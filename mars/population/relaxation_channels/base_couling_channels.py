import torch
import typing as tp
from abc import ABC, abstractmethod
import functools

from .. import transform

from ..thermal_corrections import ThermalBalanceMode, ThermalBalanceCorrector, init_thermal_balance_mode


OperatorComponentsType = tp.List[tp.Tuple[tp.Optional[torch.Tensor], tp.Optional[torch.Tensor]]]


def _expand_op_kronecker(op: torch.Tensor, left_dim: int, right_dim: int) -> torch.Tensor:
    """Helper to expand a single operator."""
    N = op.shape[-1]
    L, R = left_dim, right_dim
    device, dtype = op.device, op.dtype

    I_left = torch.eye(L, device=device, dtype=dtype) if L > 0 else None
    I_right = torch.eye(R, device=device, dtype=dtype) if R > 0 else None

    if L == 0:
        result = torch.einsum("...ij,kl->...ikjl", op, I_right)
        result = result.reshape(op.shape[:-2] + (N * R, N * R))
    elif R == 0:
        result = torch.einsum("ij,...kl->...ikjl", I_left, op)
        result = result.reshape(op.shape[:-2] + (L * N, L * N))
    else:
        result = torch.einsum("ij,...kl,mn->...ikmjln", I_left, op, I_right)
        result = result.reshape(op.shape[:-2] + (L * N * R, L * N * R))
    return result


def _expand_op_zeros(op: torch.Tensor, left_dim: int, right_dim: int) -> torch.Tensor:
    """
     Embed  operator component into a larger zero-padded matrix.

     It is used for open quantum systems where the relaxation only acts on a
     subspace.

     Mathematical formulation

     O' = [ 0_L  0   0  ]
          [ 0    O   0  ]
          [ 0    0   0_R ]
     Resulting shape: (L + N + R) x (L + N + R).

     :param left_dim: Size of the top-left zero block (L).
     :param right_dim: Size of the bottom-right zero block (R).
     """
    return torch.nn.functional.pad(op, (left_dim, right_dim, left_dim, right_dim))


class BasisChannelsManager(torch.nn.Module):
    def __init__(self, eigen_basis_flag: bool):
        """
        :param basis: basis initialized in the appropriate Context
        """
        super().__init__()
        self.eigen_basis_flag = eigen_basis_flag
        self._setup_transformers()

    def _setup_transformers(self):
        """Configure transformation methods based on the specified basis.

        This method sets up the appropriate transformation functions for operators

        When no transformation is needed (eigenbasis):
        - All transformation methods become identity operations (_transformed_skip)
        """
        if self.eigen_basis_flag:
            self.apply_basis_transform = self._transform_skip
        else:
            self.apply_basis_transform = self._transform_unitary

    def _transform_skip(
            self, operators: tp.List[torch.Tensor],  transformation_unitary: tp.Optional[torch.Tensor]
    ) -> tp.List[torch.Tensor]:
        return operators

    def _transform_unitary(
            self, operators: tp.List[torch.Tensor], transformation_unitary: torch.Tensor
    ) -> tp.List[torch.Tensor]:
        return [transform.transform_operator_to_new_basis(
            op.to(transformation_unitary.dtype), transformation_unitary
        ) for op in operators]


class BaseRelaxationChannel(torch.nn.Module, ABC):
    """
    Abstract base for relaxation channels (Redfield, Lindblad, etc.).
    """
    def __init__(self,
                 operator_components: tp.List[tp.Tuple[tp.Optional[torch.Tensor],
                 tp.Optional[torch.Tensor]]],
                 thermal_balance_mode: tp.Optional[tp.Union[str, ThermalBalanceMode]] = None):
        """
        Initialize the base relaxation channel.

        :param operator_components:
            list of tuples, where each tuple contains two base operator tensors
            used to construct a coupling channel.

            The coupling Hamiltonian is constructed as a sum over channels k:
            H_coupling = sum_k (O_static_k + field * O_dependent_k).

            For each tuple (O_static, O_dependent):
            1. The first operator (field-independent)
            is not multiplied by the external field.
            2. The second operator (field-dependent)
            is multiplied by the external field
               (e.g., Magnetic Field in Tesla).

            If a specific term is not necessary for the system, the corresponding tensor
            should be None. These tensors are constructed from fundamental operators
            (e.g., spin components Sx, Sy, Sz) and should be defined in the same basis
            as the context managing them.

            The final units of O_static we ask to be dimensionless,
            O_dependent must be 1 / Tesla.

        :param thermal_balance_mode: Strategy for enforcing thermal detailed balance.
            Accepts either a string or a `ThermalBalanceMode` enum value:

            - "skip" (or `ThermalBalanceMode.SKIP`): No modification. Use for
              high-temperature limits or when the spectral function is already balanced*
            - "symmetric" (or `ThermalBalanceMode.SYMMETRIC`): Symmetrizes the
              population rates and enforces detailed balance on all transitions.
              Both upward and downward rates are adjusted to preserve the average
              coupling strength while satisfying
            - "complement" (or `ThermalBalanceMode.COMPLEMENT`): Fills only
              missing zero entries in the upward transition matrix
            Default is "skip"
        """
        super().__init__()

        self.operator_components: tp.List[tp.Tuple[tp.Optional[torch.Tensor], tp.Optional[torch.Tensor]]] = []
        self.secular: bool = False
        self.eigen_basis_flag: tp.Optional[bool] = None
        self._initialize_components(operator_components)
        thermal_balance_mode = init_thermal_balance_mode(thermal_balance_mode)
        self.thermal_corrector_partial =\
            functools.partial(ThermalBalanceCorrector, thermal_balance_mode=thermal_balance_mode)

    def post_init(self, eigen_basis_flag: bool = False, secular: bool = False) -> None:
        """
        Post-Initialize the Redfield relaxation logic.
        :param eigen_basis_flag: The flag should the eigen basis be used. Or there is not None trasnformation matrix.
        If True then the basis will be None, and no transformation will be performed
        :param secular: Whether to apply the secular approximation.
        """
        self.secular = secular
        self.basis_manager = BasisChannelsManager(eigen_basis_flag)
        self.eigen_basis_flag = eigen_basis_flag

    def _initialize_components(
            self,
            operator_components: OperatorComponentsType) -> None:
        """
        Register and preprocess the base operator components.

        Check that init take a list
        of tuples. Each tuple contains (field-independent, field-dependent) operators.
        Adds a singleton dimension at index -3 to each tensor. This dimension
        facilitates broadcasting when applying field-dependent coefficients to build
        the final coupling operators.

        self.operator_components will be a
        List[Tuple[Optional[Tensor], Optional[Tensor]]].

        :param operator_components: operator components to set the coupling operators
        """
        if not isinstance(operator_components, list):
            raise TypeError(
                "operator_components must return a list of tuples."
            )

        self.operator_components = []
        for raw_ops in operator_components:
            self._validate_register_components(raw_ops)
            static, dep = raw_ops
            s = static.unsqueeze(-3) if static is not None else None
            d = dep.unsqueeze(-3) if dep is not None else None
            self.operator_components.append((s, d))

    def _validate_register_components(
            self,
            raw_ops: tp.Tuple[tp.Optional[torch.Tensor], tp.Optional[torch.Tensor]]
    ):
        """
        Validate the input of init function

        Ensures that exactly two operator components are returned, corresponding
        to the linear coupling model: H_coupling = O_static + field * O_dependent.
        One or both operators may be None if that term is not required for the
        specific physical system, but the tuple structure must be preserved.

        :param raw_ops: A tuple containing exactly two elements (field-independent,
            field-dependent).
        :raises ValueError: If the tuple does not contain exactly two elements.
        """
        if len(raw_ops) != 2:
            raise ValueError(
                "In the Readfield channel you should pass the "
                "tuple of two operators "
                "(field-independent, field-dependent)."
                "If one term is not needed, set it to None."
            )

    def build_coupling_operator(
            self, fields: tp.Optional[torch.Tensor]) -> tp.List[torch.Tensor]:
        """
        Construct the single final system-bath coupling operator from base components.

        This method takes the components registered in 'register_operator_components'
        and combines them to form the single effective operator that couples the
        system to the bath.

        Mathematical formulation:
        The effective coupling operator O is constructed as a linear combination:
        O(fields) = O_static + sum_k(field_k * O_dependent_k)

        In this implementation, 'fields' is broadcast against the
        'field_dependent_component'.
        If 'field_dependent_component' represents a vector of operators
        (e.g. [Sx, Sy, Sz]),
        ensure 'fields' matches the dimensions for the dot product,
        or handle the summation
        externally. Here we assume element-wise multiplication broadcasting.

        :param fields: Tensor of external fields (e.g., magnetic fields).
            Shape: (..., K), where K is number of transition.

        :return: The list of coupling  operator. Shape of each (..., 1, N, N).
        """
        operators = []
        if fields is not None:
            field_expanded = fields.unsqueeze(-1).unsqueeze(-1)
        else:
            field_expanded = 0.0

        for static, dep in self.operator_components:
            static_term = static if static is not None else 0
            dep_term = dep if dep is not None else 0

            op = dep_term * field_expanded + static_term
            operators.append(op)
        return operators

    def expand_kronecker(
            self, left_dim: int, right_dim: int) -> None:
        """
       Compute I_L ⊗ A ⊗ I_R for the coupling operator.

       It is used when the system is part of a larger composite Hilbert space.
       Note: This modifies the stored 'operator_components' list, so it should be
       called before 'build_coupling_operator' or the components should be re-registered.
       Mathematical formulation

       Given an operator component O of size N x N, the expanded component O' is:
       O' = I_L ⊗ O ⊗ I_R
       The resulting dimension is D = L * N * R.

       :param left_dim: Size of the left identity matrix (L). If 0, skipped.
       :param right_dim: Size of the right identity matrix (R). If 0, skipped.
        """
        results = []
        for static, dep in self.operator_components:
            new_static = None
            new_dep = None

            if static is not None:
                new_static = _expand_op_kronecker(static, left_dim, right_dim)
            if dep is not None:
                new_dep = _expand_op_kronecker(dep, left_dim, right_dim)

            results.append((new_static, new_dep))
        self.operator_components = results

    def expand_zeros(
            self, left_dim: int, right_dim: int) -> None:
        """
         Embed each operator component into a larger zero-padded matrix.

         It is used for open quantum systems where the relaxation only acts on a
         subspace.

         Mathematical formulation

         O' = [ 0_L  0   0  ]
              [ 0    O   0  ]
              [ 0    0   0_R ]
         Resulting shape: (L + N + R) x (L + N + R).

         :param left_dim: Size of the top-left zero block (L).
         :param right_dim: Size of the bottom-right zero block (R).
         """
        results = []
        for static, dep in self.operator_components:
            new_static = None
            new_dep = None
            if static is not None:
                new_static = torch.nn.functional.pad(
                    static, (left_dim, right_dim, left_dim, right_dim)
                )
            if dep is not None:
                new_dep = torch.nn.functional.pad(
                    dep, (left_dim, right_dim, left_dim, right_dim)
                )
            results.append((new_static, new_dep))
        self.operator_components = results

    def get_coupling_operators(
            self, transformation_unitary: tp.Optional[torch.Tensor],
            fields: torch.Tensor) -> tp.List[torch.Tensor]:
        """
        Wrapper to get the final field-dependent coupling operators in the eigenbasis.

        Passes the registered components to 'build_coupling_operator' to construct
        the list of effective operators, then transforms them to the eigenbasis
        using the provided unitary transformation matrix.

        :param transformation_unitary: Transformation matrix from one basis to another
            Shape: (..., N, N): V_new = U * V * U^dagger, where U is transformation matrix, V is coupling operator.
        :param fields: External fields (e.g., magnetic field). Shape: (..., K).
        :return: A list of coupling operators in the eigenbasis.
            Each has Shape: (..., N, N).
        """
        ops_original = self.build_coupling_operator(fields)
        ops_eigen = self.basis_manager.apply_basis_transform(
            ops_original, transformation_unitary)
        return ops_eigen

    def close(self):
        """
        close redfield channel
        """
        self.basis_manager.transformation_unitary = None

    @property
    def spin_system_dim(self):
        for operator_component in self.operator_components:
            if operator_component[0] is not None:
                return operator_component[0].shape[-1]
            if operator_component[1] is not None:
                return operator_component[1].shape[-1]
        return None


class CouplingChannelManager:
    """
    The manager is used to manage the several relaxation_coupling_channels.
    Each redfield_channel can have any number of coupling operators but only one spectral desnity
    """
    def __init__(self, relaxation_coupling_channels: tp.List[BaseRelaxationChannel]):
        self.relaxation_channels = relaxation_coupling_channels

        self.eigen_basis_flag = self.relaxation_channels[0].eigen_basis_flag
        if not all(channel.eigen_basis_flag == self.eigen_basis_flag
                   for channel in self.relaxation_channels):
            raise ValueError(
                "All redfield channels must have the same eigen_basis_flag "
                "(all True or all False)"
            )

    @property
    def spin_system_dim(self):
        for relaxation_channel in self.relaxation_channels:
            spin_system_dim = relaxation_channel.spin_system_dim
            if spin_system_dim is not None:
                return spin_system_dim
        return None

    def expand_zeros(
            self, left_dim: int, right_dim: int) -> None:
        """
         Embed each operator in each channel into a larger zero-padded matrix.

         It is used for open quantum systems where the relaxation only acts on a
         subspace.

         Mathematical formulation

         O' = [ 0_L  0   0  ]
              [ 0    O   0  ]
              [ 0    0   0_R ]
         Resulting shape: (L + N + R) x (L + N + R).

         :param left_dim: Size of the top-left zero block (L).
         :param right_dim: Size of the bottom-right zero block (R).
         """
        for channel in self.relaxation_channels:
            channel.expand_zeros(left_dim, right_dim)

    def expand_kronecker(
            self, left_dim: int, right_dim: int) -> None:
        """
       Compute I_L ⊗ A ⊗ I_R for the coupling operators in each channel.

       It is used when the system is part of a larger composite Hilbert space.
       Note: This modifies the stored 'operator_components' list, so it should be
       called before 'build_coupling_operator' or the components should be re-registered.
       Mathematical formulation

       Given an operator component O of size N x N, the expanded component O' is:
       O' = I_L ⊗ O ⊗ I_R
       The resulting dimension is D = L * N * R.

       :param left_dim: Size of the left identity matrix (L). If 0, skipped.
       :param right_dim: Size of the right identity matrix (R). If 0, skipped.
        """
        for channel in self.relaxation_channels:
            channel.expand_kronecker(left_dim, right_dim)

    def compute_transition_rates(
            self, transformation_unitary: tp.Optional[torch.Tensor],
            fields: tp.Optional[torch.Tensor], energies: torch.Tensor,
            temperature: torch.Tensor) -> torch.Tensor:
        """
        Compute the population transfer rate matrix. The diagonal elements are zeros

        :param transformation_unitary: Transformation matrix from one basis to another
            Shape: (..., N, N): V_new = U * V * U^dagger, where U is transformation matrix, V is coupling operator.
        :param fields: External magnetic fields in T. The shape [..., N, N]
        :param energies: System eigenenergies In Hz. The shape [..., N]
        :param temperature: The system temperature in K
        The shape is [] or [t], where t is number of time-steps
        :return: Rate matrix W.
        """
        return sum(
            channel.transition_rates(
                transformation_unitary, fields, energies, temperature
            ) for channel in self.relaxation_channels
        )

    def compute_dephasing(
            self, transformation_unitary: tp.Optional[torch.Tensor],
            fields: tp.Optional[torch.Tensor], energies: torch.Tensor,
            temperature: torch.Tensor) -> torch.Tensor:
        """
        Compute the population transfer rate matrix. The diagonal elements are zeros

        :param transformation_unitary: Transformation matrix from one basis to another
            Shape: (..., N, N): V_new = U * V * U^dagger, where U is transformation matrix, V is coupling operator.
        :param fields: External magnetic fields in T. The shape [..., N, N]
        :param energies: System eigenenergies In Hz. The shape [..., N]
        :param temperature: The system temperature in K
        The shape is [] or [t], where t is number of time-steps
        :return: Rate matrix W.
        """
        return sum(
            channel.dephasing_matrix(
                transformation_unitary, fields, energies, temperature
            ) for channel in self.relaxation_channels
        )

    def compute_relaxation_superoperator(
            self, transformation_unitary: tp.Optional[torch.Tensor], fields: tp.Optional[torch.Tensor],
            energies: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
        """
        Compute the 2D Liouvillian relaxation superoperator.

        Mathematical formulation

        The density matrix rho is vectorized in row-major order:
        |rho>> = [rho_00, rho_01, ..., rho_0N, rho_10, ...]^T
        The superoperator R acts as:
        d/dt |rho>> = R |rho>>
        where R_(ab),(cd) = R_abcd.

        The row-major index for element (i, j) is i * N + j.
        The superoperator maps index (c, d) to (a, b).

        :param transformation_unitary: Transformation matrix from one basis to another
            Shape: (..., N, N): V_new = U * V * U^dagger, where U is transformation matrix, V is coupling operator.
        :param fields: External magnetic fields in T. The shape [..., N, N]
        :param energies: System eigenenergies in Hz. The shape [..., N]
        :param temperature: The system temperature in Kelvin.
        The shape is [] or [t], where t is number of time-steps
        :return: Superoperator matrix. Shape: (..., N^2, N^2).
        """
        return sum(
            channel.compute_relaxation_superoperator(
                transformation_unitary, fields, energies, temperature) for channel in
            self.relaxation_channels
        )

    def __repr__(self):
        """
        Return a string representation of the CouplingChannelManager.

        Summarizes the number of managed channels and their common
        eigenbasis configuration.

        :return: String representation of the manager state.
        """
        cls_name = self.__class__.__name__
        num_channels = len(self.relaxation_channels) if hasattr(self, "relaxation_channels") else 0
        eigen_basis = getattr(self, "eigen_basis_flag",  "Not Initialized")
        return (
            f"{cls_name}( \n"
            f"num_channels={num_channels}, \n"
            f"eigen_basis={eigen_basis}) \n"
        )


def combine_coupling_managers(coupling_managers: tp.List[tp.Optional[CouplingChannelManager]]) ->\
        CouplingChannelManager:
    """
    Combine multiple Coupling managers into a single manager.

    Conceptually, this merges all relaxation channels from different subsystems
    into one unified manager, safely ignoring any missing (None) entries.

    :param coupling_managers: List of managers to combine. Some items may be None.
    :return: A new CouplingChannelManager containing all valid channels.
    """
    channels = [
        channel
        for manager in coupling_managers
        if manager is not None
        for channel in manager.relaxation_channels
    ]

    return CouplingChannelManager(channels) if channels else None

