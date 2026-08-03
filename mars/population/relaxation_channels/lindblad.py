import typing as tp
import torch

from .. import transform
from .base_couling_channels import BaseRelaxationChannel

from ..thermal_corrections import ThermalBalanceMode


class LindbladRelaxationChannel(BaseRelaxationChannel):
    """
    Lindblad relaxation channel using jump operators without spectral density.

    The master equation is: dρ/dt = Σ_k [ L_k ρ L_k† - 1/2 {L_k† L_k, ρ} ]
    Jump operators are expected to already include rate scaling (i.e., L_k = √γ_k A_k).
    """
    def __init__(self,
                 operator_components: tp.List[tp.Tuple[tp.Optional[torch.Tensor],
                 tp.Optional[torch.Tensor]]],
                 thermal_balance_mode: tp.Optional[tp.Union[str, ThermalBalanceMode]] = None):
        """
        Initialize the Lindblad relaxation channel with thermal balance settings.

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

            The Lindblad tensor scales as |A|^2 so the total
            dimension of the |A|^2 should be rad/s

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

        :raises ValueError: If `thermal_balance_mode` is a string that does not
            match one of the valid options ("skip", "symmetric", "complement").
        :raises TypeError: If `thermal_balance_mode` is neither a string,
            `ThermalBalanceMode` enum, nor `None`.
        """
        super().__init__(operator_components, thermal_balance_mode)
        if thermal_balance_mode == ThermalBalanceMode.COMPLEMENT:
            raise NotImplementedError(
                "LindbladRelaxationChannel currently supports only skip and compliment thermal balance"
            )
        self.liouvilleator = transform.Liouvilleator

    def transition_rates(
            self, transformation_unitary: tp.Optional[torch.Tensor],
            fields: tp.Optional[torch.Tensor], energies: torch.Tensor,
            temperature: torch.Tensor) -> torch.Tensor:
        """
        Compute the population transfer rate matrix.

        :param transformation_unitary: Transformation matrix from one basis to another
            Shape: (..., N, N): V_new = U * V * U^dagger, where U is transformation matrix, V is coupling operator.
        :param fields: External magnetic fields in T. The shape [..., N, N]
        :param energies: System eigenenergies in Hz. The shape [..., N]
        :param temperature: The system stationary temperature  in Kelvin.
        The shape is [] or [t], where t is number of time-steps.
        :return: Rate matrix W.
        """
        """Compute population transfer rates W_{a<-c} = Σ_k |(L_k)_{ac}|²."""
        jump_ops = self.get_coupling_operators(transformation_unitary, fields)
        return self.thermal_corrector_partial(energies).apply_matrix_transform(
            temperature, self._compute_transition_rates(jump_ops)
        )

    def _compute_transition_rates(self, operators: tp.List[torch.Tensor]) -> torch.Tensor:
        """
       Compute secular population transfer rates W_a<-c.

       Mathematical formulation

       The rate from state c to state a is given by Fermi's Golden Rule summed over
       all coupling operators k:
       W_a<-c = sum_k |(A_k)_ac|^2
       where omega_ca = E_c - E_a.

       :param operators: List of coupling operators in eigenbasis.
           Each Shape: (..., N, N).
       :return: Transition rate matrix W. Shape: (..., N, N).
           W[a, c] is the rate c -> a. Diagonal is zero.
       """
        dtype = torch.float32 if (operators[0].dtype == torch.complex64) else torch.float64
        device = operators[0].device
        shape = operators[0].shape
        W = torch.zeros(shape, dtype=dtype, device=device)

        for op in operators:
            W += op.abs() ** 2
        N = W.shape[-1]
        W.masked_fill_(torch.eye(N, device=W.device, dtype=torch.bool), 0.0)
        return W

    def dephasing_matrix(
            self, transformation_unitary: tp.Optional[torch.Tensor],
            fields: tp.Optional[torch.Tensor], energies: torch.Tensor,
            temperature: torch.Tensor) -> torch.Tensor:
        """
        Public wrapper for dephasing rate calculation.

        :param transformation_unitary: Transformation matrix from one basis to another
            Shape: (..., N, N): V_new = U * V * U^dagger, where U is transformation matrix, V is coupling operator.
        :param fields: External magnetic fields in T. The shape [..., N, N]
        :param energies: System eigenenergies in Hz. The shape [..., N]
        :param temperature: The system stationary temperature  in Kelvin.
        The shape is [] or [t], where t is number of time-steps.
        :return: Dephasing rates.
        """
        jump_ops = self._prepare_jump_operators(transformation_unitary, fields)
        return self._compute_dephasing_matrix(jump_ops)

    def _compute_dephasing_matrix(self, operators: tp.List[torch.Tensor]) -> torch.Tensor:
        """
        Compute pure dephasing rates for coherences rho_ab (a != b).

        Mathematical formulation

        In the secular approximation, the decay rate of coherence rho_ab
        due to population relaxation is:
        Gamma_ab_pop = 0.5 * (W_sum_a + W_sum_b)
        where W_sum_i is the total rate out of state i.

        Additionally, pure dephasing (elastic scattering) contributes:
        Gamma_ab_pure = 0.5 * sum_k |A^k_aa - A^k_bb|^2

        This method computes the total dephasing rate.

        :param operators: List of coupling operators. Each Shape: (..., N, N).
        :return: Dephasing rate matrix. Shape: (..., N, N).
        """
        W = self._compute_transition_rates(operators)
        rate_out = W.sum(dim=-2, keepdim=True)
        gamma_pop = 0.5 * (rate_out + rate_out.transpose(-1, -2))

        pure_dephasing = torch.zeros_like(gamma_pop)
        for op in operators:
            diag = torch.diagonal(op, dim1=-2, dim2=-1)
            diff = diag.unsqueeze(-2) - diag.unsqueeze(-1)
            pure_dephasing += diff.abs() ** 2 * 0.5
        return gamma_pop + pure_dephasing

    def compute_relaxation_superoperator(
            self, transformation_unitary: tp.Optional[torch.Tensor],
            fields: tp.Optional[torch.Tensor],
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
        :param temperature: The system temperature in K.
        The shape is [] or [t], where t is number of time-steps
        :return: Superoperator matrix. Shape: (..., N^2, N^2).
        """
        operators = self.get_coupling_operators(transformation_unitary, fields)
        R_raw = sum(self.liouvilleator.lindblad_dissipator_from_operator(operator) for operator in operators)
        R_raw = self.thermal_corrector_partial(energies).apply_superoperator_transform(
            temperature, R_raw
        )
        if self.secular:
            return transform.apply_secular_mask(R_raw)
        else:
            return R_raw

    def __repr__(self):
        return (
            f"LindbladRelaxationChannel(\n"
            f"  secular={self.secular}, \n"
            f"  eigen_basis={self.eigen_basis_flag}, \n"
            f"  num_jump_operators={len(self.operator_components)}) \n"
        )