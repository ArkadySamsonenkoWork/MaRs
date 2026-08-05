import torch

import typing as tp
import math
from enum import Enum

from .. import transform
from ... import constants
from .base_couling_channels import BaseRelaxationChannel

from ..thermal_corrections import ThermalBalanceMode


class RedfieldRelaxationChannel(BaseRelaxationChannel):
    """
    Abstract base class for computing Redfield relaxation tensors.

    Units Convention:
    ┌─────────────────────────────────────────┐
    │ Input energies/frequencies:  Hz          │
    │ Spectral density input ω:    rad/s       │
    │ Spectral density output J:   rad/s       │
    │ All rates (W, Γ, R):         rad/s       │
    │ Coupling operators A:        dimensionless│
    └─────────────────────────────────────────┘

    The Redfield tensor scales as |A|² × J(ω) so the total dimension
    of this multiplication must be rad / s
    By default we ask to set |A|^2 as dimensionless then
    J(ω) must return rad/s to produce rates in rad/s,
    or set A as rad/s and J(w) as the (rad/s)^-1

    This class implements the logic for constructing the Redfield tensor
    based on system-bath coupling operators and spectral density functions.
    It supports both secular and non-secular approximations.

    Attributes:
        secular (bool): If True, applies the secular approximation (keeps only
            energy-conserving terms). Default is True.
        operator_components (tp.List[torch.Tensor]): List of system coupling
            operators in the eigenbasis, shaped (..., 1, N, N).
        spectral_density_func (Callable): The user-provided spectral density function.
        thermal_balance_mode (ThermalBalanceMode): Strategy for enforcing detailed balance.
    """
    def __init__(self,
        operator_components: tp.List[tp.Tuple[tp.Optional[torch.Tensor], tp.Optional[torch.Tensor]]],
        spectral_density_func: tp.Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        thermal_balance_mode: tp.Optional[tp.Union[str, ThermalBalanceMode]] = None):
        """
        Initialize the Redfield relaxation channel with thermal balance settings.

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

            The Redfield tensor scales as |A|^2 × J(ω) so the total
            dimension of this multiplication must be rad / s
            By default we ask to set |A|^2 as dimensionless then
            J(ω) must return rad/s to produce rates in rad/s,
            or set A as rad/s and J(w) as the (rad/s)^-1

        :param spectral_density_func:
            Function calculating the spectral density J(ω).
                Signature: func(omega_rad_s: Tensor) -> Tensor
                - Input: Two parameters: Transition frequencies in rad/s (can be negative) and the temperature at K
                - Output: Spectral density values.
                - Units: Must match `dimension_convention`.

            The Redfield rate is: W = |A|^2 × J(ω)
            Since |A|² is dimensionless and probabilities must be positive,
            J(ω) should return non-negative values

            Mathematical formulation:
            The spectral density describes the bath correlation function
            in frequency domain:
            J(omega) = ∫ dt exp(i * omega * t) * <B(t)B(0)>

            For classical noise, J(omega) is real, even, and non-negative:
            J(-omega) = J(omega) >= 0

            For quantum baths,
            detailed balance can be enforced by _compute_density() depending of flag:
            J(-omega) = J(omega) * exp(-h * omega / (k_B * T))

        :param thermal_balance_mode: Strategy for enforcing thermal detailed balance.
            Accepts either a string or a `ThermalBalanceMode` enum value:

            - "skip" (or `ThermalBalanceMode.SKIP`): No modification. Use for
              high-temperature limits ($k_B T \gg \hbar\omega$) or non-thermal baths or
               when the spectral function is already balanced.
              *Default if None is provided.*
            - "symmetric" (or `ThermalBalanceMode.SYMMETRIC`): Symmetrizes the
              spectral density and enforces detailed balance on all transitions.
              Both upward and downward rates are adjusted to preserve the average
              coupling strength while satisfying
              $J(-\omega) = J(\omega)e^{-\hbar\omega/k_BT}$.
            - "complement" (or `ThermalBalanceMode.COMPLEMENT`): Fills only
              missing zero entries in the upward transition matrix ($\omega < 0$)
              using the corresponding downward transitions ($\omega > 0$) and the
              Boltzmann factor. Preserves user-computed downward rates exactly.

            Default is "skip"

        :raises ValueError: If `thermal_balance_mode` is a string that does not
            match one of the valid options ("skip", "symmetric", "complement").
        :raises TypeError: If `thermal_balance_mode` is neither a string,
            `ThermalBalanceMode` enum, nor `None`.
        """
        super().__init__(operator_components, thermal_balance_mode)
        self.spectral_density_func = spectral_density_func

    def _get_spectral_density_matrix(self, energies: torch.Tensor,
                                     temperature: torch.Tensor) -> torch.Tensor:
        """
        Compute the thermally balanced spectral density matrix for all pairs of eigenstates.

        Calculates transition frequencies from the provided energy levels, evaluates the
        raw spectral density, and applies the configured thermal detailed balance correction
        using :class:`ThermalBalanceCorrector`.

        :param energies: Eigenenergies of the spin Hamiltonian in Hz.
            Shape: ``(..., N)`` where ``N`` is the Hilbert space dimension.
        :param temperature: System temperature in Kelvin.
            Shape: ``(...)`` (must be broadcastable with ``energies``).
        :return: Thermally corrected spectral density matrix ``J(omega)``.
            Shape: ``(..., N, N)`` where ``spec_density[..., a, c]`` corresponds to the
            transition ``c -> a`` with frequency ``omega_ca = E_c - E_a``.
            Units: rad/s.
        """
        # omega_cd = E_up - E_down
        omega_hz = energies.unsqueeze(-2) - energies.unsqueeze(-1)
        J_raw = self.spectral_density_func(omega_hz * 2 * math.pi, temperature)
        return self.thermal_corrector_partial(energies).apply_matrix_transform(temperature, J_raw)

    def _compute_transition_rates(self, operators: tp.List[torch.Tensor],
                                  spec_density: torch.Tensor) -> torch.Tensor:
        """
       Compute secular population transfer rates W_a<-c.

       Mathematical formulation

       The rate from state c to state a is given by Fermi's Golden Rule summed over
       all coupling operators k:
       W_a<-c = sum_k |(A_k)_ac|^2 * J(omega_ca)
       where omega_ca = E_c - E_a. Note that 'spec_density' is indexed as J_ca.

       :param operators: List of coupling operators in eigenbasis.
           Each Shape: (..., N, N).
       :param spec_density: Spectral density matrix J(omega_cd). Shape: (..., N, N).
       :return: Transition rate matrix W. Shape: (..., N, N).
           W[a, c] is the rate c -> a. Diagonal is zero.
       """
        W = torch.zeros_like(spec_density)
        J_trans = spec_density

        for op in operators:
            sq = op.abs() ** 2
            W += sq * J_trans
        N = operators[0].shape[-1] if operators else spec_density.shape[-1]
        diag_mask = torch.eye(N, device=W.device, dtype=torch.bool)
        W = W.masked_fill_(diag_mask, 0.0)
        return W

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
        operators = self.get_coupling_operators(transformation_unitary, fields)
        spec_density = self._get_spectral_density_matrix(energies, temperature)
        return self._compute_transition_rates(operators, spec_density)

    def _compute_dephasing_matrix(
            self, operators: tp.List[torch.Tensor],
            spec_density: torch.Tensor) -> torch.Tensor:
        """
        Compute pure dephasing rates for coherences rho_ab (a != b).

        Mathematical formulation

        In the secular approximation, the decay rate of coherence rho_ab
        due to population relaxation is:
        Gamma_ab_pop = 0.5 * (W_sum_a + W_sum_b)
        where W_sum_i is the total rate out of state i.

        Additionally, pure dephasing (elastic scattering) contributes:
        Gamma_ab_pure = 0.5 * sum_k |A^k_aa - A^k_bb|^2 * J(0)

        This method computes the total dephasing rate.

        :param operators: List of coupling operators. Each Shape: (..., N, N).
        :param spec_density: Spectral density. Shape: (..., N, N).
        :return: Dephasing rate matrix. Shape: (..., N, N).
        """
        W = self._compute_transition_rates(operators, spec_density)

        rate_out = W.sum(dim=-2, keepdim=True)
        gamma_pop = 0.5 * (rate_out + rate_out.transpose(-1, -2))
        J0 = spec_density.diagonal(offset=0, dim1=-2, dim2=-1)

        pure_dephasing = torch.zeros_like(gamma_pop)
        for op in operators:
            op_diag = torch.diagonal(op, offset=0, dim1=-2, dim2=-1)
            diff = op_diag.unsqueeze(-2) - op_diag.unsqueeze(-1)
            sq_diff = diff.abs() ** 2
            pure_dephasing += sq_diff * J0.unsqueeze(-1).mul(0.5)

        return gamma_pop + pure_dephasing

    def dephasing_matrix(
            self, transformation_unitary: tp.Optional[torch.Tensor],
            fields: tp.Optional[torch.Tensor],
            energies: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
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
        operators = self.get_coupling_operators(transformation_unitary, fields)
        spec_density = self._get_spectral_density_matrix(energies, temperature)
        return self._compute_dephasing_matrix(operators, spec_density)

    def _secular_superoperator_4d(
            self, operators: tp.List[torch.Tensor],
            spec_density: torch.Tensor) -> torch.Tensor:
        """
        Compute the Redfield tensor under the secular approximation.

        Mathematical formulation

        The secular approximation retains only terms where omega_ab approx omega_cd.
        For non-degenerate systems, this implies a=c and b=d for coherences,
        and a=b, c=d for populations.

        The 4D tensor R_abcd has three main contributions:
        1. Population Transfer (a=b, c=d): R_aacc = W_a<-c for a != c.
        2. Coherence Decay including population loss because of population (a!=b, c=a, d=b): R_abab = -Gamma_ab.
        All other terms are zero.

        :param operators: List of coupling operators.
        :param spec_density: Spectral density matrix.
        :return: 4D Relaxation tensor. Shape: (..., N, N, N, N).
        """
        W = self._compute_transition_rates(operators, spec_density)
        gamma = self._compute_dephasing_matrix(operators, spec_density)

        N = W.shape[-1]
        device = W.device
        R = torch.zeros(
            W.shape[:-2] + (N, N, N, N), dtype=W.dtype, device=device)

        idx = torch.arange(N, device=device)
        i, j = torch.meshgrid(idx, idx, indexing="ij")

        R[..., i, i, j, j] = W
        R[..., i, j, i, j] = -gamma
        dtype = R.dtype
        complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
        return R.to(complex_dtype)

    def _non_secular_superoperator_4d(
            self, operators: tp.List[torch.Tensor],
            spec_density: torch.Tensor) -> torch.Tensor:
        """
        Compute the full Redfield tensor without secular approximation.

        Mathematical formulation (Section 3.4)

        The relaxation superoperator elements R_{abcd} are computed using the
        Bloch-Redfield formalism. The total Hamiltonian is partitioned into
        system, bath, and interaction terms, where the interaction is defined
        by coupling operators A^k and bath operators B^k.

        The intermediate tensor Gamma_{abcd}(omega) is defined as:
        Gamma_{abcd}(omega) = sum_k A^k_{ab} A^k_{cd}* J_k(omega)

        The full Redfield tensor R_{abcd} consists of four terms summed over
        intermediate states e:
        R_{abcd} = 0.5 * sum_e [
                    Gamma_{aece}(omega_{ce}) * delta_{bd} +
                    Gamma_{bede}*(omega_{de}) * delta_{ac} -
                    Gamma_{acbd}(omega_{ca}) -
                    Gamma_{bdac}*(omega_{db})
                ]
        delta_{bd} == 1 for b==d and 0 otherwise
        This includes all terms allowing for coherence transfer between transitions
        with similar frequencies (non-secular).

        :param operators: List of coupling operators A^k. Shape: (..., N, N).
        :param spec_density: Spectral density matrix J_k(omega).
                             Shape: (..., N, N) corresponding to transition frequencies.

        Remember that:
            Jca = J[a, c]

        :return: 4D Relaxation tensor R_{abcd}. Shape: (..., N, N, N, N).
        """
        ref_op = operators[0]
        spec_density = spec_density.to(ref_op.dtype)
        N = ref_op.shape[-1]
        device = ref_op.device

        A_Aconj_sum = torch.zeros(
                spec_density.shape[:-2] + (N, N, N, N), dtype=ref_op.dtype, device=device)
        for op in operators:
            A_Aconj_sum += torch.einsum("...ab,...cd->...abcd", op, op.conj())

        #1. Term 1
        eye = torch.eye(N, device=device, dtype=ref_op.dtype)
        term1 = torch.einsum("...aece,...ec,bd->...abcd", A_Aconj_sum, spec_density, eye)

        # 3. Term 2:
        term2 = torch.einsum("...bede,...ed,ac->...abcd", A_Aconj_sum.conj(), spec_density, eye)

        # 4. Term 3:
        term3 = torch.einsum("...acbd,...ac->...abcd", A_Aconj_sum, spec_density)

        # 5. Term 4:
        term4 = torch.einsum("...bdac,...bd->...abcd", A_Aconj_sum.conj(), spec_density)

        R = 0.5 * (term1 + term2 - term3 - term4)
        return R.negative_()

    def compute_relaxation_superoperator_4d(
            self, transformation_unitary: tp.Optional[torch.Tensor],
            fields: tp.Optional[torch.Tensor],
            energies: torch.Tensor,
            temperature: torch.Tensor) -> torch.Tensor:
        """
         Compute the 4D Redfield relaxation tensor.

        :param transformation_unitary: Transformation matrix from one basis to another
            Shape: (..., N, N): V_new = U * V * U^dagger, where U is transformation matrix, V is coupling operator.
        :param fields: External magnetic fields in T. The shape [..., N, N]
        :param energies: System eigenenergies in Hz. The shape [..., N]
        :param temperature: The system stationary temperature  in Kelvin.
        The shape is [] or [t], where t is number of time-steps.
        :return: Tensor R_abcd. Shape: (..., N, N, N, N).
        """
        operators = self.get_coupling_operators(transformation_unitary, fields)
        spec_density_matrix = self._get_spectral_density_matrix(
            energies, temperature
        )
        if self.secular:
            return self._secular_superoperator_4d(operators, spec_density_matrix)
        else:
            return self._non_secular_superoperator_4d(operators, spec_density_matrix)

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
        R4 = self.compute_relaxation_superoperator_4d(transformation_unitary, fields, energies, temperature)
        N = R4.shape[-1]
        return R4.reshape(R4.shape[:-4] + (N * N, N * N))

    def __repr__(self):
        """
        Return a string representation of the RedfieldRelaxationChannel.

        Includes configuration details like thermal balance mode, secular approximation,
        eigenbasis usage, and operator counts.

        :return: String representation of the channel configuration.
        """
        cls_name = self.__class__.__name__

        # Thermal Balance Mode
        if hasattr(self, "thermal_balance_mode"):
            mode = self.thermal_balance_mode
            thermal_mode = mode.value if hasattr(mode, "value") else str(mode)
        else:
            thermal_mode = "Not Initialized"

        secular = getattr(self, "secular", "Not Initialized")
        eigen_basis = getattr(self, "eigen_basis_flag", "Not Initialized")

        num_ops = len(self.operator_components) if hasattr(self, "operator_components") else 0

        spec_func = getattr(self, "spectral_density_func", None)
        spec_name = spec_func.__name__ if hasattr(spec_func, "__name__") else str(type(spec_func).__name__)

        return (
            f"{cls_name}(\n"
            f"thermal_balance_mode={thermal_mode}, \n"
            f"secular={secular}, \n"
            f"eigen_basis={eigen_basis}, \n"
            f"num_operators={num_ops}, \n"
            f"spectral_density_func={spec_name}) \n"
        )
