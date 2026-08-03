import typing as tp
from enum import Enum

import torch

from .. import constants


class ThermalBalanceMode(Enum):
    """
    Enumeration of thermal balance enforcement modes for spectral density.

    These modes control how detailed balance is enforced on the spectral density
    matrix to ensure physical consistency with thermal equilibrium.

    Attributes:
        SKIP: No thermal balance enforcement. Use for high-temperature limit
            (k_B T >> ℏω) or non-thermal baths.
        SYMMETRIC: Symmetrize spectral dens ity using Boltzmann factors. Both
            upward and downward transitions are adjusted to satisfy detailed
            balance while preserving average coupling strength.
        COMPLEMENT: Fill missing transitions using detailed balance. Only
            upward transitions (ω < 0) that are zero are filled from their
            downward counterparts. Downward transitions (ω > 0) are preserved.
    """
    SKIP = "skip"
    SYMMETRIC = "symmetric"
    COMPLEMENT = "complement"


def init_thermal_balance_mode(thermal_balance_mode: tp.Optional[tp.Union[str, ThermalBalanceMode]]):
    if thermal_balance_mode is None:
        thermal_balance_mode = ThermalBalanceMode.SKIP
    elif isinstance(thermal_balance_mode, ThermalBalanceMode):
        thermal_balance_mode = thermal_balance_mode
    elif isinstance(thermal_balance_mode, str):
        try:
            thermal_balance_mode = ThermalBalanceMode(
                thermal_balance_mode.lower()
            )
        except ValueError:
            valid_modes = [mode.value for mode in ThermalBalanceMode]
            raise ValueError(
                f"Invalid thermal_balance_mode '{thermal_balance_mode}'. "
                f"Must be one of: {valid_modes}"
            )
    else:
        raise TypeError(
            f"thermal_balance_mode must be str, ThermalBalanceMode, or None, "
            f"got {type(thermal_balance_mode)}"
        )
    return thermal_balance_mode


class ThermalBalanceCorrector:
    """Utility for enforcing thermal detailed balance on transition rates.

    This class provides configurable methods to correct raw or symmetric
    spectral density/population rate estimates so that they satisfy the
    detailed balance condition at a given temperature.

    :ivar omega_K: Transition frequencies in Kelvin. Shape: ``[..., N, N]``,
        where ``omega_K[i, j] = (E_j - E_i) / k_B``. Positive values indicate
        downward (emission) transitions.
    :ivar config_dim: Shape of the configuration/batch dimensions
        (all dimensions except the last two).
    :ivar spin_system_dim: Dimension of the spin system Hilbert space (``N``).
    :ivar _thermal_transform: Callable implementing the selected thermal
        scaling strategy.
    """
    def __init__(self, energies: torch.Tensor,
                 thermal_balance_mode: tp.Optional[tp.Union[str, ThermalBalanceMode]] = ThermalBalanceMode.SYMMETRIC):
        """
        :param energies: Energy eigenvalues of the spin Hamiltonian.
                         Expected shape: [..., N] where N is the dimension of the spin system.

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
        omega_hz = energies.unsqueeze(-2) - energies.unsqueeze(-1)
        self.omega_K = constants.unit_converter(omega_hz, "Hz_to_K")

        self.config_dim = self.omega_K.shape[:-2]
        thermal_balance_mode = init_thermal_balance_mode(thermal_balance_mode)
        self._thermal_transform = self._thermal_transform_factory(thermal_balance_mode)
        self.spin_system_dim = energies.shape[-1]

    def _thermal_transform_factory(self, thermal_balance_mode: ThermalBalanceMode) -> \
            tp.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Selects the Boltzmann transformation function based on input
        convention.

        :param thermal_balance_mode: Strategy for enforcing thermal detailed balance.
            Accepts `ThermalBalanceMode` enum value:
        :return: Callable implementing either symmetric or asymmetric
            thermal scaling.
        """
        if thermal_balance_mode == ThermalBalanceMode.SYMMETRIC:
            return self._compute_transform_symmetric
        elif thermal_balance_mode == ThermalBalanceMode.COMPLEMENT:
            return self._compute_transform_complement
        else:
            return self._compute_transform_skip

    def _compute_energy_factor(self, temperature: torch.Tensor) -> torch.Tensor:
        """
        Compute Boltzmann factor for transition FROM j TO i:

            factor[i, j] = 1 / (1 + exp(-(E_j - E_i) / k_B T))

        This ensures detailed balance:
            w_{i→j} / w_{j→i} = exp(-(E_j - E_i) / k_B T)

        ote: energy_diff[i, j] = E_j - E_i, so factor[i, j] applies to the rate w_{i→j} (i → j).

        :param temperature: Temperature tensor [...]
        :return: Factor tensor [..., N, N] where factor[i, j] corresponds to w_{j→i}
        """
        denom = 1 + torch.exp(-self.omega_K / temperature)
        return torch.reciprocal(denom)

    def _compute_transform_symmetric(self, temperature: torch.tensor, thermal_rates: torch.Tensor) -> torch.Tensor:
        """
        Convert symmetric mean strengths k'_ij into physical rates w_{j→i}.

        Given k'_ij = (thermal_rates[i, j] + thermal_rates[j, i]) / 2:

            w_{j→i} = 2 * k'_ij * factor[i, j]
            w_{i→j} = 2 * k'_ij * factor[j, i]

        where factor[i, j] is computed by _compute_energy_factor and energy_diff[i, j] = E_j - E_i.

        Returns tensor thermal_rates_update where thermal_rates_update[i, j] = w_{j→i}.

        :param temperature: Temperature in K [...]
        :param thermal_rates: Symmetric tensor [..., N, N] with thermal_rates[i, j] = k'_ij
        :return: Tensor [..., N, N] where output[i, j] = w_{j→i}
        """
        energy_factor = self._compute_energy_factor(temperature)
        return (thermal_rates + thermal_rates.transpose(-2, -1)) * energy_factor

    def _compute_transform_complement(self, temperature: torch.tensor, thermal_rates: torch.Tensor) -> torch.Tensor:
        """
        Enforce detailed balance on raw rate estimates.

        Input interpretation: thermal_rates[i, j] is initial estimate of w_{j→i}

        To enforce detailed balance:
            w_{j→i} = w_{i→j} * exp( -(E_i - E_j) / k_B T )
                    = w_{i→j} * exp( +(E_j - E_i) / k_B T )

        Since energy_diff[i, j] = E_j - E_i:

            w_{i→j} = w_{j→i} * exp(-energy_diff[i, j] / k_B T)

          If thermal_rates[i, j] > 0 and thermal_rates[j, i] == 0:
              set w_{j→i} = w_{i→j} * exp(+(E_j - E_i) / k_B T)

        The returned tensor R satisfies: R[i, j] = w_{i→j} (final physical rate).

        Implementation logic:
        ┌─────────────────────────────────────────────────────────────────────────┐
        │ Case 1: Both w[i, j] and w[j, i] are non-zero:                          │
        │   → If i→j is upward (omega < 0): preserve w[i, j]                      │
        │   → If i→j is downward (omega >= 0): compute w[i, j] from w[j, i]       │
        │     (i.e., modify downward to satisfy detailed balance w/ upward)       │
        │                                                                         │
        │ Case 2: Only one of w[i, j] or w[j, i] is non-zero:                     │
        │   → Preserve the existing non-zero rate                                 │
        │   → Compute the missing rate using detailed balance complement          │
        │                                                                         │
        │ Case 3: Both are zero:                                                  │
        │   → Leave as zero                                                       │
        └─────────────────────────────────────────────────────────────────────────┘

        :param temperature: Temperature in K [...]
        :param thermal_rates: Asymmetric input [..., N, N] where
            thermal_rates[i, j] is interpreted as the initial guess for w_{i→j}
        :return: Tensor [..., N, N] where output[i, j] = w_{i→j} satisfying
            w_{i→j} / w_{j→i} = exp( -(E_j - E_i) / k_B T )  (detailed balance)
        """
        w = thermal_rates
        w_T = w.transpose(-1, -2)
        exp_factor = torch.exp(self.omega_K / temperature)

        is_zero = torch.isclose(w, torch.zeros_like(w), atol=1e-12)
        is_zero_T = is_zero.transpose(-1, -2)

        both_nonzero = ~is_zero & ~is_zero_T
        only_ji = is_zero & ~is_zero_T
        upward = self.omega_K < 0

        compute_mask = (both_nonzero & ~upward) | only_ji

        computed_from_T = w_T * exp_factor

        result = w.clone()
        torch.where(compute_mask, computed_from_T, result, out=result)
        return result

    def _compute_transform_skip(self, temperature: torch.tensor, thermal_rates: torch.Tensor) -> torch.Tensor:
        """
        Skip applying thermal balance
        :param temperature: Temperature in K [...]
        :param thermal_rates: Asymmetric input [..., N, N] where
            thermal_rates[i, j] is interpreted as the initial guess for w_{i→j}
        :return: Tensor [..., N, N] where output[i, j] = w_{i→j} satisfying
            w_{i→j} / w_{j→i} = exp( -(E_j - E_i) / k_B T )  (detailed balance)
        """
        return thermal_rates

    def apply_matrix_transform(self, temperature: torch.Tensor, thermal_rates: torch.Tensor):
        """
        :param temperature: Temperature in K [...]
        :param thermal_rates: Asymmetric input [..., N, N] where
            thermal_rates[i, j] is interpreted as the initial guess for w_{i→j}
        """
        return self._thermal_transform(temperature, thermal_rates)

    def _get_relaxation_superop_diag_correction(self, matrix_old: torch.Tensor, matrix_new: torch.Tensor) ->\
            torch.Tensor:
        """
        Computes the diagonal correction for the relaxation superoperator.

        :param matrix_old: Original population transfer matrix of shape ``[..., N, N]``.
        :param matrix_new: Thermally transformed population transfer matrix of shape ``[..., N, N]``.
        :returns: Correction tensor of shape ``[..., N*N]``. When added to the superoperator
                  diagonal, it updates population decay and coherence dephasing rates.
        :rtype: torch.Tensor
        """
        gamma_out_old = matrix_old.sum(dim=-2)
        gamma_out_new = matrix_new.sum(dim=-2)

        diag_old = 0.5 * (gamma_out_old.unsqueeze(-1) + gamma_out_old.unsqueeze(-2))
        diag_new = 0.5 * (gamma_out_new.unsqueeze(-1) + gamma_out_new.unsqueeze(-2))

        return (diag_new - diag_old).reshape(*matrix_new.shape[:-2], -1)

    def apply_superoperator_transform(self, temperature: torch.Tensor, thermal_superop: torch.Tensor) -> torch.Tensor:
        """Apply Boltzmann scaling to population transfer rates in superoperator.

        The population-population block is defined by indices:
            pop_block[i, j] = thermal_superop[pop_i, pop_j] = initial rate from j → i (i.e., R_{iijj})

        Under the symmetric input convention:
            (pop_block[i, j] + pop_block[j, i]) / 2 = k'_ij   (mean strength)

        This method computes physical rates:
            w_{j→i} = 2 * k'_ij * factor[i, j]
            w_{i→j} = 2 * k'_ij * factor[j, i]

        where factor[i, j] = 1 / (1 + exp(-(E_j - E_i) / k_B T))

        After scaling off-diagonal elements (i ≠ j), diagonal elements are adjusted
        to preserve the original column sums of the population block. This preserves
        the external loss from populations while correctly modifying dephasing rates.

        :param temperature: Temperature tensor [...]
        :param thermal_superop: Relaxation superoperator [..., N², N²]
            Must be in the energy eigenbasis. Element [p_i, p_j] represents the rate
            from population ρ_jj to population ρ_ii (j → i transition)
        :return: Scaled superoperator [..., N², N²] with thermally corrected
            population transfer rates satisfying detailed balance while preserving
            original net loss rates from each energy level.
        """
        device = thermal_superop.device
        N_square = thermal_superop.shape[-1]

        pop_indices = torch.arange(self.spin_system_dim, device=device) * (self.spin_system_dim + 1)
        pop_block_old = thermal_superop[..., pop_indices[:, None], pop_indices[None, :]]

        pop_block_old[..., torch.arange(self.spin_system_dim), torch.arange(self.spin_system_dim)] = 0.0
        pop_block_new = self.apply_matrix_transform(temperature, pop_block_old)
        diag_correction = self._get_relaxation_superop_diag_correction(pop_block_old, pop_block_new)

        target_shape = pop_block_new.shape[:-2] + thermal_superop.shape[-2:]
        thermal_superop = thermal_superop.expand(target_shape).clone()
        diagonal_old = thermal_superop[..., torch.arange(N_square), torch.arange(N_square)]

        thermal_superop[..., pop_indices[:, None], pop_indices[None, :]] = pop_block_new
        thermal_superop[..., torch.arange(N_square), torch.arange(N_square)] = -diag_correction + diagonal_old
        return thermal_superop
