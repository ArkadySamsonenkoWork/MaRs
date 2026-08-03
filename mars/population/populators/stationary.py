import typing as tp

import torch
from ... import constants
from . import core
from .. import contexts

import torch.nn as nn


class StationaryPopulator(core.BasePopulator):
    """COmputes the population-dependent part of the transition intensity for
    stationary (CW) EPR spectra.

    The population difference between upper and lower resonant levels determines the net absorption
    (or emission) intensity. Two initialization strategies are supported:

    1. **Thermal equilibrium**: Populations follow the Boltzmann distribution at `init_temperature`:
        p_i ∝ exp(−E_i / k_B T),
       and the population difference for a transition i ← j is:
        Δp = p_j − p_i.
       This is used when no Context is provided, or when the Context does not define initial populations.

    2. **Context-defined populations**: If the Context specifies initial populations (e.g., photoexcited
       triplet sublevel polarization), these are used instead of thermal values. The populations are
       automatically transformed into the field-dependent eigenbasis using `full_system_vectors`.
    """
    def forward(self,
                res_fields: torch.Tensor,
                lvl_down: torch.Tensor,
                lvl_up: torch.Tensor,
                energies: torch.Tensor,
                full_system_vectors: tp.Optional[torch.Tensor],
                *args, **kwargs) -> torch.Tensor:
        """Computes the population difference for each resonant EPR transition.

        :param res_fields:
            Resonance magnetic field for each transition, shape [..., M],
            where M is the number of resonance conditions, (e.g. the number of resonance for each orientation)

        :param lvl_down:
            Indices of lower energy levels involved in transitions, shape [M].

        :param lvl_up:
            Indices of upper energy levels involved in transitions, shape [M].

        :param energies:
            Eigenenergies of all spin states in Hz, shape [..., M, N],
            where M is the number of resonance conditions, (e.g. the number of resonance for each orientation)
            and N is the number of energy levels.

        :param full_system_vectors:
            Eigenvectors of the full spin Hamiltonian, shape [..., N, N].
            Required only if initial populations are defined in a non-eigenbasis (e.g., ZFS basis)
            and Context provides them. Used to transform populations into the field-dependent eigenbasis.

        :return:
            Population differences Δp = p_upper − p_lower for each transition,
            shape [..., R], ready to be multiplied by transition matrix elements.
        """
        populations = self._initial_populations(energies, lvl_down, lvl_up, full_system_vectors)
        if self.context is not None:
            self.context.close_context()
        return self._out_population_difference(populations, lvl_down, lvl_up)


class StationaryPopulatorExpanded(StationaryPopulator):
    """
    Extended version of `StationaryPopulator` that supports multiple independent
    temperature values and/or additional batch dimensions.

    In the base `StationaryPopulator`, the `init_temperature` is a scalar used
    for all energy levels across all samples. In this expanded version,
    `init_temperature` can have its own batch dimensions (e.g., multiple temperature
    points per structure). These are broadcasted correctly against the energy tensor.

    **Key differences**:
    - `_temp_dependant_init_population` reshapes `self.init_temperature` to have a
      leading batch dimension followed by ones to align with the energy tensor
      dimensions, using `reshape`. This allows
      temperature to vary across different independent configurations (e.g., different
      sample points or different mean iterations).

    The temperature should have dimensions: [temp_dimension, *batch_dimensions]
    The output has shape: [temp_dimension, *bathc_dimensions, spin_system_dimensions]
    """
    def _temp_dependant_init_population(self,
                energies: torch.Tensor,
                lvl_down: torch.Tensor,
                lvl_up: torch.Tensor,
                full_system_vectors: tp.Optional[torch.Tensor],
                *args, **kwargs):
        """
        Returns the populations defined as stationary population at temperature 'init_temperature'.
        """
        new_shape = self.init_temperature.shape + (1,) * (energies.dim() - 1)
        temperature = self.init_temperature.reshape(new_shape)
        return nn.functional.softmax(
            -constants.unit_converter(energies, "Hz_to_K") / temperature, dim=-1
        )

    def _context_dependant_init_population(self,
                energies: torch.Tensor,
                lvl_down: torch.Tensor,
                lvl_up: torch.Tensor,
                full_system_vectors: tp.Optional[torch.Tensor],
                *args, **kwargs):
        """
        Returns the populations defined as polarized and given by the population defined at context
        """
        return self.context.get_transformed_init_populations(full_system_vectors, normalize=False)

