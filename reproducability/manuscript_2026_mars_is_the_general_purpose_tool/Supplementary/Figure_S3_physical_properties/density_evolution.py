"""Density-matrix evolution in MaRs: a spectra manager and a populator that return ρ(t).

DensityTimeEvolution follows the pipeline of DensityTimeSpectra (resonance fields, eigenbasis,
relaxation context), but returns the density matrices of the propagator approach instead of
the spectrum. DensityPropagatorPopulator computes them.
"""
import typing as tp
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', '..', '..')))

import torch

from mars.population import PropagatorDensityPopulator, tr_utils, rk4, transform
from mars.population.transform import Liouvilleator
from mars.spectra_manager import DensityTimeSpectra


def relaxation_superoperator(energies: torch.Tensor, temperature: torch.Tensor,
                             thermal_superop: tp.Optional[torch.Tensor],
                             driven_superop: tp.Optional[torch.Tensor],
                             secular: bool = False) -> torch.Tensor:
    """Total relaxation superoperator in the eigenbasis of the Hamiltonian.
    """
    n = energies.shape[-1]
    total = torch.zeros(*energies.shape[:-1], n * n, n * n, dtype=torch.complex128, device=energies.device)
    if thermal_superop is not None:
        evo = tr_utils.EvolutionSuper(energies)
        total = total + evo.apply_thermal_correction(temperature, thermal_superop)
    if driven_superop is not None:
        total = total + driven_superop
    if secular:
        total = transform.apply_secular_mask(total)
    return total


class DensityPropagatorPopulator(PropagatorDensityPopulator):
    def __init__(self, *args, secular: bool = False, coherent: bool = True,
                 initial_state: str = "context", **kwargs):
        super().__init__(*args, **kwargs)
        self.secular = secular
        self.coherent = coherent
        self.initial_state = initial_state

    def _compute_out(self,
                     time: torch.Tensor, res_fields: torch.Tensor,
                     lvl_down: torch.Tensor, lvl_up: torch.Tensor,
                     energies: torch.Tensor, vector_down: torch.Tensor,
                     vector_up: torch.Tensor,
                     full_system_vectors: tp.Optional[torch.Tensor],
                     initial_density: torch.Tensor,
                     Gx: torch.Tensor, Gy: torch.Tensor, Sz_eigen: torch.Tensor, Sz_prod: torch.Tensor,
                     resonance_frequency: torch.Tensor,
                     H0: torch.Tensor,
                     *args, **kwargs) -> torch.Tensor:

        generator = self._init_tr_matrix_generator(
            time, res_fields, lvl_down, lvl_up, energies, vector_down, vector_up,
            full_system_vectors, resonance_frequency, H0, *args, **kwargs)
        temperature, H, thermal_superop, driven_superop = generator(time)
        R = relaxation_superoperator(energies, temperature, thermal_superop, driven_superop, self.secular)

        n = energies.shape[-1]
        if self.initial_state == "context":
            rho0 = Liouvilleator.vec(initial_density.to(R.dtype)).unsqueeze(-1)
        else:
            rho0 = torch.eye(n * n, dtype=R.dtype, device=R.device)  # vec(|i><j|)

        tau = 1 / resonance_frequency
        periods = torch.ceil(time / tau).to(torch.int64)
        if self.coherent:
            omega = resonance_frequency * self.two_pi
            static = Liouvilleator.hamiltonian_superop(H.to(R.dtype)) + R
            dynamic = Liouvilleator.hamiltonian_superop(Gx.to(R.dtype))
            U2pi, _ = rk4.solve_matrix_ode_rk4(static / omega, dynamic / omega, self.n_steps)
            maps = (torch.linalg.matrix_power(U2pi, int(p)) for p in periods)
        else:
            maps = (torch.linalg.matrix_exp(R * (float(p) * tau)) for p in periods)

        out = []
        for P in maps:
            rho = P @ rho0
            if self.initial_state == "context":
                out.append(Liouvilleator.unvec(rho.squeeze(-1)))
            else:
                out.append(Liouvilleator.unvec(rho.transpose(-2, -1)))
        return torch.stack(out, dim=0)


class DensityTimeEvolution(DensityTimeSpectra):
    def __init__(self, freq: tp.Union[float, torch.Tensor], sample, context, temperature,
                 secular: bool = False, coherent: bool = True,
                 
                 initial_state: str = "context",
                 b1_field: float = 1e-8, n_steps: int = 16,
                 hamiltonian_mode: str = "direct",
                 device: torch.device = torch.device("cpu"),
                 
                 dtype: torch.dtype = torch.float64,
                 **kwargs):
        populator = DensityPropagatorPopulator(
            b1_field=b1_field, n_steps=n_steps, context=context, init_temperature=temperature,
            secular=secular, coherent=coherent, initial_state=initial_state, device=device, dtype=dtype)
        
        super().__init__(freq=freq, sample=sample, populator=populator, context=context,
                         temperature=temperature, hamiltonian_mode=hamiltonian_mode,
                         device=device, dtype=dtype, **kwargs)

    def _postcompute_batch_data(self, sample, res_fields, intensities, width, F, Gx, Gy, Gz,
                                full_system_vectors, time, *extras, **kwargs):
        self._transition_levels = (extras[0], extras[1])
        
        return super()._postcompute_batch_data(sample, res_fields, intensities, width, F, Gx, Gy, Gz,
                                               full_system_vectors, time, *extras, **kwargs)

    def _finalize(self, res_fields, intensities, width, gauss, lorentz, fields, lvl_down, lvl_up):
        lvl_down, lvl_up = self._transition_levels
        return {"res_fields": res_fields, "lvl_down": lvl_down, "lvl_up": lvl_up, "density": intensities}
