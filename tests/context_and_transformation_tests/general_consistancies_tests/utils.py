"""Thin MaRs adapter: prepare a field-dependent context for native EvolutionSuper.

No custom relaxation, Boltzmann correction, Choi tests, or propagator code.
Those are already implemented in ``mars.population`` and used by the notebook.
"""
from __future__ import annotations

import math
import torch

from mars.population.tr_utils import EvolutionSuper


def get_superoperators(
    sample,
    context,
    *,
    magnetic_field_t: float,
    temperature_k: float,
    thermally_corrected: bool = True,
    orientation: int = 0,
    hamiltonian_mode: str = "direct",
) -> dict:
    """Return MaRs thermal, driven, relaxation and full generators in the energy basis.

    Thermal correction uses native ``EvolutionSuper`` (``symmetric``/``skip``).
    The full generator includes the coherent Hamiltonian. Spin energies are
    in Hz; ``EvolutionSuper`` expects an angular-frequency Hamiltonian.
    """
    if temperature_k <= 0:
        raise ValueError("temperature_k must be positive")
    if hamiltonian_mode not in ("direct", "secular"):
        raise ValueError("hamiltonian_mode must be 'direct' or 'secular'")

    terms = (sample.get_hamiltonian_terms_secular()
             if hamiltonian_mode == "secular" else sample.get_hamiltonian_terms())
    F, _, _, Gz = terms
    H = F + magnetic_field_t * Gz
    if H.ndim == 2:
        H = H.unsqueeze(0)
    energies, vectors = torch.linalg.eigh(H)
    if not 0 <= orientation < energies.shape[0]:
        raise IndexError("Invalid powder orientation index")

    v = vectors.unsqueeze(-3)
    kw = dict(
        full_system_vectors=v,
        fields=torch.full(energies.shape[:-1] + (1,), magnetic_field_t,
                          dtype=energies.dtype, device=energies.device),
        energies=energies.unsqueeze(-2),
        temperature=torch.as_tensor(temperature_k, dtype=energies.dtype,
                                    device=energies.device),
    )
    thermal_raw = context.get_transformed_thermal_superop(**kw)
    driven_raw = context.get_transformed_driven_superop(**kw)
    initial_raw = context.get_transformed_init_density(full_system_vectors=v)
    n = energies.shape[-1]

    def select(tensor, side):
        if tensor is None:
            return None
        return tensor.reshape(-1, side, side)[orientation]

    energy = energies[orientation]
    thermal_raw = select(thermal_raw, n * n)
    driven_raw = select(driven_raw, n * n)
    initial = select(initial_raw, n)
    temperature = kw["temperature"]
    evo = EvolutionSuper(energy, thermal_balance_mode=(
        "symmetric" if thermally_corrected else "skip"
    ))
    zero_h = torch.zeros((n, n), dtype=H.dtype, device=H.device)
    return dict(
        energies_hz=energy,
        initial_density=initial,
        thermal=evo(temperature, zero_h, thermal_superop=thermal_raw),
        driven=evo(temperature, zero_h, driven_superop=driven_raw),
        relaxation=evo(temperature, zero_h, thermal_superop=thermal_raw,
                       driven_superop=driven_raw),
        full=evo(temperature, 2 * math.pi * torch.diag(energy),
                 thermal_superop=thermal_raw, driven_superop=driven_raw),
    )
