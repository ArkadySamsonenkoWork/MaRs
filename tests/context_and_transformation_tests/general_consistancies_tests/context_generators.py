"""Generate TWO independent MaRs contexts per sample (NO thermal flag here).

All four physical parameters (initial populations, thermal and driven rates,
dephasing, plus optionally irreversible decay) are randomized independently
in mixed bases: multiplet, zfs, product.  See utils.py to choose detailed
balance later, when the generator is evaluated at a field and temperature.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from sample_generators import SampleCase

BASES = ("multiplet", "zfs", "product")


@dataclass(frozen=True)
class ContextPair:
    first: object
    second: object
    first_basis: str
    second_basis: str
    has_decay: bool

    @property
    def contexts(self) -> tuple[object, object]:
        return self.first, self.second

    def __iter__(self):
        return iter(self.contexts)


def _basis_for_direct_sum(sample: object, components: tuple[object, object], basis: str):
    """Build a direct-sum basis from the corresponding native sample bases."""
    if basis == "zfs":
        return "zfs"
    if basis not in ("multiplet", "product"):
        raise ValueError(f"Unsupported basis {basis!r}")

    method = (
        "get_spin_multiplet_basis" if basis == "multiplet"
        else "get_product_state_basis"
    )
    F, _, _, _ = sample.get_hamiltonian_terms()
    count = F.shape[0] if F.ndim == 3 else 1
    blocks = []
    for component in components:
        vectors = torch.as_tensor(getattr(component, method)())
        if vectors.ndim == 2:
            vectors = vectors.unsqueeze(0)
        if vectors.ndim == 4 and vectors.shape[-3] == 1:
            vectors = vectors.squeeze(-3)
        if vectors.ndim != 3:
            raise ValueError(f"Unexpected {basis} basis shape: {tuple(vectors.shape)}")
        if vectors.shape[0] not in (1, count):
            raise ValueError("Component basis has incompatible orientations")
        blocks.append(vectors.expand(count, -1, -1))

    dimension = sum(block.shape[-1] for block in blocks)
    target = torch.zeros(count, 1, dimension, dimension,
                         dtype=F.dtype, device=F.device)
    offset = 0
    for block in blocks:
        width = block.shape[-1]
        target[:, 0, offset:offset+width, offset:offset+width] = block.to(F.dtype)
        offset += width
    return target


def _rates(rng: np.random.Generator, dimension: int, low: float, high: float,
           dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Fully connected, symmetric, positive jump rates (s^-1)."""
    array = rng.uniform(low, high, size=(dimension, dimension))
    array = (array + array.T) * 0.5
    np.fill_diagonal(array, 0.0)
    return torch.tensor(array, dtype=dtype, device=device)


def _basis_argument(sample, basis: str, direct_sum_components=None):
    if direct_sum_components is None:
        return basis
    return _basis_for_direct_sum(sample, direct_sum_components, basis)


def make_context_pair(
    sample: object,
    rng: np.random.Generator,
    *,
    secular: bool = False,
    decay: bool = False,
    bases: tuple[str, str] | None = None,
    direct_sum_components: tuple[object, object] | None = None,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
) -> ContextPair:
    """Make two contexts for ONE sample, with randomized rates and bases.

    The first context supplies a normalized initial state.  The second does
    not supply an initial state, preventing accidental doubling when they
    are added.  BOTH contain independent thermal/driven/dephasing channels.
    The only toggles here are secularity and external decay (as requested).
    """
    from mars import population

    device = torch.device(device)
    F, _, _, _ = sample.get_hamiltonian_terms()
    d = int(F.shape[-1])
    if d < 2:
        raise ValueError("Need a system with at least two states")
    if bases is None:
        perm = rng.permutation(BASES)
        bases = (str(perm[0]), str(perm[1]))
    if len(bases) != 2 or any(b not in BASES for b in bases):
        raise ValueError(f"bases must be 2 entries from {BASES}")

    p = rng.dirichlet(np.ones(d) * 1.3)
    init_populations = torch.tensor(p, dtype=dtype, device=device)
    rate_a = _rates(rng, d, 500.0, 2_000.0, dtype, device)
    rate_b = _rates(rng, d, 500.0, 2_000.0, dtype, device)
    driven_a = _rates(rng, d, 10.0, 120.0, dtype, device)
    driven_b = _rates(rng, d, 10.0, 120.0, dtype, device)
    deph_a = torch.tensor(rng.uniform(300.0, 900.0, d), dtype=dtype, device=device)
    deph_b = torch.tensor(rng.uniform(300.0, 900.0, d), dtype=dtype, device=device)
    losses = (torch.tensor(rng.uniform(20.0, 100.0, d), dtype=dtype,
                           device=device) if decay else None)
    shared = dict(sample=sample, enforce_secularity=secular, dtype=dtype, device=device)
    context_a = population.Context(
        basis=_basis_argument(sample, bases[0], direct_sum_components),
        init_populations=init_populations, thermal_rates=rate_a,
        driven_rates=driven_a, dephasing=deph_a, decay_rates=losses, **shared,
    )
    context_b = population.Context(
        basis=_basis_argument(sample, bases[1], direct_sum_components),
        thermal_rates=rate_b, driven_rates=driven_b, dephasing=deph_b, **shared,
    )
    return ContextPair(context_a, context_b, bases[0], bases[1], decay)


def generate_contexts(
    samples: SampleCase,
    rng: np.random.Generator,
    *,
    secular: bool = False,
    decay: bool = False,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
) -> tuple[ContextPair, ContextPair, ContextPair, ContextPair]:
    """Return 4 pairs: two contexts for each of the 4 samples.

    Each pair has fresh independent random parameters and mixed bases.
    Basis ordering cycles to cover all 3 choices in a short fixture run.
    """
    perm = list(rng.permutation(BASES))
    bases = [(str(perm[i % 3]), str(perm[(i + 1) % 3])) for i in range(4)]
    return tuple(
        make_context_pair(
            s, rng, secular=secular, decay=decay, bases=bases[i],
            direct_sum_components=(samples.sample_1, samples.sample_2) if i == 3 else None,
            dtype=dtype, device=device,
        )
        for i, s in enumerate(samples.samples)
    )
