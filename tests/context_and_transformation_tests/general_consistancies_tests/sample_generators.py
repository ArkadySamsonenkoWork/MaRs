"""Reproducible two-, three-, and four-electron MaRs relaxation samples."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch


@dataclass(frozen=True)
class SampleCase:
    """Two interacting subsystems, their coupled system, and their direct sum."""

    sample_1: object
    sample_2: object
    sample_mul: object
    sample_direct_sum: object
    spins_1: tuple[float, ...]
    spins_2: tuple[float, ...]
    interactions_hz: tuple[tuple[int, int, float], ...]
    description: str

    @property
    def samples(self) -> tuple[object, object, object, object]:
        return self.sample_1, self.sample_2, self.sample_mul, self.sample_direct_sum

    def __iter__(self):
        return iter(self.samples)


def make_sample_case(
    rng: np.random.Generator,
    *,
    spins: tuple[float, ...] = (0.5, 0.5, 0.5, 0.5),
    partition: int | None = None,
    exchange_hz: float | None = None,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
    mesh: tuple[int, int] = (2, 3),
) -> SampleCase:
    """Use connected, anisotropically coupled 2-, 3-, or 4-spin systems.

    The first ``partition`` spins form sample_1 and the remaining spins
    form sample_2. All spin pairs interact in sample_mul. Interactions
    internal to either subsystem are retained in its individual sample.
    """
    from mars import concat, spin_model

    spins = tuple(float(s) for s in spins)
    if len(spins) not in (2, 3, 4):
        raise ValueError("spins must contain 2, 3, or 4 electron spins")
    if any(s not in (0.5, 1.0) for s in spins):
        raise ValueError("This suite supports electron spins 1/2 and 1")
    if partition is None:
        partition = 2 if len(spins) == 4 else 1
    if not 0 < partition < len(spins):
        raise ValueError("partition must leave spins in both subsystems")

    device = torch.device(device)
    g_tensors = []
    local_zfs = []
    for spin in spins:
        g = rng.uniform(1.96, 2.12, size=3)
        g_tensors.append(spin_model.Interaction(g.tolist(), dtype=dtype, device=device))
        if spin == 1.0:
            d = float(rng.uniform(140e6, 750e6))
            e = float(rng.uniform(-0.3 * d, 0.3 * d))
            local_zfs.append(spin_model.DEInteraction([d, e], dtype=dtype, device=device))
        else:
            local_zfs.append(None)

    couplings = []
    interactions_hz = []
    for i in range(len(spins)):
        for j in range(i + 1, len(spins)):
            cross_subsystems = i < partition <= j
            j_hz = float(exchange_hz) if (exchange_hz is not None and cross_subsystems) else float(rng.uniform(-110e6, 110e6))
            if abs(j_hz) < 1e6:
                j_hz = 1e6 if j_hz >= 0 else -1e6
            dipolar_xy = rng.uniform(-20e6, 20e6, size=2)
            dipolar_hz = [float(dipolar_xy[0]), float(dipolar_xy[1]),
                          float(-dipolar_xy.sum())]
            exchange = spin_model.Interaction(j_hz, dtype=dtype, device=device)
            dipolar = spin_model.Interaction(dipolar_hz, dtype=dtype, device=device)
            couplings.append((i, j, exchange + dipolar))
            interactions_hz.append((i, j, j_hz))

    def spin_system(start: int, stop: int):
        local = []
        for i in range(start, stop):
            if local_zfs[i] is not None:
                local.append((i - start, i - start, local_zfs[i]))
        for i, j, interaction in couplings:
            if start <= i < j < stop:
                local.append((i - start, j - start, interaction))
        return spin_model.SpinSystem(
            electrons=list(spins[start:stop]),
            g_tensors=g_tensors[start:stop],
            electron_electron=local,
            dtype=dtype,
            device=device,
        )

    system_1 = spin_system(0, partition)
    system_2 = spin_system(partition, len(spins))
    system_mul = spin_model.SpinSystem(
        electrons=list(spins),
        g_tensors=g_tensors,
        electron_electron=[
            (i, i, zfs) for i, zfs in enumerate(local_zfs) if zfs is not None
        ] + couplings,
        dtype=dtype,
        device=device,
    )

    broadening = dict(
        gauss=0.001,
        lorentz=0.001,
        ham_strain=0.0,
        mesh=mesh,
        dtype=dtype,
        device=device,
    )
    sample_1 = spin_model.SolidSample(base_spin_system=system_1, **broadening)
    sample_2 = spin_model.SolidSample(base_spin_system=system_2, **broadening)
    sample_mul = spin_model.SolidSample(base_spin_system=system_mul, **broadening)
    sample_direct_sum = concat((sample_1, sample_2))

    dim_1 = int(np.prod([int(2 * s + 1) for s in spins[:partition]]))
    dim_2 = int(np.prod([int(2 * s + 1) for s in spins[partition:]]))
    edges = ", ".join(f"J{i}{j}={value / 1e6:+.1f}" for i, j, value in interactions_hz)
    description = (
        f"{len(spins)} spins {spins}, partition {partition}+{len(spins)-partition}; "
        f"Hilbert dimensions {dim_1}, {dim_2}, {dim_1*dim_2}, {dim_1+dim_2}; "
        f"exchange MHz: {edges}"
    )
    return SampleCase(
        sample_1,
        sample_2,
        sample_mul,
        sample_direct_sum,
        spins[:partition],
        spins[partition:],
        tuple(interactions_hz),
        description,
    )


def iter_sample_cases(
    n: int, rng: np.random.Generator, **kwargs
) -> Iterator[SampleCase]:
    """Yield n independently randomized four-sample fixtures."""
    if n < 0:
        raise ValueError("n must be nonnegative")
    for _ in range(n):
        yield make_sample_case(rng, **kwargs)


def generate_samples(
    n: int, rng: np.random.Generator, **kwargs
) -> list[tuple[object, object, object, object]]:
    """Return the requested list of four-sample tuples."""
    return [case.samples for case in iter_sample_cases(n, rng, **kwargs)]
