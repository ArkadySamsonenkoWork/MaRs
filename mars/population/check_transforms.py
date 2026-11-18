import math
import typing as tp

import torch
from . import choi_transforms


def _default_relative_tolerance(dtype: torch.dtype) -> float:
    """Return the default relative tolerance for the eigenvalue tests in this section.

    The tolerance is applied relative to the largest absolute element of the tested matrix, because
    the elements of relaxation superoperators can differ by many orders of magnitude.
    """
    if dtype in (torch.float64, torch.complex128):
        return 1e-10
    return 1e-5



def _complete_positivity_tolerance(matrix: torch.Tensor, atol: float, rtol: tp.Optional[float]) -> torch.Tensor:
    """Return the tolerance atol + rtol · max|matrix| for every matrix of the batch."""
    if rtol is None:
        rtol = _default_relative_tolerance(matrix.dtype)
    scale = matrix.abs().flatten(start_dim=-2).max(dim=-1).values
    return atol + rtol * scale


def is_hermiticity_preserving(
        superoperator: torch.Tensor, atol: float = 0.0, rtol: tp.Optional[float] = None
) -> torch.Tensor:
    """Check whether a map or a generator preserves Hermiticity.

    :param superoperator: Map or generator. Shape: [..., N², N²]
    :param atol: Absolute tolerance.
    :param rtol: Tolerance relative to the largest absolute element of the superoperator.
        If None, 1e-10 is used for double precision and 1e-5 for single precision.

    :return: torch.Tensor
        Boolean tensor of shape [...]: True if C = C† within the tolerance.
    """
    tolerance = _complete_positivity_tolerance(superoperator, atol, rtol)
    return choi_transforms.choi_hermiticity_deviation(superoperator) <= tolerance


def is_completely_positive_map(
        propagator: torch.Tensor, atol: float = 0.0, rtol: tp.Optional[float] = None
) -> torch.Tensor:
    """Check complete positivity of a map by the Choi criterion.

    The map is completely positive if its Choi matrix is Hermitian and positive semidefinite.

    :param propagator: Map (propagator) acting on vectorized density matrices. Shape: [..., N², N²]
    :param atol: Absolute tolerance.
    :param rtol: Tolerance relative to the largest absolute element of the superoperator.
        If None, 1e-10 is used for double precision and 1e-5 for single precision.

    :return: torch.Tensor
        Boolean tensor of shape [...].

    Mathematical Formulation:
    -------------------------
        C = C†  and  λ_min(C) ≥ -(atol + rtol · max|S|)
    """
    tolerance = _complete_positivity_tolerance(propagator, atol, rtol)
    hermitian = choi_transforms.choi_hermiticity_deviation(propagator) <= tolerance
    positive = choi_transforms.map_complete_positivity_margin(propagator) >= -tolerance
    return hermitian & positive


def is_completely_positive_generator(
        generator: torch.Tensor, atol: float = 0.0, rtol: tp.Optional[float] = None
) -> torch.Tensor:
    """Check that a generator produces a completely positive evolution exp(L t) for all t ≥ 0.

    The test is the conditional complete positivity of the Choi matrix together with Hermiticity
    preservation (Lindblad, Gorini-Kossakowski-Sudarshan criterion).

    :param generator: Generator L of dρ/dt = L ρ. Shape: [..., N², N²]
    :param atol: Absolute tolerance.
    :param rtol: Tolerance relative to the largest absolute element of the generator.
        If None, 1e-10 is used for double precision and 1e-5 for single precision.

    :return: torch.Tensor
        Boolean tensor of shape [...].

    Mathematical Formulation:
    -------------------------
        C(L) = C(L)†  and  λ_min(P⊥ C(L) P⊥) ≥ -(atol + rtol · max|L|)

    Notes:
    ------
    - The criterion holds with and without the Hamiltonian part of the generator.
    - The criterion does not require trace preservation: generators with external decay are allowed.
    """
    tolerance = _complete_positivity_tolerance(generator, atol, rtol)
    hermitian = choi_transforms.choi_hermiticity_deviation(generator) <= tolerance
    positive = choi_transforms.generator_complete_positivity_margin(generator) >= -tolerance
    return hermitian & positive


def extract_trace_rate_matrix(generator: torch.Tensor) -> torch.Tensor:
    """Return the operator M that gives the rate of change of the trace under a generator.

    :param generator: torch.Tensor
        Generator L of dρ/dt = L ρ. Shape: [..., N², N²]

    :return: torch.Tensor
        Operator M. Shape: [..., N, N]

    Mathematical Formulation:
    -------------------------
        d Tr(ρ)/dt = Tr(M ρ),     M[l, k] = Σ_i L[(ii), (kl)]

    In terms of the Choi matrix, M is the transposed partial trace of C(L) over the spin system.

    Notes:
    ------
    - M = 0 for a trace-preserving generator.
    - For a generator with the external decay term -½{Γ, ρ}, M = -Γ. The diagonal of -M contains the
      decay rates of the levels, and M must be negative semidefinite: decay can only remove population.
    - Diagonal elements of M are the column sums of the population block
      (see `extract_transition_matrix_from_superoperator`).
    """
    dim = choi_transforms._superoperator_dim(generator)
    population_rows = generator[..., torch.arange(dim, device=generator.device) * (dim + 1), :]
    traced = population_rows.sum(dim=-2)
    return traced.reshape(*generator.shape[:-2], dim, dim).transpose(-2, -1)


def map_trace_preservation_deviation(propagator: torch.Tensor) -> torch.Tensor:
    """Return the largest deviation of a map from trace preservation.

    :param propagator: Map acting on vectorized density matrices. Shape: [..., N², N²]

    :return: torch.Tensor
        max |Tr_system C - I|. Shape: [...]. Zero for a trace-preserving map.
    """
    dim = choi_transforms._superoperator_dim(propagator)
    reduced = choi_transforms.choi_partial_trace_system(choi_transforms.superoperator_to_choi(propagator))
    identity = torch.eye(dim, dtype=reduced.dtype, device=reduced.device)
    return (reduced - identity).abs().flatten(start_dim=-2).max(dim=-1).values


def generator_trace_preservation_deviation(generator: torch.Tensor) -> torch.Tensor:
    """Return the largest deviation of a generator from trace preservation.

    :param generator: Generator L of dρ/dt = L ρ. Shape: [..., N², N²]

    :return: torch.Tensor
        max |M|, where M is the trace rate matrix of `extract_trace_rate_matrix`. Shape: [...].
        Zero for a trace-preserving generator.
    """
    return extract_trace_rate_matrix(generator).abs().flatten(start_dim=-2).max(dim=-1).values