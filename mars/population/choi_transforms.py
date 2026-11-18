import math

import torch


def _superoperator_dim(superoperator: torch.Tensor) -> int:
    """Return the Hilbert space dimension N of a superoperator of shape [..., N², N²].

    :param superoperator: Superoperator. Shape: [..., N², N²]
    :return: Hilbert space dimension N.
    :raises ValueError: If the superoperator is not square or its size is not a perfect square.
    """
    rows, cols = superoperator.shape[-2:]
    dim = int(round(math.sqrt(cols)))
    if rows != cols or dim * dim != cols:
        raise ValueError(
            f"Expected a superoperator of shape [..., N^2, N^2], got {tuple(superoperator.shape)}"
        )
    return dim


def superoperator_to_choi(superoperator: torch.Tensor) -> torch.Tensor:
    """Convert a superoperator into its Choi matrix.

    The Choi matrix is the joint operator of the spin system (first factor) and an auxiliary copy
    of the spin system (second factor) obtained when the map acts on the spin system only:

        C = Σ_kl Φ(|k⟩⟨l|) ⊗ |k⟩⟨l|

    :param superoperator: torch.Tensor
        Superoperator acting on row-major vectorized density matrices. It can be a map (propagator)
        or a generator (relaxation superoperator). Shape: [..., N², N²]

    :return: torch.Tensor
        Choi matrix. Shape: [..., N², N²]. Row index i·N + k, column index j·N + l, where
        i, j are the indices of the spin system and k, l are the indices of the auxiliary copy.

    Mathematical Formulation:
    -------------------------
        C[(i,k), (j,l)] = S[(ij), (kl)]

    For a Kraus map Φ(ρ) = K ρ K† this gives C = |K⟩⟩⟨⟨K| with |K⟩⟩[(i,k)] = K[i, k],
    which shows that C ≥ 0 for every completely positive map.

    Notes:
    ------
    - The conversion is a permutation of elements. It is exact and is its own inverse.
    - Batch dimensions are preserved.
    """
    dim = _superoperator_dim(superoperator)
    batch_shape = superoperator.shape[:-2]
    tensor_4d = superoperator.reshape(*batch_shape, dim, dim, dim, dim)
    return tensor_4d.transpose(-3, -2).reshape(*batch_shape, dim * dim, dim * dim)


def choi_to_superoperator(choi: torch.Tensor) -> torch.Tensor:
    """Convert a Choi matrix back into the superoperator.

    :param choi: torch.Tensor
        Choi matrix with the ordering of `superoperator_to_choi`. Shape: [..., N², N²]

    :return: torch.Tensor
        Superoperator acting on row-major vectorized density matrices. Shape: [..., N², N²]

    Mathematical Formulation:
    -------------------------
        S[(ij), (kl)] = C[(i,k), (j,l)]
    """
    return superoperator_to_choi(choi)


def maximally_entangled_vector(
        dim: int, dtype: torch.dtype = torch.complex128, device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """Return the normalized maximally entangled vector of the spin system and its auxiliary copy.

    :param dim: Hilbert space dimension N of the spin system.
    :param dtype: Data type of the result.
    :param device: Device of the result.

    :return: torch.Tensor
        Vector |ω⟩ of shape [N²].

    Mathematical Formulation:
    -------------------------
        |ω⟩ = Σ_i |i⟩ ⊗ |i⟩ / √N,    ω[i·N + i] = 1 / √N, all other elements are zero.

    Notes:
    ------
    - The non-zero positions i·(N + 1) coincide with the positions of the populations in vec(ρ).
    """
    vector = torch.zeros(dim * dim, dtype=dtype, device=device)
    vector[torch.arange(dim, device=device) * (dim + 1)] = 1.0 / math.sqrt(dim)
    return vector


def choi_state(propagator: torch.Tensor) -> torch.Tensor:
    """Return the joint density matrix of the spin system and its auxiliary copy after the evolution.

    The spin system and the auxiliary copy start in the maximally entangled state |ω⟩⟨ω|, and only the
    spin system evolves by the map. The result is the Choi matrix divided by N.

    :param propagator: torch.Tensor
        Map (propagator) acting on row-major vectorized density matrices, for example exp(L t).
        Shape: [..., N², N²]

    :return: torch.Tensor
        Joint density matrix (Φ ⊗ id)(|ω⟩⟨ω|). Shape: [..., N², N²]

    Mathematical Formulation:
    -------------------------
        ρ_joint = (Φ ⊗ id)(|ω⟩⟨ω|) = (1 / N) Σ_kl Φ(|k⟩⟨l|) ⊗ |k⟩⟨l| = C / N

    Notes:
    ------
    - For a completely positive trace-preserving map ρ_joint is a valid density matrix:
      it is Hermitian, positive semidefinite and has unit trace.
    """
    dim = _superoperator_dim(propagator)
    return superoperator_to_choi(propagator) / dim


def project_choi_on_complement(choi: torch.Tensor) -> torch.Tensor:
    """Project a Choi matrix onto the complement of the maximally entangled vector.

    This projection is used to test complete positivity of a generator (conditional complete positivity).

    :param choi: torch.Tensor
        Choi matrix of a generator. Shape: [..., N², N²]

    :return: torch.Tensor
        Projected matrix P⊥ C P⊥. Shape: [..., N², N²]

    Mathematical Formulation:
    -------------------------
        P⊥ = I - |ω⟩⟨ω|
        P⊥ C P⊥ = C - |ω⟩⟨ω|C - C|ω⟩⟨ω| + |ω⟩⟨ω|C|ω⟩⟨ω|

    Notes:
    ------
    - Terms of the generator of the form Gρ + ρG† (Hamiltonian and anticommutator terms)
      are removed by the projection.
    - The projected matrix always has the eigenvalue 0 with the eigenvector |ω⟩.
    """
    dim = _superoperator_dim(choi)
    dtype = choi.dtype if choi.is_complex() else (torch.complex128 if choi.dtype == torch.float64 else torch.complex64)
    choi = choi.to(dtype)
    omega = maximally_entangled_vector(dim, dtype=dtype, device=choi.device)
    projector = torch.eye(dim * dim, dtype=dtype, device=choi.device) - torch.outer(omega, omega.conj())
    return projector @ choi @ projector


def hermitian_part(matrix: torch.Tensor) -> torch.Tensor:
    """Return the Hermitian part (A + A†) / 2 of a batch of square matrices.

    :param matrix: Square matrices. Shape: [..., M, M]
    :return: Hermitian part. Shape: [..., M, M]
    """
    return 0.5 * (matrix + matrix.conj().transpose(-2, -1))


def choi_hermiticity_deviation(superoperator: torch.Tensor) -> torch.Tensor:
    """Return the largest deviation of a superoperator from Hermiticity preservation.

    A superoperator preserves Hermiticity, Φ(A)† = Φ(A†) for every A, if and only if its Choi matrix
    is Hermitian. The same holds for a generator: if L preserves Hermiticity, so does exp(L t).

    :param superoperator: torch.Tensor
        Map or generator. Shape: [..., N², N²]

    :return: torch.Tensor
        Largest absolute element of C - C†. Shape: [...]. Zero for a Hermiticity-preserving superoperator.

    Mathematical Formulation:
    -------------------------
        max |C - C†|,  which is equivalent to  max |S[(ij), (kl)] - S[(ji), (lk)]*|
    """
    choi = superoperator_to_choi(superoperator)
    deviation = (choi - choi.conj().transpose(-2, -1)).abs()
    return deviation.flatten(start_dim=-2).max(dim=-1).values


def choi_eigenvalues(choi: torch.Tensor) -> torch.Tensor:
    """Return the eigenvalues of the Hermitian part of a Choi matrix in ascending order.

    :param choi: torch.Tensor
        Choi matrix (or projected Choi matrix). Shape: [..., N², N²]

    :return: torch.Tensor
        Real eigenvalues in ascending order. Shape: [..., N²]

    Notes:
    ------
    - Only the Hermitian part is used. Check the Hermiticity separately with `choi_hermiticity_deviation`.
    """
    return torch.linalg.eigvalsh(hermitian_part(choi))


def map_complete_positivity_margin(propagator: torch.Tensor) -> torch.Tensor:
    """Return the smallest eigenvalue of the Choi matrix of a map.

    By the Choi theorem the map is completely positive if and only if this value is non-negative.

    :param propagator: torch.Tensor
        Map (propagator) acting on row-major vectorized density matrices. Shape: [..., N², N²]

    :return: torch.Tensor
        Smallest eigenvalue of C. Shape: [...]. Negative values show the violation of complete positivity.

    Notes:
    ------
    - The value is N times the smallest eigenvalue of the joint density matrix `choi_state`.
    """
    return choi_eigenvalues(superoperator_to_choi(propagator))[..., 0]


def generator_complete_positivity_margin(generator: torch.Tensor) -> torch.Tensor:
    """Return the smallest eigenvalue of the projected Choi matrix of a generator.

    The evolution exp(L t) is completely positive for all t ≥ 0 if and only if L preserves Hermiticity
    and this value is non-negative (conditional complete positivity).

    :param generator: torch.Tensor
        Generator L of the evolution dρ/dt = L ρ, for example a relaxation superoperator with or
        without the Hamiltonian part. Shape: [..., N², N²]

    :return: torch.Tensor
        Smallest eigenvalue of P⊥ C(L) P⊥. Shape: [...].
        Because the projected matrix always has the eigenvalue 0 on |ω⟩, the value is never positive:
        it is zero (up to rounding) for a completely positive generator and negative otherwise.

    Mathematical Formulation:
    -------------------------
    For a Lindblad generator  L(ρ) = -i[H, ρ] + Σ_k (L_k ρ L_k† - ½{L_k† L_k, ρ}) - ½{Γ, ρ}:

        P⊥ C(L) P⊥ = Σ_k P⊥ |L_k⟩⟩⟨⟨L_k| P⊥ ≥ 0
    """
    return choi_eigenvalues(project_choi_on_complement(superoperator_to_choi(generator)))[..., 0]


def choi_partial_trace_system(choi: torch.Tensor) -> torch.Tensor:
    """Return the partial trace of a Choi matrix over the spin system.

    :param choi: torch.Tensor
        Choi matrix with the ordering of `superoperator_to_choi`. Shape: [..., N², N²]

    :return: torch.Tensor
        Operator on the auxiliary copy. Shape: [..., N, N]

    Mathematical Formulation:
    -------------------------
        (Tr_system C)[k, l] = Σ_i C[(i,k), (i,l)] = Σ_i S[(ii), (kl)]

    Notes:
    ------
    - A map is trace-preserving if and only if the result is the identity matrix.
    - A generator is trace-preserving if and only if the result is zero.
    """
    dim = _superoperator_dim(choi)
    tensor_4d = choi.reshape(*choi.shape[:-2], dim, dim, dim, dim)
    return torch.diagonal(tensor_4d, dim1=-4, dim2=-2).sum(dim=-1)