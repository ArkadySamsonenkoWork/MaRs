import math
import typing as tp
import functools

import torch


def basis_transformation(basis_old: torch.Tensor, basis_new: torch.Tensor) -> torch.Tensor:
    """
    Given two orthonormal bases represented as matrices whose columns are basis vectors,
    this function returns the unitary matrix :math:`U` such that for any vector `v`:

        v_in_basis_2 = U @ v_in_basis_1

    The transformation is computed as:
        U = basis_new^† @ basis_old

    where :math:`^\dagger` denotes the conjugate transpose (Hermitian adjoint).

    :param basis_old: The basis function.

    The shape is [..., K, K], where K is spin dimension size.
    The column eigenvectors[:,i] is the eigenvector corresponding to the eigenvalue eigenvalues[i].

    :param basis_new: The basis function. The shape is [..., K, K], where K is spin dimension size.
    The column eigenvectors[:,i] is the eigenvector corresponding to the eigenvalue eigenvalues[i].

    :return: A transformation matrix of shape [..., K, K] that transforms
            vectors from the `basis_old` coordinate system to the `basis_new` coordinate system.

    torch.Tensor
        A tensor of shape [..., K, K] containing the squared absolute values of the
        transformation coefficients between the two bases.

        For a 2×2 case, the output can be visualized as:

    .. code-block:: text
        :class: no-copybutton

        ┌───────────────────────────────────────────┐
        │                                           │
        │     basis_old states →                      │
        │    ┌─────────────┬─────────────┐          │
        │    │             │             │          │
        │    │ ⟨b2₀|b1₀⟩   │ ⟨b2₀|b1₁⟩   │          │
        │ b  │             │             │          │
        │ a  │             │             │          │
        │ s  │ ⟨b2₁|b1₀⟩   │ ⟨b2₁|b1₁⟩   │          │
        │ i  │             │             │          │
        │ s  │             │             │          │
        │ _  └─────────────┴─────────────┘          │
        │ 2                                         │
        │                                           │
        │ s                                         │
        │ t                                         │
        │ a                                         │
        │ t                                         │
        │ e                                         │
        │ s                                         │
        │ ↓                                         │
        └───────────────────────────────────────────┘

    ...
    Mathematical Formulation:
    -------------------------
    U = basis_new^† @ basis_old

    where ^† denotes conjugate transpose. For any state vector |ψ⟩:
        |ψ⟩₂ = U |ψ⟩₁

    Notes:
    ------
    - Both bases must be orthonormal (columns form unitary matrices)
    - U is unitary: U^† U = I

    """
    return torch.matmul(basis_new.conj().transpose(-1, -2), basis_old)


def get_transformation_probabilities(basis_old: torch.Tensor, basis_new: torch.Tensor):
    """Calculate the squared absolute values of transformation coefficients
    between two bases.

    This function computes the overlap probabilities between states in two different bases.
    The output values represent |⟨basis_2_i|basis_1_j⟩|², which are the squared magnitudes
    of probability amplitudes in quantum mechanics.

    The transformation is computed as:
        U = |basis_new^† @ basis_old|^2

    :param basis_old: (b1) torch.Tensor
        The first basis tensor with shape [..., K, K], where K is the spin dimension size.
        Each column basis_old[:,j] represents an eigenvector in the first basis.

    :param basis_new: (b2) torch.Tensor
        The second basis tensor with shape [..., K, K], where K is the spin dimension size.
        Each column basis_new[:,i] represents an eigenvector in the second basis.

    :return: torch.Tensor
        A tensor of shape [..., K, K] containing the squared absolute values of the
        transformation coefficients between the two bases.

        For a 2×2 case, the output can be visualized as:

    .. code-block:: text
        :class: no-copybutton

        ┌───────────────────────────────────────────┐
        │                                           │
        │     basis_old states →                    │
        │    ┌─────────────┬─────────────┐          │
        │    │             │             │          │
        │    │ |⟨b2₀|b1₀⟩|² │ |⟨b2₀|b1₁⟩|² │        │
        │ b  │             │             │          │
        │ a  │             │             │          │
        │ s  │ |⟨b2₁|b1₀⟩|² │ |⟨b2₁|b1₁⟩|² │        │
        │ i  │             │             │          │
        │ s  │             │             │          │
        │ _  └─────────────┴─────────────┘          │
        │ 2                                         │
        │                                           │
        │ s  Element [i, j] represents the          │
        │ t  probability of finding the system      │
        │ a  in state i of basis_new when it was    │
        │ t  prepared in state j of basis_old.      │
        │ e                                         │
        │ s                                         │
        │ ↓                                         │
        └───────────────────────────────────────────┘
    ...
    Mathematical Formulation:
    -------------------------
    P[i, j] = |⟨basis_new[:, i] | basis_old[:, j]⟩|² = |U[i, j]|²

    where U = basis_new^† @ basis_old is the unitary transformation matrix.

    Notes:
    ------
    - These coefficients apply ONLY to incoherent quantities (populations, rate matrices)
    - DO NOT use for coherent operators (Hamiltonians, density matrices) — use full complex U instead
    - Values satisfy Σ_i P[i, j] = 1 (columns sum to unity)
    """
    transforms = basis_transformation(basis_old, basis_new)
    return transforms.abs().square()


def compute_liouville_basis_transformation(basis_old: torch.Tensor, basis_new: torch.Tensor):
    """Compute the transformation matrix for superoperators between two quantum
    bases.

    This function calculates the unitary transformation matrix that converts Liouville-space
    superoperators (e.g., relaxation matrices) from an old basis to a new basis. The transformation
    preserves the structure of quantum operations under basis change.

    :param basis_old : torch.Tensor
        Original basis vectors. Shape: ``[..., K, K]`` where K is the Hilbert space dimension.
        Columns represent eigenvectors of the old basis.

    :param basis_new : torch.Tensor
        Target basis vectors. Shape: ``[..., K, K]`` where K is the Hilbert space dimension.
        Columns represent eigenvectors of the new basis.

    :return: torch.Tensor
        Transformation matrix for Liouville space operators. Shape: ``[..., K², K²]``

        For a 2×2 system (K=2), the output structure can be visualized as:
    .. code-block:: text
        :class: no-copybutton

        ┌───────────────────────────────────────────────────────  ┐
        │                                                         │
        │  Old Liouville basis states →                           │
        │  ┌─────────────┬─────────────┬─────────────┬─────────┐  │
        │  │⟨b₂₀b₂₀|b₁₀b₁₀⟩ ...       │⟨b₂₀b₂₀|b₁₁b₁₁⟩ ...     │  │
        │  │ ...         │ ...         │ ...         │ ...     │  │
        │L │⟨b₂₀b₂₁|b₁₀b₁₀⟩ ...       │⟨b₂₀b₂₁|b₁₁b₁₁⟩ ...     │  │
        │i │ ...         │ ...         │ ...         │ ...     │  │
        │o │⟨b₂₁b₂₀|b₁₀b₁₀⟩ ...       │⟨b₂₁b₂₀|b₁₁b₁₁⟩ ...     │  │
        │u │ ...         │ ...         │ ...         │ ...     │  │
        │v │⟨b₂₁b₂₁|b₁₀b₁₀⟩ ...       │⟨b₂₁b₂₁|b₁₁b₁₁⟩ ...     │  │
        │i └─────────────┴─────────────┴─────────────┴─────────┘  │
        │l                                                        │
        │l                                                        │
        │e                                                        │
        │  New Liouville basis states ↓                           │
        └─────────────────────────────────────────────────────────┘

    Let ``U = basis_new.conj().transpose(-1, -2) @ basis_old`` be the unitary that
    transforms coordinantes from basis_old to basis_new: ``v_in_basis_2 = U @ v_in_basis_1``.

    Then:
      - **Vectorized operators** (e.g., density matrices flattened via row stacking)
        transform as:
            vec(R_new) = kron(U, U.conj()) @ vec(R_old), where U.conj() is U* - complex conjugate

      - **Superoperators** (matrices acting on vectorized operators, such as relaxation
        generators L) transform by similarity:
            L_new = kron(U, U.conj()) @ L_old @ kron(U, U.conj()).conj().transpose(-1, -2)

    Thus, the returned ``T_switch`` equals ``kron(U, U.conj())``, and it is the
    fundamental building block for both operator and superoperator basis changes.

    Mathematical Formulation:
    -------------------------
    Let the unitary basis transformation be defined as:

        U = basis_new^† @ basis_old,

    where ^† denotes conj().transpose() or conjugate transpose. This maps the coordinate of vectors as |ψ⟩₂ = U |ψ⟩₁.

    For a density matrix ρ, the transformation is:

        ρ₂ = U ρ₁ U^†

    When ρ is vectorized using **row-major ordering** (as in `vec` function),
    the vectorized form transforms linearly as:

        |ρ₂⟩⟩ = T |ρ₁⟩⟩,   where   T = U ⊗ U*,

    and ⊗ denotes the Kronecker product, while U* is the element-wise complex conjugate of U.

    Consequently, any superoperator L acting in Liouville space (e.g., a Liouvillian L
    such that d|ρ⟩⟩/dt = L |ρ⟩⟩) transforms under a similarity transformation:

        L = T L T^†.

    The function returns T = U ⊗ U*, which serves as the transformation
    operator for vectorized quantum states and superoperators under the given basis change.

    Notes:
    ------
    - The returned T is unitary if U is unitary (which holds when both bases are orthonormal).
    - This formulation assumes row-major vectorization (C-order flattening), consistent with
     `vec(ρ)` implementation.
    """
    U = basis_transformation(basis_old, basis_new)
    T_switch = batched_kron(U, U.conj())
    return T_switch


def transform_operator_to_new_basis(density_old: torch.Tensor, coeffs: torch.Tensor):
    """Transform a Hilbert operator (density matrix) to a new  basis.

    Applies a unitary transformation to a density matrix using precomputed transformation coefficients.

    :param density_old : torch.Tensor
        Density matrix in original basis. Shape: [..., K, K]
        Must be Hermitian with trace 1 for physical states.

    :param coeffs : torch.Tensor
        Precomputed basis transformation matrix. Shape: [..., K, K]
        Typically from `basis_transformation`. It is equel to new_basis.conj().transpose() @ old_basis

    :return:
    torch.Tensor
        Density matrix in new basis. Shape: [..., K, K]

    Mathematical Formulation:
    -------------------------
    ρ_new = U @ ρ_old @ U†
    where U = transformation_matrix

    Notes:
    ------
    - Diagonal elements represent populations in the new basis
    - Off-diagonal elements represent dephasing in the new basis
    """
    return coeffs @ density_old @ coeffs.conj().transpose(-1, -2)


def transform_rate_matrix_to_new_basis(initial_rates: torch.Tensor, probabilities: torch.Tensor) -> torch.Tensor:
    """Transform transition rates from matrix form to new basis set.

    K(b_new_1 -> b_new_2) = |⟨b_new_1|b_old_1⟩|² * |⟨b_new_2|b_old_2⟩|² * K(b_old_1 -> b_old_2)

    WARNING: This transformation applies only when initial transition levels (i, j)
    do not transform into identical levels.
    If transitions exist between levels K1 <-> K2 and they transform into identical levels
    (N = a*K1 + b*K2), correlation terms arise between levels that pure relaxation rates
    cannot describe correctly.

    :param initial_rates: Transition rates matrix. Shape [..., K, K]. Diagonal elements must be zero
    :param probabilities: Transformation coefficients (see get_transformation_probabilities). Shape [..., K, K]
    :return: Transformed rate matrix

    Mathematical Formulation:
    -------------------------
    R_new = |U|^2 @ R_old @ |U^†|^2

    R_new[i, j] =  Σ_{m,n} |⟨i|n⟩|² R_old[n, m] |⟨j|m⟩|²
                = probabilities @ R_old @ probabilities.transpose()

    Notes:
    ------
    This applies ONLY to incoherent rate matrices (populations/relaxation).
    DO NOT use for coherent operators (Hamiltonians, density matrices) — use:
        H_new = U @ H_old @ U^†   (with full complex U, not |U|²)
    """
    return probabilities @ initial_rates @ probabilities.transpose(-1, -2)


def transform_dephasing_to_population_transfer(
    initial_dephasing_rates: torch.Tensor,
    probabilities: torch.Tensor,
) -> torch.Tensor:
    """Transform pure dephasing rates from the initial basis to an effective population-transfer matrix
    in the new basis.

    For pure dephasing defined in the initial basis by rates gamma_a, the effective population-transfer
    rates in the new basis are

        K_new[i, j] = Σ_a gamma_a * |⟨i|a⟩|² * |⟨j|a⟩|²

    or, in matrix form,
        K_new = probabilities @ diag(initial_dephasing_rates) @ probabilities.transpose(-1, -2)
    This implementation avoids creating diag(initial_dephasing_rates) explicitly by using broadcasting.

    :param initial_dephasing_rates: Pure dephasing rates in the initial basis. Shape [..., K]
    :param probabilities: Basis-transformation probabilities, defined as
                          probabilities[i, a] = |⟨i_new | a_old⟩|².
                          Shape [..., K_new, K]
    :return: Effective population-transfer matrix in the new basis. Shape [..., K_new, K_new]

    Mathematical Formulation:
    -------------------------
    K_new = C @ diag(gamma) @ C^T
    K_new[i, j] = Σ_a C[i, a] * gamma[a] * C[j, a]
    """
    return torch.einsum("...ia,...a,...ja->...ij", probabilities, initial_dephasing_rates, probabilities)


def transform_state_weights_to_new_basis(initial_rates: torch.Tensor, probabilities: torch.Tensor) -> torch.Tensor:
    """Transform a state_weights (populations, loss) from old basis to new basis using transformation
    coefficients.

    Applies the transformation: v_new[i] = Σ_j |⟨new_i|old_j⟩|² * v_old[j]

    This can be used to transform:
    - Population vectors (state occupancies)
    - Outward transition rates
    - Any other quantities that transform linearly with basis overlap probabilities

    :param initial_rates: Values in the old basis. Shape [..., K]
    :param probabilities: Transformation probabilities |⟨new|old⟩|². Shape [..., K, K]
    :return: Transformed values in the new basis. Shape [..., K]

    Mathematical Formulation:
    -------------------------
    v_new[i] = Σ_j |⟨i|j⟩|² v_old[j] = probabilities @ v_old

    Notes:
    ------
    - Valid for populations, state occupancies, and other incoherent quantities
    - Preserves total sum: Σ_i v_new[i] = Σ_j v_old[j]
    """
    return torch.einsum("...ij, ...j -> ...i", probabilities, initial_rates)


def transform_superop_to_new_basis(
    superoperator_old: torch.Tensor,
    liouville_transformation: torch.Tensor
) -> torch.Tensor:
    """Transform a superoperator to a new quantum basis in Liouville space.

    Applies a basis transformation to Liouville-space operators (e.g., relaxation matrices,
    quantum maps) using precomputed Liouville transformation coefficients.

    :param superoperator_old : torch.Tensor
        Superoperator in original Liouville basis. Shape: [..., K², K²]

    :param liouville_transformation : torch.Tensor
        Precomputed Liouville-space transformation matrix. Shape: [..., K², K²]
        from "compute_liouville_basis_transformation", for example

    :return: torch.Tensor
        Superoperator in new Liouville basis. Shape: [..., K², K²]

    Mathematical Formulation:
    -------------------------
    R_new = T @ R_old @ T†
    where T = liouville_transformation
    """
    return liouville_transformation @ superoperator_old @ liouville_transformation.conj().transpose(-1, -2)


def transform_superop_diag_to_new_basis(
    superoperator_diag: torch.Tensor,
    liouville_transformation: torch.Tensor
) -> torch.Tensor:
    """Transform a diagonal superoperator to a new basis in Liouville
    space.

    :param superoperator_diag : torch.Tensor
        Diagonal of the superoperator in the original Liouville basis.
        Shape: [..., K²]

    :param liouville_transformation : torch.Tensor
        Precomputed Liouville-space transformation matrix.
        Shape: [..., K², K²]

    :return: torch.Tensor
        Transformed superoperator in the new Liouville basis.
        Shape: [..., K², K²]

    Mathematical Formulation:
    -------------------------
    L_new = T @ diag(lambda) @ T^†

    Notes:
    ------
    - More efficient than forming full diagonal matrix first
    """
    return (liouville_transformation * superoperator_diag.unsqueeze(-2))\
        @ liouville_transformation.conj().transpose(-1, -2)


def batched_kron(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Batched Kronecker product.

    Computes the Kronecker product for batched matrices.

    :param a: torch.Tensor of shape [..., M, N]
    :param b: torch.Tensor of shape [..., P, Q]
    :return: torch.Tensor of shape [..., M*P, N*Q]
    """
    *_, M, N = a.shape
    *_, P, Q = b.shape

    a_expanded = a[..., :, None, :, None]
    b_expanded = b[..., None, :, None, :]
    result = a_expanded * b_expanded
    *batch_dims, _, _, _, _ = result.shape

    return result.reshape(*batch_dims, M * P, N * Q)


def batched_multi_kron(operators: tp.Sequence[torch.Tensor]) -> torch.Tensor:
    """Compute Kronecker product of multiple tensors.

    Computes the Kronecker product: rho_1 ⊗ rho_2 ⊗ ... ⊗ rho_n

    The Kronecker product is computed sequentially from left to right:
    ((rho_1 ⊗ rho_2) ⊗ rho_3) ⊗ ... ⊗ rho_n

    :param operators:
        torch.tensors: List of torch.Tensor, each of shape [..., n_i, m_i]
                 All tensors must have compatible batch dimensions.
                 Each tensor represents a matrix.

    :return:
        torch.Tensor of shape [..., N, M] where:
        - N = n_1 * n_2 * ... * n_k (product of all row dimensions)
        - M = m_1 * m_2 * ... * m_k (product of all column dimensions)
        - Batch dimensions are broadcast according to standard PyTorch rules

    Raises:
        ValueError: If tensors list is empty

    Example:
        >>> rho_1 = torch.randn(10, 2, 2)  # batch of 10, 2x2 matrices
        >>> rho_2 = torch.randn(10, 3, 3)  # batch of 10, 3x3 matrices
        >>> rho_3 = torch.randn(10, 2, 2)  # batch of 10, 2x2 matrices
        >>> result = batched_multi_kron([rho_1, rho_2, rho_3])
        >>> result.shape
        torch.Size([10, 12, 12])  # 2*3*2 = 12
    """
    return functools.reduce(batched_kron, operators)


def batched_sum_kron(
    operators: tp.Sequence[torch.Tensor],
) -> torch.Tensor:
    """
    Compute the sum of local operators:
        K_1 ⊗ I ⊗ ... ⊗ I
      + I ⊗ K_2 ⊗ ... ⊗ I
      + ...
      + I ⊗ ... ⊗ I ⊗ K_n

    This constructs operators like total Hamiltonians or relaxation superoperators
    where each term acts nontrivially on only one subsystem.
      
    :param operators:
        torch.tensors: List of torch.Tensor, each of shape [..., n_i, m_i]
                 All tensors must have compatible batch dimensions.
                 Each tensor represents a matrix.
    :return:
        torch.Tensor
            Tensor of shape (..., D, D), where:
                D = d_1 * d_2 * ... * d_n

    Mathematical Formulation:
    -------------------------
    For n subsystems with local operators {Kᵢ} and identity operators {Iⱼ}:

        H_global = Σᵢ (I₁ ⊗ ... ⊗ Iᵢ₋₁ ⊗ Kᵢ ⊗ Iᵢ₊₁ ⊗ ... ⊗ Iₙ)

    Notes:
    ------
    - Preserves Hermiticity: if all Kᵢ are Hermitian, H_global is Hermitian
    """

    device = operators[0].device
    dtype = operators[0].dtype
    batch_shape = operators[0].shape[:-2]

    dims = [K.shape[-1] for K in operators]
    identities = [
        torch.eye(d, device=device, dtype=dtype).expand(batch_shape + (d, d))
        for d in dims
    ]

    n = len(operators)
    left_ids = [None] * n
    right_ids = [None] * n

    for i in range(1, n):
        left_ids[i] = identities[i - 1] if i == 1 else batched_kron(left_ids[i - 1], identities[i - 1])

    for i in range(n - 2, -1, -1):
        right_ids[i] = identities[i + 1] if i == n - 2 else batched_kron(identities[i + 1], right_ids[i + 1])
    total = None
    for i, K in enumerate(operators):
        term = K
        if left_ids[i] is not None:
            term = batched_kron(left_ids[i], term)
        if right_ids[i] is not None:
            term = batched_kron(term, right_ids[i])

        total = term if total is None else total + term
    return total


def batched_sum_kron_diagonal(
        vectors: tp.Sequence[torch.Tensor],
) -> torch.Tensor:
    """Compute diagonal of summed operators.

    Computes the diagonal vector of:
        H = Σᵢ (I ⊗ ... ⊗ diag(vᵢ) ⊗ ... ⊗ I)

    where each term acts nontrivially only on subsystem i. The result is equivalent to:
        diag(H) = Σᵢ (1 ⊗ ... ⊗ vᵢ ⊗ ... ⊗ 1)

    :param vectors:
        torch.tensors: List of torch.Tensor, each of shape [..., n_i]
            All tensors must have compatible batch dimensions.
            Each tensor represents a matrix.

    :return:
        torch.Tensor
            Tensor of shape (...,  D), where:
            D = d_1 * d_2 * ... * d_n

    Mathematical Formulation:
    -------------------------
    For diagonal local operators diag(vᵢ):
        diag(H_global)[i₁,i₂,...,iₙ] = Σₖ vₖ[iₖ]
    where the composite index (i₁,i₂,...,iₙ) maps to linear index via row-major ordering.

    This avoids O(D²) memory cost of full matrix construction, using only O(D) storage
    """
    batch_shape = vectors[0].shape[:-1]
    device = vectors[0].device
    dtype = vectors[0].dtype

    dims = [v.shape[-1] for v in vectors]
    D = 1
    for d in dims:
        D *= d

    n = len(vectors)
    diag = torch.zeros(batch_shape + (D,), dtype=dtype, device=device)
    full_shape = batch_shape + tuple(dims)

    for i, v in enumerate(vectors):
        view_shape = batch_shape + tuple(
            dims[j] if j == i else 1 for j in range(n)
        )
        v_view = v.reshape(view_shape)
        v_full = v_view.expand(full_shape)
        diag += v_full.reshape(batch_shape + (D,))
    return diag


def compute_clebsch_gordan_coeffs(
        target_basis: torch.Tensor,
        basis_list: list[torch.Tensor]
) -> torch.Tensor:
    """ Compute transformation coefficients expressing a target basis as linear combinations
    of tensor product states from subsystem bases.

    For n old bases with dimensions [k1, k2, ..., kn], computes coefficients C
    that express each new basis state as a linear combination of tensor products:

    Returns coefficients C where:
        C[m, i₁, i₂, ..., iₙ] = ⟨target_m | (|i₁⟩ ⊗ |i₂⟩ ⊗ ... ⊗ |iₙ⟩)

    where ⊗ denotes the tensor (Kronecker) product.

    :param target_basis: new basis vector. Shape: [..., K, K] where K = k1*k2*...*kn
    Columns are orthonormal eigenvectors: target_basis[:, m] = |m⟩_target
    :param basis_list: List of old basis tensors.
                             Each has shape [..., k_i, k_i]
    :return: Transformation coefficients C. Shape: [..., K, k₁, k₂, ..., K]
             where C[i₁, ..., iₙ, m] = ⟨target_m | product_{i₁,...,iₙ}⟩

    Mathematical Formulation:
    -------------------------
    For product basis states |i₁,...,iₙ⟩_prod = |i₁⟩ ⊗ ... ⊗ |iₙ⟩ and target states |m⟩_target:

        |ψ⟩ = Σ_{i₁..iₙ} c_prod[i₁..iₙ] |i₁..iₙ⟩_prod
             = Σ_m c_target[m] |m⟩_target

    Coordinate transformation:
        c_target = U @ c_prod   where   U[m, composite_index(i₁..iₙ)] = ⟨m|_target |i₁..iₙ⟩_prod

    Computed via:
        C_flat  = target_basis.conj().transpose(-1,-2) @ kron_basis

    Notes:
    ------
    - Coefficients follow convention in the coordinate transformation as in function (:func:basis_transformation):
    U transforms coordinates from product → target basis
    """
    dims = [basis.shape[-1] for basis in basis_list]
    kron_basis = batched_multi_kron(basis_list)
    C_flat = torch.matmul(target_basis.conj().transpose(-1, -2), kron_basis)
    C_reshaped = C_flat.reshape(*C_flat.shape[:-2], *dims, C_flat.shape[-1])
    return C_reshaped


def compute_clebsch_gordan_probabilities(
        target_basis: torch.Tensor,
        basis_list: list[torch.Tensor]
) -> torch.Tensor:
    """Compute squared magnitudes of transformation coefficients (probabilities).

    Returns P[m, i₁, ..., iₙ] = |⟨target_m | product_{i₁..iₙ}⟩|²

    :param target_basis: New-system basis vectors. Shape: [..., K, K] where K = k1*k2*...*kn
    :param basis_list: List of old basis tensors.
                                  Each has shape [..., k_i, k_i]
    :return: Squared Clebsch-Gordan coefficients. Shape: [..., k1, k2, ..., kn, K]

    Mathematical Formulation:
    -------------------------
        P[m, i₁..iₙ] = |U[m, composite_index(i₁..iₙ)]|²
                     = |⟨target_m | product_{i₁..iₙ}⟩|²
    """
    return compute_clebsch_gordan_coeffs(target_basis, basis_list).abs().square()


def get_product_to_target_unitary(
        coeffs: torch.Tensor,
        n: int,
) -> torch.Tensor:
    """Reshape transformation coefficients into unitary matrix U.

    Converts C[i₁, ..., iₙ] → U[m, composite_index(i₁..iₙ)

    :param coeffs: Clebsch-Gordan coefficients. Shape: [..., k1, k2, ..., kn, K]
    :param n: number of subsystems
    :return: Unitary transformation matrix U. Shape: [..., K, K]

    Mathematical Formulation:
    -------------------------
        U = reshape_and_transpose(coeffs)
        such that U[m, j] = coeffs[i₁..iₙ, m] where j = composite_index(i₁..iₙ)

    This satisfies convention:
        c_target = U @ c_product
        ρ_target = U @ ρ_product @ U†
    """
    batch_shape = coeffs.shape[:-n - 1]
    uncoupled_dims = coeffs.shape[-n - 1:-1]
    total_dim = int(torch.prod(torch.tensor(uncoupled_dims)))
    K = coeffs.shape[-1]
    return coeffs.reshape(*batch_shape, K, total_dim)


def transform_kronecker_populations(
        populations_list: list[torch.Tensor],
        probabilities: torch.Tensor,
) -> torch.Tensor:
    """Transform populations from old (product) basis to new basis
    using.

    Clebsch-Gordan coefficients.

    Computes the population of each new-system state as:
    n_coupled[m] = Σ_{i1,...,in} |C[i1, ..., in, m]|² * n_old_1[i1] * ... * n_old_n[in]

    This implements the quantum mechanical rule that populations of subsystems
    multiply, weighted by the squared Clebsch-Gordan coefficients.

    Example:
    :param populations_list: List of population vectors for each  subsystem.
                            Each has shape [..., k_i]
    :param probabilities: Joint probabilities [..., k₁ * ... * kₙ, K]
                          where K = dimension of new basis
        Transformation probabilities from kronecker basis U1 ⊗ U2 ⊗ ... ⊗ Uk to system basis Us
        The U1, ..., Uk have shape [..., ki], where K = ∏ᵢ k_i
        Us has shape [..., K]
    :return: Populations in new-system basis. Shape: [..., K]

    Mathematical Formulation:
    -------------------------
        n_target[m] = Σ_{i₁..iₙ} |U[i₁..iₙ, m]|² · n₁[i₁] · n₂[i₂] · ... · nₙ[iₙ] =\
           = |U|^2 @ (n₁ ⊗ n₂ ⊗ ... ⊗ nₙ)

    Direct computation equivalent to full basis transformation:
        n_new = |U|^2 @ (n₁ ⊗ n₂ ⊗ ... ⊗ nₙ)
    where U = V_new^† @ (V₁ ⊗ V₂ ⊗ ... ⊗ Vₙ) is the composite transformation matrix
    and |U|^2 denotes element-wise squaring.
    """
    return transform_state_weights_to_new_basis(
        batched_multi_kron([v.unsqueeze(-1) for v in populations_list]).squeeze(-1),
        probabilities
    )


def transform_kronecker_rate_vector(
        vector_list: list[torch.Tensor],
        probabilities: torch.Tensor,
) -> torch.Tensor:
    """Transform rate vector from old (product) basis to new-system basis
    using.

    Computes the population of each new-system state as:
    R_coupled[m] = Σ_{i1,...,in} |C[i1, ..., in, m]|² * (K1[i1] + K2[i2] + ... + Kn[in])

    This implements the mechanical rule that rates from old subsystems
    multiply, weighted by the squared Clebsch-Gordan coefficients.

    Direct computation equivalent to full basis transformation:
        Γ_new = |U|^2 @ diag(Γ₁ ⊗ 𝟙₂ ⊗ ... + 𝟙₁ ⊗ Γ₂ ⊗ ... + ...)
    where the diagonal operator in the product basis has Kronecker-sum structure.

    Example:
    :param probabilities: Joint probabilities [..., k₁ * ... * kₙ, K]
                          where K = dimension of new basis
    Transformation probabilities from kronecker basis U1 ⊗ U2 ⊗ ... ⊗ Uk to system basis Us
    The U1, ..., Uk have shape [..., ki], where K = ∏ᵢ k_i
    Us has shape [..., K]
    :param vector_list: List of population vectors for each  subsystem.
                            Each has shape [..., k_i]
    :return: Populations in new-system basis. Shape: [..., K]

    Mathematical Formulation:
    -------------------------
        Γ_target[m] = Σ_{i₁..iₙ} |U[m, index(i₁..iₙ)]|² · (Γ₁[i₁] + Γ₂[i₂] + ... + Γₙ[iₙ])
                   = |U|^2 @ vec(diag(Γ₁) ⊗ 𝟙₂ ⊗ ... + 𝟙₁ ⊗ diag(Γ₂) ⊗ ... + ...)

        Here it is used that  |U|^2  @ 1 = 1, where 1 is vector of ones (not eye matrix)
    """
    return transform_state_weights_to_new_basis(
        batched_sum_kron_diagonal(vector_list),
        probabilities
    )


def transform_kronecker_rate_matrix(
        matrices: list[torch.Tensor],
        coeffs: torch.Tensor,
) -> torch.Tensor:
    """Transform local intra-system rate matrices to the total eigenbasis.

    For each subsystem s and each local transition j -> i with rate w^(s)_ij,
    the Lindblad jump operator is L_s(i,j) = I_1 ⊗ ... ⊗ |i><j|_s ⊗ ... ⊗ I_N.
    Under the secular approximation the eigenbasis population transfer rate is:

        Γ[a, b] = Σ_s Σ_{i≠j} w^(s)_ij · |<a| L_s(i,j) |b>|²

    The matrix element expands as a contraction over the R = K/d_s remaining
    subsystem indices (see §B.2 for derivation):

        T_s[a,b,i,j] = Σ_r  Ũ[a,i,r] · Ũ*[b,j,r]    (einsum '...air,...bjr->...abij')
        Γ^(s)[a,b]   = Σ_{i,j} w^(s)_ij · |T_s[a,b,i,j]|²   (einsum '...abij,...ij->...ab')
        Γ            = Σ_s Γ^(s)

    where Ũ is U reshaped to [..., K, d_s, R] with the s-th product-basis
    axis isolated.

    :param matrices: Local rate matrices for each subsystem.
                     matrices[s] has shape [..., d_s, d_s], where
                     matrices[s][..., i, j] is the rate of transition j -> i
                     within subsystem s. Diagonal entries are zeroed internally.
    :param coeffs:   Unitary from transformation matrix, shape [..., K, K].
                     U[..., m, j] = <m|j>, rows are eigenstates, columns are
                     flattened product-basis states.
    :return:         Eigenbasis transition rate matrix Γ, shape [..., K, K].
                     Γ[a, b] is the population transfer rate from |b> to |a>.
                     Diagonal is zero (outgoing rates must be set separately).
    """
    dims = [m.shape[-1] for m in matrices]
    nb = len(coeffs.shape) - 2
    K  = coeffs.shape[-2]

    U_full = coeffs.view(*coeffs.shape[:-1], *dims)

    result = torch.zeros(*coeffs.shape[:-2], K, K, dtype=coeffs.dtype, device=coeffs.device)

    for s, W_s in enumerate(matrices):
        perm = list(range(U_full.dim()))
        perm.pop(nb + 1 + s)
        perm.insert(nb + 1, nb + 1 + s)
        U_perm = U_full.permute(*perm)

        ds = dims[s]
        R  = K // ds
        U_flat = U_perm.reshape(*coeffs.shape[:-2], K, ds, R)

        T = torch.einsum("...air,...bjr->...abij", U_flat, U_flat.conj())
        w = W_s.clone()
        w.diagonal(dim1=-2, dim2=-1).zero_()

        result += torch.einsum("...abij,...ij->...ab", T.abs().square(), w)

    return result.real


def transform_kronecker_rate_matrix_v2(
        matrices: list[torch.Tensor],
        coeffs: torch.Tensor,
) -> torch.Tensor:
    """Transform intra‑subsystem rate matrices to total eigenbasis.

    Computes the total population rate matrix in the eigenbasis using
    the full complex Clebsch‑Gordan coefficients.

    :param matrices: List of rate matrices for each subsystem in its own initial basis.
                     Each has shape [..., k_i, k_i] where k_i is subsystem dimension.
    :param coeffs: Complex transformation coefficients.
                   Shape [..., k1, k2, ..., kn, K] where K = prod(k_i).
                   coeffs[..., i1, i2, ..., in, a] = ⟨a | i1…in⟩
    :return: Total rate matrix in eigenbasis. Shape [..., K, K].
             W_eig[..., a, b] is rate from eigenstate b to eigenstate a.
    """
    return transform_rate_matrix_to_new_basis(
        batched_sum_kron(matrices),
        coeffs
    )


def transform_kronecker_dephasing_to_population_transfer(
        dephasing_list: list[torch.Tensor],
        probabilities: torch.Tensor,
) -> torch.Tensor:
    """
    Transform pure dephasing rates from a list of tensors to an effective population-transfer matrix.

    Computes: K_new = P @ kron_sum(gamma_i) @ P^T

    where gamma_i are the dephasing rates from the input list and P is the probability matrix.
    This implementation avoids creating diag(Σ gamma) explicitly by using broadcasting.

    For pure dephasing defined in the initial basis by rates gamma_a, the effective population-transfer
    rates in the new basis are

        K_new[i, j] = Σ_a gamma_a * |⟨i|a⟩|² * |⟨j|a⟩|²

    or, in matrix form,
        K_new = probabilities @ diag(kron_sum(dephasing_rates)) @ probabilities.transpose(-1, -2)

    Works for basis. Automatically satisfies population-transfer structure
    when the basis transformation is defined by probabilities.

    :param dephasing_list: List of pure dephasing rate tensors [..., K]
                  Each tensor represents dephasing rates in the initial basis
                  The list is combined using batched_sum_kron to handle Kronecker structure

    :param probabilities: Basis-transformation probabilities, defined as
                          probabilities[i, a] = |⟨i_new | a_old⟩|².
                          Shape [..., K, K]

                          Transformation probabilities from initial basis to system basis
                          The initial basis has shape [..., K]
                          The system basis has shape [..., K]

    :return: Effective population-transfer matrix in the new basis. Shape [..., K, K]

    Mathematical Formulation:
    -------------------------
    K_new = C @ diag(gamma) @ C^T
    K_new[i, j] = Σ_a C[i, a] * gamma[a] * C[j, a]
    """
    return transform_dephasing_to_population_transfer(
        batched_sum_kron_diagonal(dephasing_list),
        probabilities
    )


def transform_kronecker_operator(
        density_list: list[torch.Tensor],
        coeffs: torch.Tensor,
) -> torch.Tensor:
    """Transform Hilbert operators matrices from old subsystem operators to new basis.

    ρ_new = U @ (ρ₁ ⊗ ρ₂ ⊗ ... ⊗ ρₙ) @ U^†

    :param density_list: Density matrices for subsystems. Each shape: [..., k_i, k_i]
    :param coeffs: Transformation basis coeffs from kronecker basis U1 ⊗ U2 ⊗ ... ⊗ Uk to system basis Us
    The U1, ..., Uk have shape [..., ki], where K = ∏ᵢ k_i
    Us has shape [..., K]
    :return: Density matrix in new-system basis. Shape: [..., K, K]
    """
    return transform_operator_to_new_basis(
        batched_multi_kron(density_list), coeffs
    )


def transform_kronecker_superoperator(
        superoperator_list: list[torch.Tensor],
        coeffs: torch.Tensor,
        apply_secular_approximation: bool,
) -> torch.Tensor:
    """Compute composite superoperator in coupled basis from subsystem superoperators.

    Constructs the Kronecker-sum superoperator in local product basis, applies index
    permutation to reconcile vectorization orderings, then transforms to the coupled basis
    using Clebsch-Gordan coefficients.

    :param superoperator_list: Subsystem superoperators in Liouville space.
                               Each shape: [..., k_i², k_i²]
    :param coeffs: Transformation basis coeffs from kronecker basis U1 ⊗ U2 ⊗ ... ⊗ Uk to system basis Us
    The U1, ..., Uk have shape [..., ki], where K = ∏ᵢ k_i
    Us has shape [..., K]
    :param apply_secular_approximation: If True, enforces the secular approximation by masking
                                        non-secular terms after Kronecker construction but before
                                        basis transformation. This ensures extracted kinetic rates
                                        remain consistent with direct computation.
                                        If False, all terms are preserved; however, kinetic rates
                                        may differ due to non-secular couplings introduced by the
                                        basis transformation.
                                        Note: The mask retains population-population (ii,jj) and
                                        coherence-coherence (ij,ij) blocks. For degenerate energy
                                        levels, coherence-coherence transitions are also masked,
                                        although in literature they are usually considered secular
                                        since ΔE_ij = ΔE_kl.

    :return: Composite superoperator in coupled Liouville basis. Shape: [..., K², K²]

    Mathematical Formulation:
    -------------------------
        Step 1 — Compute Kronecker sum in local bases:
            L_local = Σᵢ (I_1 ⊗ ... ⊗ L_i ⊗ ... ⊗ I_n)

        Step 2 — Perform Permutation for vectorization ordering:
            L_kron = P · L_local · P^†
            where P reconciles tensor-product vs. Kronecker vectorization order

        Step 3 — Perform Basis transformation:
            L_coupled = T · L_kron · T^†
            where T = U ⊗ U* and U is the product→coupled unitary from coeffs

        This implements the full transformation:
            vec(ρ_coupled) evolves under L_coupled
            when vec(ρ₁) ⊗ ... ⊗ vec(ρₙ) evolves under local Lᵢ terms
    """
    dims = [int(round(superop.shape[-1] ** 0.5)) for superop in superoperator_list]
    R = reshape_superoperator_tensor_to_kronecker_basis(batched_sum_kron(superoperator_list), subsystem_dims=dims)
    T = batched_kron(coeffs, coeffs.conj())
    if apply_secular_approximation:
        return transform_superop_to_new_basis(apply_secular_mask(R), T)
    else:
        return transform_superop_to_new_basis(R, T)


def apply_secular_mask(superoperator: torch.Tensor) -> torch.Tensor:
    """
    Apply secular mask to superoperator. The mask retains population-population (ii,jj) and
    coherence-coherence (ij,ij) blocks. For degenerate energy
    levels, coherence-coherence transitions are also masked,
     although in literature they are usually considered secular
    since ΔE_ij = ΔE_kl.

    :param superoperator: relaxation superoperator
    :return: secularized superoperator with terms ii, jj and ij, ij
    """
    dim_liouville = superoperator.shape[-1]
    N = int(dim_liouville**0.5)
    mask = torch.eye(dim_liouville, dtype=torch.bool)
    pop_indices = torch.arange(0, dim_liouville, N + 1)
    mask[pop_indices[:, None], pop_indices[None, :]] = True
    return superoperator * mask.type_as(superoperator)


def reshape_vectorized_kronecker_to_tensor_product(
    vec_rho_kron: torch.Tensor,
    subsystem_dims: tp.List[int]
) -> torch.Tensor:
    """Reshape vectorized Kronecker product state to tensor product of vectorized subsystem states.
    This transformation uses row-major order for vectorization

    Converts the vectorized form of a composite density matrix expressed as a Kronecker product:
        vec(ρ₁ ⊗ ρ₂ ⊗ ... ⊗ ρₙ)
    into the tensor product of vectorized subsystem states:
        vec(ρ₁) ⊗ vec(ρ₂) ⊗ ... ⊗ vec(ρₙ)

    This transformation reorders indices to group row-column pairs per subsystem rather than
    grouping all rows followed by all columns.

    :param vec_rho_kron: Vectorized composite density matrix in Kronecker basis.
                         Shape: [..., D²] where D = ∏ᵢ dᵢ and dᵢ are subsystem dimensions
    :param subsystem_dims: List of subsystem Hilbert space dimensions [d₁, d₂, ..., dₙ]
    :return: Vectorized state in tensor product basis.
             Shape: [..., ∏ᵢ (dᵢ²)] = [..., d₁²·d₂²·...·dₙ²]

    Mathematical Formulation:
    -------------------------
    Given composite state ρ = ρ₁ ⊗ ρ₂ ⊗ ... ⊗ ρₙ with subsystem dimensions {dᵢ}:

    Reshaping reveals subsystem structure:
        ρ → ρ[i₁,...,iₙ; j₁,...,jₙ]  (row indices iₖ, column indices jₖ)

    Kronecker vectorization orders indices as:
        vec_kron[index(i₁,...,iₙ,j₁,...,jₙ)] = ρ[i₁,...,iₙ; j₁,...,jₙ]
        where index = (i₁·d₂·...·dₙ + i₂·d₃·...·dₙ + ... + iₙ)·D +
                     (j₁·d₂·...·dₙ + j₂·d₃·...·dₙ + ... + jₙ)

    Tensor product vectorization orders indices as:
        vec_tensor[index'(i₁,j₁,i₂,j₂,...,iₙ,jₙ)] = ρ[i₁,...,iₙ; j₁,...,jₙ]
        where index' = (i₁·d₁ + j₁)·(d₂²·...·dₙ²) + ... + (iₙ·dₙ + jₙ)

    The transformation is a permutation of vector elements corresponding to index mapping:
    index' = permute_indices(index)
    """
    batch_shape = vec_rho_kron.shape[:-1]
    n_systems = len(subsystem_dims)
    total_dim = 1
    for d in subsystem_dims:
        total_dim *= d

    x = vec_rho_kron.reshape(*batch_shape, total_dim, total_dim)
    x = x.reshape(*batch_shape, *subsystem_dims, *subsystem_dims)
    perm = list(range(len(batch_shape)))

    for i in range(n_systems):
        perm.append(len(batch_shape) + i)
        perm.append(len(batch_shape) + n_systems + i)

    x = x.permute(*perm)
    dims_squared = [d * d for d in subsystem_dims]
    x = x.reshape(*batch_shape, *dims_squared)
    return x.reshape(*batch_shape, -1)


def reshape_vectorized_tensor_product_to_kronecker(
        vec_tensor_prod: torch.Tensor,
        subsystem_dims: tp.List[int]
) -> torch.Tensor:
    """Reshape tensor product of vectorized subsystem states to vectorized Kronecker product state.
    This transformation uses row-major order for vectorization

    Converts the tensor product of vectorized subsystem density matrices:
        vec(ρ₁) ⊗ vec(ρ₂) ⊗ ... ⊗ vec(ρₙ)
    into the vectorized form of their Kronecker product:
        vec(ρ₁ ⊗ ρ₂ ⊗ ... ⊗ ρₙ)

    This is the inverse operation of reshape_vectorized_kronecker_to_tensor_product,
    reordering indices to group all row indices followed by all column indices.

    :param vec_tensor_prod: Vectorized state in tensor product basis.
                            Shape: [..., ∏ᵢ (dᵢ²)] = [..., d₁²·d₂²·...·dₙ²]
    :param subsystem_dims: List of subsystem Hilbert space dimensions [d₁, d₂, ..., dₙ]
    :return: Vectorized composite density matrix in Kronecker basis.
             Shape: [..., D²] where D = ∏ᵢ dᵢ

    Mathematical Formulation:
    -------------------------
    Given subsystem states {ρᵢ} with dimensions {dᵢ}, the tensor product vectorization is:
        vec_tensor = vec(ρ₁) ⊗ vec(ρ₂) ⊗ ... ⊗ vec(ρₙ)

    Each vec(ρᵢ) has elements vec(ρᵢ)[kᵢ] = ρᵢ[iᵢ, jᵢ] where kᵢ = iᵢ·dᵢ + jᵢ

    The composite state in tensor product basis has elements:
        vec_tensor[index'(i₁,j₁,i₂,j₂,...,iₙ,jₙ)] = ∏ᵢ ρᵢ[iᵢ, jᵢ]

    The Kronecker product state ρ = ⊗ᵢ ρᵢ has matrix elements:
        ρ[i₁,...,iₙ; j₁,...,jₙ] = ∏ᵢ ρᵢ[iᵢ, jᵢ]

    Its vectorization in Kronecker basis is:
        vec_kron[index(i₁,...,iₙ,j₁,...,jₙ)] = ρ[i₁,...,iₙ; j₁,...,jₙ]
        where index = (i₁·d₂·...·dₙ + ... + iₙ)·D + (j₁·d₂·...·dₙ + ... + jₙ)
        and D = ∏ᵢ dᵢ

    The transformation applies the inverse index permutation:
        index = permute_indices⁻¹(index')
    """
    batch_shape = vec_tensor_prod.shape[:-1]
    n_systems = len(subsystem_dims)
    total_dim = 1
    for d in subsystem_dims:
        total_dim *= d
    dims_squared = [d * d for d in subsystem_dims]
    x = vec_tensor_prod.reshape(*batch_shape, *dims_squared)

    reshape_dims = []
    for d in subsystem_dims:
        reshape_dims.extend([d, d])
    x = x.reshape(*batch_shape, *reshape_dims)

    perm = list(range(len(batch_shape)))
    for i in range(n_systems):
        perm.append(len(batch_shape) + 2 * i)
    for i in range(n_systems):
        perm.append(len(batch_shape) + 2 * i + 1)

    x = x.permute(*perm)
    x = x.reshape(*batch_shape, total_dim, total_dim)
    return x.reshape(*batch_shape, -1)


def reshape_superoperator_kronecker_to_tensor_basis(
        superop_kron: torch.Tensor,
        subsystem_dims: tp.List[int]
) -> torch.Tensor:
    """Convert superoperator representation from Kronecker basis to tensor product basis.
    This transformation uses row-major order for vectorization

    Transforms a superoperator L acting on vec(ρ₁ ⊗ ... ⊗ ρₙ) to act on
    vec(ρ₁) ⊗ ... ⊗ vec(ρₙ) by reordering both input and output indices.

    :param superop_kron: Superoperator in Kronecker basis representation.
                         Shape: [..., D², D²] where D = ∏ᵢ dᵢ
    :param subsystem_dims: List of subsystem Hilbert space dimensions [d₁, d₂, ..., dₙ]
    :return: Superoperator in tensor product basis representation.
             Shape: [..., ∏ᵢ (dᵢ²), ∏ᵢ (dᵢ²)]

    Mathematical Formulation:
    -------------------------
    A superoperator L maps vectorized density matrices:
        vec(ρ') = L · vec(ρ)

    In Kronecker basis (ρ = ⊗ᵢ ρᵢ):

    In tensor product basis:
        vec_tensor(ρ) = ⊗ᵢ vec(ρᵢ)

    The basis transformation is implemented via index permutation P:
        vec_tensor(ρ) = P · vec_kron(ρ)

    Therefore the superoperator transforms as:
        L_tensor = P · L_kron · P^†

    Where P implements the index mapping:
        P[index'(i₁,j₁,...,iₙ,jₙ), index(i₁,...,iₙ,j₁,...,jₙ)] = δ_{permuted}

    For Kronecker-sum superoperators (e.g., L = Σᵢ I⊗...⊗Lᵢ⊗...⊗I),
    """
    batch_shape = superop_kron.shape[:-2]
    n_systems = len(subsystem_dims)

    total_dim = 1
    for d in subsystem_dims:
        total_dim *= d
    x = superop_kron.reshape(
        *batch_shape,
        *subsystem_dims,
        *subsystem_dims,
        *subsystem_dims,
        *subsystem_dims
    )

    batch_ndim = len(batch_shape)
    perm = list(range(batch_ndim))
    for i in range(n_systems):
        perm.append(batch_ndim + i)
        perm.append(batch_ndim + n_systems + i)
    for i in range(n_systems):
        perm.append(batch_ndim + 2 * n_systems + i)
        perm.append(batch_ndim + 3 * n_systems + i)

    x = x.permute(*perm)
    dims_squared = [d * d for d in subsystem_dims]
    x = x.reshape(*batch_shape, *dims_squared, *dims_squared)
    total_sq_dim = 1
    for d_sq in dims_squared:
        total_sq_dim *= d_sq
    return x.reshape(*batch_shape, total_sq_dim, total_sq_dim)


def reshape_superoperator_tensor_to_kronecker_basis(
        superop_tensor: torch.Tensor,
        subsystem_dims: tp.List[int]
) -> torch.Tensor:
    """Convert superoperator representation from tensor product basis to Kronecker basis.
    This transformation uses row-major order for vectorization

    Transforms a superoperator L acting on vec(ρ₁) ⊗ ... ⊗ vec(ρₙ) to act on
    vec(ρ₁ ⊗ ... ⊗ ρₙ) by reordering both input and output indices.

    This is the inverse operation of reshape_superoperator_kronecker_to_tensor_basis,
    required when interfacing with libraries that expect Kronecker-basis representations.

    :param superop_tensor: Superoperator in tensor product basis representation.
                           Shape: [..., ∏ᵢ (dᵢ²), ∏ᵢ (dᵢ²)]
    :param subsystem_dims: List of subsystem Hilbert space dimensions [d₁, d₂, ..., dₙ]
    :return: Superoperator in Kronecker basis representation.
             Shape: [..., D², D²] where D = ∏ᵢ dᵢ

    Mathematical Formulation:
    -------------------------
    Given superoperator in tensor product basis L_tensor:
        vec_tensor(ρ') = L_tensor · vec_tensor(ρ)
        where vec_tensor(ρ) = ⊗ᵢ vec(ρᵢ)

    The Kronecker-basis representation satisfies:
        vec_kron(ρ') = L_kron · vec_kron(ρ)
        where vec_kron(ρ) = vec(⊗ᵢ ρᵢ)

    With basis transformation matrix P (from Kronecker to tensor basis):
        vec_tensor(ρ) = P · vec_kron(ρ)

    The inverse transformation is:
        L_kron = P^† · L_tensor · P

    Where P^† implements the inverse index permutation:
        P^†[index(i₁,...,iₙ,j₁,...,jₙ), index'(i₁,j₁,...,iₙ,jₙ)] = δ_{inverse_permuted}

    For subsystem-local operations with Kronecker-sum structure in tensor basis:
        L_tensor = Σᵢ I⊗...⊗Lᵢ⊗...⊗I  (acting on vec(ρᵢ) spaces)
    the Kronecker-basis representation becomes:
        L_kron = Σᵢ I⊗...⊗Lᵢ⊗...⊗I  (acting on full Hilbert space)
    with appropriate reshaping of Lᵢ to superoperator form.
    """
    batch_shape = superop_tensor.shape[:-2]
    n_systems = len(subsystem_dims)

    total_dim = 1
    dims_squared = []
    for d in subsystem_dims:
        total_dim *= d
        dims_squared.append(d * d)

    x = superop_tensor.reshape(*batch_shape, *dims_squared, *dims_squared)
    reshape_dims = []

    for d in subsystem_dims:
        reshape_dims.extend([d, d])
    x = x.reshape(*batch_shape, *reshape_dims, *reshape_dims)
    batch_ndim = len(batch_shape)
    perm = list(range(batch_ndim))
    for i in range(n_systems):
        perm.append(batch_ndim + 2 * i)
    for i in range(n_systems):
        perm.append(batch_ndim + 2 * i + 1)
    for i in range(n_systems):
        perm.append(batch_ndim + 2 * n_systems + 2 * i)
    for i in range(n_systems):
        perm.append(batch_ndim + 2 * n_systems + 2 * i + 1)

    x = x.permute(*perm)
    x = x.reshape(*batch_shape, total_dim, total_dim, total_dim, total_dim)
    return x.reshape(*batch_shape, total_dim * total_dim, total_dim * total_dim)


def reshape_superoperators_list_to_direct_sum_basis(
    superoperators: tp.List[torch.Tensor]
) -> torch.Tensor:
    """Construct superoperator for direct-sum composite system with independent relaxation.

    Given subsystem superoperators {Rᵢ} acting on vectorized density matrices
    vec(ρᵢ) (row-major), constructs the full superoperator R acting on the
    vectorized block-diagonal density matrix:

        ρ = ρ_1 ⊕ ρ_2 ⊕ … ⊕ ρ_n = diag(ρ_1, ρ_2, …, ρ_n)

    The resulting superoperator preserves the block-diagonal structure of ρ
    under time evolution.

    :param superoperators:
            List of subsystem superoperators.
            Each element Rᵢ has shape ``(..., N_i^2, N_i^2)`` where N_i is the Hilbert
            space dimension of subsystem i batch dimensions (...) must be broadcastable.

    :return:
        total_superop: Composite superoperator acting on vec(ρ_1 ⊕ … ⊕ ρ_n).
            Shape: ``(..., N^2, N^2)`` where N = Σᵢ Nᵢ is the total Hilbert space dimension.

    Mathematical Formulation:
    -------------------------
    For block-diagonal ρ with subsystem dimensions {Nᵢ} and offsets
    offsetᵢ = Σ_{k<i} N_k:

        ρ[p, q] = ρᵢ[p−offsetᵢ, q−offsetᵢ]   if p,q ∈ [offsetᵢ, offsetᵢ+Nᵢ)
                = 0                           otherwise

    Row-major vectorization maps matrix element ρ[p, q] to index:

        k = p·N + q   where N = Σᵢ Nᵢ

    Subsystem i occupies non-contiguous indices in vec(ρ):

        Kᵢ = { (offsetᵢ + a)·N + (offsetᵢ + b) | a,b ∈ [0, Nᵢ) }

    The composite superoperator embeds each Rᵢ into the subspace spanned by Kᵢ:

        R_total[Kᵢ, Kᵢ] = Rᵢ
        R_total[Kᵢ, Kⱼ] = 0   for i ≠ j

    """
    dims = [int(round(math.sqrt(R.shape[-1]))) for R in superoperators]
    batch_shape = superoperators[0].shape[:-2]

    N = sum(dims)
    offsets = [0] + torch.cumsum(torch.tensor(dims[:-1]), dim=0).tolist()

    device = superoperators[0].device
    dtype = superoperators[0].dtype
    total_superop = torch.zeros(batch_shape + (N * N, N * N), dtype=dtype, device=device)

    for R_i, N_i, off in zip(superoperators, dims, offsets):
        a = torch.arange(N_i, device=device)
        b = torch.arange(N_i, device=device)
        grid_a, grid_b = torch.meshgrid(a, b, indexing="ij")

        global_idx = (off + grid_a) * N + (off + grid_b)

        out_global = global_idx.flatten()
        in_global = out_global
        total_superop[..., out_global[:, None], in_global[None, :]] = R_i
    return total_superop


def reshape_direct_sum_basis_to_superoperators_list(
        superop_tensor: torch.Tensor,
        subsystem_dims: tp.List[int]
) -> tp.List[torch.Tensor]:
    """Extract subsystem superoperators from a direct-sum composite superoperator.

      Given a full superoperator R acting on the vectorized block-diagonal density
      matrix vec(ρ_1 ⊕ … ⊕ ρ_n), extracts the individual subsystem superoperators
      {Rᵢ} acting on vec(ρᵢ).

      This is the inverse operation of `reshape_superoperators_list_to_direct_sum_basis`.

      :param superop_tensor:
              Composite superoperator acting on vec(ρ_1 ⊕ … ⊕ ρ_n).
              Shape: ``(..., N^2, N^2)`` where N = Σᵢ Nᵢ is the total Hilbert space
              dimension. Batch dimensions (...) must be consistent across extractions.

      :param subsystem_dims:
              List of Hilbert space dimensions for each subsystem [N_1, N_2, ..., N_n].
              Must satisfy Σᵢ Nᵢ = N (where N^2 is the last dimension of superop_tensor).

      :return:
          superoperators: List of subsystem superoperators.
              Each element Rᵢ has shape ``(..., N_i^2, N_i^2)`` where N_i is the Hilbert
              space dimension of subsystem i.

      Mathematical Formulation:
      -------------------------
      For a block-diagonal ρ with subsystem dimensions {Nᵢ} and offsets
      offsetᵢ = Σ_{k<i} N_k:

          ρ[p, q] = ρᵢ[p−offsetᵢ, q−offsetᵢ]   if p,q ∈ [offsetᵢ, offsetᵢ+Nᵢ)
                  = 0                           otherwise

      Row-major vectorization maps matrix element ρ[p, q] to index:

          k = p·N + q   where N = Σᵢ Nᵢ

      Subsystem i occupies non-contiguous indices in vec(ρ):

          Kᵢ = { (offsetᵢ + a)·N + (offsetᵢ + b) | a,b ∈ [0, Nᵢ) }

      The subsystem superoperator Rᵢ is recovered by slicing the total superoperator
      at the indices corresponding to subspace Kᵢ:

          Rᵢ = R_total[Kᵢ, Kᵢ]
      """
    total_vec_dim = superop_tensor.shape[-1]
    N_total = int(round(math.sqrt(total_vec_dim)))

    if len(subsystem_dims) > 1:
        offsets = [0] + torch.cumsum(torch.tensor(subsystem_dims[:-1]), dim=0).tolist()
    else:
        offsets = [0]

    extracted_superoperators = []
    device = superop_tensor.device

    for N_i, off in zip(subsystem_dims, offsets):
        a = torch.arange(N_i, device=device)
        b = torch.arange(N_i, device=device)
        grid_a, grid_b = torch.meshgrid(a, b, indexing="ij")
        global_idx = (off + grid_a) * N_total + (off + grid_b)
        flat_idx = global_idx.flatten()
        R_i = superop_tensor[..., flat_idx[:, None], flat_idx[None, :]]

        extracted_superoperators.append(R_i)

    return extracted_superoperators


def extract_transition_matrix_from_superoperator(superoperator: torch.Tensor) -> torch.Tensor:
    """Extracts the population transfer matrix from a Superoperator.

    This function get the dynamics of the diagonal elements (populations)
    of the density matrix from the full superoperator evolution. It selects
    the rows and columns corresponding to the diagonal elements of the
    vectorized density matrix.

    :param superoperator: torch.Tensor
        Superoperator acting on vectorized density matrices.
        Shape: [..., N^2, N^2]

    :return: torch.Tensor
        Population transfer matrix describing population dynamics.
        Shape: [..., N, N]

    Mathematical Formulation:
    -------------------------
    Given a superoperator L acting on vec(ρ), the population transfer matrix T
    is extracted by selecting indices corresponding to diagonal elements ρ_ii.
    In standard vectorization, these indices are k = i * (N + 1).

        T_ij = L_{k_i, k_j}
    """
    dim_vec = superoperator.shape[-1]

    N = int(dim_vec ** 0.5)
    pop_indices = torch.arange(0, dim_vec, N + 1, device=superoperator.device)
    temp = superoperator[..., pop_indices, :]
    population_transfer_matrix = temp[..., :, pop_indices]
    return population_transfer_matrix.real


def extract_pure_loss_vector(population_transfer_matrix: torch.Tensor) -> torch.Tensor:
    """
    Extracts the pure loss vector from the population transfer matrix.

    The loss is defined as the negative column sum.
    If the system is closed (conservative), column sums are 0 -> Loss is 0.
    If the system is open, column sums are negative -> Loss is positive.

    :param population_transfer_matrix: Shape [..., N, N]
    :return: Loss vector, Shape [..., N] (Positive values indicate loss)
    """
    col_sums = population_transfer_matrix.sum(dim=-2)
    loss_vector = -col_sums
    return torch.clamp(loss_vector, min=0.0)


def set_diagonal_to_pure_loss(population_transfer_matrix: torch.Tensor) -> torch.Tensor:
    """
    Returns a new matrix where the diagonal contains only the pure loss terms.

    This function removes the population exchange contribution from the diagonal,
    leaving only the net loss/decay terms. For closed systems (no loss), the
    diagonal becomes zero.

    Mathematical Formulation:
    ------------------------
    In a rate matrix M where ṗ = M p:
    - Off-diagonal elements M_ij (i≠j) represent population transfer j→i
    - Diagonal elements M_jj contain both exchange and loss contributions
    - Column sums represent net loss: Σ_i M_ij = -Γ_loss,j

    By setting M_jj = Σ_i M_ij, the diagonal contains only pure loss terms.

    :param population_transfer_matrix: torch.Tensor
        Population transfer matrix.
        Shape: [..., N, N]

    :return: torch.Tensor
        Kinetic matrix with diagonal elements set to pure loss terms.
        Shape: [..., N, N]
    """
    result = population_transfer_matrix.clone()
    col_sums = result.sum(dim=-2)
    result.diagonal(offset=0, dim1=-2, dim2=-1)[:] = col_sums
    return result


def extract_dephasing_matrix_from_superoperator(superoperator: torch.Tensor) -> torch.Tensor:
    """Extract the coherence dephasing rate matrix from a Liouville superoperator.

    This function extracts the diagonal elements of the superoperator corresponding
    to coherence decay rates. The output is an N×N matrix where element [i, j]
    (i ≠ j) represents the dephasing rate for coherence ρ_ij. Diagonal elements
    (i == j) are set to zero as populations do not dephase.

    :param superoperator: torch.Tensor
        Liouville superoperator acting on vectorized density matrices.
        Shape: [..., N², N²] where N is the Hilbert space dimension.

    :return: torch.Tensor
        Dephasing rate matrix. Shape: [..., N, N]
        Element [i, j] contains the dephasing rate for coherence ρ_ij (i ≠ j).
        Diagonal elements are zero. The non diagonal elements are greater than zero

    Mathematical Formulation:
    -------------------------
    Given superoperator L acting on vec(ρ), the dephasing matrix Γ is extracted from
    the diagonal elements corresponding to coherences:

        Γ[i, j] = Re(L[(ij), (ij)])  for i ≠ j
        Γ[i, i] = 0                   for i == j

    where index (ij) = i·N + j in row-major vectorization.

    Notes:
    ------
    - Population transfer terms are NOT included (use extract_transition_matrix_from_superoperator instead)
    - Returns positive real part since superoperator diagonals are negative for decay
    The dephasing matrix extracts L[1,1] and L[2,2] as Γ[0,1] and Γ[1,0].
    """
    dim_vec = superoperator.shape[-1]
    N = int(dim_vec ** 0.5)
    superop_diag = torch.diagonal(superoperator, dim1=-2, dim2=-1)
    superop_diag = superop_diag.reshape(*superoperator.shape[:-2], N, N)
    dephasing_matrix = torch.real(superop_diag)
    dephasing_matrix.diagonal(offset=0, dim1=-2, dim2=-1).fill_(0)
    return -dephasing_matrix


def transform_dephasing_to_new_basis(
    init_dephasing: torch.Tensor,
    init_pop_transfer: torch.Tensor,
    probabilities: torch.Tensor,
    unitary: torch.Tensor
) -> torch.Tensor:
    """Transform complete dephasing rates including population-transfer contributions.

    Combines two contributions to eigenbasis dephasing:
    - Term A: Initial coherence dephasing transformed via |U|² weights
    - Term B: Population transitions contributing to coherence damping

    This is the complete transformation required when the initial superoperator
    contains both coherence dephasing AND population transfer terms.

    :param init_dephasing: torch.Tensor
        Initial basis dephasing rates. Shape: [..., N, N]
        Element [i, j] is dephasing rate for coherence ρ_ij (i ≠ j).
        Diagonal elements should be zero.

    :param init_pop_transfer: torch.Tensor
        Initial basis population transfer matrix. Shape: [..., N, N]
        Element [i, j] is rate for population transfer j → i.
        Diagonal elements shouldn't be zero

    :param probabilities: torch.Tensor
        Transformation probabilities |⟨new|old⟩|². Shape: [..., N, N]
        From get_transformation_probabilities().

    :param unitary: torch.Tensor
        Unitary transformation matrix U. Shape: [..., N, N]
        From basis_transformation(). U[new, old] = ⟨ψ_new|ψ_old⟩.

    :return: torch.Tensor
        Complete dephasing rates in new basis. Shape: [..., N, N]
        Element [u, v] is total dephasing rate for coherence ρ_uv.

    Mathematical Formulation:
    -------------------------
    Term A (Initial coherence dephasing):
        Γ_A[u, v] = Σ_{i,j} |U[u,i]|² · |U[v,j]|² · Γ_old[i, j]

    Term B (Population transfer contribution):
        Γ_B[u, v] = -Σ_{i,k} U[u,i]·U*[v,i] · K_old[i,k] · U*[u,k]·U[v,k]

    Total:
        Γ_new[u, v] = Γ_A[u, v] + Γ_B[u, v]

    Notes:
    ------
    - Diagonal elements are zeroed (populations don't dephase)
    - Consistent with full Liouville transformation: L_new = (U⊗U*) L_old (U⊗U*)†
    """
    N = init_dephasing.shape[-1]
    gamma_new = probabilities @ init_dephasing @ probabilities.transpose(-1, -2)

    # sum_{i,k} U_ui U*_vi L_pop[i,k] U*_uk U_vk
    term_B = -torch.einsum("...ui,...vi,...uk,...vk,...ik->...uv",
                          unitary, unitary.conj(),
                          unitary.conj(), unitary,
                          init_pop_transfer)
    gamma_new = gamma_new + term_B
    gamma_new = gamma_new.real

    gamma_new[..., torch.arange(N), torch.arange(N)] = 0.0
    return gamma_new


def construct_dephasing_matrix(dephasing: torch.Tensor) -> torch.Tensor:
    """Construct symmetric dephasing rate matrix from vector coefficients.

    Computes the matrix G where each element is the arithmetic mean of
    corresponding dephasing rates:
        G_ij = 1/2 * (gamma_i + gamma_j) for i != j

    This constructs a matrix often used in decoherence models where the
    interaction rate between state i and j is the average of their individual
    dephasing rates.

    :param dephasing:
        torch.Tensor: Tensor of shape [..., N] containing the coefficients gamma_i.
            The last dimension represents the subsystem index.
            Batch dimensions are preserved and broadcasted.
            The diagonal elements are set to be zero

    :return:
        torch.Tensor: Tensor of shape [..., N, N] representing the matrix G.
            - N = length of the dephasing vector
            - Batch dimensions match the input dephasing tensor

    Mathematical Formulation:
    -------------------------
    Given a vector gamma of length N:
        G = 0.5 * (gamma.unsqueeze(-1) + gamma.unsqueeze(-2))

    Notes:
    ------
    - Symmetry: The resulting matrix G is symmetric (G_ij = G_ji)
    - Efficiency: Uses broadcasting to avoid explicit loops, O(N^2) memory
    """
    diag_indexes = torch.arange(dephasing.shape[-1])
    gamma_col = dephasing.unsqueeze(-1)
    gamma_row = dephasing.unsqueeze(-2)
    out = 0.5 * (gamma_col + gamma_row)
    out[..., diag_indexes, diag_indexes] = 0
    return out


class Liouvilleator:
    @staticmethod
    def commutator_superop(operator: torch.Tensor) -> torch.Tensor:
        """Compute the superoperator form of the commutator with a given
        operator. Here we use row-stacking for density matrix

        For an operator A, this superoperator L satisfies:
            L[ρ] = [A, ρ] = Aρ - ρA
        when applied to a vectorized density matrix.

        :param operator : torch.Tensor
            Operator for the commutator. Shape: [..., d, d]

        :return: torch.Tensor
            Commutator superoperator. Shape: [..., d^2, d^2]

        Mathematical Formulation:
        -------------------------
        ``L = A ⊗ I - I ⊗ A^T``
        where ⊗ denotes the Kronecker product, and I is the identity matrix.

        Notes:
        ------
        - Vectorization follows row-major (C) order: element ρ_ij is at position i*d + j
        - Preserves Hermiticity of density matrices when used in Liouvillian evolution
        """
        d = operator.shape[-1]
        I = torch.eye(d, dtype=operator.dtype, device=operator.device)
        batch_dims = operator.shape[:-2]

        term1 = torch.einsum("...ij,kl->...ikjl", operator, I).reshape(*batch_dims, d * d, d * d)
        term2 = torch.einsum("kl,...ij->...kilj", I, operator.transpose(-1, -2)).reshape(*batch_dims, d * d, d * d)
        return term1 - term2

    @staticmethod
    def vec(rho: torch.Tensor) -> torch.Tensor:
        """Transform density matrix to Liouvillian space from Hilbert Space.

        Example:
            rho = torch.tensor([[0.5, 0.1],
                                [0.2, 0.5]])
            return tensor([0.5, 0.1, 0.2, 0.5])
        :param ρ: density matrix in matrix form. The shape is [..., N, N], where N is number of levels
        :return density matrix in vector form. The shape is [..., N**2], where N is number of levels:

        Mathematical Formulation:
        -------------------------
        vec_row(ρ)[i·N + j] = ρ[i, j]

        Notes:
        ------
        - Uses ROW-MAJOR ordering
        - Affects Kronecker structure: vec(UρU^†) = (U ⊗ U*) vec(ρ) for row-major
        """
        shapes = rho.shape
        return rho.reshape(*shapes[:-2], shapes[-1] * shapes[-1])

    @staticmethod
    def unvec(rho: torch.Tensor) -> torch.Tensor:
        """Transform density matrix to Hilbert space from Liouvillian Space.

        Example:
            vec_rho = torch.tensor([0.6, 0.1, 0.2, 0.4])
            return tensor([[0.6, 0.1],
                           [0.2, 0.4]])
        :param rho: density matrix in vector form. The shape is [..., N**2], where N is number of levels
        :return density matrix in matrix form. The shape is [..., N, N], where N is number of levels:

        Mathematical Formulation:
        -------------------------
        ρ[i, j] = vec_row(ρ)[i·N + j]

        Notes:
        ------
        - Inverse of vec() with row-major ordering
        """
        shapes = rho.shape
        dim = int(math.sqrt(shapes[-1]))
        return rho.reshape(*shapes[:-1], dim, dim)

    @staticmethod
    def hamiltonian_superop(hamiltonian: torch.Tensor) -> torch.Tensor:
        """Compute the Liouvillian superoperator for unitary evolution under a
        Hamiltonian.

        For a Hamiltonian H, this superoperator generates:
            ``dρ/dt = -i[H, ρ]``

        :param hamiltonian : torch.Tensor
            Hamiltonian operator. Shape: [..., d, d]

        :return: torch.Tensor
            Hamiltonian superoperator. Shape: [..., d^2, d^2]

        Mathematical Formulation:
        -------------------------
        L = -i (H ⊗ I - I ⊗ H^T)

        Notes:
        ------
        - This is the unitary part of the Lindblad master equation
        - Always generates trace-preserving and positivity-preserving dynamics
        """
        return -1j * Liouvilleator.commutator_superop(hamiltonian)

    @staticmethod
    def anticommutator_superop(operator: torch.Tensor) -> torch.Tensor:
        """Compute the superoperator form of the anticommutator with a given
        operator.

        For an operator A, this superoperator L satisfies:
            ``L[ρ] = {A, ρ} = Aρ + ρA``
        when applied to a vectorized density matrix.

        :param operator: torch.Tensor
            Operator for the anticommutator. Shape: [..., d, d]

        :return: torch.Tensor
            Anticommutator superoperator. Shape: [..., d^2, d^2]

        Mathematical Formulation:
        -------------------------
        L = A ⊗ I + I ⊗ A^T

        Notes:
        ------
        - Does not preserve trace by itself (requires combination with other terms)
        """
        d = operator.shape[-1]
        I = torch.eye(d, dtype=operator.dtype, device=operator.device)
        batch_dims = operator.shape[:-2]

        term1 = torch.einsum("...ij,kl->...ikjl", operator, I).reshape(*batch_dims, d * d, d * d)
        term2 = torch.einsum("kl,...ij->...kilj", I, operator.transpose(-1, -2)).reshape(*batch_dims, d * d, d * d)
        return term1 + term2

    @staticmethod
    def anticommutator_superop_diagonal(operator: torch.Tensor) -> torch.Tensor:
        """Compute the superoperator form of the anticommutator with a given
        DIAGONAL of a operator.

        It is similar anticommutator_superop but for the special case when the operator is diagonal.
        It takes only it's diagonal and returns also diagonal

        For an operator A, this superoperator L satisfies:
            ``L[ρ] = {A, ρ} = Aρ + ρA``
        when applied to a vectorized density matrix.

        :param operator: torch.Tensor
            Operator for the anticommutator. Shape: ``[..., d,]``

        :return: torch.Tensor
            Anticommutator superoperator. Shape: ``[..., d^2]``

        Mathematical Formulation:
        -------------------------
        L = A ⊗ I + I ⊗ A^T
        """
        d = operator.shape[-1]
        batch_dims = operator.shape[:-1]
        i_indices = torch.arange(d, device=operator.device)
        j_indices = torch.arange(d, device=operator.device)
        i_grid, j_grid = torch.meshgrid(i_indices, j_indices, indexing="ij")
        anticomm_diagonal = operator[..., i_grid] + operator[..., j_grid]
        return anticomm_diagonal.reshape(*batch_dims, d * d)

    @staticmethod
    def lindblad_dissipator_from_rates(w: torch.Tensor) -> torch.Tensor:
        """Construct Lindblad dissipator superoperator from off-diagonal rates (kinetic rates).

        Models the dissipator term in the Lindblad equation:
            ``D(ρ) = Σ_{i≠j} w_{ij} [L_{ij} ρ L_{ij}^† - (1/2){L_{ij}^† L_{ij}, ρ}]``
        where ``L_{ji} = √w_{ji} |j⟩⟨i|``, and "L_{ji}" defines the jump from i to j,
        w[i, j] represents transition rate from j → i (j is source, i is destination)

        This simplifies to:
            ``D(ρ) = Σ_{i≠j} w_{ji} [|j⟩⟨i| ρ |i⟩⟨j| - (1/2){|i⟩⟨i|, ρ}]``

        :param w : torch.Tensor
            Off-diagonal rate matrix. Shape: [..., n, n]
            Element [i,j] represents transition rate for i≠j

        :return: torch.Tensor
            Lindblad dissipator superoperator. Shape: [..., n², n²]

        Mathematical Formulation:
        -------------------------
        The dissipator consists of two parts:
        1. Jump term: ``Σ_{i≠j} w_{hi} |j⟩⟨i| ρ |i⟩⟨j|``
        2. Decay term: ``-(1/2) Σ_{i≠j} w_{ji} {|i⟩⟨i|, ρ}``

        Notes:
        ------
        - Only off-diagonal elements of w are used (i≠j)
        - The decay term represents decay
        - The jump term represents population transfer
        """
        n = w.shape[-1]
        batch_shape = w.shape[:-2]

        superop_jump = torch.zeros(*batch_shape, n * n, n * n, dtype=w.dtype, device=w.device)

        i_indices = torch.arange(n, device=w.device)
        j_indices = torch.arange(n, device=w.device)
        i_grid, j_grid = torch.meshgrid(i_indices, j_indices, indexing="ij")

        offdiag_mask = i_grid != j_grid
        i_offdiag = i_grid[offdiag_mask]  # destination states
        j_offdiag = j_grid[offdiag_mask]  # source states

        row_idx = j_offdiag * n + j_offdiag
        col_idx = i_offdiag * n + i_offdiag

        superop_jump[..., row_idx, col_idx] = w[..., j_offdiag, i_offdiag]

        w_offdiag = w * (~torch.eye(n, dtype=torch.bool, device=w.device))
        decay_rates = w_offdiag.sum(dim=-2)

        superop_decay = -0.5 * Liouvilleator.anticommutator_superop_diagonal(decay_rates)

        superop_total = superop_jump + torch.diag_embed(superop_decay)
        return superop_total

    @staticmethod
    def lindblad_dephasing_from_rates(gamma: torch.Tensor) -> torch.Tensor:
        """Construct Lindblad relaxation superoperator from 'dephasing' vector.

        It models dephasing

        Models the dephasing term in the Lindblad equation:
            ``D(ρ) = Σ_i γ_{i} [L_{i} ρ L_{i}^† - (1/2){L_{i}^† L_{i}, ρ}]``
        where ``L_{i} = √γ_{i} |i⟩⟨i|``

        This simplifies to:
            ``D(ρ) = Σ_i γ_{i} [|i⟩⟨i| ρ |i⟩⟨i| - (1/2){|i⟩⟨i|, ρ}]``

        :param gamma : torch.Tensor
            dephasing rate matrix. Shape: [..., n]
            Element [i] represents dephasing rate.
            For example, if γ is not zero only for i state, then the result will be ``- γ / 2 * ρ_ij for all j != i``
            In the general case:

            ``dρ_ij / dt = - (gamma_i + gamma_j) / 2 * ρ_ij for i != j``

        :return: torch.Tensor
            Lindblad dephasing superoperator. Shape: ``[..., n², n²]``

        Mathematical Formulation:
        -------------------------
        For i ≠ j:  dρ_{ij}/dt = -1/2(γ_i + γ_j) ρ_{ij}
        For i = j:  dρ_{ii}/dt = 0

        Superoperator is diagonal with entries:
            D[i·n + j, i·n + j] = -1/2(γ_i + γ_j)  for i ≠ j
            D[i·n + i, i·n + i] = 0
        """
        dephsing = -(gamma[..., :, None] + gamma[..., None, :]) / 2
        dephsing.diagonal(dim1=-2, dim2=-1).zero_()

        *batch, n, _ = dephsing.shape
        N = n * n
        pop_indices = torch.arange(n, device=dephsing.device) * (n + 1)

        is_coherence = torch.ones(N, dtype=torch.bool, device=dephsing.device)
        is_coherence[pop_indices] = False

        rate_vector = dephsing.reshape(*batch, N)
        rate_vector = rate_vector.clone()
        rate_vector[..., pop_indices] = 0
        return torch.diag_embed(rate_vector)

    @staticmethod
    def lindblad_dissipator_from_operator(operator: torch.Tensor) -> torch.Tensor:
        """Construct the full Lindblad superoperator from a single jump operator.

        Models the complete dissipator term in the Lindblad master equation:
            ``D(ρ) = L ρ L† - (1/2){L†L, ρ}``
        where L is the jump operator.

        This is the standard form for a single quantum jump channel in open
        quantum systems, describing both the quantum jump (L ρ L†) and the
        corresponding decay (-1/2{L†L, ρ}).

        :param operator: torch.Tensor
            Lindblad jump operator L. Shape: [..., d, d]
            Can be complex-valued for general quantum operations

        :return: torch.Tensor
            Lindblad dissipator superoperator. Shape: [..., d^2, d^2]

        Mathematical Formulation:
        -------------------------
        The dissipator consists of two parts:
        1. Jump term: ``L ρ L†`` → ``L ⊗ L*`` in superoperator form
        2. Decay term: ``-(1/2){L†L, ρ}`` → ``-(1/2)(L†L ⊗ I + I ⊗ (L†L)^T)``

        Combined: ``D = L ⊗ L* - (1/2)(L†L ⊗ I + I ⊗ (L†L)^T)``

        Notes:
        ------
        - Uses row-major vectorization convention (matches other Liouvilleator methods)
        - Preserves trace: Tr(D(ρ)) = 0 for any density matrix ρ
        - Preserves positivity when combined with Hamiltonian evolution
        - For multiple jump operators, sum the individual dissipators
        - The jump operator L can be any operator (not necessarily Hermitian)
        """
        d = operator.shape[-1]
        batch_dims = operator.shape[:-2]
        device = operator.device
        dtype = operator.dtype

        L_dagger_L = operator.conj().transpose(-1, -2) @ operator

        L_conj = operator.conj()
        jump_term = torch.einsum("...ij,...kl->...ikjl", operator, L_conj).reshape(*batch_dims, d * d, d * d)

        I = torch.eye(d, dtype=dtype, device=device)
        decay_op = L_dagger_L
        decay_term1 = torch.einsum("...ij,kl->...ikjl", decay_op, I).reshape(*batch_dims, d * d, d * d)
        decay_term2 = torch.einsum("kl,...ij->...kilj", I, decay_op.transpose(-1, -2)).reshape(*batch_dims, d * d,
                                                                                               d * d)
        decay_term = -0.5 * (decay_term1 + decay_term2)

        return jump_term + decay_term
