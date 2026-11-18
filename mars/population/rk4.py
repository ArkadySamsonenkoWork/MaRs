import math

import torch


def _exp_and_phi1(M: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exp(M) and φ1(M) = Σ_k M^k / (k + 1)! = ∫_0^1 exp(M s) ds.

    Taylor series of φ1 for M / 2^s with ||M / 2^s||_1 ≤ 1/2, then s doublings
    φ1(2M) = φ1(M) (exp(M) + I) / 2, exp(2M) = exp(M)^2.
    """
    n = M.shape[-1]
    eye = torch.eye(n, dtype=M.dtype, device=M.device)
    norm = M.abs().sum(-2).amax().item()
    s = max(0, math.ceil(math.log2(norm / 0.5))) if norm > 0.5 else 0
    Ms = M / (2 ** s)

    order = 14
    phi = eye / math.factorial(order + 1)
    for k in range(order - 1, -1, -1):
        phi = Ms @ phi + eye / math.factorial(k + 1)
    expm = eye + Ms @ phi
    for _ in range(s):
        phi = 0.5 * phi @ (expm + eye)
        expm = expm @ expm
    return expm, phi


def _hermiticity_projection(M: torch.Tensor) -> torch.Tensor:
    """Project a Liouville-space map onto the maps that keep Hermiticity,
    M_(ab),(cd) -> (M_(ab),(cd) + M_(ba),(dc)^*) / 2 (row-major vectorization)."""
    n = M.shape[-1]
    d = math.isqrt(n)
    M4 = M.reshape(*M.shape[:-2], d, d, d, d)
    M4 = 0.5 * (M4 + M4.transpose(-4, -3).transpose(-2, -1).conj())
    return M4.reshape(M.shape)


def solve_matrix_ode_rk4(superop_static: torch.Tensor, superop_dynamic: torch.Tensor, n_steps: int):
    """Propagator over one period of dU/dφ = (A + B cos φ) U, U(0) = I, φ ∈ [0, 2π].

    A = superop_static and B = superop_dynamic are the static part and the amplitude of the
    oscillating part of the Liouvillian divided by the angular frequency ω, so that φ = ω t.

    Method
    ------
    The period is divided into n_steps steps h = 2π / n_steps with the nodes φ_k = k h.
    The evolution under A between the nodes is exact, and the oscillating part acts at the
    nodes as kicks K_k = exp(h cos φ_k B); the nodes φ_0 and φ_n carry half kicks:

        U(2π) = K_0^{1/2} E K_{n-1} E ... K_1 E K_0^{1/2},   E = exp(A h).

    The scheme is of second order in h, and it is exact for B = 0. Every factor is the
    exponential of A or of B, so U(2π) keeps the trace, the Hermiticity and the complete
    positivity whenever the Liouvillian does, for any n_steps. U(2π) and S are projected onto
    the maps that keep Hermiticity, M_(ab),(cd) = M_(ba),(dc)^*, so that the rounding errors of
    the products do not accumulate over many periods.

    B must be the superoperator of a Hamiltonian, B = -i[H_1, ·] / ω with Hermitian H_1.
    It is diagonalized once, B = W diag(λ) W^†, and the product is computed in this basis,
    where every kick is diagonal. One step costs one matrix multiplication.

    Between the nodes U(φ_k + s) = exp(A s) U(φ_k^+), so

        ∫_0^{2π} U(φ) sin φ dφ = Σ_k (e^{iφ_k} Φ_+ - e^{-iφ_k} Φ_-) U(φ_k^+) / (2i),
        Φ_± = ∫_0^h exp((A ± i) s) ds,

    where U(φ_k^+) is the propagator just after the kick at φ_k. The integral is returned in
    the form S used by EvolutionPropagatorSolver:

        ∫_0^{2π} U(φ) sin φ dφ = (S + (I - U(2π)) h / 12) h.

    :param superop_static: Static part of the Liouvillian divided by ω. Shape [..., N^2, N^2].
    :param superop_dynamic: Amplitude of the oscillating part of the Liouvillian divided by ω,
        -i[H_1, ·] / ω. Shape [..., N^2, N^2]. The batch shape is broadcast with superop_static.
    :param n_steps: Number of steps per period.
    :return: Tuple (U, S):
        - U: propagator U(2π). Shape [..., N^2, N^2].
        - S: integral term defined above. Shape [..., N^2, N^2].
    """
    dtype = torch.promote_types(torch.promote_types(superop_static.dtype, superop_dynamic.dtype), torch.complex64)
    device = superop_static.device
    n = superop_static.shape[-1]
    A = superop_static.to(dtype)
    B = superop_dynamic.to(dtype)
    eye = torch.eye(n, dtype=dtype, device=device)

    real_dtype = torch.empty((), dtype=dtype).real.dtype
    h = 2 * math.pi / n_steps
    phi = torch.arange(n_steps, device=device, dtype=real_dtype) * h
    theta = h * torch.cos(phi)
    theta[0] = 0.5 * h
    e_plus = torch.exp(1j * phi).to(dtype)
    e_minus = e_plus.conj()

    exp_plus, phi1_plus = _exp_and_phi1(h * (A + 1j * eye))
    _, phi1_minus = _exp_and_phi1(h * (A - 1j * eye))
    E = exp_plus * complex(math.cos(h), -math.sin(h))

    X = 1j * B
    x, W = torch.linalg.eigh(0.5 * (X + X.conj().transpose(-2, -1)))
    Wh = W.conj().transpose(-2, -1)
    kicks = torch.exp(theta.to(dtype)[:, None] * (-1j * x).to(dtype).unsqueeze(-2))
    kicks = kicks.unsqueeze(-1)

    E = Wh @ E @ W
    phi_plus = Wh @ (h * phi1_plus) @ W
    phi_minus = Wh @ (h * phi1_minus) @ W

    batch_shape = torch.broadcast_shapes(E.shape[:-2], kicks.shape[:-3])
    Y = (kicks[..., 0, :, :] * eye).expand(*batch_shape, n, n).contiguous()
    sum_plus = e_plus[0] * Y
    sum_minus = e_minus[0] * Y
    for k in range(1, n_steps):
        Y = kicks[..., k, :, :] * (E @ Y)
        sum_plus += e_plus[k] * Y
        sum_minus += e_minus[k] * Y
    U = kicks[..., 0, :, :] * (E @ Y)

    integral = (phi_plus @ sum_plus - phi_minus @ sum_minus) / 2j
    U = W @ U @ Wh
    integral = W @ integral @ Wh
    U = _hermiticity_projection(U)
    S = _hermiticity_projection(integral / h - (eye - U) * (h / 12))
    return U, S
