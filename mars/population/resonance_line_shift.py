import math
import typing as tp

import torch


class ResonanceShiftResult(tp.TypedDict):
    """Second-order resonance-shift diagnostic for one EPR transition."""
    lvl_down: int
    lvl_up: int
    bare_freq_Hz: float
    shift_Hz_2nd: float
    fwhm_Hz_2nd: float
    criterion: float
    raw_criterion: float
    mixing: float
    valid: bool


def resonance_shift_criterion(
    R: torch.Tensor,
    energies: torch.Tensor,
    lvl_down: tp.Union[tp.Sequence[int], torch.Tensor],
    lvl_up: tp.Union[tp.Sequence[int], torch.Tensor],
    mixing_limit: float = 0.1,
) -> tp.List[ResonanceShiftResult]:
    """
    Estimate resonance-frequency shifts and homogeneous linewidths to second order.

    The total relaxation superoperator must be represented in the Hamiltonian
    eigenbasis at the same magnetic field and crystallite orientation. MaRs uses
    row-major vectorization: ``rho[a, b] -> a * N + b``. No basis transformation
    is performed here. To transform a superoperator beforehand, use
    :func:`mars.transform.compute_liouville_basis_transformation` and
    :func:`mars.transform.transform_superop_to_new_basis`.

    The diagonal relaxation elements enter the unperturbed Liouvillian, whereas
    the off-diagonal elements are treated to second order. Energies are in Hz,
    and relaxation-superoperator elements are in s^-1.

    :param R: Total relaxation superoperator, of shape ``(N*N, N*N)``,
        in the Hamiltonian eigenbasis, with units s^-1.
    :param energies: Sorted Hamiltonian eigenenergies, of shape ``(N,)``,
        in Hz (not angular-frequency units).
    :param lvl_down: Lower-level indices for the selected EPR transitions.
        Shape ``(n_transitions,)``.
    :param lvl_up: Corresponding upper-level indices. Same shape as
        ``lvl_down``.
    :param mixing_limit: Maximum allowed ratio between an off-diagonal
        relaxation element and the complex unperturbed-mode separation.
        A diagnostic cutoff, not a universal physical threshold.
    :return: One result per transition, containing its level indices, bare
        frequency, second-order frequency shift (Hz), homogeneous Lorentzian
        FWHM (Hz), linewidth-normalized shift ``criterion``, ``mixing``, and
        perturbative-validity flag. If the expansion is invalid, ``criterion``
        is infinite while ``raw_criterion`` retains the formal estimate.
    :raises ValueError: If input shapes, energies, rates, or level ordering
        are invalid.
    :raises IndexError: If a transition index is out of range.

    Mathematical Formulation:
    -------------------------
    For the coherence ``(a, b)`` define ``i = a*N+b`` and
    ``lambda0[i] = R[i, i] - 2*pi*1j*(E[a] - E[b])``. The second-order correction
    to this mode is ``sum_{j!=i} R[i,j]*R[j,i]/(lambda0[i]-lambda0[j])``.
    The relaxation-induced shift is ``-Im(R[i,i] + correction)/(2*pi)`` Hz,
    and the homogeneous Lorentzian FWHM is
    ``-Re(R[i,i] + correction)/pi`` Hz.

    Notes:
    ------
    - The FWHM does not include phenomenological Gaussian broadening or
      residual Lorentzian broadening specified separately in the sample.
    - Near a degeneracy, perturbation theory may fail even when the returned
      linewidth-normalized shift is small. Check ``valid`` and ``mixing``.
    - The correction describes a Liouvillian-mode frequency; a maximum in a
      spectrum containing overlapping transitions may move differently.
    """
    R = torch.as_tensor(R).squeeze().to(dtype=torch.complex128)
    E = torch.as_tensor(
        energies, dtype=torch.float64, device=R.device
    ).squeeze()

    if E.ndim != 1:
        raise ValueError("energies must be 1D for one field and orientation")

    N = E.numel()
    if R.shape != (N * N, N * N):
        raise ValueError(
            f"R must have shape {(N * N, N * N)}, got {tuple(R.shape)}"
        )

    down = torch.as_tensor(lvl_down, dtype=torch.long, device=R.device).reshape(-1)
    up = torch.as_tensor(lvl_up, dtype=torch.long, device=R.device).reshape(-1)

    omega = (2.0 * math.pi * (E[:, None] - E[None, :])).reshape(-1)
    lam0 = torch.diagonal(R) - 1j * omega
    M = N * N
    outputs: tp.List[ResonanceShiftResult] = []

    for d, u in zip(down.tolist(), up.tolist()):
        a = u * N + d
        denom = lam0[a] - lam0
        mask = torch.ones(M, dtype=torch.bool, device=R.device)
        mask[a] = False

        coupled = torch.maximum(R[a, mask].abs(), R[mask, a].abs())
        abs_denom = denom[mask].abs()
        near_zero = abs_denom <= 1e-12
        singular = bool(((coupled > 1e-12) & near_zero).any())

        safe = torch.where(
            near_zero, torch.ones_like(abs_denom), abs_denom
        )
        mixing_terms = torch.where(near_zero,
            torch.where(
                coupled > 1e-12,
                torch.full_like(coupled, float("inf")),
                torch.zeros_like(coupled),
            ),
            coupled / safe,
        )
        mixing = mixing_terms.max().item() if M > 1 else 0.0

        if singular:
            correction = torch.tensor(
                complex(float("nan"), float("nan")),
                dtype=torch.complex128,
                device=R.device,
            )
        else:
            safe_complex = torch.where(
                near_zero, torch.ones_like(denom[mask]), denom[mask])
            correction = (
                R[a, mask] * R[mask, a] / safe_complex
            ).sum()

        delta = R[a, a] + correction
        shift_hz = (-delta.imag / (2.0 * math.pi)).item()
        fwhm_hz = (-delta.real / math.pi).item()
        valid = (
            not singular
            and math.isfinite(mixing)
            and mixing < mixing_limit
            and fwhm_hz > 0
        )
        if fwhm_hz > 0 and math.isfinite(shift_hz):
            raw_criterion = abs(shift_hz) / fwhm_hz
        elif fwhm_hz == 0 and shift_hz == 0:
            raw_criterion = 0.0
        else:
            raw_criterion = float("inf")

        outputs.append(
            {
                "lvl_down": d,
                "lvl_up": u,
                "bare_freq_Hz": (E[u] - E[d]).item(),
                "shift_Hz_2nd": shift_hz,
                "fwhm_Hz_2nd": fwhm_hz,
                "criterion": raw_criterion if valid else float("inf"),
                "raw_criterion": raw_criterion,
                "mixing": mixing,
                "valid": valid,
            }
        )

    return outputs
