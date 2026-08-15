from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import torch
import numpy as np

import typing as tp


def signal_to_amplitude(y_vals):
    """Rotate the signal so that the mean imaginary part is zero, then compute
    amplitude."""

    img = np.imag(y_vals)
    real = np.real(y_vals)
    theta = np.arctan2(np.mean(img), np.mean(real))

    real_rot = real * np.cos(theta) + img * np.sin(theta)
    img_rot = -real * np.sin(theta) + img * np.cos(theta)

    return real_rot, img_rot


def create_baseline_mask(x_vals: np.ndarray,
                         baseline_areas: list[tuple[float, float]]):
    """Create a mask that includes only baseline regions.

    Parameters:
    -----------
    x_vals : array
        X coordinates
    baseline_areas : list of tuples
        List of (x_start, x_end) tuples defining baseline regions to INCLUDE

    Returns:
    --------
    mask : array of bool
        True for baseline regions, False for peak regions
    """
    if not baseline_areas:
        return np.ones(len(x_vals), dtype=bool)
    mask = np.zeros(len(x_vals), dtype=bool)
    for x_start, x_end in baseline_areas:
        baseline_region = (x_vals >= x_start) & (x_vals <= x_end)
        mask = mask | baseline_region
    return mask


def correct_baseline_polynomial(x_vals: np.ndarray, y_vals: np.ndarray, mask: np.ndarray, poly_order: int):
    """Remove baseline by fitting polynomial to regions excluding the peak."""
    coeffs = np.polyfit(x_vals[mask], y_vals[mask], poly_order)
    baseline = np.polyval(coeffs, x_vals)
    y_corrected = y_vals - baseline
    return y_corrected, baseline


def correct_baseline_saturation(y_vals: np.ndarray, sat_last_indexes: int):
    """Remove baseline by fitting polynomial to regions excluding the peak."""
    satur_value = y_vals[-sat_last_indexes:].mean()
    baseline = np.ones_like(y_vals) * satur_value
    y_corrected = y_vals - satur_value
    return y_corrected, baseline


def correct_baseline_als(y_vals: np.ndarray, mask: np.ndarray, lam=1e6, p=0.01, niter=10):
    """ALS baseline with masking capability.

    Parameters:
    -----------
    y : array
        Input signal
    mask : array of bool, optional
        True for points to INCLUDE in baseline fitting
        False for points to EXCLUDE (e.g., peak regions)
    lam : float
        Smoothness parameter
    p : float
        Asymmetry parameter
    niter : int
        Number of iterations
    """
    L = len(y_vals)
    D = diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))

    w = np.ones(L)

    if mask is not None:
        w[~mask] = 1e-10

    for i in range(niter):
        W = diags(w, 0, shape=(L, L))
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w * y_vals)
        w_new = p * (y_vals > z) + (1 - p) * (y_vals < z)

        if mask is not None:
            w_new[~mask] = 1e-10

        w = w_new

    return y_vals - z, z


def percentile_baseline(x_vals, y_vals, window_size=None, percentile=10,
                         proximity_threshold=0.15):
    """Detect baseline using local percentile analysis.

    Points close to local low percentile are likely baseline.
    """
    if window_size is None:
        window_size = len(y_vals) // 20

    baseline_mask = np.zeros(len(y_vals), dtype=bool)
    half_window = window_size // 2
    for i in range(len(y_vals)):
        start = max(0, i - half_window)
        end = min(len(y_vals), i + half_window + 1)

        local_percentile = np.percentile(y_vals[start:end], percentile)
        if y_vals[i] <= local_percentile * (1 + proximity_threshold):
            baseline_mask[i] = True

    return baseline_mask


def correct_baseline(x_vals: np.ndarray, y_vals: np.ndarray,
                     baseline_areas: tp.Optional[list[tuple[float, float]]] = None,
                     method="poly", poly_order=0, lam=2e7, p=0.05, niter=10, sat_last_indexes=10):
    """Correct baseline line in spectral data.

    :param x_vals: np.ndarray
           X-axis coordinates (e.g., fields or time points).
    :param y_vals:np.ndarray
           Raw spectral signal values (Y-axis).
    :param baseline_areas: list of (float, float) tuples, optional
           Regions to use for baseline estimation. Each tuple defines (start_x, end_x).
           If None, automatically detects baseline regions using percentile analysis.
    :param method: str, default="poly"
           Baseline correction method:
           - "poly": Polynomial fitting (order controlled by `poly_order`)
           - "als" : Asymmetric Least Squares (robust for complex baselines)
           - "saturation": Constant baseline from signal saturation region
    :param poly_order: int, default=0
           Polynomial order for "poly" method (0 = constant offset).
    :param lam: float, default=2e7
           Smoothness parameter for "als" method (higher = smoother baseline).
    :param p: float, default=0.05
           Asymmetry parameter for "als" method (0 < p < 1; smaller = more under-estimation).
    :param niter: int, default=10
           Max iterations for "als" convergence.
    :param sat_last_indexes: int, default=10
           Number of trailing points to average for saturation baseline ("saturation" method).
    :return:
       y_corrected : np.ndarray
           Baseline-corrected signal (y_vals - baseline).
       baseline : np.ndarray
           Estimated baseline values across all x_vals.
    """
    if method == "saturation":
        y_corrected, baseline = correct_baseline_saturation(y_vals, sat_last_indexes=sat_last_indexes)
    else:
        if baseline_areas is None:
            mask = percentile_baseline(x_vals, y_vals)
        else:
            mask = create_baseline_mask(x_vals, baseline_areas)
        if method == "poly":
            y_corrected, baseline = correct_baseline_polynomial(x_vals, y_vals, mask, poly_order=poly_order)
        elif method == "als":
            y_corrected, baseline = correct_baseline_als(y_vals, mask, lam=lam, p=p, niter=niter)
        else:
            raise ValueError("Wrong method. It must be 'poly' or 'als'")
    return y_corrected, baseline


def safe_copy(x: tp.Union[torch.Tensor, np.ndarray]) -> tp.Union[torch.Tensor, np.ndarray]:
    """Return an independent copy of a numpy array or torch tensor."""
    if isinstance(x, torch.Tensor):
        return x.clone()
    return x.copy()


def save_abs(y: tp.Union[torch.Tensor, np.ndarray])-> tp.Union[torch.Tensor, np.ndarray]:
    """Unified absolute value that works for both libraries."""
    return y.abs() if isinstance(y, torch.Tensor) else np.abs(y)


def normalize_spectrum(x: tp.Union[torch.Tensor, np.ndarray],
                       y: tp.Union[torch.Tensor, np.ndarray],
                       mode: str = "integral") -> tp.Union[torch.Tensor, np.ndarray]:
    """Normalize a 1D spectrum.

    :param x:  tp.Union[torch.Tensor, np.ndarray]
        X-axis coordinates (e.g., magnetic field values).
    :param y: tp.Union[torch.Tensor, np.ndarray]
        Spectrum amplitudes (Y-axis values).
    :param mode: str, default="integral"
        Normalization method:
        - "integral": Scale so area under |y| curve = 1
        - "max"     : Scale so maximum |y| value = 1
        - "none"    : Return unmodified copy

    :return:
    torch.Tensor
        Normalized spectrum (same shape as input y).
    Notes:
    ------
    Uses trapezoidal integration for "integral" mode.
    Returns copy of y unchanged if normalization factor is zero.
    """
    if mode is None or mode == "none":
        return safe_copy(y)
    step = float(x[1] - x[0]) if len(x) > 1 else 1.0
    if mode == "max":
        denom = float(save_abs(y).max())
        if denom == 0:
            return safe_copy(y)
        return y / denom
    if mode == "integral":
        denom = float(save_abs(y).sum() * step)
        if denom == 0:
            return safe_copy(y)
        return y / denom
    raise ValueError(f"Unknown norm mode: {mode}")


def normalize_spectrum2d(
        x1: tp.Union[torch.Tensor, np.ndarray],
        x2: tp.Union[torch.Tensor, np.ndarray],
        y: tp.Union[torch.Tensor, np.ndarray],
        mode: str = "integral") -> tp.Union[torch.Tensor, np.ndarray]:
    """Normalize a 2D spectrum.

    :param x1:  tp.Union[torch.Tensor, np.ndarray]
        X1-axis coordinates (e.g., magnetic field values).
    :param x2:  tp.Union[torch.Tensor, np.ndarray]
        X1-axis coordinates (e.g., magnetic field values).
    :param y: tp.Union[torch.Tensor, np.ndarray]
        Spectrum amplitudes (Y-axis values).
    :param mode: str, default="integral"
        Normalization method:
        - "integral": Scale so area under |y| curve = 1
        - "max"     : Scale so maximum |y| value = 1
        - "none"    : Return unmodified copy
    :return:
    torch.Tensor
        Normalized spectrum (same shape as input y).
    Notes:
    ------
    Uses rectangular integration (Riemann sum) for "integral" mode.
    Returns copy of y unchanged if normalization factor is zero.
    """
    if mode is None or mode == "none":
        return safe_copy(y)
    step_1 = float(x1[1] - x1[0]) if len(x1) > 1 else 1.0
    step_2 = float(x2[1] - x2[0]) if len(x2) > 1 else 1.0
    if mode == "max":
        denom = float(save_abs(y).max())
        if denom == 0:
            return safe_copy(y)
        return y / denom
    if mode == "integral":
        denom = float(save_abs(y).sum() * step_1 * step_2)
        if denom == 0:
            return safe_copy(y)
        return y / denom
    raise ValueError(f"Unknown norm mode: {mode}")


import numpy as np
import torch
from typing import Union

def fourier_transform_time(
    time: Union[np.ndarray, torch.Tensor],
    signal: Union[np.ndarray, torch.Tensor],
    axis: int = 0,
    return_angular: bool = True,
    shift: bool = False,
    positive_only: bool = False
):
    """
    Compute Fourier transform of a time-domain signal.

    :param time: 1D array of uniformly spaced time points.
                 Can be numpy.ndarray or torch.Tensor.
    :param signal: Signal to transform. Can be real or complex.
                   For 1D: shape (n_time,).
                   For 2D EPR data with time as first dimension: shape (n_time, n_field).
                   The time dimension is specified by `axis`.
                   Can be numpy.ndarray or torch.Tensor.
    :param axis: Axis along which the time dimension lies.
                 Default is 0.
    :param return_angular: If True, return angular frequencies in rad/s (ω = 2πf).
                           If False, return frequencies in Hz.
                           Default is True.
    :param shift: If True, apply `fftshift` to both frequencies and the transform,
                  so that zero frequency is at the centre.
                  Default is False.
    :param positive_only: If True, return only non‑negative frequencies
                          (0 … Nyquist). For real signals this halves the data;
                          for complex signals it discards negative frequencies.
                          Default is False.

    :return: A tuple (freqs, ft)
             - freqs : frequency array (1D). Shape is M = n_time (full)
                       or M = n_time//2+1 (positive only).
             - ft    : Fourier transform of `signal` along the time axis.
                       Complex‑valued. Shape is the same as `signal` but with
                       the time dimension replaced by the frequency dimension.
                       For your 2D data (n_time, n_field) with axis=0,
                       ft shape becomes (M, n_field).
    """
    is_torch = isinstance(signal, torch.Tensor)
    if is_torch:
        fft = torch.fft
        if not isinstance(time, torch.Tensor):
            time = torch.as_tensor(time, dtype=torch.float)
    else:
        fft = np.fft

    N = signal.shape[axis]
    dt = float((time[1] - time[0]).item() if is_torch else time[1] - time[0])

    if positive_only:
        freqs = fft.rfftfreq(N, d=dt)
    else:
        freqs = fft.fftfreq(N, d=dt)

    if return_angular:
        freqs = freqs * (2 * np.pi)

    if positive_only:
        if is_torch:
            if signal.ndim > 1:
                ft = fft.rfftn(signal, dim=(axis,))
            else:
                ft = fft.rfft(signal, dim=axis)
        else:
            if signal.ndim > 1:
                ft = fft.rfftn(signal, axes=(axis,))
            else:
                ft = fft.rfft(signal, axis=axis)
    else:
        if is_torch:
            if signal.ndim > 1:
                ft = fft.fftn(signal, dim=(axis,))
            else:
                ft = fft.fft(signal, dim=axis)
        else:
            if signal.ndim > 1:
                ft = fft.fftn(signal, axes=(axis,))
            else:
                ft = fft.fft(signal, axis=axis)

    if shift:
        if is_torch:
            freqs = fft.fftshift(freqs)
            ft = fft.fftshift(ft, dim=axis)
        else:
            freqs = fft.fftshift(freqs)
            ft = fft.fftshift(ft, axes=axis)

    return freqs, ft
