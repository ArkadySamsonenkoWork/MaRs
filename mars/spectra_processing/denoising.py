import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import medfilt2d
import pywt
from typing import Literal


def filter_noise_2d(
    fields: np.ndarray,
    time: np.ndarray,
    result: np.ndarray,
    method: Literal["svd", "gaussian", "wavelet", "median", "combined"] = "svd",
    svd_rank: int = 10,
    sigma_t: float = 4.0,
    sigma_b: float = 2.0,
    wavelet: str = "db4",
    wavelet_level: int = 4,
    wavelet_threshold: float | None = None,
    median_kernel: int = 3,
) -> np.ndarray:
    """
    Denoise 2D spectral data I(B, t).

    :param fields: 1D array of magnetic field values in Tesla, shape (n_fields,).
    :param time: 1D array of time values in seconds, shape (n_time,).
    :param result: 2D array of spectral intensities, shape (n_fields, n_time).
        The magnetic field axis is first, the time axis is second.
    :param method: Denoising algorithm to apply. One of:
        - ``"svd"``      low-rank SVD reconstruction (best for correlated noise);
        - ``"gaussian"`` 2-D Gaussian smoothing (simplest baseline);
        - ``"wavelet"``  per-B-trace wavelet soft-thresholding (preserves sharp edges in t);
        - ``"median"``   2-D median filter (kills spikes / impulse noise);
        - ``"combined"`` SVD followed by mild Gaussian (recommended starting point).
    :param svd_rank: Number of dominant singular vectors to retain.
        Ignored by ``"gaussian"``, ``"wavelet"``, and ``"median"``.
        Plot singular values and keep components above the "elbow".
    :param sigma_t: Gaussian smoothing width along the time axis in samples.
        Ignored by every method except ``"gaussian"`` and ``"combined"``.
    :param sigma_b: Gaussian smoothing width along the field axis in samples.
        Ignored by every method except ``"gaussian"`` and ``"combined"``.
    :param wavelet: PyWavelets wavelet name, e.g. ``"db4"``, ``"sym6"``, ``"coif3"``.
        Used only by ``"wavelet"``.
    :param wavelet_level: Wavelet decomposition depth.
        Used only by ``"wavelet"``.
    :param wavelet_threshold: Hard override for the soft-threshold value.
        When ``None`` the universal threshold σ√(2 ln N) is estimated
        from the finest detail coefficients via the MAD estimator.
        Used only by ``"wavelet"``.
    :param median_kernel: Side length of the 2-D median-filter kernel in samples.
        Must be a positive odd integer; even values are silently incremented by 1.
        Used only by ``"median"``.

    :returns: Denoised spectral data with the same shape (n_fields, n_time)
        and dtype ``float64`` as *result*.

    :raises ValueError: If *result* shape does not match ``(len(fields), len(time))``.
    :raises ValueError: If *method* is not one of the supported strings.
    """
    B = np.asarray(fields, dtype=float)
    t = np.asarray(time,   dtype=float)
    I = np.asarray(result, dtype=float)

    if I.shape != (len(B), len(t)):
        raise ValueError(
            f"result shape {I.shape} does not match "
            f"(n_fields={len(B)}, n_time={len(t)})"
        )

    match method.lower():
        case "svd":
            return _svd_denoise_2d(I, svd_rank)
        case "gaussian":
            return gaussian_filter(I, sigma=(sigma_b, sigma_t))
        case "wavelet":
            return _wavelet_denoise_2d(I, wavelet, wavelet_level, wavelet_threshold)
        case "median":
            k = int(median_kernel)
            if k % 2 == 0:
                k += 1
            return medfilt2d(I, kernel_size=k)
        case "combined":
            return gaussian_filter(
                _svd_denoise_2d(I, svd_rank),
                sigma=(sigma_b * 0.5, sigma_t * 0.5),
            )
        case _:
            raise ValueError(
                f"Unknown method '{method}'. "
                "Choose from: svd, gaussian, wavelet, median, combined."
            )


def filter_noise_1d(
    axis: np.ndarray,
    signal: np.ndarray,
    method: Literal["gaussian", "wavelet", "median", "savgol"] = "wavelet",
    sigma: float = 2.0,
    wavelet: str = "db4",
    wavelet_level: int = 4,
    wavelet_threshold: float | None = None,
    median_kernel: int = 5,
    savgol_window: int = 11,
    savgol_order: int = 3,
) -> np.ndarray:
    """
    Denoise a 1-D spectral slice I(t) or I(B).

    The function is axis-agnostic: *axis* may be a time array or a field array;
    only its length is used for validation.

    :param axis: 1D coordinate array (time in seconds **or** field in Tesla),
        shape (n,).
    :param signal: 1D array of spectral intensities to denoise, shape (n,).
    :param method: Denoising algorithm to apply. One of:
        - ``"gaussian"`` Gaussian smoothing — fast, isotropic blurring;
        - ``"wavelet"``  soft-thresholding in wavelet domain — preserves peaks;
        - ``"median"``   median filter — robust against spikes / outliers;
        - ``"savgol"``   Savitzky–Golay polynomial smoothing — preserves peak
          height and position better than Gaussian.
    :param sigma: Gaussian smoothing width in samples.
        Used only by ``"gaussian"``.
    :param wavelet: PyWavelets wavelet name, e.g. ``"db4"``, ``"sym6"``.
        Used only by ``"wavelet"``.
    :param wavelet_level: Wavelet decomposition depth.
        Used only by ``"wavelet"``.
    :param wavelet_threshold: Hard override for the soft-threshold value.
        When ``None`` the universal threshold σ√(2 ln N) is estimated
        from the finest detail coefficients via the MAD estimator.
        Used only by ``"wavelet"``.
    :param median_kernel: Length of the 1-D median-filter window in samples.
        Must be a positive odd integer; even values are silently incremented by 1.
        Used only by ``"median"``.
    :param savgol_window: Length of the Savitzky–Golay filter window in samples.
        Must be odd and strictly greater than *savgol_order*.
        Used only by ``"savgol"``.
    :param savgol_order: Polynomial order for the Savitzky–Golay fit.
        Must be less than *savgol_window*.
        Used only by ``"savgol"``.

    :returns: Denoised 1-D signal with the same shape (n,) and dtype ``float64``
        as *signal*.

    :raises ValueError: If *signal* length does not match *axis* length.
    :raises ValueError: If *method* is not one of the supported strings.
    """
    from scipy.signal import medfilt, savgol_filter
    from scipy.ndimage import gaussian_filter1d

    x = np.asarray(axis,   dtype=float)
    s = np.asarray(signal, dtype=float)

    if s.shape != x.shape:
        raise ValueError(
            f"signal shape {s.shape} does not match axis shape {x.shape}."
        )

    match method.lower():
        case 'gaussian':
            return gaussian_filter1d(s, sigma=sigma)
        case 'wavelet':
            return _wavelet_denoise_1d(s, wavelet, wavelet_level, wavelet_threshold)
        case 'median':
            k = int(median_kernel)
            if k % 2 == 0:
                k += 1
            return medfilt(s, kernel_size=k)
        case 'savgol':
            w = int(savgol_window)
            if w % 2 == 0:
                w += 1
            return savgol_filter(s, window_length=w, polyorder=savgol_order)
        case _:
            raise ValueError(
                f"Unknown method '{method}'. "
                "Choose from: gaussian, wavelet, median, savgol."
            )


def _svd_denoise_2d(I: np.ndarray, rank: int) -> np.ndarray:
    """
    Reconstruct *I* from its top-*rank* singular triplets.

    :param I: 2D input array, shape (m, n).
    :param rank: Number of singular vectors to retain.
    :returns: Low-rank approximation of *I*, same shape.
    """
    U, s, Vt = np.linalg.svd(I, full_matrices=False)
    rank = min(rank, len(s))
    return (U[:, :rank] * s[:rank]) @ Vt[:rank, :]


def _wavelet_denoise_2d(
    I: np.ndarray,
    wavelet: str,
    level: int,
    threshold: float | None,
) -> np.ndarray:
    """
    Denoise each row (B-trace) of *I* independently via wavelet soft-thresholding.

    :param I: 2D array, shape (n_fields, n_time). Rows are denoised along the
        time axis.
    :param wavelet: PyWavelets wavelet name.
    :param level: Decomposition depth.
    :param threshold: Fixed threshold value, or ``None`` to use the
        MAD-based universal threshold per row.
    :returns: Denoised array, same shape as *I*.
    """
    I_out = np.empty_like(I)
    for i, row in enumerate(I):
        I_out[i] = _wavelet_denoise_1d(row, wavelet, level, threshold)
    return I_out


def _wavelet_denoise_1d(
    signal: np.ndarray,
    wavelet: str,
    level: int,
    threshold: float | None,
) -> np.ndarray:
    """
    Denoise a single 1-D signal via wavelet soft-thresholding.

    :param signal: 1D input array, shape (n,).
    :param wavelet: PyWavelets wavelet name.
    :param level: Decomposition depth.
    :param threshold: Fixed threshold value, or ``None`` to estimate the
        universal threshold σ√(2 ln N) from the finest detail band using
        the MAD noise estimator σ = median(|d|) / 0.6745.
    :returns: Denoised signal, shape (n,), dtype ``float64``.
    """
    coeffs = pywt.wavedec(signal, wavelet, level=level)

    if threshold is None:
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        thr = sigma * np.sqrt(2.0 * np.log(len(signal)))
    else:
        thr = float(threshold)

    coeffs_thresh = [coeffs[0]] + [
        pywt.threshold(c, thr, mode="soft") for c in coeffs[1:]
    ]
    return pywt.waverec(coeffs_thresh, wavelet)[:len(signal)]   
