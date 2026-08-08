from __future__ import annotations

from .fitter import BaseSpectrumFitter, FitResult, SpectrumFitter, Spectrum2DFitter

import warnings
import typing as tp
from dataclasses import dataclass
import tqdm

import numpy as np
import torch

import emcee
from scipy.optimize import minimize as scipy_minimize
import scipy
from scipy.stats import chi2 as scipy_chi2
import numdifftools as nd

from ..spectra_processing import normalize_spectrum, percentile_baseline, normalize_spectrum2d


@dataclass
class ParameterCI:
    """Confidence interval for a single parameter.

    :param lower: Lower bound of the confidence interval.
    :param upper: Upper bound of the confidence interval.
    :param best: Best-fit (maximum-likelihood) value of the parameter.
    """
    lower: float
    upper: float
    best: float

    @property
    def half_width(self) -> float:
        return 0.5 * (self.upper - self.lower)

    def __repr__(self) -> str:
        return f"{self.best:.6g}  [{self.lower:.6g}, {self.upper:.6g}]  (±{self.half_width:.6g})"


@dataclass
class UncertaintyResult:
    """Full uncertainty result for a set of parameters.

    :param intervals: Mapping from parameter name to its :class:`ParameterCI`.
    :param method: Name of the method used to compute the intervals
        (``"hessian"`` | ``"profile"`` | ``"mcmc"`` | ``"trials"``).
    :param confidence_level: Confidence level used, e.g. ``0.95`` for 95 %.
    :param extra: Method-specific extras (Hessian matrix, profile curves,
        MCMC chain, etc.).
    """
    intervals: tp.Dict[str, ParameterCI]
    method: str
    confidence_level: tp.Optional[float]
    extra: tp.Dict

    def __repr__(self) -> str:
        if self.confidence_level is not None:
            lines = [
                f"UncertaintyResult  method={self.method!r}  "
                f"confidence_level={self.confidence_level:.0%}",
                "-" * 60,
            ]
        else:
            lines = [
                f"UncertaintyResult  method={self.method!r}  "
                f"confidence_level: undefined",
                "-" * 60,
            ]
        max_len = max(len(k) for k in self.intervals) if self.intervals else 0
        for name, ci in self.intervals.items():
            lines.append(f"  {name:<{max_len}}  {ci}")
        return "\n".join(lines)

    def to_dict(self) -> tp.Dict[str, tp.Dict[str, float]]:
        """JSON dict."""
        return {
            name: {"lower": ci.lower, "upper": ci.upper, "best": ci.best}
            for name, ci in self.intervals.items()
        }


class ChiSquareComputer:
    """    This class computes the (reduced) chi‑squared statistic between experimental
    spectra and a pre‑computed best‑fit model.

    Noise estimation
    Instead of using the loss‑derived variance :math:`\\hat\\sigma^2 = \\text{loss}^*/(N-P)`
    (which would force the reduced χ² to exactly 1.0 at the optimum, making it
    useless for diagnostics), σ is estimated **independently from the data**:

        - The experimental spectrum is assumed to be baseline‑free (baseline already
          subtracted).
        - A signal‑free baseline mask is created via a local‑percentile algorithm.
        - The noise level σ is the standard deviation of the **residuals**
          (``experimental − model``) restricted to those baseline regions.

    This estimate reflects the true measurement noise only when the loss is
    proportional to SSE or MSE (the default in MaRS).  For other loss
    functions the estimated σ is not the true noise level, but the reduced χ²
    remains *qualitatively* interpretable (χ²/dof ≈ 1 indicates a good fit,
    < 1 overfitting, > 1 poor model).

    Workflow summary:
    1. Baseline detection (optional) – A signal‑free mask is generated from the
       *experimental* spectrum using a local percentile method.  This mask isolates
       regions that should contain only noise.
    2. Noise estimation – The standard deviation of the residuals
       (``experimental − model``) is computed only inside the baseline mask,
       yielding a  estimate of the noise level :math:`\\sigma`.
    3. Chi‑squared calculation – The statistic
       :math:`\\chi^2 = \\sum_i \\big((y_i^{\\text{exp}} - y_i^{\\text{model}})/\\sigma\\big)^2`
       is formed.  If reduced chi‑squared is requested, the result is divided by
       the number of degrees of freedom (``n_data − n_varying``).


    Notes
    -----
    - The baseline mask is computed on the *experimental* spectrum (not on the
      residual) because the model may already contain a fitted baseline, making
      the residual unsuitable for distinguishing signal‑free regions.
    - If fewer than 3 points fall inside the baseline mask, the noise estimate
      falls back to ``NaN``.  The caller should ensure a reliable mask is supplied.
    - For multisample fitters, a chi‑squared value (and optionally a sigma) is
      computed per individual spectrum.  The return type adapts accordingly.
    """

    def __init__(
        self,
        fitter,
        best_spectrum: tp.Union[torch.Tensor, tp.List[torch.Tensor]],
        n_varying: int,
    ):
        """
        :param fitter: A fitted SpectrumFitter or Spectrum2DFitter instance.
        :param best_spectrum: Pre-computed normalized model spectrum(s) from FitResult.
        :param n_varying: Number of varying parameters (for dof calculation).
        """
        self.fitter = fitter
        self.best_spectrum = best_spectrum
        self.n_varying = n_varying

    def _detect_baseline_1d(
        self,
        x: np.ndarray,
        y: np.ndarray,
        percentile: float = 10,
        proximity_threshold: float = 0.15,
        window_size: tp.Optional[int] = None,
    ) -> np.ndarray:
        """Detect baseline mask for 1D data using local percentile analysis."""
        return percentile_baseline(
            x_vals=x,
            y_vals=y,
            window_size=window_size,
            percentile=percentile,
            proximity_threshold=proximity_threshold,
        )

    def _detect_baseline_2d(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        y: np.ndarray,
        percentile: float = 10,
        proximity_threshold: float = 0.15,
        window_size: tp.Optional[int] = None,
    ) -> np.ndarray:
        """Detect baseline mask for 2D data by flattening temporarily."""
        y_flat = y.ravel()
        x_flat = np.arange(len(x1))
        mask_flat = percentile_baseline(
            x_vals=x_flat,
            y_vals=y_flat,
            window_size=window_size,
            percentile=percentile,
            proximity_threshold=proximity_threshold,
        )
        return mask_flat.reshape(y.shape)

    def _estimate_sigma(
        self,
        residuals: np.ndarray,
        baseline_mask: tp.Optional[np.ndarray] = None,
        min_sigma: float = 1e-12,
    ) -> float:
        """Estimate noise level σ from residuals in baseline regions.

        Computes the standard deviation of residuals restricted to points
        identified as baseline by the mask. If fewer than 3 baseline points
        are available and ``fallback_to_global=True``, falls back to the
        standard deviation over all residuals.

        :param residuals: 1D or 2D array of residuals ``y_exp - y_model``
            in the normalised space.
        :param baseline_mask: Boolean array of the same shape as ``residuals``.
            ``True`` marks points considered to be baseline (signal-free).
            If ``None`` or fewer than 3 points are ``True``, falls back to
            global std if ``fallback_to_global=True``.
        :param min_sigma: Minimum returned σ to avoid division by zero.
            Default ``1e-12``.
        :return: Estimated noise level σ > 0.
        """
        if baseline_mask is not None and baseline_mask.sum() >= 3:
            sigma = float(np.std(residuals[baseline_mask], ddof=self.n_varying))
        else:
            sigma = float('nan')
        return max(sigma, min_sigma)

    def _compute_chi2_core(
        self,
        y_exp: np.ndarray,
        y_model: np.ndarray,
        baseline_mask: np.ndarray,
        return_reduced: bool = True,
        return_sigma: bool = False,
        **baseline_kwargs,
    ) -> tp.Union[float, tp.Tuple[float, float]]:
        """Core chi-squared computation for a single normalised dataset.

        Computes:

            χ^2 = Σ ((y_exp - y_model) / σ)^2

        where σ is estimated from ``baseline_mask`` via :meth:`_estimate_sigma`.
        If ``return_reduced=True``, divides by ``dof = N - P``.

        :param y_exp: Normalised experimental spectrum, shape ``(N,)`` or ``(N, M)``.
        :param y_model: Normalised model spectrum, same shape as ``y_exp``.
        :param baseline_mask: Boolean mask identifying baseline points,
            same shape as ``y_exp``.
        :param return_reduced: If ``True``, return χ^2/dof (reduced chi-squared).
            χ^2/dof ≈ 1 indicates a good fit, < 1 overfitting, > 1 poor model.
            Default ``True``.
        :param return_sigma: If ``True``, also return the estimated σ.
            Default ``False``.
        :return: χ² or χ^2/dof as float, or tuple ``(chi2, sigma)`` if
            ``return_sigma=True``.
        """
        residuals = y_exp - y_model

        sigma = self._estimate_sigma(residuals , baseline_mask)
        chi2 = float(np.sum((residuals / sigma) ** 2))

        if return_reduced:
            n_data = residuals.size
            dof = max(n_data - self.n_varying, 1)
            chi2 = chi2 / dof

        return (chi2, sigma) if return_sigma else chi2

    def _compute_chi2_1d(
        self,
        x: np.ndarray,
        y_exp: np.ndarray,
        y_model: np.ndarray,
        baseline_mask: tp.Optional[np.ndarray] = None,
        return_reduced: bool = True,
        return_sigma: bool = False,
        **baseline_kwargs,
    ) -> tp.Union[float, tp.Tuple[float, float]]:
        """Compute chi-square for normalized 1D spectral data."""
        if baseline_mask is None:
            baseline_mask = self._detect_baseline_1d(
                x, y_exp, **baseline_kwargs
            )
        return self._compute_chi2_core(
            y_exp, y_model, baseline_mask, return_reduced, return_sigma
        )

    def _compute_chi2_2d(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        y_exp: np.ndarray,
        y_model: np.ndarray,
        baseline_mask: tp.Optional[np.ndarray] = None,
        return_reduced: bool = True,
        return_sigma: bool = False,
        **baseline_kwargs,
    ) -> tp.Union[float, tp.Tuple[float, float]]:
        """Compute chi-square for normalized 2D spectral data."""
        if baseline_mask is None:
            baseline_mask = self._detect_baseline_2d(
                x1, x2, y_exp, **baseline_kwargs
            )
        return self._compute_chi2_core(
            y_exp, y_model, baseline_mask, return_reduced, return_sigma
        )

    def compute(
        self,
        baseline: tp.Optional[tp.Union[np.ndarray, tp.List[np.ndarray]]] = None,
        return_reduced: bool = True,
        return_sigma: bool = False,
        **baseline_kwargs,
    ) -> tp.Union[float, tp.List[float], tp.Tuple[tp.Union[float, tp.List[float]], tp.Union[float, tp.List[float]]]]:
        """Compute chi-squared from the pre-computed normalised best-fit spectrum.

        Dispatches to the appropriate 1D or 2D routine based on the fitter type.
        For multisample fitters, computes one value per spectrum.

        :param baseline: Pre-computed boolean baseline mask, or list of masks
            for multisample fitters. If ``None``, detected automatically using
            the percentile method; pass ``baseline_kwargs`` to control it.
        :param return_reduced: If ``True``, return χ^2/dof (reduced chi-squared).
            χ^2/dof = 1 indicates a good fit, < 1 overfitting, > 1 poor model.
            Default ``True``.
        :param return_sigma: If ``True``, also return estimated σ value(s).
            Default ``False``.
        :param baseline_kwargs: Keyword arguments forwarded to the baseline
            detection method (``percentile``, ``proximity_threshold``,
            ``window_size``).
        :return: χ^2 or χ^2/dof as float for single spectra, or list of floats
            for multisample fitters. If ``return_sigma=True``, returns a tuple
            ``(chi2, sigma)`` or ``(list[chi2], list[sigma])``.

        """
        if isinstance(self.fitter, SpectrumFitter):
            return self._compute_spectrum_fitter(
                baseline, return_reduced, return_sigma, **baseline_kwargs
            )
        elif isinstance(self.fitter, Spectrum2DFitter):
            return self._compute_spectrum2d_fitter(
                baseline, return_reduced, return_sigma, **baseline_kwargs
            )
        else:
            raise NotImplementedError(
                f"Chi-square not implemented for fitter type {type(self.fitter).__name__}. "
                "Supported: SpectrumFitter, Spectrum2DFitter."
            )

    def _compute_spectrum_fitter(
        self,
        baseline: tp.Optional[tp.Union[np.ndarray, tp.List[np.ndarray]]],
        return_reduced: bool,
        return_sigma: bool,
        **baseline_kwargs,
    ):
        """Dispatch for SpectrumFitter (1D data) using pre-computed best_spectrum."""
        if self.fitter.multisample:
            chi2_vals, sigma_vals = [], []
            for i in range(len(self.fitter.x_exp)):
                x = self.fitter.x_exp[i].cpu().numpy()
                y_exp = self.fitter.y_exp[i].cpu().numpy()
                y_model = self.best_spectrum[i].cpu().numpy()

                norm_mode = self.fitter.norm_mode
                y_exp = normalize_spectrum(x, y_exp, norm_mode)
                y_model = normalize_spectrum(x, y_model, norm_mode)
                mask = baseline[i] if isinstance(baseline, list) else baseline

                result = self._compute_chi2_1d(
                    x, y_exp, y_model, mask, return_reduced, return_sigma, **baseline_kwargs
                )
                if return_sigma:
                    c, s = result
                    chi2_vals.append(c)
                    sigma_vals.append(s)
                else:
                    chi2_vals.append(result)

            chi2_out = chi2_vals if len(chi2_vals) > 1 else chi2_vals[0]
            if return_sigma:
                sigma_out = sigma_vals if len(sigma_vals) > 1 else sigma_vals[0]
                return chi2_out, sigma_out
            return chi2_vals if len(chi2_vals) > 1 else chi2_vals[0]

        else:
            x = self.fitter.x_exp.cpu().numpy()
            y_exp = self.fitter.y_exp.cpu().numpy()
            y_model = self.best_spectrum.cpu().numpy()

            norm_mode = self.fitter.norm_mode
            y_exp = normalize_spectrum(x, y_exp, norm_mode)
            y_model = normalize_spectrum(x, y_model, norm_mode)

            return self._compute_chi2_1d(
                x, y_exp, y_model, baseline, return_reduced, return_sigma, **baseline_kwargs
            )

    def _compute_spectrum2d_fitter(
        self,
        baseline: tp.Optional[tp.Union[np.ndarray, tp.List[np.ndarray]]],
        return_reduced: bool,
        return_sigma: bool,
        **baseline_kwargs,
    ):
        """Dispatch for Spectrum2DFitter (2D data) using pre-computed best_spectrum."""
        if self.fitter.multisample:
            chi2_vals, sigma_vals = [], []
            for i in range(len(self.fitter.x1_exp)):
                x1 = self.fitter.x1_exp[i].cpu().numpy()
                x2 = self.fitter.x2_exp[i].cpu().numpy()
                y_exp = self.fitter.y_exp[i].cpu().numpy()
                y_model = self.best_spectrum[i].cpu().numpy()

                norm_mode = self.fitter.norm_mode
                y_exp = normalize_spectrum2d(x1, x2, y_exp, norm_mode)
                y_model = normalize_spectrum2d(x1, x2, y_model, norm_mode)

                mask = baseline[i] if isinstance(baseline, list) else baseline
                result = self._compute_chi2_2d(
                    x1, x2, y_exp, y_model, mask, return_reduced, return_sigma, **baseline_kwargs
                )
                if return_sigma:
                    c, s = result
                    chi2_vals.append(c)
                    sigma_vals.append(s)
                else:
                    chi2_vals.append(result)

            chi2_out = chi2_vals if len(chi2_vals) > 1 else chi2_vals[0]
            if return_sigma:
                sigma_out = sigma_vals if len(sigma_vals) > 1 else sigma_vals[0]
                return chi2_out, sigma_out
            return chi2_vals if len(chi2_vals) > 1 else chi2_vals[0]

        else:
            x1 = self.fitter.x1_exp.cpu().numpy()
            x2 = self.fitter.x2_exp.cpu().numpy()
            y_exp = self.fitter.y_exp.cpu().numpy()
            y_model = self.best_spectrum.cpu().numpy()

            norm_mode = self.fitter.norm_mode
            y_exp = normalize_spectrum2d(x1, x2, y_exp, norm_mode)
            y_model = normalize_spectrum2d(x1, x2, y_model, norm_mode)

            return self._compute_chi2_2d(
                x1, x2, y_exp, y_model, baseline, return_reduced, return_sigma, **baseline_kwargs
            )


class UncertaintyAnalyzer:
    """Post-hoc uncertainty analysis for spectral fits.

    Computes confidence intervals for fitted parameters using one of five methods.
    The confidence intervals have correct statistical meaning only when the loss
    is proportional to SSE or MSE:

        loss ~ Σ (yᵢ - f(xᵢ; θ))²   or   loss ~ (1/N) Σ (yᵢ - f(xᵢ; θ))²

    This is the default loss in MaRs. Under this assumption the loss surface is
    proportional to -2 log L of a Gaussian model, and the chi-squared threshold
    derived from Wilks' theorem is:

        delta_thresh = chi2.ppf(confidence_level, df=1) / 2

    A parameter value is *inside* the confidence interval when:
        loss(θ) - loss(θ*) ≤ delta_thresh * scale_factor

    where ``scale_factor = loss* / (N - P)`` estimates the residual variance σ².

    ...
    Variance estimation procedures
    The definition of σ² depends on the uncertainty method:

    - ``"hessian"``, ``"profile"``, ``"mcmc"``
      Use the residual variance from the best fit:
      ``σ̂² = loss* / (N − P)`` (scale_factor).
      This is an unbiased estimate of the true noise variance **only when the loss
      is SSE or MSE** (the default in MaRS).  For other losses the value is not a
      valid variance, and confidence intervals lose their statistical interpretation.
    - ``"bootstrap"``
      Does not require an explicit σ² – uncertainty is derived directly from
      empirical refits on resampled data.
    - ``"trials"``
      No σ²; bounds are taken from the optimiser’s own trial history.
    - ``get_chi_square()`` (from ``ChiSquareComputer``)
      Estimates σ² **independently** from the baseline (signal‑free) regions of the
      experimental spectrum, avoiding the trivial χ²/dof = 1 that would occur if
      the loss‑based σ² were used.  This is only meaningful for SSE/MSE losses.
    ...
    """
    _METHODS: tp.Tuple[str, ...] = ("profile", "hessian", "mcmc", "trials", "bootstrap")

    def __init__(
        self,
        fitter: BaseSpectrumFitter,
        fit_result: FitResult,
        method: str = "hessian",
        confidence_level: float = 0.95,
    ) -> None:
        """
        :param fitter: Any :class:`BaseSpectrumFitter` or
            :class:`SpectrumCompositeFitter` that exposes
            ``_loss_from_params`` and ``param_space``.
        :param fit_result: The :class:`FitResult` returned by ``fitter.fit(...)``.
        :param method: Uncertainty method to use:

            - ``"hessian"``   — Fast, symmetric intervals via quadratic approximation
              of the loss surface. Correct statistical meaning only for SSE or MSE loss.
            - ``"profile"``   — Asymmetric intervals via re-optimisation of nuisance
              parameters. Correct statistical meaning only for SSE or MSE loss.
            - ``"mcmc"``      — Bayesian credible intervals via ensemble sampling.
              Correct statistical meaning only for SSE or MSE loss.
            - ``"trials"``    — Conservative bounding-box from optimiser history.
              No distributional assumption; requires dense trial coverage near optimum.
            - ``"bootstrap"`` — Distribution-free intervals via residual resampling.
              Correct statistical meaning for any smooth loss.

        :param confidence_level: Target coverage probability, e.g. ``0.95`` for 95 %.
        """

        if method not in self._METHODS:
            raise ValueError(f"method must be one of {self._METHODS}, got {method!r}")

        self.fitter = fitter
        self.fit_result = fit_result
        self.method = method
        self.confidence_level = confidence_level

        self.param_space = fitter.param_space
        self.best_params: tp.Dict[str, float] = {
            **self.param_space.fixed_params,
            **fit_result.best_params,
        }
        self.best_loss: float = float(fit_result.best_loss)
        self._varying_names: tp.List[str] = self.param_space.varying_names
        self._delta_thresh: float = 0.5 * float(scipy_chi2.ppf(confidence_level, df=1))
        self._n_data_points = self.fitter._n_data_points

    def __call__(
            self,
            param_names: tp.Optional[tp.List[str]] = None,
            **method_kwargs: tp.Any,
    ) -> UncertaintyResult:
        """Compute and return confidence intervals using the chosen method.

        Noise variance σ² definition
        For ``"hessian"``, ``"profile"``, and ``"mcmc"``, the estimated variance used
        to scale thresholds and covariance matrices is

            .. math::

                \hat\sigma^2 = \frac{L(\theta^*)}{N - P}

        where :math:`L(\theta^*)` is the best‑fit loss, :math:`N` is the number of
        data points, and :math:`P` is the number of varying parameters.
        This is an unbiased estimator of the true noise variance only when the loss
        is proportional to SSE or MSE (the default in MaRS).  For other losses the
        estimator is not interpretable as a variance; the resulting intervals should
        then be treated as sensitivity bounds, not as formal confidence intervals.

        The ``"bootstrap"`` method does not require an explicit σ^2, and
        ``"trials"`` does not use a noise model at all.

        For ``"hessian"``, ``"profile"``, ``"mcmc"`` -
        The resulting intervals have correct statistical meaning only when the
        loss is proportional to SSE or MSE, which is the default in MaRS.
        For custom or composite losses the intervals should be interpreted
        as sensitivity bounds only.

        :param param_names: Subset of varying parameter names to analyse.
            ``None`` analyses all varying parameters.
        :param method_kwargs: Additional keyword arguments forwarded to the
            underlying method. See each method for accepted kwargs.
        :return: :class:`UncertaintyResult` containing one
            :class:`ParameterCI` per requested parameter.
        :raises ValueError: If any name in ``param_names`` is not a varying parameter.
        """
        names: tp.List[str] = (
            param_names if param_names is not None else list(self._varying_names)
        )
        invalid: tp.List[str] = [n for n in names if n not in self._varying_names]
        if invalid:
            raise ValueError(f"Parameters not in varying space: {invalid}")

        dispatch: tp.Dict[str, tp.Callable[..., UncertaintyResult]] = {
            "hessian": self.hessian_ci,
            "profile": self.profile_likelihood,
            "mcmc": self.mcmc,
            "trials": self.trials_ci,
            "bootstrap": self.bootstrap_ci
        }
        return dispatch[self.method](names, **method_kwargs)

    def _loss_scalar(self, full_params: tp.Dict[str, float]) -> float:
        """Call fitter loss, return Python float."""
        with torch.no_grad():
            val = self.fitter._loss_from_params(full_params)
        return float(val)

    def _loss_from_vector(self, vec: np.ndarray) -> float:
        """Loss from a numpy vector."""
        params = self.param_space.vector_to_dict(vec)
        return self._loss_scalar(params)

    def _best_vector(self) -> np.ndarray:
        return self.param_space.dict_to_vector(self.best_params)

    def _vector_std(self) -> np.ndarray:
        trials = self._collect_trials()
        param_rows, _ = self._trials_to_arrays(trials)
        return np.std(param_rows, axis=-2)

    def _bounds_arrays(self) -> tp.Tuple[np.ndarray, np.ndarray]:
        lows = np.array([s.bounds[0] for s in self.param_space._varying_specs])
        highs = np.array([s.bounds[1] for s in self.param_space._varying_specs])
        return lows, highs

    def get_chi_square(
            self,
            baseline: tp.Optional[tp.Union[np.ndarray, list[np.ndarray]]] = None,
            percentile: float = 10,
            proximity_threshold: float = 0.15,
            window_size: tp.Optional[int] = None,
            return_reduced: bool = True,
            return_sigma: bool = False,
    ) -> tp.Union[float, list[float], tp.Tuple[tp.Union[float, list[float]], tp.Union[float, list[float]]]]:
        """Compute chi‑squared statistics from the best‑fit spectrum.

            Delegates to :class:`ChiSquareComputer` — no re‑simulation is performed.
            Residuals are computed in the normalised space defined by the fitter's
            ``norm_mode``.

            Noise estimation
            σ² is not derived from the best‑fit loss (which would trivially give
            χ²/dof ~ 1 and hide model inadequacies).  Instead, σ is estimated from the
            residuals in signal‑free (baseline) regions:

                - A baseline mask is created from the *experimental* spectrum using a
                  local‑percentile method (controlled by ``percentile``,
                  ``proximity_threshold``, ``window_size``).
                - The noise level σ is the standard deviation of the residuals
                  ``y_exp - y_model`` restricted to that mask.

            The experimental spectrum must have zero baseline (baseline‑subtracted
            before fitting). If a residual baseline is present, the σ estimate will be
            inflated and χ² will be underestimated.

        :param baseline: Pre-computed boolean baseline mask, or list of masks for
            multisample fitters. If ``None``, the baseline is detected automatically
            using the percentile method controlled by ``percentile``,
            ``proximity_threshold``, and ``window_size``.
        :param percentile: Percentile of local intensity used to identify baseline
            regions. Lower values select only the flattest regions.
            Default ``10``.
        :param proximity_threshold: Fractional proximity to the local percentile
            value below which a point is considered baseline. Default ``0.15``.
        :param window_size: Window size for the local percentile filter.
            ``None`` uses an automatic size based on the spectrum length.
        :param return_reduced: If ``True``, returns χ²/dof where
            ``dof = N - P`` (N data points, P varying parameters).
            Reduced χ² close to 1 indicates a good fit under Gaussian noise.
            Default ``True``.
        :param return_sigma: If ``True``, also returns the estimated noise σ
            alongside χ^2. Default ``False``.
        :return: χ^2 or χ^2/dof value (float), or a list of floats for multisample
            fitters. If ``return_sigma=True``, returns a tuple ``(chi2, sigma)``
            or ``(list[chi2], list[sigma])``.
        :raises ValueError: If ``fit_result.best_spectrum`` is ``None``. Re-fit
            with ``return_best_spectrum=True`` to populate it.
        """
        best_spec = self.fit_result.best_spectrum
        if best_spec is None:
            raise ValueError(
                "fit_result.best_spectrum is None. "
                "Re-fit with return_best_spectrum=True or compute chi-square manually."
            )

        if not self.fitter.proportional_to_mse:
            warnings.warn(
                "The fitter loss is not proportional to MSE. "
                "Chi-squared is computed by estimating σ from baseline regions of the "
                "experimental spectrum. This σ estimate is calibrated for MSE-like losses "
                "and is not the true noise level for other loss functions. "
                "As long as the loss is zero for a perfect fit, the reduced χ^2 remains "
                "qualitatively interpretable: χ^2/dof < 1 indicates overfitting, "
                "χ^2/dof > 1 indicates a poor model, and χ²/dof ≈ 1 indicates a good fit. "
                "However, the absolute value of χ^2 should not be used for formal "
                "goodness-of-fit tests or confidence interval construction with a non-MSE loss.",
                UserWarning,
                stacklevel=2,
            )

        computer = ChiSquareComputer(
            fitter=self.fitter,
            best_spectrum=best_spec,
            n_varying=len(self._varying_names),
        )
        return computer.compute(
            baseline=baseline,
            return_reduced=return_reduced,
            return_sigma=return_sigma,
            percentile=percentile,
            proximity_threshold=proximity_threshold,
            window_size=window_size,
        )

    def _find_crossings(
            self,
            grid: np.ndarray,
            delta: np.ndarray,
            threshold: float,
            best_val: float,
    ) -> tp.Tuple[float, float]:
        """Find where ``delta`` crosses ``threshold`` on each side of ``best_val``.

        :param grid: 1-D array of parameter values (the profile sweep grid).
        :param delta: 1-D array of ``profile_loss - best_loss`` values,
            same length as ``grid``.
        :param threshold: The chi-squared threshold ``delta_thresh``.
        :param best_val: Best-fit value of the profiled parameter; used to
            split the grid into left and right halves.
        :return: Tuple ``(ci_lo, ci_hi)``. Falls back to the grid edges if no
            crossing is found on the respective side.
        """
        ci_lo: float = float(grid[0])
        ci_hi: float = float(grid[-1])

        left_idx: np.ndarray = np.where(grid <= best_val)[0]
        for i in reversed(left_idx[:-1]):
            if delta[i] >= threshold > delta[i + 1]:
                ci_lo = float(np.interp(
                    threshold,
                    [delta[i + 1], delta[i]],
                    [grid[i + 1], grid[i]],
                ))
                break

        right_idx: np.ndarray = np.where(grid >= best_val)[0]
        for i in right_idx[:-1]:
            if delta[i] < threshold <= delta[i + 1]:
                ci_hi = float(np.interp(
                    threshold,
                    [delta[i], delta[i + 1]],
                    [grid[i], grid[i + 1]],
                ))
                break

        return ci_lo, ci_hi

    def get_scale_factor(self, n_params: tp.Optional[int] = None) -> float:
        """Estimate the residual variance σ² from the best-fit loss.

        Computes the unbiased estimator:

            scale_factor = loss* / (N - P)

        This equals σ̂² when ``loss = SSE = Σr_i^2``, and equals σ̂^2/N when
        ``loss = MSE = SSE/N``. In both cases the factor appears consistently
        in both the covariance formula and the profile threshold, so the 1/N
        difference between SSE and MSE cancels and the resulting CIs are identical.

        The returned value has correct statistical meaning as a variance estimate
        only when the loss is proportional to SSE or MSE, which is the default
        in MaRs. For other losses (e.g. NLL, weighted, composite) the value is
        numerically meaningless as a variance and will produce incorrectly
        scaled CIs.

        :param n_params: Number of free (varying) parameters P.
        :return: Estimated residual variance σ̂^2 (for SSE loss) or σ̂^2/N (for MSE loss).
        :raises ValueError: If degrees of freedom N - P ≤ 0.
        """
        if n_params is None:
            dof = self._n_data_points - len(self.fitter.param_space.varying_names)
            n_params = len(self.fitter.param_space.varying_names)
        else:
            dof = self._n_data_points - n_params
        if dof <= 0:
            raise ValueError(
                f"Degrees of freedom <= 0: {self._n_data_points} data points, "
                f"{n_params} varying parameters."
            )
        return self.best_loss / max(self._n_data_points - n_params, 1)

    def hessian_ci(
            self,
            param_names: tp.Optional[tp.List[str]] = None,
            step: tp.Optional[float] = 1e-4,
    ) -> UncertaintyResult:
        """Confidence intervals via numerical Hessian of the loss surface.

        Approximates the loss as quadratic near the optimum and derives symmetric
        intervals from the curvature. The covariance is estimated as:

            Cov(θ̂) = 2 · σ̂^2 · H^-1

        where ``H = ∂^2L/∂θ^2`` at ``θ*`` and ``σ̂^2 = loss* / (N - P)``.
        The factor of 2 arises from the least-squares identity ``H ≈ 2 JᵀJ / σ²``,
        so that ``2σ̂²H^-1 ≈ σ²(JᵀJ)^-1 = Cov(θ̂)``.
        The CI half-width for parameter i is then:

            Δθ_i = sqrt(2 · delta_thresh · Cov_ii)

        The intervals have correct statistical meaning only when the loss is
        proportional to SSE or MSE, which is the default in MaRs. For other
        losses the factor of 2 and the scale_factor do not cancel correctly
        against the Hessian, and the resulting intervals are not interpretable
        as confidence intervals.

        Intervals are symmetric by construction. For asymmetric uncertainty,
        strongly non-quadratic loss surfaces, or correlated parameters
        (condition number of H > 1e4), use ``profile_likelihood`` instead.

        :param param_names: Parameter names to compute CIs for.
            ``None`` uses all varying parameters.
        :param step: Finite-difference step size for Hessian estimation,
            scaled internally by the parameter standard deviations from trial
            history. Default ``1e-4``.
        :return: :class:`UncertaintyResult` with method ``"hessian"``.
            ``extra`` contains:

            - ``"hessian"``      — raw numerical Hessian matrix H.
            - ``"covariance"``   — estimated covariance matrix Cov(θ̂).
            - ``"correlation"``  — derived correlation matrix.
            - ``"scale_factor"`` — σ̂² = loss* / (N - P).
            - ``"delta_thresh"`` — chi-squared threshold (unscaled).
        """
        if not self.fitter.proportional_to_mse:
            warnings.warn(
                f"The optimizer uses a loss that is probably not proportional to MSE. "
                "The confidence intervals computed by UncertaintyAnalyzer have correct "
                "statistical meaning only when the loss is proportional to SSE or MSE "
                "(the default in MaRs). With the current loss the scale_factor, threshold, "
                "and covariance estimates are not statistically interpretable. "
                "Use method='bootstrap' for distribution-free intervals that do not "
                "rely on this assumption.",
                UserWarning,
                stacklevel=2,
            )

        names = param_names if param_names is not None else list(self._varying_names)
        lows, highs = self._bounds_arrays()
        x0 = self._best_vector()

        n_params = len(param_names)

        step_vec = step * self._vector_std()
        H = nd.Hessian(self._loss_from_vector, step=step_vec)(x0)

        eigvals = np.linalg.eigvalsh(H)
        if eigvals.min() <= 0:
            H = H + np.eye(n_params) * (abs(eigvals.min()) + 1e-10)
        try:
            H = 0.5 * (H + H.T)
            U, s, Vt = scipy.linalg.svd(H)
            rcond = 1e-8
            cutoff = rcond * s[0]
            s_inv = np.zeros_like(s)
            s_inv[s > cutoff] = 1.0 / s[s > cutoff]
            inv_H = Vt.T @ np.diag(s_inv) @ U.T

            rank = int(np.sum(s > cutoff))
            cond = s[0] / s[rank - 1] if rank > 0 else np.inf
            if cond > 1e4:
                warnings.warn(
                    f"Ill-conditioned Hessian (condition number = {cond:.2e}). "
                    f"Effective rank: {rank}/{n_params}. "
                    f"This indicates strong parameter trade-offs or near-non-identifiability. "
                    f"Confidence intervals may be unreliable. Consider: "
                    f"(1) re-parameterizing the model to reduce correlations, "
                    f"(2) using profile likelihood or "
                    f"(3) fixing parameters with large uncertainties.",
                    UserWarning,
                    stacklevel=2
                )

        except np.linalg.LinAlgError:
            inv_H = np.linalg.pinv(H)

        scale_factor = self.get_scale_factor()
        cov = 2.0 * scale_factor * inv_H

        intervals = {}
        for name in names:
            idx = self._varying_names.index(name)
            cov_ii = cov[idx, idx]

            if cov_ii <= 0:
                warnings.warn(f"Non-positive covariance diagonal for '{name}'; CI set to bounds.")
                intervals[name] = ParameterCI(lower=float(lows[idx]), upper=float(highs[idx]), best=float(x0[idx]))
                continue

            half = float(np.sqrt(2.0 * self._delta_thresh * cov_ii))
            best_val = float(x0[idx])
            lo = max(float(lows[idx]), best_val - half)
            hi = min(float(highs[idx]), best_val + half)
            intervals[name] = ParameterCI(lower=lo, upper=hi, best=best_val)

        stds = np.sqrt(np.diag(cov))
        corr = cov / np.outer(stds, stds)

        return UncertaintyResult(
            intervals=intervals,
            method="hessian",
            confidence_level=self.confidence_level,
            extra={
                "hessian": H,
                "correlation": corr,
                "covariance": cov,
                "delta_thresh": self._delta_thresh,
                "scale_factor": scale_factor,
            },
        )

    def profile_likelihood(
            self,
            param_names: tp.Optional[tp.List[str]] = None,
            n_points: int = 4,
    ) -> UncertaintyResult:
        """Asymmetric confidence intervals via profile likelihood.

        For each parameter of interest θᵢ:

        1. Sweeps θ over a grid spanning its full parameter bounds.
        2. At each grid point re-minimises the loss over all other
           (nuisance) parameters using L-BFGS-B.
        3. Finds the two crossings of the scaled threshold:

               loss_profile(θᵢ) - loss*  ≤  delta_thresh · σ̂^2

        where ``σ̂^2 = loss* / (N - P)`` scales the threshold into the
        same units as the loss.

        The intervals have correct statistical meaning only when the loss is
        proportional to SSE or MSE, which is the default in MaRs. For other
        losses the scaled threshold has no chi-squared interpretation and the
        crossing points are not true confidence bounds.

        :param param_names: Parameter names to compute CIs for.
            ``None`` uses all varying parameters.
        :param n_points: Number of grid points per parameter sweep.
            More points give smoother profiles and more accurate crossing
            interpolation at the cost of additional loss evaluations.
            ``n_points=4`` may be too coarse for narrow parameters;
            20–50 is recommended for publication-quality profiles.
        :return: :class:`UncertaintyResult` with method ``"profile"``.
            ``extra`` contains ``"profiles"``: a dict mapping each parameter
            name to ``{"grid", "delta_loss", "threshold"}``.
        """
        if not self.fitter.proportional_to_mse:
            warnings.warn(
                f"The optimizer uses a loss that is probably not proportional to MSE. "
                "The confidence intervals computed by UncertaintyAnalyzer have correct "
                "statistical meaning only when the loss is proportional to SSE or MSE "
                "(the default in MaRs). With the current loss the scale_factor, threshold, "
                "and covariance estimates are not statistically interpretable. "
                "Use method='bootstrap' for distribution-free intervals that do not "
                "rely on this assumption.",
                UserWarning,
                stacklevel=2,
            )

        names = param_names if param_names is not None else list(self._varying_names)
        lows, highs = self._bounds_arrays()
        x0 = self._best_vector()
        n_varying = len(self._varying_names)

        scale_factor = self.get_scale_factor()
        thresh = self._delta_thresh * scale_factor

        intervals = {}
        profiles = {}

        for name in tqdm.tqdm(names):
            pi = self._varying_names.index(name)
            best_val = float(x0[pi])
            lo_bound, hi_bound = float(lows[pi]), float(highs[pi])

            grid = np.linspace(lo_bound, hi_bound, n_points)
            profile_losses = np.empty(len(grid))

            nuisance_mask = np.ones(n_varying, dtype=bool)
            nuisance_mask[pi] = False
            nuisance_idx = np.where(nuisance_mask)[0]

            nu_init = x0[nuisance_mask].copy()
            for gi, pval in enumerate(grid):
                if n_varying == 1:
                    profile_losses[gi] = self._loss_from_vector(np.array([pval]))
                    continue

                def _obj(nu: np.ndarray, _pval=pval, _nu_idx=nuisance_idx) -> float:
                    full_vec = x0.copy()
                    full_vec[pi] = _pval
                    full_vec[_nu_idx] = nu
                    return self._loss_from_vector(full_vec)

                res = scipy_minimize(
                    _obj, nu_init, method="L-BFGS-B",
                    bounds=list(zip(lows[nuisance_mask], highs[nuisance_mask])),
                    options={"maxiter": 50, "ftol": 1e-12}
                )
                profile_losses[gi] = float(res.fun)
                nu_init = res.x  #

            delta_profile = profile_losses - self.best_loss
            ci_lo, ci_hi = self._find_crossings(grid, delta_profile, thresh, best_val)
            ci_lo = max(ci_lo, lo_bound)
            ci_hi = min(ci_hi, hi_bound)

            intervals[name] = ParameterCI(lower=ci_lo, upper=ci_hi, best=best_val)
            profiles[name] = {"grid": grid, "delta_loss": delta_profile, "threshold": thresh}

        return UncertaintyResult(
            intervals=intervals,
            method="profile",
            confidence_level=self.confidence_level,
            extra={"profiles": profiles},
        )

    def mcmc(
            self,
            param_names: tp.Optional[tp.List[str]] = None,
            n_steps: int = 200,
            n_walkers: int = 16,
            burn_in: int = 200,
            spread: float = 1e-3,
            seed: int = 42,
    ) -> UncertaintyResult:
        """Bayesian credible intervals via MCMC ensemble sampling (``emcee``).

        Constructs a pseudo-posterior with flat priors within parameter bounds:

            log p(θ) = -loss(θ) / (2 · σ̂²),   σ̂² = loss* / (N - P)

        Walkers are initialised in a tight Gaussian ball of radius
        ``spread * (high - low)`` around the best-fit point, then run for
        ``burn_in`` steps (discarded) followed by ``n_steps`` production steps.

        The returned intervals are Bayesian credible intervals, not frequentist
        confidence intervals. They  coincide
        with frequentist CIs asymptotically only when the loss is proportional
        to SSE or MSE, which is the default in MaRs. For other losses the
        division by ``2σ̂²`` does not correctly normalise the pseudo-posterior
        and the sampled distribution has no probabilistic interpretation.

        :param param_names: Parameter names to compute CIs for.
            ``None`` uses all varying parameters.
        :param n_steps: Number of production steps per walker after burn-in.
        :param n_walkers: Ensemble size. Raised automatically to
            ``max(n_walkers, 2 · n_dim + 2)`` and kept even.
        :param burn_in: Number of burn-in steps discarded before production.
        :param spread: Initial ball radius as a fraction of each parameter's
            bound width ``(high - low)``.
        :param seed: Integer seed for reproducible walker initialisation.
        :return: :class:`UncertaintyResult` with method ``"mcmc"``.
            ``extra`` contains ``"sampler"`` (the ``emcee.EnsembleSampler``)
            and ``"chain"`` (the flattened production chain, shape
            ``(n_walkers * n_steps, n_dim)``).
        """
        if not self.fitter.proportional_to_mse:
            warnings.warn(
                f"The optimizer uses a loss that is probably not proportional to MSE. "
                "The confidence intervals computed by UncertaintyAnalyzer have correct "
                "statistical meaning only when the loss is proportional to SSE or MSE "
                "(the default in MaRs). With the current loss the scale_factor, threshold, "
                "and covariance estimates are not statistically interpretable. "
                "Use method='bootstrap' for distribution-free intervals that do not "
                "rely on this assumption.",
                UserWarning,
                stacklevel=2,
            )

        names = param_names if param_names is not None else list(self._varying_names)
        lows, highs = self._bounds_arrays()
        x0 = self._best_vector()
        n_dim = len(x0)
        scale_factor = self.get_scale_factor()

        n_walkers = max(n_walkers, 2 * n_dim + 2)
        if n_walkers % 2 != 0:
            n_walkers += 1

        def log_prob(vec: np.ndarray) -> float:
            if np.any(vec < lows) or np.any(vec > highs):
                return -np.inf
            return -self._loss_from_vector(vec) / (2.0 * scale_factor)

        rng = np.random.default_rng(seed)
        p0 = x0 + spread * rng.standard_normal((n_walkers, n_dim)) * (highs - lows)
        p0 = np.clip(p0, lows, highs)

        sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_prob)
        sampler.run_mcmc(p0, burn_in, progress=False)
        sampler.reset()
        sampler.run_mcmc(None, n_steps, progress=False)

        chain = sampler.get_chain(flat=True)

        alpha = 1.0 - self.confidence_level
        lo_pct, hi_pct = 100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)

        intervals = {}
        for name in names:
            idx = self._varying_names.index(name)
            samples = chain[:, idx]
            lo, hi = float(np.percentile(samples, lo_pct)), float(np.percentile(samples, hi_pct))
            intervals[name] = ParameterCI(lower=lo, upper=hi, best=float(x0[idx]))

        return UncertaintyResult(
            intervals=intervals,
            method="mcmc",
            confidence_level=self.confidence_level,
            extra={"sampler": sampler, "chain": chain},
        )

    def trials_ci(
            self,
            param_names: tp.Optional[tp.List[str]] = None,
    ) -> UncertaintyResult:
        """Conservative sensitivity bounds from existing optimiser trial history.

        Collects all trials within the loss cutoff:

            cutoff = loss* · (1 + delta_thresh)

        and takes the min/max of each parameter across accepted trials as the
        CI bounds. This is a bounding-box approach rather than a percentile
        approach, because optimiser history is not a representative random sample.

        The bounds are sensitivity bounds, not formal confidence intervals.
        No distributional assumption on the loss is required, but the optimiser
        must have explored the region near the optimum densely. Sparse coverage
        produces artificially narrow bounds. A warning is raised when fewer
        than 5 trials fall within the cutoff.

        ``confidence_level`` is not used and is reported as ``None`` in the
        result, because the coverage of the bounding-box is not known.

        :param param_names: Parameter names to compute bounds for.
            ``None`` uses all varying parameters.
        :return: :class:`UncertaintyResult` with method ``"trials"``
            and ``confidence_level=None``.
            ``extra`` contains:

            - ``"n_good_trials"`` — number of trials within the cutoff.
            - ``"loss_cutoff"``   — absolute loss threshold used.
            - ``"delta_thresh"``  — relative threshold applied (``delta_thresh · loss*``).
        """
        names = param_names if param_names is not None else list(self._varying_names)
        trials = self._collect_trials()
        lows, highs = self._bounds_arrays()
        x0 = self._best_vector()

        param_rows, losses = self._trials_to_arrays(trials)
        cutoff = self.best_loss + self._delta_thresh * self.best_loss
        good = losses <= cutoff
        P_good = param_rows[good]

        if P_good.shape[0] < 5:
            warnings.warn(
                f"Only {P_good.shape[0]} trials within threshold. "
                "CIs may be unreliable due to sparse coverage."
            )

        intervals = {}
        for name in names:
            idx = self._varying_names.index(name)
            col = P_good[:, idx]
            lo = max(float(col.min()), float(lows[idx]))
            hi = min(float(col.max()), float(highs[idx]))
            intervals[name] = ParameterCI(lower=lo, upper=hi, best=float(x0[idx]))

        return UncertaintyResult(
            intervals=intervals,
            method="trials",
            confidence_level=None,
            extra={
                "n_good_trials": int(good.sum()),
                "loss_cutoff": float(cutoff),
                "delta_thresh": self._delta_thresh * self.best_loss,
            },
        )

    def bootstrap_ci(
        self,
        param_names: tp.Optional[tp.List[str]] = None,
        n_bootstrap: int = 200,
        local_opt_method: str = "L-BFGS-B",
        seed: int = 42,
    ) -> UncertaintyResult:
        """Distribution-free confidence intervals via residual bootstrap.

        For each bootstrap replicate:

        1. Draws a perturbed dataset via ``fitter._loss_from_params_random``,
           which resamples experimental targets with replacement internally.
        2. Re-fits the model from the best-fit point ``θ*`` using a local optimiser.

        Empirical percentiles of the resulting parameter ensemble define the CIs:

            CI = [percentile(α/2), percentile(1 - α/2)],   α = 1 - confidence_level

        The intervals have correct statistical meaning for any smooth loss function.
        No SSE or MSE assumption is required. This is the only method in this class
        whose intervals retain correct statistical meaning when the loss is a custom
        or composite function.

        Assumes residuals are exchangeable (i.i.d. or stationary). Systematic
        trends or heteroscedastic noise violate this assumption. Local re-optimisation
        starts from the global best-fit ``θ*``; multimodal bootstrap loss landscapes
        may converge to different local minima across replicates and produce
        unreliable intervals.

        :param param_names: Parameter names to compute CIs for.
            ``None`` uses all varying parameters.
        :param n_bootstrap: Number of bootstrap refits. ``>= 200`` is recommended
            for stable percentile estimates; fewer replicates yield coarse tail
            percentiles.
        :param local_opt_method: SciPy optimiser name for local refitting,
            e.g. ``"L-BFGS-B"`` (default) or ``"Nelder-Mead"``.
        :param seed: Base random seed. Replicate ``i`` uses seed ``seed + i``
            for reproducibility.
        :return: :class:`UncertaintyResult` with method ``"bootstrap"``.
            ``extra`` contains:

            - ``"n_bootstrap"`` — number of replicates performed.
            - ``"boot_params"`` — array of shape ``(n_bootstrap, n_params)``
              with the re-fitted parameter vectors.
        """
        names = param_names if param_names is not None else list(self._varying_names)
        lows, highs = self._bounds_arrays()
        x0 = self._best_vector()
        device = getattr(self.fitter, "device", torch.device("cpu"))

        boot_params = np.empty((n_bootstrap, len(x0)))

        for i in range(n_bootstrap):
            iter_seed = seed + i

            def loss_fn(vec, _seed=iter_seed):
                gen = torch.Generator(device=device).manual_seed(_seed)
                return self.fitter._loss_from_params_random(
                    self.param_space.vector_to_dict(vec), generator=gen
                ).item()

            res = scipy_minimize(
                loss_fn,
                x0,
                method=local_opt_method,
                bounds=list(zip(lows, highs)),
                options={"maxiter": 150, "ftol": 1e-10}
            )
            boot_params[i] = res.x

        alpha = 1.0 - self.confidence_level
        lo_pct, hi_pct = 100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)
        intervals = {}
        for name in names:
            idx = self._varying_names.index(name)
            col = boot_params[:, idx]
            lo = float(np.percentile(col, lo_pct))
            hi = float(np.percentile(col, hi_pct))
            lo = max(lo, float(lows[idx]))
            hi = min(hi, float(highs[idx]))
            intervals[name] = ParameterCI(lower=lo, upper=hi, best=float(x0[idx]))

        return UncertaintyResult(
            intervals=intervals,
            method="bootstrap",
            confidence_level=self.confidence_level,
            extra={"n_bootstrap": n_bootstrap, "boot_params": boot_params},
        )

    def _collect_trials(self) -> list:
        return UncertaintyAnalyzer._collect_trials_static(self.fit_result)

    @staticmethod
    def _collect_trials_static(fit_result) -> list:
        backend = fit_result.optimizer_info.get("backend")
        if backend == "optuna":
            study = fit_result.optimizer_info.get("study")
            if study is not None:
                return [t for t in study.trials if t.state.is_finished()]
            raw = fit_result.optimizer_info.get("trials", [])
            return raw
        elif backend == "nevergrad":
            return fit_result.optimizer_info.get("trials", [])
        return []

    def _trials_to_arrays(
        self, trials: list
    ) -> tp.Tuple[np.ndarray, np.ndarray]:
        """Convert a list of optuna/nevergrad trial objects to numpy arrays."""
        rows, losses = [], []
        for t in trials:
            if hasattr(t, "params"):
                p = t.params
                v = t.value
            else:
                p = t.get("params", {})
                v = t.get("value")
            if v is None:
                continue
            vec = [float(p.get(n, float("nan"))) for n in self._varying_names]
            if not any(np.isnan(vec)):
                rows.append(vec)
                losses.append(float(v))
        if not rows:
            return np.zeros((0, len(self._varying_names))), np.array([])
        return np.array(rows), np.array(losses)
