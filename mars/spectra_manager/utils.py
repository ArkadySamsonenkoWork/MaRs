from dataclasses import dataclass
from enum import Enum
import typing as tp

import torch


def compute_matrix_element(vector_down: torch.Tensor, vector_up: torch.Tensor, G: torch.Tensor):
    """Compute transition matrix element <ψ_up| G |ψ_down>.

    :param vector_down: Lower-state eigenvector. Shape [..., N]
    :param vector_up: Upper-state eigenvector. Shape [..., N]
    :param G: Operator matrix (e.g., g-tensor component). Shape [..., N, N]
    :return: Complex-valued transition amplitude. Shape [...]
    """
    tmp = torch.matmul(G.unsqueeze(-3), vector_down.unsqueeze(-1))
    return (vector_up.conj() * tmp.squeeze(-1)).sum(dim=-1)


class OutputSpectraMode(str, Enum):
    TOTAL = "total"
    TRANSITIONS = "transitions"


@dataclass(frozen=True)
class ComputationalDetails:
    """
    Specifies computational parameters used during the generation of EPR spectra.

    These settings control numerical integration, adaptive field resolution,
    and intensity-based filtering of transitions.

    Parameters
    ----------
    integration_chunk_size : int, default=128
        Number of magnetic field points processed together during spectrum integration.
        Larger values may improve performance but increase memory usage.

    integration_gaussian_cutoff : float, default= sqrt(5) with exp(-5) = 0.0067 (0.7 %)
        Absolute cutoff (in units of standard deviations) beyond which the Gaussian
        contribution is assumed to be zero. Used during final spectrum creation from separate lines to skip
        unnecessary evaluations when |c·(B_mean - B_val)|> cutoff.

    integration_gaussian_method : str, default="exp"
        Method used to evaluate the Gaussian function exp(-x²) during final integration:
        - "exp": uses exact PyTorch exponential (higher accuracy),
        - "approx": uses a fast 6th-order rational approximation (see ``gaussian_approx``).

    integration_level : int, default=0
        Level of geometric refinement for powder integration:
        - 0: basic centroid integration (triangle midpoint for spherical, bi-centric midpoint for axial),
        - 1–3: barycentric subdivision of orientation triangles (spherical symmetry only),
          increasing angular sampling density by a factor of 3^level.
        Higher levels improve accuracy for highly anisotropic systems but
        increase computation time. Axial integrators only support level 0.

    integration_natural_width : float, default=1e-6
        Minimum intrinsic linewidth added to every transition. Measures in FWHM
        Prevents division-by-zero or extreme sharpening when user-provided widths are
        very small or zero. Also it can be used as substitution for ordinary gaussian broadaning in the sample.

    field_factor: Scaling factor that controls the minimum contribution of the
            spectral field resolution to the effective line width. Inside
            ``_compute_effective_width`` the condition
            ``width_eff² ≥ (field_factor * dB)^2`` is enforced, where ΔB is the difference
            between consecutive spectral field points. Default is 3.0, which ensures that
            the effective width is at least three times the field step.

    integration_clamp_width_factor : float or None, optional
        Controls how strongly geometric broadening is enforced in the effective linewidth.
        The effective width combines natural width and field spread across orientations.
        This factor sets a lower bound on the relative contribution of the geometric term:
          w_eff² ≥ w₀² · (1 + clamp_width_factor · (ΔB/w₀)²)
        Defaults:
          - 3.0 for 'mean' spherical integration,
          - 2.0 for 'mean' axial integration,
          - 1.0 for 'analytical' methods (no extra clamping needed).
        If None, a sensible default is chosen based on symmetry and computation method.
        Higher values can fix problem of 'oscillating' spectrum but for too high values spectrum become broaden.
        If the oscillation is too high we recomend to use integration_computation_method == "analuytical"
        or set integration_level == 1, 2, 3

    integration_computation_method : str, default="mean"
        Strategy for evaluating transition contributions over orientation space:
        - "mean": evaluates the line shape at a effective field (e.g., triangle centroids),
        - "analytical": integrates exactly using antiderivatives over triangles (spherical)
          or line segments (axial). More accurate for broad or rapidly varying lines.

    res_field_r_tol : float, default=1e-5
        Relative tolerance for adaptive splitting of magnetic field sectors
        during res-field procedures. Smaller values yield finer
        field resolution at higher computational cost.

    res_field_split_max_iterations : int, default=20
        Maximum number of recursive sector splits allowed during res-field procedures.

    intensity_threshold : float, default=1e-2
        Transitions with intensity below this fraction of the maximum intensity
        are discarded.

    time_evolution_angle_average_steps : int, default=4
        The number of discretization steps used in the propagator computation to
        average the signal over rotations around the z-axis. This parameter controls
        the sampling density of the third Euler angle (γ) during orientational averaging.
    """
    integration_chunk_size: int = 128
    integration_gaussian_cutoff: float = 2.24
    integration_gaussian_method: str = "exp"
    integration_level: int = 0
    integration_natural_width: float = 1e-5
    field_factor: int = 3

    integration_clamp_width_factor: tp.Optional[float] = None
    integration_computation_method: str = "mean"
    res_field_r_tol: float = 1e-5
    res_field_split_max_iterations: int = 20
    intensity_threshold: float = 1e-2
    time_evolution_angle_average_steps: int = 4


@dataclass(frozen=True)
class ExpandedComputationalDetails(ComputationalDetails):
    """
    Configuration parameters for automatic field/frequency axis generation
    in expanded spectra classes (`StationarySpectraExpanded`, `TruncTimeSpectraExpanded`,
    `CoupledTimeSpectraExpanded`, `DensityTimeSpectraExpanded`, `StationaryFreqSpectraExpanded`).

    :param num_points: Number of points in the generated axis.
    :param spectral_width_part: Fraction of the estimated spectral width used to determine the sweep window.
    :param width_factor: Multiplier for the maximum linewidth to extend the sweep.
    :param min_exp_field: Absolute lower bound for the sweep (field or frequency).
    :param max_exp_field: Absolute upper bound for the sweep.
    :param width_cutoff: Only linewidths above this value are considered when estimating the sweep range.
    """
    num_points: int = 4000
    spectral_width_part: float = 0.6
    width_factor: float = 3.0
    min_exp_field: float = 0.0
    max_exp_field: float = 2.0
    width_cutoff: float = 0.5
