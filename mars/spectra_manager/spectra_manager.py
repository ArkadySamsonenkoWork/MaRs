import typing as tp
from functools import wraps
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum

import torch
import torch.fft as fft
import torch.nn as nn

from .. import constants
from .. import mesher
from .res_line_solvers import res_field_algorithm, res_freq_algorithm
from .. import spin_model

from .spectral_integration import BaseSpectraIntegrator,\
    SphereSpectraIntegrator, MeanIntegrator, AxialSpectraIntegrator
from ..population import BaseTimeDepPopulator, StationaryPopulator, LevelBasedPopulator,\
    RWADensityPopulator, PropagatorDensityPopulator, BasePopulator
from ..population import contexts


def compute_matrix_element(vector_down: torch.Tensor, vector_up: torch.Tensor, G: torch.Tensor):
    """Compute transition matrix element <ψ_up| G |ψ_down>.

    :param vector_down: Lower-state eigenvector. Shape [..., N]
    :param vector_up: Upper-state eigenvector. Shape [..., N]
    :param G: Operator matrix (e.g., g-tensor component). Shape [..., N, N]
    :return: Complex-valued transition amplitude. Shape [...]
    """
    tmp = torch.matmul(G.unsqueeze(-3), vector_down.unsqueeze(-1))
    return (vector_up.conj() * tmp.squeeze(-1)).sum(dim=-1)


class PostSpectraProcessing(nn.Module):
    """Apply line-broadening (Gaussian, Lorentzian, or Voigt) to raw stick
    spectra.

    Supports batched and non-batched inputs. Automatically selects
    broadening method based on non-zero FWHM parameters. Convolution
    performed in Fourier domain.

    :param gauss: Gaussian FWHM (in same units as magnetic_field). Shape
        [] or [*batch_dims]
    :param lorentz: Lorentzian FWHM (in same units as magnetic_field).
        Shape [] or [*batch_dims]
    """
    def __init__(self, eps: float = 1e-7, *args, **kwargs):
        """
        :param gauss: The gauss parameter.

        The shape is [batch_size] or []
        :param lorentz: The lorentz parameter. The shape is [batch_size] or []
        """
        super().__init__()
        self.register_buffer("eps", torch.tensor(1e-7))

    def _skip_broader(self, gauss: torch.Tensor, lorentz: torch.Tensor,
                      magnetic_fields: torch.Tensor, spec: torch.Tensor) -> torch.Tensor:
        return spec

    def _broading_fabric(self, gauss: torch.Tensor, lorentz: torch.Tensor) -> torch.Tensor:
        gauss_zero = (gauss == 0).all()
        lorentz_zero = (lorentz == 0).all()

        if gauss_zero and lorentz_zero:
            return self._skip_broader
        elif not gauss_zero and lorentz_zero:
            return self._gauss_broader
        elif gauss_zero and not lorentz_zero:
            return self._lorentz_broader
        else:
            return self._voigt_broader

    def forward(self, gauss: torch.Tensor, lorentz: torch.Tensor,
                magnetic_field: torch.Tensor, spec: torch.Tensor) -> torch.Tensor:
        """
        :param gauss: Tensor of shape [] or [*batch_dims].
        Values are provided as the full width at half maximum (FWHM) and are expressed in:
            - tesla (T) for field-dependent spectra,
            - hertz (Hz) for frequency-dependent spectra.

        :param lorentz: Tensor of shape [] or [*batch_dims]
        Values are provided as the full width at half maximum (FWHM) and are expressed in:
            - tesla (T) for field-dependent spectra,
            - hertz (Hz) for frequency-dependent spectra.

        :param magnetic_field: Tensor of shape [N] or [*batch_dims, N] or [*bathc_dims, T, N]
        :param spec: Spectrum tensor of shape [N] or [*batch_dims, N] or [*bathc_dims, T, N]
        :return: Broadened spectrum, same shape as spec with the shape [N] or [*batch_dims, N]
        or [*batch_dims, T, N] or [T, N] depending on input
        """
        target_batch_dims = spec.dim() - 1
        if gauss.dim() < target_batch_dims:
            gauss = gauss.reshape(*gauss.shape, *(1,) * (target_batch_dims - gauss.dim()))

        if lorentz.dim() < target_batch_dims:
            lorentz = lorentz.reshape(*lorentz.shape, *(1,) * (target_batch_dims - lorentz.dim()))

        _broading_method = self._broading_fabric(gauss, lorentz)
        return _broading_method(gauss, lorentz, magnetic_field, spec)

    def _build_lorentz_kernel(self, magnetic_field: torch.Tensor, fwhm_lorentz: torch.Tensor) -> torch.Tensor:
        """
        :param magnetic_field: Shape [*batch_dims, N].

        :param fwhm_lorentz: Shape [*batch_dims]
        :return: Kernel of shape [*batch_dims, N]
        """
        dH = magnetic_field[..., 1] - magnetic_field[..., 0]
        N = magnetic_field.shape[-1]
        device = magnetic_field.device

        idx = torch.arange(N, device=device) - N // 2

        batch_dims = magnetic_field.dim() - 1
        idx_shape = [1] * batch_dims + [N]
        idx = idx.view(*idx_shape)

        dH_expanded = dH.unsqueeze(-1)
        gamma = (fwhm_lorentz.unsqueeze(-1) / 2)
        x = idx * dH_expanded
        mask = (gamma == 0)

        safe_gamma = gamma.masked_fill(mask, 1.0)
        L = (safe_gamma / torch.pi) / (x ** 2 + safe_gamma ** 2)

        delta = torch.zeros_like(L)
        delta[..., N // 2] = 1.0 / dH
        L = torch.where(mask, delta, L)
        return L

    def _build_gauss_kernel(self, magnetic_field: torch.Tensor, fwhm_gauss: torch.Tensor) -> torch.Tensor:
        """
        :param magnetic_field: Shape [*batch_dims, N].

        :param fwhm_gauss: Shape [*batch_dims]
        :return: Kernel of shape [*batch_dims, N]
        """
        dH = magnetic_field[..., 1] - magnetic_field[..., 0]
        N = magnetic_field.shape[-1]
        device = magnetic_field.device

        idx = torch.arange(N, device=device) - N // 2

        batch_dims = magnetic_field.dim() - 1
        idx_shape = [1] * batch_dims + [N]
        idx = idx.view(*idx_shape)

        dH_expanded = dH.unsqueeze(-1)
        sigma = fwhm_gauss.unsqueeze(-1) / (2 * (2 * torch.log(torch.tensor(2.0, device=device))) ** 0.5)
        x = idx * dH_expanded

        mask = (sigma == 0)
        safe_sigma = sigma.masked_fill(mask, 1.0)
        G = torch.exp(-0.5 * (x / safe_sigma) ** 2) / (safe_sigma * (2 * torch.pi) ** 0.5)

        delta = torch.zeros_like(G)
        delta[..., N // 2] = 1.0 / dH
        G = torch.where(mask, delta, G)
        return G

    def _build_voigt_kernel(self,
                            magnetic_field: torch.Tensor,
                            fwhm_gauss: torch.Tensor,
                            fwhm_lorentz: torch.Tensor) -> torch.Tensor:
        """
        :param magnetic_field: Shape [*batch_dims, N].

        :param fwhm_gauss: Shape [*batch_dims]
        :param fwhm_lorentz: Shape [*batch_dims]
        :return: Kernel of shape [*batch_dims, N]
        """
        N = magnetic_field.shape[-1]
        G = self._build_gauss_kernel(magnetic_field, fwhm_gauss)
        L = self._build_lorentz_kernel(magnetic_field, fwhm_lorentz)

        Gf = fft.rfft(torch.fft.ifftshift(G, dim=-1), dim=-1)
        Lf = fft.rfft(torch.fft.ifftshift(L, dim=-1), dim=-1)

        Vf = Gf * Lf
        V = torch.fft.fftshift(fft.irfft(Vf, n=N, dim=-1), dim=-1)
        return V

    def _apply_convolution(self, spec: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """Apply convolution via FFT.

        :param spec: Shape [*batch_dims, N]
        :param kernel: Shape [*batch_dims, N]
        :return: Convolved spectrum of shape [*batch_dims, N]
        """
        S = fft.rfft(spec, dim=-1)
        K = fft.rfft(torch.fft.ifftshift(kernel, dim=-1), dim=-1)
        out = fft.irfft(S * K, n=spec.shape[-1], dim=-1)
        return out

    def _gauss_broader(self,
                       gauss: torch.Tensor, lorentz: torch.Tensor,
                       magnetic_field: torch.Tensor, spec: torch.Tensor) -> torch.Tensor:
        """
        :param gauss: Shape [*batch_dims].

        :param magnetic_field: Shape [*batch_dims, N]
        :param spec: Shape [*batch_dims, N]
        """
        dH = magnetic_field[..., 1] - magnetic_field[..., 0]
        kernel = self._build_gauss_kernel(magnetic_field, gauss)
        return self._apply_convolution(spec, kernel) * dH.unsqueeze(-1)

    def _lorentz_broader(self,
                         gauss: torch.Tensor, lorentz: torch.Tensor,
                         magnetic_field: torch.Tensor, spec: torch.Tensor) -> torch.Tensor:
        """
        :param lorentz: Shape [*batch_dims].

        :param magnetic_field: Shape [*batch_dims, N]
        :param spec: Shape [*batch_dims, N]
        """
        dH = magnetic_field[..., 1] - magnetic_field[..., 0]
        kernel = self._build_lorentz_kernel(magnetic_field, lorentz)
        return self._apply_convolution(spec, kernel) * dH.unsqueeze(-1)

    def _voigt_broader(self, gauss: torch.Tensor, lorentz: torch.Tensor,
                       magnetic_field: torch.Tensor, spec: torch.Tensor) -> torch.Tensor:
        """
        :param gauss: Shape [*batch_dims].

        :param lorentz: Shape [*batch_dims]
        :param magnetic_field: Shape [*batch_dims, N]
        :param spec: Shape [*batch_dims, N]
        """
        dH = magnetic_field[..., 1] - magnetic_field[..., 0]
        kernel = self._build_voigt_kernel(magnetic_field, gauss, lorentz)
        return self._apply_convolution(spec, kernel) * dH.unsqueeze(-1).pow(2)


class OutputSpectraMode(str, Enum):
    TOTAL = "total"
    TRANSITIONS = "transitions"


@dataclass
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


class BaseProcessing(nn.Module, ABC):
    """Base class for spectral processing over orientation meshes.
    """
    def __init__(self,
                 mesh: mesher.BaseMesh,
                 computational_details: ComputationalDetails = ComputationalDetails(),
                 output_mode: OutputSpectraMode = OutputSpectraMode.TOTAL,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32):
        """
        :param mesh: Mesh object defining orientation sampling grid.

        :param computational_details: The details of final spectral integration and spectra processing

        :param output_mode: Controls spectrum organization:
            - "total": returns conventional summed spectrum over all orientations
            - "transitions": returns per-orientation/transition contributions alongside level indices

        :param device: Computation device. Default is torch.device("cpu")
        :param dtype: Data type for floating point operations. Default is torch.float32
        """
        super().__init__()
        self.mesh = mesh
        self._output_factory_setter(output_mode)
        self.to(device)

    @abstractmethod
    def _compute_areas(self, batch_shape: tp.Union[torch.Size, int], device: torch.device) -> torch.Tensor:
        """Compute orientation weights for integration.

        :param batch_shape: Leading batch dimensions from intensity tensor.
        :param device: Target computation device.
        :return: Tensor of integration weights with shape broadcastable to [..., num_mesh_elements].
        """
        pass

    @abstractmethod
    def _transform_data_to_mesh_format(self, *args, **kwargs) -> torch.Tensor:
        """Map intensities onto mesh geometry.

        :param intensities: Raw intensities at mesh vertices. Shape [..., num_vertices, num_fields]
        :return: Intensities aligned with mesh simplices or discrete orientations.
        """
        pass

    @abstractmethod
    def forward(self, *args, **kwargs) ->\
            tp.Union[torch.Tensor, tp.Tuple[tp.Optional[torch.Tensor], tp.Optional[torch.Tensor], torch.Tensor]]:
        """Execute fixed-field spectral processing pipeline.

        1. Transform intensity data to mesh format
        2. Apply dimension modifiers for output mode
        3. Compute orientation weights (areas)
        4. Perform weighted orientation averaging
        5. Return spectrum in requested format

        :return: Orientation-averaged spectrum or per-orientation tuple.
        """
        pass


class BaseResProcessing(BaseProcessing):
    """Base class for spectral integration and spectral post-processing over
    orientation meshes for the resonance lines computations.

    This abstract class provides the framework for transforming resonance field data
    (fields, intensities, widths) into integrated spectra. It handles mesh-based orientation
    averaging for powder samples or single-crystal processing.

    The processing pipeline consists of:
    1. Transform resonance data to mesh format (interpolation, triangulation)
    2. Apply intensity masking based on threshold
    3. Integrate spectral contributions using the spectra integrator
    4. Apply post-processing (line broadening via convolution)
    """
    def __init__(self,
                 mesh: mesher.BaseMesh,
                 spectra_integrator: tp.Optional[BaseSpectraIntegrator] = None,
                 harmonic: int = 1,
                 post_spectra_processor: PostSpectraProcessing = PostSpectraProcessing(),
                 computational_details: ComputationalDetails = ComputationalDetails(),
                 output_mode: OutputSpectraMode = OutputSpectraMode.TOTAL,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32):
        """
        :param mesh: Mesh object defining orientation sampling grid.

        :param spectra_integrator: Integrator for computing spectra from resonance lines.
        Default is None and initialized with respect to class
        :param harmonic: Spectral harmonic (0 for absorption, 1 for first derivative). Default is 1
        :param post_spectra_processor: Processor for line broadening. Default is PostSpectraProcessing()

        :param computational_details: The details of final spectral integration and spectra processing. For example,
            -integration_natural_width : float, default=1e-6
                Minimum intrinsic linewidth added to every transition. Measures in FWHM
                Prevents division-by-zero or extreme sharpening when user-provided widths are
                very small or zero. Also it can be used as substitution for ordinary gaussian broadaning in the sample.

            - integration_gaussian_method : str, default="exp"
                Method used to evaluate the Gaussian function exp(-x²) during final integration:
                - "exp": uses exact PyTorch exponential (higher accuracy),
                - "approx": uses a fast 6th-order rational approximation (see ``gaussian_approx``).

            - chunk_size (`int`, default=128):
              Number of magnetic field points processed per integration batch.
              Larger values improve throughput but increase memory consumption.

            -for other parameters specifications, read
             docs of :class:'mars.spectra_manager.spectra_manager.ComputationalDetails'

        :param output_mode: str, OutputSpectraMode:
        Controls the organization of the computed spectrum.
        "total": returns the conventional summed spectrum over all allowed transitions (default behavior).
        "transitions": returns dict of lvl_down, lvl_up and spectrum,
        where each slice corresponds to the contribution of an individual transition
        (e.g., between specific energy levels).
        Default is "total".

        :param device: Computation device. Default is torch.device("cpu")
        :param dtype: Data type for floating point operations. Default is torch.float32
        """
        super().__init__(mesh, computational_details, output_mode, device, dtype)
        self.register_buffer("threshold", torch.tensor(
            computational_details.intensity_threshold, device=device, dtype=dtype)
        )
        self.post_spectra_processor = post_spectra_processor
        self.spectra_integrator = self._init_spectra_integrator(spectra_integrator, harmonic,
                                                                computational_details=computational_details,
                                                                device=device, dtype=dtype)
        self._output_factory_setter(output_mode)
        self.to(device)

    def _output_factory_setter(self, output_mode: OutputSpectraMode) -> None:
        """
        Set the methods for managment with respect to output mode

        :param output_mode: Controls the organization of the computed spectrum.
        :return:
        """
        if output_mode == OutputSpectraMode.TOTAL:
            self._modify_data_dimensions = self._modify_data_dimensions_total
            self._get_output = self._get_output_total
        elif output_mode == OutputSpectraMode.TRANSITIONS:
            self._modify_data_dimensions = self._modify_data_dimensions_preserve
            self._get_output = self._get_output_preserve
        else:
            raise ValueError(
                f"There are no such output method as {output_mode.value}."
                f"Use one of the {[value for value in OutputSpectraMode]}"
            )

    @abstractmethod
    def _init_spectra_integrator(self, spectra_integrator: tp.Optional[BaseSpectraIntegrator], harmonic: int,
                                 computational_details: ComputationalDetails, device: torch.device, dtype: torch.dtype):
        """
        Initialize or validate the spectra integrator used for line integration over the field axis.

        If a pre-configured integrator is provided, it may be reused or adapted;
        otherwise, a default integrator appropriate for the subclass should be created.

        :param spectra_integrator: Optional pre-defined integrator. If None, a new one is instantiated.
        :param harmonic: Spectral harmonic to compute (0 = absorption, 1 = first derivative, etc.).

        :param computational_details: Details for integrating final spectra:
              chunk_size, cutoff and so on. For more details read the dock-strings of
              :class:'mars.spectra_manager.spectra_manager.ComputationalDetails'

        :param device: Device on which the integrator should operate (e.g., CPU or CUDA).
        :param dtype: Floating-point data type for internal computations.
        :return: An instance f `BaseSpectraIntegrator`.
        """
        pass

    @abstractmethod
    def _compute_areas(self, expanded_size: torch.Tensor, device: torch.device):
        """
        Compute orientation weights (e.g., triangle areas on a sphere) for integration over the mesh.

        These weights account for the geometric contribution of each orientation sample
        and are used to average the spectrum over the powder or crystal ensemble.

        :param expanded_size: Target shape to broadcast the computed areas to.
        :param device: Device on which the area tensor should be allocated.
        :return: Tensor of integration weights with shape matching `expanded_size`.
        """
        pass

    @abstractmethod
    def _transform_data_to_mesh_format(
            self, res_fields: torch.Tensor, intensities: torch.Tensor, width: torch.Tensor) ->\
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param res_fields: the tensor of resonance fields.

        The shape is [..., num_resonance fields]
        :param intensities: the tensor of resonance fields. The shape is [..., num_resonance fields]
        :param width: the tensor of resonance fields. The shape is [..., num_resonance fields]
        :return:
        res_fields tensor with the resonance field at each triangle vertices. The shape is [..., 3] or [...]
        width tensor with the resonance field at each triangle vertices. The shape is [...]
        intensities tensor with the resonance field at each triangle vertices. The shape is [...]
        areas tensor with the resonance field at each triangle vertices. The shape is [...]
        """
        pass

    def _final_mask(self, res_fields: torch.Tensor, width: torch.Tensor,
                    intensities: torch.Tensor, areas: torch.Tensor) ->\
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply intensity-based masking to discard negligible transitions.

        Transitions are retained only if their normalized intensity exceeds
        the internal threshold. This reduces computational load
        during integration by excluding insignificant contributions.

        :param res_fields: Resonance fields at triangle vertices. Shape [..., M, 3] or [..., M]
        :param width: Linewidths associated with each transition. Shape [..., M]
        :param intensities: Transition intensities. Shape [..., M]
        :param areas: Integration weights (e.g., spherical triangle areas). Shape [..., M]
        :return: Filtered tensors (res_fields, width, intensities, areas), all with reduced last dimension
        """
        max_intensity = torch.amax(abs(intensities), dim=-1, keepdim=True)
        mask = ((intensities / max_intensity).abs() > self.threshold).any(dim=tuple(range(intensities.dim() - 1)))
        intensities = intensities[..., mask]
        width = width[..., mask]
        res_fields = res_fields[..., mask, :]
        areas = areas[..., mask]
        return res_fields, width, intensities, areas

    def _integration_precompute(self, res_fields: torch.Tensor, width: torch.Tensor,
                                intensities: torch.Tensor, areas: torch.Tensor, fields: torch.Tensor) ->\
            tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Modify resonance data to pass in into Integrator.
        """
        return res_fields, width, intensities, areas, fields

    def _modify_data_dimensions_total(
            self, res_fields: torch.Tensor, width: torch.Tensor, intensities: torch.Tensor, areas: torch.Tensor) ->\
            tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Modify data dimension to make it computable with the given type of integrator and computation method

        :param res_fields: resonance field with the shape [..., num_transitions, num_simplices, 3]
        :param width: width with the shape [..., num_transitions, num_simplices]
        :param intensities: intensities with the shape [..., num_transitions, num_simplices]
        :param areas: areas with the shape [..., num_transitions, num_simplices]
        :return: modified
         res_fields with the shape [..., num_simplices * num_transitions, 3]
         width with the shape [..., num_simplices * num_transitions]
         intensities with the shape [..., num_simplices * num_transitions]
         areas with the shape [..., num_simplices * num_transitions]
        """
        return res_fields.flatten(-3, -2), width.flatten(-2, -1), intensities.flatten(-2, -1), areas.flatten(-2, -1)

    def _modify_data_dimensions_preserve(
            self, res_fields: torch.Tensor, width: torch.Tensor, intensities: torch.Tensor, areas: torch.Tensor) ->\
            tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Modify data dimension to make it computable with the given type of integrator and computation method.
        This modifier do not flatten data into num_simplices - num_transitions dimension

        :param res_fields: resonance field with the shape [..., num_transitions, num_simplices, 3]
        :param width: width with the shape [..., num_transitions, num_simplices]
        :param intensities: intensities with the shape [..., num_transitions, num_simplices]
        :param areas: areas with the shape [..., num_transitions, num_simplices]
        :return: modified
         res_fields with the shape [..., num_simplices, num_transitions, 3]
         width with the shape [..., num_simplices, num_transitions]
         intensities with the shape [..., num_simplices, num_transitions]
         areas with the shape [..., num_simplices, num_transitions]
        """
        return res_fields, width, intensities, areas

    def _get_output_total(self, lvl_down: torch.Tensor, lvl_up: torch.Tensor, spectrum: torch.Tensor) ->\
            torch.Tensor:
        """
        Returns the final integrated spectrum as a single tensor.

        :param lvl_down: Lower energy level indices for each transition. Shape: [num_transitions]
        :param lvl_up: Upper energy level indices for each transition. Shape: [num_transitions]
        :param spectrum: Spectral contributions per transition. Shape: [..., num_transitions, N]

        :return: The single spectrum in 1D or 2D with the shpae [...., 1/2 D dimensions]
        """
        return spectrum

    def _get_output_preserve(self, lvl_down: torch.Tensor, lvl_up: torch.Tensor, spectrum: torch.Tensor) ->\
            tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns per-transition spectral contributions along with level indices.

        :param lvl_down: Lower energy level indices for each transition. Shape: [num_transitions]
        :param lvl_up: Upper energy level indices for each transition. Shape: [num_transitions]
        :param spectrum: Spectral contributions per transition. Shape: [..., num_transitions, N]

        :return: The tuple of three parameters:
        -lvl down: the index of low energy levels involved in the transition. The shape is [num_transitions]
        -lvl up: the index of high energy levels  involved in the transition. The shape is [num_transitions]
        -spectrum itself. The shape is [..., num_transitions, 1/2 D dimensions]
        """
        return lvl_down, lvl_up, spectrum

    def forward(self,
                res_fields: torch.Tensor,
                intensities: torch.Tensor,
                width: torch.Tensor,
                gauss: torch.Tensor,
                lorentz: torch.Tensor,
                fields: torch.Tensor,
                lvl_down: torch.Tensor,
                lvl_up: torch.Tensor) -> tp.Union[torch.Tensor, tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Execute the full spectral processing pipeline:

        1. Map resonance data onto mesh geometry
        2. Apply intensity-based masking
        3. Precompute integration inputs
        4. Integrate spectrum using the configured integrator
        5. Apply line broadening via PostSpectraProcessing

        :param res_fields: Resonance magnetic fields. Shape [..., num_transitions]
        :param intensities: Transition intensities. Shape [..., num_transitions]
        :param width: Inhomogeneous linewidths (Gaussian FWHM). Shape [..., num_transitions]
        :param gauss: Gaussian broadening FWHM. Scalar or batched tensor.
        :param lorentz: Lorentzian broadening FWHM. Scalar or batched tensor.
        :param fields: Field axis for output spectrum. Shape [N] or [..., N]
        :param lvl_down: Energy level indices of low spin state involved in transition. The shape is '[num_transitions]'
        :param lvl_up: Energy level indices of high spin state involved in transition. The shape is '[num_transitions]'
        :return: Broadened spectrum matching shape of `fields`
        """
        res_fields, width, intensities, areas = (
            self._transform_data_to_mesh_format(
                res_fields, intensities, width
            )
        )
        res_fields, width, intensities, areas = self._modify_data_dimensions(res_fields, width, intensities, areas)
        res_fields, width, intensities, areas = self._final_mask(res_fields, width, intensities, areas)
        res_fields, width, intensities, areas, fields = self._integration_precompute(
            res_fields, width, intensities, areas, fields
        )
        spec = self.spectra_integrator(
            res_fields, width, intensities, areas, fields
        )
        spectrum = self.post_spectra_processor(gauss, lorentz, fields, spec)
        return self._get_output(lvl_down, lvl_up, spectrum)


class PowderStationaryProcessing(BaseResProcessing):
    """Integrate stationary EPR spectra over spherical powder orientation mesh.

    This class provides the complete pipeline for transforming resonance field data
    (fields, intensities, widths) into integrated powder-averaged spectra for stationary
    (continuous-wave) EPR experiments.

    The processing pipeline consists of:
    1. Transform resonance data to mesh format (interpolation, triangulation)
    2. Apply intensity masking based on threshold
    3. Integrate spectral contributions using the spectra integrator
    4. Apply post-processing (line broadening via convolution)
    """
    def __init__(self,
                 mesh: mesher.BaseMeshPowder,
                 spectra_integrator: tp.Optional[BaseSpectraIntegrator] = None,
                 harmonic: int = 1,
                 post_spectra_processor: PostSpectraProcessing = PostSpectraProcessing(),
                 computational_details: ComputationalDetails = ComputationalDetails,
                 output_mode: OutputSpectraMode = OutputSpectraMode.TOTAL,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32
                 ):
        """
        :param mesh: Powder mesh object (BaseMeshPowder) defining spherical grid.

        :param spectra_integrator: Custom integrator. Default is None (auto-initialized based on mesh parameters)
        :param harmonic: Spectral harmonic (0 for absorption, 1 for first derivative). Default is 1
        :param post_spectra_processor: Processor for line broadening. Default is PostSpectraProcessing()

        :param computational_details: The details of final spectral integration and spectra processing. For example,
            -integration_natural_width : float, default=1e-6
                Minimum intrinsic linewidth added to every transition. Measures in FWHM
                Prevents division-by-zero or extreme sharpening when user-provided widths are
                very small or zero. Also it can be used as substitution for ordinary gaussian broadaning in the sample.

            - integration_gaussian_method : str, default="exp"
                Method used to evaluate the Gaussian function exp(-x²) during final integration:
                - "exp": uses exact PyTorch exponential (higher accuracy),
                - "approx": uses a fast 6th-order rational approximation (see ``gaussian_approx``).

            - chunk_size (`int`, default=128):
              Number of magnetic field points processed per integration batch.
              Larger values improve throughput but increase memory consumption.

            -for other parameters specifications, read
             docs of :class:'mars.spectra_manager.spectra_manager.ComputationalDetails'

        :param output_mode: str, OutputSpectraMode:
        Controls the organization of the computed spectrum.
        "total": returns the conventional summed spectrum over all allowed transitions (default behavior).
        "transitions": returns dict of lvl_down, lvl_up and spectrum,
        where each slice corresponds to the contribution of an individual transition
        (e.g., between specific energy levels).
        Default is "total".

        :param device: Computation device. Default is torch.device("cpu")
        :param dtype: Data type for floating point operations. Default is torch.float32
        """
        super().__init__(mesh, spectra_integrator, harmonic, post_spectra_processor,
                         computational_details=computational_details,
                         output_mode=output_mode, device=device, dtype=dtype)

    def _init_spectra_integrator(self, spectra_integrator: tp.Optional[BaseSpectraIntegrator],
                                 harmonic: int, computational_details: ComputationalDetails,
                                 device: torch.device, dtype: torch.dtype)\
            -> BaseSpectraIntegrator:
        """Initialize the appropriate spectra integrator based on mesh
        symmetry.

        Uses AxialSpectraIntegrator for axial powder meshes;
        otherwise uses general SphereSpectraIntegrator.

        :param spectra_integrator: Optional pre-defined integrator
        :param harmonic: Spectral harmonic (0 = absorption, 1 = first
            derivative)

        :param computational_details: Details for integrating final spectra:
              chunk_size, cutoff and so on. For more details read the dock-strings of
              :class:'mars.spectra_manager.spectra_manager.ComputationalDetails'

        :param device: Computation device
        :param dtype: Floating-point precision
        :return: Instantiated integrator object
        """
        if spectra_integrator is None:
            if self.mesh.axial:
                return AxialSpectraIntegrator(harmonic,
                                              gaussian_method=computational_details.integration_gaussian_method,
                                              chunk_size=computational_details.integration_chunk_size,
                                              natural_width=computational_details.integration_natural_width,
                                              integration_level=computational_details.integration_level,
                                              clamp_width_factor=computational_details.integration_clamp_width_factor,
                                              computation_method=computational_details.integration_computation_method,
                                              field_factor=computational_details.field_factor,
                                              device=device, dtype=dtype)
            return SphereSpectraIntegrator(
                harmonic,
                gaussian_method=computational_details.integration_gaussian_method,
                chunk_size=computational_details.integration_chunk_size,
                natural_width=computational_details.integration_natural_width,
                integration_level=computational_details.integration_level,
                clamp_width_factor=computational_details.integration_clamp_width_factor,
                computation_method=computational_details.integration_computation_method,
                field_factor=computational_details.field_factor,
                device=device, dtype=dtype)
        return spectra_integrator

    def _compute_areas(self, expanded_size: int, device: torch.device) -> torch.Tensor:
        """Compute spherical triangle areas from the powder mesh and expand.

        to match batch dimensions required for integration.

        :param expanded_size: Leading batch size before flattening
            (e.g., number of orientations)
        :param device: Target computation device
        :return: Flattened area tensor of shape [expanded_size *
            num_triangles]
        """
        grid, simplices = self.mesh.post_mesh
        areas = self.mesh.spherical_triangle_areas(grid, simplices)
        areas = areas.reshape(1, -1).expand(expanded_size, -1)
        return areas

    def _process_tensor(self, data_tensor: torch.Tensor) -> torch.Tensor:
        """Interpolate input resonance data (fields, intensities, widths) onto.

        the Delaunay triangulation defined by the powder mesh.

        :param data_tensor: Input tensor of shape [..., num_orientations, num_transitions]
        :return: Remapped tensor aligned with mesh simplices
        """
        _, simplices = self.mesh.post_mesh
        processed = self.mesh(data_tensor.transpose(-1, -2))
        return self.mesh.to_delaunay(processed, simplices)

    def _compute_batched_tensors(self, *args) -> torch.Tensor:
        """Stack multiple resonance-related tensors (e.g., fields, intensities,
        widths),.

        then remap them jointly onto the orientation mesh using `_process_tensor`.

        :param args: Tensors of identical shape [..., num_orientations, num_transitions]
        :return: Batched and mesh-aligned tensor of shape [..., 3, num_simplices, num_transitions]
        """
        batched_matrix = torch.stack(args, dim=-3)
        batched_matrix = self._process_tensor(batched_matrix)
        return batched_matrix

    def _transform_data_to_mesh_format(
            self, res_fields: torch.Tensor, intensities: torch.Tensor, width: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param res_fields: the tensor of resonance fields.
        The shape is [..., num_transitions]

        :param intensities: the tensor of resonance fields. The shape is [time, ..., num_transitions]
        :param width: the tensor of resonance fields. The shape is [..., num_transitions]
        :return:
        res_fields tensor with the resonance field at each triangle vertices.
        The shape is [..., num_transitions, num_simplices or orientations, 3]

        width tensor with the resonance field at each triangle vertices.
        The shape is [..., num_transitions, num_simplices or orientations]

        intensities tensor with the resonance field at each triangle vertices.
        The shape is [..., num_transitions, num_simplices or orientations]

        areas tensor with the resonance field at each triangle vertices.
        The shape is [..., num_transitions, num_simplices or orientations]
        """
        batched_matrix = self._compute_batched_tensors(res_fields, intensities, width)
        expanded_size = batched_matrix.shape[-3]
        res_fields, intensities, width = torch.unbind(batched_matrix, dim=-4)
        width = width.mean(dim=-1)
        intensities = intensities.mean(dim=-1)
        areas = self._compute_areas(expanded_size, device=res_fields.device)
        return res_fields, width, intensities, areas


class CrystalStationaryProcessing(BaseResProcessing):
    """Integrate stationary spectra for single-crystal or many-crystal oriented
    sample.

    This class provides the pipeline for transforming resonance field data into spectra
    for single-crystal samples or specific crystal orientations where no orientation
    averaging is required.

    The processing pipeline consists of:
    1. Transform resonance data to mesh format (interpolation, triangulation)
    2. Apply intensity masking based on threshold
    3. Integrate spectral contributions using mean contribution of each given orientation
    4. Apply post-processing (line broadening via convolution)
    """
    def __init__(self,
                 mesh: mesher.CrystalMesh,
                 spectra_integrator: tp.Optional[BaseSpectraIntegrator] = None,
                 harmonic: int = 1,
                 post_spectra_processor: PostSpectraProcessing = PostSpectraProcessing(),
                 computational_details: ComputationalDetails = ComputationalDetails(),
                 output_mode: OutputSpectraMode = OutputSpectraMode.TOTAL,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32):
        """
        :param mesh: Crystal mesh object defining single or discrete orientations.

        :param spectra_integrator: Custom integrator. Default is None (MeanIntegrator initialized)
        :param harmonic: Spectral harmonic (0 for absorption, 1 for first derivative). Default is 1
        :param post_spectra_processor: Processor for line broadening. Default is PostSpectraProcessing()

        :param computational_details: The details of final spectral integration and spectra processing. For example,
            -integration_natural_width : float, default=1e-6
                Minimum intrinsic linewidth added to every transition. Measures in FWHM
                Prevents division-by-zero or extreme sharpening when user-provided widths are
                very small or zero. Also it can be used as substitution for ordinary gaussian broadaning in the sample.

            - integration_gaussian_method : str, default="exp"
                Method used to evaluate the Gaussian function exp(-x²) during final integration:
                - "exp": uses exact PyTorch exponential (higher accuracy),
                - "approx": uses a fast 6th-order rational approximation (see ``gaussian_approx``).

            - chunk_size (`int`, default=128):
              Number of magnetic field points processed per integration batch.
              Larger values improve throughput but increase memory consumption.

            -for other parameters specifications, read
             docs of :class:'mars.spectra_manager.spectra_manager.ComputationalDetails'

        :param output_mode: str, OutputSpectraMode:
        Controls the organization of the computed spectrum.
        "total": returns the conventional summed spectrum over all allowed transitions (default behavior).
        "transitions": returns dict of lvl_down, lvl_up and spectrum,
        where each slice corresponds to the contribution of an individual transition
        (e.g., between specific energy levels).
        Default is "total".

        :param device: Computation device. Default is torch.device("cpu")
        :param dtype: Data type for floating point operations. Default is torch.float32
        """
        super().__init__(mesh, spectra_integrator, harmonic, post_spectra_processor,
                         computational_details=computational_details,
                         output_mode=output_mode, device=device, dtype=dtype)

    def _init_spectra_integrator(self, spectra_integrator: tp.Optional[BaseSpectraIntegrator], harmonic: int,
                                 computational_details: ComputationalDetails,
                                 device: torch.device, dtype: torch.dtype):
        """Initialize the appropriate spectra integrator based on mesh
        symmetry.

        :param spectra_integrator: Optional pre-defined integrator
        :param harmonic: Spectral harmonic (0 = absorption, 1 = first
            derivative)

        :param computational_details: Details for integrating final spectra:
              chunk_size, cutoff and so on. For more details read the dock-strings of
              :class:'mars.spectra_manager.spectra_manager.ComputationalDetails'

        :param device: Computation device
        :param dtype: Floating-point precision
        :return: Instantiated integrator object
        """
        if spectra_integrator is None:
            return MeanIntegrator(harmonic=harmonic,
                                  gaussian_method=computational_details.integration_gaussian_method,
                                  chunk_size=computational_details.integration_chunk_size,
                                  integration_level=computational_details.integration_level,
                                  natural_width=computational_details.integration_natural_width,
                                  clamp_width_factor=computational_details.integration_clamp_width_factor,
                                  computation_method=computational_details.integration_computation_method,
                                  field_factor=computational_details.field_factor,
                                  device=device)
        else:
            return spectra_integrator

    def _compute_areas(self, expanded_size: torch.Size, device: torch.device):
        areas = torch.ones(expanded_size, dtype=torch.float32, device=device)
        return areas

    def _transform_data_to_mesh_format(
            self, res_fields: torch.Tensor, intensities: torch.Tensor, width: torch.Tensor) ->\
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param res_fields: the tensor of resonance fields.
        The shape is [..., num_transitions]

        :param intensities: the tensor of resonance fields. The shape is [..., num_transitions]
        :param width: the tensor of resonance fields. The shape is [..., num_transitions]
        :return:

        res_fields tensor with the resonance field at each triangle vertices.
        The shape is [..., num_transitions, num_simplices or orientations, 1]

        width tensor with the resonance field at each triangle vertices.
        The shape is [..., num_transitions, num_simplices or orientations]

        intensities tensor with the resonance field at each triangle vertices.
        The shape is [..., num_transitions, num_simplices or orientations]

        areas tensor with the resonance field at each triangle vertices.
        The shape is [..., num_simplices or orientations]
        """
        expanded_size = res_fields.shape
        res_fields = res_fields.unsqueeze(-1)
        areas = self._compute_areas(expanded_size, res_fields.device)
        return res_fields, width, intensities, areas


class PowderTimeProcessing(PowderStationaryProcessing):
    """Integrate time-resolved EPR spectra over spherical powder orientation
    mesh.

    This class extends PowderStationaryProcessing to handle time-dependent intensities
    while keeping resonance fields and widths time-independent

    The processing pipeline consists of:
    1. Transform resonance data to mesh format (interpolation, triangulation)
    2. Apply intensity masking based on threshold
    3. Integrate spectral contributions.
    4. Apply post-processing (line broadening via convolution)
    """

    def _modify_data_dimensions_preserve(
            self, res_fields: torch.Tensor, width: torch.Tensor, intensities: torch.Tensor, areas: torch.Tensor) ->\
            tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Modify data dimension to make it computable with the given type of integrator and computation method.
        This modifier do not flatten data into num_simplices - num_transitions dimension

        :param res_fields: resonance field with the shape [..., num_transitions, num_simplices, 3]
        :param width: width with the shape [..., num_transitions, num_simplices, 3]
        :param intensities: intensities with the shape [..., time, num_transitions, num_simplices, 3]
        :param areas: areas with the shape [..., num_transitions, num_simplices]
        :return: modified
         res_fields with the shape [..., num_simplices, num_transitions, 3]
         width with the shape [..., num_simplices, num_transitions]
         intensities with the shape [..., num_simplices, time, num_transitions]
         areas with the shape [..., num_simplices, num_transitions]
        """
        return res_fields, width, intensities.transpose(-3, -2), areas

    def _integration_precompute(self, res_fields: torch.Tensor, width: torch.Tensor,
                                intensities: torch.Tensor, areas: torch.Tensor, fields: torch.Tensor) ->\
            tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return res_fields.unsqueeze(-3),\
            width.unsqueeze(-2), intensities, areas.unsqueeze(-2), fields.unsqueeze(-2)

    def _transform_data_to_mesh_format(self, res_fields: torch.Tensor,
                                       intensities: torch.Tensor,
                                       width: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param res_fields: the tensor of resonance fields.
        The shape is [..., num_transitions]

        :param intensities: the tensor of resonance fields. The shape is [time, ..., num_transitions]
        :param width: the tensor of resonance fields. The shape is [..., num_transitions]
        :return:
        res_fields tensor with the resonance field at each triangle vertices.
        The shape is [..., num_transitions, num_simplices or orientations, 3]

        width tensor with the resonance field at each triangle vertices.
        The shape is [..., num_transitions, num_simplices or orientations]

        intensities tensor with the resonance field at each triangle vertices.
        The shape is [time, ...,time, num_transitions, num_simplices or orientations]

        areas tensor with the resonance field at each triangle vertices.
        The shape is [..., num_transitions, num_simplices or orientations]
        """
        batched_matrix = self._compute_batched_tensors(res_fields, width)
        expanded_size = batched_matrix.shape[-3]
        intensities = self._process_tensor(intensities)

        res_fields, width = torch.unbind(batched_matrix, dim=-4)
        width = width.mean(dim=-1)
        intensities = intensities.mean(dim=-1)
        areas = self._compute_areas(expanded_size, device=res_fields.device)
        return res_fields, width, intensities, areas


class CrystalTimeProcessing(CrystalStationaryProcessing):
    """Integrate time-resolved EPR spectra over single-crystal or many-crystal
    sample.

    This class extends PowderStationaryProcessing to handle time-dependent intensities
    while keeping resonance fields and widths time-independent

    The processing pipeline consists of:
    1. Transform resonance data to mesh format (interpolation, triangulation)
    2. Apply intensity masking based on threshold
    3. Integrate spectral contributions.
    4. Apply post-processing (line broadening via convolution)
    """
    def _integration_precompute(self, res_fields: torch.Tensor, width: torch.Tensor,
                                intensities: torch.Tensor, areas: torch.Tensor, fields: torch.Tensor) ->\
            tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return res_fields.unsqueeze(-3),\
            width.unsqueeze(-2), intensities, areas.unsqueeze(-2), fields.unsqueeze(-2)


class Broadener(nn.Module):
    """Compute inhomogeneous linewidths from spin Hamiltonian strain tensors.

    Evaluates field-dependent and field-independent contributions to
    transition width using perturbation theory on strained Hamiltonian
    components. Output is FWHM of Gaussian profile.

    Input components of sample are given as FWHM (Full width at half maximum) of corresponding distributions
    """
    def __init__(self, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.to(device)

    def _compute_element_field_free(self, vector: torch.Tensor,
                          tensor_components_A: torch.Tensor, tensor_components_B: torch.Tensor,
                          transformation_matrix: torch.Tensor, correlation_matrix: torch.Tensor) -> torch.Tensor:
        return torch.einsum(
            "...pij,jkl,ikl,...bk,...bl,ph->...hb",
            transformation_matrix, tensor_components_A, tensor_components_B, torch.conj(vector), vector,
            correlation_matrix
        ).real

    def _compute_element_field_dep(self, vector: torch.Tensor,
                          tensor_components: torch.Tensor,
                          transformation_matrix: torch.Tensor, correlation_matrix: torch.Tensor) -> torch.Tensor:
        return torch.einsum(
            "...pi, ikl,...bk,...bl,ph->...hb",
            transformation_matrix, tensor_components, torch.conj(vector), vector, correlation_matrix
        ).real

    def _compute_field_strain_square(self, strained_data: tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                                     vector_down: torch.Tensor, vector_up: torch.Tensor,
                                     B_trans: torch.Tensor) -> torch.Tensor:
        correlation_matrix, tensor_components, transformation_matrix = strained_data
        return (B_trans.unsqueeze(-2) * (
                self._compute_element_field_dep(vector_up, tensor_components, transformation_matrix,
                                                correlation_matrix) -
                self._compute_element_field_dep(vector_down, tensor_components, transformation_matrix,
                                                correlation_matrix)
        )).square().sum(dim=-2)

    def _compute_field_free_strain_square(self,
                                          strained_data: tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                                          vector_down: torch.Tensor, vector_up: torch.Tensor)\
            -> torch.Tensor:
        correlation_matrix, tensor_components_A, tensor_components_B, transformation_matrix = strained_data
        return (
                self._compute_element_field_free(
                    vector_up, tensor_components_A, tensor_components_B, transformation_matrix, correlation_matrix
                ) -
                self._compute_element_field_free(
                    vector_down, tensor_components_A, tensor_components_B, transformation_matrix, correlation_matrix
                )
        ).square().sum(dim=-2)

    def add_hamiltonian_strain(self, sample: spin_model.MultiOrientedSample, squared_width: torch.Tensor) ->\
            torch.Tensor:
        """Adds residual broadening due to unresolved interactions.

        :param sample: The MultiOrientedSample object
        :param squared_width: The square of gaussian broadening
        :return: Total gaussian broadening as
        """
        hamiltonian_width = sample.build_ham_strain().unsqueeze(-1).square()
        return (squared_width + hamiltonian_width).sqrt()

    def forward(self, sample: spin_model.MultiOrientedSample,
                vector_down: torch.Tensor, vector_up: torch.Tensor,
                B_trans: torch.Tensor) -> torch.Tensor:
        """Compute total Gaussian linewidth (FWHM) for each transition by
        combining:

        - Field-dependent strain contributions (from g-, D-tensor distributions)
        - Field-independent zero-field strain terms
        - Residual Hamiltonian strain (e.g., unresolved hyperfine)
        Result is returned as FWHM

        :param sample: The MultiOrientedSample object
        :param vector_down: Lower-state eigenvector. Shape [..., N]
        :param vector_up: Upper-state eigenvector. Shape [..., N]
        :param B_trans: Magnetic fields of transitions
        :return: Return Total gaussian broadening due to the
            1) Unresolved interactions
            2) Hamiltonian parameters distribution
        """
        target_shape = vector_down.shape[:-1]
        result = torch.zeros(target_shape, dtype=B_trans.dtype, device=vector_down.device)

        for strained_data in sample.build_field_dep_strain():
            result += self._compute_field_strain_square(strained_data, vector_down, vector_up, B_trans)

        for strained_data in sample.build_zero_field_strain():
            result += self._compute_field_free_strain_square(strained_data, vector_down, vector_up)

        return self.add_hamiltonian_strain(sample, result)


class BaseIntensityCalculator(nn.Module, ABC):
    """Base class for all EPR intensity calculators.
    Provides shared infrastructure for spin system initialization, population model
    dispatch, device/dtype management, and powder/crystal magnetization routing.
    """
    def __init__(self,
                 spin_system_dim: int | list[int],
                 temperature: tp.Optional[float] = None,
                 populator: tp.Optional[tp.Union[BasePopulator, str]] = None,
                 context: tp.Optional[contexts.BaseContext] = None,
                 disordered: bool = True,
                 computational_details: ComputationalDetails = ComputationalDetails(),
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32):
        """
        :param spin_system_dim: Dimension of spin system Hilbert space.

        :param temperature: Temperature in Kelvin of a sample.
        :param populator: BasePopulator object. Default is None
        (auto-initialized based on specific calculator).
        Also, can be set as string object for some cases (for example, for density computations)

        :param context: Relaxation/population context defining relaxation and initial population. Default is None
        :param disordered: If True, use powder averaging; if False, use crystal geometry. Default is True
        :param computational_details: ComputationalDetails
            computational_details : ComputationalDetails, optional
            Configuration object that governs the numerical aspects of spectra generation.
            In this class it is used for the getting values of time-evolution equations solving

        :param device: Computation device. Default is torch.device("cpu")
        :param dtype: Data type for floating point operations. Default is torch.float32
        """
        super().__init__()
        self.populator = self._init_populator(
            temperature, populator, context, disordered, computational_details, device, dtype
        )
        self.spin_system_dim = spin_system_dim
        self.temperature = temperature
        self._compute_magnitization =\
            self._compute_magnitization_powder if disordered else self._compute_magnitization_crystal
        self.to(device)

    @abstractmethod
    def _init_populator(self,
                        temperature: tp.Optional[float],
                        populator: tp.Optional[tp.Union[BasePopulator, str]],
                        context: tp.Optional[contexts.BaseContext],
                        disordered: bool,
                        computational_details: ComputationalDetails,
                        device: torch.device,
                        dtype: torch.dtype) -> BasePopulator:
        """Initialize the population calculator for the given experiment type.

        :param temperature: Sample temperature in Kelvin.
        :param populator: Optional custom population function or identifier.
        :param context: Relaxation/population dynamics context.
        :param disordered: True for powder averaging, False for single-crystal.
        :param computational_details: Configuration object for numerical tolerances.
        :param device: Computation device.
        :param dtype: Floating-point type.
        :return: Initialized BasePopulator instance.
        """
        pass

    @abstractmethod
    def _compute_magnitization_powder\
                    (self, *args, **kwargs) -> torch.Tensor:
        """Compute powder-averaged magnetization.
        :return: Magnetization tensor. Shape [...]
        """
        pass

    @abstractmethod
    def _compute_magnitization_crystal\
                    (self, *args, **kwargs) -> torch.Tensor:
        """Compute crystal-geometry magnetization.
        :return: Magnetization tensor. Shape [...]
        """
        pass

    @abstractmethod
    def compute_intensity(self, *args, **kwargs) -> torch.Tensor:
        """Compute transition intensity based on magnetization and population.

        :param args: Experiment-specific positional arguments.
        :param kwargs: Experiment-specific keyword arguments.
        :return: Intensity tensor. Shape [...]
        """
        pass

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass alias for compute_intensity().

        :param args: Positional arguments forwarded to compute_intensity.
        :param kwargs: Keyword arguments forwarded to compute_intensity.
        :return: Intensity tensor. Shape [...]
        """
        return self.compute_intensity(*args, **kwargs)


class BaseResIntensityCalculator(BaseIntensityCalculator):
    """Base class for computing EPR transition intensities.

    Handles calculation of transition intensities based on:
    - Transition matrix elements (magnetization)
    - Level populations (thermal, time-dependent, or custom)
    """
    def _init_populator(self,  temperature: tp.Optional[float],
                        populator: tp.Optional[tp.Union[BasePopulator, str]],
                        context: tp.Optional[contexts.BaseContext], disordered: bool,
                        computational_details: ComputationalDetails,
                        device: torch.device, dtype: torch.dtype) -> BasePopulator:
        """
        :param temperature: Sample temperature in Kelvin.

        :param populator: Optional custom population function
        :param context: Relaxation/population dynamics context
        :param disordered: True for powder averaging, False for single-crystal
        :param computational_details: ComputationalDetails
            computational_details : ComputationalDetails, optional
            Configuration object that governs the numerical aspects of spectra generation.
            In this class it is used for the getting values of time-evolution equations solving
        :param device: Computation device
        :param dtype: Floating-point type
        :return: BasePopulator object
        """
        return populator

    def _compute_magnitization_powder(self, Gx: torch.Tensor, Gy: torch.Tensor, Gz: torch.Tensor,
                                      res_manifold: torch.Tensor,
                                      vector_down: torch.Tensor, vector_up: torch.Tensor,):
        """Compute powder-averaged transition intensity.

        :param Gx, Gy, Gz: Cartesian components of Zeeman operator. Shape [..., N, N]

        :param res_manifold: Resonance fields or frequencies. Shape [...]

        :param vector_down: Lower-state eigenvector. Shape [..., N]
        :param vector_up: Upper-state eigenvector. Shape [..., N]
        :return: Intensity proportional to |<up|Gx|down>|² + |<up|Gy|down>|², in (J·s/μ_B)²
        """
        magnitization = compute_matrix_element(vector_down, vector_up, Gx).square().abs() +\
                        compute_matrix_element(vector_down, vector_up, Gy).square().abs()
        return magnitization * (constants.PLANCK / constants.BOHR) ** 2

    def _compute_magnitization_crystal(self, Gx: torch.Tensor, Gy: torch.Tensor, Gz: torch.Tensor,
                                       res_manifold: torch.Tensor,
                                       vector_down: torch.Tensor, vector_up: torch.Tensor,):
        """Compute crystal transition intensity.

        The orientation of the wave magnetic field is along the x-axis.
        :param Gx, Gy, Gz: Cartesian components of Zeeman operator. Shape [..., N, N]
        :param res_manifold: Resonance fields or frequencies. Shape [...]
        :param vector_down: Lower-state eigenvector. Shape [..., N]
        :param vector_up: Upper-state eigenvector. Shape [..., N]
        :return: Intensity proportional to |<up|Gx|down>|² + |<up|Gy|down>|², in (J·s/μ_B)²
        """
        magnitization = compute_matrix_element(vector_down, vector_up, Gx).square().abs()
        return magnitization * (constants.PLANCK / constants.BOHR) ** 2

    @abstractmethod
    def compute_intensity(self, Gx: torch.Tensor, Gy: torch.Tensor, Gz: torch.Tensor,
                          res_manifold: torch.Tensor,
                          lvl_down: torch.Tensor, lvl_up: torch.Tensor, resonance_energies: torch.Tensor,
                          vector_down: torch.Tensor, vector_up: torch.Tensor,
                          full_system_vectors: tp.Optional[torch.Tensor], *args, **kwargs):
        """Compute intensity of transitions.

        :param Gx, Gy, Gz: Zeeman operator components :param
        vector_down, vector_up: Transition eigenvectors :param lvl_down,
        lvl_up: Energy level indices
        :param res_manifold: Resonance fields or frequencies. Shape [...]
        :param lvl_down, lvl_up: Energy level indices involved in transition
        :param resonance_energies: Eigenvalues of spin Hamiltonian. Shape [..., N]
        :param full_system_vectors: Optional full eigenbasis
        :return: Transition intensities
        """
        raise NotImplementedError

    def forward(self, Gx: torch.Tensor, Gy: torch.Tensor, Gz: torch.Tensor,
                res_manifold: torch.Tensor,
                lvl_down: torch.Tensor, lvl_up: torch.Tensor, resonance_energies: torch.Tensor,
                vector_down: torch.Tensor, vector_up: torch.Tensor,
                full_system_vectors: tp.Optional[torch.Tensor], *args, **kwargs) -> torch.Tensor:
        """
        :param Gx, Gy, Gz: Zeeman operator components.
        :param res_manifold: Resonance fields or frequencies. Shape [...]
        :param lvl_down, lvl_up: Energy level indices involved in transition
        :param resonance_energies: Eigenvalues of spin Hamiltonian. Shape [..., N]
        :param vector_down, vector_up: Transition eigenvectors :param
        lvl_down, lvl_up: Energy level indices
        :param full_system_vectors: Optional full eigenbasis
        :return: Transition intensities
        """
        return self.compute_intensity(Gx, Gy, Gz, res_manifold, lvl_down, lvl_up, resonance_energies,
                                      vector_down, vector_up, full_system_vectors)


class StationaryIntensityCalculator(BaseResIntensityCalculator):
    """Calculate transition intensities for stationary (CW) EPR experiments.

    Handles calculation of transition intensities based on:
    - Transition matrix elements (magnetization)
    - Level populations. Uses Boltzmann thermal populations at specified temperature
      or predefined population given in context.
    """
    def __init__(self, spin_system_dim: int, temperature: tp.Optional[float],
                 populator: tp.Optional[BasePopulator] = None,
                 context: tp.Optional[contexts.BaseContext] = None,
                 disordered: bool = True, computational_details: ComputationalDetails = ComputationalDetails(),
                 device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32):
        """
        :param spin_system_dim: Dimension of spin system Hilbert space.

        :param temperature: Temperature in Kelvin of a sample.
        :param populator: BasePopulator object. Default is None
        (auto-initialized based as stationary populator)
        :param context: Relaxation/population context defining relaxation and initial population. Default is None
        :param disordered: If True, use powder averaging; if False, use crystal geometry. Default is True
        :param computational_details: ComputationalDetails
            computational_details : ComputationalDetails, optional
            Configuration object that governs the numerical aspects of spectra generation.
            In this class it is used for the getting values of time-evolution equations solving
        :param device: Computation device. Default is torch.device("cpu")
        :param dtype: Data type for floating point operations. Default is torch.float32
        """
        super().__init__(spin_system_dim, temperature, populator, context, disordered,
                         computational_details, device=device, dtype=dtype)

    def _init_populator(self,
                        temperature: tp.Optional[float], populator: tp.Optional[BasePopulator],
                        context: tp.Optional[contexts.BaseContext],
                        disordered: bool, computational_details: ComputationalDetails,
                        device: torch.device, dtype: torch.dtype) -> BasePopulator:
        """
        :param temperature: Sample temperature in Kelvin.

        :param populator: Optional population computation instance of BasePopulator
        :param context: Relaxation/population dynamics context
        :param disordered: True for powder averaging, False for single-crystal
        :param computational_details: ComputationalDetails
            computational_details : ComputationalDetails, optional
            Configuration object that governs the numerical aspects of spectra generation.
            In this class it is used for the getting values of time-evolution equations solving
        :param device: Computation device
        :param dtype: Floating-point type
        :return: BasePopulator object
        """
        if populator is None:
            return StationaryPopulator(context=context, init_temperature=temperature, device=device, dtype=dtype)
        else:
            return populator

    def compute_intensity(self,
                          Gx: torch.Tensor, Gy: torch.Tensor, Gz: torch.Tensor,
                          res_manifold: torch.Tensor,
                          lvl_down: torch.Tensor, lvl_up: torch.Tensor, resonance_energies: torch.Tensor,
                          vector_down: torch.Tensor, vector_up: torch.Tensor,
                          full_system_vectors: tp.Optional[torch.Tensor], *args, **kwargs):
        """
        Compute CW-EPR transition intensities as the product of:

        :param Gx, Gy, Gz: Zeeman operator components. Shape [..., N, N]
        :param res_manifold: Resonance fields or frequencies. Shape [...]
        :param lvl_down, lvl_up: Energy level indices involved in transition
        :param resonance_energies: Eigenvalues of spin Hamiltonian. Shape [..., N]

        :param vector_down: Lower-state eigenvector. Shape [..., N]
        :param vector_up: Upper-state eigenvector. Shape [..., N]
        :param full_system_vectors: Full eigenbasis (optional, for advanced population models)
        :return: Intensity tensor matching transition dimension [...]
        """
        intensity = self.populator(
            res_manifold, lvl_down, lvl_up, resonance_energies, full_system_vectors, *args, **kwargs) * (
                self._compute_magnitization(Gx, Gy, Gz, res_manifold, vector_down, vector_up)
        )
        return intensity


class TimeIntensityCalculator(BaseResIntensityCalculator):
    """Calculate time-dependent transition intensities for time-resolved EPR
    experiments based on relxation of.

    populations.

    Handles calculation of transition intensities based on:
    - Transition matrix elements (magnetization)
    - Level populations. Uses relaxation parameters and initial populations given in context
    """
    def __init__(self, spin_system_dim: int, temperature: tp.Optional[float],
                 populator: tp.Optional[tp.Union[BaseTimeDepPopulator, str]],
                 context: tp.Optional[contexts.BaseContext],
                 disordered: bool = True,
                 computational_details: ComputationalDetails = ComputationalDetails,
                 device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32,
                 ):
        """
        :param spin_system_dim: Dimension of spin system Hilbert space.

        :param temperature: Temperature in Kelvin of a sample.
        :param populator:
            Specifies the population calculator to use.
            If None (default), a LevelBasedPopulator or RWADensityPopulator  is automatically initialized
            depending on class.
            Alternatively, a string may be provided to select a density-based method:
            - rwa - uses the rotating-wave approximation
            - propagator - uses full time-propagator dynamics

        :param context: Relaxation/population context defining relaxation and initial population.
        :param disordered: If True, use powder averaging; if False, use crystal geometry. Default is True
        :param computational_details: ComputationalDetails
            computational_details : ComputationalDetails, optional
            Configuration object that governs the numerical aspects of spectra generation.
            In this class it is used for the getting values of time-evolution equations solving
        :param device: Computation device. Default is torch.device("cpu")
        :param dtype: Data type for floating point operations. Default is torch.float32
        """
        super().__init__(
            spin_system_dim, temperature, populator, context, disordered, computational_details,
            device=device, dtype=dtype
        )

    def _init_populator(self,
                        temperature: tp.Optional[float], populator: tp.Optional[tp.Union[BaseTimeDepPopulator, str]],
                        context: tp.Optional[contexts.BaseContext],
                        disordered: bool, computational_details: ComputationalDetails,
                        device: torch.device, dtype: torch.dtype) -> BaseTimeDepPopulator:
        """
        :param temperature: Sample temperature in Kelvin.

        :param populator: Optional BaseTimeDepPopulator object
        :param context: Relaxation/population dynamics context
        :param disordered: True for powder averaging, False for single-crystal
        :param computational_details: ComputationalDetails
            computational_details : ComputationalDetails, optional
            Configuration object that governs the numerical aspects of spectra generation.
            In this class it is used for the getting values of time-evolution equations solving
        :param device: Computation device
        :param dtype: Floating-point type
        :return: BasePopulator object
        """
        if populator is None:
            return LevelBasedPopulator(context=context, init_temperature=temperature, device=device, dtype=dtype)
        else:
            return populator

    def compute_intensity(self, Gx: torch.Tensor, Gy: torch.Tensor, Gz: torch.Tensor,
                          res_manifold: torch.Tensor, lvl_down: torch.Tensor,
                          lvl_up: torch.Tensor, resonance_energies: torch.Tensor,
                          vector_down: torch.Tensor, vector_up: torch.Tensor,
                          full_system_vectors: tp.Optional[torch.Tensor],
                          *args, **kwargs):
        """Compute time-resolved EPR intensities based solely on transition
        matrix elements.

        Population dynamics are handled separately via `compute_population`.
        This method returns the "geometric" part of intensity `(|<ψ_up|G|ψ_down>|²)`.

        :param Gx, Gy, Gz: Zeeman operator components

        :param lvl_down:
            Energy levels of lower states from which transitions occur.
            Shape: [..., N], where
            N is the number of energy levels.

        :param lvl_up:
            Energy levels of upper states to which transitions occur.
            Shape: [..., N], where
            N is the number of energy levels.

        :param vector_down, vector_up: Eigenvectors of lower/upper states
        :param ...: Other parameters (unused here but kept for interface consistency)
        :return: Magnetization-squared term, shape [...]
        """
        intensity = (
                self._compute_magnitization(Gx, Gy, Gz, res_manifold, vector_down, vector_up)
        )
        return intensity

    def compute_population(self, time: torch.Tensor,
                           res_fields: torch.Tensor, lvl_down: torch.Tensor, lvl_up: torch.Tensor,
                           resonance_energies: torch.Tensor,
                           vector_down: torch.Tensor, vector_up: torch.Tensor,
                           full_system_vectors: tp.Optional[torch.Tensor],
                           *args, **kwargs) -> torch.Tensor:
        """
        Compute the time-dependent population differences for the resonant EPR transitions.

        This method delegates the calculation to the time-dependent populator,
        which solves the kinetic or Liouville-von Neumann equations to model the
        relaxation dynamics of the spin system over the specified time points.

        :param time: Time points at which to evaluate the populations, shape [T].
        :param res_fields: Resonance magnetic fields for each transition, shape [..., M].
        :param lvl_down: Indices of the lower energy levels involved in transitions, shape [M].
        :param lvl_up: Indices of the upper energy levels involved in transitions, shape [M].
        :param resonance_energies: Eigenenergies of all spin states, shape [..., M, N].
        :param vector_down: Eigenvectors of the lower energy states, shape [..., M, N].
        :param vector_up: Eigenvectors of the upper energy states, shape [..., M, N].
        :param full_system_vectors: Eigenvectors of the full spin Hamiltonian, shape [..., M, N, N].
        :param args: Additional positional arguments passed to the populator.
        :param kwargs: Additional keyword arguments passed to the populator.
        :return: Time-dependent population differences Δp(t) = p_upper(t) − p_lower(t)
                 for each transition, shape [..., T, M].
        """
        return self.populator(time, res_fields, lvl_down,
                              lvl_up, resonance_energies,
                              vector_down, vector_up,
                              full_system_vectors, *args, **kwargs)

    def compute_stationary_polarization(self,
                                res_fields: torch.Tensor, lvl_down: torch.Tensor, lvl_up: torch.Tensor,
                                resonance_energies: torch.Tensor,
                                vector_down: torch.Tensor, vector_up: torch.Tensor,
                                full_system_vectors: tp.Optional[torch.Tensor],
                                *args, **kwargs) -> torch.Tensor:
        """
        Compute the initial (t=0) stationary population differences for the resonant EPR transitions.

        This method calculates the population difference between the upper and lower
        resonant levels at the initial time moment, before any time-dependent relaxation
        or excitation dynamics are applied. It delegates the calculation to the
        populator `compute_stationary_polarization` method.

        :param res_fields: Resonance magnetic fields for each transition, shape [..., M].
        :param lvl_down: Indices of the lower energy levels involved in transitions, shape [M].
        :param lvl_up: Indices of the upper energy levels involved in transitions, shape [M].
        :param resonance_energies: Eigenenergies of all spin states, shape [..., M, N].
        :param vector_down: Eigenvectors of the lower energy states, shape [..., M, N].
        :param vector_up: Eigenvectors of the upper energy states, shape [..., M, N].
        :param full_system_vectors: Eigenvectors of the full spin Hamiltonian, shape [..., N, N].
        :param args: Additional positional arguments passed to the populator.
        :param kwargs: Additional keyword arguments passed to the populator.
        :return: Initial population differences Δp(t=0) = p_upper(0) − p_lower(0)
                 for each transition, shape [..., M].
        """
        return self.populator.compute_stationary_polarization(
            res_fields, lvl_down,
            lvl_up, resonance_energies,
            vector_down, vector_up,
            full_system_vectors, *args, **kwargs
        )


class TimeDensityCalculator(TimeIntensityCalculator):
    """Calculate time-dependent transition intensities for time-resolved EPR
    experiments based on.

    matrix density relaxation formalism

    Default RWADensityPopulator populator is used
    """
    def _init_populator(self, temperature: torch.Tensor,
                        populator: tp.Optional[tp.Union[BaseTimeDepPopulator, str]],
                        context: tp.Optional[contexts.BaseContext],
                        disordered: bool, computational_details: ComputationalDetails,
                        device: torch.device, dtype: torch.dtype):
        if populator is None:
            return RWADensityPopulator(
                context=context, init_temperature=temperature, disordered=disordered,
                angle_average_steps=computational_details.time_evolution_angle_average_steps,
                device=device, dtype=dtype)
        elif isinstance(populator, str):
            if populator == "rwa":
                return RWADensityPopulator(
                    context=context, init_temperature=temperature, disordered=disordered,
                    angle_average_steps=computational_details.time_evolution_angle_average_steps,
                    device=device, dtype=dtype)
            elif populator == "propagator":
                return PropagatorDensityPopulator(
                    context=context, init_temperature=temperature, disordered=disordered,
                    angle_average_steps=computational_details.time_evolution_angle_average_steps,
                    device=device, dtype=dtype)
            else:
                raise ValueError("populator can be None, user-defined or sting 'rwa' or 'propagator'")
        else:
            setattr(populator, "disordered", disordered)
            return populator


@dataclass
class ParamSpec:
    """Let's consider the Hamiltonian with shape [..., N, N], where N is spin
    system size.

    Its resonance fields have dimension [...., K]. Let's call it 'scalar'
    Its eigen values have dimension [..., K, N], where K is number of resonance transitions. Let's call it 'vector'
    Its eigen vectors have dimension [..., K, N, N], where K is number of resonance transitions. Let's call it 'matrix'

    For some purposes it is necessary to get not only intensities, res-fields and width at resonance points
    but other parameters. To generalize the approach of making these parameters it is necessary to te
    """
    category: str
    dtype: torch.dtype

    def __post_init__(self):
        assert self.category in (
            "scalar", "vector", "matrix"), f"Category must be one of 'scalar', 'vector', 'matrix', got {self.category}"


class HamComputationMethod(str, Enum):
    SECULAR = "secular"
    DIRECT = "direct"


class BaseSpectra(nn.Module, ABC):
    """Abstract base class for all EPR spectral simulations.

    Provides the common pipeline for:
    - Hamiltonian diagonalization (strategy defined by subclasses)
    - Intensity and linewidth calculation
    - Orientation averaging and line broadening

    Subclasses must implement the abstract methods that define the specific
    resonance‑finding strategy and intensity computation details.
    """

    def __init__(
        self,
        resonance_parameter: tp.Union[float, torch.Tensor],
        sample: tp.Optional[spin_model.MultiOrientedSample] = None,
        spin_system_dim: tp.Optional[int] = None,
        batch_dims: tp.Optional[tp.Union[int, tuple]] = None,
        mesh: tp.Optional[mesher.BaseMesh] = None,
        intensity_calculator: tp.Optional[BaseResIntensityCalculator] = None,
        populator: tp.Optional[tp.Union[BasePopulator, str]] = None,
        spectra_integrator: tp.Optional[BaseSpectraIntegrator] = None,
        harmonic: int = 1,
        post_spectra_processor: PostSpectraProcessing = PostSpectraProcessing(),
        temperature: tp.Optional[tp.Union[float, torch.Tensor]] = 293,
        recompute_spin_parameters: bool = True,
        computational_details: ComputationalDetails = ComputationalDetails(),
        inference_mode: bool = True,
        output_eigenvector: tp.Optional[bool] = None,
        context: tp.Optional[contexts.BaseContext] = None,
        hamiltonian_mode: tp.Union[str, HamComputationMethod] = HamComputationMethod.DIRECT,
        output_mode: tp.Union[str, OutputSpectraMode] = OutputSpectraMode.TOTAL,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        """
        :param resonance_parameter: Resonance parameter of experiment (frequency or field).
        :param sample: MultiOrientedSample used to extract meta information.
            If None, spin_system_dim, batch_dims, and mesh must be provided.
        :param spin_system_dim: Hilbert space dimension of the spin system.
        :param batch_dims: Number of batch dimensions.
        :param mesh: Orientation sampling mesh.
        :param intensity_calculator: Class to compute transition intensities.
        :param populator: Class to compute population differences.
        :param spectra_integrator: Class to integrate resonance lines into a spectrum.
        :param harmonic: Spectral harmonic (0 = absorption, 1 = first derivative).
        :param post_spectra_processor: Post‑processing and line‑broadening handler.
        :param temperature: Sample temperature in Kelvin.
        :param recompute_spin_parameters: If False, cache resonance data between calls.
        :param computational_details: Numerical settings for spectrum generation.
        :param inference_mode: If True, run forward under `torch.inference_mode()`.
        :param output_eigenvector: If True, compute and return full eigenvectors.
        :param context: Relaxation/population dynamics context.
        :param hamiltonian_mode: Method for Hamiltonian eigen‑computation ("secular" or "direct").
        :param output_mode: Output organization ("total" or "transitions").
        :param device: Computation device.
        :param dtype: Floating‑point precision.
        """
        super().__init__()

        if isinstance(hamiltonian_mode, str):
            hamiltonian_mode = HamComputationMethod(hamiltonian_mode.lower())
        elif not isinstance(hamiltonian_mode, HamComputationMethod):
            raise ValueError(f"Invalid computation_method: {hamiltonian_mode}")

        self.register_buffer(
            "resonance_parameter",
            torch.tensor(resonance_parameter, device=device, dtype=dtype),
        )
        self.register_buffer(
            "threshold",
            torch.tensor(computational_details.intensity_threshold, device=device, dtype=dtype),
        )
        self.register_buffer("tolerance", torch.tensor(1e-7, device=device, dtype=dtype))
        self.register_buffer("intensity_std", torch.tensor(1e-14, device=device, dtype=dtype))

        self.spin_system_dim, self.batch_dims, self.mesh = self._init_sample_parameters(
            sample, spin_system_dim, batch_dims, mesh
        )
        self.mesh_size = self.mesh.initial_size
        self.broader = Broadener(device=device)

        self.output_eigenvector = self._init_output_eigenvector(output_eigenvector, context)
        self.res_algorithm = self._init_res_algorithm(
            output_eigenvector=self.output_eigenvector,
            hamiltonian_mode=hamiltonian_mode,
            computational_details=computational_details,
            device=device,
            dtype=dtype,
        )

        if hamiltonian_mode == HamComputationMethod.SECULAR:
            self._hamiltonian_getter = lambda s: s.get_hamiltonian_terms_secular()
        else:
            self._hamiltonian_getter = lambda s: s.get_hamiltonian_terms()

        self.intensity_calculator = self._get_intensity_calculator(
            intensity_calculator,
            temperature,
            populator,
            context,
            computational_details,
            device=device,
            dtype=dtype,
        )
        self._param_specs = self._get_param_specs()

        if isinstance(output_mode, str):
            output_mode = OutputSpectraMode(output_mode.lower())
        elif not isinstance(output_mode, OutputSpectraMode):
            raise ValueError(f"Invalid output method: {output_mode}")

        self.spectra_processor = self._init_spectra_processor(
            spectra_integrator,
            harmonic,
            post_spectra_processor,
            computational_details=computational_details,
            output_mode=output_mode,
            device=device,
            dtype=dtype,
        )

        self.recompute_spin_parameters = recompute_spin_parameters
        self._init_cached_parameters()

        if inference_mode:
            self.forward = self._wrap_with_inference_mode(self.forward)

        self.to(device)
        self.to(dtype)

    def _init_cached_parameters(self) -> None:
        """Initialize cache placeholders for spin parameters."""
        if not self.recompute_spin_parameters:
            self._cashed_flag = False
            self._cached_data = None
        else:
            self._cashed_flag = False

    def _wrap_with_inference_mode(self, forward_fn: tp.Callable) -> tp.Callable:
        """Wrap forward to run under `torch.inference_mode`."""
        @wraps(forward_fn)
        def wrapper(*args, **kwargs):
            with torch.inference_mode():
                return forward_fn(*args, **kwargs)
        return wrapper

    def _nan_to_zeros(self, *args) -> list[torch.Tensor]:
        return [torch.nan_to_num(arg, nan=0.0) for arg in args]

    def _init_sample_parameters(
        self,
        sample: tp.Optional[spin_model.MultiOrientedSample],
        spin_system_dim: tp.Optional[int],
        batch_dims: tp.Optional[tp.Union[int, tuple]],
        mesh: tp.Optional[mesher.BaseMesh],
    ) -> tp.Tuple[int, tp.Union[int, tuple], mesher.BaseMesh]:
        """Extract or validate core sample metadata."""
        if sample is None:
            if (spin_system_dim is not None) and (batch_dims is not None) and (mesh is not None):
                return spin_system_dim, batch_dims, mesh
            raise TypeError("You must pass sample or spin_system_dim, batch_dims, mesh arguments")
        else:
            spin_system_dim = sample.base_spin_system.spin_system_dim
            batch_dims = sample.config_shape[:-1]
            mesh = sample.mesh
        return spin_system_dim, batch_dims, mesh

    def _init_output_eigenvector(
        self, output_eigenvector: tp.Optional[bool], context: tp.Optional[contexts.BaseContext]
    ) -> bool:
        """Determine if full eigenvectors are needed."""
        if output_eigenvector is not None:
            return output_eigenvector
        return context is not None

    def _freq_to_field(
        self, vector_down: torch.Tensor, vector_up: torch.Tensor, Gz: torch.Tensor
    ) -> torch.Tensor:
        """Convert frequency‑domain intensities to field‑swept representation."""
        factor_1 = compute_matrix_element(vector_up, vector_up, Gz)
        factor_2 = compute_matrix_element(vector_down, vector_down, Gz)
        diff = (factor_1 - factor_2).abs()
        safe_diff = torch.where(diff < self.tolerance * constants.BOHR / constants.PLANCK,
                                self.tolerance * constants.BOHR / constants.PLANCK, diff)
        return safe_diff.reciprocal()

    def _mask_components(self, intensities_mask: torch.Tensor, *extras) -> list:
        """Apply intensity‑based masking to auxiliary parameters using ParamSpec."""
        updated_extras = []
        for idx, param_spec in enumerate(self._param_specs):
            if param_spec.category == "scalar":
                updated_extras.append(extras[idx][..., intensities_mask])
            elif param_spec.category == "vector":
                updated_extras.append(extras[idx][..., intensities_mask, :])
            elif param_spec.category == "matrix":
                updated_extras.append(extras[idx][..., intensities_mask, :, :])
        return updated_extras

    def _mask_full_system_eigenvectors(
        self, mask: torch.Tensor, full_system_vectors: tp.Optional[torch.Tensor]
    ) -> tp.Optional[torch.Tensor]:
        """Apply transition mask to the full eigenbasis."""
        if full_system_vectors is not None:
            return full_system_vectors[..., mask, :, :]
        return full_system_vectors

    def _compute_additional(self, sample, F, Gx, Gy, Gz, full_system_vectors, *extras) -> tp.Any:
        """Hook for subclass‑specific post‑masking computations."""
        return extras

    def _get_param_specs(self) -> list[ParamSpec]:
        """Define additional parameters to extract alongside resonance data.

        Subclasses may override to request extra quantities (e.g., level indices,
        full vectors) during masking and batching. Each `ParamSpec` declares
        the tensor category ("scalar", "vector", "matrix") and dtype.

        :return: List of `ParamSpec` instances specifying auxiliary output parameters.
        """
        return []

    @abstractmethod
    def _init_res_algorithm(
        self,
        output_eigenvector: bool,
        hamiltonian_mode: HamComputationMethod,
        computational_details: ComputationalDetails,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """Instantiate the resonance/eigenvalue algorithm for this spectral type."""
        pass

    @abstractmethod
    def _init_spectra_processor(
        self,
        spectra_integrator: tp.Optional[BaseSpectraIntegrator],
        harmonic: int,
        post_spectra_processor: PostSpectraProcessing,
        computational_details: ComputationalDetails,
        output_mode: OutputSpectraMode,
        device: torch.device,
        dtype: torch.dtype,
    ) -> BaseResProcessing:
        """Create the appropriate spectra processor (powder or crystal)."""
        pass

    @abstractmethod
    def _get_intensity_calculator(
        self,
        intensity_calculator: tp.Optional[BaseResIntensityCalculator],
        temperature: float,
        populator: tp.Optional[tp.Union[BasePopulator, str]],
        context: tp.Optional[contexts.BaseContext],
        computational_details: ComputationalDetails,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """Instantiate or return the intensity calculator."""
        pass

    @abstractmethod
    def compute_parameters(self, *args, **kwargs):
        """Compute intensities, widths, and auxiliary data from eigenstates."""
        pass

    @abstractmethod
    def _postcompute_batch_data(self, *args, **kwargs):
        """Apply post‑diagonalization corrections (e.g., time‑dependent populations)."""
        pass

    @abstractmethod
    def forward(
            self, sample: spin_model.MultiOrientedSample,
            fields: torch.Tensor, time: tp.Optional[torch.Tensor] = None, **kwargs):
        """Main simulation entry point. Orchestrates the pipeline."""
        pass

    def update_context(self, new_context: contexts.BaseContext) -> None:
        """Update context.

        :param new_context: New context object with updated parameters
        :return:
        """
        self.intensity_calculator.populator.set_context(new_context)


class BaseResSpectra(BaseSpectra):
    """Base class for EPR spectral simulation.

    Provides the complete pipeline for computing EPR spectra from spin Hamiltonian:
    1. Compute resonance fields/frequencies by diagonalizing Hamiltonian
    2. Calculate transition intensities from matrix elements and populations
    3. Compute linewidths from strain tensors
    4. Integrate over orientation mesh (for powder samples)
    5. Apply line broadening (Gaussian/Lorentzian/Voigt)

    Supports both stationary (CW) and time-resolved experiments, powder and
    single-crystal samples, field-swept and frequency-swept modes.
    """
    def __init__(self,
                 resonance_parameter: tp.Union[float, torch.Tensor],
                 sample: tp.Optional[spin_model.MultiOrientedSample] = None,
                 spin_system_dim: tp.Optional[int] = None,
                 batch_dims: tp.Optional[tp.Union[int, tuple]] = None,
                 mesh: tp.Optional[mesher.BaseMesh] = None,
                 intensity_calculator: tp.Optional[BaseResIntensityCalculator] = None,
                 populator: tp.Optional[tp.Union[BasePopulator, str]] = None,
                 spectra_integrator: tp.Optional[BaseSpectraIntegrator] = None,
                 harmonic: int = 1,
                 post_spectra_processor: PostSpectraProcessing = PostSpectraProcessing(),
                 temperature: tp.Optional[tp.Union[float, torch.Tensor]] = 293,
                 recompute_spin_parameters: bool = True,
                 computational_details: ComputationalDetails = ComputationalDetails(),
                 inference_mode: bool = True,
                 output_eigenvector: tp.Optional[bool] = None,
                 context: tp.Optional[contexts.BaseContext] = None,
                 hamiltonian_mode: tp.Union[str, HamComputationMethod] = HamComputationMethod.DIRECT,
                 output_mode: tp.Union[str, OutputSpectraMode] = OutputSpectraMode.TOTAL,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32,
                 ):
        """
        :param resonance_parameter: Resonance parameter of experiment: frequency or field.

        :param sample: MultiOrientedSample.
            It is just an example of spin system to extract meta information (spin_system_dim, batch_dims, mesh)
            If it is None, then spin_system_dim, batch_dims, mesh should be given

        :param spin_system_dim: The size of spin system. Default is None
        :param batch_dims: The number of batch dimensions. Default is None
        :param mesh: Mesh object. Default is None
            If (mesh, batch_dims, spin_system_dim) are None then sample object should be given

        :param intensity_calculator:
            Class that is used to compute intensity of spectra via temperature/ time/ hamiltonian parameters.
            Default is None
            If it is None then it will be initialized as default calculator specific to given spectra_creator

        :param populator:
            Class that is used to compute part intensity due to population of levels. Default is None
            If it is None then it is initialized as default populator specific to given (default) intensity_calculator

        :param spectra_integrator:
            Class to integrate the resonance lines to get the spectrum

        :param harmonic: Harmonic of spectra: 1 is derivative, 0 is absorbance
        :param post_spectra_processor:
            Class to post process resulted resonance data (fields, intensities, width):
            integration, mesh mapping and so on. Default post_spectra_processor is powder spectra processor

        :param temperature: The temperature of an experiment. If populator is not None it takes from it
        :param recompute_spin_parameters:
            Recompute spin parameters in __call__ methods. For stationary creator is True, for time resolves is False

        :param computational_details: ComputationalDetails
            computational_details : ComputationalDetails, optional
            Configuration object that governs the numerical aspects of spectrum generation.
            Contains the following fields:

            - **chunk_size** (`int`, default=128):
              Number of magnetic field points processed per integration batch.
              Larger values improve throughput but increase memory consumption.

            - **res_field_r_tol** (`float`, default=1e-5):
              Relative tolerance for adaptive subdivision of field intervals during resolution enhancement.

            - **res_field_split_max_iterations** (`int`, default=20):
              Maximum depth of recursive field-sector splitting.

            - **intensity_threshold** (`float`, default=1e-2):
              Minimum relative intensity (as a fraction of the strongest transition) required for
              transition to be included.

            -for other parameters specifications, read
             docs of :class:'mars.spectra_manager.spectra_manager.ComputationalDetails'

        :param inference_mode: bool
            If inference_mode is True, then forward method will be performed under with torch.inference_mode():

        :param output_eigenvector: Optional[bool]
            If True, computes and returns the full system eigenvector. If False, returns None.
            For stationary computations, the default is False if context is None;
            for time-resolved simulations, the default is True.
            If set to None, the value is inferred automatically based on the population dynamics logic.

        :param context: Optional[context]
            The instance of BaseContext which describes the relaxation mechanism.
            It can have the initial population logic, transition between energy levels, dephasings, driven transition,
            out system transitions. For more complicated scenario the full relaxation superoperator can be used.

        :param hamiltonian_mode: str, HamComputationMethod
         {"secular", "direct"} or HamComputationMethod, default="direct"
            Method for Hamiltonian eigen values, eigen vectors, resonance filed computation:
            - "secular": uses secular approximation (faster)
            - "direct": use the general algorithm: res-field or res-freq (slower, the most general)

        :param output_mode: str, OutputSpectraMode:
        Controls the organization of the computed spectrum.

            "total": returns the conventional summed spectrum over all allowed transitions (default behavior).

            "transitions": returns dict of lvl_down, lvl_up and spectrum,
            where each slice corresponds to the contribution of an individual transition
            (e.g., between specific energy levels).
            Default is "total".

        :param device: cpu / cuda. Base device for computations.

        :param dtype: float32 / float64
        Base dtype for all types of operations. If complex parameters is used,
        they will be converted in complex64, complex128
        """
        super().__init__(resonance_parameter, sample, spin_system_dim, batch_dims, mesh, intensity_calculator,
                         populator, spectra_integrator, harmonic, post_spectra_processor,
                         temperature, recompute_spin_parameters,
                         computational_details,
                         inference_mode, output_eigenvector, context, hamiltonian_mode, output_mode,
                         device=device, dtype=dtype)

        if isinstance(hamiltonian_mode, str):
            hamiltonian_mode = HamComputationMethod(hamiltonian_mode.lower())
        elif not isinstance(hamiltonian_mode, HamComputationMethod):
            raise ValueError(f"Invalid computation_method: {hamiltonian_mode}")

        self.register_buffer("resonance_parameter", torch.tensor(resonance_parameter, device=device, dtype=dtype))
        self.register_buffer("threshold", torch.tensor(
            computational_details.intensity_threshold, device=device, dtype=dtype)
        )
        self.register_buffer("tolerance", torch.tensor(1e-7, device=device, dtype=dtype))
        self.register_buffer("intensity_std", torch.tensor(3.5829e-13, device=device, dtype=dtype))

        self.spin_system_dim, self.batch_dims, self.mesh =\
            self._init_sample_parameters(sample, spin_system_dim, batch_dims, mesh)
        self.mesh_size = self.mesh.initial_size
        self.broader = Broadener(device=device)

        self.output_eigenvector = self._init_output_eigenvector(output_eigenvector, context)
        self.res_algorithm = self._init_res_algorithm(
            output_eigenvector=self.output_eigenvector,
            hamiltonian_mode=hamiltonian_mode,
            computational_details=computational_details,
            device=device, dtype=dtype)

        if hamiltonian_mode == HamComputationMethod.SECULAR:
            self._hamiltonian_getter = lambda s: s.get_hamiltonian_terms_secular()
        else:
            self._hamiltonian_getter = lambda s: s.get_hamiltonian_terms()

        self.intensity_calculator = self._get_intensity_calculator(intensity_calculator,
                                                                   temperature, populator, context,
                                                                   computational_details,
                                                                   device=device, dtype=dtype)
        self._param_specs = self._get_param_specs()

        if isinstance(output_mode, str):
            output_mode = OutputSpectraMode(output_mode.lower())
        elif not isinstance(output_mode, OutputSpectraMode):
            raise ValueError(f"Invalid output method: {output_mode}")
        self.spectra_processor = self._init_spectra_processor(spectra_integrator,
                                                              harmonic,
                                                              post_spectra_processor,
                                                              computational_details=computational_details,
                                                              output_mode=output_mode,
                                                              device=device, dtype=dtype)
        self.recompute_spin_parameters = recompute_spin_parameters
        self._init_cached_parameters()

        if inference_mode:
            self.forward = self._wrap_with_inference_mode(self.forward)

        self.to(device)
        self.to(dtype)

    def _init_cached_parameters(self):
        """Initialize internal buffers to support optional caching of spin parameters.

        When `recompute_spin_parameters=False`, resonance-related tensors
        (eigenvectors, levels, fields, etc.) are computed once and stored.
        This method sets up placeholder attributes used during the first forward pass.
        """
        if not self.recompute_spin_parameters:
            self._cashed_flag = False
            self.vectors_u = None
            self.vectors_v = None
            self.valid_lvl_down = None
            self.valid_lvl_up = None
            self.res_fields = None
            self.resonance_energies = None
            self.full_eigen_vectors = None
            self._resfield_method = self._cashed_resfield

        else:
            self._resfield_method = self._recomputed_resfield

    def _wrap_with_inference_mode(self, forward_fn: tp.Callable[[tp.Any], tp.Any]):
        """Wrap a forward function to execute under `torch.inference_mode`.

        Disables gradient computation and other autograd overhead for faster inference.

        :param forward_fn: The original forward method to wrap.
        :return: A wrapped version of `forward_fn` that runs in inference mode.
        """

        @wraps(forward_fn)
        def wrapper(*args, **kwargs):
            with torch.inference_mode():
                return forward_fn(*args, **kwargs)
        return wrapper

    def _init_res_algorithm(self, output_eigenvector: bool,
                            hamiltonian_mode: HamComputationMethod,
                            computational_details: ComputationalDetails,
                            device: torch.device, dtype: torch.dtype):
        """Instantiate the resonance field computation algorithm.

        Selects an appropriate Hamiltonian eigen data backend based on
        whether full eigenvectors are needed and whether secular approximation is used.

        :param output_eigenvector: Whether full system eigenvectors should be computed.
        :param hamiltonian_mode: the method to use to compute the Hamiltonian eigen data.

        :param computational_details: The computational details to create EPR spectra:
                accuracy, number of iterations, and so on.

        :return: Configured resonance field solver.
        """
        return res_field_algorithm.ResField(
            spin_system_dim=self.spin_system_dim,
            mesh_size=self.mesh_size,
            batch_dims=self.batch_dims,
            splitting_max_iterations=computational_details.res_field_split_max_iterations,
            r_tol=computational_details.res_field_r_tol,
            output_full_eigenvector=output_eigenvector,
            device=device,
            dtype=dtype
        )

    @abstractmethod
    def _init_spectra_processor(self,
                                spectra_integrator: tp.Optional[BaseSpectraIntegrator],
                                harmonic: int,
                                post_spectra_processor: PostSpectraProcessing,
                                computational_details: ComputationalDetails,
                                output_mode: OutputSpectraMode,
                                device: torch.device,
                                dtype: torch.dtype) -> BaseResProcessing:
        """Create a processor for integrating and post-processing spectral data.

        Must be implemented by subclasses to select appropriate powder or crystal
        processing logic based on sample type.

        :param spectra_integrator: Custom integrator; if None, one is auto-selected.
        :param harmonic: Spectral harmonic (0 = absorption, 1 = first derivative).
        :param post_spectra_processor: Line-broadening and convolution handler.
        :param computational_details: The details of final spectral integration and spectra processing. For example,

            -integration_natural_width : float, default=1e-6
                Minimum intrinsic linewidth added to every transition. Measures in FWHM
                Prevents division-by-zero or extreme sharpening when user-provided widths are
                very small or zero. Also it can be used as substitution for ordinary gaussian broadaning in the sample.

            - integration_gaussian_method : str, default="exp"
                Method used to evaluate the Gaussian function exp(-x²) during final integration:
                - "exp": uses exact PyTorch exponential (higher accuracy),
                - "approx": uses a fast 6th-order rational approximation (see ``gaussian_approx``).

            - chunk_size (`int`, default=128):
              Number of magnetic field points processed per integration batch.
              Larger values improve throughput but increase memory consumption.

            -for other parameters specifications, read
             docs of :class:'mars.spectra_manager.spectra_manager.ComputationalDetails'

        :param output_mode: The mode which computes the EPR-spectroscopy output

        :return: Initialized spectra processor instance.
        """
        pass

    def _init_sample_parameters(self,
                                sample: tp.Optional[spin_model.MultiOrientedSample],
                                spin_system_dim: tp.Optional[int],
                                batch_dims: tp.Optional[tp.Union[int, tuple]],
                                mesh: tp.Optional[mesher.BaseMesh]):
        """Extract or validate core sample metadata.

        Resolves spin system dimensionality, batch shape, and orientation mesh
        either from a provided `sample` or explicit arguments.

        :param sample: Reference sample object.
        :param spin_system_dim: Hilbert space dimension of the spin system.
        :param batch_dims: Shape of batch dimensions (excluding orientation and state axes).
        :param mesh: Orientation sampling grid (powder or crystal).
        :return: `(spin_system_dim, batch_dims, mesh)` as resolved values.
        :raises TypeError: If insufficient information is provided to infer all three parameters.
        """

        if sample is None:
            if (spin_system_dim is not None) and (batch_dims is not None) and (mesh is not None):
                return spin_system_dim, batch_dims, mesh
            else:
                raise TypeError("You should pass sample or spin_system_dim, batch_dims, mesh arguments")
        else:
            spin_system_dim = sample.base_spin_system.spin_system_dim
            batch_dims = sample.config_shape[:-1]
            mesh = sample.mesh

        return spin_system_dim, batch_dims, mesh

    @abstractmethod
    def _get_intensity_calculator(self,
                                  intensity_calculator: tp.Optional[BaseResIntensityCalculator],
                                  temperature: float,
                                  populator: tp.Optional[tp.Union[BasePopulator, str]],
                                  context: tp.Optional[contexts.BaseContext],
                                  computational_details: ComputationalDetails,
                                  device: torch.device, dtype: torch.dtype):
        """Instantiate or return the intensity calculator for transition strengths.


        :param intensity_calculator: Pre-configured calculator; if None, one is created.
        :param temperature: Sample temperature in Kelvin.
        :param populator: Population model or identifier.
        :param context: Relaxation/population dynamics context.
        :param computational_details: ComputationalDetails
            computational_details : ComputationalDetails, optional
            Configuration object that governs the numerical aspects of spectrum generation.
        :param device: Computation device.
        :param dtype: Floating-point precision.
        :return: Ready-to-use intensity calculator.
        """
        if intensity_calculator is None:
            return StationaryIntensityCalculator(
                self.spin_system_dim, temperature, populator, context,
                disordered=self.mesh.disordered,
                computational_details=computational_details, device=device, dtype=dtype
            )
        else:
            return intensity_calculator

    def _freq_to_field(self, vector_down: torch.Tensor, vector_up: torch.Tensor, Gz: torch.Tensor) -> torch.Tensor:
        """Convert frequency-domain transitions to effective field units.

        Computes the reciprocal of the Zeeman splitting difference between states,
        used to transform intensities from frequency to field representation.

        :param vector_down: Lower-state eigenvector [..., N].
        :param vector_up: Upper-state eigenvector [..., N].
        :param Gz: z-component of the Zeeman operator [..., N, N].
        :return: Field conversion factor [...], with safe handling near zero splitting.
        """
        factor_1 = compute_matrix_element(vector_up, vector_up, Gz)
        factor_2 = compute_matrix_element(vector_down, vector_down, Gz)

        diff = (factor_1 - factor_2).abs()
        tolerancy = self.tolerance * constants.BOHR / constants.PLANCK
        safe_diff = torch.where(diff < tolerancy, tolerancy, diff)
        return safe_diff.reciprocal()

    def _init_output_eigenvector(
            self, output_eigenvector: tp.Optional[bool], context: tp.Optional[contexts.BaseContext]
    ) -> bool:
        """Determine whether full system eigenvectors should be computed.

        By default, eigenvectors are computed only when a `context` is provided
        (typically required for time-resolved or density-matrix simulations).

        :param output_eigenvector: Explicit override; if None, inferred from context.
        :param context: Population/relaxation context object.
        :return: True if eigenvectors should be computed.
        """
        if output_eigenvector is not None:
            return output_eigenvector
        else:
            return context is not None

    def _cashed_resfield(self, sample: spin_model.MultiOrientedSample,
                                B_low: torch.Tensor, B_high: torch.Tensor,
                                F: torch.Tensor, Gz: torch.Tensor):
        """Compute or retrieve cached resonance fields and eigensystem.

        On first call, delegates to `_recomputed_resfield` and stores results.
        Subsequent calls return the cached tensors without recomputation.

        :param sample: Spin system with orientation mesh and interactions.
        :param B_low: Lower bound of magnetic field sweep.
        :param B_high: Upper bound of magnetic field sweep.
        :param F: Field-independent part of the Hamiltonian.
        :param Gz: Zeeman operator along z.
        :return: Same as `_recomputed_resfield`.
        """
        if not self._cashed_flag:
            (self.vectors_u, self.vectors_v), (self.valid_lvl_down, self.valid_lvl_up), self.res_fields,\
                self.resonance_energies, self.full_eigen_vectors =\
                self._recomputed_resfield(sample, B_low, B_high, F, Gz)

            self._cashed_flag = True

        return (self.vectors_u, self.vectors_v), (self.valid_lvl_down, self.valid_lvl_up), self.res_fields,\
            self.resonance_energies, self.full_eigen_vectors

    def _recomputed_resfield(self, sample: spin_model.MultiOrientedSample,
                                B_low: torch.Tensor, B_high: torch.Tensor,
                                F: torch.Tensor, Gz: torch.Tensor):
        """Compute resonance fields and associated quantum states.

        Calls the configured resonance algorithm to solve for transitions
        within the specified field window.

        :param sample: Spin system definition.
        :param B_low: Lower bound of magnetic field sweep.
        :param B_high: Upper bound of magnetic field sweep.
        :param F: Static Hamiltonian term.
        :param Gz: Zeeman coupling operator.
        :return: Tuple containing:
            - (vectors_u, vectors_v): lower/upper eigenvectors [..., M, N]
            - (lvl_down, lvl_up): level indices [..., M]
            - res_fields: resonance fields [..., M]
            - resonance_energies: eigenvalues [..., N]
            - full_eigen_vectors: complete eigenbasis [..., N, N] or None
        """
        (vectors_u, vectors_v), (valid_lvl_down, valid_lvl_up), res_fields, resonance_energies, full_eigen_vectors =\
                self.res_algorithm(sample, self.resonance_parameter, B_low, B_high, F, Gz)

        return (vectors_u, vectors_v), (valid_lvl_down, valid_lvl_up), res_fields,\
            resonance_energies, full_eigen_vectors

    def _get_intensity_mask(self,
                            intensities: torch.Tensor,
                            res_manifold: torch.Tensor,
                            lvl_down: torch.Tensor,
                            lvl_up: torch.Tensor,
                            energies: torch.Tensor,
                            vector_down, vector_up,
                            full_system_vectors: tp.Optional[torch.Tensor]
                            ) -> torch.Tensor:
        """
        Generate a boolean mask to filter out transitions with intensities below a threshold.

        The mask is created by comparing the absolute intensities, normalized by the
        global maximum absolute intensity, against `self.threshold`. It evaluates to
        True if any value along all dimensions (except the last one) exceeds the threshold.

        :param intensities: Tensor of transition intensities.
        :param res_manifold: Resonance fields / frequencies (passed for signature consistency).
        :param lvl_down: Energy levels of the lower states.
        :param lvl_up: Energy levels of the upper states.
        :param energies: Resonance energies (passed for signature consistency).
        :param full_system_vectors: Eigenvectors of the full spin system.
        :return: A boolean tensor indicating which transitions have sufficient
                 intensity to be kept.
        """
        lines_dimension = tuple(range(intensities.ndim - 1))
        intensities_mask = (intensities.abs() / intensities.abs().max() > self.threshold).any(dim=lines_dimension)
        return intensities_mask


    def forward(self,
                 sample: spin_model.MultiOrientedSample,
                 fields: torch.Tensor, time: tp.Optional[torch.Tensor] = None, **kwargs):
        """Compute EPR spectrum over a given magnetic fields range.

        Orchestrates the full simulation pipeline: diagonalization, intensity
        calculation, broadening, orientation averaging, and line-shape convolution.

        :param sample: MultiOrientedSample object.
        :param fields: The magnetic fields in Tesla units
        :param time: It is used only for time resolved spectra
        :param kwargs:
        :return: spectra in 1D or 2D. Batched or un batched.
        Depending on spectra Proccessor it can be another output format
        """
        B_low = fields[..., 0]
        B_high = fields[..., -1]
        B_low = B_low.unsqueeze(-1).repeat(*([1] * B_low.ndim), *self.mesh_size)
        B_high = B_high.unsqueeze(-1).repeat(*([1] * B_high.ndim), *self.mesh_size)

        F, Gx, Gy, Gz = self._hamiltonian_getter(sample)
        (vector_down, vector_up), (lvl_down, lvl_up), res_fields,\
            resonance_energies, full_system_vectors = self._resfield_method(sample, B_low, B_high, F, Gz)

        if (vector_down.shape[-2] == 0):
            return torch.zeros_like(fields)

        res_fields, intensities, width, full_system_vectors, *extras =\
            self.compute_parameters(sample, F, Gx, Gy, Gz,
                                    res_fields,
                                    lvl_down, lvl_up,
                                    resonance_energies,
                                    vector_down, vector_up,
                                    full_system_vectors)

        res_fields, intensities, width = self._postcompute_batch_data(
            sample, res_fields, intensities, width, F, Gx, Gy, Gz, full_system_vectors, time, *extras, **kwargs
        )

        gauss = sample.gauss
        lorentz = sample.lorentz
        return self._finalize(res_fields, intensities, width, gauss, lorentz, fields, lvl_down, lvl_up)

    def _postcompute_batch_data(self, sample: spin_model.BaseSample, res_fields: torch.Tensor,
                                intensities: torch.Tensor, width: torch.Tensor,
                                F: torch.Tensor, Gx: torch.Tensor, Gy: torch.Tensor,
                                Gz: torch.Tensor,
                                full_system_vectors: tp.Optional[torch.Tensor],
                                time: tp.Optional[torch.Tensor], *extras,  **kwargs) -> tp.Any:
        """Apply post-diagonalization corrections or time-dependent population scaling.

        Base implementation returns inputs unchanged. Subclasses (e.g., time-resolved)
        override to inject population dynamics.

        :param sample: Spin system instance.
        :param res_fields: Resonance field positions.
        :param intensities: Transition strengths.
        :param width: linewidths .
        :param F, Gx, Gy, Gz: Hamiltonian components.
        :param full_system_vectors: Full eigenbasis if computed.
        :param time: Time axis for dynamic simulations.
        :param extras: Additional parameters from `_add_to_mask_additional`.
        :param kwargs: Arbitrary keyword arguments.
        :return: Potentially modified `(res_fields, intensities, width)`.
        """
        return res_fields, intensities, width

    def _finalize(self,
                  res_fields: torch.Tensor,
                  intensities: torch.Tensor,
                  width: torch.Tensor,
                  gauss: torch.Tensor,
                  lorentz: torch.Tensor,
                  fields: torch.Tensor,
                  lvl_down: torch.Tensor,
                  lvl_up: torch.Tensor):
        """Apply final spectral integration and line broadening.

        Delegates to the configured `spectra_processor` to produce the output spectrum.

        :param res_fields: Resonance field positions.
        :param intensities: Transition strengths.
        :param width: Total Gaussian linewidth (FWHM).
        :param gauss: Global Gaussian broadening (FWHM).
        :param lorentz: Homogeneous Lorentzian broadening (FWHM).
        :param fields: Output field axis.

        :param lvl_down: Energy level indices of low spin state involved in transition. The shape is '[num_transitions]'
        :param lvl_up: Energy level indices of high spin state involved in transition. The shape is '[num_transitions]'

        :return: The output of the given spectra Proccessor depending on the output_mode
        """
        return self.spectra_processor(res_fields, intensities, width, gauss, lorentz, fields, lvl_down, lvl_up)

    def _mask_components(self, intensities_mask: torch.Tensor, *extras) -> list[tp.Any]:
        """Apply intensity-based masking to auxiliary parameters.

        Uses `ParamSpec` metadata to correctly slice scalar, vector, or matrix extras.

        :param intensities_mask: Boolean mask indicating retained transitions.
        :param extras: Auxiliary tensors to mask, ordered per `_get_param_specs`.
        :return: Masked versions of each extra tensor.
        """
        updated_extras = []
        for idx, param_spec in enumerate(self._param_specs):
            if param_spec.category == "scalar":
                updated_extras.append(extras[idx][..., intensities_mask])

            elif param_spec.category == "vector":
                updated_extras.append(extras[idx][..., intensities_mask, :])

            elif param_spec.category == "matrix":
                updated_extras.append(extras[idx][..., intensities_mask, :, :])
        return updated_extras

    def _add_to_mask_additional(self,
                                lvl_down: torch.Tensor, lvl_up: torch.Tensor,
                                resonance_energies: torch.Tensor,
                                vector_down: torch.Tensor, vector_up: torch.Tensor) -> tp.Any:
        """Return additional tensors to be masked alongside intensities.

        Subclasses may override to include level indices, energies, or vectors
        in the masking step. Base class returns empty tuple.

        :param lvl_down: Index of lower energy level.
        :param lvl_up: Index of upper energy level.
        :param resonance_energies: Hamiltonian eigenvalues.
        :param vector_down: Eigenvector of lower state.
        :param vector_up: Eigenvector of upper state.
        :return: Extra tensors to mask (must match `_get_param_specs` count/order).
        """
        return ()

    def _mask_full_system_eigenvectors(
            self,
            mask: torch.Tensor,
            full_system_vectors: tp.Optional[torch.Tensor]
    ) -> tp.Optional[torch.Tensor]:
        """Optionally mask the full eigenbasis using transition selection.

        :param mask: Boolean mask over transitions.
        :param full_system_vectors: Full set of eigenvectors [..., N, N].
        :return: Masked eigenbasis [..., M, N, N] or None if input was None.
        """

        if full_system_vectors is not None:
            return full_system_vectors[..., mask, :, :]
        else:
            return full_system_vectors

    def _compute_additional(self,
                           sample: spin_model.MultiOrientedSample,
                           F: torch.Tensor,
                           Gx: torch.Tensor,
                           Gy: torch.Tensor,
                           Gz: torch.Tensor,
                           full_system_vectors: tp.Optional[torch.Tensor], *extras) -> tp.Any:
        """Compute derived quantities from masked resonance data.

        Intended for subclass extension. Base implementation returns extras unchanged.

        :param sample: Spin system.
        :param F, Gx, Gy, Gz: Hamiltonian terms.
        :param full_system_vectors: Full eigenbasis.
        :param extras: Previously masked auxiliary data.
        :return: Processed extras (same length as input).
        """
        return extras

    def compute_parameters(self, sample: spin_model.MultiOrientedSample,
                           F: torch.Tensor,
                           Gx: torch.Tensor,
                           Gy: torch.Tensor,
                           Gz: torch.Tensor,
                           res_fields: torch.Tensor,
                           lvl_down: torch.Tensor, lvl_up: torch.Tensor,
                           resonance_energies: torch.Tensor,
                           vector_down: torch.Tensor, vector_up: torch.Tensor,
                           full_system_vectors: tp.Optional[torch.Tensor]) ->\
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, tp.Optional[torch.Tensor], tuple[tp.Any]]:
        """
        :param sample: The sample which transitions must be found.

        :param F: Magnetic free part of spin Hamiltonian H = F + B * G
        :param Gx: x-part of Hamiltonian Zeeman Term
        :param Gy: y-part of Hamiltonian Zeeman Term
        :param Gz: z-part of Hamiltonian Zeeman Term

        :param res_fields: Resonance fields. The shape os [..., N]

        :param lvl_down:
            Energy levels of lower states from which transitions occur.
            Shape: [time, ..., N], where time is the time dimension and
            N is the number of energy levels.

        :param lvl_up:
            Energy levels of upper states to which transitions occur.
            Shape: [time, ..., N], where time is the time dimension and
            N is the number of energy levels.

        :param resonance_energies:
            Energies of spin states. The shape is [..., N]

        :param vector_down:
            Eigenvectors of the lower energy states. The shape is [...., M, N],
            where M is number of transitions, N is number of levels

        :param vector_up:
            Eigenvectors of the upper energy states.The shape is [...., M, N],
            where M is number of transitions, N is number of levels

        :param full_system_vectors: Eigen vector of each level of a spin system. The shape os [..., N, N]. If
        output_eigen_vectors == False, then it will be None

        :return: tuple of the next data
         - Resonance fields
         - Intensities of transitions
         - Width of transition lines
         - Full system eigen vectors or None
         - extras parameters computed in _compute_additional
        """
        intensities = self.intensity_calculator.compute_intensity(
            Gx, Gy, Gz, res_fields, lvl_down, lvl_up, resonance_energies, vector_down, vector_up, full_system_vectors
        )
        intensities = torch.nan_to_num(intensities, nan=0.0, out=intensities)
        intensities_mask = self._get_intensity_mask(
            intensities, res_fields, lvl_down, lvl_up, resonance_energies, vector_down, vector_up, full_system_vectors
        )
        intensities = intensities[..., intensities_mask]

        extras = self._add_to_mask_additional(lvl_down, lvl_up, resonance_energies, vector_down, vector_up)
        extras = self._mask_components(intensities_mask, *extras)
        full_system_vectors = self._mask_full_system_eigenvectors(intensities_mask, full_system_vectors)

        res_fields = res_fields[..., intensities_mask]
        vector_u = vector_down[..., intensities_mask, :]
        vector_v = vector_up[..., intensities_mask, :]

        freq_to_field = self._freq_to_field(vector_u, vector_v, Gz)
        intensities *= freq_to_field
        intensities = intensities / self.intensity_std
        width = self.broader(sample, vector_u, vector_v, res_fields) * freq_to_field

        extras = self._compute_additional(
            sample, F, Gx, Gy, Gz, full_system_vectors, *extras
        )
        return res_fields, intensities, width, full_system_vectors, *extras


class StationarySpectra(BaseResSpectra):
    """Simulates standard EPR experiments where microwave frequency is fixed
    and.

    magnetic field is swept. Computes absorption or first-derivative spectra
    with proper orientation averaging for powder samples.

    Provides the complete pipeline for computing EPR spectra from spin Hamiltonian:
    1. Compute resonance fields/frequencies by diagonalizing Hamiltonian
    2. Calculate transition intensities from matrix elements and populations
    3. Compute linewidths from strain tensors
    4. Integrate over orientation mesh (for powder samples)
    5. Apply line broadening (Gaussian/Lorentzian/Voigt)

    Example usage:
        spectra = StationarySpectra(freq=9.8e9, sample=sample)
        fields = torch.linspace(0.2, 0.4, 500)
        spectrum = spectra(sample, fields)
    """
    def __init__(self,
                 freq: tp.Union[float, torch.Tensor],
                 sample: tp.Optional[spin_model.MultiOrientedSample] = None,
                 spin_system_dim: tp.Optional[int] = None,
                 batch_dims: tp.Optional[tp.Union[int, tuple]] = None,
                 mesh: tp.Optional[mesher.BaseMesh] = None,
                 intensity_calculator: tp.Optional[BaseResIntensityCalculator] = None,
                 populator: tp.Optional[BasePopulator] = None,
                 spectra_integrator: tp.Optional[BaseSpectraIntegrator] = None,
                 harmonic: int = 1,
                 post_spectra_processor: PostSpectraProcessing = PostSpectraProcessing(),
                 temperature: tp.Optional[tp.Union[float, torch.Tensor]] = 293,
                 recompute_spin_parameters: bool = True,
                 computational_details: ComputationalDetails = ComputationalDetails(),
                 inference_mode: bool = True,
                 output_eigenvector: tp.Optional[bool] = None,
                 context: tp.Optional[contexts.BaseContext] = None,
                 hamiltonian_mode: tp.Union[str, HamComputationMethod] = HamComputationMethod.DIRECT,
                 output_mode: tp.Union[str, OutputSpectraMode] = OutputSpectraMode.TOTAL,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32,
                 ):
        """
        :param freq: Resonance frequency of experiment at Hz.

        :param sample: MultiOrientedSample.
            It is just an example of spin system to extract meta information (spin_system_dim, batch_dims, mesh)
            If it is None, then spin_system_dim, batch_dims, mesh should be given

        :param spin_system_dim: The size of spin system. Default is None
        :param batch_dims: The number of batch dimensions. Default is None
        :param mesh: Mesh object. Default is None
            If (mesh, batch_dims, spin_system_dim) are None then sample object should be given

        :param intensity_calculator:
            Class that is used to compute intensity of spectra via temperature/ time/ hamiltonian parameters.
            Default is None
            If it is None then it will be initialized as StationaryIntensityCalculator

        :param populator:
            Class that is used to compute part intensity due to population of levels. Default is None
            If intensity_calculator is None or StationaryIntensityCalculator
            then it will be initialized as StationaryPopulator
            In this case the population is given as Boltzmann population

        :param spectra_integrator:
            Class to integrate the resonance lines to get the spectrum.

        :param harmonic: Harmonic of spectra: 1 is derivative, 0 is absorbance. Default is 1.

        :param post_spectra_processor:
            Class to post process resulted resonance data (fields, intensities, width):
            integration, mesh mapping and so on. Default post_spectra_processor is powder spectra processor

        :param temperature: The temperature of an experiment. If populator is not None it takes from it

        :param recompute_spin_parameters:
            Recompute spin parameters in __call__ methods. For stationary creator is True.

        :param computational_details: ComputationalDetails
            computational_details : ComputationalDetails, optional
            Configuration object that governs the numerical aspects of spectrum generation.
            Contains the following fields:

            - **chunk_size** (`int`, default=128):
              Number of magnetic field points processed per integration batch.
              Larger values improve throughput but increase memory consumption.

            - **res_field_r_tol** (`float`, default=1e-5):
              Relative tolerance for adaptive subdivision of field intervals during resolution enhancement.

            - **res_field_split_max_iterations** (`int`, default=20):
              Maximum depth of recursive field-sector splitting.

            - **intensity_threshold** (`float`, default=1e-2):
              Minimum relative intensity (as a fraction of the strongest transition) required for
              transition to be included.

            -for other parameters meaning read
             docs of :class:'mars.spectra_manager.spectra_manager.ComputationalDetails'

        :param inference_mode: bool
            If inference_mode is True, then forward method will be performed under with torch.inference_mode():

        :param output_eigenvector: Optional[bool]
            If True, computes and returns the full system eigenvector. If False, returns None.
            For stationary computations, the default is False; for time-resolved simulations, the default is True.
            If set to None, the value is inferred automatically based on the population dynamics logic.

        :param context: Optional[context]
            The instance of BaseContext which describes the relaxation mechanism.
            It can have the initial population logic, transition between energy levels, dephasings, driven transition,
            out system transitions. For more complicated scenario the full relaxation superoperator can be used.

        :param hamiltonian_mode: str, HamComputationMethod
         {"secular", "direct"} or HamComputationMethod, default="direct"
            Method for Hamiltonian eigen values, eigen vectors, resonance filed computation:
            - "secular": uses secular approximation (faster)
            - "direct": use the general algorithm: res-field or res-freq (slower, the most general)

        :param output_mode: str, OutputSpectraMode:
        Controls the organization of the computed spectrum.

        "total": returns the conventional summed spectrum over all allowed transitions (default behavior).

        "transitions": returns dict of lvl_down, lvl_up and spectrum,
        where each slice corresponds to the contribution of an individual transition
        (e.g., between specific energy levels).
        Default is "total".

        :param device: cpu / cuda. Base device for computations.

        :param dtype: float32 / float64
        Base dtype for all types of operations. If complex parameters is used,
        they will be converted in complex64, complex128
        """
        super().__init__(freq, sample, spin_system_dim, batch_dims, mesh, intensity_calculator,
                         populator, spectra_integrator, harmonic, post_spectra_processor,
                         temperature, recompute_spin_parameters,
                         computational_details,
                         inference_mode, output_eigenvector, context, hamiltonian_mode, output_mode,
                         device=device, dtype=dtype)

    def _postcompute_batch_data(self, sample: spin_model.BaseSample,
                                res_fields: torch.Tensor, intensities: torch.Tensor, width: torch.Tensor,
                                F: torch.Tensor, Gx: torch.Tensor, Gy: torch.Tensor, Gz: torch.Tensor,
                                full_system_vectors: tp.Optional[torch.Tensor],
                                time: tp.Optional[torch.Tensor],  *extras, **kwargs):
        return res_fields, intensities, width

    def _init_spectra_processor(self,
                                spectra_integrator: tp.Optional[BaseSpectraIntegrator],
                                harmonic: int,
                                post_spectra_processor: PostSpectraProcessing,
                                computational_details: ComputationalDetails,
                                output_mode: OutputSpectraMode,
                                device: torch.device,
                                dtype: torch.dtype) -> BaseResProcessing:
        """Create a processor for integrating and post-processing spectral data.
        :param spectra_integrator: Custom integrator; if None, one is auto-selected.
        :param harmonic: Spectral harmonic (0 = absorption, 1 = first derivative).
        :param post_spectra_processor: Line-broadening and convolution handler.
        :param computational_details: The details of final spectral integration and spectra processing. For example,

            -integration_natural_width : float, default=1e-6
                Minimum intrinsic linewidth added to every transition. Measures in FWHM
                Prevents division-by-zero or extreme sharpening when user-provided widths are
                very small or zero. Also it can be used as substitution for ordinary gaussian broadaning in the sample.

            - integration_gaussian_method : str, default="exp"
                Method used to evaluate the Gaussian function exp(-x²) during final integration:
                - "exp": uses exact PyTorch exponential (higher accuracy),
                - "approx": uses a fast 6th-order rational approximation (see ``gaussian_approx``).

            - chunk_size (`int`, default=128):
              Number of magnetic field points processed per integration batch.
              Larger values improve throughput but increase memory consumption.

            -for other parameters specifications, read
             docs of :class:'mars.spectra_manager.spectra_manager.ComputationalDetails'

        :param output_mode: The output mode for spectra computation
        :return: Initialized spectra processor instance.
        """
        if self.mesh.disordered:
            return PowderStationaryProcessing(self.mesh, spectra_integrator, harmonic, post_spectra_processor,
                                              computational_details=computational_details,
                                              output_mode=output_mode,
                                              device=device, dtype=dtype)
        else:
            return CrystalStationaryProcessing(self.mesh, spectra_integrator, harmonic, post_spectra_processor,
                                               computational_details=computational_details,
                                               output_mode=output_mode,
                                               device=device, dtype=dtype)

    def _get_intensity_calculator(self,
                                  intensity_calculator: tp.Optional[BaseResIntensityCalculator],
                                  temperature: float,
                                  populator: tp.Optional[tp.Union[BasePopulator, str]],
                                  context: tp.Optional[contexts.BaseContext],
                                  computational_details: ComputationalDetails,
                                  device: torch.device, dtype: torch.dtype) -> BaseIntensityCalculator:
        """Instantiate or return the intensity calculator for transition strengths.

        :param intensity_calculator: Pre-configured calculator; if None, one is created.
        :param temperature: Sample temperature in Kelvin.
        :param populator: Population model or identifier.
        :param context: Relaxation/population dynamics context.
        :param computational_details: ComputationalDetails
            computational_details : ComputationalDetails, optional
            Configuration object that governs the numerical aspects of spectrum generation.
        :param device: Computation device.
        :param dtype: Floating-point precision.
        :return: Ready-to-use intensity calculator.
        """
        if intensity_calculator is None:
            return StationaryIntensityCalculator(
                self.spin_system_dim, temperature, populator, context,
                disordered=self.mesh.disordered,
                computational_details=computational_details,
                device=device, dtype=dtype
            )
        else:
            return intensity_calculator

    def __call__(self,
                sample: spin_model.MultiOrientedSample,
                fields: torch.Tensor, time: tp.Optional[torch.Tensor] = None, **kwargs):
        """
        :param sample: MultiOrientedSample object.

        :param fields: The magnetic fields in Tesla units
        :param time: It is used only for time resolved spectra
        :param kwargs:
        :return: spectra or some resonance data depending on the output_mode
        """
        return super().__call__(sample, fields, time)


class TruncTimeSpectra(BaseResSpectra):
    """Compute time-resolved EPR spectra for populations relaxation formalism.

    Uses truncated eigen vectors computation. For the general case use CoupledTimeSpectra

    Unlike CoupledTimeSpectra, only computes eigenvectors for resonant transitions
    (not full system), which improves computational efficiency.

    Provides the complete pipeline for computing EPR spectra from spin Hamiltonian:
    1. Compute resonance fields/frequencies by diagonalizing Hamiltonian
    2. Calculate transition intensities from matrix elements and populations
    3. Compute linewidths from strain tensors
    4. Integrate over orientation mesh (for powder samples)
    5. Apply line broadening (Gaussian/Lorentzian/Voigt)
    """
    def __init__(self,
                 freq: tp.Union[float, torch.Tensor],
                 sample: tp.Optional[spin_model.MultiOrientedSample] = None,
                 spin_system_dim: tp.Optional[int] = None,
                 batch_dims: tp.Optional[tp.Union[int, tuple]] = None,
                 mesh: tp.Optional[mesher.BaseMesh] = None,
                 intensity_calculator: tp.Optional[tp.Callable] = None,
                 populator: tp.Optional[BaseTimeDepPopulator] = None,
                 spectra_integrator: tp.Optional[BaseSpectraIntegrator] = None,
                 harmonic: int = 0,
                 post_spectra_processor: PostSpectraProcessing = PostSpectraProcessing(),
                 temperature: tp.Optional[tp.Union[float, torch.Tensor]] = 293,
                 recompute_spin_parameters: bool = True,
                 computational_details: ComputationalDetails = ComputationalDetails(),
                 inference_mode: bool = True,
                 output_eigenvector: tp.Optional[bool] = None,
                 context: tp.Optional[contexts.BaseContext] = None,
                 hamiltonian_mode: tp.Union[str, HamComputationMethod] = HamComputationMethod.DIRECT,
                 output_mode: tp.Union[str, OutputSpectraMode] = OutputSpectraMode.TOTAL,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32,
                 ):

        """Note that by default these spin systems (energies, vectors, etc.)
        are calculated once and then cached.

        Default harmoinc is None

        :param freq: Resonance frequency of experiment

        :param sample: MultiOrientedSample.
            It is just an example of spin system to extract meta information (spin_system_dim, batch_dims, mesh)
            If it is None, then spin_system_dim, batch_dims, mesh should be given

        :param spin_system_dim: The size of spin system. Default is None
        :param batch_dims: The number of batch dimensions. Default is None
        :param mesh: Mesh object. Default is None
            If (mesh, batch_dims, spin_system_dim) are None then sample object should be given

        :param intensity_calculator:
            Class that is used to compute intensity of spectra via temperature/ time/ hamiltonian parameters.
            Default is None
            If it is None then it will be initialized as TimeIntensityCalculator

        :param populator:
            Object that is used to compute part intensity due to the difference of population between levels.
            By default, it is initialized as LevelBasedPopulator and uses solution of kinetic equation for populations

        :param spectra_integrator:
            Class to integrate the resonance lines to get the spectrum.

        :param harmonic: Harmonic of spectra: 1 is derivative, 0 is absorbance. Default is 0.

        :param post_spectra_processor:
            Class to post process resulted resonance data (fields, intensities, width):
            integration, mesh mapping and so on. Default post_spectra_processor is powder spectra processor

        :param temperature: The temperature of an experiment. If populator is not None it takes from it

        :param recompute_spin_parameters:
            Recompute spin parameters in __call__ methods. For time resolved spectra creator is False

        :param computational_details: ComputationalDetails
            computational_details : ComputationalDetails, optional
            Configuration object that governs the numerical aspects of spectrum generation.
            Contains the following fields:

            - **chunk_size** (`int`, default=128):
              Number of magnetic field points processed per integration batch.
              Larger values improve throughput but increase memory consumption.

            - **res_field_r_tol** (`float`, default=1e-5):
              Relative tolerance for adaptive subdivision of field intervals during resolution enhancement.

            - **res_field_split_max_iterations** (`int`, default=20):
              Maximum depth of recursive field-sector splitting.

            - **intensity_threshold** (`float`, default=1e-2):
              Minimum relative intensity (as a fraction of the strongest transition) required for
              transition to be included.

            -for other parameters meaning read
             docs of :class:'mars.spectra_manager.spectra_manager.ComputationalDetails'

        :param inference_mode: bool
            If inference_mode is True, then forward method will be performed under with torch.inference_mode():

        :param output_eigenvector: Optional[bool]
            If True, computes and returns the full system eigenvector. If False, returns None.
            For stationary computations, the default is False; for time-resolved simulations, the default is True.
            If set to None, the value is inferred automatically based on the population dynamics logic.

        :param context: Optional[context]
            The instance of BaseContext which describes the relaxation mechanism.
            It can have the initial population logic, transition between energy levels, dephasings, driven transition,
            out system transitions. For more complicated scenario the full relaxation superoperator can be used.

        :param hamiltonian_mode: str, HamComputationMethod
         {"secular", "direct"} or HamComputationMethod, default="direct"
            Method for Hamiltonian eigen values, eigen vectors, resonance filed computation:
            - "secular": uses secular approximation (faster)
            - "direct": use the general algorithm: res-field or res-freq (slower, the most general)

        :param output_mode: str, OutputSpectraMode:
        Controls the organization of the computed spectrum.

        "total": returns the conventional summed spectrum over all allowed transitions (default behavior).

        "transitions": returns dict of lvl_down, lvl_up and spectrum,
        where each slice corresponds to the contribution of an individual transition
        (e.g., between specific energy levels).
        Default is "total".

        :param device: cpu / cuda. Base device for computations.

        :param dtype: float32 / float64
        Base dtype for all types of operations. If complex parameters is used,
        they will be converted in complex64, complex128
        """
        super().__init__(freq, sample, spin_system_dim, batch_dims, mesh, intensity_calculator, populator,
                         spectra_integrator, harmonic, post_spectra_processor,
                         temperature, recompute_spin_parameters,
                         computational_details,
                         inference_mode, output_eigenvector, context, hamiltonian_mode, output_mode,
                         device=device, dtype=dtype)

    def __call__(self, sample: spin_model.MultiOrientedSample, fields: torch.Tensor, time: torch.Tensor, **kwargs):
        """
        :param sample: MultiOrientedSample object.

        :param fields: The magnetic fields in Tesla units
        :param time: Time to compute time resolved spectra
        :param kwargs:
        :return: spectra or some resonance data depending on the output_mode
        """
        return super().__call__(sample, fields, time, **kwargs)

    def _get_intensity_calculator(self, intensity_calculator: tp.Optional[BaseResIntensityCalculator],
                                  temperature: float,
                                  populator: tp.Optional[BaseTimeDepPopulator],
                                  context: tp.Optional[contexts.BaseContext],
                                  computational_details: ComputationalDetails,
                                  device: torch.device, dtype: torch.dtype):
        if intensity_calculator is None:
            return TimeIntensityCalculator(
                self.spin_system_dim, temperature, populator, context,
                disordered=self.mesh.disordered,
                computational_details=computational_details,
                device=device, dtype=dtype
            )
        else:
            return intensity_calculator

    def _get_param_specs(self) -> list[ParamSpec]:
        params = [
            ParamSpec("scalar", torch.long),
            ParamSpec("scalar", torch.long),
            ParamSpec("vector", torch.float32),
            ParamSpec("vector", torch.complex64),
            ParamSpec("vector", torch.complex64)
            ]
        return params

    def _add_to_mask_additional(self,
                                lvl_down: torch.Tensor, lvl_up: torch.Tensor,
                                resonance_energies: torch.Tensor,
                                vector_down: torch.Tensor, vector_up: torch.Tensor):
        return lvl_down, lvl_up, resonance_energies, vector_down, vector_up

    def _postcompute_batch_data(self, sample: spin_model.BaseSample,
                                res_fields: torch.Tensor, intensities: torch.Tensor, width: torch.Tensor,
                                F: torch.Tensor, Gx: torch.Tensor, Gy: torch.Tensor,
                                Gz: torch.Tensor, full_system_vectors: tp.Optional[torch.Tensor],
                                time: torch.Tensor, *extras, **kwargs):
        lvl_down, lvl_up, resonance_energies, vector_down, vectors_up, *extras = extras

        res_fields, resonance_energies = self._nan_to_zeros(res_fields, resonance_energies)

        population = self.intensity_calculator.compute_population(
            time, res_fields, lvl_down, lvl_up,
            resonance_energies, vector_down, vectors_up, full_system_vectors, *extras
        )
        intensities = (intensities.unsqueeze(-3) * population)
        return res_fields, intensities, width

    def _init_spectra_processor(self,
                                spectra_integrator: tp.Optional[BaseSpectraIntegrator],
                                harmonic: int,
                                post_spectra_processor: PostSpectraProcessing,
                                computational_details: ComputationalDetails,
                                output_mode: OutputSpectraMode,
                                device: torch.device,
                                dtype: torch.dtype) -> BaseResProcessing:
        """Create a processor for integrating and post-processing spectral data.
        :param spectra_integrator: Custom integrator; if None, one is auto-selected.
        :param harmonic: Spectral harmonic (0 = absorption, 1 = first derivative).
        :param post_spectra_processor: Line-broadening and convolution handler.
        :param computational_details: The details of final spectral integration and spectra processing. For example,

            -integration_natural_width : float, default=1e-6
                Minimum intrinsic linewidth added to every transition. Measures in FWHM
                Prevents division-by-zero or extreme sharpening when user-provided widths are
                very small or zero. Also it can be used as substitution for ordinary gaussian broadaning in the sample.

            - integration_gaussian_method : str, default="exp"
                Method used to evaluate the Gaussian function exp(-x²) during final integration:
                - "exp": uses exact PyTorch exponential (higher accuracy),
                - "approx": uses a fast 6th-order rational approximation (see ``gaussian_approx``).

            - chunk_size (`int`, default=128):
              Number of magnetic field points processed per integration batch.
              Larger values improve throughput but increase memory consumption.

            -for other parameters specifications, read
             docs of :class:'mars.spectra_manager.spectra_manager.ComputationalDetails'

        :param output_mode: The output mode for spectra computation
        :return: Initialized spectra processor instance.
        """
        if self.mesh.disordered:
            return PowderTimeProcessing(self.mesh, spectra_integrator, harmonic, post_spectra_processor,
                                        computational_details=computational_details,
                                        output_mode=output_mode,
                                        device=device, dtype=dtype)
        else:
            return CrystalTimeProcessing(self.mesh, spectra_integrator, harmonic, post_spectra_processor,
                                         computational_details=computational_details,
                                         output_mode=output_mode,
                                         device=device, dtype=dtype)

    def _init_recompute_spin_flag(self) -> bool:
        """
        If flag is False: resfield data is cached.

        If flag is True: resfield recomputes every time
        :return:
        """
        return False

    def _get_intensity_mask(self,
                            intensities: torch.Tensor,
                            res_manifold: torch.Tensor,
                            lvl_down: torch.Tensor,
                            lvl_up: torch.Tensor,
                            energies: torch.Tensor,
                            vector_down: torch.Tensor,
                            vector_up: torch.Tensor,
                            full_system_vectors: tp.Optional[torch.Tensor]
                            ) -> torch.Tensor:
        """
        Generate a boolean mask to filter out transitions with intensities below a threshold.

        The mask evaluates to True if *either* of the following conditions is met:
        1. The absolute intensity multiplied by polarization, normalized by its global
           maximum, exceeds `self.threshold`.
        2. The absolute intensity alone, normalized by its global maximum, exceeds
           `self.threshold`.

        :param intensities: Tensor of transition intensities.
        :param res_manifold: Resonance fields / frequencies (passed for signature consistency).
        :param lvl_down: Energy levels of the lower states.
        :param lvl_up: Energy levels of the upper states.
        :param energies: Resonance energies (passed for signature consistency).
        :param vector_down: Eigenvectors of the lower energy states.
        :param vector_up: Eigenvectors of the upper energy states.
        :param full_system_vectors: Eigenvectors of the full spin system.
        :return: A boolean tensor indicating which transitions have sufficient
                 intensity to be kept.
        """
        res_manifold, energies = self._nan_to_zeros(res_manifold, energies)

        polarization = self.intensity_calculator.compute_stationary_polarization(
            res_manifold, lvl_down, lvl_up, energies, vector_down, vector_up,
            full_system_vectors
        )
        abs_int = intensities.abs()
        abs_pol = polarization.abs()

        abs_int_with_pop = abs_int * abs_pol
        max_with_pop = abs_int_with_pop.max()
        thresh_with_pop = self.threshold * max_with_pop
        mask_1 = abs_int_with_pop > thresh_with_pop

        max_raw = abs_int.max()
        thresh_raw = self.threshold * max_raw
        mask_2 = abs_int > thresh_raw

        combined_mask = mask_1 | mask_2

        if combined_mask.ndim > 1:
            leading_size = combined_mask.numel() // combined_mask.shape[-1]
            return combined_mask.view(leading_size, -1).any(dim=0)
        return combined_mask


class CoupledTimeSpectra(TruncTimeSpectra):
    """Compute time-resolved EPR spectra for populations relaxation formalism.

    Provides the complete pipeline for computing EPR spectra from spin Hamiltonian:
    1. Compute resonance fields/frequencies by diagonalizing Hamiltonian
    2. Calculate transition intensities from matrix elements and populations
    3. Compute linewidths from strain tensors
    4. Integrate over orientation mesh (for powder samples)
    5. Apply line broadening (Gaussian/Lorentzian/Voigt)
    """
    def _init_output_eigenvector(self, output_eigenvector: tp.Optional[bool],
                                 context: tp.Optional[contexts.BaseContext]) -> bool:
        if output_eigenvector is not None:
            return output_eigenvector
        else:
            return True


class DensityTimeSpectra(CoupledTimeSpectra):
    """Compute time-resolved EPR spectra for density matrix relaxation
    formalism.

    Default the rotating wave approximation is used

    Provides the complete pipeline for computing EPR spectra from spin Hamiltonian:
    1. Compute resonance fields/frequencies by diagonalizing Hamiltonian
    2. Calculate transition intensities from matrix elements and populations
    3. Compute linewidths from strain tensors
    4. Integrate over orientation mesh (for powder samples)
    5. Apply line broadening (Gaussian/Lorentzian/Voigt)
    """

    def __init__(self,
                 freq: tp.Union[float, torch.Tensor],
                 sample: tp.Optional[spin_model.MultiOrientedSample] = None,
                 spin_system_dim: tp.Optional[int] = None,
                 batch_dims: tp.Optional[tp.Union[int, tuple]] = None,
                 mesh: tp.Optional[mesher.BaseMesh] = None,
                 intensity_calculator: tp.Optional[tp.Callable] = None,
                 populator: tp.Optional[tp.Union[BaseTimeDepPopulator, str]] = None,
                 spectra_integrator: tp.Optional[BaseSpectraIntegrator] = None,
                 harmonic: int = 0,
                 post_spectra_processor: PostSpectraProcessing = PostSpectraProcessing(),
                 temperature: tp.Optional[tp.Union[float, torch.Tensor]] = 293,
                 recompute_spin_parameters: bool = True,
                 computational_details: ComputationalDetails = ComputationalDetails(),
                 inference_mode: bool = True,
                 output_eigenvector: tp.Optional[bool] = None,
                 context: tp.Optional[contexts.BaseContext] = None,
                 hamiltonian_mode: tp.Union[str, HamComputationMethod] = HamComputationMethod.SECULAR,
                 output_mode: tp.Union[str, OutputSpectraMode] = OutputSpectraMode.TOTAL,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32,
                 ):

        """Note that by default these spin systems (energies, vectors, etc.)
        are calculated once and then cached.

        Default harmoinc is None

        :param freq: Resonance frequency of experiment

        :param sample: MultiOrientedSample.
            It is just an example of spin system to extract meta information (spin_system_dim, batch_dims, mesh)
            If it is None, then spin_system_dim, batch_dims, mesh should be given

        :param spin_system_dim: The size of spin system. Default is None
        :param batch_dims: The number of batch dimensions. Default is None
        :param mesh: Mesh object. Default is None
            If (mesh, batch_dims, spin_system_dim) are None then sample object should be given

        :param intensity_calculator:
            Class that is used to compute intensity of spectra via temperature/ time/ hamiltonian parameters.
            Default is None
            If it is None then it will be initialized as TimeIntensityCalculator

        :param populator:
            Object used to compute the solution of the Liouville–von Neumann equation.
            If None`(the default), the solver uses the rotating-wave approximation (RWA)
            with RWADensityPopulator to evolve the density matrix.
            To use full propagator-based dynamics instead, pass the string 'propagator',
            which selects PropagatorDensityPopulator.

        :param spectra_integrator:
            Class to integrate the resonance lines to get the spectrum.

        :param harmonic: Harmonic of spectra: 1 is derivative, 0 is absorbance. Default is 0.

        :param post_spectra_processor:
            Class to post process resulted resonance data (fields, intensities, width):
            integration, mesh mapping and so on. Default post_spectra_processor is powder spectra processor

        :param temperature: The temperature of an experiment. If populator is not None it takes from it

        :param recompute_spin_parameters:
            Recompute spin parameters in __call__ methods. For time resolved spectra creator is False

        :param computational_details: ComputationalDetails
            computational_details : ComputationalDetails, optional
            Configuration object that governs the numerical aspects of spectrum generation.
            Contains the following fields:

            - **chunk_size** (`int`, default=128):
              Number of magnetic field points processed per integration batch.
              Larger values improve throughput but increase memory consumption.

            - **res_field_r_tol** (`float`, default=1e-5):
              Relative tolerance for adaptive subdivision of field intervals during resolution enhancement.

            - **res_field_split_max_iterations** (`int`, default=20):
              Maximum depth of recursive field-sector splitting.

            - **intensity_threshold** (`float`, default=1e-2):
              Minimum relative intensity (as a fraction of the strongest transition) required for
              transition to be included.

            -for other parameters meaning read
             docs of :class:'mars.spectra_manager.spectra_manager.ComputationalDetails'

        :param inference_mode: bool
            If inference_mode is True, then forward method will be performed under with torch.inference_mode():

        :param output_eigenvector: Optional[bool]
            If True, computes and returns the full system eigenvector. If False, returns None.
            For stationary computations, the default is False; for time-resolved simulations, the default is True.
            If set to None, the value is inferred automatically based on the population dynamics logic.

        :param context: Optional[context]
            The instance of BaseContext which describes the relaxation mechanism.
            It can have the initial population logic, transition between energy levels, dephasings, driven transition,
            out system transitions. For more complicated scenario the full relaxation superoperator can be used.

        :param hamiltonian_mode: str, HamComputationMethod
         {"secular", "direct"} or HamComputationMethod, default="secular"
            Method for Hamiltonian eigen values, eigen vectors, resonance filed computation:
            - "secular": uses secular approximation (faster)
            - "direct": use the general algorithm: res-field or res-freq (slower, the most general)

            For this class default is secular because RWA is default method.

        :param output_mode: str, OutputSpectraMode:
        Controls the organization of the computed spectrum.

        "total": returns the conventional summed spectrum over all allowed transitions (default behavior).

        "transitions": returns dict of lvl_down, lvl_up and spectrum,
        where each slice corresponds to the contribution of an individual transition
        (e.g., between specific energy levels).
        Default is "total".

        :param device: cpu / cuda. Base device for computations.

        :param dtype: float32 / float64
        Base dtype for all types of operations. If complex parameters is used,
        they will be converted in complex64, complex128
        """
        super().__init__(freq, sample, spin_system_dim, batch_dims, mesh, intensity_calculator, populator,
                         spectra_integrator, harmonic, post_spectra_processor,
                         temperature, recompute_spin_parameters,
                         computational_details,
                         inference_mode, output_eigenvector, context, hamiltonian_mode, output_mode,
                         device=device, dtype=dtype)

    def _postcompute_batch_data(self, sample: spin_model.BaseSample,
                                res_fields: torch.Tensor, intensities: torch.Tensor, width: torch.Tensor,
                                F: torch.Tensor, Gx: torch.Tensor, Gy: torch.Tensor,
                                Gz: torch.Tensor, full_system_vectors: tp.Optional[torch.Tensor],
                                time: torch.Tensor, *extras, **kwargs):
        lvl_down, lvl_up, resonance_energies, vector_down, vectors_up, *extras = extras
        Sz = sample.base_spin_system.get_electron_z_operator()

        res_fields, resonance_energies = self._nan_to_zeros(res_fields, resonance_energies)

        population = self.intensity_calculator.compute_population(
            time, res_fields, lvl_down, lvl_up,
            resonance_energies, vector_down, vectors_up,
            full_system_vectors,
            F, Gx, Gy, Gz, Sz,
            self.resonance_parameter, *extras
        )
        intensities = population
        return res_fields, intensities, width

    def _get_intensity_calculator(self, intensity_calculator: tp.Optional[BaseResIntensityCalculator],
                                  temperature: float,
                                  populator: tp.Optional[tp.Union[BaseTimeDepPopulator, str]],
                                  context: tp.Optional[contexts.BaseContext],
                                  computational_details: ComputationalDetails,
                                  device: torch.device, dtype: torch.dtype):
        if intensity_calculator is None:
            return TimeDensityCalculator(
                self.spin_system_dim, temperature, populator, context,
                disordered=self.mesh.disordered,
                computational_details=computational_details,
                device=device, dtype=dtype
            )
        else:
            return intensity_calculator


class StationaryFreqSpectra(StationarySpectra):
    """Compute stationary EPR spectra at frequency domain.

    Default the rotating wave approximation is used

    Provides the complete pipeline for computing EPR spectra from spin Hamiltonian:
    1. Compute resonance fields/frequencies by diagonalizing Hamiltonian
    2. Calculate transition intensities from matrix elements and populations
    3. Compute linewidths from strain tensors
    4. Integrate over orientation mesh (for powder samples)
    5. Apply line broadening (Gaussian/Lorentzian/Voigt)
    """

    def __init__(self,
                 field: tp.Union[float, torch.Tensor],
                 sample: tp.Optional[spin_model.MultiOrientedSample] = None,
                 spin_system_dim: tp.Optional[int] = None,
                 batch_dims: tp.Optional[tp.Union[int, tuple]] = None,
                 mesh: tp.Optional[mesher.BaseMesh] = None,
                 intensity_calculator: tp.Optional[BaseResIntensityCalculator] = None,
                 populator: tp.Optional[StationaryPopulator] = None,
                 spectra_integrator: tp.Optional[BaseSpectraIntegrator] = None,
                 harmonic: int = 1,
                 post_spectra_processor: PostSpectraProcessing = PostSpectraProcessing(),
                 temperature: tp.Optional[tp.Union[float, torch.Tensor]] = 293,
                 recompute_spin_parameters: bool = True,
                 computational_details: ComputationalDetails = ComputationalDetails(),
                 inference_mode: bool = True,
                 output_eigenvector: tp.Optional[bool] = None,
                 context: tp.Optional[contexts.BaseContext] = None,
                 hamiltonian_mode: tp.Union[str, HamComputationMethod] = HamComputationMethod.DIRECT,
                 output_mode: tp.Union[str, OutputSpectraMode] = OutputSpectraMode.TOTAL,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32
                 ):
        """
        :param field: Resonance field of experiment.

        :param sample: MultiOrientedSample.
            It is just an example of spin system to extract meta information (spin_system_dim, batch_dims, mesh)
            If it is None, then spin_system_dim, batch_dims, mesh should be given

        :param spin_system_dim: The size of spin system. Default is None
        :param batch_dims: The number of batch dimensions. Default is None
        :param mesh: Mesh object. Default is None
            If (mesh, batch_dims, spin_system_dim) are None then sample object should be given

        :param intensity_calculator:
            Class that is used to compute intensity of spectra via temperature/ time/ hamiltonian parameters.
            Default is None
            If it is None then it will be initialized as StationaryIntensityCalculator

        :param populator:
            Class that is used to compute part intensity due to population of levels. Default is None
            If intensity_calculator is None or StationaryIntensityCalculator
            then it will be initialized as StationaryPopulator
            In this case the population is given as Boltzmann population

        :param spectra_integrator:
            Class to integrate the resonance lines to get the spectrum.

        :param harmonic: Harmonic of spectra: 1 is derivative, 0 is absorbance. Default is 1.

        :param post_spectra_processor:
            Class to post process resulted resonance data (fields, intensities, width):
            integration, mesh mapping and so on. Default post_spectra_processor is powder spectra processor

        :param temperature: The temperature of an experiment. If populator is not None it takes from it

        :param recompute_spin_parameters:
            Recompute spin parameters in __call__ methods. For stationary creator is True.

        :param computational_details: ComputationalDetails
            computational_details : ComputationalDetails, optional
            Configuration object that governs the numerical aspects of spectrum generation.
            Contains the following fields:

            - **chunk_size** (`int`, default=128):
              Number of magnetic field points processed per integration batch.
              Larger values improve throughput but increase memory consumption.

            - **res_field_r_tol** (`float`, default=1e-5):
              It is not suppotred for StationaryFreqSpectra

            - **res_field_split_max_iterations** (`int`, default=20):
              It is not suppotred for StationaryFreqSpectra

            - **intensity_threshold** (`float`, default=1e-2):
              Minimum relative intensity (as a fraction of the strongest transition) required for
              transition to be included.

            -for other parameters specifications, read
             docs of :class:'mars.spectra_manager.spectra_manager.ComputationalDetails'

        :param inference_mode: bool
            If inference_mode is True, then forward method will be performed under with torch.inference_mode():

        :param output_eigenvector: Optional[bool]
            If True, computes and returns the full system eigenvector. If False, returns None.
            For stationary computations, the default is False; for time-resolved simulations, the default is True.
            If set to None, the value is inferred automatically based on the population dynamics logic.

        :param context: Optional[context]
            The instance of BaseContext which describes the relaxation mechanism.
            It can have the initial population logic, transition between energy levels, dephasings, driven transition,
            out system transitions. For more complicated scenario the full relaxation superoperator can be used.

        :param hamiltonian_mode: str, HamComputationMethod
         {"secular", "direct"} or HamComputationMethod, default="direct"
            Method for Hamiltonian eigen values, eigen vectors, resonance filed computation:
            FreqSpectra supports only direct conputation of spectra.

        :param output_mode: str, OutputSpectraMode:
        Controls the organization of the computed spectrum.

        "total": returns the conventional summed spectrum over all allowed transitions (default behavior).

        "transitions": returns dict of lvl_down, lvl_up and spectrum,
        where each slice corresponds to the contribution of an individual transition
        (e.g., between specific energy levels).
        Default is "total".
        """
        super().__init__(field, sample, spin_system_dim, batch_dims, mesh, intensity_calculator,
                         populator, spectra_integrator, harmonic, post_spectra_processor,
                         temperature, recompute_spin_parameters,
                         computational_details,
                         inference_mode, output_eigenvector, context, hamiltonian_mode, output_mode,
                         device=device, dtype=dtype)

    def _init_res_algorithm(self,
                            output_eigenvector: bool,
                            hamiltonian_mode: HamComputationMethod,
                            computational_details: ComputationalDetails,
                            device: torch.device, dtype: torch.dtype):
        """Instantiate the resonance field computation algorithm.

        Selects an appropriate Hamiltonian eigen data backend based on
        whether full eigenvectors are needed and whether some approximation is used.

        :param output_eigenvector: Whether full system eigenvectors should be computed.
        :param hamiltonian_mode: the method to use to compute the Hamiltonian eigen data.
        :param computational_details: The computational details to create EPR spectra:
                accuracy, number of iterations, and so on.

        :return: Configured resonance field solver.
        """
        return res_freq_algorithm.ResFreq(
            spin_system_dim=self.spin_system_dim,
            mesh_size=self.mesh_size,
            batch_dims=self.batch_dims,
            output_full_eigenvector=output_eigenvector,
            device=device,
            dtype=dtype
        )

    def _get_intensity_calculator(self,
                                  intensity_calculator: tp.Optional[BaseResIntensityCalculator],
                                  temperature: float,
                                  populator: tp.Optional[tp.Union[BasePopulator, str]],
                                  context: tp.Optional[contexts.BaseContext],
                                  computational_details: ComputationalDetails,
                                  device: torch.device, dtype: torch.dtype):
        """Instantiate or return the intensity calculator for transition strengths.

        :param intensity_calculator: Pre-configured calculator; if None, one is created.
        :param temperature: Sample temperature in Kelvin.
        :param populator: Population model or identifier.
        :param context: Relaxation/population dynamics context.
        :param computational_details: ComputationalDetails
            computational_details : ComputationalDetails, optional
            Configuration object that governs the numerical aspects of spectrum generation.
        :param device: Computation device.
        :param dtype: Floating-point precision.
        :return: Ready-to-use intensity calculator.
        """
        if intensity_calculator is None:
            return StationaryIntensityCalculator(
                self.spin_system_dim, temperature, populator, context, device=device, dtype=dtype
            )
        else:
            return intensity_calculator

    def __call__(self,
                sample: spin_model.MultiOrientedSample,
                freq: torch.Tensor, time: tp.Optional[torch.Tensor] = None, **kwargs):
        """
        :param sample: MultiOrientedSample object.
        :param freq: The frequency in Hz units
        :param time: It is used only for time resolved spectra
        :param kwargs:
        :return: spectra or some resonance data depending on the output_mode
        """
        return super().__call__(sample, freq, time)

    def compute_parameters(self, sample: spin_model.MultiOrientedSample,
                           F: torch.Tensor,
                           Gx: torch.Tensor,
                           Gy: torch.Tensor,
                           Gz: torch.Tensor,
                           res_freq: torch.Tensor,
                           lvl_down: torch.Tensor, lvl_up: torch.Tensor,
                           resonance_energies: torch.Tensor,
                           vector_down: torch.Tensor, vector_up: torch.Tensor,
                           full_system_vectors: tp.Optional[torch.Tensor]) ->\
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, tp.Optional[torch.Tensor], tuple[tp.Any]]:
        """
        :param sample: The sample which transitions must be found.

        :param F: Magnetic free part of spin Hamiltonian H = F + B * G
        :param Gx: x-part of Hamiltonian Zeeman Term
        :param Gy: y-part of Hamiltonian Zeeman Term
        :param Gz: z-part of Hamiltonian Zeeman Term

        :param res_freq: Resonance frequencies. The shape os [..., N]

        :param lvl_down:
            Energy levels of lower states from which transitions occur.
            Shape: [..., N], where
            N is the number of energy levels.

        :param lvl_up:
            Energy levels of upper states to which transitions occur.
            Shape: [..., N], where
            N is the number of energy levels.

        :param resonance_energies:
            Energies of spin states. The shape is [..., N]

        :param vector_down:
            Eigenvectors of the lower energy states. The shape is [...., M, N],
            where M is number of transitions, N is number of levels

        :param vector_up:
            Eigenvectors of the upper energy states.The shape is [...., M, N],
            where M is number of transitions, N is number of levels

        :param full_system_vectors: Eigen vector of each level of a spin system. The shape os [..., N, N]

        :return: tuple of the next data
         - Resonance fields
         - Intensities of transitions
         - Width of transition lines
         - Eigen vectors of all system levels or None
         - extras parameters computed in _compute_additional
        """

        intensities = self.intensity_calculator.compute_intensity(
            Gx, Gy, Gz, res_freq, lvl_down, lvl_up, resonance_energies, vector_down, vector_up, full_system_vectors
        )
        intensities = torch.nan_to_num(intensities, nan=0.0, out=intensities)

        intensities_mask = self._get_intensity_mask(
            intensities, res_freq, lvl_down, lvl_up, resonance_energies, vector_down, vector_up, full_system_vectors
        )

        intensities = intensities[..., intensities_mask]

        extras = self._add_to_mask_additional(lvl_down, lvl_up, resonance_energies, vector_down, vector_up)

        extras = self._mask_components(intensities_mask, *extras)
        full_system_vectors = self._mask_full_system_eigenvectors(intensities_mask, full_system_vectors)

        res_fields = res_freq[..., intensities_mask]
        vector_u = vector_down[..., intensities_mask, :]
        vector_v = vector_up[..., intensities_mask, :]

        intensities = intensities / self.intensity_std
        width = self.broader(sample, vector_u, vector_v, res_fields)

        extras = self._compute_additional(
            sample, F, Gx, Gy, Gz, full_system_vectors, *extras
        )

        return res_fields, intensities, width, full_system_vectors, *extras
