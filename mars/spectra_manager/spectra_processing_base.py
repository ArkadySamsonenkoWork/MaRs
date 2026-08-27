import typing as tp
from abc import ABC, abstractmethod

import torch
import torch.fft as fft
import torch.nn as nn

from .. import mesher

from .spectral_integration import BaseSpectraIntegrator,\
    SphereSpectraIntegrator, MeanIntegrator, AxialSpectraIntegrator

from .utils import ComputationalDetails, OutputSpectraMode


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
             docs of :class:'mars.spectra_manager.utils.ComputationalDetails'

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
              :class:'mars.spectra_manager.utils.ComputationalDetails'

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
             docs of :class:'mars.spectra_manager.utils.ComputationalDetails'

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
              :class:'mars.spectra_manager.utils.ComputationalDetails'

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
             docs of :class:'mars.spectra_manager.utils.ComputationalDetails'

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
              :class:'mars.spectra_manager.utils.ComputationalDetails'

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
