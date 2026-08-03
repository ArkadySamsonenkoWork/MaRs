import typing as tp
from dataclasses import dataclass


import torch
import torch.nn as nn

from .. import mesher
from .. import spin_model

from .spectral_integration import BaseSpectraIntegrator
from ..population import StationaryPopulatorExpanded, BasePopulator
from ..population import contexts

from .spectra_manager import CrystalStationaryProcessing, PowderStationaryProcessing,\
    CrystalTimeProcessing, PowderTimeProcessing, Broadener,\
    PostSpectraProcessing, ComputationalDetails, OutputSpectraMode, HamComputationMethod, \
    BaseResIntensityCalculator, StationaryIntensityCalculator, \
    StationarySpectra


class BroadenerExpanded(Broadener):
    """
    Extended version of `Broadener` class that supports additional independent batch dimensions
    for strain contributions.

    In the base `Broadener`, the strain tensors in the sample share the same batch
    dimension as the eigenvectors and magnetic fields. In this expanded version, the
    sample may contain an extra leading dimension (e.g., for multiple independent
    Hamiltonian strain configurations), which is handled by unsqueezing additional
    dimensions during the residual broadening addition.

    **Key difference**:
    - `add_hamiltonian_strain` now adds `hamiltonian_width` with extra dimensions
      unsqueezed to broadcast correctly against
      the squared width tensor that already includes the extra batch dimensions.
    """
    def add_hamiltonian_strain(self, sample: spin_model.MultiOrientedSampleExpandedStrain, squared_width: torch.Tensor):
        """Adds residual broadening due to unresolved interactions.

        :param sample: The MultiOrientedSample object
        :param squared_width: The square of gaussian broadening
        :return: Total gaussian broadening as
        """
        hamiltonian_width = sample.build_ham_strain().unsqueeze(-1).square()
        return (squared_width.unsqueeze(0) + hamiltonian_width.unsqueeze(1)).sqrt()


@dataclass
class ExpandedProcessingDetails:
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


class _AutoFieldAxisMixin(nn.Module):
    """
    Internal mixin that provides automatic magnetic field axis computation
    based on resonance fields and linewidths.

    This mixin is designed to be used with `PowderStationaryProcessing`,
    `CrystalStationaryProcessing`, `PowderTimeProcessing`, and
    `CrystalTimeProcessing`. It adds the ability to generate a dynamic field
    sweep range without requiring an external `fields` tensor.

    The field axis is determined by:
        1. Finding the global min and max of resonance fields across all
           transitions and orientations.
        2. Estimating the necessary spectral width using both the resonance
           field span and the maximum linewidth (scaled by `width_factor`).
        3. Adjusting the min/max fields with `spectral_width_part` to create
           margins, and clamping to absolute bounds `min_exp_field` /
           `max_exp_field`.
        4. Creating a linearly spaced field axis with `num_points` points.

    :cvar num_points: Number of points in the generated field axis.
    :cvar spectral_width_part: Fraction of the estimated spectral width used for margins.
    :cvar width_factor: Multiplier for the maximum linewidth.
    :cvar min_exp_field: Absolute lower bound for the field sweep.
    :cvar max_exp_field: Absolute upper bound for the field sweep.
    :cvar width_cutoff: Only linewidths > this value are considered for width extension.
    """
    def _init_field_axis_buffers(self, num_points: int, spectral_width_part: float,
                                 width_factor: float, min_exp_field: float, max_exp_field: float,
                                 width_cutoff: float, device: torch.device, dtype: torch.dtype) -> None:
        """
        Register persistent buffers for field‑axis parameters.

        :param num_points: Number of points in the generated field axis.
        :param spectral_width_part: Fraction of the estimated spectral width used to determine margins.
        :param width_factor: Multiplier applied to the maximum linewidth to extend the sweep.
        :param min_exp_field: Absolute minimum field value (lower clamp).
        :param max_exp_field: Absolute maximum field value (upper clamp).
        :param width_cutoff: Linewidth threshold (Tesla); linewidths above this value are considered.
        :param device: Target device for buffers.
        :param dtype: Data type for buffers.
        """
        self.register_buffer("num_points", torch.tensor(num_points, device=device))
        self.register_buffer("spectral_width_part", torch.tensor(spectral_width_part, device=device, dtype=dtype))
        self.register_buffer("width_factor", torch.tensor(width_factor, device=device, dtype=dtype))
        self.register_buffer("min_exp_field", torch.tensor(min_exp_field, device=device, dtype=dtype))
        self.register_buffer("max_exp_field", torch.tensor(max_exp_field, device=device, dtype=dtype))
        self.register_buffer("width_cutoff", torch.tensor(width_cutoff, device=device, dtype=dtype))

    def _get_new_field(self, res_fields: torch.Tensor, width: torch.Tensor,
                       intensities: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute a dynamic field axis based on resonance field distribution and widths.

        :param res_fields: Resonance fields, shape [..., num_transitions, num_simplices, 3] (after transformation)
        :param width: Linewidths, shape [..., num_transitions, num_simplices]
        :param intensities: Intensities, shape [..., num_transitions, num_simplices] (unused but kept for signature)
        :return: (fields, min_pos_batch, max_pos_batch)
                 fields: field axis [batch..., num_points]
                 min_pos_batch, max_pos_batch: lower and upper field limits per batch element
        """
        dims = res_fields.dim()
        batch_dims = tuple(range(max(dims - 2, 0), dims))

        min_pos_batch = torch.amin(res_fields, dim=batch_dims)
        max_pos_batch = torch.amax(res_fields, dim=batch_dims)
        mean_pos = (max_pos_batch + min_pos_batch) / 2

        width_criteria = width.clone()
        width_criteria[width > self.width_cutoff] = 0.0
        max_orient_width = torch.amax(width_criteria, dim=-1)

        nature_spectra_width = torch.max(max_pos_batch - min_pos_batch, max_orient_width * self.width_factor)

        min_pos_batch = mean_pos - nature_spectra_width / (2 * self.spectral_width_part)
        max_pos_batch = mean_pos + nature_spectra_width / (2 * self.spectral_width_part)

        min_pos_batch = torch.max(min_pos_batch, self.min_exp_field)
        max_pos_batch = torch.min(max_pos_batch, self.max_exp_field)

        steps = torch.linspace(0, 1, int(self.num_points), device=res_fields.device, dtype=res_fields.dtype)
        fields = steps * (max_pos_batch - min_pos_batch).unsqueeze(-1) + min_pos_batch.unsqueeze(-1)
        return fields, min_pos_batch, max_pos_batch


class PowderStationaryProcessingExpanded(_AutoFieldAxisMixin, PowderStationaryProcessing):
    """
    Expanded version of `PowderStationaryProcessing` that automatically determines
    the magnetic field axis from the resonance field distribution and linewidths.

    This class inherits all functionality of `PowderStationaryProcessing` and adds
    dynamic field‑axis generation. Instead of requiring an external `fields` tensor,
    the field sweep range is computed using the min/max resonance fields and the
    maximum linewidth (after a cutoff), with user‑controllable margins and clamping.

    The forward method returns a tuple `(spectrum, (min_field, max_field))`
    instead of just the spectrum.
    """
    def __init__(self,
                 mesh: mesher.BaseMeshPowder,
                 spectra_integrator: tp.Optional[BaseSpectraIntegrator] = None,
                 harmonic: int = 1,
                 post_spectra_processor: PostSpectraProcessing = PostSpectraProcessing(),
                 computational_details: ComputationalDetails = ComputationalDetails(),
                 output_mode: OutputSpectraMode = OutputSpectraMode.TOTAL,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32,
                 num_points: int = 4_000,
                 spectral_width_part: float = 0.6,
                 width_factor: float = 3.0,
                 min_exp_field: float = 0.0,
                 max_exp_field: float = 2.0,
                 width_cutoff: float = 0.5):
        """
        :param mesh: Powder mesh object (BaseMeshPowder).
        :param spectra_integrator: Optional custom integrator.
        :param harmonic: Spectral harmonic (0 = absorption, 1 = first derivative).
        :param post_spectra_processor: Post‑processing object for line broadening.
        :param computational_details: Details for integration (chunk size, natural width, etc.)
        :param output_mode: Must be `OutputSpectraMode.TOTAL`.
        :param device: Computation device.
        :param dtype: Floating‑point data type.
        :param num_points: Number of points in the generated field axis.
        :param spectral_width_part: Fraction of the estimated spectral width used to determine the sweep window.
        :param width_factor: Multiplier for the maximum linewidth to extend the sweep.
        :param min_exp_field: Absolute minimum field value (lower bound clamp).
        :param max_exp_field: Absolute maximum field value (upper bound clamp).
        :param width_cutoff: Only linewidths above this value (in Tesla) are considered.
        """
        super().__init__(mesh, spectra_integrator, harmonic, post_spectra_processor,
                         computational_details, output_mode, device, dtype)
        if output_mode != OutputSpectraMode.TOTAL:
            raise NotImplementedError(f"output_mode is supported only Total for expanded processing. "
                                      f"You have used {output_mode}")
        self._init_field_axis_buffers(num_points, spectral_width_part, width_factor,
                                      min_exp_field, max_exp_field, width_cutoff, device, dtype)

    def forward(self,
                res_fields: torch.Tensor,
                intensities: torch.Tensor,
                width: torch.Tensor,
                gauss: torch.Tensor,
                lorentz: torch.Tensor,
                fields: torch.Tensor,
                lvl_down: torch.Tensor,
                lvl_up: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        res_fields, width, intensities, areas = self._transform_data_to_mesh_format(res_fields, intensities, width)
        res_fields, width, intensities, areas = self._modify_data_dimensions(res_fields, width, intensities, areas)
        res_fields, width, intensities, areas = self._final_mask(res_fields, width, intensities, areas)

        fields, min_b, max_b = self._get_new_field(res_fields, width, intensities)
        res_fields, width, intensities, areas = self._final_mask(res_fields, width, intensities, areas)
        spec = self.spectra_integrator(res_fields, width, intensities, areas, fields)
        spectrum = self.post_spectra_processor(gauss, lorentz, fields, spec)

        return spectrum, (min_b, max_b)


class CrystalStationaryProcessingExpanded(_AutoFieldAxisMixin, CrystalStationaryProcessing):
    """
    Expanded version of `CrystalStationaryProcessing` for single‑crystal or discrete‑orientation
    samples, with automatic magnetic field axis generation.

    The field sweep is computed from the min/max resonance fields and the maximum linewidth
    (after applying a cutoff), with adjustable margins and clamping. The forward method
    returns `(spectrum, (min_field, max_field))`
    """
    def __init__(self,
                 mesh: mesher.CrystalMesh,
                 spectra_integrator: tp.Optional[BaseSpectraIntegrator] = None,
                 harmonic: int = 1,
                 post_spectra_processor: PostSpectraProcessing = PostSpectraProcessing(),
                 computational_details: ComputationalDetails = ComputationalDetails(),
                 output_mode: OutputSpectraMode = OutputSpectraMode.TOTAL,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32,
                 num_points: int = 4_000,
                 spectral_width_part: float = 0.6,
                 width_factor: float = 3.0,
                 min_exp_field: float = 0.0,
                 max_exp_field: float = 2.0,
                 width_cutoff: float = 0.5):
        """
        :param mesh: Crystal mesh object (CrystalMesh).
        :param spectra_integrator: Optional custom integrator.
        :param harmonic: Spectral harmonic (0 = absorption, 1 = first derivative).
        :param post_spectra_processor: Post‑processing object.
        :param computational_details: Integration details.
        :param output_mode: Must be `OutputSpectraMode.TOTAL`.
        :param device: Computation device.
        :param dtype: Data type.
        :param num_points: Number of points in the generated field axis.
        :param spectral_width_part: Fraction of the estimated spectral width.
        :param width_factor: Multiplier for the maximum linewidth.
        :param min_exp_field: Absolute lower bound for the field sweep.
        :param max_exp_field: Absolute upper bound for the field sweep.
        :param width_cutoff: Linewidth threshold (Tesla); linewidths above this are considered.
        """
        super().__init__(mesh, spectra_integrator, harmonic, post_spectra_processor,
                         computational_details, output_mode, device, dtype)
        if output_mode != OutputSpectraMode.TOTAL:
            raise NotImplementedError(f"output_mode is supported only Total for expanded processing. "
                                      f"You have used {output_mode}")
        self._init_field_axis_buffers(num_points, spectral_width_part, width_factor,
                                      min_exp_field, max_exp_field, width_cutoff, device, dtype)

    def forward(self,
                res_fields: torch.Tensor,
                intensities: torch.Tensor,
                width: torch.Tensor,
                gauss: torch.Tensor,
                lorentz: torch.Tensor,
                fields: torch.Tensor,
                lvl_down: torch.Tensor,
                lvl_up: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        res_fields, width, intensities, areas = self._transform_data_to_mesh_format(res_fields, intensities, width)
        res_fields, width, intensities, areas = self._modify_data_dimensions(res_fields, width, intensities, areas)
        res_fields, width, intensities, areas = self._final_mask(res_fields, width, intensities, areas)

        fields, min_b, max_b = self._get_new_field(res_fields, width, intensities)

        spec = self.spectra_integrator(res_fields, width, intensities, areas, fields)
        spectrum = self.post_spectra_processor(gauss, lorentz, fields, spec)
        return spectrum, (min_b, max_b)


class PowderTimeProcessingExpanded(_AutoFieldAxisMixin, PowderTimeProcessing):
    """
    Expanded version of `PowderTimeProcessing` for time‑resolved powder EPR spectra
    with automatic field axis generation.

    The field axis is computed once (ignoring the time dimension) from the resonance
    fields and linewidths, and is identical for all time points. The output spectrum
    includes the time dimension in its shape. Returns `(spectrum, (min_field, max_field))`.
    """
    def __init__(self,
                 mesh: mesher.BaseMeshPowder,
                 spectra_integrator: tp.Optional[BaseSpectraIntegrator] = None,
                 harmonic: int = 1,
                 post_spectra_processor: PostSpectraProcessing = PostSpectraProcessing(),
                 computational_details: ComputationalDetails = ComputationalDetails(),
                 output_mode: OutputSpectraMode = OutputSpectraMode.TOTAL,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32,
                 num_points: int = 4_000,
                 spectral_width_part: float = 0.6,
                 width_factor: float = 3.0,
                 min_exp_field: float = 0.0,
                 max_exp_field: float = 2.0,
                 width_cutoff: float = 0.5):
        """
        :param mesh: Powder mesh object.
        :param spectra_integrator: Optional custom integrator.
        :param harmonic: Spectral harmonic.
        :param post_spectra_processor: Post‑processing object.
        :param computational_details: Integration details.
        :param output_mode: Must be `OutputSpectraMode.TOTAL`.
        :param device: Computation device.
        :param dtype: Data type.
        :param num_points: Number of points in the generated field axis.
        :param spectral_width_part: Fraction of the estimated spectral width.
        :param width_factor: Multiplier for the maximum linewidth.
        :param min_exp_field: Absolute lower bound for the field sweep.
        :param max_exp_field: Absolute upper bound for the field sweep.
        :param width_cutoff: Linewidth threshold.
        """
        super().__init__(mesh, spectra_integrator, harmonic, post_spectra_processor,
                         computational_details, output_mode, device, dtype)
        if output_mode != OutputSpectraMode.TOTAL:
            raise NotImplementedError(f"output_mode is supported only Total for expanded processing. "
                                      f"You have used {output_mode}")
        self._init_field_axis_buffers(num_points, spectral_width_part, width_factor,
                                      min_exp_field, max_exp_field, width_cutoff, device, dtype)

    def forward(self,
                res_fields: torch.Tensor,
                intensities: torch.Tensor,
                width: torch.Tensor,
                gauss: torch.Tensor,
                lorentz: torch.Tensor,
                fields: torch.Tensor,
                lvl_down: torch.Tensor,
                lvl_up: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        res_fields, width, intensities, areas = self._transform_data_to_mesh_format(res_fields, intensities, width)
        res_fields, width, intensities, areas = self._modify_data_dimensions(res_fields, width, intensities, areas)
        res_fields, width, intensities, areas = self._final_mask(res_fields, width, intensities, areas)
        fields, min_b, max_b = self._get_new_field(res_fields, width, intensities)

        spec = self.spectra_integrator(res_fields, width, intensities, areas, fields)
        spectrum = self.post_spectra_processor(gauss, lorentz, fields, spec)
        return spectrum, (min_b, max_b)


class CrystalTimeProcessingExpanded(_AutoFieldAxisMixin, CrystalTimeProcessing):
    """
    Expanded version of `CrystalTimeProcessing` for time‑resolved single‑crystal EPR
    spectra with automatic field axis generation.

    The field axis is computed from the resonance fields and linewidths (ignoring time)
    and is the same for all time points. Returns `(spectrum, (min_field, max_field))`.
    """
    def __init__(self,
                 mesh: mesher.CrystalMesh,
                 spectra_integrator: tp.Optional[BaseSpectraIntegrator] = None,
                 harmonic: int = 1,
                 post_spectra_processor: PostSpectraProcessing = PostSpectraProcessing(),
                 computational_details: ComputationalDetails = ComputationalDetails(),
                 output_mode: OutputSpectraMode = OutputSpectraMode.TOTAL,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32,
                 num_points: int = 4_000,
                 spectral_width_part: float = 0.6,
                 width_factor: float = 3.0,
                 min_exp_field: float = 0.0,
                 max_exp_field: float = 2.0,
                 width_cutoff: float = 0.5):
        """
        :param mesh: Crystal mesh object.
        :param spectra_integrator: Optional custom integrator.
        :param harmonic: Spectral harmonic.
        :param post_spectra_processor: Post‑processing object.
        :param computational_details: Integration details.
        :param output_mode: Must be `OutputSpectraMode.TOTAL`.
        :param device: Computation device.
        :param dtype: Data type.
        :param num_points: Number of points in the generated field axis.
        :param spectral_width_part: Fraction of the estimated spectral width.
        :param width_factor: Multiplier for the maximum linewidth.
        :param min_exp_field: Absolute lower bound for the field sweep.
        :param max_exp_field: Absolute upper bound for the field sweep.
        :param width_cutoff: Linewidth threshold.
        """
        super().__init__(mesh, spectra_integrator, harmonic, post_spectra_processor,
                         computational_details, output_mode, device, dtype)
        if output_mode != OutputSpectraMode.TOTAL:
            raise NotImplementedError(f"output_mode is supported only Total for expanded processing. "
                                      f"You have used {output_mode}")
        self._init_field_axis_buffers(num_points, spectral_width_part, width_factor,
                                      min_exp_field, max_exp_field, width_cutoff, device, dtype)

    def forward(self,
                res_fields: torch.Tensor,
                intensities: torch.Tensor,
                width: torch.Tensor,
                gauss: torch.Tensor,
                lorentz: torch.Tensor,
                fields: torch.Tensor,
                lvl_down: torch.Tensor,
                lvl_up: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        res_fields, width, intensities, areas = self._transform_data_to_mesh_format(res_fields, intensities, width)
        res_fields, width, intensities, areas = self._modify_data_dimensions(res_fields, width, intensities, areas)
        res_fields, width, intensities, areas = self._final_mask(res_fields, width, intensities, areas)

        fields, min_b, max_b = self._get_new_field(res_fields, width, intensities)
        spec = self.spectra_integrator(res_fields, width, intensities, areas, fields)
        spectrum = self.post_spectra_processor(gauss, lorentz, fields, spec)
        return spectrum, (min_b, max_b)


class StationaryIntensityCalculatorExpanded(StationaryIntensityCalculator):
    """Reimplement StationaryIntensityCalculator with expanded populator.

    Handles calculation of transition intensities based on:
    - Transition matrix elements (magnetization)
    - Level populations. Uses Boltzmann thermal populations at specified temperature
      or predefined population given in context.
    """

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
            return StationaryPopulatorExpanded(
                context=context, init_temperature=temperature, device=device, dtype=dtype)
        else:
            return populator


class StationarySpectraExpanded(StationarySpectra):
    """
    Expanded version of `StationarySpectra` with automatic field‑axis generation
    and support for batch‑processed strain and temperature dimensions as an additional dimensions.


    The `forward` method returns a tuple `(spectrum, (min_field, max_field))`, where
    `min_field` and `max_field` are the computed lower and upper field limits for
    each batch element.

    Output spectrum and fields have the next dimensions order:
        -spectrum: strain_dimension, temperature_dimension, *batch_dimensions, spectral_dimension
        field_batch_positions: strain_dimension, temperature_dimension, *batch_dimensions
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
                 expended_processing_details: ExpandedProcessingDetails = ExpandedProcessingDetails(),
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
        self.expended_processing_details = expended_processing_details
        super().__init__(
            freq, sample, spin_system_dim, batch_dims, mesh,
            intensity_calculator, populator, spectra_integrator, harmonic,
            post_spectra_processor, temperature, recompute_spin_parameters,
            computational_details, inference_mode, output_eigenvector, context,
            hamiltonian_mode, output_mode, device, dtype
        )
        self.broader = BroadenerExpanded(device=device)

    def _init_spectra_processor(self,
                                spectra_integrator: tp.Optional[BaseSpectraIntegrator],
                                harmonic: int,
                                post_spectra_processor: PostSpectraProcessing,
                                computational_details: ComputationalDetails,
                                output_mode: OutputSpectraMode,
                                device: torch.device,
                                dtype: torch.dtype) -> _AutoFieldAxisMixin:
        """
        Create an expanded processor that automatically determines the field axis.
        """
        if self.mesh.disordered:
            return PowderStationaryProcessingExpanded(
                self.mesh, spectra_integrator, harmonic, post_spectra_processor,
                computational_details, output_mode, device, dtype,
                num_points=self.expended_processing_details.num_points,
                spectral_width_part=self.expended_processing_details.spectral_width_part,
                width_factor=self.expended_processing_details.width_factor,
                min_exp_field=self.expended_processing_details.min_exp_field,
                max_exp_field=self.expended_processing_details.max_exp_field,
                width_cutoff=self.expended_processing_details.width_cutoff,
            )

        else:
            return CrystalStationaryProcessingExpanded(
                self.mesh, spectra_integrator, harmonic, post_spectra_processor,
                computational_details, output_mode, device, dtype,
                num_points=self.expended_processing_details.num_points,
                spectral_width_part=self.expended_processing_details.spectral_width_part,
                width_factor=self.expended_processing_details.width_factor,
                min_exp_field=self.expended_processing_details.min_exp_field,
                max_exp_field=self.expended_processing_details.max_exp_field,
                width_cutoff=self.expended_processing_details.width_cutoff,
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
            return StationaryIntensityCalculatorExpanded(
                self.spin_system_dim, temperature, populator, context,
                disordered=self.mesh.disordered,
                computational_details=computational_details,
                device=device, dtype=dtype
            )
        else:
            return intensity_calculator

    def _get_intensity_mask(self,
                            intensities: torch.Tensor,
                            res_fields: torch.Tensor,
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
        :param res_fields: Resonance fields (passed for signature consistency).
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
        res_fields = res_fields[..., intensities_mask]
        vector_down = vector_down[..., intensities_mask, :]
        vector_up = vector_up[..., intensities_mask, :]

        extras = self._add_to_mask_additional(vector_down,
            vector_up, lvl_down, lvl_up, resonance_energies)
        extras = self._mask_components(intensities_mask, *extras)

        freq_to_field = self._freq_to_field(vector_down, vector_up, Gz)
        intensities = intensities.unsqueeze(0)

        intensities *= freq_to_field.unsqueeze(0).unsqueeze(0)
        intensities = intensities / self.intensity_std

        extras = self._compute_additional(
            sample, F, Gx, Gy, Gz, full_system_vectors, *extras
        )

        full_system_vectors = self._mask_full_system_eigenvectors(intensities_mask, full_system_vectors)
        res_fields = res_fields.unsqueeze(0)
        vector_down = vector_down.unsqueeze(0)
        vector_up = vector_up.unsqueeze(0)

        width = self.broader(sample, vector_down, vector_up, res_fields) * freq_to_field

        width_size = width.shape[0]
        temp_size = intensities.shape[1]
        common_shape = intensities.shape[2:]
        target_shape = [width_size, temp_size, *common_shape]

        res_fields = res_fields.unsqueeze(0).expand(target_shape)
        intensities = intensities.expand(target_shape)
        width = width.expand(target_shape)

        if full_system_vectors is not None:
            full_system_vectors = full_system_vectors.unsqueeze(0).unsqueeze(0)

        return res_fields, intensities, width, full_system_vectors, *extras

    def forward(self,
                sample: spin_model.MultiOrientedSample,
                fields: torch.Tensor, time: tp.Optional[torch.Tensor] = None, **kwargs):
        """
        :param sample: MultiOrientedSample object
        :param fields: The magnetic fields in Tesla units
        :param time: It is used only for time resolved spectra
        :param kwargs:
        :return:
        """

        B_low = fields[..., 0]
        B_high = fields[..., -1]
        B_low = B_low.unsqueeze(-1).repeat(*([1] * B_low.ndim), *self.mesh_size)
        B_high = B_high.unsqueeze(-1).repeat(*([1] * B_high.ndim), *self.mesh_size)

        F, Gx, Gy, Gz = self._hamiltonian_getter(sample)
        (vector_down, vector_up), (lvl_down, lvl_up), res_fields,\
            resonance_energies, full_system_vectors = self._resfield_method(sample, B_low, B_high, F, Gz)
        if (vector_up.shape[-2] == 0):
            temperature_shape = self.intensity_calculator.temperature.shape
            ham_shape = sample.base_ham_strain.shape
            width_size = ham_shape[0]
            temp_size = temperature_shape[0]
            common_shape = resonance_energies.shape[:-3]
            target_shape = [width_size, temp_size, *common_shape]
            min_pos_batch = fields[..., 0].expand(target_shape)
            max_pos_batch = fields[..., 1].expand(target_shape)
            spec = torch.zeros((*target_shape, self.spectra_processor.num_points), dtype=min_pos_batch.dtype,
                               device=min_pos_batch.device)
            return spec, (min_pos_batch, max_pos_batch)

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
