import math
import typing as tp
import warnings
from functools import wraps
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum

import torch
import torch.fft as fft
import torch.nn as nn

from .. import mesher
from .res_line_solvers import fixed_fields_algorithm
from .. import spin_model
from .spectral_integration import BaseSpectraIntegrator
from ..population import BaseTimeDepPopulator, RWADensityPopulator, PropagatorDensityPopulator, BasePopulator
from ..population import contexts
from .spectra_manager import DensityTimeSpectra, BaseSpectra,\
    HamComputationMethod, ComputationalDetails, BaseResIntensityCalculator,\
    PostSpectraProcessing, OutputSpectraMode, BaseIntensityCalculator, BaseResProcessing, BaseProcessing


class BaseDirectProcessing(BaseProcessing):
    """Base class for fixed-field spectral processing over orientation meshes.

    Designed for direct diagonalization approaches where resonance searching is bypassed.
    Computes orientation-averaged spectra directly from pre-calculated intensities
    at fixed magnetic field points without line-broadening or linewidth parameters.

    The processing pipeline consists of:
    1. Transform intensity data to mesh format (interpolation/triangulation)
    2. Compute orientation weights (areas)
    3. Perform weighted orientation averaging
    4. Return spectrum in requested output mode
    """
    def __init__(self,
                 mesh: mesher.BaseMesh,
                 computational_details: ComputationalDetails = ComputationalDetails(),
                 output_mode: OutputSpectraMode = OutputSpectraMode.TOTAL,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32):
        """
        :param mesh: Mesh object defining orientation sampling grid.

        :param computational_details: The details of final spectral integration and spectra processing.

        :param output_mode: Controls spectrum organization:
            - "total": returns conventional summed spectrum over all orientations
            - "transitions": returns per-orientation/transition contributions alongside level indices

        :param device: Computation device. Default is torch.device("cpu")
        :param dtype: Data type for floating point operations. Default is torch.float32
        """
        super().__init__(mesh, computational_details, output_mode, device, dtype)

    def _output_factory_setter(self, output_mode: OutputSpectraMode) -> None:
        """Set output management methods based on requested mode.

        :param output_mode: Controls the organization of the computed spectrum.
        :return: None
        """
        if output_mode == OutputSpectraMode.TOTAL:
            self._modify_data_dimensions = self._modify_data_dimensions_total
            self._get_output = self._get_output_total
        else:
            raise ValueError(
                f"DirectProcessor supports only {OutputSpectraMode.TOTAL}. Got {output_mode}"
            )

    @abstractmethod
    def _compute_areas(self, batch_shape: tp.Union[torch.Size, int], device: torch.device) -> torch.Tensor:
        """Compute orientation weights for integration.

        :param batch_shape: Leading batch dimensions from intensity tensor.
        :param device: Target computation device.
        :return: Tensor of integration weights with shape broadcastable to [..., num_mesh_elements].
        """
        pass

    @abstractmethod
    def _transform_data_to_mesh_format(self, intensities: torch.Tensor) -> torch.Tensor:
        """Map intensities onto mesh geometry.

        :param intensities: Raw intensities at mesh vertices. Shape [..., num_vertices, num_fields]
        :return: Intensities aligned with mesh simplices or discrete orientations.
        """
        pass

    @abstractmethod
    def _integrate(self, intensities: torch.Tensor, areas: torch.Tensor, fields: torch.Tensor) -> torch.Tensor:
        """Perform orientation averaging.

        :param intensities: Mesh-aligned intensities. Shape [..., num_mesh_elements, num_fields]
        :param areas: Orientation weights. Shape [..., num_mesh_elements]
        :param fields: Magnetic field axis. Shape [num_fields]
        :return: Orientation-averaged spectrum. Shape [..., num_fields]
        """
        pass

    def _modify_data_dimensions_total(
            self, fields: torch.Tensor, intensities: torch.Tensor, areas: torch.Tensor) ->\
            tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Modify data dimension to make it computable with the given type of integrator and computation method.
        This modifier do not flatten data into num_simplices - num_transitions dimension

        :param fields: resonance field with the shape [..., num_fields, num_simplices, 3]
        :param intensities: intensities with the shape [..., num_fields, num_simplices]
        :param areas: areas with the shape [..., num_fields, num_simplices]
        :return: modified
         fields with the shape [..., num_fields]
         intensities with the shape [..., num_simplices, num_fields]
         areas with the shape [..., num_simplices, num_fields]
        """
        return fields, intensities, areas

    def _get_output_total(self, spectrum: torch.Tensor) ->\
            torch.Tensor:
        """
        Returns the final integrated spectrum as a single tensor.
        :param spectrum: Spectral contributions per transition. Shape: [..., num_transitions, N]

        :return: The single spectrum in 1D or 2D with the shpae [...., 1/2 D dimensions]
        """
        return spectrum

    def forward(self,
                fields: torch.Tensor,
                intensities: torch.Tensor) -> torch.Tensor:
        """Execute fixed-field spectral processing pipeline.

        1. Transform intensity data to mesh format
        2. Apply dimension modifiers for output mode
        3. Compute orientation weights (areas)
        4. Perform weighted orientation averaging
        5. Return spectrum in requested format

        :param fields: Magnetic field axis. Shape [num_fields]
        :param intensities: Computed intensities at fixed field points.
            Shape [..., num_orientations, num_fields] or [..., num_vertices, num_fields]
        :return: Orientation-averaged spectrum.
        """
        intensities = self._transform_data_to_mesh_format(intensities)
        batch_dims = max(0, intensities.dim() - 2)
        batch_shape = intensities.shape[:batch_dims]

        areas = self._compute_areas(batch_shape, intensities.device)
        fields, intensities, areas = self._modify_data_dimensions(fields, intensities, areas)
        spectrum = self._integrate(intensities, areas, fields)
        return self._get_output(spectrum)


class PowderDirectProcessing(BaseDirectProcessing):
    """Integrate fixed-field EPR spectra over spherical powder orientation mesh.

    Uses Delaunay triangulation and spherical triangle areas to perform
    rigorous orientation averaging. Designed for direct diagonalization
    where resonance lines are not computed.
    """
    def __init__(self,
                 mesh: mesher.BaseMeshPowder,
                 computational_details: ComputationalDetails = ComputationalDetails(),
                 output_mode: OutputSpectraMode = OutputSpectraMode.TOTAL,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32):
        """
        :param mesh: Powder mesh object (BaseMeshPowder) defining spherical grid.

        :param output_mode: Controls spectrum organization ("total" or "transitions").
        :param device: Computation device. Default is torch.device("cpu")
        :param dtype: Data type for floating point operations. Default is torch.float32
        """
        super().__init__(mesh,computational_details, output_mode, device, dtype)

    def _compute_areas(self, batch_shape: tp.Union[torch.Size, int], device: torch.device) -> torch.Tensor:
        """Compute spherical triangle areas and expand to match batch dimensions.

        :param batch_shape: Leading batch dimensions from intensity tensor.
        :param device: Target computation device.
        :return: Expanded area tensor of shape [*batch_shape, num_simplices].
        """
        _, simplices = self.mesh.post_mesh
        areas = self.mesh.spherical_triangle_areas(*self.mesh.post_mesh)
        areas = areas.reshape(1, -1).expand(*batch_shape, -1)
        return areas

    def _transform_data_to_mesh_format(self, intensities: torch.Tensor) -> torch.Tensor:
        """Interpolate intensities from mesh vertices onto Delaunay triangulation.

        :param intensities: Intensities at mesh vertices. Shape [..., num_vertices, num_fields]
        :return: Interpolated intensities at simplex centers. Shape [..., num_simplices, num_fields]
        """
        _, simplices = self.mesh.post_mesh
        processed = self.mesh(intensities.transpose(-1, -2))
        simplex_data = self.mesh.to_delaunay(processed, simplices)
        return simplex_data.mean(dim=-1).transpose(-1, -2)

    def _integrate(self, intensities: torch.Tensor, areas: torch.Tensor, fields: torch.Tensor) -> torch.Tensor:
        """Compute powder-averaged spectrum via area-weighted summation.

        :param intensities: Simplex-aligned intensities. Shape [..., num_simplices, num_fields]
        :param areas: Spherical triangle areas. Shape [..., num_simplices]
        :param fields: Magnetic field axis (unused in direct averaging).
        :return: Powder-averaged spectrum. Shape [..., num_fields]
        """

        areas_exp = areas.unsqueeze(-1)
        total_area = areas_exp.sum(dim=-2)
        spectrum = torch.sum(intensities * areas_exp, dim=-2) / total_area
        return spectrum


class CrystalDirectProcessing(BaseDirectProcessing):
    """Integrate fixed-field EPR spectra for single-crystal or discrete orientations.

    Performs simple uniform averaging over discrete crystal orientations.
    No triangulation or area weighting is required.
    """
    def __init__(self,
                 mesh: mesher.CrystalMesh,
                 computational_details: ComputationalDetails = ComputationalDetails(),
                 output_mode: OutputSpectraMode = OutputSpectraMode.TOTAL,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32):
        """
        :param mesh: Crystal mesh object defining single or discrete orientations.
        :param computational_details: The details of final spectral integration and spectra processing.
        :param output_mode: Controls spectrum organization ("total" or "transitions").
        :param device: Computation device. Default is torch.device("cpu")
        :param dtype: Data type for floating point operations. Default is torch.float32
        """
        super().__init__(mesh, computational_details, output_mode, device, dtype)

    def _compute_areas(self, batch_shape: tp.Union[torch.Size, int], device: torch.device) -> torch.Tensor:
        """Return uniform weights (ones) for discrete crystal orientations.

        :param batch_shape: Leading batch dimensions from intensity tensor.
        :param device: Target computation device.
        :return: Unity tensor of shape [*batch_shape, num_orientations].
        """
        num_orients = self.mesh.initial_size[0]
        return torch.ones((*batch_shape, num_orients), dtype=torch.float32, device=device)

    def _transform_data_to_mesh_format(self, intensities: torch.Tensor) -> torch.Tensor:
        """Pass-through for crystal mesh. Adds orientation dimension if missing.

        :param intensities: Raw intensities. Shape [..., num_orientations, num_fields]
        :return: Unmodified intensities tensor.
        """
        return intensities

    def _integrate(self, intensities: torch.Tensor, areas: torch.Tensor, fields: torch.Tensor) -> torch.Tensor:
        """Compute crystal-averaged spectrum via arithmetic mean.

        :param intensities: Orientation-resolved intensities. Shape [..., num_orients, num_fields]
        :param areas: Unity weights (unused in mean calculation).
        :param fields: Magnetic field axis (unused in direct averaging).
        :return: Crystal-averaged spectrum. Shape [..., num_fields]
        """
        return torch.mean(intensities, dim=-2)


class BaseDirectIntensityCalculator(BaseIntensityCalculator):
    """Intensity calculator for fixed-field, pulse, and density-matrix EPR experiments.

    Operates on the complete eigensystem or density matrix rather than individual
    transition pairs.
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
            spin_system_dim, temperature, populator, context,
            disordered, computational_details,
            device=device, dtype=dtype
        )

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

    def _compute_magnitization_powder\
                    (self, *args, **kwargs) -> torch.Tensor:
        """Compute powder-averaged magnetization.
        :return: Magnetization tensor. Shape [...]
        """
        raise NotImplementedError

    def _compute_magnitization_crystal\
                    (self, *args, **kwargs) -> torch.Tensor:
        """Compute crystal-geometry magnetization.
        :return: Magnetization tensor. Shape [...]
        """
        raise NotImplementedError

    def compute_intensity(self, *args, **kwargs):
        """
        """
        raise NotImplementedError

    def compute_population(self, time: torch.Tensor,
                                    fields: torch.Tensor,
                                    energies: torch.Tensor,
                                    full_system_vectors: tp.Optional[torch.Tensor],
                                    *args, **kwargs):
        return self.populator(time, fields, None,
                              None, energies,
                              None, None,
                              full_system_vectors, *args, **kwargs)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass alias for compute_intensity().

        :param args: Positional arguments forwarded to compute_intensity.
        :param kwargs: Keyword arguments forwarded to compute_intensity.
        :return: Intensity tensor. Shape [...]
        """
        return self.compute_intensity(*args, **kwargs)


class BaseDirectSpectra(BaseSpectra):
    """Base class for fixed-field and pulse EPR spectral simulation.

    Provides a complete pipeline for computing EPR spectra by directly
    diagonalizing the spin Hamiltonian at user-defined magnetic field points,
    bypassing resonance-line searching algorithms.

    The processing pipeline consists of:
    1. Diagonalize the full Hamiltonian at each specified magnetic field point
       to obtain eigenvalues and the complete eigenbasis.
    2. Compute state populations or density-matrix evolution via the
       configured intensity calculator.

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
            It is skipped for this class.

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

        warnings.warn(
            "For direct spectra computations, all broadening parameters are skipped.",
            UserWarning,
            stacklevel=2
        )

        super().__init__(freq, sample, spin_system_dim, batch_dims, mesh, intensity_calculator,
                         populator, spectra_integrator, harmonic, post_spectra_processor,
                         temperature, recompute_spin_parameters,
                         computational_details,
                         inference_mode, output_eigenvector, context, hamiltonian_mode, output_mode,
                         device=device, dtype=dtype)

    def _init_res_algorithm(self,
                            output_eigenvector: bool,
                            hamiltonian_mode: HamComputationMethod,
                            computational_details: ComputationalDetails,
                            device: torch.device, dtype: torch.dtype) -> \
            tp.Callable[[torch.Tensor, torch.Tensor, torch.Tensor], tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]]:
        """Instantiate the resonance field computation algorithm.

        Selects an appropriate Hamiltonian eigen data backend based on
        whether full eigenvectors are needed and whether some approximation is used.

        :param output_eigenvector: Whether full system eigenvectors should be computed.
        :param hamiltonian_mode: the method to use to compute the Hamiltonian eigen data.
        :param computational_details: The computational details to create EPR spectra:
                accuracy, number of iterations, and so on.

        :return: Configured resonance field solver.
        """
        return fixed_fields_algorithm.FixedField(
            spin_system_dim=self.spin_system_dim,
            mesh_size=self.mesh_size,
            batch_dims=self.batch_dims,
            output_full_eigenvector=output_eigenvector,
            device=device,
            dtype=dtype
        )

    def _init_cached_parameters(self):
        """Initialize internal buffers to support optional caching of spin parameters.

        When `recompute_spin_parameters=False`, resonance-related tensors
        (eigenvectors, levels, fields, etc.) are computed once and stored.
        This method sets up placeholder attributes used during the first forward pass.
        """
        if not self.recompute_spin_parameters:
            self._cashed_flag = False
            self.energies = None
            self.full_eigen_vectors = None
            self._resfield_method = self._cashed_resfield

        else:
            self._resfield_method = self._recomputed_resfield

    def _cashed_resfield(self, fields: torch.Tensor, F: torch.Tensor, Gz: torch.Tensor) ->\
            tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
        """Compute or retrieve cached resonance fields and eigensystem.

        On first call, delegates to `_recomputed_resfield` and stores results.
        Subsequent calls return the cached tensors without recomputation.

        :param fields: The magnetic fields where the kinetic should be computed
        :param F: Field-independent part of the Hamiltonian.
        :param Gz: Zeeman operator along z.
        :return: Same as `_recomputed_resfield`.
        """
        if not self._cashed_flag:
            self.energies, self.full_eigen_vectors = self._recomputed_resfield(fields, F, Gz)
            self._cashed_flag = True
        return self.energies, self.full_eigen_vectors

    def _recomputed_resfield(self, fields: torch.Tensor, F: torch.Tensor, Gz: torch.Tensor) ->\
            tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
        """Compute the eigen values and eigen vectors for the given magnetic fields
        :param fields: The magnetic fields where the kinetic should be computed
        :param F: Static Hamiltonian term.
        :param Gz: Zeeman coupling operator.
        :return: Tuple containing:
            - resonance_energies: eigenvalues [..., N]
            - full_eigen_vectors: complete eigenbasis [..., N, N] or None
        """
        energies, full_eigen_vectors = self.res_algorithm(fields, F, Gz)
        return energies, full_eigen_vectors

    def _init_spectra_integrator(self, spectra_integrator: tp.Optional[BaseSpectraIntegrator],
                                 harmonic: int, computational_details: ComputationalDetails,
                                 device: torch.device, dtype: torch.dtype)\
            -> None:
        """For the diferect filed computations the integration of spectral lines is absent
        """
        return None

    def _init_spectra_processor(self,
                                spectra_integrator: tp.Optional[BaseSpectraIntegrator],
                                harmonic: int,
                                post_spectra_processor: PostSpectraProcessing,
                                computational_details: ComputationalDetails,
                                output_mode: OutputSpectraMode,
                                device: torch.device,
                                dtype: torch.dtype) -> BaseDirectProcessing:
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
            return PowderDirectProcessing(self.mesh,
                                        computational_details=computational_details,
                                        output_mode=output_mode,
                                        device=device, dtype=dtype)
        else:
            return CrystalDirectProcessing(self.mesh,
                                         computational_details=computational_details,
                                         output_mode=output_mode,
                                         device=device, dtype=dtype)

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
            return BaseDirectIntensityCalculator(
                self.spin_system_dim, temperature, populator, context,
                disordered=self.mesh.disordered,
                computational_details=computational_details,
                device=device, dtype=dtype
            )
        else:
            return intensity_calculator

    def forward(self,
                sample: spin_model.MultiOrientedSample,
                fields: torch.Tensor, time: tp.Optional[torch.Tensor] = None, **kwargs):
        """Compute EPR spectrum over a given magnetic fields range.
        :param sample: MultiOrientedSample object.
        :param fields: The magnetic fields in Tesla units, where the signal should be computed. The shape [..., K]
        :param time: It is used only for time resolved spectra
        :param kwargs:
        :return: spectra in 1D or 2D. Batched or un batched.
        Depending on spectra Proccessor it can be another output format
        """
        F, Gx, Gy, Gz = self._hamiltonian_getter(sample)
        energies, full_system_vectors = self._resfield_method(fields, F, Gz)
        fields, intensities, full_system_vectors, *extras = \
            self.compute_parameters(sample, F, Gx, Gy, Gz,
                                    fields,
                                    energies,
                                    full_system_vectors)

        fields, intensities = self._postcompute_batch_data(
            sample, fields, intensities, F, Gx, Gy, Gz, full_system_vectors, time, *extras, **kwargs
        )

        return self._finalize(fields, intensities)

    def _finalize(self,
                  fields: torch.Tensor,
                  intensities: torch.Tensor):
        """Apply final spectral integration and line broadening.

        Delegates to the configured `spectra_processor` to produce the output spectrum.

        :param fields: field positions.
        :param intensities: Transition strengths.

        :return: The output of the given spectra Proccessor depending on the output_mode
        """
        return self.spectra_processor(fields, intensities)

    def _postcompute_batch_data(self, sample: spin_model.BaseSample,
                                fields: torch.Tensor, intensities: tp.Optional[torch.Tensor],
                                F: torch.Tensor, Gx: torch.Tensor, Gy: torch.Tensor,
                                Gz: torch.Tensor, full_system_vectors: tp.Optional[torch.Tensor],
                                time: torch.Tensor, *extras, **kwargs):

        energies, *extras = extras
        Sz = sample.base_spin_system.get_electron_z_operator()
        population = self.intensity_calculator.compute_population(
            time, fields, energies,
            full_system_vectors,
            F, Gx, Gy, Gz, Sz,
            self.resonance_parameter, *extras
        )
        intensities = population
        return fields, intensities

    def compute_parameters(self, sample: spin_model.MultiOrientedSample,
                           F: torch.Tensor,
                           Gx: torch.Tensor,
                           Gy: torch.Tensor,
                           Gz: torch.Tensor,
                           fields: torch.Tensor,
                           energies: torch.Tensor,
                           full_system_vectors: tp.Optional[torch.Tensor]) ->\
            tuple[torch.Tensor, tp.Optional[torch.Tensor], tp.Optional[torch.Tensor], tuple[tp.Any]]:
        """
        :param sample: The sample which transitions must be found.

        :param F: Magnetic free part of spin Hamiltonian H = F + B * G
        :param Gx: x-part of Hamiltonian Zeeman Term
        :param Gy: y-part of Hamiltonian Zeeman Term
        :param Gz: z-part of Hamiltonian Zeeman Term

        :param fields: Resonance fields. The shape os [..., K]

        :param full_system_vectors: Eigen vector of each level of a spin system. The shape os [..., N, N]. If
        output_eigen_vectors == False, then it will be None

        :return: tuple of the next data
         - fields
         - Intensities of transitions
         - Full system eigen vectors or None
         - extras parameters computed in _compute_additional
        """
        return fields, None, full_system_vectors, *(energies, )

    def __call__(self,
                sample: spin_model.MultiOrientedSample,
                fields: torch.Tensor, time: torch.Tensor, **kwargs):
        """
        :param sample: MultiOrientedSample object.

        :param fields: The magnetic fields in Tesla units
        :param time: It is used only for time resolved spectra
        :param kwargs:
        :return: spectra or some resonance data depending on the output_mode
        """
        return super().__call__(sample, fields, time)
