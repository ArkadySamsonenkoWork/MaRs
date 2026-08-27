import typing as tp
import warnings

import torch

from .. import mesher
from .res_line_solvers import fixed_fields_algorithm
from .. import spin_model
from .spectral_integration import BaseSpectraIntegrator
from ..population import BaseTimeDepPopulator, RWADensityPopulator, PropagatorDensityPopulator, BasePopulator
from ..population import contexts
from .spectra_manager import BaseSpectra,\
    HamComputationMethod, BaseResIntensityCalculator, BaseIntensityCalculator

from .spectra_processing_base import PostSpectraProcessing
from .spectra_processing_direct import PowderDirectProcessing, CrystalDirectProcessing, BaseDirectProcessing

from .utils import ComputationalDetails, OutputSpectraMode
from .magnetization_mode import ResonatorMagnetizationConfig, MagnetizationConfig, ResonatorMode


class BaseDirectIntensityCalculator(BaseIntensityCalculator):
    """Intensity calculator for fixed-field, pulse, and density-matrix EPR experiments.

    Operates on the complete eigensystem or density matrix rather than individual
    transition pairs.
    """
    def __init__(self, spin_system_dim: int, temperature: tp.Optional[float],
                 populator: tp.Optional[tp.Union[BaseTimeDepPopulator, str]],
                 context: tp.Optional[contexts.BaseContext],
                 disordered: bool = True,
                 magnetization_config: ResonatorMagnetizationConfig = ResonatorMagnetizationConfig(),
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
        :param magnetization_config:
            Configuration of the conventional resonator excitation geometry.

            Must be ``ResonatorMagnetizationConfig``.

            ``magnetization_config.mode`` specifies the direction of the
            oscillating microwave magnetic field B1 relative to the static
            field B0:

            - ``MagneticFieldMode.PERPENDICULAR``:
              ``B1 _|_ B0``. This is the conventional transverse EPR mode.

            - ``MagneticFieldMode.PARALLEL``:
              ``B1 || B0``. The longitudinal transition-magnetization
              component is used.
            Default is perpendicular mode
            .
        :param computational_details: ComputationalDetails
            computational_details : ComputationalDetails, optional
            Configuration object that governs the numerical aspects of spectra generation.
            In this class it is used for the getting values of time-evolution equations solving
        :param device: Computation device. Default is torch.device("cpu")
        :param dtype: Data type for floating point operations. Default is torch.float32
        """
        super().__init__(
            spin_system_dim, temperature, populator, context,
            disordered, magnetization_config, computational_details,
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

    def _magnetization_factory(
        self,
        disordered: bool,
        magnetization_config: ResonatorMagnetizationConfig,
        computational_details: ComputationalDetails,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tp.Callable[..., torch.Tensor]:
        """
        Select the conventional-resonator magnetization computation.
        """
        if not isinstance(
            magnetization_config,
            ResonatorMagnetizationConfig,
        ):
            raise TypeError(
                "BaseResIntensityCalculator requires "
                "ResonatorMagnetizationConfig, got "
                f"{type(magnetization_config).__name__}."
            )

        mode = ResonatorMode(magnetization_config.mode)

        if disordered:
            if mode == ResonatorMode.PERPENDICULAR:
                return self._compute_magnetization_powder_perpendicular

            if mode == ResonatorMode.PARALLEL:
                return self._compute_magnetization_powder_parallel

        else:
            if mode == ResonatorMode.PERPENDICULAR:
                return self._compute_magnetization_crystal_perpendicular

            if mode == ResonatorMode.PARALLEL:
                return self._compute_magnetization_crystal_parallel

        raise ValueError(
            f"Unsupported resonator magnetization mode: {mode!r}"
        )

    def _compute_magnetization_powder_perpendicular\
                    (self, *args, **kwargs) -> torch.Tensor:
        """Compute powder-averaged magnetization.
        :return: Magnetization tensor. Shape [...]
        """
        raise NotImplementedError

    def _compute_magnetization_crystal_perpendicular\
                    (self, *args, **kwargs) -> torch.Tensor:
        """Compute crystal-geometry magnetization.
        :return: Magnetization tensor. Shape [...]
        """
        raise NotImplementedError

    def _compute_magnetization_powder_parallel\
                    (self, *args, **kwargs) -> torch.Tensor:
        """Compute powder-averaged magnetization.
        :return: Magnetization tensor. Shape [...]
        """
        raise NotImplementedError

    def _compute_magnetization_crystal_parallel\
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
                 magnetization_config: MagnetizationConfig = ResonatorMagnetizationConfig(),
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
             docs of :class:'mars.spectra_manager.utils.ComputationalDetails'

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
                         inference_mode, output_eigenvector, context, magnetization_config,
                         hamiltonian_mode, output_mode,
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
             docs of :class:'mars.spectra_manager.utils.ComputationalDetails'

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
                                  magnetization_config: MagnetizationConfig,
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
                magnetization_config=magnetization_config,
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
