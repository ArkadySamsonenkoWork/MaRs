import typing as tp
import warnings
from functools import wraps
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum

import torch
import torch.nn as nn

from .. import constants
from .. import mesher
from .res_line_solvers import res_field_algorithm, res_freq_algorithm
from .. import spin_model

from .spectral_integration import BaseSpectraIntegrator
from ..population import BaseTimeDepPopulator, StationaryPopulator, LevelBasedPopulator,\
    RWADensityPopulator, PropagatorDensityPopulator, BasePopulator
from ..population import contexts
from .utils import compute_matrix_element, ComputationalDetails, OutputSpectraMode

from .spectra_processing_base import PostSpectraProcessing, BaseResProcessing,\
    PowderStationaryProcessing, CrystalStationaryProcessing,\
    PowderTimeProcessing, CrystalTimeProcessing

from .magnetization_mode import ResonatorMagnetizationConfig, WaveMagnetizationConfig,\
    MagnetizationConfig, ResonatorMode

from .intensity_base import BaseIntensityCalculator, BaseResIntensityCalculator
from .intensity_wave import WaveIntensityCalculator, WaveTimeIntensityCalculator


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
                 disordered: bool = True,
                 magnetization_config: ResonatorMagnetizationConfig = ResonatorMagnetizationConfig(),
                 computational_details: ComputationalDetails = ComputationalDetails(),
                 device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32):
        """
        :param spin_system_dim: Dimension of spin system Hilbert space.

        :param temperature: Temperature in Kelvin of a sample.
        :param populator: BasePopulator object. Default is None
        (auto-initialized based as stationary populator)
        :param context: Relaxation/population context defining relaxation and initial population. Default is None
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
            Default is perpendicular mode.

        :param computational_details: ComputationalDetails
            computational_details : ComputationalDetails, optional
            Configuration object that governs the numerical aspects of spectra generation.
            In this class it is used for the getting values of time-evolution equations solving
        :param device: Computation device. Default is torch.device("cpu")
        :param dtype: Data type for floating point operations. Default is torch.float32
        """
        super().__init__(spin_system_dim, temperature, populator, context, disordered, magnetization_config,
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
                self.compute_magnetization(Gx, Gy, Gz, res_manifold, vector_down, vector_up)
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
            Default is perpendicular mode.

        :param computational_details: ComputationalDetails
            computational_details : ComputationalDetails, optional
            Configuration object that governs the numerical aspects of spectra generation.
            In this class it is used for the getting values of time-evolution equations solving
        :param device: Computation device. Default is torch.device("cpu")
        :param dtype: Data type for floating point operations. Default is torch.float32
        """
        super().__init__(
            spin_system_dim, temperature, populator, context, disordered, magnetization_config, computational_details,
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
                self.compute_magnetization(Gx, Gy, Gz, res_manifold, vector_down, vector_up)
        )
        return intensity

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
        magnetization_config: MagnetizationConfig = ResonatorMagnetizationConfig(),
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
        :param magnetization_config: Configuration for the magnetization (transition moment) calculation.
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
            magnetization_config,
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
        magnetization_config: MagnetizationConfig,
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
                 magnetization_config: MagnetizationConfig = ResonatorMagnetizationConfig(),
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
             docs of :class:'mars.spectra_manager.utils.ComputationalDetails'

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

        :param magnetization_config:
            Configuration for the transition‑moment (magnetization) calculation.

            - For conventional cavity EPR, pass a
              ``mars.spectra_manager.ResonatorMagnetizationConfig`` instance.
              The ``mode`` field selects perpendicular (default) or parallel
              orientation of the microwave field B_1 relative to B_0.

            - For propagating‑wave experiments, pass a
              ``mars.spectra_manager.WaveMagnetizationConfig`` instance.
              This allows you to define the wave polarisation (linear, circular,
              unpolarised) and the angle between the wave vector and B_0.

            Default is ``ResonatorMagnetizationConfig()`` (perpendicular mode).

            Example import:
                from mars.spectra_manager import (
                    ResonatorMagnetizationConfig, ResonatorMode,
                    WaveMagnetizationConfig, Polarization
                )

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
                         inference_mode, output_eigenvector, context, magnetization_config,
                         hamiltonian_mode, output_mode,
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
                                                                   magnetization_config,
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
             docs of :class:'mars.spectra_manager.utils.ComputationalDetails'

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
            if isinstance(magnetization_config, ResonatorMagnetizationConfig):
                return StationaryIntensityCalculator(
                    self.spin_system_dim, temperature, populator, context,
                    disordered=self.mesh.disordered,
                    magnetization_config=magnetization_config,
                    computational_details=computational_details,
                    device=device, dtype=dtype
                )
            elif isinstance(magnetization_config, WaveMagnetizationConfig):
                return WaveIntensityCalculator(
                    self.spin_system_dim, temperature, populator, context,
                    disordered=self.mesh.disordered,
                    magnetization_config=magnetization_config,
                    computational_details=computational_details,
                    device=device, dtype=dtype
                )
            else:
                raise ValueError(
                    f"magnetization_config must be either ResonatorMagnetizationConfig or "
                    f"WaveMagnetizationConfig, got {type(magnetization_config).__name__}"
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
                 magnetization_config: MagnetizationConfig = ResonatorMagnetizationConfig(),
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

        :param magnetization_config:
            Configuration for the transition‑moment (magnetization) calculation.

            - For conventional cavity EPR, pass a
              ``mars.spectra_manager.ResonatorMagnetizationConfig`` instance.
              The ``mode`` field selects perpendicular (default) or parallel
              orientation of the microwave field B_1 relative to B_0.

            - For propagating‑wave experiments, pass a
              ``mars.spectra_manager.WaveMagnetizationConfig`` instance.
              This allows you to define the wave polarisation (linear, circular,
              unpolarised) and the angle between the wave vector and B_0.

            Default is ``ResonatorMagnetizationConfig()`` (perpendicular mode).

            Example import:
                from mars.spectra_manager import (
                    ResonatorMagnetizationConfig, ResonatorMode,
                    WaveMagnetizationConfig, Polarization
                )

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
                         inference_mode, output_eigenvector, context, magnetization_config,
                         hamiltonian_mode, output_mode,
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
             docs of :class:'mars.spectra_manager.utils.ComputationalDetails'

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
                                  magnetization_config: MagnetizationConfig,
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
            if isinstance(magnetization_config, ResonatorMagnetizationConfig):
                return StationaryIntensityCalculator(
                    self.spin_system_dim, temperature, populator, context,
                    disordered=self.mesh.disordered,
                    magnetization_config=magnetization_config,
                    computational_details=computational_details,
                    device=device, dtype=dtype
                )
            elif isinstance(magnetization_config, WaveMagnetizationConfig):
                return WaveIntensityCalculator(
                    self.spin_system_dim, temperature, populator, context,
                    disordered=self.mesh.disordered,
                    magnetization_config=magnetization_config,
                    computational_details=computational_details,
                    device=device, dtype=dtype
                )
            else:
                raise ValueError(
                    f"magnetization_config must be either ResonatorMagnetizationConfig or "
                    f"WaveMagnetizationConfig, got {type(magnetization_config).__name__}"
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
                 magnetization_config: MagnetizationConfig = ResonatorMagnetizationConfig(),
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

        :param magnetization_config:
            Configuration for the transition‑moment (magnetization) calculation.

            - For conventional cavity EPR, pass a
              ``mars.spectra_manager.ResonatorMagnetizationConfig`` instance.
              The ``mode`` field selects perpendicular (default) or parallel
              orientation of the microwave field B_1 relative to B_0.

            - For propagating‑wave experiments, pass a
              ``mars.spectra_manager.WaveMagnetizationConfig`` instance.
              This allows you to define the wave polarisation (linear, circular,
              unpolarised) and the angle between the wave vector and B_0.

            Default is ``ResonatorMagnetizationConfig()`` (perpendicular mode).

            Example import:
                from mars.spectra_manager import (
                    ResonatorMagnetizationConfig, ResonatorMode,
                    WaveMagnetizationConfig, Polarization
                )

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
                         inference_mode, output_eigenvector, context, magnetization_config,
                         hamiltonian_mode, output_mode,
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
                                  magnetization_config: MagnetizationConfig,
                                  computational_details: ComputationalDetails,
                                  device: torch.device, dtype: torch.dtype):
        if intensity_calculator is None:
            if isinstance(magnetization_config, ResonatorMagnetizationConfig):
                return TimeIntensityCalculator(
                    self.spin_system_dim, temperature, populator, context,
                    disordered=self.mesh.disordered,
                    magnetization_config=magnetization_config,
                    computational_details=computational_details,
                    device=device, dtype=dtype
                )
            elif isinstance(magnetization_config, WaveMagnetizationConfig):
                return WaveTimeIntensityCalculator(
                    self.spin_system_dim, temperature, populator, context,
                    disordered=self.mesh.disordered,
                    magnetization_config=magnetization_config,
                    computational_details=computational_details,
                    device=device, dtype=dtype
                )
            else:
                raise ValueError(
                    f"magnetization_config must be either ResonatorMagnetizationConfig or "
                    f"WaveMagnetizationConfig, got {type(magnetization_config).__name__}"
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
             docs of :class:'mars.spectra_manager.utils.ComputationalDetails'

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
                 magnetization_config: ResonatorMagnetizationConfig = ResonatorMagnetizationConfig(),
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
                         inference_mode, output_eigenvector, context, magnetization_config,
                         hamiltonian_mode, output_mode,
                         device=device, dtype=dtype)

    def _postcompute_batch_data(self, sample: spin_model.BaseSample,
                                res_fields: torch.Tensor, intensities: torch.Tensor, width: torch.Tensor,
                                F: torch.Tensor, Gx: torch.Tensor, Gy: torch.Tensor,
                                Gz: torch.Tensor, full_system_vectors: tp.Optional[torch.Tensor],
                                time: torch.Tensor, *extras, **kwargs):
        lvl_down, lvl_up, resonance_energies, vector_down, vectors_up, *extras = extras
        Sz = sample.base_spin_system.get_electron_z_operator()

        if resonance_energies.dtype == torch.float32:
            warnings.warn(
                "Using float32 for density-matrix population computations can lead to "
                "incorrect results, especially for long evolution times and when using "
                "propagator-based methods. We  recommend using float64 precision, "
                "or at least verify your results against a double-precision calculation.",
                RuntimeWarning,
                stacklevel=2
            )

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
                                  magnetization_config: MagnetizationConfig,
                                  computational_details: ComputationalDetails,
                                  device: torch.device, dtype: torch.dtype):
        if intensity_calculator is None:
            if isinstance(magnetization_config, ResonatorMagnetizationConfig):
                mode = ResonatorMode(magnetization_config.mode)
                if mode == ResonatorMode.PARALLEL:
                    raise NotImplementedError(
                        "Density calculation supports only cavity detection in the Perpendicular Mode"
                    )
                return TimeDensityCalculator(
                    self.spin_system_dim, temperature, populator, context,
                    disordered=self.mesh.disordered,
                    magnetization_config=magnetization_config,
                    computational_details=computational_details,
                    device=device, dtype=dtype
                )
            elif isinstance(magnetization_config, WaveMagnetizationConfig):
                raise NotImplementedError(
                    "Density calculation supports only cavity detection in the Perpendicular Mode"
                )
            else:
                raise ValueError(
                    f"magnetization_config must be either ResonatorMagnetizationConfig or "
                    f"WaveMagnetizationConfig, got {type(magnetization_config).__name__}"
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
                 magnetization_config: MagnetizationConfig = WaveMagnetizationConfig(),
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
                         inference_mode, output_eigenvector, context, magnetization_config,
                         hamiltonian_mode, output_mode,
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
