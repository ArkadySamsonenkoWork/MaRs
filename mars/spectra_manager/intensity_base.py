import typing as tp
import warnings
from functools import wraps
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum

import torch
import torch.nn as nn

from .. import constants

from ..population import BasePopulator
from ..population import contexts
from .utils import compute_matrix_element, ComputationalDetails
from .magnetization_mode import MagnetizationConfig, ResonatorMagnetizationConfig,\
    WaveMagnetizationConfig, ResonatorMode

magnetization_func =\
    tp.Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


class BaseIntensityCalculator(nn.Module, ABC):
    """Base class for all EPR intensity calculators.
    Provides shared infrastructure for spin system initialization, population model
    dispatch, device/dtype management, and powder/crystal magnetization routing.
    """
    def __init__(self,
                 spin_system_dim: tp.Union[int, list[int]],
                 temperature: tp.Optional[float] = None,
                 populator: tp.Optional[tp.Union[BasePopulator, str]] = None,
                 context: tp.Optional[contexts.BaseContext] = None,
                 disordered: bool = True,
                 magnetization_config: MagnetizationConfig = ResonatorMagnetizationConfig(),
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

        :param magnetization_config:
            Configuration describing the experimental geometry used for the
            transition-magnetization calculation.

            Supported configuration types depend on the concrete intensity
            calculator.

            ``ResonatorMagnetizationConfig`` describes a conventional EPR
            resonator experiment. It specifies whether the microwave magnetic
            field B1 is perpendicular or parallel to the static magnetic
            field B0.

            ``WaveMagnetizationConfig`` describes an experiment in which a
            propagating electromagnetic wave is incident on the sample. It
            specifies polarization and wave geometry through ``theta`` and
            ``phi``.

            The concrete subclass validates the configuration and selects the
            corresponding magnetization method in ``_magnetization_factory``.

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
        self.magnetization_config = magnetization_config

        self._compute_magnetization_method = self._magnetization_factory(
            disordered, magnetization_config, computational_details, device, dtype
        )
        self.register_buffer(
            "_magnetization_scale",
            torch.tensor((constants.PLANCK / constants.BOHR) ** 2, device=device, dtype=dtype),
        )
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
    def _magnetization_factory(
        self,
        disordered: bool,
        magnetization_config: MagnetizationConfig,
        computational_details: ComputationalDetails,
        device: torch.device,
        dtype: torch.dtype,
    ) -> magnetization_func:
        """
        Select the magnetization computation for the current experiment.

        The factory is called once during initialization. It must validate the
        supplied ``magnetization_config`` and return a bound method implementing
        the appropriate magnetization calculation.

        The returned method must accept the standard magnetization arguments:

        ``Gx, Gy, Gz, res_manifold, vector_down, vector_up``

        and return the transition magnetization factor.

        :param disordered:
            ``True`` for powder calculations and ``False`` for crystal
            calculations.

        :param magnetization_config:
            Experiment-specific magnetization configuration.

        :param computational_details:
            Numerical configuration. It is part of the common factory
            interface even when a particular implementation does not use it.

        :param device:
            Computation device.

        :param dtype:
            Floating-point dtype.

        :return:
            Bound method used by ``compute_magnetization``.
        """
        raise NotImplementedError

    def compute_magnetization(self, *args, **kwargs) -> torch.Tensor:
        """
        Compute the transition magnetization using the method selected by
        ``_magnetization_factory``.

        :param args:
            Positional arguments passed to the selected magnetization method.

        :param kwargs:
            Keyword arguments passed to the selected magnetization method.

        :return:
            Transition magnetization factor.
        """
        return self._compute_magnetization_method(*args, **kwargs)

    @abstractmethod
    def compute_intensity(self, *args, **kwargs) -> torch.Tensor:
        """Compute transition intensity based on magnetization and population.

        :param args: Experiment-specific positional arguments.
        :param kwargs: Experiment-specific keyword arguments.
        :return: Intensity tensor. Shape [...]
        """
        pass

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
    - Transition matrix elements (magnetization). It can be computed in:
        - perpendicular mode: ``B1 _|_ B0``;
        - parallel mode: ``B1 || B0``.

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

    def _magnetization_factory(
        self,
        disordered: bool,
        magnetization_config: MagnetizationConfig,
        computational_details: ComputationalDetails,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tp.Callable[..., torch.Tensor]:
        """
        Select the conventional-resonator magnetization computation.

        :param disordered:
            Defines how the sample orientation is treated.

            - ``True`` selects a powder magnetization expression;
            - ``False`` selects a single-crystal magnetization expression.

        :param magnetization_config:
            Must be ``ResonatorMagnetizationConfig``.

            Its ``mode`` determines the orientation of the oscillating
            microwave magnetic field B1 relative to B0:

            - ``MagneticFieldMode.PERPENDICULAR`` means ``B1 _|_ B0``;
            - ``MagneticFieldMode.PARALLEL`` means ``B1 || B0``.

            The combination of ``disordered`` and ``mode`` selects one of the
            four implementations provided by this class.

        :param computational_details:
            Numerical configuration. It is not currently required by the
            resonator magnetization formulas but is retained for a common
            factory interface.

        :param device:
            Computation device. Not directly used by this factory.

        :param dtype:
            Floating-point dtype. Not directly used by this factory.

        :return:
            One of:

            - ``_compute_magnetization_powder_perpendicular``;
            - ``_compute_magnetization_powder_parallel``;
            - ``_compute_magnetization_crystal_perpendicular``;
            - ``_compute_magnetization_crystal_parallel``.

        :raises TypeError:
            If ``magnetization_config`` is not
            ``ResonatorMagnetizationConfig``.

        :raises ValueError:
            If an unsupported resonator mode is provided.
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

    def _compute_magnetization_powder_perpendicular(
        self,
        Gx: torch.Tensor,
        Gy: torch.Tensor,
        Gz: torch.Tensor,
        res_manifold: torch.Tensor,
        vector_down: torch.Tensor,
        vector_up: torch.Tensor,
    ) -> torch.Tensor:
        """Compute powder-averaged transition intensity.

        :param Gx, Gy, Gz: Cartesian components of Zeeman operator. Shape [..., N, N]

        :param res_manifold: Resonance fields or frequencies. Shape [...]

        :param vector_down: Lower-state eigenvector. Shape [..., N]
        :param vector_up: Upper-state eigenvector. Shape [..., N]
        :return: Intensity proportional to |<up|Gx|down>|² + |<up|Gy|down>|², in (J·s/μ_B)²
        """
        magnetization = compute_matrix_element(vector_down, vector_up, Gx).square().abs() +\
                        compute_matrix_element(vector_down, vector_up, Gy).square().abs()
        return magnetization * self._magnetization_scale

    def _compute_magnetization_crystal_perpendicular(
        self,
        Gx: torch.Tensor,
        Gy: torch.Tensor,
        Gz: torch.Tensor,
        res_manifold: torch.Tensor,
        vector_down: torch.Tensor,
        vector_up: torch.Tensor,
    ) -> torch.Tensor:
        """Compute crystal transition intensity.

        The orientation of the wave magnetic field is along the x-axis.
        :param Gx, Gy, Gz: Cartesian components of Zeeman operator. Shape [..., N, N]
        :param res_manifold: Resonance fields or frequencies. Shape [...]
        :param vector_down: Lower-state eigenvector. Shape [..., N]
        :param vector_up: Upper-state eigenvector. Shape [..., N]
        :return: Intensity proportional to |<up|Gx|down>|^2, in (J·s/μ_B)²
        """
        magnetization = compute_matrix_element(vector_down, vector_up, Gx).square().abs()
        return magnetization * self._magnetization_scale

    def _compute_magnetization_powder_parallel(
        self,
        Gx: torch.Tensor,
        Gy: torch.Tensor,
        Gz: torch.Tensor,
        res_manifold: torch.Tensor,
        vector_down: torch.Tensor,
        vector_up: torch.Tensor,
    ) -> torch.Tensor:
        """Compute crystal transition intensity.

        The orientation of the wave magnetic field is along the x-axis.
        :param Gx, Gy, Gz: Cartesian components of Zeeman operator. Shape [..., N, N]
        :param res_manifold: Resonance fields or frequencies. Shape [...]
        :param vector_down: Lower-state eigenvector. Shape [..., N]
        :param vector_up: Upper-state eigenvector. Shape [..., N]
        :return: Intensity proportional to  |<up|Gz|down>|², in (J·s/μ_B)²
        """
        magnetization = compute_matrix_element(vector_down, vector_up, Gz)
        return magnetization.abs().square() * self._magnetization_scale

    def _compute_magnetization_crystal_parallel(
        self,
        Gx: torch.Tensor,
        Gy: torch.Tensor,
        Gz: torch.Tensor,
        res_manifold: torch.Tensor,
        vector_down: torch.Tensor,
        vector_up: torch.Tensor,
    ) -> torch.Tensor:
        """Compute crystal transition intensity.

        The orientation of the wave magnetic field is along the x-axis.
        :param Gx, Gy, Gz: Cartesian components of Zeeman operator. Shape [..., N, N]
        :param res_manifold: Resonance fields or frequencies. Shape [...]
        :param vector_down: Lower-state eigenvector. Shape [..., N]
        :param vector_up: Upper-state eigenvector. Shape [..., N]
        :return: Intensity proportional to |<up|Gz|down>|², in (J·s/μ_B)²
        """
        magnetization = compute_matrix_element(vector_down, vector_up, Gz)
        return magnetization.abs().square() * self._magnetization_scale

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
