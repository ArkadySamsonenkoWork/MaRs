import math
import typing as tp

import torch
import torch.nn as nn

from ..population import contexts, BasePopulator, StationaryPopulator, BaseTimeDepPopulator, LevelBasedPopulator
from .spectra_manager import BaseResIntensityCalculator
from .magnetization_mode import Polarization, WaveMagnetizationConfig, MagnetizationConfig
from .utils import ComputationalDetails, compute_matrix_element


def wigner_term_square(helicity: int, num: int, theta: torch.Tensor):
    """
    Compute a squared Wigner small-d element used in powder polarization weights.

    The Wigner d-matrix describes how spherical magnetic-dipole components transform
    when the quantization axis is rotated by theta. Here theta is the angle between
    the wave propagation direction k and B0.

    For photon helicity ``h = +/-1`` and spherical component ``m = +1, 0, -1``,
    the required squared elements reduce to

    ``|d^1_{h,h}(theta)|^2 = cos(theta/2)^4``

    ``|d^1_{h,-h}(theta)|^2 = sin(theta/2)^4``

    ``|d^1_{h,0}(theta)|^2 = sin(theta)^2 / 2``.

    These terms are combined by ``PowderPlaneWaveTerms`` to build transverse,
    longitudinal and helicity-sensitive powder weights.

    :param helicity:
        Photon helicity, +1 or -1.

    :param num:
        Spherical magnetic-dipole component, +1, 0 or -1.

    :param theta:
        Angle, in radians, between k and B0.

    :return:
        Squared Wigner d-matrix element.
    """
    if helicity == num:
        return torch.pow(torch.cos(theta / 2), 4)

    elif helicity == -num:
        return torch.pow(torch.sin(theta / 2), 4)

    else:
        return torch.pow(torch.sin(theta), 2) / 2


class PlaneWaveTerms(nn.Module):
    """
    Base module for polarization-dependent geometrical terms of an incident plane
    wave.

    This class represents only the radiation geometry and polarization-dependent
    weights. It does not compute transition matrix elements itself.

    The static magnetic field B0 defines the z-axis. The propagation direction is

    ``n_k = (sin(theta), 0, cos(theta))``.

    Thus:

    - ``theta = 0`` corresponds to Faraday geometry, ``k || B0``;
    - ``theta = pi/2`` corresponds to Voigt geometry, ``k _|_ B0``.

    For linear polarization, B1 is perpendicular to k and its direction is
    determined by ``phi``. The orthonormal basis of the plane perpendicular to k is

    ``e1 = (cos(theta), 0, -sin(theta))``
    ``e2 = (0, 1, 0)``.
    The linear-polarization direction is
    ``n_1 = cos(phi) * e1 + sin(phi) * e2``

    or

    ``n_1 = (
        cos(theta) * cos(phi),
        sin(phi),
        -sin(theta) * cos(phi)
    )``.

    The actual B1-B0 angle alpha satisfies

    ``cos(alpha) = -sin(theta) * cos(phi)``.

    For circular polarization, B1 rotates in the plane perpendicular to k and the
    relevant discrete parameter is helicity, +1 or -1.

    Computationally, this base class normalizes the polarization, stores ``theta``
    and ``phi`` as PyTorch buffers, and dispatches ``forward`` to one of
    ``_circle``, ``_unpolarized`` or ``_linear``.
    """

    def __init__(
            self,
            config: WaveMagnetizationConfig,
            device: torch.device,
            dtype: torch.dtype,
    ):
        """
        Initialize polarization dispatch and tensor representation of the wave geometry.

        ``theta`` is copied from the configuration and represents the angle between k
        and B0.

        For linear polarization, ``phi`` is copied from the configuration and
        represents rotation of B1 around k in the plane perpendicular to k. It is not
        the B1-B0 angle.

        The method also normalizes the polarization enum, determines helicity for
        circular polarization, and chooses the method used by ``forward``.

        :param config:
            Incident-wave geometry and polarization configuration.

        :param device:
            Device used for theta and phi buffers.

        :param dtype:
            Floating-point dtype of theta and phi buffers.
        """
        super().__init__()
        self.config = config
        self.polarization = config.normalized_polarization()

        phi_value = math.pi / 2 if config.phi is None else config.phi
        self.register_buffer(
            "theta",
            torch.tensor(config.theta, device=device, dtype=dtype),
        )
        self.register_buffer(
            "phi",
            torch.tensor(phi_value, device=device, dtype=dtype),
        )

        self.helicity = config.helicity()
        self.output_method = self._parse_polarization(self.polarization)

    def _parse_polarization(self, polarization: Polarization):
        if polarization in (
                Polarization.CIRCULAR_PLUS,
                Polarization.CIRCULAR_MINUS,
        ):
            return self._circle
        if polarization == Polarization.UNPOLARIZED:
            return self._unpolarized
        if polarization == Polarization.LINEAR:
            return self._linear

        raise ValueError(f"Unsupported polarization: {polarization}")

    def forward(self, wave_len: tp.Optional[torch.Tensor] = None):
        return self.output_method(wave_len)


class PowderPlaneWaveTerms(PlaneWaveTerms):
    """
    Compute polarization-dependent geometrical weights for a disordered powder.

    For a powder, all molecular/crystal orientations are averaged. After this
    orientational averaging, the transition magnetic moment enters through three
    rotationally organized combinations:

    ``mag_xy = |mu_x|^2 + |mu_y|^2``

    ``mag_z = |mu_z|^2``

    ``mag_mixed = Im(mu_x * conj(mu_y))``.

    The geometrical radiation factor is written as

    ``D = mag_xy*w_xy + mag_z*w_z + mag_mixed*w_mixed``.

    This class computes the tuple ``(w_xy, w_z, w_mixed)`` for the selected
    polarization.

    For circular polarization, the angular dependence is expressed through squared
    Wigner d-matrix elements for a spin-1 rotation by theta. With helicity
    ``h = +/-1``,

    ``d_+ + d_- = (1 + cos(theta)^2) / 2``
    ``d_0 = sin(theta)^2 / 2``
    ``d_+ - d_- = h*cos(theta)``.

    The last combination changes sign with helicity and therefore contains the
    handedness-sensitive contribution.

    For unpolarized radiation, opposite helicities are averaged. The
    helicity-odd mixed term cancels, so ``w_mixed = 0``.

    For linear polarization, the key scalar is the projection of B1 onto B0:

    ``xi_1 = n_1 . n_0 = -sin(theta)*cos(phi)``.

    The implementation uses

    ``w_xy = 1 - xi_1^2``

    ``w_z = xi_1^2 / 2``

    ``w_mixed = 0``.

    Consequently, linear powder intensity depends on both theta and phi. This
    follows from the chosen physical meaning of phi as rotation around k.
    """

    def _circle(self, wave_len: tp.Optional[torch.Tensor]):
        """
        Compute powder weights for circular polarization.

        The circularly polarized wave is represented by photon helicity +1 or -1.
        Squared Wigner d-matrix elements evaluated at theta determine how the spherical
        magnetic-dipole components contribute after powder averaging.

        The returned weights multiply

        ``(|mu_x|^2 + |mu_y|^2, |mu_z|^2, Im(mu_x*conj(mu_y)))``.

        The mixed weight changes sign with helicity and therefore distinguishes the
        two circular polarizations.

        :param wave_len:
            Optional transition-dependent argument kept for interface compatibility.

        :return:
            Tuple ``(w_xy, w_z, w_mixed)``.
        """

        def _xy_term(
                helicity: int, theta: torch.Tensor, phi: torch.Tensor,
                wigners: tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
            return (wigners[0] + wigners[2]) / 2

        def _z_term(
                helicity: int, theta: torch.Tensor, phi: torch.Tensor,
                wigners: tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
            return wigners[1]

        def _mixed_term(
                helicity: int, theta: torch.Tensor, phi: torch.Tensor,
                wigners: tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
            return wigners[0] - wigners[2]

        d_pl = wigner_term_square(self.helicity, 1, self.theta)
        d_zero = wigner_term_square(self.helicity, 0, self.theta)
        d_m = wigner_term_square(self.helicity, -1, self.theta)
        wigners = (d_pl, d_zero, d_m)
        return (
            _xy_term(self.helicity, self.theta, self.phi, wigners),
            _z_term(self.helicity, self.theta, self.phi, wigners),
            _mixed_term(self.helicity, self.theta, self.phi, wigners),
        )

    def _unpolarized(self, wave_len: tp.Optional[torch.Tensor]):
        """
        Compute powder weights for unpolarized radiation.

        Unpolarized radiation is treated as an equal average over opposite circular
        helicities. The helicity-odd mixed contribution therefore cancels exactly,
        while the transverse and longitudinal weights retain their theta dependence.

        :param wave_len:
            Optional transition-dependent argument kept for interface compatibility.

        :return:
            Tuple ``(w_xy, w_z, 0)``.
        """

        def _xy_term(
                helicity: tp.Optional[int], theta: torch.Tensor, phi: torch.Tensor,
                wigners: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ):
            return (wigners[0] + wigners[2]) / 4

        def _z_term(
                helicity: tp.Optional[int], theta: torch.Tensor, phi: torch.Tensor,
                wigners: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ):
            return wigners[1] / 2

        def _mixed_term(
                helicity: tp.Optional[int], theta: torch.Tensor, phi: torch.Tensor,
                wigners: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ):
            return 0.0

        helicity = 1  # It can be 1 or -1 for this case. It doesn't matter
        d_pl = wigner_term_square(1, 1, self.theta)
        d_zero = wigner_term_square(1, 0, self.theta)
        d_m = wigner_term_square(1, -1, self.theta)
        wigners = (d_pl, d_zero, d_m)
        return (
            _xy_term(helicity, self.theta, self.phi, wigners),
            _z_term(helicity, self.theta, self.phi, wigners),
            _mixed_term(helicity, self.theta, self.phi, wigners),
        )

    def _linear(self, wave_len: tp.Optional[torch.Tensor]):
        """
        Compute powder weights for linear polarization.

        The B1 direction is

        ``n_1 = (
            cos(theta)*cos(phi),
            sin(phi),
            -sin(theta)*cos(phi)
        )``.

        Its projection onto B0 is

        ``xi_1 = n_1 . n_0 = -sin(theta)*cos(phi)``.

        The implementation uses

        ``w_xy = 1 - xi_1^2``
        ``w_z = xi_1^2 / 2``
        ``w_mixed = 0``.

        Thus both theta and phi enter the linear-polarization powder intensity.

        :param wave_len:
            Optional transition-dependent argument kept for interface compatibility.

        :return:
            Tuple ``(w_xy, w_z, w_mixed)``.
        """
        xi_1 = -torch.sin(self.theta) * torch.cos(self.phi)
        xi_1_sq = xi_1.square()

        return (
            1.0 - xi_1_sq,
            xi_1_sq / 2.0,
            0.0,
        )


class CrystalPlaneWaveTerms(PlaneWaveTerms):
    """
    Polarization-dispatch helper for single-crystal incident-wave calculations.

    A single crystal is not orientationally averaged. Therefore the complete
    complex transition magnetic-moment vector

    ``mu = (mu_x, mu_y, mu_z)``

    must be projected directly onto the incident-wave geometry.

    The wave geometry is not computed in this class. It is supplied by
    ``WaveMagnetizationConfig.compute_wave_geometry()``:

    ``n_k`` is the unit propagation direction,
    ``n_1`` is the unit B1 direction for linear polarization,

    and ``helicity`` is +1 or -1 for circular polarization.

    For linear polarization, the coupling factor is

    ``D_linear = |n_1 . mu|^2``.

    For unpolarized radiation, only the part of mu transverse to the propagation
    direction can couple to the wave:

    ``D_un = 1/2 * (|mu|^2 - |n_k . mu|^2)``.

    For circular polarization, the same transverse contribution is supplemented by
    a helicity-dependent phase term constructed from imaginary cross products of
    the Cartesian transition moments.

    The three private polarization methods in this class return placeholders
    because the complete crystal expressions are evaluated directly in
    ``WaveIntensityCalculator._compute_magnetization_crystal()``. The class remains
    as the polarization-dispatch implementation required by the common
    ``PlaneWaveTerms`` interface.
    """

    def _circle(self, wave_len: tp.Optional[torch.Tensor]):
        return None, None, None

    def _unpolarized(self, wave_len: tp.Optional[torch.Tensor]):
        return None, None, None

    def _linear(self, wave_len: tp.Optional[torch.Tensor]):
        return None, None, None


class WaveIntensityCalculator(BaseResIntensityCalculator):
    """
    Compute stationary EPR transition intensities for a propagating electromagnetic
    wave incident on the sample.

    The total transition intensity is factorized into a population contribution
    and a radiation-coupling contribution:

    ``I = population_factor * D``.

    The population factor is produced by the configured stationary populator.

    The radiation-coupling factor D is computed from the Cartesian transition
    magnetic-dipole matrix elements

    ``mu_x = <up| -Gx |down>``

    ``mu_y = <up| -Gy |down>``

    ``mu_z = <up| -Gz |down>``.

    These form the complex transition moment vector

    ``mu = (mu_x, mu_y, mu_z)``.

    For a powder sample, the vector is reduced to orientationally averaged
    combinations and multiplied by the weights returned by
    ``PowderPlaneWaveTerms``.

    For a crystal, the complete vector is projected directly onto the physical
    wave geometry supplied by ``WaveMagnetizationConfig``.

    Linear crystal polarization uses

    ``D = |n_1 . mu|^2``.

    Unpolarized crystal radiation uses

    ``D = 1/2 * (|mu|^2 - |n_k . mu|^2)``.

    """

    def __init__(self,
                 spin_system_dim: tp.Union[int, list[int]],
                 temperature: tp.Optional[float] = None,
                 populator: tp.Optional[tp.Union[BasePopulator, str]] = None,
                 context: tp.Optional[contexts.BaseContext] = None,
                 disordered: bool = True,
                 magnetization_config: WaveMagnetizationConfig = WaveMagnetizationConfig(),
                 computational_details: ComputationalDetails = ComputationalDetails(),
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32):
        """
        :param spin_system_dim:
            Dimension of the spin-system Hilbert space.

        :param temperature:
            Sample temperature used by the default population model.

        :param populator:
            Optional custom population calculator.

        :param context:
            Optional population/relaxation context.

        :param disordered:
            ``True`` for powder averaging, ``False`` for a single crystal.

        :param magnetization_config:
            Must be ``WaveMagnetizationConfig``.

            The configuration describes:

            - radiation polarization;
            - ``theta``, the angle between propagation direction k and B0;
            - ``phi``, the linear-polarization rotation around k;
            - an optional custom ``terms_computer``.

        :param device:
            Computation device.

        :param dtype:
            Floating-point dtype.
        """

        super().__init__(
            spin_system_dim=spin_system_dim,
            temperature=temperature,
            populator=populator,
            context=context,
            disordered=disordered,
            magnetization_config=magnetization_config,
            computational_details=computational_details,
            device=device,
            dtype=dtype,
        )

        self.terms_computer = self._init_terms_computer(
            config=magnetization_config,
            disordered=disordered,
            device=device,
            dtype=dtype,
        )

    def _magnetization_factory(
            self,
            disordered: bool,
            magnetization_config: MagnetizationConfig,
            computational_details: ComputationalDetails,
            device: torch.device,
            dtype: torch.dtype,
    ) -> tp.Callable[..., torch.Tensor]:
        """
        Select the magnetization computation for an incident-wave experiment.

        Unlike conventional resonator calculations, an incident-wave experiment
        does not select between a predefined perpendicular and parallel B1 mode.
        The actual orientation of the electromagnetic wave is described
        continuously by ``WaveMagnetizationConfig`` through polarization,
        ``theta`` and ``phi``.

        The factory therefore selects only the sample-orientation treatment:

        - powder/disordered sample -> ``_compute_magnetization_powder``;
        - single crystal -> ``_compute_magnetization_crystal``.

        The selected method subsequently uses the complete wave geometry from
        ``WaveMagnetizationConfig``.

        :param disordered:
            ``True`` for powder averaging and ``False`` for a single crystal.

        :param magnetization_config:
            Must be ``WaveMagnetizationConfig``.

            The configuration describes:

            - radiation polarization;
            - ``theta``, the angle between propagation direction k and B0;
            - ``phi``, the linear-polarization rotation around k;
            - an optional custom ``terms_computer``.

        :param computational_details:
            Numerical configuration. Not currently required for selecting the
            wave magnetization method.

        :param device:
            Computation device.

        :param dtype:
            Floating-point dtype.

        :return:
            ``_compute_magnetization_powder`` when ``disordered=True`` or
            ``_compute_magnetization_crystal`` otherwise.

        :raises TypeError:
            If ``magnetization_config`` is not ``WaveMagnetizationConfig``.
        """
        if not isinstance(
                magnetization_config,
                WaveMagnetizationConfig,
        ):
            raise TypeError(
                "WaveIntensityCalculator requires WaveMagnetizationConfig, got "
                f"{type(magnetization_config).__name__}."
            )

        if disordered:
            return self._compute_magnetization_powder

        return self._compute_magnetization_crystal

    def _init_terms_computer(
            self,
            config: WaveMagnetizationConfig,
            disordered: bool,
            device: torch.device,
            dtype: torch.dtype,
    ):
        """
        Initialize the polarization-dependent term computer.

        :param config:
            Wave configuration containing polarization, ``theta``, ``phi``,
            and optional ``terms_computer``.
        :param disordered:
            ``True`` for powder calculations, ``False`` for crystal
            calculations.
        :param device:
            Device for internal angle tensors.
        :param dtype:
            Floating-point dtype for internal angle tensors.
        :return:
            User-provided terms computer, or the default powder/crystal
            implementation.
        """
        if config.terms_computer is not None:
            return config.terms_computer

        if disordered:
            return PowderPlaneWaveTerms(
                config=config,
                device=device,
                dtype=dtype,
            )

        return CrystalPlaneWaveTerms(
            config=config,
            device=device,
            dtype=dtype,
        )

    def _compute_magnetization_crystal(
            self, Gx: torch.Tensor, Gy: torch.Tensor, Gz: torch.Tensor,
            res_manifold: torch.Tensor,
            vector_down: torch.Tensor, vector_up: torch.Tensor):
        """
        Compute the radiation-coupling factor for a single-crystal transition.

        First compute the Cartesian transition amplitudes

        ``mu_x = <up| -Gx |down>``
        ``mu_y = <up| -Gy |down>``
        ``mu_z = <up| -Gz |down>``.

        For linear polarization,
        ``D = |n_1 . mu|^2``.

        For unpolarized radiation,
        ``D = 1/2 * (|mu|^2 - |n_k . mu|^2)``.

        For circular polarization, the code combines the transverse contribution with
        the helicity-dependent term

        ``2*h*(n_k_x*Im(mu_y*conj(mu_z))
               + n_k_z*Im(mu_x*conj(mu_y)))``.

        This term carries the phase sensitivity and distinguishes opposite
        helicities.

        The result is scaled by ``(PLANCK/BOHR)^2``.

        :param Gx, Gy, Gz:
            Cartesian magnetic-dipole/Zeeman operator components.

        :param res_manifold:
            Resonance manifold. It is not used directly by this crystal geometry
            formula but is kept for interface compatibility.

        :param vector_down:
            Lower-state eigenvector.

        :param vector_up:
            Upper-state eigenvector.

        :return:
            Crystal radiation-coupling factor.
        """
        n_k, n_1, helicity = self.wave_config.compute_wave_geometry(
            device=Gx.device,
            dtype=Gx.real.dtype,
        )

        mu_x = compute_matrix_element(vector_down, vector_up, -Gx)
        mu_y = compute_matrix_element(vector_down, vector_up, -Gy)
        mu_z = compute_matrix_element(vector_down, vector_up, -Gz)

        is_linear = (
                self.wave_config.normalized_polarization() == Polarization.LINEAR
        )

        if is_linear:
            n1_dot_mu = n_1[0] * mu_x + n_1[1] * mu_y + n_1[2] * mu_z
            out = n1_dot_mu.abs().square()

        elif helicity is None:
            nk_dot_mu = n_k[0] * mu_x + n_k[1] * mu_y + n_k[2] * mu_z
            mu_sq = mu_x.abs().square() + mu_y.abs().square() + mu_z.abs().square()
            out = 0.5 * (mu_sq - nk_dot_mu.abs().square())

        else:
            nk_dot_mu = n_k[0] * mu_x + n_k[1] * mu_y + n_k[2] * mu_z
            mu_sq = mu_x.abs().square() + mu_y.abs().square() + mu_z.abs().square()
            D_un = 0.5 * (mu_sq - nk_dot_mu.abs().square())

            cross_z = (mu_x * mu_y.conj()).imag  # Im(mu_x conj(mu_y))
            cross_x = (mu_y * mu_z.conj()).imag  # Im(mu_y conj(mu_z))
            nk_cross = n_k[0] * cross_x + n_k[2] * cross_z

            out = 2.0 * D_un + 2.0 * helicity * nk_cross

        return out * self._magnetization_scale

    def _compute_magnetization_powder(
            self, Gx: torch.Tensor, Gy: torch.Tensor, Gz: torch.Tensor,
            res_manifold: torch.Tensor,
            vector_down: torch.Tensor, vector_up: torch.Tensor, ) -> torch.Tensor:
        """
        Compute the radiation-coupling factor for a powder transition.

        The Cartesian matrix elements are reduced to

        ``mag_xy = |mu_x|^2 + |mu_y|^2``

        ``mag_z = |mu_z|^2``

        ``mag_mixed = Im(mu_x*conj(mu_y))``.

        The configured terms computer returns

        ``(w_xy, w_z, w_mixed)``,

        and the powder factor is

        ``D = mag_xy*w_xy + mag_z*w_z + mag_mixed*w_mixed``.

        The result is scaled by ``(PLANCK/BOHR)^2``.

        :param Gx, Gy, Gz:
            Cartesian magnetic-dipole/Zeeman operator components.

        :param res_manifold:
            Resonance-dependent argument passed to the terms computer.

        :param vector_down:
            Lower-state eigenvector.

        :param vector_up:
            Upper-state eigenvector.

        :return:
            Powder-averaged radiation-coupling factor.
        """
        mu_x = compute_matrix_element(vector_down, vector_up, -Gx)
        mu_y = compute_matrix_element(vector_down, vector_up, -Gy)
        mu_z = compute_matrix_element(vector_down, vector_up, -Gz)

        magnetization_xy = mu_x.square().abs() + mu_y.square().abs()
        magnetization_z = mu_z.square().abs()
        magnetization_mixed = (mu_x * mu_y.conj()).imag

        terms = self.terms_computer(res_manifold)
        out = magnetization_xy * terms[0] + magnetization_z * terms[1] + magnetization_mixed * terms[2]
        return out * self._magnetization_scale

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

    def compute_intensity(
            self, Gx: torch.Tensor, Gy: torch.Tensor, Gz: torch.Tensor,
            res_manifold: torch.Tensor,
            lvl_down: torch.Tensor, lvl_up: torch.Tensor,
            resonance_energies: torch.Tensor,
            vector_down: torch.Tensor, vector_up: torch.Tensor,
            full_system_vectors: tp.Optional[torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        """
        Compute final stationary transition intensities.

        The intensity is the product

        ``I = population_factor * magnetization_factor``.

        The population factor is computed by ``self.populator``. The magnetization
        factor is computed by the powder or crystal path selected by the calculator.

        :param Gx, Gy, Gz:
            Cartesian magnetic-dipole/Zeeman operator components.

        :param res_manifold:
            Resonance fields or frequencies.

        :param lvl_down:
            Lower-state indices.

        :param lvl_up:
            Upper-state indices.

        :param resonance_energies:
            Energies passed to the population model.

        :param vector_down:
            Lower-state eigenvectors.

        :param vector_up:
            Upper-state eigenvectors.

        :param full_system_vectors:
            Optional complete eigenvector basis required by some population models.

        :return:
            Transition intensity tensor.
        """
        intensity = self.populator(res_manifold, lvl_down, lvl_up,
                                   resonance_energies, full_system_vectors,
                                   *args, **kwargs) * (
                        self.compute_magnetization(Gx, Gy, Gz, res_manifold, vector_down, vector_up)
                    )
        return intensity


class WaveTimeIntensityCalculator(WaveIntensityCalculator):
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
                 magnetization_config: WaveMagnetizationConfig = WaveMagnetizationConfig(),
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
            Must be ``WaveMagnetizationConfig``.

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
