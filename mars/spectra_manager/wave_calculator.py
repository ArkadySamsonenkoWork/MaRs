import math
import typing as tp

import torch
import torch.nn as nn

from ..population import contexts
from .spectra_manager import compute_matrix_element, StationaryIntensityCalculator
from .. import constants


def wigner_term_square(helicity: int, num: int, theta: torch.Tensor):
    """Compute squared Wigner d-matrix element for EPR transition probability.

    This function calculates orientation-dependent terms for different
    helicity projections in electron paramagnetic resonance simulations.

    :param helicity: Photon helicity (+1 or -1 for circular
        polarization)
    :param num: Spin quantum number projection onto z-axis (-1, 0, or
        +1)
    :param theta: Polar angle between radiation direction and
        quantization axis (radians)
    :return: Squared Wigner term as torch.Tensor
    """
    if helicity == num:
        return torch.pow(torch.cos(theta / 2), 4)

    elif helicity == -num:
        return torch.pow(torch.sin(theta / 2), 4)

    else:
        return torch.pow(torch.sin(theta), 2) / 2


class PlaneWaveTerms(nn.Module):
    """Base module for polarization-dependent term computation for plane
    waves."""
    def __init__(self, polarization: str, theta: float,
                 phi: tp.Optional[float], device: torch.device, dtype: torch.dtype):
        """
        :param polarization: Radiation polarization state.

        Must be one of:
                1) '+1' or '-1' for circular polarization
                2) 'un' for unpolarized radiation
                3) 'lin' for linear polarization
        :param theta: Polar angle between radiation direction and static magnetic field (radians)
        :param phi: An angle between static magnetic field and
        magnetic field of radiation for linear polarization orientation. It is used only fot linear polarization
        :param device: torch.device
        :param dtype: torch.dtype
        """
        super().__init__()
        self.register_buffer("theta", torch.tensor(theta, device=device, dtype=dtype))
        self.register_buffer("phi", torch.tensor(phi, device=device, dtype=dtype))
        self.output_method = self._parse_polarization(polarization)

    def _parse_polarization(self, polarization: str):
        if polarization == "+1" or polarization == "1":
            self.helicity = 1
            return self._circle

        elif polarization == "-1":
            self.helicity = -1
            return self._circle

        elif polarization == "un":
            return self._unpolarized

        elif polarization == "lin":
            return self._linear

        else:
            raise ValueError(
                "polarization must be '+1' or '-1' for circular polarization, 'lin' for linear and 'un' for unpolarized"
            )

    def forward(self, wave_len: tp.Optional[torch.Tensor] = None):
        return self.output_method(wave_len)


class PowderPlaneWaveTerms(PlaneWaveTerms):
    """Polarization weight factors for disordered (powder) samples.

       Implements Eq. (3) from Nehrkorn et al. PRL 114, 010801 (2015).

       Returns (w_xy, w_z, w_mixed) such that

           D = mag_xy * w_xy + mag_z * w_z + mag_mixed * w_mixed

       where:
           mag_xy    = |mu_x|^2 + |mu_y|^2
           mag_z     = |mu_z|^2
           mag_mixed = Im(mu_x * conj(mu_y))

       Wigner identities used (xi_k = cos theta):
           d_pl + d_m  =  (1 + cos^2 theta) / 2
           d_zero      =  sin^2(theta) / 2
           d_pl - d_m  =  helicity * cos(theta)
    """
    def _circle(self, wave_len: tp.Optional[torch.Tensor]):
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
        def _xy_term(helicity: tp.Optional[int], theta: torch.Tensor, phi: torch.Tensor):
            return torch.sin(phi).square()

        def _z_term(helicity: tp.Optional[int], theta: torch.Tensor, phi: torch.Tensor):
            return torch.cos(phi).square() / 2

        def _mixed_term(helicity: tp.Optional[int], theta: torch.Tensor, phi: torch.Tensor):
            return 0.0

        return (
            _xy_term(None, self.theta, self.phi),
            _z_term(None, self.theta, self.phi),
            _mixed_term(None, self.theta, self.phi),
        )


class CrystalPlaneWaveTerms(PlaneWaveTerms):
    """Polarization weight factors for single-crystal samples.

    theta is the angle between the radiation propagation direction k and B0,
    giving n_k = (sin theta, 0, cos theta).

    For circular and unpolarized radiation the intensity depends only on n_k,
    so theta alone fully specifies the geometry and phi is unused.

    For linear polarization B1 must lie in the plane perpendicular to k.
    phi parameterises the rotation of B1 around k within that plane, using
    the natural basis

        e1 = ( cos theta, 0, -sin theta )   (in xz-plane, perpendicular to k)
        e2 = ( 0,         1,  0         )   (y-axis)

    so that

        n1 = cos(phi) * e1 + sin(phi) * e2
           = ( cos(theta)*cos(phi),  sin(phi),  -sin(theta)*cos(phi) )

    Special cases of phi:
        phi=0    -> n1 along e1, which is perpendicular to B0 for any theta.
                    At Voigt (theta=pi/2): n1 = (0, 0, -1) = -z = -B0, i.e.
                    parallel mode.  At Faraday (theta=0): n1 = (1, 0, 0),
                    standard transverse mode.
        phi=pi/2 -> n1 = (0, 1, 0) = y-axis, always perpendicular to B0.

    Because Eq. (2) of Nehrkorn et al. introduces cross terms
    Re(mu_x conj(mu_z)) and Im(mu_y conj(mu_z)) that do not fit in the
    (mag_xy, mag_z, mag_mixed) basis, all three methods return scalar weights
    that are only exact at the special angles but the full computation is
    always delegated to _compute_magnitization_crystal via get_crystal_geometry().
    """

    def get_crystal_geometry(self) -> tuple[torch.Tensor, torch.Tensor, tp.Optional[int]]:
        """Return (n_k, n_1, helicity) geometry tensors.

        n_k : beam propagation direction, shape (3,)
              n_k = (sin theta, 0, cos theta)

        n_1 : B1 oscillation direction for linear polarization, shape (3,)
              n_1 = (cos theta * cos phi,  sin phi,  -sin theta * cos phi)
              (rotation of e1 = (cos theta, 0, -sin theta) by phi around n_k)

        helicity : +1 or -1 for circular, None otherwise
        """
        sin_t = torch.sin(self.theta)
        cos_t = torch.cos(self.theta)
        sin_p = torch.sin(self.phi)
        cos_p = torch.cos(self.phi)

        n_k = torch.stack([sin_t, torch.zeros_like(sin_t), cos_t])

        n_1 = torch.stack([cos_t * cos_p, sin_p, -sin_t * cos_p])

        helicity = getattr(self, "helicity", None)
        return n_k, n_1, helicity

    def _circle(self, wave_len: tp.Optional[torch.Tensor]):
        return None, None, None

    def _unpolarized(self, wave_len: tp.Optional[torch.Tensor]):
        return None, None, None

    def _linear(self, wave_len: tp.Optional[torch.Tensor]):
        return None, None, None


class WaveIntensityCalculator(StationaryIntensityCalculator):
    """Computes the intensity of transitions for general type of radiation,
    when the radiation has different orientation.

    with respect to static magnetic field.

    Handles calculation of transition intensities based on:
    - Transition matrix elements (magnetization).
    - Level populations. Uses Boltzmann thermal populations at specified temperature
      or predefined population given in context.
    """
    def __init__(self,
                 spin_system_dim: int, disordered: bool,
                 polarization: str, theta: float, phi: tp.Optional[float] = math.pi / 2,
                 terms_computer: tp.Optional[
                     tp.Callable[
                         [tp.Any],
                         tuple[
                             tp.Union[float, torch.Tensor],
                             tp.Union[float, torch.Tensor],
                             tp.Union[float, torch.Tensor]
                         ]
                     ]
                 ] = None,
                 temperature: tp.Optional[float] = 293.0,
                 populator: tp.Optional[tp.Callable] = None,
                 context: tp.Optional[contexts.BaseContext] = None,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32
                 ):
        """
        :param spin_system_dim: The dimension of a spin system.

        :param disordered: The flag is used for powder averaging

        :param polarization:  The polarization of radiation. It should be one of the variants:
               1) '+1' or '-1' for circular polarization
               2) 'un' for unpolarized radiation
               3) 'lin' for linear polarization. In this case the phi angle defined the direction of the polarization

        :param theta: The angle between radiation direction and stationary magnetic field

        :param phi: Meaning depends on sample type:

            Powder, linear:  angle between B1 and B0.
                phi=pi/2 -> perpendicular mode (B1 _|_ B0)
                phi=0    -> parallel mode      (B1 || B0)

            Crystal, linear:  rotation of B1 around k (see CrystalPlaneWaveTerms).
                In this case the orientation of the k-vector and oscillating magnetic field is:
                  n_k : beam propagation direction
                  n_k = (sin theta, 0, cos theta)
                  n_1 : B1 oscillation direction for linear polarization, shape (3,)
                  n_1 = (cos theta * cos phi,  sin phi,  -sin theta * cos phi)
                phi=0    -> B1 in the xz-plane, perpendicular to k
                phi=pi/2 -> B1 along y

            Circular / unpolarized:  phi is unused.

        :param terms_computer:
            Callable that computes the polarization-dependent weight factors for the three magnetization components:
                - xy-component (transverse in-plane),
                - z-component (longitudinal),
                - mixed xy-phase term (imaginary coherence).
            The callable must accept a single optional argument (e.g., transition energy or wavelength in Hz)
            and return a 3-tuple of scalars or tensors: (w_xy, w_z, w_mixed).
            If None, a default plane-wave-based implementation is used:
                • PowderPlaneWaveTerms for disordered (powder) samples,
                • CrystalPlaneWaveTerms for single-crystal samples.

        :param temperature: The temperature of an experiment. If populator is not None it takes from it
        :param populator:
            Class that is used to compute part intensity due to population of levels. Default is None
            If it is None then it will be initialized as default calculator specific to given intensity_calculator

        :param context: Optional[context]
            The instance of BaseContext which describes the relaxation mechanism.
            It can have the initial population logic, transition between energy levels, dephasing, driven transition,
            out system transitions. For more complicated scenario the full relaxation superoperator can be used.

        :param device: torch.device
        :param dtype: torch.dtype
        """
        super().__init__(
            spin_system_dim, temperature, populator, context, disordered, device=device, dtype=dtype)
        self.terms_computer = self._init_terms_computer(terms_computer=terms_computer, disordered=disordered,
                                                        theta=theta, phi=phi, polarization=polarization,
                                                        device=device, dtype=dtype)

    def _init_terms_computer(self,
                             polarization: str, theta: float, phi: tp.Optional[float],
                             terms_computer: tp.Optional[tp.Callable], disordered: bool,
                             device: torch.device, dtype: torch.dtype
                             ):
        if terms_computer is None:
            if disordered:
                return PowderPlaneWaveTerms(
                    theta=theta, phi=phi, polarization=polarization, device=device, dtype=dtype
            )
            return CrystalPlaneWaveTerms(
                theta=theta, phi=phi, polarization=polarization, device=device, dtype=dtype
            )

        else:
            return terms_computer

    def _compute_magnitization_crystal(
            self, Gx: torch.Tensor, Gy: torch.Tensor, Gz: torch.Tensor,
            res_manifold: torch.Tensor,
            vector_down: torch.Tensor, vector_up: torch.Tensor):
        """Single-crystal D from Eq. (2) of Nehrkorn et al.
        All projections are evaluated directly from mu, including the cross
        terms Re(mu_x conj(mu_z)) and Im(mu_y conj(mu_z)) that are non-zero
        at general theta.
        """
        n_k, n_1, helicity = self.terms_computer.get_crystal_geometry()

        mu_x = compute_matrix_element(vector_down, vector_up, -Gx)
        mu_y = compute_matrix_element(vector_down, vector_up, -Gy)
        mu_z = compute_matrix_element(vector_down, vector_up, -Gz)

        is_linear = (
                self.terms_computer.output_method is self.terms_computer._linear
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

        return out * (constants.PLANCK / constants.BOHR) ** 2

    def _compute_magnitization_powder(
            self, Gx: torch.Tensor, Gy: torch.Tensor, Gz: torch.Tensor,
            res_manifold: torch.Tensor,
            vector_down: torch.Tensor, vector_up: torch.Tensor,) -> torch.Tensor:
        mu_x = compute_matrix_element(vector_down, vector_up, -Gx)
        mu_y = compute_matrix_element(vector_down, vector_up, -Gy)
        mu_z = compute_matrix_element(vector_down, vector_up, -Gz)

        magnitization_xy = mu_x.square().abs() + mu_y.square().abs()
        magnitization_z = mu_z.square().abs()
        magnitization_mixed = (mu_x * mu_y.conj()).imag

        terms = self.terms_computer(res_manifold)
        out = magnitization_xy * terms[0] + magnitization_z * terms[1] + magnitization_mixed * terms[2]
        return out * (constants.PLANCK / constants.BOHR) ** 2

    def compute_intensity(
            self, Gx: torch.Tensor, Gy: torch.Tensor, Gz: torch.Tensor,
            res_manifold: torch.Tensor,
            lvl_down: torch.Tensor, lvl_up: torch.Tensor,
            resonance_energies: torch.Tensor,
            vector_down: torch.Tensor, vector_up: torch.Tensor,
            full_system_vectors: tp.Optional[torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        """Compute  EPR intensities under polarized radiation.

        :param Gx, Gy, Gz: Zeeman operator components
        :param vector_down, vector_up: Eigenvectors of lower/upper states
        :param ...: Other parameters (unused here but kept for interface consistency)
        :return: Magnetization-squared term, shape [...]
        """
        intensity = self.populator(res_manifold, lvl_down, lvl_up,
                                   resonance_energies, full_system_vectors,
                                   *args, ** kwargs) * (
                self._compute_magnitization(Gx, Gy, Gz, res_manifold, vector_down, vector_up)
        )
        return intensity
