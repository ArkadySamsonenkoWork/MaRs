from enum import Enum
import typing as tp
import math
from dataclasses import dataclass

import torch


WaveTerms = tuple[
    tp.Union[float, torch.Tensor],
    tp.Union[float, torch.Tensor],
    tp.Union[float, torch.Tensor],
]
WaveTermsComputer = tp.Callable[[tp.Optional[torch.Tensor]], WaveTerms]


class ResonatorMode(str, Enum):
    """Microwave magnetic-field mode in a conventional EPR resonator."""

    PERPENDICULAR = "perpendicular"
    PARALLEL = "parallel"

    @classmethod
    def from_value(
        cls,
        value: "ResonatorMode | str",
    ) -> "ResonatorMode":
        """Return ``value`` normalized to a :class:`ResonatorMode`."""
        if isinstance(value, cls):
            return value
        return cls(value)


class Polarization(str, Enum):
    """Polarization of an incident electromagnetic wave."""

    CIRCULAR_PLUS = "+1"
    CIRCULAR_MINUS = "-1"
    UNPOLARIZED = "un"
    LINEAR = "lin"

    @classmethod
    def from_value(
            cls,
            value: "Polarization | str",
    ) -> "Polarization":
        """Return ``value`` normalized to a :class:`Polarization`.

        Both compact values (``"lin"``, ``"un"``) and readable aliases
        (``"linear"``, ``"right-hand polarized"``, ``"left-hand polarized"``,
        ``"clockwise"``, ``"counter-clockwise"``) are accepted. Matching is
        case-insensitive and surrounding whitespace is ignored.

        Circular-polarization aliases follow the source-view convention used
        by Wikipedia for the IEEE/quantum-physics convention: the observer
        looks in the direction of propagation. With that convention:

        - right-hand / clockwise -> ``+1`` helicity;
        - left-hand / counter-clockwise -> ``-1`` helicity.
        """
        if isinstance(value, cls):
            return value

        aliases = {
            "1": "+1",
            "+1": "+1",
            "plus 1": "+1",
            "circular plus": "+1",
            "positive helicity": "+1",
            "right": "+1",
            "right hand": "+1",
            "right handed": "+1",
            "right hand polarized": "+1",
            "right handed polarized": "+1",
            "right circular": "+1",
            "rhcp": "+1",
            "clockwise": "+1",
            "cw": "+1",

            "-1": "-1",
            "minus 1": "-1",
            "circular minus": "-1",
            "negative helicity": "-1",
            "left": "-1",
            "left hand": "-1",
            "left handed": "-1",
            "left hand polarized": "-1",
            "left handed polarized": "-1",
            "left circular": "-1",
            "lhcp": "-1",
            "counter clockwise": "-1",
            "counterclockwise": "-1",
            "anti clockwise": "-1",
            "anticlockwise": "-1",
            "ccw": "-1",

            "linear": "lin",
            "linearly polarized": "lin",
            "lin": "lin",

            "unpolarised": "un",
            "un": "un",
        }
        normalized = aliases.get(value, value)
        try:
            return cls(normalized)
        except ValueError:
            raise ValueError(
                f"Unsupported polarization {value!r}. "
                "Expected one of: linear ('lin' or 'linear'), "
                "unpolarized ('un' or 'unpolarized'), "
                "right-hand circular ('+1', 'right-hand polarized', 'clockwise', or 'rhcp'), "
                "or left-hand circular ('-1', 'left-hand polarized', "
                "'counter-clockwise', or 'lhcp')."
            ) from None


@dataclass(frozen=True)
class ResonatorMagnetizationConfig:
    """
    Configuration of the microwave magnetic-field geometry for a conventional
    EPR resonator experiment.

    In this experimental setup the sample is placed inside a resonator and the
    transition magnetization is determined by the orientation of the oscillating
    microwave magnetic field B1 relative to the static magnetic field B0.

    :param mode:
       Orientation of the oscillating microwave magnetic field B1 relative to
       the static magnetic field B0.

       Supported modes are:

       - ``MagneticFieldMode.PERPENDICULAR``:
         B1 is perpendicular to B0. This is the conventional EPR excitation
         geometry.

       - ``MagneticFieldMode.PARALLEL``:
         B1 is parallel to B0. The transition magnetization is therefore
         computed from the component parallel to the static magnetic field.

       Default is ``MagneticFieldMode.PERPENDICULAR``.
   """
    mode: tp.Union[ResonatorMode, str] = ResonatorMode.PERPENDICULAR

    def normalized_mode(self) -> ResonatorMode:
        """Return ``mode`` as a normalized :class:`ResonatorMode`."""
        return ResonatorMode.from_value(self.mode)


@dataclass(frozen=True)
class WaveMagnetizationConfig:
    """
    Magnetization configuration for an experiment in which a propagating
    electromagnetic wave is incident on the sample.

    The wave geometry is described relative to the static magnetic field B0.
    The propagation vector ``k`` lies in the xz-plane and the direction of B0
    defines the z-axis.

    :param polarization:
        Polarization state of the incident electromagnetic wave.

        Supported values are:

        - ``Polarization.CIRCULAR_PLUS``, ``"+1"``, ``"right-hand polarized"``
          or ``"clockwise"`` for positive-helicity circular polarization;
        - ``Polarization.CIRCULAR_MINUS``, ``"-1"``, ``"left-hand polarized"``
          or ``"counter-clockwise"`` for negative-helicity circular polarization;
        - ``Polarization.UNPOLARIZED`` or ``"un"`` for unpolarized radiation;
        - ``Polarization.LINEAR`` or ``"lin"`` for linearly polarized
          radiation.

        For linear polarization, ``phi`` determines the orientation of the
        oscillating magnetic field B1 in the plane perpendicular to ``k``.

        Default is ``Polarization.LINEAR``.


    :param theta:
        Polar angle, in radians, between the wave propagation direction ``k``
        and the static magnetic field B0.

        With B0 along the z-axis, the unit propagation vector is

        ``n_k = (sin(theta), 0, cos(theta))``.

        Examples:

        - ``theta = 0``:
          Faraday geometry, ``k || B0``. Since B1 must be perpendicular to
          ``k``, B1 is also perpendicular to B0 for every linear-polarization
          direction.

        - ``theta = pi / 2``:
          Voigt geometry, ``k _|_ B0``. In this case ``phi`` can rotate B1
          from parallel to B0 to perpendicular to B0.

        - ``theta = pi / 4``:
          Intermediate geometry. The propagation direction is tilted by
          45 degrees relative to B0.

        Default is ``0.0``.

    :param phi:
        Linear-polarization angle, in radians, describing rotation of B1
        around the propagation direction ``k``.

        ``phi`` is measured in the plane perpendicular to ``k`` from the
        direction lying in the ``(k, B0)`` plane. Therefore ``phi`` specifies
        the polarization direction.

        Define
        ``e1 = (cos(theta), 0, -sin(theta))``
        as the direction in the ``(k, B0)`` plane perpendicular to ``k``,
        and

        ``e2 = (0, 1, 0)``
        as the direction perpendicular to the ``(k, B0)`` plane.

        Then the linear-polarization direction is

        ``n_1 = cos(phi) * e1 + sin(phi) * e2``

        or equivalently

        ``n_1 = (
            cos(theta) * cos(phi),
            sin(phi),
            -sin(theta) * cos(phi)
        )``.

        The same definition of ``phi`` is used for powder and single-crystal
        samples.

        The angle ``alpha`` between B1 and B0 is consequently determined by
        ``cos(alpha) = n_1 . n_0
                     = -sin(theta) * cos(phi)``,
        so that
        ``alpha = acos(-sin(theta) * cos(phi))``.


        Examples:

        - ``theta = pi / 2, phi = 0``:
          Voigt geometry. B1 is parallel to B0 up to its sign.

        - ``theta = pi / 2, phi = pi / 2``:
          Voigt geometry. B1 is perpendicular to B0.

        - ``theta = 0``:
          Faraday geometry. B1 is perpendicular to B0 for every ``phi``.
          Changing ``phi`` only rotates the linear polarization in the
          xy-plane.

        - ``theta = pi / 4, phi = 0``:
          B1 lies in the ``(k, B0)`` plane and has equal-magnitude transverse
          and longitudinal components relative to B0.

        ``phi`` is used only for linear polarization. For circular and
        unpolarized radiation it does not affect the calculation.

        Default is ``pi / 2``.

    :param terms_computer:
        Optional custom polarization-term computer.

        If ``None``, ``WaveIntensityCalculator`` creates:

        - ``PowderPlaneWaveTerms`` for a powder/disordered sample;
        - ``CrystalPlaneWaveTerms`` for a single-crystal sample.

        For calculations, a custom object must be callable with one
        optional tensor argument and return ``(w_xy, w_z, w_mixed)``.

        When a custom ``terms_computer`` is supplied, it is used as-is.
        The calculator does not overwrite its internal geometry or
        polarization.

        Default is ``None``.
    """

    polarization: tp.Union[Polarization, str] = Polarization.LINEAR
    theta: float = 0.0
    phi: tp.Optional[float] = math.pi / 2
    terms_computer: tp.Optional[WaveTermsComputer] = None

    def normalized_polarization(self) -> Polarization:
        """Return ``polarization`` as a normalized ``Polarization`` enum."""
        return Polarization.from_value(self.polarization)

    def helicity(self) -> tp.Optional[int]:
        """
        Return photon helicity for circular polarization.

        :return:
            ``+1`` for positive circular polarization, ``-1`` for negative
            circular polarization, and ``None`` for linear or unpolarized
            radiation.
        """
        polarization = self.normalized_polarization()
        if polarization == Polarization.CIRCULAR_PLUS:
            return 1
        if polarization == Polarization.CIRCULAR_MINUS:
            return -1
        return None

    def compute_wave_geometry(
        self,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor, tp.Optional[int]]:
        """
        Compute the incident-wave geometry relative to the static field B0.

        The coordinate system is defined with B0 along the z-axis and ``k``
        in the xz-plane.

        :param device:
            Device on which the geometry vectors are created.
        :param dtype:
            Floating-point dtype of the geometry vectors.
        :return:
            Tuple ``(n_k, n_1, helicity)`` where:

            - ``n_k`` is the unit vector along the wave propagation direction
              ``k``:

              ``n_k = (sin(theta), 0, cos(theta))``.

            - ``n_1`` is the unit vector along the oscillating magnetic field
              B1 for linear polarization:

              ``n_1 = (
                  cos(theta) * cos(phi),
                  sin(phi),
                  -sin(theta) * cos(phi)
              )``.

              B1 is perpendicular to ``n_k`` by construction. For circular
              and unpolarized radiation, ``n_1`` is not used by the intensity
              formula, but the same geometrical reference vector is returned.

            - ``helicity`` is ``+1`` for positive circular polarization,
              ``-1`` for negative circular polarization, and ``None`` for
              linear or unpolarized radiation.
        """
        theta = torch.as_tensor(self.theta, device=device, dtype=dtype)
        phi_value = math.pi / 2 if self.phi is None else self.phi
        phi = torch.as_tensor(phi_value, device=device, dtype=dtype)

        sin_t = torch.sin(theta)
        cos_t = torch.cos(theta)
        sin_p = torch.sin(phi)
        cos_p = torch.cos(phi)

        n_k = torch.stack((sin_t, torch.zeros_like(sin_t), cos_t))
        n_1 = torch.stack((
            cos_t * cos_p,
            sin_p,
            -sin_t * cos_p,
        ))
        return n_k, n_1, self.helicity()

    def compute_b1_b0_angle(self) -> float:
        """
        Compute the angle between B1 and B0 for linear polarization.

        For

        ``n_1 = (
            cos(theta) * cos(phi),
            sin(phi),
            -sin(theta) * cos(phi)
        )``

        and ``n_0 = (0, 0, 1)``, the B0 direction,

        ``cos(alpha) = n_1 . n_0 = -sin(theta) * cos(phi)``.

        Therefore,

        ``alpha = acos(-sin(theta) * cos(phi))``.

        :return:
            Angle ``alpha`` between B1 and B0 in radians, in ``[0, pi]``.
        :raises ValueError:
            If the configured polarization is not linear or ``phi`` is None.
        """
        if self.normalized_polarization() != Polarization.LINEAR:
            raise ValueError(
                "The B1-B0 angle is defined here only for linear polarization."
            )
        if self.phi is None:
            raise ValueError("phi must be specified for linear polarization.")

        cosine = -math.sin(self.theta) * math.cos(self.phi)
        cosine = max(-1.0, min(1.0, cosine))
        return math.acos(cosine)


MagnetizationConfig = tp.Union[
    ResonatorMagnetizationConfig,
    WaveMagnetizationConfig,
]