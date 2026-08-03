import math
from abc import ABC, abstractmethod
import torch
import typing as tp

from . import contexts
from .. import constants


class BaseGenerator(ABC):
    """Abstract base class for time-dependent generators of relaxation-related
    computations.

    Encapsulates a context that defines initial states, transition
    rates, and basis transformations, along with temperature and
    eigenvector data required to evaluate relaxation operators at
    arbitrary times.
    """
    def __init__(self,
                 context: contexts.BaseContext,
                 init_temperature: torch.Tensor,
                 res_fields: tp.Optional[torch.Tensor],
                 energies: tp.Optional[torch.Tensor],
                 full_system_vectors: tp.Optional[torch.Tensor],
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        """
        :param context: Context object instance.

        :param init_temperature:  initial temperature of process.
        -It can be constant during the process
        -It can be skipped if the temperature defines from profile
        :param res_fields:
            Resonance fields of transitions.
            Shape: [..., M], where M is the number of resonance energies.
        :param energies:
            Eigenenergies of the spin system, shape [..., M, N], where M is the number of resonance energies,
            and N is the number of energy levels.
        :param full_system_vectors:
            Eigenvectors of the full set of energy levels. The shape os [...., M, N, N],
            where M is number of transitions, N is number of levels
            For some cases it can be None. The parameter of the creator 'output_eigenvector- == True'
            make the creator to compute these vectors.
            The default behavior, whether to calculate vectors or not,
            depends on the specific Spectra Manager and its settings.

        :param device: Computation device
        :param dtype:
        :param args:
        :param kwargs:
        """
        super().__init__()
        self.context = context
        self.init_temperature = init_temperature
        self.full_system_vectors = full_system_vectors
        self.energies = energies
        self.res_fields = res_fields

    @abstractmethod
    def __call__(self, time: torch.Tensor):
        pass


class LevelBasedGenerator(BaseGenerator):
    """Abstract base class for generating transition rates matrices in a
    multi-level.

    system with populations and energy differences.
    The system of rate equations for two levels with populations n1, n2 and energies E1, E2 is:
        dn1/dt = -out_1 - k1 * n1 + k2 * n2
        dn2/dt = -out_2 + k1 * n1 - k2 * n2

    which can be written in matrix form:

        dN/dt = -OUT + K @ N
    where:
      - OUT is a vector of outgoing transitions from the system,
      - K is the relaxation matrix:
            K = [[-k1,  k2],
                 [ k1, -k2]]

    K itself can be rewritten via K' and driven transition DR
    K = K' + DR, where
        K'   – equilibrium relaxation (thermal),
        DR  – driven_rates transitions,

    At thermal equilibrium, transition rates satisfy detailed balance:
        k'1 / k'2 = n'2 / n'1 = exp(-(E2 - E1) / kT)
    Defining the average relaxation rate:

        k' = (k'1 + k'2) / 2

    we can compute:
        k'2 = 2k' / (1 + exp(-(E2 - E1) / kT))
        k'1 = 2k' * exp(-(E2 - E1) / kT) / (1 + exp(-(E2 - E1) / kT))

    In symmetric form, the "theraml rates" matrix (i.e. mean equilibrium transition rates) is:

        thermal_rates = [[0,  k'],
                      [k', 0]]

    DR matrix is matrix which rates are not connected by thermal equilibrium:
                     [[0,  dr_1],
                     [dr_2, 0]]
    """

    def __call__(self, time: torch.Tensor) -> tp.Tuple[
        tp.Optional[torch.Tensor],
        tp.Optional[torch.Tensor],
        tp.Optional[torch.Tensor],
        tp.Optional[torch.Tensor]
    ]:
        """Evaluate transition rates at given measurement times.

        Parameters
        :param time: torch.Tensor
        :return: tuple
            (temperature, thermal_rates, driven_rates, decay_rates)
            - temperature : torch.Tensor or None
                System temperature(s) at the given time(s).
            - thermal_rates : torch.Tensor [..., N, N]
                Thermal equilibrium (Boltzmann-weighted) transition rates.

            Example in symmetry form:
                thermal_rates = [[0,  k'],
                            [k', 0]]

            - driven_rates : torch.Tensor [..., N, N] or None
                Probabilities of driven transitions (e.g. due to external driving).

                Ind matrix is always symmetry: [[0,  i],
                                                [i, 0]]

            - decay_rates : torch.Tensor [..., N]  or None
                Out-of-system transition rates (loss terms).
        """
        if self.context.time_dependant:
            time_dep_values = self.context.get_time_dependent_values(time)
        else:
            time_dep_values = None

        temperature = self._temperature(time_dep_values)
        thermal_rates = self._thermal_transition_rates(time_dep_values, temperature)
        driven_rates = self._driven_transition_rates(time_dep_values, temperature)
        decay_rates = self._decay_transition_rates(time_dep_values, temperature)
        return temperature, thermal_rates, driven_rates, decay_rates

    def _temperature(self, time_dep_values: torch.Tensor) -> tp.Optional[torch.Tensor]:
        """Return temperature(s) at times t."""
        return self.init_temperature.unsqueeze(-1).unsqueeze(-1)

    def _thermal_transition_rates(self, time_dep_values: tp.Optional[torch.Tensor], temperature: torch.Tensor) ->\
            tp.Optional[torch.Tensor]:
        """Retrieve spontaneous (thermal) transition rates transformed
        into the eigenbasis.

        These rates are subject to Boltzmann detailed balance.
        :param time_dep_values: Optional time-dependent scaling factors from the Context profile.
        :param temperature: Optional The system temperature in K. It can be used for redfield relaxation evaluation
        The shape is [] or [t], where t is number of time-steps
        :return: Tensor of shape [..., N, N] representing equilibrium transition rates.
        """
        return self.context.get_transformed_thermal_rates(
            self.full_system_vectors, time_dep_values, self.res_fields,
            self.energies, temperature
        )

    def _driven_transition_rates(self, time_dep_values: tp.Optional[torch.Tensor], temperature: torch.Tensor) ->\
            tp.Optional[torch.Tensor]:
        """Retrieve non-thermal (driven) transition rates in the
        eigenbasis.

        These rates are not modified by thermal constraints and represent external perturbations.
        :param time_dep_values: Optional time-dependent scaling factors from the Context profile.
        :param temperature: Optional The system temperature in K. It can be used for redfield relaxation evaluation
        The shape is [] or [t], where t is number of time-steps
        :return: Tensor of shape [..., N, N] or None if no driven processes are defined.
        """
        return self.context.get_transformed_driven_rates(
            self.full_system_vectors, time_dep_values, self.res_fields, self.energies, temperature
        )

    def _decay_transition_rates(self, time_dep_values: tp.Optional[torch.Tensor], temperature: torch.Tensor) ->\
            tp.Optional[torch.Tensor]:
        """Retrieve irreversible loss rates from each energy level in the
        eigenbasis.

        These represent population removal from the spin system (e.g., phosphorescence).
        :param time_dep_values: Optional time-dependent scaling factors from the Context profile.
        :param temperature: Optional The system temperature in K. It can be used for redfield relaxation evaluation
        The shape is [] or [t], where t is number of time-steps
        :return: Vector of shape [..., N] or None if no loss processes are defined.
        """
        return self.context.get_transformed_decay_rates(
            self.full_system_vectors, time_dep_values, self.res_fields, self.energies, temperature
        )


class TempDepGenerator(LevelBasedGenerator):
    """Extension of LevelBasedGenerator where system temperature is time-
    dependent.

    This assumes the profile(t) returns absolute temperature in Kelvin
    at time t. All thermal transition rates are recomputed at each time
    step using the instantaneous temperature.
    """
    def _temperature(self, time_dep_values: torch.Tensor) -> tp.Optional[torch.Tensor]:
        """Return time-dependent temperature from the Context profile.

        Assumes time_dep_values contains temperature values evaluated at 'time'.
        :param time_dep_values: Tensor of shape compatible with [..., 1, 1] containing temperatures.
        :return: Same as time_dep_values.
        """
        return time_dep_values.unsqueeze(-1).unsqueeze(-1)


class DensityRWAGenerator(BaseGenerator):
    """Generator for Liouville-space superoperators under the rotating wave
    approximation (RWA).

    Returns the Hamiltonian H and two superoperators:
      - thermal_superop: spontaneous processes (thermal relaxation, losses, dephsing),
      - driven_superop: external non-equilibrium driving.

    The full Liouvillian is L = -i[H, ·] + R_thermal + R_driven, where [·,·] is the commutator,
    and R terms are relaxation superoperators.
    H is defined in rotating frame and equel to H =

    Unlike population-based models, this class preserves quantum coherences and operates on vectorized
    density matrices of size N^2 x N^2.
    """
    def __init__(self,
                 context: contexts.BaseContext,
                 init_temperature: torch.Tensor,
                 res_fields: torch.Tensor,
                 energies: tp.Optional[torch.Tensor],
                 full_system_vectors: tp.Optional[torch.Tensor],
                 stationary_hamiltonian: torch.Tensor,
                 lvl_down: torch.Tensor, lvl_up: torch.Tensor,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        """
        :param context: Context object instance.

        :param init_temperature:  initial temperature of process.
        -It can be constant during the process
        -It can be skipped if the temperature defines from profile

        :param res_fields:
            Resonance fields of transitions.
            Shape: [..., M], where M is the number of resonance energies.

        :param energies:
            Eigenenergies of the spin system, shape [..., M, N], where M is number of transitions each batch point,
            and N is the number of energy levels.

        :param full_system_vectors:
            Eigenvectors of the full set of energy levels. The shape os [...., M, N, N],
            where M is number of transitions, N is number of levels
            For some cases it can be None. The parameter of the creator 'output_eigenvector- == True'
            make the creator to compute these vectors
            The default behavior, whether to calculate vectors or not,
            depends on the specific Spectra Manager and its settings.


        :param stationary_hamiltonian: The Hamiltonian in the given frame. The definition depends on approximations.
            -For RWA it uses full Hamiltonian in rotating frame.
            -For Propagator it uses real stationary Hamiltonian
            -The Hamiltonian is expressed in (Hz * 2π = rad/s).

        :param lvl_down:
            Energy levels of lower states from which transitions occur.
            Shape: [time, ..., N], where time is the time dimension and
            N is the number of energy levels.

        :param lvl_up:
            Energy levels of upper states to which transitions occur.
            Shape: [time, ..., N], where time is the time dimension and
            N is the number of energy levels.
        :param device: Computation device
        :param dtype:
        :param args:
        :param kwargs:
        """
        super().__init__(context, init_temperature, res_fields, energies, full_system_vectors, device, dtype)
        self.stationary_hamiltonian = stationary_hamiltonian
        self.level_down = lvl_down
        self.level_up = lvl_up

    def __call__(self, time: torch.Tensor) -> tp.Tuple[
        tp.Optional[torch.Tensor],
        torch.Tensor,
        tp.Optional[torch.Tensor],
        tp.Optional[torch.Tensor]
    ]:
        """Evaluate time-dependent relaxation superoperators for density-matrix
        evolution.

        :param time: Tensor of shape [T] containing measurement times.
        :return: A 4-tuple:
            - temperature: scalar tensor self.init_temperature (shape broadcastable to [..., 1, 1]).
            - stationary_hamiltonian: Hermitian operator in rotating frame H of shape [..., N, N].
            In dimensionless units, it is defined as:
                H = H0 + Sz + Ht(r), where H0 is stationary Hamiltonian, Sz is z-projection of total spin,
                Ht(r) is oscillating Hamiltonian in rotating frame (in this frame it doesn't depend on time)

            - thermal_superop: Liouville-space superoperator of shape [..., N^2, N^2] for thermal relaxation.
            - driven_superop: Liouville-space superoperator of shape [..., N^2, N^2] for non-thermal driving.
        """
        if self.context.time_dependant:
            time_dep_values = self.context.get_time_dependent_values(time)
        else:
            time_dep_values = None

        temperature = self._temperature(time_dep_values)
        thermal_superop = self._base_superop(time_dep_values, temperature)
        driven_superop = self._driven_superop(time_dep_values, temperature)
        return temperature, self.stationary_hamiltonian, thermal_superop, driven_superop

    def _temperature(self, time_dep_values: tp.Optional[torch.Tensor]) -> tp.Optional[torch.Tensor]:
        """Return temperature(s) at times t."""
        return self.init_temperature.unsqueeze(-1).unsqueeze(-1)

    def _base_superop(self, time_dep_values: tp.Optional[torch.Tensor], temperature) -> torch.Tensor:
        """Retrieve spontaneous relaxation superoperator in Liouville space.

        Includes spontaneous transitions, losses, and dephsing, and is corrected for detailed balance.
        :param time_dep_values: Optional time-dependent scaling from Context profile.
        :param temperature: Optional The system temperature in K. It can be used for redfield relaxation evaluation
        The shape is [] or [t], where t is number of time-steps
        :return: Superoperator tensor of shape [..., N^2, N^2].
        """
        return self.context.get_transformed_thermal_superop(
            self.full_system_vectors, time_dep_values, self.res_fields, self.energies, temperature)

    def _driven_superop(self, time_dep_values: tp.Optional[torch.Tensor], temperature) -> tp.Optional[torch.Tensor]:
        """Retrieve non-thermal relaxation superoperator in Liouville space.

        Represents external driving not constrained by thermal equilibrium.
        :param time_dep_values: Optional time-dependent scaling from Context profile.
        :param temperature: Optional The system temperature in K. It can be used for redfield relaxation evaluation
        The shape is [] or [t], where t is number of time-steps
        :return: Superoperator tensor of shape [..., N^2, N^2] or None.
        """
        return self.context.get_transformed_driven_superop(
            self.full_system_vectors, time_dep_values, self.res_fields, self.energies, temperature)


class TempDepDensityRWAGenerator(DensityRWAGenerator):
    """Extension of DensityRWAGenerator where system temperature is time-
    dependent.

    This assumes the profile(t) returns absolute temperature in Kelvin
    at time t. All thermal transition rates are recomputed at each time
    step using the instantaneous temperature.
    """
    def _temperature(self, time_dep_values: torch.Tensor) -> tp.Optional[torch.Tensor]:
        """Return time-dependent temperature from the Context profile.

        Assumes time_dep_values contains temperature values evaluated at 'time'.
        :param time_dep_values: Tensor of shape compatible with [..., 1, 1] containing temperatures.
        :return: Same as time_dep_values.
        """
        return time_dep_values.unsqueeze(-1).unsqueeze(-1)


class DensityPropagatorGenerator(DensityRWAGenerator):
    """Generator for full propagator-based density-matrix evolution without
    RWA.

    Assumes time-independent relaxation rates. Time dependence in the
    Context profile is explicitly disallowed.
    """
    def __call__(self, time: torch.Tensor) -> tp.Tuple[
        tp.Optional[torch.Tensor],
        torch.Tensor,
        tp.Optional[torch.Tensor],
        tp.Optional[torch.Tensor]
    ]:
        """Evaluate time-dependent relaxation superoperators for density-matrix
        evolution.

        :param time: Tensor of shape [T] containing measurement times.
        :return: A 4-tuple:
            - temperature: scalar tensor self.init_temperature (shape broadcastable to [..., 1, 1]).
            - stationary_hamiltonian: Hermitian stationary Hamiltonian H0
             of shape [..., N, N]. In dimensionless units,

            - thermal_superop: Liouville-space superoperator of shape [..., N^2, N^2] for thermal relaxation.
            - driven_superop: Liouville-space superoperator of shape [..., N^2, N^2] for non-thermal driving.
        """
        if self.context.time_dependant:
            raise NotImplementedError(
                "Propagator solution of evolution doesn't support time dependant relaxation rates"
            )
        time_dep_values = None
        temperature = self._temperature(time_dep_values)
        thermal_superop = self._base_superop(time_dep_values, temperature)
        driven_superop = self._driven_superop(time_dep_values, temperature)
        return temperature, self.stationary_hamiltonian, thermal_superop, driven_superop
