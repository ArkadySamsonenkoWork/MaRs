import torch
import typing as tp
import copy

from .. import tr_utils
from .. import matrix_generators
from .. import contexts
from . import core


class LevelBasedPopulator(core.BaseTimeDepPopulator):
    """LevelBasedPopulator implements time-resolved EPR signal modeling within
    the kinetic.

    (population-based) relaxation paradigm.

    This class uses only the diagonal elements of the density matrix (populations of energy levels)
    Transitions between levels are governed by a kinetic matrix K derived from the provided Context, which may include:
      - Spontaneous (thermal) transitions respecting detailed balance,
      - Driven (induced) transitions,
      - Loss terms (e.g., phosphorescence decay from triplet states).

    The initial populations are either computed from thermal equilibrium at `init_temperature`
    or taken from the Context, if explicitly defined.
    """
    def __init__(self,
                 context: tp.Optional[contexts.BaseContext],
                 tr_matrix_generator_cls: tp.Type[matrix_generators.BaseGenerator]
                 = matrix_generators.LevelBasedGenerator,
                 solver: tp.Optional[tr_utils.EvolutionSolver] = None,
                 init_temperature: tp.Union[float, torch.Tensor] = 293.0,
                 energy_shifts: tp.Optional[tp.Union[torch.Tensor, tp.List]] = None,
                 difference_out: bool = False,
                 device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32):
        """
        :param context: context is a dataclass / Dict with any objects that are used to compute relaxation matrix.

        :param tr_matrix_generator_cls: class of Matrix Generator
            that will be used to compute rates of transitions
        :param solver: It solves the general equation dn/dt = A(n,t) @ n.

            The following solvers are available:
            - odeint_solver:  Default solver.
            It uses automatic control of time-steps. If you are not sure about the correct time-steps use it
            - stationary_rate_solver. When A does not depend on time use it.
            It just uses that in this case n(t) = exp(At) @ n0
            - exponential_solver. When A does depend on time but does not depend on n,
            It is possible to precompute A and exp(A) in all points.
            In this case the solution is n_i+1 = exp(A_idt) @ ni

            If solver is None than it will be initialized as odeint solver or stationary solver according to the context

        :param init_temperature: initial temperature. In default case it is used to find initial population

        :param energy_shifts: The additional energy shift added to the spin energies. For example, the factor TS

        :param difference_out: If True, the output intensity is expressed as the difference relative
               to the initial signal:
                       intensity(t) = intensity(t) - intensity(t=0).
                       This is useful for simulating differential or transient absorption spectra.

        :param device: device to compute (cpu / gpu)
        """
        super().__init__(
            context, tr_matrix_generator_cls, solver, init_temperature, energy_shifts, difference_out, device, dtype
        )

    def init_solver(self, solver: tp.Optional[tp.Callable]) -> tp.Callable:
        if solver is not None:
            return solver
        if self.time_dependant:
            return tr_utils.EvolutionPopulationSolver.odeint_solver
        else:
            return tr_utils.EvolutionPopulationSolver.stationary_rate_solver

    def _init_tr_matrix_generator(self,
                                  time: tp.Optional[torch.Tensor],
                                  res_fields: torch.Tensor,
                                  lvl_down: tp.Optional[torch.Tensor],
                                  lvl_up: tp.Optional[torch.Tensor],
                                  energies: torch.Tensor,
                                  vector_down: tp.Optional[torch.Tensor],
                                  vector_up: tp.Optional[torch.Tensor],
                                  full_system_vectors: tp.Optional[torch.Tensor],
                                  *args, **kwargs) -> matrix_generators.BaseGenerator:
        """
        Function creates TransitionMatrixGenerator - it is object that can compute rates of transitions.
        ----------

        :param time:
            Time points of measurements.

        :param res_fields:
            Resonance fields of transitions.
            Shape: [..., M], where M is the number of resonance energies.

        :param lvl_down:
            Energy levels of lower states from which transitions occur.
            Shape: [time, ..., N], where time is the time dimension and
            N is the number of energy levels.

        :param lvl_up:
            Energy levels of upper states to which transitions occur.
            Shape: [time, ..., N], where time is the time dimension and
            N is the number of energy levels.

        :param energies:
            The energies of spin states. The shape is [..., M, N]

        :param vector_down:
            Eigenvectors of the lower energy states. The shape is [...., M, N],
            where M is number of transitions, N is number of levels

        :param vector_up:
            Eigenvectors of the upper energy states.The shape is [...., M, N],
            where M is number of transitions, N is number of levels

        :param full_system_vectors:
            Eigenvectors of the full set of energy levels. The shape os [...., M, N, N],
            where M is number of transitions, N is number of levels
            For some cases it can be None. The parameter of the creator 'output_eigenvector- == True'
            make the creator to compute these vectors.
            The default behavior, whether to calculate vectors or not,
            depends on the specific Spectra Manager and its settings.

        :param args: tuple, optional.
        :param kwargs : dict, optional

        :param return:
        -------
        TransitionMatrixGenerator instance
        """
        return self.tr_matrix_generator_cls(
            context=self.context,
            init_temperature=self.init_temperature,
            res_fields=res_fields,
            full_system_vectors=full_system_vectors,
            energies=energies
        )

    def forward(self,
                time: torch.Tensor, res_fields: torch.Tensor,
                lvl_down: torch.Tensor, lvl_up: torch.Tensor,
                energies: torch.Tensor, vector_down: torch.Tensor,
                vector_up: torch.Tensor,
                full_system_vectors: tp.Optional[torch.Tensor],
                *args, **kwargs) -> torch.Tensor:
        """
        :param time:
            Time points of measurements. The shape is [T], where T is number of time-steps

        :param res_fields:
            Resonance fields of transitions.
            Shape: [..., M], where M is the number of resonance energies.

        :param lvl_down:
            Energy levels of lower states from which transitions occur.
            Shape: [time, ..., N], where time is the time dimension and
            N is the number of energy levels.

        :param lvl_up:
            Energy levels of upper states to which transitions occur.
            Shape: [time, ..., N], where time is the time dimension and
            N is the number of energy levels.

        :param energies:
            The energies of spin states. The shape is [..., N]

        :param vector_down:
            Eigenvectors of the lower energy states. The shape is [...., M, N],
            where M is number of transitions, N is number of levels

        :param vector_up:
            Eigenvectors of the upper energy states.The shape is [...., M, N],
            where M is number of transitions, N is number of levels

        :param full_system_vectors:
            Eigenvectors of the full set of energy levels. The shape os [...., M, N, N],
            where M is number of transitions, N is number of levels
            For some cases it can be None. The parameter of the creator 'output_eigenvector- == True'
            make the creator to compute these vectors
            The default behavior, whether to calculate vectors or not,
            depends on the specific Spectra Manager and its settings.

        :param args: additinal args from spectra creator.

        :param kwargs:
        :return: Part of the transition intensity that depends on the population of the levels.
        The shape is [T, ...., Tr]
        """
        initial_populations = self._initial_populations(energies, lvl_down, lvl_up, full_system_vectors)
        tr_matrix_generator = self._init_tr_matrix_generator(time, res_fields,
                                                             lvl_down, lvl_up, energies, vector_down,
                                                             vector_up, full_system_vectors, *args, **kwargs)
        evo = tr_utils.EvolutionMatrix(energies)
        if initial_populations.dim() == 1:
            initial_populations = initial_populations.unsqueeze(-2)
        pop_difference = self.solver(
            time, initial_populations, evo, tr_matrix_generator, lvl_down, lvl_up)
        pop_difference = self._post_compute(pop_difference)
        return pop_difference


class T1Populator(LevelBasedPopulator):
    """T1Populator models time-resolved EPR signals initiated from an inverted
    population distribution,.

    mimicking inversion-recovery relaxation experiments.

    Unlike the standard LevelBasedPopulator, which starts from thermal
    equilibrium, T1Populator swaps the populations of the resonant upper
    and lower levels at t = 0. This simulates an idealized π-pulse
    excitation that fully inverts the transition.

    This class is particularly useful for modeling relaxation dynamics
    after photoexcitation or microwave inversion in triplet or radical-
    pair systems, where the initial non-equilibrium state drives the
    subsequent relaxation.
    """
    def _initial_populations(
            self, energies: torch.Tensor, lvl_down: torch.Tensor, lvl_up: torch.Tensor,
            full_system_vectors: tp.Optional[torch.Tensor],
            *args, **kwargs
    ):
        """Computes the initial population vector with inverted populations for
        resonant transitions.

        This method first obtains the thermal equilibrium populations (or context-defined ones),
        then swaps the populations of the lower and upper states for each resonant transition
        to simulate full inversion (π-pulse condition).

        :param energies:
            The energies of spin states. The shape is [..., R, N], where R is number of resonance transitions

        :param lvl_down : array-like
            Indexes of energy levels of lower states from which transitions occur.
            Shape: [R], where R is number of resonance transitions
            N is the number of energy levels.

        :param lvl_up : array-like
            Indexes of energy levels of upper states to which transitions occur.
            Shape: [R], where R is number of resonance transitions

        :param full_system_vectors: Eigen vector of each level of a spin system. The shape os [..., N, N].
        For some cases it can be None

        :param args:
        :param kwargs:
        :return: initial populations
        """
        population = self._getter_init_population(energies, lvl_down, lvl_up, full_system_vectors)
        out_populations = copy.deepcopy(population)
        indexes = torch.arange(energies.shape[-2], device=energies.device)
        out_populations[..., indexes, lvl_down] = population[..., indexes, lvl_up]
        out_populations[..., indexes, lvl_up] = population[..., indexes, lvl_down]
        return out_populations
