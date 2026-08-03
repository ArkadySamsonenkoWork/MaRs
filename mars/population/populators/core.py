from abc import ABC, abstractmethod

import warnings

import torch
from torch import nn

from ... import constants
import typing as tp

from .. import tr_utils
from .. import matrix_generators
from .. import contexts


def transform_to_complex(vector):
    if vector.dtype == torch.float32:
        return vector.to(torch.complex64)
    elif vector.dtype == torch.float64:
        return vector.to(torch.complex128)
    else:
        return vector


class BasePopulator(nn.Module):
    """Base class for populators.

    A populator is responsible for computing the part of the EPR transition intensity
    that depends on the populations of energy levels (or the full density matrix in more advanced cases).
    This includes:
      - Thermal equilibrium populations (Boltzmann distribution),
      - Context-defined initial populations,
      - Population differences between resonant upper and lower states.

    This class supports both stationary and time-dependent scenarios through inheritance.
    It handles initialization from temperature or from a relaxation Context,
    and provides unified access to population initialization logic.

    The actual intensity computation (including matrix elements, line shapes, etc.)
    is performed downstream in the spectra creator; the populator only supplies
    the population-dependent factor.
    """
    def __init__(self,
                 context: tp.Optional[contexts.BaseContext] = None,
                 init_temperature: tp.Union[float, torch.Tensor] = 293.0,
                 energy_shifts: tp.Optional[tp.Union[torch.Tensor, tp.List]] = None,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self.register_buffer(
            "init_temperature", torch.tensor(init_temperature, device=device, dtype=dtype)
        )
        if isinstance(energy_shifts, torch.Tensor):
            self.register_buffer(
                "energy_shifts", energy_shifts.to(device=device, dtype=dtype)
            )
        elif isinstance(energy_shifts, list):
            self.register_buffer(
                "energy_shifts", torch.tensor(energy_shifts, device=device, dtype=dtype)
            )
        elif energy_shifts is None:
            self.register_buffer(
                "energy_shifts", energy_shifts
            )
        else:
            raise TypeError("energy_shift should be None or tensor")
        self._context = None
        self.set_context(context)

    def _precompute(
            self,
            res_fields: torch.Tensor, lvl_down: torch.Tensor, lvl_up: torch.Tensor,
            energies: torch.Tensor, vector_down: torch.Tensor,
            vector_up: torch.Tensor, *args, **kwargs) ->\
            tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return res_fields, lvl_down, lvl_up, energies, vector_down, vector_up

    def _init_context_meta(self):
        """Initialize data depending on context.

        :return: None
        """
        if self.context is not None:
            if self.context.contexted_init_population:
                self.contexted = True
                self._getter_init_population = self._context_dependant_init_population
            else:
                self.contexted = False
                self._getter_init_population = self._temp_dependant_init_population
            self.time_dependant = self.context.time_dependant

        else:
            self.contexted = False
            self._getter_init_population = self._temp_dependant_init_population
            self.time_dependant = False

    @property
    def context(self) -> tp.Optional[contexts.BaseContext]:
        return self._context

    def set_context(self, context: tp.Optional[contexts.BaseContext]) -> None:
        """
        :param context: Relaxtion and Polarization context
        :return:
        """
        if context is not None and not isinstance(context, nn.Module):
            raise TypeError(f"context must be an nn.Module or None, got {type(context)}")

        if self._context is not None:
            del self._modules["_context"]

        self._context = context
        if context is not None:
            if context.is_opened():
                warnings.warn(
                    "Changing the sample without creating a new context may invalidate "
                    "the cached basis transformation data. Consider calling close_context() first."
                    "Or create a new context",
                    UserWarning,
                    stacklevel=2
                )
            self.add_module("_context", context)

        self._init_context_meta()

    def _initial_populations(
            self, energies: torch.Tensor, lvl_down: torch.Tensor, lvl_up: torch.Tensor,
            full_system_vectors: tp.Optional[torch.Tensor],
            *args, **kwargs
    ) -> torch.Tensor:
        """
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
        :return: initial populations defined from thermal equilibrium
        """
        return self._getter_init_population(energies, lvl_down, lvl_up, full_system_vectors)

    def _temp_dependant_init_population(self,
                energies: torch.Tensor,
                lvl_down: torch.Tensor,
                lvl_up: torch.Tensor,
                full_system_vectors: tp.Optional[torch.Tensor],
                *args, **kwargs):
        """
        Returns the populations defined as stationary population at temperature 'init_temperature'.
        """
        return nn.functional.softmax(
            -constants.unit_converter(energies, "Hz_to_K") / self.init_temperature, dim=-1
        )

    def _temp_dependant_init_density(self,
                energies: torch.Tensor,
                lvl_down: torch.Tensor,
                lvl_up: torch.Tensor,
                full_system_vectors: tp.Optional[torch.Tensor],
                *args, **kwargs):
        """Nitializes the density matrix from thermal equilibrium at
        `self.init_temperature`.

        Populations follow the Boltzmann distribution: p_i ∝ exp(−E_i / k_B T),
        where energies are converted from Hz to Kelvin using physical constants.
        The resulting density matrix is diagonal in the Hamiltonian eigenbasis.
        :return:
            Diagonal complex-valued density matrix, shape [..., N, N].
        """
        populations = torch.nn.functional.softmax(
            -constants.unit_converter(energies, "Hz_to_K") / self.init_temperature, dim=-1
        )
        return transform_to_complex(torch.diag_embed(populations, dim1=-1, dim2=-2))

    def _context_dependant_init_density(self,
                energies: torch.Tensor,
                lvl_down: torch.Tensor,
                lvl_up: torch.Tensor,
                full_system_vectors: tp.Optional[torch.Tensor],
                *args, **kwargs):

        """Initializes the density matrix from the Context, which may define it
        in an arbitrary basis.

        The Context returns a density matrix or population vector in its native basis
        (e.g., zero-field splitting basis for triplet states).
        This method uses `full_system_vectors` to transform it into the field-dependent eigenbasis.

        :return:
            Transformed initial density matrix in the eigenbasis of the full Hamiltonian.
        """
        return self.context.get_transformed_init_density(full_system_vectors)

    def _context_dependant_init_population(self,
                energies: torch.Tensor,
                lvl_down: torch.Tensor,
                lvl_up: torch.Tensor,
                full_system_vectors: tp.Optional[torch.Tensor],
                *args, **kwargs):
        """
        Returns the populations defined as polarized and given by the
        """
        return self.context.get_transformed_init_populations(full_system_vectors, normalize=False)

    def _out_population_difference(self, populations: torch.Tensor, lvl_down: torch.Tensor, lvl_up: torch.Tensor):
        """Calculate the population difference between transitioning energy
        levels.

        Parameters
        ----------
        :param populations:
             population values.
            Shape: [..., R, N] or [N], where N is the number of energy levels. R is number of resonance transitions

        :param lvl_down : array-like
            Indexes of energy levels of lower states from which transitions occur.
            Shape: [R], where R is number of resonance transitions
            N is the number of energy levels.

        :param lvl_up : array-like
            Indexes of energy levels of upper states to which transitions occur.
            Shape: [R], where R is number of resonance transitions

        :return:
        -------
            The population difference between transitioning energy levels.
        """
        if populations.dim() == 1:
            populations = populations.unsqueeze(-2)
        indexes = torch.arange(populations.shape[-2], device=populations.device)
        return populations[..., indexes, lvl_down] - populations[..., indexes, lvl_up]


class BaseTimeDepPopulator(BasePopulator):
    """Base class for time-dependent populators that model relaxation dynamics
    in time-resolved EPR.

    This class implements the common infrastructure for solving the kinetic or Liouville-von Neumann
    equations that govern the evolution of populations or the density matrix:
      dn/dt = K(t, n) · n        (population-based)
      dρ/dt = -i[H, ρ] + R[ρ]    (density matrix-based)

    Key components:
      1. **Populator**: Defines initial state and numerical strategy (this class and subclasses).
      2. **Context**: Encodes physical relaxation mechanisms (losses, spontaneous/induced transitions,
         dephasing) and their basis of definition.
      3. **Transition matrix generator**: Constructs the relaxation operator (K or R) from Context.
      4. **Solver**: Integrates the evolution equation (stationary, quasi-stationary, or adaptive ODE).

    Subclasses must implement:
      - `init_solver`: selects appropriate integrator based on time-dependence,
      - `_init_tr_matrix_generator`: builds generator for relaxation superoperator,
      - `forward`: orchestrates the full computation pipeline.
    """
    def __init__(self,
                 context: tp.Optional[contexts.BaseContext],
                 tr_matrix_generator_cls: tp.Type[matrix_generators.BaseGenerator],
                 solver: tp.Optional[tr_utils.EvolutionSolver] = None,
                 init_temperature: tp.Union[float, torch.Tensor] = 293.0,
                 energy_shifts: tp.Optional[tp.Union[torch.Tensor, tp.List]] = None,
                 difference_out: bool = False,
                 device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32):
        """
        :param context: context is a dataclass / Dict with any objects that are used to compute relaxation matrix.

        :param tr_matrix_generator_cls: class of Matrix Generator
            that will be used to compute probabilities of transitions
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
        super().__init__(context, init_temperature, energy_shifts, device, dtype)
        self.solver = self.init_solver(solver)
        self.tr_matrix_generator_cls = tr_matrix_generator_cls
        self.difference_out = difference_out
        self.to(device)


    @abstractmethod
    def init_solver(self, solver: tp.Optional[tp.Callable]) -> tp.Callable:
        if solver is not None:
            return solver
        if self.time_dependant:
            return tr_utils.EvolutionSolver.odeint_solver
        else:
            return tr_utils.EvolutionSolver.stationary_rate_solver

    def _post_compute(self, time_intensities: torch.Tensor, *args, **kwargs):
        """
        :param time_intensities: The population difference between transitioning energy levels depending on time.

        :return: intensity of transitions due to population difference
        """
        self.context.close_context()
        if self.difference_out:
            return time_intensities - time_intensities[0].unsqueeze(0)
        else:
            return time_intensities

    @abstractmethod
    def _init_tr_matrix_generator(self,
                                  *args, **kwargs) -> matrix_generators.BaseGenerator:
        """
        Function creates TransitionMatrixGenerator - it is object that can compute rates of transitions.

        :param args: tuple, optional.
        :param kwargs : dict, optional
        :param return:
        -------
        TransitionMatrixGenerator instance
        """
        tr_matrix_generator = self.tr_matrix_generator_cls(*args, **kwargs)
        return tr_matrix_generator

    def compute_stationary_polarization(self,
                res_fields: torch.Tensor,
                lvl_down: torch.Tensor,
                lvl_up: torch.Tensor,
                energies: torch.Tensor,
                vector_down: torch.Tensor, vector_up: torch.Tensor,
                full_system_vectors: tp.Optional[torch.Tensor],
                *args, **kwargs) -> torch.Tensor:
        """Computes the population difference for each resonant EPR transition. at zero time moment

        :param res_fields:
            Resonance magnetic field for each transition, shape [..., M],
            where M is the number of resonance conditions, (e.g. the number of resonance for each orientation)

        :param lvl_down:
            Indices of lower energy levels involved in transitions, shape [M].

        :param lvl_up:
            Indices of upper energy levels involved in transitions, shape [M].

        :param energies:
            Eigenenergies of all spin states in Hz, shape [..., M, N],
            where M is the number of resonance conditions, (e.g. the number of resonance for each orientation)
            and N is the number of energy levels.

        :param vector_down: Eigenvectors of the lower energy states, shape [..., M, N].
        :param vector_up: Eigenvectors of the upper energy states, shape [..., M, N].

        :param full_system_vectors:
            Eigenvectors of the full spin Hamiltonian, shape [..., N, N].
            Required only if initial populations are defined in a non-eigenbasis (e.g., ZFS basis)
            and Context provides them. Used to transform populations into the field-dependent eigenbasis.

        :return:
            Population differences Δp = p_upper − p_lower for each transition,
            shape [..., R], ready to be multiplied by transition matrix elements.
        """
        populations = self._initial_populations(energies, lvl_down, lvl_up, full_system_vectors)
        if self.context is not None:
            self.context.close_context()
        return self._out_population_difference(populations, lvl_down, lvl_up)
