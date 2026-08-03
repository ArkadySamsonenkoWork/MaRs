import typing as tp
from enum import Enum

from scipy.optimize import linear_sum_assignment
import torch
import numpy as np
import matplotlib.pyplot as plt
from ..spin_model import MultiOrientedSample
from .. import population
from .. import constants
from .. import spectra_manager


class T1AnalysisMethod(Enum):
    PAIRWISE = "pairwise"


class RalaxatioParmaetersPopulator(population.BaseTimeDepPopulator):
    def __init__(self,
                 context: tp.Optional[population.contexts.BaseContext],
                 levels_pair: tp.Tuple[int, int],
                 comp_method: T1AnalysisMethod,
                 tr_matrix_generator_cls: tp.Type[population.matrix_generators.BaseGenerator]
                 = population.matrix_generators.LevelBasedGenerator,
                 solver: tp.Optional[population.tr_utils.EvolutionSolver] = None,
                 init_temperature: tp.Union[float, torch.Tensor] = 293.0,
                 difference_out: bool = False,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32):
        """
        :param context: context is a dataclass / Dict with any objects that are used to compute relaxation matrix.
        :param levels_pair: A pair of levels between which the relaxation rate is estimated

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
        :param difference_out: If True, the output intensity is expressed as the difference relative
               to the initial signal:
                       intensity(t) = intensity(t) - intensity(t=0).
                       This is useful for simulating differential or transient absorption spectra.

        :param device: device to compute (cpu / gpu)
        """
        super().__init__(context, tr_matrix_generator_cls, solver, init_temperature, None, difference_out, device, dtype)
        self.levels_pair = levels_pair
        self.comp_method = comp_method

    def init_solver(self, solver: tp.Optional[tp.Callable]) -> tp.Callable:
        pass

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

    def _init_tr_matrix_generator(self,
                                  time: tp.Optional[torch.Tensor],
                                  res_fields: torch.Tensor,
                                  lvl_down: tp.Optional[torch.Tensor],
                                  lvl_up: tp.Optional[torch.Tensor],
                                  energies: torch.Tensor,
                                  vector_down: tp.Optional[torch.Tensor],
                                  vector_up: tp.Optional[torch.Tensor],
                                  full_system_vectors: tp.Optional[torch.Tensor],
                                  *args, **kwargs) -> population.matrix_generators.BaseGenerator:
        """
        Function creates TransitionMatrixGenerator - it is object that can compute probabilities of transitions.
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
                res_fields: torch.Tensor,
                energies: torch.Tensor,
                lvl_down: torch.Tensor,
                lvl_up: torch.Tensor,
                full_system_vectors: tp.Optional[torch.Tensor],
                *args, **kwargs) -> torch.Tensor:
        """Computes the population difference for each resonant EPR transition.

        :param res_fields:
            Resonance magnetic field for each transition, shape [..., M],
            where M is the number of resonance conditions, (e.g. the number of resonance for each orientation)

        :param energies:
            Eigenenergies of all spin states in Hz, shape [..., M, N],
            where M is the number of resonance conditions, (e.g. the number of resonance for each orientation)
            and N is the number of energy levels.

        :param lvl_down:
            Indices of lower energy levels involved in transitions, shape [M].

        :param lvl_up:
            Indices of upper energy levels involved in transitions, shape [M].

        :param full_system_vectors:
            Eigenvectors of the full spin Hamiltonian, shape [..., N, N].
            Required only if initial populations are defined in a non-eigenbasis (e.g., ZFS basis)
            and Context provides them. Used to transform populations into the field-dependent eigenbasis.

        :return:
            Population differences Δp = p_upper − p_lower for each transition,
            shape [..., R], ready to be multiplied by transition matrix elements.
        """
        tr_matrix_generator = self._init_tr_matrix_generator(None, res_fields,
                                                             lvl_down, lvl_up, energies, None,
                                                             None, full_system_vectors, *args, **kwargs)
        evo = population.tr_utils.EvolutionMatrix(energies)
        kinetic_matrix = evo(*tr_matrix_generator(torch.tensor(0.0, device=energies.device, dtype=energies.dtype)))
        levels_pair = self.levels_pair
        if self.comp_method == T1AnalysisMethod.PAIRWISE:
            W =\
                kinetic_matrix[..., levels_pair[0], levels_pair[1]] + kinetic_matrix[..., levels_pair[1], levels_pair[0]]
        else:
            raise NotImplementedError(f"comp_method should be {T1AnalysisMethod.PAIRWISE}")
        return W


def relaxation_field_dep(context: population.BaseContext,
                         sample: MultiOrientedSample,
                         freq: float,
                         temperature: float,
                         fields: tp.Union[tp.Tuple[float, float], torch.Tensor],
                         levels_pair: tp.Tuple[int, int],
                         method: tp.Union[str, T1AnalysisMethod] = T1AnalysisMethod.PAIRWISE):
    if isinstance(method, str):
        comp_method = T1AnalysisMethod(method)
    else:
        comp_method = method

    device = sample.device
    dtype = sample.dtype
    spin_system_dim = sample.base_spin_system.spin_system_dim
    batch_dims = sample.config_shape[:-1]
    mesh = sample.mesh
    mesh_size = mesh.initial_size
    computational_details = spectra_manager.ComputationalDetails()

    if not isinstance(fields, torch.Tensor):
        fields = torch.tensor(fields, dtype=dtype, device=device)

    B_low = fields[..., 0]
    B_high = fields[..., -1]
    B_low = B_low.unsqueeze(-1).repeat(*([1] * B_low.ndim), *mesh_size)
    B_high = B_high.unsqueeze(-1).repeat(*([1] * B_high.ndim), *mesh_size)

    populator = RalaxatioParmaetersPopulator(context, levels_pair, comp_method, init_temperature=temperature)

    _resfield_method = spectra_manager.res_field_algorithm.ResField(
        spin_system_dim=spin_system_dim,
        mesh_size=mesh_size,
        batch_dims=batch_dims,
        splitting_max_iterations=computational_details.res_field_split_max_iterations,
        r_tol=computational_details.res_field_r_tol,
        output_full_eigenvector=True,
        device=device,
        dtype=dtype
    )

    F, Gx, Gy, Gz = sample.get_hamiltonian_terms()
    resonance_frequency = torch.tensor(freq, device=device, dtype=dtype)

    (vector_down, vector_up), (lvl_down, lvl_up), res_fields, \
        resonance_energies, full_system_vectors = _resfield_method(sample, resonance_frequency, B_low, B_high, F, Gz)
    kinetic_coeffs = populator(res_fields, resonance_energies, lvl_down, lvl_up, full_system_vectors)
    return res_fields, kinetic_coeffs

