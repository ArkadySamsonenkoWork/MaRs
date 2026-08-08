import math
import warnings

import torch
import typing as tp

from .. import tr_utils
from .. import transform
from .. import matrix_generators
from ... import constants
from .. import contexts
from . import core


class RWADensityPopulator(core.BaseTimeDepPopulator):
    """RWADensityPopulator.

    Computes time-dependent signal intensity using the density matrix formalism
    under the Rotating Wave Approximation (RWA).

    Relaxation dynamics are computed efficiently within the RWA framework,
    with relaxation parameters provided by the Context.

    Note: The Rotating Wave Approximation introduces several important constraints:

    1. The oscillating magnetic field (e.g., microwave or RF) is assumed to be circularly polarized.
    2. The g-tensor must be isotropic or close to isotropic,
    meaning the Zeeman operators are proportional to spin operators:
       Gx = gx * mu_B * Sx, Gy = gy * mu_B * Sy, Gz = gz * mu_B * Sz.,
    3. The static part of the Hamiltonian (denoted F) must commute with Gz (i.e., [F, Gz] = 0).
    4. The relaxation superoperator Rijkl-which couples matrix elements ρ_ij
    and rho_kl - is only non-zero when i - j equals k - l.
       This covers two processes:
       - Population transfer between energy levels (i = j, k = l), including pure decay (i = j = k = l).
       - Dephasing of coherences (i = k, j = l).
    """
    def __init__(self,
                 b1_field: tp.Optional[tp.Union[torch.Tensor, float]] = 1e-8,
                 context: tp.Optional[contexts.BaseContext] = None,
                 tr_matrix_generator_cls: tp.Type[matrix_generators.BaseGenerator] =
                 matrix_generators.DensityRWAGenerator,
                 solver: tp.Optional[tr_utils.EvolutionSolver] = None,
                 init_temperature: tp.Union[float, torch.Tensor] = 293.0,
                 energy_shifts: tp.Optional[tp.Union[torch.Tensor, tp.List]] = None,
                 difference_out: bool = False,
                 disordered: bool = True,
                 angle_average_steps: int = 4,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32):
        """
        :param b1_field:  Peak amplitude of the microwave magnetic field
        in the laboratory frame. The  field is:
        B_1(t) = b1_field * cos(ωt) * e_x
        It represents the full oscillating amplitude.
        For RWA computations it means that the oscillating part of Hamiltonian H in rotating frame will be:
        H1 = b1_field * Gx / 2, where Gx is x-component of Zeeman operator

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

        :param disordered: If True, use powder averaging; if False, use crystal geometry. Default is True

        :param angle_average_steps: The number of steps to average the signal
        intensity over the euler angle rotating over the stationary magnetic field
        For RWA methods the modification of the initial density is used and this parameter is not used.


        :param device: device to compute (cpu / gpu)
        :param dtype: dtype of computation
        """
        super().__init__(
            context, tr_matrix_generator_cls, solver, init_temperature, energy_shifts, difference_out, device, dtype)
        self.register_buffer(
            "two_pi", torch.tensor(math.pi * 2, device=device, dtype=dtype)
        )
        self.register_buffer("b1_field", torch.tensor(b1_field, device=device, dtype=dtype))
        self.liouvilleator = transform.Liouvilleator
        self.disordered: tp.Optional[bool] = disordered
        self.angle_average_steps = angle_average_steps

    def init_solver(self, solver: tp.Optional[tp.Callable]) -> tp.Callable:
        if solver is not None:
            return solver
        if self.time_dependant:
            return tr_utils.EvolutionRWASolver.odeint_solver
        else:
            return tr_utils.EvolutionRWASolver.stationary_rate_solver

    def _init_context_meta(self):
        """Itializes metadata flags based on the presence and configuration of
        the relaxation Context.

        Determines:
          - Whether initial populations/density matrices are provided by the Context (`contexted = True`),
            or should be computed from temperature (`contexted = False`);
          - Whether the relaxation parameters are time-dependent (`time_dependant` flag).

        These flags control which internal methods are used for initialization and solver selection.
        """
        if self.context is not None:
            if self.context.contexted_init_population:
                self.contexted = True
                self._getter_init_density = self._context_dependant_init_density
                self._getter_init_population = self._context_dependant_init_population
            else:
                self.contexted = False
                self._getter_init_density = self._temp_dependant_init_density
                self._getter_init_population = self._temp_dependant_init_population

            self.time_dependant = self.context.time_dependant

        else:
            self.contexted = False
            self._getter_init_density = self._temp_dependant_init_density
            self._getter_init_population = self._temp_dependant_init_population

            self.time_dependant = False

    def _get_initial_Hamiltonian(self, energies: torch.Tensor):
        """Constructs the static (time-independent) part of the spin
        Hamiltonian in the eigenbasis.

        :param energies:
            Eigenenergies of the spin system, shape [..., M, N], where M is the number of field/orientation points,
            and N is the number of energy levels.

        :return:
            Diagonal Hamiltonian tensor of shape [..., M, N, N], where N is number of levels, M is number of transition
        """
        return torch.diag_embed(energies)

    def _initial_density(
            self, energies: torch.Tensor, lvl_down: torch.Tensor, lvl_up: torch.Tensor,
            full_system_vectors: tp.Optional[torch.Tensor],
            *args, **kwargs
    ):
        """Computes the initial density matrix either from thermal equilibrium
        or from a context-defined state.

        Dispatches to one of two internal methods based on whether the Context provides an initial state.

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
        :return: density in the initial state
        """
        return self._getter_init_density(energies, lvl_down, lvl_up, full_system_vectors)

    def _transform_to_eigenbasis(self, full_basis: torch.Tensor, args_matrix: tp.Iterable[torch.Tensor]):
        """Transforms an Iterable of operators (matrices) from the
        computational basis to the Hamiltonian eigenbasis.

        Applies the unitary transformation: A_eigen = U⁺ A U,
        where U = `full_basis` contains the eigenvectors as columns.

        :param full_basis:
            Unitary transformation matrix (eigenvectors), shape [..., N, N].
        :param args_matrix:
            List of operators to transform, each of shape [..., N, N].
        :return:
            List of transformed operators in the eigenbasis.
        """
        out_matrix = []
        for matrix in args_matrix:
            out_matrix.append(full_basis.conj().transpose(-1, -2) @ matrix @ full_basis)
        return out_matrix

    def _compute_rabi_magnetic_scale(self):
        """
        Compute the magnetic coupling scale factor.

        In the RWA, the oscillating field is decomposed using Euler's formula:
            cos(ωt) = (e^(iωt) + e^(-iωt)) / 2
        This yields two counter-rotating components, each with amplitude b1_field/2.
        Only the resonant component (rotating in the same direction as the spin
        precession) is retained. Therefore, the effective static field in the
        rotating frame is exactly half of the lab-frame amplitude.
        """
        return self.two_pi * self.b1_field / 2

    def _compute_hamiltonian_operators(
            self,
            Gx: torch.Tensor, Gy: torch.Tensor, Oz: torch.Tensor,
            full_system_vectors: torch.Tensor,
    ) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Transform Hamiltonian components to eigenbasis and apply scaling."""
        Gx, Gy, Oz = self._transform_to_eigenbasis(
            full_system_vectors,
            (Gx.unsqueeze(-3), Gy.unsqueeze(-3), Oz.unsqueeze(-3))
        )
        scale = self._compute_rabi_magnetic_scale()
        Gx = scale * Gx
        Gy = scale * Gy
        return Gx, Gy, Oz

    def _init_tr_matrix_generator(self,
                                  time: torch.Tensor,
                                  res_fields: torch.Tensor,
                                  lvl_down: torch.Tensor,
                                  lvl_up: torch.Tensor, energies: torch.Tensor,
                                  vector_down: torch.Tensor,
                                  vector_up: torch.Tensor,
                                  full_system_vectors: tp.Optional[torch.Tensor],
                                  resonance_frequency: torch.Tensor,
                                  H0: torch.Tensor,
                                  Sz: torch.Tensor,
                                  Ht: torch.Tensor,
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

        :param resonance_frequency: Resonance frequency of the spin transition, in (Hz).
        Scalar value (shape: `[]`).

        :param H0: Static (time-independent) part of the spin Hamiltonian, angular frequency expressed in
                                                                                                    (Hz * 2π = rad/s).
        :param Sz: Electron z-moment projection operator.
        :param Ht: Time-dependent (oscillating) component of the Hamiltonian, given in angular
        frequency units (Hz * 2π = rad/s).

        :param args: tuple, optional.
        :param kwargs : dict, optional

        :param return:
        -------
        TransitionMatrixGenerator instance
        """
        shift = - Sz * self.two_pi * resonance_frequency
        tr_matrix_generator = self.tr_matrix_generator_cls(context=self.context,
                                                           init_temperature=self.init_temperature,
                                                           res_fields=res_fields,
                                                           energies=energies,
                                                           stationary_hamiltonian=H0 + shift + Ht,
                                                           lvl_down=lvl_down, lvl_up=lvl_up,
                                                           full_system_vectors=full_system_vectors,
                                                           )
        return tr_matrix_generator

    def _compute_average_density(
            self, initial_density: torch.Tensor, full_system_vectors: torch.Tensor, Sz_product: torch.Tensor, tol=1e-6
    ) -> torch.Tensor:
        """
        :param initial_density: Initial density matrix in the Hamiltonian eigenbasis. Shape [..., N, N].
        :param full_system_vectors: Eigenvectors of the Hamiltonian in the product basis. Shape [..., N, N].
        :param Sz_product: Sz operator in the individual spin projections basis (diagonal). Shape [N, N].
        :param tol: Tolerance for comparing Sz eigenvalues to handle numerical noise.
        :return: Phase-averaged density matrix in the Hamiltonian eigenbasis. Shape [..., N, N].
        """
        m_vals = torch.diag(Sz_product)
        diff = torch.abs(m_vals.unsqueeze(-1) - m_vals.unsqueeze(-2))
        mask = (diff < tol).to(initial_density.dtype)

        rho_product = full_system_vectors @ initial_density @ full_system_vectors.mH

        rho_product_avg = rho_product * mask
        rho_avg = full_system_vectors.mH @ rho_product_avg @ full_system_vectors

        return rho_avg

    def _compute_powder(self,
                        time: torch.Tensor, res_fields: torch.Tensor,
                        lvl_down: torch.Tensor, lvl_up: torch.Tensor,
                        energies: torch.Tensor, vector_down: torch.Tensor,
                        vector_up: torch.Tensor,
                        full_system_vectors: tp.Optional[torch.Tensor],
                        initial_density: torch.Tensor,
                        Gx: torch.Tensor, Gy: torch.Tensor, Sz_eigen: torch.Tensor, Sz_product: torch.Tensor,
                        resonance_frequency: torch.Tensor,
                        H0: torch.Tensor,
                        *args, **kwargs) -> torch.Tensor:
        """Computes the time-resolved EPR signal for a disordered (powder)
        sample by averaging over the microwave field polarization.

         In powder samples, the orientation of the microwave magnetic field relative to the molecular structure is random.
         The orientation of the molecule is described by three Euler angles.
         The first two angles determine the energy and eigenvectors.
         The last angle does not change the energy or eigenvectors, but it does alter the signal intensity.
         This can be described as a rotation between Gx and Gy. Thus, the final signal is the average of this rotation.


        :param time:
            Time points at which the signal is evaluated, shape [T].
        :param res_fields:
            Resonance fields corresponding to each orientation or transition.
        :param lvl_down:
            Energy levels of lower states from which transitions occur.
        :param lvl_up:
            Energy levels of upper states to which transitions occur.
        :param energies:
            The energies of spin states. The shape is [..., N]
        :param vector_down:
            Eigenvectors of the lower energy states.
        :param vector_up:
            Eigenvectors of the upper energy states.
        :param full_system_vectors:
            Eigenvectors of the full set of energy levels.
        :param initial_density:
            Initial density matrix in the eigenbasis of the full Hamiltonian, shape [..., N, N].
        :param Gx, Gy:
            Transformed Zeeman operators (x- and y-components) in the eigenbasis, each of shape [..., 1, N, N].
        :param Sz_eigen:
            Electron z-projection operator in the eigen basis
        :param Sz_product:
            Electron z-projection operator in the product basis. In this basis it is diagonal
        :param resonance_frequency:
            Microwave frequency in Hz (scalar).
        :param H0:
            Static Hamiltonian in eigenbasis expressed in Hz * 2pi = rad/s
        :return:
            Averaged time
        """
        evo = tr_utils.EvolutionSuper(energies)
        initial_density = self._compute_average_density(initial_density, full_system_vectors, Sz_product)
        tr_matrix_generator = self._init_tr_matrix_generator(time, res_fields,
                                                             lvl_down, lvl_up, energies, vector_down,
                                                             vector_up, full_system_vectors, resonance_frequency,
                                                             H0, Sz_eigen, Gx,
                                                             *args, **kwargs)
        out = self.solver(
            time, self.liouvilleator.vec(initial_density),
            evo, tr_matrix_generator, self.liouvilleator.vec(Gy.transpose(-2, -1))
        )
        return self._post_compute(out)

    def _compute_crystal(self,
                         time: torch.Tensor, res_fields: torch.Tensor,
                         lvl_down: torch.Tensor, lvl_up: torch.Tensor,
                         energies: torch.Tensor, vector_down: torch.Tensor,
                         vector_up: torch.Tensor,
                         full_system_vectors: tp.Optional[torch.Tensor],
                         initial_density: torch.Tensor,
                         Gx: torch.Tensor, Gy: torch.Tensor, Sz_eigen: torch.Tensor, Sz_product: torch.Tensor,
                         resonance_frequency: torch.Tensor,
                         H0: torch.Tensor,
                         *args, **kwargs) -> torch.Tensor:
        """Computes the time-resolved EPR signal for a single-crystal or many-
        crystal sample with fixed microwave polarization.

        In crystal simulations, the microwave field direction is fixed relative to the molecular frame.
        By convention, the excitation is applied along the x-axis of the laboratory frame, so only the Gx operator
        contributes to the transition intensity. No averaging over gamma is performed.

        :param time:
            Time points for signal evaluation, shape [T].
        :param res_fields:
            Resonance fields (used for shape alignment; not directly used in computation).
        :param lvl_down:
            Energy levels of lower states from which transitions occur.
        :param lvl_up:
            Energy levels of upper states to which transitions occur.
        :param energies:
            The energies of spin states. The shape is [..., N]
        :param vector_down:
            Eigenvectors of the lower energy states.
        :param vector_up:
            Eigenvectors of the upper energy states.
        :param full_system_vectors:
            Eigenvectors of the full set of energy levels.
        :param initial_density:
            Initial density matrix in the eigenbasis, shape [..., N, N].
        :param Gx, Gy:
            Zeeman operators in the eigenbasis; only Gx is used in this method.
        :param Sz_eigen:
            Electron z-projection operator in the eigen basis
        :param Sz_product:
            Electron z-projection operator in the product basis. In this basis it is diagonal
        :param resonance_frequency:
            Microwave frequency in Hz (scalar).
        :param H0:
            Static Hamiltonian in eigenbasis, expressed in Hz * 2pi = rad/s

        :return:
            Time-dependent signal intensity for a single crystal, shape [T, ..., Tr].
        """
        evo = tr_utils.EvolutionSuper(energies)
        tr_matrix_generator = self._init_tr_matrix_generator(time, res_fields,
                                                             lvl_down, lvl_up, energies, vector_down,
                                                             vector_up, full_system_vectors, resonance_frequency,
                                                             H0, Sz_eigen, Gx,
                                                             *args, **kwargs)
        out = self.solver(
            time, self.liouvilleator.vec(initial_density),
            evo, tr_matrix_generator, self.liouvilleator.vec(Gy.transpose(-2, -1))
        )
        return self._post_compute(out)

    def _compute_out(self,
                     time: torch.Tensor, res_fields: torch.Tensor,
                     lvl_down: torch.Tensor, lvl_up: torch.Tensor,
                     energies: torch.Tensor, vector_down: torch.Tensor,
                     vector_up: torch.Tensor,
                     full_system_vectors: tp.Optional[torch.Tensor],
                     initial_density: torch.Tensor,
                     Gx: torch.Tensor, Gy: torch.Tensor, Sz_eigen: torch.Tensor, Sz_product,
                     resonance_frequency: torch.Tensor,
                     H0: torch.Tensor,
                     *args, **kwargs) -> torch.Tensor:
        """Computes the time-resolved EPR signal.

        Creates the TransitionMatrixGenerator and dispatches to `_compute_crystal` or `_compute_powder`
        based on `self.disordered`.

        For powder samples: signal = (Ix + Iy) / 2 (averages over Gx and Gy)
        For crystal samples: signal = Ix (uses only Gx)

        :param time:
            Time points for signal evaluation, shape [T].
        :param res_fields:
            Resonance fields (used for shape alignment; not directly used in computation).
        :param lvl_down:
            Energy levels of lower states from which transitions occur.
        :param lvl_up:
            Energy levels of upper states to which transitions occur.
        :param energies:
            The energies of spin states. The shape is [..., N]
        :param vector_down:
            Eigenvectors of the lower energy states.
        :param vector_up:
            Eigenvectors of the upper energy states.
        :param full_system_vectors:
            Eigenvectors of the full set of energy levels.
        :param initial_density:
            Initial density matrix in the eigenbasis, shape [..., N, N].
        :param Gx, Gy:
            Zeeman operators in the eigenbasis.
        :param Sz_eigen:
            Electron z-projection operator in the eigen basis
        :param Sz_product:
            Electron z-projection operator in the product basis. In this basis it is diagonal
        :param resonance_frequency:
            Microwave frequency in Hz (scalar).
        :param H0:
            Static Hamiltonian in eigenbasis.
        :return:
            Time-dependent signal intensity, shape [T, ..., Tr].
        """

        if self.disordered:
            return self._compute_powder(
                time, res_fields,
                lvl_down, lvl_up, energies,
                vector_down, vector_up,  full_system_vectors,
                initial_density,
                Gx, Gy, Sz_eigen, Sz_product,
                resonance_frequency, H0, *args, **kwargs)
        else:
            return self._compute_crystal(
                time, res_fields,
                lvl_down, lvl_up, energies,
                vector_down, vector_up, full_system_vectors,
                initial_density,
                Gx, Gy, Sz_eigen, Sz_product,
                resonance_frequency, H0, *args, **kwargs)

    def forward(self,
                time: torch.Tensor, res_fields: torch.Tensor,
                lvl_down: torch.Tensor, lvl_up: torch.Tensor,
                energies: torch.Tensor, vector_down: torch.Tensor,
                vector_up: torch.Tensor,
                full_system_vectors: tp.Optional[torch.Tensor],
                F: torch.Tensor, Gx: torch.Tensor, Gy: torch.Tensor, Gz: torch.Tensor, Sz: torch.Tensor,
                resonance_frequency: torch.Tensor,
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

        :param F: Magnetic free part of spin Hamiltonian H = F + B * G. The shape is [...., N, N]
        :param Gx: x-part of Hamiltonian Zeeman Term. The shape is [...., N, N]
        :param Gy: y-part of Hamiltonian Zeeman Term. The shape is [...., N, N]
        :param Gz: z-part of Hamiltonian Zeeman Term. The shape is [...., N, N]
        :param Sz: z-part of electron z-projection. The shape is [...., N, N]

        :param resonance_frequency: Resonance frequency of the spin transition, in hertz (Hz).
        Scalar value (shape: `[]`).

        :param args: additional args from spectra creator.

        :param kwargs:
        :return: Part of the transition intensity that depends on the population of the levels.
        The shape is [T, ...., Tr]
        """
        H0 = self._get_initial_Hamiltonian(energies) * self.two_pi
        initial_density = self._initial_density(energies, lvl_down, lvl_up, full_system_vectors)
        Gx, Gy, Sz_eigen = self._compute_hamiltonian_operators(
            Gx, Gy, Sz.unsqueeze(-3), full_system_vectors)

        return self._compute_out(
            time, res_fields,
            lvl_down, lvl_up, energies, vector_down,
            vector_up,  full_system_vectors, initial_density,
            Gx, Gy, Sz_eigen, Sz,
            resonance_frequency,  H0, *args, **kwargs)


class PropagatorDensityPopulator(RWADensityPopulator):
    """PropagatorDensityPopulator computes time-resolved EPR signals by
    explicitly evaluating the full.

    time-evolution propagator U(t, 0).

    This method supports:
      - Arbitrary g-tensor anisotropy,
      - General relaxation superoperators (including coherence-population coupling),
      - Any kind of zero field splitting.

    The propagator is computed over one period of the microwave field and then extrapolated
    to arbitrary detection times using pereodic solution of time-evolution equation. For disordered (powder) samples,
    the signal is averaged over the Euler angle γ by evaluating two orthogonal field polarizations.

    While more computationally demanding than RWA-based methods, this approach is necessary
    for systems where RWA assumptions (close to isotropic g, circular polarization, [F, Gz] = 0) are violated,
    such as in single-molecule magnets, metal complexes, or high-field EPR.
    """
    def __init__(self,
                 b1_field: tp.Optional[tp.Union[torch.Tensor, float]] = 1e-7,
                 measurement_time: tp.Optional[float] = None,
                 context: tp.Optional[contexts.BaseContext] = None,
                 tr_matrix_generator_cls: tp.Type[matrix_generators.BaseGenerator] =
                 matrix_generators.DensityPropagatorGenerator,
                 solver: tp.Optional[tr_utils.EvolutionSolver] = tr_utils.EvolutionPropagatorSolver(),
                 init_temperature: tp.Union[float, torch.Tensor] = 293.0,
                 energy_shifts: tp.Optional[tp.Union[torch.Tensor, tp.List]] = None,
                 difference_out: bool = False,
                 disordered: bool = True,
                 angle_average_steps: int = 4,
                 n_steps: int = 100,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32):
        """
        :param b1_field:  Peak amplitude of the microwave magnetic field
        in the laboratory frame. The  field is:
        B_1(t) = b1_field * cos(ωt) * e_x
        It represents the full oscillating amplitude.

        :param measurement_time: The EPR spectrometer measurements time in seconds.
        The real experimental  time is about 40-200ns.
        By default, this parameter is None, meaning that only one period of oscillating magnetic field is measured

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

        :param disordered: If True, use powder averaging; if False, use crystal geometry. Default is True

        :param angle_average_steps: The number of steps to average the signal
        intensity over the euler angle rotating over the stationary magnetic field
        For Propagator methods this parameter define how many directions of outer magnetic field is used to compute
        the avergae signal with the general formula Gperp = Gx * cos (phi) + Gy * sin(phi)

        :param n_steps: the number of steps to find U(2pi) and Integral U sin(phi) at the interval (0, 2pi].
            This is quite crucial value for rk4 loop. If this value is small then the results will be dramatic

        :param device: device to compute (cpu / gpu)
        :param dtype: dtype of computation
        """
        super().__init__(
            b1_field, context, tr_matrix_generator_cls, solver,
            init_temperature, energy_shifts,
            difference_out, disordered, angle_average_steps, device, dtype
        )
        measurement_time =\
            torch.tensor(measurement_time, dtype=dtype, device=device)\
                if measurement_time is not None else measurement_time
        self.register_buffer("measurement_time", measurement_time)
        self.n_steps = n_steps

    def _compute_rabi_magnetic_scale(self):
        """
        Compute the magnetic coupling scale factor.
        """
        return self.two_pi * self.b1_field

    def _init_tr_matrix_generator(self,
                                  time: torch.Tensor,
                                  res_fields: torch.Tensor,
                                  lvl_down: torch.Tensor,
                                  lvl_up: torch.Tensor,
                                  energies: torch.Tensor,
                                  vector_down: torch.Tensor,
                                  vector_up: torch.Tensor,
                                  full_system_vectors: tp.Optional[torch.Tensor],
                                  resonance_frequency: torch.Tensor, H0: torch.Tensor,
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
            make the creator to compute these vectors.
            The default behavior, whether to calculate vectors or not,
            depends on the specific Spectra Manager and its settings.

        :param resonance_frequency: Resonance frequency of the spin transition, in hertz (Hz).
        Scalar value (shape: `[]`).
        :param H0: Static (time-independent) part of the spin Hamiltonian, expressed in hertz (Hz * 2π = rad / s).
        :param Ht: Time-dependent (oscillating) component of the Hamiltonian, given in angular
        frequency units (Hz * 2π = rad/s).

        :param args: tuple, optional.
        If the resfield algorithm returns full_system_vectors the full_system_vectors = args[0]

        :param kwargs : dict, optional

        :param return:
        -------
        TransitionMatrixGenerator instance
        """
        tr_matrix_generator = self.tr_matrix_generator_cls(context=self.context,
                                                           init_temperature=self.init_temperature,
                                                           res_fields=res_fields,
                                                           energies=energies,
                                                           full_system_vectors=full_system_vectors,
                                                           stationary_hamiltonian=H0,
                                                           lvl_down=lvl_down, lvl_up=lvl_up,
                                                           )
        return tr_matrix_generator

    def _compute_powder(self,
                evo: tr_utils.EvolutionSuper,
                tr_matrix_generator: tr_utils.matrix_generators.BaseGenerator,
                time: torch.Tensor, res_fields: torch.Tensor,
                initial_density: torch.Tensor,
                Gx: torch.Tensor, Gy: torch.Tensor,
                resonance_frequency: torch.Tensor,
                *args, **kwargs) -> torch.Tensor:

        """Computes the time-resolved EPR signal for a disordered (powder)
        sample.

         by averaging over the microwave field polarization.

         In powder samples, the orientation of the microwave magnetic field relative to the molecular structure is random.
         The orientation of the molecule is described by three Euler angles.
         The first two angles determine the energy and eigenvectors.
         The last angle does not change the energy or eigenvectors, but it does alter the signal intensity.
         This can be described as a rotation between Gx and Gy. Thus, the final signal is the average of this rotation.

        The method uses a Floquet-inspired stationary solver that integrates the evolution over one microwave period,
        optionally extended to a finite detector integration time (`self.measurement_time`).

        :param evo:
            Evolution superoperator generator that constructs the Liouville-space relaxation superoperator.
        :param tr_matrix_generator:
            Generator that provides the static part of the relaxation superoperator based on the spin system and context.
        :param time:
            Time points at which the signal is evaluated, shape [..., T]. where ... is batch shape
        :param res_fields:
            Resonance fields corresponding to each orientation or transition.
        :param initial_density:
            Initial density matrix in the eigenbasis of the full Hamiltonian, shape [..., N, N].
        :param Gx, Gy:
            Transformed Zeeman operators (x- and y-components) in the eigenbasis, each of shape [..., 1, N, N].
        :param resonance_frequency:
            Microwave frequency in Hz (scalar).
        :return: Signal intensity over time, shape [..., T ..., Tr],
        where the first ... is batch dimensions, the next ... is orientations, Tr is transition fields
        """
        tau = 1 / resonance_frequency
        delta_time = time[..., 1] - time[..., 0]
        ratio = delta_time / tau
        if (ratio < 100.0).any:
            warnings.warn(
                f"Time discretization may become visible: min(dt/tau)={ratio.max().item():.3g}. "
                "Increase the time step or use a coarser time grid if possible.",
                stacklevel=2,
            )

        delta_phi = self.two_pi / self.n_steps
        res_omega = resonance_frequency * self.two_pi
        superop_static = evo(*tr_matrix_generator(time))

        target_shape = list(time.shape) + list(Gx.shape[time.dim()-1:-2])
        out = torch.zeros(target_shape, dtype=res_fields.dtype, device=res_fields.device)

        z_axis_angles = torch.linspace(0, 2 * math.pi, self.angle_average_steps + 1)[:-1]
        delta_angle_z_axis = z_axis_angles[1] - z_axis_angles[0]
        for z_axis_angle in z_axis_angles:
            Gt = Gx * torch.cos(z_axis_angle) + Gy * torch.sin(z_axis_angle)
            out += self.solver.stationary_rate_solver(
                time, initial_density, Gt, superop_static,
                res_omega, tau, delta_phi, self.measurement_time, self.n_steps
            )
        return self._post_compute(out) * delta_angle_z_axis

    def _compute_crystal(self,
                         evo: tr_utils.EvolutionSuper,
                         tr_matrix_generator: tr_utils.matrix_generators.BaseGenerator,
                         time: torch.Tensor, res_fields: torch.Tensor,
                         initial_density: torch.Tensor,
                         Gx: torch.Tensor, Gy: torch.Tensor,
                         resonance_frequency: torch.Tensor,
                         *args, **kwargs) -> torch.Tensor:

        """Omputes the time-resolved EPR signal for a single-crystal or many-
        crystal sample.

        with fixed microwave polarization.

        In crystal simulations, the microwave field direction is fixed relative to the molecular frame.
        By convention, the excitation is applied along the x-axis of the laboratory frame, so only the Gx operator
        contributes to the transition intensity. No averaging over γ is performed.

        The signal is computed using a stationary Floquet-based solver that integrates the quantum evolution
        over one microwave period (or a user-defined `measurement_time`).

        :param evo:
            Evolution superoperator generator that constructs the Liouville-space relaxation superoperator.
        :param tr_matrix_generator:
            Generator that provides the static relaxation superoperator.
        :param time:
            Time points for signal evaluation, shape [T].
        :param res_fields:
            Resonance fields (used for shape alignment; not directly used in computation).
        :param initial_density:
            Initial density matrix in the eigenbasis, shape [..., N, N].
        :param Gx, Gy:
            Zeeman operators in the eigenbasis; only Gx is used in this method.
        :param resonance_frequency:
            Microwave frequency in Hz (scalar).

        :return: Signal intensity over time, shape [..., T ..., Tr],
        where the first ... is batch dimensions, the next ... is orientations, Tr is transition fields
        """
        tau = 1 / resonance_frequency
        delta_time = time[..., 1] - time[..., 0]
        ratio = delta_time / tau
        if (ratio < 100.0).any:
            warnings.warn(
                f"Time discretization may become visible: min(dt/tau)={ratio.max().item():.3g}. "
                "Increase the time step or use a coarser time grid if possible.",
                stacklevel=2,
            )

        delta_phi = self.two_pi / self.n_steps
        res_omega = resonance_frequency * self.two_pi
        superop_static = evo(*tr_matrix_generator(time))

        target_shape = list(time.shape) + list(Gx.shape[time.dim() - 1:-2])
        out = torch.zeros(target_shape, dtype=res_fields.dtype, device=res_fields.device)

        for Gt in [Gx]:
            out += self.solver.stationary_rate_solver(
                time, initial_density, Gt, superop_static,
                res_omega, tau, delta_phi, self.measurement_time, self.n_steps
            )
        return self._post_compute(out)

    def _compute_out(self,
                time: torch.Tensor, res_fields: torch.Tensor,
                lvl_down: torch.Tensor, lvl_up: torch.Tensor,
                energies: torch.Tensor, vector_down: torch.Tensor,
                vector_up: torch.Tensor,
                full_system_vectors: tp.Optional[torch.Tensor],
                initial_density: torch.Tensor,
                Gx: torch.Tensor, Gy: torch.Tensor, Sz_eigen: torch.Tensor, Sz_prod: torch.Tensor,
                resonance_frequency: torch.Tensor,
                H0: torch.Tensor,
                *args, **kwargs) -> torch.Tensor:
        """Computes the time-resolved EPR signal.

        The signal is computed using a stationary Floquet-based solver that integrates the quantum evolution
        over one microwave period (or a user-defined `measurement_time`).

        :param time:
            Time points for signal evaluation, shape [T].
        :param res_fields:
            Resonance fields (used for shape alignment; not directly used in computation).
        :param lvl_down:
            Energy levels of lower states from which transitions occur.
        :param lvl_up:
            Energy levels of upper states to which transitions occur.
        :param energies:
            The energies of spin states. The shape is [..., N]
        :param vector_down:
            Eigenvectors of the lower energy states.
        :param vector_up:
            Eigenvectors of the upper energy states.
        :param full_system_vectors:
            Eigenvectors of the full set of energy levels.
        :param initial_density:
            Initial density matrix in the eigenbasis, shape [..., N, N].
        :param Gx, Gy:
            Zeeman operators in the eigenbasis.
        :param Sz_eigen:
            Electron z-projection operator in the eigen basis
        :param Sz_product:
            Electron z-projection operator in the product basis. In this basis it is diagonal
        :param resonance_frequency:
            Microwave frequency in Hz (scalar).
        :param H0:
            Static Hamiltonian in eigenbasis. The values Hz * 2pi (rad / s)
        :return:
            Time-dependent signal intensity, shape [T, ..., Tr].
        """
        evo = tr_utils.EvolutionSuper(energies)

        tr_matrix_generator = self._init_tr_matrix_generator(
            time, res_fields,
            lvl_down, lvl_up, energies, vector_down,
            vector_up, full_system_vectors, resonance_frequency,
            H0,
            *args, **kwargs)

        if self.disordered:
            return self._compute_powder(evo, tr_matrix_generator,
                time, res_fields, initial_density, Gx, Gy, resonance_frequency, *args, **kwargs)
        else:
            return self._compute_crystal(evo, tr_matrix_generator,
                time, res_fields, initial_density, Gx, Gy, resonance_frequency, *args, **kwargs)

    def forward(self,
                time: torch.Tensor, res_fields: torch.Tensor,
                lvl_down: torch.Tensor, lvl_up: torch.Tensor,
                energies: torch.Tensor, vector_down: torch.Tensor,
                vector_up: torch.Tensor,
                full_system_vectors: tp.Optional[torch.Tensor],
                F: torch.Tensor, Gx: torch.Tensor, Gy: torch.Tensor, Gz: torch.Tensor, Sz: torch.Tensor,
                resonance_frequency: torch.Tensor,
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

        :param F: Magnetic free part of spin Hamiltonian H = F + B * G. The shape is [...., N, N]
        :param Gx: x-part of Hamiltonian Zeeman Term. The shape is [...., N, N]
        :param Gy: y-part of Hamiltonian Zeeman Term. The shape is [...., N, N]
        :param Gz: z-part of Hamiltonian Zeeman Term. The shape is [...., N, N]
        :param Sz: z-part of electron projections. The shape is [...., N, N]

        :param resonance_frequency: Resonance frequency of the spin transition, in hertz (Hz).
        Scalar value (shape: `[]`).

        :param args: additional args from spectra creator.

        :param kwargs:
        :return: Part of the transition intensity that depends on the population of the levels.
        The shape is [T, ...., Tr]
        """
        H0 = self._get_initial_Hamiltonian(energies) * self.two_pi
        Gx, Gy, Sz_eigen = self._compute_hamiltonian_operators(
            Gx, Gy, Sz.unsqueeze(-3), full_system_vectors)
        initial_density = self._initial_density(energies, lvl_down, lvl_up, full_system_vectors)
        return self._compute_out(
            time, res_fields,
            lvl_down, lvl_up,
            energies, vector_down,
            vector_up,
            full_system_vectors,
            initial_density,
            Gx, Gy, Sz_eigen, Sz,
            resonance_frequency,
            H0,
            *args, **kwargs)