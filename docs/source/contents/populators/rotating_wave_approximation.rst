.. _rotating-wave-approximation:

Rotating Wave Approximation
============================

Overview
--------

The :class:`mars.population.populators.density_population.RWADensityPopulator` computes
time-dependent EPR signals using the density-matrix formalism in a rotating reference
frame. The implementation keeps the full density matrix and solves its Liouville-space
equation, while using the rotating-wave approximation (RWA) to remove the fast
counter-rotating terms.

The RWA is useful when the rotating-frame Hamiltonian and the relaxation model have the
required symmetry. The approximation is therefore best understood as a set of structural
assumptions about the Hamiltonian, driving field, and relaxation superoperator rather than
as a restriction to one special spin system.

Theory
------

Liouville-von Neumann Equation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The density matrix :math:`\rho(t)` evolves according to

.. math::

   \frac{d\rho}{dt} = -i[H, \rho] + \mathcal{R}[\rho].

Throughout MaRs, the Hamiltonian is stored in frequency units (Hz). Consequently,
the coherent part implemented in angular-frequency form contains the factor :math:`2\pi`:

.. math::

   \frac{d\rho}{dt} = -i\,2\pi[H,\rho] + \mathcal{R}[\rho].

After vectorization, the density matrix becomes a vector of length :math:`N^2` and the
dynamics are written as

.. math::

   \frac{d\boldsymbol{\rho}}{dt} = M(t)\boldsymbol{\rho},
   \qquad
   M(t) = -i\,\mathcal{L}_H(t) + \mathcal{R}(t),

The coherent Liouville operator is represented by a matrix acting on the vectorized
density matrix. Its exact Kronecker ordering follows the convention implemented by
:meth:`mars.population.transform.Liouvilleator.vec`; the matrix is assembled by
:class:`mars.population.tr_utils.EvolutionSuper` and combined with the relaxation matrix
supplied by a matrix generator.

Rotating Frame Transformation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For an oscillating field, introduce a rotation around the static-field direction,

.. math::

   W(t) = e^{i\omega S_Z t},
   \qquad
   \tilde{\rho}(t)=W(t)\rho(t)W^\dagger(t).

The transformed Hamiltonian is

.. math::

   \tilde H(t)=W H(t) W^\dagger - \omega S_Z

in angular-frequency units (with the same convention for the factor :math:`2\pi` used
in the implementation). The RWA retains the component of the microwave interaction that
is stationary in this frame and discards the counter-rotating contribution.

For a reference transverse axis chosen as :math:`X`, the resulting structure can be written
as

.. math::

   H_{\mathrm{eff}}
   = F + \Delta S_Z + \frac{B_1}{2}G_X,

where :math:`F` is the static spin Hamiltonian in the eigenbasis used by the propagator,
:math:`\Delta S_Z` is the rotating-frame detuning term, and the factor :math:`1/2` comes
from the resonant component of a linearly oscillating field,

.. math::

   \cos(\omega t)=\frac{e^{i\omega t}+e^{-i\omega t}}{2}.

The implementation makes this scaling explicit in
:meth:`RWADensityPopulator._compute_rabi_magnetic_scale`.

Constraints and Limitations
----------------------------

The RWA requires the following structural conditions in the MaRs implementation.

Zeeman operators and rotating-frame symmetry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Zeeman operators are represented as

.. math::

   G_X,\qquad G_Y,\qquad G_Z,

and the RWA construction retains the components compatible with the chosen rotating
axis. In the current implementation the Zeeman operators are transformed to the
Hamiltonian eigenbasis and the transverse operators are scaled by the resonant
microwave amplitude.

The rotating-frame transformation must have a well-defined
spin-rotation action. Strongly anisotropic or otherwise non-standard Zeeman structure can
therefore make the RWA inaccurate. Such cases should be checked against a non-RWA
calculation. Two possible checks are:

1. Compare stationary spectra computed with the direct Hamiltonian method and
   with the secular approximation (which shares the RWA assumptions).

   .. code-block:: python

       import torch
       import matplotlib.pyplot as plt
       from mars.spectra_manager import StationarySpectra

       # Define sample, context, freq, temperature
       sample = ...  # MultiOrientedSample
       context = ...  # relaxation context
       freq = 9.8e9
       temperature = 293

       spectra_direct = StationarySpectra(
           freq=freq,
           sample=sample,
           temperature=temperature,
           context=context,
           hamiltonian_mode="direct",
       )

       spectra_secular = StationarySpectra(
           freq=freq,
           sample=sample,
           temperature=temperature,
           context=context,
           hamiltonian_mode="secular",
       )

       fields = torch.linspace(0.2, 0.4, 500)
       spectrum_direct = spectra_direct(sample, fields)
       spectrum_secular = spectra_secular(sample, fields)

       plt.plot(fields, spectrum_direct, label='Direct')
       plt.plot(fields, spectrum_secular, label='Secular')
       plt.legend()
       plt.xlabel('Magnetic field (T)')
       plt.ylabel('Intensity')
       plt.show()


2. Compare the long‑time relaxation between a kinetic (rate‑equation) model
   and the RWA density‑matrix model. After the initial decay of coherences (for
   :math:`t \gg T_2`), both approaches should give similar population evolution.

Circular and linear polarization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The rotating-frame construction is naturally expressed using circular components. A
linearly polarized field contains two counter-rotating components:

.. math::

   B_1\cos(\omega t)
   = \frac{B_1}{2}e^{i\omega t}+\frac{B_1}{2}e^{-i\omega t}.

The RWA keeps the component rotating in the resonant direction. The omitted term rotates
at approximately :math:`2\omega` in the chosen frame. The approximation is therefore
controlled by the separation between the fast counter-rotating motion and the dynamical
scales resolved by the simulation.

Commutation with the static Hamiltonian
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the rotating transformation to leave the static part invariant, the implementation
requires the relevant symmetry condition

.. math::

   [F,S_Z]=0.

Equivalently, :math:`F` must not acquire additional explicit time dependence under the
rotation around the static-field axis. This is the key condition used below in the powder-
averaging derivation.

Relaxation Superoperator Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The relaxation superoperator must be compatible with the same rotating-frame symmetry.
In the matrix-element representation, the implementation uses the secular structure

.. math::

   R_{ijkl}\neq 0
   \quad\text{only when}\quad
   i-j=k-l.

This preserves the coherence-order sectors that are needed by the rotating-frame
construction. The allowed terms include population transfer and dephasing, while direct
mixing of different coherence orders is excluded.

Detection operator and the measured signal
------------------------------------------

The measured EPR signal is a linear functional of the laboratory-frame density
matrix :math:`\rho(t)`. The detection process is derived from the time
derivative of the microwave interaction Hamiltonian, :math:`H_1(t)`. In the
Zeeman interaction, the transverse operators are

.. math::

   G_X = g_{xx} S_X,\qquad G_Y = g_{yy} S_Y,

where :math:`g_{xx}` and :math:`g_{yy}` are the corresponding components of the
g-tensor. The microwave field is linearly polarized along :math:`x` with
amplitude :math:`B_1` and angular frequency :math:`\omega` (in frequency
units), so

.. math::

   H_1(t) = -B_1 \cos(\omega t)\, G_X
           = -B_1 g_{xx} \cos(\omega t)\, S_X.

The induced signal is

.. math::

   I(t) = \operatorname{Tr}\!\left[\rho(t)\,\frac{dH_1}{dt}\right]
        = B_1\,\omega\,g_{xx}\,\sin(\omega t)\,
          \operatorname{Tr}\!\left[\rho(t)\,S_X\right].

We now express the laboratory-frame density matrix through the RWA frame. The
rotating-frame transformation is

.. math::

   \rho(t) = W^\dagger(t)\,\tilde{\rho}(t)\,W(t),
   \qquad
   W(t)=e^{i\omega S_Z t}.

Using the spin-rotation identity

.. math::

   W(t)\,S_X\,W^\dagger(t)
   = S_X\cos(\omega t) - S_Y\sin(\omega t),

we obtain

.. math::

   \operatorname{Tr}\!\left[\rho(t)\,S_X\right]
   =
   \operatorname{Tr}\!\left[\tilde{\rho}(t)\,
     \bigl(S_X\cos(\omega t) - S_Y\sin(\omega t)\bigr)\right].

Substituting into the signal expression yields

.. math::

   I(t)
   =
   B_1\,\omega\,g_{xx}
   \Bigl[
     \sin(\omega t)\cos(\omega t)\,
     \operatorname{Tr}\!\left[\tilde{\rho}(t)\,S_X\right]
     -
     \sin^2(\omega t)\,
     \operatorname{Tr}\!\left[\tilde{\rho}(t)\,S_Y\right]
   \Bigr].

The detection circuit averages over many microwave periods. The period average
of the first term is zero, and the second term averages to one-half:

.. math::

   \langle I(t)\rangle_T
   =
   -\frac{B_1\,\omega\,g_{xx}}{2}\,
   \operatorname{Tr}\!\left[\tilde{\rho}(t)\,S_Y\right].

In terms of the Zeeman operator :math:`G_Y = g_{yy} S_Y`, this becomes

.. math::

   \langle I(t)\rangle_T
   =
   -\frac{B_1\,\omega}{2}\,
   \frac{g_{xx}}{g_{yy}}\,
   \operatorname{Tr}\!\left[\tilde{\rho}(t)\,G_Y\right].


In the isotropic case :math:`g_{xx}=g_{yy}`, the
prefactor equals unity and the familiar result is recovered. In the MaRs implementation,
for simplicity, this prefactor is not computed explicitly; it is considered approximately equal to 1.0. Since the RWA is already
applied in situations where the Zeeman interaction is nearly isotropic, this is
a minor additional simplification and does not significantly affect the
relative signal amplitudes, where RWA considiration is valid.

For the Liouville-space implementation, the identity

.. math::

   \operatorname{Tr}(D\rho)
   =
   \operatorname{vec}(D^{\mathsf T})^{\mathsf T}\operatorname{vec}(\rho)

is used. Hence the detection vector supplied to the solver is

.. math::

   \mathbf d = \operatorname{vec}(G_Y^{\mathsf T})
               = g_{yy}\,\operatorname{vec}(S_Y^{\mathsf T}).


Powder Averaging
----------------
or disordered samples, spectra are averaged over molecular orientations
:math:`(\alpha,\beta)` in Euler angle in ``"zyz'"`` notation. Averaging over
:math:`\gamma` reduces to averaging the initial density matrix: under the RWA,
this dependence can be represented by the unitary rotation
:math:`e^{i\gamma S_Z}` applied to the initial density matrix. Consequently,
the powder average over :math:`\gamma` is equivalent to an average over the
corresponding initial states.

Let consider it:

.. math::

   W_\gamma=e^{iS_Z\gamma}.

Let the reference RWA Hamiltonian be

.. math::

   H_X = F + \Delta S_Z + \frac{B_1}{2}G_X,

and let the orientation-dependent Hamiltonian be

.. math::

   H_\gamma = W_\gamma^\dagger H_X W_\gamma.

Because :math:`[F,S_Z]=0` and :math:`S_Z` is unchanged by its own rotation, the only
orientation dependence is the rotation of the transverse operator. The corresponding
density-matrix evolution can therefore be written as

.. math::

   \rho_\gamma(t)
   =W_\gamma^\dagger\,
   \rho_X\!\left(t;\rho_0^{(\gamma)}\right)W_\gamma,

with the rotated initial state

.. math::

   \rho_0^{(\gamma)}
   =W_\gamma\rho_0W_\gamma^\dagger.

The detection operator transforms in the same way, e.g.

.. math::

   D_\gamma=W_\gamma^\dagger G_Y W_\gamma.

Consequently the signal is invariant under moving the orientation dependence from the
Hamiltonian and detection operator into the initial density matrix:

.. math::

   \begin{aligned}
   I_\gamma(t)
   &=\operatorname{Tr}\!\left(D_\gamma\rho_\gamma(t)\right)\\
   &=\operatorname{Tr}\!\left(G_Y\rho_X\!\left(t;\rho_0^{(\gamma)}\right)\right).
   \end{aligned}

The Liouville equation is linear in :math:`\rho`, so averaging over :math:`\gamma` can be
performed before the propagation:

.. math::

   \begin{aligned}
   \overline{I}(t)
   &=\frac{1}{2\pi}\int_0^{2\pi} I_\gamma(t)\,d\gamma\\
   &=\operatorname{Tr}\!\left(G_Y\,\rho_X(t;\overline{\rho}_0)\right),
   \end{aligned}

where

.. math::

   \boxed{\displaystyle
   \overline{\rho}_0
   =\frac{1}{2\pi}\int_0^{2\pi}
      W_\gamma\rho_0W_\gamma^\dagger\,d\gamma.}

The implementation performs a single propagation with an averaged
initial density matrix instead of explicitly sampling the final Euler angle :math:`\gamma`.

The averaging has a simple form in the eigenbasis of :math:`S_Z`. If
:math:`S_Z|m\rangle=m|m\rangle`, then

.. math::

   \left(\overline{\rho}_0\right)_{mn}
   =\rho_{0,mn}
      \frac{1}{2\pi}\int_0^{2\pi}e^{i(m-n)\gamma}\,d\gamma
   =\begin{cases}
      \rho_{0,mn}, & m=n,\\
      0, & m\neq n.
     \end{cases}

Thus the averaged state is the projection of :math:`\rho_0`
onto the commutant of :math:`S_Z`:

.. math::

   [\overline{\rho}_0,S_Z]=0.


It first transforms the density matrix to
the product basis where :math:`S_Z` is diagonal, keeps only matrix elements connecting
states with equal :math:`S_Z` eigenvalues, and transforms the result back to the Hamiltonian
eigenbasis.

Computation and solver choices
------------------------------

The RWA evolution is always reduced to the generic linear equation

.. math::

   \dot{\mathbf n}(t)=M(t)\mathbf n(t),
   \qquad
   \mathbf n(t)=\operatorname{vec}(\rho(t)).

The choice of numerical solver depends on the time dependence and numerical properties of
:math:`M(t)`.

Adaptive ODE integration
~~~~~~~~~~~~~~~~~~~~~~~~

Use :meth:`EvolutionRWASolver.odeint_solver` when the relaxation matrix or any other part
of :math:`M(t)` varies continuously with time and there is no convenient piecewise-constant
approximation. The adaptive integrator controls its internal time step automatically.
This is the default solver selected when the context is marked as time-dependent.

Piecewise matrix exponentiation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use :meth:`EvolutionRWASolver.exponential_solver` when :math:`M(t)` is known at the requested
time points and may be regarded as constant on each interval :math:`[t_i,t_{i+1}]`:

.. math::

   \mathbf n_{i+1}
   \approx e^{M(t_i)\Delta t_i}\mathbf n_i,
   \qquad
   \Delta t_i=t_{i+1}-t_i.

This method can be efficient because the matrices are assembled for all time points at once,
but its accuracy depends on the quasi-stationary assumption within each interval.

Stationary eigen-decomposition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If :math:`M` is time-independent, then

.. math::

   \mathbf n(t)=e^{Mt}\mathbf n_0.

:meth:`EvolutionRWASolver.stationary_rate_solver` diagonalizes :math:`M` and evaluates the
exponential through its eigenmodes. It is efficient when the superoperator is diagonalizable
and the eigenvector basis is numerically well conditioned.

Stationary matrix exponential
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`EvolutionRWASolver.stationary_rate_solver_expm` evaluates
:math:`e^{Mt}` directly. It avoids the diagonalizability assumption and is therefore the
robust alternative when the eigenvector matrix is singular or poorly conditioned, at the
cost of additional matrix-exponential computation. (see more in kinetic approach computation :ref:`level_based_kinetic_approach:`)

Solver configuration
~~~~~~~~~~~~~~~~~~~~~

The solver is passed directly to :class:`RWADensityPopulator`. If ``solver=None``, the
populator chooses :meth:`EvolutionRWASolver.odeint_solver` for a time-dependent relaxation
context and :meth:`EvolutionRWASolver.stationary_rate_solver` otherwise.

For example:

.. code-block:: python

   from mars.population.tr_utils import EvolutionRWASolver
   from mars.population.populators.density_population import RWADensityPopulator

   # 1. General time-dependent relaxation: adaptive ODE integration
   populator = RWADensityPopulator(
       context=context,
       solver=EvolutionRWASolver.odeint_solver,
   )

   # 2. Time-independent generator, diagonalizable superoperator
   populator = RWADensityPopulator(
       context=context,
       solver=EvolutionRWASolver.stationary_rate_solver,
   )

   # 3. Piecewise-stationary evolution on the supplied time grid
   populator = RWADensityPopulator(
       context=context,
       solver=EvolutionRWASolver.exponential_solver,
   )

   # 4. Time-independent but possibly defective / ill-conditioned generator
   populator = RWADensityPopulator(
       context=context,
       solver=EvolutionRWASolver.stationary_rate_solver_expm,
   )

A practical selection rule is:

* use ``odeint_solver`` when the generator changes smoothly in time and robustness is the
  priority;
* use ``exponential_solver`` when the generator is supplied on a time grid and is nearly
  constant over each interval;
* use ``stationary_rate_solver`` for a genuinely stationary, well-conditioned problem;
* use ``stationary_rate_solver_expm`` when the problem is stationary but the eigenmode
  representation is unreliable.

The initial-state and powder-averaging logic is independent of the solver choice. In a powder
calculation, the phase-averaged initial density is constructed first and the selected solver
then propagates that single state.

Applicability
-------------

The RWA is suitable for:

* Organic radicals with isotropic g-factors
* Triplet states with small axial zero-field splitting aligned with the field
* Systems where coherence-population coupling is negligible

The RWA should **not** be used for:

* Transition metal complexes with anisotropic g-tensors
* Systems with strong non-secular relaxation
* Single-molecule magnets with large zero-field splittings
* High-field EPR where g-anisotropy is resolved

For such systems, use the propagator-based approach which imposes no similar approximations.
