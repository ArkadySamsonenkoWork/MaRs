.. _rotating-wave-approximation:

Rotating Wave Approximation
============================

Overview
--------

The :class:`mars.population.populators.density_population.RWADensityPopulator` computes
time-dependent EPR signals using the density-matrix formalism in a rotating
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

For an oscillating field, introduce a rotation around the static-field direction
using the total electron-and-nuclear spin projection,

.. math::

   S_Z^{\mathrm{tot}}=\sum_e S_{e,Z}+\sum_n I_{n,Z},
   \qquad
   W(t)=e^{i\omega S_Z^{\mathrm{tot}} t},
   \qquad
   \tilde{\rho}(t)=W(t)\rho(t)W^\dagger(t).

For Hamiltonians expressed in Hz, with :math:`\nu=\omega/(2\pi)`, the
transformed Hamiltonian is

.. math::

   \tilde H(t)=W H(t) W^\dagger-\nu S_Z^{\mathrm{tot}}.

The RWA retains the component of the microwave interaction that is stationary
in this frame and discards the counter-rotating contribution.
The effective Hamiltonian assembled by the RWA populator has the form

.. math::

   H_{\mathrm{eff}}
   =F_{\mathrm{sec}}+B_0G_{Z,\mathrm{sec}}
   -\nu_{\mathrm{rf}}S_Z^{\mathrm{tot}}
   +\frac{B_1}{2}G_{X,\mathrm{sec}}.

Here :math:`F_{\mathrm{sec}}`, :math:`G_{Z,\mathrm{sec}}` and
:math:`G_{X,\mathrm{sec}}` are the terms selected by ``hamiltonian_mode`` (the
subscripts are omitted when ``"direct"`` is selected). The expression uses
frequency units (Hz); the implementation multiplies by :math:`2\pi` for
propagation. The factor :math:`1/2` comes from the resonant component of a
linearly oscillating field,

.. math::

   \cos(\omega t)=\frac{e^{i\omega t}+e^{-i\omega t}}{2}.

The stationary transverse-drive expression assumes the corresponding rotating
component is isolated by the RWA; with a general anisotropic g-tensor this is
an approximation rather than an exact rotating-frame identity.

Hamiltonian Secularization in the Simulation Pipeline
----------------------------------------------------

For a ``DensityTimeSpectra`` calculation, the static Hamiltonian terms are
selected by ``hamiltonian_mode``. Its default value is ``"secular"``:

.. code-block:: python

   from mars.spectra_manager import DensityTimeSpectra

   spectra = DensityTimeSpectra(
       freq=9.8e9,
       sample=sample,
       context=context,
       hamiltonian_mode="secular",
   )

Here, ``sample`` is a ``SolidSample`` and ``context`` supplies the population
and relaxation model. During spectrum evaluation,
``sample.get_hamiltonian_terms_secular()`` is called before resonance finding.
The returned terms are then used for the resonance calculation and passed to
the intensity and density-matrix population calculations. The full set of
eigenstates, when requested, is therefore calculated from the selected
Hamiltonian.

Hamiltonian secularization is applied when the sample terms are obtained.
The RWA is applied to the microwave-driven evolution by
:class:`mars.population.populators.density_population.RWADensityPopulator`.
These are distinct choices: ``hamiltonian_mode="direct"`` selects
``sample.get_hamiltonian_terms()``, but does not switch an RWA populator to a
laboratory-frame propagation method. The static Hamiltonian supplied to that
populator must still satisfy its rotating-frame symmetry requirements.
See :ref:`hamiltonian-secularization` for the Hamiltonian selection mechanism.

Constraints and Limitations
----------------------------

The RWA requires the following structural conditions in the MaRs implementation.

Zeeman operators
~~~~~~~~~~~~~~~~~

The Zeeman operators :math:`G_X`, :math:`G_Y` and :math:`G_Z` contain
both electron and nuclear contributions. For electron g-tensors diagonal in the laboratory (X, Y, Z) frame, they are

.. math::

   G_\alpha = \frac{\mu_B}{h}\sum_e g^{(e)}_{\alpha\alpha}S_{e,\alpha}
   -\frac{\mu_N}{h}\sum_n g_n I_{n,\alpha},
   \qquad \alpha\in\{X,Y,Z\}.

For general g-tensors, the electron contribution also contains off-diagonal
components. The full Zeeman operators need not be proportional to
:math:`S_\alpha^{\mathrm{tot}}` by a common scalar because the electron and
nuclear coefficients differ. The rotating frame nevertheless uses the total
:math:`S_Z^{\mathrm{tot}}` operator. An exact transverse-vector rotation law,
:math:`[S_Z^{\mathrm{tot}},G_X]=iG_Y`, requires compatible transverse
coefficients (for example, equal X and Y coefficients for each particle and no
incompatible off-diagonal terms).

The accuracy of the Hamiltonian secularization and the RWA should be assessed
separately. Two useful checks are:

1. Compare stationary spectra computed with ``hamiltonian_mode="direct"`` and
   ``hamiltonian_mode="secular"``. This tests the effect of Hamiltonian
   secularization on the stationary spectrum. Agreement does not by itself
   validate the RWA for driven time-dependent evolution. Differences can become
   significant when :math:`D`, :math:`E`, or other interaction strengths are
   comparable with the relevant Zeeman energy separations.

   .. code-block:: python

       import torch
       import matplotlib.pyplot as plt
       from mars.spectra_manager import StationarySpectra

       # Define sample, context, freq, temperature
       sample = ...  # SolidSample
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


2. Compare long-time population evolution between a kinetic (rate-equation)
   model and the RWA density-matrix model using consistent initial populations,
   relaxation rates, and driving conditions. Agreement is expected only in a
   regime where coherences have decayed and a rate-equation description is
   valid. A direct test of the RWA requires comparison with a calculation that
   retains the counter-rotating microwave contribution.

Circular and linear polarization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The rotating-frame construction is naturally expressed using circular components. A
linearly polarized field contains two counter-rotating components:

.. math::

   B_1\cos(\omega t)
   = \frac{B_1}{2}e^{i\omega t}+\frac{B_1}{2}e^{-i\omega t}.

The RWA keeps the component rotating in the resonant direction. The omitted term rotates
at approximately :math:`2\omega` in the chosen frame.

Commutation with the static Hamiltonian
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the rotating transformation to leave the static part invariant, the
required conditions are

.. math::

   [F,S_Z^{\mathrm{tot}}]=0,\qquad[G_Z,S_Z^{\mathrm{tot}}]=0.

Thus, neither the field-independent term nor the static Zeeman term acquires
additional time dependence under the rotation. The secular projector enforces
these relations for the selected Hamiltonian terms. Commutation of :math:`F`
with :math:`G_Z` alone is not equivalent to commutation with the total-spin
rotating-frame generator.

Relaxation Superoperator Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The relaxation superoperator must be compatible with the same rotating-frame symmetry.
In a basis of total-spin projection eigenstates, the required secular structure is

.. math::

   R_{ij,kl}\neq 0
   \quad\text{only when}\quad
   m_i-m_j=m_k-m_l.

Here :math:`m_i` is the eigenvalue of the total-spin projection in the
corresponding state (using a basis that diagonalizes the total projection).
This preserves the coherence-order sectors needed by the rotating-frame
construction. The relaxation model must respect the same symmetry; a
Hamiltonian projection alone does not impose it on the relaxation matrix.

Detection operator and the measured signal
------------------------------------------

The measured EPR signal is a linear functional of the laboratory-frame density
matrix :math:`\rho(t)`. In the model, the microwave interaction uses the full
transverse Zeeman operator (including electron and nuclear contributions):

.. math::

   H_1(t)=B_1\cos(\omega t)\,G_X.

Its time derivative gives

.. math::

   I(t)=\operatorname{Tr}\!\left[\rho(t)\frac{dH_1}{dt}\right]
       =-B_1\omega\sin(\omega t)\operatorname{Tr}[\rho(t)G_X].

With the total-spin rotating-frame transformation,

.. math::

   \rho(t)=W^\dagger(t)\tilde{\rho}(t)W(t),\qquad
   W(t)=e^{i\omega S_Z^{\mathrm{tot}}t}.

When the transverse Zeeman operators transform as vector components,

.. math::

   W(t)G_XW^\dagger(t)=G_X\cos(\omega t)-G_Y\sin(\omega t).

This relation is exact for compatible transverse Zeeman coefficients and is
only approximate for general anisotropic g-tensors. In that regime,

.. math::

   I(t)=-B_1\omega\Bigl[
      \sin(\omega t)\cos(\omega t)\operatorname{Tr}(\tilde\rho G_X)
      -\sin^2(\omega t)\operatorname{Tr}(\tilde\rho G_Y)
   \Bigr].

Averaging over a microwave period gives

.. math::

   \langle I(t)\rangle_T
   =\frac{B_1\omega}{2}\operatorname{Tr}[\tilde\rho(t)G_Y].

Thus, the expression uses the full :math:`G_Y`, rather than an electron-only
spin operator. For anisotropic or off-diagonal g-tensors the simple transverse
rotation relation, and hence this detection expression, requires an additional
RWA approximation. The implementation uses the displayed expression without
an orientation-dependent correction factor; this can affect relative
intensities, particularly in powder spectra.

Powder Averaging
----------------

For disordered samples, spectra are averaged over molecular orientations
:math:`(\alpha,\beta,\gamma)` in Euler angle notation. The RWA method inherently
assumes nearly isotropic :math:`g`-tensor, and for powder averaging
we make a further simplification for each orientation: :math:`g_{XX} \approx g_{YY} = g_{\perp}`.
This additional approximation affects the line intensities but not their positions.

For disordered samples, spectra are averaged over molecular orientations
:math:`(\beta, \gamma)` in Euler angle in :math:`zy'z'''` notation. Averaging over
:math:`\alpha` reduces to averaging the initial density matrix: under the RWA,
this dependence can be represented by the unitary rotation
:math:`e^{i\alpha S_Z^{\mathrm{tot}}}` applied to the initial density matrix. Consequently,
the powder average over :math:`\alpha` is equivalent to an average over the
corresponding initial states.

Let consider it:

.. math::

   W_\alpha=e^{iS_Z^{\mathrm{tot}}\alpha}.

Let the reference RWA Hamiltonian be:

.. math::

   H_X = F+B_0G_Z-\nu_{\mathrm{rf}}S_Z^{\mathrm{tot}}+\frac{B_1}{2}G_X,

and let the orientation-dependent Hamiltonian be

.. math::

   H_\alpha = W_\alpha^\dagger H_X W_\alpha.

Because :math:`[F,S_Z^{\mathrm{tot}}]=0`, :math:`[G_Z,S_Z^{\mathrm{tot}}]=0`,
and :math:`S_Z^{\mathrm{tot}}` is unchanged by its own rotation, the only
orientation dependence is the rotation of the transverse operator. The corresponding
density-matrix evolution can therefore be written as

.. math::

   \rho_\alpha (t)
   =W_\alpha^\dagger\,
   \rho_X\!\left(t;\rho_0^{(\alpha)}\right)W_\alpha,

with the rotated initial state

.. math::

   \rho_0^{(\alpha)}
   =W_\alpha\rho_0W_\alpha^\dagger.

The detection operator transforms in the same way, e.g.

.. math::

   D_\alpha=W_\alpha^\dagger G_Y W_\alpha.

Consequently the signal is invariant under moving the orientation dependence from the
Hamiltonian and detection operator into the initial density matrix:

.. math::

   \begin{aligned}
   I_\alpha(t)
   &=\operatorname{Tr}\!\left(D_\alpha\rho_\alpha(t)\right)\\
   &=\operatorname{Tr}\!\left(G_Y\rho_X\!\left(t;\rho_0^{(\alpha)}\right)\right).
   \end{aligned}

The Liouville equation is linear in :math:`\rho`, so averaging over :math:`\alpha` can be
performed before the propagation:

.. math::

   \begin{aligned}
   \overline{I}(t)
   &=\frac{1}{2\pi}\int_0^{2\pi} I_\alpha(t)\,d\alpha\\
   &=\operatorname{Tr}\!\left(G_Y\,\rho_X(t;\overline{\rho}_0)\right),
   \end{aligned}

where

.. math::

   \boxed{\displaystyle
   \overline{\rho}_0
   =\frac{1}{2\pi}\int_0^{2\pi}
      W_\alpha\rho_0W_\alpha^\dagger\,d\alpha.}

The implementation performs a single propagation with an averaged
initial density matrix instead of explicitly sampling the final Euler angle :math:`\alpha`.

The averaging has a simple form in the product basis, where the total
:math:`S_Z^{\mathrm{tot}}` is diagonal. Let
:math:`S_Z^{\mathrm{tot}}|a\rangle=m_a|a\rangle`. Then

.. math::

   \left(\overline{\rho}_0\right)_{ab}
   =\rho_{0,ab}\frac{1}{2\pi}\int_0^{2\pi}e^{i(m_a-m_b)\alpha}\,d\alpha
   =\begin{cases}
      \rho_{0,ab}, & m_a=m_b,\\
      0, & m_a\neq m_b.
     \end{cases}

Therefore matrix elements connecting states with the **same total spin
projection** survive, including coherences within degenerate sectors. The
averaged state is the projection of :math:`\rho_0` onto the commutant of
total :math:`S_Z^{\mathrm{tot}}`:

.. math::

   [\overline{\rho}_0,S_Z^{\mathrm{tot}}]=0.

The implementation transforms the density matrix to the product basis, keeps
only matrix elements connecting states with equal total :math:`S_Z` eigenvalues,
and transforms the result back to the Hamiltonian eigenbasis.

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

For the general case where R = R(rho, t), the equation is solved using adaptive Runge-Kutta methods (via ``torchdiffeq``).
This provides automatic time-step control but is computationally more expensive.

The solver is automatically selected based on the Context:

* **Stationary**: R constant → matrix exponential

* **Time-dependent**: R(t) → adaptive ODE solver by default

Additioanlly, it is possible, to configurate solver for the specific task:

Piecewise matrix exponentiation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use :meth:`mars.population.tr_utils.EvolutionRWASolver.exponential_solver` when :math:`M(t)` is known at the requested
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

:meth:`mars.population.tr_utils.EvolutionRWASolver.stationary_rate_solver` diagonalizes :math:`M` and evaluates the
exponential through its eigenmodes.

Stationary matrix exponential
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`mars.population.tr_utils.EvolutionRWASolver.stationary_rate_solver_expm` evaluates
:math:`e^{Mt}` directly. It avoids the diagonalizability assumption and is therefore the
robust alternative when the eigenvector matrix is singular or poorly conditioned, at the
cost of additional matrix-exponential computation. (see more in kinetic approach computation :ref:`level_based_kinetic_approach`)

Solver configuration
~~~~~~~~~~~~~~~~~~~~~

The solver is passed directly to :class:`RWADensityPopulator`. If ``solver=None``, the
populator chooses :meth:`EvolutionRWASolver.odeint_solver` for a time-dependent relaxation
context and :meth:`EvolutionRWASolver.stationary_rate_solver` otherwise.

For example:

.. code-block:: python

   from mars.population.tr_utils import EvolutionRWASolver
   from mars.population import RWADensityPopulator

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
