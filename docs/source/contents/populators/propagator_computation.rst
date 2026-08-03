.. _propagator-computations:

Propagator-Based Density Matrix Evolution
===========================================

Overview
--------

The :class:`mars.population.density_population.PropagatorDensityPopulator` class computes time-resolved EPR signals by explicitly calculating the full time-evolution propagator :math:`\hat{U}(t, 0)` of the density matrix.
This method imposes no approximations on the g-tensor, zero-field splitting, or relaxation superoperator, making it the most general approach for time-dependent density matrix evolution.

Theory
------

Evolution Propagator
~~~~~~~~~~~~~~~~~~~~
The core algorithm is based on the approach introduced in [Appl Magn Reson 55, 1553–1567 (2024)].

The time evolution of the spin density matrix :math:`\hat{\rho}(t)` is governed by the Liouville-von Neumann equation. In the vectorized Liouville space representation, this is written as:

.. math::

   \frac{d\vec{\rho}(t)}{dt} = \hat{\mathcal{L}}(t)\vec{\rho}(t)

where :math:`\vec{\rho}` is the vectorized density matrix and :math:`\hat{\mathcal{L}}(t) = -i\hat{\mathcal{H}}(t) + \hat{R}` is the Liouvillian superoperator. Here, :math:`\hat{\mathcal{H}}` represents the superoperator form of the commutator with the Hamiltonian, defined as:

.. math::

   \hat{\mathcal{H}} = H \otimes I - I \otimes H^T

where :math:`H` is the spin Hamiltonian in Hilbert space (in frequency units, Hz) and :math:`I` is the identity matrix. The factor of :math:`2\pi` is implicitly included in the definition of :math:`H` within MaRs to match angular frequency conventions in the exponent.

To solve this equation, we introduce the **propagator** :math:`\hat{U}(t, 0)`, which maps the initial state to the state at time :math:`t`:

.. math::

   \vec{\rho}(t) = \hat{U}(t, 0)\vec{\rho}(0)

The propagator satisfies the differential equation:

.. math::

   \frac{d\hat{U}(t, 0)}{dt} = \hat{\mathcal{L}}(t)\hat{U}(t, 0)

with the initial condition :math:`\hat{U}(0, 0) = \hat{I}` (the identity superoperator).

Floquet Theory for Periodic Hamiltonians
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Assuming a continuous microwave drive, the time-dependent Hamiltonian :math:`H(t)` is periodic with period :math:`T = 2\pi/\omega`, where :math:`\omega` is the microwave frequency (e.g., 9–10 GHz in X-band, yielding :math:`T \approx 0.1` ns). This periodicity enables efficient long-time propagation using Floquet theory.

For any time :math:`t = kT + \tau`, where :math:`k` is an integer and :math:`0 \leq \tau < T`:

.. math::

   \hat{U}(t, 0) = [\hat{U}(T, 0)]^k \hat{U}(\tau, 0)

Thus, the propagator need only be computed numerically over one microwave period :math:`[0, T]`. For longer times, the result is obtained by raising the single-period propagator to the :math:`k`-th power.

Implementation Notes
~~~~~~~~~~~~~~~~~~~~
The populator uses a Runge–Kutta method to **compute** :math:`\hat{U}(T, 0)`. This requires the parameter ``n_steps`` (see :meth:`mars.population.density_population.PropagatorDensityPopulator.__init__`).
This parameter is crucial; for systems with fast oscillating terms or strong coupling, ``n_steps`` must be increased to avoid numerical instability.

Signal Detection
~~~~~~~~~~~~~~~~

The observable EPR signal is proportional to the transverse magnetization. In the vectorized formalism, this is calculated as:

.. math::

   S(t) \propto \text{Tr}[\hat{S}_x \hat{\rho}(t)] = \mathbf{s}_x^\dagger \hat{U}(t,0) \vec{\rho}(0)

where :math:`\hat{S}_x` is the x-component of the total spin angular momentum operator.

For phase-sensitive detection locked to the microwave frequency, the integrated signal :math:`I(t)` is:

.. math::

   I(t) = \int_0^t \text{Tr}[\hat{S}_x \hat{\rho}(\tau)] \sin(\omega\tau) d\tau

Computational Implementation
----------------------------

Efficient Propagator Calculation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rather than storing the full propagator :math:`\hat{U}(\tau, 0)` for all :math:`\tau \in [0, T]`, the integration over one period computes only two quantities:

1. **Full-period propagator**: :math:`\hat{U}(T, 0)`
2. **Phase-weighted integral**:

   .. math::

      \hat{J} = \int_0^T \hat{U}(\tau, 0) \sin(\omega\tau) d\tau

These matrices are sufficient to reconstruct the detected signal at any time :math:`t` using the Floquet expansion.

Matrix Power via Diagonalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To compute :math:`[\hat{U}(T, 0)]^k` efficiently, the single-period propagator is diagonalized once:

.. math::

   \hat{U}(T, 0) = S \Lambda S^{-1}

Then:

.. math::

   [\hat{U}(T, 0)]^k = S \Lambda^k S^{-1}

where :math:`\Lambda^k` is a diagonal matrix obtained by raising the eigenvalues to the :math:`k`-th power.
This avoids repeated matrix multiplications, scaling linearly with :math:`k` rather than quadratically.

Time Discretization
~~~~~~~~~~~~~~~~~~~

Since the microwave period :math:`T \approx 0.1` ns is much shorter than typical detection timescales (hundreds of nanoseconds or longer), output time points are rounded to the nearest integer multiple of :math:`T`.
This introduces negligible error for envelope detection while greatly simplifying calculations. Note that this method does not resolve signal variations *within* a single microwave cycle.

Relaxation Parameter Constraints
---------------------------------

For the propagator method, all relaxation parameters in the Context **must be time-independent**.
The Floquet approach relies on the strict periodicity of the Liouvillian. If the relaxation superoperator :math:`\hat{R}` depends on time (e.g., due to rapid temperature jumps or time-dependent fields), the propagator loses its periodic structure and cannot be computed via :math:`[\hat{U}(T, 0)]^k`.

If relaxation parameters vary with time, use the kinetic approach or RWA with adaptive ODE integration instead.

Powder Averaging
----------------

For disordered samples, the spectrum must be averaged over all molecular orientations :math:`(\alpha,\beta,\gamma)`.
Unlike the RWA method, the propagator approach makes no simplifying assumptions about the :math:`g`-tensor, zero‑field splitting, or relaxation.
Consequently, the signal depends on all three Euler angles, and the :math:`\gamma` integration must be performed numerically.

Full numerical integration over :math:`\gamma \in [0,2\pi]` is therefore the
default. The number of :math:`\gamma` points can be controlled via the
``angle_average_steps`` parameter of
:class:`mars.population.density_population.PropagatorDensityPopulator`:

.. code-block:: python

   populator_prop = population.PropagatorDensityPopulator(
       angle_average_steps=4,         # Number of γ‑integration points
       context=context,
       measurement_time=None,         # Default: one microwave period
       init_temperature=300.0,
   )

Alternatively, ``angle_average_steps`` can be set globally via
:class:`mars.spectra_manager.spectra_manager.ComputationalDetails`.

Advantages
----------

The propagator method:

* Supports arbitrary g-tensor anisotropy without secular approximations.
* Handles any zero-field splitting tensor orientation.
* Allows general relaxation superoperators (including coherence-population coupling).
* Provides numerically exact evolution within the limits of the time-step discretization.

Computational Cost
------------------

This method is more demanding than the RWA approach because:

* It operates on the full Liouville space propagator (dimension :math:`N^2 \times N^2` versus versus :math:`N^2` for density vector evolution).
* It requires high-resolution integration over the fast microwave period :math:`T`.

Applicability
-------------

Use the propagator method when:

* g-tensor anisotropy is significant (transition metals, high-field EPR).
* Zero-field splitting has arbitrary orientation relative to the g-tensor.
* Non-secular relaxation terms are important.
* Coherence-population coupling cannot be neglected.
* RWA assumptions (slowly varying envelope) are violated.

The propagator method is essential for simulating:

* Single-molecule magnets.
* High-spin metal complexes.
* Strongly coupled radical pairs with large anisotropic interactions.

For simpler systems where the RWA is valid, use :class:`mars.population.stationary.StationaryPopulator` for significantly faster computation.