.. _propagator-computations:

Propagator-Based Density Matrix Evolution
===========================================

Overview
--------

The :class:`mars.population.populators.density_population.PropagatorDensityPopulator` class computes time-resolved EPR signals by explicitly calculating the full time-evolution propagator :math:`\hat{U}(t, 0)` of the density matrix.
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

To solve this equation, we introduce the propagator :math:`\hat{U}(t, 0)`, which maps the initial state to the state at time :math:`t`:

.. math::

   \vec{\rho}(t) = \hat{U}(t, 0)\vec{\rho}(0)

The propagator satisfies the differential equation:

.. math::

   \frac{d\hat{U}(t, 0)}{dt} = \hat{\mathcal{L}}(t)\hat{U}(t, 0)

with the initial condition :math:`\hat{U}(0, 0) = \hat{I}` (the identity superoperator).

Driving Amplitude
~~~~~~~~~~~~~~~~~~

The oscillating microwave field is applied along the laboratory-frame x-axis:

.. math::

   B_1(t) = b_1 \cos(\omega t)

where :math:`b_1` is the ``b1_field`` parameter of :class:`~mars.population.populators.density_population.PropagatorDensityPopulator` (the full peak amplitude of the oscillating field, not a rotating-frame half-amplitude). Internally this is converted to an angular Rabi coupling scale :math:`2\pi b_1` before being combined with the transformed Zeeman operators :math:`\hat{G}_x, \hat{G}_y` to build the time-dependent part of the Liouvillian.

Floquet Theory for Periodic Hamiltonians
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Assuming a continuous microwave drive, the time-dependent Hamiltonian :math:`H(t)` is periodic with period :math:`T = 2\pi/\omega`, where :math:`\omega` is the microwave frequency (e.g., 9–10 GHz in X-band, yielding :math:`T \approx 0.1` ns). This periodicity enables efficient long-time propagation using Floquet theory.

For any time :math:`t = kT + \tau`, where :math:`k` is an integer and :math:`0 \leq \tau < T`:

.. math::

   \hat{U}(t, 0) = [\hat{U}(T, 0)]^k \hat{U}(\tau, 0)

Thus, the propagator need only be computed numerically over one microwave period :math:`[0, T]`. For longer times, the result is obtained by raising the single-period propagator to the :math:`k`-th power.

Implementation Notes
~~~~~~~~~~~~~~~~~~~~
The populator uses a 4th-order Runge–Kutta method to compute :math:`\hat{U}(T, 0)` and the associated phase-weighted integral in a single loop. This requires the parameter ``n_steps`` (see :meth:`mars.population.populators.density_population.PropagatorDensityPopulator.__init__`).
This parameter is crucial; for systems with fast oscillating terms or strong coupling, ``n_steps`` must be increased to avoid numerical instability.

Signal Detection
~~~~~~~~~~~~~~~~

The observable EPR signal is proportional to the transverse magnetization. In the vectorized formalism, the "detective" vector is built from the (transposed) Zeeman operator :math:`\hat{G}_X`, and the signal at time :math:`t` is obtained by contracting it with the propagated, phase-weighted density vector:

.. math::

   S(t) \propto -\,\mathrm{Re}\Big[\,\mathrm{vec}(\hat{G}_X^T)^{\dagger} \cdot \hat{J}(t)\, \vec{\rho}(0)\,\Big]

where :math:`\hat{J}(t)` is the accumulated phase-weighted integral described below, evaluated at the propagator power corresponding to :math:`t`. Only the real part of this contraction is physically observable and is returned as the signal.

The continuous-time integrated signal that this discretized construction approximates is:

.. math::

   I(t) = \int_0^t \text{Tr}[\hat{G}_X \hat{\rho}(\tau)] \sin(\omega\tau) d\tau

Computational Implementation
----------------------------

Efficient Propagator Calculation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rather than storing the full propagator :math:`\hat{U}(\tau, 0)` for all :math:`\tau \in [0, T]`, the integration over one period computes only two quantities:

1. **Full-period propagator**: :math:`\hat{U}(T, 0)`
2. **Phase-weighted integral**:

   .. math::

      \hat{J} = \int_0^T \hat{U}(\tau, 0) \sin(\omega\tau) d\tau

These matrices are sufficient to reconstruct the detected signal at any time :math:`t` using the Floquet expansion. When a finite ``measurement_time`` is supplied (rather than the default single-period detection), this integral is further corrected to account for the sum of contributions over all :math:`M` microwave periods contained in the measurement window.

Matrix Power via Diagonalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To compute :math:`[\hat{U}(T, 0)]^k` efficiently, the single-period propagator is diagonalized once (via a general, non-symmetric eigendecomposition, since :math:`\hat{U}(T,0)` is not Hermitian):

.. math::

   \hat{U}(T, 0) = S \Lambda S^{-1}

Then:

.. math::

   [\hat{U}(T, 0)]^k = S \Lambda^k S^{-1}

where :math:`\Lambda^k` is a diagonal matrix obtained by raising the eigenvalues to the :math:`k`-th power.
This avoids repeated matrix multiplications, scaling with the number of *distinct* requested powers rather than with :math:`k` itself.

.. note::

   If :math:`\hat{U}(T, 0)` is (near-)defective, its eigenbasis can become ill-conditioned, which is flagged internally with a warning. In that regime, prefer the slower but diagonalization-free power iteration provided as a fallback (see below).

Time Discretization
~~~~~~~~~~~~~~~~~~~

Since the microwave period :math:`T \approx 0.1` ns is much shorter than typical detection timescales (hundreds of nanoseconds or longer), each requested output time is mapped to the smallest integer number of microwave periods that contains it, i.e. :math:`k = \lceil t / T \rceil` (rounded up to the next full period, not to the nearest one). This introduces negligible error for envelope detection while greatly simplifying calculations.

Relaxation Parameter Constraints
---------------------------------

For the propagator method, all relaxation parameters in the Context must be time-independent.
The Floquet approach relies on the strict periodicity of the Liouvillian. If the relaxation superoperator :math:`\hat{R}` depends on time (e.g., due to rapid temperature jumps or time-dependent fields), the propagator loses its periodic structure and cannot be computed via :math:`[\hat{U}(T, 0)]^k`.

If relaxation parameters vary with time, use the kinetic approach or RWA with adaptive ODE integration instead.

Powder Averaging
----------------

For disordered samples, the spectrum must be averaged over all molecular orientations :math:`(\alpha,\beta,\gamma)`.
Unlike the RWA method, the propagator approach makes no simplifying assumptions about the :math:`g`-tensor, zero‑field splitting, or relaxation.
Consequently, the signal depends on all three Euler angles, and the :math:`\gamma` integration must be performed numerically.

Full numerical integration over :math:`\gamma \in [0,2\pi]` is therefore the
default, implemented as a Riemann sum over polarizations :math:`\hat{G}_\perp = \hat{G}_x\cos\phi + \hat{G}_y\sin\phi`. The number of :math:`\gamma` points can be controlled via the
``angle_average_steps`` parameter of
:class:`mars.population.populators.density_population.PropagatorDensityPopulator`:

.. code-block:: python

   from mars.population.populators.density_population import PropagatorDensityPopulator

   populator_prop = PropagatorDensityPopulator(
       angle_average_steps=4,         # Number of γ‑integration points
       context=context,
       measurement_time=None,         # Default: one microwave period
       init_temperature=300.0,
   )

Alternatively, ``angle_average_steps`` can be set globally via
:class:`mars.spectra_manager.spectra_manager.ComputationalDetails`.

For ordered (single-crystal) samples, set ``disordered=False``; in that case only the fixed :math:`\hat{G}_x` polarization is used and no :math:`\gamma` averaging is performed.

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

* It operates on the full Liouville space propagator (dimension :math:`N^2 \times N^2` versus :math:`N^2` for density vector evolution).
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

For simpler systems where the RWA is valid, use :class:`mars.population.populators.density_population.RWADensityPopulator` for significantly faster computation.