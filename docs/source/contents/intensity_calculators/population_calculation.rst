.. _population_calculation:

Population Models in MaRs
=========================

Overview
--------

MaRs supports three physically distinct population models for simulating EPR spectra:

1. **Thermal equilibrium** - populations follow the Boltzmann distribution.
2. **Kinetic non-equilibrium** - populations evolve according to rate (kinetic) equations.
3. **Density-based formalism** - full time-dependent dynamics governed by the density matrix formalism.

These models are implemented through dedicated *populator* classes.

Factorization vs. Non-Factorization
-----------------------------------

- **Factorizing calculators** (:class:`mars.spectra_manager.spectra_manager.StationaryIntensityCalculator`, :class:`mars.spectra_manager.spectra_manager.TimeIntensityCalculator`, :class:`mars.spectra_manager.wave_calculator.WaveIntensityCalculator`) assume the spectral intensity for a transition :math:`i \leftrightarrow j` can be written as:

  .. math::
     I_{ij} = \underbrace{(p_j - p_i)}_{\text{population difference}} \times \underbrace{|M_{ij}|^2}_{\text{transition matrix element}}.

  This approach is valid when coherence between states can be neglected (e.g., in incoherent or high-temperature limits).

- **Non-factorizing calculators** (:class:`mars.spectra_manager.spectra_manager.TimeDensityCalculator`) compute the observable signal directly from the full density matrix without separating populations and coherences:

  .. math::
     I(t) \propto \mathrm{Tr}\!\left( \hat{G}_\perp \, \hat{\rho}(t) \right),

  where :math:`\hat{G}_\perp` is the transverse component of the spin operator (e.g., :math:`\hat{G}_x` or :math:`\hat{G}_y`), and :math:`\hat{\rho}(t)` is the time-evolved density matrix.

Thermal Equilibrium
-------------------

At temperature :math:`T`, the equilibrium population of energy level :math:`k` is given by the Boltzmann distribution:

.. math::
   p_k = \frac{e^{-E_k / k_B T}}{Z}, \quad Z = \sum_\ell e^{-E_\ell / k_B T},

where :math:`E_k` is the eigenenergy of level :math:`k`, :math:`k_B` is Boltzmann’s constant, and :math:`Z` is the partition function.

This model is used by:
- :class:`mars.population.populators.stationary.StationaryPopulator`
- :class:`mars.spectra_manager.spectra_manager.StationaryIntensityCalculator`

It serves as the default for continuous-wave (CW) EPR simulations unless overridden by a custom initial state in a :class:`mars.population.contexts.Context` (see details in :ref:`context-general-information`).

Time-Dependent Kinetics (Rate Equation Approach)
------------------------------------------------

When spin relaxation is treated classically (i.e., ignoring quantum coherences), the time evolution of level populations :math:`p_i(t)` follows a system of linear kinetic (rate) equations:

.. math::
   \frac{dp_i}{dt} = \sum_{j \ne i} \left( k_{ji} p_j - k_{ij} p_i \right) - o_i p_i,

where:
- :math:`k_{ij}` is the probabilities of transition from level :math:`i` to :math:`j`,
- :math:`o_i` represents irreversible loss (e.g., phosphorescence from triplet sublevels)


Density Matrix Formalism
-----------------------------------------------------

For systems where quantum coherences matter, MaRs solves the Liouville–von Neumann equation for the density matrix :math:`\hat{\rho}(t)`:

.. math::
   \frac{d\hat{\rho}}{dt} = -\frac{i}{\hbar} [\hat{H}, \hat{\rho}] + \hat{\mathcal{R}}[\hat{\rho}],

where:
- :math:`\hat{H}` is the spin Hamiltonian,
- :math:`\hat{\mathcal{R}}` is the relaxation superoperator.

In MaRs, :math:`\hat{\mathcal{R}}` can include:

The observable EPR signal is then computed as:

.. math::

   I(t) \propto \mathrm{Tr}\!\left( \hat{G}_\perp \, \hat{\rho}(t) \right)

Two computational strategies are supported:

1. **Rotating Wave Approximation (RWA)**  
   Assumes isotropic :math:`g`-tensor and commutation :math:`[\hat{F}, \hat{G}_z] = 0`. The Hamiltonian is transformed into a rotating frame, yielding an effective time-independent problem. Implemented in:
   - :class:`mars.population.populators.density_population.RWADensityPopulator`

2. **Full Propagator Method**  
   Solves the periodic time-dependent problem exactly over one microwave cycle and constructs the long-time evolution via Floquet-like theory.Implemented in:
   - :class:`mars.population.populators.density_population.PropagatorDensityPopulator`

Summary of Populators
----------------------

**StationaryPopulator**
   - Computes populations at specified temperature. Uses thermal equalibrium or Context-defined population
   - Default for CW-EPR simulations

**LevelBasedPopulator**
   - Handles time-dependent populations with kinetic relaxation

**RWADensityPopulator**
   - Full density matrix computation under rotating wave approximation

**PropagatorDensityPopulator**
   - Full density matrix computation under computation of evolution superoperator

**Custom populators**  
Users may implement new population dynamics by subclassing:
- :class:`mars.population.populators.core.BasePopulator` (for stationary cases)
- :class:`mars.population.populators.core.BaseTimeDepPopulator` (for time-dependent cases)

For more details on polarized spectra, time-resolved modeling, and context-based relaxation specification, see the sections 

