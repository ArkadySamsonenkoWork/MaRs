.. _intensity_calculation:

Transition Intensity Calculation
================================

Overview
--------

In electron paramagnetic resonance (EPR), the observable signal for a transition between two spin eigenstates :math:`|a\rangle` and :math:`|b\rangle` generally depends on:

1. The *magnetic transition dipole moment*, a matrix element of magnitization vector.
2. The *population difference* between the initial and final states, which can be determined from ф thermal equilibrium conditions or defined via :class:`mars.population.contexts.Context`.

This two-factor decomposition is valid for *incoherent, rate-based descriptions* of EPR (e.g., continuous-wave or time-resolved experiments with relaxation where quantum coherences are neglected).  
However, it does not hold in coherent treatments based on the density matrix, where populations and coherences evolve jointly under the Liouville–von Neumann equation. 

For stationary EPR under *thermal equilibrium*, the detected signal is proportional to the absorbed microwave power, which within Fermi golden rule is given by:

.. math::

   I_{ab} \propto (p_a - p_b) \cdot \left| \mathbf{n}_1^\top \boldsymbol{\mu}_{ab} \right|^2,

where:

- :math:`p_a = e^{-E_a / k_B T}/Z` and :math:`p_b = e^{-E_b / k_B T}/Z` are Boltzmann populations,
- :math:`\boldsymbol{\mu}_{ab} = \langle b | \hat{\boldsymbol{\mu}} | a \rangle` is the magnetic transition dipole moment vector,
- :math:`\mathbf{n}_1` is the unit polarization vector of the oscillating magnetic field :math:`\mathbf{B}_1`,
- :math:`Z` is the partition function.

For density matrix-based time-dependent methods, the signal is computed directly as :math:`\mathrm{Tr}(\hat{G}_{\perp} \hat{\rho}(t))`,
where :math:`\hat{G}_{\perp}` is the detected transverse spin component (e.g., :math:`\hat{G}_x`, :math:`\hat{G}_y`, or circular combinations depending on detection method and computational approach).

Intensity Calculator Classes
-----------------------------

MaRs provides several intensity calculator implementations through :class:`mars.spectra_manager.spectra_manager.BaseResIntensityCalculator` and its subclasses:

**StationaryIntensityCalculator**
   For continuous-wave (CW) EPR experiments 	with thermal equilibrium populations. Uses Boltzmann distributions or context-defined populations.
   Reference: :class:`mars.spectra_manager.spectra_manager.StationaryIntensityCalculator`

**TimeIntensityCalculator**
   For time-resolved EPR based on level population relaxation. Populations evolve according to kinetic equations with configurable relaxation rates.
   Reference: :class:`mars.spectra_manager.spectra_manager.TimeIntensityCalculator`

**TimeDensityCalculator**
   For time-resolved EPR using density matrix formalism. In this approach the transition matrix elements and level populations are not separable.
   Reference: :class:`mars.spectra_manager.spectra_manager.TimeDensityCalculator`

**WaveIntensityCalculator**
   For EPR experiments with non-standard excitation geometries and polarizations (circular, linear, unpolarized). Accounts for arbitrary orientation of radiation relative to the magnetic field.
   Reference: :class:`mars.spectra_manager.wave_calculator.WaveIntensityCalculator`


Computational Workflow
----------------------

*StationaryIntensityCalculator*, *TimeIntensityCalculator*, *WaveIntensityCalculator* follow a common pipeline:

1. **Receive transition parameters**
   
   - Zeeman operator components **G_x**, **G_y**, **G_z**
   - Eigenvectors of lower (:math:`|\psi_\mathrm{down}\rangle`) and upper (:math:`|\psi_\mathrm{up}\rangle`) states
   - Energy level indices of transition levels and eigenvalues of spin Hamiltonian
   - Resonance manifold values (fields or frequencies)

2. **Compute magnetization term**
   
   Calculate the squared matrix elements of Zeeman operators between transition states (see :ref:`magnetization_computation`).

3. **Evaluate population contribution**
   
   Determine the population difference (p_down - p_up) or time-dependent populations (see :ref:`population_calculation`).

4. **Combine components**
   
   Multiply magnetization and population terms to obtain final intensity.

Contents
--------

.. toctree::
   :maxdepth: 2

   general_epr_transitions
   population_calculation