Spectra Creators
================

The ``spectra_creators`` module provides a unified interface for simulating various types of Electron Paramagnetic Resonance (EPR) spectra using the MaRs library. These creators encapsulate the full computational pipeline-from Hamiltonian diagonalization and transition intensity calculation to orientation averaging, line broadening, and spectral integration-while supporting both stationary (continuous-wave) and time-resolved experiments.

Resonance-based creators inherit from :class:`mars.spectra_manager.spectra_manager.BaseResSpectra`. Direct, fixed-field calculations use :class:`mars.spectra_manager.spectra_manager_direct.BaseDirectSpectra` from :mod:`mars.spectra_manager.spectra_manager_direct`, which evaluates the Hamiltonian at user-defined field points without resonance-line searching. Both derive from :class:`mars.spectra_manager.spectra_manager.BaseSpectra`. The specific implementations differ primarily in how they treat spin dynamics: either through population kinetics, density-matrix evolution under the Lindblad master equation, or field/frequency domain assumptions.

This section documents the main spectra creator classes available in MaRs:

.. toctree::
   :maxdepth: 2

   stationary
   coupled
   density
   freq_domain
   direct
   architecture