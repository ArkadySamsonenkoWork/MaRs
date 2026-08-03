Spectra Creators
================

The ``spectra_creators`` module provides a unified interface for simulating various types of Electron Paramagnetic Resonance (EPR) spectra using the MaRs library. These creators encapsulate the full computational pipeline-from Hamiltonian diagonalization and transition intensity calculation to orientation averaging, line broadening, and spectral integration-while supporting both stationary (continuous-wave) and time-resolved experiments.

All spectra creators inherit from the abstract base class :class:`mars.spectra_manager.spectra_manager.BaseResSpectra`, which defines the core architecture and parameter handling. The specific implementations differ primarily in how they treat spin dynamics: either through population kinetics, density-matrix evolution under the Lindblad master equation, or field/frequency domain assumptions.

This section documents the main spectra creator classes available in MaRs:

.. toctree::
   :maxdepth: 2

   stationary
   coupled
   density
   freq_domain
   architecture