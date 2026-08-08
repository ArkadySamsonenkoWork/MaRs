.. MaRs documentation master file, created by
   sphinx-quickstart on Sun Jan 11 12:18:45 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to MaRs!
=================
**MaRs** is a Python library for simulating Electron Paramagnetic Resonance (EPR) spectra, optimizing spin Hamiltonian and kinetic parameters against experimental data,
and computing both continuous-wave (CW) and time-resolved EPR signals. Built on PyTorch, it supports efficient batched computations on CPU and GPU with flexible numerical precision.

Key features include:

- Construction of multi-spin systems with electrons and nuclei
- Comprehensive interaction models (Zeeman, ZFS, hyperfine, exchange, dipolar)
- Powder and single-crystal sample simulations with orientation averaging
- Population kinetics and full density matrix relaxation formalisms
- Flexible relaxation parameters management for complex experimental scenarios
- Automated parameter optimization via Optuna and Nevergrad
- Time-resolved EPR modeling with multiple relaxation approaches


.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   Installation <installation>
   Quick Start <quickstart>


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   contents/base_spin_system
   contents/sample
   contents/spectrum_constraction
   contents/fitting_tutorial
   contents/interactions/index
   contents/populators/index
   contents/context/index
   contents/spectra_creators/index
   contents/intensity_calculators/index
   contents/meshers/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/mars

.. toctree::
   :maxdepth: 2
   :caption: Tutorial Examples

   examples/example_1
   examples/example_2
   examples/example_3
   examples/example_4
   examples/example_5
   examples/example_6
   examples/example_7
   examples/example_8
   examples/example_9
   examples/example_article_1
   examples/example_article_2



.. toctree::
   :maxdepth: 1
   :caption: Project Info

    GitHub Repository <https://github.com/ArkadySamsonenkoWork/MaRs.git>