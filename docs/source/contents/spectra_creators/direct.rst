.. _direct-spectra:

Direct Fixed-Field Spectra
==========================

The :mod:`mars.spectra_manager.spectra_manager_direct` module provides
:class:`mars.spectra_manager.spectra_manager_direct.BaseDirectSpectra` for
evaluating EPR signals at user-defined magnetic field points without
resonance-line searching. It derives from
:class:`mars.spectra_manager.spectra_manager.BaseSpectra`.

Computational Workflow
----------------------

1. Diagonalize the selected Hamiltonian at each supplied field point.
2. Compute population or density-matrix evolution through
   :class:`mars.spectra_manager.spectra_manager_direct.BaseDirectIntensityCalculator`.
3. Average the resulting signal over powder or crystal orientations.

The density-matrix populator is selected with ``populator="rwa"`` (default)
or ``populator="propagator"``. See :ref:`rotating-wave-approximation` and
:ref:`propagator-computations` for their assumptions and configuration.

Hamiltonian Selection and Broadening
------------------------------------

``hamiltonian_mode="secular"`` (default) selects secular Hamiltonian terms;
``hamiltonian_mode="direct"`` selects the full static Hamiltonian. This
choice is independent of fixed-field evaluation and the populator method.
The RWA still requires its rotating-frame symmetry conditions.

Direct calculations skip broadening parameters and spectral-line integration.
The field grid must therefore resolve the computed signal itself.

Usage
------

Given a sample, relaxation context, and tensors ``fields`` (tesla) and
``time`` (seconds):

.. code-block:: python

   from mars.spectra_manager.spectra_manager_direct import BaseDirectSpectra

   creator = BaseDirectSpectra(
       freq=9.8e9,
       sample=sample,
       context=context,
       populator="propagator",
       hamiltonian_mode="direct",
   )
   signal = creator(sample, fields, time)
