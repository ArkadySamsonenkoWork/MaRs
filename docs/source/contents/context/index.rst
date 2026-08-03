.. _context-general-information:

Relaxation Context
==================

The :class:`mars.population.contexts.Context` class provides a comprehensive framework for defining relaxation mechanisms and initial states in time-resolved EPR spectroscopy. 
It encapsulates the physical model of how spin systems evolve after initial excitation.

Overview
--------

.. image:: /_static/context/context_structure.png
   :width: 70%
   :alt: The general strucrture of MaRs contexts
   :align: center



A Context object can specify:

1. The basis in which relaxation parameters, density matrix and initial populations are defined
2. Initial populations or density matrix
3. Transition probabilities (thermal, driven, and loss processes)
4. Dephasing rates
5. Custom relaxation superoperators
6. Arbitary Bloch-Redfield and Lindblad operators and the spectral density functions.

The Context class automatically handles basis transformations, and proper composition of kinetic matrices or relaxation superoperators.

Key Features
------------

- **Flexible Basis Definition**: Support for multiple predefined bases (eigen, zfs, multiplet, product)
- **Multiple Relaxation Paradigm**: Compatible with both population-based and density matrix-based dynamics
- **Lindblad Formalism**: Ensures physical validity through proper jump operators
- **Relaxation Channels**: First‑principles Redfield and arbitrary Lindblad channels via direct operator‑level coupling
- **Context Algebra**: Supports addition and multiplication operations for constructing complex relaxation scenarios

Contents
--------

.. toctree::
   :maxdepth: 2

   relaxation_parameters
   superoperator_creation
   basis_transformation
   complex_context
   relaxation_channels

Quick Example
-------------

.. code-block:: python

   import torch
   from mars import spin_model. population
   
   # Create a triplet spin system
   g_tensor = spin_model.Interaction(2.003, dtype=torch.float64)
   zfs = spin_model.DEInteraction([600e6, -80e6], dtype=torch.float64)
   
   base_system = spin_model.SpinSystem(
       electrons=[1.0],
       g_tensors=[g_tensor],
       electron_electron=[(0, 0, zfs)]
   )
   
   sample = spin_model.MultiOrientedSample(
       base_spin_system=base_system,
       ham_strain=2e7,
       gauss=0.0008,
       lorentz=0.0008
   )
   
   # Define initial populations in ZFS basis
   initial_pops = [0.6, 0.1, 0.3]
   
   # Define loss rates
   decay_rates = torch.tensor([80.0, 40.0, 60.0])  # s^-1
   
   # Create context
   context = population.Context(
       sample=sample,
       basis="zfs",
       init_populations=initial_pops,
       decay_rates=decay_rates
   )

.. note::

   For more sophisticated models you can supply **relaxation channels**
   (Redfield or Lindblad) that define the system–bath coupling at the operator level.
   See :ref:`relaxation_channels` for details.

See Also
--------

- :ref:`relaxation_parameters` - Detailed description of transition rates
- :ref:`superoperator_creation` - How MaRs constructs Lindblad superoperators
- :ref:`basis_transformation` - Transformation between different bases
- :ref:`complex_context` - Combining contexts using algebra operations
- :ref:`relaxation_channels` - Combining contexts using algebra operations

