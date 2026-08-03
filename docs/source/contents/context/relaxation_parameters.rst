.. _relaxation_parameters:

Relaxation Parameters and Initial States
========================================

The Context class in MaRs supports four types of relaxation parameters that define how spin populations and coherences evolve over time. All rate constants are expressed in **s⁻¹** (inverse seconds).

Initial States
--------------

Two mutually exclusive ways exist to specify the initial quantum state of a spin system:

1. **init_populations** (Initial populations): A real-valued vector of length N specifying the probability of occupying each energy level in the working basis.
This corresponds to a diagonal density matrix and assumes no initial coherences.

2. **init_desnity** (Initial density matrix): A full complex-valued N×N density matrix that includes both populations (diagonal) and coherences (off-diagonal).

If init_density is required for a computation but not provided, it is automatically constructed as a diagonal matrix from init_populations (if available).

Conversely, if init_populations is required but not explicitly given, the system first transforms the provided init_density into the appropriate basis using the ansformation rules.
After this transformation, the real part of the diagonal elements of the resulting density matrix is used as the initial populations.

If both of these parameters are missing, then the equilibrium initial state at the given temperature will be used.

Relaxation Overview
-------------------

MaRs defines relaxation through four distinct mechanisms:

1. **decay_rates** (Outgoing Rates)- Population loss rates
2. **thermal_rates** (Thermal rates) - Spontaneous transition rates
3. **driven_rates** (Driven rates) - Stimulated transition rates
4. **dephasing** (Dephasing) - Pure dephasing rates used only for density-based relaxation

Each mechanism serves a specific physical purpose and transforms differently under basis changes.

In addition to these rate‑matrix descriptions, MaRs supports first‑principles relaxation models via **Bloch‑Redfield** and **Lindblad relaxation channels**.
These channels define the system‑bath coupling at the operator level and, for Redfield theory, include an explicit spectral density :math:`J(\omega)`.
They are specified using the ``relaxation_coupling_channels`` argument of :class:`~mars.population.contexts.Context` and are thoroughly described in :ref:`relaxation_channels`.


Decay Rates (decay_rates)
------------------------------

.. image:: /_static/context/decay_rates.png
   :width: 70%
   :alt: Out rates diagram
   :align: center

**Physical Meaning**

Out rates describe irreversible population loss from energy levels.

**Mathematical Form**

A diagonal vector where element :math:`o_i` represents the loss rate from level :math:`|i\rangle`.


**Example: Triplet State Phosphorescence**

.. code-block:: python

   import torch
   from mars import population
   
   # Define different phosphorescence rates for each sublevel
   decay_rates = torch.tensor([120.0, 50.0, 80.0])  # s^-1
   
   # In XYZ basis: [TX, TY, TZ]
   # TX: 120 s^-1 → lifetime ~8.3 ms
   # TY: 50 s^-1  → lifetime ~20 ms  
   # TZ: 80 s^-1  → lifetime ~12.5 ms
   
   context = population.Context(
       sample=triplet_sample,
       basis="xyz",
       init_populations=[0.4, 0.2, 0.4],
       decay_rates=decay_rates
   )

**Transformation Rule**
For the most cases the parameters are set in some specific basis. If we transformation matrix from coordinates in old basis to coordinates in a new basis as :math:`U`,
then under basis transformation with matrix :math:`U`:

.. math::

   \mathbf{o}' = |U|^2 \cdot \mathbf{o}

where :math:`|U|^2` denotes element-wise squaring of the transformation matrix.

Thermal rates (thermal_rates)
--------------------------------

.. image:: /_static/context/thermal_rates.png
   :width: 70%
   :alt: Thermal rates diagram
   :align: center

**Physical Meaning**

Thermal rates represent spontaneous transitions between energy levels that obey detailed balance.

**Mathematical Form**

A matrix :math:`W` where element :math:`w_{ij}` is the transition rate from :math:`|j\rangle` to :math:`|i\rangle`.

Diagonal elements are zero (no self-transitions).

**Example: Triplet Spin-Lattice Relaxation**

.. code-block:: python

   import torch
   from mars import population
   
   # Define transition matrix in eigen basis
   # Strong relaxation between adjacent levels
   # Weak direct relaxation between extreme levels
   thermal_rates = torch.tensor([
       [0.0,    800.0,   50.0],   # to level 0
       [800.0,  0.0,     800.0],  # to level 1
       [50.0,   800.0,   0.0]     # to level 2
   ])  # s^-1
   
   # Adjacent transitions: ~1.25 ms equilibration time
   # Direct 0↔2: ~20 ms equilibration time
   
   context = population.Context(
       sample=triplet_sample,
       basis="eigen",
       init_populations=[0.35, 0.35, 0.30],
       thermal_rates=thermal_rates,
       temperature=80.0
   )

**Detailed Balance Enforcement** (see also :ref:`detailed_balance`)

MaRs automatically enforces detailed balance at temperature :math:`T`:

.. math::

   w_{ij}^* = \frac{w_{ij} + w_{ji}}{1 + e^{(E_i - E_j)/k_B T}}

.. math::

   w_{ji}^* = \frac{w_{ij} + w_{ji}}{1 + e^{(E_j - E_i)/k_B T}}

where :math:`w_{ij}^*` are the modified rates ensuring :math:`\frac{w_{ij}^*}{w_{ji}^*} = e^{-(E_i - E_j)/k_B T}`.

We additionally note, that in MaRs notation, the matrix element :math:`w_{ij}` corresponds to the physical transition rate :math:`w_{j \to i}`,
and after thermal correction it satisfies the detailed balance condition:

.. math::

   \frac{w_{j \to i}}{w_{i \to j}} = e^{-(E_i - E_j)/k_B T}.

**Transformation Rule**

Under basis transformation:

.. math::

   W' = |U|^2 \cdot W \cdot (|U|^2)^T

Driven Probabilities (driven_rates)
------------------------------------

.. image:: /_static/context/driven_rates.png
   :width: 70%
   :alt: Driven rates diagram
   :align: center

**Physical Meaning**

Driven rates describe stimulated transitions that do NOT obey detailed balance.

**Mathematical Form**

A matrix :math:`D` where :math:`d_{ij}` is the stimulated rate from :math:`|j\rangle` to :math:`|i\rangle`. This matrix is not modified by detailed balance.

**Example: Selective Microwave Excitation**

.. code-block:: python

   import torch
   from mars import population
   
   # Simulate selective excitation of specific transition
   # Strong driving of 0→1 transition
   # Moderate driving of 1→2 transition
   driven_rates = torch.tensor([
       [0.0,    2000.0,  0.0],      # to level 0
       [2000.0, 0.0,     500.0],    # to level 1  
       [0.0,    500.0,   0.0]       # to level 2
   ])  # s^-1
   
   # Strong 0↔1 transition: ~0.5 ms
   # Weaker 1↔2 transition: ~2 ms
   
   context = population.Context(
       sample=triplet_sample,
       basis="eigen",
       init_populations=None, # Then the initial condition is thermal equilibrium
       driven_rates=driven_rates
   )

**Transformation Rule**

Same as thermal rates matrix:

.. math::

   D' = |U|^2 \cdot D \cdot (|U|^2)^T

Dephasing (dephasing)
---------------------

.. image:: /_static/context/dephasing.png
   :width: 70%
   :alt: Dephasing diagram
   :align: center

**Physical Meaning**

Dephasing causes loss of coherence without changing populations. This term is used only in the relaxation of density matrix formalism (see :ref:`population_description`).

**Mathematical Form**

A vector :math:`\boldsymbol{\gamma}` where :math:`\gamma_i` is the dephasing rate for level :math:`|i\rangle`. This vector is only relevant for density matrix calculations.

**Example: T₂ Relaxation in Density Matrix Dynamics**

.. code-block:: python

   import torch
   from mars import population
   
   # Define level-specific dephasing rates
   # Faster dephasing for higher energy states
   dephasing = torch.tensor([5e3, 8e3, 12e3])  # s^-1
   
   # Level 0: T₂ ~ 200 μs
   # Level 1: T₂ ~ 125 μs
   # Level 2: T₂ ~ 83 μs
   
   context = population.Context(
       sample=triplet_sample,
       basis="eigen",
       init_populations=[0.35, 0.30, 0.35],
       dephasing=dephasing,
       decay_rates=torch.tensor([60.0, 60.0, 60.0])
   )

**Effect on Coherences**

For off-diagonal density matrix elements :math:`\rho_{ij}` (where :math:`i \neq j`):

.. math::

   \frac{d\rho_{ij}}{dt} = \cdots - \frac{\gamma_i + \gamma_j}{2} \rho_{ij}


Example
--------
Let's consider a triplet state with several realxation mechanisms:

.. code-block:: python

   import torch
   from mars import spin_model, population
   
   # Create triplet system
   g_tensor = spin_model.Interaction(2.004, dtype=torch.float64)
   zfs = spin_model.DEInteraction([450e6, -90e6], dtype=torch.float64)
   
   triplet_system = spin_model.SpinSystem(
       electrons=[1.0],
       g_tensors=[g_tensor],
       electron_electron=[(0, 0, zfs)]
   )
   
   sample = spin_model.MultiOrientedSample(
       base_spin_system=triplet_system,
       ham_strain=2.5e7,
       gauss=0.0012,
       lorentz=0.0012
   )
   
   # Initial selective population (spin polarization)
   init_populations = [0.7, 0.05, 0.25]  # Strong population of one of the states
   
   # Outgoing rates in ZFS basis
   decay_rates = torch.tensor([100.0, 45.0, 70.0])  # s^-1
   
   # Spin-lattice relaxation in eigen basis  
   thermal_rates = torch.tensor([
       [0.0,    900.0,   80.0],
       [900.0,  0.0,     900.0],
       [80.0,   900.0,   0.0]
   ])  # s^-1
   
   # Dephasing rates
   dephasing = torch.tensor([6e5, 6e5, 6e5])  # s^-1
   
   # Combine in ZFS basis for populations, eigen for transitions
   context_zfs = population.Context(
       sample=sample,
       basis="zfs",
       init_populations=init_populations,
       decay_rates=decay_rates
   )
   
   context_eigen = population.Context(
       sample=sample, 
       basis="eigen",
       thermal_rates=thermal_rates,
       dephasing=dephasing
   )
   
   # Combine both mechanisms
   context_total = context_zfs + context_eigen