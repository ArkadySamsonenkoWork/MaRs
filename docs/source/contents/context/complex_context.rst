.. _complex_context:

Complex Context Construction
============================

Real spin systems often involve multiple simultaneous relaxation mechanisms or coupled subsystems.
MaRs provides powerful algebraic operations to construct complex Context objects from simpler building blocks: addition (``+``), multiplication (``@``), and concationation (via :func:`mars.concatination.concat`).

Context Algebra Overview
------------------------

MaRs supports three fundamental operations:

1. **Addition** (``context_1 + context_2``): Combines independent relaxation processes acting on the same spin system

2. **Kronecker-Multiplication** (``context_1 @ context_2`` or ``mars.multiply((context_1, context_2))``): Creates a composite system from independent subsystems of different spin-centers

3. **Concatenation** (:func:`mars.concat((context_1, context_2)) or <mars.concatination.concat>`): Creates a composite system from independent subsystems of the spin-center via direct sum 

Operations automatically handle:

- Basis transformations to a common eigenbasis
- Detailed balance enforcement
- Proper composition of kinetic matrices or relaxation superoperators

Addition
--------

.. image:: /_static/context/summed_context.png
   :width: 100%
   :alt: addition context
   :align: center


The addition operator combines multiple relaxation pathways that act simultaneously on the same set of quantum states. This is the more common operation for building realistic relaxation models.

For the addition the :class:`mars.population.contexts.SummedContext` is used. In the general for speed up it can be instantiated directly:

.. code-block:: python

   summ_context = population.SummedContext(contexts=[context_1, context_2, ..., context_N])
   # Equivalent to: context_1 + context_2 + ... + context_N, but more efficient

Mathematical Formulation
^^^^^^^^^^^^^^^^^^^^^^^^

For **population dynamics**, the total kinetic matrix is the sum of individual contributions:

.. math::

   K_{\text{total}} = K^{(1)} + K^{(2)} + \cdots + K^{(n)}

For **density matrix dynamics**, the total relaxation superoperator becomes:

.. math::

   \hat{\mathcal{R}}_{\text{total}} = \hat{\mathcal{R}}^{(1)} + \hat{\mathcal{R}}^{(2)} + \cdots + \hat{\mathcal{R}}^{(n)}

Each component :math:`K^{(i)}` or :math:`\hat{\mathcal{R}}^{(i)}` is first calculated in its own specified basis, then all are transformed to the common eigenbasis of the full Hamiltonian before summation.

Physical Scenarios
^^^^^^^^^^^^^^^^^^

Common situations requiring addition:

1. **Multiple relaxation pathways:** Phosphorescence decay + relaxation between triplet sublevels
2. **Different bases:** Initial populations in ZFS basis + transitions in eigen basis


Example 1: Triplet State with Multiple Mechanisms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A triplet state formed by intersystem crossing typically exhibits:

- Selective initial population in the molecular frame (ZFS basis)
- Phosphorescence (radiative decay) in the molecular frame
- Spin-lattice relaxation between Zeeman levels (eigen basis)

.. code-block:: python

   import torch
   from mars import spin_model. population, spectra_manager
   
   # Define triplet system
   g_tensor = spin_modelInteraction(2.0032, dtype=torch.float64)
   zfs = spin_modelDEInteraction([540e6, 78e6], dtype=torch.float64)
   
   triplet_system = spin_modelSpinSystem(
       electrons=[1.0],
       g_tensors=[g_tensor],
       electron_electron=[(0, 0, zfs)]
   )
   
   sample = spin_modelMultiOrientedSample(
       base_spin_system=triplet_system,
       ham_strain=2.2e7,
       gauss=0.0011,
       lorentz=0.0011
   )
   
   # Context 1: Selective population and phosphorescence (ZFS basis)

   initial_pops = [0.72, 0.08, 0.20]  # [TZ, TX, TY]
   
   # Different phosphorescence rates for each sublevel
   phosphorescence = torch.tensor([105.0, 48.0, 82.0])  # s^-1
   
   context_molecular = population.Context(
       sample=sample,
       basis="zfs",
       init_populations=initial_pops,
       decay_rates=phosphorescence
   )
   
   # Context 2: Spin-lattice relaxation (eigen basis)
   # Fast equilibration between adjacent levels
   # Slower direct transitions between extreme levels
   spin_lattice = torch.tensor([
       [0.0,    1100.0,  85.0],
       [1100.0, 0.0,     1100.0],
       [85.0,   1100.0,  0.0]
   ])  # s^-1
   
   context_zeeman = population.Context(
       sample=sample,
       basis="eigen",
       thermal_rates=spin_lattice,
       temperature=140.0  # K
   )
   
   # Combine both mechanisms
   context_total = context_molecular + context_zeeman
   
   # Calculate time-resolved spectrum
   tr_spectra = spectra_manager.CoupledTimeSpectra(
       freq=9.6e9,
       sample=sample,
       harmonic=0,
       context=context_total,
       temperature=300.0
   )
   
   fields = torch.linspace(0.30, 0.40, 1000, dtype=torch.float64)
   times = torch.linspace(0, 8e-3, 600, dtype=torch.float64)  # 8 ms
   
   spectrum_2d = tr_spectra(sample, fields, times)

Physical behavior:

- **t < 1 ms:** Fast redistribution of population between Zeeman levels
- **1 ms < t < 10 ms:** Slow decay of total triplet population via phosphorescence

Example 2: Adding Dephasing to Population Dynamics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For density matrix calculations, coherence decay is crucial. Without some dephasing, the system may saturate due to an oscillating magnetic field, which is not observed experimentally.
Therefore, even if the dephasing is unknown, it is best to specify it to avoid unexpected spectral saturation effects.:

.. code-block:: python

   import torch
   from mars import population, spectra_manager
   
   # Context 1: Population dynamics (populations + transitions)
   context_pop = population.Context(
       sample=sample,
       basis="zfs",
       init_populations=[0.68, 0.12, 0.20],
       decay_rates=torch.tensor([92.0, 45.0, 75.0]),  # s^-1
       thermal_rates=torch.tensor([
           [0.0,    950.0,   70.0],
           [950.0,  0.0,     950.0],
           [70.0,   950.0,   0.0]
       ])
   )
   
   # Context 2: Pure dephasing (only affects coherences)
   T2_probs = torch.tensor([6e5, 9.5e5, 14e5])  # s^-1
   
   context_dephasing = population.Context(
       sample=sample,
       basis="eigen",
       dephasing=T2_probs
   )
   
   # Combine for full density matrix dynamics
   context_density = context_pop + context_dephasing
   
   # Use density matrix solver
   tr_spectra_density = spectra_manager.DensityTimeSpectra(
       freq=9.6e9,
       sample=sample,
       harmonic=0,
       context=context_density,
       temperature=140.0,
       populator="rwa"  # Rotating wave approximation. It is default for DensityTimeSpectra
   )
   
   spectrum_with_dephasing = tr_spectra_density(sample, fields, times)

Example 3: Multiple Independent Processes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Radical pair systems may have multiple competing decay channels:

.. code-block:: python

   # Context 1: Singlet-triplet interconversion  |S=0, Mz=0> <-> |S=1, Mz=0>
   st_mixing = torch.tensor([
       [0.0,    0.0,  2500.0,     0.0],
       [0.0,    0.0,     0.0,     0.0],
       [2500.0, 0.0,     0.0,     0.0],
       [0.0,    0.0,     0.0,     0.0]
   ])  # s^-1
   
   context_st = population.Context(
       sample=radical_pair_sample,
       basis="multiplet",
       thermal_rates=st_mixing
   )
   
   # Context 2: Radical recombination (singlet channel)
   recombination = torch.tensor([5000.0, 0.0, 0.0, 0.0])  # s^-1
   
   context_recomb = population.Context(
       sample=radical_pair_sample,
       basis="multiplet",
       decay_rates=recombination
   )
   
   # Context 3: Spin relaxation in product basis
   relaxation = torch.tensor([
       [0.0,  500.0,  500.0,  0.0],
       [500.0, 0.0,   0.0,    500.0],
       [500.0, 0.0,   0.0,    500.0],
       [0.0,   500.0, 500.0,  0.0]
   ])  # s^-1
   
   context_relax = population.Context(
       sample=radical_pair_sample,
       basis="product",
       thermal_rates=relaxation
   )
   
   # Combine all three mechanisms
   context_rp = context_st + context_recomb + context_relax

Properties of Addition
^^^^^^^^^^^^^^^^^^^^^^

1. **Commutative:** ``context_1 + context_2 == context_2 + context_1``

2. **Associative:** ``(context_1 + context_2) + context_3 == context_1 + (context_2 + context_3)``

3. **Basis-independent:** Contexts can be defined in different bases; transformation is automatic

4. **Initial state:** All initial populations are summed up. 

Kronecker-Multiplication
------------------------

.. image:: /_static/context/kronecker_multiplication.png
   :width: 100%
   :alt: Kronecker multiplication context
   :align: center

The multiplication operator (``@``) constructs a composite system from two (or more) independent subsystems.

For the multiplication the "mars.multiply" (:func:`mars.multiplication.multiply`) is used.
Also it can be done via specific function :func:`mars.population.context.multiply_contexts()`
In the general for speed up it is better instantiate it directly

.. code-block:: python

   mul_context = mars.multiply([context_1, context_2, context_N])  # The same as context_1 @ context_2 @ context_N, but more efficiently

Mathematical Formulation
^^^^^^^^^^^^^^^^^^^^^^^^

For population dynamics, the composite system is formed via tensor products:

.. math::

   \mathbf{n}_{\text{total}} = \mathbf{n}^{(1)} \otimes \mathbf{n}^{(2)}

.. math::

   K_{\text{total}} = K^{(1)} \otimes \mathbb{I}^{(2)} + \mathbb{I}^{(1)} \otimes K^{(2)}

For density matrix dynamics:

.. math::

   \hat{\rho}_{\text{total}} = \hat{\rho}^{(1)} \otimes \hat{\rho}^{(2)}

.. math::

   \hat{\mathcal{R}}_{\text{total}} = \hat{\mathcal{R}}^{(1)} \otimes \hat{\mathbb{I}}^{(2)} + \hat{\mathbb{I}}^{(1)} \otimes \hat{\mathcal{R}}^{(2)}

where :math:`\mathbb{I}` and :math:`\hat{\mathbb{I}}` are identity operators in Hilbert and Liouville space, respectively.

.. note::
   
   Vectorization requires permutation: When working with superoperators in Liouville space,
   the vectorization of a Kronecker product does not factorize directly:
   
   .. math::
   
      \operatorname{vec}(\rho_1 \otimes \rho_2) \neq \operatorname{vec}(\rho_1) \otimes \operatorname{vec}(\rho_2)
   
   Instead, a commutation (permutation) matrix :math:`\Pi` reconciles the ordering:
   
   .. math::
   
      \operatorname{vec}(\rho_1 \otimes \rho_2) = \Pi \bigl[ \operatorname{vec}(\rho_1) \otimes \operatorname{vec}(\rho_2) \bigr]
   
   To perform transformation between v


Implementation Strategy
^^^^^^^^^^^^^^^^^^^^^^^

MaRs uses Clebsch-Gordan coefficients to construct the :class:`mars.population.contexts.KroneckerContext`
object, which handles all basis transformations consistently. For relaxation, superoperators are first generated for each subsystem in its native intra-basis ("zfs", "xyz", "zeeman", ...),
and only then are these superoperators composed in the multiplied Hilbert space.

While populations, initial states,
decay rates, and density matrices admit unambiguous Kronecker-product transformations between bases,
transition rate matrices (``thermal_rates``, ``driven_rates``) require special considiration due to their
dependence on both initial and final state overlaps. The exploration for these transformations is shown in the :ref:`basis_transformation`. 

Key functions:

- :func:`mars.population.transform.compute_clebsch_gordan_probabilities` - Compute transformation coefficients
- :func:`mars.population.transform.transform_kronecker_populations` - Transform populations
- :func:`mars.population.transform.transform_kronecker_rate_matrix` - Transform probabilities matrices
- :func:`mars.population.transform.transform_kronecker_rate_vector` - Transform probabilities vector 
- :func:`mars.population.transform.transform_kronecker_operator` - Transform Hilbert operators (density matrices)
- :func:`mars.population.transform.transform_kronecker_superoperator` - Transform superoperators
- :func:`mars.population.transform.reshape_vectorized_kronecker_to_tensor_product` - Convert Kronecker-ordered vectorized states to tensor-product ordering
- :func:`mars.population.transform.reshape_vectorized_tensor_product_to_kronecker` - Convert tensor-product-ordered vectorized states to Kronecker ordering
- :func:`mars.population.transform.reshape_superoperator_kronecker_to_tensor_basis` - Convert superoperators between Kronecker and tensor-product bases
- :func:`mars.population.transform.reshape_superoperator_tensor_to_kronecker_basis` - Convert superoperators between tensor-product and Kronecker bases

.. code-block:: python

   from mars.population.transform import (
       compute_clebsch_gordan_probabilities,
       transform_kronecker_populations,
       transform_kronecker_rate_matrix
   )
   
   # Example: Two subsystems with bases basis_1 and basis_2
   # Target is the coupled basis (eigen basis of full system)
   
   # Compute Clebsch-Gordan coefficients
   coeffs = compute_clebsch_gordan_probabilities(
       target_basis=coupled_basis,
       basis_list=[basis_1, basis_2]
   )
   # Shape: [..., k1, k2, K] where K = k1*k2
   
   # Transform populations
   pops_1 = torch.tensor([0.7, 0.2, 0.1])  # 3 levels
   pops_2 = torch.tensor([0.5, 0.5])        # 2 levels  
   pops_total = transform_kronecker_populations([pops_1, pops_2], coeffs)
   # Shape: [..., 6] for combined 3×2 system
   
   # Transform rate matrices
   K1 = torch.zeros(3, 3)  # Rates for system 1
   K2 = torch.zeros(2, 2)  # Rates for system 2
   K_total = transform_kronecker_rate_matrix([K1, K2], coeffs)
   # Shape: [..., 6, 6]


Transformation Rules
^^^^^^^^^^^^^^^^^^^^

**Populations** (init_populations): Kronecker product :math:`\text{n}_1 \otimes \text{n}_2 \otimes \cdots`

**Density matrices** (init_density): Kronecker product :math:`\rho_1 \otimes \rho_2 \otimes \cdots`

**Diagonal operators** (decay_rates, dephasing): Sum of local terms :math:`v_1 \otimes I + I \otimes v_2 + \cdots`

**Matrices** (thermal_rates, driven_rates, supeoperators): Sum of local operators :math:`K_1 \otimes I + I \otimes K_2 + \cdots`

.. note::

   For all context parameters (e.g., populations, density matrices, rate/probability matrices, etc.) involved in Kronecker multiplication:
   
   - If **all** operands are ``None``, the resulting parameter in the multiplied context is ``None``.
   - If **some** operands are ``None``, the ``None`` entries are replaced by zero-valued arrays of compatible shape.
     Consequently, for populations and density matrices, the total state of the composite system becomes zero (since the Kronecker product with a zero tensor is zero).

Examples
--------

.. code-block:: python

   import torch
   from mars.population import Context
   
   ctx1 = Context(
       sample=sample,
       basis="zeeman",
       init_populations=torch.tensor([0.6, 0.4]),
       thermal_rates=torch.tensor([[0.0, 0.1], [0.1, 0.0]]),
       dtype=torch.float64
   )
   
   ctx2 = Context(
       sample=sample,
       basis="zeeman",
       init_populations=torch.tensor([0.5, 0.5]),
       thermal_rates=torch.tensor([[0.0, 0.05], [0.05, 0.0]]),
       dtype=torch.float64
   )
   
   composite = ctx1 @ ctx2  # Dimension: 4


.. note::

   Relaxation superoperators in multiplied systems obey the secular condition
   with respect to the *total* Hamiltonian, not the individual subsystems.

   Even if each subsystem superoperator contains only secular terms in its local
   eigenbasis, the composite superoperator may contain off-diagonal entries in
   Liouville space after Kronecker construction and permutation.

   These terms correspond to coherence–coherence couplings that share the same
   Bohr frequency:

   .. math::

      E_i - E_j = E_k - E_l.

   Such contributions are preserved in MaRs.

Combining Multiplication and Addition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For complex systems, both operations can be combined:

.. code-block:: python

   # Let's consider two triplet systems with defined relaxation to initial singlet state
   context_triplet_1 = population.Context(
       sample=single_sample,
       basis="zfs",
       init_populations=[0.68, 0.12, 0.20],
       decay_rates=torch.tensor([95.0, 50.0, 75.0])
   )
   
   context_triplet_2 = population.Context(
       sample=single_sample,
       basis="zfs",
       init_populations=[0.22, 0.65, 0.13],
       decay_rates=torch.tensor([95.0, 50.0, 75.0])
   )
   
   # Composite initial state and decay
   context_product = context_triplet_1 @ context_triplet_2
   
   # Add collective relaxation processes in eigenbasis


   collective_relax = torch.zeros(9, 9, dtype=torch.float64)
   # Define selected transitions between composite levels
   collective_relax[1, 3] = 200.0  # s^-1
   collective_relax[3, 1] = 200.0
   collective_relax[2, 5] = 150.0
   collective_relax[5, 2] = 150.0
   
   context_collective = population.Context(
       sample=double_sample,
       basis="eigen",
       thermal_rates=collective_relax,
       temperature=100.0
   )
   
   # Total context: product state + collective relaxation
   context_full = context_product + context_collective

Properties of Multiplication
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Non-commutative**: While mathematically symmetric, the order affects level indexing

2. **Associative:** ``(context_1 @ context_2) @ context_3 == context_1 @ (context_2 @ context_3)``

3. **Dimensionality:** :math:`N_{\text{total}} = N_1 \times N_2`

Multiplication with Summed Contexts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The behavior of Kronecker multiplication when one operand is a :class:`SummedContext` is not distributive in the naive sense because populations and kinetic matrices transform differently under tensor products.

Consider:

- Populations (density matrices) combine as:  
  :math:`(\mathbf{n_1} + \mathbf{n_2}) \otimes \mathbf{n_3} = \mathbf{n_1} \otimes \mathbf{n_3} + \mathbf{n_2} \otimes \mathbf{n_3}`

- Kinetic matrices (or superoperators) combine as:  
  :math:`(K_1 + K_2) \otimes \mathbb{I} + \mathbb{I} \otimes K_3 = (K_1 \otimes \mathbb{I} + \mathbb{I} \otimes K_3) + (K_2 \otimes \mathbb{I}) + (\mathbb{I} \otimes 0)`

In this case, MaRs rewrites the result as a sum of two elements:

1. :math:`(\mathbf{n_1}, K_1) \otimes (\mathbf{n_3}, K_3)`
2. :math:`(\mathbf{n_2}, K_2) \otimes (\mathbf{n_3}, 0)`

This rule is applied automatically when evaluating expressions like:

.. code-block:: python

   (context_A + context_B) @ context_C

The resulting object is a :class:`mars.population.contexts.SummedContext` containing two :class:`mars.population.contexts.Context` terms.

Concatenation
--------------

.. image:: /_static/context/concat_context.png
   :width: 100%
   :alt: Concatenation of contexts
   :align: center

The concatenation operation (:func:`mars.concat([context_1, context_2]) <mars.concatination.concat>`) constructs a composite system from energy-isolated subspaces within a single physical spin system.
This situation arises when a molecule can exist in multiple distinct configurations, such as different conformational isomers or spatially localized states.

In this case, the total Hilbert space decomposes as a **direct sum** of subspaces:
.. math::

   \mathcal{H}_{\text{total}} = \mathcal{H}^{(1)} \oplus \mathcal{H}^{(2)}

Accordingly, initial states and relaxation operators combine via direct summation:

For population dynamics (kinetic matrices)

.. math::

   \mathbf{n}_{\text{total}} = \mathbf{n}^{(1)} \oplus \mathbf{n}^{(2)}, \quad
   K_{\text{total}} = K^{(1)} \oplus K^{(2)}

For density matrix:

.. math::

   \rho_{\text{total}} = \rho^{(1)} \oplus \rho^{(2)} =
   \begin{pmatrix}
     \rho^{(1)} & 0 \\
     0 & \rho^{(2)}
   \end{pmatrix}


Under vectorization (row-major in MaRs), :math:`\mathrm{vec}(\rho_1 \oplus \rho_2) \neq \mathrm{vec}(\rho_1) \oplus \mathrm{vec}(\rho_2)` due to non-contiguous
index mapping. The composite relaxation superoperator requires explicit embedding into the correct
Liouville subspaces:

.. math::

   \hat{\mathcal{R}}_{\text{total}} = \mathcal{P} \cdot 
   \bigl( \hat{\mathcal{R}}^{(1)} \oplus \hat{\mathcal{R}}^{(2)} \bigr) \cdot \mathcal{P}^\top

where :math:`\mathcal{P}` is the permutation reconciling row-major vectorization of block-diagonal states.

Example: Two Triplet States in Distinct Conformers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Consider a molecule that can adopt two stable conformations, each hosting a triplet state.
The two triplets are separated by a energy gap (:math:`\sim`10–100 GHz), and interconversion between conformers is slow (:math:`\ll 10^3` s\ :sup:`-1`).
Each triplet has three spin sublevels (:math:`m_S = -1, 0, +1`), so the full system has six levels—but they form two decoupled blocks in the absence of inter-conformer relaxation.

We model each conformer independently, then concatenate their contexts:

.. code-block:: python

   import torch
   import mars
   from mars import spin_model. population
   
   device = torch.device('cpu')
   dtype = torch.float64
   
   # Define two identical triplet systems.
   g_tensor = spin_modelInteraction(2.0032, dtype=dtype)
   zfs = spin_modelDEInteraction([540e6, 78e6], dtype=dtype)
   
   triplet_1 = spin_modelSpinSystem(
       electrons=[1.0],
       g_tensors=[g_tensor],
       electron_electron=[(0, 0, zfs)]
   )
   
   triplet_2 = spin_modelSpinSystem(
       electrons=[1.0],
       g_tensors=[g_tensor],
       electron_electron=[(0, 0, zfs)],
       energy_shift=1e10,  # Shift energy (10 GHz) of second triplet to simulate the energy gap between states
   )
   
   # Combine into a single sample for concatenated system
   triplet_sample_1 = spin_modelMultiOrientedSample(
       base_spin_system=triplet_1,
       ham_strain=2.2e7,
       gauss=0.0011,
       lorentz=0.0011
   )

   triplet_sample_2 = spin_modelMultiOrientedSample(
       base_spin_system=triplet_2,
       ham_strain=2.2e7,
       gauss=0.0011,
       lorentz=0.0011
   )
   
   init_populations = [0.7, 0.2, 0.1]  # [TZ, TX, TY]
   decay_rates = torch.tensor([100.0, 50.0, 80.0], device=device, dtype=dtype)
   
   # Context for first conformer (low-energy triplet)
   context_zfs_1 = population.Context(
       sample=triplet_sample_1,
       basis="zfs",
       init_populations=init_populations,
       decay_rates=decay_rates,
       device=device,
       dtype=dtype
   )
   
   # Context for second conformer (high-energy triplet)
   context_zfs_2 = population.Context(
       sample=triplet_sample_2,
       basis="zfs",
       init_populations=init_populations,
       decay_rates=decay_rates,
       device=device,
       dtype=dtype
   )
   
   # Concatenate the two isolated triplets
   context_concat = mars.concat([context_zfs_1, context_zfs_2])
   combined_sample = mars.concat((triplet_sample_1, triplet_sample_2))

Now, suppose there is slow thermal relaxation between corresponding spin sublevels of the two triplets.
We add this as a separate context defined in the eigenbasis of the combined system:

.. code-block:: python

   dim = 6  # 3 + 3 levels
   probs_exchange = torch.zeros((dim, dim), device=device, dtype=dtype)
   relaxation_rate = 1e2  # 100 s⁻¹
   
   # Relaxation from high-energy triplet (indices 3,4,5) → low-energy triplet (0,1,2)
   probs_exchange[3, 0] = relaxation_rate  # m_S = -1 → m_S = -1
   probs_exchange[4, 1] = relaxation_rate  # m_S =  0 → m_S =  0
   probs_exchange[5, 2] = relaxation_rate  # m_S = +1 → m_S = +1
   
   
   context_exchange = population.Context(
       sample=triplet_combined,
       basis="eigen",
       thermal_rates=probs_exchange,
       temperature=300.0,
       device=device,
       dtype=dtype
   )
   
   # Full context: concatenated triplets + slow inter-conformer relaxation
   context_full = context_concat + context_exchange

Properties of Concatenation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Non-commutative**: Although the direct sum operation is mathematically symmetric, it is **non-commutative in practice** with respect to level indexing. Specifically,
:math:`\texttt{context\_1} \oplus \texttt{context\_2} \neq \texttt{context\_2} \oplus \texttt{context\_1}`
because :math:`\texttt{context\_1}` occupies the lower indices in the resulting combined system.

2. **Associative:**:   concat(concat(context_1, context_2), context_3) == concat(context_1, concat(context_2, context_3))

2. **Dimensionality**: :math:`N_{\text{total}} = N_1 + N_2`

Order of Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python

   #  They are equivalent
   result_1 = mars.concat((context_1, context_2)) + mars.concat((context_3, context_4))
   result_2 = mars.concat((context_1, context_4)) + mars.concat((context_3, context_2))


.. _dephasing_concatenation:

Dephasing in Concatenated Systems
---------------------------------

There is a distinction between concatenating Context objects directly versus concatenating their pre-computed relaxation superoperators.
When using :func:`mars.concat` on a list of Context objects, the library constructs the composite relaxation superoperator from the combined rate parameters.
This ensures that coherences between states belonging to different subsystems, for example coherence between state i in subsystem 1 and state j in subsystem 2, decay correctly according to the sum of their individual dephasing rates.

In contrast, if relaxation superoperators are computed separately for each subsystem and then combined into a block-diagonal matrix, the off-diagonal blocks connecting the subsystems remain zero.
This implies that inter-system coherences do not decay due to relaxation, meaning the time derivative of the density matrix element :math:`\rho_{12}` is zero for cross-subsystem elements.

The following example demonstrates this difference.

.. code-block:: python

    import torch
    from mars import population, concat
    sample_1, sample_2, sample_3 = create_samples()

    # Define dephasing rates for two independent subsystems
    dephasing_1 = torch.tensor([1e6, 1e6], dtype=torch.float64)
    dephasing_2 = torch.tensor([1.5e6, 1.5e6], dtype=torch.float64)

    # Create Contexts with defined dephasing and zero population transfer
    context_1 = population.Context(
       sample=sample_1,
       basis="eigen",
       dephasing=dephasing_1,
       thermal_rates=torch.zeros(2, 2, dtype=torch.float64)
    )

    context_2 = population.Context(
       sample=sample_2,
       basis="eigen",
       dephasing=dephasing_2,
       thermal_rates=torch.zeros(2, 2, dtype=torch.float64)
    )

    # Context Concatenation
    # The superoperator is generated from the combined rate structure
    context_combined = concat([context_1, context_2])
    sup_combined = context_combined.get_transformed_thermal_superop(None)

    # Superoperator Combination
    # Compute superoperators for each subsystem independently
    sup_1 = context_1.get_transformed_thermal_superop(None)
    sup_2 = context_2.get_transformed_thermal_superop(None)

    # Manually construct block-diagonal superoperator
    # This lacks the cross-terms required for inter-system coherence decay
    sup_manual = transform.reshape_superoperators_list_to_direct_sum_basis((sup_1, sup_2))

    # Verification: The two superoperators are not equal
    # The difference lies in the off-diagonal blocks corresponding to 
    # coherences between subsystem 1 and subsystem 2
    assert not torch.allclose(sup_combined, sup_manual)

    print(f"Combined superoperator includes inter-system dephasing: {not torch.allclose(sup_combined, sup_manual)}")

