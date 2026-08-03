.. _basis_transformation:

Basis Transformations
======================

The Context class in MaRs supports defining relaxation parameters in multiple bases. All parameters are automatically transformed to the eigenbasis of the full Hamiltonian in the magnetic field before computation.
This flexibility allows users to specify relaxation mechanisms in the most physically intuitive representation.

Supported Bases
---------------

MaRs provides four predefined bases:

1. **"eigen"** (default)
   
   The eigenbasis of the full Hamiltonian :math:`\hat{H}_0 = \hat{F} + g\mu_B B_z \hat{S}_z` at the resonance magnetic field.
   
   - States are sorted by increasing energy
   - This is the natural basis for most EPR calculations
   - No transformation needed if parameters are already defined here

2. **"zfs"**
   
   The eigenbasis of the zero-field splitting operator :math:`\hat{F}` (the field-independent part of the Hamiltonian).
   
   - States are sorted by increasing zero-field energy
   - For the triplet system this is the same as  "xyz" basis but usually in another order (depending on D and E values): :math:`|T_Z\rangle, |T_X\rangle, |T_Y\rangle`

3. **"multiplet"**
   
   The total spin basis :math:`|S, M\rangle`, where:
   
   - :math:`S` is the total spin quantum number
   - :math:`M` is the projection of total spin on the z-axis
   - States sorted by increasing :math:`S`, then increasing :math:`M`

4. **"product"**
   
   The uncoupled basis :math:`|m_1, m_2, \ldots, m_n\rangle` of individual spin projections.
   
   - :math:`m_k` is the projection of the :math:`k`-th electron spin
   - States sorted by **decreasing** spin projections

4. **"xyz"**
 
   A molecular-frame basis defined by the Cartesian transition moment vectors :math:`T_X`, :math:`T_Y`, :math:`T_Z`,
   and — for two-electron systems — the singlet state :math:`S_0`. The precise construction depends on
   the spin multiplicity of the system.
 
   **Single electron, spin** :math:`S = 1` **(triplet)**
 
   The basis vectors are constructed from symmetric and antisymmetric combinations of the two-electron
   product states :math:`|\alpha\alpha\rangle`, :math:`|\alpha\beta\rangle`, :math:`|\beta\alpha\rangle`,
   and :math:`|\beta\beta\rangle`, projected onto the total-spin :math:`S = 1` subspace. The three
   orthonormal triplet states are:
 
   .. math::
 
      |T_X\rangle &= \frac{1}{\sqrt{2}}\bigl(-|\alpha\alpha\rangle + |\beta\beta\rangle\bigr) \\[6pt]
      |T_Y\rangle &= \frac{i}{\sqrt{2}}\bigl(|\alpha\alpha\rangle + |\beta\beta\rangle\bigr) \\[6pt]
      |T_Z\rangle &= \frac{1}{\sqrt{2}}\bigl(|\alpha\beta\rangle + |\beta\alpha\rangle\bigr)
 
   These states correspond to the zero-field eigenstates whose transition dipole moments are aligned
   with the molecular :math:`X`, :math:`Y`, and :math:`Z` axes, respectively.
 
   **Two electrons, spin** :math:`S = 1/2` **(radical pair)**
 
   For a system of two spin-:math:`\frac{1}{2}` electrons, the basis is expressed in the uncoupled
   product basis ordered as
   :math:`|\alpha\alpha\rangle`, :math:`|\alpha\beta\rangle`, :math:`|\beta\alpha\rangle`, :math:`|\beta\beta\rangle`.
   The Clebsch–Gordan decomposition into the coupled :math:`|S, M\rangle` basis yields three triplet
   states and one singlet:
 
   .. math::
 
      |T_+\rangle &= |\alpha\alpha\rangle \\[6pt]
      |T_0\rangle &= \frac{1}{\sqrt{2}}\bigl(|\alpha\beta\rangle + |\beta\alpha\rangle\bigr) \\[6pt]
      |T_-\rangle &= |\beta\beta\rangle \\[6pt]
      |S_0\rangle &= \frac{1}{\sqrt{2}}\bigl(|\alpha\beta\rangle - |\beta\alpha\rangle\bigr)
 
   The triplet states :math:`|T_+\rangle`, :math:`|T_0\rangle`, :math:`|T_-\rangle` are then mapped
   to Cartesian transition vectors using the same convention as the spin-1 case:
 
   .. math::
 
      |T_X\rangle &= \frac{1}{\sqrt{2}}\bigl(-|T_+\rangle + |T_-\rangle\bigr)
                   = \frac{1}{\sqrt{2}}\bigl(-|\alpha\alpha\rangle + |\beta\beta\rangle\bigr) \\[6pt]
      |T_Y\rangle &= \frac{i}{\sqrt{2}}\bigl(|T_+\rangle + |T_-\rangle\bigr)
                   = \frac{i}{\sqrt{2}}\bigl(|\alpha\alpha\rangle + |\beta\beta\rangle\bigr) \\[6pt]
      |T_Z\rangle &= |T_0\rangle
                   = \frac{1}{\sqrt{2}}\bigl(|\alpha\beta\rangle + |\beta\alpha\rangle\bigr)
 
   The singlet state :math:`|S_0\rangle` is rotationally invariant (total spin :math:`S = 0`) and is
   appended as a fourth basis vector. The full basis matrix has shape :math:`(4, 4)` with columns
   :math:`[T_X,\, T_Y,\, T_Z,\, S_0]`.

5. **"zeeman"**

   The eigenbasis of the Z-projections of the Zeeman operator. It is the basis of the system in an infinite magnetic field :math:`\hat{Gz}`.
   - States are sorted by increasing energy


Custom Transformation Basis
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Users can also provide a custom transformation basis with shape :math:`[\ldots, R, 1, N, N]`: 

- :math:`R` is the number of orientations
- :math:`N` is the spin system dimension
- This basis should be defined in laboratory frame (with repspect to orientations) and in the basis of individual spin projections.

Specifying the Sample for Context Initialization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When creating a :class:`mars.population.contexts.Context`, if you set some predefined basis, you must associate it with a "sample".

In most cases, the same sample is used both to define the initial relaxation parameters (and populations) and to compute the resulting spectrum.
However, advanced workflows allow you to:

- Define populations and rates using one sample
- Compute spectra for a different sample.

Both samples must have the same Hilbert space dimension and compatible orientation grids.
Notice, that if you choose "product", "multiplet", "xyz" basis, which doesn't depend on interactions,
you will get the same result for any specified sample with the same Hilbert dimension and the same particles.

Transformation Rules
--------------------

Let :math:`V_{\text{new}}` be the eigenbasis of the full Hamiltonian and :math:`V_{\text{old}}` be the specified basis.
The transformation matrix from coordinates in old basis to coordinates in a new basis is:

.. math::

   U = V_{\text{new}}^\dagger V_{\text{old}}

This is computed using :func:`mars.population.transform.basis_transformation`:

.. code-block:: python

   from mars.population.transform import basis_transformation
   
   # basis_new and basis_old are tensors of shape [..., N, N]
   # Columns are eigenvectors
   U = basis_transformation(basis_old, basis_new)
   # Result: transformation matrix [..., N, N]

Populations and Out Probabilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These are scalar quantities associated with individual states and transform via probability conservation:

.. math::

   \mathbf{n}' = |U|^2 \cdot \mathbf{n}

.. math::

   \mathbf{o}' = |U|^2 \cdot \mathbf{o}

where :math:`|U|^2` denotes element-wise squaring of the transformation matrix.

**Implementation:** Uses :func:`mars.population.transform.get_transformation_probabilities`
and :func:`mars.population.transform.transform_state_weights_to_new_basis`:

.. code-block:: python

   from mars.population.transform import (
       get_transformation_probabilities,
       transform_state_weights_to_new_basis
   )
   
   # Get |U|^2 coefficients
   probabilities = get_transformation_probabilities(basis_old, basis_new)
   # Shape: [..., N, N] with elements |⟨new_i|old_j⟩|²
   
   # Transform populations
   populations_old = torch.tensor([0.5, 0.3, 0.2])
   populations_new = transform_state_weights_to_new_basis(populations_old, probabilities)
   
   # Transform out probabilities
   decay_rates_old = torch.tensor([100.0, 50.0, 75.0])
   decay_rates_new = transform_state_weights_to_new_basis(decay_rates_old, probabilities)

**Physical interpretation:** If state :math:`|i'\rangle` in the new basis is a superposition :math:`|i'\rangle = \sum_k U^{*}_{ik} |k\rangle`,
then its population is the sum of populations with :math:`|U^{*}_{ik}|^2` which is equal to :math:`|U_{ik}|^2`.


.. note::

   If you set the initial density as parameter of Context, then the denisty will be tranformed under desnity transforamtion rule (see futher),
   then the real part of transformed diagonal will be used as initial population if it is needed

Transition Probabilities (Free and Driven)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The library supports two types of transitions: **thermal** (spontaneous) and **driven** (induced), represented by matrices :math:`W` and :math:`D`, respectively.

Analogously to populations which transform as :math:`n'_i = \sum_k |U_{ik}|^2 n_k` - transition rates are transformed by weighting both the initial and final states with their respective overlap probabilities.
Specifically, the probability of transition from state :math:`j'` to :math:`i'` in the new basis is:

.. math::

   W'_{i'j'} = \sum_{k,l} |U_{ik}|^2 \, W_{kl} \, |U_{jl}|^2,

where :math:`|U_{ik}|^2 = |\langle i' | k \rangle|^2` is the probability that new basis state :math:`|i'\rangle` contains old basis state :math:`|k\rangle`.

In compact matrix form, this becomes:

.. math::

   W' = |U|^2 \, W \, (|U|^2)^\top,

and identically for driven transitions:

.. math::

   D' = |U|^2 \, D \, (|U|^2)^\top.


- **Free transition rates** describe spontaneous relaxation processes (e.g., spin-lattice relaxation).
- **Driven transition rates** model field-induced transitions (e.g., microwave-driven mixing).

**Implementation:** Uses :func:`mars.population.transform.transform_rate_matrix_to_new_basis`:

.. code-block:: python

   from mars.population.transform import transform_rate_matrix_to_new_basis
   # Transition probabilities between singlet state and triplet sublevels
   W_old = torch.tensor([
       [0.0,    1000.0,   1000.0, 1000.0],
       [1000.0,    0.0,   0.0,   0.0],
       [1000.0,    0.0,   0.0,   0.0],
       [1000.0,    0.0,   0.0,   0.0],
   ])
   
   # Transform to new basis
   W_new = transform_rate_matrix_to_new_basis(W_old, probabilities)
   # Applies: probabilities @ W_old @ probabilities.T

Density Matrix
^^^^^^^^^^^^^^

For full density matrix calculations:

.. math::

   \hat{\rho}' = U \hat{\rho} U^\dagger

or transforms of vectorized density matrix in Liouville space:

.. math::

   |\rho'\rangle\rangle = (U \otimes U^*) |\rho\rangle\rangle:

where :math:`\otimes` denotes the Kronecker product and :math:`U^*` is the element-wise complex conjugate of :math:`U`.

**Implementation:** Uses :func:`mars.population.transform.transform_operator_to_new_basis`:

.. code-block:: python

   from mars.population.transform import (
       transform_operator_to_new_basis
   )
   
   # Get transformation matrix (complex, not squared)
   U = basis_transformation(basis_old, basis_new)
   # This is just: basis_new.conj().T @ basis_old
   
   # Transform density matrix
   rho_old = torch.tensor([[0.6, 0.1+0.2j],
                           [0.1-0.2j, 0.4]])
   rho_new = transform_operator_to_new_basis(rho_old, U)
   # Applies: U @ rho_old @ U.conj().T


Example usage
-------------

Compute the Liouville-space transformation between eigenbases of two spin Hamiltonians and apply it to a relaxation superoperator:

.. code-block:: python

   import torch
   from mars.population.transform import (
       basis_transformation,
       compute_liouville_basis_transformation,
   )

   basis_old = torch.tensor([
       [1.0, 0.0],
       [0.0, 1.0]
   ], dtype=torch.complex64)

   basis_new = torch.tensor([
       [1.0,  1.0],
       [1.0, -1.0]
   ], dtype=torch.complex64) / torch.sqrt(torch.tensor(2.0))

   # 1. Get Hilbert-space transformation (for operators like ρ)
   U = basis_transformation(basis_old, basis_new)
   # Equivalent to: U = basis_new.conj().T @ basis_old

   # 2. Get Liouville-space transformation (for vectorized ρ or superoperators)
   T = compute_liouville_basis_transformation(basis_old, basis_new)
   # Returns: kron(U, U.conj()) with shape (4, 4) for K=2

   # 3. Transform a density matrix (Hilbert space)
   rho_old = torch.tensor([[0.6, 0.1+0.2j],
                           [0.1-0.2j, 0.4]], dtype=torch.complex64)
   rho_new = U @ rho_old @ U.conj().T

   # 4. Transform via vectorization (Liouville space) – equivalent result
   rho_old_vec = rho_old.flatten()          # Row-major: [0.6, 0.1+0.2j, 0.1-0.2j, 0.4]
   rho_new_vec = T @ rho_old_vec
   assert torch.allclose(rho_new_vec.reshape(2, 2), rho_new)

.. note::

   - For the transformation of operators from Hilbert space to Liouville space, MaRs uses **row-major** (C-order) vectorization.
     This corresponds to flattening the density matrix by stacking its rows sequentially—the default behavior in NumPy and PyTorch.
     For example, in a 2×2 system, the vectorized density matrix appears as:
     ``[ρ₀₀, ρ₀₁, ρ₁₀, ρ₁₁]``.

   - The transformation matrix for the vectorized density matrix, :math:`T = U \otimes U^*`, is unitary whenever :math:`U` is unitary.

Notes
-----

* The function assumes row-major (C-order) vectorization, consistent with PyTorch/NumPy ``.flatten()``.
* The returned for vectorized density matrix transformation :math:`T=U \otimes U^*` is unitary when :math:`U` is unitary.

Relaxation Superoperator
^^^^^^^^^^^^^^^^^^^^^^^^^

The relaxation superoperator transforms in Liouville space:

.. math::

   \hat{\mathcal{R}}' = (U \otimes U^*) \hat{\mathcal{R}} (U^\dagger \otimes U^T)

**Implementation:** Uses :func:`mars.population.transform.compute_liouville_basis_transformation` and :func:`mars.population.transform.transform_superop_to_new_basis`:

.. code-block:: python

   from mars.population.transform import (
       compute_liouville_basis_transformation,
       transform_superop_to_new_basis
   )
   
   # Get Liouville space transformation
   T_liouville = compute_liouville_basis_transformation(basis_old, basis_new)
   # Shape: [..., N², N²]
   # This is: kron(U, U.conj())
   
   # Transform superoperator
   R_old = relaxation_superop  # Shape: [..., N², N²]
   R_new = transform_superop_to_new_basis(R_old, T_liouville)
   # Applies: T @ R_old @ T.conj().T

Derivation of Transition Probabilities from the Lindblad Master Equation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The transformation rules for kinetic coefficients (loss (outgoing) rates and transition rates)
are rigorously derived from the Lindblad master equation formalism. This section connects the 
basis transformation rules used in MaRs to the transformation of Lindblad operators, ensuring
physical consistency across different representations.

The Lindblad Master Equation
""""""""""""""""""""""""""""
The time evolution of the density matrix :math:`\hat{\rho}` under Markovian dissipation is given by:

.. math::

   \frac{d\hat{\rho}}{dt} = -i[\hat{H}, \hat{\rho}] + \sum_k \gamma_k \mathcal{D}[\hat{L}_k](\hat{\rho})

where the Lindblad dissipator is:

.. math::

   \mathcal{D}[\hat{L}](\hat{\rho}) = \hat{L} \hat{\rho} \hat{L}^\dagger - \frac{1}{2} \{ \hat{L}^\dagger \hat{L}, \hat{\rho} \}

Here, :math:`\gamma_k` are the rates (kinetic coefficients) and :math:`\hat{L}_k` are the jump operators. 
When changing the basis from :math:`\{|i\rangle\}` to :math:`\{|\alpha\rangle\}` via a unitary matrix 
:math:`U` (where :math:`|\alpha\rangle = \sum_i U^{*}_{\alpha i} |i\rangle`), the operators transform, 
necessitating a corresponding transformation of the rates :math:`\gamma_k`.

Loss Probabilities (Out Probabilities)
""""""""""""""""""""""""""""""""""""""

Loss probabilities :math:`\Gamma_i` represent decay rates from state :math:`|i\rangle` to outside 
the system (e.g., radiative decay). The corresponding jump operator is:

.. math::

   \hat{L}_i = |\text{vac}\rangle\langle i|

where :math:`|\text{vac}\rangle` represents the state outside the computational subspace. 
In the new basis, the loss operator becomes:

.. math::

   \hat{L}_\alpha = |\text{vac}\rangle\langle\alpha| = \sum_i U_{\alpha i} |\text{vac}\rangle\langle i| 
   = \sum_i U_{\alpha i} \hat{L}_i

**Proof of Rate Transformation:**

The total loss rate from state :math:`|\alpha\rangle` is obtained by computing the expectation 
value of the loss superoperator:

.. math::

   \Gamma'_\alpha = \langle\alpha| \sum_i \Gamma_i \hat{L}_i^\dagger \hat{L}_i |\alpha\rangle

Substituting the transformation :math:`\hat{L}_i^\dagger \hat{L}_i = |i\rangle\langle i|`:

.. math::

   \begin{aligned}
   \Gamma'_\alpha &= \sum_i \Gamma_i \langle\alpha| i\rangle\langle i | \alpha\rangle \\
   &= \sum_i \Gamma_i |\langle\alpha|i\rangle|^2 \\
   &= \sum_i |U_{\alpha i}|^2 \Gamma_i
   \end{aligned}

In matrix form, this is :math:`\boldsymbol{\Gamma}' = |U|^2 \cdot \boldsymbol{\Gamma}`. 
This confirms that loss probabilities transform via the element-wise squared transformation 
matrix, consistent with probability conservation.

Transition Probabilities: Formal Lindblad Derivation
""""""""""""""""""""""""""""""""""""""""""""""""""""

This section provides a rigorous derivation of the transition rate transformation rule
using the Lindblad master equation formalism. We explicitly show how the secular approximation
leads to the bilinear probability rule :math:`W' = |U|^2 W (|U|^2)^\top`.

Lindblad Gain Term in Old Basis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Consider the gain part of the Lindblad dissipator for transition rates :math:`W_{ij}` 
(from state :math:`|j\rangle` to :math:`|i\rangle`) in the old basis:

.. math::

   \mathcal{L}_{\text{gain}}^{\text{old}}(\hat{\rho}) = \sum_{i,j} W_{ij} \, \hat{L}_{ij} \hat{\rho} \hat{L}_{ij}^\dagger

where the jump operators are defined as :math:`\hat{L}_{ij} = |i\rangle\langle j|`.

Transformation to New Basis
^^^^^^^^^^^^^^^^^^^^^^^^^^^

We transform the jump operators to the new basis :math:`\{|\alpha\rangle\}` using the unitary matrix 
:math:`U`, where :math:`|\alpha\rangle = \sum_k U_{\alpha k} |k\rangle`. The inverse relation is 
:math:`|k\rangle = \sum_\alpha U_{\alpha k}^* |\alpha\rangle`.

Substituting this into the jump operator definition:

.. math::

   \hat{L}_{ij} = |i\rangle\langle j| = \left( \sum_\alpha U_{\alpha i}^* |\alpha\rangle \right) 
   \left( \sum_\beta U_{\beta j} \langle\beta| \right) = \sum_{\alpha,\beta} U_{\alpha i}^* U_{\beta j} \, |\alpha\rangle\langle\beta|

Define the new basis jump operators as :math:`\hat{L}_{\alpha\beta} = |\alpha\rangle\langle\beta|`. Then:

.. math::

   \hat{L}_{ij} = \sum_{\alpha,\beta} U_{\alpha i}^* U_{\beta j} \, \hat{L}_{\alpha\beta}

Now substitute this expansion into the Lindblad gain term:

.. math::

   \begin{aligned}
   \mathcal{L}_{\text{gain}}^{\text{old}}(\hat{\rho}) &= \sum_{i,j} W_{ij} \left( \sum_{\alpha,\beta} U_{\alpha i}^* U_{\beta j} \hat{L}_{\alpha\beta} \right) \hat{\rho} 
   \left( \sum_{\gamma,\delta} U_{\gamma i} U_{\delta j}^* \hat{L}_{\gamma\delta}^\dagger \right) \\
   &= \sum_{\alpha,\beta,\gamma,\delta} \left[ \sum_{i,j} W_{ij} \, U_{\alpha i}^* U_{\gamma i} \, U_{\beta j} U_{\delta j}^* \right] 
   \hat{L}_{\alpha\beta} \hat{\rho} \hat{L}_{\gamma\delta}^\dagger
   \end{aligned}

Secular Approximation
^^^^^^^^^^^^^^^^^^^^^

The expression above contains cross-terms where :math:`(\alpha,\beta) \neq (\gamma,\delta)`. 
In the interaction picture, the operator product :math:`\hat{L}_{\alpha\beta} \hat{\rho} \hat{L}_{\gamma\delta}^\dagger` 
oscillates at frequency :math:`\omega_{\alpha\beta} - \omega_{\gamma\delta}`, where 
:math:`\omega_{\alpha\beta} = E_\alpha - E_\beta`.

The **secular approximation** assumes that relaxation processes are slow compared to coherent 
precession. Terms with non-zero frequency differences (:math:`\omega_{\alpha\beta} \neq \omega_{\gamma\delta}`) 
average to zero over the relaxation timescale. We retain only the stationary terms where 
:math:`\alpha = \gamma` and :math:`\beta = \delta`.

Applying this approximation (:math:`\delta_{\alpha\gamma} \delta_{\beta\delta}`):

.. math::

   \mathcal{L}_{\text{gain}}^{\text{new}}(\hat{\rho}) \approx \sum_{\alpha,\beta} \left[ \sum_{i,j} W_{ij} \, |U_{\alpha i}|^2 \, |U_{\beta j}|^2 \right] 
   \hat{L}_{\alpha\beta} \hat{\rho} \hat{L}_{\alpha\beta}^\dagger

Identification of New Rates
^^^^^^^^^^^^^^^^^^^^^^^^^^^

We define the new transition rates :math:`W'_{\alpha\beta}` such that the master equation retains 
its Lindblad form in the new basis:

.. math::

   \mathcal{L}_{\text{gain}}^{\text{new}}(\hat{\rho}) = \sum_{\alpha,\beta} W'_{\alpha\beta} \, \hat{L}_{\alpha\beta} \hat{\rho} \hat{L}_{\alpha\beta}^\dagger

By comparing coefficients, we obtain the transformation rule:

.. math::

   W'_{\alpha\beta} = \sum_{i,j} |U_{\alpha i}|^2 \, W_{ij} \, |U_{\beta j}|^2

In matrix notation, letting :math:`P = |U|^2` be the element-wise squared transformation matrix:

.. math::

   W' = P \, W \, P^\top

This confirms that transition rates transform via the bilinear probability rule,
consistent with the interpretation of :math:`W_{ij}` as incoherent transition rates between states.

Basis Transformation in Multiplied Contexts
-------------------------------------------

When constructing composite systems via Kronecker multiplication, each subsystem is typically defined in its own "intra-basis" (e.g., molecular frame, ZFS basis, zeeman basis).
Consider two subsystems with initial bases :math:`V^{(1)}_{\text{old}}` and :math:`V^{(2)}_{\text{old}}`. If the subsystems interact weakly, the composite eigenbasis may still decompose as:

.. math::

   V_{\text{new}} = V^{(1)}_{\text{new}} \otimes V^{(2)}_{\text{new}}

where the individual transformations are:

.. math::

   U_1 = (V^{(1)}_{\text{new}})^\dagger V^{(1)}_{\text{old}}, \quad
   U_2 = (V^{(2)}_{\text{new}})^\dagger V^{(2)}_{\text{old}}
	

Density Composition
^^^^^^^^^^^^^^^^^^^

For density matrices, the composite transformation follows the unitary rule:

.. math::

   \rho_{\text{total}} \rightarrow (U_1 \otimes U_2) \, \rho_{\text{total}} \, (U_1 \otimes U_2)^\dagger

Superoperator Composition
^^^^^^^^^^^^^^^^^^^^^^^^^^

For relaxation superoperators in Liouville space, two sequential operations are required:

1. Permutation of the Kronecker-sum structure to reconcile vectorization ordering. The identity

   .. math::
   
      \operatorname{vec}(\rho_1 \otimes \rho_2) \neq \operatorname{vec}(\rho_1) \otimes \operatorname{vec}(\rho_2)
   
   necessitates a commutation matrix :math:`\Pi` such that
   
   .. math::
   
      \operatorname{vec}(\rho_1 \otimes \rho_2) = \Pi \bigl[ \operatorname{vec}(\rho_1) \otimes \operatorname{vec}(\rho_2) \bigr]
   
   This permutation must be applied to the composite superoperator *before* basis transformation:
   
   .. math::
   
      \hat{\mathcal{R}}_{\text{initial}} = \Pi \,
      \bigl[ \hat{\mathcal{R}}^{(1)} \otimes \hat{\mathbb{I}}^{(2)} + \hat{\mathbb{I}}^{(1)} \otimes \hat{\mathcal{R}}^{(2)} \bigr] \,
      \Pi^\top

2. Joint basis transformation using the composite unitary matrix. After permutation, the superoperator transforms as:
   
   .. math::
   
      \hat{\mathcal{R}}_{\text{total}} \rightarrow
      \bigl[(U_1 \otimes U_2) \otimes (U_1^* \otimes U_2^*)\bigr] \;
      \hat{\mathcal{R}}_{\text{initial}} \;
      \bigl[(U_1^\dagger \otimes U_2^\dagger) \otimes (U_1^T \otimes U_2^T)\bigr]
   
   The transformation matrix
   
   .. math::
   
      T = (U_1 \otimes U_2) \otimes (U_1^* \otimes U_2^*)
   
   applies a joint change of basis to the vectorized composite state.

.. note::

   The secular structure of a relaxation superoperator is defined by Bohr
   frequencies of the composite system.

   After permutation and basis transformation, the superoperator generally does
   not become diagonal in Liouville space. Instead, it decomposes into blocks
   corresponding to equal energy gaps.

   As a result, coherence–coherence couplings within the same Bohr-frequency
   sector are preserved by the secular approximation.

Population Composition
^^^^^^^^^^^^^^^^^^^^^^

Populations and transition rates transform differently.
They follow probability rules rather than amplitude rules. For a population vector :math:`\mathbf{n}`:

.. math::

   \mathbf{n} \rightarrow |U|^2 \mathbf{n}, \quad

Importantly, for Kronecker products the element-wise squaring distributes exactly:

.. math::

   |U_1 \otimes U_2|^2 = |U_1|^2 \otimes |U_2|^2

Furthermore, for any unitary matrix :math:`U`, the probability matrix :math:`|U|^2` satisfies:

.. math::

   |U|^2 \, \mathbf{1} = \mathbf{1}

where :math:`\mathbf{1}` is the vector of ones. This holds because unitary matrices have orthonormal rows (:math:`\sum_j |U_{ij}|^2 = 1` for all :math:`i`), so each row of :math:`|U|^2` sums to unity.

Therefore populations and decay rates can be transformed separately in each subsystem and then combined via Kronecker product. The transformation is separable:

.. math::

   \mathbf{n}_{\text{new}} = (|U_1|^2 \mathbf{n}^{(1)}) \otimes (|U_2|^2 \mathbf{n}^{(2)}) = (|U_1 \otimes U_2|^2 (\mathbf{n}^{(1)} \otimes \mathbf{n}^{(2)})

.. math::

   \boldsymbol{\Gamma}_{\text{new}} = (|U_1|^2 \boldsymbol{\Gamma}^{(1)}) \otimes \mathbf{1}^{(2)} + \mathbf{1}^{(1)} \otimes (|U_2|^2 \boldsymbol{\Gamma}^{(2)}) = (|U_1 \otimes U_2|^2 (\boldsymbol{\Gamma}^{(1)} \otimes \mathbf{1}^{(2)} + \mathbf{1}^{(1)} \otimes \boldsymbol{\Gamma}^{(2)})


Transition Probabilities Compositions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

However, for transition rate matrices the situation is more subtle.
The full transformation of the Kronecker-sum structure does not factorize cleanly:

.. math::

   \begin{aligned}
   &|U_1 \otimes U_2|^2 \, (K_1 \otimes \mathbb{I}_2 + \mathbb{I}_1 \otimes K_2) \, |U_1 \otimes U_2|^2{}^\top \\
   &= (|U_1|^2 \otimes |U_2|^2) (K_1 \otimes \mathbb{I}_2) (|U_1|^2{}^\top \otimes |U_2|^2{}^\top) + (|U_1|^2 \otimes |U_2|^2) (\mathbb{I}_1 \otimes K_2) (|U_1|^2{}^\top \otimes |U_2|^2{}^\top) \\
   &= (|U_1|^2 K_1 |U_1|^2{}^\top) \otimes (|U_2|^2 |U_2|^2{}^\top) + (|U_1|^2 |U_1|^2{}^\top) \otimes (|U_2|^2 K_2 |U_2|^2{}^\top)
   \end{aligned}

Since :math:`|U|^2` is a doubly probability matrix (not unitary), we generally have :math:`|U|^2 |U|^2{}^\top \neq \mathbb{I}`. Therefore:

.. math::

   (|U_1|^2 K_1 |U_1|^2{}^\top) \otimes (|U_2|^2 |U_2|^2{}^\top) \neq (|U_1|^2 K_1 |U_1|^2{}^\top) \otimes \mathbb{I}_2

This non-factorization means that transforming transition rates separately and then forming the Kronecker sum does not generally yield the same result
as forming the Kronecker sum first and then transforming the composite operator.

Nevertheless, MaRs consistently interprets transition rates as probabilities for "state-to-state processes" via the deriviation from Lindblad master equation.

Lindblad Formulation for Subsystem Relaxation
"""""""""""""""""""""""""""""""""""""""""""""
Consider a composite system of :math:`N` subsystems. The relaxation originating from subsystem :math:`s` is described by a generalized jump operator:

.. math::

    \hat{L}_{ji}^{(s)} = \hat{E}_1 \otimes |j\rangle\langle i|_s \otimes \hat{E}_2,

where :math:`|j\rangle\langle i|_s` acts non-trivially only on subsystem :math:`s`, and :math:`\hat{E}_{1,2}` are identity operators on the remaining subsystems. The corresponding local transition rate :math:`k_{i \to j}^{(s)}` is an element of the intra-system rate matrix :math:`\mathbf{K}^{(s)}`, with :math:`\mathbf{K}^{(s)}_{ji} = k_{i \to j}^{(s)}`.

Let :math:`\{|\alpha\rangle\}` denote the eigenbasis of the full composite Hamiltonian. Each eigenstate is expanded in the product basis :math:`\{|\mathbf{n}\rangle = |n_1\rangle \otimes \dots \otimes |n_N\rangle\}` as:

.. math::

    |\alpha\rangle = \sum_{\mathbf{n}} c^{(\alpha)}_{\mathbf{n}} |\mathbf{n}\rangle, \quad c^{(\alpha)}_{\mathbf{n}} = \langle \alpha | \mathbf{n} \rangle.

The coefficients :math:`c^{(\alpha)}_{\mathbf{n}}` generalize Clebsch-Gordan coefficients to arbitrary coupled spin systems and are computed directly from the composite transformation matrix.

Under the secular approximation, the population dynamics are governed by the transition rates :math:`W_{\alpha\beta}^{(s)}` from eigenstate :math:`|\beta\rangle` to :math:`|\alpha\rangle` due to subsystem :math:`s`.
Evaluating the Lindblad gain term :math:`\langle \alpha | \hat{L}_{ji}^{(s)} \hat{\rho} \hat{L}_{ji}^{(s)\dagger} | \alpha \rangle` for a diagonal density matrix yields:

.. math::

    W_{\alpha\beta}^{(s)} = \sum_{i,j} k_{i \to j}^{(s)} \left| \langle \alpha | \hat{L}_{ji}^{(s)} | \beta \rangle \right|^2.

Substituting the explicit form of the jump operator and expanding in the product basis:

.. math::

    \langle \alpha | \hat{L}_{ji}^{(s)} | \beta \rangle = \sum_{\mathbf{m}, \mathbf{n}} c^{(\alpha)}_{\mathbf{m}} \langle \mathbf{m} | \hat{E}_1 \otimes |j\rangle\langle i|_s \otimes \hat{E}_2 | \mathbf{n} \rangle \left(c^{(\beta)}_{\mathbf{n}}\right)^*.

The matrix element :math:`\langle \mathbf{m} | \hat{L}_{ji}^{(s)} | \mathbf{n} \rangle` is non-zero only when the states match on all subsystems except :math:`s`, and when :math:`m_s = j`, :math:`n_s = i`.
Grouping the multi-index as :math:`\mathbf{n} = (\mathbf{r}_{\setminus s}, n_s)`, where :math:`\mathbf{r}_{\setminus s}` spans all configurations of subsystems other than :math:`s`, we obtain:

.. math::

    \langle \alpha | \hat{L}_{ji}^{(s)} | \beta \rangle = \sum_{\mathbf{r}_{\setminus s}} c^{(\alpha)}_{(\mathbf{r}_{\setminus s}, j)} \left( c^{(\beta)}_{(\mathbf{r}_{\setminus s}, i)} \right)^*.

This expression represents a coherent sum over all pathways through the untouched subsystems. Inserting this into the rate definition gives the exact transformation rule:

.. math::

    W_{\alpha\beta}^{(s)} = \sum_{i,j} \mathbf{K}^{(s)}_{ji} \left| \sum_{\mathbf{r}_{\setminus s}} c^{(\alpha)}_{(\mathbf{r}_{\setminus s}, j)} \left( c^{(\beta)}_{(\mathbf{r}_{\setminus s}, i)} \right)^* \right|^2.

Defining the overlap tensor :math:`\mathcal{M}^{(s)}_{ij,\alpha\beta} = \sum_{\mathbf{r}_{\setminus s}} c^{(\alpha)}_{(\mathbf{r}_{\setminus s}, j)} \left( c^{(\beta)}_{(\mathbf{r}_{\setminus s}, i)} \right)^*`, the total transition rate matrix for the composite system is assembled as:

.. math::

    \mathbf{W}_{\text{total}} = \sum_{s=1}^N \sum_{i,j} \mathbf{K}^{(s)}_{ji} \left| \mathcal{M}^{(s)}_{ij} \right|^{\circ 2},

where :math:`|\cdot|^{\circ 2}` denotes element-wise squared magnitude.
Expanding the squared modulus separates the rate into two contributions:

.. math::

    \left| \sum_{\mathbf{r}_{\setminus s}} c^{(\alpha)}_{(\mathbf{r}_{\setminus s}, j)} \left( c^{(\beta)}_{(\mathbf{r}_{\setminus s}, i)} \right)^* \right|^2 = \underbrace{ \sum_{\mathbf{r}_{\setminus s}} \left| c^{(\alpha)}_{(\mathbf{r}_{\setminus s}, j)} \right|^2 \left| c^{(\beta)}_{(\mathbf{r}_{\setminus s}, i)} \right|^2 }_{\text{incoherent (classical) part}} + \underbrace{ \sum_{\mathbf{r} \neq \mathbf{r}'} c^{(\alpha)}_{(\mathbf{r}, j)} \left( c^{(\beta)}_{(\mathbf{r}, i)} \right)^* \left( c^{(\alpha)}_{(\mathbf{r}', j)} \right)^* c^{(\beta)}_{(\mathbf{r}', i)} }_{\text{coherent interference terms}}.

The incoherent part recovers a double-weighted probability sum, where each local transition :math:`k_{i \to j}^{(s)}` is scaled by the probability of finding subsystem :math:`s` in state :math:`j` when the total system is in :math:`|\alpha\rangle`, and similarly for state :math:`i` and :math:`|\beta\rangle`.
The interference terms account for phase relationships between different product-basis subspaces.

This formulation does not rely on the factorization :math:`|U_1 \otimes U_2|^2 (K_1 \otimes \mathbb{I} + \mathbb{I} \otimes K_2) |U_1 \otimes U_2|^2{}^\top`.
Instead, each subsystems rate matrix is independently mapped to the composite eigenbasis using the marginalized coefficient tensor, preserving the physical independence of local relaxation channels.

For composite systems, MaRs implements the dedicated :class:`mars.population.contexts.KroneckerContext`. This class automatically computes these coefficients from the composite contexts.

Context Transformation Interface
----------------------------------

The :class:`mars.population.contexts.Context` class provides a convenient interface to transform all internally stored relaxation and initialization parameters into an arbitrary target basis.

Consider the following setup for a triplet spin system:

.. code-block:: python

    import torch
    from mars.population import Context

    # Define initial parameters in the "zfs" basis
    init_density = torch.tensor(
        [[0.2, 0.3, 0.5],
         [0.4, 0.3, 0.1],
         [0.2, 0.2, 0.5]],
        dtype=torch.complex128
    )

    decay_rates = torch.tensor([1.0, 1.0, 1.0], device=device, dtype=dtype) * 100
    thermal_rates = torch.tensor([[0.0, 1.0, 0.0],
                               [1.0, 0.0, 1.0],
                               [0.0, 1.0, 0.0]], device=device, dtype=dtype) * 1000
    driven_rates = torch.tensor([[0.0, 1.0, 0.0],
                                 [1.0, 0.0, 1.0],
                                 [0.0, 1.0, 0.0]], device=device, dtype=dtype) * 1000

    context = Context(
        sample=triplet,
        basis="zfs",
        init_density=init_density,
        thermal_rates=thermal_rates,
        decay_rates=decay_rates,
        driven_rates=driven_rates,
        device=device,
        dtype=dtype
    )

Now, suppose we wish to evaluate the model at a low magnetic field (e.g., 10 mT). We first compute the eigenbasis of the full Hamiltonian:

.. code-block:: python

    F, _, _, Gz = triplet.get_hamiltonian_terms()
    field = 0.01  # 10 mT
    values, vectors = torch.linalg.eigh(F + field * Gz)
    vectors = vectors.unsqueeze(-3)  # Shape: [orientations, 1, N, N]

The ``vectors`` tensor now represents the target basis (columns are eigenvectors of the full Hamiltonian). The :class:`mars.population.contexts.Context` instance can transform all its internal quantities into this basis using the following methods:

- **Initial populations (diagonal of the density matrix):**  
  See :meth:`mars.population.contexts.Context.get_transformed_init_populations`.

  .. code-block:: python

      pop = context.get_transformed_init_populations(full_system_vectors=vectors)
      # Returns: tensor of shape [..., N]

- **Full initial density matrix:**  
  See :meth:`mars.population.contexts.Context.get_transformed_init_density`.

  .. code-block:: python

      rho = context.get_transformed_init_density(full_system_vectors=vectors)
      # Returns: tensor of shape [..., N, N]

- **Outgoing (decay) rates (scalar rates per state):**
  See :meth:`mars.population.contexts.Context.get_transformed_decay_rates`.

  .. code-block:: python

      out = context.get_transformed_decay_rates(full_system_vectors=vectors)
      # Returns: tensor of shape [..., N]

- **Thermal (spontaneous) transition rates (between states):**
  See :meth:`mars.population.contexts.Context.get_transformed_thermal_rates`.

  .. code-block:: python

      W = context.get_transformed_thermal_rates(full_system_vectors=vectors)
      # Returns: tensor of shape [..., N, N]

- **Thermal (spontaneous) relaxation superoperator (in Liouville space):**
  See :meth:`mars.population.contexts.Context.get_transformed_thermal_superop`.

  .. code-block:: python

      R_thermal = context.get_transformed_thermal_superop(full_system_vectors=vectors)
      # Returns: tensor of shape [..., N², N²]

- **Driven relaxation superoperator (in Liouville space):**  
  See :meth:`mars.population.contexts.Context.get_transformed_driven_superop`.

  .. code-block:: python

      R_driven = context.get_transformed_driven_superop(full_system_vectors=vectors)
      # Returns: tensor of shape [..., N², N²]

