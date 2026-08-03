Base Spin System
==============================
The :class:`mars.spin_model.SpinSystem` class provides a flexible representation of spin systems commonly encountered in EPR spectroscopy. It supports arbitrary numbers of electron and nuclear spins, along with their mutual interactions (hyperfine, dipolar, exchange, zfs).


Examples
--------

**1. Simple triplet system (S = 1) with axial zero-field splitting**

Model a single electron with spin 1 and a D-tensor aligned with the lab frame:

.. code-block:: python

   from mars import spin_model, particles

   g_tensor = spin_model.Interaction(2.002)  # g = 2.002
   D_tensor = spin_model.DEInteraction(components=0.35 * 1e9)  # D = 0.35 GHz
   triplet = spin_model.SpinSystem(electrons=[1.0], g_tensors=[g_tensor], electron_electron=[(0, 0, D_tensor)])

This creates a three-level system with Hamiltonian :math:`H = gβBSz + D(S_z^2 - S(S+1)/3)`.

**2. Two coupled doublets with anisotropic g-tensors**

Model two interacting S = 1/2 centers with distinct g-anisotropy and a dipolar coupling:

.. code-block:: python

   g1 = spin_model.Interaction([2.002, 2.004, 2.008])  # gx, gy, gz
   g2 = spin_model.Interaction([1.998, 2.000, 2.006])

   anisotropic = spin_model.Interaction([0.01 * 1e9, 0.0 * 1e9, 0.02 * 1e9])  # Dx, Dy, Dz

   system = spin_model.SpinSystem(
       electrons=[1/2, 1/2],
       g_tensors=[g1, g2],
       electron_electron=[(0, 1, anisotropic)]
   )

The total Hilbert space dimension is 4, and the dipolar term is added via scalar product of spin operators.

**3. Electron coupled to a nucleus (hyperfine interaction)**

Model a nitroxide radical with isotropic hyperfine interaction with 14N nucleus (I = 1):

.. code-block:: python

   g_el = spin_model.Interaction([2.006, 2.006, 2.002])
   hyperfine = spin_model.Interaction([20.01 * 1e6, 20.0  * 1e6, 80.0  * 1e6]) # A-tensor in MHz
   system = spin_model.SpinSystem(
       electrons=[1/2],
       g_tensors=[g_el],
       nuclei=["14N"],
       electron_nuclei=[(0, 0, hyperfine)]
   )
	

The resulting system has dimension :math:`(2S+1)(2I+1) = 6`

Useful Features
---------------

:class:`mars.spin_model.SpinSystem` provides several utility methods for advanced quantum-mechanical analysis, including access to key spin operators.
These are especially useful when working with total spin manifolds or custom spectral models.

Basis and Operator Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~

The following methods allow you to construct and switch between common representations of the spin Hilbert space.  
In MaRs, all operators and vectors are initially defined in the product basis of individual spin projections, denoted as :math:`|\alpha\rangle` and :math:`|\beta\rangle`.

.. code-block:: python

   Mz = system.get_electron_z_operator()        # Total electron S_z operator
   S2 = system.get_electron_squared_operator()  # Total electron S² operator
   Me = system.get_electron_projections()       # Electron-only Mₑ per product state
   Mt = system.get_total_projections()          # Total M = Σmₑ + Σmₙ per product state

- :meth:`mars.spin_model.SpinSystem.get_electron_z_operator`
  Returns the total electron spin projection operator along the *z*-axis:

  .. math::

     \hat{S}_z = \sum_{i \in \text{electrons}} \hat{S}_{i}^{(z)}

  - **Shape**: ``(spin_dim, spin_dim)``
  - **Example**: For two spin-½ electrons, returns a 4×4 diagonal matrix with entries ``[1, 0, 0, -1]``.

- :meth:`mars.spin_model.SpinSystem.get_electron_squared_operator`
  Returns the total electron spin-squared operator:

  .. math::

     \hat{S}^2 = \left( \sum_i \hat{\mathbf{S}}_i \right) \cdot \left( \sum_j \hat{\mathbf{S}}_j \right)
               = \hat{S}_x^2 + \hat{S}_y^2 + \hat{S}_z^2

  - **Shape**: ``(spin_dim, spin_dim)``
  - **Eigenvalues**: :math:`S(S+1)`, where :math:`S` is the total electron spin quantum number.
  - **Example**: For two spin-½ electrons, eigenvalues are ``0`` (singlet) and ``2`` (triplet).

- :meth:`mars.spin_model.SpinSystem.get_electron_projections`
  Returns a 1D tensor containing the total electron magnetic quantum number :math:`M_e = \sum_i m_{e_i}` for each product state.

  - **Ignores nuclear spins** (sets their projections to zero).
  - **Shape**: ``(spin_dim,)``
  - **Example**: For one electron (:math:`S = \tfrac{1}{2}`) and one nucleus (:math:`I = \tfrac{1}{2}`), returns ``[0.5, 0.5, -0.5, -0.5]``.

- :meth:`mars.spin_model.SpinSystem.get_total_projections`
  Returns the total magnetic quantum number :math:`M = \sum_i m_{e_i} + \sum_j m_{n_j}` for every product state.

  - **Includes both electrons and nuclei**.
  - **Shape**: ``(spin_dim,)``
  - **Example**: Same system as above → ``[1.0, 0.0, 0.0, -1.0]``.

Applying Frame Rotations
~~~~~~~~~~~~~~~~~~~~~~~~

Spin systems in MaRs can be rotated as a whole relative to laboratory frame using the :meth:`mars.spin_model.SpinSystem.apply_rotation` method.

.. code-block:: python

   import torch
   from mars import spin_model

   # Define a spin system (e.g., nitroxide radical)
   g_el = spin_model.Interaction([2.006, 2.006, 2.002])
   A_tensor = spin_model.Interaction([20.0e6, 20.0e6, 80.0e6])
   system = spin_model.SpinSystem(
       electrons=[1/2],
       g_tensors=[g_el],
       nuclei=["14N"],
       electron_nuclei=[(0, 0, A_tensor)]
   )

   # Define a rotation matrix (e.g., 90° around y-axis)
   R = torch.tensor([[0., 0., 1.],
                     [0., 1., 0.],
                     [-1., 0., 0.]])

   # Apply rotation to all interaction tensors
   system.apply_rotation(R)

This operation updates the internal representation of all interaction tensors (g, hyperfine, dipolar, etc.) by left-multiplying their components with the provided rotation matrix:

.. math::

   \mathbf{T}_{\text{new}} = \mathbf{R} \cdot \mathbf{T}_{\text{old}}

Concatenating Spin Systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~

MaRs provides functionality to combine multiple independent spin systems into a single composite system using the direct sum construction.
This is not equivalent to building a true multi-particle quantum system (which would require a tensor-product Hilbert space). Instead, it creates a block-diagonal representation for specific effective models.

Use concatenation only in scenarios such as:

-Modeling an electron that may occupy distinct spin environments (e.g., two triplet states with slightly different zero-field splitting or dipolar couplings).

-Simulating polarized or time-resolved spectra where coherence or population transfer between otherwise isolated manifolds must be considered

Mathematical Formulation
^^^^^^^^^^^^^^^^^^^^^^^^

For :math:`n` independent spin systems with Hilbert spaces :math:`\mathcal{H}^{(1)}`, :math:`\mathcal{H}^{(2)}`, ..., :math:`\mathcal{H}^{(n)}` of dimensions :math:`d_1, d_2, ..., d_n`, the concatenated system has Hilbert space:

.. math::

   \mathcal{H}_{\text{total}} = \mathcal{H}^{(1)} \oplus \mathcal{H}^{(2)} \oplus \cdots \oplus \mathcal{H}^{(n)}

with total dimension :math:`D = \sum_{i=1}^{n} d_i`.

All spin operators become block-diagonal:

.. math::

   \hat{O}_{\text{total}} = \begin{pmatrix}
   \hat{O}^{(1)} & 0 & \cdots & 0 \\
   0 & \hat{O}^{(2)} & \cdots & 0 \\
   \vdots & \vdots & \ddots & \vdots \\
   0 & 0 & \cdots & \hat{O}^{(n)}
   \end{pmatrix}

The Hamiltonian decomposes as:

.. math::

   \hat{H}_{\text{total}} = \hat{H}^{(1)} \oplus \hat{H}^{(2)} \oplus \cdots \oplus \hat{H}^{(n)}


Usage
^^^^^

The concatenation can be performed using :func:`mars.concatination.concat` or directly via :func:`mars.spin_model.concat_spin_systems`:

.. code-block:: python

   from mars import concat, spin_model
   
   # Define two independent triplet systems
   g1 = spin_model.Interaction(2.002)
   D1 = spin_model.DEInteraction([350e6, 50e6])
   triplet_1 = spin_model.SpinSystem(
       electrons=[1.0],
       g_tensors=[g1],
       electron_electron=[(0, 0, D1)]
   )
   
   g2 = spin_model.Interaction(2.008)
   D2 = spin_model.DEInteraction([280e6, 35e6])
   triplet_2 = spin_model.SpinSystem(
       electrons=[1.0],
       g_tensors=[g2],
       electron_electron=[(0, 0, D2)]
   )
   
   # Concatenate into single 6-dimensional system
   composite_system = concat([triplet_1, triplet_2])
   # Equivalent to: spin_model.concat_spin_systems([triplet_1, triplet_2])

**See also**: :func:`mars.spin_model.concat_spin_systems` for implementation details.