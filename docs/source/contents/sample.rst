Sample Representation
=====================

In EPR modeling, a *sample* extends a base spin system with broadening mechanisms (Gaussian/Lorentzian), unresolved strain, for powders-orientational averaging.


Examples
--------

**1. Powder spectrum of a triplet with axial ZFS**

Simulate a frozen solution of a diradical with D = 0.35 GHz, E = 0, using a default mesh:

.. code-block:: python

   from mars import spin_model. particles

   g_tensor = spin_model.Interaction(2.002)  # g = 2.002
   D_tensor = spin_model.DEInteraction(components=0.35 * 1e9)  # D = 0.35 GHz
   triplet = spin_model.SpinSystem(electrons=[1.0], g_tensors=[g_tensor], electron_electron=[(0, 0, D_tensor)])

   sample = spin_model.MultiOrientedSample(base_spin_system=triplet, gauss="0.01") # It is better to specify some broadening

The total broadening of each spectral line is constructed from four components:

1. **Homogeneous Lorentzian broadening**, which is applied after the spectrum is constructed.
2. **Inhomogeneous Gaussian broadening**, which is applied after the spectrum is constructed.  
   More information about final spectrum postprocessing can be found in :class:`mars.spectra_manager.spectra_manager.PostSpectraProcessing`.
3. **Residual spectrum broadening** caused by unresolved interactions. This is specified by a parameter ``ham_strain`` and can be anisotropic.
4. **Broadening caused by the presence of a Hamiltonian parameter distribution**. This parameters is specified during :class:`mars.spin_model.Interaction` creation.

All widths are specified as full width at half maximum (FWHM).

**2. Anisotropic Gaussian broadening**

Add orientation-dependent inhomogeneous broadening due to unresolved hyperfine structure:

.. code-block:: python

   # Axial unresolved broadening: σ⊥ = 5 MHz, σ∥ = 15 MHz
   ham_strain = [5e-3 * 1e9, 5e-3 * 1e9, 15e-3 * 1e9]

   sample = spin_model.MultiOrientedSample(
       base_spin_system=triplet,
       ham_strain=ham_strain,
       lorentz=0.001  # 1 mT homogeneous width
   )


**3. Custom orientation mesh**

In MaRs it is possible to modify mesh to increse resolution of increase speed. Also it is possible to set a custom mesh (see :ref:`orientational_mesh`)

.. code-block:: python

   # Use only 80 orientations instead of default ~ 200
   coarse_mesh = (10, 20)  # (initial_grid_frequency, interpolation_grid_frequency)

   fast_sample = spin_model.MultiOrientedSample(
       base_spin_system=triplet,
       mesh=coarse_mesh
   )

**4. Sample spin system orientation**

In some cases, it's convenient to rotate not just one interaction, but all interactions at once (rotate the entire spin system relative laboratory frame).
This doesn't change the final spectrum of the sample, but it can be convinient for creation of complex systems.

.. code-block:: python

   rotated_sample = spin_model.MultiOrientedSample(
       base_spin_system=triplet,
       spin_system_frame=[0.0, 0.2, 0.3],
   )


Hamiltonian Terms
-----------------

The total spin Hamiltonian in the presence of a magnetic field **B** = (B_x, B_y, B_z) is expressed as:

.. math::

   \mathcal{H}(\mathbf{B}) = F + B_x G_x + B_y G_y + B_z G_z

where:

- **F** is the field-independent (zero-field) part of the Hamiltonian,
- **G_z**, **G_y**, **G_z** are the Zeeman coupling operators that encode the system’s response to the external magnetic field via electron and nuclear g‑tensors.

Explicitly, these operators are defined as:

.. math::

   \begin{aligned}
   G_x &= \frac{\mu_\mathrm{B}}{h} \sum_{i} \left( g_{i,xx} \hat{S}_i^{(x)} + g_{i,xy} \hat{S}_i^{(y)} + g_{i,xz} \hat{S}_i^{(z)} \right)
         + \frac{\mu_\mathrm{N}}{h} \sum_{j} g_{n,j} \hat{I}_j^{(x)}, \\
   G_y &= \frac{\mu_\mathrm{B}}{h} \sum_{i} \left( g_{i,yx} \hat{S}_i^{(x)} + g_{i,yy} \hat{S}_i^{(y)} + g_{i,yz} \hat{S}_i^{(z)} \right)
         + \frac{\mu_\mathrm{N}}{h} \sum_{j} g_{n,j} \hat{I}_j^{(y)}, \\
   G_z &= \frac{\mu_\mathrm{B}}{h} \sum_{i} \left( g_{i,zx} \hat{S}_i^{(x)} + g_{i,zy} \hat{S}_i^{(y)} + g_{i,zz} \hat{S}_i^{(z)} \right)
         + \frac{\mu_\mathrm{N}}{h} \sum_{j} g_{n,j} \hat{I}_j^{(z)},
   \end{aligned}

where:

- :math:`\hat{S}_i^{(x,y,z)}` are the electron spin operators for electron :math:`i`,
- :math:`\hat{I}_j^{(x,y,z)}` are the nuclear spin operators for nucleus :math:`j`,
- :math:`\mathbf{g}_i` is the (possibly anisotropic) electron g‑tensor for electron :math:`i`,
- :math:`g_{n,j}` is the (currently isotropic) nuclear g‑factor for nucleus :math:`j`,
- :math:`\mu_\mathrm{B}` is the Bohr magneton,
- :math:`\mu_\mathrm{N}` is the nuclear magneton,
- :math:`h` is Planck’s constant.

These operators act on the full product-state Hilbert space and are returned by the sample methods as complex-valued tensors.

You can retrieve them directly:

.. code-block:: python

   # Full Hamiltonian decomposition
   F, Gx, Gy, Gz = sample.get_hamiltonian_terms()

   # Secular approximation: only retain elements of F that commute with Gz.
   #   1. Zero non-commuting elements in Zeeman terms (Gx/Gy/Gz) with respect to spin projections, making Gx,y,z[Sx,y,z == 0] = 0
   #   2. Zero non-commuting elements in F with respect to Gz, making F[Gz == 0] = 0
   F_sec, Gx, Gy, Gz = sample.get_hamiltonian_terms_secular()

The secular approximation modifies the Hamiltonian terms in two steps:

1. Zeeman term modification  
   For each component :math:`\alpha \in \{x, y, z\}`, matrix elements of :math:`G_\alpha` are zeroed where the corresponding spin projection operator :math:`S_\alpha` has negligible magnitude:
   
   .. math::
   
      (G_\alpha)_{ij} \leftarrow 
      \begin{cases} 
      (G_\alpha)_{ij} & \text{if } |(S_\alpha)_{ij}| > \varepsilon \\
      0 & \text{otherwise}
      \end{cases}
   
   This enforces approximate commutation :math:`[G_\alpha, S_\alpha] \approx 0`. For single spins with isotropic g-tensors this is exact. For anisotropic cases it retains only the diagonal part of the g-tensor.

2. Zero-field term modification 
   Matrix elements of :math:`F` are zeroed where diagonal elements of :math:`G_z` differ:
   
   .. math::
   
      F_{ij}^{\text{sec}} = 
      \begin{cases}
        F_{ij}, & \text{if } |(G_z)_{ii} - (G_z)_{jj}| < \varepsilon \\
        0,      & \text{otherwise}
      \end{cases}
   
   This enforces :math:`[F^{\text{sec}}, G_z] \approx 0` with threshold :math:`\varepsilon` (default: :math:`10^{-9}`).


Useful Features
---------------

:class:`mars.spin_model.BaseSample` and :class:`mars.spin_model.MultiOrientedSample` provide several utility methods for advanced quantum-mechanical analysis, including access to key spin operators and basis transformations.
These are especially useful when working with total spin manifolds or custom spectral models.

Basis Methods
~~~~~~~~~~~~~

The following methods allow you to construct and switch between common representations of the spin Hilbert space:

.. code-block:: python

   Mul = sample.get_spin_multiplet_basis()      # |S, M> basis (eigenbasis of S^2 and S_z)
   PR  = sample.get_product_state_basis()       # Computational |m1, m2, ...> basis

- :meth:`mars.spin_model.MultiOrientedSample.get_spin_multiplet_basis`
  Constructs a unitary transformation matrix that converts from the product-state basis to the total-spin multiplet basis :math:`|S, M\rangle`.

  - **Output**: A matrix whose columns are eigenvectors of :math:`\hat{S}^2` and :math:`\hat{S}_z`, sorted first by :math:`S`, then by :math:`M`.
  - **Ordering**: States are arranged in ascending order of total spin :math:`S`, and within each :math:`S` manifold, by increasing :math:`M`.
  - **Example**: For two spin-½ electrons, the basis order is  
    :math:`|S=0, M=0\rangle,\ |S=1, M=-1\rangle,\ |S=1, M=0\rangle,\ |S=1, M=+1\rangle`.

- :meth:`mars.spin_model.MultiOrientedSample.get_product_state_basis`
  Returns the identity matrix, confirming that internal operators are represented in the standard product-state basis:

  .. math::

     |\psi\rangle = |m_{e_1}, m_{e_2}, \dots, m_{n_1}, m_{n_2}, \dots\rangle

  - **Shape**: ``(spin_dim, spin_dim)``

- :meth:`mars.spin_model.MultiOrientedSample.get_xyz_basis`
  Returns the transition moment basis vectors :math:`T_x`, :math:`T_y`, :math:`T_z` for a spin-1 system expressed in the molecular frame.

  The basis is defined in the :math:`|M_z = +1\rangle`, :math:`|M_z = 0\rangle`, :math:`|M_z = -1\rangle` eigenbasis of :math:`\hat{S}_z`.

  - **Return**: A tensor of shape ``[..., orientations, 3, 3]``, where the last two dimensions correspond to the three Cartesian components (:math:`x, y, z`) and the three :math:`M_z` states.
  - **Example**:

    .. code-block:: python

       T = system.get_xyz_basis()   # shape: [..., orientations, 3, 3]
       Tx = T[..., 0]               # x-component, shape: [..., orientations, 3]
       Ty = T[..., 1]               # y-component, shape: [..., orientations, 3]
       Tz = T[..., 2]               # z-component, shape: [..., orientations, 3]

- :meth:`mars.spin_model.MultiOrientedSample.get_zero_field_splitting_basis`
  Returns the eigenbasis of the zero-field splitting (ZFS) Hamiltonian, denoted as :math:`\mathbf{F}`.

  The eigenvectors are ordered from the lowest to the highest eigenvalue of :math:`\mathbf{F}`.

  - **Return**: A tensor of shape ``[..., N, N]``, where :math:`N` is the spin Hilbert space dimension.

- :meth:`mars.spin_model.MultiOrientedSample.get_zeeman_basis`
  Returns the eigenbasis of the Zeeman operator :math:`\mathbf{G}_z`, corresponding to the infinite magnetic field limit along the laboratory z-axis.
 
  The eigenvectors are ordered from the lowest to the highest eigenvalue of  :math:`\mathbf{G}_z`.
  - **Return**: A tensor of shape [..., N, N], where :math:N is the spin Hilbert space dimension.

Concatenating Samples
~~~~~~~~~~~~~~~~~~~~~

MaRs allows concatenation of multiple :class:`mars.spin_model.MultiOrientedSample` objects into a single composite samples using the direct sum construction of their spin systems

This is not equivalent to building a true multi-particle quantum system (which would require a tensor-product Hilbert space). Instead, it creates a block-diagonal representation suitable for specific effective models.

Use concatenation only in scenarios such as:

-Modeling an electron that may occupy distinct spin environments (e.g., two triplet states with slightly different zero-field splitting or dipolar couplings).

-Simulating polarized or time-resolved spectra where coherence or population transfer between otherwise isolated manifolds must be tracked.

Usage
^^^^^

Concatenation requires all samples to have compatible parameters:

.. code-block:: python

   from mars import concat, spin_model
   
   # Define two independent triplet samples with same broadening
   g1 = spin_model.Interaction(2.002)
   D1 = spin_model.DEInteraction([350e6, 50e6])
   triplet_1 = spin_model.SpinSystem(
       electrons=[1.0],
       g_tensors=[g1],
       electron_electron=[(0, 0, D1)]
   )
   
   sample_1 = spin_model.MultiOrientedSample(
       base_spin_system=triplet_1,
       gauss=0.0015,
       lorentz=0.0008,
       ham_strain=[3e6, 3e6, 8e6]
   )
   
   g2 = spin_model.Interaction([2.006, 2.006, 2.002])
   D2 = spin_model.DEInteraction([280e6, 35e6])
   triplet_2 = spin_model.SpinSystem(
       electrons=[1.0],
       g_tensors=[g2],
       electron_electron=[(0, 0, D2)]
   )
   
   sample_2 = spin_model.MultiOrientedSample(
       base_spin_system=triplet_2,
       gauss=0.0015,  # Must match sample_1
       lorentz=0.0008,  # Must match sample_1
       ham_strain=[3e6, 3e6, 8e6],  # Must match sample_1
   )
   
   # Concatenate samples
   mixture = concat([sample_1, sample_2])
   # Equivalent to: spin_model.concat_multioriented_samples([sample_1, sample_2])


**See also**: :func:`mars.spin_model.concat_multioriented_samples` for implementation details and validation logic.