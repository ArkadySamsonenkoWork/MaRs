Sample Representation
=====================

In EPR modeling, a *sample* extends a base spin system with broadening mechanisms (Gaussian/Lorentzian), unresolved strain, for powders-orientational averaging.


Examples
--------

**1. Powder spectrum of a triplet with axial ZFS**

Simulate a frozen solution of a diradical with D = 0.35 GHz, E = 0, using a default mesh:

.. code-block:: python

   from mars import spin_model, particles

   g_tensor = spin_model.Interaction(2.002)  # g = 2.002
   D_tensor = spin_model.DEInteraction(components=0.35 * 1e9)  # D = 0.35 GHz
   triplet = spin_model.SpinSystem(electrons=[1.0], g_tensors=[g_tensor], electron_electron=[(0, 0, D_tensor)])

   sample = spin_model.SolidSample(base_spin_system=triplet, gauss="0.01") # It is better to specify some broadening

The total broadening of each spectral line is constructed from four components:

1. **Homogeneous Lorentzian broadening**, which is applied after the spectrum is constructed.
2. **Inhomogeneous Gaussian broadening**, which is applied after the spectrum is constructed.  
   More information about final spectrum postprocessing can be found in :class:`mars.spectra_manager.spectra_processing_base.PostSpectraProcessing`.
3. **Residual spectrum broadening** caused by unresolved interactions. This is specified by a parameter ``ham_strain`` and can be anisotropic.
4. **Broadening caused by the presence of a Hamiltonian parameter distribution**. This parameters is specified during :class:`mars.spin_model.Interaction` creation.

All widths are specified as full width at half maximum (FWHM).

**2. Anisotropic Gaussian broadening**

Add orientation-dependent inhomogeneous broadening due to unresolved hyperfine structure:

.. code-block:: python

   # Axial unresolved broadening: σ⊥ = 5 MHz, σ∥ = 15 MHz
   ham_strain = [5e-3 * 1e9, 5e-3 * 1e9, 15e-3 * 1e9]

   sample = spin_model.SolidSample(
       base_spin_system=triplet,
       ham_strain=ham_strain,
       lorentz=0.001  # 1 mT homogeneous width
   )


**3. Custom orientation mesh**

In MaRs it is possible to modify mesh to increse resolution of increase speed. Also it is possible to set a custom mesh (see :ref:`orientational_mesh`)

.. code-block:: python

   # Use only 80 orientations instead of default ~ 200
   coarse_mesh = (10, 20)  # (initial_grid_frequency, interpolation_grid_frequency)

   fast_sample = spin_model.SolidSample(
       base_spin_system=triplet,
       mesh=coarse_mesh
   )

**4. Sample spin system orientation**

In some cases, it's convenient to rotate not just one interaction, but all interactions at once (express the entire spin system in the laboratory (X, Y, Z) frame).
This doesn't change the final spectrum of the sample, but it can be convinient for creation of complex systems.

.. code-block:: python

   rotated_sample = spin_model.SolidSample(
       base_spin_system=triplet,
       spin_system_frame=[0.0, 0.2, 0.3],
   )


Hamiltonian Terms
-----------------

The total spin Hamiltonian in the presence of a magnetic field **B** = (B_X, B_Y, B_Z) is expressed as:

.. math::

   \hat{\mathcal{H}}(\mathbf{B}) = \hat{F} + B_X \hat{G}_X + B_Y \hat{G}_Y + B_Z \hat{G}_Z

where:

- :math:`\hat{F}` is the field-independent (zero-field) part of the Hamiltonian,
- :math:`\hat{G}_X`, :math:`\hat{G}_Y`, :math:`\hat{G}_Z` are the Zeeman coupling operators that encode the system’s response to the external magnetic field via electron and nuclear g‑tensors.

The operators :math:`\hat{G}_\alpha` are defined as the derivatives of the Hamiltonian (in Hz) with respect to the magnetic field and are related to the magnetic dipole moment operator :math:`\hat{\mu}_\alpha` by:

.. math::

   \hat{G}_\alpha \equiv \frac{\partial \hat{\mathcal{H}}}{\partial B_\alpha} = -\frac{\hat{\mu}_\alpha}{h}, \quad \alpha \in \{X, Y, Z\}

Explicitly, these operators are defined as:

.. math::

   \begin{aligned}
   \hat{G}_X &= \frac{\mu_\mathrm{B}}{h} \sum_{i} \left( g_{i,XX} \hat{S}_i^{(X)} + g_{i,XY} \hat{S}_i^{(Y)} + g_{i,XZ} \hat{S}_i^{(Z)} \right)
         - \frac{\mu_\mathrm{N}}{h} \sum_{j} g_{n,j} \hat{I}_j^{(X)}, \\
   \hat{G}_Y &= \frac{\mu_\mathrm{B}}{h} \sum_{i} \left( g_{i,YX} \hat{S}_i^{(X)} + g_{i,YY} \hat{S}_i^{(Y)} + g_{i,YZ} \hat{S}_i^{(Z)} \right)
         - \frac{\mu_\mathrm{N}}{h} \sum_{j} g_{n,j} \hat{I}_j^{(Y)}, \\
   \hat{G}_Z &= \frac{\mu_\mathrm{B}}{h} \sum_{i} \left( g_{i,ZX} \hat{S}_i^{(X)} + g_{i,ZY} \hat{S}_i^{(Y)} + g_{i,ZZ} \hat{S}_i^{(Z)} \right)
         - \frac{\mu_\mathrm{N}}{h} \sum_{j} g_{n,j} \hat{I}_j^{(Z)},
   \end{aligned}

where:

- :math:`\hat{S}_i^{(X,Y,Z)}` are the electron spin operators for electron :math:`i`,
- :math:`\hat{I}_j^{(X,Y,Z)}` are the nuclear spin operators for nucleus :math:`j`,
- :math:`\mathbf{g}_i` is the (possibly anisotropic) electron g‑tensor for electron :math:`i`,
- :math:`g_{n,j}` is the (currently isotropic) nuclear g‑factor for nucleus :math:`j`,
- :math:`\mu_\mathrm{B}` is the Bohr magneton,
- :math:`\mu_\mathrm{N}` is the nuclear magneton,
- :math:`h` is Planck’s constant.

These operators act on the full product-state Hilbert space and are returned by the sample methods as complex-valued tensors.

You can retrieve the full Hamiltonian terms directly:

.. code-block:: python

   F, G_X, G_Y, G_Z = sample.get_hamiltonian_terms()

.. _hamiltonian-secularization:

Hamiltonian Secularization
~~~~~~~~~~~~~~~~~~~~~~~~~

Hamiltonian secularization is requested at the sample level through
``get_hamiltonian_terms_secular()``:

.. code-block:: python

   F_sec, G_X_sec, G_Y_sec, G_Z_sec = sample.get_hamiltonian_terms_secular()

The returned terms define the Hamiltonian used in a secular calculation. For a
static field along the laboratory Z-axis, it is written as

.. math::

   \hat{\mathcal{H}}_{\mathrm{sec}}(B_0)
   = \hat{F}_{\mathrm{sec}} + B_0\hat{G}_{Z,\mathrm{sec}}.

The projection uses the total spin operators, including both electrons and nuclei:

.. math::

   \hat{S}_\alpha^{\mathrm{tot}}
   =\sum_e\hat{S}_{e,\alpha}+\sum_n\hat{I}_{n,\alpha},
   \qquad \alpha\in\{X,Y,Z\}.

The field-independent term and longitudinal Zeeman operator retain all matrix
elements within each eigenspace of :math:`\hat{S}^{\mathrm{tot}}_Z` and remove
matrix elements connecting different total-spin projections. In particular,
electron--nuclear flip-flops that conserve total projection may remain. The
corresponding conditions are

.. math::

   [\hat{F}_{\mathrm{sec}},\hat{S}^{\mathrm{tot}}_Z]=0,
   \qquad
   [\hat{G}_{Z,\mathrm{sec}},\hat{S}^{\mathrm{tot}}_Z]=0.

The transverse terms are projected onto the commutants of their corresponding
total spin components:

.. math::

   [\hat{G}_{X,\mathrm{sec}},\hat{S}^{\mathrm{tot}}_X]=0,
   \qquad
   [\hat{G}_{Y,\mathrm{sec}},\hat{S}^{\mathrm{tot}}_Y]=0.

These conditions do not generally imply
:math:`[\hat{F}_{\mathrm{sec}},\hat{G}_{Z,\mathrm{sec}}]=0`.
The Zeeman operators include both electron and nuclear contributions.

The secular validity check is controlled by ``ComputationalDetails``.
With ``secular_field=None``, the discarded Frobenius norm is checked for
:math:`F` and :math:`G_Z` separately. For a specified ``secular_field``
(default ``0.3`` T), only :math:`H=F+B G_Z` is checked against its secular
counterpart. The total-spin commutator checks apply in either case.

Where Secularization Is Applied
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the spectrum calculation, Hamiltonian secularization is selected by the
``hamiltonian_mode`` argument:

.. code-block:: python

   from mars.spectra_manager import StationarySpectra

   spectra = StationarySpectra(
       freq=9.8e9,
       sample=sample,
       hamiltonian_mode="secular",
   )

With ``hamiltonian_mode="secular"``, the spectrum object obtains its terms from
``sample.get_hamiltonian_terms_secular()``. With
``hamiltonian_mode="direct"``, ``sample.get_hamiltonian_terms()`` is called.
The equivalent enum values are ``HamComputationMethod.SECULAR`` and
``HamComputationMethod.DIRECT`` from ``mars.spectra_manager``.

The method is selected during initialization of the spectrum object. During
``BaseResSpectra.forward()``, the selected sample method is called before
resonance finding. The returned :math:`F` and :math:`G_Z` are passed to the
resonance-field solver, and the returned set
:math:`(F,G_X,G_Y,G_Z)` is passed to the subsequent intensity and population
calculations. Secularization is thus applied when the Hamiltonian terms are
obtained, before eigenvalues and eigenvectors are calculated.

``StationarySpectra``, ``TruncTimeSpectra``, and ``CoupledTimeSpectra`` use
``hamiltonian_mode="direct"`` by default. ``DensityTimeSpectra`` uses
``hamiltonian_mode="secular"`` by default, consistent with its default RWA
population model.

Hamiltonian secularization and the rotating-wave approximation are separate
parts of the calculation. The former determines the static Hamiltonian terms;
the latter determines how the microwave interaction is treated during
rotating-frame density-matrix evolution. Selecting ``hamiltonian_mode="direct"``
does not by itself disable the RWA in a density-matrix populator. See
:ref:`rotating-wave-approximation` for the rotating-frame requirements.

The secular approximation is appropriate when the neglected couplings are
small compared with the relevant Zeeman energy separations. Its accuracy
should be checked against a direct calculation when the zero-field splitting
parameters :math:`D` and :math:`E`, or other interactions, become comparable
with those separations, particularly near level crossings.


Useful Features
---------------

:class:`mars.spin_model.BaseSample` and :class:`mars.spin_model.SolidSample` provide several utility methods for advanced quantum-mechanical analysis, including access to key spin operators and basis transformations.
These are especially useful when working with total spin manifolds or custom spectral models.

Basis Methods
~~~~~~~~~~~~~

The following methods allow you to construct and switch between common representations of the spin Hilbert space:

.. code-block:: python

   Mul = sample.get_spin_multiplet_basis()      # Electron |S, M> basis (S_e^2 and S_ez)
   PR  = sample.get_product_state_basis()       # Computational |m1, m2, ...> basis

- :meth:`mars.spin_model.SolidSample.get_spin_multiplet_basis`
  Constructs a basis from the total **electron** spin operators :math:`\hat S_e^2` and :math:`\hat S_{e,Z}`. Nuclear-spin projections are not included in these two operators.

  - **Output**: A matrix whose columns are eigenvectors of :math:`\hat{S}_e^2` and :math:`\hat{S}_{e,Z}`, sorted first by electron spin :math:`S`, then by electron projection :math:`M_e`.
  - **Ordering**: Electron-spin states are arranged in ascending order of :math:`S`, then :math:`M_e`; nuclear degeneracies are not resolved by these operators.
  - **Example**: For two spin-½ electrons, the basis order is  
    :math:`|S=0, M=0\rangle,\ |S=1, M=-1\rangle,\ |S=1, M=0\rangle,\ |S=1, M=+1\rangle`.

- :meth:`mars.spin_model.SolidSample.get_product_state_basis`
  Returns the identity matrix, confirming that internal operators are represented in the standard product-state basis:

  .. math::

     |\psi\rangle = |m_{e_1}, m_{e_2}, \dots, m_{n_1}, m_{n_2}, \dots\rangle

  - **Shape**: ``(spin_dim, spin_dim)``

- :meth:`mars.spin_model.SolidSample.get_xyz_basis`
  Returns the transition moment basis vectors :math:`T_x`, :math:`T_y`, :math:`T_z` for a spin-1 system expressed in the molecular frame.

  The basis is defined in the :math:`|M_z = +1\rangle`, :math:`|M_z = 0\rangle`, :math:`|M_z = -1\rangle` eigenbasis of :math:`\hat{S}_z`.

  - **Return**: A tensor of shape ``[..., orientations, 3, 3]``, where the last two dimensions correspond to the three Cartesian components (:math:`x, y, z`) and the three :math:`M_z` states.
  - **Example**:

    .. code-block:: python

       T = system.get_xyz_basis()   # shape: [..., orientations, 3, 3]
       Tx = T[..., 0]               # x-component, shape: [..., orientations, 3]
       Ty = T[..., 1]               # y-component, shape: [..., orientations, 3]
       Tz = T[..., 2]               # z-component, shape: [..., orientations, 3]

- :meth:`mars.spin_model.SolidSample.get_zero_field_splitting_basis`
  Returns the eigenbasis of the zero-field splitting (ZFS) Hamiltonian, denoted as :math:`\hat{F}`.

  The eigenvectors are ordered from the lowest to the highest eigenvalue of :math:`\hat{F}`.

  - **Return**: A tensor of shape ``[..., N, N]``, where :math:`N` is the spin Hilbert space dimension.

- :meth:`mars.spin_model.SolidSample.get_zeeman_basis`
  Returns the eigenbasis of the Zeeman operator :math:`\hat{G}_Z`, corresponding to the infinite magnetic field limit along the laboratory Z-axis.
 
  The eigenvectors are ordered from the lowest to the highest eigenvalue of  :math:`\hat{G}_Z`.
  - **Return**: A tensor of shape [..., N, N], where :math:N is the spin Hilbert space dimension.

Concatenating Samples
~~~~~~~~~~~~~~~~~~~~~

MaRs allows concatenation of multiple :class:`mars.spin_model.SolidSample` objects into a single composite samples using the direct sum construction of their spin systems

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
   
   sample_1 = spin_model.SolidSample(
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
   
   sample_2 = spin_model.SolidSample(
       base_spin_system=triplet_2,
       gauss=0.0015,  # Must match sample_1
       lorentz=0.0008,  # Must match sample_1
       ham_strain=[3e6, 3e6, 8e6],  # Must match sample_1
   )
   
   # Concatenate samples
   mixture = concat([sample_1, sample_2])
   # Equivalent to: spin_model.concat_multioriented_samples([sample_1, sample_2])


**See also**: :func:`mars.spin_model.concat_multioriented_samples` for implementation details and validation logic.