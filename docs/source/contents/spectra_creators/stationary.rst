Stationary Spectra
==================

The :class:`mars.spectra_manager.spectra_manager.StationarySpectra` class simulates standard continuous-wave (CW) EPR spectra under thermal equilibrium conditions. It assumes a fixed microwave frequency while sweeping the external magnetic field, and computes either absorption (harmonic=0) or first-derivative (harmonic=1) lineshapes.

Key Features
------------

- Full support for arbitrary spin systems (electrons, nuclei, zero-field splitting, hyperfine, exchange, dipolar couplings).
- Automatic Boltzmann population weighting at a user-specified temperature.
- Orientation averaging over powder samples using configurable meshes (Delaunay, axial, crystal).
- Line broadening via Gaussian, Lorentzian, or Voigt profiles, and also Hamiltonian parameters strain modeling and residual broadening due to unresolved interactions
- Batched computation for high-throughput simulation of parameter ensembles.
- Any spectra creator can be transfered to cuda using:

.. code-block:: python

   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   spectra_creator = spectra_creator.to(device)


Mathematical Foundation
-----------------------
   Intensities are computed by :class:`mars.spectra_manager.intensity_base.BaseResIntensityCalculator`.

   Assuming the quantization axis aligns with the magnetic field direction (e.g., :math:`\mathbf{B} \parallel z`), the intensity of a transition between eigenstates :math:`|i\rangle` and :math:`|j\rangle` is:

   .. math::
      I_{ij} \propto |\langle i | \hat{G}_\perp | j \rangle|^2 \cdot (p_j - p_i)

   where the population of state *k* for the equilibrium case is:

   .. math::
      p_k = \frac{e^{-E_k / k_B T}}{Z}, \quad Z = \sum_k e^{-E_k / k_B T} 

   In the crystal case, :math:`I_{ij} \propto \left| \langle i | \hat{G}_x | j \rangle \right|^2 \cdot (p_j - p_i)`.

   In **non-equilibrium** simulations (e.g., photoexcited states), populations :math:`p_i(t)` are not thermal and are managed via Context tool: see :class:`mars.population.contexts.Context`.

Examples
--------

**Example 1: Basic triplet spectrum with ZFS**

.. code-block:: python

   g_tensor = spin_model.Interaction(2.0023)
   zfs = spin_model.DEInteraction([500e6, 100e6])  # D=500 MHz, E=100 MHz
   sys = spin_model.SpinSystem(electrons=[1.0], g_tensors=[g_tensor],
                                electron_electron=[(0, 0, zfs)])
   sample = spin_model.MultiOrientedSample(sys, gauss=0.001, lorentz=0.001)
   creator = spectra_manager.StationarySpectra(freq=9.8e9, sample=sample)
   fields = torch.linspace(0.30, 0.40, 1000)
   spec = creator(sample, fields)

**Example 2: Batched simulation of Mn(II) with varying ZFS**

.. code-block:: python

   D_vals = torch.tensor([480e6, 520e6])
   E_vals = torch.tensor([90e6, 110e6])
   zfs_batch = spin_model.DEInteraction(torch.stack([D_vals, E_vals], dim=1))
   sys = spin_model.SpinSystem(electrons=[2.5], electron_electron=[(0,0,zfs_batch)])
   sample = spin_model.MultiOrientedSample(sys, gauss=0.002, lorentz=0.001)
   creator = spectra_manager.StationarySpectra(freq=9.5e9, sample=sample, temperature=4.0)
   fields = torch.stack([torch.linspace(0.25, 0.45, 1000)] * 2)
   spectra = creator(sample, fields)  # shape: (2, 1000)