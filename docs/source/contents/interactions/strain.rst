.. _strain-management-mars:

Strain in MaRs
=======================

"Strain" in MaRs refers to the *distribution of spin Hamiltonian parameters* due to structural disorder (e.g., in frozen solutions, powders, or amorphous matrices). This leads to inhomogeneous broadening of EPR lines.

Strain is specified in MaRs as the full width at half maximum (FWHM) of the parameter distribution and is expressed in the natural units of the corresponding Hamiltonian parameter (dimensionless for the g-tensor, Hz for zero-field splitting).

Strain vs. Explicit Modeling
-----------------------------

To understand strain modeling, consider two approaches to simulate disorder in a sample:

**Explicit Modeling (Monte Carlo)**

In explicit modeling, you would generate many spin systems with different parameter values drawn from a distribution, compute the spectrum for each, and average:

.. code-block:: python

   import torch
   from mars import spin_model
   
   # Generate 1000 samples from a Gaussian distribution
   n_samples = 1000
   g_mean = torch.tensor([2.002, 2.004, 2.008])
   g_std = torch.tensor([0.001, 0.001, 0.002]) / (2*math.sqrt(2*math.log(2)))  # to convert σ into FWHM
   
   # Sample g-values from Gaussian distribution
   g_samples = g_mean + g_std * torch.randn(n_samples, 3)
   
   # Create 1000 different spin systems
   interactions = []
   for i in range(n_samples):
       g = spin_model.Interaction(g_samples[i])
       interactions.append(g)
   
   # Compute spectrum for each and average

**Strain Modeling (Analytical)**

Strain modeling achieves the same result analytically using first-order perturbation theory. Instead of sampling, MaRs calculates how parameter variations shift the energy levels directly.

.. code-block:: python

   import torch
   from mars import spin_model
   
   # Single spin system with strain parameters
   g = spin_model.Interaction(
       components=(2.002, 2.004, 2.008),
       strain=(0.001, 0.001, 0.002)  # FWHM of distributions
   )
   
   # Compute broadening analytically

For a transition between states :math:`|u\rangle` and :math:`|v\rangle`, the linewidth contribution depends on the variance of the energy difference.
MaRs computes this via the derivatives of the diagonal Hamiltonian elements (eigenenergies):

.. math::

   \Delta\nu_{uv}^{2} \propto \sum_{i} \left( \frac{\partial (H_{vv} - H_{uu})}{\partial p_i} \right)^2 \sigma_i^2

where:
- :math:`H_{vv}` and :math:`H_{uu}` are the diagonal energy elements for states :math:`|v\rangle` and :math:`|u\rangle`.
- :math:`p_i` represents the Hamiltonian parameters (e.g., :math:`g_x, g_y, g_z`).
- :math:`\sigma_i` is the standard deviation of the :math:`i`-th parameter.

This approach is equivalent to Monte Carlo sampling in the limit of small strain but requires only a single computation.

How Strain Works
----------------

For any interaction, strain is modeled as a Gaussian distribution of the principal values. In the case of :class:`mars.spin_model.Interaction`, strain is applied independently to :math:`T_x, T_y, T_z`. For :class:`mars.spin_model.DEInteraction`, strain is applied to the underlying :math:`D` and :math:`E` parameters.

During simulation, MaRs integrates over this distribution analytically (via first-order perturbation theory) when computing the spectrum, avoiding costly Monte Carlo sampling. The result is a broadened lineshape consistent with Gaussian disorder in the Hamiltonian parameters.

Specifying Strain
-----------------

Strain is passed as a vector with the same length as the components:

.. code-block:: python

   # Isotropic g-strain
   g = Interaction(2.0023, strain=0.001)

   # Anisotropic g-strain
   g = Interaction((2.02, 2.04, 2.06), strain=(0.01, 0.01, 0.02))

   # ZFS strain on D and E
   zfs = DEInteraction([500e6, 100e6], strain=[50e6, 10e6])

Units must match those of the components (e.g., Hz for couplings, dimensionless for g).

Important Notes
---------------

- Strain in MaRs is **uncorrelated by default** for :class:`mars.spin_model.Interaction` (diagonal covariance).
- Strain distributions are assumed Gaussian and static.
- MaRs allows you to set any type of correlation within a single interaction using the :meth:`set_strain` method: ``interaction.set_strain(new_strain, new_correlation_matrix)``

Custom Strain Correlations
--------------------------

MaRs allows you to define arbitrary linear correlations between the underlying physical parameters and the tensor components via the ``strain_correlation`` attribute.

Physical Motivation
~~~~~~~~~~~~~~~~~~~

Suppose you model a system where g-strain arises from two independent structural modes:

- A **symmetric compression** affecting all axes equally (isotropic shift)
- A **tetragonal distortion** affecting only :math:`g_z`

You can represent this with two strain parameters :math:`\delta s` (symmetric) and :math:`\delta t` (tetragonal), leading to:

.. math::

   \begin{bmatrix}
   \delta g_x \\ \delta g_y \\ \delta g_z
   \end{bmatrix}
   =
   \begin{bmatrix}
   1 & 0 \\
   1 & 0 \\
   1 & 1
   \end{bmatrix}
   \begin{bmatrix}
   \delta s \\ \delta t
   \end{bmatrix}

Mathematical Framework
~~~~~~~~~~~~~~~~~~~~~~

The transition frequency :math:`\nu_{uv}` is defined by the difference in diagonal energy elements:

.. math::

   \nu_{uv} = H_{vv} - H_{uu}

Consequently, the sensitivity of the transition to a physical parameter :math:`p` is the difference of the individual energy derivatives:

.. math::

   \frac{d \nu_{uv}}{dp} = \frac{\partial H_{vv}}{\partial p} - \frac{\partial H_{uu}}{\partial p}

In MaRs, strain broadening is calculated by propagating the uncertainty of underlying physical parameters :math:`p_\alpha` (such as structural modes) through this relation. The variance of the transition frequency is:

.. math::

   \sigma_{\nu, uv}^2 = \sum_{\alpha} \left( \frac{\partial H_{vv}}{\partial p_\alpha} - \frac{\partial H_{uu}}{\partial p_\alpha} \right)^2 \sigma_{p_\alpha}^2

Using the chain rule, we connect physical parameters :math:`p_\alpha` to the Hamiltonian components :math:`g_i`:

.. math::

   \frac{\partial H_{kk}}{\partial p_\alpha} = \sum_{i} \frac{\partial H_{kk}}{\partial g_i} \frac{\partial g_i}{\partial p_\alpha} \quad \text{for } k \in \{u, v\}

Here, :math:`\frac{\partial g_i}{\partial p_\alpha}` corresponds to the user-defined ``strain_correlation`` matrix.

Implementation in MaRs
~~~~~~~~~~~~~~~~~~~~~~

In MaRs, this is implemented as follows:

.. code-block:: python

   import torch
   from mars import spin_model

   # Define base g-tensor
   g = spin_model.Interaction(
       components=(2.002, 2.004, 2.008),
       strain=torch.tensor([0.001, 0.002])  # Two strain parameters: σ_s, σ_t
   )

   # Set custom correlation matrix C = ∂g_i / ∂p_α (shape: 3 × 2)
   C = torch.tensor([
       [1.0, 0.0],  # ∂g_x/∂s = 1, ∂g_x/∂t = 0
       [1.0, 0.0],  # ∂g_y/∂s = 1, ∂g_y/∂t = 0
       [1.0, 1.0]   # ∂g_z/∂s = 1, ∂g_z/∂t = 1
   ])
   g.set_strain(strain=torch.tensor([0.001, 0.002]), correlation_matrix=C)

Examples
--------

Example 1: Uncorrelated g-strains
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When g-tensor components vary independently (default behavior):

.. code-block:: python

   import torch
   from mars import spin_model

   # Three independent Gaussian distributions for gx, gy, gz
   g = spin_model.Interaction(
       components=(2.002, 2.004, 2.008),
       strain=(0.001, 0.001, 0.002)
   )

   # Correlation matrix is identity (3×3):
   # [[1, 0, 0],
   #  [0, 1, 0],
   #  [0, 0, 1]]
   print(g.strain_correlation)

This represents three independent strain sources, each affecting only one principal axis.

Example 2: Fully correlated g-strains
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When all g-tensor components shift together (isotropic strain):

.. code-block:: python

   import torch
   from mars import spin_model

   # Single strain parameter affects all components equally
   g = spin_model.Interaction(
       components=(2.002, 2.004, 2.008)
   )  # None strain parameter

   # Correlation matrix (3×1):
   # [[1],
   #  [1],
   #  [1]]
   C = torch.tensor([[1.0], [1.0], [1.0]])
   g.set_strain(strain=torch.tensor([0.001]), correlation_matrix=C)  # Set the necessary strain here

This means:

.. math::

   \begin{bmatrix}
   \delta g_x \\ \delta g_y \\ \delta g_z
   \end{bmatrix}
   =
   \begin{bmatrix}
   1 \\ 1 \\ 1
   \end{bmatrix}
   \delta s

All components fluctuate together with the same magnitude.
