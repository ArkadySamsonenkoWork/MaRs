Interaction
===========

The :class:`mars.spin_model.Interaction` class is the general-purpose container for symmetric second-rank tensor interactions in EPR spectroscopy.
It supports isotropic, axial, and orthorhombic symmetry and can be rotated into an arbitrary molecular frame via Euler angles or a rotation matrix.

Mathematical Form
-----------------

An interaction tensor **T** is defined by its principal values :math:`(T_x, T_y, T_z)` in its intrinsic frame. In the laboratory frame, it becomes:

.. math::

   \mathbf{T}^{\text{lab}} = \mathbf{R} \cdot \text{diag}(T_x, T_y, T_z) \cdot \mathbf{R}^\top,

where :math:`\mathbf{R}` is the rotation matrix derived from the specified Euler angles (zy'z'' convention) or the spherical angles ($\theta$, $\phi$).

Construction
------------

.. code-block:: python

   # Isotropic (scalar)
   J = Interaction(1e9)  # 1 GHz isotropic exchange

   # Axial: [A_perp, A_par]
   A_axial = Interaction([20e6, 70e6])  # MHz

   # Orthorhombic: [A_x, A_y, A_z]
   A_ortho = Interaction([20e6, 25e6, 70e6])

   # With frame rotation (Euler angles in radians)
   g_rot = Interaction((2.02, 2.04, 2.06), frame=[0.0, 0.7, 0.0])

   # With strain (distribution of principal values)
   g_strained = Interaction((2.02, 2.04, 2.06), strain=[0.01, 0.01, 0.02])

.. note::
   All values must be provided in SI-compatible units (Hz for couplings, dimensionless for g-tensors).

Addition and Interaction Summation
----------------------------------

Two :class:`mars.spin_model.Interaction` and :class:`mars.spin_model.DEInteraction`  instances can be added using the ``+`` operator. The result is a new interaction that represents the sum of their second-rank tensors in the laboratory frame.

.. code-block:: python

   # Isotropic (scalar)
   J = Interaction(1e9)  # 1 GHz isotropic exchange

   # dipolar interaction:
   dipolar = DEInteraction([-30e6, -40e6, 70e6])  # MHz

   # total interaction as sum of interactions
   total = J + dipolar

Frame convention
----------------

The molecular axes are written as :math:`(x,y,z)`. The interaction principal
axes are written as :math:`(x_p,y_p,z_p)`.

For Euler angles :math:`(\alpha,\beta,\gamma)` in the ``zy'z''`` convention, start
with the molecular frame and apply the intrinsic frame sequence

.. math::

   z(\alpha) \;\rightarrow\; y'(\beta) \;\rightarrow\; z''(\gamma)

until the frame coincides with the PAS.

In MaRs, the corresponding matrix maps PAS coordinates to molecular coordinates:

.. math::

   \mathbf{v}_{m}
   = \mathbf{R}_{m\leftarrow p}\,\mathbf{v}_{p}.

Strain handling
~~~~~~~~~~~~~~~

Strain parameters describe distributions of the principal values in the intrinsic frame. During addition:

- Strain vectors from each operand are concatenated.
- A combined ``strain_correlation`` matrix is built to map this concatenated strain vector onto perturbations of the final derivatives of the principal components :math:`\partial D_x, \partial D_y, \partial D_z`.
(see also :ref:`strain-management-mars`)

This mechanism works consistently even when combining a generic :class:`mars.spin_model.Interaction` with a :class:`mars.spin_model.DEInteraction`.

For example::

    # Isotropic exchange interaction with scalar strain
    exchange = Interaction(1.5e9, strain=0.5e9)  # 1.5 GHz, strain = 0.5 GHz

    # Zero-field splitting in D–E form with strain on D and E
    zfs = DEInteraction([400e6, 80e6], strain=[20e6, 5e6])  # D = 400 MHz, E = 80 MHz

    # Combine them
    total = exchange + zfs

    print("Principal values (Hz):", total.components)
    print("Strain correlation matrix:\n", total.strain_correlation)

The resulting ``strain_correlation`` matrix has shape ``(3, 5)``: three rows for the output principal components (Dx, Dy, Dz)
and five columns—three from the isotropic interaction (treated as independent perturbations along x, y, z)
and two from the D/E parameters via the transformation:

.. math::

   \begin{bmatrix}
   \delta D_x \\ \delta D_y \\ \delta D_z
   \end{bmatrix}
   =
   \begin{bmatrix}
   1 & 0 & 0 & -\frac{1}{3} & 1 \\
   0 & 1 & 0 & -\frac{1}{3} & -1 \\
   0 & 0 & 1 & \frac{2}{3} & 0
   \end{bmatrix}
   \begin{bmatrix}
   \delta J_x \\ \delta J_y \\ \delta J_z \\ \delta D \\ \delta E
   \end{bmatrix}