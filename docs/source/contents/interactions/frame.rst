Frame, Rotations and Euler Angles
=================================

In MaRs, the orientation of any interaction tensor relative to the laboratory frame is defined by a rotation, which can be specified either as Euler angles or as a full 3×3 rotation matrix.

Euler Angle Convention
----------------------

MaRs uses the ZYZ' (proper Euler) convention with angles :math:`(\alpha, \beta, \gamma)` in radians, applied in the following order:

1. Rotation by :math:`\alpha` around the lab Z-axis,
2. Rotation by :math:`\beta` around the new Y'-axis,
3. Rotation by :math:`\gamma` around the new Z''-axis.

The total rotation matrix is:

.. math::

   \mathbf{R}(\alpha, \beta, \gamma) = \mathbf{R}_z(\gamma)\, \mathbf{R}_y(\beta)\, \mathbf{R}_z(\alpha).

This is the standard convention used in magnetic resonance for describing molecular orientations in powders or single crystals, and it matches the convention used in EasySpin and many quantum chemistry packages.

How Rotations Act on Tensors
----------------------------

An interaction tensor :math:`\mathbf{T}` is defined in its principal axis system (PAS), where it is diagonal:

.. math::

   \mathbf{T}_\text{PAS} = \operatorname{diag}(T_x, T_y, T_z).

To express this tensor in the laboratory frame, it is rotated using the rotation matrix :math:`\mathbf{R}`:

.. math::

   \mathbf{T}_\text{lab} = \mathbf{R}\, \mathbf{T}_\text{PAS}\, \mathbf{R}^\top.

Specifying Orientation at Initialization
----------------------------------------

When constructing an :class:`mars.spin_model.Interaction` or :class:`mars.spin_model.DEInteraction`, the ``frame`` argument accepts:

* ``None`` → identity (tensor aligned with lab frame),
* A sequence ``[α, β, γ]`` (in radians),
* A 3×3 rotation matrix (``torch.Tensor``).


Example:

.. code-block:: python

   import math
   
   # Rotate g-tensor: 40° around y, then 30° around new z
   g = Interaction((2.0, 2.0, 2.1), frame=[0.0, math.radians(40), math.radians(30)])

Applying Rotations After Initialization
---------------------------------------

After an interaction or a spin system has been created, you can rotate it further using the ``apply_rotation()`` method. This updates the internal rotation matrix and Euler angles accordingly.

For a single interaction:

.. code-block:: python

   import torch
   from mars import utils
   
   angles = torch.tensor([0.1, 0.2, 0.3]) # [α, β, γ] in radians
   rotation_matrix = utils.euler_angles_to_matrix(angles)
   
   dipolar_interaction.apply_rotation(rotation_matrix) # Rotate just this interaction

For an entire spin system:

.. code-block:: python

   base_spin_system.apply_rotation(rotation_matrix) # Rotate spin system relative to the laboratory frame

Orientation in Sample Construction
----------------------------------

When creating a sample, you can also specify the orientation of the entire spin system relative to the lab frame via the ``spin_system_frame`` argument in :class:`mars.spin_model.MultiOrientedSample` or :class:`mars.spin_model.BaseSample`.

Example:

.. code-block:: python

   sample = spin_model.MultiOrientedSample(
       spin_system_frame=rotation_matrix,  # or use angles directly
       base_spin_system=base_spin_system,
       ham_strain=5e7,
       gauss=0.001,
       lorentz=0.001,
   )