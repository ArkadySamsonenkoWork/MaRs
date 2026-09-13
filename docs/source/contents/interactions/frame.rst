Frame, Rotations and Euler Angles
=================================

MaRs uses passive coordinate transformations to relate coordinate frames. A
rotation matrix changes the coordinates used to represent the same physical
vector or tensor; it does not imply a physical rotation of the object.

Frame notation
--------------

The laboratory axes are written with capital letters :math:`(X,Y,Z)`. The
molecular axes are written with lowercase letters :math:`(x,y,z)`. An
interaction principal-axis system (PAS) is denoted by
:math:`(x_p,y_p,z_p)`.

For a transformation matrix :math:`\mathbf{R}` from a source frame to a target
frame,

.. math::

   \mathbf{v}_{\mathrm{target}}
   = \mathbf{R}\,\mathbf{v}_{\mathrm{source}}.

The same second-rank tensor is represented in the target frame as

.. math::

   \mathbf{T}_{\mathrm{target}}
   = \mathbf{R}\,\mathbf{T}_{\mathrm{source}}\,\mathbf{R}^{\mathsf T}.


Euler-angle convention
----------------------

Euler angles :math:`(\alpha,\beta,\gamma)` describe the orientation of one
frame relative to another. For the ``zy'z''`` convention, start with the reference
frame and apply the intrinsic frame sequence

.. math::

   z(\alpha) \;\rightarrow\; y'(\beta) \;\rightarrow\; z''(\gamma)

until the moving frame coincides with the final frame.

For the molecular frame relative to the laboratory frame, the reference axes
are :math:`(X,Y,Z)` and the final axes are :math:`(x,y,z)`. Thus the sequence is

.. math::

   Z(\alpha) \;\rightarrow\; Y'(\beta) \;\rightarrow\; Z''(\gamma),

and the resulting matrix is

.. math::

   \mathbf{R}_{L\leftarrow m}
   = \mathbf{R}_Z(\alpha)\,
     \mathbf{R}_Y(\beta)\,
     \mathbf{R}_Z(\gamma).

Its columns are the molecular axes expressed in laboratory coordinates, and it
maps molecular coordinates to laboratory coordinates:

.. math::

   \mathbf{v}_{L}
   = \mathbf{R}_{L\leftarrow m}\,\mathbf{v}_{m}.

Interaction frame
-----------------

For :class:`mars.spin_model.Interaction` and
:class:`mars.spin_model.DEInteraction`, ``frame`` describes the orientation of
the interaction PAS relative to the molecular frame.

Starting with the molecular axes :math:`(x,y,z)`, the intrinsic ``zy'z''`` Euler
sequence gives the principal axes :math:`(x_p,y_p,z_p)`. The corresponding
matrix maps PAS coordinates to molecular coordinates:

.. math::

   \mathbf{v}_{m}
   = \mathbf{R}_{m\leftarrow p}\,\mathbf{v}_{p}.

Therefore a diagonal PAS tensor

.. math::

   \mathbf{T}_{p} = \operatorname{diag}(T_x,T_y,T_z)

is represented in the molecular frame as

.. math::

   \mathbf{T}_{m}
   = \mathbf{R}_{m\leftarrow p}\,
     \mathbf{T}_{p}\,
     \mathbf{R}_{m\leftarrow p}^{\mathsf T}.

The ``frame`` argument accepts:

* ``None`` — the principal axes coincide with the molecular axes,
* Euler angles ``[alpha, beta, gamma]`` in radians,
* a ``3 x 3`` matrix that maps PAS coordinates to molecular coordinates.

Example:

.. code-block:: python

   import math
   from mars.spin_model import Interaction

   g = Interaction(
       (2.0, 2.0, 2.1),
       frame=[0.0, math.radians(40), math.radians(30)],
   )

Applying a coordinate-frame transformation
------------------------------------------

:meth:`mars.spin_model.Interaction.apply_rotation` and
:meth:`mars.spin_model.SpinSystem.apply_rotation` are interpreted passively.
The supplied matrix maps coordinates from the current frame to a target frame:

.. math::

   \mathbf{v}_{\mathrm{target}}
   = \mathbf{R}_{\mathrm{target}\leftarrow\mathrm{current}}
     \mathbf{v}_{\mathrm{current}}.

Accordingly, an interaction tensor is represented in the target frame as

.. math::

   \mathbf{T}_{\mathrm{target}}
   = \mathbf{R}_{\mathrm{target}\leftarrow\mathrm{current}}
     \mathbf{T}_{\mathrm{current}}
     \mathbf{R}_{\mathrm{target}\leftarrow\mathrm{current}}^{\mathsf T}.

Example:

.. code-block:: python

   import torch
   from mars import utils

   angles = torch.tensor([0.1, 0.2, 0.3])
   R_target_from_current = utils.euler_angles_to_matrix(angles)

   dipolar_interaction.apply_rotation(R_target_from_current)
   base_spin_system.apply_rotation(R_target_from_current)

Sample, molecular and laboratory frames
---------------------------------------

For orientation-dependent samples it is useful to distinguish three frames:

* molecular frame :math:`(x,y,z)`,
* sample or crystal reference frame,
* laboratory frame :math:`(X,Y,Z)`.

The ``molecular_frame`` argument of :class:`mars.spin_model.BaseSample` and
:class:`mars.spin_model.SolidSample` defines the molecular frame
relative to the sample/reference frame. Its matrix satisfies

.. math::

   \mathbf{v}_{s}
   = \mathbf{R}_{s\leftarrow m}\,\mathbf{v}_{m}.

The orientation mesh defines the sample/reference frame relative to the
laboratory frame:

.. math::

   \mathbf{v}_{L}
   = \mathbf{R}_{L\leftarrow s}\,\mathbf{v}_{s}.

The effective molecular-to-laboratory transformation is therefore

.. math::

   \mathbf{R}_{L\leftarrow m}
   = \mathbf{R}_{L\leftarrow s}\,
     \mathbf{R}_{s\leftarrow m}.

If ``molecular_frame=None``, the molecular and sample/reference frames
coincide.

Example:

.. code-block:: python

   sample = spin_model.SolidSample(
       base_spin_system=base_spin_system,
       molecular_frame=[0.0, 0.2, 0.0],
       ham_strain=5e7,
       gauss=0.001,
       lorentz=0.001,
   )
