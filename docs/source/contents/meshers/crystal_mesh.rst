Crystal Mesh: CrystalMesh
=========================

The :class:`mars.mesher.general_mesh.CrystalMesh` class represents one or more
fixed molecular/crystal orientations relative to the laboratory frame.

The Euler angles ``[alpha, beta, gamma]`` describe the molecular frame
``(x, y, z)`` relative to the laboratory frame ``(X, Y, Z)``. For the default
``"zyz"`` convention, start with the laboratory frame and apply the intrinsic
sequence

.. math::

   z(\alpha) \rightarrow y'(\beta) \rightarrow z''(\gamma)

to obtain the molecular/crystal frame.

The resulting matrix ``R`` maps molecular-frame coordinates to laboratory-frame
coordinates:

.. math::

   \mathbf{v}_{L} = R\,\mathbf{v}_{m}.

Key Features
------------

- Accepts a tensor of Euler angles (in radians) of shape ``(..., 3)``.
- Converts Euler angles to rotation matrices using a specified convention (default: zy'z'')

Usage
-----

.. code-block:: python

   euler = torch.tensor([[0.0, 0.0, 0.0],    # z-axis
                         [np.pi/2, 0.0, 0.0], # x-axis
                         [np.pi/2, np.pi/2, 0.0]]) # y-axis
   mesh = mesher.CrystalMesh(euler_angles=euler, device=device)

Each row specifies one fixed molecular/crystal orientation relative to the
laboratory frame.
