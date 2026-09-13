Powder Mesh: DelaunayMesh
==================================

The :class:`mars.mesher.delaunay_mesh.DelaunayMesh` class implements a spherical
triangulation for disordered (powder) samples, where molecular orientations are
sampled over the sphere based on an adaptive grid in spherical coordinates (θ, φ). 

Key Features
------------

- Supports full-sphere or restricted φ ranges via ``phi_limits``.
- Optional interpolation from a coarse initial grid to a finer evaluation grid.
- Provides molecular-to-laboratory transformation matrices for all sampled orientations.
- Provides spherical triangle areas for proper integration weighting.

Usage
-----

Typical initialization:

.. code-block:: python

   mesh = mesher.DelaunayMesh(
       initial_grid_frequency=30,
       interpolate=True,
       interpolation_grid_frequency=60,
       phi_limits=(0, 2 * np.pi)
       device=device,
       dtype=torch.float64
   )

The mesh is used internally by :class:`mars.spin_model.MultiOrientedSample` to average
the spectrum over orientations.