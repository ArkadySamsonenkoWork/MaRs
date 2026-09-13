Axial Mesh: AxialMesh
==============================

The :class:`mars.mesher.axial_mesh.AxialMesh` class is intended for systems with
axial symmetry, where observables depend only on the polar angle ``theta`` and
are independent of the azimuthal angle.


Key Features
------------

- Samples only the polar angle ``theta`` in ``[0, pi/2]`` using axial symmetry.
- Constructs line segments for integration.
- Does not use interpolation.
- Uses matrices of the form ``R = R_y(theta)``.

Usage
-----

.. code-block:: python

   mesh = mesher.AxialMesh(
       initial_grid_frequency=50,
       device=device,
       dtype=torch.float64
   )

Use this mesh when the spin system is axially symmetric and has no relevant
azimuthal dependence.

Mathematical Notes
------------------

The integration weight for a segment ``[theta_i, theta_{i+1}]`` is proportional to

.. math::

   2\pi\left(\cos\theta_i - \cos\theta_{i+1}\right),

which is the surface area of the corresponding spherical zone.