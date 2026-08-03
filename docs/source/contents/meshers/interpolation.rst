Interpolation on Spherical Meshes
=================================

MaRs supports interpolation of the three resonance data: **resonance fields**, **intensities**, and **linewidths** — from a coarse initial orientation grid to a finer evaluation grid.
This decouples the expensive Hamiltonian diagonalization (performed on the coarse grid) from the high-resolution spherical integration required for smooth spectra.

Two interpolation strategies are provided:

1. **Radial Basis Function (RBF) Interpolation**
2. **Spherical Barycentric Interpolation**

These are implemented in :class:`mars.mesher.delaunay_neighbour.RBFInterpolator` and :class:`mars.mesher.delaunay_neighbour.BarycentricInterpolator`.

When ``interpolate=True`` in :class:`mars.mesher.delaunay_neighbour.DelaunayMesh`, all three resonance quantities are interpolated onto the extended mesh before triangulation and integration.

Technical Details
-----------------

Interpolation Procedures
~~~~~~~~~~~~~~~~~~~~~~~~

Given resonance data on the base mesh:

- **Fields** :math:`B_{\text{res}}(\omega_i)` — resonance magnetic field at orientation :math:`\omega_i`
- **Intensities** :math:`I(\omega_i)` — transition intensity
- **Widths** :math:`\Delta B(\omega_i)` — linewidth

All three tensors are interpolated independently to the extended mesh vertices :math:`\{\omega'_j\}`:

.. math::

   B'_{\text{res}}(\omega'_j) = \mathcal{I}[B_{\text{res}}](\omega'_j), \quad
   I'(\omega'_j) = \mathcal{I}[I](\omega'_j), \quad
   \Delta B'(\omega'_j) = \mathcal{I}[\Delta B](\omega'_j),

where :math:`\mathcal{I}` denotes the chosen interpolator.

RBFInterpolator
~~~~~~~~~~~~~~~

Performs global interpolation using radial basis functions defined on the unit sphere.
For each resonance quantity, it solves the linear system:

.. math::

   (\mathbf{K} + \lambda \mathbf{R}) \boldsymbol{\alpha} = \mathbf{f},

where:
- :math:`\mathbf{K}_{mn} = \phi(d(\Omega_m, \Omega_n))` is the kernel matrix computed from geodesic (great-circle) distances between base vertices,
- :math:`\mathbf{R}` is a regularization matrix (see below),
- :math:`\lambda` is a small jitter term for numerical stability,
- :math:`\boldsymbol{\alpha}` are the RBF coefficients.

Interpolated values at extended points are then computed as:

.. math::

   \mathbf{f}' = \mathbf{K}_{\text{ext}} \boldsymbol{\alpha},

where :math:`\mathbf{K}_{\text{ext}}` contains kernel evaluations between extended and base vertices.

Supported kernels include:
``gaussian``, ``multiquadric``, ``inverse_multiquadric``, ``linear``, ``cubic``, and ``thin_plate``.

Regularization options:
- ``"tikhonov"``: Adds :math:`\lambda \mathbf{I}` (standard ridge regularization).
- ``"spherical"``: Uses a smoother prior based on low-order spherical harmonics (up to degree 5 by default).

BarycentricInterpolator
~~~~~~~~~~~~~~~~~~~~~~~

Performs local interpolation using true spherical barycentric coordinates.
The base mesh must be triangulated; for each target point, the algorithm:

1. Finds the spherical triangle containing the point (using oriented great-circle half-spaces),
2. Computes barycentric weights based on spherical triangle areas:
   
   .. math::
   
      w_A = \frac{\text{area}(P, B, C)}{\text{area}(A, B, C)}, \quad \text{etc.}

3. Linearly combines vertex values using these weights.

This method is geometrically exact for linear fields on the sphere and naturally respects mesh topology.

Configuring Interpolators via ``interpolator_kwargs``
-----------------------------------------------------

The ``interpolator_kwargs`` parameter allows fine-grained control over interpolator behavior.
It is passed directly to the interpolator constructor and supports the following options:

For :class:`RBFInterpolator`:
- ``kernel`` (str): Choice of RBF kernel (default: ``"thin_plate"``).
- ``regularization`` (str): ``"spherical"`` or ``"tikhonov"`` (default: ``"spherical"``).
- ``jitter`` (float): Small positive value added to diagonal for stability (default: ``1e-6``).
- ``epsilon`` (float): Shape parameter for kernels that require it (default: ``1.0``).

For :class:`BarycentricInterpolator`:
- ``tol`` (float): Numerical tolerance for point-in-triangle tests (default: ``1e-10``).

Example usage in :class:`DelaunayMesh`:

.. code-block:: python

   mesh = DelaunayMesh(
       interpolate=True,
       interpolator="rbf",
       interpolator_kwargs={
           "kernel": "gaussian",
           "regularization": "tikhonov",
           "jitter": 1e-5
       }
   )