"""
mars.mesher — Orientation meshing for powder averaging and single-crystal simulations.

This module generates and manages orientation meshes for EPR simulations:
    - Axial meshes (grids on a sphere with axial symmetry)
    - Delaunay meshes (sphere triangulation)
    - Full-sphere meshes (uniform coverage of 4π)
    - 3D meshes (full orientation space)
    - Interpolation between mesh points

Main classes:
    AxialMesh               — Axially-symmetric mesh (θ, φ grid)
    DelaunayMesh            — Delaunay triangulation of the sphere
    DelaunayMeshFullSphere  — Full 4π coverage
    Mesh3D                  — 3D orientation mesh
    CrystalMesh             — Mesh with crystal symmetry

Example:
    >>> from mars.mesher import AxialMesh, DelaunayMesh
    >>>
    >>> # Axial mesh
    >>> mesh = AxialMesh()
    >>>
    >>> # Delaunay mesh
    >>> mesh = DelaunayMesh(n_points=500)
"""

from .general_mesh import BaseMesh, BaseMeshPowder, CrystalMesh
from .delanay_mesh import DelaunayMesh, DelaunayMeshFullSphere
from .axial_mesh import AxialMesh

from .experimental import Mesh3D