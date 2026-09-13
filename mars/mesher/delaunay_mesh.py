from __future__ import annotations

import abc
import numpy as np
import math
import typing as tp
from abc import ABC
from enum import Enum

from scipy.special import sph_harm_y

import torch    
import torch.nn as nn

from sklearn.metrics import pairwise_distances
from .general_mesh import BaseMeshPowder


class BoundaryHandler:
    """Handles boundary condition logic."""
    @staticmethod
    def get_boundary(boundary: str | None, init_indexes: list[int], is_start: bool):
        if boundary == "reflection":
            offsets = (2, 1) if is_start else (-3, -2)
        elif boundary == "a":
            offsets = (-3, -2) if is_start else (1, 2)
        elif boundary is None:
            return []
        else:
            raise ValueError("Invalid phi boundary condition.")
        return [init_indexes[offset] for offset in offsets]


class ThetaLine:
    """Represents a line of constant theta with variable phi sampling.

    Used during mesh construction to generate vertices with adaptive density
    based on spherical geometry requirements.
    """
    def __init__(self, theta: float, points: int, phi_limits: tuple[float, float],
                 last_point: bool):
        """Initialize theta line with sampling parameters.

        :param theta: Polar angle in radians [0, pi]
        :type theta: float
        :param points: Number of latent sampling points along phi
        :type points: int
        :param phi_limits: (phi_min, phi_max) bounds in radians
        :type phi_limits: tuple[float, float]
        :param last_point: Whether to include the endpoint at phi_max
        :type last_point: bool
        """
        self.phi_limits = phi_limits
        self.theta = theta
        self.latent_points = points
        self.last_point = last_point

    def _compute_visible_points(self):
        """Compute number of visible points after boundary handling.

        :return: Number of points to actually generate
        :rtype: int
        """
        if self.last_point:
            return self.latent_points
        return self.latent_points if self.latent_points == 1 else self.latent_points - 1

    def phi_theta(self):
        """Generate (phi, theta) coordinate pairs for this theta line.

        :return: List of (phi, theta) tuples in radians
        :rtype: list[tuple[float, float]]
        """
        if self.latent_points == 1:
            return [(0.0, 0.0)]
        delta_phi = self.phi_limits[1] - self.phi_limits[0]
        if self.last_point:
            return [(self.phi_limits[0] + point * delta_phi / (self.latent_points - 1), self.theta) for point in
                    range(self.latent_points)]
        else:
            return [(self.phi_limits[0] + point * delta_phi / (self.latent_points - 1), self.theta) for point in
                    range(self.latent_points - 1)]


class BaseInterpolator(nn.Module, ABC):
    """
    Abstract base class for spherical mesh interpolators.

    Defines the interface for interpolating scalar fields from a coarse base mesh
    to a finer target mesh on the unit sphere. All concrete interpolators must
    implement the forward pass and provide a transformation matrix.
    """
    @abc.abstractmethod
    def __init__(self, init_vertices: tp.Union[list[tuple[float, float]], np.ndarray],
                 extended_vertices: tp.Union[list[tuple[float, float]], np.ndarray],
                 device: torch.device, dtype: torch.dtype, *args, **kwargs):
        """
        Initialize the interpolator with source and target vertex sets.

        :param init_vertices: Base mesh vertices as (phi, theta) pairs.
            Shape (N, 2) with phi ∈ [0, 2π), theta ∈ [0, π/2].
        :type init_vertices: list[tuple[float, float]] | np.ndarray
        :param extended_vertices: Target vertices for interpolation.
            Shape (M, 2) with phi ∈ [0, 2π), theta ∈ [0, π/2].
        :type extended_vertices: list[tuple[float, float]] | np.ndarray
        :param device: Computation device for storing tensors
        :type device: torch.device
        :param dtype: Floating point precision for computations
        :type dtype: torch.dtype
        :param args: Additional positional arguments for subclasses
        :param kwargs: Additional keyword arguments for subclasses
        """
        super().__init__()

    def _to_cartesian(self, vertices: np.ndarray) -> np.ndarray:
        """
        Convert spherical coordinates (phi, theta) to Cartesian unit vectors.

        Uses physics convention:
          - phi: azimuthal angle in xy-plane from x-axis [0, 2π)
          - theta: polar angle from positive z-axis [0, π]

        :param vertices: Array of shape (K, 2) containing (phi, theta) in radians
        :type vertices: np.ndarray
        :return: Normalized Cartesian vectors of shape (K, 3) on unit sphere
        :rtype: np.ndarray
        """
        phi = vertices[:, 0]
        theta = vertices[:, 1]
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        v = np.stack([x, y, z], axis=1)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def _to_lat_long(self, vertices: list[tuple[float, float]]) -> np.ndarray:
        """
        Convert (phi, theta) spherical coordinates to latitude-longitude format.

        Transforms:
          - longitude = phi
          - latitude = π/2 - theta

        :param vertices: List of (phi, theta) pairs in radians
        :type vertices: list[tuple[float, float]]
        :return: Array of shape (N, 2) with [latitude, longitude] in radians
        :rtype: np.ndarray
        """
        vertices = np.array(vertices)[:, ::-1]
        vertices[:, 0] = np.pi / 2 - vertices[:, 0]
        return np.array(vertices)

    @abc.abstractmethod
    def forward(self, f_values: torch.Tensor) -> torch.Tensor:
        """Interpolate values at extended points using RBF interpolation.

        :param f_values: Tensor of shape (..., N), where N = number of base vertices.
        :return: Interpolated values of shape (..., M), where M = number of extended vertices.
        """
        pass

    @property
    @abc.abstractmethod
    def transform_matrix(self) -> torch.Tensor:
        """
        Get the linear transformation matrix for interpolation.

        The matrix W satisfies: f_extended = f_base @ W
        where W has shape (N, M), N = base vertices, M = extended vertices.

        :return: Transformation matrix of shape (N, M)
        :rtype: torch.Tensor
        """
        pass


class RBFInterpolator(BaseInterpolator):
    """Radial Basis Function interpolator for spherical mesh refinement with symmetry awareness.

    Performs interpolation from a coarse base mesh to a finer extended mesh on the unit sphere
    using geodesic (great-circle) distances and robust regularization.
    """
    def __init__(self,
                 init_vertices: tp.Union[list[tuple[float, float]], np.ndarray],
                 extended_vertices: tp.Union[list[tuple[float, float]], np.ndarray],
                 kernel: str = "cubic",
                 regularization="tikhonov",
                 jitter: float = 1e-5,
                 epsilon: float = 1e-8,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32):
        """Initialize RBF interpolator with spherical geometry.

        :param init_vertices: Base mesh vertices as (phi, theta) pairs on upper hemisphere.
            Shape (N, 2) with phi ∈ [0, 2π), theta ∈ [0, π/2].
        :type init_vertices: list[tuple[float, float]] | np.ndarray
        :param extended_vertices: Target vertices for interpolation.
            Shape (M, 2). with phi ∈ [0, 2π), theta ∈ [0, π/2].

        :type extended_vertices: list[tuple[float, float]] | np.ndarray
        :param kernel: RBF kernel type. Default is  "thin_plate". It supporst also
            "gaussian", "multiquadric", "inverse_multiquadric", "linear", "cubic".
        :type kernel: str

        :param regularization: Regularization method. Options: "spherical", "tikhonov"
        :type regularization: str
        :param jitter: Small value is multiplied on regularization
        :type jitter: float

        :param epsilon: Shape parameter controlling kernel width. Larger values → smoother
            interpolation; smaller values → tighter fit (risk of overfitting/instability).
            For "thin_plate", typically set to 1.0 (scale-invariant).
        :type epsilon: float
        :param device: Computation device for transform matrix storage
        :type device: torch.device
        :param dtype: Floating point precision
        :type dtype: torch.dtype
        :raises ValueError: For unsupported kernel types or invalid vertex configurations
        """
        super().__init__(init_vertices, extended_vertices, device, dtype)
        self.kernel = kernel
        self.epsilon = epsilon

        base = self._to_cartesian(init_vertices)
        extended = self._to_cartesian(extended_vertices)

        base = base
        extended = extended

        dists = pairwise_distances(base, base)
        K = self._rbf(dists)

        dists_ext = pairwise_distances(extended, base)
        K_ext = self._rbf(dists_ext)
        K_torch = torch.tensor(K, dtype=dtype)
        K_ext_torch = torch.tensor(K_ext, dtype=dtype)

        if regularization == "spherical":
            regularizer = torch.from_numpy(self._spherical_harmonic_regularization(init_vertices)).to(dtype)
        elif regularization == "tikhonov":
            regularizer = torch.eye(K_torch.shape[-1], dtype=dtype)
        else:
            raise NotImplementedError("Currently only spherical and tikhonov regularization is supported")
        max_lambda = self._estimate_lambda_max(K_torch)
        K_torch.add_(jitter * regularizer * max_lambda)
        W = torch.linalg.lstsq(K_torch, K_ext_torch.mT).solution

        self.register_buffer("_transform_matrix", W.to(device))
        #self.register_buffer("_transform_matrix", torch.matmul(K_inv, K_ext_torch.mT).to(device))

    def _estimate_lambda_max(self, K_torch: torch.Tensor):
        return torch.linalg.matrix_norm(K_torch, ord=2)

    def _spherical_harmonic_regularization(self, vertices: np.ndarray, max_l=5) -> np.ndarray:
        """Use spherical harmonics as a smoother prior."""
        phi = vertices[:, 0]
        theta = vertices[:, 1]
        n_points = len(vertices)
        R = np.zeros((n_points, n_points))

        for l in range(max_l + 1):
            if l % 2 != 0:
                continue
            for m in range(-l, l + 1):
                Y_lm = sph_harm_y(m, l, phi, theta)
                Y_col = Y_lm.reshape(-1, 1)
                R += np.real(Y_col @ Y_col.conj().T) / (l * (l + 1) + 1)
        R = R * (np.trace(R) / (n_points * np.max(np.diag(R)) + self.epsilon))
        return R

    def _rbf(self, r: np.ndarray) -> np.ndarray:
        """Radial basis functions."""
        eps = self.epsilon
        if self.kernel == "gaussian":
            return np.exp(-(eps * r) ** 2)
        elif self.kernel == "multiquadric":
            return np.sqrt(1.0 + (eps * r) ** 2)
        elif self.kernel == "inverse_multiquadric":
            return 1.0 / np.sqrt(1.0 + (eps * r) ** 2)
        elif self.kernel == "linear":
            return r
        elif self.kernel == "cubic":
            return r ** 3
        elif self.kernel == "thin_plate":
            return np.where(r < eps, np.zeros_like(r), (r ** 2) * np.log(r))
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    @property
    def transform_matrix(self) -> torch.Tensor:
        """
        Return the matrix with the shape [N, M] which transforms initial function to extended function
        :return: transformation matrix
        """
        return self._transform_matrix

    def forward(self, f_values: torch.Tensor) -> torch.Tensor:
        """Interpolate values at extended points using RBF interpolation.

        :param f_values: Tensor of shape (..., N), where N = number of base vertices.
        :return: Interpolated values of shape (..., M), where M = number of extended vertices.
        """
        return torch.matmul(f_values, self._transform_matrix)


class BarycentricInterpolator(BaseInterpolator):
    """
    Spherical barycentric interpolator for data defined on a triangulated unit sphere.

    This class interpolates values from an initial spherical mesh to a new set of
    spherical points using true spherical barycentric coordinates.

    The interpolation process consists of:
        1. Converting spherical coordinates (phi, theta) to 3D Cartesian unit vectors.
        2. Finding, for each target point, the spherical triangle that contains it.
        3. Computing spherical barycentric coordinates using spherical triangle areas.
    """
    def __init__(self,
                 init_vertices: tp.Union[list[tuple[float, float]], np.ndarray],
                 extended_vertices: tp.Union[list[tuple[float, float]], np.ndarray],
                 init_triangulation: np.ndarray,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32,
                 tol: float = 1e-10):
        """
        Initialize the spherical barycentric interpolator.

        :param init_vertices:
            Initial mesh vertices as (phi, theta) pairs in radians.
            Shape (N, 2), where phi ∈ [0, 2π), theta ∈ [0, π].
        :param extended_vertices:
            Target vertices for interpolation as (phi, theta) pairs.
            Shape (M, 2).
        :param init_triangulation:
            Triangle connectivity array of shape (T, 3), containing indices
            into `init_vertices`.
        :param device:
            PyTorch device on which the interpolation matrix is stored.
        :param dtype:
            Floating-point precision for interpolation weights.
        :param tol:
            Numerical tolerance used for spherical point-in-triangle tests.
        """
        super().__init__(init_vertices, extended_vertices, device=device, dtype=dtype)
        self.init_vertices = np.array(init_vertices)
        self.extended_vertices = np.array(extended_vertices)
        self.init_triangulation = np.array(init_triangulation, dtype=int)

        self.init_cartesian = self._to_cartesian(init_vertices)
        self.extended_cartesian = self._to_cartesian(extended_vertices)

        tri_vertices_cart = self.init_cartesian[init_triangulation]
        self.tri_vertices_cart = self._orient_triangles(tri_vertices_cart)

        self.tol = tol
        self.register_buffer("_transform_matrix", self._compute_barycentric_weights(device, dtype))

    def _orient_triangles(self, triangles: np.ndarray) -> np.ndarray:
        """
        Ensure consistent orientation of spherical triangles.

        Each triangle is oriented such that the great-circle normal of edge (a, b)
        satisfies:
            dot(a × b, c) >= 0

        This guarantees consistent half-space tests during spherical
        point-in-triangle checks.

        :param triangles:
            Array of shape (T, 3, 3) containing Cartesian triangle vertices.
        :return:
            Array of shape (T, 3, 3) with consistently oriented triangles.
        """
        a = triangles[:, 0, :]
        b = triangles[:, 1, :]
        c = triangles[:, 2, :]
        n_ab = np.cross(a, b)
        check = np.einsum("ij,ij->i", n_ab, c)
        flip_mask = check < 0
        if np.any(flip_mask):
            triangles[flip_mask, 1, :], triangles[flip_mask, 2, :] = \
                triangles[flip_mask, 2, :].copy(), triangles[flip_mask, 1, :].copy()
        return triangles

    def _find_containing_triangles_spherical(self) -> tp.Tuple[np.ndarray, np.ndarray]:
        """
        Find the containing spherical triangle for each target point.

        A point is considered inside a spherical triangle if it lies on the
        same side of all three great-circle edges. The test is performed
        using oriented edge normals and dot products.

        If a point is not found inside any triangle (due to numerical
        precision or boundary cases), a fallback selects the triangle whose
        centroid has the smallest angular distance to the point.

        :return:
            triangle_indices:
                Integer array of shape (M,) giving the triangle index
                associated with each target point.
            triangles:
                Array of shape (M, 3, 3) containing the Cartesian vertices
                of the selected triangles.
        """
        P = self.extended_cartesian
        TR = self.tri_vertices_cart
        T = TR.shape[0]
        M = P.shape[0]

        a = TR[:, 0, :]
        b = TR[:, 1, :]
        c = TR[:, 2, :]

        n_ab = np.cross(a, b)
        n_bc = np.cross(b, c)
        n_ca = np.cross(c, a)

        n_ab = n_ab / np.linalg.norm(n_ab, axis=1, keepdims=True)
        n_bc = n_bc / np.linalg.norm(n_bc, axis=1, keepdims=True)
        n_ca = n_ca / np.linalg.norm(n_ca, axis=1, keepdims=True)

        P_bt = P[:, None, :]

        d1 = np.einsum("mij,tj->mt", P_bt, n_ab)
        d2 = np.einsum("mij,tj->mt", P_bt, n_bc)
        d3 = np.einsum("mij,tj->mt", P_bt, n_ca)

        tol = self.tol
        inside_mask = (d1 >= -tol) & (d2 >= -tol) & (d3 >= -tol)  # (M, T)

        triangle_indices = -1 * np.ones(M, dtype=int)
        found_any = np.any(inside_mask, axis=1)
        rows, cols = np.where(inside_mask)
        if rows.size > 0:
            for m in range(M):
                tri_candidates = np.where(inside_mask[m])[0]
                if tri_candidates.size > 0:
                    triangle_indices[m] = tri_candidates[0]

        if not np.all(found_any):
            not_found = np.where(~found_any)[0]
            centroids = np.mean(TR, axis=1)  # (T,3)
            centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
            dot_pc = P[not_found][:, None, :] @ centroids.T[None, :, :]
            dot_pc = dot_pc.squeeze(-1)
            nearest_tri = np.argmax(dot_pc, axis=1)
            triangle_indices[not_found] = nearest_tri

        triangles = TR[triangle_indices]  # (M,3,3)
        return triangle_indices, triangles

    def _spherical_triangle_area(self, u: np.ndarray, v: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        Compute the area of spherical triangles on the unit sphere.

        The area is computed using the numerically stable formula:

            area = 2 * atan2( |u · (v × w)|,
                              1 + u·v + v·w + w·u )

        All input vectors must be unit vectors.

        :param u:
            Array of shape (..., 3) representing the first vertex.
        :param v:
            Array of shape (..., 3) representing the second vertex.
        :param w:
            Array of shape (..., 3) representing the third vertex.
        :return:
            Array of spherical triangle areas with shape matching the
            broadcasted leading dimensions.
        """

        cross_vw = np.cross(v, w)
        num = np.abs(np.einsum("...i,...i->...", u, cross_vw))
        dot_uv = np.einsum("...i,...i->...", u, v)
        dot_vw = np.einsum("...i,...i->...", v, w)
        dot_wu = np.einsum("...i,...i->...", w, u)
        denom = 1.0 + dot_uv + dot_vw + dot_wu
        return 2.0 * np.arctan2(num, denom)

    def _spherical_barycentric_coords(self, points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
        """
        Compute spherical barycentric coordinates for multiple points.

        For each point P and triangle (A, B, C), the barycentric weights are:

            w_A = area(P, B, C) / area(A, B, C)
            w_B = area(P, C, A) / area(A, B, C)
            w_C = area(P, A, B) / area(A, B, C)

        where all areas are spherical triangle areas.

        :param points:
            Array of shape (M, 3) containing Cartesian unit vectors
            of target points.
        :param triangles:
            Array of shape (M, 3, 3) containing Cartesian vertices
            of the associated triangles.
        :return:
            Array of shape (M, 3) containing barycentric weights
            corresponding to triangle vertices.
        """
        a = triangles[:, 0, :]
        b = triangles[:, 1, :]
        c = triangles[:, 2, :]
        p = points

        area_a = self._spherical_triangle_area(p, b, c)
        area_b = self._spherical_triangle_area(p, c, a)
        area_c = self._spherical_triangle_area(p, a, b)

        total = area_a + area_b + area_c
        total = np.where(total <= 0, 1.0, total)
        w_a = area_a / total
        w_b = area_b / total
        w_c = area_c / total

        bary = np.stack([w_a, w_b, w_c], axis=1)
        return bary

    def _compute_barycentric_weights(self, device, dtype) -> torch.Tensor:
        """
       Construct the interpolation weight matrix.

       The resulting matrix W satisfies:

           f_extended = f_init @ W

       where:
           f_init     has shape (..., N)
           W          has shape (N, M)
           f_extended has shape (..., M)

       The matrix is first assembled in sparse COO format and then
       converted to a dense tensor for efficient multiplication.

       :param device:
           PyTorch device on which the weight matrix is stored.
       :param dtype:
           Floating-point precision of the weights.
       :return:
           Dense PyTorch tensor of shape (N, M) containing interpolation weights.
       """
        N = len(self.init_vertices)
        M = len(self.extended_vertices)

        triangle_indices, triangles = self._find_containing_triangles_spherical()
        bary_coords = self._spherical_barycentric_coords(self.extended_cartesian, triangles)
        tri_vertex_indices = self.init_triangulation[triangle_indices]  # (M,3)
        row_indices = tri_vertex_indices.flatten()  # (3M,)
        col_indices = np.repeat(np.arange(M), 3)
        values = bary_coords.flatten()

        row_indices_t = torch.tensor(row_indices, dtype=torch.long, device=device)
        col_indices_t = torch.tensor(col_indices, dtype=torch.long, device=device)
        values_t = torch.tensor(values, dtype=dtype, device=device)

        weight_matrix_sparse = torch.sparse_coo_tensor(
            torch.stack([row_indices_t, col_indices_t]),
            values_t,
            size=(N, M),
            device=device,
            dtype=dtype
        )

        weight_matrix = weight_matrix_sparse.to_dense()
        return weight_matrix

    @property
    def transform_matrix(self) -> torch.Tensor:
        """
        Return the matrix with the shape [N, M] which transforms initial function to extended function
        :return: transformation matrix
        """
        return self._transform_matrix

    def forward(self, f_values: torch.Tensor) -> torch.Tensor:
        """
        Interpolate values defined on the initial spherical mesh.

        :param f_values:
            Tensor of shape (..., N) containing values defined at the
            initial mesh vertices.
        :return:
            Tensor of shape (..., M) containing interpolated values
            at the target vertices.
        """
        result = torch.matmul(f_values, self._transform_matrix)
        return result


class MeshProcessorBase(nn.Module):
    """Base class for mesh processing pipelines with adaptive vertex generation.

    Generates spherical meshes with density increasing toward poles to maintain
    approximately uniform area per triangle.
    """
    def __init__(self,
                 init_grid_frequency: int,
                 phi_limits: tuple[float, float],
                 boundaries_cond: tp.Optional[str],
                 device: torch.device,
                 dtype: torch.dtype):
        """Initialize mesh processor with geometric parameters.

       :param init_grid_frequency: Base resolution parameter controlling vertex count
       :type init_grid_frequency: int
       :param phi_limits: (phi_min, phi_max) bounds in radians
       :type phi_limits: tuple[float, float]
       :param boundaries_cond: Boundary handling: "periodic", "reflection", "a", or None
       :type boundaries_cond: str | None
       :param device: Computation device
       :type device: torch.device
       :param dtype: Floating point precision
       :type dtype: torch.dtype
       """
        super().__init__()
        self.init_grid_frequency = init_grid_frequency
        self.phi_limits = phi_limits
        self.boundaries_cond = boundaries_cond
        self.last_point = boundaries_cond != "periodic"

    def _create_theta_lines(self, grid_frequency: int, last_point: bool) -> list[ThetaLine]:
        """
        Create theta lines with adaptive vertex density.
        Generates lines with increasing vertex count toward poles for uniform area.

        :param grid_frequency:
        :param last_point: Include or not the last point to computations. It is necessary to exclude the phi=2pi point.
        :return:
        """
        eps = 1e-8
        init_lines = [
            ThetaLine(theta=0.0, points=1, phi_limits=self.phi_limits, last_point=True),
            ThetaLine(theta=eps, points=2, phi_limits=self.phi_limits, last_point=True),
            ThetaLine(theta=2 * eps, points=3, phi_limits=self.phi_limits, last_point=last_point)
        ]

        return init_lines + [
            ThetaLine(
                theta=np.arccos(1 - (point - 3) ** 2 / (grid_frequency - 3) ** 2),
                points=point,
                phi_limits=self.phi_limits,
                last_point=last_point,
            ) for point in range(4, grid_frequency + 1)
        ]

    def _assemble_vertices(self, theta_lines: list[ThetaLine]) -> np.ndarray:
        """Concatenate vertices from all theta lines into single array.

        :param theta_lines: List of ThetaLine objects
        :type theta_lines: list[ThetaLine]
        :return: Array of shape (N, 2) with [phi, theta] coordinates
        :rtype: np.ndarray
        """
        return np.concatenate(
            [np.array(tl.phi_theta(), dtype=np.float32) for tl in theta_lines],
            axis=0
        )

    def _build_triangles(self, theta_lines: tp.List[ThetaLine]) -> np.ndarray:
        """Generate triangle connectivity for a spherical mesh with adaptive phi sampling.

        Handles variable numbers of vertices per theta line and correctly enforces periodic
        boundary conditions in the azimuthal (phi) direction. When a `ThetaLine` has
        `last_point=False`, its logical endpoint at φ = 2π is not stored as a separate vertex;
        instead, it is identified with the first vertex of the same line (φ = 0).

        As a result:
        - If `last_point=True`, all `latent_points` are stored as distinct vertices.
        - If `last_point=False`, only `latent_points - 1` vertices are stored, and any reference
          to the missing φ = 2π point during triangle construction maps to the first vertex
          of that theta line.

        The total number of triangles remains unchanged compared to a fully sampled mesh;
        only vertex indices are adjusted to reflect the wrapping.

        Example 1 (all lines include φ = 2π):
            ThetaLines (4 total, all `last_point=True`) yield vertices:
                Line 0: [v0]
                Line 1: [v1, v2]
                Line 2: [v3, v4, v5]
                Line 3: [v6, v7, v8, v9]
            Triangles include: [2, 4, 5], etc.

        Example 2 (last two lines exclude φ = 2π):
            Same logical structure, but stored vertices:
                Line 0: [v0]
                Line 1: [v1, v2]
                Line 2: [v3, v4]          # latent=3 → stores 2 points
                Line 3: [v5, v6, v7]      # latent=4 → stores 3 points
            Triangle [2, 4, 3] appears because the logical point at φ=2π on line 2
            wraps to index 3 (its first vertex).

        :param theta_lines: List of `ThetaLine` objects defining the mesh rows.
                            The first two lines must have `last_point=True` to anchor the poles.
        :type theta_lines: list[ThetaLine]
        :return: Array of shape `(M, 3)` containing vertex indices for each triangle.
        :rtype: np.ndarray
        """
        K = len(theta_lines)

        n_actual = [line._compute_visible_points() for line in theta_lines]
        offsets = np.cumsum([0] + n_actual[:-1])
        def idx(k: int, q: int) -> int:
            return offsets[k] + (q % n_actual[k]) if n_actual[k] > 0 else 0

        upward = [
            [idx(k, q), idx(k + 1, q), idx(k + 1, q + 1)]
            for k in range(K - 1)
            for q in range(theta_lines[k].latent_points)
            if len({idx(k, q), idx(k + 1, q), idx(k + 1, q + 1)}) == 3
        ]

        downward = [
            [idx(k, q), idx(k, q - 1), idx(k + 1, q)]
            for k in range(1, K - 1)
            for q in range(1, theta_lines[k].latent_points)
            if len({idx(k, q), idx(k, q - 1), idx(k + 1, q)}) == 3
        ]

        return np.array(upward + downward, dtype=int)

    def _triangulate(self, theta_lines: tp.List[ThetaLine]) -> np.ndarray:
        """Generate triangle connectivity for given grid resolution.

        :param theta_lines: List of `ThetaLine` objects defining the mesh rows.
                            The first two lines must have `last_point=True` to anchor the poles.
        :type theta_lines: list[ThetaLine]
        :return: Triangle connectivity array
        :rtype: np.ndarray
        """
        return self._build_triangles(theta_lines)


class InterpolatorsName(Enum):
    """Enumeration of supported interpolator types."""
    RBF: str = "rbf"
    BARYCENTRIC: str = "barycentric"


class InterpolatingMeshProcessor(MeshProcessorBase):
    """Mesh processor with interpolation from coarse to fine grid.

    Uses nearest-neighbor interpolation to transfer function values from
    initial mesh to higher-resolution target mesh.
    """
    def __init__(self,
                 interpolate_grid_frequency: int,
                 init_grid_frequency: int,
                 phi_limits: tuple[float, float],
                 boundaries_cond: tp.Optional[str],
                 interpolator: tp.Union[InterpolatorsName, tp.Type[BaseInterpolator]],
                 interpolator_kwargs: tp.Optional[dict[str, tp.Any]],
                 device: torch.device,
                 dtype: torch.dtype):
        """Initialize interpolating mesh processor.

       :param interpolate_grid_frequency: Target resolution for interpolation
       :type interpolate_grid_frequency: int
       :param init_grid_frequency: Base resolution for initial mesh
       :type init_grid_frequency: int
       :param phi_limits: (phi_min, phi_max) bounds
       :type phi_limits: tuple[float, float]
       :param boundaries_cond: Boundary condition type
       :type boundaries_cond: str | None

       :param interpolator: Interpolator type or custom class
       :type interpolator: tp.Union[InterpolatorsName, type[BaseInterpolator]]
       :param interpolator_kwargs: Configuration for interpolator constructor
       :type interpolator_kwargs: tp.Union[dict[str, Any], None]

       :param device: Computation device
       :type device: torch.device
       :param dtype: Floating point precision
       :type dtype: torch.dtype
       """
        super().__init__(
            init_grid_frequency=init_grid_frequency,
            phi_limits=phi_limits,
            boundaries_cond=boundaries_cond,
            device=device,
            dtype=dtype
        )
        self.interpolate_grid_frequency = interpolate_grid_frequency

        self.base_theta_lines = self._create_theta_lines(self.init_grid_frequency, last_point=False)
        self.interpolating_theta_lines = self._create_theta_lines(self.interpolate_grid_frequency, last_point=False)

        self.final_vertices, self.simplices = self._get_post_mesh()

        self.init_vertices = self._assemble_vertices(self.base_theta_lines)
        self.interpolator = self._get_interpolator(interpolator, interpolator_kwargs, device=device, dtype=dtype)

        self.extended_size = self.final_vertices.shape[0]

    def _get_post_mesh(self):
        """Generate vertices and triangles for target resolution mesh.

       :return: Tuple of (vertices, triangles)
       :rtype: tuple[np.ndarray, np.ndarray]
       """
        extended_vertices = self._assemble_vertices(self.interpolating_theta_lines)
        simplices = self._triangulate(self.interpolating_theta_lines)
        return extended_vertices, simplices

    def _get_interpolator(self,
                          interpolator: tp.Union[InterpolatorsName, tp.Type[BaseInterpolator]],
                          interpolator_kwargs: tp.Optional[dict[str, tp.Any]],
                          device: torch.device,
                          dtype: torch.dtype) -> BaseInterpolator:
        """
        Instantiate interpolator based on specification.

        :param interpolator: Enum value or BaseInterpolator subclass
        :type interpolator: InterpolatorsName | type[BaseInterpolator]
        :param interpolator_kwargs: Configuration parameters for constructor
        :type interpolator_kwargs: dict[str, Any] | None
        :param device: Computation device
        :type device: torch.device
        :param dtype: Floating point precision
        :return: Instantiated interpolator object
        :rtype: BaseInterpolator
        :raises NotImplementedError: For unsupported interpolator types
        """
        interpolator_kwargs = interpolator_kwargs or {}

        if interpolator == InterpolatorsName.RBF:
            return RBFInterpolator(init_vertices=self.init_vertices,
                                   extended_vertices=self.final_vertices,
                                   device=device, dtype=dtype, **interpolator_kwargs)

        elif interpolator == InterpolatorsName.BARYCENTRIC:
            return BarycentricInterpolator(
                init_vertices=self.init_vertices,
                extended_vertices=self.final_vertices,
                init_triangulation=self._build_triangles(self.base_theta_lines),
                device=device, dtype=dtype, **interpolator_kwargs)

        elif issubclass(interpolator, BaseInterpolator):
            return interpolator(
                init_vertices=self.init_vertices,
                extended_vertices=self.final_vertices,
                device=device, dtype=dtype, **interpolator_kwargs)
        else:
            raise NotImplementedError("The interpolator can be only rbf and baricentric or custom class")

    def forward(self, f_values: torch.Tensor) -> torch.Tensor:
        """
        Interpolate function values to higher resolution mesh.

        :param f_values: Function values at base vertices, shape (..., N)
        :type f_values: torch.Tensor
        :return: Interpolated values at extended vertices, shape (..., M)
        :rtype: torch.Tensor
        """
        shape = f_values.shape
        init_vert_dim = shape[-1]
        f_new = f_values.reshape((-1, init_vert_dim))
        out = self.interpolator(f_new)
        return out.reshape((*shape[:-1], out.shape[-1]))


class BoundaryMeshProcessor(MeshProcessorBase):
    """Mesh processor with boundary condition handling but no interpolation.

    Used when mesh resolution remains constant but boundary conditions require
    special vertex handling.
    """
    def __init__(self,
                 init_grid_frequency: int,
                 phi_limits: tuple[float, float],
                 boundaries_cond: tp.Optional[str],
                 device: torch.device,
                 dtype: torch.dtype):
        """
        Initialize boundary-aware mesh processor.

        :param init_grid_frequency: Base mesh resolution
        :type init_grid_frequency: int
        :param phi_limits: Azimuthal bounds (phi_min, phi_max)
        :type phi_limits: tuple[float, float]
        :param boundaries_cond: Boundary condition type
        :type boundaries_cond: str | None
        :param device: Computation device
        :type device: torch.device
        :param dtype: Floating point precision
        :type dtype: torch.dtype
        """
        super().__init__(
            init_grid_frequency=init_grid_frequency,
            phi_limits=phi_limits,
            boundaries_cond=boundaries_cond,
            device=device,
            dtype=dtype
        )

        self.base_theta_lines = self._create_theta_lines(self.init_grid_frequency, last_point=False)
        self.final_vertices, self.simplices = self._get_post_mesh()

        self.init_vertices = self._assemble_vertices(self.base_theta_lines)
        self.extended_size = self.final_vertices.shape[0]

    def _get_post_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate vertices and triangles for target resolution mesh.

        :return: Tuple of (vertices, triangles)
        :rtype: tuple[np.ndarray, np.ndarray]
        """
        vertices = self._assemble_vertices(self.base_theta_lines)
        simplices = self._triangulate(self.base_theta_lines)
        return vertices, simplices

    def forward(self, f_values: torch.Tensor) -> torch.Tensor:
        """
        Pass through function values unchanged (no interpolation).

        :param f_values: Function values at mesh vertices
        :type f_values: torch.Tensor
        :return: Same values (identity operation)
        :rtype: torch.Tensor
        """
        return f_values


def delaunay_mesh_processor_factory(init_grid_frequency: int,
                           interpolate_grid_frequency: int,
                           interpolate: bool,
                           interpolator: InterpolatorsName,
                           interpolator_kwargs: tp.Optional[dict[str, tp.Any]],
                           boundaries_cond: tp.Optional[str],
                           phi_limits: tuple[float, float],
                           device: torch.device,
                           dtype: torch.dtype) -> tp.Union[InterpolatingMeshProcessor, BoundaryMeshProcessor]:
    """Factory function for creating appropriate mesh processor instance.

    Selects processor type based on interpolation requirements and boundary conditions.

    :param init_grid_frequency: Base mesh resolution
    :type init_grid_frequency: int
    :param interpolate_grid_frequency: Target resolution for interpolation
    :type interpolate_grid_frequency: int
    :param interpolate: Whether to perform mesh refinement interpolation
    :type interpolate: bool
    :param boundaries_cond: Boundary condition type
    :type boundaries_cond: str | None

    :param interpolator: Enum value or BaseInterpolator subclass
    :type interpolator: InterpolatorsName | type[BaseInterpolator]
    :param interpolator_kwargs: Configuration parameters for constructor
    :type interpolator_kwargs: dict[str, Any] | None

    :param phi_limits: (phi_min, phi_max) bounds
    :type phi_limits: tuple[float, float]
    :param device: Computation device
    :type device: torch.device
    :param dtype: Floating point precision
    :type dtype: torch.dtype
    :return: Configured mesh processor instance
    :rtype: tp.Union[InterpolatingMeshProcessor, BoundaryMeshProcessor]
    """

    if interpolate:
        return InterpolatingMeshProcessor(
            interpolate_grid_frequency=interpolate_grid_frequency,
            init_grid_frequency=init_grid_frequency,
            phi_limits=phi_limits,
            boundaries_cond=boundaries_cond, interpolator=interpolator,
            interpolator_kwargs=interpolator_kwargs, device=device, dtype=dtype
        )
    elif boundaries_cond != "periodic":
        return BoundaryMeshProcessor(
            init_grid_frequency=init_grid_frequency,
            phi_limits=phi_limits,
            boundaries_cond=boundaries_cond, device=device, dtype=dtype
        )
    else:
        raise ValueError("Must specify either interpolation or non-periodic boundaries")


class DelaunayMesh(BaseMeshPowder):
    """Delaunay triangulation-based spherical mesh for powder averaging.

    Generates adaptive-resolution meshes with optional interpolation for
    efficient powder averaging in magnetic resonance simulations.
    """

    def __init__(self,
                 eps: float = 1e-7,
                 phi_limits: tuple[float, float] = (0.0, 2 * math.pi),
                 initial_grid_frequency: int = 20,
                 interpolation_grid_frequency: int = 40,
                 boundaries_cond: tp.Optional[str] = None,
                 interpolate: bool = False,
                 interpolator: tp.Union[str, InterpolatorsName, tp.Type[BaseInterpolator]] = InterpolatorsName.RBF,
                 interpolator_kwargs: tp.Optional[dict[str, tp.Any]] = None,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32):
        """Initialize Delaunay mesh with geometric and processing parameters.

        :param eps: Small value for numerical stability near poles
        :type eps: float
        :param phi_limits: (phi_min, phi_max) bounds in radians
        :type phi_limits: tuple[float, float]
        :param initial_grid_frequency: Base resolution parameter
        :type initial_grid_frequency: int
        :param interpolation_grid_frequency: Target resolution when interpolating
        :type interpolation_grid_frequency: int
        :param boundaries_cond: Doesn't support yet
        :type boundaries_cond: tp.Optional[str, None]
        :param interpolate: Whether to interpolate to higher resolution mesh

        :type interpolate: bool
        :param interpolator: The Interpolation method.
        Currently it supports rbf, barycentric interpolation or custom interpolator
        :type interpolator: tp.Union[str, InterpolatorsName, tp.Type[BaseInterpolator]]

        :param interpolator_kwargs: Optional keyword arguments passed to the interpolator constructor.
        These may include parameters such as ``jitter``, ``regularization``, ``kernel``, ``tol``, etc.,
        depending on the specific interpolator type.

        For a complete list of supported options, refer to the documentation of:
        - :class:`mars.mesher.delaunay_neighbour.RBFInterpolator`
        - :class:`mars.mesher.delaunay_neighbour.BarycentricInterpolator`

        :param device: Computation device
        :type device: torch.device
        :param dtype: Floating point precision
        :type dtype: torch.dtype
        """
        super().__init__(device=device, dtype=dtype)
        self.dtype = dtype
        self.eps = eps

        self.phi_limit = phi_limits
        self.initial_grid_frequency = initial_grid_frequency
        if interpolate:
            self.interpolation_grid_frequency = interpolation_grid_frequency
        else:
            self.interpolation_grid_frequency = initial_grid_frequency

        if boundaries_cond is not None:
            raise NotImplementedError("Boundaries condition is not supported in this version")

        if isinstance(interpolator, str):
            interpolator = InterpolatorsName(interpolator)

        self.mesh_processor = delaunay_mesh_processor_factory(initial_grid_frequency, interpolation_grid_frequency,
                                                     interpolate=interpolate,
                                                     interpolator=interpolator,
                                                     interpolator_kwargs=interpolator_kwargs,
                                                     boundaries_cond=boundaries_cond,
                                                     phi_limits=phi_limits,
                                                     device=device, dtype=dtype)

        (initial_grid,
         post_grid,
         post_simplices) = self.create_initial_cache_data(device)

        self.register_buffer("_initial_grid", initial_grid)
        self.register_buffer("_post_grid", post_grid)
        self.register_buffer("_post_simplices", post_simplices)

        self._interpolator_name = interpolator if isinstance(interpolator, str) else interpolator.value
        self._interpolator_kwargs = interpolator_kwargs or {}

        self.to(device)

    def create_initial_cache_data(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Precompute and cache mesh geometry data on specified device.

        :param device: Target computation device
        :type device: torch.device
        :return: Tuple of (initial_vertices, final_vertices, simplices)
        :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        return (
            torch.as_tensor(self.mesh_processor.init_vertices, dtype=self.dtype, device=device),
            torch.as_tensor(self.mesh_processor.final_vertices, dtype=self.dtype, device=device),
            torch.as_tensor(self.mesh_processor.simplices, device=device)
        )

    @property
    def post_mesh(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Processed mesh geometry after interpolation/boundary handling.

        :return: Tuple of (vertices, triangles) with vertices shape (N, 2),
            triangles shape (M, 3)
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        return self._post_grid, self._post_simplices

    @property
    def initial_grid(self) -> torch.Tensor:
        """Initial coarse mesh vertices before processing.

        :return: Tensor of shape (N, 2) with [phi, theta] coordinates
        :rtype: torch.Tensor
        """
        return self._initial_grid

    def to_delaunay(self,
                    f_post: torch.Tensor,
                    simplices: torch.Tensor) -> torch.Tensor:
        """Format function values per simplex for Delaunay rendering.

        :param f_post: Function values at mesh vertices, shape (..., N)
        :type f_post: torch.Tensor
        :param simplices: Triangle connectivity indices, shape (M, 3)
        :type simplices: torch.Tensor
        :return: Values arranged per simplex, shape (..., M, 3)
        :rtype: torch.Tensor
        """
        return f_post[..., simplices]

    def forward(self,  f_init: torch.Tensor) -> torch.Tensor:
        """Process function values through mesh pipeline (interpolation/boundaries).

        :param f_init: Function values at initial mesh vertices, shape (..., N)
        :type f_init: torch.Tensor
        :return: Processed values at final mesh vertices, shape (..., M)
        :rtype: torch.Tensor
        """
        return self.mesh_processor(f_init)

    def to_json_dict(self) -> dict:
        return {
            "type": "DelaunayMesh",
            "eps": self.eps,
            "phi_limits": self.phi_limit,
            "initial_grid_frequency": self.initial_grid_frequency,
            "interpolation_grid_frequency": self.interpolation_grid_frequency,
            "boundaries_cond": None,
            "interpolate": (self.interpolation_grid_frequency > self.initial_grid_frequency),
            "interpolator": self._interpolator_name,
            "interpolator_kwargs": self._interpolator_kwargs,
        }

    @classmethod
    def from_json_dict(
            cls, data: dict[str, tp.Any],
            device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32) -> DelaunayMesh:
        """Reconstruct a Delaunay mesh from a dictionary.

        :param data: Dictionary as produced by :meth:`to_json_dict`.
        :param device: Computation device.
        :param dtype: Floating point precision.
        :return: A new :class:`DelaunayMesh` instance.
        """
        return cls(
            eps=data.get("eps", 1e-7),
            phi_limits=tuple(data.get("phi_limits", (0.0, 2 * math.pi))),
            initial_grid_frequency=data.get("initial_grid_frequency", 20),
            interpolation_grid_frequency=data.get("interpolation_grid_frequency", 40),
            boundaries_cond=data.get("boundaries_cond"),
            interpolate=data.get("interpolate", False),
            interpolator=data.get("interpolator", InterpolatorsName.RBF.value),
            interpolator_kwargs=data.get("interpolator_kwargs"),
            device=device,
            dtype=dtype
        )


class DelaunayMeshFullSphere(DelaunayMesh):
    def __init__(self,
                 eps: float = 1e-7,
                 phi_limits: tuple[float, float] = (0, 2 * math.pi),
                 initial_grid_frequency: int = 20,
                 interpolation_grid_frequency: int = 40,
                 boundaries_cond=None,
                 interpolate=False,
                 dtype=torch.float32, device: torch.device = torch.device("cpu")):
        super().__init__(eps, phi_limits, initial_grid_frequency,
                         interpolation_grid_frequency, boundaries_cond, interpolate,
                         device=device, dtype=dtype
                         )
        _second_unit = self._initial_grid.clone()

        _second_unit[:, 1] = torch.pi - _second_unit[:, 1]
        self._initial_grid = torch.cat((self._initial_grid, _second_unit), dim=-2)

        _second_unit = self._post_grid.clone()
        _second_unit[:, 1] = torch.pi - _second_unit[:, 1]
        self._post_grid = torch.cat((self._post_grid, _second_unit), dim=-2)

        second_simpl = self._post_simplices.clone()
        second_simpl = second_simpl + _second_unit.shape[-2]
        self._post_simplices = torch.cat((self._post_simplices, second_simpl), dim=-2)
