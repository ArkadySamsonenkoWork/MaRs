from __future__ import annotations


from abc import ABC, abstractmethod
import typing as tp
from matplotlib import pyplot as plt

import torch
import torch.nn as nn

from .. import utils


class BaseMesh(nn.Module, ABC):
    """Abstract base class for mesh representations used in powder averaging simulations.

    Provides a common interface for crystal and powder mesh implementations,
    including rotation matrix generation and mesh properties.
    """
    @abstractmethod
    def __init__(self, device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32, *args, **kwargs):
        """Initialize the base mesh with device and dtype configuration.

        :param device: Computation device (CPU/GPU)
        :type device: torch.device
        :param dtype: Floating point precision for computations
        :type dtype: torch.dtype
        """
        super().__init__()
        self.register_buffer("_rotation_matrices", None)

    @property
    @abstractmethod
    def rotation_matrices(self) -> torch.Tensor:
        """Rotation matrices mapping laboratory frame to the frame of each molecular withing all given orientations.

        :return: Tensor of shape (..., 3, 3) containing rotation matrices
        :rtype: torch.Tensor
        """
        pass

    @property
    @abstractmethod
    def initial_size(self) -> torch.Size:
        """Size of the initial grid before any processing.

        :return: Grid dimensions excluding rotation matrix axes
        :rtype: torch.Size
        """
        pass

    @property
    @abstractmethod
    def disordered(self) -> bool:
        """Indicates whether the mesh represents a disordered (powder) sample.

        :return: True for powder samples, False for single crystals
        :rtype: bool
        """
        pass

    @property
    @abstractmethod
    def axial(self) -> bool:
        """Indicates whether the system has axial symmetry.

        :return: True if axial symmetry applies
        :rtype: bool
        """
        pass

    def is_equivalent(self, other: BaseMesh, rtol: float = 1e-5, atol: float = 1e-6) -> bool:
        """Check if two meshes are structurally and numerically equivalent.

        :param other: Another BaseMesh instance to compare against.
        :param rtol: Relative tolerance for tensor comparison.
        :param atol: Absolute tolerance for tensor comparison.
        :return: True if both meshes have the same properties and rotation matrices.
        """
        if not isinstance(other, BaseMesh):
            return False
        if type(self) is not type(other):
            return False
        if self.disordered != other.disordered:
            return False
        if self.axial != other.axial:
            return False
        if self.initial_size != other.initial_size:
            return False

        self_rot = self.rotation_matrices
        other_rot = other.rotation_matrices

        if (self_rot is None) != (other_rot is None):
            return False

        if self_rot is not None:
            if not torch.allclose(self_rot, other_rot, rtol=rtol, atol=atol):
                return False
        return True



    def to_json_dict(self) -> tp.Dict:
        """Return a JSON-serializable dict that describes this mesh and references its tensors.
        The returned dict must contain at least:
          - `"type"`: str, the class name (e.g. "CrystalMesh", "DelaunayMesh").
        All constructor arguments that are tensors should be replaced by string keys.
        """
        raise NotImplementedError("add 'to_json_dict' to make mesh serializable")


class CrystalMesh(BaseMesh):
    """Mesh representation for single-crystal samples with fixed orientation.

    Stores precomputed rotation matrices derived from Euler angles for deterministic
    orientation sampling.
    """
    def __init__(self, euler_angles: tp.Union[torch.Tensor, tp.List[float]], convention: str = "zyz",
                 device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32):
        """Initialize crystal mesh from Euler angles.

        :param euler_angles: Euler angles in radians. Shape (..., 3) where last dimension
            contains [alpha, beta, gamma] angles
        :type euler_angles: torch.Tensor

        :param convention: Euler angle convention. Supported: 'zyz', 'xyz', 'xzy',
        'yxz', 'yzx', 'zxy', 'zyx'
        :type convention: str
        :param device: Computation device
        :type device: torch.device
        :param dtype: Floating point precision
        :type dtype: torch.dtype
        """
        super().__init__(device=device, dtype=dtype)

        self.convention = convention
        if isinstance(euler_angles, list):
            self._euler_angles_list = euler_angles
            euler_angles = torch.tensor(euler_angles, dtype=dtype, device=device)
        else:
            self._euler_angles_list = euler_angles.tolist()

        if euler_angles.dim() == 1:
            euler_angles = euler_angles.unsqueeze(0)

        self.register_buffer("_rotation_matrices",
                             utils.euler_angles_to_matrix(euler_angles.to(device=device, dtype=dtype), convention)
                             )
    @property
    def rotation_matrices(self) -> torch.Tensor:
        """Precomputed rotation matrices for crystal orientations.

        :return: Tensor of shape (..., 3, 3)
        :rtype: torch.Tensor
        """
        return self._rotation_matrices

    @property
    def initial_size(self) -> torch.Size:
        """Size of the Euler angle grid.

        :return: Grid dimensions before rotation matrix axes
        :rtype: torch.Size
        """
        return self.rotation_matrices.shape[:-2]

    @property
    def disordered(self) -> bool:
        """Crystal samples are ordered systems.

        :return: Always False
        :rtype: bool
        """
        return False

    @property
    def axial(self) -> bool:
        """
        This is True for all non-powder samples

        :return: Always True for implementation
        :rtype: bool
        """
        return True

    def to_json_dict(self) -> dict:
        """Serialize crystal mesh to a dictionary."""
        return {
            "type": "CrystalMesh",
            "euler_angles": self._euler_angles_list,
            "convention": self.convention,
        }

    @classmethod
    def from_json_dict(
            cls, data: dict[str, tp.Any], device: torch.device = torch.device("cpu"),
            dtype: torch.dtype = torch.float32) -> CrystalMesh:
        """Reconstruct a crystal mesh from a dictionary.

        :param data: Dictionary as produced by :meth:`to_json_dict`.
        :param device: Computation device.
        :param dtype: Floating point precision.
        :return: A new :class:`CrystalMesh` instance.
        """
        euler_angles = data.get("euler_angles", [[0.0, 0.0, 0.0]])
        convention = data.get("convention", "zyz")
        return cls(
            euler_angles=euler_angles,
            convention=convention,
            device=device,
            dtype=dtype
        )


class BaseMeshPowder(BaseMesh):
    """Abstract base class for powder-averaged mesh implementations.

    Handles evaluation of rotation matrices and provides spherical geometry
    utilities for mesh processing.
    """
    @abstractmethod
    def __init__(self, device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32, *args, **kwargs):
        """Initialize powder mesh base with device configuration.
        :param device: Computation device
        :type device: torch.device
        :param dtype: Floating point precision
        :type dtype: torch.dtype
        """
        super().__init__(device=device, dtype=dtype)
        self._rotation_matrices: tp.Optional[torch.Tensor] = None

    @property
    def rotation_matrices(self) -> torch.Tensor:
        """Lazy-evaluated rotation matrices for powder orientations.

        Computed on first access if not already cached.

        :return: Tensor of shape (..., 3, 3) containing rotation matrices
        :rtype: torch.Tensor
        """
        if self._rotation_matrices is None:
            self._rotation_matrices = self._create_rotation_matrices()
        return self._rotation_matrices

    @property
    def initial_size(self) -> torch.Size:
        """Size of the initial spherical grid.

        :return: Grid dimensions excluding angular coordinate axes
        :rtype: torch.Size
        """
        return self.initial_grid.size()[:-1]

    def _create_rotation_matrices(self) -> torch.Tensor:
        """Given tensors phi and theta (of the same shape), returns a tensor.

        of shape (..., 3, 3) where each 3x3 matrix rotates the z-axis to the direction
        defined by the spherical angles (phi, theta).

        The rotation is computed as R =  R_y(theta) @ R_z(phi), where:
          R_z(phi) = [[cos(phi), -sin(phi), 0],
                      [sin(phi),  cos(phi), 0],
                      [      0,         0, 1]]
          R_y(theta) = [[cos(theta), 0, sin(theta)],
                        [         0, 1,          0],
                        [-sin(theta), 0, cos(theta)]]
        """
        phi = self.initial_grid[..., 0]
        theta = self.initial_grid[..., 1]
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        R = torch.empty(*phi.shape, 3, 3, dtype=phi.dtype, device=phi.device)

        R[..., 0, 0] = cos_phi * cos_theta
        R[..., 0, 0] = cos_phi * cos_theta
        R[..., 0, 1] = -sin_phi * cos_theta
        R[..., 0, 2] = sin_theta

        R[..., 1, 0] = sin_phi
        R[..., 1, 1] = cos_phi
        R[..., 1, 2] = 0

        # Third row
        R[..., 2, 0] = -sin_theta * cos_phi
        R[..., 2, 1] = sin_theta * sin_phi
        R[..., 2, 2] = cos_theta
        return R

    def areas(self) -> torch.Tensor:
        """Compute spherical areas of mesh triangles.

        :return: Areas of each triangle in the mesh (unit sphere)
        :rtype: torch.Tensor
        """
        vertices, triangles = self.post_mesh
        return self.spherical_triangle_areas(vertices, triangles)


    @staticmethod
    def spherical_triangle_areas(vertices: torch.Tensor, triangles: torch.Tensor) -> torch.Tensor:
        """Compute spherical excess areas for triangles on a unit sphere.

        Uses L'Huilier's formula for numerical stability with small triangles.

        :param vertices: Spherical coordinates [phi, theta] of shape (N, 2)
        :type vertices: torch.Tensor
        :param triangles: Triangle vertex indices of shape (M, 3)
        :type triangles: torch.Tensor
        :return: Spherical areas of shape (M,)
        :rtype: torch.Tensor
        """
        def _angle_between(u, v):
            dot = (u * v).sum(dim=1)
            dot = torch.clamp(dot, -1.0, 1.0)
            return torch.acos(dot)
        phi = vertices[:, 0]
        theta = vertices[:, 1]

        x = torch.sin(theta) * torch.cos(phi)
        y = torch.sin(theta) * torch.sin(phi)
        z = torch.cos(theta)
        xyz = torch.stack([x, y, z], dim=1)

        v0 = xyz[triangles[:, 0]]
        v1 = xyz[triangles[:, 1]]
        v2 = xyz[triangles[:, 2]]

        a = _angle_between(v1, v2)
        b = _angle_between(v2, v0)
        c = _angle_between(v0, v1)

        s = (a + b + c) / 2

        # L'Huilier's formula for spherical excess
        tan_E_4 = torch.sqrt(
            torch.clamp(
                torch.tan(s / 2) * torch.tan((s - a) / 2) * torch.tan((s - b) / 2) * torch.tan((s - c) / 2),
                min=0.0
            )
        )

        excess = 4 * torch.atan(tan_E_4)
        return excess

    @property
    @abstractmethod
    def initial_grid(self) -> torch.Tensor:
        """Initial spherical coordinate grid before processing.

        :return: Tensor of shape (..., 2) with [phi, theta] coordinates
        :rtype: torch.Tensor
        """
        pass

    @property
    @abstractmethod
    def post_mesh(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Processed mesh vertices and triangle connectivity.

        :return: Tuple of (vertices, triangles) where vertices has shape (N, 2)
            and triangles has shape (M, 3)
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        pass

    @abstractmethod
    def to_delaunay(self, f_interpolated: torch.Tensor, simplices: torch.Tensor) -> torch.Tensor:
        """Format function values for Delaunay triangulation representation.

        :param f_interpolated: Function values at mesh vertices
        :type f_interpolated: torch.Tensor
        :param simplices: Triangle connectivity indices
        :type simplices: torch.Tensor
        :return: Values arranged per simplex for rendering/processing
        :rtype: torch.Tensor
        """
        pass

    @abstractmethod
    def forward(self, f_function: torch.Tensor) -> torch.Tensor:
        """Process function values through mesh pipeline.

        :param f_function: Input function values at initial grid points
        :type f_function: torch.Tensor
        :return: Processed values at final mesh vertices
        :rtype: torch.Tensor
        """
        pass

    @property
    def disordered(self) -> bool:
        """Powder samples represent disordered ensembles.

        :return: Always True
        :rtype: bool
        """
        return True

    @property
    def axial(self) -> bool:
        return False

    def triplot(self) -> None:
        """Visualize mesh triangulation in phi-theta coordinates.

        Uses matplotlib's triplot to display triangle connectivity.
        """
        mesh, triplots = self.post_mesh
        phi, theta = mesh[..., 0], mesh[..., 1]
        plt.triplot(phi.numpy(), theta.numpy(), triplots)


class BaseMeshAxial(BaseMeshPowder):
    """Base class for axially symmetric powder averaging.

    Reduces dimensionality from 2D (phi, theta) to 1D (theta) by exploiting
    rotational symmetry around the molecular z-axis.
    """
    @abstractmethod
    def __init__(self, device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32, *args, **kwargs):
        """Initialize axial powder mesh with device configuration.

        :param device: Computation device
        :type device: torch.device
        :param dtype: Floating point precision
        :type dtype: torch.dtype
        """
        super().__init__(device=device, dtype=dtype, *args, **kwargs)

    @property
    def initial_size(self) -> torch.Size:
        """Size of the initial theta grid.

        :return: Grid dimensions excluding angular coordinate axis
        :rtype: torch.Size
        """
        return self.initial_grid.size()[:-1]

    def _create_rotation_matrices(self) -> torch.Tensor:
        """Given tensors phi and theta (of the same shape), returns a tensor.

        of shape (..., 3, 3) where each 3x3 matrix rotates the z-axis to the direction
        defined by the spherical angles (theta).

        The rotation is computed as R =  R_y(theta), where:
          R_y(theta) = [[cos(theta), 0, sin(theta)],
                        [         0, 1,          0],
                        [-sin(theta), 0, cos(theta)]]
        """
        theta = self.initial_grid[..., 0]
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        R = torch.empty(*theta.shape, 3, 3, dtype=theta.dtype, device=theta.device)

        R[..., 0, 0] = cos_theta
        R[..., 0, 1] = 0.0
        R[..., 0, 2] = sin_theta

        R[..., 1, 0] = 0.0
        R[..., 1, 1] = 1.0
        R[..., 1, 2] = 0

        # Third row
        R[..., 2, 0] = -sin_theta
        R[..., 2, 1] = 0.0
        R[..., 2, 2] = cos_theta
        return R

    @staticmethod
    def spherical_triangle_areas(vertices: torch.Tensor, triangles: torch.Tensor) -> torch.Tensor:
        """Compute spherical areas for axial symmetry mesh (effectively rings).

        Areas computed as differences in solid angle between theta boundaries.

        :param vertices: Theta coordinates of shape (N, 1)
        :type vertices: torch.Tensor
        :param triangles: Line segment indices of shape (M, 2)
        :type triangles: torch.Tensor
        :return: Ring areas of shape (M,)
        :rtype: torch.Tensor
        """
        theta = vertices[:, 0]
        end_theta = theta[triangles[:, 1]]
        start_theta = theta[triangles[:, 0]]
        excess = 2 * torch.pi * (torch.cos(end_theta) - torch.cos(start_theta))

        return excess

    @property
    def axial(self) -> bool:
        return True

    @property
    @abstractmethod
    def initial_grid(self) -> torch.Tensor:
        """Initial theta grid for axial symmetry.

        :return: Tensor of shape (..., 1) with theta coordinates
        :rtype: torch.Tensor
        """
        pass

    @property
    @abstractmethod
    def post_mesh(self):
        pass

    @abstractmethod
    def to_delaunay(self, f_interpolated: torch.Tensor, simplices: torch.Tensor) -> torch.Tensor:
        """Format function values for axial mesh representation.

        :param f_interpolated: Function values at mesh vertices
        :type f_interpolated: torch.Tensor
        :param simplices: Segment connectivity indices
        :type simplices: torch.Tensor
        :return: Values arranged per segment
        :rtype: torch.Tensor
        """
        pass

    @abstractmethod
    def forward(self, f_function: torch.Tensor) -> torch.Tensor:
        pass
