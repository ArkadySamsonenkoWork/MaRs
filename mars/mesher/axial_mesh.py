import torch
import torch.nn as nn


from .general_mesh import BaseMesh, BaseMeshAxial


class MeshProcessorAxial(nn.Module):
    def __init__(self, init_grid_frequency, device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32
                 ):
        super().__init__()
        self.init_grid_frequency = init_grid_frequency

    def _triangulate(self, grid_frequency):
        return torch.tensor([[i+1, i] for i in range(grid_frequency-1)])


class SkipMeshProcessor(MeshProcessorAxial):
    def __init__(self, device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32, *args, **kwargs):
        super().__init__(device=device, dtype=dtype, *args, **kwargs)
        self.final_vertices, self.simplices = self._get_post_mesh()
        self.init_vertices = torch.linspace(0.0, torch.pi / 2, self.init_grid_frequency).unsqueeze(-1)

    def _get_post_mesh(self):
        vertices = torch.linspace(0.0, torch.pi / 2, self.init_grid_frequency).unsqueeze(-1)
        simplices = self._triangulate(self.init_grid_frequency)
        return vertices, simplices

    def forward(self, f_values: torch.Tensor):
        return f_values


def axial_mesh_processor_factory(init_grid_frequency,
                           interpolate_grid_frequency,
                           interpolate=False,
                           device=torch.device("cpu"),
                           dtype=torch.float32):

    if interpolate:
        raise NotImplementedError

    else:
        return SkipMeshProcessor(
            device=device, dtype=dtype, init_grid_frequency=init_grid_frequency
        )


class AxialMesh(BaseMeshAxial):
    def __init__(self,
                 eps: float = 1e-7,
                 initial_grid_frequency: int = 20,
                 interpolation_grid_frequency: int = 40,
                 interpolate=False, device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32):
        """Initialize Delaunay mesh parameters.

        Args:
            eps: Small epsilon value for numerical stability
            initial_grid_frequency: Resolution of initial grid
            interpolation_grid_frequency: Resolution of interpolation grid
        """
        super().__init__(device=device, dtype=dtype)
        self.dtype = dtype
        if interpolate:
            raise NotImplementedError("Interpolation for axial case is not implemented")
        self.eps = eps
        self.initial_grid_frequency = initial_grid_frequency
        self.interpolation_grid_frequency = interpolation_grid_frequency

        self.mesh_processor = axial_mesh_processor_factory(initial_grid_frequency, interpolation_grid_frequency,
                                                           interpolate, device=device, dtype=dtype
                                                          )

        (
        initial_grid,
        post_grid,
        post_simplices
        ) = self.create_initial_cache_data(device=device)

        self.register_buffer("_initial_grid", initial_grid)
        self.register_buffer("_post_grid", post_grid)
        self.register_buffer("_post_simplices", post_simplices)
        self.to(device)

    def create_initial_cache_data(self, device: torch.device) -> tuple:
        """Create and cache initial mesh data structures."""
        return (
            torch.as_tensor(self.mesh_processor.init_vertices, dtype=self.dtype, device=device),
            torch.as_tensor(self.mesh_processor.final_vertices, dtype=self.dtype, device=device),
            torch.as_tensor(self.mesh_processor.simplices, device=device)
        )


    @property
    def post_mesh(self):
        return self._post_grid, self._post_simplices

    @property
    def initial_grid(self):
        return self._initial_grid

    def to_delaunay(self,
                    f_post: torch.Tensor,
                    simplices: torch.Tensor) -> torch.Tensor:
        """Format interpolated values for Delaunay representation.

        Args:
            f_post: Interpolated function values
            simplices: Simplices to use for final representation

        Returns:
            torch.Tensor: Values formatted for Delaunay triangulation
        """
        return f_post[..., simplices]

    def forward(self,
                    f_init: torch.Tensor) -> torch.Tensor:
        """Format interpolated values for Delaunay representation.

        Args:
            f_init: Interpolated function values

        Returns:
            torch.Tensor: Values formatted for Delaunay triangulation
        """
        return self.mesh_processor(f_init)
