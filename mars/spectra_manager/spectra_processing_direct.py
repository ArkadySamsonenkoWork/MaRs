import typing as tp
from abc import abstractmethod

import torch

from .. import mesher
from .utils import ComputationalDetails, OutputSpectraMode

from .spectra_processing_base import BaseProcessing


class BaseDirectProcessing(BaseProcessing):
    """Base class for fixed-field spectral processing over orientation meshes.

    Designed for direct diagonalization approaches where resonance searching is bypassed.
    Computes orientation-averaged spectra directly from pre-calculated intensities
    at fixed magnetic field points without line-broadening or linewidth parameters.

    The processing pipeline consists of:
    1. Transform intensity data to mesh format (interpolation/triangulation)
    2. Compute orientation weights (areas)
    3. Perform weighted orientation averaging
    4. Return spectrum in requested output mode
    """
    def __init__(self,
                 mesh: mesher.BaseMesh,
                 computational_details: ComputationalDetails = ComputationalDetails(),
                 output_mode: OutputSpectraMode = OutputSpectraMode.TOTAL,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32):
        """
        :param mesh: Mesh object defining orientation sampling grid.

        :param computational_details: The details of final spectral integration and spectra processing.

        :param output_mode: Controls spectrum organization:
            - "total": returns conventional summed spectrum over all orientations
            - "transitions": returns per-orientation/transition contributions alongside level indices

        :param device: Computation device. Default is torch.device("cpu")
        :param dtype: Data type for floating point operations. Default is torch.float32
        """
        super().__init__(mesh, computational_details, output_mode, device, dtype)

    def _output_factory_setter(self, output_mode: OutputSpectraMode) -> None:
        """Set output management methods based on requested mode.

        :param output_mode: Controls the organization of the computed spectrum.
        :return: None
        """
        if output_mode == OutputSpectraMode.TOTAL:
            self._modify_data_dimensions = self._modify_data_dimensions_total
            self._get_output = self._get_output_total
        else:
            raise ValueError(
                f"DirectProcessor supports only {OutputSpectraMode.TOTAL}. Got {output_mode}"
            )

    @abstractmethod
    def _compute_areas(self, batch_shape: tp.Union[torch.Size, int], device: torch.device) -> torch.Tensor:
        """Compute orientation weights for integration.

        :param batch_shape: Leading batch dimensions from intensity tensor.
        :param device: Target computation device.
        :return: Tensor of integration weights with shape broadcastable to [..., num_mesh_elements].
        """
        pass

    @abstractmethod
    def _transform_data_to_mesh_format(self, intensities: torch.Tensor) -> torch.Tensor:
        """Map intensities onto mesh geometry.

        :param intensities: Raw intensities at mesh vertices. Shape [..., num_vertices, num_fields]
        :return: Intensities aligned with mesh simplices or discrete orientations.
        """
        pass

    @abstractmethod
    def _integrate(self, intensities: torch.Tensor, areas: torch.Tensor, fields: torch.Tensor) -> torch.Tensor:
        """Perform orientation averaging.

        :param intensities: Mesh-aligned intensities. Shape [..., num_mesh_elements, num_fields]
        :param areas: Orientation weights. Shape [..., num_mesh_elements]
        :param fields: Magnetic field axis. Shape [num_fields]
        :return: Orientation-averaged spectrum. Shape [..., num_fields]
        """
        pass

    def _modify_data_dimensions_total(
            self, fields: torch.Tensor, intensities: torch.Tensor, areas: torch.Tensor) ->\
            tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Modify data dimension to make it computable with the given type of integrator and computation method.
        This modifier do not flatten data into num_simplices - num_transitions dimension

        :param fields: resonance field with the shape [..., num_fields, num_simplices, 3]
        :param intensities: intensities with the shape [..., num_fields, num_simplices]
        :param areas: areas with the shape [..., num_fields, num_simplices]
        :return: modified
         fields with the shape [..., num_fields]
         intensities with the shape [..., num_simplices, num_fields]
         areas with the shape [..., num_simplices, num_fields]
        """
        return fields, intensities, areas

    def _get_output_total(self, spectrum: torch.Tensor) ->\
            torch.Tensor:
        """
        Returns the final integrated spectrum as a single tensor.
        :param spectrum: Spectral contributions per transition. Shape: [..., num_transitions, N]

        :return: The single spectrum in 1D or 2D with the shpae [...., 1/2 D dimensions]
        """
        return spectrum

    def forward(self,
                fields: torch.Tensor,
                intensities: torch.Tensor) -> torch.Tensor:
        """Execute fixed-field spectral processing pipeline.

        1. Transform intensity data to mesh format
        2. Apply dimension modifiers for output mode
        3. Compute orientation weights (areas)
        4. Perform weighted orientation averaging
        5. Return spectrum in requested format

        :param fields: Magnetic field axis. Shape [num_fields]
        :param intensities: Computed intensities at fixed field points.
            Shape [..., num_orientations, num_fields] or [..., num_vertices, num_fields]
        :return: Orientation-averaged spectrum.
        """
        intensities = self._transform_data_to_mesh_format(intensities)
        batch_dims = max(0, intensities.dim() - 2)
        batch_shape = intensities.shape[:batch_dims]

        areas = self._compute_areas(batch_shape, intensities.device)
        fields, intensities, areas = self._modify_data_dimensions(fields, intensities, areas)
        spectrum = self._integrate(intensities, areas, fields)
        return self._get_output(spectrum)


class PowderDirectProcessing(BaseDirectProcessing):
    """Integrate fixed-field EPR spectra over spherical powder orientation mesh.

    Uses Delaunay triangulation and spherical triangle areas to perform
    rigorous orientation averaging. Designed for direct diagonalization
    where resonance lines are not computed.
    """
    def __init__(self,
                 mesh: mesher.BaseMeshPowder,
                 computational_details: ComputationalDetails = ComputationalDetails(),
                 output_mode: OutputSpectraMode = OutputSpectraMode.TOTAL,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32):
        """
        :param mesh: Powder mesh object (BaseMeshPowder) defining spherical grid.

        :param output_mode: Controls spectrum organization ("total" or "transitions").
        :param device: Computation device. Default is torch.device("cpu")
        :param dtype: Data type for floating point operations. Default is torch.float32
        """
        super().__init__(mesh,computational_details, output_mode, device, dtype)

    def _compute_areas(self, batch_shape: tp.Union[torch.Size, int], device: torch.device) -> torch.Tensor:
        """Compute spherical triangle areas and expand to match batch dimensions.

        :param batch_shape: Leading batch dimensions from intensity tensor.
        :param device: Target computation device.
        :return: Expanded area tensor of shape [*batch_shape, num_simplices].
        """
        _, simplices = self.mesh.post_mesh
        areas = self.mesh.spherical_triangle_areas(*self.mesh.post_mesh)
        areas = areas.reshape(1, -1).expand(*batch_shape, -1)
        return areas

    def _transform_data_to_mesh_format(self, intensities: torch.Tensor) -> torch.Tensor:
        """Interpolate intensities from mesh vertices onto Delaunay triangulation.

        :param intensities: Intensities at mesh vertices. Shape [..., num_vertices, num_fields]
        :return: Interpolated intensities at simplex centers. Shape [..., num_simplices, num_fields]
        """
        _, simplices = self.mesh.post_mesh
        processed = self.mesh(intensities.transpose(-1, -2))
        simplex_data = self.mesh.to_delaunay(processed, simplices)
        return simplex_data.mean(dim=-1).transpose(-1, -2)

    def _integrate(self, intensities: torch.Tensor, areas: torch.Tensor, fields: torch.Tensor) -> torch.Tensor:
        """Compute powder-averaged spectrum via area-weighted summation.

        :param intensities: Simplex-aligned intensities. Shape [..., num_simplices, num_fields]
        :param areas: Spherical triangle areas. Shape [..., num_simplices]
        :param fields: Magnetic field axis (unused in direct averaging).
        :return: Powder-averaged spectrum. Shape [..., num_fields]
        """

        areas_exp = areas.unsqueeze(-1)
        total_area = areas_exp.sum(dim=-2)
        spectrum = torch.sum(intensities * areas_exp, dim=-2) / total_area
        return spectrum


class CrystalDirectProcessing(BaseDirectProcessing):
    """Integrate fixed-field EPR spectra for single-crystal or discrete orientations.

    Performs simple uniform averaging over discrete crystal orientations.
    No triangulation or area weighting is required.
    """
    def __init__(self,
                 mesh: mesher.CrystalMesh,
                 computational_details: ComputationalDetails = ComputationalDetails(),
                 output_mode: OutputSpectraMode = OutputSpectraMode.TOTAL,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32):
        """
        :param mesh: Crystal mesh object defining single or discrete orientations.
        :param computational_details: The details of final spectral integration and spectra processing.
        :param output_mode: Controls spectrum organization ("total" or "transitions").
        :param device: Computation device. Default is torch.device("cpu")
        :param dtype: Data type for floating point operations. Default is torch.float32
        """
        super().__init__(mesh, computational_details, output_mode, device, dtype)

    def _compute_areas(self, batch_shape: tp.Union[torch.Size, int], device: torch.device) -> torch.Tensor:
        """Return uniform weights (ones) for discrete crystal orientations.

        :param batch_shape: Leading batch dimensions from intensity tensor.
        :param device: Target computation device.
        :return: Unity tensor of shape [*batch_shape, num_orientations].
        """
        num_orients = self.mesh.initial_size[0]
        return torch.ones((*batch_shape, num_orients), dtype=torch.float32, device=device)

    def _transform_data_to_mesh_format(self, intensities: torch.Tensor) -> torch.Tensor:
        """Pass-through for crystal mesh. Adds orientation dimension if missing.

        :param intensities: Raw intensities. Shape [..., num_orientations, num_fields]
        :return: Unmodified intensities tensor.
        """
        return intensities

    def _integrate(self, intensities: torch.Tensor, areas: torch.Tensor, fields: torch.Tensor) -> torch.Tensor:
        """Compute crystal-averaged spectrum via arithmetic mean.

        :param intensities: Orientation-resolved intensities. Shape [..., num_orients, num_fields]
        :param areas: Unity weights (unused in mean calculation).
        :param fields: Magnetic field axis (unused in direct averaging).
        :return: Crystal-averaged spectrum. Shape [..., num_fields]
        """
        return torch.mean(intensities, dim=-2)