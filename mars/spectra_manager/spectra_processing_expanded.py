import typing as tp

import torch
import torch.nn as nn

from .. import mesher

from .spectral_integration import BaseSpectraIntegrator
from .utils import OutputSpectraMode, ComputationalDetails, ExpandedComputationalDetails

from .spectra_processing_base import CrystalStationaryProcessing, PowderStationaryProcessing,\
    CrystalTimeProcessing, PowderTimeProcessing,\
    PostSpectraProcessing


class _AutoFieldAxisMixin(nn.Module):
    """
    Internal mixin that provides automatic magnetic field axis computation
    based on resonance fields and linewidths.

    This mixin is designed to be used with `PowderStationaryProcessing`,
    `CrystalStationaryProcessing`, `PowderTimeProcessing`, and
    `CrystalTimeProcessing`. It adds the ability to generate a dynamic field
    sweep range without requiring an external `fields` tensor.

    The field axis is determined by:
        1. Finding the global min and max of resonance fields across all
           transitions and orientations.
        2. Estimating the necessary spectral width using both the resonance
           field span and the maximum linewidth (scaled by `width_factor`).
        3. Adjusting the min/max fields with `spectral_width_part` to create
           margins, and clamping to absolute bounds `min_exp_field` /
           `max_exp_field`.
        4. Creating a linearly spaced field axis with `num_points` points.

    :cvar num_points: Number of points in the generated field axis.
    :cvar spectral_width_part: Fraction of the estimated spectral width used for margins.
    :cvar width_factor: Multiplier for the maximum linewidth.
    :cvar min_exp_field: Absolute lower bound for the field sweep.
    :cvar max_exp_field: Absolute upper bound for the field sweep.
    :cvar width_cutoff: Only linewidths > this value are considered for width extension.
    """
    def _init_field_axis_buffers(self, num_points: int, spectral_width_part: float,
                                 width_factor: float, min_exp_field: float, max_exp_field: float,
                                 width_cutoff: float, device: torch.device, dtype: torch.dtype) -> None:
        """
        Register persistent buffers for field‑axis parameters.

        :param num_points: Number of points in the generated field axis.
        :param spectral_width_part: Fraction of the estimated spectral width used to determine margins.
        :param width_factor: Multiplier applied to the maximum linewidth to extend the sweep.
        :param min_exp_field: Absolute minimum field value (lower clamp).
        :param max_exp_field: Absolute maximum field value (upper clamp).
        :param width_cutoff: Linewidth threshold (Tesla); linewidths above this value are considered.
        :param device: Target device for buffers.
        :param dtype: Data type for buffers.
        """
        self.register_buffer("num_points", torch.tensor(num_points, device=device))
        self.register_buffer("spectral_width_part", torch.tensor(spectral_width_part, device=device, dtype=dtype))
        self.register_buffer("width_factor", torch.tensor(width_factor, device=device, dtype=dtype))
        self.register_buffer("min_exp_field", torch.tensor(min_exp_field, device=device, dtype=dtype))
        self.register_buffer("max_exp_field", torch.tensor(max_exp_field, device=device, dtype=dtype))
        self.register_buffer("width_cutoff", torch.tensor(width_cutoff, device=device, dtype=dtype))

    def _get_new_field(self, res_fields: torch.Tensor, width: torch.Tensor,
                       intensities: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute a dynamic field axis based on resonance field distribution and widths.

        :param res_fields: Resonance fields, shape [..., num_transitions, num_simplices, 3] (after transformation)
        :param width: Linewidths, shape [..., num_transitions, num_simplices]
        :param intensities: Intensities, shape [..., num_transitions, num_simplices] (unused but kept for signature)
        :return: (fields, min_pos_batch, max_pos_batch)
                 fields: field axis [batch..., num_points]
                 min_pos_batch, max_pos_batch: lower and upper field limits per batch element
        """
        dims = res_fields.dim()
        batch_dims = tuple(range(max(dims - 2, 0), dims))

        min_pos_batch = torch.amin(res_fields, dim=batch_dims)
        max_pos_batch = torch.amax(res_fields, dim=batch_dims)
        mean_pos = (max_pos_batch + min_pos_batch) / 2

        width_criteria = width.clone()
        width_criteria[width > self.width_cutoff] = 0.0
        max_orient_width = torch.amax(width_criteria, dim=-1)

        nature_spectra_width = torch.max(max_pos_batch - min_pos_batch, max_orient_width * self.width_factor)

        min_pos_batch = mean_pos - nature_spectra_width / (2 * self.spectral_width_part)
        max_pos_batch = mean_pos + nature_spectra_width / (2 * self.spectral_width_part)

        min_pos_batch = torch.max(min_pos_batch, self.min_exp_field)
        max_pos_batch = torch.min(max_pos_batch, self.max_exp_field)

        steps = torch.linspace(0, 1, int(self.num_points), device=res_fields.device, dtype=res_fields.dtype)
        fields = steps * (max_pos_batch - min_pos_batch).unsqueeze(-1) + min_pos_batch.unsqueeze(-1)
        return fields, min_pos_batch, max_pos_batch


class PowderStationaryProcessingExpanded(_AutoFieldAxisMixin, PowderStationaryProcessing):
    """
    Expanded version of `PowderStationaryProcessing` that automatically determines
    the magnetic field axis from the resonance field distribution and linewidths.

    This class inherits all functionality of `PowderStationaryProcessing` and adds
    dynamic field‑axis generation. Instead of requiring an external `fields` tensor,
    the field sweep range is computed using the min/max resonance fields and the
    maximum linewidth (after a cutoff), with user‑controllable margins and clamping.

    The forward method returns a tuple `(spectrum, (min_field, max_field))`
    instead of just the spectrum.
    """
    def __init__(self,
                 mesh: mesher.BaseMeshPowder,
                 spectra_integrator: tp.Optional[BaseSpectraIntegrator] = None,
                 harmonic: int = 1,
                 post_spectra_processor: PostSpectraProcessing = PostSpectraProcessing(),
                 computational_details: ExpandedComputationalDetails = ExpandedComputationalDetails(),
                 output_mode: OutputSpectraMode = OutputSpectraMode.TOTAL,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32,
                 num_points: int = 4_000,
                 spectral_width_part: float = 0.6,
                 width_factor: float = 3.0,
                 min_exp_field: float = 0.0,
                 max_exp_field: float = 2.0,
                 width_cutoff: float = 0.5):
        """
        :param mesh: Powder mesh object (BaseMeshPowder).
        :param spectra_integrator: Optional custom integrator.
        :param harmonic: Spectral harmonic (0 = absorption, 1 = first derivative).
        :param post_spectra_processor: Post‑processing object for line broadening.
        :param computational_details: Details for integration (chunk size, natural width, etc.)
        :param output_mode: Must be `OutputSpectraMode.TOTAL`.
        :param device: Computation device.
        :param dtype: Floating‑point data type.
        :param num_points: Number of points in the generated field axis.
        :param spectral_width_part: Fraction of the estimated spectral width used to determine the sweep window.
        :param width_factor: Multiplier for the maximum linewidth to extend the sweep.
        :param min_exp_field: Absolute minimum field value (lower bound clamp).
        :param max_exp_field: Absolute maximum field value (upper bound clamp).
        :param width_cutoff: Only linewidths above this value (in Tesla) are considered.
        """
        super().__init__(mesh, spectra_integrator, harmonic, post_spectra_processor,
                         computational_details, output_mode, device, dtype)
        if output_mode != OutputSpectraMode.TOTAL:
            raise NotImplementedError(f"output_mode is supported only Total for expanded processing. "
                                      f"You have used {output_mode}")
        self._init_field_axis_buffers(num_points, spectral_width_part, width_factor,
                                      min_exp_field, max_exp_field, width_cutoff, device, dtype)

    def forward(self,
                res_fields: torch.Tensor,
                intensities: torch.Tensor,
                width: torch.Tensor,
                gauss: torch.Tensor,
                lorentz: torch.Tensor,
                fields: torch.Tensor,
                lvl_down: torch.Tensor,
                lvl_up: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        res_fields, width, intensities, areas = self._transform_data_to_mesh_format(res_fields, intensities, width)
        res_fields, width, intensities, areas = self._modify_data_dimensions(res_fields, width, intensities, areas)
        res_fields, width, intensities, areas = self._final_mask(res_fields, width, intensities, areas)

        fields, min_b, max_b = self._get_new_field(res_fields, width, intensities)
        res_fields, width, intensities, areas = self._final_mask(res_fields, width, intensities, areas)
        spec = self.spectra_integrator(res_fields, width, intensities, areas, fields)
        spectrum = self.post_spectra_processor(gauss, lorentz, fields, spec)

        return spectrum, (min_b, max_b)


class CrystalStationaryProcessingExpanded(_AutoFieldAxisMixin, CrystalStationaryProcessing):
    """
    Expanded version of `CrystalStationaryProcessing` for single‑crystal or discrete‑orientation
    samples, with automatic magnetic field axis generation.

    The field sweep is computed from the min/max resonance fields and the maximum linewidth
    (after applying a cutoff), with adjustable margins and clamping. The forward method
    returns `(spectrum, (min_field, max_field))`
    """
    def __init__(self,
                 mesh: mesher.CrystalMesh,
                 spectra_integrator: tp.Optional[BaseSpectraIntegrator] = None,
                 harmonic: int = 1,
                 post_spectra_processor: PostSpectraProcessing = PostSpectraProcessing(),
                 computational_details: ExpandedComputationalDetails = ExpandedComputationalDetails(),
                 output_mode: OutputSpectraMode = OutputSpectraMode.TOTAL,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32,
                 num_points: int = 4_000,
                 spectral_width_part: float = 0.6,
                 width_factor: float = 3.0,
                 min_exp_field: float = 0.0,
                 max_exp_field: float = 2.0,
                 width_cutoff: float = 0.5):
        """
        :param mesh: Crystal mesh object (CrystalMesh).
        :param spectra_integrator: Optional custom integrator.
        :param harmonic: Spectral harmonic (0 = absorption, 1 = first derivative).
        :param post_spectra_processor: Post‑processing object.
        :param computational_details: Integration details.
        :param output_mode: Must be `OutputSpectraMode.TOTAL`.
        :param device: Computation device.
        :param dtype: Data type.
        :param num_points: Number of points in the generated field axis.
        :param spectral_width_part: Fraction of the estimated spectral width.
        :param width_factor: Multiplier for the maximum linewidth.
        :param min_exp_field: Absolute lower bound for the field sweep.
        :param max_exp_field: Absolute upper bound for the field sweep.
        :param width_cutoff: Linewidth threshold (Tesla); linewidths above this are considered.
        """
        super().__init__(mesh, spectra_integrator, harmonic, post_spectra_processor,
                         computational_details, output_mode, device, dtype)
        if output_mode != OutputSpectraMode.TOTAL:
            raise NotImplementedError(f"output_mode is supported only Total for expanded processing. "
                                      f"You have used {output_mode}")
        self._init_field_axis_buffers(num_points, spectral_width_part, width_factor,
                                      min_exp_field, max_exp_field, width_cutoff, device, dtype)

    def forward(self,
                res_fields: torch.Tensor,
                intensities: torch.Tensor,
                width: torch.Tensor,
                gauss: torch.Tensor,
                lorentz: torch.Tensor,
                fields: torch.Tensor,
                lvl_down: torch.Tensor,
                lvl_up: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        res_fields, width, intensities, areas = self._transform_data_to_mesh_format(res_fields, intensities, width)
        res_fields, width, intensities, areas = self._modify_data_dimensions(res_fields, width, intensities, areas)
        res_fields, width, intensities, areas = self._final_mask(res_fields, width, intensities, areas)

        fields, min_b, max_b = self._get_new_field(res_fields, width, intensities)

        spec = self.spectra_integrator(res_fields, width, intensities, areas, fields)
        spectrum = self.post_spectra_processor(gauss, lorentz, fields, spec)
        return spectrum, (min_b, max_b)


class PowderTimeProcessingExpanded(_AutoFieldAxisMixin, PowderTimeProcessing):
    """
    Expanded version of `PowderTimeProcessing` for time‑resolved powder EPR spectra
    with automatic field axis generation.

    The field axis is computed once (ignoring the time dimension) from the resonance
    fields and linewidths, and is identical for all time points. The output spectrum
    includes the time dimension in its shape. Returns `(spectrum, (min_field, max_field))`.
    """
    def __init__(self,
                 mesh: mesher.BaseMeshPowder,
                 spectra_integrator: tp.Optional[BaseSpectraIntegrator] = None,
                 harmonic: int = 1,
                 post_spectra_processor: PostSpectraProcessing = PostSpectraProcessing(),
                 computational_details: ComputationalDetails = ComputationalDetails(),
                 output_mode: OutputSpectraMode = OutputSpectraMode.TOTAL,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32,
                 num_points: int = 4_000,
                 spectral_width_part: float = 0.6,
                 width_factor: float = 3.0,
                 min_exp_field: float = 0.0,
                 max_exp_field: float = 2.0,
                 width_cutoff: float = 0.5):
        """
        :param mesh: Powder mesh object.
        :param spectra_integrator: Optional custom integrator.
        :param harmonic: Spectral harmonic.
        :param post_spectra_processor: Post‑processing object.
        :param computational_details: Integration details.
        :param output_mode: Must be `OutputSpectraMode.TOTAL`.
        :param device: Computation device.
        :param dtype: Data type.
        :param num_points: Number of points in the generated field axis.
        :param spectral_width_part: Fraction of the estimated spectral width.
        :param width_factor: Multiplier for the maximum linewidth.
        :param min_exp_field: Absolute lower bound for the field sweep.
        :param max_exp_field: Absolute upper bound for the field sweep.
        :param width_cutoff: Linewidth threshold.
        """
        super().__init__(mesh, spectra_integrator, harmonic, post_spectra_processor,
                         computational_details, output_mode, device, dtype)
        if output_mode != OutputSpectraMode.TOTAL:
            raise NotImplementedError(f"output_mode is supported only Total for expanded processing. "
                                      f"You have used {output_mode}")
        self._init_field_axis_buffers(num_points, spectral_width_part, width_factor,
                                      min_exp_field, max_exp_field, width_cutoff, device, dtype)

    def forward(self,
                res_fields: torch.Tensor,
                intensities: torch.Tensor,
                width: torch.Tensor,
                gauss: torch.Tensor,
                lorentz: torch.Tensor,
                fields: torch.Tensor,
                lvl_down: torch.Tensor,
                lvl_up: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        res_fields, width, intensities, areas = self._transform_data_to_mesh_format(res_fields, intensities, width)
        res_fields, width, intensities, areas = self._modify_data_dimensions(res_fields, width, intensities, areas)
        res_fields, width, intensities, areas = self._final_mask(res_fields, width, intensities, areas)
        fields, min_b, max_b = self._get_new_field(res_fields, width, intensities)

        spec = self.spectra_integrator(res_fields, width, intensities, areas, fields)
        spectrum = self.post_spectra_processor(gauss, lorentz, fields, spec)
        return spectrum, (min_b, max_b)


class CrystalTimeProcessingExpanded(_AutoFieldAxisMixin, CrystalTimeProcessing):
    """
    Expanded version of `CrystalTimeProcessing` for time‑resolved single‑crystal EPR
    spectra with automatic field axis generation.

    The field axis is computed from the resonance fields and linewidths (ignoring time)
    and is the same for all time points. Returns `(spectrum, (min_field, max_field))`.
    """
    def __init__(self,
                 mesh: mesher.CrystalMesh,
                 spectra_integrator: tp.Optional[BaseSpectraIntegrator] = None,
                 harmonic: int = 1,
                 post_spectra_processor: PostSpectraProcessing = PostSpectraProcessing(),
                 computational_details: ComputationalDetails = ComputationalDetails(),
                 output_mode: OutputSpectraMode = OutputSpectraMode.TOTAL,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32,
                 num_points: int = 4_000,
                 spectral_width_part: float = 0.6,
                 width_factor: float = 3.0,
                 min_exp_field: float = 0.0,
                 max_exp_field: float = 2.0,
                 width_cutoff: float = 0.5):
        """
        :param mesh: Crystal mesh object.
        :param spectra_integrator: Optional custom integrator.
        :param harmonic: Spectral harmonic.
        :param post_spectra_processor: Post‑processing object.
        :param computational_details: Integration details.
        :param output_mode: Must be `OutputSpectraMode.TOTAL`.
        :param device: Computation device.
        :param dtype: Data type.
        :param num_points: Number of points in the generated field axis.
        :param spectral_width_part: Fraction of the estimated spectral width.
        :param width_factor: Multiplier for the maximum linewidth.
        :param min_exp_field: Absolute lower bound for the field sweep.
        :param max_exp_field: Absolute upper bound for the field sweep.
        :param width_cutoff: Linewidth threshold.
        """
        super().__init__(mesh, spectra_integrator, harmonic, post_spectra_processor,
                         computational_details, output_mode, device, dtype)
        if output_mode != OutputSpectraMode.TOTAL:
            raise NotImplementedError(f"output_mode is supported only Total for expanded processing. "
                                      f"You have used {output_mode}")
        self._init_field_axis_buffers(num_points, spectral_width_part, width_factor,
                                      min_exp_field, max_exp_field, width_cutoff, device, dtype)

    def forward(self,
                res_fields: torch.Tensor,
                intensities: torch.Tensor,
                width: torch.Tensor,
                gauss: torch.Tensor,
                lorentz: torch.Tensor,
                fields: torch.Tensor,
                lvl_down: torch.Tensor,
                lvl_up: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        res_fields, width, intensities, areas = self._transform_data_to_mesh_format(res_fields, intensities, width)
        res_fields, width, intensities, areas = self._modify_data_dimensions(res_fields, width, intensities, areas)
        res_fields, width, intensities, areas = self._final_mask(res_fields, width, intensities, areas)

        fields, min_b, max_b = self._get_new_field(res_fields, width, intensities)
        spec = self.spectra_integrator(res_fields, width, intensities, areas, fields)
        spectrum = self.post_spectra_processor(gauss, lorentz, fields, spec)
        return spectrum, (min_b, max_b)