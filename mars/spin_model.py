"""
spin_model – Definition of spin systems, interactions, and samples for EPR simulations

This module provides the core building blocks for constructing spin Hamiltonians:

- **Particles** – electrons and nuclei (from `mars.particles`)
- **Interactions** – g‑tensors, hyperfine, zero‑field splitting (ZFS), dipolar, exchange
- **SpinSystem** – a collection of spins with their interactions
- **Samples** – represent a physical sample (e.g., powder, single crystal) with broadening

The module is designed for **batched** and **orientation‑averaged** simulations.
All tensors support broadcasting over batch dimensions (e.g., multiple samples,
different parameter sets, or strain points).

Quick example – a simple S=1/2 electron with anisotropic g‑tensor:
----------------------------------------------------------------
>>> import torch
>>> from mars import spin_model
>>>
>>> # Define an anisotropic g‑tensor
>>> g = spin_model.Interaction(components=[2.0, 2.01, 2.02])
>>>
>>> # Create a spin system with one electron (spin=0.5)
>>> system = spin_model.SpinSystem(electrons=[0.5], g_tensors=[g])
>>>
>>> # Create a powder sample with Gaussian broadening
>>> sample = spin_model.MultiOrientedSample(
...     base_spin_system=system,
...     gauss=5e-4          # FWHM in tesla
... )
>>>
>>> # The sample can then be passed to a spectra manager.
"""


from __future__ import annotations

import functools
import collections
from abc import ABC, abstractmethod
import copy
import typing as tp
import math
import warnings

import torch
import torch.nn as nn

from . import constants
from . import mesher
from . import particles
from . import utils
from .mesher import BaseMesh

if tp.TYPE_CHECKING:
    from .serialization.serialization import SerializedSample, SerializedSpinSystem
    from .serialization.graph_representation import GraphSample, GraphSpinSystem


# Подумать над изменением логики.
def kronecker_product(matrices: list) -> torch.Tensor:
    """Computes the Kronecker product of a list of matrices."""
    return functools.reduce(torch.kron, matrices)


def create_operator(system_particles: list, target_idx: int, matrix: torch.Tensor) -> torch.Tensor:
    """Creates an operator acting on the target particle with identity
    elsewhere."""
    operator = []
    for i, p in enumerate(system_particles):
        operator.append(matrix if i == target_idx else p.identity)
    return kronecker_product(operator)


def scalar_tensor_multiplication(
    tensor_components_A: torch.Tensor,
    tensor_components_B: torch.Tensor,
    transformation_matrix: torch.Tensor
) -> torch.Tensor:
    """Computes the scalar product of two tensor components after applying a
    transformation.

    :param tensor_components_A:
        'torch.Tensor'
        Input tensor components with shape [..., 3, K, K].
    :param tensor_components_B:
        'torch.Tensor'
        Input tensor components with shape [..., 3, K, K].
    :param transformation_matrix:
        'torch.Tensor'
        Transformation matrix with shape [..., 3, 3].

    :return:
        'torch.Tensor'
        Scalar product with shape [..., K, K].
    """
    return torch.einsum(
        '...ij, jnl, ilm->...nm',
        transformation_matrix,
        tensor_components_A,
        tensor_components_B
    )


def transform_tensor_components(tensor_components: torch.Tensor, transformation_matrix: torch.Tensor) -> torch.Tensor:
    """Applies a matrix transformation to a collection of tensor components.

    :param tensor_components:
        'torch.Tensor'
        Input tensor components with shape [..., 3, K, K]
        where 3 represents different components (e.g., x,y,z) and K is the system dimension. For example,
        [Sx, Sy, Sz]
    :param transformation_matrix:
        'torch.Tensor'
        Transformation matrix with shape [..., 3, 3]. For example, g - tensor

    :return:
        'torch.Tensor'
        Transformed tensor components with shape [..., 3, K, K]
    """
    return torch.einsum("...ij,...jkl->...ikl", transformation_matrix, tensor_components)


def concat_spin_systems(systems: tp.Sequence[SpinSystem], mode: str = "direct_sum") -> SpinSystem:
    """
    Concatenate multiple independent spin systems into a single block-diagonal system.

    This function constructs a **direct sum** of the input Hilbert spaces.
    The resulting system represents
    effectively isolated subsystems that do not interact with each other.

    Use it only when you explicitly need to model scenarios such as:
      - Effective models for time-resolved spectroscopy where an electron may occupy
        distinct spin environments (e.g., two triplet states with slightly different
        dipolar couplings),
      - Polarized spectra involving transitions between otherwise decoupled manifolds.

    Important:
    The concatenated system is generally physically invalid as a true quantum many-body system.
    It should **not** be used to model real molecular spin clusters (e.g., diradicals or exchange-coupled pairs).
    For such cases, construct a single `SpinSystem` with explicit interactions instead.

   Performance note:
    If you only need an unpolarized, thermal-equilibrium spectrum from multiple similar systems
    (e.g., two triplet states thermally populated with ΔE << kT), it is more efficient to:
      1. Simulate each system separately,
      2. Weight and sum their spectra.
    This avoids unnecessary memory and computational overhead from block-diagonal Hamiltonians.

    :param systems: List of `SpinSystem` instances to concatenate. All must share:
        - The same floating-point `dtype`,
        - The same computation `device` (CPU/GPU).
    :return: A new `SpinSystem` with:
        - Combined particles and interactions (with globally renumbered indices),
        - Block-diagonal spin operators (no cross-subsystem terms),
        - `is_concatenated = True`.
    :raises ValueError: If input systems differ in `device` or `dtype`.
    """
    if mode != "direct_sum":
        raise NotImplementedError("Current version of 'concat_spin_systems' supports only direct_sum concatenation")

    ref_sys = systems[0]
    device = ref_sys.device
    dtype = ref_sys.dtype

    warnings.warn(
        "You are creating a block-diagonal (non-interacting) composite spin system via direct sum. "
        "This does NOT represent a true multi-particle quantum system (which would require a tensor-product space). "
        "Only use this if you are modeling effectively isolated subsystems (e.g., for polarized or time-resolved EPR). "
        "For physical spin clusters (diradicals, etc.), build a single SpinSystem with explicit couplings instead.",
        UserWarning,
        stacklevel=2
    )

    for i, sys in enumerate(systems):
        if sys.device != device:
            raise ValueError(f"System {i} device ({sys.device}) != target device ({device})")
        if sys.dtype != dtype:
            raise ValueError(f"System {i} dtype ({sys.dtype}) != target dtype ({dtype})")

    all_electrons = []
    all_g_tensors = []
    all_nuclei = []
    all_en = []
    all_ee = []
    all_nn = []

    e_offset, n_offset = 0, 0
    total_dim = 0
    total_particles = 0

    for sys in systems:
        total_dim += sys.spin_system_dim
        total_particles += len(sys.electrons) + len(sys.nuclei)
        e_offset += len(sys.electrons)
        n_offset += len(sys.nuclei)

    complex_cache = torch.zeros(
        (total_particles, 3, total_dim, total_dim),
        dtype=utils.float_to_complex_dtype(dtype),
        device=device
    )

    energy_shift_combined = torch.zeros(
        (total_dim, total_dim),
        dtype=dtype,
        device=device
    )

    e_offset = n_offset = dim_start = part_start = 0
    for sys in systems:
        sys_cache = sys.operator_cache
        d = sys.spin_system_dim
        n_particles_sys = sys_cache.shape[0]

        complex_cache[
        part_start:part_start + n_particles_sys, :, dim_start:dim_start + d, dim_start:dim_start + d] = sys_cache
        energy_shift_combined[dim_start:dim_start + d, dim_start:dim_start + d] = sys.energy_shift

        all_electrons.extend(sys.electrons)
        all_g_tensors.extend(sys.g_tensors)
        all_nuclei.extend(sys.nuclei)

        for e_idx, n_idx, inter in sys.electron_nuclei:
            all_en.append((e_offset + e_idx, n_offset + n_idx, inter))
        for e1, e2, inter in sys.electron_electron:
            all_ee.append((e_offset + e1, e_offset + e2, inter))
        for n1, n2, inter in sys.nuclei_nuclei:
            all_nn.append((n_offset + n1, n_offset + n2, inter))

        dim_start += d
        part_start += n_particles_sys
        e_offset += len(sys.electrons)
        n_offset += len(sys.nuclei)

    return SpinSystem(
        electrons=all_electrons,
        g_tensors=all_g_tensors,
        nuclei=all_nuclei or None,
        electron_nuclei=all_en or None,
        electron_electron=all_ee or None,
        nuclei_nuclei=all_nn or None,
        device=device,
        dtype=dtype,
        precomputed_operator=complex_cache,
        energy_shift=energy_shift_combined
    )


def concat_multioriented_samples(samples: tp.Sequence[MultiOrientedSample], mode: str = "direct_sum") ->\
        MultiOrientedSample:
    """
    Concatenate multiple powder-averaged spin samples into a single block-diagonal sample.

    Each input sample is first rotated from its molecular frame to the lab frame using its
    `spin_system_frame`, then all are combined into a block-diagonal spin system.
    The resulting sample shares the same broadening parameters (`lorentz`, `gauss`, `ham_strain`)
    and orientation mesh as the reference sample.

    Important:
    The concatenated system is generally physically invalid as a true quantum many-body system.
    It should **not** be used to model real molecular spin clusters (e.g., diradicals or exchange-coupled pairs).
    For such cases, construct a single `SpinSystem` with explicit interactions instead.

   Performance note:
    If you only need an unpolarized, thermal-equilibrium spectrum from multiple similar systems
    (e.g., two triplet states thermally populated with ΔE << kT), it is more efficient to:
      1. Simulate each system separately,
      2. Weight and sum their spectra.
    This avoids unnecessary memory and computational overhead from block-diagonal Hamiltonians.

    :param samples: Sequence of `MultiOrientedSample` instances. All must have:
        - Identical `lorentz`, `gauss`, and `ham_strain` parameters,
        - Identical orientation mesh (`initial_grid_frequency` and `interpolation_grid_frequency`).
    :return: A new `MultiOrientedSample` containing the concatenated, lab-frame-aligned spin system.
    :raises ValueError: If broadening parameters or meshes differ across samples.
    """
    if mode != "direct_sum":
        raise NotImplementedError("Current version of 'concat_multioriented_samples'"
                                  "supports only direct_sum concatenation")

    ref_sample = samples[0]

    for i, sample in enumerate(samples[1:], 1):
        if not torch.allclose(sample.lorentz, ref_sample.lorentz):
            raise ValueError(f"Sample {i} has different lorentz parameter than reference sample")
        if not torch.allclose(sample.gauss, ref_sample.gauss):
            raise ValueError(f"Sample {i} has different gauss parameter than reference sample")
        if (sample._ham_strain.shape != ref_sample._ham_strain.shape or
                not torch.allclose(sample._ham_strain, ref_sample._ham_strain)):
            raise ValueError(f"Sample {i} has different ham_strain parameter than reference sample")

        mesh_condition = (sample.mesh.initial_size != ref_sample.mesh.initial_size)
        if mesh_condition:
            raise ValueError(f"Sample {i} has different mesh than reference sample")

    spin_systems = []
    for sample in samples:
        spin_system = copy.deepcopy(sample.base_spin_system)
        if sample.spin_system_rot_matrix is not None:
            spin_system.apply_rotation(sample.spin_system_rot_matrix)
        spin_systems.append(spin_system)

    concatenated_spin_system = concat_spin_systems(spin_systems)
    return MultiOrientedSample(
        base_spin_system=concatenated_spin_system,
        lorentz=ref_sample.lorentz,
        gauss=ref_sample.gauss,
        mesh=ref_sample.mesh,
        spin_system_frame=None,  # Because I have already rotated spin systems
        device=ref_sample.device,
        dtype=ref_sample.dtype
        )


def init_tensor(
        components: tp.Union[torch.Tensor, tp.Sequence[float], float],
        device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Initialize a 3-component tensor of principal values (e.g., Dx, Dy, Dz).

    Accepts scalar, 1-, 2-, or 3-element inputs and expands them to three diagonal components:
      - Scalar -> [v, v, v] (isotropic)
      - 2-element -> [a, a, z] (axial symmetry)
      - 3-element -> [x, y, z] (fully anisotropic)

    :param components: float, sequence of 1–3 numbers, or torch.Tensor
        Input values defining the interaction strength along principal axes.
    :param device: torch.device
    :param dtype: torch.dtype
    :return: torch.Tensor
        Tensor of shape ``[..., 3]`` with expanded principal components.
    """
    if isinstance(components, torch.Tensor):
        tensor = components.to(device=device, dtype=dtype)
        if tensor.ndim:
            if tensor.shape[-1] == 3:
                return tensor

            elif tensor.shape[-1] == 2:
                axis_val, z_val = tensor[0], tensor[1]
                return torch.stack([axis_val, axis_val, z_val], dim=-1)

            elif tensor.shape[-1] == 1:
                tensor = tensor.squeeze(-1)
                tensor = torch.stack([tensor, tensor, tensor], dim=-1)
                return tensor
            else:
                raise ValueError(f"Tensor must have shape [..., 3] or [1] or [2], got {tensor.shape}")

        else:
            tensor = torch.stack([tensor, tensor, tensor], dim=-1)
            return tensor

    elif isinstance(components, (list, tuple)):
        if len(components) == 1:
            value = components[0]
            return torch.full((3,), value, device=device, dtype=dtype)
        elif len(components) == 2:
            axis_val, z_val = components
            return torch.tensor([axis_val, axis_val, z_val], device=device, dtype=dtype)
        elif len(components) == 3:
            return torch.tensor(components, device=device, dtype=dtype)
        else:
            raise ValueError(f"List must have 1, 2, or 3 elements, got {len(components)}")

    elif isinstance(components, (int, float)):
        return torch.full((3,), components, device=device, dtype=dtype)

    else:
        raise TypeError(f"components must be a tensor, list, tuple, or scalar, got {type(components)}")


def init_de_tensor(
        components: tp.Union[torch.Tensor, tp.Sequence[float], float],
        device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Initialize a zero-field splitting (ZFS) tensor from D and E parameters.

    Converts standard ZFS parameters (D, E) into principal components:
        Dx = -D/3 + E
        Dy = -D/3 - E
        Dz =  2D/3

    Ensures traceless tensor (Dx + Dy + Dz = 0), as required for spin S ≥ 1 systems.

    :param components : float, sequence of 1–2 numbers, or torch.Tensor
        - Scalar -> interpreted as D (E = 0)
        - 2-element -> [D, E]
        - 3-element -> used as-is (assumed already in [Dx, Dy, Dz] form)
    :param device: torch.device
    :param dtype: torch.dtype
    :return: Tensor of shape ``[..., 3]`` with [Dx, Dy, Dz] components.
    """
    if isinstance(components, torch.Tensor):
        tensor = components.to(device=device, dtype=dtype)
        if tensor.shape:
            if tensor.shape[-1] == 3:
                return tensor

            elif tensor.shape[-1] == 2:
                D, E = tensor[..., 0], tensor[..., 1]
                Dx = - D / 3 + E
                Dy = - D / 3 - E
                Dz = 2 * D / 3
                return torch.stack([Dx, Dy, Dz], dim=-1)

            elif tensor.shape[-1] == 1:
                tensor = tensor.squeeze(-1)
                Dx = - tensor / 3
                Dy = - tensor / 3
                Dz = 2 * tensor / 3
                return torch.stack([Dx, Dy, Dz], dim=-1)

            else:
                raise ValueError(f"Tensor must have shape [..., 3] or [1] or [2], got {tensor.shape}")
        else:
            Dx = - tensor / 3
            Dy = - tensor / 3
            Dz = 2 * tensor / 3
            return torch.stack([Dx, Dy, Dz], dim=-1)

    elif isinstance(components, (list, tuple)):
        if len(components) == 1:
            value = components[0]
            Dx = - value / 3
            Dy = - value / 3
            Dz = 2 * value / 3
            return torch.tensor([Dx, Dy, Dz], device=device, dtype=dtype)
        elif len(components) == 2:
            D, E = components[0], components[1]
            Dx = - D / 3 + E
            Dy = - D / 3 - E
            Dz = 2 * D / 3
            return torch.tensor([Dx, Dy, Dz], device=device, dtype=dtype)
        elif len(components) == 3:
            return torch.tensor(components, device=device, dtype=dtype)
        else:
            raise ValueError(f"List must have 1, 2, or 3 elements, got {len(components)}")

    elif isinstance(components, (int, float)):
        Dx = - components / 3
        Dy = - components / 3
        Dz = 2 * components / 3
        return torch.tensor([Dx, Dy, Dz], device=device, dtype=dtype)
    else:
        raise TypeError(f"components must be a tensor, list, tuple, or scalar, got {type(components)}")


def init_strain(strain: tp.Union[torch.Tensor, tp.Sequence, float],
                device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Initialize strain parameters as independent broadening on Dx, Dy, Dz.

    Strain here represents the full width at half maximum (FWHM) of Gaussian
    distributions for each principal component. No internal correlations are assumed.
    To take into account correlations use correlation matrix

    :param strain: None, float, sequence, or torch.Tensor
        - Scalar -> isotropic strain [s, s, s]
        - 2-element -> axial [a, a, z]
        - 3-element -> [sx, sy, sz]

    :param device: cpu / cuda
    :param dtype: float32 / float64
    :return: tensor of three components as D and E or return None
    """
    return init_tensor(strain, device, dtype)


def init_de_strain(strain: tp.Union[torch.Tensor, tp.Sequence, float],
                  device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Initialize strain for D/E-based zero-field splitting interactions.

       Strain is specified in terms of D and E parameters:
           - Scalar -> [D_strain, 0]
           - 2-element -> [D_strain, E_strain]

    :param strain: None, float, sequence of 1–2 numbers, or torch.Tensor
        FWHM values for D and optionally E.
    :param device: cpu / cuda
    :param dtype: float32 / float64
    :return:  Tensor of shape ``[..., 2]`` with [D_strain, E_strain]
    """
    if isinstance(strain, torch.Tensor):
        tensor = strain.to(device=device, dtype=dtype)
        if tensor.shape:
            if tensor.shape[-1] > 2:
                raise ValueError(f"DE Tensor Strain must have shape [..., 2] or [1] or [2], got {tensor.shape}")
            elif tensor.shape[-1] == 2:
                return tensor
            elif tensor.shape[-1] == 1:
                tensor = tensor.squeeze(-1)
                D = tensor
                E = torch.zeros_like(D)
                return torch.stack([D, E], dim=-1)
            else:
                raise ValueError(f"Tensor must have shape [..., 2] or [1] or [2], got {tensor.shape}")
        else:
            D = tensor
            E = torch.zeros_like(D)
            return torch.stack([D, E], dim=-1)

    elif isinstance(strain, (list, tuple)):
        if len(strain) == 1:
            value = strain[0]
            D = value
            E = 0
            return torch.tensor([D, E], device=device, dtype=dtype)

        elif len(strain) == 2:
            D, E = strain[0], strain[1]
            return torch.tensor([D, E], device=device, dtype=dtype)
        else:
            raise ValueError(f"List must have 1, or  2 elements, got {len(strain)}")

    elif isinstance(strain, (int, float)):
        D = strain
        E = 0
        return torch.tensor([D, E], device=device, dtype=dtype)

    else:
        raise TypeError(f"components must be a tensor, list, tuple, or scalar, got {type(strain)}")


class BaseInteraction(nn.Module, ABC):
    """Abstract base class for physical interactions in spin systems.

    Defines the interface for all interaction types (e.g., g-tensor, hyperfine, ZFS).
    Subclasses must implement core properties to describe the interaction's tensor,
    strain, and configuration shape.

    All interactions support batched evaluation and optional orientation/strain modeling.
    """
    @property
    @abstractmethod
    def tensor(self) -> torch.Tensor:
        """Return the full interaction tensor in the lab frame.

        :return: Tensor of shape ``[..., 3, 3]`` representing the interaction
                 (e.g., g-tensor, D-tensor) after applying molecular orientation.
        """
        pass

    @property
    def components(self) -> torch.Tensor:
        """Return the principal (diagonal) components of the interaction tensor.

        Equivalent to ``self.tensor`` when the frame is identity (no rotation).

        :return: Tensor of shape ``[..., 3]`` with values along the principal axes
                 (e.g., [Dx, Dy, Dz] or [gx, gy, gz]).
        """
        return self.tensor

    @property
    @abstractmethod
    def strain(self) -> tp.Optional[torch.Tensor]:
        """Return strain parameters describing distribution or broadening.

        :return: Tensor of shape ``[..., K]`` (K = number of strain parameters),
                 or ``None`` if no strain is defined.
        """
        pass

    @property
    def frame(self) -> tp.Optional[torch.Tensor]:
        """Return molecular orientation as Euler angles (ZYZ' convention).

        :return: Tensor of shape ``[..., 3]`` with angles [α, β, γ] in radians,
                 or ``None`` if the interaction is isotropic or fixed in lab frame.
        """
        return None

    @property
    @abstractmethod
    def strained_derivatives(self) -> tp.Optional[torch.Tensor]:
        """Return the tensor derivative with respect to strain parameters.

        Used in lineshape modeling to compute how small changes in Hamiltonian
        parameters (due to strain) affect the spectrum.

        :return: torch.Tensor or None
            Shape ``[..., 3, 3, 3]``, where:
              - ``strained_derivatives[..., 0, :, :]`` = ∂(interaction)/∂(component_x),
              - ``strained_derivatives[..., 1, :, :]`` = ∂(interaction)/∂(component_y),
              - ``strained_derivatives[..., 2, :, :]`` = ∂(interaction)/∂(component_z).
            Returns None if no strain is defined.
        """
        pass

    @property
    @abstractmethod
    def config_shape(self) -> torch.Size:
        """Return the batch dimensions of the interaction (excluding orientation).

        :return: ``torch.Size`` describing parameter broadcasting shape,
                 e.g., ``(N_samples,)`` or ``(N_batch, N_sites)``.
        """
        pass

    @property
    @abstractmethod
    def strain_correlation(self) -> torch.Tensor:
        """Return the correlation matrix mapping strain parameters to tensor axes.

        This matrix defines how independent strain variables affect the principal
        components. For example, in ZFS, it may map ``(δDx, δDy, δDz) -> (δD, δE)``.

        :return: Correlation matrix of shape ``(3, K)``, where K is the number
                 of independent strain parameters.
        """
        pass

    def __len__(self) -> int:
        """
        Return the number of interaction instances in the leading batch dimension.
        :return: Integer length of the first dimension of ``self.components``.
        """
        return len(self.components)

    def __repr__(self) -> str:
        """
        Return a human-readable summary of the interaction state.
        :return: Formatted string for debugging and logging.
        """
        is_batched = hasattr(self.components, 'shape') and len(self.components.shape) > 1

        if is_batched:
            batch_size = self.components.shape[0]
            lines = [f"BATCHED (batch_size={batch_size}) - showing first instance:"]

            first_components = self.components.flatten(0, -2)[0]
            if hasattr(first_components, 'tolist'):
                components_str = [f"{val:.2e}" if abs(val) >= 1e4 else f"{val:.4f}"
                                  for val in first_components.tolist()]
            else:
                components_str = [f"{val:.2e}" if abs(val) >= 1e4 else f"{val:.4f}"
                                  for val in first_components]

            lines.append(f"Principal values: [{', '.join(components_str)}]")

            # Handle batched frame
            if self.frame is not None:
                first_frame = self.frame.flatten(0, -2)[0]
                if hasattr(first_frame, 'tolist'):
                    frame_vals = first_frame.tolist()
                else:
                    frame_vals = first_frame

                if len(frame_vals) == 3:  # Euler angles
                    frame_str = f"[α={frame_vals[0]:.3f}, β={frame_vals[1]:.3f}, γ={frame_vals[2]:.3f}] rad"
                    lines.append(f"Frame (Euler angles): {frame_str}")
                else:
                    lines.append(f"Frame: {frame_vals}")
            else:
                lines.append("Frame: None")

            if self.strain is not None:
                first_strain = self.strain.flatten(0, -2)[0]
                if hasattr(first_strain, 'tolist'):
                    strain_vals = first_strain.tolist()
                    strain_str = [f"{val:.2e}" if abs(val) >= 1e4 else f"{val:.4f}"
                                  for val in strain_vals]
                    lines.append(f"Strain: [{', '.join(strain_str)}]")
                else:
                    lines.append(f"Strain: {first_strain}")
                lines.append(f"Correlation matrix: \n {self.strain_correlation}")

            else:
                lines.append("Strain: None")

        else:
            if hasattr(self.components, 'tolist'):
                components_str = [f"{val:.2e}" if abs(val) >= 1e4 else f"{val:.4f}"
                                  for val in self.components.tolist()]
            else:
                components_str = [f"{val:.2e}" if abs(val) >= 1e4 else f"{val:.4f}"
                                  for val in self.components]

            lines = [
                f"Principal values: [{', '.join(components_str)}]",
            ]

            if self.frame is not None:
                if hasattr(self.frame, 'tolist'):
                    frame_vals = self.frame.tolist()
                else:
                    frame_vals = self.frame

                if len(frame_vals) == 3:
                    frame_str = f"[α={frame_vals[0]:.3f}, β={frame_vals[1]:.3f}, γ={frame_vals[2]:.3f}] rad"
                    lines.append(f"Frame (Euler angles): {frame_str}")
                else:
                    lines.append(f"Frame: {frame_vals}")
            else:
                lines.append("Frame: Identity (no rotation)")

            if self.strain is not None:
                if hasattr(self.strain, 'tolist'):
                    strain_vals = self.strain.tolist()
                    strain_str = [f"{val:.2e}" if abs(val) >= 1e4 else f"{val:.4f}"
                                  for val in strain_vals]
                    lines.append(f"Strain: [{', '.join(strain_str)}]")
                else:
                    lines.append(f"Strain: {self.strain}")
                lines.append(f"Correlation matrix: \n {self.strain_correlation}")
            else:
                lines.append("Strain: None")

        return '\n'.join(lines)


class Interaction(BaseInteraction):
    def __init__(self, components: tp.Union[torch.Tensor, tp.Sequence, float],
                 frame: tp.Optional[tp.Union[torch.Tensor, tp.Sequence[float]]] = None,
                 strain: tp.Optional[tp.Union[torch.Tensor, tp.Sequence, float]] = None,
                 device=torch.device("cpu"), dtype=torch.float32):
        """
        :param components:

        torch.Tensor | Sequence[float] | float
            The tensor components, provided in one of the following forms:
              - A scalar (for isotropic interaction).
              - A sequence of two values (axial and z components).
              - A sequence of three values (principal components).
        The possible units are [T, Hz, dimensionless]

        :param frame:
        torch.Tensor | Sequence[float] optional
            Orientation of the tensor. Can be provided as:
              - A 1D tensor of shape (3,) representing Euler angles in ZYZ' convention.
              - A 2D tensor of shape (3, 3) representing a rotation matrix.
            Default is `None`, meaning lab frame.

        :param strain:
        torch.Tensor| Sequence[float] | float, optional
            Parameters describing interaction broadening or distribution.
            Default is `None`.
            The values are given as FWHM (full width at half maximum) of corresponding distribution
            For any number of parameters, you are assumed to specify uncorrelated principal components: Dx, Dy, Dz

        If the batched paradigm is used then only torch.Tensors with shape [..., 3] are acceptable.

        :param device:

        :param dtype:
        """
        super().__init__()
        self.register_buffer("_components", init_tensor(components, device=device, dtype=dtype))
        self.shape = self._components.shape
        batch_shape = self._components.shape[:-1]

        self._construct_rot_matrix(frame, batch_shape, device=device, dtype=dtype)

        _strain = self._init_strain_tensor(strain, device=device, dtype=dtype)
        self.register_buffer("_strain", _strain)

        _strain_correlation = torch.eye(3, device=device, dtype=dtype)
        self.register_buffer("_strain_correlation", _strain_correlation)
        self.to(device)
        self.to(dtype)

    def _init_strain_tensor(self, strain: tp.Optional[tp.Union[torch.Tensor, tp.Sequence, float]],
                            device: torch.device, dtype: torch.dtype) -> tp.Optional[torch.Tensor]:
        """Initialize strain tensor from scalar, sequence, or tensor input.

        Strain values represent FWHM of Gaussian distributions for principal components (Dx, Dy, Dz).
        Input is expanded to 3 components using axial/isotropic symmetry rules.

        :param strain: Strain parameters as scalar (isotropic), 2-element (axial), or 3-element (anisotropic).
        :param device: Target device (e.g., 'cpu' or 'cuda').
        :param dtype: Floating-point precision (e.g., torch.float32).
        :return: Strain tensor of shape [..., 3] or None if input is None.
        """
        return init_tensor(strain, device=device, dtype=dtype) if strain is not None else None

    def set_strain(self, strain: tp.Optional[torch.Tensor],
                   correlation_matrix: torch.Tensor) -> None:
        """
        Set new strain parameters from strain vector and correlation matrix
        which map strain to principle values Dx, Dy, Dz

        Defines how independent strain variables map to dDx, dDy, dDz via:
        [dDx, dDy, dDz]^T = correlation_matrix @ [ds1, ds2, ...]^T

        :param strain: Strain parameters which describe the Hamiltonian parameters distributions. IT is given at FWHM.
        The shape is [..., K]. For the None value set None
        :param correlation_matrix: Correlation matrix which match strain parameters with principle values.
        It builds transformation from Dx, Dy, Dz to strain parameters
        The shape is [3, K]
        :return: None
        """
        if strain is None:
            self._strain = strain
            self._strain_correlation = correlation_matrix
            return None

        if correlation_matrix.shape[-1] != strain.shape[-1]:
            raise ValueError(
                f"Strain and correlation matrix should be match with dimensions"
                f"get {correlation_matrix.shape[-1]} for correlation_matrix, and {self._strain.shape[-1]} for strain")
        self._strain_correlation = correlation_matrix
        self._strain = strain
        return None

    def _construct_rot_matrix(
            self, frame: tp.Optional[tp.Union[torch.Tensor, tp.Sequence[float]]], batch_shape,
            device: torch.device,
            dtype: torch.dtype
    ) -> None:
        """Construct rotation matrix and Euler angles from input frame specification.

        Supports three input formats:
          - None -> identity rotation (lab frame)
          - Sequence of 3 Euler angles (ZYZ' convention)
          - Rotation matrix (3×3)

        Internally stores both Euler angles (_frame) and rotation matrix (_rot_matrix).

        :param frame: Orientation specification as None, [α, β, γ] (radians), or 3x3 rotation matrix.
        :param batch_shape: Batch dimensions for tensor broadcasting (e.g., (N,) for N samples).
        :param device: Computation device (e.g., 'cpu' or 'cuda').
        :param dtype: Floating-point precision (e.g., torch.float32).
        :return: None
        """
        if frame is None:
            _frame = torch.zeros((*batch_shape, 3), device=device, dtype=dtype)  # alpha, beta, gamma
            _rot_matrix = self.euler_to_rotmat(_frame).to(self.components.dtype)

        else:
            if isinstance(frame, torch.Tensor):
                if frame.shape[-2:] == (3, 3) and not batch_shape:
                    _frame = utils.rotation_matrix_to_euler_angles(frame)
                    _rot_matrix = frame.to(self.components.dtype)

                elif frame.shape == (*batch_shape, 3):
                    _frame = frame.to(dtype)
                    _rot_matrix = self.euler_to_rotmat(_frame).to(self.components.dtype)

                else:
                    raise ValueError(
                        "frame must be either:\n"
                        "  • None (→ identity rotation),\n"
                        "  • a tensor of Euler angles with shape batch×3,\n"
                        "  • or a tensor of rotation matrices with shape batch×3×3."
                    )
            elif isinstance(frame, collections.abc.Sequence):
                if len(frame) != 3:
                    raise ValueError("frame must have exactly 3 values")
                _frame = torch.tensor(frame, dtype=dtype, device=device)
                _rot_matrix = self.euler_to_rotmat(_frame).to(self.components.dtype)
            else:
                raise ValueError("frame must be a Sequence of 3 values, a torch.Tensor, or None.")

        self.register_buffer("_frame", _frame)
        self.register_buffer("_rot_matrix", _rot_matrix)

    def euler_to_rotmat(self, euler_angles: torch.Tensor) -> torch.Tensor:
        """Convert ZYZ' Euler angles to rotation matrix.

        Uses the standard ZYZ' convention: R = R_z(α) R_y(β) R_z(γ).

        :param euler_angles: Tensor of shape [..., 3] with angles [α, β, γ] in radians.
        :return: Rotation matrix of shape [..., 3, 3].
        """
        return utils.euler_angles_to_matrix(euler_angles)

    def _tensor(self) -> torch.Tensor:
        """
        :return: the tensor in the spin system axis.

        the shape of the returned tensor is [..., 3, 3]
        """
        return utils.apply_single_rotation(self._rot_matrix, torch.diag_embed(self.components))

    def apply_rotation(self, rotation_matrix: torch.Tensor) -> None:
        """
        This method chagne the interaction frame rotating it with given rotation matrix:

        new_frame = rotation_matrix @ old_frame

        Update frame and rotation matrix

        :param rotation_matrix: [..., 3, 3] rotation matrix.
        :return: None
        """
        self._rot_matrix = torch.matmul(rotation_matrix, self._rot_matrix)
        self._frame = utils.rotation_matrix_to_euler_angles(self._rot_matrix)

    def _get_strained_derivatives(self) -> tp.Optional[torch.Tensor]:
        """Compute unit response of the interaction tensor to strain on each principal axis.

        For each strain direction (x, y, z), returns the derivative ∂Interaction/∂(strain_i),
        which equals the rotated dyadic product of the i-th lab-frame basis vector.
        This is used in spectral broadening simulations to weight contributions from
        parameter distributions.

        :return: torch.Tensor or None
            Shape ``[..., 3, 3, 3]``, where:
              - ``strained_derivatives[..., 0, :, :]`` = ∂(interaction)/∂(component_x),
              - ``strained_derivatives[..., 1, :, :]`` = ∂(interaction)/∂(component_y),
              - ``strained_derivatives[..., 2, :, :]`` = ∂(interaction)/∂(component_z).
            Returns None if no strain is defined.
        """
        if self._strain is None:
            return None

        else:
            return torch.einsum("...ik, ...jk->...kij", self._rot_matrix, self._rot_matrix)

    @property
    def tensor(self) -> torch.Tensor:
        """:return: the full tensor of interaction with shape [..., 3, 3] with applied rotation."""
        return self._tensor()

    @property
    def strain(self) -> tp.Optional[torch.Tensor]:
        """:return: None or tensor with shape [..., K]., where K is number of strain parameters"""
        return self._strain

    @property
    def strained_derivatives(self) -> tp.Optional[torch.Tensor]:
        """Return the tensor derivative with respect to strain parameters.

        Used in lineshape modeling to compute how small changes in Hamiltonian
        parameters (due to strain) affect the spectrum.

        :return: torch.Tensor or None
            Shape ``[..., 3, 3, 3]``, where:
              - ``strained_derivatives[..., 0, :, :]`` = ∂(interaction)/∂(component_x),
              - ``strained_derivatives[..., 1, :, :]`` = ∂(interaction)/∂(component_y),
              - ``strained_derivatives[..., 2, :, :]`` = ∂(interaction)/∂(component_z).
            Returns None if no strain is defined.
        """
        return self._get_strained_derivatives()

    @property
    def config_shape(self) -> torch.Size:
        """Interaction configureation shape. It is ..."""
        return self.shape[:-1]

    @property
    def components(self) -> torch.Tensor:
        """:return: tensor with shape [..., 3] - the principle components of a tensor."""
        return self._components

    @property
    def frame(self) -> tp.Optional[torch.Tensor]:
        """:return: angles with ZYZ' notation."""
        return self._frame

    @frame.setter
    def frame(self, frame: tp.Optional[torch.Tensor]) -> None:
        """Set new frame for computations"""
        if frame is None:
            self._frame = torch.tensor(
                [0.0, 0.0, 0.0], device=self.components.device, dtype=self.components.dtype
            )  # alpha, beta, gamma
        self._rot_matrix = self.euler_to_rotmat(self._frame)

    @property
    def strain_correlation(self) -> torch.Tensor:
        """In some cases the components of the interaction can correlate.

        To implement this correlation the strain_correlation matrix is used. For example, in the case of D/E interaction
        strain_correlation = [[-1/3, 1], [-1/3, -1], [2/3, 0]] - the matrix of trnasformation of  D and E to Dx, Dy, Dz
        :return:
        """
        return self._strain_correlation

    def __add__(self, other: Interaction) -> Interaction:
        if not isinstance(other, Interaction):
            raise TypeError("Can only add Interaction objects together")

        same_frame_flag = False
        if torch.allclose(self.frame, other.frame, atol=1e-6):
            same_frame_flag = True

        if same_frame_flag:
            new_frame = self.frame.clone()
            new_components = self.components + other.components

        else:
            tensor_self = self._tensor()
            tensor_other = other._tensor()  # [..., 3, 3]
            combined_tensor = tensor_self + tensor_other

            eigenvalues, eigenvectors = torch.linalg.eigh(combined_tensor)

            sorted_indices = torch.argsort(eigenvalues, dim=-1, descending=True)
            new_components = torch.gather(eigenvalues, -1, sorted_indices)

            new_rot_matrix = eigenvectors[..., sorted_indices].transpose(-2, -1)
            new_frame = utils.rotation_matrix_to_euler_angles(new_rot_matrix)

        correlation_matrix = torch.cat((self.strain_correlation, other.strain_correlation), dim=-1)
        if self.strain is not None and other.strain is not None:
            if not same_frame_flag:
                warnings.warn(
                    "The sum operation for the strain concatenation for interactions in different coordinate systems"
                    "is not defined correctly"
                )
            new_strain = torch.cat((self.strain, other.strain), dim=-1)

            #  It is suggested that D,E are not correlated

        elif self.strain is not None:
            if not same_frame_flag:
                warnings.warn(
                    "The sum operation for the strain concatenation for interactions in different coordinate systems"
                    "is not defined correctly"
                )
            zero_strain = torch.zeros((*other.config_shape, other.strain_correlation.shape[-1]))
            new_strain = torch.cat((self.strain, zero_strain), dim=-1)

        elif other.strain is not None:
            if not same_frame_flag:
                warnings.warn(
                    "The sum operation for the strain concatenation for interactions in different coordinate systems"
                    "is not defined correctly"
                )
            zero_strain = torch.zeros((*self.config_shape, self.strain_correlation.shape[-1]))
            new_strain = torch.cat((zero_strain, other.strain), dim=-1)

        else:
            new_strain = None
            correlation_matrix = torch.eye(3, device=new_components.device, dtype=new_components.dtype)

        interaction = Interaction(
            components=new_components,
            frame=new_frame,
            strain=None,
            device=self.components.device,
            dtype=self.components.dtype
        )
        interaction.set_strain(new_strain, correlation_matrix)
        return interaction

    def is_equivalent(self, other: Interaction, rtol: float = 1e-5, atol: float = 1e-6) -> bool:
        """Check if two interactions have numerically equivalent parameters.

        :param other: Another Interaction instance.
        :param rtol: Relative tolerance for tensor comparison.
        :param atol: Absolute tolerance for tensor comparison.
        :return: True if all internal tensors are numerically close.
        """
        if not isinstance(other, Interaction):
            return False
        if self.shape != other.shape:
            return False
        if not torch.allclose(self._components, other._components, rtol=rtol, atol=atol):
            return False
        if not torch.allclose(self._rot_matrix, other._rot_matrix, rtol=rtol, atol=atol):
            return False
        if (self._strain is None) != (other._strain is None):
            return False
        if self._strain is not None and not torch.allclose(self._strain, other._strain, rtol=rtol, atol=atol):
            return False
        if not torch.allclose(self._strain_correlation, other._strain_correlation, rtol=rtol, atol=atol):
            return False
        return True

    def get_rotation_derivative_along_axis(self, axis: tp.Union[torch.Tensor, list[float]]) -> torch.Tensor:
        """
        Compute the derivative of the interaction tensor with respect to rotation around a given axis.

        This method calculates the rate of change of the tensor Q when rotated by an infinitesimal
        angle dθ around the specified axis vector. The derivative is computed using the commutator
        formula from Lie algebra:

            dQ/dθ = [n]_× Q - Q [n]_×

        where [n]_× is the skew-symmetric cross-product matrix of the normalized axis vector,
        and Q is the full interaction tensor in the interaction frame (self.tensor).

        -----------------------------------------------------------------------
        FRAME REFERENCE AND INTERNAL STATE
        -----------------------------------------------------------------------
        This method operates on the interaction-frame tensor, which is constructed from
        internal state variables defined earlier in the class:

        • self.components  : Principal values [λ₁, λ₂, λ₃] in the principal frame
                             (diagonal tensor representation).
        • self._frame      : Euler angles [α, β, γ] (ZYZ' convention) defining
                             the rotation from principal frame to interaction frame.
        • self._rot_matrix : Rotation matrix R derived from self._frame.
        • self.tensor      : Full tensor Q = R · diag(components) · Rᵀ in the
                             interaction frame.

        The derivative is computed on self.tensor (interaction frame). Therefore, the
        rotation axis 'n' must also be specified in the interaction frame.

        Transformation Chain:
            Q_interaction = R(self._frame) · diag(self.components) · R(self._frame)ᵀ
            dQ_interaction/dθ = [n_interaction]_× Q_interaction - Q_interaction [n_interaction]_×

        :param axis: Tensor of shape [..., 3] representing the rotation axis vector(s)
                     in the interaction frame. The last dimension corresponds to Cartesian
                     components (x, y, z). Vectors will be normalized internally to
                     unit length.
        :return: Tensor of shape [..., 3, 3] containing the derivative dQ/dθ for each
                 corresponding axis vector. The derivative tensor is symmetric if Q is
                 symmetric. This derivative is expressed in the interaction frame.
        """
        axis_norm = torch.norm(axis, dim=-1, keepdim=True)
        axis_norm = torch.where(axis_norm > 0, axis_norm, torch.ones_like(axis_norm))
        n = axis / axis_norm
        Q = self.tensor
        nx, ny, nz = n[..., 0], n[..., 1], n[..., 2]

        zeros = torch.zeros_like(nx)
        Omega = torch.stack([
            torch.stack([zeros, -nz, ny], dim=-1),
            torch.stack([nz, zeros, -nx], dim=-1),
            torch.stack([-ny, nx, zeros], dim=-1)
        ], dim=-2)

        Omega_Q = torch.matmul(Omega, Q)
        Q_Omega = torch.matmul(Q, Omega)
        dQ_dtheta = Omega_Q - Q_Omega
        return dQ_dtheta

    def get_directional_derivative(self, axis: torch.Tensor,
                                   tensor_gradient: torch.Tensor) -> torch.Tensor:
        """
        Compute the directional derivative of the tensor field along a given spatial direction.

        This method calculates the rate of change of the tensor Q when displaced by an
        infinitesimal distance ds along the specified direction vector. This requires
        the spatial gradient of the tensor field ∇Q (i.e., how tensor components vary
        with position).

        The directional derivative is computed as:

            "∂Q/∂s = ∇Q · n = Σ_k (∂Q_ij/∂x_k) * n_k"

        where n is the unit direction vector and "∂Q_ij/∂x_k" are the spatial derivatives
        of each tensor component.

        :param axis: Tensor of shape [..., 3] representing the direction vector(s)
                          in the interaction frame. The last dimension corresponds to
                          Cartesian components (x, y, z). Vectors will be normalized
                          internally to unit length.

        :param tensor_gradient: Tensor of shape [..., 3, 3, 3] representing the spatial
                                gradient "∂Q_ij/∂x_k" in the interaction frame. The
                                dimensions are:
                                  - ... : Batch dimensions (must be broadcastable with
                                          direction)
                                  - 3, 3 : Tensor components (i, j)
                                  - 3 : Spatial derivatives (∂/∂x, ∂/∂y, ∂/∂z)

        :return: Tensor of shape [..., 3, 3] containing the directional derivative
                 ∂Q/∂s for each corresponding direction vector.
        """
        dir_norm = torch.norm(axis, dim=-1, keepdim=True)
        dir_norm = torch.where(dir_norm > 0, dir_norm, torch.ones_like(dir_norm))
        n = axis / dir_norm
        dQ_ds = torch.sum(tensor_gradient * n.unsqueeze(-2).unsqueeze(-2), dim=-1)

        return dQ_ds


class DEInteraction(Interaction):
    def __init__(self, components: tp.Union[torch.Tensor, tp.Sequence, float],
                 frame: tp.Optional[tp.Union[torch.Tensor, tp.Sequence[float]]] = None,
                 strain: tp.Optional[tp.Union[torch.Tensor, tp.Sequence, float]] = None,
                 device=torch.device("cpu"), dtype=torch.float32):
        """DEInteraction is given by two components D and E.

        To transform to x, y, z components the next equation is used:
        Dx = -D * 1/3 + E
        Dy = -D * 1/3 - E
        Dz = D * 2/3

        Note on DE Interaction vs. Simple Interaction
        The DE Interaction is equivalent to simple Interaction in terms of components when
        the trace of the tensor equals zero,
        but they are not equivalent in terms of strains.
        In DE Interaction, the D and E components (or only D) have a distribution,
        whereas in simple interaction, the components Dx, Dy, and Dz are distributed.

        :param components:
        torch.Tensor | Sequence[float] | float
            The tensor components, provided in one of the following forms:
              - A scalar. It is only D value.
              - A sequence of two values (D and E values).
        The possible units are [T, Hz, dimensionless]

        :param frame:
        torch.Tensor | Sequence[float] optional
            Orientation of the tensor. Can be provided as:
              - A 1D tensor of shape (3,) representing Euler angles in ZYZ' convention.
              - A 2D tensor of shape (3, 3) representing a rotation matrix.
            Default is `None`, meaning lab frame.

        :param strain:
        torch.Tensor| Sequence[float] | float, optional
            Parameters describing interaction broadening or distribution.
            Default is `None`.
            The values are given as FWHM (full width at half maximum) of corresponding distribution
            For any number of parameters, you are assumed to specify uncorrelated components.
              - A scalar. It is only D value.
              - A sequence of two values (D and E values).

        :param device: device to compute (cpu / gpu)

        :param dtype: float32 / float64
        """
        components = init_de_tensor(components, device, dtype)
        super().__init__(components, frame, strain, device, dtype)
        self._strain_correlation = torch.tensor([[-1 / 3, 1], [-1 / 3, -1], [2 / 3, 0]], device=device, dtype=dtype)

    def _init_strain_tensor(self, strain: tp.Optional[tp.Union[torch.Tensor, tp.Sequence, float]],
                            device: torch.device, dtype: torch.dtype):
        """Initialize strain tensor from scalar, sequence, or tensor input.

        Strain values represent FWHM of Gaussian distributions for D/E components .
        Input is expanded to 2 components.

        :param strain: Strain parameters as scalar (only D), 2-element (D and E).
        :param device: Target device (e.g., 'cpu' or 'cuda').
        :param dtype: Floating-point precision (e.g., torch.float32).
        :return: Strain tensor of shape [..., 3] or None if input is None.
        """
        return init_de_strain(strain, device=device, dtype=dtype) if strain is not None else None


class MultiOrientedInteraction(BaseInteraction):
    """Represents an interaction evaluated over multiple molecular orientations.

    This class stores precomputed rotated tensors for a set of orientation angles. It is generated by
    :class:`SpinSystemOrientator` and used in powder-averaged or multi-angle simulations.

    Unlike :class:`Interaction`, it does not store Euler angles or allow dynamic rotation—
    all orientation dependence is baked into the tensor arrays.
    """
    def __init__(self, oriented_tensor: torch.Tensor, strain: tp.Optional[torch.Tensor],
                 strained_derivatives: tp.Optional[torch.Tensor],
                 config_shape: torch.Size, strain_correlation: torch.Tensor,
                 device: torch.device = torch.device("cpu")):
        """
        :param oriented_tensor: torch.Tensor
            Rotated interaction tensors for each orientation.
            Shape: ``[..., orientations, 3, 3]``.

        :param: strain: Optional tensor of strain magnitudes applied to the principal components
               of the interaction (e.g., dgx, dgy, dgz).
               These values represent the full width at half maximum (FWHM) of Gaussian
               distributions describing static disorder in the sample.

               Shape: ``[..., K]``, where K is the number of independent strain parameters.
               May be ``None`` if no strain is modeled.

               !!!It is given in the principle axis of oriented_tensor

        :param strained_derivatives: torch.Tensor or None
            Strained version of the interaction, if applicable.
            Shape: ``[..., orientations, 3, 3, 3]``
            torch.Tensor or None
            Shape ``[..., 3, 3, 3]``, where:
              - ``strained_derivatives[..., 0, :, :]`` = ∂(interaction)/∂(component_x),
              - ``strained_derivatives[..., 1, :, :]`` = ∂(interaction)/∂(component_y),
              - ``strained_derivatives[..., 2, :, :]`` = ∂(interaction)/∂(component_z).
            Returns None if no strain is defined.

        :param config_shape: torch.Size
            Batch dimensions preceding the orientation axis (e.g., ``(..., N_orient)``).

        :param strain_correlation: torch.Tensor
            Correlation matrix mapping strain parameters to principal axes.
            Shape: ``(3, K)``, where ``K`` is the number of independent strain parameters.

        :param device: torch.device, optional
        """

        super().__init__()
        self.register_buffer("_strain", strain)
        self.register_buffer("_oriented_tensor", oriented_tensor)
        self.register_buffer("_strained_derivatives", strained_derivatives)
        self.register_buffer("_strain_correlation", strain_correlation)
        self._config_shape = config_shape

    @property
    def tensor(self) -> torch.Tensor:
        """Return the precomputed interaction tensors for all orientations.
        
        :return: Tensor of shape ``[..., orientations, 3, 3]``.
        """
        return self._oriented_tensor

    @property
    def strained_derivatives(self) -> tp.Optional[torch.Tensor]:
        """Return the precomputed tensor of unit excitation for all orientations.
        It is determined as delta H (strained_derivatives)

        :return: torch.Tensor or None
        Shape ``[..., 3, 3, 3]``, where:
          - ``strained_derivatives[..., 0, :, :]`` = ∂(interaction)/∂(component_x),
          - ``strained_derivatives[..., 1, :, :]`` = ∂(interaction)/∂(component_y),
          - ``strained_derivatives[..., 2, :, :]`` = ∂(interaction)/∂(component_z).
        Returns None if no strain is defined.
        """
        return self._strained_derivatives

    def apply_rotation(self, rotation_matrix: torch.Tensor) -> None:
        """
        This method change the interaction frame rotating it with given rotation matrix:

        Rotate _oriented_tensor and _strained_derivatives with according to rotation_matrix

        :param rotation_matrix: [..., 3, 3] rotation matrix.
        :return: None
        """
        self._oriented_tensor = utils.apply_single_rotation(rotation_matrix, self._oriented_tensor)
        self._strained_derivatives =\
        utils.apply_single_rotation(rotation_matrix, self._strained_derivatives)

    @property
    def config_shape(self) -> torch.Size:
        """Return the batch configuration shape (excluding orientation dimension).

        :return: ``torch.Size`` such as ``(N_batch, N_sites)``.
        """
        return self._config_shape

    @property
    def strain_correlation(self) -> torch.Tensor:
        """In some cases the components of the interaction can correlate.

        To implement this correlation the strain_correlation matrix is used. For example, in the case of D/E interaction
        strain_correlation = [[-1/3, 1], [-1/3, -1], [2/3, 0]] - the matrix of trnasformation of  D and E to Dx, Dy, Dz
        :return:
        """
        return self._strain_correlation

    @property
    def strain(self) -> tp.Optional[torch.Tensor]:
        """Return the strain parameters defining broadening of the interaction tensor.

        These values describe the distribution of values in FWHM
        in the principal components of the interaction due to microscopic disorder.
        The strain vector is not rotated with molecular orientation.

        :return:
        torch.Tensor or None
            Shape ``[..., K]``, where K is the number of independent strain parameters.
            Returns ``None`` if strain is not defined.
        """
        return self._strain


class SpinSystem(nn.Module):
    """Represents a spin system with electrons, nuclei, and interactions."""
    def __init__(self, electrons: tp.Union[list[particles.Electron], list[float]],
                 g_tensors: tp.Optional[list[BaseInteraction]] = None,
                 nuclei: tp.Optional[tp.Union[list[particles.Nucleus], list[str]]] = None,
                 electron_nuclei: tp.Optional[list[tuple[int, int, BaseInteraction]]] = None,
                 electron_electron: tp.Optional[list[tuple[int, int, BaseInteraction]]] = None,
                 nuclei_nuclei: tp.Optional[list[tuple[int, int, BaseInteraction]]] = None,
                 precomputed_operator: tp.Union[torch.Tensor, None] = None,
                 energy_shift: tp.Optional[tp.Union[float, torch.Tensor]] = None,
                 device=torch.device("cpu"), dtype: torch.dtype = torch.float32):

        """
        :param electrons:

        list[Electron] | list[float]
            Electron spins in the system. Can be specified as:
              - A list of `Electron` particle instances.
              - A list of spin quantum numbers (e.g., [0.5, 1.0]).

        :param g_tensors:
        list[BaseInteraction]
            g-tensors corresponding to each electron in `electrons`.
            Each element must be an instance of `BaseInteraction` (e.g., `Interaction`).
            Also, for a singlet state it can be passed as None. Default is None

        :param nuclei:
        list[Nucleus] | list[str], optional
            Nuclei in the system. Can be given as:
              - A list of `Nucleus` particle instances.
              - A list of isotope symbols (e.g., ["1H", "13C"]).
            Default is `None` (no nuclei).

        :param electron_nuclei:
        list[tuple[int, int, BaseInteraction]], optional
            Hyperfine interactions between electrons and nuclei.
            Each tuple is of the form (electron_index, nucleus_index, interaction_tensor).
            Default is `None`.

        :param electron_electron:
        list[tuple[int, int, BaseInteraction]], optional
            Dipolar or exchange interactions between pairs of electrons.
            Each tuple is of the form (electron_index, electron_index, interaction_tensor).
            Default is `None`.

        :param nuclei_nuclei:
        list[tuple[int, int, BaseInteraction]], optional
            Dipolar or J-coupling interactions between pairs of nuclei.
            Each tuple is of the form (nucleus_index, nucleus_index, interaction_tensor).
            Default is `None`

        :param precomputed_operator: particle operators in the Hilbert space.
            Shape: [n_particles, 3, H, H] where:
                - n_particles = total particles (electrons + nuclei):
                The order of the electrons coincides with the order of their initialization, as is the case for nuclei.
                - 3 = spin components (x, y, z)
                - H = total Hilbert space dimension (product of all spin dimensions in the default case)

            This parameter can be used when you need to create not ordinary system
            which can not be created by default methods.
            If you use it, some inner class methods may not work correctly, so be careful.
            This is a parameter for advance usage,  for ordinary workflow consider another approach

        :param energy_shift:
        float, Tensor, optional
            An additional scalar energy offset applied uniformly to the entire Hamiltonian.
            This shift does not affect the relative position of energy levels,
            doesn't change the the spectrum of the sample but changes the absolute energy reference.

        :param device: device to compute (cpu / gpu)
        :param dtype: float32/float64
        """
        super().__init__()
        self.is_artificial = False
        complex_dtype = utils.float_to_complex_dtype(dtype)

        self.electrons = self._init_electrons(electrons, device, complex_dtype)
        self.g_tensors = self._init_g_tensors(g_tensors, device, dtype)


        self.nuclei = self._init_nuclei(nuclei, device, complex_dtype) if nuclei else []
        self.electron_nuclei_interactions = nn.ModuleList()
        self.electron_electron_interactions = nn.ModuleList()
        self.nuclei_nuclei_interactions = nn.ModuleList()

        self.en_indices = []
        self.ee_indices = []
        self.nn_indices = []

        self._register_interactions(electron_nuclei, self.electron_nuclei_interactions, self.en_indices)
        self._register_interactions(electron_electron, self.electron_electron_interactions, self.ee_indices)
        self._register_interactions(nuclei_nuclei, self.nuclei_nuclei_interactions, self.nn_indices)

        if precomputed_operator is None:
            _operator_cache = self._precompute_all_operators(device=device, complex_dtype=complex_dtype)
        else:
            _operator_cache = precomputed_operator
            self.is_artificial = True

        self.register_buffer("_operator_cache_real",  _operator_cache.real)
        self.register_buffer("_operator_cache_imag", _operator_cache.imag)

        dim = _operator_cache.shape[-1]
        if energy_shift is None:
            energy_shift_tensor = torch.zeros(dim, dim, dtype=dtype, device=device)
        elif isinstance(energy_shift, torch.Tensor):
            energy_shift_tensor = energy_shift
        else:
            energy_shift_tensor = torch.eye(dim, dtype=dtype, device=device) * energy_shift
        self.register_buffer("energy_shift", energy_shift_tensor)

        self.to(device)
        self.to(dtype)

    def _init_g_tensors(self,
                        g_tensors: tp.Optional[list[BaseInteraction]],
                        device: torch.device, dtype: torch.dtype) -> nn.ModuleList:
        """Initialize and validate electron g-tensors.

        For a singlet-state system (a single spin-0 electron), `g_tensors`
        may be `None` or an arbitrary g-tensor, and a warning is emitted.
        """
        is_singlet = len(self.electrons) == 1 and self.electrons[0].spin == 0

        if is_singlet:
            warnings.warn(
                "A spin-0 electron creates a singlet-state system. "
                "g-tensors are not physically meaningful for this system.",
                UserWarning,
            )

        if g_tensors is None:
            if is_singlet:
                g_tensors = [Interaction(0.0, device=device, dtype=dtype)]
                return nn.ModuleList(g_tensors)
            raise ValueError("g_tensors must be provided for electrons with non-zero spin")

        if len(g_tensors) != len(self.electrons):
            if is_singlet:
                raise ValueError(
                    "For a singlet-state system (electrons=[0]), g_tensors must be "
                    "None or a single arbitrary g-tensor (wrapped in a list)."
                )
            raise ValueError("the number of g tensors must be equal to the number of electrons")

        return nn.ModuleList(g_tensors)

    def _register_interactions(self,
                               interactions: list[tuple[int, int, BaseInteraction]] | None,
                               module_list: nn.ModuleList,
                               index_list: list):
        """Helper to register interactions and store indices."""
        if interactions:
            for idx1, idx2, interaction in interactions:
                module_list.append(interaction)
                index_list.append((idx1, idx2))

    def _init_electrons(self, electrons: tp.Union[list[float], list[particles.Electron]],
                        device: torch.device, complex_dtype: torch.dtype) -> list[particles.Electron]:
        """Initialize electron particles from spin values or Electron instances.

        :param electrons: List of spin quantum numbers (float) or Electron objects.
        :param device: device to compute (cpu / gpu)
        :param complex_dtype: complex64/complex128
        :return: List of initialized Electron instances.
        """
        def _convert(item: tp.Union[float, particles.Electron]):
            if isinstance(item, (int, float)):
                return particles.Electron(item, device, complex_dtype)
            if isinstance(item, particles.Electron):
                return item
            raise TypeError(
                f"Expected Electron or float, got {type(item).__name__}"
            )
        return [_convert(el) for el in electrons]

    def _init_nuclei(self, nuclei: tp.Union[list[particles.Nucleus], list[str]],
                     device: torch.device, complex_dtype: torch.dtype) -> list[particles.Nucleus]:
        """Initialize nucleus particles from isotope symbols or Nucleus instances.

        :param nuclei: List of isotope strings (e.g., "1H") or Nucleus objects.
        :param device: device to compute (cpu / gpu)
        :param complex_dtype: complex64/complex128
        :return: List of initialized Nucleus instances.
        """
        def _convert(item: tp.Union[str, particles.Nucleus]):
            if isinstance(item, str):
                return particles.Nucleus(item, device, complex_dtype)
            if isinstance(item, particles.Nucleus):
                return item
            raise TypeError(f"Expected Nucleus or str, got {type(item).__name__}")
        return [_convert(n) for n in nuclei]

    @property
    def device(self) -> torch.device:
        """Get the device where the spin system tensors are stored.

        :return: The torch.device (e.g., 'cpu' or 'cuda') used for computation.
        """
        return self.operator_cache.device

    @property
    def dtype(self) -> torch.dtype:
        """Get the floating-point data type of interaction parameters."""
        return self.g_tensors[0].components.dtype

    @property
    def complex_dtype(self) -> torch.dtype:
        """Get the complex data type corresponding to the system float precision."""
        return utils.float_to_complex_dtype(self.dtype)

    @property
    def config_shape(self) -> torch.Size:
        """Get the batch configuration shape of the spin system.
        This describes parameter broadcasting dimensions (e.g., for ensembles),
        Hilbert space axes.

        :return: A tuple-like shape (e.g., (N_samples,) or (N_batch, N_sites)).
        """
        if hasattr(self, "g_tensors") and self.g_tensors and len(self.g_tensors) > 0:
            return self.g_tensors[0].config_shape
        else:
            return torch.Size([])

    @property
    def spin_system_dim(self) -> int:
        """Get the dimension of the total Hilbert space.
        Computed as the dimension of spin-operators in Hilbert Space

        :return: An integer representing the size of the full spin basis (e.g., 4 for two spin-1/2 particles).
        """
        return self.operator_cache[0].shape[-1]

    @property
    def electron_nuclei(self):
        """Get electron–nucleus hyperfine interactions as indexed tuples.
        :return: A list of tuples (electron_index, nucleus_index, interaction_tensor).
        """
        return [(idx[0], idx[1], inter) for idx, inter in zip(self.en_indices, self.electron_nuclei_interactions)]

    @electron_nuclei.setter
    def electron_nuclei(self, interactions: tp.Optional[list[tuple[int, int, BaseInteraction]]]):
        self._register_interactions(interactions, self.electron_nuclei_interactions, self.en_indices)

    @property
    def electron_electron(self):
        """Get electron–electron interactions as indexed tuples.
        :return: A list of tuples (electron_index_1, electron_index_2, interaction_tensor).
        """
        return [(idx[0], idx[1], inter) for idx, inter in zip(self.ee_indices, self.electron_electron_interactions)]

    @electron_electron.setter
    def electron_electron(self, interactions: tp.Optional[list[tuple[int, int, BaseInteraction]]]):
        self._register_interactions(interactions, self.electron_electron_interactions, self.ee_indices)

    @property
    def nuclei_nuclei(self):
        """Get nucleus–nucleus interactions as indexed tuples.
        :return: A list of tuples (nucleus_index_1, nucleus_index_2, interaction_tensor).
        """
        return [(idx[0], idx[1], inter) for idx, inter in zip(self.nn_indices, self.nuclei_nuclei_interactions)]

    @nuclei_nuclei.setter
    def nuclei_nuclei(self, interactions: tp.Optional[list[tuple[int, int, BaseInteraction]]]):
        self._register_interactions(interactions, self.nuclei_nuclei_interactions, self.nn_indices)

    @property
    def operator_cache(self) -> torch.Tensor:
        """Get precomputed spin operators for all particles in the full Hilbert space.
        Shape: [n_particles, 3, spin_dim, spin_dim], where:
          - n_particles = number of electrons + nuclei
          - 3 = x, y, z components
          - spin_dim = total Hilbert space dimension
        :return: A complex-valued tensor of spin operators.
        """
        return torch.complex(self._operator_cache_real, self._operator_cache_imag)

    def is_equivalent(self, other: SpinSystem, rtol: float = 1e-5, atol: float = 1e-6) -> bool:
        """Check equality of two spin systems.

        Compares particles, interaction tensors (including batched ones),
        index mappings, and global parameters.

        :param other: Another SpinSystem instance.
        :return: True if both systems are structurally and numerically identical.
        """
        if not isinstance(other, SpinSystem):
            return False
        if self.is_artificial != other.is_artificial:
            return False
        if len(self.electrons) != len(other.electrons) or len(self.nuclei) != len(other.nuclei):
            return False

        for e1, e2 in zip(self.electrons, other.electrons):
            if type(e1) is not type(e2) or e1.spin != e2.spin:
                return False

        for n1, n2 in zip(self.nuclei, other.nuclei):
            if type(n1) is not type(n2) or n1.spin != n2.spin:
                return False

        if len(self.g_tensors) != len(other.g_tensors):
            return False
        for g1, g2 in zip(self.g_tensors, other.g_tensors):
            if not g1.is_equivalent(g2, rtol, atol):
                return False

        for inter_self, inter_other, idx_self, idx_other in [
            (self.electron_nuclei_interactions, other.electron_nuclei_interactions,
             self.en_indices, other.en_indices),
            (self.electron_electron_interactions, other.electron_electron_interactions,
             self.ee_indices, other.ee_indices),
            (self.nuclei_nuclei_interactions, other.nuclei_nuclei_interactions,
             self.nn_indices, other.nn_indices)
        ]:
            if len(inter_self) != len(inter_other) or idx_self != idx_other:
                return False
            for i1, i2 in zip(inter_self, inter_other):
                if not i1.is_equivalent(i2, rtol, atol):
                    return False

        if not torch.allclose(self.energy_shift, other.energy_shift, rtol=rtol, atol=atol):
            return False
        return True

    def apply_rotation(self, rotation_matrix: torch.Tensor):
        """
        This method change the frame for each interaction in the spin system.
        new_frame = rotation_matrix @ old_frame
        Update each interaction frame and rotation_matrix (or tensor for multi oriented interactions)
        :param rotation_matrix: [..., 3, 3] rotation matrix.
        :return: None
        """
        for i in range(len(self.electron_nuclei_interactions)):
            self.electron_nuclei_interactions[i].apply_rotation(rotation_matrix)
        for i in range(len(self.electron_electron_interactions)):
            self.electron_electron_interactions[i].apply_rotation(rotation_matrix)
        for i in range(len(self.nuclei_nuclei_interactions)):
            self.nuclei_nuclei_interactions[i].apply_rotation(rotation_matrix)

    def _precompute_all_operators(self, device: torch.device, complex_dtype: torch.dtype):
        """Precompute spin operators for all particles in the full Hilbert space.

        Constructs Sx, Sy, Sz for each particle embedded in the total basis.

        :param device: computation  device.
        :param complex_dtype: Complex dtype (e.g., torch.complex64).
        :return: Tensor of shape [n_particles, 3, spin_dim, spin_dim].
        """
        particels = self.electrons + self.nuclei
        operator_cache = []
        for idx, p in enumerate(particels):
            axis_cache = []
            for axis, mat in zip(['x', 'y', 'z'], p.spin_matrices):
                operator = create_operator(particels, idx, mat)
                axis_cache.append(operator.to(device).to(complex_dtype))
            operator_cache.append(torch.stack(axis_cache, dim=-3))   # Сейчас каждый спин даёт матрицу [K, K] и
                                                                     # расчёт взаимодействией не оптимальный
        return torch.stack(operator_cache, dim=0)

    def get_electron_z_operator(self) -> torch.Tensor:
        """Compute the Sz operator for all electron spins in the system.

        This method sums the individual Sz operators from each electron spin operator
        cache to produce the total spin projection operator along the z-axis.

        :return: The electron Sz operator with shape [spin_dim, spin_dim], where spin_dim is the
        total dimension of the spin system Hilbert space.

        Examples
        --------
        For a system with one spin-1/2 electron:
            Returns a 2x2 matrix representing the Pauli Sz operator.
        For a system with two spin-1/2 electrons:
        Returns a 4x4 matrix representing the sum of both Sz operators.
        """
        return sum(self.operator_cache[idx][2, :, :] for idx in range(len(self.electrons)))

    def get_total_z_operator(self) -> torch.Tensor:
        """Compute the total Sz operator for all particles in the system.

        This method sums the individual Sz operators from each particle spin operator
        cache to produce the total spin projection operator along the z-axis.

        :return: The total Sz operator with shape [spin_dim, spin_dim], where spin_dim is the
        total dimension of the spin system Hilbert space.

        Examples
        --------
        For a system with one spin-1/2 electron:
            Returns a 2x2 matrix representing the Pauli Sz operator.
        For a system with one spin-1/2 electron and one nuclei spin-1/2:
        Returns a 4x4 matrix representing the sum of both Sz operators.
        """
        return sum(
            self.operator_cache[idx][2, :, :] for idx in range(len(self.electrons) + len(self.nuclei))
        )

    def get_electron_squared_operator(self) -> torch.Tensor:
        """Compute the total S² operator for all electron spins in the system.

        This method calculates S² = Sx² + Sy² + Sz² by first summing the individual
        spin vector operators from each electron, then computing the dot product of
        the total spin vector with itself.

        :return: The total S² operator with shape [spin_dim, spin_dim], where spin_dim is the
        total dimension of the spin system Hilbert space.

        Examples
        --------
        For a system with two spin-1/2 electrons:
            Eigenvalues correspond to singlet (S=0) and triplet (S=1) states.
        """
        S_vector = sum(self.operator_cache[idx] for idx in range(len(self.electrons)))
        return torch.matmul(S_vector, S_vector).sum(dim=-3)

    def get_spin_multiplet_basis(self) -> torch.Tensor:
        """Compute eigenvector in the |S, M⟩ basis (total spin and projection
        basis).

        This method diagonalizes a combination of sS² and Sz operators to obtain
        eigenvectors ordered by total spin quantum number S, then by magnetic
        quantum number M (spin projection).

        :return: torch.Tensor
        Matrix of eigenvectors with shape [spin_dim, spin_dim], where each column
        represents an eigenstate. States are ordered in ascending order of S,
        then in ascending order of M within each S manifold.

        Examples
        --------
        For two spin-1/2 electrons:
            Returns basis ordered as: |S=0, M=0⟩, |S=1, M=-1⟩, |S=1, M=0⟩, |S=1, M=1⟩
        """
        C = self.get_electron_squared_operator() + 1j * self.get_electron_z_operator()
        values, vectors = torch.linalg.eig(C)
        sorting_key = values.real * (values.imag.abs().max() + 1) + values.imag
        indices = torch.argsort(sorting_key)
        return vectors[:, indices]

    def get_product_state_basis(self) -> torch.Tensor:
        """Return the identity matrix representing the computational product
        state basis.

        The product state basis is |ms1, ms2, ..., msk, is1, ..., ism⟩ where:
        - ms1, ms2, ..., msk are magnetic quantum numbers for electrons through k
        - is1, is2, ..., ism are magnetic quantum numbers for nuclei through m
        Each state corresponds to a definite spin projection for each particle.
        :return: torch.Tensor
        Identity matrix with shape [spin_system_dim, spin_system_dim]. The identity
        matrix indicates that the current representation is already in the product
        state basis.

        Examples
        --------
        For one spin-1/2 electron and one spin-1/2 nucleus:
            Basis states: |↑, ↑⟩, |↑, ↓⟩, |↓, ↑⟩, |↓, ↓⟩
        """
        if self.is_artificial:
            raise ValueError(
                "For the concatenated spin system or complex user-created"
                "systems the product state basis is not defined"
            )
        return torch.eye(self.spin_system_dim, device=self.device, dtype=self.dtype)

    def _get_xyz_basis_two_half_spins(self) -> torch.Tensor:
        """Get basis vectors S, Tx, Ty, Tz for a spin-1 system.

        The vectors are derived by expressing the coupled triplet and singlet
        states in this uncoupled basis via Clebsch-Gordan coefficients:

            ``|T+⟩ = |↑↑⟩``
            ``|T0⟩ = (|↑↓⟩ + |↓↑⟩)/√2``
            ``|T-⟩ = |↓↓⟩``
            ``|S0⟩ = (|↑↓⟩ - |↓↑⟩)/√2``

        The triplet states ``|T+⟩``, ``|T0⟩``, ``|T-⟩`` are then mapped to
        transition vectors using the same convention as the spin-1 case
        (see :meth:`_get_xyz_basis_triplet`).

        :return: torch.Tensor
            Transition basis matrix of shape ``(4, 4)``.
            The columns represent:

            - Column 0 (S):  singlet state ``|S0⟩``
            - Column 1 (Tx): x-vector, proportional to
              ``(-|T+⟩ + |T-⟩) / sqrt(2)``
            - Column 2 (Ty): y-vector, proportional to
              ``i(|T+⟩ + |T-⟩) / sqrt(2)``
            - Column 3 (Tz): z-vector, equal to ``|T0⟩``

            In numerical form, the matrix is::

                [[ 0.000+0.j, -0.707+0.j,  0.000+0.707j,  0.000+0.j],
                 [ 0.707+0.j,  0.000+0.j,  0.000+0.000j,  0.707+0.j],
                 [-0.707+0.j,  0.000+0.j,  0.000+0.000j,  0.707+0.j],
                 [ 0.000+0.j,  0.707+0.j,  0.000+0.707j,  0.000+0.j]]

        :raises ValueError:
            If the system does not have exactly two electrons each with spin=1/2.

        Examples
        --------
        For a two-spin-1/2 system::
            T = system._get_xyz_basis_two_half_spins()  # shape (4, 4)
            S  = T[:, 0]  # singlet vector,     shape (4,)
            Tx = T[:, 1]  # x-component vector, shape (4,)
            Ty = T[:, 2]  # y-component vector, shape (4,)
            Tz = T[:, 3]  # z-component vector, shape (4,)
        """
        if len(self.electrons) != 2:
            raise ValueError("Two-spin-1/2 basis requires exactly two electrons")

        for i, electron in enumerate(self.electrons):
            if electron.spin != 0.5:
                raise ValueError(
                    f"Two-spin-1/2 basis requires spin=1/2 electrons, "
                    f"got spin={electron.spin} for electron {i}"
                )

        sqrt2 = math.sqrt(2)
        Tx = torch.tensor([-1.0 / sqrt2, 0.0, 0.0, 1.0 / sqrt2],
                          dtype=self.complex_dtype, device=self.device)
        Ty = torch.tensor([1j / sqrt2, 0.0, 0.0, 1j / sqrt2],
                          dtype=self.complex_dtype, device=self.device)
        Tz = torch.tensor([0.0, 1.0 / sqrt2, 1.0 / sqrt2, 0.0],
                          dtype=self.complex_dtype, device=self.device)
        S = torch.tensor([0.0, 1.0 / sqrt2, -1.0 / sqrt2, 0.0],
                         dtype=self.complex_dtype, device=self.device)

        return torch.stack([S, Tx, Ty, Tz], dim=1)

    def _get_xyz_basis_triplet(self) -> torch.Tensor:
        """Get basis vectors Tx, Ty, Tz for a spin-1 system.

        Returns the basis vectors representing transition moments for a spin-1
        system expressed in the ``|Mz=+1⟩``, ``|Mz=0⟩``, ``|Mz=-1⟩`` basis.

        :return: torch.Tensor
            Transition basis matrix of shape ``(3, 3)``.
            The columns represent:

            - Column 0 (Tx): x-vector, proportional to
              ``(-|+1⟩  |-1⟩) / sqrt(2)``
            - Column 1 (Ty): y-vector transitions, proportional to
              ``i(|+1⟩ + |-1⟩) / sqrt(2)``
            - Column 2 (Tz): z-vector transitions, equal to ``|0⟩``

            In numerical form, the matrix is::
                [[-0.707+0.j   ,  0.000+0.707j,  0.000+0.j],
                 [ 0.000+0.j   ,  0.000+0.j   ,  1.000+0.j],
                 [0.707+0.j   ,  0.000+0.707j,  0.000+0.j]]

        Examples
        --------
        For a spin-1 system::

            T = system.get_xyz_basis()  # shape (3, 3)
            Tx = T[:, 0]  # x-component vector, shape (3,)
            Ty = T[:, 1]  # y-component vector, shape (3,)
            Tz = T[:, 2]  # z-component vector, shape (3,)
        """
        sqrt2 = math.sqrt(2)
        Tx = torch.tensor([-1.0 / sqrt2, 0.0, 1.0 / sqrt2],
                          dtype=self.complex_dtype, device=self.device)
        Ty = torch.tensor([1j / sqrt2, 0.0, 1j / sqrt2],
                          dtype=self.complex_dtype, device=self.device)
        Tz = torch.tensor([0.0, 1.0, 0.0],
                          dtype=self.complex_dtype, device=self.device)
        return torch.stack([Tx, Ty, Tz], dim=1)

    def get_xyz_basis(self) -> torch.Tensor:
        """Get vectors  (S), Tx, Ty, Tz, for a spin system.

        Dispatches to the appropriate basis constructor depending on the system:

        - A single electron with spin=1 uses the direct spin-1 basis
          (see :meth:`_get_xyz_basis_triplet`).
        - Two electrons each with spin=1/2 uses the coupled triplet/singlet
          basis (see :meth:`_get_xyz_basis_two_half_spins`).

        :return: torch.Tensor
            Transition basis matrix.
            - For spin-1: shape ``(3, 3)``, columns are ``[Tx, Ty, Tz]``.
            - For two spin-1/2: shape ``(4, 4)``, columns are
              ``[S, Tx, Ty, Tz]``.

        :raises ValueError:
            If the system does not match a supported spin configuration.

        Examples
        --------
        For a spin-1 system::

            T = system.get_xyz_basis()  # shape (3, 3)
            Tx, Ty, Tz = T[:, 0], T[:, 1], T[:, 2]

        For two spin-1/2 electrons::

            T = system.get_xyz_basis()  # shape (4, 4)
            S, Tx, Ty, Tz = T[:, 0], T[:, 1], T[:, 2], T[:, 3]
        """
        n = len(self.electrons)
        if n == 1 and self.electrons[0].spin == 1.0:
            return self._get_xyz_basis_triplet()

        if n == 2 and self.electrons[0].spin == 0.5 and self.electrons[1].spin == 0.5:
            return self._get_xyz_basis_two_half_spins()

        raise ValueError(
            "Transition basis is only supported for a single spin-1 electron "
            "or two spin-1/2 electrons. "
            f"Got {n} electron(s) with spins "
            f"{[e.spin for e in self.electrons]}."
        )

    def get_total_projections(self) -> torch.Tensor:
        """Compute the total magnetic quantum number M for each product state.

        This method calculates M = Σmsi + Σisj (sum of all electron and nuclear
        spin projections) for each basis state in the product state representation.

        :return: torch.Tensor
        1D tensor with shape [spin_dim] containing the total spin projection
        (magnetic quantum number) for each product state basis vector.

        Examples
        --------
        For one spin-1/2 electron:
            Returns tensor([0.5, -0.5])

        For one spin-1/2 electron and one spin-1/2 nucleus:
            Returns tensor([1.0, 0.0, 0.0, -1.0])
            Corresponding to states: |↑↑⟩, |↑↓⟩, |↓↑⟩, |↓↓⟩

        For two spin-1/2 electrons:
            Returns tensor([1.0, 0.0, 0.0, -1.0])
            Corresponding to states: |↑↑⟩, |↑↓⟩, |↓↑⟩, |↓↓⟩
        """
        return torch.diagonal(self.get_total_z_operator(), offset=0, dim1=-2, dim2=-1).real

    def get_electron_projections(self) -> torch.Tensor:
        """Compute the electron-only spin projection for each product state.

        This method calculates Me = Σmsi (sum of electron spin projections only)
        for each basis state, ignoring nuclear spin contributions.

        :return: torch.Tensor
        1D tensor with shape [spin_dim] containing the total electron spin
        projection for each product state basis vector. Nuclear contributions
        are set to zero.

        Examples
        --------
        For one spin-1/2 electron:
            Returns tensor([0.5, -0.5])

        For one spin-1/2 electron and one spin-1/2 nucleus:
            Returns tensor([0.5, 0.5, -0.5, -0.5])
            Nuclear spins don't contribute, so we get: |↑_e,↑_n⟩, |↑_e,↓_n⟩, |↓_e,↑_n⟩, |↓_e,↓_n⟩

        For two spin-1/2 electrons:
            Returns tensor([1.0, 0.0, 0.0, -1.0])
            Corresponding to: |↑↑⟩, |↑↓⟩, |↓↑⟩, |↓↓⟩
        """
        return torch.diagonal(self.get_electron_z_operator(), offset=0, dim1=-2, dim2=-1).real

    def update(self,
               g_tensors: list[BaseInteraction] = None,
               electron_nuclei: tp.Union[list[tuple[int, int, BaseInteraction]], None] = None,
               electron_electron: tp.Union[list[tuple[int, int, BaseInteraction]], None] = None,
               nuclei_nuclei: tp.Union[list[tuple[int, int, BaseInteraction]], None] = None):
        """Update the parameters of spin system.

        No recomputation of spin vectors does not occur
        :param g_tensors:
        list[BaseInteraction]
            g-tensors corresponding to each electron in `electrons`.
            Each element must be an instance of `BaseInteraction` (e.g., `Interaction`).

        :param electron_nuclei:
        list[tuple[int, int, BaseInteraction]], optional
            Hyperfine interactions between electrons and nuclei.
            Each tuple is of the form (electron_index, nucleus_index, interaction_tensor).
            Default is `None`.

        :param electron_electron:
        list[tuple[int, int, BaseInteraction]], optional
            Dipolar or exchange interactions between pairs of electrons.
            Each tuple is of the form (electron_index, electron_index, interaction_tensor).
            Default is `None`.

        :param nuclei_nuclei:
        list[tuple[int, int, BaseInteraction]], optional
            Dipolar or J-coupling interactions between pairs of nuclei.
            Each tuple is of the form (nucleus_index, nucleus_index, interaction_tensor).
            Default is `None`
        """

        if g_tensors is not None:
            self.g_tensors = nn.ModuleList(g_tensors)

        if electron_nuclei is not None:
            self.electron_nuclei_interactions = nn.ModuleList()
            self.en_indices = []
            self._register_interactions(electron_nuclei, self.electron_nuclei_interactions, self.en_indices)

        if electron_electron is not None:
            self.electron_electron_interactions = nn.ModuleList()
            self.ee_indices = []
            self._register_interactions(electron_electron, self.electron_electron_interactions, self.ee_indices)

        if nuclei_nuclei is not None:
            self.nuclei_nuclei_interactions = nn.ModuleList()
            self.nn_indices = []
            self._register_interactions(nuclei_nuclei, self.nuclei_nuclei_interactions, self.nn_indices)

        self.to(self.device)
        self.to(self.dtype)

    @classmethod
    def from_serialized_spin_system(cls, serialized_spin_system: SerializedSpinSystem) -> SpinSystem:
        return serialized_spin_system.to_mars_spin_system()

    def to_serialized_spin_system(self) -> SerializedSpinSystem:
        from .serialization.serialization import SerializedSpinSystem
        return SerializedSpinSystem.from_mars_spin_system(self)

    @classmethod
    def from_graph_spin_system(cls, graph_spin_system: GraphSpinSystem) -> SpinSystem:
        return graph_spin_system.to_mars_spin_system()

    def to_graph_spin_system(self) -> GraphSpinSystem:
        from .serialization.graph_representation import GraphSpinSystem
        return GraphSpinSystem.from_mars_spin_system(self)

    def __repr__(self):
        lines = ["=" * 60]
        lines.append("SPIN SYSTEM SUMMARY")
        lines.append("=" * 60)

        lines.append("\nPARTICLES:")
        lines.append("-" * 20)

        if self.electrons:
            electron_info = []
            for i, electron in enumerate(self.electrons):
                spin_str = f"S={electron.spin} \n"
                g_gactor_str = str(self.g_tensors[i]).replace('\n', '\n      ')
                spin_str += g_gactor_str
                electron_info.append(f"  e{i}: {spin_str}")

            lines.append(f"Electrons ({len(self.electrons)}):")
            lines.extend(electron_info)
        else:
            lines.append("Electrons: None")

        if self.nuclei:
            lines.append(f"\nNuclei ({len(self.nuclei)}):")
            for i, nucleus in enumerate(self.nuclei):
                nucleus_info = f"  n{i}: "
                if hasattr(nucleus, 'isotope'):
                    nucleus_info += f"{nucleus.isotope}, "
                if hasattr(nucleus, 'spin'):
                    nucleus_info += f"I={nucleus.spin}"
                lines.append(nucleus_info)
        else:
            lines.append("\nNuclei: None")

        lines.append(f"\nSYSTEM PROPERTIES:")
        lines.append("-" * 20)
        lines.append(f"Hilbert space dimension: {self.spin_system_dim}")
        lines.append(f"Configuration shape: {tuple(self.config_shape)}")

        total_interactions = (len(self.electron_nuclei) +
                              len(self.electron_electron) +
                              len(self.nuclei_nuclei))

        if total_interactions > 0:
            lines.append(f"\nINTERACTIONS ({total_interactions} total):")
            lines.append("-" * 30)

            # Electron-nuclei interactions
            if self.electron_nuclei:
                lines.append(f"\nElectron-Nucleus ({len(self.electron_nuclei)}):")
                for i, (e_idx, n_idx, interaction) in enumerate(self.electron_nuclei):
                    lines.append(f"  {i + 1}. e{e_idx} ↔ n{n_idx}:")
                    interaction_str = str(interaction).replace('\n', '\n      ')
                    lines.append(f"      {interaction_str}")

            # Electron-electron interactions
            if self.electron_electron:
                lines.append(f"\nElectron-Electron ({len(self.electron_electron)}):")
                for i, (e1_idx, e2_idx, interaction) in enumerate(self.electron_electron):
                    lines.append(f"  {i + 1}. e{e1_idx} ↔ e{e2_idx}:")
                    interaction_str = str(interaction).replace('\n', '\n      ')
                    lines.append(f"      {interaction_str}")

            # Nucleus-nucleus interactions
            if self.nuclei_nuclei:
                lines.append(f"\nNucleus-Nucleus ({len(self.nuclei_nuclei)}):")
                for i, (n1_idx, n2_idx, interaction) in enumerate(self.nuclei_nuclei):
                    lines.append(f"  {i + 1}. n{n1_idx} ↔ n{n2_idx}:")
                    interaction_str = str(interaction).replace('\n', '\n      ')
                    lines.append(f"      {interaction_str}")
        else:
            lines.append(f"\nINTERACTIONS: None")

        lines.append("\n" + "=" * 60)
        return '\n'.join(lines)


class SpinSystemOrientator:
    """Transforms spin systems to multiple molecular orientations.

    Generates orientation-dependent versions of all interactions (g-tensors, hyperfine, ZFS)
    by applying rotation matrices. Used for powder-averaged simulations where spectra
    are averaged over many molecular orientations relative to the magnetic field.

    The output is a modified :class:`SpinSystem` where each interaction is replaced by
    a :class:`MultiOrientedInteraction` containing precomputed tensors for all orientations.
    """
    def __init__(
            self, orientation_method: tp.Callable[[torch.Tensor, torch.Tensor], torch.Tensor] =
            utils.apply_expanded_rotations
    ):
        """
        :param orientation_method: the method to used to rotate interactions. Can be expanded rotations or the
        """

    def __call__(self, spin_system: SpinSystem, rotation_matrices: torch.Tensor) -> SpinSystem:
        """
        :param spin_system: spin_system with interactions.

        :param rotation_matrices: rotation_matrices that rotate spin system
        :return: modified spin system with all rotated interactions
        """
        spin_system = self.transform_spin_system_to_oriented(copy.deepcopy(spin_system), rotation_matrices)
        return spin_system

    def interactions_to_multioriented(
            self,
            interactions: tp.List[BaseInteraction],
            rotation_matrices: torch.Tensor
    ) -> tp.Tuple[torch.Tensor, tp.List[tp.Optional[torch.Tensor]], tp.List[torch.Tensor]]:
        """Precompute rotated tensors and strain derivatives for multiple orientations."""
        oriented_tensors = torch.stack([
            utils.apply_expanded_rotations(rotation_matrices, interaction.tensor)
            for interaction in interactions
        ])

        not_none_strained = [
            interaction.strained_derivatives for interaction in interactions if interaction.strained_derivatives is not None
        ]
        none_strained_flag = [
            True if interaction.strained_derivatives is None else False for interaction in interactions
        ]
        if not_none_strained:
            strained_tensors = torch.stack(not_none_strained, dim=0)
            strained_tensors = utils.apply_expanded_rotations(rotation_matrices, strained_tensors)
            strained_tensors = strained_tensors.transpose(-3, -4)
            strained_iterator = iter(strained_tensors)
            strained_derivatives = [None if x else next(strained_iterator) for x in none_strained_flag]
        else:
            strained_derivatives = [None] * len(interactions)

        strain_correlations = [interaction.strain_correlation for interaction in interactions]

        return oriented_tensors, strained_derivatives, strain_correlations

    def _apply_reverse_transform(
            self,
            spin_system: SpinSystem,
            new_interactions: tp.List[MultiOrientedInteraction]
    ) -> SpinSystem:
        """Reassign transformed interactions to original spin system groups.

        :param spin_system : SpinSystem
            Original spin system structure.
        :param new_interactions : list of MultiOrientedInteraction
            Transformed interactions in order: g-tensors, electron-nuclei, electron-electron.

        :return:
        SpinSystem
            Spin system with interactions replaced by multi-oriented versions.
        """
        n_g = len(spin_system.g_tensors)
        n_nuc = len(spin_system.electron_nuclei_interactions)
        n_ee = len(spin_system.electron_electron_interactions)

        g_tensors = new_interactions[:n_g]
        electron_nuclei = new_interactions[n_g:n_g + n_nuc]
        electron_electron = new_interactions[n_g + n_nuc:]

        spin_system.g_tensors = nn.ModuleList(g_tensors)
        spin_system.electron_nuclei_interactions = nn.ModuleList(electron_nuclei)
        spin_system.electron_electron_interactions = nn.ModuleList(electron_electron)

        return spin_system

    def transform_spin_system_to_oriented(
            self,
            spin_system: SpinSystem,
            rotation_matrices: torch.Tensor
    ) -> SpinSystem:
        """Main transformation method: convert spin system to multi-oriented representation.

        :param spin_system:  SpinSystem
            Original spin system with interactions in molecular frame.
        :param rotation_matrices:  torch.Tensor
            Rotation matrices for each orientation. Shape: ``(n_orientations, 3, 3)``.

        :return:
        SpinSystem
            Spin system where each interaction is replaced by a :class:`MultiOrientedInteraction`
            containing precomputed tensors for all orientations.
        """
        interactions = (
                list(spin_system.g_tensors) +
                list(spin_system.electron_nuclei_interactions) +
                list(spin_system.electron_electron_interactions)
        )

        config_shape = torch.Size([*spin_system.config_shape, rotation_matrices.shape[0]])

        oriented_tensors, strained_derivatives, strain_correlations = self.interactions_to_multioriented(
            interactions,
            rotation_matrices
        )

        new_interactions = [
            MultiOrientedInteraction(
                oriented_tensor=oriented_tensor,
                strain=interaction.strain,
                strained_derivatives=strained_derivative,
                config_shape=config_shape,
                strain_correlation=strain_corr,
                device=spin_system.device
            )
            for oriented_tensor, strained_derivative, strain_corr, interaction in zip(
                oriented_tensors, strained_derivatives, strain_correlations, interactions
            )
        ]
        return self._apply_reverse_transform(spin_system, new_interactions)


class BaseSample(nn.Module):
    """Base class representing a magnetic resonance sample in a fixed molecular frame.

    This class encapsulates a spin system along with optional broadening mechanisms:
    - Homogeneous (Lorentzian) and inhomogeneous (Gaussian) line broadening,
    - Residual unresolved strain modeled as anisotropic broadening of the Hamiltonian.

    It provides methods to construct the full spin Hamiltonian in the form:

        `H = F + B_x G_x + B_y G_y + B_z G_z`,

    where `F` is the field-independent part and `G_x, G_y, G_z` are Zeeman interaction operators.

    The basis used throughout is the product-state basis of individual electron and nuclear spin projections.
    """
    def __init__(self, base_spin_system: SpinSystem,
                 ham_strain: tp.Optional[tp.Union[torch.Tensor, float]] = None,
                 gauss: tp.Optional[tp.Union[torch.Tensor, float]] = None,
                 lorentz: tp.Optional[tp.Union[torch.Tensor, float]] = None,
                 spin_system_frame: tp.Optional[tp.Union[torch.Tensor, list[float]]] = None,
                 device=torch.device("cpu"), dtype: torch.dtype = torch.float32,
                 *args, **kwargs):
        """
        :param base_spin_system:

        SpinSystem
            The spin system describing electrons, nuclei, and their interactions.

        :param ham_strain:
        torch.Tensor, float, optional
            Anisotropic line width, due to the unresolved hyperfine interactions.
            The tensor components, provided in one of the following forms:
              - A scalar (for isotropic interaction).
              - A sequence of two values (axial and z components).
              - A sequence of three values (principal components).

        :param gauss:
        torch.Tensor, float, optional
            Gaussian broadening parameter(s). Defines inhomogeneous linewidth
            contributions (e.g., due to static disorder). Default is `None`.

        :param lorentz:
        torch.Tensor, float, optional
            Lorentzian broadening parameter(s). Defines homogeneous linewidth
            contributions (e.g., due to relaxation). Default is `None`

        :param spin_system_frame:
        torch.Tensor | Sequence[float] optional
            Orientation of the spin system. Can be provided as:
              - A 1D tensor of shape (3,) representing Euler angles in ZYZ' convention.
              - A 2D tensor of shape (3, 3) representing a rotation matrix.
            Default is `None`, meaning lab frame.

        This parameter is set the orientation of the spin system relative to the molecular frame

        This parameters save base_spin_system not rotated while it rotates multioriented modified_spin_system

        :param device: device to compute (cpu / gpu)
        :param dtype: dtype
        :param args:
        :param kwargs:
        """
        super().__init__()
        self._construct_spin_system_rot_matrix(frame=spin_system_frame,
                                               config_shape=base_spin_system.config_shape, dtype=dtype, device=device)
        self.base_spin_system = base_spin_system
        self.modified_spin_system = copy.deepcopy(base_spin_system)
        self.register_buffer("_ham_strain", self._init_ham_str(ham_strain, device, dtype))

        self.base_ham_strain = copy.deepcopy(self._ham_strain)
        self.register_buffer("gauss", self._init_gauss_lorentz(gauss, device, dtype))
        self.register_buffer("lorentz", self._init_gauss_lorentz(lorentz, device, dtype))
        self.register_buffer("secular_threshold", torch.tensor(1e-9, device=device, dtype=dtype))
        self.to(device)
        self.to(dtype)

    @property
    def device(self):
        """Get the device where the spin system tensors are stored.

        :return: The torch.device (e.g., 'cpu' or 'cuda') used for computation.
        """
        return self.base_spin_system.device

    @property
    def complex_dtype(self):
        """Get the complex data type corresponding to the system float precision."""
        return self.base_spin_system.complex_dtype

    @property
    def dtype(self):
        """Get the floating-point data type of interaction parameters."""
        return self.base_spin_system.dtype

    @property
    def spin_system_dim(self):
        """Get the dimension of the total Hilbert space.

        Computed as the dimension of spin-operators in Hilbert Space

        :return: An integer representing the size of the full spin basis (e.g., 4 for two spin-1/2 particles).
        """
        return self.modified_spin_system.spin_system_dim

    @property
    def config_shape(self) -> torch.Size:
        """Get the batch configuration shape of the spin system including orientations

        This describes parameter broadcasting dimensions (e.g., for ensembles),
        Hilbert space axes.

        :return: A tuple-like shape (e.g., (N_samples,) or (N_batch, N_sites)).
        """
        return self.modified_spin_system.config_shape

    def _init_gauss_lorentz(
            self, width: tp.Optional[tp.Union[torch.Tensor, float]],
            device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if width is None:
            if self.base_spin_system.config_shape:
                width = torch.zeros(
                    (*self.base_spin_system.config_shape, ), device=device, dtype=dtype)
            else:
                width = torch.tensor(0.0, device=device, dtype=dtype)
        else:
            width = torch.tensor(width, device=device, dtype=dtype)
            if width.shape != self.base_spin_system.config_shape:
                raise ValueError(
                    f"width batch shape must be equal to base_spin_system config shape."
                    f"The current width shape is {width.shape}, expected {self.base_spin_system.config_shape}"
                )
        return width

    def _init_ham_str(
            self, ham_strain: tp.Optional[tp.Union[torch.Tensor, float]],
            device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if ham_strain is None:
            ham_strain = torch.zeros(
                (*self.base_spin_system.config_shape, 3), device=device, dtype=dtype
            )
        else:
            ham_strain = init_tensor(ham_strain, device=device, dtype=dtype)
            if ham_strain.shape[:-1] != self.base_spin_system.config_shape:
                raise ValueError(f"ham_strain batch shape must be equel to base_spin_system config shape")
        return ham_strain

    def _construct_spin_system_rot_matrix(
            self, frame: tp.Optional[torch.tensor],
            config_shape: tp.Iterable,
            device: torch.device,
            dtype: torch.dtype
    ):
        """
        :param frame: spin system euler angles or full frame
        :param config_shape:
        :param device:
        :param dtype:
        :return: None
        """
        if frame is None:
            _frame = None
            _rot_matrix = None
        else:
            if isinstance(frame, torch.Tensor):
                if frame.shape[-2:] == (3, 3) and not config_shape:
                    _frame = utils.rotation_matrix_to_euler_angles(frame)
                    _rot_matrix = frame.to(dtype)

                elif frame.shape == (*config_shape, 3):
                    _frame = frame.to(dtype)
                    _rot_matrix = utils.euler_angles_to_matrix(_frame).to(dtype)

                else:
                    raise ValueError(
                        "frame must be either:\n"
                        "  • None (→ identity rotation),\n"
                        "  • a tensor of Euler angles with shape batch×3,\n"
                        "  • or a tensor of rotation matrices with shape batch×3×3."
                    )
            elif isinstance(frame, collections.abc.Sequence):
                if len(frame) != 3:
                    raise ValueError("frame must have exactly 3 values")
                _frame = torch.tensor(frame, dtype=dtype, device=device)
                _rot_matrix = utils.euler_angles_to_matrix(_frame).to(dtype)
            else:
                raise ValueError("frame must be a Sequence of 3 values, a torch.Tensor, or None.")

        self.register_buffer("_spin_system_frame", _frame)
        self.register_buffer("_spin_system_rot_matrix", _rot_matrix)

    @property
    def spin_system_rot_matrix(self) -> tp.Optional[torch.Tensor]:
        """
        Return the rotation matrix which rotate spin system relative to frame

        :return: rotation matrix for the spin system
        """
        return self._spin_system_rot_matrix

    def is_equivalent(self, other: BaseSample, rtol: float = 1e-5, atol: float = 1e-6) -> bool:
        """Check equality of two BaseSample instances.

        Compares the base and modified spin systems, broadening parameters,
        strain, and frame configurations using appropriate numerical tolerances.

        :param other: Another BaseSample instance.
        :return: True if all parameters match within numerical tolerances.
        """
        if type(self) is not type(other):
            return False

        if not self.base_spin_system.is_equivalent(other.base_spin_system, rtol, atol):
            return False

        return all([
            utils.are_optional_tensors_close(self.base_ham_strain, other.base_ham_strain, rtol, atol),
            utils.are_optional_tensors_close(self.gauss, other.gauss, rtol, atol),
            utils.are_optional_tensors_close(self.lorentz, other.lorentz, rtol, atol),
            utils.are_optional_tensors_close(self._spin_system_frame, other._spin_system_frame, rtol, atol),
        ])

    def update(self,
               g_tensors: list[BaseInteraction] = None,
               electron_nuclei: tp.Optional[list[tuple[int, int, BaseInteraction]]] = None,
               electron_electron: tp.Optional[list[tuple[int, int, BaseInteraction]]] = None,
               nuclei_nuclei: tp.Optional[list[tuple[int, int, BaseInteraction]]] = None,
               ham_strain: tp.Optional[torch.Tensor] = None,
               gauss: tp.Union[torch.Tensor, float] = None,
               lorentz: tp.Union[torch.Tensor, float] = None
               ):
        """Update the parameters of a sample.

        No recomputation of spin vectors does not occur
        :param g_tensors:
        list[BaseInteraction]
            g-tensors corresponding to each electron in `electrons`.
            Each element must be an instance of `BaseInteraction` (e.g., `Interaction`).

        :param electron_nuclei:
        list[tuple[int, int, BaseInteraction]], optional
            Hyperfine interactions between electrons and nuclei.
            Each tuple is of the form (electron_index, nucleus_index, interaction_tensor).
            Default is `None`.

        :param electron_electron:
        list[tuple[int, int, BaseInteraction]], optional
            Dipolar or exchange interactions between pairs of electrons.
            Each tuple is of the form (electron_index, electron_index, interaction_tensor).
            Default is `None`.

        :param nuclei_nuclei:
        list[tuple[int, int, BaseInteraction]], optional
            Dipolar or J-coupling interactions between pairs of nuclei.
            Each tuple is of the form (nucleus_index, nucleus_index, interaction_tensor).
            Default is `None`

        :param ham_strain:
        torch.Tensor, optional
            Anisotropic line width, due to the unresolved hyperfine interactions.
            The tensor components, provided in one of the following forms:
              - A scalar (for isotropic interaction).
              - A sequence of two values (axial and z components).
              - A sequence of three values (principal components).

        :param gauss:
        torch.Tensor, optional
            Gaussian broadening parameter(s). Defines inhomogeneous linewidth
            contributions (e.g., due to static disorder). Default is `None`.

        :param lorentz:
        torch.Tensor, optional
            Lorentzian broadening parameter(s). Defines homogeneous linewidth
            contributions (e.g., due to relaxation). Default is `None`
        """
        raise NotImplementedError

    def build_electron_electron(self) -> torch.Tensor:
        """Constructs the zero-field Hamiltonian F."""
        F = torch.zeros((*self.config_shape,
                         self.spin_system_dim, self.spin_system_dim),
                        dtype=self.complex_dtype,
                        device=self.device)
        operator_cache = self.modified_spin_system.operator_cache
        for e_idx_1, e_idx_2, interaction in self.modified_spin_system.electron_electron:
            interaction = interaction.tensor.to(self.complex_dtype)
            F += scalar_tensor_multiplication(
                operator_cache[e_idx_1],
                operator_cache[e_idx_2],
                interaction)
        return F

    def build_electron_nuclei(self) -> torch.Tensor:
        """Constructs the hyperfine interaction."""
        F = torch.zeros((*self.config_shape,
                         self.spin_system_dim, self.spin_system_dim),
                        dtype=self.complex_dtype,
                        device=self.device)
        operator_cache = self.modified_spin_system.operator_cache
        for e_idx, n_idx, interaction in self.modified_spin_system.electron_nuclei:
            interaction = interaction.tensor.to(self.complex_dtype)
            F += scalar_tensor_multiplication(
                operator_cache[e_idx],
                operator_cache[len(self.modified_spin_system.electrons) + n_idx],
                interaction)
        return F

    def build_nuclei_nuclei(self) -> torch.Tensor:
        """Constructs the nuclei-nuclei interactions F."""
        F = torch.zeros((*self.config_shape,
                         self.spin_system_dim, self.spin_system_dim),
                        dtype=self.complex_dtype,
                        device=self.device)
        operator_cache = self.modified_spin_system.operator_cache
        for n_idx_1, n_idx_2, interaction in self.modified_spin_system.nuclei_nuclei:
            interaction = interaction.tensor.to(self.complex_dtype)
            F += scalar_tensor_multiplication(
                operator_cache[len(self.modified_spin_system.electrons) + n_idx_1],
                operator_cache[len(self.modified_spin_system.electrons) + n_idx_2],
                interaction)
        return F

    def build_first_order_interactions(self) -> torch.Tensor:
        """Constructs the zero-field Hamiltonian F of the first order
        operators."""
        return self.build_nuclei_nuclei() + self.build_electron_nuclei() + self.build_electron_electron()

    def build_zero_field_term(self) -> torch.Tensor:
        """Constructs the zero-field Hamiltonian F."""
        return self.build_first_order_interactions() + self.base_spin_system.energy_shift

    def _build_electron_zeeman_terms(self) -> torch.Tensor:
        """Constructs the Zeeman interaction terms Gx, Gy, Gz.

        for electron spins with give g-tensors
        """
        G = torch.zeros((*self.config_shape, 3,
                         self.spin_system_dim, self.spin_system_dim),
                        dtype=self.complex_dtype,
                        device=self.device)
        operator_cache = self.modified_spin_system.operator_cache
        for idx, g_tensor in enumerate(self.modified_spin_system.g_tensors):
            g = g_tensor.tensor.to(self.complex_dtype)
            G += transform_tensor_components(operator_cache[idx], g)
        G *= (constants.BOHR / constants.PLANCK)
        return G

    def _build_nucleus_zeeman_terms(self) -> torch.Tensor:
        """Constructs the Nucleus interaction terms Gx, Gy, Gz.

        for nucleus spins
        """
        G = torch.zeros((*self.config_shape, 3,
                         self.spin_system_dim,
                         self.spin_system_dim),
                        dtype=self.complex_dtype,
                        device=self.device)
        operator_cache = self.modified_spin_system.operator_cache
        for idx, nucleus in enumerate(self.modified_spin_system.nuclei):
            g = nucleus.g_factor
            G += operator_cache[len(self.modified_spin_system.electrons) + idx] * g
        G *= (constants.NUCLEAR_MAGNETRON / constants.PLANCK)
        return G

    def build_zeeman_terms(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Constructs the Zeeman interaction terms Gx, Gy, Gz.
        """
        G = self._build_electron_zeeman_terms() + self._build_nucleus_zeeman_terms()
        return G[..., 0, :, :], G[..., 1, :, :], G[..., 2, :, :]

    def calculate_derivative_max(self):
        """Calculate the maximum value of the energy derivatives with respect
        to magnetic field.

        It is assumed that B has direction along z-axis
        :return: the maximum value of the energy derivatives with
            respect to magnetic field
        """
        electron_contrib = 0
        for idx, electron in enumerate(self.modified_spin_system.electrons):
            electron_contrib += electron.spin * torch.sum(
                self.modified_spin_system.g_tensors[idx].tensor[..., :, 0], dim=-1, keepdim=True).abs()

        nuclei_contrib = 0
        for idx, nucleus in enumerate(self.modified_spin_system.nuclei):
            nuclei_contrib += nucleus.spin * nucleus.g_factor.abs()
        return (electron_contrib * (constants.BOHR / constants.PLANCK) +
            nuclei_contrib * (constants.NUCLEAR_MAGNETRON / constants.PLANCK)).squeeze(dim=-1)

    def get_hamiltonian_terms(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns F, Gx, Gy, Gz.

        F is the magnetic field-independent part of the spin Hamiltonian.
        Gx, Gy, and Gz are the Zeeman coupling matrices corresponding to the x, y, and z components
        of the external magnetic field (Bx, By, Bz). The full EPR Hamiltonian is expressed as:

            H = F + Gx * Bx + Gy * By + Gz * Bz

        All matrices are represented in the basis of product states formed from individual electron
        and nuclear spin projections.

        F includes contributions from:
          - Zero-field splitting (for systems with S >= 1)
          - Electron-nuclear hyperfine interactions
          - Nuclear-nuclear dipolar or scalar couplings
          - Other field-independent terms (e.g., exchange)

        Gx, Gy, and Gz are defined as the sum of electron and nuclear Zeeman operators weighted by
        their respective g-tensors and magnetogyric ratios:

          - For electrons: the contribution is proportional to the electron g-tensor (typically anisotropic),
            such that the electron Zeeman term is mu_B * g_tensor @ B, where mu_B is the Bohr magneton.

          - For nuclei: the contribution is gamma_n * I, where gamma_n is the nuclear gyromagnetic ratio
            and I is the nuclear spin operator.
        """
        return self.build_zero_field_term(), *self.build_zeeman_terms()

    def get_hamiltonian_terms_secular(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns F0, Gx, Gy, Gz after secular approximation.

        F0 is a part of magnetic field free term, which commutes with Gz
        Gx, Gy, Gz are terms multiplied to Bx, By, Bz respectively

        Applies two transformations:
        1. For Zeeman terms (Gx, Gy, Gz): Zero matrix elements where the absolute value
           of the corresponding total spin projection operator (Sx, Sy, Sz) is below
           threshold. This enforces approximate commutation relations:
              [Gx, Sx] ≈ 0, [Gy, Sy] ≈ 0, [Gz, Sz] ≈ 0
           For single spins with isotropic g-tensors this is exact. For anisotropic cases and one particle,
           it retains only the diagonal part of the g-tensor in the spin projection basis

        2. For zero-field term (F0): Zero matrix elements where the difference in diagonal
           elements of Gz exceeds threshold. This enforces [F0, Gz] ≈ 0.

        Both steps use self.secular_threshold. The approach assumes Gz is approximately diagonal
        """
        Gx, Gy, Gz = self.build_zeeman_terms()
        n_spins = len(self.base_spin_system.electrons) + len(self.base_spin_system.nuclei)
        dim = Gx.shape[-1]

        total_S = torch.zeros(3, dim, dim, dtype=Gx.dtype, device=Gx.device)
        [total_S.add_(self.base_spin_system.operator_cache[idx]) for idx in range(n_spins)]

        components = [Gx, Gy, Gz]
        abs_tensor = torch.zeros(
            dim, dim, dtype=self.base_spin_system.dtype, device=self.base_spin_system.device)
        for i, comp in enumerate(components):
            mask = torch.abs(total_S[i], out=abs_tensor) <= self.secular_threshold
            comp.masked_fill_(mask, 0)

        diag = torch.diagonal(Gz, dim1=-2, dim2=-1).real
        diag_i = diag.unsqueeze(-1)
        diag_j = diag.unsqueeze(-2)

        diag_diff = torch.abs(diag_i - diag_j)
        mask = diag_diff.lt_(self.secular_threshold).bool()

        F0 = self.build_zero_field_term()
        F0.masked_fill_(~mask, 0)

        return F0, Gx, Gy, Gz

    def build_field_dep_strain(self) -> tp.Generator[tuple[torch.Tensor, torch.Tensor, torch.Tensor], None, None]:
        """Calculate electron Zeeman field dependant strained part.
        :return:
        """
        operator_cache = self.modified_spin_system.operator_cache
        for idx, g_tensor in enumerate(self.modified_spin_system.g_tensors):
            g_derivatives = g_tensor.strained_derivatives
            if g_derivatives is None:
                pass
            else:
                g_derivatives = g_derivatives.to(self.complex_dtype)
                strain_matrix =\
                    g_tensor.strain_correlation.to(self.complex_dtype) *\
                    g_tensor.strain.unsqueeze(-2).to(self.complex_dtype)
                yield (
                    strain_matrix,
                    operator_cache[idx],
                    g_derivatives[..., :, 2, :] * constants.BOHR / constants.PLANCK)

    def build_zero_field_strain(self) -> torch.Tensor:
        """Constructs the zero-field strained part."""
        yield from self.build_electron_nuclei_strain()
        yield from self.build_electron_electron_strain()

    def build_electron_nuclei_strain(self) -> torch.Tensor:
        """Constructs the nuclei strained part."""
        operator_cache = self.modified_spin_system.operator_cache
        for e_idx, n_idx, electron_nuclei_interaction in self.modified_spin_system.electron_nuclei:
            electron_nuclei_derivatives = electron_nuclei_interaction.strained_derivatives
            if electron_nuclei_derivatives is None:
                pass
            else:
                electron_nuclei_derivatives = electron_nuclei_derivatives.to(self.complex_dtype)
                strain_matrix =\
                    electron_nuclei_interaction.strain_correlation.to(self.complex_dtype) *\
                    electron_nuclei_interaction.strain.unsqueeze(-2).to(self.complex_dtype)
                yield (
                    strain_matrix,
                    operator_cache[e_idx],
                    operator_cache[len(self.modified_spin_system.electrons) + n_idx],
                    electron_nuclei_derivatives)

    def build_electron_electron_strain(self) -> torch.Tensor:
        """Constructs the electron-electron strained part."""
        operator_cache = self.modified_spin_system.operator_cache
        for e_idx_1, e_idx_2, electron_electron_interaction in self.modified_spin_system.electron_electron:
            electron_electron_derivatives = electron_electron_interaction.strained_derivatives
            if electron_electron_derivatives is None:
                pass
            else:
                electron_electron_derivatives = electron_electron_derivatives.to(self.complex_dtype)
                strain_matrix =\
                    electron_electron_interaction.strain_correlation.to(self.complex_dtype) *\
                    electron_electron_interaction.strain.unsqueeze(-2).to(self.complex_dtype)
                yield (
                    strain_matrix,
                    operator_cache[e_idx_1],
                    operator_cache[e_idx_2],
                    electron_electron_derivatives)

    def get_spin_multiplet_basis(self) -> torch.Tensor:
        """Compute eigenvector in the |S, M⟩ basis (total spin and projection
        basis).

        This method diagonalizes a combination of sS² and Sz operators to obtain
        eigenvectors ordered by total spin quantum number S, then by magnetic
        quantum number M (spin projection).

        :return: torch.Tensor
        Matrix of eigenvectors with shape [spin_dim, spin_dim], where each column
        represents an eigenstate. States are ordered in ascending order of S,
        then in ascending order of M within each S manifold.

        Examples
        --------
        For two spin-1/2 electrons:
            Returns basis ordered as: |S=0, M=0⟩, |S=1, M=-1⟩, |S=1, M=0⟩, |S=1, M=1⟩
        """
        return self.base_spin_system.get_spin_multiplet_basis()

    def get_product_state_basis(self) -> torch.Tensor:
        """Return the identity matrix representing the computational product
        state basis.

        The product state basis is |ms1, ms2, ..., msk, is1, ..., ism⟩ where:
        - ms1, ms2, ..., msk are magnetic quantum numbers for electrons through k
        - is1, is2, ..., ism are magnetic quantum numbers for nuclei through m
        Each state corresponds to a definite spin projection for each particle.
        :return: torch.Tensor
        Identity matrix with shape [spin_system_dim, spin_system_dim]. The identity
        matrix indicates that the current representation is already in the product
        state basis.

        Examples
        --------
        For one spin-1/2 electron and one spin-1/2 nucleus:
            Basis states: |↑, ↑⟩, |↑, ↓⟩, |↓, ↑⟩, |↓, ↓⟩
        """
        return self.base_spin_system.get_product_state_basis()

    def get_xyz_basis(self) -> torch.Tensor:
        """Get vectors (S), Tx, Ty, Tz for a spin system.

        Dispatches to the appropriate basis constructor depending on the system:

        - A single electron with spin=1 uses the direct spin-1 basis
          (see :meth:`_get_xyz_basis_triplet`).
        - Two electrons each with spin=1/2 uses the coupled triplet/singlet
          basis (see :meth:`_get_xyz_basis_two_half_spins`).

        :return: torch.Tensor
            Transition basis matrix.
            - For spin-1: shape ``(3, 3)``, columns are ``[Tx, Ty, Tz]``.
            - For two spin-1/2: shape ``(4, 4)``, columns are
              ``[S, Tx, Ty, Tz]``.

        :raises ValueError:
            If the system does not match a supported spin configuration.

        Examples
        --------
        For a spin-1 system::

            T = system.get_xyz_basis()  # shape (3, 3)
            Tx, Ty, Tz = T[:, 0], T[:, 1], T[:, 2]

        For two spin-1/2 electrons::

            T = system.get_xyz_basis()  # shape (4, 4)
            S, Tx, Ty, Tz = T[:, 0], T[:, 1], T[:, 2], T[:, 3]
        """
        return self.base_spin_system.get_xyz_basis()

    def get_zero_field_splitting_basis(self) -> torch.Tensor:
        """
        :return: The shape is [..., N, N]
        The eigen basis of zero field splitting. The order from the
        lowest eigen value to higher eigen value
        """
        zero_field_term = self.build_zero_field_term()
        _, zfs_eigenvectors = torch.linalg.eigh(zero_field_term)
        return zfs_eigenvectors

    def get_zeeman_basis(self) -> torch.Tensor:
        """
        :return: The shape is [..., N, N]
        The eigen basis of Z-projection of Zeeman operator. This is basis in infinite magnetic field
        The order from the lowest eigen value to higher eigen value
        """
        _, _, Gz = self.build_zeeman_terms()
        _, zeeman_eigenvectors = torch.linalg.eigh(Gz)
        return zeeman_eigenvectors

    def __repr__(self):
        spin_system_summary = str(self.base_spin_system)

        lines = []
        lines.append(spin_system_summary)
        lines.append("\n" + "=" * 60)
        lines.append("GENERAL INFO: ")
        lines.append("=" * 60)

        is_batched = self.base_spin_system.config_shape

        if is_batched:
            lorentz = self.lorentz.flatten(0, -1)
            gauss = self.gauss.flatten(0, -1)
            batch_size = lorentz.shape[0] if hasattr(self.lorentz, 'shape') else len(self.lorentz)
            lines.append(f"BATCHED (batch_size={batch_size}) - showing first instance:")

            lines.append(f"lorentz: {lorentz[0].item():.5f} T (Hz)")
            lines.append(f"gauss: {gauss[0].item():.5f} T (Hz)")

            ham_str = self.base_ham_strain.flatten(0, -2)[0]
            ham_components = [f"{val:.4e}" if abs(val) >= 1e4 else f"{val:.4f}"
                              for val in ham_str.tolist()]
            ham_dim = self.base_ham_strain.shape
            lines.append(f"ham_str (dim={ham_dim}): {ham_components} Hz - first example components")
        else:
            lines.append(f"lorentz: {self.lorentz.item():.5f} T")
            lines.append(f"gauss: {self.gauss.item():.5f} T")

            ham_str = self.base_ham_strain
            ham_components = [f"{val:.4e}" if abs(val) >= 1e4 else f"{val:.4f}"
                              for val in ham_str.tolist()]
            lines.append(f"ham_str: {ham_components} Hz")
        if self._spin_system_frame is not None:
            lines.append(f"spin system frame: {self._spin_system_frame.tolist()} rad")
        return '\n'.join(lines)

    @classmethod
    def from_serialized_sample(cls, serialized_sample: SerializedSample) -> BaseSample:
        return serialized_sample.to_mars_sample()

    def to_serialized_sample(self) -> SerializedSample:
        from .serialization.serialization import SerializedSample
        return SerializedSample.from_mars_sample(self)

    @classmethod
    def from_graph_sample(cls, graph_sample: GraphSample) -> BaseSample:
        return graph_sample.to_mars_sample()

    def to_graph_sample(self) -> GraphSample:
        from .serialization.graph_representation import GraphSample
        return GraphSample.from_mars_sample(self)


class LimitedDict(collections.OrderedDict):
    def __init__(self, maxsize: int):
        super().__init__()
        self.maxsize = maxsize

    def __setitem__(self, key: tp.Any, value: tp.Any):
        if key in self:
            self.move_to_end(key)
        else:
            if len(self) >= self.maxsize:
                self.popitem(last=False)
        super().__setitem__(key, value)


class MultiOrientedSample(BaseSample):
    """Represents a solid-state sample (e.g., powder, glass, or polycrystal)
    averaged over multiple molecular orientations.

    The sample is constructed by rotating the entire spin system (including all interactions) through a set of
    orientation angles defined by a mesh. For each orientation, the Hamiltonian
    terms (`F`, `Gx`, `Gy`, `Gz`) are recomputed, enabling accurate simulation of orientation-dependent spectra.

    This class is the standard entry point for simulating frozen-solution or disordered solid EPR spectra.
    """
    _mesh_cache = LimitedDict(maxsize=3)

    def __init__(self, base_spin_system: SpinSystem,
                 ham_strain: tp.Optional[tp.Union[torch.Tensor, float]] = None,
                 gauss: tp.Optional[tp.Union[torch.Tensor, float]] = None,
                 lorentz: tp.Optional[tp.Union[torch.Tensor, float]] = None,
                 spin_system_frame: tp.Optional[tp.Union[torch.Tensor, list[float]]] = None,
                 mesh: tp.Optional[tp.Union[BaseMesh, tuple[int, int]]] = None,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32,
                 ):
        """
        :param base_spin_system:

        SpinSystem
            The spin system describing electrons, nuclei, and their interactions.

        :param ham_strain:
        torch.Tensor, optional
            Anisotropic line width, due to the unresolved hyperfine interactions.
            The tensor components, provided in one of the following forms:
              - A scalar (for isotropic interaction).
              - A sequence of two values (axial and z components).
              - A sequence of three values (principal components).
            The values are given as FWHM (full width at half maximum) of corresponding distribution and measured in Hz

        :param gauss:
        torch.Tensor, optional
            Gaussian broadening parameter(s). Defines inhomogeneous linewidth
            contributions (e.g., due to static disorder). Default is `None`.
            Values are provided as the full width at half maximum (FWHM) and are expressed in:
            - tesla (T) for field-dependent spectra,
            - hertz (Hz) for frequency-dependent spectra.

        :param lorentz:
        torch.Tensor, optional
            Lorentzian broadening parameter(s). Defines homogeneous linewidth
            contributions (e.g., due to relaxation). Default is `None`
            Values are provided as the full width at half maximum (FWHM) and are expressed in:
            - tesla (T) for field-dependent spectra,
            - hertz (Hz) for frequency-dependent spectra.

        :param spin_system_frame:
        torch.Tensor | Sequence[float] optional
            Orientation of the spin system. Can be provided as:
              - A 1D tensor of shape (3,) representing Euler angles in ZYZ' convention.
              - A 2D tensor of shape (3, 3) representing a rotation matrix.
            Default is `None`, meaning lab frame.

        This parameter is set the orientation of the spin system relative to the molecular frame

        :param mesh: The mesh to perform rotations for powder samples. It can be:
           -tuple[initial_grid_frequency, interpolation_grid_frequency],
           where initial_grid_frequency is the size of the initial mesh,
           interpolation_grid_frequency is the size of the interpolation mesh
           For this case mesh will be initialize as DelaunayMesh with given sizes

           -Inheritor of Base Mesh

        If it is None it will be initialize as DelaunayMesh with initial_grid_frequency = 20

        :param device: device to compute (cpu / gpu)
        """
        super().__init__(base_spin_system, ham_strain, gauss, lorentz, spin_system_frame, device=device, dtype=dtype)
        self.mesh = self._init_mesh(mesh, device=device, dtype=dtype)
        rotation_matrices = self.mesh.rotation_matrices

        self._ham_strain = self._expand_hamiltonian_strain(
            self.base_ham_strain,
            self.orientation_vector(rotation_matrices)
        )
        if self._spin_system_frame is None:
            self.modified_spin_system = SpinSystemOrientator()(base_spin_system, rotation_matrices)
        else:
            self.modified_spin_system = SpinSystemOrientator()(
                base_spin_system, torch.matmul(rotation_matrices, self._spin_system_rot_matrix)
            )

    @classmethod
    def _get_cached_mesh(
        cls,
        device: torch.device,
        dtype: torch.dtype,
        initial_grid_frequency: int,
        interpolation_grid_frequency: int,
        interpolate: bool
    ) -> mesher.DelaunayMesh:
        """
        Retrieve a Delaunay mesh from the class-specific cache, or create and cache it if not present.

        This method ensures that identical mesh configurations (defined by the input parameters)
        are reused across calls, avoiding redundant mesh generation.
        Retrieve or create a cached DelaunayMesh for the given parameters.

        :param device: The device (e.g., CPU or CUDA) on which mesh tensors are allocated.
        :param dtype: The data type of mesh coordinates (e.g., `torch.float32`).
        :param initial_grid_frequency: Number of points per dimension in the initial coarse grid used for triangulation.
        :param interpolation_grid_frequency:
        Number of points per dimension in the refined interpolation grid (used only if interpolation is enabled).
        :param interpolate: Flag indicating whether to generate a mesh suitable for interpolation on a finer grid.
        :return: A cached or newly created ``mesher.DelaunayMesh`` instance configured with the specified
        """
        key = (
            str(device),
            str(dtype).replace("torch.", ""),
            initial_grid_frequency,
            interpolation_grid_frequency,
            interpolate
        )

        if key not in cls._mesh_cache:
            cls._mesh_cache[key] = mesher.DelaunayMesh(
                interpolate=interpolate,
                initial_grid_frequency=initial_grid_frequency,
                interpolation_grid_frequency=interpolation_grid_frequency,
                device=device,
                dtype=dtype
            )
        return cls._mesh_cache[key]

    def _init_mesh(
            self, mesh: tp.Optional[tp.Union[BaseMesh, tuple[int, int]]],
            device: torch.device, dtype: torch.dtype
    ):
        """
        Initialize or retrieve a mesh instance for orientation sampling.

        This method handles three input cases:

        1. **``mesh=None``**: Creates a default Delaunay mesh with
           ``initial_grid_frequency=20`` and ``interpolation_grid_frequency=20``.
           Interpolation is disabled since both frequencies are equal.

        2. **``mesh=(init_freq, interp_freq)``**: Creates a Delaunay mesh with the
           specified resolution parameters. Interpolation is enabled only if
           ``init_freq < interp_freq``.

        3. **Pre-built mesh instance**: Returns the provided ``BaseMesh`` object as-is

        Meshes created from resolution parameters (cases 1 and 2) are cached per
        device and dtype to avoid redundant computation when the same configuration
        is reused across different samples.

        :param mesh: Mesh specification. Can be:
            - ``None`` for default parameters,
            - A tuple ``(initial_grid_frequency, interpolation_grid_frequency)``,
            - An existing ``BaseMesh`` instance.
        :param device: Target computation device for the mesh.
        :param dtype: Floating-point precision for mesh computations.
        :return: Initialized mesh object ready for orientation sampling.
        """
        if mesh is None:
            init_freq, interp_freq = 20, 20
            interpolate = False
        elif isinstance(mesh, tuple):
            init_freq, interp_freq = mesh
            interpolate = init_freq < interp_freq
        else:
            return mesh

        return self._get_cached_mesh(
            device=device,
            dtype=dtype,
            initial_grid_frequency=init_freq,
            interpolation_grid_frequency=interp_freq,
            interpolate=interpolate
        )

    def _expand_hamiltonian_strain(self, ham_strain: torch.Tensor, orientation_vector: torch.Tensor) -> torch.Tensor:
        ham_shape = ham_strain.shape[:-1]
        orient_shape = orientation_vector.shape[:-1]
        ham_expanded = ham_strain.view(*ham_shape, *([1] * len(orient_shape)), ham_strain.shape[-1])
        orient_expanded = orientation_vector.view(*([1] * len(ham_shape)), *orient_shape, orientation_vector.shape[-1])
        result = ((ham_expanded ** 2) * (orient_expanded ** 2)).sum(dim=-1).sqrt()
        return result

    def orientation_vector(self, rotation_matrices: torch.Tensor) -> torch.Tensor:
        """
        :param rotation_matrices: The matrix of rotations to rotate of the sample
        :return:
        """
        return rotation_matrices[..., -1, :]

    def _get_effective_rotation_matrices(self) -> torch.Tensor:
        """
        Get the effective rotation matrices combining mesh orientation and spin system frame.

        :return: Rotation matrices of shape [..., orientations, 3, 3].
        """
        rotation_matrices = self.mesh.rotation_matrices
        if self._spin_system_frame is not None:
            rotation_matrices = torch.matmul(rotation_matrices, self._spin_system_rot_matrix)
        return rotation_matrices

    def is_equivalent(self, other: BaseSample, rtol: float = 1e-5, atol: float = 1e-6) -> bool:
        """Check equality of two MultiOrientedSample instances.

        In addition to all BaseSample checks, this also verifies that the
        orientation mesh is identical.

        :param other: Another MultiOrientedSample instance.
        :return: True if all parameters and the mesh match.
        """
        if type(self) is not type(other):
            return False

        if not super().is_equivalent(other, rtol, atol):
            return False

        if not self.mesh.is_equivalent(other.mesh):
            return False

        return True

    def build_ham_strain(self) -> torch.Tensor:
        """Constructs the zero-field strained part of Hamiltonian."""
        return self._ham_strain

    def _get_xyz_basis_triplet(self) -> torch.Tensor:
        """Get transition moment basis vectors Tx, Ty, Tz for a spin-1 system
        in the frame of the molecule.

        Rotates the spin-1 triplet basis ``[Tx, Ty, Tz]`` — obtained from
        :meth:`base_spin_system.get_xyz_basis` — into the molecular frame
        using the orientation mesh rotation matrices.

        :return: torch.Tensor
            Transition basis matrix of shape ``[..., orientations, 3, 3]``,
            with columns ``[Tx, Ty, Tz]`` expressed in the molecular frame.

        Examples
        --------
        For a spin-1 system::

            T = system._get_xyz_basis_triplet()  # shape ``[..., orientations, 3, 3]``
            Tx = T[..., 0]  # shape ``[..., orientations, 3]``
            Ty = T[..., 1]  # shape ``[..., orientations, 3]``
            Tz = T[..., 2]  # shape ``[..., orientations, 3]``
        """
        triplet_basis = self.base_spin_system.get_xyz_basis()
        molecule_rotation_matrices = self._get_effective_rotation_matrices().to(triplet_basis.dtype)
        return torch.matmul(triplet_basis, molecule_rotation_matrices)

    def _get_xyz_basis_two_half_spins(self) -> torch.Tensor:
        """Get basis vectors S, Tx, Ty, Tz for two spin-1/2
        electrons in the frame of the molecule.

        The full basis is obtained from
        :meth:`base_spin_system.get_xyz_basis`, which returns a ``(4, 4)``
        matrix with columns ``[S, Tx, Ty, Tz]`` in the uncoupled
        ``|ms1, ms2⟩`` basis.

        Only the triplet columns ``[Tx, Ty, Tz]`` (indices 0–2) transform
        under rotation — they span the same 3D vector space as the spin-1
        case. The singlet column ``S`` (index 3) is rotationally invariant
        (total spin 0) and is therefore appended to the rotated triplet block
        without modification.

        :return: torch.Tensor
            Transition basis matrix of shape ``[..., orientations, 4, 4]``,
            with columns ``[S, Tx, Ty, Tz]`` where:

            - Column 0: singlet vector, unchanged under rotation
            - Columns 1–3: triplet vectors rotated into the molecular frame.

        Examples
        --------
        For a two spin-1/2 system::

            T = system._get_xyz_basis_two_half_spins()  # shape ``[..., orientations, 4, 4]``
            S  = T[..., 0]  # shape ``[..., orientations, 4]``, invariant under rotation
            Tx = T[..., 1]  # shape ``[..., orientations, 4]``
            Ty = T[..., 2]  # shape ``[..., orientations, 4]``
            Tz = T[..., 3]  # shape ``[..., orientations, 4]``
        """
        full_basis = self.base_spin_system.get_xyz_basis()
        triplet_basis = full_basis[:, 1:]
        singlet_basis = full_basis[:, :1]

        molecule_rotation_matrices = self._get_effective_rotation_matrices().to(triplet_basis.dtype)
        rotated_triplet = torch.matmul(triplet_basis, molecule_rotation_matrices)

        singlet_expanded = singlet_basis.expand(*rotated_triplet.shape[:-1], 1)
        return torch.cat([singlet_expanded, rotated_triplet], dim=-1)

    def get_xyz_basis(self) -> torch.Tensor:
        """Get basis vectors Tx, Ty, Tz (and S) for a spin
        system in the frame of the molecule.

        Dispatches to the appropriate rotation logic depending on the spin
        configuration of the base system:

        - A single spin-1 electron: rotates all three columns ``[Tx, Ty, Tz]``
          as a 3D vector basis (see :meth:`_get_xyz_basis_triplet`).
        - Two spin-1/2 electrons: rotates only the triplet columns
          ``[Tx, Ty, Tz]`` — the singlet column ``S`` is rotationally
          invariant and is appended unchanged (see
          :meth:`_get_xyz_basis_two_half_spins`).

        :return: torch.Tensor
            Transition basis matrix in the molecular frame.

            - For spin-1: shape ``[..., orientations, 3, 3]``.
            - For two spin-1/2: shape ``[..., orientations, 4, 4]``.

        Examples
        --------
        For a spin-1 system::

            T = system.get_xyz_basis()  # shape ``[..., orientations, 3, 3]``
            Tx = T[..., 0]  # shape ``[..., orientations, 3]``
            Ty = T[..., 1]  # shape ``[..., orientations, 3]``
            Tz = T[..., 2]  # shape ``[..., orientations, 3]``

        For a two spin-1/2 system::

            T = system.get_xyz_basis()  # shape ``[..., orientations, 4, 4]``
            S  = T[..., 0]  # shape ``[..., orientations, 4]``
            Tx = T[..., 1]  # shape ``[..., orientations, 4]``
            Ty = T[..., 2]  # shape ``[..., orientations, 4]``
            Tz = T[..., 3]  # shape ``[..., orientations, 4]``
        """
        electrons = self.base_spin_system.electrons
        n = len(electrons)

        if n == 1 and electrons[0].spin == 1.0:
            return self._get_xyz_basis_triplet()

        if n == 2 and electrons[0].spin == 0.5 and electrons[1].spin == 0.5:
            return self._get_xyz_basis_two_half_spins()

    def update(self,
               g_tensors: list[BaseInteraction] = None,
               electron_nuclei: tp.Optional[list[tuple[int, int, BaseInteraction]]] = None,
               electron_electron: tp.Optional[list[tuple[int, int, BaseInteraction]]] = None,
               nuclei_nuclei: tp.Optional[list[tuple[int, int, BaseInteraction]]] = None,
               ham_strain: tp.Optional[torch.Tensor] = None,
               gauss: tp.Union[torch.Tensor, float] = None,
               lorentz: tp.Union[torch.Tensor, float] = None
               ):
        """Update the parameters of a sample.

        No recomputation of spin vectors does not occur
        :param g_tensors:
        list[BaseInteraction]
            g-tensors corresponding to each electron in `electrons`.
            Each element must be an instance of `BaseInteraction` (e.g., `Interaction`).

        :param electron_nuclei:
        list[tuple[int, int, BaseInteraction]], optional
            Hyperfine interactions between electrons and nuclei.
            Each tuple is of the form (electron_index, nucleus_index, interaction_tensor).
            Default is `None`.

        :param electron_electron:
        list[tuple[int, int, BaseInteraction]], optional
            Dipolar or exchange interactions between pairs of electrons.
            Each tuple is of the form (electron_index, electron_index, interaction_tensor).
            Default is `None`.

        :param nuclei_nuclei:
        list[tuple[int, int, BaseInteraction]], optional
            Dipolar or J-coupling interactions between pairs of nuclei.
            Each tuple is of the form (nucleus_index, nucleus_index, interaction_tensor).
            Default is `None`

        :param ham_strain:
        torch.Tensor, optional
            Anisotropic line width, due to the unresolved hyperfine interactions.
            The tensor components, provided in one of the following forms:
              - A scalar (for isotropic interaction).
              - A sequence of two values (axial and z components).
              - A sequence of three values (principal components).
        The values are given as FWHM (full width at half maximum) and measured in Hz

        :param gauss:
        torch.Tensor, optional
            Gaussian broadening parameter(s). Defines inhomogeneous linewidth
            contributions (e.g., due to static disorder). Default is `None`.
            Values are provided as the full width at half maximum (FWHM) and are expressed in:
            - tesla (T) for field-dependent spectra,
            - hertz (Hz) for frequency-dependent spectra.

        :param lorentz:
        torch.Tensor, optional
            Lorentzian broadening parameter(s). Defines homogeneous linewidth
            contributions (e.g., due to relaxation). Default is `None`
            Values are provided as the full width at half maximum (FWHM) and are expressed in:
            - tesla (T) for field-dependent spectra,
            - hertz (Hz) for frequency-dependent spectra.
        """

        rotation_matrices = self.mesh.rotation_matrices
        self.base_spin_system.update(g_tensors, electron_nuclei, electron_electron, nuclei_nuclei)

        if ham_strain is not None:
            self.base_ham_strain = self._init_ham_str(ham_strain, self.device, self.dtype)
            self._ham_strain = self._expand_hamiltonian_strain(
                self.base_ham_strain,
                self.orientation_vector(rotation_matrices)
            )
        self.gauss = self._init_gauss_lorentz(gauss, self.device, self.dtype)
        self.lorentz = self._init_gauss_lorentz(lorentz, self.device, self.dtype)
        if self._spin_system_frame is None:
            self.modified_spin_system = SpinSystemOrientator()(self.base_spin_system, rotation_matrices)
        else:
            self.modified_spin_system = SpinSystemOrientator()(
                self.base_spin_system, torch.matmul(rotation_matrices, self._spin_system_rot_matrix)
        )

    def get_oriented_electron_electron_interaction(self, interaction: torch.Tensor, el_idx_1: int, el_idx_2: int) ->\
            torch.Tensor:
        """
        Compute the electron-electron operator term for all orientations.

        This method rotates the interaction tensor (e.g., dipolar or exchange coupling)
        from interaction tensor frame to the laboratory frame for each orientation in the
        mesh, then contracts it with the corresponding spin operators.

        The resulting Hamiltonian term is: ``H = S_1 · Q · S_2``, where Q is the rotated
        interaction tensor.

        :param interaction: Interaction tensor of shape ``[..., 3, 3]`` in the
                           interaction tensor frame (before orientation rotation).
        :param idx_1: Index of the first electron spin operator.
        :param idx_2: Index of the second electron spin operator.
        :return: operator of shape ``[..., orientations, dim, dim]`` in the
                complex dtype
        """
        rotation_matrices = self._get_effective_rotation_matrices()
        interaction_rotated = utils.apply_expanded_rotations(rotation_matrices, interaction).to(self.complex_dtype)
        S1 = self.base_spin_system.operator_cache[el_idx_1]
        S2 = self.base_spin_system.operator_cache[el_idx_2]
        return scalar_tensor_multiplication(S1, S2, interaction_rotated)

    def get_oriented_electron_nuclei_interaction(self, interaction: torch.Tensor, el_idx: int, nuc_idx: int) ->\
            torch.Tensor:
        """
        Compute the electron-nucleus hyperfine operator term for all orientations.

        This method rotates the hyperfine tensor (A-tensor) from its interaction tensor frame
        to the laboratory frame for each orientation in the mesh, then contracts it
        with the electron and nucleus spin operators.

        The resulting Hamiltonian term is: ``H = S · A · I``, where A is the rotated
        hyperfine tensor.

        :param interaction: Hyperfine tensor of shape ``[..., 3, 3]`` in the
                            interaction tensor frame (before orientation rotation).
        :param el_idx: Index of the electron spin operator.
        :param nuc_idx: Index of the nucleus spin operator.
        :return: operator of shape ``[..., orientations, dim, dim]`` in the
                 complex dtype
        """
        rotation_matrices = self._get_effective_rotation_matrices()
        interaction_rotated = utils.apply_expanded_rotations(rotation_matrices, interaction).to(self.complex_dtype)

        S = self.base_spin_system.operator_cache[el_idx]
        I = self.base_spin_system.operator_cache[len(self.modified_spin_system.electrons) + nuc_idx]
        return scalar_tensor_multiplication(S, I, interaction_rotated)

    def get_oriented_nuclei_nuclei_interaction(
            self, interaction: torch.Tensor, nuc_idx_1: int, nuc_idx_2: int) -> torch.Tensor:
        """
        Compute the nucleus-nucleus interaction operator for all orientations.

        This method rotates the interaction tensor (e.g., nuclear dipolar)
        from its interaction tensor frame to the laboratory frame for each orientation in the
        mesh, then contracts it with the corresponding nucleus spin operators.

        The resulting Hamiltonian term is: ``H = I_1 · Q · I_2``, where Q is the rotated
        interaction tensor.

        :param interaction: Interaction tensor of shape ``[..., 3, 3]`` in the
                            interaction tensor frame (before orientation rotation).
        :param nuc_idx_1: Index of the first nucleus spin operator.
        :param nuc_idx_2: Index of the second nucleus spin operator.
        :return: operator of shape ``[..., orientations, dim, dim]`` in the
                 complex dtype.
        """
        rotation_matrices = self._get_effective_rotation_matrices()
        interaction_rotated = utils.apply_expanded_rotations(rotation_matrices, interaction).to(self.complex_dtype)

        offset = len(self.modified_spin_system.electrons)
        I1 = self.base_spin_system.operator_cache[offset + nuc_idx_1]
        I2 = self.base_spin_system.operator_cache[offset + nuc_idx_2]
        return scalar_tensor_multiplication(I1, I2, interaction_rotated)

    def get_oriented_zeeman_interaction(self, interaction: torch.Tensor, el_idx: int) -> torch.Tensor:
        """
        Compute the electron Zeeman interaction operator for all orientations.

        Compute the electron Zeeman interaction operator for all orientations.
        This method rotates the g-tensor from its interaction tensor frame to the laboratory
        frame for each orientation in the mesh, then contracts it with the electron
        spin operator z-projection

        The external magnetic field is applied separately when constructing the full
        Hamiltonian.

        Hamiltonian term: ``H = (μ_B/h) · S · interaction · B``, where g is the rotated interaction
        and B is the external magnetic field (applied externally).

        :param interaction: interaction of shape ``[..., 3, 3]`` in the interaction tensor frame
        :param el_idx: Index of the electron spin operator in the operator cache.
        :return: Zeeman operator of shape ``[..., orientations, 3, dim, dim]`` in
                 complex dtype. The last dimension corresponds to the magnetic field
                 components (Bx, By, Bz). Multiply by the field vector to get the
                 final Hamiltonian term.

        Note:
            This method returns the operator contracted with the g-tensor but NOT yet
            multiplied by the magnetic field. To get the full Zeeman term:
            ``H_zeeman = self.get_oriented_zeeman_interaction(g_tensor, el_idx) * B_field``

            The result is scaled by μ_B/h to convert to frequency units (Hz) for
            spectral simulation.
        """
        rotation_matrices = self._get_effective_rotation_matrices()
        interaction_rotated = utils.apply_expanded_rotations(rotation_matrices, interaction).to(self.complex_dtype)

        S = self.base_spin_system.operator_cache[el_idx]
        zeeman_term = transform_tensor_components(S, interaction_rotated)
        zeeman_term *= (constants.BOHR / constants.PLANCK)
        return zeeman_term

    def get_librations_along_axis(self, axis: tp.Union[torch.Tensor, list[float]]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the Hamiltonian derivative with respect to small rotational librations
        around a given axis.

        This method calculates how the spin Hamiltonian changes when the molecular
        frame undergoes infinitesimal rotational oscillations (librations) around
        the specified axis. The result is split into field-independent and
        field-dependent contributions.

        The rotation derivative is computed using the commutator formula:
            dQ/dθ = [n]_ Q - Q [n]_
        where [n]_x is the skew-symmetric cross-product matrix of the axis vector.

        :param axis: Tensor of shape ``[..., 3]`` representing the rotation axis
                     vector(s) in the interaction frame. The last dimension
                     corresponds to Cartesian components (x, y, z).
        :return: Tuple of two operators:
                 - ``O_static``: Field-independent operator of shape
                   ``[..., orientations, dim, dim]`` from zero-field terms
                   (ZFS, hyperfine, dipolar). Not multiplied by external field.
                 - ``O_dependent``: Field-dependent operator of shape
                   ``[..., orientations, dim, dim]`` from Zeeman terms.
                 Both operators are in complex dtype.
        The total Hamiltonian derivative for a field B is:
        ``dH/dθ = O_static + O_dependent @ B``

        Note:
            O_static is measured in the Hz
            O_dependent is measured in the Hz / T

            So, for many computations they should be transformed to s^-1 by muttiplication on 2π
        """
        if isinstance(axis, list):
            axis = torch.tensor(axis, device=self.device, dtype=self.dtype)

        axis_norm = torch.norm(axis, dim=-1, keepdim=True)
        axis_norm = torch.where(axis_norm > 0, axis_norm, torch.ones_like(axis_norm))
        n = axis / axis_norm

        nx, ny, nz = n[..., 0], n[..., 1], n[..., 2]
        zeros = torch.zeros_like(nx)
        Omega = torch.stack([
            torch.stack([zeros, -nz, ny], dim=-1),
            torch.stack([nz, zeros, -nx], dim=-1),
            torch.stack([-ny, nx, zeros], dim=-1)
        ], dim=-2)

        config_shape = self.config_shape
        dim = self.spin_system_dim
        device = self.device
        complex_dtype = self.complex_dtype

        O_static = torch.zeros((*config_shape, dim, dim), dtype=complex_dtype, device=device)

        for el_idx_1, el_idx_2, interaction in self.base_spin_system.electron_electron:
            Q = interaction.tensor
            dQ_dtheta = torch.matmul(Omega, Q) - torch.matmul(Q, Omega)
            O_static += self.get_oriented_electron_electron_interaction(
                dQ_dtheta, el_idx_1=el_idx_1, el_idx_2=el_idx_2
            )

        for el_idx, nuc_idx, interaction in self.base_spin_system.electron_nuclei:
            Q = interaction.tensor
            dQ_dtheta = torch.matmul(Omega, Q) - torch.matmul(Q, Omega)
            O_static += self.get_oriented_electron_nuclei_interaction(dQ_dtheta, el_idx=el_idx, nuc_idx=nuc_idx)

        for nuc_idx_1, nuc_idx_2, interaction in self.base_spin_system.nuclei_nuclei:
            Q = interaction.tensor
            dQ_dtheta = torch.matmul(Omega, Q) - torch.matmul(Q, Omega)
            O_static += self.get_oriented_nuclei_nuclei_interaction(
                dQ_dtheta, nuc_idx_1=nuc_idx_1, nuc_idx_2=nuc_idx_2)

        O_dependent = torch.zeros((*config_shape, 3, dim, dim), dtype=complex_dtype, device=device)
        for el_idx, g_interaction in enumerate(self.base_spin_system.g_tensors):
            Q = g_interaction.tensor
            dQ_dtheta = torch.matmul(Omega, Q) - torch.matmul(Q, Omega)
            O_dependent += self.get_oriented_zeeman_interaction(
                dQ_dtheta, el_idx)

        return O_static, O_dependent[..., -1, :, :]


class MultiOrientedSampleExpandedStrain(MultiOrientedSample):
    """
    Extension of MultiOrientedSample that allows to define additional batche for hamiltonian strain.

    Compare to spin_system.MultiOrientedSample it allows to set the hamiltonian strain as an
    additional batch for the system. That is, ham_strain.shape can be any
    """
    def _init_ham_str(
            self, ham_strain: tp.Optional[tp.Union[torch.Tensor, float]],
            device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if ham_strain is None:
            ham_strain = torch.zeros(
                (*self.base_spin_system.config_shape, 3), device=device, dtype=dtype
            )
        else:
            ham_strain = init_tensor(ham_strain, device=device, dtype=dtype)
        return ham_strain


