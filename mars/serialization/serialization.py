from __future__ import annotations

import pathlib
from dataclasses import dataclass, field, fields

import typing as tp
import warnings
import json
import math

import torch

import safetensors.torch


from .. import mesher
from .. import particles
from .. import utils
from ..mesher import BaseMesh
from ..spin_model import BaseSample, MultiOrientedSample, MultiOrientedSampleExpandedStrain,\
    SpinSystem, Interaction, DEInteraction
from ..spectra_manager import StationarySpectra, BaseResSpectra


if tp.TYPE_CHECKING:
    from .graph_representation import GraphSample, GraphSpinSystem


def _save_to_disk(
        metadata: tp.Dict[str, tp.Any],
        tensor_dict: tp.Dict[str, torch.Tensor],
        base_path: tp.Union[str, pathlib.Path]) -> None:
    """Safely save metadata to JSON and tensors to Safetensors."""
    path = pathlib.Path(base_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    json_path = path.with_suffix(".json")
    safetensors_path = path.with_suffix(".safetensors")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if tensor_dict:
        safetensors.torch.save_file(tensor_dict, str(safetensors_path))


def _load_from_disk(
        base_path: tp.Union[str, pathlib.Path],
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32) -> tp.Tuple[tp.Dict[str, tp.Any], tp.Dict[str, torch.Tensor]]:
    """Safely load metadata from JSON and tensors from Safetensors."""
    path = pathlib.Path(base_path)
    json_path = path.with_suffix(".json")
    safetensors_path = path.with_suffix(".safetensors")

    with open(json_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    tensor_dict = {}
    if safetensors_path.exists():
        tensor_dict = safetensors.torch.load_file(str(safetensors_path), device=str(device))
        tensor_dict = {
            k: v.to(device=device, dtype=dtype)
            for k, v in tensor_dict.items()
        }
    return metadata, tensor_dict


@dataclass
class SpinSystemMetaData:
    """Structural information about the spin system (particles and interaction pairs).

    This class stores particle properties and the connectivity of interactions
    (which electron/nucleus indices are coupled). It does **not** store any
    tensor values; those are held in :class:`SpinSystemInteractions` and
    :class:`SpinSystemStrains`.

    :param electron_spins: List of electron spin quantum numbers (e.g., [0.5, 0.5]).
    :param nucleus_labels: List of nucleus isotope strings (e.g., ["14N", "1H"]).
    :param nuclear_spins: List of nuclear spin quantum numbers (e.g., [1.0, 0.5]).
        If not provided, it will be automatically inferred from `nucleus_labels`.
    :param electron_nucleus_pairs: List of (electron_index, nucleus_index) tuples
        for hyperfine interactions.
    :param electron_electron_pairs: List of (el_idx1, el_idx2) tuples for all
        electron‑electron interactions (both ZFS and dipolar/exchange).
    :param zfs_pairs: Subset of `electron_electron_pairs` that represent zero‑field
        splitting (ZFS) interactions. May overlap with exchange_dipolar_pairs.
    :param exchange_dipolar_pairs: Subset of `electron_electron_pairs` representing
        exchange/dipolar couplings.
    :param nucleus_nucleus_pairs: List of (nuc_idx1, nuc_idx2) tuples for
        nucleus‑nucleus interactions.
    """
    electron_spins: tp.List[float]
    nucleus_labels: tp.List[str] = field(default_factory=list)
    nuclear_spins: tp.List[float] = field(default_factory=list)

    electron_nucleus_pairs: tp.List[tp.Tuple[int, int]] = field(default_factory=list)
    electron_electron_pairs: tp.List[tp.Tuple[int, int]] = field(default_factory=list)
    zfs_pairs: tp.List[tp.Tuple[int, int]] = field(default_factory=list)
    exchange_dipolar_pairs: tp.List[tp.Tuple[int, int]] = field(default_factory=list)
    nucleus_nucleus_pairs: tp.List[tp.Tuple[int, int]] = field(default_factory=list)

    def _hilbert_dim_from_spins(self, spins: tp.List[float]) -> int:
        """Calculate the total Hilbert space dimension for a given list of spins.

        :param spins: List of spin quantum numbers (e.g., [0.5, 0.5, 1.0]).
        :return: The total Hilbert space dimension as an integer.
        """
        return math.prod(int(2 * s + 1) for s in spins)

    def _init_electron_pairs(self):
        if self.electron_electron_pairs and not self.zfs_pairs and not self.exchange_dipolar_pairs:
            self.zfs_pairs = [p for p in self.electron_electron_pairs if p[0] == p[1]]
            self.exchange_dipolar_pairs = [p for p in self.electron_electron_pairs if p[0] != p[1]]
        elif not self.electron_electron_pairs and (self.zfs_pairs or self.exchange_dipolar_pairs):
            self.electron_electron_pairs = self.zfs_pairs + self.exchange_dipolar_pairs

    def _init_nuclear_spins(self):
        """Infer or validate nuclear spins based on nucleus labels."""
        if self.nucleus_labels:
            if not self.nuclear_spins:
                self.nuclear_spins = [particles.Nucleus.get_spin(label) for label in self.nucleus_labels]
            elif len(self.nuclear_spins) != len(self.nucleus_labels):
                raise ValueError(
                    f"Length mismatch: nuclear_spins ({len(self.nuclear_spins)}) must match "
                    f"the length of nucleus_labels ({len(self.nucleus_labels)})."
                )

    def __post_init__(self):
        self._init_electron_pairs()
        self._init_nuclear_spins()

        self.num_electrons = len(self.electron_spins)
        self.num_nuclei = len(self.nucleus_labels)

        self.spin_system_dim = self._hilbert_dim_from_spins(self.electron_spins + self.nuclear_spins)

    def to_json_dict(self) -> tp.Dict[str, tp.Any]:
        """Convert metadata to a JSON‑serializable dictionary.

        :return: Dictionary containing all pair lists and particle properties.
        """
        return {
            "electron_spins": self.electron_spins,
            "nuclear_spins": self.nuclear_spins,
            "nucleus_labels": self.nucleus_labels,
            "electron_nucleus_pairs": self.electron_nucleus_pairs,
            "electron_electron_pairs": self.electron_electron_pairs,
            "zfs_pairs": self.zfs_pairs,
            "exchange_dipolar_pairs": self.exchange_dipolar_pairs,
            "nucleus_nucleus_pairs": self.nucleus_nucleus_pairs,
        }

    @classmethod
    def from_json_dict(cls, data: tp.Dict[str, tp.Any]) -> SpinSystemMetaData:
        """Reconstruct metadata from a dictionary.

        :param data: Dictionary as produced by :meth:`to_json_dict`.
        :return: A new :class:`SpinSystemMetaData` instance.
        """
        return cls(
            electron_spins=data["electron_spins"],
            nucleus_labels=data.get("nucleus_labels", []),
            nuclear_spins=data.get("nuclear_spins", []),
            electron_nucleus_pairs=data.get("electron_nucleus_pairs", []),
            electron_electron_pairs=data.get("electron_electron_pairs", []),
            zfs_pairs=data.get("zfs_pairs", []),
            exchange_dipolar_pairs=data.get("exchange_dipolar_pairs", []),
            nucleus_nucleus_pairs=data.get("nucleus_nucleus_pairs", []),
        )

    def is_equivalent(self, other: SpinSystemMetaData, rtol: float = 1e-5, atol: float = 1e-6) -> bool:
        if not isinstance(other, SpinSystemMetaData):
            return False
        return all([
            self.electron_spins == other.electron_spins,
            self.nucleus_labels == other.nucleus_labels,
            self.nuclear_spins == other.nuclear_spins,
            self.electron_nucleus_pairs == other.electron_nucleus_pairs,
            self.electron_electron_pairs == other.electron_electron_pairs,
            self.zfs_pairs == other.zfs_pairs,
            self.exchange_dipolar_pairs == other.exchange_dipolar_pairs,
            self.nucleus_nucleus_pairs == other.nucleus_nucleus_pairs
        ])


@dataclass
class SpinSystemInteractions:
    """Flat storage for all interaction tensors of a spin system.

     Each field holds the stacked components and (where applicable) orientation
     frames for one type of interaction.  Fields are ``None`` if the spin system
     contains no interaction of that type.

     :param g_tensor_components: Stacked g‑tensor principal values, shape ``(n_el, 3)``.
     :param g_tensor_orientations: Orientation frames for the g‑tensors, shape ``(n_el, 3, 3)``.
     :param hyperfine_coupling_components: Stacked HFC principal values, shape ``(n_hfc, 3)``.
     :param hyperfine_coupling_orientations: Orientation frames for HFC, shape ``(n_hfc, 3, 3)``.
     :param zfs_components: ZFS principal values (Dx, Dy, Dz), shape ``(n_zfs, 3)``.
     :param dipolar_components: Dipolar/exchange principal values, shape ``(n_dip, 3)``.
     :param electron_electron_orientations: Concatenated frames for all e‑e interactions
         (ZFS first, then dipolar), shape ``(n_ee, 3, 3)``.
     :param nuclear_coupling_components: Nuclear coupling principal values, shape ``(n_nn, 3)``.
     :param nuclear_coupling_orientations: Orientation frames for nuclear couplings, shape ``(n_nn, 3, 3)``.
     """
    g_tensor_components: torch.Tensor
    g_tensor_orientations: torch.Tensor
    hyperfine_coupling_components: tp.Optional[torch.Tensor] = field(default=None)
    hyperfine_coupling_orientations: tp.Optional[torch.Tensor] = field(default=None)
    
    zfs_components: tp.Optional[torch.Tensor] = field(default=None)
    dipolar_components: tp.Optional[torch.Tensor] = field(default=None)

    electron_electron_orientations: tp.Optional[torch.Tensor] = field(default=None)  # zfs first
    nuclear_coupling_components: tp.Optional[torch.Tensor] = field(default=None)
    nuclear_coupling_orientations: tp.Optional[torch.Tensor] = field(default=None)

    def to_tensor_dict(self) -> tp.Dict[str, torch.Tensor]:
        """Flatten interaction tensors into a dictionary.

        Extracts all non-None tensor fields
        and maps them to their respective attribute names.

        :return: A dictionary mapping string names to torch.Tensor objects.
        """
        tensor_dict = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if value is not None:
                tensor_dict[field.name] = value
        return tensor_dict

    @classmethod
    def from_tensor_dict(cls, tensor_dict: tp.Dict[str, torch.Tensor]) -> SpinSystemInteractions:
        """Reconstruct interactions from a flat tensor dictionary.

        Reads the tensor dictionary and populates
        the interaction fields, ignoring any keys that do not match the
        dataclass fields.
        :param tensor_dict: Dictionary of tensors.
        :return: A new SpinSystemInteractions instance.
        """
        kwargs = {}
        for field in fields(cls):
            if field.name in tensor_dict:
                kwargs[field.name] = tensor_dict[field.name]
        return cls(**kwargs)

    @classmethod
    def from_mars_spin_system(cls, spin_system: SpinSystem, metadata: SpinSystemMetaData) -> SpinSystemInteractions:
        """Convert all interactions of a ``SpinSystem`` into a flat structure.

        :param spin_system: The source spin system.
        :param metadata: The extracted metadata for pairing reference.
        :return: A :class:`SpinSystemInteractions` object holding stacked tensors
                 for each interaction category.
        """
        g_comp_list = [g.components for g in spin_system.g_tensors]
        g_frame_list = [g.frame for g in spin_system.g_tensors]
        g_tensor_components = torch.stack(g_comp_list, dim=-2)
        g_tensor_orientations = torch.stack(g_frame_list, dim=-2)

        hfc_comp = [inter.components for inter in spin_system.electron_nuclei_interactions]
        hfc_frame = [inter.frame for inter in spin_system.electron_nuclei_interactions]

        if hfc_comp:
            hyperfine_coupling_components = torch.stack(hfc_comp, dim=-2)
            hyperfine_coupling_orientations = torch.stack(hfc_frame, dim=-2)
        else:
            hyperfine_coupling_components = None
            hyperfine_coupling_orientations = None

        zfs_pairs = metadata.zfs_pairs or []
        ed_pairs = metadata.exchange_dipolar_pairs or []

        ee_interactions = list(spin_system.electron_electron)
        ee_dict = {(idx1, idx2): inter for (idx1, idx2, inter) in ee_interactions}

        zfs_comp = [ee_dict[pair].components for pair in zfs_pairs]
        zfs_frame = [ee_dict[pair].frame for pair in zfs_pairs]
        dip_comp = [ee_dict[pair].components for pair in ed_pairs]
        dip_frame = [ee_dict[pair].frame for pair in ed_pairs]

        zfs_components = torch.stack(zfs_comp, dim=-2) if zfs_comp else None
        dipolar_components = torch.stack(dip_comp, dim=-2) if dip_comp else None

        all_ee_frames = []
        if zfs_frame:
            all_ee_frames.append(torch.stack(zfs_frame, dim=-2))
        if dip_frame:
            all_ee_frames.append(torch.stack(dip_frame, dim=-2))

        electron_electron_orientations = torch.cat(all_ee_frames, dim=-2) if all_ee_frames else None

        nn_comp = [inter.components for inter in spin_system.nuclei_nuclei_interactions]
        nn_frame = [inter.frame for inter in spin_system.nuclei_nuclei_interactions]

        if nn_comp:
            nuclear_coupling_components = torch.stack(nn_comp, dim=-2)
            nuclear_coupling_orientations = torch.stack(nn_frame, dim=-2)
        else:
            nuclear_coupling_components = None
            nuclear_coupling_orientations = None

        return cls(
            g_tensor_components=g_tensor_components,
            g_tensor_orientations=g_tensor_orientations,
            hyperfine_coupling_components=hyperfine_coupling_components,
            hyperfine_coupling_orientations=hyperfine_coupling_orientations,
            zfs_components=zfs_components,
            dipolar_components=dipolar_components,
            electron_electron_orientations=electron_electron_orientations,
            nuclear_coupling_components=nuclear_coupling_components,
            nuclear_coupling_orientations=nuclear_coupling_orientations,
        )

    def _are_optional_tensors_close(self,
                                    first_tensor: tp.Optional[torch.Tensor],
                                    second_tensor: tp.Optional[torch.Tensor],
                                    rtol: float = 1e-5, atol: float = 1e-6) -> bool:
        """
        Compare two optional tensors for numerical equality, handling ``None`` values gracefully.

        This method safely checks if two tensors are equivalent. It first verifies that both
        tensors share the same ``None`` state. If both are tensors, it evaluates their numerical
        closeness

        :param first_tensor: The first tensor to compare. Can be ``None``.
        :param second_tensor: The second tensor to compare. Can be ``None``.
        :return: ``True`` if both tensors are ``None``, or if both are tensors and their values
                 are close
                 Returns ``False`` if one is ``None`` and the other is not, or if the tensors
                 differ.
        """
        if (first_tensor is None) != (second_tensor is None):
            return False

        if first_tensor is not None:
            if not torch.allclose(first_tensor, second_tensor, rtol=rtol, atol=atol):
                return False
        return True

    def is_equivalent(self, other: SpinSystemInteractions, rtol: float = 1e-5, atol: float = 1e-6) -> bool:
        if not isinstance(other, SpinSystemInteractions):
            return False
        return all([
                utils.are_optional_tensors_close(
                    self.g_tensor_components, other.g_tensor_components, rtol, atol),
                utils.are_optional_tensors_close(
                    self.g_tensor_orientations, other.g_tensor_orientations, rtol, atol),
                utils.are_optional_tensors_close(
                    self.hyperfine_coupling_components, other.hyperfine_coupling_components, rtol, atol),
                utils.are_optional_tensors_close(
                    self.hyperfine_coupling_orientations, other.hyperfine_coupling_orientations, rtol, atol),
                utils.are_optional_tensors_close(
                    self.zfs_components, other.zfs_components),
                utils.are_optional_tensors_close(
                    self.dipolar_components, other.dipolar_components, rtol, atol),
                utils.are_optional_tensors_close(
                    self.electron_electron_orientations, other.electron_electron_orientations, rtol, atol),
                utils.are_optional_tensors_close(
                    self.nuclear_coupling_components, other.nuclear_coupling_components, rtol, atol),
                utils.are_optional_tensors_close(
                    self.nuclear_coupling_orientations, other.nuclear_coupling_orientations, rtol, atol)
        ])


@dataclass
class SpinSystemStrains:
    """Container for strain tensors of a spin system.

    Each field may hold a stacked tensor of strain parameters or ``None`` if
    that interaction type is strain‑free.  Mixed (some strained, some not) cases
    are not supported.

    :param g_tensor_strain: Strain for g‑tensors, shape ``(n_el, …)`` or ``None``.
    :param hyperfine_coupling_strain: Strain for HFC, shape ``(n_hfc, …)`` or ``None``.
    :param zfs_strain: Strain for ZFS interactions, shape ``(n_zfs, …)`` or ``None``.
    :param dipolar_strain: Strain for dipolar/exchange interactions, shape ``(n_dip, …)`` or ``None``.
    """
    g_tensor_strain: tp.Optional[torch.Tensor] = None
    hyperfine_coupling_strain: tp.Optional[torch.Tensor] = None
    zfs_strain: tp.Optional[torch.Tensor] = None
    dipolar_strain: tp.Optional[torch.Tensor] = None

    def to_tensor_dict(self) -> tp.Dict[str, torch.Tensor]:
        """Flatten strain tensors into a dictionary.

        Extracts all non-None strain tensors
        and maps them to their respective attribute names.

        :return: A dictionary mapping string names to torch.Tensor objects.
        """
        tensor_dict = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if value is not None:
                tensor_dict[field.name] = value
        return tensor_dict

    @classmethod
    def from_tensor_dict(cls, tensor_dict: tp.Dict[str, torch.Tensor]) -> tp.Optional[SpinSystemStrains]:
        """Reconstruct strains from a flat tensor dictionary.

        Extracts strain tensors from the dictionary.
        If no strain information is present, it returns None

        :param tensor_dict: Dictionary of tensors.
        :return: A new SpinSystemStrains instance, or None if no strains are found.
        """
        kwargs = {}
        has_strain = False
        for field in fields(cls):
            if field.name in tensor_dict:
                value = tensor_dict[field.name]
                kwargs[field.name] = value
                if value is not None:
                    has_strain = True
        if not has_strain:
            return None
        return cls(**kwargs)

    def is_equivalent(self, other: SpinSystemStrains, rtol: float = 1e-5, atol: float = 1e-6) -> bool:
        if not isinstance(other, SpinSystemStrains):
            return False

        return all([
                utils.are_optional_tensors_close(self.g_tensor_strain, other.g_tensor_strain, rtol, atol),
                utils.are_optional_tensors_close(
                    self.hyperfine_coupling_strain, other.hyperfine_coupling_strain, rtol, atol),
                utils.are_optional_tensors_close(self.zfs_strain, other.zfs_strain, rtol, atol),
                utils.are_optional_tensors_close(self.dipolar_strain, other.dipolar_strain, rtol, atol)
        ])


@dataclass
class SerializedSpinSystem:
    """Complete serializable representation of a :class:`SpinSystem`.

    Stores the structural metadata, interaction tensors, and optional strain
    information.  Use :meth:`from_mars_spin_system` to create from a live system
    and :meth:`to_mars_spin_system` to reconstruct it.

    :param metadata: Structural layout (particles and pair indices).
    :param interactions: Stacked interaction tensors.
    :param strain: Strain tensors or ``None``.
    """

    metadata: SpinSystemMetaData
    interactions: SpinSystemInteractions
    strain: tp.Union[SpinSystemStrains, None]

    @classmethod
    def _serialize_electrons(cls, spin_system: SpinSystem) -> tp.List[float]:
        """Extract electron spin quantum numbers from a spin system.

        :param spin_system: Spin system to serialize.
        :return: List of spin values (e.g., 0.5, 1.0).
        """
        return [float(electron.spin) for electron in spin_system.electrons]

    @classmethod
    def _serialize_nuclei(cls, spin_system: SpinSystem) -> tp.List[str]:
        """Extract nucleus label strings from a spin system.

        :param spin_system: Spin system to serialize.
        :return: List of nucleus isotope strings.
        """
        return [nucleus.nucleus_str for nucleus in spin_system.nuclei]

    @classmethod
    def _get_spin_system_meta(cls, spin_system: SpinSystem) -> SpinSystemMetaData:
        """Create :class:`SpinSystemMetaData` from a live system.

        :param spin_system: Spin system to describe.
        :return: Populated metadata object.
        """
        electron_spins = cls._serialize_electrons(spin_system)
        nucleus_labels = cls._serialize_nuclei(spin_system)

        electron_nucleus_pairs = spin_system.en_indices
        electron_nucleus_pairs = electron_nucleus_pairs if electron_nucleus_pairs else None

        electron_electron_pairs = spin_system.ee_indices
        electron_electron_pairs = electron_electron_pairs if electron_electron_pairs else None

        nucleus_nucleus_pairs = spin_system.nn_indices
        nucleus_nucleus_pairs = nucleus_nucleus_pairs if nucleus_nucleus_pairs else None

        return SpinSystemMetaData(
            electron_spins=electron_spins,
            nucleus_labels=nucleus_labels,
            electron_nucleus_pairs=electron_nucleus_pairs,
            electron_electron_pairs=electron_electron_pairs,
            nucleus_nucleus_pairs=nucleus_nucleus_pairs
            )

    @classmethod
    def _serialize_strains(cls, spin_system: SpinSystem, metadata: SpinSystemMetaData) ->\
            tp.Optional[SpinSystemStrains]:
        """Extract strain tensors from all interactions of the spin system.

        :param spin_system: The spin system whose strains are to be serialised.
        :param metadata: Structural metadata.
        :return: A :class:`SpinSystemStrains` object if any interaction type has
                 strain, otherwise ``None``. Fields for interaction types without
                 strain are set to ``None``.
        :raises ValueError: If some (but not all) interactions of a given type
                            carry strain information.
        """
        def _stack_strains(inter_list: tp.List[Interaction]) -> tp.Optional[torch.Tensor]:
            strains = [inter.strain for inter in inter_list]
            if all(s is None for s in strains):
                return None
            if any(s is None for s in strains):
                raise ValueError(
                    "All interactions of the same type must either have strain or none. "
                    "Mixed cases are not supported."
                )
            return torch.stack(strains, dim=-2)

        strain_data = {}
        strain_data["g_tensor_strain"] = _stack_strains(spin_system.g_tensors)
        strain_data["hyperfine_coupling_strain"] = _stack_strains(
            list(spin_system.electron_nuclei_interactions)
        )
        zfs_pairs = metadata.zfs_pairs if metadata.zfs_pairs else []
        ee_dict = {(idx1, idx2): inter for (idx1, idx2, inter) in spin_system.electron_electron}
        zfs_inters = [ee_dict[pair] for pair in zfs_pairs]
        strain_data["zfs_strain"] = _stack_strains(zfs_inters)
        ed_pairs = metadata.exchange_dipolar_pairs if metadata.exchange_dipolar_pairs else []
        dip_inters = [ee_dict[pair] for pair in ed_pairs]
        strain_data["dipolar_strain"] = _stack_strains(dip_inters)

        if all(v is None for v in strain_data.values()):
            return None
        return SpinSystemStrains(**strain_data)

    @classmethod
    def from_mars_spin_system(cls, spin_system: SpinSystem) -> SerializedSpinSystem:
        """Populate the serialised representation from a live ``SpinSystem`` instance.

        :param spin_system: The spin system to be serialised.
        :return: A new :class:`SerializedSpinSystem` instance.
        """
        metadata = cls._get_spin_system_meta(spin_system)
        interactions = SpinSystemInteractions.from_mars_spin_system(spin_system, metadata)
        strain = cls._serialize_strains(spin_system, metadata)
        return cls(metadata, interactions, strain)

    def to_mars_spin_system(self) -> SpinSystem:
        """Reconstruct a ``SpinSystem`` from the serialised data.
        :return: A new ``SpinSystem`` instance with the stored particles and interactions.
        """
        sample_tensor = self.interactions.g_tensor_components
        device = sample_tensor.device
        dtype = sample_tensor.dtype

        electrons, nuclei = self._reconstruct_particles(device, dtype)
        g_tensors = self._reconstruct_g_tensors(electrons, device, dtype)
        en_interactions = self._reconstruct_hyperfine_interactions(device, dtype)
        ee_interactions = self._reconstruct_electron_electron_interactions(device, dtype)
        nn_interactions = self._reconstruct_nucleus_nucleus_interactions(device, dtype)

        return SpinSystem(
            electrons=electrons,
            g_tensors=g_tensors,
            nuclei=nuclei if nuclei else None,
            electron_nuclei=en_interactions if en_interactions else None,
            electron_electron=ee_interactions if ee_interactions else None,
            nuclei_nuclei=nn_interactions if nn_interactions else None,
            device=device,
            dtype=dtype
        )

    @classmethod
    def from_graph_sample(cls, graph_spin_system: GraphSpinSystem) -> SerializedSpinSystem:
        return graph_spin_system.to_serialized_spin_system()

    def to_graph_sample(self) -> GraphSpinSystem:
        from .graph_representation import GraphSpinSystem
        return GraphSpinSystem.from_serialized_spin_system(self)

    def _reconstruct_particles(self, device: torch.device, dtype: torch.dtype) ->\
            tp.Tuple[tp.List[particles.Electron], tp.List[particles.Nucleus]]:
        """Reconstruct electron and nucleus particles from metadata.

        :param device: The device to place the particles on.
        :param dtype: The data type for the particles.
        :return: A tuple containing the list of electrons and the list of nuclei.
        """
        complex_dtype = utils.float_to_complex_dtype(dtype)
        electrons = [
            particles.Electron(spin, device=device, complex_dtype=complex_dtype)
            for spin in self.metadata.electron_spins
        ]
        nuclei = []
        if self.metadata.nucleus_labels:
            nuclei = [
                particles.Nucleus(label, device=device, complex_dtype=complex_dtype)
                for label in self.metadata.nucleus_labels
            ]
        return electrons, nuclei

    def _reconstruct_g_tensors(self, electrons: tp.List[particles.Electron],
                               device: torch.device, dtype: torch.dtype) -> tp.List[Interaction]:
        """Reconstruct g-tensor interactions for electrons.

        :param electrons: The list of electron particles.
        :param device: The device to place the tensors on.
        :param dtype: The data type for the tensors.
        :return: A list of ``Interaction`` objects representing the g-tensors.
        """
        g_list = []
        g_strain = self.strain.g_tensor_strain if self.strain else None
        for i in range(len(electrons)):
            g_list.append(Interaction(
                components=self.interactions.g_tensor_components[..., i, :],
                frame=self.interactions.g_tensor_orientations[..., i, :],
                strain=self._get_strain(i, g_strain),
                device=device, dtype=dtype
            ))
        return g_list

    def _reconstruct_hyperfine_interactions(self, device: torch.device, dtype: torch.dtype) ->\
            tp.List[tp.Tuple[int, int, Interaction]]:
        """Reconstruct electron-nucleus hyperfine interactions.

        :param device: The device to place the tensors on.
        :param dtype: The data type for the tensors.
        :return: A list of tuples containing electron index, nucleus index, and
                 the ``Interaction`` object.
        """
        en_interactions = []
        en_strain = self.strain.hyperfine_coupling_strain if self.strain else None
        pairs = self.metadata.electron_nucleus_pairs or []
        for i, (e_idx, n_idx) in enumerate(pairs):
            en_interactions.append((
                e_idx, n_idx,
                Interaction(
                    components=self.interactions.hyperfine_coupling_components[..., i, :],
                    frame=self.interactions.hyperfine_coupling_orientations[..., i, :],
                    strain=self._get_strain(i, en_strain),
                    device=device, dtype=dtype
                )
            ))
        return en_interactions

    def _reconstruct_electron_electron_interactions(self, device: torch.device, dtype: torch.dtype) ->\
            tp.List[tp.Tuple[int, int, tp.Union[Interaction, DEInteraction]]]:
        """Reconstruct electron-electron interactions (ZFS and dipolar/exchange).

        :param device: The device to place the tensors on.
        :param dtype: The data type for the tensors.
        :return: A list of tuples containing electron indices and the corresponding
                 ``Interaction`` or ``DEInteraction`` object.
        """
        ee_interactions = []
        zfs_strain_tensor = self.strain.zfs_strain if self.strain else None
        dip_strain_tensor = self.strain.dipolar_strain if self.strain else None

        zfs_pairs = self.metadata.zfs_pairs or []
        n_zfs = len(zfs_pairs)
        for i, (e1, e2) in enumerate(zfs_pairs):
            Dx, Dy, Dz = self.interactions.zfs_components[..., i, :].unbind(-1)
            D = 1.5 * Dz
            E = (Dx - Dy) / 2.0
            D_E_tensor = torch.stack([D, E], dim=-1)

            ee_interactions.append((
                e1, e2,
                DEInteraction(
                    components=D_E_tensor,
                    frame=self.interactions.electron_electron_orientations[..., i, :],
                    strain=self._get_strain(i, zfs_strain_tensor),
                    device=device, dtype=dtype
                )
            ))

        ed_pairs = self.metadata.exchange_dipolar_pairs or []
        for j, (e1, e2) in enumerate(ed_pairs):
            i = n_zfs + j
            ee_interactions.append((
                e1, e2,
                Interaction(
                    components=self.interactions.dipolar_components[..., j, :],
                    frame=self.interactions.electron_electron_orientations[..., i, :],
                    strain=self._get_strain(j, dip_strain_tensor),
                    device=device, dtype=dtype
                )
            ))
        return ee_interactions

    def _reconstruct_nucleus_nucleus_interactions(self, device: torch.device, dtype: torch.dtype) ->\
            tp.List[tp.Tuple[int, int, Interaction]]:
        """Reconstruct nucleus-nucleus interactions.

        :param device: The device to place the tensors on.
        :param dtype: The data type for the tensors.
        :return: A list of tuples containing nucleus indices and the ``Interaction`` object.
        """
        nn_interactions = []
        pairs = self.metadata.nucleus_nucleus_pairs or []
        for i, (n1, n2) in enumerate(pairs):
            nn_interactions.append((
                n1, n2,
                Interaction(
                    components=self.interactions.nuclear_coupling_components[..., i, :],
                    frame=self.interactions.nuclear_coupling_orientations[..., i, :],
                    strain=None,
                    device=device, dtype=dtype
                )
            ))
        return nn_interactions

    @staticmethod
    def _get_strain(idx: int, strain_tensor: tp.Optional[torch.Tensor]) -> tp.Optional[torch.Tensor]:
        """Retrieve the strain tensor for a specific interaction index.

        :param idx: The index of the interaction.
        :param strain_tensor: The stacked strain tensor for the interaction type.
        :return: The strain tensor for the given index, or ``None`` if not applicable.
        """
        if strain_tensor is None:
            return None
        return strain_tensor[..., idx, :]

    def to_tensor_dict(self) -> tp.Dict[str, torch.Tensor]:
        """Flatten all interaction and strain tensors into a single dictionary.

        Delegates the tensor extraction to the
        underlying interactions and strains dataclasses, merging their
        outputs into a single flat dictionary. This centralizes the
        serialization logic and prevents code duplication.

        :return: A dictionary mapping string names to torch.Tensor objects.
        """
        tensor_dict = self.interactions.to_tensor_dict()
        if self.strain is not None:
            tensor_dict.update(self.strain.to_tensor_dict())
        return tensor_dict

    @classmethod
    def from_tensor_dict(cls, tensor_dict: tp.Dict[str, torch.Tensor],
                         metadata: SpinSystemMetaData) -> SerializedSpinSystem:
        """Reconstruct a serialised spin system from a flat tensor dictionary.

        Delegates the reconstruction of interactions
        and strains to their respective classes, using the provided metadata
        to assemble the final SerializedSpinSystem.

        :param tensor_dict: Dictionary of tensors produced by to_tensor_dict.
        :param metadata: The metadata required to reconstruct the spin system structure.
        :return: A new SerializedSpinSystem instance.
        """
        interactions = SpinSystemInteractions.from_tensor_dict(tensor_dict)
        strain = SpinSystemStrains.from_tensor_dict(tensor_dict)

        return cls(metadata=metadata, interactions=interactions, strain=strain)

    def to_file(self, base_path: tp.Union[str, pathlib.Path]) -> None:
        """Save the serialised spin system to disk."""
        _save_to_disk(
            metadata=self.metadata.to_json_dict(),
            tensor_dict=self.to_tensor_dict(),
            base_path=base_path
        )

    @classmethod
    def from_file(cls, base_path: tp.Union[str, pathlib.Path],
                  device: torch.device = torch.device("cpu"),
                  dtype: torch.dtype = torch.float32
                  ) -> SerializedSpinSystem:
        """Load a serialised spin system from disk."""
        meta_dict, tensor_dict = _load_from_disk(base_path, device=device, dtype=dtype)
        metadata = SpinSystemMetaData.from_json_dict(meta_dict)
        return cls.from_tensor_dict(tensor_dict, metadata)

    def is_equivalent(self, other: SerializedSpinSystem, rtol: float = 1e-5, atol: float = 1e-6) -> bool:
        if not isinstance(other, SerializedSpinSystem):
            return False
        if not self.metadata.is_equivalent(other.metadata, rtol, atol):
            return False
        if not self.interactions.is_equivalent(other.interactions, rtol, atol):
            return False

        if self.strain is None and other.strain is None:
            return True
        if self.strain is None or other.strain is None:
            return False
        return self.strain.is_equivalent(other.strain, rtol, atol)


@dataclass
class SerializedSampleWidth:
    ham_strain: tp.Optional[torch.Tensor] = field(default=None)
    gauss: tp.Optional[torch.Tensor] = field(default=None)
    lorentz: tp.Optional[torch.Tensor] = field(default=None)

    def to_tensor_dict(self) -> tp.Dict[str, torch.Tensor]:
        """Flatten width tensors into a dictionary.

        :return: A dictionary mapping string names to ``torch.Tensor`` objects.
        """
        tensor_dict = {}
        if self.gauss is not None:
            tensor_dict["width_gauss"] = self.gauss
        if self.lorentz is not None:
            tensor_dict["width_lorentz"] = self.lorentz
        if self.ham_strain is not None:
            tensor_dict["width_ham_strain"] = self.ham_strain
        return tensor_dict

    @classmethod
    def from_tensor_dict(cls,
                         tensor_dict: tp.Dict[str, torch.Tensor],
                         device: torch.device = torch.device("cpu"),
                         dtype: torch.dtype = torch.float32) -> SerializedSampleWidth:
        """Reconstruct width data from a flat tensor dictionary.

        :param tensor_dict: Dictionary of tensors.
        :return: A new :class:`SerializedSampleWidth` instance.
        """
        return cls(
            gauss=tensor_dict.get("width_gauss"),
            lorentz=tensor_dict.get("width_lorentz"),
            ham_strain=tensor_dict.get("width_ham_strain"),
        )

    def is_equivalent(self, other: SerializedSampleWidth, rtol: float = 1e-5, atol: float = 1e-6) -> bool:
        if not isinstance(other, SerializedSampleWidth):
            return False

        return all([
                utils.are_optional_tensors_close(self.ham_strain, other.ham_strain, rtol, atol),
                utils.are_optional_tensors_close(self.gauss, other.gauss, rtol, atol),
                utils.are_optional_tensors_close(self.lorentz, other.lorentz, rtol, atol),
        ])


@dataclass
class SampleMetaData:
    sample_type: str
    mesh_meta: tp.Optional[dict[str, tp.Any]] = None

    def to_json_dict(self) -> tp.Dict[str, tp.Any]:
        return {"sample_type": self.sample_type,
                "mesh_meta": self.mesh_meta}

    @classmethod
    def from_json_dict(cls, data: tp.Dict[str, tp.Any]) -> SampleMetaData:
        return cls(
            sample_type=data["sample_type"],
            mesh_meta=data.get("mesh_meta")
        )

    def is_equivalent(self, other: SampleMetaData, rtol: float = 1e-5, atol: float = 1e-6) -> bool:
        if not isinstance(other, SampleMetaData):
            return False
        return self.sample_type == other.sample_type and self.mesh_meta == other.mesh_meta


@dataclass
class SerializedSample:
    """Serialization of a sample into a JSON sidecar and a Safetensors file."""
    metadata: SampleMetaData
    serialized_spin_system: SerializedSpinSystem
    width: SerializedSampleWidth

    @classmethod
    def from_mars_sample(cls, sample: BaseSample) -> SerializedSample:
        """Create a serialised representation from live sample instance.

        :param sample: The source sample to be serialised.
        :return: A new :class:`SerializedSample` instance.
        """
        if isinstance(sample, MultiOrientedSampleExpandedStrain):
            sample_type = "MultiOrientedSampleExpandedStrain"
        elif isinstance(sample, MultiOrientedSample):
            sample_type = "MultiOrientedSample"
        elif isinstance(sample, BaseSample):
            sample_type = "BaseSample"
        else:
            raise TypeError(f"Unknown sample type: {type(sample)}")

        meta = SampleMetaData(sample_type=sample_type, mesh_meta=sample.mesh.to_json_dict())
        serialized_spin_system = SerializedSpinSystem.from_mars_spin_system(sample.base_spin_system)
        width = SerializedSampleWidth(
            gauss=sample.gauss,
            lorentz=sample.lorentz,
            ham_strain=getattr(sample, "base_ham_strain", None)
        )

        return cls(
            metadata=meta,
            serialized_spin_system=serialized_spin_system,
            width=width
        )

    def to_mars_sample(self) -> BaseSample:
        """Reconstruct a sample instance from the serialised data.

        :return: A new sample instance of the original type.
        """
        base_spin_system = self.serialized_spin_system.to_mars_spin_system()
        tensor_ref = self.serialized_spin_system.interactions.g_tensor_components
        device = tensor_ref.device
        dtype = tensor_ref.dtype
        mesh = self.mesh_from_json_dict(
            data=self.metadata.mesh_meta, device=device, dtype=dtype
        )

        sample_type = self.metadata.sample_type
        if sample_type == "MultiOrientedSampleExpandedStrain":
            return MultiOrientedSampleExpandedStrain(
                base_spin_system=base_spin_system,
                mesh=mesh,
                gauss=self.width.gauss,
                lorentz=self.width.lorentz,
                ham_strain=self.width.ham_strain
            )
        elif sample_type == "MultiOrientedSample":
            return MultiOrientedSample(
                base_spin_system=base_spin_system,
                mesh=mesh,
                gauss=self.width.gauss,
                lorentz=self.width.lorentz,
                ham_strain=self.width.ham_strain
            )
        else:
            return BaseSample(
                base_spin_system=base_spin_system,
                mesh=mesh,
                gauss=self.width.gauss,
                lorentz=self.width.lorentz
            )

    @classmethod
    def from_graph_sample(cls, graph_sample: GraphSample) -> SerializedSample:
        return graph_sample.to_serialized_sample()

    def to_graph_sample(self) -> GraphSample:
        from .graph_representation import GraphSample
        return GraphSample.from_serialized_sample(self)

    @staticmethod
    def mesh_from_json_dict(data: dict[str, tp.Any], device: torch.device = torch.device("cpu"),
                             dtype: torch.dtype = torch.float32) -> BaseMesh:
        """Factory function to instantiate a mesh from a JSON dictionary.

        :param data: Dictionary containing at least a ``"type"`` key specifying the mesh class.
        :param device: Computation device for tensor allocation.
        :param dtype: Floating point precision for computations.
        :return: An instantiated mesh object. Defaults to :class:`DelaunayMesh` if type is unknown.
        """
        mesh_type = data.get("type")
        if mesh_type == "CrystalMesh":
            return mesher.CrystalMesh.from_json_dict(data, device=device, dtype=dtype)
        elif mesh_type == "DelaunayMesh":
            return mesher.DelaunayMesh.from_json_dict(data, device=device, dtype=dtype)
        else:
            warnings.warn(
                f"Unknown mesh type '{mesh_type}'. Defaulting to DelaunayMesh."
            )
            return mesher.DelaunayMesh.from_json_dict(data, device=device, dtype=dtype)

    def to_json_dict(self) -> tp.Dict[str, tp.Any]:
        meta = self.metadata.to_json_dict()
        meta["spin_system_meta"] = self.serialized_spin_system.metadata.to_json_dict()
        return meta

    def to_file(self, base_path: tp.Union[pathlib.Path, str]) -> None:
        """Save the serialised sample to JSON and Safetensors files.

        :param base_path: The base path for the files (without extension).
                          Will create ``{base_path}.json`` and ``{base_path}.safetensors``.
        """
        tensor_dict = {}
        tensor_dict.update(self.serialized_spin_system.to_tensor_dict())
        tensor_dict.update(self.width.to_tensor_dict())

        _save_to_disk(metadata=self.to_json_dict(), tensor_dict=tensor_dict, base_path=base_path)

    @classmethod
    def from_file(cls, base_path: tp.Union[pathlib.Path, str],
                  device: torch.device = torch.device("cpu"),
                  dtype: torch.dtype = torch.float32) -> SerializedSample:
        """Load a serialised sample from JSON and Safetensors files.

        :param base_path: The base path for the files (without extension).
        :return: A new :class:`SerializedSample` instance.
        """
        metadata, tensor_dict = _load_from_disk(base_path, device=device, dtype=dtype)

        spin_system_meta = SpinSystemMetaData.from_json_dict(metadata["spin_system_meta"])
        serialized_spin_system = SerializedSpinSystem.from_tensor_dict(tensor_dict, spin_system_meta)
        width = SerializedSampleWidth.from_tensor_dict(tensor_dict)

        return cls(metadata=SampleMetaData.from_json_dict(metadata),
                   serialized_spin_system=serialized_spin_system, width=width)

    def is_equivalent(self, other: SerializedSample, rtol: float = 1e-5, atol: float = 1e-6) -> bool:
        if not isinstance(other, SerializedSample):
            return False
        if not self.metadata.is_equivalent(other.metadata, rtol, atol):
            return False
        if not self.serialized_spin_system.is_equivalent(other.serialized_spin_system, rtol, atol):
            return False
        if not self.width.is_equivalent(other.width, rtol, atol):
            return False
        return True


@dataclass
class PolarizationParameters:
    temperature: torch.Tensor
    basis: tp.Optional[str] = field(default=None)
    initial_populations: tp.Optional[torch.Tensor] = field(default=None)

    def __post_init__(self):
        """Ensure temperature is stored as a PyTorch tensor.

        Converts scalar values, lists, or NumPy arrays to a float32 tensor.
        """
        if not isinstance(self.temperature, torch.Tensor):
            if not self.initial_populations is not None:
                self.temperature = torch.as_tensor(self.temperature, dtype=torch.float32)
            else:
                self.temperature = torch.as_tensor(self.temperature,
                                                   dtype=self.initial_populations.dtype,
                                                   device=self.initial_populations.device)

    def to_dict(self) -> tp.Tuple[tp.Dict[str, tp.Any], tp.Dict[str, torch.Tensor]]:
        meta = {"basis": self.basis}
        tensors = {"temperature": self.temperature}
        if self.initial_populations is not None:
            tensors["initial_populations"] = self.initial_populations
        return meta, tensors

    @classmethod
    def from_dict(cls, meta: tp.Dict[str, tp.Any], tensors: tp.Dict[str, torch.Tensor]) -> PolarizationParameters:
        return cls(
            temperature=tensors["temperature"],
            basis=meta.get("basis"),
            initial_populations=tensors.get("initial_populations")
        )

    def to_file(self, base_path: tp.Union[str, pathlib.Path]) -> None:
        meta, tensors = self.to_dict()
        _save_to_disk(metadata=meta, tensor_dict=tensors, base_path=base_path)

    @classmethod
    def from_file(cls, base_path: tp.Union[str, pathlib.Path],
                  device: torch.device = torch.device("cpu"),
                  dtype: torch.dtype = torch.float32
                  ) -> PolarizationParameters:
        meta, tensors = _load_from_disk(base_path, device=device, dtype=dtype)
        return cls.from_dict(meta, tensors)


def reconstruct_positions(min_val: torch.Tensor, max_val: torch.Tensor, num_points: int) -> torch.Tensor:
    """
    General function to reconstruct positions between min and max bounds using linear interpolation.

    This function supports batched inputs. If `min_val` and `max_val` have shape `(B,)` or `(B, 1)`,
    the output will have shape `(B, num_points)`.

    If `min_val` and `max_val` have shape `()`, then the output will have shape `(num_points,)`.

    :param min_val: The starting position(s) tensor.
    :param max_val: The ending position(s) tensor. Must match the shape, device, and dtype of `min_val`.
    :param num_points: The number of points to generate between min and max.
    :return: Interpolated positions tensor.
    :rtype: torch.Tensor
    """
    if min_val.dim() == 0:
        weights = torch.linspace(
            0.0,
            1.0,
            num_points,
            device=min_val.device,
            dtype=min_val.dtype
        )
        return torch.lerp(min_val, max_val, weights)

    min_v = min_val.reshape(-1, 1)
    max_v = max_val.reshape(-1, 1)

    weights = torch.linspace(
        0.0,
        1.0,
        num_points,
        device=min_v.device,
        dtype=min_v.dtype
    ).unsqueeze(0)
    return torch.lerp(min_v, max_v, weights)


@dataclass
class TimeParameters:
    """
    Holds time parameters for temporal reconstruction in an experiment.

    :param min_time: The starting time(s) in seconds. Can be a batched tensor
                     of shape `(B,)` or `(B, 1)`.
    :param max_time: The ending time(s) in seconds. Must match the shape,
                     device, and dtype of `min_time`.
    :param num_points: The number of time points to generate between min and max.
    """
    min_time: torch.Tensor
    max_time: torch.Tensor
    num_points: int

    def reconstruct_time(self) -> torch.Tensor:
        """
        Reconstructs the time positions between min and max bounds.

        :return: The reconstructed time positions.
        :rtype: torch.Tensor
        """
        return reconstruct_positions(self.min_time, self.max_time, self.num_points)

    def to_dict(self) -> tp.Tuple[tp.Dict[str, tp.Any], tp.Dict[str, torch.Tensor]]:
        """Separates scalar metadata from tensor data for serialization."""
        meta = {"num_points": self.num_points}
        tensors = {
            "min_time": self.min_time,
            "max_time": self.max_time,
        }
        return meta, tensors

    @classmethod
    def from_dict(cls, meta: tp.Dict[str, tp.Any], tensors: tp.Dict[str, torch.Tensor]) -> TimeParameters:
        """Reconstructs the TimeParameters instance from separated dictionaries."""
        return cls(
            min_time=tensors["min_time"],
            max_time=tensors["max_time"],
            num_points=meta["num_points"],
        )

    def to_file(self, base_path: tp.Union[str, pathlib.Path]) -> None:
        """Saves the time parameters to disk."""
        meta, tensors = self.to_dict()
        _save_to_disk(metadata=meta, tensor_dict=tensors, base_path=base_path)

    @classmethod
    def from_file(cls, base_path: tp.Union[str, pathlib.Path],
                  device: torch.device = torch.device("cpu"),
                  dtype: torch.dtype = torch.float32) -> TimeParameters:
        """Loads time parameters from disk."""
        meta, tensors = _load_from_disk(base_path, device=device, dtype=dtype)
        return cls.from_dict(meta, tensors)

    @classmethod
    def parse_time(cls, time: tp.Optional[torch.Tensor]) -> tp.Optional[TimeParameters]:
        """Parses a 1D time tensor into TimeParameters."""
        if time is None:
            return None

        return cls(
            min_time=torch.min(time),
            max_time=torch.max(time),
            num_points=int(time.shape[0]),
        )


@dataclass
class ExperimentalParameters:
    """
    Holds generalized experimental parameters for 1D grid reconstruction.

    This class is domain-agnostic. Depending on the use case, the positional
    bounds (`min_pos`, `max_pos`) and the `resonance_parameter` can represent:
    - Magnetic field bounds and Larmor frequencies.
    - Frequency ranges and associated spectral parameters.
    - Arbitrary  feature space bounds and conditioning vectors.

    :param min_pos: The starting position(s) of the domain grid. Can be a
                    batched tensor of shape `(B,)` or `(B, 1)`. Represents the
                    lower bound of the field, frequency, or feature space.
    :type min_pos: torch.Tensor
    :param max_pos: The ending position(s) of the domain grid. Must match the
                    shape, device, and dtype of `min_pos`. Represents the upper
                    bound of the field, frequency, or feature space.
    :type max_pos: torch.Tensor
    :param num_points: The number of discretization points to generate linearly
                       between `min_pos` and `max_pos`. If the bounds are batched,
                       this number of points is generated for each item in the batch.
    :type num_points: int
    :param resonance_parameter: Optional context-dependent tensor associated with
                                the experiment (e.g., operating frequency for field
                                reconstruction, or a specific feature embedding).
    :type resonance_parameter: tp.Optional[torch.Tensor]
    :param time_params: Optional TimeParameters instance for temporal reconstruction
                        or time-dependent conditioning.
    :type time_params: tp.Optional[TimeParameters]
    """
    min_pos: torch.Tensor
    max_pos: torch.Tensor
    num_points: int
    resonance_parameter: tp.Optional[torch.Tensor] = field(default=None)
    time_params: tp.Optional[TimeParameters] = field(default=None)

    def to_dict(self) -> tp.Tuple[tp.Dict[str, tp.Any], tp.Dict[str, torch.Tensor]]:
        """
        Separates scalar metadata (JSON-serializable) from tensor data.

        :return: A tuple containing a dictionary of scalar values (e.g., num_points)
                 and a dictionary of PyTorch tensors to be saved via Safetensors.
        :rtype: tp.Tuple[tp.Dict[str, tp.Any], tp.Dict[str, torch.Tensor]]
        """
        meta = {"num_points": self.num_points}
        tensors = {
            "min_pos": self.min_pos,
            "max_pos": self.max_pos,
        }
        if self.resonance_parameter is not None:
            tensors["resonance_parameter"] = self.resonance_parameter

        if self.time_params is not None:
            time_meta, time_tensors = self.time_params.to_dict()
            meta["time_params"] = time_meta
            for k, v in time_tensors.items():
                tensors[f"time_{k}"] = v

        return meta, tensors

    @classmethod
    def from_dict(cls, meta: tp.Dict[str, tp.Any], tensors: tp.Dict[str, torch.Tensor]) -> ExperimentalParameters:
        """Reconstructs the ExperimentalParameters instance from separated dictionaries."""
        time_params = None
        if "time_params" in meta and meta["time_params"] is not None:
            time_meta = meta["time_params"]
            time_tensors = {
                k[len("time_"):]: v
                for k, v in tensors.items()
                if k.startswith("time_")
            }
            time_params = TimeParameters.from_dict(time_meta, time_tensors)

        return cls(
            min_pos=tensors["min_pos"],
            max_pos=tensors["max_pos"],
            num_points=meta["num_points"],
            resonance_parameter=tensors.get("resonance_parameter"),
            time_params=time_params,
        )

    def to_file(self, base_path: tp.Union[str, pathlib.Path]) -> None:
        """
        Saves the experimental parameters to disk.

        Separates the data into a JSON-compatible metadata file and a
        Safetensors file for efficient tensor storage.

        :param base_path: The base file path (without extension) to save the data to.
        :type base_path: tp.Union[str, pathlib.Path]
        """
        meta, tensors = self.to_dict()
        _save_to_disk(metadata=meta, tensor_dict=tensors, base_path=base_path)

    @classmethod
    def from_file(cls, base_path: tp.Union[str, pathlib.Path],
                  device: torch.device = torch.device("cpu"),
                  dtype: torch.dtype = torch.float32) -> ExperimentalParameters:
        """
        Loads experimental parameters from disk.

        :param base_path: The base file path (without extension) to load the data from.
        :type base_path: tp.Union[str, pathlib.Path]
        :return: An instantiated ExperimentalParameters object populated with the loaded data.
        :rtype: ExperimentalParameters
        """
        meta, tensors = _load_from_disk(base_path, device=device, dtype=dtype)
        return cls.from_dict(meta, tensors)

    def reconstruct_grid(self) -> torch.Tensor:
        """
        Reconstructs the 1D grid positions between `min_pos` and `max_pos` bounds.
        This method supports batched inputs. It generates a linearly spaced grid
        for each batch element.

        :return: The reconstructed grid positions. If inputs are batched with
                 shape `(B,)` or `(B, 1)`, the output will have shape `(B, num_points)`.
        """
        return reconstruct_positions(self.min_pos, self.max_pos, self.num_points)

    @classmethod
    def parse_grid(cls, grid: tp.Optional[torch.Tensor],
                   resonance_parameter: tp.Optional[torch.Tensor] = None,
                   time: tp.Optional[torch.Tensor] = None) -> tp.Optional[ExperimentalParameters]:
        """
        Infers and parses 1D grid parameters from a given tensor.

        This is a convenience method to automatically extract `min_pos`, `max_pos`,
        and `num_points` from an existing 1D tensor (e.g., an actual field array,
        frequency array, or feature vector), along with optional conditioning parameters.

        :param grid: The 1D tensor representing the domain (field, frequency, or features).
                     Its min, max, and shape[0] will be used to populate the parameters.
        :param resonance_parameter: Optional context-dependent tensor (e.g., frequency or feature embedding).
        :param time: Optional 1D time tensor to be parsed into TimeParameters.
        :return: An instantiated ExperimentalParameters object, or None if `grid` is None.
        """
        if grid is None:
            return None

        time_params = TimeParameters.parse_time(time) if time is not None else None

        return cls(
            min_pos=torch.min(grid),
            max_pos=torch.max(grid),
            num_points=int(grid.shape[0]),
            resonance_parameter=resonance_parameter,
            time_params=time_params,
        )


@dataclass
class CWSpectralData:
    """
    :param experimental_parameters: ExperimentalParameters from serialization module
    :param spectrum: generated EPR spectrum
    """
    experimental_parameters: ExperimentalParameters
    spectrum: torch.Tensor

    def to_dict(self) -> tp.Tuple[tp.Dict[str, tp.Any], tp.Dict[str, torch.Tensor]]:
        """
        Separates scalar metadata (JSON-serializable) from tensor data.

        :return: A tuple containing a dictionary of scalar values (e.g., num_points)
                 and a dictionary of PyTorch tensors to be saved via Safetensors.
        :rtype: tp.Tuple[tp.Dict[str, tp.Any], tp.Dict[str, torch.Tensor]]
        """
        meta, tensors = self.experimental_parameters.to_dict()
        tensors["spectrum"] = self.spectrum
        return meta, tensors

    @classmethod
    def from_dict(cls, meta: tp.Dict[str, tp.Any], tensors: tp.Dict[str, torch.Tensor]) -> CWSpectralData:
        """
        Reconstructs the ExperimentalParameters instance from separated dictionaries.

        :param meta: Dictionary containing scalar metadata.
        :type meta: tp.Dict[str, tp.Any]
        :param tensors: Dictionary containing loaded PyTorch tensors.
        :type tensors: tp.Dict[str, torch.Tensor]
        :return: An instantiated ExperimentalParameters object.
        :rtype: ExperimentalParameters
        """
        experimental_parameters = ExperimentalParameters(
            min_pos=tensors["min_pos"],
            max_pos=tensors["max_pos"],
            num_points=meta["num_points"],
            resonance_parameter=tensors.get("resonance_parameter")
        )
        spectrum = tensors["spectrum"]
        return cls(experimental_parameters=experimental_parameters, spectrum=spectrum)

    def to_file(self, base_path: tp.Union[str, pathlib.Path]) -> None:
        """
        Saves the experimental parameters to disk.

        Separates the data into a JSON-compatible metadata file and a
        Safetensors file for efficient tensor storage.

        :param base_path: The base file path (without extension) to save the data to.
        :type base_path: tp.Union[str, pathlib.Path]
        """
        meta, tensors = self.to_dict()
        _save_to_disk(metadata=meta, tensor_dict=tensors, base_path=base_path)

    @classmethod
    def from_file(cls, base_path: tp.Union[str, pathlib.Path],
                  device: torch.device = torch.device("cpu"),
                  dtype: torch.dtype = torch.float32
                  ) -> CWSpectralData:
        """
        Loads experimental parameters from disk.

        :param base_path: The base file path (without extension) to load the data from.
        :type base_path: tp.Union[str, pathlib.Path]
        :return: An instantiated ExperimentalParameters object populated with the loaded data.
        :rtype: ExperimentalParameters
        """
        meta, tensors = _load_from_disk(base_path, device=device, dtype=dtype)
        return cls.from_dict(meta, tensors)


@dataclass
class SerializedMarsSession:
    """Dataclass representing a serialized MaRs experiment.

    Contains serialized components that can be easily saved to disk
    and later deserialized into a full MaRsExperiment.
    """
    sample: SerializedSample
    experimental_parameters: tp.Optional[ExperimentalParameters] = None
    polarization: tp.Optional[PolarizationParameters] = None

    def _creator_from_sample(self,
                             sample: BaseSample,
                             sample_type: str,
                             temperature: tp.Union[float, torch.Tensor],
                             freq: tp.Union[float, torch.Tensor],
                             device: torch.device = torch.device("cpu"),
                             dtype: torch.dtype = torch.float32
                             ):
        if sample_type == "MultiOrientedSample":
            return StationarySpectra(
                sample=sample,
                temperature=temperature,
                freq=freq,
                device=device,
                dtype=dtype
            )
        else:
            raise NotImplementedError("Currently Load from file support only StationarySpectra")

    def to_experiment(self, device: torch.device = torch.device("cpu"),
                      dtype: torch.dtype = torch.float32) -> MarsSession:
        """Convert the serialized experiment into a non-serialized MaRsExperiment.

        :param device: Torch device to place the tensors on.
        :param dtype: Torch data type for the tensors.
        :return: A fully instantiated MaRsExperiment.
        """
        sample = self.sample.to_mars_sample()
        field = None
        if self.experimental_parameters is not None:
            field = self.experimental_parameters.reconstruct_grid()
        creator = None
        temperature = None
        if (self.experimental_parameters is not None and
                self.polarization is not None and
                self.experimental_parameters.resonance_parameter is not None and
                self.polarization.temperature is not None):
            creator = self._creator_from_sample(
                sample=sample,
                sample_type=self.sample.metadata.sample_type,
                temperature=self.polarization.temperature,
                freq=torch.as_tensor(self.experimental_parameters.resonance_parameter),
                device=device, dtype=dtype
            )
            temperature = self.polarization.temperature

        return MarsSession(
            sample=sample,
            creator=creator,
            field=field,
            temperature=temperature
        )

    def to_file(self, filepath: tp.Union[str, pathlib.Path]):
        """Save the serialized experiment.

        Supports both single-file and folder-based saving:
        - If filepath is a directory (or has no extension), saves components
          as separate files inside the folder using their own to_file() methods.
        - If filepath is a file, saves the entire dataclass as a single PyTorch file.

        :param filepath: Path to the file or folder.
        """
        path = pathlib.Path(filepath)
        if path.is_dir() or path.suffix == "":
            path.mkdir(parents=True, exist_ok=True)
            self.sample.to_file(path / "sample")
            if self.experimental_parameters is not None:
                self.experimental_parameters.to_file(path / "experimental_parameters")
            if self.polarization is not None:
                self.polarization.to_file(path / "polarization")
        else:
            raise NotImplementedError("filepath has wrong format. filepath should be folder")

    @classmethod
    def from_file(cls, filepath: tp.Union[str, pathlib.Path],
                  device: torch.device = torch.device("cpu"),
                  dtype: torch.dtype = torch.float32) -> SerializedMarsSession:
        """Load a serialized experiment.

        Supports folder-based loading only folder-based loading.

        :param filepath: Path to the file or folder.
        :return: The loaded MaRsExperimentData.
        """
        path = pathlib.Path(filepath)
        if path.is_dir():
            sample = SerializedSample.from_file(path / "sample", device=device, dtype=dtype)

            exp_params_path = path / "experimental_parameters"
            exp_params = None
            if exp_params_path.exists():
                exp_params = ExperimentalParameters.from_file(exp_params_path, device=device, dtype=dtype)

            pol_path = path / "polarization"
            polarization = None
            if pol_path.exists():
                polarization = PolarizationParameters.from_file(pol_path, device=device, dtype=dtype)

            return cls(sample=sample, experimental_parameters=exp_params, polarization=polarization)
        else:
            raise NotImplementedError("'filepath' should be a dir with files with suffixes: "
                                      "'sample', 'experimental_parameters', 'polarization' ")


@dataclass
class MarsSession:
    """Dataclass representing a non-serialized (fully instantiated) MaRs experiment.

    Contains actual sample objects, spectra creators, and field tensors.
    """
    sample: BaseSample
    creator: tp.Optional[BaseResSpectra] = None
    field: tp.Optional[torch.Tensor] = None
    temperature: tp.Optional[tp.Union[torch.Tensor, float, int]] = None

    def __post_init__(self):
        """Ensure temperature is stored as a PyTorch tensor.

        Converts scalar values, lists, or NumPy arrays to a float32 tensor.
        """
        if self.temperature is not None and not isinstance(self.temperature, torch.Tensor):
            self.temperature = torch.as_tensor(self.temperature, dtype=torch.float32)

    def to_serialized(self) -> SerializedMarsSession:
        """Convert the experiment into a serialized format.

        :return: A SerializedMarsSession instance.
        """
        sample_serialized = SerializedSample.from_mars_sample(self.sample)

        exp_params = None
        if self.creator is not None and self.field is not None:
            freq = getattr(self.creator, "resonance_parameter", None)
            exp_params = ExperimentalParameters.parse_grid(self.field, freq)

        polarization = None
        if self.temperature is not None:
            polarization = PolarizationParameters(temperature=self.temperature)
        elif self.creator is not None:
            temp = self.creator.intensity_calculator.temperature
            if temp is not None and not isinstance(temp, torch.Tensor):
                temp = torch.as_tensor(temp, dtype=torch.float32)
            polarization = PolarizationParameters(temperature=temp)

        return SerializedMarsSession(
            sample=sample_serialized,
            experimental_parameters=exp_params,
            polarization=polarization
        )

    def to_file(self, filepath: tp.Union[str, pathlib.Path]):
        """Save the experiment by converting it to serialized format first.
        :param filepath: Path to the file or folder.
        """
        self.to_serialized().to_file(filepath)

    @classmethod
    def from_file(cls, filepath: tp.Union[str, pathlib.Path],
                  device: torch.device = torch.device("cpu"),
                  dtype: torch.dtype = torch.float32) -> MarsSession:
        """Load an experiment from a file or folder.

        :param filepath: Path to the file or folder.
        :param device: Torch device to place the tensors on.
        :param dtype: Torch data type for the tensors.
        :return: The loaded MaRsExperiment.
        """
        serialized = SerializedMarsSession.from_file(filepath, device=device, dtype=dtype)
        return serialized.to_experiment(device=device, dtype=dtype)
