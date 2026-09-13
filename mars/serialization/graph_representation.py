from __future__ import annotations
import warnings

from dataclasses import dataclass
import typing as tp

import torch
import torch.nn as nn

from .. import particles

from ..spin_model import BaseSample, SolidSample, SolidSampleExpandedStrain,\
    SpinSystem

from . import serialization, graph_equivalence


@dataclass
class GraphSpinSystem:
    """
    Graph representation of a spin system for machine learning or graph-based processing.
    Current version of graph representation doesn't support interaction strains

    :param source: Source node indices for edges. Shape: [num_edges].
    :param destination: Destination node indices for edges. Shape: [num_edges].
    :param components: Node features representing interaction components (e.g., g-tensor, hyperfine).
                       Shape: [..., num_nodes, 3].
                       For g-tensor and hyperfine interactions this value is the g-factor (electron or nucleus)

    :param angles: Node features representing Euler angles for each node.
                   Shape: [..., num_nodes, 3].
    :param spins: Spin quantum numbers for each node. Shape: [..., num_nodes]. For interaction nodes spin is set as 0.0
    :param nucleus_labels: Optional list of nucleus isotope strings (e.g., ["14N", "1H"]).
                           Preserves nucleus identities during graph serialization/deserialization.
    :param node_types: Integer type identifiers for each node (0=Electron, 1=Nucleus, 2=Interaction).
                       Shape: [..., num_nodes].
    """
    source: torch.Tensor
    destination: torch.Tensor
    components: torch.Tensor
    angles: torch.Tensor
    spins: torch.Tensor
    node_types: torch.Tensor
    nucleus_labels: tp.Optional[tp.List[str]] = None

    ELECTRON_TYPE: tp.ClassVar[int] = 0
    NUCLEI_TYPE: tp.ClassVar[int] = 1
    INTRA_TYPE: tp.ClassVar[int] = 2

    def __post_init__(self):
        self.modified = False

    @classmethod
    def _parse_interactions(cls, base_spin_system: SpinSystem) -> tuple[
        list[torch.Tensor], list[torch.Tensor], list[int], list[int]]:
        """
        Parse all interactions (e-e, e-n, n-n) into node components, angles, sources, and destinations.

        :param base_spin_system: The spin system to parse.
        :return: A tuple containing lists of components, angles,    destination node indices, and source node indices.
        """
        components = []
        angles = []
        destination = []
        source = []

        num_electrons = len(base_spin_system.electrons)
        num_nuclei = len(base_spin_system.nuclei)
        num_particles = num_electrons + num_nuclei
        current_interaction_idx = 0

        def _process(interactions: nn.ModuleList, indices: list[tuple[int, int]], idx_1_shift: int, idx_2_shift: int):
            nonlocal current_interaction_idx
            for (idx_1, idx_2), inter in zip(indices, interactions):
                components.append(inter.components)
                angles.append(inter.frame)

                interaction_node_id = num_particles + current_interaction_idx
                current_interaction_idx += 1

                if idx_1 == idx_2:
                    source.append(interaction_node_id)
                    destination.append(idx_1 + idx_1_shift)
                else:
                    source.append(interaction_node_id)
                    destination.append(idx_1 + idx_1_shift)
                    source.append(interaction_node_id)
                    destination.append(idx_2 + idx_2_shift)

        _process(
            base_spin_system.electron_electron_interactions,
            base_spin_system.ee_indices,
            idx_1_shift=0,
            idx_2_shift=0
        )

        _process(
            base_spin_system.electron_nuclei_interactions,
            base_spin_system.en_indices,
            idx_1_shift=0,
            idx_2_shift=num_electrons
        )

        _process(
            base_spin_system.nuclei_nuclei_interactions,
            base_spin_system.nn_indices,
            idx_1_shift=num_electrons,
            idx_2_shift=num_electrons
        )
        return components, angles, destination, source

    @classmethod
    def _parse_electrons(cls, base_spin_system: SpinSystem) -> tuple[
        list[torch.Tensor], list[torch.Tensor], list[float]]:
        """
        Parse electron properties (g-tensors, frames, spins).

        :param base_spin_system: The spin system to parse.
        :return: A tuple containing lists of g-tensor components, Euler angles, and spin values.
        """
        components = []
        angles = []
        spins = []
        for electron, g_interaction in zip(base_spin_system.electrons, base_spin_system.g_tensors):
            components.append(g_interaction.components)
            angles.append(g_interaction.frame)
            spins.append(electron.spin)
        return components, angles, spins

    @classmethod
    def _parse_nuclei(cls, base_spin_system: SpinSystem) -> tuple[
        list[torch.Tensor], list[torch.Tensor], list[float]]:
        """
        Parse nucleus properties (g-factors, frames, spins).

        :param base_spin_system: The spin system to parse.
        :return: A tuple containing lists of g-factor components, Euler angles, and spin values.
        """
        components = []
        angles = []
        spins = []
        config_shape = base_spin_system.config_shape
        device = base_spin_system.device
        dtype = base_spin_system.dtype

        for nucleus in base_spin_system.nuclei:
            g_tensor_val = nucleus.g_factor
            components.append(torch.full((*config_shape, 3), g_tensor_val, dtype=dtype, device=device))
            spins.append(nucleus.spin)
            angles.append(torch.zeros((*config_shape, 3), dtype=dtype, device=device))

        return components, angles, spins

    @classmethod
    def _parse_base_spin_system(cls, base_spin_system: SpinSystem) -> dict:
        """
        Parse the entire spin system into graph representation tensors.

        :param base_spin_system: The spin system to parse.
        :return: A dictionary containing components, angles, destination, source, spins, and node_types.
        """
        components_el, angles_el, spins_el = cls._parse_electrons(base_spin_system)
        components_nuc, angles_nuc, spins_nuc = cls._parse_nuclei(base_spin_system)
        components_inter, angles_inter, destination, source = cls._parse_interactions(base_spin_system)

        num_el = len(base_spin_system.electrons)
        num_nuc = len(base_spin_system.nuclei)
        num_inter = len(components_inter)
        num_nodes = num_el + num_nuc + num_inter

        spins_list = spins_el + spins_nuc + [0.0] * num_inter
        spins = torch.tensor(spins_list, device=base_spin_system.device, dtype=base_spin_system.dtype)

        all_components = components_el + components_nuc + components_inter
        all_angles = angles_el + angles_nuc + angles_inter
        components = torch.stack(all_components, dim=-2)
        angles = torch.stack(all_angles, dim=-2)

        source_tensor = torch.tensor(source, device=base_spin_system.device, dtype=torch.long)
        destination_tensor = torch.tensor(destination, device=base_spin_system.device, dtype=torch.long)

        types_list = (
                [cls.ELECTRON_TYPE] * num_el +
                [cls.NUCLEI_TYPE] * num_nuc +
                [cls.INTRA_TYPE] * num_inter
        )
        types = torch.tensor(types_list, device=base_spin_system.device, dtype=torch.long)
        view_shape = [1] * len(base_spin_system.config_shape) + [num_nodes]
        expand_shape = list(base_spin_system.config_shape) + [num_nodes]

        spins = spins.view(view_shape).expand(expand_shape)
        types = types.view(view_shape).expand(expand_shape)

        nucleus_labels = [nucleus.nucleus_str for nucleus in base_spin_system.nuclei]

        return {
            "source": source_tensor,
            "destination": destination_tensor,
            "components": components,
            "angles": angles,
            "spins": spins,
            "node_types": types,
            "nucleus_labels": nucleus_labels
        }

    @classmethod
    def from_mars_spin_system(cls, spin_system: SpinSystem) -> GraphSpinSystem:
        """
        Create a GraphSpinSystemRepresentation from a SpinSystem instance.

        :param spin_system: The SpinSystem to convert.
        :return: A new GraphSpinSystemRepresentation instance.
        """
        parsed_data = cls._parse_base_spin_system(spin_system)
        return cls(**parsed_data)

    @classmethod
    def from_serialized_spin_system(cls,
            serialized_system: serialization.SerializedSpinSystem) -> GraphSpinSystem:
        """
        Create a graph representation from a serialized spin system.
        It directly convert serialized spin_system to graph, avoiding the creation of the spin_system

        :param serialized_system: The serialized spin system to convert.
        :return: A new GraphSpinSystemRepresentation instance.
        """
        meta = serialized_system.metadata
        inter = serialized_system.interactions
        device, dtype = inter.g_tensor_components.device, inter.g_tensor_components.dtype
        batch_shape = inter.g_tensor_components.shape[:-2]
        num_el, num_nuc = meta.num_electrons, meta.num_nuclei

        components_list, angles_list, source, destination = [], [], [], []
        current_node = num_el + num_nuc

        components_list.append(inter.g_tensor_components)
        angles_list.append(inter.g_tensor_orientations)

        if num_nuc > 0:
            nucleus_labels = meta.nucleus_labels
            g_values = [particles.Nucleus.get_g_factor(label) for label in nucleus_labels]
            nuc_comp = (
                torch.tensor(g_values, device=device, dtype=dtype)
                .unsqueeze(-1)
                .expand(-1, 3)
                .view(1, num_nuc, 3)
                .expand(*batch_shape, num_nuc, 3)
            )
            nuc_ang = torch.zeros((*batch_shape, num_nuc, 3), device=device, dtype=dtype)
            components_list.append(nuc_comp)
            angles_list.append(nuc_ang)

        if inter.hyperfine_coupling_components is not None:
            components_list.append(inter.hyperfine_coupling_components)
            angles_list.append(inter.hyperfine_coupling_orientations)
            for i, (e_idx, n_idx) in enumerate(meta.electron_nucleus_pairs):
                source.extend([current_node + i] * 2)
                destination.extend([e_idx, n_idx + num_el])
            current_node += len(meta.electron_nucleus_pairs)

        n_zfs = len(meta.zfs_pairs or [])
        if n_zfs > 0:
            components_list.append(inter.zfs_components)
            angles_list.append(inter.electron_electron_orientations[..., :n_zfs, :])
            for i, (e1, _) in enumerate(meta.zfs_pairs):
                source.append(current_node + i)
                destination.append(e1)
            current_node += n_zfs

        n_dip = len(meta.exchange_dipolar_pairs or [])
        if n_dip > 0:
            components_list.append(inter.dipolar_components)
            angles_list.append(inter.electron_electron_orientations[..., n_zfs:n_zfs + n_dip, :])
            for i, (e1, e2) in enumerate(meta.exchange_dipolar_pairs):
                source.extend([current_node + i] * 2)
                destination.extend([e1, e2])
            current_node += n_dip

        n_nn = len(meta.nucleus_nucleus_pairs or [])
        if n_nn > 0:
            components_list.append(inter.nuclear_coupling_components)
            angles_list.append(inter.nuclear_coupling_orientations)
            for i, (n1, n2) in enumerate(meta.nucleus_nucleus_pairs):
                source.extend([current_node + i] * 2)
                destination.extend([n1 + num_el, n2 + num_el])
            current_node += n_nn

        components = torch.concat(components_list, dim=-2)
        angles = torch.concat(angles_list, dim=-2)
        source_tensor = torch.tensor(source, device=device, dtype=torch.long)
        destination_tensor = torch.tensor(destination, device=device, dtype=torch.long)

        spins_list = list(meta.electron_spins) + list(meta.nuclear_spins) + [0.0] * (current_node - num_el - num_nuc)
        node_types_list = ([cls.ELECTRON_TYPE] * num_el + [cls.NUCLEI_TYPE] * num_nuc +
                           [cls.INTRA_TYPE] * (current_node - num_el - num_nuc))

        expand_shape = (*batch_shape, len(spins_list))
        spins = torch.tensor(spins_list, device=device, dtype=dtype).unsqueeze(0).expand(*expand_shape)
        node_types = torch.tensor(node_types_list, device=device, dtype=torch.long).unsqueeze(0).expand(*expand_shape)

        nucleus_labels = meta.nucleus_labels if meta.nucleus_labels else None

        return cls(source=source_tensor, destination=destination_tensor, components=components,
                   angles=angles, spins=spins, node_types=node_types, nucleus_labels=nucleus_labels)

    def to_serialized_spin_system(self) -> serialization.SerializedSpinSystem:
        """
        Reconstruct a serialized spin system from the graph representation.

        This method is invariant to the ordering of nodes in the graph.
        It uses `node_types` to dynamically identify electrons, nuclei, and interactions,
        and maps their absolute graph indices to relative indices for the serialized format.

        :return: A new SerializedSpinSystem instance.
        """
        if self.modified:
            warnings.warn("The spin-system-graph was modified. The transformation result can be incorrect", UserWarning)

        batch_shape = self.components.shape[:-2]
        idx = (0,) * len(batch_shape) if batch_shape else ()
        node_types_1d = self.node_types[idx]

        el_indices = (node_types_1d == self.ELECTRON_TYPE).nonzero(as_tuple=True)[0].tolist()

        nuc_indices = (node_types_1d == self.NUCLEI_TYPE).nonzero(as_tuple=True)[0].tolist()
        inter_indices = (node_types_1d == self.INTRA_TYPE).nonzero(as_tuple=True)[0].tolist()

        el_to_rel = {abs_idx: rel_idx for rel_idx, abs_idx in enumerate(el_indices)}
        nuc_to_rel = {abs_idx: rel_idx for rel_idx, abs_idx in enumerate(nuc_indices)}

        g_comp = self.components[..., el_indices, :]
        g_angles = self.angles[..., el_indices, :]

        hfc_c, hfc_a, hfc_p = [], [], []
        zfs_c, zfs_a, zfs_p = [], [], []
        dip_c, dip_a, dip_p = [], [], []
        nn_c, nn_a, nn_p = [], [], []

        for n_idx in inter_indices:
            mask = self.source == n_idx
            dests = self.destination[mask].tolist()
            dest_types = [node_types_1d[d].item() for d in dests]

            comp_slice = self.components[..., n_idx, :]
            ang_slice = self.angles[..., n_idx, :]

            if len(dests) == 1 and dest_types[0] == self.ELECTRON_TYPE:
                zfs_c.append(comp_slice)
                zfs_a.append(ang_slice)
                zfs_p.append((el_to_rel[dests[0]], el_to_rel[dests[0]]))

            elif len(dests) == 2 and all(t == self.ELECTRON_TYPE for t in dest_types):
                dip_c.append(comp_slice)
                dip_a.append(ang_slice)
                dip_p.append((el_to_rel[dests[0]], el_to_rel[dests[1]]))

            elif len(dests) == 2 and set(dest_types) == {self.ELECTRON_TYPE, self.NUCLEI_TYPE}:
                e_abs = dests[dest_types.index(self.ELECTRON_TYPE)]
                n_abs = dests[dest_types.index(self.NUCLEI_TYPE)]
                hfc_c.append(comp_slice)
                hfc_a.append(ang_slice)
                hfc_p.append((el_to_rel[e_abs], nuc_to_rel[n_abs]))

            elif len(dests) == 2 and all(t == self.NUCLEI_TYPE for t in dest_types):
                nn_c.append(comp_slice)
                nn_a.append(ang_slice)
                nn_p.append((nuc_to_rel[dests[0]], nuc_to_rel[dests[1]]))

        def _stack_or_none(lst: list) -> tp.Optional[torch.Tensor]:
            return torch.stack(lst, dim=-2) if lst else None

        spins_1d = self.spins[idx].tolist()
        el_spins = [spins_1d[i] for i in el_indices]
        nuc_spins = [spins_1d[i] for i in nuc_indices]

        if self.nucleus_labels is not None and len(self.nucleus_labels) == len(nuc_indices):
            final_nucleus_labels = self.nucleus_labels
        else:
            final_nucleus_labels = [f"Nucl_{i}" for i in range(len(nuc_indices))]

        meta = serialization.SpinSystemMetaData(
            electron_spins=el_spins,
            nucleus_labels=final_nucleus_labels,
            nuclear_spins=nuc_spins,
            electron_nucleus_pairs=hfc_p,
            electron_electron_pairs=zfs_p + dip_p,
            zfs_pairs=zfs_p,
            exchange_dipolar_pairs=dip_p,
            nucleus_nucleus_pairs=nn_p
        )

        ee_angles = _stack_or_none(zfs_a + dip_a)
        inter = serialization.SpinSystemInteractions(
            g_tensor_components=g_comp,
            g_tensor_orientations=g_angles,
            hyperfine_coupling_components=_stack_or_none(hfc_c),
            hyperfine_coupling_orientations=_stack_or_none(hfc_a),
            zfs_components=_stack_or_none(zfs_c),
            dipolar_components=_stack_or_none(dip_c),
            electron_electron_orientations=ee_angles,
            nuclear_coupling_components=_stack_or_none(nn_c),
            nuclear_coupling_orientations=_stack_or_none(nn_a)
        )
        return serialization.SerializedSpinSystem(metadata=meta, interactions=inter, strain=None)

    def to_mars_spin_system(self) -> SpinSystem:
        """
        Reconstruct a live SpinSystem from the graph representation.

        :return: A new SpinSystem instance with reconstructed particles and interactions.
        """
        serialized = self.to_serialized_spin_system()
        return serialized.to_mars_spin_system()

    def is_equivalent(self, other: GraphSpinSystem, rtol: float = 1e-5, atol: float = 1e-6) -> bool:
        if not isinstance(other, GraphSpinSystem):
            return False
        return graph_equivalence.are_graphs_equivalent(self, other, rtol, atol)


@dataclass
class GraphSample:
    """Serialization of a sample into a JSON sidecar and a Safetensors file."""
    metadata: serialization.SampleMetaData
    graph_spin_system: GraphSpinSystem
    width: serialization.SerializedSampleWidth

    def __post_init__(self):
        if self.graph_spin_system.modified:
            self.modified = True
        else:
            self.modified = False

    @classmethod
    def from_mars_sample(cls, sample: BaseSample) -> GraphSample:
        """Create a graph representation from sample instance.

        :param sample: The source sample to be serialised.
        :return: A new :class:`SerializedSample` instance.
        """
        if isinstance(sample, SolidSampleExpandedStrain):
            sample_type = "SolidSampleExpandedStrain"
        elif isinstance(sample, SolidSample):
            sample_type = "SolidSample"
        elif isinstance(sample, BaseSample):
            sample_type = "BaseSample"
        else:
            raise TypeError(f"Unknown sample type: {type(sample)}")

        meta = serialization.SampleMetaData(sample_type=sample_type, mesh_meta=sample.mesh.to_json_dict())
        graph_spin_system = GraphSpinSystem.from_mars_spin_system(sample.base_spin_system)

        width = serialization.SerializedSampleWidth(
            gauss=sample.gauss,
            lorentz=sample.lorentz,
            ham_strain=getattr(sample, "base_ham_strain", None)
        )
        return cls(
            metadata=meta,
            graph_spin_system=graph_spin_system,
            width=width
        )

    def to_mars_sample(self) -> BaseSample:
        """Reconstruct a sample instance from the serialised data.

        :return: A new sample instance of the original type.
        """
        if self.modified:
            warnings.warn("The sample-graph was modified. The transformation result can be incorrect", UserWarning)

        base_spin_system = self.graph_spin_system.to_mars_spin_system()
        tensor_ref = self.graph_spin_system.components[0]
        device = tensor_ref.device
        dtype = tensor_ref.dtype
        mesh = serialization.SerializedSample.mesh_from_json_dict(
            data=self.metadata.mesh_meta, device=device, dtype=dtype
        )
        sample_type = self.metadata.sample_type
        if sample_type == "SolidSampleExpandedStrain":
            return SolidSampleExpandedStrain(
                base_spin_system=base_spin_system,
                mesh=mesh,
                gauss=self.width.gauss,
                lorentz=self.width.lorentz,
                ham_strain=self.width.ham_strain
            )
        elif sample_type == "SolidSample":
            return SolidSample(
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
    def from_serialized_sample(cls,
            serialized_sample: serialization.SerializedSample) -> GraphSample:
        """
        Create a graph representation from a serialized sample
        It directly converts serialized sample to graph, avoiding the creation of the spin_system

        :param serialized_sample: The serialized sample to convert.
        :return: A new GraphSpinSystemRepresentation instance.
        """
        return cls(
            metadata=serialized_sample.metadata,
            width=serialized_sample.width,
            graph_spin_system=GraphSpinSystem.from_serialized_spin_system(serialized_sample.serialized_spin_system)
        )

    def to_serialized_sample(self) -> serialization.SerializedSample:
        """
        Reconstruct a serialized sample from the graph representation.
        :return: A new SerializedSample instance.
        """
        return serialization.SerializedSample(
            metadata=self.metadata,
            width=self.width,
            serialized_spin_system=GraphSpinSystem.to_serialized_spin_system(self.graph_spin_system)
        )

    def is_equivalent(self, other: GraphSample, rtol: float = 1e-5, atol: float = 1e-6) -> bool:
        if not isinstance(other, GraphSample):
            return False
        return all([
            graph_equivalence.are_graphs_equivalent(self.graph_spin_system, other.graph_spin_system, rtol, atol),
            self.width.is_equivalent(other.width, rtol, atol),
            self.metadata.is_equivalent(other.metadata, rtol, atol)
        ])
