import typing as tp
import torch

from . import serialization
from . import graph_representation


def _resolve_batch_dim(dim: int, tensor: torch.Tensor, protected_tail_dims: int, *, allow_end: bool = False) -> int:
    """
    :param dim: the dimension in batch
    :param protected_tail_dims: number of trailing system dimensions that must never be treated as batch dims
    :param allow_end: True for unsqueeze (dim may equal batch_ndim),
    """
    batch_ndim = tensor.ndim - protected_tail_dims
    if batch_ndim < 0:
        raise ValueError("Tensor has fewer dimensions than the protected system tail.")
    if dim < 0:
        dim += batch_ndim + (1 if allow_end else 0)

    upper = batch_ndim if allow_end else batch_ndim - 1
    if dim < 0 or dim > upper:
        raise ValueError(
            f"dim={dim} targets protected system dimensions; "
            f"valid batch dims are [0, {upper}]"
        )
    return dim


def _aggregate_optional_tensors(
        tensors: tp.List[tp.Optional[torch.Tensor]],
        op: tp.Callable,
        dim: int = -1,
        protected_tail_dims: int = 0,
        **kwargs
) -> tp.Optional[torch.Tensor]:
    """Aggregate a list of optional tensors using the provided operation.

    :param tensors: List of tensors to aggregate.
    :param op: Aggregation operation (e.g., torch.cat, torch.stack).
    :param dim: Conceptual dimension to aggregate along. Default is -1.
    :param protected_tail_dims: The number of last dimensions that should be protected
    from transformation (e.g., 'dim', 'start_dim', 'end_dim')
    :param kwargs: Additional arguments passed to `op`.
    :return: Aggregated tensor, or None if all inputs are N one.
    """
    if all(t is None for t in tensors):
        return None
    if any(t is None for t in tensors):
        raise ValueError("All tensors must have the same None/non-None status for aggregation.")
    actual_dim = _resolve_batch_dim(
                dim, tensors[0], protected_tail_dims
            )
    return op(tensors, dim=actual_dim, **kwargs)


def _transform_optional_tensor(
        tensor: tp.Optional[torch.Tensor],
        op: tp.Callable,
        dim: tp.Optional[int] = None,
        protected_tail_dims: int = 0,
        allow_end: bool = False,
        **kwargs
) -> tp.Optional[torch.Tensor]:
    """Transform an optional tensor using the provided operation.

    :param tensor: Tensor to transform.
    :param op: Transformation operation (e.g., torch.flatten, tensor.expand).
    :param dim: Conceptual dimension to transform along. Default is -1.
    :param protected_tail_dims: The number of last dimensions that should be protected
    from transformation (e.g., 'dim', 'start_dim', 'end_dim')
    to get the actual tensor dimension. Default is 0.
    :param kwargs: Additional arguments passed to `op`.
    :return: Transformed tensor, or None if input is None.
    """

    if tensor is None:
        return None

    shifted_kwargs = {}
    for k, v in kwargs.items():
        if k in ("dim", "start_dim", "end_dim", "dim0", "dim1") and isinstance(v, int):
            shifted_kwargs[k] = _resolve_batch_dim(
                v, tensor, protected_tail_dims, allow_end=allow_end
            )
        elif k == "sizes":
            tail_shape = tensor.shape[-protected_tail_dims:] if protected_tail_dims > 0 else ()
            full_shape = tuple(v) + tail_shape
            shifted_kwargs[k] = full_shape
        else:
            shifted_kwargs[k] = v
    if "dim" not in shifted_kwargs and dim is not None:
        shifted_kwargs["dim"] = _resolve_batch_dim(
            dim, tensor, protected_tail_dims, allow_end=allow_end
        )
    return op(tensor, **shifted_kwargs)


def aggregate_spin_system_meta_data(
        instances: tp.List[serialization.SpinSystemMetaData],
        op: tp.Callable = None,
        dim: int = -1,
        **kwargs
) -> serialization.SpinSystemMetaData:
    """Aggregate a list of SpinSystemMetaData instances.

    Since metadata describes structural layout, it is expected to be identical across all instances.
    This function verifies equivalence and returns the first instance.

    :param instances: List of SpinSystemMetaData instances.
    :param op: Unused, included for API consistency.
    :param dim: Unused, included for API consistency.
    :param kwargs: Unused, included for API consistency.
    :return: The first SpinSystemMetaData instance.
    """
    if not instances:
        raise ValueError("Cannot aggregate an empty list of SpinSystemMetaData")
    first = instances[0]
    for m in instances[1:]:
        if not first.is_equivalent(m):
            raise ValueError("All SpinSystemMetaData objects must be equivalent when aggregating.")
    return first


def transform_spin_system_meta_data(
        instance: serialization.SpinSystemMetaData,
        op: tp.Callable = None,
        dim: tp.Optional[int] = None,
        **kwargs
) -> serialization.SpinSystemMetaData:
    """Transform a SpinSystemMetaData instance.

    Metadata contains no tensors to transform. This function returns the instance unmodified.

    :param instance: SpinSystemMetaData instance.
    :param op: Unused, included for API consistency.
    :param dim: Unused, included for API consistency.
    :param kwargs: Unused, included for API consistency.
    :return: The unmodified SpinSystemMetaData instance.
    """
    return instance


def aggregate_spin_system_interactions(
        instances: tp.List[serialization.SpinSystemInteractions],
        op: tp.Callable,
        dim: int = -1,
        **kwargs
) -> serialization.SpinSystemInteractions:
    """Aggregate a list of SpinSystemInteractions instances.

    :param instances: List of SpinSystemInteractions instances.
    :param op: Aggregation operation (e.g., torch.cat, torch.stack).
    :param dim: Conceptual dimension to aggregate along. Default is -1.
    :param kwargs: Additional arguments passed to `op`.
    :return: A new aggregated SpinSystemInteractions instance.

    Note: `protected_tail_dims` is set to 2 to skip the last two system dimensions.
    """
    protected_tail_dims: int = 2

    def _agg(field_name: str):
        return _aggregate_optional_tensors([getattr(i, field_name) for i in instances], op, dim=dim,
                                           protected_tail_dims=protected_tail_dims, **kwargs)

    return serialization.SpinSystemInteractions(
        g_tensor_components=_agg("g_tensor_components"),
        g_tensor_orientations=_agg("g_tensor_orientations"),
        hyperfine_coupling_components=_agg("hyperfine_coupling_components"),
        hyperfine_coupling_orientations=_agg("hyperfine_coupling_orientations"),
        zfs_components=_agg("zfs_components"),
        dipolar_components=_agg("dipolar_components"),
        electron_electron_orientations=_agg("electron_electron_orientations"),
        nuclear_coupling_components=_agg("nuclear_coupling_components"),
        nuclear_coupling_orientations=_agg("nuclear_coupling_orientations"),
    )


def transform_spin_system_interactions(
        instance: serialization.SpinSystemInteractions,
        op: tp.Callable,
        dim: tp.Optional[int] = None,
        **kwargs
) -> serialization.SpinSystemInteractions:
    """Transform a SpinSystemInteractions instance.

    :param instance: SpinSystemInteractions instance.
    :param op: Transformation operation (e.g., torch.flatten).
    :param dim: Conceptual dimension to transform along.
    :param kwargs: Additional arguments passed to `op`.
    :return: A new transformed SpinSystemInteractions instance.

    Note: `protected_tail_dims` is set to 2 to skip the last two system dimensions.
    """
    protected_tail_dims: int = 2

    def _trans(field_name: str):
        return _transform_optional_tensor(getattr(instance, field_name), op, dim=dim,
                                          protected_tail_dims=protected_tail_dims, **kwargs)

    return serialization.SpinSystemInteractions(
        g_tensor_components=_trans("g_tensor_components"),
        g_tensor_orientations=_trans("g_tensor_orientations"),
        hyperfine_coupling_components=_trans("hyperfine_coupling_components"),
        hyperfine_coupling_orientations=_trans("hyperfine_coupling_orientations"),
        zfs_components=_trans("zfs_components"),
        dipolar_components=_trans("dipolar_components"),
        electron_electron_orientations=_trans("electron_electron_orientations"),
        nuclear_coupling_components=_trans("nuclear_coupling_components"),
        nuclear_coupling_orientations=_trans("nuclear_coupling_orientations"),
    )


def aggregate_spin_system_strains(
        instances: tp.List[serialization.SpinSystemStrains],
        op: tp.Callable,
        dim: int = -1,
        **kwargs
) -> serialization.SpinSystemStrains:
    """Aggregate a list of SpinSystemStrains instances.

    :param instances: List of SpinSystemStrains instances.
    :param op: Aggregation operation.
    :param dim: Conceptual dimension to aggregate along. Default is -1.
    :param kwargs: Additional arguments passed to `op`.
    :return: A new aggregated SpinSystemStrains instance.

    Note: `protected_tail_dims` is set to 2 to skip the last two system dimensions.
    """
    protected_tail_dims: int = 2

    def _agg(field_name: str):
        return _aggregate_optional_tensors([getattr(s, field_name) for s in instances], op, dim=dim,
                                           protected_tail_dims=protected_tail_dims, **kwargs)

    return serialization.SpinSystemStrains(
        g_tensor_strain=_agg("g_tensor_strain"),
        hyperfine_coupling_strain=_agg("hyperfine_coupling_strain"),
        zfs_strain=_agg("zfs_strain"),
        dipolar_strain=_agg("dipolar_strain"),
    )


def transform_spin_system_strains(
        instance: serialization.SpinSystemStrains,
        op: tp.Callable,
        dim: tp.Optional[int] = None,
        **kwargs
) -> serialization.SpinSystemStrains:
    """Transform a SpinSystemStrains instance.

    :param instance: SpinSystemStrains instance.
    :param op: Transformation operation.
    :param dim: Conceptual dimension to transform along.
    :param kwargs: Additional arguments passed to `op`.
    :return: A new transformed SpinSystemStrains instance.

    Note: `protected_tail_dims` is set to 2 to skip the last two system dimensions.
    """
    protected_tail_dims: int = 2

    def _trans(field_name: str):
        return _transform_optional_tensor(
            getattr(instance, field_name), op, dim=dim, protected_tail_dims=protected_tail_dims, **kwargs)

    return serialization.SpinSystemStrains(
        g_tensor_strain=_trans("g_tensor_strain"),
        hyperfine_coupling_strain=_trans("hyperfine_coupling_strain"),
        zfs_strain=_trans("zfs_strain"),
        dipolar_strain=_trans("dipolar_strain"),
    )


def aggregate_serialized_spin_system(
        instances: tp.List[serialization.SerializedSpinSystem],
        op: tp.Callable,
        dim: int = -1,
        **kwargs
) -> serialization.SerializedSpinSystem:
    """Aggregate a list of SerializedSpinSystem instances.

    :param instances: List of SerializedSpinSystem instances.
    :param op: Aggregation operation.
    :param dim: Conceptual dimension to aggregate along.
    :param kwargs: Additional arguments passed to `op`.
    :return: A new aggregated SerializedSpinSystem instance.

    Note: `protected_tail_dims` is set to 2 for interactions and strains.
    """
    protected_tail_dims: int = 2

    metadata = aggregate_spin_system_meta_data([s.metadata for s in instances], op=op, dim=dim, **kwargs)
    interactions = aggregate_spin_system_interactions([s.interactions for s in instances], op=op, dim=dim,
                                                      protected_tail_dims=protected_tail_dims, **kwargs)
    strains = [s.strain for s in instances]
    if all(s is None for s in strains):
        strain = None
    else:
        strain = aggregate_spin_system_strains(strains, op=op, dim=dim, protected_tail_dims=protected_tail_dims, **kwargs)

    return serialization.SerializedSpinSystem(metadata=metadata, interactions=interactions, strain=strain)


def transform_serialized_spin_system(
        instance: serialization.SerializedSpinSystem,
        op: tp.Callable,
        dim: tp.Optional[int] = None,
        **kwargs
) -> serialization.SerializedSpinSystem:
    """Transform a SerializedSpinSystem instance.

    :param instance: SerializedSpinSystem instance.
    :param op: Transformation operation.
    :param dim: Conceptual dimension to transform along. Default is None.
    :param kwargs: Additional arguments passed to `op`.
    :return: A new transformed SerializedSpinSystem instance.

    Note: `protected_tail_dims` is set to 2 for interactions and strains.
    """
    metadata = transform_spin_system_meta_data(instance.metadata, op=op, dim=dim, **kwargs)
    interactions = transform_spin_system_interactions(instance.interactions, op=op, dim=dim, **kwargs)
    strain = transform_spin_system_strains(
        instance.strain, op=op, dim=dim, **kwargs) if instance.strain is not None else None

    return serialization.SerializedSpinSystem(metadata=metadata, interactions=interactions, strain=strain)


def aggregate_serialized_sample_width(
        instances: tp.List[serialization.SerializedSampleWidth],
        op: tp.Callable,
        dim: int = -1,
        **kwargs
) -> serialization.SerializedSampleWidth:
    """Aggregate a list of SerializedSampleWidth instances.

    :param instances: List of SerializedSampleWidth instances.
    :param op: Aggregation operation.
    :param dim: Conceptual dimension to aggregate along. Default is -1.
    :param kwargs: Additional arguments passed to `op`.
    :return: A new aggregated SerializedSampleWidth instance.

    Note: `protected_tail_dims` is 1 for 'ham_strain' and 0 for 'gauss' and 'lorentz'.
    """
    def _agg(field_name: str, protected_tail_dims: int):
        return _aggregate_optional_tensors([getattr(w, field_name) for w in instances], op, dim=dim,
                                           protected_tail_dims=protected_tail_dims, **kwargs)

    return serialization.SerializedSampleWidth(
        ham_strain=_agg("ham_strain", 1),
        gauss=_agg("gauss", 0),
        lorentz=_agg("lorentz", 0),
    )


def transform_serialized_sample_width(
        instance: serialization.SerializedSampleWidth,
        op: tp.Callable,
        dim: tp.Optional[int] = None,
        **kwargs
) -> serialization.SerializedSampleWidth:
    """Transform a SerializedSampleWidth instance.

    :param instance: SerializedSampleWidth instance.
    :param op: Transformation operation.
    :param dim: Conceptual dimension to transform along.
    :param kwargs: Additional arguments passed to `op`.
    :return: A new transformed SerializedSampleWidth instance.

    Note: `protected_tail_dims` is 1 for 'ham_strain' and 0 for 'gauss' and 'lorentz'.
    """
    def _trans(field_name: str, protected_tail_dims: int):
        return _transform_optional_tensor(
            getattr(instance, field_name), op, dim=dim, protected_tail_dims=protected_tail_dims, **kwargs)

    return serialization.SerializedSampleWidth(
        ham_strain=_trans("ham_strain", 1),
        gauss=_trans("gauss", 0),
        lorentz=_trans("lorentz", 0),
    )


def aggregate_sample_meta_data(
        instances: tp.List[serialization.SampleMetaData],
        op: tp.Callable = None,
        dim: int = -1,
        **kwargs
) -> serialization.SampleMetaData:
    """Aggregate a list of SampleMetaData instances.

    Returns the first instance. Metadata is assumed to be identical across the batch.

    :param instances: List of SampleMetaData instances.
    :param op: Unused, included for API consistency.
    :param dim: Unused, included for API consistency.
    :param kwargs: Unused, included for API consistency.
    :return: The first SampleMetaData instance.
    """
    if not instances:
        raise ValueError("Cannot aggregate an empty list of SampleMetaData")
    first = instances[0]
    return first


def transform_sample_meta_data(
        instance: serialization.SampleMetaData,
        op: tp.Callable = None,
        dim: tp.Optional[int] = None,
        **kwargs
) -> serialization.SampleMetaData:
    """Transform a SampleMetaData instance.

    Returns the instance unmodified, as it contains no tensors.

    :param instance: SampleMetaData instance.
    :param op: Unused, included for API consistency.
    :param dim: Unused, included for API consistency.
    :param kwargs: Unused, included for API consistency.
    :return: The unmodified SampleMetaData instance.
    """
    return instance


def aggregate_serialized_sample(
        instances: tp.List[serialization.SerializedSample],
        op: tp.Callable,
        dim: int = -1,
        **kwargs
) -> serialization.SerializedSample:
    """Aggregate a list of SerializedSample instances.

    :param instances: List of SerializedSample instances.
    :param op: Aggregation operation.
    :param dim: Conceptual dimension to aggregate along. Default is -1.
    :param kwargs: Additional arguments passed to `op`.
    :return: A new aggregated SerializedSample instance.

    Note: `protected_tail_dims` is 2 for the spin system and passed to width aggregation.
    """
    metadata = aggregate_sample_meta_data([s.metadata for s in instances], op=op, dim=dim, **kwargs)
    serialized_spin_system = aggregate_serialized_spin_system([s.serialized_spin_system for s in instances], op=op,
                                                              dim=dim, **kwargs)
    width = aggregate_serialized_sample_width([s.width for s in instances], op=op, dim=dim,
                                              **kwargs)

    return serialization.SerializedSample(
        metadata=metadata,
        serialized_spin_system=serialized_spin_system,
        width=width
    )


def transform_serialized_sample(
        instance: serialization.SerializedSample,
        op: tp.Callable,
        dim: tp.Optional[int] = None,
        **kwargs
) -> serialization.SerializedSample:
    """Transform a SerializedSample instance.

    :param instance: SerializedSample instance.
    :param op: Transformation operation.
    :param dim: Conceptual dimension to transform along.
    :param kwargs: Additional arguments passed to `op`.
    :return: A new transformed SerializedSample instance.

    Note: `protected_tail_dims` is 2 for the spin system and passed to width transformation.
    """
    metadata = transform_sample_meta_data(instance.metadata, op=op, dim=dim, **kwargs)
    serialized_spin_system = transform_serialized_spin_system(instance.serialized_spin_system, op=op, dim=dim, **kwargs)
    width = transform_serialized_sample_width(instance.width, op=op, dim=dim, **kwargs)
    return serialization.SerializedSample(
        metadata=metadata,
        serialized_spin_system=serialized_spin_system,
        width=width
    )


def aggregate_graph_spin_system(
        instances: tp.List[graph_representation.GraphSpinSystem],
        op: tp.Callable,
        dim: int = -1,
        **kwargs
) -> graph_representation.GraphSpinSystem:
    """Aggregate a list of GraphSpinSystem instances.

    :param instances: List of GraphSpinSystem instances.
    :param op: Aggregation operation.
    :param dim: Conceptual dimension to aggregate along. Default is -1.
    :param kwargs: Additional arguments passed to `op`.
    :return: A new aggregated GraphSpinSystem instance.

    Note: 'source' and 'destination' are copied from the first instance. `protected_tail_dims` is 2 for node features.
    """
    source = instances[0].source
    destination = instances[0].destination

    def _agg(field_name: str, protected_tail_dims: int):
        return _aggregate_optional_tensors([getattr(s, field_name) for s in instances], op, dim=dim,
                                           protected_tail_dims=protected_tail_dims, **kwargs)

    return graph_representation.GraphSpinSystem(
        source=source,
        destination=destination,
        components=_agg("components", 2),
        angles=_agg("angles", 2),
        spins=_agg("spins", 1),
        node_types=_agg("node_types", 1),
    )


def transform_graph_spin_system(
        instance: graph_representation.GraphSpinSystem,
        op: tp.Callable,
        dim: tp.Optional[int] = None,
        **kwargs
) -> graph_representation.GraphSpinSystem:
    """Transform a GraphSpinSystem instance.

    :param instance: GraphSpinSystem instance.
    :param op: Transformation operation.
    :param dim: Conceptual dimension to transform along.
    :param kwargs: Additional arguments passed to `op`.
    :return: A new transformed GraphSpinSystem instance.

    Note: 'source' and 'destination' are unmodified. `protected_tail_dims` is 2 for node features.
    """

    def _trans(field_name: str, protected_tail_dims: int):
        return _transform_optional_tensor(
            getattr(instance, field_name), op, dim=dim, protected_tail_dims=protected_tail_dims, **kwargs)

    return graph_representation.GraphSpinSystem(
        source=instance.source,
        destination=instance.destination,
        components=_trans("components", 2),
        angles=_trans("angles", 2),
        spins=_trans("spins", 1),
        node_types=_trans("node_types", 1),
    )


def aggregate_graph_sample(
        instances: tp.List[graph_representation.GraphSample],
        op: tp.Callable,
        dim: int = -1,
        **kwargs
) -> graph_representation.GraphSample:
    """Aggregate a list of GraphSample instances.

    :param instances: List of GraphSample instances.
    :param op: Aggregation operation.
    :param dim: Conceptual dimension to aggregate along. Default is -1.
    :param kwargs: Additional arguments passed to `op`.
    :return: A new aggregated GraphSample instance.

    Note: `protected_tail_dims` is 2 for the graph spin system and passed to width aggregation.
    """
    metadata = aggregate_sample_meta_data([s.metadata for s in instances], op=op, dim=dim, **kwargs)
    graph_spin_system = aggregate_graph_spin_system([s.graph_spin_system for s in instances], op=op, dim=dim,
                                                    **kwargs)
    width = aggregate_serialized_sample_width([s.width for s in instances], op=op, dim=dim, **kwargs)

    return graph_representation.GraphSample(
        metadata=metadata,
        graph_spin_system=graph_spin_system,
        width=width
    )


def transform_graph_sample(
        instance: graph_representation.GraphSample,
        op: tp.Callable,
        dim: tp.Optional[int] = None,
        **kwargs
) -> graph_representation.GraphSample:
    """Transform a GraphSample instance.

    :param instance: GraphSample instance.
    :param op: Transformation operation.
    :param dim: Conceptual dimension to transform along.
    :param kwargs: Additional arguments passed to `op`.
    :return: A new transformed GraphSample instance.

    Note: `protected_tail_dims` is 2 for the graph spin system and passed to width transformation.
    """
    metadata = transform_sample_meta_data(instance.metadata, op=op, dim=dim, **kwargs)
    graph_spin_system = transform_graph_spin_system(instance.graph_spin_system, op=op, dim=dim,
                                                    **kwargs)
    width = transform_serialized_sample_width(instance.width, op=op, dim=dim, **kwargs)

    return graph_representation.GraphSample(
        metadata=metadata,
        graph_spin_system=graph_spin_system,
        width=width
    )


def aggregate_polarization_parameters(
        instances: tp.List[serialization.PolarizationParameters],
        op: tp.Callable,
        dim: int = -1,
        **kwargs
) -> serialization.PolarizationParameters:
    """Aggregate a list of PolarizationParameters instances.

    :param instances: List of PolarizationParameters instances.
    :param op: Aggregation operation.
    :param dim: Conceptual dimension to aggregate along. Default is -1.
    :param kwargs: Additional arguments passed to `op`.
    :return: A new aggregated PolarizationParameters instance.

    Note: `protected_tail_dims` is 0 for 'temperature' and -1 for 'initial_populations'.
    """
    if not instances:
        raise ValueError("Cannot aggregate an empty list of PolarizationParameters")

    temp_tensor = _aggregate_optional_tensors(
        [p.temperature for p in instances], op, dim=dim, protected_tail_dims=0, **kwargs
    )

    bases = [p.basis for p in instances]
    if any(b != bases[0] for b in bases):
        raise ValueError("All PolarizationParameters must have the same 'basis' when aggregating.")

    init_pop_tensor = _aggregate_optional_tensors(
        [p.initial_populations for p in instances], op, dim=dim, protected_tail_dims=1, **kwargs
    )

    return serialization.PolarizationParameters(
        temperature=temp_tensor,
        basis=bases[0],
        initial_populations=init_pop_tensor
    )


def transform_polarization_parameters(
        instance: serialization.PolarizationParameters,
        op: tp.Callable,
        dim: tp.Optional[int] = None,
        **kwargs
) -> serialization.PolarizationParameters:
    """Transform a PolarizationParameters instance.

    :param instance: PolarizationParameters instance.
    :param op: Transformation operation.
    :param dim: Conceptual dimension to transform along.
    :param kwargs: Additional arguments passed to `op`.
    :return: A new transformed PolarizationParameters instance.

    Note: `protected_tail_dims` is 0 for 'temperature' and -1 for 'initial_populations'.
    """
    temp_tensor = _transform_optional_tensor(
        instance.temperature, op, dim=dim, protected_tail_dims=0, **kwargs
    )

    init_pop_tensor = _transform_optional_tensor(
        instance.initial_populations, op, dim=dim, protected_tail_dims=1, **kwargs
    )

    return serialization.PolarizationParameters(
        temperature=temp_tensor,
        basis=instance.basis,
        initial_populations=init_pop_tensor
    )


def aggregate_time_parameters(
        instances: tp.List[serialization.TimeParameters],
        op: tp.Callable,
        dim: int = -1,
        **kwargs
) -> serialization.TimeParameters:
    """Aggregate a list of TimeParameters instances.

    :param instances: List of TimeParameters instances.
    :param op: Aggregation operation.
    :param dim: Conceptual dimension to aggregate along. Default is -1.
    :param kwargs: Additional arguments passed to `op`.
    :return: A new aggregated TimeParameters instance.

    Note: `protected_tail_dims` is 1 for 'min_time' and 'max_time'.
    """
    if not instances:
        raise ValueError("Cannot aggregate an empty list of TimeParameters")

    num_points_list = [p.num_points for p in instances]
    if any(n != num_points_list[0] for n in num_points_list):
        raise ValueError("All TimeParameters must have the same 'num_points' when aggregating.")

    min_time = _aggregate_optional_tensors(
        [p.min_time for p in instances], op, dim=dim, protected_tail_dims=1, **kwargs
    )
    max_time = _aggregate_optional_tensors(
        [p.max_time for p in instances], op, dim=dim, protected_tail_dims=1, **kwargs
    )

    return serialization.TimeParameters(
        min_time=min_time,
        max_time=max_time,
        num_points=num_points_list[0]
    )


def transform_time_parameters(
        instance: serialization.TimeParameters,
        op: tp.Callable,
        dim: tp.Optional[int] = None,
        **kwargs
) -> serialization.TimeParameters:
    """Transform a TimeParameters instance.

    :param instance: TimeParameters instance.
    :param op: Transformation operation.
    :param dim: Conceptual dimension to transform along.
    :param kwargs: Additional arguments passed to `op`.
    :return: A new transformed TimeParameters instance.

    Note: `protected_tail_dims` is 1 for 'min_time' and 'max_time'.
    """
    min_time = _transform_optional_tensor(
        instance.min_time, op, dim=dim, protected_tail_dims=1, **kwargs
    )
    max_time = _transform_optional_tensor(
        instance.max_time, op, dim=dim, protected_tail_dims=1, **kwargs
    )

    return serialization.TimeParameters(
        min_time=min_time,
        max_time=max_time,
        num_points=instance.num_points
    )


def aggregate_experimental_parameters(
        instances: tp.List[serialization.ExperimentalParameters],
        op: tp.Callable,
        dim: int = -1,
        **kwargs
) -> serialization.ExperimentalParameters:
    """Aggregate a list of ExperimentalParameters instances.

    :param instances: List of ExperimentalParameters instances.
    :param op: Aggregation operation.
    :param dim: Conceptual dimension to aggregate along. Default is -1.
    :param kwargs: Additional arguments passed to `op`.
    :return: A new aggregated ExperimentalParameters instance.

    Note: `protected_tail_dims` is 1 for 'min_pos', 'max_pos', and 'resonance_parameter'.
    """
    if not instances:
        raise ValueError("Cannot aggregate an empty list of ExperimentalParameters")

    num_points_list = [p.num_points for p in instances]
    if any(n != num_points_list[0] for n in num_points_list):
        raise ValueError("All ExperimentalParameters must have the same 'num_points' when aggregating.")

    min_pos = _aggregate_optional_tensors(
        [p.min_pos for p in instances], op, dim=dim, protected_tail_dims=0, **kwargs
    )
    max_pos = _aggregate_optional_tensors(
        [p.max_pos for p in instances], op, dim=dim, protected_tail_dims=0, **kwargs
    )
    resonance_parameter = _aggregate_optional_tensors(
        [p.resonance_parameter for p in instances], op, dim=dim, protected_tail_dims=0, **kwargs
    )

    position_shift = _aggregate_optional_tensors(
        [p.position_shift for p in instances], op, dim=dim, protected_tail_dims=0, **kwargs
    )

    time_params_list = [p.time_params for p in instances]
    if all(t is None for t in time_params_list):
        time_params = None
    elif any(t is None for t in time_params_list):
        raise ValueError("All ExperimentalParameters must have the same None/non-None status for time_params")
    else:
        time_params = aggregate_time_parameters(time_params_list, op=op, dim=dim, **kwargs)

    return serialization.ExperimentalParameters(
        min_pos=min_pos,
        max_pos=max_pos,
        num_points=num_points_list[0],
        resonance_parameter=resonance_parameter,
        time_params=time_params,
        position_shift=position_shift
    )


def transform_experimental_parameters(
        instance: serialization.ExperimentalParameters,
        op: tp.Callable,
        dim: tp.Optional[int] = None,
        **kwargs
) -> serialization.ExperimentalParameters:
    """Transform an ExperimentalParameters instance.

    :param instance: ExperimentalParameters instance.
    :param op: Transformation operation.
    :param dim: Conceptual dimension to transform along.
    :param kwargs: Additional arguments passed to `op`.
    :return: A new transformed ExperimentalParameters instance.
    Note: `protected_tail_dims` is 0 for 'min_pos', 'max_pos', and 'resonance_parameter'.
    """
    min_pos = _transform_optional_tensor(
        instance.min_pos, op, dim=dim, protected_tail_dims=0, **kwargs
    )
    max_pos = _transform_optional_tensor(
        instance.max_pos, op, dim=dim, protected_tail_dims=0, **kwargs
    )
    resonance_parameter = _transform_optional_tensor(
        instance.resonance_parameter, op, dim=dim, protected_tail_dims=0, **kwargs
    )

    position_shift = _transform_optional_tensor(
        instance.position_shift, op, dim=dim, protected_tail_dims=0, **kwargs
    )

    time_params = transform_time_parameters(instance.time_params, op=op, dim=dim,
                                            **kwargs) if instance.time_params is not None else None

    return serialization.ExperimentalParameters(
        min_pos=min_pos,
        max_pos=max_pos,
        num_points=instance.num_points,
        resonance_parameter=resonance_parameter,
        time_params=time_params,
        position_shift=position_shift
    )


def aggregate_cw_spectral_data(
        instances: tp.List[serialization.CWSpectralData],
        op: tp.Callable,
        dim: int = -1,
        **kwargs
) -> serialization.CWSpectralData:
    """Aggregate a list of CWSpectralData instances.

    :param instances: List of CWSpectralDatas instances.
    :param op: Aggregation operation.
    :param dim: Conceptual dimension to aggregate along. Default is -1.
    :param kwargs: Additional arguments passed to `op`.
    :return: A new aggregated ExperimentalParameters instance.

    Note: `protected_tail_dims` is 0 for 'min_pos', 'max_pos', and 'resonance_parameter'.
    """
    experimental_parameters = aggregate_experimental_parameters(
        [instance.experimental_parameters for instance in instances], op=op, dim=dim, **kwargs
    )

    spectrum = _aggregate_optional_tensors(
        [instance.spectrum for instance in instances], op=op, dim=dim, protected_tail_dims=1, **kwargs
    )

    return serialization.CWSpectralData(
        experimental_parameters=experimental_parameters,
        spectrum=spectrum,
    )


def transform_cw_spectral_data(
        instance: serialization.CWSpectralData,
        op: tp.Callable,
        dim: tp.Optional[int] = None,
        **kwargs
) -> serialization.CWSpectralData:
    """Transform an ExperimentalParameters instance.

    :param instance: CWSpectralData instance.
    :param op: Transformation operation.
    :param dim: Conceptual dimension to transform along. Default is -1.
    :param kwargs: Additional arguments passed to `op`.
    :return: A new transformed ExperimentalParameters instance.
    Note: `protected_tail_dims` is 0 for 'min_pos', 'max_pos', and 'resonance_parameter'. For spectrum is set as 1
    """
    experimental_parameters = transform_experimental_parameters(
        instance.experimental_parameters, op=op, dim=dim, **kwargs
    )
    spectrum = _transform_optional_tensor(instance.spectrum, op=op, dim=dim, protected_tail_dims=1, **kwargs)

    return serialization.CWSpectralData(
        experimental_parameters=experimental_parameters,
        spectrum=spectrum,
    )
