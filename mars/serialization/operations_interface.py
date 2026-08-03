import typing as tp
import torch

from . import serialization
from . import graph_representation
from . import operations_general


serialized_items_types = tp.Union[
    tp.Sequence[serialization.SerializedSpinSystem],
    tp.Sequence[serialization.SerializedSample],
    tp.Sequence[serialization.SerializedSampleWidth],

    tp.Sequence[graph_representation.GraphSpinSystem],
    tp.Sequence[graph_representation.GraphSample],
    tp.Sequence[serialization.ExperimentalParameters],
    tp.Sequence[serialization.PolarizationParameters],
    tp.Sequence[serialization.TimeParameters],
    serialization.CWSpectralData
]

serialized_item_type = tp.Union[
    serialization.SerializedSpinSystem,
    serialization.SerializedSample,
    serialization.SerializedSampleWidth,
    graph_representation.GraphSpinSystem,
    graph_representation.GraphSample,
    serialization.ExperimentalParameters,
    serialization.PolarizationParameters,
    serialization.TimeParameters,
    serialization.CWSpectralData,
]


def _get_aggregate_func(ref_item: serialized_item_type) -> tp.Callable[..., serialized_item_type]:
    """Returns the appropriate aggregation function from operations_general based on item type."""
    if isinstance(ref_item, serialization.SerializedSpinSystem):
        return operations_general.aggregate_serialized_spin_system
    elif isinstance(ref_item, serialization.SerializedSample):
        return operations_general.aggregate_serialized_sample
    elif isinstance(ref_item, serialization.SerializedSampleWidth):
        return operations_general.aggregate_serialized_sample_width
    elif isinstance(ref_item, graph_representation.GraphSpinSystem):
        return operations_general.aggregate_graph_spin_system
    elif isinstance(ref_item, graph_representation.GraphSample):
        return operations_general.aggregate_graph_sample
    elif isinstance(ref_item, serialization.ExperimentalParameters):
        return operations_general.aggregate_experimental_parameters
    elif isinstance(ref_item, serialization.PolarizationParameters):
        return operations_general.aggregate_polarization_parameters
    elif isinstance(ref_item, serialization.TimeParameters):
        return operations_general.aggregate_time_parameters
    elif isinstance(ref_item, serialization.CWSpectralData):
        return operations_general.aggregate_cw_spectral_data
    else:
        raise TypeError(f"Unsupported item type for aggregation: {type(ref_item)}")


def _get_transform_func(ref_item: serialized_item_type) -> tp.Callable[..., serialized_item_type]:
    """Returns the appropriate transformation function from operations_general based on item type."""
    if isinstance(ref_item, serialization.SerializedSpinSystem):
        return operations_general.transform_serialized_spin_system
    elif isinstance(ref_item, serialization.SerializedSample):
        return operations_general.transform_serialized_sample
    elif isinstance(ref_item, serialization.SerializedSampleWidth):
        return operations_general.transform_serialized_sample_width
    elif isinstance(ref_item, graph_representation.GraphSpinSystem):
        return operations_general.transform_graph_spin_system
    elif isinstance(ref_item, graph_representation.GraphSample):
        return operations_general.transform_graph_sample
    elif isinstance(ref_item, serialization.ExperimentalParameters):
        return operations_general.transform_experimental_parameters
    elif isinstance(ref_item, serialization.PolarizationParameters):
        return operations_general.transform_polarization_parameters
    elif isinstance(ref_item, serialization.TimeParameters):
        return operations_general.transform_time_parameters
    elif isinstance(ref_item, serialization.CWSpectralData):
        return operations_general.transform_cw_spectral_data
    else:
        raise TypeError(f"Unsupported item type for transformation: {type(ref_item)}")


@tp.overload
def concat(items: tp.Sequence[serialization.SerializedSpinSystem],
           mode: tp.Literal["direct_sum", "batch"] = "direct_sum",
           dim: int = -1) -> serialization.SerializedSpinSystem: ...
@tp.overload
def concat(items: tp.Sequence[serialization.SerializedSample],
           mode: tp.Literal["direct_sum", "batch"] = "direct_sum",
           dim: int = -1) -> serialization.SerializedSample: ...
@tp.overload
def concat(items: tp.Sequence[graph_representation.GraphSpinSystem],
           mode: tp.Literal["direct_sum", "batch"] = "direct_sum",
           dim: int = -1) -> graph_representation.GraphSpinSystem: ...
@tp.overload
def concat(items: tp.Sequence[graph_representation.GraphSample],
           mode: tp.Literal["direct_sum", "batch"] = "direct_sum",
           dim: int = -1) -> graph_representation.GraphSample: ...
@tp.overload
def concat(items: tp.Sequence[serialization.ExperimentalParameters],
           mode: tp.Literal["direct_sum", "batch"] = "direct_sum",
           dim: int = -1) -> serialization.ExperimentalParameters: ...
@tp.overload
def concat(items: tp.Sequence[serialization.PolarizationParameters],
           mode: tp.Literal["direct_sum", "batch"] = "direct_sum",
           dim: int = -1) -> serialization.PolarizationParameters: ...
@tp.overload
def concat(items: tp.Sequence[serialization.TimeParameters],
           mode: tp.Literal["direct_sum", "batch"] = "direct_sum",
           dim: int = -1) -> serialization.TimeParameters: ...
@tp.overload
def concat(items: tp.Sequence[serialization.CWSpectralData],
           mode: tp.Literal["direct_sum", "batch"] = "direct_sum",
           dim: int = -1) -> serialization.CWSpectralData: ...
@tp.overload
def concat(items: tp.Sequence[serialization.SerializedSampleWidth],
           mode: tp.Literal["direct_sum", "batch"] = "direct_sum",
           dim: int = -1) -> serialization.SerializedSampleWidth: ...


def concat(
        items: serialized_items_types,
        mode: tp.Literal["direct_sum", "batch"] = "direct_sum",
        dim: int = -1,
) -> serialized_item_type:
    """
    Concatenate a sequence of homogeneous serialized or graph MaRs items along an existing batch dimension.

    :param items: A non-empty sequence of identical-type serialized or graph MARS objects.
    :param mode: The concatenation mode. Only ``"batch"`` is supported for serialized/graph items.
                 If ``"direct_sum"`` is passed, a NotImplementedError is raised.
    :param dim: The dimension along which to concatenate. Default is -1.
    :return: A single serialized or graph MARS object.
    :raises NotImplementedError: If ``mode`` is not ``"batch"``.
    :raises TypeError: If the item type is not supported.
    :raises ValueError: If the sequence is empty.
    """
    if mode != "batch":
        raise NotImplementedError(
            "Serialized and graph items only support 'batch' mode. "
            "For 'direct_sum', convert to live MARS objects first."
        )

    if not items:
        raise ValueError("Cannot concatenate an empty sequence.")

    ref_item = items[0]
    agg_func = _get_aggregate_func(ref_item)
    return agg_func(list(items), op=torch.cat, dim=dim)


@tp.overload
def stack(items: tp.Sequence[serialization.SerializedSpinSystem],
          dim: int = -1) -> serialization.SerializedSpinSystem: ...
@tp.overload
def stack(items: tp.Sequence[serialization.SerializedSample], dim: int = -1) -> serialization.SerializedSample: ...
@tp.overload
def stack(items: tp.Sequence[graph_representation.GraphSpinSystem],
          dim: int = -1) -> graph_representation.GraphSpinSystem: ...
@tp.overload
def stack(items: tp.Sequence[graph_representation.GraphSample],
          dim: int = -1) -> graph_representation.GraphSample: ...
@tp.overload
def stack(items: tp.Sequence[serialization.ExperimentalParameters],
          dim: int = -1) -> serialization.ExperimentalParameters: ...
@tp.overload
def stack(items: tp.Sequence[serialization.PolarizationParameters],
          dim: int = -1) -> serialization.PolarizationParameters: ...
@tp.overload
def stack(items: tp.Sequence[serialization.TimeParameters],
          dim: int = -1) -> serialization.TimeParameters: ...
@tp.overload
def stack(items: tp.Sequence[serialization.CWSpectralData],
          dim: int = -1) -> serialization.CWSpectralData: ...
@tp.overload
def stack(items: tp.Sequence[serialization.SerializedSampleWidth],
          dim: int = -1) -> serialization.SerializedSampleWidth: ...


def stack(
        items: serialized_items_types,
        dim: int = -1,
) -> serialized_item_type:
    """
    Stack a sequence of homogeneous serialized or graph MaRs items along a new dimension.

    :param items: A non-empty sequence of identical-type serialized or graph MARS objects.
    :param dim: The dimension along which to stack. Default is -1.
    :return: A single serialized or graph MARS object with a new stacked dimension.
    :raises TypeError: If the item type is not supported.
    :raises ValueError: If the sequence is empty.
    """
    if not items:
        raise ValueError("Cannot stack an empty sequence.")

    ref_item = items[0]
    agg_func = _get_aggregate_func(ref_item)
    return agg_func(list(items), op=torch.stack, dim=dim)


@tp.overload
def flatten(item: serialization.SerializedSpinSystem, start_dim: int = 0,
            end_dim: int = -1) -> serialization.SerializedSpinSystem: ...
@tp.overload
def flatten(item: serialization.SerializedSample, start_dim: int = 0,
            end_dim: int = -1) -> serialization.SerializedSample: ...
@tp.overload
def flatten(item: graph_representation.GraphSpinSystem, start_dim: int = 0,
            end_dim: int = -1) -> graph_representation.GraphSpinSystem: ...
@tp.overload
def flatten(item: graph_representation.GraphSample, start_dim: int = 0,
            end_dim: int = -1) -> graph_representation.GraphSample: ...
@tp.overload
def flatten(item: serialization.ExperimentalParameters, start_dim: int = 0,
            end_dim: int = -1) -> serialization.ExperimentalParameters: ...
@tp.overload
def flatten(item: serialization.PolarizationParameters, start_dim: int = 0,
            end_dim: int = -1) -> serialization.PolarizationParameters: ...
@tp.overload
def flatten(item: serialization.TimeParameters, start_dim: int = 0,
            end_dim: int = -1) -> serialization.TimeParameters: ...
@tp.overload
def flatten(item: serialization.CWSpectralData, start_dim: int = 0,
            end_dim: int = -1) -> serialization.CWSpectralData: ...
@tp.overload
def flatten(item: serialization.SerializedSampleWidth, start_dim: int = 0,
            end_dim: int = -1) -> serialization.SerializedSampleWidth: ...


def flatten(
        item: serialized_item_type,
        start_dim: int = 0,
        end_dim: int = -1,
) -> serialized_item_type:
    """
    Flatten batch dimensions of a serialized or graph MaRs item.

    :param item: A single serialized or graph MARS object.
    :param start_dim: The first dimension to flatten. Default is 0.
    :param end_dim: The last dimension to flatten. Default is -1, where -1 is the last dimension of batch
    :return: A new object with flattened batch dimensions.
    """
    trans_func = _get_transform_func(item)
    return trans_func(item, op=torch.flatten, start_dim=start_dim, end_dim=end_dim)


@tp.overload
def expand(item: serialization.SerializedSpinSystem,
           sizes: tp.Union[torch.Size, tp.List[int]]) -> serialization.SerializedSpinSystem: ...
@tp.overload
def expand(item: serialization.SerializedSample,
           sizes: tp.Union[torch.Size, tp.List[int]]) -> serialization.SerializedSample: ...
@tp.overload
def expand(item: graph_representation.GraphSpinSystem,
           sizes: tp.Union[torch.Size, tp.List[int]]) -> graph_representation.GraphSpinSystem: ...
@tp.overload
def expand(item: graph_representation.GraphSample,
           sizes: tp.Union[torch.Size, tp.List[int]]) -> graph_representation.GraphSample: ...
@tp.overload
def expand(item: serialization.ExperimentalParameters,
           sizes: tp.Union[torch.Size, tp.List[int]]) -> serialization.ExperimentalParameters: ...
@tp.overload
def expand(item: serialization.PolarizationParameters,
           sizes: tp.Union[torch.Size, tp.List[int]]) -> serialization.PolarizationParameters: ...
@tp.overload
def expand(item: serialization.TimeParameters,
           sizes: tp.Union[torch.Size, tp.List[int]]) -> serialization.TimeParameters: ...
@tp.overload
def expand(item: serialization.CWSpectralData,
           sizes: tp.Union[torch.Size, tp.List[int]]) -> serialization.CWSpectralData: ...
@tp.overload
def expand(item: serialization.SerializedSampleWidth,
           sizes: tp.Union[torch.Size ,tp.List[int]]) -> serialization.SerializedSampleWidth: ...


def expand(
        item: serialized_item_type,
        sizes: tp.Union[torch.Size, tp.List[int]],
) -> serialized_item_type:
    """
    Expand batch dimensions of a serialized or graph MaRs item.

    :param item: A single serialized or graph MARS object.
    :param sizes: The desired expanded sizes.
    :return: A new object with expanded batch dimensions.
    """
    trans_func = _get_transform_func(item)
    return trans_func(item, op=lambda t, sizes: t.expand(sizes), sizes=sizes)


@tp.overload
def repeat(item: serialization.SerializedSpinSystem,
           repeats: tp.Union[torch.Size, tp.List[int]]) -> serialization.SerializedSpinSystem: ...
@tp.overload
def repeat(item: serialization.SerializedSample,
           repeats: tp.Union[torch.Size, tp.List[int]]) -> serialization.SerializedSample: ...
@tp.overload
def repeat(item: graph_representation.GraphSpinSystem,
           repeats: tp.Union[torch.Size, tp.List[int]]) -> graph_representation.GraphSpinSystem: ...
@tp.overload
def repeat(item: graph_representation.GraphSample,
           repeats: tp.Union[torch.Size, tp.List[int]]) -> graph_representation.GraphSample: ...
@tp.overload
def repeat(item: serialization.ExperimentalParameters,
           repeats: tp.Union[torch.Size, tp.List[int]]) -> serialization.ExperimentalParameters: ...
@tp.overload
def repeat(item: serialization.PolarizationParameters,
           repeats: tp.Union[torch.Size, tp.List[int]]) -> serialization.PolarizationParameters: ...
@tp.overload
def repeat(item: serialization.TimeParameters,
           repeats: tp.Union[torch.Size, tp.List[int]]) -> serialization.TimeParameters: ...
@tp.overload
def repeat(item: serialization.CWSpectralData,
           repeats: tp.Union[torch.Size, tp.List[int]]) -> serialization.CWSpectralData: ...
@tp.overload
def repeat(item: serialization.SerializedSampleWidth,
           repeats: tp.Union[torch.Size, tp.List[int]]) -> serialization.SerializedSampleWidth: ...


def repeat(
        item: serialized_item_type,
        repeats: tp.Union[torch.Size, tp.List[int]],
) -> serialized_item_type:
    """
    Repeat batch dimensions of a serialized or graph MaRs item.

    :param item: A single serialized or graph MARS object.
    :param repeats: The number of times to repeat each dimension.
    :return: A new object with repeated batch dimensions.
    """
    trans_func = _get_transform_func(item)
    return trans_func(item, op=lambda t, repeats: t.repeat(repeats), repeats=repeats)


@tp.overload
def unsqueeze(item: serialization.SerializedSpinSystem, dim: int) -> serialization.SerializedSpinSystem: ...
@tp.overload
def unsqueeze(item: serialization.SerializedSample, dim: int) -> serialization.SerializedSample: ...
@tp.overload
def unsqueeze(item: graph_representation.GraphSpinSystem, dim: int) -> graph_representation.GraphSpinSystem: ...
@tp.overload
def unsqueeze(item: graph_representation.GraphSample, dim: int) -> graph_representation.GraphSample: ...
@tp.overload
def unsqueeze(item: serialization.ExperimentalParameters, dim: int) -> serialization.ExperimentalParameters: ...
@tp.overload
def unsqueeze(item: serialization.PolarizationParameters, dim: int) -> serialization.PolarizationParameters: ...
@tp.overload
def unsqueeze(item: serialization.TimeParameters, dim: int) -> serialization.TimeParameters: ...
@tp.overload
def unsqueeze(item: serialization.CWSpectralData, dim: int) -> serialization.CWSpectralData: ...
@tp.overload
def unsqueeze(item: serialization.SerializedSampleWidth, dim: int) -> serialization.SerializedSampleWidth: ...


def unsqueeze(
        item: serialized_item_type,
        dim: int,
) -> serialized_item_type:
    """
    Insert a dimension of size one at the specified position in a serialized or graph MaRs item.

    :param item: A single serialized or graph MARS object.
    :param dim: The index at which to insert the new dimension.
    :return: A new object with the new dimension.
    """
    trans_func = _get_transform_func(item)
    return trans_func(
        item,
        op=lambda t, dim: t.unsqueeze(dim),
        dim=dim,
        allow_end=True,
    )


@tp.overload
def squeeze(item: serialization.SerializedSpinSystem, dim: tp.Optional[int] = None) ->\
        serialization.SerializedSpinSystem: ...
@tp.overload
def squeeze(item: serialization.SerializedSample, dim: tp.Optional[int] = None) ->\
        serialization.SerializedSample: ...
@tp.overload
def squeeze(item: graph_representation.GraphSpinSystem, dim: tp.Optional[int] = None) ->\
        graph_representation.GraphSpinSystem: ...
@tp.overload
def squeeze(item: graph_representation.GraphSample, dim: tp.Optional[int] = None) ->\
        graph_representation.GraphSample: ...
@tp.overload
def squeeze(item: serialization.ExperimentalParameters, dim: tp.Optional[int] = None) ->\
        serialization.ExperimentalParameters: ...
@tp.overload
def squeeze(item: serialization.PolarizationParameters, dim: tp.Optional[int] = None) ->\
        serialization.PolarizationParameters: ...
@tp.overload
def squeeze(item: serialization.TimeParameters, dim: tp.Optional[int] = None) ->\
        serialization.TimeParameters: ...
@tp.overload
def squeeze(item: serialization.CWSpectralData, dim: tp.Optional[int] = None) ->\
        serialization.CWSpectralData: ...
@tp.overload
def squeeze(item: serialization.SerializedSampleWidth, dim: tp.Optional[int] = None) ->\
        serialization.SerializedSampleWidth: ...


def squeeze(
        item: serialized_item_type,
        dim: tp.Optional[int] = None,
) -> serialized_item_type:
    """
    Squeeze batch dimensions of a serialized or graph MaRs item.

    :param item: A single serialized or graph MARS object.
    :param dim: If given, only the specified dimension will be squeezed.
        Otherwise, all dimensions of size 1 are removed.
    :return: A new object with squeezed batch dimensions.
    """
    trans_func = _get_transform_func(item)

    if dim is None:
        def _squeeze_batch_only(t: torch.Tensor, protected_tail_dims: int):
            batch_ndim = t.ndim - protected_tail_dims
            if batch_ndim < 0:
                raise ValueError("Invalid protected_tail_dims for tensor rank.")
            for d in range(batch_ndim - 1, -1, -1):
                if t.size(d) == 1:
                    t = t.squeeze(d)
            return t
        return trans_func(item, op=_squeeze_batch_only)

    return trans_func(item, op=lambda t, dim: t.squeeze(dim), dim=dim)


@tp.overload
def transpose(item: serialization.SerializedSpinSystem, dim0: int, dim1: int) ->\
        serialization.SerializedSpinSystem: ...
@tp.overload
def transpose(item: serialization.SerializedSample, dim0: int, dim1: int) ->\
        serialization.SerializedSample: ...
@tp.overload
def transpose(item: graph_representation.GraphSpinSystem, dim0: int, dim1: int) ->\
        graph_representation.GraphSpinSystem: ...
@tp.overload
def transpose(item: graph_representation.GraphSample, dim0: int, dim1: int) ->\
        graph_representation.GraphSample: ...
@tp.overload
def transpose(item: serialization.ExperimentalParameters, dim0: int, dim1: int) ->\
        serialization.ExperimentalParameters: ...
@tp.overload
def transpose(item: serialization.PolarizationParameters, dim0: int, dim1: int) ->\
        serialization.PolarizationParameters: ...
@tp.overload
def transpose(item: serialization.TimeParameters, dim0: int, dim1: int) ->\
        serialization.TimeParameters: ...
@tp.overload
def transpose(item: serialization.CWSpectralData, dim0: int, dim1: int) ->\
        serialization.CWSpectralData: ...
@tp.overload
def transpose(item: serialization.SerializedSampleWidth, dim0: int, dim1: int) ->\
        serialization.SerializedSampleWidth: ...


def transpose(
        item: serialized_item_type,
        dim0: int, dim1: int,
) -> serialized_item_type:
    """
    Squeeze batch dimensions of a serialized or graph MaRs item.

    :param item: A single serialized or graph MARS object.
    :param dim0:
    :param dim1:
    :return: A new object with squeezed batch dimensions.
    """
    trans_func = _get_transform_func(item)
    return trans_func(item, op=torch.transpose, dim0=dim0, dim1=dim1)


@tp.overload
def mask(item: serialization.SerializedSpinSystem, mask: torch.Tensor) -> serialization.SerializedSpinSystem: ...
@tp.overload
def mask(item: serialization.SerializedSample, mask: torch.Tensor) -> serialization.SerializedSample: ...
@tp.overload
def mask(item: graph_representation.GraphSpinSystem, mask: torch.Tensor) -> graph_representation.GraphSpinSystem: ...
@tp.overload
def mask(item: graph_representation.GraphSample, mask: torch.Tensor) -> graph_representation.GraphSample: ...
@tp.overload
def mask(item: serialization.ExperimentalParameters, mask: torch.Tensor) -> serialization.ExperimentalParameters: ...
@tp.overload
def mask(item: serialization.PolarizationParameters, mask: torch.Tensor) -> serialization.PolarizationParameters: ...
@tp.overload
def mask(item: serialization.TimeParameters, mask: torch.Tensor) -> serialization.TimeParameters: ...
@tp.overload
def mask(item: serialization.CWSpectralData, mask: torch.Tensor) -> serialization.CWSpectralData: ...
@tp.overload
def mask(item: serialization.SerializedSampleWidth, mask: torch.Tensor) -> serialization.SerializedSampleWidth: ...


def mask(item: serialized_item_type, mask: torch.Tensor) -> serialized_item_type:
    """
    Apply a boolean mask to the batch dimensions of a serialized or graph MaRs item.

    The mask's shape must exactly match the batch dimensions of the item
    (i.e. all dimensions except the protected system tail). All batch
    dimensions are collapsed into a single leading dimension of size
    ``mask.sum()``. Protected tail dimensions are preserved untouched.

    :param item: A single serialized or graph MARS object.
    :param mask: Boolean tensor whose shape equals the batch shape of ``item``.
    :return: A new object with masked batch dimensions.
    :raises RuntimeError: If ``mask.shape`` does not match the batch shape
        (raised by PyTorch during ``tensor[mask]``).
    """
    if mask.dtype != torch.bool:
        raise TypeError(
            f"`mask` must be a boolean tensor, got dtype={mask.dtype}. "
            "For integer-based indexing, use a different operation."
        )
    trans_func = _get_transform_func(item)
    return trans_func(item, op=lambda t, m: t[m], m=mask)
