import typing as tp

from .. import spin_model
from .. import population
from ..serialization import serialization as ser
from ..serialization import graph_representation as graph
from ..serialization import operations_interface
from ._dispatch import mars_items_types, mars_item_type



@tp.overload
def concat(items: tp.Sequence[spin_model.SpinSystem], mode: tp.Literal["direct_sum", "batch"] = "direct_sum",
           dim: int = -1) -> spin_model.SpinSystem: ...
@tp.overload
def concat(items: tp.Sequence[spin_model.MultiOrientedSample], mode: tp.Literal["direct_sum", "batch"] = "direct_sum",
           dim: int = -1) -> spin_model.MultiOrientedSample: ...
@tp.overload
def concat(items: tp.Sequence[population.BaseContext], mode: tp.Literal["direct_sum", "batch"] = "direct_sum",
           dim: int = -1) -> population.BaseContext: ...
@tp.overload
def concat(items: tp.Sequence[ser.SerializedSpinSystem], mode: tp.Literal["direct_sum", "batch"] = "direct_sum",
           dim: int = -1) -> ser.SerializedSpinSystem: ...
@tp.overload
def concat(items: tp.Sequence[ser.SerializedSample], mode: tp.Literal["direct_sum", "batch"] = "direct_sum",
           dim: int = -1) -> ser.SerializedSample: ...
@tp.overload
def concat(items: tp.Sequence[graph.GraphSpinSystem], mode: tp.Literal["direct_sum", "batch"] = "direct_sum",
           dim: int = -1) -> graph.GraphSpinSystem: ...
@tp.overload
def concat(items: tp.Sequence[ser.SerializedSampleWidth], mode: tp.Literal["direct_sum", "batch"] = "direct_sum",
           dim: int = -1) -> ser.SerializedSampleWidth: ...
@tp.overload
def concat(items: tp.Sequence[graph.GraphSample], mode: tp.Literal["direct_sum", "batch"] = "direct_sum",
           dim: int = -1) -> graph.GraphSample: ...
@tp.overload
def concat(items: tp.Sequence[ser.ExperimentalParameters], mode: tp.Literal["direct_sum", "batch"] = "direct_sum",
           dim: int = -1) -> ser.ExperimentalParameters: ...
@tp.overload
def concat(items: tp.Sequence[ser.PolarizationParameters], mode: tp.Literal["direct_sum", "batch"] = "direct_sum",
           dim: int = -1) -> ser.PolarizationParameters: ...
@tp.overload
def concat(items: tp.Sequence[ser.TimeParameters], mode: tp.Literal["direct_sum", "batch"] = "direct_sum",
           dim: int = -1) -> ser.TimeParameters: ...


def concat(
        mars_items: mars_items_types,
        mode: tp.Literal["direct_sum", "batch"] = "direct_sum",
        dim: int = -1,
) -> mars_item_type:
    """
    Concatenate a sequence of homogeneous MaRs items into a single combined item.

    This function dispatches to the appropriate concatenation routine based on the type
    of the first element in the input sequence. It supports three types of inputs:

    - A sequence of :class:`SpinSystem` -> returns a single block-diagonal :class:`SpinSystem`
    - A sequence of :class:`MultiOrientedSample` -> returns a single :class:`MultiOrientedSample`
      with a concatenated spin system and shared spectral parameters
    - A sequence of :class:`Context` (or :class:`SummedContext`) -> returns a composite
      context that computed direct sum of all contexts

    The input sequence must be non-empty and contain only one kind of item type.

    By default, the direct sum of the elements you want to concatenate is calculated.
    If ``mode="batch"`` is specified, items are concatenated along their existing batch
    dimension instead, keeping them mathematically independent.

    To understand the logic and interpretation of operation in each case, see

    - :func:`mars.spin_model.concat_spin_system`
    - :func:`mars.spin_model.concat_multioriented_samples`
    - :func:`mars.population.concatinations.concat_contexts`

    :param mars_items: A non-empty sequence of identical-type MARS objects.
                       Must be one of:
                       - ``Sequence[SpinSystem]``
                       - ``Sequence[MultiOrientedSample]``
                       - ``Sequence[BaseContext]``
    :param mode: The concatenation mode.
                 - ``"direct_sum"`` (default): Combines state spaces (block-diagonal).
                 - ``"batch"``: Concatenates along the existing batch dimension.

    :param dim: The dimension along which to concatenate (used for "batch" mode). Default is -1.
    :return: A single MARS object of the corresponding type that represents the
             logical concatenation of all input items.

    :raises TypeError: If the item type is not supported.
    :raises ValueError: If underlying concatenation logic detects inconsistencies
                        (e.g., mismatched dtypes, devices, or spectral parameters).
    """
    if not mars_items:
        raise ValueError("Cannot concatenate an empty sequence.")

    ref_item = mars_items[0]

    if isinstance(ref_item, spin_model.SpinSystem):
        return spin_model.concat_spin_systems(mars_items, mode=mode)
    elif isinstance(ref_item, spin_model.MultiOrientedSample):
        return spin_model.concat_multioriented_samples(mars_items, mode=mode)
    elif isinstance(ref_item, population.BaseContext):
        return population.concat_contexts(mars_items, mode=mode)

    if mode != "batch":
        raise NotImplementedError(
            "Serialized and graph items only support 'batch' mode. "
            "For 'direct_sum', convert to live MARS objects first."
        )

    return operations_interface.concat(mars_items, mode=mode, dim=-1)
