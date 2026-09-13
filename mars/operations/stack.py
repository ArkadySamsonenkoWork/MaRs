import typing as tp

from .. import spin_model
from .. import population
from ..serialization import serialization as ser
from ..serialization import graph_representation as graph
from ..serialization import operations_interface

from ._dispatch import mars_items_types, mars_item_type


@tp.overload
def stack(items: tp.Sequence[spin_model.SpinSystem], dim: int = -1) -> spin_model.SpinSystem: ...
@tp.overload
def stack(items: tp.Sequence[spin_model.SolidSample], dim: int = -1) -> spin_model.SolidSample: ...
@tp.overload
def stack(items: tp.Sequence[population.BaseContext], dim: int = -1) -> population.BaseContext: ...
@tp.overload
def stack(items: tp.Sequence[ser.SerializedSpinSystem], dim: int = -1) -> ser.SerializedSpinSystem: ...
@tp.overload
def stack(items: tp.Sequence[ser.SerializedSample], dim: int = -1) -> ser.SerializedSample: ...
@tp.overload
def stack(items: tp.Sequence[graph.GraphSpinSystem], dim: int = -1) -> graph.GraphSpinSystem: ...
@tp.overload
def stack(items: tp.Sequence[graph.GraphSample], dim: int = -1) -> graph.GraphSample: ...
@tp.overload
def stack(items: tp.Sequence[ser.ExperimentalParameters], dim: int = -1) -> ser.ExperimentalParameters: ...
@tp.overload
def stack(items: tp.Sequence[ser.PolarizationParameters], dim: int = -1) -> ser.PolarizationParameters: ...
@tp.overload
def stack(items: tp.Sequence[ser.TimeParameters], dim: int = -1) -> ser.TimeParameters: ...
@tp.overload
def stack(items: tp.Sequence[ser.SerializedSampleWidth], dim: int = -1) -> ser.SerializedSampleWidth: ...


def stack(
        mars_items: mars_items_types,
        dim: int = -1,
) -> mars_item_type:
    """
    """
    if not mars_items:
        raise ValueError("Cannot concatenate an empty sequence.")

    ref_item = mars_items[0]

    if isinstance(ref_item, spin_model.SpinSystem):
        raise NotImplementedError()
    elif isinstance(ref_item, spin_model.SolidSample):
        raise NotImplementedError()
    elif isinstance(ref_item, population.BaseContext):
        raise NotImplementedError()

    return operations_interface.stack(mars_items, dim=dim)
