import typing as tp

from .. import spin_model
from .. import population
from ..serialization import serialization as ser
from ..serialization import graph_representation as graph
from ..serialization import operations_interface
from ._dispatch import mars_item_type


@tp.overload
def unsqueeze(item: spin_model.SpinSystem, dim: int) -> spin_model.SpinSystem: ...
@tp.overload
def unsqueeze(item: spin_model.MultiOrientedSample, dim: int) -> spin_model.MultiOrientedSample: ...
@tp.overload
def unsqueeze(item: population.BaseContext, dim: int) -> population.BaseContext: ...
@tp.overload
def unsqueeze(item: ser.SerializedSpinSystem, dim: int) -> ser.SerializedSpinSystem: ...
@tp.overload
def unsqueeze(item: ser.SerializedSample, dim: int) -> ser.SerializedSample: ...
@tp.overload
def unsqueeze(item: graph.GraphSpinSystem, dim: int) -> graph.GraphSpinSystem: ...
@tp.overload
def unsqueeze(item: graph.GraphSample, dim: int) -> graph.GraphSample: ...
@tp.overload
def unsqueeze(item: ser.ExperimentalParameters, dim: int) -> ser.ExperimentalParameters: ...
@tp.overload
def unsqueeze(item: ser.PolarizationParameters, dim: int) -> ser.PolarizationParameters: ...
@tp.overload
def unsqueeze(item: ser.TimeParameters, dim: int) -> ser.TimeParameters: ...
@tp.overload
def unsqueeze(item: ser.CWSpectralData, dim: int) -> ser.CWSpectralData: ...
@tp.overload
def unsqueeze(item: ser.SerializedSampleWidth, dim: int) -> ser.SerializedSampleWidth: ...


def unsqueeze(
        item: mars_item_type,
        dim: int,
) -> mars_item_type:
    """
    Insert a dimension of size one at the specified position in a MaRs item.

    :param item: A single MARS object.
    :param dim: The index at which to insert the new dimension.
    :return: A new object with the new dimension.
    :raises NotImplementedError: If the item is a live MARS object (not serialized or graph).
    """
    if isinstance(item, (spin_model.SpinSystem, spin_model.MultiOrientedSample, population.BaseContext)):
        raise NotImplementedError("Unsqueeze is not implemented for live MARS objects.")

    return operations_interface.unsqueeze(item, dim=dim)
