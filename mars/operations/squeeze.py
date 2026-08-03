import typing as tp

from .. import spin_model
from .. import population
from ..serialization import serialization as ser
from ..serialization import graph_representation as graph
from ..serialization import operations_interface
from ._dispatch import mars_item_type


@tp.overload
def squeeze(item: spin_model.SpinSystem, dim: tp.Optional[int] = None) -> spin_model.SpinSystem: ...
@tp.overload
def squeeze(item: spin_model.MultiOrientedSample, dim: tp.Optional[int] = None) -> spin_model.MultiOrientedSample: ...
@tp.overload
def squeeze(item: population.BaseContext, dim: tp.Optional[int] = None) -> population.BaseContext: ...
@tp.overload
def squeeze(item: ser.SerializedSpinSystem, dim: tp.Optional[int] = None) -> ser.SerializedSpinSystem: ...
@tp.overload
def squeeze(item: ser.SerializedSample, dim: tp.Optional[int] = None) -> ser.SerializedSample: ...
@tp.overload
def squeeze(item: graph.GraphSpinSystem, dim: tp.Optional[int] = None) -> graph.GraphSpinSystem: ...
@tp.overload
def squeeze(item: graph.GraphSample, dim: tp.Optional[int] = None) -> graph.GraphSample: ...
@tp.overload
def squeeze(item: ser.ExperimentalParameters, dim: tp.Optional[int] = None) -> ser.ExperimentalParameters: ...
@tp.overload
def squeeze(item: ser.PolarizationParameters, dim: tp.Optional[int] = None) -> ser.PolarizationParameters: ...
@tp.overload
def squeeze(item: ser.TimeParameters, dim: tp.Optional[int] = None) -> ser.TimeParameters: ...
@tp.overload
def squeeze(item: ser.CWSpectralData, dim: tp.Optional[int] = None) -> ser.CWSpectralData: ...
@tp.overload
def squeeze(item: ser.SerializedSampleWidth, dim: tp.Optional[int] = None) -> ser.SerializedSampleWidth: ...


def squeeze(
        item: mars_item_type,
        dim: tp.Optional[int] = None,
) -> mars_item_type:
    """
    Remove dimensions of size one from a MaRs item.

    :param item: A single MARS object.
    :param dim: If given, only the specified dimension will be squeezed. Otherwise, all dimensions of size 1 are removed.
    :return: A new object with the specified dimensions removed.
    :raises NotImplementedError: If the item is a live MARS object (not serialized or graph).
    """
    if isinstance(item, (spin_model.SpinSystem, spin_model.MultiOrientedSample, population.BaseContext)):
        raise NotImplementedError("Squeeze is not implemented for live MARS objects.")

    if dim is None:
        return operations_interface.squeeze(item)

    return operations_interface.squeeze(item, dim=dim)