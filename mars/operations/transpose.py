import typing as tp

from .. import spin_model
from .. import population
from ..serialization import serialization as ser
from ..serialization import graph_representation as graph
from ..serialization import operations_interface
from ._dispatch import mars_item_type


@tp.overload
def transpose(item: spin_model.SpinSystem, dim0: int, dim1: int) -> spin_model.SpinSystem: ...
@tp.overload
def transpose(item: spin_model.SolidSample, dim0: int, dim1: int) -> spin_model.SolidSample: ...
@tp.overload
def transpose(item: population.BaseContext, dim0: int, dim1: int) -> population.BaseContext: ...
@tp.overload
def transpose(item: ser.SerializedSpinSystem, dim0: int, dim1: int) -> ser.SerializedSpinSystem: ...
@tp.overload
def transpose(item: ser.SerializedSample, dim0: int, dim1: int) -> ser.SerializedSample: ...
@tp.overload
def transpose(item: graph.GraphSpinSystem, dim0: int, dim1: int) -> graph.GraphSpinSystem: ...
@tp.overload
def transpose(item: graph.GraphSample, dim0: int, dim1: int) -> graph.GraphSample: ...
@tp.overload
def transpose(item: ser.ExperimentalParameters, dim0: int, dim1: int) -> ser.ExperimentalParameters: ...
@tp.overload
def transpose(item: ser.PolarizationParameters, dim0: int, dim1: int) -> ser.PolarizationParameters: ...
@tp.overload
def transpose(item: ser.TimeParameters, dim0: int, dim1: int) -> ser.TimeParameters: ...
@tp.overload
def transpose(item: ser.CWSpectralData, dim0: int, dim1: int) -> ser.CWSpectralData: ...
@tp.overload
def transpose(item: ser.SerializedSampleWidth, dim0: int, dim1: int) -> ser.SerializedSampleWidth: ...


def transpose(
        item: mars_item_type,
        dim0: int, dim1: int
) -> mars_item_type:
    """

    :param item: A single MARS object.
    :param dim0
    :param dim1
    :return: A new object with the new dimension.
    :raises NotImplementedError: If the item is a live MARS object (not serialized or graph).
    """
    if isinstance(item, (spin_model.SpinSystem, spin_model.SolidSample, population.BaseContext)):
        raise NotImplementedError("Transpose is not implemented for live MARS objects.")

    return operations_interface.transpose(item, dim0=dim0, dim1=dim1)
