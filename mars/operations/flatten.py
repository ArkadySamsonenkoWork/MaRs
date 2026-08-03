import typing as tp

from .. import spin_model
from .. import population
from ..serialization import serialization as ser
from ..serialization import graph_representation as graph
from ..serialization import operations_interface
from ._dispatch import mars_item_type


@tp.overload
def flatten(item: spin_model.SpinSystem, start_dim: int = 0, end_dim: int = -3) ->\
        spin_model.SpinSystem: ...
@tp.overload
def flatten(item: spin_model.MultiOrientedSample, start_dim: int = 0, end_dim: int = -3) ->\
        spin_model.MultiOrientedSample: ...
@tp.overload
def flatten(item: population.BaseContext, start_dim: int = 0, end_dim: int = -3) ->\
        population.BaseContext: ...
@tp.overload
def flatten(item: ser.SerializedSpinSystem, start_dim: int = 0, end_dim: int = -3) ->\
        ser.SerializedSpinSystem: ...
@tp.overload
def flatten(item: ser.SerializedSample, start_dim: int = 0, end_dim: int = -3) ->\
        ser.SerializedSample: ...
@tp.overload
def flatten(item: graph.GraphSpinSystem, start_dim: int = 0, end_dim: int = -3) ->\
        graph.GraphSpinSystem: ...
@tp.overload
def flatten(item: graph.GraphSample, start_dim: int = 0, end_dim: int = -3) ->\
        graph.GraphSample: ...
@tp.overload
def flatten(item: ser.ExperimentalParameters, start_dim: int = 0, end_dim: int = -3) ->\
        ser.ExperimentalParameters: ...
@tp.overload
def flatten(item: ser.PolarizationParameters, start_dim: int = 0, end_dim: int = -3) ->\
        ser.PolarizationParameters: ...
@tp.overload
def flatten(item: ser.TimeParameters, start_dim: int = 0, end_dim: int = -3) ->\
        ser.TimeParameters: ...
@tp.overload
def flatten(item: ser.SerializedSampleWidth, start_dim: int = 0, end_dim: int = -3) ->\
        ser.SerializedSampleWidth: ...


def flatten(
        item: mars_item_type,
        start_dim: int = 0,
        end_dim: int = -1,
) -> mars_item_type:
    """
    Flatten batch dimensions of a MaRs item.

    :param item: A single MARS object.
    :param start_dim: The first dimension to flatten. Default is 0.
    :param end_dim: The last dimension to flatten. Default is -3 (leaves the last 2 system dims intact).
    :return: A new object with flattened batch dimensions.
    :raises NotImplementedError: If the item is a MARS object (not serialized or graph).
    """
    if isinstance(item, (spin_model.SpinSystem, spin_model.MultiOrientedSample, population.BaseContext)):
        raise NotImplementedError("Flatten is not implemented for MARS objects.")

    return operations_interface.flatten(item, start_dim=start_dim, end_dim=end_dim)
