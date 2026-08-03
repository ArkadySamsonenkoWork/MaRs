import typing as tp
import torch

from .. import spin_model
from .. import population
from ..serialization import serialization as ser
from ..serialization import graph_representation as graph
from ..serialization import operations_interface
from ._dispatch import mars_item_type


@tp.overload
def expand(item: spin_model.SpinSystem, sizes: tp.Union[torch.Size, tp.List[int]]) ->\
        spin_model.SpinSystem: ...
@tp.overload
def expand(item: spin_model.MultiOrientedSample, sizes: tp.Union[torch.Size, tp.List[int]]) ->\
        spin_model.MultiOrientedSample: ...
@tp.overload
def expand(item: population.BaseContext, sizes: tp.Union[torch.Size, tp.List[int]]) ->\
        population.BaseContext: ...
@tp.overload
def expand(item: ser.SerializedSpinSystem, sizes: tp.Union[torch.Size, tp.List[int]]) ->\
        ser.SerializedSpinSystem: ...
@tp.overload
def expand(item: ser.SerializedSample, sizes: tp.Union[torch.Size, tp.List[int]]) ->\
        ser.SerializedSample: ...
@tp.overload
def expand(item: graph.GraphSpinSystem, sizes: tp.Union[torch.Size, tp.List[int]]) ->\
        graph.GraphSpinSystem: ...
@tp.overload
def expand(item: graph.GraphSample, sizes: tp.Union[torch.Size, tp.List[int]]) ->\
        graph.GraphSample: ...
@tp.overload
def expand(item: ser.ExperimentalParameters, sizes: tp.Union[torch.Size, tp.List[int]]) ->\
        ser.ExperimentalParameters: ...
@tp.overload
def expand(item: ser.PolarizationParameters, sizes: tp.Union[torch.Size, tp.List[int]]) ->\
        ser.PolarizationParameters: ...
@tp.overload
def expand(item: ser.TimeParameters, sizes: tp.Union[torch.Size, tp.List[int]]) ->\
        ser.TimeParameters: ...
@tp.overload
def expand(item: ser.SerializedSampleWidth, sizes: tp.Union[torch.Size, tp.List[int]]) ->\
        ser.SerializedSampleWidth: ...


def expand(
        item: mars_item_type,
        sizes: tp.Union[torch.Size, tp.List[int]],
) -> mars_item_type:
    """
    Expand batch dimensions of a MaRs item.

    :param item: A single MARS object.
    :param sizes: The desired expanded sizes.
    :return: A new object with expanded batch dimensions.
    :raises NotImplementedError: If the item is a MARS object (not serialized or graph).
    """
    if isinstance(item, (spin_model.SpinSystem, spin_model.MultiOrientedSample, population.BaseContext)):
        raise NotImplementedError("Expand is not implemented for MARS objects.")

    return operations_interface.expand(item, sizes=sizes)
