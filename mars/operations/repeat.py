import typing as tp
import torch

from .. import spin_model
from .. import population
from ..serialization import serialization as ser
from ..serialization import graph_representation as graph
from ..serialization import operations_interface
from ._dispatch import mars_item_type


@tp.overload
def repeat(item: spin_model.SpinSystem, repeats: tp.Union[torch.Size, tp.List[int]]) ->\
        spin_model.SpinSystem: ...
@tp.overload
def repeat(item: spin_model.MultiOrientedSample, repeats: tp.Union[torch.Size, tp.List[int]]) ->\
        spin_model.MultiOrientedSample: ...
@tp.overload
def repeat(item: population.BaseContext, repeats: tp.Union[torch.Size, tp.List[int]]) ->\
        population.BaseContext: ...
@tp.overload
def repeat(item: ser.SerializedSpinSystem, repeats: tp.Union[torch.Size, tp.List[int]]) ->\
        ser.SerializedSpinSystem: ...
@tp.overload
def repeat(item: ser.SerializedSample, repeats: tp.Union[torch.Size, tp.List[int]]) ->\
        ser.SerializedSample: ...
@tp.overload
def repeat(item: graph.GraphSpinSystem, repeats: tp.Union[torch.Size, tp.List[int]]) ->\
        graph.GraphSpinSystem: ...
@tp.overload
def repeat(item: graph.GraphSample, repeats: tp.Union[torch.Size, tp.List[int]]) ->\
        graph.GraphSample: ...
@tp.overload
def repeat(item: ser.ExperimentalParameters, repeats: tp.Union[torch.Size, tp.List[int]]) ->\
        ser.ExperimentalParameters: ...
@tp.overload
def repeat(item: ser.PolarizationParameters, repeats: tp.Union[torch.Size, tp.List[int]]) ->\
        ser.PolarizationParameters: ...
@tp.overload
def repeat(item: ser.TimeParameters, repeats: tp.Union[torch.Size, tp.List[int]]) ->\
        ser.TimeParameters: ...
@tp.overload
def repeat(item: ser.SerializedSampleWidth, repeats: tp.Union[torch.Size, tp.List[int]]) ->\
        ser.SerializedSampleWidth: ...



def repeat(
        item: mars_item_type,
        repeats: tp.Union[torch.Size, tp.List[int]],
) -> mars_item_type:
    """
    Repeat batch dimensions of a MaRs item.

    :param item: A single MARS object.
    :param repeats: The number of times to repeat each dimension.
    :return: A new object with repeated batch dimensions.
    :raises NotImplementedError: If the item is a MARS object (not serialized or graph).
    """
    if isinstance(item, (spin_model.SpinSystem, spin_model.MultiOrientedSample, population.BaseContext)):
        raise NotImplementedError("Repeat is not implemented for MARS objects.")

    return operations_interface.repeat(item, repeats=repeats)
