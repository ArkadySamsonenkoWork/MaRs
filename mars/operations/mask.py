import typing as tp

import torch

from .. import spin_model
from .. import population
from ..serialization import serialization as ser
from ..serialization import graph_representation as graph
from ..serialization import operations_interface
from ._dispatch import mars_item_type


@tp.overload
def mask(item: spin_model.SpinSystem, mask: torch.Tensor) -> spin_model.SpinSystem: ...
@tp.overload
def mask(item: spin_model.MultiOrientedSample, mask: torch.Tensor) -> spin_model.MultiOrientedSample: ...
@tp.overload
def mask(item: population.BaseContext, mask: torch.Tensor) -> population.BaseContext: ...
@tp.overload
def mask(item: ser.SerializedSpinSystem, mask: torch.Tensor) -> ser.SerializedSpinSystem: ...
@tp.overload
def mask(item: ser.SerializedSample, mask: torch.Tensor) -> ser.SerializedSample: ...
@tp.overload
def mask(item: graph.GraphSpinSystem, mask: torch.Tensor) -> graph.GraphSpinSystem: ...
@tp.overload
def mask(item: graph.GraphSample, mask: torch.Tensor) -> graph.GraphSample: ...
@tp.overload
def mask(item: ser.ExperimentalParameters, mask: torch.Tensor) -> ser.ExperimentalParameters: ...
@tp.overload
def mask(item: ser.PolarizationParameters, mask: torch.Tensor) -> ser.PolarizationParameters: ...
@tp.overload
def mask(item: ser.TimeParameters, mask: torch.Tensor) -> ser.TimeParameters: ...
@tp.overload
def mask(item: ser.CWSpectralData, mask: torch.Tensor) -> ser.CWSpectralData: ...
@tp.overload
def mask(item: ser.SerializedSampleWidth, mask: torch.Tensor) -> ser.SerializedSampleWidth: ...


def mask(
        item: mars_item_type,
        mask: torch.Tensor,
) -> mars_item_type:
    """
    Remove dimensions of size one from a MaRs item.

    :param item: A single MARS object.
    :param mask: Boolean tensor whose shape equals the batch shape of ``item``.
    :return: A new object with the specified dimensions removed.
    :raises NotImplementedError: If the item is a live MARS object (not serialized or graph).
    """
    if isinstance(item, (spin_model.SpinSystem, spin_model.MultiOrientedSample, population.BaseContext)):
        raise NotImplementedError("Squeeze is not implemented for live MARS objects.")

    return operations_interface.mask(item, mask=mask)
