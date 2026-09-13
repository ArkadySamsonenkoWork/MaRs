import typing as tp

from .. import spin_model
from .. import population
from ..serialization import serialization as ser
from ..serialization import graph_representation as graph


mars_items_types = tp.Union[
    tp.Sequence[spin_model.SpinSystem],
    tp.Sequence[spin_model.SolidSample],
    tp.Sequence[population.BaseContext],
    tp.Sequence[ser.SerializedSpinSystem],
    tp.Sequence[ser.SerializedSample],
    tp.Sequence[ser.SerializedSampleWidth],
    tp.Sequence[graph.GraphSpinSystem],
    tp.Sequence[graph.GraphSample],
    tp.Sequence[ser.ExperimentalParameters],
    tp.Sequence[ser.PolarizationParameters],
    tp.Sequence[ser.TimeParameters],
]

mars_item_type = tp.Union[
    spin_model.SpinSystem,
    spin_model.SolidSample,
    population.BaseContext,
    ser.SerializedSpinSystem,
    ser.SerializedSample,
    ser.SerializedSampleWidth,
    graph.GraphSpinSystem,
    graph.GraphSample,
    ser.ExperimentalParameters,
    ser.PolarizationParameters,
    ser.TimeParameters,
]
