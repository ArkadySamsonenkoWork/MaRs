import typing as tp

from . import population


def multiply(
        mars_items: tp.Sequence[tp.Union[population.Context, population.SummedContext, population.KroneckerContext]]
) -> tp.Union[population.Context, population.SummedContext]:
    """
    Multiply a sequence of homogeneous MaRs items into a single combined item.

    This function dispatches to the appropriate multiplication routine based on the type
    of the first element in the input sequence. It supports three types of inputs:

    - A sequence of :class:`Context` (or :class:`SummedContext`) -> returns a composite
      context that computed multiplication  of all contexts

    The input sequence must be non-empty and contain only one kind of item type.

    In all cases, the direct sum of the elements you want to concatenate is calculated.
    To understand the logic and interpretation of operation in each case, see

    - :func:`mars.population.Context.multiply_contexts`

    :param mars_items: A non-empty sequence of identical-type MARS objects.
                       Must be one of:
                       - ``Sequence[BaseContext]``

    :return: A single MARS object of the corresponding type that represents the
             logical concatenation of all input items.

    :raises TypeError: If the item type is not supported.
    :raises ValueError: If underlying multiplication logic detects inconsistencies
                        (e.g., mismatched dtypes, devices, or spectral parameters).
    """
    ref_item = mars_items[0]
    if isinstance(ref_item, population.BaseContext):
        return population.contexts.multiply_contexts(mars_items)
