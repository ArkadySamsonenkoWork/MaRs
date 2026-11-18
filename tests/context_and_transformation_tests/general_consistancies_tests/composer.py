"""Combine the eight context fixtures into (sample, context, description) cases.

The same sample is used throughout the expression, and the dimensions of
context products/sums are kept aligned with their corresponding samples.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from context_generators import ContextPair, generate_contexts
from sample_generators import SampleCase, iter_sample_cases

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..", "..", "..")))


@dataclass(frozen=True)
class ComposedCase:
    sample: object
    context: object
    description: str
    has_decay: bool
    unique_thermal_equilibrium: bool = True

    def __iter__(self):
        return iter((self.sample, self.context, self.description))

    @property
    def triple(self):
        return self.sample, self.context, self.description


def add_pair(pair: ContextPair):
    return pair.first + pair.second


def tensor_contexts(pair1: ContextPair, pair2: ContextPair):
    """(c1_a + c1_b) @ (c2_a + c2_b), matching sample_mul dimension."""
    return add_pair(pair1) @ add_pair(pair2)


def direct_sum_contexts(pair1: ContextPair, pair2: ContextPair):
    """mars.concat((c1_a+c1_b, c2_a+c2_b)), matching sample_direct_sum."""
    from mars import concat
    return concat((add_pair(pair1), add_pair(pair2)))


def compose_cases(samples: SampleCase,
                  contexts: tuple[ContextPair, ContextPair, ContextPair, ContextPair]
                  ) -> list[ComposedCase]:
    """Seven meaningful arrangements of sample and context objects."""
    a, b, mul, ds = contexts
    decay = a.has_decay or b.has_decay or mul.has_decay or ds.has_decay
    return [
        ComposedCase(samples.sample_1, add_pair(a), "sample_1 : c1_a + c1_b", a.has_decay),
        ComposedCase(samples.sample_2, add_pair(b), "sample_2 : c2_a + c2_b", b.has_decay),
        ComposedCase(samples.sample_mul, add_pair(mul), "sample_mul : cm_a + cm_b", mul.has_decay),
        ComposedCase(samples.sample_direct_sum, add_pair(ds),
                     "sample_direct_sum : cd_a + cd_b", ds.has_decay),
        ComposedCase(samples.sample_mul, tensor_contexts(a, b),
                     "sample_mul : (c1_a + c1_b) @ (c2_a + c2_b)", decay),
        ComposedCase(samples.sample_direct_sum, direct_sum_contexts(a, b),
                     "sample_direct_sum : mars.concat((c1_a+c1_b, c2_a+c2_b))",
                     decay, unique_thermal_equilibrium=False),
        ComposedCase(samples.sample_direct_sum,
                     direct_sum_contexts(a, b) + add_pair(ds),
                     "sample_direct_sum : mars.concat((c1_a+c1_b, c2_a+c2_b)) + (cd_a+cd_b)",
                     decay),
    ]


def generate_composed_cases(n, rng, *, secular=False, decay=False, **kwargs
                            ) -> Iterator[ComposedCase]:
    """Yield all seven arrangements for each of N independently randomized fixtures.

    This generator does NOT select thermal correction; that's utils.py's job.
    """
    sample_kwargs = {k: v for k, v in kwargs.items() if k in {"spins", "partition", "exchange_hz", "dtype", "device", "mesh"}}
    context_kwargs = {k: v for k, v in kwargs.items() if k in {"dtype", "device"}}
    for fixture in iter_sample_cases(n, rng, **sample_kwargs):
        pairs = generate_contexts(fixture, rng, secular=secular, decay=decay, **context_kwargs)
        yield from compose_cases(fixture, pairs)


def generate_triples(n, rng, *, secular=False, decay=False, **kwargs):
    """Simple requested interface: generator[(sample_total, context_total, description)]."""
    for case in generate_composed_cases(n, rng, secular=secular, decay=decay, **kwargs):
        yield case.triple
