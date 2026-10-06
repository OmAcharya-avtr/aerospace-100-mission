"""Interleaver construction and burst-dispersion metrics.

Four constructions -- block, convolutional, helical and S-random -- with the
spread, dispersion and burst-dispersion metrics and the latency and memory
accounting a fading-link designer needs to choose between them.

The permutation convention, which everything else depends on, is::

    y[i] = x[pi[i]]

so ``pi[i]`` is the source index that lands at transmitted position ``i``.  See
:mod:`interleavekit.base` for the full statement and for the inverse map that the
burst metrics consume.

This software is research-grade.  It is not flight-qualified, not certified, and
not approved for operational aerospace use.
"""

from __future__ import annotations

from .base import Interleaver, InterleaverCost
from .block import BlockInterleaver
from .convolutional import ConvolutionalInterleaver
from .cost import (
    CostRow,
    cheapest_for_burst,
    cost_table,
    describe,
    format_cost_table,
    row_as_dict,
)
from .helical import HelicalInterleaver
from .metrics import (
    as_permutation,
    burst_dispersion,
    burst_dispersion_by_window_scan,
    burst_dispersion_profile,
    dispersion,
    is_bijection,
    longest_consecutive_run,
    max_burst_fully_dispersed,
    minimum_spread,
    s_parameter,
    transmitted_span_profile,
)
from .srandom import SRandomInterleaver

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # constructions
    "BlockInterleaver",
    "ConvolutionalInterleaver",
    "HelicalInterleaver",
    "SRandomInterleaver",
    "Interleaver",
    # metrics
    "as_permutation",
    "burst_dispersion",
    "burst_dispersion_by_window_scan",
    "burst_dispersion_profile",
    "dispersion",
    "is_bijection",
    "longest_consecutive_run",
    "max_burst_fully_dispersed",
    "minimum_spread",
    "s_parameter",
    "transmitted_span_profile",
    # cost
    "CostRow",
    "InterleaverCost",
    "cheapest_for_burst",
    "cost_table",
    "describe",
    "format_cost_table",
    "row_as_dict",
]
