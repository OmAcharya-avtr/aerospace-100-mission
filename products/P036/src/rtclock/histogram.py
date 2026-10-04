"""Latency samples with exact, explicitly-named percentile semantics.

Two percentile definitions are implemented, and the caller names which one.
There is no default that silently differs between releases.

``"nearest_rank"``
    The order-statistic definition. For ``N`` sorted samples ``x[1..N]`` and
    ``p`` in ``[0, 100]``::

        k = max(1, ceil(p/100 * N))
        P(p) = x[k]

    The result is **always one of the stored samples**. This is the definition
    used for latency service levels in ITU-T and in most SLO practice, and it
    is the one to quote when "the 99th percentile" has to mean "a latency that
    was actually observed".

``"linear"``
    Linear interpolation between the two closest order statistics -- the
    Hyndman & Fan (1996) type 7 definition, which is also NumPy's
    ``method="linear"`` and the default of ``numpy.percentile``::

        h = (N - 1) * p/100
        P(p) = x[floor(h)+1] + (h - floor(h)) * (x[floor(h)+2] - x[floor(h)+1])

    The result is generally **not** a stored sample. Quote this one when the
    samples are a draw from a continuous distribution and the percentile is an
    estimate of that distribution's quantile.

Reference for the type-7 definition
    R. J. Hyndman and Y. Fan, "Sample Quantiles in Statistical Packages",
    *The American Statistician* **50**(4), 361-365 (1996). Definition 7.

Both definitions agree exactly at ``p = 100`` (the maximum) and, for
``nearest_rank``, ``p = 0`` returns the minimum by the ``max(1, ...)`` clamp.

A sharp edge in nearest-rank, stated rather than smoothed over
--------------------------------------------------------------
``p`` arrives as a binary64 float, and a percentile whose decimal form is not
representable makes ``p/100 * N`` land just off an integer. For ``p = 99.9``
the stored value is 99.900000000000005684..., so with ``N = 20000``::

    p/100 * N = 19980.000000000004  ->  ceil = 19981, not 19980

and the returned order statistic is the 19981st, one rank above the
"intended" one. The same happens at ``N = 1000`` and ``N = 10000``.
``p = 99.95`` and ``p = 99.0`` are unaffected at those sizes.

This package does **not** apply a tolerance to hide it. Snapping the product
to a nearby integer would make the result depend on an undocumented epsilon,
and the whole point of naming the definition is that the caller can predict
the answer. The error is bounded by one order statistic, and a caller who
needs an exact rank should index the sorted samples directly. The behaviour
is pinned by a test and demonstrated in ``validation/validate_percentiles.py``.

Units: this module is unit-agnostic and stores whatever the caller adds. Every
other module in ``rtclock`` feeds it seconds, and :meth:`LatencyHistogram.summary`
labels its output ``s``.

Overrun accounting
------------------
An iteration *overruns* a budget when its latency is **strictly greater** than
the budget: ``latency_s > budget_s``. Equality is not an overrun. The
comparison is on the raw float with no tolerance, so the count is a pure
function of the stored samples and reproduces bit-exactly across platforms
for the same inputs. This is the convention used for the P031/P036 overrun
cross-check; see ``validation/VALIDATION.md``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal

PercentileMethod = Literal["nearest_rank", "linear"]

__all__ = [
    "LatencyHistogram",
    "OverrunReport",
    "PercentileMethod",
    "overrun_report",
    "percentile",
]


def percentile(
    samples: Sequence[float], p: float, method: PercentileMethod = "nearest_rank"
) -> float:
    """Exact percentile of ``samples`` under the named definition.

    Computed by sorting the samples; no binning, no approximation, no
    streaming estimator. Cost ``O(N log N)`` per call.

    Args:
        samples: at least one finite value.
        p: percentile in ``[0, 100]``.
        method: ``"nearest_rank"`` or ``"linear"``; see the module docstring.

    Returns:
        The percentile, in the units of ``samples``.

    Raises:
        ValueError: on empty ``samples``, ``p`` outside ``[0, 100]``, a
            non-finite sample, or an unknown ``method``.
    """
    if method not in ("nearest_rank", "linear"):
        raise ValueError(f"method must be 'nearest_rank' or 'linear', got {method!r}")
    if len(samples) == 0:
        raise ValueError("samples must contain at least one value")
    if not (0.0 <= p <= 100.0):
        raise ValueError(f"p must be in [0, 100], got {p}")
    xs = sorted(float(s) for s in samples)
    for x in xs:
        if not math.isfinite(x):
            raise ValueError(f"samples must all be finite, found {x}")
    n = len(xs)
    if method == "nearest_rank":
        k = max(1, math.ceil(p / 100.0 * n))
        return xs[k - 1]
    h = (n - 1) * (p / 100.0)
    lo = math.floor(h)
    hi = min(lo + 1, n - 1)
    return xs[lo] + (h - lo) * (xs[hi] - xs[lo])


@dataclass(frozen=True)
class OverrunReport:
    """Overrun accounting over a latency trace against a single budget.

    Attributes:
        budget_s: the budget each sample was compared against, units s.
        count: number of samples with ``latency > budget`` (strict).
        indices: zero-based indices of those samples, ascending.
        total_samples: length of the trace.
        worst_overshoot_s: largest ``latency - budget`` over the overrunning
            samples, units s, or 0.0 when there are none.
        longest_consecutive_run: length of the longest run of consecutive
            overrunning iterations. A cascade of back-to-back overruns is a
            different failure from the same count scattered, so it is counted
            separately.
    """

    budget_s: float
    count: int
    indices: tuple[int, ...]
    total_samples: int
    worst_overshoot_s: float
    longest_consecutive_run: int

    @property
    def fraction(self) -> float:
        """``count / total_samples``, dimensionless. 0.0 for an empty trace."""
        return self.count / self.total_samples if self.total_samples else 0.0


def overrun_report(trace_s: Sequence[float], budget_s: float) -> OverrunReport:
    """Count overruns in a latency trace against a budget, strictly.

    An iteration overruns when ``trace_s[i] > budget_s``. Equality is not an
    overrun; no tolerance is applied.

    Args:
        trace_s: per-iteration latencies, units s, all finite and >= 0.
        budget_s: budget, units s, must be > 0.

    Returns:
        An :class:`OverrunReport`.

    Raises:
        ValueError: if ``budget_s <= 0``, or a sample is negative or
            non-finite.
    """
    if budget_s <= 0.0:
        raise ValueError(f"budget_s must be > 0 s, got {budget_s}")
    vals = [float(x) for x in trace_s]
    for i, x in enumerate(vals):
        if not math.isfinite(x):
            raise ValueError(f"trace_s[{i}] must be finite, got {x}")
        if x < 0.0:
            raise ValueError(f"trace_s[{i}] must be >= 0 s, got {x}")
    indices = tuple(i for i, x in enumerate(vals) if x > budget_s)
    worst = max((vals[i] - budget_s for i in indices), default=0.0)
    longest = 0
    run = 0
    previous = -2
    for i in indices:
        run = run + 1 if i == previous + 1 else 1
        longest = max(longest, run)
        previous = i
    return OverrunReport(
        budget_s=float(budget_s),
        count=len(indices),
        indices=indices,
        total_samples=len(vals),
        worst_overshoot_s=worst,
        longest_consecutive_run=longest,
    )


@dataclass
class LatencyHistogram:
    """Every sample stored, so percentiles are exact rather than estimated.

    Memory is ``O(N)`` -- 8 bytes per sample plus Python list overhead, about
    60 MB for 1e6 samples in CPython. That is the deliberate trade against a
    bucketed estimator such as ``hdrhistogram``, which is ``O(1)`` in memory
    with a bounded relative error. For a control loop logging 1 kHz for an
    hour (3.6e6 samples) this class is the wrong tool and the README says so.

    Attributes:
        samples: the stored values, insertion order preserved, units s by
            convention in this package.
        label: free-text label used in :meth:`summary`.
    """

    samples: list[float] = field(default_factory=list)
    label: str = "latency"

    def add(self, value_s: float) -> None:
        """Append one sample, units s.

        Raises:
            ValueError: if ``value_s`` is non-finite or negative.
        """
        v = float(value_s)
        if not math.isfinite(v):
            raise ValueError(f"value_s must be finite, got {value_s}")
        if v < 0.0:
            raise ValueError(f"value_s must be >= 0 s, got {value_s}")
        self.samples.append(v)

    def extend(self, values_s: Iterable[float]) -> None:
        """Append many samples, units s. Validates each."""
        for v in values_s:
            self.add(v)

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def count(self) -> int:
        """Number of stored samples."""
        return len(self.samples)

    def percentile(self, p: float, method: PercentileMethod = "nearest_rank") -> float:
        """Exact percentile of the stored samples; see :func:`percentile`."""
        return percentile(self.samples, p, method)

    def minimum(self) -> float:
        """Smallest stored sample, units s.

        Raises:
            ValueError: if empty.
        """
        self._require_nonempty()
        return min(self.samples)

    def maximum(self) -> float:
        """Largest stored sample, units s.

        Raises:
            ValueError: if empty.
        """
        self._require_nonempty()
        return max(self.samples)

    def mean(self) -> float:
        """Arithmetic mean, units s, summed with ``math.fsum``.

        Raises:
            ValueError: if empty.
        """
        self._require_nonempty()
        return math.fsum(self.samples) / len(self.samples)

    def stdev(self) -> float:
        """Sample standard deviation (``n-1`` denominator), units s.

        Raises:
            ValueError: if fewer than two samples.
        """
        if len(self.samples) < 2:
            raise ValueError("stdev requires at least 2 samples")
        m = self.mean()
        return math.sqrt(math.fsum((x - m) ** 2 for x in self.samples) / (len(self.samples) - 1))

    def overruns(self, budget_s: float) -> OverrunReport:
        """Overrun accounting against ``budget_s``; see :func:`overrun_report`."""
        return overrun_report(self.samples, budget_s)

    def bucket_counts(self, edges_s: Sequence[float]) -> list[int]:
        """Counts in half-open bins ``[edges[i], edges[i+1])``, plus two outer bins.

        The returned list has ``len(edges) + 1`` entries: the first counts
        samples below ``edges[0]``, the last counts samples at or above
        ``edges[-1]``. Binning is for plotting only; percentiles never go
        through it.

        Args:
            edges_s: strictly increasing bin edges, units s, at least 2.

        Returns:
            Counts, length ``len(edges_s) + 1``.

        Raises:
            ValueError: if fewer than 2 edges or not strictly increasing.
        """
        if len(edges_s) < 2:
            raise ValueError(f"edges_s must have at least 2 entries, got {len(edges_s)}")
        for a, b in zip(edges_s[:-1], edges_s[1:], strict=True):
            if not b > a:
                raise ValueError(f"edges_s must be strictly increasing, got {a} then {b}")
        counts = [0] * (len(edges_s) + 1)
        for x in self.samples:
            if x < edges_s[0]:
                counts[0] += 1
            elif x >= edges_s[-1]:
                counts[-1] += 1
            else:
                lo, hi = 0, len(edges_s) - 1
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    if x < edges_s[mid]:
                        hi = mid
                    else:
                        lo = mid
                counts[lo + 1] += 1
        return counts

    def summary(
        self,
        percentiles: Sequence[float] = (50.0, 90.0, 99.0, 99.9, 100.0),
        method: PercentileMethod = "nearest_rank",
    ) -> dict[str, float]:
        """Summary statistics, units s, with the percentile method in the keys.

        Keys are ``count``, ``min_s``, ``mean_s``, ``max_s`` and
        ``p<value>_<method>_s`` for each requested percentile, so a reader of
        the output can never be in doubt about which definition produced a
        number.

        Raises:
            ValueError: if empty.
        """
        self._require_nonempty()
        out: dict[str, float] = {
            "count": float(len(self.samples)),
            "min_s": self.minimum(),
            "mean_s": self.mean(),
            "max_s": self.maximum(),
        }
        for p in percentiles:
            key = f"p{p:g}".replace(".", "_")
            out[f"{key}_{method}_s"] = self.percentile(p, method)
        return out

    def _require_nonempty(self) -> None:
        if not self.samples:
            raise ValueError(f"histogram {self.label!r} is empty")
