"""Benchmark harness: repeated timing with the tail reported separately.

The harness exists because a control loop is sized by its worst case. A median
latency is the number that looks good in a table and the number that does not
protect a deadline, so :class:`LatencyProfile` reports p50, p90, p99, the
maximum and the mean, and every consumer of a profile has to choose which one
it means.

Method
------
One repeat is one ``perf_counter`` bracket around one call. Repeats are
preceded by ``warmup`` calls whose timings are discarded, because the first
calls into ``onnxruntime`` pay session-level lazy initialisation and a
first-touch page-fault cost that is not part of steady-state latency. The
warmup count is reported, not hidden.

Garbage collection is disabled for the duration of a timed batch and
re-enabled afterwards, because a collection landing inside a repeat produces
a latency unrelated to the callable. This removes one source of tail; the
remaining tail --- scheduler preemption on a shared host --- is real and is
reported as measured.

Everything a number needs to be interpreted travels with it: the repeat count,
the warmup count, the clock and its resolution, and the environment. There is
no code path in this package that produces a bare latency.
"""

from __future__ import annotations

import gc
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from edgeinfer.environment import EnvironmentRecord, environment_record
from edgeinfer.uncertainty import UncertaintyBudget, timer_overhead_s, uncertainty_budget

__all__ = ["LatencyProfile", "benchmark", "time_calls"]


@dataclass(frozen=True)
class LatencyProfile:
    """Latency statistics of a timed batch, tail separated from the centre.

    Attributes
    ----------
    label
        What was timed.
    samples_s
        Every retained repeat [s], in execution order. Kept so a caller can
        re-derive any statistic and so a histogram can be plotted without
        re-running.
    n_repeats, n_warmup
        Retained repeat count and discarded warm-up count.
    clock, clock_resolution_s
        Timer name and resolution [s].
    timer_bias_s
        Measured cost of one timer pair [s], a bias on short intervals.
    gc_disabled
        Whether garbage collection was disabled during the batch.
    environment
        Machine the batch ran on.
    """

    label: str
    samples_s: np.ndarray
    n_repeats: int
    n_warmup: int
    clock: str
    clock_resolution_s: float
    timer_bias_s: float
    gc_disabled: bool
    environment: EnvironmentRecord
    extra: dict[str, object] = field(default_factory=dict)

    # -- statistics ------------------------------------------------------
    @property
    def mean_s(self) -> float:
        """Arithmetic mean [s]."""
        return float(np.mean(self.samples_s))

    @property
    def std_s(self) -> float:
        """Sample standard deviation, ``ddof=1`` [s]; ``nan`` for n < 2."""
        if self.samples_s.size < 2:
            return float("nan")
        return float(np.std(self.samples_s, ddof=1))

    @property
    def min_s(self) -> float:
        """Smallest retained repeat [s]."""
        return float(np.min(self.samples_s))

    @property
    def max_s(self) -> float:
        """Largest retained repeat [s]: the observed worst case."""
        return float(np.max(self.samples_s))

    def quantile_s(self, q: float) -> float:
        """Sample quantile [s], linear interpolation (NumPy default)."""
        if not 0.0 <= q <= 1.0:
            raise ValueError(f"quantile must lie in [0, 1], got {q}")
        return float(np.quantile(self.samples_s, q))

    @property
    def p50_s(self) -> float:
        """Median [s]."""
        return self.quantile_s(0.50)

    @property
    def p90_s(self) -> float:
        """90th percentile [s]."""
        return self.quantile_s(0.90)

    @property
    def p99_s(self) -> float:
        """99th percentile [s]. With n < 100 repeats this is interpolated
        between the top two order statistics and is not a reliable tail
        estimate; the repeat count is reported alongside it for that reason."""
        return self.quantile_s(0.99)

    @property
    def tail_ratio(self) -> float:
        """``p99 / p50`` [dimensionless]: how much worse the tail is."""
        median = self.p50_s
        if median <= 0:
            return float("nan")
        return self.p99_s / median

    def uncertainty(
        self, statistic: str = "mean", *, n_resamples: int = 400, seed: int = 0
    ) -> UncertaintyBudget:
        """Uncertainty budget for one statistic.

        Parameters
        ----------
        statistic
            ``"mean"``, ``"p50"``, ``"p90"``, ``"p99"``, or ``"pNN"`` for any
            integer or decimal NN.
        n_resamples, seed
            Bootstrap settings for a quantile statistic.
        """
        method = (
            f"{self.clock} bracket per call, {self.n_repeats} repeats after "
            f"{self.n_warmup} warm-up calls, gc_disabled={self.gc_disabled}"
        )
        if statistic == "mean":
            return uncertainty_budget(
                self.samples_s,
                "mean",
                self.clock_resolution_s,
                timer_bias_s=self.timer_bias_s,
                method=method,
            )
        if statistic.startswith("p"):
            try:
                q = float(statistic[1:]) / 100.0
            except ValueError:
                raise ValueError(
                    f"statistic must be 'mean' or 'pNN', got {statistic!r}"
                ) from None
            return uncertainty_budget(
                self.samples_s,
                statistic,
                self.clock_resolution_s,
                quantile=q,
                n_resamples=n_resamples,
                seed=seed,
                timer_bias_s=self.timer_bias_s,
                method=method,
            )
        raise ValueError(f"statistic must be 'mean' or 'pNN', got {statistic!r}")

    def method_line(self) -> str:
        """One-line measurement method, for a report cell."""
        return (
            f"{self.clock} per-call bracket; n={self.n_repeats} repeats, "
            f"{self.n_warmup} warm-up discarded; clock resolution "
            f"{self.clock_resolution_s:.3g} s; timer-pair bias "
            f"{self.timer_bias_s * 1e9:.0f} ns; gc disabled={self.gc_disabled}"
        )

    def summary_lines(self) -> list[str]:
        """Human-readable profile, tail first because the tail is the point."""
        return [
            f"label                   : {self.label}",
            f"p99 (worst-case proxy)  : {self.p99_s * 1e6:.3f} us",
            f"max (observed worst)    : {self.max_s * 1e6:.3f} us",
            f"p90                     : {self.p90_s * 1e6:.3f} us",
            f"p50 (median)            : {self.p50_s * 1e6:.3f} us",
            f"mean                    : {self.mean_s * 1e6:.3f} us",
            f"sd                      : {self.std_s * 1e6:.3f} us",
            f"min                     : {self.min_s * 1e6:.3f} us",
            f"tail ratio p99/p50      : {self.tail_ratio:.3f}",
            f"method                  : {self.method_line()}",
            f"environment             : {self.environment.one_line()}",
        ]


def time_calls(
    fn: Callable[..., Any],
    *args: Any,
    repeats: int,
    warmup: int = 0,
    disable_gc: bool = True,
    **kwargs: Any,
) -> np.ndarray:
    """Time ``repeats`` calls of ``fn`` and return the per-call times [s].

    Parameters
    ----------
    fn
        Callable under test; its return value is discarded.
    repeats
        Number of retained repeats, >= 1.
    warmup
        Calls made and discarded first, >= 0.
    disable_gc
        Disable garbage collection during the timed batch.

    Returns
    -------
    numpy.ndarray
        Shape ``(repeats,)``, per-call wall time [s], in execution order.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0, got {warmup}")
    for _ in range(warmup):
        fn(*args, **kwargs)
    out = np.empty(repeats, dtype=float)
    counter = time.perf_counter
    gc_was_enabled = gc.isenabled()
    if disable_gc:
        gc.disable()
    try:
        for i in range(repeats):
            t0 = counter()
            fn(*args, **kwargs)
            out[i] = counter() - t0
    finally:
        if disable_gc and gc_was_enabled:
            gc.enable()
    return out


def benchmark(
    fn: Callable[..., Any],
    *args: Any,
    label: str,
    repeats: int = 200,
    warmup: int = 10,
    disable_gc: bool = True,
    shared_host: bool = True,
    environment_note: str = "",
    timer_bias_samples: int = 2000,
    extra: dict[str, object] | None = None,
    **kwargs: Any,
) -> LatencyProfile:
    """Time ``fn`` and return a full :class:`LatencyProfile`.

    Parameters
    ----------
    fn
        Callable under test.
    label
        What is being timed; appears in every report.
    repeats, warmup
        Retained and discarded call counts.
    disable_gc
        Disable garbage collection during the timed batch.
    shared_host
        Passed to :func:`edgeinfer.environment.environment_record`; leave
        ``True`` unless the host is known to be dedicated.
    environment_note
        Appended to the environment line, e.g. a statement that these are
        container numbers.
    timer_bias_samples
        Repeats used to measure the timer-pair overhead.
    extra
        Arbitrary key/value pairs carried into the profile, e.g. the input
        shape or the backend name.
    """
    bias = timer_overhead_s(timer_bias_samples)
    samples = time_calls(
        fn, *args, repeats=repeats, warmup=warmup, disable_gc=disable_gc, **kwargs
    )
    env = environment_record(shared_host=shared_host, note=environment_note)
    return LatencyProfile(
        label=label,
        samples_s=samples,
        n_repeats=repeats,
        n_warmup=warmup,
        clock="perf_counter",
        clock_resolution_s=env.clock_resolution_s,
        timer_bias_s=bias,
        gc_disabled=disable_gc,
        environment=env,
        extra=dict(extra or {}),
    )
