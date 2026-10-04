"""Deadline arithmetic, latency histograms and overrun accounting.

Two overrun definitions, both computed
--------------------------------------
"Did that iteration overrun?" has two defensible answers and they disagree,
so both are computed and both are reported.

**Direct (independent deadlines).** Iteration ``i`` with measured duration
``d[i]`` overruns iff ``d[i] > T``. Each iteration is judged against the
period alone; lateness does not propagate. This is the definition a
stage-latency analysis uses.

**Cascade (release-time deadlines).** Iteration ``i`` is released at
``r[i] = i * T``, starts at ``s[i] = max(r[i], c[i-1])``, completes at
``c[i] = s[i] + d[i]`` and has deadline ``D[i] = r[i] + T``; it overruns iff
``c[i] > D[i]``. One late iteration delays the next, which is then late for a
deadline it would otherwise have met. This is what a real fixed-rate loop
does and it is the model in Buttazzo, *Hard Real-Time Computing Systems*, 3rd
ed., Springer 2011, §4.1 (periodic task model, implicit deadlines ``D = T``)
and in Liu & Layland 1973 (*JACM* 20(1):46-61) where ``D_i = T_i``.

Equality is **not** an overrun: a deadline met exactly at the deadline is met.

Percentiles
-----------
Both quantile conventions of Hyndman & Fan 1996 ("Sample Quantiles in
Statistical Packages", *The American Statistician* 50(4):361-365) that matter
here are implemented and named:

* ``"nearest_rank"`` — their Type 1, the inverse empirical CDF:
  ``Q(p) = x_(ceil(n p))``, with ``Q(0) = x_(1)``. Always an observed value;
  this is the convention HdrHistogram and most latency tooling use.
* ``"linear"`` — their Type 7, linear interpolation of order statistics:
  ``h = (n-1) p + 1``, ``Q(p) = x_(floor h) + (h - floor h)(x_(ceil h) -
  x_(floor h))``. This is what ``numpy.percentile`` returns by default.

Sampling error of a quantile estimate
-------------------------------------
For a continuous distribution with density ``f`` at the true quantile
``q_p``, the sample quantile is asymptotically normal with standard error

    se(q_p) = sqrt(p (1 - p) / n) / f(q_p)                            [s]

(Serfling, *Approximation Theorems of Mathematical Statistics*, Wiley 1980,
§2.3.3, Corollary 2.3.3B). Validity: ``f(q_p) > 0``, ``n`` large, i.i.d.
samples. :func:`quantile_standard_error` evaluates it; the latency
validation uses it as the tolerance rather than a number chosen to pass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .errors import ConfigurationError, TimebaseRegressionError

__all__ = [
    "LatencyHistogram",
    "MonotonicGuard",
    "OverrunAccount",
    "PeriodSpec",
    "TimingUncertainty",
    "overrun_report",
    "quantile_standard_error",
    "timing_uncertainty",
]

_QUANTILE_METHODS = ("nearest_rank", "linear")


@dataclass(frozen=True)
class PeriodSpec:
    """Period, deadline and overrun-cascade limit of a fixed-rate loop.

    Attributes
    ----------
    period_s:
        Loop period ``T`` [s], > 0.
    deadline_s:
        Relative deadline [s]. Defaults to ``period_s`` (implicit deadline,
        the usual periodic-task assumption). Must satisfy
        ``0 < deadline_s <= period_s``; a deadline longer than the period is
        a different task model and is rejected rather than half-supported.
    cascade_limit:
        Number of *consecutive* cascade overruns that aborts the run. ``0``
        disables the abort. Must be >= 0.
    """

    period_s: float
    deadline_s: float | None = None
    cascade_limit: int = 0

    def __post_init__(self) -> None:
        if not (self.period_s > 0.0):
            raise ConfigurationError(f"period_s must be > 0, got {self.period_s!r}")
        d = self.period_s if self.deadline_s is None else self.deadline_s
        if not (0.0 < d <= self.period_s):
            raise ConfigurationError(
                f"deadline_s must satisfy 0 < deadline_s <= period_s "
                f"({self.period_s}), got {self.deadline_s!r}"
            )
        if self.cascade_limit < 0:
            raise ConfigurationError(
                f"cascade_limit must be >= 0, got {self.cascade_limit!r}"
            )

    @property
    def effective_deadline_s(self) -> float:
        """Relative deadline [s], with the implicit-deadline default applied."""
        return self.period_s if self.deadline_s is None else float(self.deadline_s)

    @property
    def rate_hz(self) -> float:
        """``1 / period_s`` [Hz]."""
        return 1.0 / self.period_s

    def release_time_s(self, index: int) -> float:
        """Release time ``index * period_s`` [s] of iteration ``index``."""
        if index < 0:
            raise ConfigurationError(f"index must be >= 0, got {index!r}")
        return index * self.period_s

    def absolute_deadline_s(self, index: int) -> float:
        """``index * period_s + effective_deadline_s`` [s]."""
        return self.release_time_s(index) + self.effective_deadline_s


class MonotonicGuard:
    """Rejects a timestamp earlier than one already seen.

    The "timebase steps backwards" failure mode. A loop that accepts a
    backwards step computes a negative duration, which then fails to exceed
    any positive deadline, so the overrun is silently lost. The guard turns
    that into :class:`~hilforge.errors.TimebaseRegressionError`.

    Parameters
    ----------
    tolerance_s:
        Backwards steps no larger than this [s] are clamped to the last value
        instead of raising, for clocks whose last digit jitters. Default 0.0,
        i.e. any decrease raises. Must be >= 0.
    """

    def __init__(self, *, tolerance_s: float = 0.0) -> None:
        if tolerance_s < 0.0:
            raise ConfigurationError(f"tolerance_s must be >= 0, got {tolerance_s!r}")
        self._tol = float(tolerance_s)
        self._last: float | None = None
        self.clamped = 0

    @property
    def last(self) -> float | None:
        """Last accepted timestamp [s], or ``None`` before the first."""
        return self._last

    def check(self, t_s: float) -> float:
        """Accept ``t_s`` [s] and return the value to use.

        Raises
        ------
        TimebaseRegressionError
            If ``t_s`` is below the last accepted value by more than
            ``tolerance_s``.
        """
        if not math.isfinite(t_s):
            raise TimebaseRegressionError(f"timestamp is not finite: {t_s!r}")
        if self._last is None:
            self._last = float(t_s)
            return self._last
        drop = self._last - t_s
        if drop > self._tol:
            raise TimebaseRegressionError(
                f"timebase stepped backwards by {drop:.9e} s "
                f"(from {self._last:.9e} s to {t_s:.9e} s, tolerance {self._tol:.9e} s)"
            )
        if t_s < self._last:
            self.clamped += 1
            return self._last
        self._last = float(t_s)
        return self._last

    def reset(self) -> None:
        """Forget the history."""
        self._last = None
        self.clamped = 0


class LatencyHistogram:
    """Latency samples with exact and binned percentile modes.

    In ``exact`` mode every sample is stored and percentiles are computed from
    the sorted samples, so the only error is sampling error. In ``binned``
    mode samples are counted into fixed-width bins and a percentile is
    reported as the **upper edge** of the containing bin, so the quantisation
    error is bounded above by one bin width and is one-sided (never
    under-reports). Both are offered because a long embedded run cannot store
    every sample and a validation run must not approximate.

    Parameters
    ----------
    mode:
        ``"exact"`` (default) or ``"binned"``.
    bin_width_s:
        Bin width [s] for ``binned`` mode, > 0.
    n_bins:
        Number of bins for ``binned`` mode, >= 1. Samples at or above
        ``n_bins * bin_width_s`` land in an overflow counter and are reported.
    units:
        Units string carried into records; seconds by default.
    """

    def __init__(
        self,
        *,
        mode: str = "exact",
        bin_width_s: float = 1.0e-4,
        n_bins: int = 4096,
        units: str = "s",
    ) -> None:
        if mode not in ("exact", "binned"):
            raise ConfigurationError(f"mode must be 'exact' or 'binned', got {mode!r}")
        if not (bin_width_s > 0.0):
            raise ConfigurationError(f"bin_width_s must be > 0, got {bin_width_s!r}")
        if n_bins < 1:
            raise ConfigurationError(f"n_bins must be >= 1, got {n_bins!r}")
        self.mode = mode
        self.bin_width_s = float(bin_width_s)
        self.n_bins = int(n_bins)
        self.units = units
        self._samples: list[float] = []
        self._counts = np.zeros(self.n_bins, dtype=np.int64)
        self.overflow = 0
        self._n = 0
        self._min = math.inf
        self._max = -math.inf
        self._sum = 0.0
        self._sumsq = 0.0

    def __len__(self) -> int:
        return self._n

    @property
    def count(self) -> int:
        """Number of samples recorded."""
        return self._n

    @property
    def minimum(self) -> float:
        """Smallest sample [s]; ``inf`` if empty."""
        return self._min

    @property
    def maximum(self) -> float:
        """Largest sample [s]; ``-inf`` if empty."""
        return self._max

    @property
    def mean(self) -> float:
        """Arithmetic mean [s]; ``nan`` if empty."""
        return self._sum / self._n if self._n else math.nan

    @property
    def stdev(self) -> float:
        """Sample standard deviation [s] (n-1); ``nan`` if n < 2."""
        if self._n < 2:
            return math.nan
        var = (self._sumsq - self._n * self.mean**2) / (self._n - 1)
        return math.sqrt(max(var, 0.0))

    def add(self, value_s: float) -> None:
        """Record one latency sample [s]. Negative or non-finite is rejected."""
        v = float(value_s)
        if not math.isfinite(v):
            raise ValueError(f"latency sample must be finite, got {value_s!r}")
        if v < 0.0:
            raise ValueError(f"latency sample must be >= 0 s, got {value_s!r}")
        self._n += 1
        self._sum += v
        self._sumsq += v * v
        self._min = min(self._min, v)
        self._max = max(self._max, v)
        if self.mode == "exact":
            self._samples.append(v)
        else:
            idx = int(v // self.bin_width_s)
            if idx >= self.n_bins:
                self.overflow += 1
            else:
                self._counts[idx] += 1

    def extend(self, values_s) -> None:
        """Record many samples [s]."""
        for v in np.asarray(values_s, dtype=np.float64).ravel():
            self.add(float(v))

    def samples(self) -> np.ndarray:
        """Sorted copy of the stored samples [s]; empty in ``binned`` mode."""
        return np.sort(np.asarray(self._samples, dtype=np.float64))

    def percentile(self, p: float, *, method: str = "nearest_rank") -> float:
        """Quantile at probability ``p`` in ``[0, 1]`` [s].

        Parameters
        ----------
        p:
            Probability in ``[0, 1]``.
        method:
            ``"nearest_rank"`` (Hyndman & Fan Type 1) or ``"linear"``
            (Type 7). Ignored in ``binned`` mode, which always reports the
            containing bin's upper edge.
        """
        if not (0.0 <= p <= 1.0):
            raise ConfigurationError(f"p must be in [0, 1], got {p!r}")
        if method not in _QUANTILE_METHODS:
            raise ConfigurationError(
                f"method must be one of {_QUANTILE_METHODS}, got {method!r}"
            )
        if self._n == 0:
            raise ValueError("cannot take a percentile of an empty histogram")
        if self.mode == "binned":
            return self._binned_percentile(p)
        xs = self.samples()
        n = xs.size
        if method == "nearest_rank":
            if p == 0.0:
                return float(xs[0])
            rank = math.ceil(p * n)
            return float(xs[min(max(rank, 1), n) - 1])
        h = (n - 1) * p + 1.0
        lo = math.floor(h)
        hi = math.ceil(h)
        frac = h - lo
        return float(xs[lo - 1] + frac * (xs[hi - 1] - xs[lo - 1]))

    def _binned_percentile(self, p: float) -> float:
        target = math.ceil(p * self._n) if p > 0.0 else 1
        cum = 0
        for idx in range(self.n_bins):
            cum += int(self._counts[idx])
            if cum >= target:
                return (idx + 1) * self.bin_width_s
        return self.n_bins * self.bin_width_s

    def binned_percentile_error_bound_s(self) -> float:
        """Upper bound [s] on the binning error of a percentile.

        One bin width in ``binned`` mode; exactly zero in ``exact`` mode.
        """
        return self.bin_width_s if self.mode == "binned" else 0.0

    def summary(self, *, method: str = "nearest_rank") -> dict[str, float]:
        """Count, mean, stdev, min, max and p50/p90/p99/p99.9 [s]."""
        if self._n == 0:
            raise ValueError("cannot summarise an empty histogram")
        out = {
            "count": float(self._n),
            "mean_s": self.mean,
            "stdev_s": self.stdev,
            "min_s": self._min,
            "max_s": self._max,
        }
        for p, key in ((0.50, "p50_s"), (0.90, "p90_s"), (0.99, "p99_s"), (0.999, "p999_s")):
            out[key] = self.percentile(p, method=method)
        return out


def quantile_standard_error(p: float, n: int, density_at_quantile: float) -> float:
    """Asymptotic standard error [s] of a sample quantile.

    ``se = sqrt(p (1 - p) / n) / f(q_p)`` (Serfling 1980, §2.3.3). Validity:
    ``0 < p < 1``, ``f(q_p) > 0``, large ``n``, i.i.d. samples.

    Parameters
    ----------
    p:
        Probability, strictly inside ``(0, 1)``.
    n:
        Sample size, >= 1.
    density_at_quantile:
        ``f(q_p)`` [1/s], > 0.
    """
    if not (0.0 < p < 1.0):
        raise ConfigurationError(f"p must be in (0, 1), got {p!r}")
    if n < 1:
        raise ConfigurationError(f"n must be >= 1, got {n!r}")
    if not (density_at_quantile > 0.0):
        raise ConfigurationError(
            f"density_at_quantile must be > 0, got {density_at_quantile!r}"
        )
    return math.sqrt(p * (1.0 - p) / n) / density_at_quantile


@dataclass
class OverrunAccount:
    """Per-iteration overrun accounting under both definitions.

    Attributes
    ----------
    period_s:
        Period ``T`` [s] used for the accounting.
    deadline_s:
        Relative deadline [s].
    direct_indices:
        Iterations with ``d[i] > deadline_s``.
    cascade_indices:
        Iterations whose completion exceeded ``i * T + deadline_s``.
    completion_s:
        ``c[i]`` [s] for every iteration, on the release-time timeline.
    lateness_s:
        ``c[i] - D[i]`` [s]; negative means the deadline was met early.
    max_consecutive_cascade:
        Longest run of consecutive cascade overruns.
    first_cascade_run_start:
        Index at which that longest run began, or ``-1`` if there were none.
    """

    period_s: float
    deadline_s: float
    direct_indices: list[int] = field(default_factory=list)
    cascade_indices: list[int] = field(default_factory=list)
    completion_s: list[float] = field(default_factory=list)
    lateness_s: list[float] = field(default_factory=list)
    max_consecutive_cascade: int = 0
    first_cascade_run_start: int = -1

    @property
    def n_iterations(self) -> int:
        """Number of iterations accounted."""
        return len(self.completion_s)

    @property
    def direct_count(self) -> int:
        """``#{i : d[i] > deadline_s}``."""
        return len(self.direct_indices)

    @property
    def cascade_count(self) -> int:
        """``#{i : c[i] > i * T + deadline_s}``."""
        return len(self.cascade_indices)

    @property
    def direct_rate(self) -> float:
        """Direct overruns per iteration [-]; ``nan`` for an empty account."""
        return self.direct_count / self.n_iterations if self.n_iterations else math.nan

    @property
    def cascade_rate(self) -> float:
        """Cascade overruns per iteration [-]; ``nan`` for an empty account."""
        return self.cascade_count / self.n_iterations if self.n_iterations else math.nan

    def as_dict(self) -> dict[str, object]:
        """JSON-serialisable summary, for the cross-check record."""
        return {
            "period_s": self.period_s,
            "deadline_s": self.deadline_s,
            "n_iterations": self.n_iterations,
            "direct_overrun_count": self.direct_count,
            "direct_overrun_indices": list(self.direct_indices),
            "cascade_overrun_count": self.cascade_count,
            "cascade_overrun_indices": list(self.cascade_indices),
            "max_consecutive_cascade": self.max_consecutive_cascade,
        }


def overrun_report(
    durations_s,
    period_s: float,
    *,
    deadline_s: float | None = None,
) -> OverrunAccount:
    """Account overruns in a sequence of iteration durations.

    Parameters
    ----------
    durations_s:
        Per-iteration durations ``d[i]`` [s], all finite and >= 0.
    period_s:
        Period ``T`` [s], > 0.
    deadline_s:
        Relative deadline [s]; defaults to ``period_s``. Must be in
        ``(0, period_s]``.

    Returns
    -------
    OverrunAccount
        Both definitions, with completions and lateness per iteration.

    Notes
    -----
    Equality (``d[i] == deadline_s``, ``c[i] == D[i]``) is a met deadline, not
    an overrun. The cascade recursion is
    ``c[i] = max(i*T, c[i-1]) + d[i]`` with ``c[-1] = 0``.
    """
    spec = PeriodSpec(period_s=period_s, deadline_s=deadline_s)
    d = np.asarray(durations_s, dtype=np.float64).ravel()
    if d.size and (not np.all(np.isfinite(d)) or np.any(d < 0.0)):
        raise ValueError("durations_s must be finite and >= 0 s")
    acc = OverrunAccount(period_s=spec.period_s, deadline_s=spec.effective_deadline_s)
    prev_completion = 0.0
    run = 0
    best_run = 0
    best_start = -1
    run_start = -1
    for i, di in enumerate(d):
        if di > spec.effective_deadline_s:
            acc.direct_indices.append(i)
        start = max(spec.release_time_s(i), prev_completion)
        completion = start + float(di)
        deadline = spec.absolute_deadline_s(i)
        acc.completion_s.append(completion)
        acc.lateness_s.append(completion - deadline)
        if completion > deadline:
            acc.cascade_indices.append(i)
            if run == 0:
                run_start = i
            run += 1
            if run > best_run:
                best_run = run
                best_start = run_start
        else:
            run = 0
        prev_completion = completion
    acc.max_consecutive_cascade = best_run
    acc.first_cascade_run_start = best_start
    return acc


@dataclass(frozen=True)
class TimingUncertainty:
    """Combined standard uncertainty of a mean duration.

    Terms, combined in quadrature per JCGM 100:2008 §5.1.2 (independent
    inputs):

    * ``u_statistical`` = ``s / sqrt(n)`` [s], the Type A term from the
      observed scatter (JCGM 100:2008 §4.2).
    * ``u_resolution`` = ``delta / sqrt(6 n)`` [s], the Type B term from clock
      quantisation of ``n`` independent two-timestamp durations, each
      contributing ``delta**2 / 6`` (Bennett 1948; JCGM 100:2008 §4.3.7 for
      the uniform-distribution Type B evaluation).

    ``call_overhead_s`` is reported separately because it is a **bias**, not
    an uncertainty: it shifts every duration the same way and must not be
    added in quadrature.

    ``expanded_k2`` is ``2 * combined`` — a coverage factor ``k = 2``, about
    95 % for a normal distribution (JCGM 100:2008 §6.3.3).
    """

    n: int
    mean_s: float
    stdev_s: float
    resolution_s: float
    u_statistical_s: float
    u_resolution_s: float
    combined_s: float
    expanded_k2_s: float
    call_overhead_s: float

    def as_text(self) -> str:
        """Human-readable uncertainty budget."""
        return (
            f"n                        : {self.n}\n"
            f"mean                     : {self.mean_s:.9e} s\n"
            f"sample stdev             : {self.stdev_s:.9e} s\n"
            f"clock resolution (delta) : {self.resolution_s:.9e} s\n"
            f"u_A statistical s/sqrt(n): {self.u_statistical_s:.9e} s\n"
            f"u_B resolution  d/sqrt(6n): {self.u_resolution_s:.9e} s\n"
            f"u_c combined (quadrature): {self.combined_s:.9e} s\n"
            f"U expanded (k=2)         : {self.expanded_k2_s:.9e} s\n"
            f"bias: timing-call overhead: {self.call_overhead_s:.9e} s "
            f"(reported, not subtracted)"
        )


def timing_uncertainty(
    durations_s,
    *,
    resolution_s: float,
    call_overhead_s: float = 0.0,
) -> TimingUncertainty:
    """Uncertainty budget for the mean of a set of measured durations.

    Parameters
    ----------
    durations_s:
        Measured durations [s], at least 2, finite.
    resolution_s:
        Clock step ``delta`` [s], > 0 — measure it with
        :func:`hilforge.timebase.measure_resolution` rather than assuming it.
    call_overhead_s:
        Measured bias of one timing-call pair [s], >= 0.
    """
    d = np.asarray(durations_s, dtype=np.float64).ravel()
    if d.size < 2:
        raise ConfigurationError(f"need at least 2 durations, got {d.size}")
    if not np.all(np.isfinite(d)):
        raise ValueError("durations_s must all be finite")
    if not (resolution_s > 0.0):
        raise ConfigurationError(f"resolution_s must be > 0, got {resolution_s!r}")
    if call_overhead_s < 0.0:
        raise ConfigurationError(
            f"call_overhead_s must be >= 0, got {call_overhead_s!r}"
        )
    n = int(d.size)
    mean = float(d.mean())
    stdev = float(d.std(ddof=1))
    u_a = stdev / math.sqrt(n)
    u_b = resolution_s / math.sqrt(6.0 * n)
    u_c = math.hypot(u_a, u_b)
    return TimingUncertainty(
        n=n,
        mean_s=mean,
        stdev_s=stdev,
        resolution_s=float(resolution_s),
        u_statistical_s=u_a,
        u_resolution_s=u_b,
        combined_s=u_c,
        expanded_k2_s=2.0 * u_c,
        call_overhead_s=float(call_overhead_s),
    )
