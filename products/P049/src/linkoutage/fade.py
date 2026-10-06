"""Fade statistics from an amplitude series and a threshold.

Everything in this module is deterministic: given the same array, the same
threshold and the same :class:`FadeDefinitions`, the numbers do not move.

Why the definitions are a first-class object
--------------------------------------------
The quantities below -- down-crossing count, level-crossing rate, mean fade
duration, outage fraction -- are not uniquely defined by their names. Four
choices change the answer by several percent on a realistic series, and they
are the usual reason two tools disagree:

1. whether ``a[n] == threshold`` is inside the fade (strict vs. non-strict),
2. how a fade run's *duration* is converted from a sample count,
3. what happens to a fade that is still in progress at either end of the
   record (censoring),
4. whether a single-sample excursion below the threshold is a fade at all.

This module makes all four explicit, defaults to the conventions stated in
:data:`DEFAULT_DEFINITIONS`, and prints them alongside every result.

References
----------
S. O. Rice, "Mathematical analysis of random noise", *Bell System Technical
Journal* 23(3):282-332 (1944) and 24(1):46-156 (1945). The level-crossing
problem and the relation between the crossing rate of a level, the fraction of
time spent below it and the mean duration of an excursion. The relation used
here as an internal consistency check,

    mean fade duration = (fraction of time below) / (down-crossing rate),

is the discrete-record form of that result: it is exact for any stationary
binary sequence in the limit of a long record, because the time below the level
is partitioned exactly into the fade runs.

L. C. Andrews and R. L. Phillips, *Laser Beam Propagation through Random
Media*, 2nd ed., SPIE Press (2005). The lognormal amplitude model whose fade
statistics this module is normally applied to; see :mod:`linkoutage.channel`.

Units
-----
Amplitude and threshold: same arbitrary linear unit (the module never assumes
volts, square-root-watts or normalised amplitude; it only compares them).
Sample rate ``fs``: Hz. Durations: seconds. Rates: Hz.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "DEFAULT_DEFINITIONS",
    "FadeDefinitions",
    "FadeRuns",
    "FadeStatistics",
    "availability",
    "below_threshold",
    "down_crossing_indices",
    "fade_durations",
    "fade_runs",
    "fade_statistics",
    "level_crossing_rate",
    "mean_fade_duration",
    "outage_fraction",
    "up_crossing_indices",
]

DurationConvention = Literal["sample_count", "interval_count", "interpolated"]
CensoringRule = Literal["exclude", "include_as_complete"]
RecordDurationConvention = Literal["intervals", "samples"]


@dataclass(frozen=True)
class FadeDefinitions:
    """The four definitional choices that fix every number in this module.

    Attributes
    ----------
    strict_below:
        ``True``  -- sample ``n`` is in fade iff ``a[n] < threshold``.
        ``False`` -- sample ``n`` is in fade iff ``a[n] <= threshold``.
        Default ``True``. On float data the difference is almost always nil;
        it is not nil on quantised data, which is what a real receiver logs.
    duration_convention:
        How a run of ``L`` consecutive below-threshold samples becomes a
        duration in seconds.

        ``"sample_count"``   -- ``L / fs``. The run occupies ``L`` sample
        periods. This is the default and the published convention.
        ``"interval_count"`` -- ``(L - 1) / fs``. The elapsed time between the
        first and last below-threshold sample. Makes a single-sample fade zero
        seconds long, which is why it is not the default.
        ``"interpolated"``   -- the elapsed time between the linearly
        interpolated down-crossing and up-crossing instants, i.e. the two
        times at which the straight line joining consecutive samples equals
        the threshold. Only available for runs that are complete at both ends.
    censoring:
        What to do with a run that touches sample 0 (left-censored: it began
        before the record) or the last sample (right-censored: it had not
        ended when the record stopped).

        ``"exclude"`` -- censored runs are counted and reported but contribute
        no duration to the mean or to the fitted distributions. Default.
        ``"include_as_complete"`` -- censored runs are treated as if they had
        ended at the record boundary. This biases the mean downwards and is
        provided only so the size of that bias can be measured.
    count_single_sample_fades:
        ``True`` (default) -- a run of length 1 is a fade. ``False`` -- runs of
        length 1 are discarded entirely: they are not fades, their samples are
        still counted in the outage fraction, and the down-crossings that begin
        them are not counted in the level-crossing rate.
    record_duration_convention:
        The denominator of the level-crossing rate for an ``N``-sample record.
        ``"intervals"`` (default) -- ``(N - 1) / fs``, the number of
        consecutive sample pairs, each of which is one opportunity for a
        crossing. ``"samples"`` -- ``N / fs``. The two differ by a factor
        ``N / (N - 1)``; at ``N = 2e6`` that is 5e-7 relative, far below any
        cross-check tolerance, but it is stated rather than left implicit.
    """

    strict_below: bool = True
    duration_convention: DurationConvention = "sample_count"
    censoring: CensoringRule = "exclude"
    count_single_sample_fades: bool = True
    record_duration_convention: RecordDurationConvention = "intervals"

    def __post_init__(self) -> None:
        if self.duration_convention not in ("sample_count", "interval_count", "interpolated"):
            raise ValueError(
                "duration_convention must be 'sample_count', 'interval_count' or "
                f"'interpolated', got {self.duration_convention!r}"
            )
        if self.censoring not in ("exclude", "include_as_complete"):
            raise ValueError(
                "censoring must be 'exclude' or 'include_as_complete', "
                f"got {self.censoring!r}"
            )
        if self.record_duration_convention not in ("intervals", "samples"):
            raise ValueError(
                "record_duration_convention must be 'intervals' or 'samples', "
                f"got {self.record_duration_convention!r}"
            )
        if self.censoring == "include_as_complete" and self.duration_convention == "interpolated":
            raise ValueError(
                "duration_convention='interpolated' is undefined for a censored run "
                "(there is no crossing to interpolate at the record boundary); use "
                "censoring='exclude'"
            )

    def describe(self) -> str:
        """One multi-line block stating the conventions in words.

        Printed verbatim into every validation output so that a disagreement
        with another implementation can be traced to a definition rather than
        argued about.
        """
        below = "a[n] < T (strict)" if self.strict_below else "a[n] <= T (non-strict)"
        dur = {
            "sample_count": "duration = L / fs for a run of L below-threshold samples",
            "interval_count": "duration = (L - 1) / fs for a run of L below-threshold samples",
            "interpolated": (
                "duration = interpolated up-crossing instant minus interpolated "
                "down-crossing instant"
            ),
        }[self.duration_convention]
        cens = {
            "exclude": (
                "a run touching sample 0 or sample N-1 is censored: counted, reported, "
                "and excluded from the mean and from any distribution fit"
            ),
            "include_as_complete": (
                "a run touching sample 0 or sample N-1 is truncated at the record "
                "boundary and treated as complete (downward-biased, diagnostic only)"
            ),
        }[self.censoring]
        single = (
            "a single below-threshold sample is a fade of duration 1/fs"
            if self.count_single_sample_fades
            else "runs of length 1 are not fades and their down-crossings are not counted"
        )
        rec = {
            "intervals": "record duration = (N - 1) / fs",
            "samples": "record duration = N / fs",
        }[self.record_duration_convention]
        return "\n".join(
            [
                "Fade definitions in force:",
                f"  in fade          : {below}",
                "  down-crossing    : a[n-1] not in fade AND a[n] in fade, for n = 1..N-1",
                "  up-crossing      : a[n-1] in fade AND a[n] not in fade, for n = 1..N-1",
                "  fade run         : a maximal run of consecutive in-fade samples",
                f"  duration         : {dur}",
                f"  censoring        : {cens}",
                f"  single sample    : {single}",
                f"  rate denominator : {rec}",
                "  level-crossing rate = (counted down-crossings) / (record duration)",
                "  outage fraction      = (in-fade samples) / N, over ALL samples including",
                "                         censored runs and discarded single-sample runs",
                "  availability         = 1 - outage fraction",
            ]
        )

    def with_(self, **changes: object) -> FadeDefinitions:
        """Return a copy with the given fields replaced."""
        return replace(self, **changes)  # type: ignore[arg-type]


DEFAULT_DEFINITIONS = FadeDefinitions()
"""The published conventions: strict below, ``L / fs``, censored runs excluded,
single-sample excursions counted, rate over ``(N - 1) / fs``."""


def _as_amplitude(amplitude: ArrayLike) -> NDArray[np.float64]:
    arr = np.asarray(amplitude, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"amplitude must be one-dimensional, got shape {arr.shape}")
    if arr.size < 2:
        raise ValueError(f"amplitude must have at least 2 samples, got {arr.size}")
    if not np.all(np.isfinite(arr)):
        n_bad = int(np.count_nonzero(~np.isfinite(arr)))
        raise ValueError(
            f"amplitude contains {n_bad} non-finite value(s); a fade statistic over "
            "NaN or inf is undefined. Mask or interpolate the gaps first."
        )
    return arr


def _check_threshold(threshold: float) -> float:
    t = float(threshold)
    if not np.isfinite(t):
        raise ValueError(f"threshold must be finite, got {t}")
    return t


def _check_fs(fs: float) -> float:
    f = float(fs)
    if not np.isfinite(f) or f <= 0.0:
        raise ValueError(f"fs must be a positive finite sample rate in Hz, got {fs!r}")
    return f


def below_threshold(
    amplitude: ArrayLike,
    threshold: float,
    *,
    definitions: FadeDefinitions = DEFAULT_DEFINITIONS,
) -> NDArray[np.bool_]:
    """Boolean in-fade indicator, one element per sample.

    Parameters
    ----------
    amplitude:
        Amplitude (or any monotone proxy for it) in arbitrary linear units.
    threshold:
        Same units as ``amplitude``.
    definitions:
        Only :attr:`FadeDefinitions.strict_below` is consulted.
    """
    arr = _as_amplitude(amplitude)
    t = _check_threshold(threshold)
    return arr < t if definitions.strict_below else arr <= t


def _transition_bounds(mask: NDArray[np.bool_]) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return ``(starts, stops)`` of maximal ``True`` runs, ``stops`` exclusive."""
    changes = np.flatnonzero(np.diff(mask.astype(np.int8)) != 0) + 1
    bounds = np.concatenate(
        (
            np.zeros(1, dtype=np.int64),
            changes.astype(np.int64),
            np.array([mask.size], dtype=np.int64),
        )
    )
    starts = bounds[:-1]
    stops = bounds[1:]
    keep = mask[starts]
    return starts[keep], stops[keep]


def down_crossing_indices(
    amplitude: ArrayLike,
    threshold: float,
    *,
    definitions: FadeDefinitions = DEFAULT_DEFINITIONS,
) -> NDArray[np.int64]:
    """Indices ``n`` with ``a[n-1]`` not in fade and ``a[n]`` in fade.

    The returned index is the index of the *first below-threshold sample*, not
    of the last above-threshold one. A record that starts already below the
    threshold yields no down-crossing at index 0: the crossing happened before
    the record began.

    When ``definitions.count_single_sample_fades`` is ``False`` the
    down-crossings that open a length-1 run are omitted.
    """
    mask = below_threshold(amplitude, threshold, definitions=definitions)
    starts, stops = _transition_bounds(mask)
    keep = starts > 0
    if not definitions.count_single_sample_fades:
        keep &= (stops - starts) > 1
    return starts[keep].astype(np.int64)


def up_crossing_indices(
    amplitude: ArrayLike,
    threshold: float,
    *,
    definitions: FadeDefinitions = DEFAULT_DEFINITIONS,
) -> NDArray[np.int64]:
    """Indices ``n`` with ``a[n-1]`` in fade and ``a[n]`` not in fade.

    The returned index is the index of the first sample back above the
    threshold. A record that ends below the threshold yields no final
    up-crossing.
    """
    mask = below_threshold(amplitude, threshold, definitions=definitions)
    all_starts, all_stops = _transition_bounds(mask)
    if not definitions.count_single_sample_fades:
        keep = (all_stops - all_starts) > 1
        all_starts, all_stops = all_starts[keep], all_stops[keep]
    return all_stops[all_stops < mask.size].astype(np.int64)


@dataclass(frozen=True)
class FadeRuns:
    """Every maximal below-threshold run in a record.

    Attributes
    ----------
    start:
        Index of the first in-fade sample of each run.
    stop:
        Index one past the last in-fade sample of each run (exclusive).
    length_samples:
        ``stop - start``.
    left_censored:
        ``start == 0``: the run was already under way when the record began.
    right_censored:
        ``stop == n_samples``: the run had not ended when the record stopped.
    n_samples:
        Length of the record the runs came from.
    fs_hz:
        Sample rate, Hz.
    definitions:
        The definitions used to extract the runs.
    """

    start: NDArray[np.int64]
    stop: NDArray[np.int64]
    left_censored: NDArray[np.bool_]
    right_censored: NDArray[np.bool_]
    n_samples: int
    fs_hz: float
    definitions: FadeDefinitions
    down_crossing_time_s: NDArray[np.float64]
    up_crossing_time_s: NDArray[np.float64]

    @property
    def length_samples(self) -> NDArray[np.int64]:
        return (self.stop - self.start).astype(np.int64)

    @property
    def censored(self) -> NDArray[np.bool_]:
        """Either end censored."""
        return self.left_censored | self.right_censored

    @property
    def n_runs(self) -> int:
        return int(self.start.size)

    def durations_s(self) -> NDArray[np.float64]:
        """Duration of every run, in seconds, under the active convention.

        Censored runs are included here regardless of the censoring rule; use
        :meth:`complete_durations_s` for the set the mean is taken over.
        """
        conv = self.definitions.duration_convention
        length = self.length_samples.astype(np.float64)
        if conv == "sample_count":
            return length / self.fs_hz
        if conv == "interval_count":
            return (length - 1.0) / self.fs_hz
        return self.up_crossing_time_s - self.down_crossing_time_s

    def complete_durations_s(self) -> NDArray[np.float64]:
        """Durations entering the mean and the distribution fits."""
        d = self.durations_s()
        if self.definitions.censoring == "exclude":
            return d[~self.censored]
        return d

    def censored_durations_s(self) -> NDArray[np.float64]:
        """Observed (lower-bound) durations of the censored runs.

        Under ``censoring='include_as_complete'`` this is empty, because those
        runs were folded into :meth:`complete_durations_s`.
        """
        if self.definitions.censoring == "include_as_complete":
            return np.empty(0, dtype=np.float64)
        return self.durations_s()[self.censored]


def fade_runs(
    amplitude: ArrayLike,
    threshold: float,
    fs_hz: float,
    *,
    definitions: FadeDefinitions = DEFAULT_DEFINITIONS,
) -> FadeRuns:
    """Extract every below-threshold run with its censoring flags.

    Parameters
    ----------
    amplitude:
        One-dimensional, finite, at least 2 samples.
    threshold:
        Fade threshold, same units as ``amplitude``.
    fs_hz:
        Sample rate in Hz, used only to convert sample counts to seconds.
    definitions:
        See :class:`FadeDefinitions`.

    Notes
    -----
    Interpolated crossing instants are defined only where the sample either
    side of the crossing exists. For a left-censored run the down-crossing
    instant is reported as ``start / fs``; for a right-censored run the
    up-crossing instant is reported as ``stop / fs``. Those two values are
    placeholders, never used by the default (``"exclude"``) censoring rule.
    """
    arr = _as_amplitude(amplitude)
    t = _check_threshold(threshold)
    fs = _check_fs(fs_hz)
    mask = below_threshold(arr, t, definitions=definitions)
    starts, stops = _transition_bounds(mask)
    if not definitions.count_single_sample_fades:
        keep = (stops - starts) > 1
        starts, stops = starts[keep], stops[keep]

    n = arr.size
    left = starts == 0
    right = stops == n

    # Linear interpolation of the instant at which the straight line through
    # (n-1, a[n-1]) and (n, a[n]) equals the threshold.
    down_t = starts.astype(np.float64) / fs
    if np.any(~left):
        i = starts[~left]
        prev = arr[i - 1]
        cur = arr[i]
        denom = prev - cur
        frac = np.where(denom != 0.0, (prev - t) / np.where(denom != 0.0, denom, 1.0), 0.0)
        down_t[~left] = ((i - 1).astype(np.float64) + frac) / fs
    up_t = stops.astype(np.float64) / fs
    if np.any(~right):
        j = stops[~right]
        prev = arr[j - 1]
        cur = arr[j]
        denom = cur - prev
        frac = np.where(denom != 0.0, (t - prev) / np.where(denom != 0.0, denom, 1.0), 0.0)
        up_t[~right] = ((j - 1).astype(np.float64) + frac) / fs

    return FadeRuns(
        start=starts.astype(np.int64),
        stop=stops.astype(np.int64),
        left_censored=left,
        right_censored=right,
        n_samples=int(n),
        fs_hz=fs,
        definitions=definitions,
        down_crossing_time_s=down_t,
        up_crossing_time_s=up_t,
    )


def fade_durations(
    amplitude: ArrayLike,
    threshold: float,
    fs_hz: float,
    *,
    definitions: FadeDefinitions = DEFAULT_DEFINITIONS,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``(complete_durations_s, censored_lower_bounds_s)``."""
    runs = fade_runs(amplitude, threshold, fs_hz, definitions=definitions)
    return runs.complete_durations_s(), runs.censored_durations_s()


def record_duration_s(n_samples: int, fs_hz: float, definitions: FadeDefinitions) -> float:
    """Record duration in seconds under the active convention."""
    fs = _check_fs(fs_hz)
    if n_samples < 2:
        raise ValueError(f"n_samples must be at least 2, got {n_samples}")
    if definitions.record_duration_convention == "intervals":
        return (n_samples - 1) / fs
    return n_samples / fs


def level_crossing_rate(
    amplitude: ArrayLike,
    threshold: float,
    fs_hz: float,
    *,
    definitions: FadeDefinitions = DEFAULT_DEFINITIONS,
) -> float:
    """Down-crossing rate of ``threshold``, in Hz.

    Counted down-crossings divided by the record duration. This is a *sample
    statistic of the supplied series*; it is not an estimate of the
    continuous-time crossing rate of the underlying process, which for a
    Gauss-Markov (Ornstein-Uhlenbeck) log-amplitude is infinite because that
    process is nowhere differentiable. See
    :func:`linkoutage.channel.analytic_level_crossing_rate` for the matching
    discrete-time analytic quantity.
    """
    arr = _as_amplitude(amplitude)
    crossings = down_crossing_indices(arr, threshold, definitions=definitions)
    return float(crossings.size) / record_duration_s(arr.size, fs_hz, definitions)


def mean_fade_duration(
    amplitude: ArrayLike,
    threshold: float,
    fs_hz: float,
    *,
    definitions: FadeDefinitions = DEFAULT_DEFINITIONS,
) -> float:
    """Arithmetic mean of the complete fade durations, in seconds.

    Returns ``nan`` when no fade satisfies the censoring rule, which is the
    honest answer for a record containing one unbroken fade.
    """
    complete, _ = fade_durations(amplitude, threshold, fs_hz, definitions=definitions)
    if complete.size == 0:
        return float("nan")
    return float(complete.mean())


def outage_fraction(
    amplitude: ArrayLike,
    threshold: float,
    *,
    definitions: FadeDefinitions = DEFAULT_DEFINITIONS,
) -> float:
    """Fraction of samples in fade, over the whole record.

    Independent of the censoring rule and of
    ``count_single_sample_fades``: every below-threshold sample counts,
    including those in a censored run and those in a discarded length-1 run.
    Stating this matters, because it is what makes
    ``mean_fade_duration == outage_fraction / level_crossing_rate`` only
    approximately true rather than exactly true on a finite record.
    """
    return float(below_threshold(amplitude, threshold, definitions=definitions).mean())


def availability(
    amplitude: ArrayLike,
    threshold: float,
    *,
    definitions: FadeDefinitions = DEFAULT_DEFINITIONS,
) -> float:
    """``1 - outage_fraction``."""
    return 1.0 - outage_fraction(amplitude, threshold, definitions=definitions)


@dataclass(frozen=True)
class FadeStatistics:
    """Every fade statistic of one record at one threshold.

    All durations in seconds, all rates in Hz.
    """

    threshold: float
    fs_hz: float
    n_samples: int
    record_duration_s: float
    definitions: FadeDefinitions
    n_down_crossings: int
    n_up_crossings: int
    n_runs: int
    n_complete_fades: int
    n_left_censored: int
    n_right_censored: int
    n_single_sample_fades: int
    level_crossing_rate_hz: float
    mean_fade_duration_s: float
    median_fade_duration_s: float
    std_fade_duration_s: float
    max_complete_fade_duration_s: float
    outage_fraction: float
    availability: float
    rice_consistency_mean_fade_duration_s: float
    in_fade_samples: int

    @property
    def rice_relative_residual(self) -> float:
        """Relative gap between the measured mean fade duration and
        ``outage_fraction / level_crossing_rate``.

        A large value is not a bug: it is the combined effect of censoring,
        discarded single-sample runs and finite-record edge effects, all of
        which break the exact partition of below-threshold time into complete
        fades.
        """
        ref = self.rice_consistency_mean_fade_duration_s
        if not np.isfinite(ref) or ref == 0.0:
            return float("nan")
        return (self.mean_fade_duration_s - ref) / ref

    def report(self) -> str:
        """Multi-line human-readable block, definitions included."""
        lines = [
            self.definitions.describe(),
            "",
            "Configuration:",
            f"  samples N              : {self.n_samples}",
            f"  sample rate fs         : {self.fs_hz!r} Hz",
            f"  record duration        : {self.record_duration_s!r} s",
            f"  amplitude threshold T  : {self.threshold!r}",
            "",
            "Results:",
            f"  down-crossings         : {self.n_down_crossings}",
            f"  up-crossings           : {self.n_up_crossings}",
            f"  fade runs (all)        : {self.n_runs}",
            f"  complete fades         : {self.n_complete_fades}",
            f"  left-censored runs     : {self.n_left_censored}",
            f"  right-censored runs    : {self.n_right_censored}",
            f"  single-sample fades    : {self.n_single_sample_fades}",
            f"  in-fade samples        : {self.in_fade_samples}",
            f"  level-crossing rate    : {self.level_crossing_rate_hz!r} Hz",
            f"  mean fade duration     : {self.mean_fade_duration_s!r} s",
            f"  median fade duration   : {self.median_fade_duration_s!r} s",
            f"  std of fade duration   : {self.std_fade_duration_s!r} s",
            f"  longest complete fade  : {self.max_complete_fade_duration_s!r} s",
            f"  outage fraction        : {self.outage_fraction!r}",
            f"  availability           : {self.availability!r}",
            f"  outage_fraction / LCR  : {self.rice_consistency_mean_fade_duration_s!r} s",
            f"  relative residual      : {self.rice_relative_residual!r}",
        ]
        return "\n".join(lines)


def fade_statistics(
    amplitude: ArrayLike,
    threshold: float,
    fs_hz: float,
    *,
    definitions: FadeDefinitions = DEFAULT_DEFINITIONS,
) -> FadeStatistics:
    """Compute every fade statistic of one record at one threshold.

    Parameters
    ----------
    amplitude:
        One-dimensional amplitude (or irradiance, or any monotone proxy, as
        long as ``threshold`` is in the same units). Finite, >= 2 samples.
    threshold:
        Fade threshold in the same units as ``amplitude``.
    fs_hz:
        Sample rate in Hz.
    definitions:
        See :class:`FadeDefinitions`. The result carries the definitions back
        out so that no number is ever reported without them.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([1.0, 0.5, 0.5, 1.0, 1.0, 0.4, 1.0])
    >>> s = fade_statistics(a, 0.6, 1.0)
    >>> s.n_down_crossings, s.n_complete_fades, s.mean_fade_duration_s
    (2, 2, 1.5)
    >>> s.outage_fraction
    0.42857142857142855
    """
    arr = _as_amplitude(amplitude)
    runs = fade_runs(arr, threshold, fs_hz, definitions=definitions)
    complete = runs.complete_durations_s()
    lengths = runs.length_samples
    mask = below_threshold(arr, threshold, definitions=definitions)
    rate = level_crossing_rate(arr, threshold, fs_hz, definitions=definitions)
    frac = float(mask.mean())
    rice = frac / rate if rate > 0.0 else float("nan")
    return FadeStatistics(
        threshold=_check_threshold(threshold),
        fs_hz=_check_fs(fs_hz),
        n_samples=int(arr.size),
        record_duration_s=record_duration_s(arr.size, fs_hz, definitions),
        definitions=definitions,
        n_down_crossings=int(down_crossing_indices(arr, threshold, definitions=definitions).size),
        n_up_crossings=int(up_crossing_indices(arr, threshold, definitions=definitions).size),
        n_runs=runs.n_runs,
        n_complete_fades=int(complete.size),
        n_left_censored=int(np.count_nonzero(runs.left_censored)),
        n_right_censored=int(np.count_nonzero(runs.right_censored)),
        n_single_sample_fades=int(np.count_nonzero(lengths == 1)),
        level_crossing_rate_hz=rate,
        mean_fade_duration_s=float(complete.mean()) if complete.size else float("nan"),
        median_fade_duration_s=float(np.median(complete)) if complete.size else float("nan"),
        std_fade_duration_s=float(complete.std(ddof=1)) if complete.size > 1 else float("nan"),
        max_complete_fade_duration_s=float(complete.max()) if complete.size else float("nan"),
        outage_fraction=frac,
        availability=1.0 - frac,
        rice_consistency_mean_fade_duration_s=rice,
        in_fade_samples=int(np.count_nonzero(mask)),
    )
