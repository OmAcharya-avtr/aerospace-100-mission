"""ARL0, ARL1 and the delay-versus-false-alarm trade-off, with stated error bars.

Definitions used throughout, so there is no ambiguity about what is measured
--------------------------------------------------------------------------
**ARL0** -- average run length to a false alarm on a stationary stream. Measured
by running a long stationary stream, resetting the detector at every alarm, and
averaging the resulting run lengths. Units: samples.

**ARL1** -- average delay from the change index to the first alarm at or after
it, measured in the **steady-state convention**. Units: samples. The detector is
run normally through a stationary pre-change segment, resetting at every false
alarm exactly as it would in service, so at the change index it is in whatever
state the pre-change stream left it in. No replicate is excluded and nothing is
conditioned on.

This convention was chosen after measuring the alternative. Conditioning on
"no alarm before the change" -- which is what the first implementation of this
module did, and what a reader might expect -- **discards 40 to 48 % of
replicates** at an ARL0 of 500 with a 300-sample pre-change segment, because
``1 - exp(-300/500) = 45 %`` of streams false-alarm first. Throwing away nearly
half the replicates on a criterion correlated with the detector's state is a
selection effect larger than any difference between the detectors. The raw
numbers from that version are in ``validation/outputs/
validate_arl1_convention.txt``.

The steady-state convention charges a detector for the cost of its own false
alarms: a windowed detector that resets 300 samples before the change is blind
when the change arrives, and that is a real property of the detector, not an
artefact. :func:`blind_fraction` measures it separately so the effect can be
attributed rather than guessed at.

A replicate with no alarm within the post-change budget is right-censored at the
budget, and the reported mean is a **lower bound** on the true ARL1 whenever the
censored fraction is nonzero.

**Monte Carlo standard error.** Every ARL figure this package publishes carries
``sem = s / sqrt(n)`` where ``s`` is the sample standard deviation of the run
lengths or delays and ``n`` their count. Run lengths on a stationary stream are
close to geometric, so ``s`` is close to the mean and the relative standard
error is close to ``1/sqrt(n)``: roughly 3.5 % at ``n = 800``. A point estimate
quoted without this number is not a measurement, and this package does not
produce one.

**Why not a closed form.** Siegmund's approximation for CUSUM ARL0 and the
Markov-chain approximations for EWMA exist and are good. They are
approximations, they do not exist for the windowed KS test or for ADWIN under
this harness's reset convention, and they would make the five detectors
non-comparable. Everything here is measured.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .detectors import Detector

__all__ = [
    "ARL0Result",
    "ARL1Result",
    "TradeoffPoint",
    "blind_fraction",
    "bootstrap_mean_ci",
    "measure_arl0",
    "measure_arl1",
    "run_lengths_on_stream",
    "wilson_interval",
]


def run_lengths_on_stream(detector: Detector, stream: np.ndarray) -> tuple[list[int], int]:
    """Collect run lengths from one stream, resetting the detector at each alarm.

    Returns
    -------
    run_lengths:
        Completed run lengths in samples. A run length is the number of samples
        consumed since the last reset, inclusive of the alarming sample.
    censored_tail:
        Samples consumed after the final alarm with no further alarm. This tail
        is **not** included in ``run_lengths``; it is returned so the caller can
        report the induced downward bias rather than ignore it.

    Notes
    -----
    A detector that declares ``self_resetting`` manages its own post-alarm state
    and is not reset here. That is how published ADWIN's window-shrinking
    convention is measured on the same footing as the full-reset convention used
    for the other four detectors.
    """
    detector.reset()
    lengths: list[int] = []
    since = 0
    external_reset = not getattr(detector, "self_resetting", False)
    for value in stream:
        since += 1
        if detector.update(value):
            lengths.append(since)
            since = 0
            if external_reset:
                detector.reset()
    return lengths, since


@dataclass(frozen=True)
class ARL0Result:
    """Measured mean time to false alarm on a stationary stream, in samples."""

    arl0: float
    sem: float
    n_runs: int
    total_samples: int
    censored_tail_samples: int
    run_lengths: np.ndarray = field(repr=False)

    @property
    def relative_sem(self) -> float:
        """``sem / arl0``, dimensionless. The honest precision of the estimate."""
        return self.sem / self.arl0 if self.arl0 > 0 else float("nan")

    @property
    def censored_fraction(self) -> float:
        """Fraction of the generated samples sitting in the uncompleted tail."""
        return self.censored_tail_samples / self.total_samples if self.total_samples else 0.0

    def summary(self) -> str:
        return (
            f"ARL0 = {self.arl0:.1f} +/- {self.sem:.1f} samples "
            f"({100 * self.relative_sem:.1f} % rel. SEM, n={self.n_runs} runs, "
            f"{self.total_samples} samples, censored tail "
            f"{100 * self.censored_fraction:.2f} %)"
        )


def measure_arl0(
    factory,
    stream_fn,
    seeds,
    stream_length: int,
) -> ARL0Result:
    """Measure ARL0 by restart-after-alarm over several seeded stationary streams.

    Parameters
    ----------
    factory:
        Zero-argument callable returning a configured :class:`Detector`.
    stream_fn:
        ``stream_fn(length, seed) -> np.ndarray``, a stationary stream generator.
    seeds:
        Iterable of integer seeds, one per replicate stream.
    stream_length:
        Samples per replicate stream.

    Notes
    -----
    Restart-after-alarm is used instead of one-run-per-stream because it costs
    ``stream_length`` samples for roughly ``stream_length / ARL0`` run lengths
    instead of one, and the run lengths are independent given the reset. The
    price is the right-censored tail after the last alarm, reported above.
    """
    seeds = list(seeds)
    if not seeds:
        raise ValueError("at least one seed is required")
    if stream_length < 1:
        raise ValueError("stream_length must be >= 1")
    all_lengths: list[int] = []
    censored = 0
    for seed in seeds:
        stream = stream_fn(stream_length, seed)
        lengths, tail = run_lengths_on_stream(factory(), stream)
        all_lengths.extend(lengths)
        censored += tail
    total = stream_length * len(seeds)
    if not all_lengths:
        # No alarm at all over the whole budget: ARL0 exceeds the budget. Report
        # the budget as a lower bound rather than returning a fabricated number.
        return ARL0Result(
            arl0=float(total),
            sem=float("nan"),
            n_runs=0,
            total_samples=total,
            censored_tail_samples=censored,
            run_lengths=np.asarray([], dtype=float),
        )
    arr = np.asarray(all_lengths, dtype=float)
    sem = float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else float("nan")
    return ARL0Result(
        arl0=float(arr.mean()),
        sem=sem,
        n_runs=arr.size,
        total_samples=total,
        censored_tail_samples=censored,
        run_lengths=arr,
    )


@dataclass(frozen=True)
class ARL1Result:
    """Measured detection delay after a declared change, in samples.

    Steady-state convention: see this module's docstring. ``n_used`` equals
    ``n_replicates`` by construction -- nothing is dropped -- and
    ``n_pre_change_alarms`` counts how many replicates had at least one false
    alarm before the change, reported as a diagnostic rather than used as a
    filter.
    """

    arl1: float
    sem: float
    n_used: int
    n_replicates: int
    n_pre_change_alarms: int
    n_censored: int
    budget: int
    delays: np.ndarray = field(repr=False)
    n_reset_at_change: int = 0

    @property
    def pre_change_alarm_rate(self) -> float:
        """Fraction of replicates with at least one false alarm before the change."""
        return self.n_pre_change_alarms / self.n_replicates if self.n_replicates else 0.0

    @property
    def censored_rate(self) -> float:
        """Fraction of replicates that never alarmed within the budget."""
        return self.n_censored / self.n_used if self.n_used else 0.0

    @property
    def blind_at_change_rate(self) -> float:
        """Fraction of replicates in which the detector was un-armed at the change.

        A detector that false-alarmed shortly before the change is still warming
        up when the change arrives. This is the mechanism by which a windowed
        detector's own false-alarm rate degrades its detection delay, and it is
        measured rather than argued about.
        """
        return self.n_reset_at_change / self.n_replicates if self.n_replicates else 0.0

    @property
    def is_lower_bound(self) -> bool:
        """True when censoring makes the reported ARL1 a lower bound only."""
        return self.n_censored > 0

    def summary(self) -> str:
        tag = " (LOWER BOUND, censored)" if self.is_lower_bound else ""
        return (
            f"ARL1 = {self.arl1:.1f} +/- {self.sem:.1f} samples{tag} "
            f"(n={self.n_used}, pre-change false alarm in "
            f"{100 * self.pre_change_alarm_rate:.1f} % of replicates, "
            f"un-armed at change {100 * self.blind_at_change_rate:.1f} %, "
            f"censored at {self.budget} {100 * self.censored_rate:.1f} %)"
        )


def measure_arl1(
    factory,
    stream_fn,
    seeds,
    pre_length: int,
    budget: int,
) -> ARL1Result:
    """Measure detection delay after a change, steady-state convention.

    Parameters
    ----------
    factory:
        Zero-argument callable returning a configured :class:`Detector`.
    stream_fn:
        ``stream_fn(pre_length, post_length, seed) -> (stream, change_index)``.
    pre_length:
        Stationary samples before the change. Should be at least one target ARL0
        so the detector reaches its steady-state distribution; the benchmark
        uses 1000 against a target ARL0 of 500.
    budget:
        Maximum post-change samples. Delays are right-censored at this value.

    Notes
    -----
    Alarms during the pre-change segment are false alarms. The detector is reset
    on each one (unless it declares ``self_resetting``) and the stream continues,
    which is what it does in service. The first alarm at or after
    ``change_index`` ends the replicate and its index minus ``change_index`` is
    the delay. A delay of 0 is possible and means the alarming sample is the
    first post-change sample.
    """
    seeds = list(seeds)
    if not seeds:
        raise ValueError("at least one seed is required")
    if budget < 1:
        raise ValueError("budget must be >= 1")
    delays: list[int] = []
    pre_alarm_replicates = 0
    censored = 0
    unarmed_at_change = 0
    for seed in seeds:
        stream, change_index = stream_fn(pre_length, budget, seed)
        det = factory()
        det.reset()
        external_reset = not getattr(det, "self_resetting", False)
        had_pre_alarm = False
        for i in range(change_index):
            if det.update(stream[i]):
                had_pre_alarm = True
                if external_reset:
                    det.reset()
        if had_pre_alarm:
            pre_alarm_replicates += 1
        if not det.is_armed():
            unarmed_at_change += 1
        alarm_at = -1
        for i in range(change_index, len(stream)):
            if det.update(stream[i]):
                alarm_at = i
                break
        if alarm_at < 0:
            delays.append(budget)
            censored += 1
        else:
            delays.append(alarm_at - change_index)
    arr = np.asarray(delays, dtype=float)
    sem = float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else float("nan")
    return ARL1Result(
        arl1=float(arr.mean()),
        sem=sem,
        n_used=arr.size,
        n_replicates=len(seeds),
        n_pre_change_alarms=pre_alarm_replicates,
        n_censored=censored,
        budget=budget,
        delays=arr,
        n_reset_at_change=unarmed_at_change,
    )


def blind_fraction(detector, stream: np.ndarray) -> float:
    """Fraction of a stationary stream during which an alarm was impossible.

    The detector is run with reset-on-alarm, as in an ARL0 measurement, and the
    samples on which ``detector.is_armed()`` is false are counted. A detector
    with no warm-up returns 0.0 by construction. A windowed detector with a
    300-sample warm-up and an ARL0 of 500 spends a large and measurable fraction
    of its life unable to detect anything, which is the cost of its window and
    is invisible in an ARL0/ARL1 pair.
    """
    detector.reset()
    external_reset = not getattr(detector, "self_resetting", False)
    blind = 0
    for value in stream:
        if not detector.is_armed():
            blind += 1
        if detector.update(value) and external_reset:
            detector.reset()
    return blind / len(stream) if len(stream) else 0.0


def bootstrap_mean_ci(
    values: np.ndarray,
    level: float = 0.95,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean.

    Used instead of a normal interval for the delay distributions, which are
    strongly right-skewed and occasionally point-massed at the censoring budget,
    so a ``mean +/- 1.96 sem`` interval is not trustworthy at these sample sizes.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    if not 0.0 < level < 1.0:
        raise ValueError("level must satisfy 0 < level < 1")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    lo = float(np.quantile(means, (1.0 - level) / 2.0))
    hi = float(np.quantile(means, 1.0 - (1.0 - level) / 2.0))
    return (lo, hi)


@dataclass(frozen=True)
class TradeoffPoint:
    """One point on a detector's delay-versus-false-alarm curve."""

    detector: str
    threshold: float
    arl0: ARL0Result
    arl1: ARL1Result


def wilson_interval(
    successes: int, trials: int, z: float = 1.959_963_984_540_054
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion, default 95 %.

    Used instead of the normal approximation for the transient-spike firing
    rates, several of which are 0/400 or 400/400. The normal interval is
    degenerate at both ends; Wilson is not, which is the only reason it is here.

    Parameters
    ----------
    successes:
        Number of successes, ``0 <= successes <= trials``.
    trials:
        Number of trials, ``>= 1``.
    z:
        Normal quantile. The default is the two-sided 95 % value.
    """
    if trials < 1:
        raise ValueError("trials must be >= 1")
    if not 0 <= successes <= trials:
        raise ValueError("successes must satisfy 0 <= successes <= trials")
    n = float(trials)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    lo = 0.0 if successes == 0 else max(0.0, centre - half)
    hi = 1.0 if successes == trials else min(1.0, centre + half)
    # At successes = 0 the Wilson lower bound is analytically 0 and at
    # successes = trials the upper bound is analytically 1 (centre + half
    # collapses to (1 + z^2/n)/denom = 1). Floating point loses the last bit, so
    # the two endpoints are set exactly rather than left at 1 - 1e-16, which
    # would make a measured rate of 1.0 sit outside its own interval.
    return (lo, hi)
