"""Baseline 2: plain Monte Carlo over the link model, reported as quantiles.

Why there is no mean in the headline
------------------------------------
A mean delivery time over an intermittent link is close to meaningless, and
this module is built so that it is hard to quote one by accident. The
distribution of delivery time over a link with contact windows is multimodal --
one mode per contact in which the transfer can finish -- and over a link with
correlated outages it is heavy-tailed, because one long outage moves a single
run by much more than the rest of the distribution spans. The mean then sits in
a gap between modes, where no run ever lands, and it moves with the worst run
in the sample rather than with the link.

So :class:`DeliveryDistribution` reports quantiles, states the tail separately,
and carries the mean only alongside the ratio of mean to median so that a
reader can see when the mean has stopped describing anything. The
completion fraction is reported as its own number because an incomplete run has
no delivery time at all, and averaging over only the runs that finished is
survivorship bias with a decimal point on it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .link import LinkConditions
from .modes import TransferMode
from .payload import FileTransferSpec
from .simulate import TransferResult, accounting_discrepancy, simulate_transfer

DEFAULT_QUANTILES: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


@dataclass(frozen=True)
class DeliveryDistribution:
    """The measured distribution of one scenario, with its tail stated apart.

    Attributes
    ----------
    mode, runs:
        What was run and how many times.
    completed:
        Runs in which the receiver held the whole file.
    completion_fraction:
        ``completed / runs``, dimensionless.
    delivery_times_s:
        Delivery times of the completed runs only, seconds, sorted ascending.
    quantiles:
        Quantile level to delivery time in seconds, over completed runs. Linear
        interpolation between order statistics (numpy's default).
    tail:
        The part a quantile table hides: the 99th percentile, the maximum, the
        ratio of maximum to median, and the number of runs that never finished.
    mean_s, median_s, mean_over_median:
        The mean is here so that ``mean_over_median`` can be read. A value far
        from 1 is the signal that the mean has stopped describing the
        distribution.
    overhead_ratios:
        Per-run ``(forward + return wire bytes) / file_bytes``, dimensionless.
    transmissions:
        Per-run forward data-unit transmission counts.
    completion_by_contact:
        Cumulative fraction of all runs (not of completed runs) that had
        finished by the end of contact ``index``. ``-1`` is the single key for
        continuous contact.
    accounting_discrepancies:
        Per-run difference between the aggregate transmission counter and the
        independent per-unit count. Every entry is zero or the harness is
        miscounting.
    """

    mode: TransferMode
    runs: int
    completed: int
    completion_fraction: float
    delivery_times_s: np.ndarray
    quantiles: dict[float, float]
    tail: dict[str, float]
    mean_s: float
    median_s: float
    mean_over_median: float
    overhead_ratios: np.ndarray
    transmissions: np.ndarray
    completion_by_contact: dict[int, float]
    accounting_discrepancies: np.ndarray

    def quantile(self, level: float) -> float:
        """Delivery time at ``level``, seconds, interpolated from the sample."""
        if self.completed == 0:
            raise ValueError("no run completed, so there is no delivery-time quantile")
        return float(np.quantile(self.delivery_times_s, level))

    def describe(self) -> list[str]:
        """Human-readable lines, quantiles first and the tail stated separately."""
        lines = [
            f"mode                    : {self.mode.value}",
            f"runs                    : {self.runs}",
            f"completed               : {self.completed} "
            f"({self.completion_fraction:.6f} of runs)",
        ]
        if self.completed == 0:
            lines.append("no run completed; there is no delivery-time distribution to report")
            return lines
        lines.append("delivery-time quantiles over completed runs, seconds:")
        for level, value in sorted(self.quantiles.items()):
            lines.append(f"  q{level:<6.3f}            : {value:.6f}")
        lines.extend(
            [
                "tail, stated separately:",
                f"  q0.990                : {self.tail['q99_s']:.6f} s",
                f"  maximum               : {self.tail['max_s']:.6f} s",
                f"  maximum / median      : {self.tail['max_over_median']:.6f}",
                f"  runs never completed  : {int(self.tail['incomplete_runs'])}",
                f"mean                    : {self.mean_s:.6f} s "
                f"(mean/median {self.mean_over_median:.6f})",
                f"overhead ratio median   : {float(np.median(self.overhead_ratios)):.6f}",
                f"transmissions median    : {float(np.median(self.transmissions)):.1f}",
            ]
        )
        if self.completion_by_contact:
            lines.append("cumulative completion by end of contact, fraction of all runs:")
            for index, fraction in sorted(self.completion_by_contact.items()):
                label = "continuous" if index < 0 else f"contact {index}"
                lines.append(f"  {label:<21} : {fraction:.6f}")
        return lines


def run_ensemble(
    spec: FileTransferSpec,
    link: LinkConditions,
    mode: TransferMode,
    runs: int,
    seed: int,
    *,
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    payload_seed: int | None = None,
    horizon_s: float | None = None,
    max_forward_transmissions: int | None = None,
) -> DeliveryDistribution:
    """Run ``runs`` independent transfers and summarise them.

    Seeds are ``seed + index``, so an ensemble is reproducible from one integer
    and extending it does not change the runs already measured. Each run
    verifies the completion invariant by hash; a violation propagates as
    :class:`relfiledelivery.payload.CompletionInvariantError` rather than being
    folded into a statistic.
    """
    if runs < 1:
        raise ValueError(f"runs must be at least 1; got {runs}")
    for level in quantiles:
        if not 0.0 < level < 1.0:
            raise ValueError(f"quantile levels must lie in (0, 1); got {level}")

    results: list[TransferResult] = [
        simulate_transfer(
            spec,
            link,
            mode,
            seed + index,
            payload_seed=payload_seed,
            horizon_s=horizon_s,
            max_forward_transmissions=max_forward_transmissions,
        )
        for index in range(runs)
    ]
    return summarise(results, link, mode)


def summarise(
    results: list[TransferResult], link: LinkConditions, mode: TransferMode
) -> DeliveryDistribution:
    """Summarise an explicit list of results. Separated so it can be tested alone."""
    if not results:
        raise ValueError("cannot summarise an empty list of results")
    runs = len(results)
    completed = [r for r in results if r.delivered]
    times = np.sort(np.array([r.delivery_time_s for r in completed], dtype=float))
    overheads = np.array([r.overhead_ratio for r in results], dtype=float)
    transmissions = np.array([r.forward_transmissions for r in results], dtype=float)
    discrepancies = np.array([accounting_discrepancy(r) for r in results], dtype=int)

    if times.size:
        quantile_map = {
            level: float(np.quantile(times, level)) for level in DEFAULT_QUANTILES
        }
        median = float(np.median(times))
        mean = float(np.mean(times))
        tail = {
            "q99_s": float(np.quantile(times, 0.99)),
            "max_s": float(times[-1]),
            "max_over_median": float(times[-1] / median) if median > 0 else math.inf,
            "incomplete_runs": float(runs - len(completed)),
        }
        mean_over_median = mean / median if median > 0 else math.inf
    else:
        quantile_map = {}
        median = math.nan
        mean = math.nan
        mean_over_median = math.nan
        tail = {
            "q99_s": math.nan,
            "max_s": math.nan,
            "max_over_median": math.nan,
            "incomplete_runs": float(runs),
        }

    return DeliveryDistribution(
        mode=mode,
        runs=runs,
        completed=len(completed),
        completion_fraction=len(completed) / runs,
        delivery_times_s=times,
        quantiles=quantile_map,
        tail=tail,
        mean_s=mean,
        median_s=median,
        mean_over_median=mean_over_median,
        overhead_ratios=overheads,
        transmissions=transmissions,
        completion_by_contact=completion_by_contact(results, link),
        accounting_discrepancies=discrepancies,
    )


def completion_by_contact(
    results: list[TransferResult], link: LinkConditions
) -> dict[int, float]:
    """Cumulative fraction of all runs complete by the end of each contact.

    The denominator is every run, including the ones that never completed,
    because the operator's question is "will this file be down by the end of
    the third pass", and dropping the failures from the denominator answers a
    different and more flattering question.
    """
    if not results:
        return {}
    runs = len(results)
    indices = sorted({r.completion_contact_index for r in results if r.delivered})
    if not indices:
        return {}
    if link.is_continuous:
        return {-1: sum(1 for r in results if r.delivered) / runs}
    out: dict[int, float] = {}
    cumulative = 0
    for index in range(len(link.windows)):
        cumulative += sum(
            1 for r in results if r.delivered and r.completion_contact_index == index
        )
        out[index] = cumulative / runs
    return out


def standard_error_of_mean(values: np.ndarray) -> float:
    """Sample standard error of the mean, same units as ``values``.

    Used to state the tolerance of the no-outage check against the closed form:
    the simulation agrees with an exact expectation only to within sampling
    error, and the tolerance has to be that error rather than a number chosen
    to pass.
    """
    array = np.asarray(values, dtype=float)
    if array.size < 2:
        raise ValueError("standard error needs at least two samples")
    return float(np.std(array, ddof=1) / math.sqrt(array.size))
