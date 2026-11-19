"""The attribution report: per-mechanism loss with an interval on each, and a residual.

This is what the package's public API returns, because it is the thing an
operator can act on. A total loss figure says a handover is expensive. A
decomposition with an interval on each mechanism says whether to buy faster
acquisition, a different protocol, or more buffer.

THE DECOMPOSITION AND ITS RESIDUAL
----------------------------------

The three mechanisms are a partition of the accounting window, so the per-
mechanism volumes sum to the total dropped volume. That alone is not evidence:
it is how the buckets were filled. The residual this module reports compares the
bucketed sum against an **independent** accounting of the same total:

* ``sum_of_mechanisms`` is the sum of the three phase buckets, accumulated slot
  by slot during the simulation;
* ``conservation_total`` is ``produced - delivered - (q_end - q_0)``, which
  never touches the buckets and follows only from the fact that every offered
  bit is delivered, still buffered, or dropped;
* ``residual = conservation_total - sum_of_mechanisms``.

The two paths share the per-slot arrival and service arrays and nothing else:
one accumulates overflow, the other accumulates arrivals, departures and the
endpoint occupancies. In exact arithmetic the residual is zero. In double
precision it is the rounding of two differently ordered sums over the same slot
count, so it is small, non-zero, and **reported as a number** rather than
asserted away. ``validation/validate_decomposition_residual.py`` reports the
distribution of the relative residual over the whole corpus and the worst case,
and ``tests/test_accounting.py`` asserts it against the conditioning-scaled
tolerance :func:`residual_tolerance_bits` rather than against a constant.

WHAT "DOMINANT" MEANS, AND WHEN IT IS DECIDED
---------------------------------------------

:attr:`AttributionReport.dominant` is the mechanism with the largest point
estimate. :attr:`AttributionReport.dominance_is_decided` is true only when the
leading mechanism's interval does not overlap the runner-up's. Those are
different claims and the second is the one worth acting on. The package reports
both and never collapses them, because a dominance that is not decided is a
reason to measure more handovers, not a reason to buy hardware.

UNITS
-----

Bits throughout, with ``*_gbit`` helpers for reporting. Shares are
dimensionless fractions of the total.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bootstrap import RESAMPLING_UNIT, BootstrapInterval, event_bootstrap
from .sequence import MECHANISMS
from .trace import TraceResult

#: Bits per gigabit, for the reporting helpers.
BITS_PER_GBIT = 1.0e9


def residual_tolerance_bits(
    produced_bits: float | np.ndarray,
    delivered_bits: float | np.ndarray,
    dropped_bits: float | np.ndarray,
    slots: int,
) -> float | np.ndarray:
    """Tolerance for the decomposition residual, bits, scaled by conditioning.

    The residual is the difference of two sums over ``slots`` terms whose
    magnitudes are of the order of the produced, delivered and dropped volumes.
    The forward error of a naive sum of ``n`` non-negative terms is bounded by
    ``n * eps`` times the sum, and there are two such sums plus the endpoint
    difference, so

    ``tolerance = 4 * slots * eps * (produced + delivered + dropped)``

    with ``eps = 2**-53``. The factor scales with the SLOT COUNT because that is
    what drives the accumulation error, and with the volume magnitudes because
    that is the conditioning. A constant tolerance would be vacuous for a long
    window and falsely tight for a short one.
    """
    if slots < 1:
        raise ValueError(f"slots must be at least 1, got {slots!r}")
    return (
        4.0
        * float(slots)
        * (2.0**-53)
        * (
            np.abs(produced_bits)
            + np.abs(delivered_bits)
            + np.abs(dropped_bits)
        )
    )


@dataclass(frozen=True)
class MechanismShare:
    """One mechanism's contribution, with its interval. Bits, plus a share.

    Attributes
    ----------
    name:
        One of :data:`~handoverloss.sequence.MECHANISMS`.
    loss_bits:
        Point estimate of this mechanism's contribution over the events
        reported, bits.
    low_bits, high_bits:
        Bootstrap bounds on :attr:`loss_bits`, bits, at the report's confidence.
    share:
        :attr:`loss_bits` divided by the report's total; zero when the total is
        zero.
    """

    name: str
    loss_bits: float
    low_bits: float
    high_bits: float
    share: float

    def __post_init__(self) -> None:
        if self.name not in MECHANISMS:
            raise ValueError(
                f"MechanismShare.name must be one of {MECHANISMS}, got {self.name!r}"
            )
        if self.low_bits > self.high_bits:
            raise ValueError(
                f"MechanismShare {self.name!r}: low_bits ({self.low_bits!r}) exceeds "
                f"high_bits ({self.high_bits!r})"
            )

    @property
    def loss_gbit(self) -> float:
        """:attr:`loss_bits` in gigabits."""
        return self.loss_bits / BITS_PER_GBIT

    @property
    def low_gbit(self) -> float:
        """:attr:`low_bits` in gigabits."""
        return self.low_bits / BITS_PER_GBIT

    @property
    def high_gbit(self) -> float:
        """:attr:`high_bits` in gigabits."""
        return self.high_bits / BITS_PER_GBIT


@dataclass(frozen=True)
class AttributionReport:
    """The headline object: which mechanism dominates, with an interval on each.

    Attributes
    ----------
    mechanisms:
        The three :class:`MechanismShare` objects in
        :data:`~handoverloss.sequence.MECHANISMS` order.
    dominant:
        Name of the mechanism with the largest point estimate.
    dominance_margin:
        Leading share minus runner-up share, dimensionless.
    dominance_is_decided:
        True only when the leading mechanism's interval is disjoint from the
        runner-up's. A false value means the data do not decide the ordering.
    total_loss_bits:
        Sum of the three mechanism point estimates, bits.
    conservation_total_bits:
        The same total from mass conservation, computed without the buckets.
    residual_bits:
        ``conservation_total_bits - total_loss_bits``.
    residual_relative:
        :attr:`residual_bits` divided by :attr:`conservation_total_bits`; zero
        when that total is zero.
    residual_tolerance_bits:
        The conditioning-scaled tolerance the residual is compared against.
    resampling_unit:
        Asserted to equal :data:`~handoverloss.bootstrap.RESAMPLING_UNIT`.
    n_events, n_resamples, confidence:
        Provenance of the intervals.
    """

    mechanisms: tuple[MechanismShare, MechanismShare, MechanismShare]
    dominant: str
    dominance_margin: float
    dominance_is_decided: bool
    total_loss_bits: float
    conservation_total_bits: float
    residual_bits: float
    residual_relative: float
    residual_tolerance_bits: float
    resampling_unit: str
    n_events: int
    n_resamples: int
    confidence: float

    def __post_init__(self) -> None:
        if len(self.mechanisms) != 3:
            raise ValueError(
                f"AttributionReport needs exactly 3 mechanisms, got "
                f"{len(self.mechanisms)}"
            )
        names = tuple(share.name for share in self.mechanisms)
        if names != MECHANISMS:
            raise ValueError(
                f"AttributionReport.mechanisms must be in the order {MECHANISMS}, got "
                f"{names}"
            )
        if self.resampling_unit != RESAMPLING_UNIT:
            raise ValueError(
                "AttributionReport.resampling_unit must be "
                f"{RESAMPLING_UNIT!r}, got {self.resampling_unit!r}. Every interval in "
                "this report was produced by resampling handover events; a report "
                "claiming any other unit did not come from this package."
            )
        if self.dominant not in MECHANISMS:
            raise ValueError(
                f"AttributionReport.dominant must be one of {MECHANISMS}, got "
                f"{self.dominant!r}"
            )

    def share(self, name: str) -> MechanismShare:
        """The :class:`MechanismShare` named ``name``."""
        for candidate in self.mechanisms:
            if candidate.name == name:
                return candidate
        raise KeyError(f"no mechanism named {name!r}; expected one of {MECHANISMS}")

    @property
    def total_loss_gbit(self) -> float:
        """:attr:`total_loss_bits` in gigabits."""
        return self.total_loss_bits / BITS_PER_GBIT

    @property
    def residual_within_tolerance(self) -> bool:
        """Whether ``|residual_bits| <= residual_tolerance_bits``.

        A floating-point comparison, so this package reports the two numbers
        wherever it reports the verdict and never asserts the boolean alone.
        """
        return abs(self.residual_bits) <= self.residual_tolerance_bits

    def ranked(self) -> list[MechanismShare]:
        """The three mechanisms sorted by point estimate, largest first."""
        return sorted(self.mechanisms, key=lambda share: share.loss_bits, reverse=True)

    def lines(self) -> list[str]:
        """Human-readable rendering, used by the CLI and the examples."""
        rows = [
            f"events                 : {self.n_events}",
            f"resampling unit        : {self.resampling_unit}",
            f"resamples / confidence : {self.n_resamples} / {self.confidence:.2f}",
            f"total loss             : {self.total_loss_gbit:.6f} Gbit",
            "",
            "mechanism      loss [Gbit]    interval [Gbit]                share",
        ]
        for share in self.mechanisms:
            rows.append(
                f"{share.name:<14} {share.loss_gbit:>12.6f}   "
                f"[{share.low_gbit:>11.6f}, {share.high_gbit:>11.6f}]   "
                f"{share.share:>7.4f}"
            )
        rows += [
            "",
            f"dominant               : {self.dominant}",
            f"dominance margin       : {self.dominance_margin:.6f} of the total",
            f"dominance decided      : {self.dominance_is_decided} "
            + (
                "(the leading interval IS disjoint from the runner-up's)"
                if self.dominance_is_decided
                else "(the leading interval OVERLAPS the runner-up's, so these events "
                "rank the mechanisms but do not decide the ordering)"
            ),
            f"conservation total     : {self.conservation_total_bits / BITS_PER_GBIT:.6f} Gbit",
            f"decomposition residual : {self.residual_bits:.6e} bits "
            f"({self.residual_relative:.3e} relative)",
            f"residual tolerance     : {self.residual_tolerance_bits:.6e} bits "
            f"-> within tolerance: {self.residual_within_tolerance}",
        ]
        return rows


def attribute(
    trace: TraceResult,
    *,
    resamples: int = 4000,
    confidence: float = 0.95,
    rng: np.random.Generator | None = None,
) -> AttributionReport:
    """Baseline 3: the attribution report from a simulated handover trace.

    The point estimate of each mechanism is the MEAN per-event dropped volume
    for that mechanism, so the report is a statement about a handover of this
    kind rather than about the particular events sampled. Intervals come from
    :func:`~handoverloss.bootstrap.event_bootstrap`, whose resampling unit is
    the handover event.

    Parameters
    ----------
    trace:
        The simulated trace, with at least 5 events.
    resamples:
        Bootstrap replicates.
    confidence:
        Nominal interval coverage.
    rng:
        Generator for the resample indices.

    Returns
    -------
    AttributionReport

    Raises
    ------
    ValueError
        Below 5 events (from :func:`event_bootstrap`), or on an invalid
        ``confidence`` or ``resamples``.
    """
    dropped = np.asarray(trace.dropped_bits, dtype=float)
    if dropped.ndim != 2 or dropped.shape[1] != 3:
        raise ValueError(
            f"trace.dropped_bits must have shape (n, 3), got {dropped.shape}"
        )
    means = dropped.mean(axis=0)
    total = float(means.sum())

    intervals: list[BootstrapInterval] = []
    for column in range(3):
        intervals.append(
            event_bootstrap(
                dropped,
                lambda rows, _c=column: float(rows[:, _c].mean()),
                resamples=resamples,
                confidence=confidence,
                rng=rng,
            )
        )

    shares = tuple(
        MechanismShare(
            name=MECHANISMS[column],
            loss_bits=float(means[column]),
            low_bits=intervals[column].low,
            high_bits=intervals[column].high,
            share=(float(means[column]) / total) if total > 0.0 else 0.0,
        )
        for column in range(3)
    )

    order = sorted(range(3), key=lambda c: float(means[c]), reverse=True)
    leader, runner_up = order[0], order[1]
    decided = intervals[leader].disjoint_from(intervals[runner_up])
    margin = shares[leader].share - shares[runner_up].share

    conservation = float(np.asarray(trace.conservation_total_bits, dtype=float).mean())
    residual = conservation - total
    tolerance = float(
        residual_tolerance_bits(
            float(np.asarray(trace.produced_bits, dtype=float).mean()),
            float(np.asarray(trace.delivered_bits, dtype=float).mean()),
            abs(total),
            max(int(trace.slot_count), 1),
        )
    )

    return AttributionReport(
        mechanisms=shares,  # type: ignore[arg-type]
        dominant=MECHANISMS[leader],
        dominance_margin=float(margin),
        dominance_is_decided=bool(decided),
        total_loss_bits=total,
        conservation_total_bits=conservation,
        residual_bits=float(residual),
        residual_relative=float(residual / conservation) if conservation > 0.0 else 0.0,
        residual_tolerance_bits=tolerance,
        resampling_unit=RESAMPLING_UNIT,
        n_events=len(trace),
        n_resamples=int(resamples),
        confidence=float(confidence),
    )
