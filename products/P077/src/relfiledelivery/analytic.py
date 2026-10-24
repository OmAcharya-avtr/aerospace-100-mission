"""Baseline 1: the closed form. Written and verified before any simulation.

Everything in this module is exact under one set of assumptions, stated here
once and repeated on every function:

A1. Each protocol data unit is lost independently with probability ``p``, the
    same ``p`` for every unit and every attempt. No bursts, no correlation.
A2. The link is up for all time. No outages, no contact windows.
A3. Control units (end-of-file on the forward link, negative acknowledgement on
    the return link) are never lost and have a fixed size, so one round
    boundary always costs the same. ``nak_bytes_per_segment`` must be zero.
A4. Transmission is back to back inside a round, with no inter-unit gap and no
    rate adaptation.

Under A1--A4 the results below are exact in exact arithmetic, and agree with
the simulation in :mod:`relfiledelivery.simulate` to within sampling error.
Outside them they are *extensions*, and this package measures how badly they
do rather than asserting that they hold. The availability-inflated extension
in :func:`extended_expected_delivery_time` is the analytic predictor the
learned model has to beat, and in the regime where A2 fails it is the number to
beat it on.

Derivations
-----------
**Attempts per unit.** Under A1 the number of transmissions of one unit until
it arrives is geometric on ``{1, 2, ...}`` with success probability ``1 - p``,
so its expectation is ``1 / (1 - p)`` (Feller 1968, *An Introduction to
Probability Theory and Its Applications*, volume 1, chapter XIII). Summing over
``n`` units, the expected total number of forward transmissions is
``n / (1 - p)`` exactly, for every mode that retransmits, and independently of
how the retransmissions are scheduled. This is why retransmission *overhead*
is mode-independent in expectation while delivery *time* is not.

**Rounds.** In deferred-NAK the retransmissions are organised into rounds: a
unit sent in round ``k`` and lost is sent again in round ``k + 1``. A unit
therefore needs ``G`` rounds where ``G`` is its geometric attempt count, and
the transfer needs ``R = max(G_1, ..., G_n)`` rounds. The ``G_i`` are
independent, so

    P(R <= k) = (1 - p**k)**n,      k = 0, 1, 2, ...

and since ``R`` is a non-negative integer,

    E[R] = sum_{k >= 0} P(R > k) = sum_{k >= 0} (1 - (1 - p**k)**n).

The ``k = 0`` term is 1. Each later term is bounded by ``n * p**k`` through
Bernoulli's inequality, so truncating after ``K`` terms leaves a remainder of
at most ``n * p**K / (1 - p)``: the series is summed to a stated tolerance
rather than guessed at, and the bound is returned with the value.

The term is evaluated as ``-expm1(n * log1p(-p**k))``, not as
``1 - (1 - p**k)**n``. The two are the same function and not the same
computation. Written the naive way, ``1 - p**k`` for a small ``p**k`` rounds
with an absolute error of about half an epsilon; raising that to the ``n``-th
power multiplies the *relative* error by ``n``; and the final subtraction from
1 then leaves an absolute error of order ``n * eps``. At ``n = 1000`` that is
2e-13, which is a thousand times the stated truncation tolerance and was
measured: ``validation/validate_analytic.py`` compared the reported truncation
bound against the measured truncation and found the naive form violating the
bound by 1.8e-13 at ``n = 1000, p = 0.3``. The bound was not wrong; the
arithmetic was. ``log1p`` and ``expm1`` are accurate near zero and the same
check then passes with a slack of 1e-15.

**Delivery time.** Delivery time is the moment the *receiver* holds the whole
file. Under A1--A4 the forward link is busy for the transmission of every
attempt, which costs ``8 * W / ((1 - p) * forward_rate)`` seconds in
expectation where ``W`` is the clean-pass wire size of the file; each of the
``R - 1`` round boundaries costs one end-of-file unit, one one-way light time,
one negative acknowledgement and one more one-way light time; and the last unit
still has to propagate. Hence

    E[T] = 8 * W / ((1 - p) * R_fwd) + (E[R] - 1) * t_turn + L

with ``t_turn = 8 * eof_bytes / R_fwd + 8 * nak_bytes / R_ret + 2 * L``. The
expectation of the first term is exact because the sum over attempts of a
random number of rounds telescopes into the per-unit expectation; the
expectation of the second is exact because ``t_turn`` is constant under A3.

**Unacknowledged.** One pass, nothing retransmitted. The file arrives only if
no unit is lost, with probability ``(1 - p)**n``; conditional on that, the
delivery time is deterministic at ``8 * W / R_fwd + L``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy import stats

from .link import LinkConditions
from .modes import TransferMode
from .payload import FileTransferSpec

DEFAULT_SERIES_TOLERANCE = 1e-14
_MAX_SERIES_TERMS = 10_000_000


def expected_transmissions_per_pdu(loss_probability: float) -> float:
    """``1 / (1 - p)`` forward transmissions per unit, dimensionless.

    Exact under A1. Raises for ``p`` outside ``[0, 1)``.
    """
    _check_probability(loss_probability)
    return 1.0 / (1.0 - loss_probability)


def expected_total_transmissions(pdu_count: int, loss_probability: float) -> float:
    """``n / (1 - p)`` forward transmissions for the whole file, dimensionless.

    Exact under A1 and independent of mode for any mode that retransmits until
    complete. Validity: ``n >= 1``, ``0 <= p < 1``.
    """
    _check_count(pdu_count)
    return pdu_count * expected_transmissions_per_pdu(loss_probability)


def rounds_cdf(pdu_count: int, loss_probability: float, k: int) -> float:
    """``P(R <= k) = (1 - p**k)**n`` for the deferred-NAK round count.

    Exact under A1. ``k`` counts rounds, so ``k = 0`` gives 0 for ``p > 0``.
    """
    _check_count(pdu_count)
    _check_probability(loss_probability)
    if k < 0:
        raise ValueError(f"k must be non-negative; got {k}")
    if k == 0:
        return 0.0 if loss_probability > 0.0 else 1.0
    if loss_probability == 0.0:
        return 1.0
    return math.exp(pdu_count * math.log1p(-(loss_probability**k)))


@dataclass(frozen=True)
class RoundsExpectation:
    """``E[R]`` with the truncation evidence that produced it.

    Attributes
    ----------
    value:
        The summed series, dimensionless number of rounds.
    terms:
        How many terms were summed.
    tail_bound:
        Rigorous upper bound ``n * p**terms / (1 - p)`` on everything left out,
        from Bernoulli's inequality. Reported rather than assumed negligible.
    """

    value: float
    terms: int
    tail_bound: float


def expected_rounds_detail(
    pdu_count: int, loss_probability: float, tolerance: float = DEFAULT_SERIES_TOLERANCE
) -> RoundsExpectation:
    """Sum ``E[R] = sum_k (1 - (1 - p**k)**n)`` to within ``tolerance``.

    Exact under A1 up to the returned ``tail_bound``. For ``p = 0`` the answer
    is exactly 1.0 with a zero tail. For ``n = 1`` it reduces to
    ``1 / (1 - p)``, which is a known-answer check in the test suite.
    """
    _check_count(pdu_count)
    _check_probability(loss_probability)
    if tolerance <= 0.0:
        raise ValueError(f"tolerance must be positive; got {tolerance}")
    p = loss_probability
    if p == 0.0:
        return RoundsExpectation(value=1.0, terms=1, tail_bound=0.0)
    total = 0.0
    k = 0
    while True:
        total += _survival_term(pdu_count, p, k)
        k += 1
        bound = pdu_count * p**k / (1.0 - p)
        if bound <= tolerance:
            return RoundsExpectation(value=total, terms=k, tail_bound=bound)
        if k >= _MAX_SERIES_TERMS:  # pragma: no cover - unreachable for p < 1
            raise RuntimeError(
                f"expected_rounds_detail did not reach tolerance {tolerance} in {k} terms "
                f"at n={pdu_count}, p={p}"
            )


def _survival_term(pdu_count: int, p: float, k: int) -> float:
    """``P(R > k) = 1 - (1 - p**k)**n``, evaluated without cancellation.

    ``-expm1(n * log1p(-x))`` instead of ``1 - (1 - x)**n``. See the module
    docstring for the measured size of the difference; it is not cosmetic.
    """
    if k == 0:
        return 1.0
    x = p**k
    if x == 0.0:
        return 0.0
    if x >= 1.0:  # pragma: no cover - p < 1 and k >= 1
        return 1.0
    return -math.expm1(pdu_count * math.log1p(-x))


def expected_rounds(
    pdu_count: int, loss_probability: float, tolerance: float = DEFAULT_SERIES_TOLERANCE
) -> float:
    """``E[R]`` for deferred-NAK, dimensionless. See :func:`expected_rounds_detail`."""
    return expected_rounds_detail(pdu_count, loss_probability, tolerance).value


def rounds_quantile(pdu_count: int, loss_probability: float, quantile: float) -> int:
    """Smallest ``k`` with ``P(R <= k) >= quantile``, exact under A1.

    Obtained by inverting the closed-form CDF, not by sampling:
    ``k = ceil(log(1 - q**(1/n)) / log p)`` for ``0 < p < 1``, evaluated by
    direct search to avoid a rounding error at the step.
    """
    _check_count(pdu_count)
    _check_probability(loss_probability)
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must lie in (0, 1); got {quantile}")
    if loss_probability == 0.0:
        return 1
    # Analytic starting point, then a short local search so that the returned k
    # satisfies the inequality as evaluated in floating point rather than as it
    # would in exact arithmetic.
    target = quantile ** (1.0 / pdu_count)
    guess = max(1, int(math.ceil(math.log1p(-target) / math.log(loss_probability))) - 2)
    k = guess
    while rounds_cdf(pdu_count, loss_probability, k) < quantile:
        k += 1
        if k > guess + 10_000:  # pragma: no cover - defensive
            raise RuntimeError("rounds_quantile failed to converge")
    while k > 1 and rounds_cdf(pdu_count, loss_probability, k - 1) >= quantile:
        k -= 1
    return k


def transmissions_quantile(pdu_count: int, loss_probability: float, quantile: float) -> float:
    """Quantile of the total forward transmission count, exact under A1.

    The total is ``n`` successes plus a negative-binomial number of failures
    with ``r = n`` and success probability ``1 - p`` (Johnson, Kemp and Kotz
    2005, *Univariate Discrete Distributions*, chapter 5), so this inverts
    ``scipy.stats.nbinom``. Dimensionless.
    """
    _check_count(pdu_count)
    _check_probability(loss_probability)
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must lie in (0, 1); got {quantile}")
    if loss_probability == 0.0:
        return float(pdu_count)
    failures = stats.nbinom.ppf(quantile, pdu_count, 1.0 - loss_probability)
    return float(pdu_count + failures)


def pdu_seconds(spec: FileTransferSpec, link: LinkConditions, sequence: int) -> float:
    """Forward-link occupancy of unit ``sequence``, seconds."""
    return 8.0 * spec.wire_bytes_of(sequence) / link.forward_rate_bps


def clean_pass_seconds(spec: FileTransferSpec, link: LinkConditions) -> float:
    """Forward time for one loss-free pass over the whole file, seconds."""
    return 8.0 * spec.total_wire_bytes / link.forward_rate_bps


def turnaround_seconds(link: LinkConditions) -> float:
    """Cost of one deferred-NAK round boundary, seconds.

    ``eof_bytes`` on the forward link, one one-way light time out, a
    fixed-size negative acknowledgement on the return link, and one one-way
    light time back. Requires A3: a NAK whose size grows with the number of
    missing segments makes this quantity random.
    """
    require_closed_form_applicable(link)
    return link.eof_seconds + link.nak_seconds(0) + 2.0 * link.one_way_light_time_s


def require_closed_form_applicable(link: LinkConditions) -> None:
    """Raise :class:`ValueError` unless A2 and A3 hold for ``link``.

    This is deliberately strict. The closed form is the reference the
    simulation is verified against, and silently applying it outside its
    assumptions is how a harness ends up validating itself against its own
    approximation.
    """
    if not link.outage.is_outage_free:
        raise ValueError(
            "the closed form assumes no outages (A2); this link declares mean_down_s="
            f"{link.outage.mean_down_s}. Use extended_expected_delivery_time for the "
            "availability-inflated extension, or simulate."
        )
    if not link.is_continuous:
        raise ValueError(
            f"the closed form assumes continuous contact (A2); this link declares "
            f"{len(link.windows)} contact windows. Use extended_expected_delivery_time, "
            "or simulate."
        )
    if link.nak_bytes_per_segment != 0.0:
        raise ValueError(
            "the closed form assumes a fixed-size NAK (A3); this link declares "
            f"nak_bytes_per_segment={link.nak_bytes_per_segment}, which makes the turnaround "
            "cost depend on how much was lost. Simulate instead."
        )


@dataclass(frozen=True)
class AnalyticDelivery:
    """Closed-form delivery prediction for one file, link and mode.

    Attributes
    ----------
    mode:
        The transfer mode the prediction is for.
    pdu_count:
        Units the file is cut into, dimensionless.
    expected_transmissions:
        Expected forward data-unit transmissions, dimensionless. Mode-independent
        for retransmitting modes.
    expected_rounds:
        ``E[R]``, dimensionless. 1.0 for unacknowledged.
    rounds_tail_bound:
        Truncation bound on the ``E[R]`` series.
    busy_seconds:
        Expected forward-link busy time, seconds. A lower bound on delivery
        time for every mode.
    turnaround_seconds:
        Cost of one round boundary, seconds. Zero for unacknowledged.
    expected_delivery_time_s:
        Expected seconds until the receiver holds the file, or ``None`` for
        unacknowledged where delivery is not certain. Exact under A1--A4 for
        deferred-NAK; for immediate-NAK this field is ``None`` and
        ``busy_seconds + light time`` is the lower bound instead, because
        pipelined retransmission makes the idle time depend on the arrival
        order and has no closed form here.
    delivery_time_lower_bound_s:
        ``busy_seconds`` plus one one-way light time, seconds. Exact lower
        bound in expectation for every retransmitting mode.
    confirmation_time_s:
        Expected seconds until the *sender* knows the transfer finished, which
        costs one further turnaround. ``None`` for unacknowledged.
    completion_probability:
        Probability the file is delivered. Exactly 1.0 for retransmitting modes
        under A1--A4 (the geometric attempt count is almost surely finite), and
        ``(1 - p)**n`` for unacknowledged.
    expected_forward_wire_bytes, expected_return_wire_bytes:
        Expected bytes on each link, including control units.
    expected_overhead_ratio:
        ``(forward + return wire bytes) / file_bytes``, dimensionless and >= 1.
    """

    mode: TransferMode
    pdu_count: int
    expected_transmissions: float
    expected_rounds: float
    rounds_tail_bound: float
    busy_seconds: float
    turnaround_seconds: float
    expected_delivery_time_s: float | None
    delivery_time_lower_bound_s: float
    confirmation_time_s: float | None
    completion_probability: float
    expected_forward_wire_bytes: float
    expected_return_wire_bytes: float
    expected_overhead_ratio: float

    def describe(self) -> list[str]:
        """Human-readable lines for a report."""
        delivery = (
            "not certain (see completion probability)"
            if self.expected_delivery_time_s is None
            else f"{self.expected_delivery_time_s:.6f} s"
        )
        return [
            f"mode                    : {self.mode.value}",
            f"PDU count               : {self.pdu_count}",
            f"E[transmissions]        : {self.expected_transmissions:.6f}",
            f"E[rounds]               : {self.expected_rounds:.6f} "
            f"(series tail bound {self.rounds_tail_bound:.3e})",
            f"forward busy time       : {self.busy_seconds:.6f} s",
            f"turnaround per round    : {self.turnaround_seconds:.6f} s",
            f"E[delivery time]        : {delivery}",
            f"delivery lower bound    : {self.delivery_time_lower_bound_s:.6f} s",
            f"completion probability  : {self.completion_probability:.9f}",
            f"E[forward wire bytes]   : {self.expected_forward_wire_bytes:.3f}",
            f"E[return wire bytes]    : {self.expected_return_wire_bytes:.3f}",
            f"E[overhead ratio]       : {self.expected_overhead_ratio:.6f}",
        ]


def analytic_delivery(
    spec: FileTransferSpec,
    link: LinkConditions,
    mode: TransferMode,
    tolerance: float = DEFAULT_SERIES_TOLERANCE,
) -> AnalyticDelivery:
    """Closed-form prediction under A1--A4. Raises outside those assumptions.

    Parameters
    ----------
    spec:
        The file and its segmentation.
    link:
        Declared link conditions. Must be outage-free, continuous and have a
        fixed-size NAK; otherwise :class:`ValueError`.
    mode:
        Transfer mode.
    tolerance:
        Truncation tolerance for the ``E[R]`` series.
    """
    require_closed_form_applicable(link)
    n = spec.pdu_count
    p = link.pdu_loss_probability
    light = link.one_way_light_time_s
    clean = clean_pass_seconds(spec, link)

    if mode is TransferMode.UNACKNOWLEDGED:
        return AnalyticDelivery(
            mode=mode,
            pdu_count=n,
            expected_transmissions=float(n),
            expected_rounds=1.0,
            rounds_tail_bound=0.0,
            busy_seconds=clean,
            turnaround_seconds=0.0,
            expected_delivery_time_s=clean + light,
            delivery_time_lower_bound_s=clean + light,
            confirmation_time_s=None,
            completion_probability=(1.0 - p) ** n,
            expected_forward_wire_bytes=float(spec.total_wire_bytes),
            expected_return_wire_bytes=0.0,
            expected_overhead_ratio=spec.total_wire_bytes / spec.file_bytes,
            )

    rounds = expected_rounds_detail(n, p, tolerance)
    busy = clean / (1.0 - p)
    turn = turnaround_seconds(link)
    boundaries = rounds.value - 1.0
    forward_bytes = spec.total_wire_bytes / (1.0 - p) + boundaries * link.eof_bytes
    return_bytes = boundaries * link.nak_bytes
    expected_time = busy + boundaries * turn + light if mode is TransferMode.DEFERRED_NAK else None
    return AnalyticDelivery(
        mode=mode,
        pdu_count=n,
        expected_transmissions=expected_total_transmissions(n, p),
        expected_rounds=rounds.value,
        rounds_tail_bound=rounds.tail_bound,
        busy_seconds=busy,
        turnaround_seconds=turn,
        expected_delivery_time_s=expected_time,
        delivery_time_lower_bound_s=busy + light,
        confirmation_time_s=None if expected_time is None else expected_time + turn,
        completion_probability=1.0,
        expected_forward_wire_bytes=forward_bytes,
        expected_return_wire_bytes=return_bytes,
        expected_overhead_ratio=(forward_bytes + return_bytes) / spec.file_bytes,
    )


def analytic_delivery_time_quantile(
    spec: FileTransferSpec,
    link: LinkConditions,
    quantile: float,
    tolerance: float = DEFAULT_SERIES_TOLERANCE,
) -> float:
    """Semi-analytic deferred-NAK delivery-time quantile, seconds.

    Built from two *exact* marginal quantiles -- the negative-binomial
    transmission count and the closed-form round count -- evaluated at the same
    probability and combined:

        T_q = 8 * W_q / R_fwd + (R_q - 1) * t_turn + L

    where ``W_q`` is the wire size implied by the transmission-count quantile.
    Evaluating two *dependent* quantiles at the same probability and adding them
    is an approximation, not an identity: the sum of quantiles is not the
    quantile of the sum. It is used as the analytic *interval* in the predictor
    benchmark, and its measured coverage against nominal is reported rather than
    claimed. Validity: A1--A4.
    """
    require_closed_form_applicable(link)
    n = spec.pdu_count
    p = link.pdu_loss_probability
    attempts = transmissions_quantile(n, p, quantile)
    mean_wire = spec.total_wire_bytes / n
    rounds = rounds_quantile(n, p, quantile)
    forward = 8.0 * attempts * mean_wire / link.forward_rate_bps
    return forward + (rounds - 1) * turnaround_seconds(link) + link.one_way_light_time_s


def _spread_over_contacts(needed_s: float, link: LinkConditions) -> float:
    """Wall-clock time at which ``needed_s`` of contact time runs out, seconds.

    Lays the required contact time over the declared windows in order, starting
    at the first window's start, and returns the elapsed time when it is used
    up. For continuous contact the answer is ``needed_s`` itself. When the
    declared windows cannot carry it, the return value is the end of the last
    window plus the unserved remainder, which is an extrapolation past the
    declared contact plan and is recognisable by exceeding the contact span.
    """
    if needed_s < 0.0:
        raise ValueError(f"needed_s must be non-negative; got {needed_s}")
    if link.is_continuous:
        return needed_s
    origin = link.windows[0].start_s
    remaining = needed_s
    for window in link.windows:
        if remaining <= window.duration_s:
            return (window.start_s - origin) + remaining
        remaining -= window.duration_s
    return (link.windows[-1].end_s - origin) + remaining


def extended_delivery_time_quantile(
    spec: FileTransferSpec,
    link: LinkConditions,
    quantile: float,
    tolerance: float = DEFAULT_SERIES_TOLERANCE,
) -> float:
    """Contact-aware extension of the semi-analytic quantile, seconds.

    The deferred-NAK quantile of :func:`analytic_delivery_time_quantile`
    evaluated on the outage-free link, divided by the outage duty cycle, laid
    out over the declared contact windows. Carries every approximation of
    :func:`analytic_delivery_time_quantile` and of
    :func:`extended_expected_delivery_time`, and is the analytic *interval* in
    the predictor benchmark; its measured coverage against nominal is reported
    rather than claimed.
    """
    base = analytic_delivery_time_quantile(spec, link.without_outages(), quantile, tolerance)
    return _spread_over_contacts(base / link.outage.duty_cycle, link)


def extended_expected_delivery_time(
    spec: FileTransferSpec,
    link: LinkConditions,
    mode: TransferMode,
    tolerance: float = DEFAULT_SERIES_TOLERANCE,
) -> float:
    """Contact-aware extension of the closed form past A2, seconds.

    Three steps, each of them arithmetic on declared quantities:

    1. Evaluate the closed form on ``link.without_outages()``. That is the
       usable-link time the transfer needs, exact under A1, A3 and A4.
    2. Divide by the outage duty cycle. A fraction ``1 - duty`` of the time
       inside a contact carries nothing, so the transfer needs that much more
       contact time. This is an approximation: it ignores that an outage
       destroys units already in flight and that a round boundary straddling an
       outage pays its turnaround twice.
    3. Lay that contact time out over the **declared contact windows, in
       order**, and return the wall-clock time at which it runs out.

    **This is an extension, not a theorem.** It is the analytic predictor the
    learned model is benchmarked against, and the size of its error in the
    correlated-outage regime is a measured result of this package, reported in
    both directions.

    Step 3 exists because the first version of this function divided by
    ``link.availability``, the product of the two duty cycles. On the worked
    example in ``validation/worked_example.py`` -- a 2 MiB file over six
    300-second contacts every 1800 seconds -- that gave 164.57 s against a
    measured median of 26.80 s, a factor of 6.14, because the transfer fits
    inside the *first* contact and the contact duty cycle never bites at all.
    An extension that wrong is not a baseline; it is a straw man, and
    benchmarking a learned model against a straw man measures nothing. Laying
    the required contact time over the declared windows costs three lines and
    removes that error exactly in the common case where the transfer fits in
    one pass.

    When the declared windows cannot carry the transfer at all, the return
    value is the end of the last window plus the unserved remainder, which is
    an extrapolation past the declared contact plan and is flagged by being
    larger than the contact span.

    For immediate-NAK, whose closed form has no delivery-time expression, the
    deferred-NAK expression is used, which overstates the idle time and is
    declared as such.
    """
    base_link = link.without_outages()
    reference_mode = (
        TransferMode.DEFERRED_NAK if mode is TransferMode.IMMEDIATE_NAK else mode
    )
    base = analytic_delivery(spec, base_link, reference_mode, tolerance)
    if base.expected_delivery_time_s is None:  # pragma: no cover - only unacknowledged
        raise ValueError(f"no closed-form delivery time for mode {mode.value}")
    duty = link.outage.duty_cycle
    if duty <= 0.0:  # pragma: no cover - rejected by OutageModel
        raise ValueError("outage duty cycle must be positive")
    return _spread_over_contacts(base.expected_delivery_time_s / duty, link)


def _check_probability(value: float) -> None:
    if not 0.0 <= value < 1.0:
        raise ValueError(f"loss probability must lie in [0, 1); got {value}")


def _check_count(value: int) -> None:
    if value < 1:
        raise ValueError(f"pdu_count must be at least 1; got {value}")
