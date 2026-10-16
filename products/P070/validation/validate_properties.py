"""Algebraic identities over a deterministic sweep of declared parameters.

The test suite drives the same identities with Hypothesis. This script sweeps
a fixed grid instead, so the counts quoted in README.md and VALIDATION.md come
from a run that produces the same numbers every time. Identities that are
only true in exact arithmetic are checked with a tolerance, and the largest
observed deviation is printed rather than being hidden behind a pass.

The one identity that is reported as FALSIFIED is the claim that equation (3)
is attained for every declared supervisor. It is not: with a poll release
jitter larger than the poll period it is a strict over-estimate. That was
found by Hypothesis and is pinned in tests/test_regression.py.
"""

from __future__ import annotations

import itertools
import math

from _support import Emitter
from watchdogspec import (
    DEFAULT_TOL_MS,
    Ladder,
    Margin,
    Rung,
    Supervisor,
    Task,
    false_trip_probability,
    max_tolerance_for_deadline,
    task_timing,
    verdict,
    worst_case_detection_latency_ms,
)
from watchdogspec.model import Action

PERIODS = (0.5, 2.0, 7.5, 20.0, 137.0, 1000.0)
JITTERS = (0.0, 0.05, 1.3, 9.0, 60.0)
TOLERANCES = (0, 1, 2, 3, 7, 15)
POLLS = (0.25, 1.0, 5.0, 10.0, 97.0)
POLL_JITTERS = (0.0, 0.7, 3.0)
OVERHEADS = (0.0, 0.5, 4.0)
DEADLINES = (10.0, 100.0, 229.0, 1000.0, 50_000.0)
LADDER = Ladder(
    id="L",
    rungs=(
        Rung(action=Action.LOG, latency_ms=2.0, recovery_cost=0.1),
        Rung(action=Action.RESTART_TASK, latency_ms=20.0, recovery_cost=2.0),
        Rung(action=Action.SAFE_MODE, latency_ms=50.0, recovery_cost=10.0),
        Rung(action=Action.RESET, latency_ms=100.0, recovery_cost=80.0),
    ),
)
SIGMAS = (0.01, 0.4, 3.0, 55.0)


def main() -> int:  # noqa: C901 - a linear sequence of independent sweeps
    """Sweep the identities. Returns a process exit code."""
    emit = Emitter(__file__)
    emit("watchdogspec validation: algebraic identities over a deterministic sweep")
    emit()

    emit("1. equation (3) against an independent re-association of the same sum")
    worst_absolute = 0.0
    worst_ulps = 0.0
    trials = 0
    for period, jitter, k, poll, poll_jitter, overhead in itertools.product(
        PERIODS, JITTERS, TOLERANCES, POLLS, POLL_JITTERS, OVERHEADS
    ):
        trials += 1
        direct = worst_case_detection_latency_ms(
            period, jitter, k, poll, poll_jitter, overhead
        )
        reassociated = (k + 1) * period + (k + 1) * jitter + poll + poll_jitter + overhead
        difference = abs(direct - reassociated)
        if difference > worst_absolute:
            worst_absolute = difference
            worst_ulps = difference / math.ulp(direct)
    emit(f"  parameter sets         : {trials}")
    emit(f"  largest absolute difference : {worst_absolute:.3e} ms")
    emit(f"  largest difference in ulp   : {worst_ulps:.1f}")
    emit.check(
        "the two summation orders agree to within 8 units in the last place",
        worst_ulps <= 8.0,
        f"{worst_ulps:.1f} ulp",
    )
    emit.check(
        "the two summation orders are NOT bit-identical, which is why the "
        "deadline verdict is three-valued",
        worst_absolute > 0.0,
        f"{worst_absolute:.3e} ms apart at worst",
    )

    emit()
    emit("2. the three-valued verdict is stable under that re-association")
    unstable_two_valued = 0
    unstable_absolute_only = 0
    unstable_scaled = 0
    for period, jitter, k, poll, poll_jitter, overhead in itertools.product(
        PERIODS, JITTERS, TOLERANCES, POLLS, POLL_JITTERS, OVERHEADS
    ):
        direct = worst_case_detection_latency_ms(
            period, jitter, k, poll, poll_jitter, overhead
        )
        reassociated = (k + 1) * period + (k + 1) * jitter + poll + poll_jitter + overhead
        deadline = reassociated
        if ((deadline - direct) > 0.0) != ((deadline - reassociated) > 0.0):
            unstable_two_valued += 1
        if verdict(deadline - direct, DEFAULT_TOL_MS) is not verdict(
            deadline - reassociated, DEFAULT_TOL_MS
        ):
            unstable_absolute_only += 1
        if verdict(deadline - direct, DEFAULT_TOL_MS, deadline) is not verdict(
            deadline - reassociated, DEFAULT_TOL_MS, deadline
        ):
            unstable_scaled += 1
    emit(f"  parameter sets         : {trials}")
    emit(f"  two-valued 'slack > 0' flips           : {unstable_two_valued}")
    emit(f"  three-valued flips, absolute band only : {unstable_absolute_only}")
    emit(f"  three-valued flips, scale-aware band   : {unstable_scaled}")
    emit.check(
        "the scale-aware three-valued verdict never flips under re-association",
        unstable_scaled == 0,
    )
    emit.check(
        "the two-valued boolean does flip, which is the behaviour being avoided",
        unstable_two_valued > 0,
        f"{unstable_two_valued} of {trials} parameter sets",
    )

    emit()
    emit("2b. the same, at latencies where one ulp exceeds the absolute tolerance")
    emit("    An ABSOLUTE band of 1e-9 ms is below one ulp above about 4.5e6 ms.")
    big_periods = (1e5, 3.7e5, 1e6)
    big_jitters = (0.0, 86413.50000000362, 1.3e5)
    big_tolerances = (20, 44, 90)
    big_trials = 0
    big_absolute_only = 0
    big_scaled = 0
    worst_band = 0.0
    for period, jitter, k, poll in itertools.product(
        big_periods, big_jitters, big_tolerances, POLLS
    ):
        big_trials += 1
        direct = worst_case_detection_latency_ms(period, jitter, k, poll)
        reassociated = (k + 1) * period + (k + 1) * jitter + poll
        deadline = reassociated
        worst_band = max(worst_band, abs(direct - reassociated) / max(math.ulp(direct), 1e-300))
        if verdict(deadline - direct, DEFAULT_TOL_MS) is not verdict(
            deadline - reassociated, DEFAULT_TOL_MS
        ):
            big_absolute_only += 1
        if verdict(deadline - direct, DEFAULT_TOL_MS, deadline) is not verdict(
            deadline - reassociated, DEFAULT_TOL_MS, deadline
        ):
            big_scaled += 1
    emit(f"  parameter sets         : {big_trials}")
    emit(f"  largest re-association error          : {worst_band:.1f} ulp")
    emit(f"  three-valued flips, absolute band only: {big_absolute_only}")
    emit(f"  three-valued flips, scale-aware band  : {big_scaled}")
    emit.check(
        "the absolute band alone is NOT sufficient at these magnitudes",
        big_absolute_only > 0,
        f"{big_absolute_only} of {big_trials} flip; found by Hypothesis at the "
        "dependency floor",
    )
    emit.check(
        "the scale-aware band is sufficient at these magnitudes",
        big_scaled == 0,
    )

    emit()
    emit("3. escalation timing identities")
    supervisor_grid = [
        Supervisor(poll_period_ms=q, poll_jitter_ms=jq, declaration_overhead_ms=o)
        for q, jq, o in itertools.product(POLLS, POLL_JITTERS, OVERHEADS)
    ]
    checked = 0
    bad_monotone = 0
    bad_first_rung = 0
    bad_total = 0
    bad_deepest = 0
    bad_deadline_monotone = 0
    for period, jitter, k, deadline, supervisor in itertools.product(
        PERIODS, JITTERS, TOLERANCES, DEADLINES, supervisor_grid
    ):
        task = Task(
            id="t",
            heartbeat_period_ms=period,
            missed_beats_tolerated=k,
            liveness_deadline_ms=deadline,
            escalation="L",
            heartbeat_jitter_ms=jitter,
        )
        timing = task_timing(task, supervisor, LADDER)
        checked += 1
        finishes = [rung.finish_ms for rung in timing.rungs]
        if any(b < a - 1e-9 for a, b in zip(finishes[:-1], finishes[1:], strict=True)):
            bad_monotone += 1
        if abs(timing.rungs[0].start_ms - timing.detection_latency_ms) > 1e-9:
            bad_first_rung += 1
        if abs(timing.full_ladder_latency_ms - finishes[-1]) > 1e-9:
            bad_total += 1
        if not 0 <= timing.deepest_rung_reachable <= len(LADDER.rungs):
            bad_deepest += 1
        wider = Task(
            id="t",
            heartbeat_period_ms=period,
            missed_beats_tolerated=k,
            liveness_deadline_ms=deadline * 2.0,
            escalation="L",
            heartbeat_jitter_ms=jitter,
        )
        if (
            task_timing(wider, supervisor, LADDER).deepest_rung_reachable
            < timing.deepest_rung_reachable
        ):
            bad_deadline_monotone += 1
    emit(f"  parameter sets         : {checked}")
    emit(f"  non-monotone rung finish times        : {bad_monotone}")
    emit(f"  first rung not starting at detection  : {bad_first_rung}")
    emit(f"  ladder total not equal to last finish : {bad_total}")
    emit(f"  deepest rung outside 0..len(rungs)    : {bad_deepest}")
    emit(f"  deepest rung falling with a wider D   : {bad_deadline_monotone}")
    emit.check("rung finish times are non-decreasing", bad_monotone == 0)
    emit.check("rung 1 starts at the detection instant", bad_first_rung == 0)
    emit.check("the ladder total equals the last rung finish", bad_total == 0)
    emit.check("the deepest reachable rung is in range", bad_deepest == 0)
    emit.check(
        "the deepest reachable rung never falls when the deadline widens",
        bad_deadline_monotone == 0,
    )

    emit()
    emit("4. the largest admissible missed-beat tolerance")
    solved = 0
    bad_fit = 0
    bad_next = 0
    for period, jitter, deadline, supervisor in itertools.product(
        PERIODS, JITTERS, DEADLINES, supervisor_grid
    ):
        k = max_tolerance_for_deadline(
            period,
            jitter,
            deadline,
            supervisor.poll_period_ms,
            supervisor.poll_jitter_ms,
            supervisor.declaration_overhead_ms,
            LADDER.total_latency_ms,
        )
        solved += 1
        window = period + jitter
        observation = supervisor.observation_delay_ms
        if k >= 0 and (k + 1) * window + observation + LADDER.total_latency_ms > deadline + 1e-6:
            bad_fit += 1
        if (k + 2) * window + observation + LADDER.total_latency_ms <= deadline - 1e-6:
            bad_next += 1
    emit(f"  parameter sets         : {solved}")
    emit(f"  returned k that does not fit          : {bad_fit}")
    emit(f"  k + 1 that would also have fitted     : {bad_next}")
    emit.check("the returned tolerance always fits", bad_fit == 0)
    emit.check("the returned tolerance is the largest that fits", bad_next == 0)

    emit()
    emit("5. false-trip monotonicity and range")
    rate_trials = 0
    out_of_range = 0
    not_monotone_in_k = 0
    not_monotone_in_jitter = 0
    not_monotone_in_sigma = 0
    for period, jitter, sigma in itertools.product(PERIODS, JITTERS, SIGMAS):
        for k in TOLERANCES:
            rate_trials += 1
            value = false_trip_probability(period, jitter, sigma, k)
            if not 0.0 <= value <= 0.5:
                out_of_range += 1
            if false_trip_probability(period, jitter, sigma, k + 1) > value + 1e-18:
                not_monotone_in_k += 1
            if false_trip_probability(period, jitter + 1.0, sigma, k) > value + 1e-18:
                not_monotone_in_jitter += 1
            if false_trip_probability(period, jitter, sigma * 2.0, k) < value - 1e-18:
                not_monotone_in_sigma += 1
    emit(f"  parameter sets         : {rate_trials}")
    emit(f"  probabilities outside [0, 0.5]        : {out_of_range}")
    emit(f"  rate rising with the tolerance        : {not_monotone_in_k}")
    emit(f"  rate rising with the jitter bound     : {not_monotone_in_jitter}")
    emit(f"  rate falling with sigma               : {not_monotone_in_sigma}")
    emit.check("every false-trip probability is in [0, 0.5]", out_of_range == 0)
    emit.check("the rate never rises with the tolerance", not_monotone_in_k == 0)
    emit.check("the rate never rises with the jitter bound", not_monotone_in_jitter == 0)
    emit.check("the rate never falls with sigma", not_monotone_in_sigma == 0)

    emit()
    emit("6. the verdict partitions the real line")
    partition_trials = 0
    wrong = 0
    for slack in (-1e6, -1.0, -1e-9, -1e-12, 0.0, 1e-12, 1e-9, 1.0, 1e6):
        for tol in (0.0, 1e-12, 1e-9, 1.0):
            partition_trials += 1
            result = verdict(slack, tol)
            expected = (
                Margin.MARGINAL
                if abs(slack) <= tol
                else (Margin.MET if slack > 0.0 else Margin.MISSED)
            )
            if result is not expected:
                wrong += 1
    emit(f"  parameter sets         : {partition_trials}")
    emit(f"  disagreements          : {wrong}")
    emit.check("the verdict agrees with its definition everywhere", wrong == 0)

    return emit.finish()


if __name__ == "__main__":
    raise SystemExit(main())
