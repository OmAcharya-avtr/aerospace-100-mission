"""The bracket from mathematics the accumulator does not use, with its controls.

"Two implementations agree" and "the answer lies inside a bound derived from
different mathematics" are not the same check, and the second catches what the
first cannot: an earlier product of this mission had 777 tests, property tests
included, that could not catch catastrophic cancellation in a closed form,
because every test compared the series against expressions sharing the same
cancellation.

The accumulator masks the attempt table on the failure flag, groups by test and
sums with ``numpy.add.at``. The two bounds here group nothing and sum no
subset: each uses an integer failure count plus statistics over the WHOLE
duration sample -- two extreme order statistics for the first, the second
moment for the second. The derivations are in
``src/flakerank/bounds.py``.

The slack follows the CONDITIONING of the bound expression and not an absolute
constant, because an absolute slack floored at 1.8e-15 once made a bracket of
width 4.5e-23 vacuous in this mission. It is the standard forward error bound
for floating-point summation, and its size relative to the bracket width is
reported so a reader can see that it is not what makes the check pass.

This script also states the LIMIT of the bound honestly: the bracket is as wide
as the duration spread, so on a history spanning four orders of magnitude it
catches structural errors and not every multiplicative one. The control that
needs a tight bracket uses a near-constant-duration history, where the bound
can see a doubling.

NO CLOCK IS READ.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

from flakerank.baselines import accumulated_wasted_time  # noqa: E402
from flakerank.bounds import SAFETY, UNIT_ROUNDOFF, wasted_time_bracket  # noqa: E402
from flakerank.history import history_from_mapping, load_history  # noqa: E402

#: Session windows the bracket is checked over, as ``(start, stop)``. The whole
#: history plus windows of several sizes, so the check is not a single cell.
WINDOWS = ((0, 330), (0, 30), (0, 110), (110, 220), (220, 330), (300, 330))


def constant_duration_history(sessions: int, duration_ms: int):
    """A history whose every attempt has the same duration.

    The order-statistic bracket is EXACT on such a history, since ``t_min`` and
    ``t_max`` coincide, so it is the right instrument for a control that has to
    catch a multiplicative error. Built in memory and validated by the same
    loader as the committed file.
    """
    attempts: list[tuple[int, int, int, int, int]] = []
    for session in range(sessions):
        attempts.append((session, 0, 1, 1, duration_ms))
        attempts.append((session, 0, 2, 0, duration_ms))
    return history_from_mapping(
        {
            "schema": "flakerank.history/1",
            "retry_budget": 3,
            "duration_unit": "millisecond",
            "tests": [
                {"name": "suite/always_fails_once", "suite": "suite",
                 "true_failure_probability": 1.0, "true_mean_ms": float(duration_ms)}
            ],
            "sessions": [{"index": index, "day": index} for index in range(sessions)],
            "attempts": {
                "session": [row[0] for row in attempts],
                "test": [row[1] for row in attempts],
                "attempt": [row[2] for row in attempts],
                "failed": [row[3] for row in attempts],
                "duration_ms": [row[4] for row in attempts],
            },
        },
        source="constant-duration",
    )


def main() -> int:
    """Check the bracket over several windows, then run its controls."""
    out = _bootstrap.Tee("validate_bounds_output.txt")
    out("The independently derived bracket on total wasted time -- P097 flakerank")
    out("")
    out("The accumulator masks, groups and sums. The bounds here use a failure")
    out("COUNT plus statistics over the whole duration sample: two extreme order")
    out("statistics, and the second moment. Neither groups anything, and min and")
    out("max are selections rather than additions, so they cannot share an")
    out("accumulation error with the thing they bracket.")
    out("")
    out("Equation (1): F * t_min <= W <= F * t_max, counting plus order statistics.")
    out("Equation (2): W <= sqrt(F * sum t^2), Cauchy-Schwarz against the all-ones")
    out("  vector; source Hardy, G. H., Littlewood, J. E. & Polya, G. (1952),")
    out("  Inequalities, 2nd edition, Cambridge University Press.")
    out("Equation (3): slack = SAFETY * F * eps * sum|t|, the standard forward")
    out("  error bound for floating-point summation; see Higham, N. J. (2002),")
    out("  Accuracy and Stability of Numerical Algorithms, 2nd edition, SIAM,")
    out("  chapter 4.")
    out("")
    out(f"unit roundoff eps = 2**-53 = {UNIT_ROUNDOFF!r}")
    out(f"SAFETY = {SAFETY!r}, covering the grouped rather than strictly recursive")
    out("  accumulation order of numpy.add.at")
    out("")
    out(f"fast path (FLAKERANK_FAST=1): {_bootstrap.FAST}")
    out(f"history: {_bootstrap.HISTORY.as_posix()} sha256 {_bootstrap.HISTORY_SHA256}")

    history = load_history(ROOT / _bootstrap.HISTORY)

    out.rule("1. THE BRACKET OVER SIX SESSION WINDOWS")
    out("")
    out(f"{'window':>12} {'F':>6} {'accumulated s':>15} {'low s':>12} {'order high s':>15}"
        f" {'C-S high s':>14} {'slack s':>11} {'in':>5}")
    all_bracket = True
    slack_ratios: list[float] = []
    for start, stop in WINDOWS:
        window = history.session_window(start, stop)
        truth = accumulated_wasted_time(window, resamples=20, seed=_bootstrap.SEED)
        bracket = wasted_time_bracket(window, truth.total_seconds)
        all_bracket = all_bracket and bracket.brackets
        width = bracket.tightest_high_seconds - bracket.order_statistic_low
        if width > 0.0:
            slack_ratios.append(bracket.slack_seconds / width)
        out(f"{f'[{start},{stop})':>12} {bracket.failed_attempts:>6}"
            f" {bracket.accumulated_seconds:>15.3f} {bracket.order_statistic_low:>12.3f}"
            f" {bracket.order_statistic_high:>15.3f} {bracket.cauchy_schwarz_high:>14.3f}"
            f" {bracket.slack_seconds:>11.3e} {str(bracket.brackets):>5}")
    out("")
    out.check(
        "the bracket contains the accumulated total in every window",
        all_bracket,
        f"{len(WINDOWS)} windows",
    )
    out("")
    out(f"slack divided by bracket width, worst of the windows: {max(slack_ratios):.3e}")
    out.check(
        "the slack is below 1e-6 of the bracket width, so it is not what makes "
        "the check pass",
        max(slack_ratios) < 1e-6,
        f"worst ratio {max(slack_ratios):.3e}",
    )

    out.rule("2. THE WHOLE HISTORY, IN FULL")
    out("")
    truth = accumulated_wasted_time(
        history, resamples=_bootstrap.resamples(), seed=_bootstrap.SEED
    )
    bracket = wasted_time_bracket(history, truth.total_seconds)
    out(f"failed attempts F                  : {bracket.failed_attempts}")
    out(f"shortest attempt t_min, s          : {bracket.min_attempt_seconds!r}")
    out(f"longest attempt t_max, s           : {bracket.max_attempt_seconds!r}")
    out(f"sum of squared durations, s^2      : {bracket.second_moment_seconds_squared!r}")
    out(f"sum of absolute durations, s       : {bracket.absolute_total_seconds!r}")
    out("")
    out(f"accumulated wasted time, s         : {bracket.accumulated_seconds!r}")
    out(f"equation (1) lower, F * t_min, s   : {bracket.order_statistic_low!r}")
    out(f"equation (1) upper, F * t_max, s   : {bracket.order_statistic_high!r}")
    out(f"equation (2) upper, Cauchy-Schwarz : {bracket.cauchy_schwarz_high!r}")
    out(f"tightest upper bound, s            : {bracket.tightest_high_seconds!r}")
    out(f"equation (3) slack, s              : {bracket.slack_seconds!r}")
    out(f"margin to the lower bound, s       : {bracket.margin_low_seconds!r}")
    out(f"margin to the upper bound, s       : {bracket.margin_high_seconds!r}")
    out(f"bracket width over accumulated     : {bracket.relative_width!r}")
    out("")
    out.check("the bracket contains the accumulated total", bracket.brackets)
    out.check(
        "Cauchy-Schwarz is the tighter of the two upper bounds on this history",
        bracket.cauchy_schwarz_high < bracket.order_statistic_high,
        f"{bracket.cauchy_schwarz_high:.3f} against {bracket.order_statistic_high:.3f}",
    )
    out("")
    out("WHAT THIS BOUND IS WORTH, stated rather than implied. The bracket's")
    out("width is set by the spread of the duration sample, and this history")
    out(f"spans {bracket.min_attempt_seconds!r} s to {bracket.max_attempt_seconds!r} s. So:")
    doubled = wasted_time_bracket(history, 2.0 * truth.total_seconds)
    tenfold = wasted_time_bracket(history, 10.0 * truth.total_seconds)
    out(f"  a DOUBLED accumulator total still fits inside it: {doubled.brackets}")
    out(f"  a TENFOLD total does not                        : {tenfold.brackets}")
    out("")
    out("The bound therefore catches structural errors -- the wrong mask, a")
    out("missing group, a sign, a double count -- and not every multiplicative")
    out("one. That is a property of the bound and not a defect in it, and the")
    out("control below uses a history where the bracket IS tight.")

    out.rule("3. A HISTORY WHERE THE BRACKET IS EXACT")
    out("")
    out("40 attempts of 1000 ms each, 20 of them failing. t_min = t_max, so")
    out("equation (1) gives [20.0, 20.0] s exactly and any multiplicative error")
    out("at all leaves it.")
    out("")
    tight = constant_duration_history(20, 1000)
    tight_truth = accumulated_wasted_time(tight, resamples=20, seed=_bootstrap.SEED)
    tight_bracket = wasted_time_bracket(tight, tight_truth.total_seconds)
    out(f"accumulated, s        : {tight_bracket.accumulated_seconds!r}")
    out(f"equation (1) bracket  : [{tight_bracket.order_statistic_low!r}, "
        f"{tight_bracket.order_statistic_high!r}]")
    out(f"equation (2) upper, s : {tight_bracket.cauchy_schwarz_high!r}")
    out(f"slack, s              : {tight_bracket.slack_seconds!r}")
    out.check(
        "the exact bracket contains the accumulated total",
        tight_bracket.brackets,
    )
    out.check(
        "the exact bracket has zero width before the slack",
        tight_bracket.order_statistic_high == tight_bracket.order_statistic_low,
        f"[{tight_bracket.order_statistic_low!r}, {tight_bracket.order_statistic_high!r}]",
    )

    out.rule("4. NEGATIVE CONTROLS, ANCHORED ON THE BOUND ITSELF")
    out("")
    out("Each injected value is derived from the BOUND, not from the")
    out("accumulator's output, and is checked to lie beyond the bound plus the")
    out("slack BEFORE the control is evaluated. An offset applied to a value")
    out("already inside a band need not leave it, which is how an earlier")
    out("product of this mission shipped two of eleven controls silently too")
    out("weak.")
    out("")
    high_injection = 1.5 * bracket.tightest_high_seconds
    out(f"1.5x the tightest upper bound: {high_injection!r} s")
    out(f"  beyond upper + slack: "
        f"{high_injection > bracket.tightest_high_seconds + bracket.slack_seconds}")
    out.control(
        "the bracket rejects a total 1.5x above its upper bound",
        not wasted_time_bracket(history, high_injection).brackets,
        must_fire=True,
    )
    low_injection = 0.5 * bracket.order_statistic_low
    out(f"0.5x the lower bound: {low_injection!r} s")
    out(f"  below lower - slack: "
        f"{low_injection < bracket.order_statistic_low - bracket.slack_seconds}")
    out.control(
        "the bracket rejects a total half its lower bound",
        not wasted_time_bracket(history, low_injection).brackets,
        must_fire=True,
    )
    out("")
    out("On the exact-bracket history, where the bound can see a multiplicative")
    out("error:")
    for factor in (2.0, 1.1, 1.001):
        injected = factor * tight_truth.total_seconds
        out.control(
            f"the exact bracket rejects a total {factor}x the truth",
            not wasted_time_bracket(tight, injected).brackets,
            must_fire=True,
            detail=f"{injected!r} s",
        )
    out("")
    out("CONTROLS that must STAY SILENT. A check that cannot pass is as useless")
    out("as one that cannot fail.")
    out.control(
        "the bracket rejects the accumulated total",
        not wasted_time_bracket(history, truth.total_seconds).brackets,
        must_fire=False,
    )
    out.control(
        "the bracket rejects a total sitting exactly on its upper bound",
        not wasted_time_bracket(history, bracket.tightest_high_seconds).brackets,
        must_fire=False,
        detail="the bracket is a closed interval and both ends are attainable",
    )
    out.control(
        "the exact bracket rejects its own accumulated total",
        not wasted_time_bracket(tight, tight_truth.total_seconds).brackets,
        must_fire=False,
    )

    out.rule("5. THE BRACKET ON RANDOM SYNTHETIC HISTORIES")
    out("")
    out("The bracket is a mathematical consequence of counting and order")
    out("statistics, so it must hold on every history the loader accepts and not")
    out("only on the committed one. 200 synthetic histories are generated at the")
    out("declared seed, with integer millisecond durations spanning four orders")
    out("of magnitude and random outcomes.")
    out("")
    generator = np.random.default_rng(_bootstrap.SEED)
    trials = 40 if _bootstrap.FAST else 200
    held = 0
    for trial in range(trials):
        sessions = int(generator.integers(5, 60))
        attempts: list[tuple[int, int, int, int, int]] = []
        for session in range(sessions):
            failed = bool(generator.integers(0, 2))
            attempts.append(
                (session, 0, 1, int(failed), int(generator.integers(1, 1_000_000)))
            )
            if failed:
                attempts.append((session, 0, 2, 0, int(generator.integers(1, 1_000_000))))
        synthetic = history_from_mapping(
            {
                "schema": "flakerank.history/1",
                "retry_budget": 3,
                "duration_unit": "millisecond",
                "tests": [{"name": "suite/t", "suite": "suite",
                           "true_failure_probability": 0.5, "true_mean_ms": 1.0}],
                "sessions": [{"index": index, "day": index} for index in range(sessions)],
                "attempts": {
                    "session": [row[0] for row in attempts],
                    "test": [row[1] for row in attempts],
                    "attempt": [row[2] for row in attempts],
                    "failed": [row[3] for row in attempts],
                    "duration_ms": [row[4] for row in attempts],
                },
            },
            source=f"synthetic-{trial}",
        )
        synthetic_truth = accumulated_wasted_time(synthetic, resamples=5, seed=trial + 1)
        held += int(wasted_time_bracket(synthetic, synthetic_truth.total_seconds).brackets)
    out(f"synthetic histories generated: {trials}")
    out(f"the bracket held on             : {held}")
    out.check(
        "the bracket holds on every synthetic history",
        held == trials,
        f"{held} of {trials}",
    )
    return out.finish()


if __name__ == "__main__":
    sys.exit(main())
