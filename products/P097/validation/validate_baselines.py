"""The three baselines, validated against the generator's ground truth, before the model.

The order matters and is required by the mission's build guide: an
under-strength baseline inflates an apparent learned win, and an earlier
product of this mission published a learned margin about 50 per cent too large
for exactly that reason. So the baselines are validated here, against a truth
that is known by construction, and only then is the learned component
benchmarked in ``validate_learned_model.py``.

THE TRUTH USED HERE
-------------------

Not the accumulated outcome -- that is baseline 3, which is one of the things
being checked -- but the GENERATOR'S OWN EXPECTATION, reconstructed from the
parameters recorded in ``data/run_history.json``:

    true expected wasted seconds
        = S_before * E[F_m(p)] * t_fail  +  S_after * E[F_m(p')] * t_fail

with ``p`` the generator's declared per-attempt failure probability, ``p'`` its
value after the declared regime change, ``t_fail`` the generator's declared
mean attempt duration times its declared fail-duration factor, ``m`` the retry
budget, and ``S_before`` / ``S_after`` the test's REALISED invocation counts in
the two regimes. Conditioning on the realised invocation counts removes the
skip process from the comparison, so what is left is the expected wasted time
given when the test actually ran.

This is an expectation computed from parameters no baseline and no model reads.
It is the right anchor for a negative control, because a control anchored on an
OUTPUT cannot be guaranteed to fire: an offset applied to a value already
inside a band need not leave it, and an earlier product of this mission shipped
two of eleven controls silently too weak that way.

WHAT IS CHECKED
---------------

1. Baseline 2 against the generator's expectation, per test and in aggregate.
2. Baseline 3's session-bootstrap interval against the generator's expectation,
   reported SEPARATELY for tests with at least one observed failure and for
   tests with none. The second group is where the interval is honest about the
   resampling distribution and misleading about the test, and the measured
   under-coverage there is published rather than averaged away.
3. Baseline 1's blindness to runtime, as a measurement: the rank correlation
   between the flake rate and the generator's expected cost, against the same
   correlation for baseline 2.
4. Negative controls, anchored on the generator's expectation and asserted to
   fire.

NO CLOCK IS READ.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402

import numpy as np  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

from flakerank.baselines import (  # noqa: E402
    accumulated_wasted_time,
    analytic_expected_cost,
    flake_rate_ranking,
)
from flakerank.disagreement import kendall_tau_b, spearman_rho  # noqa: E402
from flakerank.geometric import expected_failed_attempts  # noqa: E402
from flakerank.history import load_history  # noqa: E402


def generator_expected_wasted_seconds(history, payload) -> np.ndarray:
    """The generator's own expected wasted seconds per test, in test-index order.

    See the module docstring for the formula. Returns zeros for a test with no
    recorded invocation.
    """
    change = int(payload["regime_change_session"])
    budget = history.retry_budget
    out = np.zeros(history.n_tests, dtype=np.float64)
    for index, item in enumerate(payload["tests"]):
        rows = history.attempt_test == index
        if not np.any(rows):
            continue
        sessions = history.attempt_session[rows]
        before = int(np.unique(sessions[sessions < change]).size)
        after = int(np.unique(sessions[sessions >= change]).size)
        probability = float(item["true_failure_probability"])
        changed = min(0.95, probability * float(item["true_regime_factor"]))
        failing_seconds = (
            float(item["true_mean_ms"]) / 1000.0 * float(item["true_fail_duration_factor"])
        )
        out[index] = failing_seconds * (
            before * expected_failed_attempts(probability, budget)
            + after * expected_failed_attempts(changed, budget)
        )
    return out


def main() -> int:
    """Validate the three baselines against the generator's expectation."""
    out = _bootstrap.Tee("validate_baselines_output.txt")
    out("The three baselines against the generator's own expectation -- P097 flakerank")
    out("")
    out("Validated BEFORE the learned component is benchmarked, because an")
    out("under-strength baseline inflates an apparent learned win.")
    out("")
    out(f"fast path (FLAKERANK_FAST=1): {_bootstrap.FAST}")
    out(f"declared seed: {_bootstrap.SEED}")
    out(f"bootstrap resamples: {_bootstrap.resamples()}")
    out(f"confidence: {_bootstrap.CONFIDENCE}")
    out(f"history: {_bootstrap.HISTORY.as_posix()} sha256 {_bootstrap.HISTORY_SHA256}")

    path = ROOT / _bootstrap.HISTORY
    history = load_history(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    summaries = history.summaries()
    names = tuple(item.name for item in summaries)
    index_of = {name: index for index, name in enumerate(history.test_names)}
    selected = np.asarray([index_of[name] for name in names], dtype=np.int64)

    truth_expectation = generator_expected_wasted_seconds(history, payload)[selected]
    rate = flake_rate_ranking(summaries)
    analytic = analytic_expected_cost(summaries, history.retry_budget)
    accumulated = accumulated_wasted_time(
        history,
        resamples=_bootstrap.resamples(),
        confidence=_bootstrap.CONFIDENCE,
        seed=_bootstrap.SEED,
    ).require_resampling_unit("session")

    out.rule("1. BASELINE 2 AGAINST THE GENERATOR'S EXPECTATION")
    out("")
    out("Baseline 2 estimates p and t_fail from the recorded attempts; the")
    out("generator's expectation uses the parameters those attempts were drawn")
    out("from. Agreement is therefore a statement about the MODEL plus the")
    out("estimation error, and the estimation error dominates for a rare flake.")
    out("")
    total_analytic = float(analytic.scores.sum())
    total_expected = float(truth_expectation.sum())
    out(f"aggregate analytic cost, s          : {total_analytic!r}")
    out(f"aggregate generator expectation, s  : {total_expected!r}")
    out(f"aggregate relative difference       : "
        f"{(total_analytic - total_expected) / total_expected!r}")
    out.check(
        "the aggregate analytic cost is within 25 per cent of the generator's "
        "expectation",
        abs(total_analytic - total_expected) / total_expected <= 0.25,
        f"{(total_analytic - total_expected) / total_expected:+.6f} relative",
    )
    out("")
    out("The 25 per cent criterion is declared rather than tuned. It is set by the")
    out("estimation error, not by the model: the aggregate is dominated by a")
    out("handful of slow tests whose failure counts are in the tens, so the")
    out("relative standard error of the aggregate is of order 1/sqrt(F) for the")
    out(f"F = {int(np.count_nonzero(history.attempt_failed))} recorded failures, "
        "concentrated in far fewer tests.")
    out("")
    out("Rank agreement with the generator's expectation, which is the question")
    out("the product actually asks:")
    out("")
    out(f"{'ranking':>44} {'Kendall tau-b':>15} {'Spearman rho':>14}")
    for label, scores in (
        ("baseline 1, flake rate", rate.scores),
        ("baseline 2, analytic expected cost", analytic.scores),
        ("baseline 3, accumulated wasted time", accumulated.ranking.scores),
    ):
        out(f"{label:>44} {kendall_tau_b(scores, truth_expectation):>15.9f}"
            f" {spearman_rho(scores, truth_expectation):>14.9f}")
    tau_rate = kendall_tau_b(rate.scores, truth_expectation)
    tau_analytic = kendall_tau_b(analytic.scores, truth_expectation)
    out("")
    out.check(
        "baseline 2 ranks the generator's expectation better than baseline 1 does",
        tau_analytic > tau_rate,
        f"tau-b {tau_analytic:.9f} against {tau_rate:.9f}",
    )

    out.rule("2. BASELINE 3's INTERVAL AGAINST THE GENERATOR'S EXPECTATION")
    out("")
    out("Resampling unit, declared on every interval and asserted before use:")
    out(f"  {accumulated.resampling_unit!r}, {accumulated.n_sessions} of them")
    out("")
    out("Coverage is reported SEPARATELY for tests with at least one observed")
    out("failure and for tests with none, because the two groups behave")
    out("differently and an average over them would hide the second.")
    out("")
    covered_with = 0
    total_with = 0
    covered_without = 0
    total_without = 0
    for position, summary in enumerate(summaries):
        interval = accumulated.per_test_intervals[position]
        covers = interval.contains(float(truth_expectation[position]))
        if summary.failed_attempts > 0:
            total_with += 1
            covered_with += int(covers)
        else:
            total_without += 1
            covered_without += int(covers)
    out(f"tests with at least one observed failure: {total_with}")
    out(f"  the 95 per cent interval covers the generator's expectation: "
        f"{covered_with} of {total_with}"
        f" = {covered_with / total_with:.6f}")
    out(f"tests with no observed failure          : {total_without}")
    if total_without:
        out(f"  the interval covers the generator's expectation: "
            f"{covered_without} of {total_without}"
            f" = {covered_without / total_without:.6f}")
    out("")
    out("THE SECOND GROUP IS THE PRODUCT'S OWN SUBJECT. For a test that never")
    out("failed in the window, every resample of the recorded sessions gives")
    out("zero, so the interval is [0, 0]. That is a correct statement about the")
    out("resampling distribution and a misleading statement about the test,")
    out("whose true rate is only bounded above. It is the reason baseline 2")
    out("exists: an analytic model extrapolates where an accumulation cannot.")
    out("Those tests with a true expectation above zero are counted here and")
    out("the under-coverage is published rather than averaged away.")
    out("")
    nonzero_without = int(
        np.count_nonzero(
            [
                truth_expectation[position] > 0.0
                for position, summary in enumerate(summaries)
                if summary.failed_attempts == 0
            ]
        )
    )
    out(f"  of the {total_without} never-failing tests, "
        f"{nonzero_without} have a true expected wasted time above zero")
    out.check(
        "coverage among tests with an observed failure is at least 0.70",
        covered_with / total_with >= 0.70,
        f"{covered_with}/{total_with} = {covered_with / total_with:.6f}",
    )
    out("")
    out("The 0.70 floor is deliberately below the nominal 0.95 and is declared")
    out("here with its reason, not tuned to the measurement. Three effects pull")
    out("the realised coverage down and all three are properties of the problem:")
    out("the percentile bootstrap is uncorrected for the strong right skew of a")
    out("wasted-time total; the generator's expectation includes the runtime")
    out("distribution's mean while the accumulation saw a finite sample of it;")
    out("and a test with two or three recorded failures has a resampling")
    out("distribution with a handful of atoms, for which no percentile interval")
    out("can have calibrated coverage.")
    out("")
    out(f"aggregate accumulated wasted time, s: {accumulated.total_seconds!r}")
    out(f"  interval, s                       : [{accumulated.total_interval.low!r}, "
        f"{accumulated.total_interval.high!r}]")
    out(f"  generator expectation, s          : {total_expected!r}")
    out(f"  interval covers it                : "
        f"{accumulated.total_interval.contains(total_expected)}")
    out.check(
        "the aggregate interval covers the generator's aggregate expectation",
        accumulated.total_interval.contains(total_expected),
        f"[{accumulated.total_interval.low:.3f}, {accumulated.total_interval.high:.3f}] "
        f"against {total_expected:.3f}",
    )

    out.rule("3. BASELINE 1's BLINDNESS TO RUNTIME, AS A MEASUREMENT")
    out("")
    out("The flake rate is a ratio of integer counts, so it does not move when")
    out("every duration in the history is scaled. The ground truth scales")
    out("exactly with it. That is the defect, stated as a number.")
    out("")
    scaled = type(history)(
        retry_budget=history.retry_budget,
        test_names=history.test_names,
        test_suites=history.test_suites,
        true_failure_probability=history.true_failure_probability,
        true_mean_ms=history.true_mean_ms,
        session_days=history.session_days,
        attempt_session=history.attempt_session,
        attempt_test=history.attempt_test,
        attempt_number=history.attempt_number,
        attempt_failed=history.attempt_failed,
        attempt_duration_ms=history.attempt_duration_ms * 7,
        source=history.source,
    )
    scaled_rate = flake_rate_ranking(scaled.summaries())
    scaled_truth = accumulated_wasted_time(scaled, resamples=20, seed=_bootstrap.SEED)
    rate_shift = float(np.abs(scaled_rate.scores - rate.scores).max())
    truth_ratio = scaled_truth.total_seconds / accumulated.total_seconds
    out("every duration multiplied by 7:")
    out(f"  largest change in any flake-rate score : {rate_shift!r}")
    out(f"  ratio of accumulated wasted time       : {truth_ratio!r}")
    out.check(
        "the flake-rate ranking does not move when every duration is scaled",
        rate_shift <= 1e-15,
        f"largest change {rate_shift:.3e}",
    )
    out.check(
        "the ground truth scales by the same factor",
        abs(truth_ratio - 7.0) <= 1e-12 * 7.0,
        f"{truth_ratio!r}",
    )

    out.rule("4. NEGATIVE CONTROLS, ANCHORED ON THE GENERATOR'S EXPECTATION")
    out("")
    out("Each control perturbs the TRUE value, not an output, by a factor chosen")
    out("so that leaving the interval is guaranteed rather than likely. The")
    out("controls are restricted to tests whose interval is non-degenerate,")
    out("because a [0, 0] interval is left by any positive perturbation and a")
    out("control that cannot fail proves nothing.")
    out("")
    wide = [
        position
        for position, summary in enumerate(summaries)
        if summary.failed_attempts >= 5
        and accumulated.per_test_intervals[position].width > 0.0
    ]
    out(f"tests with at least 5 observed failures and a non-degenerate interval: "
        f"{len(wide)}")
    for factor, label in ((100.0, "100x too large"), (0.01, "100x too small")):
        fired = 0
        for position in wide:
            interval = accumulated.per_test_intervals[position]
            perturbed = float(truth_expectation[position]) * factor
            fired += int(not interval.contains(perturbed))
        out(f"  a true value {label}: the interval excludes it for "
            f"{fired} of {len(wide)} tests")
        out.control(
            f"the per-test interval rejects a true value {label}",
            fired == len(wide),
            must_fire=True,
            detail=f"{fired} of {len(wide)}",
        )
    out("")
    out("CONTROL that must STAY SILENT: the unperturbed generator expectation is")
    out("NOT rejected for a clear majority of those same tests. If it were, the")
    out("controls above would be measuring a broken interval rather than a")
    out("broken value.")
    silent = sum(
        int(accumulated.per_test_intervals[position].contains(
            float(truth_expectation[position])))
        for position in wide
    )
    out(f"  the interval covers the unperturbed true value for {silent} of {len(wide)} tests")
    out.control(
        "the per-test interval rejects the unperturbed true value for most tests",
        silent < 0.5 * len(wide),
        must_fire=False,
        detail=f"{silent} of {len(wide)} covered",
    )
    out("")
    out("CONTROL that must FIRE: the aggregate interval rejects an aggregate")
    out("expectation scaled by 2, anchored on the true aggregate and checked to")
    out("lie outside the interval before the control is evaluated.")
    inflated = 2.0 * total_expected
    out(f"  2x the true aggregate: {inflated!r} s")
    out(f"  the interval is [{accumulated.total_interval.low!r}, "
        f"{accumulated.total_interval.high!r}] s")
    out.control(
        "the aggregate interval rejects twice the true aggregate",
        not accumulated.total_interval.contains(inflated),
        must_fire=True,
        detail=f"{inflated:.3f} against upper {accumulated.total_interval.high:.3f}",
    )
    return out.finish()


if __name__ == "__main__":
    sys.exit(main())
