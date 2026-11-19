"""The headline: how far the cost ranking and the flake-rate ranking disagree.

This is the product's headline output, so this is where it is measured, with
its interval, its tie structure, the identity of the tests that move most, and
the cost-weighted capture that says what the disagreement is worth
operationally.

Four things are checked:

1. The rank statistics against ``scipy.stats``, which is an independently
   written implementation of both. Agreement there is a cross-check against
   another implementation and is NOT the independently-derived-bound check,
   which lives in ``validate_bounds.py``.
2. Hand-calculated known answers, printed with the arithmetic beside them, so a
   reader can check the implementation against the cited formulae without
   running anything.
3. The measured disagreement on the committed history, with its bootstrap
   interval over the declared resampling unit.
4. Negative controls: a ranking against itself must give 1, a reversal must
   give -1, and a permuted ranking must be rejected as agreeing.

Sources for the statistics, repeated here because this is the evidence file:

* Kendall, M. G. (1938), "A new measure of rank correlation", Biometrika
  30(1/2), 81-93 -- the statistic.
* Kendall, M. G. (1945), "The treatment of ties in ranking problems",
  Biometrika 33(3), 239-251 -- the tie correction that makes it tau-b.
* Kendall, M. G. & Gibbons, J. D. (1990), Rank Correlation Methods, 5th
  edition, Edward Arnold -- the textbook treatment of both, and of the midrank
  correction for Spearman.
* Spearman, C. (1904), "The proof and measurement of association between two
  things", American Journal of Psychology 15(1), 72-101 -- rho.
* Efron, B. (1979), "Bootstrap methods: another look at the jackknife", Annals
  of Statistics 7(1), 1-26, and Efron, B. & Tibshirani, R. J. (1993), An
  Introduction to the Bootstrap, Chapman & Hall -- the percentile interval.

NO CLOCK IS READ.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

from flakerank.disagreement import kendall_tau_b, midranks, spearman_rho  # noqa: E402
from flakerank.history import load_history  # noqa: E402
from flakerank.ranking import rank_tests  # noqa: E402

#: Hand-calculated cases, as ``(label, x, y, expected tau-b, expected rho)``.
HAND_CASES = (
    (
        "no ties, one inversion: C = 5, D = 1, tau = 4/6",
        (1.0, 2.0, 3.0, 4.0),
        (1.0, 2.0, 4.0, 3.0),
        4.0 / 6.0,
        0.8,
    ),
    (
        "one tie group each side: C = 4, D = 0, n1 = n2 = 1, tau = 4/sqrt(5*5)",
        (1.0, 1.0, 2.0, 3.0),
        (1.0, 2.0, 2.0, 3.0),
        0.8,
        3.75 / 4.5,
    ),
    (
        "perfect agreement: C = n0, D = 0",
        (3.0, 1.0, 4.0, 1.5, 9.0),
        (3.0, 1.0, 4.0, 1.5, 9.0),
        1.0,
        1.0,
    ),
    (
        "exact reversal: D = n0",
        (1.0, 2.0, 3.0, 4.0, 5.0),
        (-1.0, -2.0, -3.0, -4.0, -5.0),
        -1.0,
        -1.0,
    ),
)


def main() -> int:
    """Measure the headline disagreement and check the statistics four ways."""
    out = _bootstrap.Tee("validate_disagreement_output.txt")
    out("The headline disagreement measure -- P097 flakerank")
    out("")
    out("The product's headline output is not a ranking. It is how far the cost")
    out("ranking and the flake-rate ranking disagree, with an interval on that")
    out("disagreement and the identity of the tests that move most.")
    out("")
    out(f"fast path (FLAKERANK_FAST=1): {_bootstrap.FAST}")
    out(f"declared seed: {_bootstrap.SEED}")
    out(f"bootstrap resamples: {_bootstrap.resamples()}")
    out(f"confidence: {_bootstrap.CONFIDENCE}")
    out(f"top-k for every capture figure: {_bootstrap.TOP_K}")
    out(f"history: {_bootstrap.HISTORY.as_posix()} sha256 {_bootstrap.HISTORY_SHA256}")

    out.rule("1. HAND-CALCULATED KNOWN ANSWERS")
    out("")
    out("Kendall tau-b, equation (1) of src/flakerank/disagreement.py:")
    out("  tau_b = (C - D) / sqrt((n0 - n1)(n0 - n2))")
    out("with n0 = n(n-1)/2 pairs, n1 and n2 the tied-pair counts of each vector.")
    out("Spearman rho is the Pearson correlation of midranks; with no ties that")
    out("reduces to 1 - 6 sum(d^2)/(n^3 - n), and with ties the shortcut is wrong")
    out("and the midrank form is the standard correction.")
    out("")
    out("midranks([1, 1, 2, 3]): the two 1s span ranks 1 and 2, so both get 1.5")
    out(f"  computed: {midranks(np.asarray([1.0, 1.0, 2.0, 3.0])).tolist()}")
    out("")
    for label, x, y, tau_expected, rho_expected in HAND_CASES:
        first = np.asarray(x, dtype=np.float64)
        second = np.asarray(y, dtype=np.float64)
        tau = kendall_tau_b(first, second)
        rho = spearman_rho(first, second)
        out(f"{label}")
        out(f"  x = {list(x)}")
        out(f"  y = {list(y)}")
        out(f"  tau-b expected {tau_expected!r}, computed {tau!r}")
        out(f"  rho   expected {rho_expected!r}, computed {rho!r}")
        out.check(
            f"tau-b known answer: {label[:40]}",
            abs(tau - tau_expected) <= 1e-14 * max(1.0, abs(tau_expected)),
            f"{tau!r}",
        )
        out.check(
            f"rho known answer: {label[:40]}",
            abs(rho - rho_expected) <= 1e-14 * max(1.0, abs(rho_expected)),
            f"{rho!r}",
        )
        out("")

    out.rule("2. CROSS-CHECK AGAINST SCIPY, AN INDEPENDENT IMPLEMENTATION")
    out("")
    out("scipy.stats.kendalltau(variant='b') and scipy.stats.spearmanr are a")
    out("separately written implementation of both statistics. This is a")
    out("cross-check against another implementation; it is NOT the")
    out("independently-derived-bound check, which is in validate_bounds.py and")
    out("which catches what an implementation comparison cannot.")
    out("")
    history = load_history(ROOT / _bootstrap.HISTORY)
    report = rank_tests(
        history,
        top_k=_bootstrap.TOP_K,
        resamples=_bootstrap.resamples(),
        confidence=_bootstrap.CONFIDENCE,
        seed=_bootstrap.SEED,
    )
    truth = report.truth.ranking.scores
    rate = report.rate.scores
    analytic = report.analytic.scores
    out(f"{'pair':>34} {'ours tau-b':>14} {'scipy tau-b':>14} {'rel diff':>11}")
    worst_tau = 0.0
    worst_rho = 0.0
    for label, left, right in (
        ("ground truth vs flake rate", truth, rate),
        ("ground truth vs analytic cost", truth, analytic),
        ("analytic cost vs flake rate", analytic, rate),
    ):
        ours = kendall_tau_b(left, right)
        theirs = float(stats.kendalltau(left, right, variant="b").statistic)
        relative = abs(ours - theirs) / max(abs(theirs), 1e-300)
        worst_tau = max(worst_tau, relative)
        out(f"{label:>34} {ours:>14.11f} {theirs:>14.11f} {relative:>11.3e}")
    out("")
    out(f"{'pair':>34} {'ours rho':>14} {'scipy rho':>14} {'rel diff':>11}")
    for label, left, right in (
        ("ground truth vs flake rate", truth, rate),
        ("ground truth vs analytic cost", truth, analytic),
        ("analytic cost vs flake rate", analytic, rate),
    ):
        ours = spearman_rho(left, right)
        theirs = float(stats.spearmanr(left, right).statistic)
        relative = abs(ours - theirs) / max(abs(theirs), 1e-300)
        worst_rho = max(worst_rho, relative)
        out(f"{label:>34} {ours:>14.11f} {theirs:>14.11f} {relative:>11.3e}")
    out("")
    out.check(
        "our tau-b agrees with scipy's to 1e-12 relative on every pair",
        worst_tau <= 1e-12,
        f"worst {worst_tau:.3e}",
    )
    out.check(
        "our rho agrees with scipy's to 1e-12 relative on every pair",
        worst_rho <= 1e-12,
        f"worst {worst_rho:.3e}",
    )
    out("")
    out("The ONE declared difference: with either vector entirely tied, the")
    out("denominator of equation (1) is zero. This implementation returns 0.0 and")
    out("scipy returns NaN. The difference is declared in the module docstring")
    out("and is the only place the two are allowed to disagree.")
    tied = np.full(6, 3.0)
    untied = np.arange(6.0)
    out(f"  ours : {kendall_tau_b(tied, untied)!r}")
    out(f"  scipy: {float(stats.kendalltau(tied, untied, variant='b').statistic)!r}")
    out.check(
        "a fully tied vector gives 0.0 here and NaN in scipy, as declared",
        kendall_tau_b(tied, untied) == 0.0
        and np.isnan(stats.kendalltau(tied, untied, variant="b").statistic),
    )

    out.rule("3. THE MEASURED DISAGREEMENT ON THE COMMITTED HISTORY")
    out("")
    out(f"tests ranked: {report.n_tests}")
    out(f"sessions    : {report.n_sessions}")
    out(f"attempts    : {report.n_attempts}")
    out("")
    for disagreement in (
        report.truth_vs_rate,
        report.truth_vs_analytic,
        report.analytic_vs_rate,
    ):
        tau_interval = disagreement.tau_b_interval
        rho_interval = disagreement.rho_interval
        out(f"{disagreement.label_x}")
        out(f"  versus {disagreement.label_y}")
        out(f"  Kendall tau-b : {disagreement.tau_b!r}")
        out(f"    interval    : [{tau_interval.low!r}, {tau_interval.high!r}]")
        out(f"    resampling  : {tau_interval.resamples} resamples of "
            f"{tau_interval.n_units} {tau_interval.resampling_unit}s at "
            f"{tau_interval.confidence}")
        out(f"  Spearman rho  : {disagreement.rho!r}")
        out(f"    interval    : [{rho_interval.low!r}, {rho_interval.high!r}]")
        out(f"  pairs n0      : {disagreement.ties.n_pairs}")
        out(f"  tied in x n1  : {disagreement.ties.tied_x_pairs}")
        out(f"  tied in y n2  : {disagreement.ties.tied_y_pairs}")
        out(f"  concordant C  : {disagreement.ties.concordant}")
        out(f"  discordant D  : {disagreement.ties.discordant}")
        out(f"  tie groups in x: {list(disagreement.ties.tie_groups_x)}")
        out(f"  tie groups in y: {list(disagreement.ties.tie_groups_y)}")
        out(f"  max rank displacement      : {disagreement.max_displacement}")
        out(f"  mean absolute displacement : {disagreement.mean_absolute_displacement!r}")
        out(f"  top-{disagreement.top_k} overlap: {disagreement.top_k_overlap} of "
            f"{disagreement.top_k}")
        out("")
        out.check(
            f"the interval on tau-b contains the point estimate "
            f"({disagreement.label_y})",
            tau_interval.contains(disagreement.tau_b),
            f"{disagreement.tau_b:.9f} in [{tau_interval.low:.9f}, {tau_interval.high:.9f}]",
        )
        out.check(
            f"both intervals declare the 'test' resampling unit "
            f"({disagreement.label_y})",
            tau_interval.resampling_unit == "test"
            and rho_interval.resampling_unit == "test",
        )
        out("")

    out("TIE HANDLING, stated because it is where two implementations of tau-b")
    out("can differ. A pair tied in either vector counts as neither concordant")
    out("nor discordant AND is removed from the denominator for that vector")
    out("separately, which is what makes this tau-b rather than tau-a. The")
    out("flake-rate vector here is heavily tied -- every test with no observed")
    out(f"failure has rate exactly 0, and there are "
        f"{report.truth_vs_rate.ties.tied_y_pairs} such pairs of "
        f"{report.truth_vs_rate.ties.n_pairs} -- so the correction is not")
    out("cosmetic on this data.")

    out.rule("4. THE TESTS THAT MOVE MOST")
    out("")
    out("Ground truth against flake rate. Negative displacement means the cost")
    out("ranking puts the test HIGHER, nearer position 1, than the rate does.")
    out("")
    out(f"{'test':>44} {'cost rank':>10} {'rate rank':>10} {'move':>6} "
        f"{'wasted s':>12} {'rate':>10}")
    for mover in report.truth_vs_rate.movers:
        out(f"{mover.name:>44} {mover.rank_x:>10} {mover.rank_y:>10} "
            f"{mover.displacement:>+6} {mover.score_x:>12.3f} {mover.score_y:>10.6f}")
    out("")
    pinned = next(
        (mover for mover in report.truth_vs_rate.movers
         if mover.name == "soak/test_validation_probe_1_in_44"),
        None,
    )
    if pinned is None:
        out.unavailable(
            "the motivating case appears among the movers",
            "soak/test_validation_probe_1_in_44 is not in the top movers",
        )
    else:
        out("THE MOTIVATING CASE. This mission's own validation probe failed 1 time")
        out("in 44 and no number of agreeing serial verifications saw it. In the")
        out("committed history the test with that rate is:")
        out(f"  {pinned.name}")
        out(f"  cost rank {pinned.rank_x} of {report.n_tests}, "
            f"rate rank {pinned.rank_y} of {report.n_tests}")
        out(f"  displacement {pinned.displacement:+d} positions")
        out(f"  accumulated wasted time {pinned.score_x!r} s")
        out(f"  measured flake rate {pinned.score_y!r}")
        out.check(
            "the cost ranking puts the motivating case higher than the rate ranking",
            pinned.displacement < 0,
            f"{pinned.displacement:+d} positions",
        )

    out.rule("5. WHAT THE DISAGREEMENT IS WORTH: COST-WEIGHTED CAPTURE")
    out("")
    out("A rank correlation weights every test equally. The operational question")
    out("weights by cost: if a team fixes the top ten this ranking names, how")
    out("much of the measured waste goes away?")
    out("")
    out(f"wasted time actually recorded, s     : {report.truth.total_seconds!r}")
    out(f"  interval, s                        : [{report.truth.total_interval.low!r}, "
        f"{report.truth.total_interval.high!r}]")
    out(f"  resampling unit                    : {report.truth.resampling_unit}, "
        f"{report.truth.total_interval.n_units} of them")
    out("")
    out(f"fraction held by the flake-rate top {report.top_k}  : {report.capture_by_rate!r}")
    out(f"fraction held by the analytic top {report.top_k}    : {report.capture_by_analytic!r}")
    out(f"fraction held by the best possible top {report.top_k}: {report.capture_oracle!r}")
    out(f"shortfall of the flake-rate ranking    : {report.capture_shortfall!r}")
    out("")
    out.check(
        "the analytic top-k holds more of the measured waste than the flake-rate "
        "top-k",
        report.capture_by_analytic > report.capture_by_rate,
        f"{report.capture_by_analytic:.9f} against {report.capture_by_rate:.9f}",
    )
    out.check(
        "no ranking beats the oracle",
        report.capture_by_analytic <= report.capture_oracle + 1e-12
        and report.capture_by_rate <= report.capture_oracle + 1e-12,
    )

    out.rule("6. NEGATIVE CONTROLS")
    out("")
    out("Each control is anchored on a construction whose answer is known")
    out("exactly, not on an output.")
    out("")
    out.control(
        "tau-b of the ground truth against itself is not 1",
        abs(kendall_tau_b(truth, truth.copy()) - 1.0) > 1e-12,
        must_fire=False,
        detail=f"{kendall_tau_b(truth, truth.copy())!r}",
    )
    out.control(
        "tau-b of the ground truth against its exact reversal is not -1",
        abs(kendall_tau_b(truth, -truth) + 1.0) > 1e-12,
        must_fire=False,
        detail=f"{kendall_tau_b(truth, -truth)!r}",
    )
    generator = np.random.default_rng(_bootstrap.SEED)
    permuted = generator.permutation(truth)
    permuted_tau = kendall_tau_b(truth, permuted)
    out(f"  a permutation of the ground truth gives tau-b {permuted_tau!r}")
    out.control(
        "a permuted ranking is detected as disagreeing with the ground truth",
        abs(permuted_tau) < 0.5,
        must_fire=True,
        detail=f"tau-b {permuted_tau:.6f}, well below the measured "
               f"{report.truth_vs_rate.tau_b:.6f}",
    )
    out("")
    out("The permutation control is the one that matters: it shows the measured")
    out(f"disagreement of {report.truth_vs_rate.tau_b!r} is a MEASUREMENT of")
    out("partial agreement and not an artifact of the statistic, because the")
    out("statistic goes to roughly zero on a ranking with no relationship at")
    out("all.")
    return out.finish()


if __name__ == "__main__":
    sys.exit(main())
