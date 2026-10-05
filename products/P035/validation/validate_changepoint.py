"""Validation: change-point detection false-alarm rate and localisation accuracy.

Two quantities are measured.

**Per-segment false-alarm rate.**  The threshold is a Monte-Carlo quantile of
the maximised standardised mean-shift statistic under the no-change hypothesis.
Calibrated on one Monte-Carlo sample and measured on an independent one, the
achieved rate must match the target within the combined standard error.

**Localisation accuracy.**  For a single step of known size at a known index,
the distribution of ``tau_hat - tau_true`` is tabulated rather than summarised
to a mean, because it is heavy-tailed at small shift sizes.

A third quantity is reported but not checked: the *series-wide* false-alarm rate
of binary segmentation, which is higher than the per-segment target because the
threshold is applied at every recursion level.  That is a property of the
recursion, it is stated in the module docstring of
:mod:`telemetryool.changepoint`, and the measured size of the inflation is
printed here so a user can account for it.

References
----------
Hinkley, D. V. (1970). "Inference about the change-point in a sequence of random
    variables." *Biometrika* 57(1), 1-17.
Scott, A. J. and Knott, M. (1974). "A Cluster Analysis Method for Grouping Means
    in the Analysis of Variance." *Biometrics* 30(3), 507-512.
Killick, R., Fearnhead, P. and Eckley, I. A. (2012). "Optimal Detection of
    Changepoints With a Linear Computational Cost." *JASA* 107(500), 1590-1598.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from _reporting import Report  # noqa: E402

from telemetryool.calibration import binomial_se, estimate_rate  # noqa: E402
from telemetryool.changepoint import (  # noqa: E402
    calibrate_change_point_threshold,
    detect_change_points,
    max_mean_shift_statistic,
)

SEGMENT = 200
ALPHA = 0.05
CAL_SIMS = 40_000
MEASURE_SIMS = 40_000
TRIALS = 1000
TRUE_TAU = 120
Z_LIMIT = 3.5


def statistic_batch(draws: np.ndarray) -> np.ndarray:
    """Maximised statistic for each row, vectorised."""
    n = draws.shape[1]
    cs = np.cumsum(draws, axis=1)
    m = np.arange(1, n, dtype=float)
    centred = cs[:, :-1] - (m / n)[None, :] * cs[:, -1][:, None]
    return (np.abs(centred) / np.sqrt(m * (n - m) / n)[None, :]).max(axis=1)


def main() -> int:
    report = Report(
        "validate_changepoint",
        "Change-point detection: designed false-alarm rate and localisation accuracy",
    )
    report.line(f"segment length n        : {SEGMENT} samples")
    report.line(f"target per-segment alpha: {ALPHA}")
    report.line(f"calibration replicates  : {CAL_SIMS}")
    report.line(f"measurement replicates  : {MEASURE_SIMS} (independent seed)")
    report.line(
        f"combined SE             : "
        f"{np.sqrt(ALPHA * (1 - ALPHA) * (1 / CAL_SIMS + 1 / MEASURE_SIMS)):.7f}"
    )
    report.line(f"agreement band          : |z| < {Z_LIMIT} combined standard errors")

    report.section("Threshold calibration and independent measurement")
    threshold = calibrate_change_point_threshold(
        SEGMENT, ALPHA, CAL_SIMS, np.random.default_rng(101)
    )
    report.line(f"threshold               : {threshold.threshold:.9f} (dimensionless)")
    report.line(
        f"in-sample achieved alpha: {threshold.achieved_alpha:.7f} "
        f"+/- {threshold.standard_error:.7f}"
    )
    rng = np.random.default_rng(202)
    stats = statistic_batch(rng.standard_normal((MEASURE_SIMS, SEGMENT)))
    k = int((stats > threshold.threshold).sum())
    est = estimate_rate(k, MEASURE_SIMS)
    combined = float(
        np.sqrt(ALPHA * (1 - ALPHA) * (1 / CAL_SIMS + 1 / MEASURE_SIMS))
    )
    z = (est.rate - ALPHA) / combined
    report.line(
        f"independent measurement : {est.rate:.7f} +/- {est.standard_error:.7f} "
        f"(binomial SE), {k} of {MEASURE_SIMS}"
    )
    report.line(f"95 % Wilson             : [{est.wilson_low:.7f}, {est.wilson_high:.7f}]")
    report.line(f"z against the target    : {z:+.2f} combined SE")
    report.check(
        "per-segment false-alarm rate matches the target on independent data",
        abs(z) < Z_LIMIT,
        f"measured={est.rate:.7f} target={ALPHA} z={z:+.2f}",
    )

    report.section("Threshold grows with segment length and shrinks with alpha")
    report.line(
        f"{'n':>6s} {'alpha=0.10':>12s} {'alpha=0.05':>12s} {'alpha=0.01':>12s}"
    )
    grid = {}
    for n in (50, 100, 200, 400):
        row = []
        for alpha in (0.10, 0.05, 0.01):
            th = calibrate_change_point_threshold(n, alpha, 8000, np.random.default_rng(7))
            row.append(th.threshold)
            grid[(n, alpha)] = th.threshold
        report.line(f"{n:6d} {row[0]:12.6f} {row[1]:12.6f} {row[2]:12.6f}")
    report.check(
        "threshold increases with segment length at every alpha",
        all(
            grid[(50, a)] < grid[(100, a)] < grid[(200, a)] < grid[(400, a)]
            for a in (0.10, 0.05, 0.01)
        ),
    )
    report.check(
        "threshold increases as alpha tightens at every segment length",
        all(grid[(n, 0.10)] < grid[(n, 0.05)] < grid[(n, 0.01)] for n in (50, 100, 200, 400)),
    )

    report.section("Localisation accuracy of the maximiser")
    report.line(
        f"A single step at index {TRUE_TAU} of a {SEGMENT}-sample standard-normal series, "
        f"{TRIALS} seeds per shift size."
    )
    report.line("The error distribution is tabulated, not reduced to a mean: it is")
    report.line("heavy-tailed at small shift sizes, where the maximiser can land anywhere.")
    report.line("")
    report.line(
        f"{'shift':>6s} {'exact':>8s} {'|e|<=1':>8s} {'|e|<=2':>8s} {'|e|<=5':>8s} "
        f"{'median |e|':>11s} {'p90 |e|':>9s} {'max |e|':>9s} {'Pd':>8s}"
    )
    results = {}
    for shift in (0.5, 1.0, 2.0, 3.0, 5.0):
        errors = np.empty(TRIALS, dtype=np.int64)
        detected = np.zeros(TRIALS, dtype=bool)
        for i in range(TRIALS):
            trial_rng = np.random.default_rng(90000 + 1000 * int(shift * 10) + i)
            x = trial_rng.standard_normal(SEGMENT)
            x[TRUE_TAU:] += shift
            errors[i] = max_mean_shift_statistic(x).index - TRUE_TAU
            detected[i] = max_mean_shift_statistic(x).statistic > threshold.threshold
        absolute = np.abs(errors)
        results[shift] = (absolute, detected)
        report.line(
            f"{shift:6.1f} {float((absolute == 0).mean()):8.3f} "
            f"{float((absolute <= 1).mean()):8.3f} {float((absolute <= 2).mean()):8.3f} "
            f"{float((absolute <= 5).mean()):8.3f} {float(np.median(absolute)):11.1f} "
            f"{float(np.percentile(absolute, 90)):9.1f} {int(absolute.max()):9d} "
            f"{float(detected.mean()):8.3f}"
        )
    report.line("")
    report.line(f"Binomial SE on each fraction at {TRIALS} trials, worst case (p = 0.5): "
                f"{binomial_se(0.5, TRIALS):.4f}")
    report.line("")
    for shift, bound in ((2.0, 0.90), (3.0, 0.95), (5.0, 0.99)):
        absolute, detected = results[shift]
        frac = float((absolute <= 2).mean())
        report.check(
            f"shift {shift} sigma: maximiser within 2 samples at least "
            f"{bound:.0%} of the time",
            frac >= bound,
            f"measured {frac:.3f}",
        )
    for shift, bound in ((1.0, 0.70), (2.0, 0.99), (3.0, 0.999)):
        _absolute, detected = results[shift]
        frac = float(detected.mean())
        report.check(
            f"shift {shift} sigma: detection probability at least {bound}",
            frac >= bound,
            f"measured {frac:.3f}",
        )
    report.line("")
    report.line("At a 0.5 sigma shift over 200 samples the statistic is barely above its")
    report.line("null distribution, so neither detection nor localisation works; that row is")
    report.line("reported and not checked, because a detector that could do it would be")
    report.line("violating the information content of the data.")

    report.section("Series-wide false-alarm rate of binary segmentation")
    report.line("The threshold is calibrated per segment.  Binary segmentation applies it at")
    report.line("every recursion level, so the probability of at least one spurious change")
    report.line("point in a pure-noise series can exceed the per-segment target.  How much it")
    report.line("exceeds it is an empirical question, measured here over 4000 pure-noise")
    report.line("series:")
    report.line("")
    report.line(
        f"{'min_segment':>12s} {'max_depth':>10s} {'any false CP':>13s} {'SE':>9s} "
        f"{'mean count':>11s} {'inflation':>10s}"
    )
    noise_rng = np.random.default_rng(303)
    series = noise_rng.standard_normal((4000, SEGMENT))
    for min_segment, max_depth in ((20, 20), (20, 2), (50, 20)):
        counts = np.array(
            [
                len(detect_change_points(row, threshold.threshold, 1.0, min_segment, max_depth))
                for row in series
            ]
        )
        any_false = float((counts > 0).mean())
        report.line(
            f"{min_segment:12d} {max_depth:10d} {any_false:13.5f} "
            f"{binomial_se(any_false, 4000):9.5f} {counts.mean():11.4f} "
            f"{any_false / ALPHA:10.2f}"
        )
    report.line("")
    report.line("Reported, not checked.  The measured inflation is modest, because the")
    report.line("top-level test gates the recursion: a series that fails to produce one")
    report.line("change point never reaches a second level, so the extra tests are not")
    report.line("independent of the first.  The mean number of spurious change points per")
    report.line("series exceeds the any-false-CP probability only slightly, which is the")
    report.line("same statement.  A user who needs a guaranteed series-wide rate should")
    report.line("calibrate the threshold against this measurement at their own segment")
    report.line("length, or use a penalised method such as PELT (Killick, Fearnhead and")
    report.line("Eckley 2012), which the `ruptures` package implements and this module does")
    report.line("not.")
    return report.finish()


if __name__ == "__main__":
    sys.exit(main())
