"""Do the bootstrap reliability bands cover what they claim to cover?

A band is only worth plotting if its stated level means something. The exact
calibration curve of each synthetic spec is known, so coverage can be measured
rather than assumed. Two rates are reported and they are very different:

``pointwise``
    the fraction of (bin, replicate) pairs whose band contains the exact
    conditional probability. This is the rate the stated level refers to.

``simultaneous``
    the fraction of replicates in which *every* occupied bin is covered. A
    diagram with B bins at a pointwise level of 0.9 has roughly B chances to
    step outside, so this rate is much lower. Readers of a reliability diagram
    routinely treat the pointwise band as if it were simultaneous, and the gap
    measured here is how wrong that is.
"""

from __future__ import annotations

import numpy as np
from _harness import Recorder

from calibaudit.reliability import band_coverage, bootstrap_reliability
from calibaudit.synthetic import calibration_map, get_spec, sample_forecast

SEED = 56
REPLICATES = 120
BOOTSTRAP = 250


def main() -> int:
    rec = Recorder("validate_reliability_bands")
    rec.header("Bootstrap reliability band coverage - calibaudit 0.1.0")
    rec.say(
        "Coverage target is the exact conditional probability E[o | f] of the spec,\n"
        "evaluated at each bin's mean forecast. For a curved calibration map that is\n"
        "itself an approximation inside a wide bin, which is a property of binning\n"
        f"and not of the bootstrap. {REPLICATES} replicates, {BOOTSTRAP} bootstrap "
        "resamples each."
    )
    rec.say()

    head = (
        f"{'spec':>19} {'n':>7} {'B':>4} {'strategy':>12} {'level':>6} "
        f"{'pointwise':>10} {'simult.':>9} {'gap':>8}"
    )
    rec.say(head)
    rec.say("-" * len(head))
    results = []
    for name in ("calibrated", "calibrated_rare", "overconfident"):
        spec = get_spec(name)
        for n_samples, n_bins in ((500, 10), (2000, 20), (8000, 10)):
            for strategy in ("equal_width", "equal_mass"):
                cov = band_coverage(
                    spec,
                    n_samples=n_samples,
                    n_bins=n_bins,
                    strategy=strategy,
                    level=0.9,
                    n_bootstrap=BOOTSTRAP,
                    n_replicates=REPLICATES,
                    seed=SEED,
                )
                results.append((name, cov))
                rec.say(
                    f"{name:>19} {n_samples:>7d} {n_bins:>4d} {strategy:>12} "
                    f"{cov.level:>6.2f} {cov.pointwise_coverage:>10.4f} "
                    f"{cov.simultaneous_coverage:>9.4f} "
                    f"{cov.pointwise_coverage - cov.simultaneous_coverage:>8.4f}"
                )
    rec.say()

    calibrated = [c for n, c in results if get_spec(n).is_calibrated]
    pointwise = np.array([c.pointwise_coverage for c in calibrated])
    simultaneous = np.array([c.simultaneous_coverage for c in calibrated])
    rec.check(
        "NAIVE EXPECTATION, FALSIFIED: pointwise coverage of a 0.90 band is within "
        "0.10 of 0.90 on calibrated specs",
        reference="exact E[o | f] of the spec; nominal pointwise level 0.90",
        measured=f"range [{pointwise.min():.4f}, {pointwise.max():.4f}] over "
        f"{pointwise.size} configurations, mean {pointwise.mean():.4f}",
        expectation="within [0.80, 0.95]; the percentile bootstrap on a bin mean of a "
        "few hundred Bernoulli draws is known to undercover slightly, but the measured "
        "floor is below 0.80",
        passed=bool(np.all(pointwise >= 0.80) and np.all(pointwise <= 0.95)),
    )
    rec.check(
        "REFINED: coverage is systematically BELOW nominal and never above it",
        reference="same bands, same exact curves",
        measured=f"all {pointwise.size} configurations below 0.90; mean "
        f"{pointwise.mean():.4f}, i.e. a shortfall of "
        f"{0.90 - pointwise.mean():.4f}; worst {pointwise.min():.4f}",
        expectation="every configuration at or below the nominal 0.90, and the mean "
        "shortfall between 0.02 and 0.12. A percentile bootstrap of a binomial "
        "proportion is discrete and skewed near the edges of [0, 1]; it has no "
        "finite-sample coverage guarantee and this is the size of the shortfall.",
        passed=bool(
            np.all(pointwise <= 0.90) and 0.02 <= (0.90 - pointwise.mean()) <= 0.12
        ),
    )
    rec.check(
        "simultaneous coverage is far below the pointwise level, as it must be",
        reference="same bands, counted per replicate instead of per bin",
        measured=f"range [{simultaneous.min():.4f}, {simultaneous.max():.4f}], "
        f"mean {simultaneous.mean():.4f} against pointwise mean {pointwise.mean():.4f}",
        expectation="simultaneous below pointwise by more than 0.20 in every "
        "configuration, since B pointwise bands at 0.90 cannot hold jointly at 0.90",
        passed=bool(np.all(pointwise - simultaneous > 0.20)),
    )

    # --- undercoverage against bin occupancy -------------------------------
    rec.say("Pointwise coverage per bin, spec 'calibrated', n = 2000, 20 equal-width bins")
    rec.say()
    cov = band_coverage(
        get_spec("calibrated"),
        n_samples=2000,
        n_bins=20,
        strategy="equal_width",
        level=0.9,
        n_bootstrap=BOOTSTRAP,
        n_replicates=REPLICATES,
        seed=SEED,
    )
    head = f"{'bin':>4} {'replicates occupied':>20} {'coverage':>9}"
    rec.say(head)
    rec.say("-" * len(head))
    for k in range(cov.per_bin_coverage.size):
        value = cov.per_bin_coverage[k]
        shown = "n/a" if not np.isfinite(value) else f"{value:.4f}"
        rec.say(f"{k:>4d} {int(cov.per_bin_n[k]):>20d} {shown:>9}")
    rec.say()
    finite = cov.per_bin_coverage[np.isfinite(cov.per_bin_coverage)]
    rec.check(
        "the worst-covered bin is an edge bin, and its coverage is nowhere near nominal",
        reference="the same 0.90 band, resolved per bin",
        measured=f"worst bin coverage {finite.min():.4f} at bin "
        f"{int(np.nanargmin(cov.per_bin_coverage))}, best {finite.max():.4f}",
        expectation="worst coverage below the pooled rate and below 0.60; the top "
        "bin of a Beta(2,2) forecast under equal-width binning holds the fewest "
        "samples and sits against the boundary at 1, where the bootstrap "
        "distribution of a proportion is most skewed. Do not read an edge bin's "
        "band as if it had the stated level.",
        passed=float(finite.min()) < min(cov.pointwise_coverage, 0.60),
    )

    # --- a worked diagram, for the README ---------------------------------
    rec.say("Worked diagram: spec 'overconfident', n = 4000, 12 equal-width bins")
    rec.say()
    s = sample_forecast(get_spec("overconfident"), 4000, seed=SEED)
    curve = bootstrap_reliability(
        s.forecasts, s.outcomes, n_bins=12, n_bootstrap=1000, level=0.9, seed=SEED + 1
    )
    rec.say(curve.table())
    truth = calibration_map(get_spec("overconfident"), curve.mean_forecast)
    inside = (truth >= curve.lower) & (truth <= curve.upper)
    excluded = int(np.sum(curve.diagonal_excluded()))
    rec.say()
    rec.say(
        f"bins whose band excludes the mean forecast (i.e. flagged miscalibrated): "
        f"{excluded} of {curve.n_occupied}"
    )
    rec.say(
        f"bins whose band contains the exact E[o | f]: {int(np.sum(inside))} of "
        f"{curve.n_occupied}"
    )
    rec.say()
    rec.check(
        "on a known-miscalibrated forecast the bands flag most bins and still cover "
        "the exact curve",
        reference="exact E[o | f] = sigmoid(0.6 logit(f)) for this spec",
        measured=f"{excluded} of {curve.n_occupied} bins flagged; "
        f"{int(np.sum(inside))} of {curve.n_occupied} cover the exact curve",
        expectation=">= 7 of 12 bins flagged and >= 9 of 12 covering the exact curve",
        passed=excluded >= 7 and int(np.sum(inside)) >= 9,
    )
    return rec.finish()


if __name__ == "__main__":
    raise SystemExit(main())
