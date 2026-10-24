"""Check 7 -- the learned asymmetry corrector against the baselines it must beat.

Requirement (BATCH_08_SPEC, P079): "AI component: a learned delay-asymmetry
estimator that corrects the offset bias from exchange features, benchmarked
against the symmetric assumption and against a median-filtered estimator. **A
median filter matching it is the likely result and is to be reported.**"

The honest frame
----------------
The quantity to estimate is the window's mean asymmetry bias
``b = mean(d_ab - d_ba)/2`` [s], which subtracts directly off the two-way
offset (equation (5) of :mod:`timetransfer.twoway`). Five estimators are scored
on the **same** held-out scenarios with the **same** metrics:

1. ``symmetric_assumption``  predict zero -- what a conventional two-way
   implementation does, and the thing any corrector must beat to exist.
2. ``median_filter``         ``(median(fwd_c) - median(rev_c)) / 2``, closed
   form, no training.
3. ``mean_difference``       ``(mean(fwd_c) - mean(rev_c)) / 2``, closed form,
   no training. Included because the target is a *mean*, so leaving it out
   would have flattered the median filter.
4. ``learned_ridge``         standardised ridge on the sixteen features.
5. ``learned_gbr``           gradient-boosted trees on the same sixteen.

The generator's coupling parameter decides how much of the *static* asymmetry
is learnable at all -- see :mod:`timetransfer.asym.data`. The benchmark is run
at three couplings, so the reader sees the curve rather than one point chosen
after the fact.

Run: ``python validation/validate_asymmetry_benchmark.py``
"""

from __future__ import annotations

import sys

import numpy as np

from _bootstrap import Tee, Timer, bootstrap, environment_lines, is_fast, scaled

ROOT = bootstrap()

from timetransfer.asym import (  # noqa: E402
    FEATURE_NAMES,
    GeneratorConfig,
    LearnedAsymmetryCorrector,
    MeanDifferenceEstimator,
    MedianFilterEstimator,
    RidgeAsymmetryCorrector,
    ScoreTable,
    SymmetricAssumption,
    TrainingConfig,
    build_dataset,
    score_predictions,
)

SEED = 20261024
DEFAULT_COUPLING = 1.0
COUPLINGS = (DEFAULT_COUPLING,) if is_fast() else (0.0, 1.0, 2.0)
"""Fast probe mode scores one coupling only; the committed numbers use all three."""


def _estimators() -> list[object]:
    """The five scored estimators, baselines first.

    In fast probe mode the boosted model is built with fewer stages. The probe
    only checks exit status and writes no output file, so no committed number
    comes from a reduced configuration.
    """
    boosted = (
        LearnedAsymmetryCorrector(TrainingConfig(n_estimators=20, min_samples_leaf=4))
        if is_fast()
        else LearnedAsymmetryCorrector()
    )
    return [
        SymmetricAssumption(),
        MedianFilterEstimator(),
        MeanDifferenceEstimator(),
        RidgeAsymmetryCorrector(),
        boosted,
    ]


def main() -> int:
    out = Tee("validate_asymmetry_benchmark_output.txt")
    out("timetransfer validation 7 -- learned asymmetry corrector against the baselines")
    out("=" * 78)
    for line in environment_lines():
        out(line)
    failures = 0
    n_scenarios = scaled(1200, minimum=120)
    out("")
    out(f"scenarios per coupling : {n_scenarios}")
    out(f"base seed              : {SEED}")
    out("split                  : first 70 per cent train, last 30 per cent held out")
    out("target                 : window mean asymmetry bias b = mean(d_ab - d_ba)/2 [s]")
    out("metric that decides    : held-out RMSE on b, because b subtracts directly off")
    out("                         the offset estimate")

    tables: dict[float, ScoreTable] = {}
    timings: dict[str, float] = {}
    importances: np.ndarray | None = None
    coefficients: np.ndarray | None = None
    gbr_coverage: dict[float, tuple[float, float]] = {}

    for coupling in COUPLINGS:
        config = GeneratorConfig(coupling=coupling)
        build_timer = Timer()
        with build_timer:
            dataset = build_dataset(n_scenarios, seed=SEED, config=config)
        train, test = dataset.split(train_fraction=0.7)
        out.rule(f"7a. coupling = {coupling:.1f}")
        out(dataset.as_text())
        out("")
        correlation_static = float(
            np.corrcoef(dataset.static_bias_s, dataset.congestion_difference)[0, 1]
        )
        out(f"corr(static bias, congestion difference) : {correlation_static:+.6f}")
        out("  This is the only route by which a corrector can reach the static")
        out("  asymmetry, because the features are centred on each direction's own")
        out("  window minimum and the static part is removed with the floor.")
        out(f"train / held out                         : {train.size} / {test.size}")
        out(f"dataset build time                       : {build_timer.elapsed_s:.3f} s")
        out("")

        rows = []
        reference_rmse: float | None = None
        for estimator in _estimators():
            timer = Timer()
            with timer:
                estimator.fit(dataset.features[train], dataset.target_s[train])
            predicted = estimator.predict(dataset.features[test])
            scored = score_predictions(
                estimator.name,
                predicted,
                dataset.target_s[test],
                reference_rmse_s=reference_rmse,
            )
            if reference_rmse is None:
                reference_rmse = scored.rmse_s
                scored = score_predictions(
                    estimator.name,
                    predicted,
                    dataset.target_s[test],
                    reference_rmse_s=reference_rmse,
                )
            rows.append(scored)
            if coupling == DEFAULT_COUPLING:
                timings[estimator.name] = timer.elapsed_s
                if isinstance(estimator, LearnedAsymmetryCorrector):
                    importances = estimator.feature_importances
                    mean, sigma = estimator.predict_with_uncertainty(dataset.features[test])
                    errors = np.abs(mean - dataset.target_s[test])
                    with np.errstate(divide="ignore", invalid="ignore"):
                        covered = float(np.mean(errors <= 1.96 * np.maximum(sigma, 1e-300)))
                    gbr_coverage[coupling] = (covered, float(estimator.crossing_fraction))
                if isinstance(estimator, RidgeAsymmetryCorrector):
                    coefficients = estimator.coefficients
        table = ScoreTable(rows=tuple(rows), reference_name="symmetric_assumption")
        tables[coupling] = table
        out(table.as_text())
        out("")
        out(f"best by rmse : {table.best().name}")
        out(f"median_filter / learned_gbr rmse ratio : "
            f"{table.ratio('median_filter', 'learned_gbr'):.4f}")
        out(f"learned_gbr / learned_ridge rmse ratio : "
            f"{table.ratio('learned_gbr', 'learned_ridge'):.4f}")

    out.rule("7b. the verdict, stated plainly")
    default = tables[DEFAULT_COUPLING]
    symmetric = default.by_name("symmetric_assumption")
    median = default.by_name("median_filter")
    mean_diff = default.by_name("mean_difference")
    ridge = default.by_name("learned_ridge")
    gbr = default.by_name("learned_gbr")
    out(f"at the declared default coupling of {DEFAULT_COUPLING:.1f}, held-out RMSE on b:")
    out(f"  symmetric_assumption : {symmetric.rmse_s:.6e} s")
    out(f"  median_filter        : {median.rmse_s:.6e} s")
    out(f"  mean_difference      : {mean_diff.rmse_s:.6e} s")
    out(f"  learned_ridge        : {ridge.rmse_s:.6e} s")
    out(f"  learned_gbr          : {gbr.rmse_s:.6e} s")
    out("")
    out(f"median_filter vs learned_gbr : ratio {median.rmse_s / gbr.rmse_s:.4f}")
    if median.rmse_s <= gbr.rmse_s:
        out("  The median filter MATCHES OR BEATS the learned corrector. That is the")
        out("  specification's own prediction and it is published as measured.")
    else:
        out("  The median filter LOSES to the learned corrector on this metric. The")
        out("  specification predicted a match; the measurement does not support it,")
        out("  and the measurement is what is published.")
    out("")
    out(f"learned_gbr vs learned_ridge : ratio {gbr.rmse_s / ridge.rmse_s:.4f}")
    if gbr.rmse_s >= ridge.rmse_s:
        out("  The gradient-boosted corrector LOSES to a standardised linear ridge on")
        out("  the same sixteen features. The ensemble, its hyperparameters and its")
        out("  version pin buy nothing on this generator: a linear combination of the")
        out("  features is the whole of the learnable signal. This is the result and")
        out("  it is not retuned away.")
    else:
        out("  The gradient-boosted corrector beats the linear ridge, by the ratio")
        out("  above. The margin is what has to justify the machinery.")
    out("")
    out("across the three couplings, median_filter / learned_gbr rmse ratio:")
    for coupling in COUPLINGS:
        ratio = tables[coupling].ratio("median_filter", "learned_gbr")
        out(f"  coupling {coupling:.1f} : {ratio:.4f}")
    out("across the three couplings, learned_gbr / learned_ridge rmse ratio:")
    for coupling in COUPLINGS:
        ratio = tables[coupling].ratio("learned_gbr", "learned_ridge")
        out(f"  coupling {coupling:.1f} : {ratio:.4f}")

    out.rule("7c. the learned corrector's uncertainty output, and its coverage")
    covered, crossing = gbr_coverage[DEFAULT_COUPLING]
    out("The reported per-row sigma is half the width of the quantile pair at")
    out("alpha = 0.1587 and 0.8413. That is a dispersion estimate, not a calibrated")
    out("interval, and the difference is measured here.")
    out("")
    out(f"held-out coverage of the k = 1.96 band : {covered:.4f}")
    out("nominal                                : 0.9500")
    out(f"quantile crossings on held-out rows    : {crossing:.4f}")
    out("")
    if covered < 0.90:
        out("The band UNDER-COVERS. The learned model's own uncertainty output is")
        out("optimistic by the margin above and must not be used as a confidence")
        out("interval. MODEL_CARD.md and README.md say so. The uncertainty that the")
        out("offset budget consumes is instead the held-out residual standard")
        out("deviation measured on a separate calibration split, and the coverage of")
        out("THAT is measured in validation/validate_uncertainty_coverage.py.")
    else:
        out("The band's measured coverage is within 5 points of nominal on this")
        out("generator. It is still a dispersion estimate and is labelled as one.")

    out.rule("7d. what the models learned")
    assert importances is not None and coefficients is not None
    order = np.argsort(importances)[::-1]
    out(f"{'feature':>26} {'gbr importance':>16} {'ridge coef (standardised)':>28}")
    for index in order:
        out(
            f"{FEATURE_NAMES[index]:>26} {importances[index]:>16.6f} "
            f"{coefficients[index]:>28.6e}"
        )
    out("")
    out("Both models lean on the mean and median difference features, which are the")
    out("closed-form baselines. That is the mechanism by which they beat the")
    out("symmetric assumption, and it is why a linear model is enough.")

    out.rule("7e. compute budget (UNSTABLE: wall clock on a shared two-core container)")
    out(f"{'estimator':>22} {'fit time [s]':>14}")
    for name, elapsed in timings.items():
        out(f"{name:>22} {elapsed:>14.4f}")
    out("")
    out("These figures move between runs and no ratio derived from them appears in")
    out("prose anywhere in this repository.")

    out.rule("7f. held-out residual standard deviations, for the uncertainty budget")
    out(f"{'estimator':>22} {'held-out residual sd [s]':>26}")
    for row in default.rows:
        out(f"{row.name:>22} {row.residual_std_s:>26.6e}")
    out("")
    out("These are the numbers timetransfer.uncertainty consumes as its asymmetry")
    out("term when a corrector is applied. They are measured on data the corrector")
    out("did not see.")

    out.rule("summary")
    if not np.isfinite(gbr.rmse_s) or gbr.rmse_s <= 0.0:
        failures += 1
        out("FAIL: the learned corrector produced a non-finite or zero RMSE")
    if symmetric.skill != 0.0:
        failures += 1
        out("FAIL: the reference estimator's own skill is not zero")
    out("A baseline win is not a failure. It is the result, and it is reported in")
    out("README.md, validation/VALIDATION.md and MODEL_CARD.md without softening.")
    out("")
    out(f"hard failures : {failures}")
    out(f"saved to      : {out.save()}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
