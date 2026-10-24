"""Check 3 -- the coverage of the reported offset uncertainty, measured.

Requirement (BATCH_08_SPEC, P079): "coverage of the reported offset uncertainty
measured". This is the check the product exists for. An interval whose coverage
has not been measured is a decoration, and the measurement is published here
whether or not it reaches the nominal.

The four treatments
-------------------
The asymmetry term is the only part of the budget that is not purely
statistical, and there are exactly four things a caller can do about it. All
four are measured on the **same** scenarios so the comparison is like for like:

* ``unaccounted``      no asymmetry term at all. Expected to under-cover badly.
* ``declared_bound``   a bound ``B`` on ``|A|`` from external knowledge, giving
                       ``u = (B/2)/sqrt(3)``. Expected to over-cover when the
                       bound is honest.
* ``median_filter``    the closed-form median-filtered corrector applied, with
                       its residual spread **measured on a calibration split**
                       the corrector was neither trained on nor scored on.
* ``learned_gbr``      the gradient-boosted corrector, same treatment.

Three disjoint splits
---------------------
Scenarios 0 to 59 per cent train the correctors; 60 to 79 per cent measure
their residual spread, which becomes ``u_asymmetry``; 80 to 100 per cent are
the coverage trials. The spread that goes into the uncertainty is therefore
measured on data the corrector did not see *and* is not the data the coverage
is computed on. Reusing one split for both would make the interval look better
than it is.

Run: ``python validation/validate_uncertainty_coverage.py``
"""

from __future__ import annotations

import sys

import numpy as np

from _bootstrap import Tee, bootstrap, environment_lines, is_fast, scaled

ROOT = bootstrap()

from timetransfer.asym import (  # noqa: E402
    GeneratorConfig,
    LearnedAsymmetryCorrector,
    MedianFilterEstimator,
    TrainingConfig,
    build_dataset,
    generate_scenario,
    window_features,
)
from timetransfer.skew import fit_least_squares  # noqa: E402
from timetransfer.twoway import two_way_estimate  # noqa: E402
from timetransfer.uncertainty import (  # noqa: E402
    AsymmetrySource,
    measure_coverage,
    offset_uncertainty,
)

BASE_SEED = 70079
CONFIG = GeneratorConfig(window=150, coupling=1.0, independent_sigma_s=2.0e-4)
COVERAGE_FACTORS = (1.0, 1.645, 1.96, 2.576)



def _scenario_estimate(index: int) -> tuple[float, float, float, np.ndarray, int, float]:
    """One scenario: fitted offset, truth at the epoch, true bias, features, n, resolution."""
    exchanges, truth = generate_scenario(index, seed=BASE_SEED, config=CONFIG)
    estimate = two_way_estimate(exchanges)
    fit = fit_least_squares(estimate.midpoint_s, estimate.offset_s, order=1)
    offset, _ = fit.offset_at(fit.t_ref_s)
    assert exchanges.true_offset_s is not None
    true_at_epoch = float(
        np.interp(fit.t_ref_s, estimate.midpoint_s, exchanges.true_offset_s)
    )
    return (
        float(offset),
        true_at_epoch,
        float(truth["bias_s"]),
        window_features(exchanges),
        exchanges.n,
        float(CONFIG.resolution_ns),
    )


def _fit_with(index: int):
    """Re-fit a scenario and return the fit object, for the uncertainty budget."""
    exchanges, _ = generate_scenario(index, seed=BASE_SEED, config=CONFIG)
    estimate = two_way_estimate(exchanges)
    return fit_least_squares(estimate.midpoint_s, estimate.offset_s, order=1)


def main() -> int:
    out = Tee("validate_uncertainty_coverage_output.txt")
    out("timetransfer validation 3 -- measured coverage of the offset uncertainty")
    out("=" * 78)
    for line in environment_lines():
        out(line)
    failures = 0

    n_train = scaled(600, minimum=60)
    n_calib = scaled(200, minimum=30)
    n_cover = scaled(400, minimum=40)
    out("")
    out(f"base seed                 : {BASE_SEED}")
    out(f"generator window          : {CONFIG.window} attempted exchanges per scenario")
    out(f"generator coupling        : {CONFIG.coupling:.3f}")
    out(f"independent asymmetry sd  : {CONFIG.independent_sigma_s:.3e} s")
    out(f"train / calibrate / cover : {n_train} / {n_calib} / {n_cover} scenarios (disjoint)")

    out.rule("3a. train the correctors on the training split")
    train = build_dataset(n_train, seed=BASE_SEED, config=CONFIG)
    out(train.as_text())
    median = MedianFilterEstimator().fit(train.features, train.target_s)
    training = (
        TrainingConfig(n_estimators=20, min_samples_leaf=4)
        if is_fast()
        else TrainingConfig()
    )
    learned = LearnedAsymmetryCorrector(training).fit(train.features, train.target_s)
    out("")
    out(f"median filter in-sample residual sd : {median.residual_std_s:.6e} s")
    out(f"learned gbr  in-sample residual sd  : {learned.residual_std_s:.6e} s")
    out("Neither of those numbers is used as an uncertainty: both are in-sample.")

    out.rule("3b. measure the residual spread on the calibration split")
    calib_features = []
    calib_target = []
    for i in range(n_train, n_train + n_calib):
        _, _, bias, features, _, _ = _scenario_estimate(i)
        calib_features.append(features)
        calib_target.append(bias)
    calib_x = np.asarray(calib_features)
    calib_y = np.asarray(calib_target)
    u_median = float(np.std(calib_y - median.predict(calib_x), ddof=1))
    u_learned = float(np.std(calib_y - learned.predict(calib_x), ddof=1))
    declared_bound_s = 6.0 * float(CONFIG.independent_sigma_s)
    out(f"calibration scenarios        : {calib_y.size}")
    out(f"true bias sd on calibration  : {float(np.std(calib_y, ddof=1)):.6e} s")
    out(f"median filter residual sd    : {u_median:.6e} s  -> u_asymmetry")
    out(f"learned gbr residual sd      : {u_learned:.6e} s  -> u_asymmetry")
    out(f"declared bound B on |A|      : {declared_bound_s:.6e} s")
    out(f"  -> u_asymmetry = (B/2)/sqrt(3) = {(declared_bound_s / 2.0) / np.sqrt(3.0):.6e} s")

    out.rule("3c. the coverage trials")
    errors: dict[str, list[float]] = {
        "unaccounted": [],
        "declared_bound": [],
        "median_filter": [],
        "learned_gbr": [],
    }
    uncertainties: dict[str, list[float]] = {key: [] for key in errors}
    true_bias: list[float] = []
    predicted_bias: dict[str, list[float]] = {"median_filter": [], "learned_gbr": []}

    start = n_train + n_calib
    for i in range(start, start + n_cover):
        offset, truth, bias, features, _, resolution = _scenario_estimate(i)
        fit = _fit_with(i)
        true_bias.append(bias)
        row = features.reshape(1, -1)
        bias_median = float(median.predict(row)[0])
        bias_learned = float(learned.predict(row)[0])
        predicted_bias["median_filter"].append(bias_median)
        predicted_bias["learned_gbr"].append(bias_learned)

        plain = offset_uncertainty(fit, resolution_ns=int(resolution))
        errors["unaccounted"].append(offset - truth)
        uncertainties["unaccounted"].append(plain.u_combined_s)

        bounded = offset_uncertainty(
            fit,
            resolution_ns=int(resolution),
            asymmetry_source=AsymmetrySource.DECLARED_BOUND,
            asymmetry_bound_s=declared_bound_s,
        )
        errors["declared_bound"].append(offset - truth)
        uncertainties["declared_bound"].append(bounded.u_combined_s)

        for key, bias_hat, u_asym in (
            ("median_filter", bias_median, u_median),
            ("learned_gbr", bias_learned, u_learned),
        ):
            corrected = offset_uncertainty(
                fit,
                resolution_ns=int(resolution),
                asymmetry_source=AsymmetrySource.ESTIMATED,
                asymmetry_stderr_s=u_asym,
            )
            errors[key].append(offset - bias_hat - truth)
            uncertainties[key].append(corrected.u_combined_s)

    out(f"coverage scenarios : {n_cover}")
    out(f"true bias sd       : {float(np.std(np.asarray(true_bias), ddof=1)):.6e} s")
    out("")
    out("mean absolute offset error by treatment (the thing the interval has to cover):")
    out(f"{'treatment':>16} {'mean |error| [s]':>18} {'rms error [s]':>16} "
        f"{'mean u [s]':>14}")
    for key in ("unaccounted", "declared_bound", "median_filter", "learned_gbr"):
        err = np.asarray(errors[key])
        out(
            f"{key:>16} {float(np.mean(np.abs(err))):>18.6e} "
            f"{float(np.sqrt(np.mean(err * err))):>16.6e} "
            f"{float(np.mean(np.asarray(uncertainties[key]))):>14.6e}"
        )
    out("")
    out("Note that 'unaccounted' and 'declared_bound' have the SAME errors by")
    out("construction -- a declared bound changes the uncertainty, not the estimate.")

    out.rule("3d. measured coverage against nominal")
    out(f"{'treatment':>16} {'k':>6} {'nominal':>9} {'measured':>10} {'covered/trials':>16} "
        f"{'CP 95% interval':>24} {'med|e|/u':>10} {'verdict':>14}")
    verdicts: dict[str, str] = {}
    headline: dict[str, tuple[float, float, str]] = {}
    for key in ("unaccounted", "declared_bound", "median_filter", "learned_gbr"):
        err = np.asarray(errors[key])
        unc = np.asarray(uncertainties[key])
        for k in COVERAGE_FACTORS:
            result = measure_coverage(err, unc, coverage_factor=k)
            out(
                f"{key:>16} {k:>6.3f} {result.nominal:>9.4f} {result.measured:>10.4f} "
                f"{f'{result.n_covered}/{result.n_trials}':>16} "
                f"{f'[{result.lower:.4f}, {result.upper:.4f}]':>24} "
                f"{result.median_normalised_error:>10.4f} {result.verdict:>14}"
            )
            if abs(k - 1.96) < 1e-9:
                verdicts[key] = result.verdict
                headline[key] = (result.measured, result.nominal, result.verdict)
        out("")

    out.rule("3e. what the numbers mean, stated rather than left to the reader")
    unacc = headline["unaccounted"]
    bound = headline["declared_bound"]
    med = headline["median_filter"]
    gbr = headline["learned_gbr"]
    out(f"unaccounted    : measured {unacc[0]:.4f} against nominal {unacc[1]:.4f} "
        f"-> {unacc[2]}")
    out("  An interval with no asymmetry term is a repeatability statement. It")
    out("  under-covers by the margin above, and that margin IS the asymmetry.")
    out(f"declared_bound : measured {bound[0]:.4f} against nominal {bound[1]:.4f} "
        f"-> {bound[2]}")
    out("  A type-B bound at six times the generator's own asymmetry sd is honest and")
    out("  conservative. It is the right answer when external knowledge exists.")
    out(f"median_filter  : measured {med[0]:.4f} against nominal {med[1]:.4f} "
        f"-> {med[2]}")
    out(f"learned_gbr    : measured {gbr[0]:.4f} against nominal {gbr[1]:.4f} "
        f"-> {gbr[2]}")
    out("  Both corrected treatments use a residual spread measured on a split the")
    out("  corrector never saw. Where the verdict is not 'calibrated', the interval")
    out("  is wrong by the stated margin and README.md says so.")
    out("")
    out("HEADLINE for README.md and VALIDATION.md:")
    out(f"  k = 1.96, nominal {gbr[1]:.4f}")
    for key in ("unaccounted", "declared_bound", "median_filter", "learned_gbr"):
        measured, nominal, verdict = headline[key]
        out(f"    {key:<15} measured {measured:.4f}  ({verdict})")

    out.rule("summary")
    if not np.isfinite(sum(uncertainties["learned_gbr"])):
        failures += 1
        out("FAIL: a reported uncertainty was not finite")
    out("No coverage result is treated as a hard failure: this script measures the")
    out("calibration of the interval and the measurement is the deliverable. A")
    out("verdict other than 'calibrated' is reported in README.md, VALIDATION.md and")
    out("MODEL_CARD.md and is not retuned away.")
    out("")
    out(f"hard failures : {failures}")
    out(f"saved to      : {out.save()}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
