"""Measured coverage of the reported offset uncertainty, four treatments.

Writes ``screenshots/uncertainty_coverage.png``. What to notice: the treatment
with no asymmetry term covers almost nothing, the declared bound over-covers,
and the two corrector treatments sit on the diagonal. The left panel is
coverage against the nominal implied by each coverage factor; a point on the
diagonal is a calibrated interval.
"""

from __future__ import annotations

import numpy as np
from _common import SERIES, bootstrap, save, style

ROOT = bootstrap()
style()

import matplotlib.pyplot as plt  # noqa: E402

from timetransfer import (  # noqa: E402
    AsymmetrySource,
    fit_least_squares,
    measure_coverage,
    offset_uncertainty,
    two_way_estimate,
)
from timetransfer.asym import (  # noqa: E402
    GeneratorConfig,
    LearnedAsymmetryCorrector,
    MedianFilterEstimator,
    build_dataset,
    generate_scenario,
    window_features,
)

BASE_SEED = 70079
CONFIG = GeneratorConfig(window=150, coupling=1.0, independent_sigma_s=2.0e-4)
N_TRAIN = 500
N_CALIB = 150
N_COVER = 320
FACTORS = np.asarray([0.5, 0.8, 1.0, 1.282, 1.645, 1.96, 2.326, 2.576, 3.0])


def _scenario(index: int):
    exchanges, truth = generate_scenario(index, seed=BASE_SEED, config=CONFIG)
    estimate = two_way_estimate(exchanges)
    fit = fit_least_squares(estimate.midpoint_s, estimate.offset_s, order=1)
    offset, _ = fit.offset_at(fit.t_ref_s)
    assert exchanges.true_offset_s is not None
    true_at_epoch = float(
        np.interp(fit.t_ref_s, estimate.midpoint_s, exchanges.true_offset_s)
    )
    return fit, float(offset), true_at_epoch, float(truth["bias_s"]), window_features(exchanges)


def main() -> None:
    train = build_dataset(N_TRAIN, seed=BASE_SEED, config=CONFIG)
    median = MedianFilterEstimator().fit(train.features, train.target_s)
    learned = LearnedAsymmetryCorrector().fit(train.features, train.target_s)

    calib_x = []
    calib_y = []
    for i in range(N_TRAIN, N_TRAIN + N_CALIB):
        _, _, _, bias, features = _scenario(i)
        calib_x.append(features)
        calib_y.append(bias)
    calib_x = np.asarray(calib_x)
    calib_y = np.asarray(calib_y)
    u_median = float(np.std(calib_y - median.predict(calib_x), ddof=1))
    u_learned = float(np.std(calib_y - learned.predict(calib_x), ddof=1))
    bound = 6.0 * float(CONFIG.independent_sigma_s)

    errors: dict[str, list[float]] = {k: [] for k in
                                      ("unaccounted", "declared bound", "median filter",
                                       "learned GBR")}
    uncs: dict[str, list[float]] = {k: [] for k in errors}
    start = N_TRAIN + N_CALIB
    for i in range(start, start + N_COVER):
        fit, offset, truth, _, features = _scenario(i)
        row = features.reshape(1, -1)
        plain = offset_uncertainty(fit, resolution_ns=1)
        errors["unaccounted"].append(offset - truth)
        uncs["unaccounted"].append(plain.u_combined_s)
        bounded = offset_uncertainty(
            fit,
            resolution_ns=1,
            asymmetry_source=AsymmetrySource.DECLARED_BOUND,
            asymmetry_bound_s=bound,
        )
        errors["declared bound"].append(offset - truth)
        uncs["declared bound"].append(bounded.u_combined_s)
        for key, estimator, u_asym in (
            ("median filter", median, u_median),
            ("learned GBR", learned, u_learned),
        ):
            corrected = offset_uncertainty(
                fit,
                resolution_ns=1,
                asymmetry_source=AsymmetrySource.ESTIMATED,
                asymmetry_stderr_s=u_asym,
            )
            errors[key].append(offset - float(estimator.predict(row)[0]) - truth)
            uncs[key].append(corrected.u_combined_s)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    left, right = axes
    print(f"{'treatment':>16} {'measured at k=1.96':>20} {'nominal':>9} {'verdict':>14}")
    for index, key in enumerate(errors):
        err = np.asarray(errors[key])
        unc = np.asarray(uncs[key])
        nominal = []
        measured = []
        for k in FACTORS:
            result = measure_coverage(err, unc, coverage_factor=float(k))
            nominal.append(result.nominal)
            measured.append(result.measured)
        left.plot(nominal, measured, "o-", color=SERIES[index], ms=4, lw=1.1, label=key)
        headline = measure_coverage(err, unc, coverage_factor=1.96)
        print(f"{key:>16} {headline.measured:>20.4f} {headline.nominal:>9.4f} "
              f"{headline.verdict:>14}")
    left.plot([0, 1], [0, 1], "--", color="#808080", lw=1.0, label="calibrated")
    left.set_xlabel("nominal coverage implied by k")
    left.set_ylabel("measured coverage")
    left.set_title(f"coverage over {N_COVER} held-out scenarios")
    left.legend(loc="upper left", fontsize=7.6)
    left.set_xlim(0.3, 1.02)
    left.set_ylim(0.0, 1.02)

    width = 0.2
    positions = np.arange(len(errors))
    for index, key in enumerate(errors):
        err = np.asarray(errors[key])
        unc = np.asarray(uncs[key])
        right.bar(positions[index] - width / 2, float(np.sqrt(np.mean(err * err))) * 1e6,
                  width, color=SERIES[index], label="rms error" if index == 0 else None)
        right.bar(positions[index] + width / 2, float(np.mean(unc)) * 1e6, width,
                  color=SERIES[index], alpha=0.45,
                  label="mean reported u" if index == 0 else None)
    right.set_xticks(positions)
    right.set_xticklabels(list(errors), rotation=20, ha="right", fontsize=7.6)
    right.set_ylabel("microseconds")
    right.set_title("rms offset error against the reported uncertainty")
    right.legend(loc="upper left", fontsize=7.6)

    fig.tight_layout()
    print(save(fig, "uncertainty_coverage"))
    print(f"u_asymmetry, median filter : {u_median:.6e} s (calibration split)")
    print(f"u_asymmetry, learned GBR   : {u_learned:.6e} s (calibration split)")
    print(f"declared bound B           : {bound:.6e} s")


if __name__ == "__main__":
    main()
