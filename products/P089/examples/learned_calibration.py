"""Does the learned uncertainty mean anything? Measured against the analytic bound.

Left: reported one sigma against realised error, per component, for each declared
learned configuration and for the analytic Cramer-Rao bound. A calibrated
estimator sits on the diagonal. Middle: coverage against the nominal curve.
Right: the same for the single direction the record cannot support, where the
analytic bound is infinite and the learned ones are not.
"""

from __future__ import annotations

import sys
from pathlib import Path

from _common import BATCH, BOUND, LEARNED, TRIVIAL, bootstrap, save, style

bootstrap()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "validation"))
style()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from aligncal import batch, benchmark, dataset, features, observability  # noqa: E402

from _bootstrap import axis_aligned_record, fit_estimators  # noqa: E402

TRAIN = 1500
TEST = 300
SAMPLES = 150

COLOURS = {
    "forest": LEARNED,
    "forest_deep": "#7ab87a",
    "bagged_ridge": "#2f6f3f",
    "trivial": TRIVIAL,
}


def main() -> str:
    models = fit_estimators(count=TRAIN, samples=SAMPLES, seed=20261029)
    test = dataset.training_population(count=TEST, samples=SAMPLES, seed=70120261)
    matrix, targets, _ = features.build_population(test)

    figure, axes = plt.subplots(1, 3, figsize=(15.2, 5.0))
    figure.subplots_adjust(wspace=0.32)

    left = axes[0]
    for name, colour in COLOURS.items():
        prediction = models[name].predict(matrix)
        errors = np.abs(prediction.theta_rad - targets).reshape(-1) * 1e3
        sigmas = prediction.sigma_rad.reshape(-1) * 1e3
        left.scatter(sigmas, errors, s=4, alpha=0.3, color=colour, label=name)
    analytic_errors = []
    analytic_sigmas = []
    for generated in test:
        verdict = observability.assess(generated.record)
        sensor = verdict["secondary_1"]
        solution = batch.solve(generated.record, verdict, "secondary_1")
        truth = generated.truth["secondary_1"]
        for dof in sensor.degrees_of_freedom:
            if dof.verdict == "unobservable":
                continue
            analytic_errors.append(
                abs(float(np.dot(solution.theta_rad - truth, dof.direction))) * 1e3
            )
            analytic_sigmas.append(dof.crb_sigma_rad * 1e3)
    left.scatter(analytic_sigmas, analytic_errors, s=4, alpha=0.3, color=BATCH,
                 label="batch, Cramer-Rao")
    span = np.array([1.0e-3, 2.0e1])
    left.plot(span, span, color=BOUND, linestyle="--", linewidth=1.2, label="|error| = sigma")
    left.set_xscale("log")
    left.set_yscale("log")
    left.set_xlabel("reported one sigma, mrad")
    left.set_ylabel("realised |error|, mrad")
    left.set_title("reported against realised, per component")
    left.legend(fontsize=7, loc="upper left", markerscale=2.0)

    middle = axes[1]
    factors = np.linspace(0.25, 3.5, 28)
    for name, colour in COLOURS.items():
        prediction = models[name].predict(matrix)
        errors = np.abs(prediction.theta_rad - targets).reshape(-1)
        sigmas = prediction.sigma_rad.reshape(-1)
        middle.plot(
            factors,
            [benchmark.coverage(errors, sigmas, float(k)) for k in factors],
            color=colour,
            label=name,
        )
    middle.plot(
        factors,
        [benchmark.coverage(np.array(analytic_errors), np.array(analytic_sigmas), float(k))
         for k in factors],
        color=BATCH,
        label="batch, Cramer-Rao",
        linewidth=1.8,
    )
    from scipy.special import erf

    middle.plot(factors, erf(factors / np.sqrt(2.0)), color=BOUND, linestyle="--",
                linewidth=1.4, label="nominal, standard normal")
    middle.set_xlabel("interval width, in reported sigmas")
    middle.set_ylabel("fraction of components covered")
    middle.set_title(
        "coverage. The analytic bound is on the\nnominal curve; the learned ones are not."
    )
    middle.legend(fontsize=7, loc="lower right")

    right = axes[2]
    records = []
    for index in range(80):
        generated = axis_aligned_record(seed=55120261 + index, samples=SAMPLES,
                                        magnitude_rad=2.0e-3)
        verdict = observability.assess(generated.record)
        records.append((generated, verdict, verdict["secondary_1"]))
    for name, colour in COLOURS.items():
        errors = []
        sigmas = []
        for generated, verdict, sensor in records:
            worst = sensor.degrees_of_freedom[-1].direction[None, :]
            prediction = models[name].predict(
                features.build(generated.record, "secondary_1", verdict)[None, :]
            )
            truth = float(np.dot(generated.truth["secondary_1"], worst[0]))
            errors.append(abs(float(prediction.component_in_direction(worst)[0]) - truth))
            sigmas.append(float(prediction.sigma_in_direction(worst)[0]))
        right.plot(
            factors,
            [benchmark.coverage(np.array(errors), np.array(sigmas), float(k)) for k in factors],
            color=colour,
            label=name,
        )
    right.plot(factors, erf(factors / np.sqrt(2.0)), color=BOUND, linestyle="--",
               linewidth=1.4, label="nominal, standard normal")
    right.axhline(0.0, color=BATCH, linewidth=2.0,
                  label="batch: refuses, no interval at all")
    right.set_xlabel("interval width, in reported sigmas")
    right.set_ylabel("fraction of records covered")
    right.set_ylim(-0.05, 1.05)
    right.set_title(
        "the SAME curve for the one direction the\nrecord cannot support"
    )
    right.legend(fontsize=7, loc="upper left")

    figure.suptitle(
        "The learned uncertainty is an ensemble spread, not a calibrated bound. "
        "The Cramer-Rao bound is a bound, and it is infinite where it has to be.",
        fontsize=11,
    )
    figure.subplots_adjust(top=0.84)
    return save(figure, "learned_calibration")


if __name__ == "__main__":
    print(main())
