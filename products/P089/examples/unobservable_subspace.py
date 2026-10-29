"""The headline: what each estimator says about a degree of freedom the data cannot support.

A misalignment of 2 mrad is injected ENTIRELY along each record's own dominant
rotation axis. By equation (10) that is the direction the record cannot support
at any data volume. Every record is verified rank deficient by its own Gramian
before any estimator is run.

Left: what each estimator claims, with the one-sigma interval it reports.
Middle: how many of its own standard deviations each is wrong by. Right: record
length does not help, which is the whole point.
"""

from __future__ import annotations

import sys
from pathlib import Path

from _common import (
    ARCSEC_PER_RAD,
    BATCH,
    BOUND,
    LEARNED,
    RECURSIVE,
    TRIVIAL,
    bootstrap,
    save,
    style,
)

bootstrap()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "validation"))
style()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from aligncal import batch, features, observability, recursive  # noqa: E402

from _bootstrap import axis_aligned_record, fit_estimators  # noqa: E402

RECORDS = 60
SAMPLES = 150
INJECTED = 2.0e-3


def main() -> str:
    models = fit_estimators(count=1500, samples=SAMPLES, seed=20261029)
    figure, axes = plt.subplots(1, 3, figsize=(15.2, 5.0))
    figure.subplots_adjust(wspace=0.32)

    claims: dict[str, list[float]] = {}
    sigmas: dict[str, list[float]] = {}
    errors: dict[str, list[float]] = {}
    truths: list[float] = []
    deficient = 0

    for index in range(RECORDS):
        generated = axis_aligned_record(
            seed=55120261 + index, samples=SAMPLES, magnitude_rad=INJECTED
        )
        verdict = observability.assess(generated.record)
        sensor = verdict["secondary_1"]
        deficient += sensor.rank_deficient
        worst = sensor.degrees_of_freedom[-1].direction
        truth = float(np.dot(generated.truth["secondary_1"], worst))
        truths.append(truth)

        solution = batch.solve(generated.record, verdict, "secondary_1")
        mekf = recursive.run(generated.record, verdict, "secondary_1", store_history=False)
        vector = features.build(generated.record, "secondary_1", verdict)[None, :]
        entries = {
            "batch, eq (14)-(16)": (
                float(np.dot(solution.theta_rad, worst)),
                solution.sigma_rad_in_direction(worst),
            ),
            "recursive MEKF": (
                float(np.dot(mekf.theta_rad, worst)),
                mekf.sigma_rad_in_direction(worst),
            ),
        }
        for name in ("bagged_ridge", "forest", "trivial"):
            prediction = models[name].predict(vector)
            entries[f"learned {name}" if name != "trivial" else "trivial baseline"] = (
                float(prediction.component_in_direction(worst[None, :])[0]),
                float(prediction.sigma_in_direction(worst[None, :])[0]),
            )
        for label, (claim, sigma) in entries.items():
            claims.setdefault(label, []).append(claim)
            sigmas.setdefault(label, []).append(sigma)
            errors.setdefault(label, []).append(abs(claim - truth))

    order = [
        "batch, eq (14)-(16)",
        "recursive MEKF",
        "learned bagged_ridge",
        "learned forest",
        "trivial baseline",
    ]
    colours = [BATCH, RECURSIVE, LEARNED, "#7ab87a", TRIVIAL]

    left = axes[0]
    truth_arcsec = float(np.median(truths)) * ARCSEC_PER_RAD
    for position, (label, colour) in enumerate(zip(order, colours, strict=True)):
        claim = float(np.median(claims[label])) * ARCSEC_PER_RAD
        sigma = float(np.median(sigmas[label])) * ARCSEC_PER_RAD
        if not np.isfinite(sigma):
            left.scatter([position], [claim], s=70, marker="x", color=colour, zorder=3)
            left.annotate(
                "REFUSED\nbound is infinite",
                (position, claim),
                textcoords="offset points",
                xytext=(0, 22),
                ha="center",
                fontsize=8,
                color=colour,
            )
        else:
            left.errorbar(
                [position], [claim], yerr=[sigma], fmt="o", color=colour, capsize=5,
                markersize=6,
            )
    left.axhline(
        truth_arcsec, color=BOUND, linestyle="--", linewidth=1.4,
        label=f"injected truth, {truth_arcsec:.1f} arcsec",
    )
    left.axhline(0.0, color="#999999", linewidth=0.8)
    # Symmetric log: the recursive filter's interval is two orders of magnitude
    # wider than the learned ones, and on a linear axis it would make them
    # invisible. The linear region covers the injected truth.
    left.set_yscale("symlog", linthresh=600.0)
    left.set_yticks([-10000, -1000, -600, 0, 412.5, 600, 1000, 10000])
    left.set_yticklabels(["-10000", "-1000", "-600", "0", "412.5", "600", "1000", "10000"],
                         fontsize=7)
    left.set_xticks(range(len(order)))
    left.set_xticklabels([label.replace(", ", ",\n").replace(" ", "\n", 1) for label in order],
                         fontsize=7)
    left.set_ylabel("claimed component, arcsec")
    left.set_title(
        f"the claim and its own one sigma\n{deficient} of {RECORDS} records rank deficient"
    )
    left.legend(fontsize=8, loc="lower left")

    middle = axes[1]
    positions = []
    labels = []
    data = []
    for label in order:
        sigma_array = np.array(sigmas[label])
        error_array = np.array(errors[label])
        if not np.all(np.isfinite(sigma_array)):
            continue
        data.append(error_array / sigma_array)
        labels.append(label)
        positions.append(len(positions))
    parts = middle.boxplot(data, positions=positions, widths=0.6, patch_artist=True)
    for patch, label in zip(parts["boxes"], labels, strict=True):
        patch.set_facecolor(
            LEARNED if label.startswith("learned") else
            TRIVIAL if label.startswith("trivial") else RECURSIVE
        )
        patch.set_alpha(0.7)
    middle.axhline(2.0, color=BOUND, linestyle="--", linewidth=1.2, label="2 sigma")
    middle.set_yscale("log")
    middle.set_xticks(positions)
    middle.set_xticklabels([label.replace(" ", "\n", 1) for label in labels], fontsize=7)
    middle.set_ylabel("|error| / the estimator's own one sigma")
    middle.set_title(
        "how many of its own standard deviations\neach estimator is wrong by.\n"
        "The batch solution is absent: it claims nothing."
    )
    middle.legend(fontsize=8)

    right = axes[2]
    lengths = (150, 300, 600, 1200, 2400, 4800)
    curves: dict[str, list[float]] = {"recursive MEKF": [], "learned bagged_ridge": [],
                                      "learned sigma": [], "batch claim": []}
    for length in lengths:
        mekf_errors = []
        learned_errors = []
        learned_sigmas = []
        batch_claims = []
        for index in range(15):
            generated = axis_aligned_record(
                seed=55120261 + 30000 + index, samples=length, magnitude_rad=INJECTED
            )
            verdict = observability.assess(generated.record)
            sensor = verdict["secondary_1"]
            worst = sensor.degrees_of_freedom[-1].direction
            truth = float(np.dot(generated.truth["secondary_1"], worst))
            solution = batch.solve(generated.record, verdict, "secondary_1")
            batch_claims.append(abs(float(np.dot(solution.theta_rad, worst))))
            mekf = recursive.run(generated.record, verdict, "secondary_1", store_history=False)
            mekf_errors.append(abs(float(np.dot(mekf.theta_rad, worst)) - truth))
            prediction = models["bagged_ridge"].predict(
                features.build(generated.record, "secondary_1", verdict)[None, :]
            )
            learned_errors.append(
                abs(float(prediction.component_in_direction(worst[None, :])[0]) - truth)
            )
            learned_sigmas.append(float(prediction.sigma_in_direction(worst[None, :])[0]))
        curves["recursive MEKF"].append(float(np.median(mekf_errors)) * ARCSEC_PER_RAD)
        curves["learned bagged_ridge"].append(float(np.median(learned_errors)) * ARCSEC_PER_RAD)
        curves["learned sigma"].append(float(np.median(learned_sigmas)) * ARCSEC_PER_RAD)
        curves["batch claim"].append(max(float(np.max(batch_claims)) * ARCSEC_PER_RAD, 1.0e-12))
    right.loglog(lengths, curves["recursive MEKF"], color=RECURSIVE, marker="o", markersize=4,
                 label="recursive MEKF, |error|")
    right.loglog(lengths, curves["learned bagged_ridge"], color=LEARNED, marker="s", markersize=4,
                 label="learned bagged_ridge, |error|")
    right.loglog(lengths, curves["learned sigma"], color=LEARNED, linestyle="--", marker="^",
                 markersize=4, label="learned bagged_ridge, its own sigma")
    right.axhline(
        float(np.median(truths)) * ARCSEC_PER_RAD, color=BOUND, linestyle=":", linewidth=1.4,
        label="injected truth",
    )
    right.set_xticks(list(lengths))
    right.set_xticklabels([str(value) for value in lengths], fontsize=8)
    right.minorticks_off()
    right.set_xlabel("record length, samples")
    right.set_ylabel("arcsec")
    right.set_title(
        "a factor of 32 in record length.\nThe batch claim stays exactly zero and the\n"
        "others do not improve."
    )
    right.legend(fontsize=7, loc="lower left")

    figure.suptitle(
        "A 2 mrad misalignment injected along the record's own rotation axis. "
        "The analytic method refuses; the learned estimator answers, with a sharp interval.",
        fontsize=11,
    )
    figure.subplots_adjust(top=0.82)
    return save(figure, "unobservable_subspace")


if __name__ == "__main__":
    print(main())
