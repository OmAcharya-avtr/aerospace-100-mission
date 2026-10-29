"""The recursive estimator converging to the batch solution, per degree of freedom.

Left and middle: a rich record, where every degree of freedom clears the noise
floor and the filter converges onto the batch answer inside its own bound.
Right: a single-axis record, where the third degree of freedom is unsupported
and the filter does not converge at all -- it keeps the prior it started with
and says so through its covariance, while the batch solution withholds the
direction outright.
"""

from __future__ import annotations

from _common import (
    ARCSEC_PER_RAD,
    BATCH,
    RECURSIVE,
    VERDICT_COLOURS,
    bootstrap,
    save,
    style,
)

bootstrap()
style()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from aligncal import batch, dataset, observability, recursive  # noqa: E402


def _panel(axis, generated, dof_index, title):
    verdict = observability.assess(generated.record)
    sensor = verdict["secondary_1"]
    solution = batch.solve(generated.record, verdict, "secondary_1")
    mekf = recursive.run(generated.record, verdict, "secondary_1", store_history=True)
    dof = sensor.degrees_of_freedom[dof_index]
    direction = dof.direction

    history = mekf.history @ direction * ARCSEC_PER_RAD
    truth = float(np.dot(generated.truth["secondary_1"], direction)) * ARCSEC_PER_RAD
    reference = float(np.dot(solution.theta_rad, direction)) * ARCSEC_PER_RAD
    times = generated.record.times

    # Filter one sigma along this direction, per sample, from the stored diagonal.
    sigma = np.sqrt(np.sum((mekf.sigma_history * direction) ** 2, axis=1)) * ARCSEC_PER_RAD

    axis.plot(times, history, color=RECURSIVE, linewidth=1.0, label="recursive MEKF")
    axis.fill_between(
        times, history - sigma, history + sigma, color=RECURSIVE, alpha=0.18,
        label="filter one sigma",
    )
    axis.axhline(truth, color="#333333", linestyle=":", linewidth=1.2, label="injected truth")
    if dof.verdict == "unobservable":
        axis.axhline(
            0.0, color=BATCH, linewidth=1.6,
            label="batch: WITHHELD, exactly zero",
        )
    else:
        axis.axhline(reference, color=BATCH, linewidth=1.6, label="batch, eq (14)-(16)")
        axis.fill_between(
            times[[0, -1]],
            reference - dof.crb_sigma_arcsec,
            reference + dof.crb_sigma_arcsec,
            color=BATCH, alpha=0.15, label="Cramer-Rao one sigma",
        )
    # The first few updates move by radians from a 0.1 rad prior and would set the
    # scale of the whole panel. The limits are taken from the settled part of the
    # run, padded by the bound, so the figure shows convergence rather than the
    # transient.
    settled = history[int(0.2 * history.size) :]
    pad = (
        dof.crb_sigma_arcsec * 4.0
        if np.isfinite(dof.crb_sigma_rad)
        else float(np.ptp(settled)) * 0.5 + 1.0
    )
    low = min(float(np.min(settled)), truth, reference if np.isfinite(dof.crb_sigma_rad) else 0.0)
    high = max(float(np.max(settled)), truth, reference if np.isfinite(dof.crb_sigma_rad) else 0.0)
    axis.set_ylim(low - pad, high + pad)
    axis.set_xlabel("time, s")
    axis.set_title(
        f"{title}\ndof {dof_index}: {dof.verdict}"
        + (
            "   CRB unbounded"
            if not np.isfinite(dof.crb_sigma_rad)
            else f"   CRB {dof.crb_sigma_arcsec:.1f} arcsec"
        ),
        color=VERDICT_COLOURS[dof.verdict],
    )
    return mekf


def main() -> str:
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.8))
    figure.subplots_adjust(wspace=0.3)

    rich = dataset.generate(profile="three_axis", samples=2000, seed=20261029)
    _panel(axes[0], rich, 0, "rich excitation, best-observed dof")
    axes[0].set_ylabel("alignment component, arcsec")
    axes[0].legend(fontsize=7, loc="lower right")

    _panel(axes[1], rich, 2, "rich excitation, worst-observed dof")
    axes[1].legend(fontsize=7, loc="lower right")

    single = dataset.generate(profile="single_axis", samples=2000, seed=20261029)
    mekf = _panel(axes[2], single, 2, "single-axis record, unsupported dof")
    axes[2].legend(fontsize=7, loc="upper left")
    retention = mekf.prior_retention()
    axes[2].text(
        0.98,
        0.02,
        f"prior retained {retention[2]:.4f}\nworst/best sigma ratio {mekf.inflation_ratio():.1f}",
        transform=axes[2].transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )

    figure.suptitle(
        "Baseline 2 against Baseline 1. The filter has no refusal mechanism; its "
        "covariance is the only thing that says the record supports nothing here.",
        fontsize=11,
    )
    figure.subplots_adjust(top=0.84)
    return save(figure, "batch_versus_recursive")


if __name__ == "__main__":
    print(main())
