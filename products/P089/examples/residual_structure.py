"""The residual of equation (7) and what the estimator takes out of it.

Left: the raw residual between the de-rotated secondary and the reference, with
the fitted model on top. Middle: the residual before and after the fit, which is
what the estimator actually removes. Right: the component of the residual along
the record's own rotation axis, which carries no information about the alignment
about that axis and is what the Gramian's null space is made of.
"""

from __future__ import annotations

from _common import (
    BATCH,
    BOUND,
    UNOBSERVABLE,
    bootstrap,
    save,
    style,
)

bootstrap()
style()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from aligncal import batch, dataset, observability  # noqa: E402
from aligncal.record import residuals  # noqa: E402


def main() -> str:
    figure, axes = plt.subplots(1, 3, figsize=(15.2, 4.6))
    figure.subplots_adjust(wspace=0.3)

    # A deliberately quiet record: 1e-5 rad/s noise against a misalignment of up to
    # 3 mrad on a 0.05 rad/s rotation, so the structure of equation (7) is visible
    # rather than buried. The noise level is a tenth of this repository's default and
    # is stated here because the figure is explanatory, not a measurement.
    generated = dataset.generate(profile="single_axis", samples=1200, noise_sigma=1.0e-5,
                                 theta_max_rad=3.0e-3, seed=9091)
    record = generated.record
    verdict = observability.assess(record)
    sensor = verdict["secondary_1"]
    solution = batch.solve(record, verdict, "secondary_1")
    measured = residuals(record, "secondary_1")
    fitted = batch.predicted_residual(record, solution.theta_rad)
    window = slice(0, 400)

    left = axes[0]
    for axis in range(3):
        left.plot(
            record.times[window],
            measured[window, axis] * 1.0e6,
            linewidth=0.6,
            alpha=0.6,
            color=["#1f6fb4", "#d9850b", "#3f8f4f"][axis],
        )
        left.plot(
            record.times[window],
            fitted[window, axis] * 1.0e6,
            linewidth=1.4,
            color=["#1f6fb4", "#d9850b", "#3f8f4f"][axis],
            label=f"fitted, body {'xyz'[axis]}",
        )
    left.set_xlabel("time, s")
    left.set_ylabel("residual, micro-rad/s")
    left.set_title(
        "equation (7): r(t) = [w(t)]x theta\nthin, the measurement; thick, the fit"
    )
    left.legend(fontsize=7)

    middle = axes[1]
    before = np.linalg.norm(measured, axis=1) * 1.0e6
    after = np.linalg.norm(measured - fitted, axis=1) * 1.0e6
    bins = np.linspace(0.0, float(np.percentile(before, 99.5)), 50)
    middle.hist(before, bins=bins, color=BOUND, alpha=0.6,
                label=f"before the fit, RMS {float(np.sqrt(np.mean(before**2))):.3f}")
    middle.hist(after, bins=bins, color=BATCH, alpha=0.6,
                label=f"after the fit, RMS {float(np.sqrt(np.mean(after**2))):.3f}")
    noise = np.sqrt(3.0) * record.pair_sigma("secondary_1") * 1.0e6
    middle.axvline(noise, color="#333333", linestyle="--", linewidth=1.2,
                   label=f"sqrt(3) x pair sigma = {noise:.3f}")
    middle.set_xlabel("residual magnitude, micro-rad/s")
    middle.set_ylabel("samples")
    middle.set_title("what the estimator removes")
    middle.legend(fontsize=7)

    right = axes[2]
    axis_direction = sensor.degrees_of_freedom[-1].direction
    best_direction = sensor.degrees_of_freedom[0].direction
    along = measured @ axis_direction * 1.0e6
    across = measured @ best_direction * 1.0e6
    right.hist(across, bins=50, color=BATCH, alpha=0.65,
               label=f"along the best-observed dof\nstd {float(np.std(across)):.3f}")
    right.hist(along, bins=50, color=UNOBSERVABLE, alpha=0.65,
               label=f"along the rotation axis\nstd {float(np.std(along)):.3f}")
    right.set_xlabel("residual component, micro-rad/s")
    right.set_ylabel("samples")
    right.set_title(
        "the residual along the rotation axis is noise.\n"
        "[w]x w = 0, so no alignment about that axis\ncan produce a signal there."
    )
    right.legend(fontsize=7)

    figure.suptitle(
        f"A single-axis record at 1e-5 rad/s sensor noise, chosen so the structure is "
        f"visible. Rank {sensor.rank} of 3; the unobservable direction is "
        f"{sensor.unobservable_axis_angle_deg:.4f} deg from the rotation axis.",
        fontsize=11,
    )
    figure.subplots_adjust(top=0.82)
    return save(figure, "residual_structure")


if __name__ == "__main__":
    print(main())
