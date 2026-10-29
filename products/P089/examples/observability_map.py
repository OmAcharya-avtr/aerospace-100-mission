"""Where the three-valued verdict comes from: excitation, noise, length, requirement.

Left: the rank verdict over excitation richness and noise. Middle: the
three-valued verdict over excitation richness and the DECLARED alignment
requirement, which is an input and not a property of the data. Right: the angle
between the worst-observed Gramian direction and the record's dominant rotation
axis, over sixty seeds, which is the claim that the unobservable direction IS
the rotation axis.
"""

from __future__ import annotations

from _common import MARGINAL, OBSERVABLE, UNOBSERVABLE, bootstrap, save, style

bootstrap()
style()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from aligncal import dataset, observability  # noqa: E402

PROFILES = ("single_axis", "near_single_axis", "two_axis", "three_axis", "isotropic")
NOISES = (2.0e-5, 5.0e-5, 1.0e-4, 2.0e-4, 5.0e-4, 1.0e-3)
REQUIREMENTS = (1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4, 1.0e-4, 3.0e-5)


def main() -> str:
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))
    figure.subplots_adjust(wspace=0.38)

    ranks = np.zeros((len(NOISES), len(PROFILES)))
    for row, noise in enumerate(NOISES):
        for column, profile in enumerate(PROFILES):
            generated = dataset.generate(
                profile=profile, samples=1500, noise_sigma=noise, seed=20261029
            )
            ranks[row, column] = observability.assess(generated.record)["secondary_1"].rank

    left = axes[0]
    image = left.imshow(
        ranks, cmap=ListedColormap([UNOBSERVABLE, MARGINAL, OBSERVABLE]), vmin=0.5, vmax=3.5,
        aspect="auto", origin="lower",
    )
    left.set_xticks(range(len(PROFILES)))
    left.set_xticklabels([p.replace("_", "\n") for p in PROFILES], fontsize=7)
    left.set_yticks(range(len(NOISES)))
    left.set_yticklabels([f"{n:.0e}" for n in NOISES], fontsize=8)
    left.set_ylabel("sensor noise one sigma, rad/s")
    left.set_title("rank: how many alignment degrees of\nfreedom clear the floor of eq (11)")
    for row in range(len(NOISES)):
        for column in range(len(PROFILES)):
            left.text(
                column, row, f"{int(ranks[row, column])}",
                ha="center", va="center", color="white", fontsize=9,
            )
    left.grid(False)
    colourbar = figure.colorbar(image, ax=left, ticks=[1, 2, 3])
    colourbar.set_label("rank")

    middle = axes[1]
    codes = np.zeros((len(REQUIREMENTS), len(PROFILES)))
    for row, requirement in enumerate(REQUIREMENTS):
        for column, profile in enumerate(PROFILES):
            generated = dataset.generate(profile=profile, samples=1500, seed=20261029)
            sensor = observability.assess(
                generated.record, required_sigma_rad=requirement
            )["secondary_1"]
            counts = {"observable": 0, "marginal": 0, "unobservable": 0}
            for dof in sensor.degrees_of_freedom:
                counts[dof.verdict] += 1
            codes[row, column] = counts["observable"]
            middle.text(
                column, row,
                f"{counts['observable']}/{counts['marginal']}/{counts['unobservable']}",
                ha="center", va="center", color="white", fontsize=7,
            )
    middle.imshow(codes, cmap="viridis", aspect="auto", origin="lower", vmin=0, vmax=3)
    middle.set_xticks(range(len(PROFILES)))
    middle.set_xticklabels([p.replace("_", "\n") for p in PROFILES], fontsize=7)
    middle.set_yticks(range(len(REQUIREMENTS)))
    middle.set_yticklabels([f"{r:.0e}" for r in REQUIREMENTS], fontsize=8)
    middle.set_ylabel("declared requirement, rad, one sigma")
    middle.set_title(
        "observable / marginal / unobservable.\nThe requirement is an INPUT, not a "
        "property of the data."
    )
    middle.grid(False)

    right = axes[2]
    angles = []
    body_angles = []
    for index in range(60):
        generated = dataset.generate(profile="single_axis", samples=900, seed=7000 + index)
        sensor = observability.assess(generated.record)["secondary_1"]
        angles.append(sensor.unobservable_axis_angle_deg)
        direction = sensor.degrees_of_freedom[-1].direction
        body_angles.append(
            min(
                min(
                    float(np.degrees(np.arccos(np.clip(abs(np.dot(direction, column)), -1, 1)))),
                    90.0,
                )
                for column in np.eye(3)
            )
        )
    right.hist(
        body_angles, bins=18, color=OBSERVABLE, alpha=0.85,
        label="angle to the nearest BODY axis",
    )
    right.axvline(
        float(np.max(angles)), color=UNOBSERVABLE, linewidth=2.0,
        label=f"worst angle to the ROTATION axis\n{float(np.max(angles)):.6f} deg",
    )
    right.set_xlabel("angle, degrees")
    right.set_ylabel("records, of 60")
    right.set_title(
        "the unobservable direction is the rotation\naxis, and the rotation axis is not a body axis"
    )
    right.legend(fontsize=8, loc="upper right")

    figure.suptitle(
        "The observability verdict: what the motion supports, and what you declared you need.",
        fontsize=11,
    )
    figure.subplots_adjust(top=0.82)
    return save(figure, "observability_map")


if __name__ == "__main__":
    print(main())
