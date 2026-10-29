"""The five declared excitation profiles, and what each one makes observable.

The point of the figure: the alignment degrees of freedom a record supports are
decided by the motion, not by the instrument or the data volume. The bottom row
is equation (10) made visible -- the Gramian eigenvalue along a direction is
(W / s^2) (1 - p) with p the excitation share, so the most excited axis is the
least observable one.
"""

from __future__ import annotations

from _common import (
    FLOOR,
    PROFILE_COLOURS,
    VERDICT_COLOURS,
    bootstrap,
    save,
    style,
)

bootstrap()
style()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from aligncal import dataset, observability  # noqa: E402


def main() -> str:
    profiles = list(dataset.PROFILES)
    figure, axes = plt.subplots(3, len(profiles), figsize=(16.0, 8.4))
    figure.subplots_adjust(hspace=0.62, wspace=0.3)

    for column, profile in enumerate(profiles):
        generated = dataset.generate(profile=profile, samples=1500, seed=20261029)
        record = generated.record
        report = observability.assess(record)
        sensor = report["secondary_1"]

        top = axes[0, column]
        for axis in range(3):
            top.plot(
                record.times,
                record.reference_rates[:, axis] * 1.0e3,
                linewidth=0.7,
                color=["#1f6fb4", "#d9850b", "#3f8f4f"][axis],
                label=f"body {'xyz'[axis]}",
            )
        top.set_title(f"{profile}\n{dataset.PROFILES[profile]} excited axis/axes")
        top.set_xlabel("time, s")
        if column == 0:
            top.set_ylabel("measured rate, mrad/s")
            top.legend(fontsize=7, ncol=3)

        middle = axes[1, column]
        middle.bar(
            [0, 1, 2],
            report.excitation,
            color=PROFILE_COLOURS[profile],
            edgecolor="white",
        )
        middle.set_ylim(0.0, 1.05)
        middle.set_xticks([0, 1, 2])
        middle.set_xticklabels(["p0", "p1", "p2"])
        middle.set_title(
            f"excitation shares, eq (9)\nentropy {report.excitation_entropy:.4f} of 1.5850 bits"
        )
        if column == 0:
            middle.set_ylabel("share of rotational energy")

        bottom = axes[2, column]
        values = np.array([dof.eigenvalue for dof in sensor.degrees_of_freedom])
        colours = [VERDICT_COLOURS[dof.verdict] for dof in sensor.degrees_of_freedom]
        bottom.bar([0, 1, 2], values, color=colours, edgecolor="white")
        gate = sensor.floor_factor * sensor.noise_floor
        bottom.axhline(
            gate,
            color=FLOOR,
            linestyle="--",
            linewidth=1.1,
            label=f"noise floor gate, eq (11)\n{sensor.floor_factor:g} x {sensor.noise_floor:.0f}",
        )
        bottom.set_yscale("log")
        bottom.set_xticks([0, 1, 2])
        bottom.set_xticklabels(["dof 0", "dof 1", "dof 2"])
        bottom.set_title(f"Gramian spectrum, eq (8)\nrank {sensor.rank} of 3")
        if column == 0:
            bottom.set_ylabel("eigenvalue, 1/(s^2 rad^2)")
            bottom.legend(fontsize=7, loc="upper center")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colour, label=name)
        for name, colour in VERDICT_COLOURS.items()
    ]
    figure.legend(handles=handles, loc="lower center", ncol=3, fontsize=9)
    figure.suptitle(
        "What the motion makes observable. Bottom row: the most excited axis (middle row) "
        "is the least observable one, by equation (10).",
        fontsize=11,
    )
    figure.subplots_adjust(bottom=0.09, top=0.90)
    return save(figure, "rate_profiles")


if __name__ == "__main__":
    print(main())
