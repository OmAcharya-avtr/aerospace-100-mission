"""Where every miss in a population lands, against the certified threshold curve.

What to notice: every consequential miss (red) lies above and to the right of
the threshold curve, which is what soundness means in a picture. The blue
cloud below the curve is what the certificate disposes of without simulating
anything, and it is the large majority of the population.
"""

from __future__ import annotations

import sys  # noqa: E402

sys.dont_write_bytecode = True

import math  # noqa: E402

import numpy as np  # noqa: E402

from _common import (  # noqa: E402
    CERTIFIED,
    CONSEQUENTIAL,
    TOLERANCE,
    UNDECIDED,
    bootstrap,
    save,
    style,
)

ROOT = bootstrap()

import matplotlib.pyplot as plt  # noqa: E402

from hilverdict.certificate import (  # noqa: E402
    CertificateModel,
    certified_overrun_s,
    certify_run,
)
from hilverdict.plant import LoopModel  # noqa: E402
from hilverdict.resimulate import resimulate_run  # noqa: E402
from hilverdict.synthetic import CAMPAIGNS, generate_run  # noqa: E402

STEPS = 200
SEEDS = tuple(range(8000, 8036))


def main() -> int:
    style()
    model = LoopModel()
    constants = CertificateModel.build(model, STEPS)
    overruns = []
    steps = []
    dismissed = []
    consequential = []
    for position, seed in enumerate(SEEDS):
        profile, amplitude = CAMPAIGNS[position % len(CAMPAIGNS)]
        log = generate_run(
            seed=seed, profile=profile, amplitude_deg=amplitude, steps=STEPS
        )
        certificate = certify_run(log, 1.0)
        results = resimulate_run(log, 1.0)
        for item, result in zip(certificate.certificates, results, strict=True):
            overruns.append(item.event.overrun_s * 1e3)
            steps.append(max(math.degrees(item.event.input_step_rad), 1e-4))
            dismissed.append(item.dismissed)
            consequential.append(result.consequential)
    overruns = np.array(overruns)
    steps = np.array(steps)
    dismissed = np.array(dismissed)
    consequential = np.array(consequential)

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    groups = (
        (dismissed & ~consequential, CERTIFIED, "o", "dismissed by eq (13)"),
        (~dismissed & ~consequential, UNDECIDED, "s", "undecided, then inconsequential"),
        (~dismissed & consequential, CONSEQUENTIAL, "^", "undecided, then consequential"),
        (dismissed & consequential, "#000000", "X", "dismissed but consequential"),
    )
    for mask, colour, marker, label in groups:
        if not np.any(mask):
            label = f"{label} (none)"
        axes[0].scatter(
            np.clip(overruns[mask], 1e-3, None),
            steps[mask],
            s=14,
            c=colour,
            marker=marker,
            alpha=0.75,
            label=label,
        )
    grid_steps = np.logspace(-4, 1.6, 200)
    curve = [
        certified_overrun_s(constants, math.radians(d), STEPS, 1.0) * 1e3
        for d in grid_steps
    ]
    axes[0].plot(curve, grid_steps, color=TOLERANCE, lw=1.4, ls="--",
                 label="eq (14) threshold")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("overrun [ms]")
    axes[0].set_ylabel("input step held [deg]")
    axes[0].set_title(f"{len(overruns)} misses over {len(SEEDS)} runs")
    axes[0].legend(loc="lower left", fontsize=7.5)

    counts = [
        int(np.count_nonzero(dismissed & ~consequential)),
        int(np.count_nonzero(~dismissed & ~consequential)),
        int(np.count_nonzero(~dismissed & consequential)),
        int(np.count_nonzero(dismissed & consequential)),
    ]
    labels = ["dismissed", "undecided,\ninconsequential",
              "undecided,\nconsequential", "dismissed BUT\nconsequential"]
    colours = [CERTIFIED, UNDECIDED, CONSEQUENTIAL, "#000000"]
    bars = axes[1].bar(labels, counts, color=colours)
    for bar, count in zip(bars, counts, strict=True):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + max(counts) * 0.015,
            str(count),
            ha="center",
            fontsize=9,
        )
    axes[1].set_ylabel("misses")
    axes[1].set_title("the last bar is the error the product exists to prevent")
    axes[1].tick_params(axis="x", labelsize=7.5)

    figure.suptitle(
        "hilverdict: miss disposition against the certified threshold", fontsize=10
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    path = save(figure, "miss_disposition")
    print(f"written: {path}")
    print(f"misses {len(overruns)}, dismissed {counts[0] + counts[3]}, "
          f"consequential {counts[2] + counts[3]}")
    print(f"dismissal rate {float(np.mean(dismissed)):.6f}")
    print(f"consequential misses among the dismissed: {counts[3]}")
    print(f"np version {np.__version__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
