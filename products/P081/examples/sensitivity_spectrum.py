"""The singular-value spectrum of every declared telemetry set.

What to notice: the rate-only sets have one or two singular values at the 1e-16
level -- structurally zero, and the rank cutoff sits six decades above them --
while the two-channel sets do not. The horizontal line is the declared cutoff.
"""

from __future__ import annotations

import sys  # noqa: E402

sys.dont_write_bytecode = True

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from _common import ACCENT, ANALYTIC, LEARNED, REFUSED, bootstrap, save, style  # noqa: E402

ROOT = bootstrap()

from twindiverge.identifiability import RANK_RTOL, analyse  # noqa: E402
from twindiverge.plant import NOMINAL  # noqa: E402
from twindiverge.telemetry import TELEMETRY_SETS  # noqa: E402

COLOURS = {
    "rate_full": ANALYTIC,
    "rate_short": REFUSED,
    "rate_coast": LEARNED,
    "rate_torque_full": ACCENT,
    "rate_torque_short": "#6a3d9a",
}


def main() -> int:
    style()
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    floor = np.finfo(float).tiny
    for name, telemetry in TELEMETRY_SETS.items():
        report = analyse(telemetry, NOMINAL)
        ratios = np.maximum(report.singular_values / report.singular_values[0], floor)
        marker = "o" if "torque" in name else "s"
        axes[0].semilogy(
            np.arange(1, 6), ratios, marker=marker, color=COLOURS[name],
            label=f"{name} (rank {report.rank})",
        )
        axes[1].bar(
            name, report.rank, color=COLOURS[name],
            edgecolor="black", linewidth=0.5,
        )
    axes[0].axhline(
        RANK_RTOL, color="black", linestyle="--", linewidth=1.0,
        label=f"declared cutoff {RANK_RTOL:.0e}",
    )
    axes[0].set_xticks(np.arange(1, 6))
    axes[0].set_xlabel("singular value index")
    axes[0].set_ylabel("ratio to the largest singular value")
    axes[0].set_title("scaled sensitivity spectrum, equation (8)")
    axes[0].legend(fontsize=7, loc="lower left")
    axes[0].set_ylim(1e-20, 5.0)
    axes[1].set_ylim(0, 5.5)
    axes[1].set_ylabel("numerical rank of 5")
    axes[1].set_title("rank: what the telemetry can distinguish")
    axes[1].tick_params(axis="x", labelrotation=30, labelsize=7)
    for index, telemetry in enumerate(TELEMETRY_SETS.values()):
        report = analyse(telemetry, NOMINAL)
        axes[1].text(
            index, report.rank + 0.12, f"{len(report.refused)} refused",
            ha="center", fontsize=7,
        )
    figure.suptitle(
        "twindiverge: a structurally zero singular value is not a small one",
        fontsize=11,
    )
    path = save(figure, "sensitivity_spectrum.png")
    print(f"written: {path}")
    for name, telemetry in TELEMETRY_SETS.items():
        report = analyse(telemetry, NOMINAL)
        print(
            f"  {name:<20s} rank {report.rank} of 5, least ratio "
            f"{float(report.singular_values[-1] / report.singular_values[0]):.6e}, "
            f"refused {len(report.refused)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
