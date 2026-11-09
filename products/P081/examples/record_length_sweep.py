"""Structural identifiability does not move with the record. Practical does.

What to notice: the rank and the least singular-value ratio are flat lines --
they are properties of the model and the observation function, not of the data
-- while the Cramer-Rao bound on the worst identifiable direction falls through
four orders of magnitude as the record lengthens. The shaded region is where
that bound exceeds the declared practical limit of 1, meaning the record cannot
pin the combination to within its own magnitude.
"""

from __future__ import annotations

import sys  # noqa: E402

sys.dont_write_bytecode = True

import matplotlib.pyplot as plt  # noqa: E402

from _common import ACCENT, ANALYTIC, LEARNED, REFUSED, bootstrap, save, style  # noqa: E402

ROOT = bootstrap()

from twindiverge.identifiability import (  # noqa: E402
    PRACTICAL_LIMIT,
    RANK_RTOL,
    analyse,
)
from twindiverge.plant import NOMINAL, Campaign  # noqa: E402
from twindiverge.telemetry import TelemetrySet  # noqa: E402

SAMPLES = (51, 101, 201, 401, 801, 1201, 1601, 2001)


def main() -> int:
    style()
    lengths = []
    ranks = []
    least = []
    worst_bound = []
    best_bound = []
    for samples in SAMPLES:
        period = 0.1
        campaign = Campaign(
            f"sweep{samples}", period, samples, 1.0, period * round(0.2 * (samples - 1))
        )
        telemetry = TelemetrySet(f"sweep{samples}", campaign, ("rate",))
        report = analyse(telemetry, NOMINAL)
        lengths.append(campaign.duration_s)
        ranks.append(report.rank)
        least.append(
            float(report.singular_values[-1] / report.singular_values[0])
        )
        worst_bound.append(float(report.direction_crb_relative[-1]))
        best_bound.append(float(report.direction_crb_relative[0]))

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    axes[0].semilogy(lengths, least, marker="s", color=ANALYTIC,
                     label="least singular-value ratio")
    axes[0].axhline(RANK_RTOL, color="black", linestyle="--", linewidth=1.0,
                    label=f"declared cutoff {RANK_RTOL:.0e}")
    twin = axes[0].twinx()
    twin.plot(lengths, ranks, marker="o", color=LEARNED, label="numerical rank")
    twin.set_ylim(0, 5.5)
    twin.set_ylabel("numerical rank of 5", color=LEARNED)
    twin.grid(False)
    axes[0].set_xlabel("record length [s]")
    axes[0].set_ylabel("least singular-value ratio")
    axes[0].set_title("structural: a property of the model, not the record")
    axes[0].legend(loc="lower left", fontsize=7)
    axes[0].set_ylim(1e-20, 1.0)

    axes[1].axhspan(PRACTICAL_LIMIT, 1e5, color=REFUSED, alpha=0.15)
    axes[1].loglog(lengths, worst_bound, marker="s", color=LEARNED,
                   label="worst identifiable direction")
    axes[1].loglog(lengths, best_bound, marker="o", color=ACCENT,
                   label="best identifiable direction")
    axes[1].axhline(PRACTICAL_LIMIT, color="black", linestyle="--", linewidth=1.0,
                    label=f"declared practical limit {PRACTICAL_LIMIT:.0f}")
    axes[1].set_xlabel("record length [s]")
    axes[1].set_ylabel("Cramer-Rao bound, relative standard error")
    axes[1].set_title("practical: a property of the record")
    axes[1].legend(loc="upper right", fontsize=7)
    figure.suptitle(
        "twindiverge: the two kinds of unidentifiability behave differently",
        fontsize=11,
    )
    path = save(figure, "record_length_sweep.png")
    print(f"written: {path}")
    for index, samples in enumerate(SAMPLES):
        print(
            f"  {lengths[index]:7.1f} s ({samples:5d} samples) rank {ranks[index]} "
            f"least ratio {least[index]:.6e} worst bound {worst_bound[index]:.6f}"
        )
    print(f"  worst bound falls by a factor of "
          f"{worst_bound[0] / worst_bound[-1]:.1f} while the rank does not move")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
