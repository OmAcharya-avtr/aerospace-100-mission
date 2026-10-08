"""Residual streams for the three declared change types.

Saves ``../screenshots/residual_stream.png``. What to notice: the parameter
step moves the residual mean and leaves its spread alone; the ramp moves the
mean gradually; the noise-variance change leaves the mean exactly where it was
and only widens the band, which is why every mean-shift detector in this
package is mis-specified for it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validation"))
from _bootstrap import add_src_to_path  # noqa: E402

ROOT = add_src_to_path()

from twininvalidate import (  # noqa: E402
    SCENARIO_LABELS,
    SCENARIOS,
    AssetChange,
    StreamSpec,
    simulate_residuals,
)

ONSET = 400
N_SAMPLES = 1200
DT = 0.05


def main() -> int:
    fig, axes = plt.subplots(3, 1, figsize=(10, 8.5), sharex=True)
    t = np.arange(N_SAMPLES) * DT
    for ax, (name, change) in zip(axes, SCENARIOS.items(), strict=True):
        shifted = AssetChange(change.kind, ONSET, change.magnitude, change.ramp_samples)
        z = simulate_residuals(
            StreamSpec(change=shifted, n_runs=60, n_samples=N_SAMPLES, seed=53500)
        )
        ax.plot(t, z[0], lw=0.6, color="0.55", label="one realisation")
        ax.plot(t, z.mean(axis=0), lw=1.6, color="C0", label="ensemble mean of 60")
        band = z.std(axis=0)
        ax.fill_between(
            t,
            z.mean(axis=0) - band,
            z.mean(axis=0) + band,
            color="C0",
            alpha=0.18,
            label="ensemble mean +/- 1 sd",
        )
        ax.axvline(ONSET * DT, color="C3", ls="--", lw=1.2, label=f"onset, sample {ONSET}")
        ax.axhline(0.0, color="k", lw=0.6)
        ax.set_ylabel("normalised residual z")
        ax.set_ylim(-5, 5)
        ax.set_title(SCENARIO_LABELS[name], loc="left", fontsize=10)
        ax.grid(alpha=0.25)
    axes[0].legend(loc="upper left", fontsize=8, ncol=2)
    axes[-1].set_xlabel("time (s), 20 Hz sampling")
    fig.suptitle(
        "Residual signature of each declared change type\n"
        "twininvalidate, reference attitude channel, seed 53500",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = ROOT / "screenshots" / "residual_stream.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")
    for name, change in SCENARIOS.items():
        shifted = AssetChange(change.kind, ONSET, change.magnitude, change.ramp_samples)
        z = simulate_residuals(
            StreamSpec(change=shifted, n_runs=60, n_samples=N_SAMPLES, seed=53500)
        )
        pre, post = z[:, :ONSET], z[:, ONSET + 400 :]
        print(
            f"  {name:<16} pre mean {pre.mean():+.4f} sd {pre.std():.4f} | "
            f"post mean {post.mean():+.4f} sd {post.std():.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
