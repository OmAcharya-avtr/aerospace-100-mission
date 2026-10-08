"""Two different worlds, one residual: what this monitor cannot tell you.

Saves ``../screenshots/invalidation_vs_fault.png``. What to notice: the top two
panels are the residual of an asset fault and of a twin configuration error.
They are not similar; they are the same trace. The bottom panel is their
difference, which is floating-point rounding at the 1e-14 level. Any tool that
claims to attribute a residual like this to a cause is claiming something the
residual does not contain.
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
    DetectorSpec,
    calibrate_threshold,
    first_alarm,
    in_control_streams,
    max_absolute_difference,
    paired_streams,
)

DT = 0.05


def main() -> int:
    pair = paired_streams(offset=0.02, onset=400, n_runs=20, n_samples=1200, seed=53400)
    diff = max_absolute_difference(pair)
    bank = in_control_streams(n_runs=150, n_samples=2000, seed=53001)
    spec = DetectorSpec("cusum")
    threshold = calibrate_threshold(spec, bank, 1000.0).threshold
    stat_a = spec.statistic(pair.asset_fault)
    stat_b = spec.statistic(pair.twin_invalid)
    idx_a = first_alarm(stat_a, threshold)
    idx_b = first_alarm(stat_b, threshold)

    t = np.arange(pair.asset_fault.shape[1]) * DT
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.5), sharex=True)

    axes[0].plot(t, pair.asset_fault[0], lw=0.6, color="0.55")
    axes[0].plot(t, pair.asset_fault.mean(axis=0), lw=1.6, color="C0")
    axes[0].set_ylabel("residual z")
    axes[0].set_title(
        "World A, asset fault: the sensor develops a 0.02 rad bias; the twin is right",
        loc="left",
        fontsize=10,
    )

    axes[1].plot(t, pair.twin_invalid[0], lw=0.6, color="0.55")
    axes[1].plot(t, pair.twin_invalid.mean(axis=0), lw=1.6, color="C1")
    axes[1].set_ylabel("residual z")
    axes[1].set_title(
        "World B, twin invalidation: the sensor is fine; the twin's declared offset "
        "is revised to -0.02 rad",
        loc="left",
        fontsize=10,
    )

    for ax in axes[:2]:
        ax.axvline(pair.onset * DT, color="C3", ls="--", lw=1.2)
        ax.axhline(0.0, color="k", lw=0.6)
        ax.set_ylim(-4.5, 4.5)
        ax.grid(alpha=0.25)

    axes[2].plot(t, (pair.asset_fault - pair.twin_invalid)[0], lw=0.9, color="C2")
    axes[2].set_ylabel("world A - world B")
    axes[2].set_xlabel("time (s), 20 Hz sampling")
    axes[2].grid(alpha=0.25)
    axes[2].set_title(
        f"Difference: max |A - B| = {diff:.2e} over all {pair.asset_fault.shape[0]} runs, "
        f"about {diff / np.finfo(float).eps:.0f} ulps of a double",
        loc="left",
        fontsize=10,
    )
    axes[2].text(
        0.02,
        0.08,
        f"CUSUM at ARL0 = 1000 alarms at sample {int(idx_a[idx_a >= 0].mean())} in both "
        f"worlds\nidentical alarm indices in every run: "
        f"{bool(np.array_equal(idx_a, idx_b))}",
        transform=axes[2].transAxes,
        fontsize=8.5,
        bbox=dict(boxstyle="round", fc="white", ec="0.7"),
    )

    fig.suptitle(
        "Twin invalidation against asset fault: the residual does not distinguish them\n"
        "twininvalidate, shared noise realisation, seed 53400",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = ROOT / "screenshots" / "invalidation_vs_fault.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")
    print(f"max |world A - world B| {diff:.4e}")
    print(f"CUSUM threshold {threshold:.4f}, mean alarm sample A "
          f"{idx_a[idx_a >= 0].mean():.1f}, B {idx_b[idx_b >= 0].mean():.1f}")
    print(f"identical alarm indices in every run: {bool(np.array_equal(idx_a, idx_b))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
