"""Compare steering laws on one manoeuvre that passes close to a singularity.

Writes ``screenshots/steering_comparison.png``.  Runtime: about 20 seconds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cmgsteer.arrays import pyramid_array  # noqa: E402
from cmgsteer.dataset import manoeuvre_suite  # noqa: E402
from cmgsteer.nullmotion import GradientNullMotion, NoNullMotion  # noqa: E402
from cmgsteer.simulate import run_steering  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "screenshots" / "steering_comparison.png"
SEED = 20260902
MAX_RATE = 2.0

# gsr is drawn dashed because it sits almost exactly on top of sr; that overlap
# is a result, not a plotting accident.
CONFIGS = [
    ("pinv", "pinv", NoNullMotion(), "#b2182b", "-", 1.0),
    ("sr", "sr", NoNullMotion(), "#2166ac", "-", 2.2),
    ("gsr", "gsr", NoNullMotion(), "#762a83", "--", 1.1),
    (
        "sr + gradient null motion",
        "sr",
        GradientNullMotion(gain=1.0, max_rate=0.5),
        "#4d9221",
        "-",
        1.4,
    ),
]


def main() -> int:
    array = pyramid_array()
    suite = manoeuvre_suite(
        array, 3, seed=SEED, n_segments=3, segment_duration=5.0, dt=0.02
    )
    profile, start = suite.profiles[0], suite.initial_deltas[0]

    histories = {}
    for label, method, policy, _, _, _ in CONFIGS:
        histories[label] = run_steering(
            array,
            start,
            profile,
            method=method,
            null_policy=policy,
            max_gimbal_rate=MAX_RATE,
        )

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 7.4))

    ax = axes[0, 0]
    for label, _, _, colour, style, width in CONFIGS:
        h = histories[label]
        ax.plot(h.times, h.measure, color=colour, lw=width, ls=style, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("singularity measure $m$ [(N m s/rad)$^3$]")
    ax.set_title("how close each law comes to a singularity")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[0, 1]
    for label, _, _, colour, style, width in CONFIGS:
        h = histories[label]
        ax.plot(
            h.times[:-1], h.torque_error_norm, color=colour, lw=width, ls=style, label=label
        )
    ax.set_yscale("log")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("instantaneous torque error [N m]")
    ax.set_title("torque the array failed to deliver")
    ax.grid(alpha=0.3, which="both")

    ax = axes[1, 0]
    for label, _, _, colour, style, width in CONFIGS:
        h = histories[label]
        peak = np.max(np.abs(h.gimbal_rates), axis=1)
        ax.plot(h.times[:-1], peak, color=colour, lw=width, ls=style, label=label)
    ax.axhline(MAX_RATE, color="0.3", ls="--", lw=1.0, label=f"rate limit {MAX_RATE} rad/s")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("peak gimbal rate [rad/s]")
    ax.set_title("gimbal rate demanded, after the rate limit")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    for label, _, _, colour, style, width in CONFIGS:
        h = histories[label]
        cumulative = np.cumsum(np.linalg.norm(h.momentum_error, axis=1))
        ax.plot(h.times[:-1], cumulative, color=colour, lw=width, ls=style, label=label)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("cumulative momentum error [N m s]")
    ax.set_title("path-length momentum error, the honest bottom line")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=130)
    plt.close(fig)
    print(f"saved {OUT}")
    print(f"{'configuration':>28} {'min m':>10} {'path err':>12} {'net err':>12} {'sat':>6}")
    for label, _, _, _, _, _ in CONFIGS:
        h = histories[label]
        print(
            f"{label:>28} {h.min_measure:>10.6f} {h.total_momentum_error_path:>12.6e} "
            f"{h.accumulated_momentum_error:>12.6e} {h.n_rate_limited:>6d}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
