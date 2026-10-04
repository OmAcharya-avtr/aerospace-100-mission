"""Run the same loop against both backends and show that they agree exactly.

Produces ``screenshots/hil_loop_parity.png``: the attitude the loop drove, the
torque it commanded, the difference between the two backends' signal paths
(exactly zero), and the latency histogram of the measured run.

Run: ``python examples/hil_loop_parity.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hilforge.backends import make_backend_pair
from hilforge.loop import STAGES, HilLoop, LoopConfig
from hilforge.timing import PeriodSpec

PERIOD = 0.010
N = 1200
SEED = 20261004
OUT = ROOT / "screenshots" / "hil_loop_parity.png"


def main() -> int:
    sim, dev = make_backend_pair(seed=SEED, sample_dt_s=PERIOD)
    deterministic = LoopConfig(
        period=PeriodSpec(period_s=PERIOD),
        n_iterations=N,
        injected_durations_s=tuple(np.full(N, 0.004)),
    )
    a = HilLoop(sim, deterministic).run()
    b = HilLoop(dev, deterministic).run()
    sig_a = a.signal_matrix()
    sig_b = b.signal_matrix()
    identical = sig_a.tobytes() == sig_b.tobytes()

    # A second, wall-clock-timed run for the latency panel.
    sim2, _ = make_backend_pair(seed=SEED, sample_dt_s=PERIOD)
    measured = HilLoop(
        sim2,
        LoopConfig(
            period=PeriodSpec(period_s=PERIOD), n_iterations=N, record_signals=False
        ),
    ).run()

    print(f"period                    : {PERIOD:.6e} s")
    print(f"iterations                : {N}")
    print(f"simulated backend         : {a.backend_name} (is_hardware={a.is_hardware})")
    print(f"device backend            : {b.backend_name} (is_hardware={b.is_hardware})")
    print(f"signal bytes identical    : {identical}")
    print(f"max |difference|          : {float(np.max(np.abs(sig_a - sig_b))):.3e}")
    print(f"data digest (both)        : {a.data_digest()}")
    print(f"final theta [rad]         : {sig_a[-1, 0]:.6e}")
    print(f"final theta_hat [rad]     : {sig_a[-1, 2]:.6e}")
    print(f"measured p50 total [s]    : "
          f"{measured.stage_histograms['total'].percentile(0.50):.6e}")
    print(f"measured p99 total [s]    : "
          f"{measured.stage_histograms['total'].percentile(0.99):.6e}")
    print(f"figure                    : {OUT}")

    t = np.arange(N) * PERIOD
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.6))
    fig.suptitle(
        "HilForge: one loop, two backends — simulated plant and loopback device stub\n"
        "(Level 3, hardware-pending: neither backend is hardware)",
        fontsize=12,
    )

    ax = axes[0, 0]
    ax.plot(t, sig_a[:, 0], lw=1.0, label="measured attitude (simulated)")
    ax.plot(t, sig_b[:, 0], lw=1.0, ls="--", label="measured attitude (device stub)")
    ax.plot(t, sig_a[:, 2], lw=1.4, color="k", label="complementary-filter estimate")
    ax.set_xlabel("time [s]")
    ax.set_ylabel(r"attitude $\theta$ [rad]")
    ax.set_title("the loop drives the attitude to the command")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t, sig_a[:, 3], lw=1.0, label="commanded torque")
    ax.plot(t, sig_a[:, 4], lw=1.0, ls="--", label="applied torque")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("torque [N·m]")
    ax.set_title("command and applied command")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    diff = sig_a - sig_b
    labels = [r"$\theta_{meas}$", r"$\omega_{meas}$", r"$\hat{\theta}$", "command", "applied"]
    ax.bar(range(5), np.max(np.abs(diff), axis=0), color="#2f6f4e")
    ax.set_xticks(range(5))
    ax.set_xticklabels(labels)
    ax.set_ylabel("max |simulated − device| [SI]")
    ax.set_ylim(0, 1)
    ax.set_title(
        f"simulation/device parity: every column exactly 0\n"
        f"bytes identical = {identical}"
    )
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1, 1]
    totals = measured.durations_s() * 1e6
    ax.hist(totals, bins=80, range=(0, np.percentile(totals, 99.0)), color="#44618c")
    for p, style in ((0.50, "-"), (0.90, "--"), (0.99, ":")):
        value = measured.stage_histograms["total"].percentile(p) * 1e6
        ax.axvline(value, color="k", ls=style, lw=1.2, label=f"p{p * 100:g} = {value:.0f} µs")
    ax.set_xlabel("iteration duration [µs]")
    ax.set_ylabel("count")
    ax.set_title(
        "measured iteration latency, wall clock\n"
        "shared single-core container — not a hardware number"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)
    plt.close(fig)
    print()
    print("per-stage p50 of the measured run [µs]:")
    for stage in STAGES:
        hist = measured.stage_histograms[stage]
        print(f"  {stage:<10}: {hist.percentile(0.50) * 1e6:8.2f}")
    return 0 if identical else 1


if __name__ == "__main__":
    raise SystemExit(main())
