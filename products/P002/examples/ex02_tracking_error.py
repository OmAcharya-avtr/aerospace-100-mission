"""Example 2 - tracking error under platform jitter, with the loss-of-lock spike.

Generates ``screenshots/ex02_tracking_error.png``: line-of-sight error time
series for PID and LQR under the same synthesised jitter realisation, the
commanded torques, and the jitter PSD (target vs realised).

Run from the product root:  python examples/ex02_tracking_error.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trackbench.dynamics import JitterPSD, welch_psd  # noqa: E402
from trackbench.sim import Scenario, run_episode  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "screenshots" / "ex02_tracking_error.png"
SEED = 2026


def main() -> int:
    """Build the figure and save it."""
    scenarios = {
        "PID (5 Hz, $\\zeta$=0.707)": Scenario(controller="pid", seed=SEED),
        "LQR (5 Hz Butterworth)": Scenario(controller="lqr", seed=SEED),
    }
    results = {k: run_episode(v, seed=SEED) for k, v in scenarios.items()}
    sc = scenarios["PID (5 Hz, $\\zeta$=0.707)"]

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=False)

    ax = axes[0]
    ref = next(iter(results.values()))
    ax.plot(ref.t, ref.jitter * 1e6, color="0.6", lw=0.6,
            label="open-loop disturbance (jitter + spike)")
    for name, res in results.items():
        ax.plot(res.t, res.los_error * 1e6, lw=0.8,
                label=f"{name}: RMS {res.track_rms_rad * 1e6:.3f} $\\mu$rad")
    ax.axhline(sc.track_threshold * 1e6, color="crimson", ls="--", lw=1.0,
               label=f"lock threshold {sc.track_threshold * 1e6:.0f} $\\mu$rad")
    ax.axhline(-sc.track_threshold * 1e6, color="crimson", ls="--", lw=1.0)
    for name, res in results.items():
        if res.loss_time_s is not None:
            ax.axvline(res.loss_time_s, color="k", ls=":", lw=1.0)
    ax.set_ylabel("LOS error [$\\mu$rad]")
    ax.set_xlabel("time [s]")
    ax.set_title("Line-of-sight error under platform jitter; "
                 f"loss-of-lock spike at t = {sc.spike_time} s "
                 f"(amplitude {sc.spike_amplitude * 1e6:.0f} $\\mu$rad)", fontsize=10)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.25)

    ax = axes[1]
    peak = 0.0
    for name, res in results.items():
        ax.plot(res.t, res.torque * 1e3, lw=0.7, label=name)
        peak = max(peak, float(np.max(np.abs(res.torque))))
    ax.set_ylabel("commanded torque [mN m]")
    ax.set_xlabel("time [s]")
    ax.set_title(
        "Actuator command; peak "
        f"{peak * 1e3:.1f} mN m = {peak / sc.torque_max:.2%} of the "
        f"{sc.torque_max} N m limit (no saturation)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # dedicated spike-free 8 s runs so the Welch estimate reaches below the
    # 3 Hz jitter corner and the 5 Hz loop bandwidth
    ax = axes[2]
    psd = JitterPSD(sc.jitter_s0, sc.jitter_f_corner, sc.jitter_order)
    fs = 1.0 / sc.dt
    nperseg = 8192
    spectral = {
        name: run_episode(
            Scenario(controller=v.controller, seed=SEED, track_duration=8.0,
                     spike_time=4.0, spike_amplitude=1e-12),
            seed=SEED,
        )
        for name, v in scenarios.items()
    }
    ref_sp = next(iter(spectral.values()))
    f, p = welch_psd(ref_sp.jitter, fs, nperseg=nperseg)
    ax.loglog(f[1:], psd(f[1:]), "k-", lw=1.5, label="target PSD (eq. 4)")
    ax.loglog(f[1:], p[1:], color="#1f77b4", lw=0.8, alpha=0.8,
              label="realised open-loop PSD (Welch, 8 s spike-free run)")
    for name, res in spectral.items():
        fe, pe = welch_psd(res.los_error, fs, nperseg=nperseg)
        ax.loglog(fe[1:], pe[1:], lw=0.8, alpha=0.85, label=f"closed-loop error, {name}")
    ax.axvline(sc.bandwidth_hz, color="darkgreen", ls=":",
               label=f"design bandwidth {sc.bandwidth_hz} Hz")
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("PSD [rad$^2$/Hz]")
    ax.set_title("Disturbance and residual-error spectra "
                 "(rejection below the loop bandwidth)", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25, which="both")

    fig.suptitle("TrackBench - closed-loop pointing under synthesised platform jitter",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=130)
    plt.close(fig)

    for name, res in results.items():
        sys.stdout.write(
            f"{name}: RMS {res.track_rms_rad:.4e} rad, peak {res.track_peak_rad:.4e} rad, "
            f"lock lost = {res.lock_lost} at {res.loss_time_s} s, "
            f"saturation fraction {res.saturation_fraction:.4f}\n"
        )
    for name, res in spectral.items():
        sys.stdout.write(
            f"spike-free 8 s run, {name}: open-loop jitter RMS "
            f"{np.std(res.jitter):.4e} rad, closed-loop error RMS "
            f"{np.std(res.los_error):.4e} rad, rejection factor "
            f"{np.std(res.jitter) / np.std(res.los_error):.2f}\n"
        )
    sys.stdout.write(f"saved {OUT}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
