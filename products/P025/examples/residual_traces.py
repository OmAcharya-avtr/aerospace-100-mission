"""What a fault looks like in the residual, and when each test notices.

Injects a 3-sigma gyro bias into the closed loop and plots, on a common time
axis, the two normalised residual channels, the sliding chi-squared statistic
against its design threshold, and the CUSUM statistic against a threshold
designed for a 2000-sample mean time between false alarms.  The onset sample
and each test's first alarm are marked.

    python examples/residual_traces.py
"""

from __future__ import annotations

import numpy as np
from _plotstyle import COLORS, save

import matplotlib.pyplot as plt  # noqa: E402  (backend set in _plotstyle)

from fdiscope import (  # noqa: E402
    ChiSquaredDetector,
    CusumDetector,
    FaultSpec,
    FaultType,
    LoopConfig,
    PlantConfig,
    build_filter,
    cusum_threshold_for_arl0,
    detection_delay,
    loop_matrices,
    normalised_bias_signature,
    simulate_loop,
)

ONSET = 600
STEPS = 1600
SEED = 7
BIAS_SIGMAS = 3.0
ALPHA = 1.0e-3
CHI_WINDOW = 25
TARGET_ARL0 = 2000.0


def main() -> None:
    plant = PlantConfig()
    kf = build_filter(loop_matrices(plant))
    bias = BIAS_SIGMAS * float(np.sqrt(plant.gyro_var_rad2_s2))
    direction, mu = normalised_bias_signature(kf, [0.0, bias])
    h_cusum = cusum_threshold_for_arl0(TARGET_ARL0, mu)

    spec = FaultSpec(FaultType.SENSOR_BIAS, ONSET, bias, 1)
    run = simulate_loop(LoopConfig(n_steps=STEPS, seed=SEED), spec)

    chi = ChiSquaredDetector(window=CHI_WINDOW, dim=2, alpha=ALPHA).run(run.residual)
    cusum = CusumDetector(direction=direction, mu=mu, threshold=h_cusum).run(run.residual)
    d_chi = detection_delay(chi.alarm, ONSET)
    d_cusum = detection_delay(cusum.alarm, ONSET)

    t = run.t_s
    fig, axes = plt.subplots(3, 1, figsize=(9.0, 7.2), sharex=True)

    ax = axes[0]
    ax.plot(t, run.residual[:, 0], lw=0.6, color=COLORS["channel0"], label="attitude channel")
    ax.plot(t, run.residual[:, 1], lw=0.6, color=COLORS["channel1"], label="rate channel")
    ax.axhline(0.0, color="k", lw=0.5)
    ax.set_ylabel("normalised residual [-]")
    ax.set_title(
        f"{BIAS_SIGMAS:.0f}-sigma gyro bias at t = {ONSET * plant.dt_s:.0f} s "
        f"(steady-state residual mean mu = {mu:.2f})"
    )
    ax.legend(loc="upper left", ncol=2)

    ax = axes[1]
    ax.plot(t, chi.statistic, lw=0.8, color=COLORS["chi2_short"], label=f"chi2, W = {CHI_WINDOW}")
    ax.axhline(
        chi.threshold,
        color=COLORS["threshold"],
        lw=1.0,
        ls="--",
        label=f"threshold, alpha = {ALPHA:g} ({chi.threshold:.1f})",
    )
    ax.set_yscale("log")
    ax.set_ylabel("windowed NIS [-]")
    ax.legend(loc="upper left")

    ax = axes[2]
    ax.plot(t, cusum.statistic, lw=0.8, color=COLORS["cusum"], label="CUSUM on the bias direction")
    ax.axhline(
        h_cusum,
        color=COLORS["threshold"],
        lw=1.0,
        ls="--",
        label=f"threshold for ARL0 = {TARGET_ARL0:.0f} ({h_cusum:.2f})",
    )
    ax.set_ylabel("CUSUM statistic [-]")
    ax.set_xlabel("time [s]")
    ax.set_ylim(0.0, 8.0 * h_cusum)
    ax.text(
        0.985,
        0.90,
        f"axis clipped at {8.0 * h_cusum:.1f};\nthe statistic reaches {cusum.statistic.max():.0f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color=COLORS["cusum"],
    )
    ax.legend(loc="upper left")

    for ax in axes:
        ax.axvline(ONSET * plant.dt_s, color=COLORS["onset"], lw=1.0, ls=":")
    if np.isfinite(d_chi):
        axes[1].axvline((ONSET + d_chi) * plant.dt_s, color=COLORS["chi2_short"], lw=0.8)
    if np.isfinite(d_cusum):
        axes[2].axvline((ONSET + d_cusum) * plant.dt_s, color=COLORS["cusum"], lw=0.8)

    path = save(fig, "residual_traces.png")
    print(f"saved {path}")
    print(f"mean NIS before onset    = {np.mean(run.nis[300:ONSET]):.4f}  (expect 2.0)")
    print(f"mean NIS after onset     = {np.mean(run.nis[ONSET:]):.4f}")
    print(f"chi2 threshold           = {chi.threshold:.4f} (W = {CHI_WINDOW}, alpha = {ALPHA:g})")
    print(f"CUSUM threshold          = {h_cusum:.4f} (mu = {mu:.4f}, ARL0 = {TARGET_ARL0:.0f})")
    print(f"chi2 detection delay     = {d_chi:.0f} samples ({d_chi * plant.dt_s:.1f} s)")
    print(f"CUSUM detection delay    = {d_cusum:.0f} samples ({d_cusum * plant.dt_s:.1f} s)")
    print(
        f"false alarms before onset = chi2 {int(np.count_nonzero(chi.alarm[:ONSET]))}, "
        f"CUSUM {int(np.count_nonzero(cusum.alarm[:ONSET]))}"
    )


if __name__ == "__main__":
    main()
