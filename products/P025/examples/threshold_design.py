"""The design surface: what a threshold buys and what it costs.

Four panels, all closed-form, no simulation:

1. chi-squared threshold against design false-alarm rate, for three windows;
2. detection power of the chi-squared test against fault size, at a fixed
   design level -- the operating curve an engineer actually needs;
3. CUSUM mean time between false alarms against threshold, for three design
   mean shifts, with the Siegmund and Wald expressions compared;
4. CUSUM mean run length after the change against design mean shift, at a
   fixed 2000-sample ARL0 -- the delay/false-alarm trade in one line.

    python examples/threshold_design.py
"""

from __future__ import annotations

import numpy as np
from _plotstyle import COLORS, save

import matplotlib.pyplot as plt  # noqa: E402  (backend set in _plotstyle)

from fdiscope import (  # noqa: E402
    chi2_detection_power,
    chi2_threshold,
    cusum_arl0_siegmund,
    cusum_delay_siegmund,
    cusum_delay_wald,
    cusum_threshold_for_arl0,
)

ALPHAS = np.logspace(-6, -0.5, 200)
WINDOWS = (10, 25, 100)
MUS = (0.5, 1.0, 2.0)
TARGET_ARL0 = 2000.0


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.4))

    ax = axes[0, 0]
    for window, colour in zip(WINDOWS, ("#1f4e79", "#4f81bd", "#9ecae1"), strict=True):
        thresholds = [chi2_threshold(a, 2 * window) for a in ALPHAS]
        ax.plot(ALPHAS, thresholds, color=colour, label=f"W = {window} (dof {2 * window})")
    ax.set_xscale("log")
    ax.set_xlabel("design false-alarm rate per window [-]")
    ax.set_ylabel("chi-squared threshold [-]")
    ax.set_title("1. threshold from the design level alone")
    ax.legend()

    ax = axes[0, 1]
    shifts = np.linspace(0.0, 2.5, 200)
    for window, colour in zip(WINDOWS, ("#1f4e79", "#4f81bd", "#9ecae1"), strict=True):
        dof = 2 * window
        h = chi2_threshold(1e-3, dof)
        power = [chi2_detection_power(h, dof, window * s * s) for s in shifts]
        ax.plot(shifts, power, color=colour, label=f"W = {window}")
    ax.axhline(0.9, color=COLORS["onset"], lw=0.8, ls=":")
    ax.set_xlabel("residual mean shift per sample [sigma]")
    ax.set_ylabel("detection probability per window [-]")
    ax.set_title("2. power at alpha = 1e-3 (non-central chi-squared)")
    ax.legend(loc="lower right")

    ax = axes[1, 0]
    hs = np.linspace(1.0, 12.0, 200)
    for mu, colour in zip(MUS, ("#2e7d32", "#c0504d", "#e07b00"), strict=True):
        ax.plot(hs, [cusum_arl0_siegmund(h, mu) for h in hs], color=colour, label=f"mu = {mu}")
    ax.axhline(TARGET_ARL0, color=COLORS["onset"], lw=0.8, ls=":")
    ax.set_yscale("log")
    ax.set_xlabel("CUSUM threshold h [-]")
    ax.set_ylabel("mean time between false alarms [samples]")
    ax.set_title("3. CUSUM false-alarm design (Siegmund)")
    ax.legend()

    ax = axes[1, 1]
    mus = np.linspace(0.3, 4.0, 200)
    thresholds = np.array([cusum_threshold_for_arl0(TARGET_ARL0, m) for m in mus])
    sieg = np.array([cusum_delay_siegmund(h, m) for h, m in zip(thresholds, mus, strict=True)])
    wald = np.array([cusum_delay_wald(h, m) for h, m in zip(thresholds, mus, strict=True)])
    ax.plot(mus, sieg, color=COLORS["cusum"], label="Siegmund, overshoot corrected")
    ax.plot(mus, wald, color=COLORS["analytic"], ls="--", label="Wald, h / K")
    ax.axvline(1.0 / 1.1652, color=COLORS["onset"], lw=0.8, ls=":")
    ax.set_yscale("log")
    ax.set_xlabel("design mean shift mu [sigma]")
    ax.set_ylabel("mean run length after the change [samples]")
    ax.set_title(f"4. delay at a fixed ARL0 = {TARGET_ARL0:.0f}")
    ax.legend()

    fig.tight_layout(h_pad=2.2, w_pad=2.0)
    path = save(fig, "threshold_design.png")
    print(f"saved {path}")
    print(f"chi2 threshold, W = 25, alpha = 1e-3 : {chi2_threshold(1e-3, 50):.4f}")
    print(f"chi2 threshold, W = 100, alpha = 1e-3: {chi2_threshold(1e-3, 200):.4f}")
    for mu in MUS:
        h = cusum_threshold_for_arl0(TARGET_ARL0, mu)
        print(
            f"CUSUM mu = {mu:>3}: h = {h:6.3f}, Wald {cusum_delay_wald(h, mu):8.3f}, "
            f"Siegmund {cusum_delay_siegmund(h, mu):8.3f} samples"
        )
    crossing = 1.0 / 1.1652
    print(
        f"Wald and Siegmund cross at mu = 1/1.1652 = {crossing:.4f}: below it h/K is "
        "pessimistic, above it optimistic."
    )


if __name__ == "__main__":
    main()
