"""Sizing an AO system: the error budget and where the Strehl goes.

Produces ``screenshots/error_budget_sizing.png``.

Run from products/P011:
    PYTHONPATH=src python examples/error_budget_sizing.py
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from waveforge.errorbudget import strehl_marechal  # noqa: E402
from waveforge.loop import AOConfig, AOSystem  # noqa: E402

OUTPUT = Path(__file__).resolve().parent.parent / "screenshots" / "error_budget_sizing.png"
BASE = AOConfig(screen_pixels=256, n_subharmonics=0, photon_flux=2000.0, read_noise_e=1.0)


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    actuators = (5, 7, 9, 13, 17)
    fitting, temporal, noise, total = [], [], [], []
    pitches = []
    for n_act in actuators:
        system = AOSystem(replace(BASE, n_act=n_act, n_sub=8))
        budget = system.error_budget()
        fitting.append(budget.fitting)
        temporal.append(budget.temporal)
        noise.append(budget.noise)
        total.append(budget.total)
        pitches.append(system.mirror.pitch_m * 1e3)
    axes[0].stackplot(
        actuators,
        fitting,
        temporal,
        noise,
        labels=["fitting", "temporal", "noise"],
        colors=["tab:blue", "tab:orange", "tab:green"],
        alpha=0.85,
    )
    axes[0].plot(actuators, total, "k-o", ms=4, label="total")
    axes[0].set_xlabel("actuators across the pupil")
    axes[0].set_ylabel(r"residual variance [rad$^2$]")
    axes[0].set_title(
        f"Error budget vs actuator count\nD/$r_0$ = {BASE.d_over_r0:.0f}, "
        f"{BASE.frame_rate_hz:.0f} Hz, {BASE.photon_flux:.0f} e$^-$/subap"
    )
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    for x, pitch in zip(actuators, pitches, strict=True):
        axes[0].annotate(f"{pitch:.0f} mm", (x, 0.02), fontsize=6.5, ha="center")

    rates = np.array([250.0, 500.0, 1000.0, 2000.0, 4000.0])
    for n_act, colour in ((7, "tab:red"), (9, "tab:blue"), (13, "tab:green")):
        totals = []
        for rate in rates:
            system = AOSystem(replace(BASE, n_act=n_act, frame_rate_hz=float(rate)))
            totals.append(system.error_budget().total)
        axes[1].semilogx(rates, strehl_marechal(np.array(totals)), "-o", ms=4,
                         color=colour, label=f"{n_act} actuators across")
    axes[1].set_xlabel("WFS frame rate [Hz]")
    axes[1].set_ylabel("Strehl (extended Marechal)")
    axes[1].set_title("Analytic Strehl vs loop speed\n(photon flux held constant per frame)")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=8)
    axes[1].set_ylim(0.0, 1.0)

    r0_values = np.array([0.20, 0.15, 0.12, 0.10, 0.08, 0.06])
    analytic, simulated = [], []
    for r0 in r0_values:
        config = replace(BASE, r0_m=float(r0), screen_pixels=1024, n_subharmonics=3)
        system = AOSystem(config)
        analytic.append(float(strehl_marechal(system.error_budget().total)))
        simulated.append(system.run(300, warmup_frames=100).mean_strehl)
    axes[2].plot(BASE.diameter_m / r0_values, analytic, "k-o", ms=4, label="analytic budget")
    axes[2].plot(
        BASE.diameter_m / r0_values, simulated, "s--", ms=5, color="tab:purple",
        label="simulated closed loop",
    )
    axes[2].set_xlabel(r"$D/r_0$")
    axes[2].set_ylabel("Strehl ratio")
    axes[2].set_title("Budget vs end-to-end simulation\n(9x9 DM, 8x8 SH, gain 0.4, 2-frame lag)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=8)
    axes[2].set_ylim(0.0, 1.0)

    fig.suptitle(
        "WaveForge — adaptive-optics sizing (research-grade, not flight-qualified)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
