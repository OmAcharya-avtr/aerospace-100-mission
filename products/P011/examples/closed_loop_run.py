"""A closed-loop run: residual phase, Strehl history, and failure modes.

Produces ``screenshots/closed_loop_run.png``.

Run from products/P011:
    PYTHONPATH=src python examples/closed_loop_run.py
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from waveforge.control import stability_limit_gain  # noqa: E402
from waveforge.loop import AOConfig, AOSystem  # noqa: E402
from waveforge.pupil import piston_removed  # noqa: E402

OUTPUT = Path(__file__).resolve().parent.parent / "screenshots" / "closed_loop_run.png"
BASE = AOConfig(seed=7)
N_FRAMES = 500


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.4))
    system = AOSystem(BASE)

    result = system.run(N_FRAMES, warmup_frames=120, gain=0.4)
    frames = np.arange(N_FRAMES)
    axes[0, 0].semilogy(frames, result.open_loop_variance, color="0.6", lw=0.9, label="open loop")
    axes[0, 0].semilogy(frames, result.residual_variance, color="tab:blue", lw=1.0,
                        label="closed loop, g = 0.4")
    axes[0, 0].axvline(120, color="k", ls=":", lw=1)
    axes[0, 0].annotate("warm-up", (122, result.open_loop_variance.max() * 0.5), fontsize=7)
    axes[0, 0].set_xlabel("frame")
    axes[0, 0].set_ylabel(r"phase variance [rad$^2$]")
    axes[0, 0].set_title(
        f"Residual vs open loop  ({result.rejection_db:.1f} dB rejection, "
        f"mean Strehl {result.mean_strehl:.3f})"
    )
    axes[0, 0].grid(True, which="both", alpha=0.3)
    axes[0, 0].legend(fontsize=8)

    phase = system.atmosphere.frame(300)
    commands = system.mirror.fit(phase)
    residual = piston_removed(phase - system.mirror.surface(commands), system.pupil.mask)
    masked_phase = np.where(system.pupil.mask, phase, np.nan)
    masked_residual = np.where(system.pupil.mask, residual, np.nan)
    limit = float(np.nanmax(np.abs(masked_phase)))
    image = axes[0, 1].imshow(masked_phase, origin="lower", cmap="RdBu_r", vmin=-limit, vmax=limit)
    fig.colorbar(image, ax=axes[0, 1], label="phase [rad]", fraction=0.046)
    axes[0, 1].set_title("Incoming wavefront, frame 300")
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])
    inset = axes[0, 1].inset_axes((0.62, 0.62, 0.36, 0.36))
    inset.imshow(masked_residual, origin="lower", cmap="RdBu_r", vmin=-limit, vmax=limit)
    inset.set_title("best DM fit residual", fontsize=7)
    inset.set_xticks([])
    inset.set_yticks([])

    gains = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    for flux, colour in ((float("inf"), "tab:blue"), (1000.0, "tab:orange"), (300.0, "tab:red")):
        noisy = AOSystem(replace(BASE, photon_flux=flux, read_noise_e=1.0))
        values = [
            noisy.run(300, warmup_frames=100, gain=float(g)).mean_residual_variance
            for g in gains
        ]
        label = "noiseless" if np.isinf(flux) else f"{flux:.0f} e$^-$/subap"
        axes[1, 0].semilogy(gains, values, "-o", ms=4, color=colour, label=label)
    axes[1, 0].axvline(stability_limit_gain(2), color="k", ls="--", lw=1.2)
    axes[1, 0].annotate("stability limit (d = 2)", (0.83, 6.0), fontsize=7, rotation=90)
    axes[1, 0].set_xlabel("loop gain")
    axes[1, 0].set_ylabel(r"mean residual variance [rad$^2$]")
    axes[1, 0].set_title("Optimal gain moves down as photon noise rises")
    axes[1, 0].grid(True, which="both", alpha=0.3)
    axes[1, 0].legend(fontsize=8)

    strokes = (0.05, 0.1, 0.2, 0.4, float("inf"))
    residuals, saturation = [], []
    for stroke in strokes:
        limited = AOSystem(replace(BASE, stroke_rad=stroke))
        run = limited.run(300, warmup_frames=100, gain=0.4)
        residuals.append(run.mean_residual_variance)
        saturation.append(run.max_saturated_fraction)
    labels = [("inf" if np.isinf(s) else f"{s:g}") for s in strokes]
    positions = np.arange(len(strokes))
    axes[1, 1].bar(positions, residuals, color="tab:blue", alpha=0.85)
    axes[1, 1].set_xticks(positions)
    axes[1, 1].set_xticklabels(labels)
    axes[1, 1].set_xlabel("actuator stroke limit [rad of phase]")
    axes[1, 1].set_ylabel(r"mean residual variance [rad$^2$]")
    axes[1, 1].set_title("Failure mode: actuator saturation")
    axes[1, 1].grid(True, axis="y", alpha=0.3)
    twin = axes[1, 1].twinx()
    twin.plot(positions, np.array(saturation) * 100, "r-s", ms=5)
    twin.set_ylabel("peak actuators saturated [%]", color="r")
    twin.tick_params(axis="y", colors="r")

    fig.suptitle(
        "WaveForge — closed-loop simulation (research-grade, not flight-qualified)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
