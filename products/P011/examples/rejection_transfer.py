"""Closed-loop rejection and noise transfer functions, measured and analytic.

Produces ``screenshots/rejection_transfer.png``.

Run from products/P011:
    PYTHONPATH=src python examples/rejection_transfer.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from waveforge.control import (  # noqa: E402
    Integrator,
    noise_transfer,
    noise_variance_gain,
    rejection_transfer,
    stability_limit_gain,
)

OUTPUT = Path(__file__).resolve().parent.parent / "screenshots" / "rejection_transfer.png"
FRAME_RATE = 1000.0


def measure(frequency_hz: float, gain: float, delay: int, n_frames: int = 3000) -> float:
    integrator = Integrator(1, gain=gain, delay_frames=delay)
    command = 0.0
    residuals = np.empty(n_frames)
    for k in range(n_frames):
        error = np.sin(2.0 * np.pi * frequency_hz * k / FRAME_RATE) - command
        command = float(integrator.step(np.array([error]))[0])
        residuals[k] = error
    tail = residuals[n_frames // 2 :]
    phase = 2.0 * np.pi * frequency_hz * np.arange(n_frames // 2, n_frames) / FRAME_RATE
    return float(2.0 * abs(np.mean(tail * np.exp(-1j * phase))))


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    frequencies = np.logspace(0, np.log10(FRAME_RATE / 2 - 1), 300)
    probes = np.array([3.0, 11.0, 37.0, 97.0, 211.0, 379.0])

    for gain, colour in ((0.15, "tab:blue"), (0.35, "tab:orange"), (0.70, "tab:red")):
        axes[0].loglog(
            frequencies, np.abs(rejection_transfer(frequencies, FRAME_RATE, gain, 2)),
            "-", color=colour, label=f"analytic, g = {gain}",
        )
        axes[0].loglog(
            probes, [measure(float(f), gain, 2) for f in probes], "o", ms=6,
            mfc="none", color=colour, label=f"time domain, g = {gain}",
        )
    axes[0].axhline(1.0, color="k", ls=":", lw=1)
    axes[0].set_xlabel("frequency [Hz]")
    axes[0].set_ylabel(r"$|E/\Phi|$")
    axes[0].set_title("Error rejection, 2-frame latency\n(1 kHz frame rate)")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=7.5, ncol=1)

    for delay, colour in ((1, "tab:green"), (2, "tab:blue"), (3, "tab:purple"), (4, "tab:brown")):
        gain = 0.5 * stability_limit_gain(delay)
        axes[1].loglog(
            frequencies, np.abs(rejection_transfer(frequencies, FRAME_RATE, gain, delay)),
            "-", color=colour, label=f"d = {delay}, g = {gain:.2f}",
        )
    axes[1].axhline(1.0, color="k", ls=":", lw=1)
    axes[1].set_xlabel("frequency [Hz]")
    axes[1].set_ylabel(r"$|E/\Phi|$")
    axes[1].set_title("Latency costs bandwidth\n(each gain at half its stability limit)")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=8)

    for delay, colour in ((1, "tab:green"), (2, "tab:blue"), (3, "tab:purple")):
        limit = stability_limit_gain(delay)
        gains = np.linspace(0.02, 0.98 * limit, 60)
        axes[2].plot(
            gains, [noise_variance_gain(float(g), delay) for g in gains],
            "-", color=colour, label=f"d = {delay} (limit {limit:.3f})",
        )
        axes[2].axvline(limit, color=colour, ls=":", lw=1)
    axes[2].plot(
        np.linspace(0.02, 1.9, 60), np.linspace(0.02, 1.9, 60) / (2 - np.linspace(0.02, 1.9, 60)),
        "k--", lw=1.2, label=r"classical $g/(2-g)$",
    )
    axes[2].set_xlabel("loop gain")
    axes[2].set_ylabel("noise variance amplification")
    axes[2].set_title("Noise propagation and stability limits")
    axes[2].set_ylim(0.0, 3.0)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=8)

    inset = axes[1].inset_axes((0.08, 0.08, 0.38, 0.32))
    inset.semilogy(
        frequencies, np.abs(noise_transfer(frequencies, FRAME_RATE, 0.35, 2)), "k-", lw=1.2
    )
    inset.set_title(r"$|E/N|$, g=0.35", fontsize=7)
    inset.tick_params(labelsize=6)
    inset.set_xscale("log")
    inset.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        "WaveForge — closed-loop transfer functions (research-grade, not flight-qualified)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
