"""Loop design: bandwidth, damping, the closed-loop response and what it costs in jitter.

Writes ``../screenshots/loop_design.png``.  Runtime about 6 s.

Panel 1: the closed-loop magnitude response for several damping factors, with the
area under ``|H|^2`` that defines the noise bandwidth.  Panel 2: the closed-form
noise bandwidth against numerical quadrature.  Panel 3: the exact discrete jitter
variance against the classical small-bandwidth expression - the ratio is the cost
of the approximation.  Panel 4: an acquisition transient at three bandwidths.
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from slotsync.loop import (  # noqa: E402
    LoopDesign,
    jitter_variance_closed_form,
    jitter_variance_exact,
    noise_bandwidth_closed_form,
    noise_bandwidth_numeric,
)
from slotsync.pulses import nyquist_raised_cosine  # noqa: E402
from slotsync.scurve import scurve  # noqa: E402
from slotsync.simulate import run_timing_loop  # noqa: E402
from slotsync.ted import TedConfig  # noqa: E402

DAMPINGS = (0.3, 0.5, 1.0 / math.sqrt(2.0), 1.0, 2.0)
BANDWIDTHS = (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1)
PULSE = nyquist_raised_cosine(0.5, 4.0)


def response(theta: float, zeta: float, frequencies: np.ndarray) -> np.ndarray:
    omega = 2.0 * np.pi * frequencies
    numerator = 2.0 * zeta * theta * 1j * omega + theta**2
    denominator = (1j * omega) ** 2 + 2.0 * zeta * theta * 1j * omega + theta**2
    return np.abs(numerator / denominator)


def main() -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 9.0))

    theta = 0.05
    frequencies = np.logspace(-3.5, -0.3, 500)
    panel = axes[0, 0]
    for zeta in DAMPINGS:
        magnitude = response(theta, zeta, frequencies)
        panel.loglog(frequencies, magnitude, linewidth=1.6, label=f"zeta = {zeta:.3f}")
        panel.axvline(
            noise_bandwidth_closed_form(theta, zeta),
            color=panel.lines[-1].get_color(),
            linestyle=":",
            linewidth=1.0,
        )
    panel.set_xlabel("frequency (cycles per symbol)")
    panel.set_ylabel("|H|")
    panel.set_title(f"closed-loop response, theta = {theta}; dotted lines are B_n", fontsize=10)
    panel.legend(fontsize=8)
    panel.grid(alpha=0.25, which="both")

    panel = axes[0, 1]
    for zeta in DAMPINGS:
        thetas = np.logspace(-3, -0.3, 25)
        closed = [noise_bandwidth_closed_form(float(t), zeta) for t in thetas]
        numeric = [noise_bandwidth_numeric(float(t), zeta) for t in thetas]
        panel.semilogx(
            thetas,
            [abs(c - n) / n for c, n in zip(closed, numeric, strict=True)],
            "o-",
            markersize=3,
            linewidth=1.2,
            label=f"zeta = {zeta:.3f}",
        )
    panel.set_yscale("log")
    panel.set_xlabel("theta = omega_n T (rad per symbol)")
    panel.set_ylabel("|closed form - quadrature| / quadrature")
    panel.set_title("B_n closed form against numerical quadrature", fontsize=10)
    panel.legend(fontsize=8)
    panel.grid(alpha=0.25, which="both")

    panel = axes[1, 0]
    for zeta in (0.5, 1.0 / math.sqrt(2.0), 1.0):
        ratios = []
        for bandwidth in BANDWIDTHS:
            design = LoopDesign.from_bandwidth(bandwidth, zeta, 1.5)
            ratios.append(
                jitter_variance_exact(design, 0.3)
                / jitter_variance_closed_form(bandwidth, 1.5, 0.3)
            )
        panel.semilogx(BANDWIDTHS, ratios, "o-", markersize=4, label=f"zeta = {zeta:.3f}")
    panel.axhline(1.0, color="0.5", linestyle="--", linewidth=1.0)
    panel.set_xlabel("B_n (cycles per symbol)")
    panel.set_ylabel("exact discrete / classical closed form")
    panel.set_title("cost of the small-bandwidth approximation", fontsize=10)
    panel.legend(fontsize=8)
    panel.grid(alpha=0.25, which="both")

    panel = axes[1, 1]
    config = TedConfig("mueller-muller", "antipodal")
    gain = scurve(
        config, PULSE, np.linspace(-0.02, 0.02, 41), max_exact_symbols=16
    ).gain_central_difference
    for bandwidth in (0.002, 0.01, 0.05):
        design = LoopDesign.from_bandwidth(bandwidth, 1.0 / math.sqrt(2.0), gain)
        run = run_timing_loop(
            config,
            PULSE,
            design,
            n_symbols=6000,
            sample_snr_db=30.0,
            true_offset=0.0,
            initial_error=0.3,
            discard=0,
        )
        panel.plot(
            np.arange(run.error.size),
            run.error,
            linewidth=1.0,
            label=f"B_n = {bandwidth}, rms {run.jitter_rms:.4f}",
        )
    panel.axhline(0.0, color="0.5", linestyle="--", linewidth=1.0)
    panel.set_xlim(0, 3000)
    panel.set_xlabel("symbol")
    panel.set_ylabel("timing error (symbol periods)")
    panel.set_title(
        "acquisition from 0.3 symbol, Mueller-Mueller at 30 dB per-sample SNR", fontsize=10
    )
    panel.legend(fontsize=8)
    panel.grid(alpha=0.25)

    figure.suptitle("Second-order timing loop: design, verification and the acquisition it buys")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    figure.savefig("../screenshots/loop_design.png", dpi=130)
    plt.close(figure)
    print("wrote ../screenshots/loop_design.png")
    print(f"K_d used for the acquisition panel: {gain:.6f}")
    for bandwidth in BANDWIDTHS:
        design = LoopDesign.from_bandwidth(bandwidth, 1.0 / math.sqrt(2.0), gain)
        ratio = jitter_variance_exact(design, 0.3) / jitter_variance_closed_form(
            bandwidth, gain, 0.3
        )
        print(
            f"  B_n = {bandwidth:7.4f}  theta = {design.natural_frequency:8.5f}  "
            f"k1 = {design.k_proportional:9.6f}  k2 = {design.k_integral:9.3e}  "
            f"exact/closed = {ratio:.6f}"
        )


if __name__ == "__main__":
    main()
