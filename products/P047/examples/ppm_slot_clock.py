"""M-ary PPM slot timing: the slot S-curve, the jitter chain, and the slot error floor.

Writes ``../screenshots/ppm_slot_clock.png``.  Runtime about 40 s.

Top left: the slot S-curve against the closed form sin(2 pi eps), with the residual.
Top right: jitter in slot periods, predicted and measured, against the loop
bandwidth per update and per slot.  Bottom left: the measured order dependence of
the detector output variance and the slot error rate, against the 10 log10(M/2)
rule of thumb.  Bottom right: the slot-index error rate rising while the loop
reports no slips at all - the failure mode that binary symbol timing has no
counterpart for.
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
    jitter_variance_coloured,
)
from slotsync.ppm import (  # noqa: E402
    PpmConfig,
    duty_cycle_penalty_db,
    equivalent_slot_bandwidth,
    measure_ppm_slot_statistics,
    ppm_slot_autocovariance,
    ppm_slot_scurve,
    run_ppm_slot_loop,
)
from slotsync.pulses import half_sine  # noqa: E402

SLOT = half_sine(1.0)
ZETA = 1.0 / math.sqrt(2.0)
SHARP = np.linspace(-0.001, 0.001, 21)
ORDERS = (2, 4, 8, 16)
BANDWIDTHS = (0.002, 0.005, 0.01, 0.02)


def main() -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 9.0))

    offsets = np.linspace(-0.5, 0.5, 401)
    curve = ppm_slot_scurve(PpmConfig(4, 0.25, True), SLOT, offsets=offsets)
    inside = np.abs(offsets) <= 0.25
    panel = axes[0, 0]
    panel.plot(offsets, curve.values, color="#1f4e79", linewidth=2.0, label="measured S(eps)")
    panel.plot(
        offsets[inside],
        np.sin(2.0 * np.pi * offsets[inside]),
        "--",
        color="#c62828",
        linewidth=1.2,
        label="sin(2 pi eps), |eps| <= 1/4",
    )
    sharp = ppm_slot_scurve(PpmConfig(4, 0.25, True), SLOT, offsets=SHARP, fit_halfwidth=0.001)
    residual = float(
        np.max(np.abs(curve.values[inside] - np.sin(2.0 * np.pi * offsets[inside])))
    )
    panel.axvline(0.25, color="#ef6c00", linestyle=":", linewidth=1.2)
    panel.annotate("gates leave the slot\nat 1/4", xy=(0.26, 0.3), fontsize=8, color="#ef6c00")
    panel.set_xlabel("slot timing offset (slot periods)")
    panel.set_ylabel("mean detector output")
    panel.set_title(
        f"4-PPM slot S-curve: K_d = {sharp.gain_central_difference:.6f} against "
        f"2 pi = {2 * math.pi:.6f}\nmax residual against the closed form {residual:.2e}",
        fontsize=9,
    )
    panel.legend(fontsize=8)
    panel.grid(alpha=0.25)

    config = PpmConfig(4, 0.25, False)
    gain = ppm_slot_scurve(
        config, SLOT, offsets=SHARP, fit_halfwidth=0.001
    ).gain_central_difference
    autocovariance = ppm_slot_autocovariance(
        config, SLOT, sample_snr_db=20.0, max_lag=8, symbols=30000, seed=7
    )
    white = []
    coloured = []
    measured = []
    errors = []
    for bandwidth in BANDWIDTHS:
        design = LoopDesign.from_bandwidth(bandwidth, ZETA, gain)
        run = run_ppm_slot_loop(
            config, SLOT, design, n_symbols=60000, sample_snr_db=20.0, seed=20261006
        )
        white.append(jitter_variance_closed_form(bandwidth, gain, autocovariance[0]))
        coloured.append(jitter_variance_coloured(design, autocovariance))
        measured.append(run.jitter_variance)
        errors.append(run.jitter_variance_standard_error)
    panel = axes[0, 1]
    panel.loglog(BANDWIDTHS, white, "s--", color="#9e9e9e", label="white noise (closed form)")
    panel.loglog(BANDWIDTHS, coloured, "o-", color="#1f4e79", label="measured noise spectrum")
    panel.errorbar(
        BANDWIDTHS, measured, yerr=errors, fmt="k^", markersize=5, capsize=3, linewidth=1.0,
        label="closed-loop Monte Carlo",
    )
    secondary = panel.secondary_xaxis(
        "top", functions=(lambda x: x / 4.0, lambda x: x * 4.0)
    )
    secondary.set_xlabel("B_n per slot (4-PPM: one update per 4 slots)", fontsize=8)
    panel.set_xlabel("B_n per loop update (cycles per symbol)")
    panel.set_ylabel("slot jitter variance (slot^2)")
    panel.set_title("the jitter chain carries over, in slot periods", fontsize=9)
    panel.legend(fontsize=8)
    panel.grid(alpha=0.25, which="both")

    panel = axes[1, 0]
    for snr, colour in ((20.0, "#1565c0"), (6.0, "#c62828")):
        penalties = []
        errors_rate = []
        for order in ORDERS:
            configuration = PpmConfig(order, 0.25, False)
            penalties.append(
                duty_cycle_penalty_db(
                    configuration,
                    SLOT,
                    sample_snr_db=snr,
                    reference_gain=2.0 * math.pi,
                    symbols=6000,
                    seed=3,
                )
            )
            errors_rate.append(
                measure_ppm_slot_statistics(
                    configuration, SLOT, sample_snr_db=snr, symbols=6000, seed=3
                )["slot_error_rate"]
            )
        relative = [value - penalties[0] for value in penalties]
        panel.plot(ORDERS, relative, "o-", color=colour, label=f"measured, {snr:.0f} dB")
        print(f"  {snr:.0f} dB slot error rates by order: {errors_rate}")
    panel.plot(
        ORDERS,
        [10.0 * math.log10(order / 2) for order in ORDERS],
        "k--",
        linewidth=1.2,
        label="10 log10(M/2) rule of thumb",
    )
    panel.set_xscale("log", base=2)
    panel.set_xticks(ORDERS)
    panel.set_xticklabels([str(order) for order in ORDERS])
    panel.set_xlabel("PPM order M")
    panel.set_ylabel("loop SNR cost relative to M = 2 (dB)")
    panel.set_title("the duty-cycle penalty is not 10 log10(M/2)", fontsize=9)
    panel.legend(fontsize=8)
    panel.grid(alpha=0.25, which="both")

    panel = axes[1, 1]
    config8 = PpmConfig(8, 0.25, False)
    gain8 = ppm_slot_scurve(
        config8, SLOT, offsets=SHARP, fit_halfwidth=0.001
    ).gain_central_difference
    design8 = LoopDesign.from_bandwidth(0.01, ZETA, gain8)
    snrs = (20.0, 14.0, 10.0, 6.0, 2.0)
    rates = []
    jitters = []
    slips = []
    for snr in snrs:
        run = run_ppm_slot_loop(
            config8, SLOT, design8, n_symbols=30000, sample_snr_db=snr, seed=20261006
        )
        rates.append(max(run.slot_error_rate, 1e-5))
        jitters.append(run.jitter_rms)
        slips.append(run.slip_count)
    panel.semilogy(snrs, rates, "o-", color="#c62828", label="slot-index error rate")
    panel.semilogy(snrs, jitters, "s-", color="#1f4e79", label="rms slot jitter (slot periods)")
    panel.semilogy(
        snrs, [max(value, 1e-5) for value in slips], "^--", color="0.4", label="loop slips counted"
    )
    panel.set_xlabel("per-sample SNR (dB)")
    panel.set_ylabel("rate / count (log scale)")
    panel.set_title(
        "8-PPM: the symbol error floor arrives long before any loop slip", fontsize=9
    )
    panel.legend(fontsize=8)
    panel.grid(alpha=0.25, which="both")

    figure.suptitle(
        "PPM slot timing: the same chain in slot periods, with a different failure mode",
        fontsize=12,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    figure.savefig("../screenshots/ppm_slot_clock.png", dpi=130)
    plt.close(figure)
    print("wrote ../screenshots/ppm_slot_clock.png")
    print(f"4-PPM slot gain {gain:.9f}, closed-form 2 pi = {2 * math.pi:.9f}")
    for bandwidth, w, c, m in zip(BANDWIDTHS, white, coloured, measured, strict=True):
        print(
            f"  B_n/update {bandwidth:7.4f}  B_n/slot "
            f"{equivalent_slot_bandwidth(bandwidth, 4):8.5f}  white {w:.4e}  "
            f"coloured {c:.4e}  measured {m:.4e}  mc/col {m / c:.4f}"
        )
    print(f"8-PPM slot error rates by SNR {snrs}: {rates}")


if __name__ == "__main__":
    main()
