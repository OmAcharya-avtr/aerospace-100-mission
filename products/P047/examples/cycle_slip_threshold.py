"""Where the loop loses lock, and how far the classical slip estimate is from it.

Writes ``../screenshots/cycle_slip_threshold.png``.  Runtime about 60 s.

Left: measured slips per symbol against loop SNR for four loop bandwidths, with
the Gaussian level-crossing estimate on the same axes.  Middle: the rms timing
error against loop SNR, with the fully unlocked value 1/sqrt(12) marked - the
transition is a cliff, not a slope.  Right: the mechanism, measured - the S-curve's
local gain falls while the detector output noise rises, so the escape is a
runaway and not a crossing of a fixed barrier.
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from slotsync.loop import (  # noqa: E402
    LoopDesign,
    cycle_slip_rate_rice,
    jitter_variance_coloured,
    loop_snr_db,
)
from slotsync.pulses import nyquist_raised_cosine  # noqa: E402
from slotsync.scurve import scurve  # noqa: E402
from slotsync.simulate import (  # noqa: E402
    measure_ted_autocovariance,
    measure_ted_statistics,
    run_timing_loop,
)
from slotsync.ted import TedConfig  # noqa: E402

PULSE = nyquist_raised_cosine(0.5, 4.0)
ZETA = 1.0 / math.sqrt(2.0)
CONFIG = TedConfig("mueller-muller", "antipodal")
SNRS = (12.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0)
BANDWIDTHS = (0.005, 0.01, 0.02, 0.05)
SYMBOLS = 150000
UNIFORM_RMS = 1.0 / math.sqrt(12.0)


def main() -> None:
    gain = scurve(
        CONFIG, PULSE, np.linspace(-0.02, 0.02, 41), max_exact_symbols=16
    ).gain_central_difference
    full = scurve(CONFIG, PULSE, max_exact_symbols=16)
    boundary = full.reversal_offset or 0.5

    figure, axes = plt.subplots(1, 3, figsize=(15.0, 5.0))
    for bandwidth in BANDWIDTHS:
        design = LoopDesign.from_bandwidth(bandwidth, ZETA, gain)
        loop_snrs = []
        measured = []
        estimates = []
        rms = []
        for snr in SNRS:
            autocovariance = measure_ted_autocovariance(
                CONFIG, PULSE, sample_snr_db=snr, max_lag=24, samples=150000
            )
            variance = jitter_variance_coloured(design, autocovariance)
            run = run_timing_loop(
                CONFIG,
                PULSE,
                design,
                n_symbols=SYMBOLS,
                sample_snr_db=snr,
                divergence_limit=1.0e6,
            )
            loop_snrs.append(loop_snr_db(variance, boundary))
            measured.append(max(run.slip_rate_per_symbol, 1.0 / SYMBOLS / 10.0))
            estimates.append(max(cycle_slip_rate_rice(design, autocovariance[0], boundary), 1e-40))
            rms.append(run.jitter_rms)
        axes[0].semilogy(
            loop_snrs, measured, "o-", markersize=4, label=f"measured, B_n={bandwidth}"
        )
        axes[0].semilogy(
            loop_snrs,
            estimates,
            "s--",
            markersize=3,
            color=axes[0].lines[-1].get_color(),
            alpha=0.55,
            label=f"Rice estimate, B_n={bandwidth}",
        )
        axes[1].plot(loop_snrs, rms, "o-", markersize=4, label=f"B_n = {bandwidth}")

    axes[0].axhline(1.0 / SYMBOLS, color="0.4", linestyle=":", linewidth=1.0)
    axes[0].annotate(
        f"one slip in {SYMBOLS} symbols",
        xy=(axes[0].get_xlim()[0], 1.0 / SYMBOLS),
        fontsize=7.5,
        color="0.3",
        va="bottom",
    )
    axes[0].set_ylim(1e-40, 10.0)
    axes[0].set_xlabel("loop SNR (dB), from the predicted jitter")
    axes[0].set_ylabel("slips per symbol")
    axes[0].set_title("measured slip rate against the Gaussian estimate", fontsize=10)
    axes[0].legend(fontsize=6.5, ncol=2)
    axes[0].grid(alpha=0.25, which="both")

    axes[1].axhline(UNIFORM_RMS, color="#c62828", linestyle="--", linewidth=1.2)
    axes[1].annotate(
        "fully unlocked, 1/sqrt(12)",
        xy=(axes[1].get_xlim()[0], UNIFORM_RMS),
        fontsize=8,
        color="#c62828",
        va="bottom",
    )
    axes[1].set_xlabel("loop SNR (dB), from the predicted jitter")
    axes[1].set_ylabel("measured rms timing error (symbol periods)")
    axes[1].set_title("the loss of lock is a cliff", fontsize=10)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)

    offsets = (0.0, 0.1, 0.2, 0.3, 0.4, 0.45)
    local_gains = []
    variances = []
    for offset in offsets:
        index = int(np.argmin(np.abs(full.offsets - offset)))
        local_gains.append(
            float(
                (full.values[index + 1] - full.values[index - 1])
                / (full.offsets[index + 1] - full.offsets[index - 1])
            )
        )
        variances.append(
            measure_ted_statistics(
                CONFIG, PULSE, sample_snr_db=8.0, offset=offset, samples=200000
            ).variance
        )
    reference = variances[0]
    axes[2].plot(
        offsets,
        [value / local_gains[0] for value in local_gains],
        "o-",
        color="#1f4e79",
        label="local S-curve gain, normalised",
    )
    axes[2].plot(
        offsets,
        [value / reference for value in variances],
        "s-",
        color="#c62828",
        label="detector output variance, normalised",
    )
    axes[2].set_xlabel("timing offset held (symbol periods)")
    axes[2].set_ylabel("relative to the value at zero offset")
    axes[2].set_title("why the escape runs away, at 8 dB per-sample SNR", fontsize=10)
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.25)

    figure.suptitle(
        "Cycle slips: the measured threshold is usable, the classical level-crossing "
        "estimate is not",
        fontsize=12,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    figure.savefig("../screenshots/cycle_slip_threshold.png", dpi=130)
    plt.close(figure)
    print("wrote ../screenshots/cycle_slip_threshold.png")
    print(f"K_d = {gain:.6f}, S-curve reversal at {boundary:.4f} symbol")
    print("local gain / variance against held offset, at 8 dB per-sample SNR:")
    for offset, local, variance in zip(offsets, local_gains, variances, strict=True):
        print(
            f"  offset {offset:4.2f}  local gain {local:8.5f}  sigma_n^2 {variance:8.5f}  "
            f"ratio to zero {variance / reference:6.4f}"
        )


if __name__ == "__main__":
    main()
