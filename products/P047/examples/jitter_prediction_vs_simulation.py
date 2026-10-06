"""The product's central figure: three jitter predictions against the simulated loop.

Writes ``../screenshots/jitter_prediction_vs_simulation.png``.  Runtime about 70 s.

Top row, one panel per detector: predicted and measured jitter variance against
loop bandwidth, with Monte Carlo error bars.  Bottom left: the detector output
noise split into self-noise and channel noise, which is what decides whether the
white-noise prediction works.  Bottom middle: the measured autocovariance of the
detector output, which is why it does not.  Bottom right: the measured-to-predicted
ratio against bandwidth for all three detectors, i.e. the whole claim on one axis.
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
SNR_DB = 20.0
BANDWIDTHS = (0.001, 0.002, 0.005, 0.01, 0.02)
CONFIGS = (
    TedConfig("early-late", "antipodal", 0.25, "dd"),
    TedConfig("gardner", "antipodal"),
    TedConfig("mueller-muller", "antipodal"),
)
COLOURS = {"early-late": "#1f4e79", "gardner": "#c62828", "mueller-muller": "#2e7d32"}


def main() -> None:
    figure = plt.figure(figsize=(13.5, 9.0))
    grid = figure.add_gridspec(2, 3, hspace=0.33, wspace=0.28)

    ratios: dict[str, list[float]] = {}
    statistics = {}
    autocovariances = {}
    for column, config in enumerate(CONFIGS):
        gain = scurve(
            config, PULSE, np.linspace(-0.02, 0.02, 41), max_exact_symbols=16
        ).gain_central_difference
        stats = measure_ted_statistics(config, PULSE, sample_snr_db=SNR_DB, samples=400000)
        autocovariance = measure_ted_autocovariance(
            config, PULSE, sample_snr_db=SNR_DB, max_lag=24, samples=400000
        )
        statistics[config.detector] = stats
        autocovariances[config.detector] = autocovariance

        white = []
        coloured = []
        measured = []
        errors = []
        for bandwidth in BANDWIDTHS:
            design = LoopDesign.from_bandwidth(bandwidth, ZETA, gain)
            run = run_timing_loop(
                config, PULSE, design, n_symbols=200000, sample_snr_db=SNR_DB
            )
            white.append(jitter_variance_closed_form(bandwidth, gain, stats.variance))
            coloured.append(jitter_variance_coloured(design, autocovariance))
            measured.append(run.jitter_variance)
            errors.append(run.jitter_variance_standard_error)
        ratios[config.detector] = [m / c for m, c in zip(measured, coloured, strict=True)]

        panel = figure.add_subplot(grid[0, column])
        panel.loglog(BANDWIDTHS, white, "s--", color="#9e9e9e", label="white noise (closed form)")
        panel.loglog(BANDWIDTHS, coloured, "o-", color=COLOURS[config.detector],
                     label="measured noise spectrum")
        panel.errorbar(
            BANDWIDTHS,
            measured,
            yerr=errors,
            fmt="k^",
            markersize=5,
            capsize=3,
            linewidth=1.0,
            label="closed-loop Monte Carlo",
        )
        panel.set_xlabel("B_n (cycles per symbol)")
        if column == 0:
            panel.set_ylabel("timing jitter variance (symbol^2)")
        panel.set_title(
            f"{config.label}\nK_d = {gain:.4f}, sigma_n^2 = {stats.variance:.4f}", fontsize=9
        )
        panel.legend(fontsize=7.5, loc="upper left")
        panel.grid(alpha=0.25, which="both")

    panel = figure.add_subplot(grid[1, 0])
    names = [config.detector for config in CONFIGS]
    self_noise = [statistics[name].self_noise_variance for name in names]
    channel = [statistics[name].channel_noise_variance for name in names]
    positions = np.arange(len(names))
    panel.bar(positions, self_noise, color="#ef6c00", label="self-noise")
    panel.bar(positions, channel, bottom=self_noise, color="#1565c0", label="channel noise")
    panel.set_xticks(positions)
    panel.set_xticklabels(names, fontsize=8)
    panel.set_ylabel("detector output variance")
    panel.set_title(f"where sigma_n^2 comes from, at {SNR_DB:.0f} dB per-sample SNR", fontsize=9)
    panel.legend(fontsize=8)
    panel.grid(alpha=0.25, axis="y")

    panel = figure.add_subplot(grid[1, 1])
    for name in names:
        autocovariance = autocovariances[name]
        panel.plot(
            np.arange(9),
            autocovariance[:9] / autocovariance[0],
            "o-",
            markersize=4,
            color=COLOURS[name],
            label=name,
        )
    panel.axhline(0.0, color="0.5", linewidth=0.9)
    panel.set_xlabel("lag (symbols)")
    panel.set_ylabel("R[j] / R[0]")
    panel.set_title("the detector noise is not white, except for Mueller-Mueller", fontsize=9)
    panel.legend(fontsize=8)
    panel.grid(alpha=0.25)

    panel = figure.add_subplot(grid[1, 2])
    for name in names:
        panel.semilogx(
            BANDWIDTHS, ratios[name], "o-", markersize=4, color=COLOURS[name], label=name
        )
    panel.axhline(1.0, color="0.5", linestyle="--", linewidth=1.0)
    panel.axhspan(0.85, 1.15, color="#2e7d32", alpha=0.10)
    panel.set_ylim(0.8, 1.25)
    panel.set_xlabel("B_n (cycles per symbol)")
    panel.set_ylabel("Monte Carlo / predicted")
    panel.set_title("agreement, with the 15 % band shaded", fontsize=9)
    panel.legend(fontsize=8)
    panel.grid(alpha=0.25, which="both")

    figure.suptitle(
        "Predicted against measured timing jitter: the white-noise formula fails where the "
        "detector's own self-noise dominates",
        fontsize=12,
    )
    figure.savefig("../screenshots/jitter_prediction_vs_simulation.png", dpi=130,
                   bbox_inches="tight")
    plt.close(figure)
    print("wrote ../screenshots/jitter_prediction_vs_simulation.png")
    for name in names:
        values = ratios[name]
        print(
            f"  {name:<16} Monte Carlo / predicted over B_n {BANDWIDTHS}: "
            + ", ".join(f"{value:.4f}" for value in values)
        )


if __name__ == "__main__":
    main()
