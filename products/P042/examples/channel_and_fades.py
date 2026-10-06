"""Example 1: what the channel looks like, and why fade duration is emergent.

Writes ``../screenshots/channel_and_fades.png``.

Four panels:

1. 2 s of SNR sample path at three correlation times, with the lowest MODCOD
   threshold drawn in. Notice that the *depth* of the fades is the same in all
   three -- it is set by the scintillation index -- while their *duration* is
   not: duration is a consequence of the correlation time, never an input.
2. Fade-duration histograms at the same three correlation times, threshold held
   at the 10% quantile of each path so the outage fraction is identical by
   construction. Notice the long right tail: the mean is not the number that
   hurts, the tail is.
3. Measured mean and maximum fade duration against correlation time, log-log.
   Notice the maximum grows much faster than the mean, and that neither is a
   straight line of slope 1 -- the discretely sampled driver has a
   sampling-rate-dependent fade-duration distribution, which is a limitation of
   the model and is documented as one.
4. Measured against analytic level-crossing rate (Rice, equations (10)-(12) of
   ``acmpilot.channel``). Notice the points sit on the diagonal: this is the one
   place in the package where an analytic prediction and a sample statistic agree
   tightly, and it is the check that the sample paths are what they claim to be.

Runtime: about 15 s on one core.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from acmpilot.channel import (  # noqa: E402
    ChannelConfig,
    fade_statistics,
    rice_level_crossing_rate_hz,
    snr_db_path,
)

OUT = Path(__file__).resolve().parent.parent / "screenshots" / "channel_and_fades.png"
OUT_REL = "screenshots/channel_and_fades.png"
TAUS_MS = (2.5, 10.0, 40.0)
LOWEST_THRESHOLD_DB = 3.141  # MODCOD 0, measured, validation/modcod_thresholds.json


def main() -> int:
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.5))

    # Panel 1: sample paths
    ax = axes[0, 0]
    for tau_c_ms in TAUS_MS:
        cfg = ChannelConfig(tau_c_s=tau_c_ms * 1e-3, sigma_i2=0.5, mean_snr_db=14.0)
        snr = snr_db_path(cfg, 2000, 5)
        t_ms = np.arange(snr.size) * cfg.slot_s * 1e3
        ax.plot(t_ms, snr, lw=0.8, label=f"tau_c = {tau_c_ms:g} ms")
    ax.axhline(
        LOWEST_THRESHOLD_DB, color="k", ls="--", lw=1.0,
        label=f"lowest MODCOD threshold {LOWEST_THRESHOLD_DB:.2f} dB",
    )
    ax.set_xlabel("time, ms")
    ax.set_ylabel("SNR per symbol, dB")
    ax.set_title("Lognormal SNR paths, sigma_I^2 = 0.5\nsame fade depth, different fade duration")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)

    # Panel 2: fade-duration histograms
    ax = axes[0, 1]
    for tau_c_ms in TAUS_MS:
        cfg = ChannelConfig(tau_c_s=tau_c_ms * 1e-3, sigma_i2=0.5, mean_snr_db=14.0)
        snr = snr_db_path(cfg, 300_000, 9)
        thr = float(np.quantile(snr, 0.10))
        below = snr < thr
        padded = np.concatenate(([False], below, [False]))
        edges = np.flatnonzero(padded[1:] != padded[:-1])
        lengths = (edges[1::2] - edges[0::2]) * cfg.slot_s * 1e3
        ax.hist(
            lengths, bins=np.logspace(0, 2.6, 45), histtype="step", lw=1.3,
            label=f"tau_c = {tau_c_ms:g} ms, mean {lengths.mean():.2f} ms",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("fade duration, ms")
    ax.set_ylabel("count")
    ax.set_title(
        "Fade-duration distribution, threshold at the 10% quantile\n"
        "(outage fraction identical by construction)"
    )
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)

    # Panel 3: scaling
    ax = axes[1, 0]
    taus = np.array([2.5, 5.0, 10.0, 20.0, 40.0, 80.0])
    means, maxima = [], []
    for tau_c_ms in taus:
        cfg = ChannelConfig(tau_c_s=tau_c_ms * 1e-3, sigma_i2=0.5, mean_snr_db=14.0)
        snr = snr_db_path(cfg, 300_000, 13)
        thr = float(np.quantile(snr, 0.10))
        fs = fade_statistics(snr, thr, dt_s=cfg.slot_s)
        means.append(fs["mean_fade_duration_s"] * 1e3)
        maxima.append(fs["max_fade_duration_s"] * 1e3)
    slope_mean = float(np.polyfit(np.log10(taus), np.log10(means), 1)[0])
    slope_max = float(np.polyfit(np.log10(taus), np.log10(maxima), 1)[0])
    ax.loglog(taus, means, "o-", label=f"mean, log-log slope {slope_mean:.2f}")
    ax.loglog(taus, maxima, "s-", label=f"maximum, log-log slope {slope_max:.2f}")
    ax.loglog(taus, taus, "k--", lw=1.0, label="slope 1 reference")
    ax.set_xlabel("driver correlation time tau_c, ms")
    ax.set_ylabel("fade duration, ms")
    ax.set_title(
        "Fade duration against correlation time\n"
        "neither slope is 1: the model is sampling-rate dependent"
    )
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3, which="both")

    # Panel 4: Rice cross-check
    ax = axes[1, 1]
    meas, pred = [], []
    for tau_c_ms in (5.0, 10.0, 20.0, 40.0):
        cfg = ChannelConfig(tau_c_s=tau_c_ms * 1e-3, sigma_i2=0.5, mean_snr_db=14.0)
        snr = snr_db_path(cfg, 300_000, 17)
        for thr in (6.0, 8.0, 10.0, 12.0):
            fs = fade_statistics(snr, thr, dt_s=cfg.slot_s)
            meas.append(fs["level_crossing_rate_hz"])
            pred.append(
                rice_level_crossing_rate_hz(
                    sigma_i2=cfg.sigma_i2, tau_c_s=cfg.tau_c_s, threshold_db=thr,
                    mean_snr_db=cfg.mean_snr_db, slot_s=cfg.slot_s,
                )
            )
    meas_a, pred_a = np.array(meas), np.array(pred)
    ax.loglog(pred_a, meas_a, "o", ms=5)
    lim = [min(pred_a.min(), meas_a.min()) * 0.7, max(pred_a.max(), meas_a.max()) * 1.4]
    ax.loglog(lim, lim, "k--", lw=1.0, label="exact agreement")
    worst = float(np.max(np.abs(meas_a / pred_a - 1.0)))
    ax.set_xlabel("analytic level-crossing rate, Hz (Rice, eq 10-12)")
    ax.set_ylabel("measured level-crossing rate, Hz")
    ax.set_title(
        "Analytic against measured crossing rate\n"
        f"worst relative deviation {worst * 100:.1f}% over 16 points"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    fig.suptitle(
        "acmpilot: the correlated optical fading channel. Research-grade simulation, "
        "not measured turbulence data.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=120)
    plt.close(fig)
    print(f"mean fade duration log-log slope against tau_c: {slope_mean:.4f}")
    print(f"max  fade duration log-log slope against tau_c: {slope_max:.4f}")
    print(f"worst Rice deviation over 16 points: {worst * 100:.2f}%")
    print(f"written: {OUT_REL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
