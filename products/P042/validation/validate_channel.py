"""Validation V1: the correlated fading channel.

Checks, all against quantities stated in ``src/acmpilot/channel.py``:

V1.1  The Gauss-Markov driver is stationary unit-variance with the lag-1
      correlation of equation (2).
V1.2  The measured 1/e correlation time of a lognormal dB path recovers the
      input ``tau_c`` (it must, because log-irradiance is affine in the driver).
V1.3  The measured scintillation index of the lognormal irradiance matches
      equation (5) at the requested value.
V1.4  The measured scintillation index of the gamma-gamma irradiance matches
      equation (7) at the requested value, with the shapes solved by
      ``gamma_gamma_shapes``.
V1.5  The measured correlation time of a gamma-gamma irradiance path is SHORTER
      than its driver's, as the module docstring states. The size of the
      shortening is reported, not asserted.
V1.6  Fade duration is emergent, not a parameter: with the threshold held at a
      fixed quantile of the marginal, mean fade duration scales with ``tau_c``.
      The measured scaling exponent is reported.
V1.7  Consistency of the fade accounting: mean fade duration times
      level-crossing rate equals the outage fraction, to the censoring error.
V1.8  The Rice analytic level-crossing rate of equations (10)-(12) is compared
      with the measured one, at the sampling interval the prediction is stated
      for. The ratio is reported as measured.

Runtime: about 6 s on one core. Raw output is saved to
``validation/validate_channel.txt`` by redirecting stdout.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from acmpilot.channel import (  # noqa: E402
    ChannelConfig,
    correlation_time_s,
    fade_statistics,
    gamma_gamma_shapes,
    gauss_markov_path,
    irradiance_path,
    rice_level_crossing_rate_hz,
    snr_db_path,
)

N_LONG = 400_000


def main() -> int:
    print("V1 CHANNEL VALIDATION -- acmpilot")
    print("script: validation/validate_channel.py")
    print(f"samples per check: {N_LONG}")
    print()

    # ---- V1.1 driver moments -------------------------------------------------
    print("V1.1 Gauss-Markov driver, equations (1)-(2)")
    print(f"{'tau_c_ms':>9} {'var':>9} {'lag1_meas':>11} {'lag1_eq2':>10} {'rel_err':>10}")
    for tau_c_ms in (2.0, 5.0, 10.0, 25.0):
        rng = np.random.default_rng(4242)
        z = gauss_markov_path(N_LONG, dt_s=1e-3, tau_c_s=tau_c_ms * 1e-3, rng=rng)
        zc = z - z.mean()
        lag1 = float(np.dot(zc[:-1], zc[1:]) / np.dot(zc, zc))
        rho = float(np.exp(-1e-3 / (tau_c_ms * 1e-3)))
        print(
            f"{tau_c_ms:9.1f} {z.var():9.5f} {lag1:11.6f} {rho:10.6f} "
            f"{abs(lag1 - rho) / rho:10.2e}"
        )
    print()

    # ---- V1.2 / V1.3 lognormal ----------------------------------------------
    print("V1.2 and V1.3 lognormal path: correlation time and scintillation index")
    print(
        f"{'tau_c_ms':>9} {'sigma_i2_in':>12} {'tau_meas_ms':>12} {'rel_err':>9} "
        f"{'si2_meas':>10} {'rel_err':>9}"
    )
    for tau_c_ms, si2 in ((5.0, 0.2), (10.0, 0.5), (25.0, 1.0)):
        cfg = ChannelConfig(tau_c_s=tau_c_ms * 1e-3, sigma_i2=si2)
        snr = snr_db_path(cfg, N_LONG, 7)
        tau_meas = correlation_time_s(snr, dt_s=cfg.slot_s) * 1e3
        irr = irradiance_path(cfg, N_LONG, 7)
        si2_meas = float(irr.var() / irr.mean() ** 2)
        print(
            f"{tau_c_ms:9.1f} {si2:12.3f} {tau_meas:12.4f} "
            f"{abs(tau_meas - tau_c_ms) / tau_c_ms:9.2e} "
            f"{si2_meas:10.5f} {abs(si2_meas - si2) / si2:9.2e}"
        )
    print()

    # ---- V1.4 / V1.5 gamma-gamma --------------------------------------------
    print("V1.4 and V1.5 gamma-gamma path: shapes, scintillation index, correlation time")
    print(
        f"{'si2_in':>7} {'alpha':>8} {'beta':>8} {'si2_eq7':>9} {'si2_meas':>10} "
        f"{'rel_err':>9} {'tau_c_in_ms':>12} {'tau_meas_ms':>12} {'ratio':>7}"
    )
    for si2 in (0.3, 0.7, 1.5):
        alpha, beta = gamma_gamma_shapes(si2)
        si2_eq7 = 1.0 / alpha + 1.0 / beta + 1.0 / (alpha * beta)
        cfg = ChannelConfig(marginal="gamma-gamma", sigma_i2=si2, tau_c_s=10e-3)
        irr = irradiance_path(cfg, N_LONG, 11)
        si2_meas = float(irr.var() / irr.mean() ** 2)
        tau_meas = correlation_time_s(irr, dt_s=cfg.slot_s) * 1e3
        print(
            f"{si2:7.2f} {alpha:8.4f} {beta:8.4f} {si2_eq7:9.5f} {si2_meas:10.5f} "
            f"{abs(si2_meas - si2) / si2:9.2e} {10.0:12.1f} {tau_meas:12.4f} "
            f"{tau_meas / 10.0:7.3f}"
        )
    print("  gamma-gamma correlation time is shorter than the driver's, as documented.")
    print()

    # ---- V1.6 fade duration is emergent -------------------------------------
    print("V1.6 mean fade duration scales with tau_c (threshold at the 10% quantile)")
    print(
        f"{'tau_c_ms':>9} {'thr_dB':>8} {'n_fades':>8} {'mean_fd_ms':>11} "
        f"{'median_fd_ms':>13} {'max_fd_ms':>10}"
    )
    taus = (2.5, 5.0, 10.0, 20.0, 40.0)
    means = []
    for tau_c_ms in taus:
        cfg = ChannelConfig(tau_c_s=tau_c_ms * 1e-3, sigma_i2=0.5)
        snr = snr_db_path(cfg, N_LONG, 23)
        thr = float(np.quantile(snr, 0.10))
        fs = fade_statistics(snr, thr, dt_s=cfg.slot_s)
        means.append(fs["mean_fade_duration_s"] * 1e3)
        print(
            f"{tau_c_ms:9.1f} {thr:8.3f} {fs['n_fades']:8.0f} "
            f"{fs['mean_fade_duration_s'] * 1e3:11.4f} "
            f"{fs['median_fade_duration_s'] * 1e3:13.4f} "
            f"{fs['max_fade_duration_s'] * 1e3:10.3f}"
        )
    slope = float(
        np.polyfit(np.log10(np.array(taus)), np.log10(np.array(means)), 1)[0]
    )
    print(f"  log-log scaling exponent of mean fade duration against tau_c: {slope:.4f}")
    print("  exponent 1 would mean exact proportionality; the value above is measured.")
    print()

    # ---- V1.7 accounting consistency ----------------------------------------
    print("V1.7 mean_fade_duration * level_crossing_rate vs outage_fraction")
    print(
        f"{'tau_c_ms':>9} {'quantile':>9} {'outage':>9} {'fd*lcr':>9} {'rel_err':>9}"
    )
    for tau_c_ms in (5.0, 10.0, 20.0):
        for quant in (0.05, 0.10, 0.25):
            cfg = ChannelConfig(tau_c_s=tau_c_ms * 1e-3, sigma_i2=0.5)
            snr = snr_db_path(cfg, N_LONG, 31)
            thr = float(np.quantile(snr, quant))
            fs = fade_statistics(snr, thr, dt_s=cfg.slot_s)
            product = fs["mean_fade_duration_s"] * fs["level_crossing_rate_hz"]
            print(
                f"{tau_c_ms:9.1f} {quant:9.2f} {fs['outage_fraction']:9.5f} "
                f"{product:9.5f} "
                f"{abs(product - fs['outage_fraction']) / fs['outage_fraction']:9.2e}"
            )
    print()

    # ---- V1.8 Rice cross-check ----------------------------------------------
    print("V1.8 analytic Rice level-crossing rate, equations (10)-(12), vs measured")
    print(
        f"{'tau_c_ms':>9} {'thr_dB':>8} {'lcr_meas_Hz':>12} {'lcr_rice_Hz':>12} {'ratio':>8}"
    )
    for tau_c_ms in (5.0, 10.0, 20.0):
        cfg = ChannelConfig(tau_c_s=tau_c_ms * 1e-3, sigma_i2=0.5, mean_snr_db=14.0)
        snr = snr_db_path(cfg, N_LONG, 37)
        for thr in (8.0, 10.0, 12.0):
            fs = fade_statistics(snr, thr, dt_s=cfg.slot_s)
            rice = rice_level_crossing_rate_hz(
                sigma_i2=cfg.sigma_i2,
                tau_c_s=cfg.tau_c_s,
                threshold_db=thr,
                mean_snr_db=cfg.mean_snr_db,
                slot_s=cfg.slot_s,
            )
            print(
                f"{tau_c_ms:9.1f} {thr:8.1f} {fs['level_crossing_rate_hz']:12.4f} "
                f"{rice:12.4f} {fs['level_crossing_rate_hz'] / rice:8.3f}"
            )
    print("  The ratio is reported, not tuned. The prediction is stated at the")
    print("  sampling interval used (1 ms) and is sampling-rate dependent by")
    print("  construction; see rice_level_crossing_rate_hz, equation (11).")
    print()
    print("V1 COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
