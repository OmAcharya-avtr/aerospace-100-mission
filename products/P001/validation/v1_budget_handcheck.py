"""Validation V1: hand-checked deterministic link budget arithmetic.

Recomputes every budget term for a reference 10 km link with independent
longhand arithmetic and compares against beamtwin.budget.compute_budget.

Run: python validation/v1_budget_handcheck.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from beamtwin.budget import LinkParams, compute_budget, kim_attenuation_db_per_km  # noqa: E402

TOL_DB = 1e-9


def main() -> int:
    lam = 1550e-9
    w0 = 0.02
    rng_m = 10_000.0
    a = 0.05
    params = LinkParams(
        wavelength_m=lam,
        tx_power_dbm=20.0,
        tx_efficiency=0.8,
        rx_efficiency=0.8,
        beam_waist_radius_m=w0,
        rx_aperture_radius_m=a,
        range_m=rng_m,
        pointing_bias_rad=2e-6,
        attenuation_db_per_km=2.5,
        rx_sensitivity_dbm=-30.0,
    )
    b = compute_budget(params)

    lines = ["V1 — Deterministic link budget hand-check (10 km, 1550 nm)", "=" * 62, ""]

    # --- Step 1: Rayleigh range, z_R = pi w0^2 / lambda  (Saleh & Teich 3.1-11)
    z_r = math.pi * w0**2 / lam
    lines += [
        "Step 1  Rayleigh range  z_R = pi*w0^2/lambda",
        f"        = pi * (0.02 m)^2 / 1.55e-6 m = pi * 4.0e-4 / 1.55e-6",
        f"        = {z_r:.4f} m",
        "",
    ]

    # --- Step 2: beam radius w(z) = w0 sqrt(1 + (z/z_R)^2)  (Saleh & Teich 3.1-8)
    ratio = rng_m / z_r
    w = w0 * math.sqrt(1.0 + ratio**2)
    lines += [
        "Step 2  Beam radius at Rx  w(L) = w0*sqrt(1+(L/z_R)^2)",
        f"        L/z_R = 10000 / {z_r:.4f} = {ratio:.6f}",
        f"        w = 0.02 * sqrt(1 + {ratio:.6f}^2) = {w:.6f} m",
        f"        code: {b.beam_radius_at_rx_m:.6f} m   delta = {abs(w - b.beam_radius_at_rx_m):.3e} m",
        "",
    ]

    # --- Step 3: geometric capture eta = 1 - exp(-2 a^2 / w^2)
    eta = 1.0 - math.exp(-2.0 * a**2 / w**2)
    geo_db = -10.0 * math.log10(eta)
    lines += [
        "Step 3  Geometric capture  eta = 1 - exp(-2a^2/w^2)",
        f"        2a^2/w^2 = 2*(0.05)^2/({w:.6f})^2 = {2 * a**2 / w**2:.6f}",
        f"        eta = 1 - exp(-{2 * a**2 / w**2:.6f}) = {eta:.6f}",
        f"        loss = -10*log10({eta:.6f}) = {geo_db:.6f} dB",
        f"        code: {b.geometric_loss_db:.6f} dB   delta = {abs(geo_db - b.geometric_loss_db):.3e} dB",
        "",
    ]

    # --- Step 4: pointing loss (static bias), L_p = exp(-2 d^2 / w^2)
    d = 2e-6 * rng_m
    lp = math.exp(-2.0 * d**2 / w**2)
    lp_db = -10.0 * math.log10(lp)
    lines += [
        "Step 4  Static pointing loss  d = theta_bias*L, L_p = exp(-2d^2/w^2)",
        f"        d = 2e-6 rad * 10000 m = {d:.4f} m",
        f"        L_p = exp(-2*{d:.4f}^2/{w:.6f}^2) = {lp:.6f}",
        f"        loss = {lp_db:.6f} dB",
        f"        code: {b.pointing_loss_db:.6f} dB   delta = {abs(lp_db - b.pointing_loss_db):.3e} dB",
        "",
    ]

    # --- Step 5: optics losses -10log10(0.8) each
    opt_db = -10.0 * math.log10(0.8)
    lines += [
        "Step 5  Optics losses  -10*log10(0.8) = "
        f"{opt_db:.6f} dB each (Tx and Rx)",
        f"        code Tx: {b.tx_optics_loss_db:.6f} dB, Rx: {b.rx_optics_loss_db:.6f} dB",
        "",
    ]

    # --- Step 6: atmospheric loss alpha*L
    atm_db = 2.5 * rng_m / 1000.0
    lines += [
        "Step 6  Atmospheric loss  alpha*L = 2.5 dB/km * 10 km = "
        f"{atm_db:.6f} dB",
        f"        code: {b.atmospheric_loss_db:.6f} dB",
        "",
    ]

    # --- Step 7: totals
    p_rx = 20.0 - opt_db - geo_db - lp_db - atm_db - opt_db
    margin = p_rx - (-30.0)
    lines += [
        "Step 7  Received power  P_rx = 20 - "
        f"{opt_db:.4f} - {geo_db:.4f} - {lp_db:.4f} - {atm_db:.4f} - {opt_db:.4f}",
        f"        = {p_rx:.6f} dBm",
        f"        code: {b.received_power_dbm:.6f} dBm   delta = {abs(p_rx - b.received_power_dbm):.3e} dB",
        f"        Margin = {p_rx:.6f} - (-30) = {margin:.6f} dB",
        f"        code: {b.margin_db:.6f} dB   delta = {abs(margin - b.margin_db):.3e} dB",
        "",
    ]

    # --- Step 8: Kim model spot-check at V = 7 km, 1550 nm
    # 1 km < V <= 6 km branch is q = 0.16V+0.34; V=7 km falls in 6<V<=50 -> q=1.3
    v = 7.0
    q = 1.3
    beta = (3.91 / v) * (1550e-9 / 550e-9) ** (-q)
    kim_db = (10.0 / math.log(10.0)) * beta
    kim_code = kim_attenuation_db_per_km(v, 1550e-9)
    lines += [
        "Step 8  Kim model spot-check (Kim et al. 2001, SPIE 4214), V = 7 km, 1550 nm",
        f"        6 < V <= 50 km  ->  q = 1.3",
        f"        beta = (3.91/7)*(1550/550)^-1.3 = {3.91 / v:.6f} * "
        f"{(1550e-9 / 550e-9) ** (-q):.6f} = {beta:.6f} 1/km",
        f"        alpha = 4.343 * {beta:.6f} = {kim_db:.6f} dB/km",
        f"        code: {kim_code:.6f} dB/km   delta = {abs(kim_db - kim_code):.3e} dB/km",
        "",
    ]

    deltas = {
        "beam_radius_m": abs(w - b.beam_radius_at_rx_m),
        "geometric_loss_db": abs(geo_db - b.geometric_loss_db),
        "pointing_loss_db": abs(lp_db - b.pointing_loss_db),
        "atmospheric_loss_db": abs(atm_db - b.atmospheric_loss_db),
        "received_power_dbm": abs(p_rx - b.received_power_dbm),
        "margin_db": abs(margin - b.margin_db),
        "kim_db_per_km": abs(kim_db - kim_code),
    }
    worst = max(deltas.values())
    ok = worst < TOL_DB
    lines += [
        f"RESULT: max |hand - code| = {worst:.3e} (tolerance {TOL_DB:.0e})",
        f"STATUS: {'PASS' if ok else 'FAIL'}",
    ]
    text = "\n".join(lines)
    print(text)
    (Path(__file__).parent / "v1_budget_handcheck.txt").write_text(text + "\n", encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
