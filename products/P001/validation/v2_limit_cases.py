"""Validation V2: Monte Carlo vs analytic limits, and combined-case monotonicity.

V2a  Scintillation-only limit (jitter = 0, bias = 0): MC fade probability must
     agree with the closed-form lognormal baseline within the MC 95 % Wilson
     interval, over a range of margins.
V2b  Jitter-only limit (Cn2 = 0): MC mean pointing-loss factor must agree with
     the closed form E[L_p] = 1/(1 + 4 sigma_d^2/w^2).
V2c  Combined-case sanity: fade probability monotone increasing in
     scintillation index, monotone increasing in jitter, monotone decreasing
     in margin.

Run: python validation/v2_limit_cases.py
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from beamtwin.budget import LinkParams, compute_budget  # noqa: E402
from beamtwin.channel import (  # noqa: E402
    ChannelParams,
    build_channel_model,
    mean_pointing_loss_fraction,
    sample_received_power_dbm,
)
from beamtwin.stats import analytic_fade_probability_lognormal, fade_probability  # noqa: E402

N_MC = 400_000
SEED = 2024


def v2a(lines: list[str]) -> bool:
    lines += [
        "V2a — Scintillation-only limit vs closed-form lognormal baseline",
        "     (jitter = 0, bias = 0; Cn2 = 5e-16, 10 km, 1550 nm)",
        f"     MC n = {N_MC}, seed = {SEED}",
        "",
        f"{'margin_dB':>10}  {'sigma_ln':>9}  {'P_MC':>11}  {'CI95 low':>11}  {'CI95 high':>11}  "
        f"{'P_analytic':>11}  {'in CI':>6}",
    ]
    ok = True
    base = LinkParams(range_m=10_000.0, attenuation_db_per_km=2.5, rx_sensitivity_dbm=-30.0)
    channel = ChannelParams(cn2=5e-16, pointing_jitter_rad=0.0)
    p_rx = compute_budget(base).received_power_dbm
    for margin in (2.0, 4.0, 6.0, 8.0, 10.0):
        link = dataclasses.replace(base, rx_sensitivity_dbm=p_rx - margin)
        res = sample_received_power_dbm(link, channel, n_samples=N_MC, seed=SEED)
        est = fade_probability(res.samples_dbm, link.rx_sensitivity_dbm)
        model = build_channel_model(link, channel)
        analytic = analytic_fade_probability_lognormal(
            compute_budget(link).margin_db, model.sigma_ln
        )
        inside = est.ci_low <= analytic <= est.ci_high
        ok &= inside
        lines.append(
            f"{margin:>10.1f}  {model.sigma_ln:>9.4f}  {est.probability:>11.4e}  "
            f"{est.ci_low:>11.4e}  {est.ci_high:>11.4e}  {analytic:>11.4e}  "
            f"{'yes' if inside else 'NO':>6}"
        )
    lines += ["", f"V2a STATUS: {'PASS' if ok else 'FAIL'}", ""]
    return ok


def v2b(lines: list[str]) -> bool:
    lines += [
        "V2b — Jitter-only limit vs closed-form mean pointing loss",
        "     (Cn2 = 0, bias = 0); E[L_p] = 1/(1 + 4*sigma_d^2/w^2)",
        "",
        f"{'jitter_urad':>12}  {'sigma_d/w':>10}  {'E[L_p] MC':>11}  {'E[L_p] exact':>13}  "
        f"{'rel err':>10}  {'tol':>8}",
    ]
    ok = True
    link = LinkParams(range_m=10_000.0, attenuation_db_per_km=2.5, rx_sensitivity_dbm=-30.0)
    budget = compute_budget(link)
    p0_w = 10.0 ** ((budget.received_power_dbm + budget.pointing_loss_db) / 10.0) * 1e-3
    for jitter_urad in (2.0, 5.0, 10.0, 20.0):
        channel = ChannelParams(cn2=0.0, pointing_jitter_rad=jitter_urad * 1e-6)
        res = sample_received_power_dbm(link, channel, n_samples=N_MC, seed=SEED)
        p_w = 10.0 ** (res.samples_dbm / 10.0) * 1e-3
        mc_mean = float(np.mean(p_w / p0_w))
        model = build_channel_model(link, channel)
        exact = mean_pointing_loss_fraction(model.sigma_disp_m, model.beam_radius_at_rx_m)
        # MC standard error of the mean, converted to a relative 4-sigma tolerance.
        sem = float(np.std(p_w / p0_w)) / np.sqrt(N_MC)
        tol = 4.0 * sem / exact
        rel = abs(mc_mean - exact) / exact
        inside = rel <= tol
        ok &= inside
        lines.append(
            f"{jitter_urad:>12.1f}  {model.sigma_disp_m / model.beam_radius_at_rx_m:>10.4f}  "
            f"{mc_mean:>11.6f}  {exact:>13.6f}  {rel:>10.2e}  {tol:>8.2e}"
        )
    lines += ["", f"V2b STATUS: {'PASS' if ok else 'FAIL'}", ""]
    return ok


def v2c(lines: list[str]) -> bool:
    lines += ["V2c — Combined-case monotonicity sanity checks", ""]
    ok = True
    base = LinkParams(range_m=10_000.0, attenuation_db_per_km=2.5, rx_sensitivity_dbm=-30.0)

    # (i) monotone increasing in Cn2 (hence in scintillation index)
    lines.append("  (i) P_fade vs Cn2 (jitter = 5 urad, margin fixed):")
    probs = []
    for cn2 in (1e-16, 5e-16, 1e-15, 5e-15, 1e-14):
        ch = ChannelParams(cn2=cn2, pointing_jitter_rad=5e-6)
        res = sample_received_power_dbm(base, ch, n_samples=N_MC, seed=SEED)
        p = fade_probability(res.samples_dbm, base.rx_sensitivity_dbm).probability
        probs.append(p)
        lines.append(f"      Cn2={cn2:.1e} -> P_fade={p:.4e}")
    mono = all(b >= a for a, b in zip(probs, probs[1:]))
    ok &= mono
    lines += [f"      monotone non-decreasing: {'yes' if mono else 'NO'}", ""]

    # (ii) monotone increasing in jitter
    lines.append("  (ii) P_fade vs pointing jitter (Cn2 = 5e-16):")
    probs = []
    for j in (0.0, 2.0, 5.0, 10.0, 20.0):
        ch = ChannelParams(cn2=5e-16, pointing_jitter_rad=j * 1e-6)
        res = sample_received_power_dbm(base, ch, n_samples=N_MC, seed=SEED)
        p = fade_probability(res.samples_dbm, base.rx_sensitivity_dbm).probability
        probs.append(p)
        lines.append(f"      jitter={j:.1f} urad -> P_fade={p:.4e}")
    mono = all(b >= a for a, b in zip(probs, probs[1:]))
    ok &= mono
    lines += [f"      monotone non-decreasing: {'yes' if mono else 'NO'}", ""]

    # (iii) monotone decreasing in margin
    lines.append("  (iii) P_fade vs margin (Cn2 = 5e-16, jitter = 5 urad):")
    probs = []
    p_rx = compute_budget(base).received_power_dbm
    for margin in (2.0, 4.0, 6.0, 8.0, 12.0):
        link = dataclasses.replace(base, rx_sensitivity_dbm=p_rx - margin)
        ch = ChannelParams(cn2=5e-16, pointing_jitter_rad=5e-6)
        res = sample_received_power_dbm(link, ch, n_samples=N_MC, seed=SEED)
        p = fade_probability(res.samples_dbm, link.rx_sensitivity_dbm).probability
        probs.append(p)
        lines.append(f"      margin={margin:.1f} dB -> P_fade={p:.4e}")
    mono = all(b <= a for a, b in zip(probs, probs[1:]))
    ok &= mono
    lines += [f"      monotone non-increasing: {'yes' if mono else 'NO'}", ""]

    lines += [f"V2c STATUS: {'PASS' if ok else 'FAIL'}", ""]
    return ok


def main() -> int:
    lines = ["V2 — Monte Carlo limit cases and combined-case sanity", "=" * 62, ""]
    ok = v2a(lines)
    ok &= v2b(lines)
    ok &= v2c(lines)
    lines.append(f"OVERALL V2 STATUS: {'PASS' if ok else 'FAIL'}")
    text = "\n".join(lines)
    print(text)
    (Path(__file__).parent / "v2_limit_cases.txt").write_text(text + "\n", encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
