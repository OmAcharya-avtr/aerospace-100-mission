"""Validation Level 1: Monte Carlo cross-check of first-order propagation.

Two scenarios:
1. Mildly nonlinear regime (nonzero nominal pointing error, small sigmas):
   first-order sigma should match the MC standard deviation closely.
2. Deliberate linearity breakdown (zero nominal pointing error): the first
   derivative w.r.t. pointing error vanishes, so first-order propagation
   under-reports the jitter contribution. Reported as a documented failure
   mode of the linear method, not of the library.

Output saved to validation/mc_crosscheck_output.txt. Runtime < 30 s.

Run: python validation/mc_crosscheck.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from linkbudgetx import LinkBudget, monte_carlo_margin, propagate_margin_sigma  # noqa: E402


def main() -> None:
    lines: list[str] = ["linkbudgetx 0.1.0 — Monte Carlo cross-check (seed=2026, n=20000)", ""]

    base = LinkBudget(
        tx_power_dbm=20.0,
        wavelength_nm=1550.0,
        beam_divergence_rad=1.0e-3,
        range_km=10.0,
        rx_aperture_diameter_m=0.1,
        rx_sensitivity_dbm=-40.0,
        tx_optics_efficiency=0.8,
        rx_optics_efficiency=0.8,
        pointing_error_rad=0.25e-3,
        atmos_attenuation_db_per_km=0.5,
        beam_profile="gaussian",
    )
    sigmas = {
        "tx_power_dbm": 0.5,
        "atmos_attenuation_db_per_km": 0.05,
        "pointing_error_rad": 0.02e-3,
    }

    # Scenario 1: valid linear regime.
    unc = propagate_margin_sigma(base, sigmas)
    samples = monte_carlo_margin(base, sigmas, n_samples=20_000, seed=2026)
    mc_std = samples.std(ddof=1)
    rel = abs(mc_std - unc.sigma_margin_db) / mc_std * 100.0
    status = "PASS" if rel < 5.0 else "FAIL"
    lines += [
        "Scenario 1 — nominal pointing error 0.25 mrad (linear regime)",
        f"  nominal margin        = {unc.margin_db:.4f} dB",
        f"  first-order sigma     = {unc.sigma_margin_db:.4f} dB",
        f"  Monte Carlo std       = {mc_std:.4f} dB",
        f"  Monte Carlo mean      = {samples.mean():.4f} dB",
        f"  relative discrepancy  = {rel:.2f} %   [{status}, tol 5 %]",
        "",
    ]

    # Scenario 2: linearity breakdown at zero nominal pointing error.
    b0 = base.replace(pointing_error_rad=0.0)
    sig0 = {"pointing_error_rad": 0.05e-3}
    unc0 = propagate_margin_sigma(b0, sig0)
    samples0 = monte_carlo_margin(b0, sig0, n_samples=20_000, seed=2026)
    lines += [
        "Scenario 2 — nominal pointing error 0 (documented breakdown case)",
        "  margin is quadratic in pointing error; derivative at 0 is ~0, so",
        "  first-order propagation misses the jitter contribution entirely:",
        f"  first-order sigma     = {unc0.sigma_margin_db:.4f} dB",
        f"  Monte Carlo std       = {samples0.std(ddof=1):.4f} dB",
        f"  Monte Carlo mean shift= {samples0.mean() - unc0.margin_db:.4f} dB "
        "(bias also invisible to first order)",
        "  EXPECTED disagreement — use monte_carlo_margin() in this regime.",
    ]

    text = "\n".join(lines)
    print(text)
    (ROOT / "validation" / "mc_crosscheck_output.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
