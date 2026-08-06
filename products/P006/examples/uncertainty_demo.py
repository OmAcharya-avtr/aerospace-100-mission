"""Example: first-order uncertainty propagation vs Monte Carlo cross-check.

Propagates 1-sigma uncertainties on transmit power, atmospheric attenuation
and pointing error to the link margin (delta method, JCGM 100:2008), then
overlays the implied Gaussian on a 20 000-sample Monte Carlo histogram.
Saves ../screenshots/uncertainty_histogram.png.

Run: python examples/uncertainty_demo.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from linkbudgetx import LinkBudget, monte_carlo_margin, propagate_margin_sigma  # noqa: E402


def main() -> None:
    budget = LinkBudget(
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
        "tx_power_dbm": 0.5,          # dB — laser power calibration
        "atmos_attenuation_db_per_km": 0.05,  # dB/km — visibility estimate
        "pointing_error_rad": 0.02e-3,  # rad — pointing jitter about bias
    }

    unc = propagate_margin_sigma(budget, sigmas)
    samples = monte_carlo_margin(budget, sigmas, n_samples=20_000, seed=2026)
    mc_std = samples.std(ddof=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(samples, bins=80, density=True, alpha=0.6, color="tab:blue",
            label=f"Monte Carlo (n=20000)\nstd = {mc_std:.3f} dB")
    x = np.linspace(samples.min(), samples.max(), 400)
    pdf = (1.0 / (unc.sigma_margin_db * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((x - unc.margin_db) / unc.sigma_margin_db) ** 2
    )
    ax.plot(x, pdf, "r-", lw=2,
            label=f"First-order Gaussian\nsigma = {unc.sigma_margin_db:.3f} dB")
    ax.set_xlabel("Link margin [dB]")
    ax.set_ylabel("Probability density")
    ax.set_title("Margin uncertainty: first-order propagation vs Monte Carlo")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out = ROOT / "screenshots" / "uncertainty_histogram.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    print(f"Nominal margin        : {unc.margin_db:.4f} dB")
    print(f"First-order sigma     : {unc.sigma_margin_db:.4f} dB")
    print(f"Monte Carlo std       : {mc_std:.4f} dB")
    print(f"Relative discrepancy  : {abs(mc_std - unc.sigma_margin_db) / mc_std * 100:.2f} %")
    for name, contrib in sorted(unc.contributions_db.items(), key=lambda kv: -kv[1]):
        print(f"  contribution {name:<30} {contrib:.4f} dB")


if __name__ == "__main__":
    main()
