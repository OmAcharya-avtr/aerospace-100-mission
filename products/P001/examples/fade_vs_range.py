"""Example 2: fade probability vs link range.

Compares Monte Carlo (with 95 % Wilson bands), the analytic lognormal
scintillation-only baseline, and the ML surrogate across 1-15 km.
Saves ../screenshots/fade_vs_range.png.

Run: python examples/fade_vs_range.py
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from beamtwin.budget import compute_budget  # noqa: E402
from beamtwin.channel import build_channel_model, sample_received_power_dbm  # noqa: E402
from beamtwin.scenario import load_scenario  # noqa: E402
from beamtwin.stats import analytic_fade_probability_lognormal, fade_probability  # noqa: E402
from beamtwin.surrogate import FadeSurrogate, default_model_path  # noqa: E402

SHOTS = ROOT / "screenshots"
N_MC = 200_000
SEED = 11
FLOOR = 1e-6


def main() -> None:
    scenario = load_scenario(ROOT / "examples" / "link_10km.yaml")
    surrogate = (
        FadeSurrogate.load(default_model_path()) if default_model_path().exists() else None
    )
    ranges_km = np.linspace(1.0, 15.0, 15)

    mc, lo, hi, analytic, sur = [], [], [], [], []
    for r_km in ranges_km:
        link = dataclasses.replace(scenario.link, range_m=float(r_km) * 1000.0)
        res = sample_received_power_dbm(link, scenario.channel, n_samples=N_MC, seed=SEED)
        est = fade_probability(res.samples_dbm, link.rx_sensitivity_dbm)
        mc.append(max(est.probability, FLOOR))
        lo.append(max(est.ci_low, FLOOR))
        hi.append(max(est.ci_high, FLOOR))
        model = build_channel_model(link, scenario.channel)
        analytic.append(
            max(
                analytic_fade_probability_lognormal(
                    compute_budget(link).margin_db, model.sigma_ln
                ),
                FLOOR,
            )
        )
        if surrogate is not None:
            sur.append(max(surrogate.predict(link, scenario.channel).probability, FLOOR))
        print(
            f"range={r_km:5.1f} km  margin={compute_budget(link).margin_db:+7.2f} dB  "
            f"P_MC={est.probability:.3e}  P_analytic={analytic[-1]:.3e}"
            + (f"  P_surrogate={sur[-1]:.3e}" if surrogate is not None else "")
        )

    SHOTS.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.fill_between(ranges_km, lo, hi, color="#2b7bba", alpha=0.25, label="MC 95 % Wilson CI")
    ax.semilogy(ranges_km, mc, "o-", color="#2b7bba", label="Monte Carlo (truth)")
    ax.semilogy(
        ranges_km, analytic, "s--", color="#c0392b", label="analytic lognormal (scint-only)"
    )
    if surrogate is not None:
        ax.semilogy(ranges_km, sur, "^:", color="#27ae60", label="ML surrogate")
    ax.axhline(FLOOR, color="grey", linewidth=0.8, linestyle=":")
    ax.text(1.1, FLOOR * 1.3, f"plot floor {FLOOR:.0e}", fontsize=7, color="grey")
    ax.set_xlabel("link range [km]")
    ax.set_ylabel("fade probability  P(P_rx < sensitivity)")
    ax.set_title(
        "BeamTwin: fade probability vs range\n"
        f"1550 nm, Cn2={scenario.channel.cn2:.1e}, "
        f"jitter={scenario.channel.pointing_jitter_rad * 1e6:.0f} urad, "
        f"alpha={scenario.link.attenuation_db_per_km:.1f} dB/km"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = SHOTS / "fade_vs_range.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nPlot -> {out}")


if __name__ == "__main__":
    main()
