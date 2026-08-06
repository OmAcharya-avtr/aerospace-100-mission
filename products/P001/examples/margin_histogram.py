"""Example 3: distribution of the instantaneous fade margin.

Shows the Monte Carlo margin histogram for the 10 km scenario, decomposed
into scintillation-only and jitter-only contributions, with the fade
threshold and key percentiles marked.
Saves ../screenshots/margin_histogram.png.

Run: python examples/margin_histogram.py
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

from beamtwin.channel import sample_received_power_dbm  # noqa: E402
from beamtwin.scenario import load_scenario  # noqa: E402
from beamtwin.stats import fade_probability, margin_moments, margin_percentiles  # noqa: E402

SHOTS = ROOT / "screenshots"
N_MC = 300_000
SEED = 5


def main() -> None:
    scenario = load_scenario(ROOT / "examples" / "link_10km.yaml")
    sens = scenario.link.rx_sensitivity_dbm

    combined = sample_received_power_dbm(
        scenario.link, scenario.channel, n_samples=N_MC, seed=SEED
    ).samples_dbm
    scint_only = sample_received_power_dbm(
        scenario.link,
        dataclasses.replace(scenario.channel, pointing_jitter_rad=0.0),
        n_samples=N_MC,
        seed=SEED,
    ).samples_dbm
    jitter_only = sample_received_power_dbm(
        scenario.link,
        dataclasses.replace(scenario.channel, cn2=0.0),
        n_samples=N_MC,
        seed=SEED,
    ).samples_dbm

    est = fade_probability(combined, sens)
    pcts = margin_percentiles(combined, sens)
    mom = margin_moments(combined, sens)
    print(f"combined  : P_fade={est.probability:.4e}  mean={mom['mean_db']:+.2f} dB  "
          f"std={mom['std_db']:.2f} dB")
    print("percentiles [dB]: " + ", ".join(f"{k}={v:+.2f}" for k, v in pcts.items()))
    for name, s in (("scint-only", scint_only), ("jitter-only", jitter_only)):
        e = fade_probability(s, sens)
        m = margin_moments(s, sens)
        print(f"{name:<11}: P_fade={e.probability:.4e}  mean={m['mean_db']:+.2f} dB  "
              f"std={m['std_db']:.2f} dB")

    SHOTS.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    bins = np.linspace(
        min(combined.min(), jitter_only.min()) - sens - 1.0,
        max(combined.max(), scint_only.max()) - sens + 1.0,
        220,
    )
    ax.hist(
        scint_only - sens, bins=bins, density=True, histtype="step",
        color="#c0392b", linewidth=1.2, label="scintillation only",
    )
    ax.hist(
        jitter_only - sens, bins=bins, density=True, histtype="step",
        color="#27ae60", linewidth=1.2, label="jitter only",
    )
    ax.hist(
        combined - sens, bins=bins, density=True, histtype="stepfilled",
        color="#2b7bba", alpha=0.45, label="combined",
    )
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.4,
               label=f"fade threshold (P_fade={est.probability:.2e})")
    ax.axvline(pcts["p01"], color="#8e44ad", linestyle=":", linewidth=1.2,
               label=f"1st pct = {pcts['p01']:+.1f} dB")
    ax.axvline(mom["mean_db"], color="#e67e22", linestyle="-.", linewidth=1.2,
               label=f"mean = {mom['mean_db']:+.1f} dB")
    ax.set_xlabel("instantaneous fade margin [dB]")
    ax.set_ylabel("probability density [1/dB]")
    ax.set_title(
        f"BeamTwin margin distribution — {scenario.name}\n"
        f"n={N_MC:,} MC samples, seed={SEED}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = SHOTS / "margin_histogram.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nPlot -> {out}")


if __name__ == "__main__":
    main()
