"""Example 1: predicted Cn^2 profile with its interval band, against the baselines.

Trains the default seeded model (~16 s on 2 cores), predicts the profile for one
held-out scenario, and plots the 90 % prediction band together with the
synthetic ground truth, HV 5/7, SLC-Day and SLC-Night.

    python examples/profile_with_intervals.py

Writes ``screenshots/profile_with_intervals.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cncast.baselines import hv57, slc_day, slc_night  # noqa: E402
from cncast.dataset import default_altitude_grid, profile_cn2  # noqa: E402
from cncast.model import train_default_model  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "screenshots" / "profile_with_intervals.png"


def main() -> int:
    model, art = train_default_model()
    grid = default_altitude_grid(60)

    scenarios = [art["test_scenarios"][0], art["test_scenarios"][1]]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.5), sharey=True)

    for ax, sc in zip(axes, scenarios, strict=True):
        pred = model.predict_scenario(sc, grid)
        truth = profile_cn2(sc, grid)
        ax.fill_betweenx(
            grid,
            pred.cn2_lower,
            pred.cn2_upper,
            color="tab:blue",
            alpha=0.20,
            label=f"CnCast {pred.coverage:.0%} prediction interval",
        )
        ax.plot(pred.cn2, grid, color="tab:blue", lw=2.0, label="CnCast median")
        ax.plot(truth, grid, color="k", lw=1.6, ls="--", label="synthetic truth (generator)")
        ax.plot(hv57(grid), grid, color="tab:red", lw=1.4, label="HV 5/7 baseline")
        ax.plot(slc_day(grid), grid, color="tab:orange", lw=1.2, ls=":", label="SLC-Day")
        ax.plot(slc_night(grid), grid, color="tab:green", lw=1.2, ls=":", label="SLC-Night")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1e-19, 1e-13)
        ax.set_ylim(5.0, 20_000.0)
        ax.grid(alpha=0.3, which="both")
        ax.set_xlabel(r"$C_n^2$  [m$^{-2/3}$]")
        ax.set_title(
            f"T = {sc.surface_temp_c:.1f} $^\\circ$C, wind = {sc.surface_wind_m_s:.1f} m/s\n"
            f"RH = {sc.relative_humidity_pct:.0f} %, "
            f"hour = {sc.hour_of_day:.1f}, day = {sc.day_of_year}",
            fontsize=10,
        )
    axes[0].set_ylabel("altitude above site [m]  (log scale)")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle(
        "CnCast predicted $C_n^2$ profiles with 90 % prediction intervals vs published "
        "baselines\n"
        "Truth here is a SYNTHETIC generator, not a measurement — see DATASET_CARD.md",
        fontsize=11,
    )
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=130)
    plt.close(fig)

    band = np.mean(np.log10(pred.cn2_upper) - np.log10(pred.cn2_lower))
    sys.stdout.write(
        f"wrote {OUT}\nmean interval width for the last panel: {band:.3f} dex\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
