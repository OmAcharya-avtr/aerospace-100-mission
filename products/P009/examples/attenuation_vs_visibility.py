"""Attenuation vs visibility at 850/1310/1550 nm: Kim & Kruse baselines vs ML with 90% PI.

Runs the full pipeline (seeded synthetic data -> gradient-boosting model) and plots
specific attenuation (dB/km) against visibility (km) for the three FSO telecom
wavelengths, at RH = 85 %. Saves the figure to ../screenshots/.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fogcast import FogCastModel, kim_attenuation_db_km, kruse_attenuation_db_km

OUT = Path(__file__).resolve().parents[1] / "screenshots"
RH = 85.0  # % relative humidity used for the ML curves
WAVELENGTHS_NM = (850.0, 1310.0, 1550.0)


def main() -> None:
    model = FogCastModel.train_default()
    v = np.geomspace(0.05, 50.0, 200)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, lam in zip(axes, WAVELENGTHS_NM):
        lam_arr = np.full_like(v, lam)
        rh_arr = np.full_like(v, RH)
        kim = np.asarray(kim_attenuation_db_km(v, lam_arr))
        kruse = np.asarray(kruse_attenuation_db_km(v, lam_arr))
        point, lo, hi = model.predict(v, lam_arr, rh_arr)

        ax.fill_between(v, lo, hi, alpha=0.25, color="tab:blue", label="ML 90% PI")
        ax.loglog(v, point, color="tab:blue", lw=1.8, label="ML point (RH=85%)")
        ax.loglog(v, kim, "k--", lw=1.5, label="Kim (2001)")
        ax.loglog(v, kruse, "r:", lw=1.5, label="Kruse (1962)")
        ax.set_title(f"{lam:.0f} nm")
        ax.set_xlabel("Visibility V (km)")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("Specific attenuation (dB/km)")
    axes[0].legend(loc="lower left", fontsize=9)
    fig.suptitle(
        "FogCast: fog/aerosol attenuation vs visibility — Kim/Kruse baselines vs ML "
        "(synthetic-data model, research-grade)",
        fontsize=11,
    )
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    out_path = OUT / "attenuation_vs_visibility.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
