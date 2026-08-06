"""Example: 10 km terrestrial FSO link — margin vs range sweep.

Sweeps range 1-20 km for a 1550 nm terrestrial link (1 mrad full-angle
Gaussian beam, 10 cm receiver, 0.5 dB/km clear-air attenuation per the
Kim et al. 2001 visibility-model regime) and plots link margin vs range.
Saves ../screenshots/range_sweep.png.

Run: python examples/range_sweep.py  (from the product root or examples/)
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from linkbudgetx import LinkBudget  # noqa: E402


def main() -> None:
    base = LinkBudget(
        tx_power_dbm=20.0,  # 100 mW
        wavelength_nm=1550.0,
        beam_divergence_rad=1.0e-3,  # 1 mrad FULL angle
        range_km=10.0,
        rx_aperture_diameter_m=0.1,
        rx_sensitivity_dbm=-40.0,
        tx_optics_efficiency=0.8,
        rx_optics_efficiency=0.8,
        pointing_error_rad=0.25e-3,
        atmos_attenuation_db_per_km=0.5,  # clear air
        beam_profile="gaussian",
    )

    ranges = np.linspace(1.0, 20.0, 200)
    attenuations = {"clear (0.5 dB/km)": 0.5, "haze (4 dB/km)": 4.0, "light fog (20 dB/km)": 20.0}

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, alpha in attenuations.items():
        margins = [
            base.replace(range_km=float(r), atmos_attenuation_db_per_km=alpha)
            .compute()
            .margin_db
            for r in ranges
        ]
        ax.plot(ranges, margins, label=label)

    ax.axhline(0.0, color="k", lw=0.8, ls="--", label="0 dB (link closes)")
    r10 = base.compute().margin_db
    ax.plot(10.0, r10, "ko", ms=5)
    ax.annotate(f"10 km, clear: {r10:.1f} dB", (10.0, r10), textcoords="offset points",
                xytext=(8, 8))
    ax.set_xlabel("Range [km]")
    ax.set_ylabel("Link margin [dB]")
    ax.set_title("FSO link margin vs range (1550 nm, 1 mrad full angle, 10 cm Rx)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out = ROOT / "screenshots" / "range_sweep.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    print(f"Margin at 10 km, clear air: {r10:.3f} dB")


if __name__ == "__main__":
    main()
