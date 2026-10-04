"""RF and optical link capacity against range and against the ground-leg weather.

Produces ``screenshots/link_capacity.png``:

* left panel -- achievable rate against slant range for the Ka-band RF
  terminal and the 1550 nm optical terminal, both from
  ``constellink.synthdata``, with the RF Shannon bound shown so the gap
  between an information-theoretic bound and a modem requirement is visible;
* right panel -- achievable rate on the ground leg against elevation, for
  three visibilities, with the Kim-Kruse specific attenuation scaled to the
  slant path by the plane-parallel cosecant law.

Every terminal figure is a stated modelling assumption, not a device
datasheet.  See ``constellink.synthdata.default_rf_terminal`` and
``default_optical_terminal`` for the exact values and why they were chosen.

Run: ``python examples/example_capacity.py``
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from constellink.capacity import (  # noqa: E402
    kim_specific_attenuation_db_km,
    optical_link,
    rf_link,
    slant_path_attenuation_db,
)
from constellink.frames import WGS84_A_KM  # noqa: E402
from constellink.geometry import slant_range_at_elevation  # noqa: E402
from constellink.synthdata import (  # noqa: E402
    default_optical_terminal,
    default_rf_terminal,
)

ALT_KM = 550.0
R_ORBIT = WGS84_A_KM + ALT_KM
BOUNDARY_LAYER_KM = 2.0   # equivalent zenith path length, a stated assumption
OUT = os.path.join(os.path.dirname(__file__), "..", "screenshots",
                   "link_capacity.png")


def main() -> int:
    rf = default_rf_terminal()
    opt = default_optical_terminal()

    ranges = np.linspace(400.0, 5000.0, 160)
    rf_rate = np.array([rf_link(r, rf).achievable_rate_bps for r in ranges])
    rf_shannon = np.array([rf_link(r, rf).shannon_capacity_bps for r in ranges])
    opt_rate = np.array([optical_link(r, opt).achievable_rate_bps
                         for r in ranges])

    elevations = np.linspace(10.0, 90.0, 161)
    slant = np.array([slant_range_at_elevation(R_ORBIT, e) for e in elevations])
    visibilities = (50.0, 10.0, 2.0)
    opt_ground = {}
    rf_ground = {}
    for vis in visibilities:
        beta = kim_specific_attenuation_db_km(vis, 1550.0)
        zenith_db = beta * BOUNDARY_LAYER_KM
        atm = np.array([slant_path_attenuation_db(zenith_db, e)
                        for e in elevations])
        opt_ground[vis] = np.array([
            optical_link(s, opt, atmospheric_loss_db=a).achievable_rate_bps
            for s, a in zip(slant, atm, strict=True)])
        # The RF leg uses a cloud/gas proxy that is not visibility-driven; a
        # fixed 0.5 dB zenith excess is shown for reference only.
        rf_atm = np.array([slant_path_attenuation_db(0.5, e) for e in elevations])
        rf_ground[vis] = np.array([
            rf_link(s, rf, extra_loss_db=a).achievable_rate_bps
            for s, a in zip(slant, rf_atm, strict=True)])

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.6),
                                  constrained_layout=True)
    ax.plot(ranges, rf_rate / 1e6, lw=1.6, color="tab:blue",
            label="RF achievable rate (Eb/N0 + margin)")
    ax.plot(ranges, rf_shannon / 1e6, lw=1.1, ls="--", color="tab:blue",
            label="RF Shannon bound at 500 MHz")
    ax.plot(ranges, opt_rate / 1e6, lw=1.6, color="tab:green",
            label="optical achievable rate (500 photons/bit)")
    ax.set_xlabel("slant range [km]")
    ax.set_ylabel("rate [Mbit/s]")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Vacuum link: rate vs range\n"
                 "both curves fall as 1/R^2; the optical lead is set by the "
                 "terminal assumptions, not by physics alone")

    styles = {50.0: "-", 10.0: "--", 2.0: ":"}
    for vis in visibilities:
        ax2.plot(elevations, opt_ground[vis] / 1e6, lw=1.6, ls=styles[vis],
                 color="tab:green", label=f"optical, V = {vis:g} km")
    ax2.plot(elevations, rf_ground[visibilities[0]] / 1e6, lw=1.6,
             color="tab:blue", label="RF, 0.5 dB zenith excess")
    ax2.set_xlabel("elevation [deg]")
    ax2.set_ylabel("rate [Mbit/s]")
    ax2.set_yscale("log")
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(loc="lower right", fontsize=8)
    ax2.set_title(f"Ground leg at {ALT_KM:g} km altitude\n"
                  f"Kim-Kruse attenuation over a {BOUNDARY_LAYER_KM:g} km "
                  f"zenith path, cosecant-scaled; valid above 10 deg only")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=120)
    plt.close(fig)

    r1500 = rf_link(1500.0, rf)
    o1500 = optical_link(1500.0, opt)
    print("Range sweep (vacuum)")
    print(f"  RF  at  400 km : {rf_rate[0] / 1e6:10.2f} Mbit/s")
    print(f"  RF  at 1500 km : {r1500.achievable_rate_bps / 1e6:10.2f} Mbit/s"
          f"  (C/N0 {r1500.c_over_n0_dbhz:.2f} dB-Hz, "
          f"FSPL {r1500.fspl_db:.2f} dB)")
    print(f"  RF  at 5000 km : {rf_rate[-1] / 1e6:10.2f} Mbit/s")
    print(f"  OPT at 1500 km : {o1500.achievable_rate_bps / 1e6:10.2f} Mbit/s"
          f"  (geometric loss {o1500.geometric_loss_db:.2f} dB, "
          f"Rx {o1500.rx_power_w * 1e6:.4f} uW)")
    print(f"  RF rate ratio 400 km / 5000 km : "
          f"{rf_rate[0] / rf_rate[-1]:.2f} "
          f"(inverse-square prediction {(5000.0 / 400.0) ** 2:.2f})")
    print("")
    print("Ground leg, optical, at 10 deg and 90 deg elevation")
    for vis in visibilities:
        beta = kim_specific_attenuation_db_km(vis, 1550.0)
        print(f"  V = {vis:5g} km : beta {beta:7.4f} dB/km, "
              f"zenith {beta * BOUNDARY_LAYER_KM:6.3f} dB, "
              f"rate {opt_ground[vis][0] / 1e6:10.2f} -> "
              f"{opt_ground[vis][-1] / 1e6:10.2f} Mbit/s")
    print(f"written: {os.path.normpath(OUT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
