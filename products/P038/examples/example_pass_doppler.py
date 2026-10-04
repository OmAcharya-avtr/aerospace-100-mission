"""Doppler, range-rate and Doppler rate over a circular overhead pass.

Shows the sign convention in one figure: Doppler positive (up-shift) while
approaching, zero at closest approach, negative while receding; range-rate the
other way round; Doppler rate negative throughout and most negative at
closest approach.

Writes ../screenshots/pass_doppler_profile.png.
Runtime: under 2 s on one core.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dopplerkit.analytic import CircularOverheadPass
from dopplerkit.constants import WGS84_A_M
from dopplerkit.doppler import LinkDirection
from dopplerkit.passprofile import (
    compute_profile,
    doppler_zero_crossing_s,
    time_of_closest_approach_s,
)

CARRIER_HZ = 2.2e9
ALT_M = 500.0e3
OUT = Path(__file__).resolve().parent.parent / "screenshots" / "pass_doppler_profile.png"


def main() -> int:
    orbit = CircularOverheadPass(orbit_radius_m=WGS84_A_M + ALT_M)
    horizon = orbit.horizon_time_s()
    times = np.linspace(-horizon, horizon, 1401)

    one = compute_profile(orbit.observer_state, orbit.satellite_state, times, CARRIER_HZ)
    two = compute_profile(
        orbit.observer_state, orbit.satellite_state, times, CARRIER_HZ,
        direction=LinkDirection.TWO_WAY,
    )
    tca = time_of_closest_approach_s(
        orbit.observer_state, orbit.satellite_state, -horizon, horizon
    )
    zero = doppler_zero_crossing_s(
        orbit.observer_state, orbit.satellite_state, -horizon, horizon
    )

    print(f"altitude                 {ALT_M / 1e3:.0f} km")
    print(f"carrier                  {CARRIER_HZ / 1e9:.3f} GHz")
    print(f"horizon to horizon       {2.0 * horizon:.1f} s")
    print(f"peak one-way Doppler     {one.peak_doppler_hz():+.2f} Hz")
    print(f"one-way Doppler span     {one.doppler_span_hz():.2f} Hz")
    print(f"peak two-way Doppler     {two.peak_doppler_hz():+.2f} Hz")
    print(f"Doppler rate at TCA      {two.doppler_rate_hz_per_s.min():+.3f} Hz/s (two-way)")
    print(f"time of closest approach {tca:+.6f} s")
    print(f"Doppler zero crossing    {zero:+.6f} s")
    print(f"|TCA - zero crossing|    {abs(tca - zero):.3e} s")
    print(f"convention               {one.doppler_convention}")

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 10.0), sharex=True)

    ax = axes[0]
    ax.plot(times, one.range_rate_mps, color="#1f4e79", lw=1.8, label="range-rate")
    ax.axhline(0.0, color="0.4", lw=0.8, ls=":")
    ax.axvline(tca, color="#b03030", lw=1.0, ls="--", label=f"closest approach {tca:+.3f} s")
    ax.set_ylabel(r"$\dot{\rho}$  [m/s]")
    ax.set_title(
        f"Circular overhead pass, {ALT_M / 1e3:.0f} km, non-rotating Earth, "
        f"{CARRIER_HZ / 1e9:.1f} GHz"
    )
    ax.annotate("approaching\n($\\dot{\\rho}<0$)", xy=(-0.55 * horizon, -4000),
                ha="center", fontsize=9, color="#1f4e79")
    ax.annotate("receding\n($\\dot{\\rho}>0$)", xy=(0.55 * horizon, 4000),
                ha="center", fontsize=9, color="#1f4e79")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(times, one.doppler_hz / 1e3, color="#1f4e79", lw=1.8, label="one-way")
    ax.plot(times, two.doppler_hz / 1e3, color="#c07000", lw=1.4, ls="--",
            label="two-way (G = 1), exactly 2x")
    ax.axhline(0.0, color="0.4", lw=0.8, ls=":")
    ax.axvline(zero, color="#b03030", lw=1.0, ls="--",
               label=f"Doppler zero {zero:+.3f} s")
    ax.set_ylabel(r"$\Delta f$  [kHz]")
    ax.annotate("UP-SHIFT\n(approaching)", xy=(-0.62 * horizon, 78),
                ha="center", fontsize=9, color="#1f4e79")
    ax.annotate("DOWN-SHIFT\n(receding)", xy=(0.62 * horizon, -78),
                ha="center", fontsize=9, color="#1f4e79")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(times, one.doppler_rate_hz_per_s, color="#1f4e79", lw=1.8, label="one-way")
    ax.plot(times, two.doppler_rate_hz_per_s, color="#c07000", lw=1.4, ls="--",
            label="two-way (G = 1)")
    ax.axhline(0.0, color="0.4", lw=0.8, ls=":")
    ax.axvline(tca, color="#b03030", lw=1.0, ls="--")
    ax.set_ylabel(r"$d\Delta f/dt$  [Hz/s]")
    ax.set_xlabel("time from zenith  [s]")
    ax.annotate(
        f"most negative at closest approach:\n{two.doppler_rate_hz_per_s.min():.1f} Hz/s two-way",
        xy=(0.0, two.doppler_rate_hz_per_s.min()),
        xytext=(0.3 * horizon, 0.55 * two.doppler_rate_hz_per_s.min()),
        fontsize=9, arrowprops={"arrowstyle": "->", "color": "0.3", "lw": 0.9},
    )
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=130)
    plt.close(fig)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
