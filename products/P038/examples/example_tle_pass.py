"""Downlink Doppler over a real ISS pass from a TLE, and the station-velocity term.

Uses ``sgp4`` for the satellite state and this package's frames module for the
ground station, including the ``omega_E x r`` rotation velocity. The second
panel shows what forcing that station velocity to zero does to the predicted
Doppler -- a kilohertz-class error, three orders of magnitude larger than any
relativistic term this package reports.

TLE: the ISS element set used as the worked example in the ``sgp4`` package
documentation, epoch 2019-12-09. Station: Chilbolton, UK.

Writes ../screenshots/tle_pass_doppler.png.
Runtime: under 5 s on one core.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sgp4.api import Satrec

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dopplerkit.doppler import one_way_doppler_hz
from dopplerkit.frames import (
    elevation_deg,
    julian_date,
    satellite_state_teme,
    station_state_teme,
)
from dopplerkit.geometry import State, range_rate_mps, slant_range_m

ISS_LINE1 = "1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991"
ISS_LINE2 = "2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472202482"
LAT_DEG, LON_DEG, ALT_M = 51.1450, -1.4365, 100.0
CARRIER_HZ = 2.2e9
T_CULMINATION = datetime(2019, 12, 9, 16, 39, 5, tzinfo=UTC)
OUT = Path(__file__).resolve().parent.parent / "screenshots" / "tle_pass_doppler.png"


def main() -> int:
    satrec = Satrec.twoline2rv(ISS_LINE1, ISS_LINE2)
    offsets = np.arange(-420.0, 421.0, 5.0)

    el = np.empty(offsets.size)
    rng = np.empty(offsets.size)
    rate = np.empty(offsets.size)
    rate_still = np.empty(offsets.size)
    for i, dt_s in enumerate(offsets):
        t = T_CULMINATION + timedelta(seconds=float(dt_s))
        sat = satellite_state_teme(satrec, t)
        sta = station_state_teme(LAT_DEG, LON_DEG, ALT_M, t)
        jd, fr = julian_date(t)
        el[i] = elevation_deg(sta, sat, LAT_DEG, LON_DEG, jd, fr)
        rng[i] = slant_range_m(sta, sat)
        rate[i] = range_rate_mps(sta, sat)
        rate_still[i] = range_rate_mps(
            State(position_m=sta.position_m, velocity_mps=np.zeros(3)), sat
        )

    doppler = np.array([one_way_doppler_hz(float(r), CARRIER_HZ) for r in rate])
    doppler_still = np.array([one_way_doppler_hz(float(r), CARRIER_HZ) for r in rate_still])
    err = doppler_still - doppler
    visible = el > 0.0

    print("TLE                      ISS, sgp4 documentation example, epoch 2019-12-09")
    print(f"station                  {LAT_DEG:+.4f} deg, {LON_DEG:+.4f} deg, {ALT_M:.0f} m")
    print(f"carrier                  {CARRIER_HZ / 1e9:.3f} GHz downlink, one-way")
    print(f"culmination epoch        {T_CULMINATION.isoformat()}")
    print(f"maximum elevation        {el.max():.4f} deg")
    print(f"minimum slant range      {rng.min() / 1e3:.4f} km")
    print(f"visible span in window   {visible.sum() * 5.0:.0f} s of {np.ptp(offsets):.0f} s")
    print(f"Doppler at window start  {doppler[0]:+.2f} Hz")
    print(f"Doppler at window end    {doppler[-1]:+.2f} Hz")
    print(f"Doppler span             {doppler.max() - doppler.min():.2f} Hz")
    print(f"station inertial speed   "
          f"{station_state_teme(LAT_DEG, LON_DEG, ALT_M, T_CULMINATION).speed_mps:.4f} m/s")
    print(f"max |zero-velocity error| {np.max(np.abs(err)):.2f} Hz")

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.6), sharex=True)

    ax = axes[0]
    ax.plot(offsets, doppler / 1e3, color="#1f4e79", lw=1.8,
            label="with station rotation velocity (correct)")
    ax.plot(offsets, doppler_still / 1e3, color="#b03030", lw=1.4, ls="--",
            label="station velocity forced to zero (wrong)")
    ax.axhline(0.0, color="0.4", lw=0.8, ls=":")
    ax.fill_between(offsets, -60, 60, where=visible, color="#2e7d32", alpha=0.08,
                    label="elevation above 0 deg")
    ax.set_ylim(-60, 60)
    ax.set_ylabel(r"one-way $\Delta f$  [kHz]")
    ax.set_title(
        f"ISS downlink Doppler at {CARRIER_HZ / 1e9:.1f} GHz, "
        f"max elevation {el.max():.1f} deg, 2019-12-09"
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    ax2 = axes[0].twinx()
    ax2.plot(offsets, el, color="0.55", lw=1.0)
    ax2.set_ylabel("elevation [deg]", color="0.45")
    ax2.tick_params(axis="y", labelcolor="0.45")
    ax2.set_ylim(-20, 90)

    ax = axes[1]
    ax.plot(offsets, err, color="#b03030", lw=1.8)
    ax.axhline(0.0, color="0.4", lw=0.8, ls=":")
    ax.set_ylabel("Doppler error from a\nzero-velocity station  [Hz]")
    ax.set_xlabel("time from culmination  [s]")
    ax.annotate(
        f"peak {np.max(np.abs(err)):.0f} Hz\n"
        f"(every O($\\beta^2$) term in this package is under 1.3 Hz at this carrier)",
        xy=(offsets[int(np.argmax(np.abs(err)))], err[int(np.argmax(np.abs(err)))]),
        xytext=(-100.0, 0.55 * float(np.max(np.abs(err)))),
        fontsize=9, arrowprops={"arrowstyle": "->", "color": "0.3", "lw": 0.9},
    )
    ax.grid(alpha=0.3)

    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=130)
    plt.close(fig)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
