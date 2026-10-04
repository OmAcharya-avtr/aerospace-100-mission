"""Carrier pre-compensation profile for an uplink, and what the sign error costs.

The transmitter sends ``f_nominal + offset`` so the far end receives
``f_nominal``. The offset is the NEGATIVE of the Doppler shift. The second
panel is the point of the figure: applying the offset with the wrong sign
leaves twice the Doppler error instead of none, which is the failure this
package's sign discipline exists to prevent.

Writes ../screenshots/precompensation_profile.png.
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
from dopplerkit.constants import C_M_S, WGS84_A_M
from dopplerkit.passprofile import compute_profile

CARRIER_HZ = 2.0255e9  # a representative S-band uplink
ALT_M = 500.0e3
RX_BANDWIDTH_HZ = 10.0e3  # illustrative receiver acquisition window, half-width
OUT = Path(__file__).resolve().parent.parent / "screenshots" / "precompensation_profile.png"


def main() -> int:
    orbit = CircularOverheadPass(orbit_radius_m=WGS84_A_M + ALT_M)
    horizon = orbit.horizon_time_s()
    times = np.linspace(-horizon, horizon, 1401)
    profile = compute_profile(
        orbit.observer_state, orbit.satellite_state, times, CARRIER_HZ
    )

    rate = profile.range_rate_mps
    precomp = profile.precomp_offset_hz
    assert precomp is not None

    # Residual carrier error at the spacecraft receiver, three cases.
    # The exact classical relation is f_rx = f_tx (1 - rho_dot/c).
    uncompensated = CARRIER_HZ * (1.0 - rate / C_M_S) - CARRIER_HZ
    correct = (CARRIER_HZ + precomp) * (1.0 - rate / C_M_S) - CARRIER_HZ
    sign_flipped = (CARRIER_HZ - precomp) * (1.0 - rate / C_M_S) - CARRIER_HZ

    print(f"uplink carrier                  {CARRIER_HZ / 1e9:.4f} GHz")
    print(f"altitude                        {ALT_M / 1e3:.0f} km")
    print(f"peak Doppler                    {profile.peak_doppler_hz():+.2f} Hz")
    print(f"peak pre-compensation offset    {precomp[int(np.argmax(np.abs(precomp)))]:+.2f} Hz")
    print(f"max |residual|, uncompensated   {np.max(np.abs(uncompensated)):10.4f} Hz")
    print(f"max |residual|, correct sign    {np.max(np.abs(correct)):10.4f} Hz")
    print(f"max |residual|, sign flipped    {np.max(np.abs(sign_flipped)):10.4f} Hz")
    print(f"ratio flipped / uncompensated   "
          f"{np.max(np.abs(sign_flipped)) / np.max(np.abs(uncompensated)):.4f}")
    print(f"convention                      {profile.doppler_convention}")

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.6), sharex=True)

    ax = axes[0]
    ax.plot(times, profile.doppler_hz / 1e3, color="#1f4e79", lw=1.8,
            label=r"Doppler shift $\Delta f$")
    ax.plot(times, precomp / 1e3, color="#2e7d32", lw=1.6, ls="--",
            label=r"pre-compensation offset $= -\Delta f$")
    ax.axhline(0.0, color="0.4", lw=0.8, ls=":")
    ax.set_ylabel("frequency  [kHz]")
    ax.set_title(
        f"Uplink carrier pre-compensation, {CARRIER_HZ / 1e9:.4f} GHz, "
        f"{ALT_M / 1e3:.0f} km overhead pass"
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.semilogy(times, np.abs(uncompensated), color="#b03030", lw=1.6,
                label="no pre-compensation")
    ax.semilogy(times, np.abs(sign_flipped), color="#c07000", lw=1.6, ls="--",
                label="pre-compensation applied with the WRONG sign")
    ax.semilogy(times, np.maximum(np.abs(correct), 1e-4), color="#2e7d32", lw=1.8,
                label="pre-compensation applied with the stated sign")
    ax.axhline(RX_BANDWIDTH_HZ, color="0.3", lw=1.0, ls=":",
               label=f"illustrative {RX_BANDWIDTH_HZ / 1e3:.0f} kHz acquisition window")
    ax.set_ylabel("|residual carrier error| at the spacecraft  [Hz]")
    ax.set_xlabel("time from zenith  [s]")
    ax.set_ylim(1e-3, 2e5)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=130)
    plt.close(fig)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
