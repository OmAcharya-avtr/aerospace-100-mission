"""Keep-out-aware pointing windows for a fixed inertial target over an orbit.

Scans the worst-case clearance margin of a fixed target over two orbits, refines
the window boundaries by root finding, and plots the per-body margins together
with the fraction of the whole sky that stays available.

Run from products/P030/:  python examples/pointing_windows.py
Writes: ../screenshots/pointing_windows.png (relative to examples/)
"""

import datetime as dt
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from keepout import (  # noqa: E402
    OrbitPointingProblem,
    allowed_fraction,
    julian_date,
    spherical_to_unit,
)

OUT = pathlib.Path(__file__).resolve().parents[1] / "screenshots" / "pointing_windows.png"


def main() -> None:
    epoch = dt.datetime(2026, 3, 20, 12, 0, 0)
    problem = OrbitPointingProblem(
        epoch_jd=julian_date(epoch),
        altitude_m=550e3,
        inclination=np.radians(97.6),
        raan=0.4,
        arg_lat0=0.9,
        sun_exclusion=np.radians(45.0),
        earth_exclusion=np.radians(10.0),
        moon_exclusion=np.radians(15.0),
        reference="limb",
    )
    # M87*, RA 187.706 deg, dec +12.391 deg -- an arbitrary fixed inertial target.
    target = spherical_to_unit(np.radians(187.706), np.radians(12.391))
    period = problem.period

    step = 10.0
    t = np.arange(0.0, 2.0 * period + step, step)
    margins = np.array([problem.keepout_at(x).margins(target) for x in t])
    worst = margins.min(axis=1)
    windows = problem.windows(t, target, refine=True)

    print(f"epoch          : {epoch.isoformat()} UTC (JD {problem.epoch_jd})")
    print(f"orbit          : 550 km circular, i = 97.6 deg, "
          f"period {period:.3f} s = {period / 60.0:.3f} min")
    print("target         : RA 187.706 deg, dec +12.391 deg")
    print(f"scan           : 0 to {t[-1]:.1f} s at {step:g} s, {len(t)} samples")
    print()
    print(f"{'#':>3} {'start [s]':>13} {'end [s]':>13} {'duration [s]':>14} "
          f"{'duration [min]':>15}")
    total = 0.0
    for i, w in enumerate(windows, 1):
        total += w.duration
        print(f"{i:3d} {w.start:13.4f} {w.end:13.4f} {w.duration:14.4f} "
              f"{w.duration / 60.0:15.4f}")
    print(f"{'':3} {'':13} {'total':>13} {total:14.4f} {total / 60.0:15.4f}")
    print(f"duty cycle over the scan: {total / t[-1] * 100:.3f} %")
    print()
    for w in windows:
        if t[0] < w.start < t[-1]:
            print(f"margin at refined boundary t = {w.start:.6f} s: "
                  f"{problem.margin(w.start, target):+.3e} rad")
        if t[0] < w.end < t[-1]:
            print(f"margin at refined boundary t = {w.end:.6f} s: "
                  f"{problem.margin(w.end, target):+.3e} rad")

    t_coarse = np.arange(0.0, 2.0 * period + 60.0, 60.0)
    frac = np.array([allowed_fraction(problem.keepout_at(x)) for x in t_coarse])
    print()
    print(f"allowed sky fraction over the scan: min {frac.min() * 100:.4f} %, "
          f"mean {frac.mean() * 100:.4f} %, max {frac.max() * 100:.4f} %")

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12.0, 8.0), sharex=True, height_ratios=[2.1, 1.0]
    )
    colours = {"sun": "#d95f02", "earth": "#1b9e77", "moon": "#7570b3"}
    names = problem.keepout_at(0.0).names
    for k, name in enumerate(names):
        ax1.plot(t / 60.0, np.degrees(margins[:, k]), color=colours[name], lw=1.6,
                 label=f"{name} margin")
    ax1.plot(t / 60.0, np.degrees(worst), color="k", lw=2.2, ls="--",
             label="worst case (min over cones)")
    ax1.axhline(0.0, color="0.3", lw=1.0)
    for i, w in enumerate(windows):
        ax1.axvspan(w.start / 60.0, w.end / 60.0, color="#a6dba0", alpha=0.45,
                    label="pointing window" if i == 0 else None)
    ax1.set_ylabel("clearance margin [deg]")
    ax1.set_title(
        "Keep-out margins for a fixed inertial target, 550 km SSO, two orbits\n"
        f"Sun 45 deg + Earth 10 deg + Moon 15 deg to the limb; "
        f"{len(windows)} windows, duty cycle {total / t[-1] * 100:.1f} %",
        fontsize=12,
    )
    ax1.set_ylim(-80.0, 205.0)
    ax1.legend(loc="upper center", fontsize=9, ncol=4, framealpha=0.95)
    ax1.grid(alpha=0.3)

    ax2.plot(t_coarse / 60.0, frac * 100.0, color="#2166ac", lw=1.8)
    ax2.set_xlabel("time from epoch [min]")
    ax2.set_ylabel("allowed sky [%]")
    ax2.grid(alpha=0.3)
    ax2.set_title(
        "Fraction of the whole sky left by the three cones "
        "(band quadrature, exact in azimuth)", fontsize=10,
    )

    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=130)
    plt.close(fig)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
