"""Sky map of Sun, Earth and Moon exclusion cones and the allowed region.

Builds a realistic low-Earth-orbit keep-out set for a sensitive instrument,
draws the three exclusion cones and the surviving allowed region on a Mollweide
projection of the sky, and prints the allowed solid angle.

Run from products/P030/:  python examples/sky_map.py
Writes: ../screenshots/sky_map.png (relative to examples/)
"""

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from keepout import (  # noqa: E402
    OrbitPointingProblem,
    allowed_mask,
    allowed_solid_angle,
    allowed_solid_angle_monte_carlo,
    julian_date,
    rotation_matrix,
    spherical_to_unit,
    unit_to_spherical,
)

import datetime as dt  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "screenshots" / "sky_map.png"


def cone_boundary(axis, half_angle, n=721):
    """Points on the boundary circle of a cone, as unit vectors."""
    seed = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(seed, axis))) > 0.9:
        seed = np.array([1.0, 0.0, 0.0])
    perp = np.cross(axis, seed)
    perp /= np.linalg.norm(perp)
    start = np.cos(half_angle) * axis + np.sin(half_angle) * perp
    phis = np.linspace(0.0, 2.0 * np.pi, n)
    return np.array([rotation_matrix(axis, p) @ start for p in phis])


def wrap_ra(ra):
    """Right ascension in [0, 2 pi) to Mollweide's [-pi, pi)."""
    return np.where(ra > np.pi, ra - 2.0 * np.pi, ra)


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
    ks = problem.keepout_at(0.0)

    print(f"epoch                : {epoch.isoformat()} UTC (JD {problem.epoch_jd})")
    print(f"orbit                : {problem.altitude_m / 1e3:.0f} km circular, "
          f"i = {np.degrees(problem.inclination):.1f} deg, "
          f"period {problem.period / 60.0:.2f} min")
    print("instrument keep-out  : Sun 45 deg, Earth 10 deg, Moon 15 deg, "
          "measured to the limb")
    print()
    print(f"{'cone':>7} {'axis RA [deg]':>14} {'axis dec [deg]':>15} "
          f"{'half-angle [deg]':>17} {'solid angle [sr]':>17}")
    for c in ks:
        ra, dec = unit_to_spherical(c.axis)
        print(f"{c.name:>7} {np.degrees(ra):14.4f} {np.degrees(dec):15.4f} "
              f"{c.half_angle_deg:17.4f} {c.solid_angle:17.6f}")
    print()

    est = allowed_solid_angle(ks)
    mc = allowed_solid_angle_monte_carlo(ks, 500_000, seed=2026)
    print(f"allowed solid angle  : {est.solid_angle:.6f} sr "
          f"({est.fraction * 100:.4f} % of the sky), {est.n_samples} quadrature nodes")
    print(f"Monte Carlo check    : {mc.solid_angle:.6f} +/- {mc.standard_error:.6f} sr "
          f"({mc.n_samples} samples)")
    print()

    target = spherical_to_unit(np.radians(83.63), np.radians(22.01))
    print(f"target RA 83.63 deg, dec 22.01 deg -> violations: {ks.violations(target)}")
    print(f"worst-case margin    : {np.degrees(ks.margin(target)):+.4f} deg")

    # Sample RA directly on [-pi, pi] so the Mollweide mesh is monotonic.
    ra_grid = np.linspace(-np.pi, np.pi, 721)
    dec_grid = np.linspace(-np.pi / 2, np.pi / 2, 361)
    rr, dd = np.meshgrid(ra_grid, dec_grid)
    dirs = spherical_to_unit(rr, dd)
    mask = allowed_mask(ks, dirs)

    fig = plt.figure(figsize=(12.5, 6.8))
    ax = fig.add_subplot(111, projection="mollweide")
    ax.pcolormesh(
        rr, dd, mask.astype(float), cmap="Greys_r", vmin=-0.25, vmax=1.0,
        shading="auto", rasterized=True,
    )
    colours = {"sun": "#d95f02", "earth": "#1b9e77", "moon": "#7570b3"}
    for c in ks:
        pts = cone_boundary(c.axis, c.half_angle)
        ra, dec = unit_to_spherical(pts)
        x = wrap_ra(ra)
        jumps = np.nonzero(np.abs(np.diff(x)) > np.pi)[0]
        segments = np.split(np.arange(len(x)), jumps + 1)
        for k, seg in enumerate(segments):
            if len(seg) < 2:
                continue
            ax.plot(x[seg], dec[seg], color=colours[c.name], lw=2.0,
                    label=f"{c.name} {c.half_angle_deg:.1f} deg" if k == 0 else None)
        ra_c, dec_c = unit_to_spherical(c.axis)
        ax.plot(wrap_ra(ra_c), dec_c, marker="o", color=colours[c.name], ms=7,
                markeredgecolor="k", linestyle="none")

    ra_t, dec_t = unit_to_spherical(target)
    ax.plot(wrap_ra(ra_t), dec_t, marker="*", color="#e7298a", ms=18,
            markeredgecolor="k", linestyle="none", label="target (RA 83.63, dec 22.01)")

    ax.grid(True, color="0.55", lw=0.5)
    ticks_deg = np.arange(-150.0, 151.0, 30.0)
    ax.set_xticks(np.radians(ticks_deg))
    ax.set_xticklabels([f"{int(round((d % 360.0) / 15.0)):d}h" for d in ticks_deg],
                       fontsize=8)
    ax.set_xlabel("right ascension (increasing to the right)", fontsize=9)
    ax.set_title(
        f"Allowed sky (white) at {epoch.isoformat()} UTC, 550 km SSO\n"
        f"Sun 45 deg + Earth 10 deg + Moon 15 deg to the limb; "
        f"allowed {est.fraction * 100:.2f} % = {est.solid_angle:.4f} sr",
        fontsize=12, pad=22,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.13, 1.10), fontsize=9, frameon=True)
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
