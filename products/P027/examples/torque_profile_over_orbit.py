"""Disturbance-torque profile over one orbit for the reference LEO smallsat.

Produces ``screenshots/torque_profile_over_orbit.png``: the four contributions stacked
against argument of latitude, the body-frame components of the total, and the angular
momentum accumulated by each source.

Run from ``examples/``:  ``python3 torque_profile_over_orbit.py``
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from disturbtorque import (  # noqa: E402
    SOURCES,
    budget,
    compute_profile,
    momentum_accumulation,
    reference_orbit,
    reference_smallsat,
    sun_direction_for_beta,
)

OUT = pathlib.Path(__file__).resolve().parents[1] / "screenshots"
OUT.mkdir(exist_ok=True)

LABELS = {
    "gravity_gradient": "gravity gradient",
    "aerodynamic": "aerodynamic",
    "solar": "solar radiation",
    "magnetic": "residual magnetic dipole",
}
COLOURS = {
    "gravity_gradient": "#3b6ea5",
    "aerodynamic": "#c1683c",
    "solar": "#d9b310",
    "magnetic": "#5c8f5c",
}


def main() -> None:
    sc = reference_smallsat()
    orbit = reference_orbit(500.0)
    sun = sun_direction_for_beta(orbit.inclination_rad, orbit.raan_rad, np.radians(20.0))
    prof = compute_profile(sc, orbit, sun, n_samples=1441)
    b = budget(prof, "body")

    u_deg = np.degrees(prof.u_rad)
    mags = np.array([np.linalg.norm(prof.torque(s, "body"), axis=1) for s in SOURCES])

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 10.5), sharex=True)

    ax = axes[0]
    ax.stackplot(
        u_deg, mags * 1e6,
        labels=[LABELS[s] for s in SOURCES],
        colors=[COLOURS[s] for s in SOURCES],
        edgecolor="none",
    )
    dark = ~prof.illuminated
    if dark.any():
        top = mags.sum(axis=0).max() * 1e6
        ax.fill_between(u_deg, 0, top, where=dark, color="0.1", alpha=0.08, linewidth=0)
        edges = np.where(np.diff(dark.astype(int)) != 0)[0]
        for e in edges:
            ax.axvline(u_deg[e], color="0.3", ls=":", lw=1.0)
        i0 = int(np.argmax(dark))
        ax.annotate("eclipse", xy=(u_deg[i0] + 25, top * 0.94), fontsize=9, color="0.25")
    ax.set_ylabel(r"torque magnitude [$\mu$N m]")
    ax.set_title(
        "Disturbance torques over one orbit, body frame\n"
        "reference 100 kg smallsat, 500 km circular, i = 51.6 deg, beta = 20 deg",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.25)
    ax.set_xlim(0, 360)

    ax = axes[1]
    total = prof.torque("total", "body") * 1e6
    for k, (lab, col) in enumerate(zip("xyz", ["#8c2d04", "#31688e", "#35978f"])):
        ax.plot(u_deg, total[:, k], color=col, lw=1.4, label=f"$T_{lab}$")
    sec = prof.secular("total", "body") * 1e6
    for k, col in enumerate(["#8c2d04", "#31688e", "#35978f"]):
        ax.axhline(sec[k], color=col, ls="--", lw=0.9, alpha=0.7)
    ax.set_ylabel(r"total torque component [$\mu$N m]")
    ax.legend(loc="upper right", fontsize=9, ncol=3)
    ax.grid(alpha=0.25)
    ax.text(
        0.012, 0.06,
        "dashed: secular (orbit-averaged) component",
        transform=ax.transAxes, fontsize=8.5, color="0.3",
    )

    ax = axes[2]
    for s in SOURCES:
        h = momentum_accumulation(prof, s, "eci")
        ax.plot(u_deg, np.linalg.norm(h, axis=1) * 1e3, color=COLOURS[s], lw=1.5,
                label=LABELS[s])
    h_tot = momentum_accumulation(prof, "total", "eci")
    ax.plot(u_deg, np.linalg.norm(h_tot, axis=1) * 1e3, color="k", lw=1.8, ls="--",
            label="total")
    ax.set_xlabel("argument of latitude [deg]")
    ax.set_ylabel(r"$|\int T\,dt|$, ECI frame [mN m s]")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    path = OUT / "torque_profile_over_orbit.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)

    print(f"wrote {path}")
    print(f"period {prof.period_s:.1f} s, eclipse fraction {prof.eclipse_fraction:.4f}")
    print(f"{'source':<26}{'peak [N m]':>13}{'secular [N m]':>16}{'dh/orbit [N m s]':>19}")
    for s in (*SOURCES, "total"):
        name = LABELS.get(s, s)
        print(f"{name:<26}{b[s]['peak_nm']:>13.4e}{b[s]['secular_magnitude_nm']:>16.4e}"
              f"{b[s]['secular_momentum_per_orbit_nms']:>19.4e}")


if __name__ == "__main__":
    main()
