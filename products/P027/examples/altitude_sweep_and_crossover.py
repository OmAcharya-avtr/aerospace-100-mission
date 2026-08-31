"""How the four disturbance torques scale with altitude, and where aerodynamic torque
falls below solar radiation pressure.

Produces ``screenshots/altitude_sweep_and_crossover.png``: peak and secular torque
against altitude on log axes, the momentum accumulated per orbit, and the density
profile that drives the aerodynamic column.

Run from ``examples/``:  ``python3 altitude_sweep_and_crossover.py``
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from disturbtorque import (  # noqa: E402
    SOURCES,
    compute_profile,
    density,
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


def sweep(sc, alts, n_samples=241):
    peak = {s: [] for s in SOURCES}
    dh = {s: [] for s in SOURCES}
    for alt in alts:
        orb = reference_orbit(float(alt))
        sun = sun_direction_for_beta(orb.inclination_rad, orb.raan_rad, 0.0)
        prof = compute_profile(sc, orb, sun, n_samples=n_samples)
        for s in SOURCES:
            peak[s].append(prof.peak_magnitude(s, "body"))
            dh[s].append(float(np.linalg.norm(momentum_accumulation(prof, s, "eci")[-1])))
    return {s: np.array(v) for s, v in peak.items()}, {s: np.array(v) for s, v in dh.items()}


def main() -> None:
    sc = reference_smallsat()
    alts = np.arange(300.0, 1001.0, 25.0)
    peak, dh = sweep(sc, alts)

    def gap(alt_km: float) -> float:
        orb = reference_orbit(alt_km)
        sun = sun_direction_for_beta(orb.inclination_rad, orb.raan_rad, 0.0)
        prof = compute_profile(sc, orb, sun, n_samples=181)
        return prof.peak_magnitude("aerodynamic", "body") - prof.peak_magnitude("solar", "body")

    cross = brentq(gap, 400.0, 700.0, xtol=0.05)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.2))

    ax = axes[0]
    for s in SOURCES:
        ax.semilogy(alts, peak[s], color=COLOURS[s], lw=1.8, label=LABELS[s])
    ax.axvline(cross, color="0.35", ls="--", lw=1.1)
    ax.annotate(f"aero = solar\nat {cross:.0f} km", xy=(cross + 18, 1.2e-5), fontsize=9,
                color="0.3")
    ax.axhspan(1e-7, 1e-4, color="0.5", alpha=0.10, lw=0)
    ax.text(305, 1.15e-7, "quoted 1e-7 to 1e-4 N m band", fontsize=8, color="0.35")
    ax.set_xlabel("altitude [km]")
    ax.set_ylabel("peak torque magnitude, body frame [N m]")
    ax.set_title("Peak torque vs altitude", fontsize=11)
    ax.legend(fontsize=8.5, loc="lower left")
    ax.grid(alpha=0.25, which="both")

    ax = axes[1]
    for s in SOURCES:
        ax.semilogy(alts, dh[s], color=COLOURS[s], lw=1.8, label=LABELS[s])
    ax.semilogy(alts, sum(dh[s] for s in SOURCES), color="k", ls="--", lw=1.4,
                label="sum of magnitudes")
    ax.set_xlabel("altitude [km]")
    ax.set_ylabel(r"$|\int T\,dt|$ over one orbit, ECI [N m s]")
    ax.set_title("Momentum accumulated per orbit", fontsize=11)
    ax.legend(fontsize=8.5, loc="lower left")
    ax.grid(alpha=0.25, which="both")

    ax = axes[2]
    h = np.linspace(150e3, 1000e3, 2000)
    ax.semilogx(density(h), h / 1000.0, color="#444444", lw=1.8)
    ax.axhline(cross, color="0.35", ls="--", lw=1.1)
    ax.set_xlabel(r"density [kg m$^{-3}$]")
    ax.set_ylabel("altitude [km]")
    ax.set_title("Piecewise-exponential density\n(no solar-activity dependence)",
                 fontsize=11)
    ax.grid(alpha=0.25, which="both")

    fig.suptitle(
        "Reference 100 kg smallsat, circular orbit, i = 51.6 deg, beta = 0 deg, "
        "nadir-pointing with 5 deg pitch and roll",
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = OUT / "altitude_sweep_and_crossover.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)

    print(f"wrote {path}")
    print(f"aerodynamic and solar peak torques are equal at {cross:.1f} km")
    print(f"{'alt [km]':>9}" + "".join(f"{LABELS[s][:13]:>15}" for s in SOURCES))
    for i, alt in enumerate(alts):
        if alt % 100 == 0:
            print(f"{alt:>9.0f}" + "".join(f"{peak[s][i]:>15.4e}" for s in SOURCES))


if __name__ == "__main__":
    main()
