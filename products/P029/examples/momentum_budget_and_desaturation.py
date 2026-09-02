"""Disturbance torques, momentum accumulation, and magnetic desaturation authority.

Run: ``python3 momentum_budget_and_desaturation.py``
Writes ``../screenshots/momentum_budget_and_desaturation.png`` and prints the budget.

Four panels for the reference smallsat at 500 km, i = 51.6 deg, beta = 20 deg:
top left, the four disturbance torques over one orbit in ECI; top right, the momentum
each one accumulates; bottom left, the momentum a smallsat wheel set can hold against
what it collects per orbit; bottom right, the fraction of the wheel momentum that no
magnetorquer dipole can remove at each point of the orbit, which is the constraint the
scheduler has to work around.
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import sys  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from momentummgr import (  # noqa: E402
    SOURCES,
    averaged_controllability,
    momentum_budget,
    momentum_history_eci,
    momentum_per_orbit_eci,
    pyramid_four,
    reference_orbit,
    reference_smallsat,
    sun_direction_for_beta,
    sweep_orbit,
    thruster_dump,
    uncontrollable_fraction,
)

OUT = pathlib.Path(__file__).resolve().parents[1] / "screenshots"
OUT.mkdir(exist_ok=True)

sc = reference_smallsat()
orbit = reference_orbit(500.0)
sun = sun_direction_for_beta(orbit.inclination_rad, orbit.raan_rad, np.radians(20.0))
sweep = sweep_orbit(sc, orbit, sun, n_samples=1441)
budget = momentum_budget(sc, orbit, sun, n_samples=1441)
wheels = pyramid_four(max_momentum_nms=0.05)
envelope = wheels.guaranteed_body_envelope_nms

print(f"Reference smallsat, 500 km, i = 51.6 deg, beta = 20 deg, period "
      f"{orbit.period_s:.1f} s")
print(f"{'source':<18}{'|dh| per orbit [N m s]':>24}{'cyclic peak [N m s]':>22}")
print("-" * 64)
for name in (*SOURCES, "total"):
    print(f"{name:<18}{budget[name]['secular_per_orbit_nms']:>24.6e}"
          f"{budget[name]['cyclic_peak_nms']:>22.6e}")
secular = budget["total"]["secular_per_orbit_nms"]
orbits_to_fill = envelope / secular
print(f"\nwheel array envelope        {envelope:.6f} N m s (4 wheels at 0.05 N m s)")
print(f"orbits to fill on secular   {orbits_to_fill:.2f}")
dump = thruster_dump(envelope, 0.5, 220.0)
print(f"thruster propellant to dump a full envelope: {dump.propellant_kg:.4e} kg "
      f"(couple, 0.5 m arm, Isp 220 s)")

hours = sweep.time_s / 3600.0
fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.6))

ax = axes[0, 0]
for name in SOURCES:
    ax.plot(hours, np.linalg.norm(sweep.torque(name, "eci"), axis=1) * 1e6, lw=1.4, label=name)
ax.plot(hours, np.linalg.norm(sweep.torque("total", "eci"), axis=1) * 1e6, "k--", lw=1.6,
        label="total")
ax.set_xlabel("time from the ascending node [h]")
ax.set_ylabel(r"|T| [$\mu$N m]")
ax.set_title("Disturbance torque magnitude over one orbit (ECI)")
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.3)

ax = axes[0, 1]
for name in SOURCES:
    ax.plot(hours, np.linalg.norm(momentum_history_eci(sweep, name), axis=1) * 1e3, lw=1.4,
            label=name)
ax.plot(hours, np.linalg.norm(momentum_history_eci(sweep, "total"), axis=1) * 1e3, "k--",
        lw=1.6, label="total")
ax.set_xlabel("time from the ascending node [h]")
ax.set_ylabel("|h| accumulated [mN m s]")
ax.set_title("Momentum accumulated, per source")
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.3)

ax = axes[1, 0]
names = [*SOURCES, "total"]
sec = [budget[n]["secular_per_orbit_nms"] * 1e3 for n in names]
cyc = [budget[n]["cyclic_peak_nms"] * 1e3 for n in names]
x = np.arange(len(names))
ax.bar(x - 0.2, sec, 0.4, label="secular per orbit")
ax.bar(x + 0.2, cyc, 0.4, label="cyclic peak within the orbit")
ax.axhline(envelope * 1e3, color="crimson", ls="--", lw=1.6,
           label=f"wheel envelope {envelope * 1e3:.1f} mN m s")
ax.set_xticks(x)
ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
ax.set_ylabel("momentum [mN m s]")
ax.set_yscale("log")
ax.set_title(f"Storage against accumulation: {orbits_to_fill:.1f} orbits to fill")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")

ax = axes[1, 1]
h_dirs = {"along ECI x": [1.0, 0.0, 0.0], "along ECI z": [0.0, 0.0, 1.0],
          "along the orbit normal": None}
h_norm_dir = np.cross(sweep.r_eci[0], sweep.r_eci[len(sweep.r_eci) // 4])
h_dirs["along the orbit normal"] = list(h_norm_dir / np.linalg.norm(h_norm_dir))
for label, direction in h_dirs.items():
    d = np.repeat(np.array(direction)[None, :], sweep.r_eci.shape[0], axis=0)
    ax.plot(hours, uncontrollable_fraction(d, sweep.b_eci_t), lw=1.4, label=label)
_, eig, _ = averaged_controllability(sweep.b_eci_t, sweep.time_s)
ax.set_xlabel("time from the ascending node [h]")
ax.set_ylabel(r"$|h\cdot\hat{B}| \, / \, |h|$")
ax.set_ylim(0.0, 1.05)
ax.set_title("Fraction of the momentum no dipole can remove\n"
             f"averaged Gramian eigenvalues {eig[0]:.3f}, {eig[1]:.3f}, {eig[2]:.3f}")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

fig.suptitle("momentummgr: disturbance momentum budget and magnetic desaturation authority",
             fontsize=12)
fig.tight_layout()
path = OUT / "momentum_budget_and_desaturation.png"
fig.savefig(path, dpi=130)
print(f"\nwrote {path}")
print(f"total dh over one orbit (ECI) = "
      f"{np.round(momentum_per_orbit_eci(sc, orbit, sun, 'total'), 12).tolist()} N m s")
