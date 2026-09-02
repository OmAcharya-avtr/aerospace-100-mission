"""Wheel-level allocation: what null-space biasing buys and what it costs.

Run: ``python3 wheel_zero_speed_avoidance.py``
Writes ``../screenshots/wheel_zero_speed_avoidance.png``.

The reference smallsat runs for three orbits with no desaturation. The body momentum is
propagated from the wheel Euler equation and allocated to a four-wheel isotropic pyramid
three ways: minimum norm, biased inside the full envelope, and biased inside 70 % of it.
The bottom row shows the same trajectory on a three-wheel orthogonal array, which has no
null space and therefore no choice at all.

Look at the vertical jumps in the two biased panels. They are real and they are a
limitation of the method as implemented: the null coefficient is chosen afresh at every
sample with no memory, and the maximiser of ``min |h_i|`` can switch between symmetric
branches, which asks the wheels for a momentum step no real torque can deliver. The
largest such step is printed below. A flight implementation must rate-limit or hysteresis
the null coefficient; this package deliberately does not hide the raw behaviour behind
one.
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from momentummgr import (  # noqa: E402
    count_zero_crossings,
    orthogonal_three,
    pyramid_four,
    reference_orbit,
    reference_smallsat,
    sun_direction_for_beta,
    sweep_orbit,
)

OUT = pathlib.Path(__file__).resolve().parents[1] / "screenshots"
OUT.mkdir(exist_ok=True)

sc = reference_smallsat()
orbit = reference_orbit(500.0)
sun = sun_direction_for_beta(orbit.inclination_rad, orbit.raan_rad, np.radians(20.0))
n_per_orbit = 721
sweep = sweep_orbit(sc, orbit, sun, n_samples=n_per_orbit)
torque = np.vstack([sweep.torque("total", "body")[:-1]] * 3)
dt = orbit.period_s / (n_per_orbit - 1)
omega = orbit.body_rate_body_rad_s
i_omega = sc.inertia @ omega

h = np.zeros(3)
traj = [h.copy()]
for row in torque:
    h = h + dt * (row - np.cross(omega, i_omega + h))
    traj.append(h.copy())
traj = np.array(traj)
hours = np.arange(traj.shape[0]) * dt / 3600.0

w4 = pyramid_four(wheel_inertia_kg_m2=1.0e-3, max_momentum_nms=0.05)
w3 = orthogonal_three(wheel_inertia_kg_m2=1.0e-3, max_momentum_nms=0.05)
deadband = 0.05 * w4.max_momentum_nms

cases = [
    ("pyramid, minimum norm", w4, {"avoid_zero_speed": False}),
    ("pyramid, biased, full envelope", w4, {"avoid_zero_speed": True,
                                            "envelope_fraction": 1.0}),
    ("pyramid, biased, 70 % envelope", w4, {"avoid_zero_speed": True,
                                            "envelope_fraction": 0.7}),
    ("orthogonal three (no null space)", w3, {"avoid_zero_speed": True}),
]

fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.4), sharex=True)
print(f"{'allocation':<34}{'crossings':>12}{'dwell < 5 % h_max':>20}{'peak |h_i|/h_max':>20}"
      f"{'largest step [N m s]':>22}")
print("-" * 108)
for ax, (label, array, kwargs) in zip(axes.ravel(), cases, strict=True):
    hist = np.array([array.allocate(row, **kwargs).wheel_momentum_nms for row in traj])
    speeds = array.speeds_rad_s(hist)
    crossings = int(count_zero_crossings(hist).sum())
    dwell = float(np.mean(np.abs(hist) < deadband))
    peak = float(np.max(np.abs(hist)) / array.max_momentum_nms)
    jump = float(np.max(np.abs(np.diff(hist, axis=0))))
    print(f"{label:<34}{crossings:>12d}{dwell * 100:>19.2f}%{peak:>20.3f}{jump:>22.6f}")
    for k in range(array.n_wheels):
        ax.plot(hours, speeds[:, k], lw=1.2, label=f"wheel {k}")
    ax.axhspan(-deadband / array.wheel_inertia_kg_m2[0],
               deadband / array.wheel_inertia_kg_m2[0],
               color="crimson", alpha=0.12, label="low-speed band")
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.set_title(f"{label}\n{crossings} sign changes, {dwell * 100:.1f} % in the low-speed "
                 f"band, largest step {jump:.4f} N m s", fontsize=10)
    ax.set_ylabel(r"wheel speed [rad s$^{-1}$]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=3, loc="upper left")
for ax in axes[1]:
    ax.set_xlabel("time [h]")
fig.suptitle("momentummgr: null-space biasing keeps wheels out of the low-speed band "
             "(three orbits, no desaturation)", fontsize=12)
fig.tight_layout()
path = OUT / "wheel_zero_speed_avoidance.png"
fig.savefig(path, dpi=130)
print(f"\npeak |h_body| over three orbits {np.linalg.norm(traj, axis=1).max():.6f} N m s")
print("The 'largest step' column is the limitation named in the module docstring: the")
print("biased allocations jump when the memoryless maximiser switches branch. Minimum")
print("norm never does, because it is a continuous function of the request.")
print(f"wrote {path}")
