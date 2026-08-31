"""Self-consistency of the piecewise-exponential atmosphere table.

Each band k is rho0_k exp(-(h - h0_k)/H_k). If the table has been transcribed
correctly, evaluating band k at the base altitude of band k+1 must reproduce band k+1's
own base density, because the published table is constructed to be continuous. Any
single mistyped digit in any of the 84 numbers breaks that at one boundary. This is the
only check available for the table itself without an external data source, and it is
run because the table was transcribed rather than copied from a machine-readable file.

Run: ``python3 atmosphere_table_continuity.py``
"""

from __future__ import annotations

import numpy as np

from _common import Checks  # noqa: E402

from disturbtorque.atmosphere import EXPONENTIAL_TABLE, density  # noqa: E402

c = Checks()
print("Piecewise-exponential atmosphere: band-boundary continuity")
print("=" * 84)
print(f"{'h0 [km]':>9}{'h1 [km]':>9}{'rho0 [kg/m3]':>16}{'H [km]':>10}"
      f"{'predicted rho(h1)':>20}{'tabulated rho0':>17}{'rel':>10}")
print("-" * 91)
worst = 0.0
for (h0, r0, hs), (h1, r1, _) in zip(EXPONENTIAL_TABLE[:-1], EXPONENTIAL_TABLE[1:]):
    pred = r0 * np.exp(-(h1 - h0) / hs)
    rel = abs(pred - r1) / r1
    worst = max(worst, rel)
    print(f"{h0:>9.1f}{h1:>9.1f}{r0:>16.4e}{hs:>10.3f}{pred:>20.6e}{r1:>17.6e}{rel:>10.1e}")

print()
print("The 0-25 km band is a single coarse fit across the troposphere and lower")
print("stratosphere, so its endpoint mismatch is larger than the rest; every band above")
print("25 km closes to better than 1e-4 relative.")
c.check("worst relative mismatch over all 27 boundaries", worst, 0.0, 2e-3, kind="abs")
above25 = max(
    abs(r0 * np.exp(-(h1 - h0) / hs) - r1) / r1
    for (h0, r0, hs), (h1, r1, _) in zip(EXPONENTIAL_TABLE[1:-1], EXPONENTIAL_TABLE[2:])
)
c.check("worst relative mismatch above 25 km (26 boundaries)", above25, 0.0, 1e-4, kind="abs")

print("\nSpot values used elsewhere in this validation set:")
for h_km in (300.0, 400.0, 500.0, 600.0, 700.0, 800.0):
    print(f"  rho({h_km:6.1f} km) = {float(density(h_km * 1000.0)):.6e} kg m^-3")

print("\nMonotonicity and range behaviour:")
h = np.linspace(0.0, 1_000_000.0, 200001)
rho = density(h)
rises = np.where(np.diff(rho) >= 0.0)[0]
print(f"  non-monotonic steps on a 5 m grid over 0-1000 km: {len(rises)}")
for i in rises:
    print(
        f"    at {h[i] / 1000:.3f} -> {h[i + 1] / 1000:.3f} km, "
        f"rho rises from {rho[i]:.6e} to {rho[i + 1]:.6e} "
        f"({100 * (rho[i + 1] / rho[i] - 1):.4f} %)"
    )
print("""
  This is a real property of the published table, not a coding defect: the 0-25 km band
  is a coarse fit whose value at 25 km falls 0.14 % below the 25-30 km band's own base
  density, so the piecewise function steps up by 0.067 % at that single boundary. It is
  reported rather than smoothed away. The aerodynamic torque model is only valid in
  free-molecular flow, above roughly 150 km, where the table is strictly decreasing.
""")
c.assert_true(
    "density is strictly decreasing above 25 km (the orbital range)",
    bool(np.all(np.diff(rho[h >= 25_000.0]) < 0.0)),
)
c.assert_true(
    "exactly one non-monotonic step over 0-1000 km, at the 25 km boundary",
    len(rises) == 1 and abs(h[rises[0]] - 25_000.0) < 10.0,
    f"(found {len(rises)})",
)
c.assert_true("density is strictly positive everywhere", bool(np.all(rho > 0.0)))
try:
    density(1_000_001.0)
    c.assert_true("above 1000 km raises without allow_extrapolation", False)
except ValueError as exc:
    c.assert_true("above 1000 km raises without allow_extrapolation", True, f"({exc.args[0][:44]}...)")
try:
    density(-1.0)
    c.assert_true("negative altitude raises", False)
except ValueError:
    c.assert_true("negative altitude raises", True)

c.summary("atmosphere_table_continuity.py")
raise SystemExit(1 if c.n_fail else 0)
