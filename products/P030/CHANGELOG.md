# Changelog

All notable changes to KeepOut are recorded here.

## 0.1.0 — 2026-08-31

Initial release.

- `geometry`: unit-vector helpers, numerically stable `angular_separation`
  (`atan2` of the cross-product norm, not `arccos` of the dot product),
  Rodrigues rotations, Haar-uniform random rotations, spherical-cap solid
  angle, the closed-form two-cap intersection and union solid angles, and a
  Fibonacci sphere lattice.
- `bodies`: WGS-84, IAU and IAG constants; `angular_radius(R, d) =
  arcsin(R/d)`; `earth_angular_radius(h) = arcsin(R_E/(R_E+h))`; Julian date;
  and the low-precision Sun and Moon direction series of Vallado (2013)
  Algorithms 29 and 31, provided as a convenience with no accuracy claim.
- `cones`: `ExclusionCone` with signed clearance margins and a strict
  interior test; `body_exclusion_cone` with an explicit limb-or-centre
  convention, so a 10 deg instrument angle at 550 km becomes a 77.016 deg cone
  rather than a 10 deg one; `KeepOutSet` with per-cone margins, a worst-case
  margin, and violations reported deepest-first.
- `regions`: allowed-direction masks, an allowed solid angle by band
  quadrature that is exact in azimuth and Gauss-Legendre in `cos(theta)`
  (worst error 2.7e-11 sr against the two-cap closed form over 300 random
  configurations), and an independent Monte Carlo estimator with a binomial
  standard error.
- `windows`: circular Keplerian orbit positions, worst-case margin series for
  a fixed inertial target, and pointing windows with boundaries refined by
  Brent's method to |margin| below 1e-10 rad.
- Validation Level 1: cap areas against adaptive quadrature (worst 1.8e-15 sr),
  twelve exactly-known cap-intersection cases (worst 4.4e-16 sr), the
  Earth angular radius bit-identical to `arcsin(R_E/(R_E+h))` at 19 altitudes
  and within 1.4e-14 rad of a numerical maximisation that uses no tangency
  identity, a hand-computed two-cone case reproduced to 6.7e-16 sr, and
  margin invariance under 20 000 random rotations to 1.1e-15 rad.
- Two example figures (sky map with the allowed region, pointing-window
  timeline), 159 passing tests including Hypothesis property tests,
  ruff-clean.
