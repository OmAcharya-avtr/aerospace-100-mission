# Changelog

## 0.1.0 — 2026-08-31

Initial release.

- Four environmental disturbance-torque models, each with its source, units, assumptions
  and validity range in the docstring: gravity gradient `3 mu/R^3 u x (I u)` with the
  planar `sin 2 theta` form and its 45-degree analytic maximum; free-molecular
  aerodynamic `-1/2 rho Cd A |v| v` crossed with the centre-of-pressure offset; solar
  radiation pressure `(Phi/c) A (1+q) / d^2` anti-sunward with a cylindrical eclipse
  test; and residual magnetic dipole `m x B`.
- Supporting environment models, each with its validity stated: a 28-band
  piecewise-exponential atmosphere (0-1000 km, no solar-activity dependence, checked for
  band-boundary continuity to 9.59e-05 relative above 25 km); a centred non-tilted
  geomagnetic dipole with an exact closed form for its orbit average
  `<B> = (k/R^3)[z_hat - (3/2) sin(i) Q_hat]`; a low-precision Sun direction and distance;
  and a cylindrical umbra model with its closed-form eclipse fraction.
- Circular-orbit sweep in ECI, LVLH and body frames with fixed 3-2-1 pointing offsets
  from nadir; secular/cyclic decomposition; and momentum accumulation reported in both
  the inertial and the body frame, with the frame caveat stated in the docstring.
- Constants carry their provenance and their disagreements: the solar constant is 1361
  W m^-2 with the older textbook 1367 W m^-2 exported alongside it (0.441 % apart), and
  the Earth reduced dipole moment is the textbook 7.96e15 T m^3 with the 2.6 % spread
  against an IGRF-epoch reduction documented rather than averaged.
- Input validation with actionable messages: inertia tensors are checked for symmetry,
  positive definiteness and the rigid-body triangle inequality; direction vectors must be
  unit; reflectance must lie in [0, 1]; altitudes outside the density model's 0-1000 km
  range are rejected unless extrapolation is asked for explicitly.
- CLI: `python -m disturbtorque budget --altitude-km 500 [--beta-deg ...] [--json]` and
  `python -m disturbtorque sweep --altitude-km 400 500 600`.
- Level-2 validation with saved raw output (`validation/`): all four expressions against
  hand arithmetic; the gravity-gradient maximum located at 45.000000 deg; momentum per
  orbit against two exact closed forms and against a QUADPACK reference that never
  touches the sample grid (0.033 % at the default 721 samples); the atmosphere table's
  own continuity; the Sun model against perihelion, aphelion and the solstice
  declinations; the eclipse fraction against its closed form. One check is reported as
  FAILED and not tuned: the aerodynamic torque for the reference smallsat falls below the
  quoted 1e-7 N m band floor above 700 km, by factors of 1.54, 4.85 and 19.37 at 700, 800
  and 1000 km.
- 81 pytest tests (known-answer with hand arithmetic in comments, input validation,
  Hypothesis property tests, integration including the CLI in a child process, and pinned
  regression and benchmark tests); ruff-clean at line length 100.
- Two runnable examples writing PNGs to `screenshots/` with the Agg backend.
