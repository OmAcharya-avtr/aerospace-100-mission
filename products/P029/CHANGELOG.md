# Changelog

## 0.1.0 — 2026-09-02

Initial release.

- **Independent disturbance-torque and momentum model.** Gravity-gradient
  `3n² û × (I û)`, free-molecular aerodynamic `−½ρCdA|v|v`, solar radiation pressure
  `(Φ/c)A(1+q)/d²` with a cylindrical umbra, and residual dipole `m × B`, each with its
  source, units, assumptions and validity range in its docstring. Supporting models: the
  28-band Vallado exponential atmosphere, a centred geomagnetic dipole with optional tilt
  and Earth rotation, and closed-form eclipse boundaries and eclipse fraction.
- **Two quadratures, on purpose.** `momentum_per_orbit_eci` uses Gauss-Legendre in
  argument of latitude with the solar term split at the analytic eclipse edges;
  `momentum_history_eci` uses the cumulative trapezoid a time-stepped simulation would.
  The difference is the discretisation error a scheduler inherits and is reported, not
  hidden: 3.51e−03 relative on the solar term at the default 721 samples, and non-monotone
  in the sample count.
- **Cross-check against P027 `disturbtorque`**, required by the batch specification and
  implemented independently from the cited physics with no shared code. The total momentum
  vector over one orbit agrees to **1.8e−11 relative** and the solar vector to 1.2e−11
  against P027's QUADPACK reference. The apparent 1.9e−05 difference against P027's
  *sampled* table row is P027's own documented eclipse-edge quadrature error, and this
  package's value is six orders of magnitude closer to P027's reference than P027's
  sampled row is.
- **Reaction-wheel array algebra.** Redundant allocation with an **exact** null-space
  maximiser of the smallest wheel momentum (enumerated breakpoints, no line search, never
  beaten by a 20001-point scan), a closed-form conservative body envelope
  `h_max / maxᵢ‖(A⁺)ᵢ‖` = (4/3)h_max for the isotropic pyramid, wheel speeds, saturation
  fraction and zero-crossing counting. Pyramid, tetrahedral and orthogonal presets.
- **Desaturation with the controllability constraint made explicit.** The cross-product
  magnetic dumping law with magnitude limiting, the instantaneous uncontrollable fraction
  `|h·B̂|/|h|`, the time-averaged controllability Gramian `⟨I − B̂B̂ᵀ⟩` with its exactly-2
  trace and its eigenvalues, dipole cost `∫|m|dt`, and impulsive thruster dumping with
  impulse and propellant.
- **Desaturation scheduling.** Episodes of six orbits in 600 s decision windows, the
  wheel Euler equation integrated by Heun's method (explicit Euler was implemented first
  and rejected: it overstated the baseline's magnetorquer duty by 49 % at the default
  step), a tuned fixed-threshold baseline with hysteresis, an offline schedule search, and
  a learned scheduler with a confidence output and a classical fallback.
- **AI benchmark with intervals.** On 80 held-out episodes the learned scheduler uses
  20.7 % less magnetorquer duty (95 % CI [−0.0188, −0.0107]) and spends more time near
  saturation (+0.0035, CI [+0.0004, +0.0088]); the combined-cost difference is reported as
  **indistinguishable** at saturation weights of 2 and 4, where the interval contains
  zero. Confidence calibration is measured and is imperfect (Brier 0.039862 against
  0.040947 for a base-rate predictor; overconfidence up to +0.112).
- **Level-2 validation with saved raw output** (`validation/`): 88 checks, 88 passed. Two
  expectations that failed as first written are recorded as findings rather than removed —
  the uncontrollable fraction is not monotone in inclination (minimum 0.4627 at 51.6°,
  rising to 0.6367 at 90°), and the first-order integrator was wrong. One known defect is
  reported and not fixed: the null-space biasing is discontinuous, with a largest
  single-sample wheel-momentum step of 0.095112 N m s against 0.000060 N m s for
  minimum-norm allocation.
- **109 pytest tests** (known-answer with the hand arithmetic in the comments, input
  validation, Hypothesis property tests, wheel and desaturation unit tests, scheduler
  tests, integration including the CLI in a child process, and pinned regression tests);
  ruff-clean at line length 100.
- **CLI**: `python -m momentummgr budget|controllability|schedule`, each with `--json`.
- Three runnable examples writing PNGs to `screenshots/` with the Agg backend.
