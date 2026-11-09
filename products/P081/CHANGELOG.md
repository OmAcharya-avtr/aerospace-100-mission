# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-11-09

First release. Status: TESTING. Validation level 3, research grade.

### Added

- `twindiverge.plant`: the declared twin. A single-state reaction-wheel model,
  equation (1), with a Coulomb-plus-viscous-plus-quadratic friction
  decomposition whose FORM is cited and whose coefficients are declared; the
  forward sensitivity equations (3) and (5) integrated alongside the state in
  logarithmic parameters; and the two exact identities the whole product rests
  on, derived on paper in the module docstring. Identity A, equation (6): the
  five rate sensitivities sum to zero at every sample, for every excitation and
  at every parameter point, so a rate record cannot see the absolute torque
  scale. Identity B, equation (7): the five torque sensitivities sum to the
  torque signal itself, so a reaction-torque channel can.
- `twindiverge.reference`: an independent closed-form solution of equation (1)
  over a constant-current segment, equations (20) and (21), used by validation
  only and by nothing in the package. The integrator agrees with it to a worst
  relative difference of 6.615096e-15.
- `twindiverge.telemetry`: telemetry sets — a campaign, its channels and their
  declared noise levels — and the noise-scaled sensitivity matrix of equation
  (8). Five declared sets whose measured ranks are 4, 5, 4, 5 and 3 of 5.
- `twindiverge.identifiability`: **baseline 1, the gate.** Numerical rank from
  the singular values with a declared relative cutoff and a reported margin in
  decades; the Fisher information of equation (9), formed and reported but never
  computed from; Cramer-Rao bounds with the finiteness clause of equation (10),
  so a parameter with a component in the null space gets `+inf` and not a
  number; the identifiable combinations of equation (11) reported in its place;
  practical identifiability separated from structural; and the pseudo-inverse
  diagonal reported beside the bound under the name
  `pinv_diagonal_not_a_bound`.
- `twindiverge.attribution`: **baseline 2.** Generalised least squares on the
  identifiable subspace only, equations (12) and (13), with Wald intervals per
  identifiable direction; the first-order sensitivity attribution of equation
  (14) that the specification names; the constrained single-parameter maximum
  likelihood of equation (15) with its normalised likelihood weights (16),
  which measures stronger on the reference set; a chi-square detection test
  that must pass before anything is attributed; and
  `equivalence_family`, which prints the perturbations the telemetry cannot
  distinguish from its own answer.
- `twindiverge.profile`: **baseline 3.** The nonlinear profile likelihood of
  equation (17) on a grid sized from the linearised width, with centre-out
  continuation, a declared inner box, square-root interpolation of the threshold
  crossings, separate lower and upper limits because the profile is routinely
  one-sided, and the verdict `overrides_covariance` when a flat profile meets a
  finite Wald half-width.
- `twindiverge.features`: 16 declared residual-signature features, chosen so
  that the sensitivity projections — the sufficient statistics of the linearised
  problem — are deliberately NOT among them.
- `twindiverge.learned`: a gradient-boosted classifier with declared
  hyperparameters, two declared variants (five classes and a six-class ablation
  with a refusal class), and **no point-prediction method**: every prediction
  carries the whole class-probability vector, the confidence, the margin and an
  abstention flag.
- `twindiverge.synthetic`: five declared case kinds, all deterministic from
  integer seeds, including the `equivalent` kind whose members produce telemetry
  identical to a single-parameter fault.
- `twindiverge.pipeline`: the ordered path — identifiability, then detection,
  then attribution — and the learned study that scores five methods on five
  populations.
- `twindiverge.scoring`: exact Clopper-Pearson intervals, a multi-class Brier
  score, a reliability table, and the confident-error rate that the
  unidentifiable-subspace measurement needs.
- `twindiverge.report`: plain-text rendering that prints the standing of every
  number beside it.
- A CLI with `telemetry`, `identify`, `nullspace`, `attribute`, `profile`,
  `crb`, `sweep` and `learned`.
- Twelve validation scripts with their raw committed output, seven example
  figures, and 254 tests including Hypothesis property tests, nine pinned
  regressions and 29 negative controls.

### Measured and reported as found

- **The learned classifier returns a confident wrong attribution exactly where
  the analytic method refuses.** On 120 held-out cases inside the unidentifiable
  subspace: analytic refusal rate 1.000000 and ambiguity declared 1.000000
  against a learned mean confidence of 0.993426, accuracy 0.000000,
  confident-and-wrong 0.991667 (exact interval [0.954444, 0.999789]) and
  ambiguity declared 0.000000. This is the product's headline and nothing was
  tuned to soften it.
- **The analytic attributor beats the learned classifier on both answerable
  populations**: 1.000000 against 0.991667 on ordinary single-parameter faults,
  and 0.941667 against 0.641667 on faults a tenth the size and outside the
  training distribution.
- On the reference rate-only set, **all five parameters are refused**, the
  computed null space agrees with its on-paper derivation to a worst principal
  angle of 0.000000e+00 degrees, and the identifiable subspace is measured to BE
  the space of the four ratios `k_t/J`, `T_c/J`, `b/J`, `c_d/J` to within
  1.207418e-06 degrees.
- **Five of five parameter profiles are flat** on that set, at 7.385097e-10 in
  chi-square, while the linearised covariance claims half-widths between 0.7 and
  11 per cent. The profile overrides the covariance five times out of five.
  Where the profiles do close, on the two-channel set, the covariance is
  validated against them to 0.9972, 0.9977, 1.0024 and 1.1622.
- The Cramer-Rao bound lies below the measured scatter of 20 nonlinear fits in
  every direction on both full-rank-in-practice sets, and the linear estimator's
  ratios sit on the bound at 1.002398 to 1.209669, so the bound is tight as well
  as true. The nonlinear global fit's ratios reach 57.008044 because it is
  biased on a problem this ill-conditioned, and that is reported rather than
  smoothed over.
- **The specified attributor of equation (14) is not uniformly weaker than
  equation (15)**: equation (15)'s margin runs from -0.025000 on
  `rate_torque_short` to +0.000000 on the reference set. Both are printed on
  every case.
- **The six-class ablation's refusal class does not generalise.** Trained with
  pure-scale examples, it reaches 1.000000 on them and 0.008333 on the ambiguity
  population, and costs 0.316667 accuracy on small faults because it absorbs
  them.
- Structural rank is constant at 4 over record lengths from 5 s to 200 s while
  the bound on the worst identifiable direction falls by a factor of 347189.0
  (10 s to 200 s: 30979.4).
- The `pytest.approx` audit finds **0 of 254** assertions depending on the
  hidden 1e-12 absolute default and **2 of 254** resting on an absolute bound,
  both of them comparisons against an exact zero.

### Found during the build and fixed

- **The zero-order-hold current was evaluated at each Runge-Kutta stage's own
  time.** The fourth stage of the step ENDING at the current switch then sees
  the new current, which costs three orders of convergence at that step: the
  measured global error at the end of a 20 s driven segment was 4.102564e-02
  rad/s at a 0.1 s step and the measured convergence order over steps from 0.4 s
  to 0.025 s was **1.0000** instead of 4. The current is now sampled once per
  step at the step's left endpoint, which is also the physically correct reading
  of a zero-order hold, and the measured orders are 4.0201 to 4.2425. Pinned in
  `tests/test_regressions.py`.
- **`scipy.optimize.least_squares` started from a point of order 1e-17 stops
  without moving.** Measured against scipy 1.18.1: it terminates on its `xtol`
  test after 23 function evaluations and returns a chi-square 3151 above the
  minimum it reaches from exactly zero. The failure band was mapped by
  bisection — 0.0 converges, 1e-20 and 1e-17 fail, 1e-14 and everything larger
  converge — and it occurs with and without bounds and under two `x_scale`
  settings. It matters here because a start of order 1e-17 is exactly what the
  profile's centre-out continuation hands the next grid point when the profile
  is flat, which is the case this product exists to measure. Warm starts below a
  declared floor of 1e-12 are now snapped to exactly zero. Pinned.
- **The threshold crossing was interpolated linearly in chi-square.** On a
  profile reaching 1.8e5 at the grid edge that underestimated the 95 per cent
  crossing by a factor of 216.47. Interpolation is now linear in the square root
  of the chi-square excess, which is exact for the leading quadratic behaviour.
  Pinned with both arithmetics written out.
- **The profile grid was fixed at the declared half-width**, which asked the
  inner optimisation to compensate a parameter excursion thousands of standard
  deviations wide on well-determined directions; it could not do so inside the
  declared box and returned an upper bound on the profile rather than the
  profile. The grid is now sized from the linearised width, and a limit
  bracketed by a box-truncated grid point is suppressed rather than quoted.
- **The unbounded inner optimisation left the plant's validity window in one
  step** and sent the integrator to a rate of order 1e199. A declared box of 0.5
  log-parameter units now bounds it, and the number of grid points that hit the
  box is reported per profile.
- **Equations (14) and (15) divided by a squared column norm that is exactly
  zero** for a parameter the excitation never excites — `k_t` on the coast
  campaign — and `numpy.where` evaluates both branches, so the division happened
  and warned before its result was discarded. Pinned as a
  no-warnings regression test.
- **A Hypothesis counterexample**, `headers=[''], rows=0`: `format_table`
  renders as the single string `"\n"`, whose `splitlines` has length 1 and not
  2, because `splitlines` drops a trailing empty segment. The table was right
  and the property that counted its lines was wrong. Pinned with the literal
  input.
- **Two negative controls could not fire in their first form and were replaced
  rather than accepted.** The bound control shrank the measured estimates by a
  factor of 0.5, which an estimator whose spread is already sixty times the
  bound survives; it is now anchored on the bound itself. The coverage control
  compared a JOINT coverage over four directions against a nominal 0.95, which
  made correct intervals look broken; it is now marginal, and the realisation
  count was raised from 200 to 400 because a coverage saturating at 1.000000 is
  only 3.2444 standard errors from the nominal level at 200 and the "interval too
  wide" control could not have reached the declared four.
- **The first version of the sensitivity cross-check used central differences
  alone** and failed at 1.009055e-08 relative on the weakest column, which is
  the difference's own cancellation floor and not a defect in the
  sensitivities. It is now checked against a complex-step derivative from an
  independently written complex-arithmetic integrator — exact to 4.304265e-15,
  because the right-hand side is a polynomial and carries no cancellation — with
  the central difference reported beside it against its measured floor.

### Found after hand-back, by the coordinating session

- **`probe_working_directories.py` created the failure it reported.** It
  dispatched all 44 script-directory pairs into a two-worker pool, so two
  copies of the same script could overlap. `validate_examples.py` unlinks each
  shared `screenshots/*.png` and regenerates it, so two copies race: one
  deletes a figure the other is about to measure. Measured 1 non-zero exit in
  44 runs — `validate_examples.py` from the repository root — not reproducible
  running that script alone from that directory, where it exits 0. Each
  script's four directory runs are now grouped onto one worker and run
  serially, while different scripts still run two at a time, so the wall-clock
  cost is unchanged. The probe also reported only `stderr`, where these
  scripts report failed checks on `stdout`, so the one real failure it ever
  caught printed no detail; it now falls back to `stdout`. Re-run twice from a
  clean tree: 0 of 44 and 0 of 14 both times, with byte-identical status
  tables. Recorded in VALIDATION.md section 10.1.

### Known limitations at 0.1.0

Listed in full in `README.md` under Limitations and in
`docs/REQUIREMENTS.md` section 10. The ones that bear on how a result should be
read: every verdict is LOCAL and first-order and is not global structural
identifiability; the plant is one declared fictional single-state wheel valid
for one sign of rotation inside a 20 to 400 rad/s window; the noise is
independent between samples by assumption; baseline 2's linearisation error is
second order and at an 8 per cent perturbation exceeds its own statistical
interval in the two best-determined directions; the sparsity assumption behind
equation (15) is not testable from the telemetry; both ends of every dependency
range were run for `ruff` only; and the `git ls-files` hygiene check examined 0
files at build time, against an uncommitted tree, where it was recorded as
vacuous rather than as a pass. That last one is now resolved: committing the
product made the check real and it examines 81 files with 0 offenders.
