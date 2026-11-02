# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-11-02

First release. Status: TESTING. Validation level 3, research grade.

### Added

- `hilverdict.plant`: the declared loop. A two-state short-period plant with a
  cubic pitching moment, equations (1)-(2); a trim feedforward solved in closed
  form for the cubic plant, equations (4)-(5), so it does not degrade with
  amplitude; the zero-order-hold discretisation by Van Loan's block
  exponential, equation (7), which needs no matrix inverse; and the divergence
  metric of equation (6), whose weights ARE the declared engineering
  tolerances so that the verdict threshold is exactly 1 and nothing tunes it.
- `hilverdict.timeline`: the run log and the hold-last-value miss model, with
  both the direct and the cascade overrun accounting reported on every run
  because the two disagree by a factor of 2 to 3 on a loaded run.
- `hilverdict.certificate`: **baseline 1**. The closed-form bound of equation
  (13) built from Dahlquist's logarithmic-norm inequality (8), the hold
  injection (10), the free-settle factor (11) and the exact finite-horizon
  amplification (12); its closed-form inversion (14) into the single overrun
  threshold below which no miss needs looking at; the superposition bound (15);
  and the independent Gronwall factor (16) kept for cross-checking. Three
  declared conditions — actuator saturation, incidence beyond the declared
  validity limit, and mean utilisation at or above 1 — are **refused** rather
  than extrapolated through.
- `hilverdict.resimulate`: **baseline 2**, the reference. The declared
  nonlinear plant run with and without each miss, at a fixed declared
  Runge-Kutta step because an adaptive step would make the reference depend on
  the perturbation. Plant evaluations are counted so that "slow" is a
  measurement. An optional early exit is provided, off by default because its
  criterion rests on the same linearisation as baseline 1, and verified
  bit-identical to the full horizon over 228 misses.
- `hilverdict.clock`: host-to-device drift by least squares and by Theil-Sen,
  reset detection from the offset series against a robust scale, and the
  duration correction. A detected reset makes the run undecidable, because a
  duration spanning a reset is meaningless rather than biased.
- `hilverdict.features`: fourteen cheap log features, two of which are exactly
  the two numbers the certificate is a function of.
- `hilverdict.learned`: a gradient-boosted classifier with declared
  hyperparameters, an abstention band, and no point-prediction method.
- `hilverdict.scoring`: recall first, with the exact Clopper-Pearson interval.
- `hilverdict.verdict`: the clock, then the declared conditions, then the
  certificate, then re-simulation on the residual only, then the aggregate.
- `hilverdict.synthetic`: the declared execution-time and clock models,
  equations (17)-(19), three timing profiles and nine campaigns.
- `hilverdict.report`: plain-text rendering that prints the standing of every
  number beside it.
- A CLI with `certify`, `resimulate`, `clock`, `verdict`, `threshold` and
  `learned`.
- Eleven validation scripts with their raw output, seven example figures, and
  518 tests including Hypothesis property tests, 24 pinned regressions and 23
  negative controls.

### Measured and reported as found

- **The analytic bound wins outright.** On 841 held-out misses the certificate
  reaches a recall of 1.0000 and dismisses 0 of 47 consequential misses; the
  learned classifier, at the only threshold a user could actually have chosen,
  reaches 0.7234 and dismisses 13. On the certificate's residual set it saves
  0.000000 of the re-simulations at no cost in recall.
- The certificate is never violated by the nonlinear reference: 0 violations
  in 875 misses, worst ratio of truth to bound 0.710647, median 0.338738.
- It dismisses 0.876571 of misses and saves 0.869239 of the plant evaluations.
- Its looseness is driven by the declared metric's anisotropy: median ratio
  0.984466 at anisotropy 5 against 0.326902 at the declared 200.
- At an eight-period hold the bound overstates the divergence by about a
  factor of 1055, because equation (9) grows exponentially while this plant's
  divergence grows roughly linearly in the overrun.
- Replacing `phi(mu_S, tau)` by `tau` — which would make the bound invalid in
  general — is NOT falsified by the measured population. Reported as measured,
  and deliberately not offered as a negative control.
- The reset detector has a measured floor: a 9e-05 s offset jump is missed and
  a 1.1e-4 s one is detected.

### Found during the build and fixed

- `exponential_integral` computed as `expm1(mu t)/mu` broke at two subnormal
  arguments found by property testing: at `mu = -1.1125369292536007e-308` it
  returned a value one unit in the last place above `t`, violating the sign
  statement equation (13) relies on; at `mu = 5e-324` the `expm1` underflowed
  to exactly zero and the quotient returned `0.0` instead of `t`, a total loss
  rather than a rounding error. Both are fixed by taking the first-order series
  for `|mu t| < 1e-8` and both are pinned as named regression tests.
- Two negative controls could not fire in their first form and were replaced
  rather than accepted: the learned-pipeline controls compared recall at a
  fixed 0.25 flag rate, where every model on this population reaches recall 1,
  and the incidence-validity control cleared the 10 deg limit by 0.0059 deg
  because the cubic plant's incidence response saturates just below it.
- The learned threshold was originally selected using the test labels, which
  reaches recall 1 by construction and made the learned component look like it
  beat the certificate. It is now selected on the training runs; the leaked
  version is retained in the results table, labelled as an oracle.
- The optional early exit originally stopped at the first sample instant,
  because its criterion compared a zero error against a zero peak, and its
  reported saving was a projection rather than work not done. It now resumes in
  chunks so that stopping early actually costs less, and it is verified
  bit-identical to the full horizon.
- The integrator's right-hand side was evaluated through a numpy array path at
  12.4 microseconds per call, which dominated every re-simulation in the
  package. It is now a scalar path at 0.83 microseconds, with `derivative`
  delegating to it so that there is one arithmetic path and not two.

### Known limitations at 0.1.0

Listed in full in `README.md` under Limitations and in
`docs/REQUIREMENTS.md` section 10. The ones that bear on how a result should
be read: every number is measured against synthetic runs from a declared
plant; the certificate is a theorem about the declared linearisation and the
measurement that the linearisation covers the plant is evidence, not proof;
the miss model does not cascade; the analysis is one miss at a time; and the
`git ls-files` hygiene check examined 0 files in the environment it was run
in and is recorded as vacuous there.
