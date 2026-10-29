# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-10-29

First release. Status: TESTING. Validation level 2, research grade.

### Added

- `aligncal.geometry`: SO(3) algebra with one fixed convention, Rodrigues'
  formula and its principal logarithm with a branch at `p = pi`, and the
  identity `[a]x^T [a]x = |a|^2 I - a a^T` that the whole product rests on.
  Cross-checked against `scipy.spatial.transform.Rotation`, which this package
  did not write, to 1e-14 over 500 random rotations.
- `aligncal.record`: `SensorSpec` and `AlignmentRecord`, with the measurement
  model of equations (4) to (7) and validation that refuses a non-finite rate, a
  non-increasing time base, a non-rotation mounting or a non-identity reference
  mounting.
- `aligncal.observability`: **the first output of every call.** The Gramian of
  (8), the excitation second moment of (9), the structural identity (10), the
  noise floor of (11) that makes a bare rank test wrong, the Cramer-Rao bound of
  (12) per degree of freedom and per body axis, and an independent Gershgorin
  and Rayleigh bracket (13) on the smallest eigenvalue. A three-valued verdict
  per degree of freedom: observable, marginal, unobservable.
- `aligncal.batch`: Baseline 1. Weighted batch least squares (14)-(16) on the
  observable subspace only, Gauss-Newton on the exact rotation with a declared
  trust region, Markley's SVD solution (17) as an independent cross-check, and a
  regularised MAP variant (18) reported beside the specified baseline.
- `aligncal.recursive`: Baseline 2. A multiplicative extended Kalman filter
  (19)-(22) with a Joseph-form covariance update, scored against the batch
  solution, whose covariance is its own statement of what the record does not
  support.
- `aligncal.dataset`: a seeded synthetic generator with five declared excitation
  profiles and complete ground truth. No flight data anywhere.
- `aligncal.features`, `aligncal.learned`, `aligncal.benchmark`: twenty-four
  motion features that deliberately include the observability evidence, three
  declared learned configurations each with an ensemble-spread uncertainty, a
  trivial training-mean baseline, and scoring split by subspace with the
  standardised error of (24).
- A CLI with `info`, `observability`, `align`, `compare`, `unobservable` and
  `benchmark`.
- Nine validation scripts with their raw output, eight example figures, and
  **307 tests** including Hypothesis property tests and eleven pinned
  regressions.

### Measured and reported as found

- **The analytic baseline beats the learned component.** On 600 held-out
  150-sample records the regularised batch solution of (18) reaches 0.697326
  mrad on the observable subspace against 0.760152 mrad for the best learned
  configuration, a margin of 9.01 per cent, and on every one of the four
  excitation profiles. Published as measured.
- **The learned estimator returns a confident estimate for a degree of freedom
  the data cannot support.** With 2 mrad injected along a single-axis record's
  own rotation axis, the sharpest learned configuration claims -7.5118 arcsec
  with a reported one sigma of 14.3540 arcsec against a truth of 412.5296
  arcsec: **29.0693 of its own standard deviations, with two-sigma coverage of
  0.0000 over 120 records.** The batch solution withholds the direction and
  reports an infinite bound on all 120. This is the product's most useful
  result.
- **The overconfidence is specific to the unsupported direction.** Along the
  best-observed direction of the same records the same models cover at or above
  the nominal 0.954500 — two of them at 1.0000. Nothing in the model's own
  output distinguishes the two cases.
- **Record length does not help.** A factor of 32, from 150 to 4800 samples:
  the learned error along the unsupported direction stays between 419.8 and
  421.0 arcsec.
- **A bare numerical rank test is fooled on every single-axis record.**
  `numpy.linalg.matrix_rank` reports 3 on 5 of 5 records that support 2, because
  the reference sensor's own noise fills the null space at `2 n sigma_A^2`. The
  floor gate of (11) is not fooled on any of them. With the gate removed the
  estimator's claim along the unsupported direction reaches **1106 mrad**.
- **The recursive estimator is overconfident in the unsupported direction too.**
  Its two-sigma interval contains the truth on 0.591837 of rank-deficient
  records against a nominal 0.954500, and the under-coverage correlates at
  -0.6690 with how much of its prior it reduced on the strength of noise alone.
- **An unconstrained solution is wrong about the directions it CAN determine.**
  Markley's SVD solution picks up a gauge rotation with a median of 1.5355 rad
  in the null direction, and that rotation contaminates the observable subspace
  through the non-linearity of rotation composition — by more than the whole
  magnitude of the correct estimate on 2000-sample records. The recursive
  estimator shows the same mechanism: its leakage reaches 1.942601 of the
  Cramer-Rao one sigma on rank-deficient records against 0.352091 on full-rank
  ones.
- **The analytic win is bought with a prior.** With a prior ten times too tight
  the MAP baseline loses to the unbiased solution at all four record lengths
  measured.
- **One learned configuration is overconfident on the whole population**, not
  only in the unobservable subspace: `bagged_ridge` reports a median one sigma
  12 times below its realised error, with two-sigma coverage 0.187778.
- **The two-axis record is worse conditioned than the three-axis one** (2.037
  against 1.667), which is the opposite of what the number of excited axes
  suggests.
- **16 controls: 10 declared must fire and 10 fired; 6 declared must not fire
  and none did.**

### Defects found during the build and pinned as regressions

- The matrix logarithm of `aligncal.geometry` carried a spurious factor of two,
  returning half of every rotation vector. `(R - R^T)/2 = sin(p) [n]x` and
  `unskew` already takes the antisymmetric part, so the scale factor is
  `p / sin p`.
- The generator applied the mounting without a transpose, producing data for the
  inverse rotation. Measured consequence before the fix: the batch estimator
  failed to converge on 98.2 per cent of a 600-record population with a median
  error of 1880 mrad.
- Equation (10) pairs **ascending** eigenvalues with **descending** excitation
  shares. Comparing them in the same order turns a correct identity into a 100
  per cent error, and the first draft of that test did exactly that.
- The relative linearisation error of the residual is `O(|theta|)`, not
  `O(|theta|^2)`: the residual is itself of order `|theta| |w|`, so the same
  absolute error is a hundred times larger in relative terms at 10 mrad. A test
  written to the matrix figure failed.
- Gauss-Newton needs a trust region. Without one the iteration failed to
  converge on 7.8 per cent of a 600-record low-information population; with it,
  on none. The iteration budget was then raised from 24 to 40 because one record
  in 864 reached the 1e-12 rad step bound on its 25th iteration, with an answer
  already settled to 4.0e-10 mrad.
- The matrix logarithm of a product of subspace-restricted updates leaks a
  rounding-level component into the withheld direction. It is projected out
  explicitly so the withheld component is zero, which is the claim the product
  makes.
- A ratio of two root-mean-squares is the wrong statistic for comparing a
  measured error against the Cramer-Rao bound within a swept cell: the bound
  varies by orders of magnitude between seeds, so the ratio is decided by the
  single worst-conditioned seed and ranged from 0.775 to 1.692 while every
  standardised score stayed inside 2.4 sigma. The per-direction standardised
  score is what is checked.
- Counting an infinite reported sigma as covering the truth would have scored
  the refusing estimator as perfectly calibrated and reversed the product's
  headline measurement. `benchmark.coverage` treats it as a miss.

### Known limitations

See `README.md`. The dominant ones: every number is measured against a seeded
synthetic generator with white noise and nothing else; a mis-declared mounting
is indistinguishable from a misalignment; the learned component loses to the
strengthened analytic baseline; and an observability verdict is a statement
about the record, not about the instrument.

[0.1.0]: https://github.com/OmAcharya-avtr/aligncal/releases/tag/v0.1.0
