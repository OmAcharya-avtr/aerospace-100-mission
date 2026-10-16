# Validation evidence: acqsearch 0.1.0

Validation Level 2. Every number in this file was produced by a script in this
directory, in the build session of 2026-10-16, in a virtual environment pinned
to the versions recorded in `validate_environment_output.txt`. Each script
writes its raw output to `<script-name>_output.txt` beside it.

| script | what it establishes | output | wall clock |
|---|---|---|---|
| `validate_suite.py` | the suite count, taken from junit XML and cross-checked | `validate_suite_output.txt` | 18.1 s |
| `validate_environment.py` | the exact versions every number was measured at | `validate_environment_output.txt` | 1.2 s |
| `validate_detection_model.py` | the detection model against cited closed forms | `validate_detection_model_output.txt` | 1.2 s |
| `validate_spiral_coverage.py` | the spiral coverage closed form against Monte Carlo | `validate_spiral_coverage_output.txt` | 7.8 s |
| `validate_analytic_vs_mc.py` | expected acquisition time, analytic against Monte Carlo | `validate_analytic_vs_mc_output.txt` | 2.1 s |
| `validate_dwell_optimum.py` | the dwell trade and its sensitivity | `validate_dwell_optimum_output.txt` | 6.2 s |
| `validate_policies.py` | the head-to-head, the ablation and the interval calibration | `validate_policies_output.txt` | 103.3 s |
| `worked_example.py` | the snippet quoted in README.md | `worked_example_output.txt` | 1.2 s |
| `dependency_range_check.py` | a numeric fingerprint for diffing across dependency versions | `dependency_range_check_output.txt` | 4.3 s |

No script exceeds the three-minute compute budget. `validate_policies.py` is
the longest at 103.3 s on two cores; every other script is under 20 s. The
times above were measured with `date` around each run in the build session.

## 1. Test suite, counted from junit XML

A pytest configuration that collects nothing prints success and exits 0, so
the count is taken from the junit `testsuite` element and cross-checked
against an independent count of `testcase` elements.

| quantity | value |
|---|---|
| `tests` attribute | 242 |
| `testcase` elements | 242 |
| failures | 0 |
| errors | 0 |
| skipped | 0 |
| exit code | 0 |

Source: `validate_suite_output.txt`.

## 2. Detection model, against cited closed forms

Source: `validate_detection_model_output.txt`.

| check | reference | result | tolerance |
|---|---|---|---|
| `P(N>=1 \| 1) = 1 - e^-1` | Abramowitz & Stegun 1964, eq. 26.4 | 0.632120558828558, error 0.00e+00 | 1e-15 |
| `P(N>=2 \| 1) = 1 - 2 e^-1` | same | 0.264241117657115, error 0.00e+00 | 1e-15 |
| `P(N>=3 \| 2) = 1 - 5 e^-2` | same | 0.323323583816936, error 0.00e+00 | 1e-15 |
| `P(N>=1 \| 0) = 0` | same | 0.000000000000000, error 0.00e+00 | 1e-15 |
| `P(N>=4 \| 3)` | same | 0.352768111217769, error 0.00e+00 | 1e-15 |
| Beam shape factor at rho/w = 0, 0.5, 1.0, 1.5, 2.0 | Siegman 1986 Ch. 17, `exp(-2 rho^2/w^2)` | 5 of 5, error 0.00e+00 | 1e-14 |
| Lognormal irradiance quadrature, `sum W = 1` | Abramowitz & Stegun 1964, eq. 25.4.46 | 1.000000000000000 at every Rytov variance | 1e-12 |
| Lognormal irradiance quadrature, `E[I] = 1` | Andrews & Phillips 2005 Ch. 9 | error <= 2.22e-16 at Rytov 0.00, 0.05, 0.20, 0.60, 1.00 | 1e-6 |
| Hermite-node convergence of `P_d`, against 96 nodes | internal | 8 nodes 4.950e-05, 16 nodes 1.399e-07, 24 nodes 1.316e-09, 48 nodes 1.199e-14 | - |
| Interpolation error of the 1025-point `P_d` table | internal, 977-point probe | max 1.959952e-05 at 1.096 beam half-widths, mean 1.593020e-06 | 5e-5 |

Reference-configuration values: `P_fa` 0.008130742 at a 1.4 ms dwell, `P_d`
0.995647312 on axis, 0.963260627 at half a beam half-width, 0.339826444 at one
beam half-width.

### References, in full

- Abramowitz, M., & Stegun, I. A. (1964). *Handbook of Mathematical
  Functions*. National Bureau of Standards Applied Mathematics Series 55.
  Eq. 25.4.46 (Gauss-Hermite quadrature); Ch. 26.4 (Poisson and chi-square).
- Andrews, L. C., & Phillips, R. L. (2005). *Laser Beam Propagation through
  Random Media*, 2nd ed. SPIE Press. Ch. 9: the lognormal irradiance PDF in
  the weak-fluctuation regime, valid for Rytov variance up to about 1.
- Gagliardi, R. M., & Karp, S. (1995). *Optical Communications*, 2nd ed.
  Wiley. Ch. 5: direct-detection photon counting and threshold detection on a
  Poisson count.
- Marcum, J. I. (1960). "A statistical theory of target detection by pulsed
  radar". *IRE Transactions on Information Theory* 6(2), 59-267. The
  detection / false-alarm pair of a threshold test on a count statistic.
- Siegman, A. E. (1986). *Lasers*. University Science Books. Ch. 17: the
  far-field TEM00 irradiance profile.
- Papoulis, A., & Pillai, S. U. (2002). *Probability, Random Variables and
  Stochastic Processes*, 4th ed. McGraw-Hill. Ch. 6: the radial marginal of
  an isotropic bivariate normal is Rayleigh.
- Koopman, B. O. (1980). *Search and Screening: General Principles with
  Historical Applications*. Pergamon Press. Ch. 4-6: parallel-sweep and
  spiral search, search effort, and the detection-rate-per-unit-effort rule.
- Stone, L. D. (1975). *Theory of Optimal Search*. Academic Press
  (Mathematics in Science and Engineering vol. 118). Ch. 2-3: optimal
  allocation of search effort under an exponential detection function.
- Feller, W. (1968). *An Introduction to Probability Theory and Its
  Applications*, Vol. 1, 3rd ed. Wiley. Ch. XIII: the geometric distribution.

## 3. Spiral coverage: geometric closed form against Monte Carlo

The closed form, derived in `src/acqsearch/coverage.py`, is

```
C = 1 - (sigma / p) sqrt(pi / 2) [erf(u1 / (sigma sqrt 2)) - erf(u0 / (sigma sqrt 2))]
u0 = (n - 1) p + a,   u1 = n p + a
```

valid when the spiral leaves no gap between consecutive turns,
`p <= 2 sqrt(a^2 - (d/2)^2)` and `d <= 2a`. The Monte Carlo draws 200000
points from the same Rayleigh prior and tests membership of the union of
footprint discs; it shares no code with the closed form beyond the sampler.

Source: `validate_spiral_coverage_output.txt`. 16 cases, two geometries,
turns 1 to 8. Tolerance 3 Monte Carlo standard errors plus 2e-3 of geometric
slack for the along-track discretisation. **0 of 16 outside tolerance.**

| geometry | worst residual | at | in standard errors |
|---|---|---|---|
| pitch 70.7107 urad, along-track 35.3553 urad | +0.001840 | 3 turns | 1.76 |
| pitch 60.0000 urad, along-track 30.0000 urad | +0.003884 | 3 turns | 3.53 |

Limiting cases: at a vanishing pitch and one turn the closed form gives
0.054041582 against the exact single-disc value 0.054040531, an error of
1.05e-06 (see Limitations in README.md, item 9); at 40 turns it gives
1.000000000 against 1, error 0.00e+00.

Final covered prior mass of the four standard paths over the reference grid:
spiral 0.985133 after 227 dwells, raster 0.996717 after 169, Lissajous
0.912617 after 712, random restart 0.978300 after 890. The prior mass inside
the 89-cell grid is 0.954498.

## 4. Expected acquisition time: analytic against Monte Carlo

The analytic model, its assumptions A1 to A4, and the derivation are in the
module docstring of `src/acqsearch/acquisition.py`. The Monte Carlo samples
exactly those assumptions, so any disagreement beyond its standard error is an
error in the formula.

Source: `validate_analytic_vs_mc_output.txt`. Five seeds, 200000 trials each.
Tolerance 4 standard errors of the five-seed mean. **0 of 4 outside
tolerance.**

| pattern | pass period (s) | analytic E[T] (s) | Monte Carlo (s) | s.e. (s) | difference (s) | in s.e. |
|---|---|---|---|---|---|---|
| spiral | 1.416526 | 0.548183 | 0.548423 | 0.000567 | +0.000241 | 0.42 |
| raster | 1.420374 | 0.929262 | 0.929622 | 0.000535 | +0.000360 | 0.67 |
| Lissajous | 1.681476 | 1.084705 | 1.085138 | 0.000638 | +0.000432 | 0.68 |
| random restart | 3.880439 | 2.409379 | 2.410481 | 0.001471 | +0.001102 | 0.75 |

Deadline probability at two pass periods (2.833052 s), spiral order: analytic
0.944505, Monte Carlo 0.944690 with a binomial standard error of 0.000512, a
disagreement of 0.36 standard errors.

Drawing the false-alarm penalty binomially instead of charging its expectation
(assumption A3) changes the deadline probability by -0.002235 and leaves the
mean unchanged to within the Monte Carlo error (0.546525 s against
0.542937 s).

### Cost of the cell abstraction, reported not absorbed

| model | mean time (s) | s.e. (s) | trials |
|---|---|---|---|
| analytic cell model, target in grid | 0.548183 | - | - |
| continuous geometry, target in grid | 0.341280 | 0.001872 | 38125 |
| continuous geometry, whole prior | 0.402173 | 0.002918 | 40000, all acquired |

**The cell model is pessimistic by 37.74 percent.** The sign is known: the
abstraction forbids a dwell from acquiring a target in a neighbouring cell and
the continuous geometry allows it. The analytic number is therefore an upper
bound, and a large one, which is why both are published and why
`monte_carlo_acquisition_continuous` exists.

## 5. Dwell-time trade

Source: `validate_dwell_optimum_output.txt`. 41-point geometric sweep from
0.2 ms to 20 ms, spiral order over the 89-cell grid.

| quantity | value |
|---|---|
| grid minimum | 0.537002 s at 1.2619 ms |
| E[T] at 0.2000 ms | 28.886553 s |
| E[T] at 20.0000 ms | 13.487604 s |
| worst / best over the sweep | 53.79 |
| mean cell detection probability at the optimum | 0.861229 |
| per-dwell false-alarm probability at the optimum | 0.004489 |

Monotonicity over the sweep: the mean cell detection probability is
non-decreasing and the false-alarm probability is non-decreasing, both
confirmed. The optimum is a grid optimum; no convexity is assumed or claimed.

Sensitivity of the optimum:

| count threshold | best dwell (ms) | best E[T] (s) |
|---|---|---|
| 4 | 0.4477 | 1.110630 |
| 6 | 0.8934 | 0.691474 |
| 8 | 1.2619 | 0.537002 |
| 10 | 1.7825 | 0.467962 |
| 12 | 2.2440 | 0.434153 |

| verification time (s) | best dwell (ms) | best E[T] (s) |
|---|---|---|
| 0.00 | 2.5179 | 0.321546 |
| 0.10 | 1.5887 | 0.417046 |
| 0.50 | 1.2619 | 0.537002 |
| 2.00 | 1.1247 | 0.722037 |
| 10.00 | 0.8934 | 1.068882 |

| Rytov variance | best dwell (ms) | best E[T] (s) |
|---|---|---|
| 0.00 | 1.2619 | 0.432656 |
| 0.10 | 1.2619 | 0.483634 |
| 0.20 | 1.2619 | 0.537002 |
| 0.40 | 1.2619 | 0.646906 |
| 0.80 | 1.2619 | 0.870532 |

Scintillation moves the cost but not the optimum over this sweep, which is a
consequence of the lognormal average being close to multiplicative in the
detection probability at this operating point. It is not a general result.

## 6. Policy head-to-head: the baseline wins

Source: `validate_policies_output.txt`. 300 scenarios, 195 train and 105 held
out by id, 12000 training rows, 840 paired episodes per policy.

| policy | mean (s) | s.e. (s) | median (s) | acquired | mean dwells |
|---|---|---|---|---|---|
| fixed spiral | 2.1673 | 0.1437 | 0.7936 | 0.9881 | 45.2 |
| Bayesian-greedy | 2.1769 | 0.1492 | 0.7467 | 0.9893 | 45.0 |
| learned | 2.5128 | 0.1600 | 1.0970 | 0.9810 | 47.5 |
| fixed raster | 3.9036 | 0.2311 | 1.5647 | 0.9821 | 66.8 |
| fixed Lissajous | 4.5186 | 0.2908 | 1.9007 | 0.9702 | 71.6 |
| fixed random restart | 5.1877 | 0.2712 | 2.9111 | 0.8798 | 53.8 |

Paired on identical episodes:

| comparison | mean difference (s) | s.e. (s) | z |
|---|---|---|---|
| learned - Bayesian-greedy | +0.3359 | 0.0446 | +7.53 |
| learned - fixed spiral | +0.3455 | 0.0531 | +6.51 |
| Bayesian-greedy - fixed spiral | +0.0097 | 0.0244 | +0.40 |
| Bayesian-greedy - fixed raster | -1.7267 | 0.1737 | -9.94 |
| fixed spiral - fixed raster | -1.7363 | 0.1699 | -10.22 |
| fixed spiral - fixed random restart | -3.0204 | 0.2496 | -12.10 |

**The learned policy loses to both references at more than six standard
errors. The two references are indistinguishable from each other at z = 0.40.**
This is the result the spec said to expect and to report if it held. It holds.
It is stated in the same terms in README.md and MODEL_CARD.md and pinned as
an assertion in `tests/test_benchmark.py`.

### Shortlist ablation, first 40 held-out scenarios

| policy | mean (s) | s.e. (s) | median (s) | acquired |
|---|---|---|---|---|
| learned k=1 | 1.9532 | 0.2585 | 0.6164 | 0.9938 |
| learned k=4 | 2.0719 | 0.2568 | 0.6622 | 0.9906 |
| learned k=8 | 2.2143 | 0.2747 | 0.8727 | 0.9875 |
| learned k=12 | 2.3858 | 0.2943 | 1.0028 | 0.9875 |
| learned k=24 | 2.8390 | 0.2861 | 1.2624 | 0.8906 |
| learned k=89 | 7.1102 | 0.8039 | 3.4508 | 0.6094 |
| Bayesian-greedy | 1.9560 | 0.2591 | 0.5967 | 0.9938 |
| fixed spiral | 1.9036 | 0.2432 | 0.6679 | 0.9906 |

The learned policy degrades monotonically as it is given more freedom to
disagree with the baseline, and the unconstrained version (k = 89, the whole
grid) fails outright: it acquires in 0.6094 of episodes against 0.9938 for
the baseline. At k = 1 the model has exactly one candidate, which is the
baseline's own choice, and it reproduces the baseline to 1.9532 s against
1.9560 s; the residual difference is tie-breaking between equal greedy scores
(`argpartition` against `argmax`).

This table is on 40 of the 105 held-out scenarios because the full ablation
does not fit the compute budget. The reduction is stated in the output file
and both tables are published.

### Prediction-interval calibration

| quantity | value |
|---|---|
| held-out decisions scored | 2006 |
| empirical coverage of the (0.1, 0.9) interval | 0.7353 |
| nominal coverage | 0.8000 |
| mean interval width | 4.6773 s |
| mean reported confidence | 0.3156 |

**The interval under-covers by 6.5 percentage points.** No recalibration is
applied.

## 7. Dependency ranges, both ends executed

Every range in `pyproject.toml` is bounded on both sides and both ends were
installed in a clean virtual environment and executed: the full suite, counted
from junit XML, and `dependency_range_check.py`, whose output is a 33-line
numeric fingerprint covering the grid, the detection model, all four analytic
expected times, the Monte Carlo, the coverage closed form, the dwell optimum,
the training rows and all three policies.

| package | floor executed | ceiling executed | suite | fingerprint |
|---|---|---|---|---|
| numpy | 2.4.1 | 2.5.3 | 242 / 0 / 0 / 0 at both | byte-identical |
| scikit-learn | 1.9.0 | 1.9.1 | 242 / 0 / 0 / 0 at both | byte-identical |
| matplotlib | 3.10.8 | 3.11.2 | 242 / 0 / 0 / 0 at both | byte-identical |
| pytest | 8.4.2 | 9.1.1 | 242 / 0 / 0 / 0 at both | byte-identical |
| hypothesis | 6.168.0 | 6.170.0 | 242 / 0 / 0 / 0 at both | byte-identical |
| ruff | 0.16.8 | 0.17.1 | `ruff check src/ tests/` clean at both | - |

### The check that justifies the scikit-learn floor

scikit-learn **1.8.0**, one minor below the declared floor, was installed and
executed deliberately. **The suite passes: 242 tests, 0 failures, 0 errors, 0
skipped.** The fingerprint does not. Diff of
`dependency_range_check_output.txt` (1.9.1) against
`dependency_range_check_sklearn180_output.txt` (1.8.0):

```
policy_mean_learned        3.809425254938  ->  3.807297928186
single_scenario_learned    3.750148878426  ->  4.415262144178
decision_interval_low_s    0.171504258285  ->  0.218620563405
decision_interval_high_s   1.582596512604  ->  1.863967740266
decision_confidence        0.184748663028  ->  0.162725466124
```

Every other line of the fingerprint is byte-identical, because everything
outside the learned policy is deterministic numpy. `single_scenario_learned`
moves by 17.7 percent. This is the failure mode that no test catches: a suite
that passes at both ends of a range while the published numbers move
underneath it. The floor is 1.9.0 for that reason and the cost is recorded in
README.md Limitations, item 10.

## 8. Committed files are not gitignored

Every scratch pattern in `.gitignore` that could ever match a deliverable is
anchored with a leading slash. `git check-ignore -v` was run against all 63
files this product ships and reported **no match for any of them**, so a cold
clone gets every one.

The anchoring was then verified to still do its job, by checking that the
patterns do match where they are supposed to:

| path | result |
|---|---|
| `products/P067/junit.xml` | ignored by `/junit.xml` (line 10) |
| `products/P067/.hypothesis` | ignored by `.hypothesis/` (line 30) |
| `products/P067/.pytest_cache` | ignored by `.pytest_cache/` (line 29) |
| `products/P067/.ruff_cache` | ignored by `.ruff_cache/` (line 31) |
| `products/P067/src/acqsearch/__pycache__` | ignored by `__pycache__/` (line 27) |
| `products/P067/fixtures_junit.xml` | NOT ignored, which is the point of the leading slash |

This product ships no file named after a scratch artifact, but the anchoring
is there because P060 did and a cold clone of it failed with 17 failures and
9 errors.

## 9. Reproducing this file

```bash
python validation/validate_suite.py           > validation/validate_suite_output.txt
python validation/validate_environment.py     > validation/validate_environment_output.txt
python validation/validate_detection_model.py > validation/validate_detection_model_output.txt
python validation/validate_spiral_coverage.py > validation/validate_spiral_coverage_output.txt
python validation/validate_analytic_vs_mc.py  > validation/validate_analytic_vs_mc_output.txt
python validation/validate_dwell_optimum.py   > validation/validate_dwell_optimum_output.txt
python validation/validate_policies.py        > validation/validate_policies_output.txt
python validation/worked_example.py           > validation/worked_example_output.txt
python validation/dependency_range_check.py   > validation/dependency_range_check_output.txt
find . -type f -not -path "*/__pycache__/*" | xargs -n1 git check-ignore -v ; echo "no output above means nothing shipped is ignored"
```

## 10. What is not validated

- No hardware comparison of any kind. There is no measured acquisition time
  from any terminal in this repository.
- No validation of the photon rates. They are inputs; the link budget that
  would produce them is out of scope and is the hardest part of the problem.
- No validation of the slew model beyond its own arithmetic. A constant rate
  plus a fixed settle time is not what a fine-steering mirror does, and the
  pattern ranking is sensitive to slew cost.
- No temporal correlation in the scintillation, so the independence assumption
  behind repeated dwells on the same cell is unverified and is probably
  optimistic at millisecond dwell separations.
- The learned policy is validated as a measurement, not as a component. It
  loses, and nothing here establishes that a different learned policy would.
