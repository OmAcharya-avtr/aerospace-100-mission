# Validation evidence

**Validation label: Level 3, hardware-pending.**

Every number in this file was produced by a script in this directory, executed
in the session that wrote the file, on the environment recorded at the top of
each `*_output.txt`. No figure is carried from another session, estimated, or
taken from a datasheet. **No number anywhere in this repository was measured on
hardware**: every backend reports `is_hardware == False`, the device backend's
only drivers are a loopback stub and a recording replayer, and the Level 4
table in `validation/benchmark_record.json` has five rows with every value
`null` and every status `unmeasured`.

Environment every number was measured in: Python 3.13.17, numpy 2.5.3, scipy
1.18.1, scikit-learn 1.9.1, matplotlib 3.10.8, pytest 8.4.2, hypothesis
6.140.3, ruff 0.14.2, on 2 available cores. `allantools` and `ntplib` are
**not installed** and are not used; the Allan deviation is implemented in
`timetransfer/allan.py` and verified against closed forms.

Test suite: **335 tests, 0 failed, 0 errors, 0 skipped**, counted from
`junit.xml` (`testsuite tests="335"`, 335 `<testcase>` elements), never from
pytest's stdout.

---

## 1. The symmetric-delay estimator and its closed-form bias

`validation/validate_symmetric_bias.py` -> `validate_symmetric_bias_output.txt`

| Check | Reference | Result | Tolerance |
|---|---|---|---|
| Exact under a constructed symmetric delay, 60 exchanges across five offsets, four delays and three turnarounds | `theta` by construction | worst \|error\| **0.000e+00 s** | 1.000e-15 s |
| Bias against a declared static asymmetry, 7 values from -1.6 ms to +1.6 ms | `A/2`, equation (5) of `timetransfer.twoway` | worst \|measured - A/2\| **1.084e-19 s** | 2.000e-09 s |
| Fitted slope of error against asymmetry | 0.5 exactly | **0.500000000000**, error 1.110e-16 | 1.0e-6 |
| Bias independent of one-way floor (1 ms to 50 ms), turnaround (0 to 100 ms) and true offset (-100 ms to +100 ms) | `A/2` | worst departure **1.144e-17 s** | 2.000e-09 s |
| Averaging does not remove the bias: mean error at n = 20, 200, 2000 with jitter on | `A/2` = 4.000e-04 s | 3.839e-04, 4.265e-04, 4.055e-04 s; per-exchange sd 3.1e-04, 2.0e-04, 2.0e-04 s | reported, not a pass/fail |

The last row is the identifiability result made numerical: the per-exchange
scatter falls with `n` and the mean error does not approach zero.

## 2. Allan deviation against closed-form noise processes

`validation/validate_allan.py` -> `validate_allan_output.txt`

`allantools` is not installed and is not used. The reference is the closed form
for each process (Allan 1966; IEEE Std 1139-2008; Zucca and Tavella 2005).

| Process | Theory slope | Fitted slope | Stderr | \|diff\| | Worst amplitude error vs theory |
|---|---|---|---|---|---|
| white PM, `sigma_x = 1e-9 s` | -1.00 | **-0.9998** | 0.0005 | 0.0002 | 0.02183 |
| white FM, `q1 = 1e-20 s^2/s` | -0.50 | **-0.4834** | 0.0032 | 0.0166 | 0.27167 |
| random walk FM, `q2 = 1e-24 1/s` | +0.50 | **+0.5194** | 0.0017 | 0.0194 | 0.15248 |

N = 20000 phase samples at `tau0 = 1 s`, octave-spaced tau to `N//5`.
Tolerance on the slope 0.0500; tolerance on the amplitude 0.40.

**Confidence-interval coverage, measured**, 300 seeded realisations of N = 2048
per process, 27 (process, tau) cells: lowest **0.9233** at white PM `m = 32`,
highest **0.9767** at white FM `m = 256`, nominal 0.95. Every cell is reported
in the output file. The equivalent-degrees-of-freedom formulas are the
simplified ones of Riley, NIST SP 1065 Table 3, and they are approximations;
this is the measurement of how good they are, not a check that could only pass.

**Noise identification** (lag-1 autocorrelation method, Riley NIST SP 1065
section 5.5.3): 200/200 for each of the three processes, mean measured alpha
2.0037, 0.0038, -2.7166 against theoretical 2, 0, -2. The random-walk-FM alpha
is biased low because the method stops at the first differencing level with
`r1 < 0.25`; the classification is still correct at every trial.

## 3. The measured offset uncertainty and its coverage

`validation/validate_uncertainty_coverage.py` -> `validate_uncertainty_coverage_output.txt`

600 training, 200 calibration and 400 coverage scenarios, disjoint, from the
generator declared in `timetransfer/asym/data.py` at coupling 1.0 with an
independent-asymmetry standard deviation of 2.0e-04 s. The residual spread that
becomes `u_asymmetry` is measured on the calibration split, which the corrector
neither trained on nor is scored on.

**Coverage at k = 1.96, nominal 0.9500:**

| Treatment | Measured | Clopper-Pearson 95 per cent | median \|e\|/u | Verdict |
|---|---|---|---|---|
| `unaccounted` (no asymmetry term) | **0.1200** | [0.0898, 0.1559] | 9.5450 | **optimistic** |
| `declared_bound` (B = 1.2e-03 s) | **0.9900** | [0.9746, 0.9973] | 0.4556 | **conservative** |
| `median_filter` corrector applied | **0.9450** | [0.9179, 0.9652] | 0.6497 | calibrated |
| `learned_gbr` corrector applied | **0.9525** | [0.9268, 0.9712] | 0.7517 | calibrated |

Four coverage factors are reported for each treatment in the output file.

**This is the central result of the product.** An interval with no asymmetry
term covers 12 per cent of the time against a nominal 95, because it is a
repeatability statement and the error is dominated by accuracy. The two
corrected treatments are calibrated because the asymmetry term is a *measured*
residual spread rather than a datasheet figure.

Mean absolute offset error over the 400 coverage scenarios: 1.966e-04 s
uncorrected, 1.208e-04 s with the median filter, 9.531e-05 s with the learned
corrector.

## 4. Least squares and Kalman

`validation/validate_skew_estimators.py` -> `validate_skew_estimators_output.txt`

| Check | Reference | Result | Tolerance |
|---|---|---|---|
| Noiseless recovery of a declared quadratic, 800 exchanges | 1.250e-03 s, 1.800e-06 s/s, 2.000e-12 1/s | offset error -6.439e-12 s, skew error +1.922e-14 s/s | 1.0e-11 s/s on the skew |
| Convergence of the skew error under refinement, 40 seeds at n = 50 to 1600 | `n^-3/2`, from `var(a) = 12 sigma^2 / (n(n^2-1)dt^2)` | fitted exponent **-1.5192 +- 0.0218** | departure 0.0192, tolerance 0.25 |
| rms skew error against the reported standard error | ratio 1.0 | 1.0507, 0.9998, 0.9046, 0.9091, 1.0100, 0.9450 at the six counts | reported |

**Reported, not a pass/fail: the Kalman filter is over-confident when `R` comes
from the data.** The measurement variance defaults to a median-absolute-
deviation estimate with the Gaussian consistency factor, and the queueing delay
is a right-skewed gamma, so the MAD under-states its standard deviation. Mean
normalised innovation squared, which is 1.0 for a consistent filter: **2.6547,
2.5347, 2.5736** at jitter scales 5e-05, 2e-04 and 1e-03 s, implying the Kalman
offset uncertainty is optimistic by a factor of about 1.6.

With **no** queueing jitter the data-derived `R` collapses to its floor and the
mean normalised innovation is **8.788e+04**, an over-confidence factor of 296.
That is a trap and it is documented: pass `floor_variance_s2` or an explicit
`KalmanConfig.measurement_variance_s2` on a quiet link.

## 5. HAL simulation/device parity

`validation/validate_parity.py` -> `validate_parity_output.txt`

Four cases of 500 exchanges each, a simulated backend against a device backend
over the loopback stub, both on a virtual timebase:

| Case | Timestamp bytes | Offset max \|diff\| | LS params max \|diff\| | Kalman offset \|diff\| |
|---|---|---|---|---|
| symmetric, no jitter, no loss | identical (20000 B) | 0.000e+00 s | 0.000e+00 | 0.000e+00 s |
| asymmetric with jitter | identical (20000 B) | 0.000e+00 s | 0.000e+00 | 0.000e+00 s |
| lossy with an outage and drift | identical | 0.000e+00 s | 0.000e+00 | 0.000e+00 s |
| coarse 1 us timestamps | identical (20000 B) | 0.000e+00 s | 0.000e+00 | 0.000e+00 s |

Falsifying control: the same comparison across two different seeds gives
differing bytes and an offset max \|diff\| of 1.523e-03 s. A comparison that
cannot fail is not evidence.

The byte comparison is legitimate here and is not a violation of the rule
against bit-exact floating-point assertions: the two paths execute the
identical sequence of operations on identical integer inputs. Nothing is being
claimed about floating-point associativity. `tests/test_parity.py` asserts the
tolerance comparison first and the byte comparison as an additional, stronger
statement.

**This establishes nothing about a board.** The device driver is a stub.

## 6. Dry-run mode

`validation/validate_dryrun.py` -> `validate_dryrun_output.txt`

| Check | Result |
|---|---|
| Committed recording replayed | 227 of 240 attempted, 13 lost, longest sequence gap 9 |
| Driver operations | 240 |
| **Hardware operations** | **0** |
| Refused operations | 0 |
| Backend driver | `guarded-recorded`, `is_hardware = False` |
| Replay reproduces the recording | timestamp bytes identical, offset max \|diff\| 0.000e+00 s |
| Negative control: exchanges forbidden | 0 completed, **241 refused**, acquisition raised `InsufficientDataError` |
| A recording marked `is_hardware=true` | refused with `ConfigurationError` |

The 241 refusals are the 240 recorded sequence numbers plus the acquisition
path's own first attempt, so the counter is shown to fire on every route into
the driver.

## 7. The learned asymmetry corrector against the baselines

`validation/validate_asymmetry_benchmark.py` -> `validate_asymmetry_benchmark_output.txt`

1200 scenarios per coupling, 70/30 scenario-ordered split, scored on held-out
RMSE of the window mean asymmetry bias.

**At the declared default coupling of 1.0:**

| Estimator | Held-out RMSE [s] | MAE [s] | median \|e\| [s] | Skill vs symmetric |
|---|---|---|---|---|
| `symmetric_assumption` | 2.525593e-04 | 1.992098e-04 | 1.584618e-04 | 0.0000 |
| `median_filter` | 1.512459e-04 | 1.211701e-04 | 1.059714e-04 | 0.4011 |
| `mean_difference` | 1.335423e-04 | 1.073300e-04 | 9.018790e-05 | 0.4712 |
| `learned_ridge` | **1.039986e-04** | 8.304100e-05 | 6.947085e-05 | **0.5882** |
| `learned_gbr` | 1.050239e-04 | 8.263485e-05 | 7.060012e-05 | 0.5842 |

**Two results, both reported as measured and neither softened:**

1. **The gradient-boosted corrector loses to a standardised linear ridge on the
   same sixteen features**, at every coupling: RMSE ratio GBR/ridge **1.0273,
   1.0099, 1.0088** at couplings 0.0, 1.0, 2.0. The ensemble, its
   hyperparameters and its version pin buy nothing on this generator.
2. **Whether the median filter matches the learned model depends on the
   coupling**, which is the generator parameter that decides how much of the
   static asymmetry is learnable at all. Median/GBR RMSE ratio: **1.0394,
   1.4401, 1.8868** at couplings 0.0, 1.0, 2.0. At zero coupling -- where the
   static asymmetry is unlearnable by construction -- the median filter is
   within 4 per cent of the learned model, which is the specification's own
   prediction. At the default coupling and above it is not, and that is
   published too.

**The learned model's own uncertainty output under-covers**: held-out coverage
of the k = 1.96 quantile-pair band is **0.8917** against a nominal 0.9500, with
0.0000 quantile crossings. It is a dispersion estimate, not a confidence
interval, and `MODEL_CARD.md` says so. The uncertainty the offset budget
consumes is the calibration-split residual standard deviation instead, and the
coverage of *that* is section 3 above.

Held-out residual standard deviations, which are the numbers
`timetransfer.uncertainty` consumes: symmetric 2.525905e-04, median filter
1.513504e-04, mean difference 1.336667e-04, ridge 1.039929e-04, GBR
1.049785e-04 s.

## 8. Seeded reproducibility

`validation/validate_reproducibility.py` -> `validate_reproducibility_output.txt`

Seven levels, each checked by digest or byte comparison: the whole pipeline
twice at one seed (identical, including the exchange-set digest, the LS
parameters, the Kalman final state and the reported uncertainty); a different
seed (differs); the clock and link models separately; snapshot and restore; the
three synthetic noise processes; the simulated and device backends built twice
each (four acquisitions, all byte-identical); and the dataset generator with
both learned models (predictions and sigmas identical on a refit).

## 9. Deployment and recovery

`validation/validate_deployment.py` -> `validate_deployment_output.txt`

| Check | Result |
|---|---|
| Pre-flight on a healthy simulated backend | **10/10 passed** |
| Negative controls, four of them | each fails the expected named check; a wrong hardware declaration fails `hardware_declaration`, an absent device fails `backend_opens`, `probe_acquires` and `estimate_finite` |
| Pre-flight leaves the backend unchanged | digest after pre-flight equals the digest on an untouched backend |
| Recovery after an injected abort, five different exceptions | **5/5 restored**, 0 of 24 snapshot fields differing in each |
| Exchange lost mid-estimate | 600 planned, outage from 120 s, stopped early at seq 129 after a loss run of 10; **120 exchanges used, 0.2000 of plan**, labelled `truncated` |
| Truncated statistical term against a complete window | **2.1531**, against an independence expectation of 2.2361 |
| A window that loses everything | `InsufficientDataError` with `n_usable = 0`, `n_required = 8` |

## 10. The benchmark harness and the empty Level 4 table

`validation/validate_benchmark_table.py` -> `validate_benchmark_table_output.txt`,
`validation/benchmark_record.json`

Host table: 8 stages, 9 repeats each, measured on a shared two-core container
and **declared unstable** where it is printed. Measured clock resolution 108
ns, giving `u_q(offset) = delta/sqrt(12) = 3.117691e-08 s`. tracemalloc peak
196024 B, maxrss 173342720 B.

**Level 4 hardware table: 5 rows, every value `None`, every status
`unmeasured`.** Three guards, all exercised:

| Guard | Result |
|---|---|
| `BenchmarkRecord(is_hardware=True)` | raises `HardwareClaimError` |
| A hardware row with a status other than `unmeasured` | raises `HardwareClaimError` |
| `promote_to_hardware()` on a simulated record | raises `HardwareClaimError` |

3/3 fired. The written JSON carries `"is_hardware": false` and five rows with
`"value": null`.

## 11. The command-line interface

`validation/validate_cli.py` -> `validate_cli_output.txt`

Eleven invocations, each executed and its exit status checked: `version`,
`info`, `simulate`, `allan`, `parity`, `preflight`, `record`, `dryrun`,
`bench`, a missing recording (exit 2) and a link too lossy to estimate from
(exit 2). 0 unexpected statuses.

## 12. Every validation script runs from any working directory

`validation/validate_cwd_independence.py` -> `validate_cwd_independence_output.txt`

13 scripts x 4 working directories -- the filesystem root, the system temporary
directory, the product root and `validation/` -- **52 runs, 0 non-zero exit
statuses**. The probe runs set `TIMETRANSFER_FAST=1`, which shrinks the Monte
Carlo loops and makes the output writer write nothing, so a probe cannot
replace a committed output with reduced-sample numbers.

This check exists because three Batch 07 products passed every check that
preceded the release gate and then failed it, for exactly this reason.

## 13. The committed fixture

`validation/regenerate_fixture.py` -> `regenerate_fixture_output.txt`

`fixtures/recorded_exchange.json` is regenerated deterministically from seed
20261024: 240 attempted, 227 completed, 13 lost, longest loss run 8, digest
`6cad0b025a9b469d522a2e3458af647c6b62f79b1e5c765e5862687e379251a8`. The JSON
round trip is byte-identical.

---

## Checks that failed, under-performed, or are reported as negative results

This section exists because a validation document without one is not credible.

1. **An interval with no asymmetry term covers 0.1200 against a nominal
   0.9500.** Section 3. This is the headline, and it is the reason the product
   exists in this form.
2. **A declared asymmetry bound over-covers, 0.9900 against 0.9500.** Section
   3. A type-B bound at six times the generator's own asymmetry standard
   deviation is honest and conservative; it is the right answer when external
   knowledge exists and it is not calibrated.
3. **The learned model's own uncertainty band under-covers, 0.8917 against
   0.9500.** Section 7. It is a dispersion estimate and is labelled one.
4. **The gradient-boosted corrector loses to a linear ridge at every coupling
   measured.** Section 7. The learned component does not justify its
   machinery on this generator.
5. **The Kalman filter is over-confident when `R` is taken from the data**, by
   a factor of about 1.6 on a jittery link and 296 on a quiet one. Section 4.
   Reported, documented, and fixable by the caller.
6. **Allan confidence intervals under-cover at some tau**, lowest 0.9233
   against 0.95 at white PM `m = 32`. Section 2. The edf formulas are
   approximations.
7. **The random-walk-FM noise identifier reports a biased alpha** (-2.72
   against -2.00) while still classifying every trial correctly. Section 2.

## Defects found during this build, and where they are pinned

| Defect | How it was found | Pinned as |
|---|---|---|
| A stub driver wrapped in `GuardedDriver` reported `is_hardware == True`, so dry-run mode claimed 240 hardware operations | `tests/test_dryrun.py::test_dry_run_touches_no_hardware` | `tests/test_regressions.py::test_regression_guarded_stub_is_not_hardware` |
| A least-squares fit over identical epochs returned a skew of exactly zero with an uncertainty of exactly zero | a Hypothesis probe during the build | `tests/test_regressions.py::test_regression_identical_epochs_are_refused` |
| The simplified flicker-FM edf formula is below one degree of freedom at `m = 1` for every N, and is clipped | `tests/test_allan.py` | `tests/test_regressions.py::test_regression_flicker_fm_edf_at_m_one_is_clipped_not_fractional` |

## Reproducing every number

```bash
python -m pytest tests/ -q --junitxml=junit.xml
python validation/regenerate_fixture.py
python validation/validate_symmetric_bias.py
python validation/validate_skew_estimators.py
python validation/validate_allan.py
python validation/validate_uncertainty_coverage.py
python validation/validate_parity.py
python validation/validate_dryrun.py
python validation/validate_reproducibility.py
python validation/validate_asymmetry_benchmark.py
python validation/validate_deployment.py
python validation/validate_benchmark_table.py
python validation/worked_example.py
python validation/validate_cli.py
python validation/validate_cwd_independence.py
```

The last one re-runs every other script from four working directories and takes
roughly three minutes; the rest together take about two.
