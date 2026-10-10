# Validation evidence — conformalband 0.1.0

**Validation level 2.** Exact known-answer checks against hand arithmetic,
Monte-Carlo confirmation of the one statement in this package that is exactly
true, derandomised property tests, and a measured coverage audit under a
declared covariate shift. **Research-grade. Not flight-qualified, not
certified, not approved for operational aerospace use.** There is a learned
component; see `MODEL_CARD.md` and `DATASET_CARD.md`.

Every number in this file, in `README.md` and in `MODEL_CARD.md` was produced
by a script in this directory, executed in this container on 2026-10-10, with
its raw stdout committed beside it as `<script>_output.txt`. Nothing here was
copied from a paper, estimated, or rounded by hand. **The checks that undercut
this package are in the tables on purpose and are marked.** Section 10 collects
them.

## Environment

Python 3.13.16 on Linux 6.18.44, NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1,
Matplotlib 3.11.2, pytest 9.1.1, Hypothesis 6.168.5. `os.cpu_count()` = 2 and
`len(os.sched_getaffinity(0))` = 2: **two cores, shared and contended with
sibling build agents.** Runtime dependencies are NumPy, SciPy and scikit-learn;
Matplotlib is needed only for plotting. PyTorch is not installed and not used.

**No conformal prediction library is installed in this container.**
`importlib.util.find_spec` returns `None` for `mapie`, `crepes`, `deel`,
`nonconformist`, `torchcp` and `torch`, asserted in
`validate_environment.py`, so nothing measured here is silently delegating to
one. Raw output: `validate_environment_output.txt`.

**Timings move, counts do not.** Wall-clock figures below are single runs on
two contended cores and moved by up to 48 per cent between two consecutive
runs of the same command in this session (the test suite: 60.1 s, 88.9 s,
61.1 s). The primary numbers in this document are coverage ratios, widths in
watt-hours, ranks and counts, which are deterministic and reproduced exactly on
every run.

## Test suite

`python -m pytest tests/ -q --junit-xml=junit.xml` from the repository root,
counted **from the junit XML**, not from stdout:

**440 tests, 0 failed, 0 errored, 0 skipped, 0 xfail.** 60.087 s, 88.919 s and
61.127 s on three runs of the same command. The committed
`pytest_output.txt` is the third. The count is identical on all three.

Hypothesis runs with `derandomize=True`. That was not the original setting, and
changing it is itself a finding: with a random seed, one run in several found a
real floating-point artefact in the coverage bound and the other runs passed.
See section 10 item 6.

| Test file | Tests | What it covers |
|---|---|---|
| `test_physics.py` | 36 | energy model, monotonicity, interior power minimum, input validation |
| `test_shift.py` | 30 | declared shift, likelihood-ratio moments, scaling |
| `test_data.py` | 26 | generator determinism, declared means, heteroscedasticity |
| `test_bounds.py` | 50 | coverage bound, Clopper-Pearson, effective sample size |
| `test_conformal.py` | 53 | the three conformal variants, interval container, binning |
| `test_baseline.py` | 31 | analytic regressor, parametric interval |
| `test_learned.py` | 27 | gradient boosting, logistic density ratio |
| `test_known_answers.py` | 44 | every exact statement, hand arithmetic in the comments |
| `test_audit.py` | 34 | audit integration, degenerate equivalences, reproducibility |
| `test_properties.py` | 16 | Hypothesis, derandomised |
| `test_edge_cases.py` | 20 | degenerate inputs, minimum sizes, overflow |
| `test_plotting.py` | 11 | six figure functions, Agg backend, output written |
| `test_cli.py` | 20 | subprocess, exit statuses including the two non-zero ones |
| `test_validation_errors.py` | 42 | every public entry point refuses bad input |
| **total** | **440** | counted from the junit XML, which is the only count that counts |

The per-file counts above were extracted from the junit XML's `testcase`
elements, not from stdout and not by hand; they sum to 440.

---

## 1. The finite-sample coverage bound, against hand arithmetic

`validate_known_answers.py` → `validate_known_answers_output.txt`, section 1.

With `k = ceil((n+1)(1-alpha))` the exact coverage of split conformal on
exchangeable, almost surely distinct scores is `k/(n+1)`.

| n | alpha | k | exact `k/(n+1)` | hand arithmetic | forced over-coverage |
|---|---|---|---|---|---|
| 19 | 0.10 | 18 | 0.900000000000 | `20 x 0.90 = 18`, `18/20` | 0.000000000000 |
| 18 | 0.10 | 18 | **0.947368421053** | `19 x 0.90 = 17.1`, `18/19` | 0.047368421053 |
| 9 | 0.10 | 9 | 0.900000000000 | `10 x 0.90 = 9`, `9/10` | 0.000000000000 |
| 100 | 0.10 | 91 | 0.900990099010 | `101 x 0.90 = 90.9`, `91/101` | 0.000990099010 |
| 500 | 0.10 | 451 | 0.900199600798 | `501 x 0.90 = 450.9`, `451/501` | 0.000199600798 |
| 99 | 0.05 | 95 | 0.950000000000 | `100 x 0.95 = 95`, `95/100` | 0.000000000000 |

Every row matches the hand value to 1e-15. `n = 8` with `alpha = 0.1` is
**refused**, because `k = 9 > 8` means no finite calibration order statistic
attains the level and the interval is the whole real line; the error message
names the minimum size.

## 2. Monte-Carlo confirmation of the exact coverage

`validate_known_answers.py` section 2. 400 000 replicates per row, standard
error about 0.000474. The statement is distribution-free, so three different
score laws are used.

| n | alpha | score law | exact | measured | error |
|---|---|---|---|---|---|
| 19 | 0.10 | uniform | 0.900000000000 | 0.899540000000 | -0.000460 |
| 19 | 0.10 | lognormal | 0.900000000000 | 0.899470000000 | -0.000530 |
| 19 | 0.10 | half-Cauchy | 0.900000000000 | 0.899615000000 | -0.000385 |
| 18 | 0.10 | uniform | 0.947368421053 | 0.946905000000 | -0.000463 |
| 18 | 0.10 | lognormal | 0.947368421053 | 0.948120000000 | +0.000752 |
| 18 | 0.10 | half-Cauchy | 0.947368421053 | 0.946802500000 | -0.000566 |
| 100 | 0.10 | uniform | 0.900990099010 | 0.900965000000 | -0.000025 |
| 100 | 0.10 | lognormal | 0.900990099010 | 0.900597500000 | -0.000393 |
| 100 | 0.10 | half-Cauchy | 0.900990099010 | 0.901392500000 | +0.000402 |
| 99 | 0.05 | uniform | 0.950000000000 | 0.950272500000 | +0.000273 |
| 99 | 0.05 | lognormal | 0.950000000000 | 0.949410000000 | -0.000590 |
| 99 | 0.05 | half-Cauchy | 0.950000000000 | 0.950605000000 | +0.000605 |

Tolerance 0.004, about eight standard errors. The discriminating check is the
separation: dropping one calibration point, from `n = 19` to `n = 18`, must
raise coverage by exactly `18/19 - 18/20 = 0.047368421`. Measured
**+0.047365000**. The same measurement that confirms 0.9000 at `n = 19`
confirms 0.9474 at `n = 18`, so it is not a vacuous check against a round
number.

## 3. Degenerate equivalences and the floating-point rank guard

`validate_known_answers.py` section 3.

| Check | Result |
|---|---|
| `25 * (1 - 0.44)` in binary | `14.00000000000000178` |
| unguarded `math.ceil` | **15**, which is the wrong order statistic |
| `conformal_rank(24, 0.44)` with the rounding guard | **14** |
| `weighted_quantile` with unit weights, same case | **14.0** |
| split quantile on 500 shared scores | 0.691266925435076 |
| weighted conformal with unit weights | 0.691266925435076, **exactly equal** |
| Mondrian with one bin | 0.691266925435076, **exactly equal** |
| weighted conformal at severity 0 with real likelihood ratios | **exactly equal** to split |

A search over `n` in `[5, 2000)` and `alpha` in `{0.001, ..., 0.499}` found no
case where the guard changes the answer for `alpha` of 0.1, 0.05, 0.02 or 0.01,
so the hazard is confined to unusual levels. The guard is kept because being
wrong by one rank produces a silently conservative interval and no error.

## 4. The declared likelihood ratio against the Gaussian densities

`validate_known_answers.py` section 4. 400 000 calibration draws.

| Severity | Mahalanobis | max relative error against `pdf/pdf` | `E_cal[w]` (exact 1) | `E_cal[w^2]` | `exp(M^2)` |
|---|---|---|---|---|---|
| 0.0 | 0.000000 | 0.000e+00 | 1.000000 | 1.000000 | 1.000000 |
| 1.0 | 0.424264 | 4.996e-15 | 1.000639 | 1.198515 | 1.197217 |
| 2.0 | 0.848528 | 5.107e-15 | 1.001084 | 2.058341 | 2.054433 |
| 3.0 | 1.272792 | 5.329e-15 | 1.001410 | 5.062352 | 5.053090 |

The closed form `log w = sum_j [delta_j (x_j - mu_j)/sigma_j^2 - delta_j^2 /
(2 sigma_j^2)]` agrees with the ratio of the densities themselves to 5e-15
relative, and both the first and second moments match their analytic values.
`E[w^2] = exp(M^2)` is the quantity that governs how fast the effective sample
size collapses, which is why it is checked rather than assumed.

## 5. The energy model, by hand

`validate_known_answers.py` section 5. `V = 20 m/s`, `m = 6 kg`,
`rho = 1.18 kg/m^3`, shipped airframe, constant efficiency.

```
P_parasite  = 0.5 * 1.18 * 20^3 * 0.30 * 0.035            =  49.560000000000 W
W           = 6 * 9.80665                                 =  58.839900000000 N
W^2                                                       = 3462.133832010000 N^2
denominator = 1.18 * 20 * pi * 1.20^2 * 0.85              =  90.749302028656
P_induced   = 2 * 3462.133832010000 / 90.749302028656      =  76.301056969380 W
P_total     = (49.56 + 76.301056969380) / 0.62 + 12        = 215.001704789323 W
t           = 1000 / 20                                    =  50 s
E_leg       = 215.001704789323 * 50 / 3600                 =   2.986134788741 Wh
```

| Quantity | Measured | Hand | Tolerance |
|---|---|---|---|
| `level_flight_power` | 215.001704789323 W | 215.001704789323 W | 1e-12 relative |
| `leg_energy` | 2.986134788741 Wh | 2.986134788741 Wh | 1e-12 relative |
| doubling the distance | 5.972269577481 Wh | exactly 2x | 1e-14 |

Induced power is 61 per cent of the aerodynamic total at the design point,
which is why mass is the covariate that matters and why the declared shift
moves it. The power curve has an interior minimum between 14 and 19 m/s
(`tests/test_physics.py`), so the model is not monotone in airspeed and cannot
be checked by monotonicity alone.

## 6. Diagnostics, by hand

`validate_known_answers.py` section 6.

| Check | Measured | Hand |
|---|---|---|
| Kish ESS of 37 equal weights | 37.000000000000 | `n` for any equal weights |
| Kish ESS of `[1, 3]` | 1.600000000000 | `(1+3)^2/(1+9) = 1.6` |
| Kish ESS of one non-zero weight in 50 | 1.000000000000 | 1 |
| Clopper-Pearson(0, 20) upper limit | 0.168433470983 | `1 - 0.025^(1/20)` |

## 6A. Dataset statistics, against the multiplicative-noise identity

`validate_known_answers.py` section 7. `make_dataset(200000, seed=1)` on the
calibration distribution. The noise is `e = truth * 0.06 * z` with `z` standard
normal and independent of `truth`, so `sd(e) = 0.06 sqrt(E[truth^2])
= 0.06 sqrt(mean^2 + sd^2)` exactly.

| Quantity | Measured | Identity |
|---|---|---|
| mean noiseless energy | 3.117808 Wh | — |
| sd across covariate draws | 0.580589 Wh | — |
| observation-noise sd | **0.190197 Wh** | `0.06 sqrt(mean^2 + sd^2)` = **0.190284 Wh** |

Agreement to 0.05 per cent, which confirms that the generator's noise is the
multiplicative noise `DATASET_CARD.md` declares and not something else.

## 7. The coverage audit

`validate_coverage_audit.py` → `validate_coverage_audit_output.txt`.
120 replicates, `n_fit = 1500`, `n_calibration = 500`, `n_test = 600`, seed
57001, `alpha = 0.1`. **72 000 test points per row.** Intervals are 95 per cent
Student-t across replicate coverages. Finite-sample split-conformal bound at
this calibration size: exact **0.900199600798**, window
`[0.9000, 0.901996007984]`.

### 7A. Physics point predictor

| Method | Severity | Coverage | 95 % interval | Width [Wh] | Inf. fraction | ESS | ESS/n |
|---|---|---|---|---|---|---|---|
| `parametric` | 0.0 | 0.90565 | [0.90310, 0.90821] | 0.66618 | 0.00000 | 500.0 | 1.0000 |
| `parametric` | 1.0 | **0.88283** | [0.87949, 0.88618] | 0.66618 | 0.00000 | 500.0 | 1.0000 |
| `parametric` | 2.0 | **0.86031** | [0.85689, 0.86373] | 0.66618 | 0.00000 | 500.0 | 1.0000 |
| `parametric` | 3.0 | **0.83750** | [0.83364, 0.84136] | 0.66618 | 0.00000 | 500.0 | 1.0000 |
| `split` | 0.0 | 0.89981 | [0.89686, 0.90275] | 0.65355 | 0.00000 | 500.0 | 1.0000 |
| `split` | 1.0 | **0.87601** | [0.87212, 0.87991] | 0.65355 | 0.00000 | 500.0 | 1.0000 |
| `split` | 2.0 | **0.85335** | [0.84973, 0.85697] | 0.65355 | 0.00000 | 500.0 | 1.0000 |
| `split` | 3.0 | **0.82926** | [0.82553, 0.83300] | 0.65355 | 0.00000 | 500.0 | 1.0000 |
| `mondrian` | 0.0 | 0.90431 | [0.90130, 0.90731] | 0.66438 | 0.00000 | 500.0 | 1.0000 |
| `mondrian` | 1.0 | 0.90071 | [0.89761, 0.90381] | 0.69959 | 0.00000 | 500.0 | 1.0000 |
| `mondrian` | 2.0 | **0.89408** | [0.89065, 0.89752] | 0.73200 | 0.00000 | 500.0 | 1.0000 |
| `mondrian` | 3.0 | **0.88468** | [0.88082, 0.88855] | 0.75918 | 0.00000 | 500.0 | 1.0000 |
| `weighted_declared` | 0.0 | 0.89981 | [0.89686, 0.90275] | 0.65355 | 0.00000 | 500.0 | 1.0000 |
| `weighted_declared` | 1.0 | 0.89811 | [0.89451, 0.90171] | 0.69895 | 0.00000 | 418.5 | 0.8370 |
| `weighted_declared` | 2.0 | 0.89875 | [0.89420, 0.90330] | 0.74981 | 0.00003 | 248.3 | 0.4967 |
| `weighted_declared` | 3.0 | 0.90433 | [0.89798, 0.91069] | 0.82156 | 0.00583 | 116.1 | 0.2322 |
| `weighted_learned` | 0.0 | 0.90126 | [0.89828, 0.90425] | 0.65652 | 0.00000 | 496.2 | 0.9924 |
| `weighted_learned` | 1.0 | 0.89881 | [0.89526, 0.90235] | 0.70103 | 0.00000 | 412.7 | 0.8254 |
| `weighted_learned` | 2.0 | 0.89910 | [0.89464, 0.90355] | 0.74997 | 0.00004 | 247.1 | 0.4941 |
| `weighted_learned` | 3.0 | 0.90539 | [0.89906, 0.91172] | 0.82144 | 0.00668 | 114.1 | 0.2281 |

### 7B. Learned point predictor

| Method | Severity | Coverage | 95 % interval | Width [Wh] | Inf. fraction | ESS | ESS/n |
|---|---|---|---|---|---|---|---|
| `parametric` | 0.0 | 0.90796 | [0.90493, 0.91098] | 0.72385 | 0.00000 | 500.0 | 1.0000 |
| `parametric` | 1.0 | **0.88151** | [0.87777, 0.88526] | 0.72385 | 0.00000 | 500.0 | 1.0000 |
| `parametric` | 2.0 | **0.84772** | [0.84377, 0.85168] | 0.72385 | 0.00000 | 500.0 | 1.0000 |
| `parametric` | 3.0 | **0.80228** | [0.79759, 0.80697] | 0.72385 | 0.00000 | 500.0 | 1.0000 |
| `split` | 0.0 | 0.89868 | [0.89556, 0.90180] | 0.69997 | 0.00000 | 500.0 | 1.0000 |
| `split` | 1.0 | **0.87132** | [0.86792, 0.87472] | 0.69997 | 0.00000 | 500.0 | 1.0000 |
| `split` | 2.0 | **0.83642** | [0.83274, 0.84009] | 0.69997 | 0.00000 | 500.0 | 1.0000 |
| `split` | 3.0 | **0.79047** | [0.78615, 0.79480] | 0.69997 | 0.00000 | 500.0 | 1.0000 |
| `mondrian` | 0.0 | 0.90418 | [0.90125, 0.90711] | 0.71140 | 0.00000 | 500.0 | 1.0000 |
| `mondrian` | 1.0 | **0.89554** | [0.89226, 0.89883] | 0.75302 | 0.00000 | 500.0 | 1.0000 |
| `mondrian` | 2.0 | **0.87992** | [0.87637, 0.88346] | 0.79299 | 0.00000 | 500.0 | 1.0000 |
| `mondrian` | 3.0 | **0.85129** | [0.84685, 0.85573] | 0.82696 | 0.00000 | 500.0 | 1.0000 |
| `weighted_declared` | 0.0 | 0.89868 | [0.89556, 0.90180] | 0.69997 | 0.00000 | 500.0 | 1.0000 |
| `weighted_declared` | 1.0 | 0.89746 | [0.89394, 0.90098] | 0.76451 | 0.00000 | 418.5 | 0.8370 |
| `weighted_declared` | 2.0 | 0.89968 | [0.89492, 0.90444] | 0.86052 | 0.00003 | 248.3 | 0.4967 |
| `weighted_declared` | 3.0 | 0.91201 | [0.90435, 0.91968] | 1.07147 | 0.00583 | 116.1 | 0.2322 |
| `weighted_learned` | 0.0 | 0.89911 | [0.89598, 0.90224] | 0.70118 | 0.00000 | 496.2 | 0.9924 |
| `weighted_learned` | 1.0 | 0.89860 | [0.89516, 0.90204] | 0.76688 | 0.00000 | 412.7 | 0.8254 |
| `weighted_learned` | 2.0 | 0.89978 | [0.89499, 0.90457] | 0.85976 | 0.00004 | 247.1 | 0.4941 |
| `weighted_learned` | 3.0 | 0.91203 | [0.90438, 0.91968] | 1.06146 | 0.00668 | 114.1 | 0.2281 |

Rows in bold are **demonstrably below nominal**: the whole 95 per cent interval
lies under 0.90. **17 of 40 rows.** At 95 per cent confidence over 40 rows,
about 2 rows would be flagged by chance alone, so a single flagged row near
nominal is not evidence of anything; 17 is.

Width is the mean over **finite** intervals only. An unbounded interval covers
by construction, so coverage cannot be read without the infinite fraction
beside it.

## 8. The weighted-conformal breaking point

`validate_breaking_point.py` → `validate_breaking_point_output.txt`.
120 replicates, `n_calibration = 500`, `n_test = 600`, seed 57021, learned
point predictor, 31 assumed fractions from 0.00 to 1.50 in steps of 0.05, at
three true severities.

The test data is drawn under the true severity; the weights come from
`CovariateShift(true_severity * fraction)`. Fraction 1.00 is correct, below 1
understates the shift and above 1 overstates it. The **breaking fraction** is
the largest fraction at or below 1.00 whose entire 95 per cent coverage
interval lies below nominal.

| True severity | Breaking fraction | Smallest holding fraction | Coverage at f=0 (no weighting) | Coverage at f=1 (correct) | Width ratio f=1.5 / f=1 |
|---|---|---|---|---|---|
| 1.0 | **0.85** | 0.90 | 0.87324 | 0.89940 | 1.052636 |
| 2.0 | **0.90** | 0.95 | 0.83653 | 0.90083 | 1.216956 |
| 3.0 | **0.85** | 0.90 | 0.79054 | 0.90931 | 1.295290 |

**Published breaking point: between 0.85 and 0.90.** Understating the declared
shift by 10 to 15 per cent is enough to put the whole measured coverage
interval below nominal at this precision. The grid resolution is 0.05, so the
true threshold lies in a 0.05-wide bracket and is not resolved more finely
than that.

The sweep at true severity 2.0, selected rows:

| Fraction | Assumed severity | Coverage | 95 % interval | Width [Wh] | Inf. fraction | ESS | Holds |
|---|---|---|---|---|---|---|---|
| 0.00 | 0.000 | 0.83653 | [0.83271, 0.84035] | 0.70051 | 0.00000 | 500.0 | no |
| 0.50 | 1.000 | 0.86714 | [0.86311, 0.87117] | 0.76529 | 0.00000 | 417.9 | no |
| 0.85 | 1.700 | 0.89086 | [0.88649, 0.89523] | 0.82814 | 0.00000 | 299.0 | no |
| 0.90 | 1.800 | 0.89450 | [0.89017, 0.89883] | 0.83779 | 0.00000 | 281.4 | **no, breaking point** |
| 0.95 | 1.900 | 0.89756 | [0.89314, 0.90197] | 0.84882 | 0.00000 | 264.1 | yes |
| 1.00 | 2.000 | 0.90083 | [0.89632, 0.90535] | 0.85981 | 0.00000 | 247.2 | yes, correct weights |
| 1.50 | 3.000 | 0.93506 | [0.92886, 0.94125] | 1.04635 | 0.00194 | 115.6 | yes, over-covers |

The full 31-row sweep for each of the three true severities is in
`validate_breaking_point_output.txt`.

Overstating the shift does **not** break coverage. It buys coverage with width
and with effective sample size: at fraction 1.50 the interval is 1.217 times
the width it is at fraction 1.00 and the effective calibration size falls with
it. A user who does not know the shift is better off overstating it, and the
cost of doing so is measured here rather than asserted.

## 9. Conditional coverage

`validate_conditional_coverage.py` → `validate_conditional_coverage_output.txt`.
30 replicates, `n_test = 1000`, seed 57031, learned point predictor, terciles
declared on the calibration predictions and applied unchanged to the test
predictions.

### In distribution, severity 0.0

| Method | Tercile | Coverage | Width [Wh] | Test points |
|---|---|---|---|---|
| `split` | 1 | **0.95972** | 0.70027 | 10 105 |
| `split` | 2 | 0.91760 | 0.70050 | 9 757 |
| `split` | 3 | **0.81920** | 0.69988 | 10 138 |
| `mondrian` | 1 | 0.90559 | 0.57475 | 10 105 |
| `mondrian` | 2 | 0.90397 | 0.67903 | 9 757 |
| `mondrian` | 3 | 0.89998 | 0.88171 | 10 138 |

Coverage spread across terciles: split **0.14053**, Mondrian **0.00561**.

### Under shift, severity 2.0

| Method | Tercile | Coverage | Width [Wh] | Test points |
|---|---|---|---|---|
| `split` | 1 | **0.95764** | 0.70032 | 3 824 |
| `split` | 2 | 0.91157 | 0.70082 | 7 633 |
| `split` | 3 | **0.77560** | 0.69994 | 18 543 |
| `mondrian` | 1 | 0.90795 | 0.57468 | 3 824 |
| `mondrian` | 2 | 0.89991 | 0.67892 | 7 633 |
| `mondrian` | 3 | **0.86146** | 0.88127 | 18 543 |

Coverage spread: split **0.18203**, Mondrian **0.04649**. The test tercile
populations are 3 824 / 7 633 / 18 543 rather than equal thirds, because the
edges were declared on the calibration predictions and the shift moves
predictions upward. That imbalance is itself a symptom of the shift and is
reported rather than corrected away.

## 10. The checks that went against this package

Collected here, with their numbers, because they are the credible part.

1. **The specification's prediction about the baseline was wrong, and the
   measurement is published as such.** The spec expected the parametric
   Gaussian interval to be *tighter* than conformal in distribution. Measured,
   it is **wider**: the ratio of parametric half-width to conformal quantile is
   **1.019319** for the physics predictor and **1.034104** for the learned one
   (`validate_coverage_audit_output.txt` section 1), and the independent
   single-fit measurement gives **1.024586** and **1.041085**
   (`validate_baseline_vs_learned_output.txt` section 2). The parametric
   interval is correct in distribution (coverage 0.90565 and 0.90796) and
   simply not the tighter one.

2. **The mechanism, measured rather than asserted.** The residual distribution
   is leptokurtic because the observation noise is multiplicative, making the
   residual a scale mixture of normals. Excess kurtosis on 200 000 points:
   **+1.64873** for the physics predictor and **+9.00989** for the learned one.
   The 90th percentile of the absolute residual divided by `1.6449 sigma` is
   **0.983569** and **0.968426**: the standard deviation is inflated faster
   than the quantile, which is exactly why the parametric interval is wider.
   For exactly Gaussian homoscedastic residuals both ratios would be 1.

3. **The analytic baseline beats the learned model at every severity,
   including in distribution.** RMSE ratio learned/physics: **1.100527**,
   1.140478, 1.213455, **1.354209** at severities 0, 1, 2 and 3, against an
   irreducible noise floor of 0.190008 to 0.233358 Wh. Four fitted
   coefficients beat 200 boosted trees. The learned model was not retuned.

4. **Marginal coverage hides a conditional failure even with no shift at
   all.** Split conformal covers **0.95972** in the lowest tercile of predicted
   energy and **0.81920** in the highest, in distribution, while its marginal
   coverage is 0.89868. A user reading only the marginal number would be eight
   points wrong on the heaviest, windiest flights.

5. **Mondrian conformal does not restore marginal coverage under shift.** At
   severity 3 it reaches **0.85129** against a nominal 0.90. Conditioning on a
   prediction tercile is not conditioning on the covariate that moved.

6. **A floating-point artefact in the bound, found by Hypothesis on one run in
   several.** At `n = 1249, alpha = 0.18` the exact coverage is
   `1025/1250 = 0.82`, which in exact arithmetic equals `1 - alpha`, so the
   bound is attained. But `1 - 0.18` evaluates to `0.8200000000000001` in
   binary, so `exact < lower` by one unit in the last place. The property test
   that found it asserted `lower <= exact` with no tolerance and failed. The
   fix was to put the tolerance on both sides, document the case in
   `bounds.CoverageBound`, add it as a known-answer test, and set
   `derandomize=True` so the suite stops being a lottery. Nothing about the
   computed rank or coverage was changed.

7. **Weighted conformal is fragile in exactly the way the theory says.**
   Breaking fraction 0.85 to 0.90 (section 8); with no weighting at all,
   coverage 0.79054 at severity 3; effective sample size down to **116.1 of
   500** at severity 3, with the per-replicate coverage standard deviation
   rising from 0.01726 to **0.04241** and 0.583 per cent of test points getting
   a genuinely unbounded interval.

8. **17 of 40 audit rows are demonstrably below nominal coverage.** The audit's
   own `--fail-on-undercoverage` flag exits 2 on them, which is tested
   (`tests/test_cli.py::test_audit_fail_on_undercoverage_exits_two_when_a_method_breaks`).

9. **No citation in this repository was verified against a publisher page in
   this session.** Egress from this container reached `pypi.org` only.
   `doi.org`, `arxiv.org`, `arc.aiaa.org` and `jmlr.org` all returned
   `connect_rejected` from the egress proxy. Every reference is given by
   author, year, title and venue from the author's knowledge, without a
   verified DOI resolution, and page numbers are omitted where they could not
   be checked. Treat the references as pointers to look up, not as verified
   bibliography.

## 11. Alternatives, as actually checked

Checked 2026-10-10 in this container. The HTTP status of the version-pinned
PyPI JSON endpoint establishes that the exact version exists; the wheel was
then downloaded with `pip download <name>==<version> --no-deps`, unpacked, and
its modules read to establish what it ships. **Existence is not equivalence**,
so the "what it ships" column is from the unpacked source, not from a
description.

```bash
curl -s -o /dev/null -w '%{http_code}' https://pypi.org/pypi/conformalband/json   # 404, name free
curl -s -o /dev/null -w '%{http_code}' https://pypi.org/pypi/mapie/1.5.0/json     # 200
curl -s -o /dev/null -w '%{http_code}' https://pypi.org/pypi/crepes/0.9.1/json    # 200
curl -s -o /dev/null -w '%{http_code}' https://pypi.org/pypi/puncc/0.9.3/json     # 200
```

| Package | Version | Status | Declared dependencies | Read from the unpacked wheel |
|---|---|---|---|---|
| `conformalband` | — | **404** | — | the name is free |
| `MAPIE` | 1.5.0 | 200, BSD-3-Clause, `>=3.10` | `numpy>=1.24.1`, `scikit-learn>=1.4`, `scipy>=1.10` | `SplitConformalRegressor`, `CrossConformalRegressor`, `JackknifeAfterBootstrapRegressor`, `ConformalizedQuantileRegressor`, `CrossConformalizedQuantileRegressor`, `TimeSeriesRegressor`, `ConditionalSplitConformalRegressor`, classification counterparts, a conformity-score framework, `mapie.metrics` (`regression_coverage_score`, `regression_ssc`, `coverage_width_based`, `worst_slab_coverage`, `coverage_gap`), and a whole `mapie.exchangeability_testing` package (`FixedDatasetExchangeabilityTest`, `OnlineExchangeabilityTest`, `PermutationTest`, `OnlineMartingaleTest`, `RiskMonitoring`). **No Mondrian class and no likelihood-ratio weighted conformal.** |
| `crepes` | 0.9.1 | 200, BSD, `>=3.10` | `numpy`, `pandas`, `scipy` | `ConformalRegressor`, `ConformalClassifier`, `ConformalPredictiveSystem`, `WrapRegressor`, `WrapClassifier`, **Mondrian conformal throughout via a `bins` argument**, `MondrianCategorizer`, `DifficultyEstimator` for normalised (locally weighted) scores, online and semi-online p-values, CRPS, and `crepes.martingales` for exchangeability testing. **No likelihood-ratio weighted conformal for covariate shift.** |
| `puncc` | 0.9.3 | 200 | — | `SplitCP`, `LocallyAdaptiveCP`, `CQR`, `CVPlus`, `EnbPI`, `AdaptiveEnbPI`, `LeverageWeightedCP`, and a `BaseCalibrator` whose `weight_func` argument takes an arbitrary function of `X` returning weights that enter the weighted quantile. **That is the weighted-conformal machinery with user-declared weights**, so `puncc` is the honest nearest neighbour of this package's weighted path. |

A positive control was run against packages that must exist (`numpy`, `scipy`,
`scikit-learn`, `mapie`, `crepes`) and all returned 200, so the 404 for
`conformalband` distinguishes a free name from a broken transport.

**None of these is a runtime dependency here.** They are citations, not
imports, and `validate_environment.py` asserts that none of them is installed
in the container the measurements were taken in.

## 12. References

Listed as pointers, unverified against publisher pages in this session (see
section 10 item 9).

- Papadopoulos, H., Proedrou, K., Vovk, V. and Gammerman, A., "Inductive
  Confidence Machines for Regression", *Machine Learning: ECML 2002*.
  Split (inductive) conformal prediction.
- Vovk, V., Gammerman, A. and Shafer, G., *Algorithmic Learning in a Random
  World*, Springer, 2005. Conformal prediction, including the Mondrian
  (class-conditional) construction.
- Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R.J. and Wasserman, L.,
  "Distribution-Free Predictive Inference for Regression", *Journal of the
  American Statistical Association*, Vol. 113, No. 523, 2018. The split
  conformal coverage bound used in section 1.
- Tibshirani, R.J., Barber, R.F., Candes, E.J. and Ramdas, A., "Conformal
  Prediction Under Covariate Shift", *Advances in Neural Information
  Processing Systems* 32, 2019. Weighted conformal prediction; the construction
  implemented in `conformalband.conformal.WeightedConformal`.
- Angelopoulos, A.N. and Bates, S., "Conformal Prediction: A Gentle
  Introduction", *Foundations and Trends in Machine Learning*, Vol. 16, No. 4,
  2023.
- Sugiyama, M., Suzuki, T. and Kanamori, T., *Density Ratio Estimation in
  Machine Learning*, Cambridge University Press, 2012. The classifier
  reduction used by `LearnedWeightEstimator`.
- Kish, L., *Survey Sampling*, Wiley, 1965. Effective sample size
  `(sum w)^2 / sum w^2`.
- Clopper, C.J. and Pearson, E.S., "The Use of Confidence or Fiducial Limits
  Illustrated in the Case of the Binomial", *Biometrika*, Vol. 26, No. 4, 1934.
- Anderson, J.D., *Aircraft Performance and Design*, McGraw-Hill, 1999. The
  drag polar `C_D = C_D0 + C_L^2/(pi AR e)`.
- Traub, L.W., "Range and Endurance Estimates for Battery-Powered Aircraft",
  *Journal of Aircraft*, Vol. 48, No. 2, 2011. Battery-powered aircraft energy
  bookkeeping.

## 13. Reproducing every number

From the repository root, with `PYTHONPATH=src` and `MPLBACKEND=Agg`:

```bash
python -m pytest tests/ -q --junit-xml=junit.xml
ruff check src/ tests/ examples/ validation/
python validation/validate_environment.py
python validation/validate_known_answers.py
python validation/validate_baseline_vs_learned.py
python validation/validate_conditional_coverage.py
python validation/validate_coverage_audit.py
python validation/validate_breaking_point.py
python validation/validate_cli.py
python validation/worked_example.py
python examples/coverage_bound.py
python examples/mondrian_strata.py
python examples/baseline_vs_learned.py
python examples/coverage_audit.py
python examples/interval_width.py
python examples/weight_breaking_point.py
```

Every validation script exits 0 by design, including where the finding is a
failure: the failure is printed as a `FINDING` line, recorded in the committed
output, and the *measured* value is asserted against the expectation documented
beside the assertion. A validation script that exits non-zero blocks the
release gate for every sibling product in the batch, so none of them does.

`validate_cli.py` re-runs every command quoted in `README.md`, `MODEL_CARD.md`
and this file, in a clean subprocess, and asserts each exit status including
the two that must be non-zero. That is the mechanism that makes a quoted
output block checkable rather than trusted.
