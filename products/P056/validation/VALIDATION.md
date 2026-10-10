# Validation evidence — calibaudit 0.1.0

**Validation level 2.** Closed-form population values of synthetic
forecasters, hand-computed known answers, algebraic identities property-tested
to machine precision, cross-implementation agreement with scikit-learn 1.9.1,
and measured bias and coverage of the estimators this package reports.
**Research-grade. Not flight-qualified, not certified, not approved for
operational aerospace use.**

Every number in this file, in README.md, in MODEL_CARD.md and in
DATASET_CARD.md was produced by a script in this directory, executed in this
container on 2026-10-10, with its raw stdout committed as
`outputs/<script>_output.txt`. Nothing was copied from a paper, estimated, or
rounded by hand. **The checks that undercut this package are in section 9 on
purpose.**

## 0. Environment, compute budget and conventions

`validate_environment.py` → `outputs/validate_environment_output.txt`.

| | |
|---|---|
| python | 3.13.16 (main, Oct 1 2026, 15:40:24) [GCC 13.3.0] |
| platform | Linux-6.18.44-fc-v114-x86_64-with-glibc2.39 |
| cores | `os.cpu_count()` = 2, `len(os.sched_getaffinity(0))` = **2**, shared and contended with sibling build agents |
| numpy | 2.5.3 |
| scipy | 1.18.1 |
| scikit-learn | 1.9.1 |
| matplotlib | 3.11.2 |
| pytest / hypothesis | 9.1.1 / 6.168.5 |

Runtime dependencies are `numpy`, `scipy` and `scikit-learn`. `matplotlib` is
needed only for `calibaudit.plotting` and `examples/`, and is declared as an
extra.

**Every validation script exits 0, including when a check fails.** The release
gate requires exit 0 from every script under `validation/`, and a failing
*check* is a finding to be recorded, not a reason to break a build. Failures
are printed in a `!!!` block, counted in the summary line, and listed again at
the end of the committed output. `validation/_harness.py` carries that
contract and the reasoning. A script that fails to *run* is a different thing
and surfaces as a traceback, which is not caught.

**Timings move, counts do not.** Wall-clock figures below are single runs on
two contended cores and move by 10–40 % between runs. The primary numbers in
this document are residuals, counts, biases, coverage rates and harm rates,
which are deterministic in the recorded seeds and reproduce exactly.

**Compute budget.** The slowest committed script is
`validate_reliability_bands.py` at about 81 s. The full test suite is 405
tests in 146 s in the committed run. Nothing approaches the mission's
three-minute per-run
budget.

**Test suite.** `python -m pytest tests/ -q --junitxml=junit.xml` from the
repository root, counted from the junit XML rather than from stdout:
**405 tests, 0 failures, 0 errors, 0 skipped**, `time="145.841"`. Raw stdout:
`outputs/pytest_output.txt`. Ruff clean over `src/ tests/ examples/
validation/` with the configuration in `pyproject.toml`.

## 1. The decomposition identities

`validate_decomposition_identity.py` →
`outputs/validate_decomposition_identity_output.txt`. **Failed checks: 0.**

### 1.1 What is being claimed

Murphy (1973) partitions the Brier score of a forecast with finitely many
distinct values:

```
BS  = REL - RES + UNC
REL = sum_k (n_k/n) (f_k - o_k)^2        reliability, lower is better
RES = sum_k (n_k/n) (o_k - obar)^2       resolution, higher is better
UNC = obar (1 - obar)                    a property of the outcomes alone
```

Applying the same three formulas to *binned* continuous forecasts is not an
identity, because the forecast is no longer constant inside a bin. Writing
`f_i = f_k + d_i` with `sum_{i in k} d_i = 0`, using `o_i in {0,1}` so that
`mean_{i in k}(f_k - o_i)^2 = (f_k - o_k)^2 + o_k(1 - o_k)`, and applying the
law of total variance `sum_k (n_k/n) o_k(1-o_k) = UNC - RES`:

```
BS  = REL - RES + UNC + WBV - 2 WBC
WBV = (1/n) sum_i (f_i - f_k(i))^2       within-bin variance
WBC = (1/n) sum_i (f_i - f_k(i)) o_i     within-bin covariance
```

`WBV` and `WBC` vanish exactly when the forecast is constant in every occupied
bin, which is exactly when the three-term form holds. They are the part a
three-term report drops, and this package reports them.

### 1.2 Measured residuals

| Check | Cases | Worst `|residual|` | Where | Tolerance |
|---|---|---|---|---|
| exact three-term identity, distinct-value grouping | 72 (6 specs × 4 sizes × 3 roundings) | **1.110223e-16** | overconfident, n = 100, 3 digits | 1e-14 |
| five-term identity, binned | 480 (6 specs × 4 sizes × 10 bin counts × 2 strategies) | **1.720846e-15** | overconfident, n = 8000, B = 2, equal_width | 1e-14 |
| five-term identity, adversarial random inputs | 600 | **6.106227e-16** | — | 1e-14 |

The adversarial families are uniform forecasts, forecasts drawn only from
{0, 1}, forecasts rounded to one decimal (heavy ties), and forecasts
concentrated in a 0.05-wide band near 0.5, with bin counts from 1 to 59 and
both strategies, drawn from `default_rng(56)`.

`WBV` and `WBC` are **identically 0.0**, not merely small, for the exact
decomposition: it is passed the distinct forecast values rather than the group
means, because a floating-point mean of identical values differs from the
value itself by about one ulp and would leave 1e-32 in `WBV`. That change was
made after the first version of the test measured exactly that.

Hypothesis property tests (`tests/test_properties.py`, 250 examples each) hold
the same identities over arbitrary forecast-outcome lists and bin counts, at
1e-13.

### 1.3 How much a three-term report drops

Spec `overconfident`, n = 8000, `BS = 0.20200117`:

| B | strategy | `REL-RES+UNC` | WBV | WBC | three-term residual | % of BS |
|---|---|---|---|---|---|---|
| 2 | equal_width | 0.21540733 | 0.02119816 | 0.01730216 | −0.01340616 | **6.6367** |
| 5 | equal_width | 0.20536751 | 0.00330107 | 0.00333370 | −0.00336633 | 1.6665 |
| 10 | equal_width | 0.20261027 | 0.00081553 | 0.00071231 | −0.00060909 | 0.3015 |
| 20 | equal_width | 0.20191022 | 0.00020406 | 0.00005655 | +0.00009095 | 0.0450 |
| 50 | equal_width | 0.20199478 | 0.00003306 | 0.00001333 | +0.00000640 | 0.0032 |
| 200 | equal_width | 0.20201941 | 0.00000203 | 0.00001014 | −0.00001824 | 0.0090 |
| 2 | equal_mass | 0.21514219 | 0.02120355 | 0.01717228 | −0.01314102 | 6.5054 |
| 10 | equal_mass | 0.20229620 | 0.00083907 | 0.00056705 | −0.00029503 | 0.1461 |
| 50 | equal_mass | 0.20200053 | 0.00003394 | 0.00001665 | +0.00000064 | 0.0003 |
| 200 | equal_mass | 0.20201306 | 0.00000226 | 0.00000707 | −0.00001189 | 0.0059 |

The residual is **not** monotone in the bin count: it falls as bins narrow and
`WBV` shrinks, then rises again as bins empty and `WBC` becomes noisy. The
sign changes too. Anyone quoting a three-term decomposition at ten bins is
quoting a number with a 0.3 % error of unknown sign.

### 1.4 The terms against their population values

n = 200 000, 50 equal-mass bins, seed `56 * 3000`:

| Spec | `UNC` pop | `UNC` meas | `RES` pop | `RES` meas | `REL` pop | `REL` meas | `BS` pop | `BS` meas |
|---|---|---|---|---|---|---|---|---|
| calibrated | 0.250000 | 0.250000 | 0.050000 | 0.050237 | 0 | 4.731e-05 | 0.200000 | 0.199744 |
| calibrated_uniform | 0.250000 | 0.249999 | 0.083333 | 0.083123 | 0 | 2.711e-05 | 0.166667 | 0.166875 |
| calibrated_rare | 0.090000 | 0.089860 | 0.008182 | 0.008215 | 0 | 1.603e-05 | 0.081818 | 0.081582 |
| overconfident | 0.250000 | 0.250000 | 0.050000 | 0.050237 | 6.778e-03 | 6.834e-03 | 0.206778 | 0.206539 |
| underconfident | 0.250000 | 0.250000 | 0.050000 | 0.050237 | 4.562e-03 | 4.651e-03 | 0.204562 | 0.204359 |
| biased_high | 0.250000 | 0.250000 | 0.050000 | 0.050237 | 1.498e-02 | 1.491e-02 | 0.214976 | 0.214618 |

Worst `UNC` difference **1.400306e-04** against a documented tolerance of 5e-3
(five standard errors of a base-rate estimate at this n). Worst `RES`
difference **2.369132e-04** against 1e-2.

**Note the `REL` column for the three calibrated specs: the population value
is exactly 0 and the binned estimate is 1.6e-05 to 4.7e-05.** The reliability
term has the same upward binning bias as the ECE, for the same reason, and at
n = 200 000 with 50 bins it is still there. That is not a separate finding; it
is section 2 seen in a different statistic.

## 2. The ECE estimator's own bias — the product's main claim

`validate_ece_bias.py` → `outputs/validate_ece_bias_output.txt`.
**Failed checks: 2, both deliberately retained — see section 9.**

### 2.1 What is being claimed, and why it needs a synthetic reference

The population quantity is `ECE = E_f |E[o|f] - f|`. The estimator replaces
the conditional expectation by a bin average. `obar_k` is a mean of `n_k`
Bernoulli draws, so it misses `fbar_k` by about
`sqrt(fbar_k(1-fbar_k)/n_k)` even for a perfect forecaster, and the absolute
value turns that noise into a contribution that cannot cancel. On a perfectly
calibrated forecaster the population ECE is **exactly zero**, so every digit
the estimator reports is bias, and the bias is directly measurable.

`netcal.metrics.ECE` computes the same plug-in estimator. The difference this
package claims is not the metric; it is the measurement of the metric's error.

### 2.2 Spec `calibrated`, equal-width bins, 120 replicates per cell

Population ECE = 0 exactly. `sem` is the standard error of the cell mean.

| B \\ n | 200 | 1000 | 5000 | 20000 |
|---|---|---|---|---|
| 5 | **0.054061** ± 0.001663 | 0.022649 ± 0.000760 | 0.010230 ± 0.000346 | 0.005315 ± 0.000183 |
| 10 | 0.077287 ± 0.001805 | 0.032111 ± 0.000793 | 0.014266 ± 0.000287 | 0.007510 ± 0.000181 |
| 20 | 0.106881 ± 0.001756 | 0.044913 ± 0.000777 | 0.020159 ± 0.000354 | 0.010349 ± 0.000191 |
| 50 | 0.160791 ± 0.001559 | 0.071889 ± 0.000808 | 0.032160 ± 0.000392 | 0.016483 ± 0.000188 |
| 100 | **0.217391** ± 0.001716 | 0.101510 ± 0.000787 | 0.045650 ± 0.000391 | 0.023248 ± 0.000196 |

The practitioner's default of ten bins on a thousand cases reports
**0.032111** for a forecaster that is perfect.

### 2.3 The scaling law, with the exponent stated in advance

The leading Bernoulli-noise term scales as `sqrt(B/n)`, so the predicted
exponent is **0.5**. Fitted by least squares on `log bias` against
`log(B/n)` over each 5 × 4 grid:

| Spec | strategy | fitted exponent | intercept | R² |
|---|---|---|---|---|
| calibrated | equal_width | **0.496599** | −1.136902 | 0.999282 |
| calibrated_uniform | equal_width | 0.494336 | −1.188005 | 0.999455 |
| calibrated_rare | equal_width | 0.480494 | −1.972351 | 0.999356 |
| calibrated | equal_mass | 0.508607 | −1.002660 | 0.999728 |
| calibrated_uniform | equal_mass | 0.502992 | −1.136498 | 0.999693 |
| calibrated_rare | equal_mass | 0.492992 | −1.598473 | 0.999347 |

Range of exponents **0.480494 to 0.508607** against the predicted 0.5, with
R² between 0.999282 and 0.999728. The `bias/sqrt(B/n)` column in the raw
output is near-constant at **0.307 to 0.346** for `calibrated`/equal_width,
which is the same statement without a fit.

The intercepts differ by spec: `calibrated_rare` sits about 2.3× lower on
equal-width bins than `calibrated`, because most of its forecast mass is in
few bins. **The constant is not universal; only the exponent is.** Anyone
using `0.32 sqrt(B/n)` as a rule of thumb should measure their own constant,
which is what `null_ece_distribution` does.

### 2.4 Headline cells, for quoting

Spec `calibrated`, equal-width, 120 replicates, population ECE 0:

| B | n | mean ECE | sem |
|---|---|---|---|
| 10 | 200 | 0.077287 | 0.001805 |
| 15 | 200 | 0.092625 | 0.001708 |
| 50 | 200 | 0.160791 | 0.001559 |
| 10 | 20000 | 0.007462 | 0.000183 |
| 15 | 20000 | 0.009054 | 0.000172 |
| 50 | 20000 | 0.016408 | 0.000184 |

### 2.5 Miscalibrated specs, where the bias is a difference

60 replicates per cell, equal-width bins, population values from
`scipy.integrate.quad` with the `abserr` shown:

| Spec | B | n | population | quad abserr | mean ECE | bias | rel bias % |
|---|---|---|---|---|---|---|---|
| overconfident | 15 | 1000 | 0.075204 | 1.79e-10 | 0.085421 | **+0.010217** | +13.586 |
| overconfident | 15 | 20000 | 0.075204 | 1.79e-10 | 0.074878 | **−0.000327** | −0.434 |
| overconfident | 50 | 1000 | 0.075204 | 1.79e-10 | 0.105986 | **+0.030782** | +40.931 |
| overconfident | 50 | 20000 | 0.075204 | 1.79e-10 | 0.076170 | +0.000966 | +1.284 |
| underconfident | 15 | 1000 | 0.059354 | 3.04e-10 | 0.066112 | +0.006758 | +11.386 |
| underconfident | 15 | 20000 | 0.059354 | 3.04e-10 | 0.059232 | **−0.000122** | −0.205 |
| underconfident | 50 | 1000 | 0.059354 | 3.04e-10 | 0.085168 | +0.025814 | +43.492 |
| underconfident | 50 | 20000 | 0.059354 | 3.04e-10 | 0.060949 | +0.001595 | +2.688 |
| biased_high | 15 | 1000 | 0.117979 | 1.31e-15 | 0.121993 | +0.004013 | +3.402 |
| biased_high | 15 | 20000 | 0.117979 | 1.31e-15 | 0.117622 | **−0.000357** | −0.303 |
| biased_high | 50 | 1000 | 0.117979 | 1.31e-15 | 0.132344 | +0.014365 | +12.175 |
| biased_high | 50 | 20000 | 0.117979 | 1.31e-15 | 0.117705 | **−0.000274** | −0.232 |

At 50 bins and 1000 cases the reported ECE of the `underconfident` forecaster
is **43 % above** its true value. Four of the twelve cells have a **negative**
bias, which falsified the naive expectation; see section 9.1.

### 2.6 The parametric-bootstrap debiasing

`debiased_ece` estimates the bias by keeping the forecasts fixed and redrawing
`o_i ~ Bernoulli(f_i)`, under which the forecaster is calibrated by
construction, then subtracts the mean of the resulting ECE distribution. It
also returns a bootstrap p-value for the calibrated null.

20 estimates per configuration, 200 null replicates each, equal-mass bins:

| Spec | B | n | population | raw | debiased | `|raw−pop|` | `|deb−pop|` | helped | negatives |
|---|---|---|---|---|---|---|---|---|---|
| calibrated | 15 | 500 | 0 | 0.063861 | +0.002392 | 0.063861 | 0.002392 | yes | 9/20 |
| calibrated | 15 | 4000 | 0 | 0.021752 | +0.000139 | 0.021752 | 0.000139 | yes | 11/20 |
| calibrated | 50 | 4000 | 0 | 0.039434 | −0.000177 | 0.039434 | 0.000177 | yes | 12/20 |
| calibrated_uniform | 15 | 500 | 0 | 0.051860 | −0.002598 | 0.051860 | 0.002598 | yes | 12/20 |
| calibrated_uniform | 15 | 4000 | 0 | 0.019108 | −0.000148 | 0.019108 | 0.000148 | yes | 10/20 |
| calibrated_uniform | 50 | 4000 | 0 | 0.034006 | −0.001107 | 0.034006 | 0.001107 | yes | 12/20 |
| calibrated_rare | 15 | 500 | 0 | 0.036083 | −0.000388 | 0.036083 | 0.000388 | yes | 11/20 |
| calibrated_rare | 15 | 4000 | 0 | 0.013225 | +0.000479 | 0.013225 | 0.000479 | yes | 10/20 |
| calibrated_rare | 50 | 4000 | 0 | 0.023189 | −0.000153 | 0.023189 | 0.000153 | yes | 12/20 |
| overconfident | 15 | 500 | 0.075204 | 0.092724 | 0.039549 | 0.017519 | 0.035655 | **no** | 0/20 |
| overconfident | 15 | 4000 | 0.075204 | 0.075523 | 0.056744 | 0.000319 | 0.018460 | **no** | 0/20 |
| overconfident | 50 | 4000 | 0.075204 | 0.079969 | 0.045669 | 0.004765 | 0.029535 | **no** | 0/20 |
| underconfident | 15 | 500 | 0.059354 | 0.086998 | 0.021353 | 0.027644 | 0.038001 | **no** | 1/20 |
| underconfident | 15 | 4000 | 0.059354 | 0.064234 | 0.041121 | 0.004880 | 0.018233 | **no** | 0/20 |
| underconfident | 50 | 4000 | 0.059354 | 0.071410 | 0.028987 | 0.012056 | 0.030367 | **no** | 0/20 |
| biased_high | 15 | 500 | 0.117979 | 0.123277 | 0.063759 | 0.005298 | 0.054220 | **no** | 0/20 |
| biased_high | 15 | 4000 | 0.117979 | 0.118484 | 0.097440 | 0.000505 | 0.020539 | **no** | 0/20 |
| biased_high | 50 | 4000 | 0.117979 | 0.118816 | 0.080508 | 0.000836 | 0.037471 | **no** | 0/20 |

**9 of 18 configurations improved, and the split is exact: all 9 calibrated
configurations improved, by a median factor of 10.34×, and all 9
miscalibrated configurations got worse, by up to 57.9×.** Negative debiased
estimates occurred in **100 of 360** individual estimates, all of them on
calibrated specs. See section 9.2 for the structural reason and the usage
rule that follows from it.

## 3. Known answers, derived by hand first

`tests/test_known_answers.py` (38 tests) and section 1.4. The derivations are
in the test bodies.

| Quantity | Hand derivation | Value | Measured |
|---|---|---|---|
| `UNC`, Beta(2,2) | `obar = 2/4`, `UNC = 0.25` | 0.25 | 0.25 exactly |
| `RES`, Beta(2,2) | `ab/((a+b)²(a+b+1)) = 4/(16·5)` | 0.05 | 0.05, 1e-17 |
| `RES`, Beta(1,1) | uniform variance `1/12` | 0.0833333… | 1e-17 |
| `RES`, Beta(1,9) | `9/(100·11) = 9/1100` | 0.0081818… | 1e-17 |
| `BS`, calibrated Beta(2,2) | `UNC − RES = 0.25 − 0.05` | 0.2 | 1e-16 |
| `BS`, calibrated uniform | `0.25 − 1/12 = 1/6` | 0.1666667 | 1e-16 |
| `LS`, calibrated Beta(2,2) | density `6p(1−p)`; by `p ↔ 1−p` symmetry `LS = −12 ∫(p²−p³) ln p dp`; with `∫p^n ln p dp = −1/(n+1)²`, `= −12(−1/9 + 1/16) = −12(−7/144) = 7/12` | 0.5833333333333334 | **0.583333333333047**, difference 2.863e-13 |
| `LS`, calibrated uniform | `−2 ∫ p ln p dp = −2(−1/4) = 1/2` | 0.5 | **0.49999999999999983**, difference 1.665e-16 |
| Brier of a constant base-rate forecast | `p(1−p)` | — | 1e-15, 4 base rates |
| Brier, two-value hand case | `REL 0.0025 − RES 0.0625 + UNC 0.25` | 0.19 | 1e-15 |
| five-term terms, hand case | `WBV 0.09`, `WBC 0.15`, `BS 0.04` | — | 1e-15 |
| ECE, hand two-bin case | `0.5(0.3) + 0.5(0.2)` | 0.25 | 1e-15 |
| reliability curve, hand two-bin case | counts (2,2), mean f (0.2, 0.8), observed (0.5, 1.0) | — | 1e-15 |

Convergence of measurements to population values at n = 200 000 is asserted
with a tolerance of five standard errors of the Monte-Carlo estimate, stated
in the test as `5 × 0.5/sqrt(n) = 1.118e-3` for the Brier score, not chosen
after seeing the result.

## 4. Cross-implementation agreement with scikit-learn 1.9.1

`validate_sklearn_interop.py` → `outputs/validate_sklearn_interop_output.txt`.
**Failed checks: 0.**

| Comparison | Configurations | Worst difference | Tolerance |
|---|---|---|---|
| `brier_score` vs `sklearn.metrics.brier_score_loss` | 6 specs, n = 20 000 | **0.000e+00** | 1e-14 |
| `log_score` vs `sklearn.metrics.log_loss` | same | **0.000e+00** | 1e-12 |
| `reliability_curve(strategy="equal_width")` vs `calibration_curve(strategy="uniform")` | 6 specs × {10, 20} bins | bin counts identical in all 12; worst `|d observed|` and `|d mean forecast|` both **0.000e+00** | 1e-12 |
| `IsotonicCalibration` vs a direct `IsotonicRegression` fit | 1001-point grid | **0.000000e+00** | exact |
| `PlattScaling` vs `LogisticRegression(C=1e8, tol=1e-12)` on `logit(f)` | 4 specs × {4000, 40 000} | `|da| = 4.991e-09`, `|db| = 6.858e-09` | 1e-3 |

The equal-mass strategy is **not** claimed to match sklearn's `"quantile"`
strategy: sklearn drops duplicate quantile edges while this package keeps the
requested bin count and allows empty bins, so the occupied-bin counts differ
on tied forecasts. `tests/test_sklearn_interop.py` asserts the weaker true
statement rather than the convenient false one.

### 4.1 The scikit-learn 1.9 prefit removal

Captured verbatim in `outputs/validate_sklearn_interop_output.txt`:

```
  attempted: CalibratedClassifierCV(base, cv='prefit', method='sigmoid')
  raised   : InvalidParameterError
  message  : The 'cv' parameter of CalibratedClassifierCV must be an int in the range [2, inf), an object implementing 'split' and 'get_n_splits', an iterable or None. Got 'prefit' instead.
  replacement: CalibratedClassifierCV(FrozenEstimator(base), method='sigmoid')
  fitted, held-in Brier 0.092790644 against raw 0.092718669
```

This package never takes that path — it fits maps on forecast values and has
no estimator to freeze — but anyone recalibrating a prefit classifier on
scikit-learn 1.9 will, and an earlier session in this mission lost time to it.

## 5. Bootstrap reliability bands and their measured coverage

`validate_reliability_bands.py` →
`outputs/validate_reliability_bands_output.txt`. **Failed checks: 1, retained
— see section 9.3.**

Bands are pointwise percentile bootstrap intervals over the `(f_i, o_i)`
pairs. **Bin edges are fixed once from the original sample and reused for
every replicate**; recomputing quantile edges inside the loop would give each
replicate different bins and the percentile across replicates would then mix
bins rather than bound one. `tests/test_reliability.py` asserts the returned
edges equal a direct `bin_edges` call, which is the observable consequence.

Coverage target is the exact `E[o | f] = g⁻¹(f)` of the spec, evaluated at
each bin's mean forecast. 120 replicates, 250 bootstrap resamples, nominal
pointwise level 0.90:

| Spec | n | B | strategy | pointwise | simultaneous | gap |
|---|---|---|---|---|---|---|
| calibrated | 500 | 10 | equal_width | 0.8075 | 0.1250 | 0.6825 |
| calibrated | 500 | 10 | equal_mass | 0.8742 | 0.2417 | 0.6325 |
| calibrated | 2000 | 20 | equal_width | 0.8346 | 0.0167 | 0.8179 |
| calibrated | 2000 | 20 | equal_mass | 0.8942 | 0.1250 | 0.7692 |
| calibrated | 8000 | 10 | equal_width | 0.8875 | 0.2917 | 0.5958 |
| calibrated | 8000 | 10 | equal_mass | 0.8758 | 0.2250 | 0.6508 |
| calibrated_rare | 500 | 10 | equal_width | 0.8047 | 0.2583 | 0.5464 |
| calibrated_rare | 500 | 10 | equal_mass | **0.7842** | 0.0583 | 0.7258 |
| calibrated_rare | 2000 | 20 | equal_width | 0.7872 | 0.0500 | 0.7372 |
| calibrated_rare | 2000 | 20 | equal_mass | 0.8200 | **0.0000** | 0.8200 |
| calibrated_rare | 8000 | 10 | equal_width | 0.8002 | 0.2000 | 0.6002 |
| calibrated_rare | 8000 | 10 | equal_mass | 0.8875 | **0.3750** | 0.5125 |
| overconfident | 500 | 10 | equal_width | 0.8667 | 0.2083 | 0.6583 |
| overconfident | 500 | 10 | equal_mass | 0.8708 | 0.2417 | 0.6292 |
| overconfident | 2000 | 20 | equal_width | 0.8917 | 0.0833 | 0.8083 |
| overconfident | 2000 | 20 | equal_mass | 0.8954 | 0.1333 | 0.7621 |
| overconfident | 8000 | 10 | equal_width | 0.8700 | 0.2333 | 0.6367 |
| overconfident | 8000 | 10 | equal_mass | 0.8617 | 0.2083 | 0.6533 |

Over the 12 calibrated-spec configurations: pointwise coverage **0.7842 to
0.8942**, mean **0.8381**, **every one below the nominal 0.90**, mean
shortfall **0.0619**. Simultaneous coverage **0.0000 to 0.3750**, with the
pointwise-minus-simultaneous gap above **0.20** in every configuration tested
including the miscalibrated ones.

### 5.1 Per-bin coverage — the edge bins are much worse

Spec `calibrated`, n = 2000, 20 equal-width bins, 120 replicates. Every bin
was occupied in all 120 replicates, so no coverage figure is missing for lack
of samples. Coverage by bin index:

| bin | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| coverage | **0.3750** | 0.8583 | 0.8917 | 0.9083 | 0.8917 | 0.8833 | 0.8833 | 0.9167 | 0.8667 | 0.8417 |

| bin | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 |
|---|---|---|---|---|---|---|---|---|---|---|
| coverage | 0.8750 | 0.9000 | 0.9000 | 0.9250 | 0.8917 | 0.8917 | 0.8833 | 0.9250 | 0.8833 | **0.3000** |

The eighteen interior bins run from 0.8417 to 0.9250, which is the ordinary
bootstrap undercoverage of section 5. **The two edge bins are 0.3750 and
0.3000**, a third of the nominal level. A Beta(2,2) forecast puts least mass
against the boundaries of [0, 1], where a bootstrap distribution of a
proportion is most skewed and a percentile interval is at its worst. **Do not
read an edge bin's band as if it carried the stated level.** Nothing in the
plotted diagram distinguishes those two bins from the rest, which is why this
table is here.

### 5.2 A worked diagram

Spec `overconfident`, n = 4000, 12 equal-width bins, 1000 bootstrap resamples,
level 0.90 — the diagram reproduced in README.md as
`screenshots/reliability_diagram.png` (which uses the same settings):

| bin | n | mean f | observed | band lo | band hi | gap |
|---|---|---|---|---|---|---|
| 0 | 377 | 0.039726 | 0.153846 | 0.123664 | 0.187020 | −0.114120 |
| 1 | 386 | 0.125902 | 0.225389 | 0.190476 | 0.259390 | −0.099487 |
| 2 | 305 | 0.208412 | 0.331148 | 0.290201 | 0.375902 | −0.122736 |
| 3 | 322 | 0.290691 | 0.381988 | 0.338099 | 0.425888 | −0.091297 |
| 4 | 332 | 0.373959 | 0.424699 | 0.382332 | 0.469651 | −0.050740 |
| 5 | 315 | 0.459086 | 0.479365 | 0.431818 | 0.523707 | −0.020279 |
| 6 | 301 | 0.542627 | 0.538206 | 0.492000 | 0.586341 | +0.004421 |
| 7 | 294 | 0.626180 | 0.588435 | 0.542202 | 0.634237 | +0.037744 |
| 8 | 283 | 0.710307 | 0.618375 | 0.570320 | 0.663200 | +0.091932 |
| 9 | 314 | 0.792163 | 0.678344 | 0.631579 | 0.722231 | +0.113819 |
| 10 | 385 | 0.874442 | 0.750649 | 0.715815 | 0.786486 | +0.123792 |
| 11 | 386 | 0.959538 | 0.875648 | 0.847350 | 0.900764 | +0.083890 |

**9 of 12** bins have a band that excludes the mean forecast, i.e. are flagged
miscalibrated; **12 of 12** bands contain the exact `E[o|f]`. Bins 6 and 7
straddle the diagonal, which is correct: a sharpened-logit forecaster is well
calibrated near 0.5. A single ECE number averages that structure away.

## 6. Recalibration against the raw baseline

`validate_recalibration.py` → `outputs/validate_recalibration_output.txt`.
**Failed checks: 0.** Full tables in [../MODEL_CARD.md](../MODEL_CARD.md); the
summary is here.

| Finding | Measurement |
|---|---|
| Platt crossing point, `overconfident` | mean held-out Brier change turns negative from **n_total = 200** |
| Isotonic crossing point, `overconfident` | **n_total = 1000**, agreeing with Niculescu-Mizil and Caruana (2005) |
| Worst harm rate, Platt, `overconfident` | **0.512** at n_total = 60 (mean change **+0.005822**) |
| Worst harm rate, isotonic, `overconfident` | **0.713** at n_total = 60 (mean change **+0.016848**) |
| **Honest negative:** `calibrated` spec | both maps worse at **every** swept size, 60 to 15 000; isotonic harm rate **1.000** at n = 15 000 |
| Platt parameter recovery at n = 200 000 | worst `|da| = 0.02287`, `|db| = 0.00727` against the exact inverse, tolerance 0.03 |
| Brier and log scores disagree about isotonic | **2 of 5** sample sizes disagree in sign |
| Isotonic emits exact 0 or 1 | **24 of 200** held-out predictions after a 200-sample fit, **4** of them wrong; held-out log score **0.639511 → 1.272654** while Brier moved **0.221221 → 0.224179** |

## 7. Alternatives, verified rather than remembered

`validate_alternatives.py` → `outputs/validate_alternatives_output.txt`.
**Failed checks: 0.** The committed evidence is
`validation/alternatives_snapshot.json`, captured 2026-10-10 with:

```bash
curl -s -o /dev/null -w '%{http_code}' https://pypi.org/pypi/calibaudit/json
curl -s https://pypi.org/pypi/netcal/json
pip download --no-deps netcal==1.4.0 -d .
unzip -o netcal-1.4.0-py3-none-any.whl
```

and the same for `properscoring==0.1`. The validation script is offline and
deterministic: it reads the snapshot, so the release gate can re-run it and
get byte-identical output. Re-capturing the snapshot is a manual step, dated
in the file.

| Claim | Evidence | Result |
|---|---|---|
| PyPI name `calibaudit` is free | HTTP status of its PyPI JSON endpoint | **404** |
| `netcal` 1.4.0 exists | PyPI metadata, 15 releases, uploaded 2026-04-16, Apache-2.0 | confirmed |
| `netcal` computes ECE, ACE, MCE | `netcal/metrics/__init__.py` in the wheel exports ACE, ECE, ENCE, MCE, MMCE, NLL, PICP, PinballLoss, QCE, QuantileLoss, UCE | confirmed; **the README must not imply netcal lacks the metric** |
| `netcal` ships no Brier score or Murphy decomposition | case-insensitive search of every `.py` under `netcal/` for `brier`, `murphy`, `resolution` | **all absent** |
| `netcal` ships no bootstrap | same search for `bootstrap` | **absent**; its `ReliabilityDiagram` error bars come from Monte-Carlo-sampled input uncertainty |
| `netcal` requires PyTorch | `requires_dist`: numpy, scipy, matplotlib, scikit-learn, **torch>=2.3**, tqdm, **pyro-ppl**, **tensorboard**, **gpytorch** | confirmed; not installable in this container |
| `properscoring` 0.1 exists, one release, 2015-11-12 | PyPI metadata | confirmed; treat as unmaintained |
| `properscoring` has a Brier score but no calibration analysis | `__all__` = brier_score, crps_ensemble, crps_gaussian, crps_quadrature, threshold_brier_score; no ECE, no reliability or resolution term | confirmed |
| `properscoring`'s decomposition is the CRPS threshold decomposition, not Murphy's | `properscoring/thresholds.py` | confirmed; conflating them would be a wrong claim about a competitor |
| `sklearn.calibration` has the maps and the curve | `CalibratedClassifierCV`, `CalibrationDisplay`, `calibration_curve`, `FrozenEstimator` all present in 1.9.1 | 4 of 4 |
| `sklearn` has a Brier score but no ECE and no decomposition | `brier_score_loss`, `d2_brier_score`, `log_loss`; nothing named `calibration_error` | confirmed |

**`netcal` is a better choice than this package for anything it covers, and
the README says so before making any claim of its own.** None of the three is
a runtime dependency; `netcal` and `properscoring` are citations, `sklearn` is
a dependency used for isotonic regression.

## 8. Documented commands, re-run

`validate_cli.py` → `outputs/validate_cli_output.txt`. **Failed checks: 0.**

Three things are checked:

1. **Execution.** 17 CLI invocations run in clean subprocesses with their
   stdout, stderr and exit status recorded verbatim.
2. **The exit contract**, asserted on `subprocess.run().returncode`, never on
   a caught `SystemExit`: **7 of 7 correct**, covering exit 2 from
   `recalibrate --require-improvement` on an already-calibrated forecast,
   exit 0 from the same flag on `biased_high`, exit 1 from an invalid bin
   count and an invalid band level, and exit 2 from argparse for a bad choice,
   a bad subcommand and no subcommand.
3. **Coverage.** The script scans every fenced code block in README.md,
   MODEL_CARD.md, CHANGELOG.md and this file for shell commands, and asserts
   that each is either executed or has a recorded reason not to be:
   **28 distinct commands quoted, 0 uncovered.** A usage synopsis containing
   `[...]` placeholders is covered only when its subcommand is exercised by an
   executed invocation, so a new subcommand cannot be documented without also
   being run.

This check exists because a previous batch in this mission published a README
output block for a command that had not been run.

### 8.1 Example transcripts

`validate_examples.py` → `outputs/validate_examples_output.txt`. **Failed
checks: 0.** It runs all five example scripts in clean subprocesses, records
their stdout verbatim with the absolute PNG path reduced to a basename so the
committed file is reproducible on any machine, confirms all five exit 0,
confirms the five PNGs the README embeds exist, and asserts that **every
README line quoted from an example transcript appears character for character
in the captured stdout**: 6 quoted lines checked, 0 not found. Together with
section 8, every output block in README.md is now traceable to a committed
transcript.

## 9. The checks that currently FAIL, and what they mean

**Three checks fail, across two scripts.** Across the ten validation scripts
the totals are **65 checks, 62 passed, 3 FAILED**: two in
`validate_ece_bias.py` and one in `validate_reliability_bands.py`. All three
are retained verbatim. Each falsified an expectation written down *before* the
measurement, and each is paired with a refined check that states the correct
thing and passes. **None of the three makes a validation script exit
nonzero**, by the contract in section 0.

### 9.1 "ECE bias is always upward" — FALSIFIED

`validate_ece_bias.py`. **4 of 12** miscalibrated cells have a negative bias,
every one of them at n = 20 000; most negative **−0.000357**.

Two biases act in opposite directions. Bernoulli noise in `obar_k` biases the
estimate **up** by about `0.32 sqrt(B/n)`. Binning averages real
miscalibration inside a bin, which biases it **down** by an amount set by the
bin width and the curvature of the calibration map, and which **does not
shrink with n**. At n = 1000 the noise term dominates at every bin count
tested; at n = 20 000 with 15 bins the two are comparable and the net is
slightly negative.

Refined check, which passes: bias is positive at every n = 1000 cell and at
least one n = 20 000 cell is negative. **"ECE is biased upward" is the
small-sample statement, not a theorem**, and the crossing point depends on the
bin count.

### 9.2 "Debiasing reduces the bias in most configurations" — FALSIFIED

`validate_ece_bias.py`. **9 of 18** improved, and the split is exact: the 9
wins are the 9 calibrated configurations and the 9 losses are the 9
miscalibrated ones, the worst **57.9×** the raw absolute error.

The null is built by redrawing outcomes from the forecasts themselves, so it
describes a *calibrated* forecaster with the same forecast distribution. On a
forecaster that really is calibrated that is the right null. On a
miscalibrated one it is the wrong null, and subtracting its mean removes most
of the real miscalibration along with the bias.

Refined check, which passes: debiasing helps on **9 of 9** calibrated
configurations and hurts on **9 of 9** miscalibrated ones.

**Usage rule that follows, stated in README.md, MODEL_CARD.md and the module
docstring: read `DebiasedECE.p_value` first.** Large means the calibrated null
is tenable and `debiased` is the number to quote. Small means it is not, and
`raw` with the interval `[null_q05, null_q95]` is what to report — the gap
between the raw estimate and that interval is the part of the ECE that binning
bias cannot explain.

A related recorded finding, which passes as a check because the expectation
was documented as "the count is the evidence": the debiased estimate went
**negative in 100 of 360** individual estimates. An ECE cannot be negative, so
that is the correction overshooting. It is **not clipped**, because clipping
would hide exactly the case that tells you it overshot.

### 9.3 "Pointwise band coverage is within 0.10 of 0.90" — FALSIFIED

`validate_reliability_bands.py`. Range **0.7842 to 0.8942** over 12
configurations, mean **0.8381**; the floor is below the documented 0.80.

Refined check, which passes: coverage is **below** nominal in all 12
configurations and never above, with a mean shortfall of **0.0619** against a
documented expectation of 0.02 to 0.12. A percentile bootstrap of a binomial
proportion is discrete and skewed near the edges of [0, 1] and has no
finite-sample coverage guarantee; this is the size of the shortfall. The
roadmap entry for consistency bands (Bröcker and Smith 2007) exists because of
this measurement.

### 9.4 Checks that pass but read as negatives

These are recorded as passing because the expectation was stated correctly in
advance, but they are results against the package rather than for it:

- **Recalibration never helps an already-calibrated forecast**, at any swept
  size up to 15 000, with isotonic worse in **100 %** of 80 replicates there
  (section 6).
- **Both learned maps harm the score on small samples**: isotonic in **71.3 %**
  of replicates at 60 total samples (section 6).
- **Simultaneous band coverage is 0.0000 to 0.3750** against a nominal
  pointwise 0.90 (section 5).
- **The worst single bin's coverage is 0.3000** (section 5.1).
- **A three-term binned decomposition drops up to 6.6367 % of the Brier
  score** with a non-monotone, sign-changing error (section 1.3).
- **The binned reliability term is 1.6e-05 to 4.7e-05 on forecasters whose
  population reliability is exactly 0**, at n = 200 000 (section 1.4).

## 10. References

Implemented or cited in the code, with the claim each supports. Verified from
publisher-hosted pages or DOI landing pages; `api.crossref.org` is blocked by
this container's egress proxy.

- Brier, G. W. (1950). "Verification of forecasts expressed in terms of
  probability." *Monthly Weather Review* 78(1), 1–3.
  doi:10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2 — the score.
- Good, I. J. (1952). "Rational decision." *Journal of the Royal Statistical
  Society Series B* 14(1), 107–114 — the logarithmic score.
- Murphy, A. H. (1973). "A new vector partition of the probability score."
  *Journal of Applied Meteorology* 12(4), 595–600.
  doi:10.1175/1520-0450(1973)012<0595:ANVPOT>2.0.CO;2 — the three-term
  decomposition.
- Murphy, A. H. (1986). "A new decomposition of the Brier score: formulation
  and interpretation." *Monthly Weather Review* 114(12), 2671–2673.
  doi:10.1175/1520-0493(1986)114<2671:ANDOTB>2.0.CO;2
- Murphy, A. H. and Winkler, R. L. (1977). "Reliability of subjective
  probability forecasts of precipitation and temperature." *JRSS Series C*
  26(1), 41–47. doi:10.2307/2346866 — the reliability diagram.
- Hersbach, H. (2000). "Decomposition of the continuous ranked probability
  score for ensemble prediction systems." *Weather and Forecasting* 15(5),
  559–570. doi:10.1175/1520-0434(2000)015<0559:DOTCRP>2.0.CO;2 — the same
  partition in the continuous case.
- Gneiting, T. and Raftery, A. E. (2007). "Strictly proper scoring rules,
  prediction, and estimation." *JASA* 102(477), 359–378.
  doi:10.1198/016214506000001437 — propriety of both scores used here.
- Platt, J. C. (1999). "Probabilistic outputs for support vector machines and
  comparisons to regularized likelihood methods." In *Advances in Large Margin
  Classifiers*, MIT Press, 61–74 — `PlattScaling`, including the smoothed
  targets.
- Zadrozny, B. and Elkan, C. (2002). "Transforming classifier scores into
  accurate multiclass probability estimates." *KDD 2002*, 694–699.
  doi:10.1145/775047.775151 — isotonic recalibration.
- Niculescu-Mizil, A. and Caruana, R. (2005). "Predicting good probabilities
  with supervised learning." *ICML 2005*, 625–632. doi:10.1145/1102351.1102430
  — the sample size at which isotonic overtakes Platt, which section 6
  reproduces.
- Naeini, M. P., Cooper, G. F. and Hauskrecht, M. (2015). "Obtaining well
  calibrated probabilities using Bayesian binning." *AAAI 2015*, 2901–2907;
  PMCID PMC4410090 — the equal-width ECE.
- Guo, C., Pleiss, G., Sun, Y. and Weinberger, K. Q. (2017). "On calibration of
  modern neural networks." *ICML 2017*, PMLR 70, 1321–1330. arXiv:1706.04599 —
  temperature scaling, and the overconfidence shape the `overconfident` spec
  imitates.
- Nixon, J., Dusenberry, M. W., Zhang, L., Jerfel, G. and Tran, D. (2019).
  "Measuring calibration in deep learning." *CVPR 2019 Workshops*.
  arXiv:1904.01685 — adaptive (equal-mass) binning.
- Kumar, A., Liang, P. and Ma, T. (2019). "Verified uncertainty calibration."
  *NeurIPS 2019*, 3787–3798. arXiv:1909.10155 — states and analyses the upward
  bias of the plug-in ECE estimator; its debiased squared-ECE estimator is the
  roadmap item this package's parametric bootstrap does not replace.
- Bröcker, J. and Smith, L. A. (2007). "Increasing the reliability of
  reliability diagrams." *Weather and Forecasting* 22(3), 651–661.
  doi:10.1175/WAF993.1 — consistency bands under the calibrated null, the
  roadmap response to section 9.3.
- Efron, B. and Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*.
  Chapman and Hall, chapter 13 — the percentile interval.
- Schneider, R. (1993). *Convex Bodies: The Brunn–Minkowski Theory*. Cambridge
  University Press — support-function identities, cited in the docstrings for
  context only; this package contains no polytope algebra.
- Rockafellar, R. T. (1970). *Convex Analysis*. Princeton University Press —
  as above.

The theory is due to these authors. This package contributes the measurement
of the estimators' error, not the estimators.

## 11. Reproducing every number

From the repository root, after `pip install -e ".[test]"`:

```bash
python -m pytest tests/ -q --junitxml=junit.xml
ruff check src/ tests/ examples/ validation/
python validation/validate_environment.py
python validation/validate_alternatives.py
python validation/validate_decomposition_identity.py
python validation/validate_ece_bias.py
python validation/validate_reliability_bands.py
python validation/validate_recalibration.py
python validation/validate_sklearn_interop.py
python validation/worked_example.py
python validation/validate_cli.py
python validation/validate_examples.py
MPLBACKEND=Agg python examples/reliability_diagram.py
MPLBACKEND=Agg python examples/ece_bias_curve.py
MPLBACKEND=Agg python examples/decomposition_bars.py
MPLBACKEND=Agg python examples/recalibration_audit.py
MPLBACKEND=Agg python examples/binning_strategy.py
```

Measured wall-clock, single runs on two contended cores:

| Script | Time |
|---|---|
| `validate_environment.py` | under 2 s |
| `validate_alternatives.py` | under 2 s |
| `validate_decomposition_identity.py` | about 4 s |
| `validate_ece_bias.py` | about 22 s |
| `validate_reliability_bands.py` | about 81 s |
| `validate_recalibration.py` | about 76 s |
| `validate_sklearn_interop.py` | about 6 s |
| `worked_example.py` | about 3 s |
| `validate_cli.py` | about 48 s |
| `validate_examples.py` (runs all five examples) | about 43 s |
| `examples/*.py`, all five, run directly | about 40 s |
| `pytest tests/ -q` | 146 s in the committed run; 117 s, 136 s and 146 s in three timed runs |

Every script is deterministic in its recorded seeds, and **re-running all ten
produces byte-identical files in `outputs/`.** That was checked rather than
assumed: the whole set was run twice and the md5 sums compared, including the
three slowest scripts, with no differences. No validation script prints a
timestamp or a duration into its committed output, which is what makes that
possible.

**The one exception is `outputs/pytest_output.txt`**, which is not produced by
a validation script: it is the stdout of `python -m pytest tests/ -q`, whose
last line carries the wall-clock duration. Three timed runs on this container
gave 117 s, 136 s and 146 s, so that one line changes on every run while the
`405 passed` count does not. The committed file is the 146 s run and
README.md quotes it verbatim. The release gate re-executes `validation/*.py`
and `examples/*.py`, neither of which writes this file, so it does not drift
under the gate.

## Credits

This is under reserved rights obtained by OPTIMA Organisation.
