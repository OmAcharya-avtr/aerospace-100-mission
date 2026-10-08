# Validation evidence — twininvalidate 0.1.0

Validation level 3 (research grade). Every number below was produced by
running the scripts in this directory in the build session of 2026-10-08, in
the build container (Python 3.13.16, NumPy 2.5.3, SciPy 1.18.1,
scikit-learn 1.9.1, 2 cores, 7.8 GiB). The raw output of each script is saved
beside it as `<script>_output.txt` and is the authority for anything quoted
here or in `README.md`.

Reproduce everything:

```bash
python validation/validate_twin.py
python validation/validate_detectors.py
python validation/validate_thresholds.py
python validation/validate_arl_curve.py
python validation/validate_classifier.py
python validation/validate_ambiguity.py
python validation/validate_cli.py
python validation/worked_example.py
```

Total wall clock for the eight scripts: about 155 s on 2 cores, dominated by
`validate_classifier.py` at 65 s. No single run exceeds 3 minutes.

---

## 1. Summary of checks

| # | Check | Reference | Result | Tolerance | Verdict |
|---|---|---|---|---|---|
| 1 | Steady-state Riccati residual | Anderson & Moore 1979, eq. (5) | 7.047e-19 | < 1e-15 | PASS |
| 2 | Prediction-error spectral radius `A - KC` | stability requirement | 0.917218411 | < 1 | PASS |
| 3 | In-control residual mean | Kailath 1968 innovations property | +0.000814 (+1.03 se) | 4 se = 0.00316 | PASS |
| 4 | In-control residual sd | Kailath 1968 | 1.000302 (+0.38 se) | 4 se | PASS |
| 5 | In-control lag-1 autocorrelation | Kailath 1968 | −0.000141 (−0.18 se) | 4 se | PASS |
| 6 | In-control lag-2, lag-5 autocorrelation | Kailath 1968 | +0.001384, −0.000462 | 4 se | PASS |
| 7 | Parameter-step mean shift vs closed form, eq. (10) | derived in `twin.py` | 0.16 % error at δ = −0.01 | 3 % | PASS |
| 8 | Detector known answers (12 cases) | Page 1954, Roberts 1959, Willsky & Jones 1976 | max error 0.000e+00 | 1e-12 | PASS 12/12 |
| 9 | GLR(window=1) closed-form threshold, eq. (7) | hand calculation | 5.4137830853 | 1e-8 | PASS |
| 10 | Monte-Carlo calibrator vs closed form | internal consistency | 0.50 %, 1.79 %, 0.97 % at T = 200/1000/5000 | 5 % | PASS |
| 11 | ARL0 transfer to fresh banks, three baselines | — | +3.0 %, +3.4 %, −1.1 % (T=200); −0.2 %, +2.8 %, +3.4 % (T=1000); −1.4 %, −8.7 %, −1.6 % (T=5000) | 8.8 % (3 se) | PASS 9/9 |
| 12 | ARL0 transfer, variance-CUSUM oracle | — | **−8.6 % (T=200), +13.5 % (T=1000)**, +9.1 % (T=5000) | 8.8 % | **FAIL 2/3** |
| 13 | Residual identity, asset fault vs twin error | derivation in `ambiguity.py` | 6.17e-14 (≈ 278 ulps) | 1e-10 | PASS |
| 14 | Identical alarm indices in both worlds, 4 detectors | — | identical in every run | exact | PASS 4/4 |
| 15 | CLI exit statuses (15 cases) | — | all as specified | exact | PASS 15/15 |

Test suite: `python -m pytest tests/ -q` → **243 passed, 0 failed, 0 skipped**,
19.7 s. `ruff check src/ tests/ examples/ validation/` → clean.

---

## 2. The headline deliverable: detection delay against false-alarm rate

Source: `validate_arl_curve.py` → `validate_arl_curve_output.txt`.
Figure: `screenshots/arl_curve.png` from `examples/arl_curve.py`.

### Threshold-setting method, declared

A threshold is chosen so that the **in-control** average run length hits a
declared target, using in-control residual streams only. No out-of-control
stream and no detection delay is visible to the procedure. Two cases have an
exact closed form and are used as known answers:

- GLR with window 1 reduces to `|z| > sqrt(2h)`, a per-sample test of a single
  standard normal, so the run length is Geometric(p) with
  `p = 2(1 − Φ(sqrt(2h)))` and `h*(T) = Φ⁻¹(1 − 1/(2T))² / 2`.
- EWMA with λ = 1 reduces to `|z| > L`, so `L*(T) = Φ⁻¹(1 − 1/(2T))`.

Everything else is set by bisection on the censored maximum-likelihood ARL0
estimator, `ARL0 = (total samples at risk) / (number of alarms)`, over a fixed
bank of in-control streams. Right-censored runs contribute their full horizon
to the numerator and nothing to the denominator; discarding them would bias
every average run length downwards.

No published ARL approximation is quoted for CUSUM, for EWMA with λ < 1 or for
GLR with window > 1, because none was verified in this environment.

### Operating point at the declared target ARL0 = 1000 samples

1000 samples at 20 Hz is 50 s between false alarms, i.e. 72 000 false alarms
per 1000 operating hours. Zero-state convention, 300 runs × 3000 samples per
cell, detection fraction 1.00 everywhere except the variance-CUSUM oracle on
the noise-variance change (0.99).

| Change type | CUSUM (k=0.25) | EWMA (λ=0.1) | GLR (window=100) | variance-CUSUM oracle |
|---|---|---|---|---|
| parameter step, gain −1 % | **62.3 ± 2.0** | 72.6 ± 2.9 | 75.7 ± 2.5 | 216.4 ± 8.0 |
| slow ramp, gain −1 % over 400 samples | **232.5 ± 5.4** | 258.7 ± 6.3 | 258.7 ± 6.5 | 407.9 ± 11.0 |
| noise variance, process noise × 2 | **211.0 ± 11.5** | 223.5 ± 12.8 | 262.3 ± 14.1 | 466.6 ± 22.5 |

Delays in samples; one sample is 50 ms. Errors are one standard error of the
mean delay over detected runs.

### The change type on which every method does badly

**The noise-variance change.** Both it and the parameter step are fully
present from sample 0, yet every baseline takes 3.1 to 3.5 times as long to
see it:

| Detector | step | ramp | variance | variance ÷ step |
|---|---|---|---|---|
| CUSUM | 62.3 | 232.5 | 211.0 | 3.39 |
| EWMA | 72.6 | 258.7 | 223.5 | 3.08 |
| GLR | 75.7 | 258.7 | 262.3 | 3.47 |
| variance-CUSUM oracle | 216.4 | 407.9 | 466.6 | 2.16 |

The reason is exact rather than statistical: a process-noise change leaves
`E[z] = 0`, so the quantity all three baselines accumulate has **no
post-change drift at all** and they fire only on the increased rate of tail
excursions.

The slow ramp's delay is a different kind of number and is not evidence of the
same failure: the CUSUM alarms after 232 samples of a 400-sample ramp, i.e.
when the change has reached 58 % of its final magnitude, which is close to the
best a causal detector can do against a change that is not yet there.

---

## 3. Negative results

These are the findings this repository exists to publish. Each is measured,
none is an expectation.

### 3.1 CUSUM beats the windowed GLR on every change type

The batch specification expected the GLR to be hard to beat. It is the
**slowest** of the three declared baselines on the parameter step and the slow
ramp, and the slowest on the noise-variance change too:

```
parameter_step   cusum (62) < ewma (73) < glr (76)
slow_ramp        cusum (232) < ewma (259) < glr (259)
noise_variance   cusum (211) < ewma (224) < glr (262)
```

Structural reason: a CUSUM accumulates without bound, so a persistent shift
drives it linearly for as long as the shift lasts; a windowed GLR truncates
accumulation at its 100-sample window and pays a maximum-over-onset penalty
that raises the threshold needed for the same ARL0. The GLR's real advantage —
that it needs no declared shift magnitude — is invisible in a comparison where
the CUSUM's declared reference value happens to suit the change. On a change
far from the CUSUM's design shift the ordering could reverse; that was not
measured and is not claimed.

### 3.2 Using the correct statistic makes the variance change *worse*

The variance-CUSUM, handed the post-change residual variance in advance and
accumulating `z² − 1` — the sufficient statistic for a scale change in a
zero-mean Gaussian — is the **slowest** method on the noise-variance change:
466.6 samples against 211.0 for the mis-specified mean CUSUM.

At this effect size its per-sample standardised shift is only
`0.059 / sqrt(2) = 0.042`, because `Var(z²) = 2` under H0, while the mean
CUSUM's alarm rate responds to residual scale through an exponentially
sensitive tail. Correct specification is not the same as more information per
sample. This is a measured counterexample to the usual advice and the kind of
result that only appears if the correctly-specified test is actually run.

### 3.3 The analytic CUSUM beats the learned classifier, and the one win it keeps is a specification advantage

Source: `validate_classifier.py` → `validate_classifier_output.txt`.
Figure: `screenshots/classifier_benchmark.png`.

Protocol: thresholds set from the same declared in-control ARL0 target of 1000
samples on the same in-control bank, in-control data only; steady-state
convention with the change at sample 50 so the windowed classifier has a full
feature window; 200 runs × 2500 samples; four disjoint seed families for
training, probability calibration, threshold calibration and evaluation;
splits by run, never by window.

The classifier's confidence is quantised by the isotonic calibration (1857
distinct levels on the bank), so it **cannot** be set to ARL0 1000. Both
achievable bracketing thresholds are therefore reported: `learned-cons` at
ARL0 1516 (a *longer* ARL0 than the baselines, so fewer false alarms allowed —
a handicap) and `learned-gen` at ARL0 874 (more false alarms allowed — an
advantage). Cells are mean delay in samples / fraction of runs detected.

| Scenario | CUSUM | EWMA | GLR | varcusum | learned-cons (ARL0 1516) | learned-gen (ARL0 874) |
|---|---|---|---|---|---|---|
| parameter step, gain −1 % | **52** / 1.00 | 64 / 1.00 | 65 / 1.00 | 216 / 1.00 | 53 / 1.00 (**+2 %**) | 47 / 1.00 (−9 %) |
| slow ramp | 229 / 1.00 | 238 / 1.00 | 251 / 1.00 | 373 / 1.00 | **213** / 1.00 (−7 %) | 183 / 1.00 (−20 %) |
| noise variance × 2 | **200** / 1.00 | 225 / 1.00 | 277 / 1.00 | 462 / 0.99 | 445 / 0.99 (**+122 %**) | 338 / 1.00 (+69 %) |
| OOD: gain **+1 %**, sign flipped | 54 / 1.00 | 65 / 1.00 | 67 / 1.00 | 209 / 1.00 | 8 / **0.02** | 75 / **0.04** |
| OOD: gain −0.4 %, smaller | 225 / 1.00 | 277 / 1.00 | 312 / 1.00 | 555 / 0.98 | **174** / 1.00 | 136 / 1.00 |
| OOD: gain −3 %, larger | 20 / 1.00 | **19** / 1.00 | 20 / 1.00 | 32 / 1.00 | 26 / 1.00 | 23 / 1.00 |
| OOD: process noise × 5 | **71** / 1.00 | 71 / 1.00 | 76 / 1.00 | 145 / 1.00 | 184 / 1.00 (+159 %) | 169 / 1.00 |

**Scoreboard at a comparable false-alarm budget: the CUSUM beats the learned
classifier on 2 of 3 in-distribution scenarios and on 3 of 4
out-of-distribution scenarios.** The single in-distribution loss for the CUSUM
is the slow ramp, by 7 %.

**The decisive number is det 0.02.** On a gain change of the same magnitude
and the opposite sign, the classifier detects 2 % of runs within 2450 samples
while every baseline detects 100 % with an unchanged delay. The training set
contains only negative gain changes, so the classifier has learned an
effectively one-sided test; the two-sided CUSUM spends half its false-alarm
budget on the other direction and is tuned to a declared 0.5σ shift rather
than the true 0.404σ, which is exactly what the classifier's remaining 7 %
ramp win is made of.

Calibration quality of the confidence output on 90 120 held-out windows:
Brier score **0.1589**, expected calibration error **0.0208** over ten
equal-width bins. The Brier score cannot approach 0 here and the reason is in
the labels rather than the model — see DATASET_CARD.md.

The classifier was **not retuned** after these results. The forest size, depth,
seed, feature set, window length and training magnitudes were all declared
before the benchmark was run, and the thresholds were set on in-control data
only. The only change made after a benchmark had been run was the seed
reallocation of error 4 below, which was a correctness fix to the split
strategy rather than a tuning change, and it moved the result *against* the
learned model.

### 3.4 The learned monitor's threshold cannot be set to an arbitrary false-alarm rate

The isotonic calibration makes the confidence take finitely many values — 1857
distinct levels on the calibration bank — so the achievable in-control ARL0
jumps. The two levels bracketing the target of 1000 samples give ARL0 1516.2
and 874.4, a factor of 1.73 apart: the classifier simply cannot be operated at
the baselines' false-alarm rate. Both are reported above rather than picking
whichever flatters the model. The recursive analytic detectors have
continuously adjustable thresholds and no such constraint, which is a real
operational advantage that no delay number captures.

### 3.5 Per-decision cost

| | per decision | ratio | fraction of a 50 ms sample period |
|---|---|---|---|
| learned classifier, one window | 10.469 ms | 1712 × | 20.9 % |
| CUSUM, per sample | 0.00612 ms | 1 × | 0.012 % |

These are the figures in `validate_classifier_output.txt`. Wall-clock on this
container moves 10–20 % between runs: across the build session the learned
classifier was observed between 10.5 and 11.4 ms per decision and the CUSUM
between 0.0039 and 0.0061 ms per sample, so **the ratio is of order 2000, not
a precise number**. None of this is a hardware characteristic: 2 cores,
7.8 GiB, Python 3.13.16. The forest's
`n_jobs` is forced to 1 after fitting, because on this container `n_jobs > 1`
is several times slower at single-row inference.

### 3.6 A validation check that FAILS

Check 12 in the summary table. The threshold calibrated for the variance-CUSUM
oracle on one bank transfers to three pooled fresh banks with a bias of
**−8.6 % at a target of 200 samples and +13.5 % at 1000**, both outside the
declared tolerance of three standard errors (8.8 %). The three declared
baselines transfer within +3.4 % / −8.7 % and pass all nine of their checks.

Diagnosis: the variance CUSUM's increments `z² − 1` are right-skewed with a
heavy upper tail, so its own upper tail is built from fewer, larger excursions
and the bisected threshold lands on a bank-specific extreme; the mean CUSUM's
increments `z − k` are symmetric and light-tailed, so the same bank resolves
its tail better.

Consequence: the variance CUSUM is **not** one of the three declared baselines
and no headline curve depends on it. Its reported delay at a nominal ARL0 of
1000 is in fact measured at an effective ARL0 of about 1135, which makes it
look slightly *better* than a fairly matched comparison would, and it loses
anyway. The tolerance was not widened to absorb the failure.

---

## 4. Twin invalidation against asset fault

Source: `validate_ambiguity.py` → `validate_ambiguity_output.txt`.
Figure: `screenshots/invalidation_vs_fault.png`.

Two physically different worlds, one shared noise realisation:

- **World A, asset fault.** The sensor develops a 0.02 rad bias at sample 400;
  the twin's declared offset stays 0. Residual
  `e = C(x − x̂) + b + v`.
- **World B, twin invalidation.** The sensor is fine and the asset is
  unchanged; the twin's declared offset is revised to −0.02 rad at sample 400.
  Residual `e = C(x − x̂) + b + v`.

The two are the same function of the same quantities, and the state recursions
are identical, so the residual streams are **algebraically identical**.

| Quantity | Value |
|---|---|
| max \|world A − world B\| over 40 runs × 1200 samples | 6.17e-14 |
| typical \|residual\| | 0.964 |
| difference in units in the last place of a double | about 278 |
| CUSUM / EWMA / GLR / variance-CUSUM alarm indices | identical in every run |

This package does **not** claim the streams are bitwise identical: World A adds
`b` to the measurement and World B subtracts it from the prediction, so the
two sums round differently. 6.17e-14 is the measured number; the identity is
established by the derivation, not by the arithmetic.

**Conclusion.** An alarm means the twin and the asset no longer agree. It does
not mean the asset is faulty and it does not mean the twin is wrong. Deciding
which needs information this monitor is not given.

---

## 5. Errors made during the build, and their corrections

Recorded here rather than silently fixed.

### Error 1 — the predictor gain was missing its `A` factor

The first implementation set the steady-state gain to `K = P Cᵀ / S`, the
*filter* gain that corrects `x̂_{k|k}`, and used it in the one-step predictor
recursion `x̂_{k+1} = A x̂_k + B u_k + K e_k`, which needs the *predictor* gain
`K = A P Cᵀ / S`.

**How it was caught:** the in-control whiteness check in
`validate_twin.py`. The measured lag-1 residual autocorrelation was **+0.0105
against a standard error of 0.00079, i.e. +13 standard errors**, where the
innovations property requires 0. The mean, the variance and the tail
probability were all within tolerance, so only the autocorrelation check found
it.

**Effect if it had shipped:** a sub-optimal filter whose innovations are
correlated. Every threshold in the package assumes a white unit-variance
residual, so every ARL0 would have been wrong in a way that no single-sample
statistic would have revealed.

**After the fix:** lag-1 = −0.000141 (−0.18 se). The locked gain is
`K = [0.151684, 0.182596]`, with spectral radius 0.917218 for `A − KC`.
`tests/test_twin.py::test_predictor_gain_includes_the_A_factor` asserts the
distinction explicitly, and `tests/test_asset.py::test_in_control_residual_is_white`
is the regression guard.

### Error 2 — the variance-CUSUM reference value was derived from a wrong premise

The variance-CUSUM oracle's reference value was first set to 0.5, on the
premise that a declared scenario multiplying the asset's process noise by 2
doubles the residual variance.

It does not. The measurement noise `R = 1e-4 rad²` dominates the innovation
variance `S = 1.165e-4 rad²`, so the process-noise path contributes only about
14 % of `S` and a doubling raises the **residual** variance to 1.059, not 2.0.

**How it was caught:** the "oracle" was slower than the mis-specified
mean-shift baselines it was meant to bound — 506 samples against 213 — which
is not a result a correctly-specified test should give.

**Correction:** the reference value was recomputed by the same Page (1954)
rule from the measured residual-variance shift, 0.062/2 = 0.031, using a
measurement made while the scenarios were being declared and **before any
detection delay had been computed**.

**Residual discrepancy, not corrected:** the longer measurement now printed by
`validate_twin.py` gives a post-change residual variance of 1.0585, i.e. a
reference value of 0.029 rather than 0.031, a 7 % difference. The declared
0.031 was **not** revised to match, because revising a declared constant after
a delay is known is precisely what this package's threshold discipline
forbids. The discrepancy is recorded in the `DEFAULT_VARCUSUM_REFERENCE`
docstring.

**Note on the conclusion:** the correction did not rescue the oracle. At the
corrected reference value it is still the slowest method on the
noise-variance change (466.6 against 211.0), which is finding 3.2 — a real
result rather than the artefact the original 0.5 produced.

### Error 4 — the seed families were not disjoint as claimed

`datasets.py` originally allocated 53002 and 53003 to the classifier's
training streams and 53004 and 53005 to the probability-calibration streams.
Because `build_labelled_set` consumes `seed_changed + i` for each of the three
scenarios, the training family actually used 53003, 53004 and 53005, so its
second and third scenarios carried the same seed *numbers* as the calibration
family.

**How it was caught:** while writing the test-split section of MODEL_CARD.md,
by reading the seed table against the code rather than by a test.

**Whether it actually leaked:** no. The two families are generated at
different `n_runs`, and `numpy.random.default_rng(seed).standard_normal(n)`
draws different values for different `n`, so no array was shared. But the
allocation did not *support* the disjointness claim that the by-run split
strategy rests on, and a claim that happens to be true by accident is not
evidence.

**Correction:** the families were spaced — training changed streams at
53010–53012, probability calibration at 53020 and 53030–53032 — and the seed
table in the module docstring now records the reason.

**Effect on the results:** the classifier was retrained on the new seeds and
the benchmark re-run. The result moved **against** the learned model: its
parameter-step delay went from 50 to 53 samples against the CUSUM's 52, i.e.
from a 4 % win to a 2 % loss, and its noise-variance delay from 369 to 445,
i.e. from +84 % to +122 %. The earlier, more favourable numbers were not kept.

### Error 5 — a Hypothesis property was stated wrongly, not the code

`test_cusum_arms_never_exceed_the_running_absolute_sum` bounded the CUSUM by
the largest absolute *prefix* sum. Hypothesis found `z = [1, −1, −1]`, where
the lower arm accumulates from index 1 and reaches 1.5 against a prefix bound
of 1.0. The CUSUM maximises over onsets, so the correct bound is the largest
absolute **sub-interval** sum, `max(C) − min(C)` over the cumulative sum
prefixed by 0. The property was restated; no code changed.

A second Hypothesis finding, `a = 1e-12` in the ZOH scalar property, showed
that the *test's* closed form `(exp(a dt) − 1)/a` cancels catastrophically as
`a → 0`. The test now uses `dt · expm1(a dt)/(a dt)`. Again the code was
right and the check was wrong.

---

## 6. Compute budget

Measured in the build session on 2 cores, 7.8 GiB.

| Step | Wall clock |
|---|---|
| Simulate 100 × 2000 in-control residual samples | 0.06 s |
| GLR statistic (window 100) on 100 × 2000 | 0.07 s |
| Fit the drift classifier (13 440 training windows, 200 trees) | 4.8 s |
| `validate_arl_curve.py` (headline deliverable, 12 thresholds × 4 detectors × 3 scenarios) | 15 s |
| `validate_classifier.py` (training, calibration, 7 scenarios, latency) | 65 s |
| `python -m twininvalidate benchmark` | 52 s |
| `python -m pytest tests/ -q` (243 tests) | 20 s |
| All eight validation scripts | about 155 s |

No single run exceeds 3 minutes. The architectural reason the budget fits: a
statistic path does not depend on the threshold, so one simulation produces a
whole delay-against-false-alarm curve by sweeping thresholds over stored
matrices.

Wall-clock figures move 10–20 % between runs on this container. They are
reported as measurements of this container, never as hardware characteristics,
and the primary metrics everywhere in this repository are seeded sample counts
rather than times.

---

## 7. What was NOT validated

- **No real asset, no flight data.** Every number is from simulations of the
  declared linear-Gaussian twin.
- **No operational false-alarm rate.** The headline target ARL0 of 1000
  samples is 72 000 false alarms per 1000 h, which is far too many for a real
  monitor. Resolving the ARL0 a real requirement implies (7.2e6 samples for 10
  per 1000 h) needs about 7e9 in-control samples for 1000 alarms, which does
  not fit the compute budget. The **shape** of the curve is the deliverable;
  the operating point a real requirement implies is off its right-hand end and
  is not measured.
- **No non-Gaussian, non-white or non-stationary residual.** Every threshold
  rests on `z ~ i.i.d. N(0,1)` in control, verified here only for the shipped
  twin.
- **No multi-output twin.** The package accepts scalar measurements only.
- **No steady-state ARL1 for the headline curve.** The headline curve uses the
  zero-state convention; only the learned-model comparison uses the
  steady-state protocol, and the two are not interchangeable.
- **No published ARL approximation** for CUSUM or for EWMA with λ < 1 was
  used or checked, because none was verified in this environment.

---

## 8. References used, all verified 2026-10-08

- Page, E. S. (1954). "Continuous Inspection Schemes." *Biometrika* 41(1/2),
  100–115. DOI 10.1093/biomet/41.1-2.100. — the CUSUM recursion and the
  reference-value rule `k = δ/2`.
- Roberts, S. W. (1959). "Control Chart Tests Based on Geometric Moving
  Averages." *Technometrics* 1, 239–250. DOI 10.1080/00401706.1959.10489860.
  — the EWMA recursion.
- Lucas, J. M. & Saccucci, M. S. (1990). "Exponentially Weighted Moving
  Average Control Schemes: Properties and Enhancements." *Technometrics* 32,
  1–12. DOI 10.1080/00401706.1990.10484583. — the 0.05–0.25 range the declared
  λ = 0.1 comes from.
- Willsky, A. S. & Jones, H. L. (1976). "A Generalized Likelihood Ratio
  Approach to the Detection and Estimation of Jumps in Linear Systems."
  *IEEE Transactions on Automatic Control* 21, 108–112. — the windowed GLR.
- Lorden, G. (1971). "Procedures for reacting to a change in distribution."
  *Annals of Mathematical Statistics* 42(6), 1897–1908. — the asymptotic
  optimality framework for CUSUM and the ARL formulation.
- Basseville, M. & Nikiforov, I. V. (1993). *Detection of Abrupt Changes:
  Theory and Application*. Prentice-Hall. — the ARL0/ARL1 framework, the
  zero-state and steady-state conventions, and the scale-change CUSUM.
- Kailath, T. (1968). "An Innovations Approach to Least-Squares Estimation,
  Part I: Linear Filtering in Additive Noise." *IEEE Transactions on Automatic
  Control* 13(6), 646–655. — the innovations property that checks 3–6 test.
- Anderson, B. D. O. & Moore, J. B. (1979). *Optimal Filtering*.
  Prentice-Hall. — the steady-state filter and the algebraic Riccati equation.
- Franklin, G. F., Powell, J. D. & Workman, M. L. (1998). *Digital Control of
  Dynamic Systems*, 3rd ed. Addison-Wesley. — zero-order-hold discretisation.
- Bar-Shalom, Y., Li, X.-R. & Kirubarajan, T. (2001). *Estimation with
  Applications to Tracking and Navigation*. Wiley, ISBN 978-0-471-41655-5. —
  the continuous white-noise-acceleration discrete process-noise model.
- Brier, G. W. (1950). "Verification of forecasts expressed in terms of
  probability." *Monthly Weather Review* 78(1). — the Brier score.

No page number is quoted for the books above, because none was verified in
this environment.
