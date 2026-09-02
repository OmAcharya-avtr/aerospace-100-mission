# FDIScope 0.1.0 — Validation evidence (Level 2, Research)

Every number in this file was produced by running the scripts in this
directory in the build session, on Python 3.11 with numpy, scipy and
scikit-learn on **2 CPU cores**. Each script writes its raw stdout to
`<script>_output.txt` next to itself, and those captures are committed. Rerun
any of them from the product root with `python validation/<script>.py`.

| Script | Raw output | Wall time |
|---|---|---|
| `chi2_false_alarm.py` | `chi2_false_alarm_output.txt` | 3.7 s |
| `cusum_delay.py` | `cusum_delay_output.txt` | 133.9 s |
| `detection_benchmark.py` | `detection_benchmark_output.txt` | 62.8 s |
| `isolation_confusion.py` | `isolation_confusion_output.txt` | 26.1 s |

Total validation wall time **≈ 227 s**, every individual script inside the
3-minute, 2-core compute budget. The 435-test suite adds about 100 s.

**All spacecraft, sensor, actuator and telemetry data in this package is
SIMULATED.** No flight telemetry, no measured innovation record and no
on-orbit fault log is used anywhere. There is no external reference dataset:
every check below compares a measurement against a *closed-form expression*
from the change-detection literature, not against another dataset.

---

## Summary

| ID | Check | Reference | Result | Criterion |
|---|---|---|---|---|
| V1-A1 | Fault-free normalised residual is `N(0, I)` and white | `residuals.nis_consistency`, `whiteness` | mean NIS **2.003549** in [1.993303, 2.006708]; max \|autocorr\| 1.28e-3 vs 4-sigma 6.84e-3 | **PASS** |
| V1-A2 | Chi-squared false-alarm rate vs design `alpha`, non-overlapping windows | `chi2.isf`, 6 cells | 6 of 6 Wilson intervals contain `alpha` | **PASS** |
| V1-A3 | Same over *overlapping* windows | — | per-sample rate / `alpha` in **0.825 – 1.048** | measurement |
| V1-A3b | `chi2_false_alarm_rate(chi2_threshold(a, k), k)` | closed form | worst relative error **7.806e-15** | ≤ 1e-9 — **PASS** |
| V1-A4 | Measured CUSUM `ARL0` (reset on alarm) | Siegmund 1985 | ratios **0.9945, 1.0759, 0.8528** | factor 1.5 — **PASS** |
| V2-B1a | CUSUM run length, exact change-point model | Siegmund 1985 | 9 of 9 cells within 10 %; worst ratio **1.0294** | **PASS** |
| V2-B1b | Same against Wald `h/K` | Basseville & Nikiforov 1993 | ratio **0.914 – 1.372**, sign flip at `mu = 0.858` | measurement |
| V2-B2a | Closed-loop bias, mean-path prediction vs measured median | `analytic.cusum_delay_mean_path` | 3 of 4 cells within 25 %; the **1-sigma cell FAILS** at ratio 1.4706 | **FAIL** (1 of 4) |
| V2-B2b | Same against Siegmund at the steady-state `mu` | Siegmund 1985 | measured / analytic **1.40 – 1.88** | measurement |
| V3-C1 | Per-run false-alarm probability vs the 10 % calibration target | Wilson interval | 3 of 5 PASS; **`glr` 0.0389 and `learned` 0.1500 FAIL** | **FAIL** (2 of 5) |
| V3-C2 | Detection rate within a 600-sample horizon | — | 0.9857 – 1.0000, all ≥ 0.95 | **PASS** |
| V3-C3 | Mean detection delay at matched false-alarm rate | — | **cusum 54.33**, glr 54.27, learned 56.58, chi2 71.42 / 75.76 samples | measurement |
| V3-C4 | Detection AUC | — | **glr 0.9751**, learned 0.9695, cusum 0.9459, chi2 0.9411 | measurement |
| V4-D1 | Signature separability | `SignatureBank.gram` | worst \|cos\| **0.9966**, actuator stuck vs runaway | measurement |
| V4-D2 | Isolation accuracy, full 8×8 matrices | — | **glr 0.4667 [0.4046, 0.5298]**, **learned 0.6958 [0.6349, 0.7506]** | E1 above chance — **PASS** |
| V4-D3 | Isolation accuracy under onset misalignment | — | glr 0.2167 – 0.4917, learned 0.3542 – 0.7375 over ±50 samples | measurement |
| V4-D4 | Confidence calibration | — | glr over-confident by up to **0.5375**; learned within **±0.20** | measurement |

Two criteria failed, and both are reported rather than tuned away: the
mean-path delay prediction at the smallest bias (V2-B2a) and the false-alarm
calibration transfer for two of the five methods (V3-C1). Neither tolerance
was widened and no seed was reselected.

---

## V1 — Does the chi-squared test deliver its design false-alarm rate?

`chi2_false_alarm.py`, 60 fault-free closed-loop runs of 6000 samples,
burn-in 300, seeds 20000–20059 (342 000 samples).

### A1 — is the residual actually `N(0, I)` and white?

Everything downstream depends on this and nothing else, so it is checked
first.

| Quantity | Measured | Expected |
|---|---|---|
| channel means | −0.004025, 0.000452 | 0 |
| channel standard deviations | 1.000934, 1.000831 | 1 |
| mean NIS | **2.003549** | 2, 95 % band [1.993303, 2.006708] |
| lag-1 autocorrelation | −0.000412, 0.001251 | 0, 4-sigma band ±0.006840 |
| lag-2 autocorrelation | 0.001052, 0.001283 | same |
| lag-3 autocorrelation | 0.000356, −0.000199 | same |

**A1 PASS.** The criterion uses a 4-sigma band rather than the 5 % band the
function returns, because six statistics are checked at once and one 2-sigma
excursion among six is ordinary; both bands are printed in the raw output.

### A2 — measured against design, non-overlapping windows

| window | design `alpha` | threshold | measured | 95 % Wilson interval | n | verdict |
|---:|---:|---:|---:|---|---:|---|
| 25 | 1e-01 | 63.1671 | 0.099123 | [0.094227, 0.104243] | 13680 | PASS |
| 25 | 1e-02 | 76.1539 | 0.009211 | [0.007742, 0.010955] | 13680 | PASS |
| 25 | 1e-03 | 86.6608 | 0.001096 | [0.000665, 0.001808] | 13680 | PASS |
| 100 | 1e-01 | 226.0210 | 0.102632 | [0.092902, 0.113252] | 3420 | PASS |
| 100 | 1e-02 | 249.4451 | 0.009064 | [0.006393, 0.012837] | 3420 | PASS |
| 100 | 1e-03 | 267.5405 | 0.000292 | [0.000052, 0.001654] | 3420 | PASS |

**A2 PASS, 6 of 6.** This is the product specification's headline check: the
chi-squared test's empirical false-alarm rate matches its design value under
the fault-free hypothesis, across two decades of design level and two window
lengths, with no tuning of any kind — the threshold is `chi2.isf(alpha, dof)`
and nothing else.

### A3 — the same measurement over overlapping windows

| window | `alpha` | per-sample alarm rate | ratio to `alpha` |
|---:|---:|---:|---:|
| 25 | 1e-01 | 0.100599 | 1.006 |
| 25 | 1e-02 | 0.010327 | 1.033 |
| 25 | 1e-03 | 0.000825 | 0.825 |
| 100 | 1e-01 | 0.104776 | 1.048 |
| 100 | 1e-02 | 0.009915 | 0.991 |
| 100 | 1e-03 | 0.000905 | 0.905 |

Reported as a measurement, not a criterion. The *marginal* probability that
any one overlapping window exceeds the threshold is still `alpha`; what
overlapping destroys is the *independence* of those events, so alarms arrive
in bursts and the variance of an alarm count over a run is far larger than
binomial. That is why the benchmark in V3 calibrates on the **per-run**
probability of at least one alarm rather than on the per-sample rate: a
1e-3 per-sample quantile estimated from correlated windows is worthless (see
`evaluate.calibrate_all_thresholds`).

### A4 — CUSUM mean time between false alarms

| threshold | analytic `ARL0` | measured | ratio | alarms |
|---:|---:|---:|---:|---:|
| 4.0000 | 337.81 | 335.95 | 0.9945 | 1018 |
| 5.7500 | 1999.11 | 2150.94 | 1.0759 | 159 |
| 8.0000 | 19096.94 | 16285.71 | 0.8528 | 21 |

**A4 PASS** against a pre-registered factor-of-1.5 tolerance. The third row
rests on 21 alarms, so its interval is wide; that is a sample-size statement,
not a model statement.

**Overall V1: A1 PASS, A2 PASS, A3 measurement, A3b PASS, A4 PASS.**

---

## V2 — Does the measured detection delay match the analytic expectation?

`cusum_delay.py`. **Convention:** the analytic expressions return a *run
length* — the number of samples the test inspects, at least one —
while `detectors.detection_delay` returns an index difference, zero when the
alarm lands on the onset sample. Everything here compares run lengths, i.e.
`measured index + 1`. That one sample is a 3 % error at a delay of 30 samples
and a 50 % error at a delay of one, and until the convention was pinned down
this check disagreed with the literature by exactly that amount.

### B1 — the exact change-point model

`p_k ~ N(mu, 1)` from sample 0 with the CUSUM started at zero, 4000 trials per
cell, seed 31337. This is precisely the model Wald's and Siegmund's
expressions describe.

| `mu` | `h` | `K` | Wald `h/K` | Siegmund | measured | 95 % CI | m/S | m/W |
|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 0.5 | 4.000 | 0.1250 | 32.000 | 28.743 | 29.260 | [28.722, 29.798] | 1.0180 | 0.9144 |
| 0.5 | 5.750 | 0.1250 | 46.000 | 42.675 | 42.270 | [41.603, 42.937] | 0.9905 | 0.9189 |
| 0.5 | 8.000 | 0.1250 | 64.000 | 60.662 | 60.521 | [59.647, 61.395] | 0.9977 | 0.9456 |
| 1.0 | 4.000 | 0.5000 | 8.000 | 8.342 | 8.390 | [8.244, 8.536] | 1.0058 | 1.0488 |
| 1.0 | 5.750 | 0.5000 | 11.500 | 11.832 | 11.954 | [11.765, 12.143] | 1.0103 | 1.0395 |
| 1.0 | 8.000 | 0.5000 | 16.000 | 16.331 | 16.529 | [16.299, 16.759] | 1.0121 | 1.0330 |
| 2.0 | 4.000 | 2.0000 | 2.000 | 2.666 | 2.744 | [2.701, 2.788] | 1.0294 | 1.3722 |
| 2.0 | 5.750 | 2.0000 | 2.875 | 3.540 | 3.629 | [3.578, 3.680] | 1.0250 | 1.2623 |
| 2.0 | 8.000 | 2.0000 | 4.000 | 4.665 | 4.716 | [4.655, 4.778] | 1.0110 | 1.1791 |

**B1a PASS, 9 of 9** within the pre-registered 10 % tolerance; the worst
disagreement is 2.94 %.

**B1b (measurement).** The Wald ratio `m/W` runs from 0.914 to 1.372, and it
crosses 1 between `mu = 0.5` and `mu = 1.0`. That is where the algebra says it
should: `Siegmund − Wald = 2(e^{-b} + 1.1652 mu − 1)/mu²`, whose sign flips at
`mu = 1/1.1652 = 0.858`. Below that, `h/K` over-predicts the delay; above it,
`h/K` under-predicts by up to 37 %. Siegmund's overshoot correction removes
both errors.

### B2 — the same fault in the closed loop

Step gyro bias, 400 runs per magnitude, onset sample 600, seeds 41000+,
threshold from `cusum_threshold_for_arl0(2000, mu_ss)`.

| bias | `mu_ss` | `h` | Siegmund | mean-path | measured mean | median | m/S | path/median |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.0σ | 1.0000 | 5.750 | 11.833 | 25.0 | 19.845 | 17.0 | 1.6770 | **1.4706 FAIL** |
| 1.5σ | 1.5000 | 5.975 | 5.976 | 12.0 | 11.035 | 10.0 | 1.8465 | 1.2000 PASS |
| 2.0σ | 2.0000 | 5.966 | 3.648 | 7.0 | 6.843 | 6.0 | 1.8755 | 1.1667 PASS |
| 3.0σ | 3.0000 | 5.611 | 1.801 | 3.0 | 3.525 | 3.0 | 1.9568 | 1.0000 PASS |

**B2a FAILED in 1 of 4 cells.** The mean-path recursion is noise-free, so it
cannot see that fluctuation lets the CUSUM cross early, and that help is
largest when the drift per sample is smallest. The ratio falls monotonically
to 1.0 as the bias grows, which is exactly what that explanation predicts, but
the 1-sigma cell is outside the tolerance that was fixed before the run and it
is reported as a failure.

**B2b (measurement).** The `m/S` column — measured against Siegmund evaluated
at the *steady-state* `mu` — runs from 1.40 to 1.88. This is the price of the
closed loop, and it is a modelling statement, not an error: the Siegmund
expression assumes the residual mean steps to `mu_ss` at onset, and in a
closed loop it does not. The estimator absorbs part of the bias first and the
residual mean rises over the closed-loop error time constant (the slow
eigenvalue of `F(I − KH)` is 0.9798, a 49-sample time constant). Feeding the
actual noise-free residual profile into the same CUSUM recursion —
the mean-path column — tracks the measurement to within 47 % at worst and 0 %
at best, against 88 % at worst for the step assumption.

### B3 — warm start against zero start (measurement)

At 2.0σ, `mu_ss = 2.0000`, `h = 5.966`: zero-started mean run length 6.843
samples, warm-started 6.700, i.e. the warm CUSUM is 0.143 samples faster.
A CUSUM that has been running on fault-free data has usually accumulated a
little positive evidence, so it beats the theory it is designed by. The
zero-started figures are the ones the published expressions describe, and the
ones B1 and B2 use.

**Overall V2: B1a PASS (9/9), B1b measurement, B2a FAIL (3/4), B3 measurement.**

---

## V3 — Detection benchmark: false alarms, delay and ROC

`detection_benchmark.py`. 240 training scenarios (seeds 1000–1239), 240
held-out (seeds 5000–5239), 150 fault-free calibration runs (seeds 9000–9149),
150 held-out fault-free runs (seeds 12000–12149). Classes are exactly balanced
by construction: 30 of each of the eight classes in each set.

### The first result is the threshold table

| method | design formula | calibrated | note |
|---|---:|---:|---|
| `chi2_short` | 86.6608 | 92.1187 | closed form, needs no data |
| `chi2_long` | 267.5405 | 270.8737 | closed form, needs no data |
| `cusum` | 5.7504 | 9.1133 | closed form, needs no data |
| `glr` | — | 17.3754 | **no closed form: unusable without fault-free data** |
| `learned` | — | 0.7921 | **no closed form: unusable without fault-free data** |

The chi-squared and CUSUM thresholds follow from a distribution and a target
rate with no data at all. The GLR bank and the classifier have no such
formula: they cannot be operated until somebody has collected fault-free data
and picked a quantile. Everything below is measured at the *calibrated*
thresholds so that the delays are comparable, which hides that asymmetry —
hence stating it first.

### C1 — false-alarm rate on 180 held-out fault-free runs

| method | per-run | 95 % Wilson | target | per-sample | alarming/total | verdict |
|---|---:|---|---:|---:|---:|---|
| `chi2_short` | 0.1167 | [0.0776, 0.1718] | 0.10 | 0.000281 | 86/306000 | PASS |
| `chi2_long` | 0.1222 | [0.0821, 0.1781] | 0.10 | 0.000526 | 161/306000 | PASS |
| `cusum` | 0.1389 | [0.0959, 0.1970] | 0.10 | 0.000493 | 151/306000 | PASS |
| `glr` | **0.0389** | [0.0190, 0.0781] | 0.10 | 0.000085 | 26/306000 | **FAIL** |
| `learned` | **0.1500** | [0.1052, 0.2094] | 0.10 | 0.000719 | 220/306000 | **FAIL** |

**C1 FAILED for 2 of 5 methods.** Both failures are in the calibrated
methods, and they fail in opposite directions: the GLR bank's threshold
transfers *conservatively* (0.039 against a 0.10 target, so its delays in C2
are if anything pessimistic relative to the others), and the classifier's
transfers *permissively* (0.150, so its delays are flattered). The three
methods with a closed-form threshold all land inside their interval. This is
the calibration-transfer error a data-driven threshold carries and an
analytic one does not, measured rather than assumed.

### C2 — detection rate and delay at the matched operating point

| method | detection rate | censored | mean delay | 95 % CI | median | p90 |
|---|---:|---:|---:|---|---:|---:|
| `chi2_short` | 0.9857 | 3 | 71.42 | [59.81, 83.02] | 49.0 | 166.0 |
| `chi2_long` | 0.9857 | 3 | 75.76 | [63.88, 87.64] | 57.0 | 168.0 |
| `cusum` | **1.0000** | 0 | **54.33** | [44.90, 63.76] | **33.0** | 125.7 |
| `glr` | **1.0000** | 0 | **54.27** | [46.85, 61.69] | 41.0 | **117.1** |
| `learned` | 0.9952 | 1 | 56.58 | [48.25, 64.90] | 42.0 | 127.0 |

Delays are in samples at `dt = 0.1 s`. A censored run never alarmed inside the
600-sample horizon; those runs are counted and excluded from the mean, not
zero-filled.

**Where the classical tests win.** The sequential CUSUM has the shortest mean
and median delay of any method and misses nothing, and the classical GLR bank
has the shortest p90. The learned classifier is slower than both — by 2.25
samples on the mean against the CUSUM — while running at a *higher* measured
false-alarm rate (0.150 against 0.139). The paired intervals overlap heavily,
so the honest statement is that **the learned classifier does not beat either
classical detector on delay**, not that it is measurably worse.

### C3 — mean detection delay per fault class [samples]

| fault class | `chi2_short` | `chi2_long` | `cusum` | `glr` | `learned` |
|---|---:|---:|---:|---:|---:|
| sensor_bias | 4.2 | 6.9 | **3.2** | 13.0 | 4.9 |
| sensor_drift | 76.1 | 76.3 | **48.3** | 51.6 | 58.4 |
| sensor_stuck | 38.7 | 45.1 | **26.2** | 33.2 | 35.5 |
| sensor_dropout | 5.3 | 6.6 | **1.7** | 7.0 | 3.7 |
| actuator_loss_of_effectiveness | 91.5 | 96.5 | 62.5 | **61.1** | 70.6 |
| actuator_stuck | 99.9 | 104.4 | 83.6 | **82.4** | 85.3 |
| actuator_runaway | 189.8 | 194.0 | 154.8 | **131.6** | 136.0 |

The CUSUM is fastest on all four sensor faults; the GLR bank is fastest on all
three actuator faults, because it is matched to a *time profile* and an
actuator fault develops through the plant dynamics rather than appearing at
once. The learned classifier is never fastest on any class.

### C4 — ROC over window-level scores

840 faulted windows (offsets 0, 25, 50, 100 samples after onset) against 840
fault-free pre-onset windows.

| method | AUC | TPR at FPR 0.10 | at 0.01 | at 0.001 |
|---|---:|---:|---:|---:|
| `chi2_long` | 0.9411 | 0.8643 | 0.8167 | 0.7571 |
| `cusum` | 0.9459 | 0.9000 | 0.8643 | 0.8476 |
| `glr` | **0.9751** | **0.9417** | **0.9131** | **0.8821** |
| `learned` | 0.9695 | 0.9381 | 0.9024 | 0.8476 |

**The classical GLR bank has the highest AUC**, and beats the learned
classifier at every operating point measured. Per class:

| fault class | `chi2_long` | `cusum` | `glr` | `learned` |
|---|---:|---:|---:|---:|
| sensor_bias | 0.9036 | 0.8650 | **0.9722** | 0.9702 |
| sensor_drift | 0.9823 | 0.9928 | **0.9984** | 0.9959 |
| sensor_stuck | 0.9878 | 0.9996 | 0.9971 | **1.0000** |
| sensor_dropout | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| actuator_loss_of_effectiveness | 0.9204 | 0.9435 | **0.9824** | 0.9607 |
| actuator_stuck | 0.9706 | 0.9683 | 0.9733 | **0.9756** |
| actuator_runaway | 0.8231 | 0.8522 | **0.9020** | 0.8843 |

Every method is essentially perfect on a dropped channel and weakest on a slow
actuator runaway — the residual takes hundreds of samples to leave the noise.

**Overall V3: C1 FAIL (2 of 5 methods outside their interval), C2 PASS,
C3 and C4 measurements.**

---

## V4 — Isolation, reported in full

`isolation_confusion.py`. Both methods isolate the identical window
`[onset, onset + 100)` of every one of the 240 held-out runs.

### D1 — which hypotheses are structurally inseparable

Cosine between the unit fault signatures (rows and columns in bank order):

| | s-bias | s-drift | s-stuck | s-drop | a-LOE | a-stuck | a-run |
|---|---:|---:|---:|---:|---:|---:|---:|
| sensor_bias | 1.0000 | −0.3084 | −0.5690 | 0.4131 | −0.2350 | 0.2484 | −0.2554 |
| sensor_drift | −0.3084 | 1.0000 | 0.6408 | 0.1528 | 0.5096 | −0.6110 | 0.6047 |
| sensor_stuck | −0.5690 | 0.6408 | 1.0000 | 0.3351 | −0.1157 | −0.0847 | 0.0599 |
| sensor_dropout | 0.4131 | 0.1528 | 0.3351 | 1.0000 | −0.1621 | 0.0011 | −0.0206 |
| actuator_LOE | −0.2350 | 0.5096 | −0.1157 | −0.1621 | 1.0000 | −0.8679 | 0.9032 |
| actuator_stuck | 0.2484 | −0.6110 | −0.0847 | 0.0011 | −0.8679 | 1.0000 | **−0.9966** |
| actuator_runaway | −0.2554 | 0.6047 | 0.0599 | −0.0206 | 0.9032 | **−0.9966** | 1.0000 |

A GLR bank scores each hypothesis with `(phi_j · r)²`, so the sign is
irrelevant and −0.9966 is as bad as +0.9966: **a stuck actuator and a runaway
actuator cannot be separated by this classical isolator at any sample size.**
The three actuator signatures form a near-degenerate cluster (all pairwise
\|cos\| ≥ 0.868), and the confusion matrix below shows exactly that.

### D2 — full confusion matrices

Rows are truth, columns are prediction. Nothing is summarised away.

**Classical GLR bank**, overall accuracy **0.466667**, 95 % Wilson interval
[0.4046, 0.5298] against a chance level of 0.1250:

| true \ predicted | none | s-bias | s-drift | s-stuck | s-drop | a-LOE | a-stuck | a-run | recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| none | **30** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.0000 |
| sensor_bias | 0 | **20** | 10 | 0 | 0 | 0 | 0 | 0 | 0.6667 |
| sensor_drift | 3 | 0 | **14** | 13 | 0 | 0 | 0 | 0 | 0.4667 |
| sensor_stuck | 1 | 0 | 14 | **15** | 0 | 0 | 0 | 0 | 0.5000 |
| sensor_dropout | 0 | 5 | 10 | 14 | **1** | 0 | 0 | 0 | 0.0333 |
| actuator_LOE | 5 | 0 | 0 | 0 | 0 | **14** | 1 | 10 | 0.4667 |
| actuator_stuck | 6 | 0 | 0 | 0 | 0 | 1 | **13** | 10 | 0.4333 |
| actuator_runaway | 13 | 0 | 0 | 0 | 0 | 3 | 9 | **5** | 0.1667 |
| **precision** | 0.5172 | 0.8000 | 0.2917 | 0.3571 | 1.0000 | 0.7778 | 0.5652 | 0.2000 | |

**Learned classifier**, overall accuracy **0.695833**, 95 % Wilson interval
[0.6349, 0.7506]:

| true \ predicted | none | s-bias | s-drift | s-stuck | s-drop | a-LOE | a-stuck | a-run | recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| none | **30** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.0000 |
| sensor_bias | 1 | **29** | 0 | 0 | 0 | 0 | 0 | 0 | 0.9667 |
| sensor_drift | 5 | 3 | **22** | 0 | 0 | 0 | 0 | 0 | 0.7333 |
| sensor_stuck | 0 | 0 | 1 | **28** | 1 | 0 | 0 | 0 | 0.9333 |
| sensor_dropout | 0 | 0 | 0 | 1 | **29** | 0 | 0 | 0 | 0.9667 |
| actuator_LOE | 6 | 1 | 3 | 0 | 0 | **3** | 14 | 3 | 0.1000 |
| actuator_stuck | 7 | 0 | 0 | 0 | 0 | 0 | **16** | 7 | 0.5333 |
| actuator_runaway | 16 | 0 | 0 | 0 | 0 | 4 | 0 | **10** | 0.3333 |
| **precision** | 0.4615 | 0.8788 | 0.8462 | 0.9655 | 0.9667 | 0.4286 | 0.5333 | 0.5000 | |

**E1 PASS**: both accuracy intervals sit above chance.

Head to head:

| fault class | glr recall | learned recall |
|---|---:|---:|
| none | 1.0000 | 1.0000 |
| sensor_bias | 0.6667 | **0.9667** |
| sensor_drift | 0.4667 | **0.7333** |
| sensor_stuck | 0.5000 | **0.9333** |
| sensor_dropout | 0.0333 | **0.9667** |
| actuator_loss_of_effectiveness | **0.4667** | 0.1000 |
| actuator_stuck | 0.4333 | **0.5333** |
| actuator_runaway | 0.1667 | **0.3333** |
| **overall accuracy** | 0.4667 | **0.6958** |

The learned classifier wins overall and on six of the seven fault classes.
**The classical bank wins on loss of effectiveness, 0.4667 against 0.1000** —
a 4.7× recall advantage on the fault class the classifier is worst at. The
classifier sends 14 of 30 loss-of-effectiveness cases to *actuator stuck*; the
GLR bank's model-derived signature keeps them apart because the two faults
have different time profiles even though their directions are close.

Two GLR failures are worth naming because they are structural, not
statistical:

* **sensor_dropout recall 0.0333.** A dropped channel reports zero, so the
  residual is the negative of the true state, whose sign and size depend on
  where in the sinusoidal reference the fault starts. Averaging the signature
  over eight onset phases — the compromise `build_default_bank` makes —
  destroys it. The learned classifier reaches 0.9667 on the same windows
  because it never assumes a fixed time profile.
* **actuator_runaway recall 0.1667, precision 0.2000.** The −0.9966 signature
  cosine with `actuator_stuck` in D1 is the whole explanation.

### D3 — what "the onset is known" is worth

Both methods place their window at the *true* onset, which no detector can do.
Shifting the window measures the cost:

| offset [samples] | glr | learned |
|---:|---:|---:|
| −50 | 0.2167 | 0.3542 |
| −25 | 0.3125 | 0.4250 |
| −10 | 0.4250 | 0.5125 |
| 0 | 0.4667 | 0.6958 |
| +10 | 0.4750 | 0.7167 |
| +25 | **0.4917** | **0.7375** |
| +50 | 0.3833 | 0.6375 |

Both peak slightly *after* the true onset — a window that starts 25 samples
late contains more developed fault evidence — and both collapse when the
window starts early. At −50 samples the classifier loses 49 % of its accuracy
and the GLR bank 54 %. Anyone using the detector's own alarm time to place the
window should read this table, not the D2 headline.

### D4 — is the confidence a calibration or a number?

Accuracy within confidence buckets; a positive gap means over-confidence.

| method | bucket | n | accuracy | mean confidence | gap |
|---|---|---:|---:|---:|---:|
| glr | [0.00, 0.40) | 1 | 0.0000 | 0.3585 | +0.3585 |
| glr | [0.40, 0.60) | 14 | 0.2857 | 0.5273 | +0.2416 |
| glr | [0.60, 0.80) | 27 | 0.4815 | 0.6829 | +0.2015 |
| glr | [0.80, 0.90) | 3 | 0.6667 | 0.8533 | +0.1867 |
| glr | [0.90, 1.00) | 137 | 0.4599 | 0.9973 | **+0.5375** |
| glr | no fault declared | 58 | — | — | — |
| learned | [0.00, 0.40) | 36 | 0.4722 | 0.3428 | −0.1294 |
| learned | [0.40, 0.60) | 78 | 0.6410 | 0.5014 | −0.1396 |
| learned | [0.60, 0.80) | 57 | 0.8246 | 0.7066 | −0.1180 |
| learned | [0.80, 0.90) | 47 | 0.7872 | 0.8495 | +0.0623 |
| learned | [0.90, 1.00) | 22 | 0.7273 | 0.9259 | +0.1987 |

**The GLR posterior is badly over-confident**: 137 of its 182 declarations sit
in the top bucket with a mean confidence of 0.9973 and an accuracy of 0.4599,
a gap of 0.54. That is what happens when a posterior assumes exactly one of
seven modelled faults is present with equal prior and the signatures are
nearly collinear — the likelihood ratio between two indistinguishable
hypotheses is arbitrary, and the softmax over `exp(l_j/2)` turns a marginal
statistic difference into near-certainty. The classifier's vote fraction
tracks accuracy far better (gaps within ±0.20, under-confident at the low end
and over-confident at the high end) but is still not a calibrated probability.
Neither should be treated as one.

### D5 — what the classifier leans on

| feature | importance | | feature | importance |
|---|---:|---|---|---:|
| corr_01 | 0.1172 | | autocorr1_ch0 | 0.0565 |
| autocorr1_ch1 | 0.0984 | | max_abs_ch0 | 0.0562 |
| cusum_range_ch1 | 0.0849 | | slope_ch1 | 0.0555 |
| mean_nis | 0.0813 | | std_ch1 | 0.0534 |
| exceed_frac | 0.0590 | | max_abs_ch1 | 0.0523 |
| mean_ch1 | 0.0585 | | cusum_range_ch0 | 0.0513 |
| mean_ch0 | 0.0584 | | max_nis | 0.0475 |
| | | | std_ch0 | 0.0356 |
| | | | slope_ch0 | 0.0341 |

Impurity importance is biased toward high-cardinality features and no causal
reading should be taken from it. What it does show is that the importance is
spread across all sixteen features rather than concentrated: the two features
that reproduce the chi-squared test (`mean_nis`, `exceed_frac`) carry 0.1403
between them, so 86 % of the model's split value comes from structure the
chi-squared test throws away. That is the mechanism behind the isolation
result in D2 and it is consistent with the classifier failing to win on
*detection*, where the chi-squared statistic is most of the information.

**Overall V4: E1 PASS, D1–D5 measurements.**

---

## What is not validated

* **No comparison against a real innovation sequence.** Every residual here
  comes from a simulator whose model is exactly the filter's model. Real
  filters are mismatched — unmodelled dynamics, coloured sensor noise, an
  imperfectly known `Q` — and a mismatched filter has a non-white innovation,
  which breaks the chi-squared distribution the whole design rests on. Nothing
  in this repository measures that.
* **No comparison against another FDI implementation.** The checks above are
  against closed-form expressions, not against an independent code base.
* **One plant, one operating point.** Single axis, one inertia, one set of
  controller gains, one reference manoeuvre. Nothing here says how any of these
  numbers move with the plant.
* **Multiple simultaneous faults are not modelled or tested.** Every scenario
  injects exactly one fault.
* **The false-alarm rates are measured over 306 000 samples**, which resolves a
  1e-3 per-sample rate to about ±20 % and cannot resolve 1e-5 at all.
