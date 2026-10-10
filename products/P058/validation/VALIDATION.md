# Validation evidence — telemdrift 0.1.0

**Validation level 2.** Measured ARL against a published reference table and an
independent closed form, property tests for the algebraic identities that
actually hold, and a measured characterisation of five failure modes.
**Research-grade. Not flight-qualified, not certified, not approved for
operational aerospace use.**

Every number in this file and in README.md was produced by a script in
this directory, executed in this container on 2026-10-10, with its raw stdout
committed under `outputs/` as `<script>.txt`. Nothing here was copied
from a paper, estimated, or rounded by hand. The checks that undercut this
package are in the tables on purpose and are marked.

**Environment.** Python 3.13.16 on Linux 6.18.44, NumPy 2.5.3, SciPy 1.18.1,
scikit-learn 1.9.1, Matplotlib 3.11.2, joblib 1.6.0, pytest 9.1.1, Hypothesis
6.168.5. `os.cpu_count()` = 2 and `len(os.sched_getaffinity(0))` = 2: **two
cores, shared and contended with sibling build agents.** Declared runtime
dependencies are NumPy, SciPy, scikit-learn and joblib only; Matplotlib is an
optional extra needed for the figures. Raw output:
`outputs/validate_environment.txt`.

**Test suite.** Counted from junit XML, not from a pytest stdout line:
**302 collected, 302 passed, 0 failed, 0 errored, 0 skipped**, suite time 123.9 s.
Raw output `outputs/validate_tests.txt`, which quotes the XML's own `testsuite`
element verbatim. The count is read from the XML rather than from a pytest
stdout line because a configuration that collects nothing exits 0 and prints a
success line; that happened to another product in this portfolio and a
fabricated count was carried forward for three weeks. The XML itself is written
to a temporary directory and not committed: it is one multi-kilobyte line whose
test identifiers trip a secret scanner's high-entropy heuristic sixteen times,
and a false positive in the release gate blocks sibling products.

**Timings move, counts do not.** Wall-clock figures in this container move
10-40 % between runs under contention. Every cost in this document is a ratio or
a measured range over repeated runs, never a single-run microsecond figure. The
primary numbers here are run lengths, delays, counts and their standard errors,
which are deterministic given the seeds and reproduce exactly.

**Compute budget.** The validation suite is about 18 minutes of wall clock
standalone; the slowest single script is `validate_transient.py` at 183 s. The
slowest single computation anywhere is the 302-test suite at 123.9 s.
Nothing approaches a figure that would need splitting.

---

## 1. What is measured, and the conventions it is measured under

Three definitions, stated once, because every number below depends on them and
two of the three have defensible alternatives that give different answers.

**ARL0** — mean samples to a false alarm on a stationary stream. Measured by
running a long stationary stream, resetting the detector at every alarm, and
averaging the run lengths. The run after the last alarm is right-censored and
excluded, which biases ARL0 **downward**; the censored fraction is reported with
every figure and is under 1.4 % everywhere in this document. Section 2 shows the
bias has the predicted sign.

**ARL1** — mean delay from the change index to the first alarm at or after it,
**steady-state convention**: the detector runs normally through a 1000-sample
stationary pre-change segment, resetting at every false alarm exactly as it
would in service. No replicate is excluded. Section 6 measures the two
alternatives and explains why this one was chosen.

**Monte Carlo standard error** — `s / sqrt(n)` over the run lengths or delays,
quoted with every ARL figure. At the budgets here the relative standard error is
2 to 5 %, so a point estimate quoted to four significant figures would be a lie
about precision and this package does not produce one.

---

## 2. Known answers against a published reference

`validate_known_answers.py` → `outputs/validate_known_answers.txt`.
**FAILED CHECKS: 0.** 18 checks passed.

### 2.1 The reference chain, in the open

NIST/SEMATECH e-Handbook of Statistical Methods, section 6.3.2.3.1 "Cusum
Average Run Length",
`https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc3231.htm`, read
2026-10-10. For `k = 0.5` it tabulates the **one-sided** in-control ARL as 336
at `h = 4` and 930 at `h = 5`, and the ARL at a one-sigma mean shift as 8.38 and
10.4. The page states, verbatim: "If one has to control both positive and
negative deviations, as is usually the case, two one-sided charts are used". It
credits no external author: "This Handbook used a computer program that
furnished the required ARLs given the standardized h and k."

This package's CUSUM runs **both** one-sided charts against the same `h`, so it
false-alarms at twice the rate and the reference is the handbook value **halved**:

```
h = 4:  ARL0 = 336 / 2 = 168
h = 5:  ARL0 = 930 / 2 = 465
```

Quoting 336 for a two-sided chart would be wrong by a factor of two, and a
factor of two is exactly the error that survives review because both numbers
look plausible.

Independently, Siegmund's closed form for the one-sided ARL with `b = h + 1.166`
and `D = delta - k`:

```
h = 4:  (exp(5.166) - 5.166 - 1) / 0.5 = (175.195 - 6.166) / 0.5 = 338.06
h = 5:  (exp(6.166) - 6.166 - 1) / 0.5 = (476.265 - 7.166) / 0.5 = 938.20
```

The two references agree to 0.62 % and 0.88 %, so either can anchor the test and
neither is taken on trust.

### 2.2 Measured

12 seeds × 60 000 stationary samples = 720 000 samples per threshold.

| Quantity | Reference | Measured | Rel. diff | Runs | Status |
|---|---|---|---|---|---|
| Two-sided CUSUM ARL0, `h=4` | 168.0 | **164.6 ± 2.5** | −2.0 % | 4366 | PASS (tol 10 %) |
| Two-sided CUSUM ARL0, `h=5` | 465.0 | **458.5 ± 11.8** | −1.4 % | 1560 | PASS (tol 10 %) |
| Sign of the deviation | below, from censored-tail exclusion | below at both `h` | — | — | PASS |
| Zero-state ARL1, `h=4`, 1σ shift | 8.38 | **8.22 ± 0.19** | −1.8 % | 600 | PASS (tol 15 %) |
| Zero-state ARL1, `h=5`, 1σ shift | 10.4 | **10.49 ± 0.23** | +0.9 % | 600 | PASS (tol 15 %) |

The 10 % tolerance was set from the Monte Carlo standard error before the
measurement, not after it: the relative SEM is 1.5 % at `h = 4` and 2.6 % at
`h = 5`, so 10 % is four to seven standard errors. The halving is itself an
approximation — the two arms are driven by the same observations and are not
independent — which is why the band is not tighter.

The ARL1 rows use the **zero-state** convention to match the handbook, which is
not the steady-state convention used everywhere else in this repository. The two
are different quantities and section 6 measures the difference.

### 2.3 Detector internals, hand-computed

| Check | Hand arithmetic | Measured | Status |
|---|---|---|---|
| CUSUM recursion, `k=0.5`, `h=2`, five samples of 1.0 | `S+` = 0.5, 1.0, 1.5, 2.0 (not > 2), 2.5 → alarm on the fifth | alarm on the fifth, `S+` = 2.5, `S−` = 0.0 | PASS |
| EWMA, `r=0.5`, `L=3`, two samples of 1.0 | `z` = 0.5 then 0.75; limits 1.5 then `3·sqrt(0.3125)` = 1.677051 | 0.500000 / 1.500000, 0.750000 / 1.677051 | PASS |
| Page-Hinkley, `delta=0`, samples 0, 2, 4 | means 0, 1, 2; `m` = 0, 1, 3; `m_min` 0, `m_max` 3 | mean 2.000000, `m` 3.000000, 0.000000, 3.000000 | PASS |
| Page-Hinkley on a constant stream | cannot fire at any threshold: every increment is `−delta` | 1000 samples of 1000.0 at `lambda = 1`: no alarm | PASS |
| ADWIN `eps_cut`, `n0=n1=10`, `delta=0.05` | `sqrt(0.05 · ln 1600)` = `sqrt(0.36888794)` = 0.60736146 | 0.60736146 | PASS |
| Windowed KS statistic | `scipy.stats.ks_2samp` on 500 random pairs, sizes 5–119 | worst absolute difference **0.000e+00** | PASS (tol 1e-12) |
| KS default threshold inverts to its declared alpha | `alpha = 0.005` | `P(D > c)` = 0.005000000 | PASS (tol 1e-9) |

---

## 3. Operating points: the product's central claim

`validate_arl_calibration.py` → `outputs/validate_arl_calibration.txt`.
**FAILED CHECKS: 0.** 11 checks passed.

### 3.1 The five shipped defaults are five different false-alarm rates

8 seeds × 50 000 stationary samples = 400 000 per detector.

| Detector | Default | ARL0 | SEM | rel. SEM | Runs |
|---|---|---|---|---|---|
| CUSUM | `h = 4` | 167.9 | 3.4 | 2.05 % | 2373 |
| Page-Hinkley | `lambda = 50` | 1253.4 | 37.9 | 3.02 % | 313 |
| EWMA | `L = 3` | 891.8 | 41.9 | 4.70 % | 442 |
| Windowed KS | `c = 0.211981` | 7819.4 | 1280.7 | 16.38 % | 33 |
| ADWIN | `delta = 0.002` | 183.6 | 2.8 | 1.54 % | 2174 |

**Spread 46.6x.** CUSUM's default lands at the value the SPC literature predicts
(168 for a two-sided chart at `k = 0.5`, section 2), so this is not a
measurement artefact: the defaults genuinely encode different intentions.

### 3.2 Calibration, and the precision it actually achieves

Target ARL0 = 500 samples. Fitted on 6 seeds × 30 000 samples; reported on 8
disjoint seeds × 50 000 samples.

| Detector | Threshold | Fitted ARL0 | Held-out ARL0 | SEM | Target error |
|---|---|---|---|---|---|
| CUSUM | `h = 5.13989` | 498.7 | 581.1 | 22.7 | **+16.2 %** |
| Page-Hinkley | `lambda = 29.9244` | 499.1 | 485.7 | 9.3 | −2.9 % |
| EWMA | `L = 2.83155` | 498.8 | 527.1 | 20.4 | +5.4 % |
| Windowed KS | `c = 0.134999` | 457.5 | 467.9 | 7.3 | −6.4 % |
| ADWIN | `delta = 9.94156e-08` | 500.4 | 520.3 | 16.5 | +4.1 % |
| Learned RF | `p* = 0.68375` | 519.9 | 543.2 | 18.0 | +8.6 % |

Mean absolute held-out error 7.0 %, worst 16.2 %. **Residual spread 1.24x**
(468 to 581 samples), against 46.6x at the defaults.

The held-out number differs from the fitted one because bisecting against a
noisy objective selects the threshold whose *calibration* measurement happened
to land near target. That is a selection effect of the same order as the
combined Monte Carlo error of the two estimates. Closing it needs more
stationary samples than two cores afford, so it is reported rather than removed,
and the equal-ARL0 tables are described as approximate throughout.

### 3.3 ARL0 does not describe the run-length distribution

| Detector | ARL0 | median | p90 | CV | runs |
|---|---|---|---|---|---|
| CUSUM | 581.1 | 391 | 1336 | 1.02 | 679 |
| Page-Hinkley | 485.7 | 427 | 843 | 0.54 | 815 |
| EWMA | 527.1 | 374 | 1222 | 1.06 | 753 |
| Windowed KS | 467.9 | 400 | 720 | 0.45 | 850 |
| ADWIN | 520.3 | 385 | 1120 | 0.87 | 762 |

A CV near 1 with a median well below the mean is a heavy right tail. None of
these detectors is memoryless, so `ARL0 = 1/p` does not hold and the per-sample
false-alarm probability is not constant — pinned as a deliberate negative
property in `tests/test_properties.py`.

### 3.4 Warm-up blindness, which an ARL0/ARL1 pair hides

Fraction of a stationary stream on which no alarm is possible, at each
detector's calibrated threshold.

| Detector | un-armed | warm-up |
|---|---|---|
| CUSUM / Page-Hinkley / EWMA | **0.0 %** | 0 samples |
| Windowed KS | **65.9 %** | 300 samples |
| ADWIN | **12.1 %** | 60 samples |
| Learned RF | **9.7 %** | 50 samples |

The windowed KS test's 300-sample warm-up is comparable to its 468-sample ARL0,
so it spends two thirds of its life unable to detect anything. This is the
mechanism behind its poor ARL1 and behind the non-monotonicity in section 5.

### 3.5 The KS per-test level is not a stream-level false-alarm rate

The asymptotic KS p-value controls one test on independent data. The detector
tests every 5 samples on windows overlapping by 95 %.

| Threshold `c` | per-test `P(D>c)` | ARL0 if tests were independent | measured ARL0 | factor |
|---|---|---|---|---|
| default 0.21198 | 0.005000 | 1000.0 | 7878.1 | **7.9x** |
| calibrated 0.13500 | 0.176079 | 28.4 | 470.7 | **16.6x** |

Treating the p-value as a stream-level rate understates ARL0 by roughly an order
of magnitude at both thresholds.

---

## 4. The delay-versus-false-alarm trade-off

`validate_tradeoff.py` → `outputs/validate_tradeoff.txt`, figure
`../screenshots/tradeoff_validation.png`. **FAILED CHECKS: 0.** 14 checks passed.

36 measured points: 6 detectors × 6 thresholds, each with an ARL0 over 4 × 25 000
stationary samples and an ARL1 over 200 seeded replicates. Each sweep spans at
least a decade of ARL0 and each detector's ARL0 is monotone in its own
threshold, both checked.

**Interpolated ARL1 at ARL0 = 500 on a +1σ mean step**, by linear interpolation
in `log(ARL0)`:

| Detector | interpolated | direct measurement (section 5) |
|---|---|---|
| CUSUM | 9.6 | 9.4 ± 0.3 |
| EWMA | 9.6 | 9.2 ± 0.3 |
| Learned RF | 12.6 | 17.4 ± 2.8 |
| Page-Hinkley | 31.8 | 30.4 ± 2.6 |
| ADWIN | 34.5 | 30.3 ± 4.0 |
| Windowed KS | 104.5 | 102.1 ± 6.4 |

Two independent routes to the same quantity agree within the error bars for five
of six detectors; the learned detector's two estimates differ by about 1.7
combined standard errors, which is the Monte Carlo noise at 200 against 300
replicates.

### 4.1 Non-monotone trade-off curves

| Detector | ARL1 at lowest ARL0 | ARL1 at highest ARL0 | monotone |
|---|---|---|---|
| CUSUM | 5.3 | 15.3 | yes |
| Page-Hinkley | 8.9 | 67.4 | yes |
| EWMA | 6.0 | 13.2 | yes |
| Windowed KS | 164.8 | 68.5 | **NO** |
| ADWIN | 25.9 | 22.4 | **NO** |
| Learned RF | 16.8 | 129.9 | **NO** |

The windowed KS test's shortest measured delay (64.4 samples) is at ARL0 = 1243,
not at its tightest threshold (164.8 samples at ARL0 = 343). Making it more
sensitive makes it **slower**, because its own false alarms put it back into a
300-sample warm-up. ADWIN shows the same effect weakly. This is invisible in an
equal-ARL0 table and is the reason the curve is the headline figure.

---

## 5. Delay by change type, at equal measured ARL0

`validate_change_types.py` → `outputs/validate_change_types.txt`.
**FAILED CHECKS: 0.** 8 checks passed. 300 seeded replicates per cell,
steady-state convention, censored at 1500 samples, bootstrap 95 % intervals in
the raw output.

| Detector | ARL0 | mean step +1.0 | mean step +0.5 | variance ×2.0 | drift 0.02/sample | transient +4.0 |
|---|---|---|---|---|---|---|
| CUSUM | 581 | **9.4 ± 0.3** | 38.0 ± 2.0 | **12.5 ± 0.7** | 35.7 ± 0.7 | 0.9 |
| Page-Hinkley | 486 | 30.4 ± 2.6 | 63.9 ± 4.7 | 90.9 ± 4.8 | 44.1 ± 0.9 | 5.1 |
| EWMA | 527 | **9.2 ± 0.3** | **31.4 ± 1.4** | 21.2 ± 1.2 | **33.2 ± 0.6** | 1.2 |
| Windowed KS | 468 | 102.1 ± 6.4 | ≥125.6 ± 9.7 | ≥126.7 ± 8.7 | 104.9 ± 5.0 | 106.9 |
| ADWIN | 520 | 30.3 ± 4.0 | 56.9 ± 5.9 | 58.3 ± 2.5 | 40.9 ± 0.7 | 13.2 |
| Learned RF | 543 | 17.4 ± 2.8 | 45.5 ± 3.5 | 16.5 ± 1.6 | 34.0 ± 0.8 | ≥43.2 |

`≥` marks a right-censored mean, a lower bound. The transient column is a
negative control: a **large** number there is the good outcome.

**The change type where everything does badly is the variance step for the
self-referential detectors.** Page-Hinkley needs 90.9 samples and the windowed
KS test at least 126.7, against CUSUM's 12.5. Page-Hinkley subtracts a running
mean and a symmetric variance increase moves no mean; the KS test is in warm-up
two thirds of the time.

---

## 6. ARL1 conventions, measured rather than argued

Same seeded mean-step streams, same calibrated thresholds, three conventions.

| Detector | (a) steady-state | (b) conditioned | kept | (c) zero-state |
|---|---|---|---|---|
| CUSUM | 9.4 | 9.1 | 12 % | 10.1 |
| Page-Hinkley | 30.4 | 21.4 | 5 % | 442.2 |
| EWMA | 9.2 | 9.6 | 16 % | 7.7 |
| Windowed KS | 102.1 | 22.2 | 4 % | 445.3 |
| ADWIN | 30.3 | 17.7 | 13 % | 446.7 |

**(b) conditioned** — discard replicates that false-alarm before the change — is
what the first implementation of this package did, and it keeps 4 to 16 % of
replicates at these operating points. Discarding 85 % of the data on a criterion
correlated with the detector's state is a selection effect larger than any
difference between the detectors. Rejected.

**(c) zero-state** — start the detector at the change — splits the detectors into
two families. CUSUM and EWMA standardise against a declared `mu0` and `sigma0`
and detect a cold-started step in 10.1 and 7.7 samples. Page-Hinkley, the
windowed KS test and ADWIN are self-referential: with no pre-change history a
mean step is indistinguishable from a stationary stream at a different level,
and their delays run to the 1500-sample censoring budget. This is a practical
distinction, not a curiosity: a channel after a reboot or a mode change gives
you exactly that problem.

---

## 7. The transient negative control

`validate_transient.py` → `outputs/validate_transient.txt`, figure
`../screenshots/transient_validation.png`. **FAILED CHECKS: 0.** 3 checks passed.

A +A σ excursion lasting 20 samples, after which the channel recovers. There is
no change, so every alarm is a false alarm. A raw firing rate means nothing on
its own — a detector at ARL0 = 500 false-alarms inside any 70-sample window
about 13 % of the time — so a matched stationary baseline is measured on the
same seeds and the excess is what the transient caused. 250 replicates.

| Detector | +2σ | +4σ | +8σ | baseline | excess at +4σ |
|---|---|---|---|---|---|
| CUSUM | 100.0 % | 100.0 % | 100.0 % | 11.2 % | **+88.8 pp** |
| Page-Hinkley | 98.8 % | 100.0 % | 100.0 % | 14.4 % | **+85.6 pp** |
| EWMA | 100.0 % | 100.0 % | 100.0 % | 14.0 % | **+86.0 pp** |
| Windowed KS | 49.6 % | 51.2 % | 51.2 % | 18.4 % | **+32.8 pp** |
| ADWIN | 100.0 % | 100.0 % | 100.0 % | 13.2 % | **+86.8 pp** |
| Learned RF | 97.6 % | 94.4 % | 93.2 % | 13.2 % | **+81.2 pp** |

**All six fire.** The windowed KS test fires least often and that is not
discrimination: it is un-armed 65.9 % of the time (section 3.4). The learned
detector fires on 94.4 % but takes longer to do it (43.2 samples against CUSUM's
0.9, section 5), which is the only visible discrimination anywhere in this
table and it is not enough to be useful.

**What explicit supervision buys.** Retraining the learned detector with
transient streams as negatives and recalibrating to the same ARL0 (achieved
519.5 ± 16.5, +3.9 % of target):

| Variant | transient firing rate | excess | ARL1 on a real +1σ step |
|---|---|---|---|
| stationary negatives only | 94.4 % | +81.2 pp | 15.5 |
| transients in training | 90.8 % | +77.6 pp | 10.8 |

A −3.6 percentage-point improvement in the thing the extra supervision was for.
This is the one capability a learned detector has that an analytic one does not,
and measured, it buys very little.

---

## 8. Where the calibration stops being true

`validate_robustness.py` → `outputs/validate_robustness.txt`. **FAILED
CHECKS: 0.** 9 checks passed.

### 8.1 Autocorrelation, the limitation that matters most

AR(1) streams with the **same marginal distribution** `N(0,1)` as the
calibration stream, so anything that moves is caused by dependence alone.
Measured lag-one autocorrelations confirm the generator: −0.0037, +0.1969,
+0.3979, +0.5988, +0.7999, +0.9008.

ARL0 at the i.i.d.-calibrated thresholds, 4 seeds × 30 000 samples:

| φ | CUSUM | Page-Hinkley | EWMA | Windowed KS | ADWIN |
|---|---|---|---|---|---|
| 0.0 | 570 ± 42 | 469 ± 15 | 519 ± 35 | 463 ± 13 | 476 ± 24 |
| 0.2 | 164 ± 6 | 320 ± 9 | 171 ± 6 | 407 ± 8 | 235 ± 7 |
| 0.4 | 73 ± 2 | 231 ± 6 | 75 ± 2 | 350 ± 4 | 147 ± 3 |
| 0.6 | 41 ± 1 | 161 ± 3 | 39 ± 1 | 328 ± 2 | 106 ± 1 |
| 0.8 | 25 ± 0 | 102 ± 2 | 21 ± 0 | 317 ± 1 | 87 ± 1 |
| 0.9 | 20 ± 0 | 85 ± 1 | 15 ± 0 | 307 ± 0 | 81 ± 1 |

Degradation factor at φ = 0.9: **CUSUM 28.8x, EWMA 34.5x, ADWIN 5.9x,
Page-Hinkley 5.5x, windowed KS 1.5x.** A threshold from this package applied to
a real channel without recalibrating on that channel's own quiet data will
false-alarm far more often than its nominal rate.

The windowed KS test degrades least, and that is a property of its statistic
rather than a virtue: it compares empirical distributions, and the AR(1) stream
has the same marginal as the i.i.d. one. It is robust to the change in
dependence for exactly the reason it is blind to it.

### 8.2 ADWIN's `delta` is a very weak control

Measured ARL0, 3 seeds × 30 000 samples:

| `min_sub` | `1e-2` | `1e-4` | `1e-6` | `1e-8` | `1e-10` | `1e-12` |
|---|---|---|---|---|---|---|
| 5 | 45 | 76 | 125 | 218 | 372 | 600 |
| 15 | 90 | 145 | 238 | 375 | 604 | 1012 |
| 30 | 155 | 226 | 378 | 596 | 1034 | 1671 |
| 50 | 245 | 360 | 609 | 950 | 1392 | 2571 |

Ten orders of magnitude of `delta` buy a factor of **13.5** in ARL0 at
`min_sub = 5`, which is the `sqrt(log(1/delta))` scaling, measured. At
`min_sub = 5` the 500-sample target is crossed only between `delta = 1e-10` and
`1e-12`, and the calibration's bracket search from the shipped default of 0.002
reaches only `1e-8`, so it reports a bracketing **failure** (measured:
`delta = 9.861e-09`, held-out ARL0 227.2 ± 9.7, −54.6 % of target). At
`min_sub = 30` it brackets successfully (`delta = 9.942e-08`, ARL0 476.5 ± 24.5,
−4.7 %). `min_sub = 30` is the shipped default for that reason, declared before
any delay was measured.

Nominal `delta` against measured ARL0:

| `delta` | `1/delta` | measured ARL0 | ratio |
|---|---|---|---|
| 1e-2 | 100 | 156.4 | 1.56 |
| 1e-3 | 1 000 | 198.3 | 0.198 |
| 1e-4 | 10 000 | 238.1 | 0.0238 |
| 1e-6 | 1 000 000 | 384.7 | 0.0004 |

Reading `delta = 0.002` as "one false alarm per 500 samples" is wrong by a large
factor in the unsafe direction.

### 8.3 The harness reset convention does not disadvantage ADWIN

Published ADWIN drops the older part of its window on a cut; this harness resets
the whole window so all five analytic detectors are treated alike. The obvious
objection is that the harness convention is unfair. Measured, it is not:

| `delta` | full reset (harness) | shrink (published) | ratio |
|---|---|---|---|
| 1e-2 | 166.8 | 102.0 | 1.63x |
| 1e-4 | 249.6 | 161.1 | 1.55x |
| 1e-6 | 391.5 | 238.5 | 1.64x |
| 1e-8 | 640.9 | 367.8 | 1.74x |

The published convention gives a **shorter** ARL0 at every `delta` tested,
because after shrinking the detector immediately re-tests the window it kept.

---

## 9. The learned detector, and the honest negative

`validate_learned.py` → `outputs/validate_learned.txt`.
**FAILED CHECKS: 1.** 9 checks passed. The failure is the finding.

Architecture, data, split and metrics are in [MODEL_CARD.md](../MODEL_CARD.md) and
[DATASET_CARD.md](../DATASET_CARD.md). The numbers that belong here:

### 9.1 It loses at equal ARL0

| Change | Best analytic | Learned | Loss | Ratio | Significance |
|---|---|---|---|---|---|
| mean step +1.0 | EWMA 9.2 | 17.4 | +8.2 samples | 1.88x | +2.9 σ |
| mean step +0.5 | EWMA 31.4 | 45.5 | +14.1 samples | 1.45x | +3.7 σ |
| variance ×2.0 | CUSUM 12.5 | 16.5 | +3.9 samples | 1.31x | +2.2 σ |
| drift 0.02/sample | EWMA 33.2 | 34.0 | +0.8 samples | 1.02x | +0.8 σ |

Loses on three of four at more than two combined standard errors, ties on the
fourth, wins on none. **Not retuned.** The structural reason is in MODEL_CARD.md
section "Why the baseline wins".

### 9.2 It overfits at the window level — THE FAILED CHECK

| Split | ROC-AUC | Average precision | Brier | n | positive rate |
|---|---|---|---|---|---|
| train (8 seeds) | 0.9411 | 0.6342 | 0.09157 | 78 104 | 3.58 % |
| **held-out (6 disjoint seeds)** | **0.8229** | **0.3912** | 0.10382 | 58 578 | 3.58 % |

Declared expectation before measuring: a train-to-held-out ROC-AUC gap under
0.05. **Measured +0.1183. The check FAILS.** It is recorded in the committed
output, asserted against a documented band (0.08 to 0.16) so it cannot drift
silently, and the model was not retuned to close it — the stream-level result is
already a loss, and a smaller forest would only produce a prettier window-level
number for the same conclusion.

The split is by **seed**, not by row. A row split would leak badly: consecutive
windows overlap by 49 of 50 samples, so neighbouring rows are near-copies. The
gap is therefore capacity, not leakage.

Held-out average precision 0.391 against a no-skill rate of 0.036 is 10.9x
no-skill, which is a respectable window-level result. It does not translate into
a shorter detection delay at equal ARL0, and that gap between a good
classification metric and a bad streaming result is the single most transferable
thing in this repository.

### 9.3 Its confidence output is not calibrated

Reliability of the forest vote fraction on the held-out split:

| bin | n | mean score | observed frequency | gap |
|---|---|---|---|---|
| [0.1,0.2) | 19 467 | 0.1534 | 0.0090 | −0.1444 |
| [0.3,0.4) | 9 164 | 0.3444 | 0.0253 | −0.3191 |
| [0.5,0.6) | 3 070 | 0.5463 | 0.0612 | −0.4851 |
| [0.6,0.7) | 1 729 | 0.6430 | 0.1215 | **−0.5215** |
| [0.8,0.9) | 627 | 0.8467 | 0.5183 | −0.3284 |
| [0.9,1.0) | 367 | 0.9381 | 0.8093 | −0.1289 |

Worst absolute gap **0.522**. The score is used only as a monotone ordering
whose threshold is moved until the measured ARL0 matches a target, which
requires no calibration, and nothing in this repository calls it a probability.

### 9.4 Determinism and the batch-path identity

| Check | Result |
|---|---|
| training features regenerate bit-for-bit from the seeds | True |
| the refitted model produces identical held-out scores | True |
| batch `score_stream` against the online `update`, 4000 samples | worst absolute difference **0.000e+00** |
| batch and online alarm index sequences | identical, 28 alarms |

The last two matter because every learned ARL figure in this repository comes
from the batch path. If they diverged, those figures would describe a different
detector.

---

## 10. Cost

`validate_cost.py` → `outputs/validate_cost.txt`. **FAILED CHECKS: 0.** 5 checks
passed. Five repeats per measurement, reported as medians and ranges because
wall clock in this container moves 10-40 % between runs.

| Detector | median µs/sample | range | ratio to cheapest |
|---|---|---|---|
| CUSUM | 0.352 | 0.340–0.464 | 1.0x |
| Page-Hinkley | 0.396 | 0.384–0.460 | 1.1x |
| EWMA | 0.492 | 0.486–0.520 | 1.4x |
| ADWIN | 1.717 | 1.699–2.890 | 4.9x |
| Windowed KS | 8.910 | 8.606–10.705 | 25.3x |

CUSUM and Page-Hinkley swap places between runs; their medians differ by about
12 % and the run-to-run range of each is wider than that, so "the two cheapest"
is the only claim the measurement supports.

| Comparison | Result |
|---|---|
| this package's KS statistic against `scipy.stats.ks_2samp` | 43.2 µs against 1027.2 µs per evaluation, **23.8x** cheaper |
| learned detector online (one forest call per sample) | 7574 µs/sample, **21 538x** the cheapest analytic detector |
| learned detector batch score path | 6.802 µs/sample, **1113x** cheaper than online, bit-identical |
| `RandomForestClassifier.predict_proba` single row, `n_jobs=2` against `n_jobs=1` | **4.58x slower**, reproducing a defect an earlier session recorded as 5.7–8.9x |
| batch against single-window feature extraction | 3.202 µs against 304.9 µs per window, 95x |

The 7574 µs figure is dominated by scikit-learn's fixed per-call overhead on a
single row, not by traversing 100 shallow trees. The magnitude is API-specific;
what transfers is the direction and the order of magnitude. A detector needing a
model call per telemetry sample is in a different cost class from a three-line
recursion, and on a spacecraft processor that decides whether it runs at all.

---

## 11. Property tests

18 Hypothesis properties in `tests/test_properties.py`, all over the identities
that actually hold:

- CUSUM arms are non-negative; CUSUM is odd-symmetric; CUSUM is invariant under
  an affine rescaling of the channel when `mu0` and `sigma0` are set to match
  (the property that lets its threshold transfer between channels).
- EWMA's statistic stays inside the convex hull of its inputs; EWMA is
  odd-symmetric; its exact variance never exceeds the asymptotic one.
- Page-Hinkley's `m_min ≤ m ≤ m_max`; Page-Hinkley is translation-invariant.
- The KS statistic is bounded, symmetric in its arguments, equal to
  `scipy.stats.ks_2samp` to 1e-12, and invariant under a common increasing map.
- Batch and single-window features are bit-identical; features are always
  finite, including on a constant window where a naive autocorrelation gives
  0/0; the mean feature is translation-equivariant and the standard deviation
  invariant.
- The Wilson interval always contains its point estimate.
- The Siegmund ARL is positive, increases with `h` and decreases with the true
  shift.

**Two identities are pinned as deliberate NON-properties**, because they look
true and are not:

1. Page-Hinkley is **not** scale-invariant, unlike CUSUM. Its `lambda` does not
   transfer between channels of different variance.
2. **ARL0 is not the reciprocal of a per-sample false-alarm probability.** None
   of these detectors is memoryless; the measured coefficient of variation of
   the CUSUM run length is 0.6 to 1.3 and not exactly 1.

---

## 12. References

Each verified in this session from the source named. Where a source could not be
read, that is stated rather than guessed around.

| Reference | How verified |
|---|---|
| Page, E. S. (1954). "Continuous Inspection Schemes." *Biometrika* **41**(1-2), 100–115. DOI 10.1093/biomet/41.1-2.100 | Publisher page, `academic.oup.com`, read 2026-10-10: title, author, journal, volume, issue, year and page range all confirmed. |
| Hinkley, D. V. (1971). "Inference about the change-point from cumulative sum tests." *Biometrika* **58**(3), 509–523. DOI 10.1093/biomet/58.3.509 | Publisher page, `academic.oup.com`, read 2026-10-10: all fields confirmed. |
| Bifet, A. and Gavaldà, R. (2007). "Learning from Time-Changing Data with Adaptive Windowing." *Proc. 7th SIAM Int. Conf. on Data Mining*, 443–448. DOI 10.1137/1.9781611972771.42 | Title, authors, proceedings, publisher, year, DOI and page range from the IP Paris research-portal record, read 2026-10-10. **The SIAM-hosted PDF could not be read**: `epubs.siam.org` returns a robots-disallowed error and `doi.org` is blocked by this container's egress proxy. The **cut rule implemented** was transcribed from the authors' own technical report, "Adaptive Parameter-free Learning from Evolving Data Streams", Universitat Politècnica de Catalunya, read from `upcommons.upc.edu` 2026-10-10, section 4.1.1. |
| Roberts, S. W. (1959). "Control Chart Tests Based on Geometric Moving Averages." *Technometrics* **1**(3). DOI 10.1080/00401706.1959.10489860 | **Page range not verified.** `tandfonline.com` returned a client error and `doi.org` is blocked, so no page range is quoted anywhere in this repository. The EWMA recursion implemented is the standard one and is validated by hand arithmetic (section 2.3) rather than by the citation. |
| NIST/SEMATECH e-Handbook of Statistical Methods, section 6.3.2.3.1, "Cusum Average Run Length" | Read 2026-10-10 at `itl.nist.gov`. Table and the two-sided sentence quoted verbatim in section 2.1. |
| Siegmund's ARL approximation | Not read from a primary source. Used only as a cross-check, and **validated against the NIST table** to 0.62 % and 0.88 % (section 2.1), which is what licenses its use here. |
| `river` 0.26.1, `ruptures` 1.1.10 | PyPI JSON API, 2026-10-10, plus `pip download --no-deps` and module listing of each wheel. Claims in the alternatives table come from the files, not from memory. |
| PyPI name `telemdrift` | `curl -s -o /dev/null -w '%{http_code}' https://pypi.org/pypi/telemdrift/json` returned **404**; the same call for `numpy` returned 200 as a positive control. |

## 13. Safety statement

This software is research-grade. It is **not flight-qualified, not certified,
and not approved for operational aerospace use.** Every number in this document
was measured on synthetic streams this package generated itself, under an
independence assumption section 8.1 shows real telemetry violates. No real
spacecraft telemetry was used anywhere. An alarm from any detector here means a
statistic crossed a threshold, and section 7 shows the same alarm is raised by a
transient that resolves itself.
