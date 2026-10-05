# Validation evidence — telemetryool 0.1.0

Validation level 2 (research grade). Status: TESTING. Not flight-qualified, not
certified, not approved for operational aerospace use.

Every number in this file, in `README.md` and in `MODEL_CARD.md` was produced by
one of the five scripts in this directory, run in the session that wrote this
file. The raw stdout of each run is committed beside it as
`<script>_output.txt`, and the scripts write that file themselves, so the
committed output and a fresh run cannot drift apart.

## How to reproduce everything

```bash
cd validation
MPLBACKEND=Agg PYTHONPATH=../src python3 validate_persistence_trace.py
MPLBACKEND=Agg PYTHONPATH=../src python3 validate_far_design.py
MPLBACKEND=Agg PYTHONPATH=../src python3 validate_cusum_delay.py
MPLBACKEND=Agg PYTHONPATH=../src python3 validate_changepoint.py
MPLBACKEND=Agg PYTHONPATH=../src python3 validate_matched_far.py
```

All seeds are fixed in the scripts. Wall times on the single core that built
this repository, shared with four other build agents, over two runs:

| Script | Run A | Run B |
|---|---|---|
| `validate_persistence_trace.py` | 0.0 s | 0.0 s |
| `validate_far_design.py` | 26.3 s | 5.2 s |
| `validate_cusum_delay.py` | 19.4 s | 16.5 s |
| `validate_changepoint.py` | 5.4 s | 1.8 s |
| `validate_matched_far.py` | 47.7 s | 24.8 s |

The spread between the two columns is contention on the shared core, not
variation in the work: every script is seeded and the numbers it reports are
identical between runs. The committed `*_output.txt` files are from run B.

## Two kinds of result, and why the distinction matters

| Kind | What it is | Effect on exit status |
|---|---|---|
| **check** | A property this package must satisfy. | A failure is a defect here and makes the script exit non-zero. |
| **characterisation** | How far a *published approximation* departs from a higher-accuracy reference, against a band stated before the measurement. | Printed as `FAILED` and listed in the summary, but the exit status is unaffected: the package cannot fix someone else's closed form. |

No band in this file was widened after seeing a number. One characterisation is
out of band and is reported as such below and in the README.

## The operating-point currency

Everything is expressed as a **window false-alarm probability** `alpha_W`: the
probability that a detector, started from its reset state, raises at least one
alarm during one nominal monitoring window of `W` samples. One window is one
Bernoulli trial, so an estimate from `M` independent windows has the exact
binomial standard error `sqrt(alpha (1 - alpha) / M)`. A per-sample alarm rate
measured along a single long run does *not* have that property, because the
chart statistic is serially dependent, and no standard error quoted that way
would be meaningful.

Sample sizes follow from the precision needed, not the other way round. At
`alpha_W = 0.05`:

| windows M | binomial SE | relative SE |
|---|---|---|
| 1 900 | 0.005000 | 10.0 % |
| 20 000 | 0.0015411 | 3.08 % |
| 40 000 | 0.0010897 | 2.18 % |
| 160 000 | 0.0005449 | 1.09 % |

`telemetryool.calibration.windows_for_precision` inverts the relation.

---

## 1. Persistence and debounce reproduce a hand trace exactly

Script: `validate_persistence_trace.py` → `validate_persistence_trace_output.txt`
Tests: `tests/test_persistence.py` (the same two traces, as pytest assertions)

Two traces were worked out on paper from the update rule documented in
`OolChecker.update`, then compared sample by sample. Every compared quantity is
an integer or an enumeration member, so **the tolerance is zero**.

| Check | Reference | Result | Tolerance |
|---|---|---|---|
| Trace 1 (soft escalation, hard escalation, two-step de-escalation), 16 samples × 6 fields | hand trace in `tests/test_persistence.py` | 96 of 96 fields match | exact |
| Trace 2 (mode-dependent limits, invalid sample under `HOLD`, latch across a mode change), 10 samples × 6 fields | hand trace in `tests/test_persistence.py` | 60 of 60 fields match | exact |
| `InvalidPolicy.HOLD` lets a breach run continue across an invalid sample | hand trace: soft counts 1, 2, 2, 3 → `SOFT_ALARM` | reproduced | exact |
| `InvalidPolicy.RESET` breaks the run | hand trace: soft counts 1, 2, 0, 1 → `NOMINAL` | reproduced | exact |
| `InvalidPolicy.BREACH` raises `HARD_ALARM` on one invalid sample | fail-safe semantics | reproduced | exact |
| Clearing steps down one level per satisfied clear count | hand trace: `HARD, HARD, SOFT, SOFT, SOFT, NOMINAL, …` at `clear_persistence = 3` | reproduced | exact |

**All checks pass.** The hand traces are reproduced in full in the raw output,
with the hand value and the implementation value printed side by side in every
column, so a reader can audit any sample without running the code.

---

## 2. EWMA, CUSUM and limit-check false-alarm rates match their design values

Script: `validate_far_design.py` → `validate_far_design_output.txt`

Design method: the window false-alarm probability is computed exactly for the
discretised chain (Brook & Evans 1972 for CUSUM, Lucas & Saccucci 1990 for EWMA,
an exact persistence-counter chain for the limit check) and the threshold is
solved for by bisection. Measurement: 8 independent seeds × 20 000 windows =
**160 000 windows per chart**, nominal data iid standard normal, `W = 100`.

Agreement band, stated before the measurement: `|z| < 3.5` pooled standard
errors.

| Chart | Designed threshold | ARL0 (samples) | Design `alpha_W` | Measured `alpha_W` | Pooled SE | z | 95 % Wilson |
|---|---|---|---|---|---|---|---|
| CUSUM `k = 0.5` | `h = 6.351522849` | 1802.80 | 0.0500000 | **0.0495688** | 0.0005449 | **−0.79** | [0.0485160, 0.0506431] |
| EWMA `lam = 0.2` | `L = 3.374959514` | 1864.55 | 0.0500000 | **0.0498313** | 0.0005449 | **−0.31** | [0.0487758, 0.0509083] |
| Limit check, persistence 3 | `L = 1.734259866` | 1914.69 | 0.0500000 | **0.0506000** | 0.0005449 | **+1.10** | [0.0495368, 0.0516848] |

**All three pass.** The relative differences are −0.86 %, −0.34 % and +1.20 %.

Seed-to-seed spread against the binomial prediction (8 seeds, so the sample
standard deviation has a relative standard error of 0.267 and a ratio in
[0.47, 1.53] is consistent at 2 sigma):

| Chart | Observed sd of the 8 rates | Binomial SE | Ratio |
|---|---|---|---|
| CUSUM | 0.0009300 | 0.0015411 | 0.603 |
| EWMA | 0.0011765 | 0.0015411 | 0.763 |
| Limit check | 0.0012192 | 0.0015411 | 0.791 |

All three pass. The ratios are below 1, which is the direction a *correlated*
set of seeds would not produce; nothing here suggests the windows are dependent.

### Where the design stops working: serial correlation

The designs assume independent samples. Measured `alpha_W` with the **same
designed thresholds** on AR(1) nominal data, 20 000 windows per cell:

| rho | CUSUM `k=0.5` | EWMA `lam=0.2` | Limit check, persistence 3 |
|---|---|---|---|
| 0.0 | 0.048550 (0.97×) | 0.047400 (0.95×) | 0.049150 (0.98×) |
| 0.3 | 0.364250 (**7.3×**) | 0.318700 (**6.4×**) | 0.116900 (**2.3×**) |
| 0.6 | 0.809600 (**16.2×**) | 0.742600 (**14.9×**) | 0.402550 (**8.1×**) |
| 0.9 | 0.949550 (**19.0×**) | 0.903350 (**18.1×**) | 0.632150 (**12.6×**) |

This is reported, not checked. It is the single most important limitation of the
package: **a designed false-alarm rate is a property of the nominal model, and
housekeeping telemetry is usually not white.** A user whose channel has lag-1
autocorrelation of 0.3 and who takes the designed threshold on trust will get
roughly seven times the false alarms they asked for. The remedy — prewhitening
the channel, or calibrating the threshold empirically on the channel's own
nominal history using `telemetryool.calibration.calibrate_threshold` — is not
implemented as an automatic step and is left to the user.

---

## 3. Detection delay matches the analytic CUSUM expectation

Script: `validate_cusum_delay.py` → `validate_cusum_delay_output.txt`

The analytic expectation is derived in the script's module docstring. Two
independent evaluations are used: the **Brook & Evans (1972)** Markov-chain
discretisation (exact up to `O(1/n_states)`) and the **Siegmund (1985)** closed
form `ARL = (exp(−2 Δ b) + 2 Δ b − 1) / (2 Δ²)` with `b = h + 1.166`, reproduced
in Montgomery (2013) chapter 9.

Chart: `k = 0.5`, `h = 6.351522849` (the threshold designed in section 2).
Monte Carlo: 4 000 independent runs per row, one-sided upper arm, no
competing-risks approximation anywhere in this table. Agreement band, stated
before the measurement: `|z| < 3.5` standard errors of the measured mean.

| Shift δ [sigma] | Measured mean run length [samples] | SE | Brook-Evans ARL | z | Censored runs |
|---|---|---|---|---|---|
| 0.0 | **3617.06** | 57.48 | 3627.51 | **−0.18** | 0 |
| 0.5 | **56.5815** | 0.7303 | 56.4740 | **+0.15** | 0 |
| 1.0 | **13.1092** | 0.1002 | 13.0740 | **+0.35** | 0 |
| 1.5 | **7.1387** | 0.0409 | 7.0984 | **+0.99** | 0 |
| 2.0 | **4.8982** | 0.0224 | 4.9103 | **−0.54** | 0 |
| 3.0 | **3.1320** | 0.0113 | 3.1196 | **+1.10** | 0 |

**All six pass**, with no censored runs anywhere, so none of the means is biased
downwards by truncation.

Brook-Evans discretisation convergence, at `δ = 1.0`, against `n_states = 6400`:

| `n_states` | ARL | Relative error |
|---|---|---|
| 50 | 13.185486 | +8.54e-03 |
| 100 | 13.129051 | +4.22e-03 |
| 200 | 13.100946 | +2.07e-03 |
| 400 | 13.074016 | −4.33e-07 |
| 800 | 13.074011 | −7.99e-07 |
| 1600 | 13.074019 | −2.36e-07 |

The package default of 400 states has a relative discretisation error of
**4.33e-07** against a 6400-state reference. **Check passes.**

### The one check that FAILED

| Characterisation | Band stated beforehand | Measured | Verdict |
|---|---|---|---|
| Siegmund closed form within 2 % of Brook-Evans over `δ ∈ [0, 3]` | 2 % relative | worst **6.17 %** | **FAILED** |

Per-shift detail:

| δ | Brook-Evans | Siegmund | Relative difference |
|---|---|---|---|
| 0.0 | 3627.50774 | 3662.97231 | 9.78e-03 |
| 0.5 | 56.47398 | 56.51315 | 6.94e-04 |
| 1.0 | 13.07402 | 13.03613 | 2.90e-03 |
| 1.5 | 7.09839 | 7.01752 | 1.14e-02 |
| 2.0 | 4.91025 | 4.78946 | **2.46e-02** |
| 3.0 | 3.11958 | 2.92701 | **6.17e-02** |

The band is **not** widened to make this pass. Siegmund's derivation is
asymptotic in large `h` and rests on a continuous barrier crossing; at a 3-sigma
shift the run length is about three samples and the continuum argument does not
hold. The honest statement is: the Siegmund approximation is within 1.2 % of the
chain for `δ ≤ 1.5` and degrades to 6.2 % at `δ = 3`. Use
`cusum_arl_markov` when the number matters; `cusum_arl_siegmund` is provided
because it is the formula most practitioners will recognise, and its error is
now measured rather than assumed.

### The two-sided competing-risks approximation

The package combines the two arms as `1/ARL = 1/ARL+ + 1/ARL−`, which treats
them as independent when they share an observation. Measured against 4 000
two-sided runs:

| δ | Measured | SE | Combined analytic | z | Relative difference |
|---|---|---|---|---|---|
| 0.0 | 1826.06 | 29.02 | 1813.75 | +0.42 | +6.78e-03 |
| 1.0 | 12.9550 | 0.0981 | 13.0740 | −1.21 | −9.10e-03 |
| 2.0 | 4.9600 | 0.0229 | 4.9103 | +2.17 | +1.01e-02 |

All three pass at `|z| < 3.5`. The approximation error is about 1 % and the
`δ = 2.0` row is already at 2.17 sigma, so at larger Monte-Carlo sizes this
would become a visible bias rather than noise. It is a 1 % effect and is
documented as such.

### Warm-chart delay, the way an operator sees it

The rows above start the chart at the same instant as the shift. In a monitoring
window the chart is already running, so a mid-window onset is detected faster
than `ARL1`. Measured over 4 000 windows of length 200 with the step at sample
100:

| δ | Pd within the remaining 100 samples | Mean delay | SE | `ARL1` (Brook-Evans) | Ratio |
|---|---|---|---|---|---|
| 0.5 | 0.8273 | 39.3841 | 0.4265 | 56.4720 | 0.697 |
| 1.0 | 0.9475 | 11.3900 | 0.1040 | 13.0740 | 0.871 |
| 1.5 | 0.9565 | 5.6226 | 0.0420 | 7.0984 | 0.792 |
| 2.0 | 0.9523 | 3.5345 | 0.0245 | 4.9103 | 0.720 |
| 3.0 | 0.9440 | 1.9266 | 0.0128 | 3.1196 | 0.618 |

This is why every detection-delay table in this package states the onset index.

---

## 4. Change-point detection

Script: `validate_changepoint.py` → `validate_changepoint_output.txt`

Statistic: the maximised standardised mean-shift statistic (Hinkley 1970).
Threshold: a Monte-Carlo quantile under the no-change hypothesis, 40 000
replicates, segment length 200. Measurement: 40 000 independent replicates.

| Check | Reference | Result | Tolerance |
|---|---|---|---|
| Per-segment false-alarm rate on independent data | target `alpha = 0.05` | **0.0505500 ± 0.0010954**, 2 022 of 40 000, z = **+0.36** | `\|z\| < 3.5` combined SE (0.0015411) |
| Threshold increases with segment length at every alpha | monotonicity of the null distribution | passes at n ∈ {50, 100, 200, 400} | exact ordering |
| Threshold increases as alpha tightens | monotonicity in the quantile | passes at every n | exact ordering |

Calibrated threshold at n = 200, alpha = 0.05: **3.147854662** (dimensionless).

Localisation of the maximiser, 1 000 seeds per shift size, true change point at
index 120 of 200 samples:

| Shift [sigma] | Exact | \|e\| ≤ 1 | \|e\| ≤ 2 | \|e\| ≤ 5 | Median \|e\| | p90 \|e\| | Max \|e\| | Detection prob. |
|---|---|---|---|---|---|---|---|---|
| 0.5 | 0.086 | 0.182 | 0.263 | 0.415 | 8.0 | 51.0 | 119 | 0.757 |
| 1.0 | 0.267 | 0.480 | 0.597 | 0.809 | 2.0 | 10.0 | 43 | 1.000 |
| 2.0 | 0.647 | 0.863 | 0.938 | 0.990 | 0.0 | 2.0 | 9 | 1.000 |
| 3.0 | 0.838 | 0.979 | 0.998 | 1.000 | 0.0 | 1.0 | 3 | 1.000 |
| 5.0 | 0.980 | 1.000 | 1.000 | 1.000 | 0.0 | 0.0 | 1 | 1.000 |

Binomial SE on each fraction at 1 000 trials, worst case: 0.0158. Checks on the
2, 3 and 5 sigma rows (within 2 samples at least 90 %, 95 % and 99 % of the
time) all pass. The 0.5 sigma row is reported and **not** checked: at that shift
size over 200 samples the statistic barely leaves its null distribution, and a
detector that localised it reliably would be extracting information the data do
not contain.

Series-wide false-alarm rate of binary segmentation, 4 000 pure-noise series:

| `min_segment` | `max_depth` | P(any false change point) | SE | Mean count | Inflation over the per-segment 0.05 |
|---|---|---|---|---|---|
| 20 | 20 | 0.05150 | 0.00349 | 0.0540 | **1.03×** |
| 20 | 2 | 0.05150 | 0.00349 | 0.0540 | 1.03× |
| 50 | 20 | 0.05150 | 0.00349 | 0.0527 | 1.03× |

Reported, not checked. The inflation is small because the top-level test gates
the recursion: a series that produces no first change point never reaches a
second level.

---

## 5. Matched-false-alarm-rate comparison, with confusion matrices in full

Script: `validate_matched_far.py` → `validate_matched_far_output.txt`

Configuration: 4 channels, equicorrelation 0.6, no serial correlation,
`W = 100`, debounce 1, anomaly onset at sample 40, target `alpha_W = 0.05`.
Training 1 500 nominal windows, calibration 10 000 nominal windows, measurement
20 000 **independent** nominal windows, 2 000 windows per anomaly scenario,
base seed 20261005.

Every detector is reduced to one scalar score per sample; its threshold is set
so that exactly `ceil(0.05 × 10000) = 500` calibration windows alarm; and the
delivered rate is then measured on the third, independent block. The threshold
is itself an order statistic of the calibration sample, so the delivered rate
carries both errors: `sqrt(alpha (1 − alpha) (1/10000 + 1/20000)) = 0.0026693`.
That combined figure, not the measurement binomial error alone, is what the
agreement band uses.

### Did the match hold?

| Method | Calibrated threshold | In-sample cal. rate | Delivered `alpha_W` | Binomial SE | Combined SE | z | 95 % Wilson |
|---|---|---|---|---|---|---|---|
| `ool(p=1)` | 3.821037 | 0.05000 | 0.04675 | 0.00149 | 0.00267 | −1.22 | [0.04391, 0.04976] |
| `ewma(lam=0.2)` | 3.736748 | 0.05000 | 0.05210 | 0.00157 | 0.00267 | +0.79 | [0.04911, 0.05527] |
| `cusum(k=0.5)` | 7.582248 | 0.05000 | 0.05105 | 0.00156 | 0.00267 | +0.39 | [0.04808, 0.05419] |
| `t2q(lam=0.2)` | 1.632697 | 0.05000 | 0.04670 | 0.00149 | 0.00267 | −1.24 | [0.04386, 0.04971] |
| `gmm4(lam=0.2)` | 18.790808 | 0.05000 | 0.04855 | 0.00152 | 0.00267 | −0.54 | [0.04566, 0.05162] |
| `iforest60(lam=0.2)` | 0.680150 | 0.05000 | 0.05615 | 0.00163 | 0.00267 | +2.30 | [0.05304, 0.05943] |

**All six pass** the `|z| < 3.5` band. The spread of delivered rates is
0.04670 to 0.05615, i.e. the comparison below is at a false-alarm rate matched
to within about ±12 % relative — which is the honest statement of how tightly
"matched" means matched at this sample size.

Score-granularity diagnostic over the first 4 000 calibration windows: five of
the six detectors have 4 000 distinct window trigger levels with a largest tie
group of 1; `iforest60` has 3 988 distinct levels with a largest tie group of 3.
Calibration granularity is therefore not the limiting factor at this sample
size, and the residual discrepancies above are ordinary sampling error of the
threshold and of the measurement.

### Detection probability and mean delay at the matched operating point

Pd is the probability of an alarm at or after the onset within the window; the
delay is conditional on detection; `early` counts windows that alarmed *before*
the onset, which are excluded from the delay and are false alarms that landed in
an anomalous window. AUC is the window-level ROC area against the same 20 000
nominal windows.

**Scenario `step_1.0sigma`** (step of 1.0 sigma on channel 0 at sample 40):

| Method | Pd | SE | Mean delay | Median | p10 | p90 | Early | AUC |
|---|---|---|---|---|---|---|---|---|
| `ool(p=1)` | 0.1485 | 0.0080 | 29.61 | 29.0 | 7.0 | 53.0 | 35 | 0.7184 |
| `ewma(lam=0.2)` | 0.9695 | 0.0038 | 16.86 | 14.0 | 6.0 | 33.0 | 35 | 0.9962 |
| `cusum(k=0.5)` | 0.9875 | 0.0025 | 13.63 | 12.0 | 6.0 | 23.0 | 24 | **1.0000** |
| `t2q(lam=0.2)` | 0.9800 | 0.0031 | 14.62 | 13.0 | 6.0 | 27.0 | 34 | 0.9985 |
| `gmm4(lam=0.2)` | 0.9855 | 0.0027 | **11.90** | 10.0 | 5.0 | 21.0 | 28 | 0.9995 |
| `iforest60(lam=0.2)` | 0.0415 | 0.0045 | 31.46 | 34.0 | 4.0 | 53.0 | 41 | 0.6348 |

**Scenario `step_0.5sigma`:**

| Method | Pd | SE | Mean delay | AUC |
|---|---|---|---|---|
| `ool(p=1)` | 0.0410 | 0.0044 | 28.70 | 0.5482 |
| `ewma(lam=0.2)` | 0.3125 | 0.0104 | 29.60 | 0.8012 |
| `cusum(k=0.5)` | **0.5510** | 0.0111 | 31.41 | 0.8611 |
| `t2q(lam=0.2)` | 0.3830 | 0.0109 | 31.21 | 0.8551 |
| `gmm4(lam=0.2)` | 0.4130 | 0.0110 | 30.36 | **0.8639** |
| `iforest60(lam=0.2)` | 0.0395 | 0.0044 | 27.22 | 0.5650 |

**Scenario `drift_0.05sigma_per_sample`:**

| Method | Pd | SE | Mean delay | AUC |
|---|---|---|---|---|
| `ool(p=1)` | 0.8835 | 0.0072 | 44.94 | 0.9831 |
| `ewma(lam=0.2)` | 0.9780 | 0.0033 | 25.52 | 1.0000 |
| `cusum(k=0.5)` | 0.9790 | 0.0032 | 24.43 | 1.0000 |
| `t2q(lam=0.2)` | 0.9815 | 0.0030 | 24.58 | 1.0000 |
| `gmm4(lam=0.2)` | 0.9800 | 0.0031 | **22.89** | 1.0000 |
| `iforest60(lam=0.2)` | 0.0430 | 0.0045 | 35.50 | 0.6716 |

**Scenario `stuck_channel0`** (channel 0 frozen at its sample-40 value):

| Method | Pd | SE | Mean delay | AUC |
|---|---|---|---|---|
| `ool(p=1)` | 0.0235 | 0.0034 | 28.13 | **0.4677** (below chance) |
| `ewma(lam=0.2)` | 0.2370 | 0.0095 | 8.94 | 0.6279 |
| `cusum(k=0.5)` | 0.5430 | 0.0111 | 15.74 | 0.7666 |
| `t2q(lam=0.2)` | 0.6585 | 0.0106 | 15.65 | 0.8838 |
| `gmm4(lam=0.2)` | **0.6600** | 0.0106 | 15.74 | **0.8874** |
| `iforest60(lam=0.2)` | 0.0155 | 0.0028 | 28.48 | 0.5154 |

**Scenario `decorrelate_ch01`** (channels 0 and 1 replaced by independent
unit-variance noise from sample 40; every marginal distribution is unchanged):

| Method | Pd | SE | Mean delay | AUC |
|---|---|---|---|---|
| `ool(p=1)` | 0.0320 | 0.0039 | 30.52 | 0.5144 |
| `ewma(lam=0.2)` | 0.0355 | 0.0041 | 29.94 | 0.5386 |
| `cusum(k=0.5)` | 0.0345 | 0.0041 | 30.58 | 0.5374 |
| `t2q(lam=0.2)` | 0.9240 | 0.0059 | 18.28 | 0.9888 |
| `gmm4(lam=0.2)` | **0.9340** | 0.0056 | **17.46** | **0.9893** |
| `iforest60(lam=0.2)` | 0.0000 | 0.0000 | — | **0.4253** (below chance) |

### Full confusion matrices

Every one of the 30 method × scenario confusion matrices is printed in full in
`validate_matched_far_output.txt`, with all four counts and the derived rates
(tpr, fpr, precision, specificity, f1, accuracy) — not summarised to a single
score. Negatives are the shared 20 000-window nominal measurement set, so each
method's false-positive column is the same across scenarios. One example, the
best and the worst method on the decorrelation scenario:

```
Confusion matrix: method='gmm4(lam=0.2)' scenario='decorrelate_ch01'
                    predicted alarm   predicted nominal        total
  actual anomaly               1904                  96         2000
  actual nominal                971               19029        20000
  total                        2875               19125        22000
  tpr=0.9520 fpr=0.0486 precision=0.6623 specificity=0.9515 f1=0.7811 accuracy=0.9515

Confusion matrix: method='iforest60(lam=0.2)' scenario='decorrelate_ch01'
                    predicted alarm   predicted nominal        total
  actual anomaly                 40                1960         2000
  actual nominal               1123               18877        20000
  total                        1163               20837        22000
  tpr=0.0200 fpr=0.0561 precision=0.0344 specificity=0.9438 f1=0.0253 accuracy=0.8599
```

The precision column is the one worth reading: even the best method on its best
scenario flags 2 875 windows of which 971 are nominal, because 20 000 nominal
windows at a 5 % false-alarm rate produce a thousand false alarms whatever the
detector does. That is the arithmetic of monitoring, and it does not improve
with a better detector — it improves with a lower `alpha_W`, at the cost of
detection delay.

### Textbook expectations that must hold if the harness is correct

| Check | Result |
|---|---|
| CUSUM beats a plain limit check on a sustained 1.0 sigma step at matched FAR (Page 1954) | **PASS** — CUSUM Pd 0.9875, delay 13.63; limit check Pd 0.1485, delay 29.61 |
| No univariate method detects a pure decorrelation (Pd < 0.2 for all three) | **PASS** — 0.0320, 0.0355, 0.0345 |
| At least one multivariate method detects the decorrelation (Pd > 0.5) | **PASS** — `t2q` 0.9240, `gmm4` 0.9340 |
| Every method is at chance or better on the step scenario (AUC ≥ 0.5) | **PASS** — minimum 0.6348 |

### Methods scoring below chance, reported as measured

| Method | Scenario | AUC |
|---|---|---|
| `ool(p=1)` | `stuck_channel0` | 0.4677 |
| `iforest60(lam=0.2)` | `decorrelate_ch01` | 0.4253 |

An AUC below 0.5 means the score is anti-correlated with the anomaly on that
scenario. For the limit check on a stuck channel this is mechanical: a frozen
value is one sample's worth of noise held constant, and a held value is on
average closer to the mean than a fresh draw, so the limit check sees *less*
excursion than nominal. For the Isolation Forest on a decorrelation it is a real
failure of the model: its axis-aligned splits are fitted to the marginal
distributions, which the decorrelation leaves untouched. Both are left in the
table.

---

## Summary of failed and out-of-band results

| Item | Status | Where |
|---|---|---|
| Siegmund CUSUM ARL within 2 % of Brook-Evans over δ ∈ [0, 3] | **FAILED** (6.17 % at δ = 3) | section 3 |
| Isolation Forest is useful at all on these scenarios | **No** — Pd 0.016 to 0.043, AUC 0.425 to 0.672, below chance on one scenario | section 5 |
| Designed `alpha_W` holds under serial correlation | **No** — 7.3× at rho = 0.3 | section 2 |
| Limit check on a stuck channel | **Below chance**, AUC 0.4677 | section 5 |
| Nothing was cut from the Level 2 requirement list | — | — |

Nothing in this file was tuned after the fact. Where a band was exceeded the
measured number is printed and the band is left where it was stated.
