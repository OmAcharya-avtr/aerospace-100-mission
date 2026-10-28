# Validation

**Validation level 2.** Every number below was produced by a script in this
directory, executed on 2026-10-28, and every script writes its raw output
beside itself as `<script>_output.txt`. No figure here is carried from an
earlier session, estimated, or rounded by hand. Where a check failed or a
result is unfavourable to this package, it is in the table with the others.

Environment, as recorded in the header of every output file: Python 3.13.18,
numpy 2.5.3, scikit-learn 1.9.1, matplotlib 3.11.2, scipy 1.18.1, pandas 3.0.5,
pytest 9.1.1, hypothesis 6.170.9, ruff 0.16.9, two CPU cores. `xarray` is named
in the README's alternatives table and is **not installed**; no number in this
repository is compared against it.

## What was run

| script | what it measures | checks | failed |
|---|---|---|---|
| `validate_rule_screens.py` | the four timing and ordering screens over a swept grid, and their false positives on clean records | 6 | 0 |
| `validate_classification.py` | per-class scores of both rule baselines, the ceiling on any range screen, the worth of condition (7b), and the cost of withholding a field | 13 | 0 |
| `validate_interpolation_policy.py` | equation (3) as a theorem, equation (3') as a coverage statement, and what three gates are worth | 11 | 0 |
| `validate_refusal_controls.py` | fifteen controls, each anchored on a truth and verified to fire | 16 | 0 |
| `validate_learned.py` | three learned configurations against both rule baselines and two majority guesses | 5 | 0 |
| `validate_cli.py` | every command line entry point, in a subprocess, and the tree afterwards | 15 | 0 |
| `worked_example.py` | the worked example quoted in `README.md` | 3 | 0 |
| `probe_working_directories.py` | every script and example from five working directories | 1 | 0 |

Test suite: **334 tests, 334 passed, 0 failed, 0 skipped, 0 errors**, counted
from the `testsuite` element of a junit XML written outside the repository and
cross-checked against a count of `testcase` elements.

## 1. The rule screens

`validate_rule_screens.py`. A grid of four anomaly-rate multipliers against
four cadence-jitter fractions, 3,000 frames per cell, swept twice.

| check | reference | result | tolerance |
|---|---|---|---|
| gap runs matched, no clock reset | injected deletion runs, by length | recall **1.0000** on every one of 16 cells | exact match of the frame deficit |
| duplicates found, no clock reset | injected duplicate frames, by index | recall **1.0000**, precision 1.0000 | exact index |
| reorders found, no clock reset | injected reorders, by index | recall **1.0000**, precision 1.0000 | exact index |
| clock resets found | injected resets, by index | recall **1.0000** | exact index |
| gap recall with two clock resets per record | the same injected runs | mean **0.9602**, minimum **0.6667** | reported, not bounded |
| timing false positives on a clean record | 4 cells with nothing injected | **0** gaps, duplicates, reorders or resets | exactly zero |
| cadence screen on a clean record | the 2a rule, below | **0** findings where `2a <= jitter_s`; 115 and 697 findings where it is not | exactly zero on compliant cells |

**Two results worth reading.** A clock reset moves part of a record onto a
different time base, so the time-sorted view interleaves frames from before and
after it; gap recall falls from 1.0000 to a mean of 0.9602 and a worst cell of
0.6667. That degradation is a property of the problem, not of the screen, and
it is why the `time_source` determinability limit exists.

And `RecordSchema.jitter_s` is a tolerance on an **interval**, while a
per-sample timestamp jitter of amplitude `a` produces interval departures of up
to `2a`. A record whose timestamps carry jitter `a` must declare
`jitter_s >= 2a`. Measured: at `a = 0.08` cadences against a declared 0.25 the
cadence screen is silent; at 0.16 and 0.24 it fires 115 and 697 times on a
record with nothing injected.

## 2. The three-way classification

`validate_classification.py`, 12,000 frames of the held-out split, 45,500
frame-channel rows.

| class | baseline | precision | recall | surfaced | undecided rate |
|---|---|---|---|---|---|
| corrupt | specified rules | 0.9375 | 0.1961 | 0.6928 | 0.4967 |
| corrupt | strengthened rules | **0.9688** | 0.4052 | **0.7647** | 0.3595 |
| misparsed | specified rules | 0.9672 | 0.7284 | 0.7778 | 0.0494 |
| misparsed | strengthened rules | 0.9437 | **0.9321** | **0.9815** | 0.0494 |
| clean | strengthened rules | 0.9994 | 0.9936 | 1.0000 | 0.0064 |

The local-residual screen of equation (5) is the only difference between the
two baselines and is worth **+0.2092 corrupt recall**.

**Why the corrupt recall is not higher, with the number.** Enumerated over all
65,536 word codes of each channel, §1b of the same output:

| channel | codes decoding in range | single-bit flips staying in range | max achievable range-screen recall |
|---|---|---|---|
| battery_voltage | 0.381485 | **0.862371** | 0.137629 |
| valve_current | 0.122086 | 0.803993 | 0.196007 |
| gyro_rate | 0.183121 | 0.763676 | 0.236324 |
| tank_pressure | 0.244156 | 0.852486 | 0.147514 |

A declared-range screen cannot see any of those, and the worst-channel figure
is an upper bound on what it must miss. It is reported as the
`in_range_corruption` determinability limit by every call, and the agreement
between the enumeration and the limit is itself a check.

**What condition (7b) of the reinterpretation test is worth**, enumerated over
every word code with the local median at the range midpoint and the robust
sigma at 2 per cent of the span:

| channel | rescued by (7a) alone | rescued by (7a) and (7b) | ratio |
|---|---|---|---|
| battery_voltage | 0.619171 | 0.123856 | 5.00 |
| valve_current | 1.000000 | 0.230179 | 4.34 |
| gyro_rate | 1.000000 | 0.314270 | 3.18 |
| tank_pressure | 0.447479 | 0.080322 | 5.57 |

A damaged word lands inside a declared range by accident often enough that
condition (7a) on its own is not evidence of anything.

**The measured cost of withholding a field**, same record, four variants:

| record | corrupt verdicts | misparsed verdicts | undecided verdicts | corrupt recall | misparsed recall |
|---|---|---|---|---|---|
| complete | 161 | 186 | 352 | 0.4052 | 0.9321 |
| no validity flag | **0** | 210 | 68 | **0.0000** | 0.9383 |
| no raw words | 161 | 26 | 512 | 0.4052 | **0.0000** |
| neither | 0 | 26 | 252 | 0.0000 | 0.0000 |

Each zero is what the corresponding determinability limit asserts in prose, and
it is measured rather than claimed.

## 3. The gap policy

`validate_interpolation_policy.py`.

### Equation (3) as a theorem

Linear interpolation evaluated against the **noiseless** truth, so the only
error present is the interpolation error. Eight gap lengths, four channels, 400
start positions each: **12,800 chord-versus-truth comparisons, 0 violations in
32 cells.**

| check | reference | result | tolerance |
|---|---|---|---|
| measured error against `M2 h^2 / 8` | analytic `M2` from equation (12) | 0 of 32 cells violated | `eps * (4 * bound + amplitude)`, 2.8e-16 to 1.8e-14 |
| measured error against the Bernstein bound | equation (8), Boas 1954 Thm 11.1.2 | 0 of 32 cells violated | same form |

The Bernstein bound is the independent check. Equation (3) is a Taylor
remainder on a real interval; equation (8) is a complex-analytic bound on an
entire function of exponential type. They share no mathematics, which is the
point.

### Equation (3') as a coverage statement, and its measured failure rate

Real records, real noisy endpoints, eight seeds, 2,500 frames each.

| check | reference | result | tolerance |
|---|---|---|---|
| realised error against the policy's prediction | the noiseless truth on the generation grid | **1 exceedance in 16 channel-records, rate 0.062500** | none; the rate is the measurement |
| every exceedance explained | ground-truth class of the gap's endpoints | **1 of 1** traced to an undetected endpoint anomaly | exact |

**This is the product's main unfavourable result and it is published as
measured.** The one exceedance is seed 1, `battery_voltage`: frame 1069 carries
a bit error that left the value inside its declared range and passed the frame
error-control check, so no screen flagged it, so the endpoint gate could not
refuse the gap it anchors, and the chord is drawn from a wrong value. The
policy's prediction is sound; the classification feeding its endpoint gate is
not complete, and the two numbers multiply. Declaring a tighter range, or
supplying per-word rather than per-frame error control, is what removes it.
Nothing in this package can.

### What three gates are worth

| gate | configuration | result |
|---|---|---|
| curvature resolvability | disabled | the near-Nyquist channel exceeds its prediction on **8 of 8 seeds** |
| curvature resolvability | enabled (default) | the channel is refused on **8 of 8 seeds**, 0 permitted gaps |
| endpoint integrity | blinded by emptying the anomaly list | **3 of 8 seeds** exceed their prediction |
| endpoint integrity | enabled (default) | 1 of 8 seeds exceeds, for the reason above |

### The curvature estimator, and the channel it cannot see

| channel | analytic eq (12) | Bernstein eq (8) | estimated eq (9) | noise floor | measurable | hf fraction | resolvable | est/analytic |
|---|---|---|---|---|---|---|---|---|
| battery_voltage | 8.67779e-4 | 3.26025e-3 | — (declared) | 0 | yes | 0 | yes | — |
| valve_current | 9.8696e-7 | 9.8696e-7 | 9.51981e-4 | 9.62496e-4 | **NO** | 0.874 | NO | — |
| gyro_rate | 0.387163 | 2.32133 | 1.90974e-3 | 2.40624e-4 | yes | 0.039487 | **NO** | **0.004933** |
| tank_pressure | 6.09235e-7 | 6.09235e-7 | — (declared) | 0 | yes | 0 | yes | — |

On the **noiseless** channels the estimator recovers the analytic bound to
0.983x, 0.999x and 1.000x; on the near-Nyquist channel it returns **0.00393x**.
The reason is closed form: the quadratic-fit second-derivative filter has a
transfer of **0.00326383** at `f T = 0.35` and a half-power frequency at
`f T = 0.0640399`. The estimate is therefore **biased, not noisy**, the
measurability gate passes it, and only the resolvability gate catches it.

### The noise-scale estimator of equation (16)

| channel | true sigma | eq (16) estimate | ratio | top frequency |
|---|---|---|---|---|
| battery_voltage | 0.004 | 0.00411583 | 1.0290 | 0.0073 Hz |
| valve_current | 0.012 | 0.0120872 | 1.0073 | 0.0025 Hz |
| gyro_rate | 0.003 | 0.111369 | **37.1232** | 0.35 Hz |
| tank_pressure | 0.002 | 0.00211844 | 1.0592 | 0.00056 Hz |

A second difference cannot tell a fast sinusoid from noise. This is why the
policy prefers a declared `noise_sigma`.

## 4. Controls

`validate_refusal_controls.py`. **Fifteen controls, eleven of which must fire
and eleven of which fired; four of which must not fire and none of which did.**
Every control is anchored on a quantity true independently of anything this
package computed: an analytic bound from equation (12), a declared noise sigma,
a ground-truth class from the generator, or constructed counter arithmetic.

| # | control | anchored on | direction | outcome |
|---|---|---|---|---|
| 1 | long outage refused | a constructed 21,600-interval outage | must fire | 4 refusals, 0 permissions |
| 2 | tolerance below the analytic prediction refused | equation (12) over `h = 3 s` | must fire | 1 of 1 refused |
| 3 | tolerance above it permitted | the same anchor, doubled | must **not** fire | 0 of 1 refused |
| 4 | gap beyond `max_gap_slots` refused | a constructed 5-interval gap | must fire | `nyquist_slots` |
| 5 | gap inside it permitted | a constructed 2-interval gap | must **not** fire | 0 refused |
| 6 | tolerance below `k sigma` refused | a declared sigma of 0.5 | must fire | `tolerance_above_noise` |
| 7 | gap bounded by a failed frame refused | a declared error-control flag | must fire | `endpoints_clean` |
| 8 | the same gap with clean endpoints permitted | the identical record | must **not** fire | 0 refused |
| 9 | curvature below its own noise floor refuses linear | a constructed pure-noise channel | must fire | estimate 7.46e-4, floor 8.02e-4 |
| 10 | content above the estimator bandwidth refuses linear | a constructed 0.35 f_s sinusoid | must fire | hf fraction 0.999247 |
| 11 | withholding raw words empties the misparsed class | the generator's misparse truth | must fire | 105 value-level verdicts to 0 |
| 12 | withholding validity empties the corrupt class | the generator's corruption truth | must fire | 128 verdicts to 0 |
| 13 | a clean record produces no anomaly | a constructed flat in-range record | must **not** fire | 0 anomalies |
| 14 | a gap beyond the reorder horizon reads as a reorder | constructed counter arithmetic | must fire | delta 133 read as `reorder` |
| 15 | every injected misparse kind is detected | the generator's misparse kind | must fire | signedness 28/29, byteswap 36/36, scale 30/33 |

## 5. The learned component against the baselines

`validate_learned.py`. Three disjoint realisations of the generator: training
seed 20261028, test seed 71120261, calibration seed 44520261, 12,000 frames
each. Three declared configurations, no hyperparameter search.

### On the complete held-out population, 45,500 rows

Strongest rule baseline (strengthened) against the strongest learned
configuration (unweighted):

| class | metric | rules | learned | delta | winner |
|---|---|---|---|---|---|
| clean | recall | 0.9936 | 1.0000 | +0.0064 | learned |
| clean | precision | 0.9994 | 0.9987 | −0.0007 | rules |
| clean | surfaced | 1.0000 | 1.0000 | 0.0000 | tie |
| corrupt | recall | 0.4052 | 0.5098 | +0.1046 | learned |
| corrupt | precision | 0.9688 | 0.9750 | +0.0062 | learned |
| corrupt | surfaced | **0.7647** | 0.5098 | **−0.2549** | **rules** |
| misparsed | recall | 0.9321 | 0.9815 | +0.0494 | learned |
| misparsed | precision | 0.9437 | 0.9086 | −0.0352 | rules |
| misparsed | surfaced | 0.9815 | 0.9815 | 0.0000 | tie |

**Tally: rules 3, learned 4, tie 2.** The learned model wins four of the nine
metrics on this population, which is not what the specification predicted, and
it is reported as measured. The one large margin runs the other way: the rules
*surface* 0.7647 of the corrupt class against the learned model's 0.5098,
because the rules abstain on a row they cannot decide and the learned
configurations here never abstain. Recall and surfaced rate measure different
things and both are in the table.

### On the ambiguous residual set, 352 rows

This is the population the specification assigns to the learned component. The
majority class on the training ambiguous rows is `clean` at 0.785714.

| predictor | accuracy | delta against the majority guess | verdict |
|---|---|---|---|
| majority guess | 0.821023 | — | the line to beat |
| learned, `recall_weighted` | **0.255682** | **−0.565341** | **LOSES to the majority guess** |
| learned, `unweighted` | 0.906250 | +0.085227 | beats it |
| learned, `ambiguous_only` | **0.926136** | **+0.105114** | beats it |

Pooled over a 12-cell grid of anomaly rate against cadence jitter, 1,806
ambiguous rows: `ambiguous_only` **0.877076** against a majority guess of
**0.803433**.

**Both halves of that are published.** The configuration the imbalance handling
produces decides the answer completely: a model trained with
`class_weight="balanced_subsample"` on a subsampled clean class is 56.5
accuracy points *below* a constant on the population it exists for, and a model
trained on the ambiguous rows alone with no reweighting is 10.5 points above
it. Nothing about the architecture changes between them.

### Uncertainty outputs, measured on the independent third split

| configuration | rows | accuracy | ECE | dispersion AUC against wrong | mean dispersion |
|---|---|---|---|---|---|
| recall_weighted | 47,832 | 0.990676 | 0.054133 | 0.965703 | 0.154142 |
| unweighted | 47,832 | 0.998098 | **0.000781** | 0.886237 | 0.006279 |
| ambiguous_only | 363 | 0.884298 | 0.027389 | 0.685877 | 0.179044 |

The vote share is not post-hoc recalibrated. The dispersion is an ensemble
disagreement measure and not a confidence interval; its area under the ROC
curve against the indicator of being wrong is the only claim made for it.

### Compute

Generation, feature extraction, rule screening, three fits and three
predictions over three 12,000-frame splits plus a 12-cell swept grid:
**5.26 s** on two cores. The three-minute budget is not approached. Timings are
unstable and appear only in this table.

## 6. Command line and working directories

`validate_cli.py`: 11 commands executed in subprocesses with `HOME`,
`MPLCONFIGDIR` and `PYTHONDONTWRITEBYTECODE` set outside the repository; all
exit 0 and all print the expected fragment. An unknown subcommand and a missing
subcommand both exit 2. After every command the tree contains no `.config`, and
every tooling directory present is covered by an anchored `.gitignore` pattern.

`probe_working_directories.py`: every validation script and every example run
from the filesystem root, the system temporary directory, the repository root,
`validation/` and `src/`. **All exit 0 from all five.**

## What is not validated here

- No flight data. Every number rests on a seeded synthetic generator whose
  closed forms are printed in `telemgap/dataset.py`.
- No comparison against `xarray`, which is named in the alternatives table and
  is not installed in the measured environment.
- The policy is validated for linear interpolation and a zero-order hold only.
  No spline, no Gaussian process, no Kalman smoother.
- Equation (1) is the only decommutation modelled. A polynomial or table
  calibration is out of scope, and a misparse of that kind has no alternative
  reading in `telemgap.reparse` and will be reported undecided.
