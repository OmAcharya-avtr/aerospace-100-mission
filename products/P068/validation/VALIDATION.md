# Validation — degradepredict 0.1.0

**Validation level 2, research grade.** Every number below was produced by a
script in this directory, run in this repository, on the environment recorded
at the top of each `*_output.txt`. Nothing here is quoted from a paper, a
changelog or a previous run. Checks that failed, and comparisons the learned
model lost, are reported in the same tables as the ones it won.

Environment recorded by every run in this directory:

```
python           : 3.13.16
numpy            : 2.5.3
scipy            : 1.18.1
scikit-learn     : 1.9.1
cpu cores        : 2
```

The declared dependency ranges are `numpy>=2.5.0,<2.6`, `scipy>=1.18.0,<1.19`
and `scikit-learn>=1.9.0,<1.10`. The suite and the validation scripts were run
at **both ends** of all three ranges; see "Dependency range" below.

## Scripts and their output

| script | what it establishes | output |
|---|---|---|
| `validate_crossing.py` | crossing detection and label correctness | `validate_crossing_output.txt` |
| `validate_baselines.py` | the inverse-regression interval machinery | `validate_baselines_output.txt` |
| `validate_physics.py` | the power-law fade model and the cost of a fixed exponent | `validate_physics_output.txt` |
| `validate_headtohead.py` | the headline comparison, split by censoring | `validate_headtohead_output.txt` |
| `validate_calibration.py` | PIT calibration of the crossing-time distribution | `validate_calibration_output.txt` |
| `validate_cli.py` | every documented CLI invocation | `validate_cli_output.txt` |
| `worked_example.py` | the README worked example | `worked_example_output.txt` |

Run them all from the repository root:

```bash
python validation/validate_crossing.py
python validation/validate_baselines.py
python validation/validate_physics.py
python validation/validate_headtohead.py
python validation/validate_calibration.py
python validation/validate_cli.py
python validation/worked_example.py
```

Measured wall clock on the 2-core build machine is recorded in the README's
"Reproducing every number" section. The 180 s per-run budget is checked inside
`validate_headtohead.py` and reported in its own output.

## 1. Crossing detection and labels

`validate_crossing.py`

| check | reference | result | tolerance |
|---|---|---|---|
| Crossing of `y = 100 - 0.02 t` at the 80 % limit | closed form `1000.000000 d` | `1000.000000 d`, relative error `0.000e+00` | 1e-6 |
| Crossing of `y = 100 exp(-k t)`, `k = ln(1.25)/1000` | closed form `1000.000000 d` | relative error `0.000e+00` | 1e-6 |
| Crossing of `y = 100 (1 - B t**0.55)` | closed form `1000.000000 d` | relative error `0.000e+00` | 1e-6 |
| Interpolation error order, `y = 100 exp(-t/10)` | second order for linear interpolation | successive-error ratios 3.47, 3.62, 2.42, 3.46 on a halving grid | ratio > 2 |
| Every generated label is the crossing of its own mean curve, 450 trajectories across the three shapes | numerical location on a 60001-point grid | worst relative error `1.311e-15` | 1e-5 |
| Crossing boolean inside the declared tolerance band | — | three-valued: `within_tolerance` at margins of 1e-13 and 1e-11, ordinary comparison at 1e-9 and beyond | — |

## 2. The inverse-regression interval machinery

`validate_baselines.py`. Data generated from **exactly** the shape being
fitted, so this measures the interval arithmetic and nothing else. 2000 trials
per row, nominal two-sided level 0.90. A window the fit refuses counts as a
miss.

| shape | window | noise | Fieller coverage | delta coverage | unbounded sets | refusals |
|---|---|---|---|---|---|---|
| linear | 30 | 0.30 | **0.8925** | 0.8935 | 0.0000 | 0 |
| linear | 60 | 0.30 | **0.8835** | 0.8850 | 0.0000 | 0 |
| linear | 60 | 1.00 | **0.9010** | 0.8995 | 0.0000 | 0 |
| linear | 15 | 1.00 | **0.9275** | 0.8565 | 0.2620 | 13 |
| exponential | 30 | 0.30 | **0.9165** | 0.9105 | 0.0000 | 0 |
| exponential | 60 | 0.30 | **0.9005** | 0.8980 | 0.0000 | 0 |
| exponential | 60 | 1.00 | **0.9080** | 0.9100 | 0.0000 | 0 |
| exponential | 15 | 1.00 | **0.9405** | 0.8780 | 0.2005 | 9 |

Lowest Fieller coverage over the eight rows: **0.8835** against a nominal 0.90.
The delta-method interval under-covers in exactly the rows where the exact set
is often unbounded (0.8565 against 0.9275 at n = 15, noise 1.0), which is the
behaviour Fieller (1954) predicts and the reason both are computed.

Fieller against delta as a function of Fieller's `g`, 3696 fitted windows:

| g bin | windows | unbounded | median width ratio Fieller/delta | max |
|---|---|---|---|---|
| [0, 1e-3) | 176 | 0 | 1.000388 | 1.0010 |
| [1e-3, 1e-2) | 531 | 0 | 1.004112 | 1.0100 |
| [1e-2, 0.1) | 1192 | 0 | 1.034501 | 1.1102 |
| [0.1, 0.5) | 869 | 0 | 1.255887 | 1.9960 |
| [0.5, 1) | 278 | 0 | 3.724207 | 228.2004 |
| [1, inf) | 650 | 650 | — | — |

17.59 % of the randomly drawn windows have `g >= 1` and therefore an unbounded
exact confidence set. On the shipped telemetry windows `g` is of order 2e-3 and
the two intervals agree to one part in a thousand.

Parameter recovery, 200 replicates, n = 60, cadence 7 d:

| quantity | truth | mean estimate | sd | relative bias |
|---|---|---|---|---|
| linear slope [% / d] | -0.020000 | -0.020039 | 0.000319 | -0.0020 |
| linear crossing [d] | 1000.0000 | 998.8101 | 12.6629 | -0.0012 |
| exponential k [1/d] | 0.00022314 | 0.00022314 | 0.00000329 | -0.0000 |
| exponential crossing [d] | 1000.0000 | 1000.1636 | 11.6028 | +0.0002 |

The crossing estimate is a ratio of estimates and is biased in finite samples
even when the slope is unbiased; the table gives the size of that effect.

## 3. The physics-shaped baseline and the cost of a fixed exponent

`validate_physics.py`. Exponent `z = 0.55` from Wang et al. (2011), J. Power
Sources 196(8):3942-3948.

Recovery with the **correct** exponent, eight cases across z and crossing time:
worst relative crossing error **1.076e-07**.

Mean curve generated at `z_true`, fitted at the declared `z = 0.55`, true
crossing 1200 d, noise 0.3 %, 600 replicates per row:

| z true | mean estimate [d] | bias [d] | relative bias | mean interval width [d] | Fieller coverage |
|---|---|---|---|---|---|
| 0.45 | 1019.07 | **-180.93** | -0.1508 | 62.07 | **0.0000** |
| 0.50 | 1106.51 | -93.49 | -0.0779 | 60.16 | 0.0000 |
| 0.55 | 1199.82 | -0.18 | -0.0002 | 66.18 | **0.8933** |
| 0.60 | 1302.47 | +102.47 | +0.0854 | 77.81 | 0.0000 |
| 0.65 | 1417.65 | **+217.65** | +0.1814 | 95.55 | 0.0000 |

**This is the most important limitation of the physics-shaped model.** The
interval is tens of days wide; a 0.05 error in a declared exponent moves the
date by hundreds, and the interval gives no warning because `z` is declared,
not fitted. Profiling `z` over the grid reduces the bias where `z` is badly
wrong (mean absolute bias 181 d to 28 d at `z_true = 0.45`) and costs accuracy
where it is right (16 d to 37 d at `z_true = 0.55`), and the interval returned
after profiling does not include the selection step.

## 4. Head to head — the headline result

`validate_headtohead.py`. Train 1200 trajectories (seed 20261016), test 400
(seed 20261017), 227 of 400 test units right-censored (56.75 %). Nominal
interval level 0.90. CRPS and pinball in days, lower is better. `fb` is the
number of units for which a baseline could not fit and the declared
uninformative fallback was substituted and counted.

| model | cov complete | cov censored | cov pooled | CRPS complete | CRPS censored | pinball complete | pinball censored | fb |
|---|---|---|---|---|---|---|---|---|
| linear | 0.3237 | 0.3128 | 0.3175 | 104.17 | 368.79 | 50.82 | 178.96 | 3 |
| exponential | 0.3179 | 0.3392 | 0.3300 | 90.74 | **346.37** | 44.09 | **166.72** | 1 |
| powerlaw-fade | 0.0636 | 0.0793 | 0.0725 | 368.84 | 2373.34 | 177.96 | 1109.39 | 0 |
| learned-ipcw | **0.7630** | 0.3348 | 0.5200 | 30.80 | 435.56 | 14.00 | 212.57 | 0 |
| learned-complete-case | 0.7514 | 0.3172 | 0.5050 | **27.05** | 461.05 | **11.89** | 224.26 | 0 |

**There are two winners and they are different models.** The learned
forecaster has the lowest CRPS on complete trajectories (30.80 d for the IPCW
variant, 27.05 d for the complete-case ablation, against 90.74 d for the best
baseline). **The exponential baseline has the lowest CRPS on right-censored
trajectories, 346.37 d against 435.56 d**, and it wins on all three
independent seed pairs tested. A pooled score would have named one winner for
a comparison that has two answers.

Stability over three disjoint train/test seed pairs:

| model | cov complete mean (sd) | cov censored mean (sd) | CRPS censored mean (sd) |
|---|---|---|---|
| linear | 0.3009 (0.0307) | 0.2892 (0.0167) | 381.65 (9.66) |
| exponential | 0.2752 (0.0377) | 0.3611 (0.0253) | **354.71 (8.34)** |
| powerlaw-fade | 0.0769 (0.0107) | 0.1024 (0.0203) | 2385.22 (14.18) |
| learned-ipcw | 0.8095 (0.0344) | 0.3324 (0.0034) | 436.85 (0.93) |
| learned-complete-case | 0.7698 (0.0325) | 0.3122 (0.0074) | 460.83 (8.24) |

Lowest censored CRPS by seed pair: `exponential`, `exponential`, `exponential`.

### Why the baselines look bad pooled

Each analytic baseline is near its nominal coverage on the shape family it is
correctly specified for, and collapses off it:

| baseline | true shape | n | coverage | CRPS [d] | median absolute error [d] |
|---|---|---|---|---|---|
| linear | linear | 124 | **0.8629** | 46.20 | 15.53 |
| linear | exponential | 119 | 0.1513 | 86.20 | 69.54 |
| linear | powerlaw | 157 | 0.0127 | 546.18 | 409.46 |
| exponential | linear | 124 | 0.1855 | 106.81 | 69.22 |
| exponential | exponential | 119 | **0.9076** | 26.43 | 12.40 |
| exponential | powerlaw | 157 | 0.0064 | 496.40 | 368.53 |
| powerlaw-fade | linear | 124 | 0.0000 | 2849.00 | 1382.48 |
| powerlaw-fade | exponential | 119 | 0.0000 | 1911.48 | 1081.54 |
| powerlaw-fade | powerlaw | 157 | 0.1847 | 138.95 | 82.68 |

The power-law model reaches only 0.1847 even on its own family, because the
generator draws `z` from [0.45, 0.65] while the model holds it at 0.55; see
section 3.

### What IPCW buys

| quantity | complete-case | IPCW | change |
|---|---|---|---|
| coverage, complete | 0.7514 | 0.7630 | +0.0116 |
| coverage, censored | 0.3172 | 0.3348 | +0.0176 |
| CRPS, complete [d] | 27.0542 | 30.7972 | +3.7430 |
| CRPS, censored [d] | 461.0455 | 435.5584 | **-25.4872** |
| mean PIT, censored | 0.8043 | 0.7675 | -0.0367 |

IPCW moves every censored figure in the right direction and costs accuracy on
the complete ones. **It does not close the gap**: censored coverage stays at
0.3348 against a nominal 0.90.

Sensitivity to the declared IPCW weight cap (3, 5, 20): censored coverage
0.3040, 0.3348, 0.3348; censored CRPS 452.40, 448.09, 435.56 d. The largest
weight the estimator produces on this population is 14.0027, so a cap of 20 is
not binding and anything above it changes nothing. The censoring distribution
has bounded support here; on a fleet with heavier censoring the cap would
matter more.

### The part of the failure a deployment can see

For a right-censored unit, the only thing known at analysis time is that the
remaining life exceeds the follow-up, so an interval whose upper endpoint is
below the follow-up is refuted immediately. That falsification rate is a strict
lower bound on censored non-coverage and the only part of it that is
measurable in the field:

| model | falsified (measurable) | true non-coverage | ratio |
|---|---|---|---|
| linear | 0.2203 | 0.6872 | 0.3205 |
| exponential | 0.1762 | 0.6608 | 0.2667 |
| powerlaw-fade | 0.0088 | 0.9207 | 0.0096 |
| learned-ipcw | **0.0396** | **0.6652** | 0.0596 |
| learned-complete-case | 0.0529 | 0.6828 | 0.0774 |

A deployment that measured only what it can measure would see 4 % of the
learned model's censored failures and miss the other 62 points.

### CRPS quadrature

CRPS is twice the mean pinball loss over a probability grid. On the shipped
199-point midpoint grid against a 999-point grid the largest relative
difference across the five models is **1.056e-04** (powerlaw-fade); no ranking
changes.

## 5. Calibration of the crossing-time distribution

`validate_calibration.py`. Kolmogorov-Smirnov distance of the probability
integral transform from Uniform(0, 1); lower is better. The sampling noise
floor for a genuinely uniform sample of 173 points is **KS = 0.0398**
(p = 0.936).

| model | KS complete | KS censored | mean PIT complete | mean PIT censored |
|---|---|---|---|---|
| linear | 0.6295 | 0.6283 | 0.8373 | 0.8130 |
| exponential | 0.3584 | **0.4185** | 0.5410 | 0.5865 |
| powerlaw-fade | 0.7638 | 0.7423 | 0.1796 | 0.1770 |
| learned-ipcw | **0.1970** | 0.5980 | 0.3987 | 0.7675 |
| learned-complete-case | 0.1221 | 0.6240 | 0.4676 | 0.8043 |

Every statistic is far outside the noise floor. **No model in this repository
is probabilistically calibrated on this population.** The learned models are
the best calibrated on complete trajectories and among the worst on censored
ones, which is what a complete-case fit does even after reweighting.

## 6. Command line interface

`validate_cli.py` runs every invocation quoted in README.md. Four documented
invocations, **0 non-zero exit codes**, and `generate` is byte-identical on a
second run with the same seed.

## Dependency range

The suite and the validation scripts were installed and run at both ends of
every declared range. "Validation diff" compares the full text of every
`validation/*_output.txt` apart from the recorded version lines and the
wall-clock timings.

| declared range | low end installed and run | high end installed and run |
|---|---|---|
| `numpy>=2.5.0,<2.6` | 2.5.0 | 2.5.4 |
| `scipy>=1.18.0,<1.19` | 1.18.0 | 1.18.1 |
| `scikit-learn>=1.9.0,<1.10` | 1.9.0 | 1.9.1 |
| `pytest>=9.1.1,<9.2` (test extra) | 9.1.0 | 9.1.1 |
| `hypothesis>=6.169.0,<6.171` (test extra) | 6.169.0 | 6.170.0 |
| `matplotlib>=3.11.0,<3.12` (examples extra) | 3.11.0 | 3.11.2 |

The runtime ranges (numpy, scipy, scikit-learn) are the ones every validation
number depends on and the diff below is over those. The test and example
extras were additionally installed at both ends and the suite run (186 passed,
0 failures, 0 errors, 0 skipped at both) and all seven examples regenerated
without error at both matplotlib versions. **Note:** the suite was run at
pytest 9.1.0, which is below the declared floor of 9.1.1; the floor is set at
9.1.1 because that is the version the committed junit numbers were taken from,
and 9.1.0 is recorded here only as evidence that nothing in the suite depends
on the patch release.

| combination | tests (junit) | failures | errors | skipped | validation output |
|---|---|---|---|---|---|
| low: numpy 2.5.0, scipy 1.18.0, scikit-learn 1.9.0 | 186 | 0 | 0 | 0 | byte-identical to the committed files apart from the recorded version lines and wall-clock timings |
| high: numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1 | 186 | 0 | 0 | 0 | the committed files |

The diff was taken over the full text of all seven `validation/*_output.txt`
files with the four version lines and the two wall-clock lines excluded. **No
reported number moved.** The committed outputs are from the high end.

No trained model is serialised anywhere in this repository, so there is no
pickle-format dependency to bound beyond the ranges above.

## What is not validated

- Nothing here is validated against flight data, against a battery, or against
  any physical article. Every number is a statement about the declared
  generator in `degradepredict.generate`.
- The independent-censoring assumption that IPCW rests on is **true by
  construction** in the generator and is routinely false in a real fleet, where
  units are removed because they are degrading. The size of that error is not
  measured here and cannot be measured with this generator.
- Observation noise is independent and Gaussian. Real housekeeping telemetry
  has autocorrelated residuals, temperature dependence, quantisation and
  outliers, all of which make the residual-based intervals here optimistic.
- The forecast origin is the last sample on a fixed 7-day cadence. Irregular or
  gappy sampling is not exercised.
- Only a downward-degrading metric against a lower limit is generated. The
  `direction="above"` branch of `LimitSpec` is unit tested but has no
  end-to-end evidence.
