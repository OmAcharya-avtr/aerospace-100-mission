# Model card — thermalbudget steady-state thermal predictor

**This model is not certified for operational flight use.**

## Problem

Given a **short warm-up trace** — the first N seconds of die temperature, mean
CPU clock and achieved duty cycle from a run on an NVIDIA Jetson Orin Nano
Engineering Reference Developer Kit Super — predict the **eventual steady-state
die temperature** the run will settle at, so that a 12-minute soak does not have
to be run.

The steady state is the quantity the rest of the product turns into a sustained
throughput, so an error here propagates directly into the headline.

## Baseline, implemented and validated first

The baseline is **baseline 2, the first-order thermal RC network**
(`thermalbudget.rc_model`), fitted to **exactly the same warm-up window the model
sees** and extrapolated to its asymptote (`rc_warmup_prediction`). This matters
more than it sounds: handing the RC model the whole soak while handing the
learned model 90 seconds would inflate the learned margin, and the mission's
build guide records two products where a weak baseline did exactly that.

The baseline is validated before it is used as a yardstick. Four independent
bounds are checked against it in `validation/validate_independent_bound.py`:
monotonicity, the local-slope identity `T_inf = T + tau dT/dt` with tau taken
from the **cooldown** traces, three estimators of tau that share no algebra, and
a normal-theory interval against the bootstrap.

## Architecture

Two models. The choice between them is an argument, not a default.

**Primary — `ridge`.** Eleven features, standardised, into
`sklearn.linear_model.Ridge(alpha=1.0)`, wrapped in a **group bootstrap**: 200
ridge fits on training sets resampled at the level of the RUN. The prediction is
their mean; the uncertainty is the spread across fits (epistemic) combined in
quadrature with the in-sample residual spread (aleatoric). It is primary for one
reason that is specific to this evaluation axis: held-out duty cycle means
training on cooler duties and predicting a hotter one, which is extrapolation in
the target, and a linear model extrapolates.

**Comparison — `forest`.** `sklearn.ensemble.RandomForestRegressor`, 200 trees,
uncertainty from the spread across trees. Reported beside the ridge because a
tree ensemble is the obvious default and its failure mode here is worth showing:
its prediction is bounded by its training targets, so on the held-out duty-1.0
split it cannot predict a steady state hotter than the hottest run it trained
on, and it does not. `tests/test_predictor.py` asserts that every one of its
held-out duty-1.0 predictions is below every truth.

Features, in order (`thermalbudget.predictor.FEATURE_NAMES`): `window_s`,
`achieved_duty_mean`, `die_first_c`, `die_last_c`, `die_mean_c`, `die_rise_c`,
`die_slope_c_per_s`, `die_slope_second_half_c_per_s`, `die_slope_ratio`,
`clock_mean_khz`, `clock_rel_range`. Every one is computable from the warm-up
window alone; nothing reads the end of the run and nothing reads the target.

`alpha` is **declared, not searched**. With six runs there is no honest held-out
set to tune it on, and tuning it against the test duty would make the held-out
score meaningless.

## Dataset

See `DATASET_CARD.md`. In brief: the campaign's soak runs, one row per (run,
warm-up window length) pair over six declared window lengths from 60 s to 150 s.

**The row count overstates the information and the card says so wherever the row
count appears.** Rows from one run are nested inside it and are strongly
correlated. Every split is grouped by RUN and the evaluation is grouped by DUTY.
A random-row cross-validation score on this dataset would be meaningless and is
not offered anywhere in this repository.

A run that fails the equilibrium test has no measured steady state to be a
target and is **excluded by name**, not imputed.

## Training procedure

```bash
python validation/validate_measured_learned.py
```

Deterministic: seed 910091 everywhere, `numpy.random.default_rng` for the group
resampling, `random_state=910091` for the forest. No early stopping, no schedule,
no tuning loop. Fitting both models over all four held-out-duty splits takes
about two seconds on two cores.

## Metrics and test-split strategy

**Held-out DUTY CYCLE — measurable, and measured.** Leave-one-duty-out: train on
every duty cycle but one, predict the held-out one, pooled mean absolute error in
degrees Celsius against the RC baseline on the same windows. A duty whose removal
would leave fewer than two training runs is skipped and named.

**Held-out AMBIENT — STRUCTURALLY UNMEASURABLE.** There is one ambient in this
campaign and no way to vary it without physical access to the board, so there is
no second point to hold out. This axis has **no score**. It is not dropped, and
it is **not** approximated by the duty-cycle split: a model can pass one and fail
the other, which is precisely why the registry summary asks for both separately.
Reported as unmeasurable, with that reason, in `README.md`, in
`validation/VALIDATION.md`, in this card, in
`validation/validate_measured_learned_output.txt`, and by
`thermalbudget.limits.structurally_unmeasurable_axes()`.

**The decision rule, fixed before any number was computed.** The learned margin
is `RC error - learned error`, and it is compared against the **half-width of
baseline 3's run-to-run bootstrap interval on the steady-state die temperature**.
A margin smaller than that half-width is reported `NOT_DISTINGUISHABLE`,
**in either direction**. A margin inside the interval is never reported as an
improvement, and nothing is retuned until the model wins.

## Results

**MEASURED on the board**, from `validation/validate_measured_learned_output.txt`.
Dataset: 36 rows from 6 runs x 6 warm-up windows, 4 duty cycles, 11 features, 0
runs excluded. The RC baseline is fitted to exactly the same warm-up window.

| Model | Held-out-duty MAE | RC on the same window | Margin | Margin / run-to-run half-width | Verdict |
|---|---:|---:|---:|---:|---|
| ridge (primary) | **4.9205 degC** | 0.5120 degC | **-4.4084 degC** | **17.13** | **BASELINE_BETTER** |
| forest (comparison) | 2.9492 degC | 0.5120 degC | -2.4372 degC | 9.47 | **BASELINE_BETTER** |

The run-to-run bootstrap half-width on the steady-state die temperature is
**0.2573 degC**. Both margins are more than nine times it, so both are outside
the interval and the verdict is attributed rather than withheld.

Per-duty ridge mean absolute error, degrees Celsius:

| held-out duty | n rows | ridge | RC | margin |
|---|---:|---:|---:|---:|
| 0.25 | 6 | 2.3870 | 0.0958 | -2.2912 |
| 0.50 | 6 | **0.3496** | 0.6054 | **+0.2558** |
| 0.75 | 6 | 0.9794 | 0.5116 | -0.4678 |
| 1.00 | 18 | **8.6022** | 0.6197 | **-7.9825** |

**Two details that cut in opposite directions, both reported.** At duty 0.50 the
ridge *beat* the RC baseline by 0.2558 degC — one split of four, which does not
change the pooled verdict but is recorded because the rule cuts both ways. At
duty 1.00, the hottest duty cycle and the one held out from a training set that
saw only 0.25, 0.50 and 0.75, the ridge is wrong by 8.6022 degC. That is the
failure mode predicted before the data existed: **the model cannot extrapolate
past its training duty cycles.**

The learned model lost, decisively, and it is published as found. Nothing was
retuned, no alpha was searched against the test duty, and no split was dropped.

**On SYNTHETIC fixtures**, for comparison and labelled as such
(`examples/learned_vs_rc.py`, `no_throttle` regime, before the traces arrived):
ridge 1.4669 degC against RC 0.1280 degC, margin -1.3389 degC, 14.31 times the
fixture run-to-run half-width. The fixtures are generated *from* the first-order
RC form, so that figure is evidence about the estimators and not about the
board; the measured table above supersedes it.

## Uncertainty output

`SteadyStatePredictor.predict` returns `(prediction, sigma)`. `sigma` is a
one-sigma figure combining ensemble spread with in-sample residual spread in
quadrature; it is not an interval, and the report layer multiplies it where one
is wanted.

**MEASURED calibration: the uncertainty output is bad and must not be relied
on.** Two-sigma empirical coverage over the held-out rows is **0.6111** for the
ridge and **0.8333** for the forest, against a calibrated 0.95. The ridge's
two-sigma band misses the truth on nearly two rows in five. The cause is the
same as the point-estimate failure: on the held-out duty-1.0 split the model is
both wrong and confident, reporting sigma near 1.9 to 4.7 degC while its single worst
miss is 13.494 degC (soak_1.00_r3, 150 s window, truth 56.479, predicted 42.985,
sigma 4.722 — a 2.86-sigma miss). A band that narrow on an extrapolated duty cycle is not
supported by anything in the training data.

## Failure cases

- **Extrapolating in duty — MEASURED, and this is the dominant failure.** The
  held-out duty-1.0 split asks for a prediction hotter than anything in
  training. Measured ridge error there: **8.6022 degC** against 0.3496 to
  2.3870 degC on the cooler held-out duties. Per-run, the ridge predicted
  53.6–53.7 degC for `soak_1.00_r1` (truth 56.999), 47.4–47.6 for
  `soak_1.00_r2` (truth 56.647) and 43.0–43.2 for `soak_1.00_r3` (truth
  56.479): the error grows as the training set shrinks towards the cooler
  duties. The forest's measured duty-1.0 error is 3.7458 degC and is bounded by
  its training targets by construction.
- **A warm-up window short relative to tau.** The measured rise time constant
  is **88.85 s** on `soak_1.00_r1`, so the shortest declared warm-up window,
  60 s, is 0.68 of one time constant and the asymptote is extrapolated rather
  than observed. `RelaxationFit.span_in_tau` reports the ratio; below about 1.5
  the RC prediction should be distrusted, and the learned model has no better
  information. Measured, the RC baseline's errors on 60 s windows are 0.229,
  0.362, 1.637, 1.390, 1.592 and 0.613 degC across the six runs — the largest
  errors it makes anywhere except one, and its single largest error in the whole
  evaluation is 1.845 degC at duty 0.50 with a 75 s window. Short windows are
  where the classical baseline is weakest, and it still wins by 17.13 run-to-run
  half-widths overall.
- **A system with two time constants — MEASURED, and it is present.** The
  measured rise fit has a lag-1 residual autocorrelation of **+0.732** and two
  independent bounds put the true asymptote above the fitted value (by 0.233
  degC against the final-window mean and 0.038 degC against a cooldown-tau
  bracket). So the RC baseline this model is measured against is itself biased
  low, in a known direction, and still beats the model by 17.13 run-to-run
  half-widths. The learned model has no mechanism for a second time constant
  beyond the slope-ratio feature.
- **A run that never equilibrates.** No target exists, the run is excluded, and
  the dataset shrinks. With six runs that is expensive.
- **Co-resident load — MEASURED.** The board is shared: pooled foreign CPU
  **17.05 per cent** over 4975 samples, per-run means 14.93 to 20.62 per cent,
  per-sample maxima to 28.9 per cent. A soak taken while another process was
  busy has a different thermal input, and `foreign_cpu_pct` is the only record
  of it. It is a feature of none of the models here.
- **GPU load — UNTESTED.** The GPU devfreq sat at its 306 MHz floor for 99.578
  per cent of the campaign and GR3D utilisation was never sampled. Every figure
  in this card is a CPU-payload figure and a GPU-loaded pipeline could behave
  completely differently.

## Reproducibility

```bash
python -m pytest tests/ -q -p no:cacheprovider --junit-xml=junit.xml
python validation/validate_measured_learned.py   # requires data/traces/
python examples/learned_vs_rc.py
```

Seeds: 910091 for every analysis and fixture stream, 880091 for controls. The
synthetic fixtures are byte-for-byte reproducible at a fixed seed and
`tests/test_fixtures.py` asserts it. No model artefact is serialised: the
predictor refits in about two seconds, and a pickled estimator would bind this
repository to one scikit-learn build, which is the defect class that cost an
earlier product of this mission four test errors on a cold clone.

## Compute

Two cores, 7.8 GiB, x86_64, for the ANALYSIS. The traces themselves were
measured on the board's aarch64 cores, and the distinction is printed in the
header of every validation output file. Building the dataset, fitting 200
bootstrap ridges over four held-out-duty splits and the forest comparison: about
two seconds. The whole test suite: about 12 to 14 s. The slowest validation
script is the property hunt at 20.2 s. Nothing in this product needs a GPU and PyTorch is not
installed in the measured environment.

## Ethical and safety limits

This model predicts a die temperature on one development board under one
payload. It is research-grade. It is **not flight-qualified, not certified and
not approved for operational aerospace use**, and no figure it produces should be
used to size a thermal margin on flight hardware. Every claim the campaign cannot
support is refused by name: run `python -m thermalbudget limits`.

**This model is not certified for operational flight use.**
