# Model card — `powerenvelope` active-power predictor

**Status: TESTING.** Every measured figure below comes from
`validation/validate_measured_learned.py` run against the committed campaign in
`data/traces/`, and that script refuses and exits non-zero if the campaign is
absent.

**This model is beaten by a classical predictor on this campaign, on both
held-out schemes, and is kept in the product reported as beaten. See
*Pre-registered expectation, and its verdict*.**

**This model is not certified for operational flight use.**

## Problem

Predict the **mean active rail power**, in mW, that a duty-cycled on-board
autonomy pipeline will draw on one NVIDIA Jetson Orin Nano, from descriptors an
operator could state from the configuration and read from the platform's own
environment **without running the pipeline**.

The quantity matters because it is the one term of the analytic duty-cycle model
(`E_cycle = P_floor*t_sleep + E_ramp + P_active*t_active`) that an operator
cannot read off a datasheet, and because predicting it without running the
pipeline is what would let a mission size a power budget for a configuration it
has not yet flown.

**How it is scored.** Predicting a power is not by itself decision-relevant, so
the model is scored on the job that quantity does: supplying `P_active` to the
analytic model. Everything else Baseline 2 needs — the measured floor, the
counted inferences per cycle, the counted acceptance rate — is held at its
measured value, and the error reported is the relative error of the resulting
**energy per useful decision** against the Baseline 3 measured ground truth of
the same cell. The equivalent two-term form
`E_cycle = P_floor*t_sleep + P_mean_active*t_active` is what is substituted
into, because it consumes exactly the quantity the window dataset targets; it is
algebraically the same number as the three-term form and
`validate_measured_baselines.py` publishes the residual of that identity
(worst case 4.441e-16 relative: under the declared-phase partition both forms
use the same declared active duration, so the two are the same number to
rounding).

## Baseline, implemented and validated first

The classical baseline is implemented and checked against the ground truth
**before** the learned model is benchmarked, because an under-strength baseline
inflates a learned win.

- **`AnalyticActivePowerModel`** — Baseline 2's duty-cycle model used in
  reverse. From the measured background floor and, for each **training** cell
  only, that cell's whole-run mean power and its **declared** duty, it estimates
  the active excess as `mean((P_run - P_floor) / duty)` and predicts
  `P_floor + excess`. One parameter, fitted to nothing in the learned sense, and
  it never sees a test run.
- **`ConstantPowerPredictor`** — the training mean. A learned model that cannot
  beat this has learned nothing from the descriptors.

Both are scored on exactly the same splits as the learned model.

## Architecture

`sklearn.ensemble.RandomForestRegressor`, 300 trees, `max_depth=6`,
`random_state=930093`, `n_jobs=1`.

A forest rather than anything larger for three measured constraints, not
preferences: the dataset is a few hundred windows from one campaign; the build
environment has two cores and a three-minute budget for any fit; and the spread
across trees gives an uncertainty output without a second model.

## Descriptors, and the ones that are excluded

**Licensed** (`powerenvelope.windows.FEATURE_NAMES`, in column order):

| descriptor | where an operator gets it |
|---|---|
| `duty` | the configuration |
| `cycle_s` | the configuration |
| `frame_dim` | the configuration |
| `tau` | the configuration, via the calibration rule declared before the campaign ran |
| `cores_available` | the platform |
| `env_tj_c` | the platform's thermal zone at the window start |
| `env_load1` | the platform's 1-minute load average at the window start |
| `env_clk_mean_khz` | the platform's core clocks at the window start |
| `env_foreign_busy` | `/proc/stat` occupancy in pipeline-free monitor seconds |
| `env_foreign_imputed` | 1 when the line above could not be separated |

`cycle_s` and `cores_available` are constant across this campaign. They are kept
in the matrix because they are part of the licensed set, and a reader should see
that they were offered to the model and carried no information rather than find
them quietly absent.

**Excluded by specification, and enforced in code.**
`powerenvelope.predictor.assert_descriptors_licensed` is called inside
`ActivePowerPredictor.fit` and raises `DescriptorLeakError` for any descriptor
outside the licensed set. The excluded classes are the pipeline's own measured
power, its measured latency, its CPU occupancy, its realised inference count and
its realised acceptance rate. The check is exact-membership, so a quantity
nobody anticipated is rejected too; the name fragments in
`EXCLUDED_DESCRIPTOR_CLASSES` only make the refusal say which class the caller
tripped over. `validation/validate_negative_controls.py` asserts that it fires
on each class, and `tests/test_predictor.py` asserts it again per class.

### Two licensed descriptors that are weaker than the licence suggests

Both are stated here rather than hidden, and both are measured by
`validation/validate_descriptor_leak.py`, which refits without them and reports
the cost.

1. **`env_load1` is not pipeline-free.** The 1-minute load average includes the
   pipeline's own run-queue contribution. The specification names it as a
   licensed descriptor and it is used. The contamination is real.
2. **`env_foreign_busy` is separable only at low duty.** The monitor samples
   `/proc/stat` once a second and each sample covers the preceding second. A
   sample is pipeline-free only when no inference interval overlaps that second.
   With a 2.0 s cycle the sleep window is `2*(1-duty)` seconds, so a
   pipeline-free second is plentiful at duty 0.10 and 0.25 (39 of 77 monitor
   seconds), occurs exactly **once** per run at duty 0.50 and 0.75 because the
   sleep window and the monitor's own 1 s interval have an arbitrary relative
   phase, and **cannot** exist at duty 1.00. Where a window has none, the value
   is **imputed** from
   the floor cell's own mean occupancy and `env_foreign_imputed` is set to 1.
   That is an imputation from a measured constant of the same campaign, flagged
   in the design matrix, and it is **not** a per-window measurement. The
   specification asked for this to be said rather than pretended, and this is
   where it is said.

## Dataset

The measured campaign described in `DATASET_CARD.md`. Windows of 8.0 s, four
cycles each, cut from every cell that ran the pipeline; the background floor cell
contributes nothing because it has no active window and therefore no target. A
window with less than 0.2 s of measured active duration is dropped and the drop
is reported with its reason.

| quantity | measured |
|---|---|
| windows | 171 (9 per cell, 19 pipeline cells) |
| windows with imputed foreign occupancy | 117 |
| target range, mW | 3820.274 to 5657.178 |
| windows dropped | 1, the background floor cell, which has no active window and therefore no target |

On the committed synthetic generator at 10 cycles per run the same code yields
38 windows, 12 of them imputed, which is what `tests/test_windows.py` and
`tests/test_regressions.py` pin. Those are synthetic figures and they are not
measurements of anything.

## Training procedure and test-split strategy

Scored **separately** on two groupings, never pooled, because a model that
interpolates across duty at a fixed frame dimension is answering a different
question from one that extrapolates to an unseen workload:

- **held-out duty cycles** — leave-one-duty-out over 0.10, 0.25, 0.50, 0.75,
  1.00;
- **held-out frame dimensions** — leave-one-dimension-out over 48, 64, 96.

Grouping is by cell, so no window of a test group appears in training; the
repeats of the reference cell and the permissive control fall in the duty-0.50
and dimension-64 groups with the grid cell they repeat.
`tests/test_predictor.py::test_analytic_model_never_sees_a_test_cell` and
`tests/test_windows.py::test_grouped_splits_never_share_a_cell_between_train_and_test`
assert the grouping rather than assuming it.

## Metrics

Mean relative error of **energy per useful decision** against the measured
ground truth, with only `P_active` predicted:

| predictor | held-out duty | held-out frame dimension |
|---|---|---|
| measured `P_active` (ceiling) | 0.029186 | 0.029186 |
| classical: constant grid mean | 0.066443 | **0.066287** |
| classical: linear in duty and dimension | **0.051959** | 0.076385 |
| classical: analytic duty inversion | 0.073620 | 0.079520 |
| learned: random forest | 0.052428 | 0.072997 |
| between-run interval half-width | 0.029543 | 0.029543 |

The predictors' own relative error on the active power, for comparison: forest
0.061286 and 0.083798; linear 0.045813 and 0.096435; constant 0.072595 and
0.074642; duty inversion 0.077117 and 0.112652.

Mean one-sigma coverage of the learned predictor over the eight splits:
**0.2615**, far below 0.68. The across-tree spread is a lower bound on the
predictive uncertainty and includes neither run-to-run variation nor the
campaign's load drift; the figure is printed as it is rather than widened.

Forest fit time over all eight splits: **1.51 s** on two cores, against a
three-minute budget.

## Pre-registered expectation, and its verdict

From `SPEC.md`, written before the campaign ran:

> the analytic duty-cycle model of Baseline 2 is competitive, and the learned
> predictor's error will not be smaller than the analytic model's by more than
> the between-run interval's half-width.

`validation/validate_measured_learned.py` section 8 computes the comparison and
prints the verdict whichever way it falls. The margin is in mW of mean active
power and the interval half-width is in mJ per useful decision, which are
different units, so the comparison is made in relative form — the learned margin
as a fraction of the analytic error, against the half-width as a fraction of the
mean energy per useful decision — and that choice is stated in the output rather
than left implicit.

**Verdict: NOT REFUTED, on both splits.** The margin (strongest classical minus
learned) is **-0.000469** on held-out duty and **-0.006710** on held-out frame
dimension; both are negative — the learned forest is slightly *worse* — and both
are far inside the 0.029543 half-width. The classical predictors are
competitive, exactly as the specification expected before the data existed.

That is the finding. It is published as it stands, the learned component stays
in the product reported as beaten, and nothing was retuned. Note also that the
ceiling — the analytic form with the power known exactly — is 0.029186,
essentially at the measurement's own 0.029543 noise floor, so on this campaign the
analytic form is close to as good as any prediction of `P_active` could make
it.

## Uncertainty output

`ActivePowerPredictor.predict` returns a point estimate and a per-sample
standard deviation across the ensemble's trees. That is the ensemble's internal
disagreement and it is a **lower bound** on the predictive uncertainty: it
excludes the noise in the target.

Where the caller supplies the per-window quantisation step and sample count, the
instrument's own quantiser noise is combined in quadrature.
`target_quantisation_sd` uses `q / sqrt(12)`, the standard deviation of a
uniform quantiser of width `q`, with `q` the rail current quantum of 8 mA
expressed as power at the measured bus voltage, about 40 mW. The `sqrt(n)`
reduction applied over a window's samples is named in the docstring as a
**lower bound** rather than a correct propagation, because the INA3221 averages
512 conversions in hardware and neighbouring 20 Hz samples therefore overlap in
the instrument.

Neither component includes run-to-run variation. That is measured separately by
`baselines.between_run_spread` over the four repeats of the reference cell, and
it is the quantity the pre-registered comparison is judged against.

**Coverage**, the fraction of test windows whose true power lies within one
reported sigma, is the only check that the uncertainty output means anything. It
is reported whatever it is: **0.2615**, against the 0.68 a correctly sized
one-sigma band would give. The reported sigma is too small by roughly a factor
of two to three, which is what a lower bound that omits run-to-run variation and
a 27-minute load drift should look like. It is not widened to make the number
read better.

## Failure cases

- **Duty 0.50 and above**: `env_foreign_busy` is imputed on 117 of 171 windows,
  so for those cells the descriptor carries no per-window information. It still
  draws 0.1621 of the summed impurity importance, which is importance attached
  to what is effectively a constant plus its own flag.
- **The campaign's load drift is in the training data.** Frame dimension is
  confounded with elapsed time and with a falling foreign load, so the 0.2208
  importance the forest gives `frame_dim` is partly importance given to when the
  run happened. A model trained on this campaign would carry that confound into
  any prediction it made.
- **Extrapolation beyond the grid**: a random forest cannot predict outside the
  range of its training targets, which `tests/test_predictor.py` asserts. A duty
  or frame dimension outside the campaign's grid will be answered with the
  nearest training leaf and the across-tree spread will not grow to say so.
- **A second power mode, a second board, a second ambient, a GPU pipeline, a
  different cycle period**: all outside this campaign. The loader refuses a
  manifest declaring a different `cycle_s` rather than extrapolating.
- **The target is a mean over a window.** A window whose active duration is
  short has a noisier target; windows below 0.2 s of active duration are dropped
  rather than down-weighted.

## Reproducibility

```bash
python -m pytest tests/ -q --junit-xml=junit.xml
python validation/validate_descriptor_leak.py        # synthetic, runs today
python validation/validate_measured_learned.py       # refuses until the traces land
python examples/learned_vs_analytic.py
python -m powerenvelope --simulate learned
```

Seeds: `RANDOM_STATE = 930093` for every forest fit;
`SEED_FIXTURE = 930093` for the synthetic generator;
`SEED_CONTROL = 5150` for the controls. They are declared in
`src/powerenvelope/predictor.py` and `validation/_bootstrap.py` so that no two
reported numbers rest on different ones.

No model artifact is committed. The forest is refitted from the committed traces
in under a second; a pickled estimator would bind this repository to one
scikit-learn build whose unpickling is not part of that library's public API.

## Compute used

Two cores on the analysis machine, `x86_64 Linux`, Python 3.13.18,
scikit-learn 1.9.1. Measured forest fit time over all eight splits: **1.51 s**,
printed by `validation/validate_measured_learned.py` section 7 and asserted
there to be under the three-minute budget. On the synthetic campaign the whole
eight-split benchmark, all models, runs in under five seconds, which
`tests/test_integration.py::test_learned_predictor_runs_on_both_splits_inside_the_budget`
asserts.

The traces themselves were measured on the board, aarch64; the arithmetic over
them runs on the analysis machine. Those are different machines and every
validation output prints both.

## Ethical and safety limits

This is a power-accounting model for one engineering-reference developer board
running one synthetic CPU-only pipeline on a shared, contaminated machine. It
carries no personal data and makes no decision about any person.

It is research-grade. It is not flight-qualified, not certified, and not
approved for operational aerospace use. **This model is not certified for
operational flight use.**
