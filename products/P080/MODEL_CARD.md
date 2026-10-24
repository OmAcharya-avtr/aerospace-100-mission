# Model card — boundcompress learned residual predictor

**This model is not certified for operational flight use.**

Research grade. Not flight-qualified, not certified, not approved for
operational aerospace use. Every number below was produced by
`validation/validate_learned.py` and `validation/validate_predictors.py` in
the session that wrote this file; the raw output is in
`validation/validate_learned_output.txt` and
`validation/validate_predictors_output.txt`.

## Problem

Reduce the compressed rate of an error-bounded telemetry coder by predicting
the next quantised integer better than a finite-difference extrapolator does.

The model sits in the prediction step of a closed-loop DPCM coder. It sees
only the integer symbols the decoder has already reconstructed and emits an
integer prediction; the residual is coded losslessly. **It therefore cannot
change a single reconstructed sample or widen the declared error bound.** It
can only change the number of bytes. That is the whole reason the AI component
is here and not anywhere else in the pipeline: it is held to a rate comparison
at *identical* distortion, with nothing to trade away.

## Baseline, implemented first

Four integer finite-difference extrapolators in `boundcompress.predictors`,
all exact in integer arithmetic:

| name | prediction | exact on | noise gain |
| --- | --- | --- | --- |
| `previous` | q[i-1] | constants | 1 |
| `linear` | 2q[i-1] - q[i-2] | arithmetic progressions | 3 |
| `quadratic` | 3q[i-1] - 3q[i-2] + q[i-3] | quadratic sequences | 7 |
| `cubic` | 4q[i-1] - 6q[i-2] + 4q[i-3] - q[i-4] | cubic sequences | 15 |

Noise gain is the sum of the absolute coefficients: a first-order account of
why a higher-order extrapolator loses on a noisy source. The classical
baselines were implemented, measured and documented before the learned
component existed, and **a baseline win is never removed, softened or retuned
away**.

## Architecture

Two kinds, both ensembles of three members fitted on independent bootstrap
resamples with distinct seeds.

- `kind="linear"`: ridge regression (`sklearn.linear_model.Ridge`, alpha 1.0)
  on four features. This is a learned linear-prediction filter — the LPC idea
  of Makhoul 1975, *Linear prediction: a tutorial review*, Proc. IEEE 63(4) —
  with coefficients fitted rather than fixed. 5 parameters per member, 15 in
  total, **120 bytes** at float64.
- `kind="mlp"`: one hidden layer of 16 tanh units
  (`sklearn.neural_network.MLPRegressor`, solver L-BFGS, alpha 1.0, max_iter
  400, tol 1e-7). 97 parameters per member, 291 in total, **2328 bytes** at
  float64.

### Features, and why they are scale-free

Integer symbols scale as 1/Δ, so a model fitted at one declared bound would be
useless at another. For a window of W = 5 reconstructed symbols ending at
q[i-1], the first differences are d[j] = q[i-j] - q[i-j-1] for j = 1..4, and
the gain is

    s[i] = max(1, upper_median(|d[1]|, ..., |d[4]|))

an **exact integer function of integers**, so the decoder reproduces it
bit-for-bit. Features are d[j]/s[i], the target is (q[i] - q[i-1])/s[i], and
the prediction is q[i-1] + round(s[i] * f(d/s[i])).

A feature weight vector of [1, 0, 0, 0] would reproduce the classical `linear`
predictor exactly and [0, 0, 0, 0] would reproduce `previous`.

### Why the forward pass is hand-written

The models are fitted with scikit-learn and then their parameters are
**extracted**, and the forward pass is evaluated by `boundcompress.learned` in
explicit elementwise numpy operations, feature by feature and unit by unit.
`sklearn.predict` goes through BLAS, and a BLAS matrix-vector product and a
BLAS matrix-matrix product need not agree in the last unit in the last place.
The encoder evaluates the whole array in one batch; the decoder is inherently
sequential. A one-ulp disagreement would flip a `round` and desynchronise the
coder on whatever sample landed on a half-integer. Measured: 4091 predictions
per model, **0 differences** between the batch and sequential paths, both
kinds (`validate_roundtrip_determinism_output.txt`).

## Dataset

`boundcompress.dataset`, a seeded synthetic telemetry generator whose
equations are in its module docstring. See `DATASET_CARD.md`. **No flight data
is used anywhere.**

- **Training**: `dataset.training_signals()` — all eight channels at 2048
  samples, seeds 1000..1007, quantised at two declared bounds each (1e-3 and
  1e-4 of the dynamic range), giving 16 symbol arrays.
- **Evaluation**: `dataset.generate_all()` — the same eight channels at 4096
  samples, seeds 0..7.
- **Split strategy**: by **noise realisation**, not by channel. The model is
  evaluated on signal *shapes* it has seen with noise it has not. This is the
  easier of the two possible splits and is stated as such: a split by channel
  would be the harder question and was not run. `packet_counter` is noise-free
  by construction, so its training and evaluation arrays coincide; this is
  asserted and excluded in `tests/test_dataset.py`.
- Training rows available: 32,688 before subsampling, capped at **20,000** by
  a seeded subsample.

### Robust clipping, which is load-bearing

Training features and targets are clipped to ±8. The scale-normalised target
has a heavy tail — 99.9th percentile 8.0951, absolute maximum 4535.0 — and an
unclipped least-squares fit is dominated by a handful of step transients.
Measured by `validation/validate_learned.py` on the training rows:

| | ridge coefficients on d1..d4 | intercept | perceptron loss at 400 iterations |
| --- | --- | --- | --- |
| unclipped | [+0.0001, -0.0002, -0.0000, -0.0002] | +0.3069 | 1282.000772 |
| clipped at ±8 | [-0.3224, +0.0659, +0.1920, +0.1867] | +0.1430 | 0.631258 |

The unclipped coefficients are indistinguishable from predicting nothing.
Clipping is applied to training data only; nothing clips a prediction at
encode time.

## Training procedure and reproducibility

```bash
python validation/validate_learned.py      # the fit, the clip, the uncertainty
python validation/validate_predictors.py   # the rate comparison
```

Base seed 20261024; member m uses seed + m. The subsample uses
`numpy.random.default_rng(seed)`. Refitting the same configuration gives the
same prediction, asserted in `validate_learned_output.txt` and in
`tests/test_learned.py`.

Compute: two cores, no GPU. Fitting both kinds takes 4.9 s wall clock
(`validate_predictors_output.txt`, in a table explicitly declared unstable).
Everything in this repository finishes well inside the three-minute budget.

**Convergence is reported, not hidden.** 0 of 3 perceptron members converged
within the 400-iteration cap. Raising the cap to 800 moved the training loss
from 0.631258 to 0.629583 — 0.27 per cent — so the cap is a compute decision
rather than a silent failure. The fit is still non-converged and is labelled
as such everywhere it appears.

## Metrics, on the held-out realisations

Rate is actual encoded bytes. Distortion is identical across every predictor
at a given bound, by construction. 40 cells: eight signals, five bounds.

| item | value |
| --- | --- |
| cells won by a classical predictor | **14 of 40** |
| cells won by a learned predictor | 26 of 40 |
| median learned gain over the best classical predictor | **1.3451 %** |
| maximum learned gain | 50.4378 % |
| worst learned loss | **-55.8824 %** |
| best classical predictor overall | `previous` |
| best learned predictor overall | `learned_mlp` |

| predictor | total bytes over 40 cells | ratio to best |
| --- | --- | --- |
| learned_mlp | 136902 | 1.0000 |
| learned_linear | 150038 | 1.0960 |
| previous | 150168 | 1.0969 |
| linear | 152130 | 1.1112 |
| quadratic | 168824 | 1.2332 |
| cubic | 186657 | 1.3634 |

### The model's own bytes, charged

| predictor | params | model bytes | mean saving per array | worst saving | break-even arrays |
| --- | --- | --- | --- | --- | --- |
| learned_linear | 15 | 120 | **-264.52 B** | -7413 B | **never pays back** |
| learned_mlp | 291 | 2328 | +63.88 B | -420 B | **36.4** |

The parameters are a shared decoder asset and are **not** in the bitstream.
They are charged here rather than folded into the per-array rate, because
folding them in would make the rate depend on how many arrays the mission
sends. On a single 4096-sample `gyro_rate` container of 8386 bytes, the
`learned_mlp` parameters are 27.76 per cent of one container.

## Uncertainty output

Every predictor is an ensemble, the prediction is the member mean and the
**dispersion is the member standard deviation**, scaled into the array's own
unit by the quantiser step. It is a dispersion measure and **not a confidence
interval**; Wager, Hastie & Efron 2014, JMLR 15, is the reference for why an
ensemble spread does not give coverage.

Measured usefulness — Spearman rank correlation against the realised absolute
residual:

| model | signal | Spearman |
| --- | --- | --- |
| learned_linear | battery_voltage | 0.0648 |
| learned_linear | gyro_rate | 0.0179 |
| learned_linear | wheel_rate | **0.8329** |
| learned_linear | white_noise | 0.0286 |
| learned_mlp | battery_voltage | 0.0215 |
| learned_mlp | gyro_rate | 0.0275 |
| learned_mlp | wheel_rate | 0.5609 |
| learned_mlp | white_noise | 0.0233 |

**Median 0.0280**, range 0.0179 to 0.8329. The dispersion ranks errors usefully
on `wheel_rate`, which alternates between predictable ramps and unpredictable
slew changes, and is close to useless on the other three. A single-member
ensemble has identically zero dispersion, asserted in `tests/test_learned.py`.

## Failure cases

1. **`packet_counter` at a fine bound.** `linear` uses 838 bytes,
   `learned_linear` 8251 — a factor of 9.8. The signal is an exact arithmetic
   progression, which second-order linear extrapolation predicts with zero
   residual and which a model fitted to average over eight channels does not.
   This is the worst loss in the benchmark.
2. **`wheel_rate` at a coarse bound.** `linear` 187 bytes, `learned_mlp` 272.
   Piecewise-linear ramps are again the `linear` predictor's exact case.
3. **`step_transient` at coarse bounds.** `previous` 83 and 303 bytes against
   161 and 466 for `learned_linear`. A level change is unpredictable by
   anything, and the cheapest response is to predict no change.
4. **Non-convergence.** The perceptron members hit the iteration cap. See
   above.
5. **Uninformative uncertainty.** Median Spearman 0.0280.
6. **Cross-machine reproducibility is not claimed.** `tanh` is a libm call.
   The classical predictors have no such limitation.

## Ethical and safety limits

- The model is trained on synthetic data and every number about it is a
  statement about that generator, not about any spacecraft.
- It cannot widen the declared error bound — that is structural and asserted
  by test over 1,310,720 sample-versus-bound comparisons — but a mission that
  relied on its *rate* would be relying on a 1.35 per cent median improvement
  measured on eight synthetic channels, which is not a basis for a link
  budget.
- A flight implementation should use a standardised codec (CCSDS 121.0-B-3 or
  122.0-B-2). The contribution of this product is the certificate and the cost
  of the guarantee, not the predictor.

**This model is not certified for operational flight use.**
