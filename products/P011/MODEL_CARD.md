# Model Card — WaveForge `LinearSlopePredictor` 0.1.0

**Model:** a bagged ensemble of eight scikit-learn ridge regressions mapping the
last four pseudo-open-loop Shack-Hartmann slope vectors to the slope vector
`h` frames ahead, with a per-slope predictive standard deviation.
**Product:** P011 WaveForge, OPTIMA aerospace portfolio.
**Status:** TESTING · **Validation level:** 3 · **License:** AGPL-3.0-or-later

> **This model is not certified for operational flight use.**

> ## ⚠ Read this before using the model
>
> On held-out phase-screen realisations the learned predictive controller beats
> the classical integrator **at its own tuned gain** at every latency tested,
> reducing the residual variance by 1.25× at one frame of latency and 3.45× at
> four frames. That result holds **only inside the conditions it was trained
> for**, and the model degrades ungracefully outside them:
>
> * at **twice the training wind speed** (20 m/s against 10 m/s) it is
>   **17 % worse** than the plain integrator;
> * trained on noise-free slopes and deployed at **100 detected
>   photoelectrons per subaperture**, it is **5.9× worse** than the plain
>   integrator. Retraining at the deployed noise level is mandatory, not
>   optional.
>
> The turbulence it was trained and tested on is **single-layer frozen flow**,
> which is the most predictable atmosphere there is. The measured advantage is
> therefore an upper bound on what a real multi-layer, boiling atmosphere would
> give. No comparison against measured atmospheric data has been made.

---

## 1. Problem

A closed adaptive-optics loop applies a correction that was measured `d` frames
earlier. Under Taylor frozen flow the wavefront is partly predictable from its
own history, so the temporal error `(d T / τ₀)^(5/3)` is not a hard floor. The
model forecasts the pseudo-open-loop slope vector the controller will need,
`d` frames ahead, from a short history of measurements.

* **Input:** `n_history × n_slopes` pseudo-open-loop slopes in rad/m
  (4 × 104 = 416 numbers for the default 8 × 8 sensor).
* **Output:** `n_slopes` forecast slopes in rad/m, plus a per-slope one-sigma.

Prior art for linear prediction in AO: M. B. Jorgenson and G. J. M. Aitken,
*Opt. Lett.* **17**, 466 (1992); C. Dessenne, P.-Y. Madec and G. Rousset,
"Modal prediction for closed-loop adaptive optics", *Opt. Lett.* **22**, 1535
(1997). Pseudo-open-loop control: B. L. Ellerbroek and C. R. Vogel, *Inverse
Problems* **25**, 063001 (2009); L. Gilles, *Appl. Opt.* **44**, 993 (2005).

## 2. Baselines — implemented and validated first

Per the mission rule, the classical path was written, validated and reported
before any ML was written. Both baselines are evaluated on the **same** screens,
with the **same** noise realisations, through the **same** simulation.

| Baseline | Definition | Where |
|---|---|---|
| Classical integrator | `c_k = c_{k-1} + g R s_{k-(d-1)}` | `control.Integrator`, validated in `validation/validate_rejection_tf.py` to 1.7e-14 against the analytic rejection transfer function |
| Pure-delay POL | identical pseudo-open-loop control path, forecast = newest available frame | `predictor.PureDelayPredictor` |
| Analytic error budget | fitting + temporal + noise, each cited | `errorbudget`, validated in `validation/validate_fitting_error.py` |

**The integrator's gain is tuned** over a grid on a separate tuning screen
(seed 555) at every latency and noise level, so it is presented at its best
available configuration and is not handicapped. The pure-delay baseline exists
so that the improvement attributable to *prediction* can be separated from the
improvement attributable to the pseudo-open-loop control formulation; at three
frames of latency roughly a third of the gain over the plain integrator comes
from the formulation and two thirds from prediction.

## 3. Architecture

* **Features** — the last `n_history = 4` pseudo-open-loop slope vectors,
  flattened oldest-first (416 numbers), standardised with the training mean and
  standard deviation stored on the instance.
* **Members** — 8 × `sklearn.linear_model.Ridge`, each fitted on a bootstrap
  resample of the training rows. `alpha` is chosen from
  `{1e-3 … 1e3}` on the last 20 % of the *training* set only; the selected
  values were 0.01, 0.1, 1 and 10 for horizons 1–4 on clean data, and 100–1000
  on noisy data.
* **Parameters** — 8 × (416 × 104 + 104) ≈ **346 k** coefficients.
* **Prediction** — member mean.
* **Uncertainty** — `σ = sqrt(ensemble variance + out-of-bag residual
  variance)`, a deep-ensemble style decomposition (Lakshminarayanan, Pritzel
  and Blundell, NeurIPS 2017).
* **Alternative** — `model="mlp"` swaps the ridge members for small
  `MLPRegressor` networks. It is implemented and tested but **not** the
  benchmarked configuration; no claim is made for it.

**Deviation from what one would ideally build.** A recurrent or convolutional
model is the natural architecture for a spatio-temporal forecast. PyTorch is
not available in this build environment and scikit-learn offers no convolution,
so a linear model with an explicit lag structure was chosen. It is also the
model the AO literature above actually uses, which makes the comparison against
the classical baseline a fair one rather than a straw man.

## 4. Data

See `DATASET_CARD.md`. In short: entirely synthetic, generated by seeded
scripts in this repository, never committed as files. Training and test come
from **different phase-screen seeds** — different atmospheric realisations, not
different slices of one sequence, which would leak badly because a frozen-flow
series is strongly autocorrelated by construction.

## 5. Training procedure and reproducibility

```bash
cd products/P011
PYTHONPATH=src python validation/validate_predictor.py     # full benchmark, ~100 s
PYTHONPATH=src python -m waveforge predict --frames 400    # short CLI version
PYTHONPATH=src python examples/predictive_control.py       # figure
```

Fixed seeds: training screens 101/102/103, held-out screens 901/902, gain
tuning 555, sensor-noise streams 4242 (tuning), 8888 (training noise),
31337 (evaluation), ensemble bootstrap `random_state = 0`.

**Compute used.** Fitting one eight-member ensemble on 1200 samples takes
0.45 s on one core; the entire benchmark, including 1200 frames of data
generation, 4 horizons of training, 96 gain-tuning runs and 40 evaluation runs,
took **98 s** on the 2-core build machine. `n_jobs` is 1 everywhere.

## 6. Metrics and test split

Held-out screens only; no number below comes from a screen the model saw.

### Open-loop forecast RMSE [rad/m], seed 901

| horizon | learned | pure delay | ratio |
|---:|---:|---:|---:|
| 1 | 1.4384 | 4.3191 | 0.333 |
| 2 | 2.8387 | 8.2481 | 0.344 |
| 3 | 4.2849 | 11.6673 | 0.367 |
| 4 | 5.8890 | 14.5840 | 0.404 |

### Closed-loop residual variance [rad²] and Strehl, seeds 901 and 902

| latency | integrator | pure delay | learned | `S` int | `S` delay | `S` ML |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.6728 | 0.6728 | **0.5361** | 0.5434 | 0.5434 | **0.6085** |
| 2 | 0.9274 | 0.8637 | **0.5354** | 0.4308 | 0.4519 | **0.6093** |
| 3 | 1.2798 | 1.0669 | **0.5444** | 0.3040 | 0.3848 | **0.6042** |
| 4 | 1.7085 | 1.3736 | **0.4958** | 0.2124 | 0.2818 | **0.6384** |

**Which won:** the learned predictor, at every tested latency, on both the
open-loop forecast and the closed-loop residual, against a gain-tuned classical
integrator. Reported as measured.

## 7. Uncertainty output — and its miscalibration

| horizon | inside 1σ (nominal 68.3 %) | inside 2σ (95.4 %) | inside 3σ (99.7 %) | mean σ | RMSE |
|---:|---:|---:|---:|---:|---:|
| 1 | 82.7 % | 99.2 % | 100.0 % | 1.7244 | 1.4384 |
| 2 | 76.2 % | 97.7 % | 99.9 % | 2.3823 | 2.8387 |
| 3 | 67.4 % | 94.6 % | 99.5 % | 2.1099 | 4.2849 |
| 4 | 62.3 % | 91.9 % | 99.0 % | 2.6805 | 5.8890 |

The intervals are **conservative at horizons 1–2 and optimistic at horizons
3–4**. The aleatoric term is estimated out-of-bag on the training screens and
does not grow enough to cover the extra error the model makes further ahead on
turbulence it has not seen. Treat `σ` as a **relative** confidence signal —
useful for spotting frames the model is unsure about — and never as a
probability. Do not use it to gate a safety decision.

## 8. Failure cases

| Condition | Learned | Classical integrator | Verdict |
|---|---:|---:|---|
| Wind 5 m/s (trained 10) | 0.6178 | 0.9465 | learned wins |
| Wind 15 m/s | 0.9440 | 1.2202 | learned wins, margin halved |
| **Wind 20 m/s** | **1.8005** | **1.5356** | **learned loses by 17 %** |
| 1000 e⁻, trained clean | 1.6065 | 1.5164 | learned loses |
| 300 e⁻, trained clean | 5.7542 | 2.4786 | learned loses by 2.3× |
| **100 e⁻, trained clean** | **29.7540** | **5.0277** | **learned loses by 5.9×** |
| 100 e⁻, noise-matched training | 3.8755 | 5.0277 | learned wins |

Additional known failure modes, not separately quantified:

* **Geometry change.** The model is tied to the subaperture layout, slope
  ordering and units it was trained on. Feeding slopes from a different sensor
  is a silent error: `predict` checks only the vector length.
* **Actuator saturation.** With a stroke-limited mirror the loop is non-linear
  and the pseudo-open-loop reconstruction of Eq. (1) in `loop.py` is no longer
  exact, so the forecast target itself becomes biased.
* **Subaperture dropout.** Dropped subapertures are reported as zero slope with
  a validity flag; the predictor ignores the flag and treats the zero as a
  measurement.
* **Frozen-flow repetition.** A run longer than the atmosphere's `max_frames`
  is refused rather than allowed to wrap, but a run close to it re-uses
  correlated turbulence.

## 9. Ethical and safety limits

This is research-grade software. It is not flight-qualified, not certified, and
not approved for operational aerospace use. The model is trained purely on
synthetic turbulence from a generative model in this repository and has never
been compared against measured atmospheric data, so its accuracy is a statement
about that generative process and nothing else. It must not be used to certify
a link budget, to size flight hardware without independent analysis, or in any
control loop whose failure has safety consequences. There is no human-subject
or personal data anywhere in the pipeline.

## 10. Maintenance

Retrain whenever the aperture, subaperture layout, actuator layout, frame rate,
wind statistics or sensor noise level change. The model carries no drift
detection; the only supported check is to compare its forecast RMSE against the
pure-delay baseline on fresh data, which the CLI (`python -m waveforge
predict`) does in one command.
