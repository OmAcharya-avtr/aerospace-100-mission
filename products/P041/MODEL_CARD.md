# Model card — fade-duration exceedance predictor (`codedfade.predictor`)

**This model is not certified for operational flight use.**

Every number in this card was produced by `validation/validate_ai_vs_baseline.py`
in this repository and is reproduced in `validation/validate_ai_vs_baseline_output.txt`.
The committed artefact is regenerated deterministically by
`validation/train_model.py`.

## Problem

At the instant the received optical amplitude crosses below a threshold, estimate

```
P(T > t_target | state observable at the crossing)
```

where `T` is the duration of the fade that has just begun. That probability is what
sizes an interleaver: a depth covering the *mean* fade leaves the tail uncovered,
and the designer needs the tail conditioned on what the receiver can actually see.

Framed as binary classification: label 1 if the fade exceeds `t_target`, 0 otherwise.

## Baselines, implemented and validated before the model

Phase order is not negotiable in this portfolio: the analytic result comes first.

| Baseline | What it is | Uses channel config? | Uses state? |
|---|---|---|---|
| `analytic_configured` | `exp(-t/MFD)` with `MFD` from the **exact** sampled-Gauss-Markov mean fade duration, equation (15) of `codedfade.fade` | yes | no |
| `analytic_observed` | the same expression with `MFD` estimated from a 32-sample trailing window of the amplitude only | no | partly (as a scalar) |
| `empirical_constant` | the measured exceedance frequency on the training paths | no | no |

`analytic_configured` is given the true scintillation index and correlation time,
which the learned model never sees. That is deliberate: it is the strongest honest
form of the analytic result. `empirical_constant` is the baseline that matters for
any claim about state, because beating a miscalibrated constant proves nothing.

## Architecture

- `sklearn.ensemble.RandomForestClassifier`, 160 trees, `max_depth=8`,
  `min_samples_leaf=20`, `random_state=0`, `n_jobs=1`.
- PyTorch is unavailable in this build environment. Everything is scikit-learn and
  numpy.
- **Uncertainty output:** the reported confidence is the standard deviation across
  the 160 per-tree predicted probabilities, not a point estimate. `max_depth` is
  capped at 8 and leaves hold at least 20 samples precisely so that per-tree
  probabilities do not collapse to 0/1, which would make the spread meaningless.

### Features (10, all causal)

The window is the 32 samples **ending at the crossing sample**; nothing after the
crossing enters a feature. `tests/test_predictor.py::TestFeatures::test_features_use_only_causal_information`
verifies this by truncating the record immediately after a crossing and checking
that the feature row is bit-identical.

| Feature | Gini importance |
|---|---|
| `log_depth_at_crossing` | 0.402391 |
| `descent_slope_1` | 0.178402 |
| `log_depth_previous` | 0.095882 |
| `window_min_log_margin` | 0.095821 |
| `descent_slope_4` | 0.065231 |
| `window_log_std` | 0.047877 |
| `window_lag1_autocorr` | 0.047786 |
| `analytic_observed_mfd_samples` | 0.027298 |
| `window_fraction_below` | 0.025261 |
| `analytic_observed_exceedance` | 0.014050 |

`analytic_observed_exceedance` is the `analytic_observed` baseline's own output,
handed to the model as a feature. The model therefore starts from the baseline's
answer and can only improve on it by extracting information from the state.

## Dataset and split

See `DATASET_CARD.md`. Summary: seeded lognormal Gauss-Markov amplitude paths,
`SI = 0.6`, `tau = 2e-4` s, `fs = 1e6` Hz, threshold amplitude 0.6,
`t_target = 14` samples, 200 000 samples per path.

- Training: path seeds 0–23 → **39 687** crossing events, exceedance frequency
  **0.199486**.
- Held out: path seeds 100–111 → **19 900** crossing events, exceedance frequency
  **0.198693**.
- **Split strategy: by path seed.** Every event from one realisation is in exactly
  one side of the split, so a model cannot see a neighbouring fade from a path it
  trained on. Splitting events at random would leak.

## Metrics on the held-out seeds

| Predictor | Brier | log loss | ROC AUC | ECE (10 bins) |
|---|---|---|---|---|
| `analytic_configured` | 0.199594 | 0.591088 | 0.5000 | 0.200947 |
| `analytic_observed` | 0.198676 | 2.568769 | 0.4983 | 0.198648 |
| `empirical_constant` | 0.159215 | 0.498588 | 0.5000 | 0.000793 |
| **learned** | **0.154194** | **0.484009** | **0.6175** | **0.004311** |

### What this says, stated plainly

1. **The exponential closure of the level-crossing result is wrong by a factor of
   two.** At `t = MFD` it predicts `P(T > t) = 0.399641` against an observed
   **0.198693**, a ratio of **2.0113**. Fade durations on this channel are not
   exponentially distributed: the median fade is 3 samples against a mean of 15.5.
   This is the most useful finding in the product and it is a **defect of the
   analytic closure**, published as measured. Equation (16) was not retuned and was
   not removed.
2. **Against the honest baseline the learned model's win is small.** Brier
   0.154194 against the empirical constant's 0.159215 is a **3.15 %** relative
   improvement. That is the number to quote.
3. **State does carry information.** ROC AUC **0.6175** against 0.5000 for every
   state-blind predictor. The depth below the threshold at the crossing sample
   carries 40 % of the model's importance: a crossing that plunges deep is the start
   of a longer fade. This is a real effect and the only thing the model adds.
4. **Calibration.** ECE 0.004311 over 10 bins. The reliability table is good to
   about 0.02 up to a predicted 0.45 and degrades above it, where the bins are
   nearly empty (85 events in bin 5, 8 in bin 6, none above). **The model should not
   be used above a predicted probability of about 0.45**, because there is no
   evidence there.

### Reliability table (held-out)

| bin | mean predicted | observed | count | predicted − observed |
|---|---|---|---|---|
| 0 | 0.093189 | 0.121711 | 304 | −0.028522 |
| 1 | 0.150960 | 0.153634 | 11957 | −0.002674 |
| 2 | 0.235470 | 0.237998 | 5395 | −0.002529 |
| 3 | 0.339812 | 0.331541 | 1674 | +0.008271 |
| 4 | 0.434964 | 0.412998 | 477 | +0.021966 |
| 5 | 0.541788 | 0.482353 | 85 | +0.059435 |
| 6 | 0.640428 | 0.375000 | 8 | +0.265428 |
| 7–9 | — | — | 0 | — |

## Uncertainty output, assessed

| Quantity | Value |
|---|---|
| mean per-tree sigma | 0.066564 |
| median | 0.060739 |
| maximum | 0.205373 |
| events with \|p − y\| > 0.5 | 3959 of 19900 |
| mean sigma on those | 0.073262 |
| mean sigma on the rest | 0.064900 |
| **ratio** | **1.1289** |

The ensemble is more uncertain where it is wrong, but **only by 13 %**. That is a
weak signal. It is enough to rank cases but not enough to gate a decision on a
threshold, and it is reported as such rather than presented as a calibrated
interval.

## Failure cases

- **Above a predicted 0.45** there is almost no held-out evidence (93 events in
  total across bins 5 and above) and bin 6 is off by 0.27. Do not use the model
  there.
- **Outside the trained regime.** The model was trained at one scintillation index
  and one correlation time. It has **not** been shown to transfer to another
  turbulence condition, another kernel, or the gamma-gamma marginal. No transfer
  claim is made and none was tested.
- **Gaussian kernel.** The training data uses the `exp` kernel only. The `gauss`
  kernel has different short-fade statistics and the model was not evaluated on it.
- **Deep thresholds.** At a threshold much below 0.6 the crossing events become
  rare and the trailing-window estimators degrade; the model was not evaluated
  there.
- **The weak uncertainty signal above** means an out-of-distribution input will not
  reliably announce itself.

## Reproducibility

```bash
cd validation
PYTHONPATH=../src python3 train_model.py                # regenerates the artefact
PYTHONPATH=../src python3 validate_ai_vs_baseline.py    # regenerates every number here
```

Fixed seeds: channel path seeds 0–23 (train) and 100–111 (test); forest
`random_state=0`. The artefact is `validation/fade_exceedance_model.joblib`, a
joblib archive containing the fitted forest, the feature names, the training
configuration and the baseline constants. No `.pt`, `.pth`, `.ckpt` or `.onnx` file
is produced or committed.

## Compute used

Measured on the shared build host (2 cores, 7.8 GiB, shared with four sibling
processes), from the timings printed by `validate_ai_vs_baseline.py`:

| Stage | Wall-clock |
|---|---|
| dataset build, 36 paths of 200 000 samples | 19.1 s |
| forest fit, 160 trees | 7.8 s |
| full benchmark script | about 32 s |

Well inside the three-minute budget this portfolio imposes on any training run.

## Ethical and safety limits

- Research-grade. **Not flight-qualified, not certified, not approved for
  operational aerospace use.** This model estimates a statistic of a simulated
  channel; it has never seen a real optical link.
- It must not be used to set a link margin or an availability figure that anyone
  depends on. Its training data is synthetic, generated by the same package that
  evaluates it, which means systematic error in the channel model is invisible to
  the benchmark by construction.
- Validation level 3, hardware-pending. The words "Level 4" appear in this
  repository only to state what is missing: measured timing and resource use from a
  Jetson Orin Nano.
