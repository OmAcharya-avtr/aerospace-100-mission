# Model Card — PassPlanner pass-success (availability) model

**This model is not certified for operational flight use.**

| Field | Value |
|---|---|
| Model name | `passplanner.PassSuccessModel` |
| Version | 0.1.0 |
| Type | Bagged ensemble of gradient-boosting classifiers (scikit-learn) |
| Task | Binary probabilistic prediction: will an optical pass be usable (clear)? |
| Output | Probability in [0, 1] **plus** an ensemble-spread uncertainty |
| Training data | 100 % synthetic (see `DATASET_CARD.md`) |
| Licence | Apache-2.0 |

## 1. Problem

Optical ground-station contacts fail mainly because of cloud. The scheduler
maximises *expected* delivered data, `rate × duration × p_clear`, so it needs
a probability of success per pass, not a yes/no call. A miscalibrated
probability directly biases the schedule, so calibration matters more here
than raw discrimination.

## 2. Baseline (implemented first)

`passplanner.ClimatologyBaselineModel` — predicts the station's
climatological monthly clear-sky prior and ignores every weather feature.
This is the classical approach used when no forecast is available, and it is
the reference the ML model must beat. Both models are trained on the same
training split and scored on the same held-out test split.

## 3. Architecture

* 5 members, each a `sklearn.ensemble.GradientBoostingClassifier`
  (`n_estimators=150`, `max_depth=2`, `learning_rate=0.1`).
* Each member is fitted on an independent bootstrap resample of the training
  set; member random states are drawn from a single seeded
  `numpy.random.default_rng(seed)`, so the whole ensemble is reproducible
  from one integer.
* Prediction = mean of member probabilities.
* Uncertainty = standard deviation of member probabilities.

Shallow trees plus a small ensemble were chosen to fit the compute budget
(< 3 min on 2 cores) and to keep the model from overfitting a 7-feature
problem.

## 4. Input features

The seven features listed in `DATASET_CARD.md` (climatological prior,
relative humidity, IR cloud fraction, pressure anomaly, wind speed, and the
month encoded as sin/cos). Inputs are validated: wrong shape, wrong column
count or non-finite values raise `ValueError`.

## 5. Training procedure and reproducibility

```bash
cd products/P004
python validation/validate_availability_model.py
```

Exact configuration used for the reported metrics:

| Item | Value |
|---|---|
| Training set | `generate_dataset(8000, seed=20260301)` |
| Test set | `generate_dataset(4000, seed=20260302)` |
| Model seed | 7 |
| Ensemble | 5 members |
| Compute | 2 CPU cores, no GPU |
| Training time | 5.9 s (measured) |
| Framework | scikit-learn (PyTorch not used / not available) |

`tests/test_mlmodel.py` asserts bit-identical predictions from two models
trained with the same seed.

## 6. Test-split strategy

Train and test are independent i.i.d. draws from the same generative process
with different seeds — equivalent to a random hold-out split, with no shared
samples and no leakage. No hyperparameter search was run against the test
split; the hyperparameters above were fixed a priori from the compute budget.

## 7. Metrics (held-out synthetic test set, n = 4000)

| Model | Brier ↓ | Log loss ↓ | ROC AUC ↑ | ECE (10 bins) ↓ |
|---|---:|---:|---:|---:|
| Climatology baseline | 0.2328 | 0.6585 | 0.6358 | 0.0493 |
| **This model** | **0.1687** | **0.5086** | **0.8216** | **0.0179** |
| Oracle `p_true` (floor) | 0.1634 | 0.4961 | 0.8325 | 0.0180 |

* Brier improvement over the baseline: **27.51 %**.
* Distance to the irreducible floor: 0.0053 Brier.
* Reliability diagram: `validation/calibration_curve.png`; worst calibration
  bin deviates by 0.036 (bin 0.40–0.50, n = 367).

These figures characterise behaviour **on synthetic data only**.

## 8. Uncertainty / confidence output

`predict_with_uncertainty(x)` returns `(p_mean, p_std)`. `p_std` is the
spread of the bootstrap members: an *epistemic* signal that flags inputs the
ensemble disagrees about (feature combinations that are rare in training).

Measured on the test split: mean 0.0321, median 0.0289, p95 0.0670, max
0.2190; correlation with the realised error |p_pred − p_true| is **+0.2584**.

Read this correctly: the correlation is positive but weak, so `p_std` is a
*triage flag*, **not** a calibrated error bar. It also does **not** capture
the irreducible randomness of the weather — that is what `p_mean` itself
expresses.

## 9. Failure cases and out-of-scope use

* **Out-of-distribution inputs.** Real weather features (different scaling,
  correlations, seasonality) are outside the training distribution; the model
  will produce confident-looking numbers with no validity. `p_std` will not
  reliably catch this.
* **Extreme probabilities.** Predictions above ~0.95 and below ~0.05 sit in
  sparsely populated bins (149 samples in the 0.0–0.1 bin); calibration there
  is the least well determined.
* **No spatial or temporal correlation.** Consecutive passes at the same
  station are predicted independently; real cloud persistence would make
  those outcomes strongly correlated, so a schedule built from these
  probabilities understates the risk of losing a whole night.
* **Not a forecast model.** It has no access to observation time, lead time
  or NWP fields. For real planning, supply your own measured availability via
  `ForecastAvailability` instead.
* **Not for flight/ops decisions.** Do not use it to commit contact plans,
  size onboard storage, or justify link availability claims.

## 10. Ethical and safety limits

No personal data is involved. The material risk is over-trust: an
availability number that looks authoritative but is derived from invented
data. Every artefact (plots, CLI output, cards) is labelled accordingly, and
the shipped station priors are explicitly marked as fictional placeholders.

**This model is not certified for operational flight use.**
