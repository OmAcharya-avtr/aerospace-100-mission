# Model Card — LinkSwitch Learned Outage Predictor v0.1.0

**This model is not certified for operational flight use.**

**The fading (optical irradiance) and rain-fade (RF attenuation) processes
underlying every training and evaluation episode are SIMULATED, not
measured — see `DATASET_CARD.md`. No field-measured turbulence,
scintillometer, rain-gauge, or RF-link data is used anywhere in this
package.**

## Problem

Predict, from recent optical-link telemetry, whether the free-space optical
(FSO) channel of a hybrid RF/FSO link will suffer an outage (irradiance
below the link-margin threshold) within the next `H` steps ("imminent
outage"), with a confidence output, so a switching policy can proactively
move traffic to the RF channel *before* the optical link actually fails
rather than reacting after the fact.

## Classical baselines (implemented first)

Two classical policies are implemented, tested, and validated before any
learned component: `linkswitch.policies.FixedThresholdPolicy` (switch to RF
whenever current irradiance falls below a single threshold `tau`) and
`linkswitch.policies.HysteresisPolicy` (two thresholds `tau_low < tau_high`
to prevent chatter). The optimal fixed threshold in the frictionless limit
is derived in closed form and matches the physical outage threshold exactly
(`validation/VALIDATION.md` §V1); with realistic switch downtime, this
matching-a-single-threshold approach chatters and is beaten by hysteresis
(§V1c). Both baselines are benchmarked against the learned policy on the
*same* seeded telemetry (`validation/policy_comparison_ci.py`).

## Architecture

- `sklearn.ensemble.RandomForestClassifier` (default `n_estimators=40`,
  `max_depth=4`, `n_jobs=1`) inside an `sklearn.pipeline.Pipeline` with a
  `StandardScaler` (`linkswitch.learn.OutagePredictor`).
- Features (`linkswitch.features.rolling_features`, 5 columns, all causal /
  no lookahead): `ln I(t)`, trailing rolling mean of `ln I`, trailing
  rolling std of `ln I`, trailing rolling min of `ln I`, and a slope
  `(ln I(t) - ln I(t-window)) / window`. Default `window=6` or `8`
  (scenario-dependent, see `validation/VALIDATION.md`).
- Target: binary label, "does irradiance drop below `tau_phys` at any point
  in the next `horizon` steps?" (`linkswitch.features.label_imminent_outage`).
- A degenerate single-class fallback (constant predictor) is used when a
  training set contains no outages at all, documented in
  `linkswitch.learn.OutagePredictor.fit`, rather than raising or silently
  producing an ill-defined classifier.

## Dataset

Entirely synthetic, generated on demand by
`linkswitch.scenario.generate_telemetry` — see `DATASET_CARD.md`. Training
sets used in validation: 15 episodes × 500 steps (7500 rows before the
per-episode `horizon`-length trailing drop). No experimental, proprietary,
personal, or field-measured data is involved anywhere.

## Training procedure

```bash
python3 validation/policy_comparison_ci.py   # trains + evaluates both scenarios
python3 validation/horizon_sensitivity.py    # trains one model per horizon value
```

- `linkswitch.learn.train_outage_predictor(telemetries, tau_phys, horizon,
  window, random_state=0)` concatenates causal features + labels across
  episodes (dropping each episode's last `horizon` steps, whose labels are
  truncated by the end of the series) and fits the pipeline.
- No hyperparameter search was performed against a held-out test set; the
  architecture (`n_estimators=40, max_depth=4`) was fixed a priori as "small
  enough to be fast and to resist overfitting a rolling 5-feature window."
- Fit time: well under 1 s per model on 2 CPU cores (15 episodes × 500 steps).
- `LearnedPolicy` is evaluated on *fresh* seeded telemetry never seen during
  training (`compare_policies` generates new episodes per replicate with
  seeds disjoint from the training seed range).

## Metrics (actual measured runs)

See `validation/VALIDATION.md` §V2 for the full paired-CI comparison tables
and §V3 for the horizon sweep. Headline:

| Scenario | Winner (throughput) | Learned vs. hysteresis |
|---|---|---|
| Mild (package defaults) | learned (point estimate) | **statistical tie** — 95% CIs overlap almost entirely |
| Moderate (tighter margin, more turbulence) | **hysteresis** | **learned loses**, and loses even to the naive fixed-threshold baseline |

**The learned policy does not beat the classical hysteresis baseline in
either tested scenario.** This is reported as measured; no tolerance was
loosened and no scenario or hyperparameter was retuned to flip this result.

## Uncertainty / confidence output

`LearnedPolicy.outage_confidence(telemetry)` (backed by
`OutagePredictor.predict_proba`) returns, for every step, the trained
RandomForest's class-1 vote fraction — the fraction of the 40 trees voting
"imminent outage." This is the required confidence output for the learned
policy (used directly as the proactive-switch trigger:
`confidence_threshold`, default 0.5). **It is a raw ensemble vote
fraction, not a formally calibrated probability** (no Platt scaling /
isotonic calibration is applied); no calibration curve has been measured
in this build. Treat it as a relative signal ("the model is more or less
confident"), not a validated probability of an actual future outage.

## Failure cases

- **Over-triggering under higher scintillation.** In the "moderate" scenario
  (`sigma_i2=0.4`), the learned policy switches *more* often than the naive
  fixed-threshold baseline (158.7 vs. 178.2 switches/episode — both far
  above hysteresis's 99.3) while delivering *less* throughput. The rolling
  5-feature window produces false positives near the noisy threshold region
  under stronger turbulence, and the "return to optical" rule (which
  requires the model's own confidence to also clear — see
  `policies.py` docstring) does not compensate.
- **Horizon degeneracy at large H.** At `horizon=20` in the moderate
  scenario, throughput collapses to 168 Mb/s (near the RF-only floor of
  150 Mb/s) because almost every training window gets labelled "imminent
  outage," so the model predicts outage almost unconditionally and the
  policy parks on RF nearly permanently. See `validation/VALIDATION.md` §V3.
- **No extrapolation guard.** Features are raw log-irradiance statistics
  with no explicit domain check; querying telemetry far outside the
  training scenario's `(sigma_i2, coherence_steps, margin_db)` combination
  will produce a confident-looking but untested prediction.
- **Single-class training fallback.** If a training set happens to contain
  zero outage episodes (possible in the mild scenario with few episodes),
  the model degenerates to an always-optical constant predictor — silently
  correct in that case but untested against any real signal.
- **No calibration.** As noted above, `predict_proba` is an uncalibrated
  ensemble vote fraction; do not use it as a probability for any downstream
  risk calculation.

## Reproducibility

Exact commands and seeds:

```bash
python3 validation/analytic_threshold_check.py   # no ML, deterministic
python3 validation/policy_comparison_ci.py       # training seed 40000+i, random_state=0
python3 validation/horizon_sensitivity.py        # training seed 70000+i, random_state=0
python3 -m pytest tests/ -q                      # 201 tests
```

`OutagePredictor(random_state=r)` seeds both the `RandomForestClassifier`
and, indirectly (via the same value passed through
`train_outage_predictor`), nothing else — telemetry generation seeds are
independent and explicit at every call site. Identical `(telemetries, seeds,
random_state)` gives bit-identical `predict_proba` output
(`tests/test_learn.py::TestOutagePredictorBasics::test_seeded_reproducibility`).

## Compute used

Whole build fits well inside the 2-CPU-core / sub-2-minute-per-script
budget: `analytic_threshold_check.py` 6.6 s, `policy_comparison_ci.py`
3.7 s, `horizon_sensitivity.py` 8.3 s, full test suite ~25 s (201 tests),
examples ~6 s combined. No GPU, no PyTorch. `n_jobs=1` throughout.

## Ethical and safety limits

- Research and educational use only. All telemetry is synthetic; no
  personal data, no dual-use content beyond standard link-engineering
  simulation.
- Predictions must not be used to make go/no-go decisions for any
  operational optical or hybrid RF/FSO link, terrestrial or space-to-ground.
- Any safety-relevant use would require calibrated confidence, a converged
  (much larger, longer-horizon, multi-scenario) training campaign,
  validation against measured link telemetry, and independent review —
  none of which this product has.
- **This model is not certified for operational flight use.**
