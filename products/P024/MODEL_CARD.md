# Model Card — DetumbleSim learned B-dot gain scheduler v0.1.0

**This model is not certified for operational flight use.**

**Every scenario, orbit, magnetic-field value and telemetry sample used to
train or evaluate this model is SIMULATED — see `DATASET_CARD.md`. No flight
telemetry, no measured magnetometer record and no on-orbit detumble log is
used anywhere in this package.**

## Problem

Choose the B-dot gain `k` for a magnetorquer detumble. `k` trades detumble
time against coil energy and torquer saturation, and the best value depends on
the vehicle's inertia, its dipole limit, its initial rate and its orbit
geometry. The model predicts, from quantities a spacecraft actually has
onboard, a multiplicative correction to a single hand-tuned baseline gain,
together with a confidence.

## Classical baselines, implemented and validated first

Three classical gain rules were implemented, tested and validated **before**
any learned component, and all three are benchmarked against the learned model
on identical paired scenarios (`validation/learned_vs_fixed_ci.py`):

1. **`FixedGainPolicy`** — one constant gain for every vehicle, tuned on the
   training set by grid search (`k = 7.196857e+04 A m² s/T`, training mean
   cost 0.6525).
2. **`SizedGainPolicy`** — the rule an ADCS engineer would write:
   `k = c·m_max/(⟨|B|⟩·ω_est)` with `ω_est` from the observable rate proxy
   `|dB/dt|/|B|` over the first 40 magnetometer samples. Estimator and
   coefficient jointly tuned on training (`max`, `c = 2.3784`, training mean
   cost 0.5689).
3. **`PowerLawGainPolicy`** — a three-coefficient least-squares fit of the
   training oracle gains,
   `log10 k = 6.2816 + 0.6978 log10(m_max) + 0.4397 log10(j)`, RMS residual
   0.3050 dex. No machine learning; three numbers from
   `numpy.linalg.lstsq`. **This is the strong competitor.**

An **oracle** (best constant gain per scenario, by exhaustive grid search) is
also computed on the training set as an upper bound: it reaches mean cost
0.4926 against the fixed gain's 0.6525, so the headroom the learned model
could in principle capture is 24.51 %.

## Architecture

- `sklearn.ensemble.RandomForestRegressor(n_estimators=200, max_depth=6,
  min_samples_leaf=4, random_state=0, n_jobs=1)`
  (`detumblesim.scheduler.GainScheduler`). No PyTorch, no GPU.
- **Inputs, eight features** (`detumblesim.features.TelemetryWindow`), all
  computed from a trailing 60-step window of magnetometer samples plus two
  parameters the flight software already knows about its own vehicle:

  | # | Feature | Source |
  |---:|---|---|
  | 0 | `log10(median |dB/dt| / |B|)` [log10 rad/s] | magnetometer |
  | 1 | `log10(mean |B|)` [log10 T] | magnetometer |
  | 2 | slope of `log10(rate proxy)` per 1000 s | magnetometer |
  | 3 | saturation duty fraction | the controller's own clip flag |
  | 4 | `std(|B|)/mean(|B|)` | magnetometer |
  | 5 | `log10(per-axis dipole limit)` [log10 A m²] | known hardware |
  | 6 | `log10(nominal inertia scale)` [log10 kg m²] | known from CAD |
  | 7 | `log10(1 + elapsed control steps)` | mission elapsed time |

  Nothing uses the true body rate, the true attitude, or any quantity a
  spacecraft would not have.
- **Target**: `y = log10(k_oracle(scenario) / k_fixed)`, one label per
  scenario applied to every harvested window of that scenario.
- **Output**: a gain and a confidence,
  `k = k_fixed · 10^(confidence · clip(ŷ, −1, +1))`. The `±1 dex` clamp is a
  hard safety limit, not a tuning knob.

## Dataset

Entirely synthetic. Twenty training scenarios (seeds 1000–1019) and forty
held-out scenarios (seeds 5000–5039), drawn by
`detumblesim.scenarios.sample_scenario`, giving 1068 feature rows × 8 features
with labels spanning `[−0.2857, +1.1429]` dex. Regenerate with
`python data/generate_dataset.py`. See `DATASET_CARD.md` for the sampling
ranges and their limitations.

## Test-split strategy

Scenario seeds are disjoint: the model never sees a held-out scenario during
training, and every row of a scenario stays on the same side of the split
(there is no window-level leakage between train and test). The fixed gain, the
sized-rule estimator and coefficient, and the power-law coefficients are all
tuned on the **training** scenarios only. The held-out set is simulated
exactly once per policy; no held-out result was used to reselect a
hyperparameter, a seed or a tolerance.

## Training procedure

```bash
python validation/learned_vs_fixed_ci.py    # trains and evaluates end to end, ~98 s
python data/generate_dataset.py             # writes the same dataset to CSV, ~23 s
```

Simulation settings for every run: `duration_s=23000`, `control_dt_s=2.0`,
`substeps=1`, target rate 1.0 deg/s, no magnetometer noise. Gain grid
`np.geomspace(1e4, 1e6, 8)`. No hyperparameter search was performed against
the held-out set; the forest size was fixed a priori as "small enough to fit
in a second on two cores, large enough for a meaningful ensemble spread".

## Metrics

Cost `= t_detumble/T_orbit + w·∫|m|²dt/(m_max²·T_orbit)`, reported at
`w = 0` (time only), `w = 0.5` (default) and `w = 2`, all re-scored from the
same simulations. Paired Student-t 95 % intervals on the per-scenario
difference (`detumblesim.metrics.paired_difference_ci`).

Held-out, 40 paired scenarios, `w = 0.5`:

| policy | cost (95 % CI) | time [orbits] | energy term | failures | steps saturated | mean gain used |
|---|---|---|---|---:|---:|---|
| fixed | 1.170 [0.788, 1.552] | 0.885 [0.637, 1.133] | 0.285 [0.130, 0.441] | 0 | 16.97 % | 7.197e+04 |
| sized | 1.422 [0.795, 2.050] | 1.131 [0.585, 1.677] | 0.292 [0.176, 0.408] | **2** | 25.43 % | 2.380e+05 |
| powerlaw | 0.999 [0.622, 1.376] | 0.601 [0.410, 0.793] | 0.398 [0.205, 0.590] | 0 | 42.60 % | 1.851e+05 |
| learned | 1.061 [0.690, 1.433] | 0.680 [0.488, 0.871] | 0.382 [0.193, 0.571] | 0 | 32.46 % | 1.742e+05 |

Paired differences that this experiment **resolves** (interval excludes zero):

| w | comparison | difference | 95 % CI | who wins |
|---:|---|---:|---|---|
| 0.0 | learned − fixed | −0.206 | [−0.308, −0.103] | learned |
| 0.0 | powerlaw − fixed | −0.284 | [−0.402, −0.165] | powerlaw |
| 0.0 | learned − powerlaw | +0.078 | [+0.010, +0.146] | **powerlaw** |
| 0.5 | learned − fixed | −0.109 | [−0.196, −0.021] | learned |
| 0.5 | powerlaw − fixed | −0.171 | [−0.273, −0.069] | powerlaw |
| 2.0 | learned − fixed | +0.181 | [+0.038, +0.324] | **fixed** |
| 2.0 | powerlaw − fixed | +0.166 | [+0.010, +0.322] | **fixed** |

Paired differences this experiment **cannot resolve**: `sized − fixed` at all
three weights, `learned − powerlaw` at `w = 0.5` (+0.063 [−0.010, +0.135]) and
`w = 2` (+0.016 [−0.094, +0.125]), and `learned − sized` at `w = 0.5` and
`w = 2`.

### The honest summary

The learned scheduler beats the naive fixed gain on detumble time and loses to
it once coil energy is weighted heavily. **It does not beat the
three-coefficient log-linear regression at any weight tested**, and the
regression is fitted to the same twenty scenarios with three numbers and no
machine learning. If you want a better B-dot gain than one constant, fit the
power law; the RandomForest adds complexity and no measured benefit.

## Uncertainty / confidence output

The spread of the individual tree predictions is used as an uncertainty
estimate:

```
sigma      = std over trees of tree_i(x)          [dex]
confidence = 1 / (1 + sigma / 0.25)               in (0, 1]
k          = k_fixed * 10 ** (confidence * clip(y_hat, -1, +1))
```

Low confidence shrinks the correction toward the classical baseline, so an
out-of-distribution input degrades gracefully instead of commanding an
arbitrary gain. Over 2586 gain updates on the held-out set the confidence was
mean 0.9799, min 0.6482, max 1.0000.

**This is an ensemble-spread heuristic, not a calibrated predictive
interval.** No coverage calibration was performed and no reliability diagram
exists. A confidence of 0.98 means the trees agree, not that the prediction is
right 98 % of the time. Treat it as a relative disagreement measure only.

## Failure cases

1. **It is a lookup, not a scheduler.** 77.15 % of impurity importance is on
   `log10(max dipole)` and 22.84 % on `log10(inertia)`; the six time-varying
   magnetometer features carry **0.0000** between them. Nothing about the
   measured rate or field history changes the gain it picks.
2. **It loses to the fixed gain under an energy-weighted objective**
   (`w = 2`: +0.181 [+0.038, +0.324]). It buys speed with a ~2.4× larger gain
   that saturates the torquers on 32.5 % of steps against 17.0 %.
3. **It loses to the three-coefficient power law on detumble time**
   (`w = 0`: +0.078 [+0.010, +0.146]).
4. **Its training target caps it.** The label is the best *constant* gain per
   scenario, so a perfect learner reproduces the constant-gain oracle and
   nothing better. A genuinely time-varying optimal gain was never computed.
5. **Label resolution is coarse.** The oracle gains come from an 8-point grid
   spanning three decades (a factor of 1.93 between neighbours), and only 6 of
   the 8 grid values were ever selected, so the labels are quantised.
6. **No out-of-distribution guard beyond the clamp.** Querying with an inertia
   or dipole limit outside the sampled ranges returns a plausible-looking
   untested gain, shrunk only by whatever confidence the trees happen to
   report.
7. **Twenty training scenarios is a very small sample** for an eight-feature
   model. The power-law fit's 0.3050 dex residual over the same twenty points
   is a fair indication of how much of the variance neither model explains.

## Reproducibility

```bash
python -m pytest tests/ -q                  # 305 passed in 15.86s
python validation/learned_vs_fixed_ci.py    # the full benchmark, ~98 s
python data/generate_dataset.py             # the training CSVs + sha256 manifest
```

Seeds: training scenarios 1000–1019, held-out scenarios 5000–5039,
`random_state=0` for the RandomForest. Identical `(seeds, random_state)`
gives bit-identical predictions, checked by
`tests/test_scheduler.py::TestFit::test_is_reproducible_for_a_fixed_seed`.
`data/dataset_manifest.txt` records a SHA-256 of each generated CSV.

## Compute used

2 CPU cores, no GPU, `n_jobs=1` throughout. The full training and evaluation
campaign (20 × 8 oracle sweeps, 15 sized-rule tunings × 20, 20 training runs,
40 × 4 held-out runs, plus the forest fit) takes **98.3 s** end to end. The
forest fit itself is well under a second. Peak memory stays under 200 MB.
The whole model is 200 trees of depth ≤ 6.

## Ethical and safety limits

- **This model is not certified for operational flight use.** It must not be
  used to set a B-dot gain on a real spacecraft, to size a real magnetorquer,
  or to support any go/no-go decision.
- It was trained and evaluated entirely inside a simulation whose magnetic
  field model has a measured worst-case error of 71.86 % against IGRF-14
  (`validation/VALIDATION.md`, V1-A2, which FAILED its tolerance), whose orbit
  is unperturbed, and which models no environmental disturbance torque. Its
  outputs inherit all of that.
- It has no failure detection. If a scenario falls outside the training
  distribution it will still return a gain and a high confidence.
- The recommended default in this repository is the classical
  `PowerLawGainPolicy` or a hand-tuned `FixedGainPolicy`, not this model.
- No personal, proprietary or export-controlled data is involved anywhere:
  every input is generated by a committed seeded script.
