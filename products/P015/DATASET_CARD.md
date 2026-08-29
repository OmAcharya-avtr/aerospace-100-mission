# Dataset Card — LinkSwitch simulated telemetry v0.1.0

## Summary

LinkSwitch has **no committed dataset file**. Every telemetry episode used
anywhere in this package (tests, examples, validation) is generated on
demand, deterministically, by
`linkswitch.scenario.generate_telemetry(config, n_steps, seed)`, and is
regenerated fresh every time a script runs — nothing is cached to disk.
This satisfies the "data/model artifacts over 1 MB must not be committed"
constraint trivially (nothing is committed at all) and keeps every number
in `validation/VALIDATION.md` reproducible from source.

## THE DATA IS ENTIRELY SIMULATED

**No field-measured turbulence, scintillometer, rain-gauge, or RF-link
telemetry is used anywhere in this package.** Every irradiance and rain-rate
value is drawn from the models documented in `src/linkswitch/optical.py`
and `src/linkswitch/rf.py`:

- Optical irradiance: an AR(1) (discrete Gauss-Markov) process on
  standardised log-irradiance, with lognormal marginals parameterised by a
  scintillation index `sigma_i2` (Andrews & Phillips 2005) and a
  configurable coherence time. **The temporal AR(1) structure itself is an
  engineering approximation**, not derived from a measured or published
  turbulence temporal power spectrum — see the module docstring.
- Rain occurrence: a two-state (clear/rain) Markov chain parameterised by a
  stationary rain probability and mean event duration, both freely
  configurable, purely synthetic (not derived from any measured
  rain-event-duration climatology).
- Rain rate given raining: lognormal, median and spread configurable.
- Rain attenuation: `gamma_R = k * R^alpha` (ITU-R P.838 functional form)
  with **illustrative, unverified** `(k, alpha)` defaults, times a
  simplified effective-path-length reduction factor loosely motivated by
  the ITU-R P.618 concept (not the exact P.618-13 procedure).

## Generation

```python
from linkswitch import ScenarioConfig, generate_telemetry
cfg = ScenarioConfig()  # or customise OpticalParams / RFParams / SwitchCost
tel = generate_telemetry(cfg, n_steps=2000, seed=0)
```

Every field in `Telemetry` (`irradiance`, `opt_available`,
`rain_rate_mm_hr`, `rf_atten_db`, `rf_available`) is derived deterministically
from `(config, n_steps, seed)` — see
`tests/test_scenario.py::TestGenerateTelemetry::test_seeded_reproducibility`.
The optical and RF sub-streams are driven by independent RNG streams spawned
from one `numpy.random.SeedSequence(seed)` (turbulence-induced optical
scintillation and rain-induced RF fading are physically distinct phenomena
with no shared driver here).

## Schema (`Telemetry`, all arrays length `n_steps`)

| Field | Units | Description |
|---|---|---|
| `irradiance` | dimensionless (mean-normalised, E[I]=1) | Optical irradiance I(t) |
| `opt_available` | bool | `irradiance >= tau_phys` (physical outage threshold from link margin) |
| `rain_rate_mm_hr` | mm/hr | Simulated instantaneous rain rate (0 when not raining) |
| `rf_atten_db` | dB | Simulated RF path attenuation due to rain (0 when clear) |
| `rf_available` | bool | `snr_clear_db - rf_atten_db >= snr_min_db` |

## Training sets used in this build

Every training set consumed by `linkswitch.learn.train_outage_predictor`
during validation (`validation/policy_comparison_ci.py`,
`validation/horizon_sensitivity.py`) is 10–15 episodes of 500 steps each
(5000–7500 rows before the trailing `horizon`-length drop per episode),
generated with an explicit, documented seed offset (e.g. seeds `40000+i`,
`70000+i`) — see each script for the exact call. These sets are small by
design, sized to the 2-CPU-core / sub-2-minute compute budget, not to
convergence; see MODEL_CARD.md Failure cases for the consequences (notably
the horizon-20 degeneracy).

## Coverage of the two headline scenarios

| Parameter | Mild (package default) | Moderate |
|---|---|---|
| `sigma_i2` | 0.25 | 0.4 |
| `coherence_steps` | 5.0 | 4.0 |
| `margin_db` | 6.0 | 4.0 |
| Resulting optical outage fraction (measured, 100k-step run) | ~0.36% | ~9% |
| `rf.rate_mbps` / `optical.rate_mbps` | 150 / 1000 | 150 / 1000 |

Both scenarios use the same RF configuration (`RFParams()` defaults:
`p_rain=0.04`, giving RF availability ≈ 99.98%, measured in
`tests/test_scenario.py`).

## Known limitations and gaps

- No spatial dimension: single-link, single-point telemetry only. No
  network-level or multi-link correlation.
- No Cn²(h) altitude profile — this is a horizontal/point-to-point link
  model, not a slant-path or satellite-downlink model.
- Gamma-gamma irradiance sampling exists (`optical.sample_gamma_gamma_irradiance`)
  but is i.i.d. only — **not** wired into `generate_telemetry`, so no
  moderate/strong-turbulence *time series* data exists in this package.
- Rain-rate-when-raining and the ITU-R P.838 `(k, alpha)` coefficients are
  illustrative defaults, not fitted to any specific real climate zone or
  verified against the current ITU-R tables.
- Training sets used in validation (10–15 episodes × 500 steps) are small;
  no convergence study was performed on training-set size.
- No temporal frozen-flow / wind-speed modelling; the AR(1) coherence time
  is a free knob, not derived from a wind-speed physical model.

## Intended and unintended use

**Intended:** research and teaching; demonstrating and regression-guarding
the switching-policy simulation and comparison pipeline; a reproducible
reference implementation of classical vs. learned link-switching policies.

**Not intended:** training or evaluating any model used for operational
hybrid RF/FSO link switching decisions, availability guarantees, or any
flight or mission-critical planning. This dataset (generator) is not
certified for operational flight use.
