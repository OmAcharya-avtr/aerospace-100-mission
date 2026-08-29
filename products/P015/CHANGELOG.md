# Changelog

All notable changes to LinkSwitch are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-08-29

Initial release. Status: TESTING.

### Added

- **Optical channel** (`linkswitch.optical`): lognormal irradiance model
  (Andrews & Phillips 2005) with a temporal AR(1)/Gauss-Markov process for
  a configurable fade coherence time, an outage threshold derived from a
  dB link margin, and a gamma-gamma i.i.d. sampler (Al-Habash, Andrews &
  Phillips 2001).
- **RF channel** (`linkswitch.rf`): two-state Markov rain-occurrence
  process, lognormal rain-rate-when-raining, ITU-R P.838-form specific
  attenuation, a simplified effective-path-length reduction factor, and a
  fixed-margin availability rule.
- **Scenario / telemetry generation** (`linkswitch.scenario`): synchronised,
  seeded, reproducible dual-channel telemetry from independent optical/RF
  RNG streams.
- **Classical baseline policies** (`linkswitch.policies`), implemented and
  validated first: `FixedThresholdPolicy`, `HysteresisPolicy`.
- **Learned predictive policy** (`linkswitch.learn`, `linkswitch.features`,
  `linkswitch.policies.LearnedPolicy`): a `RandomForestClassifier` pipeline
  predicting imminent optical outage from causal rolling telemetry
  features, with a confidence (`predict_proba`) output, benchmarked against
  both baselines on identical seeded Monte Carlo scenarios.
- **Simulation / scoring** (`linkswitch.simulate`): switch-cost-aware
  per-episode scoring (delivered throughput, outage time, switch count) and
  a Monte Carlo driving loop.
- **Metrics with confidence intervals** (`linkswitch.metrics`): Student-t
  CIs on the mean, and a paired Monte Carlo policy comparison.
- **Analytic optimal-threshold model** (`linkswitch.analytic`): closed-form
  expected-throughput objective using the exact bivariate-normal
  level-crossing probability of the discrete AR(1) log-irradiance process;
  bounded-optimizer and grid-search solvers.
- **CLI** (`python -m linkswitch {threshold,simulate,compare}`).
- **Validation campaign**: `validation/analytic_threshold_check.py`,
  `validation/policy_comparison_ci.py`, `validation/horizon_sensitivity.py`,
  with `validation/VALIDATION.md` and raw stdout captures.
- **Examples**: `telemetry_and_switching.py` (one episode, per-policy
  switching visualised against the physical outage threshold) and
  `policy_comparison.py` (Monte Carlo comparison bar charts with CIs), both
  producing PNGs in `screenshots/`.
- **Documentation**: `README.md`, `MODEL_CARD.md`, `DATASET_CARD.md`.
- **Tests**: 201 tests — unit, input-validation, hand-calculated
  known-answer, edge-case, Hypothesis property tests, an end-to-end
  integration test, CLI subprocess tests, and pinned-seed benchmark/
  regression tests.

### Known issues

- **The learned policy does not beat the classical hysteresis baseline** in
  either scenario tested (mild: statistical tie on throughput, hysteresis
  wins on outage and switch count; moderate: hysteresis wins outright on
  every metric, and the learned policy underperforms even the naive
  fixed-threshold baseline). Reported as measured in `validation/VALIDATION.md`
  §V2 and `MODEL_CARD.md`; not tuned away.
- The learned policy's prediction confidence is an uncalibrated RandomForest
  vote fraction, not a validated probability.
- At long prediction horizons (H=20 in the tested moderate scenario) the
  learned policy degenerates to an almost-always-RF policy, collapsing
  throughput toward the RF-only floor (`validation/VALIDATION.md` §V3).
- RF specific-attenuation coefficients and the path reduction-factor length
  are illustrative defaults, not verified against current ITU-R tables.
- The AR(1) temporal fading model is a documented engineering approximation,
  not derived from a measured turbulence power spectrum.
- Gamma-gamma irradiance sampling is i.i.d.-only and not wired into
  telemetry generation (no temporal correlation for that model).
