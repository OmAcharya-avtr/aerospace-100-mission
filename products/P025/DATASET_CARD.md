# Dataset Card — FDIScope synthetic GNC fault scenarios v0.1.0

## THE DATA IS ENTIRELY SIMULATED

**No flight telemetry, no measured Kalman-filter innovation sequence, no
on-orbit fault log and no real spacecraft parameters are used anywhere in this
package.** Every scenario is drawn from the distributions below by a seeded
`numpy.random.Generator` and every residual sample is produced by the
closed-loop simulator in `src/fdiscope/simulate.py` from a committed, seeded
script. Nothing was measured; nothing was observed; nothing came from a
mission.

There is no externally sourced number in this dataset at all. Unlike a field
model or an ephemeris, a residual sequence has no published reference value to
be checked against, so every validation in this repository compares a
measurement with a **closed-form expression** from the change-detection
literature rather than with another dataset.

## Summary

| | |
|---|---|
| Generator | `fdiscope.scenarios.sample_scenario(seed, index)` (committed, deterministic) |
| Regeneration | `python data/generate_dataset.py` (~17 s) |
| Training scenarios | 240, seeds 1000–1239, 30 per class |
| Held-out scenarios | 240, seeds 5000–5239, 30 per class (disjoint) |
| Threshold calibration runs | 150, seeds 9000–9149, all fault-free |
| Held-out fault-free runs | 150, seeds 12000–12149 |
| Feature rows | 1440 × 16, from 6 windows per training scenario |
| Labels | class index into `fdiscope.faults.FAULT_CLASSES` |
| Committed files | `data/training_scenarios.csv` (13 kB), `data/training_features.csv` (394 kB), `data/dataset_manifest.txt` |
| Integrity | SHA-256 of both CSVs recorded in `data/dataset_manifest.txt` |

Both CSVs are well under 1 MB and are committed. They are redundant with the
generator: deleting them and rerunning `data/generate_dataset.py` reproduces
them byte for byte.

## What is modelled

One axis of a rigid spacecraft under estimator-plus-PD control:

- **Plant**: `J theta_ddot = u + w`, `J = 12 kg m^2`, zero-order-hold
  discretised at `dt = 0.1 s`, disturbance-torque spectral density
  `4e-8 N^2 m^2 s`.
- **Sensors**: an attitude sensor at 0.05 deg (1σ) and a rate gyro at
  0.02 deg/s (1σ), both additive white Gaussian. This is a coarse-pointing
  suite, not a star tracker.
- **Actuator**: a wheel with a symmetric 0.05 N m torque limit and no other
  dynamics.
- **Estimator**: a Kalman filter whose `F`, `G`, `H`, `Q`, `R` are *exactly*
  the plant's, started at the steady-state covariance.
- **Reference**: a sinusoidal attitude command, 0.02 rad amplitude, 60 s
  period, so the commanded torque is persistently non-zero. Without it an
  actuator loss of effectiveness would be unobservable, because
  `(1 - l) * 0 == 0`.

## Sampling ranges

These are **engineering choices covering a plausible coarse-pointing
small-spacecraft loop. They are not fitted to any measured population of
spacecraft, sensors or failures**, and no claim is made that they are
representative of any real fleet or of any real failure-rate distribution.

| Quantity | Distribution | Range |
|---|---|---|
| Fault class | cycled by scenario index | exactly balanced, 8 classes |
| Onset sample | uniform integer | 600 – 1300 (of 2000) |
| Sensor channel | uniform | 0 (attitude) or 1 (rate) |
| Sensor bias | uniform, random sign | 1.0 – 8.0 × that channel's σ |
| Sensor drift, attitude channel | uniform, random sign | 0.4 – 4.0 σ/s |
| Sensor drift, rate channel | uniform, random sign | 0.02 – 0.30 σ/s |
| Sensor stuck / dropout | no magnitude | — |
| Actuator loss of effectiveness | uniform | 0.20 – 1.00 (fraction lost) |
| Actuator stuck | no magnitude | — |
| Actuator runaway | uniform, random sign | 1e-5 – 2e-4 N m/s |

The two drift ranges differ by an order of magnitude because the loop absorbs
an attitude drift far more effectively than a rate drift: the DC gain from a
constant attitude-sensor bias to the innovation is **exactly zero**
(`fdiscope.analytic.innovation_dc_gain`), so the same drift rate on the two
channels produces residuals that differ by roughly 40×. Using one range would
have made every attitude-channel drift undetectable.

## How a row is produced

1. Draw a scenario from its seed: class, onset, channel, magnitude.
2. Simulate 2000 closed-loop samples with that fault injected, producing the
   normalised residual sequence `r_k = L^-1 y_k`.
3. Take six 100-sample windows: four starting 0, 10, 25 and 50 samples after
   onset, labelled with the scenario's class, and two starting 150 and 300
   samples *before* onset, labelled `none`.
4. Compute the sixteen features of `fdiscope.features.window_features` on each.

A fault-free scenario contributes six `none` rows, so the class counts are
600 `none` and 120 of each of the seven fault classes.

## Field names

`training_scenarios.csv`: `index`, `seed`, `fault_class`, `onset_step`,
`magnitude`, `channel`, `n_steps`.

`training_features.csv`: the sixteen features in `feature_names()` order —
`mean_ch0`, `std_ch0`, `slope_ch0`, `autocorr1_ch0`, `max_abs_ch0`,
`cusum_range_ch0`, `mean_ch1`, `std_ch1`, `slope_ch1`, `autocorr1_ch1`,
`max_abs_ch1`, `cusum_range_ch1`, `mean_nis`, `max_nis`, `corr_01`,
`exceed_frac` — then `label_index` and `label`.

## What is NOT modelled

This list is the dataset's most important content.

1. **No filter–plant mismatch.** The filter's model is exactly the plant's, so
   the fault-free innovation is exactly white with exactly the modelled
   covariance. That is the single largest gap between this dataset and any
   real one: a real filter has unmodelled dynamics, an imperfectly known `Q`,
   and a non-white innovation, and every method in this package — the
   classical tests included — assumes whiteness.
2. **No coloured or correlated sensor noise.** No gyro angle random walk, no
   rate random walk, no bias instability, no attitude-sensor low-frequency
   error, no scale-factor error, no misalignment, no quantisation.
3. **No actuator dynamics.** The wheel is an instantaneous torque source
   inside a clip. No motor lag, no cogging, no wheel speed, no zero-crossing
   friction, no bearing drag, no momentum storage.
4. **One axis, one inertia, one gain set, one manoeuvre.** No cross-coupling,
   no gyroscopic term, no flexible modes, no slosh, no large-angle kinematics.
5. **No environmental disturbance torque** — no gravity gradient, aerodynamic,
   solar-pressure or residual-dipole torque. The only disturbance is white
   noise with a flat spectrum, which is a modelling convenience, not physics.
6. **Exactly one fault per scenario.** No simultaneous faults, no fault that
   evolves from one mode into another, no intermittent fault, no fault that
   heals.
7. **Faults appear instantaneously at a known sample.** A real degradation has
   a soft onset; only the drift and runaway modes here have any onset shape at
   all.
8. **No FDI reconfiguration.** Nothing in the loop responds to a detected
   fault, so no post-detection behaviour is represented.
9. **The magnitude ranges are uniform in their parameters, not in mission
   likelihood.** Nothing here says a 4σ gyro bias is as probable as a 1σ one.
10. **Undetectable cases are deliberately kept.** A stuck actuator latched
    near the value a settled controller would have commanded, or a loss of
    effectiveness during a low-torque phase, is close to invisible. These are
    not filtered out, because dropping them would inflate every reported
    detection rate. They are why `actuator_stuck` recall is 0.5333 even for
    the better isolator.
11. **All rows of a scenario share one label**, so 1440 rows are far from 1440
    independent observations; the effective sample size is closer to the 240
    scenarios.

## Licence and provenance

Generated by this repository, Apache-2.0, copyright © 2026 OPTIMA
Organisation. No third-party dataset is redistributed and none is used.
