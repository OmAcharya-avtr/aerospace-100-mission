# Dataset Card — DetumbleSim synthetic detumble scenarios v0.1.0

## THE DATA IS ENTIRELY SIMULATED

**No flight telemetry, no measured magnetometer record, no on-orbit detumble
log and no real spacecraft parameters are used anywhere in this package.**
Every scenario is drawn from the distributions below by a seeded
`numpy.random.Generator` and every telemetry sample is produced by the
simulator in `detumblesim/simulate.py`. Nothing was measured; nothing was
observed; nothing came from a mission.

The only externally sourced numbers in the whole repository are:

- the three IGRF-14 degree-1 Gauss coefficients at epoch 2025.0
  (`g(1,0) = −29350.0`, `g(1,1) = −1410.3`, `h(1,1) = 4545.5` nT), from the
  IAGA `igrf14coeffs.txt` coefficient file;
- twelve IGRF-14 total-intensity values from the British Geological Survey web
  service, used **only** as the reference the field model is checked against in
  `validation/field_model_check.py`; and
- the WGS 84 Earth radius, gravitational parameter and rotation rate.

## Summary

| | |
|---|---|
| Generator | `detumblesim.scenarios.sample_scenario(seed)` (committed, deterministic) |
| Regeneration | `python data/generate_dataset.py` |
| Training scenarios | 20, seeds 1000–1019 |
| Held-out scenarios | 40, seeds 5000–5039 (disjoint) |
| Feature rows | 1068 × 8, harvested every 30 control steps from a 60-step window |
| Labels | `log10(k_oracle / k_fixed)`, range `[−0.2857, +1.1429]` dex |
| Committed files | `data/training_scenarios.csv` (3.6 kB), `data/training_features.csv` (165 kB), `data/dataset_manifest.txt` |
| Integrity | SHA-256 of both CSVs recorded in `data/dataset_manifest.txt` |

Both CSVs are well under 1 MB and are committed. They are redundant with the
generator: deleting them and rerunning `data/generate_dataset.py` reproduces
them byte for byte.

## Sampling ranges

These are **engineering choices covering the small-satellite regime that B-dot
is normally applied to. They are not fitted to any measured population of
spacecraft**, and no claim is made that they are representative of any real
fleet.

| Quantity | Distribution | Range |
|---|---|---|
| Inertia scale | log-uniform | 0.01 – 0.30 kg m² |
| Principal-axis ratios | uniform, triangle inequality enforced by resampling | 0.6 – 1.6 × the scale |
| Initial rate magnitude | log-uniform | 3 – 20 deg/s |
| Initial rate direction | isotropic on the sphere | — |
| Initial attitude | uniform random unit quaternion | — |
| Altitude | uniform | 400 – 800 km |
| Inclination | uniform | 20 – 100 deg |
| RAAN, argument of latitude | uniform | 0 – 360 deg |
| Earth rotation phase at t = 0 | uniform | 0 – 2π rad |
| Per-axis dipole limit | log-uniform, isotropic torquer set | 0.05 – 0.50 A m² |

## How a row is produced

1. Sample a scenario from its seed.
2. Run the simulator at each of the 8 grid gains
   (`np.geomspace(1e4, 1e6, 8)`), score each with
   `cost = t/T_orbit + 0.5·∫|m|²dt/(m_max²·T_orbit)`, and take the argmin as
   that scenario's **oracle gain**.
3. Run the scenario once more at the tuned fixed gain
   (`k = 7.196857e+04 A m² s/T`) and replay its magnetometer history through a
   60-step `TelemetryWindow`, emitting a feature row every 30 steps.
4. Label every row of that scenario with `log10(oracle gain / fixed gain)`.

Simulation settings throughout: `duration_s = 23000`, `control_dt_s = 2.0`,
`substeps = 1`, target rate 1.0 deg/s, magnetometer noise **zero**.

## Field names

`training_scenarios.csv`: `seed`, `ixx_kgm2`, `iyy_kgm2`, `izz_kgm2`,
`rate0_deg_s`, `altitude_km`, `inclination_deg`, `raan_deg`, `arg_lat0_deg`,
`gmst0_rad`, `max_dipole_am2`, `oracle_gain`, `oracle_cost`.

`training_features.csv`: `seed`, then the eight features in the order
`log10_rate_proxy`, `log10_mean_field_t`, `rate_trend_per_1000s`,
`saturation_duty`, `field_variability`, `log10_max_dipole`,
`log10_inertia_scale`, `log10_elapsed_s`, then
`label_log10_gain_ratio`.

## Known limitations of this dataset

1. **It inherits every modelling error of the simulator.** The magnetic field
   is a degree-1 IGRF truncation with a measured worst-case error of 71.86 %
   against IGRF-14 (`validation/VALIDATION.md`, V1-A2, **FAILED**); the orbit
   is circular and unperturbed; there are no gravity-gradient, aerodynamic,
   solar-pressure or residual-dipole torques; the spacecraft is perfectly
   rigid; the magnetorquer is an instantaneous per-axis dipole box.
2. **Twenty training scenarios is very small** for an eight-feature model.
   That size is set by the 3-minute, 2-core compute budget: the oracle sweep
   alone is 160 simulations.
3. **The labels are quantised** by the 8-point gain grid (a factor of 1.93
   between neighbouring gains, three decades total). Only 6 of the 8 grid
   values were ever selected as an oracle gain.
4. **The label is the best *constant* gain**, so the dataset cannot teach a
   genuinely time-varying schedule; see `MODEL_CARD.md`, failure case 4.
5. **All rows of a scenario share one label**, so the 1068 rows are far from
   1068 independent observations; the effective sample size is closer to the
   20 scenarios.
6. **The magnetometer is noise-free** in every generated row
   (`mag_noise_t = 0`). The simulator supports Gaussian noise but no headline
   number uses it, so the features are cleaner than any real magnetometer
   would deliver.
7. **The ranges are uniform in their parameters, not in mission likelihood.**
   Equatorial orbits below 20° inclination are excluded entirely, which is
   exactly where the controllability gap is worst
   (`validation/VALIDATION.md`, V4); the learned model has therefore never
   seen the hardest case.

## Licence and provenance

Generated by this repository, Apache-2.0, copyright © 2026 OPTIMA
Organisation. No third-party dataset is redistributed. The IGRF-14 coefficients
and the BGS reference values are cited, not redistributed as a dataset.
