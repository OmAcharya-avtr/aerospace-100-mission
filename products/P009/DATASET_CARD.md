# FogCast Dataset Card

## Summary

Seeded synthetic dataset of 6000 samples mapping (visibility, wavelength, relative
humidity) to fog/aerosol specific attenuation (dB/km) for FSO-link studies.
Generated entirely by the committed script `src/fogcast/dataset.py`
(`generate_dataset(n_samples=6000, seed=42)`, or `python -m fogcast.dataset` for CSV).
The dataset is not committed as a file; regeneration is deterministic (same seed =>
bit-identical arrays, covered by a unit test).

## THE FUNDAMENTAL LIMITATION — read this first

**This is synthetic, model-derived data, NOT field measurements.** The ground-truth
attenuation is computed from the Kim empirical model (Kim, McArthur, Korevaar, Proc.
SPIE 4214, 2001) with added perturbations. Consequently, any ML "accuracy" reported
against this dataset is accuracy **relative to a synthetic generative process**, not
relative to the real atmosphere. A model that scores perfectly here may still be wrong
about real fog. No transmissometer, visibility-sensor, or FSO-link measurement campaign
data was used anywhere in this product. This is the dataset's fundamental limitation.

## Generative process (exact, all seeded)

Features (n = 6000, `numpy.random.default_rng(seed=42)`):

| Feature | Distribution | Range |
|---|---|---|
| `visibility_km` | log-uniform | 0.05–50 km |
| `wavelength_nm` | 50 % choice of {850, 1310, 1550}; 50 % uniform | 600–1700 nm |
| `rh_percent` | uniform | 40–100 % |

Ground truth: `alpha = alpha_Kim(V, lambda; q_Kim(V) + dq) * m_rh(V, RH) * exp(eps)` with

1. **Wavelength-dependent noise**: `dq ~ N(0, 0.07)` perturbs the Kim size-distribution
   exponent (clipped so q >= 0); its multiplicative effect `(lambda/550)^(-dq)` grows
   with distance from the 550 nm reference wavelength.
2. **Humidity covariate effect**:
   `m_rh = 1 + 0.25 * ((clip(RH-40, 0, 60)/60)^2) * exp(-0.5*((ln V - ln 3)/0.8)^2)` —
   up to +25 % attenuation at RH = 100 % centred on the haze regime (V ~ 3 km).
   Qualitatively motivated by aerosol hygroscopic growth (e.g. Hänel 1976, Advances in
   Geophysics 19); the functional form and coefficients are synthetic design choices,
   **not** fitted to any measurement.
3. **Measurement noise**: multiplicative lognormal, `eps ~ N(0, 0.05)` (~5 %).

## Splits

Deterministic 70/15/15 train/validation/test via `split_indices(n, seed=42)`
(independent permutation seed = dataset seed + 1): 4200 / 900 / 900 samples.
Test split is used only for the benchmark and coverage evaluation.

## Intended use

Research/education on ML regression and uncertainty quantification for FSO link-budget
tooling; benchmarking against the Kim/Kruse analytic baselines. Not for deriving real
link availability figures, and not certified for operational flight use.

## License

Apache-2.0 (same as the package).
