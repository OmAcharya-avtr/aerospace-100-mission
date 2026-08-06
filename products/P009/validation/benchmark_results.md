# FogCast benchmark — ML vs Kim vs Kruse on held-out synthetic test data

Dataset: n=6000, seed=42; split 70/15/15 (train 4200, val 900, test 900).
Training time: 4.7 s (3x GradientBoostingRegressor, 300 estimators, depth 3, lr 0.05, 2 CPU cores).

Ground truth = synthetic generative process (Kim + perturbations); metrics measure fidelity to that process, NOT to field measurements.

## Overall (test split, dB/km)

| Predictor | MAE (dB/km) | RMSE (dB/km) |
|---|---|---|
| ML (GBR) | 2.320 | 5.390 |
| Kim baseline | 2.465 | 5.749 |
| Kruse baseline | 7.675 | 15.043 |

90 % prediction-interval empirical coverage: **0.879** (nominal 0.90; tolerance band 0.85-0.95). Median interval width: 2.600 dB/km.

## Per-regime MAE (dB/km)

| Regime | n | ML | Kim | Kruse |
|---|---|---|---|---|
| dense fog (V <= 0.5 km) | 302 | 6.204 | 6.577 | 20.927 |
| fog (0.5 < V <= 1 km) | 85 | 1.187 | 1.172 | 4.927 |
| haze (1 < V <= 6 km) | 247 | 0.417 | 0.496 | 0.646 |
| clear (V > 6 km) | 266 | 0.038 | 0.036 | 0.036 |

Interpretation: the ML model's edge over the Kim baseline comes from learning the synthetic humidity effect and averaging the exponent noise; the Kruse baseline is worst in fog because its q(V) branch overestimates the long-wavelength advantage there (the documented Kim-vs-Kruse disagreement).
