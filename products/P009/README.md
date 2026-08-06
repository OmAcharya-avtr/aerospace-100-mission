# FogCast

**Status:** TESTING · **Class:** compact · **Validation level:** 2 (Research) · **AI:** yes

## Executive overview

FogCast predicts fog/aerosol specific optical attenuation (dB/km) for free-space
optical (FSO) communication links from meteorological visibility, wavelength, and
relative humidity. It implements the two classical empirical baselines — the Kruse
(1962) and Kim (2001) visibility models — and benchmarks a scikit-learn
gradient-boosting regressor with 90 % prediction intervals against them on a seeded
synthetic dataset. The ML model is trained on synthetic, model-derived data (see
DATASET_CARD.md); its metrics measure fidelity to that synthetic process, not to the
real atmosphere.

## Aerospace problem

Fog is the dominant availability killer for FSO links (ground-to-ground,
ground-to-air, and optical ground stations for space-to-ground lasercom): dense fog
attenuation can exceed 100 dB/km, closing links within hundreds of metres. Link-budget
engineers need fast attenuation estimates from routinely available weather data
(visibility, humidity) at the FSO telecom wavelengths (850/1310/1550 nm), plus honest
uncertainty bounds for margin analysis.

## Intended users

FSO link-budget engineers (research studies), optical ground-station siting analysts,
students of atmospheric optics and ML uncertainty quantification.

## Engineering theory

Both baselines use the Koschmieder relation with a 2 % contrast threshold:

    alpha [dB/km] = (10 / ln 10) * (3.912 / V) * (lambda / 550 nm)^(-q(V))

where V is visibility (km), lambda wavelength (nm), 3.912 = ln(1/0.02), and q is an
empirical aerosol size-distribution exponent.

- **Kruse model** (P. W. Kruse, L. D. McGlauchlin, R. B. McQuistan, *Elements of
  Infrared Technology*, Wiley, 1962): q = 1.6 (V > 50 km), 1.3 (6 < V <= 50 km),
  0.585 V^(1/3) (V <= 6 km).
- **Kim model** (I. I. Kim, B. McArthur, E. Korevaar, Proc. SPIE vol. 4214, pp. 26–37,
  2001): q = 1.6 (V > 50), 1.3 (6 < V <= 50), 0.16V + 0.34 (1 < V <= 6),
  V − 0.5 (0.5 < V <= 1), 0 (V <= 0.5).

Units: V km, lambda nm, alpha dB/km. Assumptions: aerosol (Mie) scattering only; no
molecular absorption lines, rain, snow, or turbulence. Validity enforced in code:
V in [0.05, 100] km, lambda in [500, 2000] nm (visible/near-IR).

**Known disagreement:** for V < ~1 km, Kruse's low-visibility branch predicts a
substantial long-wavelength advantage (e.g. 37.7 dB/km at 1550 nm vs 56.6 at 550 nm
for V = 0.3 km) while Kim sets q = 0, making dense-fog attenuation
wavelength-independent (56.6 dB/km at every wavelength). Kim et al. (2001) argue the
Kruse branch is unsupported by fog measurements. FogCast implements both and documents
the divergence rather than resolving it.

## Architecture

```
src/fogcast/
├── baselines.py   # Kim & Kruse q(V) and attenuation (validated formulas, cited)
├── dataset.py     # seeded synthetic data generator + deterministic 70/15/15 split
├── model.py       # FogCastModel: 3x GradientBoostingRegressor (point + q05 + q95)
└── __init__.py    # public API
```

No cross-product imports; pure numpy/scikit-learn/matplotlib.

## Installation

```
pip install -e .        # from products/P009/
# or, without installing:
export PYTHONPATH=src
```

## Quick start

```python
from fogcast import kim_attenuation_db_km, kruse_attenuation_db_km, predict

kim_attenuation_db_km(0.3, 1550.0)     # 56.63 dB/km (dense fog, wavelength-independent)
kruse_attenuation_db_km(0.3, 1550.0)   # 37.74 dB/km (the documented disagreement)

point, lo, hi = predict(1.0, 1550.0, 85.0)   # ML estimate + 90 % prediction interval
alpha = predict(1.0, 1550.0, 85.0, return_interval=False)
```

`predict()` lazily trains the default model on the seeded synthetic dataset on first
call (~5 s), then caches it. For explicit control use
`FogCastModel.train_default(n_samples=6000, seed=42)`.

## Configuration

`FogCastModel(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42)`;
dataset size/seed via `generate_dataset(n_samples, seed)`;
CSV export via `python -m fogcast.dataset --out data.csv --seed 42`.

## Examples

`python examples/attenuation_vs_visibility.py` — attenuation vs visibility at
850/1310/1550 nm, Kim/Kruse baselines vs ML point + 90 % interval
(`screenshots/attenuation_vs_visibility.png`).

## Validation

Level 2 evidence in `validation/VALIDATION.md` (raw outputs committed alongside):

- Kim q(V) branches reproduce the SPIE 4214 piecewise definition exactly (5/5 PASS).
- Attenuation at reference visibilities matches independent hand computation to
  <= 2e-16 relative error (7/7 PASS), e.g. V = 10 km, 1550 nm -> 0.4418 dB/km;
  V = 0.3 km, 1550 nm -> Kim 56.632 / Kruse 37.744 dB/km.
- Published behaviours reproduced: dense-fog wavelength independence (Kim),
  Kim == Kruse for V > 6 km, Kim/Kruse ratio 1.50 at V = 0.3 km, 1550 nm.
- ML error analysis on the held-out test split and documented regime limits.

## Benchmark results

Held-out synthetic test split (n = 900, seed 42) — `validation/benchmark_results.md`:

| Predictor | MAE (dB/km) | RMSE (dB/km) |
|---|---|---|
| ML (GBR) | 2.320 | 5.390 |
| Kim baseline | 2.465 | 5.749 |
| Kruse baseline | 7.675 | 15.043 |

90 % interval empirical coverage: 0.879 (nominal 0.90). Note: the truth is perturbed
Kim, so Kim is a near-oracle baseline; the ML edge is modest and Kruse's large error
reflects the synthetic truth's construction, not real-world skill.

## AI model details

Baseline: Kim/Kruse analytic models (implemented and validated first). Dataset:
seeded synthetic generator, `DATASET_CARD.md` (synthetic, model-derived — the
fundamental limitation). Training: 3 gradient-boosting regressors on
log10-attenuation, 70/15/15 split, all seeds fixed, ~5 s on 2 cores. Metrics and
uncertainty: table above; 90 % quantile-regression prediction intervals with 0.879
empirical coverage on held-out data. Failure cases and reproducibility commands:
`MODEL_CARD.md`. **This model is not certified for operational flight use.**

## Hardware requirements

Any 2-core CPU; ~200 MB RAM; no GPU. Training < 1 minute, full test suite ~10 s.

## Limitations

- **Synthetic ground truth**: the ML model is trained on Kim-model-derived data;
  reported accuracy is relative to that synthetic process, not to real fog
  measurements. Field-measured attenuation may differ systematically.
- Baselines exclude rain, snow, molecular absorption, and turbulence/scintillation.
- Dense fog (V < 0.5 km): the published baselines themselves disagree by ~1.5x at
  1550 nm; structural uncertainty dominates.
- ML extrapolates outside its training ranges (V > 50 km, lambda outside 600–1700 nm)
  within the accepted input domain; intervals there are unreliable.
- Visibility must be the 550 nm Koschmieder (2 % contrast) value; airport METAR
  visibilities use different thresholds and need conversion before use.

## Safety statement

This software is research-grade. It is not flight-qualified, not certified, and not
approved for operational aerospace use.

## Roadmap

- Calibrate/validate against published fog-attenuation measurement campaigns.
- Add rain/snow attenuation baselines (e.g. ITU-R P.1814 framework) as separate terms.
- Conformalized prediction intervals for finite-sample coverage guarantees.

## License

Apache-2.0 — see LICENSE. Copyright © 2026 OPTIMA Organisation.

## Credits

This is under reserved rights obtained by OPTIMA Organisation.

## Citation

```
FogCast 0.1.0 (2026). Fog/aerosol optical attenuation prediction for FSO links.
OPTIMA Organisation aerospace software portfolio, product P009. Apache-2.0.
Baselines: Kruse et al. (1962); Kim, McArthur, Korevaar, Proc. SPIE 4214 (2001).
```
