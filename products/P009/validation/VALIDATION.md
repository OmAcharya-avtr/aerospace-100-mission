# FogCast — Validation Evidence (Level 2, Research)

All numbers below were produced by running the committed scripts in this directory in this
build session. Raw outputs: `validate_baselines_output.txt`, `benchmark_results.md`.

## 1. Baseline implementations reproduce the published formulas

Script: `validate_baselines.py` (rerun with `PYTHONPATH=../src python validate_baselines.py`).

Reference formula (Koschmieder 2 % contrast form used by Kim et al. 2001):
`alpha [dB/km] = (10/ln 10) * (3.912 / V) * (lambda / 550 nm)^(-q)`.

### Kim q(V) branches (Kim, McArthur, Korevaar, Proc. SPIE 4214, 2001)

| V (km) | computed q | expected q | branch | result |
|---|---|---|---|---|
| 0.30 | 0.0000 | 0.0000 | V <= 0.5: q = 0 | PASS |
| 0.75 | 0.2500 | 0.2500 | 0.5 < V <= 1: q = V - 0.5 | PASS |
| 3.00 | 0.8200 | 0.8200 | 1 < V <= 6: q = 0.16V + 0.34 | PASS |
| 10.00 | 1.3000 | 1.3000 | 6 < V <= 50: q = 1.3 | PASS |
| 60.00 | 1.6000 | 1.6000 | V > 50: q = 1.6 | PASS |

### Attenuation at reference visibilities (implementation vs independent hand computation)

| V (km) | lambda (nm) | model | computed (dB/km) | reference (dB/km) | rel. err. |
|---|---|---|---|---|---|
| 0.30 | 1550 | Kim | 56.6320 | 56.6320 | 0 |
| 0.30 | 1550 | Kruse | 37.7438 | 37.7438 | 0 |
| 0.75 | 850 | Kim | 20.3169 | 20.3169 | 0 |
| 3.00 | 1310 | Kim | 2.7797 | 2.7797 | 1.6e-16 |
| 3.00 | 1310 | Kruse | 2.7231 | 2.7231 | 0 |
| 10.00 | 1550 | Kim | 0.4418 | 0.4418 | 0 |
| 60.00 | 850 | Kim | 0.1411 | 0.1411 | 2.0e-16 |

All checks PASS (tolerance 1e-12 relative). These confirm the code implements the stated
formulas exactly; they are formula-consistency checks, not comparisons against new field data.

### Published qualitative behaviours

- Kim wavelength independence in dense fog: at V = 0.3 km, alpha(850 nm) = alpha(1550 nm)
  = 56.632 dB/km — PASS (q = 0 branch).
- Kim == Kruse for V > 6 km (identical q): V = 20 km, 1310 nm -> 0.2749 dB/km both — PASS.
- Documented Kim-vs-Kruse disagreement at low visibility: V = 0.3 km, 1550 nm ->
  Kim 56.63 dB/km vs Kruse 37.74 dB/km (ratio 1.50). Kruse's q = 0.585 V^(1/3) branch
  predicts a long-wavelength advantage in fog that Kim et al. (2001) argue is unsupported
  by fog measurements — PASS (behaviour reproduced and documented).

## 2. ML error analysis on held-out data

Script: `benchmark_ml.py`; full table in `benchmark_results.md`.
Dataset n = 6000, seed 42, split 70/15/15 (test n = 900, never seen in training).
Training time 4.7 s (3 GradientBoostingRegressor fits, 2 CPU cores) — within the
< 1 minute budget.

| Predictor | MAE (dB/km) | RMSE (dB/km) |
|---|---|---|
| ML (GBR) | 2.320 | 5.390 |
| Kim baseline | 2.465 | 5.749 |
| Kruse baseline | 7.675 | 15.043 |

90 % prediction-interval empirical coverage on the test split: **0.879**
(nominal 0.90; acceptance band 0.85–0.95 — PASS). Median interval width 2.60 dB/km.

Per-regime MAE (dB/km): dense fog 6.20 (ML) vs 6.58 (Kim) vs 20.93 (Kruse);
fog 1.19 / 1.17 / 4.93; haze 0.42 / 0.50 / 0.65; clear 0.038 / 0.036 / 0.036.

Honest reading: the ML model beats Kim only modestly overall (its ground truth *is*
perturbed Kim, so Kim is a near-oracle baseline); in the fog band (0.5–1 km) Kim is
marginally better than the ML model (1.17 vs 1.19 dB/km MAE). The large margin over
Kruse simply reflects that the synthetic truth was built on Kim's q(V). None of these
numbers demonstrate skill against real atmospheric measurements.

## 3. Regime limits (documented)

- Visibility validity: 0.05–100 km (Koschmieder 2 % contrast definition; inputs outside
  raise ValueError). Training data covers 0.05–50 km only; the ML model extrapolates
  for V in (50, 100] km — intervals there are not trustworthy.
- Wavelength validity: 500–2000 nm (visible/near-IR aerosol-scattering regime of both
  empirical models). Training data covers 600–1700 nm.
- Models exclude molecular absorption lines, rain, snow, and turbulence/scintillation.
- Below V ~ 0.5 km the two published baselines disagree materially; predictions in dense
  fog carry the highest structural uncertainty regardless of the interval width.

## Test suite

`python -m pytest tests/ -q` from `products/P009/`: **34 passed** (run in this session).
`ruff check src/ tests/`: clean.
