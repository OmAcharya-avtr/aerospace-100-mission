# CnCast — ML vs baseline benchmark (produced by validation/benchmark_ml.py)

- Python 3.11.15, numpy 2.4.4, scikit-learn 1.8.0
- Platform: Linux-6.18.5-fc-v18-x86_64-with-glibc2.39

## Setup

- scenarios: 394 fit + 131 calibration + 175 test (split by SCENARIO, seed 4242)
- rows: 11032 fit, 3668 calibration, 4900 test (28 altitudes per scenario)
- seeds: {'data_seed': 20260807, 'altitude_seed': 99, 'split_seed': 4242, 'model_random_state': 7}
- fit + calibration wall time: **15.3 s** on 2 cores (budget: 120 s)
- conformal delta applied to each interval bound: +0.0838 dex
- quantile-crossing fraction on the fit set: 0.0000

## 1. Held-out error, all altitudes (units: dex = decades of Cn^2)

| predictor | RMSE | MAE | bias | p95 abs err |
|---|---:|---:|---:|---:|
| CnCast learned model | 0.2095 | 0.1620 | -0.0201 | 0.4190 |
| HV 5/7 (mandated baseline) | 0.5665 | 0.4475 | +0.2686 | 1.1048 |
| SLC day/night | 0.7314 | 0.5199 | -0.0893 | 1.2224 |
| Training climatology | 0.3102 | 0.2395 | +0.0049 | 0.6168 |

Learned / HV 5/7 RMSE ratio: **0.370** (63.0 % reduction).

## 2. Held-out RMSE by altitude band (dex)

| band [m] | n | CnCast learned model | HV 5/7 (mandated baseline) | SLC day/night | Training climatology |
|---|---:|---:|---:|---:|---:|
| 5–50 | 1400 | 0.2122 | 0.7860 | 0.5748 | 0.3735 |
| 50–300 | 1050 | 0.2023 | 0.6911 | 0.4717 | 0.3424 |
| 300–2000 | 1050 | 0.2089 | 0.2642 | 0.4359 | 0.2105 |
| 2000–8000 | 875 | 0.2257 | 0.3039 | 0.6126 | 0.2333 |
| 8000–20000 | 525 | 0.1879 | 0.3143 | 1.6315 | 0.3351 |

## 3. Prediction-interval coverage on held-out data

| interval | nominal | empirical coverage | mean width [dex] |
|---|---:|---:|---:|
| raw quantile GBR (alpha = 0.05 / 0.95) | 0.900 | 0.8033 | 0.5575 |
| conformally calibrated (CQR) | 0.900 | 0.8988 | 0.7249 |

Binomial standard error on the calibrated coverage with n = 4900 rows is 0.0043; the rows are NOT independent (28 per scenario), so the effective n is closer to the 175 scenarios and the true standard error is larger — treat +/-0.02 as the resolution of this estimate.

Coverage by altitude band (calibrated interval):

| band [m] | n | coverage | mean width [dex] |
|---|---:|---:|---:|
| 5–50 | 1400 | 0.8821 | 0.6947 |
| 50–300 | 1050 | 0.8914 | 0.7170 |
| 300–2000 | 1050 | 0.9095 | 0.7014 |
| 2000–8000 | 875 | 0.9314 | 0.8409 |
| 8000–20000 | 525 | 0.8819 | 0.6752 |

## 4. Integrated seeing quantities from a predicted profile

Test scenario 0: T = 17.70 C, wind = 11.42 m/s, RH = 76.92 %, hour = 23.93, day-of-year = 344

| quantity | from predicted median | from truth profile | from HV 5/7 |
|---|---:|---:|---:|
| r0 [cm] | 7.3342 | 6.2482 | 5.0130 |
| theta0 [urad] | 4.0393 | 3.6102 | 7.0455 |
| f_G [Hz] | 123.8399 | 149.4365 | 122.7370 |

r0 from the interval bounds (upper Cn^2 -> smaller r0):
- r0(lower bound profile) = 13.1120 cm
- r0(upper bound profile) = 4.0565 cm

### Raw numbers for the hand check in VALIDATION.md §5

Predicted median Cn^2 on a 5-point coarse grid (same scenario):

| h [m] | Cn^2 [m^-2/3] |
|---:|---:|
| 5 | 2.852424e-15 |
| 100 | 1.197547e-15 |
| 1000 | 1.846164e-16 |
| 5000 | 2.623659e-17 |
| 20000 | 2.299238e-18 |

- trapezoid mu_0 on this 5-point grid = 1.450072e-12 m^(1/3)
- r0 on this 5-point grid = 6.431470 cm (the 24-point value above is the one to trust; the coarse grid exists only so the arithmetic can be done by hand)

## 5. Reproducibility

- identical test features on re-run: True
- max |prediction difference| across re-runs: 0.000e+00 dex
- conformal delta identical: True

Total script wall time: 51.3 s
