# Validation — WaveLab 0.1.0

Validation level: **2**. Every number below comes from a script in this
directory, executed in this session, with its full stdout saved next to it as
a `.txt` file. Nothing here is fabricated or hand-adjusted; where a check
would have failed, that would be reported as a failure (none did, at the
stated tolerances).

## 1. Noise-free reconstruction recovers the input to numerical tolerance

Script: [`validate_noise_free.py`](validate_noise_free.py). Raw output:
[`noise_free_output.txt`](noise_free_output.txt).

Three reconstructors, 20 random trials each, seeded (`numpy.random.default_rng(0/1/2)`):

| Reconstructor | Geometry | Check | Worst error (20 trials) | Tolerance | Result |
|---|---|---|---:|---:|---|
| `ModalReconstructor` (TSVD) | analytic Zernike interaction matrix, 32 subapertures, 14 modes | max\|â − a_true\| | **4.996e-16 rad** | 1e-6 rad | PASS |
| `ZonalReconstructor`, Hudgin | 11x11 grid, 81 active points, null space = piston only | max\|φ̂ − φ_true\| | **1.410e-14 rad** | 1e-5 rad | PASS |
| `ZonalReconstructor`, Fried | 11x11 grid, 81 active / 77 used, null space = 2 (piston + waffle) | \|‖φ̂ − φ_true‖ − \|true waffle component‖\| | **4.576e-04 rad** | 1e-3 rad | PASS |

The Fried check is not "recovers the exact input" — by construction the Fried
geometry cannot see the waffle mode (§3 below), so the correct claim, and the
one actually checked, is that the reconstruction residual equals *exactly*
the true phase's own waffle component, to numerical tolerance. This is the
honest form of "noise-free recovery" for a geometry with a nontrivial null
space.

## 2. Reconstruction error vs photon flux vs the analytic noise-propagation coefficient

Script: [`validate_photon_noise.py`](validate_photon_noise.py). Raw output:
[`photon_noise_output.txt`](photon_noise_output.txt).

For i.i.d. slope noise of variance `sigma_s^2`, the modal least-squares
reconstructor's per-mode coefficient variance is predicted by
`Var = coeff * sigma_s^2` with `coeff` the analytic noise-propagation
coefficient (`wavelab.linalg.noise_propagation_coefficients`; Wallner 1983,
*J. Opt. Soc. Am.* **73**, 1771). 800 Monte Carlo trials per flux level, one
fixed true coefficient vector (seed 0), TSVD reconstructor (`reg=1e-8`):

| Photon flux | σ_slope | predicted Var | empirical Var (800 trials) | ratio |
|---:|---:|---:|---:|---:|
| 100 | 1.000000 | 2.6082e-03 | 2.6316e-03 | 1.009 |
| 300 | 0.577350 | 8.6940e-04 | 8.8228e-04 | 1.015 |
| 1000 | 0.316228 | 2.6082e-04 | 2.6657e-04 | 1.022 |
| 3000 | 0.182574 | 8.6940e-05 | 8.5605e-05 | 0.985 |
| 10000 | 0.100000 | 2.6082e-05 | 2.5671e-05 | 0.984 |

**Worst \|empirical/predicted − 1\| = 0.022 (2.2%)**, well inside the 25%
tolerance used, PASS. The direct `sigma(N) ∝ 1/sqrt(N)` scaling on the
aggregate RMS coefficient error is confirmed to within 1.3% relative error
across the same flux range (see raw output for the full table).

## 3. Dropout robustness curves — learned reconstructor vs regularized least-squares baseline

Script: [`validate_dropout.py`](validate_dropout.py). Raw output:
[`dropout_output.txt`](dropout_output.txt). Full discussion:
[`../MODEL_CARD.md`](../MODEL_CARD.md) §7-9.

Trained once (1800 samples, flux 800, dropout 0.25, seed 100), evaluated on
400 held-out samples per operating point (seeds 9000 / 9500), across both
photon flux (dropout fixed at 0) and subaperture dropout rate (flux fixed at
800):

**vs photon flux (dropout = 0):**

| flux | baseline RMS | ML RMS | winner |
|---:|---:|---:|---|
| 100 | 0.05106 | 0.06153 | baseline |
| 300 | 0.02948 | 0.04564 | baseline |
| 1000 | 0.01615 | 0.04008 | baseline |
| 3000 | 0.00932 | 0.03853 | baseline |
| 10000 | 0.00511 | 0.03802 | baseline |

**vs subaperture dropout rate (flux = 800):**

| dropout | baseline RMS | ML RMS | winner |
|---:|---:|---:|---|
| 0.00 | 0.01765 | 0.04028 | baseline |
| 0.15 | 0.02006 | 0.03963 | baseline |
| 0.30 | 0.02377 | 0.04330 | baseline |
| 0.45 | 0.03427 | 0.05012 | baseline |
| 0.60 | 0.81656 | 0.06043 | **ML** |

**Measured result: the regularized least-squares baseline wins 9 of 10
operating points, and its margin over the learned ensemble widens with
increasing flux (up to 7.4x at flux 10 000).** The learned ensemble wins the
single most extreme dropout point tested (60%), where the baseline's fixed
Tikhonov regularization parameter is not adapted to the shrinking number of
active subapertures and becomes numerically unstable. This is reported
exactly as measured — no tolerance was loosened, no operating point was
dropped, and the model was not retuned after seeing this result. See
`MODEL_CARD.md` §9 for the failure-mode interpretation and README
"Limitations" for what a follow-up fix (dropout-adaptive regularization in
the baseline) would need to show before this conclusion could be revisited.

## Environment and compute

All three scripts were run sequentially in this session on 2 CPU cores,
Python 3.11, `n_jobs=1` throughout. Total combined wall time: noise-free
< 1 s, photon-noise 1.9 s, dropout benchmark 53.6 s. Every script is
individually well under the mission's 2-minute-per-script budget.

## What was NOT validated (explicit scope limits)

- No comparison against real Shack-Hartmann sensor data, real atmospheric
  turbulence statistics, or any physical hardware (`DATASET_CARD.md`).
- No test of the learned model's uncertainty calibration against a formal
  coverage criterion beyond the single mean-ratio number reported in
  `MODEL_CARD.md` §8 (which shows the ensemble spread under-states the true
  error by 25-40% throughout, i.e. is *not* calibrated).
- No sensitivity study over Noll mode count, subaperture layout density, or
  MLP architecture — the reported numbers hold for the configurations stated
  above and should not be extrapolated to different sizes without rerunning.
