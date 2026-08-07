# Batch 02 Specification

**Date:** 2026-08-01 · **Approval:** development and publication proceed without per-batch sign-off (owner waiver, 2026-08-01, recorded in `tracking/APPROVAL_LOG.md`)
**Composition:** 2 flagship, 3 medium, 5 compact · 7/10 AI-enabled · Levels: 2×L1, 6×L2, 2×L3
**Theme:** Adaptive optics, turbulence characterization, and the estimation/control core for aerospace GNC. Batch 02 extends the Batch 01 atmospheric work upward into wavefront sensing and correction, and opens the GNC line that later batches build on.
**Stack:** Python 3.11, NumPy/SciPy, scikit-learn (PyTorch unavailable), pytest + Hypothesis, Ruff. CLI + library API + plotting examples.

All names below were checked against PyPI on 2026-08-01 and are free.

---

## P011 — WaveForge (flagship, L3, AI)

**Problem:** Free-space optical links through turbulence lose Strehl ratio to wavefront distortion. Designers need to size an adaptive optics system — actuator count, sensor subapertures, loop bandwidth — against a turbulence profile before committing hardware, and classical AO control lags the atmosphere by one or more frames.

**Scope:** Kolmogorov phase-screen atmosphere reused conceptually from P003 but independently implemented; Zernike and zonal wavefront representations; Shack-Hartmann sensor model with photon and read noise; deformable mirror influence-function model with actuator limits; closed-loop integrator control with configurable gain and latency; Strehl ratio and residual-error budgeting (fitting error, temporal error, noise error, each cited). AI element: a learned predictive controller that forecasts the next-frame wavefront from a history of slope measurements, benchmarked against the classical integrator and against a pure-delay baseline, with uncertainty on the prediction. Report honestly whether prediction beats the integrator at the tested latencies.

**Level 3 requirements:** `docs/REQUIREMENTS.md` with 12–18 numbered requirements and a verification matrix; uncertainty analysis; regression suite with pinned seeded outputs; performance benchmark; failure-mode tests (actuator saturation, sensor dropout, loop instability at excessive gain).

**Validation:** Zernike orthonormality and Noll variance coefficients checked against published values; residual error versus D/r₀ compared with the standard fitting-error scaling; closed-loop rejection transfer function compared with the analytic integrator response; Strehl versus residual variance checked against the Maréchal approximation in its validity range.

## P012 — NavBench (flagship, L3, AI)

**Problem:** Attitude and navigation filters are chosen by folklore. Engineers need a controlled bench that runs the same truth trajectory and sensor suite through several estimators and reports consistency, not just error.

**Scope:** Truth trajectory generator (rigid-body attitude dynamics with torques, plus a simple orbital or airborne position track); sensor models with documented noise and bias behaviour — gyro with random walk and bias instability, star tracker or sun sensor, accelerometer, GPS-like position fix; estimators: linear Kalman filter, extended KF, unscented KF, and a multiplicative EKF for attitude with quaternion state; consistency diagnostics — NEES and NIS with chi-squared bounds, which is the part most tools omit. AI element: learned adaptive process-noise tuning that adjusts Q online from innovation statistics, benchmarked against fixed hand-tuned Q and against classical adaptive covariance estimation, with confidence output. Report honestly if the classical adaptive scheme wins.

**Level 3 requirements:** same battery as P011.

**Validation:** filter reproduces the analytic steady-state Riccati solution on a linear time-invariant case; NEES/NIS fall inside their chi-squared bounds over Monte Carlo runs for a correctly specified filter and provably leave them for a mis-specified one; UKF matches EKF on a near-linear problem and diverges more gracefully on a strongly nonlinear one; quaternion normalization and MEKF reset behaviour verified against P007's algebra.

## P013 — TurbScope (medium, L2, AI)

**Problem:** Turbulence strength along a path is rarely measured directly; it is inferred from scintillation, image motion, or meteorological proxies, each with its own bias.

**Scope:** Forward models for common Cn² estimators (scintillometer-style irradiance variance, differential image motion); synthetic measurement generator from known Cn² profiles; inversion to path-averaged Cn² with uncertainty; AI element: regression from multi-sensor features to Cn² with prediction intervals, baselined against the closed-form single-sensor inversion. Documented failure regime: saturation of scintillation at strong turbulence, where the analytic inversion becomes multi-valued.

**Validation:** round-trip recovery of known Cn² in the weak regime with reported error; demonstration and quantification of the saturation failure; interval coverage on held-out data.

## P014 — WaveLab (medium, L2, AI)

**Problem:** Wavefront reconstruction from Shack-Hartmann slopes is classically a least-squares inverse that degrades badly at low light and with missing subapertures.

**Scope:** Slope-to-phase reconstruction: least-squares and Southwell/Fried geometry matrix baselines with regularization; synthetic slope data from Kolmogorov screens; AI element: learned reconstructor mapping slopes to Zernike coefficients, trained on synthetic data, with ensemble uncertainty; benchmark against the regularized least-squares baseline across photon flux and subaperture-dropout rates. The plausible honest outcome is that least-squares wins at high flux and the learned model wins under dropout — report what is measured.

**Validation:** noise-free reconstruction recovers input Zernikes to numerical tolerance; reconstruction error versus photon flux compared with the analytic noise-propagation coefficient; dropout robustness curves.

## P015 — LinkSwitch (medium, L2, AI)

**Problem:** Hybrid RF-optical links must decide when to fall back from the high-rate optical channel to the resilient RF channel. Switching late loses data; switching early wastes capacity.

**Scope:** Dual-channel model — optical channel with fading statistics from the Batch 01 lineage, RF channel with rain-fade and lower but stable rate; switching policies: fixed-threshold baseline, hysteresis baseline, and a learned policy predicting imminent optical outage from recent link telemetry, with confidence; evaluation on delivered throughput, outage time, and switch count. Requires an explicit statement that the fading model is simulated, not measured.

**Validation:** analytic check that the optimal fixed threshold matches the value derived from the channel statistics; policy comparison with confidence intervals over seeded Monte Carlo; sensitivity to prediction horizon.

## P016 — ZernKit (compact, L1, no AI)

**Scope:** Zernike polynomial toolkit — Noll and OSA/ANSI indexing with explicit conversion, radial polynomial evaluation, orthonormality on the unit disc, gradient (slope) computation, fitting of a wavefront to coefficients, Noll residual variance table. Every convention stated; index-ordering mistakes are the classic bug this product exists to prevent.

**Validation (L1):** orthonormality integrals computed numerically against the analytic Kronecker delta; low-order Zernikes checked against their closed forms by hand; Noll variance coefficients for the first several modes compared with published values; Hypothesis property tests on index round-trips.

## P017 — EstimKit (compact, L1, no AI)

**Scope:** Compact, dependency-light Kalman family — linear KF, EKF with user-supplied Jacobians, UKF with configurable sigma-point parameters, RTS smoother, and Joseph-form covariance update for numerical stability. Documented: why Joseph form, when square-root filtering is needed instead, and the symptoms of covariance collapse.

**Validation (L1):** steady-state gain on a scalar constant-velocity model checked against the hand-solved algebraic Riccati equation; smoother reduces RMS error below the filter on the same data; UKF reduces to KF on a linear system to numerical tolerance; property tests that covariance stays symmetric positive-definite.

## P018 — ShackSim (compact, L2, AI)

**Scope:** Shack-Hartmann wavefront sensor simulator — lenslet array geometry, spot formation per subaperture with diffraction and noise, slope extraction by centre of gravity and by correlation. AI element: a learned slope estimator for low-flux and elongated-spot conditions, benchmarked against thresholded centre of gravity, with per-slope confidence. This is deliberately the sensor-level companion to P008's single-spot work and must cite it as related rather than duplicate it.

**Validation:** known tilt produces the analytically predicted slope; slope error versus photon count compared with the standard noise-propagation expression; learned-versus-classical crossover reported wherever it falls.

## P019 — CnCast (compact, L2, AI)

**Scope:** Vertical Cn² profile prediction — implements standard published profile models as baselines (Hufnagel-Valley 5/7 and SLC day/night families, each cited with its stated validity), then a learned model mapping surface meteorological features and time-of-day to a profile shape, with prediction intervals. Dataset card must state plainly that training data is synthetic, derived from the baseline models plus perturbations, so accuracy is measured against a generative process and not against radiosonde measurements.

**Validation:** baseline models reproduce their published characteristic values at reference altitudes; integrated seeing quantities (r₀, isoplanatic angle, Greenwood frequency) computed from a profile and hand-checked; interval coverage on held-out data.

## P020 — AtmoProfile (compact, L2, no AI)

**Scope:** Atmospheric turbulence integral toolkit — from any Cn²(h) profile compute Fried parameter r₀, isoplanatic angle θ₀, Greenwood frequency f_G, Rytov variance for slant paths, and scintillation index, each with the weighting integral shown in the docstring and the zenith-angle dependence handled explicitly. Deliberately deterministic: this is the reference implementation later AI products are benchmarked against.

**Validation (L2):** each integral checked against hand computation for a constant-Cn² slab where closed forms exist; standard profile models produce r₀ values consistent with the ranges quoted in the literature; zenith-angle scaling verified against the analytic sec(ζ) powers.

---

## Cumulative position after Batch 02

| Metric | After Batch 02 | Mission target |
|---|---:|---:|
| Products | 20 | 100 |
| Flagship / medium / compact | 4 / 6 / 10 | 20 / 30 / 50 |
| AI-enabled | 14 | ≥70 |
| Level 1 / 2 / 3 / 4 | 4 / 12 / 4 / 0 | 10 / 60 / 25 / 5 |

Level 3 accumulates at 2 per batch, which reaches 20 against a target of 25. Batches 08–10 should raise flagship validation depth or promote selected medium products to Level 3 to close the gap. Level 4 remains deferred until Jetson hardware batches.

## Shared core

Batch 02 introduces `shared/aerocore/turbulence.py` conventions consumed by P011, P013, P014, P019 and P020. Products remain independently installable; shared code is vendored at publication so no repository depends on another.

## Completion gate

Mission §17 for every product; §11 items 1–15 additionally for the seven AI products. Non-waivable regardless of the approval waiver.
