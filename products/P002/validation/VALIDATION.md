# TrackBench — Validation Evidence

**Product:** P002 TrackBench · **Version:** 0.1.0 · **Validation level:** 3 (Professional, at v0.1 MVP depth)
**Environment:** Python 3.11.15, numpy 2.4.4, scipy 1.17.1, Linux x86_64, 2 CPU cores
**All numbers below were produced by running the scripts in this directory during the build session.**
Raw output is saved next to each script as `vN_*.txt`.

| Script | Raw output | Result |
|--------|-----------|--------|
| `v1_spiral_coverage.py` | `v1_spiral_coverage.txt` | PASS |
| `v2_acquisition_time.py` | `v2_acquisition_time.txt` | PASS |
| `v3_control_step_response.py` | `v3_control_step_response.txt` | PASS |
| `v4_jitter_psd.py` | `v4_jitter_psd.txt` | PASS |
| `v5_reacq_benchmark.py` | `v5_reacq_benchmark.txt` | PASS |
| `v6_performance.py` | `v6_performance.txt` | PASS |
| `v7_regression_baseline.py` | `v7_regression_baseline.txt` | PASS |

Reproduce everything:

```bash
for f in validation/v*.py; do python "$f"; done
```

---

## 1. Spiral coverage vs the geometric track-spacing argument (R-01, R-03)

**Reference:** geometric argument, not a published number. An Archimedean
spiral `r(φ) = aφ` has constant radial pitch `2πa`. Setting
`2πa = s = 2·R_beam·(1 − overlap)` means consecutive turns are separated by
`s`, so a beam footprint of radius `R_beam` leaves no *radial* gap whenever
`s ≤ 2·R_beam`, i.e. for every `overlap ∈ [0, 1)`. The covered probability
mass of a 2-D Gaussian target should therefore reach the design containment.

**Method:** σ = 300 µrad, R_beam = 20 µrad, containment 0.995
(r_max = 976.574 µrad = 3.2552 σ), 200 000 Monte Carlo targets per case,
covered = "within R_beam of some dwell point" (k-d tree nearest neighbour).

| overlap | s [µrad] | s / 2R | dwells | covered mass | covered − 0.995 |
|---------|----------|--------|--------|--------------|-----------------|
| 0.00 | 40.000 | 1.000 | 7 491 | 0.98460 | −0.01040 |
| 0.10 | 36.000 | 0.900 | 8 323 | 0.99485 | −0.00015 |
| 0.25 | 30.000 | 0.750 | 9 987 | 0.99496 | −0.00004 |
| 0.40 | 24.000 | 0.600 | 12 484 | 0.99515 | +0.00015 |
| 0.50 | 20.000 | 0.500 | 14 980 | 0.99529 | +0.00029 |

Monte Carlo standard error at C = 0.995 with N = 2·10⁵ is
`sqrt(C(1−C)/N) = 1.6·10⁻⁴`.

**Reported finding (a genuine shortfall, not tuned away):** at `overlap = 0`
the tracks are exactly tangent and the measured coverage falls **0.01040
below** the design containment (0.98460 vs 0.995). The one-dimensional
"no radial gap" argument only holds along the radial direction; because the
spiral turns, tangent circles on adjacent turns leave curvature gaps near the
inner turns. This is precisely why PAT scan designs specify an overlap
margin. From `overlap ≥ 0.10` the shortfall (−1.5·10⁻⁴) is within one Monte
Carlo standard error of zero.

**Negative control:** a pattern designed for R_beam but *flown* with a beam
0.4× that size (`s / 2R = 1.875 > 1`) covers only **0.49623** — the check has
the power to detect a real gap.

**Cost comparison:** raster covers 0.99843 with 13 199 dwells; spiral covers
0.99513 with 9 987 dwells (raster/spiral dwell ratio 1.322), because the
raster sweeps the low-probability corners of the bounding square.

**Pass criterion:** min covered (overlap ≥ 0.10) ≥ 0.9940 **and** negative
control < 0.90. Achieved: 0.99485 and 0.49623. → **PASS**

---

## 2. Acquisition time: Monte Carlo vs uniform-coverage analytic model (R-06)

**Model (internal derivation, documented in `scan.py`):** a spiral of track
spacing `s` scanned at along-track speed `v` sweeps area at rate `s·v`, so
reaching radius `r` costs `t(r) ≈ πr²/(s·v)`. Averaging over the truncated
Rayleigh radial density gives `E[T | r ≤ r_max] = (π/(s·v))·E[r² | r ≤ r_max]`;
in the untruncated limit this reduces to `2πσ²/(s·v)`
(checked in `tests/test_scan.py::test_expected_time_infinite_containment_limit`).
Comparable spiral-acquisition statistics appear throughout the laser-comm PAT
literature (see Kaymak et al. 2018, *IEEE Communications Surveys & Tutorials*
20(2), acquisition section); **no page-specific published formula is claimed
here** — the derivation is internal and is validated by simulation only.

**Method:** 20 000 targets drawn from the Gaussian, dwell-by-dwell simulation,
σ = 300 µrad, R_beam = 20 µrad, overlap 0.25, dwell 1.0 ms, containment 0.995.
Scan speed 0.010002 rad/s; single-pass scan time 9.9870 s.

| p_dwell | p_crossing | detected | MC mean [s] | ± SEM | naive analytic [s] | naive dev | corrected analytic [s] | corrected dev |
|---------|-----------|----------|-------------|-------|--------------------|-----------|------------------------|---------------|
| 1.00 | 1.00000 | 19 903 | 1.8226 | 0.0123 | 1.8344 | −0.65 % | 1.8344 | **−0.65 %** |
| 0.90 | 0.99990 | 19 909 | 1.8252 | 0.0124 | 2.9439 | −38.00 % | 1.8354 | **−0.56 %** |

**Reported finding:** passing the *per-dwell* probability into the analytic
model as if it were a *per-crossing* probability overestimates acquisition
time by **38 %**. In the simulator a target is inside the footprint for about
`2/step_fraction = 4` consecutive dwells, so the per-crossing detection
probability is `1 − (1 − 0.9)⁴ = 0.99990`, not 0.9. The `scan.py` docstring
for `expected_acquisition_time_spiral` now states this explicitly and points
at this script.

**Residual systematic:** even with the correction the MC mean is 0.6 % *below*
the analytic value. This is expected and directional: the analytic model
assumes a swath of width `s`, but the simulated beam covers a disc of radius
`R_beam`, i.e. a swath wider than `s` whenever `overlap > 0`, so the simulator
acquires slightly sooner.

**Pass criterion:** |relative deviation| < 5 % using the corrected
per-crossing probability. Achieved: max 0.645 %. → **PASS**

---

## 3. Control: step-response metrics vs analytic second-order theory (R-11, R-12)

**Reference:** Ogata 2010, *Modern Control Engineering*, 5th ed., ch. 5
(canonical second-order step response); Anderson & Moore 1990, *Optimal
Control: Linear Quadratic Methods* (LQR / root-square locus).

With derivative-on-measurement PD control of `J s² + b s`, the closed loop is
**exactly** `Kp / (J s² + (Kd + b) s + Kp)` — no numerator zero — so
`ωₙ = sqrt(Kp/J)` and `ζ_eff = (Kd + b)/(2·sqrt(Kp·J))`, and the textbook
overshoot and peak-time formulas apply without approximation. The 10–90 %
rise time is obtained numerically from the analytic response, not from an
approximation formula.

**Plant:** J = 0.05 kg m², b = 0.02 N m s/rad, τ_max = 2 N m, RK4, dt = 10⁻⁴ s,
step 10⁻⁴ rad.

### 3A. PD, hand-checkable cases

| f_n [Hz] | ζ_des | ζ_eff | Mp sim | Mp analytic | ΔMp | t_p sim [s] | t_p analytic [s] | Δt_p | t_r sim [s] | t_r analytic [s] | Δt_r |
|----------|-------|-------|--------|-------------|-----|-------------|------------------|------|-------------|------------------|------|
| 2.0 | 0.500 | 0.51592 | 0.15076 | 0.15076 | −0.00 % | 0.29170 | 0.29184 | −0.05 % | 0.13270 | 0.13281 | −0.08 % |
| 5.0 | 0.707 | 0.71337 | 0.04055 | 0.04085 | −0.73 % | 0.14250 | 0.14270 | −0.14 % | 0.06890 | 0.06900 | −0.14 % |
| 5.0 | 0.900 | 0.90637 | 0.00110 | 0.00118 | −7.28 % | 0.23820 | 0.23669 | +0.64 % | 0.09260 | 0.09268 | −0.09 % |
| 10.0 | 0.707 | 0.71018 | 0.04144 | 0.04204 | −1.43 % | 0.07090 | 0.07102 | −0.17 % | 0.03420 | 0.03434 | −0.41 % |

Worked hand check for row 2: `Kp = J·ωₙ² = 0.05·(10π)² = 49.34802`,
`Kd = 2·0.707·0.05·10π = 2.22111`, `ωₙ = 31.41593 rad/s`,
`ζ_eff = (2.22111 + 0.02)/(2·sqrt(49.34802·0.05)) = 0.7133662`,
`Mp = exp(−πζ/sqrt(1−ζ²)) = 0.0408453`,
`t_p = π/(ωₙ·sqrt(1−ζ²)) = 0.1426958 s`. Simulation: 0.04055 and 0.14250 s.

The −7.28 % on the ζ = 0.906 row is a *relative* error on an overshoot of
0.00118 (0.118 %); the absolute error is 0.00008 and is limited by the
10⁻⁴ s sampling of the peak. The pass criterion for near-critical damping is
therefore stated in absolute terms.

### 3B. Pointwise trajectory error (PD, 5 Hz, ζ_des = 0.707)

- max |simulated − analytic| = **1.1552·10⁻⁷ rad = 0.1155 % of the step**
- RMS |simulated − analytic| = 2.7334·10⁻⁸ rad

### 3C. LQR pole placement from eq. (10), undamped plant

| f_n [Hz] | \|p\| simulated [rad/s] | \|p\| design [rad/s] | rel. error | ζ simulated | ζ analytic |
|----------|------------------------|----------------------|------------|-------------|------------|
| 2.0 | 12.56637 | 12.56637 | +1.41·10⁻¹⁶ | 0.707107 | 0.707107 |
| 5.0 | 31.41593 | 31.41593 | −6.45·10⁻¹⁵ | 0.707107 | 0.707107 |
| 10.0 | 62.83185 | 62.83185 | +2.61·10⁻¹² | 0.707107 | 0.707107 |
| 20.0 | 125.66371 | 125.66371 | −5.62·10⁻¹³ | 0.707107 | 0.707107 |

The weight rule `r = q/(J²ωₙ⁴)` reproduces the Butterworth pattern
`|p| = ωₙ, ζ = √2/2` to machine precision.

### 3D. Controller comparison (from actual runs)

Open-loop disturbance RMS 2.1116·10⁻⁶ rad (S₀ = 10⁻¹² rad²/Hz, f_c = 3 Hz,
order 2, dt = 2·10⁻⁴ s):

| controller | t_r [s] | Mp | t_s [s] | closed-loop RMS [rad] | rejection factor | −3 dB BW [Hz] |
|------------|---------|----|---------|-----------------------|------------------|---------------|
| PD | 0.06880 | 0.04025 | 0.18840 | 1.3099·10⁻⁶ | 1.61 | 4.921 |
| PID | 0.05900 | 0.17613 | 0.64500 | 1.3473·10⁻⁶ | 1.57 | 4.955 |
| LQR | 0.06820 | 0.04261 | 0.18900 | 1.3154·10⁻⁶ | 1.61 | 4.962 |

Reading: the integral term buys zero steady-state error at the cost of 4×
the overshoot and 3.4× the settling time; LQR at the same design bandwidth is
essentially indistinguishable from well-tuned PD here, which is expected —
for a second-order plant with a full-state-equivalent controller the two
designs span the same pole locations. The rejection factor is modest (≈1.6)
because most of the jitter power in this PSD lies **above** the 5 Hz loop
bandwidth; see `screenshots/ex02_tracking_error.png`, panel 3, where rejection
below 5 Hz is clearly visible (≈2 decades at 0.6 Hz).

**Pass criteria:** (A) worst relative metric deviation < 3 % → 1.432 %;
(A′) worst |ΔMp| < 0.005 → 0.00060; (B) trajectory error < 1 % of step →
0.1155 %; (C) LQR pole/damping error < 10⁻⁶ → 2.61·10⁻¹². → **PASS**

---

## 4. Jitter synthesis: realised PSD vs target PSD (R-08)

**Reference:** spectral representation of stationary Gaussian processes
(Shinozuka & Deodatis 1991, *Applied Mechanics Reviews* 44(4); Percival &
Walden 1993, *Spectral Analysis for Physical Applications*). Estimator:
Welch, Hann window, 50 % overlap.

**Method:** fs = 2000 Hz, N = 2¹⁸ (131.1 s), nperseg = 8192 → K = 63 segments,
expected per-bin relative scatter ≈ 1/√K = 0.126.

| PSD case | band | bins | median ratio | mean ratio | std of ratio |
|----------|------|------|--------------|-----------|--------------|
| S₀ = 10⁻¹², f_c = 3 Hz, order 2 | 1–10 Hz | 36 | 0.9834 | 0.9924 | 0.0336 |
| | 10–100 Hz | 369 | 1.0014 | 1.0026 | 0.0412 |
| | 100–900 Hz | 3277 | 1.0022 | 1.0012 | 0.0411 |
| S₀ = 4·10⁻¹², f_c = 10 Hz, order 4 | 1–10 Hz | 36 | 0.9924 | 0.9961 | 0.0414 |
| | 10–100 Hz | 369 | 1.0057 | 1.0030 | 0.0414 |
| | 100–900 Hz | 3277 | 1.0014 | 1.0005 | 0.0418 |
| S₀ = 2·10⁻¹¹, f_c = 200 Hz, order 2 | 1–10 Hz | 36 | 0.9958 | 0.9939 | 0.0323 |
| | 10–100 Hz | 369 | 1.0018 | 0.9996 | 0.0404 |
| | 100–900 Hz | 3277 | 1.0014 | 1.0006 | 0.0412 |

**Variance (Parseval) check** against the quadrature integral of the target
PSD over [0, fs/2]:

| case | measured variance [rad²] | target integral [rad²] | deviation |
|------|--------------------------|------------------------|-----------|
| 1 | 4.699574·10⁻¹² | 4.703389·10⁻¹² | −0.081 % |
| 2 | 3.140065·10⁻¹¹ | 3.141591·10⁻¹¹ | −0.049 % |
| 3 | 5.493524·10⁻⁹ | 5.493603·10⁻⁹ | −0.001 % |

An independent analytic cross-check exists for the order-2 shape:
`∫₀^F S₀/(1+(f/f_c)²) df = S₀·f_c·arctan(F/f_c)`, verified in
`tests/test_dynamics.py::test_psd_variance_matches_analytic_arctan_integral`.

**Ensemble convergence** (case 1, 1–10 Hz mean ratio): 1 realisation → 1.0405;
4 → 1.0206 (std 0.0201); 16 → 1.0157 (std 0.0290). The estimator converges
towards 1 as realisations are added, i.e. it is not accidentally matched by a
single lucky draw.

**Pass criteria:** worst |band median − 1| < 10 % → 1.660 %; worst |variance
deviation| < 10 % → 0.081 %. → **PASS**

---

## 5. Reacquisition: learned policy vs both scripted baselines (R-15, R-16)

**Configuration:** σ₀ = 5.00·10⁻⁵ rad, drift 1.00·10⁻⁴ rad/s, cone radius
1.00·10⁻³ rad, coverage rate 6.00·10⁻⁷ rad²/s, p_detect = 0.85,
max_time = 30 s, LOCAL disc = 3σ(t), RING width = 2σ(t), κ = 10.
Evaluation: 2 000 episodes, **common random numbers** (episode *i* uses seed
999 + *i* for every policy). Times are censored at max_time; read the mean
together with the success rate.

### Baselines first

| policy | mean [s] | 95 % CI [s] | median [s] | p90 [s] | success | attempts | action mix |
|--------|----------|-------------|------------|---------|---------|----------|------------|
| always-full (baseline) | 8.6485 | [8.2926, 9.0044] | 5.2360 | 30.000 | 0.877 | 1.685 | FULL 100 % |
| always-local (baseline) | 6.4395 | [5.9612, 6.9179] | 0.5762 | 30.000 | 0.838 | 4.716 | LOCAL 100 % |

Baseline evaluation wall time: 1.36 s.

### Learned (tabular Q-learning, 20 000 training episodes)

| training seed | mean [s] | 95 % CI [s] | median [s] | p90 [s] | success | attempts | action mix |
|---------------|----------|-------------|------------|---------|---------|----------|------------|
| 12345 | **4.1480** | [3.7752, 4.5209] | 0.5001 | 8.814 | 0.909 | 3.427 | LOCAL 17 % / FULL 20 % / RING 63 % |
| 20260 | 4.2095 | [3.8162, 4.6028] | 0.5001 | 30.000 | 0.898 | 3.865 | LOCAL 15 % / FULL 12 % / RING 73 % |
| 777 | 4.4380 | [4.0432, 4.8328] | 0.3969 | 30.000 | 0.896 | 3.847 | LOCAL 33 % / FULL 17 % / RING 50 % |

Training wall time 2.83–2.97 s per seed.

**Result:** mean over three training seeds **4.2652 s** (spread 4.1480–4.4380 s).

- vs always-full: **50.7 % reduction** in mean time-to-reacquire; 95 % CIs disjoint.
- vs always-local: **33.8 % reduction**; 95 % CIs disjoint.
- Success rate also improves (0.896–0.909 vs 0.877 / 0.838).

**Why it wins (mechanism, not hand-waving):** the two baselines are each
optimal in one regime and bad in the other. `always-local` is cheap
(0.24 s per attempt at t = 0) but repeatedly re-searches the *same* disc, so
when the loss was violent (large κ·u² displacement) it burns many attempts
before σ(t) grows enough to cover the target. `always-full` always covers the
target geometrically but costs 5.2 s every attempt. The learned policy spends
50–73 % of its attempts on `RING`, which sweeps only the annulus *not yet
searched* — the same coverage growth as repeated local restarts at a fraction
of the area cost — and escalates to `FULL` in the states where the
last-known-offset bin indicates a high-severity loss.

**Reproducibility:** training with the same seed twice gives bitwise identical
Q tables and identical evaluation dictionaries (verified in the script and in
`tests/test_reproducibility.py::test_q_learning_is_bitwise_reproducible`).

**Uncertainty output:** on the seed-12345 policy, 64 of 320 discrete states
were visited during training. Confidence on visited states: mean 0.204,
min 0.000, max 1.000. The remaining 256 unvisited states fall back to the
`FULL` baseline action, and 37 visited states carry confidence < 0.1 (the
policy still acts greedily there but flags low confidence). The confidence
is a heuristic margin × support score, **not a calibrated probability**.

**Pass criterion:** the best learned 95 % CI strictly below both baseline CIs,
and training bitwise reproducible. → **PASS**

---

## 6. Performance and compute budget (R-20)

Machine: Linux 6.18.5 x86_64, 2 CPU cores, Python 3.11.15, numpy 2.4.4,
scipy 1.17.1.

| simulated duration [s] | steps | wall [s] | steps/s | × realtime |
|------------------------|-------|----------|---------|-----------|
| 0.2 | 1 000 | 0.0270 | 36 972 | 7.39 |
| 1.0 | 5 000 | 0.1378 | 36 284 | 7.26 |
| 4.0 | 20 000 | 0.5513 | 36 275 | 7.25 |

Median throughput **36 284 steps/s** (7.3× realtime at dt = 2·10⁻⁴ s).

| operation | wall [s] |
|-----------|----------|
| `spiral_scan` (9 987 dwells) | 0.0034 |
| `coverage_fraction` (2·10⁵ samples) | 0.1174 |
| `synthesize_jitter` (2²⁰ samples) | 0.0781 |
| `run_episode` (default scenario) | 0.3171 |
| `run_monte_carlo` (20 episodes) | 3.0073 |
| `train_q_learning` (20 000 episodes) | **2.9665** |
| `evaluate_policy` (2 000 episodes) | **0.7251** |

Against the 180 s mission budget: training uses **1.6 %**, Monte Carlo
evaluation **0.4 %**. The full V5 benchmark (3 trainings + 5 evaluations) is
≈12.5 s. → **PASS**

---

## 7. Regression baseline (R-21)

28 pinned values (scan geometry and coverage, acquisition times, jitter
series statistics, PD step metrics, LQR gain, a full end-to-end episode, the
Q-table checksum, and both baseline Monte Carlo means) are committed in
`tests/test_regression.py::PINNED` and checked on every test run.
`validation/v7_regression_baseline.py` regenerates them and diffs against the
committed set; the run in `v7_regression_baseline.txt` reports **ok on all 28
values**. Tolerances are 10⁻⁹ relative for deterministic float pipelines,
exact for integers, and 10⁻⁶ relative for the LQR gain (BLAS/LAPACK ordering).

---

## 8. Uncertainty analysis

Sources of uncertainty in every number above, and how each is bounded:

| Source | Magnitude | Treatment |
|--------|-----------|-----------|
| **Monte Carlo sampling** (coverage, acquisition time, policy benchmarks) | coverage: SE = √(C(1−C)/N) = 1.6·10⁻⁴ at N = 2·10⁵; acquisition time: SEM 0.0123 s on 1.82 s (0.7 %); policy mean: SEM ≈ 0.19 s on 4.15 s | Reported as ± SEM or 95 % normal-approximation CI; conclusions require disjoint CIs |
| **Time discretisation** (RK4, dt) | trajectory error 0.116 % of step at dt = 10⁻⁴ s; peak-time quantisation ±dt | Compared pointwise against the closed-form response (§3B); metrics quoted to no more digits than dt supports |
| **Spectral estimation** (Welch) | per-bin relative scatter 1/√K = 12.6 % at K = 63; band medians 0.3–1.7 % | Judged on band medians over ≥36 bins, plus an independent variance (Parseval) check |
| **Model form error** (uniform-coverage acquisition model) | −0.6 % systematic, direction explained (swath width R_beam vs s) | Reported as a residual systematic, not absorbed into the tolerance |
| **Training stochasticity** (Q-learning) | spread across 3 training seeds 4.148–4.438 s (±3.4 % about the mean) | Three seeds reported; the *worst* learned seed still beats both baselines |
| **Floating-point / library version** | LQR gain reproducible to 10⁻⁶ relative; everything else to 10⁻⁹ | Regression tolerances set accordingly; library versions pinned in this document |
| **Censoring** (reacquisition timeout) | 9–16 % of episodes hit the 30 s cap | Mean is explicitly a censored mean; success rate, median and p90 reported alongside |

**What is NOT quantified:** model *validity* error — the gap between this
simulator and a real optical terminal. No number in this document has been
compared against hardware measurements or mission telemetry. All references
are textbook results or derivations internal to this package. The parameter
values in the shipped scenarios are illustrative orders of magnitude, not
design data for any system.

---

## 9. Summary of findings

1. Spiral coverage matches the geometric argument for `overlap ≥ 0.10`
   (within 1.5·10⁻⁴, i.e. one MC standard error); at `overlap = 0` it falls
   **1.04 % short** because of spiral-curvature gaps between tangent tracks.
2. The uniform-coverage acquisition-time model agrees with Monte Carlo to
   **0.65 %** once the per-crossing (not per-dwell) detection probability is
   used; using the per-dwell probability naively overestimates by **38 %**.
3. PD step-response metrics reproduce the canonical second-order results to
   **≤1.43 %** relative (≤0.0006 absolute in overshoot), and the full
   trajectory to **0.12 % of the step**.
4. LQR weights from `r = q/(J²ωₙ⁴)` place the closed-loop poles to
   **machine precision** (2.6·10⁻¹² worst relative error).
5. Synthesised jitter matches the target PSD to **≤1.7 %** in band medians
   and its variance to **≤0.081 %**.
6. Tabular Q-learning reduces mean time-to-reacquire by **33.8 %** vs
   always-local and **50.7 %** vs always-full, with disjoint 95 % CIs across
   three training seeds, at a training cost of **3 s**.
7. No validation check failed. Two checks required the *analysis* to be
   corrected rather than the tolerance (items 1 and 2); both corrections are
   documented above and reflected in the source docstrings.
