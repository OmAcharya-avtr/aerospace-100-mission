# Validation — ScintiNet 0.1.0

**Validation level:** 2 (Research). Evidence below was produced by running the
scripts in this directory in this build session on the target environment
(Python 3.11, 2 CPU cores, numpy/scikit-learn). Raw outputs are committed
alongside: `campaign_log.txt`, `sim_vs_theory.txt`, `benchmark_results.txt`,
`dataset.csv`.

## Rerun commands

```bash
python validation/run_campaign.py        # ~23 s -> dataset.csv, campaign_log.txt
python validation/validate_simulator.py  # <1 s  -> sim_vs_theory.txt
python validation/benchmark_surrogate.py # ~3 s  -> benchmark_results.txt
python -m pytest tests/ -q               # 50 tests
```

All randomness is seeded (`BASE_SEED = 2026`, split seed 0, surrogate
`random_state=0`), so reruns reproduce the numbers below exactly.

---

## V1 — Analytic core against textbook closed form (known-answer)

Reference: L. C. Andrews and R. L. Phillips, *Laser Beam Propagation through
Random Media*, 2nd ed., SPIE Press, 2005; L. C. Andrews, J. Opt. Soc. Am. A
9(4), 597 (1992) for the aperture-averaging factor.

| Quantity | Inputs | Hand calculation | `scintinet` | Tolerance |
|---|---|---|---|---|
| Plane-wave σ_R² | Cn²=1e-15, λ=1.55 µm, L=2000 m | 1.23·1e-15·5.118659e7·1.126908e6 = **7.09495e-2** | 7.09495e-2 | rel 1e-4 |
| Plane-wave σ_R² | Cn²=5e-16, λ=850 nm, L=1000 m | 1.23·5e-16·1.031702e8·3.162278e5 = **2.00646e-2** | 2.00646e-2 | rel 1e-4 |
| Spherical σ_R² | same as row 1 | (0.50/1.23)·7.09495e-2 = **2.88413e-2** | 2.88413e-2 | rel 1e-4 |
| Aperture factor A | λ=1.55 µm, L=2000 m, D=0.1 m | (1+1.062·5.06708)^(−7/6) = **0.115065** | 0.115065 | rel 1e-4 |
| σ_I²(D) | same | 0.115065·7.09495e-2 = **8.1638e-3** | 8.1638e-3 | rel 1e-3 |

Hand calculations are reproduced in the test comments in
`tests/test_rytov.py`. **Result: PASS** (5/5).

Algebraic identities are additionally checked with Hypothesis
(`tests/test_rytov.py::TestProperties`): exact linearity in Cn², the
k^(7/6) = (2π/λ)^(7/6) wavelength scaling (ratio 2^(7/6) under λ→2λ, rel 1e-9),
A ∈ (0,1] and strictly decreasing in D, and σ_R²(spherical) < σ_R²(plane).

## V2 — Simulator numerical sanity

| Check | Criterion | Measured | Result |
|---|---|---|---|
| Angular-spectrum propagation is unitary | Σ\|U\|² preserved for a random complex field over 500 m | rel error < 1e-12 | PASS |
| Zero turbulence → no scintillation | σ_I² ≈ 0, ⟨I⟩ = 1 for Cn²=0 | \|σ_I²\| < 1e-12, ⟨I⟩ = 1 ± 1e-12 | PASS |
| Energy conservation with turbulence | ⟨I⟩ = 1 ± 0.02 for Cn²=5e-16, L=1000 m | within 0.02 | PASS |
| Screen variance linear in Cn²·dz | ratio 4.0 for a 4× increase (same RNG stream) | 4.0 to rel 1e-9 | PASS |
| Screen piston removed | \|mean(φ)\| < 1e-10 rad | PASS | PASS |
| Seeded reproducibility | identical (params, seed) → bit-identical σ_I² | exact equality | PASS |
| Aperture averaging monotone | σ_I²(D=60 mm) < σ_I²(D=20 mm) < σ_I²(point) | PASS | PASS |

Source: `tests/test_simulator.py`. Measured mean intensity across the full
campaign stayed within 0.5 % of unity (e.g. 1.0021 at Cn²=1e-15, L=2000 m).

## V3 — Simulator against Rytov theory (key Level-2 evidence)

The reduced campaign (18 (Cn², L, λ) points × 3 apertures = 54 rows;
256² grid, 0.5 m domain, 8 screens, 8 realizations, 22.6 s wall time)
was compared point-by-point against weak-fluctuation theory. Full table in
`sim_vs_theory.txt`.

**PASS criterion, stated before the run:** point-like sim/theory mean ratio
within [0.6, 1.4] and every individual point within [0.5, 1.6].

| Receiver | n | mean sim/theory | min | max | Result |
|---|---|---|---|---|---|
| Point-like (D = 2 mm ≈ 1 pixel) | 18 | **0.980** | 0.907 | 1.041 | **PASS** |
| Finite apertures (D = 50, 100 mm) | 36 | 0.850 | 0.689 | 1.036 | reported, no pass gate |

Representative rows (σ_I², dimensionless):

| Cn² [m^(−2/3)] | L [m] | λ [m] | D [m] | simulated | theory | ratio |
|---|---|---|---|---|---|---|
| 1.00e-16 | 1000 | 8.50e-07 | 0.002 | 4.0128e-03 | 3.9765e-03 | 1.01 |
| 1.00e-15 | 2000 | 1.55e-06 | point | 6.3972e-02 | 7.0950e-02 | 0.90 |
| 1.00e-15 | 3000 | 8.50e-07 | 0.002 | 2.7197e-01 | 2.9982e-01 | 0.91 |
| 1.00e-15 | 3000 | 8.50e-07 | 0.100 | 1.9619e-02 | 2.8474e-02 | 0.69 |
| 1.00e-15 | 3000 | 1.55e-06 | 0.050 | 6.0059e-02 | 7.0698e-02 | 0.85 |

Point-like agreement is within ~10 % across the whole weak-regime sweep —
better than the "tens of percent" expected for a 256² grid with 8 screens.
Plot: `../screenshots/sweep_sigma_i2.png`.

**Observed biases, reported rather than tuned away:**

1. The simulated point index runs systematically ~2 % low on average and up to
   9 % low at the largest σ_R² (0.30). Two contributions: FFT screens contain
   no power below the fundamental frequency 2π/(N·dx) (no subharmonics), and
   the highest-σ_R² points begin to leave the strictly weak regime where
   σ_I² = σ_R² is exact.
2. Finite-aperture indices sit 15 % low on average and 31 % low at D = 100 mm,
   L = 3000 m. Both the missing low-frequency screen power (which most affects
   the large-scale intensity structure that survives aperture averaging) and
   the approximate nature of the Andrews (1992) factor contribute; the two
   cannot be separated with this dataset. This is a genuine known limitation,
   not a tuned tolerance.

## V4 — Benchmark: surrogate vs analytic baseline on held-out simulation

54 rows split 40 train / 14 test (shuffle, seed 0). Surrogate: 5-member MLP
ensemble, (32, 32) hidden units, L-BFGS, log-space features and target; fit
time 2.0 s. Baseline: aperture-averaged Rytov theory. Raw output:
`benchmark_results.txt`.

| Model | RMSE (log10 σ_I²) | median \|rel err\| | max \|rel err\| |
|---|---|---|---|
| MLP surrogate (5-ensemble) | 0.0781 | 0.1665 | 0.2824 |
| Rytov analytic baseline | **0.0429** | **0.0700** | **0.2276** |

**The analytic baseline wins on every metric.** This is the honest and
expected outcome: the benchmark is run entirely inside the baseline's own
validity regime, on 40 training points, so the surrogate can at best
rediscover a closed form the baseline already computes exactly. Reported as
measured; no retuning was done to reverse the ordering.

Uncertainty output: mean ensemble standard deviation on the test set is
2.97e-03 against a mean prediction of 1.77e-02 (≈17 % relative spread),
which is the same order as the surrogate's measured 16.7 % median error —
the ensemble spread is therefore an informative, though not formally
calibrated, error indicator.

Speed: 0.269 ms per surrogate prediction versus ~1.3 s per split-step
simulation point at this grid size, a ~5000× speedup. That, plus
extensibility to regimes where no closed form exists (strong fluctuations,
non-Kolmogorov spectra, inner/outer-scale effects, slant paths), is the
case for a surrogate — not accuracy against Rytov in-regime.

## Summary

| ID | Check | Result |
|---|---|---|
| V1 | Analytic core vs textbook closed form (5 known answers + property tests) | PASS |
| V2 | Simulator numerical sanity (energy, zero-turbulence, reproducibility) | PASS |
| V3 | Simulated σ_I² vs Rytov theory, point-like receiver | PASS (mean ratio 0.980) |
| V3b | Simulated σ_I² vs Rytov + Andrews aperture averaging | Reported: 15 % low on average (documented bias) |
| V4 | Surrogate vs analytic baseline on held-out points | Baseline wins (0.0429 vs 0.0781 RMSE log10) — reported as measured |

Test suite: 50 passed, 0 failed, 0 skipped (`python -m pytest tests/ -q`).

**Not validated:** strong-fluctuation regime (σ_R² > 1), non-Kolmogorov
spectra, inner/outer-scale effects, slant/vertical paths with Cn²(h)
profiles, Gaussian-beam or spherical-wave simulation, beam wander,
temporal statistics, and any comparison against field measurements. No
experimental data was used anywhere in this validation.
