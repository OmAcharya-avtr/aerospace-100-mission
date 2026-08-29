# TurbScope — Validation Evidence (Level 2, Research)

**Product:** P013 TurbScope 0.1.0 · **Status:** TESTING · **Date of run:** 2026-08-29
**Environment:** Python 3.11, numpy 2.4.4, scikit-learn, scipy, Linux, 2 CPU cores.

Every number below was produced by running the scripts in this directory during
this build. Raw output is committed:

| script | raw output | covers |
|---|---|---|
| `validation/round_trip_recovery.py` | `validation/round_trip_recovery_output.txt` | §1 |
| `validation/saturation_regime.py` | `validation/saturation_regime_output.txt` | §2 |
| `validation/benchmark_ml.py` | `validation/benchmark_results.md` | §3–§4 |

Reproduce with, from the product root:

```bash
python validation/round_trip_recovery.py > validation/round_trip_recovery_output.txt
python validation/saturation_regime.py   > validation/saturation_regime_output.txt
python validation/benchmark_ml.py        > validation/benchmark_results.md
```

**What "validation" means here.** Every quantity in this document is
validated against *closed-form mathematics this package itself implements*
(the Rytov/DIMM forward models, cited to Tatarski 1961, Andrews & Phillips
2005, Wang/Ochs/Lawrence 1978, Sarazin & Roddier 1990) or against a
*synthetic generator this product defines* (`turbscope.synthetic`). Nothing
here compares against a field measurement of the real atmosphere from any
scintillometer or DIMM instrument — see `DATASET_CARD.md`.

---

## 1. Round-trip recovery of a known Cn2 in the weak-fluctuation regime

### 1.1 Noiseless algebraic recovery vs sigma_R² (isolates weak-theory model error)

`Cn2 -> forward model (full saturation curve) -> sigma_I^2 -> weak-regime
inversion -> Cn2_recovered`, no sensor noise, L = 500 m:

| target σ_R² | Cn2_true [m⁻²/³] | Cn2_recovered | rel. error |
|---:|---:|---:|---:|
| 0.0001 | 2.2748e-17 | 2.2749e-17 | 0.0064% |
| 0.001 | 2.2748e-16 | 2.2762e-16 | 0.0641% |
| 0.01 | 2.2748e-15 | 2.2891e-15 | 0.6302% |
| 0.05 | 1.1374e-14 | 1.1706e-14 | 2.9187% |
| 0.1 | 2.2748e-14 | 2.3949e-14 | 5.2795% |
| 0.2 | 4.5495e-14 | 4.9357e-14 | 8.4866% |
| **0.3 (WEAK_REGIME_MAX)** | 6.8243e-14 | 7.5029e-14 | **9.9433%** |
| 0.5 (beyond threshold) | 1.1374e-13 | 1.2377e-13 | 8.8185% |
| 1.0 (beyond threshold) | 2.2748e-13 | 2.1220e-13 | −6.7170% |

The error grows monotonically with σ_R² up to the `WEAK_REGIME_MAX_SIGMA_R2 =
0.3` threshold (≈10% at the boundary), which is exactly what that threshold
is for: it is a *conservative* validity bound, not a point where error is
zero. Beyond it the error stops growing monotonically because the full curve
itself turns over (§2) — the sign flip at σ_R²=1.0 is a first hint of the
saturation behaviour quantified in §2, not noise.

### 1.2 DIMM noiseless round trip — exact to machine precision

Six probe cases (three σ_R² targets × two components): relative error
2.167e-16, 2.167e-16, 1.387e-16, 1.387e-16, 0.0, 0.0. This confirms the
algebraic identity proved in `tests/test_dimm.py` — DIMM's inversion has
**no model-form error at any turbulence strength** in this product's forward
model (no saturation term is defined for DIMM; see Limitations for what that
does and does not mean about real instruments).

### 1.3 Round trip with realistic sensor noise, one canonical scenario, N=1000 draws

True Cn2_path = 1.160622e-14 m⁻²/³, L = 300 m, target σ_R² = 0.02.
Measurement noise: scintillometer 8%, DIMM 10% relative 1-σ (both hand-chosen
and documented in `DATASET_CARD.md`, not vendor specifications):

| method | mean rel. error | std | RMSE | median \|err\| |
|---|---:|---:|---:|---:|
| scintillometer weak inversion | +1.62% | 8.03% | 8.19% | 5.41% |
| DIMM longitudinal inversion | −0.18% | 9.83% | 9.83% | 6.61% |
| **multi-sensor fused (inverse-variance)** | **−0.28%** | **5.42%** | **5.42%** | **3.64%** |

Fusing the three closed-form single-sensor estimates (§`inversion.py`)
reduces RMSE by ~34% relative to the scintillometer alone, consistent with
the classical inverse-variance-weighting reduction expected from combining
partially-independent unbiased estimators (Bevington & Robinson 2003, Ch. 4).

### 1.4 Headline aggregate number: 1639 independent weak-regime scenarios, one noisy draw each

| method | median \|rel err\| | mean | p90 |
|---|---:|---:|---:|
| **multi-sensor fused** | **3.86%** | 4.63% | 9.83% |
| scintillometer alone | 5.89% | 7.05% | 14.47% |

**Headline result: in the weak-fluctuation regime, TurbScope recovers a
known path-averaged Cn2 to a median relative error of 3.9% (multi-sensor
fused closed form) with realistic 8–10% sensor noise.**

---

## 2. Demonstration and quantification of the saturation failure mode

### 2.1 Shape of the heuristic saturation curve

(`turbscope.scintillometer.scintillation_index_full` — a bridging function
built for this product with the correct *qualitative* weak-limit and
strong-turbulence-asymptote physics; see the module docstring "Honesty
note" before reading any number here as a literature-exact prediction.)

* Asymptote as σ_R² → ∞: **0.999001** (design target 1.0).
* Local maximum ("focusing" overshoot) at **σ_R² = 1.8584**, σ_I² = **1.1261**.
* Overshoot above the asymptote: **+0.1271 (12.72%)**.

### 2.2 The multi-valued inversion band

Multi-valued σ_I² band: **[0.9990, 1.1261]**, width **0.1271**. Of 25 probe
measurements spanning the band, **25/25 (100%)** were found genuinely
multi-valued by `invert_cn2_all_roots` (two roots each).

### 2.3 Concrete worked example

At L = 1000 m, a measured σ_I² = 1.062536 has **two** consistent σ_R²
roots — 1.3249 and 2.6524 — giving two Cn2_path candidates, **8.457e-14** and
**1.693e-13 m⁻²/³**, a **2.00×** ratio. No information in σ_I² alone
distinguishes which is correct; only an independent sensor (DIMM, in this
product) or prior knowledge of the regime resolves it.

### 2.4 Quantified failure of the weak-regime baseline outside its validity range (noiseless)

| true σ_R² | true Cn2 | baseline Cn2 | rel. error |
|---:|---:|---:|---:|
| 0.5 | 3.1917e-14 | 3.4731e-14 | +8.82% |
| 1.0 | 6.3834e-14 | 5.9546e-14 | −6.72% |
| 1.858 (peak) | 1.1863e-13 | 7.1881e-14 | **−39.41%** |
| 3.0 | 1.9150e-13 | 6.5153e-14 | **−65.98%** |
| 10.0 | 6.3834e-13 | 5.8048e-14 | **−90.91%** |
| 50.0 | 3.1917e-12 | 6.2582e-14 | **−98.04%** |

The weak-regime baseline saturates near its own asymptote-adjacent value
regardless of the true Cn2, so its error grows toward −100% as truth grows —
the classic signature of a sensor whose signal has itself saturated.

### 2.5 Aggregate failure across 1025 independently drawn saturated scenarios (with realistic sensor noise)

**Headline result: applying the classical weak-theory inversion outside its
validity range gives a median relative error of 89.5% (mean 76.5%, p90
97.8%)** — roughly an order of magnitude worse than the same formula's 5.9%
median error in the weak regime (§1.4). This is the failure the saturation
regime requires this product to document, and it is not hidden or tuned
away: the baseline is simply wrong there, plainly and by a large margin.

---

## 3. Learned model vs classical baselines on held-out data

Split: 900 scenarios → 506 fit / 169 conformal-calibration / 225 test,
**split by scenario, never by row** (3 noisy realisations per scenario share
the same ground truth; a row-level split would leak). 675 held-out test
rows. Fit + calibration wall time **3.25 s** on 2 cores (budget 120 s).

### 3.1 Overall held-out error (dex = decades of Cn2; lower is better)

| predictor | RMSE | MAE | bias | p95 |
|---|---:|---:|---:|---:|
| **TurbScope learned model** | 0.0714 | 0.0497 | −0.0054 | 0.1224 |
| **Scintillometer weak baseline (mandated)** | 0.6577 | 0.3578 | −0.3332 | 1.6216 |
| DIMM-only baseline | **0.0318** | 0.0249 | −0.0020 | 0.0598 |
| Training mean (learned-nothing floor) | 1.6877 | 1.4449 | −0.0434 | 2.9402 |

**Learned/mandated-baseline RMSE ratio: 0.1086 (89% reduction).** Against the
mission-mandated comparator — the classical closed-form single-sensor
(scintillometer) inversion, the one with the documented saturation failure —
**the learned model wins decisively.**

### 3.2 Honest reading: a stronger baseline that the learned model does NOT beat

**The DIMM-only closed-form single-sensor baseline (0.0318 dex) beats the
learned multi-sensor model (0.0714 dex) outright.** This is reported plainly,
as required: it is a genuine negative result for the "does the learned model
win" question when the strongest available closed-form baseline is used
instead of the mandated one.

Why: in this product's synthetic design, DIMM's inversion has **zero
saturation-related model-form error at any turbulence strength** (§1.2) — the
generator defines no DIMM saturation term — so DIMM-only is limited purely by
its 10% sensor noise, propagated exactly linearly (`inversion.py`). No
learned model, however good, can beat a noise-limited unbiased linear
estimator on a metric that measures exactly that noise. The learned model's
extra error comes from (a) `GradientBoostingRegressor`'s inherent
piecewise-constant/staircase approximation, and (b) the scintillometer
feature actively misleading it in the saturated regime some of the time
(§3.3 below shows the learned model is *not* immune to the saturated
scintillometer signal, just far more robust to it than the baseline that
ignores DIMM entirely).

**Mission takeaway:** on this problem, most of the value of "multi-sensor" is
*avoiding a broken sensor*, not fusing two working ones. A much simpler rule
— trust DIMM and ignore the scintillometer once it is flagged saturated —
would likely match or beat the learned model here. That simpler rule is not
implemented as a baseline in this product (`DimmOnlyBaseline` is the closest
available proxy and is reported above); a fair test of "smart switching vs
learned fusion" is future work (`README.md` Roadmap).

### 3.3 Error broken down by TRUE regime

| predictor | RMSE weak | RMSE saturated |
|---|---:|---:|
| TurbScope learned model | 0.0648 | 0.0784 |
| Scintillometer weak baseline (mandated) | 0.0380 | **0.9665** |
| DIMM-only baseline | 0.0304 | 0.0334 |
| Training mean | 1.6053 | 1.7788 |

n = 363 weak rows, 312 saturated rows. The mandated baseline is competitive
in the weak regime (0.038, actually *better* than the learned model there)
and catastrophically worse in the saturated regime (0.9665, consistent with
§2.5's 89.5% median relative error) — the learned model is far more uniform
across regimes because it has access to the DIMM channel, which does not
share the failure.

---

## 4. Prediction-interval coverage on held-out data

Nominal central coverage 90% (5th/95th conditional quantiles from three
`GradientBoostingRegressor` models, split-conformal calibrated — Romano,
Patterson & Candès 2019, *Conformalized Quantile Regression*, NeurIPS 32):

| interval | nominal | **empirical coverage** | mean width [dex] |
|---|---:|---:|---:|
| raw quantile GBR | 0.900 | 0.7985 | 0.2869 |
| **conformally calibrated** | 0.900 | **0.8770** | 0.3333 |

The raw quantile models under-cover by ~10 points (expected — quantiles
fitted on the training set are optimistic on unseen data); conformal
calibration on 169 scenarios disjoint from both fit and test brings coverage
to 0.877 against a nominal 0.900.

By regime (calibrated model):

| regime | coverage | mean width [dex] |
|---|---:|---:|
| weak (n=363) | 0.8512 | 0.3464 |
| saturated (n=312) | 0.9071 | 0.3181 |

**Resolution of these estimates.** The binomial standard error on 675 rows is
~0.013, but rows are not independent (3 per scenario share ground truth), so
the effective sample size is nearer the 225 test scenarios (SE ≈ 0.022);
treat ±0.02–0.03 as the resolution of the coverage numbers above. Conformal
coverage is *marginal* (guaranteed in aggregate, not in every
sub-population, e.g. weak vs saturated) — the mild weak-regime
under-coverage is expected behaviour under that caveat, not a defect, but the
interval should not be trusted band-by-band to better than a few points.

---

## 5. Reproducibility

Re-running `train_default_model()` in the same session (`benchmark_ml.py`
§4): identical test features on re-run: **True**; max \|prediction
difference\| across re-runs: **0.000e+00 dex**; conformal delta identical:
**True**. `tests/test_model.py` and `tests/test_dataset.py` additionally
assert bit-level reproducibility of the scenario draw, the feature table and
the fitted predictions in CI, and that a different seed changes the result.

---

## 6. What was NOT validated (limits of this evidence)

1. **No comparison with a measured Cn2 of any kind.** No real scintillometer
   or DIMM data was used anywhere. "Truth" throughout is the seeded synthetic
   generator in `turbscope.synthetic`; see `DATASET_CARD.md`.
2. **The saturation model (`scintillation_index_full`) is a heuristic built
   for this product**, not a literature curve fit — it reproduces the
   documented qualitative shape (weak Rytov limit, focusing overshoot,
   order-unity asymptote) but its numerical peak height/location are not
   claimed to match any specific published measurement.
3. **DIMM's own real-world limitations are not modelled.** The
   long-baseline, diffraction-neglected coefficients (§`dimm.py`) are a
   documented simplification; real DIMM performance also degrades at very
   poor seeing (D comparable to or smaller than r0) — a regime this product
   does not separately simulate, which is part of why DIMM-only "wins" §3.2
   so cleanly here.
4. **The scintillometer/DIMM noise levels (8%/10% relative) are hand-chosen
   illustrative values**, not vendor specifications — see `DATASET_CARD.md`.
5. **Only one nominal coverage (90%) was calibrated and measured.** Other
   coverages would need their own calibration run.
6. **Anisoplanatism, beam wander, absorption/attenuation and non-Kolmogorov
   spectra** are out of scope; only the Kolmogorov weak-theory Rytov variance
   and its heuristic saturated extension are modelled.

---

## 7. Test suite

`python -m pytest tests/ -q` from the product root: **122 passed**, 0 failed,
0 skipped, 0 xfail. `ruff check src/ tests/ examples/ validation/`: clean.

Composition: known-answer tests with hand-derived expected values (shown in
test docstrings) for every forward/inverse formula, Hypothesis property
tests for the exact linear-inverse and scaling identities, input-validation
tests for every public entry point, edge-case tests, a CLI test suite, and
integration tests spanning the full forward → synthesis → both inversion
paths.
