# CnCast — Validation Evidence (Level 2, Research)

**Product:** P019 CnCast 0.1.0 · **Status:** TESTING · **Date of run:** 2026-08-07
**Environment:** Python 3.11.15, numpy 2.4.4, scikit-learn 1.8.0,
Linux-6.18.5-fc-v18-x86_64, 2 CPU cores.

Every number below was produced by running the scripts in this directory during
this build. Raw output is committed:

| script | raw output | covers |
|---|---|---|
| `validation/validate_baselines.py` | `validation/validate_baselines_output.txt` | §1–§3 |
| `validation/benchmark_ml.py` | `validation/benchmark_results.md` | §4–§7 |

Reproduce with, from the product root:

```bash
python validation/validate_baselines.py > validation/validate_baselines_output.txt
python validation/benchmark_ml.py       > validation/benchmark_results.md
```

**What "validation" means here.** §1–§3 validate the implementation against
*published models and closed-form mathematics*. §4–§7 validate the learned model
against a *synthetic generator of our own construction*. Nothing in this
document compares anything to a radiosonde, thermosonde, scintillometer or
SCIDAR measurement, because no such data is used anywhere in this product. See
`DATASET_CARD.md`.

---

## 1. Published baselines reproduce their documented behaviour

### 1.1 Hufnagel-Valley 5/7 at reference altitudes

Computed from `cncast.baselines.hv57`, decomposed into the three published
terms (Andrews & Phillips 2005 Eq. 12.30; Hufnagel 1974 / Valley 1980):

| h [m] | ground term `A e^(-h/100)` | tropopause `2.7e-16 e^(-h/1500)` | high-altitude term | total Cn² [m^-2/3] |
|---:|---:|---:|---:|---:|
| 0 | 1.7000e-14 | 2.7000e-16 | 0.0000e+00 | 1.7270e-14 |
| 10 | 1.5382e-14 | 2.6821e-16 | 3.5576e-43 | 1.5650e-14 |
| 100 | 6.2540e-15 | 2.5259e-16 | 3.2514e-33 | 6.5065e-15 |
| 500 | 1.1455e-16 | 1.9346e-16 | 2.1284e-26 | 3.0801e-16 |
| 1 000 | 7.7180e-19 | 1.3862e-16 | 1.3219e-23 | 1.3939e-16 |
| 5 000 | 3.2789e-36 | 9.6320e-18 | 2.3644e-18 | 1.1996e-17 |
| 10 000 | 6.3241e-58 | 3.4361e-19 | 1.6314e-17 | 1.6657e-17 |
| 15 000 | 1.2198e-79 | 1.2258e-20 | 6.3386e-18 | 6.3509e-18 |
| 20 000 | 2.3526e-101 | 4.3729e-22 | 7.5842e-19 | 7.5885e-19 |

The three terms are asserted to sum to the returned total to 1e-24 m^-2/3 in the
script itself. Known-answer unit tests re-derive the h = 0, h = 100 m and
h = 10 km values by hand (`tests/test_baselines.py`).

### 1.2 The "5/7" property is reproduced

HV 5/7 is named for the values it produces at λ = 0.5 µm on a vertical path.
Computed on a 0–20 km grid at 0.1 m resolution:

| quantity | computed | published nickname value | deviation |
|---|---:|---:|---:|
| r₀ | **4.9624 cm** | 5 cm | −0.75 % |
| θ₀ | **7.0109 µrad** | 7 µrad | +0.16 % |
| seeing FWHM (0.98 λ/r₀) | 2.0367 arcsec | — | — |
| μ₀ = ∫Cn² dh | 2.233985e-12 m^(1/3) | — | — |
| μ_5/3 = ∫Cn² h^(5/3) dh | 8.461974e-07 m² | — | — |

The nickname values are quoted to one significant figure in the literature, so
agreement at the < 1 % level is the strongest statement that can be made; this
is a **pass**.

Grid convergence of r₀ (why the resolution matters):

| grid points over 0–20 km | step | r₀ | relative difference from finest |
|---:|---:|---:|---:|
| 201 | 100 m | 4.78532 cm | 3.57e-02 |
| 2 001 | 10 m | 4.96056 cm | 3.81e-04 |
| 20 001 | 1 m | 4.96243 cm | 3.77e-06 |
| 200 001 | 0.1 m | 4.96245 cm | 0 (reference) |

**Consequence, documented as a limitation:** a 100 m grid biases r₀ by +3.6 %
because it under-resolves the 100 m-scale ground layer. Use ≤ 10 m spacing near
the surface, or a log-spaced grid. The examples use a 24–60 point log grid,
which is why the r₀ they report for HV 5/7 (5.01–5.06 cm) differs from the
4.962 cm above by ~1–2 %.

### 1.3 SLC-Day and SLC-Night branches

Each published branch evaluated against its closed form (Beland 1993;
tabulated in Andrews & Phillips 2005 §12.2.1). All relative errors are exactly
0.0e+00 — the implementation is the published formula:

| check | computed | closed form |
|---|---:|---:|
| SLC-Day, h = 100 m → 3.13e-13/h^1.05 | 2.486247e-15 | 2.486247e-15 |
| SLC-Day, h = 2 000 m → 8.87e-7/h³ | 1.108750e-16 | 1.108750e-16 |
| SLC-Day, h = 10 000 m → 2.0e-16/√h | 2.000000e-18 | 2.000000e-18 |
| SLC-Night, h = 50 m → 2.87e-12/h² | 1.148000e-15 | 1.148000e-15 |
| SLC-Night, h = 500 m → 2.5e-16 | 2.500000e-16 | 2.500000e-16 |
| SLC-Night, h = 3 000 m → 8.87e-7/h³ | 3.285185e-17 | 3.285185e-17 |

**Honest finding — the published SLC-Day fit is discontinuous.** Evaluating
1 µm either side of each branch boundary:

| model | boundary | value below | value above | jump |
|---|---:|---:|---:|---:|
| SLC-Day | 18.5 m | 1.7000e-14 | 1.4622e-14 | −14.0 % |
| SLC-Day | 240 m | 9.9157e-16 | 1.3000e-15 | **+31.1 %** |
| SLC-Day | 880 m | 1.3000e-15 | 1.3016e-15 | +0.1 % |
| SLC-Day | 7 220 m | 2.3567e-18 | 2.3538e-18 | −0.1 % |
| SLC-Night | 18.5 m | 8.4000e-15 | 8.3857e-15 | −0.2 % |
| SLC-Night | 110 m | 2.3719e-16 | 2.5000e-16 | +5.4 % |
| SLC-Night | 1 500 m | 2.5000e-16 | 2.6281e-16 | +5.1 % |
| SLC-Night | 7 200 m | 2.3764e-18 | 2.3570e-18 | −0.8 % |

This is a property of the published piecewise fits, not of this implementation:
the branches were fitted separately and were never constrained to join. SLC-Day
has a genuine 31 % step at 240 m. Users integrating these profiles should expect
small artefacts there. SLC-Night is continuous to better than 6 % everywhere.

Both models are defined as identically zero above their ceilings
(SLC-Day 20 500 m, SLC-Night 20 000 m); this is verified in §1.3 of the raw
output and in `tests/test_baselines.py`.

### 1.4 Integrated quantities for each baseline

λ = 500 nm, zenith, Bufton wind with 5 m/s ground wind, 0.1 m grid:

| model | r₀ [cm] | θ₀ [µrad] | f_G [Hz] | seeing FWHM [arcsec] |
|---|---:|---:|---:|---:|
| HV 5/7 | 4.962 | 7.011 | 71.98 | 2.037 |
| SLC-Day | 4.773 | 11.986 | 59.02 | 2.118 |
| SLC-Night | 8.904 | 13.229 | 36.90 | 1.135 |

Qualitative checks that pass: SLC-Night gives roughly twice the r₀ of SLC-Day
(night-time surface turbulence collapses); the SLC models give larger θ₀ than
HV 5/7 because they carry no jet-stream bump, and θ₀ is dominated by the
h^(5/3)-weighted high-altitude turbulence.

Greenwood frequency versus ground wind (HV 5/7):

| ground wind [m/s] | Bufton 5–20 km rms [m/s] | f_G [Hz] |
|---:|---:|---:|
| 0 | 18.68 | 39.62 |
| 5 | 22.96 | 71.98 |
| 10 | 27.49 | 111.68 |
| 21 | 37.87 | 203.61 |

**Honest note.** HV 5/7 is defined with a *pseudowind* of 21 m/s, but the Bufton
profile with a 5 m/s ground wind integrates to 22.96 m/s, not 21. The two
numbers come from different conventions and are often conflated in secondary
sources; this implementation keeps them separate arguments
(`hufnagel_valley(..., rms_wind_m_s=...)` and `rms_high_altitude_wind(w_ground)`)
so the user chooses explicitly. Nothing in the product silently converts one to
the other.

---

## 2. Closed-form cross-checks of the seeing integrals

### 2.1 Constant-Cn² slab (analytic case)

Cn² = 1.0e-15 m^-2/3, 0–10 km, λ = 1.55 µm:

| quantity | closed form | code | relative error |
|---|---:|---:|---:|
| μ₀ = Cn² H | 1.000000e-11 | 1.000000e-11 | 0 |
| r₀ | 7.848343 cm | 7.848343 cm | 0.00e+00 |
| μ_5/3 = Cn² H^(8/3)/(8/3) | 1.740596e-05 | 1.740596e-05 | 3.70e-11 |

The μ_5/3 residual is trapezoid discretisation of h^(5/3) on 100 001 points, as
expected.

### 2.2 Scaling laws (exact algebraic identities)

| identity | max relative error over the cases run |
|---|---:|
| r₀ ∝ λ^(6/5) (850 nm, 1550 nm vs 500 nm) | 1.5e-16 |
| r₀ ∝ cos(ζ)^(3/5) (ζ = 30°, 60°) | 0.0e+00 |
| θ₀ ∝ cos(ζ)^(8/5) (ζ = 30°, 60°) | 1.5e-16 |
| Greenwood 2.31 λ^(-6/5) form vs (0.102 k²)^(3/5) form | < 2e-3 (the published rounding of 2.307 → 2.31) |

Machine precision — these are exact identities of the coded formulas, checked
also as Hypothesis property tests in `tests/test_seeing.py`.

---

## 3. Physical-plausibility properties of the baselines

The claim actually tested is: **Cn² falls with altitude through the free
troposphere, except for the jet-stream bump that the H-V high-altitude term
exists to produce.** A blanket "monotone above the boundary layer" claim would
be false, and is *not* asserted anywhere in the tests.

| model | decreasing from 300 m up to | Cn²(300 m)/Cn²(19 km) |
|---|---|---:|
| HV 5/7 | 5 908 m (then the jet-stream bump begins) | 864 |
| SLC-Day | 19 km (no rise anywhere above 300 m) | 896 |
| SLC-Night | 1 499 m (the published branch join at 1 500 m steps up 5.1 %) | 172 |

* HV 5/7 pre-bump minimum: h = 5 908 m, Cn² = 1.032e-17 m^-2/3.
* HV 5/7 jet-stream bump peak: h = 9 847 m, Cn² = 1.667e-17 m^-2/3 — i.e. the
  bump peaks within 6 % of the 9.4 km Bufton jet altitude, as intended.
* Strictly decreasing on 300 m–5 km: **True** (this is what the test asserts).
* The high-altitude term scales as v²: increasing the pseudowind from 10 to
  40 m/s multiplies Cn²(10 km) by 14.73; the pure high-altitude term would give
  exactly 16, the difference being the v-independent tropopause term still
  contributing ~2 % at that altitude.

---

## 4. Learned model vs the HV 5/7 baseline on held-out data

Full table in `validation/benchmark_results.md`. Split: 700 scenarios →
394 fit / 131 conformal-calibration / 175 test, **split by scenario, never by
row** (rows from one profile share all five meteorological features; a row-level
split would leak). 28 altitudes per scenario → 4 900 held-out rows.
Fit + calibration wall time **15.9 s** on 2 cores (budget 120 s).

Errors in dex (decades of Cn²); lower is better:

| predictor | RMSE | MAE | bias | p95 abs err |
|---|---:|---:|---:|---:|
| **CnCast learned model** | **0.2095** | 0.1620 | −0.0201 | 0.4190 |
| HV 5/7 (mandated baseline) | 0.5665 | 0.4475 | +0.2686 | 1.1048 |
| SLC day/night | 0.7314 | 0.5199 | −0.0893 | 1.2224 |
| Training climatology (mean profile) | 0.3102 | 0.2395 | +0.0049 | 0.6168 |

The learned model reduces RMSE against HV 5/7 by 63 % (ratio 0.370), and against
the mean training profile by 32 %.

By altitude band:

| band [m] | n | CnCast | HV 5/7 | SLC | climatology |
|---|---:|---:|---:|---:|---:|
| 5–50 | 1400 | 0.2122 | 0.7860 | 0.5748 | 0.3735 |
| 50–300 | 1050 | 0.2023 | 0.6911 | 0.4717 | 0.3424 |
| 300–2000 | 1050 | 0.2089 | 0.2642 | 0.4359 | 0.2105 |
| 2000–8000 | 875 | 0.2257 | 0.3039 | 0.6126 | 0.2333 |
| 8000–20000 | 525 | 0.1879 | 0.3143 | 1.6315 | 0.3351 |

### How this result should be read

**The learned model wins, and the win is close to tautological.** The training
targets are generated from the Hufnagel-Valley family with the ground-layer
strength and the pseudowind driven by exactly the surface variables the model is
given (`DATASET_CARD.md` §3). A model that recovers that mapping must beat a
single fixed climatological curve that has, by construction, no meteorological
inputs at all. The honest interpretation is:

* HV 5/7 is not a weak baseline for its intended purpose — it is a
  climatological average and it is being scored on a task (conditioning on
  today's weather) that it was never designed to do.
* The result demonstrates that the regression machinery, the split protocol and
  the interval calibration work. It demonstrates **nothing about real
  atmospheric skill**.
* The interesting comparison is against the *training climatology* (0.3102 dex),
  which is the strongest non-learned predictor available. The learned model's
  0.2095 dex is a real but modest improvement, and it is a fair measure of how
  much of the generator's meteorological signal was recovered: the generator's
  irreducible scatter (three unobservable smooth modes at 0.12 dex each plus an
  unobservable elevated layer) sets a floor that no model could go below.
* At 300–2 000 m the learned model (0.2089) is statistically indistinguishable
  from the plain climatology (0.2105) — the meteorological features carry almost
  no information about that band in this generator. That is reported here rather
  than omitted.

---

## 5. Integrated seeing from a predicted profile, with one hand check

Test scenario 0: T = 17.70 °C, wind = 11.42 m/s, RH = 76.92 %, hour = 23.93,
day-of-year = 344. 24-point log-spaced grid, λ = 500 nm, zenith:

| quantity | from predicted median | from synthetic truth | from HV 5/7 |
|---|---:|---:|---:|
| r₀ [cm] | 7.3342 | 6.2482 | 5.0130 |
| θ₀ [µrad] | 4.0393 | 3.6102 | 7.0455 |
| f_G [Hz] | 123.84 | 149.44 | 122.74 |

r₀ implied by the interval bounds: 13.1120 cm (from the lower-Cn² bound) to
4.0565 cm (from the upper-Cn² bound) — the truth value 6.2482 cm is inside.

### 5.1 Hand check of r₀ from the predicted profile

The same model, same scenario, evaluated on a deliberately coarse 5-point grid
so the arithmetic can be written out in full. Predicted median Cn²
(from `validation/benchmark_results.md` §4):

| h [m] | Cn² [m^-2/3] |
|---:|---:|
| 5 | 2.852424e-15 |
| 100 | 1.197547e-15 |
| 1 000 | 1.846164e-16 |
| 5 000 | 2.623659e-17 |
| 20 000 | 2.299238e-18 |

**Step 1 — trapezoid integral μ₀ = ∫ Cn² dh, four panels:**

```
5 → 100 m    : ½(2.852424e-15 + 1.197547e-15) × 95    = 2.024985e-15 ×    95 = 1.923736e-13
100 → 1000 m : ½(1.197547e-15 + 1.846164e-16) × 900   = 6.910817e-16 ×   900 = 6.219735e-13
1000 → 5000 m: ½(1.846164e-16 + 2.623659e-17) × 4000  = 1.054265e-16 ×  4000 = 4.217060e-13
5000 →20000 m: ½(2.623659e-17 + 2.299238e-18) × 15000 = 1.426791e-17 × 15000 = 2.140187e-13
                                                                        sum  = 1.4500718e-12 m^(1/3)
```

Script output: `1.450072e-12 m^(1/3)`. **Agrees to all 7 printed digits.**

**Step 2 — r₀ = [0.423 k² sec ζ μ₀]^(−3/5), ζ = 0 so sec ζ = 1:**

```
k     = 2π / 500e-9                    = 1.2566371e7  rad/m
k²                                     = 1.5791367e14 rad²/m²
0.423 k²                               = 6.6797483e13
X = 0.423 k² μ₀ = 6.6797483e13 × 1.4500718e-12  = 96.861149     (dimensionless × m^-5/3 … see note)
ln X                                   = 4.573278
−(3/5) ln X                            = −2.7439668
r₀ = e^(−2.7439668)                    = 0.06431470 m = 6.431470 cm
```

Script output: `6.431470 cm`. **Agrees to all 7 printed digits.** (Unit note:
X carries units m^-5/3; r₀ = X^(−3/5) is then in metres, which is the standard
bookkeeping of the Fried formula.)

The coarse-grid value (6.431 cm) differs from the 24-point value (7.334 cm) by
14 %, because 4 panels cannot resolve the ground layer — a concrete illustration
of the grid-resolution limitation quantified in §1.2. The 24-point value is the
one to use; the 5-point value exists solely to make the hand check tractable.

---

## 6. Prediction-interval coverage on held-out data

Nominal central coverage 90 % (5th/95th conditional percentiles):

| interval | nominal | **empirical coverage** | mean width [dex] |
|---|---:|---:|---:|
| raw quantile GBR (α = 0.05 / 0.95) | 0.900 | **0.8033** | 0.5575 |
| conformally calibrated (CQR, Romano et al. 2019) | 0.900 | **0.8988** | 0.7249 |

The raw quantile models under-cover by ~10 points, which is the expected
consequence of fitting conditional quantiles on the training set. Split
conformal calibration on 131 scenarios that are disjoint from both fit and test
adds δ = +0.0838 dex to each bound and brings coverage to 0.8988 against a
nominal 0.900.

Resolution of this estimate: the binomial standard error on 4 900 rows is
0.0043, but the rows are not independent (28 per scenario), so the effective
sample size is nearer the 175 test scenarios. Treat ±0.02 as the resolution.

By altitude band (calibrated):

| band [m] | n | coverage | mean width [dex] |
|---|---:|---:|---:|
| 5–50 | 1400 | 0.8821 | 0.6947 |
| 50–300 | 1050 | 0.8914 | 0.7170 |
| 300–2000 | 1050 | 0.9095 | 0.7014 |
| 2000–8000 | 875 | 0.9314 | 0.8409 |
| 8000–20000 | 525 | 0.8819 | 0.6752 |

Coverage is within ±0.02 of nominal in every band. Conformal calibration is
*marginal* (it guarantees the aggregate rate, not the rate in every
sub-population), so the mild under-coverage in the top and bottom bands is
expected behaviour, not a defect — but it means the interval should not be
trusted band-by-band to better than a couple of points.

**Derived quantities are a different story.** In `examples/r0_comparison.py`,
the r₀ band obtained by integrating the lower and upper Cn² bounds contains the
truth **98.9 %** of the time against a nominal 90 %. That interval is
conservative because the bounds are perfectly correlated across altitude by
construction, whereas the true profile errors are only partly correlated.
Do not read the derived r₀ band as a calibrated 90 % interval; it is an upper
bound on the uncertainty.

---

## 7. Reproducibility check

Re-running `train_default_model()` in the same session:

* identical test features on re-run: **True**
* max |prediction difference| across re-runs: **0.000e+00 dex**
* conformal δ identical: **True**

`tests/test_dataset.py` and `tests/test_model.py` additionally assert bit-level
reproducibility of the scenario draw, the feature table, the split and the
fitted predictions, and assert that a different `data_seed` changes the model.

---

## 8. What was NOT validated (limits of this evidence)

1. **No comparison with measured Cn² of any kind.** No radiosonde, thermosonde,
   scintillometer, SCIDAR, MASS/DIMM or lidar data was used. The "truth" in
   §4–§6 is a generative process defined in `cncast/dataset.py`.
2. **The meteorology → Cn² coefficients are invented.** The signs follow the
   literature (daytime convective boundary layer, winter jet); the magnitudes
   are hand-chosen. No fit to observations exists.
3. **The baselines themselves are climatologies.** Reproducing HV 5/7 to 0.75 %
   validates the implementation, not the model's applicability to any given
   site or night.
4. **Not modelled anywhere:** real boundary-layer dynamics, terrain and local
   orography, temperature inversions and their sharp turbulent caps,
   jet-stream position variability, seasonal site climatology, humidity
   contribution to the refractive-index structure at IR/RF wavelengths, and
   optical turbulence generated by the observing platform itself.
5. **Spherical-wave and slant-path geometry** beyond the plane-parallel sec ζ
   factor is out of scope; the sec law degrades above ~70° zenith and is not
   corrected for Earth curvature.
6. **Only one nominal coverage (90 %)** was calibrated and measured. Other
   coverages would need their own calibration run.

---

## 9. Test suite

`python -m pytest tests/ -q` from the product root: **112 passed** in 17.5 s
(0 failed, 0 skipped, 0 xfail). `ruff check src/ tests/`: clean.

Composition: known-answer tests for every baseline branch and every seeing
formula (hand computations shown in the test docstrings), Hypothesis property
tests for the algebraic scaling identities, physical-plausibility tests,
input-validation tests for every public entry point, seeded-reproducibility
tests, and an interval-coverage test.
