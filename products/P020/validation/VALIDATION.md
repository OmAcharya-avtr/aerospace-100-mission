# AtmoProfile 0.1.0 — Validation evidence (Level 2, Research)

Every number in this file was produced by running the scripts in this directory
in the build session on 2026-08-07 (Python 3.11.15, numpy 2.4.4, scipy 1.17.1,
2 CPU cores). Each script is rerunnable from this directory with
`python3 <script>.py` and writes its raw output next to itself; the raw files
are the primary evidence and this document summarises them.

| Script | Raw output | Wall time | Result |
|---|---|---:|---|
| `constant_slab_closed_forms.py` | `constant_slab_closed_forms_output.txt` | 0.8 s | 0 failures |
| `zenith_scaling.py` | `zenith_scaling_output.txt` | 3.3 s | 0 failures |
| `standard_profiles_literature.py` | `standard_profiles_literature_output.txt` | 1.8 s | 0 blocking, **1 reported failure** |
| `quadrature_convergence.py` | `quadrature_convergence_output.txt` | 18.6 s | 0 failures |

Total validation wall time **24.5 s**, well inside the 3-minute compute budget.
There is no randomness anywhere in this package, so every number below is
reproducible exactly, without seeds.

**Citation policy.** Citations are given at work level (author, title, edition,
year). No page, equation, figure or table numbers are quoted anywhere in this
product, because none were verified against a physical copy during the build.
Where a value is attributed to the literature it is either (a) a named,
universally reproduced result (the HV 5/7 "5 cm / 7 µrad" property, Greenwood's
0.102, the Rytov 2.25), or (b) a coefficient derived here from such a result,
with the arithmetic shown.

---

## 1. Constant-Cn² slab: closed forms vs code, hand arithmetic

Reference case: Cn² = 1×10⁻¹⁵ m^(−2/3) constant from 0 to H = 1000 m, λ = 500 nm,
zenith angle 0. Shared arithmetic:

```
k        = 2π/500e-9            = 1.2566370614e7  rad/m
k²                               = 1.5791367042e14 m^-2
k^(7/6)                          = 1.9160706038e8  m^(-7/6)
μ_0      = Cn² H                 = 1.0e-12         m^(1/3)
μ_(5/3)  = Cn² (3/8) H^(8/3)     = 3.75e-8         m^2      [1000^(8/3) = 1e8]
μ_(5/6)  = Cn² (6/11) H^(11/6)   = 1.7248787e-10   m^(7/6)  [1000^(11/6) = 3.16227766e5]
```

**Fried parameter** — r₀ = [0.423 k² μ_0]^(−3/5):

```
bracket = 0.423 × 1.5791367042e14 × 1.0e-12 = 66.797483
ln(66.797483) = 4.201665 ; × 0.6 = 2.520999 ; exp = 12.441034
r0 = 1/12.441034 = 0.08037925 m
```

**Isoplanatic angle** — θ₀ = [2.914 k² μ_(5/3)]^(−3/5):

```
bracket = 2.914 × 1.5791367042e14 × 3.75e-8 = 1.7256022e7
ln = 16.663645 ; × 0.6 = 9.998187 ; exp = 21957.3
θ0 = 4.548159e-5 rad = 45.4816 µrad
```

**Rytov variance, plane wave** — σ_R² = 2.25 k^(7/6) μ_(5/6):

```
2.25 × 1.9160706e8 × 1.7248787e-10 = 2.25 × 0.03304990 = 0.07436226
```

**Rytov variance, spherical wave** — weight u^(5/6)(1−u/L)^(5/6) integrates to
L^(11/6)·B(11/6,11/6) with B(11/6,11/6) = Γ(11/6)²/Γ(11/3) = 0.22053566:

```
μ = 1e-15 × 0.22053566 × 3.16227766e5 = 6.973668e-11
σ² = 2.25 × 1.9160706e8 × 6.973668e-11 = 0.03006581
```

Code vs hand (all agreement tolerances 1e-9 relative; **all PASS**):

| Quantity | Hand value | Code value | Relative difference |
|---|---|---|---:|
| μ_0 | 1.0000000000e-12 | 1.0000000000e-12 | 0.00e+00 |
| μ_(5/3) | 3.7500000000e-08 | 3.7500000000e-08 | 1.24e-15 |
| μ_(5/6) | 1.7248787237e-10 | 1.7248787237e-10 | 5.99e-16 |
| r₀ (plane) | 8.0379248737e-02 m | 8.0379248737e-02 m | 0.00e+00 |
| r₀ (spherical, both directions) | 1.4478553729e-01 m | 1.4478553729e-01 m | ≤1.9e-16 |
| θ₀ | 4.5481593687e-05 rad | 4.5481593687e-05 rad | 8.94e-16 |
| h̄ = [μ_(5/3)/μ_0]^(3/5) | 5.5516075873e+02 m | 5.5516075873e+02 m | 8.19e-16 |
| σ_R² (plane) | 7.4362261899e-02 | 7.4362261899e-02 | 7.46e-16 |
| σ_R² (spherical) | 3.0065805300e-02 | 3.0065805300e-02 | 5.77e-16 |

**Cross-checks against the textbook homogeneous-path forms** (these are exact
arithmetic consequences of the slant-path constant 2.25, not independent
claims — but they are the standard published numbers, so agreement is
meaningful):

| Implied coefficient | Computed | Textbook value | Difference |
|---|---:|---:|---:|
| plane, 2.25 × 6/11 | 1.227273 | 1.23 | 0.22 % |
| spherical, 2.25 × B(11/6,11/6) | 0.496205 | 0.50 | 0.76 % |
| ratio spherical/plane | 0.404315 | 0.40 | 1.08 % |

The differences are the rounding of the published 3-significant-figure
coefficients, not modelling error.

**Coefficient derivations** (from `tests/test_known_answers.py`, all PASS):
`2·(24/5·Γ(6/5))^(5/6) = 6.883877` ("6.88"); `2.914/6.883877 = 0.4233080`
("0.423", the package uses the published 0.423, a +0.043 % effect on r₀);
`(0.423/2.914)^(3/5) = 0.3141308` ("θ₀ = 0.314 r₀/h̄"); `0.102^(3/5)(2π)^(6/5)
= 2.30662` ("2.31"). The identity θ₀ = 0.3141308 r₀/h̄ is reproduced to
≤1e-12 relative for the slab, HV5/7 and SLC-Day.

---

## 2. Zenith-angle scaling against the analytic sec(ζ) powers

`zenith_scaling.py`, λ = 500 nm, ζ ∈ {0, 10, 20, 30, 40, 50, 60}°, for all
three standard profiles. Two independent checks per quantity:

1. **direct**: max |Q(ζ)/[Q(0)·secζ^p] − 1| over the grid;
2. **blind fit**: least-squares slope of ln Q against ln sec ζ, which recovers
   the exponent without being told it — this is what would catch an exponent
   silently folded into a constant.

| Quantity | Analytic p | Fitted p (all 3 profiles) | Max deviation |
|---|---:|---:|---:|
| r₀ | −0.600000 | −0.600000000 | 2.2e-16 |
| θ₀ | −1.600000 | −1.600000000 | 2.2e-16 |
| f_G | +0.600000 | +0.600000000 | 2.2e-16 |
| σ_R² plane | +1.833333 | +1.833333333 | 2.2e-16 |
| σ_R² spherical | +1.833333 | +1.833333333 | 2.2e-16 |
| σ_I² (weak) | +1.833333 | +1.833333333 | 2.2e-16 |

All 18 checks PASS at machine precision. Worked example (HV5/7, 500 nm,
sec 45° = 1.41421356):

```
r0(0)  = 4.96245 cm ; r0(45) = 4.03076 cm ; ratio 0.81225240 ; sec^(-3/5) 0.81225240
σ_R²(0)= 0.232619   ; (45)   = 0.439125   ; ratio 1.88774863 ; sec^(11/6) 1.88774863
f_G(0) = 71.8705 Hz ; (45)   = 88.4830 Hz ; ratio 1.23114441 ; sec^(3/5)  1.23114441
```

Agreement is exact because the sec(ζ) factor is applied analytically outside
the quadrature, which is the intended design: the vertical integral is
evaluated once and the airmass power is applied per quantity, explicitly.

**Derived regime boundary.** With σ_R²(0) = 0.232619 for HV5/7 at 500 nm and a
sec^(11/6) growth, the weak-fluctuation limit σ_R² = 1 is reached at
sec ζ = (1/0.232619)^(6/11) = 2.2155, i.e. **ζ = 63.17°**. Beyond that angle
the package warns and the weak-regime scintillation index must not be used.

---

## 3. Standard profile models vs literature

`standard_profiles_literature.py`.

### 3.1 HV 5/7 definitional check — PASS

The Hufnagel-Valley 5/7 model is *named* for producing r₀ = 5 cm and
θ₀ = 7 µrad at λ = 0.5 µm on a vertical path (Andrews & Phillips, *Laser Beam
Propagation through Random Media*, 2nd ed., SPIE 2005; Hardy, *Adaptive Optics
for Astronomical Telescopes*, OUP 1998).

| Quantity | Computed | Named value | Relative | Tolerance | Result |
|---|---:|---:|---:|---:|---|
| r₀ at 500 nm | 4.9624 cm | 5 cm | 0.75 % | 2 % | PASS |
| θ₀ at 500 nm | 7.0109 µrad | 7 µrad | 0.16 % | 2 % | PASS |

This is the single strongest external check available for this package: it
exercises the profile model, the quadrature, both coefficients and the unit
handling simultaneously, against a value fixed by the model's own name.

### 3.2 All models at 500 nm and 1550 nm (vertical path, 0–20 km)

| model | λ [nm] | r₀ [cm] | seeing ["] | θ₀ [µrad] | h̄ [m] | f_G [Hz] | σ_R² plane | weak? |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| HV5/7 | 500 | 4.962 | 2.037 | 7.011 | 2223.5 | 71.87 | 0.2326 | True |
| HV5/7 | 1550 | 19.290 | 1.624 | 27.252 | 2223.5 | 18.49 | 0.0621 | True |
| SLC-Day | 500 | 4.339 | 2.329 | 10.829 | 1258.7 | 65.03 | 0.2879 | True |
| SLC-Day | 1550 | 16.867 | 1.858 | 42.096 | 1258.7 | 16.73 | 0.0769 | True |
| SLC-Night | 500 | 7.602 | 1.329 | 12.553 | 1902.4 | 41.59 | 0.1561 | True |
| SLC-Night | 1550 | 29.551 | 1.060 | 48.795 | 1902.4 | 10.70 | 0.0417 | True |

(f_G uses the Bufton wind with v_g = 5 m/s; seeing = 0.98 λ/r₀.)

### 3.3 Literature-band check at 500 nm

**Range used:** r₀ ≈ 5–20 cm at 0.5 µm for ground sites, equivalently roughly
0.5–2 arcsec of long-exposure seeing. **Source:** the range quoted in Hardy,
*Adaptive Optics for Astronomical Telescopes*, OUP 1998; Roddier, "The Effects
of Atmospheric Turbulence in Optical Astronomy", Progress in Optics XIX, 1981;
Andrews & Phillips 2005. It describes good astronomical sites, which is *not*
what two of these three models represent, so the check is stated with that
caveat and any excursion is reported with its physical explanation.

| model | r₀ at 500 nm | Inside 5–20 cm? | Interpretation |
|---|---:|---|---|
| HV5/7 | 4.962 cm | 0.75 % below the floor | HV5/7 is *defined* to give 5 cm, i.e. it sits exactly on the floor of the band; the shortfall is the model's own arithmetic |
| SLC-Day | 4.339 cm | 13 % below the floor | expected: a daytime sea-level model dominated by surface convection, a stronger regime than the astronomical-site statistics the band describes |
| SLC-Night | 7.602 cm | inside | expected |

Expected ordering SLC-Day < HV5/7 < SLC-Night: **PASS**. No value was tuned to
enter the band.

### 3.4 Wavelength scaling 500 nm → 1550 nm — PASS

r₀(1550)/r₀(500) must equal (1550/500)^(6/5) = 3.88717446 exactly. Computed for
all three models: 3.88717446, relative deviation ≤ 2.2e-16.

### 3.5 Bufton rms wind vs the HV pseudowind — **REPORTED FAILURE**

The Hufnagel-Valley model is parameterised by v = the rms wind over 5–20 km,
and the literature associates the Bufton wind model with the HV 5/7 value
v = 21 m/s. Evaluating that rms directly:

```
v_rms = [ (1/15000) ∫_5000^20000 (5 + 30 exp(-((h-9400)/4800)²))² dh ]^(1/2)
      = 22.9637 m/s          (computed, adaptive quadrature)
```

against 21 m/s → **9.35 % discrepancy, FAIL at a 2 % tolerance**. The ground
wind that *would* reproduce 21 m/s exactly is 2.7541 m/s (computed for
reference only; it is not used anywhere in the package).

This is a failure of an external consistency claim, not of the code: both the
HV model (with its published v = 21 m/s) and the Bufton model (with its
published coefficients) are implemented as published, and they do not agree.
The convention behind the quoted 21 m/s — band limits, treatment of the ground
term, or an added slew-rate term ω_s·h that some formulations of Bufton's model
include — could not be established during this build. Nothing was tuned: the
published parameters are retained and the discrepancy is documented here, in
the `hufnagel_valley` and `bufton_wind` docstrings, and in the README
Limitations. It is counted separately from the blocking self-consistency
checks, all of which pass.

---

## 4. Quadrature convergence

`quadrature_convergence.py`. Reference for the relative-error column is the
default adaptive Gauss-Kronrod (QUADPACK) result, which integrates panel by
panel between the profile's declared breakpoints.

### 4.1 r₀ under Simpson grid refinement (0–20 km, 500 nm)

| nodes | HV5/7 rel. err | SLC-Day rel. err | SLC-Night rel. err |
|---:|---:|---:|---:|
| 201 | 2.257e-03 | 4.244e-03 | 1.968e-02 |
| 801 | 9.835e-06 | 1.002e-02 | 5.555e-03 |
| 3201 | 3.869e-08 | 1.455e-03 | 3.755e-03 |
| 12801 | 1.512e-10 | 2.923e-04 | 2.914e-04 |
| 64001 | 2.418e-13 | 3.096e-05 | 7.251e-05 |

HV5/7 (smooth) converges at the expected Simpson rate and the change between
the last two grids is 1.5e-10 — the integral is converged. The SLC models are
piecewise **discontinuous**, so Simpson converges slowly and *not monotonically*
(the error depends on where the grid falls relative to each jump); this is a
property of the rule, not a defect, and is precisely why the default method is
adaptive with declared breakpoints.

### 4.2 Exactness of the adaptive rule on a discontinuous profile — PASS

For SLC-Day, μ_0 assembled by hand from the analytic integral of each band:

```
∫0^18.5   1.7e-14 dh                       = 3.145000e-13
∫18.5^110 3.13e-13 h^-1.05 dh              (= 3.13e-13 [h^-0.05/-0.05])
∫110^1500 1.3e-15 dh                       = 1.807000e-12
∫1500^7200 8.87e-7 h^-3 dh                 (= 8.87e-7 × ½ (1500^-2 - 7200^-2))
∫7200^20000 2.0e-16 h^-0.5 dh              (= 2.0e-16 × 2 (20000^0.5 - 7200^0.5))
--------------------------------------------------------------------
total (analytic)   = 2.794058935094e-12 m^(1/3)
adaptive quadrature= 2.794058935094e-12 m^(1/3)   relative difference 2.9e-16  PASS
Simpson, 64001 nodes = 2.794203126417e-12          relative difference 5.2e-05
```

The adaptive rule is exact to machine precision on the discontinuous model;
Simpson on the finest grid tested is 11 orders of magnitude worse.

### 4.3 Higher moments (HV5/7) — PASS

| nodes | μ_(5/3) rel. err | μ_(5/6) rel. err |
|---:|---:|---:|
| 201 | 9.470e-05 | 5.042e-03 |
| 801 | 1.855e-06 | 2.240e-04 |
| 3201 | 3.475e-08 | 1.578e-05 |
| 12801 | 7.825e-10 | 1.218e-06 |
| 64001 | 1.041e-11 | 6.338e-08 |

μ_(5/6) converges more slowly than μ_(5/3) because the h^(5/6) weight has an
infinite derivative at h = 0; both are converged to well below the accuracy of
any Cn² model by 12801 nodes, and the adaptive default is better still.

**Conclusion on quadrature:** the default (adaptive, breakpoint-aware) results
quoted throughout this document are converged. The residual quadrature error in
the tabulated values of §3.2 is bounded by ~1e-10 relative for HV5/7 and by the
machine-precision agreement of §4.2 for the SLC models — negligible against the
±50 % or worse uncertainty of the Cn² models themselves.

---

## 5. Test-suite evidence

`python -m pytest tests/ -q` from `products/P020/`: **117 passed, 0 failed,
0 skipped** (run in this session). The suite includes:

* known-answer tests with the hand arithmetic of §1 in the comments;
* Hypothesis property tests of λ^(6/5) (r₀, θ₀), λ^(−6/5) (f_G), λ^(−7/6)
  (σ_R²), the sec(ζ) exponents, and the Cn²-amplitude exponents;
* input validation: negative altitude, non-monotonic and duplicated tabulated
  heights, non-positive Cn², zenith ≥ 90°, negative zenith, non-finite inputs,
  radio wavelengths, integration ranges outside a model's stated validity,
  unknown method/wave/path strings, wind support shorter than the path;
* quadrature-convergence regression tests pinning the §4 numbers;
* piecewise-branch value tests for both SLC models and the analytic band sum.

`ruff check src/ tests/` — clean.

---

## 6. What is NOT validated

Stated plainly, because a Level-2 claim is only as good as its scope:

* **No comparison against measured turbulence data.** Every number here is
  either a closed form, an internal consistency check, or a comparison against
  a value quoted in the literature. No field campaign data was used.
* **No validation of the profile models themselves.** HV, SLC-Day and SLC-Night
  are implemented as published; whether they describe any particular site on
  any particular night is outside this package.
* **No strong-fluctuation regime.** Everything is first-order Rytov theory;
  σ_R² ≥ 1 raises a warning and the returned value is an extrapolation.
* **No inner/outer scale, no aperture averaging, no beam-wave (Gaussian-beam)
  geometry, no Earth curvature.** Each is a documented limitation, not a
  validated approximation.
* **The Greenwood frequency's zenith convention** (sec(ζ)^(3/5), wind taken as
  already transverse) is stated and tested for self-consistency, but the choice
  between it and the alternative sec(ζ)^(8/5) convention is not resolved by any
  evidence here — see the README Limitations.
