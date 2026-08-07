# CnCast

**Status:** TESTING · **Class:** compact · **Validation level:** 2 (Research) · **AI:** yes

## Executive overview

CnCast predicts the vertical profile of the optical turbulence strength Cn²(h)
from the surface to 20 km, and turns that profile into the numbers that actually
size hardware: the Fried parameter r₀, the isoplanatic angle θ₀ and the
Greenwood frequency f_G.

It ships two things, in this order:

1. **The published baselines**, implemented first and validated against their
   own documented behaviour: the Hufnagel-Valley family including the HV 5/7
   parameterisation, and the SLC-Day / SLC-Night piecewise profiles, each with
   its stated altitude validity range. These are **climatological** models —
   averages for a site or season, not forecasts for tonight.
2. **A learned model** mapping surface meteorology (temperature, wind speed,
   relative humidity, time of day, season) plus altitude to a profile shape,
   with a calibrated prediction interval on every point.

The learned model is trained entirely on **synthetic** data generated from those
baselines plus physically-motivated perturbations. Read `DATASET_CARD.md` before
using any accuracy figure: measured accuracy is fidelity to a generative
process, not agreement with radiosonde or scintillometer measurements.

## Aerospace problem

Turbulence-limited optical systems — free-space optical links, adaptive optics,
laser designation, ground-to-satellite communications — are sized against a Cn²
profile. In practice designers pick HV 5/7 out of a textbook and use it for
every case, because it is the only profile available without a measurement
campaign. HV 5/7 is a single fixed curve: it gives the same answer at noon in
July and at midnight in January, and the same answer for a coastal site and a
desert plateau.

The gap this product addresses is the middle ground between "one textbook curve"
and "instrument the site": conditioning the profile shape on the surface
meteorology a site already records, and being explicit about how uncertain the
result is.

## Intended users

* Optical-link and adaptive-optics engineers doing early sizing, who need r₀,
  θ₀ and f_G with an uncertainty rather than a single number.
* Researchers who need a clean, cited implementation of HV / HV 5/7 / SLC and
  the standard turbulence integrals, with the conventions written down.
* Anyone building a Cn² estimation pipeline who needs an honest baseline to beat
  and a worked example of how to benchmark and calibrate against one.

## Engineering theory

All Cn² in m^-2/3, altitude in metres, wavelength in metres.

### Hufnagel-Valley profile

```
Cn²(h) = 0.00594 (v/27)² (10⁻⁵ h)¹⁰ e^(−h/1000) + 2.7e−16 e^(−h/1500) + A e^(−h/100)
```

* Source: Hufnagel (1974); Valley (1980) two-parameter extension; form as
  Andrews & Phillips (2005) *Laser Beam Propagation through Random Media*,
  2nd ed., Eq. (12.30).
* `v` = pseudowind, the rms wind over 5–20 km (m/s); `A` = ground-level Cn².
* Validity: 0–20 000 m above ground level; mid-latitude, continental, clear-air
  climatology; horizontally homogeneous.
* **HV 5/7** sets v = 21 m/s, A = 1.7e-14 m^-2/3 — named for producing
  r₀ ≈ 5 cm and θ₀ ≈ 7 µrad at λ = 0.5 µm on a vertical path. This
  implementation gives 4.9624 cm and 7.0109 µrad (`validation/VALIDATION.md`
  §1.2).

### SLC-Day and SLC-Night

Piecewise fits at the AMOS observatory, Mt Haleakala (site ≈ 3.05 km MSL), so
h = 0 is the observatory floor, not sea level. Source: Beland (1993), *The
Infrared and Electro-Optical Systems Handbook* Vol. 2, ch. 2; tabulated in
Andrews & Phillips (2005) §12.2.1.

| band (m) | SLC-Day | band (m) | SLC-Night |
|---|---|---|---|
| 0–18.5 | 1.70e-14 | 0–18.5 | 8.40e-15 |
| 18.5–240 | 3.13e-13/h^1.05 | 18.5–110 | 2.87e-12/h² |
| 240–880 | 1.30e-15 | 110–1500 | 2.50e-16 |
| 880–7220 | 8.87e-07/h³ | 1500–7200 | 8.87e-07/h³ |
| 7220–20500 | 2.00e-16/√h | 7200–20000 | 2.00e-16/√h |
| > 20500 | 0 | > 20000 | 0 |

Validity: 0–20.5 km / 0–20 km above the site; identically zero above.
**The published SLC-Day fit is discontinuous** — +31 % at 240 m, −14 % at
18.5 m. That is a property of the published model, quantified in
`validation/VALIDATION.md` §1.3.

### Bufton wind

```
V(h) = w_ground + 30 exp[−((h − 9400)/4800)²]        [m/s]
v = [ (1/15e3) ∫₅ₖₘ²⁰ᵏᵐ V²(h) dh ]^(1/2)
```

Source: Bufton (1973); Andrews & Phillips (2005) Eqs. (12.31)–(12.32). Used only
for the Greenwood frequency. A climatological jet stream: real cores move by
kilometres and exceed 60 m/s.

### Integrated seeing quantities

With k = 2π/λ and ζ the zenith angle (plane-parallel sec law, good to ~1 % below
60°, degrading above 70°):

```
r₀ = [0.423 k² sec ζ ∫ Cn²(h) dh]^(−3/5)                       Fried (1966)
θ₀ = [2.914 k² sec^(8/3) ζ ∫ Cn²(h) h^(5/3) dh]^(−3/5)         Fried (1982)
f_G = 2.31 λ^(−6/5) [sec ζ ∫ Cn²(h) V^(5/3)(h) dh]^(3/5)       Greenwood (1977)
FWHM = 0.98 λ / r₀                                             Roddier (1981)
```

Assumptions: Kolmogorov spectrum, inner scale ≪ r₀ ≪ outer scale,
weak-fluctuation (Rytov) regime, plane-wave geometry, frozen flow for f_G.
Scalings (all verified to machine precision in the test suite): r₀ ∝ λ^(6/5),
r₀ ∝ cos ζ^(3/5), θ₀ ∝ cos ζ^(8/5).

### Learned model

Three quantile gradient-boosting regressors (α = 0.05 / 0.50 / 0.95) on
`[T, wind, RH, sin/cos hour, sin/cos day-of-year, log10 h]` → `log10 Cn²`,
plus a split-conformal offset (Romano, Patterson & Candès 2019) fitted on a
disjoint calibration set. See `MODEL_CARD.md`.

## Architecture

```
src/cncast/
├── baselines.py   HV, HV 5/7, SLC-Day, SLC-Night, Bufton wind   (no ML, no data)
├── seeing.py      turbulence moments, r₀, θ₀, f_G, seeing FWHM  (no ML, no data)
├── dataset.py     seeded synthetic scenario + profile generator
├── model.py       CnCastModel + Hv57Baseline / SlcBaseline / ClimatologyBaseline
└── __main__.py    CLI: `python -m cncast baseline|predict`
```

`baselines.py` and `seeing.py` depend only on NumPy and are usable on their own;
nothing in them imports the ML layer. No cross-product imports.

## Installation

```bash
cd products/P019
pip install -e .
# or, without installing:
export PYTHONPATH=src
```

Requires Python ≥ 3.11 with numpy, scikit-learn and matplotlib.

## Quick start

```python
import numpy as np
from cncast import hv57, fried_parameter, isoplanatic_angle, train_default_model

# 1. Published baseline
h = np.linspace(0.0, 20_000.0, 20_001)
print(fried_parameter(h, hv57(h), 500e-9) * 100)      # 4.962 cm
print(isoplanatic_angle(h, hv57(h), 500e-9) * 1e6)    # 7.011 urad

# 2. Learned model with an interval (~16 s to fit, seeded)
model, _ = train_default_model()
pred = model.predict(
    surface_temp_c=22.0, surface_wind_m_s=4.0, relative_humidity_pct=45.0,
    hour_of_day=14.0, day_of_year=200, altitude_m=np.geomspace(5.0, 20_000.0, 24),
)
print(pred.cn2_lower[0], pred.cn2[0], pred.cn2_upper[0])
print(pred.coverage, pred.extrapolating)
```

CLI:

```bash
python -m cncast baseline --model slc_night --wavelength-nm 1550 --zenith-deg 30
python -m cncast predict --temp-c 22 --wind-m-s 4 --rh-pct 45 --hour 14 --day-of-year 200
```

## Configuration

| knob | where | default | note |
|---|---|---|---|
| nominal interval coverage | `CnCastModel(coverage=...)` | 0.90 | only 0.90 has been calibrated and measured |
| model size | `CnCastModel(n_estimators, max_depth, ...)` | 300, 4 | fit budget 120 s; default uses 16 s |
| dataset size | `train_default_model(n_scenarios, n_altitudes)` | 700, 28 | |
| seeds | `train_default_model(data_seed, altitude_seed, split_seed, random_state)` | 20260807, 99, 4242, 7 | |
| conformal calibration | `train_default_model(calibrate=False)` | on | off → intervals under-cover (0.80 vs 0.90) |
| integration grid | caller's `h_m` array | — | ≤ 10 m near the surface, or log-spaced; see `VALIDATION.md` §1.2 |

## Examples

| script | output | shows |
|---|---|---|
| `examples/profile_with_intervals.py` | `screenshots/profile_with_intervals.png` | two predicted profiles with their 90 % bands against synthetic truth, HV 5/7, SLC-Day and SLC-Night |
| `examples/r0_comparison.py` | `screenshots/r0_comparison.png` | r₀ derived from predicted profiles vs truth for 175 held-out scenarios, with the r₀ band, against the single fixed HV 5/7 value; plus the diurnal cycle in 3 h medians |

Both were run to produce the committed PNGs (Agg backend, no `show()`).

## Validation

Level 2 (Research). Full evidence in `validation/VALIDATION.md`; raw script
output in `validation/validate_baselines_output.txt` and
`validation/benchmark_results.md`. Highlights:

* **HV 5/7 reproduces its nickname:** r₀ = 4.9624 cm (−0.75 % from the nominal
  5 cm), θ₀ = 7.0109 µrad (+0.16 % from 7 µrad) at 500 nm, zenith.
* **Constant-Cn² slab closed form:** r₀ = 7.848343 cm from the analytic formula
  vs 7.848343 cm from the code (relative error 0.00e+00).
* **SLC branches:** all six spot checks match their published closed forms to
  0.0e+00 relative error; the published SLC-Day fit's +31.1 % discontinuity at
  240 m is reported as a property of the model.
* **Hand check with arithmetic shown** (`VALIDATION.md` §5.1): r₀ from a
  predicted profile on a 5-point grid, trapezoid μ₀ = 1.4500718e-12 m^(1/3) and
  r₀ = 6.431470 cm by hand, matching the code to all 7 printed digits.
* **Interval coverage:** 0.8988 empirical against 0.900 nominal after conformal
  calibration (0.8033 without it), within ±0.02 of nominal in all five altitude
  bands.
* **Grid sensitivity, reported not hidden:** a 100 m integration grid biases r₀
  by +3.6 %.

Test suite: `python -m pytest tests/ -q` → **112 passed**, 0 failed, 0 skipped.
`ruff check src/ tests/` → clean.

## Benchmark results

Learned model vs the baselines on 4 900 held-out rows from 175 unseen scenarios
(scenario-level split), errors in dex:

| predictor | RMSE | MAE | bias | p95 abs err |
|---|---:|---:|---:|---:|
| **CnCast learned model** | **0.2095** | 0.1620 | −0.0201 | 0.4190 |
| HV 5/7 (mandated baseline) | 0.5665 | 0.4475 | +0.2686 | 1.1048 |
| SLC day/night | 0.7314 | 0.5199 | −0.0893 | 1.2224 |
| Training climatology | 0.3102 | 0.2395 | +0.0049 | 0.6168 |

**The learned model beats HV 5/7 by 63 %, and that result is close to
tautological.** The synthetic targets are generated from the H-V family with
parameters driven by the very surface variables the model is given, so any
method that recovers the generator must beat a fixed curve with no
meteorological inputs. HV 5/7 is not a strawman — it is a climatology being
scored on a task it was never designed for. The informative comparison is the
training climatology (0.3102 dex), which the model improves on by 32 %; in the
300–2 000 m band the two are indistinguishable (0.2089 vs 0.2105), and that is
reported rather than omitted. A baseline win would have been an acceptable
outcome.

## AI model details

Full card: **`MODEL_CARD.md`** (fifteen items). Dataset: **`DATASET_CARD.md`**.

* **Baseline first:** `cncast.baselines` was implemented and validated before any
  model was fitted; HV 5/7 is the mandated benchmark, with SLC and a training
  climatology as additional comparators.
* **Dataset:** 100 % synthetic, 700 scenarios × 28 altitudes, master seed
  20260807, generated by the committed `cncast/dataset.py`. **Accuracy is
  measured against a generative process, not against radiosonde or
  scintillometer measurements.** Not modelled: real boundary-layer dynamics,
  terrain, inversion layers, jet-stream variability.
* **Training:** three quantile GradientBoostingRegressors, 15.9 s on 2 CPU cores.
* **Test split:** grouped by scenario — 394 fit / 131 conformal calibration /
  175 test — because rows from one profile share every meteorological feature.
* **Uncertainty output:** every prediction carries `cn2_lower`/`cn2_upper` at a
  nominal 90 % coverage plus an `extrapolating` flag; empirical coverage 0.8988.
* **Failure cases:** out-of-domain confident extrapolation, staircase profiles,
  no thin layers, +0.67 cm optimistic bias in derived r₀, no forecast horizon,
  single-site assumption. See `MODEL_CARD.md` §10.
* **Reproducibility:** exact commands and all seeds in `MODEL_CARD.md` §11;
  re-running gives a max prediction difference of 0.000e+00 dex.

**This model is not certified for operational flight use.**

## Hardware requirements

2 CPU cores, ~500 MB RAM. No GPU, no PyTorch. Full fit 15.9 s, test suite 17.5 s,
full benchmark script 51 s. Nothing here needs more than a laptop.

## Limitations

1. **Synthetic training data** — the dominant limitation; see `DATASET_CARD.md`.
2. **Not a forecast.** Inputs and target are simultaneous; there is no time
   dimension and no lead time anywhere in this product.
3. **Climatological baselines.** HV, HV 5/7 and SLC describe averages. Real Cn²
   at a fixed altitude routinely departs from them by an order of magnitude.
4. **Site-agnostic.** No latitude, elevation, terrain or site descriptor is an
   input. SLC is an AMOS/Haleakala fit being offered as a generic day/night pair.
5. **Plane-parallel geometry only.** sec ζ, no Earth curvature; unreliable above
   ~70° zenith. Plane-wave r₀ only — no spherical-wave (beacon) constant.
6. **Integration accuracy is the caller's responsibility.** A coarse grid biases
   r₀ (+3.6 % at 100 m spacing); quantified in `VALIDATION.md` §1.2.
7. **One calibrated coverage.** Only 90 % has been calibrated and measured.
8. **Derived-quantity intervals over-cover** (98.9 % for r₀ against 90 %
   nominal) because the profile bounds are perfectly correlated in altitude.
9. **No Cn² inversion from measurements** — that is P013's scope, not this one.
10. **Deviation from the build guide:** none. The product is self-contained, uses
    only the permitted libraries, and commits no data files.

## Safety statement

This software is research-grade. It is not flight-qualified, not certified, and
not approved for operational aerospace use.

## Roadmap

* Replace the synthetic generator with a real radiosonde/thermosonde-derived
  dataset and re-validate; every metric in this repository would have to be
  recomputed.
* Add a genuine forecast mode (lead time as a feature) once time-correlated data
  exists.
* Site descriptors (elevation, latitude, coastal/continental) as features.
* Spherical-wave r₀ and slant-path geometry beyond the sec ζ approximation.
* Conditional (not just marginal) interval calibration by altitude band.

## License

Apache-2.0. See `LICENSE`. Copyright © 2026 OPTIMA Organisation.

## Credits

This is under reserved rights obtained by OPTIMA Organisation.

## Citation

```bibtex
@software{cncast2026,
  title  = {CnCast: vertical Cn2 profile prediction with published baselines and
            calibrated prediction intervals},
  author = {{OPTIMA Organisation}},
  year   = {2026},
  version = {0.1.0},
  note   = {Research-grade; trained on synthetic data; not certified for
            operational flight use}
}
```

Key references implemented here: Hufnagel (1974); Valley (1980), *Appl. Opt.*
19(4), 574–577; Beland (1993), *IR & EO Systems Handbook* Vol. 2 ch. 2;
Bufton (1973), *Appl. Opt.* 12(8), 1785–1793; Fried (1966), *JOSA* 56(10),
1372–1379; Fried (1982), *JOSA* 72(1), 52–61; Greenwood (1977), *JOSA* 67(3),
390–393; Roddier (1981), *Prog. Opt.* 19, 281–376; Andrews & Phillips (2005),
*Laser Beam Propagation through Random Media*, 2nd ed., SPIE Press, ch. 12;
Romano, Patterson & Candès (2019), *Conformalized Quantile Regression*,
NeurIPS 32.
