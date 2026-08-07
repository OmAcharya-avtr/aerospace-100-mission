# AtmoProfile

**Status:** TESTING · **Class:** compact · **Validation level:** 2 (Research) · **AI:** no

## Executive overview

AtmoProfile turns a refractive-index structure-constant profile Cn²(h) into the
five numbers an optical-propagation or adaptive-optics engineer actually needs:
the Fried coherence length r₀, the isoplanatic angle θ₀, the Greenwood frequency
f_G, the plane- and spherical-wave Rytov variance on a slant path, and the
weak-regime scintillation index σ_I². Every function's docstring states the
weighting integral it evaluates, its units, its assumptions and its validity
range; every zenith dependence is applied as an explicit, stated power of
sec(ζ) rather than folded into a coefficient; every coefficient is either a
value quoted in the standard literature or derived here with the arithmetic
shown.

The package is **deterministic**: no randomness, no fitting, no learned
components. Given a profile and a wavelength the answer is a quadrature, and
the quadrature is demonstrated to be converged. That is deliberate — this is
the reference implementation that later AI products in this portfolio are
benchmarked against, so its value is correctness and citation discipline, not
features.

Package `atmoprofile`, version 0.1.0, pure Python (numpy + scipy; matplotlib
only for the example plots). MIT licence.

## Aerospace problem

Turbulence sizing decisions are made early and are expensive to revisit. How
many actuators does the deformable mirror need (D/r₀)? How fast must the loop
run (f_G)? How far off-axis can the guide star be (θ₀)? How much fade margin
does the downlink need (σ_I²)? Each answer is a weighted integral of Cn² along
the path, and each integral has a *different* weight and therefore a *different*
zenith-angle dependence — r₀ falls as cos(ζ)^(3/5) while θ₀ falls as
cos(ζ)^(8/5) and the Rytov variance rises as sec(ζ)^(11/6). Tools that hide
those exponents inside an "airmass factor" produce link budgets that are right
at zenith and quietly wrong at 60°.

The second recurring failure is citation drift: coefficients (0.423, 2.914,
0.102, 2.25, 6.88, 0.314) get copied between codebases until nobody can say
which definition of r₀ or which bandwidth convention a given number belongs to.
This package writes each coefficient next to the definition it comes from and
derives the relations between them numerically, so the internal consistency is
machine-checked rather than asserted.

## Intended users

Adaptive-optics designers sizing an AO system against a turbulence profile;
free-space optical link engineers computing fade statistics and margins;
atmospheric-optics researchers who need a trustworthy integral kernel; students
verifying textbook results; and other products in this portfolio that need a
validated turbulence baseline to benchmark against.

## Engineering theory

**Conventions.** Altitude `h` in metres above the observer's ground datum;
wavelength λ in metres; wavenumber k = 2π/λ in rad/m; Cn² in m^(−2/3); zenith
angle ζ in radians with 0 ≤ ζ < π/2; wind speed v in m/s. The moment arm
u = h − h₀ is measured from the observer. Turbulence moments are
μ_m = ∫ Cn²(h) u^m dh.

**Design rule.** All weighting integrals are evaluated along the *vertical*.
The slant path enters afterwards as an explicit power of sec(ζ), stated per
quantity. This is why the zenith exponents in the table below can be verified
numerically against the analytic values (they are, to machine precision, in
`validation/`).

| Quantity | Weighting integral | sec(ζ) | λ | Source | Assumptions / validity |
|---|---|---:|---:|---|---|
| Fried parameter r₀ [m], plane wave | r₀ = [0.423 k² sec ζ ∫Cn²(h) dh]^(−3/5) | −3/5 | +6/5 | Fried 1966; Andrews & Phillips 2005; Hardy 1998 | Kolmogorov spectrum, infinite outer scale, negligible inner scale, isotropic homogeneous layers, plane-parallel atmosphere (ζ ≲ 60°) |
| r₀, spherical wave | same with W(u) = (1−u/L)^(5/3) downlink, (u/L)^(5/3) uplink | −3/5 | +6/5 | Fried 1966; Andrews & Phillips 2005 | point source at a finite distance; weight = (distance from source / total)^(5/3) |
| Isoplanatic angle θ₀ [rad] | θ₀ = [2.914 k² sec^(8/3)ζ ∫Cn²(h) u^(5/3) dh]^(−3/5) | −8/5 | +6/5 | Fried 1982; Roddier 1981; Andrews & Phillips 2005 | Fried's 1 rad² definition; plane-wave (downlink) geometry; sec^(5/3) from the moment arm × sec from the path element |
| Greenwood frequency f_G [Hz] | f_G = [0.102 k² sec ζ ∫Cn²(h) v(h)^(5/3) dh]^(3/5) | +3/5 | −6/5 | Greenwood 1977; Hardy 1998 | Taylor frozen flow; Greenwood's first-order-servo bandwidth definition; **v is taken to be already transverse to the line of sight** (see Limitations) |
| Rytov variance σ_R², plane | σ_R² = 2.25 k^(7/6) sec^(11/6)ζ ∫Cn²(h) u^(5/6) dh | +11/6 | −7/6 | Andrews & Phillips 2005 | first-order Rytov, **σ_R² < 1**; point receiver; unbounded wave; no inner scale |
| σ_R², spherical | same with W(u) = u^(5/6)(1−u/L)^(5/6) | +11/6 | −7/6 | Andrews & Phillips 2005 | weight symmetric in u ↔ L−u, so uplink and downlink coincide (reciprocity) |
| Scintillation index σ_I² | σ_I² ≃ σ_R² | +11/6 | −7/6 | Andrews & Phillips 2005 | **weak fluctuations only**; point receiver, no aperture averaging; saturates in the strong regime, which is not modelled |
| Effective height h̄ [m] | h̄ = [μ_(5/3)/μ_0]^(3/5) | — | — | Roddier 1981; Hardy 1998 | single-equivalent-layer interpretation |
| Seeing FWHM [rad] | ε = 0.98 λ/r₀ | +3/5 | −1/5 | Roddier 1981; Hardy 1998 | long exposure, infinite outer scale |

**Coefficients and where they come from** (`src/atmoprofile/constants.py`
carries the arithmetic; the tests assert it):

* `6.883877 = 2·(24/5·Γ(6/5))^(5/6)` — the constant in Fried's definition
  D_φ(r) = 6.88 (r/r₀)^(5/3).
* `0.423` — from `2.914 / 6.883877 = 0.4233080`, equating the Kolmogorov
  plane-wave structure function D_φ(r) = 2.914 k² r^(5/3) μ_0 with that
  definition. The published 3-figure value 0.423 is used; the difference is
  +0.043 % in r₀.
* `0.314` — from `(0.423/2.914)^(3/5) = 0.3141308`, giving θ₀ = 0.314 r₀/h̄.
  The package reproduces this identity to ≤1e-12 relative for every profile.
* `2.31` — from `0.102^(3/5)·(2π)^(6/5) = 2.30662`, showing that the two
  published Greenwood forms are the same expression.
* `2.25` — the slant-path Rytov coefficient; `2.25 × 6/11 = 1.227` recovers the
  textbook horizontal-path 1.23, and `2.25 × B(11/6,11/6) = 0.496` recovers the
  textbook 0.5, with the classic spherical/plane ratio 0.404.

**Profile models** (all with stated validity, all citable):

| Model | Form | Source | Validity |
|---|---|---|---|
| Hufnagel-Valley | Cn² = 0.00594(v/27)²(10⁻⁵h)¹⁰e^(−h/1000) + 2.7e-16 e^(−h/1500) + A e^(−h/100) | Hufnagel 1974 term with the Valley ground term, in the form given by Andrews & Phillips 2005 and Hardy 1998 | climatological clear-air, 0–20 km, horizontally homogeneous; v (5–20 km rms wind) and A (surface strength) are the only site/season handles |
| HV 5/7 | HV with v = 21 m/s, A = 1.7e-14 | as above | named for giving r₀ = 5 cm and θ₀ = 7 µrad at 0.5 µm (reproduced here as 4.962 cm / 7.011 µrad) |
| SLC-Day | 5-band piecewise (see docstring) | SLC/AMOS models tabulated by Beland 1993 and reproduced by Andrews & Phillips 2005 | daytime clear sky at the AMOS site (Maui), 0–20 km; discontinuous at band edges; site-specific |
| SLC-Night | 5-band piecewise (see docstring) | as above | night-time clear sky, AMOS, 0–20 km |
| Bufton wind | v(h) = v_g + 30 exp[−((h−9400)/4800)²] | Bufton 1973, in the form given by Andrews & Phillips 2005 | climatological mid-latitude average, 0–20 km, single tropopause jet, no shear layers |

**Citation policy.** Citations are given at work level (author, title, edition,
year). **No page, equation, figure or table numbers appear anywhere in this
product**, because none were verified against a physical copy during the build.
Numerical constants are either universally reproduced named results or derived
here with the arithmetic shown. Nothing was invented.

## Architecture

```
src/atmoprofile/
├── __init__.py     public API re-exports, __version__
├── constants.py    coefficients with their derivations and the exponent tables
├── _validate.py    all input validation (wavelength band, zenith range, altitudes, samples)
├── profiles.py     Cn2Profile + HV / HV5-7 / SLC-Day / SLC-Night / constant / tabulated
├── wind.py         WindProfile + Bufton / constant / tabulated, rms_upper_wind
├── integrals.py    quadrature (adaptive + Simpson), moments, convergence helper
├── metrics.py      r0, theta0, f_G, Rytov, scintillation index, seeing, summarize()
└── __main__.py     CLI: python -m atmoprofile summary|profile
```

Data flow: a `Cn2Profile` (callable + provenance + validity + breakpoints) and
optionally a `WindProfile` go into a weighted integral in `integrals.py`; each
metric in `metrics.py` supplies its own weight function, then applies its own
sec(ζ) power. The vertical integral and the airmass factor never mix.

Quadrature: the default `method="quad"` integrates panel by panel between the
profile's declared breakpoints with adaptive Gauss-Kronrod (QUADPACK), after
rescaling the integrand to O(1) — necessary because Cn² integrals are ~1e-12 in
SI units, far below QUADPACK's default absolute tolerance.
`method="simpson", n_nodes=N` gives a fixed composite-Simpson rule for grid
refinement studies.

## Installation

```bash
cd products/P020
python -m pip install -e .            # numpy, scipy
python -m pip install -e ".[dev]"     # + pytest, hypothesis, ruff, matplotlib
```

Or simply put `src/` on `PYTHONPATH`; the package has no build-time steps.

## Quick start

```python
import math
from atmoprofile import (fried_parameter, isoplanatic_angle, greenwood_frequency,
                         rytov_variance, scintillation_index, hv57, bufton_wind)

profile = hv57()                 # Hufnagel-Valley 5/7, 0-20 km
wind = bufton_wind(5.0)          # Bufton, 5 m/s ground wind
lam = 500e-9                     # metres
zen = math.radians(45.0)         # radians, 0 <= zen < pi/2

fried_parameter(profile, lam, zenith_rad=zen)        # 0.0403076 m
isoplanatic_angle(profile, lam, zenith_rad=zen)      # 4.0272e-06 rad
greenwood_frequency(profile, wind, lam, zenith_rad=zen)   # 88.483 Hz
rytov_variance(profile, lam, zenith_rad=zen)              # 0.439125
scintillation_index(profile, lam, zenith_rad=zen)         # 0.439125 (weak regime)
```

CLI:

```
$ python -m atmoprofile summary --profile hv57 --wavelength-nm 500 --zenith-deg 0 30 45 60
AtmoProfile 0.1.0 - HV5/7 at 500 nm
  Cn^2 source : Hufnagel (1974) upper-atmosphere term with the Valley ground term, ...
  wind source : Bufton, Applied Optics 12(8), 1973; ...
  path        : 0 m to 20000 m
zen[deg]    r0[cm]  r0sph[cm]  th0[urad]    fG[Hz]   sig2R,pl   sig2R,sp  seeing["]  weak?
------------------------------------------------------------------------------------------
     0.0     4.962      5.180      7.011     71.87     0.2326     0.1571      2.037   True
    30.0     4.552      4.751      5.570     78.35     0.3028     0.2046      2.220   True
    45.0     4.031      4.207      4.027     88.48     0.4391     0.2966      2.507   True
    60.0     3.274      3.417      2.313    108.94     0.8290     0.5600      3.087   True
```

`--json` emits the same rows as JSON. `python -m atmoprofile profile --profile
slc_day` prints Cn² samples with the model's provenance and validity statement.

Using your own measured profile:

```python
from atmoprofile import tabulated_profile, tabulated_wind, fried_parameter
profile = tabulated_profile(heights_m, cn2_values, name="site A 2026-08-07",
                            reference="SCIDAR campaign")
wind = tabulated_wind(heights_m, speeds_ms)
fried_parameter(profile, 1550e-9)
```

Heights must be strictly increasing and non-negative and Cn² strictly positive
(interpolation is log-linear); violations raise `ValueError` with the offending
index.

## Configuration

| Argument | Meaning | Default |
|---|---|---|
| `wavelength_m` | optical wavelength, m; must lie in 100 nm – 20 µm | — |
| `zenith_rad` | zenith angle, rad; `[0, π/2)`; warns above 60° | 0.0 |
| `h_ground`, `h_top` | path endpoints, m; must lie inside the profile's validity range | 0.0, profile top |
| `wave` | `"plane"` or `"spherical"` | `"plane"` |
| `path` | `"downlink"` or `"uplink"` (spherical wave only) | `"downlink"` |
| `method` | `"quad"` (adaptive, breakpoint-aware) or `"simpson"` | `"quad"` |
| `n_nodes` | Simpson node count (odd; ignored by `"quad"`) | 2001 |
| `warn_strong` | emit the σ_R² ≥ 1 validity warning | `True` |

## Examples

Both scripts use the Agg backend and were actually run to produce the PNGs in
`screenshots/`.

| Script | Output | Runtime | Content |
|---|---|---:|---|
| `examples/r0_theta0_vs_zenith.py` | `screenshots/r0_theta0_vs_zenith.png` | 7 s | r₀ and θ₀ vs zenith 0–70° for HV5/7, SLC-Day, SLC-Night at 500 nm and 1550 nm, with the analytic sec(ζ) laws drawn underneath as a grey halo and the >60° flat-Earth extrapolation region shaded |
| `examples/cn2_profile_comparison.py` | `screenshots/cn2_profile_comparison.png` | 1 s | the three Cn² models; the three weighting integrands (Cn², Cn²h^(5/6), Cn²h^(5/3)) plotted as contribution per unit ln h; cumulative contribution with h̄ marked — the picture of why r₀ is a ground-layer quantity and θ₀ and scintillation are high-altitude quantities |

## Validation

Level 2 (Research). Full evidence, including the raw script output, is in
`validation/VALIDATION.md` and the four `.txt` files beside it. Headlines:

1. **Closed forms, hand-computed.** For a 1e-15 m^(−2/3) slab 0–1000 m at
   500 nm: r₀ = 0.0803792 m, θ₀ = 4.548159e-5 rad, σ_R²(plane) = 0.0743623,
   σ_R²(spherical) = 0.0300658 — code agrees with the hand arithmetic to
   ≤ 1e-15 relative in every case. The implied textbook horizontal-path
   coefficients come out at 1.227 (published 1.23), 0.496 (published 0.50) and
   a spherical/plane ratio of 0.404 (published 0.4).
2. **Zenith exponents verified numerically.** For all three profiles and all six
   quantities, a blind least-squares fit of ln Q against ln sec ζ recovers the
   analytic exponent (−3/5, −8/5, +3/5, +11/6) to 1e-9, and the direct ratio
   check agrees to 2.2e-16.
3. **HV 5/7 reproduces the values it is named for**: r₀ = 4.9624 cm (vs 5 cm,
   0.75 %) and θ₀ = 7.0109 µrad (vs 7 µrad, 0.16 %) at 0.5 µm, zenith.
4. **Standard models at 500 / 1550 nm**: r₀ = 4.962 / 19.290 cm (HV5/7),
   4.339 / 16.867 cm (SLC-Day), 7.602 / 29.551 cm (SLC-Night). Compared with
   the r₀ ≈ 5–20 cm band quoted for ground sites at 0.5 µm (Hardy 1998;
   Roddier 1981; Andrews & Phillips 2005): SLC-Night is inside, HV5/7 sits on
   the floor by construction, SLC-Day is 13 % below it — expected for a daytime
   sea-level model, and reported rather than tuned.
5. **Quadrature converged.** Simpson refinement 201 → 64001 nodes changes r₀ by
   1.5e-10 relative on HV5/7; on the discontinuous SLC models the adaptive rule
   reproduces the hand-assembled analytic band sum to 2.9e-16 while the finest
   Simpson grid is only good to 5.2e-05.
6. **One reported failure.** The Bufton wind rms over 5–20 km evaluates to
   22.96 m/s for a 5 m/s ground wind, not the 21 m/s pseudowind of HV 5/7 —
   a 9.35 % discrepancy between two published models. Nothing was tuned; see
   Limitations.

## Benchmark results

There is nothing to benchmark against in the accuracy sense — the reference is
analysis, and §1 and §4 of `VALIDATION.md` are that comparison. The performance
figures, measured in this build on 2 CPU cores:

| Operation | Time |
|---|---:|
| one metric, analytic profile, adaptive quadrature | ~16 ms |
| full `summarize()` (7 integrals) | ~170 ms |
| one metric, 800-knot tabulated profile | ~0.6 s |
| full test suite (117 tests, incl. Hypothesis) | ~23 s |
| all four validation scripts | 24.5 s |

The cost is dominated by Python-level integrand evaluation, not by the
quadrature; `method="simpson"` with a modest node count is faster when many
evaluations are needed and the profile is smooth.

## AI model details

Not applicable — this product contains no AI/ML component. It is the
deterministic reference implementation against which AI products elsewhere in
this portfolio are benchmarked.

## Hardware requirements

Any machine that runs Python 3.11 with numpy and scipy. No GPU, no threads, no
network. Memory footprint is a few MB. Every operation in this package
completes in under a second on 2 CPU cores; the full validation suite runs in
25 s.

## Limitations

1. **Kolmogorov spectrum only.** Infinite outer scale, zero inner scale. A
   finite outer scale (10–100 m in practice) increases the effective coherence
   length and reduces low-order wavefront variance; no von Karman correction is
   applied, so r₀ here is the Kolmogorov value.
2. **Weak fluctuations only.** σ_R² ≥ 1 raises a `UserWarning` and the value
   returned is an extrapolation. Saturation of scintillation is not modelled.
   For HV5/7 at 500 nm this limit is reached at ζ = 63.2°.
3. **Plane-parallel atmosphere.** The airmass is sec(ζ) with no Earth curvature
   or refraction; above 60° a `UserWarning` is issued and results are
   extrapolation.
4. **Greenwood frequency zenith convention.** The wind profile supplied is
   assumed to be *already transverse to the line of sight*, so the only zenith
   factor is the path element and f_G ∝ sec(ζ)^(3/5) (Greenwood's published
   form). Some treatments instead model the apparent layer-crossing speed as
   growing like sec(ζ), which adds sec^(5/3) inside the bracket and gives
   sec(ζ)^(8/5) — larger by exactly sec(ζ), i.e. 41.4 % at ζ = 45°. This package implements the first
   convention and states it rather than choosing silently; the alternative can
   be obtained by supplying a scaled wind profile.
5. **Bufton/HV wind inconsistency, unresolved.** The Bufton model with a 5 m/s
   ground wind gives an rms of 22.96 m/s over 5–20 km, while HV 5/7 is quoted
   with a 21 m/s pseudowind. Both are implemented as published; the convention
   behind the 21 m/s (band limits, ground-term treatment, or an added
   slew-rate term) could not be established during this build, so the
   discrepancy is reported as a failed check rather than tuned away.
6. **Point receiver.** No aperture averaging of the scintillation index; a real
   receiver larger than the Fresnel scale sees substantially less.
7. **No Gaussian-beam (beam-wave) geometry.** Only unbounded plane and
   spherical waves; beam wander, beam spreading and pointing jitter are out of
   scope.
8. **Profile models are climatologies, not measurements.** HV, SLC-Day and
   SLC-Night are implemented as published and are not validated against field
   data here. Their own uncertainty (readily a factor of two in Cn²) dominates
   every number this package produces — a factor of 2 in Cn² is a factor of
   1.52 in r₀ and a factor of 2 in σ_R².
9. **No time dependence beyond f_G**, no temporal power spectra, no
   Zernike/modal decomposition, no anisoplanatism budgeting beyond θ₀.
10. **Altitudes are above the observer's own datum.** Site elevation must be
    handled by the caller through `h_ground`; the models' ground terms are not
    re-fitted for a high-altitude site.

## Safety statement

This software is research-grade. It is not flight-qualified, not certified, and
not approved for operational aerospace use.

## Roadmap

* von Karman (finite outer scale) corrections to r₀ and to the low-order
  variances, with the correction factor cited and validated.
* Aperture-averaging factor for the scintillation index.
* Strong-fluctuation scintillation model, so the σ_R² > 1 regime returns a
  physical number instead of a warning.
* Gaussian-beam geometry (beam wander and spreading) for uplink budgets.
* Greenwood-frequency slant-geometry option with the alternative sec(ζ)^(8/5)
  convention selectable and documented.
* Additional profile models (Mod-HV, CLEAR-1, ITU-R style) with their own
  validity statements.

## License

MIT — see `LICENSE`. Copyright © 2026 OPTIMA Organisation.

## Credits

This is under reserved rights obtained by OPTIMA Organisation.

## Citation

```
OPTIMA Organisation (2026). AtmoProfile 0.1.0: deterministic atmospheric
turbulence integrals (Fried parameter, isoplanatic angle, Greenwood frequency,
slant-path Rytov variance, weak-regime scintillation index). Product P020 of
the OPTIMA 100-product aerospace software portfolio.
```

Underlying physics (cite the sources, not this package, for the theory):

* D. L. Fried, "Optical Resolution Through a Randomly Inhomogeneous Medium for
  Very Long and Very Short Exposures", J. Opt. Soc. Am. 56(10), 1372–1379, 1966.
* D. L. Fried, "Anisoplanatism in adaptive optics", J. Opt. Soc. Am. 72(1),
  52–61, 1982.
* D. P. Greenwood, "Bandwidth specification for adaptive optics systems",
  J. Opt. Soc. Am. 67(3), 390–393, 1977.
* J. L. Bufton, "Comparison of vertical profile turbulence structure with
  stellar observations", Applied Optics 12(8), 1785–1793, 1973.
* R. E. Hufnagel, "Variations of atmospheric turbulence", Digest of Topical
  Meeting on Optical Propagation through Turbulence, OSA, 1974.
* R. R. Beland, "Propagation through Atmospheric Optical Turbulence", in The
  Infrared and Electro-Optical Systems Handbook, Vol. 2, SPIE/ERIM, 1993.
* F. Roddier, "The Effects of Atmospheric Turbulence in Optical Astronomy",
  Progress in Optics XIX, 281–376, 1981.
* J. W. Hardy, "Adaptive Optics for Astronomical Telescopes", Oxford University
  Press, 1998.
* L. C. Andrews and R. L. Phillips, "Laser Beam Propagation through Random
  Media", 2nd ed., SPIE Press, 2005.
