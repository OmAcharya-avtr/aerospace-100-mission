# ShackSim

**Status:** TESTING · **Class:** compact · **Validation level:** 2 (Research) · **AI:** yes

> **Headline honest result — read this before using the learned estimator.** On
> held-out synthetic data the ML ensemble beats the tuned thresholded centre of
> gravity **only below ≈ 100 photoelectrons per subaperture** (crossover between
> 50 and 100 e⁻ for round spots, between 100 and 300 e⁻ for 3× elongated spots).
> The best measured advantage is **1.38×**. Above the crossover the analytic
> estimator wins and the gap widens without limit — **10.5× at 10 000 e⁻**.
> Against the *correlation* baseline the learned model loses at 6 of 7 flux
> levels for round spots. Numbers and derivations in
> [`validation/VALIDATION.md`](validation/VALIDATION.md).

## Executive overview

ShackSim simulates a Shack-Hartmann wavefront sensor end to end and extracts the
slope vector from the simulated frame. It ships four things that belong
together:

1. **Lenslet-array geometry** — configurable pitch, focal length, subaperture
   pixel count, wavelength, circular pupil mask with optional central
   obscuration, and the derived diffraction spot size and field of view;
2. **Per-subaperture spot formation** — pixel-integrated diffraction-limited
   spot (optionally elongated), displaced by the local wavefront gradient, with
   a documented photon / background / read-noise chain;
3. **Two classical slope estimators, implemented first** — thresholded centre of
   gravity and correlation with parabolic peak interpolation — together with the
   standard analytic noise-propagation expression for the CoG;
4. **A learned slope estimator** — a 5-member scikit-learn MLP ensemble
   targeting the low-flux and elongated-spot regimes, with a per-slope
   ensemble-spread confidence output, benchmarked against the classical
   baselines on identical held-out data.

Everything is numpy / scipy / scikit-learn. No GPU, no network access, no
committed data.

## Aerospace problem

The Shack-Hartmann sensor is the workhorse wavefront sensor of free-space
optical communication terminals, laser-guide-star adaptive optics, active
telescope alignment and optical metrology. It measures the wavefront gradient
by asking, for each of dozens to thousands of lenslets, "how far has this spot
moved?" — and the answer feeds a reconstructor whose output drives a deformable
mirror or a fine-steering mirror.

Three things dominate the error budget and none of them are obvious from a
textbook diagram:

- **Photon starvation.** The read-noise term of the centroid variance grows with
  the *window area*, so halving the flux more than doubles the slope error once
  read noise dominates. At 30–100 e⁻ per subaperture — an entirely realistic
  regime for a faint beacon at high frame rate — the centre of gravity is
  barely usable.
- **Background offset.** An unsubtracted background does not add noise to the
  centroid; it *shrinks every slope toward zero by a fixed factor* `S/(S+Bp²)`.
  That is a gain error on the whole reconstructed wavefront, and in a closed
  loop it is a loop-gain error. This product measures it: at 1000 e⁻ signal and
  2 e⁻/px background, a third of the measured wavefront amplitude disappears.
- **Spot shape.** Elongated spots (laser guide stars, aberrated lenslets,
  extended sources) bias a thresholded first moment along the elongation axis.

ShackSim exists to let an engineer quantify all three against an exactly known
truth before committing to hardware or to an estimator.

## Intended users

Adaptive-optics, free-space-optical and payload engineers sizing wavefront
sensor error budgets; researchers comparing classical and learned slope
estimators on a common substrate; students learning why a Shack-Hartmann sensor
degrades the way it does. Not for flight software (see
[Safety statement](#safety-statement)).

## Related work — P008 CentroidNet

**P008 CentroidNet** covers subpixel centroiding of a *single* optical spot on
one detector window: spot generator, centre-of-gravity and quad-cell baselines,
and an ML centroider. **Its content is not duplicated or re-derived here.** If
you want single-spot centroid error versus SNR, the quad-cell linear range, or
the photon-noise limit on a single centroid, read P008.

What is different in P018, and what this product is actually about:

| | P008 CentroidNet | **P018 ShackSim** |
|---|---|---|
| Unit of work | one detector window | a lenslet **array** with a pupil mask |
| Geometry | none — pixels only | pitch, focal length, f-number, pupil, obscuration, fill fraction |
| Physics added | Gaussian PSF | diffraction spot size `1.029 λf/d`, gradient → displacement `Δx = f g/p` |
| Noise question | per-window noise | noise **per subaperture across a whole frame**, and its effect on the slope *field* |
| Output | one centroid (px) | a **slope vector** `(g_x, g_y)` per subaperture, in radians |
| Estimators | CoG, quad-cell | thresholded CoG, **correlation**, learned ensemble |
| Validation | noise-free CoG, quad-cell bias curve | **known-tilt uniformity across the pupil**, noise-propagation expression, background shrinkage |

The two products share no code and no imports.

## Engineering theory

Conventions: pupil coordinates in metres from the pupil centre; detector
coordinates in **pixels** from the geometric centre of a subaperture's pixel
block, `(p−1)/2`; `+x` = increasing column, `+y` = increasing row; images are
`image[row, col] == image[y, x]`. A "slope" is the wavefront (OPD) gradient in
radians (numerically the same as m/m in the small-angle limit).

**Diffraction-limited spot size** (Born & Wolf 1999, *Principles of Optics*,
7th ed., Cambridge Univ. Press, sec. 8.5.2):

```
FWHM = 1.0287938 · λ f / d        [m]
```

Units: `λ` wavelength [m], `f` lenslet focal length [m], `d` lenslet clear
aperture = pitch [m]. The coefficient is `2 × 1.61633 / π`, from the half-power
point of the Airy intensity `(2 J₁(v)/v)²`. **The scaling is what matters:
spot size is linear in wavelength and in f-number `f/d`.** *Assumptions:*
circular, unobscured, uniformly illuminated, unaberrated subaperture; paraxial.
*Validity:* the core only — the Gaussian approximation used here
(`σ = FWHM/2.35482`) has no rings and decays much faster than the true `r⁻³`
Airy envelope.

**Slope → spot displacement** (Hardy 1998, *Adaptive Optics for Astronomical
Telescopes*, Oxford Univ. Press, ch. 5). Derived explicitly in the docstring of
`LensletArray.slope_to_displacement`: for a wavefront `W(x,y)` with mean
gradient `g_x = ⟨∂W/∂x⟩` over a subaperture, the wavefront normal is tilted by
`θ_x = g_x` to first order, and a perfect lens maps ray angle to focal-plane
position:

```
Δx = f · tan θ_x ≈ f · g_x   [m]        Δx_px = f · g_x / p_pix   [pixels]
```

*Assumptions:* paraxial (`|g| ≪ 1`), gradient approximately constant over the
subaperture, spot inside its own pixel block. *Validity:* slopes up to
`s_max = (p/2)·p_pix/f` — 5.0 mrad for the defaults; beyond that the spot leaves
the block and the model truncates rather than modelling crosstalk.

**Pixel-integrated spot.** Exact, separable, closed form in the error function:

```
P(i,j) = N · [F_x(u+½) − F_x(u−½)] · [F_y(v+½) − F_y(v−½)],
F(u)   = ½(1 + erf((u − c)/(σ√2)))
```

*Assumptions:* 100 % fill factor, uniform pixel response, axis-aligned
elongation only (a rotated Gaussian is not separable and is not implemented).

**Detector noise chain** (Thomas et al. 2006, *MNRAS* **371**, 323): signal +
uniform background `B` [e⁻/px] → Poisson → `+ N(0, R²)` read noise [e⁻ RMS],
independent per pixel, not clipped.

**Thresholded centre of gravity** (Thomas et al. 2006; Hardy 1998 ch. 5):

```
x̂ = Σᵢ wᵢ xᵢ / Σᵢ wᵢ,      wᵢ = max(Iᵢ − t, 0)     [pixels]
```

*Assumptions:* symmetric spot fully inside the stamp, background removed.
*Validity:* unbiased noise-free (verified to 1.4×10⁻⁵ px, §Validation); variance
grows with window area × per-pixel noise variance; an unsubtracted background
shrinks the estimate by `S/(S+Bp²)`.

**Correlation with parabolic peak interpolation** (Poyneer 2003, *Applied
Optics* **42**, 5807): cross-correlate the stamp with a reference spot, take the
integer peak, refine with `δ = (c₋ − c₊)/(2(c₋ − 2c₀ + c₊))` per axis.
*Validity:* exact only for a parabolic peak, so it carries a systematic
sub-pixel "S-curve" bias — measured here as **0.030 px peak, 0.019 px RMS**.

**Noise propagation for the CoG** (standard result; the explicit variance
decomposition is given by Thomas et al. 2006, the photon-limited `σ_spot/√N`
form and the window-area read-noise scaling by Hardy 1998 ch. 5, and the
photon-limited bound by Winick 1986, *JOSA A* **3**, 1809). Re-derived from
first principles in the docstring of `cog_noise_sigma`:

```
Var(x̂) = M2/N + (B + R²)/N² · Σᵢ(xᵢ − x̄)²                     [px²]
Σᵢ(xᵢ − x̄)² = p²(p² − 1)/12 + p² d²   for a p×p window, spot at d
σ_g = sqrt(Var(x̂)) · p_pix / f                                 [rad]
```

`M2` is the second central moment of the spot profile, evaluated numerically
from the pixel-integrated, window-truncated profile (so pixel binning and edge
truncation are exact). In the photon limit this is `σ_spot/√N`, i.e. a
noise-equivalent angle `≈ 0.44 (λ/d)/√N`. *Validity:* first-order linearization
of a ratio estimator, no threshold, no clipping. **It breaks at very low flux —
measured 35× under-prediction at 100 e⁻** (§Validation §3).

## Architecture

```
src/shacksim/
├── __init__.py    public API and __version__
├── geometry.py    LensletArray — pitch, focal length, pupil mask, spot size,
│                  slope <-> displacement conversion (derivation in docstring)
├── sensor.py      subaperture_spot, simulate_frame, extract_subapertures,
│                  generate_subaperture_dataset      (synthetic data)
├── wavefront.py   tilt_slopes, defocus_slopes, random_slopes, slope_rms
├── slopes.py      cog_displacement/cog_slopes, correlation_*, reference_template,
│                  cog_noise_sigma                   (classical, implemented first)
└── ml.py          MLSlopeEstimator                  (5 x MLPRegressor ensemble)
```

No cross-product imports; the package is self-contained.

## Installation

Requires Python ≥ 3.11 with numpy, scipy, scikit-learn and matplotlib.

```bash
cd products/P018
pip install -e .            # or: export PYTHONPATH=src
```

All commands below are run from `products/P018/`.

## Quick start

```python
import numpy as np
from shacksim import (
    LensletArray, MLSlopeEstimator, cog_noise_sigma, cog_slopes,
    correlation_slopes, generate_subaperture_dataset, simulate_frame, tilt_slopes,
)

array = LensletArray()                    # 8x8, 500 um pitch, f = 50 mm, 16 px/subap
print(array.summary())                    # spot FWHM 2.084 px, 52 illuminated subaps

# 1. Known tilt in, slopes out
truth = tilt_slopes(array, gx=1.0e-3, gy=-5.0e-4)          # (52, 2) rad
frame = simulate_frame(array, truth, photons=3000.0,
                       background=1.0, read_noise=3.0, seed=42)   # (128, 128) e-
cog = cog_slopes(frame, array, threshold=10.0)             # (52, 2) rad
corr = correlation_slopes(frame, array)
print("CoG bias  [px]", array.slope_to_displacement((cog - truth).mean(axis=0)))
print("CoG sigma [px]", array.slope_to_displacement(cog.std(axis=0)))

# 2. What the standard noise model predicts for that measurement
print("predicted sigma [px]",
      array.slope_to_displacement(cog_noise_sigma(array, 3000.0, 1.0, 3.0)))

# 3. Train the learned estimator and get per-slope confidence
x_tr, y_tr = generate_subaperture_dataset(
    array, 9000, photons=(30.0, 30000.0), background=1.0, read_noise=3.0,
    elongation=(1.0, 3.0), seed=100)                       # ~40 s to fit
model = MLSlopeEstimator(array, n_estimators=5, random_state=0).fit(x_tr, y_tr)

x_te, y_te = generate_subaperture_dataset(                 # held out
    array, 2000, photons=50.0, background=1.0, read_noise=3.0, seed=9050)
pred, std = model.predict(x_te, return_std=True)
rms = np.sqrt(np.mean(array.slope_to_displacement(pred - y_te) ** 2))
print(f"ML RMS {rms:.3f} px, mean ensemble spread "
      f"{array.slope_to_displacement(std).mean():.3f} px")
```

`std` is **not** a calibrated 1-σ error bar — see [Limitations](#limitations).

## Configuration

`LensletArray` (all geometry flows from these):

| Parameter | Default | Units | Meaning |
|---|---|---|---|
| `n_lenslets` | 8 | — | Lenslets per side, ≥ 2 |
| `pitch` | 500e-6 | m | Lenslet pitch = clear aperture, > 0 |
| `focal_length` | 50e-3 | m | Lenslet focal length, > 0 |
| `pixels_per_sub` | 16 | px | Detector pixels per subaperture per axis, ≥ 4 |
| `wavelength` | 633e-9 | m | Monochromatic wavelength, > 0 |
| `pupil_diameter` | `n·pitch` | m | Illuminated pupil diameter, > 0 |
| `obscuration` | 0.0 | — | Central obscuration fraction, [0, 1) |
| `fill_threshold` | 0.5 | — | Min. in-pupil area fraction for a valid subaperture, (0, 1] |

Simulation and estimation:

| Parameter | Default | Units | Meaning |
|---|---|---|---|
| `photons` | 1000 | e⁻ | Signal per subaperture; scalar, `(n_valid,)`, or `(lo, hi)` log-uniform |
| `background` | 0.0 | e⁻/px | Uniform background, ≥ 0 |
| `read_noise` | 0.0 | e⁻ RMS | Gaussian read noise, ≥ 0 |
| `elongation` | 1.0 | — | Spot elongation ratio, ≥ 1 (axis-aligned only) |
| `elongation_axis` | `"x"` | — | `"x"` or `"y"` |
| `shot_noise` | True | — | Poisson noise on signal + background |
| `slope_fraction` | 0.6 | — | Dataset slopes drawn in ±fraction × field, (0, 1] |
| `seed` | None | — | Fixed int → bitwise reproducible output |
| `threshold` (CoG) | 0.0 | e⁻ | Subtracted before weighting; `B + 3R` is a good default |
| `clip_negative` (CoG) | True | — | `False` gives the linear estimator the noise model describes |
| `subtract_mean` (corr) | True | — | Makes the correlation estimate background-invariant |
| `n_estimators` | 5 | — | Ensemble members, ≥ 2 |
| `hidden_layer_sizes` | (96, 48) | — | MLP hidden widths |
| `alpha` / `max_iter` / `random_state` | 1e-4 / 400 / 0 | — | L2, Adam epochs, base seed |

Invalid input raises `ValueError` / `TypeError` with an actionable message
(negative pitch, `pixels_per_sub < 4`, wrong slope-array shape, non-finite
pixels, elongation < 1, `predict()` before `fit()`, a stamp size that does not
match the array, …).

## Examples

Both use the Agg backend and save PNGs to `screenshots/`.

```bash
PYTHONPATH=src python examples/spot_field.py            # ~3 s
PYTHONPATH=src python examples/slope_error_vs_flux.py   # ~25 s
```

**`spot_field.py` → `screenshots/spot_field.png`** — a simulated 128 × 128
frame for a global tilt plus defocus at 2000 e⁻/subaperture, with the measured
CoG slope vectors drawn over the 52 illuminated subapertures, and a second panel
showing the true slope field with the measurement residual magnified 200×
(residual RMS: CoG 0.0248 px, correlation 0.0344 px).

![Spot field with slope vectors](screenshots/spot_field.png)

**`slope_error_vs_flux.py` → `screenshots/slope_error_vs_flux.png`** — RMS slope
error against photons per subaperture for the thresholded CoG, the correlation
estimator and the ML ensemble (with its spread as a band), for round and 3×
elongated spots, with the analytic linear-CoG noise curve and the ML/CoG
crossover marked. This is a *reduced* run (6000 training stamps, 3 members) so
it finishes quickly; it reproduces the crossover (≈ 71 e⁻ round, ≈ 173 e⁻
elongated) but not the exact validation figures.

![Slope error vs flux](screenshots/slope_error_vs_flux.png)

## Validation

Level 2 (Research). Full evidence, derivations, tolerances and raw output:
[`validation/VALIDATION.md`](validation/VALIDATION.md) and
[`validation/validation_output.txt`](validation/validation_output.txt),
regenerated by

```bash
PYTHONPATH=src python validation/run_validation.py    # ~56 s
```

**1 — Known global tilt gives the analytically predicted uniform slope.** For
`W = g_x X + g_y Y` the gradient is constant, so every subaperture must report
the same slope and the spot must sit at `Δx = f g_x / p_pix` pixels (hand check:
`1.000e-3 × 50.0e-3 / 31.25e-6 = 1.600000 px`). Over five tilts up to 2.5 mrad
on noise-free frames the **worst-case CoG slope error is 8.702×10⁻⁹ rad =
1.392×10⁻⁵ px** (tolerance 1×10⁻⁸ rad, **PASS**); the correlation estimator
reaches 0.0281 px (tolerance 0.05 px, **PASS**), the difference being its
parabolic-interpolation bias. On a noisy frame (3000 e⁻, B = 1, R = 3) the
measured field is uniform to **0.025 px RMS across all 52 subapertures** with a
bias of 0.0008 / 0.0022 px.

**2 — Zero wavefront gives zero slopes.** `max|CoG| = 0`,
`max|correlation| = 6.1×10⁻²⁰ rad` (tolerance 1×10⁻¹² rad, **PASS**). Sweeping
sub-pixel shifts quantifies the correlation S-curve: **0.0302 px peak, 0.0185 px
RMS**, against **6.2×10⁻⁸ px** for the CoG.

**3 — Slope error vs photon count against the standard noise-propagation
expression.** Measured on the linear un-thresholded CoG (the estimator the
expression describes), 4000 stamps per point:

| N [e⁻] | measured σ_x [px] | predicted [px] | ratio |
|---|---|---|---|
| 100 | 95.8584 | 2.7202 | **35.24** |
| 300 | 0.9670 | 0.9069 | 1.066 |
| 1000 | 0.2772 | 0.2738 | 1.012 |
| 3000 | 0.0923 | 0.0921 | 1.002 |
| 10000 | 0.0292 | 0.0288 | 1.015 |
| 30000 | 0.0104 | 0.0105 | 0.990 |

**Ratio 0.990–1.066 for N ≥ 300 e⁻ (tolerance 0.85–1.15, PASS).** At 100 e⁻ the
expression under-predicts by **35×** — the first-order linearization of the
ratio estimator fails when the denominator can approach zero. Reported as a
failure, not tuned away.

**4 — Centre-of-gravity bias under a background offset.** Analytic prediction
`x̂ = d·S/(S + Bp²)`, matched to **1.39×10⁻⁵ px worst case** (tolerance 1×10⁻³
px, **PASS**) across S ∈ {1000, 5000} e⁻ and B ∈ {0, 0.5, 2, 10, 50} e⁻/px.
Magnitude: at S = 1000 e⁻ a background of **0.5 e⁻/px costs 11.3 %** of the
measured wavefront, **2 e⁻/px costs 33.9 %**, **10 e⁻/px costs 71.9 %**.

**5 — Learned vs classical across photon counts.** See
[Benchmark results](#benchmark-results).

## Benchmark results

9000 training stamps (seed 100), 2000 held-out stamps per point (seeds
`9000 + N + 7·elongation`), CoG threshold tuned per operating point on a third
disjoint seed family. RMS slope error in pixels of spot displacement:

**Round diffraction-limited spots**

| N [e⁻] | thresholded CoG | correlation | **ML ensemble** | ML/CoG |
|---|---|---|---|---|
| 30 | 2.5705 | 2.8780 | **2.2600** | 0.879 |
| 50 | 1.9161 | **0.9630** | 1.6591 | 0.866 |
| 100 | 0.3852 | **0.1944** | 0.7842 | 2.036 |
| 300 | 0.0891 | **0.0842** | 0.3118 | 3.501 |
| 1000 | **0.0367** | 0.0446 | 0.1382 | 3.766 |
| 3000 | **0.0193** | 0.0301 | 0.1160 | 6.010 |
| 10000 | **0.0098** | 0.0235 | 0.1027 | 10.494 |

**3× elongated spots**

| N [e⁻] | thresholded CoG | correlation | **ML ensemble** | ML/CoG |
|---|---|---|---|---|
| 30 | 2.6674 | 4.0164 | **2.5594** | 0.960 |
| 50 | 2.5064 | 3.0531 | **2.1165** | 0.844 |
| 100 | 1.6310 | 1.0323 | **1.1840** | 0.726 |
| 300 | **0.3053** | 0.4541 | 0.4344 | 1.423 |
| 1000 | **0.1040** | 0.1999 | 0.2304 | 2.215 |
| 3000 | **0.0622** | 0.1133 | 0.1706 | 2.742 |
| 10000 | 0.0970 | **0.0662** | 0.1756 | 1.810 |

- **Crossover, round spots: between 50 and 100 e⁻/subaperture.** ML wins by
  1.14× at 30 e⁻ and 1.15× at 50 e⁻; loses by 10.5× at 10 000 e⁻.
- **Crossover, 3× elongated spots: between 100 and 300 e⁻.** ML wins by 1.38× at
  100 e⁻, 1.19× at 50 e⁻, 1.04× at 30 e⁻; loses by up to 2.7× at 3000 e⁻.
- **The advantage is real but small (≤ 1.38×) and confined to a narrow flux
  window.** With more than ≈ 100 e⁻ per subaperture, use the analytic estimator.
- **Against the correlation estimator the learned model mostly loses** — 1 of 7
  flux levels for round spots, 3 of 7 for elongated. Benchmarking only against
  the CoG, as the specification asks, would overstate the case; the correlation
  column is reported for that reason.
- **ML error floors at ≈ 0.10–0.18 px** and stops improving with flux: a
  finite-capacity, L2-regularized, early-stopped network spanning a three-decade
  flux range and a 3× spot-shape range carries irreducible approximation error
  and shrinks toward the mean of the slope prior.
- **Best classical estimator by regime:** correlation from ≈ 50 to ≈ 300 e⁻
  (round spots), thresholded CoG above that; the correlation estimator floors at
  ≈ 0.023 px because of its S-curve bias.
- **Compute:** training 5 × MLP(96,48) on 9000 stamps takes **39.9 s** on 2 CPU
  cores (108.9 s under load, bit-identical results); full validation run 56.1 s;
  test suite 11 s; inference ≈ 0.05 ms per stamp per member.

## AI model details

Full card: [`MODEL_CARD.md`](MODEL_CARD.md). Data: [`DATASET_CARD.md`](DATASET_CARD.md).

- **Baseline first.** `cog_displacement` and `correlation_displacement` were
  implemented and validated (§Validation 1–4) before the model existed, and the
  CoG threshold is tuned per operating point so the baseline is at its best.
- **Architecture.** 5 × `MLPRegressor(hidden_layer_sizes=(96,48))`, ReLU, Adam,
  `alpha=1e-4`, `max_iter=400`, `early_stopping=True`, on a 257-vector: the
  flux-normalized 256-pixel stamp plus `log10(1 + total counts)`. ≈ 125 k
  parameters total.
- **Dataset.** Entirely synthetic from an idealized optical model, generated by
  committed scripts, deterministic under fixed seeds, not committed. Unmodelled:
  Airy rings, lenslet aberrations, chromatic effects, inter-subaperture
  crosstalk, arbitrary-angle spot elongation, turbulence statistics, dead/hot
  pixels, PRNU/DSNU, saturation, quantization, charge diffusion — full list in
  the dataset card.
- **Training procedure.** 9000 stamps, photons log-uniform 30–30 000 e⁻,
  elongation 1–3× along x, seed 100; 10 % held internally for early stopping;
  **no hyperparameter search**, so no test information entered any choice.
- **Test split.** Disjoint RNG streams, not a partition: training seed 100,
  tuning `300 + N + 11·elongation`, test `9000 + N + 7·elongation`. Same
  distribution — an in-distribution generalization test only.
- **Metrics.** See [Benchmark results](#benchmark-results).
- **Uncertainty output.** `predict(..., return_std=True)` returns the per-slope
  ensemble standard deviation in radians. Measured spread/RMS-error ratio spans
  **0.15 to 1.12**: it under-states the true error by up to **6.5×** in the
  low-flux regime where the model is meant to be used, and over-states it at
  high flux on round spots. **Not a calibrated 1-σ bound** — usable only as a
  qualitative photon-starvation flag.
- **Failure cases.** Silent accuracy loss above the crossover; extrapolation
  outside the trained envelope (a pinned test shows > 10× deficit); rotated spot
  elongation is not representable; wrong photometric units silently move the
  model along the flux axis; a changed pitch/focal length/wavelength invalidates
  it without raising. Every accuracy failure mode is silent.
- **Deviation.** A convolutional network is the natural architecture here;
  PyTorch is unavailable in this build environment and scikit-learn has no
  convolutional layers, so an MLP ensemble is used. Recorded in
  [Limitations](#limitations) and in `MODEL_CARD.md`.

**This model is not certified for operational flight use.**

## Hardware requirements

CPU only. Developed and validated on 2 x86-64 cores with Python 3.11.15,
numpy 2.4.4, scipy 1.17.1, scikit-learn 1.8.0, matplotlib 3.10.9. Peak memory
< 300 MB. No GPU, no accelerator, no network access. Full validation run under
1 minute; test suite 11 s.

## Limitations

1. **Deviation from the ideal design: MLP ensemble instead of a CNN.** PyTorch
   is unavailable in this build environment. The model lacks weight sharing and
   translation equivariance, a materially weaker inductive bias for a
   translation-estimation task. The reported accuracy — especially the
   ≈ 0.1 px high-flux floor — is a property of this substitute architecture and
   is **not** evidence about CNN slope estimation.
2. **The learned estimator only wins below ≈ 100 e⁻ per subaperture**, by at
   most 1.38×, and loses by up to 10.5× above. Against the correlation baseline
   it loses almost everywhere. There is no broad regime in which it is the right
   default.
3. **The confidence output is not calibrated** (spread/error 0.15–1.12) and must
   not be consumed as a measurement covariance by any reconstructor or filter.
4. **The standard noise-propagation expression fails below ≈ 300 e⁻** — measured
   35× under-prediction at 100 e⁻. It is a first-order linearization of a ratio
   estimator; do not use it in the photon-starved regime it is most often quoted
   for.
5. **All data is synthetic from an idealized optical model.** Not modelled:
   Airy ring structure (the Gaussian core understates wing flux), lenslet
   aberrations and manufacturing variation, array-to-detector misalignment,
   chromatic/broadband effects, vignetting and partially illuminated
   subapertures, scattered light, inter-subaperture crosstalk (a spot pushed
   past the field is truncated, not spilled into the neighbour), dead/hot
   pixels, PRNU/DSNU, dark current, charge diffusion, saturation, nonlinearity,
   ADC quantization, EMCCD excess noise. Real-hardware performance is unknown;
   every unmodelled effect can only degrade results, so these numbers are an
   **optimistic bound**.
6. **No turbulence.** Slopes are drawn independently and uniformly. Real
   atmospheric slopes are strongly spatially correlated (Kolmogorov) and
   temporally evolving; nothing here measures behaviour on turbulent
   wavefronts. Phase-screen work belongs to P003/P011.
7. **Only axis-aligned spot elongation is representable.** A rotated elliptical
   Gaussian is not separable in x and y and is not implemented, so the
   sodium-laser-guide-star geometry — radial elongation whose angle varies
   across the pupil — is out of scope for both the simulator and the model.
8. **No wavefront reconstruction.** This product outputs slopes; converting
   slopes to phase (Southwell/Fried geometry, least squares, Zernike fitting) is
   P014's job and is deliberately absent here.
9. **Narrow characterized envelope.** Everything is measured at 8 × 8 lenslets,
   500 µm pitch, f = 50 mm, 16 px/subaperture, λ = 633 nm, B = 1 e⁻/px,
   R = 3 e⁻ RMS. The code is configurable; the *numbers* are not transferable.
   In particular the spot is sampled at only 2.08 px FWHM — adequate but not
   generous — and results at coarser sampling will differ.
10. **The correlation estimator is given a perfectly matched template**
    generated from the same model that generated the data. Real template
    mismatch is an unmodelled error source that would make this baseline worse.
11. **Single frame, no temporal processing.** No frame stacking, no loop, no
    slope prediction, no drift or jitter model.
12. Reported floats may shift in the last digits with a different BLAS or
    scikit-learn version; the qualitative conclusions are robust.

## Safety statement

This software is **research-grade**. It is **not flight-qualified, not
certified, and not approved for operational aerospace use.** No DO-178C or
ECSS-E-ST-40C process was followed; there is no independent verification and no
qualification testing. Do not place the learned estimator in any adaptive-optics,
pointing or guidance control loop: its accuracy failure modes are silent and its
confidence output is not calibrated, so a downstream consumer cannot detect or
bound its errors — and a wavefront-sensor gain error feeds directly back into
loop gain.

## Roadmap

- Replace the Gaussian spot core with the true Airy pattern (and optionally a
  Kolmogorov-aberrated subaperture PSF) and re-measure how much of the reported
  accuracy survives the wings.
- Add inter-subaperture crosstalk and partially illuminated edge subapertures,
  which are the first-order realism gaps at the array level.
- Re-implement the estimator as a CNN once a deep-learning framework is
  available and re-run the identical benchmark to test whether the 0.1 px floor
  is an artefact of the MLP substitution.
- Calibrate the confidence output (variance scaling or conformal prediction)
  against held-out error and report coverage.
- Add a maximum-likelihood / matched-filter baseline and an explicit Cramér-Rao
  bound curve to the error-vs-flux comparison.
- Support rotated elliptical spots so that laser-guide-star elongation can be
  studied, and train per-flux-regime specialists to remove the mixed-regime
  compromise.

## License

Apache-2.0. See [LICENSE](LICENSE). Copyright © 2026 OPTIMA Organisation.

## Credits

This is under reserved rights obtained by OPTIMA Organisation.

## Citation

```bibtex
@software{shacksim_2026,
  title   = {ShackSim: Shack-Hartmann wavefront sensor simulation and slope
             extraction with classical and learned estimators},
  author  = {{OPTIMA Organisation}},
  year    = {2026},
  version = {0.1.0},
  license = {Apache-2.0},
  note    = {Research-grade; not flight-qualified. Product P018.}
}
```
