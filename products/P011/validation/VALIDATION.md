# WaveForge — Validation Evidence (Level 3)

Every number in this file was produced by running the scripts in this directory
during the 0.1.0 build session on **2026-08-29**, on Python 3.11.15 with
numpy 2.4.4, scipy 1.17.1, scikit-learn 1.8.0 and matplotlib 3.10.9, on a
2-core machine. The verbatim console transcript of each script is stored beside
it:

| Script | Transcript | Wall time |
|---|---|---:|
| `validate_zernike.py` | `validate_zernike_output.txt` | < 1 s |
| `validate_fitting_error.py` | `validate_fitting_error_output.txt` | 23.0 s |
| `validate_rejection_tf.py` | `validate_rejection_tf_output.txt` | 6.2 s |
| `validate_strehl_marechal.py` | `validate_strehl_marechal_output.txt` | 5.9 s |
| `validate_atmosphere.py` | `validate_atmosphere_output.txt` | 11.6 s |
| `validate_predictor.py` | `validate_predictor_output.txt` | 98.2 s |

Rerun any of them from `products/P011/`:

```bash
PYTHONPATH=src python validation/validate_zernike.py
```

All runs are deterministic apart from wall-clock timings. Seeds are partitioned
so that no block is reused:

| Purpose | Seeds |
|---|---|
| Zernike orthonormality | none (deterministic quadrature) |
| Fitting-error modal screens | `30000 + 100000·(r0 index + 1) + 977 k` |
| Fitting-error zonal screens | `30000 + 900000 + 977 k` |
| Atmosphere structure function | `41000 + 31 k` (no subharmonics), `52000 + 17 k` (subharmonic sweep) |
| Atmosphere modal content | `70000 + 23 k` |
| Strehl / Maréchal screens | `60000 + 13 k` |
| Predictor training screens | `101, 102, 103` |
| Predictor held-out screens | `901, 902` |
| Predictor gain-tuning screen | `555` (never used for a reported number) |
| Sensor-noise realisations | `4242` (tuning), `8888` (training noise), `31337` (evaluation) |

**Reading the results.** Two of the checks below record a *documented
deviation* rather than a pass. They are marked as such, their cause is
quantified, and no tolerance was widened to convert them into a pass.

---

## 1. Zernike orthonormality and Noll statistics

Script: `validate_zernike.py`. Reference: R. J. Noll, "Zernike polynomials and
atmospheric turbulence", *J. Opt. Soc. Am.* **66**(3), 207–211 (1976), Eqs. 2,
3, 8, 18, 32 and Table IV.

### 1a. Orthonormality under exact quadrature — PASS

Modes `j = 1…21`, 64-node Gauss–Legendre in `ρ²` times a 128-point uniform
angular rule, with the area weight `1/π`:

| Quantity | Measured | Tolerance |
|---|---:|---:|
| worst `\|G_ii − 1\|` | 6.217e-15 | 1e-12 |
| worst `\|G_ij\|`, `i ≠ j` | 3.478e-15 | 1e-12 |

### 1b. Orthonormality on the Cartesian grids the package actually uses

Not a pass/fail check — this is the discretisation error a user incurs.

| `n_pix` | worst `\|G_ii − 1\|` | worst `\|G_ij\|` |
|---:|---:|---:|
| 32 | 4.561e-02 | 4.150e-02 |
| 64 | 2.468e-02 | 1.660e-02 |
| 128 | 9.186e-03 | 8.955e-03 |
| 256 | 1.863e-03 | 2.204e-03 |

The deviation falls roughly as `1/n_pix` and is the dominant error in modal
fitting on a pixel grid. **A 64-pixel pupil is not orthonormal to better than
2 %**, which is why validation 2 quotes ratios rather than agreement to
numerical precision.

### 1c. Noll residual variances `Δ_J` — PASS

`Δ_J` computed from the analytic Kolmogorov Zernike variances, against Noll's
Table IV, in units of `(D/r₀)^(5/3)`:

| J | computed | Noll (1976) | rel. diff |
|---:|---:|---:|---:|
| 1 | 1.032765 | 1.029900 | +0.278 % |
| 2 | 0.583737 | 0.582000 | +0.298 % |
| 3 | 0.134708 | 0.134000 | +0.529 % |
| 4 | 0.111483 | 0.111000 | +0.435 % |
| 11 | 0.037803 | 0.037700 | +0.273 % |
| 21 | 0.020839 | 0.020800 | +0.188 % |

Worst relative difference over `J = 1…21`: **0.529 %**, against a 1 % tolerance
set by Noll's three-significant-figure table. **PASS.**

### 1d. Per-mode variances — DOCUMENTED DEVIATION, not a pass

Differencing two rounded table entries (`⟨a_j²⟩ = Δ_{j−1} − Δ_j`) loses two
digits. The test is whether the computed value lies inside the rounding
half-width of the two entries it is compared against:

* **13 of 20** modes are consistent with what the published table can resolve;
* every one of the seven inconsistent modes is **high**, by an amount matching
  the +0.25 % systematic offset established in 1e.

The tolerance was not widened. Worst case: `j = 19`, computed 0.001191 against
an implied 0.001100 (rounding tolerance 0.000050).

### 1e. Independent cross-check of the total variance — PASS

The piston-removed variance over a circular aperture computed **without
Zernikes at all**, from `σ² = ½ ⟨D_φ(|r₁ − r₂|)⟩` with the analytic separation
density of a disc:

| Route | Value, `(D/r₀)^(5/3)` |
|---|---:|
| structure-function integral | 1.032422 |
| Zernike-variance sum | 1.032765 |
| Noll (1976) published `Δ₁` | 1.029900 |

The two independent derivations inside this package agree to **0.033 %** and
both sit **+0.24 % / +0.28 %** above Noll's published third significant figure.
This is the origin of the deviation in 1d, and it is reported rather than
absorbed.

### 1f. Noll's large-`J` asymptote — reported, not a pass

`Δ_J ≈ 0.2944 J^(−√3/2)` (Noll Eq. 32) agrees with the exact sum to **1.15 %**
at `J = 21`, where it is customarily used, but drifts at larger `J`:
−5.7 % at `J = 500`, −16.2 % at `J = 20000`. The exact sum falls with a local
log-slope tending to **0.834**, close to the `5/6 = 0.8333` implied by mode
counting, whereas the asymptote uses `√3/2 = 0.8660`. The package uses the
exact sum everywhere; the asymptote is provided as published reference data.

---

## 2. Residual error versus D/r₀

Script: `validate_fitting_error.py`.

### 2a. Total variance of the generated screens — reported bias

64-pixel pupil over 0.5 m, screens 512² with six subharmonic levels, 40
independent realisations per `r₀`:

| `r₀` [m] | `D/r₀` | measured [rad²] | Noll `Δ₁` [rad²] | ratio |
|---:|---:|---:|---:|---:|
| 0.400 | 1.25 | 1.0124 | 1.4980 | 0.676 |
| 0.200 | 2.50 | 4.6012 | 4.7559 | 0.967 |
| 0.100 | 5.00 | 11.0482 | 15.0991 | 0.732 |
| 0.050 | 10.00 | 42.4872 | 47.9367 | 0.886 |

Pooled mean ratio **0.815**, range 0.676–0.967. Two effects, both reported:
the Fourier screen carries no power below `1/(N d)` and the deficit sits almost
entirely in tip and tilt; and the estimator itself is noisy because the total
variance is dominated by two modes.

### 2b. Modal residual after removing Noll modes 1…J — PASS

| J | mean ratio (measured / Noll) | spread over four `D/r₀` |
|---:|---:|---:|
| 3 | 0.975 | 0.198 |
| 6 | 0.975 | 0.060 |
| 10 | 1.009 | 0.083 |
| 15 | 0.986 | 0.082 |
| 21 | 0.996 | 0.044 |

The ratio is flat in `D/r₀` to within the Monte-Carlo scatter, which is the
`(D/r₀)^(5/3)` scaling being confirmed, and the offset from unity is
**≤ 2.5 %** — smaller than the 64-pixel discrete orthonormality error of 1b.

### 2c. Zonal fitting error versus actuator pitch — PASS on the coefficient

128-pixel pupil over 0.5 m, `r₀ = 0.10 m`, screens 1024² with five subharmonic
levels, 20 realisations. `lost` is the fraction of the fitting-error power that
lies above the grid Nyquist and so is absent from the screens:

| actuators across | `d_act` [m] | `σ²` [rad²] | `a_F` | lost | `a_F` corrected |
|---:|---:|---:|---:|---:|---:|
| 5 | 0.12500 | 0.27437 | 0.1892 | 0.3 % | 0.1897 |
| 9 | 0.06250 | 0.10068 | 0.2204 | 1.0 % | 0.2226 |
| 17 | 0.03125 | 0.03365 | 0.2338 | 3.1 % | 0.2414 |
| 33 | 0.01562 | 0.01115 | 0.2461 | 9.9 % | **0.2732** |

* corrected `a_F` at the finest pitch: **0.2732**
* derived ideal-low-pass value `0.0229 · 2π · 3/5 · 2^(5/3)`: **0.2741** → −0.3 %
* published Hudgin (1977) / Hardy (1998) Table 6.1 value **0.28** → −2.4 %

**PASS** against the 10 % criterion of requirement R-09.

The exponent measured across pitch is **1.544**, not 5/3, and the local
exponent rises monotonically with actuator count (1.446 → 1.581 → 1.593). Both
are the same finite-aperture effect: the `(d_act/r₀)^(5/3)` law is an
infinite-aperture result, whereas a mirror with only 5 × 5 Gaussian influence
functions fitted by unconstrained least squares over a 0.5 m pupil removes far
more than a hard spatial low-pass would. The two converge as the actuator count
grows, which is why the coefficient is quoted at the finest pitch and why the
law should be used for sizing only when many actuators span the aperture.

---

## 3. Closed-loop rejection transfer function

Script: `validate_rejection_tf.py`. Reference: P.-Y. Madec, "Control
techniques", in *Adaptive Optics in Astronomy*, ed. F. Roddier, CUP 1999, Ch. 3;
Hardy 1998 Sec. 7.3.

### 3a. Scalar loop against `|E(z)|` — PASS

Thirty combinations of latency `d ∈ {1,2,3}`, gain at 20 % and 50 % of the
stability limit, and frequencies 7–379 Hz at a 1 kHz frame rate, with the
residual amplitude extracted by synchronous detection over 2000 settled frames:

**worst relative difference 1.73e-14**, against a 1e-4 tolerance. The
time-domain implementation and the analytic expression are the same equation to
double precision.

### 3b. Stability boundary — PASS

| `d` | computed | closed form | abs. diff |
|---:|---:|---:|---:|
| 1 | 2.000000 | 2 | 3.46e-10 |
| 2 | 1.000000 | 1 | 1.80e-10 |
| 3 | 0.618034 | `2 sin(π/10)` | 3.21e-11 |

Time-domain confirmation: at `0.9 ×` the limit the residual stays bounded at
every latency; at `1.1 ×` it exceeds 1e6 within 400 frames at every latency.

### 3c. Noise variance amplification — PASS

`Σ h_k²` of the noise transfer impulse response against the classical
`g/(2−g)` for one frame of latency, over `g = 0.1…1.5`: **worst relative
difference 1.57e-16**.

### 3d. Full AO loop (pupil, SH sensor, DM, reconstructor) — PASS

A sinusoidal tilt of unit RMS replaces the atmosphere; the residual RMS is
compared with the same analytic `|E(z)|` over twelve (latency, gain, frequency)
combinations.

The tolerance is **measured, not chosen**: reconstructing a unit-RMS tilt and
putting it on the mirror in one perfect step leaves a residual of **0.0224**
(2.24 % of the input), and no closed-loop measurement can agree with the scalar
model better than that.

| Quantity | Value |
|---|---:|
| worst relative difference | 2.568e-02 |
| worst absolute difference | 6.735e-03 of a unit-RMS input |
| measured modelling floor | 2.235e-02 relative |

Ten of the twelve points agree to better than 1 %. The worst point is the
deepest-rejection case (`d = 2`, `g = 0.7`, 11 Hz), where the loop attenuates
the disturbance to a tenth of its input so a 2.5e-3 absolute error reads as
2.6 % relative. **PASS** against the measured floor with 20 % headroom.

---

## 4. Strehl ratio versus the Maréchal approximation

Script: `validate_strehl_marechal.py`. References: A. Maréchal, *Rev. Opt.*
**26**, 257 (1947); Born & Wolf, *Principles of Optics*, 7th ed., Sec. 9.1;
Hardy 1998 Eq. 4.20.

### 4a. Kolmogorov phase scaled to a target variance — validity range measured

200 screens per point, 64-pixel pupil, `S = |⟨exp(iφ)⟩|²`:

| `σ²` [rad²] | numerical `S` | `exp(−σ²)` | rel. err | `1 − σ²` | rel. err |
|---:|---:|---:|---:|---:|---:|
| 0.01 | 0.99004 | 0.99005 | +0.00 % | 0.99000 | −0.00 % |
| 0.10 | 0.90439 | 0.90484 | +0.05 % | 0.90000 | −0.48 % |
| 0.25 | 0.77636 | 0.77880 | +0.31 % | 0.75000 | −3.40 % |
| 0.50 | 0.59890 | 0.60653 | +1.27 % | 0.50000 | −16.51 % |
| 1.00 | 0.34979 | 0.36788 | +5.17 % | 0.00000 | −100 % |
| 2.00 | 0.11474 | 0.13534 | +17.95 % | 0.00000 | −100 % |
| 5.00 | 0.04046 | 0.00674 | −83.35 % | 0.00000 | −100 % |

* extended form within 5 %: up to **σ² = 0.5 rad²**
* quadratic form within 5 %: up to **σ² = 0.25 rad²**

Beyond `σ² ≈ 2.5` the numerical Strehl stops falling and saturates near the
speckle floor set by the number of coherence cells in the pupil: 0.040 measured
at `σ² = 5` against `exp(−5) = 0.0067`, a factor of six. The extended Maréchal
form must not be used there in either direction.

The small deficits at moderate `σ²` are far larger than the Monte-Carlo error
(12–17 standard errors), so they are reported rather than dismissed: each
screen is rescaled to a fixed *spatial* variance rather than drawn from the
free ensemble, and Kolmogorov phase on a finite pupil is not spatially
stationary.

### 4b. Genuine closed-loop residuals — DOCUMENTED DEVIATION

| gain | latency | `σ²` [rad²] | numerical `S` | `exp(−σ²)` | rel. err |
|---:|---:|---:|---:|---:|---:|
| 0.60 | 1 | 0.8388 | 0.51083 | 0.43223 | −15.39 % |
| 0.40 | 2 | 1.1873 | 0.39491 | 0.30506 | −22.75 % |
| 0.20 | 2 | 1.1583 | 0.35028 | 0.31402 | −10.35 % |
| 0.30 | 3 | 1.5258 | 0.28761 | 0.21745 | −24.39 % |
| 0.10 | 3 | 1.9432 | 0.19545 | 0.14324 | −26.71 % |

On real closed-loop residuals the extended Maréchal form **underestimates** the
Strehl by up to **26.7 %**. Two physical reasons: the residual variance
fluctuates frame to frame and the mean of `exp(−σ²)` over those fluctuations
exceeds `exp(−⟨σ²⟩)` by Jensen's inequality; and the residual is dominated by
high-order modes concentrated at the pupil edge, so its variance is not uniform
over the pupil. **Use `LoopResult.strehl` for performance claims and the
Maréchal form only for sizing.**

---

## 5. Phase-screen statistics

Script: `validate_atmosphere.py`. 128² screens at 20 mm sampling, `r₀ = 0.1 m`,
60 realisations, lags 1–16 samples.

### 5a. Implementation against the exact discrete expectation — PASS

| `r` [m] | measured [rad²] | exact [rad²] | ratio |
|---:|---:|---:|---:|
| 0.020 | 0.33627 | 0.33652 | 0.9993 |
| 0.100 | 4.34651 | 4.34972 | 0.9993 |
| 0.180 | 10.18918 | 10.19577 | 0.9994 |
| 0.300 | 20.31885 | 20.40635 | 0.9957 |

Worst deviation **0.50 %** against a 5 % tolerance. Any error here would be a
coding error; there is none at this level.

### 5b. Method bias against continuous theory — reported

| `r/r₀` | band-limited / `6.8839 (r/r₀)^(5/3)` |
|---:|---:|
| 0.2 | 0.7147 |
| 1.0 | 0.6319 |
| 1.8 | 0.5561 |
| 2.2 | 0.5260 |
| 3.0 | 0.4750 |

A Fourier screen on this grid carries only 47–71 % of the continuous structure
function. This is the method, not the implementation.

### 5c. Subharmonic augmentation — reported

Measured structure function relative to the **continuous** theory:

| levels | `r = 0.02` | `r = 0.10` | `r = 0.18` | `r = 0.32` |
|---:|---:|---:|---:|---:|
| 0 | 0.736 | 0.667 | 0.597 | 0.507 |
| 2 | 0.841 | 0.848 | 0.817 | 0.774 |
| 4 | 0.886 | 0.925 | 0.911 | 0.887 |
| 6 | 0.929 | 1.000 | 1.003 | 1.000 |

Six subharmonic levels recover the structure function to **1.00** at
separations from `0.10 m` upward. The remaining 7 % shortfall at one sample is
the high-frequency truncation at the grid Nyquist and cannot be fixed by
subharmonics.

### 5d. Modal content of pupil cut-outs — PASS for the higher orders

64-pixel pupil over 0.5 m, `D/r₀ = 5`, 150 screens of 512² with six
subharmonic levels. Monte-Carlo standard error 0.115 on each ratio:

| `j` | measured [rad²] | analytic [rad²] | ratio |
|---:|---:|---:|---:|
| 2 (tip) | 5.40898 | 6.56483 | 0.824 |
| 3 (tilt) | 5.39542 | 6.56483 | 0.822 |
| 4 | 0.29052 | 0.33956 | 0.856 |
| 8 | 0.09662 | 0.09055 | 1.067 |
| 11 | 0.03973 | 0.03589 | 1.107 |
| 21 | 0.01506 | 0.01741 | 0.865 |

Mean ratio over `j = 4…21`: **1.046**. Tip and tilt carry the whole of the
low-frequency deficit at **0.82**. Any study of overall tip/tilt magnitude with
these screens should use a larger generating screen or a von Kármán outer
scale.

---

## 6. Learned predictive control against the classical baselines

Script: `validate_predictor.py`. Configuration: 0.5 m aperture, `r₀ = 0.10 m`
(`D/r₀ = 5`), 10 m/s wind, 1 kHz frame rate, 8 × 8 Shack-Hartmann (52 valid
subapertures, 104 slopes), 9 × 9 deformable mirror with one margin ring.
Training data: seeds 101/102/103, 400 frames each (1200 samples). Held-out
data: seed 901 (open loop), seeds 901 and 902 (closed loop). Gains tuned on
seed 555, which contributes no reported number.

### 6a. Open-loop forecast accuracy — the learned model wins at every horizon

| horizon [frames] | ridge α | learned RMSE [rad/m] | pure-delay RMSE [rad/m] | ratio |
|---:|---:|---:|---:|---:|
| 1 | 0.01 | 1.4384 | 4.3191 | 0.333 |
| 2 | 0.1 | 2.8387 | 8.2481 | 0.344 |
| 3 | 1 | 4.2849 | 11.6673 | 0.367 |
| 4 | 10 | 5.8890 | 14.5840 | 0.404 |

### 6b. Uncertainty calibration — miscalibrated in both directions

| horizon | inside 1σ (68.3 %) | inside 2σ (95.4 %) | inside 3σ (99.7 %) | mean σ | RMSE |
|---:|---:|---:|---:|---:|---:|
| 1 | 82.7 % | 99.2 % | 100.0 % | 1.7244 | 1.4384 |
| 2 | 76.2 % | 97.7 % | 99.9 % | 2.3823 | 2.8387 |
| 3 | 67.4 % | 94.6 % | 99.5 % | 2.1099 | 4.2849 |
| 4 | 62.3 % | 91.9 % | 99.0 % | 2.6805 | 5.8890 |

The intervals are conservative at horizons 1–2 and **optimistic at horizons
3–4** (62.3 % inside one sigma at horizon 4 against a nominal 68.3 %, with the
mean σ well below the actual RMSE). The out-of-bag residual term is estimated
on the training screens and does not grow enough to cover the extra error the
model makes further ahead on unseen turbulence. **The output is a relative
confidence signal, not a probability.**

### 6c. Closed loop — the learned controller wins at every tested latency

Each controller at its own gain, tuned on seed 555 then frozen; results are the
mean over evaluation seeds 901 and 902, 500 frames with 150 discarded.

| latency `d` | `g` int | `g` delay | `g` ML | integrator `σ²` | pure delay `σ²` | learned `σ²` | `S` int | `S` delay | `S` ML | winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.40 | 0.40 | 0.30 | 0.6728 | 0.6728 | **0.5361** | 0.5434 | 0.5434 | **0.6085** | learned |
| 2 | 0.30 | 0.40 | 0.30 | 0.9274 | 0.8637 | **0.5354** | 0.4308 | 0.4519 | **0.6093** | learned |
| 3 | 0.20 | 0.60 | 0.30 | 1.2798 | 1.0669 | **0.5444** | 0.3040 | 0.3848 | **0.6042** | learned |
| 4 | 0.20 | 0.40 | 0.40 | 1.7085 | 1.3736 | **0.4958** | 0.2124 | 0.2818 | **0.6384** | learned |

The learned controller reduces the residual variance by **1.25×** at one frame
of latency and **3.45×** at four frames, and holds the Strehl near 0.60 while
the classical integrator falls from 0.54 to 0.21. The pure-delay baseline
running through the same pseudo-open-loop path recovers part of that gap, which
is why it is reported separately: at `d = 3` about a third of the improvement
over the plain integrator comes from the control formulation and two thirds
from the prediction itself.

**This is a positive result for the learned model in the regime it was trained
for. The next two sections show where it is not.**

### 6d. Failure mode — wind speed outside the training set

The model was trained at 10 m/s only. Fixed gain 0.4, latency 2:

| wind [m/s] | integrator `σ²` | learned `σ²` | ratio | winner |
|---:|---:|---:|---:|---|
| 5.0 | 0.9465 | 0.6178 | 0.653 | learned |
| 10.0 | 1.0351 | 0.5861 | 0.566 | learned |
| 15.0 | 1.2202 | 0.9440 | 0.774 | learned |
| 20.0 | 1.5356 | 1.8005 | 1.173 | **INTEGRATOR** |

At twice the training wind the learned controller is **17 % worse** than the
classical integrator. Its advantage decays continuously between 10 and 20 m/s;
there is no safe extrapolation.

### 6e. Failure mode — measurement noise the model was not trained for

Gain 0.4 for the learned controllers, tuned gain for the integrator, latency 2:

| flux [e⁻/subap] | slope σ [rad/m] | integrator `σ²` | ML trained clean | ML noise-matched | winner |
|---:|---:|---:|---:|---:|---|
| ∞ | 0.00 | 0.9578 | **0.5352** | 0.5352 | ML clean |
| 1000 | 7.79 | 1.5164 | 1.6065 | **0.6318** | ML matched |
| 300 | 16.91 | 2.4786 | 5.7542 | **1.0090** | ML matched |
| 100 | 39.70 | 5.0277 | 29.7540 | **3.8755** | ML matched |

A predictor trained on noise-free slopes and deployed on a noisy sensor is
catastrophic: **5.9× worse than the classical integrator at 100 e⁻ per
subaperture**. Retraining at the deployed noise level recovers and then
exceeds the classical performance, because the ridge then acts as a
noise-optimal spatio-temporal filter. Training the predictor at the noise level
it will meet is therefore mandatory, not optional; see `MODEL_CARD.md`.

---

## 7. Summary of outcomes

| # | Check | Outcome |
|---|---|---|
| 1a | Zernike orthonormality, exact quadrature | PASS (6.2e-15) |
| 1c | Noll `Δ_J` vs Table IV, `J = 1…21` | PASS (worst 0.529 %) |
| 1d | Per-mode variances vs differenced table entries | **DOCUMENTED DEVIATION** (13/20 consistent) |
| 1e | Total variance, two independent routes | PASS (agree to 0.033 %) |
| 2b | Modal residual vs Noll `Δ_J` on screens | PASS (≤ 2.5 % offset, flat in `D/r₀`) |
| 2c | Zonal fitting coefficient vs Hudgin 0.28 | PASS (0.2732, −2.4 %) |
| 3a | Scalar rejection TF vs analytic | PASS (1.7e-14) |
| 3b | Stability limits vs closed forms | PASS (≤ 3.5e-10) |
| 3c | Noise gain vs `g/(2−g)` | PASS (1.6e-16) |
| 3d | Full AO loop vs analytic `\|E(z)\|` | PASS against the measured 2.24 % modelling floor |
| 4a | Maréchal validity range | measured: extended form good to `σ² = 0.5` |
| 4b | Maréchal on closed-loop residuals | **DOCUMENTED DEVIATION** (−26.7 %) |
| 5a | Screens vs exact discrete expectation | PASS (0.50 %) |
| 5d | Modal content vs analytic variances | PASS for `j ≥ 4` (mean 1.046); tip/tilt 0.82 |
| 6c | Learned vs classical, closed loop | learned wins at `d = 1…4` |
| 6d | Learned at 2× training wind | **learned loses (17 % worse)** |
| 6e | Learned trained clean, deployed noisy | **learned loses badly (5.9× worse)** |

Nothing in this file was tuned after the fact. Where a check did not meet its
criterion it is labelled a documented deviation and its cause is quantified.
