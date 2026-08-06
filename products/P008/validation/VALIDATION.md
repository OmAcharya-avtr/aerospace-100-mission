# centroidnet — Validation Evidence (Level 2, Research)

All numbers below were produced by running `validation/run_validation.py` in the
build session on 2026-08-06 (Python 3.11.15, numpy 2.4.4, scipy 1.17.1,
scikit-learn 1.8.0, 2 CPU cores). Raw console output is saved verbatim to
`validation/validation_output.txt`; the two figures referenced below are written
by the same script.

Rerun from `products/P008/`:

```bash
PYTHONPATH=src python validation/run_validation.py
```

The run is deterministic apart from wall-clock timings: every dataset is drawn
from a fixed seed (`seed=100+i` for training, `seed=9000+i` for the held-out test
sets) and the MLP ensemble uses `random_state=0`.

Common configuration: 16×16 px window, Gaussian spot σ = 1.5 px, background
B = 2.0 e⁻/px, read noise R = 3.0 e⁻ RMS, true offsets drawn uniformly in
±2.0 px about the array centre. All errors are **radial** (2-D Euclidean)
unless stated otherwise; "bias" is the norm of the mean signed error vector,
`‖mean(estimate − truth)‖`.

---

## 1. Noise-free centre-of-gravity recovery (analytic known answer)

**What is checked.** With shot noise, read noise and background disabled, the
intensity-weighted centroid of a pixel-integrated Gaussian must return the exact
generating offset. The generator integrates the Gaussian over each pixel with the
error function, so the first moment of the noise-free image is analytically equal
to the true centre for an infinite window; the only error source is truncation of
the Gaussian tails at the window edge.

**Reference.** First-moment (CoG) estimator as described in Thomas et al. 2006,
*MNRAS* **371**, 323 ("Comparison of centroid computation algorithms in a
Shack–Hartmann sensor"); the noise-free CoG of a symmetric PSF fully contained in
the window is unbiased by symmetry.

| True offset (x, y) [px] | CoG estimate [px] | Radial error [px] |
|---|---|---|
| (+0.00, +0.00) | (−0.000000, −0.000000) | 3.216e-16 |
| (+0.50, −0.70) | (+0.499998, −0.699996) | 4.946e-06 |
| (+1.30, +2.00) | (+1.299971, +1.999793) | 2.094e-04 |
| (−2.00, +1.10) | (−1.999793, +1.099984) | 2.080e-04 |
| (+2.00, −2.00) | (+1.999793, −1.999793) | 2.934e-04 |

- **Worst-case noise-free CoG recovery error: 2.934e-04 px** (tolerance 1e-3 px)
  — **PASS**.
- At zero offset the error is **3.216e-16 px**, i.e. floating-point round-off, as
  required by the exact symmetry of a centred spot.
- The error grows monotonically with offset because more of the Gaussian tail
  falls outside the 16×16 window. This is a genuine window-truncation bias of the
  estimator, not a numerical defect: at a 2 px offset with σ = 1.5 px the
  truncated flux pulls the first moment inward by ~2.1e-04 px per axis.

---

## 2. Quad-cell bias curve versus true offset (linear-range limitation)

**What is checked.** The calibrated quad-cell output is swept against the true
offset over d ∈ [−4, +4] px (81 points, noise-free) and compared against the
closed-form quadrant response of a Gaussian spot.

**Reference and analytic model.** For a Gaussian of RMS width σ displaced by d,
the normalized quadrant imbalance is

```
(I_right − I_left) / I_total = erf( d / (σ√2) )
```

so with the calibration `scale = σ√(π/2)` (which sets the small-offset slope to
unity) the ideal response is `d̂ = σ√(π/2) · erf(d/(σ√2))`. Quad-cell position
estimation and its restricted linear range: Tyler & Fried 1982, *JOSA* **72**,
804 ("Image-position error associated with a quadrant detector"); Hardy 1998,
*Adaptive Optics for Astronomical Telescopes*, Oxford Univ. Press, ch. 5.

- Calibration constant used: **scale = σ√(π/2) = 1.8800 px**.
- **Maximum deviation between the simulated quad-cell output and the analytic erf
  response over d ∈ [−4, +4] px: 5.538e-05 px** (tolerance 1e-2 px) — **PASS**.
  The implementation therefore reproduces the published response function to
  better than 1e-4 px.
- **Linearity error versus true offset** (this is the documented limitation, not
  an implementation error):

  | True offset d [px] | d / σ | \|d̂ − d\| [px] | Relative error |
  |---|---|---|---|
  | 0.1 | 0.07 | **0.0001** | 0.1 % |
  | 1.5 | 1.00 | **0.217** | 14 % |
  | 3.0 | 2.00 | **1.206** | 40 % |

- The response **saturates at ±scale = ±1.88 px** as \|d\| → ∞: a quad-cell
  physically cannot report an offset larger than this, regardless of the true
  displacement. Beyond \|d\| ≈ σ the estimate is systematically compressed toward
  zero.
- **Conclusion.** The quad-cell is usable only as a null-seeking / fine-tracking
  sensor within roughly \|d\| ≲ 0.3 σ (≈ 0.45 px here) if better than a few
  percent linearity is required. It is not a wide-range absolute position sensor.
  This matches the linear-range caveat in Tyler & Fried 1982 and is why the
  quad-cell RMS error in §3 plateaus around 0.34 px even at SNR 88 — that residual
  is dominated by nonlinearity over the ±2 px offset distribution, not by noise.

Figure: `validation/quadcell_bias_curve.png` (simulated response, analytic erf
curve and the ideal unbiased line, which the response departs from beyond
\|d\| ≈ σ).

---

## 3. Bias and RMS error versus SNR — baselines against the ML ensemble

**What is checked.** All four estimators are evaluated on the *same* held-out
frames at six signal levels. Training and test sets are disjoint by construction
(different seeds: training `100+i`, test `9000+i`); the model never sees a test
frame.

- Training set: 6 signal levels × 700 frames = **4200 frames**, seeds 100–105.
- Model: **5 × MLPRegressor(hidden_layer_sizes=(64,))**, Adam, `max_iter=300`,
  `early_stopping=True`, `random_state=0..4`.
- Training time this run: **24.6 s** on 2 CPU cores (budget 120 s) — **PASS**.
  (An earlier run of the identical script on a more loaded machine took 71.6 s;
  both are within budget. Timing is the only non-deterministic output.)
- Test set: **500 frames per SNR point**, seeds 9000–9005.
- Thresholded CoG uses threshold = B + R = 5.0 e⁻. Frames where the threshold
  removes all flux fall back to plain CoG; **no fallbacks occurred** at any tested
  signal level in this run (a threshold of B + 3R would have zeroed entire frames
  at S = 100 e⁻, which is why B + R was chosen).

Detection SNR is `S / sqrt(S + N_pix(B + R²))` over the full 16×16 window
(standard CCD aperture-photometry SNR, Howell 2006, *Handbook of CCD Astronomy*,
2nd ed., Cambridge Univ. Press).

### RMS radial error [px]

| S [e⁻] | SNR | CoG (plain) | CoG (thresholded) | quad-cell | **ML ensemble** |
|---|---|---|---|---|---|
| 100 | 1.9 | 1.466 | 1.382 | 1.447 | **0.788** |
| 200 | 3.6 | 1.302 | 0.901 | 1.319 | **0.438** |
| 500 | 8.7 | 0.968 | 0.401 | 1.040 | **0.242** |
| 1000 | 16.2 | 0.656 | 0.208 | 0.771 | **0.153** |
| 3000 | 39.3 | 0.305 | **0.075** | 0.493 | 0.079 |
| 10000 | 88.3 | 0.107 | **0.030** | 0.340 | 0.066 |

### Bias ‖mean(estimate − truth)‖ [px]

| S [e⁻] | SNR | CoG (plain) | CoG (thresholded) | quad-cell | ML ensemble |
|---|---|---|---|---|---|
| 100 | 1.9 | 0.050 | 0.078 | 0.054 | 0.066 |
| 200 | 3.6 | 0.021 | 0.030 | 0.018 | 0.014 |
| 500 | 8.7 | 0.033 | 0.010 | 0.037 | 0.006 |
| 1000 | 16.2 | 0.063 | 0.022 | 0.076 | 0.015 |
| 3000 | 39.3 | 0.014 | 0.003 | 0.018 | 0.007 |
| 10000 | 88.3 | 0.008 | 0.002 | 0.027 | 0.006 |

All estimators are close to unbiased over a symmetric ±2 px offset distribution
(≤ 0.08 px everywhere): the *mean* error largely cancels because the quad-cell
compression and the CoG truncation pull symmetrically toward the array centre.
Bias is therefore not the discriminating metric here — RMS error is. The
quad-cell's residual 0.027 px bias at the highest SNR is the only bias that does
not fall with SNR, consistent with it being a deterministic nonlinearity rather
than noise.

Figure: `validation/ml_vs_baseline_snr.png` (log–log RMS error vs SNR, four
estimators). The example script `examples/error_vs_snr.py` reproduces the same
comparison independently with different seeds and 300 test frames per point, and
overlays the ensemble spread band; see `screenshots/error_vs_snr.png`.

### Honest reading of the comparison

1. **The ML ensemble beats plain CoG at every tested SNR**, by 1.6× to 3.0× in
   RMS. Plain CoG is dominated by background and read noise summed over all 256
   pixels, exactly the degradation reported in Thomas et al. 2006.

2. **The ML ensemble beats the thresholded CoG only below SNR ≈ 40.** At
   SNR 1.9 the ML error is 0.788 px versus 1.382 px (1.8× better); at SNR 8.7 it
   is 0.242 px versus 0.401 px (1.7× better). At **SNR 39.3 the thresholded CoG
   overtakes it** (0.075 px vs 0.079 px), and at SNR 88.3 the thresholded CoG is
   **2.2× better** (0.030 px vs 0.066 px). *The ML model does not beat the best
   analytic centroid at high SNR.*

3. **Why the ML model loses at high SNR.** Its error saturates at a floor of
   ≈ 0.066 px that does not improve with signal, while the thresholded CoG keeps
   improving as 1/SNR. The floor is a property of the estimator, not the data:
   the network is a finite-capacity regressor trained on a finite sample (4200
   frames), fitted with `early_stopping` and L2 regularization, so it carries an
   irreducible approximation/estimation error. Because the training set mixes all
   six signal levels, the learned mapping is a compromise across noise regimes and
   is mildly shrunk toward the mean of the offset distribution — a regression-to-
   the-mean effect that costs nothing at low SNR (where noise dominates) but
   dominates once the noise floor drops below it. Nothing in the architecture
   allows the ML error to fall below this floor no matter how bright the spot is.

4. **Practical conclusion.** For this sensor model the defensible operating rule
   is: use the ML ensemble below SNR ≈ 40, and use the thresholded CoG above it.
   The ML model is not a universal replacement for the analytic estimator; it buys
   accuracy specifically in the photon-starved regime where the analytic first
   moment is corrupted by background and read noise.

5. **Anchor against the photon-noise limit.** For a well-sampled Gaussian PSF the
   shot-noise-limited centroid standard deviation is σ_PSF/√N per axis (Winick
   1986, *JOSA A* **3**, 1809; Thomas et al. 2006). At S = 10⁴ e⁻ and σ = 1.5 px
   this is 1.5/100 = 0.0150 px per axis, i.e. √2 × 0.0150 = **0.0212 px radial**
   (hand-calculated analytic reference, not a measurement). The measured
   thresholded CoG achieves 0.030 px — within **1.4×** of the photon limit, which
   is the expected penalty for the residual background/read-noise contribution and
   the threshold bias. The ML ensemble at 0.066 px sits **3.1×** above the limit.
   This independently confirms that the high-SNR gap in point 2 is real and that
   the thresholded CoG, not the ML model, is the near-optimal estimator there.

---

## 4. Uncertainty output — measured, and NOT calibrated

The model exposes `predict(..., return_std=True)`, the per-estimate standard
deviation across the 5 ensemble members (deep-ensemble spread in the sense of
Lakshminarayanan, Pritzel & Blundell, "Simple and scalable predictive uncertainty
estimation using deep ensembles", NeurIPS 2017). Measured against the actual
held-out error:

| SNR | mean ensemble std [px] | actual RMS error [px] | std / RMS |
|---|---|---|---|
| 1.9 | 0.073 | 0.788 | **0.09** |
| 3.6 | 0.048 | 0.438 | **0.11** |
| 8.7 | 0.034 | 0.242 | **0.14** |
| 16.2 | 0.029 | 0.153 | **0.19** |
| 39.3 | 0.027 | 0.079 | **0.34** |
| 88.3 | 0.029 | 0.066 | **0.44** |

**Finding (reported as measured, not tuned):** the ensemble spread
**under-estimates the true error at every SNR**, by a factor of 2.3× at the best
point and 11× at the worst. It must **not** be read as a 1-σ error bar. The
members differ only in weight initialization and mini-batch shuffling, so the
spread captures initialization variance alone — it contains no term for shot
noise, read noise, or the shared systematic error of the common architecture and
training set. Its one useful property is monotonicity: the spread does rise as
SNR falls (0.027 → 0.073 px), so it carries a qualitative signal of degrading
conditions, but the ratio to true error varies by a factor of 5 across the range
and no fixed scaling would calibrate it. Any operational use would require
recalibration against held-out error (e.g. isotonic or variance scaling), which
is **not** performed in 0.1.0.

---

## Summary

| # | Check | Reference | Result | Tolerance | Verdict |
|---|---|---|---|---|---|
| 1 | Noise-free CoG recovery | Thomas et al. 2006 | worst 2.934e-04 px | 1e-3 px | PASS |
| 2 | Quad-cell vs analytic erf response | Tyler & Fried 1982; Hardy 1998 | max dev 5.538e-05 px | 1e-2 px | PASS |
| 2b | Quad-cell linear range | Tyler & Fried 1982 | 14 % error at d = σ, 40 % at d = 2σ, saturates at ±1.88 px | — | limitation confirmed and documented |
| 3 | ML vs baselines, RMS vs SNR | own held-out data | ML better below SNR ≈ 40 (1.8× at SNR 1.9), worse above (2.2× at SNR 88) | — | reported as measured; ML does **not** win at high SNR |
| 3b | Training compute | build-guide budget | 24.6 s / 4200 frames, 2 cores | < 120 s | PASS |
| 4 | Uncertainty calibration | Lakshminarayanan et al. 2017 | std/RMS 0.09–0.44 | — | **not calibrated**; documented limitation |

Level 2 (Research) scope: every check above compares against an analytic/published
reference or a held-out measurement. No comparison against real detector data or
an independent flight-heritage implementation was performed — that would be
required for Level 3 and is out of scope for 0.1.0.
