# Model Card — centroidnet `MLCentroider` 0.1.0

**Model:** ensemble of 5 scikit-learn `MLPRegressor` networks mapping a
flux-normalized 16×16 pixel vector to a subpixel centroid (x, y) in pixels.
**Status:** TESTING · **Validation level:** 2 (Research) · **License:** Apache-2.0

> ## ⚠ Deviation from the original specification
>
> **This product was specified with a small convolutional neural network (CNN).
> It is implemented instead as an ensemble of fully connected multi-layer
> perceptrons (`sklearn.neural_network.MLPRegressor`), because PyTorch is not
> available in this build environment** (see `templates/PRODUCT_BUILD_GUIDE.md`,
> "Environment": ML must use scikit-learn or numpy). scikit-learn provides no
> convolutional layers, so the spatial weight sharing, translation-equivariance
> and parameter efficiency of a CNN are **absent** from this model. The network
> sees the image as a flat 256-vector and must learn spatial structure from
> scratch, which is a materially weaker inductive bias for a translation-
> estimation task. The accuracy reported below — in particular the ≈ 0.066 px
> high-SNR error floor that a CNN would plausibly not exhibit — should be read as
> a property of *this substitute architecture*, not as evidence about CNN
> centroiding. Re-implementing as a CNN is the first roadmap item.

> **This model is not certified for operational flight use.**

---

## 1. Problem

Estimate the subpixel position of a single unresolved optical spot on a small
detector window, as required by star trackers, fine-guidance sensors, laser
communication acquisition/tracking sensors and Shack–Hartmann wavefront sensors.
Pointing error budgets are driven directly by centroid error, and the hardest
regime is low signal-to-noise: a faint star or a weak beacon where background and
read noise corrupt the classical first moment.

**Input:** one 16×16 detector frame in linear intensity units (photoelectrons).
**Output:** (x, y) centroid in pixels relative to the geometric array centre
`(N−1)/2`, plus a per-estimate ensemble spread.

## 2. Baseline (implemented first, benchmarked on identical held-out data)

Per the mission rule that the classical estimator comes first, three analytic
baselines are implemented in `src/centroidnet/baselines.py` and every ML claim is
made against them on the *same* frames:

| Baseline | Definition | Source |
|---|---|---|
| Centre of gravity (plain) | `x̂ = Σ wᵢxᵢ / Σ wᵢ`, `w = max(I, 0)` | Thomas et al. 2006, *MNRAS* **371**, 323 |
| Centre of gravity (thresholded) | as above with `w = max(I − t, 0)`, `t = B + R = 5 e⁻` | Thomas et al. 2006 |
| Quad-cell (calibrated) | `x̂ = s·(I_R − I_L)/I_tot`, `s = σ√(π/2)` | Tyler & Fried 1982, *JOSA* **72**, 804; Hardy 1998 ch. 5 |

The thresholded CoG is the strong baseline and it **wins at high SNR** — see §7.

## 3. Architecture

- **Preprocessing** (in `MLCentroider._features`): clip negative pixels to 0,
  flatten to a 256-vector, divide by the image sum → unit total flux. This makes
  the model invariant to overall detector gain and exposure. Frames with
  non-positive total intensity are rejected with `ValueError`.
- **Members:** 5 × `MLPRegressor`, one hidden layer of 64 ReLU units, 2 linear
  outputs (x, y). Roughly 256×64 + 64×2 ≈ 16.5 k parameters per member,
  ≈ 82 k total.
- **Solver:** Adam, `alpha = 1e-4` (L2), `max_iter = 300`, `early_stopping=True`
  with `validation_fraction = 0.1` and `n_iter_no_change = 15`.
- **Ensembling:** member *k* uses `random_state = random_state + k`, so members
  differ only in weight initialization and mini-batch shuffling. Prediction is the
  member mean; the spread is the member standard deviation.
- **No convolution, no pooling, no data augmentation.** See the deviation box.

## 4. Dataset — source and limitations

Full detail in **`DATASET_CARD.md`**. Summary:

- **Entirely synthetic.** Generated on the fly by
  `src/centroidnet/generator.py` from an idealized sensor model: a pixel-
  integrated 2-D Gaussian spot (erf model), Poisson shot noise, uniform
  background, additive Gaussian read noise. **No real detector data was used at
  any stage.** Nothing is committed to the repository; the generator is
  deterministic under a fixed seed, so datasets regenerate bit-for-bit.
- **Training distribution:** 16×16 window, σ = 1.5 px, B = 2.0 e⁻/px,
  R = 3.0 e⁻ RMS, true offsets uniform in ±2.0 px, six signal levels
  S ∈ {100, 200, 500, 1000, 3000, 10000} e⁻ × 700 frames = **4200 frames**
  (seeds 100–105).
- **Key limitation:** the model has only ever seen this idealized model. Effects
  *not* simulated — dead/hot pixels, pixel-response nonuniformity (PRNU/DSNU),
  optical aberrations beyond a Gaussian core, stray light and background
  gradients, detector nonlinearity and saturation, cosmic rays, multiple or
  extended sources, spacecraft jitter/smear, temporal drift — are all
  out-of-distribution and are **not** covered by any measurement in this product.
- **Fixed spot size.** σ = 1.5 px is baked into training. Defocus, wavelength
  change or a different optical design changes σ and invalidates the model
  without retraining.

## 5. Training procedure

```bash
cd products/P008
PYTHONPATH=src python validation/run_validation.py
```

which performs exactly:

1. For `i, S` in `enumerate([100, 200, 500, 1000, 3000, 10000])`:
   `generate_spots(700, 16, 1.5, S, 2.0, 3.0, seed=100+i, offset_range=2.0)`.
2. Concatenate → X (4200, 16, 16), Y (4200, 2).
3. `MLCentroider(n_estimators=5, hidden_layer_sizes=(64,), max_iter=300,
   random_state=0).fit(X, Y)`.

Each member internally holds out 10 % of the training frames for early stopping;
that internal split is *not* the evaluation split (§6). No hyperparameter search
was performed — the architecture was fixed a priori to fit the 3-minute compute
budget, so no test-set information influenced any hyperparameter choice.

## 6. Test-split strategy

**Disjoint by seed, not by partition.** Because data is generated rather than
collected, train and test sets are drawn from independent RNG streams:

| Split | Seeds | Size | Used for |
|---|---|---|---|
| Train | 100–105 | 700 × 6 = 4200 | fitting the ensemble |
| Early-stopping (internal) | sklearn `validation_fraction=0.1` | 420 (of the 4200) | stopping criterion only |
| **Held-out test** | **9000–9005** | **500 × 6 = 3000** | **all reported metrics** |
| Independent example run | 8800–8805 (train 300–305) | 300 × 6 | `examples/error_vs_snr.py` reproduction |
| Unit-test split | train 2024, test 777 | 1500 / 300 | `tests/test_ml.py` |

No test frame is ever seen during training. The held-out sets are drawn from the
*same* distribution as training — this is an in-distribution generalization test
only, **not** a test of robustness to distribution shift, which was not performed.

## 7. Metrics (measured, from `validation/validation_output.txt`)

RMS radial error [px] on 500 held-out frames per SNR point:

| S [e⁻] | SNR | CoG plain | CoG thresholded | quad-cell | **ML ensemble** |
|---|---|---|---|---|---|
| 100 | 1.9 | 1.466 | 1.382 | 1.447 | **0.788** |
| 200 | 3.6 | 1.302 | 0.901 | 1.319 | **0.438** |
| 500 | 8.7 | 0.968 | 0.401 | 1.040 | **0.242** |
| 1000 | 16.2 | 0.656 | 0.208 | 0.771 | **0.153** |
| 3000 | 39.3 | 0.305 | **0.075** | 0.493 | 0.079 |
| 10000 | 88.3 | 0.107 | **0.030** | 0.340 | 0.066 |

Bias ‖mean(est − truth)‖ ≤ 0.078 px for every estimator at every SNR (full table
in `validation/VALIDATION.md` §3); all four estimators are effectively unbiased
over the symmetric ±2 px offset distribution, so RMS is the discriminating metric.

**Headline result, stated plainly:**

- The ML ensemble beats **plain** CoG at **every** tested SNR (1.6×–3.0×).
- The ML ensemble beats the **thresholded** CoG only **below SNR ≈ 40**
  (1.8× better at SNR 1.9, 1.7× at SNR 8.7).
- **At SNR ≥ 39 the thresholded CoG is equal or better, and at SNR 88 it is 2.2×
  better** (0.030 px vs 0.066 px). The ML error hits a floor near 0.066 px that
  does not improve with more signal, because the finite-capacity, regularized,
  early-stopped network trained on 4200 frames across mixed noise regimes carries
  an irreducible approximation error and shrinks slightly toward the mean of the
  offset distribution. For reference the photon-noise limit σ/√N is 0.021 px
  radial at S = 10⁴ e⁻ (Winick 1986, *JOSA A* **3**, 1809); the thresholded CoG
  reaches 1.4× that limit, the ML model 3.1×.
- **Recommendation:** use the ML model below SNR ≈ 40, the thresholded CoG above.
  It is not a universal replacement for the analytic estimator.

## 8. Uncertainty output

`MLCentroider.predict(images, return_std=True)` returns `(mean, std)` where `std`
is the per-axis standard deviation across the 5 members (deep-ensemble spread;
Lakshminarayanan, Pritzel & Blundell, NeurIPS 2017).

| SNR | mean std [px] | actual RMS [px] | std / RMS |
|---|---|---|---|
| 1.9 | 0.073 | 0.788 | 0.09 |
| 8.7 | 0.034 | 0.242 | 0.14 |
| 88.3 | 0.029 | 0.066 | 0.44 |

**The spread is NOT a calibrated 1-σ error bar and must not be used as one.** It
under-estimates the true error by 2.3×–11× across the tested range. Members share
architecture, training data and preprocessing and differ only in initialization,
so the spread measures initialization variance and excludes shot noise, read
noise and shared systematic error. It is monotonic in SNR (0.027 → 0.073 px as
SNR falls), so it is usable as a *qualitative* degradation flag or for relative
ranking of frames, nothing more. No calibration (variance scaling, isotonic
regression, conformal prediction) is applied in 0.1.0.

## 9. Failure cases

| Case | Behaviour | Detected? |
|---|---|---|
| SNR > ~40 | Silently worse than a thresholded CoG (error floor ≈ 0.066 px) | No — spread does not flag it |
| Offset outside ±2 px training range | Extrapolation; unquantified, expect strong shrinkage toward centre | No |
| Spot width σ ≠ 1.5 px (defocus, different optics) | Out of distribution; unquantified | No |
| Dead/hot pixel, cosmic ray | Out of distribution; a hot pixel can dominate the flux-normalized input | No |
| Two spots / extended source in the window | Undefined; model assumes exactly one spot | No |
| Saturated / clipped spot core | Out of distribution (generator has no saturation model) | No |
| Frame with total intensity ≤ 0 after clipping | Raises `ValueError` | **Yes** |
| Wrong image shape, NaN/Inf pixels | Raises `ValueError` | **Yes** |
| `predict()` before `fit()` | Raises `RuntimeError` | **Yes** |

Only the input-validation failures are detected. **Every accuracy-related failure
mode above is silent** — the model returns a confident-looking number with an
uninformative spread. This is the single most important safety property of the
model to understand before use.

## 10. Reproducibility

Deterministic under fixed seeds; the only run-to-run variation is wall-clock
timing. From `products/P008/`:

```bash
# Full validation: all three checks, both figures, raw log
PYTHONPATH=src python validation/run_validation.py
#   -> validation/validation_output.txt
#      validation/quadcell_bias_curve.png
#      validation/ml_vs_baseline_snr.png

# Examples (regenerate screenshots/)
PYTHONPATH=src python examples/error_vs_snr.py     # -> screenshots/error_vs_snr.png
PYTHONPATH=src python examples/spot_gallery.py     # -> screenshots/spot_gallery.png

# Tests
python -m pytest tests/ -q
```

Seeds: training `100+i` (i = 0..5), held-out test `9000+i`, model
`random_state = 0` (members 0–4). Example script: train `300+i`, test `8800+i`.
Gallery: train `500+i`, frames `7700+col`. Unit tests: train 2024, test 777.

Environment used for the reported numbers: Python 3.11.15, numpy 2.4.4,
scipy 1.17.1, scikit-learn 1.8.0, matplotlib 3.10.9, Linux x86-64, 2 CPU cores.
Exact float values may shift in the last digits with a different BLAS or
scikit-learn version; the qualitative conclusions (crossover near SNR 40, ML
error floor, uncalibrated spread) are robust.

## 11. Compute used

| Item | Cost |
|---|---|
| Training (5 × MLP(64,), 4200 frames, 2 CPU cores) | **24.6 s** measured this run (71.6 s on a more loaded run) — budget 120 s |
| Full validation script (training + 3000 test frames + 2 figures) | ≈ 40–90 s |
| `examples/error_vs_snr.py` | 18 s |
| `examples/spot_gallery.py` | 15 s |
| Full test suite | 11 s |
| Inference | ≈ 0.1 ms per frame per member, CPU only |
| Model size | ≈ 82 k parameters total (5 members) |

No GPU is used or required. Total training energy is negligible; there is no
large-scale pretraining and no external dataset download.

## 12. Ethical and safety limits

- **Not certified for operational flight use.** Repeated for emphasis: *this
  model is not certified for operational flight use.*
- Research-grade software. Not flight-qualified, not certified, not approved for
  operational aerospace use. No DO-178C / ECSS-E-ST-40C process, no independent
  verification, no qualification testing, no configuration management beyond this
  repository.
- **Do not place this model in any pointing, tracking or guidance control loop.**
  Its failure modes are silent (§9) and its uncertainty output is not calibrated
  (§8), so a downstream estimator cannot detect or bound its errors. A quantified
  analytic estimator with a known error model is the correct choice for any loop
  where safety, mission success or hardware integrity depends on the result.
- Trained only on synthetic idealized data (§4). Behaviour on real detector data
  is **unknown and unmeasured**. Any transfer to hardware requires retraining or
  fine-tuning on that hardware's data plus independent validation.
- No personal data, no human subjects, no dual-use concern beyond the generic
  dual-use nature of optical tracking. The model is a small regressor on
  synthetic images and carries no demographic bias dimension; the meaningful
  "bias" here is the estimator bias quantified in §7.
- Users must not present ML output as flight-qualified performance, and must not
  extrapolate the numbers in §7 beyond the stated operating envelope (16×16 px
  window, σ = 1.5 px, offsets ≤ 2 px, B = 2 e⁻/px, R = 3 e⁻).

## 13. References

- K. A. Winick, "Cramér–Rao lower bounds on the performance of charge-coupled-
  device optical position estimators", *J. Opt. Soc. Am. A* **3**, 1809–1815 (1986).
- S. Thomas, T. Fusco, A. Tokovinin, M. Nicolle, V. Michau, G. Rousset,
  "Comparison of centroid computation algorithms in a Shack–Hartmann sensor",
  *Mon. Not. R. Astron. Soc.* **371**, 323–336 (2006).
- G. A. Tyler, D. L. Fried, "Image-position error associated with a quadrant
  detector", *J. Opt. Soc. Am.* **72**, 804–808 (1982).
- J. W. Hardy, *Adaptive Optics for Astronomical Telescopes*, Oxford Univ. Press
  (1998), ch. 5.
- S. B. Howell, *Handbook of CCD Astronomy*, 2nd ed., Cambridge Univ. Press (2006).
- B. Lakshminarayanan, A. Pritzel, C. Blundell, "Simple and scalable predictive
  uncertainty estimation using deep ensembles", *NeurIPS* (2017).
