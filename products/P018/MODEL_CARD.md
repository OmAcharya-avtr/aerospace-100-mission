# Model Card — shacksim `MLSlopeEstimator` 0.1.0

**Model:** ensemble of 5 scikit-learn `MLPRegressor` networks mapping one
16 × 16 Shack-Hartmann subaperture stamp to a wavefront slope `(g_x, g_y)` in
radians, with a per-slope ensemble-spread confidence output.
**Status:** TESTING · **Validation level:** 2 (Research) · **License:** Apache-2.0
**Product:** P018 ShackSim, OPTIMA aerospace portfolio.

> **This model is not certified for operational flight use.**

> ## ⚠ Read this before using the model
>
> On held-out synthetic data the learned estimator beats the tuned thresholded
> centre of gravity **only below ≈ 100 detected photoelectrons per subaperture**
> (round spots: crossover between 50 and 100 e⁻; 3× elongated spots: between
> 100 and 300 e⁻). The best measured advantage is **1.38×**. Above the crossover
> the analytic estimator is better and the gap widens without limit — **10.5×
> better at 10 000 e⁻**. Against the *correlation* baseline the learned model
> loses at 6 of 7 flux levels for round spots. There is no regime in which this
> model should replace the classical estimators wholesale.

---

## 1. Problem

Estimate the local wavefront gradient (`slope`) behind one lenslet of a
Shack-Hartmann wavefront sensor from that lenslet's detector stamp. The slope
vector over the whole array is the sensor's output and the input to every
downstream wavefront reconstructor; slope error propagates directly into
reconstruction error and, in a closed adaptive-optics loop, into residual
wavefront variance and Strehl loss.

- **Input:** one `pixels_per_sub × pixels_per_sub` stamp (16 × 16 by default) in
  photoelectrons. Values may be negative (read noise is not clipped).
- **Output:** `(g_x, g_y)` in radians, plus an optional per-component ensemble
  standard deviation in radians.

The target regime is where the classical centre of gravity degrades worst:
**low photon flux** (the read-noise term of the noise-propagation expression
grows with the window area) and **elongated spots** (a thresholded first moment
of an anisotropic spot is biased along the elongation axis).

**Scope boundary.** This is the *sensor-array* companion to product **P008
CentroidNet**, which addresses single-spot subpixel centroiding on one detector
window. P008's results are not reproduced or re-derived here. What is different
in P018: the lenslet-array geometry and pupil mask, the pixel-to-slope
conversion `g = Δx·p/f`, the per-subaperture noise applied across the whole
frame, and a slope-vector output rather than a single centroid.

## 2. Baseline — implemented first, benchmarked on identical held-out stamps

Per the mission rule that the classical estimator comes first, both baselines
live in `src/shacksim/slopes.py` and were validated (VALIDATION.md §1–§4)
before this model was written. Every ML number below is measured on the *same*
stamps as the baselines.

| Baseline | Definition | Source |
|---|---|---|
| Thresholded centre of gravity | `x̂ = Σ w_i x_i / Σ w_i`, `w = max(I − t, 0)` | Thomas et al. 2006, *MNRAS* **371**, 323; Hardy 1998 ch. 5 |
| Correlation | peak of the cross-correlation with a reference spot, refined by 3-point parabolic interpolation | Poyneer 2003, *Appl. Opt.* **42**, 5807 |
| Analytic noise model | `Var(x̂) = M2/N + (B+R²)/N²·[p²(p²−1)/12 + p²d²]` | derived in `cog_noise_sigma`; standard form in Hardy 1998 ch. 5, Thomas et al. 2006, Winick 1986 *JOSA A* **3**, 1809 |

The CoG **threshold was tuned per flux level on a separate tuning dataset**
(seeds `300 + N + 11·elongation`), so the baseline is presented at its best
available configuration and not handicapped.

## 3. Architecture

- **Preprocessing** (`MLSlopeEstimator.features`), per stamp:
  1. clip negative pixels to 0, flatten, divide by the total → a 256-vector of
     unit sum (removes overall detector gain / exposure);
  2. append `log10(1 + total counts)` — one scalar telling the network which
     noise regime it is in.
  Feature 2 means the model is **not gain-invariant** and assumes its input is
  in photoelectrons on the same scale as the training data. Feeding ADU, or a
  detector with a different conversion gain, invalidates it. Stamps whose
  clipped total is zero produce an all-zero feature vector.
- **Members:** 5 × `MLPRegressor(hidden_layer_sizes=(96, 48))`, ReLU, Adam,
  `alpha=1e-4`, `max_iter=400`, `early_stopping=True`, `n_iter_no_change=15`,
  `validation_fraction=0.1`, seeds `random_state + k` for `k = 0…4`.
  ≈ 25 k parameters per member, ≈ 125 k total.
- **Targets:** spot displacement in pixels (O(1), well conditioned); converted
  to and from slopes with `LensletArray.slope_to_displacement`.
- **Prediction:** member mean; `return_std=True` adds the member standard
  deviation (Lakshminarayanan, Pritzel & Blundell, NeurIPS 2017, deep
  ensembles).

**Deviation from what one would ideally build.** A convolutional network is the
natural architecture for a translation-estimation task. PyTorch is not
available in this build environment and scikit-learn has no convolutional
layers, so a fully connected ensemble is used instead. It has no weight
sharing and no translation equivariance; the high-flux error floor reported
below is plausibly an artefact of that substitution and should **not** be read
as evidence about CNN slope estimation.

## 4. Dataset

Full card: [`DATASET_CARD.md`](DATASET_CARD.md). Summary:

- **100 % synthetic**, generated by `shacksim.sensor.generate_subaperture_dataset`
  from an idealized optical model. No real detector, laboratory or flight data.
- **Not committed** to the repository (regenerable in < 1 s); generation is
  bit-for-bit deterministic from a fixed integer seed.
- Labels are **exact by construction** — the slope the stamp was drawn from. No
  label noise, which real data never provides.
- Unmodelled effects that would degrade real performance are listed in the
  dataset card and in the README Limitations.

## 5. Training procedure

```bash
cd products/P018
PYTHONPATH=src python validation/run_validation.py    # trains and benchmarks
```

| Item | Value |
|---|---|
| Training stamps | 9000 |
| Photon range | log-uniform, 30 – 30 000 e⁻ per subaperture |
| Elongation range | uniform, 1.0 – 3.0×, along x |
| Background / read noise | 1.0 e⁻/px / 3.0 e⁻ RMS |
| Slope range | uniform over ±0.6 × field = ±3.0 mrad (±4.8 px) |
| Training seed | 100 |
| Internal validation | 10 % of training stamps, held by `early_stopping` |
| Hyperparameter search | **none** — no test information entered any choice |
| Compute | 39.9 s wall clock, 2 CPU cores (108.9 s on a loaded repeat; results bit-identical) |

## 6. Test-split strategy

The split is by **disjoint RNG stream**, not by partitioning one dataset:

| Set | Seeds |
|---|---|
| Training | 100 |
| CoG threshold tuning | `300 + N + 11·elongation` |
| Held-out test | `9000 + N + 7·elongation` |

Train and test are drawn from the same generative process, so this measures
**in-distribution generalization only**. It is not a robustness or
domain-transfer test, and it says nothing about a real sensor.

## 7. Metrics

RMS slope error, quoted in pixels of spot displacement (1 px = 625 µrad), on
2000 held-out stamps per point. Full tables in
[`validation/VALIDATION.md`](validation/VALIDATION.md) §5.

**Round diffraction-limited spots:**

| N [e⁻] | thresholded CoG | correlation | **ML ensemble** | ML/CoG |
|---|---|---|---|---|
| 30 | 2.5705 | 2.8780 | **2.2600** | 0.879 |
| 50 | 1.9161 | **0.9630** | 1.6591 | 0.866 |
| 100 | 0.3852 | **0.1944** | 0.7842 | 2.036 |
| 300 | 0.0891 | **0.0842** | 0.3118 | 3.501 |
| 1000 | **0.0367** | 0.0446 | 0.1382 | 3.766 |
| 3000 | **0.0193** | 0.0301 | 0.1160 | 6.010 |
| 10000 | **0.0098** | 0.0235 | 0.1027 | 10.494 |

**3× elongated spots:**

| N [e⁻] | thresholded CoG | correlation | **ML ensemble** | ML/CoG |
|---|---|---|---|---|
| 30 | 2.6674 | 4.0164 | **2.5594** | 0.960 |
| 50 | 2.5064 | 3.0531 | **2.1165** | 0.844 |
| 100 | 1.6310 | 1.0323 | **1.1840** | 0.726 |
| 300 | **0.3053** | 0.4541 | 0.4344 | 1.423 |
| 1000 | **0.1040** | 0.1999 | 0.2304 | 2.215 |
| 3000 | **0.0622** | 0.1133 | 0.1706 | 2.742 |
| 10000 | 0.0970 | **0.0662** | 0.1756 | 1.810 |

**Crossover, reported where it falls:** 50–100 e⁻ for round spots, 100–300 e⁻
for 3× elongated spots. Best measured advantage 1.38× (elongated, 100 e⁻).
Worst deficit 10.5× (round, 10 000 e⁻).

## 8. Uncertainty / confidence output

`predict(stamps, return_std=True)` returns the per-component standard deviation
across the 5 members, in radians.

| Regime | measured spread / actual RMS error |
|---|---|
| Round spots, 30 → 10 000 e⁻ | 0.17 → 1.12 |
| Elongated spots, 30 → 10 000 e⁻ | 0.15 → 0.55 |

**This is not a calibrated 1-σ error bar.** In the low-flux regime — precisely
where the model is meant to be used — it under-states the true error by up to
**6.5×**, because all five members share the same systematic shrinkage toward
the mean of the slope distribution and ensembles cannot see their own common
bias. At high flux on round spots it over-states the error. It is usable as a
monotonic, qualitative "photon-starved" indicator; it must **not** be used as a
measurement covariance in a wavefront reconstructor, Kalman filter or
closed-loop controller.

## 9. Failure cases

1. **Silent accuracy loss above ≈ 100 e⁻/subaperture.** No exception, no flag —
   the confidence output does not rise. Choose the estimator by known flux.
2. **Extrapolation outside the training envelope.** Trained only for 30–30 000 e⁻,
   elongation 1–3× along **x**, slopes within ±0.6 of the field, B = 1 e⁻/px,
   R = 3 e⁻ RMS, 16 × 16 stamps, σ_spot = 0.885 px. A pinned regression test
   (`tests/test_ml.py::test_classical_wins_far_outside_the_training_flux`)
   demonstrates a > 10× deficit when a model trained on 30–300 e⁻ is used at
   3000 e⁻.
3. **Elongation at an arbitrary angle is not representable** — only axis-aligned
   elongation is modelled, so the sodium-laser-guide-star geometry (radial
   elongation with a position-dependent angle) is *outside* both the training
   data and the simulator.
4. **Wrong photometric units.** The `log10(1+counts)` feature makes the model
   gain-sensitive; ADU inputs silently move it to the wrong point on the
   flux axis.
5. **Geometry change.** A different `pixels_per_sub` raises `ValueError`; a
   different pitch, focal length or wavelength does **not** raise, but changes
   the spot size and invalidates the model silently.
6. **Degenerate stamps** (all pixels ≤ 0) produce an all-zero feature vector and
   an arbitrary learned response, not an error.
7. **Unmodelled detector physics** — dead/hot pixels, PRNU/DSNU, cosmic rays,
   saturation, charge diffusion, inter-subaperture crosstalk — is absent from
   training and its effect is unmeasured.

Only input-validation errors raise. **Every accuracy failure mode is silent.**

## 10. Reproducibility

From `products/P018/`:

```bash
PYTHONPATH=src python validation/run_validation.py       # full benchmark, ~56 s
PYTHONPATH=src python examples/slope_error_vs_flux.py    # reduced run, ~25 s
python -m pytest tests/ -q                               # 148 tests, ~11 s
```

Seeds: training 100, tuning `300 + N + 11·elongation`, test
`9000 + N + 7·elongation`, `random_state=0`. All data is regenerated from
`numpy.random.default_rng`, so results are bit-identical on the same
numpy/scikit-learn/BLAS build. Reported floats may move in the last digits with
a different BLAS or scikit-learn version; the qualitative conclusions
(crossover location within a factor of ~2, high-flux floor, uncalibrated
uncertainty) are robust.

## 11. Compute used

CPU only, 2 x86-64 cores. Training 39.9 s (108.9 s under load); inference
≈ 0.05 ms per stamp per member; peak memory < 300 MB (9000 × 16 × 16 float64 ≈
18 MB of image data plus features). No GPU, no accelerator, no network access.
Within the mission's 3-minute training budget with margin.

## 12. Ethical and safety limits

- Research and education only. **This model is not certified for operational
  flight use.** No DO-178C, ECSS-E-ST-40C or equivalent process was followed;
  there is no independent verification and no qualification testing.
- It must not be placed in a pointing, tracking, guidance or adaptive-optics
  control loop. Its accuracy failure modes are silent and its confidence output
  is not calibrated, so a downstream consumer cannot detect or bound its
  errors — and a wavefront-sensor gain error feeds straight back into loop gain.
- All performance figures characterize the model against an **idealized
  simulator**, not against hardware. They are an optimistic bound: every
  unmodelled effect can only degrade them.
- No personal data, no human subjects, no dual-use content is involved. The
  underlying physics is standard, published optics.
