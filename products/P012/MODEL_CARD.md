# Model Card — NavBench Learned Adaptive Process-Noise Tuner

**Model name:** `navbench.adaptive.LearnedAdaptiveQ`
**Version:** 0.1.0 · **Date:** 2026-08-29 · **Status:** TESTING
**Owner:** OPTIMA Organisation · **License:** AGPL-3.0-only

> **This model is not certified for operational flight use.**

> **Headline result, stated up front.** On the held-out benchmark the learned
> tuner gives the **lowest position RMSE** (1.923 m vs 2.100 m for a fixed
> hand-tuned `Q` and 2.274 m for classical Mehra IAE) and is the only method
> whose scale estimate tracks the truth (correlation **+0.59** vs Mehra's
> **0.00**). The classical scheme has the ANEES closest to the ideal 2.0, **but
> only because it saturates at its clip on 60 of 60 runs** — it applies a
> constant 64× inflation rather than adapting. **None of the three methods is
> actually consistent.** The learned tuner's confidence output is measurably
> uninformative (correlation with error **+0.22**, wrong sign). All of this is
> reported as measured; nothing was retuned to improve the appearance of the
> result.

---

## 1. Problem

A Kalman filter's process-noise covariance `Q` encodes how far the truth is
expected to deviate from the assumed dynamics. It is almost never known.

* `Q` too small → the filter becomes over-confident. NEES climbs above its
  chi-squared bound, the estimate lags manoeuvres, and downstream fusion
  over-weights it.
* `Q` too large → the filter throws away information. NEES falls below the
  bound and RMS error rises.

Neither failure is visible in RMS error alone, which is why the scoring here is
two-sided.

**Task.** Regression. Predict `log₁₀ λ` where `Q = λ Q_nominal`, from a sliding
window of the filter's own innovation statistics. Single scalar output plus an
uncertainty estimate.

**Why a single scalar.** Both the classical and the learned method are
restricted to the same one-dimensional knob so the comparison tests *how well
the scale is inferred*, not how many free parameters each method is allowed.

## 2. Baseline (implemented and validated FIRST, per mission §11)

Two baselines, both stronger than a strawman.

**(a) Fixed hand-tuned `Q`.** `λ = 1`, with `q̃_nom` set at the geometric centre
of the test distribution. This is the *best possible single fixed choice* for
the benchmark — no fixed value can do better on average.

**(b) Classical innovation-based adaptive estimation (IAE).**
`navbench.adaptive.MehraAdaptiveQ`, implementing

```
Ĉ = (1/N) Σ_{j ∈ window} ν_j ν_jᵀ          Q̂ = K Ĉ Kᵀ
```

from Mehra, R. K. (1970), "On the identification of variances and adaptive
Kalman filtering", *IEEE Trans. Automatic Control* 15(2), 175-184, and
Mehra, R. K. (1972), "Approaches to adaptive filtering", *IEEE Trans. Automatic
Control* 17(5), 693-698; compact form as in Mohamed, A. H. & Schwarz, K. P.
(1999), "Adaptive Kalman filtering for INS/GPS", *Journal of Geodesy* 73,
193-203, Eq. (12).

The scalar projection `λ = tr(Q̂)/tr(Q_nom)` and the clipping to
[1/64, 64] are **this package's** additions, not Mehra's, and are flagged as
such in the module docstring. Documented weaknesses of IAE, stated because they
bear on the comparison: `Q̂ = K Ĉ Kᵀ` has rank at most `m` (the measurement
dimension) and is biased when the window is short relative to the filter's
settling time; it also has no notion of confidence.

Both baselines were implemented, tested and validated before the learned
component was written.

## 3. Architecture

* 5 × `sklearn.ensemble.GradientBoostingRegressor`, each `n_estimators=150`,
  `max_depth=3`, `learning_rate=0.06`, `subsample=0.85`.
* Each member is fitted on an independent **bootstrap resample** of the
  training set with its own `random_state` (`20260812 + i`).
* Prediction = ensemble mean of `log₁₀ λ`; uncertainty = ensemble standard
  deviation (`ddof=1`).
* No feature scaling (tree models are scale-invariant).
* **No model artifact is committed.** The model is regenerated deterministically
  by `generate_adaptive_dataset(...)` + `LearnedAdaptiveQ(...).fit(...)` in
  12 s; a persisted artifact would be a needless binary in the repository.
* Gradient boosting is single-threaded in scikit-learn, so the `n_jobs = 1`
  budget is respected by construction (the estimator takes no `n_jobs`).

### Features (6, all dimensionless and scale-free)

Computed from a window of innovations `ν` and the filter's own reported `S`.
Every feature is invariant under a consistent rescaling of `(ν, S)`, so the
model cannot learn the measurement units.

| # | Feature | Meaning |
|---|---|---|
| 1 | `log10_mean_nis_per_dof` | `log₁₀(mean NIS / m)` — the primary signal |
| 2 | `log10_trace_ratio` | `log₁₀(tr Ĉ / tr S̄)` |
| 3 | `lag1_autocorr` | lag-1 autocorrelation of the scalarised normalised innovation |
| 4 | `log10_var_norm_ch0` | `log₁₀ Var(normalised innovation, channel 0)` |
| 5 | `log10_var_norm_ch1` | same for channel 1 (duplicates ch0 when `m = 1`) |
| 6 | `frac_abs_gt_2` | fraction of normalised innovations with \|·\| > 2 |

Queries outside the min/max box seen in training set `extrapolating = True`.

## 4. Dataset

See `DATASET_CARD.md` for the full description. Summary: 150 runs × 400 steps of
a 1-D CWNA constant-velocity truth, true acceleration PSD
`q̃_true = 0.05 · 10^u m²/s³` with `u ~ U(−1.5, 1.5)`, position-only
measurements at `σ_z = 3 m`, sliding window 40 with stride 20 → **2550 feature
vectors**. Master seed 20260812; run `i` uses seed `20260812 + i`.

**All data is synthetic.** Accuracy is measured against a generative process,
not against a real vehicle.

## 5. Training procedure and test split

```bash
cd products/P012
PYTHONPATH=src python3 validation/v6_adaptive_q_benchmark.py
```

* **Train:** seeds `20260812 + i`, `i ∈ [0, 150)` → 2550 windows.
* **Held out:** seeds `20260812 + 100000 + i`, `i ∈ [0, 60)`.
* The two seed sets are **disjoint by construction**; no held-out run
  contributes to fitting, and no feature vector crosses the boundary.
* Evaluation is **causal**: the `λ` in force at step `k` is estimated only from
  data strictly before `k`, window 40, re-estimated every 20 steps. All three
  tuners use identical windows and cadence.
* **Compute:** dataset 8.22 s + fit 1.98 s + 60 runs × 3 tuners 16.41 s =
  **25.8 s** wall clock, single-threaded, on the 2-core build machine.

## 6. Metrics — held-out results

| tuner | position RMSE [m] | velocity RMSE [m/s] | ANEES (dof 2) | ANIS (dof 1) | \|log₁₀λ − u\| |
|---|---|---|---|---|---|
| fixed (hand-tuned) | 2.09982 | 0.728119 | 5.5935 | 1.3330 | n/a |
| classical Mehra IAE | 2.27394 | 1.254886 | **1.1778** | 0.7396 | 1.8164 |
| **learned** | **1.92299** | **0.671698** | 4.4935 | 1.1691 | **0.5586** |

95 % acceptance bands over 60 runs: ANEES **[1.5262, 2.5369]**, ANIS
**[0.6747, 1.3883]**.

### Paired, run by run

| comparison | wins | mean paired RMSE difference (95 % CI) |
|---|---|---|
| learned vs fixed | **44 / 60** | **−0.17683 ± 0.09361 m** (excludes zero) |
| learned vs Mehra | **47 / 60** | −0.35094 ± 0.14531 m |
| Mehra vs fixed | 14 / 60 | +0.17411 ± 0.20862 m (does **not** exclude zero) |

### Stratified by the size of the mis-specification

| stratum | n | fixed RMSE / NEES | Mehra RMSE / NEES | learned RMSE / NEES |
|---|---|---|---|---|
| \|u\| ≤ 0.5 (within ~3×) | 20 | **1.80124** / 2.5307 | 2.27280 / **1.1310** | 1.86642 / 3.4732 |
| 0.5 < \|u\| ≤ 1.0 | 18 | 1.77850 / 2.8652 | 2.23129 / **1.1018** | **1.69454** / 3.1482 |
| \|u\| > 1.0 (>10×) | 22 | 2.63417 / 10.6102 | 2.30986 / **1.2826** | **2.16133** / 6.5218 |

### Is either adaptive method actually adapting?

| tuner | λ min | λ median | λ max | pinned at upper clip | corr(log₁₀λ, true u) |
|---|---|---|---|---|---|
| Mehra IAE | 64.0000 | 64.0000 | 64.0000 | **60/60 runs** | **−0.0000** |
| learned | 0.1460 | 0.5439 | 19.5782 | 0/60 | **+0.5943** |

The Mehra scheme never leaves its upper clip and has **zero** correlation with
the true scale: it is not adapting, it is applying a constant 64× inflation of
`Q`. `Q̂ = K Ĉ Kᵀ` over a 40-step window, on a filter whose innovations are
dominated by measurement noise, produces a trace ratio above the clip in every
run — and the clip is mandatory because the raw estimator is unbounded. The
conclusion is that **classical IAE restricted to a scalar knob does not work on
this problem at this window length.**

### Reading these numbers honestly

1. **The learned tuner wins on error, and the win is statistically real** —
   44/60 paired wins, CI on the paired mean excluding zero. It is also the only
   method whose scale estimate correlates with the truth (+0.59).
2. **Mehra has the ANEES closest to 2.0 (1.178), but by saturation, not by
   inference** (table above). Its covariance is uniformly 64× inflated. If you
   want a conservative covariance on this problem, "multiply `Q` by a large
   constant" achieves it more simply and more predictably than IAE does — and
   that is exactly what IAE reduces to here.
3. **None of the three is consistent.** Mehra is *below* the lower bound
   (pessimistic), the learned tuner and the fixed baseline are *above* the upper
   bound (optimistic). A single scalar on `Q` is not sufficient to make this
   filter consistent across a 1000× spread in true process noise. The right
   response is a richer `Q` parameterisation or joint `Q`/`R` adaptation, not a
   better regressor on the same knob.
4. **The fixed baseline is the best choice when mis-specification is small.**
   At \|u\| ≤ 0.5 the fixed `Q` beats the learned tuner (1.801 vs 1.866 m):
   adapting cannot help when nothing needs adapting, and the learned tuner's
   estimation noise costs about 3.6 %. The learned tuner earns its place only
   at \|u\| > 0.5.
5. **Mehra does not beat the fixed baseline on RMSE** (14/60 wins, CI includes
   zero). Combined with its saturation, the fair summary is that on this problem
   the classical scheme contributes a constant covariance inflation and nothing
   else.

## 7. Uncertainty / confidence output — a NEGATIVE result

The model exposes `AdaptiveQPrediction(log10_scale, log10_std, scale,
confidence, extrapolating)` with `confidence = exp(−log10_std)`.

| quantity | measured on held-out runs |
|---|---|
| mean confidence | 0.9206 |
| range | [0.9001, 0.9534] |
| **correlation(confidence, \|log₁₀λ − u\|)** | **+0.2206** |
| mean error in the high-confidence half | 0.6419 |
| mean error in the low-confidence half | 0.4752 |

**A useful confidence output must be negatively correlated with the error. This
one is positively correlated, and the high-confidence half is worse than the
low-confidence half.** The confidence output as implemented **does not carry
usable information about the prediction error and must not be relied on.**

Diagnosis, stated rather than guessed at: the ensemble spread is very narrow
(0.900–0.953). Five gradient-boosted trees fitted on bootstrap resamples of
2550 samples are near-identical models, so the spread measures *model variance*
— which is small here — rather than predictive uncertainty. This is a known
limitation of bootstrap ensembles for aleatoric uncertainty. Fixing it needs a
different mechanism (quantile regression, or conformal prediction intervals
calibrated on a held-out split); that is on the roadmap and is **not**
implemented in 0.1.0.

The `extrapolating` flag *is* meaningful and is tested
(`tests/test_adaptive.py::test_out_of_domain_flagged`): it fires whenever any
feature leaves the training box.

## 8. Failure cases

1. **Small mis-specification.** At \|u\| ≤ 0.5 the learned tuner is worse than
   doing nothing (see §6). Use the fixed `Q` if you believe your tuning to
   within a factor of ~3.
2. **Consistency.** The learned tuner leaves the filter optimistic (ANEES
   4.49). It should not be used where the reported covariance matters more than
   the point estimate.
3. **Confidence output.** Uninformative (§7).
4. **Distribution shift.** The model has only ever seen a 1-D CWNA
   constant-velocity truth with a correctly specified `R` and a *scalar* `Q`
   mis-specification. It has never seen a manoeuvring target, a different
   dynamic model, coloured process noise, cross-correlated `Q`, or a
   mis-specified `R`. Behaviour outside that is unknown; the `extrapolating`
   flag is the only guard and it only checks the feature box, not the
   generating process.
5. **Short windows.** Fewer than 3 finite innovations in a window raises rather
   than returning a degenerate feature vector.
6. **Clipping.** `λ` is clipped to [1/64, 64]. A true scale outside that range
   silently saturates; the training distribution spans 1/32 … 32, so the clip
   is not active in-distribution.

## 9. Reproducibility

```bash
cd products/P012
PYTHONPATH=src python3 validation/v6_adaptive_q_benchmark.py   # ~26 s
```

Every number in §6 and §7 comes from that command; raw stdout is committed as
`validation/v6_adaptive_q_benchmark_output.txt`. Seeds: master 20260812; train
runs `+ i`; held-out runs `+ 100000 + i`; ensemble members
`random_state = 20260812 + i`. Determinism is pinned by
`tests/test_regression.py::TestPinnedAdaptiveDataset` and
`tests/test_adaptive.py::test_deterministic_for_a_seed`.

## 10. Compute used

| stage | wall clock (2 cores, single-threaded) |
|---|---|
| dataset generation (150 runs × 400 steps → 2550 windows) | 8.2–9.9 s |
| ensemble fit (5 × GradientBoostingRegressor) | 2.0–2.1 s |
| held-out evaluation (60 runs × 3 tuners) | 16.4 s |
| **total** | **~26 s** |

No GPU. No PyTorch. Peak memory well under 200 MB.

## 11. Ethical and safety limits

* **This model is not certified for operational flight use.**
* It tunes a *simulation* filter against a *synthetic* generative process. It
  has never been exposed to real vehicle data.
* It must not be placed in any control or navigation loop whose failure could
  cause harm, loss of vehicle, or loss of mission.
* Its confidence output is measurably uninformative (§7) and must not be used
  as a health signal, an integrity monitor, or a gate on any decision.
* The classical scheme's apparent consistency advantage comes from saturating
  at its clip, not from adapting. Neither method should be presented as a solved
  answer to process-noise tuning. This model card exists partly to say that
  plainly.
