# Model Card — ScintiNet Surrogate v0.1.0

**This model is not certified for operational flight use.**

## Problem

Predict the aperture-averaged scintillation index σ_I² of a horizontal
free-space optical link from four link parameters — refractive-index
structure parameter Cn², path length L, wavelength λ, and receiver aperture
diameter D — fast enough for interactive link planning and parametric sweeps,
without running a split-step wave-optics simulation per query.

σ_I² is the normalised variance of received irradiance, ⟨I²⟩/⟨I⟩² − 1. It
drives fade statistics, required link margin, and outage probability in FSO
link budgets.

## Classical / analytic baseline (implemented first)

`scintinet.rytov_baseline` — the weak-fluctuation Rytov index with
aperture averaging:

- σ_R² = 1.23 Cn² k^(7/6) L^(11/6) (plane wave, Kolmogorov spectrum;
  Andrews & Phillips, *Laser Beam Propagation through Random Media*,
  2nd ed., SPIE Press, 2005)
- A = [1 + 1.062 kD²/(4L)]^(−7/6) (Andrews, J. Opt. Soc. Am. A 9(4), 597, 1992)
- σ_I²(D) = A · σ_R²

The baseline was implemented, hand-checked against textbook closed forms
(`validation/VALIDATION.md` §V1), and benchmarked on the *same held-out data*
as the ML model. It is the reference the surrogate must beat to justify
itself.

## Architecture

- 5-member ensemble of `sklearn.neural_network.MLPRegressor`.
- Each member: hidden layers (32, 32), ReLU, L-BFGS solver, `max_iter=2000`,
  `random_state = base + member_index`, preceded by `StandardScaler` in a
  `Pipeline`.
- Features: `[log10 Cn², log10 L, log10 λ, D]`. Log scaling handles the
  ~4-decade dynamic range of Cn² and the power-law structure of the physics;
  D is left linear because the aperture factor is not a pure power law.
- Target: `log10 σ_I²`; predictions are exponentiated back to linear σ_I²,
  which also guarantees strictly positive output.
- Total parameters per member: ~1200. Ensemble: ~6000. Chosen small
  deliberately — 40 training rows cannot support more.

## Dataset

See `DATASET_CARD.md`. 54 rows from a **reduced-scale** seeded simulation
campaign (256² grid, 8 screens, 8 realizations, 22.6 s total), covering
Cn² ∈ {1e-16, 3.16e-16, 1e-15}, L ∈ {1000, 2000, 3000} m,
λ ∈ {850 nm, 1550 nm}, D ∈ {2, 50, 100} mm. All points lie in the weak
regime (σ_R² ≤ 0.30). Key limitations: plane wave only, horizontal
homogeneous path only, no subharmonics in the phase screens, ~5–10 %
statistical noise per target value, and only 54 rows.

## Training procedure

```bash
python validation/run_campaign.py        # regenerate dataset.csv (seed 2026+i)
python validation/benchmark_surrogate.py # split, train, benchmark
```

- Split: shuffle split with `numpy.random.default_rng(0)`, 25 % test →
  40 train / 14 test rows.
- No hyperparameter search was performed against the test set; the
  architecture was fixed a priori as "small enough for 40 points".
- Fit time: **2.0 s** for all 5 members on 2 CPU cores.
- The three aperture rows sharing a simulation seed are correlated, so the
  random row split leaks mildly across train/test. A group split by
  simulation point would be stricter; documented, not corrected.

## Metrics (actual measured run)

Held-out test set, n = 14:

| Model | RMSE (log10 σ_I²) | median \|rel err\| | max \|rel err\| |
|---|---|---|---|
| MLP surrogate (5-ensemble) | 0.0781 | 0.1665 | 0.2824 |
| **Rytov analytic baseline** | **0.0429** | **0.0700** | **0.2276** |

**The analytic baseline beats the surrogate on every metric.** This is
reported as measured and was not tuned away. The result is expected and
honest: the benchmark sits entirely inside the baseline's validity regime,
where the baseline is a near-exact closed form, and 40 noisy training points
are nowhere near enough for a neural network to rediscover that closed form
to better accuracy. The surrogate's 16.7 % median error is in fact
comparable to the ~5–10 % statistical noise on its own training targets.

## When is a surrogate actually worthwhile?

Not here. It becomes worthwhile when:

1. **No closed form exists.** Strong-fluctuation regime (σ_R² > 1) with
   focusing and saturation, non-Kolmogorov spectra, inner/outer-scale
   effects, slant paths with Cn²(h) profiles, or partially coherent /
   Gaussian beams — regimes where the only ground truth is simulation.
2. **Speed matters against simulation, not against algebra.** Measured:
   0.269 ms per surrogate prediction versus ~1.3 s per split-step point at
   this grid (~5000×), and far more at production grid sizes. That gap
   enables Monte Carlo link-availability studies and optimisation loops that
   direct simulation cannot support.
3. **Enough training data exists.** With 10³–10⁴ converged simulation points
   rather than 54, the picture would be materially different.

In-regime, for a horizontal Kolmogorov plane-wave link, use
`rytov_baseline`. It is faster, exact, and interpretable.

## Uncertainty output

`Surrogate.predict(X, return_std=True)` returns the ensemble mean and the
standard deviation across the 5 members (deep-ensemble style;
Lakshminarayanan et al., NeurIPS 2017). Measured on the test set: mean std
2.97e-03 against mean prediction 1.77e-02, i.e. ≈17 % relative spread, which
tracks the measured 16.7 % median error well.

**Caveat:** this is epistemic (model-disagreement) uncertainty only. It does
not include simulation statistical noise, phase-screen bias, or the error of
the underlying physics model, and it is **not a calibrated predictive
interval**. Do not use it as a confidence bound for link margin sizing. It
is most useful as a relative extrapolation warning: spread grows where
training coverage is thin.

## Failure cases

- **Extrapolation outside the training box.** No guard rails: querying
  Cn² > 1e-15, L > 3000 m, or λ outside 850–1550 nm produces a confident
  number from a network that has never seen that region. The physics is a
  power law; the network is not.
- **Strong fluctuations.** σ_R² > 1 produces the wrong answer with no
  warning; the true σ_I² peaks near σ_R² ≈ 2–4 and then saturates toward 1,
  behaviour absent from both the training data and the analytic baseline.
- **Non-plane-wave sources.** Gaussian beams, spherical waves and finite
  transmit apertures are not represented at all.
- **Slant / vertical paths.** Constant-Cn² assumption; no altitude profile.
- **Inherited simulator bias.** The surrogate learns the simulator's ~15 %
  low bias on finite apertures, so it is biased low relative to Andrews
  theory there by construction.
- **Small-D behaviour.** D was trained on only three discrete values;
  interpolation between 2 mm and 50 mm is unconstrained by data.

## Reproducibility

Exact commands and seeds:

```bash
python validation/run_campaign.py         # seeds 2026 + i, i = 0..17
python validation/benchmark_surrogate.py  # split seed 0, Surrogate(random_state=0)
python examples/sweep_sigma_i2.py         # sweep seeds 7000 + i
python -m pytest tests/ -q                # 50 tests
```

`Surrogate(random_state=r)` gives member seeds r, r+1, …; identical
`(data, random_state)` gives bit-identical predictions
(`tests/test_surrogate.py::test_deterministic_given_random_state`).

## Compute used

Whole build fits the 2-core budget: campaign 22.6 s, surrogate fit 2.0 s,
full test suite 8.2 s, examples ~16 s. No GPU. No PyTorch. Peak memory
< 500 MB (dominated by 256² complex FFT arrays).

## Ethical and safety limits

- Research and educational use only. Synthetic data throughout; no personal
  data, no dual-use content beyond standard atmospheric-optics engineering.
- Predictions must not be used to size link margin, certify availability, or
  make go/no-go decisions for any operational optical link, terrestrial or
  space-to-ground.
- Any safety-relevant use requires converged simulation, experimental
  validation against measured scintillometer data, and independent review —
  none of which this product has.
- **This model is not certified for operational flight use.**
