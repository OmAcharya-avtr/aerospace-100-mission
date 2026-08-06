# Changelog

All notable changes to CentroidNet (`centroidnet`) are recorded here.
This project adheres to semantic versioning.

## 0.1.0 — 2026-08-06

Initial release (status: TESTING, validation Level 2 / Research, AI-enabled).

### Added

- **Synthetic spot generator** (`centroidnet.generator`):
  - `spot_image` — noise-free 2-D Gaussian spot on an N×N grid, with exact
    per-pixel integration via the error function (`pixelated=True`) or point
    sampling at pixel centres (`pixelated=False`).
  - `generate_spots` — batches of frames with exact (x, y) labels and an
    idealized noise chain: uniform background, Poisson shot noise, additive
    Gaussian read noise. Bitwise reproducible from a fixed `seed`.
  - `snr_estimate` — CCD aperture-photometry detection SNR,
    `S / sqrt(S + N_pix(B + R²))` (Howell 2006).
- **Classical analytic baselines** (`centroidnet.baselines`), implemented before
  the ML model as required for AI products:
  - `cog_centroid` — intensity-weighted centre of gravity with optional
    threshold and negative-pixel clipping (Thomas et al. 2006, MNRAS 371, 323).
  - `quadcell_centroid` — quadrant-detector estimate with configurable output
    scale; `scale = σ√(π/2)` calibrates the small-offset slope to unity
    (Tyler & Fried 1982, JOSA 72, 804; Hardy 1998 ch. 5).
- **ML estimator** (`centroidnet.ml`):
  - `MLCentroider` — ensemble of 5 `sklearn.neural_network.MLPRegressor`
    networks on the flux-normalized flattened pixel vector, with gain-invariant
    preprocessing and `predict(..., return_std=True)` exposing the ensemble
    spread as an uncertainty proxy (Lakshminarayanan et al., NeurIPS 2017).
- **Tests** (41, all passing): known-answer tests with hand-calculated values,
  input-validation tests, edge cases, Hypothesis property tests for CoG mirror
  symmetry and gain invariance, ML benchmark-vs-baseline, uncertainty-shape and
  reproducibility tests.
- **Examples** (Agg backend, PNGs to `screenshots/`):
  - `examples/error_vs_snr.py` — RMS error vs SNR for all four estimators with
    the ensemble-spread band.
  - `examples/spot_gallery.py` — eight frames across four signal levels with
    true, CoG, quad-cell and ML estimates overlaid.
- **Validation** (`validation/run_validation.py`, output in
  `validation/validation_output.txt`, documented in `validation/VALIDATION.md`):
  noise-free CoG recovery (worst-case 2.934e-04 px, PASS at 1e-3 px); quad-cell
  bias curve vs the analytic erf response (max deviation 5.538e-05 px, PASS at
  1e-2 px) quantifying the linear-range limitation; bias and RMS error vs SNR
  for baselines and the ML ensemble on 3000 held-out frames; ensemble-spread
  calibration measurement.
- `MODEL_CARD.md`, `DATASET_CARD.md`, `README.md`, `LICENSE` (Apache-2.0),
  `pyproject.toml`.

### Known limitations recorded at release

- **Deviation from specification:** the product was specified with a small CNN;
  PyTorch is not available in the build environment, so an ensemble of
  scikit-learn MLPs is used instead. The model has no convolutional inductive
  bias. Documented in `MODEL_CARD.md` and README Limitations.
- The ML ensemble beats the plain CoG at every tested SNR but beats the
  **thresholded** CoG only below SNR ≈ 40; above the crossover the analytic
  estimator is better (2.2× better at SNR 88). Reported as measured.
- The ensemble spread is **not calibrated** — it under-estimates the true error
  by 2.3×–11× and must not be used as a 1-σ error bar.
- All data is synthetic from an idealized sensor model. Dead pixels, PRNU/DSNU,
  optical aberrations, stray light and detector nonlinearity are not modelled;
  real-detector performance is unknown.
- Characterized only at 16×16 px, σ = 1.5 px, offsets ≤ 2 px, B = 2 e⁻/px,
  R = 3 e⁻ RMS.
