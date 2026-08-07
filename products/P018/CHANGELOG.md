# Changelog

All notable changes to ShackSim (`shacksim`) are recorded here.
This project adheres to semantic versioning.

## 0.1.0 — 2026-08-07

Initial release (status: TESTING, validation Level 2 / Research, AI-enabled).

### Added

- **Lenslet-array geometry** (`shacksim.geometry`):
  - `LensletArray` — configurable lenslet count, pitch, focal length, pixels per
    subaperture, wavelength, circular pupil with optional central obscuration
    and a fill-fraction validity criterion. Derived quantities: pixel size,
    pixel angle `p/f`, diffraction spot FWHM `1.0287938 λf/d`
    (Born & Wolf 1999, sec. 8.5.2), Gaussian-equivalent σ, maximum measurable
    slope, illuminated-subaperture mask and centres.
  - `slope_to_displacement` / `displacement_to_slope` — the measurement
    principle `Δx = f·g/p`, derived explicitly in the docstring
    (Hardy 1998, *Adaptive Optics for Astronomical Telescopes*, ch. 5).
- **Spot formation and noise chain** (`shacksim.sensor`):
  - `subaperture_spot` — pixel-integrated (error-function) Gaussian
    approximation to the Airy core, with optional axis-aligned elongation.
  - `simulate_frame` — full detector frame from a slope vector, with uniform
    background, Poisson shot noise and Gaussian read noise. Bitwise
    reproducible from a fixed seed.
  - `extract_subapertures`, `generate_subaperture_dataset` — frame-to-stamp
    cutting and labelled synthetic dataset generation.
- **Analytic wavefront fields** (`shacksim.wavefront`): `tilt_slopes`,
  `defocus_slopes`, `random_slopes` (documented as white noise, **not**
  turbulence), `slope_rms`.
- **Classical slope estimators** (`shacksim.slopes`), implemented before the ML
  model as required for AI products:
  - `cog_displacement` / `cog_slopes` — thresholded centre of gravity with
    optional negative clipping (Thomas et al. 2006, *MNRAS* **371**, 323).
  - `reference_template`, `correlation_displacement` / `correlation_slopes` —
    cross-correlation with 3-point parabolic peak interpolation
    (Poyneer 2003, *Applied Optics* **42**, 5807).
  - `cog_noise_sigma` — the standard centroid noise-propagation expression,
    re-derived in the docstring including the `p²d²` lever term, with `M2`
    evaluated numerically from the pixel-integrated profile
    (Hardy 1998 ch. 5; Thomas et al. 2006; Winick 1986, *JOSA A* **3**, 1809).
- **Learned slope estimator** (`shacksim.ml`):
  - `MLSlopeEstimator` — ensemble of 5 `sklearn.neural_network.MLPRegressor`
    networks on the flux-normalized stamp plus a `log10(1+counts)` regime
    feature, with `predict(..., return_std=True)` exposing the per-slope
    ensemble spread as a confidence proxy (Lakshminarayanan et al.,
    NeurIPS 2017), and `predict_frame` for whole frames.
- **Tests** (148, all passing): hand-calculated known-answer tests for geometry,
  slope conversion, the known-tilt answer, the background-shrinkage factor and
  the photon-noise limit; zero-wavefront-to-zero-slopes tests; input-validation
  tests across all public functions; Hypothesis property tests for spot mirror
  and transpose symmetry, CoG mirror/rotation/transpose/gain invariance and
  slope round-trips; seeded reproducibility tests for the generator and the
  model; and a pinned benchmark regression test covering both the ML win at
  low flux and its documented loss outside the training envelope.
- **Examples** (Agg backend, PNGs to `screenshots/`):
  - `examples/spot_field.py` — simulated frame with measured slope vectors and
    the magnified residual field.
  - `examples/slope_error_vs_flux.py` — slope error vs flux for both classical
    baselines and the ML ensemble, round and elongated spots, crossover marked.
- **Validation** (`validation/run_validation.py`, transcript in
  `validation/validation_output.txt`, written up in `validation/VALIDATION.md`,
  three figures): known-tilt uniformity (worst-case 8.702e-09 rad, PASS at
  1e-8 rad); zero-wavefront (6.15e-20 rad, PASS at 1e-12 rad) and the
  correlation S-curve (0.0302 px peak); noise-propagation agreement (ratio
  0.990–1.066 for N ≥ 300 e⁻, PASS at 0.85–1.15) with the documented breakdown
  at 100 e⁻ (35.2×); background shrinkage against the analytic `S/(S+Bp²)`
  (1.39e-05 px worst case, PASS at 1e-3 px); and the ML-vs-classical benchmark
  with the crossover reported where it falls.
- `MODEL_CARD.md`, `DATASET_CARD.md`, `README.md`, `LICENSE` (Apache-2.0),
  `pyproject.toml`.

### Known limitations recorded at release

- **Deviation from the ideal design:** a convolutional network is the natural
  architecture for this task; PyTorch is unavailable in the build environment,
  so an ensemble of scikit-learn MLPs is used instead. The model has no
  convolutional inductive bias. Documented in `MODEL_CARD.md` and README
  Limitations.
- The learned estimator beats the tuned thresholded centre of gravity **only
  below ≈ 100 e⁻ per subaperture** (crossover 50–100 e⁻ round, 100–300 e⁻ for 3×
  elongated spots), by at most 1.38×, and loses by up to 10.5× at high flux.
  Against the correlation baseline it loses at 6 of 7 flux levels for round
  spots. Reported as measured.
- The ensemble spread is **not calibrated** — spread/error ratio 0.15 to 1.12,
  under-stating the true error by up to 6.5× in the low-flux regime — and must
  not be used as a measurement covariance.
- The standard CoG noise-propagation expression under-predicts by 35× at
  100 e⁻; its first-order linearization is invalid there. Reported as a failed
  check rather than tuned away.
- All data is synthetic from an idealized optical model. Airy rings, lenslet
  aberrations, chromatic effects, inter-subaperture crosstalk, arbitrary-angle
  spot elongation, turbulence statistics, dead/hot pixels, PRNU/DSNU,
  saturation, quantization and charge diffusion are not modelled; real-hardware
  performance is unknown.
- No wavefront reconstruction (slopes only) and no temporal processing.
- Characterized only at 8 × 8 lenslets, 500 µm pitch, f = 50 mm,
  16 px/subaperture, λ = 633 nm, B = 1 e⁻/px, R = 3 e⁻ RMS.
