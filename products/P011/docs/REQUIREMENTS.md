# WaveForge — Requirements and Verification Matrix

**Product:** P011 WaveForge · **Version:** 0.1.0 · **Status:** TESTING
**Validation level:** 3 · **License:** AGPL-3.0-or-later
**Date of this revision:** 2026-08-29

This document states what `waveforge` is required to do, in numbered,
individually verifiable terms, and names for each requirement the test or
validation script that verifies it. Requirements are written so that a
verification can *fail*: "shall reproduce X within Y" rather than "shall be
accurate".

Paths are relative to `products/P011/`. Test identifiers are pytest node ids;
validation identifiers are scripts in `validation/` whose captured output is
stored alongside them as `*_output.txt`.

---

## 1. Scope and definitions

WaveForge sizes and simulates a single-conjugate adaptive-optics system for a
free-space optical terminal: Kolmogorov atmosphere, Shack-Hartmann wavefront
sensing, a deformable mirror with finite stroke, an integrator control law with
configurable gain and latency, a cited residual error budget, and an optional
learned predictive controller.

| Term | Meaning in this document |
|---|---|
| *phase* | wavefront phase in radians at the sensing wavelength |
| *slope* | subaperture-averaged phase gradient in rad/m |
| *latency* `d` | total loop delay in WFS frames, as in `control.rejection_transfer` |
| *residual variance* | piston-removed spatial variance of the corrected phase over the pupil [rad²] |
| *published value* | a number taken from a cited, externally published source |

---

## 2. Functional requirements

### R-01 — Kolmogorov phase screens
The package shall generate two-dimensional Kolmogorov (and von Kármán) phase
screens whose measured structure function agrees with the exact expectation of
the discrete spectrum they are drawn from to within **5 %** over separations
from one sample to one eighth of the screen.

*Rationale:* separating implementation error from the method's known
band-limitation is the only way to make the screen generator auditable.

### R-02 — Screen band-limitation disclosed
The package shall expose the exact band-limited structure function of its own
sampling grid, so that the difference between the generated screens and the
continuous Kolmogorov result is a computed number and not a claim.

### R-03 — Frozen-flow sequence integrity
A frozen-flow atmosphere shall refuse to produce a frame whose window would
wrap the non-periodic subharmonic component of the screen, and shall report the
maximum usable frame index.

### R-04 — Zernike conventions
Zernike modes shall use the Noll (1976) single index and orthonormalisation,
with an exact integer index map in both directions, and shall be orthonormal on
the unit disc to within **1e-12** under exact quadrature.

### R-05 — Kolmogorov modal statistics
Analytic Kolmogorov Zernike coefficient variances and the residual variances
`Δ_J` shall agree with Noll (1976) Table IV to within **1 %** for
`J = 1 … 21`.

### R-06 — Shack-Hartmann slope measurement
A wavefront of uniform gradient `g` shall produce a slope of exactly `g` in
every illuminated subaperture, to within **1e-12 rad/m**.

### R-07 — Sensor noise model
Slope measurement noise shall follow the published centre-of-gravity photon and
read-noise expressions (Rousset 1999), scale as `N_ph^(-1/2)` for photon noise,
and be reported both as a slope sigma [rad/m] and as a noise-equivalent angle
[rad].

### R-08 — Deformable mirror
The mirror shall be a linear superposition of influence functions with an
explicit nearest-neighbour coupling, shall enforce a symmetric stroke limit,
and shall report the fraction of actuators saturated on every command.

### R-09 — Fitting-error scaling
The residual left by the mirror shall follow `a_F (d_act/r0)^(5/3)`, and the
coefficient measured at the finest tested actuator pitch, after correcting for
the power the screens do not carry above the grid Nyquist, shall lie within
**10 %** of the published 0.28 (Hudgin 1977).

### R-10 — Closed-loop control law
The closed loop shall implement a leaky integrator with configurable gain,
latency `d ≥ 1` and leak, and its measured error rejection shall match the
analytic `E(z) = (1 - leak z⁻¹) / ((1 - leak z⁻¹) + g z⁻ᵈ)` to within **1e-4**
relative in the scalar case.

### R-11 — Stability limit
The computed largest stable gain shall match the closed forms `2` (`d = 1`),
`1` (`d = 2`) and `2 sin(π/10)` (`d = 3`) to within **1e-6**, and the
time-domain loop shall remain bounded below the limit and diverge above it.

### R-12 — Noise propagation
The closed-loop noise-variance amplification shall equal the classical
`g/(2-g)` for one frame of latency to within **1e-6** relative, and shall be
reported as infinite for an unstable gain.

### R-13 — Error budget
The package shall produce an additive residual error budget with separate
fitting, temporal and noise terms, each with a cited expression, and shall not
silently combine the pure-delay and servo-bandwidth temporal forms.

### R-14 — Strehl ratio
The package shall provide both a numerical Strehl from the complex pupil field
and the extended Maréchal approximation, and shall document the measured range
of residual variance over which the two agree to within 5 %.

### R-15 — Learned predictor with uncertainty
The learned predictive controller shall forecast pseudo-open-loop slopes from a
history of measurements, shall expose a per-slope one-sigma uncertainty and not
only a point estimate, and shall be benchmarked on held-out phase-screen
realisations against (a) the classical integrator at its own tuned gain and
(b) a pure-delay baseline running through the identical control path.

### R-16 — Honest reporting of the AI result
Whichever controller wins at each tested latency and noise level shall be
reported as measured, including the regimes where the learned model loses.

### R-17 — Determinism and regeneration
Every dataset, screen and simulation result shall be reproducible from an
integer seed; no data or model artefact larger than 1 MB shall be stored as a
file.

### R-18 — Command-line interface
`python -m waveforge` shall expose the Noll table, screen statistics, error
budget, closed-loop run and predictor benchmark as subcommands, and shall exit
with a diagnostic rather than a traceback on invalid input.

---

## 3. Verification matrix

| Req | Verified by | Kind |
|---|---|---|
| R-01 | `validation/validate_atmosphere.py` §1 · `tests/test_atmosphere.py::TestStructureFunction::test_matches_band_limited_prediction` | validation + test |
| R-02 | `validation/validate_atmosphere.py` §2 · `tests/test_atmosphere.py::TestStructureFunction::test_band_limited_is_below_continuous_theory` | validation + test |
| R-03 | `tests/test_atmosphere.py::TestFrozenFlow::test_max_frames_guard` · `::test_no_guard_without_subharmonics` | test |
| R-04 | `validation/validate_zernike.py` §1a, §1b · `tests/test_zernike.py::TestIndexing` · `::TestBasis::test_orthonormality_on_fine_grid` · `tests/test_properties.py::TestZernikeProperties` | validation + test |
| R-05 | `validation/validate_zernike.py` §2, §3, §3b · `tests/test_statistics.py::TestNollResidual::test_matches_published_table` · `::TestZernikeVariance` | validation + test |
| R-06 | `tests/test_sensor.py::TestKnownAnswers::test_uniform_tilt_gives_uniform_slope` · `::test_uniform_tilt_in_y` | test |
| R-07 | `tests/test_sensor.py::TestNoiseModel` (known-answer photon and read terms, flux scaling) · `::TestMeasurement::test_noise_statistics` | test |
| R-08 | `tests/test_dm.py::TestGeometry` · `::TestStroke` · `tests/test_failure_modes.py::TestActuatorSaturation` | test |
| R-09 | `validation/validate_fitting_error.py` §3 · `tests/test_errorbudget.py::TestFittingError` | validation + test |
| R-10 | `validation/validate_rejection_tf.py` §1, §4 · `tests/test_control.py::TestTimeDomainMatchesAnalytic` | validation + test |
| R-11 | `validation/validate_rejection_tf.py` §2 · `tests/test_control.py::TestStabilityLimits` · `tests/test_failure_modes.py::TestLoopInstability` | validation + test |
| R-12 | `validation/validate_rejection_tf.py` §3 · `tests/test_control.py::TestNoiseVarianceGain` | validation + test |
| R-13 | `tests/test_errorbudget.py::TestErrorBudget` · `tests/test_loop.py::TestErrorBudgetIntegration` | test |
| R-14 | `validation/validate_strehl_marechal.py` §1–§3 · `tests/test_errorbudget.py::TestStrehl` · `tests/test_pupil.py::TestStrehl::test_strehl_matches_marechal_for_small_gaussian_phase` | validation + test |
| R-15 | `validation/validate_predictor.py` §1–§3 · `tests/test_predictor.py::TestLinearSlopePredictorBehaviour` · `tests/test_loop.py::TestClosedLoop::test_pure_delay_predictor_runs` | validation + test |
| R-16 | `validation/validate_predictor.py` §3–§5 (recorded in `validation/VALIDATION.md` §6 and `MODEL_CARD.md`) · `tests/test_cli.py::TestPredictCommand::test_runs_and_reports_a_verdict` | validation + test |
| R-17 | `tests/test_regression.py` (all pinned outputs) · `tests/test_datasets.py::TestDataset::test_deterministic` · `tests/test_performance.py::TestScaling::test_memory_footprint_of_the_operators` | test |
| R-18 | `tests/test_cli.py` (all classes) | test |

---

## 4. Requirements deliberately **not** met in 0.1.0

These are recorded here so that the gap is visible rather than implied.

| Ref | Not implemented | Consequence |
|---|---|---|
| N-01 | Multi-layer atmospheres with per-layer wind | single-layer frozen flow only; predictable turbulence, so the learned controller's advantage is an upper bound |
| N-02 | Spot-image-level Shack-Hartmann model | slope-level model only; spot truncation, elongation and non-linearity are out of scope (see product P018) |
| N-03 | DM hysteresis and non-linearity | the mirror is exactly linear; stroke saturation is the only non-linearity modelled |
| N-04 | Aliasing term in the error budget | `ErrorBudget.other` is available for it, but no aliasing model is supplied |
| N-05 | Scintillation and amplitude fluctuation | pupil amplitude is uniform; Strehl is a pure-phase quantity here |
| N-06 | Non-common-path and calibration errors | not modelled |
| N-07 | WFS integration `sinc` roll-off in the transfer function | the discrete loop model omits it; it matters above ~0.3 `f_s` |

---

## 5. Verification status summary

Run in the 0.1.0 build session on 2026-08-29 (Python 3.11.15, numpy 2.4.4,
scipy 1.17.1, scikit-learn 1.8.0, matplotlib 3.10.9, 2 CPU cores):

* `python -m pytest tests/ -q` → **635 passed**
* `ruff check src/ tests/ examples/ validation/` → **All checks passed**
* six validation scripts executed, raw output stored in `validation/`

Two verifications record a **documented deviation rather than a pass**; both
are described in `validation/VALIDATION.md`:

1. R-05: the per-mode variances implied by differencing Noll's rounded table
   entries are consistent with this package for 13 of 20 modes; the remaining
   seven differ in one direction by an amount matching the +0.25 % systematic
   offset between this package's Kolmogorov normalisation and Noll's published
   third significant figure. Two independent derivations inside the package
   agree with each other to 0.03 %.
2. R-14: the extended Maréchal form is confirmed to 5 % only up to
   `σ² ≈ 0.5 rad²` for scaled Kolmogorov phase, and underestimates the Strehl
   of genuine closed-loop residuals by up to 27 % near `σ² ≈ 2 rad²`.
