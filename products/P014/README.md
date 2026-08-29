# WaveLab

**Status:** TESTING · **Class:** medium · **Validation level:** 2 · **AI:** yes

## Executive overview

WaveLab reconstructs a wavefront from Shack-Hartmann-style slope
measurements. It implements two families of classical, regularized
least-squares reconstructor first — a **modal** reconstructor that maps
slopes directly to Noll-Zernike coefficients, and a **zonal** reconstructor
that maps slopes to phase on a grid using the Hudgin and Fried
finite-difference geometries — and only then a **learned** reconstructor
(an ensemble of small neural networks) that is benchmarked against the
regularized modal baseline across photon flux and subaperture-dropout rate.
The measured result, reported without adjustment, is that the classical
baseline wins almost everywhere tested; the learned model's one measured
advantage is at extreme subaperture dropout, where the baseline's fixed
regularization strength becomes unstable. See `MODEL_CARD.md` and
`validation/VALIDATION.md` for the full numbers.

## Aerospace problem

A Shack-Hartmann wavefront sensor measures the local slope (gradient) of an
incoming wavefront in an array of subapertures, not the wavefront itself.
Recovering phase from slopes is classically a least-squares inverse problem
that (a) has a nontrivial null space — at minimum a global piston offset is
always unobservable from slopes, and some finite-difference geometries add
further unobservable modes — and (b) degrades badly at low photon flux (few
photons per subaperture, e.g. a faint guide star) and when subapertures are
missing or unusable (dead pixels, saturated or low-SNR spots, partial pupil
obscuration). This matters directly for adaptive-optics systems on
ground-based telescopes and any space-based active-optics or beam-control
system that closes a wavefront-sensing loop.

## Intended users

Researchers and engineers prototyping wavefront reconstruction algorithms,
teaching the Hudgin/Fried/modal distinction, or benchmarking a learned
reconstructor against a properly regularized classical baseline before
considering it for a real system (which this package's own results argue
against doing lightly — see "AI model details" below).

## Engineering theory

**Zernike modes** (`wavelab.zernike`). Noll (1976) single-index convention
and orthonormal scaling: R. J. Noll, "Zernike polynomials and atmospheric
turbulence", *J. Opt. Soc. Am.* **66** (3), 207-211. Radial polynomial
definition: M. Born & E. Wolf, *Principles of Optics*, 7th ed., Cambridge
University Press, 1999, Sec. 9.2. Modes are dimensionless on the normalised
unit disc (`x² + y² ≤ 1`); a coefficient vector carries whatever unit the
caller assigns (WaveLab uses radians of phase throughout). Analytic
gradients `∂Z/∂x`, `∂Z/∂y` are derived by the chain rule and are exact to
machine precision (no finite differences), valid *per unit normalised pupil
radius*.

**Zonal geometries** (`wavelab.geometry`), reviewed and compared by
Southwell (1980), "Wave-front estimator for wave-front sensing",
*J. Opt. Soc. Am.* **70** (8), 998-1006:

- **Hudgin** — R. H. Hudgin, *J. Opt. Soc. Am.* **67** (3), 375-378 (1977).
  Each slope is the finite difference of two *adjacent* grid phase points.
  Null space: **piston only** (verified numerically for grids 5x5 through
  13x13, `tests/test_geometry.py::test_hudgin_null_space_is_piston_only`).
- **Fried** — D. L. Fried, *J. Opt. Soc. Am.* **67** (3), 370-375 (1977).
  Each slope is the *average* of the two finite differences along one edge of
  a unit grid cell. Averaging decouples the grid into two independent
  checkerboard sub-lattices, giving a **two-dimensional** null space (piston
  + "waffle", the alternating `(-1)^(i+j)` pattern), documented in Hardy,
  *Adaptive Optics for Astronomical Telescopes*, Oxford, 1998, ch. 5, and
  Herrmann, "Least-squares wave front errors of minimum norm",
  *J. Opt. Soc. Am.* **70** (1), 28-35 (1980). Verified numerically
  (`tests/test_geometry.py::test_fried_null_space_is_two_dimensional_piston_and_waffle`).
  A subtlety found and fixed during development: on a *circular* (not
  rectangular) pupil mask, some boundary grid points touch no complete Fried
  cell and are trivially unconstrained — these are pruned
  (`wavelab.geometry.prune_unconstrained`) before null-space analysis,
  otherwise they inflate the apparent null space with spurious one-point
  directions that have nothing to do with piston or waffle.

**Regularization** (`wavelab.linalg`): Tikhonov (A. N. Tikhonov & V. Y.
Arsenin, *Solutions of Ill-Posed Problems*, Winston & Sons, 1977) and
truncated-SVD (P. C. Hansen, "The truncated SVD as a method for
regularization", *BIT* **27**, 534-553, 1987). TSVD gives an explicit,
inspectable null space; Tikhonov shrinks every direction (including the null
space) continuously and is mean-subtracted (piston-removed) by convention in
`ZonalReconstructor`.

**Kolmogorov phase screens** (`wavelab.screens`): power spectral density
from F. Roddier, "The effects of atmospheric turbulence in optical
astronomy", *Prog. Optics* **19**, 281-376 (1981); FFT generation method
from B. L. McGlamery, *Proc. SPIE* **74**, 225-233 (1976), discretized
following J. D. Schmidt, *Numerical Simulation of Optical Wave Propagation*,
SPIE Press, 2010, Sec. 9.3. **Known limitation, not corrected**: the pure FFT
method under-represents low-spatial-frequency (large-scale) power — Lane,
Glindemann & Dainty, *Waves in Random Media* **2**, 209-224 (1992), who
propose subharmonic compensation, not implemented here.

**Photon-noise slope model** (`wavelab.noise`): shot-noise-limited scaling
`sigma_slope(N) ∝ 1/sqrt(N)`, the photon-noise term of the classical centroid
variance result (Hardy 1998, ch. 5; Thomas et al. 2006, *MNRAS* **371**,
323). Deliberately omits the read-noise/background term (which scales with
detector window area, not photon count alone) and all pixel-level spot
physics — that is the scope of the related product ShackSim (P018), cited
here rather than duplicated.

**Noise propagation** (`wavelab.linalg.noise_propagation_coefficients`):
`Var(â_k) = coeff_k · sigma_s²` for a linear pseudo-inverse reconstructor —
Wallner, "Optimal wave-front correction using slope measurements",
*J. Opt. Soc. Am.* **73**, 1771 (1983); Hardy 1998, ch. 9.

## Architecture

```
wavelab/
├── zernike.py    # Noll indexing, radial polynomials, analytic gradients, fitting
├── geometry.py    # PupilGrid, Hudgin/Fried zonal matrices, waffle, pruning
├── linalg.py       # Tikhonov / TSVD solvers, null space, noise propagation
├── modal.py         # ModalReconstructor: regularized slopes -> Zernike coeffs
├── zonal.py           # ZonalReconstructor: regularized slopes -> grid phase
├── screens.py           # Kolmogorov phase screens (FFT method)
├── noise.py               # photon-shot-noise slope model, dropout sampling
├── dataset.py               # deterministic synthetic slopes-to-Zernike batches
├── ml.py                     # ZernikeSlopeEnsemble: learned reconstructor
└── cli.py / __main__.py        # python -m wavelab
```

All Zernike and slope machinery is implemented independently within this
package (no cross-product imports, per the mission build rules). The same
conventions are documented independently by the related products **ZernKit**
(P016, Zernike toolkit) and **ShackSim** (P018, Shack-Hartmann sensor
simulation), cited here as related prior work rather than duplicated or
imported.

## Installation

```bash
cd products/P014
python3 -m pip install -e .
```

Requires Python >= 3.11, numpy, scipy, scikit-learn, matplotlib. No GPU, no
network access, no PyTorch (not available in the target build environment;
see `MODEL_CARD.md` "Architecture" for what that ruled out).

## Quick start

```bash
cd products/P014
PYTHONPATH=src python -m wavelab geometry --n-grid 9
PYTHONPATH=src python -m wavelab reconstruct --n-side 8 --j-max 15 --flux 1000 --dropout 0.1
PYTHONPATH=src python -m wavelab demo-benchmark --n-train 300 --n-test 150
```

Or from Python:

```python
from wavelab.dataset import build_modal_geometry, generate_batch
from wavelab.modal import ModalReconstructor

geometry = build_modal_geometry(noll_indices=list(range(2, 16)), n_side=8)
recon = ModalReconstructor(geometry.noll_indices, geometry.sub_x, geometry.sub_y,
                            method="tikhonov", reg=1e-3)
batch = generate_batch(geometry, n_samples=1, photon_flux=1000.0, dropout_rate=0.1, seed=0)
coeffs_hat = recon.reconstruct(batch.slopes[0], active=batch.active[0])
```

## Configuration

Every reconstructor and generator takes its parameters as explicit
constructor/function arguments (subaperture layout, Noll mode range,
regularization method and strength, photon flux, dropout rate, random seed)
— there is no hidden global configuration file or environment variable.

## Examples

- `examples/reconstruction_demo.py` — reconstructs one synthetic Kolmogorov
  screen with the modal least-squares baseline under noise and dropout;
  saves `screenshots/reconstruction_demo.png` (true / reconstructed /
  residual wavefront maps).
- `examples/benchmark_plot.py` — a reduced-size flux-and-dropout sweep
  comparing the baseline and the learned ensemble; saves
  `screenshots/benchmark_flux_dropout.png`.

Run with `PYTHONPATH=src python examples/<script>.py` from `products/P014`.

## Validation

Level 2. Full detail, raw numbers, and every script + saved output:
[`validation/VALIDATION.md`](validation/VALIDATION.md). Summary:

1. **Noise-free reconstruction** recovers the input to numerical tolerance
   for all three reconstructors (modal: 5e-16 rad; Hudgin zonal: 1.4e-14 rad;
   Fried zonal: residual matches the true phase's own waffle component to
   4.6e-4 rad — the honest claim for a geometry with a nontrivial null
   space).
2. **Reconstruction error vs photon flux** matches the analytic
   noise-propagation coefficient to within 2.2% across a 100x flux range
   (800 Monte Carlo trials per point).
3. **Dropout robustness curves**, baseline vs learned model, across both
   photon flux and subaperture dropout rate — see "AI model details" below.

## Benchmark results

See `MODEL_CARD.md` §7 for the full flux and dropout sweep tables. Headline:
the regularized least-squares baseline wins 9 of 10 tested operating points,
by a margin that grows to 7.4x at high flux; the learned ensemble's one win
is at 60% subaperture dropout, where the baseline's fixed regularization
strength becomes numerically unstable.

## AI model details

Full card: [`MODEL_CARD.md`](MODEL_CARD.md). Summary:

- **Baseline implemented and validated first**: `ModalReconstructor`
  (regularized least-squares), validated in `validation/validate_noise_free.py`
  and `validation/validate_photon_noise.py` before the learned model existed.
- **Dataset**: 100% synthetic, from Kolmogorov phase screens fitted to
  Zernike coefficients (`DATASET_CARD.md`), not committed, regenerated
  deterministically from a seed.
- **Training procedure**: single fixed operating point (flux 800, dropout
  0.25), 1800 samples, 24.1 s wall clock, 2 CPU cores, no hyperparameter
  search.
- **Test-split strategy**: disjoint RNG seeds from training, same generative
  process — in-distribution generalization only.
- **Metrics**: RMS Zernike-coefficient error, measured and tabulated across
  5 flux levels and 5 dropout rates against the baseline on identical data
  (`MODEL_CARD.md` §7).
- **Uncertainty output**: `predict(..., return_std=True)` returns the
  per-coefficient ensemble standard deviation. Measured calibration ratio
  0.57-0.76 throughout (i.e. the spread *understates* the true error by
  25-40% everywhere tested) — **not calibrated**, usable only as a coarse
  qualitative signal.
- **Failure cases**: enumerated in `MODEL_CARD.md` §9; every accuracy failure
  mode is silent (no exception, no reliable confidence-output signal).
- **Reproducibility**: exact commands and seeds in `MODEL_CARD.md` §10.
- **Compute**: 53.6 s total for the full two-sweep benchmark, 2 CPU cores.

> **This model is not certified for operational flight use.**

## Hardware requirements

CPU only; every script in this package (tests, validation, examples) runs
in well under 2 minutes on 2 shared CPU cores. No GPU, no accelerator, no
network access. Peak memory is dominated by the training batch
(a few thousand `float64` samples of size `~3 * n_sub`), a few MB.

## Limitations

- **The learned reconstructor loses to the regularized baseline at every
  tested operating point except one** (60% subaperture dropout), where the
  baseline's *fixed* regularization strength is the actual cause of its
  failure, not a fundamental limit of regularized least squares — a
  dropout-adaptive baseline (choosing `reg`, or the active-row count
  threshold, per sample) was not implemented and would be the fair next
  comparison before drawing a broader conclusion about ML robustness under
  dropout (see Roadmap).
- **The learned model's uncertainty output is not calibrated** (§ AI model
  details) and must not be used as a reconstruction covariance.
- **Kolmogorov screens under-represent low-order turbulent content**
  (no subharmonic compensation; `wavelab.screens` module docstring), which
  affects both the absolute magnitude of low-order Zernike coefficients in
  every synthetic sample and, by extension, any performance number that
  depends on the mix of low- vs high-order content in the test data.
  Anisoplanatism, scintillation, temporal evolution, and multi-layer
  turbulence are not modeled at all (single static equivalent-layer screen).
  See `DATASET_CARD.md` §5 for the complete list.
- **The photon-noise model is shot-noise-only** — no read noise, no
  background, no pixel-level spot physics (that is the scope of the related
  product ShackSim, P018, cited not duplicated); `wavelab.noise` module
  docstring.
- **The slope forward model is point-sampled at each subaperture centre**,
  not averaged over the subaperture area (`wavelab.zernike.zernike_slope_matrix`
  docstring) — exact only where the wavefront varies slowly across one
  subaperture, a standard simplification but one that would need revisiting
  for strongly aliased (small `r0`, coarse subaperture grid) regimes.
- **The Fried zonal reconstructor cannot recover the waffle mode by
  construction** — this is documented and validated (§ Engineering theory,
  Validation §1), not a bug, but any caller treating the Fried baseline as a
  complete phase reconstruction needs to know this null space exists.
- No obscured/annular pupil support in the dataset generator by default
  (`PupilGrid` supports an obscuration parameter, but `wavelab.dataset` does
  not use it), no atmospheric dispersion, no polychromatic effects.

## Safety statement

This software is research-grade. It is not flight-qualified, not certified,
and not approved for operational aerospace use.

## Roadmap

- Dropout-adaptive regularization for `ModalReconstructor` (scale `reg` or
  switch to TSVD with a row-count-aware threshold when the active-subaperture
  count drops), to test whether the one measured ML advantage survives a
  fairer baseline.
- Subharmonic compensation in `kolmogorov_screen` (Lane, Glindemann & Dainty
  1992) to remove the low-frequency deficit documented above.
- A convolutional or graph-structured learned architecture, contingent on a
  build environment with PyTorch or an equivalent framework available.

## License

Apache-2.0. See [`LICENSE`](LICENSE). Copyright © 2026 OPTIMA Organisation.

## Credits

This is under reserved rights obtained by OPTIMA Organisation.

## Citation

```
OPTIMA Organisation (2026). WaveLab: slope-to-phase wavefront reconstruction
toolkit (zonal Hudgin/Fried and modal Zernike least squares, plus a learned
slopes-to-Zernike ensemble). Version 0.1.0. Aerospace 100-Product Mission,
Product P014.
```
