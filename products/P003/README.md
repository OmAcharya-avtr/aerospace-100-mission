# ScintiNet

**Status:** TESTING · **Class:** medium · **Validation level:** 2 (Research) · **AI:** yes

## Executive overview

ScintiNet estimates optical scintillation — the fluctuation of received
irradiance caused by atmospheric turbulence — for free-space optical (FSO)
link planning. It provides three layers that can be compared against one
another on identical inputs:

1. **Analytic core** — weak-fluctuation Rytov theory with circular-aperture
   averaging (the classical baseline, implemented first).
2. **Split-step phase-screen simulator** — angular-spectrum wave-optics
   propagation through Kolmogorov phase screens, used as the data generator
   and validated against the analytic core.
3. **Learned surrogate** — a small MLP ensemble trained on simulation output,
   predicting the scintillation index with an uncertainty estimate ~5000×
   faster than running the simulation.

The headline validation result: the simulator reproduces Rytov theory to a
mean ratio of **0.980** for a point receiver across the weak regime, and the
surrogate **does not beat** the analytic baseline in-regime (RMSE log10
0.0781 vs 0.0429). Both numbers are reported as measured. See
[Benchmark results](#benchmark-results).

## Aerospace problem

Free-space optical links — ground-to-ground, ground-to-air, and
satellite-to-ground downlinks — carry high data rates but suffer deep,
fast irradiance fades from refractive-index turbulence along the path. The
scintillation index σ_I² = ⟨I²⟩/⟨I⟩² − 1 sets the required link margin, the
fade statistics, and ultimately the achievable availability.

Link planners face a gap: closed-form Rytov theory is instant but valid only
under weak fluctuations for idealised geometries, while wave-optics
simulation is general but far too slow for the parametric sweeps, Monte Carlo
availability studies, and optimisation loops that planning requires. ScintiNet
is a research vehicle for closing that gap with a learned surrogate — and for
honestly measuring whether the surrogate is actually worth it.

## Intended users

- Atmospheric-propagation and FSO researchers evaluating surrogate modelling
  for link-budget tools.
- Optical communications engineers building parametric link-planning studies
  who need a reproducible reference implementation of Rytov theory and a
  split-step simulator.
- Graduate students and instructors in atmospheric optics who want a
  documented, tested, runnable implementation of textbook results.

Not intended for operational link certification, availability guarantees, or
any flight or mission-critical decision.

## Engineering theory

All results assume a **Kolmogorov** refractive-index spectrum with no inner or
outer scale, a **horizontally homogeneous** path (constant Cn²), and **weak
fluctuations**.

### Rytov variance

Plane wave:

    σ_R² = 1.23 · Cn² · k^(7/6) · L^(11/6)

Spherical wave:

    β_0² = 0.50 · Cn² · k^(7/6) · L^(11/6)

- **Source:** L. C. Andrews and R. L. Phillips, *Laser Beam Propagation
  through Random Media*, 2nd ed., SPIE Press, 2005 — standard
  Kolmogorov-spectrum results.
- **Units:** Cn² [m^(−2/3)], k = 2π/λ [rad/m], L [m]; σ_R² dimensionless.
- **Assumptions:** Kolmogorov spectrum, constant Cn², paraxial propagation,
  first-order Rytov (weak scattering).
- **Validity range:** σ_R² < ~1 (best below ~0.5). Beyond this the Rytov
  approximation breaks down; σ_I² rises to a focusing peak near σ_R² ≈ 2–4
  and then saturates toward 1. ScintiNet does **not** model that regime.

### Weak-fluctuation scintillation index

In the weak regime the point scintillation index equals the Rytov variance:

    σ_I² = σ_R²

- **Source:** Andrews & Phillips 2005. **Validity:** σ_R² < ~1 only.

### Aperture averaging (circular aperture)

    A = [1 + 1.062 · k D² / (4 L)]^(−7/6),   σ_I²(D) = A · σ_R²

- **Source:** L. C. Andrews, "Aperture-averaging factor for optical
  scintillations of plane and spherical waves in the atmosphere,"
  *J. Opt. Soc. Am. A* **9**(4), 597–600, 1992; also Andrews & Phillips 2005.
- **Units:** D [m], L [m], k [rad/m]; A dimensionless, in (0, 1].
- **Assumptions:** plane-wave illumination, Kolmogorov spectrum, weak
  fluctuations, inner scale ≪ Fresnel scale √(L/k) ≪ outer scale.
- **Validity range:** A → 1 for D ≪ √(L/k) (point receiver) and decreases
  monotonically with D. This is an **approximation** to the exact
  aperture-averaging integral; a few-percent to ~10 % discrepancy against the
  exact form is expected, and ScintiNet's measured aperture bias
  (see Validation) partly reflects this. It is implemented for plane waves
  only — requesting it with `wave="spherical"` raises `ValueError`.

### Phase-screen synthesis

Kolmogorov phase power spectral density for a screen of thickness Δz:

    Φ_φ(κ) = 2π · k² · (Cn² Δz) · 0.033 · κ^(−11/3)

- **Source:** Andrews & Phillips 2005 (Kolmogorov phase spectrum); FFT
  synthesis recipe per J. D. Schmidt, *Numerical Simulation of Optical Wave
  Propagation with Examples in MATLAB*, SPIE Press, 2010.
- **Units:** κ [rad/m], Cn²Δz [m^(1/3)]; Φ_φ [rad² m²].

### Angular-spectrum propagation

    H(κ) = exp[ −i (κ_x² + κ_y²) Δz / (2k) ]

- **Source:** J. W. Goodman, *Introduction to Fourier Optics*, 3rd ed., 2005;
  Schmidt 2010. **Assumptions:** paraxial (Fresnel) approximation, scalar
  field. The operator is unitary — total energy Σ|U|² is preserved exactly,
  which is asserted in the test suite to rel 1e-12.

### Sampling rules (enforced at run time)

The simulator raises `ValueError` on violation of any of:

1. **Fresnel resolution:** dx ≤ √(λL)/4 — the speckle scale √(λL) must span
   at least 4 samples.
2. **Domain size:** N·dx ≥ 4√(λL) — the grid must hold several speckle cells
   for meaningful statistics.
3. **Screen phase resolution:** r₀ ≥ 2·dx per screen, where
   r₀ = (0.423 k² Cn² Δz)^(−3/5) is the plane-wave Fried parameter.

Additionally, requested aperture diameters must not exceed grid_width/4.

## Architecture

```
src/scintinet/
├── rytov.py       analytic core: rytov_variance, aperture_averaging_factor,
│                  scintillation_index_weak  (no dependencies beyond numpy)
├── simulator.py   SimParams, SimResult, kolmogorov_phase_screen,
│                  angular_spectrum_propagate, simulate_scintillation
└── surrogate.py   Surrogate (MLP ensemble + uncertainty), rytov_baseline
```

Dependency direction is strictly one-way: `surrogate` → `rytov`,
`simulator` → `rytov`. The analytic core has no knowledge of the simulator or
the ML layer, so the baseline can never be contaminated by the model it is
benchmarking. No cross-product imports.

Data flow: `run_campaign.py` → `validation/dataset.csv` → `Surrogate.fit` →
benchmark against `rytov_baseline` on held-out rows.

## Installation

Requires Python 3.11+, numpy, scikit-learn (matplotlib for the examples).

```bash
cd products/P003
pip install -e .            # or: pip install -e ".[examples]"
```

No installation is strictly necessary — the tests and scripts add `src/` to
`sys.path` themselves. Nothing outside the pre-installed environment is
needed.

## Quick start

```python
from scintinet import rytov_variance, scintillation_index_weak
from scintinet import SimParams, simulate_scintillation, Surrogate

# 1. Analytic: 2 km link at 1550 nm through Cn2 = 1e-15 m^(-2/3)
sigma_r2 = rytov_variance(1e-15, 1.55e-6, 2000.0)          # 0.070950
sigma_i2 = scintillation_index_weak(1e-15, 1.55e-6, 2000.0,
                                    aperture_diameter=0.1)  # 0.008164

# 2. Wave-optics simulation of the same link (~1.3 s)
params = SimParams(cn2=1e-15, wavelength=1.55e-6, path_length=2000.0,
                   aperture_diameters=(0.1,), grid_size=256,
                   grid_width=0.5, n_screens=8, n_realizations=8)
result = simulate_scintillation(params, seed=42)
print(result.sigma_i2_point)          # 0.063972
print(result.sigma_i2_aperture[0.1])  # aperture-averaged

# 3. Surrogate with uncertainty (columns: Cn2, L, lambda, D)
import csv
import numpy as np

rows = list(csv.DictReader(open("validation/dataset.csv")))
X_train = np.array([[float(r["cn2"]), float(r["path_length_m"]),
                     float(r["wavelength_m"]), float(r["aperture_d_m"])]
                    for r in rows])
y_train = np.array([float(r["sigma_i2_sim"]) for r in rows])

surrogate = Surrogate(n_members=5, random_state=0).fit(X_train, y_train)
mean, std = surrogate.predict(np.array([[1e-15, 2000.0, 1.55e-6, 0.1]]),
                              return_std=True)
```

Reminder: in this regime `rytov_baseline` is more accurate than the
surrogate — see [Benchmark results](#benchmark-results).

## Configuration

`SimParams` (all units SI):

| Field | Default | Meaning |
|---|---|---|
| `cn2` | — | Refractive-index structure parameter [m^(−2/3)], ≥ 0 |
| `wavelength` | — | Wavelength [m], > 0 |
| `path_length` | — | Path length L [m], > 0 |
| `aperture_diameters` | `()` | Receiver diameters [m] to evaluate |
| `grid_size` | 256 | Samples per side (power of 2 recommended) |
| `grid_width` | 0.5 | Physical grid side [m] |
| `n_screens` | 8 | Phase screens (applied at segment midpoints) |
| `n_realizations` | 8 | Independent turbulence realizations averaged |

`Surrogate`: `n_members` (≥ 2, default 5), `hidden_layer_sizes`
(default `(32, 32)`), `max_iter` (default 2000), `random_state` (default 0;
member *i* uses `random_state + i`).

Cost scales as `n_realizations × n_screens × grid_size² log(grid_size)`.
The defaults are sized for the 2-core, sub-3-minute budget.

## Examples

Both scripts run standalone and write PNGs to `screenshots/`.

```bash
python examples/sweep_sigma_i2.py         # ~11 s
python examples/phase_screen_speckle.py   # ~5 s
```

- **`screenshots/sweep_sigma_i2.png`** — σ_I² versus Rytov variance σ_R²
  across a 7-point Cn² sweep at λ = 1550 nm, L = 2000 m: split-step
  simulation, analytic theory, and the surrogate with its ±2σ ensemble band.
  Simulation points track the theory line closely and fall slightly below it
  at the top of the range.
- **`screenshots/phase_screen_speckle.png`** — one Kolmogorov phase screen
  (Cn²Δz = 2.5e-13 m^(1/3)) beside the resulting intensity speckle field
  after 2 km through 8 screens (σ_I² = 0.074), showing the characteristic
  Fresnel-scale irradiance structure.

## Validation

Full evidence, criteria and raw script outputs: **`validation/VALIDATION.md`**,
with `sim_vs_theory.txt`, `benchmark_results.txt`, `campaign_log.txt` and
`dataset.csv` committed alongside. Every number below came from running those
scripts in this build session.

| ID | Check | Result |
|---|---|---|
| V1 | Analytic core vs textbook closed forms (5 hand-checked known answers, rel 1e-4) | PASS |
| V2 | Simulator sanity: energy conservation (rel < 1e-12), zero turbulence → σ_I² < 1e-12, seeded reproducibility | PASS |
| V3 | Simulated σ_I² vs Rytov theory, point receiver, 18 sweep points | PASS — mean ratio 0.980 (0.907–1.041) |
| V3b | Simulated σ_I² vs Rytov + Andrews aperture averaging, 36 points | Reported: mean ratio 0.850 (0.689–1.036) |
| V4 | Surrogate vs analytic baseline, 14 held-out points | Baseline wins — reported as measured |

The pass criterion for V3 was fixed before the run (point-like mean ratio
within [0.6, 1.4], every point within [0.5, 1.6]) and is stated in the
validation script itself.

The V3b aperture bias is a genuine, documented limitation: FFT phase screens
without subharmonics lack sub-fundamental spatial frequency power, which most
affects the large-scale intensity structure that survives aperture averaging;
the Andrews factor is itself an approximation. The two contributions cannot
be separated with this dataset. Tolerances were not adjusted to hide it.

Test suite: **50 passed, 0 failed, 0 skipped** (`python -m pytest tests/ -q`,
8.2 s), including Hypothesis property tests, an end-to-end integration test
(generate → train → predict), and a fixed-seed regression test
(σ_I² = 0.0639720889 at seed 42) that guards the screen and propagator
normalisation.

## Benchmark results

Surrogate versus analytic baseline on the same 14 held-out simulation points
(54-row dataset, 40/14 shuffle split, seed 0):

| Model | RMSE (log10 σ_I²) | median \|rel err\| | max \|rel err\| |
|---|---|---|---|
| MLP surrogate (5-member ensemble) | 0.0781 | 0.1665 | 0.2824 |
| **Rytov analytic baseline** | **0.0429** | **0.0700** | **0.2276** |

**The analytic baseline wins on every metric.** This is the honest outcome
and it is not a defect: the benchmark runs entirely inside the baseline's own
validity regime, where Rytov theory is a near-exact closed form, and 40
training rows carrying ~5–10 % statistical noise cannot support a network
that rediscovers that closed form more accurately. The surrogate's 16.7 %
median error is comparable to the noise on its own training targets.

Where a surrogate *does* earn its place: regimes with no closed form (strong
fluctuations, non-Kolmogorov spectra, slant paths with Cn²(h) profiles,
Gaussian beams), and speed against **simulation** rather than against algebra
— 0.269 ms per surrogate prediction versus ~1.3 s per split-step point here,
a ~5000× speedup that makes Monte Carlo availability studies feasible.
In-regime for a horizontal Kolmogorov plane-wave link, use `rytov_baseline`.

Runtime budget (2 CPU cores): campaign 22.6 s, surrogate fit 2.0 s, test
suite 8.2 s, examples ~16 s. All well inside the 3-minute limit.

## AI model details

Full detail in **`MODEL_CARD.md`** and **`DATASET_CARD.md`**.

- **Baseline (implemented first):** `rytov_baseline` — aperture-averaged
  weak-fluctuation Rytov index, benchmarked on the identical held-out split.
- **Architecture:** 5-member ensemble of scikit-learn `MLPRegressor`,
  hidden (32, 32), L-BFGS, `StandardScaler` pipeline. Features
  `[log10 Cn², log10 L, log10 λ, D]`; target `log10 σ_I²` (exponentiated back,
  guaranteeing positive output). ~1200 parameters per member.
- **Dataset:** 54 rows from a **reduced-scale** seeded simulation campaign
  (256² grid, 8 screens, 8 realizations, 22.6 s). Weak regime only
  (σ_R² ≤ 0.30), plane wave, horizontal homogeneous path. Regenerate with
  `python validation/run_campaign.py` (seeds 2026 + i).
- **Training:** 40 train / 14 test shuffle split, seed 0; fit time 2.0 s. No
  hyperparameter search against the test set.
- **Test split caveat:** the three aperture rows from one simulation run share
  a seed and are correlated, so the random row split leaks mildly. A group
  split by simulation point would be stricter. Documented, not corrected.
- **Metrics:** see [Benchmark results](#benchmark-results). The baseline wins.
- **Uncertainty output:** `Surrogate.predict(X, return_std=True)` returns the
  ensemble-member standard deviation (deep-ensemble style; Lakshminarayanan
  et al., NeurIPS 2017). Measured 2.97e-03 mean std against 1.77e-02 mean
  prediction (≈17 %), tracking the 16.7 % median error. This is epistemic
  disagreement only — **not a calibrated predictive interval**, and not to be
  used as a confidence bound for link margin sizing.
- **Failure cases:** silent extrapolation outside the training box
  (Cn² > 1e-15, L > 3000 m, λ outside 850–1550 nm); strong fluctuations
  (σ_R² > 1) where the true σ_I² peaks and saturates; non-plane-wave sources;
  slant paths; inherited simulator low-bias on finite apertures;
  unconstrained interpolation between the three trained D values.
- **Reproducibility:** every command and seed is listed in `MODEL_CARD.md`;
  identical `(data, random_state)` gives bit-identical predictions.

**This model is not certified for operational flight use.**

## Hardware requirements

- CPU only. Developed and validated on 2 cores; no GPU, no PyTorch.
- Peak memory < 500 MB (dominated by 256² complex FFT working arrays).
  A 1024² grid would need roughly 16× that per array.
- Disk: < 1 MB including the committed dataset and PNGs.
- Python 3.11 with numpy and scikit-learn; matplotlib for examples only.

## Limitations

1. **Weak-fluctuation regime only.** Everything here — theory, simulation
   validation, training data, surrogate — is confined to σ_R² ≲ 1
   (max 0.30 in the dataset). The focusing peak and saturation regime are not
   modelled and will be silently wrong if queried.
2. **No subharmonics in the phase screens.** FFT synthesis provides no power
   below 2π/(N·dx). Measured consequence: σ_I² ~2 % low (point) and ~15 % low
   (50–100 mm apertures). Do not use this simulator for beam-wander, tilt, or
   long-exposure phase statistics, which are dominated by exactly the
   missing low frequencies.
3. **Plane wave only.** No Gaussian-beam or spherical-wave simulation; the
   analytic core has spherical-wave σ_R² but aperture averaging is
   plane-wave only.
4. **Horizontal homogeneous path only.** Constant Cn²; no Cn²(h) altitude
   profile, so no slant-path or satellite downlink capability.
5. **Kolmogorov spectrum with no inner or outer scale.** No
   von Kármán / modified-atmospheric-spectrum options.
6. **Reduced-scale dataset (54 rows).** Not converged; ~5–10 % statistical
   noise per target from only 8 realizations. See `DATASET_CARD.md` for what
   a full campaign would require.
7. **Surrogate does not beat the analytic baseline in-regime** and has no
   extrapolation guard rails.
8. **Ensemble std is not calibrated** as a predictive interval.
9. **No experimental validation.** Nothing here has been compared against
   measured scintillometer or field-trial data. Level 2 (Research) evidence
   is analytic and self-consistency only.
10. **No temporal modelling.** No wind, frozen flow, fade duration, fade rate,
    or outage-probability statistics.
11. **No CLI.** The spec names no CLI for this product; use the Python API
    and the committed scripts.

## Safety statement

This software is research-grade. It is not flight-qualified, not certified,
and not approved for operational aerospace use.

## Roadmap

- Subharmonic phase-screen augmentation (Lane et al. 1992) to correct the
  low-frequency deficit and the finite-aperture bias.
- Gaussian-beam and spherical-wave sources; exact aperture-averaging integral
  alongside the Andrews approximation.
- Extension into the strong-fluctuation regime with comparison against the
  Andrews–Phillips gamma-gamma / effective-spectrum models — the regime where
  a surrogate has a real accuracy argument.
- Cn²(h) altitude profiles (e.g. Hufnagel–Valley) for slant and downlink
  geometries.
- Converged campaign at 1024² with 500+ realizations per point and a
  group-wise train/test split, plus surrogate recalibration.
- Temporal frozen-flow simulation for fade-duration and outage statistics.

## License

Apache-2.0. See `LICENSE`. Copyright © 2026 OPTIMA Organisation.

## Credits

This is under reserved rights obtained by OPTIMA Organisation.

Theory references: L. C. Andrews and R. L. Phillips, *Laser Beam Propagation
through Random Media*, 2nd ed., SPIE Press, 2005; L. C. Andrews,
*J. Opt. Soc. Am. A* **9**(4), 597 (1992); J. D. Schmidt, *Numerical
Simulation of Optical Wave Propagation with Examples in MATLAB*, SPIE Press,
2010; J. W. Goodman, *Introduction to Fourier Optics*, 3rd ed., 2005;
B. Lakshminarayanan, A. Pritzel and C. Blundell, NeurIPS 2017 (deep
ensembles).

## Citation

```bibtex
@software{scintinet_2026,
  title   = {ScintiNet: learned scintillation surrogate for FSO link planning},
  author  = {{OPTIMA Organisation}},
  year    = {2026},
  version = {0.1.0},
  license = {Apache-2.0},
  note    = {Research-grade software; not certified for operational aerospace use}
}
```
