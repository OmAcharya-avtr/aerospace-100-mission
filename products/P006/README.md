# LinkBudgetX

**Status:** TESTING · **Class:** compact · **Validation level:** 1 (Educational) · **AI:** no

## Executive overview

`linkbudgetx` is a deterministic, unit-aware free-space optical (FSO)
link-budget library. It decomposes an optical link into transmit power,
optics efficiencies, geometric spreading loss (Gaussian or flat-top beam),
Gaussian-beam pointing loss, user-supplied atmospheric attenuation, and
receiver sensitivity, returning every intermediate term in dB and the link
margin. It adds first-order (delta-method) uncertainty propagation from
1-sigma input uncertainties to a sigma on `margin_db`, with a seeded Monte
Carlo cross-check. A YAML-driven CLI prints a formatted budget table or JSON.

## Aerospace problem

Sizing an optical communication link — terrestrial FSO, ground-to-air, or
inter-satellite — starts with a power budget: does enough light reach the
detector, with how much margin, and how sensitive is that margin to the
inputs you know least well (pointing jitter, atmospheric attenuation, laser
power calibration)? This package provides the standard first-order budget
arithmetic in a transparent, testable form for teaching and early trade
studies.

## Intended users

Students, educators, and engineers doing first-cut FSO feasibility or
sensitivity studies. Not a substitute for a professional link-analysis tool.

## Engineering theory

All losses are positive dB terms; the budget is

```
P_rx[dBm] = P_tx[dBm] − L_tx_optics − L_geo − L_point − L_atm − L_rx_optics
margin[dB] = P_rx[dBm] − S_rx[dBm]
```

(FSO budget structure per Majumdar & Ricklin, *Free-Space Laser
Communications: Principles and Advances*, Springer 2008; conceptually the
optical adaptation of Friis 1946, Proc. IRE 34.)

**Angle convention (explicit).** The input `beam_divergence_rad` is the
**full** divergence angle θ_full; for a Gaussian beam this is the full angle
between 1/e² intensity directions in the far field. All Gaussian formulas
below use the **half** angle θ_half = θ_full/2, matching
θ_half = λ/(π w₀) in Saleh & Teich, *Fundamentals of Photonics*, 2nd ed.,
Ch. 3 (Gaussian beam optics).

**Geometric spreading loss** (units: dB; far-field assumption
R ≫ Rayleigh range z_R = π w₀²/λ):

- Beam radius at range R: `w(R) ≈ θ_half·R` [m].
- Gaussian: centred circular aperture of radius a captures
  `f = 1 − exp(−2a²/w²)` (Saleh & Teich, 2nd ed., Ch. 3 — Gaussian power
  through a circular aperture). Small-aperture limit `f ≈ 2a²/w²`.
- Flat-top: uniform disc of radius θ_half·R; `f = min(1, (a/(θ_half·R))²)`.
- `L_geo = −10 log₁₀ f`.

**Pointing loss** (units: dB). The Gaussian far-field angular intensity is
`I(θ) = I₀ exp(−2θ²/θ_half²)`, so a static radial offset θ_err gives

```
L_point(linear) = exp(−2 θ_err² / θ_half²)
```

This is the classic `exp(−2θ_err²/θ_div²)` form quoted in FSO literature
(Majumdar & Ricklin 2008), **in which θ_div denotes the 1/e² half angle** —
since this library's input is the full angle, it uses
θ_half = `beam_divergence_rad`/2 inside the formula. Validity: receiver
small relative to the beam (a ≪ w); with large apertures, truncation and
offset couple and the multiplicative treatment is approximate. Applied to
the flat-top profile only as a first-order approximation.

**Atmospheric attenuation** (units: dB): `L_atm = α[dB/km] · R[km]`, with α
supplied by the user (e.g. from the visibility-based models of Kim, McArthur
& Korevaar, Proc. SPIE 4214, 2001). No turbulence/scintillation model.

**Uncertainty propagation** (units: dB). For independent 1-sigma input
uncertainties σᵢ,
`σ_margin² ≈ Σᵢ (∂margin/∂xᵢ)² σᵢ²` (JCGM 100:2008, Sec. 5.1), with partials
by central finite differences (one-sided at domain boundaries). **Linearity
assumption:** valid only when margin is nearly linear in each input over
±3σ. It breaks when the margin is curved across the sigma range — the
canonical case is pointing jitter about a **zero** nominal pointing error,
where the derivative vanishes and first order predicts zero contribution;
use `monte_carlo_margin()` there (demonstrated in validation Scenario 2).

## Architecture

```
src/linkbudgetx/
├── units.py        # dBm/W, dB/linear, length conversions (exact identities)
├── core.py         # LinkBudget dataclass, per-term losses, compute()
├── uncertainty.py  # propagate_margin_sigma(), monte_carlo_margin()
├── cli.py          # YAML config → table / JSON
└── __main__.py     # python -m linkbudgetx
```

Pure Python + numpy (Monte Carlo) + pyyaml (CLI). No cross-product imports.

## Installation

From the product root (`products/P006/`):

```bash
pip install .            # or: pip install -e .[dev]
```

Or without installing: `PYTHONPATH=src python -m linkbudgetx ...`

## Quick start

```python
from linkbudgetx import LinkBudget, propagate_margin_sigma

budget = LinkBudget(
    tx_power_dbm=20.0,
    wavelength_nm=1550.0,
    beam_divergence_rad=1.0e-3,   # FULL angle
    range_km=10.0,
    rx_aperture_diameter_m=0.1,
    rx_sensitivity_dbm=-40.0,
    tx_optics_efficiency=0.8,
    rx_optics_efficiency=0.8,
    pointing_error_rad=0.25e-3,
    atmos_attenuation_db_per_km=0.5,
    beam_profile="gaussian",
)
result = budget.compute()
print(result.format_table())          # every term in dB
print(result.margin_db)               # 13.900 dB

unc = propagate_margin_sigma(budget, {"tx_power_dbm": 0.5})
print(unc.sigma_margin_db)            # 0.500 dB (margin linear in Tx power)
```

## Configuration

CLI reads a YAML file mapping 1:1 onto `LinkBudget` fields plus an optional
`uncertainties` block (see `examples/example.yaml`):

```bash
python -m linkbudgetx --config examples/example.yaml        # table
python -m linkbudgetx --config examples/example.yaml --json # machine-readable
```

Exit code 2 with an actionable message on invalid input (negative range,
zero divergence, unknown keys, ...).

## Examples

Run from the product root; PNGs are written to `screenshots/`:

- `python examples/range_sweep.py` — 10 km terrestrial link, margin vs range
  (1–20 km) for clear air / haze / light fog → `screenshots/range_sweep.png`.
- `python examples/uncertainty_demo.py` — first-order Gaussian overlaid on a
  20 000-sample Monte Carlo margin histogram →
  `screenshots/uncertainty_histogram.png`.

## Validation

Level 1 evidence in `validation/VALIDATION.md` (raw script output committed
alongside; all numbers from running `validation/*.py` in this session):

- Flat-top geometric loss, 1 mrad full angle at 10 km, 0.1 m aperture:
  hand 40.000000 dB = library, diff 0.
- Gaussian geometric loss, same geometry: hand 36.990134 dB = library, diff 0
  (and Gaussian−flat-top gap = 3.010 dB ≈ 10 log₁₀ 2 small-aperture check).
- Pointing loss at θ_err = θ_full/2 (= half angle): hand 20 log₁₀ e =
  8.685890 dB = library; θ_err = 0 → exactly 0 dB.
- Full hand-summed budget: margin +13.900193 dB = library, diff 0.
- Monte Carlo cross-check (n = 20 000, seed 2026): first-order σ 0.7879 dB vs
  MC std 0.7850 dB → 0.37 % discrepancy (PASS, 5 % tolerance). Breakdown
  scenario at zero nominal pointing error: first-order 0.0000 dB vs MC
  0.1221 dB — expected failure of the linear method, documented.

## Benchmark results

Not applicable at compact/Level 1 beyond runtime: a single `compute()` is
microseconds; the 20 000-sample Monte Carlo cross-check runs in ~1 s on the
2-core build environment (well under the 3-minute budget).

## AI model details

Not applicable — this product contains no AI/ML components.

## Hardware requirements

Any machine running Python ≥ 3.11 with numpy and pyyaml; no GPU. Examples
additionally need matplotlib (Agg backend, no display required).

## Limitations

- Far-field only (R ≫ Rayleigh range); no near-field or defocus modelling.
- Pointing loss assumes a receiver small relative to the beam; truncation ×
  offset coupling is ignored for large apertures, and the Gaussian pointing
  form is applied to flat-top beams only as a first-order approximation.
- Atmospheric attenuation is a single user-supplied dB/km scalar; no
  turbulence, scintillation, beam wander, or pointing-jitter statistics.
- `wavelength_nm` is validated but informational: the model is parameterised
  by divergence, so wavelength does not currently enter the numbers.
- First-order uncertainty propagation assumes near-linearity over ±3σ and
  independent Gaussian inputs; it fails at curvature-dominated points (zero
  nominal pointing error — see validation Scenario 2). Monte Carlo redraws
  unphysical samples (truncated-Gaussian behaviour), so keep sigmas small
  relative to nominals.
- Educational validation level: hand calculations and internal consistency
  only; no comparison against measured link data or professional tools.

## Safety statement

This software is educational. It is not flight-qualified, not certified, and
not approved for operational aerospace use.

## Roadmap

- Diffraction-limited divergence helper θ_half = λ/(π w₀) tying
  `wavelength_nm` into the model.
- Pointing-jitter statistical loss (Rayleigh-distributed radial jitter).
- Optional Kim/Kruse visibility-to-attenuation helper.

## License

MIT — see `LICENSE`. Copyright © 2026 OPTIMA Organisation.

## Credits

This is under reserved rights obtained by OPTIMA Organisation.

## Citation

```
OPTIMA Organisation (2026). LinkBudgetX: deterministic free-space optical
link-budget library (v0.1.0) [Computer software]. Educational validation
level 1.
```
