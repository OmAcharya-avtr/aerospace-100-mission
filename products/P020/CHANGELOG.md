# Changelog

## 0.1.0 — 2026-08-07

Initial release.

- Turbulence metrics from any Cn²(h) profile, each with its weighting integral,
  units, assumptions and validity range in the docstring: Fried parameter r₀
  (plane and spherical wave, uplink and downlink geometry), isoplanatic angle
  θ₀, Greenwood frequency f_G, slant-path Rytov variance (plane and spherical),
  weak-regime scintillation index, effective turbulence height h̄ and seeing
  FWHM.
- Zenith-angle dependence applied as an explicit, stated power of sec(ζ) per
  quantity (−3/5 for r₀, −8/5 for θ₀, +3/5 for f_G, +11/6 for the Rytov
  variance and scintillation index), never folded into a coefficient. The
  exponents are exported machine-readably as `EXPONENT_SEC_ZENITH` and
  `EXPONENT_WAVELENGTH`.
- Coefficients with their derivations in `constants.py`: 6.883877 =
  2(24/5·Γ(6/5))^(5/6); 0.423 = 2.914/6.883877; 0.3141308 = (0.423/2.914)^(3/5);
  2.30662 = 0.102^(3/5)(2π)^(6/5); 2.25 with its 1.23 / 0.50 / 0.40
  horizontal-path consequences. Every relation is asserted by a test.
- Profile models with source and stated validity: Hufnagel-Valley (free v and
  A), HV 5/7, SLC-Day, SLC-Night, homogeneous slab, and log-linear tabulated
  profiles from user samples. Wind models: Bufton, constant, tabulated, plus
  `rms_upper_wind`.
- Quadrature: adaptive Gauss-Kronrod integrating panel by panel between the
  profile's declared breakpoints, with the integrand rescaled to O(1) so
  QUADPACK's absolute tolerance is meaningful; fixed composite Simpson for grid
  refinement; `grid_convergence()` helper returning per-grid change and error.
- Input validation with actionable messages: wavelength restricted to the
  100 nm – 20 µm optical/IR band, zenith angle in [0, π/2) with a
  flat-Earth warning above 60°, non-negative and strictly increasing tabulated
  heights, positive Cn², integration limits inside each model's stated validity
  range, and a `UserWarning` when σ_R² ≥ 1 leaves the weak-fluctuation regime.
- CLI: `python -m atmoprofile summary --profile hv57 --wavelength-nm 500
  --zenith-deg 0 30 45 60 [--json]` and `python -m atmoprofile profile`.
- Level-2 validation with saved evidence (`validation/`): constant-slab closed
  forms vs hand arithmetic (≤1e-15 relative), zenith exponents recovered by a
  blind log-log fit (≤1e-9), HV 5/7 reproducing its named 5 cm / 7 µrad,
  standard-model r₀ at 500 nm and 1550 nm against the literature band, and a
  grid-refinement convergence study. One check is reported as FAILED and not
  tuned: the Bufton rms wind (22.96 m/s) does not match the 21 m/s HV
  pseudowind.
- 117 pytest tests (known answers with hand arithmetic in comments, Hypothesis
  property tests for the wavelength/zenith/amplitude scaling laws, input
  validation, quadrature-convergence regressions, CLI); ruff-clean.
- Two runnable examples saving PNGs to `screenshots/` with the Agg backend.
