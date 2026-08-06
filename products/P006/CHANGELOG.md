# Changelog

## 0.1.0 — 2026-08-01

Initial release.

- `LinkBudget` dataclass with unit-explicit field names; `compute()` returns
  every intermediate term in dB plus `margin_db`.
- Gaussian and flat-top geometric spreading loss (far-field), centred-aperture
  capture fraction, Gaussian pointing loss (half-angle convention documented),
  user-supplied dB/km atmospheric attenuation, optics efficiencies, margin
  against receiver sensitivity.
- First-order (delta-method) uncertainty propagation to `margin_db` via
  numerical partial derivatives, with per-input contributions; Monte Carlo
  cross-check helper (`monte_carlo_margin`, seeded, reproducible).
- CLI: `python -m linkbudgetx --config example.yaml` (table) and `--json`.
- Examples: range sweep (margin vs range PNG), uncertainty histogram vs
  Monte Carlo PNG.
- Level 1 validation: hand-calculated known-answer cases and Monte Carlo
  cross-check documented in `validation/VALIDATION.md`.
