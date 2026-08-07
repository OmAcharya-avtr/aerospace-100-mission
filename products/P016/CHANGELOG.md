# Changelog

All notable changes to ZernKit are recorded here.

## 0.1.0 — 2026-08-07

Initial release.

- `indexing`: exact integer maps Noll (1-based) <-> `(n, m)` <-> OSA/ANSI
  (0-based), with `noll_to_osa`/`osa_to_noll`, index validation, and
  traditional aberration names. No floating point, so no rounding failure at
  large `j`.
- `polynomials`: radial coefficients from the Born & Wolf factorial sum,
  `R_n^m`, full modes in polar and Cartesian coordinates, Noll/ANSI
  orthonormal and unnormalised (unit-peak) conventions, unit-disc grid helper.
- `gradients`: closed-form Cartesian `dZ/dx`, `dZ/dy` with the `1/rho`
  singularity removed at the coefficient level (exact at `rho = 0`, no `nan`),
  plus a Shack-Hartmann style point-sampled slope matrix.
- `fitting`: SVD least-squares wavefront fit with an explicit out-of-disc
  policy (`raise` / `drop` / `extrapolate`), condition-number reporting, and
  results labelled in both index conventions.
- `statistics`: analytic Kolmogorov Zernike coefficient variances, Noll
  residual variances `Delta_J` by direct summation, the large-`J` asymptote,
  and Noll's published `Delta_J` table as reference data only.
- CLI `python -m zernkit` with `index` and `noll-table` subcommands.
- Validation Level 1: orthonormality vs the analytic Kronecker delta (worst
  deviation 5.2e-14), twelve hand-evaluated low-order closed forms (worst
  2.2e-16), analytic gradients vs Richardson-extrapolated finite differences
  (worst 2.9e-11), and Noll residual variances vs the published table (worst
  0.954 %) with an independent structure-function cross-check of `Delta_1`
  agreeing to 0.033 %.
- Two example figures (mode gallery, wavefront fit and residual), 158 passing
  tests including Hypothesis property tests, ruff-clean.
