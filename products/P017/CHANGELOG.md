# Changelog

## 0.1.0 — 2026-08-07

Initial release.

- `KalmanFilter`: discrete linear Kalman filter with optional control
  input, per-step overrides of `F`/`Q`/`H`/`R` for time-varying systems,
  single-step `predict`/`update` and a batch `filter` returning prior and
  posterior histories, gains, innovations and per-step NIS.
- Joseph-form covariance update (`joseph_update`) used by the linear and
  extended filters, with the short form (`simple_update`) kept alongside
  for comparison and teaching. `symmetrize` returns a bit-exactly
  symmetric matrix; `covariance_health` reports asymmetry, extreme
  eigenvalues, trace and condition number.
- `steady_state`: fixed-point solution of the filtering algebraic Riccati
  equation, returning `P⁻_∞`, `P⁺_∞`, `K_∞` and the iteration count.
- `ExtendedKalmanFilter`: EKF taking user-supplied `f_jac`/`h_jac`, with
  `numerical_jacobian` (central differences, per-component step scaling)
  as a documented fallback carrying its accuracy caveat.
- `UnscentedKalmanFilter`, `MerweSigmaPoints`, `unscented_transform`:
  scaled unscented transform with configurable `alpha`/`beta`/`kappa`,
  weight convention stated explicitly in the module docstring and README,
  Cholesky failure reported as covariance collapse. Stores an effective
  transition matrix from the sigma-point cross-covariance so the linear
  RTS recursion reproduces the unscented RTS smoother.
- `rts_smooth`: Rauch-Tung-Striebel fixed-interval smoother consuming the
  output of any of the three filters.
- `models`: constant-velocity CWNA and DWNA models and the scalar random
  walk, all from Bar-Shalom, Rong Li & Kirubarajan 2001, Ch. 6.
- CLI: `python -m estimkit steady-state` and `python -m estimkit track`,
  with `--json` output and exit code 2 on invalid input.
- Examples: filter-vs-smoother tracking comparison and a UKF-vs-EKF
  comparison on long-range polar radar tracking, both saving PNGs to
  `screenshots/` with the Agg backend.
- Level 1 validation in `validation/VALIDATION.md`: hand-solved scalar
  Riccati equation, Kalata α–β closed form and a SciPy DARE cross-check
  for the 2-state constant-velocity case, smoother-vs-filter RMS over 300
  seeds, UKF↔KF reduction over an `(α, β, κ)` grid with the `eps/α²`
  round-off model, and covariance symmetry/positive-definiteness over a
  200 000-step run plus float32 and sub-optimal-gain stress cases.
- Hypothesis property tests for covariance symmetry/positive
  semi-definiteness, zero-measurement-noise state collapse, and affine
  exactness of the unscented transform.
