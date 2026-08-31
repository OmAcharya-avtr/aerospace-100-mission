# Changelog

All notable changes to WahbaKit are recorded here.

## 0.1.0 — 2026-08-31

Initial release.

- `conventions`: the single definition of every convention the package uses —
  scalar-first Hamilton quaternions, `dcm_from_quat` matching
  `scipy.spatial.transform.Rotation.as_matrix` to 5.6e-16, Shepperd's
  matrix-to-quaternion extraction with all four branches, canonical `w >= 0`
  sign, and the body-frame attitude error `log(A_est A_true^T)`.
- `observations`: `VectorObservations` with per-observation sigmas [rad],
  inverse-variance weights, the Wahba attitude profile matrix, and an
  observability metric evaluated in both the body and the reference frame.
  `DegenerateObservationsError` (a `ValueError`) is raised below
  `lambda_min = 1e-6`, which is a 0.1146 deg separation for a pair.
- `triad`: Black's (1964) two-observation algorithm, with the primary
  observation reproduced exactly and the choice of primary exposed.
- `davenport`: the q-method as a 4x4 symmetric eigenproblem, with the
  scalar-last Shuster convention converted at the boundary and the eigenvalue
  gap reported.
- `quest`: the characteristic quartic, Newton from `lambda_0 = 1` (2 to 5
  iterations observed), and the adjugate closed form, agreeing with the
  q-method eigenvector to 1.5e-14 rad over 500 random problems. Shuster's
  method of sequential rotations is on by default and removes the 180 deg
  singularity (1.1e-06 rad -> 2.7e-17 rad).
- `olae`: Mortari, Markley and Singla's (2007) Cayley-transform linear
  estimator, with the same sequential-rotation remedy. Its first-order
  departure from the Wahba optimum is measured, not assumed: 0.18 sigma on the
  four-observation geometry in `validation/`.
- `covariance`: the Cramer-Rao attitude covariance and the asymmetric TRIAD
  covariance, both in rad^2 and both matched against 10 000-trial seeded Monte
  Carlo to 1.6 % and 1.1 % against a 1.4 % sampling error.
- `solve`: one dispatcher, `solve_wahba(obs, method, with_covariance=)`.
- CLI `python -m wahbakit` with `demo` and `conventions` subcommands.
- Validation Level 1: four scripts with committed raw output, covering the
  conventions and frame order, four-method agreement, covariance against Monte
  Carlo, and near-parallel geometry. Two checks failed on their first
  formulation and are documented in `validation/VALIDATION.md` section 5 with
  the reason the reference, not the library, was corrected.
- Three example figures, 145 passing tests including Hypothesis property tests
  for rotation invariance in both frames, orthogonality, quaternion norm and
  relabelling invariance, ruff-clean.
