# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-08

First release. Status: TESTING. Validation level 2.

### Added

- `Polytope`, an H-representation convex polyhedron with exact support-function
  evaluation by linear programme, a closed form for axis-aligned boxes, LP-based
  redundant-row removal, brute-force vertex enumeration with an explicit work
  budget, and Qhull volumes.
- Set algebra: `intersect`, `pontryagin_difference` (exact in halfspace form),
  `minkowski_sum` (via vertex enumeration and convex hull), `is_subset` (exact,
  by support function on facet normals) and `support_gap`.
- `maximal_robust_invariant_set`: the one-step-set recursion with a declared
  convergence criterion, an iteration cap, and an explicit `iteration_cap`
  termination that reports non-convergence instead of returning the last
  iterate as if it were the answer.
- `verify_robust_invariance`: an independent exact facet-by-facet check of the
  invariance condition, used to validate every computed set.
- `pre_set`, handling the degenerate rows that a singular system matrix
  produces.
- Diagnostics: `growth_table` (per-iteration row, facet and vertex counts) and
  `tolerance_sweep` (the same problem solved once per redundancy tolerance,
  with the answers compared).
- Eight built-in example systems, four of them with hand-computable answers.
- CLI `python -m invariantset` with `systems`, `invariant`, `tolerance-sweep`
  and `algebra` subcommands, text and JSON output, and a non-zero exit status
  when the recursion does not converge.
- Five runnable examples, each writing a PNG into `screenshots/`.
- Seven validation scripts with their raw output committed beside them.
- 200 tests: unit, input validation, known-answer with the hand arithmetic in
  the test comments, edge cases, Hypothesis property tests for the set algebra
  identities, diagnostics regressions and CLI subprocess tests.

### Known limitations at this release

See the Limitations section of README.md. The ones that bite first: brute-force
vertex enumeration is `C(m, n)` and is refused above `max_bases`; the recursion
has no finite-termination guarantee and two of the shipped systems do not
converge at any practical cap; and the redundancy tolerance changes the computed
answer on the shipped attitude loop between 3e-3 and 4e-3.
