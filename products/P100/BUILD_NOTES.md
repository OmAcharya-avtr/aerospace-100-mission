# P100 SeedAudit — build state note

Running state, kept for resumption. Not a deliverable. The coordinator may
delete it before publishing; nothing else references it.

## Status: BUILD COMPLETE

Every deliverable is written, every validation script exits 0, and every
number quoted in `README.md` and `validation/VALIDATION.md` was re-executed
after the final code change.

## Measured, final

- Environment 2026-11-16: Python 3.13.18, numpy 2.5.3, scipy 1.18.1,
  matplotlib 3.11.2, pytest 9.1.1, hypothesis 6.170.15. `ruff` 0.16.9 on PATH
  and 0.17.5 as a module; clean under both over `src/ tests/ examples/
  validation/`.
- Tests: 181 passed, 0 failures, 0 errors, 0 skipped, counted from `junit.xml`
  and cross-checked against 181 `testcase` elements.
- PyPI 2026-11-16T04:17:23Z: `seedaudit` HTTP 404 (FREE), `numpy` HTTP 200
  (live control).
- Repository audit: headline **5683** seed-attributable moved numbers across
  103 scripts; **345** moving at the declared seed across 46 scripts; **45**
  excluded as the audit's own footprint; 229 of 229 candidates attempted;
  110 compared, 73 no_output, 37 run_failed, 9 timeout; 3337.1 s.
- Static screen: 3564 files parsed, 53 flagged, 275 findings, 0 parse failures.
- Baseline 1 at p = 1/44: measured 0.05136363636363636 against derived
  0.044421487603305776.

## What the build changed in the product, and why

Each was forced by a measurement, not by review.

1. Static screen over-flagged on name collisions (`str.replace`, `json.dump`)
   and on receivers that are not paths (`fig.savefig`). Fixed at the cause:
   441 findings fell to 275 and the reported paths became correct.
2. A vacuous path expression cannot carry a containment claim; such uses are
   skipped and counted as blind coverage.
3. The audit reported 38 findings that were its own perturbation echoed back by
   scripts printing their own seed. `is_instrument_footprint` was added and
   those are now excluded and reported separately.
4. The skeleton placeholder was NUL-delimited and made every committed output a
   binary file. Changed to printable `<#>` with the collision declared.
5. Running the classifier over the corpus misclassified an intrinsic rate-0.5
   script as self-polluting. The closed form `q(1-q)` was derived, tested and
   published; its worst case is 0.25 at every repeat budget.
6. Wilson intervals fail at a proportion of exactly 1; replaced with
   Clopper-Pearson, the same family as the product's own escaped-rate bound.
7. The percentile bootstrap is degenerate at zero events; exact intervals are
   reported beside it and are the pass criterion.
8. Calibration and detection mathematics were confounded in one check; split
   into derived@injected and derived@realised, with a declared Bonferroni
   family-wise criterion.
9. The fast path's statistical checks were flaky and made the working-directory
   probe report a sampling outcome as a path defect; they are now recorded as
   UNAVAILABLE on the fast path.

## Rules held to

- **No `git` command was run by this agent**, and no script in this product
  invokes one. `validate_tree_hygiene.py` re-implements the `.gitignore`
  subset it needs and says so; release-gate check 9 remains authoritative.
- Nothing was written outside `products/P100/` and temporary directories. No
  other product was modified.
- `products/P094/` excluded from the repository scan (concurrent agent);
  `products/P100/` excluded from its own scan and audited separately.
