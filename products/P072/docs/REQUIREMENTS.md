# Requirements — downlinkpick 0.1.0

Validation level 3. Each requirement names the check that demonstrates it and
the output file that records the measurement. A requirement with no check is not
a requirement, and is not listed.

Status vocabulary: **met** (a check demonstrates it), **met with a stated
limit** (a check demonstrates it within a documented boundary), **not met**
(deliberately out of scope, with the reason).

## Functional

| ID | Requirement | Status | Evidence |
|---|---|---|---|
| F-01 | Compute a contact window's deliverable capacity as the integral of its rate profile over the window, in bits. | met | `validate_capacity.py` checks 1, 2, 7; worst 8.384938e-09 relative against adaptive quadrature |
| F-02 | Invert that integral: given a bit count, return the contact time needed. | met | `validate_capacity.py` check 5; worst 9.574388e-13 relative over 25 points |
| F-03 | Select a subset of a recorded-data inventory for downlink over a contact plan, respecting per-window capacity, per-product expiry, no double downlink, and dependency precedence. | met | `validate_invariants.py` check 1; 0 violations in 4000 selections |
| F-04 | Provide a greedy baseline that an operator would recognise as the obvious procedure. | met | `downlinkpick.greedy.greedy_select`; two placement rules, documented in the module docstring |
| F-05 | Provide an exact solver that proves optimality on instances small enough to solve. | met with a stated limit | `validate_exact.py` check 1; optimal to within the solver's 1e-6 absolute MIP gap, which `SOLVER_ABS_GAP_VALUE_UNITS` records |
| F-06 | Score the greedy baseline against the exact optimum and report the gap as a distribution. | met | `validate_greedy_gap.py`; five families of 300 instances with deciles, percentiles, maximum and histogram |
| F-07 | Report the marginal value of the next unit of contact time, per window. | met with a stated limit | `validate_marginal.py` checks 3, 4; a finite difference with an explicit step, not a dual variable, and an upper estimate by a factor of 2.000000 at a 300 s step |
| F-08 | Report the identity of everything left behind, with a reason and a shortfall. | met | `validate_marginal.py` check 6; four reasons, whole-inventory accounting to 1.937e-16 relative |
| F-09 | Estimate a value for a product whose declared value is missing or stale. | met | `validate_valuemodel.py` checks 3, 4 |
| F-10 | Estimate a compression ratio for a product whose compressibility is unmeasured. | met | `validate_compress.py` checks 1, 4 |
| F-11 | Expose an uncertainty or confidence output on both learned components. | met with a stated limit | `MODEL_CARD.md`; a dispersion measure over the ensemble, not a calibrated interval, with measured coverage reported |
| F-12 | Provide a command-line interface. | met | `validate_cli.py`; 7 commands exit 0, 3 failure paths exit 2 with a message and no traceback |
| F-13 | Load an inventory and a contact plan from a file. | met | `downlinkpick.cases.load_case`; three shipped cases, round-trip tested |
| F-14 | Emit the MILP in a portable form for a reader with a different solver. | met | `python -m downlinkpick lp`; the PuLP LP file, checked term by term against the arrays actually solved |

## Verification

| ID | Requirement | Status | Evidence |
|---|---|---|---|
| V-01 | The exact solver's optimum must be checked against a route that contains no solver. | met | `validate_exact.py` check 1; exhaustive enumeration over 240 instances, worst 0.000000e+00 relative |
| V-02 | The capacity integral must be checked against an independent integration. | met | `validate_capacity.py` checks 1 and 2; two independent references with different error mechanisms |
| V-03 | The greedy gap must be reported as a distribution, not a mean. | met | `validate_greedy_gap.py`; the mean is reported last, after the deciles and the histogram |
| V-04 | Value conservation and no-double-downlink must be checked on the output of every selector. | met | `validate_invariants.py` check 1 |
| V-05 | The invariant checker itself must be shown to fail when an invariant is broken. | met | `validate_invariants.py` check 4; 6 of 6 deliberate corruptions detected |
| V-06 | Selection stability under perturbed value estimates must be measured. | met | `validate_stability.py`; Jaccard and retained value at seven noise levels |
| V-07 | Every learned component must be benchmarked against the classical baseline on the same held-out data and through the same solver. | met | `validate_valuemodel.py`, `validate_compress.py`; one arm changes per comparison |
| V-08 | Every validation script must run from any working directory. | met | verified from `/`, `/tmp`, the product root and `validation/`, all exit 0 |
| V-09 | No committed file may contain an absolute path. | met | `validate_cli.py` final check, plus a repository-wide grep before hand-off |
| V-10 | Every Hypothesis counterexample must be pinned as an explicit regression test. | met | `tests/test_regressions.py`; 15 pinned cases with literal inputs |

## Non-functional

| ID | Requirement | Status | Evidence |
|---|---|---|---|
| N-01 | No single training, sweep or benchmark may exceed three minutes on two cores. | met | longest measured step is the 48-curve stability sweep at 61.151 s |
| N-02 | Ruff-clean at line length 100 over `src/`, `tests/`, `examples/` and `validation/`. | met | `ruff check` passes |
| N-03 | Type hints on the public API and units in every public docstring. | met | module and function docstrings throughout `src/downlinkpick/` |
| N-04 | No `print` in library code. | met | `tests/test_report.py::test_no_print_in_the_library` scans every module but `__main__.py` |
| N-05 | Every dependency bounded on both sides at the measured version. | met | `pyproject.toml`; six runtime dependencies, each with a floor and a ceiling |
| N-06 | Deterministic: the same inputs and seeds reproduce the same outputs. | met | `tests/test_dataset.py`, `tests/test_greedy.py`, `tests/test_valuemodel.py`, and the byte-level case check in `regenerate_cases.py` |
| N-07 | Matplotlib must use the Agg backend and never call `show()`. | met | `examples/_common.py` sets it before any pyplot import |

## Deliberately not met

| ID | Requirement | Why not |
|---|---|---|
| X-01 | Model the link budget that produces the rate profile. | Out of scope. The profile is an input; a link budget needs an antenna pattern, an atmospheric model and a modcod table, none of which this package has. |
| X-02 | Measure a real compression ratio. | Out of scope and unnecessary: `numcodecs` does this and is named in the alternatives table. This package consumes a ratio or predicts one. |
| X-03 | Re-plan after a failed pass. | Out of scope. Reliability multiplies value in a single-shot expected-value objective; there is no recourse model and the limitation is stated in `README.md`. |
| X-04 | Order the transmission queue within a pass. | Out of scope. Precedence is by window, which is stated wherever precedence is defined. |
| X-05 | Support more than one transmit chain. | Out of scope. `ContactPlan` rejects overlapping windows for exactly this reason, with the assumption named in the error message. |
| X-06 | Beat OR-Tools CP-SAT on search. | Not attempted, and `README.md` says plainly that CP-SAT is the better solver. The contribution here is the contact-window and expiry accounting plus the left-behind report. |
