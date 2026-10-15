# Validation evidence

Product P064 `regressselect` · version 0.1.0 · validation Level 2 ·
status TESTING

Research-grade software. Not flight-qualified, not certified, not approved for
operational aerospace use.

Every number below was produced by a script in this directory, run in the build
session of 2026-10-15, and the raw output of each run is committed beside the
script as `<script-name>_output.txt`. A check that failed is reported as
failed. Nothing was retuned to make a result come out better.

## Environment the numbers were measured at

From `validate_environment_output.txt`:

| item | value |
|---|---|
| Python | 3.13.16, CPython |
| platform | Linux-6.18.44-fc-v114-x86_64-with-glibc2.39, x86-64, 2 CPUs |
| numpy | 2.5.3 (declared `>=2.5.3,<2.6`) |
| scikit-learn | 1.9.1 (declared `>=1.9.1,<1.10`) |
| matplotlib | 3.11.2 (declared `>=3.11.2,<3.12`) |
| pytest | 9.1.1 (declared `>=9.1.1,<9.2`) |
| hypothesis | 6.169.3 (declared `>=6.169.3,<6.170`) |
| ruff | 0.17.0 (declared `>=0.17.0,<0.18`) |

Every range is bounded on both sides. The floor of each range **is** the
version executed, because the build environment forbids installing any other
version, so no wider floor is claimed. This is recorded as a limitation in
`README.md` rather than resolved by removing a bound.

## Scripts and what each one establishes

| script | what it checks | checks | failures |
|---|---|---|---|
| `validate_environment.py` | records interpreter and library versions against the declared ranges | record only | - |
| `validate_importgraph.py` | Baseline 1 selection and hop distances against a hand-written table | 19 | 0 |
| `validate_apfd.py` | APFD against eight hand-worked cases, four undefined-input rejections, and the fixture history against a no-information control | 13 | 0 |
| `validate_benchmark.py` | all three strategies on the committed history: APFD, safety loss, equal-cost comparison, Brier scores | 6 | 0 |
| `validate_mutation.py` | seeded faults executed against the bundled sample project; faults detected vs tests run with the full-suite upper bound | 5 | 0 |
| `validate_self_selection.py` | Baseline 1 on this repository's own suite, with and without pytest's conftest coupling | 4 | 0 |
| `validate_fixtures.py` | `git check-ignore -v` on every shipped fixture, cross-checked against `git ls-files --others --exclude-standard` | 29 | 0 |
| `validate_cli.py` | every CLI subcommand, including the three that must exit 2 | 12 | 0 |
| `worked_example.py` | the worked example the README quotes | output only | - |

Total: **88 checks, 0 failures.**

---

## 1. APFD, known-answer

Reference: G. Rothermel, R. H. Untch, C. Chu and M. J. Harrold, "Prioritizing
Test Cases For Regression Testing", IEEE Transactions on Software Engineering,
27(10), 929-948, October 2001, doi:10.1109/32.962562.

    APFD = 1 - (TF_1 + ... + TF_m) / (n * m) + 1 / (2 n)

Source: `validate_apfd_output.txt`. Tolerance 1e-12, two-sided.

| case | hand-worked | computed | abs diff | verdict |
|---|---|---|---|---|
| n=5 m=2, TF=(1,3) | 0.700000000000 | 0.700000000000 | 0.000e+00 | PASS |
| n=5 m=2, TF=(5,5) | 0.100000000000 | 0.100000000000 | 0.000e+00 | PASS |
| n=4 m=3, TF=(1,1,1) | 0.875000000000 | 0.875000000000 | 0.000e+00 | PASS |
| n=4 m=3, TF=(4,4,4) | 0.125000000000 | 0.125000000000 | 0.000e+00 | PASS |
| n=1 m=1, TF=(1,) | 0.500000000000 | 0.500000000000 | 0.000e+00 | PASS |
| n=10 m=1, TF=(1,) | 0.950000000000 | 0.950000000000 | 0.000e+00 | PASS |
| n=10 m=1, TF=(10,) | 0.050000000000 | 0.050000000000 | 0.000e+00 | PASS |
| n=24 m=4, TF=(1,2,3,4) | 0.916666666667 | 0.916666666667 | 0.000e+00 | PASS |

Four undefined inputs are rejected rather than substituted: an empty order, an
empty fault set, a fault detected by no test in the order, and an order with
duplicate test ids. The third matters: a fault that nothing detects has no
`TF_i`, the formula has no value for it, and the honest companion number is the
safety loss.

## 2. Baseline 1, import-graph reachability, known-answer

Source: `validate_importgraph_output.txt`. The sample project's import
structure was written out by hand from the source files and the computed
selection compared against it.

| changed module | selected test modules | of 7 | fraction |
|---|---|---|---|
| `sampleproj.vecops` | 3 | 7 | 0.4286 |
| `sampleproj.stats` | 2 | 7 | 0.2857 |
| `sampleproj.quat` | 2 | 7 | 0.2857 |
| `sampleproj.integrate` | 1 | 7 | 0.1429 |
| `sampleproj.orbit` | 1 | 7 | 0.1429 |
| `sampleproj.filters` | 1 | 7 | 0.1429 |
| `sampleproj.kinematics` | 1 | 7 | 0.1429 |
| `sampleproj` (package `__init__`) | 7 | 7 | 1.0000 |

All eight selections and all eleven hop distances match the hand-written table.
Mean over the eight single-module changes: 0.3214.

## 3. Baseline 1 on this repository's own suite, and the conftest gap

Source: `validate_self_selection_output.txt`. This is the check that keeps the
README's claim about Baseline 1 honest, and it produced a result that
contradicts the usual phrasing.

| measure | imports only | with conftest coupling |
|---|---|---|
| mean single-module selection fraction | 0.4015 | 0.6364 |
| maximum | 0.6364 | 1.0000 |
| source modules selecting the whole suite | 0 of 12 | 6 of 12 |

From the import text alone, no single-module change on this repository selects
the whole suite, so the common claim that a dependency-based selector is
near-useless because it selects almost everything **does not hold at this
scale**. It holds once pytest's `conftest.py` dependency is counted, which is a
real dependency and not a modelling choice: `tests/conftest.py` imports five of
the twelve source modules, and with those edges six of the twelve modules reach
every test module.

The same measurement is the soundness statement for Baseline 1: for
`regressselect.features`, the import-only selection **misses 0.5455** of the
test modules that actually depend on it, through fixtures. A selector built
from imports alone is not safe for a pytest suite.
`ImportGraph.with_conftest_coupling()` adds those edges.

## 4. Three-strategy benchmark on the committed fixture history

Source: `validate_benchmark_output.txt`. 320 revisions, chronological split at
0.70, 224 train and 96 held out, of which 37 have a real failure. Fixture
declares out-of-closure coupling 0.010 and flakiness 0.003 per test per
revision.

### APFD over the full order

| strategy | APFD |
|---|---|
| lexical control, no information | 0.703266 |
| **Baseline 1, dependency** | **0.836149** |
| learned selector | 0.814752 |
| Baseline 2, history | 0.795608 |

### Safety loss against budget

| strategy | budget | cost | safety loss | regressions caught |
|---|---|---|---|---|
| dependency | any | 0.4077 | 0.172072 | 0.8649 |
| history | 0.10 | 0.1250 | 0.740970 | 0.4865 |
| history | 0.25 | 0.2500 | 0.491549 | 0.7297 |
| history | 0.4167 | 0.4583 | 0.271396 | 0.8649 |
| history | 0.50 | 0.5000 | 0.232282 | 0.8919 |
| history | 0.75 | 0.7500 | 0.122523 | 0.9459 |
| history | 1.00 | 1.0000 | 0.000000 | 1.0000 |
| learned | 0.10 | 0.1250 | 0.633569 | 0.6486 |
| learned | 0.25 | 0.2500 | 0.484599 | 0.7027 |
| learned | 0.4167 | 0.4583 | 0.261411 | 0.8108 |
| learned | 0.50 | 0.5000 | 0.214264 | 0.8378 |
| learned | 0.75 | 0.7500 | 0.063063 | 0.9459 |
| learned | 1.00 | 1.0000 | 0.000000 | 1.0000 |

### Equal cost

| strategy | cost | safety loss | APFD |
|---|---|---|---|
| **dependency** | 0.4077 | **0.172072** | **0.836149** |
| learned | 0.4167 | 0.282883 | 0.814752 |
| history | 0.4167 | 0.299399 | 0.795608 |

### Dependency-baseline cost over all 320 revisions

mean 0.4066, min 0.1250, max 1.0000; 14 of 320 revisions (0.0437) select the
whole suite, all of them changes that touch `sampleproj/__init__.py`.

### Learned selector diagnostics

| quantity | value |
|---|---|
| Brier score, training slice | 0.025500 |
| Brier score, held-out slice | 0.061151 |
| combined importance of `distance_hops`, `inverse_distance`, `reachable` | 0.0061 |
| largest importance | `log1p_nearest_churn` 0.2891 |

## 5. Mutation-style evaluation, Level 2 requirement

Source: `validate_mutation_output.txt`. Fourteen declared single-line mutations
applied one at a time to a throwaway copy of the bundled sample project; the
full suite executed against each; failing nodes read from the junit XML with a
`tests` attribute against `testcase` element cross-check on every run.

Clean run: tests 24, failures 0, errors 0, skipped 0, testcase elements 24,
cross-check PASS.

| quantity | value |
|---|---|
| seeded faults | 14 |
| detected by the full suite | 10 |
| **full-suite detection upper bound** | **0.714286** |
| detected by no test | F04, F12, F13, F14 |
| committed `fault_matrix.json` reproduces this run | PASS |

### Faults detected against tests run, churn = 1 line (the truthful churn)

| strategy | budget | cost | detected / 14 | detected / 10 detectable | upper bound |
|---|---|---|---|---|---|
| **dependency** | any | **0.2589** | **0.7143** | **1.0000** | 0.7143 |
| history | 0.10 | 0.1250 | 0.2857 | 0.4000 | 0.7143 |
| history | 0.25 | 0.2500 | 0.3571 | 0.5000 | 0.7143 |
| history | 0.50 | 0.5000 | 0.5714 | 0.8000 | 0.7143 |
| history | 0.75 | 0.7500 | 0.7143 | 1.0000 | 0.7143 |
| learned | 0.10 | 0.1250 | 0.2857 | 0.4000 | 0.7143 |
| learned | 0.25 | 0.2500 | 0.5714 | 0.8000 | 0.7143 |
| learned | 0.50 | 0.5000 | 0.6429 | 0.9000 | 0.7143 |
| learned | 0.75 | 0.7500 | 0.7143 | 1.0000 | 0.7143 |

### Sensitivity: the same faults presented as 20-line changes

| strategy | budget | detected / 10 detectable, churn 1 | churn 20 |
|---|---|---|---|
| learned | 0.10 | 0.4000 | 0.9000 |
| learned | 0.25 | 0.8000 | 1.0000 |
| history | 0.25 | 0.5000 | 0.5000 |
| dependency | any | 1.0000 | 1.0000 |

The learned selector's detection moves with churn and the baselines' do not,
which is the measurement behind failure case 1 in `MODEL_CARD.md`.

Multi-fault APFD over a single order for a change touching every mutated
module: dependency 0.591667, learned 0.550000, history 0.525000.

## 6. Cold-clone fixture check

Source: `validate_fixtures_output.txt`. 29 checks, 0 failures. 22 required
fixtures confirmed present, not ignored, and present in
`git ls-files --others --exclude-standard`; 7 artifact paths confirmed ignored.

A measurement trap was found while writing this check and is recorded because
it is the same shape as the defect being guarded against: **`git check-ignore
-v` exits 0 whenever any pattern matches, including a negation.** Run on
`fixtures/sample_project/junit.xml` it exits 0 and prints
`!products/*/fixtures/**/junit.xml`, so an exit-status-only check reports the
required fixture as ignored. The verdict used here is "a pattern matched and it
is not a negation", cross-checked independently against the list of files git
would actually add.

| path | verdict | matching pattern |
|---|---|---|
| `fixtures/sample_project/junit.xml` | not ignored, git would add | `.gitignore:60:!products/*/fixtures/**/junit.xml` |
| `fixtures/faults/faults.json` | not ignored, git would add | no pattern matches |
| `fixtures/faults/fault_matrix.json` | not ignored, git would add | no pattern matches |
| `fixtures/history/history.json` | not ignored, git would add | no pattern matches |
| `fixtures/history/generation_parameters.json` | not ignored, git would add | no pattern matches |
| 7 sample-project source files, 7 test files, `pytest.ini`, `regenerate_fixtures.py` | not ignored, git would add | no pattern matches |
| `junit.xml` (product root) | ignored | `products/P064/.gitignore:12:/junit.xml` |
| `.coverage` (product root) | ignored | `products/P064/.gitignore:18:/.coverage` |
| `fixtures/sample_project/__pycache__/anything.pyc` | ignored | `products/P064/.gitignore:31:__pycache__/` |

A `__pycache__` fixture is unshippable rather than merely unwanted: the
repository root ignores `__pycache__/` at every depth and git cannot re-include
a file whose parent directory is excluded. The sample project needs no bytecode
fixture, and the mutation harness copies the project with `__pycache__`
excluded before every run.

## 7. Test suite

Counts parsed from `junit.xml` (`testsuite` element), cross-checked against a
count of `testcase` elements. Source: `pytest_output.txt`.

| quantity | value |
|---|---|
| tests | 140 |
| failures | 0 |
| errors | 0 |
| skipped | 0 |
| `testcase` elements | 140 |
| cross-check | PASS |
| wall clock | 18.71 s on 2 CPU cores |

Property-based tests use Hypothesis for the metric algebra: APFD closed form,
APFD monotonicity under moving a detecting test earlier, APFD invariance under
relabelling, safety-loss monotonicity in the selection, safety-loss bounds, and
detection-curve monotonicity. No property asserts bit-exact equality over a
floating-point pipeline and none asserts the equality of a boolean produced by
a strict threshold comparison; every comparison carries a two-sided tolerance.
`.hypothesis/` is not committed, so the three counterexamples found while
developing those properties are pinned as named regression tests with literal
inputs in `tests/test_properties.py`.

The published numbers in sections 4 and 5 are also pinned as regression
assertions in `tests/test_benchmark.py`, with a tolerance of 1e-6 for the two
deterministic baselines and 5e-3 for the learned selector, so a change that
moves a published number fails the suite instead of silently republishing a
different one. The assertion that the dependency baseline wins is among them.

## 8. What failed, and what was not done

- **No check failed in the final run.** Two checks failed during development
  and both were corrected by changing the claim, not the tolerance:
  `validate_self_selection.py` initially asserted that at least one module
  selects the whole suite on this repository, which is false from the import
  text alone; the script now measures the conftest coupling instead, and the
  narration was rewritten to state what was measured. `validate_fixtures.py`
  initially reported the committed `junit.xml` fixture as ignored, which was a
  defect in the check, not in the `.gitignore`.
- **Dependency ranges were executed at one point each**, their floor. The
  build environment forbids installing other versions, so the ceiling of each
  range is excluded by declaration and not by execution. Recorded as a
  limitation in `README.md`.
- **The history is synthetic.** No real repository history was mined, so every
  number in sections 4 and 5 is conditional on the generation model in
  `fixtures/history/generation_parameters.json`.
- **Calibration is not measured.** `SelectorConfig(calibrate=True)` exists and
  is exercised by the test suite, but no number here was measured with it and
  no calibration improvement is claimed.
- **Ranking inside a dependency-based selection is not implemented.** It is
  what the measured results point to as the useful configuration, and it is
  named as future work rather than claimed.
