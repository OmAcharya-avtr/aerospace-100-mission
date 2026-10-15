# Validation evidence — hazardtrace 0.1.0

**Validation level 2 (research grade).** Every number in this document was
produced by running the script named beside it, in this container, on
2026-10-15, and the raw output of each run is committed next to the script as
`validation/<script-name>_output.txt`. No number here was typed from memory or
carried over from an earlier run.

**What a clean run means.** The analysis is fully drawn and the identifiers it
cites resolve. It is not a statement that any system is safe, not a statement
that the hazard analysis is correct or adequate, and it discharges no safety
obligation of any kind. This software is research-grade. It is not
flight-qualified, not certified, and not approved for operational aerospace use.

**Method references.** STPA Handbook, Leveson and Thomas, March 2018 (the four
analysis steps, the control-structure model, the four UCA classes). Leveson,
*Engineering a Safer World: Systems Thinking Applied to Safety*, MIT Press,
2011, ISBN 978-0-262-01662-9. Leveson, "A new accident model for engineering
safer systems", *Safety Science* 42(4), 2004, pp. 237–270,
doi:10.1016/S0925-7535(03)00047-X. Thomas, "Extending and automating a
systems-theoretic hazard analysis for requirements generation and analysis",
PhD thesis, MIT, 2013 (formal UCA context tables, which this release does
**not** implement; see Limitations).

## How to reproduce every number

From the repository root, with `pip install -e ".[test]"`:

```bash
python -m pytest tests/ -q --junitxml=junit.xml
cd validation
python validate_environment.py
python validate_known_answers.py
python validate_enumeration.py
python validate_parser_rejections.py
python validate_crosslinks.py
python validate_cli.py
python validate_properties.py
cd ..
python validation/worked_example.py
```

`worked_example.py` is run from the repository root because the code block it
contains — the one quoted in README.md — uses repository-relative paths on
purpose. Each script exits 0 when every check in it passed and 1 otherwise.

## 1. Test suite

Counted from the junit XML, never from pytest's summary line. A suite that
collects zero tests prints success and exits 0; that defect shipped elsewhere in
this portfolio and the count below is parsed from the `testsuite` element and
cross-checked against a count of `testcase` elements.

| quantity | value |
|---|---|
| `testsuite/@tests` | 215 |
| `testcase` elements counted | 215 |
| cross-check | MATCH |
| failures | 0 |
| errors | 0 |
| skipped | 0 |
| wall clock | 15.49 s on 2 contended cores |

No test is marked xfail, and no test is skipped.

## 2. Known answers on hand-constructed cases

`validation/validate_known_answers.py`

`examples/cases/planted_gaps.yaml` was built to contain exactly one instance of
each of the sixteen checks. The expected code-to-subject mapping is written out
in the script and in `tests/test_known_answers.py`, so the answer is known by
construction rather than read off a run.

| check | expected | measured | verdict |
|---|---|---|---|
| findings raised on the planted case | 16, one per check | 16 | pass |
| subject of each finding | the planted object | all 16 match | pass |
| severity split | 14 error, 2 warning, 0 info | 14 / 2 / 0 | pass |
| enumeration coverage | 7/8 | 7/8 = 87.5000 % | pass |
| cited requirements resolved | 1/2 | 1/2 = 50.0000 % | pass |
| cited tests resolved | 2/3 | 2/3 = 66.6667 % | pass |
| cited tests passing | 1/3 | 1/3 = 33.3333 % | pass |
| exit code | 1 | 1 | pass |
| `complete_minimal.yaml` findings | 0 | 0 | pass |
| `complete_minimal.yaml` enumeration | 4/4 | 4/4 = 100.0000 % | pass |
| `complete_minimal.yaml` exit code | 0 | 0 | pass |

## 3. Enumeration completeness against exhaustive enumeration

`validation/validate_enumeration.py`

The grid has `4n` cells for `n` control actions, so for small `n` the whole
space of filled-cell patterns can be enumerated and compared against a
brute-force count that shares no code with the checker.

| check | reference | measured | verdict |
|---|---|---|---|
| completeness vs exhaustive enumeration, n = 1 and n = 2 | every subset of the 4n cells filled with a UCA and every subset of the remainder excluded; 6642 cases | 0 mismatches | pass |
| HT001 count equals the uncovered-cell count, same 6642 cases | brute force | 0 mismatches | pass |
| grid size is 4n, 2000 seeded random trials, n = 0 to 8 | arithmetic | 0 wrong | pass |
| covered count equals the union of filled cells | set union computed from the pattern | 0 wrong | pass |
| covered + gaps = total | partition identity | 0 broken | pass |
| `run_checks` determinism over the same trials | repeat invocation | 0 non-deterministic | pass |

Seed 20261015. Run time 1.6 s.

## 4. Malformed-document rejection

`validation/validate_parser_rejections.py`

The parser's contract is that a document it accepts has no dangling reference,
so every check downstream may assume references resolve. The failure that
matters is silent acceptance: a dropped section makes the gap report wrong in
the direction of looking **better**, which is the one direction a gap-reporting
tool must never be wrong in.

| check | expected | measured | verdict |
|---|---|---|---|
| schema rejection paths exercised | — | 49 | — |
| raised `AnalysisFormatError` | 49/49 | 49/49 | pass |
| named the expected YAML key path | 49/49 | 49/49 | pass |
| malformed documents accepted | 0 | 0 | pass |
| I/O rejection paths (missing file, directory, parent directory) | 3/3 rejected | 3/3 | pass |
| the well-formed control document parses | yes | yes | pass |

**Found during this validation, not before it.** PyYAML's default behaviour on
a duplicate mapping key is last-one-wins, silently. The first draft of this
script accidentally wrote a second `hazards:` section and watched the first one
disappear. A document with two `ucas:` sections would therefore have lost one
without a word, and the report would have looked more complete than the
document was. The parser now rejects a duplicate key in any mapping; that is
rejection path 49, and `tests/test_parse_malformed.py` carries three tests for
it.

## 5. Cross-links computed from the repository

`validation/validate_crosslinks.py`

| check | reference | measured | verdict |
|---|---|---|---|
| requirement declaration forms | hand count: heading, bullet and bold declare; a mid-sentence mention and a fenced line do not; one id declared twice | 4 declarations over 3 unique ids | pass |
| junit outcome classification | all five outcomes pytest writes | passed 1, failed 1, errored 1, skipped 1, xfailed 1 | pass |
| identifier alias forms accepted | 4 forms for one case | 4/4 resolve; a wrong-file identifier resolves to 0 | pass |
| P024 junit fixture | a real run of P024's suite | 305 cases, 305 passed | pass |
| P024 requirements fixture | hand-written for this example | 13 declared; REQ-021 and REQ-022 deliberately absent | pass |
| worked example: cited requirements resolved | — | 8/10 = 80.0000 % | pass |
| worked example: cited tests resolved | — | 7/8 = 87.5000 % | pass |
| worked example: cited tests passing | — | 7/8 = 87.5000 % | pass |
| planted case: cited tests resolved / passing | 2/3 and 1/3 | 2/3 and 1/3 | pass |
| with no requirements document and no junit report | the three ratios must be **undefined**, not 100 % | undefined | pass |
| supplying the sources can only add findings | set inclusion | holds | pass |

**Fixture provenance, stated plainly.** `fixtures/p024_bdot/junit.xml` is a real
pytest run of P024 DetumbleSim's own test suite — 305 tests, 0 failures,
0 errors, 0 skipped, 18.36 s — performed on a **copy** of P024 in a scratch
directory so that nothing in another product was written to, then normalised
(times to 0.000, a fixed timestamp, hostname `fixture`) by
`fixtures/regenerate_fixtures.py`. `fixtures/p024_bdot/docs/REQUIREMENTS.md` is
**hand-written for this worked example**: P024 ships no requirements document,
and that absence is itself one of the findings. `fixtures/planted/junit.xml` is
hand-written, because this container's own suites never produce a failing,
erroring or skipped case and the HT014 path needs all of them; its header says
so.

## 6. The Level 2 worked example — P024 DetumbleSim's B-dot loop

`validation/worked_example.py`

An STPA of the magnetorquer B-dot detumble loop implemented in P024
DetumbleSim, carried end to end. **Every statement in the analysis was read out
of P024's own source** (`control.py` for the B-dot law and the
perpendicular-torque result, `simulate.py` for the backward-difference
derivative, the zero-order hold, the first-step zero command, the seeded
magnetometer noise and the interpolated threshold crossing, `spacecraft.py` for
the per-axis dipole clipping).

**The analysis is incomplete and is published incomplete.** The eleven findings
below are its real gaps. Eight of them correspond to functions P024 does not
have.

| figure | value |
|---|---|
| control actions / feedback paths | 2 / 1 |
| UCAs / justified exclusions | 7 / 1 |
| enumeration completeness | 6/8 = 75.0000 % |
| hazards with a constraint | 5/6 = 83.3333 % |
| constraints with a requirement | 4/5 = 80.0000 % |
| UCAs with a loss scenario | 6/7 = 85.7143 % |
| scenarios with a mitigation | 5/6 = 83.3333 % |
| mitigations with evidence | 5/6 = 83.3333 % |
| cited requirements resolved | 8/10 = 80.0000 % |
| cited tests resolved / passing | 7/8 and 7/8 = 87.5000 % |
| findings | 11 error, 0 warning |
| exit code | 1 |

The eleven findings, and the fact behind each:

| code | subject | the fact in P024 |
|---|---|---|
| HT001 | `CA-DIPOLE` × stopped too soon / applied too long | Actuation ends when the simulation reaches `duration_s`. There is no onboard termination of the dipole command, no timeout, and no declared response when the rate threshold is never reached, so there is nothing to write a UCA context against and nothing that would justify excluding the class. |
| HT001 | `CA-EXIT` × wrong timing or order | There is no mode sequencing at all. Detumble entry and exit are not implemented and `detumble_time_s` is computed in post-processing from the true body rate, so there is no order of operations to analyse. |
| HT004 | hazard H-4 has no system-level constraint | H-4 is the body rate increasing while the controller is active. P024's own suite carries an explicit counterexample in which the angular-momentum magnitude rises for an asymmetric inertia under the ideal law, and there is no divergence monitor anywhere in the package. |
| HT005 | constraint SC-5 cites no requirement | SC-5 requires that the magnetic measurement not be corrupted by the dipole the control law is commanding. P024 models no coil-to-magnetometer cross-talk, so the constraint is an unbacked assertion. |
| HT006 | UCA-3 has no loss scenario | UCA-3 is the command being dominated by magnetometer noise late in the detumble. The backward difference divides per-sample noise by the control interval, so the noise term in the estimated derivative scales as 1/dt; P024 declares no noise floor, no filter and no deadband. |
| HT007 | loss scenario LS-5 has no mitigation | LS-5 is the magnetometer sampling while a coil is driven. `simulate.py` samples and commands at the same instant and there is no sample-and-settle schedule. |
| HT008 | mitigation M-4 cites no evidence | M-4 bounds the deployment tumble rate by operational agreement: a procedural control with nothing in the repository behind it. |
| HT015 | controller CTRL-MODE has no feedback path | **The sharpest one.** B-dot is a magnetometer-only law, P024 carries no rate estimator, and the completion criterion is a threshold on the true body rate. The controller that decides detumble is finished has no measurement of the quantity it decides on. |
| HT012 | REQ-021 is not declared | Cited by M-5, the proposed divergence monitor. The requirement does not exist because the function does not exist. |
| HT012 | REQ-022 is not declared | Cited by SC-4, the energy-exhaustion constraint. P024 has no power or energy model. |
| HT013 | `tests/test_divergence_monitor.py::test_rate_increase_commands_safe_mode` | Cited by EV-6; absent from a junit report of 305 real P024 tests, because the test and the code it would test are both unwritten. |

**What the worked example does not claim.** None of its UCAs or loss scenarios
was reviewed by anybody, none is traced to a hazard log, and the losses and
hazards are the author's judgement about a 3U-class mission that does not
exist. It demonstrates that the checks compute something real on a real control
loop. It is not a hazard analysis of any flight system.

## 7. Identities, measured rather than asserted

`validation/validate_properties.py` — 1500 seeded random analyses, seed
20261015, 0 to 6 control actions each. The quantities are counts over finite
sets, so these are exact integer identities; the one involving a ratio is
checked with a tolerance of 1e-12 on both sides.

| identity | holds | of applicable |
|---|---|---|
| grid size is 4n | 1500 | 1500 |
| cell states partition the grid | 1500 | 1500 |
| covered + gaps = total | 1500 | 1500 |
| HT001 count equals the gap count | 1500 | 1500 |
| HT003 count equals the conflict count | 1500 | 1500 |
| every cell appears exactly once | 1500 | 1500 |
| `run_checks` is deterministic | 1500 | 1500 |
| declaration order does not change the findings | 1500 | 1500 |
| enumeration ratio agrees with the grid to 1e-12 | 1276 | 1276 |
| enumeration ratio lies in [0, 1] | 1276 | 1276 |
| excluding a gap cell removes exactly that gap | 1110 | 1110 |
| adding a UCA to a filled cell leaves the gap set alone | 1041 | 1041 |
| adding an orphan component adds exactly one HT016 | 1500 | 1500 |

**The claim that was FALSIFIED, reported as a result.** The claim "filling an
enumeration gap never increases the total finding count" is **false**, and it
failed in **1110 of 1110 applicable trials**. Filling a cell removes one HT001
and adds an HT006 and an HT010 whenever the new UCA has no loss scenario and no
hazard, which is the state a freshly written UCA is in. The finding **count**
is therefore not a progress measure; the enumeration ratio is. Hypothesis found
this independently; the smallest counterexample — one control action, no UCAs,
no exclusions, 5 findings before and 6 after — is pinned with literal inputs in
`tests/test_regression.py::test_filling_an_enumeration_gap_can_increase_the_finding_count`,
because `.hypothesis/` is gitignored and a counterexample that lives only there
is lost on a fresh clone.

`tests/test_properties.py` runs the same identities under Hypothesis (100 to 150
examples each). No other falsifying example was found.

## 8. CLI exit-code contract

`validation/validate_cli.py` — 28 real `python -m hazardtrace` subprocesses.

| quantity | value |
|---|---|
| invocations run | 28 |
| exit-code mismatches against the documented contract | 0 |
| every check code present in the planted report | 16/16 |
| planted report severity line | `14 error, 2 warning` |
| worked-example report severity line | `11 error, 0 warning` |
| a cannot-run failure writes to stderr, not stdout | yes |

The contract: 0 when the command succeeded and no blocking finding was raised,
1 when at least one blocking finding was raised, 2 when the command could not
run. A warning-only analysis exits 0 and exits 1 under `--strict`; both are
measured.

## 9. Dependency ranges, measured at both ends

`validation/validate_environment.py`

Every range in `pyproject.toml` is bounded on both sides and **both ends were
installed and the whole suite plus every validation script run at them**. No
bound was derived by reading a changelog or reasoning about an API.

| package | declared range | floor measured | ceiling measured |
|---|---|---|---|
| `pyyaml` | `>=6.0.1,<6.1` | 6.0.1 | 6.0.3 |
| `matplotlib` (optional) | `>=3.11.0,<3.12` | 3.11.0 | 3.11.2 |
| `pytest` (test extra) | `>=8.3.5,<9.2` | 8.3.5 | 9.1.1 |
| `hypothesis` (test extra) | `>=6.100.0,<6.170` | 6.100.0 | 6.169.3 |
| `ruff` (test extra) | `>=0.16.8,<0.17` | 0.16.8 | 0.16.8 |

Python 3.13.16, CPython, Linux x86_64, 2 cores.

| measurement | floor | ceiling |
|---|---|---|
| `python -m pytest tests/ -q` | 215 passed | 215 passed |
| all eight validation scripts | all exit 0 | all exit 0 |
| diff of the committed validation output between the two | **only `validate_environment_output.txt`, which records the versions themselves** | — |

That last line is the point of the exercise. A suite that passes at both ends
while its validation numbers move is the failure mode that no test catches; the
`diff -r` between the two runs was taken and every number is identical.

**Two floors are set by measurement, not by preference, and both cost
something:**

- **PyYAML 6.0** has no cp313 wheel and fails to build against Cython 3 on
  Python 3.13 (`AttributeError: 'build_ext' object has no attribute
  'cython_sources'`). 6.0.1 is the lowest version this package could be
  measured at here, so 6.0 is excluded even though the code would almost
  certainly work with it.
- **matplotlib 3.9.2 and 3.10.0 both fail this suite at collection time.** Both
  call pyparsing's deprecated `oneOf`; pyparsing 3.3.3 raises
  `PyparsingDeprecationWarning`, and `[tool.pytest.ini_options] filterwarnings
  = ["error::DeprecationWarning"]` turns it into a collection error in
  `tests/test_diagram.py` and `tests/test_integration.py`. matplotlib 3.11.0
  does not. The declared floor is therefore 3.11.0, which is narrow, and the
  narrowness is a measured fact rather than a choice.

The ceilings are a real cost, not free safety: this package cannot take a
future major of PyYAML, or matplotlib 3.12, without a port and a
re-measurement. That is recorded in README.md under Limitations rather than
resolved by deleting the ceiling.

## 10. Shipped-file ignore check

`git check-ignore -q` was run against every file this product ships, from the
portfolio root, on 2026-10-15.

| path | status | required |
|---|---|---|
| `products/P065/junit.xml` | ignored | ignored, it is a per-run artifact |
| `products/P065/fixtures/p024_bdot/junit.xml` | tracked | tracked, it is a committed fixture |
| `products/P065/fixtures/planted/junit.xml` | tracked | tracked, it is a committed fixture |
| the three `examples/cases/*.yaml` | tracked | tracked |
| the two `fixtures/**/docs/REQUIREMENTS.md` | tracked | tracked |
| `screenshots/*.png`, `screenshots/*.mmd` | tracked | tracked |
| `validation/*_output.txt` | tracked | tracked |

The product `.gitignore` anchors every scratch pattern with a leading slash, so
`/junit.xml` matches only at the product root. An unanchored `junit.xml` would
match at any depth and would have excluded both committed fixtures; that exact
defect shipped in P060 of this portfolio and the anchoring here is why it does
not repeat.

## 11. Checks that are not here, and why

| not checked | why |
|---|---|
| whether a declared UCA is a real UCA | It reads structure, never content. A UCA whose description is the word "bad" is indistinguishable from a considered one. |
| whether the four classes were applied **thoughtfully** | Cell coverage is a count. A cell occupied by a vacuous UCA is covered. |
| context tables in the sense of Thomas (2013) | Not implemented in this release. A context table enumerates process-model variable combinations per control action and would make the coverage denominator far larger and far more honest. |
| whether a hazard is a hazard, or a loss a loss | No taxonomy, no ontology, no judgement. |
| whether cited evidence supports the mitigation | The tool checks that a cited test exists and passed. Whether a passing test is evidence of anything is the question it refuses. |
| whether the control structure matches the system | Nothing is read from source code or a model. The structure is a declaration. |
