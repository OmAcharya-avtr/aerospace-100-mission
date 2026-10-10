# Validation evidence — traceaudit 0.1.0

**Validation level 1.** Every check in this document is against text, XML and
test sources constructed by hand for the purpose, with the expected answer
worked out by reading the fixture before the code was run. **Research-grade.
Not flight-qualified, not certified, not approved for operational aerospace
use. Not a DO-178C or ARP4754A compliance tool.** No learned component, no AI,
and therefore no MODEL_CARD.md and no DATASET_CARD.md: there is no model and
no training set. Section 11 says what the Level 1 label excludes.

Every number in this file and in README.md was produced by a script in this
directory, executed in this container on **2026-10-10**, with its raw stdout
committed beside it as `<script>_output.txt`. Nothing here was copied from a
paper, estimated, or rounded by hand. The checks that undercut this package
are in section 9 on purpose.

**The claim this package must not make.** Traceability is a **necessary
bookkeeping condition and not evidence of adequacy**. A requirement traced to
a test that asserts nothing is still traced, and this tool reports it as
covered. Section 5 measures how badly the one check that gestures at adequacy
performs, and the answer is: badly enough that it is labelled a heuristic
everywhere it appears.

**Environment.** Python 3.13.16 (CPython) on Linux 6.18.44-fc-v114 x86_64,
pytest 9.1.1, Hypothesis 6.168.5, Matplotlib 3.11.2, NumPy 2.5.3.
`os.cpu_count()` = 2 and `len(os.sched_getaffinity(0))` = 2: **two cores,
shared and contended with sibling build agents.** The package declares **no
runtime dependencies**; it uses `re`, `ast`, `xml.etree.ElementTree`, `json`,
`argparse`, `pathlib`, `dataclasses` and `enum` from the standard library.
Matplotlib is an optional extra used only by `plotting.py`, and NumPy arrives
with it. Raw output: `validate_environment_output.txt`.

**Timings move, counts do not.** The suite runs in 7 to 12 s on two contended
cores and the figure moves by 10 to 40 % between runs. Every primary number
in this document is a **count**: declarations, test cases, findings,
numerators, denominators, confusion-matrix cells. Those are deterministic and
reproduce exactly.

**Compute budget.** The slowest thing in the repository is
`validate_self_trace.py`, which runs the whole suite in a subprocess: about
12 s. `validate_cli.py` runs 20 subprocesses including that suite and the four
examples: about 40 s. Nothing approaches the mission's 3-minute budget.

**Test suite, counted from junit XML.** `python -m pytest tests/ -q
--junitxml=...` writes `<testsuite tests="236" failures="0" errors="0"
skipped="0">`. **236 passed, 0 failed, 0 errored, 0 skipped, 0 xfailed.** The
count is read from that element, not from pytest's stdout summary, for the
reason given in section 7. Raw output: `validate_self_trace_output.txt`,
`pytest_output.txt`.

---

## 1. Requirement parsing — known answers

`validate_parsers.py` → `validate_parsers_output.txt`. **FAILED CHECKS: 0.**

The fixture `fixtures/sample_project/docs/REQUIREMENTS.md` was written to
contain exactly eight requirements and one duplicate declaration, and the
expected counts below were obtained by reading it.

| Check | Reference / method | Result | Tolerance |
|---|---|---|---|
| Declarations in the sample document | hand count: REQ-001..REQ-008 as headings plus one repeat of REQ-007 in the appendix | **9** | exact |
| Unique ids | same | **8** | exact |
| Duplicated ids | same | **`['REQ-007']`**, 2 declarations | exact |
| First declaration's title | hand read | **"Wheel speed input validation"** | exact |
| A mid-sentence mention of REQ-002 does not declare it | the document says "unlike REQ-002, ..." on a prose line | **1 declaration of REQ-002, not 2** | exact |
| Declarations in this package's own document | hand count: REQ-001..REQ-026, all level-3 headings | **26 declarations, 26 unique, 0 duplicates, contiguous** | exact |
| Heading, bullet, bold-bullet and bare declaration forms | document written for this check | **all four parsed**, in line order | exact |
| Mid-line mention excluded | same | **excluded** | exact |
| Fenced declaration excluded by default | same | **excluded** | exact |
| Fenced declaration included at `skip_code_fences: false` | same | **included** | exact |

**Why the model is line-oriented.** A markdown parser would let an id in a
table cell or an inline span declare a requirement, and a document that
discusses its own ids — as both documents in this repository do — would then
declare phantom requirements. The restriction is deliberate and is listed as
limitation 5 in README.md.

## 2. Test-report parsing — known answers

Same script. The junit fixture is a **real pytest 9.1.1 run** over
`fixtures/sample_project/tests/test_monitor.py`, normalised for
reproducibility (section 6), not a hand-written XML file.

| Check | Reference / method | Result | Tolerance |
|---|---|---|---|
| Cases parsed | hand count of the fixture suite | **8** | exact |
| Outcome classification | pytest's junit writer: `<skipped type="pytest.skip">`, `<skipped type="pytest.xfail">`, `<failure>`, `<error>`, bare | **passed 5, failed 1, errored 0, skipped 1, xfailed 1, unknown 0** | exact |
| `<testsuite tests>` header | the fixture | **8** | exact |
| `<testsuite skipped>` header counts xfails too | the fixture | **2**, against 1 skipped + 1 xfailed counted from the cases | exact |
| Claimed ids, in first-seen order | hand read of the markers | **REQ-001, REQ-002, REQ-003, REQ-004, REQ-006, REQ-007, REQ-008, REQ-042** | exact |
| Skip reason preserved | the fixture | "simulator fixture not available in this environment" | exact |
| xfail distinguished from skip | the `type` attribute | **xfailed**, not skipped | exact |
| Source file recovered from the `file` attribute | `junit_family = xunit1` | **tests/test_monitor.py** | exact |
| Collection report: cases | `pytest --collect-only -q` on the same suite | **8** | exact |
| Collection report: outcomes | the format carries none | **all `unknown`** | exact |
| Collection report: claims | the format carries no properties | **none at all** | exact |
| Collection report: node ids | compared element by element against the junit ones | **8 of 8 identical** | exact |

**The header is not trusted.** `skipped="2"` in the header is one skip and one
xfail; a reader who reports "2 skipped" from the header is wrong about one of
them. The audit cross-checks the header against the counted cases and raises
a note when they disagree (`findings.py`), which is REQ-009 of this package.

## 3. Findings — one input per code

`validate_findings.py` → `validate_findings_output.txt`. **FAILED CHECKS: 0.**

The sample project was built so that every code has exactly one input that
produces it. The enumeration below was written before the code was run and is
reproduced in the module docstring of `tests/test_known_answers.py`.

| Code | Input that produces it | Expected | Measured |
|---|---|---|---|
| TA001 | REQ-005 is declared and no test claims it | 1 | **1** |
| TA002 | `test_claims_an_undeclared_requirement` claims REQ-042 | 1 | **1** |
| TA003 | REQ-007 is declared at lines 40 and 54 | 1 | **1** |
| TA004 | REQ-003 traced only by a skipped test; REQ-004 only by an xfailed one | 2 | **2** |
| TA005 | REQ-008 traced only by a failing test | 1 | **1** |
| TA006 | `test_request_counter` delegates its assertion to `_check_nonneg` | 1 | **1** |
| total | | 7 | **7**, exit status **1** |

| Check | Result |
|---|---|
| Only TA006 is marked heuristic | **yes** |
| TA003's message names both declaration lines | `docs/REQUIREMENTS.md:40`, `docs/REQUIREMENTS.md:54` |
| TA004's message names the outcome per requirement | `[skipped]` for REQ-003, `[xfailed]` for REQ-004 |
| Withholding `--test-root` removes TA006 and nothing else | 7 → **6** findings, with a note saying why |
| `--ignore TA006` keeps the finding and drops it from the exit status | **7 findings, 6 blocking, exit 1** |
| Ignoring every code | **7 findings, 0 blocking, exit 0** |
| The same project from a collection report | TA001 **1 → 8**, TA004 and TA005 **not computed**, with a note |

**The one that matters.** REQ-003 and REQ-004 are counted by nominal coverage
and not by executed coverage: `len(traced) - len(executed_traced) = 2`. Those
are the requirements that look covered on a spreadsheet and are not.

## 4. Coverage arithmetic

`validate_coverage.py` → `validate_coverage_output.txt`. **FAILED CHECKS: 0.**

Hand arithmetic for the sample project, denominator **8 unique declared ids**
(REQ-007 is declared twice and counts once):

```
nominal  : 001 002 003 004 006 007 008   -> 7/8
executed : 001 002 006 007 008           -> 5/8   (003 skipped, 004 xfailed)
passing  : 001 002 006 007               -> 4/8   (008 failed)
```

| Figure | Definition | Measured | Percent |
|---|---|---|---|
| nominal coverage | requirements with ≥1 claiming test of any outcome | **7/8** | **87.5000 %** |
| executed coverage | requirements with ≥1 claiming test that actually ran | **5/8** | **62.5000 %** |
| passing coverage | requirements with ≥1 claiming test that passed | **4/8** | **50.0000 %** |

**Quoting the first alone overstates the third by 37.5000 percentage points on
this fixture.** That spread is the reason all three are printed with their
numerator and denominator rather than one being chosen.

| Check | Reference / method | Result | Tolerance |
|---|---|---|---|
| Coverage chain `nominal ≥ executed ≥ passing` | 2000 pseudo-random requirement/test sets, seed 60011 | **0 violations**; **822 of 2000** inputs had a strict gap | exact |
| Zero denominator | empty requirements document | percent is **`None`**, rendered as **"undefined"**, not 0 % and not 100 % | exact |
| A duplicated declaration counts once in the denominator | two declarations of one id | denominator **1** | exact |

The same chain is property-tested with Hypothesis over 200 generated examples
in `tests/test_properties.py`; this section is the deterministic, seeded
restatement of it with the seed written down.

## 5. The assertion heuristic — measured, and it is not good

`validate_heuristic.py` → `validate_heuristic_output.txt`. **FAILED CHECKS: 0**
(the checks assert the confusion matrix, which is stable; the matrix itself is
the finding).

`fixtures/heuristic/` holds **23 test functions** in three files, each
hand-labelled in `labels.json` with `asserts_something` and a one-line reason.
The label is a human judgement made by reading the function before the
heuristic was run against it: true when the function would fail if the
behaviour under test were wrong, false when it would pass regardless. Eight
were labelled as verifying nothing, fifteen as verifying something.

The positive class is **"the heuristic flags this test as having no
assertion"**.

| | truly empty | truly asserts |
|---|---|---|
| **flagged empty** | 5 | **3** |
| **not flagged** | **3** | 12 |

| Metric | Value | In words |
|---|---|---|
| precision | **0.625000** | 3 of 8 flags are wrong |
| recall | **0.625000** | 3 of 8 empty tests are caught; the rest are missed |
| false-positive rate | **0.200000** | 3 of 15 asserting tests are wrongly flagged |
| false-negative rate | **0.375000** | 3 of 8 empty tests are missed |
| accuracy | **0.739130** | 17 of 23 |

**Both error modes have exactly one cause, and both are structural.**

*False positives — the assertion is one call away and calls are not followed:*

| Function | Why the label says it asserts |
|---|---|
| `test_delegates_to_module_helper` | the assertion is in `_check_nonneg`, one call away |
| `test_delegates_to_fixture` | the assertion is in the `checker` fixture |
| `test_delegates_to_verifier_object` | `Verifier.must_equal` raises `AssertionError` |

*False negatives — the `assert` exists and is syntactically non-constant, but
is vacuous:*

| Function | Why the label says it verifies nothing |
|---|---|
| `test_vacuous_self_comparison` | `value == value` is true for every non-NaN value |
| `test_always_true_predicate` | `len(result) >= 0` is true for every list |
| `test_assert_short_circuited_by_true` | the `or True` arm makes the assert unconditional |

**What would move these numbers, and why it was not done.** Following one
level of call into a module-level helper would remove two of the three false
positives, and would introduce a new error mode: a helper that only logs would
then read as an assertion. Catching the three false negatives needs data flow,
not syntax, and no syntactic rule can do it. Neither is implemented. Instead,
TA006 is marked `heuristic` in every rendering, is the only code the README
suggests ignoring, and is omitted entirely when no `--test-root` is given.

**The corpus is small.** Twenty-three functions is a small sample for an error
rate near 0.2, and the three false positives are three instances of one
pattern rather than three independent observations. These numbers describe
this corpus. They are not an estimate of the rate on an arbitrary codebase,
and nothing in this repository claims they are.

## 6. The committed fixtures are regenerable

`validate_fixtures.py` → `validate_fixtures_output.txt`. **FAILED CHECKS: 0.**

`fixtures/sample_project/junit.xml` and `collect_only.txt` are produced by
`fixtures/regenerate_fixtures.py`, which runs pytest over the fixture suite
and normalises the parts of a pytest report that are not reproducible:
`time`, `timestamp` and `hostname` attributes, and the absolute path pytest
writes into the text of a `<skipped>` element. Element structure, attributes,
outcomes and properties are exactly as pytest wrote them.

| Check | Result |
|---|---|
| `junit.xml` regenerates byte-identically | **yes**, 2513 bytes |
| `collect_only.txt` regenerates byte-identically | **yes**, 401 bytes |
| Neither contains an absolute home or root path | **confirmed** |

`fixtures/sample_project/pytest.ini` pins the fixture suite's rootdir to that
directory. Without it, pytest finds this package's own `pyproject.toml`,
rootdir becomes the repository root, and the paths written into the fixture
change — which is how this check earned its place: the first generated
fixture was 2897 bytes for exactly that reason.

## 7. The self-audit, and why the test count comes from junit XML

`validate_self_trace.py` → `validate_self_trace_output.txt`.
**FAILED CHECKS: 0.**

A suite that collects nothing prints a success line and exits 0. A sibling
product in this portfolio shipped publicly with a pytest configuration that
collected zero tests and a README claiming 251 passing for three weeks. The
count below is therefore read from the junit XML `<testsuite>` element and
cross-checked against the outcomes counted from the individual `<testcase>`
elements by this package's own parser.

| Source | tests | failures | errors | skipped | xfailed | passed |
|---|---|---|---|---|---|---|
| `<testsuite>` attributes | **236** | **0** | **0** | **0** | — | 236 by arithmetic |
| counted from `<testcase>` elements by `parse_junit_xml` | **236** | **0** | **0** | **0** | **0** | **236** |
| pytest's stdout summary line | `236 passed in 6.73s` (from `pytest_output.txt`) — **quoted for comparison only, not used** | | | | | |

| Check | Result |
|---|---|
| Suite collected more than zero tests | **yes** |
| Case count matches the header | **236 = 236** |
| skipped + xfailed matches the header's `skipped` | **0 = 0** |
| Declared requirements in `docs/REQUIREMENTS.md` | **26** |
| Findings auditing this repository against itself | **0** |
| Audit exit status | **0** |
| nominal / executed / passing coverage | **26/26 on all three = 100.0000 %** |
| Claiming test cases | **236 of 236** |

**There is a pleasing recursion here and it should not be mistaken for a
proof.** This package parses junit XML, and its own test count is taken from
junit XML by the same parser. The cross-check against the raw
`<testsuite>` attributes, read with `xml.etree` directly rather than through
the package, exists so that the recursion cannot hide a parser bug.

**And the clean run means bookkeeping.** 100.0000 % passing coverage over 26
requirements says that each of the 26 statements in `docs/REQUIREMENTS.md` is
claimed by at least one test that passed. It does not say the statements are
true, that the tests are adequate, or that anything is verified.

## 8. Alternatives — how each was checked

Checked **2026-10-10** in this container. Method: `curl` against
`https://pypi.org/pypi/<name>/json` for existence and version, then
`pip download --no-deps` and `unzip` of the wheel for the first three, and
reading of the unpacked source before describing it.

| Package | HTTP | Version seen | What was read | What was concluded |
|---|---|---|---|---|
| `traceaudit` | **404** | — | — | **name free**, which is the condition for using it |
| `doorstop` | 200 | **3.2** (110 releases, LGPLv3) | wheel unpacked: `doorstop/{cli,core,gui,server}`, `core/validators/item_validator.py`, `core/reference_finder.py`, `core/publishers/{html,latex,markdown,text}.py`, `core/vcs/{git,mercurial,subversion}.py`, entry points `doorstop`, `doorstop-gui`, `doorstop-server` | a requirements **management** system with link validation, suspect links, VCS reference resolution and publishers. A recursive grep of the wheel for `junit` returns **nothing**: it does not read test reports |
| `sphinx-needs` | 200 | **8.5.0** (32 releases) | wheel unpacked: `directives/need*.py` (needflow, needtable, needpie, needgantt, needuml, needextend, needimport, needservice), `warnings.py`, `need_constraints.py`, `needs_schema.py`, `services/` | typed need objects, links, filters, diagrams and a configurable warnings mechanism inside a Sphinx build. Grep for `junit`: **nothing** — the junit half is a separate package |
| `sphinx-test-reports` | 200 | **2.0.0** | wheel unpacked: `sphinxcontrib/test_reports/` with `schemas/JUnit.xsd`, `directives/test_{suite,case,results,env,file,report}.py`, `pytest_plugin.py`; depends on `sphinx-needs>=6.0.1` under the `sphinx` extra | **the mature answer to this problem.** It ingests junit XML and links results to needs. Conceded first in the README |
| `pytest-requirements` | 200 | **0.3.1** | wheel unpacked and `__init__.py` read in full: registers `verifies_requirement` and `verifies_usecase` markers, and `pytest_runtest_call` appends `("requirement_id", value)` to `item.user_properties`, which pytest writes into junit XML as a `<property>` | **the exact mechanism this package reads by default.** It records claims; it does not analyse them. The two compose with no change to either |
| `strictdoc` | 200 | **0.30.2** | metadata only (summary, version) | a full technical-documentation and requirements-management system. **Described from its PyPI summary only**; the wheel was not unpacked, and the README says no more about it than that |
| `pytest` | 200 | **9.1.1** | installed and used throughout | markers select tests; they cannot report what no test claims |
| `reqif` | 200 | 0.1.0 | metadata only | ReqIF parsing, noted here only to record that it was checked and is not described in the README |

**Two traps recorded by earlier batches, re-confirmed here so that nobody
reaches for them:** PyPI `gsn` 0.2.0 is "a python wrapper for Global Sensor
Networks API" (EPFL), **not** Goal Structuring Notation; PyPI `advocate` 1.0.0
is an SSRF-protection wrapper around `requests`, **not** NASA AdvoCATE.
Neither is mentioned in README.md and neither is relevant to this product.

**Not verified:** IBM DOORS Next, Polarion, Jama Connect and codeBeamer are
named in the alternatives table as a category. They are commercial products
with no PyPI presence, nothing about them was measured, and the README says
plainly that the comparison is not a comparison.

## 9. Honest negatives, collected

The mission values these above anything else in the document, so they are
gathered here rather than left scattered.

1. **The assertion heuristic is wrong about one asserting test in five.**
   False-positive rate **0.200000**, false-negative rate **0.375000**,
   precision **0.625000** on 23 hand-labelled functions (section 5). Every
   false positive is a test whose assertion lives one call away; every false
   negative is a vacuous `assert`. This is the only check in the package that
   can be wrong about a correct input, and it is the one a user is most likely
   to want, which is the worst possible combination.
2. **The corpus behind those rates is 23 functions**, and the three false
   positives are three instances of one pattern. The numbers describe the
   corpus, not an arbitrary codebase.
3. **An xpassed test is indistinguishable from a pass.** pytest 9.1.1 writes a
   test marked `xfail` that unexpectedly passed as a bare `<testcase>` with no
   child element. TA004 therefore cannot see it, and a requirement traced only
   to an xpassed test is reported as **fully covered on all three figures**.
   Verified directly against pytest output in this container and pinned by
   `tests/test_edge_cases.py::test_an_xpassed_test_is_indistinguishable_from_a_pass`.
   There is no fix available from junit XML alone.
4. **A pytest collection report is much weaker input than the README's
   headline suggests a "report" would be.** It carries no outcomes and no
   properties. Measured on the sample project: TA001 rises from **1 to 8**,
   TA004 and TA005 are **not computed**, executed and passing coverage are
   **undefined**, and nominal coverage falls from 87.5000 % to **0.0000 %**.
   Supporting the format at all is arguably generous; it is supported, and
   loudly caveated.
5. **An unclosed code fence hides the rest of the document.** The fence state
   is a toggle, so a stray triple backtick silently removes every later
   requirement from the denominator — which makes coverage look *better*.
   Tested and documented, not fixed, because parsing inside fences would stop
   a requirements document from showing example requirements.
6. **The id pattern is case-sensitive everywhere**, so `test_req_014` does not
   claim REQ-014 while `test_REQ_014` does. This surprises people and is
   pinned by a test rather than smoothed over.
7. **The first generated fixture was wrong and the check caught it.** Without
   `fixtures/sample_project/pytest.ini`, pytest resolved rootdir to the
   repository root and wrote different paths into the fixture (2897 bytes
   instead of 2513). The fixture-reproducibility check in section 6 exists
   because of that, not in anticipation of it.
8. **Three of this package's own checks were written wrong first.** During the
   build, `validate_findings.py` keyed its message assertions by code alone
   and so tested only the last TA004 finding; two tests asserted `assert 1`,
   which the heuristic correctly treats as no assertion, and a third asserted
   a column width that was off by one space. All three were script defects,
   not product defects, and all three are recorded here rather than quietly
   corrected.
9. **100 % coverage of this package's own requirements is a bookkeeping
   result.** It is stated in README.md next to the number, and it is the
   clearest possible demonstration of the product's own central caveat.

## 10. Reproducing every number

From the repository root. Each command's raw output is committed in this
directory under the name given.

| Command | Produces | Output file |
|---|---|---|
| `python -m pytest tests/ -q --junitxml=junit.xml` | the 236-test count | `pytest_output.txt` |
| `ruff check src/ tests/ examples/ validation/` | lint clean | — |
| `python validation/validate_environment.py` | versions, core count, fixture sizes | `validate_environment_output.txt` |
| `python validation/validate_parsers.py` | sections 1 and 2 | `validate_parsers_output.txt` |
| `python validation/validate_findings.py` | section 3 | `validate_findings_output.txt` |
| `python validation/validate_coverage.py` | section 4 | `validate_coverage_output.txt` |
| `python validation/validate_heuristic.py` | section 5 | `validate_heuristic_output.txt` |
| `python validation/validate_fixtures.py` | section 6 | `validate_fixtures_output.txt` |
| `python validation/validate_self_trace.py` | section 7 | `validate_self_trace_output.txt` |
| `python validation/validate_cli.py` | every README command, 20 of them | `validate_cli_output.txt` |
| `python validation/worked_example.py` | the README worked example | `worked_example_output.txt` |
| `MPLBACKEND=Agg python examples/audit_sample_project.py` | `screenshots/trace_matrix.png` | — |
| `MPLBACKEND=Agg python examples/coverage_definitions.py` | `screenshots/coverage_definitions.png` | — |
| `MPLBACKEND=Agg python examples/finding_breakdown.py` | `screenshots/finding_breakdown.png` | — |
| `MPLBACKEND=Agg python examples/heuristic_error_rates.py` | `screenshots/heuristic_confusion.png` | — |
| `python fixtures/regenerate_fixtures.py` | the committed junit and collection fixtures | — |

**Every script in this directory exits 0.** `validate_cli.py` deliberately
runs commands that exit 1 — that is the product reporting findings, which is
correct behaviour — and it does so through `subprocess.run`, comparing each
`returncode` with the status the README documents. A non-zero exit from a
script in this directory would be a defect in the script.

## 11. What Level 1 excludes

Stated explicitly, because a validation level that is not defined is
decoration.

**What was validated.** That the parsers recover exactly the declarations and
test cases that a human reading the fixture counted; that each finding code
fires on an input built to produce it and on nothing else in that input; that
the coverage arithmetic matches hand arithmetic and that its ordering property
holds over 2000 seeded random inputs; that the heuristic's error rates against
23 hand labels are what section 5 says; that the committed fixtures regenerate
byte-identically; that the package audits itself cleanly; and that every
command quoted in the README produces the output and exit status the README
shows, re-executed in this session.

**What was not, and cannot be.**

- **No physical reference.** There is no equation, no unit and no physical
  quantity anywhere in this product. There is nothing to compare against a
  measurement, a published dataset or an independent reference implementation.
  This is the whole reason the level is 1 rather than 2, and labelling it 2
  would be dishonest.
- **No cross-tool agreement.** No output of this package has been compared
  against `doorstop`, `sphinx-needs` with `sphinx-test-reports`, or any
  commercial tool. None of them computes quite this set of findings and none
  is installed here.
- **No large-scale corpus.** The heuristic has been measured on 23 functions
  and the parsers on two documents and two reports. Nothing here has been run
  over a large body of real repositories.
- **No qualification of any kind.** No tool qualification, no DO-330
  consideration, no DO-178C objective mapping, no ARP4754A process claim. The
  word "compliance" does not describe anything in this repository.
- **No claim about adequacy.** The product reports that a requirement is
  traced. Whether the test that traces it verifies anything useful is outside
  the model, and the one check that reaches toward it is the least reliable
  thing here.

## 12. References

The conventions this package reads, rather than theory it implements:

- pytest's junit XML writer — `<skipped type="pytest.skip">` versus
  `<skipped type="pytest.xfail">`, the `file` and `line` attributes under
  `junit_family = xunit1`, and `<property>` elements from
  `record_property` and `item.user_properties`. Verified empirically against
  **pytest 9.1.1** in this container; the probe output is reproduced in
  section 2.
- `pytest-requirements` **0.3.1**, whose `requirement_id` property name is the
  default this package reads. Read from the unpacked wheel, section 8.
- The JUnit XML schema as shipped in `sphinx-test-reports` **2.0.0**
  (`schemas/JUnit.xsd`), consulted for the element names only; this package
  does not validate against it.

No aerospace standard is implemented, cited as implemented, or claimed to be
supported by anything in this repository.
