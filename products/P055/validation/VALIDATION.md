# Validation evidence — assuregraph 0.1.0

**Validation level 2** — the package is checked against hand-worked known
answers, exact graph identities, and its own documented behaviour. There is no
physical reference to validate against, because this product computes nothing
physical: every number in it is a count of nodes, a count of findings, a byte
length or a wall-clock second. Level 2 is the honest label; calling it level 3
would imply a reference model that does not exist.

**Scope statement, which bounds everything below.** A well-formed assurance case
is not a safe system. assuregraph checks the structure of a case and the
freshness of the artifacts it cites. It evaluates no argument's soundness. It is
**not a DO-178C or ARP4754A compliance tool**, and nothing in this document is
evidence about any system. Research-grade software: **not flight-qualified, not
certified, not approved for operational aerospace use.**

Every number below was produced by a script in this directory, executed in this
container on 2026-10-08, with its raw stdout committed beside it as
`<script>_output.txt`. Nothing here was copied from a paper, estimated or
rounded by hand.

**Environment.** Python 3.13.16, PyYAML 6.0.3 (built against libyaml, so the
`CSafeLoader` path is the one measured), pytest 9.1.1, Hypothesis 6.168.5,
matplotlib 3.11.2, Ruff 0.16.8. Two CPU cores, 7.8 GiB RAM, shared with sibling
build agents. Every wall-clock figure is measured under that contention and
moves by 10–20 % run to run; the accuracy figures are exact counts and do not
vary.

**Re-running this evidence.** Ten of the twelve committed raw-output files are
byte-identical across re-runs; the two that are not are
`validate_scale_output.txt` and `validate_staleness_output.txt`, which report
wall-clock timings on purpose. Filesystem paths in the committed outputs are
redacted to `<repo>` or `<scratch>` so a checkout in a different directory does
not change them.

**Test suite.** `python -m pytest tests/ -q` from the repository root:
**248 passed, 0 failed, 0 skipped, 0 xfail**, in 1.93 s (junit XML:
`tests=248 failures=0 errors=0 skipped=0`). No test is marked xfail or skipped;
there is nothing hidden behind either.

**Reference for the notation.** Goal Structuring Notation Community Standard,
**Version 3**, document **SCSC-141C**, published by the **SCSC Assurance Case
Working Group**, **May 2021** (the standard's own document history records
Version 1 issued 16 November 2011 and Version 2 in January 2018). The version,
number, publisher and date were read from `scsc.uk/scsc-141c` and
`scsc.uk/resources/citation_r1386.html` on 2026-10-08 before being written here.
**No clause or page number is quoted**, because none was verified in this
environment. What is taken from the standard is: the six Core GSN element types,
the two relationship types, the identifier prefixes used in its examples, the
*Undeveloped* decorator's meaning as an author declaration, and the shapes the
notation draws.

**No DO-178C or ARP4754A clause is cited anywhere in this repository, and none
should be.** Those documents were not consulted and this tool has no
relationship to them.

---

## 1. Parser: every rejection path

`validate_parser_rejections.py` → `validate_parser_rejections_output.txt`.

A malformed case must be rejected with a message that says where the problem is
and what to do about it. The script builds one deliberately malformed document
per rejection path and records the message verbatim.

| Check | Method | Result | Tolerance |
|---|---|---|---|
| Schema rejection paths exercised | one malformed document each | **51** | — |
| Rejected with `CaseFormatError` | expected type | **51 / 51** | exact |
| Message carried a YAML key path | `CaseFormatError.location` not `None` | **51 / 51** | exact |
| I/O rejection paths exercised | missing file, a directory, unparsable YAML | **3** | — |
| Rejected with `CaseIOError` | expected type | **3 / 3** | exact |
| Documents wrongly accepted | — | **0** | exact |

The paths include every one of: a non-mapping document, unknown keys at every
level, a missing or blank `name`, empty `nodes`, a duplicate node id, an unknown
element type, an unknown relationship type, the *Undeveloped* decorator on a
non-claim, `discharged` on a non-Assumption, `discharged_by` without
`discharged`, a dangling `discharged_by`, a Solution with no evidence, evidence
on a non-Solution, a digest that is not 64 hexadecimal characters, a self-loop, a
duplicate edge, a dangling edge endpoint, a top goal that is not a Goal, and the
two Core GSN relationship-table violations below.

**Two rules that come from the notation rather than from software hygiene** are
enforced at parse time, so a document that breaks them never becomes an
`AssuranceCase`:

- only a Goal or a Strategy may be the source of a relationship;
- `SupportedBy` may terminate only on a Goal, Strategy or Solution, and
  `InContextOf` only on a Context, Assumption or Justification.

Both are rejections, not findings, because a document that breaks them is not a
GSN case at all.

---

## 2. The six checks against hand-worked answers

`validate_known_answers.py` → `validate_known_answers_output.txt`.

The expected finding set for each shipped example was written by hand from the
YAML **before** the script was run, and is committed inside the script as
`EXPECTED`. The same expectations are asserted by `tests/test_integration.py`.

| Case | Nodes / edges | Expected findings | Computed findings | Sets match | Expected exit | Computed exit |
|---|---|---|---|---|---|---|
| `complete_case.yaml` | 18 / 17 | 0 | **0** | yes | 0 | **0** |
| `incomplete_case.yaml` | 14 / 12 | 9 | **9** | yes | 1 | **1** |
| `cyclic_case.yaml` | 5 / 5 | 1 | **1** | yes | 1 | **1** |

The incomplete case's nine findings, with the node ids each check reports:

| Check | Severity | Node ids | Why |
|---|---|---|---|
| `unsupported_claims` | error | `G5` | no `SupportedBy` child and not declared undeveloped |
| `unsupported_claims` | info | `G7` | declared `undeveloped: true` — a declaration, not a defect |
| `unsupported_claims` | info | `G8` | declared `undeveloped: true` |
| `missing_evidence` | error | `Sn3` | cites `evidence/detection_latency_study_v2.txt`, not on disk |
| `stale_evidence` | warning | `Sn1` | no recorded digest, so freshness is not computable |
| `stale_evidence` | error | `Sn2` | recorded digest differs from the file's current digest |
| `undischarged_assumptions` | error | `A1` | `discharged` absent |
| `orphan_nodes` | error | `G8` | unreachable from the declared top goal `G1` |
| `orphan_nodes` | error | `C3` | unreachable from `G1` |

Totals: **6 error, 1 warning, 2 info**. Coverage computed for that case:
claims argued or declared undeveloped **7/8 = 87.5 %**, artifacts present
**2/3 = 66.7 %**, artifacts fresh of those checkable **0/1 = 0.0 %**, assumptions
discharged **0/1 = 0.0 %**, unverifiable artifacts **1**, orphans **2**, cyclic
regions **0**.

For the complete case: claims **8/8**, artifacts present **6/6**, fresh of
checkable **6/6**, assumptions discharged **1/1**, unverifiable **0**, orphans
**0**, cyclic regions **0**.

---

## 3. Content-hash staleness: the measurement that undercuts the check's name

`validate_staleness.py` → `validate_staleness_output.txt`.

This is the most important table in the document, because the check's name is
stronger than the check. SHA-256 (FIPS 180-4, NIST, August 2015) detects that an
artifact's bytes changed. It carries no information about whether the change
matters. The script makes 20 edits that invert an artifact's meaning and 20 that
preserve it exactly, and counts how many of each are reported stale.

| Check | Method | Result | Tolerance |
|---|---|---|---|
| Material edits reported stale | 20 synthetic edits that invert a verdict or a count | **20 / 20 = 1.000** | exact |
| Immaterial edits reported stale | 20 synthetic edits that add whitespace or a comment | **20 / 20 = 1.000** | exact |
| **Discrimination between the two** | difference of the two rates | **0.000** | exact |
| Unchanged artifact | recorded digest equals current digest | `fresh` | exact |
| Artifact changed then reverted | state after the revert | `fresh` — the change leaves no trace | exact |
| Artifact present, no digest recorded | — | `unverifiable`, never `fresh` | exact |
| Cited path absent | — | `absent` | exact |
| Cited path is a directory | — | `unreadable` | exact |
| `sha256_file` against `hashlib.sha256` of the whole file | 32 MiB artifact | digests agree | exact |
| Hashing throughput | 32 MiB, two shared contended cores | **1 179 MiB/s**, 0.027 s | ±20 % run to run |

**A discrimination of 0.000 is the intended result and the reason the package
says "the bytes changed" rather than "the evidence is invalid".** It is also the
reason the README states the limit in its own voice rather than in a footnote.

Assumptions of the method, which this package does not verify:

- that a SHA-256 digest is a usable proxy for an artifact's identity. This is a
  cryptographic assumption about SHA-256, not an engineering one about the
  artifact, and nothing here tests it.
- that the artifact on disk is the artifact the claim meant. A file moved,
  regenerated byte-identically by a different process, or replaced by an
  identical copy is indistinguishable from the original.
- the check holds no history. **Measured above:** an artifact changed and
  changed back reports `fresh`.

---

## 4. Graph identities over random cases

`validate_graph_identities.py` → `validate_graph_identities_output.txt`,
seed 20261008.

Tarjan's algorithm (Tarjan 1972, "Depth-first search and linear graph
algorithms", *SIAM Journal on Computing* 1(2)) for the strongly connected
components; breadth-first search for reachability. Both are exact, so these are
pass/fail identities with no tolerance.

| Check | Method | Result | Tolerance |
|---|---|---|---|
| A DAG never reports a cycle | 400 random cases, every `SupportedBy` edge from lower to higher index; mean 10.57 nodes, 11.65 edges | **0 false positives, rate 0.000000** | exact |
| A planted cycle is always found | one back-edge added to 372 of 400 generated cases (28 had no eligible edge) | **0 false negatives, rate 0.000000** | exact |
| The reported walk is a real walk | first id equals last; every consecutive pair is an edge of the case | **0 bad walks of 372** | exact |
| Adding orphans does not change the unsupported-claim findings | 400 cases, 1–5 detached Contexts added to each | **0 cases changed** | exact |
| Adding orphans raises the orphan count by exactly that many | same 400 cases | **0 cases wrong** | exact |

Hypothesis covers the same identities over its own generated cases in
`tests/test_properties.py` (12 properties, 60 examples each), including that
node and edge document order does not change the finding set.

---

## 5. CLI exit codes, measured from real subprocesses

`validate_cli.py` → `validate_cli_output.txt`.

The nonzero exit on an incomplete case is a contract. Checking it in-process
would not prove the process exit status, so each of the **22** rows below is an
actual `python -m assuregraph` invocation and the number recorded is
`returncode`.

| Check | Method | Result | Tolerance |
|---|---|---|---|
| Invocations probed | real subprocesses | **22** | — |
| Exit codes matching the contract | 0 complete, 1 incomplete, 2 malformed or usage | **22 / 22** | exact |
| `--help` exits 0 | top level and all three subcommands | **4 / 4** | exact |
| `check complete_case.yaml` | — | **0** | exact |
| `check incomplete_case.yaml` | — | **1** | exact |
| `check cyclic_case.yaml` | — | **1** | exact |
| Missing file, empty-`nodes` case, unparsable YAML | — | **2, 2, 2** | exact |
| A malformed input writes nothing to stdout | — | **0 bytes** | exact |
| `--help` carries the scope statement | whitespace-normalised text search | all 3 phrases present | exact |

The JSON payload of `check incomplete_case.yaml --json` carries
`exit_code = 1`, `counts = {error: 6, warning: 1, info: 2}`, 9 findings,
3 evidence entries, and every coverage fraction next to its denominator
(`claim_support 0.875000 / 8`, `evidence_present 0.666667 / 3`,
`evidence_fresh 0.000000 / 1`, `assumption_discharged 0.000000 / 1`).

---

## 6. Mermaid rendering

`validate_mermaid.py` → `validate_mermaid_output.txt`.

| Check | Method | Result | Tolerance |
|---|---|---|---|
| Six element types render with the documented shape | compare `MERMAID_SHAPES` against the emitted text | **6 / 6** | exact |
| Both relationship arrows emitted | `-->` and `--o` | **2 / 2** | exact |
| Substitution markers present | `A1 [A]`, `J1 [J]`, `(undeveloped)` | **3 / 3** | exact |
| Double quote escaped to `#quot;` | — | yes | exact |
| Newline becomes `<br/>` | — | yes | exact |
| One declaration line per node, one per edge | three shipped cases | **18/18 + 17/17, 14/14 + 12/12, 5/5 + 5/5** | exact |
| Two renders are byte-identical | three shipped cases | **3 / 3** | exact |

**External check, performed once during the build and not re-run by the script.**
A seven-node excerpt of the complete case's diagram — one node of each of the six
element types plus both relationship arrows — was passed to an external Mermaid
renderer on 2026-10-08. It reported `valid: true`, `diagramType: flowchart`,
rendered all 7 node labels, and produced SVG containing `rect`, `polygon` and
`circle` elements, consistent with the four GSN shapes plus the two
substitutions. **The full 18-node diagram was never passed to a renderer**, and
no Mermaid implementation is available in this container, so the repository
cannot re-verify this and does not claim to.

**Where the rendering is not GSN, stated plainly.** Mermaid has a rectangle, a
parallelogram, a circle and a stadium, which carry Goal, Strategy, Solution and
Context. It has no ellipse and no decorator, so Assumption becomes a hexagon
marked `[A]`, Justification an asymmetric shape marked `[J]`, and *Undeveloped*
the text `(undeveloped)` in the label. It has no hollow arrowhead, so
`InContextOf` is drawn `--o`. A reader who knows GSN will recognise the first
four and must be told the rest.

---

## 7. Run cost against case size

`validate_scale.py` → `validate_scale_output.txt`, whole script 18.5 s.

Synthetic cases whose claims form a complete binary tree, with one 1 KiB
evidence artifact per claim leaf, every digest matching. A correct run finds
**zero** findings at every size, which is asserted. Wall clock on two shared
contended cores; the committed output is the run these numbers came from.

| Nodes | Edges | Solutions | Parse s | `run_checks` s | Mermaid s | Findings | YAML KiB | `run_checks` µs/node |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 150 | 149 | 50 | 0.007 | 0.001 | 0.001 | 0 | 20.6 | 8.99 |
| 750 | 749 | 250 | 0.044 | 0.007 | 0.003 | 0 | 105.9 | 9.26 |
| 3 000 | 2 999 | 1 000 | 0.181 | 0.023 | 0.011 | 0 | 433.1 | 7.59 |
| 7 500 | 7 499 | 2 500 | 0.468 | 0.063 | 0.029 | 0 | 1 090.8 | 8.45 |
| 30 000 | 29 999 | 10 000 | 3.048 | 0.337 | 0.168 | 0 | 4 457.5 | 11.25 |

The per-node cost stays between **7.6 and 11.3 µs/node** across a 200× range of
sizes. From 3 000 to 30 000 nodes, **10×** the nodes cost **14.8×** the
`run_checks` wall clock. That is *consistent with* the linear complexity the
modules document and is **not a proof of it**: five sizes on one contended
machine cannot distinguish O(V+E) from O(V log V), and the wall-clock figures
move by 10–20 % between runs. The 150-node row is dominated by fixed overhead
and is not used for the ratio.

**Parsing dominates, not checking.** At 30 000 nodes the YAML load plus schema
validation takes 3.05 s against 0.34 s for all six checks including hashing
10 000 artifacts. Validity range of this table: about 150 to about 30 000 nodes
with 1 KiB artifacts. Published assurance cases run to tens or low hundreds of
elements, so the largest row is a stress case, not a representative one.

---

## 8. Worked example and example figures

`worked_example.py` → `worked_example_output.txt` produces, verbatim, the output
shown in the README's worked-example section.

The four example scripts write their figures to `screenshots/` and their stdout
to this directory:

| Script | Figure | Raw output |
|---|---|---|
| `examples/coverage_summary.py` | `screenshots/coverage_summary.png` | `example_coverage_summary_output.txt` |
| `examples/findings_by_check.py` | `screenshots/findings_by_check.png` | `example_findings_by_check_output.txt` |
| `examples/render_case_graph.py` | `screenshots/complete_case_graph.png`, `screenshots/incomplete_case_graph.png`, plus `.mmd` sources | `example_render_case_graph_output.txt` |
| `examples/staleness_demo.py` | `screenshots/staleness_timeline.png` | `example_staleness_demo_output.txt` |

---

## 9. Errors made during this build, and what was done about them

Recorded here rather than quietly fixed, because a build log that contains no
mistakes is a build log that is not being kept.

1. **The Mermaid line-break substitution was dead code.** `mermaid._label`
   normalised a statement with `" ".join(statement.split())`, which collapses
   newlines along with every other whitespace run, so the `<br/>` substitution
   the module's docstring promised could never fire and a multi-line GSN
   statement rendered as one long line. Found by a test written from the
   docstring. Fixed by normalising each line separately;
   `tests/test_regression.py::test_regression_mermaid_label_keeps_explicit_newlines`
   and its companion now pin both halves of the behaviour.
2. **Every evidence artifact was hashed twice per run.** The first draft of
   `run_checks` let `check_missing_evidence` and `check_stale_evidence` each call
   `inspect_case_evidence`, so each cited file was read from disk twice. Found
   while writing the scale measurement. Fixed by hashing once in `run_checks`
   and passing the reports into both checks; pinned by
   `tests/test_regression.py::test_regression_evidence_is_hashed_once_per_run`.
3. **The first scale generator produced a case full of errors.** It attached
   solutions by index rather than at the leaves, so every claim leaf was an
   unsupported claim and the measurement ran against a case with 8 750 error
   findings at the largest size — which measured the finding-formatting path
   rather than the clean path. The run was discarded, the generator rewritten as
   a binary tree with a Solution under every leaf, and the script now **fails**
   if any finding appears at any size.
4. **A validation script used a hard-coded `/tmp` path.** Replaced with
   `tempfile.TemporaryDirectory` so the script is re-runnable on a machine where
   that path is not writable and leaves nothing behind.
5. **An assertion about `--help` failed for the wrong reason.** The CLI
   validation checked for the phrase "not a DO-178C or ARP4754A compliance tool"
   in `--help` output; argparse had wrapped the epilog across a line break, so
   the substring was absent even though the sentence was present. The check was
   wrong, not the CLI. Fixed by comparing against whitespace-normalised text.

6. **A SHA-256 digest was being split across a line break.** `format_findings`
   wrapped a finding's message with `textwrap.wrap`'s defaults, which break a
   token longer than the wrap column — so a 64-character digest in a
   `stale_evidence` message arrived in the report in two pieces and could not
   be copied. Found while adding a test for the new `width` parameter on
   `format_report`. Fixed with `break_long_words=False, break_on_hyphens=False`,
   which means a line may now exceed the wrap column; that is the better
   defect, and it is documented on the function.
   `tests/test_report.py::test_a_digest_is_never_split_across_a_line_break`
   pins it, and the existing width test was corrected, since it had been
   asserting the wrong contract.

None of these was hidden with `xfail` or a skipped test, and none changed a
published number without the number being recomputed.

---

## 10. What this validation does not establish

- **Nothing about any system.** Every case in this repository is synthetic and
  every evidence artifact under `examples/cases/evidence/` says so in its first
  line. The claims in them are fabricated for demonstration.
- **Nothing about argument soundness.** No check in this package reads the
  meaning of a statement. A case can satisfy all six checks while every Solution
  cites an empty file whose digest was recorded from that empty file.
- **Nothing about evidence adequacy.** Structural completeness and evidence
  adequacy are different properties and this package measures only the first.
- **Nothing about materiality of change.** Measured in section 3: the
  discrimination is 0.000 by construction.
- **No certification support of any kind.** This tool is not part of any
  compliance argument, and a clean exit code from it means only that the case is
  fully drawn and its cited files are the files that were cited.
