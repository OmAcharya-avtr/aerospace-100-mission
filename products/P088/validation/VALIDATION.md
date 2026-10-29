# Validation — tmtcdiff 0.1.0

Validation level 2. Every number in this document was produced by a script in
this directory, executed in the session that wrote the document. Each script
writes its raw output beside itself as `<script>_output.txt`, and those files
are the record; this document summarises them and does not replace them.

**Environment, recorded in the header of every output file.** Python 3.13.18,
numpy 2.5.3, scipy 1.18.1, matplotlib 3.11.2, pytest 9.1.1, hypothesis
6.170.13, ruff 0.17.3 reported by `importlib.metadata` (the package was also
checked clean under the 0.16.9 on `PATH`), 2 CPU cores. `deepdiff` is **not
installed**; it is named in the README's alternatives table and no number here
is taken from it or from a comparison against it. `jsonschema` 4.26.0 is
installed and is not used.

**What everything is measured against.** A seeded synthetic generator,
`src/tmtcdiff/synth.py`, described in `DATASET_CARD.md`. No flight database is
used, read, imported or approximated anywhere in this repository.

**Totals.** 9 scripts, 51 named checks, 0 failures. 608 tests, 0 failures, 0
errors, 0 skipped, counted from the `tests` attribute of the junit XML and
cross-checked against a literal count of `testcase` elements (608 and 608).

---

## 1. Per-class detection — `validate_change_classes.py`

Two regimes over 21 change classes, 2 dictionary sizes (30, 120) and 10 seeds.

**Isolated** — one class injected per pair, 420 injections:

| | precision | recall | TP | FP | FN |
|---|---|---|---|---|---|
| micro-averaged, all 21 classes | 1.0000 | 1.0000 | 420 | 0 | 0 |
| silent tier, 11 classes | 1.0000 | 1.0000 | 220 | 0 | 0 |
| loud tier, 7 classes | 1.0000 | 1.0000 | 140 | 0 | 0 |

Every class fired on every one of its 20 pairs.

**Combined** — all 21 classes injected into the same pair, 420 injections.
Micro-averaged precision 0.9665, recall 0.9619, TP 404, FP 14, FN 16. The
classes that lost ground, with the cause:

| class | precision | recall | what happened |
|---|---|---|---|
| `command_removed` | 1.0000 | **0.7000** | 6 of 20 removals were absorbed into an inferred rename |
| `command_added` | 1.0000 | **0.7000** | the other side of the same 6 pairings |
| `command_renamed` | **0.7692** | 1.0000 | 6 false renames, the same 6 |
| `removed` | 1.0000 | **0.9000** | 2 of 20 absorbed into an inferred rename |
| `added` | 1.0000 | **0.9000** | the other side of the same 2 |
| `renamed` | **0.9091** | 1.0000 | 2 false renames |
| `description_changed` | **0.8696** | 1.0000 | 3 false renames carried a description difference |
| `bit_shifted` | **0.9091** | 1.0000 | 2 false renames carried an offset difference |
| `reenumerated` | **0.9524** | 1.0000 | 1 false rename carried an enumeration difference |

No silent-tier class lost recall in either regime.

**Detection floor on a relative scale change**, by bisection rather than read
off the configuration:

| configured `numeric_rel_tol` | smallest detected relative change | ratio |
|---|---|---|
| 1e-12 | 9.999778782798785e-13 | 0.999978 |
| 1e-09 | 9.9999997171806854e-10 | 1.000000 |
| 1e-06 | 1.0000009998956119e-06 | 1.000001 |

**No-change control**: a dictionary diffed against itself reported 0 changes
on all 20 pairs.

Checks: 5, failures 0.

---

## 2. The silent tier, separately — `validate_silent_changes.py`

Sizes 40 and 160, 10 seeds, all 21 classes injected into each pair.

| tier | injected | TP | FP | FN | precision | recall |
|---|---|---|---|---|---|---|
| silent | 220 | 220 | 0 | 0 | 1.0000 | 1.0000 |
| loud | 140 | 139 | 1 | 1 | 0.9929 | 0.9929 |

Detection keys are deduplicated per pair, the same convention
`validate_change_classes.py` uses: a class detected on one item through two
fields counts once. `argument_reordered` always moves two arguments, so
counting raw changes would charge it a false positive for every true
detection and report a precision of 0.5000 that is an artefact of the counting
rule. That was found and corrected during the build.

**Ranking.** 20 pairs, all 20 carrying both tiers: 0 ordering violations in
the ranked diff and 0 in the rendered report.

**Silence is testable.** Over isolated injections:

- every silent change leaves the old by-name reference resolving: **220 of 220**;
- every resolution-breaking loud class stops the old reference resolving:
  **100 of 100** (`removed`, `renamed`, `command_removed`, `command_renamed`
  by name, `path_moved` by path).

**The one tier assignment this package cannot verify.** `argument_added` is
classified loud on the assumption that a command system rejects a call missing
an argument. Measured over 20 injections: the command still resolves 20 of 20
and every pre-existing argument is still present 20 of 20. Nothing here
confirms the call fails. If an uplink system defaults the missing argument
instead, this class is silent and is ranked one tier too low by this taxonomy.

**Why a removed enumeration label is silent.** Demonstrated on `OBDH_S_GYRO_23`:
the label is removed, the parameter still resolves, and an equality test
against the removed label never fires and raises nothing. The same run shows a
second finding: the removed label is in use on another parameter, so a bare
label reference cannot be attributed to one of them. That ambiguity is counted
in `CoverageVerdict.ambiguous_enum_labels`.

Checks: 7, failures 0.

---

## 3. Rename detection — `validate_rename_sweep.py`

A grid of 12 thresholds (0.40 to 0.95 in steps of 0.05), 5 dictionary sizes
(20, 50, 100, 200, 400), 12 seeds and 2 matching strategies. Each pair carries
`size // 10` true renames plus 2 unrelated removals and 2 unrelated additions;
the unrelated items are what make a false match possible at all.

**False-match rate, optimal matcher** (false accepted pairs over accepted
pairs):

| threshold | n=20 | n=50 | n=100 | n=200 | n=400 | false | accepted |
|---|---|---|---|---|---|---|---|
| 0.40 | 0.2258 | 0.1781 | 0.0840 | 0.0438 | 0.0224 | 53 | 977 |
| 0.50 | 0.1111 | 0.0625 | 0.0551 | 0.0244 | 0.0144 | 27 | 951 |
| 0.60 | 0.0769 | 0.0476 | 0.0551 | 0.0164 | 0.0062 | 19 | 943 |
| 0.70 | 0.0400 | 0.0164 | 0.0164 | 0.0000 | 0.0000 | 4 | 928 |
| 0.75 | 0.0400 | 0.0000 | 0.0083 | 0.0000 | 0.0000 | 2 | 926 |
| **0.80** | **0.0000** | **0.0000** | **0.0000** | **0.0000** | **0.0000** | **0** | **924** |
| 0.90 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 794 |

**Recall, optimal matcher**, pooled over sizes: 924/924 = 1.0000 at every
threshold up to and including 0.85; 794/924 = 0.8593 at 0.90, 95% interval
[0.8352, 0.8811]; 674/924 = 0.7294 at 0.95, interval [0.6996, 0.7579].

**A confound, named rather than left in the table.** The number of injected
renames grows with dictionary size while the unrelated removals and additions
stay at 2 each, so the denominator of the rate is dominated by true renames at
the larger sizes and the rate falls with size partly by construction. Counted
against the opportunity instead — 24 unrelated removals and 24 unrelated
additions per cell — the dependence on size is weak and not monotone, and the
dependence on threshold is the strong effect. The full table is in the output.

**Threshold selection.** The criterion was declared in the script before the
grid was run: *rule 1 — among thresholds whose measured false-match rate is
exactly zero at every swept size, take the lowest; recall is non-increasing in
the threshold, so the lowest qualifying threshold is the highest-recall
qualifying one.* Rule 1 fired and selected **0.80** for both strategies. The
script compares that selection against the shipped constant
`tmtcdiff.diff.DEFAULT_RENAME_THRESHOLD` and **fails** on a disagreement
rather than writing the value back; it failed exactly that way on the first
run of this build, when the constant was still a placeholder.

**Baseline comparison, and an unfavourable result.** At the selected threshold
the greedy baseline and the optimal rectangular assignment are
indistinguishable: 924 correct and 0 false for both, at every size. Across the
entire grid the largest difference between them in any cell is **2 false
matches**, and they differ in both directions. The optimal matcher remains the
default because it cannot be worse by construction, not because it measured
better. It did not.

**The independently derived bound.** Exact structural-signature collisions,
counted from the signature multiset by the occupancy argument (Feller 1968,
chapter II), using nothing the matcher computes:

| size | parameters | distinct signatures | colliding items | fraction |
|---|---|---|---|---|
| 20 | 20 | 20 | 0 | 0.0000 |
| 50 | 50 | 50 | 0 | 0.0000 |
| 100 | 100 | 100 | 0 | 0.0000 |
| 200 | 200 | 197 | 6 | 0.0300 |
| 400 | 400 | 388 | 24 | 0.0600 |

A non-zero collision fraction is why the score carries a name term at all.

Checks: 2, failures 0.

---

## 4. The negative control — `validate_negative_control.py`

The control required of this product: a constructed pair with **no** semantic
change but substantial textual reordering, verified to produce an empty
semantic diff. It is built by re-serialising one dictionary with its entries,
its keys inside every entry, its enumerations and its command-argument
declarations shuffled, with a different indent, and with a different accepted
spelling for most keys (`mnemonic` for `name`, `bits` for `bit_width`,
`factor` for `scale`).

24 pairs over sizes 20, 60, 150 and 300 and 6 seeds. **0 semantic changes on
all 24**, and 0 round-trip identity failures. "Substantial" as a number: at
size 60 seed 7400, 2526 of 1267 canonical lines differ (every line deleted and
re-added), a fraction of 1.9937; at size 300, 12572 lines.

**What a text diff says on the same pair.** `difflib`, which is named in the
README's alternatives table and is doing exactly its job:

| size | text lines | difflib lines differing | difflib changed blocks | difflib line ratio | tmtcdiff semantic changes |
|---|---|---|---|---|---|
| 20 | 435 | 862 | 3 | 0.0092 | 0 |
| 60 | 1267 | 2526 | 3 | 0.0032 | 0 |
| 150 | 3142 | 6276 | 3 | 0.0013 | 0 |
| 300 | 6290 | 12572 | 3 | 0.0006 | 0 |

**The control on the control.** A control that cannot fail is not evidence.
Each of the 11 silent classes was injected into the reordered document, 6
pairs each: **66 of 66 still detected**, with **0 extra changes** reported
across all 66. Both mechanisms were also run alone — reorder only, and reorder
plus respell — and each independently produces an empty semantic diff.

Checks: 6, failures 0.

---

## 5. Impact classification and the coverage verdict — `validate_impact_report.py`

**Part 1, the classifier.** For each of 17 change classes, 8 seeds, a pair with
that class injected alone and a one-file corpus holding exactly three
references: one to the injected item in the form that class affects, one to an
item no injection touched, and one assembled by a run-time expression. 408
references probed, **0 misclassified**; the confusion table is diagonal.
Rename suggestions: **8 of 8** named the injected target.

**Part 2, the verdict.** Five constructed cases:

| case | references | undecidable | coverage | items unreferenced | determinable | reasons |
|---|---|---|---|---|---|---|
| A full coverage | 75 | 0 | 1.0000 | 0 | True | 0 |
| B changed item outside the corpus | 10 | 0 | 0.1333 | 65 | False | 2 |
| C unreadable source present | 10 | 2 | 0.0267 | 73 | False | 3 |
| D empty corpus | 0 | 0 | 0.0000 | 75 | False | 3 |
| E full coverage, nothing changed | 75 | 0 | 1.0000 | 0 | True | 0 |

An independent count of referenced items agrees with the verdict's own count
(75 against 75).

**Part 3, the statement this package must not make.** Case B has a silent
change in the dictionary and a corpus that does not reference the changed
item. The actionable finding list is **empty**. The rendered interpretation
reads:

> No affected reference was found among the references this tool could
> resolve, and the assessment is NOT complete: 65 of 75 dictionary items are
> never referenced by this corpus, so this assessment says nothing about
> consumers outside it; 1 changed item(s) are not referenced by this corpus:
> THER_T_BUS_24. An empty finding list under partial coverage is an absence of
> evidence, not evidence of absence.

Case E, where the assessment *is* complete and empty, still scopes itself:
"That is a statement about this corpus and these two dictionary versions
only." Five rendered reports were scanned for the words *safe*, *safely*,
*certified*, *flight-qualified* and *production-ready* as whole words: **0
occurrences**.

Checks: 13, failures 0.

---

## 6. Independently derived bounds — `validate_bounds.py`

"Two implementations agree" and "the answer lies inside a bound derived from
different mathematics" are not the same check. Five routes, 24 pairs each over
sizes 30, 80 and 200 and 8 seeds.

| bound | what it compares against | violations |
|---|---|---|
| 1. cardinality conservation | set cardinalities alone, not the diff walk | 0 of 24 |
| 2. a silent change exists iff the structural signature tuple differs | one tuple equality against per-field classification, with the tolerance set to exactly zero so the two predicates coincide | 0 items disagreed |
| 3. accepted perfect-structural matches ≤ identical-signature pairs | the signature multisets, by the occupancy counting argument | 0 of 24 |
| 4. missed renames = renames scoring below the threshold | the score distribution alone | 0 of 24 |
| 5. union bounds on the impact findings | counts over the reference list | 0 of 24 |

Bound 3's measured slack is **0 at every swept size**, so on this generator the
bound is tight rather than loose: every injected rename preserves the
structural signature exactly and every such pair also clears the combined
threshold. A generator that perturbed a field across the rename would open the
slack up, and the bound would still hold. Every bound here is an integer
counting bound, so no slack constant is involved and the question of how a
slack should scale with the conditioning of the bound expression does not
arise — a property of this problem, not a general escape.

A structural control is also run: renaming an item and clearing its path
leaves its structural signature identical and its structural term at 1.0000,
while its name term falls. The signature cannot see a name, which is what
makes a rename inferable at all.

Checks: 6, failures 0.

---

## 7. The command-line interface — `validate_cli.py`

14 invocations run as subprocesses, each with a temporary `HOME` and
`MPLCONFIGDIR` outside the repository and `PYTHONDONTWRITEBYTECODE=1`. Every
invocation exited with its expected status, including the three that are
expected to fail: a missing file (1), an out-of-range threshold (1) and an
unknown subcommand (2), and `--fail-on-silent` (2).

The impact output names the unreadable corpus source and prints the coverage
verdict. No transcript contains a safety or certification word as a whole
word, and none contains the absolute repository path. The tree is compared
before and after: **0 files gained beyond the script's own output, 0 lost**.

Checks: 6, failures 0.

---

## 8. The worked example — `worked_example.py`

Two dictionaries written out by hand, four parameters and one command, with a
procedure and an analysis script written against the older one. 7 silent
changes, 1 loud, 1 advisory. 7 silently changed references, 0 broken, 1 renamed
away, 1 undecidable. Coverage of the dictionary by the corpus 1.0000, and the
assessment is still **not fully determinable**, because 1 of 9 references is
assembled by a run-time expression.

The entry worth reading twice: `SEND SET_HEATER(HEATER_ID=3, DUTY=50)` parses
in both versions, is accepted by both, and in version 1.1 puts 50 into the
heater identifier.

Checks: 5, failures 0.

---

## 9. Working directory — `probe_working_directories.py`

Every validation script and the worked example run from five working
directories: the filesystem root, the system temporary directory, the
repository root, `validation/` (which is what the release gate uses) and
`src/`. **40 of 40 exit 0.** Elapsed times are recorded in the output in a
table explicitly declared unstable; no conclusion in this repository rests on
a timing or on a ratio of timings.

Checks: 1, failures 0.

---

## Checks that failed

None in the final run. Two failed during the build and were fixed at their
cause rather than by moving a tolerance:

1. `validate_rename_sweep.py` failed its own threshold check on the first run,
   because the shipped `DEFAULT_RENAME_THRESHOLD` was a placeholder and the
   criterion selected a different value. The constant was set to the selected
   value; the check is permanent and will fail again if the two diverge.
2. `validate_impact_report.py` failed its full-coverage determinability check
   because every enumeration label shared by two parameters was reported as
   undetermined even with no corpus reference to it. The library was corrected
   to report an ambiguous label only when the corpus references it.

A third problem was not a failed check but a vacuous one: the first generator
gave every parameter a globally unique bit offset, so the measured rename
false-match rate was exactly zero at every threshold from 0.40 to 0.95. That
is a broken experiment, not a result. The generator was changed so bit offsets
wrap inside each packet, and the sweep then produced the curve above.

## What this validation does not establish

- That detection rates transfer to a real mission database. Everything here is
  measured on a seeded synthetic generator whose biases are listed in
  `DATASET_CARD.md`.
- That the within-tier ordering is correct. Only the three tiers are argued
  for, and only the silent-above-loud claim is checked. Nothing here ranks a
  rescale against a re-enumeration.
- That `argument_added` belongs in the loud tier. See section 2.
- That an upgrade is safe. This package does not produce that verdict and no
  measurement here supports one.

Research grade. Not flight-qualified, not certified, not approved for
operational aerospace use.
