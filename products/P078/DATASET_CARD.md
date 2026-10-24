# Dataset card — the seeded synthetic corpus

**No flight data is used anywhere in this repository.** Every learned result
and every miss-rate figure in this repository is a statement about the
generator described here, and not about any real mission's command sequences.
Nothing here was recorded from a spacecraft, derived from a published command
dictionary, or validated against operational data, because no such data was
available to this build.

## What is committed, and what is not

Nothing in this repository is a data file larger than a few kilobytes. The
corpus is **not committed**: it is regenerated deterministically from seeds by
`cmdguard.dataset.build_corpus`, and every script that uses it states its seeds.
What *is* committed is five hand-written case files in `examples/cases/`, which
are the independently checkable part of the validation and are small enough to
read.

## Generation

`cmdguard.dataset`. Three parts.

**A fixed command dictionary.** 29 commands over seven subsystems — power,
thermal, ADCS, payload, recorder, comms, propulsion, structures and OBDH — with
five modes, ten predicates, two resource ceilings (120.0 W and 40.0 Mbps before
per-case jitter), one mutual-exclusion group of three commands, and four
commands flagged irreversible (`ANT_DEPLOY`, `MEM_SCRUB`, `PROP_BURN`,
`PROP_VALVE_OPEN`).

The dictionary is **fixed and hand-written, not random**. A randomly wired
dictionary produces sequences whose legality is an accident of the wiring, and a
risk ranker trained on it learns the wiring. It is a plausible shape for a small
Earth-observation spacecraft and it is not any real vehicle's dictionary; no
numerical value in it is taken from a published source.

**Per-case variation.** What varies per seed is what varies on a real mission:
which conditions are verified at the start (`unverified_fraction`, default
0.25), where the resource ceilings sit (`limit_jitter`, ±10 per cent), the
per-command draws and holds (`draw_jitter` ±15 per cent, `hold_jitter` ±20 per
cent), and the sequence itself.

**Goal-directed sequence construction.** The generator repeatedly draws an
activity from a weighted list — a payload scan, a downlink, a propulsive burn, a
memory scrub, a safing — resolves its prerequisite chain over the declared
dictionary and mode machine, and appends the chain one command at a time. Each
append is accepted only if the forward executor reports no violation at the new
step, so the result is **violation-free by construction**.

The goal-directed design is a correction of a measured defect in the first
version of this generator. A uniform random walk over the dictionary placed
mostly teardown commands, which have no preconditions and therefore always
succeed: measured during this build over 40 seeds, it produced 125
`THERM_HEATER_OFF` and **one** `PROP_VALVE_OPEN`, no `PAYLOAD_SCAN` at all, and
two of the nine defect families were uninjectable because the commands they need
never appeared. That version of the generator is not in this repository, so
those three counts are the one set of figures here that a reader cannot
re-derive; they are recorded because they are the reason for the design. The weights are
in `cmdguard.dataset.ACTIVITY_WEIGHTS` and the property is asserted by
`tests/test_dataset.py::test_generated_sequences_reach_the_interesting_commands`.

## The defect population

`cmdguard.defects`, nine families, each a textual edit of the sequence or the
dictionary under its own seed. Eight are reachable only by a sequence-aware
check; `undeclared_typo` is the control that a per-command lint finds every
time, so the headline miss rate is a rate over a mixed population rather than
one selected to embarrass the baseline.

| Family | The edit |
|---|---|
| `swap_enabler` | A step that satisfies a later step's precondition is moved after it. |
| `delete_enabler` | That step is deleted. |
| `insert_invalidator` | A command that falsifies the needed predicate is inserted between them. |
| `mode_drop` | A transition out of the mode a later gated command needs is inserted. |
| `compress_gap` | A step is moved earlier so a declared `min_gap_after_s` is not met. |
| `overlap_exclusion` | A step is moved inside another exclusion-group member's hold. |
| `resource_spike` | A drawing step is moved inside other holds so a ceiling is crossed. |
| `irreversible_early` | An irreversible command is moved ahead of its verification, with the tail shifted so the edit is a reordering and not also a timing defect. |
| `undeclared_typo` | A mnemonic is mistyped. |

**No-op injections are counted, not retried.** An edit that produces no new
violation is recorded as a no-op and the count is printed; over the 60-seed
corpus in `validation/validate_lint_missrate_output.txt` there were 14 no-ops in
540 injections, concentrated in `irreversible_early` (7) and `resource_spike`
(6). Retrying until an injection bites would silently reshape the population.

Two arithmetic defects in these families were found by their own no-op counts
and are pinned as regression tests:
`tests/test_regressions.py::test_compress_gap_family_no_longer_uses_min_interval_as_a_gap`
records the first, where 18 of 40 injections were no-ops because
`min_interval_s` was treated as if it were measured from the previous step.

## Size, as measured

From `validation/validate_lint_missrate_output.txt` §1 and
`validation/validate_ranker_output.txt` §1:

| Corpus | Seeds | Cases | Command occurrences |
|---|---|---|---|
| Miss-rate corpus | 60 (0–59) | 600 (60 clean, 540 defective, 526 applied) | 21660 |
| Ranker training | 70 (0–69) | 630 | 22750 |
| Ranker held-out | 35 (100000–100034) | 315 | 11375 |
| Negative control | 60 per shape, 7 shapes | 420 | 15317 |

Labels for the ranker come from the forward walk and nothing else; the features
come from the declarations and nothing else.

## Splits

Training and held-out seeds are disjoint ranges, checked by
`cmdguard.benchmark.assert_disjoint`, which raises. There is no validation
split: no hyperparameter was selected on measured performance. The forest's 200
trees and depth 8 were fixed before the comparison was run, and the one
variation reported — the timing-feature ablation — is reported as an ablation
beside the full model and not used to choose between them.

## Known limitations of this corpus

1. **It is synthetic.** Nothing here is evidence about the frequency or the
   shape of real commanding defects. A real mission's sequences are written by
   people with intentions, and its defects are correlated with schedule
   pressure, hand-offs and replans. None of that is modelled.
2. **The defects are injected, and the injection leaves traces.** Four families
   edit the time column, and the ranker can learn that. The size of the effect
   is measured — 0.056291 of precision at k=5 — and reported in `MODEL_CARD.md`
   rather than left for a reader to suspect.
3. **Clean cases are clean by construction, using the executor under test.** A
   false alarm the executor raises consistently would not show up in the
   negative control. The hand-written cases are the independent part and there
   are two of them. This is stated in
   `validation/validate_negative_control_output.txt` §4.
4. **One dictionary.** Every generated case uses the same 29 commands with
   jittered numbers. Generalisation across dictionaries is not measured and
   there is no basis here for claiming any.
5. **The sequences are short.** 36 steps by default, 240 in the largest timing
   probe. Real uplink loads run to thousands of commands, and while the forward
   walk is linear in length, nothing about the ranker's behaviour at that length
   is measured.

## Licence and provenance

Generated by code in this repository, Apache-2.0, © 2026 OPTIMA Organisation.
No third-party data, no scraped content, no personal data.

## Reproducing the corpus

```bash
python -c "import sys; sys.path.insert(0, 'src'); \
from cmdguard.dataset import build_corpus; \
print(len(build_corpus(range(60))))"
python validation/validate_negative_control.py
python validation/validate_lint_missrate.py
python validation/validate_ranker.py
```
