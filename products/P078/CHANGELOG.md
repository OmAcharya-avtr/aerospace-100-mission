# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-10-24

First release. Status: TESTING. Validation level 2, research grade.

### Added

- `cmdguard.model`: the declarations — `CommandSpec` with preconditions, mode
  gating, effects, resource draw, hold, minimum same-command interval, minimum
  inter-command gap, an irreversibility flag and a mutual-exclusion group;
  `CommandDictionary`, `ResourceLimits`, `ModeMachine`, `InitialState`,
  `SequenceStep`, `CommandSequence` and `Case`. Predicates are three-valued
  (`Verity.TRUE`, `FALSE`, `UNKNOWN`) and `UNKNOWN` never satisfies a
  precondition.
- `cmdguard.trace`: baseline 1 and the ground truth — forward symbolic
  execution of a linear sequence through the mode machine, reporting ten
  violation kinds, each with the `Provenance` chain of earlier steps that
  produced the state it tripped over, plus a `StateSnapshot` per step and an
  order-insensitive `fingerprint` used by the determinism tests.
- `cmdguard.lint`: baseline 2 — the per-command static precondition lint, at
  two strengths, sharing the `(step, kind)` identity of a trace violation so
  the miss rate is computable.
- `cmdguard.gate`: the irreversible-command gate, whose two verdicts are
  `REFUSE` and `NO_REFUSAL_FOUND`.
- `cmdguard.defects`: nine seeded defect families, eight reachable only by a
  sequence-aware check and one control that a per-command lint always finds.
- `cmdguard.dataset`: a fixed 29-command dictionary, a five-mode machine, and a
  goal-directed generator that builds violation-free sequences by rejection
  against the forward executor.
- `cmdguard.features` and `cmdguard.ranker`: twenty static features and a
  random-forest risk ranker with per-tree dispersion as an uncertainty output,
  scored against the irreversibility-flag ordering, the resource-draw ordering
  and a seeded random control on precision and recall at k.
- `cmdguard.benchmark`: the two published measurements — the static-lint miss
  rate and the precision-at-k comparison — with a paired percentile bootstrap.
- `cmdguard.report` and a CLI with `info`, `check`, `lint`, `compare`, `gate`,
  `rank` and `benchmark`, exiting 0, 1, 2 or 3.
- Five committed case files, two of them clean, and one deliberately broken
  declaration that the loader must reject.
- Eleven validation scripts writing their raw output beside themselves, five
  examples writing PNGs to `screenshots/`, and 295 tests.

### Known at release

- The learned ranker beats both free orderings on this corpus at every cut-off
  measured, and roughly a third of its advantage at k=5 comes from three timing
  features that encode the defect-injection process rather than anything about
  commanding. The ablated figure is reported beside the full one.
- The risk score is not calibrated: the weighted mean absolute calibration gap
  is 0.331780. It is a ranking score and the model card says so.
- Dependency ceilings are declared at the minor versions measured, but only the
  single installed patch release of each was executed, because installing into
  this build container is prohibited. See README.md under Limitations.
