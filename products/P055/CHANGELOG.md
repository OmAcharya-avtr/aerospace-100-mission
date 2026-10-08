# Changelog

All notable changes to assuregraph are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-10-08

First release. Status **TESTING**: the test suite passes and the validation
evidence is committed, but the package has had no external review.

### Added

- **Case format.** A YAML declarative format for a Goal Structuring Notation
  assurance case, following the GSN Community Standard Version 3 (SCSC-141C,
  SCSC Assurance Case Working Group, May 2021), Core GSN: the six element types
  (Goal, Strategy, Solution, Context, Assumption, Justification), the two
  relationship types (`SupportedBy`, `InContextOf`), and the *Undeveloped*
  decorator.
- **`assuregraph.parse`.** Schema validation that rejects a malformed case with
  a message naming the offending YAML key path. 51 schema rejection paths and 3
  I/O rejection paths are exercised by `validation/validate_parser_rejections.py`.
  Two Core GSN relationship rules are enforced at parse time rather than
  reported as findings.
- **`assuregraph.checks`.** The six computed checks: `unsupported_claims`,
  `missing_evidence`, `stale_evidence`, `undischarged_assumptions`, `cycles`,
  `orphan_nodes`. Every finding carries the offending node ids and a severity.
- **Evidence staleness by SHA-256 content hash** (`assuregraph.evidence`), with
  five distinct states: `fresh`, `stale`, `unverifiable`, `absent`,
  `unreadable`. An artifact with no recorded digest is never counted as fresh.
- **`assuregraph.graph`.** Iterative Tarjan strongly connected components and
  breadth-first reachability, so a 4 000-deep case cannot exhaust the recursion
  limit.
- **`assuregraph.mermaid`.** Mermaid flowchart rendering with GSN shapes, a
  documented substitution table for the three shapes Mermaid lacks, and optional
  highlighting of nodes named by an error finding.
- **`assuregraph.report`.** Text, Markdown and JSON reports. Every coverage
  fraction is printed with its denominator; an undefined fraction prints as
  `n/a (denominator 0)` rather than as 0 % or 100 %.
- **CLI** `python -m assuregraph` with `check`, `mermaid` and `summary`
  subcommands. Exit codes 0 (complete), 1 (incomplete, `check` only), 2
  (malformed input or usage error). The nonzero exit is a tested contract,
  verified by 22 real subprocesses.
- **Three example cases** under `examples/cases/`: complete (exit 0), incomplete
  (exit 1, 6 error / 1 warning / 2 info findings), cyclic (exit 1). All
  synthetic, with synthetic evidence artifacts that say so in their first line.
- **Four example scripts** writing figures to `screenshots/`, and eight
  validation scripts writing raw output to `validation/`.
- **248 tests**, including malformed-input tests for every rejection path,
  known-answer tests with the expected findings written in the test comments,
  12 Hypothesis property tests, integration tests against the shipped cases,
  and six regression tests.

### Deliberately not added

- **Any AI or learned component.** A learned judgement about whether an argument
  is sound is the thing this tool must not automate. This is a design decision,
  recorded in the README, not a gap.
- **Modular GSN, Argument Patterns, Confidence arguments, Dialectic arguments,
  Argument Claim Points.** Core GSN only.
- **A graphical editor.** ASCE, AdvoCATE and the other tools named in the
  README's alternatives table do that, and better.

### Known limitations at this release

See the README's Limitations section. The two that matter most: content-hash
staleness detects that an artifact changed and **not** whether the change
matters (measured discrimination: 0.000), and structural completeness says
nothing about whether the evidence is adequate.
