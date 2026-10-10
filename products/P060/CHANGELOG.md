# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-10

First release. Status: TESTING. Validation level 1. No AI component, and
therefore no MODEL_CARD.md and no DATASET_CARD.md.

### Added

- `requirements.py`: a line-oriented parser for numbered requirements in
  markdown, recognising heading, bullet and bold declaration forms, skipping
  fenced code blocks, retaining duplicate declarations, and deliberately not
  treating a mid-sentence mention of an id as a declaration.
- `testreports.py`: a junit XML parser that classifies each case as passed,
  failed, errored, skipped or xfailed following pytest's own writer, reads
  requirement claims from configured `<property>` names and optionally from
  the node id, and records the `file` and `line` attributes when
  `junit_family = xunit1` supplies them. Also a parser for
  `pytest --collect-only -q`, whose cases carry no outcomes by construction.
- `matrix.py`: the bidirectional mapping and three coverage figures —
  nominal, executed and passing — each carrying its numerator, denominator
  and definition, and each reported as undefined rather than zero when the
  denominator is zero or the report carries no outcomes.
- `findings.py`: the six checks. TA001 untraced requirement, TA002 undeclared
  claim, TA003 duplicate id, TA004 requirement traced only by skipped or
  xfailed tests, TA005 requirement traced only by failing tests, TA006 the
  heuristic empty-assertion flag. Any code can be moved out of the
  exit-status calculation while still being reported.
- `assertions.py`: a syntactic heuristic for whether a test function asserts
  anything, with six documented rules, no call following, and measured error
  rates — precision 0.625000, recall 0.625000, false-positive rate 0.200000,
  false-negative rate 0.375000 on a 23-function hand-labelled corpus.
- `report.py`: text, JSON and markdown-table rendering, every one of which
  carries the statement that traceability is a bookkeeping condition and not
  evidence of adequacy.
- `markers.py`: a helper for a three-line `conftest.py` that records
  `@pytest.mark.verifies("REQ-NNN")` ids into junit XML, using the same
  `requirement_id` property name as `pytest-requirements` 0.3.1.
- `config.py`: a JSON configuration covering the id pattern, the declaration
  pattern, fence skipping, claim property names and separators, node-id
  claiming, ignored codes and the heuristic switch, rejecting an unknown key
  by name rather than ignoring it.
- `plotting.py`: four Agg figures — trace matrix, coverage definitions,
  finding breakdown and the heuristic's confusion matrix.
- CLI `python -m traceaudit` with `audit`, `requirements`, `tests`,
  `assertions`, `codes` and `config-template`; exit 0 with no blocking
  finding, 1 with at least one, 2 when the audit could not run.
- Bundled fixtures: a sample project with a requirements document, a pytest
  suite and a committed junit XML and collection report produced by a real
  pytest run and normalised for reproducibility; and a hand-labelled corpus
  of 23 test functions for the heuristic.
- `docs/REQUIREMENTS.md`: 26 numbered requirements for this package itself,
  each claimed by at least one test, so that the tool audits its own
  repository as the second worked example.
- Four runnable examples, each writing a PNG into `screenshots/`.
- Nine validation scripts with their raw output committed beside them. All
  nine exit 0, including the one that runs commands which deliberately exit 1.
- 236 tests: unit, input validation, known-answer with the hand arithmetic in
  the test comments, edge cases, Hypothesis property tests for the coverage
  chain and the mapping partition, plotting smoke tests, and CLI subprocess
  tests that assert on exit status.

### Known limitations at this release

See the Limitations section of README.md. The ones that bite first: the
assertion heuristic is wrong about one asserting test in five; an xpassed test
is indistinguishable from a pass in junit XML, so a requirement traced only to
one is reported as fully covered; a pytest collection report carries no
outcomes and no claims, which costs two checks and seven traces on the sample
project; and an unclosed code fence hides every requirement after it, which
makes coverage look better rather than worse.
