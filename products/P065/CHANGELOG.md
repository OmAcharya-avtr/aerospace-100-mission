# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-15

First release. Status: TESTING. Validation level 2. No AI component, by design,
and therefore no MODEL_CARD.md and no DATASET_CARD.md.

### Added

- `model.py`: a typed model of an STPA analysis in the vocabulary of the STPA
  Handbook (Leveson and Thomas, 2018) - losses, hazards, system-level
  constraints, the control structure, unsafe control actions in the four
  classes, justified exclusions, loss scenarios, mitigations and evidence.
- `parse.py`: a strict YAML reader. Unknown keys are rejected rather than
  ignored, dangling references are rejected at parse time, duplicate keys in one
  mapping are rejected rather than silently collapsed, and every rejection names
  the offending key path. 49 schema rejection paths and 3 I/O paths exercised.
- `enumeration.py`: the `4n` UCA enumeration grid, with a cell covered by a UCA,
  by a justified exclusion, by both (a reported contradiction) or by nothing (an
  enumeration gap).
- `checks.py`: sixteen checks, HT001 to HT016, each naming the objects it is
  about, with coverage ratios that always carry their denominator and are
  reported as undefined rather than as 100 % when the denominator is zero.
- `crosslinks.py`: independent parsers for a markdown requirements document and
  a junit XML report, so that requirement and test coverage is computed from the
  repository rather than asserted in the analysis. Four identifier alias forms
  are accepted and all five pytest outcomes are classified.
- `report.py`: text, JSON and markdown renderers, each of which repeats the
  scope statement next to the verdict.
- `diagram.py`: a Mermaid renderer for review in a pull request, and a
  matplotlib (Agg) figure of the control structure beside the enumeration grid.
- CLI `python -m hazardtrace` with `check`, `grid`, `coverage`, `diagram` and
  `codes`; exit 0 with no blocking finding, 1 with at least one, 2 when the
  command could not run.
- Three shipped cases: a clean minimal one, a hand-constructed one carrying
  exactly one instance of every check, and the Level 2 worked example - an STPA
  of the B-dot detumble loop implemented in P024 DetumbleSim of this portfolio,
  published with its eleven real gaps rather than tidied.
- Two committed cross-link fixtures: a junit report of a real 305-test run of
  P024's own suite, normalised by `fixtures/regenerate_fixtures.py`, and a
  hand-written five-outcome report for the checks that need a failing test.
- Four runnable examples, each writing a PNG into `screenshots/`.
- Eight validation scripts with their raw output committed beside them, in the
  flat `validation/<script-name>_output.txt` layout.
- 215 tests: unit, input validation, known-answer with the hand arithmetic in
  the test comments, edge cases, Hypothesis property tests over the grid algebra,
  pinned regression tests including the one counterexample Hypothesis found,
  diagram smoke tests and CLI subprocess tests that assert on exit status.

### Known limitations at this release

See the Limitations section of README.md. The ones that bite first: the tool
reads structure and never content, so a UCA whose description is the word "bad"
is indistinguishable from a real one; the finding count is not monotone, because
filling an enumeration gap removes one HT001 and can add two downstream
findings; a context table in the sense of Thomas (2013) is not implemented, so
cell coverage is coarser than a full STPA would require; and an xpassed test is
indistinguishable from a passing one in junit XML, which is pytest's limitation
and is inherited here.
