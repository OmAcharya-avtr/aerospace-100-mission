# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-16

First release. Status: TESTING. Validation level 1, which is the educational
level of this portfolio and was chosen deliberately: the subject is arithmetic
that everyone does in a spreadsheet and gets subtly wrong. No AI component,
by design, and therefore no MODEL_CARD.md and no DATASET_CARD.md.

### Added

- `model.py`: a typed model of a hierarchical budget that keeps deterministic
  contingency and random uncertainty strictly apart. Contingency is summed,
  never root-sum-squared; the module docstring says why, with its sources,
  units, assumptions and validity range.
- `correlation.py`: the declared leaf-to-leaf correlation structure and the
  quadratic form `sqrt(w^T R w)` it defines, with a positive semi-definiteness
  test on the declared matrix. `R = I` reduces the form to root-sum-square and
  `R = J` to the linear sum; those two reductions are the algebraic reference
  points for the whole package.
- `aggregate.py`: the rollup. Three combination rules per node - `linear`,
  `rss`, `correlated` - with margin, margin fraction and margin at k sigma at
  every node that declares an allocation, and the exact correlated form
  recorded beside the declared one at every node so the two can be compared.
- `sensitivity.py`: three separate answers to three separate questions - value
  share, variance share under the declared correlations, and the measured
  change in the top-level margin when one leaf is perturbed by a declared
  relative step.
- `checks.py`: ten checks, BR001 to BR010, each naming the objects it is about.
  BR003 and BR004 carry a rounding band so that the sign of a margin that is
  zero to within floating point is never reported as an overrun.
- `parse.py`: a strict YAML reader. Unknown keys rejected rather than ignored,
  duplicate keys in one mapping rejected rather than silently collapsed,
  dangling correlation references rejected at load time, and every rejection
  naming the offending key path. 36 schema rejection paths and 3 input and
  output paths exercised.
- `report.py`: text, markdown and JSON renderings, each repeating the scope
  statement beside the verdict.
- CLI `python -m budgetroll` with `check`, `rollup`, `sensitivity`, `compare`
  and `codes`; exit 0 with no blocking finding, 1 with at least one, 2 when the
  command could not run. 32 real subprocesses measured against that contract.
- Four shipped cases: a clean control that raises nothing, a planted case that
  raises all ten checks, a 3U CubeSat orbit-average power budget that closes on
  predicted values and does not close at two standard deviations, and a CubeSat
  S-band downlink degradation budget whose allocation is the gross link margin
  computed in `validation/validate_link_budget.py`.
- Four runnable examples, each writing a PNG into `screenshots/`.
- Eight validation scripts with their raw output committed beside them, in the
  flat `validation/<script-name>_output.txt` layout.
- 264 tests: unit, input validation, known-answer with the hand arithmetic in
  the test comments, edge cases, Hypothesis property tests over the five
  algebraic identities, pinned regression tests carrying the literal inputs of
  every counterexample found, and CLI subprocess tests that assert on exit
  status.

### Known limitations at this release

See the Limitations section of README.md. The ones that bite first: the package
does no unit conversion and no dimensional analysis, so a budget that mixes kg
and lb is summed without complaint; the additive model is valid only for
quantities that combine additively in the declared unit; a k-sigma figure is a
multiple of a standard deviation and not a confidence level, because no
distribution is assumed anywhere; a 1 sigma value below about 2.22e-162 in the
budget's unit has a square that underflows to zero and is lost from every
root-sum-square; and the three margin policies shipped in the example cases are
declared by the author of those files and are not reproduced from any standard.
