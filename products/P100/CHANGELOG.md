# Changelog

All notable changes to this product are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the versioning
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.1.0 — 2026-11-16

First release. Status: TESTING.

### Added

- `seedaudit.power`: the closed-form detection power for intermittent
  non-determinism in both sampling designs, its two inversions, and three
  independently derived brackets (first-order union, two-term
  inclusion-exclusion, and the exponential inequality on `log(1-p)`). The
  escaped-rate upper bound `1 - alpha**(1/n)` is what a clean audit reports
  instead of zero.
- `seedaudit.numeric`: numeric-claim extraction, skeleton alignment and
  comparison of a committed text against a re-measured one, with both
  tolerance bounds explicit and defaulting to zero.
- `seedaudit.seeds`: syntax-tree discovery and rewriting of the seed literals
  a script declares, with every unresolvable seed position reported rather
  than silently skipped.
- `seedaudit.static`: the static screen for the self-polluting-check class,
  with its own coverage reported as a count of path uses it cannot reason
  about.
- `seedaudit.dynamic`: pristine-copy execution and the two-arrangement
  classifier that separates intrinsic non-determinism from self-pollution.
- `seedaudit.corpus`: Baseline 3, the injected corpus of known rate, with
  disjoint-block detection-power measurement.
- `seedaudit.audit`: whole-repository auditing with coverage reported as a
  number.
- `python -m seedaudit` with the subcommands `power`, `static`,
  `arrangements`, `corpus` and `audit`.

### Known limitations recorded at release

- The candidate rule requires a seed literal in the script itself, so a script
  whose seed is declared in an imported module is not a candidate. The size of
  that blind spot is measured in `validation/validate_self_audit_output.txt`
  and published in README.md.
- The suite has not been run at both ends of every declared dependency range.
- The repeat axis of the repository audit is sampled once per arrangement, so
  its escaped-rate bound is weak and is stated with every result.
