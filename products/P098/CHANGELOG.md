# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-11-19

First release. Status `TESTING`.

### Added

- `evidencestale.graph` — dependency graph with closure and impact
  reachability, cycle reporting, and a depth guard that raises on malformed
  input. Cycles are handled rather than assumed away.
- `evidencestale.replay` — replay of a change history, with the value-based
  definition of true staleness and the by-construction ground truth.
- `evidencestale.baselines` — baseline 1 (timestamp comparison, the thing being
  refuted), baseline 1b (its strongest honest variant over direct
  dependencies), baseline 2 (SHA-256 comparison over the dependency closure)
  and baseline 3 (the ground truth).
- `evidencestale.bounds` — an independently derived bracket on the stale
  fraction from the Moore bound and a union bound, consuming only counts.
- `evidencestale.bootstrap` — cluster and item bootstrap intervals, with the
  resampling unit required rather than defaulted.
- `evidencestale.learned` — random-forest staleness-risk model with a
  confidence output and a per-tree vote spread.
- `evidencestale.benchmark` and `evidencestale.evaluation` — held-out artifact
  and held-out time splits, scored separately.
- `data/generate_corpus.py` and two committed corpora, one layered and acyclic
  and one deliberately cyclic, with a digest manifest.
- Seven validation scripts and their committed raw output, four examples and
  their figures, 213 tests.

### Fixed during the build, each pinned as a named regression test

- An off-by-one in the closure depth guard: a closure whose fixed point needs
  `d` levels requires `d + 1` breadth-first expansions, so the guard fired on
  well-formed input at `depth_limit == d`.
- The bracket's lower bounds were evaluated against a UNION of the per-item
  change sets, which is valid for the upper bound and invalid for the lower
  bounds. Measured consequence at the time: a lower bound above the measured
  fraction at 6 of 10 evaluation revisions.
- A docstring claimed the pairwise digest-collision bound underflows to zero at
  corpus scale. It does not; it is of order 1e-73 and representable.
