# Changelog

All notable changes to this project are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this
project uses [semantic versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-16

First release. Status: TESTING. Research-grade; not flight-qualified, not
certified, not approved for operational aerospace use.

### Added

- Declarative YAML schema (`spec_version: 1`) for a supervisor, escalation
  ladders and supervised tasks, with strict parsing: unknown keys rejected,
  no invented defaults for any quantity the arithmetic reads, and a key path
  on every rejection.
- Closed-form worst-case detection latency
  `WCDL = (k + 1)(P + J) + Q + Jq + O` and the escalation composition
  `WCRL(r) = WCDL + sum of rung latencies`, with the deepest rung reachable
  inside each declared liveness deadline.
- False-trip rate from the declared jitter statistics,
  `p = 0.5 erfc((k P + (k+1) J) / (2 sigma))` per heartbeat gap, with the
  per-hour rate and the mean time between false trips.
- Seeded discrete-event simulation of heartbeat arrivals and the supervisor
  deadline chain and poll loop, written here rather than on top of a
  simulation framework, used to cross-check both of the above.
- Twelve consistency checks, `WS001` to `WS012`, with a documented precedence
  so that one cause produces one finding.
- Three-valued deadline verdict (`met`, `marginal`, `missed`) with an explicit
  comparison band of `max(tol_ms, 16 ulp of the larger quantity compared)`, so
  that no two-valued boolean is ever produced by a strict comparison of a
  floating-point sum against a threshold. The relative part of the band was
  added after a property-based run **at the declared dependency floor** found
  that an absolute tolerance of 1e-9 ms is narrower than one unit in the last
  place above about 4.5e6 ms; see `validation/VALIDATION.md` section 5.2.
- Command line interface `python -m watchdogspec` with `check`, `latency`,
  `simulate`, `mermaid` and `codes`, and a measured exit-code contract.
- Text, JSON and markdown reports; Mermaid and Agg PNG renderings.
- Three shipped specifications: a consistent minimal one, a planted fixture
  carrying exactly one instance of each check, and a worked example published
  with its two findings in place.
- Validation level 2: eight rerunnable scripts under `validation/` with their
  committed output.

### Known limitations at this release

- The closed form is tight only while the supervisor poll release jitter does
  not exceed the poll period; beyond that it is a strict over-estimate. Found
  by a property-based run, pinned in `tests/test_regression.py`.
- The far tail of the false-trip rate is unvalidated by sampling below about
  `2e-4` per gap and is only as good as the Gaussian independence assumption.
- Both constants in the comparison band, 1e-9 ms and 16 ulp, are chosen rather
  than derived. The measured worst re-association error is 2 ulp.
- Nothing here measures a running system. The whole package checks a
  declaration against itself.
