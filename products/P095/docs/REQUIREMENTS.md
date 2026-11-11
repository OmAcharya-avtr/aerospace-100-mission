# Requirements — boardprobe

Requirements for a Level 3 product of the aerospace 100-product mission. Each
requirement has an identifier, a statement, a verification method and the
artifact that verifies it. A requirement that is NOT met is recorded as not
met, with what is missing.

Status key: **MET** — verified by the named artifact in this repository.
**NOT MET (hardware)** — cannot be verified without an NVIDIA Jetson Orin
Nano, which was not reachable when this was built.

## 1. Provenance

| id | requirement | verification | status |
| --- | --- | --- | --- |
| R-1.1 | Every reading carries its provenance, units, timestamp and source channel in the returned value. | `Reading` fields; `tests/test_readings.py` | MET |
| R-1.2 | Provenance is one of at least `LIVE`, `RECORDED`, `SIMULATED`, totally ordered by entitlement. | `boardprobe.provenance`; `validation/validate_provenance_lattice_output.txt` §1 | MET |
| R-1.3 | Provenance propagates through every derivation as the lattice meet of its inputs. | `attribute`, `combine`, `derive`, `summarise`, `rail_power`; `validation/validate_provenance_lattice_output.txt` §3 | MET |
| R-1.4 | The meet operation is monotone, associative, commutative and idempotent with an identity. | exhaustive enumeration in `validation/validate_provenance_lattice_output.txt` §2; Hypothesis in `tests/test_provenance_properties.py` | MET |
| R-1.5 | A derivation with no inputs is refused rather than taking the lattice identity. | `meet_all`, `attribute`; `validation/validate_provenance_lattice_output.txt` §4 | MET |
| R-1.6 | Provenance cannot be strengthened by any operation in the package. | `meet` has no inverse; backend floor only weakens; `validation/validate_adversarial_label_output.txt` | MET |

## 2. Labelling

| id | requirement | verification | status |
| --- | --- | --- | --- |
| R-2.1 | A result may be labelled `MEASURED` only when its propagated provenance is `LIVE` throughout. | `boardprobe.certify.certify`; `tests/test_certify.py` | MET |
| R-2.2 | Labelling passes through exactly one choke point, with no second path. | `Certificate` is constructible only by `certify`; `validation/validate_adversarial_label_output.txt` route 15 | MET |
| R-2.3 | An attempt to label otherwise raises with an actionable message naming the offending channel. | `LabelViolation` message; `validation/worked_example_output.txt` | MET |
| R-2.4 | No simulated or recorded reading can reach a result labelled measured, by any route the API exposes. | 15 adversarial routes in `validation/validate_adversarial_label_output.txt` | MET |
| R-2.5 | The benchmark harness, the report writer and the CLI all label through the choke point. | `harness.benchmark`, `report.render_*`, `__main__`; `validation/validate_harness_output.txt` §1 | MET |
| R-2.6 | A measurement taken off the declared target cannot be rendered as a bare `MEASURED`. | `Certificate.headline`; `tests/test_certify.py`, `tests/test_integration.py` | MET |

## 3. Hardware abstraction layer

| id | requirement | verification | status |
| --- | --- | --- | --- |
| R-3.1 | One uniform interface over the board's clock, die-temperature, rail-power and performance-counter sources. | `Source.read`; 14 channels across 4 kinds in `validation/validate_backends_output.txt` §1 | MET |
| R-3.2 | Board paths and counter names are declared as data, not embedded in logic. | `boardprobe.orin_nano.ORIN_NANO`; backends dispatch on `Access` alone | MET |
| R-3.3 | Every declared interface cites its documentation source. | the reference table in `validation/validate_backends_output.txt` §1 | MET |
| R-3.4 | Declared paths are marked unverified until read on hardware. | `Channel.verified_on_hardware` is `False` on all 14; asserted in `validation/validate_backends_output.txt` §1 | MET |
| R-3.5 | The live backend reads the declared interfaces on a real board. | — | **NOT MET (hardware)**: structurally complete, never executed against the board. |
| R-3.6 | The live backend fails cleanly and informatively when the board is absent. | 13 of 13 board channels refused with an actionable message; `validation/validate_backends_output.txt` §4 | MET |
| R-3.7 | The recorded backend replays a committed trace and stamps `RECORDED`, weakened by the document's own content provenance. | `validation/validate_backends_output.txt` §3 | MET |
| R-3.8 | The simulated backend is deterministic from a fixed seed. | 14 of 14 channels bit-identical; `validation/validate_backends_output.txt` §2 | MET |
| R-3.9 | A trace document cannot declare `live` content. | `Trace.from_mapping`; adversarial route 6 | MET |

## 4. Dry-run mode

| id | requirement | verification | status |
| --- | --- | --- | --- |
| R-4.1 | A dry run exercises every code path that does not require a device. | 16 named code paths; `validation/validate_dry_run_output.txt` §1 | MET |
| R-4.2 | A dry run performs no I/O against a device. | 0 accesses against the real filesystem root; §1 | MET |
| R-4.3 | A dry run is asserted side-effect free, by measurement and not by promise. | before-and-after tree snapshot, 0 files added; §1 | MET |
| R-4.4 | The device-access ledger is the mechanism, and the dry run reports its contents. | `boardprobe.sysio.LEDGER`, `ledger_window`; §2 | MET |
| R-4.5 | The dry run's own check agrees with itself when run twice from the same tree. | §5 | MET |

## 5. Benchmark harness

| id | requirement | verification | status |
| --- | --- | --- | --- |
| R-5.1 | Runs a user-supplied callable under sampling and produces timing and resource figures. | `harness.benchmark`; `validation/validate_harness_output.txt` §3 | MET |
| R-5.2 | Every figure carries propagated provenance. | `Statistic.attribution`; §1 | MET |
| R-5.3 | Refuses to call its output measured on anything but live sources. | §1, 6 of 6 backend combinations | MET |
| R-5.4 | The harness's own overhead is characterised and reported. | `characterise_overhead`; §4 | MET |
| R-5.5 | The overhead characterisation is checked against independently measured bounds. | the sandwich in `validation/validate_independent_bound_output.txt` §3 | MET |
| R-5.6 | Timing figures measured on the declared target. | — | **NOT MET (hardware)**: every figure is a build-host figure, labelled as such. |
| R-5.7 | A channel the environment backend cannot serve is reported, never silently dropped. | `read_available` returns refusals; `BenchmarkResult.refusals`; `tests/test_harness.py` | MET |

## 6. Procedures

| id | requirement | verification | status |
| --- | --- | --- | --- |
| R-6.1 | A deployment procedure someone can execute. | `docs/DEPLOYMENT.md` | MET |
| R-6.2 | A recovery procedure covering board unreachable mid-run, thermal throttle, a partially written trace, a stale lock and unestablishable provenance. | `docs/RECOVERY.md`, five numbered procedures | MET |
| R-6.3 | A run whose provenance cannot be established aborts rather than downgrading silently. | `ProvenanceError` from `meet_all` and `attribute`; `docs/RECOVERY.md` §5 | MET |
| R-6.4 | Deployment and recovery procedures executed on the board. | — | **NOT MET (hardware)**: both are written and neither has been executed against a Jetson Orin Nano. |

## 7. Interfaces

| id | requirement | verification | status |
| --- | --- | --- | --- |
| R-7.1 | A CLI via `python -m boardprobe`, argparse, `--help` exiting 0 in a clean subprocess. | `tests/test_cli.py`; `validation/validate_cli_output.txt` §1 | MET |
| R-7.2 | Subcommands for listing channels, taking a sample, running the benchmark, replaying a trace and printing a provenance report. | `channels`, `sample`, `benchmark`, `replay`, `provenance`, plus `dry-run` and `attest` | MET |
| R-7.3 | A bad argument exits non-zero with a message and no traceback. | `validation/validate_cli_output.txt` §2 | MET |
| R-7.4 | A committed trace fixture small enough to track. | 36 KB and 12 KB; `tests/test_fixtures.py`; `git check-ignore -v` in `validation/validate_cli_output.txt` §3b | MET |

## 8. Engineering process

| id | requirement | verification | status |
| --- | --- | --- | --- |
| R-8.1 | No figure in the repository presented as a hardware measurement. | every `MEASURED` certificate reads `MEASURED OFF-TARGET`; `tests/test_integration.py` | MET |
| R-8.2 | Validation level labelled `Level 3, hardware-pending` everywhere. | `README.md`, `validation/VALIDATION.md`, `boardprobe.VALIDATION_LEVEL`, the badge | MET |
| R-8.3 | Every validation number from a script run in the build session, with raw output committed beside it. | 11 `validation/*_output.txt` files | MET |
| R-8.4 | Every validation and example script exits 0 from four working directories. | 40 validation runs and 8 example runs, all 0; `validation/probe_working_directories_output.txt` | MET |
| R-8.5 | No `pytest.approx(rel=)` call depends on the hidden 1e-12 absolute default. | 0 of 291; `validation/validate_approx_audit_output.txt` | MET |
| R-8.6 | At least one bound check from mathematics the implementation does not use. | three, in `validation/validate_independent_bound_output.txt` | MET |
| R-8.7 | Negative controls anchored on true values, each verified to fire or stay silent. | 21 controls; `validation/validate_negative_controls_output.txt` | MET |
| R-8.8 | Test count taken from junit XML and cross-checked against `testcase` elements. | 291 declared, 291 counted; `validation/validate_approx_audit_output.txt` | MET |
| R-8.9 | Running a validation script adds no cache directory or bytecode file. | `validation/validate_cli_output.txt` §3a | MET |
| R-8.10 | Git tracks no cache directory, bytecode file or build artifact. | `git ls-files` | **MET**, measured after committing: 78 tracked files examined, 0 offenders, in `validation/validate_cli_output.txt` §3b. Not measurable by the build agent itself, whose tree is uncommitted. |
| R-8.11 | No committed file contains an absolute host path. | 0 offenders in the files scanned; `validation/validate_cli_output.txt` §4 | MET |
| R-8.12 | Ruff-clean under both installed ruff versions. | 0.16.9 and 0.17.4, both clean over `src/ tests/ examples/ validation/` | MET |

## What Level 4 requires

Four things, all of them needing the board:

1. R-3.5 — the live backend executed against the Jetson Orin Nano, with every
   channel's locator, raw unit and scale confirmed and
   `verified_on_hardware` flipped to `True` channel by channel.
2. R-5.6 — timing and resource figures measured on the board, with the
   harness overhead re-characterised there, because the overhead is a
   property of the host and the figures in this repository are x86_64.
3. R-6.4 — `docs/DEPLOYMENT.md` and `docs/RECOVERY.md` executed end to end on
   the board, with the steps corrected where they turn out wrong.
4. A `PlatformAttestation` whose `matches_target` is `True` from a real
   `proc/device-tree/model` read, so that a certificate reads `MEASURED` with
   the board named rather than `MEASURED OFF-TARGET`.

Until all four are done, the product is `Level 3, hardware-pending`.
