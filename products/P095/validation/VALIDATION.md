# Validation — boardprobe

**Validation level: Level 3, hardware-pending.**

The registry records `validation_level: 4` for this product's slot. That is the
slot's TARGET. This product is **Level 3, hardware-pending**, and the reason is
one sentence: **Level 4 requires measured timing and resource figures from an
NVIDIA Jetson Orin Nano, the board was not reachable from the environment this
package was built in, and no figure in this repository was measured on the
board.**

What that means concretely, stated before any number below:

- Every sysfs path, counter name, raw unit and scale in
  `src/boardprobe/orin_nano.py` is declared from the documentation cited
  beside it and is **UNVERIFIED ON HARDWARE**. Not one of those files was
  opened. `verified_on_hardware` is `False` on all 14 channels and a check
  asserts that it is.
- The live backend's reading path is structurally complete and has never been
  executed against the declared target. What IS measured here is its
  **refusal**: 13 of 13 board channels refuse with an actionable message on a
  host without the board.
- Every timing and resource figure is a **container number**. The container:
  Python 3.13.18, numpy 2.5.3, 2 cores, 7.8 GiB, **x86_64 Linux**. The
  declared target is **aarch64**, which is itself a limitation: even the
  instruction counts a working perf adapter would report on the target are not
  comparable with anything here.
- Every `MEASURED` certificate this repository can produce renders as
  `MEASURED OFF-TARGET`, naming both model strings, because the platform
  attestation does not match the declared board. A bare `MEASURED` is
  unreachable here by construction.

`docs/REQUIREMENTS.md` lists the four requirements that are NOT MET without the
board, and they are the four things Level 4 needs.

## How to reproduce everything below

```bash
python -m pytest tests/ -q -p no:cacheprovider --junit-xml=junit.xml
python validation/validate_provenance_lattice.py
python validation/validate_adversarial_label.py
python validation/validate_independent_bound.py
python validation/validate_backends.py
python validation/validate_harness.py
python validation/validate_dry_run.py
python validation/validate_negative_controls.py
python validation/validate_approx_audit.py
python validation/worked_example.py
python validation/validate_cli.py
python validation/probe_working_directories.py
python examples/provenance_lattice.py
python examples/backend_comparison.py
python examples/harness_overhead.py
python examples/label_matrix.py
```

Each script writes its raw output beside itself as
`validation/<script-name>_output.txt`. Every number in this file and in
`README.md` comes from one of those files, from a script run in the build
session, and the file is named beside the number.

## Summary

| quantity | value |
| --- | --- |
| tests, from the junit `testsuite` element | **291** |
| independent count of `testcase` elements | **291** |
| failures / errors / skipped | **0 / 0 / 0** |
| named checks across the 11 validation scripts | **122** |
| failed checks | **0** |
| adversarial routes attempted / refused | **15 / 15** |
| routes that succeeded during the build and were closed by a design change | **2** |
| negative controls declared | **21** |
| must-fire controls that fired | **13 of 13** |
| must-NOT-fire controls that stayed silent | **8 of 8** |
| `pytest.approx` assertions depending on the hidden 1e-12 default | **0 of 291** |
| validation script runs from four working directories | **40, all exit 0** |
| example runs from two working directories | **8, all exit 0** |
| channels declared / verified on hardware | **14 / 0** |
| figures in this repository measured on the declared target | **0** |

## 1. The provenance lattice — `validate_provenance_lattice_output.txt`

Reference for the axioms: B. A. Davey and H. A. Priestley, *Introduction to
Lattices and Order*, 2nd edition, Cambridge University Press 2002, chapter 2,
definitions 2.1–2.8.

| check | reference | result | tolerance |
| --- | --- | --- | --- |
| meet is commutative | all 9 pairs enumerated | 9 of 9 | exact (identity on enum members) |
| meet is associative | all 27 triples enumerated | 27 of 27 | exact |
| meet is idempotent | all 3 elements | 3 of 3 | exact |
| `LIVE` is the identity of meet | both sides, 3 elements | 3 of 3 | exact |
| `SIMULATED` absorbs | 3 elements | 3 of 3 | exact |
| meet is monotone in each argument | every triple with `a ≤ b` | 0 violations | exact |
| meet is the greatest lower bound | all 9 pairs, lower bound and greatest among them | 9 of 9 | exact |
| the absorption laws | all 9 pairs, both laws | 9 of 9 | exact |
| the lattice is distributive | all 27 triples, both laws | 27 of 27 | exact |
| `meet_all` is the weakest element present, order-independent, duplication-invariant | **1092 assignments** of length 1 to 6 | 0 disagreements | exact |
| the empty fold is REFUSED, not taken as the identity | `ProvenanceError` | raised | — |
| exactly one lattice element maps to `MEASURED` | enumeration over the chain | `['live']` | — |

The empty-fold row is the one worth reading twice. `LIVE` is the identity of
`meet`, so the ordinary monoid fold over an empty collection returns the
STRONGEST element in the lattice, for a derivation with no inputs at all.
`meet_all([])` raises instead, and so does `attribute([])` and `summarise([])`.

## 2. The adversarial check — `validate_adversarial_label_output.txt`

**The check the specification asks for: no simulated reading can reach a result
labelled measured.** Fifteen routes, grouped by what each attacks.

| # | route | refused by |
| --- | --- | --- |
| 1 | label a wholly simulated reading set `MEASURED`, directly | `LabelViolation` |
| 2 | aggregate first, then label the statistic | `LabelViolation` |
| 3 | dilute: 999 live readings and one simulated one | `LabelViolation` |
| 4 | derive a second-order figure from a live and a simulated statistic | `LabelViolation` |
| 5 | replay a trace and label the replay | `LabelViolation` |
| 6 | edit a trace document to declare `live` content | `TraceError` |
| 7 | serialise a simulated reading, edit it to `live`, deserialise | `LabelViolation` |
| 8 | rewrite provenance with `object.__setattr__`, past the frozen guard | `TamperError` (seal) |
| 9 | a `Reading` subclass overriding the seal check and the token | `TamperError` (exact type) |
| 10 | **`dataclasses.replace(reading, provenance=LIVE)`** | `TypeError` — **this route SUCCEEDED** |
| 11 | a `Source` subclass overriding `read()` to stamp `LIVE` | `TypeError` at class creation |
| 12 | subclass `LiveSource` and serve it trace values | `TypeError` — the backends are sealed |
| 13 | **rebind `SimulatedSource.backend_provenance` to `LIVE`** | `RuntimeError` — **this route SUCCEEDED** |
| 14 | exploit the lattice identity: meet of an empty set is `LIVE` | `ProvenanceError` |
| 15 | construct a `Certificate` directly, bypassing `certify()` | `TypeError` |

**Routes attempted: 15. Routes refused: 15. Routes that succeeded: 0.**

Three positive controls accompany them, because a check that only ever
observes refusals cannot distinguish a working invariant from a function that
raises unconditionally. The `SYNTHETIC`, `REPLAYED` and `MEASURED` labels are
each granted on inputs that admit them, and the `MEASURED` one is granted on a
live clock reading and renders as `MEASURED OFF-TARGET`.

### The two routes that succeeded, and what was changed

Both were found while writing this check, on 2026-11-11, and in both cases the
DESIGN was changed rather than the check.

**Route 10, `dataclasses.replace`.** The exact attack:

```python
laundered = dataclasses.replace(simulated_reading, provenance=Provenance.LIVE)
certify("figure", attribute([laundered]), Label.MEASURED, platform)
```

`dataclasses.replace` is public API on every dataclass. It re-ran the
`Reading`'s `__post_init__`, which **recomputed the tamper seal**, so the
laundered reading was validly sealed, carried the simulated reading's own
channel and value, claimed `LIVE`, and the choke point granted `MEASURED`. One
line, no private state touched. The seal did not help, because the seal was
recomputed by the very mechanism being abused.

The fix: a `Reading`'s first field is now a `dataclasses.InitVar` holding a
module-private minting token, so `dataclasses.replace` and `copy.replace` both
raise `TypeError: InitVar 'mint' must be specified with replace()`, and the
single minting entry point is `boardprobe.readings.mint()`, whose only
in-package callers are `Source.read()` and `from_mapping()`. Pinned as a
regression test in `tests/test_readings.py`.

**Route 13, rebinding the class attribute.** Provenance was read from
`type(self).backend_provenance`, which is an ordinary class attribute and can
be assigned to. The fix: the three backends' declared provenances are captured
in a module-private registry when `boardprobe.sources` finishes importing, and
the `provenance` property compares the class attribute against the registry on
every read.

### What is NOT closed, named rather than omitted

None of these is a route from an existing simulated reading to a measured
label, which is what the check is about.

1. `boardprobe.readings.mint()` lets its caller choose a provenance. A caller
   who passes `LIVE` for a generated number has made a false statement at the
   source. No type system can prevent that; a hardware abstraction layer is a
   program that believes the kernel. What changed is that the false statement
   is now made in one named function, is attributable to a `source_id` that
   travels with the reading, and cannot be made by accident or by a refactor.
2. A `PlatformAttestation` can be constructed by hand with
   `matches_target=True`. That is a false claim about WHICH machine, not about
   provenance, and every certificate records the attestation's own `source`
   string so the claim is legible in the output.
3. Code running in the same process can reach into module-private state
   (`boardprobe.readings._KEY`, `boardprobe.sources._SEALED` and `_DECLARED`)
   and defeat all of the above. This is true of every Python program.

## 3. Independent bound checks — `validate_independent_bound_output.txt`

Three checks whose mathematics the implementation does not use.
`boardprobe.provenance` computes `meet` as `min` over an integer rank;
everything here avoids that encoding.

| check | the independent mathematics | result | tolerance |
| --- | --- | --- | --- |
| the chain re-derived from its Hasse diagram | **Warshall's transitive closure** (Warshall 1962, *JACM* 9(1), 11–12) over a boolean reachability matrix built from the two covering edges; the meet defined order-theoretically with no ranks | **1092 assignments, 0 disagreements** | exact |
| the closure is reflexive, transitive, antisymmetric and total | the same matrix | 3 + 27 + 9 + 9 cases, all pass | exact |
| the propagation counted by **inclusion and exclusion** | `#{meet = simulated} = 3ⁿ − 2ⁿ`, `#{meet = recorded} = 2ⁿ − 1`, `#{meet = live} = 1`, summing to `3ⁿ` | **9840 assignments enumerated for n = 1..8, 0 mismatched rows** | exact integer arithmetic |
| the harness in-bracket overhead inside an independently measured sandwich | `cost(one sealed Reading) ≤ in-bracket overhead ≤ cost(one clock read pair)`, both bounds measured by a DIFFERENT arrangement (one bracket around a block of N operations, not N brackets differenced) | median **2.740 ≤ 4.348 ≤ 7.685 µs**; lower bound **7/7** trials, upper bound **7/7** trials | zero slack; the inequalities are structural |

Why the first two are genuinely independent: the implementation's claim is that
an integer rank encoding is an order embedding of the chain. Warshall's closure
tests exactly that claim, from the covering relation alone, and would disagree
if the ranks were out of order or if `meet` were not the greatest lower bound.
The inclusion–exclusion identity tests the dilution rule from the other side:
get the direction of propagation wrong and `3ⁿ − 2ⁿ` disagrees immediately, at
n = 2, before any enumeration gets interesting.

The timing sandwich is a weaker check and is reported as such: it is wall-clock
on a shared two-core host, so the HIT RATE over 7 trials is reported alongside
the median rather than a single trial being asserted. **Both bounds held on
7 of 7 trials.**

## 4. The backends — `validate_backends_output.txt`

| check | reference | result | tolerance |
| --- | --- | --- | --- |
| no channel claims to be verified on hardware | the shipped table | **0 of 14** | — |
| every channel cites a reference and a relative locator | the table | 14 of 14 | — |
| all four source classes the specification names are declared | clock, die-temperature, rail-power, performance-counter | 4 of 4 | — |
| the simulated backend reproduces bit for bit at one seed | two independent instances at seed 770095 | **14 of 14 channels identical** | bit-exact, and correct here: the claim is reproducibility of a pseudo-random stream, not the value of a physical quantity |
| reading a subset gives the same values as reading every channel | one generator per channel | 2 of 2 | bit-exact |
| the idle fixture replays as `simulated` | `meet(recorded, simulated)` | `simulated` | — |
| the replay-path fixture replays as `recorded` | `meet(recorded, recorded)` | `recorded` | — |
| every first sample of each trace replays with its own value and unit | the committed document | 10 of 10 and 5 of 5 | bit-exact on the document's own floats |
| every board channel is refused on a host without the board | the real filesystem root | **13 of 13 refused, 0 served** | — |
| every refusal says what a reader should check | the message text | 13 of 13 | — |
| no refusal message quotes an absolute host path | needles assembled from `os.sep` | 0 offenders | — |
| the monotonic clock is served and stamped live | `clock_gettime(2)`, `CLOCK_MONOTONIC_RAW` | served, `live`, unit `s` | — |
| the perf adapter fails with the C `errno` and what to check | `perf_event_open(2)` | `errno 2 (No such file or directory)`: no PMU driver bound in this virtual machine | — |
| `sizeof(struct perf_event_attr)` as declared | `linux/perf_event.h` through `config3` | **136 bytes** | exact |
| no sealed backend can be subclassed | `__init_subclass__` | 3 of 3 refused at class creation | — |

The perf failure is worth its own sentence. The adapter was written from
`perf_event_open(2)` and exercised only on x86_64, where the syscall number is
298 and the call returns `-1 ENOENT` because the virtual machine exposes no
PMU. On an aarch64 Jetson the syscall number is 241. **Both numbers are
declared in `src/boardprobe/orin_nano.py` and the aarch64 one has never been
executed.**

## 5. The harness — `validate_harness_output.txt`

**Every figure in this section is a container number.** Container: Python
3.13.18, 2 cores, 7.8 GiB, x86_64 Linux. Target: aarch64.

| check | result |
| --- | --- |
| no backend combination available on this host yields a hardware measurement | **6 of 6 combinations: NO** |
| a live clock with a simulated environment | timing `MEASURED`, overall **`SYNTHETIC`** |
| a live clock with a recorded environment | overall **`REPLAYED`** |
| a simulated clock can never yield a `MEASURED` timing label | 3 of 3 combinations |
| a live environment backend on a board-less host | `HarnessError`, pointing at `docs/RECOVERY.md` |
| the reported durations against the clock's resolution at this uptime | **1.721e+08 times the floor** |

Per-repeat cost over 400 repeats, microseconds, on the build host:

| quantity | µs |
| --- | --- |
| null payload, sampling on | 4.376 |
| null payload, sampling off | 0.038 |
| overhead inside the bracket | **4.337** |
| one clock read (measured independently) | 4.035 |
| one sealed `Reading` | 2.652 |
| reference payload, sampling off | 73.040 |
| reference payload, sampling on | 79.678 |

A 200-repeat timing-only benchmark of `reference_payload(2000)` on the build
host: mean **80.522 µs**, sd **7.431 µs**, median **78.275 µs**, clock
resolution at this uptime **4.547e-13 s**.

**The perturbation ratio is noisy and that is reported rather than hidden.**
Single trials of it have come back BELOW 1.0 on this host — not a negative
overhead, but a statement that the perturbation is at the level of the
measurement noise for a 70 µs payload. `examples/harness_overhead.py`
measures the whole curve over **five trials per payload size**, reporting the
median and the min-to-max band. Median ratio against the median payload cost
without sampling:

| payload [µs] | 2.28 | 4.51 | 9.29 | 19.75 | 38.41 | 67.38 | 141.42 | 255.46 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| median of 5 | 3.330 | 2.153 | 1.610 | 1.252 | 1.165 | 1.123 | 1.056 | 1.037 |
| min | 3.001 | 1.736 | 1.549 | 1.211 | 0.864 | 1.054 | 0.957 | 0.922 |
| max | 4.044 | 2.769 | 2.304 | 1.360 | 1.251 | 1.152 | 1.083 | 1.082 |

**Below a few microseconds this harness is not an instrument**, and above about
140 µs on this host the perturbation is comparable with the run-to-run noise --
the min column at 38 µs is 0.864 and at 255 µs is 0.922, and the max at 9 µs is
2.304 against a median of 1.610. Those are a shared two-core host descheduling
the measuring process and are left in the table rather than trimmed. That is
the honest statement of this harness's resolution.

## 6. Dry-run mode — `validate_dry_run_output.txt`

| check | result |
| --- | --- |
| named code paths exercised | **16** |
| device accesses against the real filesystem root | **0** |
| attempts against a root asserted not to exist (the live refusal leg) | **13**, reported in their own count |
| files added under the watched tree | **0** |
| every code path name distinct, so the count is a count | 16 of 16 |
| every attempted locator relative | 13 of 13 |
| the choke point exercised in both directions | 2 granted, 3 refused |
| the second run agrees with the first on every reported quantity | paths, accesses, attempts, files added and file count all equal |

The separation of the two access counts is deliberate and is the reason the
dry run can make its claim honestly. Folding the live-refusal attempts into
the main ledger would make the main count non-zero for accesses that provably
cannot reach a device; omitting them would be a self-reporting scanner. Both
numbers are printed, and the absent root is asserted not to exist before the
leg runs.

Running the scanner twice and comparing is the fifth-instance precaution:
**a check whose own execution can create the condition it tests is not a
check.**

## 7. Negative controls — `validate_negative_controls_output.txt`

**21 controls declared. 13 must fire; 13 fired. 8 must NOT fire; 8 stayed
silent.**

Every control is anchored on a value known independently of the code under
test: the provenance a backend DECLARES as a class constant; a raw sysfs
integer this script writes itself, so the correct value is known by hand
(48123 millidegrees Celsius is 48.123 °C; 4870 mV × 4860 mA is 23.6682 W); or
a declared plausibility window.

**The two plausibility windows are declared in the script and are NOT quoted
from any datasheet**, because no NVIDIA document was read against hardware for
this build and a datasheet citation would be unsupported. Die temperature:
−40 to 125 °C, on the reasoning that a silicon die reading outside that is a
scaling or wiring error and not a temperature. Rail power: 0 to 60 W,
spanning every published Orin Nano power mode with large margin. Both are
deliberately wider than any operating range: they catch the factor-of-1000
class of error and nothing subtler, which is what they are for.

### Must fire (13 of 13 fired)

| control | why it must fire |
| --- | --- |
| a wholly simulated reading set labelled `MEASURED` | true provenance is simulated, declared by the backend class |
| a replayed reading set labelled `MEASURED` | true provenance is recorded |
| five live readings and one simulated one labelled `MEASURED` | the meet of the true provenances is simulated |
| a simulated reading set labelled `REPLAYED` | simulated admits `SYNTHETIC` only |
| a live reading set labelled `SYNTHETIC` | the label must follow the provenance in BOTH directions |
| provenance rewritten with `object.__setattr__` | the seal is recomputed and compared |
| `dataclasses.replace(reading, provenance=LIVE)` | the minting token; this route SUCCEEDED before the design was changed |
| an aggregate over an empty reading set | the lattice identity is `live` |
| a trace document edited to declare `live` content | a replay is not a live reading |
| the millidegree scale dropped from the thermal channel | **48123.0 °C** against a window of (−40, 125); ratio to the true value **1000.0** |
| millivolts labelled as volts in the rail-power derivation | **23668.2 W** against a window of (0, 60); ratio **1000.0** |
| a trace recording kelvin where the table declares °C | a unit mismatch is a wrong number |
| a clock channel whose values decrease during a benchmark | a monotonic clock that decreases is not monotonic |

### Must NOT fire (8 of 8 stayed silent)

a wholly live set labelled `MEASURED`; a replayed set labelled `REPLAYED`; a
simulated set labelled `SYNTHETIC`; an untouched reading passed to
`attribute()`; the correctly scaled die temperature (**48.123 °C**, inside the
window); the correctly scaled rail power (**23.6682 W**, inside the window); a
clock channel whose values increase; and a trace document declaring `recorded`
content.

No control's criterion was weakened and no seed was changed to make a control
behave.

## 8. The `pytest.approx` audit — `validate_approx_audit_output.txt`

`pytest.approx(expected, rel=r)` passes if EITHER bound is met and the absolute
default is `1e-12`, so a bare `rel=` on a small quantity silently stops testing
what it says.

| part | question | result |
| --- | --- | --- |
| static scan of the test sources | how many `approx` calls carry `rel=` with no explicit `abs=`? | **33 calls carry `rel=`; 0 carry it without an explicit `abs=`** |
| the suite with a bare `rel=` made to raise | how many assertions rely on the default? | **0 of 291** |
| the suite with every `abs` forced to `0.0` | for how many assertions is the ABSOLUTE bound doing the work? | **0 of 291** |

The third result is the strongest of the three: every relative bound in the
suite holds on its own with the absolute bound removed entirely. The `abs=0.0`
arguments throughout are written to say that the relative bound is the one
meant, not to widen anything.

Counts in all three parts come from the junit `testsuite` element and are
cross-checked against a count of `testcase` elements: **291 declared, 291
counted.** A suite that collects zero tests prints success and exits 0, which
is how an earlier product of this mission shipped with a recorded 251 tests
and a config that collected none.

## 9. The command line and the tree — `validate_cli_output.txt`

| check | result |
| --- | --- |
| every subcommand, run as a user runs it, with the README's fragments present | **10 of 10 commands exit 0** |
| an unknown backend exits 2 and lists the three that exist | exit 2 |
| a live environment backend on a board-less host exits 1 with the recovery pointer and no traceback | exit 1 |
| running this script adds no cache directory and no bytecode file | **0 added** |
| git tracks no cache directory, bytecode file or build artifact | **STRUCTURALLY UNMEASURABLE BY A BUILD AGENT** — see below |
| neither shipped fixture is excluded by `.gitignore` | 2 of 2 not ignored |
| no text file in the tree contains an absolute host path | **0 offenders** |

**The tracked-file check cannot be measured here, and is reported as
unmeasurable rather than passed on an empty set.** This product's working tree
is uncommitted while it is being built: the mission's coordinator holds every
commit under ADR-010, so `git ls-files` inside this directory returns nothing
and a check over nothing would report a vacuous PASS — which is worse than
reporting that it could not be measured, because it reads as evidence. What is
measured instead is `git check-ignore -v --no-index` against every shipped
fixture, which is the P060 defect class: a `.gitignore` pattern with no leading
slash matches at any depth and silently excluded a required committed fixture.
Both fixtures are **not ignored**. The tracked-file claim is re-measurable, and
must be re-measured, by whoever commits this tree; release-gate check 9
enforces it independently over the whole repository.

## 10. Working directories — `probe_working_directories_output.txt`

Three products in one batch of this mission passed every check in their own
build and then failed the release gate, which runs each validation script with
its working directory set to the script's own directory rather than to the
product root.

| scripts | directories | runs | non-zero exits |
| --- | --- | --- | --- |
| the 10 validation scripts | `/`, the system temporary directory, the product root, `validation/` (the gate's cwd) | **40** | **0** |
| the 4 examples | the product root, `examples/` (the gate's cwd for examples) | **8** | **0** |

The three underscore-prefixed support modules (`validation/_bootstrap.py`,
`validation/_approx_audit_plugin.py`, `examples/_common.py`) were additionally
executed as scripts from their own directories, because the release gate runs
every `*.py` in those directories and not only the ones meant to be run. All
three exit 0.

## 11. What this product CANNOT claim

Listed as its own section because a validation document whose weaknesses are
only reachable through its strengths is a sales document.

1. **No figure here was measured on an NVIDIA Jetson Orin Nano.** Not one.
2. **Every declared sysfs path, counter name, raw unit and scale is
   unverified.** They come from the kernel ABI documents, manual pages and the
   Jetson Linux Developer Guide cited in `src/boardprobe/orin_nano.py`. The
   thermal-zone INDEX for a given zone name is not fixed by any document and
   is the single most likely line in the table to be wrong on a particular
   L4T release.
3. **The live reading path has never run against the board.** Its refusal path
   has, 13 channels of 13.
4. **The `perf_event_open` adapter's aarch64 syscall number has never been
   executed.** The x86_64 path was exercised only as a clean failure.
5. **`docs/DEPLOYMENT.md` and `docs/RECOVERY.md` have never been executed on
   the board.** Steps 1 to 3 of deployment were executed on the build host;
   steps 4 onward are written from documentation.
6. **The harness overhead figures are x86_64 figures and will not transfer.**
   The overhead must be re-characterised on the board, and a payload within a
   few times the overhead is being perturbed by the instrument.
7. **The dependency ranges were measured at one version each, except `ruff`.**
   `numpy>=2.5,<2.6` was measured at 2.5.3; numpy 2.5.4 exists on PyPI, is
   admitted by that ceiling and was NOT tested, because installing a second
   numpy would have required running `pip` against a shared pre-pinned
   interpreter this product is forbidden to modify. `ruff` is the exception
   and is measured at both ends, 0.16.9 and 0.17.4.
8. **The tracked-file check is unmeasurable by a build agent** and is reported
   as such rather than passed.
9. **The tamper seal is not a security boundary.** It detects mutation,
   mislabelling and forged documents. Code in the same process can import
   module-private state and forge it, and no design in Python can prevent
   that.
