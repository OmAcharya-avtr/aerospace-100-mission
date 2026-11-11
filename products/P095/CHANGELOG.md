# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-11-11

First release. Status: TESTING. **Validation level: Level 3,
hardware-pending.** Research grade.

### Added

- `boardprobe.provenance`: the lattice. `SIMULATED < RECORDED < LIVE` as a
  chain, with `meet` as the greatest lower bound and `join` as its dual.
  `meet` is the propagation rule for every derived figure in the package. The
  monoid identity is `LIVE`, which makes the empty fold a trap — a derivation
  with no inputs would claim the strongest provenance in the lattice — so
  `meet_all` **raises** on an empty collection instead of returning it.
  Reference for the axioms: Davey and Priestley, *Introduction to Lattices and
  Order*, 2nd edition, Cambridge University Press 2002, chapter 2.
- `boardprobe.readings`: the `Reading` value object, carrying value, unit,
  integer-nanosecond timestamp, source channel, backend identity and
  provenance; a keyed BLAKE2b tamper seal over its own fields; `attribute` and
  `combine`, which are the only ways an `Attribution` is built and are where
  the meet is taken. A `Reading` cannot be constructed: its first field is a
  `dataclasses.InitVar` holding a module-private minting token, and
  `mint()` is the single entry point.
- `boardprobe.certify`: the choke point. `certify(subject, attribution, label,
  platform)` is the only way a figure in this package acquires a label, and
  `Certificate` refuses construction without a private token. The label is
  DERIVED from the provenance by `label_for`, a total function on the lattice
  with exactly one element mapping to `MEASURED`; the `label` argument exists
  so a caller can state what they believe, and a disagreement raises
  `LabelViolation` naming the offending channels and their sources.
- `boardprobe.platform_attest`: which machine, as a fact independent of
  provenance. `MEASURED` says a device responded; it does not say which
  device, and a `MEASURED` certificate earned off the declared target renders
  as `MEASURED OFF-TARGET` with both model strings. A bare `MEASURED` is
  unreachable off-target by construction.
- `boardprobe.channels` and `boardprobe.orin_nano`: 14 channels across the
  four source classes the specification names, declared as DATA with their
  paths, raw units, scales and the documentation reference behind each.
  Backends dispatch on `Access` alone.
- `boardprobe.sources`: three sealed backends behind one interface.
  `Source.read` stamps provenance from the backend's class and is final —
  `__init_subclass__` refuses a subclass that defines `read`, `read_all`,
  `provenance` or `_stamp_ns`, and the three concrete backends cannot be
  subclassed at all. Provenance is `meet(class constant, instance floor)`, so
  it can only ever weaken.
- `boardprobe.sysio`: the single device funnel and the access ledger the dry
  run's claim rests on, plus the `perf_event_open(2)` adapter with
  `struct perf_event_attr` as documented in `linux/perf_event.h`.
- `boardprobe.trace`: trace documents. A document declares the provenance of
  its own SAMPLES, and **may not declare `live`**: replaying a file is not a
  live reading, and a document that could claim it would be a one-line route
  from a text editor to a measured label.
- `boardprobe.statistics`: aggregates that cannot exist without an
  attribution. Count, mean, sample sd, min, median, p95 and max, with `sd`
  omitted at n = 1 rather than returned as `nan`; `derive` for second-order
  figures; `rail_power`, because the INA3221 driver exposes volts and amperes
  and not watts.
- `boardprobe.harness`: the benchmark harness, with timing and resource
  figures carrying propagated provenance, refusals reported rather than
  dropped, and `characterise_overhead` measuring the harness's own per-repeat
  cost and the perturbation it causes.
- `boardprobe.dryrun`: a dry-run walk over every code path that does not
  require a device, with the device-access ledger and a before-and-after tree
  snapshot as the evidence that it stayed dry.
- `boardprobe.report`: plain-text rendering in which the certificate headline
  always precedes the figures.
- A CLI with `channels`, `sample`, `benchmark`, `replay`, `provenance`,
  `dry-run` and `attest`.
- Two committed trace fixtures, both synthesised in a container at declared
  seeds, both saying so in their own `origin` field, re-synthesised and
  compared value by value by `tests/test_fixtures.py`.
- `docs/DEPLOYMENT.md`, `docs/RECOVERY.md` and `docs/REQUIREMENTS.md`. Eleven
  validation scripts with their raw output, four example figures, and 291
  tests including Hypothesis property tests over the lattice and the whole
  propagation path.

### Measured and reported as found

- **No figure in this repository was measured on an NVIDIA Jetson Orin Nano.**
  The board was not reachable. Every declared sysfs path, counter name, raw
  unit and scale is UNVERIFIED on hardware; `verified_on_hardware` is `False`
  on all 14 channels and a check asserts that it is.
- **15 adversarial routes attempted, 15 refused, 0 succeeded** in the final
  design. Two succeeded while the check was being written; see below.
- **21 negative controls: 13 must fire and 13 fired; 8 must not fire and 8
  stayed silent.** The two plausibility windows are declared in the script
  with their reasoning and are NOT quoted from any datasheet, because no
  NVIDIA document was read against hardware for this build.
- **The lattice laws hold over all 9 pairs and all 27 triples**, and
  `meet_all` agrees with the weakest-element rule over 1092 assignments of
  length 1 to 6.
- **The chain re-derived from its Hasse diagram by Warshall's transitive
  closure agrees with `meet_all` on 1092 assignments, 0 disagreements**, and
  the propagation counted by inclusion and exclusion matches `3ⁿ − 2ⁿ`,
  `2ⁿ − 1`, `1` over 9840 assignments for n = 1..8 with 0 mismatches. Both are
  mathematics the implementation does not use.
- **The live backend refuses 13 of 13 board channels** on a host without the
  board, every refusal naming a relative locator and what to check. The one
  channel it serves is the monotonic clock, and that is a live reading of the
  build host's clock and not of a Jetson.
- **The harness in-bracket overhead sits inside an independently measured
  sandwich** — median 2.740 ≤ 4.348 ≤ 7.685 µs on the build host, with both
  inequalities holding on 7 of 7 trials.
- **The harness is not an instrument below a few microseconds.** The measured
  perturbation of a 2.3 µs payload is a median 3.33× over five trials, falling
  to 1.037× at 255 µs. Single trials have come back below 1.0 at payload sizes
  above about 38 µs, which is reported as a statement that the perturbation
  there is comparable with the measurement noise rather than as a negative
  overhead; the median and a min-to-max band are reported for that reason.
- **0 of 291 `pytest.approx` assertions depend on the hidden 1e-12 absolute
  default**, and 0 of 291 rest on an absolute bound when every `abs` is forced
  to 0.0.
- **48 script runs across four working directories for the validation scripts
  and two for the examples, 0 non-zero exits.**
- **The tracked-file self-check is structurally unmeasurable by a build
  agent** and is reported as unmeasurable rather than passed on an empty set.
  The coordinator re-ran it after committing, where it is real: **78 tracked
  files examined, 0 offenders.**

### Defects found during the build and pinned as regressions

- **A simulated reading could be laundered into a figure labelled `MEASURED`
  in one line, and it was.** `dataclasses.replace(simulated_reading,
  provenance=Provenance.LIVE)` re-ran the `Reading`'s `__post_init__`, which
  **recomputed the tamper seal**, so the laundered reading was validly sealed,
  carried the simulated reading's own channel and value, claimed `LIVE`, and
  the choke point granted `MEASURED`. No private state was touched;
  `dataclasses.replace` is public API on every dataclass. The seal did not
  help because the seal was recomputed by the mechanism being abused. **Fixed
  in the design, not the check:** a `Reading`'s first field is now a
  `dataclasses.InitVar` holding a module-private minting token, so
  `dataclasses.replace` and `copy.replace` both raise, and `mint()` is the
  single entry point. Pinned with literal inputs in `tests/test_readings.py`
  and as route 10 of the adversarial check.
- **A backend's declared provenance could be rebound at run time.**
  Provenance was read from `type(self).backend_provenance`, an ordinary class
  attribute, so `SimulatedSource.backend_provenance = Provenance.LIVE` made a
  simulated backend stamp `LIVE`. **Fixed in the design:** the three backends'
  declared provenances are captured in a module-private registry when
  `boardprobe.sources` finishes importing and the `provenance` property
  compares the class attribute against the registry on every read. Route 13 of
  the adversarial check.
- **`numpy.mean` returned `inf` for a mean that is representable.** Hypothesis
  falsified "the mean lies between the min and the max" at
  `[8.988465674311579e+307, 8.98846567431158e+307]`: `numpy.mean` sums before
  dividing, the sum exceeds the float64 maximum, and the mean came back `inf`
  for a true mean of 8.98846567431158e+307. `numpy.std` is worse because it
  squares. Every aggregate is now computed in `numpy.longdouble`, the standard
  deviation on centred values, and a result that is still not representable as
  a float64 is **refused by name rather than reported as infinity**. Both the
  overflow input and the genuinely-unrepresentable case are pinned with
  literal inputs in `tests/test_regressions.py`.
- **A finiteness check placed before a cast does not check anything.** The
  first version of the overflow guard tested `numpy.isfinite` on the
  longdouble value, which is finite at 2.4e+308, and then cast it to a float64
  `inf`. The check now runs after the cast.
- **A committed file that names an absolute-path pattern as a literal is what
  an absolute-path scanner looks for.** Two files in this product contained
  the literal prefix while asserting its absence; both now assemble the needle
  from `os.sep`. This is the fourth occurrence of that class in this mission
  and it was introduced here, again, in the act of writing about it.
- **A self-check that reported a vacuous PASS.** The tracked-file check
  inherited from an earlier product passed on an empty `git ls-files` listing,
  which is what a build agent always sees because the coordinator holds every
  commit. It now reports STRUCTURALLY UNMEASURABLE, and measures
  `git check-ignore -v --no-index` against every shipped fixture instead —
  which is the P060 defect class, where a `.gitignore` pattern with no leading
  slash silently excluded a required committed fixture.

### Known limitations

See `README.md` for the full list. The dominant ones: no figure here was
measured on an NVIDIA Jetson Orin Nano; every declared sysfs path and counter
name is unverified, and the thermal-zone index for a given zone name is not
fixed by any document; the `perf_event_open` aarch64 syscall number has never
been executed; `docs/DEPLOYMENT.md` and `docs/RECOVERY.md` have never been
executed on a board; the harness overhead figures are x86_64 figures and will
not transfer; the tamper seal is not a security boundary and `mint()` lets its
caller choose a provenance; and `numpy>=2.5,<2.6` was measured at 2.5.3 only,
with 2.5.4 admitted by the ceiling and untested because installing a second
numpy would have required `pip` against a shared pre-pinned interpreter this
product is forbidden to modify.

[0.1.0]: https://github.com/OmAcharya-avtr/boardprobe/releases/tag/v0.1.0
