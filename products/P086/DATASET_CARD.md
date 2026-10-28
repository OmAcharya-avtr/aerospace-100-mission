# Dataset card — `telemgap.dataset`

**No flight data is used anywhere in this package, and no number in it is taken
from a real spacecraft.** Everything is generated from closed forms printed in
the module docstring of `src/telemgap/dataset.py`, so every figure can be
regenerated from a seed and the true class of every anomaly is known exactly
rather than adjudicated.

## What it contains

A `GeneratedRecord` is a `TelemetryRecord` as a user would receive it, plus its
complete ground truth:

| field | shape | what it is |
|---|---|---|
| `record` | — | timestamps, counters, engineering values, raw words, per-frame validity flag |
| `value_truth` | `(n, 4)` | 0 clean, 1 corrupt, 2 misparsed, per frame and channel |
| `misparse_kind` | `(n, 4)` | `""`, `"signedness"`, `"byteswap"` or `"scale_decade_down"` |
| `frame_truth` | `(n,)` | `""`, `"duplicate"`, `"reorder"` or `"clock_reset"` |
| `corruption_in_range` | `(n, 4)` | whether an injected corruption left the value inside its declared range |
| `truth_times`, `truth_values` | `(m,)`, `(m, 4)` | the noiseless truth on the generation grid |
| `missing_slots` | `(k,)` | grid slots of the deleted frames |

Nothing is stored on disk. A 12,000-frame record generates in about 0.05 s, so
regeneration is cheaper than committing it, and no data file is committed.

## The four channels

Each is a sum of sinusoids plus an optional linear ramp, equation (11), whose
curvature and slew bounds follow by term-by-term differentiation, equation
(12). Cadence 1 s, 16-bit words, counter modulus 16384 (the CCSDS space-packet
width).

| channel | unit | model | noise sigma | declared range | curvature declared? |
|---|---|---|---|---|---|
| `battery_voltage` | V | 28.5 + 1.2 sin(2π t/600) + 0.35 sin(2π t/137 + 1.1) | 0.004 | [26, 31] | yes |
| `valve_current` | A | 0.030 + 0.004 sin(2π t/400 + 0.3) | 0.012 | [−0.08, 0.08] | no |
| `gyro_rate` | deg/s | 0.40 sin(2π t/240) + 0.08 sin(2π·0.35·t + 0.7) | 0.003 | [−3, 3] | no |
| `tank_pressure` | bar | 13.0 − 2.0e-4 t + 0.05 sin(2π t/1800 + 0.6) | 0.002 | [8, 16] | yes |

Three of these carry a constructed property that the validation depends on, and
each is deliberate:

**`valve_current` is declared unsigned and its field is signed.** Its mean is
0.030 A with a measurement noise of 0.012 A, so roughly six samples in a
thousand have a negative true value, and each of those is stored as a
near-full-scale word and read back at about 0.64 A. This is a *standing
decommutation fault*, not an injected event — it is what that fault looks like
in a real archive, a sparse scatter of impossible readings on an otherwise
quiet channel. The signedness misparse class is entirely this.

**`gyro_rate` carries a term at 0.35 of the sampling frequency.** Below
Nyquist, far above the scale a smoothed second difference can resolve. It
exists so that the curvature estimator of equation (9) can be shown to
under-predict by a factor derivable in closed form, and so that the policy's
resolvability gate has something to catch. Measured: the estimator returns
0.00393x of the analytic curvature on the noiseless channel.

**Only two channels declare a curvature bound.** That asymmetry is realistic —
a sensor's noise is on its datasheet and its curvature bound is not — and it is
what forces both the declared and the estimated path of `telemgap.policy` to be
exercised by every run.

## What is injected, and how

Every injection is recorded in the ground truth, and every one is constructed
so that the stored record stays **self-consistent**: `values ==
ChannelSpec.decode(raw_words)` holds for every frame and channel of every
generated record, including the corrupted and misparsed ones. This is asserted
by test (`tests/test_dataset.py::test_record_is_self_consistent`). An
inconsistent record would let any classifier separate the classes by
re-decoding the words, which no real archive permits, and every measured score
would be meaningless.

| injection | how | default count per 12,000 frames |
|---|---|---|
| `missing` | runs of whole frames deleted, run length drawn from (1,1,1,2,2,3,5,8,20,600) | 10 runs |
| `corruption` | 1 to 3 bits flipped in the stored word; 60 per cent also have the frame error-control flag set to failed | 160 |
| `misparse_byteswap` | the word stored with its bytes reversed, on `tank_pressure` | 40 |
| `misparse_scale` | the word stored for a calibration a decade smaller than declared, on `gyro_rate` | 40 |
| `misparse_signedness` | not injected; see `valve_current` above | about 90 |
| `duplicate` | a frame re-inserted immediately after itself, byte-identical | 12 |
| `reorder` | a frame moved later by 2 to 5 positions, timestamp and counter unchanged | 12 |
| `clock_reset` | a constant subtracted from every subsequent timestamp | 2 |

**Whether a corruption lands inside or outside its declared range is not
chosen.** It falls out of the bit flip, and the split is measured: on the test
split, 110 of 153 corruptions stay inside the declared range and are therefore
invisible to a declared-range screen.

**Frame-level injections reserve the frames they touch.** A reorder refuses a
window that overlaps a duplicate pair or an earlier reorder. Without that
guard a reorder whose window ends on the source of a duplicate pair moves a
third frame between the two copies, the copies are no longer adjacent, and the
counter screen cannot see the duplicate at all. Measured during this build as a
duplicate recall of 0.9688 that was an artefact of the injection order and not
of the screen; with the guard, recall is 1.0000 on every cell of the swept
grid.

## Splits

Three independent realisations, not a partition of one record:

| split | seed | frames | purpose |
|---|---|---|---|
| training | 20261028 | 12,000 | fitting the learned model |
| test | 71120261 | 12,000 | every score reported |
| calibration | 44520261 | 12,000 | the uncertainty measurements only |

Sweep cells use seeds derived deterministically from the cell indices inside
each validation script.

## Limitations

- **It is synthetic.** The anomaly mix, the noise model and the channel
  dynamics are chosen, not observed. Scores measured here do not transfer to a
  real archive and no claim is made that they do.
- **Four channels.** A real spacecraft has hundreds, with correlations between
  them that this generator does not model.
- **One decommutation form.** Equation (1), affine. A polynomial or table
  calibration is out of scope, and a misparse of that kind has no alternative
  reading in `telemgap.reparse`.
- **Three misparse kinds.** Byteswap, signedness and a decade scale error.
  Real decommutation faults include bit-field misalignment, wrong word offset
  and stale calibration, none of which are modelled.
- **The corruption model is uniform bit flips.** A real link produces burst
  errors shaped by the coding scheme, and a real ground chain produces
  structured faults that are not bit flips at all.
- **The frame error-control model is a declared flag, not a computed CRC.**
  Sixty per cent of corruptions set it; the remaining forty per cent model
  corruption introduced after the error-control check, in the ground
  processing chain. That split is a modelling choice and is the direct cause of
  the size of the ambiguous residual set.
- **Timestamp jitter is per sample.** An interval therefore departs from the
  cadence by up to twice the per-sample amplitude, which is why
  `RecordSchema.jitter_s` must be declared at least twice the jitter. Measured
  in `validation/validate_rule_screens_output.txt`.

## Licence and provenance

Generated by this repository's own code, Apache-2.0, copyright © 2026 OPTIMA
Organisation. There is no third-party data in it and therefore no third-party
licence to observe.
