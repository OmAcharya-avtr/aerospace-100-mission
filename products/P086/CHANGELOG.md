# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-10-28

First release. Status: TESTING. Validation level 2, research grade.

### Added

- `telemgap.record`: `ChannelSpec`, `RecordSchema` and `TelemetryRecord`, with
  the affine decommutation of equation (1), a declared measurement-noise
  sigma, and `without_validity()` / `without_raw_words()` views for
  determinability experiments.
- `telemgap.determinability`: the seven limits, computed from the record
  supplied and returned as the **first** field of every report — frame
  validity, raw words, counter horizon, in-range corruption, cadence
  declaration, time source and gap interior. The in-range corruption limit is
  computed exactly by enumerating all 65,536 word codes of a 16-bit channel.
- `telemgap.screens`: six screens — counter continuity, timestamp
  monotonicity, cadence, declared range, frame error control, and a
  strengthened robust local-residual screen — returning evidence and no
  verdict.
- `telemgap.reparse`: the reinterpretation test, with five declared
  alternative readings of a raw word and the two conditions (7a) and (7b) a
  candidate must satisfy to count as evidence of a misparse.
- `telemgap.classify`: the three-way verdict with `undecided` as a first-class
  fourth, a decision table printed in full in the module docstring, and a
  stated reason on every undecided verdict.
- `telemgap.policy`: the refusable gap policy. Equation (3) Taylor remainder,
  equation (4) zero-order hold, equation (3′) coverage statement, equation (9)
  Savitzky-Golay curvature estimate with its noise floor and bandwidth,
  equation (16) robust noise scale, equation (17) high-frequency variance
  share, and six gates per gap and channel.
- `telemgap.interpolate`: fills only the permitted gaps and leaves the refused
  ones `nan`, with the refusal mask returned.
- `telemgap.dataset`: a seeded synthetic generator with complete ground truth,
  self-consistent by construction.
- `telemgap.features`, `telemgap.learned`, `telemgap.benchmark`: fifteen
  frame-channel features with channel identity deliberately excluded, a random
  forest with two uncertainty outputs, a majority-class guess, and per-class
  scoring by equations (13) to (15b).
- A CLI with `info`, `check`, `policy`, `refuse`, `determinability` and
  `benchmark`.
- Eight validation scripts with their raw output, nine example figures, and
  334 tests including Hypothesis property tests and six pinned regressions.

### Measured and reported as found

- **The declared-range screen cannot see 0.862371 of single-bit errors** on the
  worst generated channel, enumerated over all 65,536 word codes. That is an
  upper bound on what it must miss and it is reported as a determinability
  limit, not a caveat.
- **Equation (3) held on all 12,800 chord-versus-truth comparisons** across 32
  swept cells, against the analytic curvature of equation (12) and against the
  independently derived Bernstein bound of equation (8).
- **The policy's coverage statement failed once in sixteen channel-records**,
  rate 0.0625. The failure traces to a bit error that stayed inside its
  declared range and passed the frame error-control check, so no screen saw it
  and the endpoint gate could not refuse the gap it anchored. Published as
  measured.
- **An estimated curvature bound under-predicts a near-Nyquist channel by a
  factor of 203**, with the closed-form transfer of the estimator at
  `f T = 0.35` being 0.00326383. The estimate sits above its own noise floor,
  so the measurability gate passes it and only the resolvability gate catches
  it. With that gate disabled the policy exceeds its prediction on 8 of 8
  seeds.
- **Blinding the endpoint integrity gate makes the policy exceed its
  prediction on 3 of 8 seeds.** The integrity classification is an input to
  the policy, not documentation beside it.
- **On the full held-out population the learned model wins four of nine
  metrics against the strengthened rule baseline** — rules 3, learned 4, tie 2
  — which is *not* what the product specification predicted, and is reported
  as measured. The rules keep the one large margin, surfacing 0.7647 of the
  corrupt class against 0.5098, because they abstain where they cannot decide.
- **One of the three declared learned configurations loses to a majority-class
  guess by 56.5 accuracy points** on the ambiguous residual set it exists for;
  another beats it by 10.5. The imbalance handling decides the answer
  completely and the architecture is identical.
- **The best corrupt recall reached by any predictor here is 0.5098.** The
  ceiling is the in-range corruption fraction above.
- **A clock reset degrades gap recall from 1.0000 to a mean of 0.9602** and to
  0.6667 on the worst swept cell, because the record's time base is not
  single-valued after one.
- **The robust noise-scale estimator of equation (16) overstates the noise of
  a near-Nyquist channel by a factor of 37.1**, because a second difference
  cannot tell a fast sinusoid from noise.
- **Fifteen controls, eleven of which must fire and eleven of which did; four
  of which must not fire and none of which did.**

### Defects found during the build and pinned as regressions

- A left-insertion lookup against the truth grid reported a one-cadence signal
  difference as a reconstruction error, inflating the measured error on
  `battery_voltage` from 0.0049 to 0.84. The record's timestamp jitter is what
  puts the reconstruction grid anchor between two truth samples. Fixed with a
  nearest-index lookup and pinned.
- The gap policy's error prediction was violated by a factor of 23.8 where a
  gap endpoint was itself anomalous. Equations (3) and (4) bound an
  interpolation anchored on *correct* endpoints. Fixed by the `endpoints_clean`
  gate, which makes the integrity classification an input to the policy.
- A frame whose error-control field failed produced no anomaly at all unless a
  value screen also fired, so a known-damaged frame could anchor a gap. Fixed
  by the `frame_validity_failed` observation and its `value_in_failed_frame`
  companion, which is where most of the ambiguous residual set now comes from.
- The generator's frame-level injections overwrote one another's ground-truth
  labels, and a reorder whose window ended on the source of a duplicate pair
  separated the two copies so the counter screen could not see the duplicate.
  Measured as a duplicate recall of 0.9688 and a clock-reset precision of
  0.5000 that were artefacts of the labelling, not of the screens. Fixed with a
  reservation array; recall is now 1.0000 on every swept cell.
- The curvature estimator was a plain second difference and was dominated by
  measurement noise by five orders of magnitude, making every estimated bound
  vacuous. Replaced by a Savitzky-Golay quadratic fit with a computed noise
  floor and a measurability verdict.
- `np.rint` rounds half to even, so an interval of exactly 1.5 cadences rounds
  to two slots and the cadence residual is negative. Pinned.
- A window containing only non-finite neighbours fell through to a `nanmedian`
  over an empty slice. Pinned with its fallback.

### Known limitations

See `README.md`. The dominant ones: every number is measured against a seeded
synthetic generator and not against flight data; the corrupt class has a
ceiling set by the in-range flip fraction that this package cannot raise; the
coverage statement of equation (3′) has a measured failure rate; and an
estimated curvature bound is not trustworthy on a channel with near-Nyquist
content.

[0.1.0]: https://github.com/OmAcharya-avtr/telemgap/releases/tag/v0.1.0
