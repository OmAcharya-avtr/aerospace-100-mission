# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-10-29

First release. Status: TESTING. Validation level 2, research grade.

### Added

- `tlecheck.elements`: `ElementSet`, `parse_tle`, `format_tle`, `checksum` and
  `apply_checksum`. Parsing is by fixed column span and never raises; a
  malformed line produces `faults`. `format_tle` reproduces a published element
  set for object 25544 character for character, all 69 columns of both lines.
- `tlecheck.determinability`: eleven limits, computed before any screen runs and
  returned as the **first** field of every verdict — three structural, four
  computed, four data-dependent. Equation (13) is exact by exhaustive
  enumeration over 91,368 two-column digit substitutions and is cross-checked
  against a closed form of 72/81 derived by a congruence argument.
- `tlecheck.bounds`: physical, format and policy bands kept apart, with
  equations (1) to (4) and the WGS-72 constants the propagator itself uses.
- `tlecheck.epoch`: equations (5) and (6), with the two-digit year window and
  its 2057 cliff reported as a limit.
- `tlecheck.history`: the object against itself, with equations (7) to (10), a
  quantisation floor at one count of each element's own text field, and an
  explicit refusal when there is not enough history to estimate a scale.
- `tlecheck.continuity`: ephemeris continuity against the object's own
  predecessor, in kilometres, with a self-calibrated tolerance and the screen's
  own resolution returned as equation (27).
- `tlecheck.divergence`: cross-source separation in kilometres by equation (12),
  with `which_source_is_correct` present as a field and always `None`.
- `tlecheck.propagate`: a failure-reporting wrapper over `sgp4`, where `loud` is
  the propagator's own verdict and nothing this package computes.
- `tlecheck.consequence`: equation (11) and the three-class boundary, where the
  class is a **measured** position error in kilometres and the threshold is a
  declared policy value.
- `tlecheck.screens` and `tlecheck.verdict`: six screens returning evidence with
  no verdict, then a decision table printed in full in the module docstring,
  equation (15) as a sweepable score, and `integrity_ok` as a separate output
  from `predicted_class`.
- `tlecheck.dataset`: a seeded synthetic catalogue built with the propagator
  itself, six orbit families, eight corruption families, equations (16) to (18)
  and the Kozai inverse.
- `tlecheck.features`, `tlecheck.learned`, `tlecheck.benchmark`: three declared
  feature configurations, a random forest with two uncertainty outputs, a
  majority-class baseline, and equations (21) to (23).
- A command line with `info`, `check`, `determinability`, `consequence`,
  `screen`, `benchmark` and `divergence`.
- Ten validation scripts with their raw output, ten example figures, and
  **491 tests** including Hypothesis property tests and twelve pinned
  regressions.

### Measured and reported as found

- **The product specification's central prediction was wrong, and the result is
  published as measured.** It predicted the explicit rule screens would match
  or beat the learned classifier on every class they cover. On a 420-object
  held-out split the learned model reaches a silent-class recall of **0.7946**
  at a false-positive rate of 0.0455 where the swept rule baseline reaches
  **0.4286**, and takes 3 of 6 per-class metrics against the rules' 2.
- **The baseline was strengthened five times before that was published**, and
  the margin fell from about **+0.78** recall to **+0.3661**. Each
  strengthening and what it was worth is in `validation/VALIDATION.md`
  section 7, where every superseded behaviour is reinstated and re-measured by
  `validation/validate_baseline_history.py` and the overall build-time scores
  are labelled as not reproducible from the shipped code.
- **The rules keep three things the learned model does not have.** The
  integrity verdict is exact: precision and recall of 1.0000 on 57 element sets
  with **0** false positives in 363, because a checksum is a digit sum and not
  a statistic. Loud-failure recall is 0.9444 from a check against a number
  rather than an estimate. And on two corruption families withheld from
  training the rules win by **+0.2153** and **+0.1463** of area, with the
  learned model recovering when the family is restored.
- **The checksum is blind to exactly one in nine two-digit corruptions and to
  every digit transposition**, enumerated over 91,368 and 992 cases and
  confirmed against a closed form. More than that, **0 of 85** re-checksummed
  corruptions on the test split were seen by it, which is what a ground system
  that formats and then checksums produces.
- **The corruption with the loudest screen has exactly zero consequence.** All
  31 `checksum_digit` element sets have a propagated position error of
  **0.0000 km**. The screen fires on every one and is right to.
- **Four of six orbit families cannot be screened for a one-kilometre
  consequence by comparing element sets at all**, with equation (27)
  resolutions from 1.7898 km to 29.4676 km. The screen withholds its absolute
  test and says so as computed output.
- **The learned model answers confidently where the analytic screen withholds.**
  7 of 32 errors on the unresolvable subset carry a decision margin above 0.5,
  against 0 of 11 on the resolvable one.
- **The cross-source screen's precision is capped at 0.8889 by a symmetry it
  cannot break.** The corpus contains the same observable situation labelled
  both ways and the separation distributions overlap, so 3 of 130 records are
  irreducible false positives. Equation (12) forward and backward on one pair:
  10.44971342 km both ways, bit-exactly.
- **A single flipped drag-term sign is worth 1.8993 km after one day, 93.0648 km
  after seven and 1704.5723 km after thirty**, with a valid checksum and every
  element inside every bound, and the propagator reporting success at every
  sample.
- **The declared 7 day epoch requirement and the declared 1 km silent threshold
  disagree by a factor of about four** on the constructed stale control: the
  requirement flags from 7.5 days and the measured consequence first exceeds
  1 km at 30 days. Reported rather than resolved.
- **The history screen's false-positive rate is 0.0433 per element** at a
  four-sigma threshold against the 0.0001 a Gaussian tail would give, from the
  small-sample variance of a median absolute deviation over twelve points.
  Reported rather than corrected.
- **A threshold of 8.0 scores better than the declared 4.0 on this corpus** and
  was not adopted, because choosing it for that reason would be fitting the
  threshold to the corruption distribution.
- **Equation (5) is one day wrong outside 1900 March to 2100 February**, which
  is asserted as a test. No decodable element set reaches there.
- **Equation (1) is biased by up to 2.9231 km** across the six families, and the
  cross-check requires the bias to be non-zero as well as bounded, because a
  vanishing difference would mean the check is not exercising the Kozai
  transformation at all.
- **Seventeen controls: twelve that must fire did, five that must not did not**,
  each asserted individually and each anchored on a constructed element set
  rather than on an output of this package.

### Defects found during the build and pinned as regressions

- **Hypothesis falsified the affine invariance of equation (8) immediately.**
  With `values=[0.0, 1.0, 6.663237840215803e-40]` and a shift of 1.0, adding
  1.0 to every element collapses the subnormal element onto 1.0 exactly, the
  shifted sample is constant, and the modified z-score jumps from
  1.0122567671316239e+39 to an infinity. The identity is exact in exact
  arithmetic and conditional in floating point, the docstring now says which,
  and the counterexample is pinned with its literal inputs.
- **Robust scales with no quantisation floor produced z-scores of order 1e6 on
  a quantised history.** A history whose mean motion is constant to the eighth
  written decimal has a detrended scale of order 1e-8. Measured by
  `validation/validate_baseline_history.py` on a constructed unjittered history
  whose candidate sits exactly one count from its trend prediction: the
  unfloored z-score is **2.531364e+06** times the floored one. Fixed with a
  floor at one count of each element's own text field, derived from the format
  rather than tuned. **The honest other half, also published:** on the corpus
  this repository now ships the floor changes nothing, because the generator's
  declared mean-motion jitter already keeps the scale above one count — the
  largest finite z-score over 217 uncorrupted element sets is 6.6304 either
  way.
- **The right ascension of the ascending node was in the level-test group.** The
  node regresses secularly, so a level test against a median gave a modified
  z-score between 1.3787 and 1.5246 on every one of 217 uncorrupted element
  sets, a spread of 0.1459 — a fixed offset carrying no information at all.
  Moved to the trend group and pinned there.
- **The generator's element histories were not dynamically consistent, twice
  over.** All three versions are reinstated and re-measured side by side by
  `validation/validate_baseline_history.py`. Median propagated disagreement
  between consecutive uncorrupted element sets, low-Earth and sun-synchronous:
  version 1, with first-order J2 rates and a mean anomaly at `360 n dt`,
  **21.03 km** and **49.33 km**; version 2, with the propagator run backwards
  but `Satrec.nm` written straight into the Kozai field, **112.09 km** and
  **603.64 km**; version 3, the current one, **0.0564 km** and **0.0231 km**.
  A continuity screen measuring either of the first two is measuring the
  generator. The two superseded versions are not ordered against each other and
  the measurement says so.
- **The checksum indicator in equation (15) put 31 zero-consequence element
  sets into the silent class**, 31 of the baseline's 56 avoidable false
  positives. The checksum screen was right and the consequence prediction was
  wrong and they were the same decision. Fixed by making integrity a separate
  output.
- **Predicting a loud failure from this package's own parse failure drove an
  operating characteristic below chance.** The rules predicted loud for 13
  field-shifted element sets of which the propagator refused 1, and the
  field-shift area was 0.3686. Fixed by setting equation (15) to infinity for
  an unreadable field instead; the area became 0.9065 and loud recall fell from
  1.0000 to 0.9444, which is the published cost.
- **The valid-but-stale control was constructed wrongly and reported a
  meaningless number.** Stamping an earlier epoch onto an element set without
  moving its angles produces a different orbit, not an older element set, and
  the control measured 9752.6918 km at an age of 1.5 days. Fixed with
  `tlecheck.dataset.rewound`, and the wrong number is kept in
  `validation/VALIDATION.md` because it is instructive.
- **A parse-fault string without a line prefix was attributed to the wrong
  line.** The eccentricity fault read `eccentricity: columns 27-33 ...` and the
  format screen guessed line 1. Every fault string now carries its line and a
  regression asserts it.
- **A known-answer test for equation (22) had the wrong hand arithmetic.** It
  reasoned that a target false-positive rate of 0.25 over four negatives would
  stop the threshold at 0.4 for a recall of 0.75; the threshold comes down to
  0.35 and the recall is 1.0. The test was the defect, not the code, and the
  corrected arithmetic is pinned.
- **The working-directory probe reported a path defect that was a timing.**
  `validate_learned.py` printed its own process time, so its output hash
  differed between every run. Timing lines are now tagged `[unstable]` by the
  scripts and dropped before hashing, so a timing cannot make a path check pass
  or fail.
- **The probe overwrote the committed full-size outputs with small-corpus
  numbers.** Fixed with `TLECHECK_OUTPUT_DIR` and `TLECHECK_FIGURE_DIR`, and
  the probe now asserts afterwards that every committed output still carries the
  full-size corpus line.

### Known limitations

See `README.md`. The dominant ones: a clean verdict is not a statement that a
propagated ephemeris is accurate and nothing here can establish that; every
screening number is measured against a seeded synthetic corpus and not against
flight data; this package ships no catalogue and cannot get you one; the
checksum is blind to a whole enumerated family of corruptions; the continuity
screen cannot resolve the declared threshold on four of six orbit families; and
the cross-source screen cannot say which source is wrong at any threshold.

[0.1.0]: https://github.com/OmAcharya-avtr/tlecheck/releases/tag/v0.1.0
