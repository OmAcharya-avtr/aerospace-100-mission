# Dataset card — `tlecheck.dataset`

**This package ships no element catalogue, and that is a deliberate limitation
rather than an oversight.** The authoritative source of two-line element sets is
an operator's catalogue — Space-Track, or a mission's own orbit determination —
and it is not redistributable, not reproducible from a repository, and not
something a screening tool should bundle a stale copy of. Every element set in
this repository's tests, validation and examples is constructed from declared
element values by this module.

That has one large advantage for a validation exercise and one large cost, and
both are stated plainly below.

## What it contains

A seeded synthetic catalogue. Each record is one screening case with complete
ground truth:

| field | what it is |
|---|---|
| `satnum` | object number, unique within a split |
| `family` | one of six orbit archetypes |
| `corruption` | the ground-truth corruption family, one of eight |
| `candidate` | the element set to be screened |
| `reference` | the uncorrupted element set at the candidate's intended epoch. Used **only** to measure consequence; never given to a screen or to the model, and not reachable from `tlecheck.features` |
| `history` | the object's prior element sets, oldest first |
| `other_sources` | element sets for the same object from other catalogues, possibly empty |
| `use_jd` | the Julian date the ephemeris is wanted at |

Default size: 420 objects per split, 12 prior element sets per object at a 1.5
day cadence, with the ephemeris wanted 1 day after the current epoch. That is
2,730 element sets per split written as conforming two-line text, and the whole
corpus regenerates deterministically in under a second. **Nothing is committed
as data.** The generator is the dataset.

## How the element histories are built, and why it matters

Each object's history is built **with the SGP4 propagator itself**. One base
element set is drawn per object from its family's declared centres and spreads;
each history entry is then produced by running SGP4 backwards from that base to
the entry's epoch, reading the mean elements SGP4 computed there, converting the
mean motion back to the Kozai form the text format carries by equation (18),
adding seeded orbit-determination jitter, and writing a conforming element set.

The result is an element history that is **dynamically consistent under the
propagator the screens are screening for**. That matters because the continuity
screen of `tlecheck.continuity` measures exactly that consistency, and a
generator whose own histories were inconsistent would make that screen measure
the generator.

Two earlier versions of this generator were replaced and the measurements are
recorded here because they are the reason the current construction exists.

All three versions are reinstated and re-measured side by side on the same two
orbits by `validation/validate_baseline_history.py`, so the figures below are
reproducible rather than remembered. The quantity is the median propagated
disagreement between consecutive uncorrupted element sets of the same object.

| version | what it did | low-Earth | sun-synchronous |
|---|---|---|---|
| 1 | advanced the elements with the first-order J2 secular rates of equations (16) and (17) and a mean anomaly advanced at `360 n dt` | **21.03 km** | **49.33 km** |
| 2 | ran SGP4 backwards and wrote `Satrec.nm` straight into the mean-motion field | **112.09 km** | **603.64 km** |
| 3, current | the same, with the Kozai inverse of equation (18) applied first | **0.0564 km** | **0.0231 km** |

The two superseded versions are **not** ordered against each other, and the
measurement says so: version 2 is worse than version 1 on both orbits. Version
2's error is a systematic half-kilometre bias in the semi-major axis, which is a
period error and therefore an along-track drift; version 1's was a kinematic
mean anomaly whose error happens to be smaller on these two orbits. Version 3
beats both by three orders of magnitude, and the only difference between
versions 2 and 3 is the fixed-point inverse of equation (18) — one line, worth a
factor of 1987 and 26175 respectively.

Equations (16) and (17) are retained as a first-order analytic statement of the
same secular theory and are compared against the propagator's own `nodedot` and
`argpdot` in `validation/validate_bounds.py`, where the worst relative
difference is 0.001579 and 0.279937.

## The six orbit families

Element centres are representative of the regime named. **No numerical value
here is taken from any specific object or from any catalogue.**

| family | mean motion rev/day | inclination deg | eccentricity | B* 1/Re | why it is in the corpus |
|---|---|---|---|---|---|
| `leo_sun_synchronous` | 14.60 | 98.20 | 0.0012 | 1.1e-4 | the commonest Earth-observation regime, and the one whose node regresses fastest |
| `leo_low_inclination` | 15.50 | 51.60 | 0.0008 | 4.0e-5 | a low-drag low-Earth object; the family with the best continuity resolution |
| `leo_decaying` | 15.92 | 52.00 | 0.0021 | 1.0e-3 | high drag, so its element sets genuinely do not predict their own successors; the hardest family for every screen here |
| `meo_navigation` | 2.0056 | 55.00 | 0.0019 | 1.0e-6 | twelve-hour orbit, negligible drag, propagated by SGP4's deep-space path |
| `geo_stationary` | 1.00272 | 0.0600 | 0.00020 | **−2.0e-5** | the drag term is legitimately **negative**, which is why the bounds screen does not treat a negative B* as a violation |
| `heo_critical_inclination` | 2.0060 | 63.40 | 0.7200 | 1.2e-4 | sits deliberately at the critical inclination, where equation (17) vanishes and an apsidal-rate check carries no information |

Each object's elements are drawn with a declared spread about those centres; the
node, argument of perigee and mean anomaly are drawn uniformly on their ranges.

### Orbit-determination jitter

Every generated element set carries seeded jitter, so the history is not
perfectly self-consistent and the screens have something to be right or wrong
about.

| element | standard deviation |
|---|---|
| inclination | 2.0e-4 deg |
| right ascension of the ascending node | 2.0e-3 deg |
| argument of perigee | 4.0e-3 deg |
| mean anomaly | 4.0e-3 deg |
| eccentricity | 2.0e-6 |
| mean motion | 1.5e-8, relative |
| B* | 0.03, relative |

These were chosen to put the generated element-to-element continuity of a
low-Earth object in the range a real catalogue achieves, tens to hundreds of
metres, **not** to produce a particular screening result. The resulting
continuity is measured by family in
`validation/validate_rule_screens_output.txt`: a median resolution of 0.3895 km
for the sun-synchronous family and 29.4676 km for the decaying one.

## What is injected, and how

Eight families. One is drawn uniformly from the seven corrupting ones with
probability equal to the declared corruption rate, 0.5 by default.

| family | what is changed | checksum afterwards | measured on the test split |
|---|---|---|---|
| `none` | nothing | valid | 218 records, all benign, position error exactly 0 km |
| `checksum_digit` | the checksum column alone; the data is untouched | **invalid** | 31 records, all benign, position error exactly **0.0000 km** |
| `data_digit` | one digit of one numeric element field; the stated checksum is left as it was | **invalid** | 26 records: 15 silent, 11 benign, median 5.80 km |
| `field_shift` | one character deleted from the line body, everything to its right shifted one column left, a space padding column 68, and **the checksum recomputed** | valid | 27 records: 1 loud, 18 silent, 8 benign, median 35.84 km |
| `sign_flip` | the sign of the B* drag term, checksum recomputed | valid | 32 records: 17 silent, 15 benign, median 1.25 km |
| `implausible_eccentricity` | the eccentricity replaced with a value far outside the object's regime, checksum recomputed | valid | 26 records: 17 loud, 9 silent, median 34269.84 km |
| `stale_epoch` | nothing. The candidate is a genuine, uncorrupted element set from earlier in the object's own history | valid | 35 records: 29 silent, 6 benign, median 4.93 km |
| `cross_source_divergence` | nothing in the candidate. A second source publishes a slightly different element set and the candidate is that second source | valid | 25 records: 24 silent, 1 benign, median 6.02 km |

Three design points in that table are load-bearing.

**`checksum_digit` exists to separate two statements.** Its corruption is
detected with certainty by the checksum screen and its measured propagated
consequence is exactly zero. A screening tool that reported one number would
have to choose which of those to say; this package returns both, which is why
`Verdict.integrity_ok` is a separate field from `Verdict.predicted_class`.

**`field_shift` recomputes the checksum, and that is realistic.** A ground
system formats a line and then checksums it, so a formatting fault produces an
internally self-consistent line that no checksum screen can see. 13 of the 27
field-shifted element sets on the test split stayed fully readable by this
package's own parser.

**`data_digit` is confined to the numeric element fields.** The spans are
declared in `tlecheck.dataset.ELEMENT_SPANS`. A corruption of the element set
number or the revolution counter would have no propagated consequence by
construction, and a family whose measured consequence is zero for a reason that
has nothing to do with screening would make the benign class meaningless.

### The symmetric pair, which exists on purpose

A `cross_source_divergence` record and an uncorrupted record with a perturbed
second source attached are **the same observable situation with the labels
swapped**. In the first the candidate is the perturbed source and the measured
consequence is real; in the second the candidate is the clean one and the
measured consequence is zero. From the two element sets alone they are
indistinguishable, because the cross-source comparison of equation (12) is
symmetric to the last bit.

This is the one construction that makes a structural limit measurable rather
than merely stated. Measured on the test split: 105 records where the candidate
is the clean source, with separations from 0.0000 to 4.0612 km and 3 above the
declared 1 km threshold; and 25 where it is the perturbed source, from 0.8786 to
89.1645 km. **The distributions overlap, so no threshold on equation (12)
separates them**, and the 3 are irreducible false positives for that screen.

The perturbation scales are declared: `AGREEING_SOURCE_SCALE = (0.002, 0.060)`
for a second source that agrees and `DIVERGING_SOURCE_SCALE = (0.5, 3.0)` for
one that does not, with the same code producing both so the two constructions
cannot drift apart.

## Splits

| split | seed | first object number | objects |
|---|---|---|---|
| `train` | 20261029 | 10000 | 420 |
| `test` | 20261030 | 40000 | 420 |
| `calibration` | 20261031 | 70000 | 419 |

Disjoint object-number blocks and different seeds, so no object appears in two
splits and the element text of one is not reachable from another. A test asserts
the disjointness and a test asserts that regeneration is byte-identical.

The calibration split has 419 and not 420 objects because the propagator refused
one object's rewind, and the generator **omits** that object rather than quietly
replacing it with a different one. A short history is a real outcome and the
screens must handle it; that is what their determinability verdicts are for.

## Limitations

- **It is not a catalogue and its corruption distribution is invented.** The
  precision and recall figures in this repository are measured against this
  generator. A real corruption distribution would give different numbers. The
  *consequence* figures in kilometres are different in kind: those are real
  measurements of what `sgp4` does with these element sets.
- **The corruption families are the ones the product specification names, and
  nothing else.** There is no modelling of a wholly fabricated element set, a
  mis-attributed object number, a mixed-epoch catalogue dump, or a line that
  arrived truncated. Those are real faults and they are absent.
- **The eight families are drawn uniformly from the corrupting seven.** A real
  catalogue's faults are not uniform and are probably dominated by staleness.
  The corruption rate is swept in `validation/validate_rule_screens.py`; the
  mix within it is not.
- **No frame, time-scale or polar-motion effect is modelled.** Positions are
  differenced in the TEME frame of date that SGP4 returns, which needs no
  transformation for a difference, and nothing here converts to any other frame.
- **The orbit families are representative, not sampled.** Six archetypes with
  declared spreads is not a catalogue's distribution of objects, and the
  decaying family in particular is a single drag regime rather than a
  distribution over solar activity.
- **One jitter model for every object.** A real catalogue's element-set
  agreement depends on tracking coverage, which is not modelled at all.
- **The two-digit epoch year is always in the 2026 window.** Nothing in the
  corpus exercises the 1957-2056 boundary, although
  `validation/validate_bounds.py` tests the underlying date arithmetic across
  the whole window and one day either side of its declared range.

## Licence and provenance

Generated entirely by committed code in this repository under its Apache-2.0
licence. No element set, element value or numerical constant in
`tlecheck.dataset` is taken from any catalogue. The one published element set
that appears anywhere in this repository is a two-line element set for object
25544 at epoch 2019-343.69339541, distributed in the `sgp4` project's own
documentation; it is used in `tests/conftest.py` and in three validation
scripts as a known answer for the checksum, the formatting round trip and the
epoch decoding, and **no screening or benchmarking number is derived from it.**

The WGS-72 constants the generator uses — `J2 = 0.001082616` and
`XKE = 0.07436691613317342` — are read from `sgp4.api.Satrec` rather than
recopied, and a test asserts they are equal to the propagator's own.
