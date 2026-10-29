# Validation

**Validation level 2.** Every number below was produced by a script in this
directory, executed on 2026-10-29, and every script writes its raw output
beside itself as `<script>_output.txt`. No figure here is carried from an
earlier session, estimated, or rounded by hand. Where a check failed, where a
result is unfavourable to this package, or where the baseline beat the learned
component, it is in the table with the others.

Environment, as recorded in the header of every output file: Python 3.13.18,
numpy 2.5.3, scikit-learn 1.9.1, sgp4 2.27, scipy 1.18.1, matplotlib 3.11.2,
pandas 3.0.5, pytest 9.1.1, hypothesis 6.170.13, ruff 0.16.9, two CPU cores.
`astropy` and `skyfield` are named in the README's alternatives table and are
**not installed**; no number in this repository is compared against either.

Declared settings behind every number: 420 objects per split, 12 prior element
sets per object at a 1.5 day cadence, a 1 day consequence horizon at 25 samples,
a 1 km silent threshold, a 7 day epoch requirement and a target false-positive
rate of 0.05. Every one of those is a policy choice and every one is swept
somewhere below.

## What was run

| script | what it measures | checks | failed |
|---|---|---|---|
| `validate_baseline_history.py` | every superseded behaviour this document quotes, reinstated and re-measured beside the current one | 6 | 0 |
| `validate_elements.py` | the checksum and the formatting round trip against a published element set, and equation (13) by enumeration against a closed form | 14 | 0 |
| `validate_bounds.py` | equations (1), (3), (5), (16) and (17) against `sgp4`'s own internals and against an independent day count, plus the bound findings on constructed violations | 19 | 0 |
| `validate_consequence.py` | the measured consequence class, swept over the threshold and the horizon, and the rules' loud prediction against the propagator's own error codes | 12 | 0 |
| `validate_rule_screens.py` | per-class precision and recall over a swept corruption rate, the four-sigma threshold swept, the history false-positive rate, equation (27) by orbit family, and the structural limit of the cross-source screen | 13 | 0 |
| `validate_controls.py` | 17 constructed controls, 12 that must fire and 5 that must not, each asserted in one direction | 17 | 0 |
| `validate_learned.py` | the rule baseline, three learned configurations and a majority-class baseline on the same split, with two transfer experiments and both uncertainty outputs | 9 | 0 |
| `validate_cli.py` | every command line entry point in a subprocess, the error paths, and the tree afterwards | 39 | 0 |
| `worked_example.py` | the worked example quoted in `README.md` | 3 | 0 |
| `probe_working_directories.py` | every script and example from five working directories, by exit status and output hash | 7 | 0 |

Test suite: **491 tests, 491 passed, 0 failed, 0 skipped, 0 errors**, counted
from the `tests` attribute of a junit XML written outside the repository and
cross-checked against a literal count of `testcase` elements. The two agree.

## 1. The element set format

`validate_elements.py`. The only external known answer in this repository is a
two-line element set for object 25544 at epoch 2019-343.69339541, as
distributed in the `sgp4` project's documentation.

| check | reference | result | tolerance |
|---|---|---|---|
| checksum, line 1 | the digit in column 69 | computed **1**, stated **1** | exact integer |
| checksum, line 2 | the digit in column 69 | computed **2**, stated **2** | exact integer |
| formatting round trip | both published lines | **all 69 columns** of each identical | exact string |
| field round trip over the corpus | 2,730 generated element sets | worst reparse difference **0.000e+00** on every element | bit-exact |
| corpus parse faults | the same | **0** of 2,730 | exact |
| corpus checksums | the same | **0** bad lines | exact |

### Equation (13), two ways

The enumeration loops over every digit column and every replacement digit. The
closed form counts the same sets by a congruence argument that enumerates
nothing: for one column the nine replacements give the nine distinct non-zero
residues modulo 10, so a single substitution always changes the digit sum; a
transposition permutes the digits and changes nothing; and for a pair of columns
exactly one of the nine residues at the second column cancels each residue at
the first, so 9 of every 81 pairs are undetected.

| family | line | enumerated | closed form | cases |
|---|---|---|---|---|
| single substitution | 1 | **1.0000000000** | 1 | 432 |
| transposition | 1 | **0.0000000000** | 0 | 992 |
| two-column substitution | 1 | **0.8888888889** | 72/81 | 91,368 |
| single substitution | 2 | **1.0000000000** | 1 | 477 |
| transposition | 2 | **0.0000000000** | 0 | 1,244 |
| two-column substitution | 2 | **0.8888888889** | 72/81 | 111,618 |

The closed form also proves something the enumeration cannot: the answer does
not depend on the digits of the particular line. That is checked by enumerating
a line with twelve digits replaced and getting the identical fraction.

**So exactly one in nine two-digit corruptions, and every digit transposition,
reaches the propagator with a valid checksum.** That is a determinability limit
and it is returned with every verdict.

## 2. The analytic expressions, against mathematics they do not use

`validate_bounds.py`. Three of these compare this package's arithmetic against
`sgp4`'s internals, which perform a different calculation, and one compares a
Julian date against a day count from a different origin by a different
algorithm. In every case the disagreement is bounded and reported rather than
asserted away.

### Equation (1) against the propagator's own semi-major axis

| family | n rev/day | equation (1) km | `sgp4` km | bias km |
|---|---|---|---|---|
| leo_sun_synchronous | 14.60000 | 7071.3131 | 7068.3900 | **+2.9231** |
| leo_low_inclination | 15.50000 | 6794.8651 | 6795.3755 | −0.5104 |
| leo_decaying | 15.92000 | 6674.8257 | 6675.2781 | −0.4524 |
| meo_navigation | 2.00560 | 26560.6740 | 26560.6632 | +0.0108 |
| geo_stationary | 1.00272 | 42164.6843 | 42165.7288 | −1.0445 |
| heo_critical_inclination | 2.00600 | 26557.1430 | 26556.1543 | +0.9887 |

The check requires the bias to be below 20 km **and above 0.001 km**. The
second half matters: a vanishing difference would mean the cross-check is not
exercising the Kozai transformation at all, and the test would be measuring
nothing.

### Equations (16) and (17) against `nodedot` and `argpdot`

| family | eq (16) deg/day | `sgp4` deg/day | relative | eq (17) deg/day | `sgp4` deg/day | relative |
|---|---|---|---|---|---|---|
| leo_sun_synchronous | 0.990415 | 0.988853 | 0.0016 | −3.118844 | −3.116241 | 0.0008 |
| leo_low_inclination | −4.959309 | −4.960693 | 0.0003 | 3.709107 | 3.701236 | 0.0021 |
| leo_decaying | −5.231963 | −5.233277 | 0.0003 | 3.803737 | 3.794856 | 0.0023 |
| meo_navigation | −0.038781 | −0.038781 | 0.0000 | 0.021803 | 0.021798 | 0.0002 |
| geo_stationary | −0.013413 | −0.013415 | 0.0001 | 0.026827 | 0.026830 | 0.0001 |
| heo_critical_inclination | −0.130586 | −0.130551 | 0.0003 | 0.000356 | 0.000278 | **0.2799** |

The last cell is 0.2799 and the bound on it is 0.50, which is loose on purpose
and is said to be loose. The critically inclined family sits where equation (17)
vanishes — the rate is 3.56e-4 deg/day — and a relative comparison of two
near-zero numbers carries no information. Equation (17) at exactly 63.4349488
degrees evaluates to **2.334e-10 deg/day**, which is the real check.

### Equation (5) against an independent day count

| check | result |
|---|---|
| month starts compared, 1957 to 2056 | **1,200** |
| bit-exact disagreements | **0** |
| 1900 January 1 | equation (5) is **one day low** |
| 1900 February 1 | equation (5) is **one day low** |
| 2100 March 1 | equation (5) is **one day high** |
| 2100 December 1 | equation (5) is **one day high** |

The out-of-range disagreements are asserted as tests rather than left as a
claim. The two-digit epoch-year window resolves into 1957 to 2056, which lies
entirely inside the agreeing range, so no element set this package can decode
reaches them.

### Epoch decoding over the corpus

5,460 element sets, worst difference against `sgp4`'s own decoding
**0.000e+00 seconds**. Not a tolerance — bit-exact on every one.

## 3. The measured consequence

`validate_consequence.py`. Nothing here assumes a class from a corruption
family. Every label comes from propagating the candidate and its uncorrupted
reference with `sgp4` over the same absolute grid and differencing the
positions.

| injected corruption | n | loud | silent | benign | median max km | largest max km |
|---|---|---|---|---|---|---|
| none | 218 | 0 | 0 | 218 | **0.000000** | 0.0000 |
| checksum_digit | 31 | 0 | 0 | 31 | **0.000000** | **0.0000** |
| data_digit | 26 | 0 | 15 | 11 | 5.802850 | 16479.1961 |
| field_shift | 27 | 1 | 18 | 8 | 35.837773 | 44276.0284 |
| sign_flip | 32 | 0 | 17 | 15 | 1.246071 | 1920.7668 |
| implausible_eccentricity | 26 | 17 | 9 | 0 | 34269.839604 | 72754.6552 |
| stale_epoch | 35 | 0 | 29 | 6 | 4.931675 | 2910.9148 |
| cross_source_divergence | 25 | 0 | 24 | 1 | 6.024142 | 89.1645 |

**The `checksum_digit` row is the one to read twice.** Its corruption is the
checksum column alone, the data is intact, and the propagated position error is
exactly 0.0000 km on all 31. The checksum screen fires on every one and it is
right to, because the element set is not what its source wrote. Those are two
different statements and this package returns them separately, which is why
`Verdict.integrity_ok` exists.

### The class boundary moves with the declared threshold

| threshold km | benign | silent | loud |
|---|---|---|---|
| 0.1 | 278 | 124 | 18 |
| 0.5 | 282 | 120 | 18 |
| **1.0** | **290** | **112** | **18** |
| 5.0 | 324 | 78 | 18 |
| 50.0 | 361 | 41 | 18 |
| 500.0 | 374 | 28 | 18 |

The loud count does not move at all, across four orders of magnitude of
threshold. A loud failure is the propagator's own verdict and no policy value
of this package changes it.

### The consequence grows with the horizon

| horizon days | benign | silent | loud | median silent km |
|---|---|---|---|---|
| 0.25 | 298 | 107 | 15 | 11.9186 |
| 1.00 | 290 | 112 | 18 | 9.3469 |
| 3.00 | 285 | 117 | 18 | 17.2932 |
| 7.00 | 285 | 114 | 21 | 31.3619 |

### One flipped sign on a published element set

The two lines differ in a single column, the sign of the drag term. Both carry
a valid checksum. Every element is inside every physical bound. The propagator
reports success at every sample.

| horizon days | max error km | rms error km | class |
|---|---|---|---|
| 1.0 | **1.8993** | 0.8759 | silently_wrong |
| 3.0 | 17.0910 | 7.8059 | silently_wrong |
| 7.0 | **93.0648** | 42.2865 | silently_wrong |
| 14.0 | 372.5879 | 168.7601 | silently_wrong |
| 30.0 | **1704.5723** | 773.1676 | silently_wrong |

### The rules' loud prediction against the propagator

| quantity | result |
|---|---|
| records | 420 |
| rules predicted a loud failure | 20 |
| propagator failed loudly | 18 |
| both | 17 |
| loud recall | **0.9444** |
| loud precision | **0.8500** |
| SGP4 error codes seen | code 6 on all 18 |
| misses that were an unreadable field | **1** |
| misses flagged silently wrong instead | 0 |
| misses not flagged at all | **0** |

The rules both over-predict and under-predict a loud failure, and both
directions are published. They over-predict because an element set whose
eccentricity is implausible for its object sometimes still propagates. They
under-predict because an element set with a field this package cannot parse is
classified silently wrong rather than loud, deliberately; see section 7.

## 4. The rule screens over a swept grid

`validate_rule_screens.py`. Six corruption rates, 420 objects per cell, each
cell screened and measured independently.

| rate | loud P | loud R | silent P | silent R | benign P | benign R | accuracy |
|---|---|---|---|---|---|---|---|
| 0.10 | 1.0000 | 1.0000 | 0.1982 | 0.9167 | 0.9935 | 0.7747 | 0.7833 |
| 0.25 | 0.7059 | 1.0000 | 0.3333 | 0.8182 | 0.9813 | 0.7450 | 0.7619 |
| 0.40 | 0.5000 | 0.8462 | 0.5000 | 0.8283 | 0.9701 | 0.7370 | 0.7619 |
| **0.50** | **0.8500** | **0.9444** | **0.5105** | **0.8661** | **0.9429** | **0.6828** | **0.7429** |
| 0.75 | 0.7073 | 0.8286 | 0.6349 | 0.8644 | 0.9130 | 0.6058 | 0.7333 |
| 1.00 | 0.6889 | 0.9118 | 0.7097 | 0.8722 | 0.8438 | 0.5094 | 0.7381 |

Silent-class recall across the grid: minimum 0.8182, median 0.8652, maximum
0.9167. Silent-class precision moves from 0.1982 to 0.7097 across the same
grid, which is the point: precision is a property of the population a screen is
run on and recall is closer to a property of the screen.

### The rule baseline as a swept score

| rate | own recall | own fpr | swept recall at fpr 0.05 | achieved fpr | area |
|---|---|---|---|---|---|
| 0.25 | 0.8182 | 0.2466 | 0.5455 | 0.0438 | 0.8284 |
| **0.50** | **0.8661** | **0.3019** | **0.4286** | **0.0455** | **0.8677** |
| 0.75 | 0.8644 | 0.3621 | **0.0000** | 0.0000 | 0.7707 |
| 1.00 | 0.8722 | 0.4197 | **0.0000** | 0.0000 | 0.7331 |

The two zeros are real and are worth reading. At a corruption rate of 0.75 and
above there are few uncorrupted element sets left, and the unreadable records —
which carry an infinite equation (15) score by design — exhaust the
false-positive budget before any finite threshold can be reached. The rules'
own operating point is unaffected; it is the low-false-positive region of the
swept curve that collapses.

### The declared four-sigma threshold, swept

| threshold | silent P | silent R | accuracy | area |
|---|---|---|---|---|
| 2.0 | 0.3442 | 0.9018 | 0.5143 | 0.8309 |
| 3.0 | 0.4387 | 0.8750 | 0.6667 | 0.8571 |
| **4.0** | **0.5105** | **0.8661** | **0.7429** | **0.8677** |
| 5.0 | 0.5455 | 0.8571 | 0.7762 | 0.8712 |
| 8.0 | 0.6346 | 0.8839 | 0.8167 | 0.8745 |

The declared 4.0 is not the best cell in this table and it was not chosen by
fitting it. It is a four-sigma threshold on a robust z-score, declared in
advance; the sweep shows that 8.0 would score better on this corpus, and
choosing it for that reason would be fitting the threshold to the corruption
distribution. The figures above are what 4.0 is worth.

### The history screen's false-positive rate, on uncorrupted element sets only

| history length | z tests | fraction above 4 | p95 | p99 | max |
|---|---|---|---|---|---|
| 4 | 1,035 | **0.2116** | 12.9284 | 37.8628 | 129.1525 |
| 7 | 1,050 | 0.0867 | 5.3959 | 11.3011 | 12778.2416 |
| **12** | **1,085** | **0.0433** | **3.8566** | **6.5087** | 14034.0863 |
| 18 | 1,055 | 0.0152 | 2.8939 | 4.2772 | 9990.4655 |

At the declared history length the measured per-element false-positive rate is
0.0433, against the 0.0001 a Gaussian four-sigma tail would give. The excess is
the small-sample variance of a median absolute deviation over twelve points, and
it is **reported rather than corrected**: a scale estimated from twelve numbers
is not a scale, and the screen's operating point reflects that. The maxima in
the right-hand column are the residual zero-scale cases, where the quantisation
floor is the only thing keeping the z-score finite.

### Equation (27), the continuity screen's resolution, by orbit family

| family | n | median offset km | median normal km | median resolution km | resolves 1 km |
|---|---|---|---|---|---|
| geo_stationary | 39 | 4.1977 | 4.5579 | **3.3303** | no |
| heo_critical_inclination | 40 | 4.6428 | 5.0580 | **2.1048** | no |
| leo_decaying | 33 | 50.3644 | 41.5192 | **29.4676** | no |
| leo_low_inclination | 31 | 0.8501 | 0.8558 | 0.4723 | **yes** |
| leo_sun_synchronous | 35 | 0.9948 | 0.8064 | 0.3895 | **yes** |
| meo_navigation | 39 | 2.9360 | 2.8900 | **1.7898** | no |

**Four of the six families cannot be screened for a one-kilometre consequence
by comparing element sets at all.** The screen says so, as computed output,
instead of returning a verdict that is noise. For the decaying family the
reason is physical: a high-drag object's element set genuinely does not predict
its own successor's ephemeris to within tens of kilometres.

### The structural limit of the cross-source screen, measured

The corpus contains two constructions of the same observable situation. In the
`cross_source_divergence` family the candidate is the perturbed source; in the
uncorrupted family with a second source attached the candidate is the clean one.
From the two element sets alone they are indistinguishable, because equation (12)
is symmetric.

| construction | n | min km | median km | max km | above 1 km |
|---|---|---|---|---|---|
| candidate is the clean source | 105 | 0.0000 | 0.0819 | **4.0612** | **3** |
| candidate is the perturbed source | 25 | **0.8786** | 6.0241 | 89.1645 | 24 |

The two distributions overlap: the clean construction reaches 4.0612 km and the
perturbed one starts at 0.8786 km. The divergence screen fired on 27 records of
which 24 were measured silently wrong, a precision of **0.8889**, and the 3
false positives are irreducible. Equation (12) forward and backward on one such
pair: **10.44971342 km** both ways, bit-exactly, and
`which_source_is_correct` is `None` in both directions.

### Every screen's firing rate on uncorrupted element sets

| screen finding | fired | rate |
|---|---|---|
| history, bstar | 14 | 0.0642 |
| history, mean motion | 12 | 0.0550 |
| history, node | 9 | 0.0413 |
| history, eccentricity | 6 | 0.0275 |
| history, inclination | 6 | 0.0275 |
| continuity z exceeded | 3 | 0.0138 |
| cross-source diverged | 3 | 0.0138 |
| bounds, perigee below policy | 1 | 0.0046 |
| **checksum mismatch** | **0** | **0.0000** |
| **parse fault** | **0** | **0.0000** |

218 uncorrupted element sets. The format and checksum screens have no false
positive rate at all, because they are arithmetic.

## 5. Controls

`validate_controls.py`. Seventeen controls, each anchored on a **constructed
element set whose corruption is known exactly** and never on an output of this
package. Twelve must fire and five must not, and both directions are asserted.

| control | expected | fired | predicted class |
|---|---|---|---|
| 1 invalid checksum (the declared control) | `checksum_mismatch` | yes | benign |
| 2 non-digit checksum column | `checksum_column_not_a_digit` | yes | silently_wrong |
| 3 valid but stale, 30 days (the declared control) | `epoch_stale` | yes | silently_wrong |
| 4 perigee below the surface | `perigee_below_surface` | yes | will_fail_loudly |
| 5 mean motion above the equation (3) ceiling | `mean_motion_above_physical_ceiling` | yes | will_fail_loudly |
| 6 drag sign flip against a positive history | `history_outlier_bstar` | yes | silently_wrong |
| 7a one character dropped from line 1 | `parse_fault` | yes | silently_wrong |
| 7b one character dropped from line 2 | `history_outlier_raan_deg` | yes | silently_wrong |
| 8 cross-source separation | `cross_source_divergence` | yes | silently_wrong |
| 9 continuity break, 2e-5 rev/day | `ephemeris_continuity_exceeded` | yes | silently_wrong |
| 10 epoch five days in the future | `epoch_in_future` | yes | silently_wrong |
| 11 unexpected ephemeris type | `ephemeris_type_unexpected` | yes | benign |
| 12 the constructed truth itself | nothing | **no** | benign |
| 13 valid and fresh, 4 days | nothing | **no** | benign |
| 14 the mean anomaly moved by one count | nothing | **no** | benign |
| 15 negative drag against a negative history | nothing | **no** | benign |
| 16 an agreeing second source | nothing | **no** | benign |

Controls 1 and 11 fire and predict `benign`, which is the integrity/consequence
separation working: a problem is reported, and the consequence prediction comes
from the screens that have a magnitude. Control 7b is the one worth reading:
the same fault class as 7a on the other line produced **no** unreadable field
and **no** bound violation, so only the object-against-itself screens could see
it. A shift is not one fault class; it is whatever the shifted characters
happen to say.

### The two specification-named controls, with their measured consequence

| control | screen verdict | measured position error |
|---|---|---|
| invalid checksum | `checksum_mismatch` fires, integrity not intact | **exactly 0.000000 km** |
| valid but stale, 1.5 days | not stale against a 7 day requirement | 0.0193 km |
| valid but stale, 3.0 days | not stale | 0.0462 km |
| valid but stale, 7.5 days | **stale** | 0.2458 km, class benign |
| valid but stale, 15.0 days | **stale** | 0.8737 km, class benign |
| valid but stale, 30.0 days | **stale** | **3.2964 km, class silently wrong** |
| valid but stale, 60.0 days | **stale** | 12.7868 km, class silently wrong |

**A disagreement reported rather than resolved.** The declared 7 day epoch
requirement flags this element set from 7.5 days, and the measured consequence
first exceeds the declared 1 km threshold at 30 days. For this object the
staleness policy is the more conservative of the two by a factor of about four.
Neither number is wrong; they are two different operational choices, and a
screen that reported only one would be hiding the gap. The determinability
report returns the requirement with every verdict for exactly this reason.

The valid-but-stale control was **wrong in its first version** and the number it
produced is kept here because it is instructive. Stamping an earlier epoch onto
an element set without moving its angles does not produce an older element set;
it produces a different orbit. Measured that way, the control reported
**9752.6918 km at an age of 1.5 days**, which is meaningless. The control is now
built with `tlecheck.dataset.rewound`, which runs the propagator back to the
older epoch and writes the mean elements it computed there.

## 6. The learned component against the baselines

`validate_learned.py`. Same held-out split, same labels, and the learned model
given **every feature the rule baseline uses**, including the continuity offset
and the screen's own resolution. A learned win over a baseline that was shown
less information is not a result.

Corpus: train 420 objects (loud 20, silent 122, benign 278), test 420 (18, 112,
290), calibration 419 (16, 127, 276). Silent-class prevalence on the test split
0.2667.

### The headline

| predictor | recall | achieved fpr | area | accuracy |
|---|---|---|---|---|
| rules at their own threshold | **0.8661** | 0.3019 | 0.8677 | 0.7429 |
| rules swept, equation (15) | **0.4286** | 0.0455 | 0.8677 | 0.7429 |
| majority-class baseline | 0.0000 | 0.0000 | **0.5000** | 0.6905 |
| learned `elements_and_history` | 0.7768 | 0.0487 | 0.9583 | 0.8976 |
| learned `elements_only` | 0.5804 | 0.0487 | 0.7960 | 0.8238 |
| learned `with_rule_score` | **0.7946** | 0.0455 | **0.9629** | **0.9095** |

**This is not what the product specification predicted.** It predicted in
writing that the explicit rule screens would match or beat the learned
classifier on every class they cover, and that the learned model's only
defensible contribution would be the silent class. On this corpus the learned
model wins the silent class by **+0.3661 recall** at a comparable
false-positive rate, and wins on accuracy and on area as well. The result is
published as measured.

Both the rules and every learned configuration beat the majority-class
baseline, whose accuracy is 0.6905 and whose area is 0.5000 by construction
because its score is a constant.

### Per class

| predictor | loud P | loud R | silent P | silent R | benign P | benign R |
|---|---|---|---|---|---|---|
| rules | 0.8500 | **0.9444** | 0.5105 | **0.8661** | **0.9429** | 0.6828 |
| majority | nan | 0.0000 | nan | 0.0000 | 0.6905 | 1.0000 |
| learned `elements_and_history` | **0.9000** | **1.0000** | **0.8723** | 0.7321 | 0.9052 | **0.9552** |
| learned `elements_only` | 0.8182 | 1.0000 | 0.6792 | 0.6429 | 0.8767 | 0.8828 |
| learned `with_rule_score` | 0.8947 | 0.9444 | 0.8713 | 0.7857 | 0.9233 | 0.9552 |

Tally against the best learned configuration: **rules 2, learned 3, tie 1.**
The rules keep silent-class recall and benign precision; the learned model keeps
loud precision, silent precision and benign recall; loud recall is a tie at
0.9444.

### What the rules hold that a classifier is not even asked

| quantity | result |
|---|---|
| element sets whose checksum should be inconsistent | 57 |
| checksum screen flagged | 57 |
| precision | **1.0000** |
| recall | **1.0000** |
| false positives in 363 consistent element sets | **0** |
| element sets corrupted and then re-checksummed | 85 |
| of which the checksum screen saw | **0** |
| element sets with a field this package could not read | 14 |
| field-shifted element sets that stayed fully readable | 13 of 27 |

Precision and recall of exactly 1 here are not statistics. The checksum is a
digit sum: there is no threshold, no training set and no error rate. The 0 of
85 is the determinability limit of equation (13) made concrete, and it is not a
defect.

### Transfer to an unseen orbit family

| withheld family | n | silent | rules area | learned, family withheld | learned, all families |
|---|---|---|---|---|---|
| geo_stationary | 70 | 16 | 0.6528 | 0.8438 | 0.8819 |
| heo_critical_inclination | 70 | 12 | 0.7744 | 0.9095 | 0.9583 |
| leo_decaying | 70 | 22 | **0.8442** | 0.8305 | 0.8741 |
| leo_low_inclination | 70 | 25 | 0.9858 | 0.9973 | 1.0000 |
| leo_sun_synchronous | 70 | 16 | 0.9340 | 0.9722 | 0.9769 |
| meo_navigation | 70 | 21 | 0.9417 | 0.9776 | 0.9815 |

The learned model keeps its advantage on 5 of 6 withheld orbit families, so its
advantage is **not** a memorised map from orbit regime to consequence. The rules
take the decaying family by 0.0137.

### Transfer to an unseen corruption family, where the rules win

| withheld corruption | n | silent | rules area | learned, withheld | learned, all | rules win |
|---|---|---|---|---|---|---|
| data_digit | 244 | 15 | 0.8285 | 0.9045 | 0.9409 | no |
| field_shift | 245 | 18 | 0.9065 | 0.9534 | 0.9425 | no |
| sign_flip | 250 | 17 | 0.9488 | 0.9813 | 0.9912 | no |
| implausible_eccentricity | 244 | 9 | 0.7858 | 0.9863 | 0.9910 | no |
| **stale_epoch** | 253 | 29 | **0.8601** | **0.6448** | 0.9438 | **yes, +0.2153** |
| **cross_source_divergence** | 243 | 24 | **0.9798** | **0.8335** | 0.9825 | **yes, +0.1463** |

**This is the rules' defensible advantage and it is a large one.** The learned
model's area falls by the margins above when the corruption family is withheld
and recovers when it is restored, so the drop is the unseen-family effect and
nothing else. A real corruption is not drawn from the distribution a model was
trained on. None of the rule screens was fitted to a corruption distribution at
all: each is an explicit statement about what a plausible element set looks
like.

### Confident where the analytic screen withholds its test

| subset | n | errors | error rate | margin above 0.5 | fraction | mean margin |
|---|---|---|---|---|---|---|
| equation (27) resolves the threshold | 142 | 11 | 0.0775 | **0** | 0.0000 | 0.7291 |
| equation (27) does **not** resolve it | 278 | 32 | 0.1151 | **7** | **0.2188** | 0.6403 |

On the subset where the analytic continuity test cannot resolve the declared
threshold, the learned model still answers with a mean decision margin of
0.6403, and 7 of its 32 errors there carry a margin above 0.5 — against 0 of 11
on the resolvable subset. That is the failure mode this batch of the mission
exists to look for: a confident answer about something the data does not
support. The analytic screen's answer on those records is that it cannot tell.

### Uncertainty outputs, on the independent calibration split

| group | n | mean equation (20) margin | mean equation (19) dispersion |
|---|---|---|---|
| correct | 384 | **0.6871** | 0.2827 |
| wrong | 35 | **0.2839** | 0.4211 |

The margin carries information: it is higher on predictions that turned out
right. It is not a probability of being right and nothing here calibrates it to
one.

## 7. How the baseline reached its published strength

The mission's build guide requires an under-strength baseline to be
strengthened and the smaller margin published. The baseline was changed five
times before the comparison above was published, and each change is recorded
with what it was worth, because a reader is entitled to know how much of the
final margin is the problem and how much was the first implementation.

**Two kinds of figure appear below and they are labelled.** The "why" column is
reproducible: `validate_baseline_history.py` reinstates each superseded
behaviour alongside the current one and measures both on the same corpus, and
the field-shift and checksum figures come from the committed outputs of
`validate_rule_screens.py` and `validate_learned.py`. The "measured effect"
column is **not** reproducible from this repository: those are the overall
baseline scores as they stood at each step of the build, measured against code
that is no longer here, and they are marked `[build-time]` for that reason. They
are kept because dropping them would leave the published margin uncheckable,
and they are marked because a reader must not take them for something a script
here will reproduce.

| change | why, reproducible | measured effect, `[build-time]` |
|---|---|---|
| 1. A quantisation floor on every robust scale, at one count of the element's own text field | A history whose mean motion is constant to the eighth written decimal has a scale at the quantiser. On a constructed unjittered history whose candidate sits exactly one count from its trend prediction, the unfloored modified z-score is **2.531364e+06** times the floored one. **The honest other half:** on the corpus this repository ships the floor changes nothing — largest finite z-score **6.6304** either way over 217 uncorrupted element sets — because the generator's declared jitter already keeps the scale above one count. The floor is kept because it follows from the text format and not from a corpus. | `[build-time]` rule area 0.6261 to 0.8086 on the training split; recall at a 0.05 false-positive rate from **0.0000** to 0.4457 |
| 2. The right ascension of the ascending node moved from the level test to the trend test | The node regresses secularly. Both tests run on the same 217 uncorrupted element sets: the level test gives a z-score between **1.3787 and 1.5246**, a spread of **0.1459**, which is a fixed offset carrying no information at all | `[build-time]` included in the figures for change 1 |
| 3. The ephemeris continuity screen added, with its self-calibrated tolerance and equations (24) to (27) | Every earlier screen tested element values against element bands while the label is a position error in kilometres. Nothing translated between the two | `[build-time]` rule area 0.8342 to 0.8431 on the test split; recall at a 0.05 false-positive rate from 0.6387 to 0.7311 at the then-current corpus |
| 4. The checksum indicator removed from equation (15), and integrity made a separate output | The indicator put **31** element sets whose only corruption was the checksum column, with a measured position error of exactly **0.0000 km**, into the silent class. Both figures are in `validate_consequence_output.txt` section 1 | `[build-time]` silent-class precision 0.4245 to 0.4943; accuracy 0.6500 to 0.7214 |
| 5. An unreadable field stopped predicting a loud failure and started setting equation (15) to infinity | The propagator reads a shifted line without complaint far more often than it refuses one. In the committed output of `validate_rule_screens.py`, **14** test-split element sets have a field this package cannot read and **13 of 27** field-shifted ones stay fully readable; the field-shift operating characteristic now stands at **0.9065** | `[build-time]` field-shift area **0.3686, below chance**, to 0.9065; overall rule area 0.7937 to **0.8677**; loud recall fell from 1.0000 to **0.9444**, which is the published cost |

The learned model wins the headline after all five. The `[build-time]` figures
put the margin before the strengthening at about **+0.78 recall** at a fixed
false-positive rate against the **+0.3661** published above, which is less than
half. Publishing the first figure would have overstated the learned contribution
by more than a factor of two, and that is the whole reason this section exists.

## 8. Command line and working directories

`validate_cli.py` runs all ten command line invocations and three error paths in
subprocesses with a temporary `HOME` and `MPLCONFIGDIR` and
`PYTHONDONTWRITEBYTECODE=1`. Every invocation exits 0 and every error path exits
2 with a message and no traceback. The repository contains no `.config`
directory afterwards.

`probe_working_directories.py` runs every one of the 18 scripts in `validation/`
and `examples/` from the filesystem root, the system temporary directory, the
repository root, `validation/` and `src/` — 90 runs — and requires the exit
status and a hash of the complete standard output to be **identical** across all
five. That is a stronger contract than requiring exit zero: a script whose
numbers changed with the working directory would pass an exit-zero check and
fail this one. Timing lines are tagged `[unstable]` by the scripts themselves and
are dropped before hashing, so a timing cannot make a path check pass or fail.

The probe runs at a reduced corpus of 24 objects, which it states in its own
output, and redirects every output file and figure into a temporary directory so
it cannot overwrite the committed full-size ones. Several measurements in
`validate_learned.py` and `validate_rule_screens.py` are population statements
that do not hold on 24 objects, so those two scripts exit 1 under the probe —
identically from all five directories, which is the thing being asserted.

## 9. Compute

Timings, **declared unstable**. They move between runs on a shared machine, no
conclusion in this repository rests on any of them, and no ratio of them appears
in prose anywhere.

The table itself is in `validate_learned_output.txt` section 10, where every
line is tagged `[unstable]`, and it is **not** copied here: a timing changes on
every re-run and a figure quoted in this document would be falsified by the next
one. The three stages measured are screening, measuring and featurising all
three 420-object splits; fitting all three learned configurations; and
predicting the test split. All three are of order seconds and the whole
benchmark is well inside the mission's three-minute budget.

Two CPU cores, no GPU. PyTorch is not installed and nothing here uses a neural
network.

## What is not validated here

- **Any number against flight data.** There is none in this repository and
  there is no catalogue in it either.
- **Whether a screened element set produces an accurate ephemeris.** Not
  determinable here, for the reason the first limit of every report states.
- **The dependency ranges at both ends.** The interpreter is shared and
  pre-pinned and this product may not run `pip`; see `README.md` under
  Limitations. `ruff` is the exception and was measured at both 0.16.9 and
  0.17.3.
- **Behaviour on `sgp4` deep-space element sets specifically.** The corpus
  contains geostationary, navigation and critically inclined
  high-eccentricity objects, which `sgp4` propagates with its deep-space
  path, and they are in every table above; but no check isolates that path or
  tests its boundary at 225 minutes of period.
