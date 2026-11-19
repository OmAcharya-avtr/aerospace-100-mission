# Validation evidence — P099 PassYield

Status: **TESTING**. Research-grade software. Not flight-qualified, not
certified and not approved for operational aerospace use.

Every number in this file, in `README.md`, in `MODEL_CARD.md` and in
`DATASET_CARD.md` comes from a script in this directory that was run in this
session, and each row below names the script and the output file it came from.
The output files are committed. Every script was re-executed after the last
code change and the quoted figures replaced with what the run printed.

Measured on: Python 3.13.18, numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1,
sgp4 2.27, pytest 9.1.1, hypothesis 6.170.15, ruff 0.16.9, matplotlib 3.11.2,
x86_64 Linux, 2 cores shared with four other concurrent builds.

Test suite: **284 tests, 284 passed, 0 failed, 0 skipped, 0 errors**, counted
from the `testsuite` element of the junit XML and cross-checked against a
count of 284 `testcase` elements. Never from pytest stdout: a suite that
collects zero tests prints success and exits 0.

## 1. The geometry, against the propagator's own published reference

`validate_geometry.py` → `validate_geometry_output.txt`

| check | reference | result | tolerance |
|---|---|---|---|
| the two committed TLE lines appear verbatim in the installed `SGP4-VER.TLE` | Vallado, Crawford, Hujsak & Kelso (2006), AIAA 2006-6753, as distributed by the `sgp4` package | both found, character for character | exact string match |
| propagation of catalogue 06251 against the official `tcppver.out` | same paper, 25 reference epochs from 0 to 2880 minutes since epoch | worst position disagreement **7.266e-09 km**, worst velocity **6.894e-10 km/s** | 1e-7, the file's own printed resolution |
| our IAU-1982 GMST polynomial against `sgp4.propagation.gstime` | Vallado (2013) Algorithm 15 | worst disagreement **1.757e-11 rad** over 5 epochs | 1e-9 rad |
| slant range against an independent closed form | Maral, Bousquet & Sun (2020) ch. 2 spherical-Earth circular-orbit expression | median relative disagreement **2.6912e-02**, worst **3.8428e-02** over 40 passes | below 0.10, and a control requires it to be above 1e-9 |
| pass-search resolution margin | the scan cannot find a pass shorter than its 5 s step | shortest window found over 10 days **42.668 s**, which is **8.53 coarse steps** | at least 4 steps |
| cost of the TEME-to-pseudo-Earth-fixed approximation | polar motion under 0.3 arcsec, UT1-UTC under 0.9 s, together under 1e-5 degrees of elevation | worst rate error **2.323e-05 Mbit/s**, which is **9.292e-08** of the 250 Mbit/s ceiling | below 1e-6 of the ceiling |

The closed-form cross-check **must** disagree, and the control that requires
it to disagree by more than 1e-9 fired: a disagreement of 1e-15 would mean
the two expressions were not independent, and a few per cent is what a
spherical Earth and a circular orbit cost against an element set of
eccentricity 0.0030035.

## 2. The link budget

`validate_link.py` → `validate_link_output.txt`

| check | reference | result | tolerance |
|---|---|---|---|
| the Boltzmann term | `k = 1.380649e-23` J/K, exact SI defining constant since 2019 | **228.5991671732 dB** | 1e-9 dB |
| doubling the range costs `20*log10(2)` dB | Friis (1946) | **6.020599913 dB** | 1e-9 dB |
| slant attenuation at 30 degrees is twice the zenith value | ITU-R P.618 secant scaling, `sin(30) = 0.5` | **0.800000000 dB** from `A_z = 0.4` dB | 1e-12 dB |
| 6 dB more figure of merit is 6 dB more `C/N0` | standard Earth-space link equation (Maral et al. 2020 ch. 5; Ippolito 2017 ch. 4) | **6.000000000 dB** | 1e-9 dB |
| `R = B * eta` | Shannon (1948) capacity with an implementation margin | **124.005868316 Mbit/s** both ways at 30 degrees, G/T 21 dB/K | 1e-12 Mbit/s |
| the rate varies across a pass | the whole point of the geometric term | smallest spread `R(90)/R(mask)` = **3.163**, largest **5.629** across the eight stations | at least 2.0 |
| the rate is non-decreasing in elevation | slant range and attenuation both fall | most negative step **0.000e+00 Mbit/s** over 2001 points per station | at least -1e-12 |
| the validity range is enforced, not extrapolated | ITU-R P.618 states the secant law above about 10 degrees | **7 of 7** refusal controls fired at -1, 0, 5, 9.999, 90.001 and 180 degrees; the model accepts exactly 10 | both directions asserted |

The largest saturated fraction of an elevation sweep is **0.2743**, at the
best station in the catalogue. That is not zero, and it matters: the
`min(eta_max, ...)` cap makes the rate profile only C0 at the saturation
boundary, which is the mechanism behind the decomposition residual below.

## 3. The decomposition identity and its reported residual

`validate_decomposition.py` → `validate_decomposition_output.txt`

Re-derived from the committed event timelines over all **1990** passes, not
read from the record's own decomposition columns.

| quantity | result |
|---|---|
| median `|rho|` | **2.125497e-04 Mbit** |
| 95th percentile `|rho|` | **4.364125e-03 Mbit** |
| worst `|rho|` | **5.065014e+00 Mbit** |
| median `|rho| / |S|` | **1.463911e-08** |
| 95th percentile `|rho| / |S|` | **3.888794e-07** |
| worst `|rho| / |S|` | **8.705712e-04** |
| largest mask clamp applied | **5.108738e-06 deg** against a declared 1e-4 tolerance |
| passes with acquisition past mask set | **44** of 1990 |

**THE REPORTED RESIDUAL OF THIS PRODUCT IS 3.888794e-07 relative**, the 95th
percentile. The test suite's identity tolerance is 1.0e-5, that figure rounded
up by more than an order of magnitude.

**Where the residual comes from, measured and not asserted.** Stratifying by
peak elevation at the 65 degree saturation threshold: mean `|rho|` is
**1.208408e-01 Mbit** on the 404 passes above it and **5.486052e-04 Mbit** on
the 1586 below, a ratio of **220.3**. The kink is the mechanism. The
corroborating check: on the worst pass the measured fourth derivative of the
rate profile is 2.068936e+02 Mbit/s per s^4, giving a composite-Simpson bound
of 8.484974e+06 Mbit, and the measured residual of 5.065014e+00 Mbit is
inside it — but that bound is only valid on the smooth part of the profile,
which is why the stratification is the primary evidence.

**Grid refinement on the worst pass:**

| nodes/interval | nodes/window | `rho`, Mbit | `rho/S` |
|---|---|---|---|
| 9 | 65 | 9.492440e+01 | 1.654e-03 |
| 33 | 513 | 5.065014e+00 | 8.828e-05 |
| 129 | 2049 | 1.194297e-01 | 2.082e-06 |
| 513 | 8193 | -5.888118e-02 | -1.026e-06 |

Asserted as a direction only. The reduction factors are 18.74, 42.41 and
2.03, which is not the C4 order of Simpson's rule and should not be: the
kink's position relative to the grid changes with every refinement.

**The three documented edge cases, all exercised:**

| case | behaviour | evidence |
|---|---|---|
| a pass that never rises above the mask | `find_passes` returns an empty list; `decompose_pass` on a sub-mask window raises `ValueError` naming the peak elevation and the mask | 0 windows for a station at 85 N against a 58.06 degree inclination; control fired |
| a zero-duration pass | `decompose_pass` raises `ValueError`: every ratio the product reports would be undefined | control fired |
| an acquisition time exceeding the whole pass | **returns**, because it is an operational outcome and not a malformed input: the interval is clipped, `acquisition_truncated` is True, the realised yield is **exactly 0.0 Mbit**, and the acquisition term carries the whole capacity to within 2.017e-04 relative | **44 of the 1990 committed passes are of this kind**; control fired |

No path produces a silent negative yield. The record loader refuses a
negative `realised_mbit` with that stated as the reason.

## 4. The convolution: two methods and one independent bound

`validate_convolution.py` → `validate_convolution_output.txt`

**Numerical, not closed form.** The cumulative volume is an SGP4-driven
integral with no elementary inverse, so neither pushforward has an elementary
density and the compound Poisson sum has no elementary transform. The work is
a `numpy.convolve` on a uniform grid of data volume in Mbit.

**Discretisation, measured.** On the reference pass the median quantile moves
by 302.6241, 144.2426 and 87.6921 Mbit as the grid goes 256 → 512 → 1024 →
2048 bins, an implied convergence order of 0.894 in the step. First order is
what a mass-preserving binned pushforward gives.

**Against a direct Monte Carlo of the same laws**, 200000 draws, 24 passes
spread across the catalogue, 7 quantile levels each:

| quantity | result |
|---|---|
| quantile comparisons | **161** |
| outside the derived tolerance | **0** |
| worst `|difference| / tolerance` | **0.2607** |
| worst `|difference|` in grid steps | **7.06** |

The tolerance is derived, not tuned: a discrete convolution adds **bin
indices**, not values, so each stage can displace the result by up to one grid
step. The stages on a pass are one for acquisition, one for idle, and one per
outage in the compound sum truncated at its 1-in-1e9 quantile, so the term is
`(1 + stages)` grid steps plus three Monte Carlo quantile standard errors.

**The structural approximation, measured.** The convolution treats the three
losses as independent and additive. They are not — inside one window the
intervals must be disjoint — so any probability either method puts above the
window capacity is probability the assumption invented:

| quantity | convolution | Monte Carlo |
|---|---|---|
| median truncated mass | **2.072740e-02** | **2.065000e-02** |
| worst truncated mass | **6.260082e-01** | **6.234050e-01** |

Worst disagreement between the two: **2.603172e-03**. On the worst pass, 63
per cent of baseline 2's own law is physically impossible. **That is published
as a limitation of baseline 2**, and it is why baseline 2 is not ground truth.

**One bound from mathematics neither method uses.** `0 <= Y_real <= R_max T`
gives `Y_nom - R_max T <= S <= Y_nom` path by path, from the pass duration and
the 250 Mbit/s waveform ceiling alone: no quadrature of the rate profile, no
pushforward, no convolution, no random numbers. **394 quantiles and accounted
values bracketed, 0 outside.**

## 5. The brackets over the whole campaign

`validate_bounds.py` → `validate_bounds_output.txt`

| check | result |
|---|---|
| accounted shortfalls outside the shortfall bracket, 1990 passes | **0** |
| Simpson capacities outside the mean-value bracket, 1990 passes | **0** |
| median shortfall-bracket width | **85441.9 Mbit** |
| median slack | **1.518e-10 Mbit**, which is **1.776e-15** of the width |
| where the true shortfall sits inside the bracket | minimum **0.3649**, median **0.7390**, maximum **1.0000** of the width |
| negative controls that must fire | **7 of 7 fired** |
| controls that must stay silent | **2 of 2 silent** |

The controls are anchored on the true value and constructed to be outside by
the bracket's own algebra, not by offsetting an output that already sits
inside the band: a Mbit-reported-as-bit factor of 1e6, a shortfall exceeding
the booked volume, a negative realised yield, a sign flip on the capacity, a
capacity above `R_max T` and a capacity below `R_min T`.

**The slack follows the conditioning.** `8 * eps * max(|operands|)`, measured
at operand magnitudes 1, 1e3, 1e6, 1e9 and 1e12: the ratio is
**1.776357e-15** at every magnitude, a ratio spread of **0.000e+00**. A slack
floored at an absolute 1.8e-15 would be vacuous at magnitude 1e12 against the
conditioning-following 1.776e-03, and that control fired.

## 6. The committed record

`validate_record.py` → `validate_record_output.txt`

| check | result |
|---|---|
| the record's SHA-256 against `data/manifest.json` | **match** |
| size | **754690 bytes**, 0.7197 of 1 MB, so it is committed |
| rows against the manifest | **1990** |
| every decomposition column recomputed from the committed event timeline | **0 of 1990 passes outside tolerance** |
| worst absolute difference, any column | **1.825e-04 Mbit** on `nominal_mbit`, against a derived bound of 1e-3 Mbit, a margin of **5.48x** |
| worst relative difference over columns of at least 1 Mbit | **1.881e-05** on `shortfall_mbit`, which is a difference of two nine-digit columns and conditions worse than either |
| the stored columns close the identity | worst **1.211e-04 Mbit** against a derived printing bound of **1.915e-04 Mbit** |
| required files excluded by the product's own `.gitignore` | **0** |
| scratch-file patterns that must ignore | **6 of 6 controls fired** |

The closure bound is absolute and derived: each column is written at nine
significant digits, so six columns of magnitude up to 6.4e4 Mbit can close to
`6 * 0.5e-9 * 6.4e4`. A relative bound would be the wrong shape, because the
identity's left-hand side is the shortfall and on some passes that is two
orders of magnitude smaller than the largest column.

## 7. The learned component against the baselines

`validate_learned.py` → `validate_learned_output.txt`

Baseline 3 over the whole campaign, bootstrapped by **pass**, 2000 resamples,
95 per cent:

| quantity | value |
|---|---|
| passes | **1990** |
| campaign booked (baseline 1) | **74152404.0 Mbit** |
| campaign capacity | **63654338.3 Mbit** |
| campaign realised | **45322772.6 Mbit** |
| campaign shortfall | **28829631.5 Mbit** |
| mean shortfall per pass | **14487.252 Mbit**, 95% bootstrap interval **[14122.511, 14888.071]** |
| realised / capacity | **0.712014**, 95% bootstrap interval **[0.702095, 0.721268]** |
| geometry share of the shortfall | **36.4142 %** |
| acquisition share | **23.2756 %** |
| re-sync share | **18.1909 %** |
| idle share | **22.1193 %** |
| residual share (reported) | **1.369e-05 %** |

**A schedule built on baseline 1 books 1.64 times the volume the record
accounts for.**

### Held-out stations (GS-03 and GS-08, declared before any measurement)

1443 training passes, 547 held out.

| predictor | CRPS, Mbit | MAE, Mbit | PIT cal err | cov80 |
|---|---|---|---|---|
| baseline 1 booked | 17732.36 | 17732.36 | 1.0000 | 0.0000 |
| null constant fraction | 4891.64 | 6859.17 | **0.1961** | 0.7185 |
| baseline 2 convolution | 7014.63 | 7831.87 | 0.6709 | 0.2194 |
| baseline 2b backlog-aware | 4184.76 | 5271.86 | 0.3601 | 0.3455 |
| learned tree-spread | 3203.30 | 4110.59 | 0.2455 | 0.5558 |
| **learned residual (declared)** | **3086.86** | 4140.69 | 0.2410 | 0.6344 |

### Held-out time (last 25 per cent of the campaign, chronologically)

1492 training passes, 498 held out.

| predictor | CRPS, Mbit | MAE, Mbit | PIT cal err | cov80 |
|---|---|---|---|---|
| baseline 1 booked | 14380.29 | 14380.29 | 1.0000 | 0.0000 |
| null constant fraction | 3685.53 | 5839.60 | 0.0483 | 0.7932 |
| baseline 2 convolution | 4102.21 | 4808.50 | 0.5341 | 0.3534 |
| baseline 2b backlog-aware | 3892.15 | 4908.58 | 0.2590 | 0.4237 |
| **learned tree-spread** | **1682.22** | 2284.37 | 0.1300 | 0.6165 |
| learned residual (declared) | 1701.40 | 2326.78 | **0.0345** | **0.8052** |

### The gap between the splits

| predictor | CRPS stations | CRPS time | ratio | PIT stations | PIT time |
|---|---|---|---|---|---|
| baseline 1 booked | 17732.36 | 14380.29 | 1.23 | 1.0000 | 1.0000 |
| null constant fraction | 4891.64 | 3685.53 | 1.33 | 0.1961 | 0.0483 |
| baseline 2 convolution | 7014.63 | 4102.21 | 1.71 | 0.6709 | 0.5341 |
| baseline 2b backlog-aware | 4184.76 | 3892.15 | 1.08 | 0.3601 | 0.2590 |
| learned tree-spread | 3203.30 | 1682.22 | 1.90 | 0.2455 | 0.1300 |
| learned residual (declared) | 3086.86 | 1701.40 | 1.81 | 0.2410 | 0.0345 |

The declared learned configuration costs **1.81 times** the CRPS on a station
it never trained on, and its calibration error is **7.0 times** worse there.

### Findings, with nothing retuned away

1. **Baseline 1 is refuted on both splits.** Its CRPS is 5.7 times the
   learned model's on held-out stations and 8.5 times on held-out time, and
   its 80 per cent interval covers nothing, because it is a point.
2. **The specified baseline 2 is beaten by a trivial null on both splits.**
   The constant-fraction null, which knows nothing but the training mean of
   the realised fraction, has a lower CRPS than baseline 2 on both. The reason
   is diagnosable: baseline 2's idle law is unconditional, while the idle time
   in the record is driven by the onboard backlog, which is a pre-pass
   quantity. **Baseline 2b**, the same convolution with the backlog-limited
   dry time in place of the fitted idle law, is the better analytic form on
   both splits and beats the null on held-out stations (4184.76 against
   4891.64) while losing to it on held-out time (3892.15 against 3685.53). It
   is reported beside the specified form and does not replace it.
3. **The learned forest has the lowest CRPS on both splits.**
4. **On the harder split, the trivial null is better calibrated than the
   learned model** (PIT 0.1961 against 0.2410; cov80 0.7185 against 0.6344
   for a nominal 0.80). Calibration is what a scheduler needs in order to
   book a quantile, so that is a baseline result on the dimension that
   matters most and it is published as one.
5. The declared uncertainty output, the out-of-bag residual law, is better
   calibrated than the across-tree spread on both splits (0.2410 against
   0.2455, and 0.0345 against 0.1300). On held-out time it reaches cov80
   **0.8052** against a nominal 0.80, which is the one place in this product
   where the learned uncertainty output is usable as it stands.

## 8. Hidden-tolerance audit

`validate_approx_audit.py` → `validate_approx_audit_output.txt`

`pytest.approx(expected, rel=r)` passes if either bound is met and the
absolute default is 1e-12, so for a quantity of order 1e-5 every relative
tolerance tighter than 2e-8 is silently replaced.

| measurement | result |
|---|---|
| test files scanned | 13 |
| `approx` calls found, from the syntax tree | **80** |
| `rel=` and `abs=` both given | **80** |
| `rel=` only — the defect | **0** |
| the whole suite re-run with a plugin that makes a bare `rel=` raise | **exit 0** |
| **assertions depending on the hidden default** | **0** |

## 9. Tree hygiene

`validate_tree_hygiene.py` → `validate_tree_hygiene_output.txt`

This check asserts what the product owns and deliberately not the third
thing: that running a script **adds** no cache directory or bytecode file, and
that none **would reach a cold clone** — not that none exists, because the
release gate's own `ruff` run writes `.ruff_cache` into the tree one check
before the validation scripts are executed.

| check | result |
|---|---|
| artifacts inherited at entry, reported and not failed on | **29** |
| artifacts added by running a validation script and an example in a pristine copy | **0** |
| runnable scripts setting `sys.dont_write_bytecode` first | **16 of 16** |
| artifacts a cold clone would carry | **0** |
| the scanner run twice from the same tree | **agrees**, 29 and 29, difference empty |
| committed evidence regenerated by this script | **0 files** |

## 10. Working-directory probe

`probe_working_directories.py` → `probe_working_directories_output.txt`

Every script under `validation/` and `examples/` run from the filesystem
root, the system temporary directory, the product root and the script's own
directory, **inside a pristine copy** so that no committed output file is
touched.

| quantity | result |
|---|---|
| validation scripts probed | **10** |
| examples probed | **4** |
| scripts returning the same exit status from all four directories | **14 of 14** |
| scripts exiting 0 from all four directories | **14 of 14** |
| committed output files changed by the probe, excluding its own | **0** |

## 11. Checks that did not pass first time

Recorded because a validation file that reports only successes is not
evidence.

- The acquisition-truncation edge case was first checked against a 1e-5
  relative tolerance and **failed at 2.017e-04** on the worst pass of the
  campaign. The tolerance was wrong, not the code: the acquisition term and
  the capacity cover the same interval with different quadratures, so their
  difference is the residual class measured in section 3, and the bound used
  is now that section's measured worst case.
- The convolution-versus-Monte-Carlo comparison first used a tolerance of
  three grid steps and **28 of 161 comparisons failed**, by factors of 1.2 to
  1.6. The derivation was wrong, not the code: a discrete convolution adds bin
  indices, so the error accumulates over the convolution stages. With the
  stage count derived from the Poisson truncation point, 0 of 161 fail and the
  worst disagreement is 7.06 grid steps.
- The conditioning demonstration in section 5 first divided the slack by the
  smaller of its two operands and reported a ratio spread of 1.0. The slack is
  defined against the largest operand; dividing by the right one gives a
  spread of 0.
- The `.gitignore` re-implementation first matched a trailing-slash directory
  pattern against the parent components only, so it reported that
  `.hypothesis` and `src/passyield/__pycache__` would reach a cold clone when
  they would not. One implementation now serves both scripts that need it.
- A line-based scan for the bytecode flag read a docstring sentence beginning
  "from that paper:" as an import and failed a script that was correct. It
  parses the syntax tree now.
