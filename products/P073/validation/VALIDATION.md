# Validation evidence

Level 2. Every number in this document was produced by a script in this
directory, executed in the session this repository was built. Each script writes
its raw output beside itself as `<script>_output.txt`, and the header of every
output file records the exact package versions and core count the numbers were
measured on.

Nothing here is carried from another product, estimated, or quoted from a
reference work. Where a check is against an external reference, the reference is
named and its presence in the environment is recorded; where there is no
external reference, the script says so and states what it is comparing instead.

## Environment

Measured on Python 3.13.17 with `numpy` 2.5.3, `scipy` 1.18.1, `scikit-learn`
1.9.1, `sgp4` 2.27, `pandas` 3.0.5, `matplotlib` 3.10.8, `pytest` 8.4.2,
`hypothesis` 6.140.3, `ruff` 0.14.2, on two cores. The suite was additionally run
on the system interpreter of the same container, which carries `pytest` 9.1.1,
`hypothesis` 6.170.9, `matplotlib` 3.11.2 and `ruff` 0.17.2 with identical
`numpy`, `scipy`, `scikit-learn` and `sgp4`; 611 tests pass on both and
`ruff check src/ tests/` exits 0 on both.

**`astropy` is not installed.** It is named in `README.md` as the mature
alternative for frames and time scales, and no number anywhere in this
repository is a cross-check against it. The independent geometric check in
`validate_visibility.py` is a second derivation written in this package, not a
third-party reference.

**`pulp` is installed and ships no solver**, and OR-Tools is not installed. That
measurement is the reason the exact baseline runs on HiGHS; see section 4.

## Test count

Read from the junit XML, never from pytest stdout. A suite that collects zero
tests prints success and exits 0.

```
tests=611  failures=0  errors=0  skipped=0
literal <testcase> element count: 611
```

The declared attribute and the literal element count agree.

## 1. Geometry, sidereal time and the independently derived bound

`validate_geometry.py` -> `validate_geometry_output.txt`

| Check | Result | Tolerance |
| --- | --- | --- |
| WGS-84 constants against their closed forms | agree | 1e-10 relative |
| Maximum ellipsoid-normal-to-radial deviation, closed form against a brute-force latitude scan at 1e-4 deg | 0.192424301 deg both, difference 2.0e-14 deg | 1e-11 deg |
| GMST against `sgp4.propagation.gstime` over 120 years of span | worst 2.211e-11 rad, 1.267e-09 deg of Earth rotation | 1e-8 rad |
| Geocentric elevation from three lengths against the vector dot product, 40000 samples | worst 2.615e-12 deg | 1e-6 deg |
| Geodetic against geocentric elevation of the same slant vector, 36000 samples over nine latitudes | worst 0.192424299 deg against a derived bound of 0.192424301 deg | the bound, margin 1.682e-09 deg |
| Radial point above an equatorial station | 90.000000000 deg | 1e-9 deg |
| Due-east tangential point | elevation 0.000000000 deg, azimuth 90.000000000 deg, range 500.000000000 km | 1e-12 |

The fifth row is the check the build-guide defect class calls for: a computed
quantity against an **independently derived bound**, not against a second
implementation of the same expression. The bound comes from the WGS-84
ellipsoid and shares no expression with either elevation computation. It earned
its place: a hard-coded `MAX_GEODETIC_GEOCENTRIC_DEVIATION_DEG = 0.19242218`
stood in `src/contactsched/geometry.py` until this check failed by 1.0e-06 deg,
and the closed form is 0.19242430115956036.

**A result that contradicted the prose written before it.** The law-of-cosines
elevation was expected to lose precision near the horizon, where its numerator
is a difference of nearly equal large numbers. Measured, the disagreement is
largest near the **zenith**: the cancellation leaves an absolute error in
`sin(elevation)` of order `eps * r_station / (2 * range)`, which is nearly flat
in elevation, and the conversion through `arcsin` multiplies it by
`1 / cos(elevation)`, which is 1 at the horizon and 57 at 89 degrees. The
measured maxima per band are 2.869e-14 deg below 0.1 deg elevation and
2.615e-12 deg above 60 deg. The script says so in place of the original claim.

## 2. Visibility windows

`validate_visibility.py` -> `validate_visibility_output.txt`

24-hour horizon, four antennas, six spacecraft, 165 windows.

| Check | Reference | Result | Tolerance |
| --- | --- | --- | --- |
| Elevation against an independent derivation | spherical law of cosines on geocentric latitude and longitude | worst 8.086e-12 deg over 57624 samples | 1e-9 deg |
| Window boundaries | dense one-second scan of the same margin function | worst 0.5859 s, zero count mismatches | 0.75 s |
| Window durations | closed-form spherical upper bound on the TLE mean motion | 0 of 22 station-spacecraft pairs exceed it; ratios 0.758 to 0.924 | the bound |
| Union antenna supply | interval merge | overlap 0.0 s at KOUROU to 4694.5 s at SVALBARD-1; union never above the sum | 1e-6 s |

The second row checks the **root search** and not the geometry: both sides use
the same elevation and the same mask. The first and third rows check the
geometry, by two different routes.

## 3. Convergence under refinement

`validate_convergence.py` -> `validate_convergence_output.txt`

The scan step and the bisection tolerance are refined separately, because they
control different errors: the step decides whether a pass is found, the
tolerance decides how precisely a found boundary is located.

- **Scan step.** Against the 2 s reference, a 240 s step loses 2 boundaries on
  one of the three cases measured. Steps of 30, 45, 60 and 120 s lose none on
  any case. The defaults are 30 s in `compute_windows` and 45 s in
  `generate_instance`, both in that list. No general claim is made: any sampled
  root search can miss a pass shorter than its step, and the loss at 240 s is
  exactly that.
- **Bisection tolerance.** At a fixed 30 s scan, every boundary shift against
  the 0.001 s reference is between 0.2884 and 0.4678 of its own declared
  tolerance, so the bisection delivers what it promises.
- **Total visible seconds.** 6138.750 s at the coarsest setting against
  6283.314 s at the finest, converging to within 0.084 s at a 10 s step.

## 4. The exact baseline

`validate_exact.py` -> `validate_exact_output.txt`

### The measurement that moved the baseline off PuLP

The product specification named a PuLP mixed-integer program. In the environment
this product was built and measured in:

```
pulp.__version__                        : 4.1.0
pulp.listSolvers(onlyAvailable=True)    : []
'PULP_CBC_CMD' in pulp.listSolvers()    : False
len(pulp.listSolvers())                 : 27
```

An empty availability list means no solver is installed, and `PULP_CBC_CMD` is
absent from the full solver list as well, CBC having been unbundled in the 4.x
series. A PuLP-backed exact baseline would raise at solve time rather than
return a schedule, here and on the release gate. The baseline therefore runs on
`scipy.optimize.milp`, which dispatches to HiGHS and is part of a dependency the
package already needs. `pulp` is **not** a dependency of this package and is
named in `README.md` among the alternatives together with this measurement.
Reproduce it with `python -m contactsched backends`.

An unknown backend name raises rather than falling back silently, which the
script also exercises.

### The optimum against exhaustive enumeration

280 reduced instances were scanned and 140 kept, those whose candidate count
lands between 8 and 18. Each was solved by the clique mixed-integer program on
HiGHS and by a literal `2 ** n` bitmask enumeration that shares no part of that
formulation, under four configurations.

| Configuration | Instances | Agreements | Worst difference |
| --- | --- | --- | --- |
| per-contact objective, floors off | 140 | 140 | 0.000e+00 |
| per-second objective, floors off | 140 | 140 | 0.000e+00 |
| per-contact objective, floors on | 140 | 140 | 0.000e+00 |
| per-second objective, floors on | 140 | 140 | 0.000e+00 |

6964736 subsets were enumerated in total. Both methods returned feasible
schedules on every instance: zero feasibility violations.

Wall clock for the whole comparison, declared unstable and used in no
conclusion: of order 8 s for 560 enumerations and of order 1.3 s for 560 HiGHS
solves on a shared two-core machine. The committed
`validate_exact_output.txt` records the exact figures of the run that produced
it; no ratio of them belongs in a sentence.

### What the candidate grid costs

The optimum is exact over the discretised grid and not over continuous
placement. Refining the grid from 600 s to 60 s on five full-size instances
changed the optimum by 0.0000 in every case; the optimum never fell, which it
cannot, since 60 divides 120, 300 and 600 and every coarse candidate is also a
fine one.

## 5. Invariants and negative controls

`validate_invariants.py` -> `validate_invariants_output.txt`

120 instances, 118 usable.

| Check | Result |
| --- | --- |
| Feasibility violations, three methods, 118 instances each | 0 |
| Requests short in the 16 feasible fairness-constrained schedules | 0 |
| Allocated antenna time above union visibility | 0 breaches |
| Contacts per request above standalone capacity | 0 breaches |
| Objective above the per-request weight ceiling | 0 breaches |
| Fairness metrics outside their declared ranges | 0 breaches |
| Selection mismatches over three repeated runs, 40 instances, three methods | 0 |

### Negative controls

Eight controls on a hand-built instance: one antenna with a 200 s turnaround and
a single 2000 s window, so every corruption stays inside the window and the
check under test is the only one that can fire.

| Control | Fired | Expected |
| --- | --- | --- |
| positive control: 800 s apart against a 200 s turnaround | False | False |
| separation exactly one turnaround, 200 s | False | False |
| separation one second inside the turnaround, 199 s | True | True |
| overlap: second contact starts 1 s after the first ends | True | True |
| contact relocated outside every window | True | True |
| the same candidate selected twice | True | True |
| three compatible contacts against a ceiling of two | True | True |
| one spacecraft on two antennas at once | True | True |

The two rows expecting **False** are what make the other six meaningful. A
checker that fires on everything is not a checker, and the boundary is tested
from both sides because an offset that merely reaches a bound does not cross it.

Starvation is deliberately *not* a feasibility violation: a schedule that misses
a declared minimum is feasible and short, and treating the miss as infeasibility
would throw away the number this product exists to report.

## 6. Fairness

`validate_fairness.py` -> `validate_fairness_output.txt`

- **Known answers.** Eight hand-calculated allocations, including the all-zero
  case where Jain's index returns its degenerate 1.0 and the max-min share
  returns 0.0, and a single denormal share of 5e-324 which is a pinned
  Hypothesis counterexample. All eight agree to 1e-12.
- **Behaviour under priority inversion.** Reversing the declared priority order
  on 24 instances, leaving the contention structure untouched, changed the
  schedule on **24 of 24**. Mean inversions 4.2917 before and 3.9583 after; the
  aggregate promise held on 9 instances before and 9 after. The count is not
  expected to fall and the script reports what happened rather than asserting a
  direction. A reversal that changed no schedule would mean the priority weights
  never reach the solver, and the script fails in that case.
- **A constructed inversion.** A priority-1 request declaring three contacts
  with one usable pass, beside a priority-4 request declaring one with a
  disjoint pass: 1 inversion of 1 comparable pair, gap exactly 0.666667, and the
  aggregate promise fails. No scheduler could have avoided it, which is the
  useful content.
- **A constructed contested instance.** Where the two requests do contest one
  antenna, the higher priority wins under both objective modes, 1.0 against
  0.375 per contact and 900 against 112.5 weighted seconds. The declared
  geometric weights do what they say; the inversion above comes from the
  geometry and from differing declared need.

### The price of the fairness constraints, measured

| Quantity | Value |
| --- | --- |
| instances where the constrained program is feasible | 16 of 160 |
| mean objective loss on those 16 | 0.000000 |
| worst objective loss on those 16 | 0.000000 |
| mean Jain gain | 0.002511 |
| mean max-min gain | 0.046875 |

**The measured price is zero**, which is not what the prose written before the
measurement said. Where the declared floors can be met at all, they are met by
an alternative optimal schedule rather than by giving up objective, because the
per-contact objective is indifferent between placements of the same request. A
non-zero price needs a specific structure, and one is constructed rather than
claimed: three slots on one antenna, two of them visible to a second spacecraft,
costs exactly 1.000000 of a 3.000000 free optimum, one third, and raises the
max-min share from 0.000000 to 1.000000.
`examples/fairness_metrics.py` pays 0.375 of 18.500 on the renegotiated seed-7
instance.

**144 of 160 instances are infeasible under their own declared floors.** That is
the headline finding and it is not a solver failure: on those instances at least
one request declares more contacts than its own passes can carry. The actionable
output is the diagnostic, which names the request and its standalone capacity.

## 7. Baseline against baseline

`validate_baselines.py` -> `validate_baselines_output.txt`

160 instances. Gap of priority-ordered first-fit against the proven optimum,
per-contact objective:

| Statistic | Value |
| --- | --- |
| mean | 0.029962 |
| median | 0.013007 |
| p75 | 0.045665 |
| p90 | 0.096503 |
| p99 | 0.142725 |
| maximum | 0.176871 |
| exact ties | 75 of 160 |
| tie fraction | 0.468750 |

The distribution is reported in bands, not summarised, because 46.9 per cent
ties and a tail to 0.1769 are both the result and a mean of 0.0300 hides both.

Head to head between the two heuristic orderings over 160 instances: priority
order strictly better on 78, demand order strictly better on 31, tied on 51.
Neither dominates, so both are shipped.

Beyond the objective, per-instance means:

| Method | Objective | Starved | Jain | Max-min | Inversions | Instances with a schedule |
| --- | --- | --- | --- | --- | --- | --- |
| greedy[priority] | 10.755469 | 3.0438 | 0.779122 | 0.133646 | 4.9688 | 160 |
| greedy[demand] | 10.257031 | 3.1875 | 0.774858 | 0.126771 | 6.9062 | 160 |
| exact | 11.115625 | 2.8062 | 0.803596 | 0.171458 | 4.7812 | 160 |
| exact + fairness floors | 10.437500 | 0.0000 | 1.000000 | 1.000000 | 0.0000 | 16 |

The last row is over a biased subset -- the 16 instances whose floors are
satisfiable at all -- and must not be compared with the rows above it.

## 8. Dataset

`validate_dataset.py` -> `validate_dataset_output.txt`

| Check | Result |
| --- | --- |
| Instance determinism over 40 re-generations | 0 mismatches |
| Dataframe rebuild over 20 seeds | byte-identical |
| Rows over 320 instances | 2903, positive rate 0.308302 |
| Build wall clock, unstable, 2 cores | 22.9 to 28.6 s, 71.5 to 89.2 ms per instance, observed this session |
| Shared `(seed, request_id)` keys between train and test | 0 |
| Shared seeds | 0 |
| Overlapping seed ranges | raise |
| Features containing NaN | 0 |

The compute budget check is against this mission's three-minute limit and the
script fails if the build exceeds 180 s.

## 9. The learned conflict predictor

`validate_predictors.py` -> `validate_predictors_output.txt`

240 training instances, 80 test instances, disjoint seed ranges, split by
instance. 2178 training rows and 725 test rows. Full tables are in
`MODEL_CARD.md`; the result in brief:

| Metric | Best baseline | Baseline | Learned | Margin |
| --- | --- | --- | --- | --- |
| area under the ROC curve | `ranking[demand_over_usable]` | 0.922952 | 0.984596 | +0.061644 |
| average precision | `ranking[demand_over_usable]` | 0.885048 | 0.969138 | +0.084090 |
| Brier score | `ranking[demand_over_usable]` | 0.098911 | 0.046725 | +0.052185 |
| expected calibration error | `logistic[demand_over_usable]` | 0.012237 | 0.027302 | **-0.015065** |
| F1 at the training-chosen threshold | `ranking[demand_over_usable]` | 0.733728 | 0.886256 | +0.152528 |
| area under the ROC curve, hard subset | `ranking[ratio_weighted]` | 0.838021 | 0.952907 | +0.114887 |

**The baseline wins on calibration** and the result is published unchanged.

Two further findings that bear on the comparison's honesty:

1. The contention ratio the specification names -- demand over supply **per
   station-window** -- is the weakest of the three available, at 0.6010 and
   0.6228 area under the ROC curve against 0.9230 for the per-request ratio, and
   on a smaller 108-row split both per-bin ratios fall below 0.5. The strongest
   is therefore the baseline. Taking the specified one would have inflated the
   learned margin.
2. The learned model is compared against the best baseline **per metric**, not
   against a single nominated baseline, because the ranking and logistic forms
   of the same feature tie on rank statistics and differ on Brier score.

## 10. The `pytest.approx` audit

`validate_approx_audit.py` -> `validate_approx_audit_output.txt`

`pytest.approx(expected, rel=r)` carries a default absolute tolerance of 1e-12
and passes on either bound, so for a quantity of order 1e-5 every relative
tolerance tighter than 2e-8 is silently replaced.

| Measurement | Value |
| --- | --- |
| `pytest.approx` calls in the suite | 276 |
| calls passing `rel=` | 269 |
| calls passing both `rel=` and `abs=` | 269 |
| **calls relying on the 1e-12 default** | **0** |
| calls with a non-zero `rel=`, re-run with `abs` forced to 0.0 | 44 |
| of those, assertions that then failed | 0 |
| calls written `rel=0.0` with an explicit absolute bound | 225 |

No assertion in this suite depends on the default. Every assertion that declares
a non-zero relative tolerance passes on that tolerance alone with no absolute
bound whatever. The 225 pure-absolute assertions compare angles, times and
quantities whose expected value is exactly zero, where a relative bound is
identically zero and bit-exact equality is what this mission's build guide
forbids asserting.

The script also runs an unmodified **control** copy, so that any failure can be
attributed to the rewrite rather than to the act of copying the tests out of the
repository, and demonstrates the trap on four lines that can be run by hand.

## 11. Working directories and absolute paths

`probe_working_directories.py` -> `probe_working_directories_output.txt`

Every `.py` file in `validation/` and `examples/`, excluding support modules and
the probe itself, is run from the filesystem root, the system temporary
directory, the repository root and its own directory.

| Mode | Scripts | Result |
| --- | --- | --- |
| module body, all four directories | 19 | every exit status 0 |
| full end to end, all four directories | 8 fast scripts | every exit status 0 |
| repository-wide absolute-path scan | 77 text files | 0 offending lines |

The exhaustive form, which runs every script end to end from all four
directories, is available as `CONTACTSCHED_PROBE_FULL=1` and was executed in the
build session. Every script exited 0 from every directory in that form as well,
including `validate_predictors.py`, `validate_approx_audit.py`,
`validate_exact.py`, `validate_baselines.py`, `validate_fairness.py`,
`validate_invariants.py`, `validate_dataset.py`, `validate_cli.py`,
`examples/priority_inversion.py`, `examples/greedy_optimality_gap.py` and
`examples/predictor_benchmark.py`, which the default mode covers by module body
only because the exhaustive form takes several minutes.

The probe found three defects on its first three runs, all recorded in
`README.md`. `runpy.run_path` does not put a script's own directory on
`sys.path` the way `python script.py` does, so the module-body mode failed from
every directory. The absolute-path scan then reported its own pattern literals,
and on the next run its own previous output. Finally it reported
`.github/workflows/tests.yml`, whose CI step forbidding absolute paths contained
one in its own grep pattern. All three are fixed by assembling path prefixes
from a separator rather than writing them out, and by deleting the probe's
output file immediately before the scan.

## 12. Every quoted command, re-run

`validate_cli.py` -> `validate_cli_output.txt`

Twelve commands, each run as a subprocess from the repository root with
`PYTHONPATH=src` and nothing else inherited, which is how the release gate runs
them. Every exit status matched its declared value, including the deliberate 2
for an instance with no placeable candidate, and no command printed an absolute
path.

`worked_example.py` -> `worked_example_output.txt` holds the full transcript of
the example quoted in `README.md`, including the renegotiation loop.

## What is not validated

- There is no comparison against any real ground-station schedule, because no
  such data is available here. Everything is measured against the declared
  model.
- There is no comparison against `astropy`, `pulp` or OR-Tools, because the
  first and third are not installed and the second ships no solver.
- The pass-duration bound is loose by construction and is not a model of pass
  duration; it is a falsifier.
- No statement is made about accuracy against a real sky. UT1 against UTC, polar
  motion and the equation of the equinoxes are omitted, and the resulting
  look-angle error is stated in `README.md` under Limitations rather than
  measured, because measuring it would require an Earth-orientation model this
  package does not have.

This software is research-grade. It is not flight-qualified, not certified, and
not approved for operational aerospace use.
