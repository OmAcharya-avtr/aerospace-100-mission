# Validation evidence -- scenariocov 0.1.0

Validation **Level 2**. Every number below was produced by running the script
named beside it, in the build session of 2026-10-16, on the environment
recorded in `validate_environment_output.txt` (Python 3.13.16, 2 CPU cores).
The raw output of every script is committed next to it as
`<script-name>_output.txt`. Nothing here was transcribed by hand.

Reproduce everything:

```bash
python validation/validate_environment.py      > validation/validate_environment_output.txt
python validation/validate_tway_exhaustive.py  > validation/validate_tway_exhaustive_output.txt
python validation/validate_planted_gaps.py     > validation/validate_planted_gaps_output.txt
python validation/validate_occupancy.py        > validation/validate_occupancy_output.txt
python validation/validate_uncertainty.py      > validation/validate_uncertainty_output.txt
python validation/validate_benchmark.py        > validation/validate_benchmark_output.txt
python validation/validate_dependency_ends.py  > validation/validate_dependency_ends_output.txt
python validation/validate_fixtures.py         > validation/validate_fixtures_output.txt
python validation/validate_cli.py              > validation/validate_cli_output.txt
python validation/validate_suite.py            > validation/validate_suite_output.txt
python validation/worked_example.py            > validation/worked_example_output.txt
```

Total wall clock for all eleven scripts: **49 s** on 2 cores. The longest
single script is `validate_suite.py`, which runs the whole test suite in a
subprocess; `validate_benchmark.py` takes about 7.4 s.

## Summary

| area | script | checks | failures |
|---|---|---|---|
| t-way coverage against exhaustive enumeration | `validate_tway_exhaustive.py` | 160 comparisons over 5 spaces | 0 |
| planted-gap detection and false positives | `validate_planted_gaps.py` | 8 | 0 |
| grid occupancy, Feller reference, empty blocks | `validate_occupancy.py` | 10 | 0 |
| uncertainty output | `validate_uncertainty.py` | 4 | 0 |
| head-to-head benchmark, both metrics | `validate_benchmark.py` | measurement, no assertions | - |
| dependency range, both ends | `validate_dependency_ends.py` | 16 quantities compared | 0 moved |
| shipped files not gitignored | `validate_fixtures.py` | 71 paths + 6 size checks | 0 |
| command line | `validate_cli.py` | 6 invocations | 0 |
| test suite counted from junit XML | `validate_suite.py` | 6 | 0 |

## 1. t-way coverage against exhaustive enumeration

`validate_tway_exhaustive.py`. Three independent checks on five spaces with
level counts (2,2), (2,3), (3,3,2), (2,2,2,2) and (4,3,3,2,2), at every
strength from 1 to the parameter count.

| check | reference | result | tolerance |
|---|---|---|---|
| closed form `\|T_t\|` versus enumerated tuple count | equation (1), NIST SP 800-142 section 2 | 16 of 16 agree | exact integers |
| package coverage count versus a slow independent reference | enumerate `T_t`, test membership | 128 of 128 agree | exact integers |
| exhaustive campaign | coverage must be complete | 16 of 16 complete, 0 uncovered | exact integers |

No floating-point comparison enters any of these: coverage is a set size over
an integer denominator.

Cost, from the same output, which is where the method stops being usable:

| parameters | levels | t | \|T_t\| |
|---|---|---|---|
| 7 | 3 | 2 | 189 |
| 7 | 3 | 3 | 945 |
| 20 | 4 | 2 | 3040 |
| 20 | 4 | 3 | 72960 |
| 20 | 4 | 6 | 158760960 -- refused, `max_tuples` is 2000000 |
| 40 | 5 | 4 | 57118750 -- refused |

## 2. Planted gaps detected, and nothing else

`validate_planted_gaps.py`. The committed campaign has five deliberate holes,
declared in `fixtures/generation_parameters.json` and reproduced by
`fixtures/regenerate_fixtures.py` at seed 20261016.

| planted gap | measure that must find it | result |
|---|---|---|
| G1 lighting = eclipse AND closing_speed >= 0.20 m/s | 2-way tuple | reported |
| G2 thruster_health = one_failed AND sensor_mode = camera_only | 2-way tuple | reported |
| G3 closing_speed >= 0.20 m/s AND range >= 200 m | 2-way tuple, 9 empty cells | reported, all 9 cells empty |
| G4 nav_position_error >= 1.5 m never executed | equivalence class | `nav_position_error/excessive` uncovered |
| G5 nothing within 2.5 m of range = 200 m | boundary | missed, nearest 3.586 m against a 2.0 m tolerance |

**False positives: zero.** 2-way coverage is 223/226 = 0.986726 and the three
uncovered tuples are exactly G1, G2 and G3. Class coverage is 14/15 and the
single uncovered class is exactly G4.

G4 and G5 are invisible to the combinatorial measures by construction: both sit
inside a single declared bin. That is the reason this package carries three
measures and not one.

**The near tie that was not planted.** The declared boundary
`target_tumble_rate/tumble_limit_2p0` is reported as missed with a nearest
approach of 0.021 deg/s against a declared tolerance of 0.020 deg/s, a margin
of -0.001 deg/s. The hit flag is a strict floating-point comparison one part in
twenty from flipping. The margin is reported beside every boundary for exactly
this case, and no test in this repository asserts that boolean.

Occupancy, same script: 108 of 144 cells occupied = 0.750000, against an
equiprobable-cell reference of 0.812219. The largest empty connected block is
11 cells and contains all 9 planted cells. **Occupancy alone does not isolate
the planted region** at this campaign size -- it reports a larger contiguous
untested region of which the planted one is a part. The 2-way enumeration is
what names the region exactly.

## 3. Grid occupancy and the Feller reference

`validate_occupancy.py`.

| check | reference | result | tolerance |
|---|---|---|---|
| hand-counted occupancy on a 3 x 4 grid | counted by hand in the script | PASS | exact integers |
| equation (3) versus Monte Carlo, equiprobable cells | Feller 1968 Vol. I Ch. II | 5 of 5 agree | within 4 standard errors, 4000 replicates |
| generalised expectation `sum_c (1 - (1 - p_c)^n)`, unequal cells | same argument, per-cell probabilities | 3 of 3 agree | within 4 standard errors |
| empty-block sizes sum to the empty-cell count | exact | PASS | exact integers |

Worked numbers from the equiprobable check: n = 30 into C = 25 predicts
17.6536 occupied cells and measures 17.7127 with a 4-standard-error band of
0.1014; n = 500 into C = 144 predicts 139.5829 and measures 139.5620, band
0.1229.

**Where equation (3) does not apply, stated because this repository's own
fixture is such a case.** Drawing a continuous parameter uniformly over its
declared interval does not make unequal-width bins equiprobable. For the
`closing_speed  x  range_at_handover` grid the bin probabilities under
uniform-in-value sampling are (0.081633, 0.102041, 0.204082, 0.612245) and
(0.078947, 0.131579, 0.263158, 0.526316), and the equiprobable formula is high
by 3.7 cells of 16 at n = 40 and by 0.5 at n = 240. The package reports the
equiprobable reference anyway, because it is the only one computable without a
model of how the campaign was sampled, and says so in the report text.

## 4. The head-to-head, with the spread

`validate_benchmark.py`. Paired protocol: starting campaign of 20 scenarios
drawn from the committed campaign, 20 scenarios added from a 200-scenario pool,
**60 seeds**, the same starting campaign and pool shared by every strategy at
each seed. The starting size of 20 was chosen because a 20-scenario draw sits
at a mean 2-way coverage of about 0.82, which leaves headroom; at 40 the mean
start is 0.95 and every strategy saturates.

### Metric 1, 2-way combinatorial coverage -- the headline

| strategy | final mean | sd | min | max | gain per scenario | seconds |
|---|---|---|---|---|---|---|
| uniform random | 0.943215 | 0.018598 | 0.898230 | 0.977876 | 0.006549 | 0.00 |
| greedy max-min | 0.951254 | 0.018781 | 0.898230 | 0.982301 | 0.006951 | 0.03 |
| learned gap finder | 0.955457 | 0.015815 | 0.920354 | 0.982301 | 0.007161 | 1.33 |
| **coverage-greedy reference** | **0.993584** | 0.006010 | 0.977876 | 1.000000 | **0.009067** | 0.67 |

Paired per-seed differences, with win / tie / loss counts over the 60 seeds:

| comparison | mean | sd | W/T/L | separated |
|---|---|---|---|---|
| learned - greedy max-min | +0.004204 | 0.015119 | 30/10/20 | **no** |
| learned - uniform random | +0.012242 | 0.019199 | 41/6/13 | no |
| greedy max-min - uniform random | +0.008038 | 0.020650 | 39/8/13 | no |
| coverage-greedy reference - learned | +0.038127 | 0.014868 | 60/0/0 | **yes** |

### Metric 2, grid occupancy over the continuous parameters

The metric closest to what the density model actually sees, reported because
the coverage-greedy reference optimises metric 1 directly.

| strategy | final mean | sd | gain per scenario |
|---|---|---|---|
| uniform random | 0.219329 | 0.014706 | 0.004583 |
| greedy max-min | 0.217477 | 0.016646 | 0.004491 |
| learned gap finder | 0.218056 | 0.012926 | 0.004520 |
| coverage-greedy reference | 0.238310 | 0.012429 | 0.005532 |

Paired: learned - uniform random = **-0.001273** (sd 0.017177, W/T/L 25/5/30).
On the metric most favourable to it, the learned gap finder does not beat the
null strategy.

### What this experiment supports

1. The deterministic coverage-greedy reference beats every strategy under test,
   on every one of the 60 seeds, by 0.038127 in mean final coverage. If 2-way
   coverage is what you want, greedily maximising 2-way coverage is what to do,
   and no model is needed for it.
2. The learned gap finder is **not separated** from greedy max-min: the paired
   mean difference is 0.28 of the standard deviation of that same difference,
   and the learned model loses on 20 of 60 seeds.
3. The learned gap finder costs 1.33 s against 0.03 s for greedy max-min on the
   same work, a factor of about 40, for a difference of unestablished sign.
   Wall clock varies with the machine; the coverage numbers do not.

**This result is not removed, softened or retuned.** It is the result.

### Batch-rule ablation, same seeds, 2-way metric

| batch rule | final mean | sd | gain per scenario | shipped |
|---|---|---|---|---|
| sequential refit plus 0.5-bandwidth separation | 0.955457 | 0.015815 | 0.007161 | yes |
| separation only, no refit | 0.934956 | 0.019006 | 0.006136 | no |
| lowest-density block, no refit, no separation | 0.933628 | 0.020104 | 0.006069 | no |
| greedy max-min, for reference | 0.951254 | 0.018781 | 0.006951 | - |

Both unshipped rules are **worse than the greedy max-min baseline**. The
shipped default is the only configuration of this model that is not, and it is
still not separated from that baseline.

## 5. The uncertainty output

`validate_uncertainty.py`, 24 bootstrap replicates.

| check | result |
|---|---|
| bootstrap spread falls with more data | mean sd 0.144124 nats at n=20 falling to 0.063551 at n=240, monotone across 20/40/80/160/240 |
| agreement separates empty from occupied grid cells | mean agreement 0.181100 (153 candidates in empty cells) against 0.049764 (247 in occupied cells), difference +0.131336 |
| the agreement number is not saturated | 4.50 % of the pool at 1.0, 80.50 % at 0.0, 15.00 % strictly between, against a declared sparsity quantile of 0.10 |
| fitted bandwidth reported | 0.724415 in normalised-embedding units at n = 240 |

The agreement number is **not a calibrated probability**. It is the fraction of
bootstrap resamples that place the candidate in the sparsest declared quantile,
and nothing in this repository compares it against an observed frequency.

## 6. Dependency ranges, both ends executed

`validate_dependency_ends.py`. Both ends of every declared range were installed
and the full suite plus this script run at each.

| end | numpy | scikit-learn | matplotlib | pytest | hypothesis | ruff | suite |
|---|---|---|---|---|---|---|---|
| floor | 2.3.4 | 1.8.0 | 3.10.7 | 9.0.1 | 6.169.3 | 0.17.0 | 255 passed, 0 failed |
| ceiling | 2.5.3 | 1.9.1 | 3.11.2 | 9.1.1 | 6.170.0 | 0.17.1 | 255 passed, 0 failed |

**16 of 16 tracked quantities moved by less than 1e-9** between the two ends,
including the fitted KDE bandwidth (0.7244152329), the top proposal's log
density (-10.97837978) and its bootstrap sd (0.06439671852), and all four
benchmark means. The committed numbers are therefore the numbers at both ends
of every declared range, not just at one.

## 7. Shipped files and the suite count

`validate_fixtures.py`: 71 of 71 shipped paths are not gitignored; all six
fixture files are under 1 MB and total 163293 bytes. The probe section shows
`fixtures/junit.xml` resolving to **not ignored** -- this product anchors
`/junit.xml`, and the check reads the matching pattern rather than trusting the
`git check-ignore` exit status, which returns 0 for a negation match too.

`validate_suite.py`: `testsuite @tests` = 255, `@failures` = 0, `@errors` = 0,
`@skipped` = 0, and an independent count of `testcase` elements = 255. The two
counts agree and neither is zero.

## 8. Checks that are not made

- No comparison against `allpairspy`, PICT or any other covering-array
  generator. This package measures coverage; it does not generate covering
  arrays, so there is nothing to compare.
- No real verification campaign. Every number here is measured on a synthetic
  240-scenario fixture generated by a committed script.
- No calibration of the confidence output against observed frequencies.
- No statistical test on the benchmark. The per-seed differences share a
  starting campaign and a candidate pool, which no two-sample test models, so
  the win/tie/loss split and the spread are reported instead of a p-value.
