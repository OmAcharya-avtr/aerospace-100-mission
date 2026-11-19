# Validation evidence — flakerank

Validation level **2**. Every number here was produced by a script in this
directory, run in this session, over
`data/run_history.json` (402100 bytes, sha256
`aa12f8cc861ba66008a0d3762cd68fa24a24b013f897da67c8151996c531b18a`). Each
script writes its raw transcript to `validation/<script-name>_output.txt` beside
itself, and every table below names the file its figures came from.

This software is research-grade. It is not flight-qualified, not certified and
not approved for operational aerospace use.

## How to reproduce all of it

```bash
cd products/P097
python3 -m pytest tests/ -q -p no:cacheprovider --junit-xml=junit.xml
ruff check src/ tests/
for s in validation/validate_*.py validation/probe_working_directories.py; do python3 "$s"; done
for e in examples/*.py; do python3 "$e"; done
```

Every script anchors on `__file__` and runs from any working directory;
`probe_working_directories.py` proves that rather than asserting it.

## Summary

| script | checks | failures | controls that must fire | controls that must stay silent |
|---|---|---|---|---|
| `probe_working_directories.py` | 4 | 0 | — | — |
| `validate_approx_audit.py` | 4 | 0 | 2 of 2 | — |
| `validate_baselines.py` | 10 | 0 | 3 of 3 | 1 of 1 |
| `validate_bounds.py` | 15 | 0 | 5 of 5 | 3 of 3 |
| `validate_disagreement.py` | 23 | 0 | 1 of 1 | 2 of 2 |
| `validate_geometric.py` | 17 | 0 | 3 of 3 | 1 of 1 |
| `validate_history.py` | 19 | 0 | 11 of 11 | 1 of 1 |
| `validate_learned_model.py` | 6 | 0 | — | — |
| `validate_no_live_timing.py` | 14 | 0 | 5 of 5 | 3 of 3 |
| `validate_tree_hygiene.py` | 34 | 0 | — | — |
| **total** | **146** | **0** | **30 of 30** | **11 of 11** |

Test suite: **259 tests collected, 259 passed, 0 failed, 0 skipped**, counted
from the junit XML `testsuite` element and cross-checked against a count of
`testcase` elements. `ruff check src/ tests/` reports `All checks passed!`.

## 1. The mathematical core — `validate_geometric.py`

The closed-form geometric re-run count, checked three ways in increasing order
of what each could have caught.

**Sources.** Feller, W. (1968), *An Introduction to Probability Theory and Its
Applications, Volume I*, 3rd edition, Wiley — the geometric distribution and its
mean. Johnson, N. L., Kemp, A. W. & Kotz, S. (2005), *Univariate Discrete
Distributions*, 3rd edition, Wiley, chapter 5 — the same result with the
shifted-support convention used here.

| check | reference | result | tolerance |
|---|---|---|---|
| `E[N]` at `p = 1/4` | `1/0.75 = 4/3` by hand | 1.3333333333333333 | 1e-15 relative |
| `E[F]` at `p = 1/4` | `0.25/0.75 = 1/3` by hand | 0.3333333333333333 | 1e-15 relative |
| `E[F_3]` at `p = 1/4` | `(p - p³)/(1-p) + p³ = 0.328125` by hand | 0.328125 | 1e-15 absolute |
| `E[F]` at `p = 1/44` | `(1/44)/(43/44) = 1/43` by hand | 0.023255813953488372 | 1e-15 relative |
| equation (5) against its division-free recursion, 30 cells | `E[F_m] = p(1 + E[F_{m-1}])` | worst relative difference 1.493e-16 | 1e-12 relative |
| `1/(1-p)` against the fixed-point iteration, 6 cells | `E[N] = 1 + p·E[N]` | worst relative difference 2.220e-16 | 1e-12 relative |
| **Monte Carlo of the retry process, 36 cells** | 20000 simulated invocations per cell | 0 of 36 outside 4 standard errors, worst 2.484 sigma | 4 standard errors, declared in advance |

The Monte Carlo is the only one of the three that could catch a wrong *model*
rather than a wrong evaluation of the right model: it draws Bernoulli attempts
and counts failures with no reference to any closed form.

**The divergence at `p → 1`.** Three controls, all firing: `expected_failed_attempts(1.0)`,
`expected_attempts_to_first_pass(1.0)` and `expected_wasted_seconds(1.0, 10.0)`
each raise `ValueError` rather than returning `inf`. One control staying silent:
at `p = 1 - 1e-12` the unbounded form returns 1000022122208.5028, a large finite
number, so the error is a statement about `p = 1` and not a refusal to work near
it. With a retry budget the same quantity is finite and equals the budget —
checked at `m` = 1, 2, 3, 7 and 100.

**The dashboard inversion, which is the product's whole argument, as arithmetic.**

| test | flake rate | expected wasted seconds per invocation |
|---|---|---|
| 1-in-44 failure in a 600 s test | 0.022727272727272728 | 13.953488372093023 |
| 1-in-2 failure in a 0.1 s test | 0.5 | 0.1 |

Cost ratio 139.53488372093022 one way; rate ratio 22.0 the other. The two
rankings disagree on this single pair by a factor of 3069.7674418604647.

## 2. The committed history — `validate_history.py`

| quantity | measured |
|---|---|
| sha256 | `aa12f8cc861ba66008a0d3762cd68fa24a24b013f897da67c8151996c531b18a` |
| bytes | 402100 (under the mission's 1 MB limit) |
| tests / sessions / attempts | 72 / 330 / 23836 |
| failed attempts | 1417 |
| pooled per-attempt rate | 0.059447893941936565 |
| total attempt time | 2953413293 ms |
| wasted (failed-attempt) time | 253493035 ms |
| wasted fraction of total | 0.08583053228642727 |
| shortest / longest attempt | 11 ms / 1848115 ms |
| correlation of true `p` with `log` true mean duration | -0.08985773213530207, 0.762 standard errors from zero |

**Eleven loader rejections, all firing**, each applied to a mutated copy of the
real committed history: an unknown schema, a duration unit other than
millisecond, a retry budget below 1, a negative duration, a fractional duration,
a non-binary failure flag, a session index out of range, a test index out of
range, an attempt number that is not 1-based, unequal attempt-array lengths and
a duplicate test name. **One control staying silent**: the unmodified history
loads.

Durations are exact integer milliseconds and the total 2953413293 is far below
`2**53 = 9007199254740992`, so every partial sum is exactly representable and
the input contributes no representation error to any accumulation.

## 3. The three baselines against the generator's own expectation — `validate_baselines.py`

The baselines were validated **before** the learned component was benchmarked.
The truth used here is not the accumulated outcome — that is baseline 3, one of
the things being checked — but the generator's own expectation, reconstructed
from parameters no baseline and no model reads.

| ranking | Kendall tau-b against the generator's expectation | Spearman rho |
|---|---|---|
| baseline 1, flake rate | 0.729281462 | 0.860134282 |
| **baseline 2, analytic expected cost** | **0.896547852** | **0.920010948** |
| baseline 3, accumulated wasted time | 0.895432743 | 0.919895282 |

| check | result |
|---|---|
| aggregate analytic cost | 254695.0609070536 s |
| aggregate generator expectation | 225460.93388879017 s |
| relative difference | +0.1296638247435066, inside the declared 0.25 |
| aggregate accumulated wasted time | 253493.03500000003 s |
| its 95 % interval, resampling 330 sessions | [221006.75767500003, 285213.18475] s |
| the interval covers the generator's aggregate expectation | yes |

**Per-test interval coverage, reported separately for two groups** because they
behave differently and an average would hide the second:

| group | tests | the 95 % interval covers the generator's expectation |
|---|---|---|
| at least one observed failure | 31 | 30 of 31 = 0.967742 |
| no observed failure | 41 | 38 of 41 = 0.926829 |

Of the 41 never-failing tests, 3 have a true expected wasted time above zero and
an interval of exactly `[0, 0]`. That is a correct statement about the
resampling distribution and a misleading statement about the test, and it is the
reason baseline 2 exists: an analytic model extrapolates where an accumulation
cannot.

**Baseline 1's blindness to runtime, as a measurement.** Multiplying every
duration in the history by 7 changes no flake-rate score at all (largest change
0.0) and multiplies the accumulated wasted time by 6.999999999999999.

**Negative controls, anchored on the generator's expectation and not on any
output**: a true value 100x too large is rejected by the interval for 26 of 26
tests with at least 5 observed failures; 100x too small, 26 of 26; twice the
true aggregate is rejected by the aggregate interval. **One control staying
silent**: the unperturbed true value is covered for 25 of those 26 tests.

## 4. The independently derived bracket — `validate_bounds.py`

The accumulator masks the attempt table on the failure flag, groups by test and
sums. Neither bound here groups anything or sums a subset: each uses an integer
failure count plus statistics over the **whole** duration sample.

- Equation (1): `F·t_min ≤ W ≤ F·t_max` — counting plus two extreme order
  statistics. `min` and `max` are selections, not additions, so they cannot
  share an accumulation error with the thing they bracket.
- Equation (2): `W ≤ sqrt(F · Σt²)` — Cauchy–Schwarz against the all-ones
  vector. *Source:* Hardy, G. H., Littlewood, J. E. & Pólya, G. (1952),
  *Inequalities*, 2nd edition, Cambridge University Press.
- Equation (3): `slack = 4·F·eps·Σ|t|` — the standard forward error bound for
  floating-point summation. *Source:* Higham, N. J. (2002), *Accuracy and
  Stability of Numerical Algorithms*, 2nd edition, SIAM, chapter 4. **The slack
  follows the conditioning of the bound expression, not an absolute constant.**

Over six session windows:

| window | F | accumulated (s) | lower (s) | order upper (s) | Cauchy–Schwarz (s) | slack (s) | inside |
|---|---|---|---|---|---|---|---|
| [0,330) | 1417 | 253493.035 | 15.587 | 2618778.955 | 1575872.940 | 1.859e-06 | yes |
| [0,30) | 120 | 21677.570 | 2.040 | 169312.080 | 135313.432 | 1.396e-08 | yes |
| [0,110) | 399 | 59118.799 | 5.187 | 737397.885 | 476173.684 | 1.706e-07 | yes |
| [110,220) | 369 | 76843.540 | 5.535 | 639092.502 | 465354.947 | 1.607e-07 | yes |
| [220,330) | 649 | 117530.696 | 7.139 | 1066917.709 | 622672.980 | 2.910e-07 | yes |
| [300,330) | 177 | 27357.105 | 2.478 | 283274.694 | 166631.562 | 2.131e-08 | yes |

**The slack is not what makes the check pass**: worst slack-to-width ratio
across the six windows is 1.179e-12, against a declared threshold of 1e-6.

The bracket also holds on **200 of 200** randomly generated synthetic histories.

**What the bound is worth, stated rather than implied.** The bracket's width is
set by the spread of the duration sample, and this history spans 0.011 s to
1848.115 s. A doubled accumulator total still fits inside it; a tenfold total
does not. The bound catches structural errors — a wrong mask, a missing group, a
sign, a double count — and not every multiplicative one. So the controls that
need a tight bracket use a near-constant-duration history (40 attempts of
1000 ms, 20 of them failing), where equation (1) gives `[20.0, 20.0]` exactly:
there a 2x, a 1.1x and even a 1.001x error are all caught.

**Five controls firing, three staying silent.**

## 5. The headline disagreement — `validate_disagreement.py`

**Sources.** Kendall (1938) *Biometrika* 30(1/2) 81–93 for tau; Kendall (1945)
*Biometrika* 33(3) 239–251 for the tie correction that makes it tau-b; Kendall
& Gibbons (1990) *Rank Correlation Methods*, 5th edition, Edward Arnold, for
both and for the midrank correction; Spearman (1904) *American Journal of
Psychology* 15(1) 72–101 for rho; Efron (1979) *Annals of Statistics* 7(1) 1–26
and Efron & Tibshirani (1993) *An Introduction to the Bootstrap* for the
percentile interval.

**Tie handling, stated.** A pair tied in either vector counts as neither
concordant nor discordant **and** is removed from the denominator for that
vector separately, which is what makes this tau-b rather than tau-a. On this
history the flake-rate vector is heavily tied — 41 tests have rate exactly 0,
giving 820 tied pairs of 2556 — so the correction is not cosmetic.

| hand-calculated case | expected | computed |
|---|---|---|
| no ties, one inversion: `C=5, D=1`, `4/6` | 0.6666666666666666 | 0.6666666666666666 |
| one tie group each side: `4/sqrt(5·5)` | 0.8 | 0.8 |
| Spearman with ties: `3.75/4.5` | 0.8333333333333334 | 0.8333333333333334 |
| perfect agreement / exact reversal | 1.0 / −1.0 | 1.0 / −1.0 |

**Cross-check against `scipy.stats`, an independently written implementation**:
worst relative difference 1.355e-16 for tau-b and 1.172e-16 for rho, across
all three pairs. The one declared difference is that a fully tied vector gives 0.0
here and `NaN` in scipy.

### The measured disagreement, over 72 tests and 330 sessions

| comparison | Kendall tau-b | 95 % interval (2000 resamples of 72 tests) | Spearman rho | max rank move | top-10 overlap |
|---|---|---|---|---|---|
| **ground truth vs flake rate** | **0.8191244239631337** | **[0.7375535244988303, 0.8992828376783003]** | 0.947669374556353 | 21 | **5 of 10** |
| ground truth vs analytic cost | 0.9988479262672811 | [0.993177650637593, 1.0] | 0.9999605647133054 | 1 | 10 of 10 |
| analytic cost vs flake rate | 0.8202764976958525 | [0.73197735429303, 0.8980427807486631] | 0.9481425979966874 | 21 | 5 of 10 |

Pair counts behind the first row: 2556 pairs, 820 tied in each vector, 1579
concordant, 157 discordant.

### What the disagreement is worth

| quantity | measured |
|---|---|
| wasted time actually recorded | 253493.03500000003 s |
| its 95 % interval, resampling 330 sessions | [221006.75767500003, 285213.18475] s |
| fraction held by the flake-rate top 10 | **0.7916781500525251** |
| fraction held by the analytic top 10 | **0.9936751713908036** |
| fraction held by the best possible top 10 | 0.9936751713908036 |
| shortfall of the flake-rate ranking | 0.20199702133827857 |

### The motivating case

`soak/test_validation_probe_1_in_44` — named for a failure this mission actually
shipped, a 1-in-44 intermittent failure in a validation probe that no number of
agreeing serial verifications could see.

| quantity | value |
|---|---|
| rank by accumulated wasted time | 8 of 72 |
| rank by flake rate | 27 of 72 |
| displacement | −19 positions |
| accumulated wasted time | 2223.9179999999997 s |
| measured flake rate | 0.012698412698412698 |

**Controls.** A permutation of the ground-truth ranking gives tau-b
−0.03110599078341014, so the measured 0.8191244239631337 is a measurement of
partial agreement and not an artifact of the statistic. Two controls stay
silent: a ranking against itself gives exactly 1.0 and against its reversal
exactly −1.0.

## 6. The learned component — `validate_learned_model.py`

Full tables are in `MODEL_CARD.md`. The finding, counted over two splits and
four metrics, with an exact draw counted as a shared cell:

| predictor | share of 8 cells | outright wins |
|---|---|---|
| analytic expected cost | 2.83 | 2 |
| flake rate | 2.00 | 2 |
| accumulated persistence | 1.83 | 1 |
| **learned forest** | **1.33** | **1** |

A classical baseline matches or beats the learned forest. The forest's three
largest feature importances are `analytic_cost_per_session` (0.3399693389703807),
`wasted_seconds` (0.32644603859189514) and `log1p_wasted_seconds`
(0.2818601186220956) — the baselines' own outputs. This is published as the
finding and was not retuned away.

The learned model's nominal 90 % interval covers 0.909091 of held-out-TESTS
targets and 0.830000 of held-out-TIME targets.

## 7. No live timing anywhere — `validate_no_live_timing.py`

The defect class this checks for cost this mission a whole product: five
intermittent failures and a sixth caught in flight all traced to checks
comparing two wall-clock readings on a shared two-core runner. This product is
*about* intermittency, so the rule is applied in its strongest checkable form.

All 41 Python files in `src/`, `tests/`, `examples/`, `validation/` and `data/`
are parsed with `ast` — not grepped, so a mention in a comment or docstring
cannot trip the check and a clock behind a line continuation cannot hide from
it.

| screen | offenders |
|---|---|
| imports of `time`, `timeit`, `datetime`, `resource`, `sched` | 0 |
| reads of `perf_counter`, `monotonic`, `process_time`, `clock`, `getrusage`, … | 0 |
| `/usr/bin/time`, `/proc/uptime`, `/proc/stat`, `clock_gettime` strings | 0 |

**Five controls firing** on synthetic sources that do read a clock, and **three
staying silent** on a clean source, a comment mentioning `perf_counter` and a
docstring mentioning `time.monotonic`.

## 8. The `pytest.approx` hidden-default audit — `validate_approx_audit.py`

`pytest.approx(expected, rel=r)` passes if *either* bound is met and the
absolute default is `1e-12`, so for a quantity of order 1e-5 every relative
tolerance tighter than 2e-8 is silently replaced.

| measurement | result |
|---|---|
| `approx` calls parsed from 13 test files | 116 |
| with both `rel=` and `abs=` | 107 |
| with `abs=` only | 9 |
| **relying on the hidden default** | **0** |
| the whole suite re-run with a bare `rel=` made an error | passes, exit 0 |

Two controls, both firing: an assertion built to pass *only* because of the
hidden default (`1e-15` against `2e-15` with `rel=1e-30`) passes without the
plugin and is refused with it.

## 9. Tree hygiene — `validate_tree_hygiene.py`

A self-check asserts what the product owns and not what a harness left behind.

| check | result |
|---|---|
| files a cold clone would carry | 65 |
| of those, cache or build artifacts | 0 |
| serialised model files anywhere in the tree | 0 |
| 22 required files exist and are not ignored | all pass |
| 8 scratch paths are ignored, including `model.joblib` and `src/flakerank/forest.pkl` | all pass |
| tracked text files carrying an absolute home or temp path | 0 of 59 scanned |
| cache directories or bytecode files **added** by running the script | 0 |

Artifacts already present at entry are reported, **named as inherited**, and do
not fail: release-gate check 2 runs `ruff check` with its working directory set
to the product root and writes `.ruff_cache` into the tree one check before
check 6 re-executes this script, and no change to this product can fix that.

No `git` command is run. The build agent for this product is forbidden from
running `git`, so the `.gitignore` semantics the check needs are re-implemented
and the covered subset is declared in the output. The authoritative check is
release-gate check 9, which runs against the real index.

## 10. Every script runs from any working directory — `probe_working_directories.py`

Three products in one batch of this mission failed the release gate after
passing every preceding check, because a validation script passed a bare
relative path to a loader. The build agent, its self-report and the
coordinator's re-measurement all ran the script from the product root; the gate
runs it with `cwd=validation/`.

Each script is run from the filesystem root, the system temporary directory, the
product root and its own directory, **inside a pristine copy of this product**,
with `FLAKERANK_FAST=1`.

| script | `/` | tmp | product root | own directory | same |
|---|---|---|---|---|---|
| `examples/cost_versus_rate.py` | 0 | 0 | 0 | 0 | yes |
| `examples/geometric_rerun_cost.py` | 0 | 0 | 0 | 0 | yes |
| `examples/model_against_baselines.py` | 0 | 0 | 0 | 0 | yes |
| `validation/validate_baselines.py` | 0 | 0 | 0 | 0 | yes |
| `validation/validate_bounds.py` | 0 | 0 | 0 | 0 | yes |
| `validation/validate_disagreement.py` | 0 | 0 | 0 | 0 | yes |
| `validation/validate_geometric.py` | 0 | 0 | 0 | 0 | yes |
| `validation/validate_history.py` | 0 | 0 | 0 | 0 | yes |
| `validation/validate_learned_model.py` | 0 | 0 | 0 | 0 | yes |
| `validation/validate_no_live_timing.py` | 0 | 0 | 0 | 0 | yes |
| `validation/validate_tree_hygiene.py` | 0 | 0 | 0 | 0 | yes |

`validate_approx_audit.py` is excluded from the probe with its reason recorded:
it runs the whole pytest suite in a subprocess, which four times over would
dominate the probe's cost for a question it cannot answer differently from the
other scripts, since it resolves every path through `_bootstrap` exactly as they
do. It is covered by its own run from `validation/`, which is the gate's own
arrangement.

**The probe did not touch the tree**, which is checked rather than asserted: the
bytes of all 10 committed `*_output.txt` files and all 3 committed PNGs are
recorded before the probe and compared after it. 0 output files other than the
probe's own changed, and 0 screenshots changed.

## What is NOT validated

Stated because a validation document that lists only successes is not evidence.

1. **No real CI run history.** Everything is measured on a synthetic generator.
   Every figure here is a property of that generator as much as of the code.
2. **The declared dependency ranges were not exercised at both ends.** The build
   guide asks for the suite and the committed output to be compared at the floor
   and the ceiling of each range; the build agent for this product is forbidden
   from running `pip` at all, because a concurrent install mutated numpy
   mid-build in an earlier batch of this mission. The declared ranges are the
   minor series that was measured.
3. **Retry dependence is not modelled or measured.** The generator makes
   attempts independent; real retries frequently are not.
4. **The bootstrap of a rank correlation is not bias-corrected.** A resample
   that draws the same test twice creates an artificial tie, which inflates the
   tie counts and biases tau-b upward. The interval is better read as a width
   than as a calibrated coverage statement.
5. **The percentile bootstrap is uncorrected for skew.** The wasted-time total
   is strongly right-skewed; BCa would be better and needs an acceleration
   estimate this product does not compute.

## Credits

This is under reserved rights obtained by OPTIMA Organisation.
