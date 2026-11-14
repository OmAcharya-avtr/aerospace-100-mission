# Validation — P093 PowerEnvelope

**Status: TESTING.** Every number in this file comes from a script in this
directory that was run on 2026-11-14 against the committed hardware campaign in
`data/traces/`, and each row names the script and the output file it came from.
Failed and weak checks are published here, not removed.

**One check is FALSIFIED and the falsification is the finding, not a defect in
the run.** The duty-cycle factor's lower bound built from the campaign floor's
minimum power is violated on exactly 7 of the 19 pipeline cells, because the
bound's premise is false on this campaign: the run order was not randomised, the
board's foreign load fell throughout, and the floor run was therefore measured
at the highest load of the whole campaign. The bound is not edited, the check is
not skipped and the tolerance is not loosened. `validate_measured_headline.py`
reports `FALSIFIED` as its finding, names all seven cells with their sleep
minima, and **pins those seven as literals**: it exits 0 while the falsification
set is exactly that list and non-zero the moment a cell joins it or leaves it.
Details in *The check that is FALSIFIED, published*.

## Measurement model, stated once

The platform rail is VDD_IN, measured by the on-board INA3221 as bus voltage
(mV) times shunt current (mA); `P(t) = V(t)*I(t)`. Energy over a window is the
**trapezoidal** integral of `P(t)` over the sampler's `CLOCK_MONOTONIC`
timestamps. A cycle is 2.0 s: the pipeline runs back-to-back inferences for
`duty * 2.0` s and sleeps the rest; a run is 38 cycles. An inference is one pass
over one frame, one row of the decisions CSV. A **useful decision** is an
inference that passes the gate — a candidate exists, its box does not touch the
border, its area is within the declared bounds, and its score is at least `tau`
— so the acceptance rate is counted from the committed CSV and never asserted.
**Energy per useful decision** is the integral of `P(t)` over the **run window
`[t0_mono, t1_mono]` exactly**, with `P` interpolated at both ends, sleep
windows included, divided by the number of accepted decisions in that run. The
sampler brackets the run on purpose, starting early and stopping late, so its
span is **wider** than the run and is never what is integrated. `E_active` and
`E_sleep` partition that same window using the declared per-cycle phases, with
`P` interpolated at every boundary.

The same paragraph appears verbatim in `README.md`, in
`src/powerenvelope/trace.py`'s module docstring, in
`powerenvelope.report.MEASUREMENT_MODEL`, and at the head of every validation
output that reports a measured quantity.

## Environment every number was produced in

| component | version |
|---|---|
| Python | 3.13.18 |
| numpy | 2.5.3 |
| scipy | 1.18.1 |
| scikit-learn | 1.9.1 |
| pytest | 9.1.1 |
| hypothesis | 6.170.15 |
| matplotlib | 3.11.2 |
| ruff | 0.16.9 on the path, 0.17.5 as a Python module — clean under both |
| analysis machine | x86_64 Linux, 2 cores |

The **traces** are measured on the board, aarch64. The **arithmetic** over them
runs on the analysis machine. Those are different machines and every validation
output prints both.

## Test suite

| quantity | value | source |
|---|---|---|
| tests | 360 | `junit.xml` `testsuite/@tests` |
| failures | 0 | `junit.xml` `testsuite/@failures` |
| errors | 0 | `junit.xml` `testsuite/@errors` |
| skipped | 0 | `junit.xml` `testsuite/@skipped` |
| `testcase` elements | 360 | cross-check against the attribute |

`tests/test_measured_campaign.py` adds 16 tests over the committed campaign.
They are skipped with a stated reason if `data/traces/` is absent, because a
missing measurement is not a code defect and turning it into a test failure
would hide the refusal the measured validation scripts exist to make visible.
With the campaign present, none is skipped.

The count is taken from the junit XML and cross-checked against a count of
`testcase` elements, never from pytest's stdout: a suite that collects nothing
prints success and exits 0.

## The checks

| # | check | script | what it would take to fail it | result |
|---|---|---|---|---|
| 1 | The loader accepts a complete campaign and refuses every malformed one | `validate_loader_contract.py` | a header that moved being accepted; a ragged row, a non-numeric field, a decreasing timestamp, an `accepted` value outside {0,1}, an `accepted` value disagreeing with its reason, an inverted inference interval, an undeclared reject reason, or a power column that is not `V*I/1000` being accepted | **PASS**, 35 checks, 26 controls (25 must-fire all fired, 1 must-not-fire silent) |
| 2 | The acceptance rate is reconstructed by counting, never asserted | `validate_loader_contract.py` §3 | the reason histogram not summing to the inference count, or `accepted` not equalling the count of `accept` rows | **PASS** |
| 3 | Whole-run energy matches the synthetic closed form on 15 cells | `validate_baselines_synthetic.py` §2 | a relative error above 1e-9 against a closed form derived on paper from the generator's declared constants | **PASS**, largest relative error below 1e-9 |
| 4 | Active-window energy matches the per-cycle closed form | `validate_baselines_synthetic.py` §3 | the same, for duties 0.10–0.75; at duty 1.00 the identity `E_active == E_total` is checked instead | **PASS** |
| 5 | Baseline 1, both variants, against the closed form | `validate_baselines_synthetic.py` §4 | the mean active power not being the closed-form active energy over the active duration, or the nominal variant not being below the measured one by exactly the voltage ratio | **PASS** |
| 6 | Baseline 2's three components sum to the cycle energy, and the ramp is reported as not separable at duty 0.10 rather than estimated | `validate_baselines_synthetic.py` §5 | the components not summing to 1e-12; a ramp figure being produced where the active phase is shorter than three settling windows | **PASS** |
| 7 | Baseline 2 is within 10 % of Baseline 3 on all 15 synthetic grid cells | `validate_baselines_synthetic.py` §5 | the analytic model drifting from the ground truth by more than the declared 10 % | **PASS** |
| 8 | Baseline 3 against the closed form, and the between-run interval over four repeats | `validate_baselines_synthetic.py` §7 | the numerator or the counted denominator disagreeing with the closed form | **PASS**; **the spread is exactly zero**, see Weak checks below |
| 9 | The three factors multiply to F exactly, and the declared phases partition the run | `validate_baselines_synthetic.py` §8 | a closure error above 1e-12, or `E_active + E_sleep` disagreeing with `E_total` over the run window | **PASS**, closure error −1.11e-16 |
| 10 | The trapezoid lies inside the Riemann bracket on every cell | `validate_independent_bound.py` §1 | the trapezoid falling outside the left/right Riemann sums, which it cannot in exact arithmetic because it is their mean | **PASS**, 20 cells |
| 11 | The trapezoid lies inside the mean-value bracket on every cell | `validate_independent_bound.py` §2 | the integral falling outside `[span*min P, span*max P]`, which would mean the wrong samples or the wrong span were integrated | **PASS**, 20 cells |
| 12 | The duty-cycle factor satisfies a bound from the sleep window's own power, on SYNTHETIC data | `validate_independent_bound.py` §3 | `E_total` being below `E_active + floor_min * t_sleep` | **PASS**, 19 pipeline cells on the generator, whose floor is constant by construction. On the MEASURED campaign the same bound is falsified: row 27 |
| 13 | The headline factor satisfies a bound from extreme values alone | `validate_independent_bound.py` §4 | F falling below a bound built with no integral, no acceptance rate as a multiplier, and no part of the decomposition | **PASS**, 19 cells; the measured factor exceeds the bound by 1.08 to 1.34 times, so the bound is tight enough to catch a small error on this data |
| 14 | A bound slack follows the conditioning of its own expression | `validate_independent_bound.py` §5 | a slack floored at an absolute constant, which makes a 4.5e-23-wide bracket admit anything | **PASS**; the floored form is shown to admit an intruder 4.0e7 widths outside the bracket |
| 15 | The quantiser noise floor is the limit on every energy figure | `validate_independent_bound.py` §6 | the 40 mW quantum's implied energy spread falling below the ~1e-8 timestamp-resolution floor, which would mean the arithmetic, not the instrument, limits the result | **PASS** |
| 16 | 26 negative controls, each anchored on the true value and asserted to fire in its declared direction | `validate_negative_controls.py` | any must-fire control not firing, or any must-not-fire control firing | **PASS**: 18/18 must-fire fired, 8/8 must-not-fire silent |
| 17 | Seven claimed identities survive a property hunt | `validate_property_hunt.py` | a counterexample at 600 examples per property | **PASS** — but see Falsified below: one property was falsified and replaced |
| 18 | No assertion depends on `pytest.approx`'s hidden 1e-12 absolute default | `validate_approx_audit.py` | any call site passing `rel=` without `abs=`, or the suite failing with the default forced to 0.0 | **PASS**: 0 bare `rel=` call sites, 360 tests pass with `abs=0.0` forced |
| 19 | A narrower descriptor set is refitted and its cost reported | `validate_descriptor_leak.py` | the restriction being unmeasured, or a narrower set beating the wider one by more than noise | **PASS** on synthetic data; the measured version is row 25 |
| 20 | Git tracks no cache, bytecode or build artifact, and nothing this product must ship is excluded by `.gitignore` | `validate_tree_hygiene.py` §2–3 | a `junit.xml` pattern without a leading slash; a required file matched by an ignore rule | **PASS** |
| 21 | No absolute path in any tracked text file | `validate_tree_hygiene.py` §4 | a committed file carrying an absolute path rooted at a home directory, a user directory, the superuser's directory or a temporary directory. The four patterns are assembled from `os.sep` inside the scanner and are deliberately not written out here: an earlier draft of this row contained them as literals and the scanner reported this file. Three consecutive sessions of this mission introduced that defect in the act of writing about it. | **PASS**, 60+ files scanned |
| 22 | Running a validation script adds no cache directory and no bytecode file | `validate_tree_hygiene.py` §6 | a script's own interpreter writing `.pyc` into the tree | **PASS** |
| 23 | Every validation and example script gives the same exit status from four working directories | `probe_working_directories.py` | a bare relative path reaching a loader, which is the defect class that blocked three products of this mission in one batch | **PASS**, 15 checks: 12 validation scripts and 3 examples, each agreeing with itself across all four directories. With the campaign committed, `validate_measured_headline.py` agrees at exit status 0, the status its pinned falsification reports, and the other two at 0. |
| 24 | The CLI's four subcommands run, label synthetic output, and use the declared exit statuses | `validate_cli.py` | `--help` not exiting 0; a `--simulate` run not labelling its output; absent traces not exiting 3; a corrupted header not exiting 2; the refusal falling back to the generator | **PASS**, 28 checks |
| 25 | The measured headline and its decomposition | `validate_measured_headline.py` §3–4 | the three factors not multiplying to F, or the declared active and sleep energies not partitioning the run window | **PASS**: F = 5.544866, closure error 0.000e+00, partition residual −1.320e-05 |
| 26 | The measured headline against the Riemann, mean-value and extreme-value brackets | `validate_measured_headline.py` §6 | the integral or F falling outside a bracket from other mathematics | **PASS** on all three; F exceeds the extreme-value bound by 3.849 times |
| 27 | **The measured duty-cycle factor against the campaign-floor sleep bound** | `validate_measured_headline.py` §6 | the premise that a sleep window draws at least the floor run's minimum | **FALSIFIED on exactly 7 of 19 cells**, pinned as literals; the script exits 0 on that set and non-zero if it changes — published, see below |
| 28 | The same bound built from each cell's own sleep-window minimum | `validate_measured_headline.py` §6 | the same arithmetic with a premise measured in the cell | **PASS**, 19 of 19; the weaker companion, see Weak checks |
| 29 | The permissive-gate control | `validate_measured_headline.py` §8 | the rejection factor not falling when the gate is relaxed | **PASS**: `1/a` falls 2.8986 → 1.1094, `a` rises 0.3450 → 0.9014, duty-cycle factor unmoved |
| 30 | The measured floor, the floor-occupancy regression, and the floor-sensitivity table | `validate_measured_baselines.py` §2–4 | the intercept exceeding the floor mean, or the correlation being unreported | **PASS**: floor 4258.633 mW, slope 2034.282 mW per unit occupancy, r = 0.693158, foreign share 5.895 per cent |
| 31 | The run-order drift table, and whether a running cell undercuts the floor | `validate_measured_baselines.py` §5 | nothing — this check is written to FIRE on the drift so it cannot be overlooked | **PASS (it fired)**: `w96_d010` draws 3716.75 mW against a 4258.63 mW floor |
| 32 | Baseline 2 against Baseline 3 on all 15 grid cells | `validate_measured_baselines.py` §9 | a gap above the declared 20 per cent | **PASS**, worst 13.48 per cent on `w96_d010` |
| 33 | The three-term and two-term cycle energies agree | `validate_measured_baselines.py` §8 | a residual above 1e-4 relative | **PASS**, worst 4.441e-16; under the declared-phase partition both forms use the same active duration, so they are the same number to rounding |
| 33a | The declared phases partition every run, worst residual named | `validate_measured_baselines.py` §6 | `\|E_active + E_sleep − E_total\| / E_total` above the declared 1e-4 on any cell | **PASS**, worst **8.114e-05** on `w96_d100`, which is 0.81 of the tolerance and is named rather than passed silently |
| 34 | The drift-adjusted Baseline 2, using each cell's own sleep power | `validate_measured_baselines.py` §10 | the adjusted ratios not collapsing | **PASS**: adjusted ratios uniform to 0.0000, against absolute errors up to 0.1348 unadjusted |
| 35 | The measured rail-accounting ratio against load | `validate_measured_baselines.py` §11 | a ratio outside (0, 1) | **PASS**: 0.494338 to 0.644352, R² = 0.118710 against load |
| 36 | The between-run interval from the four reference repeats | `validate_measured_baselines.py` §12 | a zero-width interval | **PASS**: 73.4156 ± 2.1689 mJ, 2.9543 per cent relative |
| 37 | **The reject-reason census** | `validate_measured_baselines.py` §13, `validate_measured_headline.py` §7 | a reason outside the declared set appearing | **PASS on the set; NOT MEASURED on two branches** — see below |
| 38 | The learned predictor against three classical ones, both held-out schemes | `validate_measured_learned.py` §5–7 | the comparison not being computable, or the ceiling being worse than a predictor | **PASS**; the pre-registered expectation is **NOT REFUTED** on both splits |
| 39 | Measurement back-action | `validate_measured_baselines.py` §14 | nothing; it is a reported quantity | 1.058 s of sampler CPU per 77 s run, 1.374 per cent of one core |

## The check that is FALSIFIED, published

One, and it is a finding about the campaign rather than a defect in the code.

The bound is `E_total >= E_active + floor_min * t_sleep`, whose premise is that
a sleep window draws at least the minimum power ever seen on the no-pipeline
floor run, 4113.4 mW. **That premise is false on this campaign.** The run order
was not randomised and the board's foreign load fell over the 27.05 minutes of
the campaign, so the floor was measured at the highest load of the whole run and
later cells' sleep windows draw as little as 3480 mW. The arithmetic is sound;
the premise is an assumption about the board's state and the campaign falsifies
it. This is exactly what a bound derived from different mathematics is for: it
caught a methodological defect that no agreement between two implementations
could have caught.

A script that correctly establishes a falsification has **succeeded**, so
`validate_measured_headline.py` reports the falsification as its finding and
exits **0** — but only while the falsification set is exactly the seven cells
pinned as literals in `PINNED_FALSIFICATION`:

```
pinned falsification set (7 cells):
  w64_d050_permissive    sleep-window minimum 3560.0 mW < floor-run minimum 4113.4 mW
  w64_d050_rep2          sleep-window minimum 3600.0 mW < floor-run minimum 4113.4 mW
  w64_d050_rep3          sleep-window minimum 3560.0 mW < floor-run minimum 4113.4 mW
  w64_d050_rep4          sleep-window minimum 3560.0 mW < floor-run minimum 4113.4 mW
  w96_d010               sleep-window minimum 3480.0 mW < floor-run minimum 4113.4 mW
  w96_d025               sleep-window minimum 3520.0 mW < floor-run minimum 4113.4 mW
  w96_d050               sleep-window minimum 3560.0 mW < floor-run minimum 4113.4 mW
```

If a cell ever joins that set or leaves it, the pinned list no longer matches
and the script **fails for real**. The row in the table above therefore reads
`FALSIFIED`, not `PASS`: the bound does not hold, and the table says so.

The three duty-1.00 cells are **not** in the set. The sleep duration the bound
uses is the declared one, `cycles * (1 - duty) * cycle_s`, so a duty-1.00 run
has no sleep phase at all and the bound degenerates to `factor >= 1`, which
`w48_d100`, `w64_d100` and `w96_d100` satisfy (1.000021, 1.000022, 1.000081).

What was **not** done: the bound was not weakened, the slack was not widened,
the floor was not re-chosen, no cell was excluded, and nothing was re-run. What
was done: the violation is published with the per-cell table, the premise's
falsity and the non-randomised run order that causes it are recorded in
`powerenvelope.bounds.duty_cycle_lower_bracket`'s own docstring, in
`README.md`'s *Limitations* and in `DATASET_CARD.md`, and a variant using each
cell's **own** measured sleep-window minimum is reported beside it — that one
holds on all 19 cells and is labelled as the weaker argument it is, because it
reads its minimum from the same samples the integral it checks is taken over.

**The headline does not depend on this bound, and F does not depend on the floor
at all.** `F` is `(E_total / A)` divided by `(E_active / N)`: four measured
integrals and two counts, none of which is the floor cell. The floor enters
Baseline 2 only, and its sensitivity is published as a three-row table in
`validate_measured_baselines.py` §4. That distinction is the reason the headline
is robust to the contaminated floor and the analytic model is not.

## Checks that are WEAK, published

1. **The in-cell sleep-minimum bound shares an input with what it checks.** It
   reads its minimum from the same samples the integral it bounds is taken
   over, so it is a weaker argument than the campaign-floor form it replaces.
   It is reported as the sound-but-weaker companion, never as a replacement.
2. **The extreme-value bound on the headline is a lower bound only.** Its upper
   end is infinite, so it can catch a factor that is too small and cannot catch
   one that is too large. On the measured reference cell F exceeds it by 3.849
   times, so it would catch only a gross error.
3. **The between-run interval contains the campaign's drift as well as the
   run-to-run variation.** The first of the four repeats is the grid cell
   itself, which ran 651 s into the campaign; the other three ran at the end.
   Its value, 75.4567 mJ, is the outlier of the four (72.8447, 72.6578,
   72.7033). The interval is therefore wider than pure run-to-run variation
   would make it, which is the honest direction to err in, and it is the
   comparator every error in this file is judged against.
4. **`validate_descriptor_leak.py` runs on synthetic data.** The measured
   descriptor comparison is `validate_measured_learned.py` §5; the synthetic
   script exercises the shape of the restriction check, and on synthetic data
   the environment descriptors are deterministic functions of duty, so its cost
   figures are a property of the generator.
5. **The learned predictor's one-sigma coverage is 0.2615**, against the 0.68 a
   correctly sized band would give. The across-tree spread is a lower bound on
   predictive uncertainty: it omits run-to-run variation and the campaign's load
   drift. It is published as measured and not widened.
6. **`env_foreign_busy` is imputed on 117 of 171 windows** and still carries
   0.1621 of the forest's summed importance, which is importance attached to a
   near-constant plus its own flag.
7. **`matplotlib` was measured at one version only, 3.11.2.** It draws the
   screenshots and is in no reported number, so it is outside the both-ends
   sweep below. Every other range was run at both ends on 2026-11-14 in an
   isolated virtual environment outside this tree: floors numpy 2.5.0, scipy
   1.18.0, scikit-learn 1.9.0, pytest 9.1.0, hypothesis 6.170.0; ceilings numpy
   2.5.4, scipy 1.18.1, scikit-learn 1.9.1, pytest 9.1.1, hypothesis 6.170.15,
   each the highest release below the declared bound per the PyPI JSON API on
   that date. Both ends gave **360 passed**, and every committed validation
   output regenerated at both ends differs from the pinned one only in the
   environment stanza, which prints the installed versions, and in two
   wall-clock readings — the suite duration the approx audit quotes and the
   forest's measured fit time. No reported figure moves across any range. The
   examples were not re-run at the floors because `matplotlib` is not installed
   there, which is the same reason it is outside the sweep. `ruff` is clean at
   0.16.9 and 0.17.5.
8. **`validate_tree_hygiene.py`'s "git tracks no artifact" check is vacuous in
   this tree.** The product is not committed at build time — the coordinator
   commits — so `git ls-files --full-name` returns nothing under this subtree
   and the check cannot fail here. The script prints that note only when the
   list is genuinely empty, so it cannot mask a populated tree, and the
   `.gitignore` coverage check in the same section is the one with content.
9. **Two of the gate's five reject branches never fired.** `no_candidate` and
   `border` have zero occurrences across 268,425 measured inferences, so the
   measured acceptance rate — and therefore the rejection factor `1/a` the
   headline decomposes into — is a property of a gate two of whose branches
   were never reached by measurement. Those paths are exercised only by the
   synthetic generator. No occurrence was synthesised to fill the table, and
   `tests/test_measured_campaign.py` asserts the census so the fact cannot
   change silently.
10. **The measured `E_ramp` is negative on 12 of the 15 grid cells** (the three
    duty-0.10 cells report it as not separable and therefore zero), down to
    −202.1 mJ per wake on `w96_d100`. That is a
    measurement of the instrument: the INA3221's 512-conversion hardware
    averaging means a reading within the 0.15 s settling window still carries
    the sleep window's power. The settling window was declared before the
    campaign ran and was not re-chosen after seeing this.
11. **The floor's regression on occupancy has r = 0.6932** (r² = 0.4805). Over
    half of the floor's second-to-second variance is unexplained by the
    occupancy counter, so the intercept is an extrapolation below the observed
    range and is used only as one row of the sensitivity table. The fit uses
    the 75 monitor seconds that lie wholly inside the run window; a second
    straddling `t0_mono` or `t1_mono` is dropped rather than clipped.
12. **The rail-accounting ratio is poorly described as a line in load**,
    R² = 0.118710. The fit is published with that figure rather than replaced
    by a better-fitting form chosen after seeing the data.

## A property this product claimed and had falsified

`validate_property_hunt.py` searched the claim that the trapezoidal energy
integral is **invariant under a shift of the timestamps** and falsified it on
the first run, at

```
t = [0.0, 1e-4],  power = [1.0, 1.0],  shift = 2048.0
```

where the shifted spacing is `1.0000000020227162e-4` rather than `1e-4` and the
integral changes by 2.0e-9 relative, because 2048.0001 is not representable.

The claim was not weakened into vacuity and not deleted. It was replaced by a
quantitative one — the relative change is bounded by
`4 * EPS * max|t| / min(dt)` — which the hunt then could not falsify at 600
examples, and the counterexample is pinned as
`tests/test_regressions.py::test_time_shift_counterexample_t_zero_to_1e_minus_4_shift_2048`
with the literal inputs in the test body, because `.hypothesis/` is gitignored
and an example found by one run is lost on a fresh clone.

The practical consequence, which is why it is in `DATASET_CARD.md` as well: the
campaign's `t_mono` is the board's uptime, of order 1e5 to 1e6 seconds, with
0.05 s spacings, so every energy figure here carries about 1e-8 relative
uncertainty from the timestamp representation alone. That is ten orders of
magnitude below the 40 mW current-quantisation floor and changes no reported
figure.

## Reproducing all of it

```bash
python -m pytest tests/ -q --junit-xml=junit.xml
ruff check src/ tests/ examples/ validation/
python -m ruff check src/ tests/ examples/ validation/
python validation/probe_working_directories.py   # FIRST: it sets POWERENVELOPE_FAST=1
rm -rf .ruff_cache .hypothesis                   # so the hygiene check sees a clean tree
find . -name __pycache__ -type d -prune -exec rm -rf {} +
for script in validation/*.py; do
  case "$(basename "$script")" in _*|probe_*) continue;; esac
  python "$script"
done
for example in examples/*.py; do
  case "$(basename "$example")" in _*) continue;; esac
  python "$example"
done
```

The probe runs first because it sets `POWERENVELOPE_FAST=1`, which reduces the
property hunt from 600 examples to 60; running the individual scripts afterwards
leaves the committed outputs at their full settings.

Every script exits **0**, `validate_measured_headline.py` included: its
campaign-floor sleep bound is falsified, and a script that correctly establishes
a falsification against its pinned set has succeeded. It exits non-zero if that
set ever changes. If `data/traces/` is ever absent, all three
`validate_measured_*.py` scripts exit 1 on the refusal instead, which is the
behaviour `tests/test_refusal.py` verifies hermetically.
