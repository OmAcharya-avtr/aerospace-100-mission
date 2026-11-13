# Validation — deadlinerisk

Validation level: **Level 3, hardware-pending.** The mission coordinator decides
the level at Phase 9 against the evidence below. This document does not claim a
level.

**No measured campaign is present in this repository yet.** `data/traces/` was
empty for the whole of this build. Every section below that would report a
measured figure carries the literal token `MEASURED_PENDING` and names the script
that fills it; the measured scripts refuse with a non-zero exit until
`data/traces/CAMPAIGN_COMPLETE` exists, and nothing was substituted from the
committed synthetic generator. The sections without that token were measured and
are complete.

Every number in this file was produced by a named script in this directory,
whose raw output is committed beside it as `<script>_output.txt`. Nothing here
was typed from memory, and the mission's build guide requires every quoted
command and figure to be re-executed as the last step before hand-back.

## The measurement model, stated once

Every number in this repository depends on this and nothing else.

- The loop has period `T` and an **implicit deadline** `D = T`.
- Job `i` is **released** at `t0 + i*T`. Its **response time** is
  `end - release`, not `end - start`.
- A job whose response time exceeds `D` is a **miss**.
- If a job finishes after one or more later releases have already passed, those
  releases are never executed. They are **dropped**, and **each dropped release
  is also a miss**. Every executed row carries `dropped_before`, so
  `releases = sum(1 + dropped_before)` and `misses = sum(missed + dropped_before)`
  are both reconstructible from the committed traces.
- The response times in a trace are therefore those of the **executed** jobs
  only. A dropped release has no response time. The mean-plus-margin rule is
  computed from response times and so never sees one; it is nonetheless compared
  against the counted rate over **all** releases, because that is the comparison
  the practice makes.

Source: the implicit-deadline periodic task model of Liu & Layland 1973, JACM
20(1):46-61, with the dropped-release accounting a non-queueing periodic loop
forces.

## What is and is not hardware-backed, by name

| Claim | Backed by |
|---|---|
| the headline understatement factor and its interval | MEASURED, `validate_measured_campaign.py` |
| the counted miss rate, releases, misses and drops on every run | MEASURED, `validate_measured_campaign.py` |
| the block-length selection and the autocorrelation it came from | MEASURED, `validate_measured_campaign.py` |
| the EVT threshold-stability diagnostic and the chosen fit | MEASURED, `validate_measured_evt.py` |
| the learned predictor's scores on both splits, and both baselines' | MEASURED, `validate_measured_learned.py` |
| foreign CPU occupancy, clock and temperature per run | MEASURED, `validate_measured_campaign.py` |
| the loader's rejections and the measurement-model identities | SYNTHETIC and MEASURED, `validate_loader_contract.py` |
| the three independent bounds | SYNTHETIC and MEASURED, `validate_independent_bound.py` |
| the four negative controls | SYNTHETIC, `validate_negative_controls.py` |
| the `pytest.approx` audit | the test suite itself, `validate_approx_audit.py` |
| the Hypothesis hunt and the pinned counterexample | the test suite itself, `validate_property_hunt.py` |
| tree hygiene and `.gitignore` coverage | this repository's tree, `validate_tree_hygiene.py` |
| path independence of every script | this repository's scripts, `probe_working_directories.py` |

A SYNTHETIC figure comes from the committed deterministic generator in
`tests/fixtures/synthetic_trace.py` and measures no board. The distinction is
printed in the header of every output file and is never collapsed.

## How to reproduce everything below

```bash
pip install -e ".[dev]"
python -m pytest tests/ -q --junit-xml=junit.xml
ruff check src/ tests/ examples/ validation/
python -m ruff check src/ tests/ examples/ validation/
for s in validation/validate_*.py validation/worked_example.py; do python "$s"; done
python validation/probe_working_directories.py
for e in examples/*.py; do case "$(basename "$e")" in _*) continue;; esac; python "$e"; done
```

Declared seeds: analysis `920092`, fixtures `920092`, controls `5150`.
Bootstrap resamples for every reported interval: `4000`.

## The suite

| Quantity | Value | Source |
|---|---|---|
| tests, from the junit XML `testsuite` element | 219 | `python -m pytest tests/ -q --junit-xml=junit.xml` |
| `testcase` elements counted in the same file | 219 | cross-check, agrees |
| failures | 0 | junit XML |
| errors | 0 | junit XML |
| skipped | 0 | junit XML |

The count is taken from the junit XML and cross-checked against a count of
`testcase` elements, never from pytest's stdout: a suite that collects zero
tests prints success and exits 0, which is how an earlier product of this
mission shipped with a recorded 251 passing tests and a config that collected
none.

## The measured headline

MEASURED_PENDING — the hardware campaign had not written any file into `data/traces/` when this document was generated, so there is no measured number to put here and none was invented. This slot is filled by `validate_measured_campaign.py`, which refuses with a non-zero exit until `data/traces/CAMPAIGN_COMPLETE` exists.

## Baselines against the measured ground truth

MEASURED_PENDING — the hardware campaign had not written any file into `data/traces/` when this document was generated, so there is no measured number to put here and none was invented. This slot is filled by `validate_measured_evt.py section 3`, which refuses with a non-zero exit until `data/traces/CAMPAIGN_COMPLETE` exists.

## The EVT threshold-choice diagnostic

MEASURED_PENDING — the hardware campaign had not written any file into `data/traces/` when this document was generated, so there is no measured number to put here and none was invented. This slot is filled by `validate_measured_evt.py section 2`, which refuses with a non-zero exit until `data/traces/CAMPAIGN_COMPLETE` exists.

## The learned predictor against both baselines

MEASURED_PENDING — the hardware campaign had not written any file into `data/traces/` when this document was generated, so there is no measured number to put here and none was invented. This slot is filled by `validate_measured_learned.py`, which refuses with a non-zero exit until `data/traces/CAMPAIGN_COMPLETE` exists.

## Independent bounds

Three bounds, none of them used by any estimator in this product.

| Bound | Statement | Uses | What it catches that an agreement check cannot |
|---|---|---|---|
| Markov | `P(R > D) <= E[R]/D` for non-negative `R` | the first moment only | a counted rate inconsistent with the measured mean |
| Cantelli (one-sided Chebyshev) | `P(R - mean >= t) <= s^2/(s^2 + t^2)` | the first two moments, no distributional form | a Gaussian tail computed wrongly, independently of any data |
| Bonferroni bracket | `max(p1, p2) <= P(miss in a consecutive pair) <= p1 + p2` | the monotonicity of probability and the union bound | a miss-indicator expansion that has lost or duplicated releases |

**Every slack follows the conditioning of its own bound expression and none is
floored at an absolute constant.** A floor makes a narrow comparison vacuous: an
earlier product of this mission floored a slack at an absolute 1.8e-15, which
made a bracket of width 4.5e-23 vacuous and admitted the very expression the
check existed to catch.

- Markov slack: `4*eps*bound`. The bound is a quotient of two positive
  quantities, so its relative condition number is 1 in each argument.
- Cantelli slack: `8*eps*((|D|+|mean|)/|D-mean|)*bound`. The ill-conditioned step
  is the subtraction `D - mean`; the slack is its relative condition number
  times the bound's own magnitude.
- Bonferroni slack: `4*eps*max(upper, observed, lower)`. Every quantity is a sum
  of 0/1 over a common denominator; the slack is zero on an all-zero indicator.

Demonstration that the slack is not doing the work, from
`validate_independent_bound.py` section 2: with `sd = 1e-14` and `D - mean = 1`
the Cantelli bound is `1.000000e-28` and its slack is `1.776357e-43`, a ratio of
`1.776e-15`. An absolute floor of 1.8e-15 would exceed that bound by thirteen
orders of magnitude and admit anything.

The bounds have been checked against the SYNTHETIC campaign and hold on all ten of its runs: 10 of 10 Markov checks, 10 of 10 Cantelli checks and 10 of 10 Bonferroni brackets pass. The same checks against the MEASURED campaign are
MEASURED_PENDING — the hardware campaign had not written any file into `data/traces/` when this document was generated, so there is no measured number to put here and none was invented. This slot is filled by `validate_independent_bound.py section 4`, which refuses with a non-zero exit until `data/traces/CAMPAIGN_COMPLETE` exists.

## Negative controls

Four controls, each anchored on a **counted** quantity of a committed
deterministic trace, each asserted to fire in its declared direction. A control
that does not fire is a failure of the control, not a success of the product.

Declared criterion, fixed before the controls were run and not adjusted
afterwards:

```
factor = executed_miss_rate / mean_plus_margin_probability
understating when factor >= 3.0 ; overstating when factor <= 1/3
drop inflation when counted_rate / executed_miss_rate >= 1.5
```

The executed-job rate is used rather than the all-release rate because it is the
quantity the Gaussian rule actually estimates. Control C isolates the drop
accounting on its own.

| Control | Declared direction | Measured factor | Events | Fired |
|---|---|---|---|---|
| A right-skewed lognormal response | must FIRE (understatement) | 4.7011 | 193 executed misses in 300001 releases | yes, clearing the criterion by 5.03 counting standard errors |
| B Gaussian response | must NOT fire | 0.9637 | 217 executed misses in 12001 releases | no, as declared |
| C dropped-release accounting | must FIRE (inflation) | ratio 1.963836 | 217 dropped releases | yes |
| D Pareto-spiked response | must FIRE (overstatement) | 0.0479 | 223 executed misses in 12001 releases | yes |

Source: `validate_negative_controls.py`, which reports
`CONTROLS: 4 declared, 3 fired; must-fire 3/3, must-NOT-fire silent 1/1`.

**Direction D was measured, not assumed, and it is the more interesting result.**
A Pareto spike fired on 2 per cent of jobs inflates the sample standard
deviation more than it inflates the exceedance count, so the Gaussian rule's `k`
collapses and its probability **overshoots** the counted rate by a factor of
about 21. A heavy tail therefore does not always make the mean-plus-margin rule
optimistic. The understatement in control A comes from moderate right skew with
a standard deviation that is not dominated by rare giant outliers, which is a
narrower mechanism than "the tail is heavy" and is why the controls assert a
direction rather than a magnitude.

Control A's criterion is not a 95 per cent interval, and its margin is reported
rather than asserted. With 193 measured events the counting standard error on
the numerator is 7.2 per cent, so the absolute standard error on the factor is
about 0.34 and the measured 4.7011 clears the declared 3.0 by 5.03 standard
errors.

**A first draft of control A was too weak and was strengthened the honest way.**
It ran a 600 s trace, carried 38 events, measured a factor of 4.5382 and sat
only 2.1 counting standard errors above the criterion, which is close enough to
an expected outcome to be weak evidence. The synthetic trace was LENGTHENED from
600 s to 3000 s. The declared criterion of 3.0 was not moved and the seed was
not changed. `tests/test_negative_controls.py` now asserts both the event count
(at least 100) and the margin (at least four counting standard errors), so the
control's strength is itself a test rather than a remark.

## Results of the code and estimator checks

| Check | Result | Source |
|---|---|---|
| `rel=` call sites in `tests/` with an explicit `abs=` | 69 | `validate_approx_audit.py` |
| `rel=` call sites without one | 0 | same |
| assertions depending on `pytest.approx`'s absolute default, measured by re-running the suite with it forced to 0 | **0** | same, dynamic audit |
| suite exit status with the default forced off | 0 | same |
| properties whose example count was raised for the hunt | 12 | `validate_property_hunt.py` |
| pinned Hypothesis/benchmark regression tests | 5 | same |
| schema and loader rejections exercised, each with the message it produced | 9 | `validate_loader_contract.py` |
| measurement-model identity checks on the synthetic campaign | 60 | same |

### The counterexample Hypothesis found

A 4000-example run at seed 1 on 2026-11-13 falsified
`test_mean_plus_margin_k_is_invariant_under_a_common_rescaling` at

```
responses = [0.09479453252194603, 0.09496984880350386]
deadline  = 0.03125
scale     = 0.001
```

with `k = -513.2980590228497` unscaled against `-513.2980590228335` scaled: a
discrepancy of 1.62e-11 against a tolerance of 1.45e-11.

**The tolerance was wrong, not the implementation.** It followed the conditioning
of `D - mean` and ignored that of the sample variance. Two values differing in
their fifth significant figure give a sample standard deviation whose relative
condition number is about 1.2e6, and `k` inherits half of that. The tolerance now
carries both terms:

```
condition_subtraction = (|D| + |mean|) / |D - mean|
condition_variance    = sum(x^2) / ((n-1) * sd^2)
tolerance = 64 * eps * (condition_subtraction + 0.5 * condition_variance) * max(|k|, 1)
```

The case is pinned as `test_regression_k_invariance_two_nearly_equal_responses`
with the literal inputs in the test body, because `.hypothesis/` is gitignored
and a falsifying example found by one run is lost on a fresh clone. The pinned
test also asserts the new tolerance is still far below `|k|`, so the fix did not
make the check vacuous.

### Tree hygiene

From `validate_tree_hygiene.py`, run twice in a row from a clean tree with
identical output both times, which is the property the mission's build guide
requires of any scanner: a check whose own execution can create the condition it
tests is not a check.

| Check | Result |
|---|---|
| artifact-shaped entries present at entry | none, after the build cleared its own bytecode caches |
| artifact-shaped files tracked by git under `products/P092/` | 0 |
| files this product must ship that exist on disk | 15 of 15 |
| files this product must ship that `.gitignore` excludes | 0 of 15 |
| cache and build paths that `.gitignore` excludes | 8 of 8 |
| scratch patterns in this product's `.gitignore` that are anchored or a suffix | 38 lines checked, all |
| `junit.xml` anchored as `/junit.xml` rather than bare | yes |
| text files scanned for an absolute path | 76, zero hits |
| cache directories or bytecode files the script itself added | none |

Section 2 is vacuous for this product's own files at the moment it runs, and the
script says so rather than reporting a pass as if it had content: this build does
not commit, by instruction, so nothing under `products/P092/` is tracked yet and
the `.gitignore` coverage check in section 3 is the one with content.

**Inherited finding, reported and not failed.** `tests/fixtures/junit.xml` is
excluded by `**/junit.xml` in the **repository-level** `.gitignore`, not by this
product's. That pattern would swallow a nested `junit.xml` fixture in any product
that shipped one, which is the exact defect class that cost an earlier product of
this mission a cold-clone failure with 17 failures and 9 errors. P092 ships no
such fixture, so nothing of this product's is lost; the finding belongs to
whoever owns the repository-level file.

**A defect found in this check itself, and fixed.** The first version of the
`.gitignore` coverage test keyed on `git check-ignore`'s exit status. Measured in
a scratch repository with this product's own `.gitignore`: `git check-ignore -v
data/traces/jobs_probe.csv` exits **0** and prints `!/data/traces/*.csv`, while
`git status --ignored` shows the same file as stageable. `git check-ignore` exits
0 whenever **any** pattern matched, including a negation pattern, so its exit
status means "a pattern mentioned this path", not "this path is ignored". The
check would therefore have reported every committed measured trace as wrongly
excluded the moment the campaign wrote one. It now parses the `-v` output and
treats a path as ignored only when the last matching pattern is not a negation.

### Path independence

From `probe_working_directories.py`. Each of the 10 validation scripts and 4
example scripts is executed from the filesystem root, the system temporary
directory, the product root and `validation/`, and the **exit status must be the
same from all four**. Three products in one batch of this mission passed every
check in their own build and then failed the release gate, which runs each
validation script with its working directory set to the script's own directory;
three independent verifications agreeing does not help when they share an
assumption.

| Scripts probed | Directories | Disagreements |
|---|---|---|
| 10 validation + 4 examples | 4 | 0 |

The agreed status is 0 for the scripts that need no measured data and 1 for the
measured ones when the traces are absent. **A path-independent refusal is not a
pass**, and the probe reports it as the agreement it is rather than counting it
as success.

## Structurally unmeasurable from this campaign

These are not checks that failed. They are checks that cannot exist here, and
each is reported as such by the script that would otherwise have run it.

| Check | Why it cannot run |
|---|---|
| held-out ambient temperature | one uninstrumented ambient; there is no second ambient to hold out |
| a second power mode | `sudo -n true` returns non-zero on the board, so `nvpmodel` is unreachable unattended; every figure is a `MAXN_SUPER` figure |
| held-out board | one board, with its own heatsink and fan |
| held-out payload and period | one CPython CPU-only payload at one period; no descriptor varies with either |
| attribution of a miss rate to foreign versus declared load | the board was shared and the foreign load was not controlled; the contamination is measured and reported, not separable |
| a dependency range verified by installing both ends | the build environment is a shared pre-pinned interpreter and this product is forbidden to run `pip`; only `ruff` was measured at two versions, 0.16.9 and 0.17.5 |

## Known gaps in this validation

1. **No repeatability figure.** The campaign is a single pass; no run is
   repeated, so a per-run repeatability interval cannot be computed from the
   committed traces. The reported intervals are sampling intervals within one
   run, not run-to-run intervals.
2. **No sensitivity analysis over the window length or the block-rule cap.**
   Both are declared constants, chosen before the data existed. Their effect on
   the reported numbers is not measured.
3. **The monitor is 1 Hz and the loop is 100 Hz.** Sub-second contention bursts
   are invisible in the descriptors while fully visible in the response times,
   which bounds what any descriptor-based predictor can do and is not quantified
   here.
4. **The GPD fit's own parameter uncertainty is not reported.** The
   threshold-stability diagnostic shows how the point estimates move with the
   threshold, which is a different and weaker statement than a confidence region
   on `(xi, beta)`. `pyextremes` would give the latter; this product does not.
5. **The dependency ceilings are not verified at both ends**, for the reason in
   the table above.
