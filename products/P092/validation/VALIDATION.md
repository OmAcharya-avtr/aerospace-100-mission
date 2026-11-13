# Validation — deadlinerisk

Validation level: **Level 4, measured on hardware, with the scope recorded as
part of the result.** The mission coordinator set it at Phase 9 against the
evidence below, on the mission's criterion that the timing and resource figures
come from the board rather than from a model of one. They do. **This document is
written so that either reading is correct**: a reader who holds that Level 4
requires more than one board, more than one power mode and an instrumented
ambient should read it as Level 3 with a measured headline, and nothing below
changes.

**The measured campaign is present and complete**: 10 runs in `data/traces/`,
190 662 executed jobs over 195 010 releases, `CAMPAIGN_COMPLETE` written last.
Every section below was produced by running its named script against those
traces. The document was written before the campaign, with every measured figure
carrying the literal token `MEASURED_PENDING`; the measured scripts exited
non-zero until the traces landed and nothing was ever substituted from the
committed synthetic generator.

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

**The mean-plus-margin rule understates the measured deadline-miss rate on the
ground-truth run by a factor of 1.528423e+05**, 95 per cent moving-block interval
[3.635383e+04, 9.160321e+05], which **excludes 1**. Measured by
`validate_measured_campaign.py`.

| quantity | measured |
|---|---|
| releases | 60001 |
| misses | 816 |
| of which releases never executed | 417 (51.10 per cent of all misses) |
| counted miss rate, all releases | 1.359977e-02 |
| counted miss rate, executed jobs only | 6.696429e-03 |
| ratio all-release / executed-only | 2.030899 |
| mean response | 6.277342 ms |
| sd response | 0.713016 ms |
| k with mean + k*sd = D | 5.221001 |
| mean-plus-margin P(resp > D) | 8.897912e-08 |
| counted rate 95 per cent interval | [1.219938e-02, 1.501642e-02] |
| block length, releases | 7 |
| lag-1 autocorrelation of the miss indicator | 0.547771 |
| cut-off lag L | 60 |
| integrated autocorrelation time | 6.276304 |
| degenerate bootstrap replicates | 0 of 4000 |

**Why the block bootstrap and not the ordinary one.** The miss indicator has a
lag-1 autocorrelation of 0.547771, so consecutive releases are not independent.
An i.i.d. bootstrap on the same series returns an interval **1.44 times
narrower** than the moving-block one. That factor is the precision an i.i.d.
resample would have invented.

## Baselines against the measured ground truth

| baseline | P(miss) on `gt_long` | inside the counted rate's 95 per cent interval |
|---|---|---|
| 1 mean-plus-margin (Gaussian) | 8.897912e-08 | NO — low by 1.53e+05 |
| 2 GPD peaks over threshold | 6.727478e-03 | NO — low by 2.0215 |
| 3 counted rate over all releases | 1.359977e-02 | — (it is the reference) |

**Both baselines are outside the interval and the product says so.** Being five
orders of magnitude better than the thing being refuted is not the same as being
right, and the EVT fit is reported as low by a factor of two rather than as
correct.

The same comparison on every run, from `validate_measured_evt.py` section 4:

| run | counted | Gaussian P | GPD P | k |
|---|---|---|---|---:|
| `gt_long` | 1.359977e-02 | 8.897912e-08 | 6.727478e-03 | 5.221 |
| `mix_c0` | 1.333244e-04 | 4.457871e-69 | 9.515956e-05 | 17.527 |
| `mix_c1` | 3.999733e-04 | 5.279235e-23 | 3.648369e-05 | 9.806 |
| `mix_c2` | 1.333244e-04 | 1.411300e-91 | 7.954718e-06 | 20.261 |
| `mix_c2mem1` | 7.999467e-04 | 9.015874e-91 | 8.247439e-05 | 20.170 |
| `mix_c3` | 2.533164e-03 | 8.749451e-35 | 2.531523e-03 | 12.247 |
| `mix_c4` | 1.586561e-02 | 2.469598e-06 | 7.826897e-03 | 4.567 |
| `mix_c5` | 1.135258e-01 | 9.099267e-02 | 6.286643e-02 | 1.335 |
| `mix_c6` | 3.825745e-01 | 2.507859e-01 | 1.800099e-01 | 0.672 |
| `mix_mem2` | 0.000000e+00 | 7.939543e-165 | 7.678624e-09 | 27.336 |

The Gaussian rule is worst exactly where a practitioner would trust it most: at
`mix_c2` a margin of 20.261 standard deviations corresponds to 1.411300e-91,
and the board missed 2 of 15 001 deadlines. Where the margin is small, at
`mix_c5` and `mix_c6`, the rule is within a factor of 1.25 and 1.53 — the
Gaussian is not useless, it is useless in the regime anyone would ship.

## The EVT threshold-choice diagnostic

The declared rule: among candidate thresholds at sample quantiles 0.80 to 0.99
leaving at least 30 exceedances, take the one minimising the standard deviation
of the GPD shape estimate over the forward window of 5 consecutive candidates
starting there; ties and short windows go to the lowest threshold. It is applied
by code, not by eye, and it is a stability-window rule in the sense of Coles
2001 section 4.3.1.

| quantity | measured |
|---|---|
| admissible candidates | 20 of 20 |
| exceedance counts across candidates | 596 to 11917 |
| shape estimate across candidates | 0.242682 to 0.702151 |
| chosen quantile | 0.8400 |
| chosen threshold u | 6.263055 ms |
| chosen exceedances | 9534 |
| chosen shape xi | 0.594814 |
| chosen scale beta | 0.397903 ms |
| chosen modified scale | -3.327452 ms |
| forward-window shape sd at the chosen candidate | 0.013192 |
| chosen GPD P(resp > D) | 6.727478e-03 |

**The shape is positive at every candidate**, so the response-time tail is heavy
and the Gaussian rule is not merely mis-scaled; it is the wrong family.

**The diagnostic reports against the fit as well as for it.** The modified scale
`beta - xi*u` should be constant in `u` above a threshold at which the GPD
approximation holds (Coles 2001 eq. 4.10). It is not constant here: it runs from
about -4.06 to -0.52 across the candidate range. That is the honest reason the
EVT probability is low by a factor of two rather than exact — the asymptotic
regime is not fully reached at the lower thresholds — and it is reported rather
than smoothed over.

## The learned predictor against both baselines

195 windows over 10 runs, 0 discarded. Scored separately on held-out workload
mixes (leave one run out) and held-out time (the last 30 per cent of every run).

| split | learned forest | speed scaling | constant rate |
|---|---:|---:|---:|
| held-out mixes, MAE | **0.025022** | 0.043799 | 0.071455 |
| held-out mixes, RMSE | 0.065346 | 0.112132 | 0.112982 |
| held-out mixes, bias | -0.015675 | -0.043749 | +0.003364 |
| held-out time, MAE | **0.007330** | 0.040879 | 0.061431 |
| held-out time, RMSE | 0.017055 | 0.104257 | 0.096236 |
| held-out time, bias | +0.003146 | -0.040822 | +0.004320 |

**The pre-registered expectation is refuted and is recorded as refuted.**
`docs/SPEC.md`, written before the campaign, says the classical predictor would
be competitive and that any learned margin would be smaller than the bootstrap
interval on the counted reference. The margins are 0.018776 and 0.033550 against
a half-width of 1.4085e-03 — 13.3 and 23.8 half-widths. The specification was
wrong about this board and the result stands as measured.

**Three qualifications, all of which cut against the learned model.**

1. **Speed scaling wins 5 of the 10 held-out runs**: `mix_c0`, `mix_c1`,
   `mix_c2`, `mix_c2mem1` and `mix_mem2`. Every one is a low-contention run where
   the right answer is nearly zero, and the classical model says nearly zero for
   a structural reason. The forest wins the aggregate because the aggregate is
   dominated by the high-rate runs.
2. **The dominant descriptor is one the specification does not license.**
   `task_pct`, the task's own CPU occupancy, carries importance 0.654312, and it
   is only observable to someone already running the task. Refitting without it,
   and without `load1`, costs a factor of 1.1354 on held-out mixes and 1.2619 on
   held-out time (`validate_descriptor_leak.py`): grouped MAE 0.028411 and time
   MAE 0.009249. The restricted figure is the honest forecasting claim; it still
   beats both baselines.
3. **The grouped bias is negative at -0.015675**, so on an unseen mix the forest
   under-predicts risk on average, which is the dangerous direction here.

Uncertainty: measured 1-sigma coverage 0.7282 (grouped) and 0.7619 (time)
against about 0.68 for a calibrated Gaussian-like sigma, minimum reported sigma
3.013e-04, no window with sigma zero.

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
On the MEASURED campaign the same two bounds hold on all ten runs: 10 of 10
Markov checks and 10 of 10 Cantelli checks, 0 failures, from
`validate_independent_bound.py` section 4. On `gt_long` the counted
executed-job rate 6.696429e-03 sits under a Markov bound of 6.277342e-01, and
the Gaussian mean-plus-margin tail 8.897912e-08 under a Cantelli bound of
3.538714e-02 with a slack of 2.75e-16 that is not load-bearing. These are weak
bounds by construction — they are there to catch an implementation that is wrong
by orders of magnitude, using mathematics the implementation does not use, and
they would not have caught the mean-plus-margin rule's error, because that rule
is not wrong about its own model. It is wrong about the board.

## Three campaigns ran on the board, and the two discarded ones are the lesson

The committed campaign is the third of three run on the same board on the night
of 2026-11-13. The first two are not in this repository and no figure anywhere
in it comes from them. They are recorded because each failed in a way that a
single-run campaign would have published without noticing.

**Campaign 1 (board time 23:14:39, ran to completion, 36 minutes, discarded).**
All ten job traces were written and every one is a valid measurement. **Not one
monitor file was.** The 1 Hz monitor is launched as a subprocess of the campaign
script, and the launch omitted the `--out` argument that the script's own
`argparse` declares required, so every monitor process exited 2 before writing a
row, with its stderr sent to `DEVNULL`. The campaign reported success. The
defect was visible only as ten absent `mon_*.csv` files after the run, and it
would have removed every contention descriptor — which is to say the entire
learned half of this product — while leaving the headline looking healthy.

**Campaign 1 also mis-sized the payload, and that is the second lesson.** It
calibrated to 0.4499 of the deadline from a 1.2 s probe taken immediately after
the board went idle, when the `schedutil` governor had already begun to drop the
clock. The sustained median execution time across its ten runs then came out
near 0.19 of the deadline — a factor of 2.3 — so the task loaded one core to
about 19 per cent, **six of its ten runs recorded zero deadline misses**, and
its miss-rate curve was not monotone in the contention level (0.0015 at four
stressors, 0.0345 at five, 0.0005 at six). A tail product calibrated that way
measures almost nothing.

**Campaign 2 (02:59:19, killed after 3 minutes).** Launched with the monitor
fixed; killed once the first campaign's calibration defect was understood,
because it carried the same 1.2 s calibration. Killing it left **five orphaned
CPU stressors running at about 97 per cent each**, which were found by listing
the board's processes and killed by PID before campaign 3 started. A campaign
launched on top of them would have measured a board with five uncontrolled
busy threads and called the result a contention level.

**Campaign 3 (03:02:08, 37 minutes, published).** Monitor launch fixed; the
payload sized by a **sustained** calibration — a 30 s run with the first third
discarded as governor ramp — which reached 0.7315 of the deadline on its first
attempt with the clock already at 1728000 kHz on all six cores. Every run
carries its monitor file. The miss-rate curve is monotone from two stressors
upward and no run but `mix_mem2` recorded zero misses.

The general lesson, and it is the same one this mission keeps relearning: a
campaign that completes is not a campaign that measured. Both defects above
produced a clean exit status, a full log and a plausible set of files.

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
