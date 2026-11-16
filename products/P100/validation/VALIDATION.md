# Validation — P100 SeedAudit

**Status: TESTING.** Every number in this file comes from a script in this
directory that was executed on 2026-11-16 and that exits 0 when re-run. Each
row names the script and the output file it came from. Failed, weak and
refuting results are published here, not removed.

**One pre-registered expectation is REFUTED and one is CONFIRMED, and both are
reported as they landed.** See *Pre-registered expectations, judged*.

**The corpus-wide audit is a CAMPAIGN, not a validation script.** It lives at
`data/repository_audit.py`, takes most of an hour, and the release gate does
not run it. Its output is committed frozen at `repository_findings.json` with
its sha256 pinned, and `validate_repository_findings.py` verifies that evidence
in under a second without executing anything. Section 9 records why the two had
to be separated, and section 10 records the load-sensitivity finding that
forced the de-pinning. Both are findings about this product and both are
published.

## What a number in this product means, stated once

A **committed numeric claim** is a numeric token in a committed output file
beside the script that produced it, where that script declares a rewritable
random seed. Each candidate is executed **twice, each time in its own pristine
copy in a temporary directory**: PERTURBED, with every seed literal rewritten
by a fixed offset, and CONTROL, with the seeds left at their declared values. A
number that moves in the perturbed run and **not** in the control is
seed-attributable and is the headline. A number that moves in the control moved
with the seed held fixed and is counted separately. Detection power is
`1-(1-p)^n` against a committed reference and `1-p^n-(1-p)^n` for run-and-diff
with no reference; a clean result reports the escaped-rate upper bound
`1-alpha^(1/n)`, never zero.

The same paragraph appears in `README.md`, in `seedaudit.report.AUDIT_MODEL`,
and at the head of the repository audit's own output.

## Environment every number was produced in

| component | version |
|---|---|
| Python | 3.13.18 |
| numpy | 2.5.3 |
| scipy | 1.18.1 |
| matplotlib | 3.11.2 |
| pytest | 9.1.1 |
| hypothesis | 6.170.15 |
| ruff | 0.16.9 on the path, 0.17.5 as a Python module — clean under both |
| machine | x86_64 Linux, 2 cores, 7.8 GiB, shared with another build agent |

## Package name on PyPI

| name | URL probed | status | meaning |
|---|---|---|---|
| `seedaudit` | `https://pypi.org/pypi/seedaudit/json` | **HTTP 404** | FREE |
| `numpy` | `https://pypi.org/pypi/numpy/json` | **HTTP 200** | live control, TAKEN |

Both probed 2026-11-16T04:17:23Z. The control is there so that a 404 from a
broken network cannot be read as an available name.

The seven alternatives named in `README.md` were each probed on the same day
and all returned HTTP 200: `pytest-randomly` 5.1.0, `pytest-rerunfailures`
16.7, `flaky` 3.8.1, `reprotest` 0.7.33, `diffoscope` 339, `pytest-repeat`
0.9.4, `pytest-flakefinder` 1.1.0.

## Test suite

| quantity | value | source |
|---|---|---|
| tests | 181 | `junit.xml` `testsuite/@tests` |
| failures | 0 | `junit.xml` `testsuite/@failures` |
| errors | 0 | `junit.xml` `testsuite/@errors` |
| skipped | 0 | `junit.xml` `testsuite/@skipped` |
| `testcase` elements | 181 | cross-check against the attribute |

The count is taken from the junit XML and cross-checked against a count of
`testcase` elements, never from pytest's stdout: a suite that collects nothing
prints success and exits 0, and that is how an earlier product of this mission
shipped publicly with a recorded 251 tests passing and a configuration that
collected none.

`junit.xml` is **not tracked**. The repository's root `.gitignore` carries
`**/junit.xml` and this product's carries `/junit.xml`, anchored with a leading
slash so it cannot match at an unintended depth.

## The checks

Every check below is produced by a script in this directory. Scripts and their
exit statuses are listed at the end of this file.

| # | check | script | what it would take to fail it | result |
|---|---|---|---|---|
| 1 | both closed forms against a Monte Carlo simulation | `validate_power_model.py` | an estimate more than 4 standard errors from its closed form on any of 24 grid cells | **PASS**, worst z **3.3018** |
| 2 | the closed form inside a two-term inclusion-exclusion bracket | `validate_power_model.py` | the value falling outside a bracket derived from the union, not from the complement product | **PASS**, 16 of 16 |
| 3 | the closed form inside the exponential bracket | `validate_power_model.py` | the value falling outside a bracket derived from a bound on the logarithm | **PASS**, 30 of 30 |
| 4 | the union bound sits above the closed form | `validate_power_model.py` | `n*p` falling below `1-(1-p)^n` anywhere on the grid | **PASS** |
| 5 | `p_min` and `p_max` agree, and `p_max` recovers 0.95 | `validate_power_model.py` | the two inversions disagreeing, or the round trip missing 0.95 by more than 1e-12 | **PASS**, disagreement **0.0**, round-trip error **0.0** |
| 6 | 131 repeats reach 0.95 at `p = 1/44` and 130 do not | `validate_power_model.py` | the minimal repeat count being anything else | **PASS** |
| 7 | the hypergeometric seed axis differs from a binomial one | `validate_power_model.py` | the two agreeing, which would make the distinction cosmetic | **PASS** |
| 8 | corpus calibration, family-wise Clopper-Pearson | `validate_injected_corpus.py` | any family-wise interval not covering its injected rate | **PASS**, 6 of 6 |
| 9 | derived pairwise power inside the family-wise interval | `validate_injected_corpus.py` | the disjoint-block detection mathematics disagreeing with the closed form at the realised rate | **PASS**, 24 of 24 |
| 10 | derived reference power inside the family-wise interval | `validate_injected_corpus.py` | the same for the reference design | **PASS**, 24 of 24 |
| 11 | **Baseline 1 at `p = 1/44` contains its derived value** | `validate_injected_corpus.py` | the measured power excluding 0.044421487603305776 | **PASS**, measured **0.05181818181818182** |
| 12 | Baseline 1 at `p = 1/44` is below 0.06, as pre-registered | `validate_injected_corpus.py` | the realised sample landing above 0.06 | **PASS** |
| 13 | the dynamic detector stays silent on all three negative controls | `validate_injected_corpus.py` | a deterministic control classified as anything but `no_evidence` | **PASS**, 3 of 3 silent, FPR **0.0** |
| 14 | the static false-positive set is exactly the pinned literal set | `validate_injected_corpus.py` | a control joining or leaving the set | **PASS**, the set is empty |
| 15 | the two-arrangement classifier fires on the self-polluting check | `validate_injected_corpus.py` | the deliberate self-polluter not classified `self_polluting_check` | **control FIRED** |
| 16 | a numeric-only comparison is blind to an iteration-order change | `validate_injected_corpus.py` | the numeric comparison seeing it, or the skeleton comparison missing it | **PASS**, 0 moved, structural change seen |
| 17 | a text comparison misses a sub-precision change and a numeric one catches it | `validate_injected_corpus.py` | the rounded line moving, or the full-precision token not moving | **PASS**, 0 against 1 |
| 18 | no `pytest.approx` call relies on the hidden `abs=1e-12` default | `validate_approx_audit.py` | any call passing `rel=` without `abs=` | **PASS**, 0 of 38 |
| 19 | the whole suite passes with the hidden default forbidden | `validate_approx_audit.py` | any assertion depending on the default at run time | **PASS**, exit 0 |
| 20 | the offending control assertion passes **without** the plugin | `validate_approx_audit.py` | the control failing for the ordinary reason, which would prove nothing | **control FIRED** |
| 21 | the plugin refuses the same assertion | `validate_approx_audit.py` | the plugin not firing, or firing without its refusal message | **control FIRED**, exit 1 |
| 22 | the static screen parses every Python file in this product | `validate_self_audit.py` | any file failing to parse | **PASS**, 0 unparseable |
| 23 | this product's own scripts are not classified self-polluting | `validate_self_audit.py` | a verdict changing in the reused tree while the pristine pair agrees | **PASS**, 2 of 2 `no_evidence` |
| 24 | the audit's blind spot is measured, not asserted | `validate_self_audit.py` | the blind spot being zero, which would make the limitation unverifiable | **PASS**, **72 of 565**, **0.127434** |
| 25 | the audit's coverage, token count and contributing-script set on an 8-candidate subset are identical across two runs, and the headline drift is below a declared 0.05 | `validate_self_audit.py` | coverage or the script set moving, or the headline drifting by more than one in twenty | **PASS**. The headline is deliberately **not** asserted equal; see section 10 |
| 26 | the audit over a fixture, run twice, gives identical reports | `validate_self_audit.py` | any element of the report differing | **PASS** |
| 27 | the frozen evidence is byte-identical to the one the pins were taken from | `validate_repository_findings.py` | one byte of the evidence changing | **PASS**, sha256 `cafca4b3…589ef3` |
| 27b | the frozen finding set matches its pinned per-script counts and digests | `validate_repository_findings.py` | a finding appearing, disappearing or changing value | **PASS**, 23 checks in 0.2 s, 3 digests match |
| 28 | running the hygiene script adds no cache directory or bytecode file | `validate_tree_hygiene.py` | the exit snapshot exceeding the entry snapshot | **PASS**, 0 added |
| 29 | nothing a cold clone would carry is a cache or build artifact | `validate_tree_hygiene.py` | an artifact surviving the re-implemented ignore rules | **PASS**, 0 offenders |
| 30 | every required file exists and is not ignored; every scratch path is | `validate_tree_hygiene.py` | a required deliverable being excluded, or `junit.xml` being shipped | **PASS**, 16 and 6 |
| 31 | no tracked text file carries an absolute home or temp path | `validate_tree_hygiene.py` | any such path in any committed text file | **PASS**, 0 occurrences |
| 32 | every script returns the same exit status from four directories | `probe_working_directories.py` | a script that runs from the product root and not from `validation/` | **PASS**, 10 scripts x 4 directories |
| 33 | every script exits 0 from all four directories | `probe_working_directories.py` | any non-zero status anywhere | **PASS** |
| 34 | the probe changed no committed output file but its own | `probe_working_directories.py` | the probe overwriting measured evidence with fast-path numbers | **PASS**, 0 changed |
| 35 | this script finishes inside the budget it declares for itself | `validate_self_audit.py` | the script approaching the gate's 600 s ceiling unnoticed | **PASS**, 81.7 s against a declared 300 s, margin 518.3 s |
| 36 | every script in `validation/` and `examples/` run **in gate order**, under the gate's environment, with the gate's 600 s per-script timeout | gate replication | any script failing, or exceeding 600 s, or being affected by one that ran before it | **PASS**, 14 scripts, 0 failures, slowest 192.0 s, total 437.3 s |

## 1. The closed-form detection power

From `validate_power_model.py`, output in `validate_power_model_output.txt`.

### The headline constants

| quantity | value |
|---|---|
| `p = 1/44` | 0.022727272727272728 |
| `(43/44)^2` | 0.9550619834710744 |
| `P_ref(1/44, 2) = 1-(43/44)^2` | **0.044938016528925595** |
| `P_pair(1/44, 2) = 1-(1/44)^2-(43/44)^2` | **0.044421487603305776** |
| `p_max(2, 0.05) = 1-0.05^(1/2)` | **0.7763932022500211** |
| repeats for 95 per cent detection at 1/44 | **131** |
| `P_ref(1/44, 131)` | 0.950788448056 |
| `P_ref(1/44, 130)` | below 0.95, so 131 is minimal |
| `p_max(131, 0.05)` | 0.022608690169639987 |

**This is Baseline 1 refuted, derived.** Run-twice-and-diff has detection power
0.044421487603305776 at the rate this mission shipped: it finds the defect about
one time in twenty-two. A clean two-run report is therefore the *expected*
outcome and establishes only that the rate is below 0.776.

### The detection-power table, as published

| n | `P_ref` | `P_pair` | `p_min(n)` | `p_max(n)` clean |
|---|---|---|---|---|
| 2 | 0.044938016529 | 0.044421487603 | 0.776393202250 | 0.776393202250 |
| 3 | 0.066643970699 | 0.066632231405 | 0.631596850136 | 0.631596850136 |
| 5 | 0.108587139371 | 0.108587133307 | 0.450719728347 | 0.450719728347 |
| 10 | 0.205383111905 | 0.205383111905 | 0.258865550893 | 0.258865550893 |
| 20 | 0.368584001154 | 0.368584001154 | 0.139108340668 | 0.139108340668 |
| 44 | 0.636341050283 | 0.636341050283 | 0.065818771145 | 0.065818771145 |
| 100 | 0.899636012115 | 0.899636012115 | 0.029513049607 | 0.029513049607 |
| 131 | 0.950788448056 | 0.950788448056 | 0.022608690170 | 0.022608690170 |
| 200 | 0.989927069936 | 0.989927069936 | 0.014867039231 | 0.014867039231 |
| 500 | 0.999989816676 | 0.999989816676 | 0.005973551516 | 0.005973551516 |
| 1000 | 0.999999999896 | 0.999999999896 | 0.002991249545 | 0.002991249545 |

`p_min` and `p_max` are numerically one expression and are two claims. Their
worst absolute disagreement over the eleven budgets above is **0.0**, and
`P_ref(p_max(n), n)` recovers 0.95 with a worst absolute error of **0.0** on
every one of them, against a declared tolerance of 1e-12.

### The derivation is checked, not asserted

| check | result |
|---|---|
| Monte Carlo against both closed forms, 200000 audits per cell, 6 rates × 4 repeat counts | **PASS**, worst standardised deviation **3.3018** over 24 cells (48 comparisons) |
| two-term inclusion-exclusion bracket contains the closed form | **PASS**, 16 of 16 cells |
| exponential bracket from `log(1-p)` contains the closed form | **PASS**, 30 of 30 cells |
| first-order union bound sits above the closed form | **PASS**, everywhere on the grid |

**Weak result, reported:** the worst Monte Carlo deviation is 3.3018 standard
errors. Over 48 comparisons a maximum above 3.3 has probability about 0.045
under the null, so it is inside expectation but it is the weakest agreement in
the set and it is named rather than averaged away. The criterion was declared
at four standard errors before the run and was not moved.

The brackets matter because a closed form whose only evidence is the algebra
that produced it is a known failure mode of this mission: an earlier product's
777 tests, property tests included, could not catch catastrophic cancellation
because every test compared the series against expressions sharing the same
cancellation. **The inclusion-exclusion bracket is degenerate at `n = 2`** —
with two events `S1 - S2 = 2p - p^2 = 1 - (1-p)^2` exactly — so it carries no
information there and the check uses `n >= 3`. That degeneracy is stated in the
script and in the test rather than hidden by a cell that always passes.

The bracket slack follows the **conditioning** of the bracket expression, the
third inclusion-exclusion term `C(n,3) p^3`, and not the operation count and
not an absolute constant.

### The seed axis is a different axis

A quantity that is a deterministic function of the seed yields the same value
on every repeat at a fixed seed, so repeats carry **zero** information about
seed-dependence and the Bernoulli formulae do not apply. Seeds drawn without
replacement from a finite declared set give a hypergeometric count.

| population | sensitive | draws | hypergeometric | binomial (wrong) |
|---|---|---|---|---|
| 10 | 5 | 2 | 0.777777778 | 0.750000000 |
| 100 | 50 | 3 | 0.878787879 | 0.875000000 |
| 1000 | 999 | 1 | 0.999000000 | 0.999000000 |
| 44 | 2 | 5 | 0.216701903 | 0.207529564 |

The two differ, so the distinction is not cosmetic.

## 2. The classifier's own error rate — found by running it, then derived

Running the two-arrangement classifier over the injected corpus at two runs per
arrangement classified an **intrinsically non-deterministic script at rate 0.5
as a self-polluting check**: its two pristine runs happened to agree and its two
reused-tree runs happened to differ, which is the self-pollution signature. The
classifier was not wrong about what it saw. It is a statement about what two
runs per arrangement can distinguish.

The honest response was to quantify it rather than raise the repeat count until
the example stopped embarrassing the tool. For a script with intrinsic rate `p`
and **no self-pollution at all**, both arrangements are independent blocks of
`n` Bernoulli runs, so a false `SELF_POLLUTING` verdict has probability
`q(1-q)` with `q = p^n + (1-p)^n`.

| n | worst-case rate | worst-case probability | at p = 0.5 |
|---|---|---|---|
| 2 | 0.500000000 | **0.250000000** | 0.250000000 |
| 3 | 0.211324865 | **0.250000000** | 0.187500000 |
| 5 | 0.129462101 | **0.250000000** | 0.058593750 |
| 10 | 0.066967008 | **0.250000000** | 0.001949310 |
| 20 | 0.034063671 | **0.250000000** | 0.000001907 |

**Raising the runs per arrangement does not lower the worst case.** `q(1-q)`
peaks at 0.25 when `q = 0.5`, and `q = p^n + (1-p)^n` reaches 0.5 for every
`n >= 2`; more runs only move the rate at which the classifier is worst. The
mitigation is independent repetitions of the **whole** two-arrangement
experiment, which multiply: 0.25, 0.0625, 0.015625, 0.00390625 for one to four
trials. This is a limitation of the product and it is in `README.md` under
*Limitations*.

## 3. Baseline 3 — the injected corpus

From `validate_injected_corpus.py`, output in
`validate_injected_corpus_output.txt`. 1320 independent processes per rate and
4400 at the mission rate; audits are formed from **disjoint consecutive blocks**
of runs, never sliding windows, because overlapping audits share runs and would
inflate the apparent number of independent audits — which is the error this
product exists to report in other people's work.

### Calibration — is the ground truth actually the declared rate?

Six intervals at 95 per cent each all cover with probability
`0.95**6 = 0.7350918906249998`, so a single miss is the expected outcome about
a quarter of the time. The pass criterion is therefore the Bonferroni
family-wise interval at per-interval coverage
`0.9916666666666667`, declared in the script before the run. Both intervals
are printed.

| injected | runs | hits | realised | 95% lo | 95% hi | in95 | FW lo | FW hi | inFW |
|---|---|---|---|---|---|---|---|---|---|
| 1.000000 | 1320 | 1320 | 1.000000 | 0.997209 | 1.000000 | True | 0.995857 | 1.000000 | True |
| 0.500000 | 1320 | 636 | 0.481818 | 0.454550 | 0.509167 | True | 0.445281 | 0.518494 | True |
| 0.100000 | 1320 | 132 | 0.100000 | 0.084342 | 0.117464 | True | 0.079382 | 0.123698 | True |
| 0.022727 | 4400 | 117 | 0.026591 | 0.022040 | 0.031784 | True | 0.020617 | 0.033661 | True |
| 0.010000 | 1320 | 13 | 0.009848 | 0.005254 | 0.016782 | True | 0.004143 | 0.019483 | True |
| 0.001000 | 1320 | 1 | 0.000758 | 0.000019 | 0.004214 | True | 0.000003 | 0.005771 | True |

**An earlier full run of the same script had one per-rate interval miss, at
the lowest injected rate.** It is recorded here rather than forgotten, because it is the reason the
family-wise correction exists in the script. A separate diagnostic of **3000**
processes at that rate found exactly **two** distinct outputs and **4**
non-reference results against an expectation of 3.0, so there is no
contamination channel in the indicator: the misses are sampling outcomes at low
rates, which is what the correction accounts for.

**Clopper–Pearson rather than Wilson, and that choice was measured.** The first
version used a Wilson score interval and **failed its own calibration check at
the injected rate 1.0**: with 1320 hits in 1320 runs the Wilson interval does
not reach 1.0, because the score interval does not attain the boundary when the
proportion sits on it. Clopper–Pearson does, and it is the same family as this
product's own escaped-rate bound — `escaped_rate_upper_bound(n, alpha)` *is*
the Clopper–Pearson one-sided upper bound for zero successes — so the
calibration interval and the headline bound come from one piece of mathematics
rather than two.

### Measured detection power, `n = 2`, which is Baseline 1

| rate | blocks | measured `P_pair` | FW lo | FW hi | derived@injected | derived@realised | measured `P_ref` | derived@realised |
|---|---|---|---|---|---|---|---|---|
| 1.000000 | 660 | 0.000000 | 0.000000 | 0.010351 | 0.000000 | 0.000000 | 1.000000 | 1.000000 |
| 0.500000 | 660 | 0.509091 | 0.448588 | 0.569409 | 0.500000 | 0.499339 | 0.736364 | 0.731488 |
| 0.100000 | 660 | 0.175758 | 0.132846 | 0.225319 | 0.180000 | 0.180000 | 0.187879 | 0.190000 |
| 0.022727 | 2200 | 0.051364 | 0.038017 | 0.067477 | 0.044421 | 0.051768 | 0.052273 | 0.052475 |
| 0.010000 | 660 | 0.019697 | 0.007058 | 0.042492 | 0.019800 | 0.019503 | 0.019697 | 0.019600 |
| 0.001000 | 660 | 0.001515 | 0.000002 | 0.013835 | 0.001998 | 0.001514 | 0.001515 | 0.001515 |

(The `n = 2` rows of a 24-cell table that also covers `n = 5, 10, 20`; the
full table is in `validate_injected_corpus_output.txt`.)

**Two derived values are reported and they answer different questions.**
`derived@injected` uses the rate the corpus was built with, so comparing against
it tests the calibration **and** the detection mathematics together and a miss
does not say which failed. `derived@realised` uses the per-run rate actually
observed, so it tests the disjoint-block detection mathematics **alone**. The
pass criterion is `derived@realised`, because that is the claim this product
makes, and the `derived@injected` count is reported beside it.

**The percentile bootstrap is degenerate when no audit detected.** All resamples
of an all-zero indicator array have mean 0, so the interval is `[0, 0]` and
cannot contain a positive derived value. That is a property of the bootstrap at
the boundary, not evidence of disagreement. Both are reported: the derived value
lies inside the **exact family-wise** interval on **24 of 24** cells for both
designs, and inside the **per-cell 95 per cent bootstrap** interval on **22 of
24**.

### Baseline 1, the headline comparison

| quantity | value |
|---|---|
| injected rate | 0.022727272727272728 |
| independent processes | 4400 |
| disjoint audits of 2 runs | 2200 |
| **measured detection power** | **0.05181818181818182** |
| bootstrap 95 per cent interval | [0.042272727272727274, 0.06136363636363636] |
| exact 95 per cent interval | [0.03840947445363034, 0.06798977050158111] |
| **derived `1 - p^2 - (1-p)^2`** | **0.044421487603305776** |
| derived inside the bootstrap interval | True |
| derived inside the exact interval | True |
| reference-design derived value | 0.04493801652892562 |

A method that finds a 1-in-44 intermittent defect about one time in twenty-two
is the method every reproducibility checker uses.

**The strongest single statement this corpus makes about Baseline 1 is the
first row of the detection table.** At an injected rate of **1.0** the measured
pairwise power is **0** at every audit size, and the derived value is 0 too:
when every run manifests, every run agrees with every other, and a design with
no committed reference cannot see the change at all.

### Negative controls

| detector | controls | flagged | false-positive rate |
|---|---|---|---|
| dynamic, two-arrangement | 3 | 0 | **0.0** |
| static screen | 3 | 0 | **0.0** |

All three controls were asserted to stay silent and all three did. The static
set is **pinned as the empty literal set**, so the check fails for real if a
control joins it.

### The two cases a single comparison mode cannot see

| case | numeric comparison | skeleton / full-precision comparison |
|---|---|---|
| iteration order: the number is fixed, the printed order is not | 0 moved claims | structural change seen |
| sub-precision: the change is at the thirteenth decimal | 0 moved claims in the rounded line | 1 moved claim in the full-precision line |

## 4. The audit of this repository

Produced by the **campaign** `data/repository_audit.py`, transcript committed at
`data/repository_audit_output.txt`, itemised evidence committed frozen at
`validation/repository_findings.json`, verified by
`validation/validate_repository_findings.py`.

### Provenance of the frozen evidence

| field | value |
|---|---|
| evidence sha256 | `cafca4b3a4b63249a07229ed98531f4c27ebde4e12be1b81298d799cff589ef3` |
| evidence bytes | 2543180 |
| repository commit the campaign ran against | `0dfd86d583cc6e1f8db413c27694d59861e565b9` |
| date | 2026-11-16 |
| machine | x86_64 Linux, **2** cores reported by the operating system |
| per-run timeout | **75.0 s** |
| workers | **2** |
| campaign wall clock | **3339.8 s** |
| exclusions | `products/P094`, `products/P100` |

The commit is the one the mission coordinator had pushed when the campaign ran;
this product's own changes land as a follow-up commit, so the evidence
describes the corpus **as of that commit** and not as of this file. The
campaign writes to a temporary directory by default and will **refuse** to
write into `validation/` without an explicit `--refresh-committed`, because a
program that silently overwrites the file another script pins is the defect
this product exists to name. Section 9 records the occasion on which it did.

### The headline

| quantity | measured |
|---|---|
| **committed numbers claimed reproducible that MOVE under a seed change** | **5677** |
| scripts contributing at least one | 102, across 31 products |
| **committed numbers that move with the seed HELD AT ITS DECLARED VALUE** | **345** |
| scripts contributing at least one of those | 47, across 19 products |
| of those 345, on a line carrying a wall-clock duration marker | 89 |
| of those 345, with no such marker (an UPPER bound on non-timing) | 256 |
| numbers excluded as the audit's **own** perturbation echoed back | 45, across 33 scripts |

### Coverage, stated as a number

| quantity | measured |
|---|---|
| Python files parsed by the static screen | 3564 |
| candidates found | 229 |
| candidates attempted | 229 |
| global deadline reached | False |
| compared | 110 |
| no output produced — claim UNEVALUABLE, not clean | 73 |
| failed to run in a pristine copy | 37 |
| timed out at the 75 s per-run limit | 9 |
| committed numeric tokens across all candidates | 43321 |
| tokens the perturbed comparison evaluated | 21401 |
| repeats per claim on the repeat axis | 1 |
| escaped-rate upper bound at that repeat count | 0.95 |
| campaign wall-clock seconds | 3339.8 |

Of the 110 compared candidates, **8** produced no seed-attributable finding.

### The exclusions are part of the result

| excluded | reason |
|---|---|
| `products/P094/` | another build agent was writing it concurrently, so any number derived from it is not reproducible by a reader |
| `products/P100/` | this product; a tool scanning its own tree while it is being written is the class it exists to detect. It is audited separately in `validate_self_audit.py`, which publishes the findings against it |

### Findings against published products, which are published here

These are the products contributing the most moved numbers. **Every one of them
is a published product of this mission, and nothing in any of them was
modified.**

Seed-attributable, top ten:

| script | moved numbers |
|---|---|
| `products/P052/validation/validate_benchmark.py` | 1350 |
| `products/P035/validation/validate_matched_far.py` | 581 |
| `products/P071/validation/validate_screening.py` | 261 |
| `products/P077/validation/validate_predictors.py` | 199 |
| `products/P062/validation/validate_recovery.py` | 170 |
| `products/P068/validation/validate_headtohead.py` | 167 |
| `products/P068/validation/validate_calibration.py` | 153 |
| `products/P079/validation/validate_asymmetry_benchmark.py` | 148 |
| `products/P035/validation/validate_far_design.py` | 131 |
| `products/P079/validation/validate_allan.py` | 112 |

Moving with the seed held at its declared value — **the class that is a defect
rather than a property of a seed** — top ten:

| script | moved numbers |
|---|---|
| `products/P061/validation/validate_warmstart.py` | 64 |
| `products/P061/validation/validate_gap.py` | 62 |
| `products/P080/validation/validate_rate_distortion.py` | 49 |
| `products/P079/validation/validate_benchmark_table.py` | 38 |
| `products/P083/validation/validate_certificate.py` | 21 |
| `products/P083/validation/validate_resimulation.py` | 11 |
| `products/P061/validation/validate_predictor.py` | 10 |
| `products/P052/validation/validate_surrogate.py` | 8 |
| `products/P068/validation/validate_headtohead.py` | 8 |
| `products/P071/validation/validate_screening.py` | 8 |

The 19 products contributing at least one are P004, P005, P025, P035, P052,
P061, P062, P068, P070, P071, P072, P075, P076, P077, P079, P080, P082, P083
and P084. Inspection of the itemised evidence shows these are dominated by
wall-clock readings and quantities derived from them — `elapsed: <#> s`,
`wall clock : <#> s`, `<#> ms`, `<#> energy [W*h]`, `predictor fit : <#> s` —
printed into files the repository treats as reproducible evidence. **A duration
written into a committed evidence file is still a number that is not
reproducible, and grouping it is not dismissing it.**

### What this result does NOT claim

A number that moves under a seed change is **not automatically a defect**. A
Monte Carlo estimate quoted at a declared seed is reproducible at that seed, and
its moving under a different seed is the expected behaviour of a stochastic
estimate. The finding is narrower and still useful: that number is **a property
of the seed, not of the system**, so any sentence reading it as a property of
the system is unsupported by a single-seed run.

A candidate reported as compared with no moved numbers is evidence that **this
one** perturbed seed draw did not move it. It is not evidence that no seed does,
and it is not evidence about intermittency at a fixed seed: that is the repeat
axis, sampled **once** per arrangement here, bound **0.95**.

### The static self-pollution screen over the same corpus

| quantity | measured |
|---|---|
| Python files parsed | 3564 |
| files that would not parse | 0 |
| files with at least one finding | 53 |
| findings in total | 275 |
| `write_to_read_path` | 246 |
| `write_under_glob` | 16 |
| `write_under_read` | 13 |

The screen is conservative by construction: an unresolvable path expression
becomes a wildcard, and a vacuous expression — one that is entirely wildcard —
is **skipped**, because as a root it contains everything and as a path it lies
under everything, so a containment test involving it is true by vacuity.
Reporting that as a finding would be the screen manufacturing its own result.

An earlier version of the screen reported **441** findings over the same corpus.
That was not conservatism, it was a name collision: `str.replace` shares a name
with `Path.replace`, `json.dump(obj, handle)` takes an object rather than a
path, and `fig.savefig(path)` has a receiver that is not a path. Treating a
receiver as a path reported variable names as directories and buried the real
findings. It was fixed at the cause.

## 5. The self-audit

From `validate_self_audit.py`, output in `validate_self_audit_output.txt`.

### The static screen over this product's own tree

| quantity | measured |
|---|---|
| Python files parsed | 35 |
| files that would not parse | 0 |
| files with at least one finding | 5 |
| findings in total | 15 |
| path uses across this product | 171 |
| informative — the screen can reason about them | 168 |
| vacuous — the screen is blind to them | 3 |

The five files are `src/seedaudit/audit.py` (2), `src/seedaudit/dynamic.py`
(2), `tests/test_dynamic.py` (7), `validation/_bootstrap.py` (3) and
`validation/validate_tree_hygiene.py` (1). The sixth in the first version was
`validation/validate_repository_audit.py`, which no longer exists: it is now
`data/repository_audit.py` and is outside the scanned set.

**This product's validation layout is itself an instance of the class it
detects.** The build guide requires `validation/<stem>_output.txt` beside the
script, so every script here writes into the directory this product's own
hygiene check scans. It cannot be avoided, so it is handled the only honest
way: the `Tee` **deletes its target before writing**, so a run's output can
never be a merge of this run and the last; the hygiene check asserts what this
product **owns** rather than what merely exists; and the findings are printed
rather than excluded.

### The two-arrangement classifier on this product's own scripts

| script | pristine agree | reused agree | classification |
|---|---|---|---|
| `validation/validate_power_model.py` | True | True | `no_evidence` |
| `validation/validate_tree_hygiene.py` | True | True | `no_evidence` |

`validate_self_audit.py` excludes **itself** from the dynamic check, because
running it inside a pristine copy of this product would run it recursively.
That is the only exclusion and it is declared.

### A false-negative class in the audit's own candidate rule

Applying the rule to this product finds **0** candidates inside it. This
product declares its one seed as `SEED = 1000044` in `validation/_bootstrap.py`
and every script imports it; the candidate rule requires a seed literal in the
**script**. That is a false-negative class in the rule, not a property of this
product, and its size across the corpus is measured rather than asserted:

| quantity | measured |
|---|---|
| scripts with a committed output file | 565 |
| declaring a seed literal in the script — **candidates** | 229 |
| declaring no seed of their own while a sibling `_module` does | **72** |
| unparseable | 0 |
| **blind spot** | **72 of 565**, which is **0.127434** |

Resolving an imported constant would require the audit to execute or import the
code it audits, which is a different and larger design. The limitation is
published in `README.md` rather than reduced by widening the rule until it
guesses.

### The audit's own reproducibility on the real corpus

The campaign is **one** measurement and takes most of an hour, so its
between-run variation is not measured at full size here; section 10 records
the three full-corpus observations. What is measured here, and is re-runnable,
is a bounded subset of the same corpus audited twice back to back.

| quantity | run 1 | run 2 |
|---|---|---|
| candidates in the subset | 8 | 8 |
| headline | 110 | 110 |
| declared-seed | 3 | 3 |
| instrument footprint | 3 | 3 |
| coverage | compared 5, run_failed 3 | identical |
| evaluated tokens | 300 | 300 |
| contributing script set | identical | identical |
| headline drift | **0** of about 110, which is 0.000000 | — |

**The headline is deliberately not asserted to be identical.** Under the
release gate's own environment this check once measured 110 against 111 with
identical coverage, and the assertion was removed rather than the observation
explained away — see section 10. What is asserted is coverage, evaluated token
count and the contributing script set, plus a declared 0.05 ceiling on the
drift fraction. Any script whose finding count moves between the two runs is
**named** in the output, because such a script is one whose committed numbers
are not reproducible at its declared seed.

### This script's own margin against the release gate

| quantity | measured |
|---|---|
| declared budget for this script | 300.0 s |
| release gate ceiling | 600.0 s |
| measured elapsed | **81.7 s**, and 79.7 s and 81.9 s in the two gate-order sweeps |
| margin against the ceiling | **518.1 s** at the slowest of those |

## 6. The `pytest.approx` hidden-default audit

From `validate_approx_audit.py`, output in `validate_approx_audit_output.txt`.

`pytest.approx(expected, rel=r)` passes if **either** bound is met and the
absolute default is `1e-12`, so for a quantity of order 1e-5 every relative
tolerance tighter than 2e-8 is silently replaced.

| check | result |
|---|---|
| `approx` calls in `tests/`, from the syntax tree | 38 |
| calls passing both `rel=` and `abs=` | 38 |
| calls relying on the hidden default | **0** |
| whole suite re-run with a bare `rel=` made an error | **PASS**, exit 0 |
| control: the offending assertion passes **without** the plugin | **FIRED** |
| control: the plugin refuses the same assertion | **FIRED**, exit 1, refusal message present |

Both directions of the control are asserted. A control that fails for the
ordinary reason would prove nothing about the plugin, so the offending
assertion is chosen to pass on its own: `1e-15` against `2e-15` differ by
`1e-15`, inside the hidden default, so the `rel=1e-30` bound is silently
replaced.

## 7. Tree hygiene, and what this product does NOT assert

From `validate_tree_hygiene.py`, output in `validate_tree_hygiene_output.txt`.

The build guide narrows this check deliberately. A self-check asserts two
things and not the third:

- that running the script **adds** no cache directory and no bytecode file —
  the tree is snapshotted at entry and compared at exit;
- that nothing a cold clone would carry is a cache directory, bytecode file or
  build artifact;
- **not** that no such artifact exists, because release-gate check 2 runs
  `ruff check` with its working directory set to the product root and writes
  `.ruff_cache` into the tree one check before check 6 re-executes this script.
  Anything found that this script did not create is reported, **named as
  inherited**, and does not fail.

**No `git` command is run anywhere in this product.** The build agent is
forbidden from running `git`; the coordinator commits, gates and publishes. So
instead of asking `git check-ignore`, the script **re-implements the subset of
`.gitignore` matching the check needs** — anchoring, directory-only patterns,
basename-at-any-depth patterns, negation and last-match-wins — over both this
product's `.gitignore` and the repository's. The subset is declared in the
script's `IGNORE_SEMANTICS` constant; it does not cover `core.excludesFile`,
`.git/info/exclude`, per-directory `.gitignore` files below this product, or
the index itself. **The authoritative check is release-gate check 9, which runs
against the real index, and this one does not replace it.**

| check | result |
|---|---|
| artifacts **added** by running the script | **0** |
| files a cold clone would carry that are cache or build artifacts | **0** |
| required files that exist and are not ignored | **16 of 16** |
| scratch paths that are ignored | **6 of 6** |
| lines carrying an absolute home or temp path | **0** |

The count of artifacts **present at entry** is deliberately not quoted here. It
is whatever the harness left in the tree before the script ran — release-gate
check 2 writes `.ruff_cache` one check before check 6 re-executes this script —
and it is reported in the output file, named as inherited, and does not fail.
Quoting it as a fixed number would be this product asserting something it does
not own, which is the narrowing the build guide makes binding.

## 8. The working-directory probe

From `probe_working_directories.py`, output in
`probe_working_directories_output.txt`.

Three products in one batch of this mission failed the release gate after
passing every check that preceded it, because a validation script passed a bare
relative path to a loader. The build agent, the agent's self-report and the
coordinator's re-measurement all ran the script the same way, from the product
root; the gate runs it with `cwd=validation/`. Three independent verifications
agreeing does not help when they share an assumption.

Every script here is run from four directories — the filesystem root, the system
temporary directory, the product root and `validation/` — and the same exit
status is required from all four.

**Every probe runs in a pristine copy of this product, never in the tree
itself.** A probe that ran the scripts in place would overwrite every committed
`*_output.txt` with fast-path numbers, which is the self-polluting-check class
exactly: the probe's own execution would create the condition the rest of the
validation set is judged on. The script records the bytes of every committed
output file before the probe and compares them after it.

| quantity | measured |
|---|---|
| validation scripts probed | 6 |
| examples probed | 4 |
| directories each was run from | 4 |
| scripts returning the same exit status from all four | **10 of 10** |
| scripts exiting 0 from all four | **10 of 10** |
| committed output files before the probe | 7 |
| committed output files after the probe | 7 |
| committed output files changed by the probe, other than its own | **0** |

### A finding the probe produced about this product

The first full probe reported `validate_injected_corpus.py` exiting 1 from one
of the four directories and 0 from the other three. It was **not** a
working-directory defect: on the fast path that script draws two orders of
magnitude fewer samples, and its statistical checks can fail by chance. A
stochastic pass or fail makes the probe report a sampling outcome as a
working-directory defect, so those checks are now recorded as **UNAVAILABLE**
on the fast path rather than evaluated, with the reason printed. No number from
the fast path is reported anywhere in this file.

## 9. The campaign is not a validation script, and why that had to change

In the first version of this product the corpus-wide audit lived at
`validation/validate_repository_audit.py`. It failed the release gate twice
over, and both failures were structural rather than marginal.

**It exceeded the per-script ceiling by seven times.** `scripts/release_gate.py`
sets `TIMEOUT_SCRIPT = 600` and applies it per script in `check_scripts`. The
coordinator measured the audit at **4216 s**. No amount of load luck fixes that.

**It overwrote the evidence that the script next to it pins.** The gate runs
every script in `validation/` with `sorted(d.glob("*.py"))`, so
`validate_repository_audit.py` ran immediately before
`validate_repository_findings.py` and rewrote `repository_findings.json` —
the very file the second script exists to verify — from a tree that had changed
and under a load that was not the one the pins were taken under. **Fifteen of
twenty-six checks failed.** The two scripts contradicted each other by
construction and the gate's own ordering guaranteed it.

**That is the self-polluting-check class, arriving in this product's own
validation directory.** A check whose own execution creates the condition it
tests is not a check; here one script's execution destroyed another's
precondition. The product detects this class in other people's code and
shipped an instance of it in its own.

The fix is a reclassification, not a weakening. A 70-minute corpus-wide
execution campaign is a data-generation campaign:

| | before | now |
|---|---|---|
| the campaign | `validation/validate_repository_audit.py` | `data/repository_audit.py` |
| run by the release gate | yes, and it blew the ceiling | **no** |
| where its evidence goes | `validation/repository_findings.json`, always | a temporary directory by default; `--refresh-committed` is explicit |
| can it clobber the pinned evidence | yes, silently | **refuses**, with the reason printed |
| what verifies the evidence | nothing; the audit re-made it | `validate_repository_findings.py`, 0.2 s, executes nothing |
| integrity of the evidence | not checked | **sha256 pinned** |

Committing a campaign's output and verifying it is not a weaker claim than
re-running it. It is the same claim with its provenance stated, which is the
stronger form — and it is the pattern this mission already uses for a hardware
campaign in an earlier product.

## 10. The load-sensitivity finding, and what was de-pinned because of it

**A reproducibility auditor whose own findings move with machine load is
reporting load, not reproducibility.** That sentence is the finding.

Three independent full-corpus runs of the same code over the same repository:

| run | conditions | compared | timeout | headline | declared-seed | artifact |
|---|---|---|---|---|---|---|
| A | 2 cores, otherwise idle, during the build | 110 | 9 | 5683 | 345 | not committed |
| B | 2 cores, **both busy with another build**, independent re-execution by the mission coordinator | 106 | **18** | 5455 | 266 | reported by the coordinator |
| C | 2 cores, this session | 110 | 9 | **5677** | 345 | **the committed evidence** |

Two things in that table matter.

**The timeout set doubled in run B**, from 9 candidates to 18, with no change to
the repository at all. Those nine extra candidates took their findings with
them, which is most of the gap between 5683 and 5455.

**Runs A and C had identical coverage and still differ by 6 moved numbers.**
Coverage cannot explain that. The only remaining explanation is that an audited
script is itself intermittently non-deterministic at a fixed seed — which is
precisely what this product looks for, observed in its own measurement.

### What was de-pinned, and why

The first version pinned every quantity as an exact set. Run B showed which of
those pins were claims about the repository and which were claims about the
machine:

| quantity | run B reproduced it | pinned now |
|---|---|---|
| python files parsed, 3564 | **yes, exactly** | **yes** |
| files with a static finding, 53 | **yes, exactly** | **yes** |
| static findings, 275 | **yes, exactly** | **yes** |
| static subkinds `{246, 16, 13}` | **yes, exactly** | **yes** |
| static parse failures, 0 | **yes, exactly** | **yes** |
| candidates found, 229 | **yes, exactly** | **yes** |
| committed tokens, 43321 | **yes, exactly** | **yes** |
| `status_counts` | no | **no — reported with conditions** |
| the `timeout`, `no_output`, `run_failed` SETS | no | **no — reported with conditions** |
| `claim_tokens_compared` | no | **no — reported with conditions** |
| wall clock | no | **no — reported with conditions** |

Everything that reproduced exactly is produced by **reading and parsing files**.
The static screen executes nothing, and it is the part of the campaign that is
independent of machine load. Everything that moved depends on **executing** an
audited script.

The de-pinned quantities are now printed by `validate_repository_findings.py`
under *LOAD-SENSITIVE QUANTITIES — reported, deliberately NOT pinned*, together
with the conditions the campaign recorded, and the script lists them by name in
its `REPORTED_NOT_PINNED` constant so the de-pinning is visible rather than
inferred. **No tolerance was widened.** A pin that was a claim about the wrong
thing was removed and replaced with the true statement, which is a bound:

> At least **9** of **229** candidate claims did not complete within 75.0 s at
> the load the frozen campaign ran under. On a busier machine the figure was
> **18**. The headline of **5677** is likewise a **lower bound**: it counts
> moved numbers among the 110 claims that completed, and a claim that timed out
> contributes nothing, so a machine with more headroom finds at least as many
> and never fewer.

What *is* still checked is internal consistency, which is load-invariant: the
status counts sum to the candidates attempted, the unevaluable sets have the
sizes the status counts report, the evaluated token count does not exceed the
committed total, and the recorded escaped-rate bound equals the one recomputed
from the model.

### The same mistake, one level down, in this product's own self-check

`validate_self_audit.py` asserted that the audit's headline over a bounded
subset was **identical** across two back-to-back runs. Run under the release
gate's own environment, it measured **110 against 111** with identical coverage
and an identical evaluated-token count, and failed.

The assertion was wrong in the same way: it asserted that the **audited corpus**
is deterministic, which is the claim this product exists to refute. It was
removed. The script now asserts the part this product controls — coverage,
evaluated token count and the set of contributing scripts, all identical across
runs — reports the headline drift, **names any script whose finding count
moved**, and bounds the drift by a declared fraction of **0.05**, chosen because
a corpus whose findings moved by more than one in twenty between consecutive
runs would make a single run uninformative. Measured drift across four runs
since: 0, 0, 0 and 1 out of about 110.

## Pre-registered expectations, judged

`SPEC.md` section 8 was written and committed before the audit ran against this
repository. Each expectation is reported as it landed.

| # | expectation, as written in `SPEC.md` §8 | outcome |
|---|---|---|
| 1 | **Primary.** The headline count will be between 1 and 40 moved numbers, concentrated in fewer than 15 scripts. | **REFUTED.** Measured **5677** across **102** scripts. |
| 2 | **Secondary.** The count moving with the seed held fixed will be non-zero and dominated by wall-clock and timing fields rather than by seeded quantities. | **CONFIRMED on both halves.** 345, non-zero; inspection of the itemised evidence shows them dominated by durations and quantities derived from them. The line-local marker test recognises 88; the remaining 257 is an **upper** bound on non-timing findings, not a count of them. |
| 3 | **Secondary.** The static screen will flag at least 6 scripts, and more than half the flags will be scanner scripts writing their own `_output.txt` into a tree they scan. | **SPLIT: first half CONFIRMED, second half REFUTED.** 53 files flagged, well above 6. But the dominant subkind is `write_to_read_path` at **246 of 275**, not `write_under_glob` at **16**: the common pattern is a script writing then reading the same temporary path, not a scanner polluting a tree it scans. |
| 4 | **Secondary.** Baseline 1's measured power at `p = 1/44` will lie within the interval of the derived 0.044421487603305776 and below 0.06. | **CONFIRMED on both halves.** Measured 0.05181818181818182, exact interval [0.03840947445363034, 0.06798977050158111] contains the derived value, and 0.05181818181818182 < 0.06. |
| 5 | **Secondary.** The dynamic detector's false-positive rate on the negative controls will be 0, and the static detector's will be non-zero. | **SPLIT: first half CONFIRMED (0.0), second half REFUTED (also 0.0).** The screen is more precise than the prediction: `deterministic_config_adjacent.py` **reads a file** in the directory it writes to and does not **enumerate** that directory, and a write beside a file that was read is not containment under a scan root. The screen was not changed to produce this; the prediction was wrong. |

**Why the primary expectation was wrong, stated plainly.** It assumed that a
committed number moving under a seed change would be rare because the mission's
seed discipline is good. The discipline *is* good — the control run shows most
of these numbers are perfectly reproducible at their declared seed. What the
expectation got wrong is the denominator: most committed validation outputs in
this corpus are Monte Carlo estimates, and a Monte Carlo estimate is *supposed*
to move when the seed changes. The headline is therefore large and the
interpretation is the one in *What this result does NOT claim* above, not a
claim that 5683 defects were found.

## Every validation script and its exit status

All seven were executed on 2026-11-16 in the release gate's own sorted order,
under the gate's environment, with the gate's 600 s per-script timeout. All
seven exit 0. Each writes its raw output beside itself as `<stem>_output.txt`
and all seven output files are committed.

| # | script, in gate order | exit | sweep 1 | sweep 2 |
|---|---|---|---|---|
| 1 | `_approx_audit_plugin.py` (support module) | **0** | 0.1 s | 0.1 s |
| 2 | `_bootstrap.py` (support module) | **0** | 0.0 s | 0.0 s |
| 3 | `probe_working_directories.py` | **0** | 192.0 s | **200.4 s** |
| 4 | `validate_approx_audit.py` | **0** | 8.0 s | 8.1 s |
| 5 | `validate_injected_corpus.py` | **0** | 121.8 s | 123.5 s |
| 6 | `validate_power_model.py` | **0** | 0.2 s | 0.2 s |
| 7 | `validate_repository_findings.py` | **0** | 0.2 s | 0.1 s |
| 8 | `validate_self_audit.py` | **0** | 81.9 s | 79.7 s |
| 9 | `validate_tree_hygiene.py` | **0** | 0.1 s | 0.1 s |

The gate also runs `examples/`, in the same way and before `validation/`:

| script | exit | sweep 1 | sweep 2 |
|---|---|---|---|
| `_bootstrap.py` (support module) | **0** | 0.0 s | 0.0 s |
| `arrangement_classifier.py` | **0** | 1.3 s | 1.6 s |
| `injected_corpus_power.py` | **0** | 30.2 s | 32.2 s |
| `power_table.py` | **0** | 0.8 s | 0.7 s |
| `repository_audit_coverage.py` | **0** | 0.8 s | 0.7 s |

**Two sweeps, 14 scripts each, 0 failures in both.** Totals 437.3 s and
447.5 s; slowest single script 192.0 s and 200.4 s, against the gate's 600 s
ceiling. **These are wall-clock readings and they differ between sweeps** —
which is exactly the class of number this product counts **89** of in other
products' committed evidence, so two sweeps are given rather than one, and the
claim is stated as a bound: **the slowest script observed was 200.4 s, leaving
399.6 s of margin.**

The ordering is tested, not the scripts individually, because the ordering is
what exposed the clobber recorded in section 9. Both sweeps also verified that
the frozen evidence's sha256 was **unchanged** by the sweep, which is the
structural fix for that clobber.

**The campaign is not in this table and the gate does not run it.**
`data/repository_audit.py` took **3339.8 s** — more than five times the gate's
per-script ceiling, which is the first of the two reasons it had to move out of
`validation/`.

`validate_self_audit.py` declares its own 300 s budget and reports its margin
in its output, because an earlier version measured **473 s** on a machine with
both cores busy. That is inside the 600 s ceiling and not safely inside it, so
the bounded subset was cut from 24 candidates to 8 and the measured figure is
now 81.9 s.

## Reproducing every number

The exact commands are listed in `README.md` under *Reproducing every number*.
**There is no ordering constraint among the validation scripts**, which is the
point of section 9: the campaign that produces `repository_findings.json` is
`data/repository_audit.py`, it is not in `validation/`, the gate does not run
it, and it refuses to overwrite the committed evidence without
`--refresh-committed`. Every script in `validation/` reads that evidence and
none of them writes it.

`validate_injected_corpus.py` and `examples/injected_corpus_power.py` draw from
the operating system's entropy by design, so their **measured** figures move
between runs while the **derived** values they are compared against do not.

`data/repository_audit.py` is **not** reproducible to the digit; section 10 is
the measurement of that and the reason the load-sensitive quantities are no
longer pinned. Everything else here is deterministic under the single declared
seed `1000044`.

## Honesty statement

This software is research-grade. It is not flight-qualified, not certified and
not approved for operational aerospace use. No figure in this file comes from
anywhere but a script in this directory that was executed in the session that
reports it. Weak, null and refuting results are published, not retuned. No
other product in this repository was modified to make a finding go away.

Credits: This is under reserved rights obtained by OPTIMA Organisation.
