# Validation — relfiledelivery 0.1.0

**Validation level 2.** Research grade. Not flight-qualified, not certified, not
approved for operational aerospace use.

Every number in this document was produced by a script in this directory,
executed in the session that wrote the document, on the environment recorded at
the top of each committed output file. Nothing here is carried from an earlier
session or estimated.

Layout is flat: each script `validation/<name>.py` writes
`validation/<name>_output.txt` beside itself. Every script anchors its paths on
`__file__` and changes to the repository root, and each one was executed from
`/`, `/tmp`, the repository root and `validation/` with exit 0 from all four.

## Environment

```
python           : 3.13.17
numpy            : 2.5.3
scipy            : 1.18.1
scikit-learn     : 1.9.1
pandas           : 3.0.5
matplotlib       : 3.10.8
joblib           : 1.5.2
pytest           : 8.4.2
hypothesis       : 6.140.3
ruff             : 0.14.2
cfdp-py          : not installed
spacepackets     : not installed
cpu cores        : 2
```

**`cfdp-py` and `spacepackets` are not installed.** They are named in the
README as the mature implementations of the protocol itself and **no number
anywhere in this repository was taken from either package**. There is no
comparison against them because there could not be one.

## Test suite

Counted from `junit.xml`, never from pytest stdout, and cross-checked against a
literal count of `testcase` elements.

```
python -m pytest tests/ -q --junitxml=junit.xml
```

| Quantity | Value |
|---|---|
| `testsuite` declared tests | 777 |
| `testcase` elements counted | 777 |
| failures | 0 |
| errors | 0 |
| skipped | 0 |

`ruff check src/ tests/ examples/ validation/` reports no findings.

## What was checked, against what, with what result

### 1. The closed form against independent derivations — `validate_analytic.py`

No simulation in this script. Four checks, none of which compares the closed
form against a remembered constant.

| Check | Reference | Result | Tolerance | Verdict |
|---|---|---|---|---|
| `E[R]` series | inclusion-exclusion over geometric minima, 48 `(n, p)` pairs | worst relative difference 1.258780e-13 | 1e-09 | PASS |
| series truncation | Bernoulli bound against the measured change when the tolerance tightens by 1e-06 | worst slack +7.297573e-16 over 9 pairs | 16·eps·E[R] | PASS |
| round-count CDF | independent product over units, 48 triples | worst absolute difference 2.220446e-16 | 1e-12 | PASS |
| transmission quantile | 200000 draws from `scipy.stats.nbinom`, 27 triples | worst absolute difference 0.0 transmissions | 1 transmission | PASS |

**This check failed twice before it passed, and both failures are recorded in
the committed output rather than in a commit message.**

The first run compared the truncation slack against a flat 1e-18 and reported
four of nine rows as failures at a slack of -1e-16. That is the rounding of the
subtraction that produces the measurement, not a violated bound: the
measurement is a difference of two partial sums of size about 2.5, so it
carries an error of order eps times the sum. The comparison was wrong, not the
bound.

With the tolerance corrected to 16·eps·E[R], one row still failed: `n = 1000`,
`p = 0.3`, slack **-1.83e-13** against a tolerance of 2.39e-14 — eight times
too large to be the subtraction. It traced to the series term itself. Written
as `1 - (1 - p**k)**n`, a small `p**k` rounds with an absolute error of half an
epsilon, the `n`-th power multiplies the *relative* error by `n`, and the final
subtraction from 1 leaves an absolute error of order `n * eps`, which at
`n = 1000` is 2e-13 — a thousand times the stated truncation tolerance of the
series itself. **The bound was not wrong; the arithmetic was.** The term is now
`-expm1(n * log1p(-p**k))`, the same function and not the same computation, and
the worst slack moved to +7.30e-16. `rounds_cdf` was changed the same way.

This is the defect of the build that matters most: it was invisible to 777
hand-written and property-based tests, all of which compared the series against
other expressions that shared the same cancellation, and it was found only by a
validation check that compared a computed quantity against a bound derived
independently of it.

### 2. The simulation against the closed form — `validate_simulation_vs_closed_form.py`

In the no-outage limit the closed form is exact, so it is the reference and the
simulation is what is being checked. **The tolerance is the sampling standard
error of the simulated mean, not a number chosen to pass**: four standard
errors, declared before the loop, giving a per-cell false-alarm probability of
6.3e-05.

| Check | Cells | Runs per cell | Worst discrepancy | Tolerance | Outside |
|---|---|---|---|---|---|
| deferred-NAK delivery time vs exact `E[T]` | 12 | 1200 | 1.9000 standard errors, 1.537594e-02 relative | 4 standard errors | 0 of 12 |
| forward transmission count vs `n/(1-p)` | 20 | 1200 | 1.2187 standard errors | 4 standard errors | 0 of 20 |
| immediate-NAK mean vs its busy-time lower bound | 11 | 400 | — | bound must hold | PASS |
| immediate-NAK idle vs one one-way light time, every run | 11 | 400 | -1.804e-16 s | 1e-09 s | PASS |

**The immediate-NAK bound check failed on its first run**, reporting seven of
eleven cells outside. The bound `8W/((1-p)R_fwd) + L` contains an *expectation*,
so it bounds the mean and not a realisation; a lucky run with fewer
transmissions than the expectation legitimately falls below it. The check was
comparing it against the minimum over runs. It now checks two separate
statements: the mean is at least the bound, and on every run the idle time is at
least one one-way light time, which is the per-realisation statement.

**Reported, not asserted: immediate-NAK is not always faster than
deferred-NAK.** Over the same eleven cells and the same seeds, the ratio of
median delivery times ranges from 0.8662 to 1.2425, and immediate-NAK is slower
in **1 of 11** cells — at 120 units, 25 per cent loss and a 2.5 s light time,
where the light time is 4.97 clean passes. Pipelining the retransmission
requests shortens the transfer when the forward stream is long relative to the
round trip and lengthens it when it is not, because a gap revealed late costs
its own two light times and with many gaps the sender idles once per gap
instead of once per round. This is a property of the declared model under M5,
not of any protocol, and it is reported because the opposite is the natural
assumption.

### 3. The completion invariant — `validate_completion_invariant.py`

The central claim of this package is that its numbers describe transfers that
actually delivered the file. The invariant — delivered bytes equal source bytes,
verified by SHA-256 — is checked **inside the simulator on every completed run**
and raises `CompletionInvariantError` on a mismatch. It is not a summary
statistic and a caller cannot get a delivery time without it.

| Check | Scope | Result |
|---|---|---|
| digest comparison on every delivered run | 4 files × 5 links × 3 modes × 40 seeds = 2400 runs | 1829 delivered, **0 mismatches** |
| no digest reported on an undelivered run | 571 undelivered runs | **0 violations** |
| the array exercises both branches | — | PASS, 1829 delivered and 571 not |
| **negative control**: one flipped bit must be caught | 3 modes × 5 byte offsets | **15 of 15 raised** |
| source payload reproducible from its seed | 4 files × 2 seeds | PASS |

The negative control is what makes the invariant evidence. A check that has
never been observed to fail is indistinguishable from a check that does nothing.
One of the five offsets is past the end of the file and wraps, so the hook
cannot silently no-op on an out-of-range offset.

The invariant is also asserted as a hard test, one pytest case per run, over an
independent array in `tests/test_invariants.py`: 4 links × 3 files × 3 modes ×
12 seeds = 432 parametrised cases, plus the corruption controls.

### 4. Retransmission accounting — `validate_retransmission_accounting.py`

Three routes to the same number.

| Check | Scope | Result |
|---|---|---|
| aggregate transmission counter vs per-unit attempt sum | 7 links × 3 modes × 80 seeds = 1680 runs | **0 discrepancies**, exact integer equality |
| sample mean transmission count vs `n/(1-p)` | 4 loss levels × 2 modes × 600 runs | worst 1.4971 standard errors, tolerance 4 |
| forward wire bytes recomputed from the attempt array | 21 link/mode cells × 40 runs | **0 mismatches**, exact integer equality |
| overhead ratio never below the lossless floor | same | PASS, floor 1.023440171 |

**Reported, not asserted: the one-over-duty-cycle inflation of the transmission
count over-predicts.**

| Link | Outage duty | Exact `n/(1-p)` | Measured mean | Measured/exact | Naive `1/duty` | Runs with no outage |
|---|---|---|---|---|---|---|
| bursty continuous | 0.7500 | 126.32 | 150.12 | 1.1885 | 1.3333 | 0.3550 |
| windowed bursty | 0.7143 | 126.32 | 149.77 | 1.1857 | 1.4000 | 0.5350 |
| growing NAK | 1.0000 | 141.18 | 141.29 | 1.0008 | 1.0000 | 1.0000 |

The naive factor over-predicts by 12.2 and 18.1 per cent, and the last column
says why: the clean-pass time of this file is about half a second while the
mean outage-free interval is one to two seconds, so a third to a half of runs
never meet an outage at all and the ones that do pay far more than their share.
An inflation factor derived from a duty cycle assumes the outage process is
fine-grained relative to the transfer; when it is not, the duty cycle describes
the long run and the transfer does not live in the long run. This is the same
failure the analytic delivery-time extension has, measured on the transmission
count instead of on the time.

### 5. Seeded reproducibility — `validate_reproducibility.py`

| Check | Scope | Result |
|---|---|---|
| every field of a repeated run | 3 links × 3 modes × 25 seeds = 225 runs, 23 fields each | **0 field mismatches** |
| a repeated Monte Carlo ensemble | 9 link/mode pairs × 150 runs | **0 mismatches**, array and quantile table |
| a repeated scenario table, target column included | 120 rows, 30 columns | identical |
| scenario construction alone | 80 scenarios | stable |
| independence from the surrounding call order | 9 pairs, interleaved with unrelated draws | **0 order-dependent results** |
| different seeds give different runs | 6 stochastic link/mode rows | PASS |

The call-order check exists because a simulator sharing one global generator
passes the first three checks and fails this one. Each run derives its
generators from its own seed through a `numpy.random.SeedSequence`.

**The seed-sensitivity check failed on its first run**, reporting three of nine
rows outside. Unacknowledged mode sends every unit exactly once whatever the
seed, so its transmission count is deterministic by construction and the
quantity that has to vary there is how many units *arrived*. The criterion was
wrong, not the simulator; the output now names which quantity each row is
judged on.

### 6. Distributions — `validate_distributions.py`

This script substantiates one claim: **a mean delivery time over an intermittent
link is close to meaningless.**

| Scenario | Mean (s) | Median (s) | Mean/median | Within 1% of the mean | Within 5% | Max/median |
|---|---|---|---|---|---|---|
| no outages, continuous | 4.5373 | 5.1676 | 0.8780 | **0.0000** | **0.0000** | 2.2978 |
| correlated outages, continuous | 7.0532 | 6.0142 | 1.1728 | 0.0240 | 0.1340 | 4.8386 |
| correlated outages, 6 contacts | 12.8544 | 11.6930 | 1.0993 | 0.0177 | 0.1571 | 4.5144 |

3000 runs per scenario. On the no-outage link **not one run of three thousand**
lands within five per cent of the sample mean: the distribution is quantised by
retransmission rounds and the mean falls in a gap between two modes. A uniform
distribution over the same support would put about two per cent of its mass in a
plus-or-minus-one-per-cent band around its mean.

Per-contact completion, 6 contacts, 3000 runs per mode:

| Mode | by pass 0 | by pass 1 | by pass 2 | by pass 3 | by pass 4 | by pass 5 |
|---|---|---|---|---|---|---|
| unacknowledged | 0.0010 | 0.0010 | 0.0010 | 0.0010 | 0.0010 | 0.0010 |
| deferred_nak | 0.2637 | 0.7873 | 0.9353 | 0.9870 | 0.9973 | 0.9997 |
| immediate_nak | 0.2590 | 0.7467 | 0.9270 | 0.9823 | 0.9967 | 0.9993 |

The unacknowledged row is flat because with no retransmission more passes do
not help. No single expected delivery time can express any of these rows.

**Reported, not asserted**: the semi-analytic delivery-time quantile is an
approximation and its worst relative error against the measured quantiles on
the no-outage scenario is **0.007348**. It is the reason the analytic interval
under-covers in check 7.

### 7. The learned predictor against the baselines — `validate_predictors.py`

Four predictors on 1127 held-out scenarios from a disjoint seed range. The full
table is in `MODEL_CARD.md`; the result is here.

| Regime | Rows | analytic | learned_quantile | mean_lookup | Winner |
|---|---|---|---|---|---|
| no_outage | 600 | **0.051928** | 0.104687 | 0.489155 | **analytic**, by a factor of 2.0160 |
| correlated_outage | 527 | 0.368928 | **0.292259** | 0.574066 | **learned_quantile**, by a factor of 1.2623 |

Median absolute relative error of the point prediction. **The specification for
this product predicted that the closed form would win where its assumptions
hold and that the learned model would earn its place only in the
correlated-outage regime, if at all. Both halves are measured above and both
are published as measured.** The closed form uses no training data.

The analytic baseline was strengthened before this comparison was published.
An earlier extension divided by the product of the contact and outage duty
cycles and was wrong by a factor of 6.14 on the worked-example scenario; with
it, the learned model's win in the correlated-outage regime was a factor of
1.9036 rather than 1.2623. Benchmarking against a straw man measures nothing
and the weaker baseline's figure is not the one reported.

**NOT CALIBRATED.** 10 of 12 predictor/regime rows fall outside the declared
0.03 calibration band at nominal 0.90, where the standard error of a coverage
estimate on 1127 rows is 0.0089.

| Predictor | Coverage at nominal 0.90 | Error | Median width / observed |
|---|---|---|---|
| analytic | 0.5093 | -0.3907 | 0.4327 |
| analytic_mean | 0.5093 | -0.3907 | 0.4327 |
| mean_lookup | 0.8784 | -0.0216 | 7.4233 |
| learned_quantile | 0.8146 | -0.0854 | 0.8662 |

Only the lookup table is within the band, and it pays with an interval 8.57
times wider than the learned model's. The closed form's interval is much the
worst because it combines two exact but *dependent* marginal quantiles at the
same probability, and the sum of quantiles is not the quantile of a sum.

**Reported**: the closed-form *expectation* falls outside its own 0.90 interval
on **2 of 1127** rows, worst excess 0.018720 relative. This is not a defect to
clamp away — the expectation and the quantiles are two different summaries of a
tail-heavy distribution with no containment relation between them — and the
sanity assertion was narrowed to the three predictors whose point prediction is
the 0.50 quantile of the same expression that produces their interval, with the
narrowing stated in the output.

**Reported**: 939 of 6762 adjacent quantile-level pairs (0.1389) had to be
sorted into order, because independent per-level fits do not guarantee
monotonicity.

### 8. The command line — `validate_cli.py`

7 invocations and 7 rejected inputs, **0 failures**. Run in process rather than
as a subprocess, because a subprocess is launched through an interpreter path
and an interpreter path on this machine is absolute, which no committed file in
this repository may contain.

### 9. The worked example — `worked_example.py`

The README's worked example, run so that the quoted output is the output. A
2 MiB file, 8 per cent loss, 2.5 s one-way light time, 100:1 asymmetry, outages
averaging 20 s at an 80 per cent duty cycle, six 30-second contacts every 180 s.

| Quantity | Value |
|---|---|
| closed form on the link with outages removed | 25.4821 s |
| contact time the extension needs | 31.8527 s |
| declared contact window | 30.0000 s |
| analytic extension | 181.8527 s |
| measured median, 600 runs | 26.7995 s |
| measured q0.95 | 205.1822 s |
| measured maximum / median | 20.8086 |
| complete by pass 0 | 0.8250 |
| complete by pass 1 | 0.9600 |

The extension is a step function of the contact plan and this scenario sits six
per cent on the wrong side of a step, so it deposits the whole transfer into the
second contact while the simulation says 82.5 per cent of runs finish in the
first. The arithmetic is not wrong; it has one number where the answer is a
distribution, and a distribution straddling a window edge is where that fails
hardest.

## Defects found during this build

Six, all fixed, all with the finding recorded where it was found.

| Defect | Found by | Fix |
|---|---|---|
| `1 - (1 - p**k)**n` loses `n * eps` to catastrophic cancellation; 2e-13 at `n = 1000`, a thousand times the series truncation tolerance | `validate_analytic.py` check 2 | `-expm1(n * log1p(-p**k))`; worst slack -1.83e-13 → +7.30e-16 |
| `LinkConditions` rejected a contiguous window list built by its own `uniform_windows`, as overlapping by 2.3e-13 s | Hypothesis, at `uniform_windows(11, 114.06854261351316, 114.06854261351316)` | named relative tolerance; pinned in `tests/test_regressions.py` |
| `contact_duty_cycle` returned 1.0000000000000002, which is not a probability and is used as a divisor | Hypothesis, same example | clamped, with the reason documented at the clamp |
| the truncation-bound check compared a slack against a flat 1e-18 | first run of `validate_analytic.py` | comparison tolerance 16·eps·E[R], stated per row |
| the immediate-NAK bound check compared an expectation bound against a minimum over runs | first run of `validate_simulation_vs_closed_form.py` | split into a mean check and a per-run idle check |
| the seed-sensitivity check judged unacknowledged mode on a quantity that is deterministic for it | first run of `validate_reproducibility.py` | per-mode criterion, named in the output |

Hypothesis was run at 700 examples per property over 26 properties before
release with no further falsifying example found. Every counterexample it did
find is pinned with literal inputs in `tests/test_regressions.py`, because
`.hypothesis/` is not committed and a falsifying example found by one run is
otherwise lost on a fresh clone.

## What was not checked

- **No comparison against `cfdp-py` or `spacepackets`.** Neither is installed
  in this environment and no number here was taken from either.
- **No comparison against a real link.** Everything is a statement about the
  declared model.
- **No conformance checking against CCSDS 727.0-B-5.** The transfer-mode names
  come from it and nothing else does.
- **No check that the Gilbert two-state process describes any real link.** It
  does not; it is the simplest process that produces correlated loss.
- **No calibration of the uncertainty outputs.** The measured shortfall is
  reported instead, and a conformal recalibration layer is not implemented.
- **No measurement on Python 3.11 or 3.12.** The package declares `>=3.11` and
  CI covers all three, but every number here was produced on 3.13.17.
