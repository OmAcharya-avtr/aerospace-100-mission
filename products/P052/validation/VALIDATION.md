# falsifyloop — validation evidence

**Validation level 3.** Every number in this file and in `README.md` was
produced by a script in this directory, executed on 2026-10-08 in the build
container (Python 3.13.16, NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1,
Hypothesis 6.168.5, 2 CPU cores reported by `os.sched_getaffinity`). The raw
stdout of each script is committed beside it as `<script>_output.txt`. Nothing
below is quoted from memory, from a textbook, or from a previous run.

**Falsification is one-sided: finding no violation is not evidence of
correctness.** Nothing in this file is a verification claim, and nothing in this
repository verifies anything.

---

## What was checked, against what, with what result

| # | Check | Reference | Result | Tolerance / criterion | Script |
|---|---|---|---|---|---|
| 1 | Robustness semantics against hand calculation | 10 hand-worked cases in the script, each shown with its arithmetic | 10/10 match | 1e-12 absolute | `validate_semantics.py` check 1 |
| 2 | Sign agreement, exhaustive | Boolean semantics, independently implemented | **44694 formula-trace pairs, 0 disagreements** | exact, no tolerance band | `validate_semantics.py` check 2 |
| 2a | — of which visited the hard cases | — | 32070 samples at exactly zero robustness, 16848 at infinite robustness | the claim is only interesting if these are nonzero | `validate_semantics.py` check 2 |
| 3 | Sign agreement on simulator traces | Boolean semantics | 2400 pairs over 8 instances, 156 violating, **0 disagreements** | exact | `validate_semantics.py` check 3 |
| 4 | Empty-window conventions | identities of `min`/`all` and `max`/`any` | `+inf`/`True` and `-inf`/`False` as required, 0 failures | exact | `validate_semantics.py` check 4 |
| 5 | Off-grid time bound refused | — | `ValueError` raised for 0.3 s against dt = 0.25 s; 0.5 s accepted | must raise, not round | `validate_semantics.py` check 5 |
| 6 | Positive normalisation cannot flip a verdict | — | 0 verdict flips over 480 draws at scales 0.1, 1, 10, 1000 | 0 flips | `validate_semantics.py` check 6 |
| 7 | Euler against the exact zero-order-hold solution | `scipy.linalg.expm` on the unsaturated linear loop | worst `theta` error **0.009032 deg** at the shipped dt = 0.005 s | first-order convergence | `validate_simulator.py` check 1 |
| 7a | — observed convergence order | halving dt | error ratios **2.034, 2.017, 2.008, 2.004** | ratio ≈ 2 for first order | `validate_simulator.py` check 1 |
| 8 | Euler self-convergence with both clips active | same scheme at dt = 0.000125 s | worst `theta` discrepancy **0.429770 deg** at the shipped dt; ratios 2.136, 2.118, 2.169 | reported, not bounded | `validate_simulator.py` check 2 |
| 9 | Declared actuator limits over the search box | 4000 uniform draws | worst deflection **19.614588 deg** against 20 deg; worst slew **120.000000 deg/s** against 120 deg/s; worst command 20.000000 deg | must not exceed | `validate_simulator.py` check 3 |
| 10 | Determinism | re-simulation | 500 draws, **0 non-reproducible** | bit equality | `validate_simulator.py` check 4 |
| 11 | **Do counterexamples survive a finer step** | same scheme at dt/4 | **22 of 304 violating draws flip** (7.2 %); worst flipped robustness **0.135888** | reported, not filtered | `validate_simulator.py` check 5 |
| 12 | Instance difficulty | 15000 uniform draws per instance at seed 52052, against a pilot at seed 13 | measured p from **0.001267** to **0.321933**, a 254-fold span | Clopper-Pearson exact 95 % | `validate_difficulty.py` |
| 13 | Tier labels against measured difficulty | the declared tier bands | **0 of 8** instances outside their band | all inside | `validate_difficulty.py` |
| 14 | Design target inside the re-measured interval | the pilot targets | **3 of 8 outside**, which is expected and explained | not a pass criterion | `validate_difficulty.py` |
| 15 | Baseline curve against its closed form | `1 - (1-p)^n` | **1 of 40** interval comparisons disagree | ≈5 % expected at 95 % intervals | `validate_benchmark.py` section 4 |
| 16 | Surrogate held-out accuracy | predicting the training mean | RMSE ratio **1.523 to 2.425** over the trivial predictor; R² 0.5631 to 0.8270 | must beat the trivial predictor | `validate_surrogate.py` check 1 |
| 17 | **Surrogate uncertainty coverage** | nominal 0.9500 and 0.6827 | measured **0.9356** mean (range 0.8950–0.9650) at 1.96σ and **0.7631** pooled at 1σ | reported as measured | `validate_surrogate.py` check 2 |
| 18 | Spread as a ranking signal | — | mean Spearman correlation **0.4853** across the suite; 0.6343 pooled | positive is the criterion | `validate_surrogate.py` check 3 |
| 19 | `n_jobs > 1` slower at small-batch inference | `n_jobs = 1` | single-row predict **8.6x slower**; three runs in this build gave 8.9x, 7.9x, 8.6x | the ratio is the claim; the absolute µs move 10-40 % | `validate_surrogate.py` check 5 |
| 20 | CLI exit statuses and vocabulary | 14 invocations plus `--help` | **0 failures**; no forbidden word in any output | 0 failures | `validate_cli.py` |

---

## The headline result: sample efficiency

`validate_benchmark.py`, 8 instances × 5 strategies × 30 seeds, budget 100
simulations, base seed 52000. The full per-instance table is in
`validate_benchmark_output.txt` section 1; the aggregate is section 5.

**Aggregate mean curve probability** — the average probability of having found a
violation over the whole 100-simulation budget, averaged unweighted over the
eight instances:

| strategy | aggregate mean P | vs baseline | instances the baseline beats it on |
|---|---:|---:|---:|
| uniform-random (baseline) | 0.6130 | — | — |
| latin-hypercube | 0.5229 | −0.0902 | 8 of 8 |
| simulated-annealing | 0.6097 | −0.0034 | 4 of 8 |
| cross-entropy | 0.2955 | −0.3175 | 8 of 8 |
| surrogate-guided | 0.8016 | **+0.1885** | 1 of 8 |

**Per-instance, on the two hardest instances** (`attitude-envelope`, measured
p = 0.003733; `rate-envelope`, measured p = 0.001267), median simulations to the
first violation, `> 100` meaning more than half the runs found nothing:

| strategy | attitude-envelope | rate-envelope |
|---|---:|---:|
| uniform-random | > 100 (12/30 found) | > 100 (2/30 found) |
| latin-hypercube | > 100 (8/30) | > 100 (2/30) |
| simulated-annealing | 60.0 (28/30) | 79.5 (20/30) |
| cross-entropy | > 100 (2/30) | > 100 (**0/30**) |
| surrogate-guided | **28.5** (30/30) | **51.0** (30/30) |

---

## Honest negatives, stated as findings

These are published because they are the result, not in spite of being the
result. None of them was retuned away.

1. **Latin hypercube never beats uniform random on any instance in this suite,
   and loses resolvably on two.** Point estimates: −0.0020 to −0.2523 in mean
   curve probability. The bootstrap resolves the loss on `attitude-envelope`
   (95 % interval [−0.3307, −0.0280]) and `nested-capture` ([−0.4224, −0.0907]);
   the other six are undecided at 30 seeds. The structural reason is in the
   method: a Latin hypercube's stratification guarantee is a property of the
   **whole** design, and a falsification run that stops at the first violation
   evaluates a small prefix of it, which has none of that guarantee. The
   stratification also costs correlation structure that uniform draws do not
   have to pay for.

2. **Cross-entropy is the worst strategy in the suite, by a wide margin, and
   loses resolvably on six of eight instances.** Aggregate 0.2955 against the
   baseline's 0.6130. On `rate-envelope` it found **nothing in 30 runs of 100
   simulations** and its best robustness over all 3000 simulations was
   **+0.0213**, never once negative. The structural reason: cross-entropy spends
   its early budget on a population it then fits a Gaussian to, and on an
   instance where the violating set is 0.13 % of the box, every early population
   is entirely satisfying, so the fitted distribution concentrates on the region
   of least-positive robustness — which on this problem is not where the
   violating corner is. The `min_sigma_fraction` floor stops it collapsing to a
   point but cannot make it look in the right place.

3. **Simulated annealing loses resolvably to uniform random on all four
   easy-to-moderate instances** (`overshoot-loose` −0.0657, `settling-band`
   −0.1263, `multi-requirement` −0.2023, `command-rate` −0.2050, all with 95 %
   intervals excluding zero) and only beats it resolvably on the single hardest
   one (`rate-envelope`, +0.2507, interval [+0.0983, +0.3823]). On an easy
   instance a random walk is strictly worse than independent draws, because its
   proposals are correlated and it wastes simulations re-examining a
   neighbourhood that a fresh uniform draw would have left.

4. **The surrogate's advantage is entirely on the hard instances, and on the
   four easiest the comparison is undecided.** It beats the baseline resolvably
   on `overshoot-tight` (+0.2370), `attitude-envelope` (+0.4420),
   `nested-capture` (+0.2973) and `rate-envelope` (+0.4670), and the other four
   comparisons straddle zero. Its one point-estimate **loss**, on
   `settling-band` (−0.0050, interval [−0.0397, +0.0320]), is reported here
   because the reporting rule is that ties are not wins.

5. **On the easy instances the surrogate is not a learned method at all.** It
   spends its first 16 simulations on a Latin hypercube warm start. On
   `overshoot-loose` all 30 of its violations came during that warm start, and
   on `settling-band` 28 of 30 did. The learned part only gets a turn where the
   baseline is already struggling, which is also where it wins.

6. **The surrogate's reported uncertainty is not calibrated, and the number that
   looks good is misleading.** Nominal-95 coverage averages 0.9356, close to
   nominal. Nominal-68 coverage is 0.7631 pooled — well *above* the 0.6827 a
   Gaussian gives. Both at once means the error is not Gaussian around the
   forest mean: the spread is conservative through the body of the distribution
   and too thin in the tails, and the near-nominal 95 % figure is two errors
   partly cancelling. It is reported as a ranking signal, which it measurably
   is (Spearman 0.4853), and never as a confidence interval.

7. **The surrogate's sample-efficiency gain is paid for in CPU time.** One
   simulation costs 0.31–0.37 ms; one forest refit on 120 points costs 19–27 ms,
   roughly 60–90 simulations' worth, and scoring 256 candidates costs about 9–14
   simulations' worth. At the shipped `refit_every = 8` the overhead is roughly
   15–25 simulations' worth of wall clock per simulation bought. The ranges are
   over the runs in this build; a single figure would be stale by the next one. On a
   simulator cheaper than this one the trade goes the other way, and the curves
   count simulations rather than seconds precisely so that this is visible.

8. **7.2 % of counterexamples do not survive a four-times-finer integration
   step.** 22 of 304 violating uniform draws flip to satisfying at dt/4, with a
   worst flipped robustness of 0.135888. A counterexample with a margin smaller
   than that should be re-simulated before it is believed. This is a property of
   the benchmark's cheap explicit-Euler simulator, and it is reported rather
   than filtered out, because the search paid a simulation for each of those
   draws and they appear in its curve.

9. **Half the per-instance win/loss ledger is statistically undecided at 30
   seeds.** 14 of 32 strategy-instance comparisons have a bootstrap interval on
   the difference that straddles zero. Running the grid at a different base seed
   moves exactly those entries: at base seed 7300 with 20 seeds,
   `examples/per_instance_comparison.py` reports Latin hypercube losing to the
   baseline on 2 instances rather than 8, and the surrogate on 0 rather than 1.
   A per-instance ledger at this repeat count resolves a large difference and
   nothing smaller, and anyone quoting a single such table as settled is
   over-reading it.

10. **The bootstrap band is degenerate where the curve is flat at 0 or 1.** 11 of
    40 of the band-width values in the closed-form consistency check were
    exactly zero, because the percentile bootstrap of a sample whose runs all
    agree cannot produce any spread. This is why that check was rewritten to use
    an interval-overlap test instead, and why the bands in section 3 of
    `validate_benchmark_output.txt` must be read as estimates of run-to-run
    spread rather than as coverage statements at a curve's floor or ceiling.

---

## Errors made during this build, and corrected

Recorded rather than quietly fixed, as the build guide requires.

1. **A wrong prediction about the surrogate's uncertainty, written into the
   module docstring before it was measured.** `src/falsifyloop/surrogate.py`
   originally asserted that the tree spread "systematically understates the
   error" and that its coverage would be "well below 95 %". The measurement said
   the opposite: the mean spread is 1.05 to 1.37 times the mean absolute error
   on every instance, and nominal-95 coverage is 0.9356. The docstring was
   rewritten to the measurement, and now says explicitly that the earlier
   prediction was wrong. The substantive conclusion — that the spread is not a
   confidence interval — survives, but for a different reason than the one
   originally given: the interval shape is wrong, not its scale.

2. **A wrong claim about convergence in the saturated regime.**
   `validate_simulator.py` check 2 originally said convergence would be "slower
   than first order" because the clip switching instants move with the step. The
   measured ratios when the step halves are 2.136, 2.118 and 2.169, which is
   first order or slightly better. The text was corrected to the measurement and
   the point that survives — that the absolute error in the saturated regime is
   about fifty times the unsaturated one — is now stated as the finding instead.

3. **A wrong hand calculation in a known-answer test.**
   `tests/test_requirements.py::test_nested_eventually_always_known_answer`
   originally expected 0.0 for `eventually[0,1] always[0,0.2] y <= 2` on the
   triangle trace. The correct answer is 2.0, attained at the last sample, where
   the inner window has been clipped to a single point. The test now carries the
   full index-by-index calculation in a comment and additionally asserts that
   `check_horizon` rejects that formula-trace pair, because the maximum is
   attained for a vacuous reason. The error was in the test, not the code, and
   finding it is what the clipped-window convention needs a test for.

4. **The closed-form consistency check was initially wrong, and reported 17 of
   40 failures.** It compared the closed form against the percentile bootstrap
   band, which is degenerate at a flat curve, so most of those "failures" were
   an artefact of the comparison and not of the harness. Rewritten as an
   interval-overlap test between a Clopper-Pearson interval on the empirical
   count and the closed-form interval implied by the Clopper-Pearson interval on
   p, it reports 1 of 40, which is the expected rate for 95 % intervals. Both
   the defect and the fix are described in the script's own section 4.

5. **A README code block quoted an output that had not been produced.** The
   "first run" section originally showed `falsify --instance rate-envelope
   --budget 100 --seed 0` printing `NO VIOLATION FOUND` with a minimum
   robustness of `+0.111672`. That command, actually run, finds a violation at
   simulation 16 with robustness `-0.048916`; the quoted figure corresponded to
   nothing. It was caught by re-running every quoted command as the final build
   step. The block now shows the budget-5 invocation whose verbatim output is
   committed in `validate_cli_output.txt`, and the figure it quotes is that
   file's. **This is the exact failure the "every number comes from a script you
   ran" rule exists to prevent, and it happened once in this build.**

6. **A plotting defect in `examples/requirement_monitor.py`.** The unsatisfied
   shading was drawn with data-coordinate y limits under an axes-coordinate
   transform, so it covered the wrong vertical extent. Fixed to use axes
   coordinates. No number depended on it.

---

## Checks that did NOT fail but that a reader should not over-read

* **Section 4's single disagreement** (`overshoot-tight` at n = 50) is within
  what 95 % intervals produce by chance across 40 comparisons. It is reported as
  a disagreement anyway rather than explained away.
* **The 3 of 8 design targets outside their re-measured interval** are not
  defects. The targets came from a 40000-draw pilot at a different seed and were
  rounded to convenient bounds; the re-measured column is the one quoted
  everywhere else. The tier labels, which are the claim that matters, are all
  inside their bands.
* **Every instance's measured p is an estimate.** `rate-envelope`'s is based on
  19 violations out of 15000 draws, with a 95 % interval of
  [0.000763, 0.001977] — a factor of 2.6 wide. Anything derived from it inherits
  that width.

---

## Compute budget

| run | simulations | wall clock | note |
|---|---:|---:|---|
| `validate_semantics.py` | 2400 simulator traces + 44694 enumerated pairs | 4.1 s | the enumeration dominates |
| `validate_simulator.py` | ~24000 | 7.0 s | |
| `validate_difficulty.py` | 120000 | 51.0 s | 425.1 µs per simulation |
| `validate_benchmark.py` | 64000 + up to 120000 in searches | 62.0 s | the longest run in the repository |
| `validate_surrogate.py` | 2560 | 7.8 s | |
| `validate_cli.py` | ~1500 | 3.7 s | |
| `worked_example.py` | ~2500 | 2.2 s | |
| `examples/` (six scripts) | ~110000 | 72.8 s total | |

Every single run finishes well inside the three-minute budget. The container
reports **2 available CPU cores** (`os.sched_getaffinity`), everything runs in
one process with no parallelism, and scikit-learn forests are fit and queried
with `n_jobs = 1`. **These are wall clocks on a shared container. They are not
hardware characteristics, they move by 10–20 % between runs, and no figure in
this repository presents a timing as a property of any hardware.** The primary
metric everywhere is a seeded simulation count, which does not move at all.

---

## Reproducing every number

From the repository root, with `pip install -e ".[dev]"`:

```bash
python -m pytest tests/ -q                    # 282 tests
ruff check src/ tests/ examples/ validation/

python validation/validate_semantics.py       # sign agreement, 6 checks
python validation/validate_simulator.py       # numerics, 5 checks
python validation/validate_difficulty.py      # instance difficulty
python validation/validate_benchmark.py       # the headline, 8 sections
python validation/validate_surrogate.py       # the learned component, 5 checks
python validation/validate_cli.py             # exit statuses and vocabulary
python validation/worked_example.py           # the README's worked example

MPLBACKEND=Agg python examples/sample_efficiency_curves.py
MPLBACKEND=Agg python examples/per_instance_comparison.py
MPLBACKEND=Agg python examples/robustness_landscape.py
MPLBACKEND=Agg python examples/surrogate_uncertainty.py
MPLBACKEND=Agg python examples/counterexample_trace.py
MPLBACKEND=Agg python examples/requirement_monitor.py
```

Every seed is fixed in the scripts. The benchmark uses base seed 52000, the
difficulty measurement seed 52052, the surrogate split seeds 4100 and 9400, and
the examples seeds 7100, 7300, 5 and 0. Re-running any script reproduces its
committed output exactly, except for the wall-clock lines, which move.
