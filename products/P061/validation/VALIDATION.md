# activityplan — validation evidence

**Validation level 3.** Every number in this file and in `README.md`,
`MODEL_CARD.md` and `DATASET_CARD.md` was produced by a script in this directory,
executed on 2026-10-15 in the build container:

```
python           : 3.13.16  (Linux-6.18.44-fc-v114-x86_64-with-glibc2.39)
numpy            : 2.5.3
scipy            : 1.18.1
scikit-learn     : 1.9.1
matplotlib       : 3.11.2
pytest           : 9.1.1
hypothesis       : 6.169.3
ruff             : 0.16.8 on PATH, 0.17.0 as the installed package (both run, both clean)
cpu cores        : 2 reported by os.sched_getaffinity
```

The raw stdout of each script is committed beside it as `<script>_output.txt`.
Nothing below is quoted from memory, from a textbook, or from a previous run.

**This software is research-grade. It is not flight-qualified, not certified,
and not approved for operational aerospace use.** Everything validated here is
validated against a *declaration*: a plan this package calls feasible is
feasible with respect to the inputs you supplied, on the discretised timeline
you chose, under the resource model whose assumptions are listed in the module
docstrings.

---

## What was checked, against what, with what result

| # | Check | Reference | Result | Tolerance / criterion | Script |
|---|---|---|---|---|---|
| 1 | Closed-form solar integral | `scipy.integrate.quad`, adaptive, independent | worst relative difference **4.743e-10** over 8 profile/horizon pairs | < 1e-8 | `validate_energy.py` check 1 |
| 2 | Per-step closed-form integrals sum to the whole-horizon closed form | the whole-horizon closed form | worst absolute difference **1.776e-15 W\*h** over 5 instances | < 1e-9 W\*h | `validate_energy.py` check 2 |
| 3 | Midpoint sampling, smooth profile | the closed form | error ratios **4.003, 4.001, 4.000** under step halving | second order | `validate_energy.py` check 3 |
| 3a | Midpoint sampling, across the eclipse kink | the closed form | ratios **5.593, 6.848, 1.544** — not the second-order 4, and the last refinement gained almost nothing | reported, not bounded | `validate_energy.py` check 3 |
| 4 | **Battery energy balance over the timeline** | `(E_final - E_initial)` against `generated - consumed - shunted` | worst residual **2.274e-13 W\*h** over **21 traces** (5 instances × empty/greedy/exact, plus 6 sampled instances) | < 1e-9 W\*h | `validate_energy.py` check 4 |
| 5 | Fixed-load closed form against the per-step sum | the closed form | worst absolute difference **0.000e+00 W\*h** | < 1e-9 W\*h | `validate_energy.py` check 5 |
| 6 | **MILP optimum against exhaustive enumeration** | every assignment of start-step-or-not-placed, tested with the resource simulator | **9 instances, 0 disagreements**, worst objective difference **0.000e+00** priority units | difference ≤ 1e-9 | `validate_exact.py` |
| 6a | — candidates actually enumerated | — | 2401 to 24336 per instance | must be > 1000 | `validate_exact.py` |
| 7 | **Greedy against the exact optimum** | the proven optimum over 24 seeded instances | mean ratio **0.9511**, median 0.9555, **worst 0.8500**, best 1.0000, sd 0.0479 | reported, not bounded | `validate_gap.py` part 2 |
| 7a | — pooled shortfall | — | **4.97 %** | reported | `validate_gap.py` |
| 7b | — instances where greedy ties the optimum | — | **9 of 24** | reported | `validate_gap.py` |
| 8 | **Dominance lemma**: every MILP plan simulated | the resource simulator's fixed shunt and drain policies | **0 of 24** MILP plans rejected; 0 greedy plans rejected; 0 instances not proven optimal | all three zero | `validate_gap.py` |
| 9 | Learned duration against the analytic model, held out | the analytic model on the same rows | MAE ratio **0.9249** mean, range 0.8641–0.9968 over 5 seeds | ratio < 1 on every seed | `validate_predictor.py` check 1 |
| 10 | Learned energy against the analytic model, held out | as above | MAE ratio **0.4116** mean, range 0.3646–0.4591 | ratio < 1 on every seed | `validate_predictor.py` check 1 |
| 11 | Irreducible noise floor, duration | closed form `sigma*sqrt(2/pi)` against 200-draw Monte Carlo | **26.74746 s** against **26.83000 s**, relative difference **0.0031** | the two must agree | `validate_predictor.py` check 2 |
| 11a | — how much of the available gap the learned model closes | — | duration **51.1 %**, energy **86.1 %** | reported | `validate_predictor.py` check 2 |
| 12 | **Prediction-interval marginal coverage** | nominal 0.95 / 0.90 / 0.80, both conformal variants, both targets | **0 of 12** rows below nominal by more than three binomial standard errors | 3·sqrt(a(1-a)/400) band | `validate_predictor.py` check 3 |
| 13 | **Conditional coverage by duration tercile** | nominal 0.90 | **absolute variant covers 0.6782 on the long tercile, a shortfall of 0.2218**; normalised variant 0.8677, shortfall 0.0323 | reported, no pass criterion | `validate_predictor.py` check 4 |
| 14 | **Is there a search for a warm start to prune** | HiGHS node count, presolve on and off | **31 of 34** census solves have a node count of 1 | reported | `validate_warmstart.py` part 1 |
| 15 | Warm start where the solver does branch | the cold solve | nodes **263 → 1**, **2064 → 420**, **46 → 1**; time ratios 0.65, 0.24, 0.68 | optimum must be preserved | `validate_warmstart.py` part 2 |
| 16 | **Learned against analytic as the warm-start source** | each other, on the learned-prediction target problem | learned candidate verifies **8 of 8**, analytic candidate **5 of 8**; node count unchanged on every verified case | 0 optima lost | `validate_warmstart.py` part 3 |
| 17 | **Realised feasibility of the four planning modes** | one shared realisation per instance, 36 plans per mode | oracle **1.0000**, analytic **0.2500**, ml-point **0.4722**, **ml-robust 0.4167** | reported, no pass criterion | `validate_robust.py` check 2 |
| 18 | **Why the duration-robust plan is worse** | realised against planned data rate | ml-robust underbooks the data rate on **99.40 % (0.9940)** of activities, mean rate ratio **1.1911** | reported | `validate_robust.py` check 5 |
| 19 | Analytic duration model bias | 600 fresh draws | mean signed error **-16.7371 s**, optimistic on **67.83 % (0.6783)** of activities | reported | `validate_robust.py` check 6 |
| 20 | CLI exit statuses and vocabulary | 14 documented invocations plus `--help` | **0 failures**, 0 forbidden words | 0 | `validate_cli.py` |

---

## The energy check, in full

`validate_energy.py`. The spec this product was built against asks for "energy
conservation over the timeline checked against a closed-form integral of the
power profile". That is checked four ways, in increasing strength, because the
weak version is a tautology.

1. **Is the closed form right?** `SolarProfile.energy_wh` accumulates whole
   orbits analytically and handles the partial arcs at both ends separately, so
   it is not obviously correct. Checked against `scipy.integrate.quad` on four
   declared profiles at two horizons each: worst relative difference
   **4.743e-10**, against quad's own tolerance.

2. **Does the per-step decomposition agree with the whole-horizon form?**
   Different branches of the same function. Worst absolute difference
   **1.776e-15 W\*h** over the five shipped instances.

3. **What would sampling cost?** The package integrates the profile in closed
   form per step rather than sampling it. On a smooth profile the midpoint rule
   converges at second order — measured ratios **4.003, 4.001, 4.000** — but
   across the eclipse kink it does not: the measured ratios are **5.593, 6.848,
   1.544**, and at the 150 s step used by four of the five shipped instances the
   sampled estimate is **0.09919 W\*h** out against a horizon total of 207.98
   W\*h. A separate measurement at a coarser grid, pinned as a test, shows a
   refinement from 100 s to 50 s making the estimate *worse*. That is the
   reason for the design choice and it is pinned as a test
   (`tests/test_timeline.py::test_midpoint_sampling_does_not_converge_cleanly_across_the_eclipse_kink`).

4. **Does the battery recursion close?** `(E_final - E_initial)` against
   `effective_generated - consumed - shunted`, over 21 traces: five shipped
   instances under the empty, greedy and exact plans, plus six sampled
   instances under the greedy plan. Worst residual **2.274e-13 W\*h**.

---

## The exactness check, and the defect it found

`validate_exact.py`. Nine small instances, each enumerated exhaustively — every
assignment of a start step or not-placed to every activity, 2401 to 24336
candidates each, every candidate tested through the resource **simulator** — and
each solved by the MILP. Zero disagreements, worst objective difference
0.000e+00.

**This check earned its place during the build.** An earlier draft of
`activityplan.feasibility` treated the aggregate downlink volume — total data
produced against total declared contact capacity — as a hard constraint. The
MILP never had that constraint. On a four-activity instance the MILP returned
**6.5** priority units and the enumeration **5.0**.

The MILP was right. Data left on the recorder at the end of the horizon is not
an infeasibility; the physical limit is the recorder capacity, which is
evaluated step by step. Aggregate downlink volume is now a **diagnostic**, and
the operations-concept requirement that the recorder be empty at the horizon is
available as the declared option `Problem.require_empty_recorder`, which imposes
the strictly stronger end-of-horizon constraint. Both readings of the instance
are reproduced in `validate_exact_output.txt` (6.50 without the option, 5.00
with it) and pinned in `tests/test_regressions.py`.

A second declaration defect was found the same way: the first draft of the
`energy-bound` instance did not admit **the empty plan** — the fixed loads alone
broke the depth-of-discharge floor by 1.44 W\*h, so no plan was feasible and
every planner correctly returned nothing. `solve_exact` now names the
empty-plan binding constraint when a model is infeasible, and
`tests/test_instances.py` checks the condition for every shipped instance and
for the sampled family.

---

## The optimality gap

`validate_gap.py`, 24 instances from seed 9000, family defaults (14 activities,
2 orbits, 150 s step).

| statistic | greedy / exact |
|---|---:|
| mean | **0.9511** |
| median | 0.9555 |
| minimum | **0.8500** |
| maximum | 1.0000 |
| standard deviation | 0.0479 |
| 10th percentile | 0.8842 |
| instances where greedy ties the optimum | **9 of 24** |
| pooled `sum(greedy)/sum(exact)` | 0.9503 |
| pooled shortfall | **4.97 %** |

Cost: greedy 0.216 s mean, exact 0.635 s mean and 1.264 s worst, an exact/greedy
time ratio of **2.94** on this family. On the shipped `leo-imaging` instance the
gap is much larger — greedy 15.0 against an optimum of 23.0, a **34.78 %**
shortfall — which is why that instance is the worked example.

**The binding constraint the exact optimum leaves**, over the 24 instances:

| constraint | instances |
|---|---:|
| `horizon` | 12 |
| `recorder-capacity` | 10 |
| `battery-depth-of-discharge` | 2 |

`horizon` appears half the time and that is partly an artifact: the objective
gives no reward for finishing early, so an optimal plan may place an activity
arbitrarily late and leave a one-step horizon slack. The tightest **resource**
constraint is reported alongside it for exactly this reason, and over the same
24 instances it is `recorder-capacity` 22 times and
`battery-depth-of-discharge` twice.

---

## The learned component

`validate_predictor.py`, 5 dataset seeds, 1400 activities each, 400 held out,
160-tree forests.

### Accuracy, and the floor it should be read against

| target | analytic MAE | learned MAE | irreducible floor | gap closed |
|---|---:|---:|---:|---:|
| duration [s] | 31.60685 | **29.12174** | **26.74746** | 51.1 % |
| energy [W\*h] | 1.00330 | **0.40974** | **0.31358** | 86.1 % |

The duration floor is computed in closed form — the generator adds
`Normal(0, 0.08 · analytic_duration)` and the mean absolute deviation of a
centred normal is `sigma·sqrt(2/pi)` — and cross-checked by 200-draw Monte Carlo
over 150 requests: **26.74746 s** against **26.83000 s**, a relative difference
of 0.0031.

**The learned model wins because the analytic model is misspecified**, by
construction, in a way this repository declares in full in
`activityplan/features.py`: the generator adds a cold-settling term, a nonlinear
gain term and a heater term that the analytic model omits. The learned model is
also handed the analytic prediction as a feature, so it fits the correction
rather than relearning the model. **Neither of those facts is evidence about a
real spacecraft, and no number in this section should be read that way.**

### Marginal coverage: attained

| alpha | mode | target | nominal | mean coverage | min | max | mean width |
|---:|---|---|---:|---:|---:|---:|---:|
| 0.05 | absolute | duration [s] | 0.9500 | 0.9385 | 0.9250 | 0.9500 | 175.5717 |
| 0.05 | absolute | energy [W\*h] | 0.9500 | 0.9435 | 0.9300 | 0.9600 | 2.3987 |
| 0.05 | normalised | duration [s] | 0.9500 | 0.9540 | 0.9425 | 0.9750 | 156.3841 |
| 0.05 | normalised | energy [W\*h] | 0.9500 | 0.9555 | 0.9375 | 0.9650 | 2.2263 |
| 0.10 | absolute | duration [s] | 0.9000 | 0.8820 | 0.8550 | 0.9075 | 129.5453 |
| 0.10 | absolute | energy [W\*h] | 0.9000 | 0.8975 | 0.8900 | 0.9075 | 1.8155 |
| 0.10 | normalised | duration [s] | 0.9000 | 0.9025 | 0.8750 | 0.9475 | 125.6657 |
| 0.10 | normalised | energy [W\*h] | 0.9000 | 0.8980 | 0.8650 | 0.9150 | 1.7568 |
| 0.20 | absolute | duration [s] | 0.8000 | 0.7795 | 0.7150 | 0.8175 | 85.2116 |
| 0.20 | absolute | energy [W\*h] | 0.8000 | 0.8095 | 0.7950 | 0.8225 | 1.2666 |
| 0.20 | normalised | duration [s] | 0.8000 | 0.8190 | 0.7875 | 0.8375 | 96.1656 |
| 0.20 | normalised | energy [W\*h] | 0.8000 | 0.8030 | 0.7775 | 0.8150 | 1.3580 |

None of the twelve rows falls below nominal by more than three binomial standard
errors at n = 400.

### Conditional coverage: not attained by the constant-width variant

At alpha 0.10, duration, by tercile of the analytic duration:

| mode | tercile | mean coverage | mean width [s] | deviation from nominal |
|---|---|---:|---:|---:|
| absolute | short | 1.0000 | 129.5453 | +0.1000 |
| absolute | middle | 0.9669 | 129.5453 | +0.0669 |
| absolute | **long** | **0.6782** | 129.5453 | **-0.2218** |
| normalised | short | 0.9343 | 62.2834 | +0.0343 |
| normalised | middle | 0.9053 | 99.5930 | +0.0053 |
| normalised | **long** | **0.8677** | 215.5972 | **-0.0323** |

The absolute variant covers the longest third of activities **0.6782** of the
time against a nominal 0.90. This is not a defect in the implementation: split
conformal guarantees marginal coverage and nothing more, and the generator's
noise is heteroscedastic by declaration. It is the reason the normalised variant
exists, and the normalised variant does not fix it either — it reduces the
shortfall from 0.2218 to 0.0323 and does not remove it.

---

## The warm start: it does not help, except where it does

`validate_warmstart.py`. **`scipy.optimize.milp` exposes no MIP start**, so the
warm start here is a primal-bound cut: a verified feasible plan of value V
licenses the row `objective >= V`, which gives branch-and-bound the pruning
information a MIP start's incumbent would. It does not supply an integer
assignment to repair.

### Part 1: on most instances there is nothing to prune

Of **34** cold solves across the five shipped instances and twelve sampled ones,
with presolve on and off, **31 have a node count of 1**. HiGHS closes the gap on
the time-indexed relaxation at the root. **A warm start cannot speed up a search
that does not happen**, and reporting a speed-up from a node count of 1 in both
arms would be measuring presolve.

### Part 2: where the solver does branch, the cut is worth a lot

| instance | presolve | cold nodes | warm nodes | node ratio | cold s | warm s | time ratio | optimum kept |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `energy-bound` | on | 263 | **1** | 0.0038 | 1.508 | 0.979 | 0.6493 | yes |
| `leo-imaging` | off | 2064 | **420** | 0.2035 | 22.260 | 5.411 | 0.2431 | yes |
| `family-9005` | on | 46 | **1** | 0.0217 | 1.248 | 0.845 | 0.6768 | yes |

### Part 3: learned against analytic as the source of the warm start

Target problem built from the learned point predictions, 8 sampled instances:

| metric | greedy-ml | greedy-analytic |
|---|---:|---:|
| candidate verified feasible on the target | **8 of 8** | **5 of 8** |
| bound as a fraction of the optimum, mean | 0.9534 | 0.9765 |
| bound as a fraction of the optimum, minimum | 0.8696 | 0.8824 |
| optima lost to the bound | 0 | 0 |
| node count changed | no | no |

**The honest result: the learned warm start does not help the solver on these
instances.** The cold node count is 1, so neither source changes anything to
measure. What the learned predictor does buy is that its candidate plan
*verifies* on the target problem 8 times out of 8, where a plan built from the
analytic model fails verification 3 times out of 8 — once on
`recorder-capacity` by 10.1567 Mbit and twice on `battery-depth-of-discharge` by
2.81873 and 3.14506 W\*h. Without the verification gate those three would have
been applied as cuts on a problem they are infeasible for, which can remove the
optimum. The gate, not the predictor, is what makes the mechanism safe, and that
is reported here rather than dressed up as a speed-up.

On the one sampled instance in part 3 that does branch, `family-9003`, the cut
reduced the node count from **613 to 510** and *increased* the wall clock from
**3.460 s to 4.380 s**. That is in the committed output too.

---

## Robust planning: the result is negative, and the mechanism is measured

`validate_robust.py`, 12 instances × 3 realisations = 36 scored plans per mode.
A realised plan that violates any declared constraint scores zero, not partial
credit; the same rule is applied to all four modes.

| mode | feasibility rate | mean realised objective | mean activities placed | realised / oracle |
|---|---:|---:|---:|---:|
| oracle (unachievable) | 1.0000 | 28.083 | 8.333 | 1.0000 |
| analytic | 0.2500 | 5.500 | 8.667 | 0.1958 |
| **ml-point** | **0.4722** | **12.472** | 8.500 | **0.4441** |
| ml-robust | 0.4167 | 11.333 | 8.000 | 0.4036 |

**Planning against the conformal upper bound made things worse, not better.**
That is not a bug and it is not noise. The mechanism, measured in check 5:

| mode | mean realised rate / planned rate | activities whose rate was underbooked |
|---|---:|---:|
| oracle | 1.0000 | 0.00 % (0.0000) |
| analytic | 0.9473 | 23.81 % (0.2381) |
| ml-point | 1.0032 | 50.60 % (0.5060) |
| **ml-robust** | **1.1911** | **99.40 % (0.9940)** |

An activity's **data volume is declared, not predicted**, so its production rate
is `data_mbit / duration_s` — inversely proportional to the duration. Booking the
conformal **upper** bound on the duration books a **lower** production rate. The
realised duration is shorter, the real rate is higher, and the recorder fills
faster than the plan allowed for. **Planning against an upper bound on duration
is conservative for time and energy and anti-conservative for the recorder.**

The failure modes confirm it:

| mode | binding constraints of the infeasible plans |
|---|---|
| oracle | none |
| analytic | `battery-depth-of-discharge` ×22, `exclusion:act004\|act007` ×3, `recorder-capacity` ×2 |
| ml-point | `battery-depth-of-discharge` ×9, `recorder-capacity` ×7, `exclusion:act003\|act011` ×3 |
| ml-robust | **`recorder-capacity` ×21** and nothing else |

The analytic model underbooks **duration** and breaks the battery; it is
optimistic by **-16.7371 s** on average and on **67.83 % (0.6783)** of activities. The
duration-robust plan underbooks the **rate** and breaks the recorder. The two
mechanisms are opposite and neither is fixed by a better duration predictor.
`activityplan` does not claim to fix either; it measures both.

---

## Dependency bounds

The dependency ranges in `pyproject.toml` are **pinned to the minor version
actually executed**, because the mission's own cold-clone audit on 2026-10-13
found five of fifty published products broken or silently wrong by a floor with
no ceiling.

| dependency | declared range | version executed | how the range was chosen |
|---|---|---|---|
| `numpy` | `>=2.5.3,<2.6` | **2.5.3** | floor equals the executed version, ceiling excludes the next minor |
| `scipy` | `>=1.18.1,<1.19` | **1.18.1** | as above. `scipy.optimize.milp` and its HiGHS options are the whole exact solver |
| `scikit-learn` | `>=1.9.1,<1.10` | **1.9.1** | as above |
| `pytest` | `>=9.1.1,<9.2` | **9.1.1** | as above |
| `hypothesis` | `>=6.169.3,<6.170` | **6.169.3** | as above |
| `matplotlib` | `>=3.11.2,<3.12` | **3.11.2** | as above |
| `ruff` | `>=0.16.8,<0.18` | **0.16.8 and 0.17.0** | both ends executed; see below |

**What was and was not tested at both ends.** For `ruff` the container happens
to carry two installations — `0.16.8` on `PATH` as a user-local binary and `0.17.0` as
the pip package, reachable as `python -m ruff` — and `ruff check src/ tests/
examples/ validation/` passes clean under **both**, so that range was measured at
both ends. For every other dependency the declared floor **is** the executed
version and the ceiling is the next minor, so the range contains one measured
minor series and a single run covers it by construction. **No other version of
numpy, scipy, scikit-learn, pytest, hypothesis or matplotlib was installed or
tested**, and the ranges are tight for exactly that reason rather than being
tight on evidence that a wider range would break. This package commits no
pickled or otherwise serialised model artifact, so the scikit-learn
serialisation hazard that broke P001 does not apply here; the bound is still
pinned to the minor version because the validation numbers in this file would
move with a different forest implementation even if every test still passed,
which is the P025 failure mode.

The cost of a tight pin is real and is recorded in the README's Limitations: the
product cannot take a future security release of any of these without a port.

---

## Test suite

```
python -m pytest tests/ -q --junitxml=junit.xml
```

Counts parsed from the `testsuite` element of `junit.xml` and cross-checked
against a count of `testcase` elements:

```
tests 217 · failures 0 · errors 0 · skipped 0
testcase elements counted: 217  (cross-check: equal)
```

`junit.xml` is gitignored with a leading slash (`/junit.xml`), so the pattern
cannot match a fixture at any other depth. `git check-ignore -v` was run against
every file under `products/P061/`: the only matches are `__pycache__/`,
`.pytest_cache/`, `.ruff_cache/`, `.hypothesis/` and `/junit.xml`. **No shipped
source file, test, validation output, screenshot or document is ignored.**

Property-based tests use Hypothesis. `.hypothesis/` is gitignored, so every
counterexample Hypothesis produced during this build is pinned with its literal
inputs in `tests/test_regressions.py`:

- **greedy returns the empty plan when the declaration is infeasible**
  (`peak_power_w = 20.0`, `max_depth_of_discharge = 0.1`, `bus_load_w = 21.0`).
  The unconditional claim that the greedy planner always returns a feasible plan
  is false; the claim is conditional on the empty plan being feasible, and the
  docstring in `activityplan/greedy.py` now says so.
- **`energy_between_wh` refuses a reversed interval** (a 1 s segment and a split
  at 3.0 s). The property test now expresses the split as a fraction of the
  duration.
- The downlink-volume disagreement described above, both readings.

---

## Reproducing every number

From the repository root:

```bash
pip install -e ".[dev]"
python -m pytest tests/ -q --junitxml=junit.xml
ruff check src/ tests/ examples/ validation/
python -m activityplan --help

python validation/validate_energy.py
python validation/validate_exact.py
python validation/validate_gap.py
python validation/validate_predictor.py
python validation/validate_warmstart.py
python validation/validate_robust.py
python validation/validate_cli.py
python validation/worked_example.py

MPLBACKEND=Agg python examples/timeline_plot.py
MPLBACKEND=Agg python examples/feasibility_slack.py
MPLBACKEND=Agg python examples/instance_portraits.py
MPLBACKEND=Agg python examples/greedy_vs_exact.py
MPLBACKEND=Agg python examples/prediction_intervals.py
MPLBACKEND=Agg python examples/robust_tradeoff.py
MPLBACKEND=Agg python examples/warmstart_nodes.py
```

Every seed is a literal in the script that uses it; there is no environment
variable and no configuration file. The validation scripts write their raw
output beside themselves as `<script>_output.txt`, which is committed, and every
number in this file appears in one of those.

### Measured wall clocks in this build

| run | wall clock |
|---|---:|
| `python -m pytest tests/ -q` | 48.7 s |
| `validate_energy.py` | 3.8 s |
| `validate_exact.py` | 19.4 s |
| `validate_gap.py` | 23.7 s |
| `validate_predictor.py` | 73.8 s |
| `validate_warmstart.py` | 95.1 s |
| `validate_robust.py` | 38.6 s |
| `validate_cli.py` | 18.6 s |
| `worked_example.py` | 5.5 s |

All of them are inside the three-minute budget. **These are wall clocks on a
shared 2-core container; they move by tens of percent between runs and they are
not hardware characteristics.** The primary metrics everywhere — objective
values, node counts, coverage fractions — do not move at all at a fixed seed.

---

## What failed, and what was cut

- **Nothing in the validation suite is reported as failing.** Checks 13, 17, 18
  and 19 have no pass criterion because they are measurements of a trade, and
  three of them are unfavourable to the learned component: the constant-width
  conformal interval undercovers long activities by 0.2218, the robust plan is
  *worse* than the point plan on both feasibility and realised objective, and
  the warm start changes nothing on 31 of 34 solves.
- **The MILP refuses `charge_efficiency < 1`** rather than solving a different
  problem. The efficiency kink at zero surplus is piecewise linear and is not
  represented; the greedy planner and the simulator handle any efficiency. This
  is a documented limitation, not a defect.
- **No peak-power constraint is implemented.** The off-grid final step of an
  activity carries a reduced *average* power by construction, so a peak-power
  constraint would be wrong at that step without a finer treatment. Energy is
  exact; instantaneous power is not constrained.
- **No hardware measurement of any kind appears in this repository.** Nothing
  here is hardware-pending and nothing should be read as a hardware
  characteristic.
