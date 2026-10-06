# Validation — coderateopt 0.1.0

**Validation level 2 (research grade).** Everything below was produced by
running the scripts in this directory in the build session of 2026-10-06, on
Python 3.13.16 with NumPy 2.5.3 and SciPy 1.18.1, on two shared cores
contended with four sibling build agents. Each script's raw stdout is committed
beside it. Nothing here is flight-qualified, certified or approved for
operational aerospace use.

Where a check undercuts the package, it is in this document with the number
that undercuts it. Two of them are load-bearing: the greedy heuristic
essentially wins on a well-behaved MODCOD table, and the integer variables in
the MILP do no work on the base formulation.

## What is being validated, and what cannot be

This package computes a decision from a *model* of a link: a marginal fade
distribution, a set of MODCOD thresholds, a margin and an availability target.
The validation therefore covers

* the **internal identities** of the fade models (moments, normalisations,
  survival functions) against their defining equations and against Monte Carlo;
* the **correctness of the optimisation** against two independent
  implementations, one of which calls no solver at all;
* the **behaviour at the boundaries** — infeasibility, ties, non-monotone
  goodput, degenerate dwell fractions;
* the **sensitivity machinery**, by checking that a reported stability interval
  really is stable inside and really does change outside;
* the **gap to the rules of thumb** this package competes with.

It does **not** and cannot validate that the fade model describes any real
atmosphere, that the MODCOD thresholds describe any real modem, or that the
outage approximation to coded performance is accurate near threshold. Those
are inputs and modelling choices, stated in `coderateopt/problem.py` and in the
README's Limitations.

## 1. Fade models — `validate_fade_moments.py` → `fade_moments_output.txt`

| Check | Reference | Result | Tolerance |
|---|---|---|---|
| `sigma_lnI = sqrt(ln(1+sigma_I^2))` | Andrews & Phillips 2005 | re-derived independently in the script at 5 scintillation indices; max deviation below 1e-14 | abs 1e-14 |
| `sigma_dB = (10/ln10) sigma_lnI` | same | `sigma_dB = 1.8543995855453355` at `sigma_I^2 = 0.2` | abs 1e-13 |
| Lognormal mean of `10 log10 I` under `E[I] = 1` | same | `-0.39590623023812405` dB; negative, as the normalisation requires | abs 1e-12 |
| `E[I] = 1` by Monte Carlo, n = 400000 | — | within 4 standard errors at `sigma_I^2` = 0.05, 0.2, 0.6 | \|z\| < 4 |
| Lognormal availability, closed form vs Monte Carlo | — | 9 comparisons, all within 4 binomial standard errors | \|z\| < 4 |
| Gamma-gamma pdf total mass | Al-Habash, Andrews & Phillips 2001 | 1.0 to better than 1e-7 at 8 (sigma_I^2, ratio) pairs | abs 1e-7 |
| Gamma-gamma `E[I]` | same | 1.0 to better than 1e-7 at the same 8 pairs | abs 1e-7 |
| Gamma-gamma scintillation index from quadrature moments vs `1/a + 1/b + 1/(ab)` | same | agrees to better than 1e-5 relative | rel 1e-5 |
| Gamma-gamma survival vs a product-of-gammas Monte Carlo, n = 300000 | the model's own construction | 9 comparisons, all within 4 binomial standard errors | \|z\| < 4 |
| Gamma-gamma vs lognormal in weak turbulence | two distinct models | **reported, not asserted**: worst absolute availability difference over the block is printed in the raw output | — |
| Empirical survival on a hand-written sample | — | exact at all 6 probe levels; `1/n` resolution floor confirmed | exact |
| `exceedance(quantile(p)) == p` | — | 8 cases across both parametric models | abs 1e-7 |

The gamma-gamma survival function was **wrong when first written**: integrating
the pdf from 0 to a large upper limit in one `quad` call misses the mass near
the mode and silently returns a near-zero answer. The fix integrates the
shorter side of the mode, and
`test_gamma_gamma_upper_tail_is_not_silently_zero` guards against the
regression. The defect and its cause are recorded in
`GammaGammaFade.survival_linear`.

## 2. Solver agreement — `validate_milp_vs_exhaustive.py` → `milp_vs_exhaustive_output.txt`

Three implementations on the same seeded random instances (2 to 7 MODCODs,
lognormal or empirical fade, margin 0–25 dB, target 0.05–0.995, K 1–4, both
availability modes, minimum dwell drawn from {0, 0, 0.05, 0.2, 0.4}):

* `solve_milp` — `scipy.optimize.milp`, binary selection variables, linking
  rows, cardinality row, dwell rows;
* `solve_exhaustive` — `itertools.combinations` over subsets, one plain LP per
  subset, **no integer variables anywhere**;
* `solve_closed_form_k2` — vertex enumeration in NumPy, **no solver call**.

| Quantity | Result |
|---|---|
| Instances generated | 3000 |
| Infeasible in both, agreed | 784 |
| Feasibility disagreements | **0** |
| Feasible instances compared | 2216 |
| `milp` vs `exhaustive` mismatches | **0** |
| Closed form applicable (effective K ≤ 2) | 1456 |
| Closed form vs `exhaustive` mismatches | **0** |
| Worst relative goodput difference | **0.000e+00** |
| Optimal support sizes | 2178 single-entry, 38 two-entry, 0 larger |
| Mean solve time | `milp` 3.48 ms, `exhaustive` 11.20 ms |

Agreement tolerance was rel 1e-7 or abs 1e-9; the observed worst difference is
exactly zero, which is a stronger statement than the tolerance.

**This cross-check found two real defects, both now handled.** Neither was
visible from a single implementation.

## 3. The two solver defects

### 3.1 The HiGHS default MIP gap — `validate_mip_gap.py` → `mip_gap_output.txt`

`scipy.optimize.milp` does not set `mip_rel_gap`, so HiGHS stops when the
incumbent is within its default relative tolerance of the bound. The goodput
error that buys is negligible; the *decision* error is not.

| Quantity | Default gap | `mip_rel_gap = 0.0` (this package) |
|---|---|---|
| Support differences from enumeration, random instances | **3 of 481 feasible** | **0** |
| Objective shortfalls > 1e-9 | 3 | 0 |
| Worst relative shortfall | **6.8651e-05** | 0 |

The named instance, carried in `coderateopt.milp.MILP_OPTIONS`: illustrative
table, margin 12.00 dB, target 0.990000, K = 2, long-run,
scintillation index `0.5281008782676444`.

| Arm | Goodput, bits/symbol | Support |
|---|---|---|
| Enumerated optimum | 0.495018082635 | `ook-r1/2` + `ook-r2/3` |
| Default gap | 0.495001474400 | `ook-r1/2` |
| `mip_rel_gap = 0.0` | 0.495018082635 | `ook-r1/2` + `ook-r2/3` |

Relative difference 3.3551e-05. A uniform sweep of the scintillation index
shows **0** differences, because the grid steps over the narrow bands where a
mix is only marginally better; the frequency above comes from random search,
and the raw output carries both arms so the difference in method is visible.

### 3.2 Primal feasibility slack on a tight availability row

HiGHS accepts an allocation that violates the availability row within its
primal feasibility tolerance, and on a tight row that violation buys goodput
that is not available. Instance: illustrative table, margin 15.50 dB,
scintillation index 0.05, target 0.990000, K = 2, long-run.

| Arm | Achieved availability | Goodput, bits/symbol |
|---|---|---|
| Raw HiGHS allocation | 0.989999983417085 (short by **1.66e-08**) | 1.43572616018188 |
| True optimum (enumeration) | 0.990000000000000 | 1.435725437609653 |
| `solve_milp` after the polish step | 0.990000000000000 | 1.435725437609653 |

Relative goodput excess of the raw allocation: **5.0e-07**. `solve_milp`
therefore uses HiGHS to choose the *support* and re-solves the continuous part
exactly on that support with `coderateopt.lp.restricted_lp`. Guarded by
`test_raw_highs_allocation_can_violate_the_availability_row` and
`test_polished_solution_is_exactly_feasible_on_a_grid`.

## 4. Sensitivity — `validate_sensitivity.py` → `sensitivity_output.txt`

A reported stability interval is only worth something if it is checked.

| Check | Result |
|---|---|
| Support unchanged at 25 interior points of each reported interval, 5 instances | **0 failures** |
| Support changed 1e-3 past every *resolved* endpoint | **0 failures** |
| Grid instances evaluated (margin 9–22 dB in 0.5 dB steps × 5 targets) | 135 |
| With both boundaries resolved (not scan-bounded) | 84 |
| Relative width: min / median / max | **0.251465 / 0.757324 / 3.368164** |
| Knife-edge at ±10 % | **24 of 84** |
| Ratio of widest to narrowest | **13.39** |

Two instances of the same nine-entry table, in full:

| | knife-edge | flat |
|---|---|---|
| Margin | 14.00 dB | 22.00 dB |
| Availability target | 0.9990 | 0.9900 |
| Chosen MODCOD | `ook-r5/6` | `qpsk-r3/4` |
| Stable over (scintillation index) | **[0.154590, 0.204883]** | **[0.081641, 0.755273]** |
| Relative width | **0.251465** | **3.368164** |
| Headroom down / up | 0.227051 / **0.024414** | 0.591797 / **2.776367** |
| Knife-edge at ±10 % | **True** | False |
| Answer just below / above | `ook-r9/10` / `ook-r3/4` | `qam16-r3/4` / `qpsk-r1/2` |
| Margin stability interval | **[13.927, 14.726] dB**, width 0.800 dB | [17.911, 23.709] dB, width 5.798 dB |

The knife-edge case tolerates a **2.44 %** overestimate of the scintillation
index before the selected MODCOD changes; the flat case tolerates **278 %**.
Both come from the same table and differ only in operating point.

Two honest caveats, both reported by the code rather than left to the reader:
a signature change narrower than one grid step can be missed, and an interval
that reaches the scan bound is a lower bound, flagged by
`bounded_below_by_scan` / `bounded_above_by_scan`.

## 5. Degenerate and adversarial cases — `validate_degenerate.py` → `degenerate_output.txt`

17 checks, **0 failures**.

| Case | Required behaviour | Result |
|---|---|---|
| Infeasible instance | raise on every path, never return the least-bad MODCOD | `solve_milp`, `solve_exhaustive`, `solve_closed_form_k2` and `select_rate` all raised `InfeasibleProblem` |
| Infeasibility diagnostic | name the best achievable availability, its MODCOD and a sufficient margin | all three present; **the margin named in the message was checked to make the instance solve** |
| Exact three-way tie | deterministic canonical choice plus the full tie list | canonical support `(a,)`, alternatives `(b,)` and `(c,)` reported; **1 distinct support over 20 repeated solves** |
| Constraint satisfied only by the lowest rate | select it | `m1` at goodput 1.0, the lowest rate in the table |
| Non-monotone goodput over the rate set | find the interior peak | optimum `c` at 3.6; **greedy "fastest that closes" takes `e` at 1.6, a relative loss of 0.5556** |
| Zero-availability MODCOD with the largest rate | never select it | `useful` selected; `hopeless` (rate 1000, availability 0) ignored |
| Minimum dwell above 0.5 | collapse the cardinality limit to 1 | `effective_k == 1`, single entry, target still met |

The non-monotone case is the one that matters: a MODCOD table where goodput
rises then falls is exactly where the greedy rule fails, and it fails by more
than half.

## 6. The rules of thumb — `validate_heuristic_gap.py` → `heuristic_gap_output.txt`

Grid: margins 6–24 dB in 1 dB steps × 5 scintillation indices × 5 availability
targets. 430 feasible points, 45 infeasible for every MODCOD.

| Rule | Constraint-respecting choices | Matched the optimum | Loss (mean / max) | Choices that MISS the target |
|---|---|---|---|---|
| `highest_feasible_rate` ("fastest that closes") | 430 | **429 (0.9977)** | 0.000002 / **0.000865** | **0** |
| `highest_rate_within_reserve` (3 dB fixed reserve) | 238 | 201 (0.8445) | 0.037651 / **0.496837** | **192 of 430** |

**This is the result that decides whether the package is worth installing, and
it is not flattering.** On a well-behaved MODCOD table the greedy rule is
right 99.77 % of the time and never misses the availability target, because it
checks it. Its single loss on this grid is 0.0865 %, at margin 10.0 dB,
scintillation index 0.05, target 0.9, where `ook-r5/6` at goodput 0.820904
beats `ook-r9/10` at 0.820194 — a difference no link budget would act on.

The fixed-reserve rule is a different matter. It never looks at the fade
distribution, so **192 of 430** of its choices sit below the availability
target they were given. `examples/heuristic_gap.py` reproduces this on a finer
margin grid at a single target: **99 of 200**. That is a specification miss,
not a throughput loss, and it is the failure this package's constraint check
actually prevents.

## 7. The two readings of availability — `validate_mode_divergence.py` → `mode_divergence_output.txt`

| Check | Result |
|---|---|
| `per_interval` and `long_run` agree at K = 1 | **684 of 684** feasible grid points identical |
| Answers that differ at K = 2 | **511 of 684 (0.7471)** |
| Goodput gain from `long_run`: mean / median / max | 0.113952 / **0.046195** / **0.977107** |
| Worst-interval availability shortfall: mean / median / max | 0.070091 / **0.020729** / 0.476618 |
| Every divergence trades goodput for worst-interval availability | confirmed, no exceptions |

The largest divergence: margin 13.00 dB, scintillation index 0.4, target 0.999.
`per_interval` selects `ook-r1/4` at goodput 0.249994 with worst-interval
availability 0.999977. `long_run` mixes `ook-r1/4` (x = 0.020938) with
`ook-r1/2` (x = 0.979062) for goodput 0.494266 — a **97.71 %** gain — at a
worst-interval availability of 0.998979. Unavailability during the worst
interval rises from 2.286e-05 to 1.021e-03, a factor of **44.65**.

At the documented operating point (margin 12.00 dB, scintillation index 0.2,
target 0.99): `per_interval` gives `ook-r3/4` at 0.746408 bits/symbol;
`long_run` gives 0.660014 of `ook-r3/4` plus 0.339986 of `ook-r5/6` at 0.770151
bits/symbol, a 3.18 % gain, with worst-interval availability 0.979885 against a
0.99 target.

Nobody writing "99 % availability" in a requirement has usually said which of
these they meant. The two differ by up to a factor of 44 in the unavailability
seen during the worst part of the operating cycle.

## 8. Do the integer variables do anything? — `validate_support_size.py` → `support_size_output.txt`

Two structural results are claimed in `coderateopt/problem.py` and checked by
brute force here:

* in `per_interval` mode the optimum never exceeds the best single allowed
  entry — mixing in a second entry moves time to a strictly lower `R_m A_m`,
  which a linear objective never rewards;
* in `long_run` mode with no minimum dwell the optimum needs at most two
  entries, because the linear programme has two active constraints and
  therefore a basic optimal solution with at most two non-zeros.

A minimum dwell fraction breaks the second argument, so a second, targeted
search looks for a three-or-more entry optimum under one.

| Search | Instances | Feasible | Result |
|---|---|---|---|
| Wide random (3–7 MODCODs, both modes, K 1–4, dwell in {0, 0.05, 0.1, 0.2, 0.3}) | 1200 | 909 | `per_interval` optimum never exceeded the best single entry (**0 violations**); `long_run` with no dwell never exceeded two entries (**0 violations**); largest support found **2** |
| Targeted (one very-high-availability low-rate entry plus three faster ones, dwell > 0, K 3–4) | 2000 | 1922 | **13 instances with a three-entry optimum** |

The largest such instance, printed in full in the raw output:

```
rates          [0.3151 2.6378 5.5532 6.2442]
thresholds dB  [-3.381  2.351  3.982  9.308]
availabilities [0.993083 0.75069  0.566959 0.068128]
target 0.756620  K 3  dwell 0.1
mix m0 x=0.100265, m1 x=0.799735, m2 x=0.100000
goodput 1.929836415
best pair  ('m0', 'm2') at 1.886390780
best single 'm0' at 0.312888822
```

The three-entry optimum beats **every** pair by **4.345e-02 bits/symbol**
(2.30 % relative), verified by enumerating all pairs with
`restricted_lp`. `m0` and `m2` are each pinned at the dwell floor purely to
satisfy the availability row while `m1` carries the rate; no two-entry mix can
reproduce that, which is exactly the indicator implication
`x_m > 0 => x_m >= d` doing work that no linear programme expresses.

**So the honest reading is split, and both halves are in the README.** On the
base formulation (`min_dwell_fraction = 0`) the binary variables changed
nothing in 909 feasible instances, and the honest alternative is one line of
NumPy for `per_interval` and an `O(M**2)` closed form for `long_run`. With a
minimum dwell the MILP is doing real work on **0.68 %** of the targeted
instances, and the closed form is not merely slower there but wrong.

## 9. Worked example — `worked_example.py` → `worked_example_output.txt`

The example printed in the README is produced by this script, so the two
cannot drift. It is the knife-edge instance: margin 14.00 dB, target 0.999,
`ook-r5/6` at 0.832272 bits/symbol with achieved availability 0.99912568, a
unique optimum, stable over scintillation index [0.154590, 0.204883], with the
long-run reading buying 0.374 % more goodput at a worst-interval availability
of 0.996517.

## 10. Examples — `example_*_output.txt`

Each example's stdout is committed, so the screenshots in the README can be
tied to numbers:

* `knife_edge_vs_flat` — the two instances of §4, with their intervals;
* `sensitivity_map` — 37 margins × 50 scintillation indices; all 9 MODCODs
  appear as the optimum somewhere; 522 of 1850 grid cells infeasible;
  relative stability width min 0.251660, median 0.819141, max 2.743359;
  **8 of 37** margins knife-edge at ±10 %;
* `goodput_vs_target` — 140 targets; 21 mode-comparison points; **0 points
  with a positive goodput gain at a non-positive availability shortfall**,
  asserted by the script;
* `heuristic_gap` — 200 grid points; greedy loss identically 0; fixed-reserve
  choices below the 0.99 target at **99 of 200**.

## 11. Test suite

`python -m pytest tests/ -q --junit-xml=...` on the build container:

| tests | failures | errors | skipped | wall clock |
|---|---|---|---|---|
| **152** | **0** | **0** | **0** | 24.6 s |

Composition: unit tests for every module; input-validation tests for every
public constructor; known-answer tests in `tests/test_known_answers.py` whose
arithmetic is shown in the comments (including a hand-solved two-entry mix
with an exact tie between two different supports); Hypothesis property tests
asserting three-way solver agreement and the solution invariants over 600
generated instances; degenerate-case tests mirroring §5; and CLI tests that
run `python -m coderateopt` in a clean subprocess.

## 12. Compute budget

2 shared cores and 7.8 GiB, contended with four sibling build agents. Test
suite 24.6 s. Validation scripts, measured in this session:
`validate_fade_moments` 2 s, `validate_degenerate` 2 s, `worked_example` 2 s,
`validate_heuristic_gap` 4 s, `validate_sensitivity` 35 s,
`validate_milp_vs_exhaustive` about 60 s, `validate_mip_gap` 71 s after
trimming its sweep, `validate_mode_divergence` about 70 s,
`validate_support_size` 60 s. Examples: 4 s to 22 s each. Nothing needs
more than one core and nothing exceeds the 3-minute budget.

## 13. What failed, and what was cut

* **The gamma-gamma survival function was wrong on first writing** (single
  `quad` call from 0 to a large limit; silently near-zero). Found by a
  quantile inversion that could not bracket its root. Fixed and regression
  tested.
* **The MILP disagreed with enumeration twice**, both times caught by the
  cross-check rather than by a test written in advance: the HiGHS default
  `mip_rel_gap` (§3.1) and primal feasibility slack on a tight availability row
  (§3.2). Both are handled in the library and documented at the handling site.
* **A known-answer test was written with wrong arithmetic** (a minimum dwell of
  0.45 was predicted to destroy a two-entry mix; it only shrinks it, from
  goodput 1.8 to 1.775). The test now carries the corrected derivation.
* **`validate_mip_gap.py` initially ran 182 s**, over the 3-minute budget under
  contention. Its sweep was reduced from 401 to 221 points and its random
  search from 900 to 500 instances. The qualitative result is unchanged; the
  counts in §3.1 are from the committed run of the trimmed script.
* **No cross-check against another product in this batch was performed.** The
  batch specification assigns none to P050, and nothing here computes a
  quantity another product in Batch 05 computes from the same seeded sample
  path.
* **`pulp` was not used**, because `pulp.listSolvers(onlyAvailable=True)`
  returns `[]` in this container. It is named in the README's alternatives
  table as a modelling front end that needs a solver this environment lacks.
  `cvxpy`, `pyomo` and `mip` exist on PyPI (versions checked with
  `pip index versions`) but are not installed here and were **not**
  benchmarked; the alternatives table says so rather than implying a
  comparison that was not run.
* **No learned model was built**, by design. P050 is specified as a
  deterministic optimisation library and stays one.
