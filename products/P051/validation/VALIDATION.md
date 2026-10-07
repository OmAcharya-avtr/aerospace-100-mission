# Validation evidence — simplexguard 0.1.0

Validation level 3, research grade. **Not flight-qualified, not certified, not
approved for operational aerospace use.** This is not a verification tool: it
computes a robust invariant set for the discrete-time linear plant it is given,
under the disturbance bound it is told to assume, and proves nothing about any
plant it was not given.

Every number below was produced by a script in this directory, executed in this
container on 2026-10-07, with its raw stdout committed beside it as
`<script>_output.txt`. Nothing here was copied from a paper, estimated or
rounded by hand. The checks that undercut this package are in the tables on
purpose and are marked.

**Timings vary between runs.** Every microsecond figure below is wall-clock on
two shared contended cores and moves by 10-20 % run to run; the committed
`*_output.txt` files are the run the tables were taken from. The accuracy
figures are seeded and do not vary.

**Environment.** Python 3.13.16, NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1,
joblib 1.6.0, pytest 9.1.1, Hypothesis 6.168.5. Two shared CPU cores, contended
with four sibling build agents. Every timing below is wall-clock under that
contention and is reported as such.

**Test suite.** `python -m pytest tests/ -q` from the repository root:
**218 passed, 0 failed, 0 skipped, 0 xfail**, in 91 s.

**What is being validated.** The shipped illustrative plant is a single-axis
attitude loop, `x = [theta, theta_dot]` in `[rad, rad/s]`, exact zero-order-hold
discretisation of a double integrator at `dt = 0.05 s`; declared constraints
`|theta| <= 0.30 rad`, `|theta_dot| <= 0.50 rad/s`, `|u| <= 3.0 rad/s^2`;
declared disturbance an unmodelled angular acceleration of `0.12 rad/s^2` acting
for one sample, entered as the state-increment box with half-widths
`[1.500000000e-04 rad, 6.000000000e-03 rad/s]`. **These five numbers are illustrative**, chosen so
that the robust invariant set is non-trivial and the guard fires at a measurable
rate. They are not a measurement of any spacecraft.

---

## 1. Support-function and set algebra

`validate_support_function.py` → `validate_support_function_output.txt`,
elapsed 13.8 s.

| Check | Reference / method | Result | Tolerance |
|---|---|---|---|
| Box closed form against the generic linear programme | `h_box(c) = c.m + abs(c).r` against `scipy.optimize.linprog` on the same halfspaces; Rockafellar 1970 §13 | 4000 directions in dimensions 1–5, worst relative difference **8.284e-16** | 1e-9 |
| Box support against explicit vertex enumeration | the definition `sup_{x in P} c^T x` over all `2**n` vertices | 2000 directions in dimensions 1–4, worst absolute difference **1.776e-15** | 1e-12 |
| Positive homogeneity `h(t c) = t h(c)` | Rockafellar 1970 §13 | 3000 cases, worst relative difference **3.585e-16** | 1e-12 |
| Subadditivity `h(c1+c2) <= h(c1) + h(c2)` | same | 3000 cases, most negative slack **-3.553e-15** | -1e-12 |
| Minkowski additivity `h_{P+W} = h_P + h_W` | Schneider 1993 §1.7 | 2000 cases, worst relative difference **2.774e-16** | 1e-12 |
| Pontryagin difference against vertex shifts | Kolmanovsky and Gilbert 1998, eq. for `P (-) W` in halfspace form | **20000** points over 100 random polytopes, **0 mismatches** | exact membership |
| Erosion monotone in the disturbance set | `W1 ⊆ W2 ⇒ P(-)W2 ⊆ P(-)W1`, tested by LP subset check | 400 nested pairs, **0 violations** | 1e-9 |
| Preimage identity `x in M^-1 P ⟺ M x in P` | definition | **10000** points over 200 random maps, **0 mismatches** | exact membership |
| Redundancy removal preserves the set | LP-based minimal representation | 8000 points over 200 polytopes, **0 mismatches**, **1200** redundant rows removed | exact membership |

## 2. The robust invariant set (the certificate)

`validate_invariant_set.py` → `validate_invariant_set_output.txt`,
elapsed 64.5 s.

| Check | Reference / method | Result | Tolerance |
|---|---|---|---|
| One-dimensional closed form | hand arithmetic, shown in full in the script and in `tests/test_invariant.py`: `S = Omega_0` if `b0 >= 2 wmax`, empty otherwise | 4 hand-computed cases, worst absolute difference **0.000e+00**, including one case that must empty and does | 1e-12 |
| `S` is a subset of the declared `X` | LP subset test, facet by facet | **true**; X margin **0.000e+00** — `S` touches the angle facets of `X` exactly, so there is **no slack at all** in that direction | 1e-9 |
| The baseline input is admissible on all of `S` | `-K_b x in U` by support function | **true**; U margin **2.076789588 rad/s^2**, so the baseline uses at most **0.923210412** of its 3.0 rad/s^2 authority inside `S` | 1e-9 |
| `S` is robustly invariant, verified **exactly** | for every facet `(c,e)`: `h_S(A_b' c) + h_W(c) <= e`, one LP per facet, no sampling | **true**; worst facet margin **-5.551e-17**, i.e. zero to floating point, because `S` is the **maximal** such set and invariance holds with equality on some facet | 1e-9 |
| Sampled one-step invariance, as a second opinion | all 50 vertices of `S` plus 4000 interior samples, each pushed one step under all 4 vertices of `W` | **14604** (state, disturbance-vertex) pairs, worst residual **5.551e-17** | 1e-12 |
| Recursion is monotone | `S ⊆ Omega_0` by LP subset test | **true**; area of `S` **0.513045535** against area of `Omega_0` **0.600000000** | 1e-9 |
| A larger declared `W` gives a smaller `S` | areas at `W` scale 0.5x, 1x, 1.5x, 2x | **0.525200, 0.513046, 0.494365, 0.457606** — strictly decreasing | — |
| A certificate exists only below a measured declared-disturbance scale | recursion run at 2.0x and 2.5x the shipped declared bound | non-empty at **2.0x** (area 0.457606), **empty at 2.5x**, so the threshold lies in (2.0, 2.5]. This is headroom in the **declaration**, which is not headroom in the realised disturbance; section 5 measures the latter and it is far smaller | — |
| The three failure modes raise rather than returning a non-certificate | too weak a baseline (LQR R = 400), `W` as large as `X`, recursion cap at 3 | emptied at iteration **52**, emptied at iteration **1**, and `RecursionDidNotConverge` respectively | exact |
| Facet growth, reduced against unreduced | with and without redundancy removal | reduced: 4 facets at `Omega_0` growing by exactly 2 per iteration to **50** at convergence in **3.14 s**. Unreduced and capped at 10 iterations: **6144** halfspaces in **26.47 s**, because the intersection doubles its row count every iteration (6 × 2^10) | — |
| How much of `X` the certificate covers | area ratio and Chebyshev radii | area(S)/area(X) = **0.85507589**; Chebyshev radius of `S` **0.298406159**, of `S (-) W` **0.296722550**, so the erosion costs **1.684e-03** of inscribed radius | — |

## 3. The exact switching condition

`validate_guard_exactness.py` → `validate_guard_exactness_output.txt`,
elapsed 7.9 s.

| Check | Reference / method | Result | Tolerance |
|---|---|---|---|
| Support-function form against vertex enumeration of `W` | the condition `c_j^T (A x + B u) + h_W(c_j) <= d_j` against enumerating all 4 vertices of the box `W` | **40000** (state, input) pairs, 10000 of them bisected to within 1e-9 of the switching boundary: **0 mismatches** with a margin above 1e-12, and **4 disagreements AT the boundary** where \|margin\| ≤ 1e-12, worst \|margin\| **5.551e-17**. **This is a reported floating-point finding, not a tuned tolerance**: the eroded offset is rounded once, the vertex form rounds per vertex, and a state bisected onto the facet can fall either side of the 1-ulp gap | 1e-12 on the margin |
| Admitted inputs survive a dense interior sampling of `W` | 200 interior samples per admitted pair, which can falsify but not prove | 5274 admitted pairs × 200 = **1054800** successor states, **0 outside S** | exact membership |
| The margin equals the binding facet residual | facet normals are unit length after reduction, so the margin is a signed distance in state units | 5000 pairs, worst absolute difference **1.110e-16** | 1e-12 |
| Horizon-0 of the exact lead predictor equals the one-step condition | two independent code paths | **20000** random (state, reference) pairs, **0 disagreements** | exact |
| Guarded episodes never leave `S` under the declared bound | worst-case vertex sampler | 8 episodes × 1500 steps = **12000** steps, **0 exits from S**, **0 violations of X** | exact |
| An input outside `U` is refused and flagged | — | mode `baseline`, `input_admissible` false | exact |
| A state outside `S` is flagged as certificate lost | state [0.15, 0.45], slack **-3.017058e-02** in `S` | `certificate_lost` true, mode `baseline` | exact |

## 4. The assurance accounting

`validate_accounting.py` → `validate_accounting_output.txt`, elapsed 5.9 s.
Reference scenario: seed 51, 2000 steps (100 s), uniform sampler, square-wave
reference of 0.18 rad at a 4 s half period, no hysteresis.

| Quantity | Result | Notes |
|---|---|---|
| Authority changes | **140** over 2000 steps = **70.0 per 1000 steps** = **1.4 per second** | — |
| Baseline dwell | n = **70**, min 1, median **2.00**, mean **2.400**, max **5**; **31.4 %** of intervals last one step | histogram `{1: 22, 2: 24, 3: 0, 4: 22, 5: 2}` |
| Performance dwell | n = **71**, min 1, median **1.00**, mean **25.803**, max **83**; **64.8 %** of intervals last one step | **the guard chatters.** The mean hides it; the median does not. Reported as a limitation |
| Fraction under baseline authority | **0.084000** | identical on all 10 seeds, because the disturbance is small relative to the reference transients |
| Conservatism cost, guarded / unguarded | **1.198341** (guarded 258.634365, unguarded 215.826971) | paired on the same disturbance realisation |
| Baseline-only cost | **650.257772** | the price of never attempting the task |
| Fraction of the baseline-to-unguarded gap recovered | **0.901463** | — |
| Guarded constraint violations | **0** | worst row residual **-1.612008461e-03** |
| Unguarded constraint violations | **168** | worst row residual **+3.197937271e-01**, i.e. the unguarded run exceeds the 0.50 rad/s rate limit by 0.32 rad/s |
| Guarded invariant-set exits | **0** | — |
| Unguarded invariant-set exits | **189** | more than the 168 violations, because `S ⊂ X` |
| Switching margin | minimum **-1.226786374e-01**, median **+1.189731172e-01**, median at the firing steps **-4.822443534e-02** | negative exactly at the firing steps |

Across 10 seeds (51–60), 2000 steps each:

| Quantity | mean | min | max | sd |
|---|---|---|---|---|
| Switches per 1000 steps | 67.900000 | 65.000000 | 70.000000 | 1.523884 |
| Baseline fraction | 0.084000 | 0.084000 | 0.084000 | 0.000000 |
| Baseline dwell mean | 2.475365 | 2.400000 | 2.584615 | 0.056340 |
| Conservatism cost ratio | 1.199811 | 1.198341 | 1.201612 | 0.000905 |
| Gap recovered | 0.900232 | 0.897001 | 0.902130 | 0.001477 |
| Unguarded violations | 168.000000 | 168.000000 | 168.000000 | 0.000000 |
| Unguarded S exits | 188.500000 | 187.000000 | 190.000000 | 0.971825 |

**Zero guarded violations and zero guarded invariant-set exits on all 10 seeds
and under all three samplers (zero, uniform, worst-case vertex).** Totals: 0
guarded violations against 1680 unguarded violations over 20000 steps.

Minimum-baseline-dwell trade, seed 51 (safety is unaffected at every setting):

| min dwell | switches / 1000 steps | baseline fraction | cost ratio | steps held by dwell | guarded violations |
|---|---|---|---|---|---|
| 1 | 70.00 | 0.084000 | 1.198341 | 0 | 0 |
| 2 | **70.00** | 0.096000 | 1.201536 | 26 | 0 |
| 4 | 46.00 | 0.096000 | 1.214139 | 50 | 0 |
| 8 | 26.00 | 0.104000 | 1.273843 | 108 | 0 |
| 16 | 24.00 | 0.192000 | 1.392117 | 286 | 0 |

**A dwell of 2 is the worst setting in the table and this is a negative result:**
it raises the authority fraction from 0.084 to 0.096 and the cost ratio from
1.198341 to 1.201536 for **no reduction in the switch rate at all**, because a
two-step minimum extends baseline intervals that were already going to end in a
switch rather than merging two of them.

Reference-amplitude sweep (seed 51, 2000 steps):

| amplitude [rad] | switches / 1000 | baseline fraction | cost ratio | guarded X viol | unguarded X viol | unguarded S exits |
|---|---|---|---|---|---|---|
| 0.06 | 0.00 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| 0.10 | 0.00 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| 0.14 | 40.00 | 0.044000 | 1.096702 | 0 | 120 | 120 |
| 0.18 | 70.00 | 0.084000 | 1.198341 | 0 | 168 | 189 |
| 0.22 | 98.00 | 0.133000 | 1.307431 | 0 | 213 | 249 |
| 0.26 | 117.00 | 0.209000 | 1.405351 | 0 | 244 | 323 |

**Below an amplitude of about 0.12 rad the guard never fires and buys nothing**,
because the performance controller never leaves the certified envelope. That
region is in the table on purpose.

## 5. The deliberate bound-violation experiment

`validate_bound_violation.py` → `validate_bound_violation_output.txt`,
elapsed 10.6 s. 6 episodes × 1500 steps per scale. The guard is **not** rebuilt:
the invariant set, the eroded set and the switching condition are all still
computed from the declared `W`; only the realised disturbance is inflated.

| Scale `rho` | sampler | X violations | S exits | worst X residual | steps with `w` outside declared `W` |
|---|---|---|---|---|---|
| 1.00 | uniform | **0** | **0** | -1.1619e-03 | 0.0000 |
| 1.10 | uniform | 0 | 0 | -1.8027e-04 | 0.1681 |
| 1.25 | uniform | 1 | 1 | 1.2922e-03 | 0.3527 |
| 1.50 | uniform | 2 | 2 | 6.5015e-04 | 0.5469 |
| 2.00 | uniform | 8 | 10 | 3.7911e-03 | 0.7430 |
| 12.00 | uniform | 236 | 294 | 7.9681e-02 | 0.9944 |
| 1.00 | **vertex** | **0** | **0** | -6.8195e-04 | 0.0000 |
| 1.10 | vertex | **3** | **3** | 4.0275e-04 | 1.0000 |
| 1.25 | vertex | 5 | 5 | 1.1763e-03 | 1.0000 |
| 1.50 | vertex | 14 | 17 | 2.8098e-03 | 1.0000 |
| 2.00 | vertex | 30 | 37 | 5.9759e-03 | 1.0000 |
| 12.00 | vertex | 539 | 659 | 1.7476e-01 | 1.0000 |

**The headline negative result of this package.** Under the worst-case vertex
sampler the first constraint violation appears at `rho = 1.10`: a **10 %**
underestimate of the disturbance bound is already enough. Under the uniform
sampler it appears at `rho = 1.25`. There is no margin beyond the declared
bound, and the reason is structural rather than accidental: the facet-wise slack
of `S` inside `X` is **[0, 0, 0, 0]**, so `S` touches every facet of `X`, and
the recursion makes invariance hold with equality (worst facet margin
**-5.551e-17**) because it computes the **maximal** robust invariant set. A
maximal set has, by definition, spent all of its margin.

What the guard is still worth as the assumption degrades (vertex sampler,
6 episodes × 1500 steps per point, `validate_bound_violation.py` check 6):

| `rho` | unguarded X violations | guarded X violations | fraction eliminated |
|---|---|---|---|
| 1.00 | 665 | 0 | **1.0000** |
| 1.50 | 662 | 14 | 0.9789 |
| 2.00 | 663 | 30 | 0.9548 |
| 4.00 | 676 | 76 | 0.8876 |
| 6.00 | 674 | 167 | 0.7522 |
| 12.00 | 672 | 539 | 0.1979 |

The guard degrades rather than failing outright, and at 12x the declared bound
it eliminates only **19.8 %** of the violations. **Above `rho = 1` nothing is
guaranteed**, and every number in this table is a measurement of this plant,
this baseline and this sampler, not a property of the Simplex architecture.

In every row with any failure, the invariant-set exit count is at least the
constraint-violation count — the certificate is lost before, or at the same time
as, the constraint is broken, with an exits-to-violations ratio up to **1.31**.
An observer watching only `X` would see nothing wrong while the guard was
already operating outside its own theory.

## 6. The learned switch predictor against the exact analytic condition

`validate_predictor.py` → `validate_predictor_output.txt`, elapsed 36.1 s.
40 episodes × 1200 steps, split **by episode** into 26 train / 6 calibration /
8 test. Features: `theta`, `theta_dot`, `reference`, `tracking_error`,
`u_perf_unsaturated` — raw measurable quantities only. **The switching-condition
margin is deliberately not a feature**: a model handed the exact computation's
output is not an alternative to it.

Base rates: lead-0 **0.081687** overall, **0.083333** on the test split; lead-5
**0.148201** overall, **0.149477** on test. Performance-input saturation rate
**0.093729**.

### Task A, lead 0: does the guard fire at this step — the condition's own criterion

| Predictor | Precision | Recall | F1 | Brier | ECE | µs / single-row decision |
|---|---|---|---|---|---|---|
| **Exact one-step guard condition** | **1.00000** | **1.00000** | **1.00000** | n/a | n/a | **8.68** |
| Learned forest, 150 trees, isotonic | 0.97764 | 0.98375 | 0.98069 | 0.002629 | 0.00258 | 7343.16 |
| Learned logistic, isotonic | undefined | 0.00000 | undefined | 0.072448 | 0.02798 | 819.25 |
| Always-negative at the training base rate | undefined | 0.00000 | undefined | 0.076467 | 0.00885 | — |

**The learned model loses on accuracy and loses on cost, and both losses are
published without retuning.** 18 false positives and 13 false negatives in 9600
held-out steps, against zero of each for the exact computation. The structural
reason is that the exact boundary is the intersection of 50 halfspaces in
`(x, u)`, none of them axis-aligned, and an axis-aligned tree ensemble
approximates a tilted hyperplane with staircase error. On cost the factor is
**846x**: at `dt = 0.05 s` the exact condition uses **0.0174 %** of the sample
interval and the forest **14.69 %**.

The logistic model predicts no positives at all at a 0.5 threshold, which is the
expected behaviour of a single hyperplane asked to represent a 50-facet
intersection. It is in the table as a measured lower bound.

Latency against forest size — **no size reaches the exact condition's cost**:

| trees | F1 | Brier | µs / decision | fit time |
|---|---|---|---|---|
| 20 | 0.978301 | 0.002615 | 1717.9 | 0.3 s |
| 50 | 0.981250 | 0.002374 | 3247.3 | 0.5 s |
| 150 | 0.980685 | 0.002629 | 6723.7 | 1.7 s |
| 300 | 0.977472 | 0.002760 | 12424.5 | 3.1 s |
| exact | **1.000000** | n/a | **8.68** | — |

A 20-tree forest is already **198x** the exact condition's latency at F1
0.978301, and latency rises by only 7.23x from 20 to 300 trees, so the gap is
scikit-learn's per-call overhead and not tree traversal.

### Task B, lead 5: does the guard fire within the next 5 steps

| Predictor | Precision | Recall | F1 | Brier | µs / decision |
|---|---|---|---|---|---|
| Exact worst-case reachability, L = 5 | 0.43477 | 0.86284 | 0.57819 | n/a | 13.84 |
| Exact nominal reachability, L = 5 | 0.44448 | 0.86284 | 0.58672 | n/a | 18.70 |
| Persistence (fires now ⇒ will fire) | 0.85875 | 0.48076 | 0.61642 | n/a | — |
| **Learned forest, 150 trees, isotonic** | **0.96829** | 0.83345 | **0.89583** | 0.026397 | 7123.80 |
| Always-negative at the base rate | undefined | 0.00000 | undefined | 0.127245 | — |

**The learned model wins on F1 here, and the reason is stated rather than
claimed.** The exact predictor answers "may fire under *some* admissible
disturbance sequence", which over-predicts a realised episode by construction:
its precision is 0.43 against a recall of 0.86. The learned model is fitted to
the realised label, which is a different and easier question. It still costs
**515x** more per decision.

**A second honest negative, about the exact predictor.** Its recall on realised
episodes is **0.86284**, not 1.0, so it misses **13.7 %** of realised firings
despite being a worst-case over-approximation. Two stated approximations are
responsible, both quantified rather than assumed harmless: it ignores the
saturation of the performance input (measured at **9.3729 %** of steps) and it
ignores the guard's own intervention inside the horizon, which changes the
realised trajectory.

### Lead time before each firing event (287 events on the test split)

| Predictor | mean | median | max | fraction with zero lead |
|---|---|---|---|---|
| Exact worst-case, L = 5 | **12.861** | 8.0 | 40.0 | **0.0000** |
| Exact nominal, L = 5 | 12.861 | 8.0 | 40.0 | 0.0000 |
| Learned forest, lead-5 target | 6.624 | 8.0 | 16.0 | 0.0418 |
| Learned forest, lead-0 target | 0.115 | 0.0 | 5.0 | 0.9686 |
| Exact one-step condition | 0.000 | 0.0 | 0.0 | 1.0000 |

The exact worst-case predictor anticipates **every** event, with a mean lead of
12.861 steps (0.6430 s). The forest misses 4.18 % of them entirely.

### Calibration — the one thing the exact computation cannot do

Expected calibration error **0.002583** over 10 equal-width bins, Brier
**0.002629**, against **0.076467** for an always-negative forecaster at the
training base rate. Reliability bins (Task A):

| bin | n | mean forecast | observed frequency |
|---|---|---|---|
| [0.0, 0.1) | 8646 | 0.000211 | 0.000000 |
| [0.1, 0.2) | 126 | 0.122057 | 0.039683 |
| [0.3, 0.4) | 23 | 0.324280 | 0.347826 |
| [0.5, 0.6) | 12 | 0.502476 | 0.666667 |
| [0.6, 0.7) | 9 | 0.680715 | 0.777778 |
| [0.7, 0.8) | 3 | 0.722847 | 0.666667 |
| [0.8, 0.9) | 13 | 0.849174 | 0.538462 |
| [0.9, 1.0) | 768 | 1.000000 | 0.993490 |

Neither exact predictor produces a probability at all, so neither has a Brier
score or a reliability curve. That asymmetry is the learned model's only
structural advantage in this package, and it is not enough to recommend it for
the switching decision itself.

## 6b. The command-line interface

`validate_cli.py` → `validate_cli_output.txt`, elapsed 25.0 s. Every subcommand
is executed in a clean subprocess and its raw output is committed, so the
README quickstart block cannot drift from the code.

| Check | Result |
|---|---|
| `--help` exits 0 and carries the research-grade and not-a-verification-tool statements | **pass** |
| `plant`, `invariant`, `guard`, `run`, `accounting` all exit 0 | **pass** |
| An empty invariant set exits **3** and names the iteration and what to change | **pass** |
| A non-converged recursion exits **3** and says the last iterate is not returned | **pass** |
| A bad argument exits **2** | **pass** |

Failed checks: 0.

## 7. Checks that failed, were weakened, or went against this package

Collected in one place so they are not spread thin.

1. **The learned predictor loses on the switching condition's own criterion**,
   on both accuracy (F1 0.98069 against 1.0) and cost (781x). Published, not
   retuned. Section 6.
2. **The guarantee has essentially no margin.** A 10 % underestimate of the
   declared disturbance bound already produces constraint violations under the
   worst-case sampler. Section 5.
3. **The guard chatters.** 64.8 % of performance intervals and 31.4 % of
   baseline intervals last a single step. Section 4.
4. **A minimum baseline dwell of 2 costs authority and tracking for no reduction
   in the switch rate.** Section 4. The check that asserted a strictly
   decreasing switch rate was **weakened to non-increasing** when this was
   measured, and the plateau is reported rather than removed.
5. **The exact worst-case lead predictor is not sound on realised episodes**
   (recall 0.86284), because of two approximations it makes and states.
   Section 6.
6. **The exact switching condition and brute-force vertex enumeration disagree
   on 4 of 10000 states bisected onto the switching boundary**, at
   \|margin\| ≤ 5.551e-17. Floating point, not algebra; reported rather than
   tolerance-tuned. Section 3.
7. **Below a reference amplitude of 0.12 rad the guard never fires and provides
   no benefit at all** on this scenario — pure overhead. Section 4.
8. **At 12x the declared bound the guard eliminates only 10.4 % of the
   unguarded run's violations.** Section 5.
9. A known-answer test originally written against the *minimal* robust invariant
   set `wmax/(1-|lambda|)` was **wrong**: the switching condition needs the
   *maximal* set, which for the one-dimensional case is `Omega_0` whenever
   `b0 >= 2 wmax` and empty otherwise. The arithmetic was redone and is shown in
   full in `tests/test_invariant.py` and in `validate_invariant_set.py` check 1.
10. `S` covers only **85.5 %** of the declared constraint set `X`, so **14.5 %**
    of the states a user may consider acceptable are outside the certificate and
    the guard will refuse to operate there.

## 8. Reproducing every number

From a cold clone, no install step needed (`pythonpath = ["src"]` is set in
`pyproject.toml`):

```bash
python -m pytest tests/ -q --junit-xml=junit.xml     # 218 tests
ruff check src/ tests/ examples/ validation/
python -m simplexguard --help

python validation/validate_support_function.py       # set algebra, 14 s
python validation/validate_invariant_set.py          # the certificate, 64 s
python validation/validate_guard_exactness.py        # the switching condition, 8 s
python validation/validate_accounting.py             # the accounting, 6 s
python validation/validate_bound_violation.py        # the bound-violation sweep, 11 s
python validation/validate_predictor.py              # the AI benchmark, 36 s
python validation/validate_cli.py                    # the CLI in a subprocess, 25 s
python validation/worked_example.py                  # the README worked example

MPLBACKEND=Agg python examples/invariant_set.py
MPLBACKEND=Agg python examples/guarded_episode.py
MPLBACKEND=Agg python examples/bound_violation.py
MPLBACKEND=Agg python examples/predictor_benchmark.py
MPLBACKEND=Agg python examples/accounting_tradeoff.py
```

Every script is seeded and deterministic. Seeds: 51051 (set algebra), 51052
(invariant set), 51053 (guard exactness), 51 and 51–60 (accounting), 51055
(bound violation), 5101 (predictor). The committed `*_output.txt` files are the
raw stdout of exactly these commands, and the committed PNGs are what exactly
these example scripts produce.

## 9. References

Real and verifiable; no page number is quoted that was not checked in a source
available here, so most citations name author, title, venue and year only.

- Rockafellar, R. T., *Convex Analysis*, Princeton University Press, 1970,
  section 13 — the support function.
- Schneider, R., *Convex Bodies: the Brunn-Minkowski Theory*, Cambridge
  University Press, 1993, section 1.7 — support functions and Minkowski
  additivity.
- Boyd, S. and Vandenberghe, L., *Convex Optimization*, Cambridge University
  Press, 2004, section 8.5.1 — the Chebyshev-centre linear programme.
- Gilbert, E. G. and Tan, K. T., "Linear systems with state and control
  constraints: the theory and application of maximal output admissible sets",
  *IEEE Transactions on Automatic Control* 36(9), 1991, pp. 1008–1020 — the
  one-step-set recursion.
- Kolmanovsky, I. and Gilbert, E. G., "Theory and computation of disturbance
  invariant sets for discrete-time linear systems", *Mathematical Problems in
  Engineering* 4(4), 1998, pp. 317–367 — the Pontryagin difference in halfspace
  form and robust invariance.
- Blanchini, F. and Miani, S., *Set-Theoretic Methods in Control*, Birkhäuser,
  2008 — invariant sets for constrained linear systems.
- Borrelli, F., Bemporad, A. and Morari, M., *Predictive Control for Linear and
  Hybrid Systems*, Cambridge University Press, 2017 — the invariant-set
  algorithms, chapter 10.
- Raković, S. V., Kerrigan, E. C., Kouramas, K. I. and Mayne, D. Q., "Invariant
  approximations of the minimal robust positively invariant set", *IEEE
  Transactions on Automatic Control* 50(3), 2005, pp. 406–410 — the **minimal**
  robust invariant set, named here because it is a different object from the
  maximal set this package computes, and confusing the two is the mistake item 9
  of section 7 records.
- Seto, D., Krogh, B., Sha, L. and Chutinan, A., "The Simplex architecture for
  safe online control system upgrades", *Proceedings of the American Control
  Conference*, 1998 — the architecture.
- Sha, L., "Using simplicity to control complexity", *IEEE Software* 18(4),
  2001, pp. 20–28 — the architecture.
- Schierman, J. D., DeVore, M. D., Richards, N. D. and Clark, M. A., "Runtime
  assurance for autonomous aerospace systems", *Journal of Guidance, Control,
  and Dynamics*, 2020 — runtime assurance in an aerospace setting. No page
  number is quoted because none was verified here.
- Anderson, B. D. O. and Moore, J. B., *Optimal Control: Linear Quadratic
  Methods*, Prentice-Hall, 1990, chapter 3 — the discrete LQR gain.
- Franklin, G. F., Powell, J. D. and Workman, M. L., *Digital Control of Dynamic
  Systems*, 3rd ed., Addison-Wesley, 1998, chapter 4 — zero-order-hold
  discretisation.
- Zadrozny, B. and Elkan, C., "Transforming classifier scores into accurate
  multiclass probability estimates", *KDD*, 2002 — isotonic calibration.
- Niculescu-Mizil, A. and Caruana, R., "Predicting good probabilities with
  supervised learning", *ICML*, 2005 — calibration of classifier scores.
- Brier, G. W., "Verification of forecasts expressed in terms of probability",
  *Monthly Weather Review* 78(1), 1950 — the Brier score.
- Naeini, M. P., Cooper, G. F. and Hauskrecht, M., "Obtaining well calibrated
  probabilities using Bayesian binning", *AAAI*, 2015, and Guo, C., Pleiss, G.,
  Sun, Y. and Weinberger, K. Q., "On calibration of modern neural networks",
  *ICML*, 2017 — the expected calibration error and its binning bias.
- Huangfu, Q. and Hall, J. A. J., "Parallelizing the dual revised simplex
  method", *Mathematical Programming Computation* 10(1), 2018 — HiGHS, the
  solver behind `scipy.optimize.linprog` as used here.
