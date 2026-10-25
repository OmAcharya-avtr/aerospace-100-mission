# Method: every derivation in one place

Every closed form this package implements is derived here, with the reference
it stands on and the independent check it is scored against. Where a reference
is cited for a *method* and not for a number, that is stated.

Nothing in this document is a number taken from a source. Every numerical
figure quoted is measured by a script in `validation/` and names the output
file it comes from.

## Notation

`R` reliability, `Q = 1 - R` unreliability, `H = -log R` cumulative hazard, all
dimensionless. `t` in hours, rates in 1/hour. For a block, `q_i` is its
unreliability at its effective age.

## 1. The spaces, and why there are three

`R`, `Q` and `H` carry the same information in exact arithmetic and completely
different information in float64.

- `H` is **additive over series composition and over mission phases**, so a
  series chain and a phased profile are both sums in this space, with one
  rounding per term and no cancellation.
- `Q` is **multiplicative over parallel composition**, so a redundant group's
  small unreliability survives as the exact product.
- `R` is what gets reported, and is never used as the carrier from which a
  small `Q` is later recovered.

Measured cost of getting this wrong, from
`validation/validate_precision_output.txt`:

- `1 - prod(1 - q_i)` for a parallel pair returns exactly `0.0` at and below
  `q = 5.748e-09`, where the correct answer is `3.3e-17`. For a triple the
  cliff is at `q = 3.502e-06`.
- For a ten-block series chain at `q = 1e-12`, the naive expression returns
  `9.999778782798785e-12` against the hazard-space `9.999999999955000e-12`:
  4.66 significant figures lost, and **outside** the first- and second-order
  Bonferroni bracket `[9.999999999954999e-12, 9.999999999999999e-12]`.

The last point is the one that matters methodologically. A property test
asserting the implementation against the same expression would pass. Only a
bound derived from a *different* argument catches it, which is why
`reliablock.bounds` exists.

## 2. Series

A series structure survives only if every block survives, so `R = prod R_i`
and therefore `H = sum H_i` exactly. Implemented as
`numerics.series_hazard`, with `Q = -expm1(-H)`.

**Checked against**: the product of the block reliabilities (75 cases, worst
relative difference `7.544748e-16`), and the Bonferroni bracket as an
independently derived bound (0 of 75 outside).

## 3. Parallel

A parallel structure fails only if every block fails, so `Q = prod Q_i`
exactly. `R = 1 - Q` is the derived figure.

**Checked against**: the exact product (worst relative `1.836710e-16`) and
exhaustive `2**n` state enumeration over the structure function (worst
relative `0.000000e+00`).

## 4. k-of-n

`R` is the Poisson-binomial survival `P(at least k survive)` and `Q` is
`P(at least n - k + 1 fail)`. Each is evaluated by its own exact convolution
recurrence over the number of survivors or failures:

```
d_0 = [1]
d_{m+1}[j] = d_m[j] (1 - p_m) + d_m[j-1] p_m
```

`O(n^2)` multiplications, all terms non-negative, so there is no cancellation
in either. Neither is derived from the other, so neither inherits the other's
conditioning; their sum differs from 1 only by accumulated rounding, which
`Quantity.complement_residual` reports (worst measured `2.220446e-16`).

For identical blocks both reduce to the binomial sum
`sum_{i=k}^{n} C(n,i) p^i (1-p)^(n-i)`.

**Checked against**: `scipy.stats.binom.sf`, 300 cases, worst relative
`4.281257e-16`; the series chain at `k = n` (`3.017899e-16`); the parallel
group at `k = 1` (`1.147943e-16`); exhaustive enumeration for non-identical
blocks (`3.990254e-16`).

**Defect found here by Hypothesis**: the tail sum is a sum of non-negative
float64 terms that add to 1 in exact arithmetic, so it can land a few ulps
*above* 1. It returned `1.0000000000000002` at
`rates = [0.0078125, 5e-05, 5e-05, 1.0000000000000003e-09, 1e-09]` over
229415.5 hours, and `hazard_from_unreliability` then raised on a perfectly
valid diagram. The result is now clamped into `[0, 1]`; pinned as
`test_poisson_binomial_sum_clamped_to_one`.

## 5. Standby with a declared switching reliability

### Erlang form, `m` identical cold standby units

```
R(t) = exp(-lambda t) * sum_{j=0}^{m} (p_s lambda t)^j / j!
```

**Derivation.** With perfect switching the system life is the sum of `m + 1`
independent exponentials, so it is Erlang-distributed with shape `m + 1`, whose
survival function is `sum_{j=0}^{m} exp(-lambda t) (lambda t)^j / j!`. The
`j`-th term is the event that exactly `j` changeovers have been required, and
each of those `j` changeovers must have succeeded, so the term carries
`p_s^j`. At `m = 1` this is `R = exp(-lambda t)(1 + p_s lambda t)`.

### General form, one standby unit, any distributions, any hazard fraction

```
R(t) = R_p(t) + p_s * integral_0^t f_p(u) exp(-H_s(f u + (t - u))) du
```

**Derivation.** The system survives either because the primary survives to `t`,
or because the primary fails at some `u < t`, the changeover succeeds, and the
standby survives from `u` to `t`. A standby unit that has waited `u` accruing
hazard at a fraction `f` of its operating rate has effective age
`f u + (t - u)` at `t`. At `f = 0` this is the ordinary convolution
`integral f_p(u) R_s(t - u) du`; at `f = 1` the effective age is `t` and the
expression collapses to parallel-with-switch. Both limits are asserted as
property tests.

Evaluated with `scipy.integrate.quad`; the quadrature's reported absolute
error bound is carried in `Quantity.detail["quadrature_abs_error"]`, because a
convolution-based figure whose quadrature error is not stated is not a
checkable number.

**Checked against**: the convolution integral itself for the closed form (81
cases spanning three rates, three durations, three switch reliabilities and
three hazard fractions, worst relative `2.211217e-16`); the Erlang series
directly (`2.220446e-16`); a Weibull primary and standby (`0.000000e+00`); and
a seeded simulation that follows the physical story with no algebra in common
(within `1.8208` standard errors on the warm-standby case,
`validation/validate_montecarlo_output.txt`).

## 6. Shared load, two units

Each unit's declared rate `lambda_i` is its full-load rate; while both survive
each carries `s lambda_i` for a declared sharing factor `s` in `(0, 1]`. With
`L = s (lambda_1 + lambda_2)`:

```
R(t) = exp(-L t)
     + s lambda_1 (exp(-lambda_2 t) - exp(-L t)) / (L - lambda_2)
     + s lambda_2 (exp(-lambda_1 t) - exp(-L t)) / (L - lambda_1)
```

**Derivation.** The four-state Markov chain has states (both alive), (only 1
alive), (only 2 alive), (down). The sojourn in (both alive) is exponential with
rate `L`. Leaving it, unit 1 is the one that failed with probability
`s lambda_1 / L`, after which unit 2 carries the whole load at `lambda_2`.
Integrating the survivor's own survival over the first failure time `u`,
`integral_0^t s lambda_1 exp(-L u) exp(-lambda_2 (t - u)) du`, gives the second
term; the third is the mirror image.

**The degeneracy, and how it is avoided.** Both denominators vanish when
`L = lambda_i`, which happens exactly at `s = 0.5` with `lambda_1 = lambda_2`.
The first implementation switched to an analytic limit below a declared
relative distance from the vanishing denominator; the switch was a
discontinuity and the threshold would have had to be tuned to hide it. The
algebraically identical form

```
(exp(-a t) - exp(-L t)) / (L - a) == t exp(-L t) * expm1((L - a) t) / ((L - a) t)
```

is exact for every argument including zero, so the implementation has **no
threshold and no tuning constant**. `expm1(x)/x` has a removable singularity at
zero, filled in with its limit of 1.

**Checked against**: `scipy.integrate.solve_ivp` with DOP853 at rtol 1e-12 on
the Markov system, which never forms the closed form at all -- 108 cases, worst
relative `4.038264e-14`; the analytic limit `(1 + x) exp(-x)` at exact
degeneracy (`2.220447e-16`); a parallel pair at `s = 1` (`1.120369e-16`); and a
two-stage simulation (`0.7233` standard errors).

## 7. A general two-terminal network

Pivotal (Shannon) decomposition on one edge at a time:

```
Q_sys = q_e Q(e failed) + (1 - q_e) Q(e working)
```

recursing until the residual network is decided -- the sink unreachable with
every undecided edge working (certain failure), or reachable using only edges
already declared working (certain success). The recursion is in unreliability
space throughout and its terms are non-negative, so a network of
high-reliability edges keeps its answer: the five-edge bridge retains
`Q = 2.000000e-22` at an edge unreliability of `1e-11`, where its reliability
is exactly `1.0` in float64.

Barlow, R. E. and Proschan, F. (1975), *Statistical Theory of Reliability and
Life Testing*, Holt, Rinehart and Winston, chapter 1, for the decomposition.
**Cited for the method; no number here comes from it.**

## 8. The bridge network, derived

Four vertices `s`, `a`, `b`, `t` and five edges: `e1: s-a`, `e2: s-b`,
`e3: a-b`, `e4: a-t`, `e5: b-t`.

Minimal paths `{e1,e4}`, `{e2,e5}`, `{e1,e3,e5}`, `{e2,e3,e4}`; minimal cuts
`{e1,e2}`, `{e4,e5}`, `{e1,e3,e5}`, `{e2,e3,e4}`.

Pivotal decomposition on `e3`, with equal edge reliability `p`:

- `e3` working merges `a` and `b`, leaving two parallel pairs in series:
  `(1 - (1-p)^2)^2 = (2p - p^2)^2 = 4p^2 - 4p^3 + p^4`
- `e3` failed leaves two disjoint paths in parallel:
  `1 - (1 - p^2)^2 = 2p^2 - p^4`

```
R = p (4p^2 - 4p^3 + p^4) + (1 - p)(2p^2 - p^4)
  = 4p^3 - 4p^4 + p^5 + 2p^2 - p^4 - 2p^3 + p^5
  = 2p^5 - 5p^4 + 2p^3 + 2p^2
```

At `p = 0.9`: `2(0.59049) - 5(0.6561) + 2(0.729) + 2(0.81) = 0.97848`.

**What is cited and what is derived.** The closed form and that value are
**derived above**. The bridge network and the conditional-probability method
are standard material, cited for the method only: Barlow and Proschan (1975),
chapter 1, and Billinton, R. and Allan, R. N., *Reliability Evaluation of
Engineering Systems*, 2nd edition, Plenum Press, 1992. **No page number and no
published figure is quoted**, because this build had no access to either volume
and a page number that cannot be checked is worse than none.

**What stands in its place**: five independent routes, all reproducible from
this repository. `validation/validate_bridge_output.txt` has the table; worst
relative difference `9.673179e-16` among the four algebraic routes and
`1.128538e-14` against the 32-state enumeration. The polynomial is also
self-dual (`R(p) + R(1-p) = 1`, worst absolute residual `0.000000e+00` over 8
values of `p`), which is a property of the structure that no part of the
implementation was written to satisfy.

**Why the bridge is the right check**: it is not series-parallel, and the two
decompositions a reader might reach for *bracket* the true answer rather than
giving it. At `p = 0.9`: two-disjoint-paths `0.963900000000000`, exact
`0.978480000000000`, merged-vertices `0.980100000000000`.

## 9. Common cause: the beta-factor model

A component's total failure rate splits into an independent part and a part
attributed to a cause shared with its redundant group:

```
lambda_total = lambda_independent + lambda_common,  lambda_common = beta lambda_total
```

The shared cause is a **single additional basic event in series with the whole
group**. For a group of `n` members each with cumulative hazard `H` under a
structure function `phi`:

```
R_group = exp(-beta H) * R_phi((1 - beta) H)
```

Both limits are exact and asserted as property tests: at `beta = 0` the first
factor is 1 and the second is the independent result (reproduced bit for bit,
because no common-cause block is created at all); at `beta = 1` the second
factor is `R_phi(0) = 1` and the group reduces to `exp(-H)`, one member's
reliability (worst relative `0.000000e+00` over `n = 2..5`).

The model is applied by **rewriting the diagram**, so the with-and-without
comparison runs one quantifier over two diagrams rather than two code paths
over one diagram, and the shared-cause event appears in the cut-set listing as
a singleton -- which is the qualitative half of the answer.

Fleming, K. N. (1975), General Atomic report GA-A13284, for the origin.
Mosleh, A. et al. (1988), NUREG/CR-4780 / EPRI NP-5613, for the standard
treatment. IEC 61508-6:2010 Annex D for the functional-safety quantification.
**All three cited for the model's structure. No numerical beta in this
repository comes from any source**: every one is a declared input chosen to
span a range.

### Where the model stops being conservative

The beta-factor model has **exactly one** common-cause mode: the one that fails
every member. A `k`-of-`n` group fails once `n - k + 1` members have failed,
which for `k >= 2` is fewer than all of them, so moving failure probability
into the single mode can *reduce* the computed group unreliability.

Measured by bisection at `beta = 0.1`, the member cumulative hazard at which
the ratio `Q(with cc) / Q(independent)` crosses 1
(`validation/validate_commoncause_output.txt`):

| group | failures needed | ratio at `H = 0.01` | ratio at `H = 3` | crossing `H` |
|---|---|---|---|---|
| 1-of-2 | 2 | 10.905368 | 1.000960 | none, conservative everywhere |
| 1-of-3 | 3 | 1015.322358 | 1.002914 | none, conservative everywhere |
| 2-of-3 | 2 | 4.198132 | 0.997584 | 0.302944 |
| 2-of-4 | 3 | 256.285271 | 0.995519 | 0.653729 |
| 3-of-4 | 2 | 2.516174 | 0.999621 | 0.124016 |
| 2-of-5 | 4 | 20557.783027 | 0.993070 | 0.924649 |
| 3-of-5 | 3 | 103.720793 | 0.999115 | 0.340012 |
| 4-of-5 | 2 | 1.841492 | 0.999958 | 0.067949 |
| 2-of-6 | 5 | 1722249.954759 | 0.990350 | 1.141930 |
| 3-of-6 | 4 | 6907.998017 | 0.998346 | 0.533470 |
| 4-of-6 | 3 | 52.612956 | 0.999882 | 0.217911 |
| 5-of-6 | 2 | 1.503309 | 0.999996 | 0.042978 |

`k = n` is absent because `Diagram.validate` refuses a beta factor there. That
refusal was added **because** an earlier run of the same sweep measured `k = n`
as non-conservative at every hazard -- a ratio of `0.918939` at `k = n = 6` --
which is correct arithmetic for a series group and meaningless as a redundancy
model. The refusal message quotes the measured ratio.

This is the known inadequacy of the beta-factor model for groups larger than
two, and the reason NUREG/CR-4780 introduces the multiple-Greek-letter and
alpha-factor models, which carry a separate parameter per failure
multiplicity. Neither is implemented here.
`reliablock.commoncause.group_beta_effects` reports the effect per group and
`common_cause_comparison` prints a warning naming the alternatives when any
group crosses.

## 10. Importance measures

With `Q_sys` the system unreliability and `q_i` an event's unreliability:

- **Birnbaum** `I_B(i) = dQ_sys / dq_i = Q_sys(q_i=1) - Q_sys(q_i=0)`.
  Birnbaum, Z. W. (1969), in *Multivariate Analysis II*, Academic Press,
  pp. 581-592.
- **Criticality** `I_C(i) = I_B(i) q_i / Q_sys`.
- **Fussell-Vesely** `I_FV(i) = P(some minimal cut set containing i occurred) / Q_sys`.
  Fussell, J. B. (1975), IEEE Transactions on Reliability R-24(3), 169-174.

`Q_sys` is **multilinear** in the `q_i` for any coherent structure of
independent components -- each `q_i` appears to at most the first power in every
term -- so `I_B` is constant in `q_i` and a finite difference is exact for any
step size in exact arithmetic. The agreement is therefore measured across five
decades of step and reported not to degrade beyond the cancellation in the
numerator: worst relative difference `3.490329e-09` over 135 comparisons, with
the error growing with the step's reciprocal exactly as cancellation
predicts -- on `mixed_rates`, from `2.170840e-13` at a step of `1e-3` to
`3.490329e-09` at `1e-7`.

The identity `I_C(i) = (Q_sys - Q_sys(q_i=0)) / Q_sys` follows from the pivotal
decomposition:

```
Q_sys - Q_sys(q_i=0) = q_i [Q_sys(q_i=1) - Q_sys(q_i=0)] = q_i I_B(i)
```

and is checked to `1.789122e-15` over 27 rows.

**Fussell-Vesely is computed exactly**, by enumerating the states of the events
appearing in the relevant cut sets, not by the rare-event approximation
`sum of cut-set probabilities`. The approximation overestimates, and by how
much is measured: up to `1.268384e-01` absolute on the diagrams in
`validation/validate_importance_output.txt`, which is the usual reason a
published Fussell-Vesely figure exceeds 1.

Importance measures are **refused** for a diagram containing a `standby` or
`shared_load` node. Those have an exact Boolean structure function and
therefore exact minimal cut sets, but their quantification is not the
independent-event arithmetic over that function, so `dQ_sys/dq_i` is not the
derivative of the number `quantify()` returns.

## 11. Minimal cut sets

Structural recursion over node kinds:

- `series`: the union of the children's cut sets.
- `parallel` and `shared_load`: one cut from each child, unioned.
- `k_of_n`: one cut from each of every `(n - k + 1)`-subset of children.
- `standby`: the primary's cuts crossed with (the switch) and with (one cut
  from each standby unit) -- the switch being a real basic event with a
  declared failure probability and a cut set of its own with the primary.
- `network`: the minimal transversals of the minimal path sets, built by
  incremental crossing with minimalisation at each step.

Minimalised at every level, so the returned count is a count of *minimal* cut
sets, ordered by size then lexicographically so the listing is diffable.

**Checked against**: exhaustive `2**n` state enumeration, which evaluates the
structure function on every state and keeps the minimal failures with no code
in common -- identical **sets** on all 13 diagrams measured, over 147712 states
(`validation/validate_cutsets_output.txt`). Plus three properties checked
directly against the structure function: every reported cut set fails the
system, restoring any one of its events restores the system, and no reported
cut set contains another. Zero violations in all three.

## 12. Bounds, and why they are a different class of check

**Esary-Proschan** (Esary, J. D. and Proschan, F., 1963, *Technometrics*
5(2), 191-209). For a coherent system of independent components with minimal
path sets `P_j` and minimal cut sets `K_j`:

```
prod_j (1 - prod_{i in K_j} q_i)  <=  R_sys  <=  1 - prod_j (1 - prod_{i in P_j} p_i)
```

Exact when no two cut sets (respectively paths) share a component, which is the
series-parallel case; strict for the bridge, where the measured bracket at
`p = 0.9` is `[9.781407801000e-01, 9.973487799000e-01]` of width
`1.920800e-02` around an exact `0.97848`.

**Bonferroni** orders 1 and 2 on a series chain:

```
sum_i q_i - sum_{i<j} q_i q_j  <=  Q_sys  <=  sum_i q_i
```

Both sides are sums of non-negative terms, so neither inherits the
cancellation in `1 - prod(1 - q_i)` -- which is why this is the check that
detects it. The naive expression falls outside the bracket in 13 of 20
measured cases.

**The slack took three attempts**, and both wrong answers are worth recording:

1. A slack floored at an absolute constant. The Bonferroni bracket for ten
   blocks at `q = 1e-12` has width `4.500056e-23`; a slack of
   `8 eps * max(scale, 1.0)` is `1.8e-15` there, forty million times the
   bracket's own width, and calls every value in the neighbourhood inside.
2. A slack proportional to the operation count. Hypothesis falsified this on a
   two-cut-set series-parallel structure at
   `rates = [0.0078125, 5.960464477539063e-08]`, `hours = 762213`, where the
   exact reliability sits `2.7e-17` below the lower bound while the
   operation-count slack allowed `3.4e-18`.

The reason is **conditioning**, not operation count: the min-cut bound
evaluates `1 - prod q_i` with every `q_i` close to 1, whose absolute error is
of order `eps` whatever the result, so its relative error is
`eps / (1 - prod q_i)` -- 9e-13 at those inputs. The slack is now propagated
from the expression's own conditioning, term by term, by
`esary_proschan_bounds` itself. Both counterexamples are pinned as named
regression tests.

## 13. Monte Carlo

For any continuous lifetime `L` with cumulative hazard `H`, the quantity `H(L)`
is exactly `Exponential(1)`-distributed, so a draw is `E ~ Exponential(1)`,
`L = H^-1(E)`. This is inverse transform written in hazard space; it needs no
distribution-specific sampler and is exact for every distribution here,
including `ScaledHazard`, whose inverse is the base's at `E / factor`.

Intervals are **Wilson score** intervals (Wilson, E. B., 1927, *JASA* 22,
209-212), which stay inside `[0, 1]` and do not collapse to zero width when no
failure is observed -- both of which matter for a redundant diagram whose
unreliability is small by construction.

**Budgets are derived, not chosen.** `n = z^2 Q (1 - Q) / h^2` at a pilot
estimate of `Q`, for a declared target half-width of 1 percent of the system
unreliability. The resulting counts are not round -- 14403160 for the redundant
pair, 1221257 for the bus, 70010 for the load-sharing pair -- and the interval
actually achieved is reported rather than assumed. All 8 runs lie within
`1.9129` standard errors of their closed form, 0 of 8 outside the interval.

The simulation shares no algebra with the closed forms it is scored against:
standby is simulated by drawing the primary's lifetime once, trying the switch
and ageing the spare at its declared fraction; shared load by drawing the first
failure and redrawing the survivor's residual life at the full rate; a network
by testing its minimal path sets.

## 14. The mission profile

`age_eff = sum over phases of duration * duty_cycle * stress_factor`, and
`H = H_block(age_eff)`. This is the cumulative-exposure model (Nelson, W.,
1990, *Accelerated Testing*, Wiley, chapter 10; **structure only, no number
taken from it**).

**Exact for an exponential block**: a constant hazard rate makes `H` linear in
age, so the sum of phase contributions is the integral. Verified against a
`quad` integration of the piecewise-constant stress profile, worst relative
difference `0.000000e+00` over four rates.

**An approximation otherwise, and the check is declared and FAILED.**
`validation/validate_mission.py` declares that the model agrees with
calendar-time integration of a scaled hazard rate to within 1 percent for a
non-exponential block, and reports that it does not: worst divergence
**1852.73 percent** at Weibull shape 3.5 under a stress factor of 4, with 4 of
20 cases agreeing and all four being the exponential ones. The two models
answer different physical questions -- cumulative exposure says the stress
advances the block along its own ageing curve, calendar integration says the
stress multiplies its instantaneous hazard at its current calendar age -- and
this package implements the first. The check is published rather than removed
because a reader who sees "stress factor 3.0" may assume the second.
