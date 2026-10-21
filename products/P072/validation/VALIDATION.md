# Validation — downlinkpick 0.1.0

**Validation level 3 (research grade).** Not flight-qualified, not certified,
not approved for operational aerospace use.

Every number in this file was produced by running a script in this directory
during the build session of 2026-10-21, and every script writes its raw output
beside itself as `<script-name>_output.txt`. Nothing here is estimated, carried
from a previous session, or transcribed by hand. The commands that reproduce
all of it are at the bottom.

## Environment the numbers were measured against

| Component | Version |
|---|---|
| python | 3.13.16 |
| numpy | 2.5.3 |
| scipy | 1.18.1 |
| scikit-learn | 1.9.1 |
| matplotlib | 3.11.2 |
| pulp | 4.1.0 |
| pyyaml | 6.0.3 |
| pandas | 3.0.5 |
| joblib | 1.6.0 |
| pytest | 9.1.1 |
| hypothesis | 6.170.5 |
| ruff | 0.17.1 |
| astropy | not installed |
| CPU cores available | 2 |

Recorded at the head of every output file by `validation/_bootstrap.py`.

**PuLP has no solver attached in this environment.**
`pulp.listSolvers(onlyAvailable=True)` returns the empty list: PuLP 4.x does not
bundle CBC and `pulp.PULP_CBC_CMD` no longer exists. The MILP is therefore
*modelled* with PuLP — which is also what writes the portable LP file — and
*solved* with `scipy.optimize.milp`, the HiGHS branch-and-bound solver shipped
inside SciPy. Check 3 of `validate_exact.py` compares the two emissions of the
model term by term so that this is a statement about which solver ran and not
about which problem was solved.

## Summary

| # | Check | Reference | Result | Tolerance | Output |
|---|---|---|---|---|---|
| 1 | Window capacity against adaptive quadrature | `scipy.integrate.quad` | worst **8.384938e-09** relative over five profiles | 1e-7 relative | `validate_capacity_output.txt` |
| 2 | Window capacity against a midpoint Riemann sum | 400 000 sub-intervals | worst **6.960605e-02 of the derived bound** | `4 * sum(step heights) * h / capacity`, derived per profile | `validate_capacity_output.txt` |
| 3 | Capacity additivity over a split | integral identity | worst **1.225630e-16** relative over 15 splits | 1e-12 relative | `validate_capacity_output.txt` |
| 4 | `time_for_bits` inverted | `delivered_bits` | worst **9.574388e-13** relative over 25 points | 1e-9 relative | `validate_capacity_output.txt` |
| 5 | Shipped cases' capacities | `scipy.integrate.quad` | worst **4.031808e-15** relative over 7 windows | 1e-7 relative | `validate_capacity_output.txt` |
| 6 | MILP optimum against exhaustive enumeration | enumeration, no solver | worst **0.000000e+00** relative over **240** instances | 1e-9 relative plus the solver's 1e-6 absolute MIP gap | `validate_exact_output.txt` |
| 7 | PuLP emission against the SciPy arrays | same `MilpModel` | **0.000000e+00** maximum coefficient difference | 0 absolute | `validate_exact_output.txt` |
| 8 | LP relaxation as an upper bound | `scipy.optimize.linprog` | **0** violations in 40 instances; integrality gap mean **1.901325e-01** | 1e-9 relative | `validate_exact_output.txt` |
| 9 | Greedy optimality gap, tight family | proved optimum | mean **4.843046e-02**, median **3.776537e-02**, q0.95 **1.447293e-01**, max **2.491356e-01**, optimal on **51 of 300** | distribution reported, not a threshold | `validate_greedy_gap_output.txt` |
| 10 | Greedy never beats the optimum | proved optimum | **0** of 1500 instances | 1e-9 relative plus the 1e-6 MIP gap | `validate_greedy_gap_output.txt` |
| 11 | Invariants over both selectors | `check_invariants` | **0** violations in **4000** selections | capacity `max(1 bit, 1e-6 x capacity)`; value 1e-9 relative | `validate_invariants_output.txt` |
| 12 | Transitive precedence from direct rows only | explicit closure walk | **7726** transitive pairs checked, **0** violations, deepest closure **8** | exact set and ordering test | `validate_invariants_output.txt` |
| 13 | The invariant checker's own sensitivity | six deliberate corruptions | **6 of 6** detected | detection, not a number | `validate_invariants_output.txt` |
| 14 | Capacity sweep monotone where no expiry is crossed | integer program | **0** decreasing steps crossing no expiry; **1** that does; **173** convexity violations | 1e-9 relative | `validate_marginal_output.txt` |
| 15 | Marginal value against an actual extension | re-solve | worst **0.000000e+00** relative over **90** window-prices | 1e-9 relative | `validate_marginal_output.txt` |
| 16 | Left-behind value accounting | inventory total | **40** instances, residual at or below **1.937e-16** | 1e-9 relative | `validate_marginal_output.txt` |
| 17 | Left-behind capacity shortfall frees the room claimed | re-measured capacity | **0 of 101** failed | 1 bit absolute | `validate_marginal_output.txt` |
| 18 | Selection stability under perturbed values | unperturbed selection | at sigma 0.05 the exact solver keeps Jaccard **0.99313** and value **0.999747**; at sigma 0.40, **0.95919** and **0.992742** | measured curve | `validate_stability_output.txt` |
| 19 | Learned value model against its baselines | oracle selection | declared wins scenario 1 (**0.000404** against **0.009582**); learned or hybrid wins scenarios 2-5 | measured, no threshold | `validate_valuemodel_output.txt` |
| 20 | Learned compressibility predictor against its baselines | oracle ratios | delivered value **0.962760** (learned+2sigma) against **0.900303** (instrument constant) and **0.827558** (entropy floor) | measured, no threshold | `validate_compress_output.txt` |
| 21 | CLI, including every failure path | exit status and stderr | **7** commands exit 0, **3** failure paths exit 2 with a message and no traceback | exact exit status | `validate_cli_output.txt` |
| 22 | Committed example cases reproduce from their seeds | byte comparison | **3 of 3** unchanged | byte-identical | `regenerate_cases_output.txt` |
| 23 | The three solver counterexamples, re-measured post-fix | enumeration and Dantzig bound | all three feasible, none exceeding the enumeration optimum or the fractional bound | 1e-6 absolute MIP gap | `validate_exact_output.txt` |
| 24 | The pinned seed-36 drop, re-measured | capacity sweep | **40.146248887935315** to **34.97126699929167**, 1 expiry crossed | reproduces exactly | `validate_marginal_output.txt` |

## 1. Contact-time accounting (`validate_capacity.py`)

The capacity of a window is `C = integral of R(t) dt` over the window, in bits.
The library evaluates it in closed form as a trapezoidal sum over the
piecewise-linear breakpoints, which is exact for a linear integrand on each
segment. Two independent references are used: adaptive Gauss-Kronrod quadrature
through `scipy.integrate.quad`, which shares no code with the sum, and a
midpoint Riemann sum over 400 000 sub-intervals, which has a different error
mechanism again.

| Profile | Closed form [bit] | quad relative | Riemann relative | Riemann tolerance |
|---|---|---|---|---|
| flat 50 Mbit/s, 300 s | 1.5000000000e+10 | 0.000e+00 | 1.272e-16 | 1.000e-09 |
| sine floor 0.22, 9 points, 480 s | 4.0905538588e+10 | 1.865e-16 | 0.000e+00 | 1.000e-09 |
| sine floor 0.0, 25 points, 612 s | 3.1124385444e+10 | 7.354e-16 | 2.295e-12 | 1.000e-09 |
| irregular breakpoints with a step | 8.6546250000e+10 | 8.385e-09 | 5.200e-07 | 7.470e-06 |
| zero-rate opening segment | 9.5000000000e+09 | 2.008e-16 | 1.551e-12 | 1.000e-09 |

The Riemann tolerance is derived, not chosen. A profile containing a *rate step*
has one sub-interval straddling the discontinuity, where the midpoint value is
wrong by up to the step height, so the reference's own error is bounded by
`4 x sum(step heights) x h`. That bound is computed per profile and the measured
disagreement sits at 6.960605e-02 of it. The disagreement is a property of the
reference, not of the library, which is why it is bounded rather than tuned.

**This check found a real defect on its first run.** The closed-form integral
read the rate at each segment's endpoints back through `rate_bps`, which is
right-continuous at a rate step, so the segment *ending* at a step was
integrated with the step's higher value. The error was **6.065e-02 relative** on
the stepped profile and additivity failed by **2.953e-02** at a split of 73.270 s.
The integral now interpolates within each segment. Both are pinned in
`tests/test_regressions.py`.

## 2. The exact solver against exhaustive enumeration (`validate_exact.py`)

240 instances of 7 products over 2 windows, seeds 1 to 240, each solved twice:
once by `scipy.optimize.milp` and once by enumerating every one of up to 2187
complete assignments with no solver in the loop.

```
instances compared                  : 240
worst relative objective difference : 0.000000e+00 at seed 1
selections with an equal-value tie  : 0
solutions needing overflow repair   : 0
enumeration assignments, min / max  : 48 / 2187
```

**This check found the single most serious defect in the build.** With the
capacity rows left in bits — coefficients of order 1e9 against objective
coefficients of order 1e1 — HiGHS returned `status == 0` with an objective
**below** the true optimum on **12 of 240** instances, the worst by
**4.568e-01 relative** at seed 69, and a hard solve error on one more. Nothing
in the solver's own status reporting indicated this; only the enumeration did.
Each capacity row is now divided by that window's capacity, so its coefficients
are the fraction of the window a product occupies and its right-hand side is 1.
Pinned as `test_regression_unscaled_capacity_rows_made_highs_report_a_wrong_optimum`.

The two emissions of the model agree exactly:

| seed | columns | rows | max abs objective diff | max abs row diff | max abs rhs diff |
|---|---|---|---|---|---|
| 3 | 13 | 9 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 17 | 12 | 9 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 101 | 13 | 12 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 240 | 10 | 12 | 0.000e+00 | 0.000e+00 | 0.000e+00 |

The LP relaxation is a valid upper bound on all 40 instances tested, with an
integrality gap of mean **1.901325e-01**, median **1.550808e-01** and maximum
**8.088386e-01**. That is loose, and is the reason the integer program is solved
rather than bounded.

## 3. The greedy optimality gap, as a distribution (`validate_greedy_gap.py`)

300 instances per family, seeds 1 to 300, 24 products over 3 windows. The gap is
`(optimum - greedy) / optimum`.

| Family | Matched optimum | Mean | Median | q0.95 | Max |
|---|---|---|---|---|---|
| tight (capacity 45% of queue) | 51 / 300 | 4.8430e-02 | 3.7765e-02 | 1.4473e-01 | 2.4914e-01 |
| very tight (capacity 20%) | 79 / 300 | 4.4287e-02 | 2.4051e-02 | 1.5516e-01 | 2.9662e-01 |
| loose (capacity 85%) | 42 / 300 | 4.4650e-02 | 3.4288e-02 | 1.3132e-01 | 2.6770e-01 |
| tight, no dependencies | 58 / 300 | 3.7845e-02 | 3.0326e-02 | 1.0804e-01 | 2.3542e-01 |
| tight, dense dependencies | 46 / 300 | 9.3385e-02 | 6.6936e-02 | 2.9428e-01 | 5.8157e-01 |

Histogram of the tight family, which is the regime this product is for:

| Gap range | Count | Fraction |
|---|---|---|
| exactly optimal (<= 1e-12) | 51 | 0.1700 |
| 1e-12 to 1e-2 | 34 | 0.1133 |
| 1e-2 to 2e-2 | 26 | 0.0867 |
| 2e-2 to 5e-2 | 71 | 0.2367 |
| 5e-2 to 1e-1 | 77 | 0.2567 |
| 1e-1 to 2e-1 | 35 | 0.1167 |
| above 2e-1 | 6 | 0.0200 |

The mean is the least useful number in that table. Greedy is already optimal on
17 per cent of the tight family and within one per cent on another 11 per cent,
and it is more than 20 per cent off on 2 per cent of it. Dependency density
costs more than capacity tightness: the dense-dependency family has roughly
twice the mean gap and more than twice the 95th percentile, which is the
bundle-rollback behaviour of the baseline showing up in the measurement.

Dantzig's fractional bound is valid on all 60 instances tested and loose by
mean **1.254391e-01**, range **2.584912e-02** to **4.307902e-01**.

## 4. Invariants (`validate_invariants.py`)

4000 selections — both selectors over 400 instances of five shapes — with no
violation of any of: no double downlink, expiry, capacity, dependency closure,
precedence by window, value conservation. The worst value-conservation residual
over all 4000 is **0.000e+00** and the worst capacity headroom is negative on
every window of every instance, which is the expected sign.

The transitive-precedence claim is checked rather than assumed: the model
imposes the precedence row on *direct* prerequisites only, and **7726**
transitive pairs over **400** instances containing closures of size up to **8**
all hold.

The checker itself is checked, because a checker that never fails is not
evidence. Six deliberate corruptions of a valid selection — a duplicated
assignment, an inflated total, an inflated per-assignment value, an understated
size, an unknown product, and a delivery after expiry — were all detected.

## 5. The marginal value of contact time (`validate_marginal.py`)

**A result worth stating plainly: the marginal value of contact time can be
negative, and this is not a defect.** A window's end is also the epoch at which
everything in it is delivered. Extending a window moves that epoch later, so a
product whose expiry falls inside the extension loses the window entirely. The
capacity sweep of seed 36 falls from **40.146248887935315** to
**34.97126699929167** between a 440-second and a 460-second extension of pass1,
a **12.89 per cent** drop, because the longer pass is too late for a product the
shorter one would have delivered in time. `WindowMarginal.expiry_crossings`
reports the products responsible so that a negative price arrives with its
explanation.

Measured over 40 instances swept from 0 to 600 added seconds in 31 points:

```
decreasing steps crossing no expiry        : 0 (must be 0)
decreasing steps that cross an expiry      : 1
worst relative drop                        : 1.289032e-01 at seed 36
convexity violations across all sweeps     : 173
```

173 convexity violations is the expected result and is reported rather than
fixed: the objective of an integer program is a step function of capacity, so
the marginal value of contact time is not a derivative and the curve is not
concave.

The reported marginal value matches an actual re-solve to **0.000000e+00**
relative over 90 window-prices, and the extension's bias is measured: holding
the window's final rate across the extension delivers **2.000000** times the
bits that continuing the falling profile would, at a 300-second step, so the
marginal value reported here is an upper estimate by that factor.

The left-behind report accounts for the whole inventory on all 40 instances
tested, with a worst residual of **1.937e-16**. Its capacity shortfalls are
checked two ways, and the two counts differ for a reason worth recording:

```
capacity shortfalls probed                 : 101
shortfalls that do not free enough room    : 0 (must be 0)
shortfalls after which a re-solve selects  : 39 of 101
```

The first count is the claim the report makes and it holds everywhere. The
second is a question the report does not answer: re-solving with the extra room
may spend it on a different product of higher value, so a product that fits is
not a product that is chosen. The gap between 101 and 39 is the honest limit of
a per-product shortfall number.

## 6. Selection stability under perturbed values (`validate_stability.py`)

24 instances, 24 trials per noise level, values perturbed multiplicatively by
`exp(N(0, sigma))`.

| sigma | exact: mean Jaccard | exact: value kept | greedy: mean Jaccard | greedy: value kept |
|---|---|---|---|---|
| 0.00 | 1.00000 | 1.000000 | 1.00000 | 1.000000 |
| 0.01 | 0.99904 | 0.999990 | 0.99863 | 1.000183 |
| 0.02 | 0.99763 | 0.999971 | 0.99615 | 1.000611 |
| 0.05 | 0.99313 | 0.999747 | 0.99279 | 1.001491 |
| 0.10 | 0.98356 | 0.998976 | 0.99013 | 1.001615 |
| 0.20 | 0.97131 | 0.997548 | 0.98146 | 1.001526 |
| 0.40 | 0.95919 | 0.992742 | 0.95829 | 0.998671 |

Greedy's retained fraction above 1.0 is not an error. Greedy's unperturbed
selection is not optimal, so a perturbation that reshuffles its density ordering
can land on a better selection, and the fraction is measured against greedy's
own unperturbed value. The exact solver has no such headroom.

The practical conclusion is in the last section of that output. On this family
the mean greedy gap is **4.758763e-02** and the value the exact solver loses to
value noise is **9.943610e-06** at sigma 0.01 and **7.257854e-03** even at
sigma 0.40, so **the optimality is worth having at every noise level measured**.
There is no crossover up to sigma 0.4.

## 7. The learned value model (`validate_valuemodel.py`)

Trained on 6000 synthetic products from seed 1001, 70/30 split; benchmark
instances from seeds 5001 to 5030, which share no product with the training
table. Regression error on the held-out split:

| Estimator | MAE | RMSE | Spearman | bias |
|---|---|---|---|---|
| learned forest | 0.962341 | 1.585054 | 0.912832 | 5.278e-03 |
| size/age fitted | 1.863225 | 2.561772 | 0.557234 | 4.825e-03 |
| train mean | 2.233730 | 3.034266 | nan (a constant vector has no ranks) | 5.277e-02 |

**End-to-end regret is the number that matters and the specification's
prediction is the measured result, with one refinement.** The prediction was
that the exact solver on declared values beats everything and the learned model
helps only where declarations are absent. Measured:

| Scenario | Best non-oracle arm | declared-median regret fraction | learned regret fraction |
|---|---|---|---|
| 1 complete and fresh | **declared-median** | **0.000404** | 0.009582 |
| 2 complete but stale | learned | 0.015860 | **0.009582** |
| 3 30 per cent missing | **hybrid**, at 0.002253 | 0.015418 | 0.005488 |
| 4 all missing | learned | 0.986665 | **0.008156** |
| 5 40 per cent missing, half of the rest stale | learned | 0.021350 | **0.005799** |

The declared-value passthrough is the best arm in scenario 1 and in no other. It
is beaten by a factor of 24 in scenario 1 terms as soon as the declarations go
stale, and it collapses entirely when they are absent: with no declaration
anywhere, both fallbacks reduce to zero, the passthrough has no information, and
its regret fraction is **0.986665** — it delivers almost nothing of value.

In scenario 3 the hybrid arm, which keeps every declaration and fills only the
gaps, is the best of all at **0.002253**.

The refinement to the prediction is that the learned model also helps where
declarations are *stale*, not only where they are absent, which the
specification's own wording ("missing or stale") anticipated and its one-line
prediction did not.

The ranking is a property of the estimators and not of the solver: through the
greedy selector the ranking on scenario 5 is identical — learned, hybrid,
declared-median, size-age, declared-zero — with regret fractions 0.011111,
0.011621, 0.033167, 0.042057 and 0.315365 against the exact solver's 0.005799,
0.009625, 0.021350, 0.036160 and 0.319707.

Uncertainty output: the forest reports the standard deviation of its per-tree
predictions. Measured coverage of the bands, against the nominal Gaussian
figures:

| Band | Nominal | Measured |
|---|---|---|
| +-1 sigma | 0.6827 | 0.688889 |
| +-1.96 sigma | 0.9500 | 0.941111 |
| +-2 sigma | 0.9545 | 0.947222 |

The band is **not calibrated** and is reported as a dispersion measure, which is
the expected behaviour of a random forest's tree spread (Wager, Hastie and Efron
2014). It is still informative about *where* the model is wrong: the rank
correlation between the dispersion and the absolute error is **0.601302**.

## 8. The learned compressibility predictor (`validate_compress.py`)

Trained on 6000 synthetic products from seed 1002, 70/30 split.

| Estimator | MAE | RMSE | bias | max abs error |
|---|---|---|---|---|
| learned forest (mean) | 0.018918 | 0.026270 | 8.335e-04 | 0.175484 |
| learned forest (+2 sigma) | 0.040664 | 0.051649 | 3.954e-02 | 0.206918 |
| per-instrument constant | 0.066879 | 0.086329 | 1.632e-03 | 0.421233 |
| global mean | 0.148160 | 0.178668 | 9.558e-03 | 0.638845 |
| entropy floor H/B | 0.228995 | 0.248018 | 2.287e-01 | 0.516601 |

**The entropy floor is not a bound for a correlated source, and the measurement
says by how much.** Shannon's source-coding theorem bounds the ratio below by
`H/B` for a *memoryless* source; the generator, like a real codec, includes a
correlation gain. **0.997778** of held-out products have a realised ratio below
the floor, the mean floor is **0.580345** against a mean realised ratio of
**0.351597**, and the worst over-estimate is **0.516601**. Planning against the
floor over-reserves capacity by about 65 per cent on average.

End to end, charging the true size at transmission time so that an
under-estimate costs real products:

| Arm | MAE | bias | overflow rate | mean overflow [Gbit] | delivered | unspent |
|---|---|---|---|---|---|---|
| learned-mean | 0.018147 | 9.350e-04 | 0.144444 | 0.376175 | 0.947669 | 0.143863 |
| learned+2sigma | 0.040371 | 3.969e-02 | 0.000000 | 0.000000 | **0.962760** | 0.212954 |
| instrument-constant | 0.066061 | 1.395e-03 | 0.200000 | 1.136124 | 0.900303 | 0.244360 |
| global-mean | 0.154795 | 8.132e-03 | 0.788889 | 2.531673 | 0.710566 | 0.391047 |
| entropy-floor | 0.224043 | 2.240e-01 | 0.000000 | 0.000000 | 0.827558 | 0.468298 |
| oracle | 0.000000 | 0.000e+00 | 0.000000 | 0.000000 | 1.000000 | 0.089909 |

This is the one place in the package where the uncertainty output changes a
decision rather than annotating one. Planning with `ratio + 2 sigma` from the
ensemble dispersion removes every overflow — 0.000000 against 0.144444 — at a
cost of 6.9 points of unspent capacity, and delivers **more** value overall
(0.962760 against 0.947669) because an overflow loses not only the product that
did not fit but everything queued behind it in that pass.

**Here the learned component beats the analytic and operational baselines, and
this is reported as measured in the same spirit as the value-model result
above.** The ranking by delivered value is: learned+2sigma, learned-mean,
instrument-constant, entropy-floor, global-mean.

## Checks that failed, and what was done

Every one of these was found by a check in this directory or by a property test,
and every one is pinned as a named regression test with literal inputs in
`tests/test_regressions.py`.

| Found by | Defect | Measured error | Resolution |
|---|---|---|---|
| `validate_capacity.py` | The closed-form capacity integral read segment endpoints through a right-continuous `rate_bps`, over-integrating any segment ending at a rate step | 6.065e-02 relative on the stepped profile; additivity off by 2.953e-02 | Interpolate within each segment |
| `validate_exact.py` | Unscaled capacity rows made HiGHS report a wrong optimum with `status == 0` | 12 of 240 instances wrong, worst 4.568e-01 relative; 1 hard solve error | Divide each capacity row by its window capacity |
| `validate_marginal.py` | A decreasing step in the capacity sweep, initially treated as a monotonicity failure | 12.89 per cent drop at seed 36 | Not a defect. Modelled and reported as `expiry_crossings`; the check now distinguishes the two cases |
| Hypothesis `test_milp_agrees_with_enumeration` | An absolute one-bit capacity tolerance in the enumeration disagreed with the MILP's relative one | optimum 1.0 against 0 on a window of 8 bits | Feasibility tolerances made relative throughout |
| Hypothesis `test_milp_agrees_with_enumeration` | Rounding a near-integral HiGHS vector at 0.5 and charging full sizes over-committed a window by 5 bits | reported objective 2.0 where the optimum is 1.0 | Dependency-safe overflow repair; status becomes `optimal-rounded` |
| Hypothesis `test_dantzig_bound_dominates_the_optimum` | The same rounding defect, exposed through a violated upper bound | 6.0 against a fractional bound of 5.999998749938903 | As above |
| Hypothesis `test_greedy_never_exceeds_the_optimum` | Greedy beat the MILP by 5.960464477539063e-08 | within the solver's absolute MIP gap | Not a defect. `SOLVER_ABS_GAP_VALUE_UNITS` documents it; the claim is now "optimal to within the gap" |
| `tests/test_contacts.py` | `ContactPlan.extended_window` silently returned an unchanged plan for an unknown window name | a mispriced window would read as worth zero | Raise `KeyError` |
| `tests/test_cli.py` | A PyYAML `ParserError` reached the user as a traceback | exit status was not 2 | Translate to `ValueError` |

**Nothing was resolved by loosening a tolerance to fit a measurement.** The one
tolerance that was changed — the Riemann reference in check 2 — was replaced by
a bound derived from the reference's own error mechanism, and the measured
disagreement sits at 7 per cent of that bound.

The error magnitudes in the table above are measurements taken in this build
session from the runs that found each defect, **before** it was fixed. They are
by construction not reproducible from the current code, which is the point of
fixing them; what *is* reproducible is each pinned regression test, and the
three Hypothesis counterexamples against the solver are additionally re-measured
in check 7 of `validate_exact.py` and check 8 of `validate_marginal.py` so that
their post-fix values appear in a committed output file.

## What is not validated

- **Nothing here is measured against a real recorder inventory, a real contact
  plan or a real codec.** Every number is a statement about the synthetic
  generator in `src/downlinkpick/dataset.py`, whose equations are printed in its
  module docstring. The learned components' performance is a statement about
  that generator and not about any spacecraft.
- **The rate profile is a shape, not a link budget.** There is no antenna
  pattern, no atmospheric model and no modcod table anywhere in this package.
- **Reliability is a per-window scalar.** There is no partial pass, no
  re-planning after a failed pass and no correlation between window outcomes.
- **Precedence is by window, not by time within a window.** Ordering the queue
  inside a pass is not modelled.
- **The PuLP solve path is exercised and tested but was never run**, because no
  solver is available to PuLP in this environment. Its model emission is checked
  term by term against the arrays that were solved; its solve is not.
- **No timing in this document supports a conclusion.** The compute figures
  below are wall clock on two shared cores and move between runs.

## Compute budget

Wall clock on 2 shared cores. **Unstable between runs; recorded for the compute
budget only and no conclusion rests on them.**

| Script | What it does | Wall clock |
|---|---|---|
| `validate_greedy_gap.py` | 1500 greedy and 1500 exact solves | greedy 1.891 s total (1.261 ms mean), exact 45.992 s total (30.662 ms mean) |
| `validate_exact.py` | 240 MILP and 240 enumeration solves | MILP 0.688 s total (2.868 ms mean), enumeration 0.337 s total (1.404 ms mean) |
| `validate_stability.py` | 48 stability curves, 8112 selector invocations | 61.151 s |
| `validate_valuemodel.py` | forest fit plus 5 benchmark scenarios | fit 3.886 s, benchmark 54.716 s |
| `validate_compress.py` | forest fit plus 1 benchmark | fit 1.876 s, benchmark 7.624 s |

Every script finishes well inside the three-minute budget this mission imposes
on any single training, sweep or Monte Carlo run.

## Reproducing every number

From the repository root, after `pip install -e ".[dev]"`:

```bash
python validation/validate_capacity.py
python validation/validate_exact.py
python validation/validate_greedy_gap.py
python validation/validate_invariants.py
python validation/validate_marginal.py
python validation/validate_stability.py
python validation/validate_valuemodel.py
python validation/validate_compress.py
python validation/validate_cli.py
python validation/regenerate_cases.py
python validation/worked_example.py
```

Each writes `validation/<script-name>_output.txt` and exits 0 on success and 1
on any failed check. Every script runs identically from any working directory —
they anchor on `__file__` and change to the repository root — and this is
verified, not assumed:

```bash
for dir in / /tmp . validation; do
  (cd "$dir" && for s in <repo>/validation/validate_*.py <repo>/validation/worked_example.py; do
     python "$s" > /dev/null || echo "FAILED $s from $dir"
   done)
done
```

The test suite and the lint:

```bash
python -m pytest tests/ -q --junitxml=junit.xml
ruff check src/ tests/ examples/ validation/
```

The figures:

```bash
MPLBACKEND=Agg python examples/contact_timeline.py
MPLBACKEND=Agg python examples/solver_comparison.py
MPLBACKEND=Agg python examples/greedy_gap_distribution.py
MPLBACKEND=Agg python examples/marginal_value_curve.py
MPLBACKEND=Agg python examples/left_behind_report.py
MPLBACKEND=Agg python examples/value_model_benchmark.py
MPLBACKEND=Agg python examples/compressibility_benchmark.py
MPLBACKEND=Agg python examples/stability_curve.py
```
