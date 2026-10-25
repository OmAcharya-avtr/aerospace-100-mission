# Methods

What this document is for: the formulations, the choices behind them, and the
deviations from the product specification, in one place. Every number quoted
here comes from a script in `validation/` and the script is named.

## 1. Visibility

### The mask margin

A contact window is a maximal interval on which

```
m(t) = elevation(t) - mask(azimuth(t))  >=  0
```

in degrees. Elevation is measured against the local WGS-84 ellipsoid normal,
which is what a levelled antenna mount is aligned to, and the mask is evaluated
at the instantaneous azimuth. `contactsched.visibility.mask_margin_deg` is the
single place elevation and mask are combined, so the coarse scan, the bisection
and every test agree by construction about what visibility means.

### The search

A uniform coarse scan locates sign changes; bisection refines each to a declared
tolerance. Bisection and not a secant or Newton step, because `m` is not smooth:
the mask is only piecewise linear in azimuth, so `m` has corners and a
derivative-based method can step across one. Bisection converges on the bracket
regardless, in `ceil(log2(step / tolerance))` evaluations, and all brackets in
one station-spacecraft pair are refined simultaneously so that one batched
propagation serves every root.

A sampled root search can miss a pass shorter than its step. That is a property
of the method, not a defect, and `validate_convergence.py` measures it: against a
2 s reference, a 240 s step loses two boundaries on one of three cases, and steps
of 30, 45, 60 and 120 s lose none on any of them.

### Frames

`ECEF` here means TEME rotated about the z-axis by the Greenwich Mean Sidereal
Angle, which is the pseudo-Earth-fixed frame with polar motion omitted. The
omissions and their magnitudes:

| Omitted | Magnitude | Effect |
| --- | --- | --- |
| polar motion | under 0.5 arcsecond of pole displacement | under 15 m of station displacement, under 0.0002 deg of elevation |
| equation of the equinoxes, TEME against PEF | under 1.2 arcsecond | comparable |
| UT1 against UTC | under 0.9 s | up to 0.0037 deg of Earth rotation, roughly 0.4 km of station displacement |

The third dominates and is the reason window boundaries are reported to the
*search* resolution rather than claimed accurate to it. Use `astropy` where
arcsecond frames matter; it is not installed here and nothing in this repository
is checked against it.

### The independently derived bound

`contactsched.geometry.max_pass_duration_s` is a closed-form upper bound on one
pass, derived from the sine rule in the triangle with vertices at the Earth's
centre, the station and the spacecraft:

```
gamma_max = arccos((R / r) cos el) - el
duration <= 2 gamma_max / (n - omega_E)
```

with `R` the station's geocentric radius, `r = a (1 + e)` the orbit's apogee
radius, `n = sqrt(mu / a**3)` the mean motion and `omega_E` the Earth rotation
rate. It is deliberately loose -- spherical Earth, circular orbit at apogee
radius, mean rather than instantaneous angular rate -- because its purpose is to
falsify a window computation that has gone wrong, not to predict a pass.
Measured ratios of the longest observed pass to the bound run from 0.758 to
0.924 over 22 station-spacecraft pairs. It is not valid above geosynchronous and
raises there; just below geosynchronous it is correct and vacuous, of order 1e9
s, which `tests/test_geometry.py` asserts explicitly so that a returned number is
not mistaken for an informative one.

## 2. The scheduling model

### Candidates

A `ContactRequest` asks for `min_contacts` to `max_contacts` contiguous contacts
of `duration_s` seconds, for one spacecraft, on any antenna from a declared
allowed set. A `Candidate` is one concrete placement: a request, an antenna and a
start time inside a visibility window. Candidates are generated on a grid inside
each window, so the model is a **discretisation** and its optimum is the optimum
over that grid.

One candidate is added flush with the end of each window whenever the grid does
not already land there. Without it a request whose duration is just under the
window length can be unplaceable purely because the grid missed, which is a
discretisation artefact rather than a contention result.

### Feasibility

An assignment is feasible when no two selected candidates on the same antenna
overlap once the antenna's turnaround is added to each, no two for the same
spacecraft overlap, no request exceeds its ceiling, and every contact lies inside
the window it came from.

Minimum contact counts are deliberately **not** in that list.
`contactsched.problem.check_feasible` does not check them and
`contactsched.problem.shortfalls` measures them. A schedule that misses a
declared minimum is feasible and short. Treating the miss as infeasibility would
throw away the number this product exists to report.

### Priority weights

`priority_weight(p, base) = base ** (1 - p)`, so priority 1 weighs 1 and each
step down halves at the default base of 2.

This function is the **declared promise** of the priority scheme, and naming it
matters. A geometric weight says that one priority-1 contact outweighs any number
of lower-priority contacts only when `base` exceeds the number of competitors. At
`base = 2` with five priority-2 requests, the solver will correctly drop a
priority-1 contact to gain three priority-2 contacts. That is not a bug in the
solver; it is what the weights asked for, and it is exactly the gap between
promise and delivery that `contactsched.fairness` measures.

## 3. The exact method

One binary variable per candidate, maximising the priority-weighted objective
subject to:

- **Antenna exclusion.** For each antenna and each candidate start time on it,
  the candidates whose inflated interval `[start, end + turnaround)` covers that
  instant sum to at most one.
- **Spacecraft exclusion.** The same, per spacecraft.
- **Contact ceilings.** Each request takes at most `max_contacts`.
- **Declared fairness constraints**, optionally: a hard floor at each request's
  `min_contacts`, and a cap on each mission's share of all allocated seconds.

### Why testing start times is sufficient

For intervals the point formulation is **exact**, not a relaxation. A family of
intervals is pairwise disjoint if and only if no instant is covered twice, and
every maximal clique of an interval graph contains the left endpoint of one of
its members, so testing the left endpoints suffices (Wolsey 1998, chapter 1). It
is also far smaller than the pairwise encoding.

That argument is the thing most likely to be wrong, which is why
`contactsched.bruteforce` exists: literal `2 ** n` bitmask enumeration sharing no
part of the formulation, compared against HiGHS on 140 instances under four
configurations with a worst objective difference of 0.000e+00 over 6964736
enumerated subsets (`validate_exact.py`).

### Infeasibility is an output, not an error

When the fairness floors cannot be met, `solve_exact` returns a status of
`infeasible` together with notes naming the binding floors. For each request with
a floor, `standalone_capacity` computes the most contacts it could get with the
whole network to itself, by earliest-finish-time selection over its own
candidates. A floor above that number cannot be met by any schedule, whatever the
contention, so the diagnostic separates "asked for too much" from "lost the
contention". An operator cannot act on "infeasible" but can act on "TER-1 asked
for three contacts and has at most one mutually compatible candidate".

Measured on 160 instances, 144 are infeasible under their own declared floors.

## 4. Deviation from the specification: HiGHS, not PuLP

The product specification named a PuLP mixed-integer program. Measured in the
environment this product was built in:

```
pulp.__version__                        : 4.1.0
pulp.listSolvers(onlyAvailable=True)    : []
'PULP_CBC_CMD' in pulp.listSolvers()    : False
```

`pulp` ships no solver, and `PULP_CBC_CMD` is absent from the full solver list
entirely, CBC having been unbundled in the 4.x series. A PuLP-backed exact
baseline would raise at solve time rather than return a schedule.

`scipy.optimize.milp` wraps HiGHS, is part of a dependency the package already
needs, and solves every instance here to proven optimality. `pulp` is **not** a
dependency of this package; it is named in `README.md` among the alternatives
together with this measurement, which any reader can reproduce with
`python -m contactsched backends`.

`contactsched.exact.available_backends` probes for `pulp` and OR-Tools at
runtime and reports them absent or unusable rather than raising. HiGHS is the
default and the only backend any test or validation script exercises; nothing in
this repository has ever been measured on another one, and passing another
backend name to `solve_exact` raises with the probe's output in the message.

## 5. Contention

For an antenna `s` and a time bin `b`:

- **supply** is the **union** of the visibility windows on `s` inside `b`. It
  must be a union, because an antenna serves one spacecraft at a time: two
  overlapping windows supply the time they jointly cover exactly once. Measured
  overlap ranges from 0.0 s at an equatorial antenna to 4694.5 s at a polar one
  over a 24-hour horizon.
- **demand** apportions each request's declared requirement across the
  opportunities it has: a request needing `requested_seconds` with
  `effective_seconds` of usable window contributes
  `requested_seconds * overlap(window, bin) / effective_seconds` to each bin. A
  request with many opportunities presses lightly on each; a request with one
  presses its whole requirement onto that one. That property is what makes the
  ratio informative: a station that is merely busy is not contended, a station
  that is the *only* option for several requests at once is.

"Usable" window excludes any window shorter than the request's contact duration.
Without that exclusion a request surrounded by useless three-second windows would
appear to have ample opportunity.

The ratio is an apportionment and not a feasibility test. It knows nothing about
turnaround times, spacecraft radio exclusion, or the combinatorial structure that
decides who actually loses. `validate_predictors.py` measures exactly how large
that gap is: the per-bin ratio reaches 0.6010 to 0.6228 area under the ROC curve
as a predictor of starvation, against 0.9230 for the per-request ratio.

## 6. Fairness

- **Requested seconds** are `min_contacts * duration_s`, the number the
  requester declared they needed. Not `max_contacts`: measuring satisfaction
  against a ceiling would make every schedule look worse than it is.
- **Satisfaction** is allocated over requested seconds, capped at 1 for the
  indices and reported uncapped as well. A request with `min_contacts = 0` has
  no demand, cannot be starved, and is excluded from the indices and listed
  separately so the exclusion is visible.
- **Jain's index** is `(sum x)**2 / (n * sum x**2)` over the capped
  satisfactions, in `[1/n, 1]`. The all-zero case returns 1.0 by convention --
  every request got the same, namely nothing -- and the max-min share of the
  same allocation returns 0.0. Both are always reported together for that
  reason, and `format_fairness` prints a sentence naming the convention whenever
  it applies. The index is computed after normalising the shares by their
  maximum, which is mathematically free and which fixes a
  `ZeroDivisionError` Hypothesis found at a share of 5e-324.
- **Priority inversion.** The declared promise of an integer priority is that a
  lower number is served no worse. An inversion is an ordered pair of requests
  with strictly different priorities in which the more urgent one has the
  strictly lower satisfaction, with a tolerance, because a difference of 1e-16 is
  a floating-point artefact. Both the per-pair count and the per-priority mean
  delivered satisfaction are reported, because the aggregate promise can hold
  while individual inversions exist.

Jain's index and the max-min share are **descriptions of one schedule**. They do
not say a schedule is fair; they say how equal this allocation is against this
declared demand.

## 7. The learned component

Covered in full in `MODEL_CARD.md`. The design constraint worth repeating here:
every feature must be computable **before** the assignment is solved, because a
conflict predictor whose inputs include the schedule is not a predictor and
would be useless for the task it exists for, which is telling a planner at
submission time that a request is about to be dropped.

The second constraint is the baseline. Three contention ratios are available;
the one the specification named is the weakest; the strongest is the baseline;
all three are reported; and the learned model is compared against the best
baseline **per metric**, because nominating one baseline would have handed the
learned model an easier comparison than the baselines actually offer. The
baseline wins on calibration and that result is published unchanged.

## 8. What is deliberately not here

- No propagator. `sgp4` is the propagator.
- No frame model beyond a GMST rotation. `astropy` is the frame model.
- No search beyond HiGHS. OR-Tools CP-SAT is the better scheduler at scale.
- No model of a real ground-station network: no catalogue, no keyhole regions,
  no slew rates, no elevation-dependent data rates, no weather, no service level
  agreements, no real request mix.

What is here, and is not available off the shelf, is the accounting: which
request was starved, by how many seconds, how unequally the satisfied fraction
was distributed, and whether the delivered order matched the declared priority
order.

This software is research-grade. It is not flight-qualified, not certified, and
not approved for operational aerospace use.
