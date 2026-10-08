# Validation evidence — invariantset 0.1.0

**Validation level 2.** Analytic known-answer checks against sets computed by
hand, property tests against the set-algebra identities, and a measured
characterisation of the three numerical failure modes. **Research-grade. Not
flight-qualified, not certified, not approved for operational aerospace use.**
No learned component, no AI, no MODEL_CARD.

Every number in this file and in README.md was produced by a script in this
directory, executed in this container on 2026-10-08, with its raw stdout
committed beside it as `<script>_output.txt`. Nothing here was copied from a
paper, estimated, or rounded by hand. The checks that undercut this package
are in the tables on purpose and are marked.

**Which set this is.** This package computes the **maximal robust invariant
set** `S_inf` inside a declared constraint set `X`: the set of states from
which `x in X` can be held for every admissible disturbance sequence, forever.
It does **not** compute the **minimal robust positively invariant set**
`F_inf = sum_{i>=0} A^i W` of Raković et al. 2005. Section 2 check 1 shows the
two numbers side by side on the same data, where they differ by a factor of
two. Confusing them is the only real correctness question in this package.

**Environment.** Python 3.13.16 on Linux 6.18.44, NumPy 2.5.3, SciPy 1.18.1,
Matplotlib 3.11.2, pytest 9.1.1, Hypothesis 6.168.5. `os.cpu_count()` = 2 and
`len(os.sched_getaffinity(0))` = 2: **two cores, shared and contended with
sibling build agents.** LP solver `scipy.optimize.linprog(method="highs")`;
convex hulls and volumes from `scipy.spatial.ConvexHull` (Qhull). Runtime
dependencies are NumPy and SciPy only; Matplotlib is needed for plotting.
Raw output: `validate_environment_output.txt`.

**Timings move, counts do not.** The wall-clock figures quoted below are
single runs on two contended cores and move by 10-20 % between runs. The
primary numbers in this document are **iteration counts, facet counts, vertex
counts and margins**, which are deterministic and reproduce exactly.

**Compute budget.** The whole test suite is 200 tests in about 58 s. The
slowest single computation committed anywhere in this repository is the
400-iteration non-converging run at `redundancy_tol = 5e-2` in section 4C,
about 20 s. No computation here approaches the 3-minute budget.

**Test suite.** `python -m pytest tests/ -q` from the repository root:
**200 passed, 0 failed, 0 skipped, 0 xfail, 0 error**, in 57.83 s
(`200 passed in 57.83s`; the figure moves 10-20 % between runs, the count does not).
Raw output: `pytest_output.txt`.

---

## 1. Support function and set algebra

`validate_support_algebra.py` → `validate_support_algebra_output.txt`, seed
59001.

| Check | Reference / method | Result | Tolerance |
|---|---|---|---|
| Box closed form against the generic LP | `h(c) = c.m + abs(c).r` against `scipy.optimize.linprog` on the same halfspaces | 1000 directions in dimensions 1–4, worst absolute difference **3.552714e-15** | 1e-9 |
| Support function against explicit vertex enumeration | `sup_{x in P} c^T x` over all enumerated vertices | 1800 directions over 180 random polytopes in dimensions 1–3, worst absolute difference **8.881784e-16** | 1e-12 |
| Positive homogeneity `h(t c) = t h(c)`, `t >= 0` | Rockafellar 1970 | 400 cases, worst relative difference **2.865772e-16** | 1e-12 |
| Subadditivity `h(c1+c2) <= h(c1) + h(c2)` | same | 400 cases, worst slack **+8.881784e-16**, i.e. zero to floating point and marginally the wrong sign | <= 1e-12 |
| Minkowski additivity `h_{P+Q} = h_P + h_Q` | Schneider 1993 | 960 cases over 120 pairs, worst absolute difference **2.109424e-15** | 1e-9 |
| `h_{P & Q} <= min(h_P, h_Q)` | definition of the intersection | 300 cases, worst slack **0.000000e+00** | <= 1e-9 |
| Redundancy removal preserves membership | LP-based removal at `tol = 1e-10`, membership compared pointwise | 4000 points over 200 polytopes, **0 mismatches**, **207** rows removed | exact membership |

Boundary points within 1e-7 of a facet are excluded from the membership
comparison: a point exactly on a tangent facet can legitimately flip when that
facet is dropped, and counting those as mismatches would be measuring the
tolerance rather than the algorithm.

## 2. Known answers, computed by hand first

`validate_known_answers.py` → `validate_known_answers_output.txt`.
**FAILED CHECKS: 0.** The hand derivations are reproduced in full in the
docstrings of `tests/test_known_answers.py`.

### 2.1 One dimension, maximal set equals `X`

`lambda = 0.8`, `|w| <= 0.1`, `|x| <= 1`.

```
Omega_0      = [-1, 1]
Pre(Omega_0) = {x : |0.8 x| <= 1 - 0.1} = [-1.125, 1.125]   (0.9 / 0.8 = 1.125)
Omega_1      = Omega_0 & Pre(Omega_0) = [-1, 1] = Omega_0    -> terminate at k = 1
```

Equivalently `S_inf = X` exactly when `b >= w / (1 - |lambda|)`, here
`0.1 / 0.2 = 0.5 <= 1`.

| Quantity | Hand value | Computed | Status |
|---|---|---|---|
| iterations | 1 | 1 | PASS |
| length of `S_inf` | 2.0 | 2.000000000000 | PASS |
| worst facet margin | `0.8 + 0.1 - 1 = -0.1` | -1.000000000000e-01 | PASS |
| **minimal RPI half-width, Raković et al. 2005** | **0.5** | 0.500000000000 | different object |

The maximal half-width is **1.0** and the minimal RPI half-width is **0.5**, a
factor of two apart on identical data. A sibling product in this portfolio
(P051 SimplexGuard) first wrote its 1-D known-answer test against `0.5` and
had to redo it; this package pins both numbers in the same test so the
distinction cannot be lost again.

### 2.2 One dimension, maximal set empty, with a closed form

`lambda = 0.9`, `|w| <= 0.2`, `|x| <= 1`. With `Omega_k = [-r_k, r_k]`,

```
r_{k+1} = min(r_k, (r_k - w) / lambda) = (r_k - 0.2) / 0.9   while r_k < r*
r*      = w / (1 - lambda) = 0.2 / 0.1 = 2.0
s_k     = r_k - r*,  s_{k+1} = s_k / lambda,  s_0 = 1 - 2 = -1
=> r_k  = 2 - (10/9)^k
```

| k | `(10/9)^k` | hand `r_k` | measured `r_k` | abs diff |
|---|---|---|---|---|
| 1 | 1.111111 | 0.888889 | 0.888888888889 | 1.110e-16 |
| 2 | 1.234568 | 0.765432 | 0.765432098765 | 1.110e-16 |
| 3 | 1.371742 | 0.628258 | 0.628257887517 | 1.110e-16 |
| 4 | 1.524158 | 0.475842 | 0.475842097241 | 5.551e-17 |
| 5 | 1.693508 | 0.306491 | 0.306491219157 | 1.665e-16 |
| 6 | 1.881676 | 0.118324 | 0.118323576841 | 8.327e-17 |
| 7 | 2.090751 | **-0.090752** | infeasible | emptiness detected at **k = 7** |

### 2.3 One dimension, exactly on the threshold

`lambda = 0.5`, `|w| <= 0.25`, `|x| <= 0.5`. Here `w / (1 - lambda) = 0.5 = b`
exactly, so the pre-set bound `(0.5 - 0.25) / 0.5 = 0.5` equals `b`: `X` is
invariant with **zero** margin. Measured: converged at k = 1, worst facet
margin **0.000e+00 exactly**. Lowering `b` to 0.49 gives **empty**, as the
closed form requires.

### 2.4 Two dimensions, singular `A`

`A = [[0, 1], [0, 0]]`, box `X` of half-width 1, box `W` of half-width 0.1.
`A` maps `(x1, x2)` to `(x2, 0)`, so `Pre(Omega_0)` is

```
c = e1  : c^T A = (0, 1),  h_W(e1) = 0.1  ->   x2 <= 0.9
c = -e1 : c^T A = (0,-1),  h_W(-e1)= 0.1  ->  -x2 <= 0.9
c = e2  : c^T A = (0, 0),  h_W(e2) = 0.1  ->    0 <= 0.9   (degenerate, trivial)
c = -e2 : c^T A = (0, 0)                  ->    0 <= 0.9   (degenerate, trivial)
```

so `Omega_1 = {|x1| <= 1, |x2| <= 0.9}`, and `Omega_2 = Omega_1`.

| Quantity | Hand value | Computed | Status |
|---|---|---|---|
| iterations | 2 | 2 | PASS |
| facets / vertices | 4 / 4 | 4 / 4 | PASS |
| `h(e1)` | 1.0 | 1.000000000000 | PASS |
| `h(e2)` | 0.9 | 0.900000000000 | PASS |
| area | `2 * 1.8 = 3.6` | 3.600000000000 | PASS |
| worst facet margin | `0.9 + 0.1 - 1 = 0` | 0.000e+00 exactly | PASS |

This case also exercises the degenerate-row path, which a singular `A`
produces and which an implementation that assumes invertible `A` gets wrong.

### 2.5 Two dimensions, decoupled

`A = 0.5 I`, box `X` half-width 1, box `W` half-width 0.1. The recursion
separates into two 1-D recursions with threshold `0.1 / 0.5 = 0.2 <= 1`, so
`S_inf = X`. Measured: k = 1, area **4.000000000000**, worst facet margin
**-0.400000000000** (hand: `0.5 + 0.1 - 1 = -0.4`).

## 3. The illustrative attitude loop

A rigid single-axis attitude channel, `x = [theta, theta_dot]` in `[rad,
rad/s]`, exact zero-order-hold discretisation of `theta_ddot = u` at
`dt = 0.05 s`:

```
A_plant = [[1, dt], [0, 1]] = [[1, 0.05], [0, 1]]
B       = [[dt^2/2], [dt]]  = [[0.00125], [0.05]]
```

Valid for small angles about one axis, rigid body, no actuator lag, control
held constant over each sample.

The gain `K = [8.0, 3.8]` in `[1/s^2, 1/s]` is **derived by hand**, not tuned.
Requiring closed-loop poles at `0.9 +/- 0.1j`, i.e. `z^2 - 1.8 z + 0.82`:

```
trace(A_plant - B K) = 2 - dt^2/2 k1 - dt k2 = 1.8   ->  dt^2/2 k1 + dt k2 = 0.2
det  (A_plant - B K) = 1 - dt k2 + dt^2/2 k1  = 0.82 ->  dt^2/2 k1 - dt k2 = -0.18
adding:  dt^2 k1 = 0.02  ->  k1 = 0.02 / 0.0025 = 8.0
then  :  dt k2 = 0.2 - 0.01 = 0.19  ->  k2 = 3.8
```

Measured: `trace = 1.800000000000`, `det = 0.820000000000`, eigenvalues
`0.900000000 -/+ 0.100000000j`. **These are illustrative design numbers, not
a measurement of any vehicle.**

Constraints `|theta| <= 0.30 rad`, `|theta_dot| <= 0.50 rad/s`, and the input
constraint `|u| = |K x| <= 3.0 rad/s^2` written as two state halfspaces.
Disturbance: an unmodelled angular acceleration of `0.12 rad/s^2` held for one
sample, entering the state as the box with half-widths
`[dt^2/2 * 0.12, dt * 0.12] = [1.500000000e-04 rad, 6.000000000e-03 rad/s]`
(measured, matching the hand values exactly).

| Quantity | Value | Units |
|---|---|---|
| iterations to convergence | **5** | — |
| facets of `S_inf` | **16** | — |
| vertices of `S_inf` | **16** | — |
| area of `X` | 0.544407894737 | rad·rad/s |
| area of `S_inf` | **0.538347284806** | rad·rad/s |
| `S_inf` as a fraction of `X` | **98.886752 %** | — |
| worst facet margin of `S_inf` | **0.000000e+00** | rad or rad/s |
| worst facet margin of `X` itself | **+3.100000e-02** | rad or rad/s |

`X` itself is **not** robustly invariant (positive margin), which is why the
recursion has anything to do; `S_inf` is invariant with **exactly zero**
margin, which is what being maximal means. **1.113 % of the declared
constraint set is outside the certificate**, and from those states the
constraints cannot be held against every admissible disturbance sequence.

## 4. Where a tolerance changes the answer — the required deliverable

`validate_tolerance_sensitivity.py` → `validate_tolerance_sensitivity_output.txt`.
System: `attitude_loop`. Areas in rad·rad/s, margins in `b` units.

### 4A. Redundancy-tolerance sweep, `max_iter = 60`, `convergence_tol = 1e-9`

| `redundancy_tol` | termination | iters | facets | area | invariance margin |
|---|---|---|---|---|---|
| 0 | converged | 5 | 16 | 0.538347285 | 0.000000e+00 |
| 1e-12 | converged | 5 | 16 | 0.538347285 | 0.000000e+00 |
| 1e-09 | converged | 5 | 16 | 0.538347285 | 0.000000e+00 |
| 1e-06 | converged | 5 | 16 | 0.538347285 | 0.000000e+00 |
| 1e-04 | converged | 5 | 16 | 0.538347285 | 0.000000e+00 |
| 1e-03 | converged | 5 | 16 | 0.538347285 | 0.000000e+00 |
| 2e-03 | converged | 5 | 16 | 0.538347285 | 0.000000e+00 |
| 3e-03 | converged | 5 | 16 | 0.538347285 | 0.000000e+00 |
| **4e-03** | **iteration_cap** | 60 | 12 | — | — |
| **1e-02** | **iteration_cap** | 60 | 10 | — | — |
| **5e-02** | **iteration_cap** | 60 | 8 | — | — |

**Two distinct answers across the sweep. The tolerance changed the answer.**
One digit of the redundancy tolerance is the difference between an exact set
and no answer at all.

### 4B. Why, exactly: the threshold is the final shrink margin

The per-iteration shrink margins of the converging run are
`3.100000e-02, 2.150272e-02, 1.250574e-02, 3.631988e-03`. The smallest
positive one is **3.631987773647e-03**. Bisecting the redundancy tolerance
over 30 steps:

| | value |
|---|---|
| converges up to | **3.631987710483e-03** |
| fails from | **3.631987802684e-03** |
| smallest positive shrink margin | **3.631987773647e-03** |

The threshold brackets the final shrink margin to ten significant figures.
**Mechanism:** once the redundancy tolerance reaches the shrink margin, the
rows `Pre(Omega_k)` contributes look redundant and are discarded, so
`Omega_{k+1}` equals `Omega_k` in representation, the sequence stops
contracting, and the recursion neither contracts nor terminates. Since the
shrink margin decays geometrically, **there is no safe fixed redundancy
tolerance for a system whose recursion runs long enough.**

### 4C. What a loose tolerance actually returns

| `redundancy_tol` | cap | termination | facets | area | vs reference | invariance margin |
|---|---|---|---|---|---|---|
| 4e-03 | 150 | iteration_cap | 12 | 0.538885985 | **+0.1001 %** | **+4.294737e-03** |
| 1e-02 | 150 | iteration_cap | 10 | 0.539214085 | **+0.1610 %** | **+5.488889e-03** |
| 5e-02 | 150 | iteration_cap | 8 | 0.595476829 | **+10.6120 %** | **+4.148278e-02** |
| 5e-02 | 400 | iteration_cap | 6 | 0.588122298 | **+9.2459 %** | **+4.158156e-02** |

Each of these is a **wrong** answer, not a coarse one: the returned set is
larger than `S_inf` and fails the independent invariance test. Note also that
`5e-2` returns a different set at cap 150 than at cap 400: **a non-converged
outer bound is a function of the cap, not of the system.**

### 4D. The convergence tolerance is a slack on the guarantee

`redundancy_tol = 1e-9`, `max_iter = 60`.

| `convergence_tol` | iters | facets | area | invariance margin | invariant to 1e-9 |
|---|---|---|---|---|---|
| 0 | 5 | 16 | 0.538347285 | 0.000000e+00 | True |
| 1e-09 | 5 | 16 | 0.538347285 | 0.000000e+00 | True |
| 1e-06 | 5 | 16 | 0.538347285 | 0.000000e+00 | True |
| 1e-04 | 5 | 16 | 0.538347285 | 0.000000e+00 | True |
| 1e-03 | 5 | 16 | 0.538347285 | 0.000000e+00 | True |
| 3e-03 | 5 | 16 | 0.538347285 | 0.000000e+00 | True |
| **5e-03** | 4 | 14 | 0.538430410 | **+3.631988e-03** | **False** |
| **1e-02** | 4 | 14 | 0.538430410 | **+3.631988e-03** | **False** |
| **5e-02** | 1 | 6 | 0.544407895 | **+3.100000e-02** | **False** |

At `5e-3` the recursion stops one iteration early and the set it returns
violates the invariance condition by `3.63e-3` in `b` units. On the `theta`
facet that is **3.63e-3 rad of constraint the returned set does not actually
hold**. The tolerance buys one iteration and costs exactly that much of the
guarantee.

### 4E. A system where no tolerance changes the answer

On `decoupled_2d` the sweep from 1e-12 to 1e-1 returns `converged, 4 facets,
area 4.000000000` every time. `X` is already invariant with margin -0.4, far
larger than any tolerance swept. **Tolerance sensitivity is a property of the
system and its margin, not of the implementation alone**, and the honest
statement is that the user has to check it per problem — which is what
`tolerance_sweep` is for.

## 5. Measured representation growth

`validate_growth_and_stall.py` → `validate_growth_and_stall_output.txt`.
These are counts, not estimates.

### 5A. Two dimensions: `damped_rotation_2d`, `A = 0.995 R(0.2 rad)`

| k | rows before removal | facets kept | removed | vertices | vtx/facet | volume | shrink margin |
|---|---|---|---|---|---|---|---|
| 1 | 8 | 8 | 0 | 8 | 1.0000 | 3.682831465 | 1.748422e-01 |
| 2 | 16 | 12 | 4 | 12 | 1.0000 | 3.473921179 | 1.382588e-01 |
| 3 | 24 | 16 | 8 | 16 | 1.0000 | 3.341039555 | 1.071330e-01 |
| 4 | 32 | 20 | 12 | 20 | 1.0000 | 3.262161266 | 7.971337e-02 |
| 5 | 40 | 24 | 16 | 24 | 1.0000 | 3.221307998 | 5.471075e-02 |
| 6 | 48 | 28 | 20 | 28 | 1.0000 | 3.206039893 | 3.094795e-02 |
| 7 | 56 | 32 | 24 | 32 | 1.0000 | 3.205016434 | 6.667467e-03 |
| 8 | 64 | 32 | 32 | 32 | 1.0000 | 3.205016434 | 2.220446e-16 |

Measured law: **raw rows `8k`, facets kept `4(k+1)`, vertices equal to
facets.** `S_inf` has 32 facets and area 3.205016434, **80.1254 %** of `X`.
Without redundancy removal the representation would reach 64 rows at k = 8 and
keep doubling; with it, growth is linear at **+4 facets per iteration**.

### 5B. Three dimensions: `damped_rotation_3d`, `A = diag(0.99, 0.99, 0.95) R_z(0.2 rad)`

| k | rows before removal | facets kept | removed | vertices | vtx/facet | volume | shrink margin |
|---|---|---|---|---|---|---|---|
| 1 | 12 | 10 | 2 | 16 | 1.6000 | 7.394605991 | 1.699485e-01 |
| 2 | 20 | 14 | 6 | 24 | 1.7143 | 7.001386683 | 1.327821e-01 |
| 3 | 28 | 18 | 10 | 32 | 1.7778 | 6.757995926 | 1.009867e-01 |
| 4 | 36 | 22 | 14 | 40 | 1.8182 | 6.621443104 | 7.268357e-02 |
| 5 | 44 | 26 | 18 | 48 | 1.8462 | 6.559763500 | 4.635276e-02 |
| 6 | 52 | 30 | 22 | 56 | 1.8667 | 6.545828125 | 2.028413e-02 |
| 7 | 60 | 30 | 30 | 56 | 1.8667 | 6.545828125 | 1.110223e-16 |

Measured law: **facets `6 + 4k`, vertices `8 + 8k`.** In three dimensions the
**vertex count grows at twice the facet rate**, and the vertex-to-facet ratio
rises from 8/6 = 1.3333 at `Omega_0` to 56/30 = **1.8667** at convergence.
This is why the V-representation, not the H-representation, is the first thing
to become unusable.

### 5C. Where brute-force vertex enumeration gives up

`Polytope.vertices` solves `C(m, n)` square systems. Measured counts:

| dimension | rows `m` | `C(m, n)` bases |
|---|---|---|
| 2 | 32 | 496 |
| 3 | 30 | 4,060 |
| 3 | 60 | 34,220 |
| 4 | 40 | 91,390 |
| 5 | 40 | 658,008 |

The default `max_bases = 200000` admits 4-D with 40 rows and **refuses** 5-D
with 40 rows, raising `VertexEnumerationError` rather than hanging. A 4-D
example with 11 rows enumerated 25 vertices in 0.02 s. When the budget is
exceeded during a recursion, `track_geometry` records `None` for that
iteration instead of raising, because growth that outruns the enumerator is
itself a measurement.

## 6. Where the iteration stalls, and non-convergence

### 6A. `slow_pair`: `A = [[0.999, 0.05], [0, 0.999]]`, `|w| <= 0.002`

Repeated pole at 0.999. Measured over 80 iterations: shrink margin min
**2.730605e-02**, max **5.100000e-02**, per-iteration ratio between
**0.976652761** and **1.000000000**. The margin does not decay, so the
recursion **cannot** terminate.

| cap | termination | facets | area | invariant | worst facet margin |
|---|---|---|---|---|---|
| 10 | iteration_cap | 6 | 3.440009340 | False | +5.100000e-02 |
| 20 | iteration_cap | 6 | 2.879521799 | False | +5.100000e-02 |
| 40 | iteration_cap | 4 | 1.758263874 | False | +5.003431e-02 |
| 80 | iteration_cap | 4 | 0.591043051 | False | +2.703876e-02 |

The cap-80 result is a subset of the cap-10 result (checked exactly), which is
the nesting the recursion guarantees. **Every row of this table is an outer
bound on `S_inf`, and none of them is `S_inf`.** The package reports this as
`termination = "iteration_cap"` and the report text says `DID NOT CONVERGE ...
the returned polytope is an OUTER bound on S_inf, not S_inf`. The CLI exits
with status 2.

### 6B. Eigenvalues exactly on the unit circle: `A = R(0.3 rad)`

| cap | facets | area | invariant | worst facet margin |
|---|---|---|---|---|
| 10 | 20 | 2.545854330 | False | +6.403736e-02 |
| 30 | 20 | 1.301222810 | False | +6.108756e-02 |
| 60 | 16 | 0.204587260 | False | +4.511011e-02 |

Nothing contracts, so the recursion shaves a sliver forever. **No cap value
fixes this**; the right answer is that `S_inf` for this system and this `W` is
degenerate or empty and the recursion is the wrong tool for finding out.

## 7. The identity that does not hold, measured

`validate_support_algebra.py` checks 6, 7, 8 and 11; `examples/set_algebra_gap.py`
draws it.

**The Pontryagin difference is not an inverse of the Minkowski sum.** The
identity that holds is a containment, and the equality holds in the other
order:

| Identity | Status | Measured |
|---|---|---|
| `(P (-) Q) (+) Q  subset  P` | **holds** | 120 random pairs, worst escape margin **+2.220446e-16** (floating point, not algebra) |
| `(P (-) Q) (+) Q  =  P` | **FALSE in general** | area deficit over 120 random pairs: median **0.0000 %**, max **0.2750 %**, min -0.0000 % |
| `(P (+) Q) (-) Q  =  P` | **holds** for compact convex sets | 120 random pairs, worst margin either way **+2.442491e-15** |

The median deficit being zero is the reason people believe the false identity:
most randomly generated polytopes here are boxes or box-dominated, and for
aligned boxes the equality does hold.

Deterministic witness, hand-computable. `P` is the triangle
`{x1 >= 0, x2 >= 0, x1 + x2 <= 1}` and `Q` the box of half-width 0.1:

| Quantity | Hand value | Measured |
|---|---|---|
| area(`P`) | 0.5 | 0.500000000 |
| area(`P (-) Q`), a right triangle with legs `0.8 - 0.1 - 0.1 = 0.6` | `0.5 * 0.6^2 = 0.18` | 0.180000000 |
| area(`(P (-) Q) (+) Q`), by `area(A) + area(B) + sum_i abs(e_i) h_B(n_i)` = `0.18 + 0.04 + 0.06 + 0.06 + 0.12` | 0.46 | 0.460000000 |
| **deficit** | **0.04** | **0.040000000** |
| deficit as a fraction of `P` | 8 % | **8.000000 %** |
| area(`(P (+) Q) (-) Q`) | 0.5 | 0.500000000 |
| `P subset (P (-) Q) (+) Q` | False | **False** |
| `(P (-) Q) (+) Q subset P` | True | **True** |

**When does the assumed identity happen to hold?** The textbook condition is
that `Q` be a *summand* of `P`, i.e. `P = R (+) Q` for some convex `R`
(Schneider 1993, summands and decomposition). For a box `Q` in the plane that
shows up concretely as: the opening returns `P` when the box's own facet
normals are already facet normals of `P`. Measured:

| `P` | area | opened area | deficit |
|---|---|---|---|
| triangle (`e1`, `e2` are **not** facet normals) | 0.500000000 | 0.460000000 | **+4.000e-02 (+8.0000 %)** |
| pentagon (`e1`, `e2` **are** facet normals) | 0.460000000 | 0.460000000 | -5.551e-17 (-0.0000 %) |

The opened triangle *is* the pentagon: the pentagon is a fixed point of the
opening and the triangle is not. The same effect appears on the shipped `S_inf`, whose
facet normals include `+/-e1` and `+/-e2`. Scaling `W` up (check 11 of
`validate_support_algebra.py`):

| `W` scaled by | facets of `S_inf` after erosion | area after opening | deficit |
|---|---|---|---|
| 1 (as shipped) | 16 of 16 | 0.538347285 | -0.0000 % |
| 20 | 16 of 16 | 0.538347285 | -0.0000 % |
| 40 | **14** of 16 | 0.537410343 | **+0.1740 %** |
| 60 | **8** of 16 | 0.485505791 | **+9.8155 %** |

The deficit appears exactly when facets start disappearing in the erosion.

### 7B. A maximal set has no margin left, so the strict invariance check can read `False`

`verify_robust_invariance` defaults to zero tolerance. On a correct **maximal**
set the true margin is exactly zero, so the computed one lands on either side
of zero at the 1e-16 level:

| system | worst facet margin | strict (`tol = 0`) | `tol = 1e-9` |
|---|---|---|---|
| `attitude_loop` | 0.000000e+00 | **True** | True |
| `damped_rotation_2d` | **+2.220446e-16** | **False** | True |
| `damped_rotation_3d` | **+1.110223e-16** | **False** | True |

This is reported rather than tolerance-tuned away: the default stays strict so
the margin is visible, and the `tol` argument exists for the engineering
question. Two of the three shipped converging systems fail the strict check by
one or two units in the last place.

## 8. Errors made during this build, and corrected

1. **An interval Pontryagin difference computed wrongly in a test.** The first
   version of `tests/test_edge_cases.py::TestOneDimensional::test_interval_algebra`
   asserted `[-1, 2] (-) [-0.25, 0.5] = [-0.5, 1.75]`. That is wrong. The
   correct erosion is
   `{x : x - 0.25 >= -1 and x + 0.5 <= 2} = [-0.75, 1.5]`, which is what the
   code returned and what made the test fail. The test comment now carries the
   corrected derivation. **The implementation was right and the hand
   arithmetic was wrong**; recording it here rather than silently editing the
   number is the point.
2. **A `scipy.spatial.qhull` import that raised a `DeprecationWarning`.** The
   first version imported `QhullError` from `scipy.spatial.qhull`, which SciPy
   1.18 deprecates. Moved to `scipy.spatial`. Caught because
   `pyproject.toml` sets `filterwarnings = ["error::DeprecationWarning"]`.
3. **A determinant-based rank test in vertex enumeration** produced
   `RuntimeWarning: divide by zero encountered in det` on badly scaled rows,
   which is exactly the case that arises after several recursion iterations.
   Replaced with a singular-value test at `RANK_REL_TOL = 1e-10`. This was not
   only quieter but faster: the suite went from 83 s to 60 s.
4. **Blanchini 1999 page range.** The sibling product P051 cites
   pp. 1747-1767. The author's own publication list gives **1747-1768**, and
   the title is "Set invariance in control — a survey". This package uses the
   verified form. Every other citation in section 9 was checked the same way
   during this build.

## 9. References, each verified during this build

Book section and chapter numbers are **deliberately not quoted**: no copy of
any of the books was available in the build container to check them against.
Journal volume, issue, year and page ranges below were each checked against a
publisher, repository or author listing during this build.

- Gilbert, E. G. and Tan, K. T., "Linear systems with state and control
  constraints: the theory and application of maximal output admissible sets",
  *IEEE Transactions on Automatic Control* **36**(9), 1991, pp. 1008–1020 —
  the one-step-set recursion and its finite-determination theory.
- Kolmanovsky, I. and Gilbert, E. G., "Theory and computation of disturbance
  invariant sets for discrete-time linear systems", *Mathematical Problems in
  Engineering* **4**(4), 1998, pp. 317–367 — the Pontryagin difference in
  halfspace form, robust invariance, and the identity `(P (+) Q) (-) Q = P`.
- Blanchini, F., "Set invariance in control — a survey", *Automatica*
  **35**(11), 1999, pp. 1747–1768 — the survey; the distinction between
  maximal and minimal invariant sets.
- Raković, S. V., Kerrigan, E. C., Kouramas, K. I. and Mayne, D. Q.,
  "Invariant approximations of the minimal robust positively invariant set",
  *IEEE Transactions on Automatic Control* **50**(3), 2005, pp. 406–410 — the
  **minimal** robust positively invariant set, named here because it is a
  different object from the maximal set this package computes.
- Borrelli, F., Bemporad, A. and Morari, M., *Predictive Control for Linear
  and Hybrid Systems*, Cambridge University Press, 2017, ISBN 9781107016880 —
  invariant-set algorithms and the LP-based redundancy test.
- Rockafellar, R. T., *Convex Analysis*, Princeton University Press, 1970 —
  the support function and its properties.
- Schneider, R., *Convex Bodies: the Brunn-Minkowski Theory*, Cambridge
  University Press, 1993 — Minkowski additivity of the support function, and
  summands.
- Franklin, G. F., Powell, J. D. and Workman, M. L., *Digital Control of
  Dynamic Systems*, 3rd edition, Addison-Wesley, 1997 — the zero-order-hold
  discretisation of the double integrator used in section 3.
- Herceg, M., Kvasnica, M., Jones, C. N. and Morari, M., "Multi-Parametric
  Toolbox 3.0", *Proceedings of the European Control Conference*, Zürich,
  2013, pp. 502–510 — MPT3, named in the alternatives table.
- Heirung, T. A. N., `pytope` 0.0.4, https://github.com/heirung/pytope —
  named in the alternatives table; version, dependencies and self-description
  read from its PyPI metadata during this build.
- `polytope` 0.2.5, TuLiP control project,
  https://github.com/tulip-control/polytope — named in the alternatives table;
  version and dependencies read from its PyPI metadata during this build.

## 10. What this validation does **not** establish

1. **No physical validation.** There is no flight data, no hardware, and no
   comparison against a measured spacecraft anywhere in this repository. The
   attitude loop of section 3 is an illustrative model with chosen numbers.
   Level 2 is the honest label.
2. **No proof about any plant you did not supply.** The package computes a
   robust invariant set for the matrix and disturbance bound it is given. It
   proves nothing about a plant it was not given, and the disturbance bound is
   something you declared and it cannot check.
3. **No guarantee of finite termination.** Sections 6A and 6B are systems on
   which the recursion does not terminate. The package reports that; it does
   not fix it.
4. **No large-dimension evidence.** Everything validated here is dimension 1
   to 4. The brute-force vertex enumeration is `C(m, n)` and the redundancy
   removal is one LP per row per iteration; neither has been measured beyond
   the sizes in section 5C.
5. **No comparison against `pytope`, `polytope` or MPT3 outputs.** None of
   them is installed in this container and installing them was out of scope,
   so the alternatives table describes what they ship (read from their
   published metadata) and does not claim a numerical agreement that was never
   measured.

## 11. Reproducing every number

From a cold clone, no install step needed (`pythonpath = ["src"]` is set in
`pyproject.toml`):

```bash
python -m pytest tests/ -q --junit-xml=junit.xml        # 200 tests, ~60 s
ruff check src/ tests/ examples/ validation/
python -m invariantset --help

python validation/validate_environment.py               # <1 s
python validation/validate_support_algebra.py           # set algebra, ~40 s
python validation/validate_known_answers.py             # hand answers, ~3 s
python validation/validate_growth_and_stall.py          # growth and stalling, ~20 s
python validation/validate_tolerance_sensitivity.py     # the tolerance case, ~90 s
python validation/validate_cli.py                       # the CLI in subprocesses, ~25 s
python validation/worked_example.py                     # the README worked example

MPLBACKEND=Agg python examples/maximal_invariant_set_2d.py
MPLBACKEND=Agg python examples/set_algebra_gap.py
MPLBACKEND=Agg python examples/tolerance_sensitivity.py
MPLBACKEND=Agg python examples/iteration_growth.py
MPLBACKEND=Agg python examples/non_convergence.py
```

Every script is deterministic. The only seeded one is
`validate_support_algebra.py` (seed 59001); everything else involves no
randomness at all, because the recursion, the LP solver and the hull are
deterministic given the same inputs. The committed `*_output.txt` files are
the raw stdout of exactly these commands, and the committed PNGs are what
exactly these example scripts produce.
