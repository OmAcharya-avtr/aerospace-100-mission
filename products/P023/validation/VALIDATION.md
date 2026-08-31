# Validation evidence — AllocLab 0.1.0

Validation Level 2. Every number below was produced by running the script named
beside it, in this repository, on 2 x86-64 cores under Python 3.11.15,
numpy 2.4.4, scipy 1.17.1, scikit-learn 1.8.0, pulp 3.3.2. Raw stdout is saved
next to each script.

| Script | Raw output | What it establishes |
|---|---|---|
| `validate_exact_allocation.py` | `exact_allocation_output.txt` | Allocation reproduces a commanded torque inside the attainable set |
| `validate_ams.py` | `ams_output.txt` | The attainable moment set matches its closed forms |
| `validate_bounds.py` | `bounds_output.txt` | Every allocation respects the actuator bounds |
| `validate_failure.py` | `failure_output.txt` | Failed-effector reallocation meets what it can and reports what it cannot |
| `validate_ml_vs_qp.py` | `ml_vs_qp_output.txt` | The learned allocator against the exact QP |

Reproduce all five:

```bash
python validation/validate_exact_allocation.py    # ~40 s
python validation/validate_ams.py                 # ~1 s
python validation/validate_bounds.py              # ~110 s
python validation/validate_failure.py             # ~130 s
python validation/validate_ml_vs_qp.py            # ~180 s (trains the model)
```

Three effector configurations recur throughout:

| Name | m | Command box | Notes |
|---|---:|---|---|
| Orthogonal triad | 3 | ±1 (dimensionless) | `B = I`, so the AMS is the cube `[-1,1]³` |
| Pyramid wheel array | 4 | ±0.1 N·m | Spin axes at 54.7356° from +z, azimuths 0/90/180/270 |
| Reference thruster cluster | 8 | [0, 1] N | Three antiparallel couples plus a skew corner couple, arm 0.5 m |

---

## 1. Allocation reproduces the commanded torque inside the attainable set

`validate_exact_allocation.py`, seed 20260831, 400 commands per configuration
placed at 80% of the attainable-moment-set boundary radius along uniformly
random directions.

| Configuration | Method | Max residual [N·m] | Max bound violation | Infeasible of 400 | Mean solve [µs] | Outcome |
|---|---|---:|---:|---:|---:|---|
| Orthogonal triad | pinv | 0.000000e+00 | 0.000000e+00 | 0 | 53.3 | PASS |
| Orthogonal triad | wpinv | 0.000000e+00 | 0.000000e+00 | 0 | 117.9 | PASS |
| Orthogonal triad | rpi | 0.000000e+00 | 0.000000e+00 | 0 | 159.2 | PASS |
| Orthogonal triad | lp | 0.000000e+00 | 0.000000e+00 | 0 | 2036.8 | PASS |
| Orthogonal triad | qp | 1.363919e-12 | 0.000000e+00 | 0 | 122.1 | PASS |
| Pyramid wheels | pinv | 1.144854e-16 | **1.914405e-02** | **100** | 65.7 | bounds not enforced |
| Pyramid wheels | wpinv | 8.946869e-17 | **1.914405e-02** | **100** | 94.6 | bounds not enforced |
| Pyramid wheels | rpi | 8.946869e-17 | 0.000000e+00 | 0 | 167.8 | PASS |
| Pyramid wheels | lp | 6.245005e-17 | 0.000000e+00 | 0 | 2999.7 | PASS |
| Pyramid wheels | qp | 1.765844e-13 | 0.000000e+00 | 0 | 225.7 | PASS |
| Thruster cluster | pinv | 6.479604e-16 | **5.614578e-01** | **400** | 57.3 | bounds not enforced |
| Thruster cluster | wpinv | 7.153458e-16 | **6.145776e-02** | **94** | 158.9 | bounds not enforced |
| Thruster cluster | rpi | 7.285508e-16 | 0.000000e+00 | 0 | 254.0 | PASS |
| Thruster cluster | lp | 8.777084e-17 | 0.000000e+00 | 0 | 3578.8 | PASS |
| Thruster cluster | qp | 2.105728e-12 | 0.000000e+00 | 0 | 211.8 | PASS |

Tolerance: 1e-8 N·m on the residual, 1e-9 on the bound violation.

The pseudo-inverse rows are the intended reading of this table, not a defect:
`pinv` reproduces the commanded torque to 6.5e-16 N·m on the thruster cluster
and does so with a command every single one of whose 400 answers violates the
0–1 N thrust box, by up to **0.561 N** on a 1 N thruster. On a one-sided
effector set an unconstrained generalized inverse is never usable on its own.

### 1b. QP residual versus `gamma`

The QP penalises torque error rather than constraining it, so its residual on
an attainable command falls as `1/gamma`. Measured on the pyramid array, 200
interior commands:

| gamma | Max residual [N·m] | Max bound violation |
|---:|---:|---:|
| 1e4 | 1.744154e-05 | 0.000000e+00 |
| 1e6 | 1.744672e-07 | 0.000000e+00 |
| 1e8 | 1.744677e-09 | 0.000000e+00 |
| 1e10 | 1.744677e-11 | 0.000000e+00 |
| **1e12 (default)** | **1.744348e-13** | 0.000000e+00 |
| 1e14 | 1.725001e-15 | 0.000000e+00 |
| 1e16 | 9.557745e-17 | 0.000000e+00 |

Exactly `1/gamma` over twelve decades, with no conditioning breakdown. The
default 1e12 leaves about five orders of margin against the 1e-8 N·m tolerance.

### 1c. LP solver cross-check, HiGHS versus CBC

The `min_error` linear programme built twice, once through
`scipy.optimize.linprog` (HiGHS dual simplex) and once through PuLP/CBC, 40
interior commands each:

| Configuration | Max \|achieved torque difference\| [N·m] | Max HiGHS residual | Max CBC residual | HiGHS / CBC mean solve |
|---|---:|---:|---:|---|
| Orthogonal triad | 5.720588e-09 | 0.000000e+00 | 5.720588e-09 | 2.623 / 5.332 ms |
| Pyramid wheels | 6.603284e-10 | 3.704356e-17 | 6.603284e-10 | 2.060 / 4.929 ms |
| Thruster cluster | 5.104357e-09 | 1.000742e-16 | 5.104357e-09 | 2.195 / 5.452 ms |

Two independent simplex implementations agree to 6e-9 N·m. The residual gap is
CBC's looser default optimality tolerance, not a formulation difference.

### 1d. Commands outside the attainable set

200 commands at 150% of the boundary on the thruster cluster:

| Method | Min residual [N·m] | Max bound violation [N] | Reported infeasible |
|---|---:|---:|---:|
| pinv | 2.081668e-17 | 1.059814e+00 | 200 / 200 |
| wpinv | 4.857226e-17 | 5.598137e-01 | 200 / 200 |
| rpi | 3.354102e-01 | 0.000000e+00 | 200 / 200 |
| lp | 3.361275e-01 | 0.000000e+00 | 200 / 200 |
| qp | 3.354102e-01 | 0.000000e+00 | 200 / 200 |

All five report `feasible=False` on all 200, but for different reasons: the
generalized inverses because their commands are unexecutable, the bounded
methods because the torque cannot be delivered. The LP's 2-norm residual is
slightly *worse* than the QP's (3.361e-01 versus 3.354e-01) because it
minimises the 1-norm error, which is the correct behaviour for its objective.

---

## 2. The attainable moment set against its closed forms

`validate_ams.py`.

### 2a. Cube known answer

Orthogonal triad, bounds ±1: the AMS is exactly `[-1,1]³`.

| Quantity | Computed | Closed form | Error |
|---|---:|---:|---:|
| Vertices | 8 | 8 | exact |
| Max \|vertex coordinate error\| | — | — | 0.000000e+00 |
| Volume [(N·m)³] | 8 | 8 | 0.000000e+00 |
| Surface area [(N·m)²] | 24 | 24 | 0.000000e+00 |
| Boundary scale along +x | 1 | 1 | 0.000000e+00 |
| Boundary scale along (1,1,1) | 1.73205080756888 | √3 | 2.220446e-16 |

A second box case, `B = I` with bounds ±(1, 1.5, 2.5): volume 30 (error 0),
area 62 (error 0), 8 vertices.

### 2b. Hull volume against the zonotope volume formula

`V = Σ_{i<j<k} |det(b_i, b_j, b_k)| L_i L_j L_k`, computed independently of the
convex hull.

| Configuration | Hull volume | Closed form | Relative error |
|---|---:|---:|---:|
| Orthogonal triad, bound 1.0 | 8 | 8 | 0.00e+00 |
| Orthogonal triad, bound 2.0 | 64 | 64 | 0.00e+00 |
| Pyramid wheels, 4 × 0.1 N·m | 0.0246336114854 | 0.0246336114854 | 4.23e-16 |
| Pyramid wheels, 5 × 0.05 N·m, 35° | 0.00463656224447 | 0.00463656224447 | 0.00e+00 |
| Thruster cluster, 8 × 1 N | 3.82842712475 | 3.82842712475 | 5.80e-16 |
| Thruster cluster, t1 failed off | 2.26776695297 | 2.26776695297 | 5.87e-16 |
| Thruster cluster, t1 stuck at 1 N | 2.26776695297 | 2.26776695297 | 3.92e-16 |

Tolerance: 1e-10 relative. Worst measured 5.87e-16. PASS.

### 2c. Vertex count against `g(g-1) + 2`

| Configuration | Computed | Formula | Match |
|---|---:|---:|---|
| Orthogonal triad, bound 1.0 | 8 | 8 | yes |
| Orthogonal triad, bound 2.0 | 8 | 8 | yes |
| Pyramid wheels, 4 × 0.1 N·m | 14 | 14 | yes |
| Pyramid wheels, 5 × 0.05 N·m, 35° | 22 | 22 | yes |
| Thruster cluster, 8 × 1 N | 14 | 14 | yes |
| Thruster cluster, t1 failed off | 14 | 14 | yes |
| Thruster cluster, t1 stuck at 1 N | 14 | 14 | yes |

`g` counts distinct generator *lines*: the eight-thruster cluster has only four,
because its three antiparallel couples and its skew couple each collapse to one
line, giving `4·3 + 2 = 14`. The formula is an upper bound only when three
generator lines are coplanar; none of the seven configurations above is in that
case, so all seven match exactly.

### 2d. Pairwise facet construction against brute-force box enumeration

| Configuration | Vertices (pairwise) | Vertices (brute) | Volume rel. err. | t pairwise [ms] | t brute [ms] |
|---|---:|---:|---:|---:|---:|
| Orthogonal triad, 1.0 | 8 | 8 | 0.00e+00 | 1.241 | 0.294 |
| Orthogonal triad, 2.0 | 8 | 8 | 0.00e+00 | 1.364 | 0.322 |
| Pyramid wheels 4 | 14 | 14 | 2.82e-16 | 1.883 | 0.356 |
| Pyramid wheels 5 | 22 | 22 | 5.61e-16 | 4.101 | 0.428 |
| Thruster cluster | 14 | 14 | 4.64e-16 | 3.002 | 0.686 |
| Cluster, t1 off | 14 | 14 | 7.83e-16 | 3.597 | 0.692 |
| Cluster, t1 stuck | 14 | 14 | 7.83e-16 | 3.518 | 0.814 |

Brute force is faster at these sizes; it is `O(2^m)` and stops being usable
above `m ≈ 18`, where the pairwise `O(m²)` construction is the only option.

**A defect this cross-check found.** The first implementation of the pairwise
construction enumerated only the four `(u_i, u_j)` corners of each facet, which
is correct only when exactly two generators lie in the facet plane. Hypothesis,
comparing the two constructions on random configurations, produced a
configuration with three coplanar generators where that lost a vertex 0.5 N·m
away and under-reported the volume by 3.6%. Facets are now enumerated exactly
as 2-D zonotopes by an angular walk; the case is pinned in
`tests/test_ams.py::test_coplanar_generators_do_not_lose_facet_vertices`.

### 2e. Failure shrinks the AMS

Reference thruster cluster, nominal volume 3.82842712475 (N·m)³:

| Failed effectors | Volume [(N·m)³] | Ratio | Remaining rank |
|---|---:|---:|---:|
| none | 3.828427125 | 1.0000 | 3 |
| [t1] | 2.267766953 | 0.5923 | 3 |
| [t1, t2] | 0.7071067812 | 0.1847 | 3 |
| [t1, t3] | 1.310660172 | 0.3423 | 3 |
| [t7] | 2.414213562 | 0.6306 | 3 |
| [t7, t8] | 1.000000000 | 0.2612 | 3 |
| [t1, t2, t3, t4] | 0 | 0.0000 | 2 |
| [t5, t6] | 1.414213562 | 0.3694 | 3 |

Losing the ±x and ±y couples together collapses the set to a plane: rank 2,
volume 0, and every command with a non-zero x or y component becomes
unattainable.

---

## 3. Every allocation respects the actuator bounds

`validate_bounds.py`, seed 4242. 300 random full-rank effector configurations
(`m` between 3 and 8, random Gaussian columns, random possibly-one-sided
bounds) × 8 random commands each = 1505 evaluated commands, 642 of them inside
the attainable moment set as decided by the convex hull.

| Method | Max bound violation | Commands violating | % violating | Max residual on attainable commands | Attainable commands missed |
|---|---:|---:|---:|---:|---:|
| pinv | 2.029687e+00 | 1444 | 95.95 | 2.113598e-14 | 0 |
| wpinv | 1.879693e+00 | 1334 | 88.64 | 7.016278e-14 | 0 |
| rpi | 0.000000e+00 | 0 | 0.00 | 7.656943e-01 | **23** |
| lp | 0.000000e+00 | 0 | 0.00 | 1.609823e-15 | 0 |
| qp | 2.220446e-16 | 0 | 0.00 | 1.473295e-11 | 0 |

PASS criterion, max bound violation ≤ 1e-9 for the bounds-aware methods:

- `lp`: **PASS**, 0.000000e+00
- `qp`: **PASS**, 2.220446e-16
- `rpi`: **PASS**, 0.000000e+00

**The redistributed pseudo-inverse missed 23 of 642 attainable commands (3.6%)**,
with a worst residual of 0.766 while staying inside the box. That is not a
defect in this implementation: the redistributed pseudo-inverse is a heuristic
and both Bodson (2002) §V.A and Härkegård (2002) §2.2.1 state that it is not
guaranteed to find a feasible command when one exists. It is reported here
because it is the reason to prefer the LP or QP. Note that on the *structured*
configurations of §4 it missed none, so the failure rate depends strongly on
geometry.

The same claim is also a Hypothesis property over random configurations and
random commands. Executed as part of this script:

```
tests/test_properties.py::test_qp_allocation_always_respects_actuator_bounds
tests/test_properties.py::test_lp_allocation_always_respects_actuator_bounds
tests/test_properties.py::test_redistributed_pseudo_inverse_always_respects_actuator_bounds
tests/test_properties.py::test_allocation_reproduces_any_attainable_command_exactly
4 passed in 3.61s
```

pytest exit code 0. PASS.

---

## 4. Failed-effector reallocation

`validate_failure.py`, seed 90210. For each failure case: 60 random directions ×
6 magnitudes (0.25, 0.6, 0.9, 1.1, 1.5, 2.5 of the *degraded* attainable
boundary) = 360 commands. Feasibility is decided independently of the allocator
by the exact LP certificate `is_attainable`.

Thruster cluster, `method="qp"`, all 8 single failures and 8 double failures;
pyramid array, all 4 single failures:

| Aggregate check | Count | Required | Outcome |
|---|---:|---:|---|
| Attainable commands the QP failed to meet | 0 | 0 | **PASS** |
| Unattainable commands not reported infeasible | 0 | 0 | **PASS** |
| Feasibility verdicts disagreeing with the LP certificate | 0 | 0 | **PASS** |

Worst-case numbers within that sweep: max residual on an attainable command
3.941e-12 N·m (thruster cluster, t4 failed), 2.615e-13 N·m (pyramid array);
max bound violation 7.112e-17.

Six of the ten pyramid double-failure cases leave rank 2 — two wheels of a
four-wheel pyramid is exactly the point at which an axis is lost — and are
reported as degenerate rather than swept.

The same sweep with `method="rpi"` on the eight single-failure cases missed 0
of 1440 attainable commands, with a worst residual of 2.038e-15 N·m. The
heuristic handles this particular geometry; §3 shows it does not handle all of
them.

### Worked cases

**Command still attainable after failure** — +0.05 N·m about x with t1 failed
off: `attainable=True`, `status="saturated"`, residual **9.798418e-13 N·m**,
bound violation 0, AMS volume ratio 0.592350, failure margin 5.0000 (the
command could grow 5× before leaving the degraded set).

**Command no longer attainable** — +0.6 N·m about x with t1 failed off:
`attainable=False`, `status="infeasible"`, achieved torque
`[0.32, 6.4e-13, -0.14]`, residual **3.130495e-01 N·m**, bound violation 0,
failure margin 0.4167. The message carries the LP certificate value
(3.5e-01 N·m optimal 1-norm error) alongside the QP's 2-norm shortfall.

**Thruster stuck open** — t1 pinned at 1 N, zero torque commanded: the
remaining seven cancel the 0.5 N·m bias exactly, achieved torque
`[7.8e-13, -1.1e-13, 2.2e-13]`, residual **8.162595e-13 N·m**.

**Rank collapse** — t5, t6, t7, t8 all lost, 0.2 N·m about z commanded:
remaining rank 2, `attainable=False`, residual **2.000000e-01 N·m** (the whole
command), AMS volume ratio 0.

**`require_feasible=True`** raises `InfeasibleAllocationError` naming the
effectors, the LP certificate, the shortfall, the bound violation and the
remaining rank, instead of returning a clipped command.

---

## 5. The learned allocator against the exact QP

`validate_ml_vs_qp.py`. The classical allocators above were implemented and
validated first; the model is trained to imitate `qp_allocate`.

**Setup.** Reference thruster cluster (m = 8, u ∈ [0, 1] N, arm 0.5 m), AMS
boundary radius 1.707107 N·m. 4000 training samples (seed 1234) and 2000
held-out test samples (seed 5678), each a (torque, health-mask) pair labelled
with the exact QP solution at `u_pref = 0` (minimum total thrust), gamma 1e12.
0 to 2 effectors failed off per sample, 50% of samples with at least one
failure. Model: 5 × `MLPRegressor(64, 48)`, `max_iter=300`, `alpha=1e-4`,
`early_stopping=True`, `random_state=0..4`.

**Compute budget, 2 CPU cores, no GPU, no PyTorch:**

| Step | Wall clock |
|---|---:|
| Training-set generation (4000 QP solves) | 4.10 s |
| Test-set generation (2000 QP solves) | 1.98 s |
| Model fit (5 MLPs) | **115.22 s** |
| Total | 121.29 s |

Under the 180 s budget. Two of the five members emit a scikit-learn
`ConvergenceWarning` at `max_iter=300`; they are present in the raw output.
Attainable fraction: 0.9550 train, 0.9575 test.

### 5a. Allocation error and constraint satisfaction, all 2000 test samples

| Allocator | Mean [N·m] | Median | p95 | Max | % inside actuator bounds |
|---|---:|---:|---:|---:|---:|
| Exact QP | 7.183888e-04 | 1.400865e-12 | 6.743076e-12 | 5.413941e-02 | **100.00** |
| Learned (raw) | **1.043167e-02** | 7.605578e-03 | 2.704358e-02 | 8.937423e-02 | **4.85** |
| Learned (clipped into the box) | 1.538573e-02 | 8.987115e-03 | 4.674646e-02 | 3.361737e-01 | 100.00 |

Restricted to the 1915 samples the QP meets exactly:

| Allocator | Mean [N·m] | Median | p95 | Max | % inside bounds |
|---|---:|---:|---:|---:|---:|
| Exact QP | 1.618039e-12 | 1.314036e-12 | 4.801879e-12 | 8.327863e-12 | 100.00 |
| Learned (raw) | 1.011498e-02 | 7.389733e-03 | 2.592084e-02 | 8.937423e-02 | 5.01 |
| Learned (clipped) | 1.392367e-02 | 8.627559e-03 | 4.215541e-02 | 3.208724e-01 | 100.00 |

The QP's non-zero *mean* over all 2000 samples comes entirely from the 85
samples whose command is outside the attainable set, where no allocation can
be exact.

### 5b. Bound violations of the learned output — the finding

| Quantity | Learned (raw) | Exact QP |
|---|---:|---:|
| Samples violating at least one bound | **1903 / 2000 (95.15%)** | 0 / 2000 |
| Max violation | **3.294447e-01 N** (32.9% of the 1 N thrust limit) | 0 |
| Mean violation over violating samples | 1.762133e-02 N | — |

Per-effector maximum violation [N]:
`[0.0796, 0.1971, 0.1983, 0.1600, 0.1765, 0.1592, 0.2273, 0.3294]` — every one
of the eight thrusters is commanded out of range somewhere in the test set.

This is the honest headline: **the learned allocator violates actuator bounds
where the QP does not, on 95% of held-out samples, by up to a third of a
thruster's full range.** Nothing in a plain regression enforces the box, and
the residual bound violation does not shrink to zero with more training — it is
structural. Clipping restores 100% bound compliance at the cost of a 47% larger
mean allocation error (1.0432e-02 → 1.5386e-02 N·m) and a 3.8× larger worst
case (8.94e-02 → 3.36e-01 N·m).

### 5c. Runtime

| Path | Per allocation | Speed-up vs QP |
|---|---:|---:|
| Exact QP, single solve | 696.3 µs | 1.00× |
| Learned, single solve (batch of 1) | 1417.4 µs | **0.49×** |
| Learned, 2000 batched at once | 64.0 µs | 10.88× |

The expected speed-for-exactness trade only half materialised. Called one
command at a time — which is how a control loop calls an allocator — the
learned model is **twice as slow** as the exact QP, because five scikit-learn
`predict` calls plus feature scaling cost more than one bounded least-squares
solve on a 3×8 problem. The 10.9× advantage exists only when 2000 commands are
batched, which a real-time loop cannot do.

### 5d. Error by number of failed effectors

| Failures | Count | QP mean error [N·m] | Learned mean error | Ratio | Learned % in bounds |
|---:|---:|---:|---:|---:|---:|
| 0 | 964 | 9.494661e-04 | 7.273390e-03 | 7.7× | 3.22 |
| 1 | 495 | 5.312145e-04 | 1.256916e-02 | 23.7× | 8.28 |
| 2 | 541 | 4.778948e-04 | 1.410363e-02 | 29.5× | 4.62 |

The learned model degrades monotonically with the number of failures — exactly
the regime the health mask was added to cover, and exactly where it is least
trustworthy.

### 5e. The confidence output

Ensemble spread across the five members, mapped to a `[0, 1]` confidence.

| Quantity | Value |
|---|---:|
| Pearson r(confidence, allocation error) | **−0.6245** |
| Pearson r(ensemble spread, allocation error) | +0.7045 |
| Mean ensemble spread / RMS command error | 0.6333 |

| Confidence decile | Count | Mean allocation error [N·m] | % in bounds |
|---:|---:|---:|---:|
| 1 (lowest) | 200 | 2.563590e-02 | 3.50 |
| 2 | 200 | 1.581155e-02 | 5.50 |
| 3 | 200 | 1.300535e-02 | 2.50 |
| 4 | 200 | 1.078942e-02 | 4.50 |
| 5 | 200 | 8.761695e-03 | 5.50 |
| 6 | 200 | 7.678611e-03 | 2.00 |
| 7 | 200 | 6.631749e-03 | 8.50 |
| 8 | 200 | 6.000610e-03 | 4.00 |
| 9 | 200 | 5.343445e-03 | 4.50 |
| 10 (highest) | 200 | 4.658391e-03 | 8.00 |

The confidence is monotone in error across all ten deciles and spans a 5.5×
error range, so it does rank predictions usefully. It is **not calibrated**: the
mean ensemble spread is 0.633 of the RMS command error, so the members agree
with each other more than they agree with the truth, and it must not be used as
a covariance. It is also nearly useless as a bound-violation predictor: the
top confidence decile still violates the box on 92% of its samples.

### 5f. Verdict

| Axis | Winner | Margin |
|---|---|---|
| Allocation error | **exact QP** | 14.5× on the mean |
| Constraint satisfaction | **exact QP** | 100.00% vs 4.85% |
| Runtime, one command at a time | **exact QP** | 2.0× |
| Runtime, 2000 commands batched | learned | 10.9× |

The learned allocator loses three of four axes. On this problem — a 3×8
bounded least-squares solve that a direct active-set factorisation finishes in
under a millisecond — there is no operating point at which it is the right
choice for a control loop.

---

## Not validated

- Any comparison against hardware, a test stand, flight telemetry, or a
  higher-fidelity simulator. Every number here is self-consistency against
  closed forms and against independently implemented solvers.
- Actuator dynamics of any kind: rate limits, minimum impulse bit, pulse-width
  modulation, thruster rise and fall, wheel friction or zero-crossing,
  motor speed-torque roll-off.
- Reaction-wheel momentum saturation and the gyroscopic `ω × h` term. The wheel
  model is torque-limited only, so an allocation this package calls feasible
  may be unreachable once wheel speeds are accounted for.
- Closed-loop behaviour. Nothing here has been placed in an attitude control
  loop, so nothing is known about stability, chatter at the attainable
  boundary, or the effect of allocator switching.
- Numerical behaviour for `m > 20` or for effectiveness matrices whose
  condition number exceeds about 1e3; the property tests filter both out.
- Formal coverage or calibration of the learned model's confidence output
  beyond the decile ranking in §5e.
