# Validation evidence — rareverify 0.1.0

**Validation level 2 (research grade).** Status: `TESTING`.

Every number in this file and in `README.md` was produced by a script in this
directory, executed on 2026-10-08 in the build container, with its raw output
committed beside it as `validation/<script>.txt`. Nothing is quoted from a
textbook, estimated, or carried over from another product.

## Environment the numbers were measured in

| Item | Value |
|---|---|
| Python | 3.13.16 |
| Platform | Linux-6.18.44-fc-v80-x86_64-with-glibc2.39 |
| `os.cpu_count()` | 2 |
| Schedulable cores | 2 |
| `MemTotal` | 7.84 GiB |
| numpy / scipy / scikit-learn | 2.5.3 / 1.18.1 / 1.9.1 |

Every estimator in the package is single-threaded numpy. Wall-clock figures
move 10–20 % between runs on this container and are reported for budgeting
only; none of them is a hardware characteristic, and no benchmark in this
package is waiting on hardware.

## What is being validated, and what is not

The package estimates the probability that **a simulated model** violates a
requirement, and bounds that probability. Validation therefore means: the
intervals have the coverage they claim, the estimators converge to known
analytic answers, and the claimed variance reductions are measured rather than
predicted. Nothing here validates any model against any vehicle, and no
interval in this package covers model error.

## Scripts and what each one establishes

| Script | Raw output | Run time | Checks |
|---|---|---|---|
| `validate_intervals.py` | `validate_intervals.txt` | 5.2 s | 13 |
| `validate_planner.py` | `validate_planner.txt` | 1.6 s | 10 |
| `validate_quadrature.py` | `validate_quadrature.txt` | 1.7 s | 5 |
| `validate_design_point.py` | `validate_design_point.txt` | 2.6 s | 5 |
| `validate_known_answer.py` | `validate_known_answer.txt` | 1.8 s | 25 |
| `validate_variance_reduction.py` | `validate_variance_reduction.txt` | 9.7 s | 19 |
| `validate_is_worse.py` | `validate_is_worse.txt` | 12.0 s | 3 |
| `validate_surrogate.py` | `validate_surrogate.txt` | 97.6 s | 10 |
| `validate_compute_budget.py` | `validate_compute_budget.txt` | 22.7 s | 2 |

All nine scripts exit 0; 92 checks in total, 0 failures. Sum of run times:
154.9 s. The longest single script is 97.6 s, inside the 180 s per-run budget
the build was held to. `python -m pytest tests/ -q` reports **203 passed in
36.1 s**. These wall times moved by up to 20 % between two consecutive runs of
the same scripts during the build (`validate_surrogate.py` 109.9 s then
97.6 s), which is why none of them is presented as a property of anything but
this machine on this day.

---

## 1. Interval coverage — exact, not simulated

`validate_intervals.py`. Coverage is computed by summing the binomial
probability mass over every `k` whose interval contains the given `p`, so these
figures carry no Monte-Carlo error.

| Check | Reference | Result | Tolerance |
|---|---|---|---|
| Clopper-Pearson two-sided coverage, minimum over `n` in {20, 50, 100, 1000} and nine `p` | Clopper & Pearson 1934; nominal 0.95 | **0.951900** | must be ≥ 0.95 |
| Wilson two-sided coverage, minimum over the same grid | Wilson 1927; nominal 0.95 | **0.904610** at `n = 20`, `p = 0.005` | reported, not required |
| Clopper-Pearson one-sided coverage, minimum over the grid | nominal 0.95 | **0.951973** | must be ≥ 0.95 |
| Wilson one-sided coverage, minimum over the grid | nominal 0.95 | **0.930825** | reported, not required |
| `k = 0`, `n = 100`, 95 % one-sided | `1 - 0.05**(1/100)` by hand = 0.0295130496070399 | **0.029513049607039932** | < 1e-15 absolute |
| `k = 0`, `n = 100`, 95 % two-sided upper | `1 - 0.025**(1/100)` by hand = 0.0362166926 | **0.036216692645** | < 1e-12 |
| `k = 0`, `n = 100`, Wilson two-sided upper | `z^2/(n+z^2)` by hand = 0.0369934982 | **0.036993498207** | < 1e-12 |
| Closed form vs Beta-quantile code path at `n` in {1, 2, 10, 137, 5000, 29956} | each other | agree | ≤ 1e-12 |

**The Wilson interval under-covers, by 4.5 percentage points at its worst on
this grid.** It is implemented and offered because it is shorter and widely
recommended, and the exact coverage study is in the repository so that choice
is made with the number in view.

### Measured result that contradicts the usual rule of thumb

"Clopper-Pearson is always the wider interval" is false at `k = 0`. The sign of
(Wilson upper − Clopper-Pearson upper) changes **exactly once for 2 ≤ n ≤ 400,
at n = 46**:

| n | CP two-sided upper | Wilson two-sided upper | Wilson − CP |
|---|---|---|---|
| 10 | 3.0849710782e-01 | 2.7753279986e-01 | −3.096431e-02 |
| 45 | — | — | −5.350167e-05 |
| 46 | — | — | +1.173603e-05 |
| 100 | 3.6216692645e-02 | 3.6993498207e-02 | +7.768056e-04 |
| 10000 | 3.6881991462e-04 | 3.8399837068e-04 | +1.517846e-05 |

No test in this package asserts containment of one interval in the other. In
the interior (`k/n = 0.005`) Clopper-Pearson is wider at every `n` tested, by
1.1 % to 5.2 %.

### The rule of three does not converge

`3/n` is widely quoted as the 95 % zero-failure bound. Its relative error tends
to `3 / (−ln 0.05) − 1 = 1.4246021e-3` **from above**, not to zero:

| n | exact one-sided bound | `3/n` relative error |
|---|---|---|
| 100 | 2.9513049607e-02 | 1.6499494e-02 |
| 1000 | 2.9912495451e-03 | 2.9253510e-03 |
| 10000 | 2.9952835978e-04 | 1.5746096e-03 |
| 100000 | 2.9956874019e-05 | 1.4396022e-03 |
| 1000000 | 2.9957277864e-06 | 1.4261021e-03 |

`3/n` is therefore conservative by about 0.14 % forever, which is harmless, but
it is not an approximation that improves.

---

## 2. Sample-size planning

`validate_planner.py`.

| Check | Reference | Result | Tolerance |
|---|---|---|---|
| `p_target = 1e-3`, 95 % | `ceil(ln 0.05 / ln(1 − 1e-3))` = `ceil(2994.23)` | **2995** | exact |
| `p_target = 1e-4`, 95 % | `ceil(29955.8)` | **29956** | exact |
| `p_target = 1e-3`, 99 % | `ceil(4602.87)` | **4603** | exact |
| Minimality: bound met at `n`, missed at `n − 1` | — | holds in **15 of 15** combinations | exact |
| `P(k = 0)` at `p_target` and the planned `n` | must equal alpha = 0.05 by construction | 0.049962 to 0.049999 | < 2e-3 |
| Estimation size for width 0.5 p at `p = 1e-4` | — | **658 443** runs, achieved width/p = 0.499999 | ≤ 0.5 |
| Crude runs for a 10 % coefficient of variation at `p = 1e-6` | `(1−p)/(p c^2)` | **99 999 900** | exact |

**The Wilson plan is smaller, and that makes it the weaker claim.** The Wilson
one-sided zero-failure bound is `z^2/(n+z^2)` with `z = 1.6448536`, i.e.
`2.70554/n` asymptotically against the exact `2.99573/n`. Planning with Wilson
asks for about 9.7 % fewer runs:

| `p_target` | Clopper-Pearson `n` | Wilson `n` | ratio |
|---|---|---|---|
| 1e-2 | 299 | 268 | 0.89632 |
| 1e-3 | 2995 | 2703 | 0.90250 |
| 1e-4 | 29956 | 27053 | 0.90309 |
| 1e-5 | 299572 | 270552 | 0.90313 |

Combined with the one-sided coverage of 0.9308 measured above, that is a
smaller campaign bought by weakening the guarantee. The package default is
Clopper-Pearson.

**Non-monotonicity audit.** The estimation predicate uses `k = round(n p)`,
which jumps, so the interval width is not monotone in `n`. In an 800-wide
window around the returned `n = 65332` for `p = 1e-3` there is **1
non-monotone step of 799**, and **no feasible `n` smaller than the returned one
inside the 512-step back-scan window**. The search is documented as returning
the smallest `n` located by bracketing plus a bounded downward scan, not a
proven global minimum.

---

## 3. Reference probabilities

`validate_quadrature.py`.

| Limit state | Reference kind | Value | Evidence |
|---|---|---|---|
| Linear Gaussian, `beta = 3.719` | closed form `Phi(-beta)` | 1.0000652593416135e-04 | exact |
| Lognormal ratio, defaults | closed form `Phi(-beta)`, `beta = 3.7190003` | 1.0000646652128005e-04 | exact |
| Rippled, `A = 0.8`, `w = 1.5` | Gauss-Hermite, 400 nodes | 4.4462124889798390e-04 | converged to 1e-15 relative |
| Rippled, `beta = 5.5`, `A = 2.5`, `w = 2.0` | Gauss-Hermite, 400 nodes | 1.8530815483852215e-04 | converged |
| Rippled, `A = 0.8`, `w = 6.0` | Gauss-Hermite, 400 nodes | 4.4893846920763430e-04 | converged |

- At amplitude 0 the quadrature reproduces `Phi(-beta)` to better than 1e-13 at
  five values of `beta`: the one-dimensional reduction is right, not only
  converged.
- Against a direct 2 000 000-sample count at `beta = 2.0`, the quadrature
  reference agrees within 0.26 to 2.59 counting-noise floors at four
  (amplitude, frequency) settings. Tolerance 4 floors.
- Node convergence: at the default 400 nodes the worst relative error over
  frequencies 1.5, 2.0, 6.0 and 12.0 is **2.524e-06**.

**Error made during the build and corrected.** The default was 200 nodes, which
is converged to only **5.709e-05** relative at frequency 12 — enough to make a
tolerance stated in counting-noise floors meaningless for a high-frequency
instance. The default was raised to 400 **after** this measurement, and the
pinned regression values in `tests/test_regression.py` were updated with it.

**Tool defect found and worked around.** On numpy 2.5.3,
`numpy.polynomial.hermite.hermgauss(n)` overflows and returns NaN weights for
`n >= 400` (verified: finite at 200 and 300, NaN at 400 and 800). The first
implementation of the rippled reference used it and silently produced NaN.
`scipy.special.roots_hermite` is finite to at least 3200 nodes and is what the
package uses.

---

## 4. Design-point search — two independent paths

`validate_design_point.py`. The design point is the basis of every
importance-sampling tilt in the package, so it is found twice: by a vectorised
ray search and by SLSQP, and compared with the closed form where one exists.

| Instance | Exact `beta` | Ray search | SLSQP | Ray relative error |
|---|---|---|---|---|
| linear `beta = 2.5`, d = 2 | 2.5 | 2.500000 | 2.500000 | 0.000e+00 |
| linear `beta = 3.719`, d = 2 | 3.719 | 3.719000 | 3.719000 | 0.000e+00 |
| linear `beta = 4.753`, d = 5 | 4.753 | 4.753000 | 4.753000 | 0.000e+00 |
| linear `beta = 3.719`, d = 8, skew direction | 3.719 | 3.719000 | 3.719000 | −9.543e-12 |
| lognormal ratio | 3.719 | 3.719000 | 3.719000 | 4.036e-08 |

On the rippled instances, where no closed form exists, the two searches agree
to between 3.1e-14 and 1.7e-13 relative:

| Instance | Smooth-part `beta` | True `beta` (both searches) |
|---|---|---|
| rippled `A = 0.8`, `w = 1.5` | 3.719000 | **3.072735** |
| rippled `beta = 5.0`, `A = 2.0`, `w = 2.0` | 5.000000 | **3.097125** |
| rippled `beta = 5.5`, `A = 2.5`, `w = 2.0` | 5.500000 | **3.097896** |
| rippled `A = 0.8`, `w = 6.0` | 3.719000 | **2.930579** |

### Measured limitation of the fast search, published rather than hidden

The ray search samples directions, so its accuracy degrades with dimension.
Raw relative error at 192 directions on the rippled limit state, whose true
answer is 3.072735 in every dimension:

| dimension | 192 directions | 512 | 2048 |
|---|---|---|---|
| 2 | 9.702e-07 | 3.980e-07 | −7.977e-08 |
| 4 | 5.052e-04 | 1.917e-03 | 4.322e-04 |
| 6 | **8.665e-02** | 2.499e-02 | 2.352e-03 |
| 10 | **3.706e-02** | 3.186e-02 | 1.735e-02 |

With the SLSQP polish step that the search runs by default, the relative error
is **−8.852e-08 in every dimension and at every direction count tested**. The
polish is on by default for that reason; `polish=False` exposes the raw
behaviour.

A design point outside `max_radius` returns `converged = False` and the origin,
rather than a plausible-looking wrong point.

---

## 5. Known-answer tests

`validate_known_answer.py`. 400 000 samples, tolerance **4 counting-noise
floors**, where the floor is `sqrt(p(1−p)/n_eff)` of the run that produced the
estimate, or the estimator's own standard error if that is larger. A 4-floor
two-sided normal tolerance has a 6.3e-5 false-alarm rate per check. No
tolerance was adjusted after seeing a result.

Eight instances × crude and analytic importance sampling = 16 checks, plus 8
subset-simulation checks on the mean of 20 replications, plus one negative
control. **All 25 pass.** Worst case:

| Instance | Estimator | Reference | Estimate | Error in floors | Limit |
|---|---|---|---|---|---|
| lognormal `sigma_r=0.35` | crude | 5.670938e-04 | 5.300000e-04 | **0.985** | 4.0 |
| linear `beta=2.5` | crude | 6.209665e-03 | 6.092500e-03 | 0.943 | 4.0 |
| linear `beta=3.719` | crude | 1.000065e-04 | 8.750000e-05 | 0.791 | 4.0 |
| linear `beta=4.753` | crude | 1.002102e-06 | 0.000000e+00 | 0.633 | 4.0 |
| rippled `beta=5.5 A=2.5` | analytic-IS | 1.853082e-04 | 1.931952e-04 | 0.366 | 4.0 |
| linear `beta=4.753` | analytic-IS | 1.002102e-06 | 9.991756e-07 | **0.002** | 4.0 |

The crude run at `beta = 4.753` observed **zero** failures in 400 000 samples,
which is correct behaviour at `p = 1.0e-06` and is the case the zero-failure
machinery exists for; it passes the known-answer check because its error of one
reference probability is 0.633 counting-noise floors.

Subset simulation is biased at finite samples per level, so its check is on the
mean of 20 independent runs with the tolerance set by the standard error of
that mean. Worst `z` across the eight instances: **0.982**.

**Negative control.** Comparing a correct importance-sampling estimate against
a deliberately wrong reference of 1e-2 is reported as **FAIL at 31.466
floors**, so the harness is known to be capable of failing.

---

## 6. Measured variance reduction

`validate_variance_reduction.py`. 100 000 samples, 40 independent
replications, equal true-evaluation basis. The empirical standard deviation at
R = 40 has a relative uncertainty of **0.113**, so a variance ratio is good to
roughly a factor of 1.25.

| Instance | reference `p` | analytic-IS VRF | subset VRF | subset evaluations |
|---|---|---|---|---|
| linear `beta=3.719` d=2 | 1.00006526e-04 | **3405** | 12.94 | 8525 |
| linear `beta=4.753` d=2 | 1.00210174e-06 | **2.697e+05** | 508.4 | 12305 |
| linear `beta=3.719` d=6 | 1.00006526e-04 | **1489** | 7.11 | 8345 |
| lognormal default | 1.00006467e-04 | **2101** | 11.60 | 8300 |
| rippled `A=0.8 w=1.5` | 4.44621249e-04 | **110.9** | 2.78 | 7400 |
| rippled `beta=5.5 A=2.5 w=2.0` | 1.85308155e-04 | **14.41** | 3.59 | 7490 |

Every method's mean is within 4 replication standard errors of its reference on
every instance. The two definitions of the variance reduction factor — against
the measured crude variance and against the exact `p(1−p)/n` — agree within
1.6 % and 52 % — measured disagreements of 11 %, 12 %, 31 %, 1.6 %, 52 % and
26 % across the six instances in the order of the table above — which is
consistent with the replication noise on a crude estimator that sees only a
handful of failures per run.

### The reported error bar is not always honest, and the ratio is measured

`se_ratio` is the mean reported standard error divided by the measured
replication spread. 1.0 is honest.

| Instance | crude | analytic-IS | subset simulation |
|---|---|---|---|
| linear `beta=3.719` d=2 | 0.967 | 1.172 | **0.330** |
| linear `beta=4.753` d=2 | **0.281** | 1.102 | **0.342** |
| linear `beta=3.719` d=6 | 1.194 | 1.280 | **0.442** |
| lognormal default | 0.954 | 1.225 | **0.421** |
| rippled `A=0.8 w=1.5` | 0.960 | 0.919 | **0.395** |
| rippled `beta=5.5 A=2.5` | 0.919 | 0.906 | **0.300** |

- **Subset simulation's reported standard error understates its true spread by
  a mean factor of 2.69** (worst 3.33). The package labels that standard error
  `subset-independence-lower-bound`, and any interval built on it carries the
  words `LOWER BOUND` in its rationale string. This is a property of the
  independence assumption, not a defect in the implementation, and it is the
  reason the README quotes replication spreads rather than that standard error.
- **The crude estimator's own plug-in standard error understates its spread by
  a factor of 3.6 at `p = 1e-6` with 100 000 runs** (`se_ratio` 0.281), because
  most replications observe zero or one failure and `sqrt(p_hat(1−p_hat)/n)`
  collapses with them. That is exactly the regime in which a campaign must
  quote the Clopper-Pearson zero-failure bound rather than a standard error,
  and the package's `CampaignReport` does.

---

## 7. Where importance sampling is WORSE — a required deliverable

`validate_is_worse.py`. 100 000 samples, 40 replications, two instances, 20
tilts each. **26 of the 40 tilts are measurably worse than plain Monte Carlo
at the same cost.** Selected rows, linear `beta = 3.719`:

| Tilt | mean estimate | relative bias | VRF | MSERF | mean ESS | verdict |
|---|---|---|---|---|---|---|
| `scale=-1` | 0.00000e+00 | −1.0000 | — | — | 0.00 | degenerate: no replication reached the failure region |
| `scale=-0.5` | 1.58115e-03 | +14.8105 | 9.994e-06 | 9.785e-06 | 0.03 | WORSE |
| `scale=-0.25` | 2.33302e-04 | +1.3329 | 0.006366 | 0.005722 | 0.35 | WORSE |
| `scale=-0.1` | 1.02464e-04 | +0.0246 | **0.1879** | 0.1878 | 2.13 | WORSE |
| `scale=1` (design point) | 1.00136e-04 | +0.0013 | 2743 | 2623 | 19268.65 | better |
| `scale=2.5` | 1.02645e-05 | −0.8974 | 0.6326 | 0.1038 | 2.47 | WORSE |
| `scale=3` | 4.82921e-10 | −1.0000 | **1.836e+08** | **0.09998** | 2.06 | WORSE |
| `orthogonal=0.25` | 1.12062e-04 | +0.1206 | **0.3263** | 0.3117 | 6.27 | WORSE |
| `orthogonal=0.5` | 1.49688e-04 | +0.4968 | **0.009271** | 0.009068 | 3.52 | WORSE |
| `orthogonal=1` | 3.54168e-04 | +2.5414 | **2.536e-04** | 2.497e-04 | 1.97 | WORSE |

The crude baseline in this sweep is a separate 40-replication run from the one
in section 6, so its variance reduction factors differ slightly from those
reported there: 2743 here against 3405 there for the design-point tilt. Both
are inside the replication noise on a variance ratio, which is a factor of
about 1.25 at R = 40, and both are reported rather than reconciled.

Three distinct failure modes, all measured:

1. **Tilting away from the failure region** (`scale < 0`). At `scale = -1` no
   replication of 100 000 samples ever reaches the failure region; the
   estimator returns exactly 0 with exactly 0 standard error.
2. **Tilting sideways** (`orthogonal`). The failure region is no rarer under
   the proposal, but every contributing sample carries a weight far from 1.
   At `orthogonal = 0.5` the variance is **107.9 times worse** than plain Monte
   Carlo (VRF 0.009271), and at `orthogonal = 1` it is **3943 times worse**.
3. **Over-tilting** (`scale >= 2`). Almost every sample fails, but the samples
   that would carry the weight are never drawn. At `scale = 3` the estimate is
   **4.82921e-10 against a true 1.00006526e-04**, a factor of 2.07e5 too
   small, with a standard error small enough to match.

### Error made during the build and corrected

The first version of this sweep reported mode 3 as **"better", with a variance
reduction factor of 1.8e+08**, because it compared variances only. A variance
reduction factor is not a quality metric for a biased estimator. The harness
was changed to also compute

    MSERF = [p(1−p)/E] / [(mean − p)^2 + Var]

against the exactly unbiased crude estimator, and
`VarianceReduction.worse_in_mse` is the field to believe when the two verdicts
disagree. At `scale = 3` the two verdicts disagree by nine orders of
magnitude. The defect is recorded here rather than quietly fixed;
`tests/test_benchmark.py::test_variance_and_mse_verdicts_disagree_for_an_over_tilted_sampler`
is the regression that keeps it visible.

An estimator that collapses to 0 has `MSERF` approaching `p(1−p)/n / p^2`,
which is 0.0999 at `p = 1e-4` and `n = 1e5`: returning zero is ten times worse
than plain Monte Carlo at this budget, and the metric says so.

---

## 8. The AI component against the analytic baseline

`validate_surrogate.py`. The analytic/importance-sampling baseline was
implemented first, in `rareverify.tilting`, and benchmarked before the
surrogate existed. Budget: **60 000 true limit-state evaluations for every
method, including the surrogate's training set.** 40 replications for the
analytic methods, 12 for the surrogate.

### Design-point recovery

| Instance | `n_train` | `beta_hat` | relative error | 2-sigma band | fit s |
|---|---|---|---|---|---|
| smooth (true 3.719) | 25 | 3.718990 | −2.607e-06 | 3.71791 – 3.72007 | 0.04 |
| smooth | 50 | 3.719079 | 2.113e-05 | 3.71814 – 3.72002 | 0.03 |
| smooth | 100 | 3.719014 | 3.793e-06 | 3.71810 – 3.71993 | 0.03 |
| smooth | 200 | 3.719052 | 1.398e-05 | 3.71813 – 3.71997 | 0.10 |
| rough (true 3.097896) | 25 | 3.099064 | 3.768e-04 | 3.09561 – 3.10383 | 0.02 |
| rough | 50 | 3.098429 | 1.720e-04 | 3.08309 – 3.11384 | 0.01 |
| rough | 100 | 3.097933 | 1.175e-05 | 3.09756 – 3.09956 | 0.11 |
| rough | 200 | 3.097919 | 7.244e-06 | 3.09888 – 3.10015 | 0.17 |

The analytic smooth-part design point on the rough instance is 5.5, which is
**+77.5 % wrong**.

### Headline result, replicated

| Instance | estimator | mean | measured spread | cov | VRF vs crude |
|---|---|---|---|---|---|
| smooth | crude | 1.016667e-04 | 4.557e-05 | 0.4483 | 1 |
| smooth | analytic-IS | 9.993796e-05 | 7.918e-07 | 0.0079 | 3313 |
| smooth | oracle-IS | 1.000075e-04 | 9.407e-07 | 0.0094 | 2347 |
| smooth | **surrogate-guided-IS** | 1.000785e-04 | 7.547e-07 | 0.0075 | 3646 |
| rough | crude | 1.995833e-04 | 5.371e-05 | 0.2691 | 1 |
| rough | analytic-IS | 1.854003e-04 | 1.405e-05 | 0.0758 | 14.62 |
| rough | oracle-IS | 1.842105e-04 | 3.587e-06 | 0.0195 | 224.2 |
| rough | **surrogate-guided-IS** | 1.847665e-04 | 4.303e-06 | 0.0233 | 155.8 |

Head to head against the analytic baseline at equal true evaluations:

| Instance | variance ratio surrogate / analytic | replication-noise 3-sigma band | verdict |
|---|---|---|---|
| smooth | **1.1006** | 0.485 – 2.063 | **not resolved; no evidence of a gain** |
| rough | **10.6563** | 0.485 – 2.063 | **surrogate wins** |

### The honest negative, with its structural reason

**On a smooth analytic limit state the surrogate cannot win, and the
measurement says so as loudly as 12 replications allow.** The reason is not
tuning:

- The analytic design point is `[3.719, 0.0]`. The surrogate's, at
  `n_train = 100`, is `[3.71900589, 0.0]`. The distance between the two tilts
  is **5.890e-06 standard-normal units**, so the two estimators are the same
  estimator to within sampling noise and there is no variance headroom to
  compete for.
- The surrogate therefore pays, deterministically, the 100 of 60 000 true
  evaluations it spent on training: a **0.167 % variance penalty** at equal
  budget.
- It also pays, on this container, a measured **0.07 to 6.75 s** of
  Gaussian-process fit plus **1.98 to 2.76 s** of design-point search, against
  a closed form that costs nothing.
- The observed variance ratio of 1.1006 is well inside the ±3-sigma
  replication-noise band of 0.485 to 2.063, so it is not evidence of anything.
  **That is the result: the surrogate's best achievable outcome on a smooth
  limit state is to be indistinguishable from a free closed form while costing
  more.** No retune was attempted and none would change it.

### The win, and its limit

On the rough instance, where the analytic smooth-part design point is 77.5 %
too far out, the surrogate's tilt gives a **10.66-times smaller variance than
the analytic baseline at equal true-evaluation budget**, well outside the
replication noise. It does **not** reach the oracle tilt (VRF 155.8 against
224.2, a shortfall of 31 %), because the oracle is handed unlimited exact
evaluations inside an optimiser and is a reference, not a method.

### A second honest negative: the sample-efficiency curve is flat

| Instance | `n_train` | estimate | relative error | cov | straddle fraction |
|---|---|---|---|---|---|
| smooth | 25 | 9.942885e-05 | −0.0058 | 0.0084 | 0.0010 |
| smooth | 100 | 9.998262e-05 | −0.0002 | 0.0084 | 0.0000 |
| smooth | 300 | 1.006698e-04 | +0.0066 | 0.0084 | 0.0000 |
| rough | 25 | 1.845715e-04 | −0.0040 | 0.0225 | 0.0040 |
| rough | 100 | 1.834032e-04 | −0.0103 | 0.0222 | 0.0010 |
| rough | 300 | 1.852828e-04 | −0.0001 | 0.0235 | 0.0010 |

On these two-dimensional limit states 25 training evaluations already place the
tilt well enough that more do not help, so there is no interesting
sample-efficiency trade-off to report. That is a property of the test problems,
not a general result, and it is stated as such rather than dressed up as a
finding. The surrogate's cost here is dominated by its fit and search wall
time, not by its training-set size.

### The surrogate-only estimator is biased, and its error bar does not know

Replacing the true limit state by the surrogate gives an estimator whose error
is the surrogate's error near the boundary and is not covered by its own
standard error. Measured on the high-frequency rippled instance
(`beta = 3.719`, `A = 0.8`, `w = 6.0`, reference 4.48938469e-04):

| `n_train` | surrogate-only estimate | relative error | its own se | \|error\| / se | surrogate-guided (true g) | guided relative error |
|---|---|---|---|---|---|---|
| 20 | 3.298484e-05 | −0.9265 | 8.129e-07 | **511.66** | 7.506879e-04 | +0.6721 |
| 30 | 2.823918e-04 | −0.3710 | 9.682e-06 | **17.20** | 4.363961e-04 | −0.0279 |
| 60 | 4.500197e-04 | +0.0024 | 1.110e-05 | 0.10 | 4.410407e-04 | −0.0176 |
| 120 | 4.451116e-04 | −0.0085 | 1.108e-05 | 0.35 | 4.387880e-04 | −0.0226 |
| 240 | 4.369609e-04 | −0.0267 | 1.063e-05 | 1.13 | 4.708746e-04 | +0.0489 |

The nuance, reported rather than suppressed: **at 60 training evaluations and
above, on these two-dimensional problems, the surrogate-only bias falls below
its own Monte-Carlo noise and is no longer detectable at that budget.** The
failure is a small-design failure, not a universal one. The surrogate-guided
estimator, which evaluates the true limit state, stays within 67.2 % of the
reference even at the training size where the surrogate-only estimator is wrong
by a factor of 14 — the bad surrogate costs it variance, not correctness.

### Compute: restarts buy nothing

| Instance | `n_restarts_optimizer` | fit + search seconds | `beta_hat` |
|---|---|---|---|
| smooth | 0 | 2.14 | 3.719036 |
| smooth | 2 | 2.15 | 3.719036 |
| rough | 0 | 1.95 | 3.097913 |
| rough | 2 | 2.04 | 3.097913 |

Identical to six decimal places at two to three times the cost. The default is
0 restarts and the budget goes to samples.

Fit wall times on this container are erratic at the 2x level between otherwise
identical calls, because the L-BFGS iteration count varies with the random
initial kernel. They are budgeting figures, not method characteristics.

---

## 9. Compute budget

`validate_compute_budget.py`. Measured on 2 cores.

| Operation | Size | Seconds |
|---|---|---|
| crude Monte Carlo, d = 2 | 4 000 000 | 0.111 (36.2 M samples/s) |
| crude Monte Carlo, d = 10 | 4 000 000 | 0.558 (7.2 M samples/s) |
| importance sampling, d = 2 | 1 000 000 | 0.059 |
| subset simulation | 10 000 per level | 0.005 |
| surrogate fit + design point | `n_train = 100` | 2.28 |
| surrogate fit + design point | `n_train = 800` | 9.51 |

Peak array memory is bounded by the 250 000-sample batch: 4.0 MB at dimension
2, 20.0 MB at dimension 10. The surrogate's kernel matrix at the `n_train`
cap of 2000 is 32.0 MB.

---

## 10. Errors made during this build and corrected

Recorded here rather than quietly fixed, as the build standard requires.

1. **The variance reduction factor was reported without the bias.** The first
   tilt sweep labelled an over-tilted sampler "better" with VRF 1.8e+08 while
   its estimate was five orders of magnitude below the truth. Fixed by adding
   the mean-squared-error reduction factor and making `worse_in_mse` the
   verdict. Section 7.
2. **The campaign sized itself with one bound and judged itself with another.**
   `run_campaign` planned from the one-sided Clopper-Pearson limit
   (9.999942e-05 at `n = 29956`) and then compared the result against the upper
   end of the two-sided interval (1.2313568e-04), so a campaign that had
   demonstrated its target reported that it had failed. Fixed by adding
   `CampaignReport.one_sided_upper` and basing the verdict on it;
   `tests/test_integration.py::test_campaign_report_and_plan_agree_on_the_bound_they_use`
   is the regression.
3. **The Gauss-Hermite reference defaulted to too few nodes.** 200 nodes is
   converged to only 5.709e-05 relative at ripple frequency 12. Raised to 400
   after measuring it. Section 3.
4. **`numpy.polynomial.hermite.hermgauss` returns NaN at 400 nodes and above**
   on numpy 2.5.3 and was used in the first implementation. Replaced with
   `scipy.special.roots_hermite`. Section 3.
5. **The ray-search design point was wrong by 8.7 % in six dimensions.** The
   first version had no polish step; the error was found by comparing against
   SLSQP, not by inspection. Section 4.
6. **The surrogate's design-point search originally took 91 s** for two
   instances because SLSQP called the Gaussian process one row at a time. It
   was replaced with a vectorised ray search plus one SLSQP polish, measured at
   2.1 s for the same work, and the two paths are cross-checked in section 4.
7. **A placeholder check with a hardcoded `True`** was written into
   `validate_surrogate.py` section 4 while the script was being drafted and was
   replaced with the computed condition before the final run. It is listed here
   because an always-passing check is worse than no check.

## 11. What this validation does not establish

- Nothing here is evidence about any vehicle, any flight-control law, or any
  requirement other than the synthetic limit states defined in
  `rareverify.limitstates`.
- No interval in this package accounts for model error, for a mis-specified
  input distribution, or for a wrongly written requirement. All three are
  normally larger than the statistical uncertainty a campaign reports.
- Correlated runs break the independence assumption every interval here rests
  on, and nothing in the package detects that.
- Every measured variance reduction factor is for a two- to eight-dimensional
  problem with a single, connected, mildly non-convex failure region. The
  mean-shift family is known to fail on multi-modal failure regions and that
  failure is not characterised here.
- Subset simulation's bias and its correlated-sample variance are measured only
  at `n_per_level` of 1000 to 10000 and `p0 = 0.1`.
