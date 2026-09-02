# Validation — momentummgr 0.1.0

**Validation level 2.** Every number below was produced by running the script named
beside it in this session, on Python 3.11 with numpy 2.4.4, scipy 1.17.1 and
scikit-learn 1.8.0, on two CPU cores. Each script writes its raw stdout to the
`*_output.txt` file next to it, and those files are the evidence; this page is a summary
of them and adds nothing they do not contain.

Reproduce everything, from `products/P029/`:

```bash
cd validation
python3 hand_calculations.py          #  32 checks,   0.1 s
python3 p027_cross_check.py           #  22 checks,  30.8 s
python3 magnetic_controllability.py   #  18 checks,   6.5 s
python3 wheel_allocation.py           #   9 checks,   9.5 s
python3 learned_vs_fixed_ci.py        #   7 checks, 116.2 s
```

Totals: **88 checks, 88 passed, 0 failed.** Two expectations the author held before
running were wrong and are recorded below as findings rather than removed.

---

## 1. The cross-check against P027 DisturbTorque

`p027_cross_check.py` — 22 checks, all passed.

The batch specification requires this package's momentum accumulation to reproduce P027
`disturbtorque`'s for the same environment, from a torque model implemented
independently. It does, to eleven significant figures.

**What is shared and what is not.** Shared: the *inputs* — the reference vehicle
(inertia diag(4, 8, 10) kg m², drag 0.6 m² at Cd 2.2, sunlit 1.2 m² at q = 0.6, both
centres of pressure at (0.02, 0.02, 0.05) m, residual dipole (0.05, 0.05, 0.10) A m²), the
orbit (circular 500 km, i = 51.6°, RAAN 0, nadir with 5° pitch and 5° roll), the beta
angle (20°), the Vallado exponential density table, the centred non-tilted dipole with
B₀Re³ = 7.96e15 T m³, the 1361 W m⁻² solar constant and the cylindrical umbra. Not shared:
any line of code. `momentummgr` implements the four torques, the frames, the field and the
eclipse from the cited sources, and integrates them by **Gauss-Legendre quadrature in
argument of latitude with the solar term split at closed-form eclipse boundaries**, where
P027's package function uses a uniform-grid trapezoidal rule.

| Check | Reference | This package | Agreement | Tolerance |
|---|---|---|---|---|
| Orbital period | P027 5676.9780 s | 5676.9780285 s | 5.0e-09 rel | 1e-7 |
| cos β | P027 0.9396926208 | 0.9396926208 | 1.5e-11 rel | 1e-10 |
| Eclipse entry / exit, argument of latitude | P027 113.473596° / 246.526404° | 113.473596° / 246.526404° | 2.3e-07° / 2.3e-07° | 1e-6° |
| Eclipse fraction | P027 0.3695911346 | 0.3695911346 | 5.7e-11 rel | 1e-10 |
| Gravity-gradient body torque | P027 (6.332938, 1.907139, −0.1112353) µN m | same to printed precision | 3.2e-13 N m | 2e-6 rel |
| Aerodynamic body torque (no co-rotation) | P027 (−0.03615203, −1.281033, 0.5268739) µN m | same to printed precision | 3.1e-13 N m | 2e-6 rel |
| Gravity-gradient \|Δh\| per orbit | P027 1.084062e−02 N m s | 1.0840622384e−02 | 2.2e-07 rel | 2e-6 |
| Aerodynamic \|Δh\| per orbit | P027 6.911776e−03 N m s | 6.9117755125e−03 | 7.1e-08 rel | 2e-6 |
| Magnetic \|Δh\| per orbit | P027 2.236218e−03 N m s | 2.2362180824e−03 | 3.7e-08 rel | 2e-6 |
| **Solar Δh vector** | P027 QUADPACK (−1.2065216122, −1.2863853089, 3.7137042242)e−04 N m s | identical to 11 figures | **1.2e-11 rel** | 5e-10 |
| **Total Δh vector** | P027 (−2.4484285744, 2.8372528957, −2.2167544874)e−03 N m s | identical to 11 figures | **1.8e-11 rel** | 5e-10 |
| Smooth-source \|Δh\| | P027 4.573122077267e−03 N m s | same | 5.4e-14 rel | 5e-12 |

**The one apparent disagreement, and what it actually is.** Against P027's *sampled*
per-source table at N = 11521 the solar and total magnitudes differ by 1.9e−05 and
3.9e−06 relative, outside the 2e−06 that seven printed figures would justify. That is not
a disagreement between the implementations. P027's table row for the solar term is its
trapezoidal value on a uniform grid, and P027's own output states that this row sits
1.69e−04 relative from its QUADPACK reference, with a derived eclipse-edge bound of
1.20e−03. This package's Gauss-Legendre value is 4.44e−15 N m s from P027's QUADPACK
reference (1.08e−11 relative), while P027's own N = 11521 row is 7.80e−09 N m s from it
(1.90e−05 relative) — six orders of magnitude further. Both codes agree on the physics; they differ on quadrature, in the direction the
error analysis predicts. Checked and recorded as such.

**Convergence of this package's own quadrature.** 8, 16, 32, 64, 96, 192 nodes: the total
changes by 5.7e−06 relative between 8 and 384 nodes and by at most 1.0e−13 for any node
count of 16 or more. The agreement above is not an accident of one node count.

**The sampled trapezoid this package also provides**, against its own Gauss-Legendre
reference:

| N | total | solar | three smooth sources |
|---|---|---|---|
| 181 | 7.04e−04 | 7.46e−03 | 1.6e−14 |
| 361 | 3.81e−05 | 4.04e−04 | 1.5e−14 |
| 721 (default) | 3.31e−04 | 3.51e−03 | 1.4e−14 |
| 1441 | 1.46e−04 | 1.55e−03 | 1.8e−14 |
| 2881 | 5.35e−05 | 5.67e−04 | 1.5e−14 |

The solar error does not fall as a power of N and cannot: the torque steps to and from
zero between two adjacent samples, so the error is set by where the two eclipse edges fall
inside their sample intervals. This is the same behaviour P027 documents, reproduced
independently.

**Internal consistency of the wheel dynamics.** Integrating
`dh_w/dt = T_dist − ω × (Iω + h_w)` over one orbit with 200 000 trapezoidal steps and
converting to inertial gives Δh = (−2.4484292103, 2.8372489870, −2.2167566048)e−03 N m s
against the quadrature's (−2.4484285744, 2.8372528957, −2.2167544874)e−03 N m s: 1.03e−06
relative. The scheduler's dynamics and the cross-checked accumulation are therefore the
same physics, which is the only reason the cross-check bears on the rest of the package.

---

## 2. Hand calculations

`hand_calculations.py` — 32 checks, all passed. Every target is worked out from the
printed formula with the constants written out; the script shows the arithmetic.

| Quantity | Hand value | Code |
|---|---|---|
| P₁AU = 1361/299792458 | 4.5398073356e−06 N m⁻² | agrees to 1e−15 |
| Equatorial surface field 7.96e15/6371200³ | 3.0778635e−05 T | agrees to 1e−7 |
| Period at 500 km | 5676.9780 s | agrees to 1e−7 |
| Gravity gradient, I = diag(4,8,10), 30° off nadir | 3.1825644e−06 N m about x, zero elsewhere | agrees to 1e−7 |
| Worst-case gravity gradient (3µ/2R³)ΔI | 3.6749088e−06 N m, at 45° | agrees to 1e−15 |
| ρ(500 km), a table base altitude | 6.967e−13 kg m⁻³ exactly | exact |
| Aerodynamic torque, 0.05 m arm, 7600 m s⁻¹ | −1.3279659e−06 N m about y | agrees to 1e−7 |
| SRP torque, 1.2 m², q = 0.6, 0.02 m arm | 1.7432860e−07 N m about y | agrees to 1e−7 |
| Dipole torque, m = (0.05,0.05,0.10), B = 3e−05 ẑ | (1.5, −1.5, 0)e−06 N m | agrees to 1e−15 |
| Dipole field at 7000 km equator / pole | 2.3206997e−05 T / −2× that | agrees to 1e−14 |
| Pyramid A Aᵀ | (4/3)I | agrees to 1e−14 |
| Pyramid guaranteed envelope | (4/3)h_max = 0.0666667 N m s | agrees to 1e−14 |
| Thruster couple, Δh = 0.05, arm 0.5 m, Isp 220 s | 0.2 N s, 9.2701474e−05 kg | agrees to 1e−7 |

---

## 3. Magnetic desaturation controllability

`magnetic_controllability.py` — 18 checks, all passed. The spec requires the
field-direction controllability constraint to be demonstrated and the uncontrollable
direction quantified rather than hidden.

**The constraint.** Over 200 random fields the cross-product operator [B×] has singular
values (|B|, |B|, 0) to 1.56e−16 and 8.88e−16 relative: rank 2, null direction along B.
Every torque the dumping law commands is perpendicular to B to 1.42e−16 relative, and the
law removes exactly `h − (h·B̂)B̂` to 1.04e−15 relative. A momentum lying along B produces a
zero dipole command and is 100 % uncontrollable, exactly.

**How much is uncontrollable on a real orbit**, for a momentum along ECI z, 500 km,
3601 samples per orbit:

| Inclination | mean uncontrollable fraction | fraction of the orbit above 0.5 | above 0.9 |
|---|---|---|---|
| 0° | 1.0000 | 1.0000 | 1.0000 |
| 28.5° | 0.5952 | 0.5618 | 0.2063 |
| 51.6° | 0.4627 | 0.3119 | 0.1241 |
| 70° | 0.5811 | 0.6279 | 0.1041 |
| 90° | 0.6367 | 0.6668 | 0.2869 |
| 98° | 0.6275 | 0.6612 | 0.2669 |

**Finding, and an expectation of the author's that was wrong.** The mean uncontrollable
fraction is **not monotone in inclination**. It falls to a minimum of 0.4627 at 51.6° and
rises again to 0.6367 at 90°: a near-polar orbit is *worse* than a mid-inclination one for
dumping momentum along the Earth's rotation axis, because near the poles the dipole field
is itself close to z. The check as originally written asserted monotonicity, failed, and
was replaced by a check that asserts the non-monotonicity and reports the minimum. Anyone
who assumes "more inclination means more magnetic authority" will size a polar mission
wrongly.

**Time-averaged controllability Gramian** `G = ⟨I − B̂B̂ᵀ⟩` over one orbit. Its trace is
exactly 2 at every inclination (worst deviation 3.6e−15). Eigenvalues are the fraction of
the orbit for which each principal direction is dumpable:

| Inclination | λ_min | λ_mid | λ_max | λ_max/λ_min |
|---|---|---|---|---|
| 0° | 0.000000 | 1.000000 | 1.000000 | ∞ (one axis blocked outright) |
| 28.5° | 0.380764 | 0.805869 | 0.813367 | 2.136 |
| 51.6° | 0.605071 | 0.616911 | 0.778018 | 1.286 |
| 70° | 0.530333 | 0.530838 | 0.938829 | 1.770 |
| 90° | 0.500000 | 0.500000 | 1.000000 | 2.000 |
| 98° | 0.504872 | 0.504884 | 0.990245 | 1.961 |

**Closed-loop demonstration.** Starting with 0.02 N m s along the instantaneous field and
running the cross-product law for one orbit at gain 5e−4 s⁻¹, with no disturbance and no
dipole limit:

* with the field **frozen**, the component along B is unchanged to 1e−12 relative and
  |h| is unchanged to 1e−12 relative — nothing at all is dumped;
* with the **real rotating field**, |h| falls from 2.0e−02 to 2.8108e−03 N m s, a factor of
  7.115, i.e. 85.9 % removed.

That pair is the constraint and its resolution: magnetic desaturation is never
instantaneously three-axis controllable, and is controllable on average only because the
field direction sweeps.

---

## 4. Wheel allocation and zero-speed avoidance

`wheel_allocation.py` — 9 checks, all passed.

* **Exactness.** Over 500 random requests per array, biased and unbiased, the worst
  relative reconstruction error of the body momentum is 9.34e−13 (pyramid) and 2.19e−13
  (tetrahedron).
* **The exact maximiser is a maximiser.** Against a 20001-point brute-force scan of the
  null coefficient on 200 random requests, the exact enumeration is never beaten; worst
  shortfall 0.0 of h_max.
* **What biasing buys and costs**, over 2000 directions at 40 % of the envelope, as
  fractions of h_max:

| Mode | mean min\|h\| | worst min\|h\| | mean max\|h\| | worst max\|h\| |
|---|---|---|---|---|
| minimum norm | 0.0707 | 0.0001 | 0.3457 | 0.4000 |
| biased, envelope 1.0 | 0.4045 | 0.3468 | 1.0000 | 1.0000 |
| biased, envelope 0.7 | 0.1757 | 0.1035 | 0.4013 | 0.7000 |
| biased, envelope 0.5 | 0.1754 | 0.0976 | 0.3985 | 0.5000 |

  The minimum-norm worst case is a wheel at 0.0001 of h_max, i.e. essentially at rest.

* **On a real trajectory** (reference smallsat, three orbits, no desaturation, 2163
  samples): minimum-norm allocation spends **47.70 %** of the run with a wheel inside a
  5 %-of-h_max low-speed band; biasing at 0.7 envelope spends **0.00 %**. The crossing
  counts are 20 either way — biasing does not stop a wheel changing sign, it removes the
  dwell, and quoting crossings alone would have hidden the whole effect.
* **The honest control.** A three-wheel orthogonal array has a null space of dimension 0.
  Allocation with and without biasing is bit-identical, and it spends 38.92 % of the same
  run in the low-speed band. Nothing in this module can help it.

**Finding: biasing introduces discontinuous wheel commands.** The null coefficient is
chosen afresh at every sample with no memory, and the maximiser switches between symmetric
branches. Largest single-sample step in wheel momentum along the trajectory, at a 7.9 s
sample interval:

| Allocation | largest step |
|---|---|
| minimum norm | 0.000060 N m s |
| biased, full envelope | 0.095112 N m s |
| biased, 70 % envelope | 0.065112 N m s |

0.0951 N m s in 7.9 s is a torque demand of order 1e−02 N m against wheels whose useful
torque is of order 1e−03 N m: not realisable. This is a defect of the implementation, is
visible in `screenshots/wheel_zero_speed_avoidance.png`, and is **not fixed** in 0.1.0. A
flight implementation must rate-limit the null coefficient or add hysteresis to the branch
choice. It is repeated in the README Limitations and in the `allocate` docstring.

---

## 5. Learned scheduler against the tuned baseline

`learned_vs_fixed_ci.py` — 7 checks, all passed, 116.2 s.

**Protocol.** Four disjoint simulated seed blocks: fitting 1000–1059 (60 episodes),
knob tuning 2000–2024 (25), held out 5000–5079 (80), calibration 5000–5024 (a subset of
held out, used only for classification metrics). The classical baseline is tuned on the
same 85 non-held-out episodes the learned model gets, by grid search over both of its
thresholds (28 pairs; the best five differ by 0.011 in mean cost, so the surface is flat).
Both policies get the identical safety override at 0.95 of the envelope. Differences are
**paired by episode** and carry a 95 % percentile bootstrap interval over 10 000 resamples.

Chosen baseline: on = 0.60, off = 0.48, training mean cost 0.074266.
Learned model: gradient-boosted trees, 150 estimators of depth 3, 3413 rows, positive
label rate 0.0847, decision threshold 0.05, deferral band 0.70, 31.3 s to train.

**Held-out results, 80 episodes:**

| Metric | baseline | learned | difference | 95 % CI | verdict |
|---|---|---|---|---|---|
| Magnetorquer duty | 0.070837 | 0.056150 | −0.014687 | [−0.01882, −0.01072] | **learned better** |
| Time near saturation | 0.000044 | 0.003586 | +0.003542 | [+0.00035, +0.00883] | **baseline better** |
| Dipole cost [A m² s] | 3665.18 | 2842.37 | −822.81 | [−1044.79, −605.68] | **learned better** |
| Combined episode cost | 0.070880 | 0.059736 | −0.011145 | [−0.01639, −0.00501] | **learned better** |
| Peak \|h\|/envelope | 0.653456 | 0.685692 | +0.032236 | [+0.01965, +0.04561] | **baseline better** |

Envelope exceedances: 0 of 80 for both policies.

**How to read that.** The learned scheduler buys a real 20.7 % reduction in magnetorquer
duty by spending saturation margin. It is not uniformly better and is not reported as
such.

**Sensitivity to the cost weight** (policies not re-tuned; only the weight used to score
their recorded outcomes changes):

| Saturation weight | baseline | learned | difference | 95 % CI | verdict |
|---|---|---|---|---|---|
| 0.25 | 0.070848 | 0.057046 | −0.013801 | [−0.01795, −0.00990] | learned better |
| 0.50 | 0.070859 | 0.057943 | −0.012916 | [−0.01720, −0.00866] | learned better |
| 1.00 | 0.070880 | 0.059736 | −0.011145 | [−0.01646, −0.00503] | learned better |
| 2.00 | 0.070924 | 0.063322 | −0.007602 | [−0.01543, +0.00356] | **indistinguishable** |
| 4.00 | 0.071012 | 0.070495 | −0.000517 | [−0.01410, +0.02101] | **indistinguishable** |

The combined-cost verdict is a verdict about the weight from 2.0 upward. Reported as
indistinguishable there, not as a win.

**Confidence output.** On 1426 held-out classification rows with positive rate 0.0428, the
Brier score is 0.039862 against 0.040947 for a constant base-rate predictor: a skill of
+0.0265. Reliability:

| Confidence bin | n | mean predicted | observed rate | gap |
|---|---|---|---|---|
| [0.00, 0.05) | 1072 | 0.0192 | 0.0252 | −0.0059 |
| [0.05, 0.10) | 178 | 0.0702 | 0.0337 | +0.0365 |
| [0.10, 0.20) | 95 | 0.1379 | 0.1053 | +0.0327 |
| [0.20, 0.40) | 56 | 0.2727 | 0.1607 | **+0.1120** |
| [0.40, 1.00) | 25 | — | — | fewer than 20 rows per bin, not scored |

The model is **overconfident above 0.05**, by up to 0.112 in the [0.20, 0.40) bin. The
confidence is a decision score with modest skill, not a calibrated posterior, and is
reported as such.

**Headroom.** On the 25-episode calibration subset: tuned baseline 0.061755, learned
0.045545, non-causal offline search 0.030835. The learned scheduler captures **52.4 %** of
the headroom the search shows to exist, and the search cannot fly.

**Integrator sensitivity.** Repeating the entire held-out evaluation at 10 substeps per
window instead of 5 moves no reported mean by more than 0.001148.

**Finding, from an integrator that was rejected.** Explicit Euler was used first. At the
default 5 substeps per 600 s window it overstated the baseline's magnetorquer duty by
49 % against a converged reference (0.1292 against 0.0866 at 160 substeps), because inside
an active window the dipole command shrinks as momentum is removed and a first-order step
does not see it. The integrator was replaced with Heun's method, which at 5 substeps is
within 4e−05 of its own 80-substep value. Every scheduler number in this repository is
from the second-order scheme. This is recorded because the first set of benchmark numbers
was wrong and the integrator convergence test is what caught it.

---

## What is not validated

* No comparison against a real spacecraft, real telemetry or a real desaturation log
  exists anywhere in this repository. There is none to compare against.
* The geomagnetic field is a centred dipole (tilted and Earth-rotating in the scheduling
  episodes, untilted for the P027 cross-check). Against IGRF its *direction* errs by tens
  of degrees, and magnetic desaturation is a function of direction. Every controllability
  number above is therefore a geometry study of a dipole field, not a prediction for a
  specific vehicle on a specific day.
* The atmosphere table has no solar-activity dependence, so the aerodynamic column above
  400 km carries at least a factor-of-several uncertainty from its input model.
* The learned scheduler is validated only inside the episode distribution in
  `DATASET_CARD.md`. Nothing here says how it behaves outside it.
