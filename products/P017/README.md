# EstimKit

**Status:** TESTING · **Class:** compact · **Validation level:** 1 (Educational) · **AI:** no

## Executive overview

`estimkit` is a compact, dependency-light Kalman filter family for
aerospace state estimation. It provides the linear Kalman filter, an
extended Kalman filter that takes user-supplied Jacobians (with a
central-difference fallback whose accuracy limits are documented rather
than hidden), an unscented Kalman filter with configurable
`alpha`/`beta`/`kappa` sigma-point parameters and an explicitly stated
weight convention, and the Rauch-Tung-Striebel fixed-interval smoother.
Every covariance update in the linear and extended filters uses the
Joseph form, and the package documents why that matters, when it stops
being enough, and what covariance collapse and filter divergence look like
when they happen to you.

The library depends on NumPy only. Matplotlib and SciPy are used by the
examples and validation scripts respectively, never by the library.

## Aerospace problem

Navigation, guidance and tracking all reduce to the same question: given a
dynamics model you half-trust and sensors you trust even less, what is the
state and how sure are you? The Kalman family answers it, and every GNC
codebase ends up containing some version of it. The recurring failure is
not the algebra — that is in every textbook — but the numerics and the
conventions:

- covariance matrices that silently lose symmetry, then positive
  definiteness, and take the estimate with them;
- unscented filters whose sigma-point weight convention is left
  unstated, so results cannot be reproduced across implementations;
- extended filters fed a numerically differentiated Jacobian whose error
  is invisible until the filter is inconsistent;
- steady-state gains that nobody has checked against the Riccati solution
  the filter is supposed to converge to.

This package exists to be the small, auditable version: each of those four
points is addressed explicitly, checked numerically, and written down.

## Intended users

Students and educators learning state estimation; GNC engineers who want a
transparent reference implementation to check a larger filter against, or a
quick bench for a filter/smoother trade study. It is deliberately small:
if you need square-root filtering, multiple-model estimation, or
data-association, this is not that library.

## Engineering theory

Notation: `x` state, `z` measurement, `F` transition, `H` measurement
matrix, `Q` process-noise covariance, `R` measurement-noise covariance,
`P⁻`/`P⁺` prior/posterior covariance, `K` gain. Units are the caller's,
used consistently: `P` carries squared state units, `R` squared
measurement units, `K` state per measurement unit.

### Linear Kalman filter

Model (Bar-Shalom, Rong Li & Kirubarajan, *Estimation with Applications to
Tracking and Navigation*, Wiley 2001, Ch. 5; Simon, *Optimal State
Estimation*, Wiley 2006, Ch. 5):

```
x_k = F x_{k-1} + B u_{k-1} + w_{k-1},   w ~ N(0, Q)
z_k = H x_k + v_k,                       v ~ N(0, R)
```

Recursion:

```
predict:  x⁻ = F x⁺ + B u
          P⁻ = F P⁺ Fᵀ + Q
update :  y  = z − H x⁻                      (innovation)
          S  = H P⁻ Hᵀ + R                   (innovation covariance)
          K  = P⁻ Hᵀ S⁻¹
          x⁺ = x⁻ + K y
          P⁺ = (I − K H) P⁻ (I − K H)ᵀ + K R Kᵀ      (Joseph form)
```

*Assumptions:* linear dynamics and measurement; `w`, `v` zero-mean, white,
mutually uncorrelated with the stated covariances. *Validity:* under those
assumptions the filter is the minimum-mean-square-error estimator; under
non-Gaussian noise it remains the best **linear** unbiased estimator.
Correlated, coloured or time-correlated noise requires state augmentation
and is not handled here.

Each update also returns the **normalised innovation squared**
`NIS = yᵀ S⁻¹ y`, which is dimensionless and, for a correctly specified
filter, chi-squared distributed with `m` degrees of freedom (so its
expectation is `m`). It is the cheapest available consistency diagnostic.

### Why the Joseph form

`P⁺ = (I − KH)P⁻` and the Joseph form are algebraically identical **for
the optimal gain**. They are not numerically identical, and they are not
even mathematically equivalent for any other gain.

- The short form is a **difference** of positive semi-definite matrices.
  Cancellation can drive `P⁺` indefinite, and no structural property
  prevents it.
- The Joseph form is a **sum** of two positive semi-definite congruences,
  `A P⁻ Aᵀ` with `A = I − KH`, plus `K R Kᵀ`. In exact arithmetic it is
  symmetric and positive semi-definite for **any** gain, and its
  first-order sensitivity to gain error vanishes.

That matters in practice because non-optimal gains are common: fixed-gain
(α–β) implementations, gain scheduling, coefficient quantisation on
embedded targets, deliberately detuned filters, and the EKF/UKF where `H`
is a linearisation rather than the true measurement map. Cost: roughly
twice the flops of the short form, which is irrelevant at the state
dimensions this package targets.

Sources: Bar-Shalom, Rong Li & Kirubarajan 2001, Ch. 5; Simon 2006, Ch. 6;
the form is generally attributed to P. D. Joseph (Bucy & Joseph,
*Filtering for Stochastic Processes with Applications to Guidance*, 1968).

Measured in `validation/covariance_health.py`: with an over-relaxed gain
`K = 1.5 K_opt` the short form yields a covariance with minimum eigenvalue
**−0.4952** while the Joseph form gives **+0.2456**; in float32 with a
near-singular measurement geometry the short form reaches **−0.2779**
where the Joseph form stays at **+2.0e−09**.

### When Joseph form is not enough: square-root and UD filtering

The Joseph form still forms `P` explicitly. The condition number of `P` is
the **square** of the condition number of any of its square-root factors,
so when the eigenvalue spread of `P` approaches the reciprocal of the
machine epsilon, no update written in terms of `P` is safe. Typical
triggers:

- nearly-exact measurements combined with nearly-unobservable states (a
  large `P`-eigenvalue ratio by construction);
- single-precision or fixed-point embedded arithmetic, where
  `eps ≈ 1.2e−07` and a condition number of `1e7` already exhausts the
  precision;
- large integrated navigation states (GNSS/INS with many bias and
  scale-factor states) with tight `R`;
- long runs with very small `Q`, where the filter's own convergence drives
  the condition number up.

In those regimes the correct answer is a **factorised** filter: Potter or
Carlson square-root, or Bierman's UD decomposition, or the Andrews
square-root time update. These propagate a factor whose condition number
is the square root of that of `P`, and they cannot represent a
negative-definite covariance at all, because `P` is never formed.
References: Bierman, G. J., *Factorization Methods for Discrete Sequential
Estimation*, Academic Press 1977; Maybeck, P. S., *Stochastic Models,
Estimation, and Control*, Vol. 1, Academic Press 1979, Ch. 7; Grewal &
Andrews, *Kalman Filtering: Theory and Practice*. **This package does not
implement them** — see Limitations.

### Observable symptoms of covariance collapse and filter divergence

`estimkit.covariance_health(P)` returns `asymmetry`, `min_eig`, `max_eig`,
`trace` and `condition` so these can be watched directly.

| symptom | what you observe | usual cause |
|---|---|---|
| **Covariance collapse** | `trace(P)` decays monotonically towards zero while innovations stay large or grow; `K → 0`; the filter stops responding to measurements and coasts on the model | `Q` too small (or zero) for the real dynamics; repeated updates with an over-optimistic `R` |
| **Loss of positive definiteness** | `min_eig` crosses zero; a state reports negative "variance"; Cholesky failure in UKF sigma-point generation (`estimkit` raises a `ValueError` naming covariance collapse) | short-form covariance update, cancellation, ill-conditioned `H` |
| **Asymmetry growth** | `asymmetry` climbs above round-off level; usually precedes the loss of definiteness | update not re-symmetrised |
| **Divergence** | `trace(P)` and NIS both grow without bound; **or** NIS sits far above its chi-squared bound (`m` dof) while `trace(P)` stays small — a confident, wrong filter | unmodelled dynamics; wrong `Q`/`R` ratio; EKF linearisation error large compared with the state uncertainty |
| **Non-white innovations** | innovation sequence visibly autocorrelated | model mismatch (the innovation of a correctly specified filter is white) |

### Extended Kalman filter

```
x_k = f(x_{k-1}) + w,   z_k = h(x_k) + v
F_k = ∂f/∂x |_{x̂_{k-1}},   H_k = ∂h/∂x |_{x̂⁻_k}
```

*Validity:* first order. The neglected term in the propagated mean is
`½ tr(Hess · P)`; when that becomes comparable to the state standard
deviation, the estimate is biased and the covariance optimistic.
References: Bar-Shalom, Rong Li & Kirubarajan 2001, Ch. 10; Simon 2006,
Ch. 13.

**Numerical Jacobian caveat.** `numerical_jacobian` uses central
differences with a per-component step `cbrt(eps)·max(|x_j|, 1)`, balancing
`O(h²)` truncation against `O(eps/h)` round-off; best achievable accuracy
is about `eps^(2/3) ≈ 4e−11` relative for well-scaled smooth functions.
It degrades badly when `f` is evaluated to less than full double precision
(loose ODE tolerance, interpolated tables), when state components have
wildly different scales (metres and radians in the same vector), and it is
simply **wrong** — not merely inaccurate — near a kink from angle
wrapping, `abs`, `min`/`max`, saturation or table interpolation. The error
never raises; it shows up as an inconsistent NIS or slow divergence. Cost
is `2n` function evaluations per Jacobian per step. Supply analytic
Jacobians for anything beyond exploration, and use the helper to *check*
them.

### Unscented Kalman filter — weight convention, stated

Scaled symmetric sigma-point set. With `λ = α²(n + κ) − n` and `S` the
**lower-triangular Cholesky factor** of `(n + λ)P` (so `S Sᵀ = (n+λ)P`),
the `2n+1` points are the **columns** of `S`:

```
X_0     = x̄
X_i     = x̄ + S[:, i]        i = 1 … n
X_{i+n} = x̄ − S[:, i]        i = 1 … n

W_0^m = λ/(n+λ)
W_0^c = λ/(n+λ) + (1 − α² + β)
W_i^m = W_i^c = 1/(2(n+λ))    i = 1 … 2n
```

The **mean** weights sum to exactly 1. The **covariance** weights do not,
unless `β = α² − 1`; that is intentional, and it is why `W_0^c` is usually
negative. Because `X_0 − x̄ = 0`, `W_0^c` has no effect on reconstructing
`P` from the points nor on transforming an affine map.

- `α ∈ (0, 1]` sets the spread. Small `α` keeps points local (less
  higher-order leakage) at the cost of a large negative `W_0^m` and, as
  measured in `validation/`, a round-off amplification of about `1/α²`.
- `β ≥ 0` folds in prior knowledge; `β = 2` is optimal for a Gaussian.
- `κ` is a secondary scaling; `0` and `3 − n` are both standard. Requires
  `n + κ > 0` (enforced), which with `α > 0` guarantees `n + λ > 0`.

References: Julier & Uhlmann, "A New Extension of the Kalman Filter to
Nonlinear Systems", Proc. SPIE AeroSense, 1997; Julier, "The scaled
unscented transformation", Proc. American Control Conference, 2002;
Julier & Uhlmann, "Unscented filtering and nonlinear estimation",
Proceedings of the IEEE, Vol. 92, No. 3, 2004; Wan & van der Merwe,
"The unscented Kalman filter for nonlinear estimation", Proc. IEEE
AS-SPCC, 2000; Simon 2006, Ch. 14.

The UKF here is the **additive-noise** formulation (`Q` and `R` added to
the transformed covariances). Its covariance update is
`P⁺ = P⁻ − K S Kᵀ`, the standard UKF form, since no `H` exists to write a
Joseph form with; the result is re-symmetrised each step.

### Rauch-Tung-Striebel smoother

```
A_k       = P⁺_k F_{k+1}ᵀ (P⁻_{k+1})⁻¹
x̂_{k|T}   = x̂⁺_k + A_k (x̂_{k+1|T} − x̂⁻_{k+1})
P_{k|T}   = P⁺_k + A_k (P_{k+1|T} − P⁻_{k+1}) A_kᵀ
```

initialised with `x̂_{T-1|T} = x̂⁺_{T-1}`, `P_{T-1|T} = P⁺_{T-1}`.
References: Rauch, Tung & Striebel, "Maximum likelihood estimates of
linear dynamic systems", AIAA Journal, Vol. 3, No. 8, 1965; Bar-Shalom,
Rong Li & Kirubarajan 2001, Ch. 6; Simon 2006, Ch. 9; Särkkä, *Bayesian
Filtering and Smoothing*, Cambridge 2013, Ch. 8.

For a correctly specified linear-Gaussian model `P_{k|T} ⪯ P⁺_k`, so
smoothing reduces the expected squared error at every interior step, with
the largest gain in the middle of the interval and none at the last step.
The same recursion is applied to EKF output (using the per-step transition
Jacobian) and to UKF output (using the sigma-point cross-covariance recast
as an effective transition matrix), where it is the extended/unscented RTS
smoother — an approximation, and the covariance ordering above is then no
longer guaranteed.

### Kinematic models

`constant_velocity_cwna(dt, q_psd)` — continuous white-noise acceleration:
`F = [[1, T], [0, 1]]`, `Q = q̃ [[T³/3, T²/2], [T²/2, T]]`, `q̃` in m²/s³.
`constant_velocity_dwna(dt, sigma_a)` — discrete white-noise acceleration:
`Q = σ_a² Γ Γᵀ` with `Γ = [T²/2, T]ᵀ` (rank 1, hence positive
*semi*-definite — a useful stress case). Both from Bar-Shalom, Rong Li &
Kirubarajan 2001, Ch. 6. `random_walk(q, r)` gives the scalar model whose
Riccati equation is solved by hand in `validation/VALIDATION.md`.

## Architecture

```
src/estimkit/
├── covariance.py   Joseph update, symmetrize, health diagnostics
├── linear.py       KalmanFilter, UpdateResult, FilterResult, steady_state
├── ekf.py          ExtendedKalmanFilter, numerical_jacobian
├── ukf.py          MerweSigmaPoints, unscented_transform, UnscentedKalmanFilter
├── smoother.py     rts_smooth, SmootherResult
├── models.py       constant_velocity_cwna / _dwna, random_walk
├── cli.py          argparse CLI (the only module that prints)
└── __main__.py     python -m estimkit
```

All three filters return the same `FilterResult` container, so
`rts_smooth` consumes any of them unchanged; the UKF stores an effective
transition matrix derived from the sigma-point cross-covariance for
exactly that reason. No cross-product imports; NumPy is the only runtime
dependency.

## Installation

```bash
cd products/P017
pip install -e .          # runtime: numpy
pip install -e ".[dev]"   # + pytest, hypothesis, ruff, matplotlib, scipy
```

Or run in place without installing:

```bash
PYTHONPATH=src python examples/tracking_filter_vs_smoother.py
```

## Quick start

```python
import numpy as np
from estimkit import KalmanFilter, constant_velocity_cwna, rts_smooth

F, Q = constant_velocity_cwna(dt=1.0, q_psd=0.01)   # [pos m, vel m/s]
H = np.array([[1.0, 0.0]])                          # position-only sensor
R = np.array([[4.0]])                               # sigma = 2 m

kf = KalmanFilter(F, H, Q, R)
res = kf.filter(x0=np.zeros(2), p0=np.diag([100.0, 100.0]), measurements=z)

res.x_post          # (T, 2) filtered means
res.p_post          # (T, 2, 2) Joseph-form covariances
res.nis             # (T,) normalised innovation squared, expectation m = 1

sm = rts_smooth(res)
sm.x                # (T, 2) fixed-interval smoothed means
```

Single steps, if you want the loop yourself:

```python
x, P = kf.predict(x, P)
upd  = kf.update(x, P, z_k)         # upd.x, upd.p, upd.gain, upd.nis
```

Steady-state gain, checked against the hand-solved Riccati equation:

```python
from estimkit import random_walk, steady_state
F, H, Q, R = random_walk(q=1.0, r=1.0)
P_prior, P_post, K, iters = steady_state(F, H, Q, R)
K            # [[0.618033988749895]] = 1/phi, the golden-ratio solution
```

UKF with an explicit sigma-point parameterisation:

```python
from estimkit import UnscentedKalmanFilter
ukf = UnscentedKalmanFilter(f=my_f, h=my_h, process_noise=Q,
                            measurement_noise=R,
                            alpha=1.0, beta=2.0, kappa=0.0)
```

## Configuration

There is no configuration file: the API is the configuration. The CLI
covers the two canned scenarios.

```bash
python -m estimkit steady-state --model random-walk --q 1 --r 1
python -m estimkit steady-state --model constant-velocity --dt 1 --q 0.01 --r 4
python -m estimkit track --steps 200 --dt 1 --q 0.01 --r 4 --seed 2026
python -m estimkit --json track --steps 200        # machine-readable
```

`steady-state` prints the converged `P⁻`, `P⁺` and `K`; `track` runs a
seeded constant-velocity scenario and prints filter and smoother RMS
errors plus the mean NIS. Invalid input exits with code 2 and an
actionable message (e.g. `r must be a positive finite number, got -1.0`).

Filter tuning knobs, all explicit constructor arguments: `Q` (trust in the
dynamics model), `R` (trust in the sensor), `P0`/`x0` (initial belief),
and for the UKF `alpha`/`beta`/`kappa`. The library validates shapes,
squareness, and the semi-definiteness of `Q`/`R`, and raises `ValueError`
or `TypeError` with an actionable message otherwise.

## Examples

Run from the product root; both write PNGs to `screenshots/` using the Agg
backend and were actually run to produce the committed images.

- `python examples/tracking_filter_vs_smoother.py` →
  `screenshots/tracking_filter_vs_smoother.png`. 300-step constant-velocity
  track, `σ_z = 2 m`, seed 2026. Three panels: track with the nominal
  10 m/s ramp removed (otherwise 3 km of travel hides metre-level
  differences), position error with ±1σ envelopes, velocity error with ±1σ
  envelopes. Prints RMS position 1.078105 m (filter) vs 0.629825 m
  (smoother) and RMS velocity 0.399002 m/s vs 0.133870 m/s. Runtime < 1 s.

- `python examples/ukf_vs_ekf_nonlinear.py` →
  `screenshots/ukf_vs_ekf_nonlinear.png`. Long-range polar radar tracking
  of a 2-D constant-velocity target (start 50 km × 20 km, range
  `σ_r = 50 m`), Cartesian state, `h(x) = [range, bearing]`. The EKF
  replaces the arc swept by the bearing uncertainty with its tangent; the
  neglected curvature grows with the cross-range uncertainty `r·σ_θ`.
  Three panels: a single run at `σ_θ = 10°`, mean position RMSE vs `σ_θ`
  over 50 seeds, and mean NIS vs `σ_θ` against the `m = 2` consistency
  line. Runtime ~12 s.

## Validation

Full evidence, hand arithmetic and raw script output in
`validation/VALIDATION.md`. All four scripts pass, and all numbers below
come from running them in this session.

1. **Scalar Riccati, solved by hand.** `p² − qp − qr = 0` →
   `P⁻_∞ = ½(q + √(q² + 4qr))`. For `q = r = 1` that is the golden ratio
   `(1+√5)/2 = 1.618033988749895` and `K_∞ = 1/φ = 0.618033988749895`.
   Filter (converged Riccati iteration, 19 iterations): `P⁻ = 1.618033988749895`
   (|diff| 2.220e−16), `K = 0.618033988749895` (|diff| 0), `P⁺ = 0.618033988749895`
   (|diff| 0). Two further hand cases (`q=0.25, r=4`; `q=2, r=0.5`) agree
   to ≤ 1.8e−15.
2. **2-state constant velocity vs Kalata's published α–β closed form**
   (IEEE Trans. AES-20(2), 1984): three parameter sets agree to
   ≤ 4.441e−16 on the gain; the predicted covariance additionally matches
   `scipy.linalg.solve_discrete_are` to ≤ 1.377e−12.
3. **Smoother beats filter.** Seed 2026, 300 steps: RMS position
   **1.078105 m → 0.629825 m** (41.58 % reduction), RMS velocity
   **0.399002 m/s → 0.133870 m/s** (66.45 %). Over 300 seeds the smoother
   wins on position in **300/300** runs and on velocity in **300/300**
   (mean 1.053928 → 0.567928 m, 0.420932 → 0.129166 m/s). The theoretical
   ordering `P⁺_k − P_{k|T} ⪰ 0` holds at every step (worst minimum
   eigenvalue 0.000e+00). Forward-filter mean NIS 1.0260 for 1 dof, so the
   comparison is not an artefact of a mistuned filter.
4. **UKF reduces to KF on a linear-Gaussian system.** Over a grid of six
   `(α, β, κ)` settings spanning `α = 1 … 1e−3`, the **worst relative
   deviation is 4.259e−10** (tolerance 1e−9), worst absolute 5.118e−07 on
   a 3120 m state. The relative deviation tracks the predicted `eps/α²`
   round-off amplification of the scaled transform to within about a
   factor of 2 across five decades of `α`.
5. **Joseph form preserves symmetry and positive definiteness.** 200 000
   predict/update steps of a 4-state filter: max `|P − Pᵀ|` = **0.000e+00**
   (bit-exact), minimum eigenvalue over the whole run **2.659427e−02**,
   max condition number 51.02. In float32 with a near-singular `H` the
   short form reaches minimum eigenvalue **−0.2778614** while Joseph stays
   at **+1.992e−09**; with an over-relaxed gain `1.5 K_opt` in double
   precision the short form gives **−0.4952091** and Joseph **+0.2456274**.

Property-based tests (Hypothesis, 120 examples each) cover: covariance
stays symmetric and positive semi-definite under Joseph updates with
arbitrary gains; zero-measurement-noise updates collapse the state onto
the measurement for an observable system; the unscented transform is exact
for affine maps at any admissible `(α, β, κ)`.

## Benchmark results

Measured on the 2-core build environment, double precision, 10 000
predict/update steps, timing the full batch `filter()`/`rts_smooth()`
path (which includes shape validation, symmetrisation and NIS on every
step):

| operation | state dim | measurement dim | per step |
|---|---|---|---|
| `KalmanFilter.filter` | 2 | 1 | 73 µs |
| `ExtendedKalmanFilter.filter` (analytic Jacobians) | 2 | 1 | 70 µs |
| `UnscentedKalmanFilter.filter` (α=1, β=2, κ=0) | 2 | 1 | 174 µs |
| `rts_smooth` | 2 | — | 27 µs |
| `KalmanFilter.filter` | 6 | 3 | 65 µs |
| `UnscentedKalmanFilter.filter` | 6 | 3 | 210 µs |

The UKF costs roughly 2.4–3× the linear filter at these dimensions, which
is the expected `2n+1` sigma-point propagation plus a Cholesky per
transform. Validation-script runtimes: Riccati < 1 s, smoother Monte Carlo
(300 seeds × 300 steps) 30 s, UKF↔KF grid 3 s, covariance health
(200 000 steps + float32 stress) 43 s. Everything is far inside the
3-minute compute budget.

## AI model details

Not applicable — this product contains no AI/ML components.

## Hardware requirements

Any machine running Python ≥ 3.11 with NumPy. No GPU, no compiled
extensions beyond NumPy's own. Memory is dominated by the stored
covariance history: `T·n²` doubles, i.e. about 2.4 MB for 10 000 steps of
a 6-state filter. Examples additionally need Matplotlib (Agg backend, no
display); validation additionally uses SciPy for one independent DARE
cross-check.

## Limitations

- **No square-root or UD-factorised filter.** The README explains when one
  is required; the package does not provide it. If your problem sits in
  that regime, use Bierman's UD implementation, not this.
- **Additive noise only.** The UKF is the additive-noise formulation; the
  augmented-state UKF needed for multiplicative or state-dependent noise
  is not implemented. `Q` and `R` are added directly to the transformed
  covariances.
- **No quaternion/manifold handling.** States are treated as elements of
  Rⁿ. Attitude estimation needs a multiplicative EKF with a reset step;
  that lives in P012 (NavBench) in this portfolio, not here.
- **No adaptive tuning, no multiple-model estimation, no data
  association.** `Q` and `R` are whatever you supply, fixed.
- **No smoothing-consistency guarantee for nonlinear filters.** The
  extended/unscented RTS smoother reuses the linear recursion with a
  linearised or effective transition; `P_{k|T} ⪯ P⁺_k` is only proved in
  the linear-Gaussian case.
- **NIS only, no NEES.** Consistency diagnostics are limited to the
  per-step normalised innovation squared. Monte Carlo NEES with
  chi-squared bounds is out of scope here.
- **Numerical Jacobian is a convenience, not a substitute.** See the
  accuracy caveat in Engineering theory; it is silently wrong at kinks and
  for poorly scaled states.
- **The UKF's covariance update is not Joseph form** (`P⁻ − K S Kᵀ`),
  because no `H` exists; it is re-symmetrised but carries a weaker
  structural guarantee than the linear filter's update. With a negative
  `W_0^c` and a strongly nonlinear map the transformed covariance can lose
  positive definiteness; positive-definite `R`/`Q` is what keeps it in
  check in practice.
- **Small `α` costs precision.** Quantified in validation §3: relative
  agreement degrades as `eps/α²`. `α = 1e−3` leaves roughly 6 significant
  digits.
- **Educational validation level.** Hand calculations, one published
  closed form, one independent DARE solver, and internal consistency. No
  comparison against measured flight, radar or IMU data, and no comparison
  against a certified filter implementation.
- **Deviation from the build guide:** the spec names no CLI for this
  product, but the batch stack line calls for one, so a small
  `python -m estimkit` CLI is included and tested.

## Safety statement

This software is educational. It is not flight-qualified, not certified,
and not approved for operational aerospace use.

## Roadmap

- Bierman UD and Carlson square-root implementations, benchmarked against
  the Joseph form on the ill-conditioned case already in
  `validation/covariance_health.py`.
- NEES with chi-squared bounds over Monte Carlo ensembles, to complete the
  consistency toolkit alongside NIS.
- Information-filter form for the multi-sensor / delayed-measurement case.
- Sequential (scalar) measurement processing, which avoids the `S`
  inversion entirely and is the usual companion to UD filtering.
- Iterated EKF and a second-order EKF, to bracket the linearisation-error
  discussion with measurements rather than argument.

## License

MIT — see `LICENSE`. Copyright © 2026 OPTIMA Organisation.

## Credits

This is under reserved rights obtained by OPTIMA Organisation.

## Citation

```
OPTIMA Organisation (2026). EstimKit: compact Kalman filter family
(KF/EKF/UKF/RTS) for aerospace estimation (v0.1.0) [Computer software].
Educational validation level 1.
```
