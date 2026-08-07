# Validation — estimkit 0.1.0 (Level 1, Educational)

Evidence level: hand-solved algebraic cases, published closed-form
comparisons, and internal-consistency checks. **Every number below was
produced by running the committed scripts in this session**; the raw
stdout of each run is committed alongside.

| Script | Raw output | Result |
|---|---|---|
| `validation/riccati_steady_state.py` | `riccati_steady_state_output.txt` | PASS |
| `validation/smoother_rms.py` | `smoother_rms_output.txt` | PASS |
| `validation/ukf_vs_kf_linear.py` | `ukf_vs_kf_linear_output.txt` | PASS |
| `validation/covariance_health.py` | `covariance_health_output.txt` | PASS |

All scripts are rerunnable from the product root with
`PYTHONPATH=src python validation/<script>.py` and exit non-zero on failure.
Total runtime of the four scripts: about 75 s on the 2-core build
environment (well inside the 3-minute budget).

---

## 1. Steady-state algebraic Riccati equation

### 1a. Scalar random walk — solved by hand, full arithmetic

Model: `x_k = x_{k-1} + w`, `Var(w) = q`; `z_k = x_k + v`, `Var(v) = r`
(`F = H = 1`).

Write `p` for the steady-state **predicted** variance `P⁻_∞` and `p⁺` for
the steady-state **posterior** variance. The two filter recursions at the
fixed point are

```
update :  p⁺ = (1 − K) p,        K = p / (p + r)
predict:  p  = p⁺ + q
```

Substituting `K`:

```
p⁺ = (1 − p/(p+r)) p = p · r/(p + r)
```

and inserting into the predict equation:

```
p = p·r/(p + r) + q
```

Multiply through by `(p + r)`:

```
p(p + r) = p r + q(p + r)
p² + p r = p r + q p + q r
p² − q p − q r = 0
```

Positive root (the only admissible one, since a variance must be ≥ 0):

```
p = P⁻_∞ = ½ ( q + √(q² + 4 q r) )
K_∞ = p / (p + r)
P⁺_∞ = p − q          (directly from the predict equation)
```

**Worked case q = 1, r = 1.**

1. `p² − p − 1 = 0`
2. `p = (1 + √5)/2 = (1 + 2.2360679775)/2 = 1.61803398874989…` — the golden
   ratio φ.
3. `K = p/(p+1) = 1.6180339887 / 2.6180339887 = 0.61803398874989…` = 1/φ.
4. `P⁺ = p − q = 1.6180339887 − 1 = 0.61803398874989…`.

| quantity | hand value | filter (converged Riccati iteration) | \|diff\| |
|---|---|---|---|
| `P⁻_∞` | 1.618033988749895 | 1.618033988749895 | 2.220e−16 |
| `K_∞`  | 0.618033988749895 | 0.618033988749895 | 0.000e+00 |
| `P⁺_∞` | 0.618033988749895 | 0.618033988749895 | 0.000e+00 |

Converged in 19 iterations. Tolerance 1e−12 → **PASS**.

**Worked case q = 0.25, r = 4.0.**

1. `q² + 4qr = 0.0625 + 4·0.25·4 = 0.0625 + 4 = 4.0625`
2. `√4.0625 = 2.01556443707…`
3. `p = (0.25 + 2.01556443707)/2 = 1.13278221853…`
4. `K = 1.13278221853 / 5.13278221853 = 0.22069555463…`
5. `P⁺ = 1.13278221853 − 0.25 = 0.88278221853…`

Filter: `P⁻ = 1.132782218537317` (|diff| 1.776e−15),
`K = 0.220695554634329` (|diff| 2.498e−16),
`P⁺ = 0.882782218537318` (|diff| 1.110e−15). **PASS**.

**Worked case q = 2.0, r = 0.5.**

1. `q² + 4qr = 4 + 4 = 8`; `√8 = 2.82842712474619…`
2. `p = (2 + 2.82842712474619)/2 = 2.41421356237310…` = 1 + √2
3. `K = 2.41421356237310 / 2.91421356237310 = 0.82842712474619…`
4. `P⁺ = 2.41421356237310 − 2 = 0.41421356237310…` = √2 − 1

Filter matches to ≤ 4.441e−16. **PASS**.

### 1b. 2-state constant velocity — published closed form (Kalata)

Model: discrete white-noise acceleration constant-velocity model
(Bar-Shalom, Rong Li & Kirubarajan 2001, Ch. 6), position-only
measurement:

```
F = [[1, T], [0, 1]]        Q = σ_a² [[T⁴/4, T³/2], [T³/2, T²]]
H = [1, 0]                  R = σ_v²
```

Kalata's *tracking index* gives the exact steady-state α–β gains
(Kalata, P. R., "The Tracking Index: A Generalized Parameter for α-β and
α-β-γ Target Trackers", IEEE Transactions on Aerospace and Electronic
Systems, Vol. AES-20, No. 2, 1984):

```
Λ = σ_a T² / σ_v
ρ = (4 + Λ − √(8Λ + Λ²)) / 4
α = 1 − ρ²
β = 2(2 − α) − 4√(1 − α)
K_∞ = [α, β/T]ᵀ
```

| T [s] | σ_a [m/s²] | σ_v [m] | Λ | Kalata K_∞ | filter K_∞ | max \|diff\| |
|---|---|---|---|---|---|---|
| 1.0 | 0.10 | 2.0  | 0.050 | [0.270867118992629, 0.042694639037219] | [0.270867118992628, 0.042694639037219] | 3.331e−16 |
| 0.5 | 1.00 | 0.5  | 0.500 | [0.628373457204967, 0.609611796797792] | [0.628373457204967, 0.609611796797792] | 4.441e−16 |
| 2.0 | 0.02 | 10.0 | 0.008 | [0.118799444828764, 0.003754891327687] | [0.118799444828764, 0.003754891327687] | 1.505e−16 |

Tolerance 1e−12 → **PASS**.

### 1c. Independent solver cross-check (SciPy DARE)

The same 2-state predicted covariance solved with
`scipy.linalg.solve_discrete_are(Fᵀ, Hᵀ, Q, R)` — the control-form DARE
applied to the filtering dual. SciPy is used only in this validation
script, never by the library.

| case | max \|P⁻(iterated) − P⁻(SciPy DARE)\| |
|---|---|
| T=1.0, σ_a=0.10, σ_v=2.0 | 1.088e−14 |
| T=0.5, σ_a=1.00, σ_v=0.5 | 2.554e−15 |
| T=2.0, σ_a=0.02, σ_v=10  | 1.377e−12 |

Tolerance 1e−10 → **PASS**. (The third case converges slowest — 281
iterations — and carries the largest residual, consistent with a
fixed-point iteration stopped at a 1e−15 increment.)

---

## 2. RTS smoother reduces RMS error below the forward filter

Scenario (seeded, reproducible): 1-D constant velocity with continuous
white-noise acceleration, `dt = 1 s`, acceleration PSD `q̃ = 0.01 m²/s³`,
position measurements with `σ_z = 2 m`, 300 steps, `x₀ = [0, 0]`,
`P₀ = diag(100, 100)`, seed 2026.

**Single seed (2026):**

| quantity | forward filter | RTS smoother | reduction |
|---|---|---|---|
| RMS position | **1.078105 m** | **0.629825 m** | 41.58 % |
| RMS velocity | **0.399002 m/s** | **0.133870 m/s** | 66.45 % |

**Monte Carlo, seeds 0…299 (300 runs):**

| quantity | forward filter | RTS smoother | smoother wins |
|---|---|---|---|
| mean RMS position | 1.053928 m | 0.567928 m | 300/300 |
| mean RMS velocity | 0.420932 m/s | 0.129166 m/s | 300/300 |

**Covariance-ordering guarantee.** Theory says `P_{k|T} ⪯ P⁺_k`. Checked
directly by taking the minimum eigenvalue of `P⁺_k − P_{k|T}` at every
step: worst value over the run **0.000e+00** (tolerance −1e−12) — i.e. the
difference is positive semi-definite everywhere, and exactly zero at the
final step where the smoother coincides with the filter by construction.

**Filter consistency.** Mean NIS over the run = **1.0260** for `m = 1`
degree of freedom (expectation 1.0), so the forward filter is consistent
and the comparison is not an artefact of a mistuned filter.

The same scenario is reproduced by
`examples/tracking_filter_vs_smoother.py`, which prints identical numbers
and saves `screenshots/tracking_filter_vs_smoother.png`.

---

## 3. UKF reduces to the linear KF on a linear-Gaussian system

Scenario: 2-state constant velocity (CWNA), `dt = 1 s`, `q̃ = 0.05 m²/s³`,
`σ_z = 3 m`, 200 steps, seed 7. The same measurement sequence is run
through `KalmanFilter` and `UnscentedKalmanFilter` for a grid of
sigma-point parameters, and through the RTS smoother afterwards.

Peak magnitudes used to normalise (from the KF run): `|x| = 3120`,
`|P| = 52.18`, `|K| = 0.9569`, `|S| = 209`, `|x_s| = 3120`, `|P_s| = 2.883`.

| α | β | κ | abs Δx | abs ΔP | abs ΔK | abs ΔS | abs Δx_s | abs ΔP_s | **worst relative** | eps/α² |
|---|---|---|---|---|---|---|---|---|---|---|
| 1     | 2 | 0 | 9.095e−13 | 3.944e−13 | 7.777e−14 | 1.515e−12 | 9.095e−13 | 8.065e−13 | **2.798e−13** | 2.220e−16 |
| 1     | 0 | 1 | 4.547e−13 | 3.837e−13 | 3.053e−14 | 5.969e−13 | 9.095e−13 | 9.543e−13 | **3.311e−13** | 2.220e−16 |
| 0.5   | 2 | 1 | 1.364e−12 | 1.349e−12 | 8.044e−14 | 1.567e−12 | 2.274e−12 | 1.436e−12 | **4.981e−13** | 8.882e−16 |
| 1e−1  | 2 | 0 | 4.684e−11 | 4.138e−12 | 4.156e−13 | 8.098e−12 | 4.229e−11 | 9.599e−12 | **3.330e−12** | 2.220e−14 |
| 1e−2  | 2 | 0 | 5.175e−09 | 3.386e−11 | 4.725e−12 | 9.204e−11 | 4.655e−09 | 2.845e−11 | **9.871e−12** | 2.220e−12 |
| 1e−3  | 2 | 0 | 4.880e−07 | 4.771e−10 | 3.556e−11 | 6.924e−10 | 5.118e−07 | 1.228e−09 | **4.259e−10** | 2.220e−10 |

**Worst deviation over the whole grid: 4.259e−10 relative** (tolerance
1e−9) → **PASS**. In absolute terms the worst deviation is 5.118e−07 on a
smoothed position of 3120 m.

**Why the tolerance is relative, stated plainly.** The scaled transform
places the sigma points at `α√(n+κ)` standard deviations and then divides
the resulting spread by `2α²(n+κ)`. Cancellation entering the sigma points
at the `eps·|x|` level is therefore amplified by roughly `1/α²`. The
measured relative deviations track the predicted `eps/α²` column to within
a factor of about 2 across five orders of magnitude in α. The algebra of
the reduction is exact; the arithmetic is not, and small α genuinely costs
significant digits. An absolute tolerance of 1e−9 would have failed the
α = 1e−2 and α = 1e−3 rows purely because the state is ~3 km in magnitude.

The same identity is exercised in the test suite over Hypothesis-generated
affine maps (`tests/test_properties.py`).

---

## 4. Covariance symmetry and positive definiteness under Joseph form

### 4a. Long run

4-state 2-D constant-velocity filter (`[x, vx, y, vy]`, `dt = 0.1 s`,
acceleration PSD 0.02 m²/s³ per axis), 2-D position measurements with
`R = diag(4, 4) m²`, `P₀ = diag(1e4, 1e2, 1e4, 1e2)`, seed 31337,
**200 000 predict/update steps**. Every step is checked.

| quantity | value |
|---|---|
| steps | 200 000 |
| max \|P − Pᵀ\| over the run | **0.000e+00** (bit-exact symmetry) |
| min eigenvalue over the run | **2.659427e−02** (> 0 throughout) |
| max condition number | 51.0215 |
| final trace(P) | 0.635146 |

**PASS.** Bit-exact symmetry is not accidental: `symmetrize()` writes the
same floating-point sum into both triangles rather than relying on
`(P + Pᵀ)/2` alone.

### 4b. Ill-conditioned measurement geometry in float32

Two nearly parallel measurement rows `H = [[1, 1], [1, 1.001]]`, so the
second row contributes information of relative size `δ² = 1e−6`;
`R = 1e−8·I`; float32 arithmetic (eps = 1.19e−07); 500 updates with a
`1e−5·I` re-inflation between updates. Neither variant is re-symmetrised,
so the raw behaviour of each formula is visible.

| form | min eigenvalue | max \|P − Pᵀ\| |
|---|---|---|
| Joseph `(I−KH)P(I−KH)ᵀ + KRKᵀ` | **+1.992004e−09** | 9.313e−10 |
| short `(I−KH)P` | **−2.778614e−01** | 2.634e−02 |

The short form loses positive definiteness catastrophically; the Joseph
form does not. **PASS** (criterion: Joseph stays positive definite).

### 4c. Sub-optimal gain, double precision

The Joseph form is valid for any gain; the short form only for the optimal
Kalman gain. With `P = [[1, 0.2], [0.2, 0.5]]`, `H = [1, 0]`, `R = 0.01`,
the optimal gain is `K_opt = [0.99009901, 0.1980198]`; an over-relaxed
`K = 1.5 K_opt = [1.48514851, 0.2970297]` is applied to both forms:

| form | min eigenvalue |
|---|---|
| Joseph | **+2.456274e−01** |
| short `(I−KH)P` | **−4.952091e−01** |

`KH > I` along the measured direction, so `(I − KH)P` has a negative
eigenvalue — the covariance is no longer a covariance. The Joseph form
remains positive definite. **PASS.**

This is the practical argument: fixed-gain (α–β) implementations, gain
scheduling, quantised coefficients and deliberately detuned filters all
produce non-optimal gains routinely.

---

## 5. Property-based checks (Hypothesis)

Run as part of `python -m pytest tests/ -q`; 120 examples per property.

1. **Symmetry and positive semi-definiteness under Joseph form** —
   `joseph_update` with randomly drawn `P ≻ 0`, `R ≻ 0`, arbitrary `H` and
   an arbitrary (deliberately non-optimal) `K`, for state dimensions 1–4
   and measurement dimensions 1–3. Asserts bit-exact symmetry and a
   minimum eigenvalue above `−1e−9·max|P⁺|`. Also exercised over runs of
   up to 25 predict/update steps of a full `KalmanFilter`.
2. **Zero measurement noise collapses the state onto the measurement** —
   with `R = 0` and an invertible (well-conditioned) `H`, `H x⁺ = z` to
   `1e−6·scale`, and the posterior covariance vanishes. A partial-
   observation variant pins only the measured component.
3. **The unscented transform is exact for affine maps** — for
   `g(x) = Ax + b` with random `A`, `b`, `P ≻ 0` and any
   `α ∈ [0.01, 1]`, `β ∈ [0, 3]`, `κ ∈ [−0.5, 3]` with `n + κ > 0`, the
   transform reproduces `A x̄ + b` and `A P Aᵀ` to `1e−10/α²` relative,
   the `1/α²` factor being the round-off amplification quantified in
   section 3.

---

## What was NOT validated

- No comparison against measured flight, radar or IMU data; every scenario
  in this document is synthetic and generated by the committed scripts.
- No comparison against an independent third-party filter implementation
  beyond the SciPy DARE solver used in section 1c.
- The EKF and UKF have **no** analytic reference solution here. Their
  correctness is established indirectly: both reduce exactly to the linear
  KF on linear-Gaussian problems (sections 3 and `tests/test_ekf.py`), and
  the UKF's advantage on a nonlinear problem is demonstrated empirically in
  `examples/ukf_vs_ekf_nonlinear.py`, not proved.
- Square-root and UD-factorised filtering are discussed but not
  implemented, so the claim that they are needed beyond the Joseph form's
  range is a cited statement (Bierman 1977; Maybeck 1979), not a measured
  result of this package.
- Consistency testing is limited to the mean NIS. NEES over a Monte Carlo
  ensemble with chi-squared bounds is not implemented here (it belongs to
  P012 NavBench in this portfolio).
