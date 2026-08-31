# WahbaKit — Validation evidence (Level 1, Educational)

**Product:** P026 WahbaKit · **Version:** 0.1.0 · **Date of run:** 2026-08-31
**Environment:** Python 3.11.15, NumPy 2.4.4, SciPy 1.17.1 (validation and
cross-checks only; the library itself needs NumPy alone), 2 CPU cores.
Every number below was produced by running the scripts in this directory in the
session that wrote this file. Each script's raw stdout is committed beside it.

| # | Check | Script | Raw output | Result |
|---|---|---|---|---|
| 1 | Conventions, frame order, and exact recovery from noise-free observations | `validate_conventions.py` | `conventions_output.txt` | PASS 15/15 |
| 2 | The four methods against each other and against the Davenport algebra | `validate_agreement.py` | `agreement_output.txt` | PASS 7/7 |
| 3 | Attitude covariance against seeded Monte Carlo | `validate_covariance.py` | `covariance_output.txt` | PASS 7/7 |
| 4 | Near-parallel observations, and the 180-degree singularity | `validate_degeneracy.py` | `degeneracy_output.txt` | PASS 9/9 |

Reproduce, from `products/P026/`:

```bash
python validation/validate_conventions.py    #  ~2 s
python validation/validate_agreement.py      # ~13 s
python validation/validate_covariance.py     # ~78 s
python validation/validate_degeneracy.py     #  <1 s
```

Total under 100 s on the 2-core build machine. The test suite (`python -m
pytest tests/ -q`) reports **145 passed, 0 failed, 0 skipped** in 12.9 s, and
`ruff check src/ tests/` is clean.

Nothing in this file was tuned to pass. Two checks failed on their first
formulation and are documented as such in §5 below: in both cases the reference
value, not the library, was wrong, and the correction is stated with its reason.

---

## Notation

Observations are pairs `(b_i, r_i)`, `b_i` in the body frame and `r_i` in the
reference frame. The attitude matrix `A` satisfies `b_i ≈ A r_i`. Quaternions
are scalar-first with `A = dcm_from_quat(q)`. The attitude error is
`δθ = log(A_est A_trueᵀ)` in the body frame, in rad, and all covariances are
`E[δθ δθᵀ]` in rad². Equation numbers refer to the module docstrings:

| Eq. | Where | What |
|---|---|---|
| O1 | `observations.py` | Shuster's unit-vector measurement model, `E[n nᵀ] = σ²(I − bbᵀ)` |
| O2 | `observations.py` | Wahba's loss `L(A) = ½ Σ w_i \|b_i − A r_i\|²` |
| O3 | `observations.py` | Fisher information `F = Σ σ_i⁻² (I − b_i b_iᵀ)` |
| O4 | `observations.py` | Observability gate `F̂ = (1/N) Σ (I − b_i b_iᵀ)`, `λ_min ≥ 1e-6` |
| D1–D3 | `davenport.py` | `B`, `S`, `z`, `σ`, and the 4×4 Davenport matrix `K` |
| Q1–Q3 | `quest.py` | QUEST characteristic quartic and the closed-form eigenvector |
| L1–L3 | `olae.py` | OLAE's Cayley/Gibbs linear system and its cost function |
| T1–T3 | `triad.py` | TRIAD's two orthonormal triads |
| V1, V2 | `covariance.py` | Optimal (Cramér–Rao) and TRIAD attitude covariance |

---

## 1. Conventions and noise-free exactness

**References.** Wahba (1965) *SIAM Review* **7**(3), 409; Shuster & Oh (1981)
*Journal of Guidance and Control* **4**(1), 70–77; Markley & Crassidis (2014),
*Fundamentals of Spacecraft Attitude Determination and Control*, Ch. 5;
Shepperd (1978) *Journal of Guidance and Control* **1**(3), 223–224 for the
matrix-to-quaternion extraction.

**Method.** 500 random quaternions compared against
`scipy.spatial.transform.Rotation`, three hand-evaluated closed forms, an
explicit test that the returned matrix is `A` and not `Aᵀ`, and noise-free
problems at N = 2, 3, 4, 6, 12 solved by all four methods.

| Check | Result | Tolerance |
|---|---|---|
| `dcm_from_quat(q)` vs `Rotation.from_quat(roll(q,-1)).as_matrix()`, 500 quaternions | **5.551e-16** | 1e-14 |
| `quat_from_dcm(dcm_from_quat(q)) − q` (all four Shepperd branches) | **2.220e-16** | 1e-12 |
| Hand closed forms: `q=[1,0,0,0]→I`; `q=[c45,0,0,s45]→90° about z`; `q=[0,1,0,0]→diag(1,−1,−1)` | **2.220e-16** | 1e-15 |
| Angle from the returned `A` to `A_true` | **7.364e-16 rad** | 1e-12 |
| Angle from the returned `A` to `A_trueᵀ` (must be large — this is the frame-order test) | **3.062 rad** | ≥ 1e-3 |
| Noise-free attitude error, TRIAD (N = 2) | **2.675e-16 rad** | 1e-11 |
| Noise-free attitude error, q-method (N = 2…12) | **1.980e-14 rad** | 1e-11 |
| Noise-free attitude error, QUEST (N = 2…12) | **9.626e-15 rad** | 1e-11 |
| Noise-free attitude error, OLAE (N = 2…12) | **3.304e-15 rad** | 1e-11 |
| `max \|AᵀA − I\|` over 200 noisy problems × 3 methods | **8.882e-16** | 1e-13 |
| `max \|det A − 1\|` | **1.332e-15** | 1e-13 |
| `max \|\|q\| − 1\|`; quaternions returned with `w < 0` | **2.220e-16**; **0** | 1e-14; 0 |
| q-method vs `scipy Rotation.align_vectors` (Kabsch), 200 noisy problems | **7.279e-15 rad** | 1e-10 |

The frame-order row is the one worth reading twice. A package that returned the
transpose would still produce orthogonal matrices, unit quaternions and a
plausible loss; only a check against a known truth distinguishes them.

---

## 2. Agreement of the four methods

**References.** Davenport (1968) NASA TN D-4696; Shuster & Oh (1981);
Mortari, Markley & Singla (2007) *Journal of Guidance, Control, and Dynamics*
**30**(6), 1619–1627; Black (1964) *AIAA Journal* **2**(7), 1350–1351.

**Method.** 500 random problems with N drawn from 2…8 and σ log-uniform on
1e-5…1e-1 rad; then a fixed four-observation geometry (three body axes plus the
body diagonal, smallest separation 54.74°) at six noise levels, 400 trials each.

| Check | Result | Tolerance |
|---|---|---|
| `max \|λ_max(QUEST) − λ_1(K)\|` — Eq. Q2 really is the characteristic polynomial of Eq. D3 | **1.443e-15** | 1e-12 |
| `max \|ψ(λ_i)\|` at **all four** eigenvalues of `K`, not just the largest | **2.443e-15** | 1e-12 |
| **QUEST's closed-form quaternion vs the q-method eigenvector** | **1.489e-14 rad** | 1e-9 |
| Worst Newton iteration count from `λ₀ = 1` | **5** | — |
| Worst pairwise disagreement of all four methods, noise-free, 200 problems | **9.068e-13 rad** | 1e-11 |
| `max \|QUEST − q-method\|` across all six noise levels | **2.038e-15 rad** | 1e-9 |
| Spread of the OLAE-to-optimum gap divided by σ, over σ ≤ 1e-2 | **0.0212** | 0.05 |
| Worst amount by which any method beat the q-method on the Wahba loss | **5.551e-16** | 1e-15 |

Departure from the Wahba optimum, four-observation geometry, 400 trials per row:

| σ [rad] | RMS \|QUEST − q-method\| | RMS \|OLAE − q-method\| | ratio to σ | RMS error, q-method | RMS error, OLAE |
|---:|---:|---:|---:|---:|---:|
| 1e-1 | 5.637e-16 | 2.099e-01 | 2.0995 | 1.051e-01 | 2.351e-01 |
| 1e-2 | 5.437e-16 | 1.840e-03 | 0.1840 | 1.095e-02 | 1.112e-02 |
| 3e-3 | 5.453e-16 | 5.925e-04 | 0.1975 | 3.280e-03 | 3.349e-03 |
| 1e-3 | 5.319e-16 | 1.763e-04 | 0.1763 | 1.131e-03 | 1.145e-03 |
| 3e-4 | 5.353e-16 | 5.659e-05 | 0.1886 | 3.221e-04 | 3.263e-04 |
| 1e-4 | 5.542e-16 | 1.802e-05 | 0.1802 | 1.084e-04 | 1.095e-04 |

Two things are established here and both are worth stating plainly.

**QUEST is the q-method, numerically.** The gap between them is 5e-16 rad at
every noise level — round-off, and independent of σ. That is what the check is
for: a QUEST implementation with the wrong sign in Eq. Q3 or a stale `κ` still
converges and still returns a unit quaternion.

**OLAE is not.** The gap is a *constant fraction of σ* — 0.176 to 0.198 over
four decades — not a constant fraction of σ². That is a first-order departure,
so OLAE is a different estimator, not a numerically noisier version of the same
one. Its RMS attitude error is correspondingly 1.3 % larger than the q-method's
in this geometry (see §3.4). This follows from Eq. L3: OLAE minimises
`Σ w_i [(1 + |g|²)|e_i|² − (g·e_i)²]`, whose gradient at the Wahba optimum is
O(σ), not O(σ²).

The σ = 1e-1 row is outside the regime the argument assumes (5.7° of sensor
noise) and is kept in the table for that reason: the ratio jumps to 2.10 and
OLAE's RMS error becomes **2.2× the q-method's**. The check on the constancy of
the ratio is therefore evaluated over σ ≤ 1e-2 only, and the excluded row is
printed by the script.

---

## 3. Attitude covariance against Monte Carlo

**References.** Shuster (1978) AIAA-78-1249; Shuster & Oh (1981); Markley &
Crassidis (2014), Ch. 5.

**Method.** Body vectors are sampled from Eq. O1 (Gaussian in the two
directions orthogonal to the true direction, then re-normalised), 10 000 seeded
trials per case, seed 20260831. The sample covariance of `δθ` is compared with
the closed form. The sampling error on a covariance entry is `√(2/10 000)` =
**1.41 %**, so the 6 % gate is loose enough not to be flaky and tight enough
that a wrong formula fails outright.

### 3.1 Eq. V1 — optimal covariance (q-method, QUEST)

`P_opt = [Σ_i σ_i⁻² (I − b_i b_iᵀ)]⁻¹`

| Case | Worst relative deviation from Monte Carlo | Tolerance |
|---|---|---|
| Equal sigmas, 1e-3 rad, four observations | **1.554e-02** | 0.06 |
| Unequal sigmas [1, 2, 5, 1] × 1e-3 rad | **1.151e-02** | 0.06 |

Per-axis diagonal deviations were 1.7 %, 1.4 %, 1.2 % (equal) and 0.8 %, 1.7 %,
0.2 % (unequal), all inside the 1.4 % sampling error to within a factor of 1.3.

### 3.2 Eq. V2 — TRIAD covariance

| Case | Worst relative deviation | Tolerance |
|---|---|---|
| `primary = 0` (σ = 1e-3 exact, σ = 5e-3 secondary) | **2.615e-03** | 0.06 |
| `primary = 1` (the roles swapped) | **1.059e-02** | 0.06 |

Both choices of primary are checked because Eq. V2 is not symmetric in the two
observations: TRIAD reproduces the primary exactly and passes its error straight
into the attitude.

### 3.3 `P_TRIAD − P_opt` is positive semi-definite

TRIAD can never beat the Cramér–Rao bound. Over a 3 × 3 grid of
(σ₁, σ₂) ∈ {1e-4, 1e-3, 1e-2}²:

| σ₁ | σ₂ | min eig of `P_TRIAD − P_opt` | relative to max\|P_TRIAD\| | excess total variance |
|---|---|---|---|---|
| 1e-4 | 1e-4 | −2.023e-24 | −2.02e-16 | 20.000 % |
| 1e-4 | 1e-3 | −1.408e-20 | −2.32e-14 | 0.010 % |
| 1e-4 | 1e-2 | −2.249e-16 | −3.74e-12 | 0.000 % |
| 1e-3 | 1e-4 | −1.597e-20 | −1.84e-14 | 97.078 % |
| 1e-3 | 1e-3 | −4.340e-23 | −4.34e-17 | 20.000 % |
| 1e-3 | 1e-2 | −1.797e-18 | −2.96e-14 | 0.010 % |
| 1e-2 | 1e-4 | −1.777e-16 | −2.05e-12 | 99.970 % |
| 1e-2 | 1e-3 | −1.248e-18 | −1.44e-14 | 97.078 % |
| 1e-2 | 1e-2 | −6.782e-21 | −6.78e-17 | 20.000 % |

Worst relative negative eigenvalue **3.735e-12**, gated at 1e-9. The gate is not
zero because Eq. V1 inverts a matrix whose condition number reaches 1.0e+04 on
this grid, and `eps × cond ≈ 2e-12` is the attainable numerical bound on an
eigenvalue that is exactly zero in exact arithmetic. The two rows where the
excess is 0.000 % are those where the primary sensor is 100× better than the
secondary: TRIAD is then effectively optimal, exactly as the σ₁ → 0 limit of
Eq. V2 predicts. The rows where the excess is 97–100 % are the reverse case,
and are the reason `primary` must be the accurate sensor.

### 3.4 OLAE against Eq. V1

| Quantity | Value |
|---|---|
| `trace(P_MC)`, OLAE | **1.168621e-06 rad²** |
| `trace(P_MC)`, q-method | **1.167665e-06 rad²** |
| `trace(P)` analytic, Eq. V1 | **1.166667e-06 rad²** |
| OLAE excess total variance over the optimum | **0.082 %** |

Eq. V1 is therefore an *optimistic* covariance for OLAE, by 0.08 % in trace on
this geometry at σ = 1e-3 rad. `attitude_covariance(obs, "olae")` returns
Eq. V1 and its docstring says so.

### 3.5 Where the first-order covariance stops describing reality

Eq. V1 and Eq. V2 are first-order results valid for σ ≪ 1 rad. 3 000 trials
per row:

| σ [rad] | trace `P` analytic [rad²] | trace `P` Monte Carlo [rad²] | ratio |
|---:|---:|---:|---:|
| 1e-4 | 1.166667e-08 | 1.170736e-08 | 1.0035 |
| 1e-3 | 1.166667e-06 | 1.181041e-06 | 1.0123 |
| 1e-2 | 1.166667e-04 | 1.143564e-04 | 0.9802 |
| 3e-2 | 1.050000e-03 | 1.038019e-03 | 0.9886 |
| 1e-1 | 1.166667e-02 | 1.151535e-02 | 0.9870 |
| 3e-1 | 1.050000e-01 | 9.223586e-02 | 0.8784 |

Worst `|ratio − 1|` for σ ≤ 1e-2 is **1.980e-02**, inside the 6 % gate. At
σ = 0.3 rad (17°) the analytic covariance overestimates the true variance by
**12 %**, because the attitude error saturates while the linear model does not.
This is reported, not gated.

### 3.6 Cross-check against SciPy — and a correction to the positioning

`scipy.spatial.transform.Rotation.align_vectors(..., return_sensitivity=True)`
returns a sensitivity matrix which, per its Notes, must be multiplied by the
harmonic mean of the observation variances, with weights inversely proportional
to those variances, to give the covariance.

| Case | max relative deviation from Eq. V1 |
|---|---|
| Equal sigmas, 1e-3 rad | **2.427e-16** |
| Unequal sigmas [1, 2, 5, 1] × 1e-3 rad | **8.503e-16** |

This check was written expecting SciPy's scalar scaling to fail for unequal
sigmas. **It does not.** SciPy already yields exactly Eq. V1, to round-off, in
both cases. The optimal attitude covariance is therefore *not* something this
package provides and SciPy does not; the difference is that WahbaKit returns it
directly in rad² rather than requiring the harmonic-mean scaling to be worked
out from the Notes, and that SciPy has no analogue of the TRIAD covariance
(Eq. V2). The README's alternatives table was rewritten to say so.

---

## 4. Near-parallel observations

**Method.** Two observations at a controlled separation `η`, swept from 90° to
0.001°, with and without the observability gate.

### 4.1 The gate follows its closed form

For a pair, `λ_min(F̂) = sin²(η/2)` exactly.

| η [deg] | `λ_min` measured | closed form | absolute dev | relative dev |
|---:|---:|---:|---:|---:|
| 90.0000 | 5.000000e-01 | 5.000000e-01 | 5.55e-17 | 1.11e-16 |
| 30.0000 | 6.698730e-02 | 6.698730e-02 | 1.94e-16 | 2.90e-15 |
| 10.0000 | 7.596123e-03 | 7.596123e-03 | 1.22e-16 | 1.61e-14 |
| 1.0000 | 7.615242e-05 | 7.615242e-05 | 1.31e-16 | 1.72e-12 |
| 0.1150 | 1.007141e-06 | 1.007141e-06 | 8.68e-17 | 8.62e-11 |
| 0.0100 | 7.615435e-09 | 7.615435e-09 | 8.19e-17 | 1.08e-08 |

Worst **absolute** deviation **1.943e-16**, tolerance 1e-13. The default gate
`λ_min = 1e-6` corresponds to `η = 0.1146°`, confirmed to 8.4e-06 deg.

### 4.2 Every solver raises below the gate

At a noise-free 0.05° separation (`λ_min = 1.904e-07`), `triad`, `q_method`,
`quest` and `olae` all raise `DegenerateObservationsError`, and every message
names the limiting frame and the separation angle:

```
observations are degenerate in the body frame: lambda_min = 1.904e-07 <
tol = 1.000e-06 (Eq. O4), equivalent to a separation of 0.0500 deg between two
equally weighted observations; smallest body-vector separation is 0.0500 deg.
The rotation about the common direction is not determined by this data. Add an
independent observation, or lower tol only if you accept an arbitrary answer
about that axis.
```

A degenerate *reference* catalogue is caught too: body vectors 90° apart with
reference vectors 1e-5 rad apart gives `λ_min(body) = 5.000e-01`,
`λ_min(reference) = 2.500e-11`, and the error names the reference frame.

### 4.3 With the gate disabled: no exception, no NaN, a wrong answer

σ = 1e-4 rad, `check_degeneracy=False`, attitude error in rad:

| η [deg] | `λ_min` | 1/sin η | TRIAD | q-method | QUEST | OLAE |
|---:|---:|---:|---:|---:|---:|---:|
| 90.000 | 4.999e-01 | 1.000e+00 | 1.368e-04 | 1.342e-04 | 1.342e-04 | 1.431e-04 |
| 10.000 | 7.596e-03 | 5.759e+00 | 3.202e-04 | 2.805e-04 | 2.805e-04 | 4.400e-04 |
| 1.000 | 7.615e-05 | 5.730e+01 | 1.800e-02 | 1.800e-02 | 1.800e-02 | 1.718e-02 |
| 0.100 | 7.615e-07 | 5.730e+02 | 5.487e-02 | 5.487e-02 | 5.484e-02 | 4.973e-02 |
| 0.010 | 7.615e-09 | 5.730e+03 | 1.582e+00 | 1.582e+00 | 1.575e+00 | 1.616e+00 |
| 0.001 | 7.615e-11 | 5.730e+04 | 2.470e+00 | 2.470e+00 | 2.144e+00 | 9.867e-01 |

The q-method error grows by a factor **1.84e+04** from 90° to 0.001°, while the
sensor noise is unchanged. `error × sin η` is flat to within a factor of 7 over
`η ≤ 10°`, confirming the `1/sin η` law. Every returned matrix is finite and
proper orthogonal at every row: nothing announces the failure, which is the
entire argument for the gate. The script catches `RuntimeError` per cell,
because in this regime QUEST's Newton iteration can stall on a near-double root
and OLAE's normal equations can go singular; in the committed run that happened
only at the π rotations of §4.4, and every cell of the table above returned a
finite number.

### 4.4 The 180-degree parametrisation singularity

QUEST's Eq. Q3 and OLAE's Gibbs vector both fail at a rotation of exactly π.
The method of sequential rotations (Shuster & Oh 1981) cures both. Attitude
error in rad, four well-separated observations, noise free:

| π − θ | QUEST, no sequential | QUEST, sequential | OLAE, no sequential | OLAE, sequential |
|---:|---:|---:|---:|---:|
| 1e-02 | 1.189e-14 | 2.139e-17 | 2.167e-14 | 1.516e-17 |
| 1e-04 | 1.537e-12 | 1.561e-17 | 1.918e-12 | 1.686e-17 |
| 1e-06 | 3.904e-10 | 2.532e-17 | 1.506e-10 | 1.477e-17 |
| 1e-08 | 2.291e-08 | 2.652e-17 | RuntimeError | 2.555e-17 |
| 1e-10 | 1.146e-06 | 8.768e-18 | RuntimeError | 2.244e-17 |
| 0 | RuntimeError | 8.768e-18 | RuntimeError | 1.901e-17 |

Worst error **with** sequential rotation: **2.652e-17 rad**, tolerance 1e-12.
Worst error **without**: **1.146e-06 rad**, growing as `1/(π − θ)`.
`sequential_rotation=True` is the default for both methods.

### 4.5 What SciPy does on the same inputs

| Input | `align_vectors` error | Warning raised |
|---|---|---|
| 0.05° apart, noise free | 1.040e-13 rad | none |
| Exactly parallel | **1.176 rad** | `Optimal rotation is not uniquely or poorly defined for the given sets of vectors.` |

SciPy warns on exactly parallel input and returns a value; it does not warn at
0.05°. WahbaKit raises in both cases. Neither behaviour is wrong — they are
different defaults — and this difference, not the covariance, is the one the
README claims.

---

## 5. Checks that failed on their first formulation

Both are recorded because the corrections changed a *reference*, not a
tolerance, and a reader is entitled to see that the tolerance was not moved.

**5.1 `λ_min` against `(1 − |cos η|)/2` at η = 0.01°.** The first version of
§4.1 gated a *relative* deviation at 1e-10 and measured 7.3e-09, i.e. FAIL. Two
things were wrong with the check, neither in the library. First, evaluating
`(1 − cos η)/2` in floating point at η = 1.7e-4 rad loses nine significant
digits by cancellation; the algebraically identical `sin²(η/2)` does not.
Second, `λ_min` comes from `eigvalsh` on a matrix of trace 2, so its error is
**absolute** (about `eps × 2 = 4e-16`), not relative: a `λ_min` of 7.6e-09
carries only about eight correct relative digits no matter how it is computed.
The check is now on the absolute deviation (**1.943e-16**, tolerance 1e-13) and
the relative deviation is reported alongside. The default gate sits at 1e-6, a
hundred times above the 1e-8 relative-accuracy floor, so nothing about the gate
depends on this.

**5.2 `P_TRIAD − P_opt ⪰ 0` gated at 1e-18 absolute.** Measured −2.249e-16,
i.e. FAIL. The bound is exact in exact arithmetic, but Eq. V1 inverts a Fisher
matrix whose condition number reaches 1.0e+04 when σ₁ and σ₂ differ by 100×, so
the numerically attainable bound on a zero eigenvalue is `eps × cond ≈ 2e-12`,
not zero. The check is now relative to `max|P_TRIAD|` with a 1e-9 gate; the
worst value is 3.735e-12, and the condition number is printed on every row so
the reader can see where the round-off comes from.

---

## 6. What this validation does **not** establish

* Level 1 (Educational). Every reference here is analytic, hand-derived, or an
  independent implementation of the same mathematics (SciPy). **Nothing is
  compared against a real star tracker, a real sun sensor, a flight data set, or
  an independently published numerical example.**
* The Monte Carlo comparisons use exactly the measurement model of Eq. O1.
  They confirm that the covariance formulae are the right consequence of that
  model; they say nothing about whether that model describes any particular
  sensor. Real star trackers have along-boresight roll error, field-dependent
  bias, and non-Gaussian tails, none of which are modelled.
* Reference-vector error is assumed zero throughout. A real catalogue has
  position and aberration errors that add to the attitude covariance and are not
  in Eq. V1.
* The TRIAD covariance of Eq. V2 was derived for this package by first-order
  propagation through Eq. T1; the same result is in Shuster & Oh (1981) and
  Markley & Crassidis (2014). The evidence that it is right is the Monte Carlo
  agreement in §3.2, not the citation.
* No timing or throughput claim is made. QUEST is described as avoiding an
  eigendecomposition, which is a statement about the algorithm, not a measured
  speedup; this implementation evaluates Eq. Q3 four times for the sequential
  rotation and is not optimised.
