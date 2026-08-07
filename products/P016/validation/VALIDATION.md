# ZernKit — Validation evidence (Level 1, Educational)

**Product:** P016 ZernKit · **Version:** 0.1.0 · **Date of run:** 2026-08-07
**Environment:** Python 3.11, NumPy, SciPy, 2 CPU cores. Every number below was
produced by running the scripts in this directory in the session that wrote this
file; the raw stdout is committed alongside each script.

| # | Check | Script | Raw output | Result |
|---|---|---|---|---|
| 1 | Orthonormality on the unit disc vs the analytic Kronecker delta | `validate_orthonormality.py` | `orthonormality_output.txt` | PASS |
| 2 | Low-order Zernikes vs hand-evaluated closed forms; index conventions | `validate_closed_forms.py` | `closed_forms_output.txt` | PASS |
| 3 | Analytic gradients vs high-accuracy finite differences | `validate_gradients.py` | `gradients_output.txt` | PASS |
| 4 | Noll residual variances vs published values (Noll 1976) | `validate_noll_variance.py` | `noll_variance_output.txt` | PASS |

Reproduce with, from `products/P016/`:

```bash
python validation/validate_orthonormality.py
python validation/validate_closed_forms.py
python validation/validate_gradients.py
python validation/validate_noll_variance.py
```

Total runtime of all four scripts: under 30 s on the 2-core build machine.

---

## 1. Orthonormality on the unit disc

**Reference.** The analytic orthonormality relation, Noll (1976), *JOSA* 66(3),
207–211, Eq. (3):

```
(1/pi) * int_0^{2pi} int_0^1 Z_i(rho,theta) Z_j(rho,theta) rho drho dtheta = delta_ij
```

**Method.** Modes Noll `j = 1..36` (radial orders 0–7). Two independent
quadratures.

**A — Gauss-Legendre in ρ × uniform trapezoid in θ.** The integrand is a
polynomial in ρ and a trigonometric polynomial in θ, so both rules are exact
to round-off once there are enough nodes. This measures the *library*.

| n_ρ | n_θ | max &#124;G−I&#124; on the diagonal | max &#124;G&#124; off-diagonal |
|---:|---:|---:|---:|
| 20 | 64 | 1.266e-14 | 1.077e-14 |
| 40 | 128 | 2.442e-14 | 2.309e-14 |
| 80 | 256 | **2.109e-15** | **1.091e-15** |
| 120 | 512 | 5.218e-14 | 5.115e-14 |

**Worst deviation from the identity: 5.218e-14** at the finest rule, and
2.109e-15 at the best-conditioned rule (80 × 256). Tolerance 1e-12 → **PASS**.
The mild growth from 80×256 to 120×512 is accumulation of floating-point
round-off across ~61 000 quadrature points, not a modelling error.

Sample diagonals at 120 × 512 (should be exactly 1):

```
  j=  1  (n=0, m=+0)  piston                         <Z,Z> = 1.000000000000001
  j=  4  (n=2, m=+0)  defocus                        <Z,Z> = 1.000000000000014
  j= 11  (n=4, m=+0)  primary spherical              <Z,Z> = 1.000000000000030
  j= 36  (n=7, m=+7)  Z(7, 7)                        <Z,Z> = 1.000000000000050
```

**B — Uniform Cartesian pupil grid with a circular mask.** This is what a real
sampled wavefront looks like, and it is reported so nobody confuses A with B.

| n_pix | points in pupil | max &#124;G−I&#124; diagonal | max &#124;G&#124; off-diagonal |
|---:|---:|---:|---:|
| 32 | 740 | 1.637e-01 | 1.702e-01 |
| 64 | 3096 | 7.911e-02 | 5.273e-02 |
| 128 | 12644 | 2.527e-02 | 2.319e-02 |
| 256 | 51040 | 8.058e-03 | 7.519e-03 |
| 512 | 205012 | 3.328e-03 | 3.117e-03 |

A masked square grid is a first-order-accurate quadrature of a disc (its
boundary is jagged), so orthogonality on sampled data is only approximate and
converges like `1/n_pix`. This is the reason `zernkit.fitting` uses least
squares rather than projection integrals — a projection would inherit these
errors directly.

---

## 2. Low-order Zernikes against closed forms, by hand

**Reference.** The closed forms as printed by Noll (1976), *JOSA* 66(3),
207–211. Every "hand" value below was worked out by hand and typed into
`validate_closed_forms.py` as a literal before the library was called.

**Evaluation point:** ρ = 0.5, θ = 30° = 0.523598775598299 rad.
Constants used: √3 = 1.7320508075688772, √5 = 2.23606797749979,
√6 = 2.449489742783178, √8 = 2.8284271247461903, √10 = 3.1622776601683795,
cos 30° = 0.8660254037844386, sin 30° = 0.5, cos 60° = 0.5,
sin 60° = 0.8660254037844386.

### The arithmetic

**Z₁ (piston), (n,m) = (0,0):** `Z = 1` → **1.000000000000000**

**Z₂ (x tilt), (1,+1):** `Z = 2ρ cos θ = 2(0.5)(0.8660254037844386)`
= `1 × 0.8660254037844386` → **0.866025403784439**

**Z₃ (y tilt), (1,−1):** `Z = 2ρ sin θ = 2(0.5)(0.5)` = `1 × 0.5` → **0.500000000000000**

**Z₄ (defocus), (2,0):** `Z = √3 (2ρ² − 1) = √3 (2 × 0.25 − 1) = √3 (0.5 − 1) = √3 × (−0.5)`
= `1.7320508075688772 × (−0.5)` → **−0.866025403784439**

**Z₅ (oblique astigmatism), (2,−2):** `Z = √6 ρ² sin 2θ = √6 (0.25) sin 60°`
= `2.449489742783178 × 0.25 = 0.6123724356957945`, then
`0.6123724356957945 × 0.8660254037844386` → **0.530330085889911**

**Z₆ (vertical astigmatism), (2,+2):** `Z = √6 ρ² cos 2θ = 0.6123724356957945 × cos 60°`
= `0.6123724356957945 × 0.5` → **0.306186217847897**

**Z₇ (vertical coma), (3,−1):** `Z = √8 (3ρ³ − 2ρ) sin θ`.
`3(0.125) − 2(0.5) = 0.375 − 1.0 = −0.625`;
`√8 × (−0.625) = 2.8284271247461903 × (−0.625) = −1.7677669529663689`;
`× sin 30° = × 0.5` → **−0.883883476483184**

**Z₈ (horizontal coma), (3,+1):** same radial factor `−1.7677669529663689`,
`× cos 30° = × 0.8660254037844386` → **−1.530931089239486**

**Z₉ (oblique trefoil), (3,−3):** `Z = √8 ρ³ sin 3θ = √8 (0.125) sin 90°`
= `0.35355339059327373 × 1` → **0.353553390593274**

**Z₁₀ (vertical trefoil), (3,+3):** `Z = √8 ρ³ cos 3θ = 0.35355339059327373 × cos 90° = 0`
→ **0.000000000000000**

**Z₁₁ (primary spherical), (4,0):** `Z = √5 (6ρ⁴ − 6ρ² + 1)`.
`6(0.0625) − 6(0.25) + 1 = 0.375 − 1.5 + 1 = −0.125`;
`√5 × (−0.125) = 2.23606797749979 × (−0.125)` → **−0.279508497187474**

**Z₁₂ (vertical secondary astigmatism), (4,+2):** `Z = √10 (4ρ⁴ − 3ρ²) cos 2θ`.
`4(0.0625) − 3(0.25) = 0.25 − 0.75 = −0.5`;
`√10 × (−0.5) × cos 60° = 3.1622776601683795 × (−0.5) × 0.5` → **−0.790569415042095**

### Result

| j | mode | hand | library | &#124;diff&#124; |
|---:|---|---:|---:|---:|
| 1 | piston | 1.000000000000000 | 1.000000000000000 | 0.00e+00 |
| 2 | horizontal tilt (x) | 0.866025403784439 | 0.866025403784439 | 0.00e+00 |
| 3 | vertical tilt (y) | 0.500000000000000 | 0.500000000000000 | 5.55e-17 |
| 4 | defocus | −0.866025403784439 | −0.866025403784439 | 0.00e+00 |
| 5 | oblique astigmatism | 0.530330085889911 | 0.530330085889911 | 0.00e+00 |
| 6 | vertical astigmatism | 0.306186217847897 | 0.306186217847897 | 1.11e-16 |
| 7 | vertical coma | −0.883883476483184 | −0.883883476483184 | 1.11e-16 |
| 8 | horizontal coma | −1.530931089239486 | −1.530931089239487 | 2.22e-16 |
| 9 | oblique trefoil | 0.353553390593274 | 0.353553390593274 | 5.55e-17 |
| 10 | vertical trefoil | 0.000000000000000 | 0.000000000000000 | 9.24e-33 |
| 11 | primary spherical | −0.279508497187474 | −0.279508497187474 | 0.00e+00 |
| 12 | vertical secondary astigmatism | −0.790569415042095 | −0.790569415042095 | 2.22e-16 |

**Worst |library − hand| = 2.220e-16** (one unit in the last place). Tolerance
1e-15 → **PASS**.

### Auxiliary identities

* `R_n^m(1) = 1` for all 231 legal `(n, m)` up to `n = 20`: worst
  `|R − 1| = 0.000e+00`. Tolerance 1e-8 → **PASS**.
* Noll index → `(n, m)` reproduces Noll's own listing of Z₁…Z₁₅ exactly
  (all 15 pairs) → **PASS**.
* OSA/ANSI closed form `j = (n(n+2) + m)/2` verified against the implementation
  for all 496 pairs with `n ≤ 30` → **PASS**.

The index correspondence, which is the failure mode this product exists to
prevent:

```
 Noll j   expected (n,m)          library   OSA j  name
      1           (0, 0)           (0, 0)       0  piston
      2           (1, 1)           (1, 1)       2  horizontal tilt (x)
      3          (1, -1)          (1, -1)       1  vertical tilt (y)
      4           (2, 0)           (2, 0)       4  defocus
      5          (2, -2)          (2, -2)       3  oblique astigmatism
      6           (2, 2)           (2, 2)       5  vertical astigmatism
      7          (3, -1)          (3, -1)       7  vertical coma
      8           (3, 1)           (3, 1)       8  horizontal coma
      9          (3, -3)          (3, -3)       6  oblique trefoil
     10           (3, 3)           (3, 3)       9  vertical trefoil
     11           (4, 0)           (4, 0)      12  primary spherical
     12           (4, 2)           (4, 2)      13  vertical secondary astigmatism
     13          (4, -2)          (4, -2)      11  oblique secondary astigmatism
     14           (4, 4)           (4, 4)      14  vertical quadrafoil
     15          (4, -4)          (4, -4)      10  oblique quadrafoil
```

Note that Noll 2 ↔ OSA 2 and Noll 3 ↔ OSA 1: the two conventions already
disagree at tip/tilt, and the gap reaches 5 places by `j = 15`.

---

## 3. Analytic gradients vs high-accuracy finite differences

**Reference.** Richardson-extrapolated central differences, base `h = 1e-2`,
two extrapolation steps giving an `O(h⁶)` reference (~1e-12 relative for these
smooth polynomials, without the round-off blow-up of a single tiny `h`).

**Method.** 400 points drawn uniformly on the disc `ρ ≤ 0.9` with seed
20260807; all 66 modes with `n ≤ 10`. Errors are scaled by
`max(1, max|dZ/dx|, max|dZ/dy|)` so high-order modes with large slopes are not
flattered.

Representative per-mode results (orders 0–4):

```
 Noll j     (n, m)   max|dZ/dx err|   max|dZ/dy err|  mode
      1     (0, 0)        0.000e+00        0.000e+00  piston
      2     (1, 1)        9.881e-14        1.099e-13  horizontal tilt (x)
      4     (2, 0)        5.290e-14        6.595e-14  defocus
      7    (3, -1)        3.804e-14        5.060e-14  vertical coma
     11     (4, 0)        6.781e-14        5.937e-14  primary spherical
     14     (4, 4)        9.211e-14        6.622e-14  vertical quadrafoil
```

**Worst scaled deviation over all 66 modes: 2.884e-11.** Tolerance 1e-9 →
**PASS**. (The worst case is a high radial order where the finite-difference
reference itself is limited by round-off, not the analytic gradient.)

**Closed-form hand checks** (exact, no finite differences), at `(x, y) = (0.37, −0.21)`:

| check | library | exact | diff |
|---|---:|---:|---:|
| j=2 dZ/dx (exact 2) | 2.000000000000000 | 2.000000000000000 | 0.00e+00 |
| j=2 dZ/dy (exact 0) | 0.000000000000000 | 0.000000000000000 | 0.00e+00 |
| j=4 dZ/dx (exact 4√3·0.37) | 2.563435195201938 | 2.563435195201938 | 0.00e+00 |
| j=4 dZ/dy (exact 4√3·(−0.21)) | −1.454922678357857 | −1.454922678357857 | 2.22e-16 |
| j=6 dZ/dx (exact 2√6·0.37) | 1.812622409659552 | 1.812622409659552 | 0.00e+00 |
| j=6 dZ/dy (exact −2√6·(−0.21)) | 1.028785691968935 | 1.028785691968935 | 2.22e-16 |
| j=7 dZ/dx at origin (exact 0) | 0.000000000000000 | 0.000000000000000 | 0.00e+00 |
| j=7 dZ/dy at origin (exact −2√8) | −5.656854249492381 | −5.656854249492381 | 0.00e+00 |

Working for the last two: `Z₇ = √8 (3ρ³ − 2ρ) sin θ = √8 (3ρ² − 2) y`, so at
the origin `dZ₇/dy = −2√8 = −5.656854249492381` and `dZ₇/dx = 0`. This is the
case that would produce `nan` in a naive `1/ρ` implementation.

**Worst |library − exact| over the eight hand checks: 2.220e-16.** Tolerance
1e-15 → **PASS**.

**Non-finite check:** 0 of the 66 modes produce `nan` or `inf` at `ρ = 0` →
**PASS**. The `1/ρ` factor in the chain rule is cancelled at the *coefficient*
level (`R_n^m(ρ)/ρ` is itself a polynomial when `|m| ≥ 1`), not by an epsilon.

---

## 4. Noll residual variance coefficients vs published values

**Reference.** R. J. Noll, "Zernike polynomials and atmospheric turbulence",
*Journal of the Optical Society of America* **66**(3), 207–211 (1976): the
residual mean-square wavefront error `Δ_J` remaining after the first `J`
Zernike terms are removed, in units of `(D/r₀)^(5/3)`. The 21 tabulated values
are reproduced verbatim in `zernkit.NOLL_TABLE_IV` and are **never used in any
computation** — the computed column comes from the analytic coefficient
variances. *(No page or table number beyond the paper's own labelling is
asserted here.)*

**Computed model.** Projecting the Kolmogorov phase spectrum onto the Zernike
basis and evaluating the resulting Bessel integral with Weber–Schafheitlin
(Gradshteyn & Ryzhik 6.574.2) gives a coefficient variance that depends only on
the radial degree:

```
<a_j^2> = 8 C_psd pi^(8/3) (n+1) (D/r0)^(5/3)
          * Gamma(14/3) Gamma(n - 5/6) / [2^(14/3) Gamma(17/6)^2 Gamma(n + 23/6)]
```

Two spectral constants are compared:
`C_psd = 0.023` (Noll's Eq. 4, rounded to two significant figures) and
`C_psd = 0.490/(2π)^(5/3) = 0.0229032`, the unrounded equivalent of the
standard phase PSD `Φ_φ(κ) = 0.490 r₀^(−5/3) κ^(−11/3)` (Roddier 1981,
*Progress in Optics* XIX; Hardy 1998, *Adaptive Optics for Astronomical
Telescopes*).

| J | published | computed (0.023) | rel | computed (0.02290) | rel |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.0299 | 1.037130 | +0.70 % | 1.032765 | +0.28 % |
| 2 | 0.5820 | 0.586204 | +0.72 % | 0.583737 | +0.30 % |
| 3 | 0.1340 | 0.135278 | +0.95 % | 0.134709 | +0.53 % |
| 4 | 0.1110 | 0.111954 | +0.86 % | 0.111483 | +0.44 % |
| 5 | 0.0880 | 0.088630 | +0.72 % | 0.088257 | +0.29 % |
| 6 | 0.0648 | 0.065307 | +0.78 % | 0.065032 | +0.36 % |
| 7 | 0.0587 | 0.059087 | +0.66 % | 0.058838 | +0.24 % |
| 8 | 0.0525 | 0.052867 | +0.70 % | 0.052645 | +0.28 % |
| 9 | 0.0463 | 0.046648 | +0.75 % | 0.046451 | +0.33 % |
| 10 | 0.0401 | 0.040428 | +0.82 % | 0.040258 | +0.39 % |
| 11 | 0.0377 | 0.037963 | +0.70 % | 0.037803 | +0.27 % |
| 15 | 0.0279 | 0.028102 | +0.73 % | 0.027984 | +0.30 % |
| 21 | 0.0208 | 0.020927 | +0.61 % | 0.020839 | +0.19 % |

(Full 21-row table in `noll_variance_output.txt`.)

**Worst relative deviation, `C_psd = 0.023`: 0.954 %** (at J = 3).
**Worst relative deviation, `C_psd = 0.0229032`: 0.529 %** (also J = 3).
Tolerance 1 % → **PASS**.

### Per-mode variances against differences of the published table

Because the variance depends only on `n`, consecutive `Δ_J` differences divided
by the number of modes in that order must reproduce `<a_j²>`:

| n | modes | computed | from table | rel | from |
|---:|---:|---:|---:|---:|---|
| 1 | 2 | 0.449028 | 0.447950 | +0.24 % | (Δ₁ − Δ₃)/2 |
| 2 | 3 | 0.023226 | 0.023067 | +0.69 % | (Δ₃ − Δ₆)/3 |
| 3 | 4 | 0.006193 | 0.006175 | +0.30 % | (Δ₆ − Δ₁₀)/4 |
| 4 | 5 | 0.002455 | 0.002440 | +0.60 % | (Δ₁₀ − Δ₁₅)/5 |
| 5 | 6 | 0.001191 | 0.001183 | +0.63 % | (Δ₁₅ − Δ₂₁)/6 |

That the *differences* of the published table are flat within each radial order
independently confirms the Noll index → `(n, m)` map: a mis-ordered index would
break this pattern immediately.

### Independent cross-check of Δ₁ (no Zernike series involved)

For a stationary phase screen the aperture-averaged, piston-removed variance is
`σ² = ½ ⟨D_φ(|r₁ − r₂|)⟩` with both points uniform on the pupil. Using the
Kolmogorov structure function `D_φ(r) = 6.883877 (r/r₀)^(5/3)` (Fried 1965;
the coefficient is `2(24/5 Γ(6/5))^(5/6)`) and the known distance density for
two points in a disc gives a one-dimensional quadrature:

```
  structure-function integral      : 1.032422 (D/r0)^(5/3)
  Zernike series, C_psd = 0.022904 : 1.032765
  Zernike series, C_psd = 0.023    : 1.037130
  published Noll Delta_1           : 1.029900
```

**|series − structure function| / structure function = 0.033 %.** Tolerance
0.5 % → **PASS**.

**Honest reading of this result.** Two entirely independent routes — a
Zernike-series sum and a real-space structure-function integral — agree to
0.03 % and both land ~0.3 % *above* Noll's published 1.0299. The residual gap
is therefore in the rounding of Noll's published constants, not in this
implementation. We do **not** rescale anything to close it, and the library
default remains Noll's `C_psd = 0.023` so results are directly comparable with
his table.

### Large-J asymptote

`Δ_J ≈ 0.2944 J^(−√3/2) (D/r₀)^(5/3)` (Noll 1976):

| J | published | asymptotic | rel |
|---:|---:|---:|---:|
| 10 | 0.0401 | 0.040079 | −0.05 % |
| 15 | 0.0279 | 0.028211 | +1.11 % |
| 20 | 0.0220 | 0.021989 | −0.05 % |
| 21 | 0.0208 | 0.021080 | +1.34 % |

### Numerical convergence of the residual sum

`Δ_5` with `C_psd = 0.023`, versus a reference with `n_max = 10⁶`:

| n_max | Δ₅ | delta vs 10⁶ |
|---:|---:|---:|
| 100 | 0.088424149174 | −2.061e-04 |
| 1 000 | 0.088625749651 | −4.541e-06 |
| 20 000 | 0.088630259445 | −3.084e-08 |
| 200 000 (default) | 0.088630289666 | −6.200e-10 |

The default cutoff contributes a truncation error of 6.2e-10 `(D/r₀)^(5/3)`,
four orders of magnitude below the discrepancy against the published table.

---

## 5. End-to-end check embedded in the example

`examples/wavefront_fit.py` synthesises a Kolmogorov wavefront from Noll modes
`j = 2..120` at `D/r₀ = 8`, fits the first 21 modes on 28 600 pupil samples,
and compares the residual variance with the analytic `Δ₂₁`:

```
residual RMS            : 0.5769 rad
this realisation        : 0.3328 rad^2
ensemble mean (exact)   : 0.5100 rad^2
analytic Noll Delta_21  : 0.6697 rad^2
tail beyond j=120       : 0.1517 rad^2
Delta_21 minus tail     : 0.5180 rad^2
condition number        : 1.0135
```

The ensemble mean is computed exactly (not by Monte Carlo) by projecting each
truth column out of the fit basis. **0.5100 vs 0.5180 rad² = 1.5 % agreement**,
which exercises indexing, normalisation, fitting and the turbulence statistics
together. A single realisation scatters widely about the ensemble mean (0.333
here), which is why the ensemble quantity is the one compared.

---

## Test suite

`python -m pytest tests/ -q` from `products/P016/`: **158 passed, 0 failed,
0 skipped** (runtime ~16 s, includes Hypothesis property tests). `ruff check
src/ tests/`: clean.

Property-based tests (Hypothesis) cover:

* Noll ↔ `(n, m)` round-trip for `j = 1..20 000`;
* OSA/ANSI ↔ `(n, m)` round-trip for `j = 0..20 000`;
* Noll → OSA → Noll and OSA → Noll → OSA round-trips;
* both conventions naming the same physical `(n, m)`;
* Noll's parity rule (even `j` ⇒ cosine) holding for arbitrary `j`;
* radial-polynomial parity — `R_n^m` contains only powers with the parity of
  `n`, and rejects `n − |m|` odd, for which `R_n^m ≡ 0`;
* `R_n^m(1) = 1`;
* `|R_n^m(ρ)| ≤ 1` on the disc;
* analytic gradient vs central difference at random points;
* noise-free fitting recovering injected coefficient vectors and isolating a
  single injected mode.

## Known limitations of this validation

1. **Level 1 (Educational).** All evidence is analytic, hand-calculated or
   internal-consistency. Nothing here is compared against a measured wavefront,
   a physical Shack-Hartmann sensor, or an independent third-party code.
2. The comparison with Noll (1976) is against values transcribed from the
   published paper. The transcription itself is a single point of failure; it is
   partly self-checked by the flatness of the per-order differences (§4) but not
   independently verified against the original document in this session.
3. Orthonormality was checked to radial order 7 (`j ≤ 36`). High-order radial
   polynomials evaluated from the explicit factorial sum lose precision as `n`
   grows; the worst `|R_n^m(1) − 1|` was 0 up to `n = 20`, but no claim is made
   beyond `n = 20`.
4. The Kolmogorov statistics assume an infinite outer scale and an unobscured
   circular pupil. Finite outer scale (von Kármán) reduces the low-order
   variances substantially and is not modelled; annular pupils need Mahajan's
   annular polynomials (JOSA **71**, 75–85, 1981), which are not implemented.
5. The fitting validation uses noise-free data. No noise-propagation or
   error-budget analysis for the least-squares estimator is provided.
