# ZernKit

**Status:** TESTING · **Class:** compact · **Validation level:** 1 (Educational) · **AI:** no

## Executive overview

`zernkit` is a small, dependency-light Zernike polynomial toolkit for adaptive
optics and wavefront analysis. It implements **both** common single-index
orderings — Noll (1976) and OSA/ANSI — as exact integer maps with tested
conversion between them, evaluates radial polynomials and full modes in both the
orthonormal and unnormalised conventions, provides closed-form Cartesian
gradients for Shack-Hartmann use, fits sampled wavefronts to coefficients by
least squares with an explicit policy for samples outside the unit disc, and
computes Noll's Kolmogorov residual variance coefficients `Δ_J` from the
analytic coefficient variances (not from a lookup table).

Every convention is stated in the code, in this README, and in the figures.
Index-ordering and normalisation confusion is the exact class of bug this
package exists to prevent.

## Aerospace problem

Wavefront error in an optical system — a laser communications terminal, a
space telescope, an airborne imaging turret — is almost universally reported as
a vector of Zernike coefficients. But "Z11 = 0.08 waves" is meaningless on its
own: it could be primary spherical (Noll) or vertical secondary astigmatism
(OSA/ANSI), and it could be RMS-normalised or peak-normalised, a factor of
√5 apart. Coefficient sets are routinely exchanged between a sensor vendor, a
simulation tool and an analysis script, each with a different default, and the
resulting errors are silent: the numbers still look plausible.

This package makes the convention explicit at every call site, converts between
the conventions exactly, and ships the low-order closed forms and Noll's
turbulence statistics as tested reference points so a mistake shows up as a
failing check rather than as a slightly wrong Strehl ratio.

## Intended users

Students, educators, and engineers doing early adaptive-optics or wavefront
analysis work who need a transparent, auditable Zernike implementation — and
anyone who has to reconcile Zernike coefficients coming from two tools that
disagree about indexing. It is a building block, not an AO simulator: sensor
modelling, phase screens, and closed-loop control are out of scope.

## Engineering theory

### Radial polynomials

Source: M. Born and E. Wolf, *Principles of Optics*, 7th (expanded) ed.,
Cambridge University Press 1999, Sec. 9.2 and Appendix VII. For `n − |m|` even
(the polynomial is identically zero otherwise):

```
R_n^m(rho) = sum_{k=0}^{(n-|m|)/2}   (-1)^k (n-k)!
             ------------------------------------------------------- rho^(n-2k)
             k! ((n+|m|)/2 - k)! ((n-|m|)/2 - k)!
```

*Units:* dimensionless. *Domain:* `0 ≤ ρ ≤ 1` (unit disc; ρ is the pupil radius
normalised to the pupil edge — the physical aperture radius never enters).
*Properties used as tests:* `R_n^m(1) = 1`; only powers of ρ with the parity of
`n` appear; `|R_n^m(ρ)| ≤ 1` on the disc.

### Full modes and the two normalisation conventions

```
unnormalised (Born & Wolf, unit peak):
    Z_n^m = R_n^|m|(rho) * cos(m theta)      m > 0
          = R_n^|m|(rho) * sin(|m| theta)    m < 0
          = R_n^0(rho)                       m = 0

normalised (Noll 1976 Eq. 2; identical in ANSI Z80.28):
    Z_n^m = N_n^m * (unnormalised),   N_n^m = sqrt(2(n+1)) for m != 0
                                      N_n^0 = sqrt(n+1)
```

*Orthonormality* (Noll 1976, Eq. 3), with the **area-normalised weight `1/π`**:

```
(1/pi) int_0^{2pi} int_0^1 Z_i Z_j rho drho dtheta = delta_ij
```

Consequence: in the normalised convention the RMS wavefront over the pupil is
`sqrt(sum_{j != piston} a_j^2)` directly. In the unnormalised convention it is
not, and the two coefficient sets differ mode-by-mode by `N_n^m`.

*Units:* the library is unit-agnostic for the wavefront — coefficients come out
in whatever unit the samples went in (waves, radians, metres). It never
converts. *Assumptions:* circular, unobscured pupil. *Validity:* the unit disc
only. On an annulus the circle polynomials are **not** orthogonal; that case
needs the annular polynomials of V. N. Mahajan, *JOSA* **71**, 75–85 (1981),
which are not implemented.

### Index conventions (state these before quoting any coefficient)

**Noll (1976)**, *JOSA* **66**(3), 207–211 — `j` starts at **1**:

* order `n` occupies `n(n+1)/2 + 1 ≤ j ≤ (n+1)(n+2)/2`, i.e. `n+1` modes;
* within an order `|m|` increases, each `|m| > 0` pair taking two consecutive `j`;
* **even `j` is the cosine (`m > 0`) member, odd `j` the sine (`m < 0`) member.**

**OSA/ANSI** — ANSI Z80.28; equivalently L. N. Thibos, R. A. Applegate,
J. T. Schwiegerling, R. Webb, "Standards for reporting the optical aberrations
of eyes", *J. Refract. Surg.* **18**, S652–S660 (2002) — `j` starts at **0**:

* closed form both ways: `j = (n(n+2) + m)/2`;
* within an order `m` runs `−n, −n+2, …, +n`.

**Sign convention, shared by both:** `m > 0` ⇒ `cos(m θ)`, `m < 0` ⇒
`sin(|m| θ)`, with `θ` counter-clockwise from `+x`. Neither standard fixes the
handedness of *your* pupil coordinates: if the optical train flips the pupil,
every `m < 0` coefficient changes sign.

The two orderings agree only at piston. They are already swapped at tip/tilt
(Noll 2 = OSA 2, Noll 3 = OSA 1) and by `j = 15` the gap is five places:

| Noll j | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (n, m) | (0,0) | (1,+1) | (1,−1) | (2,0) | (2,−2) | (2,+2) | (3,−1) | (3,+1) | (3,−3) | (3,+3) | (4,0) | (4,+2) | (4,−2) | (4,+4) | (4,−4) |
| OSA j | 0 | 2 | 1 | 4 | 3 | 5 | 7 | 8 | 6 | 9 | 12 | 13 | 11 | 14 | 10 |

### Cartesian gradients (Shack-Hartmann slopes)

With `ρ = √(x²+y²)`, `θ = atan2(y, x)`, the chain rule gives

```
dZ/dx = N [ R'(rho) Theta(theta) cos(theta) - (R(rho)/rho) Theta'(theta) sin(theta) ]
dZ/dy = N [ R'(rho) Theta(theta) sin(theta) + (R(rho)/rho) Theta'(theta) cos(theta) ]
```

The `1/ρ` is removable: for `|m| ≥ 1`, `R_n^m(ρ) = O(ρ^|m|)`, so `R/ρ` is itself
a polynomial; the implementation divides the *coefficient array*, so `ρ = 0` is
evaluated exactly with no `nan` and no epsilon. For `m = 0`, `Θ' ≡ 0` and the
term is dropped before it can form.

*Units:* per unit **normalised** pupil radius. For a pupil of radius `R_pupil`
[m] and a wavefront in metres, physical slope in radians = returned value /
`R_pupil`. The library does not do this conversion.

*Assumption in `zernike_slope_matrix`:* point-sampled slopes. A real
Shack-Hartmann measures the subaperture-*averaged* slope; the two agree only
when the mode varies slowly across a subaperture.

### Least-squares fitting

Given samples `w_p` at `(x_p, y_p)`, solve `a = argmin ||A a − w||₂` with
`A[p,k] = Z_k(x_p, y_p)` via SVD (`numpy.linalg.lstsq`).

Orthonormality holds under a *continuous* `1/π` weight. On a finite, non-uniform
or clipped sample set the modes are only approximately orthogonal, so a
projection integral and the least-squares solution differ — see the measured
discretisation error in `validation/VALIDATION.md` §1B (2.5 % off-diagonal
leakage at 128 px across the pupil). Least squares is the honest estimator for
sampled data. `FitResult.condition_number` reports how far the sampling has
pushed the basis from orthogonality.

**Policy for samples outside the unit disc (explicit).** Zernikes are orthogonal
only on `ρ ≤ 1` and grow rapidly outside it, so an out-of-disc sample can
dominate a fit. `fit_wavefront(..., outside=...)` takes:

* `"raise"` **(default)** — `ValueError` naming the offending count and the
  worst radius. Fails loudly.
* `"drop"` — exclude samples with `ρ > 1 + tol`, counted in
  `FitResult.n_dropped`. The right choice for a square array whose corners are
  simply outside the pupil.
* `"extrapolate"` — keep everything and evaluate outside the disc. Provided for
  completeness; the result is **not** an orthogonal decomposition.

`tol` (default `1e-9`) absorbs round-off for points meant to sit exactly on the
rim. Non-finite wavefront values are always rejected, because `lstsq` would
otherwise return an all-`nan` solution silently.

### Kolmogorov coefficient statistics and Noll residual variances

Source: R. J. Noll, *JOSA* **66**(3), 207–211 (1976). Kolmogorov turbulence
(infinite outer scale), circular unobscured pupil of diameter `D`, phase PSD in
cyclic spatial frequency `Φ(k) = C_psd r₀^(−5/3) k^(−11/3)` [rad² m²]. Noll's
Eq. (4) quotes `C_psd = 0.023`; the unrounded equivalent
`0.490/(2π)^(5/3) = 0.0229032` follows from the standard angular-frequency form
`Φ_φ(κ) = 0.490 r₀^(−5/3) κ^(−11/3)` (F. Roddier, *Progress in Optics* **XIX**,
281–376, 1981; J. W. Hardy, *Adaptive Optics for Astronomical Telescopes*, OUP
1998). Both are exposed; the default is Noll's value.

Projecting the spectrum onto the Zernike basis and evaluating the Bessel
integral with Weber–Schafheitlin (Gradshteyn & Ryzhik 6.574.2, `λ = 14/3`,
`ν = n+1`) gives a variance that depends **only** on the radial degree:

```
<a_j^2> = 8 C_psd pi^(8/3) (n+1) (D/r0)^(5/3)
          * Gamma(14/3) Gamma(n - 5/6) / [2^(14/3) Gamma(17/6)^2 Gamma(n + 23/6)]
```

and `Δ_J = Σ_{j>J} <a_j²>`, summed here over radial orders to `n_max = 200 000`
(truncation < 1e-9 `(D/r₀)^(5/3)`).

*Units:* rad², for `r₀` the Fried parameter in the same length unit as `D` and
at the same wavelength. *Validity:* infinite outer scale; a finite outer scale
(von Kármán) reduces the low-order variances substantially and is not modelled.
Piston is excluded — its Kolmogorov variance diverges, which is why Noll's
series starts at `j = 2`.

`zernkit.NOLL_TABLE_IV` holds Noll's published `Δ_J` for `J = 1..21` as
reference data only; it is never used in a computation. Measured agreement:
worst 0.954 % with `C_psd = 0.023`, 0.529 % with the unrounded constant — see
Validation.

## Architecture

```
src/zernkit/
├── indexing.py     # Noll <-> (n,m) <-> OSA/ANSI, pure integer arithmetic
├── polynomials.py  # radial coefficients, R_n^m, Z_n^m, normalisation, disc grid
├── gradients.py    # analytic dZ/dx, dZ/dy; Shack-Hartmann slope matrix
├── fitting.py      # design matrix, least-squares fit, FitResult, disc policy
├── statistics.py   # Kolmogorov <a_j^2>, Noll Delta_J, published reference table
├── cli.py          # argparse CLI
└── __main__.py     # python -m zernkit
```

Runtime dependencies: NumPy (everywhere) and SciPy (`gamma`/`gammaln` in
`statistics.py` only). Matplotlib is needed for the examples, not for the
library. No cross-product imports.

## Installation

From the product root (`products/P016/`):

```bash
pip install .            # or: pip install -e ".[dev]"
```

Or without installing: `PYTHONPATH=src python -c "import zernkit"`.

## Quick start

```python
import numpy as np
import zernkit as zk

# 1. Never guess an index again.
zk.noll_to_nm(11)        # (4, 0)  -> primary spherical
zk.noll_to_osa(11)       # 12      -> the SAME mode in OSA/ANSI numbering
zk.osa_to_nm(11)         # (4, -2) -> a DIFFERENT mode with the same number
zk.mode_name(4, 0)       # 'primary spherical'

# 2. Evaluate modes (normalised by default).
x, y, mask = zk.unit_disc_grid(128)
z = zk.zernike_cartesian(4, 0, x[mask], y[mask])          # orthonormal
z_peak = zk.zernike_cartesian(4, 0, x[mask], y[mask], normalized=False)

# 3. Analytic slopes for a Shack-Hartmann interaction matrix.
gx, gy = zk.zernike_gradient_noll(4, x[mask], y[mask])    # defocus: 4*sqrt(3)*x, 4*sqrt(3)*y

# 4. Fit a sampled wavefront (units in = units out).
w = 0.3 * zk.zernike_cartesian(2, 0, x[mask], y[mask])    # 0.3 waves of defocus
fit = zk.fit_wavefront(x[mask], y[mask], w, n_modes=15)   # Noll ordering by default
fit.coefficient(2, 0)          # 0.3
fit.noll_indices               # [1, 2, 3, ..., 15]
fit.osa_indices                # [0, 2, 1, 4, 3, 5, 7, 8, 6, 9, 12, 13, 11, 14, 10]
fit.residual_rms               # ~1e-16
fit.condition_number           # ~1.0 on a well-sampled disc

# 5. Turbulence statistics.
zk.coefficient_variance(2, d_over_r0=10.0)   # <a^2> for any n=2 mode, rad^2
zk.residual_variance(21, d_over_r0=10.0)     # Noll Delta_21, rad^2
zk.NOLL_TABLE_IV[21]                         # 0.0208, published reference
```

A square array whose corners fall outside the pupil:

```python
x, y, _ = zk.unit_disc_grid(64)
zk.fit_wavefront(x, y, wavefront_array, 15)                  # raises: corners outside
zk.fit_wavefront(x, y, wavefront_array, 15, outside="drop")  # explicit and safe
```

## Configuration

No configuration files. Behaviour is set by keyword arguments, all defaulting to
the safest choice:

| Argument | Default | Effect |
|---|---|---|
| `normalized` | `True` | Noll/ANSI orthonormal scaling; `False` gives unit-peak Born & Wolf modes |
| `indexing` | `"noll"` | Mode ordering for `n_modes`; `"osa"`/`"ansi"` for OSA/ANSI |
| `outside` | `"raise"` | Out-of-disc sample policy; `"drop"` or `"extrapolate"` |
| `tol` | `1e-9` | Radial tolerance for "on the rim" |
| `psd_constant` | `0.023` | Kolmogorov PSD constant (Noll Eq. 4); `KOLMOGOROV_PSD_CONSTANT` for the unrounded value |
| `n_max` | `200000` | Radial-order cutoff of the `Δ_J` summation |

CLI:

```bash
python -m zernkit index --noll 8            # -> n=3, m=+1, OSA 8, horizontal coma
python -m zernkit index --osa 11            # -> n=4, m=-2, Noll 13
python -m zernkit index --max-order 4       # full Noll <-> OSA correspondence table
python -m zernkit noll-table --j-max 21     # computed vs published Delta_J
python -m zernkit noll-table --d-over-r0 8  # scaled to D/r0 = 8
```

Exit code 2 with an actionable message on invalid input.

## Examples

Run from the product root; PNGs land in `screenshots/` (Agg backend, no display
needed):

* `python examples/mode_gallery.py` — the first 21 Noll modes, every panel
  labelled with its Noll index, its OSA/ANSI index, `(n, m)` and its traditional
  name, so the reordering between conventions is visible rather than asserted.
  → `screenshots/zernike_mode_gallery.png`
* `python examples/wavefront_fit.py` — a seeded synthetic Kolmogorov wavefront
  (Noll `j = 2..120`, `D/r₀ = 8`) fitted with 21 modes on 28 600 pupil samples:
  truth, fit, residual, injected-vs-recovered coefficient bars, and a residual
  variance check against the analytic `Δ₂₁`.
  → `screenshots/wavefront_fit_residual.png`

Both run in a few seconds.

## Validation

Level 1 (Educational). Full evidence, with the hand arithmetic written out, is
in `validation/VALIDATION.md`; the raw stdout of each script is committed
alongside it. All numbers below come from running those scripts in the build
session.

1. **Orthonormality vs the analytic Kronecker delta.** Modes Noll `j = 1..36`,
   Gauss-Legendre(ρ) × uniform(θ) quadrature. **Worst deviation of the Gram
   matrix from the identity: 5.218e-14** at 120×512 nodes (2.109e-15 at
   80×256). Tolerance 1e-12 → PASS. A masked uniform Cartesian grid — what real
   sampled data looks like — is reported separately and converges only as
   `1/n_pix` (2.5e-2 leakage at 128 px, 3.3e-3 at 512 px).
2. **Low-order closed forms, by hand.** Twelve modes (piston, tip, tilt,
   defocus, both astigmatisms, both comas, both trefoils, primary spherical,
   secondary astigmatism) evaluated by hand at ρ = 0.5, θ = 30° and compared
   with the library. **Worst |library − hand| = 2.220e-16** (1 ulp). Tolerance
   1e-15 → PASS. `R_n^m(1) = 1` for all 231 legal pairs to `n = 20`, worst
   error 0.0. Noll's own listing of Z₁…Z₁₅ reproduced exactly.
3. **Analytic gradients vs high-accuracy finite differences.** Richardson-
   extrapolated central differences (`O(h⁶)`), 400 seeded points on `ρ ≤ 0.9`,
   all 66 modes to `n = 10`. **Worst scaled deviation 2.884e-11**; tolerance
   1e-9 → PASS. Eight closed-form hand checks agree to 2.220e-16, including
   `dZ₇/dy(0,0) = −2√8` — the case a naive `1/ρ` implementation turns into
   `nan`. Zero modes produce `nan`/`inf` at the origin.
4. **Noll residual variances vs published values.** Computed `Δ_J` compared
   with Noll (1976) for `J = 1..21`. **Worst relative deviation 0.954 %** with
   Noll's rounded `C_psd = 0.023`, **0.529 %** with the unrounded
   `0.490/(2π)^(5/3)`. Tolerance 1 % → PASS.
5. **Independent cross-check of Δ₁.** A real-space route that never touches the
   Zernike series — `σ² = ½⟨D_φ(|r₁−r₂|)⟩` over two uniform points in the pupil,
   with `D_φ(r) = 6.883877 (r/r₀)^(5/3)` — gives **1.032422 (D/r₀)^(5/3)** vs
   the Zernike series' **1.032765**, i.e. **0.033 % agreement**. Both land
   ~0.3 % above Noll's published 1.0299; the gap is attributed to rounding in
   the published constants and **is not tuned away**.
6. **End-to-end.** In `examples/wavefront_fit.py`, the exact ensemble-mean
   residual variance after a 21-mode fit is **0.5100 rad²** against
   `Δ₂₁ − (tail beyond j=120)` = **0.5180 rad²**, a 1.5 % agreement that
   exercises indexing, normalisation, fitting and statistics together.

Test suite: **158 passed, 0 failed, 0 skipped**, including Hypothesis
property tests on Noll↔OSA round-trips (`j` to 20 000), radial-polynomial
parity, `R_n^m(1) = 1`, gradient-vs-finite-difference, and noise-free
coefficient recovery.

## Benchmark results

Not a performance product; measured on the 2-core build machine (`timeit`,
28 600 pupil samples = a 192-px grid masked to the unit disc):

| Operation | Time |
|---|---|
| `noll_to_nm` (pure integer arithmetic) | 0.36 µs |
| `zernike_cartesian`, 28 600 points | 1.3 ms |
| `zernike_gradient`, 28 600 points | 5.1 ms |
| `zernike_design_matrix`, 28 600 × 21 | 41 ms |
| `fit_wavefront`, 28 600 × 21 (includes the SVD) | 297 ms |
| `residual_variance(J)`, default `n_max = 200 000` | 19 ms |
| Full test suite (158 tests, incl. Hypothesis) | ~16 s |
| All four validation scripts | < 30 s |

Well inside the 3-minute compute budget; nothing here needs more than one core.

## AI model details

Not applicable — this product contains no AI/ML components.

## Hardware requirements

Any machine running Python ≥ 3.11 with NumPy and SciPy. No GPU. Memory scales
with the number of pupil samples times the number of modes (a 28 600 × 21
design matrix is ~5 MB). Examples additionally need Matplotlib (Agg backend, no
display).

## Limitations

* **Circular, unobscured pupils only.** Zernike circle polynomials are not
  orthogonal on an annulus; a central obscuration requires Mahajan's annular
  polynomials (*JOSA* **71**, 75–85, 1981), which are not implemented. Fitting
  an obscured pupil with this package will silently give a non-orthogonal
  decomposition.
* **Only two index conventions.** Fringe/University-of-Arizona and Wyant
  orderings, which are also common in commercial interferometry software, are
  not implemented.
* **No pupil-coordinate handedness handling.** The library uses `θ`
  counter-clockwise from `+x`; if your optical train flips the pupil you must
  flip the sign of the `m < 0` coefficients yourself.
* **High radial orders lose precision.** Radial coefficients come from the
  explicit alternating factorial sum; cancellation grows with `n`. Verified
  clean to `n = 20` (`|R_n^m(1) − 1| = 0`); no claim is made beyond that. A
  recurrence-based evaluation would be needed for very high orders.
* **Turbulence statistics assume an infinite outer scale** and an unobscured
  circular pupil, and the computed `Δ_J` sit ~0.5–1 % above Noll's published
  table (documented, not tuned). Finite outer scale (von Kármán) is not
  modelled and would reduce low-order variances substantially.
* **Slope model is point-sampled**, not subaperture-averaged; it is an
  approximation to a real Shack-Hartmann measurement.
* **Fitting is unweighted ordinary least squares.** No measurement-noise
  weighting, no regularisation, no noise-propagation analysis. Slope-to-phase
  reconstruction is deliberately out of scope (see P014 WaveLab).
* **Educational validation level.** All evidence is analytic, hand-calculated or
  internally consistent. Nothing is compared against measured wavefronts, a
  physical sensor, or an independent third-party code, and Noll's published
  values were transcribed by hand from the paper.

## Safety statement

This software is educational. It is not flight-qualified, not certified, and not
approved for operational aerospace use.

## Roadmap

* Fringe/Wyant index conventions with the same explicit conversion treatment.
* Mahajan annular polynomials for obscured pupils.
* Recurrence-based radial evaluation for high `n`.
* von Kármán (finite outer scale) coefficient variances alongside Kolmogorov.
* Weighted and regularised fitting with noise propagation to coefficient
  uncertainties.

## License

MIT — see `LICENSE`. Copyright © 2026 OPTIMA Organisation.

## Credits

This is under reserved rights obtained by OPTIMA Organisation.

## Citation

```
OPTIMA Organisation (2026). ZernKit: Zernike polynomial toolkit for adaptive
optics and wavefront analysis (v0.1.0) [Computer software]. Educational
validation level 1.
```
