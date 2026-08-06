# Validation — linkbudgetx 0.1.0 (Level 1, Educational)

Evidence level: hand-calculated known-answer cases plus a Monte Carlo
cross-check of the first-order uncertainty propagation. All numbers below
were produced by running the committed scripts in this session:

- `validation/known_answer_check.py` → raw output in `known_answer_output.txt`
- `validation/mc_crosscheck.py` → raw output in `mc_crosscheck_output.txt`

Baseline configuration used throughout (also `examples/example.yaml`):
Tx power 20 dBm, wavelength 1550 nm, beam divergence 1.0 mrad **full angle**
(1/e² full angle), range 10 km, Rx aperture diameter 0.1 m, Tx/Rx optics
efficiency 0.8 each, atmospheric attenuation 0.5 dB/km, Rx sensitivity
−40 dBm.

Angle convention: the library input is the FULL divergence angle
θ_full; all Gaussian far-field formulas use the HALF angle
θ_half = θ_full/2 = 0.5 mrad (Saleh & Teich, *Fundamentals of Photonics*,
2nd ed., Ch. 3).

## Case 1 — Flat-top geometric spreading loss (hand calculation)

Uniform disc of half-angle θ_half = 0.5 mrad:

1. Spot radius at R = 10 km: r_spot = θ_half · R = 0.5×10⁻³ · 10⁴ m = **5 m**.
2. Aperture radius a = 0.1/2 = 0.05 m.
3. Capture fraction f = (a/r_spot)² = (0.05/5)² = (0.01)² = **1×10⁻⁴**.
4. Loss L = −10 log₁₀(10⁻⁴) = **40.000000 dB**.

Library result: 40.000000 dB, |diff| = 0.0 (tol 1e−9). **PASS**

## Case 2 — Gaussian geometric spreading loss (hand calculation)

Gaussian aperture-capture formula f = 1 − exp(−2a²/w²)
(Saleh & Teich, 2nd ed., Ch. 3, power through a circular aperture):

1. 1/e² beam radius at R = 10 km: w = θ_half · R = **5 m** (far field).
2. 2a²/w² = 2·(0.05)²/5² = 2·0.0025/25 = 2×10⁻⁴.
3. f = 1 − exp(−2×10⁻⁴) = 1.9998000×10⁻⁴
   (series: 2×10⁻⁴ − (2×10⁻⁴)²/2 = 2×10⁻⁴ − 2×10⁻⁸).
4. L = −10 log₁₀(1.9998000×10⁻⁴) = **36.990134 dB**.

Library: capture fraction 1.9998000×10⁻⁴ (|diff| = 0, tol 1e−12), loss
36.990134 dB (|diff| = 0, tol 1e−9). **PASS**

Sanity check: for a ≪ w the Gaussian fraction is ≈ 2(a/w)², exactly twice
the flat-top (a/w)² — 40.000 − 36.990 = 3.010 dB ≈ 10 log₁₀ 2. Consistent.

## Case 3 — Pointing loss at θ_err = θ_full/2 (hand calculation)

Far-field Gaussian angular intensity I(θ) = I₀ exp(−2θ²/θ_half²), so
L_p(linear) = exp(−2 θ_err²/θ_half²), where θ_half is the 1/e² HALF angle
(this is the classic exp(−2θ_err²/θ_div²) form with θ_div meaning the half
angle; Majumdar & Ricklin, *Free-Space Laser Communications*, Springer 2008).

At θ_err = θ_full/2 = θ_half = 0.5 mrad:

1. 2 θ_err²/θ_half² = 2.
2. L_p(linear) = e⁻².
3. L_p[dB] = −10 log₁₀(e⁻²) = 20 log₁₀(e) = **8.685890 dB**.

Library: 8.685890 dB, |diff| = 0 (tol 1e−9). **PASS**

Edge case 3b: θ_err = 0 → L_p = exp(0) = 1 → **0 dB exactly**.
Library: 0.0 dB, exact. **PASS**

## Case 4 — Full budget (hand-summed)

With pointing_error_rad = 0.25 mrad (= θ_half/2 → 2 θ_err²/θ_half² = 0.5):

| Term | Hand value |
|---|---|
| Tx power | +20.000000 dBm |
| Tx optics, −10 log₁₀ 0.8 | 0.969100 dB |
| Geometric (case 2) | 36.990134 dB |
| Pointing, 5 log₁₀ e | 2.171472 dB |
| Atmospheric, 0.5 dB/km × 10 km | 5.000000 dB |
| Rx optics | 0.969100 dB |
| **Rx power** | **−26.099807 dBm** |
| **Margin vs −40 dBm** | **+13.900193 dB** |

Library: Rx power −26.099807 dBm, margin 13.900193 dB, |diff| = 0
(tol 1e−9). **PASS**

## Monte Carlo cross-check of first-order uncertainty propagation

Inputs (1-sigma): tx_power_dbm 0.5 dB, atmos_attenuation_db_per_km
0.05 dB/km, pointing_error_rad 0.02 mrad. n = 20 000 samples, seed 2026.

**Scenario 1 — linear regime (nominal pointing error 0.25 mrad):**

- First-order sigma (delta method, JCGM 100:2008): **0.7879 dB**
- Monte Carlo std: **0.7850 dB**
- Relative discrepancy: **0.37 %** → PASS (tolerance 5 %)
- MC mean 13.8897 dB vs nominal 13.9002 dB (small quadratic bias, expected).

**Scenario 2 — deliberate breakdown (nominal pointing error 0,
sigma 0.05 mrad):** margin is quadratic in the pointing offset, so the
first derivative at 0 vanishes:

- First-order sigma: **0.0000 dB** (misses the jitter entirely)
- Monte Carlo std: **0.1221 dB**; MC mean shift: **−0.0877 dB**

This disagreement is EXPECTED and documents where the linearity assumption
breaks (see `uncertainty.py` module docstring and README Limitations). It is
a failure mode of the linear method itself, not a library defect; use
`monte_carlo_margin()` in this regime.

## Test suite

`python -m pytest tests/ -q` → 54 passed, 0 failed, 0 skipped (this session).
`ruff check src/ tests/` → clean.

## References

- B. E. A. Saleh & M. C. Teich, *Fundamentals of Photonics*, 2nd ed., Wiley —
  Gaussian beam optics chapter (divergence, aperture transmission).
- A. K. Majumdar & J. C. Ricklin (eds.), *Free-Space Laser Communications:
  Principles and Advances*, Springer, 2008 — FSO link-budget structure and
  pointing-error treatment.
- H. T. Friis, "A Note on a Simple Transmission Formula", Proc. IRE 34, 1946 —
  transmitter/receiver/free-space-loss budget concept adapted to optics.
- I. I. Kim, B. McArthur, E. Korevaar, "Comparison of laser beam propagation
  at 785 nm and 1550 nm in fog and haze for optical wireless communications",
  Proc. SPIE 4214, 2001 — context for user-supplied dB/km attenuation values.
- JCGM 100:2008 (GUM), Sec. 5.1 — first-order (delta-method) uncertainty
  propagation for uncorrelated inputs.

No page numbers are cited because none were verified against physical copies.
