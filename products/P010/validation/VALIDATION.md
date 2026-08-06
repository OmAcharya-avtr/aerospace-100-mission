# BERBench 0.1.0 — Validation evidence (Level 2, Research)

All numbers below were produced by running the scripts in this directory in
the build session on 2026-08-01 (Python 3.11.15, numpy 2.4.4, scipy 1.17.1,
2 CPU cores). Every script is rerunnable from this directory with
`python3 <script>.py`; each writes its raw output next to itself.

| Script | Raw output | Wall time |
|---|---|---|
| `mc_vs_analytic.py` | `mc_vs_analytic_results.md` | 8.8 s |
| `bpsk_textbook.py` | `bpsk_textbook_output.txt` | < 2 s |
| `ppm_bound_vs_exact.py` | `ppm_bound_vs_exact_output.txt` | ~1 s |
| `lognormal_gh_vs_quad.py` | `lognormal_gh_vs_quad_output.txt` | ~1 s |
| `ci_coverage.py` | `ci_coverage_output.txt` | 4.6 s |

Total Monte Carlo wall time across all validation scripts: **~15 s**, well
inside the 3-minute compute budget.

## 1. Monte Carlo vs analytic across an SNR sweep

`mc_vs_analytic.py` — 10 modulation/channel cases, 54 (SNR, case) points,
each point sized for >= ~150 expected errors, seeded (base 20260801),
Wilson 95% CI on the MC estimate.

**Result: 49/54 points (90.7%) contain the analytic BER inside the 95% CI.**
Expected coverage for a correct implementation is ~95%; the binomial
2-sigma acceptance band for 54 trials is 48–54 points, so the run is
consistent with correctness. The 5 misses are mixed-sign (both above and
below), all within ~2.5 sigma, i.e. statistical fluctuation, not bias. A
repeat with an independent seed base (987654) gave 51/54 (94.4%). Full
per-point table with error counts and intervals:
`mc_vs_analytic_results.md`.

Representative rows (from the saved table):

| case | Eb/N0 | BER analytic | BER MC | errors | Wilson 95% CI |
|---|---|---|---|---|---|
| bpsk [awgn] | 8 dB | 1.9091e-04 | 1.8327e-04 | 144 | [1.5568e-04, 2.1575e-04] |
| ook [awgn] | 12 dB | 3.4303e-05 | 3.4303e-05 | 150 | [2.9235e-05, 4.0249e-05] |
| ppm M=4 [lognormal, s2=0.2] | 12 dB | 2.0697e-03 | 2.1525e-03 | 156 | [1.8404e-03, 2.5173e-03] |

## 2. BPSK against textbook values

`bpsk_textbook.py`:

- Pb at Eb/N0 = 9.6 dB: **9.736e-06** vs the classic textbook benchmark
  "~1e-5 at 9.6 dB" (Proakis & Salehi 2008, Fig. 4.3-1 region; Sklar 2001,
  Sec. 4.7.1). Inverting our curve: Pb = 1e-5 at **9.5879 dB** — within
  0.02 dB of the quoted 9.6 dB. PASS.
- Pb at 10 dB = 3.872108216e-06, identical (rel diff 0.0) to the independent
  scipy path `norm.sf(sqrt(20))`. PASS.
- Q(1) = 0.1586552539 vs Abramowitz & Stegun Table 26.1 value 0.1586552539.
  PASS.
- OOK(γ) − BPSK(γ/2) identity (exact 3.0103 dB penalty): difference
  0.000e+00. PASS.

## 3. M-PPM: exact expression vs union bound

`ppm_bound_vs_exact.py`:

- M=2 exact expression reduces to the closed form Q(sqrt(Eb/N0)):
  max |difference| = 5.6e-17 over 0–12 dB. PASS.
- Union bound >= exact at every tested (M, SNR), M ∈ {4, 16, 64}. PASS.
- Bound tightens with SNR as documented, e.g. M=16: ratio bound/exact =
  2.0254 at 0 dB → 1.0173 at 8 dB → 1.0000 at 12 dB. PASS.
- Exact expression cross-checked against an independent adaptive-quadrature
  (QUADPACK) evaluation of Proakis Eq. (4.4-17): worst relative difference
  **3.1e-07** (M=64, 0 dB; Gauss-Hermite truncation), machine-precision
  (~1e-15) in the deep tail down to BER 2.9e-21. PASS (< 1e-6).
  Note: the *naive* 1−∫φΦ^(M−1) reference itself loses all precision below
  SER ~1e-12 (catastrophic cancellation); the reference integrates the
  error-form integrand instead. This is why berbench computes the exact
  expression in expm1/log_ndtr form.

## 4. Lognormal fading average: Gauss-Hermite vs adaptive quadrature

`lognormal_gh_vs_quad.py`:

- Parameterisation: for sigma_I^2 ∈ {0.1, 0.3, 0.8}, the implemented
  lognormal has E[I] = 1.000000000000 and Var[I] = sigma_I^2 to 12 decimal
  places. PASS.
- Adaptive-threshold OOK and BPSK fading averages agree with independent
  scipy.integrate.quad to **< 2.0e-10 relative** (8 cases,
  sigma_I^2 ∈ {0.1, 0.3}, 6–12 dB; worst 1.99e-10). PASS.
- Fixed-threshold OOK (step-like integrand): rel. diff 7.9e-13 at 12 dB,
  1.7e-10 at 20 dB, **4.7e-3 at 30 dB** with GH-256. Documented limitation:
  the fixed-threshold analytic average degrades to ~0.5% relative error at
  30 dB; use Monte Carlo there if tighter accuracy is needed. PASS within
  documented limits.

## 5. Confidence-interval coverage (and a defect this process caught)

`ci_coverage.py` — 200 independent seeded MC replicates per case; count how
often the analytic BER falls inside the reported 95% CI:

| case | coverage |
|---|---|
| bpsk awgn 4 dB | 185/200 = 0.925 |
| ook awgn 6 dB | 185/200 = 0.925 |
| ppm M=4 awgn 4 dB | 180/200 = 0.900 |
| ppm M=16 awgn 4 dB | 181/200 = 0.905 |
| ook lognormal s2=0.3, 8 dB | 190/200 = 0.950 |

Nominal 0.95; 2-sigma band for 200 replicates is ±0.031. Binary cases are
consistent with nominal. PPM sits ~0.90–0.93: the interval neglects the
variance of the wrong-bits-per-symbol-error ratio (documented in
`montecarlo.py`), leaving it ~8% too narrow in half-width.

**Honest disclosure:** during development this check exposed a real defect —
the original bit-level Wilson interval for PPM had only **75.7% coverage**
at M=16 (bit errors within a symbol are correlated, violating the binomial
assumption). The interval was rebuilt on symbol-error counts and rescaled,
restoring coverage to ~0.90–0.93 (a 300-replicate spot check during the fix
measured 0.923/0.930/0.927 for M=16 AWGN / M=4 AWGN / M=16 lognormal). The
residual narrowing is documented rather than hidden.

## Known deviations / limitations recorded here

1. PPM CI residual narrowing (~8% half-width; coverage ~0.90–0.93 vs 0.95).
2. Fixed-threshold OOK lognormal analytic average: ~0.5% relative error at
   30 dB (GH-256 vs step-like integrand); accurate to <1e-9 at <= 20 dB.
3. The 54-point MC sweep coverage (90.7%) is at the low edge of, but inside,
   the 2-sigma acceptance band; misses are mixed-sign near-misses.
