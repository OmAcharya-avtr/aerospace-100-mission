# Changelog

## 0.1.0 — 2026-08-01

Initial release.

- Analytic BER: BPSK Q(sqrt(2·gamma)); OOK with optimal and fixed decision
  thresholds; M-ary PPM exact orthogonal-signalling expression (stable
  expm1/log_ndtr Gauss-Hermite evaluation) and union bound, clearly
  labelled exact vs bound. All expressions cited (Proakis & Salehi 2008;
  Zhu & Kahn 2002; Andrews & Phillips 2005; Popoola & Ghassemlooy 2009).
- Lognormal fading (weak-turbulence FSO): scintillation-index
  parameterisation with E[I] = 1, Gauss-Hermite BER averaging, documented
  sigma_I² < ~1 validity range with UserWarning beyond it; no-CSI fixed
  threshold reproduces the irreducible BER floor.
- Monte Carlo engine: vectorised numpy, batched (2^21 values), seeded
  (PCG64), Wilson score confidence intervals (symbol-level counting for
  PPM), `n_bits_for_target` sample-sizing helper, `max_seconds` hard
  runtime budget with `budget_exhausted` flag.
- API: `analytic_ber(...)` / `mc_ber(...)` returning frozen
  `AnalyticResult` / `MCResult` dataclasses; log-domain Q-function helpers.
- CLI: `python -m berbench sweep --mod ook bpsk ppm --snr 0:20:2
  --channel awgn [--mc] [--png out.png]`.
- Level-2 validation with saved evidence (validation/): 54-point MC-vs-
  analytic sweep, BPSK textbook cross-check, PPM bound-vs-exact tables,
  Gauss-Hermite vs QUADPACK fading-average comparison, 200-replicate CI
  coverage audit.
- 68 pytest tests (known answers, hypothesis property tests, MC statistical
  agreement, input validation, edge cases); ruff-clean.
