# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-06

First release. Validation level 2 (research grade). Status: TESTING.

### Added

- `softdecode.channel` - unit-mean lognormal fading (Andrews & Phillips 2005,
  chapter 9) and unit-mean gamma-gamma fading (Al-Habash, Andrews & Phillips,
  *Opt. Eng.* 40(8), 2001), each with density, log-density, sampler, closed-form
  moments, scintillation index and a quadrature rule. Gauss-Hermite for the
  lognormal, capped at 300 nodes because the weights underflow above that; an
  adaptive log-spaced trapezoid rule for the gamma-gamma, whose range is derived
  from the small-`h` power law, the large-`h` Bessel decay and the exact
  log-moments `E[log h] = psi(a) - log a + psi(b) - log b`,
  `Var[log h] = psi'(a) + psi'(b)`. The tensor Gauss-Laguerre rule is also
  provided and is **not** the default, because it stalls near 1e-2 in LLR.
- `softdecode.detection` - the thermal-limited signal-independent AWGN
  detection model, with the Eb/N0 conventions for OOK, M-PPM and BPSK written
  down once and used everywhere.
- `softdecode.llr` - exact OOK LLRs with known CSI (the textbook affine LLR),
  exact LLRs with the channel state marginalised by quadrature, the max-log
  approximation with its one-signed error property, exact and max-log M-ary PPM
  bit LLRs under a Gray labelling, LLR clipping and scalar rescaling.
- `softdecode.csi` - multiplicative (bias plus lognormal jitter) and stale
  (log-correlation `rho`) channel-estimate error models, their closed-form
  Gaussian posteriors on `log h` for a lognormal prior, a numeric posterior for
  the gamma-gamma prior, and the vectorised CSI-aware LLR that averages the
  likelihood over `p(h | h_hat)`.
- `softdecode.ldpc` - a (3,6)-regular LDPC code of length 96 built by Gallager's
  construction with a seeded permutation search, a GF(2) systematic generator
  obtained by elimination with column pivoting, and a vectorised flooding
  sum-product decoder whose check update uses exact leave-one-out cumulative
  products. The realised dimension is 50, not 48, because the construction
  leaves two dependent rows; the rate and the length-4 cycle count are reported
  rather than assumed.
- `softdecode.codes` - the extended Hamming (8,4) code with exhaustive
  maximum-likelihood soft decoding and minimum-Hamming-distance hard decoding,
  used for cross-check X3.
- `softdecode.metrics` - bit error rate with its binomial standard error, LLR
  error statistics, and the generalised mutual information of a set of LLRs.
- `softdecode.simulate` - the end-to-end OOK-over-fading simulation with a
  frozen random-draw order, six named demappers and the decoded-BER helper.
- `softdecode.corrector` - a random-forest LLR corrector with an
  ensemble-dispersion uncertainty output, plus the tuning of the non-learned
  competitors' free parameters on a split disjoint from training and reporting.
- `softdecode.datasets` - the three disjoint seeded splits and the multi-Eb/N0
  training set.
- CLI `python -m softdecode` with `llr`, `maxlog`, `clip`, `mismatch`, `ppm`,
  `crosscheck` and `ldpc` subcommands.
- 161 tests including Hypothesis property tests, a brute-force known-answer test
  for the sum-product decoder on a cycle-free graph, and the AWGN-limit
  known-answer test. Six validation scripts plus the worked example, with their
  committed raw output; four examples, each producing a PNG in `screenshots/`.

### Known limitations in this release

- Fading is drawn **independently per channel bit**, which is the
  ideal-interleaving limit. Real atmospheric fades last milliseconds; every
  coded number here is optimistic for a real link by an amount this package does
  not estimate.
- The noise model is thermal-limited and signal-independent. A
  shot-noise-limited or avalanche-photodiode receiver has signal-dependent noise
  and every LLR here is wrong for it.
- The learned corrector loses to the analytic CSI-aware LLR in all three
  mismatch regimes measured. That is published as the result.
- The forest's per-tree dispersion is not a calibrated interval, and by itself
  it ranks confidence the wrong way round; normalised by `1 + |L|` it orders
  risk correctly. Both forms are reported as measured.
- The 80-node Gauss-Hermite rule gives a worst LLR deviation of 1.7e-3 at
  scintillation index 0.3 but 9.7e-2 at 1.0; strong scintillation needs more
  nodes, and the convergence table says how many.
- No model binary is shipped. The corrector is retrained deterministically from
  the committed seeds, which takes about 12 s.
