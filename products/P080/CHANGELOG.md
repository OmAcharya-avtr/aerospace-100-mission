# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-10-24

First release. Status: TESTING. Validation level 2, research grade.

### Added

- `boundcompress.quantise`: the bound-implied quantiser. Three declared modes
  — absolute, pointwise relative by log-magnitude quantisation, and
  range-relative — with encode-time verification of every sample, a one-step
  correction for samples floating-point rounding puts outside the bound, and
  `BoundNotAchievable` rather than an unsupportable certificate when no
  admissible integer exists.
- `boundcompress.predictors`: four integer finite-difference extrapolators
  (`previous`, `linear`, `quadratic`, `cubic`), exact in integer arithmetic
  and therefore bit-identical on any machine with int64.
- `boundcompress.rans`: a static range Asymmetric Numeral Systems coder, so
  every rate figure is actual bytes rather than an entropy estimate.
- `boundcompress.entropy`: zigzag folding, escape varints, frequency-table
  serialisation, and two entropy figures — over the coded symbol alphabet and
  over the distinct residual values — both labelled as references and not as
  floors.
- `boundcompress.container`: a 38-byte fixed header and three
  variable-length blocks, with per-block byte accounting that sums to the
  rate.
- `boundcompress.certificate`: `certify`, independent of the encoder, and the
  `Certificate` record — maximum per-sample error, the index where it
  occurred, the margin, the violations, the rate and the container digest.
- `boundcompress.baselines`: `zlib`, `lzma` and `bz2` over the same quantised
  symbol and residual streams at exactly matched distortion, plus three
  lossless references at zero distortion, each round-trip verified.
- `boundcompress.learned`: an ensemble residual predictor in ridge and
  perceptron flavours, with scale-free features, an exact integer gain the
  decoder reproduces, a hand-written elementwise forward pass so the
  encoder's batch evaluation and the decoder's sequential one agree
  bit-for-bit, and an ensemble dispersion as an uncertainty output.
- `boundcompress.ratedistortion`, `boundcompress.benchmark`,
  `boundcompress.dataset`, `boundcompress.report`, and a CLI with `info`,
  `certify`, `sweep`, `compare` and `negative-control`.
- Nine validation scripts with their raw output, nine example figures, and
  287 tests including Hypothesis property tests and eight pinned
  counterexamples.

### Measured and reported as found

- **The declared bound held on all 1,310,720 sample-versus-bound comparisons**
  across every signal, mode, predictor and bound, with a tightest margin of
  1.543376e-12 and zero rounding corrections needed.
- **Eleven deliberate bound violations were caught and located**, including a
  violation of one part in 1e9 of the bound and a reconstruction one unit in
  the last place past it.
- **The package's own entropy coder loses to `lzma` or `bz2` on 6 of 8
  signals** at the median swept bound, by a factor of 4.788 on
  `packet_counter`. The contribution of this product is the certificate and
  the cost accounting, not the coder.
- **A classical predictor wins 14 of 40 benchmark cells.** The learned
  perceptron wins the median cell by 1.3451 per cent and loses one by 55.8824
  per cent; the learned ridge predictor has a negative mean saving and never
  pays back its own 120 bytes. The perceptron pays back its 2328 bytes after
  36.4 arrays.
- **The strongest classical predictor on these signals is first-order
  `previous`, not second-order `linear`.** `linear` wins outright on the two
  piecewise-linear channels and loses elsewhere.
- **A coder with memory went below the first-order entropy reference on 3 of
  8 signals**, by a factor of 6.7 on one. The entropy is not a floor for a
  correlated source.
- **The learned predictors' uncertainty output is mostly uninformative**:
  median Spearman rank correlation 0.0280 against the realised residual,
  range 0.0179 to 0.8329.
- **The perceptron did not converge** within its 400-iteration cap; doubling
  the cap moves the training loss from 0.631258 to 0.629583, 0.27 per cent.

### Defects found during the build and pinned as regressions

- An int64 wrap in the zigzag fold produced a negative symbol index and an
  `IndexError` three frames away. Found by Hypothesis at six samples spanning
  8.26e11 with an absolute bound of 1.192092896e-07. Fixed by the derived
  guard `MAX_SYMBOL_MAGNITUDE = 2**57`.
- An `OverflowError` escaped the quantiser's correction loop at 1e300 with a
  bound of 1e-300. Same guard, applied before the cast.
- A known-answer test asserted 0.01990049875623903 for `2*log1p(0.01)`, wrong
  in the seventh significant figure.
- `entropy.SymbolError` escaped `container.unpack` on a truncated buffer.
- Re-exporting the function `quantise` shadowed the submodule of the same
  name, so `quantise.MODES` raised `AttributeError` from the CLI.
- Two of the first eleven negative controls were too weak to guarantee a
  violation and were not caught. The certificate was correct; the control was
  not. Recorded in `validation/VALIDATION.md` §2.

### Known limitations

See `README.md`. The dominant ones: every number is measured against a seeded
synthetic generator and not against flight data; the package is 1-D only; the
learned predictors are not bit-exact across machines because `tanh` is a libm
call; and a flight implementation should use CCSDS 121.0-B-3 or 122.0-B-2
rather than this coder.

[0.1.0]: https://github.com/OmAcharya-avtr/boundcompress/releases/tag/v0.1.0
