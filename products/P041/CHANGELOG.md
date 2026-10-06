# Changelog

All notable changes to `codedfade` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## 0.1.0 — 2026-10-06

First release. Status: TESTING. Validation level 3, hardware-pending.

### Added

- `codedfade.channel`: temporally correlated optical fading as filtered-Gaussian
  sample paths. Gauss-Markov (`exp`) and Gaussian (`gauss`) correlation kernels;
  lognormal and gamma-gamma marginals; plane-wave Rytov-to-gamma-gamma parameter
  relations with the saturation branch detected and refused rather than solved on
  the wrong side; sample autocorrelation and 1/e correlation-time estimator.
- `codedfade.fade`: fade statistics with the sample definitions written down
  exactly (level-crossing rate, mean and median fade duration, outage fraction,
  censored-run handling), Rice's (1945) continuous-time crossing rate for the
  Gaussian kernel, and the **exact** crossing rate and mean fade duration of the
  sampled Gauss-Markov path from the bivariate-normal orthant probability.
- `codedfade.gf`: GF(2^m) arithmetic by log/antilog tables for m in {3,4,5,6,8},
  with primitivity verified at construction.
- `codedfade.reedsolomon`: systematic narrow-sense Reed-Solomon encoder and
  Berlekamp-Massey / Chien / Forney decoder in numpy.
- `codedfade.convolutional`: rate-1/n feedforward convolutional encoder and
  hard-decision Viterbi decoder over the terminated trellis.
- `codedfade.interleave`: block and Forney convolutional interleavers with their
  latency and memory cost in symbols, milliseconds and bytes.
- `codedfade.link`: coded OOK link over the correlated channel, and the
  interleaver-depth sweep that is the product's central output. Every depth in a
  sweep sees an identical channel record.
- `codedfade.hal`: one modem/codec backend interface; a fully implemented
  simulated backend; a contract-only device backend; simulation and dry-run modes;
  executable preflight checks; a run capture carrying no filesystem path; and an
  idempotent backout path.
- `codedfade.predictor`: learned fade-duration-exceedance classifier with a
  per-tree uncertainty output, and the analytic and empirical state-blind baselines
  it is benchmarked against.
- `python -m codedfade` with the `channel`, `fade`, `depth`, `code`, `preflight`
  and `dryrun` subcommands.
- `benchmark/run_benchmark.py`, which writes its own measurement method and
  environment into its output file.
- `docs/REQUIREMENTS.md`: 33 numbered requirements, each naming the test that
  exercises it, plus an explicit list of what is **not** claimed.

### Known limitations at 0.1.0

See the Limitations section of `README.md`. The ones that change what a reader
should believe:

- No hardware measurement exists. The device backend raises
  `NotImplementedError`. This product is **Level 3, hardware-pending** and is never
  labelled Level 4.
- The gamma-gamma temporal correlation is imposed through a Gaussian copula, so the
  specified correlation time governs the latent Gaussian rather than the
  log-irradiance. The measured disagreement is published.
- The exponential closure of the level-crossing result over-predicts
  fade-duration exceedance at `t = MFD`. The measured ratio is published and the
  closure is not retuned.
- The learned predictor's advantage over a state-blind constant equal to the
  measured exceedance frequency is small. Both numbers are published.
