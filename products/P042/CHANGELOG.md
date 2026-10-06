# Changelog

All notable changes to acmpilot. Format follows Keep a Changelog; versioning is
semantic.

## 0.1.0 — 2026-10-06

First release. Status: TESTING. Validation level 3. Research-grade.

### Added

**Channel (`acmpilot.channel`)**
- Stationary first-order Gauss-Markov driver, exact discrete sampling of the
  Ornstein-Uhlenbeck process, with the correlation time as a stated input.
- Lognormal irradiance (Andrews & Phillips 2005) and gamma-gamma irradiance
  (Al-Habash, Andrews & Phillips, *Opt. Eng.* 40(8), 2001) with unit mean, the
  latter through a Gaussian copula that preserves the marginal exactly.
- Emergent fade statistics: outage fraction, fade-duration distribution with
  right-censored runs excluded, downward level-crossing rate.
- Analytic Rice level-crossing rate for cross-checking the sample statistic.
- `detector_exponent` to switch between coherent detection and
  thermal-noise-limited direct detection.

**Modulation and coding (`acmpilot.modulation`, `acmpilot.coding`)**
- Gray-mapped unit-energy BPSK, QPSK, 8PSK and 16QAM with minimum-distance
  detection; Monte Carlo BER and per-bit error positions.
- Closed-form BER for all four, labelled exact for BPSK/QPSK and high-SNR
  approximate for 8PSK/16QAM, used only as checks.
- Reed-Solomon bounded-distance accounting from channel BER to post-decoding
  frame, symbol and bit error rate, with the code parameters following from the
  MDS property rather than a table.

**MODCOD ladder (`acmpilot.modcod`)**
- Eight MODCODs over four constellations and four RS rates, spanning 0.498 to
  3.749 bit/symbol.
- Thresholds **measured** by this package at a target post-decoding BER of 1e-6,
  each with a Monte Carlo 1-sigma uncertainty (worst 0.045 dB). JSON
  serialisation that contains no filesystem path.

**Policies and accounting (`acmpilot.policy`, `acmpilot.simulate`,
`acmpilot.accounting`)**
- Explicit round-trip feedback delay: the policy at slot `n` sees slot `n - d`.
- Fixed margin, threshold with hysteresis, and a clairvoyant upper bound labelled
  in code, docs and every plot legend as unachievable by any causal policy.
- Mis-selection split into too-aggressive and too-conservative with their separate
  costs in bit/symbol, plus the unavoidable share the channel caused.
- Delay sweep with the standard error of the mean over seeds.

**Learned component (`acmpilot.predictor`, `acmpilot.benchmark`)**
- Analytic AR(1) MMSE predictor, **not learned**, implemented first.
- Learned quantile predictor: three gradient-boosting models at the 10th, 50th and
  90th percentiles, emitting a state-dependent confidence that gates aggressive
  rate choices.
- Calibration reporting: interval coverage, tail exceedance, pinball loss,
  quantile-crossing fraction.
- A benchmark protocol with three disjoint seed sets that tunes every baseline's
  free parameter before comparing it to the model.

**Interface and evidence**
- `python -m acmpilot` with `thresholds`, `simulate`, `sweep` and `predict`.
- Five validation scripts with their raw saved output; five examples, each writing
  a figure to `screenshots/`.
- `docs/REQUIREMENTS.md`: 73 numbered requirements, each naming the test or
  validation section that exercises it, including six written down as deliberately
  not met in 0.1.0.

### Published results

- **The learned predictor does not beat the analytic AR(1) MMSE predictor at any
  feedback delay, on either channel marginal.** It is nominally behind at all five
  delays on the lognormal channel and never separated by two combined standard
  errors on either. The lognormal channel is AR(1) in log-amplitude by
  construction, so a three-parameter linear predictor is already optimal. This is
  the published result and no baseline was removed or retuned to change it.
- The learned model's only measured advantage is calibration on the non-Gaussian
  gamma-gamma channel: nominal-80% coverage 0.794 to 0.800 against 0.821 to 0.823
  for the analytic Gaussian interval.
- Both predictive policies beat the **tuned** non-learned baselines by +1.82% at
  `tau` = 2 ms rising to +11.48% at 20 ms. At `tau` = 1 ms a tuned fixed margin is
  statistically indistinguishable from both, so prediction earns nothing there.
- The confidence gate is worth +0.15 to +0.28 bit/symbol; the choice of predictor
  is worth 0.002 to 0.005. The uncertainty output matters far more than the
  regression.
- The shipped 3 dB default fixed margin is not optimal at any delay measured; the
  measured optimum is 0 dB at zero delay and 1 to 2.5 dB above it.

### Fixed during development

- `rice_level_crossing_rate_hz` used `1/(2*pi*tau_c)` as its rate prefactor, which
  contradicted its own docstring and disagreed with the measured crossing rate by a
  factor of 3 to 6 that grew with the correlation time. The docstring was correct;
  the implementation now uses the documented discrete-sampling derivative variance
  and agrees with the measurement to 1.7% worst case over nine cases.

### Test and lint status at release

`python -m pytest tests/ -q --junit-xml` reports 398 tests, 0 failures, 0 errors,
0 skipped, in 34 s on 2 contended cores. `ruff check src/ tests/ examples/
validation/` is clean with `line-length = 100` and
`select = ["E", "F", "W", "I", "UP", "B"]`.

### Known limitations at 0.1.0

The feedback report is noiseless, only stale. Policies cannot mute. Reed-Solomon
miscorrection is neglected. Interleaver depth, latency and memory are not modelled
or charged. The fade-duration distribution is sampling-rate dependent. Nothing is
validated against measured turbulence data. See the README and
`docs/REQUIREMENTS.md` section 7.
