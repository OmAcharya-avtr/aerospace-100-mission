# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-06

First release. Validation level 2 (research grade). Status: TESTING.

### Added

- `slotsync.pulses` - finite-support pulse shapes in symbol-period units
  (rectangle, triangle, raised cosine in time, half-sine, and a truncated
  frequency-domain Nyquist raised cosine), each normalised to unit peak with its
  support, inter-symbol span, residual value at one symbol and energy exposed.
  Finite support is what makes the S-curves exact.
- `slotsync.detectors` - the early-late gate (raw, squared and decision-directed
  forms), Gardner's detector for real and complex baseband samples, and the
  Mueller-Müller detector, as pure functions of already-taken samples, all under
  one sign convention: a positive output means the sampling instant is late.
  Gardner's invariance to a constant carrier phase is stated in the docstring,
  proved in one line, and checked by a Hypothesis property test.
- `slotsync.ted` - the configuration and dispatch layer, including the map from
  detector taps to **physical sample instants**, so that a simulation draws one
  noise sample per instant rather than one per tap. Consecutive updates share
  samples for Gardner and Mueller-Müller, and getting that wrong whitens the
  detector noise artificially.
- `slotsync.stream` - analytic sampling of a pulse-amplitude-modulated stream at
  arbitrary instants, data-pattern enumeration, and the per-sample noise scale.
- `slotsync.scurve` - **exact** S-curves by enumerating every data pattern in the
  pulse's span (32 to 4096 patterns), with the detector gain `K_d` measured two
  ways, the lock-point bias, the 10 % linear range, the flattening point, the
  reversal point and the detector self-noise at zero offset.
- `slotsync.loop` - the second-order loop: the noise-bandwidth relation verified
  against quadrature, coefficients by exact pole matching with a closed-form
  inverse, the timing jitter variance by three routes (classical closed form,
  exact discrete-time Lyapunov, and the same with a measured noise
  autocovariance), the loop SNR, and a Gaussian level-crossing estimate of the
  cycle-slip rate.
- `slotsync.simulate` - open-loop detector statistics (mean, variance, self-noise
  share and autocovariance) and a closed-loop Monte Carlo of the actual loop with
  batch-means standard errors on both the jitter variance and the static offset.
- `slotsync.ppm` - the M-ary PPM slot clock: an exact slot S-curve, the open-loop
  statistics and autocovariance, the closed-loop slot timing loop, the
  per-update to per-slot bandwidth conversion, the measured duty-cycle cost, and
  the slot-index error rate. The module docstring states what carries over from
  binary symbol timing and what does not, including why Gardner and
  Mueller-Müller have no slot-clock analogue.
- CLI `python -m slotsync` with `scurve`, `loop`, `jitter`, `slip` and `ppm`
  subcommands.
- 234 tests including Hypothesis property tests and five hand-computable
  known-answer families with their derivations in the test comments; six
  validation scripts plus a worked example, each with its committed raw output;
  five examples, each producing a PNG in `screenshots/`.
- `docs/TIMING_MODEL.md` with every derivation the package relies on and the four
  verified bibliographic references.

### Known limitations in this release

- The per-sample SNR is an electrical sample SNR, not an optical `Es/N0`. There
  is no photon-counting model, no receiver filter, no shot noise and no
  conversion to an optical sensitivity.
- Additive noise is independent per physical sample instant. Samples shared
  between consecutive updates are handled correctly, but noise between distinct
  instants is not correlated as a real matched-filter output's would be.
- The classical white-noise jitter expression over-predicts by 15.3x for the
  decision-directed early-late gate and 3.1x for Gardner's detector on a Nyquist
  pulse at 20 dB per-sample SNR, because both are dominated by detector
  self-noise. It is right to 0.5 % for Mueller-Müller, which has none. Shipped
  with that measurement attached.
- The Gaussian level-crossing cycle-slip estimate is low by 6 to 16 orders of
  magnitude in the regime where slips occur. Its functional form is wrong, not
  its scale; no fitted prefactor is offered. The measured threshold is published
  instead.
- The closed-loop Monte Carlo deviates systematically from the coloured-noise
  prediction, from about 3 % at `B_n` = 0.001 to 14 % at 0.02, which is up to
  9.96 Monte Carlo standard errors. It is the S-curve nonlinearity and the
  decision errors; no linearised prediction contains either.
- Detector self-noise makes the loop lock away from the true symbol centre by
  0.383 (early-late) or 0.575 (Gardner) symbol per unit `B_n`. Measured and
  characterised; not corrected.
- Mueller-Müller has exactly zero gain on a rectangular and on a half-sine pulse
  and is ill-conditioned on a time-domain raised cosine.
- One loop topology only: second order, type 2, one update per symbol. No
  carrier recovery, no frequency-offset estimation, no acquisition aid, no
  frame synchronisation.
- The PPM slot clock implements the early-late energy family only.
