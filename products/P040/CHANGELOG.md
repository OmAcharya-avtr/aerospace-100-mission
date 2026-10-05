# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-05

First release. Validation level 2 (research grade). Status: TESTING.

### Added

- `bitflipsim.bitlayout` - IEEE 754 binary16/32/64 and two's-complement
  int8/16/32 layouts discovered from `numpy.finfo` and re-derived independently
  from probe encodings; exact single-bit-flip prediction from the layout
  formula in integer arithmetic (`predict_flip`,
  `exponent_field_prediction`, `exponent_scale_factor`), plus bit-pattern
  conversion and flipping primitives.
- `bitflipsim.flux` - the rate model `lambda = flux * cross_section * bits`
  with units, the FIT conversion, the Poisson count model, a validity check on
  the small-per-bit-probability assumption, and sample statistics carrying the
  Poisson sampling errors of both the mean and the variance.
- `bitflipsim.injection` - uniform bit-site sampling with and without
  replacement, exact bitwise injection into any 8/16/32/64-bit numpy array,
  and an independent population-count check on the injector.
- `bitflipsim.network` - a flat, bit-addressable two-layer MLP whose forward
  pass reproduces `MLPClassifier.predict_proba` bit-exactly in float64, with
  stated conventions for non-finite logits, and symmetric per-tensor int8
  quantization.
- `bitflipsim.datasets` - the deterministic synthetic three-class problem, its
  train / calibration / evaluation splits, and the seeded reference model.
- `bitflipsim.criticality` - the total-variation degradation metric, the
  exhaustive per-bit ground-truth sweep for float32 and int8, the magnitude
  and exponent-bit baselines, and degradation avoided per protected byte under
  a bit-granular and a word-granular cost model with exact tie expectations.
- `bitflipsim.predictor` - a random-forest criticality predictor on 14
  injection-free features with an ensemble standard deviation as its
  uncertainty output, a grouped parameter split, and a coverage measurement of
  that uncertainty.
- `bitflipsim.mitigation` - parameter range clamping with a derived
  single-upset logit-deviation bound, word-level and bitwise majority voters,
  periodic-reload accounting including `E[live upsets] = lambda T_s / 2`, and
  memory and latency cost records for all three schemes.
- `bitflipsim.campaign` - Poisson-sized parameter, activation and flux
  campaigns reporting standard errors, and campaign sizing from a target
  standard error.
- `bitflipsim.onnx_io` - a minimal ONNX reader, writer and in-file initializer
  bit patcher implemented directly against the protobuf encoding, because the
  `onnx` package is not available in the build environment. Verified against
  onnxruntime.
- CLI `python -m bitflipsim` with `layout`, `flip`, `flux`, `campaign`,
  `criticality`, `mitigate` and `onnx` subcommands.
- 126 tests including Hypothesis property tests and two exhaustive
  enumerations; six validation scripts with their committed raw output; four
  examples, each producing a PNG in `screenshots/`.

### Known limitations in this release

- The upset model is single-bit, independent and uniform over bit sites.
  Multiple-bit upsets from one particle, single-event functional interrupts,
  latch-up and total-dose effects are not modelled.
- The cross-section is a single effective number, not an integral of a
  measured LET-dependent cross-section over an environment spectrum. The flux
  and cross-section shipped with the package are round illustrative constants
  and are not attributed to any measured environment or part.
- The reference model is 147 float32 parameters. Nothing here establishes that
  the criticality ordering or the learned predictor's margin transfers to a
  model of realistic size, and the README says so.
- The clamping bound is derived for the two-layer ReLU network of
  `bitflipsim.network` and for single upsets. It is not a general bound for an
  arbitrary graph, and it is loose by a factor of 7.86 against the worst of
  the 4704 exhaustive single-upset cases.
- The random forest's ensemble dispersion is a confidence signal, not a
  calibrated interval: measured coverage at two standard deviations is 0.8782
  against a Gaussian reference of 0.9545.
- `bitflipsim.onnx_io` handles float32 initializers stored in `raw_data` only.
  It does not execute ONNX graphs, and reports `float_data` tensors, non-float32
  dtypes, external data and sparse initializers as unsupported.
