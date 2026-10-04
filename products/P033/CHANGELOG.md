# Changelog

All notable changes to `edgeinfer` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-10-04

Initial release. Status `TESTING`; validation level **3, hardware-pending**.

### Added

- **Declared budget as a first-class object** (`edgeinfer.budget`): worst-case
  latency, median latency, peak memory, power ceiling, duty cycle and control
  period, with per-limit pass/fail, a `MARGINAL` verdict when the margin is
  below two combined standard uncertainties, and detection of a budget that is
  **infeasible by construction** before any model is loaded.
- **Analytic cost model** (`edgeinfer.ops`, `edgeinfer.roofline`,
  `edgeinfer.analytic`): per-operator operation counts and compulsory memory
  traffic with cited formulas; a roofline latency bound (Williams, Waterman &
  Patterson 2009) with explicit per-node and fixed overhead terms; peak memory
  by liveness analysis (Aho et al. 2006 §8.4); and a four-parameter device
  calibration labelled `calibrated-fit`.
- **Model-graph IR and ONNX I/O** (`edgeinfer.graph`, `edgeinfer.pbwire`,
  `edgeinfer.onnx_io`): build and parse ONNX ModelProto bytes with a
  hand-written protobuf codec, because the `onnx` package is unavailable in the
  target environment. Field numbers verified against models shipped inside
  `onnxruntime`.
- **Backends behind one contract** (`edgeinfer.backends`): a simulated backend
  driven by an injected per-stage cost model, an `onnxruntime` backend and a
  scikit-learn backend, with a dry-run mode that exercises the real command path
  and discards the output.
- **Benchmark harness** (`edgeinfer.harness`): repeated timing with warm-up and
  garbage collection disabled, reporting p50, p90, p99, mean, standard deviation
  and the observed maximum separately, each with its measurement method and
  repeat count.
- **Uncertainty budget** (`edgeinfer.uncertainty`): GUM Type A (`s/sqrt(n)`,
  bootstrap for a quantile) and Type B (clock quantisation) contributions, with
  the measured timer-call cost reported as a bias rather than folded in.
- **Peak-memory measurement** (`edgeinfer.memtrace`): `tracemalloc`-based peak
  with the `onnxruntime` arena's invisibility documented and demonstrated.
- **Declared thermal throttling** (`edgeinfer.thermal`), with independent
  compute and bandwidth factors and the basis recorded in the device's source
  string.
- **Learned predictor** (`edgeinfer.dataset`, `edgeinfer.features`,
  `edgeinfer.predictor`): a reproducible synthetic ONNX population, 17 graph
  features, a random forest on `log10` latency with an ensemble-disagreement
  uncertainty, and a paired-bootstrap comparison that reports *no measurable
  advantage* when the difference is inside the bootstrap standard deviation.
- **Result-file writer** (`edgeinfer.report`): every output file states the
  environment it was measured on and carries a target-device column that is
  left empty, with no API that would let a caller fill it.
- **CLI**: `python -m edgeinfer {analyse,profile,check,feasibility,env}`.
- 467 tests, including hand-count known-answer tests, Hypothesis property tests,
  failure-mode tests for all four modes the specification names, a pinned seeded
  regression suite and a performance benchmark.
- Eight validation scripts with their raw captured stdout, `docs/REQUIREMENTS.md`
  with 18 numbered requirements and a verification matrix, `MODEL_CARD.md`,
  `DATASET_CARD.md`, and four runnable examples each saving a figure.
- `validation/crosscheck_pipeline.json` for the independent cross-check against
  P039 LatencyNet.

### Results reported, including the unfavourable ones

- Over five independent measurement passes, the learned predictor beats the
  analytic baseline on **median** latency (8.42–15.20 % against 21.20–24.96 %
  median absolute relative error) and shows **no measurable advantage** on
  **worst-case** latency on every pass. On **peak memory the analytic model wins
  outright**, being exact by construction.
- The first cost-model recovery check asserted on the arithmetic mean and
  **failed** at a measured/injected ratio of 1.494. The check was moved to the
  median rather than the band being widened; the mean and trimmed mean are still
  reported and never asserted.
- The learned model's uncertainty output is **not calibrated**; its measured
  coverage is 75.0 % within ±1 u and 91.7 % within ±2 u on the p50 target.

### Known limitations

See `README.md` §Limitations and `validation/VALIDATION.md` §10. In particular:
no measurement from any edge device exists, power is never measured, the
analytic peak memory is a lower bound, and the operator set is narrow and
refuses rather than approximating.

[0.1.0]: https://github.com/OmAcharya-avtr/edgeinfer/releases/tag/v0.1.0
