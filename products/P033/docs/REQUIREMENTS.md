# EdgeInfer — Requirements and Verification Matrix

**Product:** P033 EdgeInfer · **Package:** `edgeinfer` · **Version:** 0.1.0 · **Status:** TESTING
**Validation level:** 3, hardware-pending · **Licence:** Apache-2.0
**Date of this revision:** 2026-10-04

This document states what `edgeinfer` is required to do, in numbered,
individually verifiable terms, and names for each requirement the test or
validation script that verifies it. Requirements are written so that a
verification can *fail*: "shall agree with X to within Y" rather than "shall be
accurate".

Paths are relative to `products/P033/`. Test identifiers are pytest node ids;
validation identifiers are scripts in `validation/` whose captured stdout is
stored beside them as `*_output.txt`.

---

## 1. Scope and definitions

`edgeinfer` answers one question about a candidate inference model: does it fit
inside a declared deployment envelope of latency, peak memory, power and duty
cycle, and how confident is that answer? It contains no training, no model
optimisation, no quantisation and no device driver.

| Term | Meaning in this document |
|---|---|
| *budget* | a declared envelope: worst-case latency [s], optional median latency [s], peak memory [B], optional power [W] and duty cycle [dimensionless] |
| *worst case* | a stated high quantile of the measured latency distribution, p99 by default; the observed maximum is reported alongside it |
| *median* | the p50 of the measured latency distribution |
| *analytic estimate* | latency and peak memory computed from the model graph with nothing executed |
| *roofline bound* | `t >= max(F/peak_flops, B/peak_bw)` [s], Williams, Waterman & Patterson 2009 |
| *compulsory traffic* | each tensor read once and written once [B]; a lower bound on real DRAM traffic |
| *peak activation bytes* | the largest total live non-weight tensor size over a sequential execution [B], by liveness analysis |
| *calibrated baseline* | the analytic model with its four device parameters fitted on a training split; labelled `calibrated-fit`, never a datasheet figure |
| *declared* | a value the caller supplied; never measured by this package |
| *simulated backend* | a pipeline whose per-stage cost is injected by the caller; its output is never a device measurement |
| *u_c* | combined standard uncertainty, root-sum-square of independent contributions (JCGM 100:2008 §5.1.2) |
| *hardware-pending* | Level 3 with the Level 4 groundwork present and no measurement from the target device |

---

## 2. Functional requirements

### R-01 — Budget as a declared object
The package shall represent a deployment envelope as a single object carrying
worst-case latency [s], peak memory [B], and optionally median latency [s],
average power [W], duty cycle [dimensionless] and control period [s]; and shall
reject a non-positive latency, a non-positive memory ceiling, a duty cycle
outside `(0, 1]`, a non-positive period or power, and a worst-case quantile
outside `(0.5, 1.0)`, each with a diagnostic naming the offending field.

### R-02 — Infeasible-by-construction detection
The package shall detect, **without loading or measuring any model**, a budget
whose declarations contradict each other: a median ceiling above the worst-case
ceiling, a duty-cycle-implied ceiling below the declared worst-case ceiling, and
a median ceiling above the duty-cycle-implied ceiling. It shall report every
contradiction found, not only the first, and shall return `FAIL` for such a
budget whatever is measured against it.

### R-03 — Worst case and median reported separately
Every latency the package reports shall carry p50, p90, p99, the mean, the
standard deviation and the observed maximum as separate figures, and the budget
check shall decide the worst-case row against the declared quantile and the
median row against the median ceiling independently. A candidate inside the
median ceiling and outside the worst-case ceiling shall produce an overall
verdict of `FAIL` or `MARGINAL FAIL`.

### R-04 — Every number carries its measurement method and repeat count
No code path shall emit a latency without its clock, clock resolution, repeat
count, warm-up count and garbage-collection state. A result row without a
measurement-method string shall be refused at construction.

### R-05 — Analytic operation counts match a hand count
The analytic operation count, memory-traffic count, resident-weight count and
peak-activation count shall agree **exactly** (integer equality, tolerance 0)
with a hand count for two fixed small networks: a 4-6-3 MLP with ReLU and a
one-block CNN with `Conv(1->2, 3x3, pad 1)`, `Relu`, `MaxPool(2x2, s2)`,
`Reshape` and `Gemm(18, 2)`.

### R-06 — ONNX output shapes follow the standard
Convolution and pooling output spatial sizes shall follow the ONNX operator-set-17
formula `floor((in + pad_begin + pad_end - dilation*(kernel-1) - 1)/stride) + 1`,
and a kernel whose effective extent exceeds the padded input shall be rejected
rather than producing a non-positive dimension.

### R-07 — Unsupported operators are refused, never charged zero
A graph containing an operator with no cost rule shall raise
`UnsupportedOperatorError` from both shape inference and cost evaluation, naming
the operator, the node and the supported set. The analytic estimate shall not be
produced for such a graph.

### R-08 — Roofline bound is a bound
For device peaks declared well above the host's effective rate, the pure
roofline estimate (both overhead terms zero) shall lie below the measured
latency of every model in the held-out set. The time form and the rate form of
the roofline shall agree to floating-point tolerance.

### R-09 — Peak memory by liveness analysis
Peak activation memory shall be computed by liveness analysis over the
topologically ordered node list, with a node's outputs allocated before its dead
inputs are released, weights excluded and graph outputs never released. The
result shall match a measured execution of the same dataflow in NumPy to within
one allocator header per live tensor, and to within 5 % relative for a graph
whose payload dominates the headers.

### R-10 — Peak-memory measurement reproduces a known allocation pattern
The `tracemalloc`-based measurement shall reproduce hand-counted allocation
patterns — one array, two simultaneously live arrays, two sequentially live
arrays, and a dtype-width change — to within 1024 B of the payload, and the
float64/float32 peak ratio shall be 2.0 to within 1 %.

### R-11 — Measured latency reproduces an injected cost model
The timing harness, applied to a simulated backend that consumes a sampled
per-stage duration, shall recover the injected mean. The check shall be made on
the **median**, which must lie within `[1.00, 1.15]` of the injected mean; the
arithmetic mean and the trimmed mean shall be reported alongside it so that the
size of the scheduler contamination on the host is visible.

### R-12 — Uncertainty budget over timing
Every reported statistic shall carry a combined standard uncertainty with its
Type A contribution (`s/sqrt(n)` for a mean, JCGM 100:2008 §4.2.3; a
nonparametric bootstrap for a quantile, Efron & Tibshirani 1993 §6), its Type B
clock-quantisation contribution (`d/sqrt(12)` per read, §4.3.7, two reads in
quadrature), and the measured timer-call bias stated separately as a bias rather
than folded into the uncertainty.

### R-13 — Budget verdicts carry their uncertainty
A pass or fail whose margin is smaller than two combined standard uncertainties
shall be reported as `MARGINAL PASS` or `MARGINAL FAIL`, not as `PASS` or
`FAIL`.

### R-14 — Power is never reported as measured
This package measures no power. A power row shall always carry the verdict
`NOT MEASURED` and a method string saying so, whatever value the caller declares,
and shall not contribute to the overall verdict.

### R-15 — Declared throttling, never simulated as measurement
A thermally throttled device shall be expressed as a derating of a declared
device model with independent compute and bandwidth factors, the bandwidth
factor defaulting to 1.0; and the resulting device's name and source string
shall record both factors and the caller's stated basis, so that a throttled
figure cannot be read as a nominal or a measured one.

### R-16 — The analytic baseline precedes and is benchmarked against the ML model
The analytic cost model shall be implemented and validated before the learned
predictor. Both shall be evaluated on the same held-out split with the same
features and the same training set, on median latency, worst-case latency and
peak memory. Where the difference in median absolute relative error lies inside
a paired-bootstrap standard deviation, the result shall be reported as **no
measurable advantage**. The outcome shall be reported in `README.md` whichever
way it falls, and shall be established over several independent measurement
passes rather than one.

### R-17 — The learned model exposes an uncertainty, with its coverage measured
The learned predictor shall return an uncertainty with every point prediction,
propagated from the fitting space by the GUM sensitivity coefficient, and its
empirical coverage on the held-out set shall be measured and reported rather
than asserted.

### R-18 — Reproducibility and the output artefact
The synthetic model population, the train/test split, the calibration fit and
the forest fit shall be byte- or value-identical for a given seed, and pinned
outputs shall be regression-tested. Every results file this package writes shall
state, in its own text, the environment the numbers came from — including that
the host is a shared single-CPU-core cloud container — and shall carry a
target-device column that is left empty, with no API that would let a caller
fill it.

---

## 3. Verification matrix

| Req | Verified by | Kind |
|---|---|---|
| R-01 | `tests/test_budget.py::TestBudgetValidation` · `::TestReportText` | test |
| R-02 | `validation/validate_budget.py` §(4) · `tests/test_budget.py::TestFeasibility` · `tests/test_failure_modes.py::TestInfeasibleBudget` · `tests/test_properties.py::TestBudgetProperties` · `tests/test_cli.py::TestFeasibilityCommand` | validation + test |
| R-03 | `validation/validate_budget.py` §(5)/(6) · `tests/test_harness.py::TestLatencyProfile` · `tests/test_budget.py::TestVerdicts::test_worst_case_failure_overrides_a_median_pass` · `tests/test_cli.py::TestProfile` | validation + test |
| R-04 | `tests/test_report.py::TestResultsTable::test_a_row_without_a_method_is_rejected` · `::test_every_method_is_listed` · `tests/test_harness.py::TestLatencyProfile::test_method_line_names_the_clock_and_the_repeat_count` · every `validation/*_output.txt` | validation + test |
| R-05 | `validation/validate_opcounts.py` §hand MLP, §hand CNN · `tests/test_ops.py::TestHandCountedMlp` · `::TestHandCountedCnn` | validation + test |
| R-06 | `tests/test_graph.py::TestConvOutputSpatial` · `tests/test_properties.py::TestConvFormulaProperties` · `tests/test_ops.py::TestPerOperatorCosts` | test |
| R-07 | `validation/validate_budget.py` §(2) · `tests/test_ops.py::TestUnsupportedOperator` · `tests/test_failure_modes.py::TestUnsupportedOperator` · `tests/test_features.py::TestFeatureVector::test_an_unsupported_operator_is_refused_not_featurised_as_zero` | validation + test |
| R-08 | `validation/validate_predictor.py` §(i-a) · `tests/test_roofline.py::TestRooflineTime` · `tests/test_analytic.py::TestAnalyticEstimate::test_estimate_is_a_lower_bound_on_a_measured_onnxruntime_latency` · `tests/test_properties.py::TestCostIdentities` | validation + test |
| R-09 | `validation/validate_memory.py` §(b) · `tests/test_analytic.py::TestLiveness` · `tests/test_ops.py::TestHandCountedCnn::test_peak_activations_match_the_hand_count` · `tests/test_properties.py::TestLivenessProperties` | validation + test |
| R-10 | `validation/validate_memory.py` §(a) · `tests/test_memtrace.py::TestKnownAllocationPattern` | validation + test |
| R-11 | `validation/validate_backend_cost_model.py` §(a), §(b) · `tests/test_harness.py::TestBenchmark::test_measured_mean_recovers_an_injected_cost_model` · `tests/test_backends.py::TestSimulatedBackend` · `tests/test_integration.py::TestCrosscheckExport` | validation + test |
| R-12 | `validation/validate_uncertainty.py` §(a)-(d) · `tests/test_uncertainty.py` (all classes) · `tests/test_harness.py::TestLatencyProfile::test_uncertainty_for_the_mean_uses_the_standard_error` | validation + test |
| R-13 | `validation/validate_budget.py` §(5) · `tests/test_budget.py::TestVerdicts::test_a_pass_inside_the_uncertainty_is_marked_marginal` · `::test_a_fail_inside_the_uncertainty_is_marked_marginal` | validation + test |
| R-14 | `validation/validate_budget.py` §(5) full report · `tests/test_budget.py::TestVerdicts::test_power_is_always_not_measured_even_when_a_value_is_passed` · `tests/test_cli.py::TestCheck::test_the_power_row_is_never_measured` | validation + test |
| R-15 | `validation/validate_budget.py` §(3) · `tests/test_failure_modes.py::TestThermallyThrottledDevice` · `tests/test_integration.py::TestFullCandidateCheck::test_the_same_candidate_can_fail_a_throttled_device_budget` | validation + test |
| R-16 | `validation/validate_predictor.py` §(2), §(3), §(4) · `tests/test_predictor.py::TestComparison` · `tests/test_integration.py::TestPredictorPipeline` · reported in `README.md` §Validation evidence and §Result | validation + test |
| R-17 | `validation/validate_predictor.py` §(v) · `tests/test_predictor.py::TestUncertaintyOutput` | validation + test |
| R-18 | `tests/test_regression.py` (all pinned outputs) · `tests/test_dataset.py::TestReproducibility` · `tests/test_report.py::TestWriteResultsFile` · `::TestResultsTable` · `validation/performance_results.md` | validation + test |

Performance is reported by `validation/validate_performance.py` and bounded by
`tests/test_performance.py`; the compute budget those enforce is in
`README.md` §Limitations.

---

## 4. What is required for validation level 4, and is absent

Level 4 requires measured timing and resource use **from a Jetson Orin Nano
itself**. This repository has none. Specifically, the following do not exist and
cannot be produced in a cloud container:

1. `validation/validate_performance.py` run on the device, with its raw stdout
   captured, and the device column of `validation/performance_results.md`
   transcribed from that output.
2. A measured peak working set from the device's allocator, rather than the
   analytic lower bound and the `tracemalloc` Python-side figure this
   repository has (R-09, R-10).
3. A measured power trace from the device's own rails, which would make R-14's
   `NOT MEASURED` row decidable.
4. A measured DVFS frequency schedule and thermal-zone log under sustained
   load, which would replace the declared throttle factors of R-15 with
   measured ones.
5. A re-fit of the analytic baseline's four device parameters against
   on-device measurements, and a re-run of R-16's comparison there, since a
   result obtained against `onnxruntime`'s CPU execution provider on a shared
   x86 container transfers to an Orin Nano only as a hypothesis.

Until a session is given raw output from the device, this product is
`Level 3, hardware-pending`, the device column stays empty, and the mission's
Level 4 count stays at zero. A promoted label without a measurement is the same
defect class as a fabricated benchmark.
