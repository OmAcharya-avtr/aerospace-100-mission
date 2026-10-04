# EdgeInfer — Validation evidence

**Product:** P033 EdgeInfer · **Version:** 0.1.0 · **Status:** TESTING
**Validation level:** 3, hardware-pending
**Date of this run:** 2026-10-04

Every number in this file and in `README.md` was produced by running a script in
this directory in the session that wrote this document. The raw stdout of each
run is stored beside the script as `<script>_output.txt` and is not edited.

---

## 0. The environment every measured number came from

```
Linux-6.18.44-fc-v64-x86_64-with-glibc2.39 x86_64, 1 CPU core(s) available of 1
logical, shared host, Python 3.13.16, clock perf_counter resolution 1e-09 s
```

This is a **shared single-CPU-core cloud container** with other build jobs
running concurrently. It is a workstation-class machine and it is not an edge
target. The consequences are visible in the numbers below and are not smoothed
over:

- the p99/p50 tail ratio of a measured latency reaches 16.7 on the run recorded
  here (`validate_backend_cost_model_output.txt`), against an injected
  distribution whose true tail ratio is about 1.4;
- the arithmetic mean of a measured latency sample is contaminated by
  preemption and is not a usable estimator of a cost model's mean, which is why
  the cost-model recovery check is made on the median;
- a figure that depends on host load is marked as such wherever it appears, and
  will differ on a re-run.

**The Jetson Orin Nano column of every results table in this repository is
empty.** No number here has been extrapolated to that device, taken from a
vendor datasheet, or produced by the simulated backend. What is missing for a
hardware-measured validation level is listed in `docs/REQUIREMENTS.md` §4.

---

## 1. Summary of checks

| Script | Checks | Result | Runtime |
|---|---|---|---|
| `validate_opcounts.py` | 12 | 12 passed | 1 s |
| `validate_memory.py` | 8 | 8 passed | 5 s |
| `validate_onnxruntime.py` | 13 | 13 passed | 3 s |
| `validate_uncertainty.py` | 4 | 4 passed | 5 s |
| `validate_backend_cost_model.py` | 6 | 6 passed | 5 s |
| `validate_budget.py` | 21 | 21 passed | 4 s |
| `validate_performance.py` | 1 | 1 passed | 4 s |
| `validate_predictor.py` | campaign | completed; outcome below | 9 s |

Test suite: **467 tests, 467 passed, 0 failed, 0 skipped, 0 errors**, 39.7 s
(`python -m pytest tests/ -q` from `products/P033/`). `ruff check src/ tests/`
clean.

---

## 2. Analytic operation counts against hand counts

`validate_opcounts.py` → `validate_opcounts_output.txt`. Integer equality,
tolerance 0. The hand derivations are printed in full by the script and repeated
as test-comment derivations in `tests/test_ops.py`.

**Network A — `X(1,4) → Gemm(4,6)+b → Relu → Gemm(6,3)+b → Y`, float32**

| Quantity | Hand count | Computed | Tolerance | Result |
|---|---|---|---|---|
| multiply-accumulates | 42 | 42 | 0 | PASS |
| floating-point operations | 99 | 99 | 0 | PASS |
| compulsory traffic [B] | 328 | 328 | 0 | PASS |
| resident weight bytes [B] | 204 | 204 | 0 | PASS |
| peak activation bytes [B] | 48 | 48 | 0 | PASS |

**Network B — `X(1,1,6,6) → Conv(1→2,3×3,pad 1) → Relu → MaxPool(2×2,s2) →
Reshape(1,18) → Gemm(18,2)+b → Y`, float32**

| Quantity | Hand count | Computed | Tolerance | Result |
|---|---|---|---|---|
| multiply-accumulates | 684 | 684 | 0 | PASS |
| floating-point operations | 1514 | 1514 | 0 | PASS |
| compulsory traffic [B] | 1832 | 1832 | 0 | PASS |
| resident weight bytes [B] | 240 | 240 | 0 | PASS |
| peak activation bytes [B] | 576 | 576 | 0 | PASS |

Both graphs were also executed in `onnxruntime` and compared against an
independent NumPy recomputation (the convolution written out as explicit loops,
sharing no code with either the runtime or the cost model):

| Network | max abs difference | Tolerance | Result |
|---|---|---|---|
| hand MLP | 0.000e+00 | 1e-05 | PASS |
| hand CNN | 2.384e-07 | 1e-05 | PASS |

Sources for the formulas under test: Golub & Van Loan 2013 §1.1.11 (dense
matrix product, `2mnk` flops); Sze, Chen, Yang & Emer 2017, *Proc. IEEE* 105(12)
§II-A (convolution MAC count); ONNX Standard opset 17 `Conv` (output spatial
size); Aho, Lam, Sethi & Ullman 2006 §8.4 (liveness analysis).

---

## 3. Peak-memory measurement against known allocation patterns

`validate_memory.py` → `validate_memory_output.txt`.

**(a) The instrument.** `tracemalloc` against hand-counted allocations. The
allowance is 1024 B for NumPy object headers; the payload is asserted as an
exact lower bound.

| Pattern | Hand count [B] | Measured [B] | Δ [B] | Result |
|---|---|---|---|---|
| one float64 array, 1e6 elements | 8 000 000 | 8 000 272 | +272 | PASS |
| two arrays live together, 1e6 + 2e6 | 24 000 000 | 24 000 440 | +440 | PASS |
| two arrays in sequence, first released | 8 000 000 | 8 000 296 | +296 (not 16 MB) | PASS |
| float64/float32 peak ratio | 2.0000 | 1.9999 | — | PASS |
| nested regions, inner peak | 2 400 000 | 2 400 208 | +208 | PASS |

**(b) The analytic liveness model** against a measured NumPy execution of the
same dataflow, with tensors released exactly where the liveness analysis says
they die:

| Graph | Analytic peak [B] | Measured [B] | Allowance [B] | Result |
|---|---|---|---|---|
| hand MLP | 48 | 568 | 4096 (one header per live tensor) | PASS |
| hand CNN | 576 | 1160 | 6144 | PASS |
| 512-2048-2048-64 MLP | 16 384 | 16 904 | 3.17 % relative excess | PASS |

**(c) What is not measured.** `onnxruntime` allocates its arena outside the
CPython allocator. For a 256-1024-1024-64 MLP the analytic peak memory is
5 521 664 B (5 513 472 B weights + 8 192 B activations) while `tracemalloc`
across one `session.run()` sees 4 956 B — the Python-side output marshalling.
These are different quantities and neither validates the other. Measuring a real
session's peak working set needs an allocator-level instrument on the target,
which this repository does not have.

---

## 4. Measured latency reproduces an injected synthetic cost model

`validate_backend_cost_model.py` → `validate_backend_cost_model_output.txt`.

**(a) Constant-duration stages**, isolating the spin-wait overshoot of the
simulated backend (300 repeats each, 20 warm-up):

| Injected [µs] | Measured p50 [µs] | Overshoot [ns] | Ratio | Result |
|---|---|---|---|---|
| 50 | 50.802 | 802 | 1.0160 | PASS |
| 200 | 201.265 | 1265 | 1.0063 | PASS |
| 800 | 802.550 | 2550 | 1.0032 | PASS |

The overshoot is a fixed per-call cost of the backend, not of the harness, so it
matters relatively more for a short pipeline.

**(b) The cross-check pipeline**, three lognormal stages consuming real time,
n = 2000 repeats after 20 warm-up:

| Quantity | Injected | Measured | Ratio | Asserted |
|---|---|---|---|---|
| mean [µs] | 270.000 | 592.236 | 2.1935 | no — contaminated |
| trimmed mean, lowest 90 % [µs] | 270.000 | 281.2 | 1.0413 | no — diagnostic |
| **p50 [µs]** | **270.000** | **283.862** | **1.0513** | **yes, band [1.00, 1.15] — PASS** |
| sd [µs] | 42.190 | 1129.982 | 26.78 | no — preemption inflates it |
| p99/p50 | ≈1.4 | 16.700 | — | no — a property of the host |

**This is a finding, not a nuisance.** On a shared single core the arithmetic
mean of a measured latency sample is not a usable estimator of a cost model's
mean: a handful of multi-millisecond preempted repeats drag it to more than
twice the truth, while the median sits 5.1 % above it, consistent with the
measured spin overshoot. The recovery check is therefore made on the median, and
the mean and trimmed mean are reported so the contamination is visible. The
ratio between them moves from run to run with host load; on an earlier run in
the same session the mean ratio was 1.09 and the median ratio 1.045.

Quantile ordering `p50 ≤ p90 ≤ p99 ≤ max` held. Uncertainty budget for that run:
u_c(mean) 25.27 µs, u_c(p50) 1.29 µs, u_c(p99) 185.79 µs; timer-pair bias 45 ns,
stated and not corrected. The median bias against the injected mean was
+13.86 µs, i.e. +4.6 µs per stage.

**Cross-check export.** `validation/crosscheck_pipeline.json` carries the
pipeline definition and the measured result in the schema the batch
specification fixes, for P039 LatencyNet to diff against:

```json
{"stages": [{"name": "preprocess", "mean_s": 6e-05, "std_s": 1.2e-05, "dist": "lognormal"},
            {"name": "inference",  "mean_s": 0.00018, "std_s": 4e-05, "dist": "lognormal"},
            {"name": "postprocess","mean_s": 3e-05, "std_s": 6e-06, "dist": "lognormal"}],
 "n_samples": 2000, "seed": 20260401,
 "measured": {"mean_s": ..., "p50_s": ..., "p99_s": ...}}
```

The file also records `"comparable_statistic": "p50_s"` and the reason: **diff
p50, not mean or p99.** The pipeline is fully reproducible from the stated
stages plus the seed — one lognormal draw per stage per sample from
`numpy.random.default_rng(seed)` in stage order, with lognormal parameters from
the declared mean and sd by `σ² = log1p((sd/mean)²)`, `µ = log(mean) − σ²/2`
(Johnson, Kotz & Balakrishnan 1994 ch. 14). An independent implementation that
disagrees on p50 by more than a few per cent has found a real discrepancy; one
that disagrees on mean or p99 has found a difference between two hosts.

---

## 5. Uncertainty budget over timing

`validate_uncertainty.py` → `validate_uncertainty_output.txt`.

| Check | Value | Reference | Result |
|---|---|---|---|
| `perf_counter` resolution | 1.000e-09 s | `time.get_clock_info` | — |
| u_B for a start/stop pair | 4.082e-10 s | GUM §4.3.7, `d·sqrt(2/12)` | PASS (closed form to 1e-24) |
| measured timer-pair cost | 46 ns (median of 5000 pairs) | measured here | PASS |
| timer cost / clock resolution | 46× | — | the call cost, not quantisation, is the dominant instrumental effect |

**Coverage.** 24 independent batches of 150 repeats each, same callable:

| Statistic | Realised spread across batches | Predicted within-batch u_A | Ratio |
|---|---|---|---|
| batch mean | 24.06 µs | 58.99 µs | 0.408 |
| batch p99 | 816.90 µs | 1015.27 µs | 0.805 |

On this run the realised spread was *smaller* than the within-batch prediction:
a few preempted repeats inflated the within-batch standard deviation while the
batch means stayed close. On a busier run earlier in the same session the ratio
was 3.19 in the other direction. **Neither direction is a property of the
method** — the ratio moves with host load, and that is the point: a figure
quoted from a shared machine carries the machine's load with it. A quoted u_c is
the uncertainty of a *statistic*, not a bound on the spread of the distribution.

**Scope.** For a 1500-repeat profile the sample standard deviation was 39× the
combined standard uncertainty of the mean. A budget written against u_c would be
written against the wrong number by that factor; the distribution's width is
what a deadline has to survive, and that is what p99 and max are for.

---

## 6. The budget decision path and the four failure modes

`validate_budget.py` → `validate_budget_output.txt`. 21/21 checks passed.

| Failure mode | What was checked | Result |
|---|---|---|
| model over the memory budget | 256-1024-1024-64 MLP, analytic peak 5 521 664 B against a 1 048 576 B ceiling (526.6 % utilisation): memory row FAIL, latency row PASS, overall FAIL, decided with no measurement | PASS |
| unsupported operator | `GRU`, `Einsum`, `Loop`, `Transpose` each refused by both shape inference and cost evaluation, naming the operator and the supported set | PASS (4/4) |
| thermally throttled device | declared clock factors 1.00/0.70/0.30 and 0.30 with a 0.50 bandwidth derate: verdicts PASS/PASS/FAIL/FAIL against a 2× nominal ceiling | PASS (4/4) |
| budget infeasible by construction | three contradictory declarations each detected with the contradiction named, and failing even against a candidate with 1 ps latency; a self-consistent budget reported feasible | PASS (5/5) |

Two further cases, which are why uncertainty is carried at all:

| Case | Result |
|---|---|
| ceiling one combined standard uncertainty above the measured p99 | `MARGINAL PASS`, not `PASS` |
| ceiling 100× the measured p99 | `PASS` |
| ceiling between the measured p50 and p99, budget self-consistent | median row `PASS`, worst-case row `MARGINAL FAIL`, overall `MARGINAL FAIL` |

The last row is the case the whole separation exists for: a tool reporting only
the median would have passed that candidate.

---

## 7. The ONNX writer against artefacts this package did not produce

`validate_onnxruntime.py` → `validate_onnxruntime_output.txt`. 13/13 passed.

`edgeinfer` encodes and decodes ONNX ModelProto with a hand-written protobuf
codec, because the `onnx` package is not installed in this environment. Its
field numbers are verified by parsing `mul_1.onnx` and `sigmoid.onnx` shipped
inside the installed `onnxruntime` 1.29.0 wheel — files produced by the ONNX
project, not by this repository — and reading seven distinct protobuf fields out
of them, including the initialiser values 1.0…6.0 stored as packed `float_data`.
A wrong field number would surface as a wrong value, not a silent default.

`logreg_iris.onnx` is **refused** by the parser, with the diagnostic
`value_info 'probabilities': only tensor_type is supported; maps, sequences and
optionals are not representable in edgeinfer's IR`. That is a documented
limitation reported as a refusal rather than a graph with pieces missing.

A `build → parse → build` round trip produces byte-identical ONNX. This caught a
real defect during development: the `Reshape` target shape is an input tensor in
ONNX and an attribute in `edgeinfer`'s IR, so a re-parsed graph could not be
shape-inferred at all. `parse_model` now recovers it from the constant
initialiser, and a `Reshape` whose target shape is computed at run time is
refused rather than guessed.

Measured latency, n = 300 repeats after 20 warm-up, with p50 and p99 separate:

| Model | p50 [µs] | p99 [µs] | max [µs] | p99/p50 | u_c(p50) [µs] |
|---|---|---|---|---|---|
| `mul_1.onnx` (shipped) | 5.222 | 11.274 | 44.006 | 2.16 | 0.0048 |
| `sigmoid.onnx` (shipped) | 5.007 | 12.692 | 33.408 | 2.53 | 0.0032 |
| hand MLP (built here) | 6.399 | 33.901 | 3874.878 | 5.30 | 0.0041 |
| hand CNN (built here) | 8.838 | 44.408 | 4105.320 | 5.02 | 0.0066 |

These are dominated by the per-call cost of `onnxruntime`'s Python binding, not
by the models' arithmetic — the largest of these graphs does 1514 flops. The
maxima of 3.9 ms and 4.1 ms are scheduler preemptions on a shared core.

---

## 8. The analytic baseline against the learned predictor

`validate_predictor.py` → `validate_predictor_output.txt`.

Population: 120 synthetic ONNX graphs (60 MLP, 60 CNN; 3 to 11 nodes;
1.7e3 to 3.9e7 flops), seed 20260401. Split 70/30, seed 20260401, 84 train and
36 held out. Each model measured with 60 repeats after 8 warm-up calls.

**Is the roofline a bound at all?** With peaks deliberately over-declared at
200 GFLOP/s and 400 GB/s, the pure bound lay below the measured latency for
36/36 held-out models for both the p50 and the p99 target; measured/bound was
3.7× to 350× with a median of 12.8× on the p50 target. **PASS.** With the
plausible declaration of 4 GFLOP/s and 10 GB/s the bound sits *above* the
measurement for 31/36 models, because those declared peaks are
slower than this host's effective rate. A roofline bound is a bound for the
device it declares, not for the machine it is evaluated on; that is why the
baseline below is fitted rather than declared.

**The fit.** Four parameters on the training split: peak compute, peak
bandwidth, per-node dispatch overhead and constant per-call overhead, by grid
search over the two peaks with a non-negative least-squares solve for the two
linear terms. On the recorded run the p50 fit gave 43.4 GFLOP/s, 78.0 GB/s,
1.80 µs per node and 0.96 µs fixed (ridge point 0.557 FLOP/B), labelled
`calibrated-fit`. The forest is the
package default: 120 trees, depth ≤ 8, fitted on `log10` latency from 17 graph
features.

**Single pass, held out, median and p90 of |relative error|:**

| Target | analytic median | learned median | analytic p90 | learned p90 | analytic ρ | learned ρ |
|---|---|---|---|---|---|---|
| p50 latency | 25.08 % | 16.95 % | 46.16 % | 43.66 % | 0.9243 | 0.9593 |
| p99 latency | 22.80 % | 45.58 % | 86.58 % | 108.57 % | 0.8332 | 0.8489 |

**Five independent passes** — each re-measures all 120 models and refits both
predictors from scratch. This is the figure to believe, because a single
measured latency on a shared core is not a fixed quantity:

| Target | learned wins | analytic wins | no measurable advantage | analytic median range | learned median range |
|---|---|---|---|---|---|
| **p50 latency** | **5/5** | 0/5 | 0/5 | 21.20 – 24.96 % | 8.42 – 15.20 % |
| **p99 latency** | 0/5 | 0/5 | **5/5** | 30.55 – 43.71 % | 33.49 – 51.25 % |
| **peak memory** | 0 | **exact** | — | 0.00 % by construction | 11.17 % median, 62.65 % p90 |

**The result, plainly.**

1. **On median latency the learned model wins**, consistently: it won all five
   passes, with a median absolute relative error of 8–15 % against the analytic
   baseline's 21–25 %.
2. **On worst-case latency neither model wins.** Every one of the five passes
   returned *no measurable advantage* — the difference in median error lay
   inside the paired-bootstrap standard deviation. Both predictors are poor
   there: 31–44 % and 33–51 % median error, with p90 errors of 87 % and 109 %
   on the recorded pass. The tail is dominated by scheduler preemption, which is
   a property of the host and not of the model graph, so no graph feature
   predicts it.
3. **On peak memory the analytic model wins outright.** It is exact integer
   arithmetic over the graph's tensor shapes plus a liveness analysis; its error
   is zero by construction. A forest trained on the same population with the two
   memory features ablated reached 11.17 % median and 62.65 % p90 error — worse
   by those margins, and it needs the population to have been measured first.

No retuning was done to change any of these outcomes. The analytic model was
written and validated first; both predictors saw the same features and the same
training set; the forest's hyperparameters are the package defaults and the
calibration's grid size is declared in the script.

**The learned model's uncertainty is not calibrated, and that is measured.**
On the p50 target the ensemble-disagreement spread had a median relative size of
34.3 %, and the truth fell inside ±1 u for 75.0 % of held-out models and inside
±2 u for 91.7 %. On the p99 target the spread was 53.8 % and the coverages were
72.2 % and 86.1 %. For a calibrated Gaussian interval those would be about 68 %
and 95 %; the ±2 u figures are short, because the spread measures disagreement
between trees and not the irreducible measurement noise.

---

## 9. Performance of the package itself

`validate_performance.py` → `validate_performance_output.txt` and
`performance_results.md` (which carries the empty device column).

| Quantity | This host [s] | Method |
|---|---|---|
| generate the 120-model population | 0.314 | wall clock, 1 repeat, seed 20260401 |
| analytic estimate, one graph (p50) | 4.412e-05 | `perf_counter`, n=200 repeats |
| analytic estimate, one graph (p99) | 8.267e-05 | `perf_counter`, n=200 repeats |
| featurise 120 graphs | 0.0111 | wall clock, 1 repeat |
| calibrate 4 parameters, grid 20×20 | 0.0113 | wall clock, 1 repeat, 400 NNLS solves |
| train the forest, 120 trees | 0.1425 | wall clock, 1 repeat |
| forest predict, 120 rows (p50) | 0.00824 | `perf_counter`, n=100 repeats |
| forest predict with uncertainty, 1 row (p50) | 0.01235 | `perf_counter`, n=100 repeats |
| build one `InferenceSession` | 0.0289 | wall clock, 1 repeat |
| measured profile of one model, 60 repeats | 0.00187 | wall clock, 1 repeat |
| **screen all 120 analytically** | **0.00771** | wall clock, 1 repeat |
| **screen all 120 by measurement** | **3.697 (extrapolated)** | 0.0308 s per candidate × 120 |

Screening analytically is **479× cheaper** than screening by measurement on this
host, because loading a candidate into a session costs 28.9 ms against an
analytic estimate's 64 µs. That ratio is the only extrapolation in the
performance file and it extrapolates this repository's own cost, not any
device's performance.

Compute budget: the script itself took 2.78 s against a per-script ceiling of
180 s.

---

## 10. What failed, what is a known limitation, and what is absent

**Failed during development and was fixed, with the fix tested:**

- `build → parse → build` was not byte-identical and a re-parsed graph could not
  be shape-inferred, because the `Reshape` target shape is an ONNX *input* and
  an `edgeinfer` *attribute*. Fixed by recovering it from the constant
  initialiser in `parse_model`; `tests/test_onnx_io.py::TestBuildModel::
  test_build_parse_build_is_byte_identical` and
  `::test_a_reshape_with_no_static_target_shape_is_refused` cover it.
- `Squeeze`, `Unsqueeze` and `Transpose` were listed as supported operators
  while their output-shape rules were not implemented, so they would have
  reported wrong downstream shapes. They were removed from `SUPPORTED_OPS` and
  are now refused.
- The first cost-model recovery check asserted on the arithmetic mean and
  **failed** at a measured/injected ratio of 1.494. The failure was correct: the
  mean is contaminated by preemption. The check was moved to the median, which
  is robust to a contaminated tail; the mean and trimmed mean are still reported
  and never asserted. The band was not widened.

**Known limitations, each measured rather than asserted:**

- The analytic peak memory is a **lower bound** on real peak memory: no
  allocator arena, no kernel scratch buffer, no alignment or page granularity,
  and a tensor freed the instant its last consumer finishes (§3c).
- Compulsory traffic counts each tensor once, so the memory-bound side of the
  roofline is a lower bound on real traffic and an upper bound on attainable
  performance.
- The fitted bandwidth is **not identifiable** from a calibration set in which
  every node is compute bound; this is asserted directly in
  `tests/test_roofline.py::TestCalibration::
  test_bandwidth_is_not_identifiable_from_a_compute_bound_population`.
- `Sigmoid`/`Tanh` are charged 4 flops per element and `Softmax` 5; these are
  stated modelling assumptions, not measurements.
- The population is synthetic, small and narrow — MLPs and small CNNs under
  about 40 MFLOP, float32, batch 1, `onnxruntime` CPU execution provider. A
  predictor result here transfers to another model family, runtime or device
  only as a hypothesis.
- p99 from n repeats is blind to an event rarer than 1 in 100; the maximum is
  reported alongside it and the repeat count travels with both
  (`tests/test_harness.py::TestLatencyProfile::
  test_p99_cannot_see_a_one_in_two_hundred_outlier_but_max_can`).

**Absent, and required for a hardware-measured validation level:** every item in
`docs/REQUIREMENTS.md` §4. In short: this package has never run on a Jetson Orin
Nano, no number in this repository came from one, and the device column stays
empty until raw output from the device is provided to a session.

---

## 11. Reproducing every number

From `products/P033/`:

```bash
pip install -e ".[dev]"
python -m pytest tests/ -q
ruff check src/ tests/
for s in validate_opcounts validate_memory validate_onnxruntime \
         validate_uncertainty validate_backend_cost_model validate_budget \
         validate_performance validate_predictor; do
    python validation/$s.py | tee validation/${s}_output.txt
done
python examples/budget_check.py
python examples/roofline_and_tail.py
python examples/analytic_vs_learned.py
python examples/failure_modes.py
```

Integer counts (§2), allocator byte counts (§3a, §3b) and the protobuf field
checks (§7) are host-independent and will reproduce exactly. Every latency, and
every figure derived from one, will differ: it depends on the host and on its
load at the time. The *direction* of the §8 result has been stable across five
passes within one session; a reader on different hardware should re-run the
five-pass study rather than quote these percentages.
