# P033 edgeinfer -- performance results

## Environment (applies to every number in this file)

Linux-6.18.44-fc-v64-x86_64-with-glibc2.39 x86_64, 1 CPU core(s) available of 1 logical, shared host, Python 3.13.16, clock perf_counter resolution 1e-09 s. 

All numbers in this file are WORKSTATION numbers measured in a shared, single-CPU-core cloud container with other build jobs running concurrently. They are not measurements of any edge or flight target. The target-device column is empty because no measurement from that device exists; it must be filled only from raw output produced on the device itself.

## Results

| Quantity | Unit | This host (shared 1-core container) | u (std) | Jetson Orin Nano (measured on device) |
|---|---|---|---|---|
| generate 120-model population | s | 0.314207 | - | (not measured) |
| analytic estimate, one graph (p50) | s | 4.412e-05 | - | (not measured) |
| analytic estimate, one graph (p99) | s | 8.26698e-05 | - | (not measured) |
| featurise 120 graphs | s | 0.0111126 | - | (not measured) |
| calibrate analytic baseline (4 params, grid 20x20) | s | 0.0112591 | - | (not measured) |
| train random forest (120 trees) | s | 0.142545 | - | (not measured) |
| forest predict, 120 rows (p50) | s | 0.00824024 | - | (not measured) |
| forest predict with uncertainty, 1 row (p50) | s | 0.0123494 | - | (not measured) |
| build one onnxruntime InferenceSession | s | 0.0289388 | - | (not measured) |
| measured profile of one model (60 repeats) | s | 0.0018686 | - | (not measured) |
| analytic estimate for all 120 graphs | s | 0.00771223 | - | (not measured) |
| measure one candidate (session build + 60-repeat profile) | s | 0.0308074 | - | (not measured) |
| this validation script, end to end | s | 2.78317 | - | (not measured) |

Measurement methods:
- generate 120-model population: wall clock, 1 repeat, seed 20260401
- analytic estimate, one graph (p50): perf_counter per-call bracket; n=200 repeats, 1 warm-up discarded; clock resolution 1e-09 s; timer-pair bias 45 ns; gc disabled=True
- analytic estimate, one graph (p99): perf_counter per-call bracket; n=200 repeats, 1 warm-up discarded; clock resolution 1e-09 s; timer-pair bias 45 ns; gc disabled=True
- featurise 120 graphs: wall clock, 1 repeat
- calibrate analytic baseline (4 params, grid 20x20): wall clock, 1 repeat, 400 NNLS solves
- train random forest (120 trees): wall clock, 1 repeat, n=120 training rows
- forest predict, 120 rows (p50): perf_counter per-call bracket; n=100 repeats, 1 warm-up discarded; clock resolution 1e-09 s; timer-pair bias 45 ns; gc disabled=True
- forest predict with uncertainty, 1 row (p50): perf_counter per-call bracket; n=100 repeats, 1 warm-up discarded; clock resolution 1e-09 s; timer-pair bias 72 ns; gc disabled=True
- build one onnxruntime InferenceSession: wall clock, 1 repeat
- measured profile of one model (60 repeats): wall clock, 1 repeat, 60 timed calls plus 8 warm-up
- analytic estimate for all 120 graphs: wall clock, 1 repeat
- measure one candidate (session build + 60-repeat profile): wall clock, 1 repeat; sum of the two rows above
- this validation script, end to end: wall clock, 1 repeat

## What the empty column needs

The Jetson Orin Nano column is empty because no measurement from that device exists. To fill it: run `validation/validate_performance.py` on the device, capture its raw stdout, and transcribe the figures from that output into this column. No number may be extrapolated from the host column, taken from a vendor datasheet, or produced by the simulated backend. Until then this product is `Level 3, hardware-pending` and the mission's Level 4 count stays at zero.

## How to read these numbers

Every row is the cost of *using* edgeinfer, not the latency of any model. The one model latency that appears in the text above (p50 11.316 us, p99 34.005 us over 60 repeats) is a property of this shared container and of onnxruntime's CPU execution provider, and it is not quoted in the table for that reason.
