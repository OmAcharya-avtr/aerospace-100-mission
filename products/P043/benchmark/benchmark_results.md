# photoncount benchmark results

**These numbers are not a Level 4 measurement.** They were produced on the machine described below. Level 4 for this product requires this same script run on a Jetson Orin Nano with its raw output kept; no figure here may stand in for that.

## Environment

| key | value |
|---|---|
| `timestamp_utc` | 2026-10-06T04:23:40Z |
| `python` | 3.13.16 |
| `platform` | Linux-6.18.44-fc-v70-x86_64-with-glibc2.39 |
| `machine` | x86_64 |
| `processor` | x86_64 |
| `cpu_count` | 2 |
| `numpy` | 2.5.3 |
| `scipy` | 1.18.1 |
| `scikit_learn` | 1.9.1 |
| `photoncount` | 0.1.0 |
| `measured_on` | shared cloud build container, two cores, running concurrently with other build jobs. NOT a Jetson Orin Nano and NOT a flight-representative board. |

## Method

Each timed operation is run 3 times to warm up, then repeated with `time.perf_counter()` around each call and `gc.collect()` once before the measured loop. Percentiles are over the repeats, not over an internal loop, so a single slow call shows up in p99 rather than being averaged away. Memory is reported both as the `tracemalloc` peak of the operation (Python allocations only) and as the process-wide maximum resident set size from `resource.getrusage`, which never decreases and therefore bounds the whole run rather than the operation.

Workload: a paralyzable detector with tau = 50 ns, afterpulse probability 0.050 with a 200 ns mean delay, illuminated at 4e+06 counts/s (n tau = 0.200), read out in 1.0 ms windows. Rate corrector: validation/rate_corrector.joblib.

## Latency

| operation | repeats | mean (ms) | p50 (ms) | p90 (ms) | p99 (ms) | min (ms) | max (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `acquisition_one_window` | 60 | 1.6394 | 1.6160 | 1.7556 | 1.9149 | 1.5191 | 1.9930 |
| `inference_single_row` | 100 | 0.6354 | 0.6163 | 0.7580 | 1.0690 | 0.4958 | 1.2826 |
| `inference_batch_1000` | 20 | 8.7740 | 8.5773 | 9.8784 | 10.2957 | 8.2860 | 10.3851 |
| `ppm_exact_error_M256` | 50 | 0.5077 | 0.3870 | 0.4946 | 3.1218 | 0.3482 | 3.4983 |
| `bit_llrs_10000_symbols` | 30 | 149.3768 | 149.7213 | 161.8618 | 168.2672 | 127.4177 | 169.5543 |
| `simulator_one_window` | 60 | 2.0064 | 1.8314 | 2.3546 | 4.1231 | 1.4156 | 4.2128 |

## Throughput

| quantity | value |
|---|---:|
| `observed_rate_hz` | 3.36645e+06 |
| `registered_events_per_window` | 3366.45 |
| `simulated_events_per_second_of_cpu` | 1.83817e+06 |
| `acquisition_realtime_factor` | 0.618814 |
| `inference_rows_per_second_batched` | 116587 |
| `ppm_symbols_per_second_soft_metrics` | 66790.8 |

`acquisition_realtime_factor` is the ratio of simulated window length to the wall-clock time taken to produce it. Above 1 means the simulator runs faster than real time at this loading; it falls as the loading rises, because the event loop is linear in events.

## Memory

| quantity | KiB |
|---|---:|
| `acquisition_one_window_peak_kib` | 98.2 |
| `inference_batch_1000_peak_kib` | 58.0 |
| `bit_llrs_10000_symbols_peak_kib` | 35940.7 |
| `process_max_rss_kib` | 201624.0 |

## What is still missing for Level 4

- Latency, memory and throughput from a Jetson Orin Nano, produced by this script on that board.
- A measured dead time, dead-time model, afterpulse probability and delay distribution, dark-count rate, timestamp resolution and jitter from the actual detector, replacing the declared values in `photoncount.hal.DeviceBackend`.
- A `DeviceBackend` implementation of `open`, `self_test` and `acquire` against that hardware.

No extrapolation from this container, and no vendor datasheet, substitutes for any of the three.
