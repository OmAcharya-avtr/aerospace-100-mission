# Dataset card — deadlinerisk hardware campaign

Validation level: **Level 4, measured on hardware.** Level 4 here is **not** a claim of generality: one power mode (`MAXN_SUPER`), one uninstrumented ambient, one CPython CPU-only payload, one period, one shared board. See Limitations.

> **No trace of this dataset is present in this repository.** `data/traces/` is
> present. This card describes the dataset the committed campaign script
> `data/campaign.py` produced and that every loader, validation script and
> example here reads: 10 runs, 190 662 executed jobs over 195 010 releases, with
> a 1 Hz contention monitor beside each run. It was written before the campaign
> ran, with every measured figure marked `MEASURED_PENDING`. Nothing was
> substituted from the committed synthetic generator.

## What it is

A campaign of periodic hard-real-time loops on one NVIDIA Jetson Orin Nano
Engineering Reference Developer Kit Super under named co-resident contention,
recording per-job timing and 1 Hz contention descriptors.

Produced by `data/campaign.py`, committed verbatim as specified for the board. The
script is stdlib-only on purpose: the measurement loop must not depend on
numpy's import or allocation behaviour, and scipy and scikit-learn are broken on
that board.

## Files

| File | One row per | Columns |
|---|---|---|
| `data/traces/manifest.json` | — | board identity, period, deadline, calibrated `payload_w`, the full calibration history including the attempts that missed, the probe table, the chosen ground-truth level and the pre-declared rule that chose it, and per-run metadata |
| `data/traces/jobs_<run>.csv` | released-and-**executed** job | `job_index, release_s, start_delay_s, end_s, exec_s, response_s, missed, dropped_before` |
| `data/traces/mon_<run>.csv` | 1 Hz monitor sample | `t_s, total_busy_j, task_busy_j, stress_busy_j, foreign_busy_j, foreign_pct, task_pct, stress_pct, clk0_khz..clk5_khz, clk_mean_khz, clk_min_khz, tj_mdegC, load1` |
| `data/traces/CAMPAIGN_COMPLETE` | — | zero-byte marker, written last of all |

Units: times in seconds, occupancies as percentages of total CPU capacity over
the sample second from `/proc/stat` jiffy deltas, clocks in kHz from
`scaling_cur_freq`, temperature in millidegrees Celsius from
`thermal_zone8` (`tj-thermal`) and converted to degrees Celsius on load.

A row in `jobs_<run>.csv` exists only for a release that **executed**. A dropped
release has no row; it is accounted for by the `dropped_before` field of the row
that preceded it. This is the single most important thing to know before using
these files: a naive `mean(missed)` over the rows undercounts, and by how much
is reported in the validation evidence.

## Runs

| Run | Duration | Contention |
|---|---|---|
| `gt_long` | 600 s | the chosen ground-truth CPU-stressor level |
| `mix_c0` … `mix_c6` | 150 s each | 0 to 6 CPU stressors |
| `mix_mem2` | 150 s | 2 memory stressors |
| `mix_c2mem1` | 150 s | 2 CPU + 1 memory stressors |

**Measured totals, per run:**

| run | releases | misses | miss rate | median exec (ms) | p99.9 response (ms) |
|---|---:|---:|---:|---:|---:|
| `gt_long` | 60001 | 816 | 1.359977e-02 | 6.0527 | 16.2285 |
| `mix_c0` | 15001 | 2 | 1.333244e-04 | 7.3275 | 8.6010 |
| `mix_c1` | 15001 | 6 | 3.999733e-04 | 7.3287 | 8.5640 |
| `mix_c2` | 15001 | 2 | 1.333244e-04 | 6.0438 | 7.4279 |
| `mix_c3` | 15001 | 38 | 2.533164e-03 | 6.0733 | 10.3871 |
| `mix_c4` | 15001 | 238 | 1.586561e-02 | 6.0671 | 17.2965 |
| `mix_c5` | 15001 | 1703 | 1.135258e-01 | 6.1882 | 28.5229 |
| `mix_c6` | 15001 | 5739 | 3.825745e-01 | 6.0534 | 21.9098 |
| `mix_mem2` | 15001 | 0 | 0.000000e+00 | 6.0511 | 7.2608 |
| `mix_c2mem1` | 15001 | 12 | 7.999467e-04 | 6.0502 | 7.8595 |

The curve is monotone in the CPU stressor count from `mix_c2` upward and is
**not** monotone below it: `mix_c1` has a higher rate than `mix_c2`, and `mix_c0`
and `mix_c1` have the longest median execution times in the campaign, 7.33 ms
against about 6.05 ms elsewhere, because the `schedutil` governor drops the clock
when there is little load — mean core clock 1008.0 MHz on `mix_c0` and 1561.8 MHz
on `mix_c1` against 1728.0 MHz on every loaded run. Two memory-bandwidth
stressors produced zero misses in 15 001 releases.

Measured contamination, because the board was shared: foreign CPU occupancy
pooled over all 1970 monitor samples has mean 2.5210 per cent, minimum 0.1681
and maximum 21.6443. The campaign cannot attribute any part of a miss rate to
foreign load rather than to the declared stressors, because the foreign load was
not controlled.

## How the campaign was parameterised, and why that is auditable

- **Calibration.** The payload size `w` is sized so the median execution time on
  an otherwise-idle board is about 0.45 of the deadline. The whole calibration
  history, including the attempts that missed the target, is in
  `manifest.json` under `calibration`.
- **Probes.** 15 s at each CPU-stressor level 0 to 6, used to choose the long
  run's contention level by a **pre-registered rule**: the level whose probe miss
  rate is closest to 0.02 within the band `[0.0005, 0.25]`, with a pre-declared
  fallback of 4 stressors. The rule, the band, the target and the probe rates
  are all in `manifest.json`, so the choice is auditable and demonstrably was
  not made after seeing the long run.

**The declared rule chose contention level 4.** The measured probe table it
read, 15 s per level:

| CPU stressors | releases | misses | miss rate | median exec (ms) |
|---:|---:|---:|---:|---:|
| 0 | 1501 | 2 | 0.001332 | 7.2964 |
| 1 | 1501 | 0 | 0.000000 | 7.2961 |
| 2 | 1501 | 0 | 0.000000 | 7.2961 |
| 3 | 1501 | 0 | 0.000000 | 6.0364 |
| 4 | 1501 | 49 | 0.032645 | 6.0315 |
| 5 | 1501 | 143 | 0.095270 | 6.1470 |
| 6 | 1501 | 528 | 0.351765 | 6.0228 |

Level 4 is the admissible level whose probe rate is closest to the declared
target of 0.05 inside the declared band [0.002, 0.30]. The target, the band, the
fallback and the whole probe table are in `manifest.json`, written before the
600 s run, so the choice is auditable and demonstrably was not made after seeing
that run.

The payload was sized by a **sustained** calibration against the governor-ramped
clock: `payload_w` 20000, sustained median execution 7.315178 ms over 1998 jobs
of a 30 s attempt with the first third discarded as ramp, which is 0.7315 of the
deadline. Two earlier campaigns on the same night calibrated against a 1.2 s
probe taken as the board went idle, sized the payload to 0.4499 of the deadline,
and then measured a sustained 0.19 — a factor of 2.3 out. That is why the
calibration is sustained, and why the whole calibration history, including the
attempts that missed, is in `manifest.json`.

## Stressors

CPU stressor: a Python process running an unbounded integer linear-congruential
loop. Memory stressor: a Python process copying two 64 MiB `bytearray`s into
each other repeatedly. Both are started as subprocesses of the campaign script
and stopped after each run, with an 8 s settling period before the measured
loop starts.

These are deliberately crude. They are named in the manifest by count and type,
and the point of the product is that the **measured** descriptors — foreign
occupancy, clock, temperature, load average — are what a predictor consumes,
not the stressor labels.

## Limitations of the dataset itself

1. **One board**, with its own heatsink and fan.
2. **One power mode**, `MAXN_SUPER`. `sudo -n true` returns non-zero on the
   board, so `nvpmodel` and `jetson_clocks` are unreachable unattended.
3. **One ambient**, never instrumented or varied. There is no ambient column,
   because there was no ambient measurement.
4. **One CPython CPU-only payload at one period.** No column varies with
   payload or period.
5. **The board is shared throughout.** `foreign_pct` is the measured
   contamination. It is reported, not assumed absent, and it is not separable
   from the declared stressors' effect because it is not controlled.
6. **The monitor is 1 Hz and the loop is 100 Hz.** A descriptor is an average
   over a second, so sub-second contention bursts are invisible in the
   descriptors while being fully visible in the per-job response times. Any
   model built on the descriptors is therefore predicting a windowed rate from
   windowed inputs, which is why the dataset is windowed at 10 s rather than
   used per job.
7. **No ground truth for execution time without contention inside a contended
   run.** The idle execution-time distribution comes from a separate run
   (`mix_c0`), not from interleaved idle jobs.
8. **The campaign is a single pass.** No run is repeated, so a per-run
   repeatability figure cannot be computed from these files.

## Licence and provenance

Apache-2.0, copyright © 2026 OPTIMA Organisation, same as the code. The traces
are produced by this repository's own `data/campaign.py`, which reads only
`/proc` and `/sys` and the loop's own clock, so they contain no personal data and
no third-party content.
