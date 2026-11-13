# Dataset card — deadlinerisk hardware campaign

Validation level: **Level 3, hardware-pending.**

> **No trace of this dataset is present in this repository.** `data/traces/` is
> empty. This card describes the dataset the committed campaign script
> `data/campaign.py` produces and that every loader, validation script and
> example here is written against; it does not describe files that exist. Every
> figure that would come from the data is marked `MEASURED_PENDING`. Nothing was
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

MEASURED_PENDING — the hardware campaign had not written any file into `data/traces/` when this document was generated, so there is no measured number to put here and none was invented. This slot is filled by `validation/validate_measured_campaign.py`, which refuses with a non-zero exit until `data/traces/CAMPAIGN_COMPLETE` exists.

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

MEASURED_PENDING — the hardware campaign had not written any file into `data/traces/` when this document was generated, so there is no measured number to put here and none was invented. This slot is filled by `validation/validate_measured_campaign.py`, which refuses with a non-zero exit until `data/traces/CAMPAIGN_COMPLETE` exists.

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
