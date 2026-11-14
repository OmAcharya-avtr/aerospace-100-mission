# Dataset card — the P093 PowerEnvelope campaign traces

**Status: TESTING.** The campaign described here completed on 2026-11-14 and
its output is committed under `data/traces/`: 125 files, 19 MB, 20 of 20 planned
runs, `aborted: false`, `CAMPAIGN_COMPLETE` present, all 40 subprocess return
codes zero and every captured `.stderr` empty.

## What the dataset is

The output of one hardware measurement campaign, run by the committed program
`data/campaign.py` (byte-identical to `scripts/p093_campaign.py` in the mission
repository), on one NVIDIA Jetson Orin Nano. It is the only measured input to
this product; everything else in the repository is code, documentation, or the
output of the committed synthetic generator, which is labelled synthetic
wherever it appears.

## The board

| property | value |
|---|---|
| board | NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super |
| architecture | aarch64, 6 CPU cores |
| power mode | `MAXN_SUPER` — the only one measured |
| governor | `schedutil` |
| kernel, uname | `Linux Project2025 5.15.148-tegra #1 SMP PREEMPT Thu Sep 18 15:08:33 PDT 2025 aarch64` |
| `cpuinfo_max_freq` | 1,728,000 kHz |
| on-board Python | 3.10.12 (the campaign script is stdlib-only) |
| 1-minute load average, campaign start → end | 2.324 → 1.050 |
| `sudo -n true` | returns a password prompt, so `nvpmodel` and `jetson_clocks` are unreachable unattended |

## The instrument

The on-board **INA3221**, three channels, read through
`/sys/class/hwmon/hwmon1`. Channel 1 is the platform input rail **VDD_IN**;
channels 2 and 3 are the two downstream rails the board exposes,
`VDD_CPU_GPU_CV` and `VDD_SOC`. Power is bus voltage (mV) times shunt current
(mA), divided by 1000, in mW.

Two properties of the instrument limit every number in this product and are
reported rather than smoothed over.

1. **The rail current quantises at 8 mA.** At the nominal 5 V rail that is a
   **40 mW least-significant bit** on VDD_IN. A uniform quantiser of width `q`
   has standard deviation `q / sqrt(12)`, so a single sample carries about
   11.75 mW of quantisation noise. `validation/validate_independent_bound.py`
   section 6 propagates that into an energy figure and reports it beside the
   energy, and notes that the `sqrt(n)` reduction over a window's samples is a
   **lower bound** rather than a correct propagation, for the reason below.
2. **The INA3221 averages 512 conversions in hardware** (`samples = 512` in the
   sysfs configuration, recorded in `manifest.json` under `ina3221`).
   Neighbouring 20 Hz samples therefore overlap inside the instrument and are
   not independent draws. Two consequences, both stated in code:
   - a bootstrap over samples within one run would report an interval narrower
     than the data supports, which is why the uncertainty on the headline is
     taken from the **between-run** spread of four repeats instead
     (`baselines.BetweenRunSpread`);
   - the *shape* of the wake transient is bandwidth-limited by the averaging, so
     `E_ramp` in Baseline 2 is a **bandwidth-limited estimate of excess energy**
     and never a rise time (`baselines.RampEnergy.bandwidth_limited`).

## The background, which is contaminated and is reported as contaminated

The board was **not idle** at campaign time. It had five logged-in users, a
desktop Chrome running for nearly two days, swap in use and a 1-minute load
average above 1. So:

- the `floor` cell is a **background floor on a shared machine**, not an idle
  floor, and `report.floor_lines` says so in every output that prints it;
- the foreign CPU occupancy is **measured once per second** rather than assumed
  absent, and the measured figure is printed whatever it is;
- the contamination is **not separable** from the platform's own idle draw by
  this campaign, and `docs/REQUIREMENTS.md` lists that as out of scope rather
  than approximating it away.

**Measured background floor power: 4264.428 mW** (sample sd 123.957 mW, minimum
4113.408 mW, maximum 5716.480 mW, over 1540 samples and 76.950 s).
**Measured foreign occupancy on the floor run: 0.123514**, mean die temperature
48.640 °C.

Regressing the floor run's per-second mean rail power on that second's
`/proc/stat` occupancy — in that run the pipeline is not running, so every busy
cycle is foreign work — over its 77 usable seconds gives

| quantity | value |
|---|---|
| slope | 2013.347 mW per unit occupancy |
| intercept at zero occupancy | 4015.319 mW (**extrapolated**; the lowest occupancy observed is 0.078859) |
| Pearson r | 0.510626 (r² = 0.260739) |
| implied foreign contribution at the mean occupancy | 248.676 mW = **5.832 per cent** of the measured floor |

The correlation is **weak**, and that is part of the result: about three
quarters of the floor's second-to-second variance is not explained by the
occupancy counter. The published headline uses the measured floor mean and
`validation/validate_measured_baselines.py` section 4 publishes a three-row
sensitivity table beside it.

## Structure

Per run cell, three CSVs; per campaign, one `manifest.json` and one
`campaign.log`. The schemas are declared once in
`src/powerenvelope/schema.py` and the loader rejects any file whose header is
not exactly the declared one, including a header with the right names in the
wrong order.

`power_<cell>.csv`, 20 Hz rail sampler:

```
t_mono,vdd_in_mv,vdd_in_ma,cpu_gpu_cv_mv,cpu_gpu_cv_ma,soc_mv,soc_ma,vdd_in_mw
```

`monitor_<cell>.csv`, 1 Hz environment monitor:

```
t_mono,cpu_busy_fraction,tj_c,load1,clk0_khz,clk1_khz,clk2_khz,clk3_khz,clk4_khz,clk5_khz
```

`decisions_<cell>.csv`, one row per inference:

```
t_start,t_end,cycle,frame_idx,score,n_candidates,area,touches_border,accepted,reject_reason
```

`t_mono`, `t_start` and `t_end` are `CLOCK_MONOTONIC` seconds. The sampler and
the pipeline are separate processes; `CLOCK_MONOTONIC` is a system-wide clock,
which is why intersecting their timestamps to derive the active windows is exact
and not an approximation. `accepted` is 0 or 1 and `reject_reason` is one of
`accept`, `no_candidate`, `border`, `area_out`, `low_score`, so the acceptance
rate is **reconstructible by counting** and is never asserted anywhere in this
product.

`manifest.json` carries the cycle length, the cycles per run, the sampler
period, the gate constants `grad_thr`, `area_ref`, `area_min` and `area_max`,
the declared `tau_rule`, the INA3221 `update_interval` and `samples`, the
environment block, a `calibration` list with one record per frame dimension
(the probe's latency and score quantiles and the chosen `tau_strict` and
`tau_permissive`), the `plan`, and a `runs` list with per-run `t0_mono`,
`t1_mono`, `epoch_at_t0`, `inferences`, `accepted`, `active_cpu_s`, `wall_s` and
both subprocess return codes.

`campaign.log` is evidence in its own right: it carries the warm-up, the
calibration probe table and the per-run completion lines in the order they
happened. The repository-level `.gitignore` has a bare `*.log`, which matches at
any depth; this product's `.gitignore` re-includes `/data/traces/campaign.log`
by negation so that it reaches a cold clone.

`CAMPAIGN_COMPLETE` is written **last of all**. Every measured script refuses
unless it exists, because a figure computed from a directory without it would be
a figure from a partial campaign.

## The cells

| group | cells | what they are for |
|---|---|---|
| background floor | `floor` | the sampler and monitor run, the pipeline does not |
| grid | `w48_d010` … `w96_d100`, 15 cells | 3 frame dimensions (48, 64, 96) by 5 duty cycles (0.10, 0.25, 0.50, 0.75, 1.00) |
| repeats | `w64_d050_rep2`, `_rep3`, `_rep4` | with the grid cell `w64_d050`, four runs of the reference cell, from whose spread the between-run interval is taken |
| control | `w64_d050_permissive` | the reference cell at `tau_permissive`, where the rejection factor should nearly vanish: the negative control for the rejection half of the headline |

Each run is 38 cycles of 2.0 s, 76 s. Twenty cells, about 25 minutes of
measurement plus warm-up, calibration and settling.

### Campaign totals, measured

| quantity | value |
|---|---|
| board time, first run start to last run end | 1622.887 s = **27.05 minutes** |
| runs | 20 of 20 planned, none aborted |
| rail samples per cell | 1540 (20 Hz over 77 s) |
| monitor samples per cell | 77 (1 Hz) |
| **total inferences** | **268,425** |
| **total accepted decisions** | **100,789** |
| reject census | `accept` 100,789 · `low_score` 146,410 · `area_out` 21,226 · `no_candidate` **0** · `border` **0** |
| pooled foreign-plus-own CPU occupancy over all cells | 0.161347 |
| current quantum | 8 mA = 39.883 mW at the measured bus voltage |
| INA3221 hardware averaging | `samples = 512`, `update_interval = 1` |
| files on disk | 125, 19 MB |

**Two of the gate's five branches never fired.** `no_candidate` and `border`
have zero occurrences in the entire campaign. The measured acceptance rate is
therefore a property of a gate whose empty-candidate and border paths were never
reached by measurement; those branches are exercised only by the committed
synthetic generator. No occurrence was synthesised to fill the table, and
`tests/test_measured_campaign.py` asserts the census so the fact cannot change
silently.

### Measurement back-action, measured

The rail sampler draws power it is measuring. Its own CPU time is recorded per
run from `getrusage` into `power_<cell>.csv.rusage`:

| quantity | value |
|---|---|
| mean sampler CPU per run | 1.058 s (0.317 s user, 0.741 s system) |
| as a share of the 77 s run | **1.374 per cent of one core**, about 0.23 per cent of the six-core capacity |

It is **not corrected for**: a correction would need a model of the board's
power against occupancy, which is the thing being measured. The pipeline's own
summed inference wall time is recorded per run in `manifest.json` as
`active_cpu_s`, up to 75.994 s on the duty-1.00 cells.

### The run order was NOT randomised

The plan ran `floor`, then the five frame-48 duties, then the five frame-64, the
five frame-96, the three repeats and the permissive control: 20 sequential 76 s
runs. **Frame dimension is therefore confounded with elapsed time and with a
falling foreign load**, and the floor was measured at the highest foreign load
of the campaign. `w96_d010` draws 3714.66 mW on average while running the
pipeline, which is *below* the 4264.43 mW floor. This is a defect in the
campaign's design; it is published with the per-cell table in
`validation/validate_measured_baselines.py` section 5 rather than corrected
away, and it is the honest explanation for the analytic model's error being
largest on the frame-96 cells.

## How the run cells were produced, and what was declared beforehand

- **Warm-up**, 60 s at full occupancy, before any calibration, so the
  calibration probe is not taken while `schedutil` is still ramping the clock.
- **Calibration**, a sustained 8 s probe per frame dimension. `tau` is set by a
  rule written into the script **before the campaign ran**: `tau_strict` is the
  65th percentile of the probe's score distribution, so the strict acceptance
  rate is about 0.35 by construction, and `tau_permissive` is the 5th
  percentile. The probe scores, both quantiles and the full latency distribution
  go into `manifest.json`, so the choice is auditable and was not made after
  seeing the runs.
- **Per-run verification.** The orchestrator checks after **every** run that
  each expected file exists with rows, checks both subprocess return codes, and
  **aborts the whole campaign** if not. Each subprocess's stderr is captured to
  a file and not to `DEVNULL`.
- **A 60-second instrumentation dry run** is executed and must pass before the
  full campaign is launched.

## The pipeline whose decisions these are

A CPU-only CPython detection pipeline, stdlib-only, in `data/campaign.py`. Per
frame: a deterministic synthetic frame from a per-frame LCG seeded by the frame
index, a 2x2 decimation, a Sobel-style absolute gradient, 4-connected
labelling of the pixels above `grad_thr = 90`, and a score
`mean_gradient * min(1, area / area_ref)` on the best component. The gate
accepts when a candidate exists, its bounding box does not touch the frame
border, its area is in `[2, 60]` and its score is at least `tau`.

**The decision pipeline is deterministic in the frame index alone.** That means
the decision sequence of any run is exactly reproducible off-board from the
committed CSV. The power is not: it depends on the board, the ambient, the
governor and whatever else was running.

## Licence and provenance of the data

Apache-2.0, the same as the code. The traces were produced by this project on
its own hardware; they contain no personal data and no third-party content. The
frames the pipeline processes are generated by a linear congruential generator
in the committed script and are not images of anything.

## Known limitations of the dataset

- One power mode, one board, one uninstrumented ambient, one CPython CPU-only
  pipeline, one 2.0 s cycle period. Nothing in it supports a general statement
  about energy per decision on Orin Nano.
- The background is contaminated and the contamination is measured, not
  controlled.
- The 40 mW current quantum is the floor on every power figure.
- The 512-conversion hardware averaging low-passes the wake transient.
- **The sampler draws power it is measuring.** Its own CPU time is recorded per
  run from `getrusage` and reported as measurement back-action. It is not
  corrected for: a correction would need a model of the board's power against
  occupancy, which is the thing being measured.
- The run order was not randomised; see above.
- Two of the gate's five reject branches never fired; see above.
- `t_mono` is the board's uptime, of order 1e5 to 1e6 seconds, and a 0.05 s
  spacing at that magnitude is represented to about 1e-8 relative. That is a
  property of the timestamps and it puts a floor of about 1e-8 relative on every
  energy integral — ten orders of magnitude below the quantisation floor, so it
  changes no reported figure. It is recorded because
  `validation/validate_property_hunt.py` found it rather than because anyone
  expected it.
