# Dataset card — the P091 ThermalBudget measurement campaign

## What it is

Per-second traces from a measurement campaign run on an **NVIDIA Jetson Orin
Nano Engineering Reference Developer Kit Super** (`/proc/device-tree/model`),
aarch64, JetPack R36.4.7, kernel 5.15.148-tegra, 6 cores, 7619 MB, governor
`schedutil`, `cpuinfo_max_freq` 1728000 kHz, power mode **MAXN_SUPER**.

The campaign is `data/campaign.py`, which runs **on the board** and writes one
CSV per run plus `manifest.json`. The traces belong in `data/traces/`.

**State at the time this package was built: `data/traces/` was empty**, the
campaign had not finished, and every script that needs the traces refused by
name and exited non-zero rather than substituting anything. **The campaign has
since completed.** `data/traces/` now holds all 17 CSV traces plus
`manifest.json`, transferred from the board with a verified sha256 at both ends,
and every measured figure in this repository was computed from them.

Campaign window, from `manifest.json`: started 1794457188.947, finished
1794462341.063, a little under 86 minutes of wall clock for the 17 runs.
`manifest.json` also records that the per-run `csv` fields were rewritten to
basenames by the coordinating session; the campaign wrote absolute paths under
the board user's home directory, which is provenance of no analytical value and
is not published.

**What the campaign measured, in one line:** the board did not throttle. Across
the three duty-1.0 soaks the die rose 10.94 degC, from 46.34 to 57.28, and not
one of the 9654 per-core clock samples fell below `cpuinfo_max_freq`
1728000 kHz.

## The payload

One iteration is a fixed-work, GNC-representative kernel: an extended Kalman
filter propagate-and-update on a 12-state / 6-measurement system
(`x=Fx`, `P=FPF'+Q`, `S=HPH'+R`, `K=PH'S^-1`, `x+=K(z-Hx)`, `P=(I-KH)P`) plus one
96x96 float64 GEMM. Deterministic; no RNG inside the loop; matrices seeded once
at 20261112. Six worker processes, one per core, `OMP_NUM_THREADS=1` so BLAS does
not oversubscribe. Throughput is iterations per wall second summed across
workers. Duty cycle is enforced per worker over a 50 ms window and the
**achieved** duty is measured and recorded, not assumed.

## The runs

| Label | Duty | Seconds | Purpose |
|---|---:|---:|---|
| `idle_pre` | 0 | 120 | ambient / idle floor |
| `burst_1.00_a`, `burst_1.00_b` | 1.0 | 20 | baseline 1, the cold-burst figure, twice |
| `soak_1.00_r1` | 1.0 | 720 | baseline 3 ground truth, primary soak |
| `soak_1.00_r2`, `soak_1.00_r3` | 1.0 | 480 | run-to-run spread for the bootstrap interval |
| `soak_0.25` / `soak_0.50` / `soak_0.75` | 0.25 / 0.5 / 0.75 | 420 / 480 / 480 | held-out duty-cycle axis |
| `cool_*` | 0 | 180–300 | RC decay traces, independent tau estimate |

## Schema, binding

CSV, one header row, one row per second, 20 columns:

```
t_wall, t_rel, iters_cum, iters_per_s, achieved_duty, loadavg1, foreign_cpu_pct,
temp_cpu-thermal, temp_gpu-thermal, temp_soc0-thermal, temp_soc1-thermal,
temp_soc2-thermal, temp_tj-thermal, cpu0_khz .. cpu5_khz, gpu_hz
```

- Temperatures are **milli-degrees Celsius** as the kernel reports them
  (Linux `Documentation/driver-api/thermal/sysfs-api.rst`). The loader
  **infers** the scale from the data against a declared rule and refuses on
  anything outside both accepted bands, rather than assuming.
- Clocks are kHz (`cpufreq/scaling_cur_freq`); the GPU devfreq frequency is Hz.
- **`-1` means UNREADABLE and is never a value.** It becomes `nan` on load,
  before any unit conversion, and the count of converted samples travels with
  the trace. Dividing the sentinel by 1000 would give `-0.001 degC`, a
  plausible-looking number that would drag a mean towards zero and bias a slope.
  `tests/test_regressions.py` pins that.
- `manifest.json` carries the board identification, the run plan and the per-run
  wall-clock window.

Five real rows of the completed cold-burst run are quoted verbatim in
`docs/SPEC.md` and are loaded and checked by
`validation/validate_loader_contract.py`. In all five, every one of the six cores
sits at 1728000 kHz.

## Declared limits of the campaign, recorded BEFORE it ran

1. **Power mode could not be varied.** `sudo -n true` returns
   `sudo: a password is required`; `nvpmodel` and `jetson_clocks` need root. The
   board was characterised at its as-found mode, **MAXN_SUPER**, only.
2. **Ambient could not be varied.** One ambient, no physical access. The
   pre-run idle temperature of every run is recorded so ambient **drift** is at
   least visible. The held-out-ambient evaluation axis is therefore
   **structurally unmeasurable** in this campaign.
3. **The board is shared.** `foreign_cpu_pct` — the total CPU percentage of every
   process that is not the campaign — is sampled every second, so contamination
   is visible in the data rather than assumed absent. At campaign start it sat
   near 20–25 per cent, mostly `avahi-daemon`.
4. **Three thermal zones expose no reading.** `thermal_zone2/3/4`
   (`cv0/cv1/cv2-thermal`) return no bytes at all: `pathlib.read_text` raises
   `TypeError: can't concat NoneType to bytes`, while a shell `cat` prints
   nothing and exits 0. Six zones are readable and `tj-thermal` is the die
   channel.

## The synthetic fixtures, and why they can never be mistaken for this

`tests/fixtures/synthetic_trace.py` emits **schema-identical** traces for unit
tests and for the examples. They are not measurements of anything, and four
separate things say so:

- every file it writes is named `SYNTHETIC_<label>.csv`, and its manifest is
  `SYNTHETIC_manifest.json`;
- that manifest's `model` field reads
  `"SYNTHETIC fixture written by tests/fixtures/synthetic_trace.py; not a board
  and not a measurement"`, so `Campaign.board_matches_declared_target()` is
  False;
- the loader requires an explicit `Origin` argument with no default and no
  inference from the filename, and the fixture loader passes `SYNTHETIC`;
- `Origin` is a two-element lattice with `SYNTHETIC` absorbing, so any result
  touching a fixture is `SYNTHETIC`, and `require_measured` raises rather than
  returning a label.

The generator has two declared regimes and **neither is a default**:
`no_throttle` holds the clock at its maximum, which is what the one real
cold-burst trace shows; `throttled` rolls it off above a declared knee. Both
exist so that the degenerate and the identifiable code paths are both exercised
and so that nothing in this product presupposes that throttling happens.

## Licence and provenance of the data

The traces are produced by this repository's own campaign script on hardware
belonging to the campaign. No third-party dataset is used, redistributed or
cited. Apache-2.0, © 2026 OPTIMA Organisation, as for the code.

## Measured quality of the dataset

All from `validation/validate_measured_campaign_output.txt`.

- **Missing data: zero.** Not one sentinel-coded or unparseable sample across
  all 17 runs. The `-1` handling was never exercised by the real campaign, only
  by the fixtures and by the hand-built rows in
  `validate_loader_contract.py` — worth knowing, because it means the measured
  traces give no evidence that the sentinel path works, and the synthetic ones
  are what does.
- **Contamination, measured rather than assumed absent.** Pooled foreign CPU
  **17.05 per cent** over 4975 samples. Per-run means from **14.93** (`cool_r3`)
  to **20.62** (`idle_pre`) per cent, with per-sample maxima up to 28.9 per
  cent, mostly `avahi-daemon`. Note the drift: the early runs sit near 20 per
  cent and the late ones near 15, so the co-resident load was not stationary
  across the campaign.
- **Achieved duty against commanded duty.** 0.2527 / 0.5030 / 0.7507 for the
  0.25 / 0.50 / 0.75 soaks and 0.9998–0.9999 for the duty-1.0 soaks, with
  per-sample values ranging from 0.9494 to 1.2041 at duty 1.0. Achieved duty can
  exceed the commanded value because the control window is 50 ms and an
  iteration that starts inside it runs past its end. Every duty-indexed figure
  in this product is indexed by the COMMANDED duty, with the achieved value
  reported beside it.
- **Start-up transient.** Every one of the eight non-zero-duty runs reports a
  first one-second sample below its own post-skip mean: 25.4 and 26.6 per cent
  for the two bursts, 4.2 / 17.3 / 24.2 per cent for duty 0.25 / 0.50 / 0.75,
  and 25.6 / 26.1 / 24.3 per cent for the three duty-1.0 soaks. The deficit is
  **not** constant across duty cycles. The declared 2 s skip is justified in
  every case.
- **GPU devfreq.** At its **306 MHz floor in 4954 of 4975 samples (99.578 per
  cent)**; 21 samples (0.422 per cent) at 1020 MHz, scattered across 13 runs
  including idle and cooldown runs, so they are not caused by the payload.
  **GR3D utilisation is not a column in this schema and was never sampled**, so
  "the GPU was idle" is an inference from the devfreq floor and not a
  measurement.
- **Clock levels.** The governor used **14 distinct levels** campaign-wide, from
  729600 to 1728000 kHz, with a smallest adjacent gap of **76800 kHz**.

## Known limitations of the dataset

- **Six soak runs across four duty cycles.** Three at duty 1.0. A run-level
  bootstrap over three runs has at most 27 distinct resample multisets and the
  interval is granular; over the two burst runs, 4. More runs is the only fix,
  and with only two burst runs the burst side dominates the width of the
  headline interval.
- **One ambient, never instrumented in degrees.** The idle-run die temperature,
  **44.489 degC**, is its only proxy.
- **One board, one fan, one enclosure.** Thermal behaviour is a property of the
  whole assembly and nothing here transfers to another unit. The measured
  "no throttling" result is **not** a claim about Jetson Orin Nano modules in
  general.
- **One CPU-only payload.** A memory-bound or GPU-bound workload dissipates
  differently and none was run.
- **720 s is the longest soak.** The die's measured time constant is 88.85 s but
  the enclosure's thermal mass has a far longer one, and the die was still
  rising at +0.000353 degC/s at the end of that soak. A multi-hour drift, a dust
  effect or an enclosure heat-soak would not appear in these traces.
- **Co-resident processes were present throughout**, drifting from about 20 to
  about 15 per cent across the campaign, and are recorded rather than
  eliminated.

Research-grade. Not flight-qualified, not certified and not approved for
operational aerospace use.
