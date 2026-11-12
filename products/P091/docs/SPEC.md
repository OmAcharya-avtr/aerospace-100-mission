# P091 ThermalBudget — specification (Phase 2)

**Class** flagship · **Category** hardware-in-the-loop · **Package** `thermalbudget`
**Validation level** target 4; achieved level is decided by what is measured, see below.
**Licence** Apache-2.0 · **Registry** `products.yaml` entry P091

## The question the product answers

A datasheet or a bench benchmark reports throughput measured in a short burst
from a cold board. An on-board autonomy or GNC pipeline runs for hours. Between
the two sits thermal throttling, and the gap is not a footnote: it is the
difference between a schedule that holds and one that does not.

**Headline output: the cold-burst overstatement** — sustained throughput after
the board reaches thermal equilibrium, as a fraction of the short-burst figure,
with an interval attached.

## Measurement basis — WHAT MAKES THIS DIFFERENT FROM EVERY PRIOR BATCH

This is the first product in the mission with access to the target board. Every
thermal, clock and throughput figure in this product is measured on an
**NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super**
(`/proc/device-tree/model`), aarch64, JetPack R36.4.7, kernel 5.15.148-tegra,
6 cores, 7619 MB, governor `schedutil`, `cpuinfo_max_freq` 1728000 kHz,
power mode `MAXN_SUPER`.

### Declared limits of the campaign, recorded before it ran

1. **Power mode could not be varied.** `sudo -n true` on the board returns
   `sudo: a password is required`. `nvpmodel` and `jetson_clocks` need root, so
   the board was characterised at its as-found mode, `MAXN_SUPER`, only. Any
   figure here is a MAXN_SUPER figure and the product must say so wherever one
   is quoted.
2. **Ambient could not be varied.** There is one ambient and no way to change
   it without physical access. The registry summary asks for the learned
   predictor to be scored on held-out **ambient** conditions as well as
   held-out duty cycles. **The ambient axis is therefore structurally
   unmeasurable in this campaign and must be reported as such, not silently
   dropped and not approximated by the duty-cycle split.** The pre-run idle
   temperature of every run is recorded so that ambient *drift* across the
   campaign is at least visible.
3. **The board is shared.** `foreign_cpu_pct` — total CPU percentage of every
   process that is not the campaign — is sampled every second alongside the
   temperatures, so contamination is visible in the data rather than assumed
   absent. At campaign start it sat near 20–25 per cent, mostly `avahi-daemon`.
4. **Three thermal zones expose no reading.** `thermal_zone2/3/4`
   (`cv0/cv1/cv2-thermal`) return no bytes at all: `pathlib.read_text` raises
   `TypeError: can't concat NoneType to bytes`, while a shell `cat` prints
   nothing and exits 0. Six zones are readable: `cpu-thermal`, `gpu-thermal`,
   `soc0/1/2-thermal` and `tj-thermal`. `tj-thermal` is the junction
   temperature and is the product's die-temperature channel.

### The payload

One iteration is a fixed-work GNC-representative kernel: an EKF
propagate-and-update on a 12-state / 6-measurement system
(`x=Fx`, `P=FPFᵀ+Q`, `S=HPHᵀ+R`, `K=PHᵀS⁻¹`, `x+=K(z-Hx)`, `P=(I-KH)P`) plus
one 96×96 float64 GEMM. Deterministic; no RNG inside the loop; matrices seeded
once at 20261112. Six worker processes, one per core, `OMP_NUM_THREADS=1` so
BLAS does not oversubscribe. Throughput is counted iterations per wall second
summed across workers. Duty cycle is enforced per worker over a 50 ms window
and the **achieved** duty is measured and recorded, not assumed.

### The runs

| Label | Duty | Seconds | Purpose |
|---|---:|---:|---|
| `idle_pre` | 0 | 120 | ambient / idle floor |
| `burst_1.00_a`, `burst_1.00_b` | 1.0 | 20 | baseline 1, the cold-burst figure, twice |
| `soak_1.00_r1` | 1.0 | 720 | baseline 3 ground truth, primary soak |
| `soak_1.00_r2`, `soak_1.00_r3` | 1.0 | 480 | run-to-run spread for the bootstrap interval |
| `soak_0.25/0.50/0.75` | 0.25/0.5/0.75 | 420/480/480 | held-out duty-cycle axis |
| `cool_*` | 0 | 180–300 | RC decay traces, independent τ estimate |

### Trace schema — binding

CSV, one header row, one row per second:

```
t_wall,t_rel,iters_cum,iters_per_s,achieved_duty,loadavg1,foreign_cpu_pct,
temp_cpu-thermal,temp_gpu-thermal,temp_soc0-thermal,temp_soc1-thermal,
temp_soc2-thermal,temp_tj-thermal,cpu0_khz,...,cpu5_khz,gpu_hz
```

Temperatures are milli-degrees Celsius as the kernel reports them; the loader
converts and must not assume the scale. A field that could not be read is
`-1` and **must be treated as missing, never as a value**. `manifest.json`
carries the board identification, the run plan and the per-run wall-clock
window.

## The three baselines, in build order

Baseline 1 and 2 are implemented and reported **before** any learned model.

- **Baseline 1 — cold-burst extrapolation.** Sustained throughput ≈ burst
  throughput. Implemented deliberately as the thing being refuted, because it
  is what a bench measurement produces and what a link-budget-style
  calculation assumes. Expected to be wrong in a specific direction and by a
  measurable amount; that amount is the product's headline.
- **Baseline 2 — first-order thermal RC network, closed form.** Fit
  `T(t) = T∞ − (T∞ − T₀)·exp(−t/τ)` to the measured `tj` rise; take τ
  independently from the cooldown traces as a cross-check and report both.
  Map steady-state die temperature to the throttled steady-state clock using
  the measured clock-versus-temperature relation, then clock to throughput.
  Analytic, no fitting of the throughput itself.
- **Baseline 3 — ground truth.** Directly measured sustained throughput and die
  temperature from the soak runs, with the run-to-run spread reported as a
  **bootstrap confidence interval** over the three duty-1.0 soaks and over
  within-run equilibrium samples. Equilibrium is not assumed: it is tested, by
  the slope of a linear fit over the final window with a stated threshold, and
  a run that fails the test is reported as not-equilibrated rather than
  averaged anyway.

## The learned component

A thermal-state predictor mapping a **short warm-up trace** (the first N
seconds of `tj` and clock) to the eventual steady state.

- Scored on **held-out duty cycles** — train on a subset of duty cycles,
  predict a duty cycle never seen. Measurable.
- Scored on **held-out ambient** — **structurally unmeasurable here**, see
  limit 2. Report as unmeasurable with the reason; do not substitute.
- **Expected and acceptable result: the RC model is competitive and any learned
  margin is smaller than the run-to-run bootstrap interval.** If that is what
  the data says, that is the finding and it is published as the finding. The
  learned model is not retuned until it wins, and a margin inside the interval
  is reported as "not distinguishable from the baseline", never as an
  improvement.

## Validation-level rule, binding

The product's own README, badge, `VALIDATION.md` and
`thermalbudget.VALIDATION_LEVEL` state the achieved level. A figure measured on
the board is a measured figure and may be labelled so. **A figure that was not
measured on the board — anything depending on a power mode other than
MAXN_SUPER, on a varied ambient, or on hardware the campaign did not touch —
must be labelled unmeasured, by name.** The registry slot keeps
`validation_level: 4` as its target and carries `validation_level_achieved`
set to what this session actually achieved, decided from the evidence at
Phase 9 and not promised here.

## Prohibitions

No marketing language, no emoji, no exclamation marks. Never "flight-safe",
"certified", "mission-ready" or "production-ready". Every number in a
validation document comes from a script executed this session. The classical
baselines are implemented before the model and a baseline win is never
removed, softened or retuned away. Credits line verbatim, once:
"This is under reserved rights obtained by OPTIMA Organisation."
