# P093 PowerEnvelope — specification

**Class** medium · **Category** hardware-in-the-loop ·
**AI** yes · **Validation level target** 4 (hardware) ·
**Package** `powerenvelope` · **License** Apache-2.0

Written before the campaign ran. The pre-registered expectation in the
*Learned component* section is part of the specification and is reported
against whatever the data says.

## The question

An on-board autonomy pipeline is quoted in **energy per inference**: take the
power drawn while the pipeline is running, multiply by the inference latency,
divide by nothing else. Every accelerator datasheet, every edge-AI benchmark and
every power budget in a mission proposal is written in that unit.

A mission does not act on inferences. It acts on **decisions it accepted**. Two
things sit between the two numbers and both cost energy:

1. **Rejection.** An inference whose output fails the confidence gate, or whose
   geometry is invalid, consumed its energy and produced nothing actionable.
2. **Duty cycling.** An on-board pipeline that runs 10 per cent of the time
   still draws the platform's standby power for the other 90 per cent, and pays
   a transient on every wake. Energy per inference charges none of that to the
   decisions it enabled.

**The headline output of this product is the factor by which energy per
inference understates the measured energy per useful decision, decomposed into
the rejection factor, the duty-cycle factor and the nominal-versus-measured
active-power error.** The product is specified to report that factor whatever
its value, including a value near 1, which would say the quoted unit is adequate
for this pipeline on this board.

## What it does that existing tools do not

`tegrastats`, `jetson_stats`/`jtop`, `powertop`, RAPL readers and the INA3221
sysfs interface all report **watts over time**. `pyJoules`, `CodeCarbon` and
`experiment-impact-tracker` integrate that into **joules per code region**.
MLPerf Tiny and MLPerf Inference report **energy per inference** — the very unit
being refuted here.

None of them divides by the number of decisions a mission could act on, and
none of them separates the duty-cycle and rejection components of the gap. The
contribution is not a better power meter; it is the denominator, and the
decomposition that says which part of the gap an engineer can actually reduce.
The alternatives table in the README names each of these and says when to use
it instead — for a pure power trace, use `tegrastats` or `jtop`; for per-region
joules in a server context, use `pyJoules`.

## Measurement model, stated once

- The platform rail is **VDD_IN**, measured by the on-board **INA3221** at
  `/sys/class/hwmon/hwmon1`, channel 1, as bus voltage (mV) × shunt current
  (mA). Instantaneous power `P(t) = V(t)·I(t)`.
- Energy over a window is the **trapezoidal integral** of `P(t)` over the
  sampler's timestamps, which are `CLOCK_MONOTONIC` and therefore directly
  comparable with the pipeline's own per-inference timestamps taken in a
  different process.
- A **cycle** is `CYCLE_S = 2.0` s long. The pipeline runs back-to-back
  inferences for `duty · CYCLE_S` and then sleeps the remainder. A run is 38
  cycles, 76 s.
- An **inference** is one pass of the pipeline over one frame.
- A **useful decision** is an inference whose output passes the gate: a
  candidate exists, its bounding box does not touch the frame border, its area
  is within declared bounds, and its score is at least `tau`. Every rejection
  is recorded with its reason, so the acceptance rate is reconstructible from
  the committed decisions CSV and never asserted.
- **Energy per useful decision** over a run is the trapezoidal integral of
  `P(t)` across **the whole run, sleep windows included**, divided by the
  number of accepted decisions in that run.

This definition is the one every number in the product depends on and it is
stated in `README.md`, `VALIDATION.md` and the loader's docstring.

## Baselines — implement and validate in this order

1. **Baseline 1 — energy per inference. The thing being refuted.** Two
   variants, both implemented because both are quoted in practice:
   - **measured-active**: the mean measured `P(t)` over the active windows only,
     times the mean inference latency. No idle, no ramp, no rejection.
   - **nominal-rail**: the same, with the **nominal 5.0 V** rail voltage
     substituted for the measured bus voltage, which is the specific
     simplification a power budget makes when it quotes a rail rather than
     measuring one.
2. **Baseline 2 — the analytic duty-cycle model.** Closed form from measured
   components: `E_cycle = P_floor·t_sleep + E_ramp + P_active·t_active`, with
   `P_floor` the measured background draw with the pipeline not running,
   `P_active` the measured steady active draw, and `E_ramp` the measured excess
   energy in the settling interval after each wake. Energy per useful decision
   follows as `E_cycle / (n_inf_per_cycle · a)` with `a` the measured acceptance
   rate. A **measured rail-accounting curve** is applied rather than assumed:
   the ratio `(VDD_CPU_GPU_CV + VDD_SOC) / VDD_IN` is measured as a function of
   load across the workload grid and fitted, which is what is actually
   observable on this board — it is **not** a converter-efficiency curve from a
   datasheet and the product says so.
3. **Baseline 3 — ground truth.** The directly integrated rail energy over the
   whole run divided by the counted accepted decisions, with the **between-run
   spread** reported as an interval from the four repeats of the reference cell.
   A bootstrap over samples within one run is **wrong** for this quantity and
   the product says why: the sampler's values are hardware-averaged over 512
   conversions, so neighbouring samples are not independent, and the quantity
   whose uncertainty matters is run-to-run, not sample-to-sample.

## Learned component

A power predictor mapping **descriptors an operator could state from the
configuration and the platform's own environment, without running the
pipeline** — duty cycle, cycle period, frame dimension, gate threshold `tau`,
cores available, and the environment at the window start: die temperature,
1-minute load average, mean core clock, measured foreign CPU occupancy — to the
run's **mean active rail power**, with an uncertainty output.

**Explicitly excluded as descriptors**, because the specification does not
license them and because on 2026-11-13 another product in this mission found its
own top feature was a quantity its specification forbade: the pipeline's own
measured power, its own measured latency, its own CPU occupancy, its realised
inference count, and its realised acceptance rate.

Scored **separately** on:

- **held-out duty cycles** — grouped so no window of a test duty is in training;
- **held-out workloads** — grouped by frame dimension, so a test workload's
  windows are never in training.

**Pre-registered expectation, recorded here before the data exists: the analytic
duty-cycle model of Baseline 2 is competitive, and the learned predictor's
error will not be smaller than the analytic model's by more than the between-run
interval's half-width.** If the learned predictor wins by more than that, it is
published as a win. If the analytic model wins, that is published as the
finding and the learned component stays in the product reported as beaten.
Neither outcome is retuned away or removed.

## Hardware campaign (this is a Level 4 product)

Measured on the **NVIDIA Jetson Orin Nano Engineering Reference Developer Kit
Super**, aarch64, 6 cores, power mode `MAXN_SUPER`, governor `schedutil`.

`scripts/p093_campaign.py` (committed) does the measuring. It is **stdlib-only
on purpose**: the rail sampler must not be delayed by numpy's allocation or
threading behaviour, and the decision pipeline must be byte-for-byte
reproducible off-board.

Campaign structure:

- **Warm-up**, 60 s at full occupancy, before any calibration. A cold probe on
  this board is taken while `schedutil` is still ramping the clock, which on
  2026-11-13 mis-sized another product's payload by a factor of 2.3.
- **Calibration**, a sustained 8 s probe per workload. `tau` is set by a rule
  **declared in the script before the campaign ran**: `tau_strict` is the 65th
  percentile of the probe's score distribution, so the strict acceptance rate is
  about 0.35 by construction, and `tau_permissive` is the 5th percentile. The
  probe scores, both quantiles and the full latency distribution go into
  `manifest.json`, so the choice is auditable and was not made after seeing the
  runs.
- **Grid**: 3 frame dimensions (48, 64, 96) × 5 duty cycles (0.10, 0.25, 0.50,
  0.75, 1.00), 76 s each.
- **Background floor**: one 76 s run with the sampler and monitor running and
  the pipeline **not** running.
- **Between-run spread**: the reference cell (dim 64, duty 0.50) run four times
  in total.
- **A permissive-gate control**: the reference cell at `tau_permissive`, where
  the rejection factor should nearly vanish. This is the negative control for
  the rejection half of the headline.
- A **20 Hz rail sampler** and a **1 Hz environment monitor**, each a separate
  process with its own required `--out`, its stderr captured **to a file and not
  to DEVNULL**, and its return code checked after every run. The orchestrator
  verifies that every expected file exists with rows after **every single run**
  and aborts the campaign if not.

A **60-second instrumentation dry run** is executed and must pass before the
full campaign is launched.

## Scope, which is part of the result

- **One power mode** (`MAXN_SUPER`). `sudo -n true` returns
  `sudo: a password is required` on this board, so `nvpmodel` and
  `jetson_clocks` are unreachable unattended and **every figure is a
  MAXN_SUPER figure.**
- **The board is not idle and the floor is not an idle floor.** At campaign time
  it had five logged-in users, a desktop Chrome running for nearly two days,
  swap in use and a 1-minute load average above 1. The measured floor is a
  **background floor on a shared machine**, it is reported as such, and the
  foreign CPU occupancy is measured once per second rather than assumed absent.
- **The rail current quantises at 8 mA**, which at 5 V is a **40 mW
  least-significant bit** on VDD_IN. Every power figure carries that floor.
- **The INA3221 averages 512 conversions in hardware.** The energy integral is
  sound, but the *shape* of the wake transient is bandwidth-limited by that
  averaging, so `E_ramp` in Baseline 2 is a bandwidth-limited estimate and is
  reported as one, with the limitation stated rather than smoothed over.
- **The sampler draws power it is measuring.** Its own CPU time is recorded per
  run from `getrusage` and reported as measurement back-action.
- **One ambient**, never instrumented or varied. Held-out ambient is
  structurally unmeasurable from here.
- **One CPU-only CPython pipeline.** Nothing here supports a statement about a
  GPU or DLA pipeline, a compiled pipeline, or a different cycle period.
- One board, with its own heatsink and fan.

Nothing in this product supports a general statement about energy per decision
on Orin Nano, and nothing in it may be called flight-safe, certified,
mission-ready or production-ready.

## Deliverables

Standard product layout per `templates/PRODUCT_BUILD_GUIDE.md`.
`MODEL_CARD.md` and `DATASET_CARD.md` required. Validation scripts flat in
`validation/`, each anchored on `__file__`, each setting
`sys.dont_write_bytecode = True` first, each runnable from any working
directory. Committed traces under `data/traces/`, including the campaign log and
`manifest.json`. No absolute path in any tracked file. Dependencies bounded on
both sides at versions actually measured.
