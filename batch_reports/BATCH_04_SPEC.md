# Batch 04 Specification — Onto hardware: HIL, embedded timing, onboard inference, fault injection, and the satellite link layer

**Date:** 2026-10-03 · **Authorization:** ADR-017 standing publication authorization, conditional on `scripts/release_gate.py` exiting 0. **No per-batch approval is cited and none exists.** This spec was written by an unattended session; under ADR-016 it records no approval and attributes no words to the owner.
**Composition:** 2 flagship, 3 medium, 5 compact · 7/10 AI-enabled · Levels: 2 x L1, 5 x L2, 3 x L3
**Theme:** Everything built so far runs on a workstation. Batch 04 is the mission's move toward hardware: a hardware-in-the-loop harness with a real hardware abstraction layer, embedded timing and schedulability, onboard inference sizing, fault and radiation-effect injection, and the satellite-communications link layer the mission has not touched in thirty products. This is the **Level 4 entry batch**.
**Stack:** Python 3.11, NumPy/SciPy, scikit-learn (PyTorch unavailable in the build container), pytest + Hypothesis, Ruff. CLI plus library API plus plotting examples.
**Repositories:** one per product (ADR-018), named for the package, authored solely as `Om Acharya <145807881+OmAcharya-avtr@users.noreply.github.com>`.

## Build precondition — read this before starting

**This batch must not be built until `connectedFolders` is non-empty inside the
scheduled run.** As of 2026-10-03 it is empty, `device_commit_files` is refused,
and the container-to-Mac transfer leg of the publication path is down. Source
trees would survive the osascript text route; screenshots, bundles and any
binary would not. Building a batch that cannot be published is the failure this
mission has already paid for twice. See `nightly_reports/2026-10-03.md`.

This specification is text, publishes through the working leg, and exists so
that the first run with a working transfer path can start at Phase 2 instead of
Phase 0.

## PyPI name check

All ten names checked 2026-10-03 by `GET https://pypi.org/pypi/<name>/json`,
HTTP 404 on every one, therefore **all free**: `hilforge`, `constellink`,
`edgeinfer`, `faultinject`, `telemetryool`, `rtclock`, `framesync`,
`dopplerkit`, `latencynet`, `bitflipsim`.

`ccsdskit` was checked, is free, and was **dropped anyway** — see the note under
P037. Availability of a name is not a reason to build the product.

Packages confirmed to **exist** on PyPI the same way, for use in alternatives
tables: `ccsdspy`, `spacepackets`, `reedsolo`, `commpy`, `galois`, `crcmod`,
`itur`, `sgp4`, `skyfield`, `poliastro`, `pyod`, `alibi-detect`, `river`,
`ruptures`, `statsmodels`, `labgrid`, `pyvisa`, `cocotb`, `simpy`,
`pytest-benchmark`, `hdrhistogram`, `psutil`, `py-spy`, `onnxruntime`,
`pytorchfi`, `tmtccmd`, `hypothesis`, `scikit-learn`. Build agents must still
read what each one actually ships before naming it; existence is not
equivalence.

## Quota gap this batch closes

Before (`scripts/quota_report.py`, 2026-10-03): 14 flagship, 21 medium, 35
compact remaining; AI 21 of 70; L1 6, L2 18, L3 6, L4 0.
After, if all ten pass the gate: 12 flagship, 18 medium, 30 compact; AI 28 of
70; L1 8 of 10, L2 23 of 60, L3 9 of 25, **L4 still 0 of 5**.

L1 is nearly closed — only two slots remain after this batch, so later batches
must stop producing L1 products. AI is at 21 of a 70 minimum with 70 products
left, which is comfortable at 7 per batch.

## The Level 4 entry plan

Level 4 is zero and the Batch 05 deadline is one batch away. It cannot be
closed in the cloud, and it will not be faked.

**Built in the container this batch (P031 HilForge, P033 EdgeInfer):**

1. A hardware abstraction layer — sensors, actuators and timers behind one
   interface, with a simulated backend and a device backend that share the
   same contract and the same test suite.
2. Simulation mode: the full loop against models, deterministic and seeded.
3. Dry-run mode: the real command path, real timing, outputs discarded, so a
   deployment can be rehearsed without moving an actuator.
4. Deployment and recovery procedures, written as executable checks rather
   than prose: what is verified before a run, what is captured during it, and
   how a half-finished run is backed out.
5. A benchmark harness that records latency, memory and throughput with the
   measurement method and environment stated in the output file.

**Labelling rule, absolute:** products carrying this groundwork are
`validation_level: 3` with `hardware_pending: true` in `products.yaml` and the
phrase **Level 3, hardware-pending** in the README. They are **never** labelled
Level 4, and the word Level 4 does not appear in their documentation except to
say what is still missing. Level 4 requires measured timing and resource use
**from the Jetson Orin Nano itself**. No number in any Level 4 claim may come
from a simulated backend, an extrapolation, a vendor datasheet, or a
workstation run.

**What the owner will have to do, later:** run the benchmark harness on the
Jetson and make its raw output available to a session. Until that happens, L4
stays at 0 and should be reported as 0 in every status document. A promoted
label without a measurement is the same defect class as a fabricated benchmark.

---

## P031 — HilForge (flagship, L3 hardware-pending, AI)

**Problem:** Flight software is validated against models, then meets hardware
and discovers that the model never had a deadline, a dropped sample or a bus
that stalls for 40 ms. The bridge between the two is a hardware-in-the-loop
harness, and in open source it effectively does not exist for this domain:
there are board-farm provisioners and instrument-control libraries, but nothing
that gives a GNC or comms loop a hardware abstraction layer with
simulation/device parity, dry-run rehearsal, deadline accounting and a
repeatable benchmark record.

**Scope:** The five items of the Level 4 entry plan above, concretely — HAL
interfaces for sensor read, actuator write and timebase; a simulated backend
driving the existing mission models and a device backend reaching real hardware
through a driver shim; identical test suite run against both backends;
deterministic seeded replay of a recorded run; deadline and overrun accounting
per loop iteration with latency histograms; dry-run mode; deployment and
recovery checks; the benchmark harness and its output format.

**AI element:** a deadline-overrun predictor over recent per-stage latency
features, which flags an iteration likely to miss its deadline early enough to
shed load. Baselines, both implemented first: a fixed-threshold rule on the
previous iteration, and a queueing/Markov model of stage latency. Benchmarked
on lead time, false-alarm rate and missed-overrun rate. **The honest expected
outcome is that the fixed threshold is hard to beat on jittery real traces;
report what is measured and keep the result either way.**

**Level 3 requirements:** `docs/REQUIREMENTS.md` with 12-18 numbered
requirements and a verification matrix; uncertainty analysis over timing
measurements, including clock resolution as a stated error term; regression
suite with pinned seeded outputs; performance benchmark; failure-mode tests —
device absent, device disconnects mid-run, sample dropped, timebase steps
backwards, actuator write rejected, loop overrun cascade.

**Validation:** simulated and device backends produce bit-identical results on
a seeded deterministic case where the device backend is pointed at a loopback
stub; latency histogram percentiles reproduce an injected known latency
distribution to within its sampling error; overrun accounting reproduces a
hand-counted overrun sequence exactly; dry-run mode provably issues no write
(verified by a counting stub, not by inspection); recovery procedure restores
the pre-run state after an induced mid-run abort.

**Alternatives honesty:** `labgrid` for board provisioning and `pyvisa` for
instrument control are real and better at those jobs; `cocotb` is a different
layer entirely (HDL testbenches). Say so, and say when a reader should use them
instead of this.

## P032 — ConstelLink (flagship, L3, AI)

**Problem:** A constellation link plan is a time-varying graph problem, and the
mission has touched only the single-ground-station case (P004 PassPlanner).
Inter-satellite topology, routing under contact windows, and capacity under
hybrid RF/optical links are where constellation design actually gets decided,
and the tooling is mostly proprietary.

**Scope:** Orbit propagation via `sgp4` for a TLE set or a generated Walker
constellation; contact-window computation for inter-satellite and
satellite-to-ground links with geometric blockage by the Earth limb and
range limits; the time-varying contact graph; routing by Dijkstra over a
time-expanded graph and by an ILP for maximum-flow scheduling with `pulp`;
per-link capacity from an RF and an optical model with pointing and atmospheric
loss at the ground leg; queueing and latency accounting end to end.

**AI element:** a learned link-availability predictor (will this contact close
at the achieved data rate) from geometry, elevation and weather-proxy features,
against a climatology baseline and a logistic-regression baseline, with
calibrated probabilities and a reliability diagram. Calibration, not accuracy,
is the headline number.

**Level 3 requirements:** as P031, with failure modes covering an empty contact
graph, a partitioned constellation, a satellite loss mid-horizon, and a TLE
epoch far from the requested window (which must raise, not silently
extrapolate).

**Validation:** contact windows for a two-body circular case match the
closed-form central-angle expression; `sgp4` propagation reproduces a published
TLE test vector to the documented tolerance; Dijkstra on a time-expanded graph
agrees with exhaustive enumeration on small instances; the ILP objective
matches the brute-force optimum on instances small enough to enumerate; link
capacity reduces to the standard free-space expression when losses are zeroed.

**Alternatives honesty:** `sgp4`, `skyfield` and `poliastro` do propagation and
geometry and are depended on, not replaced. Research codebases for
constellation routing exist outside PyPI; the build agent must look for them
and name them in the alternatives table if they fit, rather than claiming an
empty field.

## P033 — EdgeInfer (medium, L3 hardware-pending, AI)

**Problem:** Putting a model on an aerospace edge computer is a budget
question, not an accuracy question: latency, memory, power and worst-case
behaviour against a declared envelope. Teams answer it by flashing the board
and finding out.

**Scope:** A declared budget (latency, peak memory, power ceiling, duty cycle)
as a first-class object; candidate model characterization through
`onnxruntime` and scikit-learn backends; an analytic cost model from operation
counts and memory traffic (a roofline-style estimate) as the deterministic
baseline; measured profiling via the HilForge-style benchmark harness,
implemented independently here; pass/fail against the budget with stated
uncertainty; worst-case versus median latency separated, because the median is
not what a control loop cares about.

**AI element:** a learned latency/memory predictor from model-graph features,
benchmarked against the analytic roofline baseline on held-out models.
**The analytic baseline may well win, and that is a publishable result**;
the README reports it plainly if so.

**Level 3 requirements:** as above, with failure modes for a model exceeding
memory, an unsupported operator, a thermally throttled device and a budget that
is infeasible by construction.

**Validation:** the analytic operation count matches a hand count for two small
networks; measured latency on the simulated backend reproduces an injected
synthetic cost model; peak-memory measurement reproduces a known allocation
pattern; every reported number carries its measurement method and repeat count.

**Hardware-pending scope:** all cloud numbers are workstation numbers and are
labelled as such in the output file. The Jetson Orin Nano columns stay empty.

## P034 — FaultInject (medium, L2, AI)

**Problem:** Fault injection decides whether a fault-detection scheme works,
and it is usually a handful of ad-hoc test cases. What is missing is a campaign
tool: a fault taxonomy, systematic injection, coverage accounting, and an
honest search over a space too large to enumerate.

**Scope:** A fault taxonomy for GNC and comms software — sensor bias, drift,
stuck, dropout, quantization collapse; actuator loss of effectiveness, stuck,
runaway; bus delay, reorder, loss; timing faults (late sample, overrun);
numerical faults (NaN propagation, denormal, overflow). Injection into a target
loop through a wrapper that does not require modifying the target; campaign
definition, execution and a coverage metric over the taxonomy cross product;
severity scoring from the target system response; reproducible seeded replay of
any single injected case.

**AI element:** a learned campaign prioritizer that orders untried
fault/parameter combinations by predicted severity, against two baselines
implemented first: uniform random search and coverage-greedy search. Metric:
severe faults found per campaign budget, with confidence intervals over seeds.
**If random search is inside the interval, the result is `no measurable
advantage` and is reported as such.**

**Validation:** every injected fault is reproducible from its seed and
parameters; coverage accounting matches a hand enumeration on a small taxonomy
subset; an injected fault with a known analytic effect produces that effect
(a constant sensor bias shifts the filter innovation mean by the expected
amount); NaN injection is detected rather than silently absorbed.

**Alternatives honesty:** `pytorchfi` does bit-level injection for PyTorch
models and is the closer prior art for P040 than for this product; `hypothesis`
generates adversarial inputs and is a dependency here, not a competitor. State
both.

## P035 — TelemetryOOL (medium, L2, AI)

**Problem:** Housekeeping telemetry monitoring in practice is a limit table and
an alarm that cries wolf. The limit table is necessary; the thing nobody
characterizes is the false-alarm rate it actually delivers on real noisy
channels, and what a detector costs to add on top.

**Scope:** Out-of-limit checking with the full operational semantics — soft and
hard limits, per-channel validity masks, persistence/debounce counts, mode
dependence; EWMA and CUSUM drift detection with designed false-alarm rates;
change-point detection; a multivariate novelty detector; calibrated alarm rates
throughout, with the design value and the measured value side by side.

**AI element:** the multivariate novelty detector, benchmarked against the OOL,
EWMA and CUSUM baselines on detection delay at matched false-alarm rate.
**Matched false-alarm rate is mandatory** — a detector compared at a different
operating point is not compared at all. ROC and detection-delay curves for
every method.

**Validation:** the EWMA and CUSUM empirical false-alarm rates match their
design values under the nominal hypothesis across seeds; detection delay for a
step change matches the analytic CUSUM expectation; persistence logic
reproduces a hand-traced alarm sequence exactly; confusion matrices reported in
full.

**Alternatives honesty:** this is the most crowded product in the batch.
`pyod`, `alibi-detect`, `river`, `ruptures` and `statsmodels` all exist and are
mature. The narrow defensible claim is the operational OOL semantics and the
matched-false-alarm-rate comparison harness, **not** a better detector. If a
`pyod` model beats the one shipped here, the README says so and tells the reader
to use `pyod`. If the build agent cannot write that comparison honestly, it
must report that in the readiness report rather than pad the table.

## P036 — RtClock (compact, L1, no AI)

**Scope:** Real-time timing primitives for a Python control loop — a monotonic
timebase abstraction with stated resolution; period/deadline arithmetic with
units; a fixed-rate loop driver that reports drift rather than absorbing it;
latency histograms with exact percentile semantics; schedulability arithmetic
for a periodic task set (rate-monotonic and EDF utilization bounds,
response-time analysis, blocking under a priority ceiling); timing-budget
composition across stages.

**Validation (L1):** utilization bounds reproduce the textbook values for the
standard task sets, including the n(2^(1/n)-1) RM bound; response-time analysis
converges to hand-computed values for published examples; histogram percentiles
match an exact computation over the stored samples; the loop driver's reported
drift matches an injected clock skew; clock resolution is measured and reported,
not assumed.

**Alternatives honesty:** `hdrhistogram` for histograms, `psutil` and `py-spy`
for process measurement, `simpy` for discrete-event simulation. The claim here
is the arithmetic and the schedulability analysis, held to L1 rigor.

## P037 — FrameSync (compact, L2, no AI)

**Scope:** CCSDS telemetry **frame-level link performance** — attached sync
marker correlation detection, the sync acquisition/flywheel/loss state machine
with its transition thresholds, slip and false-sync behaviour, and frame-loss
and frame-error rate against Eb/N0 for an uncoded link, an RS(255,223) link and
a convolutionally coded link, using `reedsolo` and `commpy` as dependencies.

**Why not `ccsdskit`:** the obvious product here was a CCSDS packet and framing
library. `ccsdspy` and `spacepackets` already do that well, so building it
would have been duplicated effort with no honest answer to the ADR-018
question. The name `ccsdskit` is free and was dropped anyway. What is **not**
available as a package is the frame-level performance harness — sync
acquisition statistics and frame-loss rate versus Eb/N0 — which is the gap
between a codec library and a link budget (P006 LinkBudgetX). That is the
narrow claim, and the README must state it in exactly these terms: this is a
performance harness over existing codecs, not a codec, and not a packet parser.

**Validation (L2):** uncoded frame-error rate matches the analytic
(1-(1-BER)^n) expression; RS(255,223) corrects up to 16 symbol errors and fails
at 17, verified exhaustively on seeded patterns; measured RS coding gain at
10^-5 falls in the range the standard references quote, with the reference
cited; false-sync probability for a random bit stream matches the combinatorial
expression for the ASM correlation threshold; the acquisition state machine
reproduces a hand-traced sequence exactly.

## P038 — DopplerKit (compact, L1, no AI)

**Scope:** Doppler and range-rate geometry for a satellite link — range,
range-rate and Doppler shift from propagated states; Doppler rate; carrier
pre-compensation profiles over a pass; time-of-flight and light-time
correction; the one-way and two-way cases kept explicitly separate with the
sign convention stated at every interface, because that is where this is
usually got wrong.

**Validation (L1):** range-rate from finite-differenced range agrees with the
analytic dot-product expression to numerical tolerance; Doppler shift for a
circular orbit overhead pass matches the closed-form expression; the zero
crossing occurs at closest approach to within the integration step; two-way
Doppler is exactly twice one-way in the non-relativistic limit and the
relativistic correction term is reported with its magnitude; light-time
iteration converges and its residual is reported.

**Alternatives honesty:** `sgp4`, `skyfield` and `poliastro` provide the
states; this provides the link-geometry layer on top and depends on them.

## P039 — LatencyNet (compact, L2, AI)

**Scope:** End-to-end latency prediction for a staged embedded pipeline from
per-stage features — a sum-of-stages analytic model and a linear regression as
the two baselines, a learned predictor as the third, prediction intervals on
all three, and an honest comparison on held-out pipelines. Tail latency
(p99, p99.9) treated as the primary target, since that is what a deadline
cares about.

**Validation (L2):** the analytic sum-of-stages model is exact on an injected
synthetic pipeline with independent stage latencies, which is the correctness
check for the whole harness; prediction intervals achieve their nominal
coverage on held-out data, reported as measured coverage against nominal;
tail-latency estimates converge with sample count at the rate order statistics
predict. **The analytic baseline is expected to win where stages are
independent; the learned model should only help where they are not, and the
README must show which case the data falls in.**

## P040 — BitFlipSim (compact, L2, AI)

**Scope:** Radiation-induced single-event-upset effects on onboard inference —
bit-flip injection into model parameters and activations at a specified flux and
exposure time, per-bit criticality by position (exponent versus mantissa for
floats, and the quantized-integer case), accuracy and output-distribution
degradation versus upset rate, and mitigation evaluation (parameter range
clamping, selective triplication, periodic reload) with its cost in memory and
latency.

**AI element:** a learned criticality predictor that identifies which
parameters matter, against a magnitude-based baseline and an exponent-bit
heuristic baseline. Metric: degradation avoided per protected byte.
**The exponent-bit heuristic is a strong baseline and is expected to be hard to
beat; if it wins, that is the result.**

**Validation (L2):** injected upset counts match the Poisson expectation for
the stated flux and exposure; flipping a float exponent sign bit produces the
analytically predicted magnitude change; a clamped parameter range provably
bounds output deviation, verified against the bound; triplication with a
majority vote recovers exactly the single-upset cases and fails on the
double-upset cases, verified exhaustively on a small parameter set.

**Alternatives honesty:** `pytorchfi` is the closest prior art and does this
for PyTorch, which is unavailable in this build environment. The README must
name it, state that it is the right tool for PyTorch models, and confine this
product's claim to the scikit-learn/ONNX path plus the flux-model and
mitigation-cost accounting.

---

## Shared conventions

Conventions follow QuatKit (P007) for attitude, EstimKit (P017) for
estimators, and LinkBudgetX (P006) for link quantities and units.
**No cross-product imports** — each repository stays independently
installable, and a sibling is cited as related work, never imported.

Deliberate independent cross-checks, where disagreement is a finding rather
than a nuisance:

- P033 EdgeInfer and P039 LatencyNet must agree on measured latency for the
  same synthetic pipeline from independent implementations.
- P036 RtClock and P031 HilForge must agree on overrun counts for the same
  injected timing trace.
- P037 FrameSync frame-error rate must be consistent with P010 BERBench bit
  error rate for the same channel at the same Eb/N0, through the analytic
  frame/bit relation.

## Build order and concurrency

Five build agents maximum, in two waves, with a commit, a checkpoint and a
**push** between them (usage limits have twice terminated every concurrent
agent at once; recovery is `SendMessage` to the agent id, which resumes from
its transcript rather than restarting):

- Wave 1: P031, P032, P033, P036, P038 — the two flagships, the
  hardware-pending medium, and the two L1 compacts whose validation the others
  depend on for conventions.
- Wave 2: P034, P035, P037, P039, P040.

Each agent reads `templates/PRODUCT_BUILD_GUIDE.md` first and
`templates/REPO_README_STANDARD.md` before writing its README.

## Completion gate

Mission section 17 for every product; section 11 items 1-15 additionally for
the seven AI products. Test counts come from junit XML and nowhere else — never
from pytest stdout, never from an agent's self-report. The coordinating session
re-runs every validation script itself; an implementing model is never the sole
validator of its own critical numerical claims.

`scripts/release_gate.py` must exit 0 before anything is published. On a
non-zero exit, every blocked product is set to `NEEDS HARDENING` in
`products.yaml`, the exact gate output goes into the nightly report, and
nothing from the batch is published — not even the products that passed.

**Two operational notes for the night that builds this batch**, both found on
2026-10-03: run the gate with named products (`release_gate.py P031 P032 ...`)
because the unargumented form re-runs all thirty built products and does not
finish in a usable time; and `git checkout --` the `products/*/screenshots/` and
`products/*/validation/` paths after a gate run, because the gate re-executes
every example and validation script and the regenerated PNGs differ
byte-for-byte even when the numbers are identical.
