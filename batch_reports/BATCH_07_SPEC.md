# Batch 07 Specification — Onboard autonomy: deciding, fitting, and the testing that justifies it

**Date:** 2026-10-14 · **Authorization:** ADR-017 standing publication
authorization, conditional on `scripts/release_gate.py` exiting 0. **No
per-batch approval is cited and none exists.** This spec was written by an
unattended session; under ADR-016 it records no approval and attributes no words
to the owner.

**Composition:** 2 flagship, 3 medium, 5 compact · **7/10 AI-enabled** ·
Levels: **1 × L1, 6 × L2, 3 × L3** · **no Level 4 and no new Level 4
groundwork** — see "Level 4" below.

**Quota arithmetic this batch is sized against** (`scripts/quota_report.py`,
run 2026-10-14, Registered 60/100 · Built 60/100 · Published 60/100):

| quota | target | built | remaining | this batch | after |
|---|---|---|---|---|---|
| flagship | 20 | 12 | 8 | 2 | 14 |
| medium | 30 | 18 | 12 | 3 | 21 |
| compact | 50 | 30 | 20 | 5 | 35 |
| ai (minimum) | 70 | 42 | 28 | 7 | 49 |
| L1 | 10 | 9 | **1** | **1** | **10 — closed** |
| L2 | 60 | 36 | 24 | 6 | 42 |
| L3 | 25 | 15 | 10 | 3 | 18 |
| L4 | 5 | 0 | **5** | **0** | **0** |

L1 has exactly one slot left and this batch spends it (P066), closing that
quota. L4 is addressed below and deliberately not touched.

**Stack:** Python 3.13 in the build container, NumPy/SciPy, scikit-learn
(PyTorch unavailable), pytest + Hypothesis, Ruff, PuLP/`scipy.optimize.milp`
where an exact optimum is needed. 2 cores; every run under 3 minutes. Library
API plus `python -m <package>` CLI plus plotting examples.

**Repositories:** one per product (ADR-018), named for the package, authored
solely as `Om Acharya <145807881+OmAcharya-avtr@users.noreply.github.com>`.

---

## Why this batch, stated against what already exists

Sixty products in, the portfolio computes aerospace quantities (Batches 01–02,
05), controls attitude (03), runs things on hardware and watches telemetry (04),
and — since Batch 06 — can say when a learned component is wrong and what
happens then.

What it still cannot do is **decide**. There is no product that takes a set of
goals, a power and data budget and a contact schedule and produces a timeline;
nothing that fits a digital twin to the asset it is supposed to mirror (P053
only detects that it no longer does); nothing that turns an FMEA into a
fault-response policy and checks that every fault reaches a safe mode; and
almost nothing about the *testing* of such a thing — one product, P058, carries
the `testing` category.

The honest case for the batch in one sentence: **the mature libraries here are
each excellent at one primitive and silent about the aerospace accounting that
turns the primitive into a decision you could defend.**

`PuLP`, `python-mip` and `OR-Tools` solve the integer program. `lmfit`,
`scipy.optimize` and `emcee` fit parameters. `SALib` does sensitivity.
`pytest-testmon` selects tests by coverage. `allpairspy` and PICT generate
t-way combinations. `scikit-survival` and `lifelines` fit degradation and
survival models. `transitions` implements state machines. Every one of them is
a better choice than anything in this batch for the primitive it owns, and
**every README below must say so, by name, in its alternatives table, before
making its own case.**

What none of them answers:

- Given goals, a power and energy budget, a data-volume budget and a contact
  schedule, **is this timeline feasible, by how much, and which constraint is
  binding** — and does a learned duration/energy predictor beat the analytic
  one well enough to warm-start the solver, measured against the exact optimum.
- Given a twin and telemetry, **which parameters are identifiable from the data
  you actually have**, before anyone reports a fitted value for one that is not.
- Given an FMEA table and an observability model, **does every declared fault
  reach a safe mode**, and under what detection assumption.
- Given a 40-minute numerical suite and a diff, **which tests can be skipped
  without losing a fault**, and what is the measured loss when you skip them.
- Given a scenario space and a campaign that has already run, **which regions
  are uncovered**, computed rather than asserted.
- Given housekeeping telemetry with a trend, **when does this subsystem cross
  the limit, with what interval, and is that interval calibrated** under
  censoring.

Each product is scoped to one of those gaps. **Every product must state what it
does that the named alternative does not.** If an agent concludes while
building that the honest answer is "nothing", it must say so in its final
report rather than pad the README; that is a finding, not a failure.

---

## Level 4 — why this batch does not touch it

L4 is 0 of 5 and **ten batches overdue**; it is the mission's largest unmet
quota. It is not advanced here, and that is a decision, not an omission.

Four products — P031 HilForge, P033 EdgeInfer, P041 CodedFade, P043 PhotonCount
— are already complete to Level 3 and labelled **"Level 3,
hardware-pending"**. Each carries the full cloud-side groundwork the standing
plan prescribes: hardware abstraction layer, simulation mode, dry-run mode,
deployment and recovery procedures, benchmark harness. All four are parked on
the identical blocker: **measured timing and resource use from the Jetson Orin
Nano itself.**

Building an eleventh hardware-pending product does not move L4 from 0. It adds
groundwork to a queue that already has four items in it and no measurement
session. What is needed is the Jetson powered on and reachable, its hostname or
IP, and passwordless SSH from the Mac — an owner hardware session. Until that
happens, nothing is relabelled Level 4, because the only way to do so from here
is to label an unmeasured product Level 4, which the mission's own rules forbid.

---

## PyPI name check — method stated, because the first attempt was wrong

All ten names were probed with `pip index versions <name>` against the live
index. **The first probe reported all ten FREE and was wrong**: it tested for
the string `available versions`, while `pip index` prints `name (version)` for a
package that exists, so a taken name matched nothing and was scored free. The
probe was re-run testing for `No matching distribution found`, and validated
first against four controls — `numpy` → 2.5.4, `pulp` → 4.0.0 (both correctly
TAKEN) and `beamtwin`, `traceaudit`, two of this mission's own unpublished
package names (both correctly FREE).

The corrected run found **`covmetrics` is TAKEN (0.1.2, 0.1.1, 0.1.0)**. P069
was renamed to `scenariocov`, which is free and is the better name anyway.

This is the mission's own recurring defect class — a check that prints success
while measuring nothing, the same shape as P001's zero-collecting pytest config
— reached in the first hour of the session that is supposed to be guarding
against it. Recorded here rather than quietly fixed.

Final result, nine free plus one renamed:

| product | package | PyPI |
|---|---|---|
| P061 | `activityplan` | FREE |
| P062 | `twincalib` | FREE |
| P063 | `safemodetree` | FREE |
| P064 | `regressselect` | FREE |
| P065 | `hazardtrace` | FREE |
| P066 | `budgetroll` | FREE |
| P067 | `acqsearch` | FREE |
| P068 | `degradepredict` | FREE |
| P069 | `scenariocov` | FREE (`covmetrics` was taken) |
| P070 | `watchdogspec` | FREE |

---

## P061 ActivityPlan — flagship · L3 · AI · autonomous-aerospace-systems

Package `activityplan`. **Onboard activity planning with a timeline you can
audit.**

Takes a declared set of activities (duration, power profile, data produced,
priority, time windows, precedence, exclusions), a declared resource model
(battery capacity and depth-of-discharge limit, solar input profile, solid-state
recorder capacity, downlink contact windows with rates) and produces a timeline,
plus the thing that matters: **a feasibility report naming the binding
constraint and the slack on every other one.**

- **Baseline 1 (analytic/greedy):** priority-ordered earliest-feasible-start
  insertion with explicit resource simulation. Fast, documented, suboptimal.
- **Baseline 2 (exact):** MILP over a discretised timeline via
  `scipy.optimize.milp` or PuLP, giving the true optimum on problems small
  enough to solve, which is what the other two are scored against.
- **AI component:** a learned predictor of activity duration and energy draw
  from activity features, with a **calibrated prediction interval**, used (a) to
  warm-start the MILP and (b) to produce a robust plan under predicted-duration
  uncertainty. It must be benchmarked against the analytic duration model on
  held-out activities, and against the exact optimum on optimality gap and
  solve time. **Report honestly if the warm start does not help.**
- **Validation (L3):** energy conservation over the timeline checked against a
  closed-form integral of the power profile; MILP optimum cross-checked against
  exhaustive enumeration on small instances; greedy-vs-MILP optimality gap
  measured over a seeded instance family; prediction-interval coverage measured.
- **Alternatives table must name:** PuLP, OR-Tools CP-SAT, `python-mip`,
  `simpy`. Say plainly that CP-SAT is the better scheduler and that this
  product's contribution is the spacecraft resource accounting and the
  feasibility report, not the search.

## P062 TwinCalib — flagship · L3 · AI · digital-twins

Package `twincalib`. **Fit a digital twin, and say which parameters the data
can actually pin down.**

P053 TwinInvalidate detects that a twin has stopped matching its asset. Nothing
in the portfolio fits one, and nothing checks whether a fitted parameter is
identifiable before reporting it.

- **Identifiability first, fitting second.** Local identifiability from the
  Fisher information matrix / sensitivity Jacobian: rank, condition number,
  and the near-null directions reported as the *parameter combinations* that are
  unconstrained, not just a warning. Profile-likelihood intervals for each
  parameter, which reveal practical non-identifiability that a Hessian
  approximation hides.
- **Baseline:** weighted nonlinear least squares / maximum likelihood on the
  physics model alone, with parameter covariance and correlation reported.
- **AI component:** a learned residual (discrepancy) model on top of the fitted
  physics, with uncertainty, benchmarked against physics-only on held-out
  trajectories — and a measured statement of **how much of the improvement is
  real and how much is absorbing identifiability failure**, which is the honest
  risk of residual models and must be addressed, not elided.
- **Validation (L3):** recovery of known parameters on synthetic data with
  known ground truth across noise levels; profile-likelihood intervals checked
  for coverage over seeded repetitions; a deliberately non-identifiable
  configuration included and shown to be *reported as such* rather than fitted
  confidently; residual-model improvement measured on held-out data only.
- **Alternatives table must name:** `lmfit`, `scipy.optimize.least_squares`,
  `emcee`, `SALib`, `pysindy`.

## P063 SafeModeTree — medium · L3 · AI · autonomous-aerospace-systems

Package `safemodetree`. **From an FMEA table to a fault-response policy, with a
reachability proof that every fault reaches a safe mode.**

P025 FDIScope detects and isolates faults. P034 FaultInject injects them.
Nothing builds or checks the *response*.

- Declarative input: faults (effect, detectability, declared detection latency),
  modes (including declared safe modes), responses (transition, cost,
  precondition), and an observability model saying which faults are
  distinguishable from which observations.
- Deterministic core: build the response automaton; **check that from every
  fault state a declared safe mode is reachable under the declared detection
  assumptions**; report unreachable safe modes, fault states with no response,
  responses whose preconditions can never hold, and ambiguity classes where two
  faults are indistinguishable but demand different responses — that last one is
  the finding the product exists for.
- **AI component:** a learned fault classifier over the observation vector with
  **calibrated confidence**, benchmarked against the declared threshold logic
  baseline, and — the part that makes it meaningful rather than bolted on — the
  policy consumes the confidence: a low-confidence classification routes to the
  conservative response for the whole ambiguity class instead of the
  fault-specific one. Measure the cost of that conservatism.
- **Validation (L3):** reachability results cross-checked against exhaustive
  enumeration on small automata; a planted unreachable-safe-mode case detected;
  classifier calibration measured (reliability curve, ECE) and compared to the
  threshold baseline at equal false-alarm rate.
- **Alternatives table must name:** `transitions`, `python-statemachine`, NuSMV
  / nuXmv, `pyNuSMV`. Say that a model checker is the right tool for large
  reachability problems and that this product's contribution is the FMEA-shaped
  input and the ambiguity-class analysis.

## P064 RegressSelect — medium · L2 · AI · testing

Package `regressselect`. **Which tests can this diff skip, and what did skipping
them cost.**

Aimed squarely at repositories like this one: numerical suites where a full run
is tens of minutes and most of it is untouched by any given change.

- **Baseline 1 (safe, dependency-based):** import-graph reachability from
  changed modules to tests. Safe by construction, usually selects almost
  everything, and the README must say so.
- **Baseline 2 (history-based):** prioritisation by historical failure
  co-occurrence; measured with **APFD** (average percentage of faults detected).
- **AI component:** a learned selector over change features (files touched,
  churn, test–module distance, historical co-failure) producing a *ranked* order
  with a probability, benchmarked against both baselines on **APFD and
  safety-loss** — the fraction of real failures a selection would have missed.
  The safety-loss number is the one that decides whether this is usable, and it
  must be reported even when it is bad.
- **Validation (L2):** mutation-style evaluation — seeded faults injected into a
  bundled sample project, each selection strategy scored on faults-detected vs
  tests-run, with the full-suite upper bound shown; APFD computed on a committed
  fixture history so the number is reproducible.
- **Alternatives table must name:** `pytest-testmon`, `pytest-picked`,
  `coverage.py`, `pytest-xdist`. Be explicit that `pytest-testmon` already does
  coverage-based selection well and that this product's contribution is the
  measured safety-loss accounting, not the selection itself.

## P065 HazardTrace — medium · L2 · no AI · aerospace-assurance

Package `hazardtrace`. **STPA control-structure hazard analysis as a
machine-checkable artifact.**

P055 AssureGraph checks an assurance argument; P060 TraceAudit links
requirements to tests. Neither produces or checks the hazard analysis those
arguments rest on.

- Declarative YAML: losses, hazards, system-level constraints, the control
  structure (controllers, actuators, sensors, controlled process, control
  actions, feedback), and unsafe control actions (UCAs) in the four STPA classes
  — not provided, provided unsafely, wrong timing/order, stopped too soon or
  applied too long.
- Computed checks, the product's whole value: **UCA enumeration completeness**
  (every control action × four classes either has a UCA or an explicit
  justified exclusion), hazards with no constraint, constraints with no
  requirement, UCAs with no loss scenario, loss scenarios with no mitigation,
  and mitigations with no evidence artifact. Cross-links to P060-style
  requirement identifiers and junit test identifiers so the coverage is computed
  from the repository, not asserted.
- Deterministic, deliberately. There is no AI component because the thing being
  checked is a declared argument, and a learned component here would generate
  text that looks like analysis. Say that in the README.
- **Validation (L2):** completeness checker proved on hand-constructed cases
  with known answers (shown in test comments); a worked example hazard analysis
  of a real control loop from this portfolio (P024 DetumbleSim's B-dot loop)
  carried end to end with its genuine gaps reported rather than tidied.
- **Alternatives table must name:** XSTAMPP / STPA-Tool, `safety-architect`,
  and plain spreadsheets, which is what most teams actually use.

## P066 BudgetRoll — compact · L1 · no AI · aerospace-assurance

Package `budgetroll`. **Margin and budget rollup arithmetic that does not lie
about how uncertainties combine.**

The L1 slot that closes that quota. Educational level, and chosen for it: this
is arithmetic everyone does in a spreadsheet and gets subtly wrong.

- Hierarchical budgets (mass, power, data volume, link margin) with per-item
  contingency, declared maturity-based margin policies, and aggregation that is
  **explicit about linear vs root-sum-square** combination, with the
  independence assumption RSS requires stated at every node and checked against
  a declared correlation structure.
- Sensitivity: which leaf item moves the top-level margin most; what margin is
  left at each level; which nodes are RSS-aggregated under an assumption the
  declaration contradicts.
- **Validation (L1):** hand-calculated known-answer tests shown in test
  comments for each aggregation rule; a worked CubeSat-scale power and link
  budget reproduced against textbook values with the reference cited; property
  tests for the algebraic identities (RSS ≤ linear, associativity of
  independent aggregation).
- **Alternatives table must name:** spreadsheets, `uncertainties`, `pint`,
  `openmdao`. `uncertainties` does linear error propagation better; the
  contribution here is the margin policy and the assumption checking.

## P067 AcqSearch — compact · L2 · AI · pointing-acquisition-tracking

Package `acqsearch`. **Spatial acquisition search for an optical terminal:
expected time to acquire, and which scan pattern actually wins.**

The PAT domain is covered for *tracking* (P002 TrackBench, P005 JitterScope,
P008 CentroidNet) and not at all for *acquisition*.

- Scan patterns — spiral, raster, Lissajous, random-restart — over a declared
  initial pointing-uncertainty distribution (2-D Gaussian cone), with declared
  beam divergence, detector field of view, dwell time, slew rate and detection
  probability per dwell including false alarms.
- Deterministic core: coverage fraction vs time per pattern; **expected
  acquisition time** computed analytically where the geometry allows and by
  seeded Monte Carlo otherwise, with the two cross-checked; optimal dwell time
  from the detection-probability/dwell trade; probability of acquisition within
  a declared deadline.
- **AI component:** a learned adaptive scan-order policy that reorders the next
  dwell from the posterior over target location given the misses so far
  (Bayesian update is the analytic baseline; the learned policy must be
  benchmarked against both the fixed pattern and the Bayesian-greedy one). The
  honest expected result is that Bayesian-greedy is hard to beat — **report
  that if it holds.**
- **Validation (L2):** analytic expected acquisition time vs Monte Carlo over
  seeds; coverage fraction checked against a geometric closed form for the
  spiral; detection-probability model cited (Andrews & Phillips 2005 for the
  irradiance statistics; a cited reference for the dwell detection model).
- **Alternatives table must name:** there is no direct open-source equivalent —
  say so plainly and name the closest primitives (`numpy` geometry, `poliastro`
  for pointing geometry) rather than inventing a competitor.

## P068 DegradePredict — compact · L2 · AI · autonomous-aerospace-systems

Package `degradepredict`. **When does this subsystem cross its limit, and is the
interval around that date calibrated.**

P058 TelemDrift says *something changed*. P035 TelemetryOOL says *it is out of
limits now*. Neither forecasts the crossing.

- **Baselines:** linear and exponential degradation fits with analytic
  prediction intervals; a declared physics-shaped model (e.g. battery capacity
  fade, wheel friction growth) where the form is known and cited.
- **AI component:** a learned forecaster (gradient-boosted or Gaussian-process
  regression on lag features) producing a **distribution over crossing time**,
  benchmarked against both baselines on held-out trajectories.
- The part that makes it honest: **censoring**. Most trajectories have not
  crossed the limit yet, and scoring only the ones that have is survivorship
  bias. The product must handle right-censored data explicitly and measure
  prediction-interval coverage on censored trajectories, not just complete ones.
- **Validation (L2):** coverage of the predicted crossing-time interval measured
  over seeded synthetic trajectories with known crossing times, complete and
  censored, reported separately; CRPS or pinball loss against baselines;
  calibration curve for the crossing-time distribution.
- **Alternatives table must name:** `lifelines`, `scikit-survival`, `statsmodels`,
  `prophet`. Say that `lifelines` is the better survival library and that the
  contribution is the limit-crossing formulation and the censored coverage
  measurement.

## P069 ScenarioCov — compact · L2 · AI · testing

Package `scenariocov` (`covmetrics` was taken on PyPI). **Which regions of the
scenario space has this campaign not touched.**

P054 RareVerify sizes a campaign for a rare event. P034 FaultInject accounts for
fault coverage. Neither answers the input-space question.

- Deterministic core: **t-way combinatorial coverage** over discretised
  scenario parameters (computed, with the uncovered t-tuples enumerated, not
  just a percentage); equivalence-class and boundary coverage against declared
  classes; a grid/cell occupancy measure over continuous parameters with the
  discretisation stated as an assumption, because it is one.
- **AI component:** a learned **gap finder** — a density or one-class model over
  executed scenarios that proposes the next batch of scenarios in sparse
  regions, benchmarked against uniform random and greedy-max-min-distance on
  coverage achieved per scenario executed. If greedy max-min wins, which is
  plausible, **say so**.
- **Validation (L2):** t-way coverage cross-checked against exhaustive
  enumeration on small parameter spaces; a planted uncovered region detected;
  coverage-per-scenario curves for all three strategies over seeded runs with
  the spread shown, not just the mean.
- **Alternatives table must name:** `allpairspy`, Microsoft PICT, `hypothesis`,
  `scenariogeneration` (OpenSCENARIO). `hypothesis` already searches input
  spaces far better; the contribution is the *reporting* of what a finished
  campaign missed.

## P070 WatchdogSpec — compact · L2 · no AI · hardware-in-the-loop

Package `watchdogspec`. **Declare your heartbeat and liveness deadlines, then
measure whether the watchdog would actually have caught it.**

P036 RtClock provides timing primitives and schedulability arithmetic. Nothing
specifies or checks the supervision layer on top.

- Declarative spec: tasks with heartbeat periods, missed-beat tolerances,
  liveness deadlines, escalation ladders (log → restart task → safe mode →
  reset) with per-rung latency, and declared recovery costs.
- Deterministic core: **worst-case detection latency** per fault class computed
  from the spec arithmetic and cross-checked against a seeded discrete-event
  simulation of beat arrivals with jitter; false-trip probability from declared
  jitter statistics; escalation ladders that can never reach their final rung
  within a declared deadline; tasks whose tolerance exceeds the deadline they
  are supposed to protect — a specification defect the arithmetic finds for free.
- Deterministic, deliberately: there is nothing to learn here, and the README
  must say that rather than attach a model.
- **Validation (L2):** closed-form worst-case latency vs discrete-event
  simulation over seeds; hand-calculated known-answer tests for the escalation
  arithmetic shown in test comments; a planted
  tolerance-exceeds-deadline spec detected.
- **Alternatives table must name:** `watchdog` (the filesystem library, a name
  collision worth noting), systemd watchdog semantics, `simpy` for the
  simulation. No open tool does the aerospace escalation-ladder arithmetic.

---

## Binding rules for every build agent in this batch

1. Read `templates/PRODUCT_BUILD_GUIDE.md` **first**, including the
   shipped-defect classes at the end. The dependency-bound section added
   2026-10-14 is new and it is binding: **bound every dependency on both sides
   with a range you actually ran the suite at.**
2. Validation output layout is **flat**: `validation/<script-name>_output.txt`
   beside the script. Binding for Batch 07 onward.
3. Count tests from **junit XML**, never pytest stdout. Report the junit numbers
   in your final message.
4. Re-run every README-quoted command as the **last** step before handing back,
   and replace quoted output with what the run actually printed.
5. Never assert bit-exact equality over a floating-point pipeline, and never
   assert equality of a boolean produced by a strict threshold comparison.
6. Anchor scratch patterns in `.gitignore` with a leading slash, and run
   `git check-ignore -v` against every fixture you are supposed to ship.
7. Implement the classical/analytic baseline **before** any ML model, and report
   honestly when the baseline wins. Those results are research evidence and are
   never removed, softened or retuned away.
8. Never describe the product as flight-safe, certified, mission-ready or
   production-ready. Research-grade, not flight-qualified.
9. No marketing language, no emoji, no exclamation marks, anywhere.
10. Every equation carries source, units, assumptions and validity range, and
    every citation is real and verifiable. Inventing one is the single worst
    thing you can do here.
