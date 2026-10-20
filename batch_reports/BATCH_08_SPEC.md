# Batch 08 Specification — The measurement layer around mature tools: safety of flight, what to send, and whether the clock is right

Products P071–P080. Composition: **2 flagship, 3 medium, 5 compact.**
**Eight of ten carry a meaningful AI component**, against the mandated minimum
of seven. Validation: **3 × L3, 6 × L2, 1 × L3-hardware-pending.**

Specified 2026-10-20 from a clean, unblocked state: Batch 07 was published the
same session and nothing is held behind this batch.

| ID | Name | Package | Class | Validation | AI | Category |
|---|---|---|---|---|---|---|
| P071 | ConjProb | `conjprob` | flagship | L3 | yes | aerospace-gnc |
| P072 | DownlinkPick | `downlinkpick` | flagship | L3 | yes | satellite-communications |
| P073 | ContactSched | `contactsched` | medium | L2 | yes | satellite-communications |
| P074 | OrbitDrift | `orbitdrift` | medium | L3 | yes | aerospace-gnc |
| P075 | ReliaBlock | `reliablock` | medium | L2 | no | aerospace-assurance |
| P076 | PointAhead | `pointahead` | compact | L2 | no | pointing-acquisition-tracking |
| P077 | RelFileDelivery | `relfiledelivery` | compact | L2 | yes | satellite-communications |
| P078 | CmdGuard | `cmdguard` | compact | L2 | yes | autonomous-aerospace-systems |
| P079 | TimeTransfer | `timetransfer` | compact | **L3, hardware-pending** | yes | hardware-in-the-loop |
| P080 | BoundCompress | `boundcompress` | compact | L2 | yes | satellite-communications |

## Why this batch, stated against what already exists

The honest problem with a mission at 70 products is that the easy gaps are
gone. Every product below had to survive one question: **what does it do that
an existing, mature tool does not?** Where the answer was "nothing", the
product was not specified.

The answer that recurs — and it is the mission's most defensible position, not
a dodge — is that the mature tool **computes** and this batch **measures**.
`astropy` will give an aberration correction; it will not give a point-ahead
error budget with each term attributed. `numcodecs` and `zfpy` will compress an
array; they will not certify that a declared error bound held on every sample
and report the rate-distortion cost of guaranteeing it. `cfdp-py` implements
CFDP; it does not tell an operator what delivery time to expect over a link
with a declared outage distribution. Six of the ten below are explicitly the
verification layer around a tool that is named, credited, and recommended in
the product's own README.

Four gaps are genuine absences rather than measurement layers:

- **Conjunction assessment is entirely absent from 70 products** — the single
  largest uncovered area in safety of flight. P030 KeepOut is attitude keep-out
  (slew constraints against bright bodies), not orbital close approach.
- **Downlink value triage is absent.** P061 ActivityPlan schedules activities
  against resources; nothing decides which of the data already recorded is
  worth the contact time.
- **Reliability/redundancy arithmetic is absent.** P063 SafeModeTree does FMEA
  response policy, P065 HazardTrace does STPA, P055 AssureGraph does assurance
  arguments — none computes a redundancy configuration.
- **Time transfer is absent.** P036 RtClock is local timing primitives and
  schedulability; P079 is offset and skew estimation across a link.

Products deliberately **not** specified after failing the worth-building test:

- A second onboard-inference product. P033 EdgeInfer already does budget
  verification with an analytic roofline baseline and P039 LatencyNet already
  does tail-latency prediction. A third would be re-measurement.
- A HAL/HIL harness product. **P031 HilForge already is one**, with
  simulation/device parity, dry-run rehearsal and a benchmark harness. The L4
  scaffolding that the task prompt prescribes is therefore applied *inside*
  P079 rather than sold as its own product.
- A change-detection product. P053 TwinInvalidate and P058 TelemDrift cover
  CUSUM/EWMA/GLR/Page-Hinkley/KS/ADWIN between them, both scored at equal
  operating point. There is nothing left to add honestly.
- A radiation-environment product. P040 BitFlipSim covers SEU injection, and a
  dose-accumulation model would depend on environment data this mission cannot
  measure or honestly source offline.

## Level 4 — what this batch does and does not do about it

**L4 stands at 0 of 5 for the fifteenth consecutive batch and this batch does
not change that number.** This is stated plainly because the task prompt
contains two clauses that cannot both be satisfied:

> *"Level 4 validation is 0 of 5 and is overdue — start it in the next batch."*

> *"...label those products `Level 3, hardware-pending` — never Level 4 — until
> measured timing and resource use come from the Jetson Orin Nano itself."*

Taken together these order the L4 *work* to start while forbidding the L4
*label*. That is the correct instruction and this batch honours it literally:
**P079 TimeTransfer is built with the full L4 scaffolding** — hardware
abstraction layer with simulation/device parity, simulation mode, dry-run mode,
documented deployment and recovery procedure, and a benchmark harness — and is
labelled **`Level 3, hardware-pending`**, never L4.

P079 was chosen for it because clock offset and skew estimation is the one
product in this batch whose central claims are *inherently* hardware
measurements: a simulated clock cannot evidence a real oscillator's drift. A
product whose claims are numerical would gain nothing but a label.

What this does not do is move 0/5. **Four products — P031, P033, P041, P043 —
are already complete to Level 3 and already labelled `hardware-pending` on the
identical blocker.** P079 will be the fifth. Adding a fifth waiting product is
not progress toward 5/5; it is five products waiting instead of four. The
blocker is unchanged and is owner action: **the Jetson Orin Nano powered on and
reachable, its hostname or IP, and passwordless SSH from the Mac.** With that,
five products reach L4 in one session and the quota closes. Without it, no
cloud session can move it, and no session should relabel on simulated numbers.

## PyPI name check — method and result

All ten names probed with `pip index versions` against the live index on
2026-10-20. **The probe was validated against controls before being trusted**,
because a name probe that cannot distinguish "absent" from "network failure"
reports everything as free:

- Known-taken controls: `numpy` → `(2.5.4)`, `pulp` → `(4.1.0)`. Both TAKEN.
- Known-free controls: `beamtwin`, `traceaudit` — this mission's own previously
  validated free names. Both FREE.

With the probe shown to discriminate in both directions, all ten batch names
returned FREE: `conjprob`, `downlinkpick`, `contactsched`, `orbitdrift`,
`reliablock`, `pointahead`, `relfiledelivery`, `cmdguard`, `timetransfer`,
`boundcompress`.

**Every package named in an alternatives table below was also probed and
confirmed to exist**, so no product recommends a package that is not there:
`reliability (0.9.0)`, `spacepackets (0.32.0)`, `cfdp-py (0.7.0)`,
`numcodecs (0.17.0)`, `zfpy (1.0.1)`, `ntplib (0.4.0)`, `sgp4 (2.27)`,
`astropy (8.0.2)`, `pulp (4.1.0)`.

**Reproducibility note carried into this batch:** `numpy` resolved to `2.5.4`
in this session's container. The 2026-10-18 session resolved `2.5.3` on the
same unbounded specifier, and the committed Batch 07 outputs name `2.5.4`. The
`unbounded_bare_names` carry-over is therefore live and has now produced three
different observations. **Every product in this batch declares every dependency
bounded on both sides**, with no bare floors.

---

## P071 ConjProb — flagship · L3 · AI · aerospace-gnc

Package `conjprob`. **Collision probability for a close approach, with the
regime it is valid in stated next to the number.**

Takes two states with covariance at a common epoch (or TLE pairs propagated to
one), a combined hard-body radius, and produces the time of closest approach,
the miss distance, the relative geometry, and **a probability of collision
computed by more than one method with their disagreement reported as the
primary output rather than a footnote.**

The reason this is a flagship and not a one-function utility: the published Pc
methods disagree, each has a validity regime, and **the regimes are where the
operational mistakes live.** In particular the dilution region — where
*increasing* covariance *decreases* computed Pc — produces a reassuring number
from a worse state, and a tool that returns a scalar without naming it is
actively misleading.

- **Baseline 1 (analytic, 2-D):** Foster–Estes short-encounter formulation on
  the conjunction plane, the operational default.
- **Baseline 2 (independent analytic):** Chan's equivalent-area series and
  Alfano's method, implemented separately. Three analytic paths over the same
  integral, cross-checked against each other.
- **Baseline 3 (ground truth):** direct 2-D numerical quadrature on the
  encounter plane and 3-D Monte Carlo over the relative state, with confidence
  intervals. This is what the analytic methods are scored against — Monte Carlo
  is the reference, not a competitor.
- **AI component:** a learned screening classifier that predicts, from cheap
  geometry and covariance features alone, whether a conjunction will exceed a
  declared Pc threshold — the operational problem of triaging thousands of
  screened pairs without running the full computation on each. It is scored
  against the obvious analytic screens (miss distance thresholding, and
  Mahalanobis distance in the encounter plane) on **recall at fixed
  false-positive rate**, because a missed high-Pc event is not symmetric with a
  wasted computation. **Report honestly if a Mahalanobis threshold matches it**
  — that is the likely outcome and it is a publishable result.
- **Validation (L3):** analytic methods against numerical quadrature to a
  stated tolerance across a swept grid of miss distance, covariance ratio and
  hard-body radius, with the **regime boundaries where each method departs
  reported as measured curves, not assertions**; Monte Carlo convergence shown;
  the dilution region exhibited explicitly as a Pc-versus-covariance curve with
  its non-monotonic turn, and a dilution warning raised by the library itself;
  linear-relative-motion assumption violated deliberately at long encounter
  durations to show where the 2-D formulation fails.
- **Alternatives table must name:** `astropy`, `sgp4`, and Orekit (Java,
  via a note that it is the mature full-stack option and this is not a
  replacement for it). Say plainly that operational conjunction assessment is
  done by CARA/CSpOC-grade tooling against catalogues this product does not
  have, and that this product's contribution is **method cross-validation and
  regime reporting on a conjunction you already have**, not screening a
  catalogue.
- **Must not say:** that any output is suitable for a manoeuvre decision.

## P072 DownlinkPick — flagship · L3 · AI · satellite-communications

Package `downlinkpick`. **Choose what to downlink when the contact is shorter
than the queue, and say what the choice cost.**

Takes a recorded-data inventory (size, age, declared science or operational
value, expiry, dependency on other products, compressibility) and a contact
plan (windows, rates, per-window reliability) and produces a selection,
**plus the marginal value of the next unit of contact time and the identity of
what was left behind.**

- **Baseline 1 (greedy):** value-density ordering with window packing. The
  operational default and genuinely hard to beat.
- **Baseline 2 (exact):** multi-knapsack MILP over windows via PuLP, giving the
  true optimum on instances small enough to solve. Greedy is scored against it.
- **AI component:** a learned value model that predicts realised value from
  product features where declared value is missing or stale, and a learned
  compressibility predictor feeding the size estimate. Benchmarked against the
  declared-value passthrough and against a simple size/age heuristic. **The
  likely honest finding is that the exact solver on declared values beats
  everything and the learned value model only helps where declarations are
  absent; state that if it is what the measurement shows.**
- **Validation (L3):** MILP optimum against exhaustive enumeration on small
  instances; greedy optimality gap measured over a seeded instance family with
  its distribution reported, not just a mean; value conservation and
  no-double-downlink invariants; contact-time accounting cross-checked against
  an independent integral of rate over window; selection stability under
  perturbed value estimates measured, because a selector that flips on noise is
  unusable regardless of its optimality.
- **Alternatives table must name:** `pulp`, OR-Tools CP-SAT, `numcodecs`.
  Say plainly that CP-SAT is the better solver and the contribution here is the
  **contact-window and expiry accounting plus the left-behind report**, not the
  search.

## P073 ContactSched — medium · L2 · AI · satellite-communications

Package `contactsched`. **Ground-station contact scheduling when several
missions want the same antenna, with the fairness cost measured.**

Takes station locations and masks, spacecraft ephemerides (via `sgp4`), per-
mission requests with priorities and minimum-contact requirements, and produces
an assignment — **plus contention and fairness metrics: who was starved, by
how much, and what the priority scheme actually delivered versus what it
promised.**

- **Baseline 1 (analytic):** visibility computation from ephemeris and mask,
  then priority-ordered first-fit. Documented and suboptimal.
- **Baseline 2 (exact):** MILP assignment with declared fairness constraints.
- **AI component:** a learned conflict predictor that estimates, before
  scheduling, which requests will be unsatisfiable — so a request can be
  renegotiated rather than silently dropped. Scored against a demand-over-
  supply ratio baseline per station-window.
- **Validation (L2):** visibility windows cross-checked against an independent
  geometric computation and against `sgp4` at finer time steps for convergence;
  no-overlap and minimum-contact invariants asserted; fairness metrics
  (max-min share, Jain index) computed and their behaviour under priority
  inversion exhibited; MILP against enumeration on small instances.
- **Alternatives table must name:** `sgp4`, `astropy`, `pulp`, OR-Tools
  CP-SAT. Note that real ground-station networks schedule with vendor-specific
  systems against catalogues and SLAs this does not model.

## P074 OrbitDrift — medium · L3 · AI · aerospace-gnc

Package `orbitdrift`. **How wrong the ephemeris has become, as a measured curve
against epoch age.**

Takes a TLE or state at an epoch and produces position-error growth against
propagation time — **with a prediction interval whose coverage is measured, not
assumed** — so an operator can answer "is this ephemeris still good enough for
this decision" instead of guessing from a rule of thumb.

Honest scoping, stated in the README: this product has **no access to precision
ephemerides or a TLE archive**, so it cannot measure true error against truth.
What it measures is **model divergence**, which is a lower bound on error and
is useful as such:

- **Baseline 1:** SGP4 self-consistency — propagate from an earlier epoch and
  compare against SGP4 from a later epoch of the same object where both are
  supplied by the user, the standard practical proxy.
- **Baseline 2 (independent dynamics):** numerical integration with declared
  perturbations (J2 through J4, drag with a declared atmosphere, third-body
  luni-solar), diverging from SGP4 at a rate that is itself the signal.
- **AI component:** a learned error-growth predictor from orbit and object
  features (altitude, eccentricity, inclination, B\*, epoch age) producing a
  calibrated interval, benchmarked against the analytic growth models
  (linear-in-time, quadratic-in-time, and drag-dominated exponential-in-time
  forms) on interval coverage and width. **If an analytic form matches, say so
  — that is the expected outcome at most altitudes.**
- **Validation (L3):** energy and angular-momentum conservation on the
  integrator with no non-conservative terms, to a stated tolerance; integrator
  convergence under step refinement; SGP4 reproduced against its own published
  test vectors; divergence curves reproduced across seeds; interval coverage
  measured against held-out cases and reported as a coverage table, including
  where it is miscalibrated.
- **Alternatives table must name:** `sgp4`, `astropy`, Orekit. Say plainly
  that for real accuracy assessment the answer is precision ephemerides and an
  orbit-determination package, and this product is for **deciding whether the
  TLE you have is stale**, which those do not answer directly.
- **Must not say:** that any output is a covariance suitable for conjunction
  assessment. Cross-reference P071 and state that its covariance must come from
  orbit determination, not from this.

## P075 ReliaBlock — medium · L2 · no AI · aerospace-assurance

Package `reliablock`. **Redundancy arithmetic that is explicit about the
assumption doing the work.**

Takes a declared reliability block diagram — series, parallel, k-of-n, standby
with switching reliability, shared-load, common-cause via a declared beta
factor — with per-block failure distributions, and computes system reliability
over a mission profile, importance measures (Birnbaum, Fussell-Vesely),
and a minimal-cut-set enumeration.

**No AI component, deliberately.** This is arithmetic with a correct answer;
a learned model would be a liability and there is nothing to learn.

The contribution, stated against the mature alternative: **the `reliability`
package (0.9.0) exists, is good, and the README must say so and say when to use
it instead** — it is the better choice for distribution fitting, ALT and
repairable-system analysis. What it does not do is take a *declared diagram* as
a checkable artifact and report **how much of the answer rests on the
independence assumption.** The headline output here is reliability computed
with and without a declared common-cause beta, side by side, because
independence is the assumption that fails in flight hardware and the one most
often left implicit.

- **Validation (L2):** closed-form checks on every structure (series, parallel,
  k-of-n against the binomial sum, standby against the convolution integral)
  to a stated tolerance; Monte Carlo simulation of the whole diagram agreeing
  with the analytic result within its confidence interval; minimal cut sets
  cross-checked against exhaustive state enumeration on small diagrams;
  importance measures verified against finite-difference perturbation of block
  reliabilities; a known textbook bridge network reproduced with its published
  answer cited.
- **Alternatives table must name:** `reliability`, `lifelines`, and say when
  each is the right tool.

## P076 PointAhead — compact · L2 · no AI · pointing-acquisition-tracking

Package `pointahead`. **Point-ahead angle for an optical crosslink, with every
term in the error budget attributed.**

Computes the point-ahead angle from relative geometry and velocity — the
transmit beam must lead the apparent receive direction because the far terminal
moves during the light transit — and produces **an error budget in which each
contribution is a separate, individually checkable line**: relative velocity
knowledge, ephemeris staleness, light-time iteration residual, stellar and
velocity aberration terms, and the beam-divergence context that says whether
the total matters.

Small and sharply scoped, which is why it is compact. The value is that the
total is useless without the breakdown: a 5 µrad point-ahead error means one
thing on a 50 µrad beam and another on a 5 µrad beam, and the dominant term
determines what to fix.

- **Baseline:** closed-form first-order aberration, and an iterated light-time
  solution as the reference the closed form is scored against.
- **Validation (L2):** aberration terms cross-checked against `astropy`'s
  independent implementation to a stated tolerance; light-time iteration
  convergence shown and its residual reported; error budget verified by
  propagating each input perturbation numerically and confirming the
  linearised sensitivity matches; geometric sanity invariants (point-ahead
  vanishing at zero relative transverse velocity, sign reversal under reversed
  geometry).
- **Alternatives table must name:** `astropy`, `sgp4`. Say plainly that
  `astropy` is the correct tool for aberration itself and the contribution here
  is **the crosslink-specific geometry and the attributed budget**.

## P077 RelFileDelivery — compact · L2 · AI · satellite-communications

Package `relfiledelivery`. **What a file transfer will actually cost over an
intermittent link — measured, not another protocol implementation.**

**This is explicitly not a CFDP implementation.** `cfdp-py` (0.7.0) and
`spacepackets` (0.32.0) exist, are maintained, and the README must name them
and recommend them for the protocol itself. What they do not provide is an
answer to the operator's question: *given this link — these outage durations,
this loss rate, this asymmetry, these contact windows — how long will this file
take, how much overhead will I pay, and what is the spread?*

So this is a **measurement harness over declared link conditions**, driving
deferred-NAK, immediate-NAK and unacknowledged transfer modes against a seeded
link model, reporting delivery time distributions, retransmission overhead and
completion probability per contact.

- **Baseline 1 (analytic):** closed-form expected delivery time under
  geometric loss with no outages — exact, and the thing the simulation is
  verified against.
- **Baseline 2:** plain Monte Carlo over the link model.
- **AI component:** a learned delivery-time predictor from link and file
  features producing a calibrated interval, benchmarked against the analytic
  model and against a measured-mean lookup. **The analytic model is expected to
  win wherever its assumptions hold; the learned model earns its place only in
  the regime with correlated outages, if it does at all. Report either way.**
- **Validation (L2):** simulation against the closed form in the no-outage
  limit to a stated tolerance; completion invariant (delivered bytes equal
  source bytes, verified by hash) on every run; retransmission accounting
  cross-checked against an independent count; seeded reproducibility;
  distributions reported as quantiles with the tail stated separately, since a
  mean delivery time over an intermittent link is close to meaningless.
- **Alternatives table must name:** `cfdp-py`, `spacepackets`.

## P078 CmdGuard — compact · L2 · AI · autonomous-aerospace-systems

Package `cmdguard`. **Check a command sequence against the spacecraft state it
will actually meet.**

Takes a declared command dictionary (preconditions, mode gating, resource draw,
minimum inter-command timing, mutual exclusions, irreversibility flags), a
declared mode machine, and a candidate sequence — and reports violations
**with the state trace that reaches each one**, because "command 47 is illegal"
is not actionable and "command 47 is illegal because command 12 left the
payload in SAFE and nothing re-enabled it" is.

- **Baseline 1 (exhaustive):** forward symbolic execution of the sequence
  through the mode machine, which is exact for a single linear sequence and is
  the ground truth here.
- **Baseline 2:** per-command static precondition lint, the thing people
  actually do, which misses every ordering defect — and the measurement of how
  many it misses is a deliverable.
- **AI component:** a learned risk ranker over commands, ordering them by
  likelihood of being involved in a violation, for triaging long sequences
  where exhaustive tracing is cheap but human review is not. Scored against
  irreversibility-flag ordering and against resource-draw ordering on
  precision at k. **A flag-based ordering matching it is a result to publish,
  not to retune away.**
- **Validation (L2):** forward execution verified against hand-constructed
  sequences with known violations, including a negative control set with no
  violations that must report none; determinism under re-run; the static-lint
  miss rate measured on a seeded family of injected ordering defects; timing
  constraint arithmetic unit-checked; an irreversible-command gate that must
  refuse to pass a sequence where an irreversible command precedes an
  unverified precondition.
- **Alternatives table:** no mature equivalent was found on the index; the
  README must say that plainly rather than inventing a comparison, and name the
  honest substitute, which is mission-specific scripted checks plus review.
- **Must not say:** that a passing sequence is safe to uplink.

## P079 TimeTransfer — compact · **L3, hardware-pending** · AI · hardware-in-the-loop

Package `timetransfer`. **Clock offset and skew across a link, with the
uncertainty measured rather than quoted.**

Two-way time transfer: estimates offset, skew and drift from timestamp
exchanges under asymmetric and variable delay, and reports stability as an
Allan deviation over tau with the confidence interval at each tau — plus the
thing that distinguishes it, **a measured uncertainty on the offset estimate
itself**, not a nominal figure from a datasheet.

**This is the batch's L4-scaffolded product** and carries the full set, per the
task prompt, while being labelled `Level 3, hardware-pending`:

1. **Hardware abstraction layer** over the timebase with simulation/device
   parity — the same estimator code paths run against a simulated clock and a
   real one, and parity is itself asserted by test.
2. **Simulation mode** with a seeded clock model (declared drift, jitter,
   asymmetry, outages) — fully reproducible, no hardware.
3. **Dry-run mode** that exercises the full acquisition and estimation path
   against a recorded exchange without touching a device.
4. **Documented deployment and recovery procedure** for the Jetson Orin Nano,
   including what to do when the exchange is lost mid-estimate.
5. **Benchmark harness** that produces the timing and resource table, ready to
   run, with the table's hardware rows left **empty and explicitly marked
   unmeasured** rather than filled from simulation.

- **Baseline 1 (analytic):** the standard two-way offset estimator assuming
  symmetric delay — exact under its assumption, and the asymmetry-induced bias
  is derived in closed form and verified.
- **Baseline 2:** least-squares and Kalman skew estimation over the exchange
  history.
- **AI component:** a learned delay-asymmetry estimator that corrects the
  offset bias from exchange features, benchmarked against the symmetric
  assumption and against a median-filtered estimator. **A median filter
  matching it is the likely result and is to be reported.**
- **Validation (L3, hardware-pending):** closed-form bias reproduced against a
  declared asymmetry; estimator consistency and convergence under refinement;
  Allan deviation verified against synthetic noise processes with known slopes
  (white phase, white frequency, random walk) and the fitted slopes reported
  against theory; seeded reproducibility; **HAL simulation/device parity
  asserted by test**; coverage of the reported offset uncertainty measured.
  **The L4 rows — measured timing and resource use on the Jetson — stay empty
  and marked unmeasured. The product is never labelled L4 on simulated
  numbers.**
- **Alternatives table must name:** `ntplib`, and PTP/`linuxptp` as the mature
  answer for real time synchronisation, with a plain statement that this is a
  measurement and characterisation tool, not a time-sync daemon.

## P080 BoundCompress — compact · L2 · AI · satellite-communications

Package `boundcompress`. **Certify that the error bound held, and report what
the guarantee cost.**

**Not a codec.** `numcodecs` (0.17.0) and `zfpy` (1.0.1) exist and are the
right tools for compressing an array; the README must say so. What they do not
do is answer the flight-software question: *can I prove that no sample exceeded
my declared error bound, and what did I pay in rate to be able to prove it?*

Takes a telemetry array and a declared per-sample absolute or relative error
bound, and produces a compressed representation plus **a certificate: the
measured maximum per-sample error, where it occurred, and a rate-distortion
curve against the named codecs at matched distortion** — so the cost of the
guarantee is visible rather than asserted.

- **Baseline 1 (analytic):** lossless entropy coding of quantised residuals at
  the bound-implied quantiser step — the honest, boring, hard-to-beat default,
  with the achievable rate bounded below by the measured residual entropy.
- **Baseline 2:** the named third-party codecs at matched distortion.
- **AI component:** a learned residual predictor (the prediction step of a
  predictive coder) reducing residual entropy ahead of the same entropy coder,
  benchmarked against linear and polynomial extrapolation predictors on
  compressed rate at identical, verified distortion. **If a second-order linear
  predictor matches it on real-shaped telemetry, that is the result.**
- **Validation (L2):** **the error bound asserted on every sample of every
  test array, as a hard test, not a summary statistic** — the central claim and
  the thing most easily faked by reporting a mean; round-trip reconstruction
  verified; rate measured as actual encoded bytes, never estimated from
  entropy; rate-distortion comparison against the named codecs at matched
  measured distortion; a deliberate bound-violation negative control that the
  certificate must catch and fail on; determinism across runs.
- **Alternatives table must name:** `numcodecs`, `zfpy`, and CCSDS 121.0/122.0
  as the standards, with a plain statement that a flight implementation should
  use a standardised codec and this product's contribution is the
  **certificate and the cost of the guarantee**.

---

## Binding rules for every build agent in this batch

Read `templates/PRODUCT_BUILD_GUIDE.md` first. In addition:

1. **Pin a virtual environment per agent at the start, before measuring
   anything.** On 2026-10-16 three of five concurrent agents independently
   observed the container's shared system Python mutating mid-build — `numpy`
   2.5.3 → 2.5.4 → 2.5.0 with a transient broken install and bus errors,
   `scikit-learn` metadata disagreeing with the imported module. Concurrent
   agents in one container are not working against a stable interpreter.
2. **Declare every dependency bounded on both sides.** No bare floors. `numpy`
   has now resolved to three different patch releases across sessions on an
   unbounded specifier.
3. **Implement the classical or analytic baseline first, and report honestly
   when it wins.** Eight of the ten products above predict in writing that a
   baseline will match or beat the learned component. Those predictions are
   not hedges — they are the expected result, and a baseline win is research
   evidence that is never removed, softened or retuned away.
4. **Every number in a validation document comes from a script executed that
   session.** No carried figures, no estimates.
5. **Timings belong in tables declared unstable, never in prose conclusions.**
   A sentence reading "a factor of 40" became "a factor of 38" on re-run in
   Batch 07 because it was derived from wall clock.
6. **Count tests from junit XML, never from pytest stdout.** A suite that
   collects nothing prints success.
7. **Validation layout is flat:** `validation/<script>_output.txt`. Not
   `validation/outputs/`.
8. **`git check-ignore` exits 0 when a negation pattern matches.** Any fixture
   check that reads only the exit status will report a deliberately
   re-included file as ignored. Parse the matched pattern and treat a leading
   `!` as not-ignored.
9. **Anchor every path on `Path(__file__)` and `chdir` to the product root in
   validation support modules.** The gate runs each script with `cwd` set to
   the script's own directory, not the product root. This defect shipped in
   four Batch 07 products and was caught only by the gate, because the build
   agent, its self-report and the coordinator re-measurement all ran from the
   product root — three independent verifications that shared an assumption
   rather than a method.
10. **No absolute container paths in any committed file**, including probe and
    analysis scripts. Two consecutive sessions introduced this defect in the
    act of writing about it.
11. **READMEs follow `templates/REPO_README_STANDARD.md`:** measured test
    counts in badges, the problem in three sentences, who it is **not** for, an
    alternatives table naming only packages verified to exist, a cold-clone
    quickstart with real output, a worked example with real output, a mermaid
    diagram using real module names, embedded screenshots, validation evidence
    **including failed checks**, limitations, reproduction commands. No
    marketing language, no emoji, no exclamation marks.
12. **Never describe a product as flight-safe, certified, mission-ready or
    production-ready.**
