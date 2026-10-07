# Batch 06 Specification — Autonomy assurance: runtime guarding, falsification, calibration and traceable evidence

**Date:** 2026-10-07 · **Authorization:** ADR-017 standing publication authorization, conditional on `scripts/release_gate.py` exiting 0. **No per-batch approval is cited and none exists.** This spec was written by an unattended session; under ADR-016 it records no approval and attributes no words to the owner.
**Composition:** 2 flagship, 3 medium, 5 compact · 7/10 AI-enabled · Levels: **1 x L1**, 6 x L2, 3 x L3 · **0 new Level 4 groundwork, deliberately**
**Theme:** The portfolio now has fifty products that compute aerospace quantities and several that learn them. It has almost nothing that answers the question a reviewer asks about a learned component: *how do you know when it is wrong, and what happens then.* Batch 06 builds that layer — the runtime guard, the adversarial test search, the drift and calibration audit, and the machine-checkable link from a claim to the artifact that supports it.
**Stack:** Python 3.13.16 in the build container, NumPy/SciPy, scikit-learn (PyTorch unavailable), pytest + Hypothesis, Ruff. Library API plus `python -m <package>` CLI plus plotting examples.
**Repositories:** one per product (ADR-018), named for the package, authored solely as `Om Acharya <145807881+OmAcharya-avtr@users.noreply.github.com>`.

## Why this batch, stated against what already exists

The honest case for the batch in one sentence: **the mature Python libraries in
this space are each excellent at one primitive and silent about the aerospace
accounting that turns the primitive into evidence.**

`rtamt` monitors STL. `MAPIE` and `crepes` produce conformal intervals. `netcal`
computes calibration error. `river` streams change detectors. `SALib` does
global sensitivity. `pytope` manipulates polytopes. Every one of them is a better
choice than anything in this batch for the primitive it owns, and every README
below must say so, by name, in its alternatives table, before making its own
case.

What none of them answers:

- Given a learned controller and a conservative certified one, **how often does
  the guard fire, what does the switching cost, and is the safety argument
  actually preserved** under the disturbance bound you declared.
- Given a requirement and a simulator, **how many samples does it take to find a
  violation**, and does a surrogate-guided search beat uniform random — measured,
  not assumed.
- Given a digital twin and a stream of residuals, **how many samples after the
  physical asset changes do you notice**, against how many false alarms per
  thousand hours.
- Given a probabilistic output, **which part of a bad Brier score is
  miscalibration and which part is irreducible**, and does recalibration help or
  only move the error.
- Given a pile of validation scripts and a requirements document, **which claims
  have no evidence behind them**, computed rather than asserted.

Each product is scoped to one of those gaps. **Every product must state what it
does that the named alternative does not.** If an agent concludes while building
that the honest answer is "nothing", it must say so in its final report rather
than pad the README; that is a finding, not a failure.

## PyPI name check — controlled method, ADR-018 discipline

All ten names checked 2026-10-07 in the build container with
`pip index versions <name>`. **"No matching distribution found" for all ten,
therefore all free:** `simplexguard`, `falsifyloop`, `twininvalidate`,
`rareverify`, `assuregraph`, `calibaudit`, `conformalband`, `telemdrift`,
`invariantset`, `traceaudit`.

**The negative result was not trusted on its own.** The same command was run as a
positive control against nine packages that must exist — `numpy`, `requests`,
`mapie`, `crepes`, `rtamt`, `SALib`, `river`, `netcal`, `pytope` — and returned a
version list for all nine. The absences therefore distinguish a free name from a
broken transport.

**Two candidate names were rejected by this check and replaced**, which is the
check earning its place rather than decorating the spec:

| Rejected | Found on PyPI | Replacement |
|---|---|---|
| `driftwatch` | 0.4.0, 0.3.0, 0.2.0 | `telemdrift` (free) |
| `tracelint` | 0.10.0 and eleven earlier releases | `traceaudit` (free) |

Versions observed for the alternatives, for use in README alternatives tables —
**agents must still read what the package actually ships before describing it;
existence is not equivalence, and a wrong claim about a competitor is as much a
defect as a wrong number** (the Batch 05 `komm` lesson, which cost four
independent re-checks):
`mapie` 1.5.0 · `crepes` 0.9.1 · `rtamt` 0.3.5 · `SALib` 1.6.0 · `river` 0.26.1 ·
`netcal` 1.4.0 · `pytope` 0.0.4.

**None of these may be added as a runtime dependency.** The declared runtime
dependency set for this batch is `numpy`, `scipy`, `scikit-learn`, `joblib` only,
unchanged from Batch 05, so that `pip-audit` remains diffable against the
declared union. A package named in an alternatives table is a citation, not an
import.

## Quota gap this batch closes

Before (`scripts/quota_report.py`, run 2026-10-07): **50/100 built and
published.** Remaining: 10 flagship, 15 medium, 25 compact; AI 35 of 70;
L1 8 of 10, L2 30 of 60, L3 12 of 25, **L4 0 of 5**.

After, if all ten pass the gate: 60/100; 8 flagship, 12 medium, 20 compact
remaining; AI 42 of 70; L1 9 of 10, L2 36 of 60, L3 15 of 25, **L4 still 0 of 5
pending hardware.**

**Arithmetic, checked rather than asserted.** Five batches remain including this
one, and 50 products remain, so 2 flagship / 3 medium / 5 compact per batch is
not a convention here — it is the only split that lands exactly on 20/30/50.
AI at 7 per batch gives 35 + 35 = 70, exactly the floor. L2 at 6 per batch gives
30 + 30 = 60, exactly the target.

**L1 spends one of its last two slots, and this is a change from Batch 05.**
Batch 05 froze L1 at 8 with six batches left, correctly: 2 slots over 6 batches
cannot be spread. With five batches left the position is different — two slots
over five batches still cannot be spread evenly, so they must be spent as
single-slot batches whenever a genuinely educational-depth product appears.
P060 TraceAudit is that product: a static linter over text and test node ids, with
no physics to validate against a reference, is honestly L1 and would be
dishonestly labelled L2. One slot goes now, one remains for the final four
batches.

**L3 arithmetic.** 12 built, target 25, five batches left including this one. At
3 per batch: 12 + 15 = 27, clearing 25 with two slots of slack — the same margin
Batch 05 computed, one batch later, so the rate holds at 3 and no flagship needs
its validation depth inflated to make a number.

## The Level 4 position — this batch adds no new candidate, on purpose

L4 is 0 of 5 and **Batch 06 does not try to move it.** Four products already wait
on the same measurement: P031 HilForge, P033 EdgeInfer, P041 CodedFade and
P043 PhotonCount, each labelled `Level 3, hardware-pending`, each with a
`benchmark/run_benchmark.py` that needs a Jetson Orin Nano the container does not
have. **A fifth candidate waiting on the identical measurement would add nothing
and would read as progress while being none.** The constraint is one hardware
session by the owner, not more groundwork.

No product in this batch may be labelled Level 4, and no product may add a new
hardware-pending benchmark harness. Where a product's figures depend on compute
(P052 and P054 both report sample-efficiency curves), the README must state the
container's measured core count and the wall-clock budget, and must not present a
timing as a hardware characteristic.

---

# The ten products

Agents: read `templates/PRODUCT_BUILD_GUIDE.md` first, then
`templates/REPO_README_STANDARD.md`. Every number in a README or VALIDATION.md
comes from a script you executed this session and committed. Implement the
classical or analytic baseline **before** any learned model, and **publish the
result even when the baseline wins** — sixteen such honest negatives are already
in this portfolio and four of Batch 05's ten found one. They are the most
credible thing the mission has produced. Never describe anything as flight-safe,
certified, mission-ready or production-ready.

## P051 SimplexGuard — flagship · L3 · AI · autonomous aerospace systems / aerospace assurance

Package `simplexguard`. **Runtime-assurance (Simplex) architecture benchmark for
a learned controller wrapped by a conservative certified one.**

Scope:
- Discrete-time linear plant with a bounded additive disturbance, declared
  explicitly, plus a declared state constraint set. A conservative baseline
  controller with a computed robust invariant set, and a performance controller
  which may be learned.
- The **switching condition**: hand control to the baseline when one step of the
  performance controller could leave the invariant set under the worst admissible
  disturbance. Implement it exactly (support-function evaluation on the declared
  polytope), not heuristically.
- The **assurance accounting** that is the actual product: switch rate, dwell
  time distribution, fraction of the episode under baseline control, the
  *conservatism cost* (performance lost against an unguarded run), and the
  constraint-violation count — which must be zero under the declared bound, and
  must be reported as nonzero if it is not.
- A deliberate **bound-violation experiment**: run with a disturbance larger than
  declared and report where the guarantee fails. A guard whose assumption is
  false is the realistic failure mode and must be measured, not asserted safe.
- AI component: a learned switch predictor (classifier on the state, trained on
  simulated episodes) that tries to anticipate the exact condition earlier, with
  a calibrated confidence output. **Baseline is the exact analytic condition.**
  The honest expected outcome is that the learned predictor cannot beat an exact
  computation on its own criterion; measure precision/recall and lead time, and
  publish the loss if it loses. A learned predictor that is merely *equal* at
  lower cost is also a result — report the cost.
- Alternatives table must name: `pytope`/`polytope` for polytope algebra, `rtamt`
  for STL monitoring, and any control-barrier-function package the agent verifies
  exists. State plainly that this is not a verification tool and proves nothing
  about the plant it was not given.
- L3 obligations: `docs/REQUIREMENTS.md` with numbered requirements, each mapped
  to the test that exercises it.

## P052 FalsifyLoop — flagship · L3 · AI · testing / aerospace assurance

Package `falsifyloop`. **Requirement falsification for autonomous control loops:
find the input or parameter setting that violates a stated requirement, and
measure how many simulations it took.**

Scope:
- A robustness semantics over traces for a small, fully documented requirement
  language: bounded always/eventually, bounds on signals and their differences.
  Robustness must be negative exactly when the requirement is violated —
  property-test that equivalence with Hypothesis.
- Search strategies, all over the same seeded instances and the same budget:
  uniform random, Latin hypercube, simulated annealing, cross-entropy, and a
  Gaussian-process / random-forest surrogate-guided search (the AI component).
- The deliverable is the **sample-efficiency curve**: probability of having found
  a violation against simulation count, with bootstrap confidence bands, over a
  suite of at least six seeded benchmark instances of stated difficulty.
- **Uniform random is the baseline and it is a strong one.** Report per-instance
  results, not only the aggregate, and report the instances where random wins.
  A method that wins on average while losing on the hardest instance must say so.
- Falsification is one-sided: finding no violation is not evidence of
  correctness. The README must say that in those words, and the CLI must not
  print anything that reads as a pass.
- Alternatives table must name `rtamt` (and S-TaLiRo / Psy-TaLiRo if the agent
  verifies them) and concede their specification languages are far more complete.
- L3 obligations as P051.

## P053 TwinInvalidate — medium · L3 · AI · digital twins

Package `twininvalidate`. **Model-invalidation monitor for a digital twin: decide
when the twin no longer describes the asset, and report how long that took.**

Scope:
- A declared linear-Gaussian twin with a residual generator, and injected
  asset-side changes of three kinds: a parameter step, a slow ramp, and a noise
  variance change. Generated by committed seeded scripts.
- Analytic baselines first: CUSUM, EWMA, and a windowed generalised-likelihood-
  ratio test, each with its threshold set from a declared false-alarm target
  rather than tuned to the result.
- The deliverable is the **detection-delay against false-alarm-rate curve**
  (ARL0/ARL1), per change type, with the threshold-setting method stated. Report
  the change type where every method does badly.
- AI component: a learned drift classifier on windowed residual features with a
  confidence output. **The analytic GLR is the baseline and is expected to be
  hard to beat on a Gaussian residual by construction** — if so, publish that and
  explain the structural reason rather than retuning.
- Must distinguish *twin invalidation* from *asset fault*: the same residual can
  mean either, the product cannot tell them apart, and the README must say so.
- Alternatives table: `river` for streaming detectors, `scipy` for the tests,
  any change-point package the agent verifies.
- L3 obligations as P051.

## P054 RareVerify — medium · L2 · AI · testing

Package `rareverify`. **Planning and executing a Monte-Carlo verification
campaign for a rare requirement violation, with defensible interval bounds.**

Scope:
- Sample-size planner: given a target violation probability and a confidence
  requirement, how many runs. Clopper-Pearson and Wilson intervals, with the
  zero-failure case handled explicitly, because that is the case a campaign
  actually hits.
- Variance reduction: importance sampling with a declared tilting family, and
  subset simulation. Report the **measured** variance reduction factor against
  plain Monte Carlo on instances whose answer is known analytically, and report
  the instances where importance sampling is *worse* — a badly chosen tilt is
  worse than none and the product must be able to show that.
- Known-answer tests against analytic tail probabilities (Gaussian, lognormal
  limit states) with the tolerance stated as a multiple of the counting-noise
  floor.
- AI component: a learned surrogate of the limit-state function used to steer
  sampling, against the analytic/IS baseline. **Expected to lose on a smooth
  analytic limit state and possibly to win on a rough one** — report both.
- No claim about any real vehicle. The README states that the probability
  estimated is the probability *of the model*, which is the number people
  misquote.
- Alternatives table: `SALib` for sensitivity, `scipy.stats` for the intervals,
  `UQpy`/`OpenTURNS` if the agent verifies them, and concede their breadth.

## P055 AssureGraph — medium · L2 · no AI · aerospace assurance

Package `assuregraph`. **An assurance case as a machine-checkable evidence graph.**

Scope:
- A declarative case format (YAML) of claims, argument steps, assumptions,
  context and evidence artifacts, following Goal Structuring Notation naming so a
  reader who knows GSN recognises it. Parse, validate, and reject malformed cases
  with actionable errors.
- Computed checks, which are the product: unsupported claims, evidence
  referenced but absent from disk, **evidence staleness by content hash** (the
  artifact changed after the claim cited it), undischarged assumptions, cycles,
  and orphan nodes. Every check reported with the node ids, exit code nonzero
  when the case is incomplete.
- Mermaid rendering of the graph so a case is reviewable in a GitHub README,
  plus a coverage summary table.
- Honest scope statement, mandatory and prominent: **a well-formed assurance case
  is not a safe system**, this tool checks structure and freshness only, it
  evaluates no argument's soundness, and it is not a DO-178C or ARP4754A
  compliance tool. Any phrasing that could be read as certification support is a
  defect.
- No AI, by design. A learned judgement about whether an argument is sound is
  exactly the thing that must not be automated here, and the README says so.
- Alternatives table: commercial GSN tools (ASCE, AdvoCATE) by name, `graphviz`,
  and any open GSN package the agent verifies. Concede the commercial tools'
  notation coverage.

## P056 CalibAudit — compact · L2 · AI · aerospace assurance

Package `calibaudit`. **Audit of a probabilistic forecast: decompose the score,
quantify the estimator's own bias, then recalibrate and measure whether it
helped.**

Scope:
- Murphy decomposition of the Brier score into reliability, resolution and
  uncertainty, with the identity `BS = REL - RES + UNC` property-tested to
  machine precision.
- Reliability diagrams with bootstrap bands; Brier and log skill scores against a
  stated reference forecast.
- **Expected calibration error with its binning bias quantified, not just
  reported.** ECE is biased by bin count and sample size; measure that bias on
  synthetic forecasts of known calibration and publish the curve. This is the
  product's main contribution over `netcal`, and the README must make the
  comparison explicitly rather than implying `netcal` lacks the metric.
- Recalibration: Platt scaling and isotonic regression (the learned components),
  each with a held-out split, against the **raw forecast as baseline**. Report
  the case where recalibration makes the score worse, which happens on small
  samples and is the practically important case.
- Alternatives table: `netcal` 1.4.0, `sklearn.calibration`, `properscoring` if
  verified. Say when to use `netcal` instead.

## P057 ConformalBand — compact · L2 · AI · autonomous aerospace systems

Package `conformalband`. **Distribution-free prediction intervals for an
aerospace regression, with coverage measured under a declared covariate shift.**

Scope:
- Split conformal, Mondrian (class-conditional) conformal, and weighted conformal
  with declared likelihood-ratio weights. Exchangeability stated as the
  assumption it is.
- The deliverable is a **coverage audit**: empirical coverage and interval width
  against nominal level, on in-distribution data and under three declared shifts
  of increasing severity, with the finite-sample coverage bound plotted alongside.
- **Baseline is the parametric Gaussian interval from the model's own residual
  variance.** Expected outcome: parametric is tighter and correct in
  distribution, conformal is wider and holds up under shift until the weights are
  wrong. Measure where weighted conformal breaks and publish the breaking point.
- Known-answer test: exact finite-sample coverage of split conformal on exchangeable
  data, against the `ceil((n+1)(1-alpha))/(n+1)` bound, hand-computed in a test
  comment.
- Alternatives table: `MAPIE` 1.5.0 and `crepes` 0.9.1 **named first and
  recommended for general use.** This product's narrow case is the measured
  shift audit, and the README must not imply more.

## P058 TelemDrift — compact · L2 · AI · testing / satellite communications

Package `telemdrift`. **Streaming change detection on a univariate telemetry
channel, scored on detection delay against false-alarm rate.**

Scope:
- Detectors: Page-Hinkley, CUSUM, EWMA, a windowed Kolmogorov-Smirnov test, and
  an ADWIN-style adaptive window implemented from its published rule, each with a
  stated threshold-setting procedure.
- Scoring: ARL0 (mean time to false alarm on a stationary stream) and ARL1
  (detection delay after a change), both measured over seeded replicates with
  confidence intervals, plus the **delay-versus-false-alarm trade-off curve that
  makes the detectors comparable at equal operating point** — comparing at
  default thresholds is the usual error and the README must name it as such.
- Change types: mean step, variance step, drift ramp, and a transient spike that
  should *not* trigger a change alarm. Report every detector's behaviour on the
  transient, including the ones that fire.
- AI component: a learned detector on windowed features, against the analytic
  detectors at equal ARL0. Publish the loss if it loses.
- Alternatives table: `river` 0.26.1 named first for production streaming,
  `ruptures` if verified for offline change-point detection. Say plainly that
  this is a benchmark harness, not a streaming framework.

## P059 InvariantSet — compact · L2 · no AI · GNC

Package `invariantset`. **Robust invariant and reachable sets for discrete-time
linear systems with bounded disturbances.**

Scope:
- Polytope representation with support-function evaluation, Minkowski sum,
  Pontryagin difference, and intersection; the recursive maximal robust invariant
  set computation with a declared convergence criterion and iteration cap, and an
  explicit report when it does not converge.
- Known-answer tests against hand-computed sets for 1-D and 2-D systems, shown in
  test comments, plus Hypothesis property tests for the set algebra identities
  (Pontryagin difference is not the inverse of Minkowski sum — test the actual
  identity, not the one people assume).
- **Numerical honesty is the hard part and is the product's real content:**
  redundant-constraint removal tolerance, how vertex counts grow, where the
  iteration stalls, and a documented case where a tolerance change alters the
  answer. Report the growth measured, not estimated.
- Self-contained; P051 may describe it as related but must not import it
  (no cross-product imports).
- Alternatives table: `pytope` 0.0.4, `polytope`, and MPT3/MPC toolboxes by name,
  conceding their maturity. Say when to use them instead.

## P060 TraceAudit — compact · L1 · no AI · aerospace assurance

Package `traceaudit`. **Requirements-to-test traceability, computed from the
repository rather than maintained by hand.**

Scope:
- Parse numbered requirements from markdown (`REQ-NNN` style, format documented
  and configurable), parse test identifiers and markers from a pytest collection
  report or a junit XML file, and compute the bidirectional mapping.
- Report, with nonzero exit on any finding: requirements with no test, tests
  claiming a requirement that does not exist, duplicate requirement ids,
  requirements traced only by a skipped or xfailed test (the one that looks
  covered and is not), and a coverage percentage with its denominator stated.
- Works on this mission's own L3 products as the worked example — run it against
  a committed sample `docs/REQUIREMENTS.md` and junit XML fixture bundled in the
  repository, and show the real output.
- **Level 1, labelled honestly.** This is a text-and-id linter with no physical
  reference to validate against; calling it L2 would be dishonest labelling, and
  the README states the validation level and what it means.
- The README must state that traceability is a necessary bookkeeping condition
  and not evidence of adequacy: a requirement traced to a test that asserts
  nothing is still traced. Where possible, flag empty-assertion tests and admit
  the check is heuristic.
- Alternatives table: `pytest`'s own markers, `sphinx-needs`, `doorstop` if
  verified, and commercial requirements tools. Concede them.

---

## Binding reminders for every agent

1. **Classical or analytic baseline first.** The learned component is benchmarked
   against it on the same held-out data. If the baseline wins, that is the
   published result, with the structural reason. Do not retune, soften or remove
   it. Four of Batch 05's ten products reported a baseline win.
2. **Every number comes from a script you ran this session**, committed, with its
   raw output saved under `validation/`. Re-run every producing script as your
   final step so `git status` over your product directory is clean — the gate
   re-executes them and a drifted artifact fails the batch.
3. **Test counts are measured from junit XML by the coordinator, never from your
   report.** Report your own count anyway; the comparison is the point.
4. **Name real alternatives and read them before describing them.** `pip download`
   or `pip index versions` plus unpacking the wheel. Existence is not equivalence.
5. **No flight-safety, certification, mission-ready or production-ready claim.**
   Research-grade. For P055 and P060 especially, no phrasing that reads as
   DO-178C / ARP4754A compliance support.
6. **Compute budget:** 2 cores, shared five ways. Any training or Monte-Carlo run
   must finish in under 3 minutes standalone; expect a 5-6x slowdown under
   contention and size accordingly. Matplotlib Agg only.
7. **Declared runtime dependencies are `numpy`, `scipy`, `scikit-learn`, `joblib`
   only.** Nothing else, so the batch `pip-audit` stays diffable.
8. **Known tool defects to avoid** (found and recorded by earlier batches):
   `scipy.optimize.milp` needs `options={'mip_rel_gap': 0.0}`;
   `numpy.polynomial.hermite` weights overflow to NaN above ~350 nodes;
   `scipy.stats.kstest(x,'norm',args=(loc,scale))` raises TypeError on the
   installed SciPy — standardise and call `kstest(z,'norm')`.
9. **Credits line, verbatim, once, in Credits only:** "This is under reserved
   rights obtained by OPTIMA Organisation."
