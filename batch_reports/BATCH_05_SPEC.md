# Batch 05 Specification — Optical modulation, coding and the fading-channel link layer

**Date:** 2026-10-06 · **Authorization:** ADR-017 standing publication authorization, conditional on `scripts/release_gate.py` exiting 0. **No per-batch approval is cited and none exists.** This spec was written by an unattended session; under ADR-016 it records no approval and attributes no words to the owner.
**Composition:** 2 flagship, 3 medium, 5 compact · 7/10 AI-enabled · Levels: **0 x L1**, 7 x L2, 3 x L3
**Theme:** The portfolio has forty products and exactly one that touches modulation (P010 BERBench) and one that touches frame-level coding (P037 FrameSync). Batch 05 builds the coding and modulation layer for the channel this mission actually cares about: a *correlated* fading optical channel, where fades last milliseconds and the design variable is not code rate alone but code rate against interleaver depth against feedback delay.
**Stack:** Python 3.13.16 in the build container, NumPy/SciPy, scikit-learn (PyTorch unavailable), pytest + Hypothesis, Ruff. CLI plus library API plus plotting examples.
**Repositories:** one per product (ADR-018), named for the package, authored solely as `Om Acharya <145807881+OmAcharya-avtr@users.noreply.github.com>`.

## Why this batch, stated against what already exists

The honest case for the batch as a whole is one sentence: **the mature Python
communications libraries are AWGN-centric, and the atmospheric optical channel
is not AWGN.** `komm`, `commpy`, `scikit-dsp-comm`, `pyldpc` and `galois` give
excellent codes, modulators and LLR machinery over memoryless channels. None of
them answers the question an FSO link designer actually has: given a channel
whose fades are correlated over tens of milliseconds, how deep must the
interleaver be, what does that cost in latency and memory, what rate can be
sustained at a stated availability, and how much does the whole design degrade
when the channel state estimate is stale by one round trip.

Every product below is scoped to that gap. Where a product's function is
already well covered by an existing package, its README names that package
first and makes the narrow case second (ADR-018). Build agents must read what
each named alternative actually ships before describing it; existence is not
equivalence, and a wrong claim about a competitor is as much a defect as a
wrong number.

## PyPI name check

All ten names checked 2026-10-06 in the build container with
`pip index versions <name>`, which reported **"No matching distribution found"**
for every one, therefore **all free**: `codedfade`, `acmpilot`, `photoncount`,
`aperturediv`, `arqlonghaul`, `interleavekit`, `slotsync`, `softdecode`,
`linkoutage`, `coderateopt`.

**A negative result from this method was not trusted on its own.** The same
command was run as a control against nine packages that must exist — `galois`,
`reedsolo`, `commpy`, `komm`, `scikit-dsp-comm`, `pyldpc`, `itur`, `sgp4`,
`numpy` — and returned a version list for all nine. The method therefore
distinguishes absence from a transport failure, which is the only reason the ten
absences mean anything. A previous session's habit of quoting an HTTP 404 from
`pypi.org/pypi/<name>/json` is equivalent; this is the same check through the
sanctioned package path.

Packages confirmed to **exist** the same way, for use in alternatives tables:
`galois`, `reedsolo`, `commpy`, `komm`, `scikit-dsp-comm`, `pyldpc`, `itur`,
`sgp4`. Agents may name others only after verifying them the same way and
recording the check.

## Quota gap this batch closes

Before (`scripts/quota_report.py`, 2026-10-06): 40/100 built and published;
12 flagship, 18 medium, 30 compact remaining; AI 28 of 70; L1 8 of 10,
L2 23 of 60, L3 9 of 25, **L4 0 of 5**.

After, if all ten pass the gate: 50/100; 10 flagship, 15 medium, 25 compact
remaining; AI 35 of 70; **L1 8 of 10 (deliberately unchanged)**, L2 30 of 60,
L3 12 of 25, L4 still 0 of 5 pending hardware.

**L1 is deliberately frozen at 8.** Two slots remain against a target of 10 and
six batches remain; spending them now would overshoot the distribution and
force later batches to invent shallow products to hit a number. Batch 05
produces no L1 product. This is the first batch to hold that line.

**L3 arithmetic, checked rather than asserted.** 9 built, target 25, six
batches left including this one. At 3 per batch: 9 + 18 = 27, which clears 25
with two slots of slack. The previous checkpoint's worry that L3 "will not
reach 25" assumed a lower rate; at 3 per batch it closes. Batch 05 therefore
holds at 3 and does not need to raise flagship validation depth.

## The Level 4 position — restated precisely, because it is the easiest thing in this mission to get wrong

L4 is 0 of 5. Batch 04 was the L4 **entry** batch and it did not move the
number, because it cannot be moved in a cloud container. That was the correct
outcome, not a failure.

Batch 05 adds two more products carrying the full hardware-pending groundwork,
so that the eventual Jetson session has four candidates (P031, P033 from Batch
04; P041, P043 from this batch) rather than two:

1. A hardware abstraction layer — the device under test behind one interface,
   with a simulated backend and a device backend that share one contract and
   one test suite.
2. Simulation mode: the full loop against models, deterministic and seeded.
3. Dry-run mode: the real command path and real timing, outputs discarded.
4. Deployment and recovery procedures written as executable checks, not prose:
   what is verified before a run, what is captured during it, how a
   half-finished run is backed out.
5. A benchmark harness recording latency, memory and throughput with the
   measurement method and environment stated in its output file.

**Labelling rule, absolute.** These products are `validation_level: 3` with
`hardware_pending: true` in `products.yaml` and the phrase
**Level 3, hardware-pending** in the README. They are **never** labelled Level
4. The words "Level 4" appear in their documentation only to state what is
still missing. No number in any future L4 claim may come from a simulated
backend, an extrapolation, a vendor datasheet, or a workstation run — only from
the Jetson Orin Nano itself.

**What the owner must do to close L4:** run each product's
`benchmark/run_benchmark.py` on the Jetson Orin Nano and make its raw output
file available to a session. Nothing an automated run can do substitutes.

## The ten products

### P041 — CodedFade (flagship · L3, hardware-pending · AI)
`codedfade` · optical-modulation · AGPL-3.0 (open-core)

Coded FSO link performance over a **correlated** fading channel, with
interleaver depth as the primary design variable.

- Channel: lognormal and gamma-gamma amplitude fading with a specified
  temporal correlation (Gauss-Markov / filtered-Gaussian sample paths), so fade
  *duration* is a property of the model rather than an afterthought.
- Codes: Reed-Solomon and convolutional coding implemented in-package
  (`reedsolo` and `commpy` named as alternatives; note from Batch 04 that
  `commpy` does not install in this container, so it is cited, not used).
- Interleaving: block and convolutional interleavers; the central output is
  post-decoding BER/FER as a function of interleaver depth against the channel
  correlation time, with the latency and memory cost of each depth stated.
- HAL: modem/codec backend behind one interface, simulated + device contract,
  simulation and dry-run modes, deployment/recovery checks, benchmark harness.
- **AI:** a learned predictor of fade-duration exceedance used to *size* the
  interleaver, benchmarked against the analytic level-crossing-rate baseline
  implemented first. Must expose an uncertainty output. If the analytic
  baseline wins, that is the published result.

Why not just use `galois`/`reedsolo`: they give you the code, correctly and
faster than this will. Neither tells you the depth your interleaver needs
against a 20 ms fade, which is the only question that changes the design.

### P042 — AcmPilot (flagship · L3 · AI)
`acmpilot` · optical-modulation · AGPL-3.0 (open-core)

Adaptive coding and modulation on a single optical link **under feedback
delay** — the regime where rate adaptation stops being obvious.

- A declared MODCOD set with measured thresholds; rate adaptation policies:
  fixed margin, threshold with hysteresis, and a clairvoyant upper bound that
  no causal policy can beat.
- Explicit round-trip feedback delay, so the policy acts on a channel state
  that is stale by construction; throughput, outage and
  mis-selection accounting are all reported against delay.
- **AI:** a learned channel predictor over the feedback horizon driving MODCOD
  selection, benchmarked against the three baselines above on the same seeded
  sample paths, with a confidence output that gates aggressive rate choices.
- Level 3 obligations: `docs/REQUIREMENTS.md`, traceability from each
  requirement to the test that exercises it.

**Relationship to P015 LinkSwitch, which must be stated in the README:** P015
chooses *which link* (RF or optical) and is a failover problem. P042 chooses
*which rate* on one optical link and is a prediction-under-delay problem. The
README names P015 and says so, so a reader does not buy the same thing twice.

### P043 — PhotonCount (medium · L3, hardware-pending · AI)
`photoncount` · optical-modulation · Apache-2.0

Photon-counting receiver chain for deep-space and photon-starved optical links.

- Poisson and Webb detection statistics; PPM slot statistics; SPAD
  non-idealities: paralyzable and non-paralyzable dead time, afterpulsing.
- Soft-decision metrics for PPM over the Poisson channel; Poisson channel
  capacity bounds for sanity-checking a link design.
- HAL: photon-counting detector behind one interface, simulated + device
  contract, simulation and dry-run modes, deployment/recovery checks, benchmark
  harness.
- **AI:** a learned dead-time and afterpulsing correction (recovering incident
  rate from observed counts) benchmarked against the closed-form paralyzable
  and non-paralyzable corrections implemented first, with an uncertainty
  interval. The closed forms are strong; if they win, the README says so.

### P044 — ApertureDiv (medium · L2 · AI)
`aperturediv` · optical-link-engineering · Apache-2.0

Multi-aperture receive diversity for FSO — the cheapest real mitigation for
scintillation, and the one with the most folklore around it.

- Channel statistics: gamma-gamma and lognormal pdf/cdf, aperture averaging
  factor, spatial correlation between apertures as a function of separation.
- Combining: maximal-ratio, equal-gain and selection combining; outage
  probability, diversity order, and the gap between the correlated and
  independent-aperture idealisation.
- **AI:** learned combiner weighting under imperfect/stale channel-state
  estimates, against the analytic MRC optimum (which is optimal under perfect
  CSI by construction — so the only honest claim available is about the
  imperfect-CSI regime, and the README must say that plainly).
- **Cross-check (binding):** BER for BPSK over lognormal fading at a stated
  turbulence strength must agree with **P010 BERBench** within the stated
  statistical tolerance. Disagreement is reported, not tuned away.

### P045 — ArqLongHaul (medium · L2 · AI)
`arqlonghaul` · satellite-communications · Apache-2.0

ARQ and hybrid-ARQ throughput over links whose round-trip time dominates
everything.

- Stop-and-wait, go-back-N and selective-repeat; window sizing against RTT and
  bandwidth-delay product; HARQ with incremental redundancy.
- Goodput against raw BER, burst structure and RTT; the regime where
  retransmission is cheaper than redundancy, and the regime where it is not.
- **AI:** a learned retransmission/redundancy policy against fixed-rate HARQ
  and the analytic throughput expressions implemented first.

### P046 — InterleaveKit (compact · L2 · no AI)
`interleavekit` · optical-modulation · Apache-2.0

Interleaver construction and burst-dispersion metrics as a standalone library.

- Block, convolutional, helical and S-random interleavers; minimum-spread and
  dispersion metrics; latency and memory cost per construction.
- Known-answer tests with hand-computed permutations shown in test comments;
  Hypothesis property tests for the de-interleave/interleave identity.
- **Honesty requirement:** `commpy` ships a random interleaver and `komm` ships
  interleaving primitives. The agent must read both, state exactly what they
  cover, and make the narrow case — the burst-dispersion metrics and the cost
  accounting — or report back that there is no case. **If the honest answer is
  that this adds nothing, say so in the return message and do not pad it.**

### P047 — SlotSync (compact · L2 · no AI)
`slotsync` · optical-modulation · Apache-2.0

Slot and symbol timing recovery for OOK and PPM optical receivers.

- Early-late, Gardner and Mueller-Müller timing-error detectors; S-curve
  computation; loop bandwidth and damping against jitter variance; cycle-slip
  rate.
- Validated against the classical loop-noise results implemented from stated
  references, with the reference named per equation.

### P048 — SoftDecode (compact · L2 · AI)
`softdecode` · optical-modulation · Apache-2.0

Soft-decision demapping over *fading* optical channels, including the penalty
nobody budgets for: a stale channel-state estimate.

- Exact LLRs for OOK/PPM over lognormal and gamma-gamma fading; max-log
  approximation and its error; LLR clipping effects.
- The mismatched-CSI penalty as a measured quantity: LLRs computed with a
  channel estimate that is wrong by a stated amount, and what that costs.
- **AI:** a learned LLR corrector for the mismatched-CSI regime, against exact
  LLR (upper bound) and max-log (baseline).
- **Honest positioning:** `komm` and `scikit-dsp-comm` cover AWGN LLRs well and
  the README says to use them for AWGN. The case here is fading with imperfect
  CSI.

### P049 — LinkOutage (compact · L2 · AI)
`linkoutage` · optical-link-engineering · Apache-2.0

Outage and availability statistics for an optical link, as a standalone
library.

- Fade-duration distributions, level-crossing rates, outage probability and
  availability computed from a seeded or supplied amplitude series.
- Markov and semi-Markov channel-state fitting, with a goodness-of-fit report
  rather than a bare fitted model.
- **AI:** a short-horizon outage classifier against the analytic
  level-crossing-rate predictor and a logistic-regression baseline, with
  calibration reported (not just accuracy).
- **Cross-check (binding):** level-crossing rate and mean fade duration on a
  **specified seeded sample path** must agree with **P041 CodedFade** to the
  tolerance stated in the spec of the check.

### P050 — CodeRateOpt (compact · L2 · no AI)
`coderateopt` · optical-modulation · Apache-2.0

Code-rate and margin selection as the constrained optimisation it actually is.

- Maximise expected goodput subject to a stated availability constraint over a
  discrete MODCOD/rate set; `scipy.optimize.milp` for the discrete problem
  (**not** `pulp`, which per the Batch 04 checkpoint ships no usable solver in
  this container).
- Sensitivity of the chosen rate to the fade statistics, so a user can see when
  the optimum is knife-edge and when it is flat.
- Known-answer tests against hand-solved small instances.

## Cross-checks — specified here so they are reproducible, not re-litigated

Batch 04 produced one cross-check that will disagree forever because the
compared statistic was never pinned down (P033 vs P039 p50: one a wall-clock
median on a contended core, the other the median of a cost model). That is a
specification defect and this spec does not repeat it.

**Rule: a cross-check compares a quantity that both implementations compute
from the same seeded sample path, by a definition written here. No cross-check
compares a wall-clock measurement to a model output.**

| # | Products | Quantity compared | Definition | Tolerance |
|---|---|---|---|---|
| X1 | P041 ↔ P049 | mean fade duration and level-crossing rate | computed from the identical seeded filtered-Gaussian amplitude series, same threshold, same sample count, both defined as sample statistics of that series | relative difference ≤ 2 % on both, and the sample count must be stated |
| X2 | P044 ↔ P010 | BER, BPSK over lognormal fading | sample BER at a stated Eb/N0 and scintillation index, same seed count | within 3 binomial standard errors, SE reported |
| X3 | P048 ↔ P010 | hard- vs soft-decision ordering | soft-decision BER must be ≤ hard-decision BER at every stated Eb/N0 on the same channel realisations | strict inequality direction; any violation is a defect, reported |

A disagreement is **reported, not fixed by retuning**. If a disagreement traces
to the specification rather than to either product, say so and amend the
specification in the report.

## Binding rules for every build agent

Read `templates/PRODUCT_BUILD_GUIDE.md` first; it is binding. Then:

1. **Deterministic baseline before any ML.** Phase order is not negotiable. If
   the baseline wins, the baseline winning is the published result (ADR-011).
   Nine such cases are already in the portfolio and none was removed.
2. **Every number from a script run this session.** Save raw output into
   `validation/`. Never invent a citation, a page number or a figure.
3. **No absolute paths in anything you commit.** The release gate fails the
   whole repository on a single `/home/<user>/...` or `/Users/<name>/...` string
   in tracked content, including inside raw validation output. Print and
   serialise repository-relative paths. This cost the 2026-10-05 session a
   second full gate run over thirteen files.
4. **Do not commit model binaries.** The gate rejects `.pt`, `.pth`, `.ckpt`,
   `.onnx` and `__pycache__` outright. Persist models with `joblib`/`.npz` and
   commit the regeneration script, deterministic under a fixed seed.
5. **A suite that collects zero tests is a failure.** Confirm your own count
   from junit XML, not from pytest's stdout line.
6. **Compute budget:** 2 cores, 7.8 GiB, shared with four sibling agents. Any
   training or Monte Carlo run finishes in under 3 minutes. Size accordingly
   and state the budget in the README.
7. **No flight-safety, certification, mission-ready or production-ready
   claims.** Research-grade. L1 wording does not apply — there are no L1
   products in this batch.
8. **Alternatives table is mandatory and must be true.** Read what the package
   actually ships before you characterise it.
9. **Report what failed.** The return message carries files created, test
   counts from an actual run, validation numbers with references, limitations,
   and anything cut or broken. No marketing language.

## Publication plan

Per ADR-017 and ADR-018, conditional on `scripts/release_gate.py` exiting 0:
ten standalone public repositories named `codedfade`, `acmpilot`,
`photoncount`, `aperturediv`, `arqlonghaul`, `interleavekit`, `slotsync`,
`softdecode`, `linkoutage`, `coderateopt`, created with `gh` and pushed with
the Keychain credential (the `gh` token lacks `workflow` scope and the CI
workflow file would be rejected). Monorepo commits are capped at five for the
night. No approval row is written and no product is set to `APPROVED`
(ADR-016).
