# P094 HilTimingDiff — Specification

**Product:** P094 · **Package:** `hiltimingdiff` · **Class:** medium ·
**Category:** testing · **Target validation level:** 4 (hardware-measured)
**Written:** 2026-11-15, **before** any measured timing data existed.

> This document is a **pre-registration**. Baseline 1's declared timing model
> and the pre-registered expectation in §7 were both written and committed
> before the campaign's data was read. The commit that introduces this file
> precedes the commit that introduces the traces, and the ordering is
> verifiable in the repository history. A timing-conformance product whose
> expectation was formed after seeing the measurements would be measuring
> nothing.

## 1. What it does

`hiltimingdiff` executes one pipeline in two modes — **simulation mode** on a
host and **hardware mode** on a Jetson Orin Nano — and reports, stage by
stage, where the simulation's timing assumptions diverge from the board and by
how much. The divergence is the headline output, not a by-product.

The pipeline is a six-stage PAT (pointing, acquisition, tracking) frame chain:

| # | Stage | Work |
|---|---|---|
| 0 | `ingest` | frame assembly: cast and reshape a raw buffer |
| 1 | `background` | exponential running background estimate and subtract |
| 2 | `threshold` | threshold and group bright pixels into candidate regions |
| 3 | `centroid` | intensity-weighted centroid per candidate, 5×5 window |
| 4 | `gate` | validity gating: score, area and border predicates |
| 5 | `track` | six-state constant-acceleration track update, fixed gain |

The same file, `pat_pipeline.py`, runs in both modes, and the same
instrumented driver, `staged_exec.py`, times it in both modes. A per-stage
difference is therefore a property of the platform, not of two
implementations. Both files are committed byte-identical to the copies that
executed on the board, with md5 sums recorded in `validation/VALIDATION.md`.

## 2. What it does that existing tools do not

HIL frameworks run a pipeline in both modes and compare **outputs**. This
compares the **timing itself**, stage by stage, and names which simulated
stage is least trustworthy as a schedulability input.

That question has a consumer. A schedulability analysis — a WCET budget, a
frame-rate feasibility argument, a decision about which stage to move to an
accelerator — is built on per-stage time estimates that come out of a
simulation. If the simulation's per-stage error is *uniform*, the analysis is
merely scaled and its ordering survives. If the error is *stage-dependent*,
the analysis can rank the stages wrongly, and a budget built on it allocates
headroom to the wrong stage. No HIL tool reports which of those two
situations you are in. This one does, and states the interval on the answer.

Honest alternatives, named because they exist and are mature:

| Tool | What it does | When to use it instead |
|---|---|---|
| `pytest-benchmark` | statistically careful timing of a callable | you need one function timed well on one platform |
| `perf` / `ftrace` | kernel-level profile and trace | you need to know *why* a stage is slow, not whether a model predicted it |
| `cProfile` / `yappi` | per-call attribution within a process | you are attributing cost inside one mode |
| NVIDIA `tegrastats` | board-level resource sampling | you need board power and utilisation, not per-stage conformance |
| MATLAB/Simulink HIL | model-in-loop vs hardware-in-loop output comparison | you are validating *behaviour*, not timing conformance |

None of these answers "is my simulation's per-stage timing error uniform or
stage-dependent, and by how much, with an interval". That is the gap.

## 3. Modes and the three time series

For each stage `s` and cell `c` the product holds three series:

1. **`declared[s,c]`** — the simulation's own declared timing model. Derived
   from the operation count in `pat_pipeline.declared_ops(W, K)` multiplied by
   a single scalar `ns_per_op` calibrated **once**, globally, on the
   simulation host. This is what a schedulability analysis reads off the
   source, and it is **the object being refuted**.
2. **`sim_measured[s,c]`** — wall-clock time of the same stage measured in
   simulation mode on the host.
3. **`hw_measured[s,c]`** — wall-clock time of the same stage measured in
   hardware mode on the board. **Ground truth.**

## 4. Baselines

Implemented in this order; the classical work is complete before any fitting.

- **Baseline 1 — the declared timing model.** `declared[s,c]` against
  `hw_measured[s,c]`. Implemented deliberately as the thing being refuted,
  because a correct operation count used as a timing proxy is exactly what a
  schedulability analysis does, and it is wrong for reasons that are not
  arithmetic.
- **Baseline 2 — per-stage analytic scaling, fitted on a training subset of
  stages and evaluated on held-out stages.** A single scalar
  `k = median(hw_measured / sim_measured)` fitted over the training stages
  only, then applied to the held-out stages. This asks the transfer question:
  *can a factor learned on some stages be trusted on a stage you did not
  measure on hardware?* That is the question a team actually faces, because
  they cannot measure every stage on the board.
- **Baseline 3 — ground truth with an interval.** `hw_measured[s,c]` with a
  **bootstrap confidence interval whose resampling unit is the pass, not the
  frame.** Adjacent frames share cache and governor state and are serially
  correlated; resampling frames would understate the interval. The pass is a
  far closer approximation to an independent draw and is the declared unit.

**No learned component.** `ai_enabled: false` in the registry. A model fitted
to the same measurements it would be scored against would add a second thing
to trust and no honest information. The product's value is the paired
per-stage comparison and the ordering of divergences.

## 5. Headline outputs

1. **Per-stage divergence ratio** `R[s] = median(hw_measured[s]) /
   median(declared[s])`, with a bootstrap interval over passes.
2. **Divergence spread** `max(R)/min(R)` across the six stages. **This is the
   number that matters.** At 1.0 the declared model is uniformly wrong and a
   schedulability ordering built on it survives. Far from 1.0 the model is
   stage-dependently wrong and the ordering does not survive.
3. **Rank disagreement** between the stage ordering by `declared` and by
   `hw_measured`, as Kendall's tau-b and as the explicit permutation.
4. **The least trustworthy stage**: `argmax(R)` and `argmin(R)`, named.
5. **Baseline 2's held-out transfer error**, which bounds how much a team can
   rely on extrapolating a measured scaling factor to an unmeasured stage.

## 6. The unpinned clock, declared as scope rather than hidden

Measured on the board on 2026-11-15, before the campaign: governor
`schedutil`; per-core `scaling_cur_freq` spanning 729 600–1 190 400 kHz across
the six cores at idle; `nvpmodel` reporting `MAXN_SUPER`; **`sudo -n true`
fails, so passwordless sudo is unavailable and neither the governor nor the
power mode can be changed by this session.**

Every figure this product reports is therefore **a figure at an unpinned,
schedutil-governed clock in MAXN_SUPER**. That is stated with the headline and
not relegated to a limitations list, and it is **measured rather than
asserted**: per-core frequency and every thermal zone are sampled before each
cell, at six points within each cell and after each cell, and the frequency
trace is a committed output. The product reports how much of the measured
per-stage variance co-varies with core frequency.

This is a real restriction on the result and it is also the honest operating
condition of most HIL rigs, which do not run with pinned clocks either.

## 7. Pre-registered expectation — written before the data was read

**The declared operation-count model will be stage-dependently wrong, not
uniformly wrong, and the divergence spread `max(R)/min(R)` will exceed 3.**

Mechanism, stated in advance so the claim is falsifiable: `declared_ops`
counts arithmetic on elements. Three of the six stages — `ingest`,
`background`, `threshold` — are vectorised numpy operating on whole arrays, so
their per-operation cost is a memory-bandwidth cost and is small. Two —
`centroid` and `gate` — contain Python-level loops over candidates, so their
per-operation cost carries interpreter overhead one to two orders of magnitude
larger. `track` is a fixed-size matrix update dominated by call overhead
rather than by its 180 declared operations.

**This expectation is refuted if the spread is below 3**, and the result is
reported either way. A spread near 1 would mean the declared model is a sound
schedulability input, which would make this product's premise wrong and would
be published as such.

**Secondary pre-registration:** baseline 2's single global scaling factor will
transfer poorly to held-out stages, with a mean relative error above 0.5,
because a single factor cannot absorb a vectorised/interpreted split.

## 8. Campaign design

Nine grid cells: `W ∈ {48, 64, 96}` × density `∈ {sparse: 2 targets, dense: 12
targets}`, plus three repeats of the reference cell `w64_sparse`.

- **Order randomised** under recorded seed `20261115`, with the permutation
  committed. P093's cells ran in ascending frame dimension, which confounded
  dimension with elapsed time and with a falling foreign load.
- **Drift calibration before every cell and once after the last** — ten
  blocks, each 2000 repetitions of a fixed 160×160 float64 matmul that shares
  no code with the pipeline. The 2026-11-14 carry-over requires a minimum of
  start and end; ten blocks let board drift be *fitted* over the campaign
  rather than only bracketed.
- 1500-frame stack per cell, 60 passes, **90 000 timed frames per cell**.
- Per-pass summaries computed over **every** frame; raw per-frame arrays
  committed subsampled every third frame, which is stated wherever the raw
  arrays are used.
- Instrument measured before subject: empty-bracket cost (4000 reps) and clock
  resolution recorded per cell.

## 9. Gate branch reachability

P093 shipped a gate two of whose five reject reasons never fired in 268 425
inferences, and that is recorded as a weak result. Here all four reject
reasons plus accept are counted per cell and **a test asserts every branch
fires at least once across the campaign**. If a branch does not fire, that is
reported as a measured property of the gate and the acceptance figures are
qualified accordingly — not synthesised.

## 10. Deliverables

`src/hiltimingdiff/` (pipeline, instrumented executor, three baselines,
divergence analysis, bootstrap, CLI), `tests/`, `examples/` with real output,
`validation/VALIDATION.md` with every number traced to a re-runnable script,
`data/traces/` with the campaign outputs and `meta.json`, `screenshots/`
generated from measured data, `README.md` per
`templates/REPO_README_STANDARD.md`.

## 11. Honesty constraints

Research-grade; not flight-qualified, not certified, not approved for
operational aerospace use. No figure in `VALIDATION.md` comes from anywhere
but a script executed in the session that reports it. The pre-registered
expectation in §7 is reported as it lands. Weak, null and refuting results are
published, not retuned.

Credits: This is under reserved rights obtained by OPTIMA Organisation.
