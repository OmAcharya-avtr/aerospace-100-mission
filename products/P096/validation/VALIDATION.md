# Validation evidence — handoverloss

**Validation level 2.** The evidence here is agreement between a closed form, an
independently derived bound and a simulated trace. **Nothing in this repository
is measured on hardware or on a real optical link**, and every figure below is a
property of this repository's own synthetic corpus and arithmetic.

Research-grade; not flight-qualified, not certified, and not approved for
operational aerospace use.

Environment measured this session, recorded in the header of every output file:
Python 3.13.18, numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1, pytest 9.1.1,
hypothesis 6.170.15, matplotlib 3.11.2, ruff 0.17.5 as a module and 0.16.9 on
the path, x86_64 Linux, 2 cores, under contention from four other concurrent
build agents.

Corpus: `data/corpus/handovers.csv`, 1536 events, sha256
`3a0343a6c2c51a91eba0679856700f5495d8814c3bcdc94e6b710ba300c59706`, checked on
every load.

## Resampling unit

**The handover event.** Every interval in this repository, and every interval
any part of this package returns, comes from resampling **whole handover
events**, never accounting slots.

Slots within one handover are not independent: the occupancy at the start of a
slot is the occupancy at the end of the previous one, and once the buffer is
full it tends to stay full, so consecutive slots drop together. A bootstrap over
slots would treat dependent observations as independent and report an interval
narrower than the data support (Davison and Hinkley 1997, section 8.2; Field and
Welsh 2007).

The unit is not merely documented. `BootstrapInterval` and `AttributionReport`
both carry a `resampling_unit` field and **refuse construction with any value
other than `"handover event"`**. `validate_negative_controls.py` measures what
resampling below the unit would cost: splitting each event's volume into ten
pieces and treating them as units gives an interval **0.314950** times as wide,
against the derived expectation `1/sqrt(10) = 0.316228`.

## Reading order

`validate_baselines_against_truth.py` first. It establishes that baseline 1 is
the full-strength form of the assumption being refuted and that baseline 2
reproduces the ground-truth decomposition. Nothing in
`validate_learned_vs_baselines.py` can be interpreted without it.

## Scripts and what each one is evidence of

| Script | Checks | Controls | Evidence of |
|---|---|---|---|
| `validate_baselines_against_truth.py` | 7 | — | Both baselines against the ground truth, BEFORE any benchmark |
| `validate_acquisition_model.py` | 4 | — | Three independent paths to the acquisition-time moments |
| `validate_decomposition_residual.py` | 11 | — | The headline attribution, its residual, and how the answer changes by regime |
| `validate_independent_bound.py` | 11 | 3 | A bracket from mathematics neither model uses |
| `validate_learned_vs_baselines.py` | 4 | — | The benchmark on both held-out splits |
| `validate_negative_controls.py` | 14 | 14 | Every guard asserted to fire, anchored on the true value |
| `validate_approx_audit.py` | 3 | — | No assertion depends on `pytest.approx`'s hidden absolute default |
| `validate_tree_hygiene.py` | 10 | — | What this product owns: adds no artifact, git tracks none |
| `probe_working_directories.py` | 13 | — | Every script agrees with itself from four working directories |
| `worked_example.py` | 3 | — | The README's worked example, executed |

Every script writes its raw output beside itself as
`<script-name>_output.txt`. **Every number in `README.md`, `MODEL_CARD.md`,
`DATASET_CARD.md` and this file comes from one of those files**, and the
document says which.

## 1. Baseline validation — done FIRST, before any benchmark

`validate_baselines_against_truth.py`. The mission's build guide records that an
under-strength analytic baseline inflated an earlier product's apparent learned
win by about 50 per cent. The only defence is to check each baseline against the
ground truth before scoring anything against it.

| Check | Reference | Result | Tolerance |
|---|---|---|---|
| Baseline 1 equals baseline 2's acquisition term | The identity derived in `handoverloss.baselines` step 4 | **400 of 400 bit-identical**, worst relative discrepancy **0.000000e+00** | `8·eps = 8.881784e-16` |
| Baseline 2's dominance agreement with the trace, at realised durations | Baseline 3, the simulated trace | **0.983724** | ≥ 0.90 declared |
| Baseline 2's median total ratio | Baseline 3 | **0.994247**, 5th/95th percentile 0.900798 / 1.072828 | [0.80, 1.20] declared |
| Baseline 2's log10 MAE on the total | Baseline 3 | **0.033592** | reported |
| Baseline 2's log10 MAE per mechanism | Baseline 3 | acquisition **0.067127**, restart **0.067798**, drain **0.076714** | reported |
| Baseline 1 at its conservative setting is strictly larger | — | holds on all 256 events checked | — |
| Baseline 1 log10 MAE, mean setting / 95th-percentile setting | Baseline 3 | **2.104258** / **3.316295** | reported |

**The slot-width and burstiness terms, separated.** Re-running the simulator with
the burst and jitter shapes set to zero makes it a pure discretisation of the
fluid model, so the two sources of baseline-2 discrepancy can be measured
apart. Over 512 events at a 0.25 s slot width:

| Term | Median relative | Worst absolute relative |
|---|---|---|
| Slot width | **+8.514534e-04** | 2.065002e-01 |
| Burstiness | **+0.000000e+00** | 3.697603e+00 |

The burstiness term's median is exactly zero because on most events the buffer
is saturated for the whole excursion and the same volume is lost whatever the
arrival pattern. **164 of 512 events** had a burstiness term of exactly zero.
Where the term is positive the closed form understates, because the fluid model
cannot produce loss from a burst arriving at a non-full buffer.

## 2. The headline — attribution with an interval on each mechanism

`validate_decomposition_residual.py`, 512 corpus events, 4000 resamples, 95 per
cent confidence.

| Mechanism | Loss [Gbit] | 95% interval [Gbit] | Share |
|---|---|---|---|
| acquisition | **20.847705** | [16.005656, 26.229194] | 0.6037 |
| restart | **8.575911** | [7.631991, 9.518379] | 0.2483 |
| drain | **5.110302** | [3.772034, 6.552979] | 0.1480 |

Total **34.533918 Gbit**; dominant **acquisition**; dominance margin **0.355355**
of the total; **ordering DECIDED** (the leading interval is disjoint from the
runner-up's).

**Which mechanism dominates changes with the regime, which is the product's
substantive claim.** Per-event dominance mix, cut on quantities an operator knows
before the handover:

| Regime | n | acquisition | restart | drain |
|---|---|---|---|---|
| low utilisation (`rho < 0.25`) | 576 | 0.6667 | 0.2795 | 0.0538 |
| high utilisation (`rho > 0.75`) | 384 | 0.3672 | 0.3203 | 0.3125 |
| tight buffer (headroom < 10 s) | 559 | 0.4919 | 0.3220 | 0.1860 |
| ample buffer (headroom > 120 s) | 150 | 0.9133 | 0.0800 | 0.0067 |
| slow restart (declared > 30 s) | 672 | 0.3735 | **0.6190** | 0.0074 |
| fast restart (declared < 8 s) | 480 | 0.7000 | 0.0146 | 0.2854 |

**A finding worth stating separately: the MEAN-weighted attribution and the
PER-EVENT dominance mix disagree.** The mean-weighted dominant mechanism is
`acquisition` in every regime, because the mean is carried by the
largest-loss events, which are acquisition-heavy throughout. In the
slow-restart regime the per-event modal dominant is `restart` instead. Both
statistics are printed for every regime, because reporting only one would be
misleading — a tool that reported only the mean would tell an operator with a
slow protocol to buy faster acquisition.

**The ordering is not always decided.** Two of six regimes report
`dominance decided: False` — ample buffer and slow restart — meaning the leading
interval overlaps the runner-up's and those events rank the mechanisms without
deciding them. That is a reason to measure more handovers, not a reason to buy
hardware, and the package never collapses the two claims.

**One regime cell is empty and is reported as such.** `rho > 0.75` combined with
a headroom above 120 s contains **0 events**, because the generator draws the
buffer capacity independently of the source rate and
`headroom_s = (B - q0) / R_s`. It is recorded as `STRUCTURALLY UNMEASURABLE`
with its reason, not dropped.

## 3. The decomposition residual — a reported number, not an assumption

The attribution sums to the total *by construction*, because the phase buckets
partition the window. That alone is not evidence. The residual compares the
bucketed sum against an **independent** accounting of the same total:

- `sum_of_mechanisms` — the three phase buckets, accumulated slot by slot;
- `conservation_total` — `produced − delivered − (q_end − q0)`, which never
  touches the buckets;
- `residual = conservation_total − sum_of_mechanisms`.

**Tolerance.** `4 · slots · eps · (produced + delivered + dropped)`. It scales
with the slot count, because that drives the accumulation error, and with the
volume magnitudes, because that is the conditioning. **It is not a constant**:
a constant would be vacuous for a long window and falsely tight for a short one.

| Check | Result | Tolerance |
|---|---|---|
| Worst absolute residual over the whole corpus | **8.544922e-03 bits** | smallest per-event tolerance 4.494728e-02 bits |
| Median absolute residual | **2.574921e-04 bits** | — |
| Worst \|residual\| / its own tolerance, whole corpus | **3.534252e-03** | ≤ 1 |
| Worst \|residual\| / its own tolerance, per event over 512 events | **3.120787e-03**, 0 of 512 outside | ≤ 1 |
| Worst relative residual on the 1320 events with a loss above 1e6 bits | **1.161965e-12** | derived bound `tolerance/loss` = **4.165621e-09** |

**Two checks in this script were wrong before they were right, and both
failures are recorded rather than removed.**

1. The first version divided the absolute residual by the total over **all**
   events, including the 216 with no loss at all, and reported a worst relative
   residual of **1.632690e-03**. That figure is an artefact of the denominator,
   not a measurement of the accounting, and the check FAILED. The criterion was
   corrected to the absolute residual against its conditioning-scaled tolerance,
   and the relative figure is now reported only on events where it means
   something.
2. The second version then used a literal `1e-12` as the criterion on the
   restricted relative residual and FAILED at **1.161965e-12** — a round number
   that happened to sit just under the measured value. That is exactly the kind
   of bound the mission's build guide forbids. The criterion is now the derived
   `tolerance/loss`.

## 4. The independent bound — mathematics neither model uses

`validate_independent_bound.py`. "Two implementations agree" cannot catch an
error they share; this mission has shipped that defect.

`handoverloss.bounds` is derived from conservation and capacity counting:
`max(0, A − S − (B − q0)) ≤ L ≤ A`, with `A` the offered volume and
`S = ½ R_d T_ramp + R_d · max(0, H − T_o − T_ramp)` the integral of the declared
egress schedule. References: Cruz, R. L., *IEEE Transactions on Information
Theory* 37(1):114–131, 1991; Le Boudec and Thiran, *Network Calculus*, Springer
LNCS 2050, 2001, chapter 1.

| Check | Reference | Result | Tolerance |
|---|---|---|---|
| The bounds module imports no model module | `ast` parse of `src/handoverloss/bounds.py` | imports `__future__`, `dataclasses`, `numpy` only | — |
| Egress integral against hand arithmetic | `½·1e9·60 + 1e9·588 = 6.18e11` | **6.180000e+11 bits** | `8·eps·6.18e11` |
| Bracket contains baseline 2's total, whole corpus | Baseline 2 | **0 of 1536 outside** (0 below, 0 above) | conditioning-scaled slack |
| Bracket contains baseline 3's trace total, measured arrivals | Baseline 3 | **0 of 512 outside** | same |
| Fraction of the swept envelope where the lower bound is non-zero | 2000 random handovers | **1448 / 2000 = 0.724000** | ≥ 0.05 declared |
| Median position of the closed form inside the bracket | — | **0.084174**; 5th/95th percentile −0.000000 / 0.498654 | reported |
| Tightest case, distance to the nearer bound as a fraction of the width | — | **−2.010989e-15** | within the slack |
| Slack scales linearly over six decades | `8·eps·(\|A\|+\|S\|+B+\|q0\|)` | ratio **100.000000000000** at each decade step | 1e-12 relative |

**The bracket is tight, not vacuous.** The lower bound binds on 72.4 per cent of
the swept envelope and the median closed-form total sits 8.4 per cent of the way
up the bracket. A bracket nobody can fail is not evidence, so the tightness is
measured rather than claimed.

## 5. The acquisition-time distribution — three independent paths

`validate_acquisition_model.py`, five declared cases spanning the generator's
envelope.

| Check | Reference | Result | Tolerance |
|---|---|---|---|
| Closed form against Gauss-Legendre quadrature, mean | 512-node quadrature of the truncated density | worst relative **6.695914e-15** | 1e-9 declared |
| Closed form against quadrature, standard deviation | same | worst relative **2.820248e-15** | 1e-9 declared |
| Closed form against 200 000-draw Monte Carlo | Monte Carlo | worst z-score **+1.4049**, 0 of 5 cases outside 4σ | 4 standard errors, derived |
| Memoised exact quantile against a 200 000-draw quantile | Monte Carlo | worst relative **3.276024e-02** | derived sampling bound **1.067708e-01** |
| Declared validity range enforced | — | **4 of 4** out-of-range combinations refused | — |

**A check in this script found a real defect and the implementation was changed
rather than the check loosened.** Baseline 1's conservative setting evaluates the
acquisition distribution at an upper quantile. The first implementation used a
normal approximation `mean + z·sd`. Measured against 200 000 draws it
**overstates the 95th percentile by a factor of 2.032448** at (median 6 s,
log-sigma 0.25, abort 0.01, window 70 s) and **understates by up to 14.2 per
cent** (ratio 0.858176) at a high abort probability. The cause is structural: a
geometric retry tail inflates the variance through a rare long attempt while
leaving the 95th percentile inside the lognormal body, so a symmetric
two-parameter approximation cannot track both ends. It was replaced with a
memoised exact quantile over the 128 distinct parameter tuples a corpus of 1536
events carries.

## 6. The benchmark — an analytic baseline wins both splits

`validate_learned_vs_baselines.py`. Full tables are in `MODEL_CARD.md`.

| Split | baseline 1 | **baseline 2** | learned forest | Winner |
|---|---|---|---|---|
| held-out station pairs, n=384 | 0.536458 | **0.653646** [0.606771, 0.700521] | 0.390625 [0.343750, 0.440104] | **baseline 2**, interval disjoint from the learned component's |
| held-out time, n=576 | 0.506944 | **0.682292** [0.644097, 0.718750] | 0.600694 [0.560764, 0.640625] | **baseline 2**, interval disjoint |

On the station-pair split the learned component does not reach baseline 1's
always-acquisition floor of 0.536458. **This result is published as the finding.
It was not retuned away and the learned component was not removed.**

**What the finding does and does not support.** This corpus is generated by a
mechanism baseline 2 nearly reproduces in closed form (dominance agreement
0.983724 at realised durations), so a closed-form win here is close to the best
case for the closed form and says nothing about measured handover data. The
honest statement is the narrow one: **when the handover sequence is known and
the source is close to constant-rate, the closed form is the better predictor,
and a forest fitted on sixteen station configurations does not beat it.**

**Hyperparameters were selected by three-fold grouped cross-validation over
station pairs, on the training side only.** Six candidates spanned
0.342882 to 0.360243 — a spread of **0.017361** — so the weakness on held-out
station pairs is structural, not a tuning matter.

**Uncertainty output, achieved against nominal 0.900000:**

| Split | acquisition | restart | drain | all |
|---|---|---|---|---|
| held-out station pairs | 0.875000 | 0.835938 | 0.976562 | **0.895833** |
| held-out time | 0.802083 | 0.862847 | 0.873264 | **0.846065** |

Undercoverage is expected and stated: the across-tree spread captures model
uncertainty and not the irreducible noise of the handover. Use the measured
figures.

## 7. Negative controls — 11 of 11 must-fire fired, 3 of 3 must-not-fire silent

`validate_negative_controls.py`. **Every control is anchored on the TRUE value,
not on an output.** Offsetting an output that already sits inside a band does not
guarantee leaving it; an earlier product of this mission shipped two of eleven
controls silently too weak for exactly that reason.

| Control | Must | Fired | Anchor |
|---|---|---|---|
| acquisition bucket inflated 1 % | fire | yes, 3.613e+08× the tolerance | the TRUE bucket value |
| acquisition bucket zeroed | fire | yes, 3.784e+10× | the TRUE bucket value |
| restart bucket inflated 1 % | fire | yes, 1.486e+08× | the TRUE bucket value |
| restart bucket zeroed | fire | yes, 1.515e+10× | the TRUE bucket value |
| drain bucket inflated 1 % | fire | yes, 8.858e+07× | the TRUE bucket value |
| drain bucket zeroed | fire | yes, 8.958e+09× | the TRUE bucket value |
| the unperturbed trace | **NOT** fire | silent | — |
| one-bit error against a zero-magnitude window | fire | yes | the derived zero tolerance |
| 1 % error against a 1e3-bit loss | fire | yes, 10 bits against 8.886225e-08 | — |
| bootstrap resampled below the unit of independence | fire | yes, width ratio **0.314950** | derived `1/sqrt(10) = 0.316228` |
| a report claiming the accounting slot as its unit | fire | yes, refused at construction | — |
| a prediction above the training target range | **NOT** fire | silent | the TRUE training ceiling |
| a prediction call with the wrong feature width | fire | yes | — |
| a degenerate zero-width uncertainty band | **NOT** fire | silent, 1.000000 of cells carry a band | — |

Plus three controls on the independent bound in
`validate_independent_bound.py`: **2 of 2 must-fire fired, 1 of 1 must-not-fire
silent**, both injected values anchored on the true bound rather than on a model
output.

## 8. `pytest.approx`'s hidden absolute default — audited, not assumed

`validate_approx_audit.py`. `pytest.approx(expected, rel=r)` passes if EITHER
bound is met and the absolute default is `1e-12`, so for a quantity of order
1e-5 every relative tolerance tighter than 2e-8 is silently replaced. This
product's quantities span fifteen orders of magnitude, so it is exactly the case
where the default bites.

| Check | Result |
|---|---|
| Static scan of 14 test files: `rel=` sites with an explicit `abs=` | **50** |
| Static scan: `rel=` sites WITHOUT an explicit `abs=` | **0** |
| Whole suite re-run with the default forced to 0 | exit status **0** |
| Assertions that depended on the default, recorded dynamically | **0** |

## 9. Hypothesis counterexamples — both found real defects, both pinned

`.hypothesis/` is gitignored, so a falsifying example found by one run is lost on
a fresh clone. Both counterexamples are pinned in `tests/test_regressions.py`
with literal inputs in the body.

1. **A cancellation in the closed form.** `test_acquisition_term_equals_baseline_one`
   was falsified at `R_s = 1.0e6`, `B = q0 = 15608060.0`,
   `T_acq = 1.1691562848899153`. The first implementation evaluated
   `(q0 + R_s·T_acq) − B`, a difference of two nearly equal large numbers, and
   returned `1169156.284889914` where baseline 1's `R_s·T_acq − (B − q0)`
   returned `1169156.2848899153` — a relative discrepancy of **9.96e-16** that
   baseline 1's own expression does not suffer. **The whole decomposition was
   moved into headroom space**, where every subtraction is between the free space
   and the volume offered against it. The two now agree to the last bit on that
   input and across the whole declared strategy domain. Pinned as
   `test_headroom_form_removes_the_cancellation_hypothesis_found`, which
   recomputes the occupancy-space value so the test shows the discrepancy rather
   than merely asserting its absence.
2. **A one-ulp horizon reconstruction.** `test_a_longer_outage_never_reduces_the_total`
   was falsified at `restart_s = 213.0`, `ramp_s = 299.49081121423325`,
   `extra_s = 1.490811214233247`: rebuilding the window as `horizon_s + extra_s`
   gave `513.9816224284664` while `acquisition_s + extra_s + restart_s + ramp_s`
   gave `513.9816224284665`, so the constructor's bound refused the result. **The
   constructor is right and the caller was wrong.** The property test now rebuilds
   from the phases, and both halves of the asymmetry are pinned as
   `test_rebuilt_horizon_must_come_from_the_phases_not_the_old_horizon`. The
   generator's own capping carries a declared `CAP_MARGIN_S = 1e-6` for the same
   reason.

Property tests pass at the default Hypothesis seed and at seeds 7 and 19,
checked this session.

## 10. Working-directory probe

The release gate runs each validation script with its working directory set to
`validation/`. Three products in one batch of this mission failed the gate on
this after passing every earlier check, because the build agent, its own
self-report and the coordinator all ran the scripts from the product root — three
verifications sharing one assumption.

Every script under `validation/` and `examples/` is executed from **the
filesystem root, the system temporary directory, the product root and
`validation/`**, and the exit status must be identical from all four.
`HANDOVERLOSS_FAST=1` is set for the probed runs, which reduces the resample
count from 4000 to 200, the trace event count from 512 to 96, and makes
`validate_approx_audit.py` run two test files rather than the whole suite —
without the last of those the probe would run the suite sixteen times. The
committed figures in each script's own output file come from the full runs.

Result: see `probe_working_directories_output.txt`. Reported in full in the
build report.

## 11. Tree hygiene — what this product owns

`validate_tree_hygiene.py` asserts two things and deliberately not a third:

- that running the script **adds** no cache directory and no bytecode file —
  the tree is snapshotted at entry and compared at exit;
- that **git tracks** no cache directory, bytecode file or build artifact;
- **not** that no such artifact exists, because release-gate check 2 runs `ruff`
  with its working directory set to the product root and writes `.ruff_cache`
  into the tree one check before check 6 re-executes this script. Inherited
  artifacts are reported, named as inherited, and do not fail.

It also checks `.gitignore` in both directions — that nothing this product must
ship is excluded, and that every cache and build path is — and scans every text
file for an absolute path. One **inherited** finding is reported and not failed:
`tests/fixtures/junit.xml` is excluded by `**/junit.xml` at the mission
repository root, not by this product's `.gitignore`. P096 ships no such file, so
nothing of this product's is lost.

Every runnable script under `validation/` and `examples/` sets
`sys.dont_write_bytecode = True` as its first statement after
`from __future__ import annotations`, because CPython writes a module's bytecode
cache before executing it and the setting therefore cannot be delegated to the
shared `_bootstrap` module.

## 12. Compute, measured

Reported, never asserted as a relationship between two wall-clock readings. That
mistake cost this mission a whole product.

All figures below are single wall-clock readings taken this session on the
two-core build machine, after the final code change, with four other build
agents competing for the same cores. They are **reported**, never asserted.

| Step | Measured | Declared budget |
|---|---|---|
| `data/generate_corpus.py`, 1536 events × 2800 slots | **8.119 s** | 60 s |
| Whole pytest suite, 222 tests | **6.75 s** (6.3 to 10.1 s across runs) | 1800 s per suite |
| `validate_baselines_against_truth.py` | **1.16 s** | 600 s per script |
| `validate_acquisition_model.py` | **2.01 s** | 600 s |
| `validate_decomposition_residual.py` | **2.76 s** | 600 s |
| `validate_independent_bound.py` | **1.17 s** | 600 s |
| `validate_learned_vs_baselines.py` | **10.78 s** | 600 s |
| `validate_negative_controls.py` | **1.58 s** | 600 s |
| `validate_approx_audit.py` (spawns the whole suite) | **9.64 s** | 600 s |
| `validate_tree_hygiene.py` | **0.17 s** | 600 s |
| `worked_example.py` | **1.15 s** | 600 s |
| `probe_working_directories.py` (13 scripts × 4 directories = 52 runs) | **136 s** | 600 s |
| Forest fit + predict + bootstrap, one split | **0.812 s** and **0.758 s** on the two splits (0.71 to 0.84 s across runs) | 60 s |

The one timing assertion in the suite,
`tests/test_regressions.py::test_benchmark_simulation_throughput_direction_only`,
asserts only that the elapsed time is strictly positive and that every event
produced a row. The measured band — 128 events over 2800 slots in 0.6 to 1.4 s
on the two-core build machine under contention from four other agents — is
published here and is deliberately **not** asserted.

## 13. Checks that are NOT evidence, stated

- **No hardware, no measured link, no measured handover.** Validation level 2.
- **One accounting window (700 s) and one slot width (0.25 s).** The slot-width
  term is measured but not varied.
- **No dependency range was tested at both ends** except `ruff`. The build
  environment is a shared pre-pinned interpreter this product is forbidden to run
  `pip` against, so a second version of numpy, scipy, scikit-learn, pytest,
  hypothesis or matplotlib could not be installed. Every range is bounded at the
  single version it was measured at, widened only to the patch floor the code
  plainly supports. `ruff` is clean under both 0.16.9 and 0.17.5.
- **The empty regime cell** (`rho > 0.75` with headroom above 120 s) is reported
  as structurally unmeasurable, not as a pass.
