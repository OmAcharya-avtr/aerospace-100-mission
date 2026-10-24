# Validation evidence — cmdguard 0.1.0

**Validation level 2.** Research grade. Not flight-qualified, not certified,
not approved for operational aerospace use.

Every number below was produced by a script in this directory, executed in the
session that built this repository, on the environment recorded at the head of
each `*_output.txt`. Each row names the script and the transcript it came from.
Nothing here is an estimate and nothing is carried from an earlier session.

The layout is flat: each script writes `validation/<script name>_output.txt`
beside itself.

| Script | What it establishes | Transcript |
|---|---|---|
| `validate_forward_known.py` | Forward execution against hand-constructed cases with hand-derived violations and shortfalls | `validate_forward_known_output.txt` |
| `validate_negative_control.py` | No violation is reported on sequences that have none | `validate_negative_control_output.txt` |
| `validate_timing.py` | Timing constraint arithmetic, unit-checked, and the tolerance band probed from both sides | `validate_timing_output.txt` |
| `validate_irreversible_gate.py` | The gate refuses, and refuses exactly the right orderings | `validate_irreversible_gate_output.txt` |
| `validate_lint_missrate.py` | What a per-command static lint misses, by family and by kind | `validate_lint_missrate_output.txt` |
| `validate_ranker.py` | The learned ranker against the two free orderings, with an ablation and a calibration measurement | `validate_ranker_output.txt` |
| `validate_determinism.py` | Every component reproducible from its seed; the cost table | `validate_determinism_output.txt` |
| `validate_cli.py` | Every CLI subcommand and its exit code, as a subprocess | `validate_cli_output.txt` |
| `regenerate_cases.py` | The committed case files round-trip through the loader unchanged | `regenerate_cases_output.txt` |
| `worked_example.py` | The README worked example, with the output the README quotes | `worked_example_output.txt` |
| `probe_pypi_names.py` | Every package the README names exists on PyPI; the probe validated in both directions | `probe_pypi_names_output.txt` |

Support: `_bootstrap.py`. Every script calls `bootstrap()` first, which puts
`src` on `sys.path` and changes the working directory to the repository root,
derived from `__file__`. Every path printed goes through `relative()`.

## 1. Forward execution against hand-derived answers

`validate_forward_known.py`. The three committed cases, with the arithmetic
written out beside each assertion.

| Check | Reference | Measured | Tolerance |
|---|---|---|---|
| `nominal_science_pass` violations | hand: the sequence satisfies every declaration | 0 | exact count |
| `nominal_science_pass` peak held power | hand: heater 24.0 + enable 18.0 + scan 46.0 = 88.0 W at t=80 s, against a declared 120.0 W | 88.000000 W | 1e-9 absolute |
| `nominal_science_pass` peak data rate | hand: the scan's declared 22.0 Mbps, against 40.0 | 22.000000 Mbps | 1e-9 absolute |
| `late_payload_enable` violations | hand: exactly one, at step 6 | 1, `precondition_false` | exact count |
| `late_payload_enable` trace | hand: step 5, `PAYLOAD_SAFE`, t=80.0 s | step 5, `PAYLOAD_SAFE`, 80.0 s | 1e-9 on t |
| `resource_and_exclusion` violations | hand: six, derived step by step in the case file's notes | 6 | exact count |
| exclusion shortfall at step 2 | hand: `HEATER_A` holds `thermal_pair` over [10, 210); `HEATER_B` at t=60 is 150 s early | 150.000000 s | 1e-9 |
| resource shortfall at step 3 | hand: 40 + 40 + 55 = 135 W against 100 W | 35.000000 W | 1e-9 |
| gap shortfall at step 4 | hand: `XMIT` declares 20 s and ran at t=70; `SSR_WRITE` at t=80 | 10.000000 s | 1e-9 |
| resource shortfall at step 5 | hand: 30 + 25 = 55 Mbps against 40 Mbps | 15.000000 Mbps | 1e-9 |
| interval shortfall at step 6 | hand: `XMIT` declares 300 s and last ran at t=70; repeat at t=200 | 170.000000 s | 1e-9 |
| every declared violation kind exercised | 10 kinds declared | 10 exercised | exact set |

## 2. Negative control

`validate_negative_control.py`. Two populations.

| Population | Cases | Command occurrences | False alarms | Rate |
|---|---|---|---|---|
| Hand-constructed clean cases | 2 | 11 | 0 | 0 |
| Generated clean sequences, 7 generator shapes × 60 seeds | 420 | 15317 | 0 | 0.000000 |
| Gate refusals on the same 420 | 420 | — | 0 | 0 |

Both effect policies (`apply` and `block`) are checked on the hand cases.

**What this does not establish**, printed in §4 of the transcript and repeated
here because it matters: the generated population is clean *by construction*,
and the construction uses the executor under test, so a false alarm the
executor raises consistently would not appear. The hand-constructed cases are
the independent part and there are two of them. A zero rate over 420 cases is
evidence about consistency, not about correctness.

## 3. Timing arithmetic

`validate_timing.py`. Nineteen checks, every expected value derived by hand in
the transcript. Highlights:

| Check | Hand derivation | Measured |
|---|---|---|
| Gap met exactly | `FIRST` declares 12.5 s; `SECOND` at t=12.5 | no violation |
| Gap short by 0.5 s | `SECOND` at t=12.0 | shortfall 0.500000 s |
| Simultaneous commands | `SECOND` at t=0.0 | shortfall 12.500000 s |
| Same-command interval | `REPEATER` declares 300 s, runs at t=0 and t=200 | shortfall 100.000000 s, trace names step 0 |
| Interval is per-command | `A`, `B`, `A` at 0, 1, 2 with 300 s each | one violation, at step 2 |
| Hold interval half-open | `HOLDER` holds [0, 10); `TAKER` at t=10.0 | no violation |
| | `TAKER` at t=9.5 | mutual exclusion reported |
| Tolerance band, inside | gap short by 1e-10 s (0.1 × `TIME_TOL_S`) | not reported |
| Tolerance band, outside | gap short by 1e-6 s (1000 × `TIME_TOL_S`) | reported |
| Generator spacing arithmetic | 2.0 s owed to any next command, 30.0 s to a repeat, 5.0 s from a last step at t=25 | all three match, 1e-12 |

No boolean produced by a strict threshold comparison is asserted near its
boundary: the two tolerance probes sit four decades apart on either side of it.

## 4. The irreversible-command gate

`validate_irreversible_gate.py`.

| Check | Measured |
|---|---|
| `irreversible_before_verify` | `REFUSE`, exit code 2, naming `ANT_DEPLOY` and the unverified declaration |
| `irreversible_after_verify` | `NO_REFUSAL_FOUND`, exit code 0 |
| Refusal holds under both effect policies | yes, both |
| All six orderings of {`FIRE`, `VERIFY_P`, `FILLER`} | refuses exactly the 3 that put `FIRE` before `VERIFY_P`; 0 mismatches of 6 |
| A *false* precondition on an irreversible command | `precondition_false`, not escalated, gate does not refuse |
| 80 generated seeds with `irreversible_early` injected | 72 applied, 8 no-ops, 18 carried a gate-blocking kind, 18 refused |
| 80 clean generated cases | 0 refusals |

The clean verdict is called `NO_REFUSAL_FOUND` and its rendering says, every
time, that it is not a statement that the sequence is safe to uplink.

## 5. Static-lint miss rate — the deliverable

`validate_lint_missrate.py`. Population: 60 seeds × 9 defect families, 540
defective cases of which 526 injections applied and 14 were no-ops, plus 60
clean cases; 21660 command occurrences; 1894 violations found by forward
execution.

A violation and a lint finding are the same finding when they agree on
`(step, kind)`, which is generous to the baseline.

| Lint strength | Violations | Missed | **Miss rate** | Ordering-only violations | Missed | **Ordering-only miss rate** |
|---|---|---|---|---|---|---|
| `per_command` | 1894 | 993 | **0.524287** | 833 | 556 | **0.667467** |
| `with_timing` | 1894 | 805 | **0.425026** | 833 | 368 | **0.441777** |

The timing pass recovers 188 violations. Adding a check cannot increase the
miss count, and that is asserted rather than assumed.

The other half of the measurement, which decides whether anybody keeps running
the tool:

| Lint strength | Findings emitted | Not confirmed by the forward walk | **False-positive rate** |
|---|---|---|---|
| `per_command` | 17122 | 16221 | **0.947378** |
| `with_timing` | 17310 | 16221 | **0.937088** |

The lint re-evaluates every command occurrence against the declared initial
state, so it re-reports conditions an earlier command already fixed.

Per family, at `with_timing`:

| Family | Violations | Missed | Miss rate |
|---|---|---|---|
| `compress_gap` | 64 | 1 | 0.015625 |
| `irreversible_early` | 96 | 10 | 0.104167 |
| `undeclared_typo` (control) | 180 | 64 | 0.355556 |
| `swap_enabler` | 339 | 137 | 0.404130 |
| `delete_enabler` | 367 | 156 | 0.425068 |
| `mode_drop` | 209 | 105 | 0.502392 |
| `overlap_exclusion` | 211 | 108 | 0.511848 |
| `insert_invalidator` | 250 | 129 | 0.516000 |
| `resource_spike` | 178 | 95 | 0.533708 |

The control family's own kind, `undeclared_command`, is missed **zero** times
at either strength; the misses in that row are the other violations the typo
causes downstream.

## 6. The learned ranker against the free orderings

`validate_ranker.py`. Full tables are in `MODEL_CARD.md`; the headline, on 302
scored held-out cases from seeds the model never saw:

| Ordering | P@5 | P@10 | P@20 |
|---|---|---|---|
| learned, all features | **0.270199** | **0.210265** | **0.144040** |
| learned, timing features withheld | 0.213907 | 0.174503 | 0.135265 |
| irreversibility flag | 0.156954 | 0.150331 | 0.102483 |
| resource draw | 0.183444 | 0.157285 | 0.102152 |
| random control | 0.071523 | 0.084106 | 0.098344 |

**The specification predicted the flag ordering would match or beat the learned
model. On this corpus it did not**, and the paired bootstrap intervals exclude
zero at every cut-off (+0.113245 [+0.088079, +0.138411] at k=5). That is
reported as measured, in the direction it came out.

### Results where a baseline won, or where nothing was distinguished

- **At k=20 the free orderings are not distinguished from a random shuffle.**
  Flag minus random: +0.004139 [-0.002649, +0.010762]. Resource draw minus
  random: +0.003808 [-0.003146, +0.010596]. Both intervals include zero.
- **About a third of the learned margin at k=5 is an artefact of the
  generator.** Withholding three timing-shape features costs 0.056291 at k=5,
  0.035762 at k=10, 0.008775 at k=20, because four of the nine defect families
  edit the time column.
- **The irreversibility flag is the least important feature to the forest**
  (importance 0.003546 of 1.0) and the flag ordering still beats a shuffle at
  k=5 and k=10.
- **The risk is not calibrated.** Weighted mean absolute calibration gap
  **0.331780**. In the [0.4, 0.5) risk bin the observed involved fraction is
  0.079782. The model is used as a ranking and is not corrected.

## 7. Determinism under re-run

`validate_determinism.py`. Asserted by test as well as measured here
(`tests/test_trace.py::test_determinism_under_rerun`,
`tests/test_properties.py::test_forward_execution_is_deterministic`).

| Component | Repeats | Distinct results |
|---|---|---|
| Forward walk, 5 hand cases | 20 each | 1 fingerprint each |
| Forward walk, 30 generated cases × 2 effect policies | 5 each | 1 each; 0 unstable |
| Generator, 30 seeds | 2 each | 0 differing sequences |
| Injections, 12 seeds × 9 families | 2 each | 0 differing of 108 |
| Static lint, both strengths, 12-seed corpus | 3 each | 1 digest each |
| Miss rate, `with_timing` | 3 | 1 value, bit-identical: 0.46701846965699206 |
| Ranker, two fits at seed 99 | 2 | identical risks and spreads, max difference 0.000e+00 |
| Ranker at seed 100 | 1 | different, as it must be |

## 8. Cost — table explicitly declared unstable

Wall clock on a shared two-core container, mean over 5 repeats
(`validate_determinism_output.txt` §6). **These numbers move between runs and
between machines, and no conclusion in this repository is derived from them.**

| Case | Steps | Forward walk (ms) | Static lint (ms) |
|---|---|---|---|
| `irreversible_before_verify` | 3 | 0.0387 | 0.0095 |
| `irreversible_after_verify` | 3 | 0.0813 | 0.0099 |
| `late_payload_enable` | 7 | 0.0690 | 0.0210 |
| `resource_and_exclusion` | 7 | 0.1298 | 0.0155 |
| `nominal_science_pass` | 8 | 0.8950 | 0.0347 |
| generated, seed 0 | 36 | 0.2658 | 0.1088 |
| generated, seed 1 | 36 | 0.2495 | 0.0860 |
| generated, 240 steps | 240 | 2.3347 | 1.3127 |

| Component | Measured |
|---|---|
| Ranker fit, 200 trees on 6500 rows | 1.453 s |
| Ranker fit, 200 trees on 22750 rows | 2.636 s |
| Whole corpus build, 70 + 35 seeds | 1.240 s |

The `nominal_science_pass` row is the first case the timer touches and it
carries the process's warm-up: 0.8950 ms for an 8-step sequence against
2.3347 ms for a 240-step one. That is exactly the kind of artefact a derived
ratio in prose would turn into a false claim, which is why there is none.

## 9. CLI contract

`validate_cli.py`. 21 subprocess invocations, every exit code checked, and
every line of every stdout and stderr scanned for an absolute path (0
offenders).

| Exit code | Meaning | Example |
|---|---|---|
| 0 | nothing found against the declared model | `check examples/cases/nominal_science_pass.yaml` |
| 1 | violations found, none gate-blocking | `check examples/cases/late_payload_enable.yaml` |
| 2 | the irreversible-command gate refused | `gate examples/cases/irreversible_before_verify.yaml` |
| 3 | the input could not be loaded, or bad arguments | `check examples/cases/missing_declarations.yaml` |

## 10. Working-directory independence

Required because four Batch 07 products of this mission failed a release gate
on exactly this. Every validation script was executed from `/`, `/tmp`, the
product root, `validation/`, `src/`, `examples/` and `tests/` — the two
expensive ones (`validate_cli.py`, `validate_ranker.py`) from the first four —
and every run exited 0. The examples were run from four directories, and the
test suite from three. The probe used `python <script>`, never `python -I`,
because `-I` strips the script's own directory from `sys.path` and produces
false failures.

## 11. Checks that failed during the build

Reported because a check that fails and is reported is the credible kind.

1. **A Hypothesis property was false, and the property was wrong, not the
   code.** The claim was that `EffectPolicy.BLOCK` never reports fewer
   violations than `EffectPolicy.APPLY`. Hypothesis minimised it to one
   command gated to a mode it is not in, which also declares a transition the
   machine forbids: two violations under `APPLY`, one under `BLOCK`, because a
   command that changes no state cannot make an illegal transition. The
   behaviour is correct; the property was replaced with the identity that does
   hold — the two policies agree on whether the sequence is clean and on the
   first violating step — and the counterexample is pinned with literal inputs
   in `tests/test_regressions.py::test_block_policy_can_report_fewer_violations_than_apply`.
2. **A defect family's arithmetic was wrong and its no-op count exposed it.**
   `compress_gap` used `max(previous.min_gap_after_s, own.min_interval_s)` as
   the spacing to break, but `min_interval_s` is measured from the command's
   own previous execution. 18 of 40 injections were no-ops. The family now
   targets `min_gap_after_s` only and applies on 40 of 40; pinned in
   `tests/test_regressions.py::test_compress_gap_family_no_longer_uses_min_interval_as_a_gap`.
3. **The first generator could not reach four of the nine defect families.** A
   uniform random walk placed mostly teardown commands and never placed
   `PAYLOAD_SCAN`; `overlap_exclusion` and `resource_spike` were no-ops on
   40 of 40 seeds. The generator was rewritten as goal-directed and the
   property is asserted in
   `tests/test_dataset.py::test_generated_sequences_reach_the_interesting_commands`.
4. **`irreversible_early` injections also broke a timing gap, 36 times out of
   36.** The intended finding was buried under `timing_gap`. The family now
   shifts the sequence tail by the moved command's own `min_gap_after_s`, so
   the edit is a reordering; 10 of 36 still produced no timing finding at all
   and 7 produced no violation, and both counts are printed rather than the
   injection being retried until it bit.

None of these was resolved by widening a tolerance or by removing a check.

## 12. Dependency ranges — a limitation, stated

Every dependency is bounded on both sides at the minor version measured. **Only
the single installed patch release of each was executed.** Installing into this
build container is prohibited after a shared interpreter was mutated mid-build
on 2026-10-16 (numpy 2.5.3 → 2.5.4 → 2.5.0, with transient broken installs and
bus errors), so the suite was **not** run at both ends of each declared range.
The ceilings are the ranges this product claims, not two measured endpoints.
Versions actually executed are at the head of every transcript in this
directory: Python 3.13.17, numpy 2.5.3, scikit-learn 1.9.1, pyyaml 6.0.3,
matplotlib 3.10.8, pytest 8.4.2, hypothesis 6.140.3, ruff 0.14.2.
