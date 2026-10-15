# Requirements — safemodetree 0.1.0

Validation level 3. Every requirement below is numbered, carries the test or
validation script that checks it, and is either SATISFIED or carries a stated
deviation. Nothing here is a flight-software requirement; this is a research
package and the requirement list exists so a reader can see what was checked
and what was not.

Requirement identifiers are stable and are referenced from
`validation/VALIDATION.md`.

## 1. Declarative input

| id | requirement | evidence | status |
|---|---|---|---|
| R-1.1 | The input declares faults with an effect, a severity category, a detectability flag and a detection latency in seconds. | `tests/test_spec.py::test_fmea_shape`, `::test_severity_is_ordinal_and_ranked` | SATISFIED |
| R-1.2 | The input declares modes, of which at least one is marked safe; a declaration with no safe mode is refused. | `tests/test_validation_errors.py::test_declaration_without_safe_mode_is_rejected` | SATISFIED |
| R-1.3 | The input declares responses with a target mode, a cost in minutes of mission outage and a precondition. | `tests/test_spec.py::test_minimal_yaml_round_trip` | SATISFIED |
| R-1.4 | The input declares an observability model: one binary signature per fault over the declared observation channels. | `tests/test_spec.py::test_observability_expected_vector_and_nominal` | SATISFIED |
| R-1.5 | Severity categories are treated as ordinal only; no arithmetic is performed on them and no Risk Priority Number is computed. | source inspection, `src/safemodetree/spec.py` module docstring | SATISFIED |
| R-1.6 | Every structural error in a declaration raises `ValueError` or `TypeError` with a message naming the offending identifier. An unknown key is an error, not a silently ignored field. | `tests/test_validation_errors.py`, 39 cases | SATISFIED |

## 2. Deterministic core

| id | requirement | evidence | status |
|---|---|---|---|
| R-2.1 | The response automaton has one state per (mode, active fault) pair plus the no-fault state per mode. | `tests/test_known_answers.py::test_state_and_transition_counts` | SATISFIED |
| R-2.2 | A transition is enabled only when the declared `from_mode`, the declared precondition and the detection assumption all permit it. No transition leaves a mode declared absorbing. | `tests/test_automaton.py`, 18 cases | SATISFIED |
| R-2.3 | From every fault state, the checker reports whether a declared safe mode is reachable under the detection assumption in force, with a witness path and its cost. | `tests/test_reach.py::test_all_fault_states_reach_a_safe_mode_in_the_shipped_declaration` | SATISFIED |
| R-2.4 | Reachability and cheapest-path results agree with exhaustive enumeration of every simple path on small automata. | `validation/validate_reachability.py`, 2000 declarations | SATISFIED |
| R-2.5 | Declared safe modes that no fault state can reach are reported. | `tests/test_reach.py::test_planted_case_is_detected` | SATISFIED |
| R-2.6 | Fault states with no enabled response are reported, excluding those already in a declared safe mode. | `tests/test_diagnostics.py::test_fault_state_without_response_is_reported` | SATISFIED |
| R-2.7 | Responses whose precondition can never hold in any candidate source mode are reported, with the reason per mode. | `tests/test_diagnostics.py::test_vacuous_responses_are_named_with_reasons` | SATISFIED |
| R-2.8 | Ambiguity classes — faults with equal declared signatures — are reported, and a class whose members prefer different responses is flagged as conflicting. | `tests/test_diagnostics.py::test_ambiguity_classes_known_answers` | SATISFIED |
| R-2.9 | For each ambiguity class the checker computes the class-conservative route, or reports that none exists. | `tests/test_diagnostics.py::test_conservative_response_returns_none_when_nothing_is_agnostic` | SATISFIED |
| R-2.10 | Precondition emptiness is decided exactly, not heuristically, over the declared fragment (conjunctions of literals, per-mode availability, an always-false flag). | `tests/test_properties.py::test_precondition_emptiness_matches_a_direct_check` | SATISFIED |
| R-2.11 | The whole core is deterministic: the same declaration and assumption give the same report, with deterministic tie-breaking on witness paths. | `src/safemodetree/reach.py::min_cost_to_safe` tie-break on (cost, hops, response ids) | SATISFIED |

## 3. AI component

| id | requirement | evidence | status |
|---|---|---|---|
| R-3.1 | The declared threshold-logic baseline is implemented and measured before the learned model. | commit order; `src/safemodetree/baseline.py` precedes `src/safemodetree/learned.py` in the build, and the baseline's decision rule is declared, not fitted | SATISFIED |
| R-3.2 | The learned classifier exposes a calibrated probability over every declared class, not only a point estimate. | `tests/test_learned.py::test_posterior_rows_sum_to_one` | SATISFIED |
| R-3.3 | The learned classifier is benchmarked against the baseline on held-out data at an equal false-alarm rate, with the matching done on a separate calibration split. | `validation/validate_calibration.py` section 2 | SATISFIED |
| R-3.4 | Calibration is measured as a reliability curve and an expected calibration error, at more than one bin count. | `validation/validate_calibration.py` sections 3 and 4 | SATISFIED |
| R-3.5 | The policy consumes the confidence: an identification below the declared threshold is routed to the class-conservative response. | `tests/test_policy.py::test_low_confidence_routes_to_the_conservative_response` | SATISFIED |
| R-3.6 | The cost of that conservatism is measured in declared units and reported with its sign, including where it is negative. | `validation/validate_conservatism.py` section 3 | SATISFIED |
| R-3.7 | Where the baseline wins, it is reported. | README "What this does", `validation/validate_calibration.py` section 2 FINDING | SATISFIED — on identification accuracy at an equal false-alarm rate the baseline wins at 2 of 5 noise scales, the learned model at 2, and 1 is a tie |

## 4. Validation level 3

| id | requirement | evidence | status |
|---|---|---|---|
| R-4.1 | Reachability results cross-checked against exhaustive enumeration on small automata. | `validation/validate_reachability.py` | SATISFIED |
| R-4.2 | A planted unreachable-safe-mode case is detected and localised. | `validation/validate_planted_case.py` | SATISFIED |
| R-4.3 | Classifier calibration is measured and compared to the baseline at an equal false-alarm rate. | `validation/validate_calibration.py` | SATISFIED |
| R-4.4 | Every number in `validation/VALIDATION.md` comes from a script in `validation/` that was run, with its raw output committed beside it. | `validation/*_output.txt` | SATISFIED |
| R-4.5 | Hand-calculated known answers exist for every arithmetic rule. | `validation/validate_known_answers.py`, 35 checks | SATISFIED |
| R-4.6 | Property-based tests cover the algebraic identities, and every boundary case the strategies can draw is pinned as an explicit regression test with literal inputs. | `tests/test_properties.py` | SATISFIED |
| R-4.7 | Both ends of every declared dependency range were installed and the whole suite plus every validation script was run at each end, and the committed validation output was diffed. | `validation/VALIDATION.md` section 9 | SATISFIED |
| R-4.8 | No physical validation of any kind. | — | DEVIATION, stated: there is no hardware in this product and no measurement of any spacecraft. Level 3 here means numerically cross-checked, not physically validated. |

## 5. Non-requirements, stated so they are not assumed

| id | statement |
|---|---|
| N-5.1 | This package does not detect or isolate faults from telemetry in flight. The classifier is trained on a synthetic generator defined by the declaration itself. |
| N-5.2 | This package does not verify temporal properties. It decides reachability of a declared safe mode over a static observation vector, not LTL or CTL properties over traces. A model checker does that; see the README alternatives table. |
| N-5.3 | This package does not handle simultaneous independent faults. One fault is active at a time. |
| N-5.4 | This package does not decide diagnosability in the discrete-event sense of Sampath et al. 1995. It decides equality of declared static signatures, which is strictly weaker. |
| N-5.5 | General Boolean preconditions are out of scope. The declared fragment is conjunctions of literals, chosen so emptiness is decidable in linear time without a SAT solver. |
| N-5.6 | Declared response costs are assumed additive along a path and independent of order. The package does not check that assumption and cannot. |
