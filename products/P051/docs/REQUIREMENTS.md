# simplexguard — requirements

Validation level 3. Every requirement below names the test or validation script
that exercises it. A requirement with no such name is not a requirement; it is a
wish, and none appear here.

**Conventions.** `tests/...` are pytest tests, run by `python -m pytest tests/ -q`
from the repository root. `validation/...` are standalone scripts whose raw
stdout is committed beside them. Verification method is one of **T** (automated
test), **V** (validation script with committed output), **A** (analysis, stated
in a module docstring and checked by a T or a V), **I** (inspection).

**Scope of every requirement.** All of them are conditional on the declared
plant, the declared disturbance set and the declared constraint sets supplied by
the caller. Nothing here is a statement about any physical system.

---

## 1. Declared model

| # | Requirement | Method | Verified by |
|---|---|---|---|
| REQ-001 | The plant shall be a discrete-time linear system `x_{k+1} = A x_k + B u_k + w_k` with `A`, `B` exactly known, and shall validate the shapes and finiteness of `A`, `B`, `W`, `X`, `U` on construction. | T | `tests/test_plant.py::test_plant_constructor_validation`, `::test_step_rejects_wrong_shapes` |
| REQ-002 | The disturbance set `W` shall be declared explicitly, shall be bounded, and shall contain the origin; a set that does not shall be rejected with an actionable error. | T | `tests/test_plant.py::test_plant_rejects_a_disturbance_set_without_the_origin`, `tests/test_polytope.py::test_erode_rejects_an_unbounded_disturbance_set` |
| REQ-003 | The state constraints `X` and input constraints `U` shall be declared as polytopes in the units of the state and input respectively. | T, I | `tests/test_plant.py::test_describe_mentions_the_declared_sets`, `::test_reference_plant_disturbance_box_known_answer` |
| REQ-004 | The shipped reference plant shall be the exact zero-order-hold discretisation of a double integrator at the stated sample interval, and its disturbance box shall be the stated angular acceleration acting for one sample. | T | `tests/test_plant.py::test_reference_plant_zoh_discretisation_known_answer`, `::test_reference_plant_disturbance_box_known_answer`, `::test_step_known_answer` |
| REQ-005 | `Plant.step` shall **not** validate the disturbance against the declared set, so that the bound-violation experiment can pass an inadmissible disturbance through the plant. | T | `tests/test_plant.py::test_step_does_not_check_the_disturbance_bound` |
| REQ-006 | Invalid plant parameters (non-positive `dt`, non-positive limits, non-finite matrices) shall raise `ValueError` with a message naming the parameter. | T | `tests/test_plant.py::test_reference_plant_rejects_bad_parameters` |

## 2. Support-function and set algebra

| # | Requirement | Method | Verified by |
|---|---|---|---|
| REQ-010 | The support function `h_P(c) = sup_{x in P} c^T x` shall be computed exactly: in closed form for a box, by linear programme otherwise, and the two shall agree. | T, V | `tests/test_polytope.py::test_box_support_known_answer`, `::test_generic_polytope_support_matches_box_closed_form`, `tests/test_properties.py::test_box_closed_form_equals_the_generic_linear_programme`; `validation/validate_support_function.py` checks 1, 2 |
| REQ-011 | The support function shall satisfy positive homogeneity and subadditivity. | T, V | `tests/test_properties.py::test_support_is_positively_homogeneous`, `::test_support_is_subadditive`; `validation/validate_support_function.py` check 3 |
| REQ-012 | The support function of a Minkowski sum shall equal the sum of the support functions. | T, V | `tests/test_properties.py::test_support_of_a_minkowski_sum_adds`; `validation/validate_support_function.py` check 4 |
| REQ-013 | The Pontryagin difference `P (-) W` shall be computed by the exact halfspace identity `b_i -> b_i - h_W(a_i)`, and shall be equivalent to "`x + w in P` for every `w` in `W`". | T, V | `tests/test_polytope.py::test_erode_known_answer`, `::test_erode_is_exactly_the_pontryagin_difference`; `tests/test_properties.py::test_erosion_is_the_pontryagin_difference`; `validation/validate_support_function.py` check 5 |
| REQ-014 | Erosion shall be monotone in the disturbance set, and eroding by a strictly larger set shall empty the result. | T, V | `tests/test_properties.py::test_erosion_is_monotone_in_the_disturbance_set`, `::test_eroding_by_a_larger_box_empties_the_set`; `validation/validate_support_function.py` check 6 |
| REQ-015 | The preimage `M^{-1} P` shall satisfy `x in M^{-1} P` if and only if `M x in P`. | T, V | `tests/test_polytope.py::test_preimage_identity`; `tests/test_properties.py::test_preimage_membership_identity`; `validation/validate_support_function.py` check 7 |
| REQ-016 | Redundancy removal shall preserve the set exactly, and shall keep exactly one copy of a duplicated row. | T, V | `tests/test_polytope.py::test_minimal_removes_redundant_rows_and_keeps_the_square`, `::test_minimal_keeps_exactly_one_of_a_duplicated_row`; `tests/test_properties.py::test_minimal_preserves_membership`; `validation/validate_support_function.py` check 8 |
| REQ-017 | Emptiness shall be detected by a signed Chebyshev radius, negative exactly when the polytope is empty, and the support function of an empty polytope shall be `-inf`. | T | `tests/test_polytope.py::test_empty_polytope_is_detected_and_support_is_minus_infinity`, `::test_chebyshev_radius_known_answer` |
| REQ-018 | The support function of an unbounded polytope shall be `+inf` in an unbounded direction, not an exception. | T | `tests/test_polytope.py::test_unbounded_polytope_support_is_plus_infinity` |
| REQ-019 | All public set operations shall raise `ValueError` or `TypeError` on dimension mismatch, non-finite input or a degenerate request. | T | `tests/test_polytope.py::test_dimension_mismatch_errors`, `::test_polytope_constructor_rejects_bad_input`, `::test_support_rejects_wrong_length_and_non_finite_direction`, `::test_box_constructor_rejects_bad_bounds`, `::test_scaled_rejects_non_positive_factors`, `::test_box_refuses_to_enumerate_too_many_vertices`, `::test_vertices_2d_requires_two_dimensions` |

## 3. Controllers

| # | Requirement | Method | Verified by |
|---|---|---|---|
| REQ-020 | The discrete LQR gain shall solve the discrete algebraic Riccati equation and shall reproduce the scalar closed form. | T | `tests/test_controllers.py::test_scalar_lqr_known_answer`, `::test_lqr_gain_stabilises_the_closed_loop` |
| REQ-021 | A larger LQR input weight shall give a lower-authority gain, and the shipped baseline shall be slower than the shipped performance controller. | T | `tests/test_controllers.py::test_larger_input_weight_gives_a_lower_authority_gain`, `::test_reference_controllers_baseline_is_slower_than_performance` |
| REQ-022 | Both controllers shall saturate into the declared input box, and the performance controller shall report whether the saturation binds. | T | `tests/test_controllers.py::test_saturate_known_answer`, `::test_baseline_controller_is_minus_gain_times_state_then_saturated`, `::test_performance_controller_tracks_the_reference`, `::test_saturates_at_flag` |
| REQ-023 | Invalid gains, weights and input sets shall raise. | T | `tests/test_controllers.py::test_dlqr_rejects_bad_weights`, `::test_controller_constructors_validate`, `::test_controller_calls_validate_shapes`, `::test_closed_loop_matrix_rejects_a_misshaped_gain`, `::test_saturate_rejects_a_non_box_input_set` |

## 4. The robust invariant set (the certificate)

| # | Requirement | Method | Verified by |
|---|---|---|---|
| REQ-030 | The computed set `S` shall be a subset of the declared `X`. | T, V | `tests/test_invariant.py::test_invariant_set_is_a_strict_subset_of_the_constraint_set`, `::test_reference_plant_set_is_verified_exactly`; `validation/validate_invariant_set.py` check 2a |
| REQ-031 | The baseline input shall be admissible everywhere in `S`. | T, V | `tests/test_invariant.py::test_reference_plant_set_is_verified_exactly`; `validation/validate_invariant_set.py` check 2b |
| REQ-032 | `S` shall be robustly positively invariant under the baseline closed loop for every disturbance in the declared `W`, and this shall be verified **exactly** by support function and not by sampling. | T, V | `tests/test_invariant.py::test_reference_plant_set_is_verified_exactly`, `::test_one_dimensional_boundary_case_is_exactly_invariant`; `validation/validate_invariant_set.py` checks 2c, 3 |
| REQ-033 | The verifier shall report the three properties separately, so that a set which is invariant but outside `X` is distinguished from one which is inside `X` but not invariant. | T | `tests/test_invariant.py::test_verify_detects_a_set_that_is_not_invariant`, `::test_scaling_an_invariant_set_keeps_invariance_but_leaves_the_constraints` |
| REQ-034 | The one-dimensional case shall reproduce its closed-form answer exactly, including the boundary case and the case where the input constraint truncates the initial set. | T, V | `tests/test_invariant.py::test_one_dimensional_invariant_set_known_answer`, `::test_one_dimensional_set_truncated_by_the_input_constraint_known_answer`, `::test_one_dimensional_boundary_case_is_exactly_invariant`, `::test_one_dimensional_set_empties_when_b0_is_below_twice_wmax`; `validation/validate_invariant_set.py` check 1 |
| REQ-035 | The recursion shall be monotone decreasing, and a larger declared `W` shall give a smaller `S`. | T, V | `tests/test_invariant.py::test_recursion_is_monotone_decreasing`, `::test_larger_disturbance_gives_a_smaller_set`; `validation/validate_invariant_set.py` checks 4a, 4b |
| REQ-036 | If the recursion empties the set, the function shall raise `EmptyInvariantSet` naming the iteration, and shall not return a set that is not invariant. | T, V | `tests/test_invariant.py::test_a_too_conservative_baseline_empties_the_set`, `::test_a_disturbance_larger_than_the_constraints_empties_the_set`, `tests/test_degenerate.py::test_very_tight_constraints_empty_the_set`, `::test_a_degenerate_disturbance_equal_to_the_constraint_set_empties`; `validation/validate_invariant_set.py` check 5 |
| REQ-037 | If the recursion reaches its iteration cap, the function shall raise `RecursionDidNotConverge` and shall not return the last iterate as a certificate. | T, V | `tests/test_invariant.py::test_recursion_cap_raises_rather_than_returning_a_non_certificate`; `validation/validate_invariant_set.py` check 5 |
| REQ-038 | Redundancy removal shall be optional and shall not change the computed set. | T, V | `tests/test_invariant.py::test_unreduced_recursion_reaches_the_same_set`, `::test_halfspace_count_grows_by_two_per_iteration`; `validation/validate_invariant_set.py` check 6 |
| REQ-039 | A zero disturbance shall be accepted and shall reduce the recursion to the maximal output admissible set. | T | `tests/test_degenerate.py::test_zero_disturbance_gives_the_maximal_output_admissible_set`, `::test_zero_disturbance_guard_has_a_zero_erosion`, `tests/test_plant.py::test_zero_disturbance_is_allowed_and_gives_a_degenerate_box` |
| REQ-040 | The result shall carry the iteration count, the facet count per iteration and the closed-loop matrix, so the computation can be audited. | T | `tests/test_invariant.py::test_describe_reports_convergence_and_area`, `::test_halfspace_count_grows_by_two_per_iteration` |

## 5. The exact switching condition

| # | Requirement | Method | Verified by |
|---|---|---|---|
| REQ-050 | The performance input shall be admitted if and only if `A x + B u + w` lies in `S` for **every** `w` in the declared `W`, evaluated by support function on the declared polytope and not by sampling. | T, V | `tests/test_guard.py::test_one_dimensional_switching_condition_known_answer`, `::test_support_function_form_agrees_with_vertex_enumeration`; `validation/validate_guard_exactness.py` checks 1, 2 |
| REQ-051 | The eroded set `S (-) W` used by the condition shall be precomputed once and shall equal `S` eroded by the declared `W`. | T | `tests/test_guard.py::test_one_dimensional_eroded_set_known_answer`, `::test_eroded_set_is_strictly_inside_the_invariant_set` |
| REQ-052 | The condition shall report a signed margin equal to the residual on the binding facet, in state units, whether or not the input is admitted. | T, V | `tests/test_guard.py::test_margin_is_reported_even_when_the_performance_input_is_admitted`; `validation/validate_guard_exactness.py` check 3 |
| REQ-053 | An input outside the declared `U` shall be refused and flagged. | T, V | `tests/test_guard.py::test_an_inadmissible_input_is_refused`; `validation/validate_guard_exactness.py` check 6a |
| REQ-054 | A state outside `S` shall be flagged as certificate lost, and the decision record shall say so. | T, V | `tests/test_guard.py::test_a_state_outside_the_invariant_set_is_reported_as_certificate_lost`, `tests/test_degenerate.py::test_an_initial_state_outside_the_invariant_set_is_reported_not_hidden`; `validation/validate_guard_exactness.py` check 6b |
| REQ-055 | A guarded episode started inside `S` shall produce zero constraint violations and zero invariant-set exits for every disturbance sequence inside the declared bound. | T, V | `tests/test_simulate.py::test_guarded_run_has_no_violations_inside_the_declared_bound`, `tests/test_integration.py::test_guarded_states_stay_in_the_invariant_set_for_every_sampler`; `validation/validate_guard_exactness.py` check 5, `validation/validate_accounting.py` checks 2, 3 |
| REQ-056 | A minimum baseline dwell shall be available, shall never be applied to the performance side, and shall not cause a violation. | T, V | `tests/test_guard.py::test_minimum_baseline_dwell_holds_the_baseline`, `::test_reset_clears_the_dwell_counter`, `::test_advance_dwell_false_leaves_the_counter_alone`, `tests/test_accounting.py::test_minimum_dwell_reduces_the_switch_rate`; `validation/validate_accounting.py` check 5c |
| REQ-057 | The guard shall refuse to accept a set that fails the certificate check when asked to verify it. | T, V | `tests/test_guard.py::test_verify_true_rejects_a_set_that_is_not_a_certificate`, `::test_verify_true_accepts_the_computed_set` |
| REQ-058 | The guard shall validate the shapes and finiteness of its arguments. | T | `tests/test_guard.py::test_decide_validates_shapes`, `::test_guard_constructor_validation`, `::test_brute_force_needs_a_box_disturbance_set` |

## 6. Simulation and pairing

| # | Requirement | Method | Verified by |
|---|---|---|---|
| REQ-060 | Guarded, unguarded and baseline-only episodes shall be runnable on the same disturbance sequence, and the accounting shall refuse unpaired episodes. | T | `tests/test_simulate.py::test_pairing_is_exact`, `tests/test_accounting.py::test_account_refuses_unpaired_episodes`, `::test_account_refuses_different_lengths` |
| REQ-061 | Three disturbance samplers shall be available (zero, uniform, worst-case vertex), all inside the declared bound, and all shall give zero guarded violations. | T, V | `tests/test_simulate.py::test_disturbance_sequence_respects_the_declared_box`, `::test_vertex_sampler_only_produces_vertices`, `::test_zero_sampler_is_exactly_zero`, `tests/test_accounting.py::test_vertex_sampler_also_gives_zero_violations`; `validation/validate_accounting.py` check 3 |
| REQ-062 | A scaled sampler shall be available for the bound-violation experiment and shall leave the declared set above a scale of 1. | T | `tests/test_simulate.py::test_scaled_sampler_leaves_the_declared_box` |
| REQ-063 | Simulation shall be deterministic for a given disturbance sequence, and the episode record shall include the terminal state. | T | `tests/test_simulate.py::test_simulation_is_deterministic`, `::test_episodes_record_the_terminal_state` |
| REQ-064 | Simulation shall validate its arguments. | T | `tests/test_simulate.py::test_simulate_validates_shapes`, `::test_disturbance_sequence_validation`, `::test_square_wave_reference_validation` |

## 7. The assurance accounting

| # | Requirement | Method | Verified by |
|---|---|---|---|
| REQ-070 | The report shall give the switch rate per 1000 steps and per second. | T, V | `tests/test_accounting.py::test_report_is_internally_consistent`; `validation/validate_accounting.py` checks 1, 2 |
| REQ-071 | The report shall give the dwell-time distribution of each authority, including the minimum, median, mean, maximum and the fraction of one-step intervals, with a histogram. | T, V | `tests/test_accounting.py::test_dwell_statistics_known_answer`, `::test_dwell_statistics_histogram_known_answer`, `::test_mode_runs_known_answer`, `::test_mode_runs_cover_the_whole_sequence`, `tests/test_regression.py::test_the_guard_chatters_on_the_reference_scenario`; `validation/validate_accounting.py` check 1 |
| REQ-072 | The report shall give the fraction of the episode under baseline authority, and it shall agree with the dwell totals exactly. | T, V | `tests/test_accounting.py::test_report_is_internally_consistent`; `validation/validate_accounting.py` check 4 |
| REQ-073 | The report shall give the conservatism cost as a paired comparison against an unguarded run on the same disturbance, and against a baseline-only run, with the fraction of the gap recovered. | T, V | `tests/test_accounting.py::test_conservatism_cost_sits_between_the_two_extremes`, `tests/test_regression.py::test_reference_scenario_accounting_is_pinned`; `validation/validate_accounting.py` checks 1, 2b, 6 |
| REQ-074 | The report shall give the constraint-violation count, which must be zero for a guarded run under the declared bound, and shall report it as non-zero if it is not. | T, V | `tests/test_accounting.py::test_guard_eliminates_the_violations_the_unguarded_run_makes`; `validation/validate_accounting.py` checks 2, 3, 6 |
| REQ-075 | The report shall give the invariant-set exit count separately from the constraint-violation count. | T, V | `tests/test_accounting.py::test_guard_eliminates_the_violations_the_unguarded_run_makes`, `tests/test_boundviolation.py::test_the_certificate_is_lost_at_least_as_often_as_a_constraint_is_broken`; `validation/validate_bound_violation.py` check 4 |
| REQ-076 | The report shall give the distribution of the switching margin and show it is negative exactly at the firing steps. | T | `tests/test_accounting.py::test_margin_is_negative_exactly_at_the_firing_steps` |
| REQ-077 | The report shall be printable as a fixed-width block containing all six headings. | T | `tests/test_accounting.py::test_describe_contains_all_six_headings` |
| REQ-078 | Dwell statistics shall reject invalid lengths and shall handle an empty interval list. | T | `tests/test_accounting.py::test_dwell_statistics_empty_and_invalid` |

## 8. The bound-violation experiment

| # | Requirement | Method | Verified by |
|---|---|---|---|
| REQ-080 | The sweep shall run the guard unchanged, computed from the declared `W`, while the realised disturbance is scaled. | T, V | `tests/test_boundviolation.py::test_fraction_outside_the_declared_set_rises_with_the_scale`; `validation/validate_bound_violation.py` checks 2, 3 |
| REQ-081 | At a scale of 1 the sweep shall report zero constraint violations and zero invariant-set exits. | T, V | `tests/test_boundviolation.py::test_scale_one_has_no_violations_and_no_exits`; `validation/validate_bound_violation.py` check 1 |
| REQ-082 | Above a scale of 1 the sweep shall report where the guarantee fails, for both samplers, including the smallest scale at which each failure first appears. | T, V | `tests/test_boundviolation.py::test_violations_appear_and_grow_beyond_the_declared_bound`, `::test_the_vertex_sampler_breaks_the_guarantee_sooner_than_uniform`, `::test_describe_reports_both_thresholds`; `validation/validate_bound_violation.py` checks 2, 3 |
| REQ-083 | The invariant-set exit count shall be at least the constraint-violation count at every scale, since `S` is a subset of `X`. | T, V | `tests/test_boundviolation.py::test_the_certificate_is_lost_at_least_as_often_as_a_constraint_is_broken`; `validation/validate_bound_violation.py` check 4 |
| REQ-084 | The sweep shall refuse a scale list that does not include the guaranteed case. | T | `tests/test_boundviolation.py::test_sweep_validation` |

## 9. The learned switch predictor (AI component)

| # | Requirement | Method | Verified by |
|---|---|---|---|
| REQ-090 | The analytic baseline shall be implemented and benchmarked **before** the learned model, on the same held-out episodes. | V, I | `validation/validate_predictor.py` checks 2, 3; `simplexguard/reachability.py` and `simplexguard/guard.py` predate `simplexguard/predictor.py` in the dependency graph, which imports both |
| REQ-091 | The exact multi-step predictor shall reproduce the one-step switching condition exactly at horizon 0. | T, V | `tests/test_reachability.py::test_horizon_zero_reproduces_the_one_step_guard_condition`; `validation/validate_guard_exactness.py` check 4 |
| REQ-092 | The exact multi-step predictor shall be monotone in the horizon, and its worst-case variant shall dominate its nominal variant. | T | `tests/test_reachability.py::test_prediction_is_monotone_in_the_horizon`, `::test_worst_case_dominates_nominal_on_identical_inputs` |
| REQ-093 | Datasets shall be split by episode, never by row, and shall be deterministic for a given seed. | T, V | `tests/test_predictor.py::test_split_by_episode_is_disjoint`, `::test_dataset_is_deterministic`; `validation/validate_predictor.py` check 1 |
| REQ-094 | The learned model shall be given raw state and reference quantities only, and shall **not** be given the switching-condition margin. | T, I | `tests/test_predictor.py::test_features_are_the_documented_raw_quantities`, `::test_dataset_shape_and_feature_names`; `simplexguard/predictor.py` `FEATURE_NAMES` |
| REQ-095 | The learned model shall expose a calibrated probability, calibrated on a split the base estimator never saw, and shall report its Brier score and expected calibration error. | T, V | `tests/test_predictor.py::test_expected_calibration_error_known_answer`, `::test_expected_calibration_error_is_zero_for_a_perfect_forecaster`; `validation/validate_predictor.py` checks 2d, 6 |
| REQ-096 | On the switching condition's own criterion, the exact computation shall be reported as exactly right, and the learned model's loss shall be published with its structural reason and without retuning. | T, V | `tests/test_predictor.py::test_exact_one_step_condition_is_perfect_on_the_lead_zero_target`, `::test_learned_model_loses_to_the_exact_condition_on_its_own_criterion`, `tests/test_regression.py::test_the_exact_condition_stays_perfect_on_its_own_criterion`; `validation/validate_predictor.py` checks 2a, 2b |
| REQ-097 | The per-decision cost of each predictor shall be measured and reported, and the comparison shall include at least four forest sizes. | T, V | `tests/test_predictor.py::test_learned_model_is_slower_per_decision_than_the_exact_condition`, `tests/test_regression.py::test_exact_condition_is_cheaper_than_the_learned_model`; `validation/validate_predictor.py` checks 2c, 3c, 5 |
| REQ-098 | Precision, recall and lead time shall be reported for every predictor on both tasks. | T, V | `tests/test_predictor.py::test_lead_times_known_answer`, `::test_score_binary_known_answer`, `::test_exact_lead_predictor_has_high_recall_but_poor_precision`; `validation/validate_predictor.py` checks 2, 3, 4 |
| REQ-099 | A linear learned model shall be reported as a measured lower bound on what a learned model can do here. | T, V | `tests/test_predictor.py::test_logistic_model_is_a_weak_lower_bound`; `validation/validate_predictor.py` checks 2, 3 |
| REQ-100 | The measured failure of the exact worst-case predictor to be sound on realised episodes shall be reported together with the two stated approximations that cause it, each quantified. | V | `validation/validate_predictor.py` check 3b; `simplexguard/reachability.py` module docstring |
| REQ-101 | Predictor functions shall validate their arguments and shall refuse a single-class training split. | T | `tests/test_predictor.py::test_predictor_input_validation`, `::test_fit_refuses_a_single_class_split`, `tests/test_reachability.py::test_predictor_validation` |

## 10. Interface and packaging

| # | Requirement | Method | Verified by |
|---|---|---|---|
| REQ-110 | `python -m simplexguard --help` shall exit 0 in a clean subprocess and shall print the research-grade disclaimer. | T | `tests/test_cli.py::test_help_exits_zero_in_a_clean_subprocess` |
| REQ-111 | The CLI shall expose the plant, the invariant set, the guard, an episode, the accounting, the bound sweep and the predictor benchmark. | T | `tests/test_cli.py::test_plant_subcommand`, `::test_invariant_subcommand`, `::test_guard_subcommand`, `::test_run_subcommand`, `::test_accounting_subcommand`, `::test_bound_sweep_subcommand`, `::test_predict_subcommand` |
| REQ-112 | The CLI shall exit 2 on bad input and 3 when the invariant set is empty or the recursion does not converge, so a script can tell a modelling outcome from a crash. | T | `tests/test_cli.py::test_bad_input_exits_two`, `::test_argument_validation_exits_two`, `::test_empty_invariant_set_exits_three`, `::test_recursion_cap_exits_three`, `::test_no_subcommand_is_an_error` |
| REQ-113 | The declared runtime dependencies shall be numpy, scipy, scikit-learn and joblib only; matplotlib shall be used by examples only, with the Agg backend. | I | `pyproject.toml` `[project] dependencies`; no `import matplotlib` anywhere under `src/` |
| REQ-114 | The whole pipeline shall run end to end from the public API. | T | `tests/test_integration.py::test_full_pipeline_end_to_end` |
| REQ-115 | The package shall be importable and testable from a cold clone with no install step. | I, T | `pyproject.toml` `[tool.pytest.ini_options] pythonpath = ["src"]`; the full suite runs that way |
