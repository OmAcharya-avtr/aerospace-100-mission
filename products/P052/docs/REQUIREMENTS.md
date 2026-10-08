# falsifyloop — requirements

Validation level 3. Every requirement below names the test or validation script
that exercises it. A requirement with no such name is not a requirement; it is a
wish, and none appear here.

**Conventions.** `tests/...` are pytest tests, run by `python -m pytest tests/ -q`
from the repository root. `validation/...` are standalone scripts whose raw
stdout is committed beside them. Verification method is one of **T** (automated
test), **V** (validation script with committed output), **A** (analysis, stated
in a module docstring and checked by a T or a V), **I** (inspection).

**Scope of every requirement.** All of them are conditional on the simulator,
the search box and the requirement the caller supplies. Nothing here is a
statement about any physical system, and nothing here is a verification claim:
**falsification is one-sided, and finding no violation is not evidence of
correctness.**

---

## 1. Traces

| # | Requirement | Method | Verified by |
|---|---|---|---|
| REQ-001 | A trace shall be a finite, uniformly sampled record of named real-valued signals over a common time vector in seconds. | T | `tests/test_traces.py::test_trace_known_answer_properties`, `::test_multiple_signals_are_sorted_in_names` |
| REQ-002 | A non-uniform time vector shall be rejected, because the time-bounded operators convert a window in seconds to an integer sample offset and that conversion is exact only on a uniform grid. | T | `tests/test_traces.py::test_trace_rejects_bad_input[times4-signals4-uniformly spaced]` |
| REQ-003 | A trace shall have at least two samples, strictly increasing finite times, and no non-finite signal value; `nan` in particular shall be rejected because it would break the sign agreement of REQ-020. | T | `tests/test_traces.py::test_trace_rejects_bad_input` (10 parametrised cases) |
| REQ-004 | A missing signal shall raise `KeyError` naming the signals the trace does have. | T | `tests/test_traces.py::test_missing_signal_raises_keyerror_naming_what_exists` |
| REQ-005 | The uniformity check shall tolerate the floating-point spacing that `numpy.arange` produces at the shipped sample interval, rather than demanding bit-equal steps. | T | `tests/test_traces.py::test_uniformity_tolerance_accepts_floating_point_arange` |

## 2. The requirement language and its two semantics

| # | Requirement | Method | Verified by |
|---|---|---|---|
| REQ-010 | The language shall provide terms `Signal`, `Difference` and `Abs`, and formulas `Predicate` (`<=`, `>=`), `And`, `Or`, bounded `Always` and bounded `Eventually`, and nothing else. | T, I | `tests/test_requirements.py` known-answer tests for each; `src/falsifyloop/requirements.py` module docstring |
| REQ-011 | `Difference` shall be the backward difference with a one-sided forward difference at the left edge, in signal units per second. | T, V | `tests/test_requirements.py::test_difference_term_known_answer`; `validation/validate_semantics.py` check 1 |
| REQ-012 | Each operator shall compute the robustness stated in the module docstring, to hand-calculated known answers. | T, V | `tests/test_requirements.py` (10 known-answer tests); `validation/validate_semantics.py` check 1 (10 cases) |
| REQ-013 | A time bound that is not an integer multiple of the trace sample interval shall raise `ValueError` rather than be silently rounded. | T, V | `tests/test_requirements.py::test_time_bound_off_the_sample_grid_is_rejected`, `::test_time_bound_within_tolerance_is_accepted`; `validation/validate_semantics.py` check 5 |
| REQ-014 | An empty time window shall score `+inf`/`True` under `Always` and `-inf`/`False` under `Eventually`, these being the identities of the reductions. | T, V | `tests/test_requirements.py::test_empty_window_conventions`; `validation/validate_semantics.py` check 4 |
| REQ-015 | `check_horizon` shall raise when a trace is too short for a formula's windows, so that a vacuously satisfied verdict cannot be reported as a real one. | T | `tests/test_requirements.py::test_check_horizon_rejects_a_short_trace`, `::test_nested_eventually_always_known_answer` |
| REQ-016 | The language shall contain **no negation operator**, so that every formula is in negation normal form and the sign agreement of REQ-020 is exact at the boundary. | A, I | `src/falsifyloop/requirements.py` module docstring; enforced by the absence of a `Not` class and by REQ-020 holding at zero robustness |
| REQ-017 | `scale` shall be strictly positive and shall divide the robustness, so a normalisation can change the magnitude and never the sign. | T, V | `tests/test_requirements.py::test_constructor_validation`; `tests/test_sign_agreement.py::test_positive_scaling_of_every_predicate_cannot_change_the_verdict`; `validation/validate_semantics.py` check 6 |
| REQ-018 | Every constructor shall reject malformed input with `ValueError` or `TypeError` naming the parameter. | T | `tests/test_requirements.py::test_constructor_validation` (11 cases), `::test_type_validation` (4 cases), `::test_top_level_functions_type_check_their_arguments` |
| REQ-019 | A formula evaluated against a trace lacking one of its signals shall raise `KeyError` listing both sets. | T | `tests/test_requirements.py::test_missing_signal_in_formula_raises_keyerror` |

## 3. The sign agreement — the core claim

| # | Requirement | Method | Verified by |
|---|---|---|---|
| REQ-020 | **Robustness shall be negative exactly when the requirement is violated**: `robustness(phi, w) >= 0` if and only if `satisfies(phi, w)`, with no tolerance band, where the two sides are computed by two independent implementations. | T, V | `tests/test_sign_agreement.py::test_robustness_is_negative_exactly_when_the_requirement_is_violated` (Hypothesis, 400 examples); `validation/validate_semantics.py` checks 2 and 3 |
| REQ-021 | The agreement shall hold **elementwise**, not only at `t = 0`. | T | `tests/test_sign_agreement.py::test_sign_agreement_holds_elementwise_not_just_at_time_zero` |
| REQ-022 | The agreement shall hold over an exhaustive enumeration of small traces and formulas, and the enumeration shall be shown to visit the zero-robustness and infinite-robustness cases. | V | `validation/validate_semantics.py` check 2: 44694 formula-trace pairs, 32070 samples at exactly zero robustness, 16848 at infinite robustness, 0 disagreements |
| REQ-023 | The agreement shall hold on real simulator traces from every shipped instance. | T, V | `tests/test_instances.py::test_evaluate_sign_agrees_with_the_boolean_semantics_on_every_instance`; `validation/validate_semantics.py` check 3: 2400 pairs, 156 violating, 0 disagreements |
| REQ-024 | Robustness shall never be `nan`. | T | `tests/test_sign_agreement.py::test_robustness_is_never_nan` |

## 4. The simulator

| # | Requirement | Method | Verified by |
|---|---|---|---|
| REQ-030 | The simulator shall implement the declared second-order attitude loop with a first-order actuator, a slew-rate limit and a deflection limit, by explicit Euler, with every equation and unit stated. | A, I | `src/falsifyloop/systems.py` module docstring |
| REQ-031 | No parameter value in the simulator shall be presented as identified from any aircraft. | I | `src/falsifyloop/systems.py` module docstring; `validation/validate_simulator.py` header; README Limitations |
| REQ-032 | In the unsaturated regime the scheme shall agree with the exact zero-order-hold matrix-exponential solution, and shall show first-order global convergence. | T, V | `tests/test_systems.py::test_euler_matches_the_exact_zoh_solution_in_the_unsaturated_regime`; `validation/validate_simulator.py` check 1: error ratios 2.034, 2.017, 2.008, 2.004 as dt halves; 0.009032 deg at the shipped dt |
| REQ-033 | In the saturated regime the scheme shall self-converge as the step is refined, and the discretisation budget shall be reported rather than assumed small. | T, V | `tests/test_systems.py::test_euler_self_converges_as_dt_shrinks`; `validation/validate_simulator.py` check 2: 0.429770 deg at the shipped dt against a dt/40 reference |
| REQ-034 | The declared actuator deflection and slew-rate limits shall hold over the whole search box. | T, V | `tests/test_systems.py::test_deflection_and_rate_limits_are_respected`; `validation/validate_simulator.py` check 3: 4000 draws, worst `delta` 19.614588 deg against 20 deg, worst slew 120.000000 deg/s against 120 deg/s |
| REQ-035 | Simulation shall be bit-for-bit deterministic in its inputs. | T, V | `tests/test_systems.py::test_simulation_is_deterministic`; `validation/validate_simulator.py` check 4: 500 draws, 0 non-reproducible |
| REQ-036 | The fraction of counterexamples that do not survive a four-times-finer step shall be measured and published, not assumed zero. | V | `validation/validate_simulator.py` check 5: 22 of 304 violating draws flip, worst flipped robustness 0.135888 |
| REQ-037 | `simulate` shall reject a non-positive `dt`, a `dt` above the effective actuator time constant, and a horizon shorter than two steps. | T | `tests/test_systems.py::test_simulate_rejects_bad_timing`, `::test_simulate_type_validation` |
| REQ-038 | `LoopParameters` and `LoopInput` shall validate every field with an actionable message. | T | `tests/test_systems.py::test_loop_input_validation` (6 cases), `::test_loop_parameters_validation` (8 cases), `::test_loop_input_from_array_rejects_the_wrong_length` |

## 5. The benchmark suite

| # | Requirement | Method | Verified by |
|---|---|---|---|
| REQ-040 | The suite shall contain **at least six** seeded instances sharing one declared search box. | T | `tests/test_instances.py::test_suite_has_at_least_six_instances`, `::test_every_instance_shares_the_declared_box` — the suite ships eight |
| REQ-041 | Every instance shall declare a difficulty tier, and the tier shall be checked against an independently measured violation probability. | T, V | `tests/test_instances.py::test_every_tier_label_is_declared`, `::test_suite_spans_at_least_three_tiers`; `validation/validate_difficulty.py`: 0 of 8 instances outside their declared tier band |
| REQ-042 | Each instance's uniform-violation probability shall be reported with an exact binomial interval from a seed different from the one the bound was chosen with. | V | `validation/validate_difficulty.py`: 15000 draws at seed 52052 against a pilot at seed 13; measured p from 0.001267 to 0.321933 |
| REQ-043 | Every instance's requirement horizon shall fit inside its simulated trace, checked at construction. | T | `tests/test_instances.py::test_every_requirement_horizon_fits_inside_the_simulated_trace`, `::test_instance_rejects_a_requirement_longer_than_its_horizon` |
| REQ-044 | Every instance shall be falsifiable by the baseline given a stated budget, so that no column of the benchmark is vacuous. | T | `tests/test_integration.py::test_every_instance_is_falsifiable_by_the_baseline_given_enough_draws` |
| REQ-045 | One simulation shall mean exactly one call to `Instance.evaluate`, which is the unit the sample-efficiency curve counts. | T, A | `tests/test_instances.py::test_evaluate_agrees_with_robustness_of_the_simulated_trace`; `src/falsifyloop/search.py` `_Run.evaluate` is the only counter |

## 6. The search strategies

| # | Requirement | Method | Verified by |
|---|---|---|---|
| REQ-050 | **Uniform random shall be the baseline** and shall be the first registered strategy. | T, I | `tests/test_search.py::test_the_baseline_is_uniform_random_and_is_registered_first`; `src/falsifyloop/search.py` ordering |
| REQ-051 | Five strategies shall be provided: uniform random, Latin hypercube, simulated annealing, cross-entropy, and a surrogate-guided search. | T | `tests/test_search.py::test_registry_has_the_five_required_strategies` |
| REQ-052 | Every strategy shall spend at most its budget, counting every evaluation. | T | `tests/test_search.py::test_strategy_respects_its_budget` (5 cases) |
| REQ-053 | Every strategy shall be exactly reproducible from `(instance, budget, seed)` and shall differ between seeds. | T | `tests/test_search.py::test_strategy_is_reproducible_from_its_seed`, `::test_different_seeds_give_different_runs` (5 cases each) |
| REQ-054 | Every strategy shall stop at its first violation, under the same rule. | T | `tests/test_search.py::test_strategy_stops_at_the_first_violation` (5 cases) |
| REQ-055 | Every proposal shall lie inside the declared box. | T | `tests/test_search.py::test_best_vector_lies_inside_the_declared_box` (5 cases); `_reflect` known answers in `::test_reflect_known_answers` |
| REQ-056 | The analytic uniform-random curve `1 - (1-p)^n` shall be available and shall match a hand calculation. | T | `tests/test_search.py::test_analytic_random_curve_known_answer`, `::test_analytic_random_curve_is_monotone_and_bounded` |
| REQ-057 | The surrogate strategy's warm start shall be exactly the Latin hypercube prefix, so that the learned part's share of the budget is auditable. | T, V | `tests/test_search.py::test_surrogate_guided_warm_start_is_the_latin_hypercube_prefix`, `::test_surrogate_guided_uses_the_surrogate_after_the_warm_start`; `validation/validate_benchmark.py` section 6 |
| REQ-058 | Every strategy parameter shall be validated with an actionable message. | T | `tests/test_search.py::test_simulated_annealing_validation`, `::test_cross_entropy_validation`, `::test_surrogate_guided_validation`, `::test_zero_budget_is_rejected` |

## 7. The learned component

| # | Requirement | Method | Verified by |
|---|---|---|---|
| REQ-060 | The classical baseline shall be implemented and benchmarked **before** the learned model, on the same instances, budget and seeds. | I, V | `src/falsifyloop/search.py` function order and module docstring; `validation/validate_benchmark.py` runs the full grid at one `base_seed` |
| REQ-061 | The surrogate shall expose an uncertainty or confidence output alongside every point prediction. | T | `tests/test_surrogate.py::test_predict_shapes_and_nonnegative_spread`, `::test_spread_is_larger_away_from_the_training_data` |
| REQ-062 | The empirical coverage of that output shall be measured on an independently seeded held-out split and published as measured. | V | `validation/validate_surrogate.py` check 2: nominal-95 coverage 0.9356 mean, range 0.8950–0.9650; nominal-68 coverage 0.70–0.80 against a Gaussian 0.683 |
| REQ-063 | The usefulness of the spread as a ranking signal shall be measured, since that is the only property the acquisition uses. | V | `validation/validate_surrogate.py` check 3: mean Spearman correlation 0.4853 across the suite |
| REQ-064 | The surrogate's held-out accuracy shall be compared against a trivial predictor. | V | `validation/validate_surrogate.py` check 1: RMSE ratio 1.523–2.425 over the trivial mean predictor |
| REQ-065 | The surrogate's compute cost shall be reported against the price of a simulation. | V | `validation/validate_surrogate.py` check 4 |
| REQ-066 | The forest shall be fit and queried with `n_jobs = 1`, on the strength of a measurement and not an assumption. | V, I | `validation/validate_surrogate.py` check 5; `src/falsifyloop/surrogate.py` hard-codes it and does not expose it |
| REQ-067 | The surrogate shall refuse to fit through an infinite robustness value. | T | `tests/test_surrogate.py::test_fit_validation` |
| REQ-068 | A fit with a fixed seed and fixed data shall be bit-identical. | T | `tests/test_surrogate.py::test_fit_with_the_same_seed_is_bit_identical` |

## 8. Curves, bands and the report

| # | Requirement | Method | Verified by |
|---|---|---|---|
| REQ-070 | The deliverable shall be a sample-efficiency curve: probability of having found a violation against simulation count. | T, V | `tests/test_curves.py::test_efficiency_curve_known_answer`; `validation/validate_benchmark.py` section 3; `examples/sample_efficiency_curves.py` |
| REQ-071 | Each curve shall carry a bootstrap confidence band whose resampling unit is one run, and the band shall bracket the curve. | T, V | `tests/test_curves.py::test_bootstrap_band_brackets_the_curve_and_is_reproducible`, `::test_bootstrap_band_is_non_decreasing`; `validation/validate_benchmark.py` section 3 |
| REQ-072 | Runs that found nothing shall be treated as right-censored and never as late successes; the summary statistic shall be a median reported as `> budget` when censored. | T | `tests/test_curves.py::test_median_first_violation_known_answers`, `::test_median_returns_none_when_nothing_was_found` |
| REQ-073 | The aggregate shall be the unweighted mean of per-instance curves, with a stratified bootstrap. | T | `tests/test_curves.py::test_aggregate_curve_known_answer`, `::test_aggregate_weights_every_instance_equally`, `::test_aggregate_band_brackets_the_aggregate_curve` |
| REQ-074 | **Per-instance results shall be reported before the aggregate**, and the instances where the baseline wins shall be named. | T, V | `tests/test_cli.py::test_benchmark_prints_per_instance_before_aggregate`; `tests/test_benchmark.py::test_baseline_wins_names_instances_and_the_baseline_never_beats_itself`; `validation/validate_benchmark.py` sections 1 and 2 |
| REQ-075 | Whether each win or loss is statistically distinguishable at the repeat count used shall be reported, not left implied. | V | `validation/validate_benchmark.py` section 2b: 14 of 32 comparisons undecided at 30 seeds |
| REQ-076 | The empirical baseline curve shall be checked against the closed form `1 - (1-p)^n` as a check on the whole measurement chain. | V | `validation/validate_benchmark.py` section 4: 1 of 40 interval comparisons disagree |
| REQ-077 | Instance difficulty shall use an exact binomial interval, because the hardest instances have `p` of order `1e-3`. | T | `tests/test_curves.py::test_clopper_pearson_known_answers`, `::test_clopper_pearson_brackets_the_point_estimate`, `::test_clopper_pearson_narrows_with_more_trials` |

## 9. One-sidedness, vocabulary and the CLI

| # | Requirement | Method | Verified by |
|---|---|---|---|
| REQ-080 | **Nothing this package renders or prints shall read as a verdict of correctness.** | T, V | `tests/test_vocabulary.py` (9 rendered strings + 7 CLI invocations + `--help`); `validation/validate_cli.py`: 0 failures over 14 invocations |
| REQ-081 | A search that found nothing shall say so, and shall restate that falsification is one-sided. | T | `tests/test_report.py::test_render_no_violation_states_the_one_sidedness`; `tests/test_cli.py::test_falsify_reports_nothing_found_without_implying_correctness` |
| REQ-082 | There shall be no `print` in library code; the CLI shall be the only writer. | T | `tests/test_vocabulary.py::test_library_source_has_no_print_statement` |
| REQ-083 | `python -m falsifyloop --help` shall exit 0. | T, V | `tests/test_cli.py::test_module_help_exits_zero_as_a_subprocess`; `validation/validate_cli.py` subprocess check |
| REQ-084 | Exit status 0 shall mean the command ran, never that anything was or was not found; `--exit-on-violation` shall make a found violation exit 1 and shall leave the absence of one at 0. | T, V | `tests/test_cli.py::test_exit_on_violation_changes_only_the_found_case`; `validation/validate_cli.py` exit-status table |
| REQ-085 | Usage and validation errors shall exit 2. | T, V | `tests/test_cli.py` (6 cases); `validation/validate_cli.py` rows 11–14 |
| REQ-086 | The `SearchResult` object shall expose no Boolean that could be read as a verdict about the requirement. | T | `tests/test_vocabulary.py::test_search_result_never_reports_absence_as_a_positive_verdict` |

## 10. Reproducibility and compute

| # | Requirement | Method | Verified by |
|---|---|---|---|
| REQ-090 | The benchmark grid shall be deterministic given its configuration. | T | `tests/test_benchmark.py::test_run_benchmark_is_deterministic` |
| REQ-091 | A pinned seeded regression shall exist so that a change in any layer surfaces as a failing test rather than a quietly different figure. | T | `tests/test_integration.py::test_benchmark_regression_pinned_seeded_numbers` |
| REQ-092 | Every number in README.md and validation/VALIDATION.md shall come from a committed script's committed output. | I | `validation/VALIDATION.md` "Reproducing every number" |
| REQ-093 | The compute budget and the container's measured core count shall be stated, and no timing shall be presented as a hardware characteristic. | V, I | `validation/validate_benchmark.py` section 8 and `validate_difficulty.py` COMPUTE; README "Compute budget" |
| REQ-094 | No hardware-pending benchmark harness shall be added, and the product shall not be labelled Level 4. | I | repository contains no `benchmark/` directory; README header states Level 3 |
