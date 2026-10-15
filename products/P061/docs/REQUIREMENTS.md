# Requirements — activityplan

Level 3 products carry a requirement list that the test suite and the validation
scripts can be traced to. Each requirement below names the artifact that
discharges it. A requirement with no evidence is listed as **not discharged**
rather than quietly dropped.

Identifiers are stable. `R-` requirements are functional, `V-` are validation,
`N-` are non-functional, `L-` are declared limitations that scope the rest.

---

## Functional

| ID | Requirement | Discharged by |
|---|---|---|
| R-01 | The package shall accept a declared set of activities, each with a piecewise-constant power profile, a data volume, a priority and a time window. | `activityplan.model.Activity`; `tests/test_model.py` |
| R-02 | An activity's duration shall be derived from its power profile, so that a declared duration and a declared profile cannot disagree. | `Activity.duration_s`; `tests/test_model.py::test_activity_energy_and_duration_known_answer` |
| R-03 | The package shall accept a declared resource model with battery capacity, a depth-of-discharge limit, an initial state of charge, a bus load, a solar input profile, a recorder capacity and contact windows with rates and transmitter power. | `activityplan.model.ResourceModel`; `tests/test_model.py` |
| R-04 | The solar input profile shall have an exact closed-form energy integral over any interval. | `SolarProfile.energy_wh`; `validate_energy.py` check 1 |
| R-05 | The package shall accept declared precedence and exclusion constraints and shall refuse a cyclic precedence graph. | `Problem.__post_init__`; `tests/test_model.py::test_problem_rejects_duplicates_unknown_names_and_cycles` |
| R-06 | The package shall project a declared problem onto a uniform time grid, exactly once, in a form both planners and the simulator consume. | `activityplan.timeline.RasterisedProblem`; `tests/test_timeline.py` |
| R-07 | Per-step activity energy and data volume shall sum to the declared totals. | `tests/test_timeline.py::test_rasterised_activity_energy_sums_to_the_exact_activity_energy`; `tests/test_properties.py` |
| R-08 | A horizon that is not an integer multiple of the grid step shall be refused, not rounded. | `Grid.for_horizon`; `tests/test_timeline.py::test_grid_refuses_an_off_grid_horizon_rather_than_rounding` |
| R-09 | Grid snapping shall only tighten a declared window, never loosen it, and the lost float shall be reportable. | `RasterisedProblem.window_tightening_s`; `tests/test_timeline.py::test_allowed_starts_tighten_the_window_inward` |
| R-10 | The package shall simulate the battery with a shunt at capacity and **no** clamp at the depth-of-discharge floor. | `activityplan.simulate.simulate`; `tests/test_simulate.py::test_battery_is_not_clamped_below_the_floor_or_below_zero` |
| R-11 | The package shall simulate the recorder with production then maximal drain within each step, and shall report the within-step peak as a diagnostic. | `simulate`; `tests/test_simulate.py::test_recorder_fills_then_drains_at_the_contact_rate` |
| R-12 | The package shall produce a feasibility report naming the binding constraint and giving the slack on every other declared constraint, in its physical unit and normalised by a declared scale. | `activityplan.feasibility.evaluate`; `tests/test_feasibility.py` |
| R-13 | Quantities that cannot be violated by construction shall be reported as diagnostics and shall never be named as binding. | `ConstraintSlack.binding_candidate`; `tests/test_feasibility.py::test_downlink_volume_is_a_diagnostic_not_a_constraint` |
| R-14 | The feasibility verdict shall use a tolerance relative to each constraint's own declared scale, not a strict comparison against zero. | `ConstraintSlack.satisfied`; `tests/test_feasibility.py::test_constraint_slack_normalisation_and_tolerance` |
| R-15 | **Baseline 1.** The package shall provide a priority-ordered earliest-feasible-start planner with explicit resource simulation at every candidate start. | `activityplan.greedy.plan_greedy`; `tests/test_greedy.py` |
| R-16 | The greedy planner shall consider activities in a precedence-respecting order. | `greedy.topological_order`; `tests/test_greedy.py::test_topological_order_respects_precedence_then_priority` |
| R-17 | The greedy planner shall record, for every activity it drops, the binding constraint of each refused start. | `greedy.RejectionRecord`; `tests/test_greedy.py::test_greedy_drops_an_activity_and_records_the_reason` |
| R-18 | **Baseline 2.** The package shall provide the exact optimum of the discretised problem by mixed-integer programming. | `activityplan.exact.solve_exact`; `tests/test_exact.py` |
| R-19 | The exact solver shall report its dual bound, MIP gap and node count, and shall state whether optimality was proven. | `ExactResult.proven_optimal`; `tests/test_exact.py::test_milp_is_at_least_as_good_as_greedy_on_every_shipped_instance` |
| R-20 | A plan the exact solver returns shall be feasible under the resource simulator. | `tests/test_exact.py::test_milp_plans_are_feasible_under_the_simulator`; `validate_gap.py` |
| R-21 | When the model is infeasible and no primal bound was applied, the error shall name the empty plan's binding constraint. | `solve_exact`; `tests/test_exact.py::test_milp_names_the_empty_plan_violation_when_the_declaration_is_infeasible` |
| R-22 | **AI.** The package shall provide a learned predictor of activity duration and energy with a prediction interval. | `activityplan.predictor.DurationEnergyPredictor`; `tests/test_predictor.py` |
| R-23 | The learned predictor's interval shall carry a stated coverage guarantee and shall report when the requested level is unattainable for the calibration set size. | `ConformalRegressor.level_attainable`; `tests/test_predictor.py::test_conformal_interval_is_infinite_when_the_level_is_unattainable` |
| R-24 | The learned predictor shall be benchmarked against the analytic model on held-out activities. | `validate_predictor.py` check 1; `tests/test_predictor.py::test_learned_model_beats_the_analytic_model_on_energy_held_out` |
| R-25 | A warm start derived from predictions shall be verified against the target problem by the simulator before its value is used as a bound. | `warmstart.warm_start_from_plan`; `tests/test_warmstart.py::test_an_infeasible_candidate_is_refused_as_a_bound` |
| R-26 | The package shall provide a robust planning mode that books the upper end of the prediction interval, and shall measure what it costs. | `robust.estimate_for_mode`; `validate_robust.py` |
| R-27 | The package shall provide a CLI at `python -m activityplan` covering instance listing, declaration, planning, comparison, prediction, robustness and warm start. | `activityplan.__main__`; `tests/test_cli.py`; `validate_cli.py` |
| R-28 | No library module shall write to a stream. | `activityplan.report` returns strings; `tests/test_report.py` |

## Validation

| ID | Requirement | Discharged by | Result |
|---|---|---|---|
| V-01 | Energy conservation over the timeline shall be checked against a closed-form integral of the power profile. | `validate_energy.py` check 4 | worst residual 2.274e-13 W\*h over 21 traces |
| V-02 | The closed-form solar integral shall itself be checked against an independent implementation. | `validate_energy.py` check 1 | worst relative difference 4.743e-10 against `scipy.integrate.quad` |
| V-03 | The MILP optimum shall be cross-checked against exhaustive enumeration on small instances. | `validate_exact.py` | 9 instances, 0 disagreements, worst difference 0.000e+00 |
| V-04 | The greedy-against-exact optimality gap shall be measured over a seeded instance family. | `validate_gap.py` | mean ratio 0.9511, worst 0.8500, ties 9 of 24, pooled shortfall 4.97 % |
| V-05 | Prediction-interval coverage shall be measured. | `validate_predictor.py` checks 3 and 4 | marginal: 0 of 12 rows outside the binomial band. Conditional: **0.6782 on the long tercile against a nominal 0.90** for the constant-width variant |
| V-06 | The effect of the warm start on the exact solver shall be measured and reported honestly, including a null result. | `validate_warmstart.py` | 31 of 34 cold solves have a node count of 1; **the warm start does not help on those**, and reduces 263 → 1, 2064 → 420 and 46 → 1 on the three that branch |
| V-07 | The effect of planning under predicted-duration uncertainty shall be measured on realised outcomes. | `validate_robust.py` | **ml-robust 0.4167 feasible against ml-point 0.4722** — the robust plan is worse, for a measured structural reason |
| V-08 | Every documented CLI invocation shall be executed and recorded. | `validate_cli.py` | 14 invocations plus `--help`, 0 failures, 0 forbidden words |
| V-09 | Every counterexample a property test produces shall be pinned as a named regression test with literal inputs. | `tests/test_regressions.py` | 2 Hypothesis counterexamples plus the downlink-volume defect, pinned |
| V-10 | Test counts shall be read from junit XML and cross-checked against a count of `testcase` elements. | `validation/VALIDATION.md` | 217 tests, 0 failures, 0 errors, 0 skipped; 217 `testcase` elements counted |

## Non-functional

| ID | Requirement | Discharged by |
|---|---|---|
| N-01 | Every run shall finish inside three minutes on two CPU cores. | Compute-budget table in `README.md`; slowest run 92.7 s |
| N-02 | No GPU and no PyTorch. | `pyproject.toml` dependencies; both forests use `n_jobs = 1` |
| N-03 | Matplotlib shall use the Agg backend and shall never call `show()`. | `examples/_common.py` |
| N-04 | The package shall be ruff-clean at line length 100. | `ruff check src/ tests/ examples/ validation/` passes under 0.16.8 and 0.17.0 |
| N-05 | Public APIs shall carry type hints and docstrings stating units. | module docstrings throughout; API table in `README.md` |
| N-06 | Every dependency shall be bounded on both sides at a version actually executed. | `pyproject.toml`; dependency table in `validation/VALIDATION.md` |
| N-07 | Every equation shall carry its source, units, assumptions and validity range. | module docstrings in `model.py`, `simulate.py`, `exact.py`, `features.py` |
| N-08 | No output of any kind shall claim flight-safety, certification or production readiness. | `tests/test_report.py::test_no_forbidden_vocabulary_in_any_rendered_output`; `tests/test_cli.py`; `validate_cli.py` |

## Declared limitations that scope the above

| ID | Limitation |
|---|---|
| L-01 | Start times are restricted to grid points. The optimum reported is that of the discretised problem and is a lower bound on the continuous-time optimum. |
| L-02 | Activity occupancy rounds up and windows tighten inward. Both are conservative and both make some legal continuous plans unreachable. |
| L-03 | The final partially occupied step of an off-grid activity carries a reduced average power. Energy is exact; instantaneous power is not constrained, and **no peak-power constraint is implemented**. |
| L-04 | The MILP requires `charge_efficiency == 1` and raises otherwise. The efficiency kink at zero surplus is piecewise linear and is not represented. |
| L-05 | `scipy.optimize.milp` exposes no MIP start, so the warm start is a primal-bound cut and supplies no integer assignment. |
| L-06 | The battery has no voltage, resistance, temperature, self-discharge or ageing model. The recorder has no file structure, packet overhead or retransmission. |
| L-07 | Contact rates are declared, not computed. Elevation-dependent rate variation is not modelled, which overstates a low-elevation pass. |
| L-08 | The learned model's advantage over the analytic model is a direct consequence of a declared misspecification in the synthetic generator, and the learned model is given the analytic prediction as a feature. |
| L-09 | The conformal interval has no conditional coverage guarantee and does not attain conditional coverage here. |
| L-10 | The instance family is a declared experimental design. Every measured rate, gap and node count is a statement about that population only. |
| L-11 | No value anywhere was identified from a spacecraft, instrument or mission. |

## Not discharged

| ID | Requirement | Why not |
|---|---|---|
| ND-01 | Dependency ranges executed at **both** ends. | Only `ruff` was executed at both ends of its declared range, because the container happens to carry two installations. For numpy, scipy, scikit-learn, pytest, hypothesis and matplotlib the declared floor **is** the executed version and the ceiling is the next minor, so the range contains one measured minor series; **no other version was installed or tested.** Recorded in `validation/VALIDATION.md` and in the README's Limitations rather than claimed as discharged. |
| ND-02 | A peak-power constraint. | Would be wrong at the final partially occupied step of an off-grid activity without a finer treatment of that step. Listed in the README's roadmap. |
| ND-03 | A rate-aware robust planning mode. | The measurement in `validate_robust.py` shows that bounding duration from above is the wrong conservatism for a declared data volume. The fix is identified but not implemented. Listed in the README's roadmap. |
| ND-04 | Any hardware measurement. | There is none in this repository and nothing here is hardware-pending. |
