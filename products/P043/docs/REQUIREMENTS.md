# photoncount — requirements

Validation level 3, hardware-pending. Every requirement below names the test or
validation script that exercises it. A requirement with no such name is not a
requirement; it is a wish, and none appear here.

Conventions. `tests/...` are pytest tests, run by `python -m pytest tests/ -q`.
`validation/...` are standalone scripts whose raw output is committed beside
them. Verification method is one of **T** (automated test), **V** (validation
script with committed output), **A** (analysis, stated in a docstring and
checked by T or V), **I** (inspection).

---

## 1. Detection statistics

| # | Requirement | Method | Verified by |
|---|---|---|---|
| R-1.1 | The mean detected count in a slot shall be computed from optical power, wavelength, slot length and detection efficiency as `eta P T lambda / (h c)`, with SI defining constants. | T | `tests/test_poisson.py::test_photon_rate_known_answer`, `::test_mean_counts_scales_with_efficiency_and_time` |
| R-1.2 | Signal, background and dark contributions shall combine additively as Poisson rates. | T | `tests/test_poisson.py::test_invalid_inputs_raise_value_error`, `tests/test_ppm.py::test_slot_means_known_answer` |
| R-1.3 | Threshold detection and missed-detection probabilities shall be exact for a Poisson count and shall sum to 1. | T | `tests/test_poisson.py::test_threshold_detection_probability_known_answers`, `::test_missed_and_detection_sum_to_one`, `::test_detection_probability_agrees_with_scipy` |
| R-1.4 | A two-sided interval for a Poisson mean from a single count shall be the exact (Garwood) interval, not a normal approximation. | T | `tests/test_poisson.py::test_exact_count_interval_matches_published_garwood_values`, `::test_exact_count_interval_zero_count_has_zero_lower_bound`, `::test_exact_interval_brackets_the_count` |
| R-1.5 | The Fano factor shall be reported as `Var/mean` and shall be 1 for Poisson samples within sampling error. | T | `tests/test_poisson.py::test_fano_factor_is_one_for_poisson_samples` |

## 2. Webb distribution (APD excess-noise counting statistics)

| # | Requirement | Method | Verified by |
|---|---|---|---|
| R-2.1 | The Webb density shall be implemented in the gain-normalised variable with parameters `(m, F, G)` and shall be zero outside its support `y > m - mF/(F-1)`. | T | `tests/test_webb.py::test_pdf_is_zero_below_the_support` |
| R-2.2 | The first three moments shall be `m`, `mF` and `3 m F (F-1)`, and these closed forms shall be verified against quadrature of the density, not assumed. | T, V | `tests/test_webb.py::test_closed_form_moments_match_numerical_quadrature`, `::test_skewness_known_answer`; `validation/validate_webb_limits.py` check 1 |
| R-2.3 | At `F = 1` the density shall equal `N(m, m)` exactly to floating point, and the distance to `N(m, mF)` shall fall monotonically as `F -> 1`. | T, V | `tests/test_webb.py::test_f_equal_one_is_exactly_gaussian`, `::test_gaussian_limit_is_approached_as_excess_noise_falls`; `validation/validate_webb_limits.py` check 2 |
| R-2.4 | The Poisson limit shall be stated and measured as total variation falling as `m^(-1/2)`, and shall not be claimed to be tighter. | T, V | `tests/test_webb.py::test_poisson_limit_total_variation_falls_as_inverse_sqrt_mean`; `validation/validate_webb_limits.py` check 3 |
| R-2.5 | Integer-binned probabilities shall be normalised, non-negative, and shall absorb the negative-`y` mass of the continuous approximation into bin 0. | T | `tests/test_webb.py::test_binned_pmf_is_normalised_and_non_negative` |
| R-2.6 | Sampling shall be deterministic for a given generator state and shall reproduce the closed-form moments. | T, V | `tests/test_webb.py::test_sampling_reproduces_closed_form_moments`, `::test_sampling_is_deterministic_for_a_seed`; `validation/validate_webb_limits.py` check 4 |
| R-2.7 | The excess noise factor shall satisfy `F = kG + (2 - 1/G)(1-k)`, with `F = G` at `k = 1` and `F -> 2` at `k = 0`. | T, V | `tests/test_webb.py::test_excess_noise_factor_limits`; `validation/validate_webb_limits.py` check 5 |

## 3. PPM slot statistics

| # | Requirement | Method | Verified by |
|---|---|---|---|
| R-3.1 | Slot counts shall be independent Poisson with mean `n_0 + n_s` in the signalled slot and `n_0` elsewhere. | T, V | `tests/test_ppm.py::test_sample_counts_matches_the_poisson_means` |
| R-3.2 | The hard-decision symbol error probability shall be computed exactly by summation with uniform random tie-breaking, and the truncation shall be reported as a Poisson tail mass. | T, V | `tests/test_ppm.py::test_exact_error_probability_is_a_probability`; `validation/validate_ppm.py` check 6 |
| R-3.3 | With no background the exact result shall equal `exp(-n_s)(M-1)/M` to within 1e-12. | T, V | `tests/test_ppm.py::test_background_free_symbol_error_known_answer`, `::test_background_free_matches_closed_form`; `validation/validate_ppm.py` check 1 |
| R-3.4 | With background the exact result shall agree with Monte Carlo within 4 binomial standard errors. | T, V | `tests/test_ppm.py::test_exact_error_probability_agrees_with_monte_carlo`; `validation/validate_ppm.py` check 2 |
| R-3.5 | The erasure probability shall be `exp(-(n_s + M n_0))`, reducing to `exp(-n_s)` for a background-free channel. | T | `tests/test_ppm.py::test_erasure_probability_known_answer`, `::test_erasure_channel_probabilities_requires_background_free` |
| R-3.6 | The per-slot soft metric shall be `k_i ln(1 + n_s/n_0)` and the symbol posterior shall equal the full Poisson product likelihood. | T, V | `tests/test_ppm.py::test_soft_metric_is_linear_in_the_count`, `::test_slot_metric_scale_known_answer`; `validation/validate_ppm.py` check 3 |
| R-3.7 | The maximum-likelihood decision shall be `argmax_i k_i`, with ties broken uniformly at random. | T, V | `tests/test_ppm.py::test_hard_decision_is_argmax_and_breaks_ties_uniformly`, `::test_posterior_normalises_and_peaks_at_the_largest_count`; `validation/validate_ppm.py` check 4 |
| R-3.8 | Bit log-likelihood ratios shall follow a natural-binary mapping, be positive for bit 0, reproduce the symbol posterior marginals, and be exactly zero for an all-empty symbol. | T, V | `tests/test_ppm.py::test_bit_llrs_sign_convention_and_shape`, `::test_bit_llrs_are_zero_on_an_empty_symbol`; `validation/validate_ppm.py` check 5 |
| R-3.9 | A background-free channel shall refuse to produce a finite soft-metric scale, and shall be described by the erasure-channel functions instead. | T | `tests/test_ppm.py::test_slot_metric_scale_diverges_without_background` |

## 4. Detector non-idealities

| # | Requirement | Method | Verified by |
|---|---|---|---|
| R-4.1 | The non-paralyzable relation `m = n/(1 + n tau)` and its inverse shall be exact to 1e-12 over five decades of `n tau`. | T, V | `tests/test_deadtime.py::test_nonparalyzable_known_answer`, `::test_nonparalyzable_round_trip`; `validation/validate_deadtime.py` checks 1, 7 |
| R-4.2 | The paralyzable relation `m = n exp(-n tau)` shall be implemented, and its maximum at `n = 1/tau`, `m = 1/(e tau)`, shall be verified against a dense sweep and a finite-difference derivative. | T, V | `tests/test_deadtime.py::test_paralyzable_known_answer`, `::test_paralyzable_maximum_known_answer`, `::test_paralyzable_forward_map_is_non_monotonic`; `validation/validate_deadtime.py` check 2 |
| R-4.3 | The paralyzable inverse shall require an explicit branch, shall return both roots on demand, and shall refuse an observed rate above the maximum. | T, V | `tests/test_deadtime.py::test_paralyzable_inverse_is_two_valued`, `::test_paralyzable_true_requires_an_explicit_branch`, `::test_paralyzable_inverse_rejects_unreachable_rate`; `validation/validate_deadtime.py` check 3 |
| R-4.4 | The conditioning of the paralyzable inverse shall be reported, showing the amplification of a relative error diverging at the maximum. | V | `validation/validate_deadtime.py` check 4 |
| R-4.5 | Loss and live-time fractions shall be distinct quantities per model, with `1 - m tau` for non-paralyzable and `exp(-n tau)` for paralyzable live time. | T, V | `tests/test_deadtime.py::test_loss_fractions_known_answers`, `::test_live_time_fractions_differ_by_model`; `validation/validate_deadtime.py` check 5 |
| R-4.6 | Afterpulsing shall be modelled as a Poisson cluster process in two stated variants, with closed-form cluster moments and Fano factors verified by direct sampling. | T, V | `tests/test_afterpulse.py::test_cascading_moments_known_answers`, `::test_first_order_moments_known_answers`, `::test_fano_factor_known_answers`; `validation/validate_afterpulse.py` checks 1-3 |
| R-4.7 | The afterpulse rate relations shall be exactly invertible in both variants. | T, V | `tests/test_afterpulse.py::test_forward_and_inverse_are_exact_inverses`; `validation/validate_afterpulse.py` check 4 |
| R-4.8 | The effective afterpulse probability behind a dead time shall be `p exp(-tau/t_ap)`, verified against the exponential survival fraction. | T, V | `tests/test_afterpulse.py::test_effective_probability_known_answer`, `::test_effective_probability_limits`; `validation/validate_afterpulse.py` check 5, `validation/validate_simulator.py` check 6 |
| R-4.9 | Dead time shall push the Fano factor below 1 and afterpulsing above 1, and the simulator shall reproduce both signs. | T, V | `tests/test_simulate.py::test_fano_factor_below_one_with_dead_time_and_above_one_with_afterpulsing`; `validation/validate_simulator.py` check 5 |

## 5. Event-level simulator

| # | Requirement | Method | Verified by |
|---|---|---|---|
| R-5.1 | With no dead time and no afterpulsing the simulator shall reproduce the incident rate within Poisson sampling error. | T, V | `tests/test_simulate.py::test_ideal_counter_reproduces_the_incident_rate`; `validation/validate_simulator.py` check 1 |
| R-5.2 | With afterpulsing disabled the simulator shall reproduce both dead-time rate laws over `n tau` from 0.01 to 3, including the fall past the paralyzable maximum. | T, V | `tests/test_simulate.py::test_nonparalyzable_rate_equation_recovered`, `::test_paralyzable_rate_equation_recovered`, `::test_paralyzable_observed_rate_falls_past_the_maximum`; `validation/validate_simulator.py` checks 2, 3 |
| R-5.3 | With dead time disabled the simulator shall reproduce the cascading afterpulse rate law over `p` from 0.02 to 0.4. | T, V | `tests/test_simulate.py::test_afterpulsing_rate_equation_recovered_without_dead_time`; `validation/validate_simulator.py` check 4 |
| R-5.4 | Registered timestamps shall be strictly increasing and shall respect the dead time. | T | `tests/test_simulate.py::test_timestamps_are_sorted_and_respect_the_dead_time` |
| R-5.5 | Counts shall be conserved: registered = primary-registered + afterpulse-registered, and primary events = primary-registered + primary-lost. | T | `tests/test_simulate.py::test_counts_are_conserved` |
| R-5.6 | The simulator shall be deterministic for a given generator state. | T | `tests/test_simulate.py::test_determinism_for_a_seed` |
| R-5.7 | The composition of dead time and afterpulsing shall differ measurably from the product of their separate corrections, and that difference shall be quantified. | V | `validation/validate_simulator.py` check 7 |
| R-5.8 | A run larger than 5e6 expected events shall be refused rather than attempted. | T | `tests/test_simulate.py::test_oversized_run_is_refused` |

## 6. Hardware abstraction layer

| # | Requirement | Method | Verified by |
|---|---|---|---|
| R-6.1 | One abstract backend interface shall define the whole contract, and every backend shall expose the same description keys without requiring hardware. | T, V | `tests/test_hal_contract.py::test_describe_returns_every_required_key`, `::test_describe_does_not_open_the_backend`; `validation/validate_hal_contract.py` check 1 |
| R-6.2 | A simulated backend shall implement the contract fully and deterministically. | T | `tests/test_hal_contract.py::test_simulated_backend_is_deterministic`, `::test_simulated_backend_self_test_passes` |
| R-6.3 | A device backend shall carry the documented contract and shall raise `NotImplementedError`, naming what is missing, for every operation requiring hardware. | T, V | `tests/test_hal_contract.py::test_hardware_operations_refuse_only_when_hardware_is_required`, `::test_device_backend_never_returns_counts`; `validation/validate_hal_contract.py` check 1 |
| R-6.4 | One contract test suite shall be parameterised over both backends. | T, I | `tests/test_hal_contract.py` (the `backend` fixture) |
| R-6.5 | An invalid request shall raise `ValueError` before any hardware is touched, on every backend. | T, V | `tests/test_hal_contract.py::test_bad_request_raises_value_error_not_notimplemented`, `::test_bad_mode_raises_value_error_not_notimplemented`; `validation/validate_hal_contract.py` check 1 |
| R-6.6 | Simulation mode shall run the full chain against the models; dry-run mode shall run the real command path with real timing and discard every count. | T, V | `tests/test_hal_contract.py::test_dry_run_discards_counts_but_keeps_real_timing`; `validation/validate_hal_contract.py` checks 2, 3 |
| R-6.7 | No simulated or discarded result shall report itself as a measurement. | T, V | `tests/test_hal_contract.py::test_is_measurement_is_false_for_simulated_counts`; `validation/validate_hal_contract.py` check 2 |
| R-6.8 | `close` shall be idempotent and safe before or after a failed `open`, and the backend shall be a context manager. | T | `tests/test_hal_contract.py::test_close_is_idempotent_and_safe_before_open`, `::test_context_manager_closes` |

## 7. Deployment, capture and recovery

| # | Requirement | Method | Verified by |
|---|---|---|---|
| R-7.1 | Preflight shall run a fixed, named set of checks in a fixed order, each returning a value as well as a verdict. | T, V | `tests/test_ops.py::test_preflight_runs_every_check_in_order`, `::test_report_helpers`; `validation/validate_hal_contract.py` check 4 |
| R-7.2 | A window shorter than 1000 dead times shall fail preflight. | T, V | `tests/test_ops.py::test_short_window_fails_preflight`; `validation/validate_hal_contract.py` check 4 |
| R-7.3 | An expected rate at or above `1/tau` on a paralyzable detector shall fail preflight; above `0.3/tau` it shall warn. | T, V | `tests/test_ops.py::test_saturated_paralyzable_rate_fails_preflight`, `::test_near_saturation_warns_but_does_not_block`; `validation/validate_hal_contract.py` check 4 |
| R-7.4 | An expected observed rate within 10 % of the paralyzable maximum shall warn that the inverse branch is ambiguous. | T | `tests/test_ops.py::test_branch_ambiguity_warns_near_the_maximum` |
| R-7.5 | A declared afterpulse probability of zero shall warn rather than pass silently. | T | `tests/test_ops.py::test_zero_afterpulsing_warns` |
| R-7.6 | A device backend that cannot self-test for want of hardware shall produce a warning, not a failure. | T, V | `tests/test_ops.py::test_device_backend_self_test_is_a_warning_not_a_failure`; `validation/validate_hal_contract.py` check 4 |
| R-7.7 | A run record shall capture the backend description, mode, simulated flag, request, preflight report, per-window counts, wall time, seed and environment, and shall serialise without NaN tokens. | T, V | `tests/test_ops.py::test_run_record_round_trip`, `::test_run_record_json_has_no_nan_tokens`; `validation/validate_hal_contract.py` check 5 |
| R-7.8 | A run shall write an in-progress journal before its first window and remove it only on commit; beginning a run over an existing in-progress journal shall be refused. | T | `tests/test_ops.py::test_journal_marks_and_clears_a_run`, `::test_journal_refuses_to_overwrite_evidence_of_a_failure` |
| R-7.9 | Backout shall convert every in-progress journal to an aborted record without deleting anything, shall be idempotent, shall leave completed records untouched, and shall survive an unreadable journal. | T, V | `tests/test_ops.py::test_backout_renames_in_progress_journals`, `::test_backout_is_idempotent`, `::test_backout_leaves_completed_records_alone`, `::test_backout_survives_an_unreadable_journal`; `validation/validate_hal_contract.py` check 5 |
| R-7.10 | No function in this package shall emit or store an absolute filesystem path; labels and filenames shall be bare names. | T, V | `tests/test_ops.py::test_environment_summary_contains_no_filesystem_path`, `::test_read_run_record_rejects_a_path`, `::test_journal_rejects_a_path_label`; `validation/validate_hal_contract.py` check 6 |
| R-7.11 | A benchmark harness shall record latency, memory and throughput together with the measurement method and the environment in its own output file. | V, I | `benchmark/run_benchmark.py`, `benchmark/benchmark_results.md` |

## 8. Capacity and rate bounds

| # | Requirement | Method | Verified by |
|---|---|---|---|
| R-8.1 | The background-free PPM erasure capacity `(1 - exp(-n_s)) log2 M` shall be exact and shall agree with an independent Monte Carlo. | T, V | `tests/test_capacity.py::test_erasure_capacity_known_answer`, `::test_erasure_capacity_limits`; `validation/validate_capacity.py` check 1 |
| R-8.2 | Photons per bit shall approach `1/log2 M` from above as `n_s -> 0` and shall never cross it. | T, V | `tests/test_capacity.py::test_photons_per_bit_approaches_the_limit_from_above`, `::test_photons_per_bit_never_below_the_limit`; `validation/validate_capacity.py` check 2 |
| R-8.3 | Bits per photon shall be shown to grow without bound in `M`. | T, V | `tests/test_capacity.py::test_bits_per_photon_is_unbounded_in_order`; `validation/validate_capacity.py` check 3 |
| R-8.4 | The hard-decision capacity of the induced symmetric channel shall be exact, bounded by `log2 M`, and monotone in the signal level. | T, V | `tests/test_capacity.py::test_hard_decision_capacity_is_below_log2_m`, `::test_hard_decision_capacity_saturates_on_a_clean_channel`, `::test_hard_decision_capacity_is_bounded`; `validation/validate_capacity.py` check 5 |
| R-8.5 | The soft-decision achievable rate shall be at least the hard-decision capacity for every configuration, satisfying the data-processing inequality, and shall carry a standard error. | T, V | `tests/test_capacity.py::test_soft_decision_rate_exceeds_hard_decision_rate`, `::test_soft_decision_rate_is_below_log2_m`; `validation/validate_capacity.py` check 4 |
| R-8.6 | No capacity bound shall be cited or evaluated that was not verified in this environment. | I | `src/photoncount/capacity.py` module docstring, "What is not shipped" |

## 9. Learned rate correction

| # | Requirement | Method | Verified by |
|---|---|---|---|
| R-9.1 | The closed-form inversions shall be implemented and measured before the learned model, on the same held-out rows. | V, I | `validation/validate_correction.py` sections 1-3 |
| R-9.2 | Both textbook inversions and a composed closed form that receives the same inputs as the learned model shall be reported. | T, V | `tests/test_correction.py::test_composed_baseline_removes_the_afterpulse_inflation_first`, `::test_composed_equals_matched_without_afterpulsing`; `validation/validate_correction.py` section 3 |
| R-9.3 | A baseline that cannot answer shall return `nan` and shall be reported with its defined fraction, not silently dropped. | T | `tests/test_correction.py::test_baselines_return_nan_where_undefined`, `::test_rate_error_metrics_known_answer`, `::test_rate_error_metrics_all_undefined` |
| R-9.4 | The learned model shall emit an uncertainty interval, and the interval's coverage shall be measured on held-out data and reported. | T, V | `tests/test_correction.py::test_interval_is_ordered_per_row`, `::test_interval_coverage_known_answer`; `validation/validate_correction.py` section 4 |
| R-9.5 | Results shall be reported per regime, including the regime where the closed forms win and the regime where no estimator works. | T, V | `tests/test_correction.py::test_closed_forms_win_where_their_assumptions_hold`, `::test_nobody_inverts_the_paralyzable_upper_branch`, `::test_regime_table_covers_every_estimator`; `validation/validate_correction.py` section 5 |
| R-9.6 | The dataset shall be deterministic in its seed and shall straddle the paralyzable maximum. | T | `tests/test_correction.py::test_dataset_is_deterministic_in_the_seed`, `::test_dataset_straddles_the_paralyzable_maximum` |
| R-9.7 | No feature shall encode the label. The removal of the mean-count feature shall be enforced by a test. | T, V | `tests/test_correction.py::test_dataset_has_no_mean_count_feature`; `validation/validate_correction.py` section 7 (Fano ablation) |
| R-9.8 | The model shall persist with joblib and shall refuse `.pt`, `.pth`, `.ckpt` and `.onnx` suffixes. | T | `tests/test_correction.py::test_save_and_load_round_trip`, `::test_save_rejects_a_non_joblib_suffix`, `::test_load_rejects_an_unknown_version` |
| R-9.9 | Training and evaluation shall fit inside the stated compute budget of two shared cores and three minutes. | V | `validation/validate_correction.py` sections 0, 2, 8 (runtimes printed) |

## 10. Interfaces, style and provenance

| # | Requirement | Method | Verified by |
|---|---|---|---|
| R-10.1 | `python -m photoncount --help` shall exit 0 in a clean subprocess. | T | `tests/test_cli.py::test_module_help_exits_zero_in_a_clean_subprocess`, `::test_module_version_exits_zero_in_a_clean_subprocess` |
| R-10.2 | Every CLI subcommand shall emit JSON and nothing else. | T | `tests/test_cli.py` (all `_run` based tests) |
| R-10.3 | A CLI acquisition shall abort with a non-zero status when preflight fails, and shall report a device backend's refusal distinctly. | T | `tests/test_cli.py::test_acquire_subcommand_aborts_on_failed_preflight`, `::test_acquire_subcommand_device_backend_reports_not_implemented` |
| R-10.4 | Invalid inputs shall raise `ValueError` or `TypeError` with an actionable message, never produce a silent wrong answer. | T | `tests/test_poisson.py::test_invalid_inputs_raise_value_error`, `tests/test_webb.py::test_invalid_parameters_raise`, `tests/test_ppm.py::test_invalid_config_raises`, `tests/test_deadtime.py::test_bad_dead_time_raises`, `tests/test_simulate.py::test_invalid_spec_raises` |
| R-10.5 | The published numbers shall be pinned by regression tests so that a code change that moves them fails the suite. | T | `tests/test_integration.py::test_regression_pinned_symbol_error_probability`, `::test_regression_pinned_erasure_capacity`, `::test_regression_pinned_paralyzable_maximum`, `::test_regression_pinned_webb_moments` |
| R-10.6 | The full chain from link budget to corrected rate shall be exercised end to end. | T, V | `tests/test_integration.py::test_link_budget_to_slot_statistics_to_decoder_input`, `::test_full_operational_sequence_with_recovery`, `::test_acquisition_feeds_the_closed_form_corrections`; `validation/worked_example.py` |
| R-10.7 | `ruff check src/ tests/ examples/ validation/ benchmark/` shall report no findings under the configured rule set. | T, I | CI workflow `.github/workflows/tests.yml`, lint step |
| R-10.8 | The product shall be labelled validation level 3, hardware-pending, and shall never be labelled level 4. | I | `README.md` header, `src/photoncount/__init__.py`, `src/photoncount/hal.py`, `benchmark/benchmark_results.md` |

---

## Requirements deliberately not met

| # | Requirement a reader might expect | Why it is not met |
|---|---|---|
| N-1 | Recover the incident rate past the paralyzable maximum. | The forward map is two-valued there. Measured: every estimator, learned and closed-form, has a median relative error above 0.8 in that band (`validation/validate_correction_output.txt`, regime `paralyzable & x>1.5`). Resolving it needs a second observable this package does not have, such as a second detector or a modulated source. |
| N-2 | Measured latency, memory and throughput on flight-representative hardware. | Level 4. Requires `benchmark/run_benchmark.py` run on a Jetson Orin Nano. No number in this repository substitutes. |
| N-3 | A validated dead-time model, afterpulse probability or dark-count rate for any real detector. | All detector parameters here are inputs, not measurements. `photoncount.hal.DeviceBackend` declares them and says so. |
| N-4 | Multipulse (multi-slot) PPM, as in the Hamkins & Moision report. | Only the single-pulse M-ary case is implemented. |
| N-5 | A peak-and-average-constrained Poisson capacity bound. | Not verifiable in this environment; omitted rather than cited without verification (R-8.6). |
| N-6 | Multi-exponential or power-law afterpulse release statistics. | A single exponential is implemented; real SPADs show heavier tails. |
