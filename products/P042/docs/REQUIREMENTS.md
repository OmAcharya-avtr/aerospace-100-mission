# Requirements and traceability — acmpilot 0.1.0

Validation level 3 obligation: every requirement is numbered, and every
requirement names the test or validation script that exercises it. A requirement
with no named test is a defect in this document.

Test names below are pytest node identifiers relative to the repository root.
Where a requirement is demonstrated by a validation script rather than a unit
test, the script and the section of its saved output are named instead, and that
is marked **V** in the Kind column.

Status vocabulary: **MET** means the named test passes in the run recorded in
`CHANGELOG.md`; **MET, with stated limitation** means the test passes and the
limitation is recorded in the README; **NOT MET** means the requirement is
written down and deliberately not satisfied in 0.1.0.

## 1. Channel model

| # | Requirement | Kind | Exercised by | Status |
|---|---|---|---|---|
| R-001 | The driver process shall be a stationary unit-variance first-order Gauss-Markov process whose lag-1 correlation is `exp(-dt/tau_c)`. | test | `tests/test_channel.py::TestGaussMarkovPath::test_stationary_variance_and_lag1`, `::test_known_answer_rho_equals_one_over_e` | MET |
| R-002 | A sample path shall be exactly reproducible from its seed. | test | `tests/test_channel.py::TestGaussMarkovPath::test_reproducible_from_seed`, `tests/test_channel.py::TestPaths::test_snr_path_reproducible` | MET |
| R-003 | The lognormal irradiance shall have unit mean and the requested scintillation index. | test | `tests/test_channel.py::TestLognormalIrradiance::test_unit_mean`, `::test_scintillation_index_recovered` | MET |
| R-004 | The gamma-gamma shape parameters shall solve the Al-Habash 2001 scintillation-index relation exactly at any requested index and shape ratio. | test | `tests/test_channel.py::TestGammaGamma::test_shapes_satisfy_equation_7`, `::test_property_shape_solution_round_trips` | MET |
| R-005 | The gamma-gamma irradiance shall have unit mean and the requested scintillation index. | test | `tests/test_channel.py::TestGammaGamma::test_unit_mean_and_scintillation_index` | MET |
| R-006 | Fade duration shall be an emergent statistic of the sample path, never an input parameter, and shall increase with the correlation time. | test | `tests/test_channel.py::TestFadeStatistics::test_mean_fade_duration_grows_with_tau_c` | MET |
| R-007 | The fade-duration scaling with correlation time shall be measured and reported, not assumed. | **V** | `validation/validate_channel.py` V1.6; `examples/channel_and_fades.py` panel 3 | MET, with stated limitation: the measured log-log slope is 0.467, not 1 |
| R-008 | Fade statistics shall be internally consistent: mean fade duration times level-crossing rate shall equal the outage fraction. | test | `tests/test_channel.py::TestFadeStatistics::test_duration_times_rate_equals_outage` | MET |
| R-009 | Fade runs truncated by the start or end of the series shall be excluded from the duration statistics and still counted in the outage fraction. | test | `tests/test_channel.py::TestFadeStatistics::test_censored_runs_excluded_from_durations` | MET |
| R-010 | The measured level-crossing rate shall agree with the analytic Rice prediction, at the sampling interval the prediction is stated for, to within 3%. | test | `tests/test_channel.py::TestRiceCrossingRate::test_matches_measured_within_three_percent` | MET |
| R-011 | The measured 1/e correlation time of the lognormal dB series shall recover the input correlation time. | test | `tests/test_channel.py::TestPaths::test_lognormal_db_correlation_time_recovers_tau_c` | MET |
| R-012 | The gamma-gamma irradiance correlation time shall be shorter than its driver's, and the discrepancy shall be measured and documented rather than hidden. | test + **V** | `tests/test_channel.py::TestPaths::test_gamma_gamma_correlation_time_shorter_than_driver`; `validation/validate_channel.py` V1.5 | MET |
| R-013 | The mapping from irradiance to electrical SNR shall be parameterised by a documented detector exponent (1 coherent, 2 thermal-limited direct detection). | test | `tests/test_channel.py::TestPaths::test_detector_exponent_doubles_db_swing` | MET |
| R-014 | All channel inputs shall be validated with actionable messages. | test | `tests/test_channel.py::TestChannelConfig::test_rejects_invalid_config`, `TestGaussMarkovPath::test_rejects_invalid_input` | MET |

## 2. Modulation, coding and the MODCOD ladder

| # | Requirement | Kind | Exercised by | Status |
|---|---|---|---|---|
| R-015 | Every constellation shall be Gray-mapped and normalised to unit average symbol energy. | test | `tests/test_modulation.py::TestConstellations::test_unit_average_energy`, `::test_qpsk_gray_adjacent_points_differ_in_one_bit`, `::test_8psk_gray_adjacent_points_differ_in_one_bit`, `::test_16qam_nearest_neighbours_differ_in_one_bit` | MET |
| R-016 | Monte Carlo uncoded BER shall agree with the exact closed form for BPSK and QPSK within 4 standard errors. | test | `tests/test_modulation.py::TestMeasureBer::test_matches_exact_closed_form_within_four_sigma` | MET |
| R-017 | The 8PSK and 16QAM closed forms shall be labelled as high-SNR approximations and shall not be used as a threshold source. | **V** | `validation/validate_modcod_thresholds.py` V2.1; `src/acmpilot/modulation.py` module docstring | MET |
| R-018 | Reed-Solomon code parameters shall follow from the MDS property, with no table lookup. | test | `tests/test_coding.py::TestConstruction::test_mds_properties`, `::test_known_answer_rs_255_223` | MET |
| R-019 | Post-decoding frame, symbol and bit error rates shall be monotone in the channel bit error rate and shall never exceed it. | test | `tests/test_coding.py::TestPostDecodingRates::test_monotone_in_channel_ber`, `::test_output_ber_never_exceeds_channel_ber` | MET |
| R-020 | The post-decoding combinatorics shall be verified against a direct Monte Carlo through the real modulator. | test + **V** | `tests/test_coding.py::TestDirectMonteCarloAgreement::test_formula_matches_direct_simulation_for_bpsk`; `validation/validate_modcod_thresholds.py` V2.3 | MET |
| R-021 | The cost of the ideal-bit-interleaving assumption shall be measured, not assumed away. | **V** | `validation/validate_modcod_thresholds.py` V2.3, the 16QAM interleaved/non-interleaved pairs | MET: without a bit interleaver the measured error rate is up to 3.8% higher than the model |
| R-022 | Every MODCOD threshold shall be measured by this package, not quoted from a standard. | test + **V** | `tests/test_modcod.py::TestMeasureThresholds::test_provenance_recorded_and_path_free`; `validation/validate_modcod_thresholds.py` V2.4 | MET |
| R-023 | Every MODCOD threshold shall carry a Monte Carlo uncertainty. | test | `tests/test_modcod.py::TestMeasureThresholds::test_uncertainties_are_small_and_positive` | MET |
| R-024 | The threshold ladder shall be strictly monotone in spectral efficiency and shall contain no dominated MODCOD. | test | `tests/test_modcod.py::TestMeasureThresholds::test_table_is_monotone`, `::test_no_dominated_modcod` | MET |
| R-025 | Lowering the target BER shall never lower a threshold. | test | `tests/test_modcod.py::TestMeasureThresholds::test_lower_target_ber_needs_more_snr` | MET |
| R-026 | The threshold table shall serialise to JSON and back without loss and without any filesystem path. | test | `tests/test_modcod.py::TestModcodTable::test_round_trip_json_file`, `::test_serialised_form_contains_no_filesystem_path` | MET |
| R-027 | Re-measuring the thresholds with the shipped seed shall reproduce the committed table within 0.1 dB. | test | `tests/test_modcod.py::TestRegressionAgainstCommittedTable::test_measured_thresholds_match_committed_within_tenth_of_a_db` | MET |

## 3. Feedback delay and policies

| # | Requirement | Kind | Exercised by | Status |
|---|---|---|---|---|
| R-028 | The round-trip feedback delay shall be an explicit parameter, and the channel-state report at slot n shall be exactly the channel state at slot n-d. | test | `tests/test_simulate.py::TestDelayedObservation::test_known_answer_shift`, `::test_property_tail_matches_shifted_head` | MET |
| R-029 | Three non-learned policies shall be provided, in the order fixed margin, threshold with hysteresis, clairvoyant upper bound. | test | `tests/test_policy.py::TestBaselinePolicies::test_returns_three_in_specified_order` | MET |
| R-030 | The fixed-margin policy shall select the highest MODCOD supported by the delayed report minus the margin. | test | `tests/test_policy.py::TestFixedMargin::test_known_answer_selection`, `::test_zero_margin_matches_best_supported_floored` | MET |
| R-031 | The hysteresis policy shall have a configurable dead band and shall change MODCOD by at most one step per slot. | test | `tests/test_policy.py::TestThresholdHysteresis::test_known_answer_dead_band_holds`, `::test_index_changes_by_at_most_one_per_slot` | MET |
| R-032 | Widening the hysteresis dead band shall reduce the switch rate. | test + **V** | `tests/test_policy.py::TestThresholdHysteresis::test_switch_count_falls_as_dead_band_widens`; `validation/validate_policies.py` V3.5 | MET |
| R-033 | The clairvoyant policy shall be labelled, in code, documentation and every plot legend, as an upper bound no causal policy can achieve. | test | `tests/test_policy.py::TestClairvoyantUpperBound::test_is_labelled_acausal`, `::test_label_constant_states_unachievability`, `tests/test_integration.py::TestRepositoryHygiene::test_readme_labels_the_clairvoyant_bound` | MET |
| R-034 | The clairvoyant policy's goodput shall be independent of the feedback delay. | test + **V** | `tests/test_simulate.py::TestRunPolicy::test_clairvoyant_is_independent_of_delay`; `validation/validate_policies.py` V3.2 | MET: spread exactly 0 |
| R-035 | No causal policy's goodput shall exceed the clairvoyant bound on the same sample path. | test + **V** | `tests/test_simulate.py::TestRunPolicy::test_clairvoyant_dominates_causal_policies`, `tests/test_integration.py::TestEndToEnd::test_clairvoyant_bound_holds_across_the_whole_sweep`; `validation/validate_policies.py` V3.1 | MET |
| R-036 | No causal policy shall read the true channel state. | test | `tests/test_policy.py::TestFixedMargin::test_ignores_true_snr`, `TestThresholdHysteresis::test_ignores_true_snr`, `tests/test_predictor.py::TestPredictivePolicy::test_does_not_read_the_true_snr` | MET |
| R-037 | Goodput shall fall and outage shall rise as the feedback delay grows. | test | `tests/test_simulate.py::TestRunPolicy::test_goodput_falls_as_delay_grows`, `::test_outage_rises_as_delay_grows` | MET |
| R-038 | Every policy shall return an in-range MODCOD index at every slot. | test | `tests/test_policy.py::TestFixedMargin::test_never_returns_out_of_range_index`, `TestThresholdHysteresis::test_index_range`, `tests/test_predictor.py::TestPredictivePolicy::test_selection_shape_and_range` | MET |

## 4. Accounting

| # | Requirement | Kind | Exercised by | Status |
|---|---|---|---|---|
| R-039 | Mis-selection shall be reported split into too-aggressive and too-conservative, never as one number. | test | `tests/test_accounting.py::TestKnownAnswers::test_hand_computed_four_slot_case` | MET |
| R-040 | The three outcome categories (exact, conservative, aggressive) shall partition the accounted slots. | test | `tests/test_accounting.py::TestInvariants::test_three_categories_partition_the_slots` | MET |
| R-041 | Outage caused by the channel supporting no MODCOD at all shall be reported separately from policy-caused outage. | test | `tests/test_accounting.py::TestKnownAnswers::test_hand_computed_four_slot_case`, `TestInvariants::test_avoidable_plus_unavoidable_equals_aggressive` | MET |
| R-042 | The two mis-selection kinds shall carry their separate costs in bit/symbol, and goodput plus wasted plus lost shall equal the clairvoyant goodput. | test | `tests/test_accounting.py::TestInvariants::test_goodput_plus_wasted_plus_lost_equals_clairvoyant` | MET |
| R-043 | Efficiency shall lie in [0, 1] and shall be reported as not-a-number when the channel never supports any MODCOD. | test | `tests/test_accounting.py::TestInvariants::test_goodput_never_exceeds_clairvoyant`, `TestAccountingDataclass::test_efficiency_is_nan_when_channel_never_supports_anything` | MET |
| R-044 | Every reported quantity shall be a function of the feedback delay and of the channel correlation time. | **V** | `validation/validate_policies.py` V3.3 and V3.4 | MET |
| R-045 | A warm-up period shall be excluded from the accounting, at least as long as the feedback delay. | test | `tests/test_accounting.py::TestKnownAnswers::test_warmup_excludes_leading_slots`, `tests/test_simulate.py::TestRunPolicy::test_default_warmup_is_at_least_the_delay` | MET |

## 5. The learned component

| # | Requirement | Kind | Exercised by | Status |
|---|---|---|---|---|
| R-046 | A non-learned analytic predictor shall be implemented first and reported alongside the learned one. | test + **V** | `tests/test_predictor.py::TestGaussMarkovPredictor::test_not_learned`, `tests/test_benchmark.py::TestBenchmarkAtDelay::test_variants_present`; `validation/validate_predictor.py` V4.4a | MET |
| R-047 | The learned predictor shall emit a confidence output, not only a point estimate. | test | `tests/test_predictor.py::TestQuantilePredictor::test_spread_is_non_negative`, `::test_spread_varies_across_rows` | MET |
| R-048 | The confidence output shall gate aggressive rate choices: a larger gate shall never select a higher MODCOD, and shall reduce outage. | test | `tests/test_predictor.py::TestPredictivePolicy::test_larger_gate_never_selects_higher`, `::test_gate_reduces_outage` | MET |
| R-049 | Calibration of the stated confidence shall be reported, not only mean error. | test + **V** | `tests/test_predictor.py::TestCalibrationReport::test_coverage_near_nominal`, `::test_tail_fractions_sum_with_coverage`; `validation/validate_predictor.py` V4.1 | MET |
| R-050 | The learned predictor shall be benchmarked against all three non-learned baselines on the same seeded sample paths. | **V** | `validation/validate_predictor.py` V4.2b; `validation/validate_predictor_gammagamma.py` V4.2b | MET |
| R-051 | Every baseline's free parameters shall be tuned before the comparison, on data disjoint from the test data. | test + **V** | `tests/test_benchmark.py::TestTuning::test_tune_fixed_margin_returns_the_argmax_of_its_trace`, `tests/test_benchmark.py::TestBenchmarkSplit::test_rejects_overlap`; `validation/validate_predictor.py` V4.2a | MET |
| R-052 | If a baseline wins, the baseline winning shall be the published result. | **V** | `validation/validate_predictor.py` V4.4a and V4.4b; README "The result" | MET: the non-learned analytic predictor is nominally ahead of the learned one at every delay on both channels, and a tuned fixed margin is indistinguishable from both at `tau` = 1 ms |
| R-053 | The learned predictor shall be persistable without committing a model binary, and shall be regenerable deterministically from a seed. | test | `tests/test_predictor.py::TestQuantilePredictor::test_round_trip_joblib`, `::test_deterministic_given_the_same_data_and_seed`, `tests/test_integration.py::TestRepositoryHygiene::test_no_model_binaries_present` | MET |
| R-054 | Prediction error shall grow with the feedback horizon. | test | `tests/test_predictor.py::TestCalibrationReport::test_error_grows_with_horizon` | MET |
| R-055 | The confidence output shall be informative: a wider stated interval shall correspond to a larger realised error. | **V** | `examples/predictor_vs_baselines.py` panel 4 | MET |
| R-056 | The learned predictor shall not be given information the baselines lack; its parameters shall be estimated from the training data only. | test | `tests/test_benchmark.py::TestFitPredictors::test_training_uses_only_the_training_seeds`, `tests/test_predictor.py::TestGaussMarkovPredictor::test_fit_estimates_three_parameters` | MET |

## 6. Interface, packaging and documentation

| # | Requirement | Kind | Exercised by | Status |
|---|---|---|---|---|
| R-057 | `python -m acmpilot --help` shall exit 0 in a clean subprocess. | test | `tests/test_cli.py::TestSubprocess::test_help_exits_zero` | MET |
| R-058 | Every CLI subcommand shall have help and shall exit 0 on a valid invocation. | test | `tests/test_cli.py::TestSubprocess::test_each_subcommand_has_help`, `TestInProcess` | MET |
| R-059 | No tracked file shall contain an absolute filesystem path. | test | `tests/test_integration.py::TestRepositoryHygiene::test_no_absolute_paths_in_tracked_text` | MET |
| R-060 | No model binary shall be tracked. | test | `tests/test_integration.py::TestRepositoryHygiene::test_no_model_binaries_present` | MET |
| R-061 | No documentation shall contain an unqualified certification, flight-safety, mission-ready or production-ready claim. | test | `tests/test_integration.py::TestRepositoryHygiene::test_no_certification_claims_in_documentation` | MET |
| R-062 | Every example shall use the Agg backend, never call `show()`, and write its figure to `screenshots/`. | test | `tests/test_integration.py::TestRepositoryHygiene::test_every_example_writes_to_screenshots`, `::test_every_screenshot_exists_for_every_example` | MET |
| R-063 | Every validation script shall have its raw saved output committed beside it. | test | `tests/test_integration.py::TestRepositoryHygiene::test_every_validation_script_has_saved_output` | MET |
| R-064 | The README shall name P015 LinkSwitch and state the distinction. | test | `tests/test_integration.py::TestRepositoryHygiene::test_readme_names_p015_linkswitch` | MET |
| R-065 | The credits line shall appear verbatim exactly once. | test | `tests/test_integration.py::TestRepositoryHygiene::test_credits_line_appears_exactly_once_in_readme` | MET |
| R-066 | `pyproject.toml` shall set `pythonpath = ["src"]` for pytest and the specified ruff configuration. | test | `tests/test_integration.py::TestRepositoryHygiene::test_pyproject_has_pytest_pythonpath`, `::test_ruff_configuration_as_specified` | MET |
| R-067 | The licence shall be AGPL-3.0 with the 2026 OPTIMA Organisation copyright. | test | `tests/test_integration.py::TestRepositoryHygiene::test_licence_is_agpl` | MET |

## 7. Requirements deliberately NOT met in 0.1.0

These are written down so that their absence is a recorded decision rather than
an oversight. Each is also in the README limitations.

| # | Requirement | Why not met |
|---|---|---|
| R-068 | The feedback report shall be quantised and corrupted by a measurement error of stated magnitude. | **NOT MET.** The report is the exact SNR of slot `n-d`. Isolating staleness is the point of the product, but it makes every goodput number optimistic for a real terminal. |
| R-069 | A causal policy shall be able to mute the transmitter rather than transmit the lowest MODCOD into a fade. | **NOT MET.** Flooring at MODCOD 0 keeps the outage accounting legible. A real terminal mutes, which would move that outage out of goodput and into an availability figure. |
| R-070 | Reed-Solomon decoder miscorrection shall be accounted for. | **NOT MET.** Neglected, which makes the post-decoding rates and therefore the thresholds mildly optimistic. A real decoder (`reedsolo`, `galois`) would settle it. |
| R-071 | Interleaver depth, latency and memory shall be modelled and charged. | **NOT MET** by design. `coding.py` assumes an interleaver deep enough to decorrelate a codeword and charges nothing. P041 CodedFade is the product that sizes it. |
| R-072 | The channel model shall be validated against measured turbulence data. | **NOT MET.** No measured data is available to this build. Until it is, nothing here is evidence about a real link. |
| R-073 | The fade-duration distribution shall be independent of the slot rate. | **NOT MET**, and cannot be with this driver: see R-007. The distribution is sampling-rate dependent and must be re-measured at the user's own slot rate. |
