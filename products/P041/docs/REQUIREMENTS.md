# Requirements and traceability — codedfade 0.1.0

Validation level 3 obliges this product to carry numbered requirements and to name
the test that exercises each one. The table below is the authority; the test names
are the `pytest` node identifiers, so each row is runnable as written:

```bash
python -m pytest "tests/test_fade.py::TestFadeStatisticsDefinitions::test_hand_computed_record" -q
```

Requirements are written as verifiable statements. Where a requirement is satisfied
only within a stated range or under a stated assumption, that limit is part of the
requirement text and not a footnote.

Status vocabulary: **MET** means the named test passes in the suite run recorded in
`validation/VALIDATION.md`. **MET (measured, not asserted)** means the test records
what the implementation does rather than comparing it to an external figure, which
is the honest status where no external reference was verifiable.

## 1. Correlated fading channel

| # | Requirement | Test | Status |
|---|---|---|---|
| REQ-01 | A channel configuration shall reject a non-positive scintillation index, correlation time or sample rate, an unknown marginal or kernel, and a sample rate that gives fewer than 2 samples per correlation time. | `tests/test_channel.py::TestConfigValidation` (7 tests) | MET |
| REQ-02 | The Gauss-Markov (`exp`) kernel shall produce a unit-variance, zero-mean Gaussian path whose sample autocorrelation matches `exp(-k/Lc)` to within 0.02 over lags 0–40 at `Lc = 20`. | `tests/test_channel.py::TestCorrelatedGaussian::test_ar1_autocorrelation_matches_exp_kernel` | MET |
| REQ-03 | The Gaussian (`gauss`) kernel shall produce a path whose sample autocorrelation matches `exp(-(k/Lc)^2)` to within 0.03 over the same lags, and both kernels shall reach `1/e` at lag `Lc` to within 0.03. | `tests/test_channel.py::TestCorrelatedGaussian::test_gauss_kernel_autocorrelation_is_gaussian`, `::test_one_over_e_point_is_the_correlation_time` | MET |
| REQ-04 | The lognormal marginal shall have unit mean irradiance to within 3 %, a sample scintillation index within 10 % of the target, and a log-irradiance that passes a Kolmogorov–Smirnov test against `N(-s^2/2, s^2)` at `p > 0.01` on thinned samples. | `tests/test_channel.py::TestLognormalMarginal` (5 tests) | MET |
| REQ-05 | The gamma-gamma marginal shall satisfy the scintillation-index identity of equation (6) exactly under the plane-wave relations (7)–(9), shall invert that relation on its monotone branch to within `1e-6` relative, and shall refuse a target above the model's saturation peak. | `tests/test_channel.py::TestGammaGamma` (7 tests) | MET |

## 2. Codes

| # | Requirement | Test | Status |
|---|---|---|---|
| REQ-06 | GF(2^m) arithmetic shall satisfy commutativity, associativity, distributivity over XOR, the multiplicative identity and inverses for `m` in {3,4,5,6,8}, with the exponential table a permutation of the non-zero elements, and shall reject out-of-range elements and division by zero. | `tests/test_gf.py::TestFieldAxioms`, `::TestValidation`, `::TestKnownAnswers` (24 tests) | MET |
| REQ-07 | Reed-Solomon encoding shall be systematic, shall produce an all-zero syndrome vector for every codeword, and shall reject a message of the wrong shape or with out-of-range symbols. | `tests/test_reedsolomon.py::TestEncode` (5 tests) | MET |
| REQ-08 | RS(15,11) over GF(2^4) shall correct **every** single-symbol error pattern (225) and **every** double-symbol error pattern (23 625 patterns, exhaustive). | `tests/test_reedsolomon.py::TestKnownAnswersAtTheRadius::test_every_single_error_is_corrected`, `::test_every_double_error_is_corrected_exhaustively` | MET |
| REQ-09 | At `t+1` symbol errors the decoder shall never report success with the correct message; it shall either declare failure or miscorrect, and both outcomes shall occur. | `tests/test_reedsolomon.py::TestKnownAnswersAtTheRadius::test_t_plus_one_errors_are_never_decoded_successfully_to_the_original`, `TestWorkingCode::test_t_plus_one_errors_never_decode_successfully_on_rs_31_21` | MET |
| REQ-10 | The rate-1/2, K=3, (0o7, 0o5) convolutional encoder shall produce `111000010111` for the message `1011`, matching the hand trace shown in the test, and shall terminate in the all-zero state. | `tests/test_convolutional.py::TestHandTrace::test_encode_1011_matches_the_hand_trace`, `::test_encode_terminates_in_the_zero_state` | MET |
| REQ-11 | The Viterbi decoder shall recover the message from any single bit error in the codeword, shall report a path metric equal to the Hamming distance to the codeword it chose, and shall fail on a contiguous burst longer than the measured minimum terminated weight. | `tests/test_convolutional.py::TestViterbi` (4 tests), `TestHandTrace::test_minimum_terminated_weight_is_measured_not_asserted` | MET (measured, not asserted) |

## 3. Interleaving

| # | Requirement | Test | Status |
|---|---|---|---|
| REQ-12 | The block interleaver shall be a bijection on whole blocks, shall have an exact inverse, and shall place adjacent symbols of one codeword exactly `depth` symbol periods apart in the channel stream. | `tests/test_interleave.py::TestBlockInterleaver::test_known_answer_depth_3_span_4`, `::test_deinterleave_inverts_interleave`, `::test_adjacent_codeword_symbols_are_depth_apart`, `::test_block_interleaver_is_a_permutation` | MET |
| REQ-13 | The convolutional interleaver composed with its de-interleaver shall be a pure delay of exactly `M*B*(B-1)` symbols, measured by pushing a ramp through the implementation. | `tests/test_interleave.py::TestConvolutionalInterleaver::test_end_to_end_delay_is_measured_not_asserted`, `::test_convolutional_interleaver_is_a_pure_delay` | MET |
| REQ-14 | Both interleavers shall report latency in symbols and in milliseconds at a stated symbol rate, and memory in symbols and bytes at a stated bits-per-symbol, and shall reject a non-positive symbol rate or bits-per-symbol. | `tests/test_interleave.py::TestBlockInterleaver::test_cost_formulas`, `::test_rejects_bad_cost_arguments`, `TestConvolutionalInterleaver::test_cost_formulas`, `::test_rejects_bad_cost_arguments` | MET |

## 4. Fade statistics and the analytic baseline

| # | Requirement | Test | Status |
|---|---|---|---|
| REQ-15 | Level-crossing rate, mean fade duration and outage fraction shall follow the definitions in `codedfade.fade`, including the exclusion of censored runs and a non-zero duration for a one-sample fade, verified against a hand-computed seven-sample record. | `tests/test_fade.py::TestFadeStatisticsDefinitions` (6 tests), `TestFadeRuns` (5 tests) | MET |
| REQ-16 | Rice's (1945) continuous-time crossing-rate formula shall be implemented for the Gaussian kernel and shall give `sqrt(2)/(2*pi*tau)` at `u = 0`. | `tests/test_fade.py::TestAnalyticCrossingRates::test_rice_formula_known_answer`, `::test_rice_mfd_is_phi_over_rate` | MET |
| REQ-17 | The exact crossing rate of the **sampled** Gauss-Markov path shall agree with a direct Monte Carlo of the bivariate normal to within 2 %, and with a generated AR(1) path to within 3 %; it shall increase without bound as the sample rate rises, because the Ornstein-Uhlenbeck process has no finite continuous-time crossing rate. | `tests/test_fade.py::TestAnalyticCrossingRates::test_markov_rate_matches_a_monte_carlo_of_the_bivariate_normal`, `::test_markov_rate_matches_a_generated_path`, `::test_markov_rate_grows_with_sample_rate`, `::test_markov_mfd_shrinks_with_sample_rate` | MET |
| REQ-18 | On a 2 000 000-sample lognormal Gauss-Markov record the sample mean fade duration shall agree with equation (15) to within 6 % and the sample level-crossing rate with equation (14) to within 5 %. | `tests/test_fade.py::TestSampleAgainstAnalytic` (2 tests) | MET |

## 5. Coded link

| # | Requirement | Test | Status |
|---|---|---|---|
| REQ-19 | The conditional bit-error probability shall be `Q(sqrt(gbar)*I)`, shall equal `Q(10) = 7.6198530e-24` at 20 dB and `I = 1`, shall be monotone decreasing in irradiance, shall tend to 1/2 in a deep fade, and shall reject negative irradiance. | `tests/test_link.py::TestDetectionModel` (6 tests) | MET |
| REQ-20 | Every point of a depth sweep shall see the identical channel record, so that the pre-decoding symbol error rate and the codeword count are the same at every depth. | `tests/test_link.py::TestDepthSweep::test_every_point_sees_the_same_channel_record` | MET |
| REQ-21 | A depth sweep shall report frame error rate with its binomial standard error, post-decoding bit error rate, decoder failures, miscorrections, latency and memory, with latency and memory linear in depth, and shall be reproducible for a fixed seed. | `tests/test_link.py::TestDepthSweep` (remaining 8 tests) | MET |
| REQ-22 | The decoding shortcut (skipping the decoder when the symbol error count is at most `t`) shall give results identical to running the full decoder on the same realisation. | `tests/test_link.py::TestDecodingShortcut::test_shortcut_and_exact_decoding_agree`, `validation/validate_decode_shortcut.py` | MET |

## 6. Hardware abstraction layer

| # | Requirement | Test | Status |
|---|---|---|---|
| REQ-23 | One backend interface shall be defined, with a simulated implementation and a device implementation, and **one** contract test suite shall be parameterised over both, so the device backend is held to the same contract when it is written. | `tests/test_hal_contract.py::TestSharedContract` (13 tests, each run against both backends) | MET |
| REQ-24 | The simulated backend shall refuse `RunMode.LIVE`, shall be deterministic, shall correct a burst of `depth * t` channel symbols after de-interleaving, and shall remain closed if `open` fails. | `tests/test_hal_contract.py::TestSimulatedOnly` (4 tests) | MET |
| REQ-25 | Every operational method of the device backend shall raise `NotImplementedError` naming the Jetson Orin Nano as the missing evidence, and its self-test shall report the missing implementation without raising. | `tests/test_hal_contract.py::TestDeviceOnly` (3 tests) | MET |
| REQ-26 | A preflight check set shall run before any run, shall include a check that the interleaver depth is at least the fade correlation length, shall refuse a configuration whose block exceeds the preflight work limit, and shall contain no absolute filesystem path in any result. | `tests/test_hal_contract.py::TestPreflight` (8 tests) | MET |
| REQ-27 | A session shall refuse to start when preflight fails, shall discard the payload but keep the timing in dry-run mode, shall produce a JSON-serialisable capture with no absolute path, and shall provide an idempotent backout path that survives a backend whose methods raise. | `tests/test_hal_contract.py::TestSession` (7 tests) | MET |

## 7. Learned fade-exceedance predictor

| # | Requirement | Test | Status |
|---|---|---|---|
| REQ-28 | Features shall be causal: truncating the amplitude record immediately after a crossing shall not change that crossing's feature row, and a crossing closer to the record start than the window length shall be dropped. | `tests/test_predictor.py::TestFeatures::test_features_use_only_causal_information`, `::test_window_is_respected` | MET |
| REQ-29 | The predictor shall expose an uncertainty output that is not identically zero, shall be reproducible for a fixed seed, and shall refuse to predict before being fitted or to fit on single-class labels. | `tests/test_predictor.py::TestPredictor` (6 tests) | MET |
| REQ-30 | Brier score, log loss, ROC AUC, reliability table and expected calibration error shall be implemented with known-answer tests, including the degenerate cases (one class present, all-tied scores, empty input). | `tests/test_predictor.py::TestMetrics` (12 tests) | MET |
| REQ-31 | The learned model shall be benchmarked against the analytic exceedance baseline on held-out path seeds, and the measured miscalibration of the exponential closure shall be asserted rather than hidden. | `tests/test_predictor.py::TestBaselineComparison` (3 tests), `validation/validate_ai_vs_baseline.py` | MET |

## 8. Interfaces and reproducibility

| # | Requirement | Test | Status |
|---|---|---|---|
| REQ-32 | `python -m codedfade <command>` shall exit 0 for every subcommand in a clean subprocess with no `PYTHONPATH` beyond `src`, shall exit non-zero on a failed preflight and on an unknown command, and shall print no absolute filesystem path. | `tests/test_cli.py` (22 tests) | MET |
| REQ-33 | Every published number shall be produced by a committed script in `validation/` whose raw output is committed beside it, and the model artefact shall be regenerable from a committed script with fixed seeds. | `validation/VALIDATION.md` (each row names its script), `validation/train_model.py` | MET |

## Requirements deliberately **not** claimed

These are recorded so that a reader does not infer them from the presence of the
surrounding machinery.

| Not claimed | Why |
|---|---|
| Any timing or resource figure as evidence for validation level 4 | Level 4 requires measured timing and resource use from a Jetson Orin Nano. Everything in `benchmark/benchmark_results.json` was measured on a shared cloud build container and says so in its own output. |
| That the gamma-gamma path reproduces a physical two-scale temporal spectrum | The temporal correlation is imposed through a Gaussian copula, which fixes the marginal exactly and the correlation structure only approximately. `validation/validate_channel_statistics.py` reports the size of the disagreement. |
| That `exp(-t/MFD)` is the fade-duration distribution | Level-crossing theory fixes the mean only. The measured error of the closure is published in `validation/validate_ai_vs_baseline.py`. |
| A free distance for the convolutional code taken from a reference | The minimum terminated codeword weight is enumerated by the implementation and reported; no published figure is cited for it. |
| Any claim about a shot-noise-limited or APD receiver | Equation (26) assumes signal-independent noise and is wrong for those receivers. |
