# Requirements — P092 DeadlineRisk

Requirements are numbered and each names the artifact that satisfies it and the
artifact that demonstrates it. A requirement with no demonstrating artifact is
listed as unsatisfied rather than omitted.

Validation level: **Level 4, measured on hardware.** Level 4 here is **not** a claim of generality: one power mode (`MAXN_SUPER`), one uninstrumented ambient, one CPython CPU-only payload, one period, one shared board. See Limitations. The coordinator decided the
level at Phase 9 against the measured evidence. This document does not claim a
level.

## Functional

| ID | Requirement | Satisfied by | Demonstrated by |
|----|-------------|--------------|-----------------|
| F-01 | Load the campaign's `jobs_<run>.csv`, `mon_<run>.csv` and `manifest.json` and reject any file whose header is not exactly the declared schema. | `src/deadlinerisk/trace.py`, `src/deadlinerisk/schema.py` | `tests/test_trace.py`, `tests/test_schema.py`, `validation/validate_loader_contract.py` |
| F-02 | Reconstruct `releases = sum(1 + dropped_before)` and `misses = sum(missed + dropped_before)` from the committed traces. | `src/deadlinerisk/accounting.py` | `tests/test_accounting.py`, `validation/validate_loader_contract.py` |
| F-03 | Produce the per-release miss indicator in release order, of length `releases`, summing to `misses`. | `accounting.miss_indicator` | `tests/test_accounting.py`, `tests/test_properties.py` |
| F-04 | Report the mean-plus-margin rule both as a probability and as the `k` for which `mean + k*sd = D`. | `baselines.mean_plus_margin` | `tests/test_baselines.py`, `validation/validate_measured_campaign.py` |
| F-05 | Fit a generalised Pareto tail to peaks over threshold and report `P(resp > D)` in closed form. | `baselines.gpd_tail` | `tests/test_baselines.py`, `validation/validate_measured_evt.py` |
| F-06 | Report the GPD shape and modified scale over a declared range of thresholds, and choose the threshold by a rule stated in code. | `baselines.threshold_stability`, `baselines.choose_threshold`, `baselines.THRESHOLD_RULE` | `validation/validate_measured_evt.py`, `screenshots/evt_threshold_stability.png` |
| F-07 | Report the counted miss rate with a moving-block bootstrap interval whose block length is selected from the measured autocorrelation of the miss indicator, and report the selection. | `bootstrap.counted_rate_interval`, `bootstrap.block_length_from_autocorrelation`, `bootstrap.BLOCK_RULE` | `tests/test_bootstrap.py`, `validation/validate_measured_campaign.py` |
| F-08 | Report the factor by which the mean-plus-margin rule understates the counted miss rate, with an interval, whatever its value, including an interval containing 1. | `bootstrap.understatement_factor` | `validation/validate_measured_campaign.py`, `validation/worked_example.py` |
| F-09 | Provide the classical speed-scaling predictor `available_fraction = min(1, m/(n+1))` using only descriptors and the idle-run execution-time distribution. | `src/deadlinerisk/speed_scaling.py` | `tests/test_speed_scaling.py`, `validation/validate_measured_learned.py` |
| F-10 | Window the campaign, extract descriptors observable without running the task, and produce a counted miss rate per window. | `src/deadlinerisk/windows.py` | `tests/test_windows.py`, `validation/validate_measured_learned.py` |
| F-11 | Provide a learned predictor with an uncertainty output, scored separately on held-out workload mixes and on held-out time, against both baselines on the same splits. | `src/deadlinerisk/predictor.py`, `windows.grouped_splits`, `windows.time_split` | `tests/test_predictor.py`, `validation/validate_measured_learned.py` |
| F-12 | Provide a command-line interface. | `src/deadlinerisk/__main__.py` | `tests/test_cli.py` |
| F-13 | Refuse, naming every absent file, rather than substituting synthetic data for a measured figure. | `trace.TraceDataAbsentError`, `validation/_bootstrap.require_measured_campaign`, `examples/_common.measured_campaign` | `tests/test_trace.py`, every measured script's own refusal path |

## Verification

| ID | Requirement | Satisfied by | Demonstrated by |
|----|-------------|--------------|-----------------|
| V-01 | At least one bound check whose bound comes from mathematics the implementation does not use. | `src/deadlinerisk/bounds.py` (Markov, Cantelli, Bonferroni bracket) | `tests/test_bounds.py`, `validation/validate_independent_bound.py` |
| V-02 | Every bound slack follows the conditioning of its own expression, with no absolute floor. | `bounds.markov_slack`, `bounds.cantelli_slack`, `bounds.PairBracket.slack` | `tests/test_bounds.py`, `tests/test_regressions.py`, `validation/validate_independent_bound.py` section 2 |
| V-03 | Negative controls anchored on the true value, each asserted to fire in its declared direction. | `tests/test_negative_controls.py`, `validation/validate_negative_controls.py` | the same two files, which report firing as its own check |
| V-04 | No assertion depends on `pytest.approx`'s hidden absolute default. | an explicit `abs=` on every `rel=` call | `validation/validate_approx_audit.py`, which reports the count |
| V-05 | Every Hypothesis counterexample is pinned as a named literal regression test. | `tests/test_regressions.py` | `validation/validate_property_hunt.py` |
| V-06 | Every validation and example script exits identically from four working directories. | `ROOT = Path(__file__).resolve().parent.parent` then `os.chdir(ROOT)` in `validation/_bootstrap.py` and `examples/_common.py` | `validation/probe_working_directories.py` |
| V-07 | No absolute path in any tracked file, and no tracked cache or build artifact. | `validation/_bootstrap.relative`, `.gitignore` | `validation/validate_tree_hygiene.py` |
| V-08 | The test count is taken from junit XML and cross-checked against a count of `testcase` elements. | `--junit-xml` in the reproduction commands | `README.md` badge row and Reproducing every number |

## Measurement

| ID | Requirement | Satisfied by | Demonstrated by |
|----|-------------|--------------|-----------------|
| M-01 | The payload is calibrated so the median idle execution time is about 0.45 of the deadline, and the whole calibration history including failed attempts is recorded. | `data/campaign.py` `calibrate()` | `data/traces/manifest.json` `calibration`, printed by `validation/validate_measured_campaign.py` section 2 |
| M-02 | The ground-truth contention level is chosen by a rule declared before the long run, from a probe table, and both the rule and the table are recorded. | `data/campaign.py` `main()` | `data/traces/manifest.json` `gt_reason`, `gt_rate_band`, `probes` |
| M-03 | Foreign CPU occupancy is measured per second and reported, not assumed absent. | `data/campaign.py` `monitor()` | `validation/validate_measured_campaign.py` section 7, `screenshots/contention_descriptors.png` |
| M-04 | Die temperature, all six core clocks and the load average are recorded at 1 Hz. | `data/campaign.py` `monitor()` | `validation/validate_measured_campaign.py` section 8 |

## Explicitly out of scope, and structurally unmeasurable from this campaign

| ID | Not satisfied | Why |
|----|---------------|-----|
| X-01 | Any statement about a second power mode. | `sudo -n true` returns non-zero on the board, so `nvpmodel` and `jetson_clocks` are unreachable unattended. Every figure is a `MAXN_SUPER` figure. |
| X-02 | Any statement about held-out ambient temperature. | One uninstrumented ambient. There is no second ambient to hold out, so the check cannot exist in this campaign rather than having failed. |
| X-03 | Any statement about a second board, or about Orin Nano in general. | One board, with its own heatsink and fan. |
| X-04 | Any statement about a compiled payload, a GPU payload, or a different period. | One CPython CPU-only payload at one period. |
| X-05 | Attribution of any part of a measured miss rate to foreign load rather than to the declared stressors. | The board was shared and the foreign load was not controlled. The contamination is measured and reported; it is not separable. |
| X-06 | A dependency range verified at both ends by installing both. | The build environment is a shared pre-pinned interpreter and this product is forbidden to run `pip`. Ranges are bounded at the single version each was measured at, except `ruff`, which was measured at 0.16.9 and 0.17.5. |
