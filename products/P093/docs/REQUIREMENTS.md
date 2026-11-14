# Requirements — P093 PowerEnvelope

Requirements are numbered and each names the artifact that satisfies it and the
artifact that demonstrates it. A requirement with no demonstrating artifact is
listed as unsatisfied rather than omitted.

Validation level target: **Level 4, measured on hardware.** Level 4 here is
**not** a claim of generality: one power mode (`MAXN_SUPER`), one uninstrumented
ambient, one CPython CPU-only pipeline, one cycle period, one shared board. See
the README's Limitations. The coordinator decides the level against the measured
evidence; this document does not claim a level.

The measured campaign is committed under `data/traces/`: 20 of 20 planned runs,
`aborted: false`, `CAMPAIGN_COMPLETE` present, 268,425 inferences and 100,789
accepted decisions over 27.05 minutes of board time. Every requirement below is
demonstrated against it.

Two requirements are **not** satisfied by this campaign and are listed as such
rather than omitted: V-10, the campaign-floor sleep-power bound, which is
**falsified** on exactly 7 of the 19 pipeline cells because the campaign's run
order was not randomised; and M-06, coverage of all five gate branches, of which
two never fired.

## Functional

| ID | Requirement | Satisfied by | Demonstrated by |
|----|-------------|--------------|-----------------|
| F-01 | Load a campaign directory into typed records and reject any file whose header is not exactly the declared schema, including a header with the right names in the wrong order. | `src/powerenvelope/trace.py`, `src/powerenvelope/schema.py` | `tests/test_trace.py`, `tests/test_schema.py`, `validation/validate_loader_contract.py` |
| F-02 | State the measurement model in one place and carry it verbatim into the loader docstring, the README, VALIDATION.md and every validation output. | `report.MEASUREMENT_MODEL`, `trace` module docstring | `validation/validate_loader_contract.py`, `validation/VALIDATION.md` |
| F-03 | Integrate rail power trapezoidally over `t_mono`, whole-run and restricted to the active windows derived by intersecting the sampler's timestamps with the decisions' intervals. | `src/powerenvelope/energy.py` | `tests/test_energy.py`, `validation/validate_baselines_synthetic.py` |
| F-04 | Document and count the edge case of a sampler timestamp falling between two inferences inside an active window. | `energy` module docstring, `energy.interstitial_samples` | `tests/test_energy.py`, `report.baseline1_lines` |
| F-05 | Implement Baseline 1 in both variants, `measured-active` and `nominal-rail`. | `baselines.energy_per_inference` | `tests/test_baselines.py`, `validation/validate_baselines_synthetic.py` |
| F-06 | Implement Baseline 2 from measured components, with `P_floor` from the floor cell and `E_ramp` as the measured excess in the settling interval after each wake. | `baselines.floor_power`, `baselines.ramp_energy`, `baselines.duty_cycle_model` | `tests/test_baselines.py`, `validation/validate_baselines_synthetic.py` |
| F-07 | Fit the measured rail-accounting ratio against load across the grid and report it as a rail-accounting ratio, not a converter efficiency. | `baselines.rail_accounting_curve`, `report.rail_lines` | `tests/test_baselines.py`, `validation/validate_baselines_synthetic.py` |
| F-08 | Implement Baseline 3 as integrated whole-run energy over counted accepted decisions, with the between-run spread over the four reference repeats as the interval. | `baselines.ground_truth`, `baselines.between_run_spread` | `tests/test_baselines.py`, `validation/validate_baselines_synthetic.py` |
| F-09 | Say in code why a within-run bootstrap is wrong for this quantity. | `baselines.BetweenRunSpread` docstring, `report.spread_lines` | the docstring and every output file that prints the spread |
| F-10 | Report the headline factor and its decomposition into the rejection factor, the duty-cycle factor and the nominal-versus-measured active-power error, with the closure error reported explicitly rather than claimed. Under the run-window definition the three factors are exact and there is **no residual term**; the phase occupancy `T_act/(N·L)` is reported separately as a diagnostic, and the declared-phase partition's own closure is checked on every cell. | `src/powerenvelope/decomposition.py`, `energy.run_energy`, `energy.declared_active_energy`, `energy.phase_occupancy` | `tests/test_decomposition.py`, `tests/test_properties.py`, `validation/validate_baselines_synthetic.py`, `validation/validate_measured_baselines.py` §6 |
| F-11 | Window the campaign and extract only the licensed descriptors, with the foreign-occupancy imputation flagged in the design matrix. | `src/powerenvelope/windows.py` | `tests/test_windows.py`, `MODEL_CARD.md` |
| F-12 | Provide a learned predictor of mean active rail power with an uncertainty output, scored separately on held-out duty cycles and held-out frame dimensions, grouped so no window of a test group is in training. | `src/powerenvelope/predictor.py`, `windows.grouped_splits` | `tests/test_predictor.py`, `tests/test_integration.py`, `validation/validate_measured_learned.py` §5 |
| F-18 | Implement and report the classical alternatives — a constant, a least-squares fit in the configuration descriptors, and the analytic duty inversion — before the learned model, and score every one of them on the decision-level target rather than on the power alone. | `predictor.ConstantPowerPredictor`, `predictor.LinearActivePowerModel`, `predictor.AnalyticActivePowerModel`, `baselines.two_term_cycle_energy_mj` | `tests/test_predictor.py`, `validation/validate_measured_learned.py` §5–7 |
| F-19 | Load a header-only decisions file for the background floor cell without error, and refuse one for a cell the manifest says ran the pipeline. | `trace.load_decisions(allow_empty=...)`, `trace.load_campaign` | `tests/test_trace.py`, `tests/test_measured_campaign.py` |
| F-20 | Measure how much of the background floor is foreign load, rather than assuming, and publish a floor-sensitivity table. | `baselines.floor_occupancy_regression`, `baselines.floor_variants` | `validation/validate_measured_baselines.py` §3–4 |
| F-21 | Publish the campaign's run order with the per-cell environment, so the drift confound is visible. | `baselines.run_order_rows`, `report.run_order_lines` | `validation/validate_measured_baselines.py` §5, `tests/test_measured_campaign.py` |
| F-13 | Enforce the excluded descriptors in code with a named check. | `predictor.assert_descriptors_licensed`, `predictor.DescriptorLeakError` | `tests/test_predictor.py`, `validation/validate_negative_controls.py` |
| F-14 | Report what a narrower descriptor set costs. | `validation/validate_descriptor_leak.py` | the same script's output |
| F-15 | Provide a documented synthetic generator writing the same CSV schema, used by the tests and by a `--simulate` path, labelled synthetic at the point of use, and never a fallback for a missing measured file. | `src/powerenvelope/synthetic.py` | `tests/test_synthetic.py`, `tests/test_cli.py`, `validation/validate_negative_controls.py` section 7 |
| F-16 | Provide a CLI with `--help` exiting 0, a `headline` subcommand and a `report` subcommand, and a console entry point. | `src/powerenvelope/__main__.py`, `pyproject.toml` `[project.scripts]` | `tests/test_cli.py`, `validation/validate_cli.py` |
| F-17 | Refuse, naming every absent file, rather than substituting synthetic data for a measured figure, and exit non-zero when doing so. | `trace.TraceDataAbsentError`, `validation/_bootstrap.require_measured_campaign`, `examples/_common.measured_campaign` | `tests/test_refusal.py`, `tests/test_cli.py`, the three `validate_measured_*.py` scripts |

## Verification

| ID | Requirement | Satisfied by | Demonstrated by |
|----|-------------|--------------|-----------------|
| V-01 | At least one bound check whose bound comes from mathematics the implementation does not use. | `src/powerenvelope/bounds.py` (Riemann, mean-value, sleep-power, extreme-value) | `tests/test_bounds.py`, `validation/validate_independent_bound.py` |
| V-02 | Every bound slack follows the conditioning of its own expression, with no absolute floor. | `bounds.slack_from_conditioning` | `tests/test_bounds.py`, `validation/validate_independent_bound.py` section 5 |
| V-03 | Negative controls anchored on the true value, each asserted to fire in its declared direction. | `validation/validate_negative_controls.py`, `tests/test_bounds.py` | the same files, which report firing as its own check |
| V-04 | No assertion depends on `pytest.approx`'s hidden absolute default. | an explicit `abs=` on every `rel=` call | `validation/validate_approx_audit.py`, which reports the count |
| V-05 | Every Hypothesis counterexample is pinned as a named literal regression test. | `tests/test_regressions.py` | `validation/validate_property_hunt.py` |
| V-06 | Every validation and example script exits identically from four working directories. | `ROOT = Path(__file__).resolve().parent.parent` then `os.chdir(ROOT)` in `validation/_bootstrap.py` and `examples/_common.py` | `validation/probe_working_directories.py` |
| V-07 | No absolute path in any tracked file, and no tracked cache or build artifact. | `validation/_bootstrap.relative`, `.gitignore` | `validation/validate_tree_hygiene.py` |
| V-08 | The test count is taken from junit XML and cross-checked against a count of `testcase` elements. | `--junit-xml` in the reproduction commands | `README.md` badge row and Reproducing every number |
| V-09 | Every `MEASURED_PENDING` token in the tree is enumerated with its file and line. | `validation/validate_tree_hygiene.py` section 5 | the same script's output |
| V-10 | **NOT SATISFIED — published as a PINNED FALSIFICATION.** The duty-cycle factor lies above a lower bound built from the campaign floor's minimum power. | `bounds.duty_cycle_lower_bracket`, whose docstring records the false premise and names the non-randomised run order as its cause | `validation/validate_measured_headline.py` §6 — **falsified on exactly 7 of the 19 pipeline cells**: `w64_d050_permissive`, `w64_d050_rep2`, `w64_d050_rep3`, `w64_d050_rep4`, `w96_d010`, `w96_d025`, `w96_d050`. The bound, its arithmetic and its tolerance are unchanged and no cell is excluded. The seven cells are pinned as literals in `PINNED_FALSIFICATION`; the script reports `FALSIFIED` as its finding and exits 0 while the set matches, non-zero if it changes. The three duty-1.00 cells are outside the set because a duty-1.00 run has no declared sleep phase and the bound degenerates to `factor >= 1`. The in-cell variant, V-11, holds. |
| V-11 | The duty-cycle factor lies above a lower bound built from each cell's own measured sleep-window minimum. | `bounds.duty_cycle_lower_bracket` with an in-cell premise | `validation/validate_measured_headline.py` §6 — passes on 19 of 19, reported as the weaker argument it is |

## Measurement

| ID | Requirement | Satisfied by | Demonstrated by |
|----|-------------|--------------|-----------------|
| M-01 | `tau` is set by a rule declared in the campaign script before the campaign ran, and the probe scores and both quantiles are recorded. | `data/campaign.py` `calibrate()`, `TAU_STRICT_Q`, `TAU_PERMISSIVE_Q` | `data/traces/manifest.json` `tau_rule` = "tau_strict = quantile(probe_scores, 0.65); tau_permissive = quantile(probe_scores, 0.05); declared before the campaign ran", with per-dimension records: 153.0263/137.75 at 48, 154.7619/139.875 at 64, 157.8333/144.1538 at 96 |
| M-02 | A 60 s warm-up at full occupancy runs before any calibration. | `data/campaign.py` `role_campaign()` | `data/traces/campaign.log` lines 1–2: 20,426 warm-up inferences before the first probe |
| M-03 | Foreign CPU occupancy, die temperature, all six core clocks and the load average are recorded at 1 Hz, and the floor is reported as a background floor on a shared machine. | `data/campaign.py` `role_monitor()`, `report.floor_lines` | `validation/validate_measured_baselines.py` §2–3: floor occupancy 0.123514, 77 samples per cell |
| M-04 | The sampler's own CPU time is recorded per run as measurement back-action. | `data/campaign.py` `role_sampler()` writes `<out>.rusage`; the manifest records `active_cpu_s` | `validation/validate_measured_baselines.py` §14; 1.058 s per 77 s run, 1.374 per cent of one core |
| M-05 | Every expected file is checked for rows after every run and the campaign aborts if one is missing. | `data/campaign.py` `check_file()` and `run_cell()` | `data/traces/campaign.log`; `manifest.json` `aborted: false`, all 40 subprocess return codes 0 |
| M-06 | **NOT SATISFIED.** All five gate branches are exercised by the measurement. | `data/campaign.py` `gate()` | `no_candidate` and `border` have **zero** occurrences across 268,425 inferences. Those branches are exercised only by the synthetic generator; no occurrence was synthesised. `tests/test_measured_campaign.py` asserts the census so it cannot change silently. |
| M-07 | **NOT SATISFIED.** The run order is randomised so that frame dimension is not confounded with elapsed time. | the campaign plan in `data/campaign.py` `role_campaign()` | The plan is sequential by frame dimension. Over 27.05 minutes the 1-minute load average fell from 2.324 to 1.050, so the floor ran at the highest load and the frame-96 cells at the lowest. Published with the per-cell table in `validation/validate_measured_baselines.py` §5 and not corrected out of any figure. |

## Explicitly out of scope, and structurally unmeasurable from this campaign

| ID | Not satisfied | Why |
|----|---------------|-----|
| X-01 | Any statement about a second power mode. | `sudo -n true` returns a password prompt on the board, so `nvpmodel` and `jetson_clocks` are unreachable unattended. Every figure is a `MAXN_SUPER` figure. |
| X-02 | Any statement about held-out ambient temperature. | One uninstrumented ambient. There is no second ambient to hold out, so the check cannot exist in this campaign rather than having failed. |
| X-03 | Any statement about a second board, or about Orin Nano in general. | One board, with its own heatsink and fan. |
| X-04 | Any statement about a GPU or DLA pipeline, a compiled pipeline, or a different cycle period. | One CPython CPU-only pipeline at one 2.0 s cycle. The loader refuses a manifest declaring a different `cycle_s` rather than extrapolating. |
| X-05 | Separation of the measured floor into platform idle and foreign load. | The board was shared and the foreign load was not controlled. The contamination is measured per second and reported; it is not separable. |
| X-06 | The shape, as opposed to the energy, of the wake transient. | The INA3221 averages 512 conversions in hardware, so the transient is bandwidth-limited by the instrument. Measured, `E_ramp` comes out NEGATIVE on 13 of 15 grid cells, between -72 and -202 mJ per wake, because a reading within the 0.15 s settling window still carries the sleep window's power. It is reported as measured and the settling window was not re-chosen after seeing it. |
| X-07 | A ramp-versus-steady decomposition at duty 0.10. | The 0.2 s active phase is shorter than three settling windows. `ramp_energy` reports `separable=False` and the cycle energy is unaffected. |
| X-08 | A `matplotlib` range verified at both ends. | It draws the screenshots and is in no reported number; measured at 3.11.2 only. Every other range WAS run at both ends on 2026-11-14 in an isolated virtual environment — floors numpy 2.5.0 / scipy 1.18.0 / scikit-learn 1.9.0 / pytest 9.1.0 / hypothesis 6.170.0, ceilings numpy 2.5.4 / scipy 1.18.1 / scikit-learn 1.9.1 / pytest 9.1.1 / hypothesis 6.170.15 — both giving 360 passed, with regenerated validation output differing only in the environment stanza that prints the installed versions and in two wall-clock readings. |
| X-09 | Separation of the floor's variation into foreign load and platform idle. | Regressed on its own occupancy the floor gives Pearson r = 0.6932 (r² = 0.4805), so over half of its second-to-second variance is unexplained by the occupancy counter. The regression intercept is an extrapolation below the observed range and is used only as one row of a sensitivity table. |
