# Requirements — thermalbudget

What this product must do, and the state of each requirement as measured in the
build session. A requirement that is NOT MET says so.

Level 3 products in this mission ship this file. It is here because the
product's target level is 4 and the gap between the target and what was achieved
is exactly the content of this file.

## Functional requirements

| # | Requirement | State | Evidence |
|---|---|---|---|
| F1 | Load the campaign's CSV traces against the binding schema, rejecting a damaged file by what is wrong with it | **MET** | `validate_loader_contract_output.txt`, 29 checks |
| F2 | Treat `-1` as missing, never as a value, before any unit conversion, and carry the count | **MET** | `validate_loader_contract_output.txt`; `tests/test_regressions.py` |
| F3 | Infer the temperature scale from the data and refuse when it cannot be inferred | **MET** | `validate_loader_contract_output.txt` |
| F4 | Carry the origin of every figure and refuse to label a synthetic one measured | **MET** | `validate_loader_contract_output.txt`, `validate_negative_controls_output.txt` |
| F5 | TEST equilibrium rather than assuming it, and exclude a failing run by name | **MET** | `validate_equilibrium_output.txt`, 9 controls |
| F6 | Baseline 1, cold-burst extrapolation, with an interval | **MET.** Measured 26160.398 [25478.165, 26923.403] iters/s | `validate_measured_campaign_output.txt` |
| F7 | Baseline 2, closed-form first-order RC, with tau from the rise AND independently from the cooldown, reported side by side | **MET.** Measured rise mean 80.99 s, cooldown mean 110.69 s, ratio 1.367; area rule 81.76 s; log-difference route refused by name | `validate_measured_campaign_output.txt` |
| F8 | Map steady die temperature to clock by the MEASURED relation, and clock to throughput | **MET.** The measured relation is DEGENERATE: 1 clock level over 45.875-57.281 degC, so baseline 2 predicts the unthrottled clock | `validate_measured_campaign_output.txt`, `validate_clock_confound_output.txt` |
| F9 | Baseline 3, measured ground truth with a bootstrap interval across runs AND within runs | **MET.** 25741.404 [25593.028, 25885.792] iters/s across runs; within-run half-widths 226.86, 221.65, 225.65 | `validate_measured_campaign_output.txt` |
| F10 | Headline: the cold-burst overstatement with an interval, reporting INDISTINGUISHABLE when the interval contains zero | **MET.** Measured +1.6277 per cent [-1.0870, +4.6481], VERDICT INDISTINGUISHABLE | `validate_measured_campaign_output.txt` |
| F11 | Learned steady-state predictor with an uncertainty output, sklearn only | **MET**, and the uncertainty is **badly calibrated**: measured two-sigma coverage 0.6111 (ridge), 0.8333 (forest) against 0.95 | `validate_measured_learned_output.txt` |
| F12 | Score the learned model on held-out DUTY CYCLES | **MET.** ridge MAE 4.9205 degC against RC 0.5120; margin -4.4084 = 17.13 half-widths, BASELINE_BETTER | `validate_measured_learned_output.txt` |
| F13 | Score the learned model on held-out AMBIENT | **NOT MET — STRUCTURALLY UNMEASURABLE.** One ambient, no physical access. Reported as unmeasurable with the reason, not approximated | `validate_measured_learned_output.txt`, `MODEL_CARD.md` |
| F14 | Never report a margin inside the run-to-run interval as an improvement | **MET** | `validate_negative_controls_output.txt`, 6 controls on the rule |
| F15 | CLI with an explicit origin on every loading subcommand | **MET** | `tests/test_cli.py` |

## Measurement requirements, which are what Level 4 needs

| # | Requirement | State |
|---|---|---|
| M1 | The campaign's CSV traces and `manifest.json` present under `data/traces/` | **MET.** 17 CSV traces plus `manifest.json`, `model` matching the declared target. Not met at build time; the campaign finished afterwards. |
| M2 | A MEASURED cold-burst overstatement with its interval | **MET.** +1.6277 per cent [-1.0870, +4.6481], INDISTINGUISHABLE. |
| M3 | A MEASURED tau from the rise and independently from the cooldown, with the residual autocorrelation that says whether the first-order form fits the real die | **MET**, and the answer is that it does NOT fit well: rho +0.732, and two independent bounds put the true asymptote above the fitted 56.765 degC. |
| M4 | A MEASURED held-out-duty score for the learned predictor against the RC baseline | **MET.** BASELINE_BETTER by 17.13 run-to-run half-widths. |
| M5 | A MEASURED clock-versus-temperature relation, identifiable or degenerate | **MET: DEGENERATE.** 0 of 9654 per-core samples below 1728000 kHz over a 10.94 degC rise; within-duty slope at duty 1.00 exactly zero. |
| M6 | Figures at a power mode other than MAXN_SUPER | **NOT MET and NOT ACHIEVABLE** without root on the board. Refused by name. |
| M7 | Figures at a second ambient | **NOT MET and NOT ACHIEVABLE** without physical access. Refused by name. |

M1 to M5 are met. They became met when the campaign's traces were placed in
`data/traces/` and the measured scripts were re-executed; nothing else in the
product had to change. M6 and M7 cannot be met by this campaign at all and are
the reason the product's own documents refuse those claims by name rather than
hedging them.

A requirement the specification did not name but the campaign exposed: **M8, a
GPU-loaded payload, is NOT MET and was never attempted.** The GPU devfreq sat at
its 306 MHz floor for 99.578 per cent of the campaign and GR3D utilisation is
not a column in the trace schema, so every figure here is a CPU-payload figure
and "the GPU was idle" is an inference rather than a measurement.

## Non-functional requirements

| # | Requirement | State |
|---|---|---|
| N1 | Every validation script behaves identically from any working directory | **MET.** Every script, four directories each, every one agreeing. |
| N2 | No absolute path in any tracked file | **MET.** Scanner in `validate_tree_hygiene.py`, which deletes its own previous output before scanning. |
| N3 | No cache directory or build artifact added by any script here | **MET.** Tree snapshotted at entry and compared at exit. |
| N4 | Ruff-clean under both installed ruff versions | **MET.** 0.16.9 and 0.17.4, over `src/ tests/ examples/ validation/`. |
| N5 | No assertion resting on `pytest.approx`'s hidden 1e-12 absolute default | **MET.** 51 `rel=` sites, 51 with an explicit `abs=`, 0 bare; suite re-run with the default forced off, 0 failures. |
| N6 | Every training or Monte Carlo run under three minutes on two cores | **MET.** Slowest validation script 20.2 s (the property hunt); the whole suite about 12 to 14 s; the measured campaign 3.0 s and the measured learned model 3.7 s. |
| N7 | Dependency ranges bounded on both sides | **MET**, but exercised at both ends only for `ruff`; see `README.md` Limitations. |
