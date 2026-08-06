# BeamTwin — Requirements and Verification Matrix

Version 0.1.0 · Validation Level 3 (Professional, at v0.1 depth) · Status: TESTING

Requirement IDs are stable. `F` = functional, `N` = non-functional.
Verification methods: **T** = automated test, **A** = analysis/hand-calculation,
**D** = demonstration (runnable example), **I** = inspection.

---

## 1. Functional requirements

| ID | Requirement | Rationale | Verification method | Evidence |
|----|-------------|-----------|---------------------|----------|
| **R01** | The system shall compute a deterministic FSO link budget comprising transmit power, transmit/receive optics efficiencies, Gaussian-beam geometric capture on a circular aperture, static pointing loss, atmospheric attenuation, and the resulting received power and margin. | Core capability. | T, A | `tests/test_budget.py::TestComputeBudget`; `validation/v1_budget_handcheck.txt` (all terms, max deviation 0.0) |
| **R02** | Gaussian-beam propagation shall follow Saleh & Teich (2007) Eqs. (3.1-8), (3.1-11), (3.1-21): `w(z)=w0*sqrt(1+(z/z_R)^2)`, `z_R=pi*w0^2/lambda`, `theta=lambda/(pi*w0)`, with the far-field and waist limits recovered exactly. | Physical correctness. | T, A | `tests/test_budget.py::TestGaussianBeam` (9 tests incl. waist, Rayleigh-range and far-field limits); `tests/test_properties.py::TestGaussianBeamProperties` |
| **R03** | Geometric capture on a centred circular aperture shall be `eta = 1 - exp(-2a^2/w^2)`, recovering `eta -> 2a^2/w^2` for `a << w` and `eta -> 1` for `a >> w`. | Physical correctness. | T, A | `tests/test_budget.py::TestGeometricCapture` (6 tests); validation V1 step 3 |
| **R04** | The system shall provide a Kim-model helper converting visibility to attenuation per Kim et al. (2001), SPIE 4214, including the wavelength-independent dense-fog branch (`V <= 0.5 km`, `q = 0`). | Usability: visibility is the commonly available meteorological input. | T, A | `tests/test_budget.py::TestKimModel` (6 tests); validation V1 step 8 (0.630817 dB/km at V=7 km, 1550 nm) |
| **R05** | The system shall model scintillation as lognormal irradiance with scintillation index from the plane-wave Rytov variance `sigma_R^2 = 1.23 Cn2 k^(7/6) L^(11/6)` (Andrews & Phillips 2005), preserving `E[I] = <I>`. | Core stochastic model. | T, A | `tests/test_channel.py::TestRytovVariance` (6 tests incl. exponent checks); `test_mean_irradiance_is_preserved_by_scintillation` (within 1 % at 4e5 samples) |
| **R06** | The system shall model per-axis Gaussian pointing jitter producing a random transverse displacement and pointing-loss factor, and shall agree with the closed form `E[L_p] = 1/(1 + 4 sigma_d^2/w^2)` in the jitter-only limit. | Core stochastic model; analytic cross-check. | T, A | `validation/v2_limit_cases.txt` V2b — 4 jitter levels, relative error 8.4e-5 to 1.0e-3, all inside the 4-sigma MC tolerance |
| **R07** | The system shall produce Monte Carlo samples of received power combining scintillation and jitter, vectorised with numpy and fully seeded. | Performance + reproducibility. | T, D | `tests/test_channel.py::TestMonteCarlo` (12 tests); `validation/v3_performance.txt` (1.3e7 samples/s peak) |
| **R08** | The system shall report fade probability `P(P_rx < sensitivity)` with a 95 % Wilson score confidence interval, plus fade-margin percentiles and mean/variance. | Decision-relevant output with honest uncertainty. | T, A | `tests/test_stats.py::TestFadeProbability`, `::TestMarginStatistics`; `validation/v4_uncertainty.txt` U1 (empirical/binomial std ratio 0.906–1.035) |
| **R09** | The system shall provide a closed-form lognormal fade-probability baseline, and Monte Carlo shall agree with it within the MC confidence interval in the scintillation-only limit. | The ML surrogate must be benchmarked against a classical baseline (mission §11). | T, A | `validation/v2_limit_cases.txt` V2a — 5 margins from 2 to 10 dB, analytic value inside the MC 95 % CI in **5/5** cases |
| **R10** | The system shall provide an ML surrogate predicting fade probability from link parameters, trained on the twin's own seeded Monte Carlo output, exposing an uncertainty output alongside the point estimate, and flagging extrapolation outside the training domain. | AI product requirement; uncertainty is mandatory. | T, D | `tests/test_surrogate.py` (35 tests); `MODEL_CARD.md`; `validation/surrogate_benchmark.txt` |
| **R11** | Every invalid scenario input (missing file, malformed YAML, unknown key, wrong type, out-of-range physics value, conflicting attenuation specification) shall raise `ScenarioError` naming the offending key; the CLI shall convert this to exit code 2 with a one-line message and no traceback. | Error-handling policy (see §3). | T | `tests/test_scenario.py::TestFailureModes` (25 tests); `tests/test_cli.py::TestModuleInvocation::test_python_dash_m_bad_file_exit_code_2_no_traceback` |
| **R12** | The system shall load scenarios from YAML and provide `python -m beamtwin run` (text + JSON report) and `python -m beamtwin sweep` (parameter sweep with PNG output). | Usability / integration. | T, D | `tests/test_cli.py` (17 tests); `examples/link_10km.yaml`; `screenshots/fade_vs_range.png` |

## 2. Non-functional requirements

| ID | Requirement | Rationale | Verification method | Evidence |
|----|-------------|-----------|---------------------|----------|
| **R13** | An end-to-end scenario report with 2e5 Monte Carlo samples shall complete in under 5 s on a 2-core machine. | Interactive usability. | T, D | `validation/v3_performance.txt` — measured **0.0169 s**; bound enforced by `tests/test_performance.py::test_scenario_report_under_5s` |
| **R14** | Monte Carlo sampling throughput shall exceed 1e5 samples/s. | Makes 1e5–1e6-sample studies practical. | T, D | `validation/v3_performance.txt` — measured **1.31e7 samples/s** peak; bound enforced by `tests/test_performance.py::test_monte_carlo_throughput_above_100k_per_second` |
| **R15** | Dataset generation plus surrogate training shall complete in under 2 minutes on 2 cores, deterministically from committed seeds. | Guide compute budget; reproducibility. | T, D | Measured: dataset 12.3 s (4000 scenarios × 5e4 samples, seed 42) + training 5.8 s = **18.1 s**; `tests/test_surrogate.py::test_deterministic_for_fixed_seed` |
| **R16** | Identical seed and scenario shall produce byte-identical reports; drawing more samples shall extend rather than change the sample stream. | Reproducibility is a Level-3 requirement. | T | `tests/test_regression.py::TestReproducibility`; `tests/test_cli.py::test_run_reproducible_across_invocations` |
| **R17** | Numerical behaviour shall be pinned by regression tests against recorded seeded outputs. | Detects silent numerical drift. | T | `tests/test_regression.py` — 12 pinned values incl. exact fade count (55/50000) and raw sample stream |
| **R18** | The source shall be ruff-clean at line length 100, with type hints and unit-carrying docstrings on all public APIs. | Maintainability. | I | `ruff check src/ tests/` → "All checks passed!"; inspection of module docstrings |
| **R19** | Every physics equation in code and docs shall carry source, units, assumptions, and validity range; model validity flags shall be surfaced in output when violated. | Engineering honesty (binding guide rule). | I, T | Module docstrings in `src/beamtwin/*.py`; `weak_regime_valid` flag surfaced — `tests/test_scenario.py::test_weak_regime_warning_in_text` |

---

## 3. Error-handling policy

1. **Input validation at construction.** `LinkParams` and `ChannelParams` validate in `__post_init__`. Invalid values raise `ValueError`; wrong types raise `TypeError`. Both name the parameter and give the offending value.
2. **Unit-mistake guards.** Values that are physically impossible but arise from common unit errors are rejected with a message that names the suspected mistake — wavelength outside [100 nm, 20 µm] ("check units (metres expected)"), `Cn2 > 1e-11` ("check units"), `tx_power_dbm > 50` ("implausible for the FSO terminals this model covers").
3. **Scenario errors are one class.** Every problem with a scenario file raises `ScenarioError` (a `ValueError` subclass) identifying the offending key. Unknown keys are rejected rather than ignored, so typos cannot silently select a default.
4. **CLI never shows a traceback for user error.** `main()` catches `ScenarioError`, `ValueError`, and `TypeError`, writes `error: <message>` to stderr, and returns exit code 2. Exit code 0 means success; a traceback would indicate a genuine internal defect.
5. **Resource guards.** `n_samples > 2e7` is rejected before allocation to prevent multi-GB allocations.
6. **Model-validity flags, not silent extrapolation.** Conditions that invalidate a model are reported, not suppressed: `weak_regime_valid=False` when `sigma_R^2 >= 1`, `margin_negative=True` when the link fails deterministically, and `extrapolating=True` when a surrogate query lies outside the training domain. All three appear in the text report.
7. **No silent numerical fallback.** The surrogate raises `RuntimeError` when unfitted and `FileNotFoundError` (naming the training script) when the model file is absent; it never returns a placeholder value.

---

## 4. Requirement coverage summary

| Requirement | R01 | R02 | R03 | R04 | R05 | R06 | R07 | R08 | R09 | R10 | R11 | R12 | R13 | R14 | R15 | R16 | R17 | R18 | R19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Verified | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |

All 19 requirements are verified by the evidence listed above (251 automated tests, 4 validation scripts, 3 runnable examples). No requirement is waived or deferred.

**Caveat on the meaning of "verified".** Verification here means *internal consistency*: agreement with the cited closed-form results, self-consistency of the Monte Carlo, and reproducibility. No requirement has been verified against measured hardware data from a physical FSO link — see `validation/VALIDATION.md` §6 and README Limitations.
