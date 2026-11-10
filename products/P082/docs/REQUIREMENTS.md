# Requirements — sitediverse (P082)

Level 3 validation requires a stated requirement set with each requirement
traced to the code that implements it and the evidence that it holds. Every
"verified by" entry below names a committed file in this repository.

Status key: **MET** — verified by a check that ran in the build session and
passed. **MET WITH A STATED LIMITATION** — the requirement holds within a
boundary that is written down. **NOT MET** — recorded rather than removed.

## 1. Functional requirements

| id | requirement | implemented in | verified by | status |
| --- | --- | --- | --- | --- |
| F-01 | Compute the joint unavailability of a station set under the independence assumption, as a product of long-term marginal cloud probabilities. | `independence.product_of_marginals`, equation (14) | `tests/test_independence.py`; `validate_overstatement_output.txt` | MET |
| F-02 | Compute the same under epoch-conditional independence, honouring a time-varying marginal climatology. | `independence.conditional_product`, equation (15) | `tests/test_independence.py` hand-built co-varying and anti-varying cases | MET |
| F-03 | Compute the directly observed joint unavailability from the record. | `empirical.joint_rate`, equation (20) | `tests/test_empirical.py`; cross-checked against the matrix form on all 190 pairs | MET |
| F-04 | Report the overstatement as the headline: the ratio and the difference between F-01 and F-03. | `independence.Overstatement`, CLI `overstate` | `validate_overstatement_output.txt`; six networks, factors 0.9552 to 22.9809 | MET |
| F-05 | Decompose the overstatement exactly into a stationarity term and a spatial-correlation term. | `Overstatement.comodulation_term`, `.correlation_term`, equation (16) | residual **0.00e+00** on all six networks, `validate_overstatement_output.txt` | MET |
| F-06 | Compute the Gaussian-copula joint unavailability of a station pair in closed form. | `copula.bivariate_orthant` through the Plackett identity, equations (8) and (9) | agreement with SciPy's independent implementation to **1.088539e-16** absolute | MET |
| F-07 | Compute the same for more than two stations. | `copula.network_joint_monte_carlo`, equation (10) | agrees with the exact pair quadrature to 1.9705 sigma | MET WITH A STATED LIMITATION — Monte Carlo, not closed form; the standard error is reported with every figure and it is never called closed form |
| F-08 | Fit a two-parameter exponential correlation-length model over station separation. | `correlation.fit_exponential`, equation (23) | recovers a noiseless exponential to 1e-06 relative; function error 0.014631 rms against the generator on the correctly-specified record | MET |
| F-09 | Report a strengthened baseline beside the specified one. | `correlation.fit_per_regime`, equation (24) | both reported in `validate_correlation_output.txt` and by CLI `correlate`; margin 0.000674 | MET |
| F-10 | Attach bootstrap confidence intervals to the empirical reference. | `empirical.bootstrap_joint_rate`, equation (21) | `validate_empirical_output.txt` | MET |
| F-11 | Expose a learned spatio-temporal correlation model with an uncertainty output. | `learned.LearnedCorrelationModel`, equations (35) to (38) | `tests/test_learned.py` asserts structurally that no bare point-estimate method exists | MET |
| F-12 | Score the learned model on held-out SITES and held-out TIME separately. | `splits`, `study.run_study`, equations (29) and (41) | `validate_learned_model_output.txt`; four margins, never merged | MET |
| F-13 | Propagate the record's sampling uncertainty and report it beside every availability figure. | `uncertainty`, equations (27) and (28); `FigureWithUncertainty` | every table in `validate_overstatement_output.txt` and in CLI `overstate` | MET |
| F-14 | Provide a command line. | `sitediverse.__main__`, eight subcommands | `validate_cli_output.txt`, 14 subcommand invocations and 6 bad-argument cases | MET |

## 2. Numerical requirements

| id | requirement | implemented in | verified by | status |
| --- | --- | --- | --- | --- |
| N-01 | A joint unavailability must never be computed through a subtraction of two nearly equal probabilities. | `copula.correlation_excess`; `independence.product_of_marginals` in log space; `availability_nines` from the unavailability | the integral form keeps full relative precision at `rho = 1e-20` where the subtraction returns exactly 0.0 | MET |
| N-02 | Every computed joint unavailability must satisfy a bound derived from mathematics the implementation does not use. | `bounds.frechet_hoeffding`, equation (17) | **0 violations in 380 comparisons** over 190 pairs by two routes, and 0 in 18 over six networks by three routes | MET |
| N-03 | A bound-check slack must follow the conditioning of the bound expression. | `bounds.bound_slack`, equation (19) | the relative slack rises by **8.333e+15** over sixteen decades while the absolute slack varies by **1.200000** | MET |
| N-04 | A bound that falls below its own slack must be reported as uninformative, not as a pass. | `FrechetBounds.lower_informative` | printed for every case in `validate_bounds_output.txt`; informative for 8 of 190 pairs | MET |
| N-05 | No test may assert bit-exact equality over a floating-point pipeline. | the whole suite | `validate_approx_audit_output.txt`: 90 of 90 `approx` calls carry an explicit `abs=` | MET |
| N-06 | No test may assert equality of a boolean produced by a strict threshold comparison. | `tests/test_field.py`; `Record.margin_to_threshold` | every flipped indicator is shown to have a margin inside the perturbation that flipped it | MET |
| N-07 | No assertion may depend on the hidden `pytest.approx` absolute default. | the whole suite | **0 of 433**, measured by re-running the suite with a bare `rel=` made to raise | MET |
| N-08 | An n-station correlation matrix assembled from pair predictions must be positive definite, and any repair must be reported. | `correlation.matrix_from_pairs`, equation (26) | a negative control constructs an indefinite matrix and the projection fires | MET |

## 3. Validation requirements

| id | requirement | verified by | status |
| --- | --- | --- | --- |
| V-01 | Every analytic baseline is validated against the simulation BEFORE the learned model is benchmarked. | `validate_correlation.py` runs and reports first; `study.run_study` fits both baselines before touching the learned model | MET |
| V-02 | Every negative control is anchored on a true value, not on an output. | `validate_negative_controls.py`; every offset is a multiple of the check's own criterion | MET |
| V-03 | Every negative control is asserted to have fired as declared. | **26 declared; 15 must fire and all 15 did; 11 must not and none did** | MET |
| V-04 | Every validation script exits zero from any working directory. | `probe_working_directories.py`: **0 non-zero exits in 66 runs** across four directories | MET WITH A STATED LIMITATION — `validate_approx_audit.py` is probed in its declared paths-only mode; the reason and what is still exercised are in that script's docstring and in VALIDATION.md |
| V-05 | No committed file contains an absolute host path. | `validate_cli.py` part 4: **0 offenders** in the text files scanned | MET |
| V-06 | Running a validation script adds no cache directory and no bytecode file. | `validate_cli.py` part 3a: **0 added** | MET |
| V-07 | git tracks no cache directory, bytecode file or build artifact. | `validate_cli.py` part 3b | NOT MET AT BUILD TIME — the product was uncommitted when measured, so the check examined 0 files and is recorded as **VACUOUS** in the output. Release-gate check 9 enforces the same claim independently |
| V-08 | Test counts come from junit XML, never from pytest stdout. | `validate_approx_audit.py` parses the `testsuite` element and cross-checks `testcase` elements: **433 and 433** | MET |
| V-09 | Every scanner is run twice in a row from a clean tree and the two runs agree. | `probe_working_directories.py` and `validate_cli.py` were each run twice during the build | MET |
| V-10 | Every figure in README.md is produced by a committed example. | `validate_examples.py`: 7 examples, 7 figures, all rewritten and all above 20 KiB | MET |

## 4. Non-functional requirements

| id | requirement | verified by | status |
| --- | --- | --- | --- |
| P-01 | Any training or Monte Carlo run completes in under three minutes on two cores. | learned fit **0.904 s**; whole study **6.521 s**; 400000-draw Monte Carlo about 0.05 s | MET |
| P-02 | Every validation script completes inside the release gate's 600-second limit. | slowest is `probe_working_directories.py` at **281.5 s** | MET |
| P-03 | The package is ruff-clean. | clean under **both** ruff 0.16.9 and ruff 0.17.4 over `src/ tests/ examples/ validation/` | MET |
| P-04 | Every dependency is bounded on both sides. | `pyproject.toml` | MET |
| P-05 | The suite is run at both ends of every declared dependency range. | `ruff` only | NOT MET — the build environment is a shared pre-pinned interpreter this product is forbidden to modify, and installing a second version of any other dependency would have required running pip. Recorded in README.md under Limitations and in a `pyproject.toml` comment rather than claimed |
| P-06 | No network connection or data download is required to reproduce any number. | the record is regenerated from a declared integer seed; no fixture exceeds a kilobyte | MET |
| P-07 | Matplotlib uses the Agg backend and never calls `show`. | `examples/_common.py` | MET |

## 5. Scope boundaries, stated as requirements that are deliberately NOT met

| id | statement |
| --- | --- |
| S-01 | **No real climatology.** Every number is measured on a synthetic record from a declared generator. `DATASET_CARD.md` states what that costs. |
| S-02 | **No propagation physics.** Cloud blockage is a binary indicator. There is no scintillation, fade depth, aperture averaging, elevation dependence or pointing loss. |
| S-03 | **One copula family.** The Gaussian copula is a modelling assumption, not a consequence of Sklar's theorem, which gives no unique copula for discrete margins. |
| S-04 | **No flight qualification of any kind.** Research grade: not flight-qualified, not certified, not approved for operational aerospace use. |
