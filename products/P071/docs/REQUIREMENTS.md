# Requirements — conjprob 0.1.0

Validation level 3 requires a written requirement set with each requirement
traced to the evidence that it was met. Each row below names the script that
produced the evidence; every number in the Evidence column was produced by
running that script in this session, and the raw output is committed beside it.

Status vocabulary: **MET** means the stated tolerance was achieved and the
evidence file says so. **MET WITH MEASURED LIMIT** means the requirement is
satisfied only inside a regime that was measured rather than assumed, and the
measured boundary is given. **NOT MET** means the check failed and the failure
is published.

## Functional requirements

| ID | Requirement | Evidence | Status |
|---|---|---|---|
| F-01 | Compute time of closest approach and miss distance for two states under rectilinear relative motion. | `validate_geometry.py` check 1: tca 10.000000000000000 s against a hand calculation of 10 s, miss 0.000000e+00 m. | MET |
| F-02 | Construct an encounter-plane basis and project a 3x3 position covariance onto it as an exact marginal. | `validate_geometry.py` checks 2 and 3: worst normalised perpendicularity residual 0.000000e+00 over 200 random geometries; worst relative in-plane trace error 0.000000e+00 over 200 isotropic cases. | MET |
| F-03 | Compute Pc by three independently implemented analytic methods. | `validate_analytic.py` check 2: Foster-Estes, Chan and Alfano all evaluated over a 30-cell grid against `scipy.stats.ncx2`. | MET |
| F-04 | Provide a numerical reference that shares no algebra with the analytic paths. | `validate_analytic.py` check 2: `pc_quadrature` worst relative error 5.551115e-16 against `scipy.stats.ncx2`; it uses neither the error-function reduction nor the eigendecomposition. | MET |
| F-05 | Provide a 3-D Monte Carlo reference over the relative state with confidence intervals. | `validate_montecarlo.py` checks 1 and 2: 6 of 6 intervals contain the analytic value; measured coverage 0.960000 over 300 seeds against a nominal 0.95. | MET |
| F-06 | Report the disagreement between methods as a primary output, not a footnote. | `worked_example.py`: max absolute spread 1.655544e-05, max relative spread 4.460264e-03, worst method `small_body`; `ConjunctionReport` has no scalar `pc` attribute, asserted in `tests/test_report.py`. | MET |
| F-07 | Report the validity regime of each method as a measured curve. | `validate_regimes.py` check 2: boundary tables at 1e-2 and 1e-3 relative tolerance over 30 cells per method. | MET |
| F-08 | Exhibit the dilution region explicitly as a Pc-against-covariance curve with its non-monotonic turn. | `validate_dilution.py` check 3: maximum of the sampled curve 7.20571701e-03 at scale 0.482318, strictly increasing left of it and strictly decreasing right of it over 241 sampled scales; the golden-section maximum for the same case is 7.209827e-03 at scale 0.473905 (`worked_example.py`). | MET |
| F-09 | Raise a dilution warning from the library itself, not from the caller inspecting a flag. | `validate_dilution.py` check 5: `DilutionWarning` raised on both dilution-branch cases and not on the one case that is not on the branch. | MET |
| F-10 | Measure where the rectilinear 2-D formulation fails at long encounter durations. | `validate_nonlinear.py` checks 1 and 2: finite-window ratio 0.913176 at half a crossing time; 2-D error reaches 1 per cent at curvature parameter 3.288387 and 10 per cent at 6.702773e+02. | MET |
| F-11 | Provide a learned screening classifier with an uncertainty output, benchmarked against analytic screens on recall at fixed false-positive rate. | `validate_screening.py` checks 2 and 5: recall tables for four screens at three operating points on three regimes; per-tree dispersion ratio between misclassified and correct cases 4.894, 3.121 and 7.932. | MET |
| F-12 | Accept a state from a two-line element set as an optional entry point. | `validate_tle.py` check 1: worst position error 6.819541e-06 m and worst velocity error 7.704807e-07 m/s against the published SGP4 verification output over 31 epochs. | MET |
| F-13 | Provide a command-line interface. | `validate_cli.py` check 1: 9 invocations, exit statuses and required output fragments all as expected. | MET |

## Accuracy requirements

| ID | Requirement | Evidence | Status |
|---|---|---|---|
| A-01 | Foster-Estes agrees with an exact distribution function to 1e-9 relative. | `validate_analytic.py` check 2: worst 6.661338e-16 against `scipy.stats.ncx2`. | MET |
| A-02 | Chan's series at its recommended order agrees with an exact distribution function to 1e-9 relative for a circular covariance. | `validate_analytic.py` check 2: worst 1.110223e-15. | MET |
| A-03 | Alfano's fixed-node rule agrees to 5e-4 relative at 2001 nodes, with its convergence order measured rather than assumed. | `validate_analytic.py` checks 2 and 4: worst 1.479235e-04; measured order 1.5004 against Simpson's nominal 4; 1.468000e-04 at the package default of 201 nodes. | MET WITH MEASURED LIMIT |
| A-04 | The Chan series order required for convergence is characterised, not assumed. | `validate_analytic.py` check 3: required order grows with both M and R/sigma; the default order of 3 is 1.515e-01 wrong at M = 12, R/sigma = 0.5 and 1.000e+00 wrong at M = 12, R/sigma = 3. | MET |
| A-05 | The closed-form maximum Pc over covariance scale reproduces R^2 / (e d^2). | `validate_dilution.py` check 1: worst relative difference 4.440892e-16 against a 1e-13 tolerance. | MET |
| A-06 | The covariance scale at which Pc is maximised equals M / sqrt(2). | `validate_dilution.py` check 4: worst relative difference 6.257130e-04 against a 1e-3 tolerance. | MET |
| A-07 | Pc is invariant under a rigid rotation of the whole problem and under a common rescaling of every length. | `validate_geometry.py` checks 4 and 5: worst rotation-induced relative change 4.070440e-10 over 32 rotations per case; worst rescaling-induced change 5.995204e-15 over factors from 1e-4 to 1e4. | MET |
| A-08 | The Monte Carlo interval attains its nominal coverage. | `validate_montecarlo.py` check 2: 0.960000 measured, interval on the coverage itself [0.931171, 0.979163], nominal 0.95. | MET |
| A-09 | The Monte Carlo interval width scales as the inverse square root of the sample count. | `validate_montecarlo.py` check 1: measured log-log slope -0.505200 against -0.5, tolerance 0.05. | MET |
| A-10 | The discretisation bias of the curved Monte Carlo estimator is measured, not assumed negligible. | `validate_nonlinear.py` check 3: the package default of 129 nodes and 4 refinements agrees with a 513-node, 6-refinement grid to +0.000000e+00 relative. | MET |
| A-11 | The learned screen's predicted probabilities are calibrated to within 0.15 on a reliability table. | `validate_screening.py` check 4: worst bin gap 0.272486. | **NOT MET** — published and not repaired; see `MODEL_CARD.md`. |

## Regime-reporting requirements

| ID | Requirement | Evidence | Status |
|---|---|---|---|
| R-01 | Report the hard-body-size boundary of the leading-order approximation. | `validate_regimes.py` check 2: 1 per cent boundary at R/sigma_eff from 0.2818 (aspect 1, M = 1) to 0.01958 (aspect 30, M = 4); worst error anywhere on the grid 3.933262e+00. | MET |
| R-02 | Report the aspect-ratio boundary of Chan's equivalent-area substitution. | `validate_regimes.py` check 2: exact at aspect 1 (tolerance never exceeded), boundary 0.5609 at aspect 2 falling to 0.02024 at aspect 30; worst error anywhere on the grid 2.906285e+01. | MET |
| R-03 | Show that Foster-Estes and the quadrature reference hold everywhere on the swept grid. | `validate_regimes.py` checks 1 and 2: Foster-Estes worst error 1.408800e-13 and the 1e-3 tolerance never exceeded in any of 30 cells. | MET |
| R-04 | Express the curvature breakdown in a quantity an operator can act on. | `validate_nonlinear.py` check 4: the measured curvature parameters correspond to relative speeds of 0.810120 m/s (1 per cent) and 0.056743 m/s (10 per cent) for a declared 200 m / 300 m low Earth encounter. | MET |

## Reproducibility requirements

| ID | Requirement | Evidence | Status |
|---|---|---|---|
| P-01 | Every validation script runs from any working directory. | All ten scripts executed from the filesystem root, the system temporary directory, the product root and `validation/`; exit 0 from all forty runs. | MET |
| P-02 | No committed file contains an absolute path. | A recursive scan of every tracked file for the build machine's home, root and site-packages prefixes returns no match outside the gitignored `.hypothesis/` cache. | MET |
| P-03 | The synthetic population is regenerated deterministically from a committed script and a fixed seed; no model artefact is committed. | `tests/test_dataset.py::test_population_is_reproducible_for_a_seed`; `conjprob.screening.ScreeningClassifier` is trained in-process on every run. | MET |
| P-04 | Test counts are taken from junit XML, not from pytest stdout. | `junit.xml`: `tests="269"`, `failures="0"`, `errors="0"`, `skipped="0"`, cross-checked against 269 `testcase` elements. | MET |
| P-05 | Every dependency is bounded on both sides at the version it was measured against. | `pyproject.toml`: `numpy>=2.5,<2.6`, `scipy>=1.18,<1.19`, `scikit-learn>=1.9,<1.10`, `pyyaml>=6.0,<6.1`, `sgp4>=2.27,<2.28`. | MET |
| P-06 | Every counterexample found by property-based testing is pinned as a named regression test with literal inputs. | `tests/test_regressions.py`: seven pinned cases. | MET |

## Explicit non-requirements

These are stated so that nobody infers them from the requirements above.

- **No output of this package is an input to a manoeuvre decision.** The package
  computes a probability under a covariance the user supplied and reports how
  much the methods disagree about it. It does not decide anything.
- **No covariance is produced.** A covariance must come from orbit determination.
  The TLE entry point returns a state and stops.
- **No catalogue is screened.** There is no conjunction detection, no catalogue
  ingest and no screening volume. Operational conjunction assessment is done
  with CARA- and CSpOC-grade tooling against catalogues this product does not
  have.
- **No curved-trajectory Pc method is implemented.** The curvature breakdown is
  measured and the established remedy (Patera 2001) is cited, not implemented.
- **No covariance realism assessment.** Whether the supplied covariance is right
  is outside the package, and it is the question that matters most in practice.
