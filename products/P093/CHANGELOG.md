# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-11-14

First release. Status: **TESTING**.

### Added

- `powerenvelope.trace` — loader for a campaign directory, validating every
  header against the declared schema and refusing a file whose columns moved,
  with the measurement model stated in its module docstring.
- `powerenvelope.schema` — the three CSV schemas, the reject-reason enum, the
  8 mA current quantum and the 5.0 V nominal rail, in one place.
- `powerenvelope.energy` — trapezoidal energy over the whole run and restricted
  to the active windows derived by intersecting the sampler's timestamps with
  the decisions' intervals, with the inter-inference edge case documented and
  counted rather than absorbed.
- `powerenvelope.baselines` — Baseline 1 (energy per inference, `measured-active`
  and `nominal-rail`), Baseline 2 (the analytic duty-cycle model with every
  component measured, plus the measured rail-accounting ratio against load), and
  Baseline 3 (integrated whole-run energy over counted accepted decisions, with
  a between-run interval from four repeats).
- `powerenvelope.decomposition` — the headline understatement factor and its
  decomposition into the rejection factor, the duty-cycle factor and the
  nominal-versus-measured active-power error, with the residual reported
  explicitly and checked against an independently computed accounting factor.
- `powerenvelope.windows` — the window dataset and the licensed descriptor set,
  with the foreign-occupancy imputation flagged in the design matrix.
- `powerenvelope.predictor` — a random-forest mean-active-power predictor with
  an uncertainty output, the analytic and constant baselines it is scored
  against, and `assert_descriptors_licensed`, which raises `DescriptorLeakError`
  for any descriptor outside the licensed set.
- `powerenvelope.bounds` — four brackets derived from mathematics the
  integrator does not use, each with a slack that follows the conditioning of
  its own expression and has no absolute floor.
- `powerenvelope.synthetic` — a deterministic synthetic campaign generator
  writing the same CSV schema, used by the tests and by the CLI's `--simulate`
  path, labelled synthetic at the point of use and never a fallback for a
  missing measured file.
- `powerenvelope.__main__` — `manifest`, `headline`, `report` and `learned`
  subcommands, with exit status 3 for absent traces and 2 for an input error.
- The measured hardware campaign under `data/traces/`: 20 runs, 125 files,
  19 MB, 268,425 inferences and 100,789 accepted decisions over 27.05 minutes of
  board time.
- 356 tests, including a full synthetic-campaign integration test, 16 smoke
  tests over the committed measured campaign, a regression suite with every
  Hypothesis counterexample pinned as a literal, and tests that the
  measured-data refusal path exits non-zero and does not read the committed
  campaign.
- Thirteen validation scripts, flat in `validation/`, each anchored on
  `__file__`, each writing its raw output beside itself.

### Measured at release

- **Headline, reference cell `w64_d050`: F = 5.605882.** Energy per inference
  13.608877 mJ against 76.289767 mJ per useful decision; rejection factor
  2.898629, duty-cycle factor 1.933305, residual 1.000348, closure error
  0.000e+00. Across the grid F runs from 2.820 to 28.989.
- Between-run interval over the four reference repeats: 74.2009 ± 2.2198 mJ,
  2.9916 per cent relative.
- **The pre-registered expectation is NOT REFUTED on both held-out schemes.**
  The learned random forest does not beat the strongest classical predictor of
  `P_active`; it is slightly worse on both, by margins inside the between-run
  half-width. The learned component stays in the product reported as beaten.

### Dependency ranges

Both ends of every declared range were run on 2026-11-14 in an isolated virtual
environment outside the product tree, and the committed validation output was
regenerated and diffed at each end, not only the test verdict. Floors numpy
2.5.0, scipy 1.18.0, scikit-learn 1.9.0, pytest 9.1.0, hypothesis 6.170.0;
ceilings numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1, pytest 9.1.1,
hypothesis 6.170.15. Both gave 356 passed and byte-identical validation output
apart from the forest's measured fit time. `matplotlib` is outside the sweep: it
draws the screenshots and is in no reported number.

### Known at release

- **One validation check fails and is published failing.** The duty-cycle
  factor's lower bound built from the campaign floor's minimum power is violated
  on 9 of 19 pipeline cells, because the campaign's run order was not randomised
  and the floor was measured at the highest foreign load of the run. The bound
  is sound arithmetic on a false premise; it was not weakened.
  `validate_measured_headline.py` exits 1 for this reason.
- Two of the gate's five reject branches, `no_candidate` and `border`, never
  fired across 268,425 measured inferences.
- The measured `E_ramp` is negative, which is the INA3221's 512-conversion
  hardware averaging and not a physical energy deficit.
- A ramp-charging defect was found by the measured data through the two-term
  cycle-energy identity: `E_ramp` is charged per WAKE and scaled by
  wakes-per-cycle, which matters at duty 1.00 where a run has one wake and not
  38.
- `validation/validate_property_hunt.py` falsified the claim that the
  trapezoidal energy integral is invariant under a shift of the timestamps, at
  `t = [0.0, 1e-4]`, `shift = 2048.0`. The claim was replaced by a bound that
  follows the timestamp resolution, and the counterexample is pinned in
  `tests/test_regressions.py`.
