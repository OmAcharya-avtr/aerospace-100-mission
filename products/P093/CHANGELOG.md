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
- `powerenvelope.energy` — trapezoidal energy over the run window
  `[t0_mono, t1_mono]` exactly, with `P` interpolated at both ends, and over the
  declared per-cycle active and sleep phases that partition that same window,
  again interpolated at every boundary. The sampler's own span is wider than the
  run on purpose and is exposed separately as a diagnostic. The inter-inference
  edge case is documented and counted rather than absorbed, and the phase
  occupancy `T_act/(N·L)` is reported as a diagnostic rather than as a factor.
- `powerenvelope.baselines` — Baseline 1 (energy per inference, `measured-active`
  and `nominal-rail`), Baseline 2 (the analytic duty-cycle model with every
  component measured, plus the measured rail-accounting ratio against load), and
  Baseline 3 (integrated whole-run energy over counted accepted decisions, with
  a between-run interval from four repeats).
- `powerenvelope.decomposition` — the headline understatement factor and its
  **exact** decomposition into three factors — the rejection factor, the
  duty-cycle factor and the nominal-versus-measured active-power error — with no
  residual term, because `E_active` and `E_sleep` partition the same run window
  the numerator integrates. The closure error of the three and the partition's
  own closure are both reported rather than claimed.
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
- 360 tests, including a full synthetic-campaign integration test, 16 smoke
  tests over the committed measured campaign, a regression suite with every
  Hypothesis counterexample pinned as a literal, and tests that the
  measured-data refusal path exits non-zero and does not read the committed
  campaign.
- Thirteen validation scripts, flat in `validation/`, each anchored on
  `__file__`, each writing its raw output beside itself.

### Measured at release

- **Headline, reference cell `w64_d050`: F = 5.544866.** Energy per inference
  `E_active/N` = 13.608386 mJ against `E_total/A` = 75.456680 mJ per useful
  decision; rejection factor 2.898629, duty-cycle factor 1.912927, rail factor
  1.000000, closure error 0.000e+00 with no residual term. Across the grid F
  runs from 2.791586 (`w96_d100`) to 28.597765 (`w48_d010`).
- Worst partition closure `|E_active + E_sleep − E_total| / E_total` over the 19
  pipeline cells: **8.114e-05** on `w96_d100`, against a declared 1e-4.
- Phase-occupancy diagnostic on the reference cell: the inferences occupy
  37.989458 s of the 38.000000 s declared active window, occupancy 0.999723, a
  shortfall of 10.542 ms. The loop will not start an inference that would run
  past a phase boundary, so it undershoots the phase; it is not a factor of F.
- Between-run interval over the four reference repeats: 73.4156 ± 2.1689 mJ,
  2.9543 per cent relative.
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
hypothesis 6.170.15. Both gave 360 passed, and the regenerated validation output
differs only in the environment stanza that prints the installed versions and in
two wall-clock readings, the suite duration and the forest's measured fit time.
`matplotlib` is outside the sweep: it draws the screenshots and is in no
reported number.

### Known at release

- **One validation check is FALSIFIED and is published as a falsification.**
  The duty-cycle factor's lower bound built from the campaign floor's minimum
  power is violated on exactly 7 of the 19 pipeline cells —
  `w64_d050_permissive`, `w64_d050_rep2`, `w64_d050_rep3`, `w64_d050_rep4`,
  `w96_d010`, `w96_d025`, `w96_d050` — because the campaign's run order was not
  randomised and the floor was measured at the highest foreign load of the run.
  The bound is sound arithmetic on a false premise; it was not weakened, no cell
  was excluded and the check was not skipped. Those seven cells are pinned as
  literals in `validate_measured_headline.py`, which reports `FALSIFIED` as its
  finding and exits 0 on that set, non-zero if it ever changes. The three
  duty-1.00 cells are outside the set because a duty-1.00 run has no declared
  sleep phase and the bound degenerates to `factor >= 1`.
- Two of the gate's five reject branches, `no_candidate` and `border`, never
  fired across 268,425 measured inferences.
- The measured `E_ramp` is negative on 12 of the 15 grid cells, down to
  −202.1 mJ per wake, which is the INA3221's 512-conversion hardware averaging
  and not a physical energy deficit. The 0.15 s settling window was declared
  before the campaign ran and was not re-chosen after seeing this.
- A ramp-charging defect was found by the measured data through the two-term
  cycle-energy identity: `E_ramp` is charged per WAKE and scaled by
  wakes-per-cycle, which matters at duty 1.00 where a run has one wake and not
  38.
- `validation/validate_property_hunt.py` falsified the claim that the
  trapezoidal energy integral is invariant under a shift of the timestamps, at
  `t = [0.0, 1e-4]`, `shift = 2048.0`. The claim was replaced by a bound that
  follows the timestamp resolution, and the counterexample is pinned in
  `tests/test_regressions.py`.
- **The energy denominator was corrected before release.** An earlier
  implementation integrated the *sampler's* full span rather than the run window
  `[t0_mono, t1_mono]`. Because the campaign's sampler deliberately brackets the
  run — starting 0.5 s early and running `duration + 1.0` s — that charged the
  lead-in and tail-out to the decisions and overstated `E_total` by 1.1041 per
  cent on the reference cell. Every measured figure in this release is computed
  over the run window, and the published headline is the smaller, correct
  number. The synthetic generator was also corrected to bracket its run the same
  way, because the previous fixture, whose sampler span equalled its run, could
  not have caught this class of defect. As a consequence `T_act/(N·L)` is no
  longer a factor of the decomposition but a separately reported diagnostic, and
  the loop is shown to *undershoot* the declared active phase by 10.542 ms
  rather than overshoot it.
