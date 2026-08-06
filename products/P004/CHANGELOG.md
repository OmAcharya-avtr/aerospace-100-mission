# Changelog

All notable changes to PassPlanner are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-08-06

Initial release.

### Added
- **Orbit / pass core** (`passplanner.frames`, `passplanner.passes`)
  - SGP4 propagation via the `sgp4` package; TEME→ECEF rotation using the
    IAU 1982 GMST polynomial, with the neglected terms and accuracy class
    documented.
  - WGS-84 geodetic↔ECEF conversion and topocentric azimuth/elevation/range
    (SEZ frame, `atan2` form for zenith conditioning).
  - `find_passes(tle, station, t0, t1, min_elev_deg)`: coarse elevation scan
    plus bisection refinement of rise/set against an elevation mask, with
    culmination refinement; `find_passes_from_position_fn` for arbitrary
    (e.g. analytic) ephemerides.
  - Two checksummed public historic TLE fixtures (ISS 2008-09-20 epoch,
    NOAA 14 1997-11-16 epoch).
- **Optical availability** (`passplanner.availability`, `passplanner.stations`)
  - Per-station climatological monthly clear-sky priors loaded from YAML.
  - `ForecastAvailability` override layer for user-supplied forecasts.
  - `expected_data(pass, availability)` = rate × duration × p_clear.
  - Example station files with clearly labelled fictional placeholder sites.
- **AI availability model** (`passplanner.mlmodel`, `passplanner.synthdata`)
  - Classical `ClimatologyBaselineModel` implemented first.
  - `PassSuccessModel`: bagged gradient-boosting ensemble with an
    ensemble-spread uncertainty output.
  - Seeded synthetic weather dataset generator (no data files committed).
  - `MODEL_CARD.md`, `DATASET_CARD.md`, calibration curve in `validation/`.
- **Scheduling** (`passplanner.scheduler`)
  - `schedule_greedy(...)` baseline and exact `schedule_ilp(...)` (PuLP/CBC)
    maximising expected delivered data under per-station and per-satellite
    no-overlap constraints, with optional setup/slew padding.
- **CLI**: `python -m passplanner plan --config scenario.yaml` printing the
  schedule table and summary statistics.
- **Examples**: pass timeline / schedule Gantt figure and an
  availability-weighted vs unweighted comparison, both writing PNGs to
  `screenshots/`.
- **Validation** (Level 2): rise/set cross-check against a dense-grid
  recomputation (worst Δ 0.0146 s), ILP vs hand-solved and exhaustive optima,
  greedy-vs-ILP gap tables, and availability-model calibration
  (Brier 0.1687 vs baseline 0.2328, ECE 0.0179).
- 106 tests covering known answers, input validation, edge cases,
  property-based checks (Hypothesis), an end-to-end integration test and two
  runtime benchmark tests.

### Known limitations
See `README.md` § Limitations — notably: no atmospheric refraction, GMST-only
Earth rotation, all-or-nothing per-pass cloud model, synthetic-only ML
training data, and single-telescope/single-terminal resource assumptions.
