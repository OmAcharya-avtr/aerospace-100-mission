# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-11-12

First release. Research-grade; not flight-qualified, not certified and not
approved for operational aerospace use.

### Added

- **Trace loader and binding schema** (`thermalbudget.schema`,
  `thermalbudget.trace`). The 20-column per-second CSV of the campaign, with the
  `-1` missing sentinel converted to `nan` before any unit conversion and
  counted per column, and the temperature scale **inferred** from the data
  against a declared rule rather than assumed.
- **Origin lattice** (`thermalbudget.provenance`). Two elements, `SYNTHETIC`
  absorbing, carried on every trace and every derived result. No function
  promotes a synthetic figure to a measured one; `require_measured` raises, and
  rejects a figure derived from zero inputs.
- **Equilibrium test** (`thermalbudget.equilibrium`). The slope of a linear fit
  over a declared final window against a declared threshold, with an
  AR(1)-corrected standard error reported beside the uncorrected one, a verdict
  ladder across five thresholds, and `INSUFFICIENT_SAMPLES` as a status that is
  explicitly not a pass.
- **Baseline 1, cold-burst extrapolation** (`thermalbudget.baselines`),
  implemented as the thing being tested and given the same sample hygiene,
  equilibrium test and bootstrap treatment as the ground truth.
- **Baseline 2, first-order thermal RC network** (`thermalbudget.rc_model`,
  `thermalbudget.clock_throughput`). A separable least-squares relaxation fit
  plus two estimators of tau that share no algebra with it, tau from the
  cooldown traces reported **beside** tau from the rise rather than merged, and
  the measured clock-versus-temperature and clock-to-throughput relations — each
  of which reports itself **not identifiable** rather than fitting noise when the
  clock does not move.
- **Baseline 3, measured ground truth** with a two-level (cluster) bootstrap
  across runs and within runs, an independent Student-t interval over per-run
  means, and the number of distinct run-level resample multisets printed beside
  every interval.
- **The headline**, the cold-burst overstatement with its interval, reporting
  `INDISTINGUISHABLE` when that interval contains zero.
- **Learned steady-state predictor** (`thermalbudget.predictor`), group-
  bootstrapped ridge with an uncertainty output, scored on held-out duty cycles
  against the RC baseline given the **same** warm-up window, with a random
  forest reported beside it, and a comparison rule that reports
  `NOT_DISTINGUISHABLE` for any margin inside baseline 3's run-to-run interval.
- **Declared limits and refusals** (`thermalbudget.limits`): the campaign's four
  pre-recorded limits, and eight claims the campaign cannot support, each refused
  by name.
- **CLI**: `python -m thermalbudget` with `schema`, `limits`,
  `validation-level`, `equilibrium`, `tau` and `report`. Every loading
  subcommand requires an explicit `--origin`; there is no default.
- **Synthetic fixture generator** (`tests/fixtures/synthetic_trace.py`) emitting
  schema-identical `SYNTHETIC_*` traces in two declared regimes, neither a
  default.
- **221 tests**, counted from junit XML. Eleven validation scripts, four
  examples, four PNGs.

### Measured results

The hardware campaign finished after the package was built and its 17 traces
are in `data/traces/`. Every figure below comes from a script in `validation/`
executed against them.

- **Headline: the cold-burst overstatement is +1.6277 per cent, 95 per cent
  interval [-1.0870, +4.6481]. VERDICT INDISTINGUISHABLE** — the interval
  contains zero. Burst 26160.398 iters/s, sustained 25741.404
  [25593.028, 25885.792], steady-state die 56.7085 [56.4813, 56.9959] degC.
- **The board did not throttle.** 0 of 9654 per-core clock samples across the
  three duty-1.0 soaks fell below `cpuinfo_max_freq` 1728000 kHz, over a die
  rise of 10.94 degC. The within-duty clock-versus-temperature slope at duty
  1.00 is exactly zero.
- **Baseline 2 reduces to baseline 1**, because the clock-to-throughput chain
  has no thermal signal to act on. A measured property of this board under this
  payload, not an implementation defect.
- **The learned model lost**, by 17.13 run-to-run half-widths: ridge MAE
  4.9205 degC against the RC baseline's 0.5120 degC on the same warm-up window.
  Its two-sigma coverage is 0.6111 against a calibrated 0.95.
- **The first-order model is inadequate for this board**, by two independent
  bounds and by a lag-1 residual autocorrelation of +0.732 — reported as a
  finding about the board, not as a failed check.

### Added after the traces arrived

- `validation/validate_clock_confound.py`, which decomposes the
  clock-versus-temperature relation within duty cycle and between it, and shows
  that the pooled +46110.5 kHz/degC slope is generated entirely between duty
  groups.
- Sections 1b, 1c and 8b of `validation/validate_measured_campaign.py`: the GPU
  devfreq and achieved-duty census, the per-core clock census and the governor's
  resolution, and an independent bound on the measured asymptote.
- `examples/clock_relation.py` rewritten to colour by duty cycle, draw the
  within-duty fits and label the pooled fit CONFOUNDED.

### Corrected after the traces arrived

Each is an instance of the same error class — a confident sentence about data
that had not been seen — and each is recorded rather than quietly applied. The
full list, with the superseded claims, is in `validation/VALIDATION.md`.

- `thermalbudget.baselines`: the claim that the measured cold-burst trace's
  first row carries `iters_cum = 0` (it is 19468), and then the replacement
  claim that the start-up deficit is "consistently near 25 per cent" across
  "all six load runs" (there are eight, and the deficit is 4.2 per cent at duty
  0.25).
- `thermalbudget.clock_throughput`: a claim about the whole 20 s burst run made
  from the five rows quoted in `docs/SPEC.md`, and a claim that the governor's
  steps are "far larger" than the identifiability floor made without measuring
  one (measured: 76800 kHz, 8.9 times the floor).
- `tests/conftest.py`, `tests/fixtures/synthetic_trace.py`,
  `tests/test_trace.py`: docstrings and a comment describing the measured
  regime from partial evidence.

### Known at release

- **The validation badge and `thermalbudget.VALIDATION_LEVEL` still read
  "hardware-measurement pending at build time"** and have deliberately not been
  updated. The achieved level is the mission coordinator's decision at Phase 9;
  the evidence is in `validation/VALIDATION.md`.
- The held-out-**ambient** evaluation axis is **structurally unmeasurable** in
  this campaign and has no score.
- **CPU-only.** The GPU devfreq sat at its 306 MHz floor for 99.578 per cent of
  the campaign and GR3D utilisation was never sampled, so a GPU-loaded pipeline
  is untested.
- MAXN_SUPER only, one ambient, one board, soaks of at most 720 s.
- Dependency ranges are bounded on both sides but were exercised at both ends
  only for `ruff`.

[0.1.0]: https://github.com/OmAcharya-avtr/thermalbudget/releases/tag/v0.1.0
