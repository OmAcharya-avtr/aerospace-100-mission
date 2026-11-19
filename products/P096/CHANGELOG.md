# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.1.0 - 2026-11-19

First release. Research-grade; not flight-qualified, not certified, and not
approved for operational aerospace use.

### Added

- `handoverloss.sequence`: the declared four-phase handover timeline
  (acquisition, protocol restart, drain with a linear egress ramp) with
  validation at construction, in SI units.
- `handoverloss.acquisition`: the acquisition-time distribution, a lognormal
  body with a geometric retry tail, with exact closed-form moments and a
  Gauss-Legendre cross-check.
- `handoverloss.baselines`: baseline 1, the acquisition-only assumption,
  implemented at full strength as the thing being refuted; and baseline 2, the
  analytic fluid buffer-drain and restart model, in closed form, evaluated in
  headroom space.
- `handoverloss.trace`: the discrete-slot tail-drop simulator, vectorised over
  handover events, with every dropped bit tagged by the phase it was dropped in.
- `handoverloss.accounting`: `attribute`, baseline 3, returning an
  `AttributionReport` with a per-mechanism bootstrap interval, the dominant
  mechanism, whether the ordering is DECIDED, and the decomposition residual
  between the phase-bucket accounting and the mass-conservation accounting as a
  reported number with its conditioning-scaled tolerance.
- `handoverloss.bootstrap`: event-level percentile bootstrap. The resampling
  unit is the handover event and is asserted on every returned object.
- `handoverloss.bounds`: an independently derived bracket on the total from
  conservation and capacity counting alone, importing no model module of this
  package.
- `handoverloss.corpus`: loader for the committed corpus with a sha256 check,
  and the two held-out splits (station pairs, time).
- `handoverloss.predictor`: a random-forest predictor of the three
  per-mechanism losses with an across-tree uncertainty output and an ensemble
  dominance vote share, plus the scoring harness both baselines run through.
- `python -m handoverloss` with `attribute`, `baselines`, `learned` and
  `sequence` subcommands.
- `data/generate_corpus.py` and the committed corpus of 1536 synthetic
  handovers over 16 station pairs and 8 epochs, hashed in
  `data/corpus/manifest.json`.
- Four examples writing to `screenshots/`, ten validation scripts with their
  committed raw output, `validation/VALIDATION.md`, `MODEL_CARD.md` and
  `DATASET_CARD.md`.

### Measured in this release

- The analytic fluid model (baseline 2) beats the learned predictor on BOTH
  held-out splits: 0.653646 against 0.390625 on held-out station pairs, and
  0.682292 against 0.600694 on held-out time. Published as the finding.
- A Hypothesis property test falsified the first, occupancy-space form of the
  closed-form decomposition at a nearly full buffer. The decomposition was moved
  into headroom space; the counterexample is pinned in
  `tests/test_regressions.py`.
- `validation/validate_acquisition_model.py` found the normal quantile
  approximation used for baseline 1's conservative setting wrong by a factor of
  2.032448. It was replaced with a memoised exact quantile.
