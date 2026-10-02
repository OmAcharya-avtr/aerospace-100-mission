# Changelog

## 0.1.0 — 2026-10-02

Initial release.

- **Synthetic star catalogue and its preparation.** Whole-sky counts from a power law in
  magnitude (Eq. C1, `N(<m) = 4800·10^0.52(m−6)`, sampled by inverse transform, Eq. C2) with
  isotropic positions, and close-pair removal that deletes **both** members of an unresolved
  pair because the blended centroid belongs to neither. Checked against Eq. C1 to 8.2e−05,
  against Eq. C2 by Kolmogorov–Smirnov (D = 1.0489e−02 against a 5 % critical 1.4552e−02),
  for isotropy against an empirical null (dipole 8.7607e−03 against a 95th percentile of
  2.3582e−02), and against Eq. C3 for the close-pair count. **No fit to a real catalogue was
  performed and none is bundled**; `DATASET_CARD.md` states what that bounds.
- **Pair table with three indexes over the same pairs**, because the three queries the
  matcher makes are different: sorted separations for "which pairs are this far apart",
  per-star adjacency searched by a composite `10·star + θ` key for "which stars are this far
  from star s", and an ordered-pair key for "how far apart are these two". Sorted arrays plus
  `searchsorted` rather than Mortari and Neta's k-vector, which is a documented choice and
  not an oversight: the whole query vectorises in NumPy. All four query paths verified
  **exactly** equal to brute force on a 796-star catalogue, and the table size verified
  against Eq. P1 over five magnitude limits (worst 1.1e−02). The real cost is reported:
  10.96× pairs per magnitude, 17.9 MB at magnitude 6.0 and 196.6 MB at 7.0.
- **Pinhole star camera and a scene simulator.** Ideal gnomonic projection (Eq. K1–K2) with
  a measured 1.3 % centre-to-corner departure of the nominal plate scale (Eq. K3) from the
  true local scale, and a frame simulator that applies a Haar-uniform attitude, field
  clipping, dropout, per-axis Gaussian centroid noise, false detections and
  brightest-N truncation **in that order** — so a bright false detection displaces a real
  star instead of being appended, which is what makes the false-star failure regime behave
  like the real one.
- **Triangle matching as a three-way vectorised join** (Eq. T1) and **Pyramid fourth-star
  confirmation** (Eq. Y1) vectorised over every (candidate triangle, fourth spot) pair at
  once, with the gap-ordered triple scan of Mortari, Samaan, Bruccoleri & Junkins
  (*Navigation* 51(3), 171–183, 2004). Angular separations use `atan2(‖a×b‖, a·b)` rather
  than `arccos(a·b)` throughout: measured, `arccos` is wrong by 1.6e−08 rad at a true angle
  of 1e−08 rad while the `atan2` form is accurate to 8.7e−17.
- **Davenport's q-method reimplemented** rather than imported from a sibling product, so the
  repository stands alone, with an explicit regression test for the Shuster-convention
  transpose that would otherwise return the inverse rotation silently (measured 2.7e−16 rad
  to `A` against 2.96 rad to `Aᵀ`).
- **Both error rates, everywhere, with intervals.** Three exhaustive, mutually exclusive
  outcomes — identified / false identification / no solution — are reported for every
  decision rule at every operating point, each with a Wilson (1927) interval, and a measured
  zero is reported as its upper bound rather than as zero. The classical Pyramid rule made
  **0 false identifications in 660 trials** over centroid noise from 1 to 60 arcsec, stated
  as "below 0.0058 at 95 % confidence".
- **The regime where the Pyramid rule's guarantee fails, quantified.** A tolerance sized for
  60 arcsec of noise applied to 5-arcsec frames, plus four false detections: **46 of 120
  frames return a confidently wrong attitude, false-identification rate 0.3833, 95 % CI
  [0.3012, 0.4727]**. The mis-sized gate alone does this 0 times in 120; false stars alone do
  it 0 times in 200 at every count tried. The two together do it.
- **The dense-false-star failure regime, measured to collapse.** The triangle rule keeps
  answering and starts being wrong (false-ID rate up to **0.4550** at 20 false stars); the
  Pyramid rule keeps being right and stops answering (identification 1.0000 → 0.0050 at
  0.0000 false). The *search ceiling* is reported alongside, so a decision failure is never
  confused with a search failure: it falls to 0.0300 at 20 false stars and 0.0100 at 30,
  below which no decision rule can recover the frame in principle.
- **Learned candidate ranker, benchmarked against the classical rule on identical candidate
  lists.** `HistGradientBoostingClassifier` over 13 hand-built features, trained in 3.5 s
  (16.3 s including dataset generation) on 21 150 candidate rows from 600 frames, split by
  frame and by seed. Identification-rate gain at threshold 0.5: **+0.0000** on a clean sky,
  **+0.0000** at 40 arcsec noise, **+0.0000** at 4 false stars, **+0.1600** at 8 false stars,
  **+0.0933** at 12, **+0.3667** against the 12×-too-wide gate. Eight of nine thresholds
  reach a higher identification rate than the Pyramid rule without exceeding the upper end of
  the Pyramid rule's own false-ID interval. **The losses are reported in the same place as
  the wins**: no gain at all in three of six regimes, a 27.93× runtime penalty on a clean
  frame (21.3 ms against 0.76 ms), a 0.0133 false-identification rate at 12 false stars where
  the classical rule measures 0.0000, and a wrong acceptance observed at confidence 0.97908.
  **This model is not certified for operational flight use.**
- **Confidence output, measured rather than asserted.** `predict_proba` with a reliability
  table, Brier score (4.6670e−04 against 2.3027e−01 for a base-rate predictor) and expected
  calibration error (5.9771e−04). The card says what those numbers do *not* establish: 10 587
  of 10 600 held-out rows fall in the two extreme bins and 13 rows cover everything between
  0.05 and 0.95, so no calibration is claimed in the middle of the range.
- **A mitigation that beats the learned ranker in its own regime, reported anyway.** Raising
  `max_stars` from 10 to 20 at 10 false stars takes the search ceiling from 0.4200 to 0.9867
  and the *classical* Pyramid identification rate from 0.2267 to 0.7267, for about 1 ms per
  frame. The package default stays at 10, and `README.md` Limitation 3 says to change it
  before reaching for the model.
- **Level-2 validation with saved raw output** (`validation/`): **49 checks, 49 passed, 0
  failed**, 138 s, across five rerunnable scripts. **Three expectations held before running
  were wrong and are recorded as findings rather than removed:** a one-sigma tolerance gate
  does not lose true matches (ceiling 1.000 at k = 1, because 25 triples are scanned; it
  takes k = 0.25 to move the ceiling, to 0.275); using more observed spots does rescue the
  dense false-star regime; and the 11× OpenMP small-batch penalty asserted in
  `src/skymatch/ranker.py` **does not reproduce** on the validation machine — measured 1.29×
  (1.657 ms against 1.287 ms). The docstring was left unchanged and the discrepancy
  documented.
- **167 pytest tests**, ruff-clean at line length 100 with `E,F,I,UP,B,SIM`. **Two gaps are
  documented rather than hidden:** `ranker.py`, `dataset.py`, `benchmark.py` and `cli.py` have
  no test file and are exercised only by the validation scripts, and `hypothesis` is declared
  as a test dependency but imported by no test.
- **CLI**: `python -m skymatch catalogue|identify|sweep|conventions`.
- **Three runnable examples** writing PNGs to `screenshots/` with the Agg backend (6.5 s,
  32 s, 34 s).
