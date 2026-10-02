# Model card — skymatch learned candidate ranker v0.1.0

**This model is not certified for operational flight use.**

**Every star, every frame, every centroid, every false detection and every label used to
train or evaluate this model is SIMULATED — see `DATASET_CARD.md`. No real star catalogue
is bundled or downloaded, no star-tracker image, no flight telemetry and no on-orbit
identification record is used anywhere in this package. The results characterise the
algorithms; they do not predict on-sky performance.**

Every number on this page was produced by running `validation/validate_ml_vs_classical.py`
in this session; its raw output is `validation/ml_vs_classical_output.txt` and the summary
is `validation/VALIDATION.md` section 5.

The fifteen items the build guide requires are numbered below.

---

## 1. Problem

In the **lost-in-space** problem a star tracker has a list of detected spots and no attitude
prior at all, and must decide which catalogue star each spot is. The geometric search
(classical, shared by every rule here) proposes catalogue triangles that match three
observed inter-star angles inside a tolerance, and tests each against a fourth spot. That
search typically produces 1 to 130 candidate triangles per frame — 35.3 on average over the
training grid — of which at most one is correct and frequently none is.

The learned component is the **decision**: given the candidate list for one frame, pick one
candidate or refuse, and attach a confidence. It replaces the decision rule only; it does
not replace the search, and both classical and learned rules are scored on the identical
candidate list on every frame, so any difference between them is a difference of decision.

Three outcomes are scored, exhaustive and mutually exclusive:

* **identified correctly** — all three core correspondences are the truth;
* **false identification** — a candidate was accepted and at least one correspondence is
  wrong;
* **no solution** — nothing was accepted.

These are never summed into an "accuracy". A star tracker that reports nothing is a
known-unknown the spacecraft can wait out; one that reports a confidently wrong attitude
will be believed.

## 2. Classical baseline, implemented and validated first

Two baselines, both implemented, tested and validated (`validation/VALIDATION.md` sections
3 and 4) before any learning:

* **`skymatch.identify.triangle_decision`** — accept the first observed triple whose
  catalogue triangle is unique. No fourth star. The weak baseline, present so that the
  fourth star's contribution can be measured rather than asserted. Its worst measured
  false-identification rate is **0.4550** (91 of 200 frames at 20 false stars).
* **`skymatch.identify.pyramid_decision`** — the classical **Pyramid** rule (Mortari,
  Samaan, Bruccoleri & Junkins, *Navigation* 51(3), 171–183, 2004): accept the first
  observed triple with exactly one candidate that a fourth spot confirms uniquely. **This
  is the baseline the learned ranker is measured against.** Over 660 trials spanning
  centroid noise from 1 to 60 arcsec it made **0 false identifications**, 95 % Wilson
  interval [0.00000, 0.00579], and identified 1.0000 of frames at every noise level.

The Pyramid rule is a **hard rule with no operating point**: it accepts on a uniqueness test
and otherwise returns nothing, with no graded output and no threshold to move. That is the
structural difference the learned ranker addresses, and the comparison is therefore a curve
against a point, not one number against another.

## 3. Architecture

`sklearn.ensemble.HistGradientBoostingClassifier` wrapped by `skymatch.ranker.LearnedRanker`:

```
max_iter=200, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=40,
l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
n_iter_no_change=20, random_state=0
```

No PyTorch and no GPU: PyTorch is not available in the target environment and there are two
CPU cores, so a deep model was never an option. A gradient-boosted tree ensemble is also the
right shape for 13 heterogeneous hand-built features and tens of thousands of rows. **This
is a constraint on the result, not evidence about what other architectures would do.**

Small-batch inference runs under `threadpoolctl.threadpool_limits(limits=1)` for batches
below 1000 rows, because thread-pool entry and exit cost more than the trees do at the ~25
rows a frame produces. Measured here the saving is **1.29× median** (1.657 ms → 1.287 ms per
25-row call), not the 11× the source docstring claims; see `validation/VALIDATION.md`
section 7.3, which records the discrepancy rather than hiding it.

**Inputs: the 13 features of `skymatch.identify.FEATURE_NAMES`**, all computable onboard from
the frame and the catalogue with no truth and no attitude prior:

| Feature | What it is |
|---|---|
| `resid_max_norm` | largest of the three triangle edge residuals, in units of the tolerance |
| `resid_rms_norm` | RMS of the three edge residuals, in units of the tolerance |
| `n_confirm` | number of fourth spots that confirmed this candidate (Eq. Y1) |
| `confirm_fraction` | `n_confirm` divided by the number of fourth spots tried |
| `confirm_resid_norm` | mean confirmation residual in units of the tolerance; 2.0 if nothing confirmed |
| `log1p_rivals` | `log1p` of the number of *other* catalogue triangles matching the same observed triple |
| `min_edge_rad`, `mean_edge_rad` | the observed triangle's smallest and mean edge angle [rad] |
| `magnitude_resid_rms` | RMS magnitude residual after removing the unknown zero point [mag] |
| `magnitude_rank_agreement` | fraction of the three pairwise brightness comparisons that agree |
| `log10_tolerance_arcsec` | the operating point's tolerance [log10 arcsec] |
| `expected_stars_in_field` | catalogue density × field solid angle |
| `n_spots` | spots handed to the matcher |

Five of these (`n_confirm`, `confirm_fraction`, `confirm_resid_norm` and the two residual
norms) encode the **same fourth-star evidence the classical Pyramid rule uses**. The model is
therefore being handed the classical rule's own discriminant and asked to weight it softly
instead of thresholding it. That is the whole mechanism, and it is why the row-level metrics
in item 7 are near-perfect and the frame-level gains are confined to two regimes.

**Output.** `score()` returns `predict_proba(...)[:, 1]`, a probability per candidate.
`decide(candidates, threshold)` returns the highest-scoring candidate if it clears the
threshold, else `(None, best_score)` so a caller can see how close the frame came.

## 4. Dataset source

`skymatch.dataset.generate_candidate_dataset`, committed and deterministic in an integer
seed. Nothing is committed as data; the generator *is* the dataset. A **row is one
candidate**, not one frame; the label is 1 if all three of its correspondences are the truth,
else 0; `groups` records which frame it came from.

Frames are drawn from a 12-cell grid of operating points (`DEFAULT_GRID`: magnitude limit
5.5–6.5, centroid noise 1–40 arcsec, 0–16 false stars) which **deliberately includes cells
nothing can solve**, so the ranker sees what no evidence looks like. Catalogues are synthetic
(`skymatch.catalogue.generate_catalogue`, Eq. C1/C2, isotropic positions).

| | train | test |
|---|---|---|
| Frames | 600 | 300 |
| Candidate rows | 21 150 | 10 600 |
| Rows per frame | 35.25 | 35.33 |
| Positive fraction | 0.3664 | 0.3595 |
| Solvable frames (the ceiling) | 0.8233 | 0.8333 |
| Frame seed | 1234 | 5678 |
| Catalogue seed | 20260902 | 20260902 |

Class balance follows from the search rather than from a choice, and is **not rebalanced**:
rebalancing would destroy the base rate the probability output is meant to express.

## 5. Dataset limitations

Full detail in `DATASET_CARD.md`. The ones that bear on this model:

* **The catalogue is synthetic and was never fitted to a real one.** Eq. C1's power law has
  no galactic-latitude dependence; real counts vary by roughly an order of magnitude between
  galactic pole and galactic plane, and a real star tracker's hardest fields are the dense
  ones this model has never seen.
* **Not modelled at all:** proper motion, binary and multiple stars, instrument spectral
  response, and optical distortion. The last is the most dangerous for this model
  specifically, because radial distortion of a few pixels at the field edge shifts the
  inter-star angles that *every one of the 13 features* is built from.
* **Photometry is flattered.** The simulator hands the instrument the catalogue's own
  magnitude scale plus Gaussian noise — no colour term, no zero-point drift, no saturation.
  `magnitude_resid_rms` is the single most important feature by permutation importance
  (0.00202), so item 7's ablation is the honest lower bound for this feature set.
* **Centroid noise is independent zero-mean Gaussian** on the two focal-plane axes. Real
  centroid error correlates with sub-pixel position, brightness and PSF shape.
* **False detections are uniform on the detector with uniform magnitudes.** Real false spots
  cluster: hot-pixel columns, cosmic-ray tracks, stray light gradients.

## 6. Training procedure

1. Build one catalogue and pair table per magnitude limit in the grid (0.9 s).
2. Generate 600 training frames from the grid; run the **classical** geometric search on each;
   label every candidate it proposes (11.8 s, 21 150 rows).
3. Fit the classifier on all 21 150 rows × 13 features (**3.5 s**). Internal early stopping
   holds back 15 % of rows and stops after 20 iterations without improvement.
4. Fit the 11-feature no-photometry ablation on the same rows (0.2 s).

**Total training cost 16.3 s** on two cores, against a documented budget of 180 s. No
hyperparameter search was performed: the values in item 3 are the defaults chosen once and
left, so none of them is tuned on the test set — and equally, no claim is made that they are
optimal.

## 7. Metrics

**Row-level, 10 600 held-out candidate rows** (base rate 0.35953):

| Model | ROC AUC | Average precision |
|---|---|---|
| full, 13 features | 1.00000 | 0.99999 |
| no photometry, 11 features | 0.99999 | 0.99999 |

**Read these as a warning, not a result.** An AUC of 1.00000 means the labelling problem is
nearly separable given features that already contain the classical rule's discriminant, on
synthetic data. The ablation losing essentially nothing (0.99999 → 0.99999 AP) says the same
thing from the other side. The metric that matters is the frame-level one.

**Frame-level, 150 trials per case, identical candidate lists** (ident / false-ID):

| Case | Ceiling | pyramid | ranker@0.5 | ranker@0.9 | ident gain @0.5 |
|---|---|---|---|---|---|
| clean sky, σ 5 arcsec | 1.0000 | 1.0000 / 0.0000 | 1.0000 / 0.0000 | 1.0000 / 0.0000 | **+0.0000** |
| σ 40 arcsec, no false stars | 1.0000 | 1.0000 / 0.0000 | 1.0000 / 0.0000 | 1.0000 / 0.0000 | **+0.0000** |
| 4 false stars | 1.0000 | 1.0000 / 0.0000 | 1.0000 / 0.0000 | 1.0000 / 0.0000 | **+0.0000** |
| 8 false stars | 0.6133 | 0.4333 / 0.0000 | 0.5933 / 0.0000 | 0.5733 / 0.0000 | **+0.1600** |
| 12 false stars | 0.2200 | 0.1067 / 0.0000 | 0.2000 / **0.0133** | 0.1667 / 0.0000 | **+0.0933** |
| gate 12× too wide + 4 false | 0.9067 | 0.5400 / **0.4133** | 0.9067 / 0.0067 | 0.9067 / 0.0000 | **+0.3667** |

**Threshold curve**, the operating point the classical rule does not have (8 false stars,
σ 5 arcsec, 300 trials, ceiling 0.6767): the Pyramid rule sits at 0.4733 / 0.0000, and the
ranker runs from 0.6667 / 0.0033 at threshold 0.02 to 0.4733 / 0.0000 at threshold 0.99 —
landing exactly on the classical point at the top end, which is the sanity check that the
curve passes through it. **8 of 9 thresholds** reach a higher identification rate without
exceeding the upper end of the Pyramid rule's own false-ID interval; at threshold 0.3 and
above the learned false count is also 0 / 300.

### The honest summary

The learned ranker **does** beat the classical Pyramid rule, and the win is narrower than
the headline number looks:

* **In three of six regimes the gain is exactly zero**, because the Pyramid rule is already
  at 1.000. On a clean sky, and at centroid noise up to 40 arcsec, this model has nothing to
  contribute and costs **28× the runtime** to contribute it (item 13).
* **The largest gain, +0.3667, is against a misconfigured classical rule**, handed a 12×
  too wide tolerance. It measures robustness to misconfiguration, not skill on a correctly
  configured problem. Size the gate correctly and that case disappears.
* **The genuine win is one band: 6–12 false stars**, where it recovers +0.1600 and +0.0933
  of a shortfall that section 4c of `validation/VALIDATION.md` measures at 0.1850 and 0.0800
  — most of the available headroom, and nothing beyond it.
* **It introduces false identifications the Pyramid rule does not make**: 0.0133 at 12 false
  stars against the Pyramid rule's 0.0000, and 0.0033 at thresholds below 0.3. Raising the
  threshold to 0.9 removes them and cuts the gains to +0.1400 and +0.0600.
* **Raising `max_stars` from 10 to 20 beats this model outright** in the regime it exists to
  address. At 10 false stars that single configuration change takes the ceiling from 0.42 to
  0.9867 and the *classical* Pyramid rule from 0.2267 to 0.7267, for about 1 ms per frame
  (`validation/VALIDATION.md` section 4d). The learned ranker's +0.16 at 8 false stars is
  small next to that. **If you only read one line of this card: try more spots before you try
  this model.**

## 8. Test-split strategy

Splitting is **by frame and by seed, never by row**. Candidates from one frame share observed
spots and are correlated through them, so a random row split would leak the answer across
the split. Train frames come from seed 1234 and test frames from seed 5678, both disjoint by
construction.

Train and test **share the catalogues** (catalogue seed 20260902 for both). That is
deliberate and is the split a star tracker actually faces: the catalogue is fixed and loaded
once, and the sky — the attitude, the noise, the false detections — is what varies. It also
means **nothing here measures generalisation to a different catalogue**, and a user who wants
that must regenerate with a different `catalogue_seed` and rerun.

All 13 feature columns are computed inside the classical search from the frame and the
catalogue alone. No feature reads `Scene.truth_index` or `Scene.attitude`; those exist only
for scoring.

## 9. Uncertainty / confidence output

`LearnedRanker.score` returns a probability per candidate, and `decide` returns it as the
confidence of the accepted candidate. Measured on 10 600 held-out rows:

* Brier score **4.6670e−04**, against **2.3027e−01** for predicting the base rate.
* Expected calibration error **5.9771e−04** over 10 equal-width bins.
* Reliability: the bin at mean prediction 0.0001 observes 0.0000 over 6779 rows (gap
  +0.0001); the bin at 0.9998 observes 0.9989 over 3808 rows (gap +0.0008).

**Those aggregate numbers are good for an uninteresting reason.** 10 587 of the 10 600 rows
fall in the two extreme bins. The middle of the range holds **13 rows in total**, across
which the measured gaps are +0.1326, −0.4396, +0.1583, −0.2693 and −0.1506 on 1 to 5
observations each. **No calibration is claimed between 0.05 and 0.95**, because nothing here
measures it there. The confidence is usable as an operating-point knob (item 7's threshold
curve) and as a flag for "this candidate is not like the others"; it is not a calibrated
posterior in the middle of its range, and `confidence = 0.6` must not be read as "60 %
chance this is right".

## 10. Failure cases

1. **High-confidence wrong answers.** At threshold 0.05 over 300 frames there were **8 wrong
   acceptances**, with median confidence when wrong 0.79070 and **maximum confidence when
   wrong 0.97908**. A wrong identification carrying 0.979 confidence is indistinguishable
   from a right one by its confidence alone, and item 9 gives no basis for discriminating in
   that range. This is the failure that matters for a star tracker.
2. **It introduces false identifications where the classical rule has none** — 0.0133 at 12
   false stars (item 7). Buying identification rate with false-identification rate is exactly
   the trade the Pyramid rule refuses to make, and this model makes it unless the threshold
   is raised to 0.9.
3. **It cannot identify what the search never proposed.** On the 8-false-star case the
   ceiling is 0.3867 and the learned rate 0.3767: **only 0.0100 of the shortfall is a
   decision problem** and the rest is the search. Below a ceiling of about 0.1 (12+ false
   stars at `max_stars = 10`) the frame is unrecoverable in principle and no decision rule
   helps.
4. **No gain on a clean sky, at 28× the cost.** Item 13.
5. **Nothing outside `DEFAULT_GRID` is validated.** Magnitude limits outside 5.5–6.5, centroid
   noise outside 1–40 arcsec, false-star counts above 16, cameras other than 12 deg / 1024 px,
   and `max_stars` other than 10 are extrapolation. In particular the `max_stars = 20`
   configuration that works best (item 7) is *not* in the training grid, so the model is
   untested there.
6. **The features it leans on hardest are the ones the simulator flatters.**
   `magnitude_resid_rms` has the largest permutation importance (0.00202). On a real
   instrument with a colour term, zero-point drift and saturation that feature degrades, and
   nothing here says by how much. The no-photometry ablation exists for this reason and loses
   nothing on synthetic data — which is evidence about the simulator, not about the sky.
7. **Optical distortion is absent from the training distribution** and would shift every
   angle-derived feature at once. A model trained on undistorted geometry has no
   representation of it.
8. **It is unit-tested nowhere.** `src/skymatch/ranker.py` and `src/skymatch/dataset.py` have
   **no test file** in `tests/` (`validation/VALIDATION.md` section 6). They are exercised
   only by the validation script, which is not part of CI.

## 11. Reproducibility — exact commands

From `products/P028/`:

```bash
# the full benchmark: dataset, training, calibration, frame rates, threshold
# curve, runtime and failure analysis. 8 checks, ~60 s on two cores.
cd validation && python validate_ml_vs_classical.py

# the classical baselines it is measured against, first
python validate_identification.py     # 10 checks, ~41 s
python validate_failure_regime.py     #  5 checks, ~28 s

# a reduced-size rerun that also draws the figure, ~34 s
cd .. && python examples/learned_vs_pyramid.py
```

```python
# regenerate the exact training and test matrices
from skymatch import CameraModel, DEFAULT_GRID, build_catalogue_tables, generate_candidate_dataset
cam = CameraModel(fov_deg=12.0, pixels=1024)
tables = build_catalogue_tables(tuple(p.magnitude_limit for p in DEFAULT_GRID), cam, 20260902)
train = generate_candidate_dataset(600, 1234, camera=cam, tables=tables)   # 21150 rows
test  = generate_candidate_dataset(300, 5678, camera=cam, tables=tables)   # 10600 rows
```

## 12. Seeds

| Thing | Seed |
|---|---|
| Catalogues (all splits, all magnitude limits) | 20260902 |
| Training frames | 1234 |
| Test frames | 5678 |
| Classifier `random_state` | 0 |
| Frame-rate benchmark (item 7, 5e) | 20260922 (`SEED + 20`) |
| Threshold curve (5f) | 20260923 (`SEED + 21`) |
| Permutation importance RNG | 20260924 (`SEED + 22`) |
| Failure analysis (5h) | 20260925 (`SEED + 23`) |

All frame generation goes through `numpy.random.default_rng(seed)` and is deterministic.
Two caveats, recorded in `validation/VALIDATION.md` section 7: every timing column varies
run to run, and a few BLAS-threaded quantities move in their last digits (the ablated
model's average precision moved 0.99999 ↔ 0.99998 between two runs on the same seed, and the
ordering of the near-zero permutation-importance rows is not stable).

## 13. Compute used

Two CPU cores, no GPU, a few hundred MB of RAM.

| Stage | Time |
|---|---|
| Catalogues and pair tables (3 magnitude limits) | 0.9 s |
| Training dataset, 600 frames | 11.8 s |
| Test dataset, 300 frames | 5.7 s |
| Fit, full model | 3.5 s |
| Fit, no-photometry ablation | 0.2 s |
| **Total training cost** | **16.3 s** (budget 180 s) |
| Whole benchmark script | 60.3 s |

**Inference, measured over 120 frames** — the learned path is slower, not faster:

| n false stars | classical early exit | full scan (needed by the ranker) | ranker scoring | ratio |
|---|---|---|---|---|
| 0 | 0.76 ms | 16.18 ms | 5.12 ms | **27.93 ×** |
| 4 | 2.08 ms | 9.82 ms | 5.04 ms | 7.14 × |
| 8 | 5.18 ms | 8.04 ms | 4.24 ms | 2.37 × |

The classical rule stops at the first confirmed triple; the ranker needs the whole candidate
list and so pays for the full scan plus inference. On a clean frame that is 21.3 ms against
0.76 ms, for zero gain. The penalty shrinks only because the early exit stops firing as the
frame gets hard.

## 14. Ethical and safety limits

* This model decides what a spacecraft believes its attitude to be. It is **not certified for
  operational flight use**, has never been run against a real star field or real telemetry,
  and must not be placed in a loop that matters.
* Its advantage over the classical baseline is confined to one regime, is partly a measure of
  robustness to a misconfigured baseline, and is bought with a measured non-zero
  false-identification rate. It is **not a general improvement and must not be presented as
  one**. A simple configuration change to the classical pipeline (`max_stars` 10 → 20) beats
  it in that same regime.
* The confidence output is a decision score with no measured calibration in the middle of its
  range. Presenting it to an operator as a probability would overstate what is known.
* The safe failure mode is available and simple: remove the model and the system degrades to
  `pyramid_decision`, which over 660 trials of validated operating points made zero false
  identifications. Nothing in this package depends on the model being present.
* The results are from a synthetic sky. Publishing an identification rate from this package
  as if it characterised a real star tracker would be misleading, and `DATASET_CARD.md` says
  so in its own words.

## 15. Required statement

**This model is not certified for operational flight use.**
