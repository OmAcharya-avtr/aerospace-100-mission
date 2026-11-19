# Model card — passyield per-pass yield predictor

Validation level: **Level 2**. Research-grade software.

**This model is not certified for operational flight use.**

Every metric in this card was produced by `validation/validate_learned.py` in
this session, against the committed synthetic campaign in `data/passes.csv`,
and the output file `validation/validate_learned_output.txt` is committed
beside it. The architecture, the feature set, the two splits, the uncertainty
definitions and the held-out stations were all fixed in
`src/passyield/dataset.py` and `src/passyield/predictor.py` **before** any
score was computed, and none of them was changed afterwards.

## Problem

Predict the **realised fraction** `Y_real / Y_cap` of a scheduled
ground-station contact — the volume the record accounts for over the volume
the window could have carried — from quantities a scheduler knows **before the
contact starts**. Multiplied by the known capacity and subtracted from the
known booked volume, that is a predictive distribution of the
nominal-versus-realised shortfall in Mbit, which is the product's headline
quantity and the axis everything here is scored on.

Normalising by capacity rather than by the booked volume divides the geometry
out of the target. That is what makes the target transferable to a station the
model has never seen: the geometry and the link budget are computable in
advance, so a model that predicts the fraction and is handed the capacity
predicts the yield.

## Baselines, implemented and validated first

All three baselines were implemented and validated before the learned
component was benchmarked, in `validation/validate_decomposition.py`,
`validation/validate_convolution.py` and `validation/validate_bounds.py`,
which run before `validate_learned.py` in the release gate's own alphabetical
order. The build guide requires that order because an under-strength baseline
inflates an apparent learned win.

1. **Baseline 1, the booked figure.** `T * R_nom`, predicting a shortfall of
   zero. Built at full strength: the duration is the true geometric
   mask-to-mask window from the committed SGP4 element set and the rate is the
   link budget's own rate at a declared 30 degree reference elevation with the
   station's own figure of merit. This is the thing being refuted.
2. **Baseline 2, the analytic law**, by numerical convolution: a
   mass-preserving pushforward of a fitted Gamma acquisition law through the
   cumulative volume, a compound-Poisson convolution of the per-outage loss,
   and a pushforward of a fitted Gamma idle law, all convolved on a uniform
   grid of data volume. Fitted on the **training side only** of each split.
3. **Baseline 2b**, the same convolution with the backlog-limited dry time in
   place of the fitted idle law. The dry time is knowable before the pass, so
   an analytic model is entitled to use it.
4. **Baseline 3, accounted ground truth**, with a percentile bootstrap whose
   resampling unit is the **pass**. This is what everything is scored against,
   so it is not a row in the comparison tables.
5. **The constant-fraction null.** Every held-out pass receives the training
   set's empirical law of the realised fraction. A learned model that cannot
   beat this has learned nothing from the features; an analytic baseline that
   cannot beat it is weaker than knowing nothing but the mean.

## Architecture

`sklearn.ensemble.RandomForestRegressor`, `n_estimators=300`, `max_depth=10`,
`min_samples_leaf=4`, `random_state=99099`, `oob_score=True`, `n_jobs=1`.

A forest and nothing larger, for three reasons that are hard constraints:

- the dataset is 1990 passes from one campaign, so a model with many
  parameters would memorise passes;
- the build environment has two cores shared with four other concurrent
  builds and a 60 s budget for any single fit;
- a forest supplies two different uncertainty outputs for free, and the
  difference between them turns out to be the most useful thing the model
  says.

`max_depth=10` with `min_samples_leaf=4` caps the effective leaf count well
below the training size on both splits.

## Features: pre-pass only, and no station identity

Seventeen columns, in `passyield.predictor.FEATURE_NAMES`:

`window_duration_s`, `max_elevation_deg`, `min_slant_range_km`,
`capacity_mbit`, `mean_rate_mbit_s`, `station_gain_over_temperature_db`,
`station_zenith_attenuation_db`, `station_mask_deg`,
`station_abs_latitude_deg`, `slew_angle_deg`, `backlog_mbit`,
`hours_since_last_contact`, `prior_pass_outage_count`, `local_hour_sin`,
`local_hour_cos`, `rise_azimuth_sin`, `rise_azimuth_cos`.

Two properties are enforced by tests, not by intention:

- **Nothing the pass did is a feature.** The acquisition time, this pass's
  outage count, the outage durations and the idle time are the target's
  causes, not its predictors; a model given any of them is a measurement.
  `tests/test_predictor.py::test_feature_names_are_all_pre_pass` scans the
  names. The one outage-flavoured feature is `prior_pass_outage_count`, which
  refers to the **previous** contact at the same station and whose name says
  so.
- **The station enters only through its own physical figures** — figure of
  merit, zenith attenuation, mask, latitude — and never through its identity.
  A model with a station-identity feature cannot be scored on a held-out
  station at all.

## Uncertainty output

Two modes, both implemented, both scored, both published.

- `"tree-spread"` — the 300 per-tree predictions. This is the output people
  reach for first and it is **structurally over-confident**: the spread across
  trees measures disagreement about the conditional mean and omits the
  irreducible noise entirely.
- `"residual"` — the point prediction plus the empirical law of the
  **out-of-bag** training residuals, as a deterministic 256-member quantile
  ensemble, clipped into `[0, 1]` because a realised fraction outside that is
  not a possible outcome. Out-of-bag rather than in-sample, because in-sample
  forest residuals are shrunk by the fit and would reproduce the
  over-confidence in a subtler form.

**`"residual"` is the product's declared uncertainty output**
(`DECLARED_UNCERTAINTY_MODE`). It is measurably wider than the tree spread —
`tests/test_predictor.py` asserts that on the median held-out interval width —
and measurably better calibrated on both splits.

## Test-split strategy

Both splits are declared in `passyield.dataset` and neither was changed after
a score was seen.

- **Held-out stations.** `GS-03` and `GS-08`, chosen before any measurement as
  the worst and the best figure of merit in the catalogue, so the split tests
  extrapolation in the station's own physics rather than interpolation between
  two similar sites. 1443 training passes, 547 held out. No station appears on
  both sides.
- **Held-out time.** The last 25 per cent of the campaign in chronological
  order, never a random subset, because a random subset lets the model
  interpolate between passes that straddle the one it is forecasting and a
  schedule cannot do that. 1492 training passes, 498 held out; every held-out
  pass is later than every training pass, asserted by
  `tests/test_dataset.py`.

The event laws for baselines 2 and 2b are refitted on each split's training
side, so no held-out pass contributes to the law it is scored under.

## Metrics

Scored on the **shortfall of a pass in Mbit**. Primary score the **CRPS**
(Gneiting & Raftery 2007, JASA 102(477), section 4.2 and equation 21), which
is strictly proper over a whole predictive distribution; a point forecast is
the one-member ensemble, for which the CRPS is the absolute error exactly, so
baseline 1 sits on the same axis without special pleading. Calibration is
reported beside it and is the figure that decides usability: the **PIT**
Kolmogorov distance from uniform (Dawid 1984; Gneiting, Balabdaoui & Raftery
2007) and the coverage of the nominal **80 per cent** interval, for which the
target is 0.800 and either direction away from it is a defect.

### Held-out stations (the harder split)

| predictor | CRPS, Mbit | MAE, Mbit | PIT cal err | cov80 |
|---|---|---|---|---|
| baseline 1 booked | 17732.36 | 17732.36 | 1.0000 | 0.0000 |
| null constant fraction | 4891.64 | 6859.17 | **0.1961** | 0.7185 |
| baseline 2 convolution | 7014.63 | 7831.87 | 0.6709 | 0.2194 |
| baseline 2b backlog-aware | 4184.76 | 5271.86 | 0.3601 | 0.3455 |
| learned tree-spread | 3203.30 | 4110.59 | 0.2455 | 0.5558 |
| **learned residual (declared)** | **3086.86** | 4140.69 | 0.2410 | 0.6344 |

### Held-out time

| predictor | CRPS, Mbit | MAE, Mbit | PIT cal err | cov80 |
|---|---|---|---|---|
| baseline 1 booked | 14380.29 | 14380.29 | 1.0000 | 0.0000 |
| null constant fraction | 3685.53 | 5839.60 | 0.0483 | 0.7932 |
| baseline 2 convolution | 4102.21 | 4808.50 | 0.5341 | 0.3534 |
| baseline 2b backlog-aware | 3892.15 | 4908.58 | 0.2590 | 0.4237 |
| **learned tree-spread** | **1682.22** | 2284.37 | 0.1300 | 0.6165 |
| learned residual (declared) | 1701.40 | 2326.78 | **0.0345** | **0.8052** |

### The gap between the splits

The declared configuration costs **1.81 times** the CRPS on a station it never
trained on (3086.86 against 1701.40 Mbit) and its calibration error is **7.0
times** worse there (0.2410 against 0.0345). The gap is wider for the learned
model than for any baseline: baseline 2b's ratio is 1.08 and the null's is
1.33.

## Findings, with nothing retuned away

1. **Baseline 1 is refuted on both splits.** Its CRPS is 5.7 times the learned
   model's on held-out stations and 8.5 times on held-out time, and its 80 per
   cent interval covers nothing, because it is a point. The campaign books
   1.64 times the volume the record accounts for.
2. **The specified baseline 2 is beaten by a trivial null on both splits.**
   7014.63 against 4891.64 on stations and 4102.21 against 3685.53 on time.
   The cause is diagnosed, not shrugged at: baseline 2's idle law is
   unconditional while the record's idle time is driven by the onboard
   backlog, which is a pre-pass quantity. Baseline 2b uses it and is the
   better analytic form on both splits; it is reported **beside** the
   specified form and does not replace it, because the specification names the
   unconditional form and the measurement of that form is the finding.
3. **The learned forest has the lowest CRPS on both splits.**
4. **On the harder split, the trivial null is better calibrated than the
   learned model** — PIT 0.1961 against 0.2410, cov80 0.7185 against 0.6344
   for a nominal 0.80. Calibration is what a scheduler needs in order to book
   a quantile, so this is a baseline result on the dimension that matters most
   and it is published as one.
5. **The declared uncertainty output is better calibrated than the tree spread
   on both splits** — 0.2410 against 0.2455, and 0.0345 against 0.1300. On
   held-out time it reaches cov80 0.8052 against a nominal 0.80, which is the
   one place in this product where the learned uncertainty output is usable as
   it stands.

## Failure cases

- **A station the model never trained on.** 80 per cent interval coverage
  0.6344 against a nominal 0.80, and a PIT calibration error of 0.2410. The
  model is over-confident there. The null's interval measured 0.7185 and
  0.1961 on the same rows, so on a new station the null's interval is the
  better one to book against.
- **A pass whose acquisition exceeds the whole window.** 44 of the 1990
  committed passes are of this kind, and the target is exactly 0.0 there. The
  forest cannot predict an exact zero and the residual ensemble is clipped at
  zero, so these rows sit at the boundary of the predictive support.
- **Extrapolation in the station's physics.** The two held-out stations were
  chosen as the extremes of the catalogue by figure of merit, so the station
  split is extrapolation and not interpolation. A station outside the measured
  19.5 to 24.0 dB/K range, or with a mask outside 10 to 15 degrees, is outside
  everything this card reports.
- **A different campaign.** Every figure here is from one synthetic campaign
  over one element set. The model has not been tested against a real contact
  record and there is none in this repository to test it against.

## Reproducibility

```bash
python data/generate_passes.py            # the record, seed 990099
python validation/validate_learned.py     # every figure in this card, seed 99099
python examples/learned_vs_baselines.py   # the figure
```

Seeds: `99099` for the forest's `random_state`, the bootstrap and every
validation script; `990099` for the campaign generator, recorded in
`data/manifest.json`. `tests/test_determinism.py` asserts that the validation,
examples and test seeds are the same number and that the generator's seed
matches the manifest.

**No model is serialised.** The forest is refitted from the committed record
by every script that needs it, which takes about 2.5 s on one core. The
`scikit-learn>=1.9,<1.10` ceiling in `pyproject.toml` is therefore about the
**numbers** — the published CRPS, calibration and coverage are a property of
this minor version's tree building and bootstrap — and not about a pickle
format. `.gitignore` carries patterns for serialised models so that a future
change cannot add one silently and leave that ceiling reading as merely
advisory.

## Compute used

Measured on x86_64 Linux, 2 cores shared with four other concurrent builds,
Python 3.13.18:

| step | measured |
|---|---|
| 300-tree fit, one split | about 2.5 s on one core |
| `validate_learned.py` end to end, both splits, all six predictors | about 41 s wall, 20 s CPU |
| whole test suite, 284 tests | about 6 s |
| whole validation set, 11 scripts | under 2 minutes |

Well inside the 60 s per-fit budget and the 600 s per-script gate limit. No
number in this card or anywhere in this product is a comparison between two
live wall-clock measurements.

## Ethical and safety limits

- **This model is not certified for operational flight use.**
- The training data is synthetic. The eight ground stations and the entire
  event timeline are declared design values; only the orbit is real and
  published. Nothing here is a measurement of, or a claim about, any real
  ground station, network, operator or mission.
- The model is over-confident on an unseen station, which is exactly the case
  an operator would most want to use it for. Using its interval to size a
  contact schedule for a new station would under-book the shortfall. The
  limitation is quantified above and in `README.md`; it is not a caveat to be
  read past.
- No personal data of any kind is involved.
