# Model card — orbitdrift drift predictors

Four families of predictor ship in this package. Three are analytic and one is
learned, and they are documented together because the only thing that makes
either interesting is the comparison between them.

**This model is not certified for operational flight use.**

## 1. Problem

Given a two-line element set and an epoch age, predict the **model divergence**
— how far SGP4 and an independent numerical propagator have separated — with a
prediction interval whose coverage is measured rather than assumed, so that an
operator can answer "is this element set still good enough for what I am about
to do" instead of guessing from a rule of thumb.

The target is not ephemeris error against truth. There are no precision
ephemerides and no element-set archive in this environment. What is predicted is
the separation of two propagations, and `orbitdrift.divergence` derives the one
rigorous claim that supports: half the separation is a lower bound on the larger
of the two models' position errors.

**No output of any model here is a covariance, and none may be used as the input
covariance of a conjunction assessment.** P071 ConjProb is the product that
computes collision probability, and its covariance must come from orbit
determination — a filter with measurement residuals, or a published covariance
in a conjunction data message. `orbitdrift.guard` enforces this in code and
`validation/validate_guard.py` records 23 refusals firing and 15 legitimate
calls passing.

## 2. The baseline, which was built and validated first

The order is not cosmetic. The analytic forms were implemented, then validated
against the simulation in `validation/validate_growth_baseline.py`, and only
then was anything learned. The ordering inside the benchmark is enforced in
`orbitdrift.benchmark.run_benchmark` rather than left to the caller, because a
baseline weaker than it could be manufactures a learned win.

Each analytic form is what the dynamics produce under a stated dominant error
term:

| Form | Physics | Fitted parameters |
|---|---|---|
| linear in time | a constant mean-motion error: along-track error `-(3/2) n (da/a) t` | amplitude only |
| quadratic in time | a constant error in the *rate* of change of semi-major axis, which a mis-modelled drag gives: `da(t) = adot_err t`, so the along-track error integrates to `t^2` | amplitude only |
| free power law | `A t^p` with `p` free: the measurement of which of the two above is in force | amplitude and exponent |
| altitude-dependent power | `p = p0 + p1 log10(perigee altitude)`, because the dominant term changes with altitude | amplitude and two exponent coefficients |
| drag-dominated exponential | `A (exp(t/tau) - 1)`: at very low altitude the altitude loss feeds back into the density | amplitude and timescale |

Also in the benchmark, because it is the thing the product is positioned
against: a **rule of thumb** of one kilometre per day with a factor-of-three
band, fitted to nothing. Its measured coverage is 0.3017 at a claimed 0.90.

### The baseline's physics, measured

`validate_growth_baseline.py`, per-object free-exponent fits over the committed
population of 192 objects:

| perigee band | n | mean p | sd p |
|---|---|---|---|
| 300-450 km | 42 | 1.2683 | 0.3914 |
| 450-600 km | 32 | 1.0249 | 0.3590 |
| 600-800 km | 31 | 1.1197 | 0.3383 |
| 800-1100 km | 41 | 1.0175 | 0.2020 |
| 1100-1700 km | 46 | **0.9685** | **0.0628** |

Correlation of `p` with `log10(perigee altitude)`: **-0.2914**. With
`log10(C_D A/m * rho_perigee)`: **+0.5090**. Above 1100 km the growth is linear
in time to within the fitted scatter, which is what a constant mean-motion error
gives, and that is the specification's predicted outcome confirmed.

## 3. The learned model

`orbitdrift.predictors.LearnedQuantilePredictor`. Three
`sklearn.ensemble.GradientBoostingRegressor` fits with the quantile loss, at the
lower quantile, the median and the upper quantile, on the logarithm of drift.
PyTorch is **not installed** in the environment these numbers were measured in
and nothing here uses it.

| Hyperparameter | Value |
|---|---|
| loss | quantile |
| alpha | 0.05, 0.50, 0.95 |
| n_estimators | 300 |
| max_depth | 3 |
| learning_rate | 0.05 |
| min_samples_leaf | 20 |
| subsample | 0.9 |
| random_state | 20261025 |

**Nothing was tuned.** These are the values in the dataclass defaults and they
were not searched. The reason is stated rather than apologised for: the compute
budget is two shared cores, the dataset is a few thousand rows, and an unbounded
hyperparameter search against a closed-form least-squares baseline would turn
the comparison into a comparison of search budgets. A reader who wants a tuned
comparison should run one and say so.

`LearnedQuantilePredictor.predict_interval_km` **refuses** a coverage other than
the quantile pair it was fitted with, rather than rescaling. Quantile crossing —
a lower prediction above the upper one — is **reported** by
`interval_crossing_fraction` rather than sorted away, because a crossing is a
statement that the fit is not supported at that point.

## 4. Features

The ten static features plus epoch age, identical for every predictor:

| Feature | Units | Why |
|---|---|---|
| `perigee_altitude_km` | km | from `satrec.altp`, scaled by the WGS72 radius SGP4 used |
| `apogee_altitude_km` | km | from `satrec.alta` |
| `eccentricity` | — | |
| `inclination_deg` | deg | |
| `bstar` | 1/earth-radii | signed, as the element set carries it |
| `log10_abs_bstar` | — | because `B*` spans four decades |
| `bstar_sign` | — | separate, so a negative or zero `B*` has a defined logarithm |
| `period_min` | min | `2 pi / no_kozai` |
| `log10_perigee_density` | — | the physically correct variable: density varies by five orders of magnitude over the altitude range, and a model given altitude alone must learn that exponential relation from data |
| `log10_drag_parameter` | — | `log10(C_D A/m * rho_perigee)`, the combination that actually appears in the drag acceleration; `B*` alone cannot express it because the same `B*` means a very different acceleration at 400 km and 1400 km |
| `epoch_age_days` | days | the one feature that varies along a curve |

Giving the model the logarithm of the density rather than the altitude alone is
not a trick that helps the analytic form: it is the physically correct variable,
and withholding it to make the learned model look better would be the trick.
Both families see the same columns.

In the extrapolation task the learned model receives three further features —
the object's own inside-horizon fitted log amplitude, exponent and log residual
spread — so it is not handicapped by being denied what the analytic baseline
has.

### The feature set is strongly collinear, and the model is only valid on its manifold

Perigee altitude, apogee altitude, period, the log perigee density and the log
drag parameter are all functions of the same orbit and move together. A
log-linear fit on such data carries large coefficients in nearly opposite
directions along the collinear combinations: harmless **on** the manifold the
training data occupy, catastrophic off it.

Measured: a hand-assembled feature vector with a 1300 km perigee, an apogee 25
km above it and a period of 105.6 minutes — internally inconsistent, because a
1300 by 1325 km orbit has a period of 112 minutes — drew a predicted divergence
of **2.5e8 km** from `analytic_power_altitude`, against the 17.7 km measured for
a real object at that altitude. The model was not wrong; it was asked about a
point no orbit occupies.

`AnalyticGrowthPredictor.leverage` is the diagnostic: the Mahalanobis distance
of a query from the training features in their standardised, pseudo-inverted
covariance. Measured on the committed population: the training maximum is
**7.04**, the held-out objects reach **8.45**, and the inconsistent vector
scores **259**.

**Build the feature vector with `orbitdrift.features.features_from_satrec` from
an actual element set. Do not assemble one by hand.**

## 5. Dataset and split

See `DATASET_CARD.md`. 192 synthetic objects, three epoch groups, 5376 rows, a
seven-day span at 360 integration steps per revolution, committed as
`fixtures/divergence_dataset.npz` with the population as
`fixtures/population.tle`.

The split is **by object, never by row**: 134 training and 58 test objects,
disjoint. Splitting by row would put an object's early ages in training and its
late ages in test, where the features are identical and only the age column
differs, so a model could interpolate its own training rows.

Conformal calibration splits **by object too**, so no object appears in both the
fitting and the calibration part. `validation/validate_predictors.py` reports
the half-width under both splits, so the size of the leak a row-wise split would
cause is visible.

## 6. Uncertainty output, and what it promises

Every predictor exposes `predict_interval_km`. Three mechanisms appear:

| Mechanism | What it assumes | What it promises |
|---|---|---|
| Gaussian from the training log-residual spread | log residuals are Gaussian and the training spread carries over | nothing that is checked; this is the row that shows what assuming costs |
| quantile regression | nothing parametric | the fitted quantiles, which are not the same as coverage |
| split conformal | the rows are exchangeable | marginal coverage at or above the nominal level |

**The exchangeability assumption is violated here and that is the point rather
than a footnote.** Twenty-eight epoch ages of one object share its features and
its physics, so a calibration set drawn by object is exchangeable at the level
of objects and not at the level of rows. Marginal coverage therefore comes out
close to nominal while coverage conditional on altitude or on epoch age does
not. Measured: `analytic_power_altitude_conformal` covers 0.8768 marginally with
altitude cells spanning 0.7802 to 0.9375, a spread of 0.1573. Conformal
prediction promises marginal coverage and this package does not claim it
promises more.

## 7. Metrics, and how the winner is decided

Reported per predictor: measured coverage with a Clopper-Pearson 95 per cent
interval; object-level coverage with its own interval; median and mean interval
width in km; median `log(upper/lower)`, which is the scale-free width;
the mean Winkler interval score (Gneiting and Raftery 2007, eq. 43); the
fraction of crossed intervals; and the median absolute log point error.

Narrowing an interval always improves its score until it stops covering, so
**width is only comparable among predictors whose measured coverage is
consistent with the level they claim.** Three rankings are reported: over
everything, over the predictors whose Clopper-Pearson interval contains the
nominal level, and over the predictors that do not under-cover at all.

The third is the operational one. The two failure directions are not symmetric
for a staleness decision: an interval that covers 95 per cent when it claims 90
gives a horizon that is early, so the user refreshes sooner than strictly
necessary, which is a cost. An interval that covers 86 per cent when it claims
90 gives a horizon that is **late**, so the user acts on an element set already
outside the tolerance they declared, which is a defect.

## 8. Results

Full tables in `validation/VALIDATION.md` §10. The verdicts, as measured:

### Task 1 — cold prediction from elements and epoch age alone

1624 held-out rows over 58 objects.

- **The learned model wins the point prediction.** `learned_gbr_quantile` 0.5942 median absolute log error against the best analytic `analytic_power_conformal` 0.7410.
- Lowest interval score overall: `learned_gbr_conformal`, 270.64 km, coverage 0.8849.
- Calibrated predictors: `analytic_quadratic` (0.8898, CI [0.8735, 0.9046]) and `analytic_power_altitude` (0.8861, CI [0.8696, 0.9011]). Best of those: `analytic_power_altitude`, 477.60 km.
- **No learned predictor is calibrated on this task.**
- **Every predictor under-covers at 0.90.** The table has no operationally usable row.

### Task 2 — extrapolation beyond a two-day fitting horizon (the realistic task)

1160 held-out rows over 58 objects.

- **Best point prediction is analytic**: `own_curve_drag_exponential` 0.1299 against the best learned `learned_with_curve_features` 0.1391 — the analytic form beats the learned model by 7 per cent.
- **The only calibrated predictor is analytic**: `own_curve_linear_conformal`, coverage 0.8991 [0.8804, 0.9159], score 554.32 km.
- **The only predictor that does not under-cover is analytic**: `own_curve_power_conformal`, coverage 0.9474, score 281.53 km.
- **No learned predictor reaches its nominal coverage.** The best learned coverage is 0.8750.
- Lowest interval score overall is `learned_with_curve_features_conformal` at 164.44 km, bought with a coverage of 0.8681 against a claimed 0.90.

### What the learning is worth, separately from the curve

| | median \|log error\| | interval score km |
|---|---|---|
| learned, with the object's own curve as features | 0.1391 | 200.91 |
| learned, features only | 0.5439 | 446.79 |
| ratio | **3.9095** | 2.2238 |

The information is in the curve, not in the learning. A learned model denied the
curve is four times worse at the point prediction, and the analytic
extrapolation of that same curve beats both on the point prediction and is the
only family that reaches its nominal coverage.

### The recommendation this card makes on its own measurements

**For the extrapolation task, use `own_curve_power_conformal`.** It is analytic,
it is the only predictor measured not to under-cover, its interval score of
281.53 km is within 2 per cent of the best analytic score, and 0 of its 7
stratified cells under-cover.

**For the cold task, use `analytic_power_altitude` and treat its horizon as
optimistic.** Every predictor on that task under-covers, including the learned
one, so there is no row that can be recommended without that caveat.

**The learned model is not recommended for either task.** It wins the cold-task
point prediction and it has the lowest interval score on both, and neither of
those is the criterion a staleness decision needs. Its intervals do not cover at
the level they claim, and a horizon taken from an interval that under-covers is
late.

## 9. Failure cases

| Case | Behaviour |
|---|---|
| a hand-assembled or internally inconsistent feature vector | prediction wrong by orders of magnitude; `leverage` detects it (training maximum 7.04, inconsistent vector 259) |
| an object with a negative `B*` | passed through; the drag term becomes an along-track thrust, which is the dynamics the element set actually describes |
| an eccentric orbit | the population is capped near `e = 0.05` because a fixed-step integrator is poor at a fast perigee passage: relative position error 1.014e-06 at `e = 0.59` against 5.004e-10 at `e = 0.001` |
| an object above about 2000 km | outside the training range entirely; the atmosphere table is extrapolated and the population does not go there |
| high solar activity | the atmosphere model is static, with no solar index; published solar-cycle density variation at 400 km is about an order of magnitude and every drag-driven prediction inherits it |
| `own_curve_*` forms with fewer than three usable ages inside the horizon | refused with `InsufficientDataError` rather than fitted |
| `own_curve_drag_exponential` extrapolated far beyond its horizon | its interval score is 874131.70 km, four orders worse than the other forms: an exponential extrapolated past its fitting range occasionally explodes. The number is in the table rather than removed from it |
| a coverage other than the one a model was fitted or calibrated for | refused, not rescaled |

## 10. Reproducibility

Exact commands in `validation/VALIDATION.md` §15. Every seed is in the source:
population `20740825`, split `20740825`, learned model `20261025`, conformal
split `20740825`.

Verified in `validation/validate_reproducibility.py`: the population reproduces
byte for byte from its seed; two runs of the divergence computation agree bit
for bit; both benchmarks are bit-reproducible across runs (0 mismatched rows of
13 and of 12); and a different split seed changes 13 of 13 measured coverages,
so the split is in use.

## 11. Compute used

| Step | Wall clock (UNSTABLE: shared two-core container) |
|---|---|
| dataset regeneration, 192 objects, 7 days, 360 steps/rev | 190.9 s |
| cold-prediction benchmark, 13 predictors | 8.4 s |
| extrapolation benchmark, 12 predictors | 15.7 s |
| test suite, 273 tests | 72.9 s |
| the `pytest.approx` audit, three suite runs | 232.9 s |
| the working-directory matrix, 104 invocations | 329.5 s |

Timings appear in tables declared unstable and never in a prose conclusion.

## 12. Ethical and safety limits

This is research-grade software. It is **not flight-qualified, not certified,
and not approved for operational aerospace use.**

It measures model divergence between two propagators. It does not measure
ephemeris error against truth, and no output of it is a covariance or an input
to a conjunction assessment.

The specific misuse guarded against: a probability of collision is a ratio of
two small numbers and is dominated by the covariance it is given. Supplying a
model-divergence spread in that slot would produce a number with a plausible
magnitude and no meaning, and the failure would be silent. See
`orbitdrift.guard` and P071 ConjProb.

## Licence

Apache-2.0, copyright 2026 OPTIMA Organisation. The credits line required by
the mission appears once, in README.md under Credits.
