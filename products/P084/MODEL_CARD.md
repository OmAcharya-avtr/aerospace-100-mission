# Model card — `transfergap` learned discrepancy model

**Product:** P084 `transfergap` 0.1.0 · **Status:** TESTING · **Validation
level:** 2 · **Licence:** Apache-2.0 · © 2026 OPTIMA Organisation

**This model is not certified for operational flight use.**

Every number on this page was produced by a script in this repository, run in
the session that wrote the page, with the raw output committed beside the
script. The primary source is
`validation/validate_learned_model_output.txt`; the negative controls are in
`validation/validate_negative_controls_output.txt`.

## 1. Problem

A digital twin predicts a scalar quantity — here, wing-root bending moment in
kN.m — and a field test measures the same quantity at the same operating
condition. The **sim-to-field gap** is `d = min(|y − s|, R)` by equation (1) of
`transfergap.discrepancy`, with `R` a declared finite instrument range.

The marginal bounds in `transfergap.bounds` and `transfergap.resample` bound
the gap over the operating-condition population as a whole. The thing they
cannot do is say that the twin is fine at Mach 0.6 and poor at Mach 0.9. That
is the model's job: a map from the operating condition `x` to the conditional
discrepancy distribution.

The model is **not** the product's answer to "what is the discrepancy". The
distribution-free bound is. The model's role is conditional discrimination,
and the measured separation between the two roles is the research content of
this product.

## 2. Baseline

**The baseline was implemented and validated first**, in
`validation/validate_bounds.py`, before any learned component existed. It is
not a stub and it is not an under-strength comparator; it is validated on its
own terms against hand calculations, published tolerance-factor tables, the
exact binomial tail, the conformal coverage bracket of Lei et al. (2018)
Theorem 2.2 and the `Beta(n, 1)` order-statistic law.

| baseline | target | standing | measured coverage of the truth |
| --- | --- | --- | --- |
| Hoeffding (1963), eq. (3) | mean gap | distribution-free | **1.0000** at n = 15, 25, 40, 60 |
| Maurer–Pontil (2009), eq. (5) | mean gap | distribution-free | **1.0000** at all four |
| bootstrap percentile, eq. (14) | mean gap | asymptotic | 0.9067 ± 0.0168 at n = 40, measurably below 0.95 |
| Wilks (1941), eq. (7) | future gap, `p = 0.95` | distribution-free, training-conditional | 0.9868, tolerance rate 0.9750, n = 60 only |
| split conformal, eq. (12) | future gap | distribution-free, marginal | 0.9524 to 0.9577 at n ≥ 25 |
| normal-theory tolerance, eq. (9) | future gap | needs normality | 0.9535 at n = 60, measurably below 0.95 at the 300-campaign measurement |

The model is benchmarked against these on the **same** held-out pool, at the
**same** declared confidence, with the **same** scoring rule.

## 3. Architecture

Three independent `sklearn.ensemble.GradientBoostingRegressor` ensembles
fitted with the pinball (quantile) loss of equation (23),

```
L_q(y, f) = q (y - f)_+ + (1 - q) (f - y)_+
```

whose population minimiser is the conditional `q`-quantile. Koenker & Bassett
(1978), *Econometrica* 46(1), 33–50, equation (1.2); the boosting algorithm is
Friedman (2001), *Annals of Statistics* 29(5), 1189–1232.

| ensemble | quantile | used for |
| --- | --- | --- |
| lower | `1 − p`, default 0.05 | the lower edge of the uncertainty interval |
| median | 0.5 | the point estimate and the discrimination measurement |
| upper | `p`, default 0.95 | the conditional upper band that is scored on coverage |

Inputs: a `(n, 4)` array of operating conditions — Mach, altitude in km, angle
of attack in degrees, load factor. Output: a conditional quantile in kN.m, one
per row. No feature engineering, no standardisation (trees are invariant to
monotone rescaling of a feature), no feature selection.

**Declared hyperparameters. There is no tuning loop in this repository.**

```
n_estimators     = 80
learning_rate    = 0.12
max_depth        = 2
min_samples_leaf = 3
subsample        = 1.0
```

Small trees and a slow learning rate because the training sets this product is
about have **tens** of rows. The values are declared rather than tuned, which
means they are certainly not optimal; tuning them on the same synthetic
population the model is scored against would make the coverage measurement
meaningless.

`fit_full=False` fits only the upper ensemble. It is three times cheaper and
is what the repeated-draw coverage study uses, since only the upper band is
scored; with it, the median and lower predictors **raise** rather than
returning the upper band under another name.

### The conformal repair

`ConformalisedDiscrepancyModel` splits the field campaign half and half, fits
the model on one half and calibrates an additive offset on the other:
`s_i = y_i − upper(x_i)`, `qhat` the conformal quantile of equation (12), band
`upper(x) + qhat`. Romano, Patterson & Candès (2019), "Conformalized quantile
regression", *NeurIPS* 32. Under exchangeability the band has the
finite-sample marginal guarantee of equation (13) whatever the underlying
model does, including when the underlying model is useless.

The cost is explicit: the repair adds **one constant** to every condition, so
it fixes the average coverage and not the conditional coverage; and it needs
at least 19 calibration scores for equation (12), so at a half-and-half split
it does not exist below n = 40 while the raw model happily returns a number
from 15 points.

## 4. Dataset source and limitations

Full card: [`DATASET_CARD.md`](DATASET_CARD.md).

**Synthetic, generated by committed code from declared seeds.** No measurement
of any real aircraft, engine or test article appears anywhere in this
repository. `transfergap.synthetic` implements equations (19)–(21):

```
e(x) = b(x) + s(x) Z,   Z ~ Normal(0, 1)
b(x) = 2.0 + 6.0 sigmoid((M - 0.82)/0.05) + 3.0 (alpha/10)^2      [kN.m]
s(x) = 1.5 + 4.0 sigmoid((M - 0.85)/0.04) + 1.0 (nz - 1)          [kN.m]
d    = min(|e(x)|, 40.0)
```

Operating envelope, uniform and independent in the `random` campaign: Mach
0.55–0.95, altitude 6–12 km, angle of attack 0–9 degrees, load factor 0.8–3.0.

**This is a declared fiction with an engineering shape, not a physical
derivation**, and it is labelled as such everywhere it is used. What it encodes
is the one qualitative fact that makes the problem interesting: a twin's error
is not uniform over the envelope, and the growth is put in the transonic
corner because a steady linear-aerodynamics twin loses fidelity as shock
formation begins. The specific constants were chosen so the mean discrepancy
is a few per cent of `R` and the upper tail reaches a substantial fraction of
it, which is the regime where the choice of bound matters.

Limitations of the dataset, each of which bounds what the measurements on this
page can mean:

- The error law is a **folded normal** at every condition. A real twin error is
  not, and a heavier tail would make every learned quantile worse and every
  distribution-free bound unchanged.
- The conditional mean and scale are **smooth** in Mach and angle of attack. A
  real twin can have a discontinuity at a configuration change, which trees
  would fit better and a smooth model worse.
- The operating conditions are **independent** across features in the `random`
  campaign. Real flight-test conditions are strongly correlated — Mach,
  altitude and load factor are not independently selectable.
- Nothing is **censored**: the measured clip fraction at `R = 40 kN.m` is
  0.000000 over 400 000 draws, so the censoring path of equation (1) is
  exercised only by the input-validation tests and never by the study.
- The `swept` campaign is the only non-exchangeable alternative, and it is a
  monotone Mach sweep. A real campaign can depart from exchangeability in ways
  neither test statistic has power against.

## 5. Training procedure

```
for each independent field campaign:
    draw n paired measurements from the `random` campaign at seed SEED_FIELD + k*n
    fit the upper-quantile ensemble on (features, gaps)        [raw model]
    or: permute, split half/half, fit on one half,
        calibrate the offset on the other                      [conformal model]
    predict on a disjoint 2500-point pool
    score the achieved coverage against the pool's realised gaps
```

No early stopping, no validation split inside the fit, no regularisation path,
no class weights. The fit is deterministic given the seed: `random_state` is
set on every ensemble and the conformal split is drawn from
`numpy.random.default_rng(seed)`.

Repetitions: **80 independent campaigns** per sample size for the coverage
study, **12** for the discrimination study, at n ∈ {15, 25, 40, 60}.

## 6. Metrics

**The scoring rule is achieved coverage at the declared confidence.** Width is
reported in its own column and never enters the verdict. A tighter interval
that under-covers is worse than a loose one that holds: the loose one is
conservative and the engineer reading it knows it, while the tight one is
wrong and reads as a measurement.

| metric | definition | why it is here |
| --- | --- | --- |
| marginal coverage | eq. (25): mean over campaigns of `P(d_new ≤ upper)` on a fixed pool | the primary score |
| tolerance rate | eq. (26): fraction of campaigns whose own coverage reached `p` | the training-conditional statement a tolerance interval makes and a marginal one does not |
| mean limit | the band's mean value, kN.m | the width, reported beside coverage, never instead |
| rank correlation | Spearman's between the conditional median and the true conditional mean gap of eq. (22) | discrimination, which is what the model is actually for |
| quantile crossing rate | fraction of conditions where `upper < median` or `median < lower` | a symptom of thin training data, reported rather than sorted away |

Monte Carlo error is reported with every rate. For a prediction target the
standard error is equation (27),
`sqrt(s²/repetitions + p(1−p)/m)`, which adds the fixed pool's own sampling
error to the repetition error; at `m = 2500` and `p = 0.95` the pool term alone
is 0.0044.

### Measured results — the quantile target, 80 campaigns per cell

| method | n | coverage | ± se | tolerance rate | mean limit [kN.m] | dist-free |
| --- | --- | --- | --- | --- | --- | --- |
| **learned quantile (raw)** | 15 | **0.8761** | 0.0109 | 0.2125 | **10.233** | NO |
| **learned quantile (raw)** | 25 | **0.8754** | 0.0077 | 0.0625 | **9.799** | NO |
| **learned quantile (raw)** | 40 | **0.8769** | 0.0078 | 0.0625 | **9.539** | NO |
| **learned quantile (raw)** | 60 | **0.8843** | 0.0063 | 0.0500 | **9.433** | NO |
| learned, conformalised | 15, 25 | — | — | — | — | no limit exists |
| learned, conformalised | 40 | 0.9571 | 0.0064 | 0.6375 | 14.591 | yes |
| learned, conformalised | 60 | 0.9721 | 0.0050 | 0.8625 | 14.445 | yes |
| split conformal (baseline) | 25 | 0.9577 | 0.0068 | 0.7125 | 14.701 | yes |
| split conformal (baseline) | 40 | 0.9524 | 0.0059 | 0.6250 | 13.528 | yes |
| split conformal (baseline) | 60 | 0.9558 | 0.0053 | 0.6375 | 13.468 | yes |
| Wilks tolerance (baseline) | 60 | 0.9868 | 0.0046 | 0.9750 | 18.292 | yes |

**The result the specification stated in advance, and it is confirmed.** The
raw learned conditional quantile under-covers at **4 of 4** field sample sizes,
by 6 to 9 standard errors, and it is the **tightest** limit at every one of
them. Its per-campaign tolerance rate falls to 0.0500 at n = 60: one campaign
in twenty produces a learned band that covers the declared proportion. This is
a baseline win and it is reported as the research evidence it is. Nothing was
retuned, softened or removed to change it.

### Measured results — the mean target, 80 campaigns per cell

| method | n | coverage | mean upper [kN.m] | excess over the truth | dist-free |
| --- | --- | --- | --- | --- | --- |
| **learned mean plug-in, eq. (29)** | 15 | **0.5750** | 5.2758 | 0.1565 | NO |
| **learned mean plug-in** | 25 | **0.6375** | 5.4187 | 0.2993 | NO |
| **learned mean plug-in** | 40 | **0.7000** | 5.4709 | 0.3516 | NO |
| **learned mean plug-in** | 60 | **0.7750** | 5.4588 | 0.3395 | NO |
| Hoeffding (baseline) | 15 | 1.0000 | 17.7105 | 12.5912 | yes |
| Hoeffding (baseline) | 60 | 1.0000 | 11.5470 | 6.4277 | yes |
| bootstrap percentile (baseline) | 40 | 0.9500 | 6.1660 | 1.0467 | NO |

Equation (29), `upper = mean_x[f(x)] + z_{1−δ}·s_resid/sqrt(n)`, is the
construction a programme would reach for: fit a conditional-mean model,
average over the envelope, attach a normal interval from the residual spread.
Every step is defensible and the result covers the true mean 57.5 per cent of
the time at n = 15 against a declared 95. The in-sample residual standard
deviation understates the error of a model fitted to tens of points, the
normal multiplier assumes a sampling distribution nothing has established, and
the average over the pool ignores the model's bias entirely.

### Measured results — discrimination, 12 draws per cell

| n | rank correlation with the true conditional mean | ± sd | quantile crossing rate |
| --- | --- | --- | --- |
| 15 | 0.5371 | 0.2707 | 0.025300 |
| 25 | 0.6007 | 0.2732 | 0.043600 |
| 40 | 0.6358 | 0.1887 | 0.031333 |
| 60 | 0.6793 | 0.1385 | 0.019867 |

Discrimination and calibration are different things. At these sample sizes the
model ranks operating conditions usefully and does not cover. A reader who
wants to know **where** the twin is worst should use it; a reader who wants a
number to put in a report should use Hoeffding's bound.

## 7. Test-split strategy

- **Training set:** one field campaign of `n` ∈ {15, 25, 40, 60} paired
  measurements, drawn from the `random` (exchangeable) campaign at a seed
  derived from `SEED_FIELD = 770000` and the repetition index.
- **Scoring set:** a **disjoint** 2500-point pool drawn from the same envelope
  at `SEED_POOL = 880000`. The pool is **fixed across repetitions** so that the
  Monte Carlo noise of the inner coverage estimate does not enter the
  comparison between methods, which is the comparison being made; the pool's
  own sampling error is added into the reported standard error by equation
  (27) instead of being ignored.
- **Conformal split:** the campaign is permuted with
  `numpy.random.default_rng(seed)` and split at `calibration_fraction = 0.5`.
  The fit half and the calibration half never overlap, and the conformal
  offset is computed only from the calibration half.
- **No test data is used in any fit**, and no hyperparameter was selected using
  any of it.

## 8. Uncertainty and confidence output

The model is **never** used as a point predictor anywhere in this package.
`DiscrepancyModel.predict_with_uncertainty(features)` returns a
`ConditionalPrediction` carrying, per operating condition:

| field | meaning | unit |
| --- | --- | --- |
| `point` | conditional median discrepancy | kN.m |
| `lower` | conditional `1 − p` quantile | kN.m |
| `upper` | conditional `p` quantile | kN.m |
| `width` | `upper − lower` | kN.m |
| `crossing` | boolean: `upper < point` or `point < lower` | — |
| `lower_level`, `upper_level` | the quantile levels fitted | — |
| `crossing_rate` | fraction of conditions where the quantiles cross | — |

The interval **is** the model's uncertainty statement, and the `crossing` mask
is the model's own admission that the training data was too thin at that
condition. The crossing is reported rather than repaired by sorting, because
sorting hides the thinness.

`ConformalisedDiscrepancyModel.predict_upper` returns the calibrated band, and
`offset` is the additive conformal correction in kN.m — on the declared n = 60
example it is **+16.2046 kN.m**, which is itself the measurement of how far the
raw band was from where it needed to be.

## 9. Failure cases

1. **Under-coverage at every field sample size.** 0.8761 at n = 15 rising only
   to 0.8843 at n = 60, against a declared 0.95. Per-campaign tolerance rate
   as low as 0.0500. This is the dominant failure and the product's headline
   finding.
2. **Tightest and wrongest.** The raw band is 9.4 to 10.2 kN.m against 13.5 to
   18.3 for the limits that hold. A user selecting on sharpness would select
   exactly the band that does not cover.
3. **The plug-in mean bound of equation (29) covers 57.5 per cent of the time
   at n = 15** against a declared 95.
4. **Quantile crossing at 2 to 4 per cent of conditions.** Three independently
   fitted ensembles are not constrained to be ordered.
5. **The learned 0.95 quantile sits below the true conditional 0.95 quantile
   over 85.5 per cent of a Mach sweep** on the declared n = 60 example; see
   `screenshots/learned_discrepancy.png`.
6. **Shuffled labels destroy it completely**, as they must: rank correlation
   with the conditional truth falls from 0.7309 ± 0.0306 to 0.0215 ± 0.0613.
   This is a must-fire negative control and it fired.
7. **Refuses below 5 training rows** rather than returning a number. Three
   gradient-boosted ensembles cannot be fitted to four points in any sense
   worth reporting.
8. **The conformal repair does not exist below n = 40** at a half-and-half
   split, because equation (12) needs 19 calibration scores. The repair is
   unavailable at exactly the sample sizes where the raw model is worst.
9. **Discrimination variance is large at small n:** the rank correlation's
   standard deviation across draws is 0.2707 at n = 15, so a single campaign's
   ranking can be close to useless.
10. **No guarantee under a distribution shift.** The model is fitted and
    scored on the same envelope. Nothing here measures what happens when the
    field campaign and the operational envelope differ, which is the usual
    situation.

## 10. Reproducibility

Exact commands, from `products/P084`:

```bash
python3 validation/validate_bounds.py            # the baseline, FIRST
python3 validation/validate_learned_model.py     # the model, benchmarked against it
python3 validation/validate_negative_controls.py # including the shuffled-label control
python3 examples/learned_discrepancy.py          # the conditional band figure
python3 examples/coverage_vs_width.py            # the coverage-versus-width figure
python3 -m transfergap learned --n 60 --seed 202 # one campaign, interactively
```

Declared seeds, all in `validation/_bootstrap.py` unless noted:

| seed | value | used for |
| --- | --- | --- |
| `SEED_TRUTH` | 20260101 | the population truth, 400 000 draws |
| `SEED_FIELD` | 770000 | field campaigns, offset by sample size and repetition |
| `SEED_POOL` | 880000 | the held-out scoring pool |
| `SEED_MODEL` | 990000 | every ensemble's `random_state` and the conformal split |
| example seeds | 202, 880000, 990000, 5150 | `examples/learned_discrepancy.py` |
| frozen-benchmark seed | 202 | `tests/test_regressions.py` |

**No model artifact is serialised or committed.** Every model reported anywhere
in this repository is refitted from a declared seed by the script that reports
it, so there is no pickle whose format has to be bounded to a library minor
version. `scikit-learn` is still bounded `>=1.9,<1.10`, because the
quantile-loss tree builder's tie-breaking is not part of its public API
contract and the model's measured coverage is a threshold comparison on its
output.

## 11. Compute used

| run | wall clock | hardware |
| --- | --- | --- |
| one upper-quantile fit on 60 rows + predict on 2500 | about 0.06 s | 2 cores |
| one full three-quantile fit + predict | about 0.18 s | 2 cores |
| the whole coverage study, 4 sizes × 80 campaigns | **55.7 s** on the internal timer, **66.5 to 68.8 s** for the whole script across four working directories | 2 cores |
| `validate_negative_controls.py`, including the shuffled-label study | 19.8 to 21.5 s | 2 cores |
| declared budget per script | under 180 s | 2 cores |

No GPU. PyTorch is not a dependency and is not installed in the measured
environment; the model is `scikit-learn` throughout. Peak resident memory for
the coverage study is dominated by the 2500-point pool and is under 400 MB.

## 12. Ethical and safety limits

- **This model is not certified for operational flight use.**
- It is research-grade software. It is not flight-qualified, not certified, and
  not approved for operational aerospace use.
- It is fitted and scored entirely on **synthetic** data. Nothing here
  establishes that it behaves the same way on a real test campaign, and no
  measurement of any real aerospace article appears anywhere in this
  repository.
- **It under-covers its declared confidence at every sample size measured.** It
  must not be used to produce a discrepancy bound for a safety argument, a
  certification artefact, a clearance decision or an airworthiness case. The
  distribution-free bounds in `transfergap.bounds` are what this package
  offers for that purpose, and they are loose on purpose.
- The model's output is a conditional quantile of a synthetic discrepancy. It
  carries no information about the physical cause of a discrepancy and must
  not be read as a diagnosis of a twin, a structure or an instrument.
- No personal data of any kind is involved at any stage.
- The package's own design is adversarial toward its model: the scoring rule
  is coverage, the baseline is validated first, and the negative controls
  include one whose job is to catch the model under-covering. That control
  fired.
