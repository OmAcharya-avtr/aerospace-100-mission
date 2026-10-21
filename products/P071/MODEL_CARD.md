# Model card — conjprob screening classifier

**This model is not certified for operational flight use.**

## Problem

Conjunction screening produces more candidate close approaches than anyone wants
to compute a full collision probability for. The task here is triage: decide,
from a cheap closed-form description of the encounter geometry and the combined
covariance, whether the full Foster-Estes collision probability will exceed a
declared threshold, so that the expensive computation is spent on the cases that
matter.

The declared threshold is `Pc > 1e-4`.

The cost of the two mistakes is not symmetric. A missed high-probability
conjunction is a different kind of failure from a wasted computation, so the
figure of merit is **recall at a fixed false-positive rate**, not accuracy and
not the area under the curve. Accuracy on a population that is two-thirds
negative is uninformative and is not reported.

## Baselines, which were implemented first

Three analytic screens, in increasing order of the information they use. None was
tuned, and none was changed after the learned model was measured against it.

| Screen | Score | Information used |
|---|---|---|
| `miss_distance` | `-log10(d)` | the miss distance alone; thresholding it is the operational practice of flagging everything inside a fixed distance |
| `mahalanobis` | `-M` | the miss distance relative to the covariance, from the encounter-plane projection; invariant to the hard-body radius and to the absolute covariance scale |
| `small_body` | `log10(R^2 / (2 sigma_x sigma_y)) - M^2 / (2 ln 10)` | exactly the features the learned model is given; a monotone function of the leading-order Pc |

The third screen is **not** in the product specification's list of baselines. It
was added because leaving it out would have made the learned model look better
than it is. A screen that is a monotone function of an approximation to the
target is the strongest cheap screen available, and a learned model that cannot
beat it has not earned its place.

## Architecture

A scikit-learn `RandomForestClassifier` (Breiman 2001) over ten features:

- 300 trees, `min_samples_leaf=4`, unlimited depth, `random_state=0`, 2 workers.
- No feature scaling, no feature selection, no hyperparameter search. The forest
  size is the only hyperparameter that was examined, in check 7 of
  `validation/validate_screening_output.txt`: recall at a 1 per cent
  false-positive rate is 0.99046 at 25 trees, 0.99284 at 200, unchanged at 300
  and 0.99213 at 400, so 300 sits on the flat part.
- PyTorch is not available in the build environment and was not needed; the
  problem has ten features and a closed-form solution that is nearly as good.

### Features

| Feature | Units | Note |
|---|---|---|
| `log10_miss_distance_m` | log metres | |
| `log10_radius_m` | log metres | combined hard-body radius |
| `log10_sigma_x_m` | log metres | larger principal standard deviation |
| `log10_sigma_y_m` | log metres | smaller principal standard deviation |
| `log10_aspect_ratio` | dimensionless | |
| `mahalanobis` | dimensionless | the `mahalanobis` baseline is a threshold on this one feature |
| `log10_radius_over_sigma_eff` | dimensionless | |
| `log10_relative_speed_mps` | log m/s | **cannot affect Pc**: it cancels out of the short-encounter integral |
| `abs_cos_psi` | dimensionless | miss-vector direction in the principal frame |
| `log10_sigma_isotropic_3d_m` | log metres | **cannot affect Pc**: the out-of-plane direction is marginalised away |

Two features are included knowing they carry no information about the target.
Their measured importance is the check on whether the model is being misled by
generator structure, and it is reported below.

## Dataset

See `DATASET_CARD.md`. In summary: a committed deterministic synthetic generator
in the dimensionless encounter-plane parameterisation, three declared regimes, no
data committed, nothing identified from any catalogue or vehicle. **The headline
limitation is that it is not the real conjunction population and the performance
figures below do not transfer to a real catalogue.**

## Training procedure

```bash
python -m conjprob screen --n-train 24000 --n-test 12000 --seed 0
python validation/validate_screening.py
```

24000 conjunctions from the `nominal` regime at seed 0, 8279 of them positive
(base rate 0.34496). Trained once and evaluated unchanged on all three regimes;
the fit time is in the compute table below, is wall clock on a shared two-core
container and is unstable between runs. **No model artefact is
committed**: the forest is refitted from the committed generator on every run, so
there is no pickle whose format could outlive the scikit-learn version that wrote
it.

## Test-split strategy

By **seed**, not by shuffling one sample. Training is seed 0; every test set is
seed 1000. No conjunction appears in both and there is no shared draw through
which information could leak. 12000 conjunctions per test regime.

Two of the three test regimes were never trained on, and are reported separately
rather than averaged in.

## Metrics

All figures from `validation/validate_screening_output.txt`, check 2, at a
false-positive-rate target of 0.01. Recall intervals are 95 per cent
Clopper-Pearson.

### `nominal` — the training regime, 4191 positives of 12000

| Screen | Achieved FPR | Recall | 95 per cent interval | Precision |
|---|---|---|---|---|
| `miss_distance` | 0.00999 | 0.30112 | [0.28726, 0.31526] | 0.94179 |
| `mahalanobis` | 0.00999 | 0.03436 | [0.02905, 0.04033] | 0.64865 |
| `small_body` | 0.00999 | 0.97733 | [0.97236, 0.98162] | 0.98131 |
| **`learned`** | 0.00999 | **0.99284** | [0.98980, 0.99517] | 0.98160 |

### `homogeneous` — never trained on, 3016 positives of 12000

| Screen | Achieved FPR | Recall | 95 per cent interval | Precision |
|---|---|---|---|---|
| `miss_distance` | 0.00991 | 0.91280 | [0.90216, 0.92263] | 0.96868 |
| `mahalanobis` | 0.00991 | 0.83720 | [0.82353, 0.85021] | 0.96595 |
| **`small_body`** | 0.00991 | **1.00000** | [0.99878, 1.00000] | 0.97134 |
| `learned` | 0.00991 | 0.98840 | [0.98390, 0.99190] | 0.97101 |

### `dilution` — never trained on, 1388 positives of 12000

| Screen | Achieved FPR | Recall | 95 per cent interval | Precision |
|---|---|---|---|---|
| `miss_distance` | 0.00999 | 0.02954 | [0.02128, 0.03986] | 0.27891 |
| `mahalanobis` | 0.00999 | 0.01297 | [0.00770, 0.02042] | 0.14516 |
| **`small_body`** | 0.00999 | **1.00000** | [0.99735, 1.00000] | 0.92905 |
| `learned` | 0.00999 | 0.99784 | [0.99370, 0.99955] | 0.92891 |

### Paired comparison, learned against the closed form

A per-positive paired count at the same operating point, with a two-sided exact
binomial test on the discordant pairs (check 3).

| Regime | Both found | Learned only | Closed form only | Neither | p |
|---|---|---|---|---|---|
| `nominal` | 4076 | 85 | 20 | 10 | 1.008e-10 |
| `homogeneous` | 2981 | 0 | 35 | 0 | 5.821e-11 |
| `dilution` | 1385 | 0 | 3 | 0 | 0.25 |

### The verdict, stated as measured

- On the regime it was trained on, the learned model beats the one-line closed
  form by **+0.01551 recall** (0.99284 against 0.97733), and the paired test says
  that difference is real (p = 1.008e-10).
- On both regimes it was not trained on, **the closed form wins**: 1.00000
  against 0.98840 on `homogeneous`, which the paired test also calls real
  (p = 5.821e-11), and 1.00000 against 0.99784 on `dilution`, where 3 discordant
  pairs are not enough to resolve the difference (p = 0.25).
- So the learned model is ahead on one regime of three and behind on two. A
  closed-form expression that fits on one line, uses the same features and has no
  training step is the better screen in two of the three populations measured.
- **The product specification predicted that a Mahalanobis threshold would match
  the learned model. On the broad `nominal` population it does not**: it is the
  worst of the three analytic screens, at 0.03436 recall. The reason is
  measurable and was measured. The Mahalanobis distance is invariant to
  `R / sigma_eff`, which spans 3.92251 decades on `nominal` and 0.26125 decades
  on `homogeneous`, and `R / sigma_eff` is half of the leading-order Pc. On
  `homogeneous` the same screen recovers to 0.83720. The specification's
  prediction holds under a condition it did not state, and the condition is the
  spread of `R / sigma_eff` in the population.
- Nothing was retuned after this was measured, and no screen was removed.

### Calibration — a failed check, published rather than repaired

| Regime | Brier score | Base-rate Brier | Skill |
|---|---|---|---|
| `nominal` | 0.011060 | 0.227274 | 0.95133 |
| `homogeneous` | 0.017690 | 0.188165 | 0.90599 |
| `dilution` | 0.005964 | 0.102288 | 0.94170 |

The reliability table on `nominal` (check 4) has a **worst bin gap of 0.272486
against a declared tolerance of 0.15, so this check FAILS.** The forest is
over-confident in the middle bins — the 0.10 to 0.20 bin predicts 0.144293 and
observes 0.000000 — and under-confident at the top, where the 0.70 to 0.80 bin
predicts 0.755616 and observes 0.984375.

It is **not repaired**, for a stated reason. Recall at a fixed false-positive rate
depends only on the ordering of the scores and is invariant under any strictly
monotone recalibration, so fitting an isotonic or Platt recalibration would move
the Brier score and this table and would not move a single recall number above.
Repairing it would make this card look better and change nothing a user of the
screen experiences.

**The consequence for a user: the predicted probability is usable as a ranking
and is not usable as a probability.**

## Uncertainty and confidence output

`ScreeningClassifier.predict_with_uncertainty` returns the ensemble mean
probability and the standard deviation of the individual trees' predicted
probabilities.

That standard deviation is a **disagreement measure across the ensemble, not a
confidence interval**. The distinction is the subject of Wager, Hastie and Efron
(2014), JMLR 15, 1625-1651, and this package does not claim coverage for it.

What was measured instead (check 5) is whether it ranks errors. The mean
dispersion on cases the model gets wrong at the 0.5 decision threshold, against
the mean dispersion on the cases it gets right:

| Regime | Mean dispersion, correct | Mean dispersion, wrong | Ratio | Wrong |
|---|---|---|---|---|
| `nominal` | 0.085890 | 0.420312 | 4.894 | 113 |
| `homogeneous` | 0.138100 | 0.430981 | 3.121 | 115 |
| `dilution` | 0.052934 | 0.419853 | 7.932 | 49 |

So the dispersion is three to eight times larger on the cases that are wrong. It
is usable to route cases to the full computation and is not usable as a
probability interval.

## Feature importances

Mean decrease in impurity, from check 6.

| Feature | Importance | Can affect Pc |
|---|---|---|
| `mahalanobis` | 0.292114 | yes |
| `log10_radius_over_sigma_eff` | 0.245839 | yes |
| `log10_miss_distance_m` | 0.214054 | yes |
| `log10_radius_m` | 0.064877 | yes |
| `log10_sigma_y_m` | 0.063057 | yes |
| `log10_sigma_isotropic_3d_m` | 0.054180 | **no** |
| `log10_sigma_x_m` | 0.043597 | yes |
| `abs_cos_psi` | 0.011828 | yes |
| `log10_aspect_ratio` | 0.007328 | yes |
| `log10_relative_speed_mps` | 0.003125 | **no** |

**0.057305 of the total importance is assigned to two features that provably
cannot affect the target.** Under the short-encounter assumptions the relative
speed cancels out of the collision-probability integral entirely and the
out-of-plane standard deviation is marginalised away. Any importance assigned to
them is the model fitting structure in the generator rather than physics. This is
a limitation of the model, not of the problem, and it is one reason the closed
form generalises better: it cannot be misled this way because it does not look at
those numbers.

The top two features reproduce the algebra of the leading-order approximation,
`Pc ~= R^2 / (2 sigma_x sigma_y) exp(-M^2 / 2)`, which depends on exactly
`mahalanobis` and `log10_radius_over_sigma_eff`. The model has rediscovered the
closed form and does not beat it by much.

## Failure cases

1. **It loses to a one-line closed form on both out-of-distribution regimes.**
   Measured above.
2. **Its probabilities are miscalibrated** by up to 0.272486 on a reliability
   table. Measured above; not repaired.
3. **It assigns 5.7 per cent of its importance to features that cannot matter**,
   so it is partly fitting the generator.
4. **It was trained on a synthetic population that is not the real one.** The
   base rate here is about one third; a real screening population is
   overwhelmingly negative, so every precision figure above is optimistic by a
   factor this product cannot estimate. The recall-at-fixed-false-positive-rate
   figures are computed within class and transfer better, but the *ordering* of
   screens could change on a real population and nothing here can rule that out.
5. **It cannot know when the Pc it is predicting is itself not to be believed.**
   A conjunction on the dilution branch has a Pc that is small for the wrong
   reason, and the model predicts the Pc faithfully, including when the Pc is
   misleading. The dilution diagnosis is a separate, analytic part of the
   package; the model does not substitute for it.
6. **No threshold other than 1e-4 was examined.** A different declared threshold
   is a different classification problem and would need retraining and
   re-measuring.

## Reproducibility

```bash
git clone https://github.com/OmAcharya-avtr/conjprob.git
cd conjprob
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python validation/validate_screening.py     # every number in this card
python examples/screening_performance.py    # screenshots/screening_performance.png
```

Seeds: training population seed 0, test populations seed 1000,
`RandomForestClassifier(random_state=0)`. The measured coverage study in
`validate_montecarlo.py` uses seeds 10000 to 10299. The worked example uses
Monte Carlo seed 20261021.

Determinism: the generator is bit-reproducible for a seed on the pinned numpy.
The forest's predicted probabilities are reproducible to within rounding, not bit
for bit, because the ensemble is averaged across two worker processes and the
summation order is not fixed; `tests/test_screening.py` asserts agreement to
1e-12 relative and says why.

Environment measured against: Python 3.13.16, numpy 2.5.3, scipy 1.18.1,
scikit-learn 1.9.1, 2 CPU cores. Every dependency is bounded on both sides in
`pyproject.toml`; `scikit-learn>=1.9,<1.10` in particular, because the measured
numbers above are a property of that minor version.

## Compute used

| Item | Measured |
|---|---|
| dataset generation, 24000 conjunctions with Foster-Estes labels | under 1 s |
| one forest fit, 300 trees, 24000 x 10 | 6.107 s |
| `validate_screening.py` in full, including six extra fits for check 7 | 31.75 s |
| hardware | 2 shared CPU cores, no GPU |

Declared compute budget for the product: no single training or Monte Carlo run
may exceed three minutes. The largest run here is 32.90 s.

## Ethical and safety limits

- **This model is not certified for operational flight use.**
- No output of this package, learned or analytic, is suitable as an input to a
  manoeuvre decision.
- The model is a triage aid for deciding which conjunctions to compute properly.
  It is not a collision probability and must not be substituted for one.
- The training data is synthetic and declared. Performance on a real catalogue is
  unknown and unmeasurable within this product.
- There is no personal data anywhere in the package.
- A screen that misses a high-probability conjunction is the failure that
  matters; that is why the metric is recall at a fixed false-positive rate, and
  why the closed-form baseline — which wins on two of three regimes — is the one
  the README recommends using.
