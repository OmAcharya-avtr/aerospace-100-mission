# Model card — conformalband learned components

**Package:** `conformalband` 0.1.0 · **Modules:** `conformalband.learned`,
`conformalband.conformal` · **Status:** `TESTING` · **Validation level:** 2

**This model is not certified for operational flight use.**

## 1. Problem

Predict the battery energy a small fixed-wing electric UAV consumes on one
straight leg [Wh] from five covariates (true airspeed, all-up mass, air
density, headwind component, ground distance), and attach a prediction
interval whose coverage can be audited when the deployment covariate
distribution is not the calibration one.

The interval, not the point estimate, is the product. A point prediction with
no uncertainty is not usable for an energy-reserve decision, and an interval
whose coverage was only ever measured in distribution is not usable either.

## 2. Baselines, implemented first

Two baselines, written, tested and benchmarked before any learned component
existed.

**Point prediction: `conformalband.baseline.PhysicsRegressor`.** Bounded
nonlinear least squares on the closed-form energy model of
`conformalband.physics`, four free coefficients (`cd0`, `oswald`, `eta_prop`,
`avionics_power`), airframe geometry treated as known. It is deliberately
misspecified: it assumes a constant propulsive efficiency while the generator
has a quadratic fall-off away from the design airspeed.

**Interval: `conformalband.baseline.GaussianResidualInterval`.**
`yhat +/- z_{1-alpha/2} sigma_hat` with `sigma_hat^2 = RSS/(n - p)` computed on
the held-out calibration sample. Assumes homoscedastic Gaussian residuals and
an unbiased mean model; both are false on this data.

**The baseline wins, and that is the published result.** Root mean squared
error on the same held-out test samples, 40 replicates, seed 57041
(`validation/validate_baseline_vs_learned_output.txt`):

| Shift severity | Physics baseline [Wh] | Learned model [Wh] | Ratio learned/physics | Irreducible noise [Wh] |
|---|---|---|---|---|
| 0.0 | **0.202899** | 0.223296 | **1.100527** | 0.190008 |
| 1.0 | **0.216194** | 0.246565 | **1.140478** | 0.202917 |
| 2.0 | **0.232029** | 0.281556 | **1.213455** | 0.218479 |
| 3.0 | **0.249318** | 0.337629 | **1.354209** | 0.233358 |

The four-coefficient analytic model beats a 200-stage gradient-boosted tree
ensemble at every severity, including in distribution, and its margin widens
with severity because a tree ensemble has no functional form to extrapolate
with. The learned model was not retuned to close the gap and will not be.

The learned model is in this package because the weighted-conformal machinery
and the learned weight estimator have to be exercised on a predictor that has
no physics in it, which is the situation a reader is most likely to be in.

## 3. Architecture

**`LearnedRegressor`** — `sklearn.ensemble.GradientBoostingRegressor` with

```
n_estimators  = 200
max_depth     = 3
learning_rate = 0.05
random_state  = <seed>        # the fit is deterministic given the data
```

squared-error loss, five raw covariates, no feature engineering, no scaling.
Fit cost is about 0.1 s on 1500 rows on one contended core.

**`LearnedWeightEstimator`** — `sklearn.linear_model.LogisticRegression`
(`C = 1.0`, `max_iter = 2000`) on standardised inputs, separating calibration
covariates (label 0) from test covariates (label 1), restricted by default to
the two columns the shift is declared on (`mass_kg`, `headwind_mps`). The
posterior odds are converted to a density ratio,

```
w(x) = [p(x) / (1 - p(x))] * (n_calibration / n_test)
```

which is the classifier reduction of density-ratio estimation (Sugiyama,
Suzuki and Kanamori, *Density Ratio Estimation in Machine Learning*, Cambridge
University Press, 2012, chapter 4). Weights are clipped to `[1/100, 100]` and
the clipped fraction is reported rather than hidden.

**No PyTorch.** Not installed in the build container and not used.

## 4. Uncertainty output

Neither learned component has a native uncertainty output, and neither is used
without one. A point predictor becomes an interval predictor only through
`conformalband.conformal`:

| Wrapper | Guarantee | Assumption |
|---|---|---|
| `SplitConformal` | `P(Y in C) = ceil((n+1)(1-alpha))/(n+1)` | exchangeability of the `n+1` conformity scores |
| `MondrianConformal` | the same, within each declared bin | exchangeability within the bin |
| `WeightedConformal` | `P(Y in C) >= 1 - alpha` under covariate shift | the weights are the **true** likelihood ratio |
| `GaussianResidualInterval` | none; asymptotic at best | homoscedastic Gaussian residuals, unbiased mean |
| `ConformalizedRegressor` | inherits its calibrator's | as above |

`LearnedWeightEstimator.auc` is the weight model's own confidence output: the
in-sample ROC AUC of the calibration-versus-test classifier. A value near 0.5
means the two samples are indistinguishable and the learned weights carry no
information; it does **not** mean the shift is absent.

## 5. Training and test-split strategy

Three disjoint samples per replicate, drawn in this order from one seeded
generator:

| Sample | Size | Distribution | Use |
|---|---|---|---|
| fit | 1500 | calibration | fits the point predictor only |
| calibration | 500 | calibration | conformity scores, `sigma_hat`, Mondrian bin edges, weight-model class 0 |
| test | 600–1000 | **shifted** | evaluation only |

The point predictor never sees the calibration sample, and nothing sees the
test targets. `LearnedWeightEstimator` sees test **covariates** and no test
targets, which is the transductive setting weighted conformal assumes.

Replicate `r` uses `seed + r`. All reported figures pool over 30 to 120
replicates.

## 6. Metrics

Primary metric is **empirical coverage against the nominal level**, with
interval width as the cost. Reported with a 95 per cent Student-t interval
across replicate coverages, because test points inside one replicate share a
calibration quantile and a Clopper-Pearson interval on the pooled count is two
to three times too narrow. Both are printed.

Headline rows from `validation/validate_coverage_audit_output.txt` (120
replicates, `n_calibration = 500`, `n_test = 600`, seed 57001, alpha 0.1,
72 000 test points per row, learned point predictor):

| Method | Severity 0 | Severity 1 | Severity 2 | Severity 3 | Width at severity 0 [Wh] | Width at severity 3 [Wh] |
|---|---|---|---|---|---|---|
| `parametric` (baseline) | 0.90796 | 0.88151 | 0.84772 | **0.80228** | 0.72385 | 0.72385 |
| `split` | 0.89868 | 0.87132 | 0.83642 | **0.79047** | 0.69997 | 0.69997 |
| `mondrian` | 0.90418 | 0.89554 | 0.87992 | **0.85129** | 0.71140 | 0.82696 |
| `weighted_declared` | 0.89868 | 0.89746 | 0.89968 | 0.91201 | 0.69997 | 1.07147 |
| `weighted_learned` | 0.89911 | 0.89860 | 0.89978 | 0.91203 | 0.70118 | 1.06146 |

The finite-sample split-conformal bound at `n = 500, alpha = 0.1` is
`451/501 = 0.900199600798`, inside the window `[0.9000, 0.901996007984]`.

Learned weights match the exact declared weights to within 0.0015 of coverage
at every severity on this problem. That is not a general result: the declared
shift is a Gaussian mean shift in two coordinates, so the log likelihood ratio
is exactly linear in them and the logistic model is correctly specified for it.

## 7. Failure cases

1. **Under shift, every marginal method loses coverage.** Worst measured row:
   `learned/split` at severity 3, coverage **0.79047**, 95 per cent interval
   `[0.78615, 0.79480]`, against a nominal 0.90. The baseline parametric
   interval is no better: 0.80228.
2. **Weighted conformal breaks when the declared weights are wrong.** Measured
   breaking fraction (the largest assumed-shift fraction whose whole 95 per
   cent coverage interval lies below nominal) is **0.85** at true severity 1,
   **0.90** at true severity 2 and **0.85** at true severity 3. Understating
   the declared shift by 10 to 15 per cent is enough to lose the guarantee.
   With no weighting at all the coverage is 0.87324, 0.83653 and 0.79054.
3. **Weighted conformal runs out of calibration points.** Kish effective
   sample size falls from 500 to **418.5, 248.3, 116.1** at severities 1, 2 and
   3, and the per-replicate coverage standard deviation rises from 0.0173 to
   **0.0424**. At severity 3, 0.58 per cent of test points get a genuinely
   unbounded interval, which is reported as `inf` rather than truncated.
4. **Conditional coverage is much worse than marginal coverage, with no shift
   at all.** Marginal split conformal covers **0.95972** in the lowest tercile
   of predicted energy and **0.81920** in the highest, in distribution.
   Mondrian reduces the spread from 0.14053 to **0.00561** there, and from
   0.18203 to 0.04649 at severity 2, by making the interval 1.53 times wider in
   the top tercile than the bottom.
5. **Mondrian does not restore marginal coverage under shift.** At severity 3
   it reaches 0.85129 against a nominal 0.90. Conditioning on a prediction
   tercile is not conditioning on the covariate that moved.
6. **The learned regressor degrades faster than the analytic one under
   shift**, by construction of what a tree ensemble is: see the table in
   section 2.
7. **The learned weight estimator reports AUC in-sample**, so it is optimistic
   on small samples. It is a diagnostic, not a hypothesis test.

## 8. Reproducibility

Exact commands, run from the repository root with `PYTHONPATH=src`:

```bash
python -m pytest tests/ -q --junit-xml=junit.xml
python validation/validate_environment.py
python validation/validate_known_answers.py
python validation/validate_baseline_vs_learned.py
python validation/validate_coverage_audit.py
python validation/validate_breaking_point.py
python validation/validate_conditional_coverage.py
python validation/validate_cli.py
python validation/worked_example.py
```

Seeds: 57001 (coverage audit, worked example), 57021 (breaking point), 57031
(conditional coverage), 57041 (baseline comparison), 57101 (known answers).
Every dataset is a deterministic function of its seed; the two consecutive full
runs recorded in this session reproduced every quoted figure exactly.

## 9. Compute used

Two shared, contended CPU cores, under 1 GiB resident. No GPU, no compiled
extension, no hardware in the loop.

| Script | Wall clock |
|---|---|
| `python -m pytest tests/ -q` (440 tests) | 60 s, 89 s and 61 s on three runs |
| `validate_coverage_audit.py` | 106–156 s |
| `validate_breaking_point.py` | 202–237 s |
| `validate_conditional_coverage.py` | 35–38 s |
| `validate_baseline_vs_learned.py` | 28 s |
| `validate_cli.py` | 33–35 s |
| `validate_known_answers.py` | 12 s |

Timings move 10 to 40 per cent between runs on contended cores. Quote the
ratios and the counts, not the seconds. No single computation approaches the
three-minute standalone budget except `validate_breaking_point.py`, which
sweeps 31 weight models at three true severities over 120 replicates and is
the slowest thing in the repository.

## 10. Ethical and safety limits

- **Research-grade.** Not flight-qualified, not certified, not approved for
  operational aerospace use. It is not evidence for any airworthiness or
  certification argument.
- **The data is synthetic and the airframe is illustrative.** No number here
  describes a real vehicle, and no coverage figure transfers to one.
- **Every guarantee is conditional on a declaration this software cannot
  check.** Exchangeability, for split and Mondrian conformal, is an assumption
  about your data and is false under the shift this package was built to
  measure. The likelihood ratio, for weighted conformal, is something the user
  declares; section 7 item 2 is the measured cost of declaring it wrong.
- **Using an interval from this package as an energy reserve is a safety
  decision this package does not support.** A 90 per cent interval is wrong one
  flight in ten by construction, and under an undeclared shift it is wrong far
  more often.

## 11. Alternatives

For conformal prediction in production use `MAPIE` 1.5.0 or `crepes` 0.9.1, and
for weighted conformal with user-supplied weights use `puncc` 0.9.3. All three
are named, version-checked and described in README.md, and none is a runtime
dependency here.
