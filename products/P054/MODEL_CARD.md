# Model card — rareverify Gaussian-process limit-state surrogate

**Package:** `rareverify` 0.1.0 · **Module:** `rareverify.surrogate` ·
**Status:** `TESTING` · **Validation level:** 2

**This model is not certified for operational flight use.**

## 1. Problem

Importance sampling in the mean-shift family needs a mean shift. The
variance-minimising choice within the family is the design point of the limit
state — the most probable point of the failure region. For a linear or
lognormal limit state that point is available in closed form and costs nothing.
For a limit state available only as a simulator it is not, and a campaign that
guesses it badly gets an estimator that is worse than plain Monte Carlo
(measured: 26 of 40 swept tilts, `validation/validate_is_worse.py`).

The model learns the limit-state function from a small number of true
evaluations, so that the design point can be located on the learned function
instead. It is a simplified, one-shot version of the Kriging-plus-Monte-Carlo
family of Echard, B., Gayton, N. and Lemaire, M. (2011), "AK-MCS: an active
learning reliability method combining Kriging and Monte Carlo Simulation",
*Structural Safety* 33(2):145–154. No active-learning loop is implemented.

## 2. Baseline, implemented first

`rareverify.tilting.analytic_mean_shift`: the tilt at the limit state's
analytically known design point. It was written, tested and benchmarked before
any learned component existed, and every comparison in this card is against it
on an **equal budget of true limit-state evaluations**, with the surrogate's
training set counted against the surrogate.

A second reference, `oracle_mean_shift`, locates the true design point by
numerical optimisation on the true limit state. It is not a method — it spends
unlimited exact evaluations inside an optimiser — and it exists to bound what
any mean shift could achieve.

## 3. Architecture

`sklearn.gaussian_process.GaussianProcessRegressor` with

```
kernel = ConstantKernel(var(g), (1e-6, 1e10))
       * RBF(ones(d), (1e-2, 1e4))                  # anisotropic
       + WhiteKernel(1e-4 var(g), (1e-8 var(g), 1e1 var(g)))
normalize_y = True
n_restarts_optimizer = 0
```

Two distinct uses, and the difference is the whole argument:

1. **Surrogate as the proposal** (`surrogate_guided_importance_sampling`). The
   surrogate supplies `theta` only; every sample is evaluated on the **true**
   limit state. The estimator is unbiased for any surrogate. A bad surrogate
   costs variance, not correctness. This is the configuration benchmarked and
   the one a campaign should use.
2. **Surrogate as the estimator** (`surrogate_probability`). The probability is
   computed from the learned limit state. **Biased**, by exactly the
   surrogate's error near the boundary, and the bias is not covered by the
   estimator's reported standard error. Implemented so the bias can be
   measured, not as a recommendation.

Notes on the kernel, both measured rather than assumed:

- The white-noise lower bound is `1e-8 var(g)` rather than zero. The limit
  states are deterministic, so the marginal likelihood drives the noise to its
  bound; at a strictly zero bound scikit-learn 1.9.1 produces numerically
  negative posterior variances, which propagate straight into the uncertainty
  output.
- On an exactly linear limit state the RBF length scale saturates at its upper
  bound and scikit-learn emits a `ConvergenceWarning` saying so. That is the
  correct diagnosis — a linear function is the infinite-length-scale limit of
  an RBF — and it is one structural reason the surrogate has nothing to add on
  a smooth instance.

## 4. Dataset

Fully synthetic, generated deterministically by committed code. See
[DATASET_CARD.md](DATASET_CARD.md). Design: directions uniform on the unit
sphere, radii uniform on `[0, radius_scale * |Phi^-1(p_prior)|]`, with
`p_prior` a declared prior order of magnitude for the failure probability and
`radius_scale = 1.4` by default.

**Limitation stated rather than hidden:** that design uses prior information
about where the failure region is. Without it, a design drawn from the input
distribution itself would contain no point anywhere near a `1e-4` boundary and
the surrogate would be useless. The analytic baseline is given the exact design
point, so neither side is unfairly handicapped, but neither is operating
without prior knowledge.

## 5. Training procedure

```python
fit = fit_surrogate(limit_state, n_train=100, p_prior=1e-4,
                    radius_scale=1.4, rng=np.random.default_rng(606),
                    n_restarts=0)
```

Exactly `n_train` true limit-state evaluations are spent; `n_train` is capped
at 2000 because a Gaussian-process fit is cubic in the design size. The
marginal likelihood is maximised once from the initial kernel.

**Restarts buy nothing, measured** (`validation/validate_surrogate.py` §6):

| Instance | `n_restarts_optimizer` | fit + search seconds | recovered `beta_hat` |
|---|---|---|---|
| smooth | 0 | 2.14 | 3.719036 |
| smooth | 2 | 2.15 | 3.719036 |
| rough | 0 | 1.95 | 3.097913 |
| rough | 2 | 2.04 | 3.097913 |

Identical to six decimal places. The default is 0 and the compute budget goes
to samples.

## 6. Test-split strategy

There is no held-out split in the usual sense, because the quantity being
tested is not a prediction accuracy. The surrogate is fitted on `n_train` true
evaluations and then evaluated on an **entirely disjoint** set of
`budget - n_train` freshly drawn samples, each of which is evaluated on the
**true** limit state, and the resulting probability estimate is compared with a
reference probability the surrogate never saw — closed form for the linear and
lognormal limit states, 400-node Gauss-Hermite quadrature for the rippled ones.
The reference is independent of both the model and the sample.

Replication, not a single split, is how spread is measured: 12 independent
surrogate runs (fit and sample, independent seeds) against 40 independent runs
of each analytic method.

## 7. Metrics

Measured at a budget of **60 000 true limit-state evaluations for every
method**, 2-core container, seeds fixed in `validation/validate_surrogate.py`.

### Design-point recovery

| Instance | `n_train` | `beta_hat` | relative error | 2-sigma band |
|---|---|---|---|---|
| smooth (true 3.719) | 25 | 3.718990 | −2.607e-06 | 3.71791 – 3.72007 |
| smooth | 100 | 3.719014 | 3.793e-06 | 3.71810 – 3.71993 |
| smooth | 200 | 3.719052 | 1.398e-05 | 3.71813 – 3.71997 |
| rough (true 3.097896) | 25 | 3.099064 | 3.768e-04 | 3.09561 – 3.10383 |
| rough | 100 | 3.097933 | 1.175e-05 | 3.09756 – 3.09956 |
| rough | 200 | 3.097919 | 7.244e-06 | 3.09888 – 3.10015 |

The analytic smooth-part design point on the rough instance is 5.5, which is
**77.5 % too large**.

### Replicated benchmark

| Instance | Estimator | mean estimate | measured spread | cov | VRF vs crude |
|---|---|---|---|---|---|
| smooth, `p` = 1.00006526e-04 | crude | 1.016667e-04 | 4.557e-05 | 0.4483 | 1 |
| | analytic-IS (baseline) | 9.993796e-05 | 7.918e-07 | 0.0079 | 3313 |
| | oracle-IS (reference) | 1.000075e-04 | 9.407e-07 | 0.0094 | 2347 |
| | **surrogate-guided-IS** | 1.000785e-04 | 7.547e-07 | 0.0075 | 3646 |
| rough, `p` = 1.85308155e-04 | crude | 1.995833e-04 | 5.371e-05 | 0.2691 | 1 |
| | analytic-IS (baseline) | 1.854003e-04 | 1.405e-05 | 0.0758 | 14.62 |
| | oracle-IS (reference) | 1.842105e-04 | 3.587e-06 | 0.0195 | 224.2 |
| | **surrogate-guided-IS** | 1.847665e-04 | 4.303e-06 | 0.0233 | 155.8 |

Head to head against the baseline at equal true evaluations:

| Instance | variance ratio surrogate / baseline | 3-sigma replication-noise band | verdict |
|---|---|---|---|
| smooth | **1.1006** | 0.485 – 2.063 | **no resolvable gain** |
| rough | **10.6563** | 0.485 – 2.063 | **surrogate wins** |

## 8. Uncertainty and confidence output

The model does not return a point estimate alone.

- `SurrogateFit.predict(x, return_std=True)` returns the Gaussian-process
  posterior mean and standard deviation of `g`.
- `surrogate_design_point(fit, k_sigma=2.0)` locates the design point three
  times — on the posterior mean and on the mean plus and minus `k_sigma`
  posterior standard deviations — and returns `beta`, `beta_lower`,
  `beta_upper` and the corresponding first-order probabilities. That band is
  the model's statement of how well it knows where the failure boundary is. At
  `n_train = 100` on the rough instance it is 3.09756 – 3.09956 around 3.097933,
  i.e. ±0.003 % on the reliability index.
- `SurrogateFit.straddle_fraction(x, k=2.0)` returns the fraction of points
  whose posterior band crosses zero, i.e. where the model does not know which
  side of the limit state it is on. Measured around the design point: 0.0000 to
  0.0040 across the sample-efficiency curve.
- `SurrogateFit.diagnostics` carries `train_rmse`, `noise_level`,
  `g_train_std` and `log_marginal_likelihood`.

## 9. Failure cases, measured

1. **It cannot beat an exact free design point.** On a smooth analytic limit
   state the surrogate's tilt is `[3.71900589, 0.0]` against the analytic
   `[3.719, 0.0]` — a distance of 5.890e-06 standard-normal units — so the two
   estimators are the same estimator and the surrogate's 100 training
   evaluations are a deterministic 0.167 % variance penalty plus its fit and
   search time. The observed ratio of 1.1006 is inside the ±3-sigma
   replication-noise band and is not evidence of a gain. **No retune was
   attempted; the result is structural.**
2. **It does not reach the oracle.** On the rough instance it achieves VRF
   155.8 against the oracle's 224.2, a shortfall of 31 %.
3. **Used as the estimator rather than the proposal, it is biased and its error
   bar does not know.** On a high-frequency rippled limit state (`w = 6`,
   reference 4.48938469e-04):

   | `n_train` | surrogate-only estimate | relative error | its own se | \|error\|/se |
   |---|---|---|---|---|
   | 20 | 3.298484e-05 | −0.9265 | 8.129e-07 | **511.66** |
   | 30 | 2.823918e-04 | −0.3710 | 9.682e-06 | **17.20** |
   | 60 | 4.500197e-04 | +0.0024 | 1.110e-05 | 0.10 |
   | 240 | 4.369609e-04 | −0.0267 | 1.063e-05 | 1.13 |

   The nuance, reported rather than suppressed: at `n_train` of 60 and above on
   these two-dimensional problems the bias falls below the Monte-Carlo noise
   and stops being detectable at that budget. It is a small-design failure, not
   a universal one. The surrogate-**guided** estimator at the same `n_train` of
   20 is wrong by 67.2 %, not by a factor of 14 — the bad surrogate costs it
   variance, not correctness.
4. **The sample-efficiency curve is flat.** 25 training evaluations already
   place the tilt well enough on these two-dimensional problems; more do not
   help. There is no interesting training-size trade-off to report here, and
   that is stated rather than dressed up as a finding.
5. **It inherits the ray search's dimensional limitation.** Without the SLSQP
   polish step the design-point search is wrong by 8.7 % in six dimensions at
   192 directions. The polish is on by default; `polish=False` exposes it.
6. **It is not adaptive.** A one-shot design cannot refine itself where the
   posterior is uncertain, which is exactly what AK-MCS does and this does not.

## 10. Reproducibility

Exact commands and seeds. Library versions: Python 3.13.16, numpy 2.5.3, scipy
1.18.1, scikit-learn 1.9.1.

```bash
python validation/validate_surrogate.py      # 10 checks, 97.6 s, writes validate_surrogate.txt
python examples/surrogate_benchmark.py       # writes screenshots/surrogate_benchmark.png
python -m rareverify surrogate --limit-state rippled --n-train 100 --samples 100000 --seed 0
```

Seeds inside `validation/validate_surrogate.py`: 555 (design-point recovery,
per `n_train`), 606 (structural comparison), 701–704 (replicated benchmark,
with replication `i` using `numpy.random.default_rng([seed, i])`), 808
(sample-efficiency curve), 909 (surrogate-only bias), 1010 (restart study), 1
and 3 (oracle searches). All randomness goes through
`numpy.random.default_rng`; the Gaussian process's own `random_state` is drawn
from that generator, so a fixed seed reproduces the fit exactly.

## 11. Compute used

Training is minutes of CPU, not hours. On the 2-core build container
(`validation/validate_compute_budget.py`):

| `n_train` | fit (s) | design-point search (s) | total (s) |
|---|---|---|---|
| 50 | 0.13 | 2.08 | 2.21 |
| 100 | 0.07 | 2.21 | 2.28 |
| 200 | 0.28 | 2.29 | 2.57 |
| 400 | 1.14 | 1.98 | 3.12 |
| 800 | 6.75 | 2.76 | 9.51 |

The full surrogate validation, including 24 replicated surrogate runs and the
bias study, takes 97.6 s. Fit times are erratic at the 2x level between
otherwise identical calls because the L-BFGS iteration count varies with the
initial kernel; they are budgeting figures only and are not a characteristic of
the method or of any hardware. No GPU, no accelerator, no distributed training.

## 12. Ethical and safety limits

- **This model is not certified for operational flight use.**
- It is trained on synthetic limit states defined in
  `rareverify.limitstates`. It has never seen data from any vehicle, test
  campaign, or physical system, and a probability it helps estimate is the
  probability of **the model that was simulated**, not of any real failure.
- Used as the estimator rather than the proposal it is biased and reports a
  standard error that does not cover the bias; §9 case 3 quantifies that. Any
  use of `surrogate_probability` in evidence would be a misuse of this package.
- The surrogate's uncertainty band is a Gaussian-process posterior under a
  chosen kernel. It is a statement about interpolation uncertainty on the
  training design, not about whether the limit state itself is the right one.
- No personal data, no human subjects, no dual-use concern beyond the general
  one that any reliability tool can be quoted out of context. The README and
  this card both state the model-versus-vehicle distinction explicitly for
  exactly that reason.
