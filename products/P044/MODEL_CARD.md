# Model card: learned combiner weighting under imperfect channel-state information

**Package:** `aperturediv` 0.1.0 · **Module:** `aperturediv.learned` ·
**Status:** TESTING · Validation level 2 · Apache-2.0 · © 2026 OPTIMA Organisation

**This model is not certified for operational flight use.** It is
research-grade software for studying a channel model. It has not been
validated against a measured optical link, and nothing in it substitutes for
one.

---

## 1. The problem, and what can honestly be claimed about it

A receiver with `L` apertures must choose a weight vector `w` to apply to the
branch amplitudes before summing. For unit-norm `w` and true branch
amplitudes `h`, the post-combining SNR is proportional to `(w . h)^2`, and
Cauchy–Schwarz gives

```
(w . h)^2 <= ||h||^2      with equality only for w ∝ h
```

So **maximal-ratio combining with the true channel state is optimal, and no
learned combiner can beat it.** This is Brennan's result (*Proc. IRE* 47(6),
1959). A model card that claimed a learned combiner outperformed MRC under
perfect CSI would be describing a defect — most likely a label leak — not a
result.

The only undetermined question is what to do when the state the receiver has
is **wrong**: noisy, stale, or both. That is the entire scope of this model.
It is also where equal-gain combining becomes interesting, because equal gain
never looks at the estimate and is therefore immune to estimation error.

Two consequences are stated up front because they are the published result:

1. **At zero estimation error the analytic baseline is exactly optimal and
   the learned model is worse.** Measured: at `sigma_e = 0`, MRC-with-the-
   estimate has a mean penalty of `-0.0000 dB` (it *is* MRC-with-the-truth)
   and the learned combiner has `+0.0084 dB`. The baseline wins. That is not
   tuned away.
2. **Once the estimation error passes a measured threshold, equal-gain
   combining — which needs no model and no estimate — is the better choice
   than trusting the estimate.** Measured crossover: `sigma_e = 0.8320`,
   i.e. **3.613 dB of irradiance estimation error**
   (`validation/validate_learned_combiner.py`, check 4). This is an analytic
   finding, not a learned one, and it is the most useful output of the whole
   exercise.

## 2. Baselines, all evaluated on the same held-out rows

Four non-learned references. The first is an upper bound; the other three are
things a real receiver can do.

| reference | weights | CSI needed | role |
|---|---|---|---|
| `mrc_true` | `w ∝ sqrt(I)` | the truth | unreachable upper bound; **0 dB by construction** |
| `mrc_estimated` | `w ∝ sqrt(Ihat)` | the estimate | what a real receiver does |
| `egc` | `w = 1/sqrt(L)` | none | immune to estimation error |
| `shrinkage(p)` | `w ∝ Ihat^(p/2)` | the estimate | one-parameter analytic family; `p = 1` is `mrc_estimated`, `p = 0` is `egc`; `p` fitted by grid search on the **validation** split |

`shrinkage` exists specifically so that any advantage the learned model shows
can be attributed. If the learned model does not clearly beat `shrinkage`,
the honest conclusion is that the whole benefit of learning here is
*shrinkage towards equal gain*, and the one-line analytic rule is the result
worth publishing. Fitted value on the headline configuration: **p* = 0.1600**.

## 3. Architecture

PyTorch is not available in this environment. The model is scikit-learn.

- **Weight head.** One `sklearn.ensemble.HistGradientBoostingRegressor` per
  aperture (`L` of them), each predicting one component of the unit-norm
  optimal weight direction `h/||h||`. Predictions are clipped to be
  non-negative (a negative weight against a non-negative amplitude can only
  reduce `w . h`) and then renormalised to unit norm.
- **Uncertainty head.** Three further `HistGradientBoostingRegressor`s with
  `loss="quantile"` at quantiles 0.1, 0.5 and 0.9, predicting the **SNR
  penalty in dB that this combiner will incur on this row**. They are fitted
  on the penalties the weight head actually incurs on the training rows, so
  the output describes this combiner rather than a generic one.
- Hyperparameters: `max_iter=100`, `learning_rate=0.12`,
  `max_leaf_nodes=31`, `early_stopping=False`, `random_state=44044`. Chosen
  for a training run that fits the compute budget; no hyperparameter search
  was performed, and that is a limitation, not a claim of optimality.

Public entry point: `LearnedCombiner.combine(irradiance_estimated, sigma_e)`
returns `(weights, penalty_quantiles_db)`. It takes only quantities a
receiver has.

## 4. Features — and why the truth cannot reach the model

`aperturediv.datasets.build_features(irradiance_estimated, sigma_e)` returns
`L + 3` columns: `ln Ihat_k` for each aperture, then the mean and the
standard deviation of those logs across apertures, then `sigma_e` itself (an
estimator knows its own accuracy).

**The function takes no argument that carries the true irradiance.** Its
signature is checked in `tests/test_datasets.py`
(`test_takes_no_truth_argument`) and the feature matrix is re-derived from
the estimate alone in `validation/validate_learned_combiner.py` (check 2).
The truth enters only as the regression target.

## 5. Dataset

See `DATASET_CARD.md`. Summary for the headline configuration:

| field | value |
|---|---|
| rows | 120000, generated, nothing committed |
| split | 70000 train / 25000 validation / 25000 test, by row index |
| apertures `L` | 4, line array, pitch 0.15 m |
| irradiance correlation scale `rho_c` | 0.10 m |
| adjacent-aperture log correlation | 0.105399 |
| marginal scintillation index | 0.9 |
| estimation error `sigma_e` | uniform on [0, 3.0] per row, i.e. [0, 13.03] dB |
| seed | 44044 |

Rows are i.i.d. draws of one symbol interval, so an index split is a valid
held-out split: there is no temporal structure to leak across it. The
evaluation sets used for the per-error-level curves are drawn from separate
seeds (`44544 + level`, `44944 + step`), so they share no row with training.

## 6. Metrics

Primary metric: **SNR penalty against MRC-with-the-truth**, in dB,

```
penalty_dB = 10 log10( ||h||^2 ||w||^2 / (w . h)^2 )   >= 0 for every row
```

Secondary metric: mean BPSK BER at a stated branch `Eb/N0`, averaging the
conditional error probability `Q(sqrt(2 gamma_out))` over held-out rows.

### Pooled held-out results (25000 rows, `sigma_e` uniform on [0, 3])

| combiner | mean dB | median dB | p90 dB | max dB | mean BER at 10 dB |
|---|---|---|---|---|---|
| `mrc_true` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.082324e-08 |
| `mrc_estimated` | 1.2278 | 0.7275 | 3.1630 | 14.6171 | 2.059059e-05 |
| `egc` | 0.4411 | 0.3603 | 0.8941 | 2.9260 | 1.281703e-07 |
| `shrinkage p=0.160` | 0.3667 | 0.2892 | 0.7549 | 3.1880 | 1.239550e-07 |
| **`learned`** | **0.2948** | **0.2124** | **0.6676** | **2.6016** | **1.225447e-07** |

Read honestly:

- The learned combiner recovers **0.933 dB** of the 1.228 dB that naive
  MRC-on-the-estimate loses.
- Against the best *analytic* rule it gains **0.072 dB**. That is the real
  size of the machine-learning contribution on this configuration.
- It is still **0.295 dB** short of the unreachable bound, and that gap
  cannot be closed, because the truth is not available.
- Its BER is 1.730x the bound; equal gain's is 1.810x. The difference between
  the learned combiner and the no-model option is **4.4 %** in BER.

### Resolved by estimation error level (20000 fresh rows per level)

Mean penalty in dB:

| `sigma_e` | dB | `mrc_estimated` | `egc` | `shrinkage` | `learned` | best non-learned |
|---|---|---|---|---|---|---|
| 0.000 | 0.000 | **-0.0000** | 0.4402 | 0.3005 | 0.0084 | `mrc_estimated` |
| 0.250 | 1.086 | 0.0441 | 0.4373 | 0.3002 | 0.0447 | `mrc_estimated` |
| 0.500 | 2.171 | 0.1708 | 0.4361 | 0.3036 | 0.1299 | `mrc_estimated` |
| 0.750 | 3.257 | 0.3668 | 0.4344 | 0.3095 | 0.2127 | `shrinkage` |
| 1.000 | 4.343 | 0.6023 | 0.4401 | 0.3217 | 0.2730 | `shrinkage` |
| 1.250 | 5.429 | 0.8809 | 0.4394 | 0.3341 | 0.3175 | `shrinkage` |
| 1.500 | 6.514 | 1.1714 | 0.4395 | 0.3484 | 0.3470 | `shrinkage` |
| 2.000 | 8.686 | 1.7498 | 0.4344 | 0.3832 | 0.3809 | `shrinkage` |
| 2.500 | 10.857 | 2.2805 | 0.4345 | 0.4265 | 0.3973 | `shrinkage` |
| 3.000 | 13.029 | 2.7497 | 0.4369 | 0.4846 | 0.4176 | `egc` |

The `egc` column is flat to within Monte Carlo noise across a 13 dB range of
estimation error, because equal gain does not read the estimate. That flat
line is what the crossover is measured against.

The learned combiner is ahead of every non-learned option at **8 of 10**
levels, by at most **0.097 dB** (at `sigma_e = 0.75`). At `sigma_e = 0` it is
behind by 0.008 dB — the regime where the baseline is provably unbeatable —
and at `sigma_e = 0.25` it is behind by 0.0006 dB, which is below the
resolution of a 20000-row evaluation and should be read as a tie rather than
as a loss.

### Sensitivity to aperture count

| `L` | `p*` | `mrc_estimated` dB | `egc` dB | `shrinkage` dB | `learned` dB | best |
|---|---|---|---|---|---|---|
| 2 | 0.140 | 0.6817 | 0.2533 | 0.2121 | 0.1728 | learned |
| 3 | 0.160 | 1.0001 | 0.3680 | 0.3085 | 0.2488 | learned |
| 4 | 0.160 | 1.1971 | 0.4382 | 0.3602 | 0.2899 | learned |

Refitted at a smaller size (45000 rows, 27000 train, 60 boosting iterations)
because this is a sensitivity check and not the headline fit. The learned
margin over `shrinkage` grows slowly with `L`
(0.039 / 0.060 / 0.070 dB) and stays under a tenth of a decibel.

## 7. Uncertainty output

`predict_penalty_quantiles` returns the predicted 10th, 50th and 90th
percentile of the penalty for each row, clipped at 0 dB. Held-out calibration
(25000 rows):

| nominal | empirical coverage | mean predicted dB |
|---|---|---|
| 0.10 | 0.094360 | 0.0563 |
| 0.50 | 0.465720 | 0.2167 |
| 0.90 | 0.873240 | 0.5489 |

Within 0.035 absolute at every level. All three are **under**-covered, i.e.
mildly optimistic, and the median is the worst of the three; a user who needs
a conservative bound should treat the 0.9 output as roughly a 0.87 bound, not
a 0.9 one.

## 8. Error model, and why there is only one error axis

`aperturediv.estimation` builds the estimate from the channel as it was
`tau` ago plus measurement noise, both in the log domain. Both mechanisms
enter only through

```
sigma_e = sqrt(2 s^2 (1 - rho_t) + sigma_m^2),    s^2 = ln(1 + si)
```

so a stale estimate and a noisy one are the same thing to a combiner at equal
`sigma_e`. Measured (check 6, 40000 rows per row of the table):

| `rho_t` | `sigma_m` | `sigma_e` | `mrc_estimated` penalty dB | `egc` penalty dB |
|---|---|---|---|---|
| 1.00 | 1.0000 | 1.0000 | 0.6089 | 0.4343 |
| 0.95 | 0.9674 | 1.0000 | 0.6178 | 0.4363 |
| 0.90 | 0.9336 | 1.0000 | 0.6227 | 0.4360 |
| 0.80 | 0.8621 | 1.0000 | 0.6325 | 0.4350 |

Spread 0.024 dB over the four rows against a mean of 0.62 dB, so the single
axis is sufficient at this resolution. There is a weak monotone drift with
`rho_t` at fixed `sigma_e`, which is recorded here as a residual effect
rather than claimed to be zero.

## 9. Failure cases

- **Zero estimation error.** The learned model is 0.008 dB worse than the
  analytic optimum. Use `mrc_estimated` when the estimate is good.
- **Small estimation error.** At `sigma_e = 0.25` the learned model and
  `mrc_estimated` are level to 0.0006 dB, which is below the resolution of
  the evaluation. There is nothing to gain from the model in that regime.
- **Very large estimation error.** At `sigma_e = 3.0` the learned combiner
  leads by 0.019 dB over `egc`; below that resolution the model is not worth
  shipping and `egc` is the right answer.
- **Worst-case rows.** The learned combiner's maximum penalty on the
  held-out set is 2.60 dB, against 2.93 dB for `egc` and 3.19 dB for
  `shrinkage`. It reduces the mean far more than it reduces the tail: on the
  worst rows, where one aperture's estimate is badly wrong, no weighting rule
  recovers much.
- **Extrapolation in `sigma_e`.** `sigma_e` is a feature. Queried outside the
  trained range [0, 3.0], the gradient-boosted trees extrapolate flat; the
  model does not know it is out of range. There is no guard for this and that
  is a limitation.
- **Extrapolation in geometry.** The model is trained at one `si`, one pitch
  and one correlation scale. It is not expected to transfer to a different
  correlation structure, and that has not been tested.
- **Aperture count is fixed at fit time.** `L` is the number of regressors; a
  model fitted at `L = 4` cannot serve `L = 3`.

## 10. Reproducibility

Exact commands, from the repository root:

```bash
pip install -e ".[dev]"
python validation/validate_learned_combiner.py     # every number in sections 1-8
python examples/learned_combiner_vs_baselines.py   # the figure
python -m aperturediv combiner                     # the pooled table, smaller
```

Every seed is in the script as a module constant: dataset `44044`, model
`random_state=44044`, per-level evaluation `44044 + 500 + level`, crossover
bisection `44044 + 900 + step`, `sigma_e` sufficiency `44044 + 1300 +
100*rho_t`. No data file and no model binary is committed; `fit` is
deterministic given the dataset and `random_state`, which is asserted in
`tests/test_learned.py::TestLearnedCombiner::test_deterministic_fit`.

## 11. Compute used

Measured in the build container: Python 3.13.16, 2 CPU cores, 7.8 GiB RAM,
shared with four sibling build agents and a concurrent gate run.

| step | wall time |
|---|---|
| generate 120000 rows | 0.4 s |
| fit 4 weight regressors + 3 quantile regressors on 70000 rows | about 7 s on an idle pair of cores |
| full `validate_learned_combiner.py` | **184 s measured** with ten competing Python processes on the two cores; about 30 s when the cores are idle |
| full `learned_combiner_vs_baselines.py` | about 25 s idle |

The sizing was cut specifically to fit this budget. An earlier 250000-row
version of the validation script took **440 s** of wall clock under the same
contention, which is why the committed configuration is 120000 rows with 100
boosting iterations. The conclusions did not change across 250000, 200000 and
120000 rows: the crossover moved from 3.599 dB to 3.613 dB, and the learned
gain over the best analytic rule stayed at 0.072 dB.

No GPU is used or needed. PyTorch is not installed and is not required.

## 12. Ethical and safety limits

- The model is a receiver-design study tool. It produces combining weights
  for a simulated channel; it does not control hardware and has no interface
  that could.
- The data are synthetic and contain no personal or operational information.
- **Not flight-qualified, not certified, not approved for operational
  aerospace use.** Do not use its outputs for a link-availability commitment:
  the channel model is an idealisation whose correlation scale is a free
  input, and the validity limits in the README apply to everything above.
- A single-digit-decibel result produced on an idealised channel should not
  be presented as a receiver design margin.

## Credits

This is under reserved rights obtained by OPTIMA Organisation.
