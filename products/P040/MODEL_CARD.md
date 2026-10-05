# Model card: bitflipsim learned criticality predictor

**This model is not certified for operational flight use.** It is a
research-grade tool for ranking which bits of a stored model matter under
single-event upsets. It models upsets; it does not qualify parts, and nothing
in it substitutes for radiation testing of real hardware.

Version 0.1.0 · Status: TESTING · Validation level 2 · MIT ·
© 2026 OPTIMA Organisation

## 1. Problem

Given a trained inference model stored in memory, decide which bits to protect
with a byte budget. The ground truth - flip every bit, run inference, measure
the damage - costs one forward pass per bit site (4704 for the 147-parameter
reference model) and cannot be run on orbit or on a large model. The predictor
estimates the same quantity from features that need no injection at all.

**Target.** `log1p(D)` where `D` is the mean total-variation distance between
the golden and the faulty output distribution over an evaluation batch,
bounded in `[0, 1]`:

```
D = (1/n) sum_i 0.5 * sum_c |p_ic - p'_ic|
```

Conventions for faulty outputs that are not distributions, stated rather than
inherited: a NaN row gives `D = 1`; `+inf` logits give probability spread
uniformly over the infinite entries; an argmax over a NaN row reports class
`-1`, always counted wrong.

**Metric.** Degradation avoided per protected byte, under two cost models (see
§6). This is the metric the product specification names, and it is the metric
the baselines are compared on.

## 2. Baselines, implemented first

Both were written, run and recorded before the learned model existed, and both
are given to the learned model as input features, so it can only win by adding
information they do not carry.

| Baseline | Score for bit site `(parameter i, bit b)` | What it cannot do |
|---|---|---|
| **Magnitude** | `\|w_i\|`, identical for every bit of a parameter | no bit-position resolution at all |
| **Exponent-bit heuristic** | base-2 log of the worst-case relative perturbation of bit `b`, derived from the IEEE 754 layout alone: exponent bit at position `b` -> `2**(b - nmant)`; sign bit -> 1; mantissa bit -> `b - nmant` | no parameter resolution at all |

Two further reference points are reported with every result: an **oracle** that
ranks by the measured degradation itself (unachievable - it is the answer), and
a **random** ranking.

## 3. Architecture

`sklearn.ensemble.RandomForestRegressor`, 160 trees, `max_depth=10`,
`min_samples_leaf=4`, `random_state=7`, `n_jobs=1`, fitted on `log1p(D)`.

A forest rather than a linear model because the target spans ten orders of
magnitude and the interaction between bit position and parameter magnitude is
multiplicative. A forest rather than a boosted ensemble because a forest gives
a distribution-free uncertainty output for free (§7). This is not a deep model
and does not need to be: the whole problem has 14 features and 4704 sites.

## 4. Features (14, all injection-free)

| Feature | Meaning |
|---|---|
| `log2_abs_weight`, `abs_weight` | the magnitude baseline |
| `weight_sign` | sign of the parameter |
| `biased_exponent` | the stored exponent field `E` of the parameter |
| `bit_position` | storage-bit index |
| `is_sign_bit`, `is_exponent_bit`, `is_mantissa_bit` | role indicators from the discovered layout |
| `heuristic_log2_relative` | the exponent-bit baseline score |
| `layer_index`, `is_bias` | structural position in the network |
| `downstream_weight_l1` | L1 norm of the second-layer weights fed by the hidden unit this parameter affects |
| `upstream_activation_mean_abs` | mean absolute value of the activation this parameter multiplies, on the calibration batch |
| `hidden_active_fraction` | fraction of calibration samples for which the relevant hidden unit is past the ReLU |

Every one is computable from the golden parameters and a calibration batch. No
feature requires an injection, which is the point.

Measured importances (top five, from
`validation/criticality_methods_output.txt`): `heuristic_log2_relative`
0.664403, `biased_exponent` 0.118038, `log2_abs_weight` 0.081391, `abs_weight`
0.077793, `bit_position` 0.031904. Two thirds of the model is the exponent-bit
baseline. Its own contribution comes mainly from `biased_exponent`, which lets
it separate the parameters for which an exponent-MSB flip saturates to
infinity, lands on a subnormal, or scales cleanly - a distinction the pure bit
heuristic cannot make because it never looks at the value.

## 5. Data and splits

Dataset: the deterministic synthetic three-class, eight-feature problem
described in [`DATASET_CARD.md`](DATASET_CARD.md). It is synthetic and carries
no physical meaning.

| Split | Size | Used for |
|---|---|---|
| train | 900 | fitting the reference MLP whose bits are then flipped |
| calibration | 300 | measuring the ground-truth degradation used as the predictor's targets, and the activation statistics used as features |
| evaluation | 300 | measuring the degradation every method is finally scored on |

**Split strategy: grouped by parameter, and disjoint in the data as well.**
Parameter indices are shuffled with seed 7 and split 60/40 (88 train / 59 test
parameters); every bit of a parameter goes to the same side, so no bit of a
test parameter is ever seen in training. Targets come from the calibration data
split, the reported metric from the disjoint evaluation data split. Neither the
parameters nor the data overlap between fitting and scoring.

## 6. Training procedure and results

```bash
PYTHONPATH=src python validation/validate_criticality_methods.py
```

Deterministic: seeds 20261005 (dataset and reference model) and 7 (split and
forest). Fit time about 0.9 s on one contended core; the ground-truth sweeps
that produce the targets cost 9408 forward passes and about 6 s.

### Degradation avoided per protected byte, test parameters, evaluation split

**Bit-granular cost model** (`cost = protected bits / 8` bytes; idealised, no
real scheme is this fine-grained)

| budget | magnitude | exponent heuristic | **learned** | oracle | random |
|---|---|---|---|---|---|
| 2 % | 0.498414 | 4.886449 | **6.298675** | 6.532930 | 0.312257 |
| 5 % | 0.364869 | 3.164676 | **3.550208** | 3.616972 | 0.258950 |
| 10 % | 0.362017 | 1.713467 | **1.983816** | 2.031046 | 0.155522 |
| 25 % | 0.369763 | 0.834949 | **0.871761** | 0.872797 | 0.159885 |
| 50 % | 0.331287 | 0.440233 | **0.440244** | 0.440246 | 0.223585 |

**Word-granular cost model** (`cost = protected parameters x 4` bytes; what
word-level ECC or TMR actually charges)

| budget | magnitude | exponent heuristic | learned | oracle | random |
|---|---|---|---|---|---|
| 2 % | **0.547011** | 0.220135 | 0.362759 | 0.547011 | 0.095307 |
| 5 % | 0.367204 | 0.220135 | **0.448575** | 0.540700 | 0.109166 |
| 10 % | 0.350601 | 0.220135 | **0.388010** | 0.461151 | 0.146595 |
| 25 % | **0.372746** | 0.220135 | 0.360320 | 0.398163 | 0.200680 |
| 50 % | **0.333600** | 0.220135 | 0.331964 | 0.335665 | 0.201788 |

### The result, without spin

1. **At bit granularity the learned predictor beats the exponent-bit heuristic**
   by **+0.385533 avoided per byte (+12.2 %)** at a 5 % budget, reaching 98.2 %
   of the oracle against the heuristic's 87.5 %.
2. **The heuristic costs nothing.** Zero injections, zero training, and it
   already captures 87.5 % of what perfect knowledge would achieve. The learned
   predictor's 12.2 % margin is bought with 2816 injections and forward passes
   on the training parameters plus the forest fit. If you cannot afford a
   ground-truth sweep on your own model - which is the situation this product
   exists for - **use the heuristic**.
3. **At word granularity the heuristic carries no information at all.** It
   gives every word the same score, so its avoided-per-byte figure is pinned at
   0.220135 - exactly the random-selection expectation - at every budget. There
   the plain magnitude baseline beats the learned predictor at 4 of the 6
   budgets tested.
4. **No single method wins everywhere.** Which criticality method is best
   depends on the protection granularity the hardware offers. That is the
   result; it has not been retuned, and no baseline has been weakened to make
   the learned model look better.

## 7. Uncertainty output

The predictor returns, for every bit site, the mean and the **standard
deviation of the individual trees' predictions**:

```
mean(x)  = (1/T) sum_t h_t(x)
sigma(x) = sqrt( (1/T) sum_t (h_t(x) - mean(x))**2 )
```

exposed as `CriticalityPrediction.log_mean`, `.log_sigma`, `.expected`
(= `expm1(log_mean)`) and `.relative_sigma`.

**Measured coverage**, from `uncertainty_calibration` on held-out sites:
**0.878178** of held-out targets lie within two ensemble standard deviations of
the ensemble mean, against the Gaussian reference of **0.954500**. Mean
absolute error in `log1p` space is 4.624248e-03; median sigma is 8.45e-06.

So the uncertainty **understates** the predictive spread, which is the known
behaviour of forest tree dispersion: it measures model disagreement, not
predictive variance. It is a usable confidence signal for ranking and for
flagging sites the model is unsure about. **It is not a calibrated predictive
interval and must not be used as one.** No recalibration layer was added to
make the number look better; the number is published as measured.

## 8. Failure cases

* **Word-granular protection.** The predictor's word score is the sum of its
  bit scores, and at 4 of 6 budgets that ranks worse than simply sorting by
  `|w|`. If your hardware protects whole words, start with the magnitude
  baseline.
* **Very small and very large budgets.** At a 2 % word budget only one word is
  protected and the predictor picks the wrong one (0.362759 against the
  magnitude baseline's 0.547011, which happens to equal the oracle). At a
  100 % budget every method is identical by construction.
* **Sites the ground truth cannot distinguish.** Mantissa bits below about
  bit 10 have degradations of order `1e-9`; the forest's ordering among them is
  noise, and its uncertainty output does not flag that, because the trees agree
  that the value is tiny.
* **Out-of-distribution parameters.** The model has seen 88 parameters of one
  147-parameter network. A parameter whose exponent field, magnitude or fan-out
  lies outside the training range gets an extrapolated prediction from a tree
  ensemble, which means a constant - the nearest leaf - with a small sigma.
  That failure mode is silent.
* **Transfer to other models.** Nothing here establishes that the ordering or
  the margin transfers to a different architecture, a different dataset or a
  model of realistic size. This is the most important limitation in this card.

## 9. Reproducibility

```bash
git clone https://github.com/OmAcharya-avtr/bitflipsim.git
cd bitflipsim
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python validation/validate_criticality_methods.py          # every number in §6 and §7
MPLBACKEND=Agg python examples/criticality_methods.py      # the plot and the budget sweep
python -m bitflipsim criticality --budget-fraction 0.05    # the headline table from the CLI
```

Seeds: dataset and reference model `20261005`; parameter split and forest `7`.
Library versions that produced the committed numbers: Python 3.13.16, numpy
2.5.3, scipy 1.18.1, scikit-learn 1.9.1, onnxruntime 1.29.0. The committed raw
output is in `validation/criticality_methods_output.txt`.

**Compute used.** Whole pipeline: about 8 s on one shared CPU core - 9408
forward passes for the two ground-truth sweeps, 4704 feature rows, one
160-tree forest fit. Peak memory well under 200 MB. There is no GPU step and no
PyTorch dependency; PyTorch is not available in this environment, which is why
the product's claim is confined to the scikit-learn / ONNX path.

## 10. Ethical and safety limits

* **Not certified for operational flight use.** Research-grade. Nothing in this
  repository has been through a qualification process of any kind.
* **This product models upsets; it does not qualify parts.** The cross-section
  and flux are inputs the user must supply from their own part test data and
  environment model. The constants shipped with the package are round
  illustrative numbers and are labelled as such in the code, in the docs and in
  the output of every script that uses them.
* **A criticality ranking is not a protection scheme.** Protecting the top-k
  bits by any of these rankings leaves everything else unprotected, and the
  degradation avoided is an expectation over uniformly placed upsets, not a
  guarantee about any particular upset.
* **The derived clamping bound is the only guarantee in this repository**, it
  is for single upsets on the two-layer network of `bitflipsim.network`, and it
  is verified exhaustively for that case and no other.
* The model is trained on synthetic data with no human subjects, no personal
  data and no provenance concerns. Its only misuse risk is being cited as
  evidence that a particular flight model is or is not upset-tolerant, which it
  cannot support.
