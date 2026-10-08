# Model card — falsifyloop surrogate

**This model is not certified for operational flight use.**

Research-grade. Not flight-qualified, not certified, not approved for
operational aerospace use. Every number below came from a script in
`validation/` executed on 2026-10-08 in the build container; the raw output is
committed beside each script.

---

## 1. Problem

Predict the **requirement robustness** `rho(x)` of a simulated closed-loop
trace from the six decision variables `x` that produced it, cheaply enough to be
refit inside a falsification search, so that the search can rank candidate
points by how likely they are to violate.

- **Task**: scalar regression, 6 continuous inputs, 1 continuous output.
- **Target unit**: dimensionless. Every shipped instance normalises its
  predicates by the requirement's own tolerance.
- **Use inside the product**: the model's output is never reported as a result.
  It is used only to pick which point to simulate next. Every verdict in this
  package comes from an actual simulation, never from the model.

## 2. Baseline, implemented first

**Uniform random sampling of the declared box** is the baseline for the search,
and **predicting the training mean** is the baseline for the regression. Both
were implemented and measured before the model.

The search baseline is a strong one. On a box whose violating set has volume
fraction `p`, uniform random finds a violation within `n` draws with probability
exactly `1 − (1−p)^n`, with no tuning, no state and no assumption that the
response is smooth. The full comparison, including the instances where it wins,
is in `README.md` and `validation/VALIDATION.md`.

Against the regression baseline, on an independently seeded held-out split
(`validate_surrogate.py` check 1):

| instance | RMSE forest | RMSE training mean | ratio | R² |
|---|---:|---:|---:|---:|
| overshoot-loose | 0.522819 | 0.908043 | 1.737 | 0.6670 |
| settling-band | 0.398536 | 0.623474 | 1.564 | 0.5884 |
| multi-requirement | 0.277745 | 0.422974 | 1.523 | 0.5631 |
| command-rate | 0.140700 | 0.307459 | 2.185 | 0.7876 |
| overshoot-tight | 0.117796 | 0.201787 | 1.713 | 0.6577 |
| attitude-envelope | 0.068163 | 0.165290 | 2.425 | 0.8270 |
| nested-capture | 0.114479 | 0.236708 | 2.068 | 0.7644 |
| rate-envelope | 0.072211 | 0.137141 | 1.899 | 0.7184 |

The forest beats the trivial predictor by a factor of 1.5 to 2.4 in RMSE. That
is a modest model, and it is enough, because the search needs an ordering rather
than an accurate value: the Spearman correlation between prediction and truth is
0.8795 to 0.9606 across the suite.

## 3. Architecture

`sklearn.ensemble.RandomForestRegressor`:

| setting | value | why |
|---|---|---|
| `n_estimators` | 25 | refit every 8 simulations inside a 100-simulation budget; a larger forest costs more than it buys |
| `min_samples_leaf` | 1 | the target is noiseless, so there is nothing to smooth against |
| `bootstrap` | True | the tree-to-tree spread is the uncertainty output and needs resampling to exist |
| `n_jobs` | **1, hard-coded, not exposed** | `n_jobs = 2` is **about 8 times slower** at the single-row inference this search does on a 2-core container; three runs in this build gave 8.9x, 7.9x and 8.6x (`validate_surrogate.py` check 5) |
| `random_state` | the search seed | makes a seeded search bit-reproducible |

**Why a forest and not a Gaussian process.** Two reasons, both measurable. The
response is non-smooth: two clips in the simulator and the `min`/`max` of the
robustness semantics put kinks and ridges all over `rho`, which a stationary GP
kernel is a poor prior for. And cost: an exact GP is `O(n³)` in the
observations, refit inside a loop, on a simulator that costs about 0.33 ms per
call. The measured forest refit cost is already 19-27 ms on 120 points, 60-90
simulations' worth.

**Acquisition.** `mean − 2·spread`, minimised over 256 uniform candidate points
per simulation. This is the lower-confidence-bound rule of Srinivas et al.
(2010) with a forest dispersion substituted for a GP posterior standard
deviation — a substitution that **voids the regret bound that paper proves**,
and is used here because it works, measurably, not because it is justified.

## 4. Dataset: source and limitations

Full detail in [`DATASET_CARD.md`](DATASET_CARD.md). In brief:

- **Source**: generated on demand by `falsifyloop.systems.simulate`, a
  deterministic synthetic benchmark closed loop. Nothing is downloaded, nothing
  is committed, and no real-world data is used.
- **Inputs**: uniform draws from the declared six-dimensional box.
- **Targets**: the requirement robustness of the resulting trace.
- **Limitations, in order of importance**:
  1. **The simulator is synthetic and no parameter in it was identified from any
     aircraft.** A model fit to it has learned a benchmark, not a vehicle.
  2. **No noise.** The simulator is deterministic, so the dataset has no
     aleatoric component at all. Every bit of the model's error is its own bias,
     which is why the ensemble-dispersion uncertainty behaves as section 6
     reports.
  3. **Violating points are rare.** On the hardest instance 0.1267 % of uniform
     draws violate, so a 120-point training set typically contains **none**. The
     model learns the shape of the satisfying region and extrapolates towards
     the violating corner; the search works because that extrapolation is
     directionally right, not because the model has seen a violation.
  4. **In-search training data is not uniform.** After the warm start the
     surrogate trains on points it chose itself, which are concentrated where it
     already expects low robustness. The held-out measurements in this card use
     uniform training data and therefore describe the model in a more favourable
     setting than the one it runs in.
  5. **One box, one simulator, eight requirements.** Nothing here generalises
     off that.

## 5. Training procedure

Inside the search (`falsifyloop.search.surrogate_guided`):

1. **Warm start**: 16 simulations on a Latin hypercube design. During these the
   strategy is not learned at all, and on the four easiest instances the
   violation is usually found before the model is ever fit.
2. **Fit**: a fresh forest on every observation so far. No warm-starting of
   trees, no incremental update.
3. **Propose**: draw 256 uniform candidates, score `mean − 2·spread`, simulate
   the minimiser.
4. **Refit** every 8 simulations; repeat until a violation or the budget.
5. **Stop at the first violation**, under the same rule as every other strategy.

For the held-out measurements in this card: 120 uniform training points and 200
uniform test points per instance, drawn at **different seeds** (4100 and 9400).

No hyperparameter was tuned against the benchmark result. The forest settings
were chosen from the compute budget before the benchmark was run, and the
reported comparison is the first one executed with them.

## 6. Metrics, including the uncertainty output

### Search performance (the metric that matters)

Mean curve probability — the average probability of having found a violation
over a 100-simulation budget, 30 seeds, base seed 52000:

| strategy | aggregate | hardest instance (`rate-envelope`) |
|---|---:|---:|
| uniform-random (baseline) | 0.6130 | 0.0630 |
| surrogate-guided | **0.8016** | **0.5300** |

The surrogate beats the baseline resolvably (bootstrap interval on the
difference excluding zero) on the four hardest instances: `overshoot-tight`
+0.2370, `attitude-envelope` +0.4420, `nested-capture` +0.2973, `rate-envelope`
+0.4670. On the four easiest the comparison is **undecided**, and on
`settling-band` the point estimate is a **loss** of −0.0050.

### Regression accuracy

Section 2 above. RMSE ratio 1.523–2.425 over the trivial predictor.

### Uncertainty / confidence output

`ForestSurrogate.predict(x)` returns `(mean, spread)`, where `spread` is the
standard deviation of the 25 individual trees' predictions at `x`, in the unit
of the target.

Measured on the held-out split (`validate_surrogate.py` check 2):

| instance | coverage of ±1.96·spread | coverage of ±1·spread | mean spread / mean \|error\| |
|---|---:|---:|---:|
| overshoot-loose | 0.9450 | 0.7450 | 1.2567 |
| settling-band | 0.9400 | 0.7900 | 1.2124 |
| multi-requirement | 0.9250 | 0.8000 | 1.1743 |
| command-rate | 0.8950 | 0.7000 | 1.0546 |
| overshoot-tight | 0.9450 | 0.7450 | 1.2620 |
| attitude-envelope | 0.9300 | 0.7900 | 1.3191 |
| nested-capture | 0.9650 | 0.7650 | 1.3666 |
| rate-envelope | 0.9400 | 0.7700 | 1.2065 |
| **mean** | **0.9356** | — | — |
| **pooled over the suite** | 0.9356 | **0.7631** | — |

**Read both columns together.** Nominal-95 coverage is close to nominal.
Nominal-68 coverage is 0.7631 against the 0.6827 a Gaussian gives — well above.
Both at once means the held-out error is **not Gaussian** around the forest
mean: the spread is conservative through the body of the distribution and too
thin in the tails, and the near-nominal 95 % figure is two errors partly
cancelling, not evidence of calibration.

**This contradicts what the module docstring predicted before the measurement
was taken**, which said the spread would understate the error badly. It does
not. The prediction was wrong, the docstring was corrected to the measurement,
and the mistake is recorded in `validation/VALIDATION.md`.

What the spread **is** good for is ranking, which is all the acquisition uses.
Spearman correlation between the reported spread and the absolute held-out
error: 0.3607 to 0.6124 per instance, mean **0.4853**, pooled **0.6343**. Mean
absolute error in the top spread decile is 3.3 to 11.5 times that in the bottom
decile.

**The spread must never be reported as a confidence interval.** Wager, Hastie &
Efron (2014) give the jackknife corrections that would be needed; this package
does not implement them and says so instead.

## 7. Test-split strategy

- **For the held-out measurements in this card**: 120 training and 200 test
  points per instance, both drawn uniformly from the same declared box, at
  **different seeds** (4100 and 9400). No point appears in both. There is no
  temporal or grouping structure in the data, so a random split is the right
  one, and the two-seed construction makes the independence explicit rather than
  relying on a shuffle.
- **For the search comparison**: there is no split, because there is no held-out
  claim. The surrogate is refit online and judged only on how many simulations
  it took to find a counterexample, which is measured against the baseline on
  the same instances, the same budget and the same seeds.
- **Seeds**: repeat `r` of every benchmark cell uses `base_seed + r` for every
  strategy, so the comparison is not confounded by which seeds a strategy got.

## 8. Failure cases

1. **On easy instances the model never gets a turn.** On `overshoot-loose` all
   30 of the surrogate's violations came during the 16-simulation warm start,
   and on `settling-band` 28 of 30 did. Reported as a win for the strategy, it
   would be a win for Latin hypercube.
2. **Its only point-estimate loss is `settling-band`** (−0.0050), which the
   bootstrap cannot resolve from zero.
3. **The training set usually contains no violation at all** on the hard
   instances, so the model is extrapolating towards a region it has never seen.
   It works; it is not guaranteed to, and on a simulator whose robustness is
   non-monotone towards the violating corner it would not.
4. **The uncertainty is the wrong shape** (section 6) and would mislead anyone
   who treated it as a posterior.
5. **Cost.** The gain is paid for in CPU: roughly 15-25 simulations' worth of
   wall clock per simulation bought, on a simulator costing about 0.33 ms. On a cheaper
   simulator the surrogate is a net loss in wall clock even where it wins in
   sample count.
6. **`n_jobs > 1` is a trap.** About eight times slower at single-row inference on this
   container, which is why the parameter is not exposed.
7. **Fitting through an infinite robustness value is refused**, not clipped,
   because a surrogate fit through an infinity is meaningless. Infinite
   robustness comes from an empty time window; the shipped instances check their
   horizon at construction so it cannot arise there.

## 9. Reproducibility — exact commands and seeds

```bash
pip install -e ".[dev]"

# The held-out measurements in this card (train seed 4100, test seed 9400):
python validation/validate_surrogate.py

# The search comparison (base seed 52000, 30 seeds, budget 100):
python validation/validate_benchmark.py

# The figures:
MPLBACKEND=Agg python examples/surrogate_uncertainty.py
MPLBACKEND=Agg python examples/sample_efficiency_curves.py   # base seed 7100

# A single seeded run, reproducible bit for bit:
python -m falsifyloop falsify --instance rate-envelope \
    --strategy surrogate-guided --budget 100 --seed 5
```

Seeds in use: benchmark base 52000; difficulty 52052; surrogate split 4100 and
9400; `sample_efficiency_curves.py` 7100; `per_instance_comparison.py` 7300;
`robustness_landscape.py` and `counterexample_trace.py` 5; bootstraps 0 and
52000. A pinned regression test,
`tests/test_integration.py::test_benchmark_regression_pinned_seeded_numbers`,
fails if any layer changes the seeded result.

## 10. Compute used

| step | cost |
|---|---|
| one simulation + requirement evaluation | 0.31–0.37 ms |
| one forest refit on 120 points | 19–27 ms (≈60–90 simulations) |
| scoring 256 candidates (mean + spread) | 2.5–4.2 ms (≈8–14 simulations) |
| held-out measurements, whole suite | 2560 simulations, ≈8 s |
| the full 1200-search benchmark | 26–50 s |

Trained and measured on a container reporting **2 available CPU cores**
(`os.sched_getaffinity`), single process, `n_jobs = 1`. No GPU, no PyTorch, no
pretrained weights, no external data. **These are wall clocks on a shared
container; they move by 10–20 % between runs and are not hardware
characteristics.** The primary metric everywhere is a seeded simulation count,
which does not move.

## 11. Ethical and safety limits

- **This model is not certified for operational flight use.**
- It predicts a number about a synthetic benchmark. It has no bearing on any
  vehicle, any certification activity, or any operational decision.
- **It must never be used in place of a simulation.** Its output is a ranking
  signal inside a search; every verdict this package reports comes from an
  actual simulation.
- **Its uncertainty output must never be reported as a confidence interval**, on
  the measurement in section 6.
- **A search that found nothing found nothing.** Falsification is one-sided and
  the absence of a counterexample is not evidence of correctness, whether the
  search was guided by this model or by anything else.
- There is no personal data anywhere in this package and no human subject is
  involved at any point.

## 12. References

- Breiman, L. (2001), "Random forests", *Machine Learning* 45(1), 5–32.
- Efron, B. (1979), "Bootstrap methods: another look at the jackknife", *Annals
  of Statistics* 7(1), 1–26.
- Wager, S., Hastie, T. and Efron, B. (2014), "Confidence intervals for random
  forests: the jackknife and the infinitesimal jackknife", *JMLR* 15,
  1625–1651.
- Srinivas, N., Krause, A., Kakade, S. and Seeger, M. (2010), "Gaussian process
  optimization in the bandit setting", *ICML 2010*.
- Annpureddy, Y., Liu, C., Fainekos, G. and Sankaranarayanan, S. (2011),
  "S-TaLiRo: a tool for temporal logic falsification for hybrid systems",
  *TACAS 2011*, LNCS 6605.
- Fainekos, G. E. and Pappas, G. J. (2009), "Robustness of temporal logic
  specifications for continuous-time signals", *Theoretical Computer Science*
  410(42), 4262–4291.
