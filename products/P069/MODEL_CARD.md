# Model card -- scenariocov learned gap finder 0.1.0

**This model is not certified for operational flight use.**

**Headline result, stated before anything else: on the bundled fixtures this
model is not separated from the greedy max-min-distance baseline, it is behind
uniform random on the grid-occupancy metric, and a deterministic
coverage-greedy chooser beats it on every one of 60 seeds. The model is
shipped because the measurement is the contribution. It is not recommended
over the baselines.**

## Problem

Given a verification campaign that has already run and a pool of scenarios
that could be executed next, propose the next batch, preferring regions of the
scenario space where the executed campaign is thin. The deterministic core of
`scenariocov` answers a related but different question -- which *declared*
t-way tuples, equivalence classes and grid cells were never reached -- and that
core was built, tested and validated before this model was written. The model
exists to see thinness *inside* a declared bin, which the combinatorial
measures cannot.

## Baselines

Three, all implemented before the model and all measured on the same paired
seeds:

1. **Uniform random** -- sample the batch without replacement from the pool.
   The null strategy.
2. **Greedy max-min distance** -- farthest-point sampling in the normalised
   embedding: take the candidate whose distance to the nearest already
   selected or already executed point is largest. Gonzalez 1985, *Theoretical
   Computer Science* 38, 293-306.
3. **Coverage-greedy reference** -- take the candidate adding the most new
   t-way tuples. This optimises the benchmark metric directly, so it is
   reported as a ceiling rather than as a competitor. It needs the declared
   discretisation and the chosen t; the other three do not.

## Architecture

A Gaussian kernel density estimate over the normalised embedding of the
executed scenarios:

    f_hat(x) = (1 / (n h^d)) * sum_i K((x - x_i) / h),  K standard Gaussian

`sklearn.neighbors.KernelDensity(kernel="gaussian", bandwidth="scott")`.
Bandwidth by Scott's rule, `h = n^(-1/(d+4))` on standardised data; the fitted
value on the 240-scenario campaign is **0.724415** in normalised-embedding
units. No neural network, no PyTorch, nothing is trained by gradient descent,
and **no model is ever serialised to disk**.

Embedding: one column per continuous parameter, scaled linearly onto [0, 1]
over its declared interval; `n_levels` one-hot columns per categorical
parameter, scaled by `1/sqrt(2)` so that two different levels of one
categorical parameter are exactly 1 apart. The bundled space has 7 parameters
and a 13-column embedding.

Batch rule (shipped default): proposals are taken in ascending density order,
each accepted proposal is added to the model's data and the density refitted
before the next is chosen, and a candidate within 0.5 fitted bandwidths of an
accepted proposal is skipped. Two simpler rules were measured and are worse --
see the ablation below.

Sources: Silverman 1986 chapter 4; Scott 1992 chapter 6; Efron 1979 for the
bootstrap.

## Dataset and its limitations

`fixtures/campaign_executed.json`, 240 synthetic scenarios over a 7-parameter
close-range rendezvous scenario space, generated at seed 20261016 by
`fixtures/regenerate_fixtures.py`; `fixtures/candidate_pool.json`, 400
candidates from the same space with the planted regions included. Full detail
in `DATASET_CARD.md`.

Limitations that matter for this model:

- The campaign is **synthetic**. No real verification campaign was used, and
  no claim is made about behaviour on one.
- It has **five deliberate holes**, so the gap-finding task is easier here
  than on a campaign whose holes are accidental and scattered.
- 240 scenarios in a 13-column embedding is a small sample for a density
  estimate. The benchmark starts from 20, which is smaller still: the fitted
  bandwidth at n = 20 is 0.838434, close to the span of the embedding itself.

## Training procedure

There is no training loop. `GapFinder.fit` stores the executed scenarios and
builds the kernel density estimate, plus `n_bootstrap` further estimates on
nonparametric bootstrap resamples. Fitting on 240 scenarios with 16 bootstrap
replicates takes well under a second on 2 CPU cores.

Hyperparameters, all declared rather than fitted:

| name | value | basis |
|---|---|---|
| kernel | gaussian | scikit-learn default for continuous KDE |
| bandwidth | `"scott"` | Scott 1992; no cross-validation is performed |
| `n_bootstrap` | 16 in the examples and the CLI, 0 in the benchmark | the benchmark does not use the uncertainty output and the bootstrap would multiply its cost |
| `sparsity_quantile` | 0.10 | declared definition of "a sparse region" |
| `min_separation` | 0.5 bandwidths | two proposals closer than half a bandwidth are the same query to this model |

## Test-split strategy

There is no fitted parameter to overfit, so a conventional train/test split
would measure nothing. The split that matters is the **paired seeded
benchmark**: at each seed, a starting campaign of 20 scenarios is drawn from
the committed campaign and a candidate pool of 200 from the committed pool,
both shared by every strategy at that seed, and each strategy adds 20
scenarios. The model never sees the pool before proposing from it, and the
coverage it is scored on is computed over the whole declared space, not over
the sample it was fitted on. 60 seeds.

## Metrics

Primary: 2-way combinatorial coverage reached after the added scenarios, and
the coverage gain per scenario, `(C_t(end) - C_t(start)) / n_steps`.
Secondary: grid-cell occupancy over the continuous parameters.

From `validation/validate_benchmark_output.txt`, 60 paired seeds:

| strategy | 2-way final mean | sd | gain per scenario | occupancy final mean | seconds |
|---|---|---|---|---|---|
| uniform random | 0.943215 | 0.018598 | 0.006549 | 0.219329 | 0.00 |
| greedy max-min | 0.951254 | 0.018781 | 0.006951 | 0.217477 | 0.03 |
| **learned gap finder** | 0.955457 | 0.015815 | 0.007161 | 0.218056 | 1.33 |
| coverage-greedy reference | 0.993584 | 0.006010 | 0.009067 | 0.238310 | 0.67 |

Paired per-seed differences, win / tie / loss over 60 seeds:

| comparison | metric | mean | sd | W/T/L | separated |
|---|---|---|---|---|---|
| learned - greedy max-min | 2-way | +0.004204 | 0.015119 | 30/10/20 | **no** |
| learned - uniform random | 2-way | +0.012242 | 0.019199 | 41/6/13 | no |
| learned - greedy max-min | occupancy | +0.000579 | 0.013669 | 23/10/27 | no |
| **learned - uniform random** | **occupancy** | **-0.001273** | 0.017177 | 25/5/30 | no |
| coverage-greedy reference - learned | 2-way | +0.038127 | 0.014868 | 60/0/0 | **yes** |

No p-value is quoted. The per-seed differences share a starting campaign and a
pool, which no two-sample test models, so the win/tie/loss split and the
spread are reported instead.

### Ablation of the batch rule, same seeds, 2-way metric

| batch rule | final mean | sd | gain per scenario | shipped |
|---|---|---|---|---|
| sequential refit plus 0.5-bandwidth separation | 0.955457 | 0.015815 | 0.007161 | yes |
| separation only, no refit | 0.934956 | 0.019006 | 0.006136 | no |
| lowest-density block, no refit, no separation | 0.933628 | 0.020104 | 0.006069 | no |
| greedy max-min, for reference | 0.951254 | 0.018781 | 0.006951 | - |

Both unshipped rules are worse than the greedy max-min baseline. The shipped
default is the only configuration of this model that is not, and it is still
not separated from that baseline.

## Uncertainty and confidence output

Every proposal carries three numbers, not one:

| field | meaning | units |
|---|---|---|
| `log_density` | mean log density over the bootstrap ensemble, or the single-fit value when `n_bootstrap == 0` | nats |
| `log_density_sd` | standard deviation of the log density across bootstrap replicates | nats |
| `sparsity_agreement` | fraction of replicates placing the candidate inside the sparsest `sparsity_quantile` of the pool | dimensionless, [0, 1] |

Measured behaviour, from `validation/validate_uncertainty_output.txt`:

- the bootstrap spread falls monotonically with more data, 0.144124 nats at
  n = 20 to 0.063551 at n = 240;
- `sparsity_agreement` separates candidates whose grid cell is empty from
  those whose cell is occupied, mean 0.181100 against 0.049764;
- the number is not saturated: 4.50 % of the pool at 1.0, 80.50 % at 0.0,
  15.00 % strictly between.

**It is not a calibrated probability.** It quantifies sensitivity to the
executed sample and nothing else -- not the binning, not the embedding, not the
kernel choice. Nothing in this repository compares it against an observed
frequency.

## Failure cases

1. **It does not beat the deterministic baselines.** Documented above in the
   model's own terms; this is the main failure case and it is not going to be
   tuned away.
2. **It is blind to what the combinatorial core sees directly.** The model
   works in a continuous embedding and has no notion of a declared t-way
   tuple, so a tuple that is missing for a categorical reason is found only if
   that region also happens to be geometrically sparse.
3. **It crowds without the batch rule.** With neither refit nor separation,
   all proposals come from one corner and the final coverage drops to 0.933628,
   below uniform random's 0.943215.
4. **The embedding is mixed-type and the bandwidth is global.** Three of the
   seven parameters are categorical one-hot blocks; one Gaussian bandwidth is
   applied across all 13 columns. No consistency claim is made.
5. **Small executed samples give a bandwidth close to the span of the space.**
   At n = 20 the fitted bandwidth is 0.838434, so the density is close to flat
   and the ranking is close to arbitrary. That is the regime the benchmark
   runs in, deliberately, because it is the regime where a gap finder would be
   useful.
6. **Cost.** 1.33 s against greedy max-min's 0.03 s for the same work, a factor
   of about 40, which is before the bootstrap is switched on. Wall clock varies
   with the machine; the coverage numbers do not.
7. **It cannot know that a region is forbidden.** A scenario space with
   physically impossible combinations has no way to declare them here, so the
   model will happily propose into a region that cannot occur.

## Reproducibility

Exact commands, run from the product root:

```bash
python fixtures/regenerate_fixtures.py                 # seed 20261016
python validation/validate_benchmark.py    > validation/validate_benchmark_output.txt
python validation/validate_uncertainty.py  > validation/validate_uncertainty_output.txt
python examples/strategy_comparison.py
python -m pytest tests/ -q --junitxml=junit.xml
```

Seeds: fixtures 20261016; benchmark seeds 0 to 59; the examples'
`strategy_comparison.py` uses seeds 0 to 29 and reports its own numbers, which
differ from the 60-seed figures above because it is a different number of
seeds and says so. `GapFinder(random_state=...)` seeds the bootstrap only; the
density fit is deterministic given its input.

## Compute used

2 CPU cores, no GPU, Python 3.13.16. Fitting with 16 bootstrap replicates on
240 scenarios: under a second. Full benchmark (60 seeds, four strategies, two
metrics, plus a three-way ablation): 7.4 s. Full test suite: 7.5 s. All eleven
validation scripts: 49 s. Nothing in this repository approaches the mission's
3-minute budget.

## Dependency versions this model's numbers were measured at

Both ends of every declared range were installed and the full suite plus the
numeric comparison run at each
(`validation/validate_dependency_ends_output.txt`):

| end | numpy | scikit-learn | python |
|---|---|---|---|
| floor | 2.3.4 | 1.8.0 | 3.13.16 |
| ceiling | 2.5.3 | 1.9.1 | 3.13.16 |

16 of 16 tracked quantities moved by less than 1e-9, including the fitted
bandwidth and the top proposal's log density and bootstrap sd. No model is
serialised, so no dependency bound here is a serialisation-format bound; the
scikit-learn ceiling is a numeric-behaviour bound, because `KernelDensity`
bandwidth selection and `score_samples` are not contractually stable across
minor releases.

## Ethical and safety limits

This model proposes software test scenarios. It does not fly anything, control
anything or make an operational decision. Its outputs are suggestions for what
to simulate next, and acting on them affects a test schedule and nothing else.

A coverage figure produced with this package's help is a statement about a
declared discretisation of a declared scenario space. Complete coverage of
that space is not evidence that a system is safe, and neither this model nor
the deterministic core should ever be cited as such. The space is only as good
as the person who declared it: a parameter nobody listed is a region nobody
measured, and this package cannot tell you that it is missing.

**This model is not certified for operational flight use.** The software is
research-grade: not flight-qualified, not certified, not approved for
operational aerospace use.
