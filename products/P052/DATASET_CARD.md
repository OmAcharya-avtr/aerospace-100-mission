# Dataset card — falsifyloop

**No data file is committed to this repository, and none is downloaded.** Every
training and test point is generated on demand by a deterministic committed
script, so the dataset is reproducible from a seed rather than stored.

Research-grade. Not flight-qualified, not certified, not approved for
operational aerospace use.

## What the data is

| field | description |
|---|---|
| **Inputs** | Six decision variables: `step_amplitude` [deg], `kp_factor`, `kd_factor`, `tau_factor` (all dimensionless multipliers), `gust_amplitude` [deg/s²], `gust_frequency` [Hz]. |
| **Target** | The requirement robustness of the simulated trace, dimensionless on every shipped instance because each predicate is normalised by its own tolerance. Negative exactly when the requirement is violated. |
| **Generator** | `falsifyloop.systems.simulate` — a single-axis attitude-hold loop with a first-order actuator, a slew-rate limit, a deflection limit and a sinusoidal gust, integrated by explicit Euler at dt = 0.005 s over a 2.0 s horizon (401 samples). |
| **Sampling** | Uniform over the declared box `falsifyloop.instances.SEARCH_BOX`, from a `numpy.random.Generator` with a stated seed. |
| **Size** | Whatever the caller asks for. The held-out measurements use 120 train + 200 test per instance; the difficulty measurement uses 15000 per instance; the full benchmark consumes up to 120000. |
| **Licence** | Apache-2.0, same as the code that generates it. |

## Source and provenance

Entirely synthetic. There is **no external data source**, no download, no
scraped corpus, no measurement campaign, and no human subject.

The simulator's structure — second-order attitude dynamics with aerodynamic
damping, a PD attitude controller, a first-order actuator with rate and
deflection limits — is standard and cited in `src/falsifyloop/systems.py`
against Etkin & Reid (1996), Stevens, Lewis & Johnson (2015) and Franklin,
Powell & Emami-Naeini (2015). **The parameter values are not.** They are
declared constants of this benchmark, chosen so the nominal loop is stable and
lightly damped (`wn` = 5.019960 rad/s, `zeta` = 0.747018) and so the declared box
contains both satisfying and violating settings. **No value was identified from
any aircraft and none is traceable to flight data.**

## Limitations

1. **It is a benchmark, not a vehicle.** A model fit to this data has learned
   this simulator.
2. **It is noiseless.** The simulator is deterministic to the bit
   (`validate_simulator.py` check 4: 500 re-simulations, 0 differences), so the
   dataset has no aleatoric component. Any residual a model leaves is its own
   bias. This is the structural reason the surrogate's ensemble-dispersion
   uncertainty has the shape reported in `MODEL_CARD.md` section 6.
3. **The target is severely imbalanced for the hard instances.** Measured
   uniform-violation probabilities, 15000 draws per instance at seed 52052, with
   exact Clopper-Pearson 95 % intervals:

   | instance | tier | violations | p | 95 % interval |
   |---|---|---:|---:|---|
   | overshoot-loose | easy | 4829 | 0.321933 | [0.314458, 0.329478] |
   | settling-band | easy | 1592 | 0.106133 | [0.101248, 0.111172] |
   | multi-requirement | moderate | 986 | 0.065733 | [0.061819, 0.069817] |
   | command-rate | moderate | 470 | 0.031333 | [0.028603, 0.034247] |
   | overshoot-tight | hard | 145 | 0.009667 | [0.008163, 0.011365] |
   | nested-capture | hard | 68 | 0.004533 | [0.003522, 0.005744] |
   | attitude-envelope | hard | 56 | 0.003733 | [0.002821, 0.004845] |
   | rate-envelope | very hard | 19 | 0.001267 | [0.000763, 0.001977] |

   A 120-point uniform training set on `rate-envelope` contains a violating
   point with probability about 14 %.
4. **The physics stops being valid where the violations are.** The declared
   second-order model assumes small-angle single-axis motion;
   `validation/validate_simulator.py` check 3 observed a worst `|theta|` of
   **24.1727 deg** over 4000 uniform draws. Violating settings routinely leave
   the regime in which the model is even nominally valid. That
   is acceptable for a search benchmark and unacceptable for any claim about a
   vehicle.
5. **7.2 % of violating draws are discretisation artefacts.** 22 of 304
   violating points flip to satisfying when the integration step is refined by a
   factor of four, with a worst flipped robustness of 0.135888
   (`validate_simulator.py` check 5).
6. **In-search data is not uniform.** After its warm start the surrogate trains
   on points it chose, which are not a uniform sample of the box. The held-out
   metrics in `MODEL_CARD.md` use uniform training data and therefore describe a
   more favourable setting than the one the model runs in.
7. **One box, one simulator, eight requirements.** Nothing generalises off that.

## Regenerating it

Deterministic from the seeds below. Nothing is cached and nothing is committed.

```bash
# The difficulty measurement (15000 uniform draws per instance, seed 52052):
python validation/validate_difficulty.py

# The surrogate's train/test split (seeds 4100 and 9400):
python validation/validate_surrogate.py
```

Or directly:

```python
import numpy as np
from falsifyloop import instance

inst = instance("rate-envelope")
rng = np.random.default_rng(52052)
x = inst.sample(rng, 1000)                       # (1000, 6) uniform in the box
y = np.array([inst.evaluate(p) for p in x])      # robustness, dimensionless
```

`Instance.evaluate` is one simulation, and it is the unit every
sample-efficiency curve in this repository counts.
