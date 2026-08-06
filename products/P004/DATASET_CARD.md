# Dataset Card — PassPlanner synthetic availability dataset

**This dataset is entirely SYNTHETIC. It contains no real meteorological
observations, no real optical-link measurements and no real ground-station
statistics.** It exists to exercise and calibrate the availability-prediction
component; it is not evidence about any real site, and results obtained on it
do not transfer to operational weather prediction.

## Identity

| Field | Value |
|---|---|
| Name | PassPlanner synthetic weather/availability dataset |
| Version | 0.1.0 (ships with `passplanner` 0.1.0) |
| Generator | `src/passplanner/synthdata.py`, function `generate_dataset(n_samples, seed)` |
| Committed data files | none — data is regenerated deterministically from the seed (nothing is stored on disk, so the >1 MB commit rule does not apply) |
| Licence | Apache-2.0, same as the package |

## Regeneration

```python
from passplanner import generate_dataset
train = generate_dataset(8000, seed=20260301)   # validation training split
test  = generate_dataset(4000, seed=20260302)   # validation test split
```

`numpy.random.default_rng(seed)` is used throughout, so identical seeds give
bit-identical arrays (asserted in `tests/test_mlmodel.py`). Generating 12 000
samples takes well under a second.

## Instances and features

One instance = one hypothetical satellite pass at a hypothetical station, with
a binary outcome "was the optical path usable?".

| # | Feature | Units | Meaning |
|---|---|---|---|
| 0 | `prior_p_clear` | – (0–1) | climatological monthly clear-sky prior for the site |
| 1 | `rel_humidity_pct` | % | synthetic relative humidity, clipped to [5, 100] |
| 2 | `ir_cloud_fraction` | – (0–1) | synthetic IR-derived cloud fraction proxy |
| 3 | `pressure_anom_hpa` | hPa | synthetic surface-pressure anomaly |
| 4 | `wind_speed_ms` | m/s | synthetic 10 m wind speed, clipped at 0 |
| 5 | `month_sin` | – | sin(2π·month/12) |
| 6 | `month_cos` | – | cos(2π·month/12) |

Label `y ∈ {0, 1}`: 1 = pass succeeded (clear line of sight).
The generator also returns `p_true`, the exact probability used to draw `y`;
it is used only to compute an oracle reference in validation and is never
given to any model as a feature.

## Generative process (the "ground truth" is a construct, not physics)

1. Draw a site type (3 archetypes: temperate, arid, highland) and a calendar
   month uniformly; look up the archetype's monthly prior `p0`.
2. Draw a latent synoptic state `s ~ N(0, 1)`.
3. `logit(p_true) = logit(p0) − 1.6·s` — positive `s` means a cloudier
   synoptic situation.
4. Draw `y ~ Bernoulli(p_true)`.
5. Emit noisy observables correlated with `s` and `p0` (humidity, IR cloud
   fraction, pressure anomaly, wind), plus the month encoding.

The archetype priors used *inside the generator* (0.34–0.89 depending on month
and archetype) are invented, plausible-looking placeholders on the same
footing as the example station files.

## Splits

Train and test are **independent draws from the same generative process**
using different seeds, not a partition of one sample; because draws are i.i.d.
this is equivalent to a random hold-out split and guarantees no leakage. The
validation run uses train 8000 / test 4000.

## Known limitations and biases

* **No real data.** Absolute metrics measured on it (Brier 0.169, AUC 0.822)
  describe the synthetic generator only.
* Feature–label relationships are smooth, low-dimensional and stationary. Real
  cloud fields are spatially and temporally correlated, seasonal-regime
  dependent, and have heavy-tailed persistence — none of that is present.
* Pass success is drawn once per pass (all-or-nothing). Real passes can be
  partially obscured.
* Site archetypes are balanced by construction; a real network is not.
* No terrain, aerosol, turbulence (Cn²), daylight/background-radiance or
  satellite-geometry effects.
* Class balance follows the priors (about 60 % successes overall); no
  rebalancing is applied.

## Ethical / safety notes

No personal data, no human subjects, no protected attributes. The only misuse
risk is treating model outputs as real availability forecasts: **the model
trained on this data is not certified for operational flight use** and must
not be used to make real contact-plan commitments.
