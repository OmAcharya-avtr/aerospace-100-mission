# Dataset card — conformalband synthetic energy-per-leg regression

**Package:** `conformalband` 0.1.0 · **Generator:** `conformalband.data` and
`conformalband.physics` · **Status:** `TESTING` · **Validation level:** 2

## Summary

Fully synthetic. Nothing is downloaded, no data file is committed, and no file
in this repository contains a dataset. Every array is produced at run time by
committed code from a `numpy.random.Generator` seeded in the caller, so every
number in `validation/` is reproduced exactly by re-running the script that
printed it.

This is **not** a measurement of any vehicle. The airframe coefficients are
chosen to make the induced-power term the larger one at the design airspeed,
so that all-up mass matters; they are not taken from a datasheet and are not
validated against flight test data.

## Generation

```python
from conformalband.data import make_audit_split, make_dataset
from conformalband.shift import CovariateShift

split = make_audit_split(seed=57001, n_fit=1500, n_calibration=500,
                         n_test=1000, severity=2.0)
shifted = make_dataset(1000, seed=11, shift=CovariateShift(severity=3.0))
```

### Covariates

Five columns, `conformalband.data.FEATURE_NAMES`, all independent Gaussians.

| Column | Symbol | Unit | Calibration distribution | Shifted distribution (severity `s`) |
|---|---|---|---|---|
| `airspeed_mps` | `V` | m/s | `N(20.0, 1.50^2)` | unchanged |
| `mass_kg` | `m` | kg | `N(6.00, 0.60^2)` | `N(6.00 + 0.18 s, 0.60^2)` |
| `air_density_kgm3` | `rho` | kg/m^3 | `N(1.18, 0.035^2)` | unchanged |
| `headwind_mps` | `w` | m/s | `N(0.00, 2.00^2)` | `N(0.00 + 0.60 s, 2.00^2)` |
| `distance_m` | `d` | m | `N(1000.0, 80.0^2)` | unchanged |

The two shifted coordinates move by 0.3 standard deviations per unit severity,
so the Mahalanobis displacement is `0.3 sqrt(2) s = 0.4243 s`. Shift severities
0, 1, 2 and 3 therefore have displacements 0, 0.4243, 0.8485 and 1.2728.
Because both the calibration and test laws are declared Gaussians with the same
standard deviation, the likelihood ratio is available in closed form and is
**exact**, not estimated. That is the only reason this package can measure what
a wrong weight costs: it has a right one to compare against.

Nothing is truncated or rejected, so the declared laws are the laws. At the
most severe shift the ground speed `V - w` is distributed as
`N(16.4, 2.5^2)`, so `P(V - w <= 0)` is about `3e-11` and the generator's
positive-ground-speed guard has never fired in any run recorded here.

### Target

`energy` [Wh], the battery energy for one straight leg:

```
truth  = leg_energy(V, m, rho, w, d)                  # conformalband.physics
energy = truth * (1 + 0.06 * z),      z ~ N(0, 1)
```

The noise is **multiplicative**, so the residual scale grows with the energy.
That heteroscedasticity is deliberate: it is why the marginal conformal band is
the wrong width in the tails (measured: 0.9597 coverage in the lowest tercile
of predicted energy against 0.8192 in the highest, with no shift at all,
`validation/validate_conditional_coverage_output.txt`) and why the Mondrian
variant exists.

Measured on 200 000 draws from the calibration distribution
(`validation/validate_known_answers.py` section 7,
`make_dataset(200000, seed=1)`): mean noiseless energy **3.117808 Wh**,
standard deviation across covariate draws **0.580589 Wh**, observation-noise
standard deviation **0.190197 Wh**, against the multiplicative-noise identity
`0.06 sqrt(mean^2 + sd^2)` = **0.190284 Wh**. The irreducible noise standard deviation rises to
0.233358 Wh at severity 3 because the mean energy rises
(`validation/validate_baseline_vs_learned_output.txt`).

### Model misspecification, on purpose

The generator uses a propulsive efficiency that falls off quadratically away
from the design airspeed,
`eta(V) = 0.62 (1 - 2.5 ((V - 20)/20)^2)`. The analytic baseline
(`conformalband.baseline.PhysicsRegressor`) fits the same equations with a
**constant** efficiency, so it carries an airspeed-dependent bias that no
amount of data removes. Measured: RMSE 0.2029 Wh in distribution against an
irreducible 0.1900 Wh, so about 0.071 Wh of the error is bias.

## Sizes used

| Use | n_fit | n_calibration | n_test | Replicates | Total test points |
|---|---|---|---|---|---|
| `validate_coverage_audit.py` | 1500 | 500 | 600 | 120 | 72 000 per row |
| `validate_breaking_point.py` | 1500 | 500 | 600 | 120 | 72 000 per row |
| `validate_baseline_vs_learned.py` | 1500 | 500 | 1000 | 40 | 40 000 per row |
| `validate_conditional_coverage.py` | 1500 | 500 | 1000 | 30 | 30 000 per severity |
| `examples/*.py` | 1500 | 500 | 600–1000 | 20–40 | — |

The largest array materialised anywhere is 200 000 x 5 float64 = 8 MB, in the
residual-kurtosis measurement in `validate_baseline_vs_learned.py`. Nothing is
written to disk, so no regeneration step is needed and no data file approaches
the 1 MB commit threshold.

## Determinism

Every dataset is a deterministic function of its seed. `make_audit_split(seed)`
draws the fit, calibration and test samples from one generator in that order,
so the whole replicate is reproduced from the seed alone. The audit uses
`seed + replicate_index`. Two consecutive full runs of the validation
directory reproduced every quoted figure exactly; the test suite's wall-clock
time moved from 60.1 s to 88.9 s between runs on contended cores, and not one
measured value changed.

## Limitations of this dataset

1. **It is synthetic, and the model that generated it is the model under
   test.** The physics baseline is misspecified only in the one way the
   generator was written to misspecify it. Real misspecification is not a
   single missing quadratic term.
2. **The covariate shift is a Gaussian mean shift in two coordinates.** That is
   the easiest possible case for a logistic density-ratio estimator, because
   the log ratio is exactly linear in those two coordinates and the logistic
   model is correctly specified for it. The measured agreement between learned
   and declared weights (within 0.0015 of coverage at every severity) should
   not be read as evidence about harder shifts.
3. **No label shift, no concept drift, no sensor faults, no missing data, no
   outliers, no temporal correlation.** The conditional law of energy given the
   covariates is identical in calibration and test, which is exactly the
   assumption weighted conformal needs and which a real deployment does not
   get for free.
4. **Covariates are independent.** Real flight logs have mass correlated with
   mission type, and airspeed correlated with wind.
5. **One airframe, one flight condition.** Steady level flight only: no climb,
   no turn, no acceleration, no battery voltage sag, no temperature effect.
