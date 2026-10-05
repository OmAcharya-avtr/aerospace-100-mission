# Dataset card — telemetryool synthetic housekeeping telemetry, v0.1.0

Research grade. Not flight-qualified, not certified, not approved for
operational aerospace use.

## 1. What it is

Synthetic multichannel housekeeping telemetry, generated on demand by
`src/telemetryool/synthetic.py`. Nothing is committed as data: every array is
regenerated deterministically from a seed, so the repository stores the
generator rather than its output.

Each block has shape `(n_windows, window_length, n_channels)` and holds
*standardised deviates*: every nominal channel is marginally `Normal(0, 1)`.
Working in sigma units means the control-chart designs in `telemetryool.arl`
apply without a scale conversion and the "shift in sigma" axis of every
detection-delay curve is unambiguous.

## 2. Why synthetic, and what that costs

The subject of this package is the **designed** false-alarm rate, and a designed
rate can only be checked against a known nominal hypothesis. Real housekeeping
telemetry carries no ground-truth label for "nominal": a measured false-alarm
rate on it would be an estimate of an unknown quantity, with no way to tell a
genuine false alarm from an undetected real event.

The cost is stated plainly and applies to every number in this repository:

> **Results here characterise the detectors under this generator's model, not
> under any real spacecraft's telemetry.** No real mission data was used at any
> point. A detector that wins here may lose on a real channel, and the
> false-alarm rates measured here will not transfer to a channel whose nominal
> behaviour differs from the model below — most importantly, to a channel with
> serial correlation.

That last point is not hypothetical. Section 2 of `validation/VALIDATION.md`
measures a 7.3× inflation of the designed false-alarm rate at an AR(1)
coefficient of only 0.3.

## 3. Nominal model

`x_t = L e_t`, where `e_t` is an `n_channels`-vector AR(1) process

```
e_t = rho e_{t-1} + sqrt(1 - rho^2) n_t,     n_t ~ Normal(0, I)
```

so each component is marginally unit variance for any `rho` in `[0, 1)`, and `L`
is the lower Cholesky factor of the target correlation matrix (unit diagonal, so
the marginals stay unit variance). The AR(1) state is burned in for `burn_in`
samples (default 200) before the window starts, so no window contains a
transient. The variance relation `var = sigma_n^2 / (1 - rho^2)` is the standard
AR(1) result; see Box, Jenkins and Reinsel, *Time Series Analysis: Forecasting
and Control*, 4th ed., Wiley 2008, chapter 3.

Settings used by the headline comparison (`validation/validate_matched_far.py`):

| Parameter | Value |
|---|---|
| `n_channels` | 4 |
| cross-channel correlation | equicorrelation 0.6 |
| `rho_time` (serial correlation) | 0.0 |
| `window_length` | 100 samples |
| `burn_in` | 200 samples (unused at `rho = 0`) |

`validation/validate_far_design.py` additionally sweeps
`rho_time ∈ {0, 0.3, 0.6, 0.9}` on a single channel, specifically to measure
what the independence assumption is worth.

Verified properties (`tests/test_synthetic.py`, with the stated tolerances):

| Property | Check | Tolerance |
|---|---|---|
| Marginal standard deviation is 1 | 600 000 samples per channel; sd of the sample sd is `1/sqrt(2N) = 0.00091` | 0.005 (5 sigma) |
| Marginal mean is 0 | same block | 0.01 |
| Lag-1 autocorrelation equals `rho_time` | at `rho = 0.6` | 0.01 |
| Cross-channel correlation equals the target | at 0.6, 4 channels | 0.01 |
| Generation is bit-identical for a fixed seed | two generators, same seed | exact |

## 4. Anomaly models

Each is injected into every window of a block at a stated onset index.

| Kind | Definition | Magnitude units | What it is meant to represent |
|---|---|---|---|
| `step` | `+delta` added to the affected channels from `onset` | sigma | A sustained bias: a shifted reference, a degraded sensor |
| `drift` | `+slope * (t - onset)` from `onset` | sigma per sample | A ramp: a slow leak, a thermal trend |
| `stuck` | Affected channels hold their `onset` value for the rest of the window | — | A frozen sensor or a latched data path |
| `spike` | One sample at `onset` offset by `+delta` | sigma | A transient, a single corrupted word |
| `decorrelate` | Affected channels replaced from `onset` by independent unit-variance noise | — | A broken physical coupling: every marginal distribution is unchanged and only the cross-channel structure breaks |

`decorrelate` exists specifically so that the comparison contains an anomaly a
univariate monitor is **blind to by construction**, and the measurement confirms
it: all three univariate methods sit at the false-alarm rate (Pd 0.0320, 0.0355,
0.0345) while the multivariate ones reach 0.9240 and 0.9340.

## 5. Splits used by the headline comparison

Four disjoint blocks, each from its own derived seed off a base seed of
20261005, so no sample is used in two roles:

| Block | Seed | Windows | Samples | Role |
|---|---|---|---|---|
| Train | base + 1 | 1 500 | 150 000 | Fit the models and the feature standardisation |
| Calibrate | base + 2 | 10 000 | 1 000 000 | Set each detector's threshold |
| Measure | base + 3 | 20 000 | 2 000 000 | Measure the delivered false-alarm rate; the negatives of every confusion matrix |
| Anomalous ×5 | base + 10 + i | 2 000 each | 200 000 each | Detection probability, delay, ROC |

Sizes were chosen from the precision required, not from convenience. At
`alpha_W = 0.05`, 20 000 windows give a binomial standard error of 0.0015411
(3.1 % relative); the threshold's own sampling error adds
`sqrt(0.05 × 0.95 / 10000) = 0.00218`, for a combined 0.0026693 on the delivered
rate.

## 6. Regeneration

```bash
PYTHONPATH=src python3 -c "
import numpy as np
from telemetryool.synthetic import NominalModel, equicorrelation, generate_nominal
model = NominalModel(4, correlation=equicorrelation(4, 0.6))
block = generate_nominal(model, 20000, 100, np.random.default_rng(20261005 + 3))
print(block.shape, block.std(axis=(0, 1)))
"
```

Every validation script regenerates everything it needs from its own fixed
seeds; nothing is cached and nothing large is committed. The four PNGs in
`screenshots/` are the only committed binaries, and each is produced by the
example script of the same name.

## 7. Licence and provenance

Generated by code in this repository, licensed Apache-2.0, © 2026 OPTIMA
Organisation. No third-party data, no mission data, no personal data, and
nothing that could identify a vehicle, an operator or an individual.

## 8. Known limitations of the generator

1. **Gaussian marginals.** Real housekeeping channels are quantised to a
   telemetry word, are often bounded, and frequently have skewed or bimodal
   distributions driven by duty cycling. None of that is modelled.
2. **No mode-dependent nominal distribution.** `telemetryool.limits` supports
   mode-dependent *limit tables*, but the generator's nominal distribution does
   not change with mode. A real channel's mean and variance usually do.
3. **No calibration drift, no telemetry gaps, no packet loss.** The validity
   mask in `telemetryool.limits` is exercised by hand-built traces in the tests
   and the example, not by the generator.
4. **Stationary cross-channel correlation.** Only the `decorrelate` anomaly
   breaks it, and it breaks it completely rather than gradually.
5. **One anomaly per window, with a known onset.** Real faults overlap, recur,
   and have no label. Detection-delay numbers measured against a known onset are
   an upper bound on what is knowable in practice.
