# Dataset card — degradepredict synthetic degradation fleet

**Version 0.1.0 · 2026-10-16 · Apache-2.0 · © 2026 OPTIMA Organisation**

Research-grade. Not flight-qualified, not certified, not approved for
operational aerospace use.

## What it is and why it is synthetic

A fleet of degrading subsystems, each with a housekeeping health metric
sampled on a fixed cadence, a declared limit, a forecast origin, a follow-up
period and a **known true crossing time**.

The last of those is the reason the data set exists. The quantity this package
forecasts is only observable after it happens. To measure whether a predicted
interval around a crossing date is calibrated, the true crossing time has to be
known for every unit in the test set **including the units that never crossed
inside the record** — and those are the majority. No flight data set supplies
that, and no amount of waiting supplies it for the units still in service. A
declared generator does.

The price is stated plainly: every number in this repository is a statement
about this generator, not about any spacecraft.

## Generation

```bash
python -m degradepredict generate --n 400 --seed 20261017 --out population.csv
```

Nothing is committed. The CSV is 2.26 MB for the 400-trajectory test
population and 6.72 MB for the 1200-trajectory training population, both over
the 1 MB limit this portfolio places on committed data, and both deterministic,
so they are regenerated rather than stored. A second run with the same seed produces
a byte-identical file (`validation/validate_cli_output.txt`).

Determinism: each candidate unit draws from its own
`numpy.random.SeedSequence` child, so the k-th accepted unit is identical for
any requested population size at or above k, and the selection step for one
unit cannot perturb another.

## Schema

One row per telemetry sample. Trajectory-level fields repeat on every row of
that trajectory.

| column | unit | description |
|---|---|---|
| `traj_id` | index | unit index within the generated population |
| `family` | — | `linear`, `exponential` or `powerlaw`: the true mean shape |
| `noise_sd_percent` | percent of BOL | observation-noise standard deviation |
| `crossing_time_days` | d | **true** crossing time of the noise-free mean curve |
| `origin_days` | d | forecast origin; observations run over `[0, origin]` |
| `followup_days` | d | follow-up duration after the origin, drawn independently |
| `remaining_true_days` | d | `crossing_time_days - origin_days`, always positive |
| `remaining_observed_days` | d | `min(remaining_true_days, followup_days)` |
| `event` | 0 or 1 | 1 if the crossing was observed inside the follow-up |
| `time_days` | d | sample service age |
| `capacity_percent_bol` | percent of BOL | noisy metric value at that sample |

An analyst in the field would have every column except `crossing_time_days`
and `remaining_true_days` for the censored units. Those two are the oracle and
are used only by the scoring layer.

## The metric and the limit

Battery usable capacity as a percentage of beginning-of-life capacity,
decreasing with service age in days, against a limit of **80 percent of BOL**.

That limit is a **declared convention**, borrowed from IEEE Std 1188-2005
(IEEE Recommended Practice for Maintenance, Testing, and Replacement of
Valve-Regulated Lead-Acid Batteries for Stationary Applications), which defines
end of service life as the point at which measured capacity falls below 80
percent of rated capacity. It is used because it is a published, checkable
number. It was **not** derived from any spacecraft battery qualification, and
the standard it comes from is about a different cell chemistry.

## Mean shapes

`y` is percent of BOL, `t` is service age in days, `y0` is beginning-of-life
calibration drawn uniformly from [99, 101] percent.

| family | mean curve | parameter | source for the form | validity |
|---|---|---|---|---|
| `linear` | `y = y0 - b t` | `b` in % / d | none; a first-order expansion | only over a window short compared with the curvature scale |
| `exponential` | `y = y0 exp(-k t)` | `k` in 1/d | none; a declared alternative shape with a loss rate proportional to the remaining metric | no mechanism claimed |
| `powerlaw` | `y = y0 (1 - B t**z)` | `B` in d^-z, `z` dimensionless | Wang et al. (2011), *J. Power Sources* 196(8):3942-3948, exponent 0.55 against charge throughput; Broussely et al. (2005), *J. Power Sources* 146(1-2):90-96, square-root-of-time SEI-limited calendar fade, z = 0.5 | one ageing regime only; `B t**z < 1`; no lithium plating, no sudden death, no thermal excursion |

Weights are 0.30 linear, 0.30 exponential, 0.40 power-law, so **no model in
this package is correctly specified for the whole population**, which is the
point. `z` is drawn per trajectory from [0.45, 0.65], bracketing both cited
values, so the physics-shaped baseline's fixed `z = 0.55` is slightly wrong
even on its own family.

Wang et al. fit the power law against **charge throughput at a controlled
temperature**. Using service age in days instead assumes a duty cycle and a
thermal environment constant enough for throughput to be proportional to
elapsed time. That is a strong assumption and it is declared, not justified.

## Declared generator parameters

From `GeneratorConfig`, all defaults:

| parameter | value | meaning |
|---|---|---|
| `sample_interval_days` | 7.0 | housekeeping cadence |
| `y0_range` | (99.0, 101.0) % | BOL calibration scatter |
| `log_crossing_mean`, `log_crossing_sd` | 7.0, 0.55 | lognormal true crossing time in days; the crossing is drawn first and the shape parameter solved from it, so the crossing-time distribution is exactly the declared one |
| `t0_range` | (119.0, 700.0) d | forecast-origin service age |
| `followup_range` | (120.0, 1200.0) d | follow-up after the origin, independent of the degradation state |
| `noise_sd_range` | (0.15, 0.60) % | observation-noise standard deviation |
| `powerlaw_z_range` | (0.45, 0.65) | fade exponent |
| `family_weights` | (0.30, 0.30, 0.40) | shape mixture |

## Censoring model

A unit is observed from service age 0 to the forecast origin, then followed for
a further duration drawn **independently of its degradation state** — the unit
leaves the monitored fleet when it is decommissioned, when the housekeeping
channel is lost, or when the archive ends. If the crossing falls inside that
follow-up the crossing time is recorded (`event = 1`); otherwise all that is
known is that the remaining life exceeds the follow-up (`event = 0`).

Resulting censoring at the shipped seeds
(`validation/validate_headtohead_output.txt`):

| split | n | complete | censored | censored fraction | median true remaining life, complete | median true remaining life, censored |
|---|---|---|---|---|---|---|
| train, seed 20261016 | 1200 | 502 | 698 | 0.5817 | — | — |
| test, seed 20261017 | 400 | 173 | 227 | 0.5675 | 402 d | 1126 d |

Family counts: train 377 linear / 329 exponential / 494 power-law; test 124 /
119 / 157.

**Independence of censoring and degradation is true here by construction.** In
a real fleet it is routinely false, because units are removed precisely because
they are degrading. Every inverse-probability-of-censoring result in this
repository rests on an assumption the generator satisfies and reality often
does not.

## Selection

A unit is only a forecasting case if it has not already crossed the limit at
the forecast origin, so candidates with `origin >= crossing` are rejected. This
is the same selection a real fleet imposes and it makes the retained sample
**length-biased** with respect to the crossing-time distribution: short-lived
units are under-represented. It is left in deliberately and stated rather than
corrected.

## Observation noise

Independent, zero-mean Gaussian, constant standard deviation per trajectory,
added to the mean shape at each sample.

Real housekeeping telemetry is not like this. It has autocorrelated residuals,
temperature and state-of-charge dependence, telemetry quantisation, dropouts,
calibration steps at ground contacts, and outliers. This generator has none of
them, and every residual-based uncertainty estimate in this repository is
therefore optimistic relative to flight data by an amount this data set cannot
quantify.

## Known limitations, collected

1. Synthetic. No flight data, no laboratory data, no physical article.
2. Three mean shapes. No regime change, no knee, no sudden-death failure, no
   recovery, no step from a reconfiguration.
3. Gaussian independent noise only.
4. Fixed 7-day cadence, no gaps, no irregular sampling, no missing samples.
5. Censoring independent of state by construction.
6. One metric, one limit, one direction. The `direction="above"` branch of
   `LimitSpec` is unit tested but has no end-to-end evidence here.
7. The 80 percent limit is a borrowed convention from a standard about a
   different chemistry.
8. The power-law form is cited against charge throughput and used against
   calendar time.
9. No covariates: no temperature, no depth of discharge, no cycle count. A real
   degradation model would use them and a real censoring model would condition
   on them.

## Licence and provenance

Generated by code in this repository under Apache-2.0, © 2026 OPTIMA
Organisation. No third-party data is included or required.
