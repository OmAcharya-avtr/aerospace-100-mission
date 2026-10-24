# Dataset card — relfiledelivery scenario generator

Version 0.1.0 · Apache-2.0 · © 2026 OPTIMA Organisation

**No flight data is used anywhere in this repository.** Every row is generated
by `relfiledelivery.dataset` from a seeded `numpy.random.Generator` and every
target is produced by `relfiledelivery.simulate.simulate_transfer`. Every result
derived from this dataset is a statement about this generator, and the
generator's equations are in its module docstring so that the statement can be
checked rather than taken.

## What a row is

One declared scenario — a file, a link and a transfer mode — plus the realised
delivery time of **one** simulated transfer over it. One run, not an average:
the learned component is asked for a calibrated interval, so its training target
has to carry the run-to-run spread as well as the scenario-to-scenario spread.

## Not committed

The tables are not committed. They are regenerated deterministically from two
integers by

```bash
python validation/validate_predictors.py
```

The full benchmark table is 4200 scenarios and takes about 13 seconds to
generate on two cores. Committing it would add a megabyte of data that the
generator reproduces exactly, and `validation/validate_reproducibility.py`
asserts that it does, column by column including the simulated target.

## Regimes

Two, and they exist so that the learned component can be scored separately in
the regime where the closed form is exact and the regime where it is not.

| Regime | Outage process | Contact | Closed form |
|---|---|---|---|
| `no_outage` | none | continuous | **exact** under A1–A4 |
| `correlated_outage` | Gilbert two-state | 4 to 10 declared windows | an extension |

## Sampling

Per scenario, from a generator seeded on `(seed, regime index)`:

| Quantity | Distribution | Range |
|---|---|---|
| `pdu_count` | `round(10 ** U(1.6, 2.7))` | 40 to 501 units |
| `pdu_payload_bytes` | uniform over `{256, 512, 1024}` | bytes |
| `file_bytes` | `(pdu_count - 1) * payload + U{1, payload}` | so the last unit is short on all but a measure-zero set |
| `pdu_header_bytes` | fixed | 12 bytes |
| `forward_rate_bps` | `10 ** U(4.7, 6.3)` | 50 kbit/s to 2 Mbit/s |
| asymmetry | `10 ** U(0.7, 2.3)` | 5 to 200 |
| `one_way_light_time_s` | `10 ** U(-1.4, 0.6)` | 0.04 s to 4 s |
| `pdu_loss_probability` | `U(0, 0.25)` | dimensionless |
| mode | uniform over deferred-NAK and immediate-NAK | — |

`pdu_count` is sampled directly rather than derived from a file size because it
is the quantity that sets the cost of a simulated run, and bounding it bounds
the compute budget.

In `correlated_outage`, additionally, with `C` the clean-pass forward time of
this file on this link:

| Quantity | Distribution | Range |
|---|---|---|
| `mean_down_s` | `C * 10 ** U(-0.6, 0.9)` | 0.25 to 8 clean passes |
| outage duty | `U(0.55, 0.95)`; `mean_up_s = mean_down_s * duty / (1 - duty)` | dimensionless |
| window count | uniform over `{4, …, 10}` | — |
| `window_duration_s` | `C * 10 ** U(-0.2, 0.8)` | 0.63 to 6.3 clean passes |
| contact duty | `U(0.25, 0.90)`; window period is duration / duty | dimensionless |

**The outage and contact time scales are drawn relative to the clean-pass
time**, not from absolute second ranges. This is a declared design choice and it
is the reason the two regimes differ. The first version of the generator drew
them absolutely, and it produced a regime that was not a regime: outages an
order of magnitude shorter or longer than the transfer either never bite or make
every run fail, and both regimes completed 100 per cent of runs with
overlapping delivery-time distributions. With relative scales the
correlated-outage regime completes about 86 per cent of runs and its 95th
percentile delivery time is roughly twice the no-outage regime's.

## Unacknowledged mode is excluded

The predictor dataset draws only deferred-NAK and immediate-NAK. Unacknowledged
mode completes with probability `(1 - p) ** n`, which over 40 to 501 units at
up to 25 per cent loss is effectively zero, so nearly every such row would be
dropped for having no delivery time. Unacknowledged mode is measured
everywhere else in the package — the mode comparison, the per-contact
completion table and the worked example all include it — and it is excluded
here only from the regression.

## Censoring

A scenario whose run did not finish inside its declared horizon — the end of
the last contact window, or the transmission cap — carries `completed = False`
and no target, and is dropped by `fit_split`.

| Split | Scenarios | Usable | Dropped | Completion rate |
|---|---|---|---|---|
| train, seeds from 100000 | 3000 | 2810 | 190 | 0.9367 |
| test, seeds from 900000 | 1200 | 1127 | 73 | 0.9392 |

Measured by regime on the test split: `no_outage` completes 1.0000 of 600
scenarios and `correlated_outage` completes 0.8783 of 600. **Every learned
prediction is therefore conditional on completion** and nothing in this package
learns to predict *whether* a transfer completes. That is a real limitation and
it is in `MODEL_CARD.md` as one.

## Target distribution

Delivery time in seconds, on the test split:

| Regime | q0.05 | q0.50 | q0.95 | max |
|---|---|---|---|---|
| no_outage | 0.8301 | 6.0359 | 31.9983 | 81.6823 |
| correlated_outage | 0.9344 | 8.2814 | 55.9166 | 960.7278 |

The correlated-outage maximum of 960.73 s against a median of 8.28 s is the
tail this package exists to report: one scenario in 527 took a hundred and
sixteen times its own regime's median.

## Known limitations of the generator

- **The loss process is independent loss plus a two-state Gilbert burst.** A
  real link has structure neither term has: weather, pointing, interference,
  rate adaptation, queueing at the ground station.
- **Contact windows are uniform.** Real passes vary in duration and spacing with
  the orbit and the station; `uniform_windows` is a convenience, not an orbit
  propagation, and `LinkConditions` accepts any non-overlapping window list.
- **Control PDUs are never lost** in the simulation that produces the target, so
  every target is optimistic by whatever a NAK timer would have cost.
- **The ranges are declared, not derived from any mission.** They are wide
  enough to span low Earth orbit to beyond the Moon in light time, and they are
  not a model of any particular link.
- **One run per scenario.** The scenario-to-scenario and run-to-run variance are
  not separately identifiable from this table, which is the right design for
  training a quantile model and the wrong one for estimating either variance
  component on its own.

## Reproducing

```python
from relfiledelivery import generate_table, fit_split

train = fit_split(generate_table(1500, 100_000))
test = fit_split(generate_table(600, 900_000))
```

Deterministic in the two integers. `validation/validate_reproducibility.py`
asserts that a regenerated table is identical column by column, including the
simulated delivery time.
