# Dataset card — photoncount rate-correction dataset 0.1.0

## What it is

Synthetic acquisitions from an event-level photon-counting detector simulator.
Each row is one acquisition: a detector with a known dead-time model, dead time,
afterpulse probability and afterpulse delay, illuminated at a known true rate,
read out as 25 counting windows and reduced to five features and one label.

There is no real data in it. There is no public dataset of
dead-time-and-afterpulsing-corrupted photon counts with known incident rates,
because the incident rate is exactly what a detector cannot tell you; the only
way to have the label is to simulate.

## Generation

```python
from photoncount.dataset import generate_dataset
train = generate_dataset(6000, 20261006)
test = generate_dataset(2000, 99001)
```

Deterministic in the seed. The generator is
`src/photoncount/dataset.py`; the simulator it calls is
`src/photoncount/simulate.py`. Nothing is committed as data: the dataset is about
4 × 10⁷ simulated events and is regenerated in about 20 s; the exact figure for the committed run is printed in
`validation/validate_correction_output.txt` §0.

## Sampling

| Quantity | Range | Distribution |
|---|---|---|
| dead time `τ` | 20 ns to 1 µs | log-uniform |
| loading `x = n τ` | 0.002 to 3.0 | log-uniform |
| afterpulse probability `p` | 0 to 0.15 | uniform |
| delay ratio `t_ap / τ` | 0.3 to 30 | log-uniform |
| dead-time model | paralyzable / non-paralyzable | uniform, half each |
| windows per row | 25 | fixed |
| expected primary events per window | 200 | fixed |

**`x = n τ` is the design variable, not the rate.** Everything about dead time
depends only on that product. The range deliberately straddles the paralyzable
maximum at `x = 1`: below it the closed-form inversions are excellent, at it they
are ill-conditioned, above it the lower-branch inversion is the wrong root. A
dataset that stopped at `x = 0.1` could show nothing, because there the textbook
answer is right.

In the 2000-row held-out set, `n τ` spans 0.0020 to 2.9984 and 14.6 % of rows are
past the paralyzable maximum.

## Features and label

Five features: `log10(m τ)`, `fano_factor`, `afterpulse_probability`,
`log10(t_ap/τ)`, `is_paralyzable`. Label: `log10(n τ)`. Full definitions in
`MODEL_CARD.md` and in the `photoncount.dataset` module docstring.

## Splits

Train and test are generated from **different seeds**, so they are independent
draws rather than a partition of one draw. No row appears in both. The default
split used throughout this repository is 6000 / 2000.

## Known limitations

1. **Uniform counting statistics.** The window length is `200 / n`, so every row
   integrates about 5000 counts and has a counting-noise floor of about 1.4 %
   relative on the observed rate. **Nothing in this dataset says how any method
   behaves at 50 counts or at 10⁶ counts.** That is a real gap: at very low counts
   the closed forms would be noise-dominated and a learned prior might help more;
   at very high counts the floor falls and the model's residual bias would
   dominate.
2. **The window length depends on the true rate.** That is a cost-control choice,
   and it leaves a weak residual coupling between the true rate and the
   Fano-factor estimate. The ablation in `validate_correction_output.txt` §7
   shows the learned margin does not come from it: flattening the Fano column
   leaves the model slightly better. A sixth feature, the mean count per window,
   was removed for exactly this reason — it carried the label outright.
3. **One afterpulse model.** Cascading clusters with a single exponential release
   delay. Real SPADs show multi-exponential or power-law afterpulse tails, which
   this dataset does not contain and the model therefore has never seen.
4. **No dark counts, no background, no gain fluctuation.** Rows are pure
   signal-driven detection events. A real measurement includes a dark-count rate
   and, for an APD, the excess-noise statistics of `photoncount.webb`. Neither is
   in the generator.
5. **No timing jitter, no dead-time dispersion.** The dead time is a fixed number
   with no spread.
6. **No drift.** Detector parameters are constant within a row and across a run.
   Temperature-driven drift in dead time or afterpulse probability, which is the
   usual reason a correction goes stale in the field, is absent.
7. **The simulator is the ground truth, and it is a model.**
   `validation/validate_simulator_output.txt` checks it against every closed-form
   rate law it generalises — the ideal counter, both dead-time laws including past
   the paralyzable maximum, the cascading afterpulse law, the Fano-factor signs,
   and the effective-probability relation. That is the strongest check available
   without hardware. It is not a check against a detector.

## Provenance and licence

Generated entirely by code in this repository, released under the same Apache-2.0
licence. No third-party data, no personal data, no measured data.
