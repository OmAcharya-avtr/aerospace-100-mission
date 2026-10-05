# Dataset card — faultinject campaign severity data

**Version:** 0.1.0 · **Date:** 2026-10-05 · **Status: TESTING**

## What the dataset is

A set of `(fault case, observed severity)` pairs. Each case is one injection —
fault kind, channel, parameters, start step, duration — plus a seed and a run
length. Each severity is the score `faultinject.severity.score` assigns to the
closed-loop response of the reference target, relative to the fault-free run of
the same seed.

There is no downloaded corpus and no file of data in this repository. The data
is generated on demand by committed code, deterministically, and the generation
is cheap enough that storing it would only create a chance for it to drift from
the code.

## How it is generated

```bash
cd products/P034
python -c "
from faultinject.campaign import build_pool, evaluate_pool
pool = build_pool(pool_seed=1, replicates=2)   # 496 cases, all 248 cells twice
sev = evaluate_pool(pool)                      # one target execution per case
print(len(pool), sum(1 for s in sev if s >= 0.6))
"
```

- `build_pool(pool_seed, replicates)` walks the 248 coverage cells in canonical
  order and constructs `replicates` cases inside each, drawing parameters
  log-uniformly or uniformly inside the cell's bins and integer quantities from
  the integers that actually bin to the requested bin. Every constructed case is
  re-binned and checked against the cell it was asked for.
- `evaluate_pool(pool)` executes each case against the reference target and
  returns its severity.
- Randomness: `numpy.random.default_rng([pool_seed, 9])` for pool construction,
  `[case seed, 1]` for plant and measurement noise, `[case seed, 2]` for handler
  draws. PCG64 throughout.

## Size and cost

| Quantity | Value |
|---|---|
| Coverage cells | 248 |
| Benchmark pool, `replicates=2` | 496 cases |
| Pools used in the benchmark | 3 (`pool_seed` 1, 2, 3) → 1488 cases |
| Run length per case | 150 steps at dt = 0.02 s (3.0 s of simulated time) |
| Cost per case | about 1 ms on one shared CPU core |
| Cost of one 496-case pool | about 1 s |
| Training set actually seen by the model | ≤ 60 cases, the campaign budget |
| Stored bytes in the repository | 0 — the data is regenerated, never committed |

## Source and limitations

- **Entirely synthetic.** The source is the simulator in `faultinject.target`: a
  double-integrator plant, a fixed-gain Kalman filter at the Riccati fixed
  point, and a saturating PD controller tracking a 0.2 Hz sinusoid. It is a
  textbook loop chosen because it has a closed form, not because it resembles
  any particular vehicle.
- **One target.** Every severity in the dataset comes from that one loop. The
  severity distribution, and therefore the whole benchmark, would change with a
  different plant, different gains, a different reference signal or a different
  noise level.
- **The label is a design choice.** Severity is
  `0.5 · deviation + 0.3 · RMSE excess + 0.2 · violation fraction`, with
  overrides to 1.0 for a non-finite trace and a floor of 0.9 for a divergent
  one. It is not a measured physical quantity and not a criticality rating.
- **Coverage-stratified, not representative.** The pool is built to cover every
  cell equally, which is deliberate for a campaign benchmark but means the
  severity distribution is a property of the taxonomy's binning, not of any
  real-world fault frequency. Nothing in this repository claims to know how
  often a real sensor sticks.
- **Class balance.** 28.49 % of the 1488 benchmark cases are severe at the 0.6
  threshold (`validation/benchmark_output.txt`). Over one 496-case pool the
  severity labels fall as 235 negligible, 48 minor, 71 moderate, 142 severe.
- **One fault per case.** No case contains two simultaneous faults.
- **No personal data, no third-party data, no licensing constraints.** Nothing
  was downloaded; everything is produced by the code in this repository.

## Intended use

Benchmarking campaign search strategies against each other on a known ground
truth, and fitting the campaign prioritiser inside a campaign. It is not a
benchmark for fault detection, not a dataset about any real system, and not
suitable for training anything intended to make a claim about hardware.

## Maintenance

The data regenerates from the code, so it cannot go stale relative to it. Any
change to `faultinject.target`, `faultinject.faults` or `faultinject.severity`
changes the dataset; `tests/test_regression.py` holds six recorded severities
and a plausible-band check on the severe fraction so that such a change is
noticed rather than absorbed.

## Credits

MIT, © 2026 OPTIMA Organisation. The credits line for this product is in
README.md.
