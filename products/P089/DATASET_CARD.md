# Dataset card — `aligncal.dataset`

Version 0.1.0. Research grade.

**No flight data is used anywhere in this repository.** Every number in
`README.md`, `MODEL_CARD.md` and `validation/VALIDATION.md` is measured against
the seeded synthetic generator described here. Nothing is downloaded, nothing is
cached, and no data file is committed: the largest configuration used anywhere
is 2500 records of 150 samples, about 2 MB of float64, generated in a few
seconds.

## What it contains

A record is a time series of angular rates from a **reference** sensor and one
or more **secondary** sensors, each with a declared nominal mounting rotation
and an injected relative misalignment that the estimators are asked to recover.

| field | shape | unit | note |
|---|---|---|---|
| `times` | (n,) | s | uniform, strictly increasing |
| `reference_rates` | (n, 3) | rad/s | the body rate plus the reference's noise |
| `secondary_rates[name]` | (n, 3) | rad/s | in that sensor's own frame, not de-rotated |
| `secondaries[name].mounting` | (3, 3) | — | the **declared** nominal rotation, a random seeded rotation of magnitude about 1.5 rad |
| `truth[name]` | (3,) | rad | the injected misalignment, ground truth, never visible to an estimator |
| `true_rates` | (n, 3) | rad/s | the noiseless body rate, used only to measure what noise in the Gramian costs |

## The motion, equation (23)

```
w(t) = sum_k a_k sin(2 pi f_k t + phi_k) u_k        rad/s
```

with base frequencies 0.31, 0.17 and 0.23 Hz and phases drawn per seed. The
excitation axes are rotated by a **seeded random rotation**, so no result in this
repository depends on the motion happening to lie along a body axis; measured
over 60 single-axis seeds, the motion is never closer than **7.0421 deg** to a
canonical axis.

| profile | axes excited | excitation shares `p` | rank of the alignment problem |
|---|---|---|---|
| `single_axis` | 1 | (1, 0, 0) | **2** — the component about the rotation axis is unobservable at any data volume |
| `near_single_axis` | 2, one at a `leak` fraction | (1 - O(leak^2), ..., 0) | 3, poorly conditioned; this is what makes the verdict three-valued |
| `two_axis` | 2, equal amplitude | (1/2, 1/2, 0) | **3** — two independent rotation axes suffice |
| `three_axis` | 3, unequal amplitude | (0.505, 0.319, 0.176) measured | 3 |
| `isotropic` | band-limited random | (0.379, 0.336, 0.286) measured | 3 |

`two_axis` giving rank 3 is worth stating because it is easy to assume three
excited axes are needed. By equation (10) the Gramian eigenvalue along a
direction is `(W / s^2) (1 - p)`, so the **unexcited** direction is the
best-observed one.

## The instruments

Each sensor adds independent zero-mean Gaussian white noise with a declared
per-axis one sigma. **That is the whole error model.** There is no bias, no
scale-factor error, no angular random walk, no g-sensitivity, no quantisation,
no latency and no time-tag error. None of the scores in this repository
therefore says anything about any of those, and a real relative-alignment
estimate is usually limited by one of them.

The misalignment is applied as the **exact** rotation of equation (4), not its
linearisation, so the linearisation error of every estimator here is measured
rather than cancelled by construction.

## The populations

| population | function | records | samples | seed | used for |
|---|---|---|---|---|---|
| training | `training_population` | 2500 | 150 | 20261029 | fitting every learned estimator |
| test | `training_population` | 600 | 150 | 70120261 | every score in §5 of VALIDATION.md |
| control | `axis_aligned_record` | 120 | 150 | 55120261 | the unobservable-subspace measurement |
| grid | `generate` | 864 | 150 / 600 / 2400 | 31420261 + cell | the Level 2 recovery sweep |

`training_population` draws per record:

- profile uniformly from `single_axis`, `near_single_axis`, `two_axis`,
  `three_axis` — so the training data **deliberately contains unobservable
  degrees of freedom**;
- peak body rate log-uniformly from 0.01 to 0.08 rad/s;
- per-axis sensor noise one sigma log-uniformly from 5e-5 to 8e-4 rad/s;
- misalignment magnitude uniform on [0, 3] mrad, direction uniform on the
  sphere, so the per-component standard deviation is 1.0 mrad and the population
  mean is within rounding of zero.

That last fact is why the trivial baseline of `aligncal.learned.TrivialEstimator`
is, in effect, "the declared mounting is right": it predicts the training mean.

`axis_aligned_record` is a two-step construction used for the negative control.
The record is generated once to find its dominant rotation axis, then
regenerated with the misalignment placed **exactly along that axis**. The
control is therefore anchored on a chosen truth and not on any estimator's
output, and it is verified to fire: 120 of 120 records rank deficient, with the
injection within 0.000001 deg of the worst-observed Gramian eigendirection.

## Determinism

Every function takes an explicit integer seed and uses
`numpy.random.default_rng`. A given seed reproduces a given record bit for bit
on a given numpy. `tests/test_dataset.py` checks this directly, and checks that
different seeds give different records.

Verified in the test suite: the magnitude bound is respected over 30 seeds, each
secondary gets a distinct mounting and misalignment, the declared mountings are
rotations to 1e-10, the measured rates differ from the noiseless ones by the
declared sigma to 15 per cent, and the exact rotation of equation (4) is
reproduced to 1e-10.

## Limitations

- **Synthetic only.** No flight data, no hardware, no real gyro. The scores do
  not transfer.
- **White noise only.** See above. This is the largest single gap between this
  generator and a real record.
- **The misalignment is constant.** No thermal drift, no launch shift, no
  time-varying misalignment is generated or scored anywhere, although the
  recursive estimator accepts a random-walk process noise for one.
- **Sinusoidal motion.** Four of the five profiles are sums of sinusoids at
  fixed frequencies. A real vehicle's rate history is not, and the excitation
  shares of a real record would have to be measured rather than assumed.
- **No gaps, no dropouts, no time-tag error.** `AlignmentRecord` refuses a
  non-finite rate outright rather than carrying it, because the Gramian of (8)
  would silently become `nan` and every verdict with it.
- **One reference.** Secondaries are estimated independently against a single
  reference; a joint solution over a sensor network is not implemented.
- **The declared mounting is assumed correct.** Control C8 measures what a wrong
  one costs: a 5 mrad declaration error moves the estimate by 5.000000 mrad, and
  nothing here can separate the two.

## Licence and provenance

Generated by code in this repository, released under Apache-2.0 with it.
Copyright 2026 OPTIMA Organisation. No third-party data, no personal data, no
export-controlled content. The closed forms are printed in the module docstring
of `src/aligncal/dataset.py`.
