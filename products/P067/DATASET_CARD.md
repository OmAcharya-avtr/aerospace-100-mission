# Dataset card: acqsearch scenarios

Version 0.1.0, 2026-10-16. File: `data/scenarios.json`, 147067 bytes,
SHA-256 `8c320d4a436ef7faf378139669c747e001ce4f416c46730df7d99d26049334f7`.

## What it is

300 synthetic acquisition scenarios. A scenario is a complete set of declared
terminal and link parameters: the pointing uncertainty, the beam and detector
geometry, the gimbal rate and settle time, the photon rates and count
threshold, the dwell time, the false-alarm verification time and the
scintillation strength. Nothing in a scenario is a result; every field is an
input to the physics in `acqsearch.geometry` and `acqsearch.detection`, so the
dataset cannot carry a conclusion. What it determines is which corners of the
parameter space the scan patterns and the policies are compared on.

## Source

**Entirely synthetic. No hardware was measured and no datasheet was
transcribed.** The ranges below were chosen to bracket values that are
plausible for a small free-space optical crosslink terminal. They are a
declared sampling envelope and nothing more; they are not a survey of flown
hardware and should not be cited as one.

## Generation

```bash
python data/regenerate_data.py
```

`generate_scenarios(300, 20261016)` draws each parameter independently and
uniformly from its range using `numpy.random.default_rng(20261016)`. The
output is byte-identical on every run; `tests/test_scenarios.py` regenerates
the file into a temporary directory and compares it with the committed bytes.

## Declared ranges

| field | unit | declared range | observed min | observed max | observed mean |
|---|---|---|---|---|---|
| `sigma_rad` | rad | 1.00e-4 to 2.20e-4 | 1.00169e-4 | 2.19847e-4 | 1.62064e-4 |
| `beam_divergence_rad` | rad, full, 1/e^2 | 8.0e-5 to 1.4e-4 | 8.04993e-5 | 1.39919e-4 | 1.10651e-4 |
| `detector_fov_rad` | rad, full | divergence x (1.5 to 3.0) | 1.28186e-4 | 3.94450e-4 | 2.47036e-4 |
| `slew_rate_rad_s` | rad/s | 5.0e-3 to 2.0e-2 | 5.01103e-3 | 1.99260e-2 | 1.19709e-2 |
| `settle_time_s` | s | 1.0e-3 to 4.0e-3 | 1.00066e-3 | 3.99583e-3 | 2.49650e-3 |
| `signal_rate_hz` | 1/s, on-axis | 1.2e4 to 3.5e4 | 12059 | 34783 | 23561.4 |
| `background_rate_hz` | 1/s | 1.0e3 to 4.0e3 | 1004.13 | 3991.07 | 2587.56 |
| `threshold_counts` | counts | integer 6 to 10 | 6 | 10 | - |
| `dwell_s` | s | 8.0e-4 to 2.2e-3 | 8.10754e-4 | 2.19109e-3 | 1.49061e-3 |
| `verification_time_s` | s | 0.2 to 1.0 | 0.202607 | 0.995255 | 0.600216 |
| `rytov_variance` | dimensionless | 0.05 to 0.60 | 0.0522419 | 0.598333 | 0.324475 |

Threshold counts: 58 scenarios at 6, 45 at 7, 56 at 8, 66 at 9, 75 at 10.

The deadline is not a sampled field. It is three times the scenario's nominal
pass period, so scenarios with different timing are censored at a comparable
point in their own scan rather than at a common wall clock.

## What the scenarios turn into

Measured over the first 60 scenarios:

| derived quantity | min | median | max |
|---|---|---|---|
| cells in the grid | 25 | 85 | 233 |
| prior-weighted mean cell detection probability | 0.4228 | 0.9413 | 0.9996 |
| nominal pass period (s) | 0.4967 | 2.7399 | 84.5831 |

The 84.58 s pass period is a scenario with a wide uncertainty cone, a narrow
beam and a slow gimbal. It is inside the declared ranges and it is kept,
because an easy-only benchmark is not a benchmark.

## Split

By `scenario_id`, deterministically: ids 0 to 194 train (195 scenarios), ids
195 to 299 held out (105 scenarios). `split_scenarios` sorts by id before
cutting, so the split does not depend on list order. No scenario appears in
both halves, and `tests/test_benchmark.py` asserts that no held-out scenario
contributes a training row.

## Limitations

1. **The parameters are sampled independently and they are not independent.**
   A terminal with a 140 urad beam does not also have a 5 mrad/s gimbal and a
   35 kHz signal rate for any physical reason; the joint distribution here is
   a product of marginals and that is wrong. Some of the 300 scenarios
   describe terminals that could not be built. This is the dataset's
   principal limitation and it affects every number fitted or measured on it.
2. **No hardware, no flight data, no literature survey.** The ranges are
   declared by the author. Treat them as the envelope of an experiment, not
   as a claim about real terminals.
3. **The pointing prior is isotropic, zero-mean and stationary in every
   scenario.** Real open-loop pointing error is usually anisotropic, often
   biased, and drifts during the search.
4. **The link is modelled only through two photon rates.** Range, aperture,
   transmit power, atmospheric attenuation and detector efficiency all
   collapse into `signal_rate_hz` and `background_rate_hz`, so the dataset
   cannot express a correlation between them that a link budget would impose.
5. **Scintillation enters only as a Rytov variance with a lognormal
   irradiance model**, which Andrews & Phillips 2005 Ch. 9 gives for the
   weak-fluctuation regime. The sampled range stays inside it (maximum
   0.598), but there is no temporal correlation: each dwell draws an
   independent irradiance, which is optimistic for a millisecond-scale scan.
6. **There is no target motion.** The target position is fixed for the
   duration of an episode.
7. **300 scenarios is small.** The held-out half is 105 scenarios, and with 8
   target draws each the standard error on a policy mean is about 0.15 s
   against means of 2 to 5 s. Paired comparison is used precisely because the
   unpaired standard error is too large to resolve the gaps that matter.

## Licence and provenance

Generated by this repository, released under the same Apache-2.0 licence.
Copyright (c) 2026 OPTIMA Organisation. No third-party data is included and no
personal data of any kind is involved.
