# Validation evidence — acmpilot 0.1.0

Validation level 3. Every number below was produced by running the script named
beside it, in this container, on 2026-10-06, with Python 3.13.16, NumPy 2.5.3,
SciPy 1.18.1 and scikit-learn 1.9.1. The raw standard output of each script is
saved next to it as `validate_*.txt` and is the authority; this file is a
summary of those files and adds nothing to them.

**Everything validated here is a property of this package's simulator.** No
measured scintillometer record, no field link telemetry and no laboratory modem
measurement is used anywhere. The analytic references are mathematics (Rice's
level-crossing formula, the Gaussian AR(1) conditional moments, the Reed-Solomon
MDS bound, the AWGN error-rate closed forms), and checking an implementation
against mathematics is not the same thing as checking a model against reality.
The second has not been done.

## How to reproduce every number

```bash
cd acmpilot
pip install -e ".[dev,examples]"
cd validation
python validate_channel.py                  > validate_channel.txt
python validate_modcod_thresholds.py        > validate_modcod_thresholds.txt
python validate_policies.py                 > validate_policies.txt
python validate_predictor.py                > validate_predictor.txt
python validate_predictor_gammagamma.py     > validate_predictor_gammagamma.txt
```

Run `validate_modcod_thresholds.py` first: it writes `modcod_thresholds.json`,
which the other three load so that they score policies against the same measured
ladder. If the file is absent they measure it in-process instead, with the same
seed, so the numbers do not change.

Measured wall-clock on 2 contended cores: 6 s, 18 s, 6 s, 65 s, 77 s. All five
are inside the 3-minute budget stated in the README.

## V1 — the channel (`validate_channel.py`, `validate_channel.txt`)

400 000 samples per check, slot 1 ms.

| # | Check | Reference | Result | Tolerance | Verdict |
|---|---|---|---|---|---|
| V1.1 | Gauss-Markov driver lag-1 correlation | `rho = exp(-dt/tau_c)`, eq (2) of `channel.py` | worst relative error 2.18e-3 over `tau_c` = 2, 5, 10, 25 ms; variance 0.989 to 0.997 | 1e-2 relative | PASS |
| V1.2 | measured 1/e correlation time of the dB path | the input `tau_c` | worst relative error 2.02e-2 over `tau_c` = 5, 10, 25 ms | 5e-2 relative | PASS |
| V1.3 | lognormal scintillation index | `sigma_I^2 = exp(4 sigma_chi^2) - 1`, eq (5) | worst relative error 2.06e-2 at `sigma_I^2` = 0.2, 0.5, 1.0 | 5e-2 relative | PASS |
| V1.4 | gamma-gamma scintillation index | Al-Habash, Andrews & Phillips 2001, eq (7) | worst relative error 1.15e-2 at `sigma_I^2` = 0.3, 0.7, 1.5 | 5e-2 relative | PASS |
| V1.5 | gamma-gamma correlation time against its driver's | the module docstring's own statement that it is shorter | ratio 0.931, 0.874, 0.792 at `sigma_I^2` = 0.3, 0.7, 1.5 | ratio < 1 | PASS, and the size is reported |
| V1.6 | fade duration is emergent, not a parameter | — | mean fade duration 1.80, 2.41, 3.31, 4.63, 6.57 ms at `tau_c` = 2.5, 5, 10, 20, 40 ms; log-log slope **0.467**, not 1 | none; measurement | reported, see below |
| V1.7 | fade accounting consistency | `mean_fade_duration * level_crossing_rate = outage_fraction` | worst relative error 2.8e-16 over 9 cases | 1e-6 relative | PASS |
| V1.8 | level-crossing rate | Rice 1944/1945, eqs (10)-(12) with the discrete-sampling derivative variance | measured/analytic ratio 0.983 to 1.015 over 9 (`tau_c`, threshold) pairs | 3e-2 relative | PASS |

**V1.6 is the honest limitation, not a pass.** Mean fade duration scales as
roughly `tau_c**0.47`, not `tau_c**1`, because at a fixed 1 ms slot rate the
*median* fade stays pinned at 2 ms (short re-crossings of the threshold dominate
the count) while only the tail lengthens — the measured maximum fade grows from
14 ms to 156 ms over the same range, a log-log slope of about 0.68
(`examples/channel_and_fades.py`). The fade-duration distribution of this model
is therefore **sampling-rate dependent**: resampling the same physical channel at
a different slot rate changes it. Anyone sizing an interleaver or a
retransmission buffer from these numbers must resample at their own slot rate
first.

V1.8 found and fixed a real defect during this build: the first implementation of
`rice_level_crossing_rate_hz` used `1/(2*pi*tau_c)` as the rate prefactor, which
disagreed with its own docstring and with the measurement by a factor of 3 to 6
that grew with `tau_c`. The docstring was right and the code was wrong. With the
documented eq (11) prefactor the agreement is 1.7% worst case.

## V2 — MODCOD thresholds (`validate_modcod_thresholds.py`)

| # | Check | Reference | Result | Tolerance | Verdict |
|---|---|---|---|---|---|
| V2.1 | Monte Carlo uncoded BER, BPSK and QPSK | exact closed forms, eqs (2)-(3) of `modulation.py` | worst standardised deviation 1.69 sigma over 10 points with > 50 observed errors | 3 sigma | PASS |
| V2.1 | Monte Carlo uncoded BER, 8PSK and 16QAM | nearest-neighbour / high-SNR approximations, eqs (4)-(5) | agreement within 1.4 sigma at and above 14 dB; deviations of 7 to 82 sigma below 10 dB | none claimed | expected, reported |
| V2.2 | Reed-Solomon MDS properties | `d_min = n-k+1`, `t = floor((n-k)/2)` | all four shipped codes PASS | exact | PASS |
| V2.3 | post-decoding FER and output SER, BPSK | direct Monte Carlo of 10 000 codewords through the real modem | measured/formula 0.998, 0.999, 0.981 at 4.5, 4.8, 5.1 dB (5679, 2266, 510 observed failures) | 5e-2 relative | PASS |
| V2.3 | the same, 16QAM **with** a bit interleaver | the same | measured/formula 1.000, 1.007, 1.000 | 5e-2 relative | PASS |
| V2.3 | the same, 16QAM **without** a bit interleaver | the same | measured/formula 1.006, 1.019, 1.038 — the real system is **worse** than the model | none; this measures the cost of the assumption | reported |
| V2.5 | ladder sanity | — | thresholds strictly increasing, efficiencies strictly increasing, no dominated MODCOD, span 13.391 dB | monotone | PASS |

### The measured MODCOD table

Target post-decoding BER 1e-6. Measured, not taken from any standard. `sigma`
is the Monte Carlo 1-sigma uncertainty of the threshold, from propagating one
binomial standard error on the measured channel BER.

| idx | MODCOD | mod bits | code rate | eta, bit/symbol | threshold, dB | sigma, dB |
|---|---|---|---|---|---|---|
| 0 | BPSK RS(255,127) | 1 | 0.49804 | 0.4980 | 3.141 | 0.045 |
| 1 | BPSK RS(255,191) | 1 | 0.74902 | 0.7490 | 4.583 | 0.031 |
| 2 | QPSK RS(255,191) | 2 | 0.74902 | 1.4980 | 7.602 | 0.024 |
| 3 | QPSK RS(255,223) | 2 | 0.87451 | 1.7490 | 8.810 | 0.032 |
| 4 | 8PSK RS(255,191) | 3 | 0.74902 | 2.2471 | 12.378 | 0.030 |
| 5 | 8PSK RS(255,223) | 3 | 0.87451 | 2.6235 | 13.677 | 0.025 |
| 6 | 16QAM RS(255,223) | 4 | 0.87451 | 3.4980 | 15.451 | 0.023 |
| 7 | 16QAM RS(255,239) | 4 | 0.93725 | 3.7490 | 16.532 | 0.022 |

Ladder steps, dB: 1.442, 3.019, 1.209, 3.568, 1.299, 1.774, 1.080. The two
3-dB-plus steps are the constellation changes BPSK to QPSK and QPSK to 8PSK; the
granularity of this ladder, not the 0.02 to 0.05 dB threshold uncertainty, is
what stops any policy reaching the clairvoyant bound.

**These are not DVB-S2 thresholds.** DVB-S2 (ETSI EN 302 307) uses LDPC plus BCH
and its thresholds are several dB better at comparable spectral efficiency. It is
named in the README as the reference for standardised numbers; nothing here
implements it.

## V3 — the three non-learned policies (`validate_policies.py`)

Lognormal channel, slot 1 ms, `tau_c` = 10 ms, `sigma_I^2` = 0.5, SNR at mean
irradiance 14 dB. 20 000 slots per episode, 5 seeds, paired across policies.

| # | Check | Result | Verdict |
|---|---|---|---|
| V3.1 | the clairvoyant bound dominates both causal policies at every delay | 14 of 14 comparisons PASS | PASS |
| V3.2 | the clairvoyant bound is independent of the feedback delay | spread 0.000e+00 bit/symbol across 7 delays, exactly | PASS |
| V3.6 | the best fixed margin depends on the delay | 0 dB at `tau` = 0, 1 dB at 2 ms, 2 dB at 5 to 40 ms; the shipped 3 dB default is **not** optimal anywhere | reported |

### Goodput and the two kinds of mis-selection against delay

Fixed margin 3 dB (the shipped default) and hysteresis +2/-0.5 dB, 5 seeds.
`unavoid` is the share of slots where the channel supported no MODCOD at all.

| tau, ms | policy | goodput | sem | efficiency | outage | unavoid | conservative | wasted | lost | switches/slot |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | fixed margin 3 dB | 1.7078 | 0.0111 | 0.714 | 0.00028 | 0.00028 | 0.912 | 0.685 | 0.000 | 0.384 |
| 0 | hysteresis +2/-0.5 | 2.0754 | 0.0138 | 0.868 | 0.00470 | 0.00028 | 0.541 | 0.309 | 0.008 | 0.217 |
| 0 | clairvoyant (upper bound, unachievable) | 2.3923 | 0.0152 | 1.000 | 0.00028 | 0.00028 | 0.000 | 0.000 | 0.000 | 0.432 |
| 5 | fixed margin 3 dB | 1.5898 | 0.0099 | 0.665 | 0.0492 | 0.00028 | 0.736 | 0.714 | 0.089 | 0.384 |
| 5 | hysteresis +2/-0.5 | 1.6369 | 0.0100 | 0.684 | 0.1735 | 0.00028 | 0.503 | 0.422 | 0.333 | 0.217 |
| 20 | fixed margin 3 dB | 1.4056 | 0.0068 | 0.588 | 0.1224 | 0.00028 | 0.671 | 0.777 | 0.210 | 0.383 |
| 20 | hysteresis +2/-0.5 | 1.4123 | 0.0057 | 0.590 | 0.2499 | 0.00028 | 0.506 | 0.521 | 0.459 | 0.217 |
| 40 | fixed margin 3 dB | 1.3674 | 0.0050 | 0.572 | 0.1371 | 0.00028 | 0.657 | 0.792 | 0.233 | 0.383 |
| 40 | hysteresis +2/-0.5 | 1.3641 | 0.0044 | 0.570 | 0.2658 | 0.00028 | 0.503 | 0.543 | 0.485 | 0.217 |

Two things in that table are the product's whole argument. First, at zero delay
hysteresis is far better than a 3 dB margin (2.075 against 1.708) and by 40 ms
they are indistinguishable (1.364 against 1.367) — the ranking of two sensible
rules inverts across the delay range, so a policy chosen at one delay is the
wrong policy at another. Second, the two mis-selection kinds move in opposite
directions: hysteresis always has roughly half the conservative share and two to
four times the outage, at the same goodput. A single "mis-selection rate" would
report those two policies as similar and would be useless.

### The hysteresis dead band, `tau` = 10 ms

| up, dB | down, dB | dead band, dB | goodput | outage | conservative | switches/slot |
|---|---|---|---|---|---|---|
| 0.5 | 0.5 | 0.0 | 1.4224 | 0.3050 | 0.417 | 0.436 |
| 2.0 | 0.5 | 1.5 | 1.5045 | 0.2201 | 0.504 | 0.217 |
| 4.0 | 0.5 | 3.5 | 1.5455 | 0.1297 | 0.617 | 0.103 |
| 6.0 | 0.5 | 5.5 | 1.5152 | 0.0681 | 0.714 | 0.054 |
| 6.0 | 4.0 | 2.0 | 1.2631 | 0.0220 | 0.869 | 0.159 |

Widening the dead band from 0 to 5.5 dB cuts the switch rate eightfold and the
outage by a factor of 4.5, and goodput peaks in the middle at a 3.5 dB band.

### `tau/tau_c` is the governing ratio, not `tau`

At a fixed `tau` = 10 ms, sweeping `tau_c`:

| tau_c, ms | tau/tau_c | fixed margin 3 dB | hysteresis +2/-0.5 | clairvoyant (unachievable) |
|---|---|---|---|---|
| 2.5 | 4.00 | 1.3641 | 1.4037 | 2.3854 |
| 10.0 | 1.00 | 1.4875 | 1.5045 | 2.3923 |
| 40.0 | 0.25 | 1.6793 | 1.7954 | 2.4111 |

## V4 — learned predictor against every baseline, lognormal (`validate_predictor.py`)

Protocol: predictors fitted on seeds (101, 102, 103); every policy's free
parameter — fixed margin, hysteresis dead band, confidence gate — tuned on seeds
(201 … 205); **every number below measured on seeds (301 … 310)**, which no
fitting or tuning touched. 20 000 slots per scored episode, 4 lags, 60 trees per
quantile model.

### V4.1 calibration of the stated intervals (nominal coverage 0.80)

| tau, ms | predictor | coverage | below q10 | above q90 | MAE, dB | RMSE, dB | mean spread, dB |
|---|---|---|---|---|---|---|---|
| 1 | learned quantile GBR | 0.7963 | 0.0981 | 0.1056 | 0.949 | 1.186 | 3.031 |
| 1 | analytic AR(1) MMSE (not learned) | 0.7937 | 0.0983 | 0.1080 | 0.945 | 1.182 | 2.996 |
| 5 | learned quantile GBR | 0.7887 | 0.0985 | 0.1128 | 1.762 | 2.205 | 5.548 |
| 5 | analytic AR(1) MMSE (not learned) | 0.7918 | 0.0963 | 0.1119 | 1.757 | 2.198 | 5.588 |
| 20 | learned quantile GBR | 0.7891 | 0.0889 | 0.1220 | 2.204 | 2.765 | 6.872 |
| 20 | analytic AR(1) MMSE (not learned) | 0.7953 | 0.0867 | 0.1180 | 2.198 | 2.759 | 6.952 |

Both predictors under-cover slightly, by 0.4 to 2.0 percentage points. No
quantile crossing was observed at any delay (`crossing_fraction` 0.0000
throughout), so the non-crossing repair never fired.

### V4.2 held-out goodput, bit/symbol

| tau, ms | variant | policy | goodput | sem | efficiency | outage | conservative |
|---|---|---|---|---|---|---|---|
| 1 | default | fixed margin 3 dB | 1.6943 | 0.0062 | 0.713 | 0.0013 | 0.852 |
| 1 | tuned | fixed margin 1 dB | 1.9640 | 0.0073 | 0.826 | 0.0645 | 0.452 |
| 1 | tuned | hysteresis +1/-0.5 dB | 1.9396 | 0.0070 | 0.816 | 0.0981 | 0.397 |
| 1 | predictive | analytic AR(1) MMSE (not learned), gate 0.25 | 1.9759 | 0.0075 | 0.831 | 0.0816 | 0.400 |
| 1 | predictive | learned quantile GBR, gate 0.25 | 1.9737 | 0.0074 | 0.830 | 0.0812 | 0.402 |
| 1 | bound | clairvoyant (upper bound, unachievable) | 2.3777 | 0.0083 | 1.000 | 0.0002 | 0.000 |
| 10 | default | fixed margin 3 dB | 1.4775 | 0.0046 | 0.621 | 0.0906 | 0.692 |
| 10 | tuned | fixed margin 2 dB | 1.5060 | 0.0041 | 0.633 | 0.1562 | 0.588 |
| 10 | tuned | hysteresis +4/-0.5 dB | 1.5409 | 0.0033 | 0.648 | 0.1276 | 0.616 |
| 10 | predictive | analytic AR(1) MMSE (not learned), gate 0.25 | 1.6636 | 0.0027 | 0.700 | 0.0876 | 0.579 |
| 10 | predictive | learned quantile GBR, gate 0.25 | 1.6618 | 0.0025 | 0.699 | 0.0913 | 0.572 |
| 20 | tuned | fixed margin 2.5 dB | 1.4134 | 0.0032 | 0.595 | 0.1533 | 0.621 |
| 20 | tuned | hysteresis +5/-0.5 dB | 1.4787 | 0.0034 | 0.622 | 0.1157 | 0.649 |
| 20 | predictive | analytic AR(1) MMSE (not learned), gate 0.25 | 1.6485 | 0.0018 | 0.693 | 0.0578 | 0.606 |
| 20 | predictive | learned quantile GBR, gate 0.25 | 1.6478 | 0.0019 | 0.693 | 0.0593 | 0.605 |

The full five-delay table, including the shipped-default hysteresis rows, is in
`validate_predictor.txt`.

### V4.3 what the confidence gate is worth

Tuned gate (0.25 at every delay) against `gate_k = 0`, which selects on the
point estimate alone. Same predictor, same test seeds.

| tau, ms | predictor | goodput, tuned gate | goodput, gate 0 | delta | outage, tuned | outage, gate 0 |
|---|---|---|---|---|---|---|
| 1 | learned quantile GBR | 1.9737 | 1.8200 | +0.1537 | 0.0812 | 0.2136 |
| 5 | learned quantile GBR | 1.7260 | 1.5571 | +0.1689 | 0.1108 | 0.3012 |
| 10 | learned quantile GBR | 1.6618 | 1.4792 | +0.1826 | 0.0913 | 0.3137 |
| 20 | learned quantile GBR | 1.6478 | 1.3636 | +0.2842 | 0.0593 | 0.3914 |
| 20 | analytic AR(1) MMSE (not learned) | 1.6485 | 1.3789 | +0.2696 | 0.0578 | 0.3799 |

The gate is worth +0.15 to +0.28 bit/symbol. The choice of predictor is worth
0.002 to 0.005 bit/symbol (next table). **The confidence output is between 30 and
140 times more valuable here than the regression it gates.**

### V4.4a learned against analytic (both are predictive; one is not learned)

"separated" means the gap exceeds twice the combined standard error of the two
means over 10 test seeds.

| tau, ms | learned | analytic (not learned) | gap | gap/sem | separated | winner |
|---|---|---|---|---|---|---|
| 1 | 1.9737 | 1.9759 | -0.0022 | -0.21 | NO | indistinguishable |
| 2 | 1.8598 | 1.8647 | -0.0050 | -0.55 | NO | indistinguishable |
| 5 | 1.7260 | 1.7290 | -0.0030 | -0.50 | NO | indistinguishable |
| 10 | 1.6618 | 1.6636 | -0.0018 | -0.48 | NO | indistinguishable |
| 20 | 1.6478 | 1.6485 | -0.0007 | -0.27 | NO | indistinguishable |

**The learned predictor does not beat the analytic AR(1) MMSE predictor at any
delay, and is nominally behind it at all five.** This is the expected result and
it is published as the result: the lognormal channel of `channel.py` is an AR(1)
process in log-amplitude by construction, so its minimum-mean-square-error
predictor is exactly linear in the last report, and three estimated parameters
already attain it. Gradient boosting on four lags has nothing left to find. This
is not a defect in the learned model; it is a property of the channel, and anyone
publishing an ML win on a synthetic AR(1) fading channel without this comparison
has published an artefact.

### V4.4b best predictive policy against the best **tuned** baseline

| tau, ms | best predictive | goodput | best tuned baseline | goodput | gap | gap, % | gap/sem | separated |
|---|---|---|---|---|---|---|---|---|
| 1 | analytic AR(1) MMSE, gate 0.25 | 1.9759 | fixed margin 1 dB | 1.9640 | 0.0119 | +0.61% | 1.14 | **NO** |
| 2 | analytic AR(1) MMSE, gate 0.25 | 1.8647 | fixed margin 1.5 dB | 1.8314 | 0.0333 | +1.82% | 3.73 | yes |
| 5 | analytic AR(1) MMSE, gate 0.25 | 1.7290 | hysteresis +3/-0.5 dB | 1.6448 | 0.0842 | +5.12% | 11.95 | yes |
| 10 | analytic AR(1) MMSE, gate 0.25 | 1.6636 | hysteresis +4/-0.5 dB | 1.5409 | 0.1227 | +7.96% | 28.73 | yes |
| 20 | analytic AR(1) MMSE, gate 0.25 | 1.6485 | hysteresis +5/-0.5 dB | 1.4787 | 0.1698 | +11.48% | 43.73 | yes |

At `tau` = 1 ms a tuned fixed margin is statistically indistinguishable from the
best predictor. **Prediction only earns its complexity at `tau/tau_c` of roughly
0.2 and above.** Below that, tune the margin and ship nothing else.

## V5 — the same protocol on the gamma-gamma channel (`validate_predictor_gammagamma.py`)

The gamma-gamma channel is built by pushing two independent Gauss-Markov drivers
through a Gaussian copula, so the dB series is a nonlinear function of two latent
AR(1) states: the conditional mean is not linear in the last report and the
conditional variance is not constant. This is where a learned predictor has
something structural to find.

| tau, ms | learned | analytic (not learned) | gap | gap/sem | separated |
|---|---|---|---|---|---|
| 2 | 1.7926 | 1.8006 | -0.0080 | -0.82 | NO |
| 5 | 1.6428 | 1.6420 | +0.0008 | +0.09 | NO |
| 10 | 1.5828 | 1.5773 | +0.0055 | +0.66 | NO |
| 20 | 1.5681 | 1.5699 | -0.0018 | -0.25 | NO |

**Still indistinguishable on goodput.** The learned predictor is nominally ahead
at 5 and 10 ms and behind at 2 and 20 ms, and no gap reaches two combined
standard errors.

Where the learned predictor *is* measurably better is calibration:

| tau, ms | predictor | coverage (nominal 0.80) | below q10 | above q90 |
|---|---|---|---|---|
| 2 | learned quantile GBR | 0.7981 | 0.0931 | 0.1087 |
| 2 | analytic AR(1) MMSE (not learned) | 0.8206 | 0.0901 | 0.0893 |
| 10 | learned quantile GBR | 0.7994 | 0.0821 | 0.1185 |
| 10 | analytic AR(1) MMSE (not learned) | 0.8217 | 0.0822 | 0.0961 |
| 20 | learned quantile GBR | 0.7944 | 0.0831 | 0.1224 |
| 20 | analytic AR(1) MMSE (not learned) | 0.8229 | 0.0801 | 0.0970 |

The learned intervals land within 0.6 percentage points of nominal 80% at every
delay; the analytic Gaussian intervals over-cover by 2.1 to 2.3 points because
the gamma-gamma dB marginal is not Gaussian and a symmetric Gaussian interval is
the wrong shape for it. **That is the only measured advantage of the learned
model anywhere in this package, and it is an advantage in the stated uncertainty,
not in the throughput.** Against the tuned baselines the predictive policies win
by +1.67% at 2 ms rising to +14.25% at 20 ms, all separated.

## Checks that are absent, and what that costs

| Not checked | Consequence |
|---|---|
| Any comparison against measured turbulence data | The channel is a model with a free correlation-time knob. Nothing here is evidence about a real link. |
| Any comparison against a real modem or decoder | The MODCOD thresholds describe this package's AWGN Monte Carlo plus an exact RS accounting, not hardware. |
| Reed-Solomon decoder miscorrection | Neglected. It makes the post-decoding rates, and therefore the thresholds, mildly optimistic. `reedsolo` or `galois` with a real decoder would settle it. |
| Feedback report quantisation and measurement noise | Not modelled; the report is the exact SNR, only stale. Every goodput number here is optimistic for a real terminal. |
| Interleaver latency and memory | Not modelled. The i.i.d.-bit assumption of `coding.py` assumes an interleaver deep enough to decorrelate a codeword and charges nothing for it. P041 CodedFade is the product that sizes it. |
| Muting | A causal policy always transmits, flooring at MODCOD 0. A real terminal would mute below threshold, which would move outage out of the goodput accounting and into a separate availability figure. |
| Any delay longer than 40 ms, or `sigma_I^2` above 1.5 | Outside the swept range. |
