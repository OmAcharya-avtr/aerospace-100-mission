# Validation evidence — codedfade 0.1.0

**Validation level 3, hardware-pending.** Everything below was produced by running
the scripts in this directory in the build session of 2026-10-06. Each script's raw
stdout is committed beside it as `<script>_output.txt`, so a reader can check that
the numbers in this file and in `README.md` came from a run rather than from a
draft. Re-running a script overwrites its output file, which is the point: the
committed output matches a fresh run.

**Environment.** Python 3.13.16, numpy and scipy from the build container, 2 CPU
cores and 7.8 GiB RAM shared with four sibling processes. Every Monte Carlo run and
model fit is sized to finish in under three minutes.

**Nothing here is a hardware measurement.** The device backend raises
`NotImplementedError`. The timings in `validate_decode_shortcut_output.txt` and in
`benchmark/benchmark_results.json` are wall-clock figures on a shared cloud core,
reported as compute-budget notes, and are never compared to a model output. What is
missing for validation level 4 is measured timing and resource use from a Jetson
Orin Nano.

## Test suite

Counts read from the junit XML `testsuite` element, not from pytest's stdout line:

```
python3 -m pytest tests/ -q --tb=no -p no:cacheprovider --junit-xml=/tmp/p041.xml
tests=315  failures=0  errors=0  skipped=0  time=146.669 s
```

No test is marked `xfail` and none is skipped.

## Scripts and what each one establishes

| Script | Raw output | Runtime |
|---|---|---|
| `validate_channel_statistics.py` | `validate_channel_statistics_output.txt` | ~30 s |
| `validate_crossing_convergence.py` | `validate_crossing_convergence_output.txt` | ~90 s |
| `validate_rs_known_answers.py` | `validate_rs_known_answers_output.txt` | ~70 s |
| `validate_interleaver_cost.py` | `validate_interleaver_cost_output.txt` | ~5 s |
| `validate_cross_check_x1.py` | `validate_cross_check_x1_output.txt` | ~10 s |
| `validate_decode_shortcut.py` | `validate_decode_shortcut_output.txt` | ~10 s |
| `train_model.py` | `train_model_output.txt` | ~30 s |
| `validate_ai_vs_baseline.py` | `validate_ai_vs_baseline_output.txt` | ~35 s |
| `validate_depth_sweep.py` | `validate_depth_sweep_output.txt` | ~46 s |
| `worked_example.py` | `worked_example_output.txt` | ~25 s |
| `examples/*.py` (4 figures) | `example_*_output.txt` | ~4 min total |

---

## 1. Channel model

`validate_channel_statistics.py`. 400 000-sample paths, `fs = 1e6` Hz,
`tau = 2e-4` s (`Lc = 200` samples). A different seed per row, so the five lognormal
rows are independent checks rather than one latent Gaussian seen through five
transforms.

### 1.1 Lognormal marginal

| Target SI | mean `I` | sample SI | relative error | KS statistic | KS p |
|---|---|---|---|---|---|
| 0.1 | 1.000532 | 0.103234 | 0.032343 | 0.052340 | 0.124684 |
| 0.3 | 0.985502 | 0.305166 | 0.017220 | 0.022647 | 0.954568 |
| 0.6 | 1.000394 | 0.575837 | 0.040272 | 0.028597 | 0.797098 |
| 1.0 | 0.976463 | 0.954666 | 0.045334 | 0.036902 | 0.492271 |
| 1.5 | 1.028944 | 1.248874 | **0.167418** | 0.030689 | 0.721994 |

The KS test compares `(ln I + sigma^2/2)/sigma` against the standard normal on
samples thinned to one per four correlation times, so it checks the distribution
shape and the model's claimed parameters at once. All five pass at `p > 0.01`.

**The `SI = 1.5` row fails the 10 % tolerance the test suite applies at `SI = 0.6`,
at 16.7 %.** The sample scintillation index requires the fourth moment of a
lognormal and is badly behaved in strong turbulence. This is reported rather than
fixed by lengthening the record until it passed.

### 1.2 Autocorrelation of the latent Gaussian, `Lc = 20` samples

| Kernel | Model | max \|ACF − model\|, lags 0–40 | ACF at lag `Lc` |
|---|---|---|---|
| `exp` | `exp(-k/Lc)` | **0.002240** | 0.365679 (1/e = 0.367879) |
| `gauss` | `exp(-(k/Lc)^2)` | **0.003695** | 0.364550 |

### 1.3 Measured 1/e correlation time of log-irradiance

| Marginal | Kernel | Target | Measured | Relative error |
|---|---|---|---|---|
| lognormal | `exp` | 2.000000e-04 s | 1.910364e-04 s | 0.044818 |
| lognormal | `gauss` | 2.000000e-04 s | 1.942576e-04 s | 0.028712 |
| gammagamma | `exp` | 2.000000e-04 s | 1.938845e-04 s | 0.030577 |
| gammagamma | `gauss` | 2.000000e-04 s | 2.010274e-04 s | 0.005137 |

These are measurements, not checks against a reference. The gamma-gamma rows carry
the extra caveat that the correlation is imposed through a Gaussian copula, so the
specified `tau` governs the latent Gaussian rather than the log-irradiance. **At
this configuration the gamma-gamma deviation is the same order as the lognormal
rows' own estimator bias, so nothing larger than estimator error was detected.**
That is reported; it is not asserted as agreement, and no claim is made that the
copula reproduces a physical two-scale temporal spectrum.

### 1.4 Gamma-gamma parameter relations

Equation (6) applied to the plane-wave equations (7)–(9) must equal
`exp(sigma_x^2 + sigma_y^2) - 1` identically:

| `sigma_R^2` | alpha | beta | SI from (6) | `exp(sx2+sy2)-1` | relative difference |
|---|---|---|---|---|---|
| 0.05 | 41.773624 | 39.336094 | 0.049969059 | 0.049969059 | 0.000e+00 |
| 0.30 | 8.431713 | 6.922053 | 0.280199300 | 0.280199300 | 1.981e-16 |
| 1.00 | 4.393859 | 2.563632 | 0.706438496 | 0.706438496 | 1.572e-16 |
| 4.00 | 4.340663 | 1.308803 | 1.170459847 | 1.170459847 | 0.000e+00 |
| 16.00 | 6.763441 | 1.056484 | 1.234338709 | 1.234338709 | 0.000e+00 |

Inversion round trip, worst relative error **3.701e-16** over targets 0.1, 0.5, 0.8,
1.0, 1.2.

**The plane-wave scintillation index is not monotone in the Rytov variance.** It
peaks at **SI = 1.2432** at `sigma_R^2 = 10.3218` (located by golden-section search
at import, `GAMMA_GAMMA_SI_PEAK`) and falls again. The inversion is restricted to the
increasing branch and refuses a higher target rather than returning the wrong branch.
A link needing `SI > 1.24` cannot use the plane-wave gamma-gamma path here.

### 1.5 Gamma-gamma sample marginal

| Target SI | mean `I` | sample SI | relative error |
|---|---|---|---|
| 0.3 | 0.985003 | 0.284680 | 0.051065 |
| 0.8 | 0.973286 | 0.758841 | 0.051448 |
| 1.1 | 0.969861 | 1.046897 | 0.048276 |

---

## 2. Level-crossing rate: which kernel converges

`validate_crossing_convergence.py`. `SI = 0.6`, `tau = 2e-4` s, amplitude threshold
0.6, standard level `u = -1.147441698282337`. Every row of sections 2.1 and 2.2 uses
a **4-second record**, so the number of crossings stays comparable across sample
rates and the convergence statement is about the kernel rather than about how many
crossings happened to be observed.

### 2.1 `exp` kernel: the rate must NOT converge

| `fs` (Hz) | `Lc` | sample LCR (s^-1) | equation (14) (s^-1) | relative difference | crossings | 1/sqrt(N) |
|---|---|---|---|---|---|---|
| 1.0e5 | 20 | 2603.500 | 2570.000 | 0.013035 | 10414 | 0.009799 |
| 2.0e5 | 40 | 3720.750 | 3659.630 | 0.016701 | 14883 | 0.008197 |
| 5.0e5 | 100 | 5831.500 | 5810.379 | 0.003635 | 23326 | 0.006548 |
| 1.0e6 | 200 | 8234.750 | 8228.472 | **0.000763** | 32939 | 0.005510 |

The sample rate rises by 10x and both columns rise by 3.2x, close to `sqrt(10)`.
**That is the Ornstein-Uhlenbeck process having no finite mean-square derivative: its
continuous-time level-crossing rate does not exist.** Equation (14), the exact
bivariate-normal expression for the sampled path, tracks the measurement to within
1.7 % at every rate and to 0.08 % at the highest. This is a property of the model,
asserted in `tests/test_fade.py::TestAnalyticCrossingRates::test_markov_rate_grows_with_sample_rate`
so that it cannot be silently "fixed" later.

### 2.2 `gauss` kernel: the rate MUST converge to Rice (1945)

Rice's equation (12) gives **582.645238 s^-1** in continuous time.

| `fs` (Hz) | `Lc` | sample LCR (s^-1) | relative difference | crossings | deviation in SE |
|---|---|---|---|---|---|
| 1.0e5 | 20 | 594.000 | 0.019488 | 2376 | 0.95 |
| 2.0e5 | 40 | 595.750 | 0.022492 | 2383 | 1.10 |
| 5.0e5 | 100 | 589.500 | 0.011765 | 2358 | 0.57 |
| 1.0e6 | 200 | 596.250 | 0.023350 | 2385 | 1.14 |

Every row is within 1.2 Poisson standard errors of Rice's value and **the offset does
not grow with `fs`**, which is the claim being made. All four rows share one seed, so
their offsets are correlated.

### 2.3 Convergence of the sample statistics with record length

Exact results at `fs = 1e6` Hz: LCR **8228.471844 s^-1**, MFD
**1.526401990e-05 s**.

| samples | complete fades | sample LCR | relative | sample MFD (s) | relative |
|---|---|---|---|---|---|
| 100 000 | 744 | 7440.000 | −0.095822 | 1.484946e-05 | −0.027159 |
| 400 000 | 3177 | 7942.500 | −0.034754 | 1.553982e-05 | +0.018068 |
| 1 000 000 | 8457 | 8457.000 | +0.027773 | 1.502625e-05 | −0.015577 |
| 4 000 000 | 33 313 | 8328.250 | +0.012126 | 1.521934e-05 | −0.002927 |

Both converge. A 100 000-sample record disagrees with the exact result by 9.6 % on
the crossing rate, which is why the suite's agreement tests use 2 000 000 samples.

---

## 3. Codes

`validate_rs_known_answers.py`.

### 3.1 RS(15,11) over GF(2^4), `t = 2`, `d = n-k+1 = 5` (MDS)

Generator polynomial `[12, 1, 3, 15, 1]` (ascending degree), degree 4 = 2t. Message
`[0..10]`, codeword `[0,1,2,3,4,5,6,7,8,9,10,10,12,0,13]`, syndromes of the codeword
all zero.

Four buckets, because "the decoded message equals the transmitted message" is not the
same as "decoding succeeded":

| errors | patterns | recovered | miscorrected | failed, message intact | failed, message wrong |
|---|---|---|---|---|---|
| 1 | 225 | **225** | 0 | 0 | 0 |
| 2 | **23 625** | **23 625** | 0 | 0 | 0 |

Both rows are **exhaustive**: every single-symbol and every double-symbol error
pattern over the whole codeword and all 15 non-zero error values. This is the code's
guarantee, so it is enumerated and not sampled.

At `t+1 = 3` errors, 20 000 random patterns:

| Outcome | Count | Fraction |
|---|---|---|
| **recovered (must be 0)** | **0** | 0.000000 |
| miscorrected | 5807 | 0.290350 |
| failed, message intact | 137 | 0.006850 (expected `C(4,3)/C(15,3) = 0.008791`) |
| failed, message wrong | 14 056 | 0.702800 |

The "failed, message intact" column is not a recovery: those are patterns where all
three errors fell in the four parity symbols, so the untouched message symbols came
back unchanged while the decoder correctly reported failure. Counting that as a
recovery was a genuine defect in an earlier version of the test, caught by this run.

### 3.2 RS(31,21) over GF(2^5), `t = 5`, rate 0.677419

| errors | trials | recovered | miscorrected | failed, intact | failed, wrong |
|---|---|---|---|---|---|
| 5 | 4000 | **4000** | 0 | 0 | 0 |
| 6 | 4000 | **0** | 5 | 1 | 3994 |
| 8 | 2000 | **0** | 8 | 0 | 1992 |

### 3.3 Convolutional code, rate 1/2, K = 3, `(0o7, 0o5)`

| Check | Result |
|---|---|
| `encode([1,0,1,1])` | **111000010111** |
| hand trace (6 steps, in the test docstring) | 11 10 00 01 01 11 — **match** |
| Viterbi on a clean input | `[1,0,1,1]`, path metric 0 |
| minimum terminated codeword weight, L = 12 | **5 bits**, enumerated over all 4095 non-zero messages |
| single-bit errors in a 60-bit message's codeword | **124 of 124** recovered |

**No published free distance is cited.** The weight above is what the implementation
produces, enumerated.

Burst sensitivity — the reason interleaving exists:

| burst length (bits) | recovered |
|---|---|
| 1 | 12/12 |
| 2 | 12/12 |
| 3 | **1/11** |
| 4, 6, 8, 12 | **0/11** |

---

## 4. Interleavers

`validate_interleaver_cost.py`. Span 31 symbols, `Rs = 1e6` Hz, 5 bits per symbol,
`t = 5`.

### 4.1 Block interleaver

| depth | measured spacing | equation (24) latency (symbols) | latency (ms) | memory (B) | longest survivable burst `t*D` |
|---|---|---|---|---|---|
| 1 | 1 | 62 | 0.0620 | 38.8 | 5 |
| 16 | 16 | 992 | 0.9920 | 620.0 | 80 |
| 64 | 64 | 3968 | 3.9680 | 2480.0 | 320 |
| 256 | 256 | 15 872 | 15.8720 | 9920.0 | 1280 |
| 1024 | 1024 | 63 488 | 63.4880 | 39 680.0 | 5120 |
| 4096 | 4096 | 253 952 | 253.9520 | 158 720.0 | 20 480 |

The measured spacing equals the depth **exactly at all 13 depths from 1 to 4096**,
which is the property equation (23) relies on. Full table in the raw output.

### 4.2 Burst dispersion, equation (22) — `ceil(L/D)`

| burst L | D=1 | D=8 | D=64 | D=512 |
|---|---|---|---|---|
| 20 | 20 | 3 | 1 | 1 |
| 200 | 200 | 25 | 4 | 1 |
| 2000 | 2000 | 250 | 32 | 4 |

### 4.3 Convolutional interleaver, measured against `M*B*(B-1)`

| branches B | increment M | `M*B*(B-1)` | **measured delay** | memory (symbols) | block equivalent (symbols) |
|---|---|---|---|---|---|
| 2 | 1 | 2 | **2** | 2 | 124 |
| 4 | 1 | 12 | **12** | 12 | 248 |
| 4 | 2 | 24 | **24** | 24 | 496 |
| 8 | 1 | 56 | **56** | 56 | 496 |
| 8 | 4 | 224 | **224** | 224 | 1984 |
| 16 | 1 | 240 | **240** | 240 | 992 |
| 31 | 1 | 930 | **930** | 930 | 1922 |

Measured by pushing a ramp through the real implementation and finding the shift at
which input and output agree, not by asserting the formula. At B = 31 the
convolutional construction stores 930 symbols against the block interleaver's 1922
for a comparable depth, at the cost of a start-up transient.

---

## 5. Coded link and the depth sweep

`validate_depth_sweep.py`. RS(31,21) over GF(2^5), OOK with
`p_b = Q(sqrt(gbar) I)`, mean electrical SNR 14.000 dB, `SI = 0.6`, `Rs = 1e6` Hz,
channel seed 7, 2048 codewords per point. A frame is one codeword; FER counts both
decoder failures and miscorrections. Every depth in a sweep sees the **identical**
channel record, so the pre-decoding symbol error rate is the same at every depth —
visible in the `raw SER` column, which does not move.

### 5.1 `tau = 2e-5` s, `Lc = 20` symbols

Sample MFD **4.90 symbols** (exact 4.89), worst fade 76 symbols over 10 350 fades,
uncoded BER 1.559065e-02.

| depth | D/Lc | raw SER | post BER | FER | FER SE (binomial) | latency (ms) |
|---|---|---|---|---|---|---|
| 1 | 0.050 | 0.065319 | 2.1382e-02 | 0.124512 | 0.005159 | 0.062 |
| 4 | 0.200 | 0.065319 | 1.2684e-02 | 0.094727 | 0.004576 | 0.248 |
| 16 | 0.800 | 0.065319 | 4.0016e-03 | 0.035889 | 0.002906 | 0.992 |
| 64 | 3.200 | 0.065319 | 1.8276e-03 | 0.017334 | 0.002039 | 3.968 |
| 256 | 12.800 | 0.065319 | 1.4021e-03 | 0.014160 | 0.001846 | 15.872 |
| 1024 | 51.200 | 0.065319 | 1.6741e-03 | 0.016113 | 0.001967 | 63.488 |
| 4096 | 204.800 | 0.065319 | 1.6113e-03 | 0.015381 | 0.001923 | 253.952 |

The curve flattens at `D/Lc ~ 3` and the last three rows are indistinguishable. The
extra 250 ms of latency at `D = 4096` buys nothing.

### 5.2 `tau = 2e-4` s, `Lc = 200` symbols

Sample MFD **14.37 symbols** (exact 15.26), worst fade 580 symbols over 3583 fades,
uncoded BER 1.583933e-02. Depth sized to the mean fade: 15 symbols; to three times
the mean: 44.

| depth | D/Lc | post BER | FER | FER SE | failures | miscorrections | worst errors in a codeword | latency (ms) |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.005 | 2.6216e-02 | 0.133057 | 0.005307 | 545 | 0 | 30 | 0.062 |
| 4 | 0.020 | 2.3640e-02 | 0.131836 | 0.005286 | 539 | 1 | 30 | 0.248 |
| 16 | 0.080 | 1.8113e-02 | 0.119385 | 0.005066 | 489 | 0 | 19 | 0.992 |
| 64 | 0.320 | 6.8545e-03 | 0.055664 | 0.003582 | 227 | 1 | 10 | 3.968 |
| 256 | 1.280 | 1.5532e-03 | 0.014893 | 0.001893 | 61 | 0 | 9 | 15.872 |
| 1024 | 5.120 | 1.2719e-03 | 0.012207 | 0.001716 | 50 | 0 | 9 | 63.488 |
| 4096 | 20.480 | 8.0683e-04 | **0.007568** | 0.001354 | 31 | 0 | 7 | 253.952 |

**A depth of 15 or 44 — what the mean fade duration asks for — is not enough.** At
`D = 16` the FER is 0.119 against 0.133 with no interleaving at all. Almost all the
gain arrives between `D/Lc = 0.08` and `D/Lc = 1.28`. Sizing to the mean fade
duration instead of the correlation length is the mistake this product exists to make
visible.

### 5.3 `tau = 1e-3` s, `Lc = 1000` symbols

Sample MFD **33.56 symbols** (exact 34.09), worst fade 1548 symbols.

| depth | D/Lc | FER | FER SE | latency (ms) |
|---|---|---|---|---|
| 1 | 0.001 | 0.120117 | 0.005080 | 0.062 |
| 64 | 0.064 | 0.121582 | 0.005106 | 3.968 |
| 256 | 0.256 | 0.053223 | 0.003507 | 15.872 |
| 1024 | 1.024 | 0.001953 | 0.000690 | 63.488 |
| 4096 | 4.096 | 0.011719 | 0.001682 | 253.952 |

`D = 64` gives **no improvement at all** over no interleaving, within the error bar,
for 3.968 ms of latency. The `D = 1024` and `D = 4096` rows appear to contradict each
other, which is the subject of the next section.

### 5.4 How big is the real error bar?

The FER standard error above is a **binomial** standard error, which assumes codeword
failures are independent. They are not: they are driven by a handful of deep fades.
Measured over five channel seeds at `tau = 1e-3` s, 2048 codewords
(63 488 symbols = 63 correlation times):

| depth | seed 7 | seed 11 | seed 23 | seed 37 | seed 53 | mean | across-seed sd | binomial SE |
|---|---|---|---|---|---|---|---|---|
| 64 | 0.137207 | 0.139160 | 0.224609 | 0.112793 | 0.032227 | 0.129199 | **0.068810** | 0.007073 |
| 256 | 0.082520 | 0.051758 | 0.145508 | 0.132812 | 0.004883 | 0.083496 | **0.058021** | 0.005561 |
| 1024 | 0.003906 | 0.013184 | 0.067383 | 0.022461 | 0.000488 | 0.021484 | **0.027048** | 0.002640 |
| 4096 | 0.011719 | 0.004395 | 0.023682 | 0.005127 | 0.003906 | 0.009766 | **0.008399** | 0.001436 |

The across-seed standard deviation is **5.8x to 10.4x the binomial standard error**
at every depth. The mean FER is monotone decreasing in depth, but **above `D/Lc ~ 1`
the ordering between individual depths is not resolvable at this sample size and
reverses between channel seeds**. This is published instead of presenting one seed's
ordering as a result, and it is the reason the README's Limitations section says to
treat the binomial SE as a lower bound.

### 5.5 The decoding shortcut

`validate_decode_shortcut.py`. The sweep skips the Berlekamp-Massey decoder for
codewords whose symbol error count is at most `t`, which is sound because
Reed-Solomon codes are maximum distance separable — the property enumerated
exhaustively in section 3.1. Both paths on the identical realisation, 512 codewords,
`tau = 2e-4` s, seed 9:

| depth | FER (shortcut) | FER (exact) | failures | miscorrections | identical? | speed-up |
|---|---|---|---|---|---|---|
| 1 | 0.091797 | 0.091797 | 47 / 47 | 0 / 0 | **yes** | 4.98x |
| 8 | 0.109375 | 0.109375 | 56 / 56 | 0 / 0 | **yes** | 5.45x |
| 64 | 0.035156 | 0.035156 | 18 / 18 | 0 / 0 | **yes** | 24.44x |
| 512 | 0.000000 | 0.000000 | 0 / 0 | 0 / 0 | **yes** | 664.56x |

**0 depths with a mismatch.** The wall-clock figures are compute-budget notes on a
shared core, not hardware evidence.

---

## 6. Cross-check X1 (binding)

`validate_cross_check_x1.py`. Compared against **P049 LinkOutage** at a tolerance of
2 % relative difference on each of the two quantities.

### Configuration, frozen

| Key | Value |
|---|---|
| distribution | lognormal |
| kernel | `exp` (Gauss-Markov / AR(1), filtered Gaussian) |
| scintillation index | 0.6 |
| correlation time | 2.0e-4 s |
| sample rate | 1.0e6 Hz |
| sample count | 2 000 000 |
| threshold amplitude | 0.6 |
| seed | 41 |
| random source | `numpy.random.default_rng(41).standard_normal(2000000)` in one call |
| normalisation | `E[I] = 1`, `a = sqrt(I)` |
| `rho = exp(-1/(fs*tau))` | 0.995012479192682 |
| `sigma_lnI^2 = ln(1+SI)` | 0.470003629245736 |
| `Lc = tau*fs` | 200.000000 samples |

Series fingerprint, so a reproduction can be checked before the statistics are
compared:

```
amplitude[0:5]   [0.582925424356399 0.589514039810461 0.590583695809805
                  0.602026167532692 0.57644569736477 ]
amplitude[-5:]   [0.601259433341748 0.629124368200058 0.674698323595734
                  0.66970961177348  0.666937533144782]
mean irradiance          0.998907972681559
sample scintillation idx 0.609378559983851
```

### The two X1 quantities

| Quantity | Value |
|---|---|
| **LEVEL_CROSSING_RATE_HZ** | **8191.000000000000000** |
| **MEAN_FADE_DURATION_S** | **1.549737516786717e-05** |

Supporting counts: outage fraction 0.126940500000000, down-crossings 16 382,
complete fades 16 382, censored fades 1, record duration 2.000000000 s.

Context only, not part of the comparison: the exact sampled-Gauss-Markov analytic
values are LCR 8228.471844 s^-1 (sample/analytic − 1 = **−0.004554**) and MFD
1.526401990e-05 s (**+0.015288**). **X1 compares the two sample statistics between
products. It does not compare either to the analytic value, and no quantity in this
check is a wall-clock measurement.**

Additional sample statistics for diagnosis: median fade duration 3.000000000e-06 s,
maximum 6.900000000e-04 s, duration p50/p90/p99 = 3.0 / 37.0 / 199.4 samples. The
median being 3 samples against a mean of 15.5 is the fact that defeats the
exponential closure in section 7.

---

## 7. AI against the analytic baseline

`validate_ai_vs_baseline.py`. Lognormal, `SI = 0.6`, `tau = 2e-4` s, `fs = 1e6` Hz,
`exp` kernel, threshold amplitude 0.6, `t_target = 14` samples. 200 000 samples per
path. Train seeds 0–23 (**39 687** events, exceedance frequency 0.199486); held-out
seeds 100–111 (**19 900** events, 0.198693). **Split by path seed**, so no event from
a training realisation appears in the test set.

Phase order: the analytic level-crossing result was implemented and validated in
section 2 before any model was fitted.

### 7.1 The analytic closure, measured

| Quantity | Value |
|---|---|
| equation (16) prediction `P(T > MFD)` | **0.399641** |
| observed frequency | **0.198693** |
| **ratio predicted / observed** | **2.0113** |

**The memoryless closure of the level-crossing result over-predicts the exceedance at
`t = MFD` by a factor of two.** Level-crossing theory fixes the mean and nothing
else, and the fade-duration distribution on this channel is far from exponential —
the median fade is 3 samples against a mean of 15.5. This is the most useful finding
in the product. Equation (16) was **not** retuned and **not** removed; it is labelled
an approximation in `codedfade.fade` and its error is published here.

### 7.2 Benchmark on identical held-out events

| Predictor | Brier | log loss | ROC AUC | ECE (10 bins) |
|---|---|---|---|---|
| `analytic_configured` (exact MFD, given the true channel config) | 0.199594 | 0.591088 | 0.5000 | 0.200947 |
| `analytic_observed` (MFD from a 32-sample window) | 0.198676 | **2.568769** | 0.4983 | 0.198648 |
| `empirical_constant` (measured training frequency) | 0.159215 | 0.498588 | 0.5000 | **0.000793** |
| **learned** | **0.154194** | **0.484009** | **0.6175** | 0.004311 |

| | |
|---|---|
| best baseline Brier | 0.159215 |
| learned Brier | 0.154194 |
| **relative improvement** | **+3.1536 %** |

### 7.3 Verdict, stated honestly

**The learned model wins, but only just, and only against the right baseline.**

- Against the **exponential closure** the Brier improvement is 23 %, and that number
  would be misleading: almost all of it is a calibration correction, because the
  closure predicts 0.40 where the frequency is 0.20.
- Against the **empirical constant** — the same state-blind structure with the
  measured frequency instead of the exponential assumption — the improvement is
  **3.15 %**. That is the number to quote.
- The real evidence that state carries information is the **ROC AUC of 0.6175**
  against 0.5000 for every state-blind predictor. `log_depth_at_crossing` carries
  40.2 % of the model's Gini importance: a crossing that plunges deep is the start of
  a longer fade. That effect is real and it is the only thing the model adds.
- `analytic_observed` has a **log loss of 2.5688**, five times the empirical
  constant's, because estimating the mean fade duration from a 32-sample window
  sometimes yields a near-0 or near-1 exceedance. Its Brier looks fine; its log loss
  shows it is confidently wrong on a minority of events. Reported rather than dropped.

### 7.4 Reliability table, learned model

| bin | mean predicted | observed | count | predicted − observed |
|---|---|---|---|---|
| 0 | 0.093189 | 0.121711 | 304 | −0.028522 |
| 1 | 0.150960 | 0.153634 | 11 957 | −0.002674 |
| 2 | 0.235470 | 0.237998 | 5395 | −0.002529 |
| 3 | 0.339812 | 0.331541 | 1674 | +0.008271 |
| 4 | 0.434964 | 0.412998 | 477 | +0.021966 |
| 5 | 0.541788 | 0.482353 | 85 | +0.059435 |
| 6 | 0.640428 | 0.375000 | 8 | **+0.265428** |
| 7–9 | — | — | 0 | — |

Calibrated to about 0.02 up to a predicted 0.45 and not beyond. **There are 93
held-out events in total above a predicted 0.45 and the model should not be used
there** — stated in `MODEL_CARD.md` as a failure case.

### 7.5 Uncertainty output

| Quantity | Value |
|---|---|
| mean per-tree sigma | 0.066564 |
| median | 0.060739 |
| maximum | 0.205373 |
| events with \|p − y\| > 0.5 | 3959 of 19 900 |
| mean sigma on those | 0.073262 |
| mean sigma on the rest | 0.064900 |
| **ratio** | **1.1289** |

The ensemble is more uncertain where it is wrong, **by 13 %**. That is enough to rank
cases and not enough to gate a decision on a threshold. No recalibration layer was
added to improve the appearance of this number.

### 7.6 What the probability is for

Depth required to cover a fade of a given held-out quantile at `Rs = 1` Mbaud, with
the latency that depth costs at span 31:

| quantile | duration (symbols) | depth | latency (ms) |
|---|---|---|---|
| 0.50 | 3.0 | 3 | 0.186 |
| 0.90 | 36.0 | 36 | 2.232 |
| 0.99 | 205.0 | 205 | 12.710 |
| 1.00 | 640.0 | 640 | 39.680 |

### 7.7 Model artefact

`train_model.py` regenerates `fade_exceedance_model.joblib` deterministically:
channel seeds 0–23, forest `random_state=0`, 160 trees, `max_depth=8`,
`min_samples_leaf=20`. The archive holds the fitted forest, the feature names, the
training configuration and both baseline constants. **No `.pt`, `.pth`, `.ckpt` or
`.onnx` file is produced or committed**, and no cache directory is tracked.

---

## 8. Benchmark harness

`benchmark/run_benchmark.py` writes `benchmark/benchmark_results.json`, which
contains its own measurement method and environment. Method: `time.perf_counter_ns`
with a measured resolution of **119 ns**, one discarded warm-up execution per
stage, **7 timed repeats**, **median** reported (the mean is contaminated by
scheduling on a shared host), throughput computed as symbols divided by the median,
and `tracemalloc` peak captured in a separate untimed execution so that tracing never
perturbs a timing. Resident-set size is deliberately not reported: on a shared
container it is dominated by the interpreter and the imported libraries.

Environment of the committed run: **2 CPU cores**, 1-minute load average
**1.06**, Python 3.13.16, numpy 2.5.3.

| Stage | median (s) | throughput (symbols/s) | tracemalloc peak (B) |
|---|---|---|---|
| channel path generation, 400 000 samples | 0.078852 | 5 072 784 | 9 601 424 |
| fade statistics, 400 000 samples | 0.000791 | 505 524 754 | 1 652 244 |
| RS(31,21) encode, one codeword | 0.000201 | 154 052 | 2 682 |
| RS(31,21) decode, clean codeword | 0.002355 | 13 161 | 2 921 |
| **RS(31,21) decode, `t` errors** | 0.007436 | 4 169 | 3 680 |
| block interleave, 7936 symbols | 0.000036 | 1 770 391 233 | 1 016 096 |
| block de-interleave | 0.000044 | 1 434 335 675 | 1 016 096 |
| HAL encode block, depth 256 | 0.047793 | 166 050 | 190 928 |
| HAL decode block, depth 256 | 0.778071 | 10 200 | 127 264 |
| full depth-sweep point, 256 codewords | 0.002043 | 3 883 546 | 445 244 |

Interleaver cost model for the benchmarked configuration (depth 256, span 31,
5 bits/symbol, 1 Mbaud): latency 15 872 symbols = **15.872 ms**, memory
15 872 symbols = **9920 B**.

**Every figure in this section describes this package on a shared cloud container.**
They move with the load average — an earlier run on the same host gave 14.9 ms for
the `t`-error decode against 7.44 ms here — which is exactly why the harness records
its method, its repeat count and the load average alongside the numbers. They are
**not** Jetson Orin Nano measurements and cannot support a validation-level-4 claim.
The output file says so in its own `environment.note` and `level_4_gap` fields.

---
## 9. What is still missing for validation level 4

One thing, stated so it cannot be mistaken for anything else:

> **Measured timing and resource use from a Jetson Orin Nano**, produced by running
> `benchmark/run_benchmark.py` on that board and keeping its raw output file.

No simulated backend, no extrapolation from the figures in section 8, no vendor
datasheet and no workstation run substitutes for it. Until that run exists this
product is **Level 3, hardware-pending**, and it is never labelled Level 4.
