# Validation evidence

**Package:** `aperturediv` 0.1.0 · **Validation level 2** (research grade) ·
Status **TESTING** · Apache-2.0 · © 2026 OPTIMA Organisation

Research-grade. Not flight-qualified, not certified, not approved for
operational aerospace use.

Every number in this file, in `README.md` and in `MODEL_CARD.md` was produced
by running one of the scripts in this directory in the build container
(Python 3.13.16, NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1, 2 CPU cores).
The raw stdout of each run is committed beside the script. No figure is
quoted from a reference, and no page number is cited anywhere in this
repository.

## How to reproduce everything

```bash
pip install -e ".[dev]"
python validation/validate_channel_stats.py          #  ~2 s
python validation/validate_aperture_averaging.py     #  ~2 s
python validation/validate_correlation.py            #  ~5 s
python validation/validate_combining.py              # ~130 s
python validation/validate_learned_combiner.py       # ~185 s
python validation/validate_cross_check_x2.py         #  ~20 s
```

Each exits 0 on success and 1 if any check fails, printing which. The times
above are the measured wall clock in the build container **under four-way
agent contention on two cores**, which is the state the container was in
tonight; on an idle pair of cores the two slow scripts take about 45 s and
30 s. The learned-combiner script was deliberately resized down from 250000
rows to 120000 after the 250000-row version measured 440 s under that
contention; the conclusions were identical at both sizes and the committed
numbers are from the committed size. Nothing in this repository approaches
the 600 s per-script limit.

## Scripts and their raw output

| script | raw output | what it establishes |
|---|---|---|
| `validate_channel_stats.py` | `channel_stats_output.txt` | both irradiance densities normalise and reproduce their closed-form moments and scintillation index; samplers match their marginals; the Rytov mapping is internally consistent |
| `validate_aperture_averaging.py` | `aperture_averaging_output.txt` | the kernel normalisation identity; the two derived closed-form limits of `A(D)` and where each stops being usable |
| `validate_correlation.py` | `correlation_output.txt` | the exact lognormal log-to-irradiance correlation relation; both copula samplers reproduce their inputs |
| `validate_combining.py` | `combining_output.txt` | combiner ordering (including the one that is false); outage against the closed form; diversity order, independent and correlated; the correlation gap in dB |
| `validate_learned_combiner.py` | `learned_combiner_output.txt` | MRC-with-the-truth optimality numerically; the learned combiner against four non-learned references; the EGC crossover; uncertainty calibration |
| `validate_cross_check_x2.py` | `cross_check_x2_output.txt` | **binding cross-check X2** against P010 BERBench |

Example scripts also save their stdout here
(`example_*_output.txt`), because the figures in `screenshots/` are produced
by the same runs and the numbers printed alongside them should be auditable.

---

## 1. Channel statistics

Reference for the models: Andrews & Phillips, *Laser Beam Propagation through
Random Media* (SPIE Press, 2005); Al-Habash, Andrews & Phillips, *Optical
Engineering* 40(8), 2001 for gamma-gamma. The checks below compare the code
against the **models' own closed forms** and against Monte Carlo, which is a
check rather than a quotation.

| check | reference | result | tolerance | verdict |
|---|---|---|---|---|
| lognormal density integrates to 1, si = 0.1…0.9 | definition | worst deviation 0 to 1e-9 | 1e-8 abs | pass |
| lognormal scintillation index recomputed from quadrature moments | `si = exp(s^2) - 1` | agrees at every si on the grid | 1e-7 abs | pass |
| gamma-gamma density integrates to 1, Rytov 0.2…5.0 | definition | worst deviation below 1e-7 | 1e-7 abs | pass |
| gamma-gamma si from quadrature vs `1/a + 1/b + 1/(ab)` | Al-Habash et al. 2001 | agrees at every Rytov variance | 1e-6 rel | pass |
| lognormal CDF closed form vs quadrature of its own pdf | definition | worst absolute difference **2.220e-16** | 1e-10 abs | pass |
| gamma-gamma CDF (quadrature) vs 400000 samples | — | worst **\|z\| = 1.298** over six points | 4.0 | pass |
| lognormal sampler, standardised `ln I` vs N(0,1) | Kolmogorov-Smirnov | statistic **0.001047**, p = **0.772** | p > 0.001 | pass |
| gamma-gamma sampler vs its own quadrature CDF, 60 quantiles | KS statistic | **0.001019** against a 1 % critical value of **0.002577** | below critical | pass |
| Rytov → (alpha, beta) internal consistency, Rytov 0.2…5.0 | model's own closed form | worst **\|z\| = 0.536** | 4.0 | pass |

Representative mapped shapes: Rytov variance 1.0 gives
**alpha = 4.393859, beta = 2.563632, si = 0.706438**; the sampled
scintillation index at 400000 draws was **0.710416** (z = +0.536).

## 2. Aperture averaging

`A(D)` is computed from its defining integral over the circular-aperture
overlap kernel. The two closed forms used as targets are **derived in the
source docstrings**, not quoted.

| check | reference | result | tolerance | verdict |
|---|---|---|---|---|
| kernel normalisation `int_0^1 u W(u) du = pi/16` | derived | difference **below 1e-11** | 1e-11 abs | pass |
| `A(0) = 1` exactly | definition | exactly 1.0 | exact | pass |
| `A` strictly decreasing over `D/rho_c` in [1e-3, 300] | definition | 119 of 119 forward differences negative | all | pass |
| small-D limit `A ~ 1 - D^2/(4 rho_c^2)` | derived | residual/(D/rho_c)^4 constant to **under 5 %** over `D/rho_c <= 0.2` | 5 % spread | pass |
| large-D limit `A ~ 4(rho_c/D)^2 - (8/sqrt pi)(rho_c/D)^3` | derived | worst relative difference **3.740e-05** for `D/rho_c >= 20` | 1e-4 rel | pass |
| leading coefficient `A D^2/rho_c^2 -> 4` | derived | **3.99549** at `D/rho_c = 1000` | 2e-2 abs | pass |
| exponential vs Gaussian covariance cross exactly once | structural | one sign change, between `D/rho_c` **2.39 and 2.49** | exactly one | pass |

**A result that contradicted the first expectation written for it.** The
check on the exponential covariance was first written asserting
`A_exponential >= A_gaussian` at every `D`, on the reasoning that the heavier
covariance tail must average less. That check **failed**, and it failed
because the reasoning was wrong: `exp(-x) < exp(-x^2)` for `x < 1`, so at
equal `rho_c` the exponential field is already decorrelated across a small
aperture and averages *less*, while its tail makes it average less across a
large one too. The two curves cross once. The check now records the crossover
instead of the false inequality, and the episode is kept in the raw output
rather than deleted.

Where each closed form is usable, measured in
`example_aperture_averaging_output.txt`: the small-D form is within 1 % up to
`D/rho_c = 0.637`; the two-term large-D form is within 0.01 % from
`D/rho_c = 14.87` upwards. Between those the integral has to be done.

Worked case, 1.55 um over 2 km (`rho_c = sqrt(lambda L) = 0.055678 m`,
point si 0.6):

| `D` (m) | `D/rho_c` | `A(D)` | `si(D)` |
|---|---|---|---|
| 0.02 | 0.359 | 0.968590 | 0.581154 |
| 0.05 | 0.898 | 0.828002 | 0.496801 |
| 0.10 | 1.796 | 0.532486 | 0.319492 |
| 0.20 | 3.592 | 0.214569 | 0.128741 |
| 0.40 | 7.184 | 0.065387 | 0.039232 |

## 3. Inter-aperture correlation

| check | reference | result | tolerance | verdict |
|---|---|---|---|---|
| `corr(I_j,I_k) = (exp(R s^2) - 1)/(exp(s^2) - 1)` vs 400000 samples, 32 (si, R) points | derived, exact for lognormal | worst absolute deviation **1.41e-03** | `4/sqrt(n)` = 6.32e-03 | pass |
| copula sampler: column means | `E[I] = 1` | worst \|mean - 1\| **below 0.01** | 0.01 | pass |
| copula sampler: column scintillation index | requested `si` | worst deviation **below 0.02** | 0.02 | pass |
| copula sampler: log-domain correlation matrix | requested `R` | worst deviation **below 0.01** | 0.01 | pass |
| copula sampler: irradiance correlation matrix | derived relation | worst deviation **below 0.01** | 0.01 | pass |
| irradiance correlation never exceeds log correlation, si 0.05…2.0 | derived | strict on `0 < R < 1` at every si; endpoints exact to 0 | strict | pass |
| gamma-gamma copula: marginals | closed-form si | worst mean error **0.00113**, worst si error **0.00660** | 0.02 / 0.03 | pass |
| `nearest_psd` identity on a valid matrix | — | worst change **8.33e-16** | 1e-10 | pass |
| `nearest_psd` repairs an indefinite matrix | — | eigenvalues (-0.8, 1.9, 1.9) → (7.9e-13, 1.5, 1.5), unit diagonal kept | min eig >= -1e-12 | pass |

**The number this section exists for.** Quoting a log-domain correlation
overstates the correlation a combiner actually sees, by up to **0.0866** at
si = 0.9 on the tested grid (and by 0.135 at si = 2.0). A worked case: four
apertures at 0.05 m pitch with `rho_c = 0.10 m` have adjacent log correlation
**0.778801** and adjacent *irradiance* correlation **0.736686** at si = 0.6.
Any diversity figure that does not say which one it used is ambiguous.

For the correlated gamma-gamma sampler with only the large-scale factor
correlated, a large-scale copula correlation of **0.7788** between adjacent
apertures produces an irradiance correlation of only **0.2484**, because the
independent small-scale factor contributes variance but no covariance.

## 4. Combining, outage and diversity order

Configuration: lognormal si = 0.9 or gamma-gamma at Rytov variance 1.0
(alpha 4.393859, beta 2.563632); four apertures at 0.05 m pitch,
`rho_c = 0.10 m`, adjacent log correlation 0.778801; outage threshold 5 dB;
2000000 realisations; SNR grid step 0.25 dB.

| check | reference | result | tolerance | verdict |
|---|---|---|---|---|
| `gamma_MRC >= gamma_EGC` pointwise, L = 1…4 | Cauchy-Schwarz | true on all 2000000 realisations at every L | no exception | pass |
| `gamma_MRC >= gamma_SC` pointwise, L = 1…4 | Cauchy-Schwarz | true on all 2000000 realisations at every L | no exception | pass |
| five hand-calculated known answers for all three gains | hand arithmetic, shown in the script | exact to 1e-12 | 1e-12 | pass |
| `E[sum_k I_k] = L` | `E[I] = 1` | 1.00085, 2.00185, 3.00251, 4.00272 | 1 % rel | pass |
| `var(sum_k I_k) = si * sum_jk corr(I_j,I_k)` | derived | worst relative difference **9.89e-04** | 2 % rel | pass |
| L = 1 outage vs the lognormal CDF, 5…25 dB | closed form | worst **\|z\| = 1.652** | 4.0 | pass |
| lognormal slope grows as the window moves down, every L | structural | monotone at all four L | monotone | pass |

### The inequality that is false, measured rather than asserted

`gamma_EGC >= gamma_SC` is **not** true pointwise. With `L = 2` and
`I = (1, 0)`, EGC gives 0.5 and SC gives 1. Measured fraction of
realisations on which selection beats equal gain:

| L | P(SC > EGC) |
|---|---|
| 1 | 0.000000 (the three schemes are the same quantity) |
| 2 | 0.000929 |
| 3 | 0.002392 |
| 4 | 0.002302 |

Small, but not zero, and concentrated on exactly the deep-fade realisations
that diversity is bought for. Only the two true inequalities are tested.

### Diversity order, gamma-gamma, window `P_out` 1e-5 to 1e-3

| L | measured, independent | measured, correlated | asymptotic `L min(a,b)` | loss from correlation |
|---|---|---|---|---|
| 1 | 2.482 | 2.479 | 2.564 | 0.003 |
| 2 | 4.410 | 3.914 | 5.127 | 0.496 |
| 3 | 6.265 | 5.016 | 7.691 | 1.249 |
| 4 | **7.528** | **5.608** | **10.255** | **1.921** |

Fit quality: 10 to 34 points per fit, residual RMS 0.014 to 0.035 decades.
The SNR windows are reported in the raw output (for L = 4, 5.75-8.00 dB
independent and 7.50-11.00 dB correlated).

Two things are wrong with the number usually quoted, and both are measured
here. First, the **asymptotic** value is not reached in the outage range a
link is designed to: at L = 4 the measured slope is 7.528 against an
asymptote of 10.255, a shortfall of **2.726**. Second, the
**independent-aperture** idealisation overstates the correlated array by a
further **1.921** in slope. A quoted "diversity order 4" for this channel is
neither of the measured numbers.

### The correlation gap in dB, target outage 1e-3

Branch mean SNR needed, from matched seeded sample paths:

| L | scheme | independent (dB) | correlated (dB) | gap (dB) | gain over L = 1 (dB) |
|---|---|---|---|---|---|
| 2 | MRC | 11.932 | 13.328 | **1.396** | 8.747 |
| 3 | MRC | 8.022 | 9.798 | **1.777** | 12.657 |
| 4 | MRC | 5.563 | 7.371 | **1.808** | 15.116 |
| 2 | EGC | 12.216 | 13.574 | 1.357 | 8.463 |
| 3 | EGC | 8.389 | 10.152 | 1.763 | 12.290 |
| 4 | EGC | 5.982 | 7.784 | 1.802 | 14.697 |
| 2 | SC | 13.944 | 15.312 | 1.368 | 6.735 |
| 3 | SC | 11.303 | 13.015 | 1.712 | 9.376 |
| 4 | SC | 9.763 | 11.453 | 1.690 | 10.916 |

So on this geometry the independence assumption is worth **1.4 to 1.8 dB**,
nearly independently of the combining scheme and of the aperture count above
two. EGC costs 0.3 to 0.4 dB against MRC; SC costs 2.0 to 4.2 dB.

### The lognormal channel has no finite diversity order at all

Measured slope for independent lognormal branches in three successive outage
windows. A finite asymptotic diversity order would give the same slope in all
three:

| L | window 1e-2…1e-1 | window 1e-3…1e-2 | window 1e-5…1e-3 |
|---|---|---|---|
| 1 | 2.738 | 3.721 | 4.693 |
| 2 | 3.783 | 5.253 | 6.805 |
| 3 | 4.577 | 6.299 | 8.666 |
| 4 | 5.220 | 7.204 | 9.154 |

The slope increases monotonically at every L. The lognormal outage decays
faster than any power of the mean SNR, so "diversity order L" is not a
property of a lognormal channel; it is a property of the window. For
gamma-gamma it *is* a property of the channel, because the gamma-gamma
density has a power-law tail at small irradiance.

This is also why the diversity-order tables above use gamma-gamma, and why
`diversity_order` returns the window, the point count and the fit residual
alongside the slope instead of returning a bare number.

### Model divergence, measured

At matched scintillation index (Rytov 0.3, si 0.280199) the two channel
models agree within **2 %** on outage while the outage is above 0.4, and
diverge monotonically into the tail, reaching **3.675x** at outage 1.6e-3
with gamma-gamma the pessimistic one
(`tests/test_integration.py::test_gamma_gamma_and_lognormal_diverge_only_in_the_tail`).

## 5. Learned combiner under imperfect CSI

Full tables in `MODEL_CARD.md`. The checks, and the verdicts:

| check | reference | result | tolerance | verdict |
|---|---|---|---|---|
| penalty of MRC-with-the-truth is zero on every held-out row | Cauchy-Schwarz | max \|penalty\| **3.857e-15 dB** | 1e-9 | pass |
| no combiner has a negative penalty on any held-out row | Cauchy-Schwarz | 0 of 25000 rows below -1e-9 dB, for all five | none | pass |
| features reproducible from the estimate alone | by construction | exact array equality | exact | pass |
| uncertainty calibration, nominal 0.1 / 0.5 / 0.9 | — | **0.094360 / 0.465720 / 0.873240** | 0.05 abs | pass |
| `sigma_e` is a sufficient error axis | derived | four (rho_t, sigma_m) pairs at `sigma_e = 1.0` give 0.6089 / 0.6178 / 0.6227 / 0.6325 dB | Monte Carlo | pass, with a weak residual drift recorded |
| EGC/MRC-estimated crossover exists and is locatable | — | **`sigma_e` = 0.8320 = 3.613 dB**, bracketed to 0.0078 by bisection | found | pass |

### The published result, stated plainly

**The analytic baseline is unbeatable at zero estimation error, and that is
not a tolerance issue — it is Cauchy-Schwarz.** At `sigma_e = 0`,
MRC-with-the-estimate *is* MRC-with-the-truth and scores -0.0000 dB; the
learned combiner scores +0.0084 dB. The baseline wins and the result stands.
At `sigma_e = 0.25` the two are level to 0.0006 dB, below the resolution of a
20000-row evaluation, so the learned model earns nothing there either.

Pooled over `sigma_e` uniform on [0, 3] (25000 held-out rows), mean SNR
penalty against the unreachable bound:

| combiner | mean penalty dB | worst-row penalty dB | mean BER at branch Eb/N0 10 dB |
|---|---|---|---|
| `mrc_true` (bound) | 0.0000 | 0.0000 | 7.082324e-08 |
| `mrc_estimated` | 1.2278 | 14.6171 | 2.059059e-05 |
| `egc` (no CSI) | 0.4411 | 2.9260 | 1.281703e-07 |
| `shrinkage p=0.160` (analytic) | 0.3667 | 3.1880 | 1.239550e-07 |
| `learned` | 0.2948 | 2.6016 | 1.225447e-07 |

- Learning recovers **0.933 dB** of what naive MRC-on-the-estimate loses.
- Against the best **analytic** rule it gains **0.072 dB**. That is the
  honest size of the machine-learning contribution.
- In BER it is **4.4 %** better than equal gain, which needs no model at all
  (1.730x the bound against 1.810x).

### The crossover, which is the useful finding

Equal-gain combining overtakes MRC-with-the-estimate at
**`sigma_e` = 0.8320, i.e. 3.613 dB of irradiance estimation error.** Below
that, use the estimate. Above it, the correct engineering decision is to stop
estimating the channel and combine with equal gain — no model, no training,
no inference. The `egc` penalty is flat to within 0.006 dB across a 13 dB
range of estimation error, because equal gain never reads the estimate.

## 6. Cross-check X2 (binding)

Batch 05 specification, X2: P044 and **P010 BERBench** must agree on the
sample BER of BPSK over lognormal fading within 3 binomial standard errors.
**P010's value is not known to this repository and is not read, imported or
approximated anywhere.** The coordinating session holds it and performs the
comparison.

### Frozen configuration

| field | value |
|---|---|
| modulation | BPSK, coherent, ideal phase reference, hard decision |
| receive apertures | **1** (single aperture, no diversity) |
| `Eb/N0` | **10.0000 dB** (10.000000 linear) |
| `Eb/N0` reference | the **mean** received irradiance, `E[I] = 1` |
| fading law | `gamma = (Eb/N0) * I`, `I` lognormal, `E[I] = 1` |
| scintillation index | **0.300000**, defined as `var(I)/E[I]^2` |
| `sigma_log = sqrt(ln(1+si))` | 0.512215056854 |
| `E[ln I]` | -0.131182131824 |
| channel realisations | **20000000** |
| bits per realisation | **1** (so bits are independent and the binomial SE is exact) |
| total bits | 20000000 |
| RNG | `numpy.random.default_rng(44044)` (PCG64) |
| seed | **44044** |
| chunk size | 2000000 bits per chunk |
| draw order per chunk of m bits | `z = rng.standard_normal(m)`; `bits = rng.integers(0,2,m)`; `noise = rng.standard_normal(m)` |
| received sample | `r = sqrt(2 (Eb/N0) I) s_tx + noise`, `s_tx = 1 - 2 bits` |
| decision | `bit_hat = (r < 0)` |
| BER definition | observed bit errors / total bits — a **sample** statistic |

### Result

| quantity | value |
|---|---|
| **sample BER** | **6.039000000e-04** |
| bit errors observed | **12078** of 20000000 |
| **binomial standard error** | **5.493338260e-06** |
| SE as a fraction of the BER | 0.9096 % |
| 3 binomial standard errors | 1.648001478e-05 |
| X2 agreement interval | [5.874199852e-04, 6.203800148e-04] |

### Internal consistency of that number (not the comparison)

| check | result | tolerance | verdict |
|---|---|---|---|
| Gauss-Hermite quadrature of the same expectation, 300 nodes | 6.002225662e-04 | — | — |
| quadrature convergence, 300 vs 150 nodes | relative difference 3.613e-16 | 1e-6 | pass |
| `z = (sample BER - quadrature) / binomial SE` | **+0.6694** | \|z\| < 3 | pass |
| bit-for-bit reproducible under the same seed and chunk size | identical, 12078 errors | exact | pass |
| independent seed 44045 | BER 6.080000000e-04, z = -0.527 against seed 44044 | — | pass |
| AWGN BER at the same `Eb/N0`, no fading | 3.872108216e-06 | — | — |
| fading penalty | **155.962x** the AWGN BER | — | — |

Supplementary operating points, secondary and provided only so that a
disagreement can be attributed to a convention rather than to noise:

| `Eb/N0` dB | si | realisations | sample BER | binomial SE | z vs quadrature |
|---|---|---|---|---|---|
| 6.00 | 0.300 | 20000000 | 1.024845e-02 | 2.252046e-05 | +0.958 |
| 14.00 | 0.300 | 20000000 | 7.000000e-06 | 5.916059e-07 | -1.230 |
| 10.00 | 0.100 | 20000000 | 7.415000e-05 | 1.925416e-06 | -0.645 |
| 10.00 | 0.600 | 20000000 | 2.189850e-03 | 1.045240e-05 | +1.414 |

### Conventions a disagreement would most likely trace to

Written out in the script, because each is a factor-of-two-or-more
difference and a disagreement traceable to one of them is a **specification**
defect rather than a product defect in either repository:

- **C1** SNR is **linear** in irradiance. Not `gamma ∝ I^2`, which is the
  thermal-noise-limited intensity-modulation convention.
- **C2** `si` is defined on the irradiance, so `var(ln I) = ln(1 + si)`. Not
  `si` as `var(ln I)` directly, nor as the log-amplitude variance `si/4`.
- **C3** `Eb/N0` is referred to the **mean** irradiance, not the median. The
  two differ by `s^2/2` nepers, which at si = 0.3 is 0.57 dB.
- **C4** Coherent BPSK, `Q(sqrt(2 gamma))`. Not differential or non-coherent,
  which would give `0.5 exp(-gamma)`.
- **C5** Memoryless across bits: one irradiance draw per bit. Not block
  fading, which leaves the BER unchanged in expectation but inflates the true
  standard error above the binomial figure.

**No wall-clock measurement is compared to a model anywhere in this
product**, so X2 cannot repeat the Batch 04 cross-check defect.

## 7. Test suite

```
python3 -m pytest tests/ -q --tb=no -p no:cacheprovider --junit-xml=p044.xml
```

Counts are read from the junit XML `testsuite` element, not from pytest's
stdout line. See the README badge and the build report for the figures.

Coverage of the minimums for a medium-class product:

- unit tests for every public function, in `tests/test_channel.py`,
  `test_aperture.py`, `test_correlation.py`, `test_combining.py`,
  `test_estimation.py`, `test_ber.py`, `test_datasets.py`, `test_learned.py`;
- input-validation tests on every raising path;
- known-answer tests with the hand arithmetic written in the test comment
  (lognormal `s = sqrt(ln 2)` at si = 1; `A` kernel at `u = 0, 0.5, 1`;
  EGC of `(1,0)` and `(4,1)`; `Q(1) = 0.158655`; gamma-gamma
  `si = 1.25` at `a = b = 2`; `d = D/sqrt(n)`);
- edge cases: zero diameter, single aperture, zero estimation error, total
  fade on one branch, outage below the Monte Carlo floor, a diversity-order
  window with too few points;
- Hypothesis property tests for the algebraic identities: the
  lognormal `si <-> s` round trip, quantile inverting the CDF, irradiance
  correlation never exceeding log correlation, equal-area diameters
  preserving total area, `gamma_MRC >= gamma_EGC` and `>= gamma_SC`, and the
  non-negativity and minimiser property of the penalty metric;
- an integration test walking the whole pipeline from link geometry to a
  learned-combiner comparison, `tests/test_integration.py`;
- regression pins on the aperture-averaging quadrature, the Rytov mapping,
  the irradiance correlation matrix, the seeded sample BER (exactly 1180
  errors in 2000000 bits), the seeded outage (exactly 228 of 200000) and the
  seeded diversity order (7.630 over 10 fit points).

Nothing is marked `xfail`. Nothing is skipped.

## 8. What is not validated

Stated so no reader has to discover it:

- **No comparison against measured data.** Every number here is internal to
  the model or to the sampler. The models are cited to the literature; their
  agreement with an atmosphere is not established by this repository.
- **The irradiance correlation scale is a free input.** It is not derived
  from a `C_n^2` profile, a wind speed or a path geometry. Taking it as the
  Fresnel scale, as the CLI does by default, is a convenience, not a result.
- **Aperture averaging is applied by scaling the scintillation index** and
  keeping the distribution family. Aperture averaging does not strictly
  preserve the lognormal or gamma-gamma family, so `si(D) = A(D) si(0)` is an
  approximation whose error is not quantified here.
- **No co-phasing, no pointing, no misalignment, no detector model.** Perfect
  phase reference, perfect alignment, no thermal or shot-noise accounting, no
  background light. The SNR convention is stated and nothing beyond it is
  modelled.
- **The gamma-gamma copula correlates the two factors separately** with the
  small-scale factor independent by default. That is a modelling choice; the
  joint distribution it induces is not the only gamma-gamma field with those
  marginals.
- **The learned combiner is not validated outside its training geometry.**
  One `si`, one pitch, one correlation scale, `sigma_e` in [0, 3]. Outside
  that range the gradient-boosted trees extrapolate flat and do not signal
  it.

## Credits

This is under reserved rights obtained by OPTIMA Organisation.
