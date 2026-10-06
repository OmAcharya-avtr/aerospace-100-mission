# Validation evidence — softdecode

**Validation level 2 (research grade). Status: TESTING.**

Every number in this file, in `README.md` and in `MODEL_CARD.md` was produced by
a script in this directory, run on 2026-10-06 in the build container, with its
raw standard output committed beside it. The commands that reproduce each one
are at the end of this file. Nothing here is quoted from memory, and no
reference is cited that was not read.

Build host: Python 3.13.16, numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1,
pytest 9.1.1. Two CPU cores and 7.8 GiB of RAM **shared with four other build
agents and a concurrent release-gate run**. Runtimes quoted below are therefore
upper bounds and vary between runs; **no result in this repository is compared
against a wall-clock measurement.**

Test suite, counted from the junit XML rather than from pytest's stdout line:
**161 tests, 0 failures, 0 errors, 0 skipped.**

## Summary

| # | Check | Reference | Result | Tolerance / gate | Script |
|---|---|---|---|---|---|
| 1 | Quadrature moments against closed-form moments | `E[h]=1`; `E[h^2]=1+sigma_I^2` (lognormal), `(1+1/a)(1+1/b)` (gamma-gamma) | worst relative error **5.3e-13** over 7 models | < 1e-9 | `validate_quadrature.py` |
| 2 | **`p(y\|b=1)` by quadrature against Monte Carlo of the same integral** | 4 000 000 draws of `h`, MC standard error propagated | lognormal worst **\|z\| = 2.807**, worst relative difference 6.0e-03; gamma-gamma worst **\|z\| = 2.839** | \|z\| < 4 | `validate_quadrature.py` |
| 3 | The same comparison in LLR units | as above, `se(LLR) = se(p1)/p1` | lognormal worst **\|z\| = 2.962**; gamma-gamma worst **\|z\| = 2.142** | \|z\| < 4 | `validate_quadrature.py` |
| 4 | Node convergence, lognormal `sigma_I^2 = 0.3` | the 300-node rule | 80 nodes (the default): **1.748e-03** max LLR deviation; 160 nodes 1.07e-05 | reported | `validate_quadrature.py` |
| 5 | Node convergence, lognormal `sigma_I^2 = 1.0` | the 300-node rule | 80 nodes: **9.717e-02**. The default is **not** adequate for strong scintillation and the README says so | reported | `validate_quadrature.py` |
| 6 | Node convergence, gamma-gamma (4,2) and (2.2,1.3) | the 3201-point rule | the default adaptive grid (265 / 379 points): **5.9e-11** and **6.8e-12** | reported | `validate_quadrature.py` |
| 7 | **The tensor Gauss-Laguerre rule stalls** | the same reference | 1600 nodes: 7.38e-03 at (4,2) and **1.01e-01** at (2.2,1.3); 6400 nodes only reaches 8.13e-03 at (2.2,1.3). This is why it is **not** the default | reported | `validate_quadrature.py` |
| 8 | Posterior quadrature of `softdecode.csi` | Monte Carlo over the same posterior, 1 000 000 draws per point | worst **\|z\| = 1.982**, worst absolute LLR difference 1.086e-03 | \|z\| < 4 | `validate_quadrature.py` |
| 9 | Samplers, moments | closed form | gamma-gamma mean z **+0.142**; lognormal mean z **−3.137** (reported, see §1) | \|z\| < 4 | `validate_quadrature.py` |
| 10 | Lognormal sampler, Kolmogorov-Smirnov | standardised log-amplitude against `N(0,1)`, 5 draws of 400 000 | **5 of 5** draws with p > 0.01, smallest p **0.392653** | p > 0.01 | `validate_quadrature.py` |
| 11 | Gamma-gamma sampler, Kolmogorov-Smirnov | trapezoid CDF of its own density, 200 000 draws | statistic 0.001432, **p = 0.806452** | p > 0.01 | `validate_quadrature.py` |
| 12 | **AWGN known-answer limit, lognormal** | `L = (a^2 - 2ay)/2sigma^2` (Proakis & Salehi, ch. 4) | the deviation is **first order** in `sigma_I^2`: coefficient spread over five decades **1.000098** | spread < 1.05 | `validate_awgn_limit.py` |
| 13 | The same limit, gamma-gamma | `alpha = beta -> infinity` | `err/sigma_I^2` rises to **1689.3** at `A = 1e4`, approaching the lognormal coefficient **1718.9** on the same grid | reported | `validate_awgn_limit.py` |
| 14 | The same limit, 4-PPM bit LLRs | known-CSI PPM metric at `h = 1` | coefficient **1371.2**, constant over four decades | reported | `validate_awgn_limit.py` |
| 15 | One-node quadrature equals the known-CSI formula | algebraic identity | max difference **3.55e-15** | < 1e-12 | `validate_awgn_limit.py` |
| 16 | **Max-log LLR error, no CSI, lognormal** | exact marginal LLR, same samples | rmse **1.684** at 2 dB falling to **1.329** at 10 dB; one-signed (`L_maxlog >= L_exact`) at **every** sample, 5 of 5 points | one-signed required | `validate_maxlog_clipping.py` |
| 17 | **What max-log costs after decoding** | same channel realisations, 1200 blocks | decoded BER ratio **1.433** at 2 dB rising to **3.194** at 10 dB; GMI 0.3343 → 0.0422 at 2 dB | reported | `validate_maxlog_clipping.py` |
| 18 | Max-log, gamma-gamma (4,2) | as above | rmse **2.934**, GMI **−0.5949** at 2 dB: max-log LLRs there are confidently wrong | reported | `validate_maxlog_clipping.py` |
| 19 | Max-log with an estimated channel state | exact posterior-aware LLR | BER ratio **5.750** at 10 dB, rmse 1.243 | reported | `validate_maxlog_clipping.py` |
| 20 | **Clipping: the two costs disagree** | unclipped exact LLRs, same realisations | at 8 dB, `L_max = 5`: relative LLR rmse **0.8177**, 26.4 % of LLRs saturated, decoded BER **2.590e-02** against 2.608e-02 unclipped (ratio 0.993, within one binomial se of 6.5e-04) | reported | `validate_maxlog_clipping.py` |
| 21 | Where clipping does bite | as above | at 8 dB, `L_max = 1`: BER ratio **3.873**; `L_max = 2`: **2.028** | reported | `validate_maxlog_clipping.py` |
| 22 | 4-PPM max-log error bound | `0 <= logsumexp(a,b) - max(a,b) <= log 2`, so the bit LLR error lies in `[-log 2, +log 2]` | measured maximum **0.692962**, i.e. **0.999733** of the bound `log 2 = 0.693147` | <= log 2 | `validate_maxlog_clipping.py` |
| 23 | **Mismatched CSI, over-estimating** | plug-in LLR, 8 dB, 1200 blocks, pure bias | +2 dB bias: decoded BER **8.415e-02** against **1.917e-03** unbiased, a factor **43.9**; +4 dB: 4.851e-01, factor 253.1 | reported | `validate_csi_mismatch.py` |
| 24 | **Mismatched CSI, under-estimating** | the same | −2 dB bias: decoded BER **1.168e-02**, factor **6.10**; −4 dB: 7.630e-02, factor 39.8 | reported | `validate_csi_mismatch.py` |
| 25 | **The asymmetry** | the ratio of the two above at equal magnitude | **7.203** at 2 dB, 1.926 at 1 dB, 7.873 at 3 dB, 6.358 at 4 dB | reported | `validate_csi_mismatch.py` |
| 26 | **Decomposition of the asymmetry** | raw hard-decision BER is scale invariant, so its ratio isolates the threshold shift | at 2 dB: decoded ratio 7.203 = threshold **1.339** x over-confidence **5.381** | reported | `validate_csi_mismatch.py` |
| 27 | **The mechanism** | raw hard errors split by transmitted bit | missed-one share **0.1029** at −4 dB, 0.5055 at 0 dB, **0.9912** at +4 dB | reported | `validate_csi_mismatch.py` |
| 28 | GMI goes negative | plug-in LLRs, 8 dB | **−1.170** at +3 dB bias and **−7.616** at +4 dB: the demapper is then actively misleading the decoder | reported | `validate_csi_mismatch.py` |
| 29 | A known bias is fully correctable | posterior-aware LLR with the bias known | decoded BER **1.916667e-03 at every bias from −4 to +4 dB**, identical to the true-CSI result | identity expected | `validate_csi_mismatch.py` |
| 30 | Pure jitter still costs | plug-in LLR, no bias | 1 dB jitter: **1.200e-02** against 5.567e-03 posterior-aware; 3 dB jitter: **2.113e-01** against 1.598e-02 | reported | `validate_csi_mismatch.py` |
| 31 | A stale estimate | correlated lognormal estimate | `rho = 0.8`: plug-in **2.852e-02**, posterior-aware **9.633e-03**; `rho = 0.6`: 8.058e-02 against 1.577e-02 | reported | `validate_csi_mismatch.py` |
| 32 | **Learned corrector against the realistic baselines** | report split, 1500 blocks, 8 dB, disjoint from train and tune | bias +2 dB: learned **6.533e-03** against max-log-with-estimated-CSI 2.199e-02 (**3.365x better**) and plug-in 1.855e-01 (**28.388x better**) | reported | `validate_corrector.py` |
| 33 | **Learned corrector against plain scalar rescaling** | `alpha` tuned on the disjoint tuning split | bias +2 dB: rescaling **1.540e-01** (alpha = 0.20), learned **23.576x better**; bias −2 dB: the tuner chose **alpha = 1.00**, so rescaling recovers **nothing** | reported | `validate_corrector.py` |
| 34 | **Learned corrector against the analytic posterior-aware LLR — the baseline wins** | the same split | learned loses by **1.121x** (bias +2 dB), **1.078x** (bias −2 dB) and **1.060x** (stale `rho=0.8`) | reported, not tuned away | `validate_corrector.py` |
| 35 | Distance to the unreachable bound | true-CSI LLR, same split | learned is **2.356x**, **2.264x** and **3.139x** worse than true CSI in the three regimes | reported | `validate_corrector.py` |
| 36 | Uncertainty output | report split | dispersion correlates **0.711** with `\|learned − clipped analytic\|`, **−0.241** with the error indicator; error rate **decreases** monotonically across its quintiles, so as a risk score it points the wrong way | reported, not gated | `validate_corrector.py` |
| 37 | **CROSS-CHECK X3** | soft-decision BER <= hard-decision BER at every Eb/N0, same realisations, extended Hamming (8,4), both decoders exhaustive ML | **PASS at all 7 points**; gain rises from **1.409x** at 0 dB to **25.569x** at 12 dB; 160 000 information bits per point | strict ordering | `validate_cross_check_x3.py` |
| 38 | X3 pairing evidence | per-block agreement counts | at 12 dB, 693 blocks soft-right-hard-wrong against **5** soft-wrong-hard-right | reported | `validate_cross_check_x3.py` |
| 39 | **X3 comparand**, uncoded hard-decision BPSK over lognormal fading | sample BER with binomial se, against `E_h[Q(h sqrt(2 Eb/N0))]` by quadrature | `sigma_I^2 = 0.30`, Eb/N0 = 10 dB, 2 000 000 bits, seed 20261361: **7.549000e-03 ± 6.120e-05**, quadrature reference **7.518905e-03** | reported for the coordinating session | `validate_cross_check_x3.py` |
| 40 | Comparand grid against the analytic reference | 15 cells | worst **\|z\| = 2.215** over the 14 cells with a nonzero error count | \|z\| < 4 | `validate_cross_check_x3.py` |
| 41 | Sum-product decoder known answer | brute-force bit a-posteriori LLRs on a cycle-free parity-check matrix | max deviation **< 1e-09** over 20 blocks | < 1e-09 | `tests/test_ldpc.py` |
| 42 | LDPC code parameters reported, not assumed | GF(2) elimination with column pivoting | `n = 96`, `m = 48`, **rank 46**, so `k = 50` and rate **50/96 = 0.520833**, not 1/2; **26** length-4 cycles | reported | `validate_maxlog_clipping.py`, `python -m softdecode ldpc` |

## 1. Fading quadrature and samplers (`validate_quadrature.py`)

Raw output: [`quadrature_output.txt`](quadrature_output.txt). Runtime about 10 s.

Every exact LLR in this package replaces an integral over the fading density by
a quadrature sum. That substitution is the thing most likely to be silently
wrong, so it is measured against a Monte Carlo estimate of the same integral
with the Monte Carlo standard error propagated, rather than against a second
quadrature rule.

The quadrature moments are exact to **5.3e-13** or better for seven models
spanning scintillation indices from 0.021 to 6.0. The likelihood comparison
against 4 000 000 Monte Carlo draws agrees to within **2.8 Monte Carlo standard
errors** at the worst of eight `y` values for the lognormal model and **2.8**
for gamma-gamma; in LLR units the worst discrepancies are **2.96** and **2.14**
standard errors.

**Two quadrature rules were tried for the gamma-gamma model and the obvious one
lost.** The tensor product of two generalised Gauss-Laguerre rules is exact for
the moments at very few nodes, which is why it was the first choice, but on the
peaked likelihood integrands this package actually evaluates it stalls: 1600
nodes give a worst LLR deviation of **1.01e-01** at `(alpha, beta) = (2.2, 1.3)`
and 6400 nodes only reach **8.13e-03**. A trapezoid rule on the density in
`log h`, with its range derived from the small-`h` power law, the large-`h`
Bessel decay and the exact log-moments, reaches **6.8e-12** at 379 points. The
Laguerre rule is kept in the API as `laguerre_quadrature` and is **not** the
default; both are in the table so the choice can be checked.

The lognormal sample mean came out at **z = −3.137** against its own standard
error on the draw reported, with the sample variance 0.2967 against 0.3000.
Five independent Kolmogorov-Smirnov tests of the standardised log-amplitude all
pass with p from 0.393 to 0.973, so the sampler is not biased; the mean z score
is a property of that one draw and of the slow convergence of the sample mean of
a lognormal. It is reported rather than re-drawn.

The lognormal Kolmogorov-Smirnov test standardises the sample and calls
`kstest(z, "norm")`, because `kstest(x, "norm", args=(loc, scale))` raises
`TypeError` on the installed SciPy. The two forms are equivalent.

## 2. The AWGN limit as a known answer (`validate_awgn_limit.py`)

Raw output: [`awgn_limit_output.txt`](awgn_limit_output.txt). Runtime about 2 s.

A tolerance alone would prove nothing here, because any tolerance can be met by
choosing a small enough scintillation index. The statement checked is stronger:
the deviation from the textbook affine LLR is **first order** in the
scintillation index, with the ratio `max|L_exact − L_linear| / sigma_I^2`
constant to **1.000098** across five decades from 1e-10 to 1e-6.

The coefficient itself (1718.9 on the validation grid, 96.0 on the narrower grid
the unit test uses) is a property of the grid and of `a/sigma`, not of the
package: the leading term is the curvature of `L` in `h` times `Var[h]/2`. The
gamma-gamma model approaches the **same** coefficient as `alpha = beta` grows —
1689.3 at `A = 1e4` against the lognormal's 1718.9 — which is a consistency
check between two independently implemented densities. The 4-PPM bit LLRs are
first order with coefficient 1371.2.

Evaluating the gamma-gamma density at `alpha = beta = 1e4` requires the
log-space form `log K_v(z) = log kve(v, z) − z`; the direct product overflows
above a few hundred. That is implemented as `GammaGammaFading.log_pdf` and
pinned by a test.

## 3. Max-log and clipping (`validate_maxlog_clipping.py`)

Raw output: [`maxlog_clipping_output.txt`](maxlog_clipping_output.txt). Runtime
about 31 s.

**Max-log is not cheap on this channel.** Because `logsumexp >= max` termwise,
the max-log LLR of the fading-marginalised demapper is one-signed above the
exact one, which holds at every one of the 115 200 samples at every Eb/N0
point. The RMS error is 1.3 to 1.7 nats, and the decoded cost grows with
Eb/N0: the BER ratio against exact rises from **1.433** at 2 dB to **3.194** at
10 dB for lognormal fading, and to **5.750** at 10 dB when the fading average
is over an estimated channel state. On gamma-gamma fading at 2 dB the max-log
LLRs have a GMI of **−0.5949**, meaning a bit-metric decoder fed them is worse
off than one fed zeros.

**Clipping is much cheaper than its LLR error suggests, and that gap is the
useful result.** At Eb/N0 = 8 dB, saturating at `L_max = 5` saturates 26.4 per
cent of the LLRs and introduces a relative RMS LLR error of **0.8177** — and
leaves the decoded BER at 2.590e-02 against 2.608e-02 unclipped, a ratio of
0.993 which is inside one binomial standard error. The decoded cost only appears
below `L_max = 3`: a ratio of 1.250 at 3, 2.028 at 2 and 3.873 at 1. An LLR-error
budget would have forced a far wider fixed-point word than the decoder needs.

The 4-PPM max-log error has an analytic bound: each of the two sums in a bit
LLR of a 4-ary constellation has two terms, `0 <= logsumexp(a,b) − max(a,b) <= log 2`,
so the bit-LLR error lies in `[−log 2, +log 2]`. The measured maximum is
**0.692962** against `log 2 = 0.693147`, i.e. 0.999733 of the bound, so the
bound is tight and attained.

## 4. The mismatched-CSI penalty (`validate_csi_mismatch.py`)

Raw output: [`csi_mismatch_output.txt`](csi_mismatch_output.txt). Runtime about
24 s.

The bias sweep changes **only** the estimate: the transmitted bits, the fading
and the noise are identical at every bias because `simulate_ook` draws them in
a fixed order from the same seed and the bias is applied afterwards. The
penalty is therefore a property of the demapper.

At Eb/N0 = 8 dB with the shipped LDPC code, decoded bit error rate of the
plug-in demapper:

| bias (dB) | −4 | −3 | −2 | −1 | 0 | +1 | +2 | +3 | +4 |
|---|---|---|---|---|---|---|---|---|---|
| decoded BER | 7.630e-02 | 3.527e-02 | 1.168e-02 | 3.600e-03 | 1.917e-03 | 6.933e-03 | 8.415e-02 | 2.777e-01 | 4.851e-01 |
| ratio to unbiased | 39.8 | 18.4 | 6.10 | 1.88 | 1.00 | 3.62 | **43.9** | 144.9 | 253.1 |
| GMI (bit/bit) | +0.549 | +0.608 | +0.666 | +0.714 | +0.737 | +0.689 | +0.391 | **−1.170** | **−7.616** |

**The two directions are not equivalent.** At equal magnitude the
over-estimating case costs **7.20x** more decoded errors than the
under-estimating case at 2 dB, 7.87x at 3 dB and 6.36x at 4 dB. The GMI goes
negative above about 2 dB of over-estimate, which is the quantitative statement
that the demapper has stopped being merely inaccurate and become misleading.

**The mechanism, measured.** The plug-in LLR crosses zero at `y = a h_hat / 2`
while the correct threshold is `y = a h / 2`, so a high estimate raises the
threshold and misses ones. Splitting the raw hard-decision errors by
transmitted bit gives a missed-one share of **0.1029** at −4 dB, 0.5055 at 0 dB
and **0.9912** at +4 dB. Because the raw hard-decision BER is invariant to any
positive LLR scaling, its over/under ratio isolates the threshold effect:
at 2 dB that ratio is **1.339**, so of the 7.203x decoded penalty, **5.381x** is
attributable to over-confidence in the LLR magnitudes and the rest to the
shifted threshold.

**A bias that the receiver knows about costs nothing.** The posterior-aware LLR
gives **1.916667e-03 at every bias from −4 to +4 dB**, exactly the true-CSI
figure, because the posterior removes a deterministic log-offset exactly. This
is the single most actionable result in the repository: characterise the
estimator's bias and marginalise, rather than training anything.

Jitter and staleness are not free even when unbiased in `log h`: 1 dB of jitter
raises the plug-in BER to 1.200e-02 against 5.567e-03 for the posterior-aware
LLR, and a stale estimate at `rho = 0.6` to 8.058e-02 against 1.577e-02.

## 5. The learned corrector (`validate_corrector.py`)

Raw output: [`corrector_output.txt`](corrector_output.txt). This script fits
three forests, tunes four free parameters by decoded BER and evaluates eight
demappers in three regimes; it took **175 s and 276 s** on two runs of the same
deterministic code, with byte-identical output, and the spread is contention
from the four sibling build agents. The three-minute budget applies to a single
training or Monte Carlo run: one forest fit on 240 000 samples takes about 12 s.

Splits are disjoint by construction and every free parameter — `alpha`,
`L_max`, and the forest's own output clip — is chosen on the tuning split, never
on the reporting split. Tuned values, from the raw output:

| regime | `plugin_scaled` alpha | `plugin_clipped` L_max | joint (alpha, L_max) | learned clip |
|---|---|---|---|---|
| bias +2 dB | 0.20 | 2.00 | (0.30, 2.00) | 12.0 |
| bias −2 dB | **1.00** | 12.00 | (1.00, 12.00) | 8.0 |
| stale rho = 0.8 | 0.70 | 5.00 | (0.70, 5.00) | 8.0 |

Decoded bit error rate on the report split, 1500 blocks (75 000 information
bits) at Eb/N0 = 8 dB:

| method | bias +2 dB | bias −2 dB | stale rho = 0.80 |
|---|---|---|---|
| `known_csi` (unreachable bound) | 2.773e-03 | 2.773e-03 | 2.773e-03 |
| **`csi_aware_exact` (analytic)** | **5.827e-03** | **5.827e-03** | **8.213e-03** |
| `csi_aware_maxlog` | 2.199e-02 | 2.199e-02 | 2.708e-02 |
| `plugin` | 1.855e-01 | 1.488e-02 | 2.664e-02 |
| `plugin_scaled` | 1.540e-01 | 1.488e-02 | 2.241e-02 |
| `plugin_clipped` | 1.573e-01 | 1.495e-02 | 2.140e-02 |
| `plugin_scaled_clipped` | 1.521e-01 | 1.495e-02 | 1.977e-02 |
| `learned` | 6.533e-03 | 6.280e-03 | 8.707e-03 |

### Where the baseline won

| Item | What happened |
|---|---|
| **The analytic posterior-aware LLR beats the learned corrector in all three regimes** | by **1.121x**, **1.078x** and **1.060x**. It is also closed form for a lognormal channel, costs one 32-node Gauss-Hermite quadrature per sample, and needs no training data. It does need the error-model parameters, which the corrector is not given; that is the only axis on which the corrector wins, and it is stated wherever the comparison appears. No retuning was attempted and the baseline was not removed. |
| **Plain scalar rescaling, the obvious non-learned competitor, barely helps** | With `alpha` tuned on a disjoint split it recovers **17.0 per cent** of the plug-in's excess BER when the estimate is biased high (1.855e-01 → 1.540e-01), and in the under-estimating regime the tuner selected **alpha = 1.00**, i.e. no rescaling at all, because none helped. The reason is structural and is measured in §4: a positive scalar cannot move the zero crossing `y = a h_hat / 2`, and the threshold error is a large part of the damage. Clipping and clip-plus-rescale behave the same way. Anyone expecting a scale factor to fix mismatched CSI on an OOK link should read that row. |
| **The learned corrector is still far from the bound** | 2.356x, 2.264x and 3.139x worse than true CSI. Pilot bits are not a substitute for knowing the channel. |
| **The uncertainty output does not rank risk** | The per-tree dispersion correlates 0.711 with the model's own deviation from the analytic reference, but **−0.241** with the hard-decision error indicator, and the error rate falls monotonically across its quintiles (0.2237, 0.1468, 0.0483, 0.0099, 0.0068). It grows with `\|L\|`, which is where decisions are reliable. Normalising by `1 + \|L\|` removes the scale but gives a non-monotone ordering (0.0178, 0.1002, 0.1347, 0.1380, 0.0449). `\|L\|` itself orders risk cleanly (0.3268 → 3.5e-05). Published as measured; no recalibration layer was added. |
| **Feature importances say the model is mostly re-reading the plug-in LLR** | plug-in LLR 0.569, `y/sigma` 0.400, the other four features 0.031 in total. The model is correcting a function it was handed rather than discovering a new statistic. |

## 6. Cross-check X3 and the comparand (`validate_cross_check_x3.py`)

Raw output: [`cross_check_x3_output.txt`](cross_check_x3_output.txt). Runtime
about 3 s.

### X3: PASS

The configuration is frozen in the script's own docstring and printed in the raw
output: OOK, extended Hamming (8,4) with weight enumerator `[1,0,0,0,14,0,0,0,1]`
confirming `d_min = 4`, unit-mean lognormal fading at scintillation index 0.30
drawn independently per channel bit, `a = 2 sigma sqrt(R Eb/N0)` with `R = 1/2`
and Eb per information bit, 40 000 blocks (160 000 information bits) per point,
seed 20261006 with one generator per point as
`default_rng(20261006 + 1000 * round(Eb/N0))`, draws in the order information
bits, then fading, then noise. **Both** decoders are exhaustive maximum
likelihood over all 16 codewords, and they are handed the same LLR array — the
hard decoder sees only its sign — so the realisations are identical by
construction rather than by seed agreement.

| Eb/N0 (dB) | 0 | 2 | 4 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|---|---|
| soft BER | 2.0953e-01 | 1.3696e-01 | 7.7375e-02 | 3.2925e-02 | 1.0544e-02 | 2.2375e-03 | 3.6250e-04 |
| hard BER | 2.9522e-01 | 2.3286e-01 | 1.7129e-01 | 1.0826e-01 | 5.9069e-02 | 2.6913e-02 | 9.2688e-03 |
| soft <= hard | yes | yes | yes | yes | yes | yes | yes |
| hard / soft | 1.409 | 1.700 | 2.214 | 3.288 | 5.602 | 12.028 | 25.569 |

The ordering holds strictly at every stated Eb/N0. Per-block agreement counts
are in the raw output; at 12 dB, 693 blocks are decoded correctly by the soft
decoder and not by the hard one, against **5** the other way.

### The comparand

Uncoded hard-decision BPSK over lognormal fading, with the complete
configuration printed in the raw output. **No comparison to any other product's
value is made here**, because the value another product computes for this
quantity was deliberately not supplied to this build; the grid is published so
the coordinating session can do the comparison.

Headline cell:

| field | value |
|---|---|
| modulation | antipodal BPSK, `s` in `{−1, +1}`, one sample per bit |
| detection | thermal-limited signal-independent AWGN, `sigma = 1` |
| amplitude | `a = sigma sqrt(2 Eb/N0)`, uncoded so `R = 1`; `a = 4.472135955` |
| fading | unit-mean lognormal, independent per bit, `sigma_I^2 = 0.30` |
| detector | hard decision `sign(y)` |
| Eb/N0 | 10.00 dB, per information bit |
| bits | 2 000 000 |
| seed | 20261361 (`20261048 + 101*round(10*sigma_I2) + round(Eb/N0)`) |
| draw order | symbols, then fading, then noise |
| **sample BER** | **7.549000000e-03** (15 098 errors) |
| **binomial standard error** | **6.120463e-05** |
| three-se interval | [7.365386e-03, 7.732614e-03] |
| quadrature reference `E_h[Q(h sqrt(2 Eb/N0))]` | 7.518905101e-03 |

The full grid covers `sigma_I^2` in {0.1, 0.3, 1.0} and Eb/N0 in
{0, 5, 10, 15, 20} dB; the worst deviation from the analytic quadrature over the
14 cells with a nonzero error count is **2.215** binomial standard errors. The
`sigma_I^2 = 0.1`, 20 dB cell produced **0 errors in 2 000 000 bits** against a
quadrature prediction of 9.08e-09, which is consistent and is reported as a zero
rather than as an upper bound.

## 7. Worked example (`worked_example.py`)

Raw output: [`worked_example_output.txt`](worked_example_output.txt).

Produces every number printed in the README's worked-example block, so the two
cannot drift. The last two lines are the product in one line: at `y = 3.0` with
`h_hat = 1.4` the plug-in LLR says **+12.636868** — a confident zero — and the
posterior-aware LLR says **−2.519930**, a one. Opposite signs, same sample.

## Reproducing every number

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

ruff check src/ tests/ examples/ validation/
python -m pytest tests/ -q --junit-xml=junit.xml   # 161 tests
python -m softdecode --help

python validation/validate_quadrature.py        # quadrature vs Monte Carlo, samplers
python validation/validate_awgn_limit.py        # the AWGN known-answer limit
python validation/validate_maxlog_clipping.py   # max-log and clipping costs
python validation/validate_csi_mismatch.py      # the mismatch penalty, both directions
python validation/validate_corrector.py         # the learned corrector vs everything
python validation/validate_cross_check_x3.py    # cross-check X3 and the comparand
python validation/worked_example.py             # the README worked example

MPLBACKEND=Agg python examples/llr_curves.py
MPLBACKEND=Agg python examples/mismatch_asymmetry.py
MPLBACKEND=Agg python examples/clipping_cost.py
MPLBACKEND=Agg python examples/corrector_benchmark.py
```

Every script is deterministic given its seeds, which are printed in its own
output: `20261006` for the LDPC construction, the dataset splits and the forest,
`20261048` for the comparand grid. Committed raw output for all of them is
beside this file.

Research-grade software. Not flight-qualified, not certified, not approved for
operational aerospace use.
