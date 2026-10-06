# Validation evidence

**Status: TESTING** · Validation level 2 (research grade) · Apache-2.0 ·
© 2026 OPTIMA Organisation

Every number in this file was produced by a script in this directory, run on
2026-10-06 on the build machine (Python 3.13.16, numpy 2.5.3, scipy 1.18.1,
scikit-learn 1.9.1, two cores shared with four sibling build agents). The raw
stdout of each run is committed beside the script as `*_output.txt`. No number
below was typed in from anywhere else.

Where a check did not come out clean, or where a baseline beat the learned
model, it is in this file with its number. Those are sections 2, 3, 6 and 7.

## Summary

| # | check | reference | result | tolerance | verdict |
|---|---|---|---|---|---|
| 1 | CRC catalogue check values, 5 CRCs | `crcmod` 1.7 `predefined.py` | 5 of 5 agree | exact | pass |
| 1 | CRC-32 against `zlib.crc32` | Python standard library | 0 mismatches in 3000 payloads / 763709 bytes | exact | pass |
| 1 | table-driven against bit-at-a-time | internal, independent code path | 0 mismatches in 2000 comparisons | exact | pass |
| 2 | simulator against the classical closed forms | Lin & Costello; Tanenbaum | worst 0.616 % relative, worst \|z\| 2.36, over 78 cases | 1 % relative where rel SE < 0.5 %, \|z\| ≤ 4 | pass |
| 3 | window-limited selective-repeat expression | same | **misstates by up to 145 %** for 1 < W < N | none; reported | **expression is an upper bound, documented** |
| 3 | ideal `eta = 1-p` at the textbook window | same | **43.8 to 63.7 % shortfall at W = N** | none; reported | **textbook window is not enough, documented** |
| 4 | Gilbert-Elliott marginal rate | equation (3), this repo | worst \|z\| 2.60 over 6 cases | \|z\| ≤ 4 | pass |
| 4 | Gilbert-Elliott mean sojourn | equation (4) | worst \|z\| 2.66 over 6 cases | \|z\| ≤ 4 | pass |
| 4 | Gilbert-Elliott autocorrelation | equation (5) | worst 2.87e-03 absolute over 12 lags | 5e-03 absolute | pass |
| 5 | `block_fer` against a `math.comb` sum | equation (7) | worst 3.33e-16 absolute over 25 cases | 1e-12 absolute | pass |
| 5 | type-I chase analytic against Monte Carlo | internal | worst \|z\| 1.33 over 6 cases | \|z\| ≤ 4 | pass |
| 5 | type-II exact DP against Monte Carlo | internal | worst \|z\| 1.34 over 12 cases | \|z\| ≤ 4 | pass |
| 5 | independent-round approximation error | exact nested DP | **up to 1.54 % on the residual FER** | none; reported | approximation characterised |
| 6 | marginal window gain against finite difference | equation (6) | worst 6.94e-17 absolute over 36 cases | 1e-12 absolute | pass |
| 7 | learned policy against tuned fixed schedule | this repo | **learned loses by 1.66 % and 3.03 % of cost** | none; reported | **baseline wins, published** |
| 7 | learned policy against analytic fixed schedule | this repo | learned wins by 24.1 % and 36.4 % of cost | none; reported | learned wins |

Test suite: **229 tests, 0 failures, 0 errors, 0 skipped** (counted from
`pytest --junit-xml`, 47.3 s).

---

## 1. Frame check sequence (`validate_crc.py`)

An ARQ simulator needs a frame check, because "the receiver detected the error"
has to be made concrete somewhere. `crcmod` and `commpy` are the established
libraries and neither installs in this build environment, so the check is
implemented in-package and verified against two independent references.

Catalogue check values, defined as the CRC of the ASCII string `123456789`,
read from the `crcmod` 1.7 source distribution file
`python3/crcmod/predefined.py` on 2026-10-06:

| CRC | width | computed | catalogue | agree |
|---|---|---|---|---|
| crc-32 | 32 | 0xcbf43926 | 0xcbf43926 | yes |
| crc-32c | 32 | 0xe3069283 | 0xe3069283 | yes |
| crc-ccitt-false | 16 | 0x29b1 | 0x29b1 | yes |
| crc-16 | 16 | 0xbb3d | 0xbb3d | yes |
| crc-8 | 8 | 0xf4 | 0xf4 | yes |

`crcmod` parameterises the initial value *after* the final XOR, so its table
lists `init = 0` for CRC-32 where this package lists `init = 0xFFFFFFFF`. Both
describe the same function; the agreement on the check value is what shows it.

CRC-32 against `zlib.crc32`, an entirely independent implementation that is
always available: **0 mismatches over 3000 random payloads, 763709 bytes.**

Table-driven against bit-at-a-time, for all five CRCs over 400 random payloads
each: **0 mismatches in 2000 comparisons.**

Frame-check detection of 1 to 8 *distinct* bit flips over a 544-bit frame,
8000 trials each:

| CRC | detected | rate | 2^-w |
|---|---|---|---|
| crc-32 | 8000 / 8000 | 1.00000 | 2.33e-10 |
| crc-32c | 8000 / 8000 | 1.00000 | 2.33e-10 |
| crc-ccitt-false | 7999 / 8000 | 0.99987 | 1.53e-05 |
| crc-16 | 8000 / 8000 | 1.00000 | 1.53e-05 |
| crc-8 | 7974 / 8000 | 0.99675 | 3.91e-03 |

CRC-8's miss rate of 3.25e-03 is consistent with its 2^-8 = 3.91e-03 random
coincidence probability, which is the expected behaviour of an 8-bit check and
is why it is not the default.

**A check that was initially wrong, and how.** The first version of this script
sampled flip positions *with* replacement, so a repeated position flipped a bit
twice and left the frame unaltered. That produced 7 "undetected errors" in
20000 for CRC-32, none of which was an error at all. Sampling without
replacement removed the artefact. The number is recorded in
`validate_crc_output.txt` because a validation file that only contains its
final state is less useful than one that says what went wrong.

These rates are for a pattern a CRC is good at. They are **not** a residual
undetected-error model for a real channel, and this package does not provide
one.

---

## 2. The simulator against the classical closed forms (`validate_closed_form.py`)

This is the check that licenses everything else in the repository. The three
state-machine simulators are run on *independent* frame errors with the window
each closed form assumes, which is the one regime where the textbook
expressions are exact, and the measured goodput is compared with them. Only
after this agreement is established is the simulator used on a correlated
channel, where no closed form exists.

The expressions, with their sources:

```
stop-and-wait      eta = (1-p)/N
go-back-N          eta = (1-p)/(1 - p + N p)          requires W >= ceil(N)
selective repeat   eta = 1 - p                        requires W unbounded
```

S. Lin and D. J. Costello, Jr., *Error Control Coding*, 2nd ed., Prentice Hall,
2004, chapter on ARQ error control; the same results appear in A. S. Tanenbaum,
*Computer Networks* as the sliding-window efficiency derivations. No page
numbers are quoted because none were verified against the printed text in this
build.

Grid: `N` in {5, 20, 60, 200} × `p` in {0.001, 0.01, 0.05, 0.10, 0.20, 0.40},
three protocols, plus six window-one cases. 1200000 slots per sliding-window
run; stop-and-wait runs are sized at 40000 attempts, so 40000·N slots, because
stop-and-wait makes only one attempt per N slots and a fixed slot count would
leave it the noisiest measurement in the table.

### Worst-case agreement, 78 cases

| protocol | worst \|z\| case | rel % | \|z\| |
|---|---|---|---|
| stop-and-wait | N=60, p=0.01 | 0.076 | 2.06 |
| go-back-N | N=5, p=0.2, W=5 | 0.238 | 1.48 |
| selective repeat | N=60, p=0.1, W=2400 | 0.048 | 1.53 |
| selective repeat, W=1 | N=200, p=0.01, W=1 | 0.251 | 2.36 |

| protocol | worst relative case (own rel SE < 0.5 %) | rel % | \|z\| |
|---|---|---|---|
| stop-and-wait | N=200, p=0.4 | 0.529 | 1.36 |
| go-back-N | N=20, p=0.1, W=20 | **0.616** | 1.46 |
| selective repeat | N=20, p=0.4, W=800 | 0.069 | 0.87 |
| selective repeat, W=1 | N=200, p=0.05, W=1 | 0.518 | 2.17 |

**Worst relative deviation: 0.616 per cent. Worst \|z\|: 2.36.**

Tolerance adopted: `|z| <= 4` on every case, and `|relative| <= 1.0 %` on every
case whose own Monte Carlo relative standard error is below 0.5 per cent. Both
pass. The `z` criterion is the real one — it is what separates Monte Carlo
noise from a modelling error — and the relative criterion is the engineering
statement, meaningful only where the simulation is precise enough to carry it.
Standard errors are batch means over 20 blocks, which is the correct estimator
for a correlated time series; using an independent-sample standard error here
would overstate the precision and make a correct simulator look broken.

The identity `sr_throughput(p, N, 1) == sw_throughput(p, N)` holds to
**2.78e-17** absolute over 24 grid points, which is floating-point exact.

---

## 3. Where the closed forms break (`validate_closed_form_output.txt`, section 5)

Two of the textbook expressions are **upper bounds, not answers**, and this
repository measures by how much. Neither of these is a failure of the
simulator; both are properties of the expressions.

### 3a. The window-limited expression `eta = (1-p) W/N`

It assumes the sender always holds `W` transmissions in flight. A real
selective-repeat sender cannot advance its window base past an unacknowledged
frame, so a frame under retransmission holds its sequence number and the window
fills behind it. The expression is exact at `W = 1` (where it is stop-and-wait)
and progressively optimistic above:

| N | p | W | W/N | simulated | `(1-p)W/N` | misstated by |
|---|---|---|---|---|---|---|
| 60 | 0.01 | 1 | 0.017 | 0.016490 | 0.016500 | +0.06 % |
| 60 | 0.01 | 30 | 0.500 | 0.422046 | 0.495000 | +17.29 % |
| 60 | 0.05 | 30 | 0.500 | 0.314908 | 0.475000 | +50.84 % |
| 60 | 0.20 | 7 | 0.117 | 0.066557 | 0.093333 | +40.23 % |
| 60 | 0.20 | 30 | 0.500 | 0.201509 | 0.400000 | +98.50 % |
| 200 | 0.05 | 100 | 0.500 | 0.259667 | 0.475000 | +82.93 % |
| 200 | 0.20 | 25 | 0.125 | 0.054351 | 0.100000 | +83.99 % |
| 200 | 0.20 | 100 | 0.500 | 0.163181 | 0.400000 | **+145.13 %** |

`arqlonghaul.closedform.sr_throughput` documents this in its docstring and
points the reader at the simulator.

### 3b. The ideal expression `eta = 1-p` and the textbook window

The ideal expression assumes an unbounded window. The textbook recommendation
is a window of one bandwidth-delay product plus one frame, which is `W = N`.
At `N = 60`:

| p | ideal 1−p | W=N | W=2N | W=3N | W=4N | W=6N | W=8N |
|---|---|---|---|---|---|---|---|
| 0.01 | 0.99000 | 0.7321 | 0.9950 | 1.0002 | 1.0002 | 1.0002 | 1.0002 |
| 0.05 | 0.95000 | 0.5653 | 0.9088 | 0.9942 | 1.0001 | 1.0004 | 1.0004 |
| 0.10 | 0.90000 | 0.5024 | 0.8155 | 0.9690 | 0.9972 | 1.0001 | 1.0001 |
| 0.20 | 0.80000 | 0.4324 | 0.7163 | 0.9075 | 0.9837 | 0.9994 | 0.9998 |
| 0.40 | 0.60000 | 0.3613 | 0.6081 | 0.8015 | 0.9257 | 0.9953 | 0.9996 |

(fraction of the ideal achieved; `validate_window.py` section 4.)

Smallest window reaching 99 per cent of the ideal, at `N = 60` and a
1115-octet frame:

| p | multiple of N | frames | bytes |
|---|---|---|---|
| 0.01 | 2 | 120 | 133800 |
| 0.05 | 3 | 180 | 200700 |
| 0.10 | 4 | 240 | 267600 |
| 0.20 | 6 | 360 | 401400 |
| 0.40 | 6 | 360 | 401400 |

**The textbook window is adequate only on an error-free link.** At a 20 per
cent frame error rate the window needed is six times it, which at `N = 60` and
1115-octet frames is 401400 bytes of send buffer rather than 66900. That is a
procurement decision, not a rounding error.

---

## 4. The Gilbert-Elliott channel against its own analytic statistics (`validate_burst_channel.py`)

Before the correlated channel is used to contradict the independent-error
expressions, it has to be shown to be the channel it claims to be. 2000000
slots per case.

Model: two states, Good and Bad, a first-order Markov chain on the frame slot
index, with per-state error probabilities (E. N. Gilbert, *Capacity of a
burst-noise channel*, Bell System Technical Journal 39(5):1253–1265, 1960;
E. O. Elliott, *Estimates of error rates for codes on burst-noise channels*,
BSTJ 42(5):1977–1997, 1963. Titles and years were confirmed against the Nokia
Bell Labs publication pages for both papers on 2026-10-06; volume, issue and
page numbers are as given in the reference list of arXiv:2005.06921, read the
same day).

```
pi_b   = p_gb / (p_gb + p_bg)                                              (2)
p_bar  = (1 - pi_b) eps_g + pi_b eps_b                                     (3)
E[Bad sojourn] = 1 / p_bg                                                  (4)
rho_k  = pi_b (1-pi_b) (eps_b - eps_g)^2 lam^k / (p_bar (1 - p_bar))       (5)
```

with `lam = 1 - p_gb - p_bg`.

### Marginal frame error rate, equation (3)

The standard error of the sample mean of a correlated binary series is inflated
by `sqrt((1+lam)/(1-lam))`; at a burst length of 25 that factor is about 6.8,
so an independent-sample standard error would be wrong by that much and would
fail a correct channel.

| target FER | burst | analytic | sampled | iid SE | correlated SE | z |
|---|---|---|---|---|---|---|
| 0.02 | 2 | 0.020000 | 0.019936 | 9.90e-05 | 1.69e-04 | −0.38 |
| 0.05 | 5 | 0.050000 | 0.050291 | 1.54e-04 | 4.49e-04 | +0.65 |
| 0.05 | 25 | 0.050000 | 0.049119 | 1.54e-04 | 1.05e-03 | −0.84 |
| 0.10 | 25 | 0.100000 | 0.100385 | 2.12e-04 | 1.41e-03 | +0.27 |
| 0.10 | 100 | 0.100000 | 0.102579 | 2.12e-04 | 2.84e-03 | +0.91 |
| 0.30 | 50 | 0.300000 | 0.292990 | 3.24e-04 | 2.69e-03 | **−2.60** |

**Worst \|z\|: 2.60.** Tolerance `|z| <= 4`. Pass.

### Mean Bad-state sojourn, equation (4), by run-length counting

**Worst \|z\|: 2.66** over six cases (worst at target 25, measured 26.0883
from 3997 runs, SE 0.4084). Tolerance `|z| <= 4`. Pass.

### Autocorrelation, equation (5)

**Worst absolute error 2.87e-03** over 12 (burst, lag) combinations at lags 1,
5, 25 and 100. Tolerance 5e-03 absolute, which is about the sampling standard
error of an autocorrelation at n = 2000000. Pass.

The inverse construction `from_mean_and_burst` returns its inputs to 8 decimal
places in all six cases.

---

## 5. Hybrid ARQ (`validate_harq.py`)

### 5a. `block_fer`, equation (7)

```
P_F(n, t, p) = 1 - sum_{i=0}^{t} C(n,i) p^i (1-p)^(n-i)                    (7)
```

the standard block error probability of a bounded-distance decoder on a binary
symmetric channel (Lin & Costello, 2nd ed., 2004). Checked against a direct
`math.comb` summation, which shares no code with the scipy implementation the
package uses: **worst absolute difference 3.33e-16** over 25 (n, t, p) cases.
Tolerance 1e-12. Pass.

Correcting power from the Singleton bound, `d <= n - k + 1`:

```
t_m = floor(alpha (n_m - k) / 2),   alpha in (0, 1]                        (8)
```

`alpha` is an explicit code-family efficiency: 1.0 is the MDS limit, which is
the best any family can do. **No named standardised code is claimed.** The
sensitivity of every result to `alpha` is reported in section 5d.

### 5b. Type-I chase combining, analytic against Monte Carlo

60000 frames per point, 6 configurations. **Worst \|z\| on the residual frame
error rate: 1.33.** Tolerance `|z| <= 4`. Pass.

### 5c. Type-II incremental redundancy, three computation paths

The package computes IR goodput two ways and a third as a check:

- the **independent-round approximation** `q_m = P_F(n_m, t_m, p_b)`, which
  ignores that a frame reaching round `m` is already known to have failed
  rounds 1..m−1;
- the **exact nested dynamic program**, which carries the sub-probability
  distribution of the accumulated symbol error count over frames not yet
  decoded and absorbs the decoded states after each round;
- **Monte Carlo**.

| n1 | Es/N0 dB | approx residual | exact residual | MC residual | approx error | MC z vs exact |
|---|---|---|---|---|---|---|
| 240 | 0 | 1.1353e-02 | 1.1341e-02 | 1.1000e-02 | +0.105 % | −0.78 |
| 260 | 0 | 2.5212e-03 | 2.5160e-03 | 2.4167e-03 | +0.206 % | −0.48 |
| 300 | 0 | 8.3825e-05 | 8.3359e-05 | 1.3333e-04 | +0.559 % | +1.34 |
| 340 | 0 | 1.8733e-06 | 1.8522e-06 | 0 | +1.137 % | −0.11 |
| 340 | 2 | 2.1844e-20 | 2.1512e-20 | 0 | **+1.542 %** | 0.00 |

**The approximation overstates the residual frame error rate by up to 1.54 per
cent.** It is reported, not bounded: the exact DP is what the package
publishes, `harq_goodput` is labelled as the approximation in its docstring and
its `method` field, and a test asserts the approximation is always an upper
bound on the exact residual. **Worst \|z\| of the Monte Carlo against the exact
DP: 1.34**, tolerance `|z| <= 4`, pass — noting that the Monte Carlo column is
only informative where the residual exceeds 1/60000 = 1.7e-05, so the rows
showing 0 are consistent with the analytic value without testing it.

### 5d. The crossover, and the assumptions that fix it

Baseline: `k = 200`, Es/N0 = 1.0 dB, `alpha = 0.5`, `delta = 40`, `M = 4`,
comparing a rate-0.90 first transmission with IR against a rate-0.50 first
transmission with IR.

**Crossover D = 85.87 channel symbol times.** Below it, retransmission is
cheaper; above it, paying for redundancy up front is.

| assumption varied | value | crossover D | change |
|---|---|---|---|
| alpha | 0.4 | 49.18 | −42.7 % |
| alpha | **0.5** | **85.87** | — |
| alpha | 0.6 | 118.20 | +37.7 % |
| alpha | 0.8 | 160.44 | +86.8 % |
| Es/N0 dB | −1 | no crossing in grid | reported, not extrapolated |
| Es/N0 dB | 0 | 31.02 | −63.9 % |
| Es/N0 dB | **1** | **85.87** | — |
| Es/N0 dB | 2 | 162.29 | +89.0 % |
| Es/N0 dB | 3 | 407.26 | +374.3 % |
| delta | 20 | 37.54 | −56.3 % |
| delta | **40** | **85.87** | — |
| delta | 80 | 96.61 | +12.5 % |
| delta | 160 | 20.34 | −76.3 % |
| max_rounds | 2 | no crossing in grid | reported, not extrapolated |
| max_rounds | 3 | 80.59 | −6.2 % |
| max_rounds | **4** | **85.87** | — |
| max_rounds | 8 | 85.95 | +0.1 % |

**The crossover is not a constant of nature.** It moves by a factor of four
over a 4 dB range of Es/N0 and by a factor of three over the `alpha` range
tested. Quoting it without those five numbers would be dishonest.

### 5e. Optimal first rate against D, and where real links sit

| D symbols | optimal R1 | n1 | goodput | mean rounds | residual |
|---|---|---|---|---|---|
| 1 | 0.7576 | 264 | 0.717741 | 1.3330 | 6.23e-07 |
| 50 | 0.7220 | 277 | 0.586137 | 1.1580 | 1.16e-07 |
| 300 | 0.6601 | 303 | 0.327455 | 1.0229 | 3.63e-09 |
| 10000 | 0.6079 | 329 | 0.019343 | 1.0011 | 3.63e-11 |
| 1000000 | 0.5450 | 367 | 0.000200 | 1.0000 | 1.18e-13 |

Monotone: the longer the round trip, the more redundancy it is worth paying for
in advance.

Taking BPSK so one channel symbol carries one bit, `D` equals the
bandwidth-delay product in bits:

| preset | D symbols | D / crossover | side of the crossover |
|---|---|---|---|
| geo | 5.175e+05 | 6026 | pay redundancy up front |
| leo | 5.836e+05 | 6796 | pay redundancy up front |
| lunar | 2.614e+06 | 3.045e+04 | pay redundancy up front |
| mars_near | 9.393e+08 | 1.094e+07 | pay redundancy up front |

**Every preset is four to seven orders of magnitude past the crossover.** For a
single HARQ process with no pipelining the answer on a space link is always to
pay up front. That conclusion rests entirely on the no-pipelining assumption: a
sender running enough parallel HARQ processes to fill the round trip hides `D`
completely, and then the crossover moves back toward `D = 0` and retransmission
wins again. **The crossover is a statement about one process, not about the
link.**

---

## 6. Window sizing (`validate_window.py`)

Link geometry, with the frame size on every row because a frame count is
meaningless without one:

| preset | rate bit/s | RTT s | frame B | N slots | BDP B | BDP frames | knee frames |
|---|---|---|---|---|---|---|---|
| geo | 2e+06 | 0.2587 | 1115 | 59.01 | 64684.6 | 58.01 | 60 |
| leo | 5e+07 | 0.01167 | 1115 | 66.42 | 72945.5 | 65.42 | 67 |
| lunar | 1e+06 | 2.614 | 1115 | 294.10 | 326805 | 293.10 | 295 |
| mars_near | 2.56e+05 | 3669 | 1115 | 105305.54 | 1.17415e+08 | 105304.54 | 105306 |

`bdp_frames == N - 1` to **0.000e+00** absolute over all four presets.

At the knee, at a 5 per cent frame error rate (closed form):

| preset | knee frames | knee bytes | SR eta | SR bit/s | GBN eta | GBN bit/s |
|---|---|---|---|---|---|---|
| geo | 60 | 66900 | 0.95000 | 1.87274e+06 | 0.24355 | 480108 |
| leo | 67 | 74705 | 0.95000 | 4.68184e+07 | 0.22243 | 1.09617e+07 |
| lunar | 295 | 328925 | 0.95000 | 936368 | 0.06068 | 59812.9 |
| mars_near | 105306 | 1.17416e+08 | 0.95000 | 239710 | 0.00018 | 45.5184 |

At the same window and the same frame error rate, **go-back-N delivers 6.39 per
cent of selective repeat's goodput on the lunar preset and 0.019 per cent on the
Mars preset.** On a link that long go-back-N is not a protocol choice, it is a
way of not using the link.

The marginal throughput gained per extra frame of window, `(1-p)/N` below the
knee and exactly 0 at or above it, agrees with a finite difference of the
throughput expression to **6.94e-17** absolute over 36 cases. Tolerance 1e-12.
Pass.

The simulated window requirement is in section 3b above, because it is a place
where the closed form is wrong rather than a place where it is right.

---

## 7. The learned redundancy policy (`validate_policy.py`)

Full detail in `MODEL_CARD.md`. Three disjoint seed sets: fit 1000–1119 trains
the model, tune 2000–2039 selects every free parameter of every policy
including the learned one's forest depth and leaf size, report 3000–3299
carries the published numbers and is read once. `SeedSplit` raises if the sets
overlap, and a test asserts it.

Report: 300 seeds × 150 frames = 45000 frames per policy. Lower cost is better.

### long-burst regime (mean faded sojourn 25 rounds, autocorrelation 1/e 19.50 rounds)

| policy | cost/frame | ± | goodput | ± | residual FER |
|---|---|---|---|---|---|
| analytic-fixed | 443.733 | 8.377 | 0.59130 | 0.00382 | 6.240e-02 |
| **tuned-fixed** | **331.401** | **1.541** | **0.60758** | **0.00253** | 2.222e-04 |
| escalating | 337.514 | 1.878 | 0.59754 | 0.00301 | 0.000e+00 |
| learned-forest | 336.902 | 1.875 | 0.59940 | 0.00295 | 3.556e-04 |

### short-burst regime (mean faded sojourn 1.5 rounds, autocorrelation 1/e 0.56 rounds)

| policy | cost/frame | ± | goodput | ± | residual FER |
|---|---|---|---|---|---|
| analytic-fixed | 550.983 | 3.228 | 0.52643 | 0.00136 | 1.054e-01 |
| **tuned-fixed** | **339.893** | **0.381** | **0.58864** | **0.00066** | 0.000e+00 |
| escalating | 345.650 | 0.463 | 0.57893 | 0.00077 | 0.000e+00 |
| learned-forest | 350.194 | 0.217 | 0.57118 | 0.00035 | 0.000e+00 |

### Verdicts, paired over the report seeds

| regime | learned against | goodput diff | cost diff | z on cost | verdict |
|---|---|---|---|---|---|
| long-burst | analytic-fixed | +1.370 % | −24.076 % | −12.45 | learned |
| long-burst | tuned-fixed | −1.346 % | +1.660 % | +2.27 | **tuned-fixed** |
| long-burst | escalating | +0.312 % | −0.182 % | −0.23 | tie |
| short-burst | analytic-fixed | +8.500 % | −36.442 % | −62.06 | learned |
| short-burst | tuned-fixed | −2.967 % | +3.031 % | +23.50 | **tuned-fixed** |
| short-burst | escalating | −1.339 % | +1.315 % | +8.88 | **escalating** |

`|z| < 2` is reported as a tie, because at this episode length a difference
smaller than two standard errors is not a result.

### Where a baseline won, stated without spin

**The tuned fixed schedule — two integers chosen by grid search on 40 tune
seeds — beats the learned random forest in both regimes.** It is marginal in
the long-burst regime (`|z| = 2.27`) and decisive in the short-burst regime
(`|z| = 23.50`), where the adaptive heuristic also beats the learned model
(`|z| = 8.88`). The learned model was not retuned after this was found, and it
has not been removed.

The structural reason was written down before the measurement: on a long link
feedback is stale by at least one round trip, so a policy's only purchase on
the channel is through its *statistics*, and a fixed schedule already encodes
the statistics perfectly. The measurement then says that the residual
correlation surviving the staleness — substantial in the long-burst regime, 19.5
rounds of memory against a 1-round decision interval — is still not enough to
beat a well-chosen constant. That is the result.

### The one comparison the learned model wins, and why it is not flattering

The learned model beats the analytic fixed schedule by 24.1 and 36.4 per cent
of cost. The reason is a real lesson rather than a success: choosing the
schedule that is optimal *at the mixture error probability* is not the same as
choosing the schedule that minimises *expected cost under the mixture*, because
cost is convex in the error probability. The analytic baseline consequently
ships a residual frame error rate of 6.2e-02 and 1.05e-01 where every other
policy achieves 1e-4 or better. It optimises for an average channel the link is
never in. **That is the analytic baseline's failure, not the learned model's
achievement.**

### Uncertainty output

Across-tree standard deviation of the chosen action's predicted cost: median
**68.99** symbol times (10th percentile 6.66, 90th 193.49). Decision margin to
the runner-up action: median **0.65 pooled standard deviations**, with **63.3
per cent of states below one standard deviation**.

At nearly two thirds of the states it visits the model cannot distinguish its
two best actions. That is an honest account of why a grid search beats it.

### Action usage, long-burst regime, one report seed, 2000 frames

| policy | 10 | 25 | 50 | 100 | 200 |
|---|---|---|---|---|---|
| analytic-fixed | 0.0000 | 0.1694 | 0.8306 | 0.0000 | 0.0000 |
| tuned-fixed | 0.0000 | 0.0000 | 0.8525 | 0.1475 | 0.0000 |
| escalating | 0.0000 | 0.0000 | 0.8543 | 0.1017 | 0.0440 |
| learned-forest | 0.0034 | 0.0289 | 0.8342 | 0.0668 | 0.0668 |

The learned policy spends 0.34 per cent of rounds at the 10-symbol increment,
a first-transmission code rate of 0.952 that essentially cannot decode at this
Es/N0. No baseline ever chooses it. Those rounds are close to wasted and they
are part of why it loses.

---

## 8. The model artifact (`export_policy_table.py`)

No model binary is committed. The greedy policy is a finite map from a
quantised observation to an action, exported as compressed `.npz`:

| file | states | bytes |
|---|---|---|
| `policy_table_long_burst.npz` | 200 | 7936 |
| `policy_table_short_burst.npz` | 255 | 10761 |

SHA-256 of each exported array and the numpy and scikit-learn versions are
printed in `export_policy_table_output.txt`, because a random forest is not
guaranteed to be bit-identical across scikit-learn releases and claiming
otherwise would be false.

---

## Reproducing every number

From the repository root, with `pip install -e ".[dev]"`:

```bash
python -m pytest tests/ -q --junit-xml=junit.xml      #  229 tests,  47 s
python validation/validate_crc.py                     #  section 1,  12 s
python validation/validate_closed_form.py             #  sections 2 and 3,  75 s
python validation/validate_burst_channel.py           #  section 4,   9 s
python validation/validate_burst_goodput.py           #  the headline, 33 s
python validation/validate_window.py                  #  sections 3b and 6, 14 s
python validation/validate_harq.py                    #  section 5,   7 s
python validation/validate_policy.py                  #  section 7, 100 s
python validation/export_policy_table.py              #  section 8,  15 s
python validation/worked_example.py                   #  the README example, 2 s
```

Total about 320 s on two contended cores. No single run exceeds three minutes,
which was the compute budget.
