# rtclock — Validation evidence (Level 1, educational to research grade)

**Product:** P036 RtClock · **Version:** 0.1.0 · **Date of run:** 2026-10-04
**Environment:** Python 3.13.16, NumPy 2.5.3, SciPy 1.18.1, pytest 9.1.1,
Hypothesis, Ruff. Build container: **1 CPU core**, 7.8 GiB RAM, **four
concurrent build agents**. Every number in this file was produced by running
the scripts in this directory in the session that wrote it; each script's raw
stdout is committed beside it.

**What "measured on a loaded shared machine" means here.** This product
measures timing, so the state of the host matters. Only two classes of number
in this file come from the host clock: the clock-resolution measurement
(section 5) and the two explicitly-labelled "for the record" real-clock runs
(4e, and the tail of `examples/loop_drift.py`). **Neither is asserted against
any bound**, because a noisy neighbour would break any such assertion and a
bound that passes only when the machine is quiet is not evidence. Everything
else is asserted against arithmetic — closed forms, hand computations, exact
order statistics, independent high-precision reference values — or against
injected synthetic traces on a deterministic simulated clock.

| # | Check | Script | Raw output | Result |
|---|---|---|---|---|
| 1 | Utilization bounds vs textbook and 40-digit decimal values | `validate_utilization_bounds.py` | `utilization_bounds_output.txt` | PASS |
| 2 | Response-time analysis vs hand computations, 4 task sets | `validate_response_times.py` | `response_times_output.txt` | PASS |
| 3 | Exact percentiles, both definitions, vs order statistics and NumPy | `validate_percentiles.py` | `percentiles_output.txt` | PASS |
| 4 | Loop drift vs injected clock skew, 60 000 comparisons | `validate_loop_drift.py` | `loop_drift_output.txt` | PASS (4e reported, not asserted) |
| 5 | Clock resolution measured, with its error term | `validate_clock_resolution.py` | `clock_resolution_output.txt` | consistency PASS; values reported |
| 6 | Overrun accounting and the P031/P036 cross-check | `validate_crosscheck_overruns.py` | `crosscheck_overruns_output.txt` | PASS; **cross-check AGREES on both definitions** |

Reproduce, from `products/P036/`:

```bash
python validation/validate_utilization_bounds.py
python validation/validate_response_times.py
python validation/validate_percentiles.py
python validation/validate_loop_drift.py
python validation/validate_clock_resolution.py
python validation/validate_crosscheck_overruns.py
```

Wall clock on the 1-core build container, including interpreter start:
2.1 s, 0.1 s, 3.6 s, 1.8 s, 1.0 s and 0.4 s respectively — **9.0 s in total**.
The test suite (`python -m pytest tests/ -q`) is **252 tests**, all passing,
in roughly 21 s on this 1-core container. These figures are from a shared machine and will vary.

---

## 1. Utilization bounds

**Reference.** C. L. Liu and J. W. Layland, "Scheduling Algorithms for
Multiprogramming in a Hard-Real-Time Environment", *Journal of the ACM*
**20**(1), 46–61 (1973), Theorem 5 (the rate-monotonic least upper bound) and
Theorem 7 (the EDF condition). Textbook anchor: G. C. Buttazzo, *Hard
Real-Time Computing Systems*, 3rd ed., Springer (2011), Sec. 4.3.

**Assumptions** (Liu & Layland Sec. 2): one processor, fully preemptive, zero
context-switch cost, independent tasks, no release jitter, deadlines equal to
periods. **Validity:** implicit-deadline sets only; `rm_utilization_test` and
`hyperbolic_bound_test` raise rather than return a number for `D < T`.

### 1a. `U_lub(n) = n(2^(1/n) - 1)` for n = 1…10

Computed against the same expression evaluated at **40 decimal digits** with
Python's `decimal` module. Tolerance 5e-13 absolute.

| n | rtclock | decimal (40 digits) | abs diff |
|---|---|---|---|
| 1 | 1.000000000000000 | 1.000000000000000 | 0.000e+00 |
| 2 | 0.828427124746190 | 0.828427124746190 | 2.220e-16 |
| 3 | 0.779763149684620 | 0.779763149684619 | 1.110e-16 |
| 10 | 0.717734625362931 | 0.717734625362932 | 3.331e-16 |

Worst absolute deviation over n = 1…10: **5.551e-16**. All PASS.

### 1b. Named values

| check | computed | expected | tolerance |
|---|---|---|---|
| `U_lub(2)` vs textbook 0.8284 | 0.8284271247461903 | 0.8284 | 5e-5 |
| `U_lub(2)` full precision | 0.8284271247461903 | 0.8284271247461903 | 1e-15 |
| `U_lub(3)` | 0.7797631496846196 | 0.7797631496846196 | 1e-15 |
| `U_lub(10^6)` → ln 2 | 0.6931474207938493 | 0.6931471805599453 | 2.5e-7 |

The n = 2 textbook value **0.8284** and the n → ∞ limit **ln 2 =
0.6931471805599453** both reproduce. Both were verified, not taken on trust.

### 1c. Asymptotic expansion

`n(2^(1/n)-1) = ln2 + (ln2)^2/(2n) + O(1/n^2)`. The leading correction
`(ln 2)^2/2 = 0.2402265069591007`.

| n | `U_lub(n) − ln2` | `(ln2)^2/(2n)` | ratio |
|---|---|---|---|
| 100 | 2.407825111943e-03 | 2.402265069591e-03 | 1.002315 |
| 1 000 | 2.402820207960e-04 | 2.402265069591e-04 | 1.000231 |
| 10 000 | 2.402320505202e-05 | 2.402265069591e-05 | 1.000023 |
| 100 000 | 2.402260017687e-06 | 2.402265069591e-06 | 0.999998 |

All ratios within 1 % of 1. PASS.

### 1d–1f. Sufficiency, exactness, and bound ordering

- EDF `U ≤ 1` is reported as **exact** for implicit deadlines and gives the
  right verdict on a set at `U = 1.000000000000` (feasible) and one at
  `U = 1.200000000000` (infeasible).
- The harmonic set a(T=4,C=1) b(T=8,C=2) c(T=16,C=8) has `U = 1.0`, is
  **rejected by the Liu & Layland bound** (0.7797631496846196 for n = 3) and
  **is nevertheless schedulable** — exact RTA gives R = {a: 1, b: 3, c: 16}
  against D = {4, 8, 16}. This is the check that proves the bound is
  sufficient and not necessary.
- Over a **200 000-set random sweep** of 2–6 task sets, the hyperbolic bound
  accepted **181 017** sets and the Liu & Layland bound **179 801**; the
  number accepted by Liu & Layland but rejected by the hyperbolic bound was
  **0**, as the dominance result requires.

---

## 2. Response-time analysis against hand computation

**Reference.** M. Joseph and P. Pandya, "Finding Response Times in a
Real-Time System", *The Computer Journal* **29**(5) (1986); the iterative
fixed-point form of N. C. Audsley, A. Burns, M. Richardson, K. Tindell and
A. J. Wellings, "Applying new scheduling theory to static priority
pre-emptive scheduling", *Software Engineering Journal* **8**(5) (1993).
Blocking: L. Sha, R. Rajkumar and J. P. Lehoczky, "Priority Inheritance
Protocols: An Approach to Real-Time Synchronization", *IEEE Transactions on
Computers* **39**(9) (1990). Textbook anchor: Buttazzo (2011), Sec. 4.5–4.6
and 7.5.

Recurrence: `R_i = C_i + B_i + sum_{j in hp(i)} ceil(R_i/T_j) * C_j`.
Tolerance on every response time: **1e-12 s absolute**. The full hand
computation for every case is printed in `response_times_output.txt`, and the
complete iterate sequence is compared term by term, not just the fixed point.

### 2.1 Standard set — t1(T=7,C=3) t2(T=12,C=3) t3(T=20,C=5), U = 0.928571428571

| task | R by hand | R computed | D | diff |
|---|---|---|---|---|
| t1 | 3.000000 | 3.000000 | 7 | 0.00e+00 |
| t2 | 6.000000 | 6.000000 | 12 | 0.00e+00 |
| t3 | 20.000000 | 20.000000 | 20 | 0.00e+00 |

Iterates for t3, hand and computed: **5, 11, 14, 17, 20, 20**. t3 meets its
deadline *exactly*, with zero slack — the case a utilization bound can never
resolve.

### 2.2 Harmonic set at U = 1 — a(4,1) b(8,2) c(16,8)

R = {a: 1, b: 3, c: 16}; iterates for c: **8, 12, 15, 16, 16**. All met.

### 2.3 Priority-ceiling blocking turns a met deadline into a missed one

Four tasks t1(7,3) t2(12,3) t3(20,5) t4(50,5); semaphore `s1` used by t3
(0.001 s) and t4 (1.0 s); RM priorities t1=3 > t2=2 > t3=1 > t4=0, hence
ceiling(s1) = 1.

| task | B by hand | B computed |
|---|---|---|
| t1 | 0.0 | 0.0 |
| t2 | 0.0 | 0.0 |
| t3 | 1.0 | 1.0 |
| t4 | 0.0 | 0.0 |

With B_3 = 1 s, iterates for t3 are **6, 12, 15, 21** and R_3 = **21.000000**
against D_3 = 20: slack **−1.000000 s**, deadline **missed**. t4's iterates
are **5, 16, 25, 36, 42, 50, 59**, R_4 = 59 against D_4 = 50 — also missed,
and correctly so, since U = 1.028571 > 1.

### 2.4 Constrained deadlines under deadline-monotonic priorities

a(T=50,C=5,D=10) b(T=20,C=4,D=20): R = {a: 5, b: 9}, iterates for b
**4, 9, 9**. Deadline-monotonic gives `a` the higher priority; rate-monotonic
would have given it to `b`.

---

## 3. Exact percentiles, both definitions named

**Reference.** R. J. Hyndman and Y. Fan, "Sample Quantiles in Statistical
Packages", *The American Statistician* **50**(4), 361–365 (1996),
Definition 7, for the `linear` method. The `nearest_rank` method is the
elementary order-statistic definition.

| check | reference | result | tolerance |
|---|---|---|---|
| `nearest_rank` vs explicit order statistic, 110 comparisons over N ∈ {1,2,3,5,10,13,64,101,1000,20000} | index `k = max(1, ceil(pN/100))` computed in the validation script | worst deviation **0.0e+00** | bit-exact equality required |
| `linear` vs `numpy.percentile(method="linear")`, 110 comparisons | NumPy 2.5.3 | worst deviation **1.084e-19 s** | 1e-16 s |
| 11-row hand-computed table on samples 1…10 | hand, printed in full in the raw output | all PASS | 1e-12 |

Invariants over a 20 000-sample right-skewed synthetic trace: monotone in `p`
across 401 percentiles for both methods; `nearest_rank` always returns a
stored sample; both agree at p = 0 and p = 100; invariant to sample order;
positively homogeneous under a 1e3 scaling. All PASS.

The definitions are shown to **disagree where they must**: p50 on the samples
1…10 is **5.0** under `nearest_rank` and **5.5** under `linear`; on the
20 000-sample trace the p99.5 difference is **+2.413732e-09 s**.

### 3f. A documented sharp edge, not smoothed over

`p` arrives as a binary64 float. The stored value of 99.9 is
99.900000000000005684…, so `p/100 * N` can land just above an integer:

| p | N | `p/100*N` | `ceil` | naive | shifted? |
|---|---|---|---|---|---|
| 99.90 | 20000 | 19980.000000000004 | 19981 | 19980 | **yes, +1 rank** |
| 99.90 | 10000 | 9990.000000000002 | 9991 | 9990 | **yes, +1 rank** |
| 99.90 | 1000 | 999.0000000000001 | 1000 | 999 | **yes, +1 rank** |
| 99.95 | 20000 | 19990.0 | 19990 | 19990 | no |
| 99.00 | 100 | 99.0 | 99 | 99 | no |

No tolerance is applied to hide this. The error is bounded by exactly one
order statistic, it is documented in the `rtclock.histogram` module docstring,
and it is pinned by `test_nearest_rank_float_edge_is_pinned_not_smoothed`.

---

## 4. Loop drift vs injected clock skew

**Injection.** `SkewedSimulatedTimebase.sleep(d)` advances the timestamping
clock by `d*(1 + skew_ppm*1e-6) + wake_delay_s`. Deterministic and
noise-free, so the comparison is against arithmetic.

**Closed forms** (derived in `rtclock.loop`, with `a = T*eps + d`):

```
relative mode:  drift_k = k * a
absolute mode:  drift_k = a (1 - (-eps)^k) / (1 + eps),   k >= 1
```

**Grid:** skews {−1000, −1, 1, 250} ppm × wake delays {0, 1e-7, 5e-6} s ×
periods {1e-4, 1e-3, 2.5e-3, 1e-2, 0.1} s, **500 iterations each, every
iteration compared** — 30 000 comparisons per mode.

**Tolerance** per comparison: `1e-9 * |closed form| + 4*k*ulp(k*T)`, the second
term being binary64 accumulation noise over `k` periods.

| check | comparisons | worst absolute deviation | worst deviation / tolerance |
|---|---|---|---|
| 4a relative mode | 30 000 | **5.709e-13 s** | 7.075e-02 |
| 4b absolute mode | 30 000 | **3.385e-15 s** | 8.762e-02 |

### 4c. Recovering the injected skew from the drift slope

Least-squares slope of drift against iteration index, divided by the period,
over 2000 iterations:

| injected [ppm] | T [s] | recovered [ppm] | relative error |
|---|---|---|---|
| −1000.000 | 0.0010 | −1000.000000020 | 2.04e-11 |
| −10.000 | 0.0010 | −10.000000041 | 4.07e-09 |
| 0.500 | 0.0010 | 0.499999978 | 4.35e-08 |
| 100.000 | 0.0100 | 100.000000032 | 3.17e-10 |
| 1000.000 | 0.0100 | 999.999999987 | 1.34e-11 |

All twelve swept combinations within 1e-6 relative. PASS.

### 4d. Accumulation, over 50 000 iterations at T = 1 ms, skew 200 ppm

`a = T*eps = 2.000000e-07 s`.

| mode | final drift measured | closed form | relative deviation |
|---|---|---|---|
| relative | 9.999800000323e-03 s | 9.999800000000e-03 s | 3.225e-11 |
| absolute | 1.999600058866e-07 s | 1.999600079984e-07 s | 1.056e-08 |

Ratio relative/absolute: **50 009×**. Drift slope: 2.000000000e-07 s/iteration
(relative) against 4.797429630e-16 s/iteration (absolute) — the latter is
roundoff, 2.4e-9 of `a`.

### 4e. A real-clock run on this machine — REPORTED, NOT ASSERTED

300 iterations at a 2 ms period, absolute mode, on the host monotonic clock,
while four build agents shared one core:

| quantity | value |
|---|---|
| measured clock tick | 5.400000e-08 s |
| max \|drift\| | **4.581682e-03 s** ± 5.4e-08 s (clock quantization) |
| final drift | 9.252100e-05 s |
| late releases | 6 of 300 |
| body duration p50 / p99 / max (nearest-rank) | 9.660e-07 / 6.621e-06 / 6.698e-06 s |

A max drift of 4.58 ms on a 2 ms period means the loop was more than two
periods late at its worst point. That is a true statement about a 1-core
container running five Python processes, and it is exactly why this product
does not claim to deliver real-time behaviour. The number will be different
on every run and on every machine; it is published because hiding it would be
worse.

---

## 5. Clock resolution — measured, with its error term

**Method.** N back-to-back reads of `time.monotonic_ns()` into a preallocated
list, then forward differences. The **observable tick** is the smallest
non-zero forward difference; the **per-read cost** is the median forward
difference; the **identical-reading fraction** says whether the clock or the
read loop is the limiting factor.

**Error term.** For a clock of step `q`, treating each endpoint's
quantization as independent and uniform on `[−q/2, q/2]` (JCGM 100:2008, the
GUM, Sec. 4.3.7 and 5.1.2):

```
worst case on a duration   |dt - dt_true| <= q
standard uncertainty       u(dt) = sqrt(2) * q/sqrt(12) = q/sqrt(6)
```

### Reported values for this machine

`time.monotonic` is `clock_gettime(CLOCK_MONOTONIC)`, **advertised**
resolution 1e-09 s.

| samples | measured tick [s] | median per-read [s] | identical fraction |
|---|---|---|---|
| 2 000 | 6.500000e-08 | 1.200000e-07 | 0.000000 |
| 20 000 | 5.800000e-08 | 6.900000e-08 | 0.000000 |
| 100 000 | 5.700000e-08 | 6.700000e-08 | 0.000000 |
| **400 000** | **5.300000e-08** | **6.600000e-08** | 0.000000 |

**Primary reported value: q = 5.300000e-08 s** (53 ns), from 400 000 reads of
`time.monotonic_ns()`.

- worst-case duration error: **± 5.300000e-08 s**
- standard uncertainty `q/sqrt(6)`: **2.163716e-08 s**
- identical-reading fraction **0.000000**, so **the read loop was slower than
  the clock and 53 ns is an UPPER BOUND on the hardware granularity**, not the
  granularity itself. The advertised 1 ns is a claim about representation, not
  about what can be observed from Python.

**Repeatability**, 10 independent measurements of 50 000 samples each:
observable tick min/median/max **5.200000e-08 / 5.400000e-08 / 5.700000e-08 s**,
spread (max−min)/median **0.093**. That spread is the honest statement of how
well this quantity is known on a loaded shared host.

### What the error term costs at the rates this package targets

Using q = 5.300000e-08 s:

| rate [Hz] | period [s] | q/T | q/T [ppm] |
|---|---|---|---|
| 100 | 1.0e-02 | 5.300e-06 | 5.3 |
| 400 | 2.5e-03 | 2.120e-05 | 21.2 |
| 1 000 | 1.0e-03 | 5.300e-05 | 53.0 |
| 10 000 | 1.0e-04 | 5.300e-04 | 530.0 |
| 100 000 | 1.0e-05 | 5.300e-03 | 5300.0 |

At 100 kHz the clock quantization alone is 5300 ppm of the period, so no drift
figure quoted at that rate on this machine can be trusted below that level,
whatever the loop does.

---

## 6. Overrun accounting and the P031 / P036 cross-check

**Convention, stated so a disagreement can only be a real disagreement.** An
iteration overruns when `latency_s > period_s`, **strictly**. Equality is not
an overrun. No tolerance is applied. Indices are **zero-based**. The count is a
pure function of the stored trace and reproduces bit-exactly.

### 6a. Hand computation

Trace (ms): 1.0, 2.0, 2.5, 2.5000001, 3.0, 3.1, 1.0, 9.0, 2.4999999, 2.5
against a 2.5 ms period.

| quantity | by hand | computed |
|---|---|---|
| count | 4 | 4 |
| indices | [3, 4, 5, 7] | [3, 4, 5, 7] |
| longest consecutive run | 3 | 3 |
| worst overshoot | 6.5 ms | 6.5000000 ms |

### 6b. The strict-inequality boundary at one ulp

| latency | overruns | expected |
|---|---|---|
| exactly 2.5e-03 s | 0 | 0 |
| `nextafter(2.5e-03, +inf)` | 1 | 1 |
| `nextafter(2.5e-03, 0)` | 0 | 0 |

One ulp at the period: 4.336809e-19 s.

### 6c. Independent recomputation

On the cross-check trace, `rtclock`'s accounting and a separate NumPy
expression (`(arr > period).sum()`) in the validation script give **30 and 30**,
with **identical index lists**.

### 6d. Cross-check with P031 HilForge — AGREEMENT

`products/P031/validation/crosscheck_overruns.json` **was present** when this
validation last ran. P036 read its `trace` (1200 samples) and `period_s`
(1.0e-02 s), ran its own accounting over them, and wrote the independent
result to `validation/crosscheck_overruns.json`. **P031's file was not
edited.**

| field | P036 (rtclock) | P031 (hilforge) |
|---|---|---|
| trace length | 1200 | 1200 |
| period_s | 1.0e-02 | 1.0e-02 |
| direct overrun_count | **176** | **176** |
| overrun indices | identical, element by element | identical |
| overrun fraction | 0.146667 | — |
| longest consecutive run | 4 | — |
| worst overshoot | 2.817417400000e-02 s | — |

The two implementations were written independently and agree on the count and
on every index. The convention they share — strict `>`, zero-based indices,
no tolerance — is the reason a disagreement would have meant something.

### 6e. The second accounting P031 publishes — CASCADE overruns, also agreeing

P031's file also carries a *cascade* count under a different and genuinely
different question: not "did this iteration's duration exceed the budget" but
"did this iteration finish late once earlier overruns had pushed its start":

```
r[i] = i*T ;  s[i] = max(r[i], c[i-1]) ;  c[i] = s[i] + d[i] ;  c[-1] = 0
overrun iff c[i] > r[i] + deadline
```

`rtclock.loop.FixedRateLoop` on a `SimulatedTimebase` *implements* that
recurrence — release at `t0 + k*T`, wait skipped when already past it — so
driving the loop with P031's trace reproduces the quantity rather than
reimplementing P031's expression:

| quantity | P036 (FixedRateLoop) | P031 (hilforge) |
|---|---|---|
| cascade overrun count | **307** | **307** |
| cascade indices | identical, element by element | identical |
| longest consecutive cascade | **16** | **16** |
| late releases reported by the loop driver | 307 of 1200 | — |

Both cross-checks PASS. The trace is synthetic and injected (P031's note says
so explicitly); neither product measured hardware.

---

## Checks that are not here, and why

- **No assertion on achieved real-time performance.** The build machine has
  one core and four concurrent agents. Any wall-clock bound would be a
  statement about the neighbours, not about this code. 4e and the tail of
  `examples/loop_drift.py` report host numbers and assert nothing.
- **No processor-demand criterion.** The exact EDF test for constrained
  deadlines is not implemented; `edf_test` returns `strength="sufficient"` in
  that case and says so in `detail`.
- **No level-i busy-period analysis.** The response-time recurrence is the
  exact worst case only while `R_i <= T_i`. Past that point the package
  reports the recurrence value, flags the deadline miss, and documents that
  the number is not the exact worst case (Lehoczky 1990; Buttazzo 2011
  Sec. 4.6).
- **No multiprocessor analysis.** Everything here is single-processor.
