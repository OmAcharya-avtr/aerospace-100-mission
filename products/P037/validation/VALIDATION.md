# framesync 0.1.0 — validation evidence (Level 2, research-grade)

Every number in this file and in `README.md` was produced by running the
scripts in this directory in the build session on **2026-10-05**, on Python
3.13.16 with numpy 2.5.3, scipy 1.18.1, reedsolo 1.7.0, on **one CPU core
shared with four other concurrent build agents**. Each script writes its raw
stdout to the `*_output.txt` file next to it, and each is rerunnable with
`python3 <script>.py` from this directory.

| Script | Raw output | Wall time | Result |
|---|---|---|---|
| `validate_uncoded_fer.py` | `validate_uncoded_fer_output.txt` | 9.9 s | PASS |
| `validate_rs_correction.py` | `validate_rs_correction_output.txt` | 1.8 s | PASS |
| `validate_false_sync.py` | `validate_false_sync_output.txt` | 0.7 s | PASS |
| `validate_state_machine.py` | `validate_state_machine_output.txt` | 0.2 s | PASS |
| `validate_rs_coding_gain.py` | `validate_rs_coding_gain_output.txt`, `rs_coding_gain.json` | 44.3 s | PASS (one check NOT VERIFIED, see §5) |
| `validate_conv_fer.py` | `validate_conv_fer_output.txt` | 17.6 s | PASS |
| `crosscheck_berbench.py` | `crosscheck_berbench_output.txt`, `crosscheck_berbench.json` | 3.0 s | PASS |

Total validation wall time **≈ 78 s**, every script inside the 60 s
per-script budget. The slowest is the Reed-Solomon sweep, bounded by
`reedsolo`'s pure-Python decoder. Wall times vary by a factor of about two
with how many other build agents are sharing the core; the numbers above are
from the final run, and every *result* is seed-fixed and reproduces exactly
regardless of timing.

## Conventions, stated once

`ebn0_db` is **energy per information bit** over noise spectral density,
Eb/N0, in dB. A code of rate R puts Es/N0 = R·Eb/N0 on the channel, so the
channel bit error probability is

    p = Q(sqrt(2 · R · Eb/N0))                                    (Eq. 1)

Modulation is coherent BPSK over memoryless AWGN with ideal synchronisation
and no implementation loss. This is the same Eb/N0 convention and the same
BPSK expression as P010 BERBench; §7 is the cross-check that holds the two
products together.

---

## 1. Uncoded frame error rate against the analytic expression

`validate_uncoded_fer.py`. Frame: 1115 octets + 2-octet Frame Error Control
Field = **8936 bits**. Checked against

    FER = 1 − (1 − p)^n                                          (Eq. 11)

Acceptance: measured within 4 binomial standard errors of Eq. (11). Frame
counts sized for a 5 % relative standard error, clamped to [400, 12000].

| Eb/N0 (dB) | p_bit | N | errors | FER measured | stderr | FER Eq. (11) | dev/σ | result |
|---|---|---|---|---|---|---|---|---|
| 7.00 | 7.7267e-04 | 400 | 400 | 1.0000e+00 | 2.500e-03 | 9.9900e-01 | 0.40 | PASS |
| 8.00 | 1.9091e-04 | 400 | 329 | 8.2250e-01 | 1.910e-02 | 8.1843e-01 | 0.21 | PASS |
| 9.00 | 3.3627e-05 | 1142 | 297 | 2.6007e-01 | 1.298e-02 | 2.5955e-01 | 0.04 | PASS |
| 10.00 | 3.8721e-06 | 11362 | 368 | 3.2389e-02 | 1.661e-03 | 3.4009e-02 | 0.98 | PASS |
| 11.00 | 2.6131e-07 | 12000 | 31 | 2.5833e-03 | 4.634e-04 | 2.3323e-03 | 0.54 | PASS |

**Known-answer anchor:** CRC-16 of `b"123456789"` = **0x29B1**, the published
check value of the CRC-16/IBM-3740 parameter set (poly 0x1021, init 0xFFFF,
no reflection, no final XOR), which is the CCSDS 132.0-B Frame Error Control
Field polynomial g(x) = x¹⁶ + x¹² + x⁵ + 1 with an all-ones preload. PASS.

**Frame Error Control Field detection gap (reported, not a pass criterion):**
of 1425 frames carrying at least one bit error, the FECF rejected **1425**
and missed **0**. The 2⁻¹⁶ = 1.526e-05 asymptote predicts 0.022 misses over
that sample, so zero is the expected outcome and the sample is far too small
to measure the asymptote. Most frames here carry a single bit error, which a
CRC-16 always detects, so the measured gap is expected to sit *below* 2⁻¹⁶.

## 2. RS(255,223) corrects 16 symbol errors and fails at 17

`validate_rs_correction.py`. n = 255, k = 223, d_min = n − k + 1 = **33**,
E = (d_min − 1)/2 = **16**. 12 seeded patterns per error count — four
structured families (random positions, first E, last E, evenly spaced) across
three seeds — with non-zero error values so every chosen position is a real
symbol error.

| error count | patterns | corrected | declared failure | wrong message | result |
|---|---|---|---|---|---|
| 0 – 16 | 12 each | 12 each | 0 | 0 | PASS |
| 17 | 12 | **0** | 12 | 0 | PASS |
| 18 | 12 | 0 | 12 | 0 | PASS |
| 19 | 12 | 0 | 12 | 0 | PASS |
| 20 | 12 | 0 | 12 | 0 | PASS |

No miscorrection was observed: `reedsolo` declared failure on every pattern
beyond the radius rather than returning a wrong message. That is what
Eq. (9) of `framesync.rs` assumes.

Interleaved codeblocks, I ∈ {1, 3, 5}: 16 errors in **every** codeword →
frame recovered, 0 failed codewords. 17 errors in codeword 0 → frame lost,
1 failed codeword. PASS for all three depths.

## 3. False-sync probability against the combinatorial expression

`validate_false_sync.py`. Derivation: a correlation detector declares the
marker when the window Hamming distance D ≤ T. For an i.i.d. uniform bit
stream each of the L comparisons mismatches with probability 1/2,
independently, so D ~ Binomial(L, 1/2) and

    P_fa(T) = P[D ≤ T] = 2^(−L) · Σ_{k=0..T} C(L, k)              (Eq. 3)

**Check 1** — exact expression against `scipy.stats.binom.cdf(T, 32, 0.5)`
over every T from 0 to 32: worst relative difference **0.000e+00**
(tolerance 1e-12). PASS.

Exact values at operational thresholds, with the expected false-sync count
over a 10⁹-bit stream (linearity of expectation over the N − L + 1 window
positions; overlapping windows are correlated so only the expectation is
claimed):

| T | P_fa(T) | expected in 1 Gbit |
|---|---|---|
| 0 | 2.328306e-10 | 2.328306e-01 |
| 1 | 7.683411e-09 | 7.683411e+00 |
| 2 | 1.231674e-07 | 1.231674e+02 |
| 3 | 1.278007e-06 | 1.278007e+03 |
| 4 | 9.650597e-06 | 9.650597e+03 |

**Check 2** — Monte Carlo over 2,000,000 random window positions. Band is
6σ of the binomial standard error because overlapping windows are
correlated; the widening is for the correlation, not to accommodate a miss.

| T | hits | measured | Eq. (3) | σ | dev/σ | result |
|---|---|---|---|---|---|---|
| 8 | 7065 | 3.532500e-03 | 3.500183e-03 | 4.1761e-05 | 0.77 | PASS |
| 10 | 50022 | 2.501100e-02 | 2.505123e-02 | 1.1051e-04 | 0.36 | PASS |
| 12 | 215529 | 1.077645e-01 | 1.076636e-01 | 2.1917e-04 | 0.46 | PASS |
| 14 | 596164 | 2.980820e-01 | 2.983074e-01 | 3.2351e-04 | 0.70 | PASS |
| 16 | 1139000 | 5.695000e-01 | 5.699750e-01 | 3.5007e-04 | 1.36 | PASS |

**Check 3** — exhaustive, not statistical: for a 16-bit marker the whole
2¹⁶ window space is enumerated and counted. All 17 thresholds match
`2¹⁶ · P_fa(T)` exactly (T = 0 → 1, T = 1 → 17, T = 2 → 137, T = 3 → 697,
T = 4 → 2517, T = 8 → 39203, T = 16 → 65536). PASS.

**Marker autocorrelation (reported):** minimum Hamming distance over the 31
non-zero cyclic shifts of 0x1ACFFC1D is **12 bits**, mean 15.94 bits. A
random 32-bit pattern would average 16.

## 4. Acquisition state machine reproduces a hand trace exactly

`validate_state_machine.py`. The full 18-step hand derivation, with counter
values at every step, is a comment at the top of `tests/test_sync.py`.
Configuration: search_tolerance 0, lock_tolerance 3, check_required 2,
flywheel_max 3.

Result: **all 18 states and all 18 miss-counter values match the hand trace
exactly**, and the number of observations in a delivering state (LOCK or
FLYWHEEL) is 12, which is what the hand trace says. PASS.

The sequence exercises every transition: a false start that drops out of
CHECK, promotion to LOCK, a single-miss flywheel entry with recovery, a
two-miss flywheel with recovery, a three-miss flywheel that expires to
SEARCH, and re-acquisition.

**Machine driven by real correlator output** (reported, not a pass
criterion), 60 marker-prefixed frames per point:

| Eb/N0 (dB) | p | observations in LOCK | final state |
|---|---|---|---|
| −2.00 | 1.3064e-01 | 1 / 60 | SEARCH |
| 0.00 | 7.8650e-02 | 45 / 60 | LOCK |
| 2.00 | 3.7506e-02 | 59 / 60 | LOCK |
| 6.00 | 2.3883e-03 | 60 / 60 | LOCK |
| 12.00 | 9.0060e-09 | 60 / 60 | LOCK |

The marker survives far below the data threshold because the lock tolerance
is 3 bits in 32: at 6 dB the expected marker errors are 32p = 0.08 bits, so
the synchroniser holds lock in a region where the frame error rate is 1.

**Slip:** across insertions of +3 and +1 bits and deletions of −2 and −7
bits before frame 7 of 16, the synchroniser delivers exactly **4 frames at
the wrong phase** and then re-enters SEARCH, for all four slip sizes. With
`flywheel_max = 4` that is the hand expectation: one flywheel entry plus
three further misses. PASS.

## 5. RS(255,223) coding gain at 10⁻⁵

`validate_rs_coding_gain.py`, raw output `validate_rs_coding_gain_output.txt`,
JSON `rs_coding_gain.json`.

**First, the semi-analytic curve is validated where it can be measured.**
Monte Carlo with the real `reedsolo` decoder against Eqs. (7)/(8), I = 5,
acceptance 4σ:

| Eb/N0 (dB) | p_bit | N | errors | FER measured | stderr | FER Eq. (8) | dev/σ | result |
|---|---|---|---|---|---|---|---|---|
| 5.25 | 7.7505e-03 | 100 | 90 | 9.0000e-01 | 3.000e-02 | 9.0209e-01 | 0.07 | PASS |
| 5.50 | 6.3668e-03 | 150 | 86 | 5.7333e-01 | 4.038e-02 | 5.2463e-01 | 1.21 | PASS |
| 5.75 | 5.1755e-03 | 200 | 34 | 1.7000e-01 | 2.656e-02 | 1.5398e-01 | 0.60 | PASS |
| 6.00 | 4.1607e-03 | 250 | 9 | 3.6000e-02 | 1.178e-02 | 2.4350e-02 | 0.99 | PASS |

Reaching FER 10⁻⁵ by Monte Carlo would need of order 10⁶ frames through a
pure-Python Reed-Solomon decoder, which is not affordable on this hardware.
The gain at 10⁻⁵ is therefore obtained by **inverting the semi-analytic
curve validated above**, not by direct measurement, and that is stated
wherever the number appears.

**Uncoded anchor:** Eb/N0 for uncoded BPSK at BER 10⁻⁵ = **9.5879 dB**
against the textbook-quoted ~9.6 dB for coherent BPSK [Proakis & Salehi
2008, Sec. 4.3; Sklar 2001, Sec. 4.7.1]. P010 BERBench records the same
inversion at 9.5879 dB in `validation/bpsk_textbook_output.txt`. Within
0.05 dB. PASS.

**Measured gain:**

| metric | uncoded Eb/N0 (dB) | RS Eb/N0 (dB) | gain (dB) |
|---|---|---|---|
| output BER = 10⁻⁵ | 9.5879 | 6.2938 | **3.2941** |
| FER = 10⁻⁵ (n = 8936 bits, I = 5) | 12.5230 | 6.7023 | **5.8207** |

The two differ because an 8936-bit frame needs a far lower bit error rate to
reach 10⁻⁵ at the frame level than a single bit does. Both are in Eb/N0 per
information bit, so the 0.5824 dB rate loss of the coded link is already
inside the numbers.

**Reference check that passed:** asymptotic coding gain of a hard-decision
bounded-distance decoder, G_∞ = 10 log₁₀(R(E+1)) = 10 log₁₀(0.874510 × 17)
= **11.7221 dB** [standard result; Lin & Costello 2004, *Error Control
Coding* 2nd ed., coding-gain discussion; Sklar 2001, *Digital
Communications* 2nd ed., Ch. 8]. Real coding gain at a finite error rate
must be positive and strictly below G_∞. Both measured gains satisfy
0 < gain < 11.7221 dB. PASS.

### Reference check that is NOT VERIFIED — recorded as a deviation

The build specification requires the measured gain to fall "in the range the
standard references quote, with the reference cited". The performance curves
for the CCSDS Reed-Solomon code are carried by **CCSDS 130.1-G**, *TM
Synchronization and Channel Coding — Summary of Concept and Rationale*. This
build container has no network access and that document was not available to
be read.

**No numeric range is quoted from it here.** Attaching a half-remembered
figure to a page number would be an invented citation, which the build guide
forbids outright, and quoting a range that the measured value happens to fall
inside would amount to inventing a tolerance. The comparison against a
published range therefore **remains open**, is reported as NOT VERIFIED in
the script output and in `rs_coding_gain.json`
(`"reference_range_status": "NOT_VERIFIED_NO_DOCUMENT_ACCESS"`), and is
listed in README Limitations. A reviewer with access to CCSDS 130.1-G can
close it by comparing against 3.2941 dB (BER metric) and 5.8207 dB (FER
metric) as recorded above, which is the reason both numbers are stated with
their definitions rather than a single headline figure.

**Convolutional leg:** free distance computed from the trellis by
shortest-weight search: **d_free = 10**, matching the literature value for
the (171, 133) rate-1/2 K = 7 code. Unchanged at 10 with the G2 inversion
switched off, as it must be, since the inversion is a fixed XOR. PASS.

## 6. Convolutionally coded frame error rate

`validate_conv_fer.py`. CCSDS (171, 133) rate-1/2, K = 7, 64 states, G2
inverted. Frame: 512 information bits + 6 tail bits → 1036 channel bits.
No exact frame-error expression exists for this code and none is claimed, so
the checks are structural and ordinal.

| Eb/N0 (dB) | mode | p_chan | N | errors | FER | uncertainty | decoded BER |
|---|---|---|---|---|---|---|---|
| 2.00 | hard | 1.0403e-01 | 300 | 295 | 9.8333e-01 | ±7.391e-03 | 1.0513e-01 |
| 2.00 | soft | 1.0403e-01 | 300 | 77 | 2.5667e-01 | ±2.522e-02 | 4.3685e-03 |
| 3.00 | hard | 7.8896e-02 | 300 | 239 | 7.9667e-01 | ±2.324e-02 | 2.9876e-02 |
| 3.00 | soft | 7.8896e-02 | 300 | 9 | 3.0000e-02 | ±9.849e-03 | 3.3203e-04 |
| 4.00 | hard | 5.6495e-02 | 400 | 131 | 3.2750e-01 | ±2.347e-02 | 4.9268e-03 |
| 4.00 | soft | 5.6495e-02 | 400 | 1 | 2.5000e-03 | ±2.497e-03 | 1.9531e-05 |
| 5.00 | hard | 3.7679e-02 | 500 | 21 | 4.2000e-02 | ±8.971e-03 | 3.9844e-04 |
| 5.00 | soft | 3.7679e-02 | 500 | 0 | 0.0000e+00 | <6.000e-03 (95 % one-sided) | 0.0000e+00 |

Zero-error points carry the Poisson rule-of-three one-sided 95 % upper limit
3/N instead of a binomial standard error, which would be 0 and uninformative.

- Frame error rate monotone decreasing in Eb/N0, both modes: PASS.
- Soft decision never worse than hard at the same Eb/N0, ratios 3.83 →
  26.56 → 131.00 → ∞ over 2 to 5 dB: PASS. The separation is **reported, not
  asserted at a dB figure**: these frame counts do not resolve one.
- Soft-decision coded frame error rate below the uncoded rate for the same
  512-bit payload at every point: PASS.
- Noiseless round-trip exact over 16 frames: PASS.

## 7. Mandatory cross-check against P010 BERBench

`crosscheck_berbench.py`, raw output `crosscheck_berbench_output.txt`,
machine-readable `crosscheck_berbench.json`.

Channel: coherent BPSK, AWGN, uncoded, Eb/N0 per bit, no fading. P010's
expression, from `P010/src/berbench/analytic.py`, is Pb = Q(sqrt(2γ)) with
γ = 10^(dB/10) [Proakis & Salehi 2008, Eq. (4.3-13)]. It is **recomputed
here from that formula via `scipy.stats.norm.sf`**, never imported —
cross-product imports are forbidden, and importing would make the comparison
circular. framesync's own path uses `scipy.special.erfc`.

**Step 1 — P010's expression against P010's own committed numbers:**

| Eb/N0 | P010 recorded | recomputed | rel diff | framesync | rel diff | result |
|---|---|---|---|---|---|---|
| 8.0 dB | 1.909100e-04 | 1.909078e-04 | 1.17e-05 | 1.909078e-04 | 0.00e+00 | PASS |
| 10.0 dB | 3.872108e-06 | 3.872108e-06 | 1.23e-10 | 3.872108e-06 | 3.72e-15 | PASS |

The 1.17e-05 residual at 8 dB is the five significant figures P010 printed in
its `VALIDATION.md` table, not a disagreement.

**Step 2 and 3 — conversion through the frame/bit relation and comparison
against a measured frame error rate.** Eb/N0 = 9.00 dB, n = 8936 bits,
20,000 Monte Carlo frames, seed 20261005:

| quantity | value |
|---|---|
| P010 BER (its formula, independent `norm.sf` path) | 3.362722842e-05 |
| framesync BER (its formula, `erfc` path) | 3.362722842e-05 |
| relative difference in BER | 2.821e-15 |
| **FER from P010's BER via FER = 1 − (1 − Pb)ⁿ** | **2.595505895e-01** |
| **framesync measured FER (20,000 frames, 5130 errors)** | **2.565000000e-01** |
| binomial standard error on the measurement | 3.087942276e-03 |
| **relative difference** | **1.175e-02** |
| deviation | **0.988 σ** |

Acceptance: relative difference < 5 % and deviation < 4σ. **Both PASS.**

**Step 4 — inverted anchor:** framesync gives Eb/N0 = 9.5879 dB at BER 10⁻⁵;
P010 records 9.5879 dB. Difference 4.17e-05 dB. PASS.

**Finding: NO DISAGREEMENT.** framesync's BER expression reproduces P010
BERBench's recorded BPSK/AWGN values to better than 1e-4 relative, limited
only by P010's printed precision; the inverted Eb/N0 at BER 10⁻⁵ agrees to
4e-05 dB; and framesync's measured frame error rate sits 0.99 binomial
standard errors from the value obtained by converting P010's bit error rate
through the analytic frame/bit relation. The two products are consistent.

---

## What is not validated

- **No comparison against a published RS coding-gain range** (§5). Open.
- **No bit-level CCSDS interoperability.** `reedsolo` implements RS(255,223)
  in the conventional basis with its own generator polynomial; CCSDS
  specifies a dual basis. Same code, same E = 16, same statistics, but not
  wire-compatible with a flight decoder. Nothing here was checked against a
  CCSDS reference codeblock.
- **No turbo, LDPC or the longer turbo ASMs.** Only the 32-bit marker.
- **No implementation loss, no fading, no burst errors, no phase or timing
  error.** The channel is ideal coherent BPSK over memoryless AWGN
  throughout, so every number is an upper bound on real performance.
- **The convolutional soft/hard separation is not resolved to a dB figure**;
  the frame counts affordable on one contended core do not support one.
- **Equations (9) and (10)** (post-decoding symbol and bit error rate) are
  approximations, not exact, and are labelled as such wherever printed. They
  rise above the channel bit error rate below about 5.46 dB; see README
  Limitations.

## Reproducing every number

```bash
cd validation
PYTHONPATH=../src python3 validate_uncoded_fer.py
PYTHONPATH=../src python3 validate_rs_correction.py
PYTHONPATH=../src python3 validate_false_sync.py
PYTHONPATH=../src python3 validate_state_machine.py
PYTHONPATH=../src python3 validate_rs_coding_gain.py
PYTHONPATH=../src python3 validate_conv_fer.py
PYTHONPATH=../src python3 crosscheck_berbench.py
```

All seeds are fixed in the scripts (base 20261005), so every number above is
reproducible exactly.
