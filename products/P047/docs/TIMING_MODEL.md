# Timing model: every derivation this package relies on

Each section is referenced by name from the docstring that uses it. Nothing here
is quoted from a page of a textbook; where a classical result is used, it is
either derived below or checked numerically by a script in `validation/`, and the
script is named.

Units throughout: time in **symbol periods** (or **slot periods** for PPM),
frequency in **cycles per symbol**, angular frequency in **radians per symbol**.
Timing errors and jitter are in symbol periods and squared symbol periods.

---

## 1. The signal and noise model

The received waveform is

```
x(t) = sum_m  a_m  p(t - m - tau)
```

with `a_m` the data symbols, `p` a finite-support pulse shape normalised to
`p(0) = 1`, and `tau` the transmitter's timing offset. Samples are evaluated
analytically at the instants the receiver chooses; no dense waveform is generated
and no interpolation is used in the open-loop measurements.

Additive noise is independent and Gaussian **per physical sample instant**, with
standard deviation `sigma = 10**(-SNR_dB / 20)`. Because consecutive detector
updates re-use samples (a strobe is both the current strobe of symbol `k` and the
previous strobe of symbol `k+1`), the simulation draws one noise value per
instant and maps taps onto instants; `slotsync.ted.ted_sample_slots` does that
bookkeeping and `tests/test_ted.py` fixes the mapping for each detector.

What this model does **not** contain: a receive filter, so there is no filter
noise colouring between distinct instants; dispersion; shot noise or any other
signal-dependent optical noise; and any amplitude or frequency offset. Those are
limitations, listed in the README.

---

## 2. S-curves and the detector gain

The S-curve is the mean detector output at a held timing error,
`S(eps) = E[e | eps]`, averaged over the data. Because every pulse shape has
finite support, only a finite window of symbols can reach the detector's samples,
so the expectation is computed by **enumerating every data pattern in that
window** rather than by Monte Carlo. The detector gain is

```
K_d = dS/d(eps) at eps = 0
```

in detector output per symbol period of timing error. It is measured two ways -
a least-squares line through the origin over `|eps| <= 0.05`, and a central
difference at the grid spacing - and a large disagreement between them is the
package's signal that the S-curve is not differentiable at the origin.

### 2.1 Early-late gate on a triangular pulse: `K_d = 2`

With `p(t) = 1 - |t|` on `|t| <= 1` and gates at `eps -+ d`,

```
e(eps) = p(eps - d) - p(eps + d)
       = [1 - (d - eps)] - [1 - (d + eps)]      for |eps| < d
       = 2 eps
```

so `K_d = 2` for every gate spacing `d` in `(0, 1/2]`, exactly and independently
of `d`. Measured to 1.3e-15 absolute by
`validation/validate_scurve_gains.py` over four gate spacings, with the full
32-pattern ensemble average rather than the isolated-symbol case.

### 2.2 Mueller-Mueller: the S-curve depends only on `h(+-1)`

For data that is independent with `E[a] = 0` and `E[a^2] = s2`, the detector
`e[k] = a[k] x[k-1] - a[k-1] x[k]` has mean

```
E[a[k] x[k-1]] = s2 * h(eps - 1)
E[a[k-1] x[k]] = s2 * h(eps + 1)
S(eps)         = s2 * ( h(eps - 1) - h(eps + 1) )
```

because the only term of `x[k-1] = sum_m a_m h(k - 1 - m + eps)` correlated with
`a[k]` is `m = k`. For the triangle this gives `S(eps) = s2 * eps` and
`K_d = s2`, i.e. exactly 1 for antipodal data.

For **unipolar OOK** the data mean is not zero and the cross terms survive.
Writing `S_all(eps) = sum_j h(j + eps)` over all integers `j`, with `E[a] = 1/2`
and `E[a^2] = 1/2`:

```
E[a[k] x[k-1]] = (1/2) h(eps - 1) + (1/4)[ S_all(eps) - h(eps - 1) ]
E[a[k-1] x[k]] = (1/2) h(eps + 1) + (1/4)[ S_all(eps) - h(eps + 1) ]
S(eps)         = (1/4) ( h(eps - 1) - h(eps + 1) )
```

The `S_all` terms cancel exactly, so there is **no lock-point bias** and the gain
is a quarter of the antipodal one. Measured as 0.250000 with a bias of 0.0e+00.

### 2.3 Mueller-Mueller on a Nyquist raised cosine: `K_d = 2 cos(pi a) / (1 - 4 a^2)`

The frequency-domain raised cosine is
`h(t) = sinc(t) cos(pi a t) / (1 - (2 a t)^2)`. It is even, so from §2.2

```
K_d = S'(0) = h'(-1) - h'(1) = -2 h'(1).
```

At `t = 1`, `sinc(1) = 0`, so in the product rule only the term that
differentiates the sinc survives:

```
h'(1) = sinc'(1) * cos(pi a) / (1 - 4 a^2),   sinc'(1) = -1
```

(`sinc'(t) = d/dt [sin(pi t) / (pi t)]` evaluated at `t = 1` is
`[pi cos(pi) * pi - sin(pi) * pi] / pi^2 = -1`). Hence

```
K_d = 2 cos(pi a) / (1 - 4 a^2)
```

which is exactly `2` at `a = 0` and, by l'Hopital at the removable singularity
`a = 1/2`, exactly `pi / 2`. Measured to 2.9e-06 relative over five rolloffs.

### 2.4 PPM slot early-late on a half-sine slot: `S(eps) = sin(2 pi eps)`

With a half-sine slot pulse `p(t) = cos(pi t)` on `|t| <= 1/2` - zero at the slot
edges, so no slot overlaps another - and gates at `-+1/4` slot, the squared
early-late output on the slot carrying the pulse is

```
e(eps) = cos^2(pi(eps - 1/4)) - cos^2(pi(eps + 1/4))
       = [ cos(2 pi eps - pi/2) - cos(2 pi eps + pi/2) ] / 2
       = [ sin(2 pi eps) + sin(2 pi eps) ] / 2
       = sin(2 pi eps)
```

valid for `|eps| <= 1/4`, where both gates stay inside the pulse. Hence
`K_d = 2 pi` exactly, the S-curve peaks at a quarter slot, and its useful range
is a quarter slot. Measured residual 2.7e-15 and gain error 6.6e-08.

---

## 3. The second-order loop and its noise bandwidth

The analogue prototype is

```
H(s) = (2 zeta w_n s + w_n^2) / (s^2 + 2 zeta w_n s + w_n^2)
```

and its one-sided noise bandwidth, in hertz with `w_n` in rad/s, is

```
B_L = int_0^inf |H(j 2 pi f)|^2 df = (w_n / 2)(zeta + 1 / (4 zeta))
    = w_n (1 + 4 zeta^2) / (8 zeta).
```

This is the classical second-order-loop result and it is **verified rather than
quoted**: `slotsync.loop.noise_bandwidth_numeric` evaluates the integral by
quadrature with an analytic tail correction, and
`validation/validate_loop_coefficients.py` compares the two over 18 points,
worst relative error 5.3e-09. The first draft of this package carried the same
expression with a spurious factor of `2 pi`; that check is what caught it.

Normalised to the symbol period, with `theta = w_n T`:

```
B_n = B_L T = theta (1 + 4 zeta^2) / (8 zeta),    theta = 8 zeta B_n / (1 + 4 zeta^2).
```

The tail correction: for large `w` the integrand tends to `(2 zeta theta / w)^2`,
whose tail beyond `f_max` integrates to `(2 zeta theta)^2 / (4 pi^2 f_max)`.
Without it, truncating at `limit` multiples of `theta / (2 pi)` leaves a relative
error of `16 zeta^3 / (pi * limit * (1 + 4 zeta^2))`, which is 6.0e-03 at
`limit = 400, zeta = 2` - large enough to be mistaken for a wrong closed form.

---

## 4. The discrete loop

### 4.1 Coefficients by exact pole matching

The loop updates, with `e` the detector output, `w` the integrator state and
`tau_hat` the timing estimate:

```
w       <- w - k2 e
tau_hat <- tau_hat - k1 e + w
```

Linearising with `e = K_d eps + n` and writing `alpha = k1 K_d`,
`beta = k2 K_d`, the error state `[eps, w]` evolves as

```
eps[k+1] = (1 - alpha - beta) eps[k] + w[k] - (k1 + k2) n[k]
w[k+1]   = -beta eps[k]            + w[k] - k2 n[k]
```

so

```
A = [[1 - alpha - beta, 1], [-beta, 1]],    B = [[-(k1 + k2)], [-k2]]
```

with characteristic polynomial `z^2 - (2 - alpha - beta) z + (1 - alpha)`.
Matching it term by term to the exponential image of the analogue poles,
`z^2 - 2 exp(-zeta theta) cos(theta sqrt(1 - zeta^2)) z + exp(-2 zeta theta)`:

```
alpha = 1 - exp(-2 zeta theta)
beta  = 2 - alpha - 2 exp(-zeta theta) cos(theta sqrt(1 - zeta^2))
```

with `cos` replaced by `cosh(theta sqrt(zeta^2 - 1))` when `zeta >= 1`. The
discrete poles are then **exactly** `exp(s_i T)`; measured worst error 8.3e-09
over 32 designs. Dividing by `K_d` to get `k1, k2` makes the closed-loop
dynamics independent of the detector, which is the whole reason `K_d` has to be
measured.

The inverse is closed form: `u = zeta theta = -log(1 - alpha) / 2` from the
constant term, then `c = (2 - alpha - beta) / (2 exp(-u))` gives
`theta = sqrt(u^2 + arccos(c)^2)` (or `sqrt(u^2 - arccosh(c)^2)` when `c >= 1`)
and `zeta = u / theta`. Round-trip error 9.8e-11.

### 4.2 Jitter variance, three ways

**White-noise closed form.** Treating the detector noise as white with variance
`sigma_n^2` at the symbol rate, its two-sided power spectral density is
`sigma_n^2 T`, the error transfer is `eps = -(1/K_d) H n`, and

```
sigma_eps^2 = (sigma_n^2 T / K_d^2) * 2 B_L = 2 B_n sigma_n^2 / K_d^2.
```

**Exact discrete, still white.** Solve `P = A P A' + B B' sigma_n^2` for the
stationary covariance and take `P[0,0]`. This drops the small-bandwidth
approximation only. The ratio to the closed form is 1.0004 at `B_n = 0.0005`,
1.009 at 0.01 and 1.093 at 0.1 for `zeta = 1/sqrt(2)`.

**Coloured.** Detector self-noise is not white. Writing the noise-to-error
impulse response `h[m] = C A^(m-1) B` and the measured autocovariance `R_n[j]`,

```
sigma_eps^2 = sum_{j=-J}^{J} R_n[j] rho[j],    rho[j] = sum_m h[m] h[m+j].
```

The lagged products have the closed form `rho[j] = C P (A')^j C'` with `P` the
unit-variance Lyapunov solution, which avoids truncating an impulse response
thousands of samples long. Setting `R_n[j] = 0` for `j != 0` recovers the exact
white result exactly, which `tests/test_loop.py` checks.

---

## 5. Cycle slips

A slip is the loop losing a whole symbol. The classical estimate treats the
timing error as a stationary Gaussian process and counts crossings of the
S-curve's reversal point `a`, using the standard level-crossing rate

```
nu = (1 / 2 pi) (sigma_dot / sigma) exp(-a^2 / (2 sigma^2))
```

with the one-step difference `eps[k+1] - eps[k]` standing in for the derivative;
the total rate is `2 nu` for the two boundaries `+-a`. Its three approximations:
Gaussianity, every crossing counting as a slip, and the discrete difference
standing in for a derivative.

**This estimate fails by 6 to 16 orders of magnitude** against the measured slip
rate (`validation/validate_cycle_slips.py`). It fails because the mechanism is
not a Gaussian excursion over a fixed barrier. §4 of that script measures both
halves of the real mechanism: as the timing error grows, the S-curve's local gain
*falls* (1.571 to 1.246 from 0 to 0.45 symbol) while the detector output variance
*rises* (by a factor of 2.24 over the same range). The restoring force weakens and
the disturbance strengthens together, so the escape is a positive-feedback runaway.

What is usable instead is the measured threshold. Across loop bandwidths from
0.005 to 0.05, the slip rate crosses 1e-04 per symbol at a predicted **rms timing
error of 0.063 to 0.079 symbol** - nearly independent of the bandwidth. The design
rule this package supports is therefore a jitter rule, not a bandwidth rule.

---

## 6. The static lock offset from self-noise

A type-2 loop drives the **time average of the detector output** to zero, not the
time average of the timing error. When the detector's output noise is correlated
with the data the loop has already tracked - which is what self-noise is - these
are not the same condition, and the loop locks away from the true symbol centre.

The offset is invisible in the open-loop S-curve, which passes through zero at
zero offset to within 1e-16 for every detector here.
`validation/validate_self_noise_bias.py` establishes the mechanism with three
measurements: the offset is proportional to `B_n` (offset / `B_n` constant to
2.2 % for the early-late gate and 7.1 % for Gardner over a factor of eight in
bandwidth), it does not move when the channel SNR changes by 40 dB, and it is
zero to within its standard error for Mueller-Mueller, the one detector with no
self-noise on a Nyquist pulse.

Measured coefficients: **0.383 symbol per unit `B_n`** for the decision-directed
early-late gate and **0.575** for Gardner. At `B_n = 0.01` that is a static
timing error of 0.4 % and 0.6 % of a symbol respectively, which is a third to a
half of the rms jitter at 20 dB per-sample SNR.

---

## 7. PPM slot timing: what changes

See the module docstring of `slotsync.ppm` for the full list. The three points
that change the arithmetic:

1. **Update rate.** One update per symbol, `M` slots per symbol, so a loop with
   normalised bandwidth `B_n` per update has `B_n / M` cycles per slot.
2. **Duty cycle.** The measured loop-SNR cost from `M = 2` to `M = 16` is
   **0.0 to 0.22 dB at 20 dB per-sample SNR and 0.6 to 2.2 dB at 6 dB**, against
   the `10 log10(M/2)` rule of thumb's 0 to 9.03 dB. The rule of thumb is wrong
   in both directions and the measured numbers are published instead.
3. **Failure mode.** The receiver picks the pulse slot by index, so a slot-timing
   failure is a wrong symbol, not a loop slip. At 8-PPM and `B_n = 0.01` the
   slot-index error rate runs from 0 to 0.81 across 18 dB of SNR while the loop
   records **zero** slips.

---

## 8. References

Verified bibliographically on 2026-10-06. No page number, equation number or
numerical value is taken from any of them; they are named because a reader
working in this area will want them, and because the detectors are theirs.

- F. M. Gardner, "A BPSK/QPSK Timing-Error Detector for Sampled Receivers",
  *IEEE Transactions on Communications*, vol. 34, no. 5, pp. 423-429, 1986.
  DOI `10.1109/TCOM.1986.1096561`, checked against the Crossref record.
- K. H. Mueller and M. Mueller, "Timing Recovery in Digital Synchronous Data
  Receivers", *IEEE Transactions on Communications*, vol. 24, no. 5,
  pp. 516-531, 1976. DOI `10.1109/TCOM.1976.1093326`, checked against the
  Crossref record.
- U. Mengali and A. N. D'Andrea, *Synchronization Techniques for Digital
  Receivers*, Springer (Applications of Communications Theory), 1997,
  ISBN 978-0-306-45725-8. Checked against the publisher's catalogue record.
- F. M. Gardner, *Phaselock Techniques*, 3rd edition, Wiley, 2005,
  ISBN 978-0-471-43063-6. Checked against the publisher's catalogue record; its
  chapter list includes "Effects of Additive Noise" and "Digital (Sampled)
  Phaselock Loops".

The early-late gate is elementary and is not attributed. The definition of the
detector gain as the slope of the S-curve at zero offset is also the one GNU
Radio's Symbol Sync block documents, under the name "Expected TED Gain"; that
page was read on 2026-10-06 and is the source of that wording only.
