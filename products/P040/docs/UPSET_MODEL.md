# The upset model, its derivations and its validity range

Every equation used by `bitflipsim` is derived here, with its units, its
assumptions and the validation script that checks it. Nothing in this document
is quoted from a reference the author has not read; where a result is standard,
the derivation is given in full so that it can be checked rather than trusted.

## 1. Storage layout

### 1.1 IEEE 754 binary interchange formats

For a storage word of `w` bits with `nmant` trailing significand bits and
`nexp` biased-exponent bits, `w = 1 + nexp + nmant`, a **normal** number is

```
value = (-1)**s * 2**(E - bias) * (1 + m / 2**nmant),   bias = 2**(nexp-1) - 1
```

with `s` the sign bit, `E` the biased exponent in `[1, 2**nexp - 2]` and `m`
the trailing significand field. `E = 0` is zero or a subnormal, `E = 2**nexp-1`
is an infinity (`m = 0`) or a NaN (`m != 0`).

`bitflipsim.bitlayout.float_layout` obtains `nmant` and `nexp` from
`numpy.finfo` and computes `bias` from them; `verify_float_layout` re-derives
all of them a second way from probe encodings:

| field | probe |
|---|---|
| `nmant` | `bits(2.0) - bits(1.0)`, which the formula makes exactly `1 << nmant` because both have `m = 0` and `E` differing by one |
| `bias` | the exponent field of `1.0`, which the formula requires to equal the bias |
| `nexp` | the bit length of `bits(inf) >> nmant`, the all-ones field |
| sign bit | the single set bit of `-0.0` |

`validation/validate_ieee754_layout.py` runs this for float16, float32 and
float64 and reports agreement field by field.

### 1.2 Single-bit flips

Flipping storage bit `p`:

* `p = w - 1` (sign): the magnitude is unchanged, the ratio is exactly `-1`.
* `nmant <= p <= w - 2` (exponent): `E` changes by `d = +- 2**(p - nmant)`, `s`
  and `m` are unchanged, so **while both operand and result are normal** the
  magnitude is multiplied by exactly `2**d`. For the exponent MSB
  `p = w - 2`, `d = +- 2**(nexp - 1)`: `+- 128` for float32, `+- 16` for
  float16, `+- 1024` for float64.
* `0 <= p <= nmant - 1` (mantissa): `m` changes by `+- 2**p`, so the value
  changes **additively** by `+- 2**(E - bias - nmant + p)`, a relative change of
  at most `2**(p - nmant) <= 2**-1`.

The exponent case has three regimes outside the normal one, all predicted
exactly from the fields rather than from a float ratio: the new `E` may
saturate to all ones (infinity if `m = 0`, NaN otherwise), or land on zero
(subnormal), or the operand may already have been special.

`validation/validate_ieee754_layout.py` checks the field-level prediction
(integer arithmetic, no overflow) and, where the ratio is a representable
float, the ratio itself. For float64 the factor `2**1024` is not representable,
which is exactly why the field-level form is the one validated.

### 1.3 Two's-complement integers

```
value = -2**(w-1) * b_{w-1} + sum_{k < w-1} 2**k * b_k
```

so flipping bit `k < w-1` changes the value by `+- 2**k`, and flipping the sign
bit changes it by `-2**(w-1)` when that bit goes 0 -> 1 and `+2**(w-1)` when it
goes 1 -> 0. There is no exponent, so there is no multiplicative blow-up: the
worst single-bit perturbation of an int8 code is 128 code steps, which after
dequantization is `128 * scale`. Checked exhaustively over all 256 int8 values
and all 8 bit positions.

## 2. Upset rate and counts

### 2.1 Rate

For `N` exposed bits under an omnidirectional particle flux `phi` with an
effective per-bit upset cross-section `sigma`,

```
lambda = phi * sigma * N          [upsets s^-1]
phi    [particles cm^-2 s^-1]
sigma  [cm^2 bit^-1]
```

This is dimensional accounting, not a fitted model: `sigma` is *defined* as the
upset probability per unit fluence per bit, so the product is the rate. Over an
exposure `t`, the expected count is `mu = lambda * t`.

In FIT (failures per `10**9` device-hours), `lambda_FIT = lambda * 1e9 * 3600 =
lambda * 3.6e12`.

### 2.2 Poisson counts and their sampling errors

Treating the `N` bits as independent with per-bit upset probability
`p = phi * sigma * t`, the count is binomial `(N, p)` and tends to Poisson with
mean `mu = N p` as `p -> 0`. `bitflipsim.flux.poisson_validity` reports `p` and
flags `p >= 1e-3` rather than assuming the limit holds.

For a Poisson sample of size `n`:

* `Var[mean] = mu / n`, so `se(mean) = sqrt(mu / n)`.
* The unbiased sample variance has, for large `n`,
  `Var[S2] = (mu_4 - mu_2**2) / n`. The Poisson central moments are
  `mu_2 = mu` and `mu_4 = mu + 3 mu**2`, so
  `Var[S2] = (mu + 2 mu**2) / n` and `se(S2) = sqrt((mu + 2 mu**2) / n)`.

Both are reported as z scores against the model in
`validation/validate_poisson_counts.py`, together with a chi-square fit of the
whole count histogram to the Poisson pmf.

### 2.3 What this model does not include

A single effective cross-section collapses the real object - a cross-section
that is a function of linear energy transfer, measured by a heavy-ion or proton
test and usually fitted with a Weibull curve - into one number, and the correct
rate is an integral of that curve over the LET spectrum of the environment.
Multiple-bit upsets from one particle, angular dependence, shielding,
single-event functional interrupts, latch-up, total ionising dose and
displacement damage are all outside the model. A result from this package is a
first-order sensitivity study of a model's numerics, not a radiation
qualification of anything.

## 3. Degradation metric

```
D = (1/n) sum_i TV(p_i, p'_i),    TV(p, q) = 0.5 * sum_c |p_c - q_c|
```

bounded in `[0, 1]`, defined for every pair of distributions, and not saturating
on a small evaluation split the way accuracy does. Conventions for outputs that
are not distributions at all, stated rather than inherited from numpy:

* a logit vector containing a NaN gives an all-NaN probability vector, and
  `TV = 1` - the maximum;
* a logit vector containing `+inf` gives probability spread uniformly over the
  `+inf` entries;
* an argmax over a NaN row reports class `-1`, which always counts as wrong.

## 4. Mitigations

### 4.1 Parameter range clamping

Parameters are read back through a clip into `[-C, C]`, so for any upset
`|dw| <= 2C`. For the two-layer network `z1 = x W1 + b1`, `a1 = relu(z1)`,
`z2 = a1 W2 + b2` with every parameter clamped and `Xmax = max_i |x_i|`:

| upset in | bound on `max_k |dz2_k|` | why |
|---|---|---|
| `W1[i,j]` | `2 C**2 Xmax` | `dz1_j = dw x_i`; ReLU is 1-Lipschitz; `dz2_k = da1_j W2[j,k]`, `|W2[j,k]| <= C` |
| `b1[j]` | `2 C**2` | as above with `x_i` replaced by 1 |
| `W2[j,k]` | `2 C**2 (n_in Xmax + 1)` | `dz2_k = dw a1_j` and `a1_j <= sum_i |x_i||W1[i,j]| + |b1_j| <= C(n_in Xmax + 1)` |
| `b2[k]` | `2C` | `dz2_k = dw` |

Worst of the four:

```
B = 2 C * max( C * (n_in * Xmax + 1), 1 )
```

`validation/validate_clamp_bound.py` checks `B` against **every** one of the
4704 single-upset sites at three clamps, and reports the tightness ratio rather
than claiming the bound is tight. The bound is for **single** upsets; two
simultaneous upsets obey `2B` by the triangle inequality only if their effects
are in the same layer, and the package does not claim a general multi-upset
bound.

The clamp also fixes the regime where no bound exists at all: without it,
50 of the 4704 single upsets drive the logits to a non-finite value.

### 4.2 Triplication and majority voting

Two voters, with different failure modes:

* **word-level**: return the value held by at least two copies; flag the
  element uncorrectable when all three differ.
* **bitwise**: `(a & b) | (a & c) | (b & c)` on storage bits. Always returns a
  value, never signals.

Enumerated exhaustively in `validation/validate_triplication.py` over 3
float16 parameters in 3 copies (144 single-upset cases, 10296 double-upset
pairs) and over 1 float32 parameter in 3 copies:

| double-upset case | word voter | bitwise voter |
|---|---|---|
| both in the same copy | correct | correct |
| different copies, different elements | correct | correct |
| different copies, same element, different bits | cannot correct, flags every one | correct |
| different copies, same element, same bit | **silently wrong** | **silently wrong** |

The last row is the classic TMR double-upset failure: two copies agree on the
same wrong value, so there is nothing for a voter to detect.

### 4.3 Periodic reload

Upsets arrive as a Poisson process of rate `lambda` and are cleared at every
reload. Observed at a time `tau` uniform on `[0, T_s)` into the current scrub
interval, `E[live | tau] = lambda tau`, so

```
E[live upsets] = lambda * T_s / 2
```

Half of what an unscrubbed interval accumulates by its end. Costs one golden
copy and a duty cycle `t_reload / T_s` of lost inference time. Checked against a
Monte Carlo of the arrival-and-scrub process.

## 5. Validity range, in one place

| quantity | valid when | enforced or reported by |
|---|---|---|
| exponent-flip ratio `2**(+-2**(p-nmant))` | operand and result both normal | `predict_flip` returns the regime; other regimes get their own exact prediction |
| Poisson count model | `phi * sigma * t << 1` per bit | `poisson_validity`, threshold `1e-3` |
| clamping bound `B` | single upset, inputs obeying `Xmax`, the two-layer ReLU network of `bitflipsim.network` | derivation above; exhaustively checked |
| `E[live] = lambda T_s / 2` | Poisson arrivals, complete reload, observation uniform in the interval | derivation above; Monte Carlo checked |
| TMR single-upset recovery | at most one upset per voted triple | exhaustive enumeration |
| criticality ordering | this 147-parameter model, this dataset | not established for larger models; see README limitations |
