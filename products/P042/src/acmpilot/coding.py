"""Reed-Solomon bounded-distance decoding accounting over a memoryless bit channel.

Why Reed-Solomon, and why only the accounting
---------------------------------------------
This package needs, for each MODCOD, a defensible map from *channel* bit error
rate to *post-decoding* error rate. It does not need a working decoder: no
information bits are ever transported. Reed-Solomon is used because the only
code property required is exact and needs no table lookup --- an RS code over
GF(2**m) with block length ``n`` and ``k`` information symbols is maximum
distance separable, so

    d_min = n - k + 1,    t = floor((n - k) / 2)                          (1)

follows from the Singleton bound being met with equality. Quoting a BCH
``(n, k, t)`` table instead would introduce a reference this package cannot
check for itself. ``reedsolo`` and ``galois`` (both on PyPI, verified
2026-10-06) ship real RS encoders and decoders and should be used if bits must
actually be moved; see the README alternatives table.

Error model
-----------
Channel bit errors are independent and identically distributed with probability
``p_b`` (equation (1) of :mod:`acmpilot.modulation` supplies ``p_b``). A code
symbol of ``m`` bits is in error if any of its bits is:

    p_s = 1 - (1 - p_b)**m                                                (2)

and the number of symbol errors in a block is binomial:

    S ~ Binomial(n, p_s)                                                  (3)

A bounded-distance decoder corrects the block if and only if ``S <= t``.

**The i.i.d. assumption is an interleaving assumption, and it is the single
strongest assumption in this package.** On the correlated fading channel of
:mod:`acmpilot.channel` the raw bit errors are emphatically *not* independent:
they arrive in bursts lasting the whole fade. Equations (2)-(3) therefore
describe a system with an interleaver deep enough to decorrelate a codeword,
and the latency that interleaver costs is not modelled here. Sizing that
interleaver is the subject of a different product (P041 CodedFade); what this
package assumes is that it has been done. Within one slot the SNR is treated as
constant, which is the usual block-fading idealisation and is reasonable while
the slot is short against the correlation time --- 1 ms against 10 ms by
default.

Post-decoding rates
-------------------
Codeword (frame) error rate, exact under (3):

    FER = P(S > t)                                                        (4)

Output symbol error rate, exact under (3) and the assumption that a failed
decode passes the received word through unchanged (no miscorrection):

    SER_out = E[S * 1{S > t}] / n
            = sum_{i=t+1}^{n} (i/n) * C(n,i) * p_s**i * (1-p_s)**(n-i)    (5)

Output bit error rate:

    BER_out = SER_out * p_b / p_s                                         (6)

``p_b / p_s`` is the expected fraction of wrong bits inside a symbol that is
known to contain at least one wrong bit, ``(m*p_b)/(m*p_s)``. Equation (6)
treats that fraction as independent of ``S``; this is the only approximation in
the chain and it is weak, because the fraction varies slowly with ``p_b``.
Equation (5) is verified against a direct Monte Carlo through the real
modulator in ``validation/validate_modcod_thresholds.py``.

Miscorrection is neglected. A bounded-distance RS decoder presented with more
than ``t`` errors can land on a wrong codeword; for high-rate RS codes that
probability is small relative to (4) but it is not zero, and neglecting it makes
(5) and (6) mildly optimistic. This is recorded in the README limitations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class ReedSolomonCode:
    """An RS code over GF(2**bits_per_symbol), identified only by ``(n, k, m)``.

    Attributes
    ----------
    n
        Block length in code symbols, ``1 <= n <= 2**m - 1``.
    k
        Information symbols per block, ``1 <= k < n``.
    bits_per_symbol
        ``m``, the GF extension degree. 8 for the shipped codes.
    """

    n: int
    k: int
    bits_per_symbol: int = 8

    def __post_init__(self) -> None:
        if self.bits_per_symbol < 1:
            raise ValueError(f"bits_per_symbol must be >= 1, got {self.bits_per_symbol}")
        if not 1 <= self.n <= (1 << self.bits_per_symbol) - 1:
            raise ValueError(
                f"n must be in [1, 2**m - 1] = [1, {(1 << self.bits_per_symbol) - 1}], "
                f"got {self.n}"
            )
        if not 1 <= self.k < self.n:
            raise ValueError(f"k must satisfy 1 <= k < n = {self.n}, got {self.k}")

    @property
    def rate(self) -> float:
        """Code rate ``k/n``, dimensionless."""
        return self.k / self.n

    @property
    def d_min(self) -> int:
        """Minimum distance ``n - k + 1`` symbols, exact by the MDS property, eq (1)."""
        return self.n - self.k + 1

    @property
    def t(self) -> int:
        """Guaranteed correctable symbol errors ``floor((n-k)/2)``, eq (1)."""
        return (self.n - self.k) // 2

    @property
    def label(self) -> str:
        """Short label, e.g. ``"RS(255,223)"``."""
        return f"RS({self.n},{self.k})"

    def symbol_error_rate(self, p_b: float | np.ndarray) -> np.ndarray:
        """Code-symbol error probability from channel bit error rate, eq (2)."""
        p = np.asarray(p_b, dtype=float)
        if np.any(p < 0) or np.any(p > 1):
            raise ValueError("p_b must lie in [0, 1]")
        return 1.0 - (1.0 - p) ** self.bits_per_symbol

    def frame_error_rate(self, p_b: float | np.ndarray) -> np.ndarray:
        """Post-decoding codeword error rate, eq (4)."""
        p_s = self.symbol_error_rate(p_b)
        return np.asarray(stats.binom.sf(self.t, self.n, p_s), dtype=float)

    def output_symbol_error_rate(self, p_b: float | np.ndarray) -> np.ndarray:
        """Post-decoding code-symbol error rate, eq (5)."""
        p_s = np.atleast_1d(self.symbol_error_rate(p_b))
        counts = np.arange(self.t + 1, self.n + 1)
        pmf = stats.binom.pmf(counts[None, :], self.n, p_s[:, None])
        out = (pmf * counts[None, :]).sum(axis=1) / self.n
        return out.reshape(np.shape(p_b)) if np.ndim(p_b) else out.reshape(())

    def output_bit_error_rate(self, p_b: float | np.ndarray) -> np.ndarray:
        """Post-decoding bit error rate, eq (6). Zero where ``p_b == 0``."""
        p = np.asarray(p_b, dtype=float)
        p_s = self.symbol_error_rate(p)
        ser_out = self.output_symbol_error_rate(p)
        with np.errstate(divide="ignore", invalid="ignore"):
            factor = np.where(p_s > 0.0, p / np.where(p_s > 0.0, p_s, 1.0), 0.0)
        return np.asarray(ser_out * factor, dtype=float)


#: The RS codes used by the shipped MODCOD ladder. All over GF(256), n = 255,
#: so ``t`` follows from equation (1) with no table lookup.
SHIPPED_CODES: tuple[ReedSolomonCode, ...] = (
    ReedSolomonCode(255, 127),
    ReedSolomonCode(255, 191),
    ReedSolomonCode(255, 223),
    ReedSolomonCode(255, 239),
)
