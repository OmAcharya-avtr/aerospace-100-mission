"""Systematic Reed-Solomon code over GF(2^m): encoder and Berlekamp-Massey decoder.

Construction
------------
``RS(n, k)`` over GF(2^m) with ``n <= 2^m - 1`` and ``n - k = 2t`` parity
symbols. The generator polynomial is

    g(X) = prod_{i=0}^{2t-1} (X - alpha**(i))                             (18)

with ``alpha`` the primitive element of the field and the first consecutive root
at ``alpha**0 = 1`` (the "narrow-sense" convention). Encoding is systematic:

    c(X) = m(X) * X**(n-k) + (m(X) * X**(n-k) mod g(X))                   (19)

so the first ``k`` symbols of the codeword are the message and the last ``n-k``
are parity.

Decoding is the textbook syndrome / Berlekamp-Massey / Chien / Forney chain:

1. Syndromes ``S_j = r(alpha**j)`` for ``j = 0 .. 2t-1``. All zero means no
   detected error.
2. Berlekamp-Massey returns the error-locator polynomial ``Lambda(X)`` of degree
   ``v <= t``.
3. Chien search evaluates ``Lambda`` at ``alpha**(-i)`` for every position ``i``
   to find the error locations; if the number of distinct roots found is not
   ``deg Lambda``, the received word is beyond the correction radius and
   decoding is declared a failure rather than guessed at.
4. Forney's formula gives the error magnitudes from ``Lambda`` and the syndrome
   polynomial.

Guarantee and its boundary
--------------------------
The code has minimum distance ``d = n - k + 1`` (Reed-Solomon codes are maximum
distance separable), so bounded-distance decoding corrects **any** pattern of up
to ``t = floor((n-k)/2)`` symbol errors and **no** decoder of this class corrects
all patterns of ``t+1``. ``validation/validate_rs_known_answers.py`` verifies
both halves of that statement exhaustively for a small code and by sampling for
a larger one: every ``t``-error pattern is corrected, and ``t+1`` errors either
fail loudly or decode to the wrong codeword, never silently to the right one by
design.

A decoder failure is reported, not hidden: :class:`RSDecodeResult` carries
``success`` and ``corrected_symbols``. A *miscorrection* (decoder succeeds but
the result is wrong) is possible above the radius and is counted separately by
the validation script.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gf import GF2m


@dataclass(frozen=True)
class RSDecodeResult:
    """Outcome of one Reed-Solomon decode.

    Attributes
    ----------
    message:
        Recovered ``k`` message symbols (the received message symbols unchanged
        if decoding failed).
    success:
        ``True`` if the decoder produced a consistent error pattern within the
        correction radius.
    corrected_symbols:
        Number of symbol positions the decoder changed. ``0`` with
        ``success=True`` means the received word was already a codeword.
    """

    message: np.ndarray
    success: bool
    corrected_symbols: int


class ReedSolomon:
    """Systematic narrow-sense Reed-Solomon code over GF(2^m).

    Parameters
    ----------
    n:
        Codeword length in symbols, ``2 <= n <= 2**m - 1``.
    k:
        Message length in symbols, ``1 <= k < n``, with ``n - k`` even.
    m:
        Field extension degree; symbols carry ``m`` bits each.
    """

    def __init__(self, n: int, k: int, m: int = 4) -> None:
        self.field = GF2m(m)
        n, k = int(n), int(k)
        if not 2 <= n <= self.field.order - 1:
            raise ValueError(
                f"n must satisfy 2 <= n <= {self.field.order - 1} for m={m}, got {n!r}"
            )
        if not 1 <= k < n:
            raise ValueError(f"k must satisfy 1 <= k < n = {n}, got {k!r}")
        if (n - k) % 2 != 0:
            raise ValueError(f"n - k must be even, got n-k = {n - k}")
        self.n = n
        self.k = k
        self.m = int(m)
        self.t = (n - k) // 2
        self.generator = self._build_generator()

    @property
    def rate(self) -> float:
        """Code rate ``k/n``, dimensionless."""
        return self.k / self.n

    @property
    def bits_per_symbol(self) -> int:
        """``m``, bits per code symbol."""
        return self.m

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return f"ReedSolomon(n={self.n}, k={self.k}, m={self.m}, t={self.t})"

    def _build_generator(self) -> np.ndarray:
        f = self.field
        g = np.array([1], dtype=np.int64)
        for i in range(2 * self.t):
            root = f.alpha_power(i)
            # (X - alpha**i) == (X + alpha**i) in characteristic 2
            g = f.poly_mul(g, np.array([root, 1], dtype=np.int64))
        return g

    def encode(self, message: np.ndarray) -> np.ndarray:
        """Encode ``k`` message symbols into an ``n``-symbol systematic codeword.

        Returns an int64 array of length ``n``; ``out[:k] == message``.
        """
        msg = np.asarray(message, dtype=np.int64)
        if msg.shape != (self.k,):
            raise ValueError(f"message must have shape ({self.k},), got {msg.shape}")
        f = self.field
        f._check(msg)
        parity = np.zeros(self.n - self.k, dtype=np.int64)
        gen = self.generator[:-1]  # g is monic; drop the leading 1
        for sym in msg.tolist():
            feedback = int(sym) ^ int(parity[-1])
            parity[1:] = parity[:-1]
            parity[0] = 0
            if feedback:
                parity ^= f.mul(feedback, gen)
        out = np.empty(self.n, dtype=np.int64)
        out[: self.k] = msg
        out[self.k :] = parity[::-1]
        return out

    def syndromes(self, received: np.ndarray) -> np.ndarray:
        """``S_j = r(alpha**j)``, ``j = 0 .. 2t-1``. int64 array of length ``2t``."""
        r = np.asarray(received, dtype=np.int64)
        if r.shape != (self.n,):
            raise ValueError(f"received must have shape ({self.n},), got {r.shape}")
        f = self.field
        f._check(r)
        # r(X) with r[0] the highest-order coefficient (first transmitted symbol)
        coeffs = r[::-1]
        return np.array(
            [f.poly_eval(coeffs, f.alpha_power(j)) for j in range(2 * self.t)],
            dtype=np.int64,
        )

    def decode(self, received: np.ndarray) -> RSDecodeResult:
        """Bounded-distance decode of an ``n``-symbol received word."""
        r = np.asarray(received, dtype=np.int64).copy()
        if r.shape != (self.n,):
            raise ValueError(f"received must have shape ({self.n},), got {r.shape}")
        f = self.field
        f._check(r)
        syn = self.syndromes(r)
        if not np.any(syn):
            return RSDecodeResult(r[: self.k].copy(), True, 0)

        lam = self._berlekamp_massey(syn)
        v = int(lam.size - 1)
        if v == 0 or v > self.t:
            return RSDecodeResult(r[: self.k].copy(), False, 0)

        positions: list[int] = []
        roots: list[int] = []
        for i in range(self.n):
            # position i (0 = first transmitted symbol) has locator alpha**(n-1-i)
            xinv = f.alpha_power(-(self.n - 1 - i))
            if f.poly_eval(lam, xinv) == 0:
                positions.append(i)
                roots.append(f.alpha_power(self.n - 1 - i))
        if len(positions) != v:
            return RSDecodeResult(r[: self.k].copy(), False, 0)

        magnitudes = self._forney(syn, lam, roots)
        for pos, mag in zip(positions, magnitudes, strict=True):
            r[pos] ^= mag
        if np.any(self.syndromes(r)):
            return RSDecodeResult(
                np.asarray(received, dtype=np.int64)[: self.k].copy(), False, 0
            )
        return RSDecodeResult(r[: self.k].copy(), True, len(positions))

    def _berlekamp_massey(self, syndromes: np.ndarray) -> np.ndarray:
        """Error-locator polynomial, ascending-degree coefficients, ``lam[0] == 1``."""
        f = self.field
        lam = np.array([1], dtype=np.int64)
        b = np.array([1], dtype=np.int64)
        length = 0
        shift = 1
        b_inv_scale = 1
        for i in range(2 * self.t):
            delta = int(syndromes[i])
            for j in range(1, lam.size):
                delta ^= int(f.mul(lam[j], syndromes[i - j]))
            if delta == 0:
                shift += 1
                continue
            scale = int(f.mul(delta, b_inv_scale))
            shifted = np.zeros(shift + b.size, dtype=np.int64)
            shifted[shift:] = f.mul(scale, b)
            size = max(lam.size, shifted.size)
            new = np.zeros(size, dtype=np.int64)
            new[: lam.size] ^= lam
            new[: shifted.size] ^= shifted
            if 2 * length <= i:
                b = lam
                b_inv_scale = int(f.inv(delta))
                length = i + 1 - length
                shift = 1
            else:
                shift += 1
            lam = new
        nz = np.nonzero(lam)[0]
        return lam[: int(nz[-1]) + 1] if nz.size else np.array([1], dtype=np.int64)

    def _forney(
        self, syndromes: np.ndarray, lam: np.ndarray, roots: list[int]
    ) -> list[int]:
        """Error magnitudes by Forney's formula.

        With ``S(X) = sum_j S_j X**j``, ``Omega(X) = S(X) Lambda(X) mod X**(2t)``
        and ``Lambda(X) = prod_j (1 - X_j X)``, substituting the syndrome
        definition ``S_j = sum_i e_i X_i**j`` gives
        ``Omega(X_i**-1) = e_i prod_{l != i} (1 - X_l / X_i)`` and
        ``Lambda'(X_i**-1) = X_i prod_{l != i} (1 - X_l / X_i)`` (signs vanish in
        characteristic 2), hence

            e_i = X_i * Omega(X_i**-1) / Lambda'(X_i**-1)                  (20)

        The leading ``X_i`` is the factor that the ``b = 1`` form of the formula
        omits; this code uses ``b = 0`` (first syndrome ``r(alpha**0)``), so it
        is required. Omitting it was an actual defect during development, caught
        by the exhaustive known-answer test.
        """
        f = self.field
        two_t = 2 * self.t
        omega = f.poly_mul(syndromes, lam)[:two_t]
        # formal derivative over GF(2^m): only odd-degree terms survive
        lam_prime = np.zeros(max(lam.size - 1, 1), dtype=np.int64)
        for j in range(1, lam.size):
            if j % 2 == 1:
                lam_prime[j - 1] = lam[j]
        out: list[int] = []
        for root in roots:
            xinv = f.inv(root)
            num = f.poly_eval(omega, xinv)
            den = f.poly_eval(lam_prime, xinv)
            if den == 0:
                out.append(0)
                continue
            out.append(int(f.mul(root, f.div(num, den))))
        return out


def symbols_to_bits(symbols: np.ndarray, bits_per_symbol: int) -> np.ndarray:
    """Expand symbols to bits, most significant bit first. uint8 array of 0/1."""
    s = np.asarray(symbols, dtype=np.int64)
    shifts = np.arange(bits_per_symbol - 1, -1, -1, dtype=np.int64)
    return ((s[:, None] >> shifts) & 1).astype(np.uint8).reshape(-1)


def bits_to_symbols(bits: np.ndarray, bits_per_symbol: int) -> np.ndarray:
    """Pack bits (most significant first) into symbols. int64 array.

    ``bits.size`` must be a multiple of ``bits_per_symbol``.
    """
    b = np.asarray(bits, dtype=np.int64).reshape(-1)
    if b.size % bits_per_symbol:
        raise ValueError(
            f"bits.size {b.size} is not a multiple of bits_per_symbol {bits_per_symbol}"
        )
    grid = b.reshape(-1, bits_per_symbol)
    weights = (1 << np.arange(bits_per_symbol - 1, -1, -1, dtype=np.int64))[None, :]
    return (grid * weights).sum(axis=1).astype(np.int64)
