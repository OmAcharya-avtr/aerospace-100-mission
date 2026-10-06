"""GF(2^m) arithmetic by log/antilog tables, in numpy.

Why this exists at all
----------------------
``galois`` (0.4.11, MIT) is a faster and far more general finite-field library
and the README says to use it for field arithmetic. This module exists because
the build rules for this product forbid new dependencies and cross-product
imports, and because the Reed-Solomon decoder here must run inside a Monte Carlo
loop whose cost budget is stated in the README. It implements only what that
loop needs.

Construction
------------
``GF(2^m)`` is represented as the integers ``0 .. 2^m - 1`` interpreted as
polynomial coefficient vectors over GF(2), reduced modulo a primitive polynomial
``p(x)`` of degree ``m``. A primitive element ``alpha = x`` generates the
multiplicative group, so every non-zero element is ``alpha**i`` for exactly one
``i`` in ``0 .. 2^m - 2``. Multiplication is then

    a * b = alpha ** ((log(a) + log(b)) mod (2^m - 1)),   a, b != 0        (17)

and the exponential and logarithm are precomputed tables. This is the standard
table construction; it is exact integer arithmetic with no approximation.

Primitive polynomials, in the conventional binary form (bit ``i`` is the
coefficient of ``x**i``):

===  ========================  ======
m    p(x)                      binary
===  ========================  ======
3    x^3 + x + 1               0b1011
4    x^4 + x + 1               0b10011
5    x^5 + x^2 + 1             0b100101
6    x^6 + x + 1               0b1000011
8    x^8 + x^4 + x^3 + x^2 + 1 0b100011101
===  ========================  ======

Each is verified to be primitive at construction time by checking that the
generated exponential table visits all ``2^m - 1`` non-zero elements exactly
once, so a wrong table cannot silently produce wrong codes.
"""

from __future__ import annotations

import numpy as np

PRIMITIVE_POLYNOMIALS: dict[int, int] = {
    3: 0b1011,
    4: 0b10011,
    5: 0b100101,
    6: 0b1000011,
    8: 0b100011101,
}


class GF2m:
    """Arithmetic in GF(2^m) by table lookup.

    Parameters
    ----------
    m:
        Extension degree, one of the keys of :data:`PRIMITIVE_POLYNOMIALS`.

    Attributes
    ----------
    m:
        Extension degree.
    order:
        ``2**m``, the number of field elements.
    exp:
        int64 array of length ``2*(order-1)``, ``exp[i] = alpha**i``. Doubled so
        that index sums up to ``2*(order-2)`` need no modulo.
    log:
        int64 array of length ``order``; ``log[0]`` is set to ``-1`` and must
        never be used.
    """

    __slots__ = ("m", "order", "exp", "log", "primitive_polynomial")

    def __init__(self, m: int = 4) -> None:
        if m not in PRIMITIVE_POLYNOMIALS:
            raise ValueError(
                f"m must be one of {sorted(PRIMITIVE_POLYNOMIALS)}, got {m!r}"
            )
        self.m = int(m)
        self.order = 1 << self.m
        self.primitive_polynomial = PRIMITIVE_POLYNOMIALS[self.m]
        n = self.order - 1
        exp = np.zeros(2 * n, dtype=np.int64)
        log = np.full(self.order, -1, dtype=np.int64)
        x = 1
        for i in range(n):
            exp[i] = x
            if log[x] != -1:
                raise RuntimeError(
                    f"polynomial {self.primitive_polynomial:#b} is not primitive for m={m}"
                )
            log[x] = i
            x <<= 1
            if x & self.order:
                x ^= self.primitive_polynomial
        if x != 1:
            raise RuntimeError(
                f"polynomial {self.primitive_polynomial:#b} is not primitive for m={m}"
            )
        exp[n:] = exp[:n]
        self.exp = exp
        self.log = log

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return f"GF2m(m={self.m}, p={self.primitive_polynomial:#b})"

    def mul(self, a: np.ndarray | int, b: np.ndarray | int) -> np.ndarray:
        """Element-wise product in GF(2^m). Accepts scalars or integer arrays."""
        a_arr = np.asarray(a, dtype=np.int64)
        b_arr = np.asarray(b, dtype=np.int64)
        self._check(a_arr)
        self._check(b_arr)
        zero = (a_arr == 0) | (b_arr == 0)
        la = self.log[np.where(a_arr == 0, 1, a_arr)]
        lb = self.log[np.where(b_arr == 0, 1, b_arr)]
        out = self.exp[la + lb]
        return np.where(zero, 0, out).astype(np.int64, copy=False)

    def inv(self, a: np.ndarray | int) -> np.ndarray:
        """Multiplicative inverse. Raises ``ZeroDivisionError`` on a zero element."""
        a_arr = np.asarray(a, dtype=np.int64)
        self._check(a_arr)
        if np.any(a_arr == 0):
            raise ZeroDivisionError("0 has no multiplicative inverse in GF(2^m)")
        n = self.order - 1
        return self.exp[(n - self.log[a_arr]) % n].astype(np.int64, copy=False)

    def div(self, a: np.ndarray | int, b: np.ndarray | int) -> np.ndarray:
        """``a / b``. Raises ``ZeroDivisionError`` if any ``b`` is zero."""
        return self.mul(a, self.inv(b))

    def power(self, a: int, k: int) -> int:
        """``a**k`` for integer ``k`` (may be negative)."""
        a = int(a)
        self._check(np.asarray(a))
        if a == 0:
            if k == 0:
                return 1
            if k < 0:
                raise ZeroDivisionError("0 cannot be raised to a negative power")
            return 0
        n = self.order - 1
        return int(self.exp[(self.log[a] * int(k)) % n])

    def alpha_power(self, k: int) -> int:
        """``alpha**k``, ``k`` any integer."""
        n = self.order - 1
        return int(self.exp[int(k) % n])

    def poly_eval(self, coefficients: np.ndarray, x: int) -> int:
        """Evaluate a polynomial at ``x`` by Horner's rule.

        ``coefficients[i]`` is the coefficient of ``X**i``.
        """
        coeffs = np.asarray(coefficients, dtype=np.int64)
        self._check(coeffs)
        acc = 0
        for c in reversed(coeffs.tolist()):
            acc = int(self.mul(acc, x)) ^ int(c)
        return acc

    def poly_mul(self, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Polynomial product over GF(2^m), ascending-degree coefficient order."""
        p = np.asarray(p, dtype=np.int64)
        q = np.asarray(q, dtype=np.int64)
        self._check(p)
        self._check(q)
        out = np.zeros(p.size + q.size - 1, dtype=np.int64)
        for i, pi in enumerate(p.tolist()):
            if pi == 0:
                continue
            out[i : i + q.size] ^= self.mul(pi, q)
        return out

    def _check(self, a: np.ndarray) -> None:
        if a.size and (int(a.min()) < 0 or int(a.max()) >= self.order):
            raise ValueError(
                f"field elements must lie in [0, {self.order - 1}], got "
                f"[{int(a.min())}, {int(a.max())}]"
            )
