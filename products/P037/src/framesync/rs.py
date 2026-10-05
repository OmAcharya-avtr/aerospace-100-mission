"""RS(255,223) over GF(2^8): interleaved codec and analytic failure rates.

The code
--------
CCSDS 131.0-B specifies a Reed-Solomon (255, 223) code over GF(2^8) with
E = 16 correctable symbols (minimum distance d_min = 2E + 1 = 33), applied
to a codeblock of ``I`` interleaved codewords. The transfer frame therefore
carries ``223 * I`` octets of data inside ``255 * I`` octets on the channel,
for a code rate R = 223/255 = 0.874510.

CCSDS specifies the field in a **dual basis** and a particular generator
polynomial. ``reedsolo``, used here for the encode/decode leg, implements
the conventional basis with its own generator. The two are
representation-different and not bit-compatible with a CCSDS flight
decoder, but they are the same (255, 223) bounded-distance code with the
same E = 16 and therefore the same error-correction *statistics*, which is
what a performance harness measures. This package makes no claim of
bit-level CCSDS interoperability, and anything that must interoperate on
the wire needs a dual-basis implementation.

Analytic expressions
--------------------
With independent channel bit errors at probability p, a symbol of m = 8
bits is in error with probability

    p_s = 1 - (1 - p)^m                                                 (6)

A bounded-distance decoder fails when more than E symbols are in error, so
the codeword decoding failure probability is exactly

    P_cw = sum_{i=E+1}^{n} C(n, i) p_s^i (1 - p_s)^(n-i)                (7)

i.e. the upper tail of Binomial(n, p_s) above E. For a codeblock of I
interleaved codewords on a memoryless channel the codewords see independent
symbol errors, so interleaving changes nothing and

    FER = 1 - (1 - P_cw)^I                                              (8)

The post-decoding symbol and bit error rates use the standard
bounded-distance approximations

    P_s,out ~= (1/n) sum_{i=E+1}^{n} i C(n, i) p_s^i (1 - p_s)^(n-i)    (9)
    P_b,out ~= (2^(m-1) / (2^m - 1)) * P_s,out                          (10)

[Standard results for Reed-Solomon over a memoryless symbol channel; see
Sklar 2001, "Digital Communications: Fundamentals and Applications" 2nd ed.,
Ch. 8 (Reed-Solomon performance), and Lin & Costello 2004, "Error Control
Coding" 2nd ed., Ch. 7. Equation (9) assumes a decoding failure is declared
rather than miscorrected, which is how ``reedsolo`` behaves: it raises
``ReedSolomonError``. Equation (10) is the usual uniform-error assumption
that a symbol error puts each of its m bits in error with probability
2^(m-1)/(2^m - 1).]

Validity: memoryless channel, hard-decision symbol errors, no erasures, no
miscorrection. Equations (7) and (8) are exact under those assumptions;
(9) and (10) are approximations and are labelled as such wherever printed.

Known non-monotonicity of Eq. (9)/(10): Eq. (9) models a decoder that emits
the *uncorrected* codeword on failure, so below the code threshold -- around
p = 1e-2, where the expected symbol-error count approaches E -- it returns
a post-decoding bit error rate *above* the channel bit error rate. That is
the error-amplification regime of a high-rate block code and is a property
of the expression and of hard-decision Reed-Solomon, not a defect; the
``reedsolo`` decoder used here declares failure instead, so the measured
frame error rate of Eqs. (7)/(8) is the quantity to trust in that regime.
``README.md`` lists this under Limitations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import binom

__all__ = [
    "RS_N",
    "RS_K",
    "RS_E",
    "RS_RATE",
    "symbol_error_probability",
    "codeword_failure_probability",
    "rs_frame_error_rate",
    "rs_output_bit_error_rate",
    "ReedSolomonLink",
]

RS_N = 255
"""Codeword length in octets."""

RS_K = 223
"""Information length in octets."""

RS_E = 16
"""Correctable symbols per codeword, E = (n - k) / 2."""

RS_RATE = RS_K / RS_N
"""Code rate R = 223/255 = 0.8745098039215686 (dimensionless)."""


def symbol_error_probability(p_bit: float | np.ndarray, m: int = 8) -> np.ndarray:
    """Equation (6): symbol error probability from channel bit error rate."""
    p = np.asarray(p_bit, dtype=float)
    if np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError("p_bit must lie in [0, 1]")
    if m < 1:
        raise ValueError(f"m must be a positive bit count, got {m}")
    # 1 - (1-p)^m via log1p/expm1 for relative precision at small p; p == 1
    # would give log1p(-1) = -inf, so that endpoint is set directly.
    safe = np.minimum(p, np.nextafter(1.0, 0.0))
    return np.asarray(
        np.where(p >= 1.0, 1.0, -np.expm1(int(m) * np.log1p(-safe))), dtype=float
    )


def codeword_failure_probability(
    p_sym: float | np.ndarray, n: int = RS_N, e: int = RS_E
) -> np.ndarray:
    """Equation (7): exact P[more than ``e`` symbol errors in ``n`` symbols].

    Evaluated as the binomial survival function, which keeps full relative
    precision in the far tail where the direct sum of (9) would not.
    """
    ps = np.asarray(p_sym, dtype=float)
    if np.any(ps < 0.0) or np.any(ps > 1.0):
        raise ValueError("p_sym must lie in [0, 1]")
    if not 0 <= int(e) < int(n):
        raise ValueError(f"e must lie in [0, n), got e={e}, n={n}")
    return np.asarray(binom.sf(int(e), int(n), ps), dtype=float)


def rs_frame_error_rate(
    p_bit: float | np.ndarray, interleave: int = 5, n: int = RS_N, e: int = RS_E, m: int = 8
) -> np.ndarray:
    """Equation (8): codeblock (frame) error rate for I interleaved codewords."""
    i = int(interleave)
    if i < 1:
        raise ValueError(f"interleave depth must be >= 1, got {interleave}")
    pcw = codeword_failure_probability(symbol_error_probability(p_bit, m), n, e)
    # 1 - (1 - pcw)^I via log1p/expm1 for relative precision when I*pcw << 1;
    # pcw == 1 would make log1p(-1) = -inf, so that saturated case is set
    # directly rather than letting a warning escape.
    safe = np.minimum(pcw, np.nextafter(1.0, 0.0))
    out = np.where(pcw >= 1.0, 1.0, -np.expm1(i * np.log1p(-safe)))
    return np.asarray(out, dtype=float)


def rs_output_bit_error_rate(
    p_bit: float | np.ndarray, n: int = RS_N, e: int = RS_E, m: int = 8
) -> np.ndarray:
    """Equations (9) and (10): approximate post-decoding bit error rate.

    Labelled an approximation everywhere it is printed; see module docstring
    for the two assumptions it rests on.
    """
    ps = np.atleast_1d(symbol_error_probability(p_bit, m))
    i = np.arange(int(e) + 1, int(n) + 1)
    pmf = binom.pmf(i[None, :], int(n), ps[:, None])
    p_s_out = (i[None, :] * pmf).sum(axis=1) / int(n)
    scale = (2 ** (int(m) - 1)) / (2 ** int(m) - 1)
    return scale * p_s_out


@dataclass
class RSDecodeResult:
    """Per-codeword outcome of :meth:`ReedSolomonLink.decode_codeword`."""

    corrected: bool
    n_symbol_errors_in: int
    message: bytes | None


class ReedSolomonLink:
    """RS(255,223) encode/decode over ``I`` interleaved codewords.

    Uses ``reedsolo`` for the Galois-field arithmetic (see module docstring
    on the dual-basis caveat). ``reedsolo`` raises ``ReedSolomonError`` on a
    decoding failure, so a failure is always declared and never silently
    miscorrected, which is the behaviour Eq. (9) assumes.

    Parameters
    ----------
    interleave : int
        Interleaving depth I, >= 1. Frame data length is ``223 * I`` octets.
    """

    def __init__(self, interleave: int = 5) -> None:
        if int(interleave) < 1:
            raise ValueError(f"interleave depth must be >= 1, got {interleave}")
        import reedsolo

        self._rs_mod = reedsolo
        self.interleave = int(interleave)
        self.codec = reedsolo.RSCodec(RS_N - RS_K)

    @property
    def frame_data_octets(self) -> int:
        """Information octets per frame, 223 * I."""
        return RS_K * self.interleave

    @property
    def codeblock_octets(self) -> int:
        """Channel octets per frame, 255 * I."""
        return RS_N * self.interleave

    @property
    def rate(self) -> float:
        """Code rate, 223/255 (dimensionless)."""
        return RS_RATE

    def encode_codeword(self, message: bytes) -> bytes:
        """Encode 223 octets into a 255-octet codeword."""
        if len(message) != RS_K:
            raise ValueError(f"message must be exactly {RS_K} octets, got {len(message)}")
        return bytes(self.codec.encode(bytes(message)))

    def decode_codeword(self, codeword: bytes) -> RSDecodeResult:
        """Decode a 255-octet codeword; report success and nothing invented."""
        if len(codeword) != RS_N:
            raise ValueError(f"codeword must be exactly {RS_N} octets, got {len(codeword)}")
        try:
            msg = bytes(self.codec.decode(bytes(codeword))[0])
        except self._rs_mod.ReedSolomonError:
            return RSDecodeResult(False, -1, None)
        return RSDecodeResult(True, -1, msg)

    def encode_frame(self, data: bytes) -> bytes:
        """Encode ``223 * I`` octets into an interleaved ``255 * I`` codeblock.

        Interleaving is symbol-wise: codeword j takes octets j, j+I, j+2I...
        of the frame, and the output is the I codewords interleaved the same
        way, which is the CCSDS arrangement. On a memoryless channel it
        changes no statistics; it matters only for burst errors.
        """
        if len(data) != self.frame_data_octets:
            raise ValueError(
                f"frame data must be {self.frame_data_octets} octets "
                f"(223 * I, I={self.interleave}), got {len(data)}"
            )
        arr = np.frombuffer(data, dtype=np.uint8).reshape(RS_K, self.interleave)
        words = [self.encode_codeword(bytes(arr[:, j])) for j in range(self.interleave)]
        block = np.stack([np.frombuffer(w, dtype=np.uint8) for w in words], axis=1)
        return block.tobytes()

    def decode_frame(self, block: bytes) -> tuple[bytes | None, int]:
        """Decode an interleaved codeblock.

        Returns ``(data, n_failed_codewords)``; ``data`` is None if any
        codeword failed, because the frame is then not deliverable.
        """
        if len(block) != self.codeblock_octets:
            raise ValueError(
                f"codeblock must be {self.codeblock_octets} octets, got {len(block)}"
            )
        arr = np.frombuffer(block, dtype=np.uint8).reshape(RS_N, self.interleave)
        msgs, failed = [], 0
        for j in range(self.interleave):
            res = self.decode_codeword(bytes(arr[:, j]))
            if not res.corrected:
                failed += 1
                msgs.append(None)
            else:
                msgs.append(res.message)
        if failed:
            return None, failed
        out = np.stack([np.frombuffer(m, dtype=np.uint8) for m in msgs], axis=1)  # type: ignore[arg-type]
        return out.tobytes(), 0
