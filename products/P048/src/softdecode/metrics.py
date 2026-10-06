"""Quality measures for LLRs and for the decoders that consume them.

Three different things are measured, because they disagree and a user who
conflates them will make the wrong design choice:

**LLR error** (:func:`llr_error`) -- root-mean-square and maximum absolute
deviation from a reference LLR. Cheap, and almost useless on its own: a
demapper can have a large RMS LLR error and lose nothing after decoding.

**Generalised mutual information** (:func:`generalised_mutual_information`) --
for a bit-metric decoder fed LLRs ``L`` with transmitted bits ``b``,

    GMI = 1 - E[ log2( 1 + exp( -x * L ) ) ],   x = 1 - 2b,

in bits per channel bit. This is the achievable rate of bit-metric decoding
with those LLRs (the standard mismatched-decoding lower bound; see e.g.
G. Böcherer, F. Steiner and P. Schulte, "Bandwidth efficient and
rate-matched low-density parity-check coded modulation", *IEEE Transactions
on Communications* 63(12), 4651-4665, 2015, section II, for this form). It is
the right scalar for *calibration*: it rewards being right and penalises
being confidently wrong, so a demapper with over-confident LLRs loses GMI even
when its hard decisions are unchanged. It is bounded above by 1 for binary
input and may go negative for badly scaled LLRs.

**Decoded bit error rate** (:func:`bit_error_rate`) -- the only quantity a
link designer is paid to care about, reported with its binomial standard
error ``sqrt(p (1-p) / n)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "BitErrorRate",
    "LlrError",
    "bit_error_rate",
    "generalised_mutual_information",
    "llr_error",
]


@dataclass(frozen=True)
class BitErrorRate:
    """A bit error rate with the binomial standard error of the estimate."""

    errors: int
    trials: int

    def __post_init__(self) -> None:
        if self.trials <= 0:
            raise ValueError(f"trials must be positive, got {self.trials!r}")
        if not 0 <= self.errors <= self.trials:
            raise ValueError(f"errors must lie in [0, trials], got {self.errors!r}")

    @property
    def rate(self) -> float:
        """``errors / trials``, dimensionless."""
        return self.errors / self.trials

    @property
    def standard_error(self) -> float:
        """``sqrt(p (1-p) / n)``; zero when no errors were observed."""
        p = self.rate
        return float(np.sqrt(p * (1.0 - p) / self.trials))

    def __str__(self) -> str:
        return f"{self.rate:.6e} +- {self.standard_error:.3e} ({self.errors}/{self.trials})"


def bit_error_rate(decoded, reference) -> BitErrorRate:
    """Compare two 0/1 arrays of equal shape."""
    a = np.asarray(decoded)
    b = np.asarray(reference)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return BitErrorRate(int(np.count_nonzero(a != b)), int(a.size))


@dataclass(frozen=True)
class LlrError:
    """Deviation of an LLR array from a reference LLR array."""

    rmse: float
    max_abs: float
    mean_signed: float
    reference_rms: float

    @property
    def relative_rmse(self) -> float:
        """``rmse / rms(reference)``, dimensionless."""
        return self.rmse / self.reference_rms if self.reference_rms > 0 else float("nan")

    def __str__(self) -> str:
        return (
            f"rmse {self.rmse:.6e}  max|e| {self.max_abs:.6e}  "
            f"mean signed {self.mean_signed:+.6e}  relative {self.relative_rmse:.6e}"
        )


def llr_error(llr, reference) -> LlrError:
    """RMS, maximum and mean signed deviation ``llr - reference``."""
    a = np.asarray(llr, dtype=float)
    b = np.asarray(reference, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    d = a - b
    return LlrError(
        float(np.sqrt(np.mean(d**2))),
        float(np.max(np.abs(d))),
        float(np.mean(d)),
        float(np.sqrt(np.mean(b**2))),
    )


def generalised_mutual_information(llr, bits) -> float:
    """GMI in bits per channel bit for LLRs ``log P(0)/P(1)`` and bits 0/1."""
    lam = np.asarray(llr, dtype=float)
    b = np.asarray(bits)
    if lam.shape != b.shape:
        raise ValueError(f"shape mismatch: {lam.shape} vs {b.shape}")
    if np.any((b != 0) & (b != 1)):
        raise ValueError("bits must be 0/1")
    x = 1.0 - 2.0 * b.astype(float)
    return float(1.0 - np.mean(np.logaddexp(0.0, -x * lam) / np.log(2.0)))
