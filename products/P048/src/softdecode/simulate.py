"""End-to-end OOK-over-fading simulation with a frozen random-draw order.

Every validation number in this repository comes from :func:`simulate_ook`,
whose random draws happen in one fixed order so that a seed reproduces a run
exactly:

1. information bits, ``rng.integers(0, 2, (blocks, k))``;
2. fading ``h`` per channel bit, from the fading model's own ``sample``
   (one standard normal per sample for lognormal, two gammas for
   gamma-gamma);
3. channel-state estimate ``h_hat``, from the error model (one standard
   normal per sample, or none when the jitter is zero);
4. noise, ``rng.standard_normal((blocks, n))``.

Interleaving assumption
-----------------------
``h`` is drawn **independently per channel bit**. That is the ideal-
interleaving limit. A real atmospheric channel has fades lasting
milliseconds, so the required interleaver depth and its latency cost are a
separate question which this package does not answer and which the README
points elsewhere for. Every coded result here is therefore an ideally
interleaved result and is optimistic for a real link.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .channel import GammaGammaFading, LognormalFading, amplitude_quadrature
from .csi import MultiplicativeCsiError, StaleCsiError, csi_aware_llr_ook
from .detection import DetectionModel
from .ldpc import LdpcCode, sum_product_decode
from .llr import clip_llr, llr_ook_known_csi, llr_ook_marginal, llr_ook_maxlog, rescale_llr
from .metrics import bit_error_rate

Demapper = Literal[
    "known_csi",
    "plugin",
    "csi_aware_exact",
    "csi_aware_maxlog",
    "no_csi_exact",
    "no_csi_maxlog",
]

DEMAPPERS: tuple[str, ...] = (
    "known_csi",
    "plugin",
    "csi_aware_exact",
    "csi_aware_maxlog",
    "no_csi_exact",
    "no_csi_maxlog",
)

__all__ = ["DEMAPPERS", "OokRealisation", "demap_ook", "decode_ber", "simulate_ook"]


@dataclass(frozen=True)
class OokRealisation:
    """One frozen batch of OOK channel realisations.

    Attributes
    ----------
    messages:
        ``(blocks, k)`` information bits.
    codeword:
        ``(blocks, n)`` channel bits.
    h, h_hat:
        ``(blocks, n)`` true and estimated normalised irradiance.
    y:
        ``(blocks, n)`` matched-filter samples.
    amplitude, sigma:
        Peak amplitude at ``h = 1`` and noise standard deviation.
    ebn0_db, rate:
        The Eb/N0 and code rate the amplitude was derived from.
    """

    messages: np.ndarray
    codeword: np.ndarray
    h: np.ndarray
    h_hat: np.ndarray
    y: np.ndarray
    amplitude: float
    sigma: float
    ebn0_db: float
    rate: float

    @property
    def blocks(self) -> int:
        """Number of codeword blocks."""
        return int(self.codeword.shape[0])

    @property
    def channel_bits(self) -> int:
        """Total channel bits in the batch."""
        return int(self.codeword.size)


def simulate_ook(
    code: LdpcCode,
    ebn0_db: float,
    fading: LognormalFading | GammaGammaFading,
    error: MultiplicativeCsiError | StaleCsiError | None,
    blocks: int,
    seed: int,
    detection: DetectionModel | None = None,
) -> OokRealisation:
    """Draw ``blocks`` codewords through the fading OOK channel.

    ``error`` of ``None`` means perfect CSI (``h_hat = h``).
    """
    det = DetectionModel() if detection is None else detection
    rng = np.random.default_rng(int(seed))
    rate = code.rate
    amplitude = det.ook_amplitude(ebn0_db, rate)
    messages = rng.integers(0, 2, size=(int(blocks), code.dimension)).astype(np.int8)
    codeword = code.encode(messages)
    h = fading.sample(codeword.shape, rng)
    if error is None:
        h_hat = h.copy()
    elif isinstance(error, StaleCsiError):
        h_hat = error.estimate(h, rng, fading)
    else:
        h_hat = error.estimate(h, rng)
    noise = rng.standard_normal(codeword.shape)
    y = amplitude * h * codeword + det.sigma * noise
    return OokRealisation(
        messages, codeword, h, h_hat, y, amplitude, det.sigma, float(ebn0_db), rate
    )


def demap_ook(
    realisation: OokRealisation,
    method: str,
    fading: LognormalFading | GammaGammaFading,
    error: MultiplicativeCsiError | StaleCsiError | None = None,
    nodes: int | None = None,
    clip: float | None = None,
    scale: float | None = None,
) -> np.ndarray:
    """Compute ``(blocks, n)`` LLRs from a realisation by the named demapper.

    Methods
    -------
    ``known_csi``
        Equation (1) with the true ``h``. Unreachable upper bound.
    ``plugin``
        Equation (1) with ``h_hat`` substituted for ``h``. The naive
        mismatched receiver.
    ``csi_aware_exact`` / ``csi_aware_maxlog``
        Likelihood averaged over ``p(h | h_hat)``, exactly or by max-log.
        Requires ``error``.
    ``no_csi_exact`` / ``no_csi_maxlog``
        Likelihood averaged over the fading prior, ignoring ``h_hat``
        entirely.

    ``scale`` is applied before ``clip``, both optional.
    """
    y = realisation.y
    a = realisation.amplitude
    s = realisation.sigma
    if method == "known_csi":
        out = llr_ook_known_csi(y, a, realisation.h, s)
    elif method == "plugin":
        out = llr_ook_known_csi(y, a, realisation.h_hat, s)
    elif method in ("csi_aware_exact", "csi_aware_maxlog"):
        if error is None:
            raise ValueError(f"{method} requires an error model")
        flat = csi_aware_llr_ook(
            y.ravel(),
            a,
            realisation.h_hat.ravel(),
            fading,
            error,
            s,
            nodes=32 if nodes is None else nodes,
            max_log=method.endswith("maxlog"),
        )
        out = flat.reshape(y.shape)
    elif method in ("no_csi_exact", "no_csi_maxlog"):
        hq, wq = amplitude_quadrature(fading, nodes)
        fn = llr_ook_maxlog if method.endswith("maxlog") else llr_ook_marginal
        out = fn(y, a, hq, wq, s)
    else:
        raise ValueError(f"unknown demapper {method!r}; expected one of {DEMAPPERS}")
    if scale is not None:
        out = rescale_llr(out, scale)
    if clip is not None:
        out = clip_llr(out, clip)
    return out


def decode_ber(
    code: LdpcCode,
    realisation: OokRealisation,
    llr: np.ndarray,
    iterations: int = 20,
):
    """Sum-product decode ``llr`` and return the information-bit error rate."""
    bits, _, _ = sum_product_decode(code, llr, iterations=iterations)
    hat = bits[:, code.message_positions()]
    return bit_error_rate(hat, realisation.messages)
