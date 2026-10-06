"""BPSK bit error rate over a fading optical channel.

Convention (identical to :mod:`aperturediv.combining`): the instantaneous
SNR is ``gamma = (Eb/N0) * I`` with ``I`` the unit-mean normalised
irradiance, so the fading *power* gain is the lognormal or gamma-gamma
irradiance and the fading amplitude gain is ``sqrt(I)``. Coherent BPSK with
ideal phase reference then has conditional bit error probability

``p(gamma) = Q(sqrt(2 gamma))``,  ``Q(x) = 0.5 erfc(x / sqrt 2)``

the standard AWGN result (any digital communications text; e.g. Proakis,
*Digital Communications*). The channel BER is its expectation over ``I``.

Two routines are provided and they are not interchangeable:

:func:`bpsk_ber_lognormal_sample`
    A **sample** BER: irradiance and receiver noise are both drawn, bits are
    decided one at a time, and the returned figure is the observed fraction
    of bit errors. One BPSK bit per channel realisation, so the bits are
    independent and the binomial standard error is exact. This is the
    quantity the X2 cross-check against P010 BERBench compares.

:func:`bpsk_ber_lognormal_gauss_hermite`
    An **analytic** quadrature of the same expectation, provided as a
    diagnostic for the sample figure. It is not the compared quantity.

Nothing here is flight-qualified or certified.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import special

from .channel import lognormal_sigma_log

MAX_HERMGAUSS_NODES = 350

__all__ = [
    "MAX_HERMGAUSS_NODES",
    "SampleBer",
    "bpsk_ber_awgn",
    "bpsk_ber_from_snr",
    "bpsk_ber_lognormal_gauss_hermite",
    "bpsk_ber_lognormal_sample",
    "q_function",
]


def q_function(x: np.ndarray | float) -> np.ndarray:
    """Gaussian tail ``Q(x) = 0.5 erfc(x / sqrt 2)``, dimensionless."""
    return 0.5 * special.erfc(np.asarray(x, dtype=float) / np.sqrt(2.0))


def bpsk_ber_from_snr(snr_linear: np.ndarray | float) -> np.ndarray:
    """``Q(sqrt(2 gamma))`` for a given instantaneous SNR (linear)."""
    g = np.asarray(snr_linear, dtype=float)
    if np.any(g < 0.0):
        raise ValueError("snr_linear must be >= 0")
    return q_function(np.sqrt(2.0 * g))


def bpsk_ber_awgn(ebn0_db: np.ndarray | float) -> np.ndarray:
    """Coherent BPSK BER on a non-fading AWGN channel, ``Q(sqrt(2 Eb/N0))``."""
    ebn0 = 10.0 ** (np.asarray(ebn0_db, dtype=float) / 10.0)
    return bpsk_ber_from_snr(ebn0)


def bpsk_ber_lognormal_gauss_hermite(
    ebn0_db: float, si: float, n_nodes: int = 200
) -> float:
    """Gauss-Hermite quadrature of ``E[Q(sqrt(2 (Eb/N0) I))]``.

    With ``ln I = -s^2/2 + s Z``, ``Z ~ N(0,1)``, the expectation becomes
    ``(1/sqrt(pi)) sum_i w_i Q(sqrt(2 (Eb/N0) exp(-s^2/2 + s sqrt(2) x_i)))``
    over the ``n_nodes`` Gauss-Hermite abscissae ``x_i`` and weights ``w_i``
    of ``numpy.polynomial.hermite.hermgauss``.

    Diagnostic only; the cross-check quantity is the sample BER.

    ``n_nodes`` is capped at :data:`MAX_HERMGAUSS_NODES`. Above roughly 350
    nodes the installed NumPy's ``hermgauss`` overflows while forming the
    weights and returns NaN -- measured in this environment: 350 nodes gives
    a weight sum of 1.7724538509 (= sqrt(pi), correct), 380 gives NaN. The
    guard raises rather than silently returning NaN, and the computed
    weights are checked for finiteness as well.
    """
    s = lognormal_sigma_log(si)
    ebn0 = 10.0 ** (float(ebn0_db) / 10.0)
    n = int(n_nodes)
    if n < 2:
        raise ValueError(f"n_nodes must be >= 2, got {n!r}")
    if n > MAX_HERMGAUSS_NODES:
        raise ValueError(
            f"n_nodes must be <= {MAX_HERMGAUSS_NODES}; the installed NumPy's "
            f"hermgauss overflows above roughly 350 nodes and returns NaN weights"
        )
    x, w = np.polynomial.hermite.hermgauss(n)
    if not np.all(np.isfinite(w)):
        raise RuntimeError(f"hermgauss({n}) returned non-finite weights")
    irr = np.exp(s * np.sqrt(2.0) * x - 0.5 * s * s)
    return float(np.sum(w * bpsk_ber_from_snr(ebn0 * irr)) / np.sqrt(np.pi))


@dataclass(frozen=True)
class SampleBer:
    """Result of a sample-BER run.

    Attributes
    ----------
    ber
        Observed fraction of bit errors, dimensionless.
    n_errors
        Observed bit-error count.
    n_bits
        Total bits transmitted, equal to the number of channel realisations
        (one bit per realisation).
    binomial_se
        ``sqrt(ber (1 - ber) / n_bits)``. Exact here because one bit per
        independent channel realisation makes the bits independent.
    """

    ber: float
    n_errors: int
    n_bits: int
    binomial_se: float

    @property
    def relative_se(self) -> float:
        """Standard error as a fraction of the BER; ``inf`` if no errors."""
        return float(self.binomial_se / self.ber) if self.ber > 0.0 else float("inf")


def bpsk_ber_lognormal_sample(
    ebn0_db: float,
    si: float,
    n_bits: int,
    seed: int,
    *,
    chunk_size: int = 2_000_000,
) -> SampleBer:
    """Sample BER of coherent BPSK over unit-mean lognormal fading.

    One bit per channel realisation. The draw order inside each chunk is
    fixed and documented so that the figure is reproducible bit for bit:
    for every chunk of ``m`` bits, from a single
    ``numpy.random.default_rng(seed)`` stream,

    1. ``z = rng.standard_normal(m)``   -> irradiance ``I = exp(s z - s^2/2)``
    2. ``bits = rng.integers(0, 2, m)`` -> BPSK symbols ``s_tx = 1 - 2 bits``
    3. ``noise = rng.standard_normal(m)``

    the received sample is ``r = sqrt(2 (Eb/N0) I) s_tx + noise`` and the
    decision is ``bit_hat = (r < 0)``. The ``sqrt(2 Eb/N0)`` scaling with
    unit-variance noise is the standard unit-energy BPSK form whose
    non-fading error probability is ``Q(sqrt(2 Eb/N0))``.

    Parameters
    ----------
    ebn0_db
        ``Eb/N0`` in dB, referred to the *mean* received irradiance.
    si
        Scintillation index of the irradiance, > 0.
    n_bits
        Number of bits, equal to the number of channel realisations.
    seed
        Seed for ``numpy.random.default_rng``.
    chunk_size
        Bits per chunk; affects memory only, not the result, as long as it
        is unchanged. Changing it changes the stream partition and therefore
        the exact sample, so it is part of the reported configuration.
    """
    n = int(n_bits)
    if n < 1:
        raise ValueError(f"n_bits must be >= 1, got {n!r}")
    chunk = int(chunk_size)
    if chunk < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk!r}")
    s = lognormal_sigma_log(si)
    amp = np.sqrt(2.0 * 10.0 ** (float(ebn0_db) / 10.0))
    rng = np.random.default_rng(int(seed))
    errors = 0
    remaining = n
    while remaining > 0:
        m = min(chunk, remaining)
        z = rng.standard_normal(m)
        irr = np.exp(s * z - 0.5 * s * s)
        bits = rng.integers(0, 2, m)
        noise = rng.standard_normal(m)
        received = amp * np.sqrt(irr) * (1.0 - 2.0 * bits) + noise
        errors += int(np.count_nonzero((received < 0.0) != (bits == 1)))
        remaining -= m
    ber = errors / n
    return SampleBer(
        ber=float(ber),
        n_errors=int(errors),
        n_bits=n,
        binomial_se=float(np.sqrt(ber * (1.0 - ber) / n)),
    )
