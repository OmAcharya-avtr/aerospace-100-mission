"""Eb/N0 conventions and the hard-decision BPSK/AWGN bit channel.

Conventions (stated once, used everywhere in this package)
----------------------------------------------------------
* ``ebn0_db`` is the energy per **information** bit divided by the one-sided
  noise spectral density, Eb/N0, in dB. Dimensionless ratio in dB.
* A code of rate ``R = k/n`` spends Es = R * Eb per transmitted channel
  symbol, so the **channel** symbol energy ratio is Es/N0 = R * Eb/N0.
* Modulation is BPSK with coherent detection over AWGN. The channel bit
  error probability is therefore

      p = Q( sqrt( 2 * R * Eb/N0 ) )                                    (1)

  with Q the Gaussian tail function Q(x) = 0.5 * erfc(x / sqrt(2)).
  For the uncoded link R = 1 and (1) reduces to Pb = Q(sqrt(2 Eb/N0)).
  [Proakis & Salehi 2008, "Digital Communications" 5th ed., Eq. (4.3-13)]

* Validity: ideal coherent detection, perfect carrier and symbol
  synchronisation, memoryless AWGN, no fading, no implementation loss.
  Real receivers add 0.5-2 dB of implementation loss which this package
  does **not** model.

Consistency with P010 BERBench
------------------------------
P010 BERBench defines ``snr_db`` as Eb/N0 per bit in dB and evaluates BPSK
over AWGN as Pb = Q(sqrt(2 * gamma)), gamma = 10**(snr_db/10). Equation (1)
with R = 1 is that same expression, implemented independently here (scipy
``erfc`` rather than BERBench's ``qfunc``), which is what makes the
cross-check in ``validation/crosscheck_berbench.py`` meaningful rather than
circular.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erfc

__all__ = [
    "qfunc",
    "bpsk_ber",
    "ebn0_to_esn0_db",
    "bsc_flip",
    "awgn_bpsk_samples",
]


def qfunc(x: float | np.ndarray) -> np.ndarray:
    """Gaussian tail Q(x) = 0.5 * erfc(x / sqrt(2)), dimensionless.

    Parameters
    ----------
    x : float or array
        Argument (dimensionless). Finite values only.
    """
    arr = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError("qfunc argument must be finite")
    return 0.5 * erfc(arr / np.sqrt(2.0))


def _check_ebn0(ebn0_db: float | np.ndarray) -> np.ndarray:
    arr = np.atleast_1d(np.asarray(ebn0_db, dtype=float))
    if arr.ndim != 1:
        raise ValueError("ebn0_db must be a scalar or 1-D array of Eb/N0 values in dB")
    if not np.all(np.isfinite(arr)):
        raise ValueError("ebn0_db must be finite (negative dB is allowed, NaN/inf is not)")
    return arr


def _check_rate(code_rate: float) -> float:
    r = float(code_rate)
    if not np.isfinite(r) or not (0.0 < r <= 1.0):
        raise ValueError(f"code_rate must lie in (0, 1], got {code_rate!r}")
    return r


def ebn0_to_esn0_db(ebn0_db: float | np.ndarray, code_rate: float = 1.0) -> np.ndarray:
    """Channel symbol Es/N0 in dB from information-bit Eb/N0 in dB.

    Es/N0 = R * Eb/N0, i.e. ``esn0_db = ebn0_db + 10*log10(R)``. Units: dB.
    """
    arr = _check_ebn0(ebn0_db)
    r = _check_rate(code_rate)
    return arr + 10.0 * np.log10(r)


def bpsk_ber(ebn0_db: float | np.ndarray, code_rate: float = 1.0) -> np.ndarray:
    """Channel bit error probability of coherent BPSK over AWGN, Eq. (1).

    Parameters
    ----------
    ebn0_db : float or 1-D array
        Energy per **information** bit over noise density, Eb/N0, in dB.
    code_rate : float
        Code rate R = k/n in (0, 1]. R = 1 is the uncoded link.

    Returns
    -------
    ndarray
        Channel bit error probability, dimensionless, in [0, 0.5].

    Reference
    ---------
    Proakis & Salehi 2008, Eq. (4.3-13). Same expression and same Eb/N0
    convention as P010 BERBench ``analytic_ber("bpsk", ...)``.
    """
    arr = _check_ebn0(ebn0_db)
    r = _check_rate(code_rate)
    gamma = np.power(10.0, arr / 10.0)
    return qfunc(np.sqrt(2.0 * r * gamma))


def bsc_flip(bits: np.ndarray, p: float, rng: np.random.Generator) -> np.ndarray:
    """Pass a packed-uint8 or 0/1 bit array through a binary symmetric channel.

    Parameters
    ----------
    bits : ndarray of uint8
        Array of 0/1 values (not bit-packed). Any shape.
    p : float
        Crossover probability in [0, 1], dimensionless.
    rng : numpy.random.Generator
        Seeded generator; the caller owns reproducibility.

    Returns
    -------
    ndarray of uint8
        ``bits`` with each position independently flipped with probability p.
    """
    if not 0.0 <= float(p) <= 1.0:
        raise ValueError(f"crossover probability p must lie in [0, 1], got {p!r}")
    arr = np.asarray(bits, dtype=np.uint8)
    if arr.size and arr.max() > 1:
        raise ValueError("bsc_flip expects an array of 0/1 bits, not packed bytes")
    flips = (rng.random(arr.shape) < float(p)).astype(np.uint8)
    return arr ^ flips


def awgn_bpsk_samples(
    bits: np.ndarray, ebn0_db: float, code_rate: float, rng: np.random.Generator
) -> np.ndarray:
    """Matched-filter output samples for BPSK over AWGN, unit signal amplitude.

    Mapping: bit 0 -> +1, bit 1 -> -1. Noise standard deviation is
    ``sigma = 1 / sqrt(2 * R * Eb/N0)`` so that the hard-decision error rate
    of ``sign()`` on the returned samples equals Eq. (1). Units: normalised
    amplitude (dimensionless).
    """
    arr = np.asarray(bits, dtype=np.uint8)
    r = _check_rate(code_rate)
    gamma = 10.0 ** (float(ebn0_db) / 10.0)
    sigma = 1.0 / np.sqrt(2.0 * r * gamma)
    return (1.0 - 2.0 * arr.astype(float)) + sigma * rng.standard_normal(arr.shape)
