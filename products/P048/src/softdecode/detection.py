"""Detection model and the energy conventions every number in this package uses.

Noise assumption, written down explicitly
-----------------------------------------
**Thermal-limited, signal-independent additive white Gaussian noise.** After
photodetection and matched filtering of one slot, the sample is

    y = a * h * s + n,     n ~ N(0, sigma**2),

where ``h`` is the normalised irradiance (``E[h] = 1``, :mod:`softdecode.channel`),
``s`` is the slot amplitude (``s in {0, 1}`` for OOK, a one-hot vector for
PPM), ``a`` is the peak electrical amplitude at ``h = 1`` and ``sigma**2`` is
the noise variance of one sample. ``sigma**2`` does **not** depend on ``s`` or
on ``h``.

This is the standard assumption for a receiver whose noise is dominated by the
load resistance and the preamplifier, and it is the assumption under which
Zhu & Kahn derive the free-space optical maximum-likelihood detection and
imperfect-channel-knowledge results (X. Zhu and J. M. Kahn, "Free-space
optical communication through atmospheric turbulence channels", *IEEE
Transactions on Communications* 50(8), 1293-1300, 2002).

**Not modelled.** A shot-noise-limited or avalanche-photodiode receiver has
signal-dependent noise, ``sigma**2 = sigma_th**2 + k * a * h * s``, and a
Poisson or Webb-McIntyre-Conradi count statistic. Every LLR in this package is
derived for the signal-independent case and is *wrong* for the shot-limited
case; that is a different product and the README says so. Background light is
absorbed into ``sigma**2`` only to the extent that it is signal-independent.

Energy conventions
------------------
Fixed here once and used everywhere, so that an Eb/N0 figure in this
repository means one definite thing:

* one-sided noise power spectral density ``N0 = 2 * sigma**2`` (unit-duration
  slot, unit-energy matched filter);
* **OOK**, equiprobable bits: average electrical energy per channel bit at
  ``h = 1`` is ``Ec = a**2 / 2``;
* **M-ary PPM**: one pulse of amplitude ``a`` in one of ``M`` unit-duration
  slots, so energy per channel symbol is ``a**2`` and per channel bit
  ``Ec = a**2 / log2(M)``;
* for a code of rate ``R``, energy per *information* bit is ``Eb = Ec / R``.

Inverting, with ``gamma = Eb/N0`` as a ratio (not dB):

* OOK:     ``a = 2 * sigma * sqrt(R * gamma)``
* M-PPM:   ``a = sigma * sqrt(2 * log2(M) * R * gamma)``
* BPSK (antipodal, ``s in {-1, +1}``): ``a = sigma * sqrt(2 * R * gamma)``

These are conventions, not results. Comparing an Eb/N0 figure from this
repository with one from elsewhere requires checking that the other side uses
the same three definitions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def db_to_ratio(value_db: float | np.ndarray) -> np.ndarray:
    """Convert dB to a power ratio."""
    return np.power(10.0, np.asarray(value_db, dtype=float) / 10.0)


def ratio_to_db(value: float | np.ndarray) -> np.ndarray:
    """Convert a power ratio to dB."""
    v = np.asarray(value, dtype=float)
    if np.any(v <= 0.0):
        raise ValueError("ratio_to_db requires strictly positive input")
    return 10.0 * np.log10(v)


@dataclass(frozen=True)
class DetectionModel:
    """Thermal-limited AWGN detection with a stated energy convention.

    Parameters
    ----------
    sigma:
        Noise standard deviation of one slot sample, in the same arbitrary
        electrical units as the amplitude. Must be > 0.
    """

    sigma: float = 1.0

    def __post_init__(self) -> None:
        s = float(self.sigma)
        if not np.isfinite(s) or s <= 0.0:
            raise ValueError(f"sigma must be a finite positive number, got {self.sigma!r}")

    @property
    def noise_variance(self) -> float:
        """``sigma**2`` of one slot sample."""
        return float(self.sigma) ** 2

    @property
    def n0(self) -> float:
        """One-sided noise PSD ``N0 = 2 sigma**2``."""
        return 2.0 * self.noise_variance

    def ook_amplitude(self, ebn0_db: float, rate: float = 1.0) -> float:
        """Peak OOK amplitude ``a = 2 sigma sqrt(R Eb/N0)`` at ``h = 1``."""
        _check_rate(rate)
        gamma = float(db_to_ratio(ebn0_db))
        return float(2.0 * self.sigma * np.sqrt(rate * gamma))

    def ppm_amplitude(self, ebn0_db: float, order: int, rate: float = 1.0) -> float:
        """Pulse amplitude ``a = sigma sqrt(2 log2(M) R Eb/N0)`` at ``h = 1``."""
        _check_rate(rate)
        m = _check_order(order)
        gamma = float(db_to_ratio(ebn0_db))
        return float(self.sigma * np.sqrt(2.0 * np.log2(m) * rate * gamma))

    def bpsk_amplitude(self, ebn0_db: float, rate: float = 1.0) -> float:
        """Antipodal amplitude ``a = sigma sqrt(2 R Eb/N0)`` at ``h = 1``."""
        _check_rate(rate)
        gamma = float(db_to_ratio(ebn0_db))
        return float(self.sigma * np.sqrt(2.0 * rate * gamma))

    def ook_ebn0_db(self, amplitude: float, rate: float = 1.0) -> float:
        """Inverse of :meth:`ook_amplitude`, in dB."""
        _check_rate(rate)
        a = float(amplitude)
        if a <= 0.0:
            raise ValueError(f"amplitude must be positive, got {amplitude!r}")
        return float(ratio_to_db(a**2 / (4.0 * self.noise_variance * rate)))


def _check_rate(rate: float) -> float:
    r = float(rate)
    if not np.isfinite(r) or not 0.0 < r <= 1.0:
        raise ValueError(f"rate must lie in (0, 1], got {rate!r}")
    return r


def _check_order(order: int) -> int:
    m = int(order)
    if m < 2 or (m & (m - 1)) != 0:
        raise ValueError(f"PPM order must be a power of two and at least 2, got {order!r}")
    return m
