"""Coherent constellations, Gray mapping and a Monte Carlo uncoded BER measurement.

Why a Monte Carlo and not a formula
-----------------------------------
The MODCOD thresholds that drive every policy in this package are **measured**
here rather than quoted from a standard's table. The closed forms below exist
and are used, but only as a *check* on the measurement
(``validation/validate_modcod_thresholds.py``), never as the source of a
threshold. A table of thresholds taken from a standard would describe that
standard's codes, not the codes in :mod:`acmpilot.coding`.

Signal model
------------
Unit-average-energy complex constellation ``s``, complex AWGN:

    y = s + n,    n ~ CN(0, N0/Es)                                        (1)

so the noise variance per real dimension is ``1 / (2 * Es/N0)`` and the single
parameter of the measurement is ``Es/N0`` in dB, called ``snr_db`` throughout.
Detection is minimum Euclidean distance over the full constellation.

Gray mapping is used for every constellation, so that a nearest-neighbour
symbol error costs one bit error in the high-SNR limit. BPSK and QPSK carry
natural Gray maps; 8-PSK uses the reflected binary code around the circle;
16-QAM is the Cartesian product of two Gray-coded 4-PAM axes.

Closed forms used only as checks
--------------------------------
Standard results for Gray-coded constellations in complex AWGN, all with
``Es/N0 = gamma`` and ``Q(x) = 0.5*erfc(x/sqrt(2))``:

    BPSK   BER = Q(sqrt(2*gamma))                                exact    (2)
    QPSK   BER = Q(sqrt(gamma))                                   exact    (3)
    M-QAM  SER = 1 - (1 - p)**2,
           p   = 2*(1 - 1/sqrt(M)) * Q(sqrt(3*gamma/(M-1)))      exact    (4)
           BER ~ SER / log2(M)                             high-SNR only
    M-PSK  SER ~ 2*Q(sqrt(2*gamma)*sin(pi/M))          nearest-neighbour  (5)
           BER ~ SER / log2(M)                             high-SNR only

(2)-(5) are textbook results; see J. G. Proakis and M. Salehi, *Digital
Communications* (McGraw-Hill), chapter on optimum receivers for AWGN channels.
No page or edition-specific number is quoted here. (2) and (3) are exact for
Gray-coded BPSK/QPSK; (4) is exact for square-QAM *symbol* error rate, and the
division by ``log2(M)`` to get a bit error rate is a high-SNR approximation;
(5) is the nearest-neighbour union bound approximation and is not exact at any
SNR. The validation script states which comparisons are exact and which are
approximate, and reports the measured discrepancy for both.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import special


def _gray_code(n_bits: int) -> np.ndarray:
    """Reflected binary (Gray) code words 0..2**n_bits-1 as integers."""
    idx = np.arange(1 << n_bits, dtype=np.int64)
    return idx ^ (idx >> 1)


def _bits_of(values: np.ndarray, n_bits: int) -> np.ndarray:
    """Unpack ``values`` to a (len, n_bits) bit array, most significant first."""
    shifts = np.arange(n_bits - 1, -1, -1, dtype=np.int64)
    return ((values[:, None] >> shifts) & 1).astype(np.uint8)


@dataclass(frozen=True)
class Constellation:
    """A Gray-mapped unit-average-energy complex constellation.

    Attributes
    ----------
    name
        Short label, e.g. ``"QPSK"``.
    points
        Complex constellation points, mean squared magnitude exactly 1.
    bit_map
        ``(M, bits_per_symbol)`` uint8 array; row ``i`` is the bit label of
        ``points[i]``.
    """

    name: str
    points: np.ndarray
    bit_map: np.ndarray

    @property
    def order(self) -> int:
        """Constellation order M (number of points)."""
        return int(self.points.size)

    @property
    def bits_per_symbol(self) -> int:
        """log2(M), bits per channel symbol."""
        return int(self.bit_map.shape[1])


def _normalise(points: np.ndarray) -> np.ndarray:
    return points / np.sqrt(float(np.mean(np.abs(points) ** 2)))


def _psk(order: int, name: str) -> Constellation:
    n_bits = int(np.log2(order))
    angles = 2.0 * np.pi * np.arange(order) / order
    points = np.exp(1j * angles)
    labels = _gray_code(n_bits)
    return Constellation(name=name, points=_normalise(points), bit_map=_bits_of(labels, n_bits))


def _square_qam(order: int, name: str) -> Constellation:
    side = int(round(np.sqrt(order)))
    if side * side != order:
        raise ValueError(f"{order}-QAM is not square")
    half = int(np.log2(side))
    pam = 2.0 * np.arange(side) - (side - 1)
    gray = _gray_code(half)
    # Gray-coded PAM: level index k carries label gray[k].
    i_grid, q_grid = np.meshgrid(pam, pam, indexing="ij")
    points = (i_grid + 1j * q_grid).ravel()
    i_lab, q_lab = np.meshgrid(gray, gray, indexing="ij")
    labels = (i_lab.ravel().astype(np.int64) << half) | q_lab.ravel().astype(np.int64)
    return Constellation(
        name=name, points=_normalise(points), bit_map=_bits_of(labels, 2 * half)
    )


_BUILDERS = {
    "BPSK": lambda: Constellation(
        name="BPSK",
        points=np.array([-1.0 + 0j, 1.0 + 0j]),
        bit_map=np.array([[0], [1]], dtype=np.uint8),
    ),
    "QPSK": lambda: _psk(4, "QPSK"),
    "8PSK": lambda: _psk(8, "8PSK"),
    "16QAM": lambda: _square_qam(16, "16QAM"),
}

#: Constellation names this package ships, in increasing spectral efficiency.
CONSTELLATION_NAMES: tuple[str, ...] = ("BPSK", "QPSK", "8PSK", "16QAM")


def constellation(name: str) -> Constellation:
    """Look up a shipped constellation by name.

    Raises
    ------
    ValueError
        If ``name`` is not one of :data:`CONSTELLATION_NAMES`.
    """
    try:
        return _BUILDERS[name]()
    except KeyError:
        raise ValueError(
            f"unknown constellation {name!r}; available: {sorted(_BUILDERS)}"
        ) from None


def qfunc(x: np.ndarray | float) -> np.ndarray | float:
    """Gaussian tail ``Q(x) = 0.5*erfc(x/sqrt(2))``, dimensionless."""
    return 0.5 * special.erfc(np.asarray(x, dtype=float) / np.sqrt(2.0))


def theoretical_ber(name: str, snr_db: np.ndarray | float) -> np.ndarray:
    """Closed-form BER for ``name`` at ``Es/N0 = snr_db`` dB, equations (2)-(5).

    Exact for BPSK and QPSK; a high-SNR approximation for 8PSK and 16QAM. Used
    as a check on :func:`measure_ber`, never as a threshold source.
    """
    gamma = 10.0 ** (np.asarray(snr_db, dtype=float) / 10.0)
    if name == "BPSK":
        return np.asarray(qfunc(np.sqrt(2.0 * gamma)), dtype=float)
    if name == "QPSK":
        return np.asarray(qfunc(np.sqrt(gamma)), dtype=float)
    if name == "16QAM":
        p = 2.0 * (1.0 - 1.0 / 4.0) * np.asarray(qfunc(np.sqrt(3.0 * gamma / 15.0)))
        ser = 1.0 - (1.0 - p) ** 2
        return np.asarray(ser / 4.0, dtype=float)
    if name == "8PSK":
        ser = 2.0 * np.asarray(qfunc(np.sqrt(2.0 * gamma) * np.sin(np.pi / 8.0)))
        return np.asarray(ser / 3.0, dtype=float)
    raise ValueError(f"no closed form shipped for {name!r}")


def simulate_bit_errors(
    name: str,
    snr_db: float,
    n_symbols: int,
    *,
    rng: np.random.Generator,
    chunk: int = 50_000,
) -> np.ndarray:
    """Per-bit error indicators for ``n_symbols`` symbols of ``name`` at ``snr_db`` dB.

    Equation (1) with minimum-distance detection, exactly as :func:`measure_ber`.
    This returns the error *positions* rather than only the rate, so that a
    caller can group them into code symbols and test the block-error
    combinatorics of :mod:`acmpilot.coding` against the real modem.

    Returns
    -------
    ndarray of bool, shape ``(n_symbols * bits_per_symbol,)``
        Bit stream order is symbol-major, most significant bit of each symbol
        first. Consecutive bits from one modulation symbol are therefore
        **correlated**, which is what a bit interleaver exists to remove.
    """
    if n_symbols < 1:
        raise ValueError(f"n_symbols must be >= 1, got {n_symbols}")
    con = constellation(name)
    points, bits, m = con.points, con.bit_map, con.order
    sigma = float(np.sqrt(1.0 / (2.0 * 10.0 ** (snr_db / 10.0))))
    pieces = []
    sent = 0
    while sent < n_symbols:
        n = int(min(chunk, n_symbols - sent))
        idx = rng.integers(0, m, size=n)
        noise = sigma * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
        y = points[idx] + noise
        dec = np.argmin(np.abs(y[:, None] - points[None, :]) ** 2, axis=1)
        pieces.append((bits[idx] != bits[dec]).ravel())
        sent += n
    return np.concatenate(pieces)


def measure_ber(
    name: str,
    snr_db: float,
    *,
    rng: np.random.Generator,
    target_errors: int = 2000,
    min_symbols: int = 20_000,
    max_symbols: int = 400_000,
    chunk: int = 50_000,
) -> tuple[float, int, int]:
    """Monte Carlo uncoded bit error rate of ``name`` at ``Es/N0 = snr_db`` dB.

    Simulation stops once ``target_errors`` bit errors have been observed and at
    least ``min_symbols`` symbols sent, or when ``max_symbols`` is reached,
    whichever comes first. Equal a-priori symbols are drawn uniformly, so the
    estimate is unbiased.

    Returns
    -------
    (ber, n_bit_errors, n_bits)
        ``ber`` dimensionless; ``n_bits`` is the number of bits actually sent,
        from which the binomial standard error is
        ``sqrt(ber*(1-ber)/n_bits)``.
    """
    if max_symbols < min_symbols:
        raise ValueError(
            f"max_symbols ({max_symbols}) must be >= min_symbols ({min_symbols})"
        )
    con = constellation(name)
    points = con.points
    bits = con.bit_map
    m = con.order
    k = con.bits_per_symbol
    sigma = float(np.sqrt(1.0 / (2.0 * 10.0 ** (snr_db / 10.0))))
    errors = 0
    sent = 0
    while sent < max_symbols and not (errors >= target_errors and sent >= min_symbols):
        n = int(min(chunk, max_symbols - sent))
        idx = rng.integers(0, m, size=n)
        noise = sigma * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
        y = points[idx] + noise
        # Minimum-distance detection; M <= 16 so the full distance matrix is cheap.
        dec = np.argmin(np.abs(y[:, None] - points[None, :]) ** 2, axis=1)
        errors += int(np.count_nonzero(bits[idx] != bits[dec]))
        sent += n
    n_bits = sent * k
    return errors / n_bits, errors, n_bits
