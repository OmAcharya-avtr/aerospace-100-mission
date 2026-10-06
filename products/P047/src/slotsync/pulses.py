"""Finite-support pulse shapes for slot and symbol timing analysis.

All shapes are expressed as a function of time ``t`` measured in **symbol
periods** (dimensionless: ``t = 1.0`` is one symbol period after the pulse
centre).  Every shape is even, peaks at ``t = 0`` and is zero outside
``|t| > half_support``, so a sample at time ``t`` is influenced by at most
``2*ceil(half_support)+1`` neighbouring symbols.  Finite support is a
deliberate restriction: it makes the inter-symbol interference span exact and
therefore makes the S-curve in :mod:`slotsync.scurve` an exact ensemble
average over an enumerable set of data patterns rather than a Monte Carlo
estimate.

Units and conventions
---------------------
* ``t`` - time in symbol periods (dimensionless).
* amplitude - dimensionless; every shape is normalised to ``p(0) == 1``.
* ``half_support`` - symbol periods; ``p(t) == 0`` exactly for
  ``|t| > half_support``.

The Nyquist raised-cosine shape (:func:`nyquist_raised_cosine`) is the
frequency-domain raised cosine, which has infinite support and is **truncated**
here.  Truncation is a documented approximation: the truncated shape is no
longer exactly Nyquist, so ``p(+-1)`` is not exactly zero.  The measured value
is reported by :attr:`PulseShape.value_at_one_symbol` so a user can see the
residual rather than assume it away.

References
----------
No pulse shape in this module is attributed to a specific reference.  The
rectangular, triangular, raised-cosine-in-time and half-sine shapes are
elementary; the frequency-domain raised cosine is standard textbook material
and is implemented here from its usual closed form, which the module's own
tests check against the defining properties (unit peak, evenness, zero
crossings at integer multiples of the symbol period for zero excess
bandwidth).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "PulseShape",
    "half_sine",
    "nyquist_raised_cosine",
    "pulse_by_name",
    "raised_cosine_time",
    "rectangular",
    "triangular",
]


@dataclass(frozen=True)
class PulseShape:
    """A finite-support, even, unit-peak pulse shape in symbol-period units.

    Parameters
    ----------
    name
        Short identifier, used in reports and figure labels.
    half_support
        Symbol periods.  ``amplitude(t)`` is zero for ``|t| > half_support``.
    _function
        Callable evaluated on an array of times in symbol periods.  It is not
        required to zero itself outside the support; :meth:`amplitude` applies
        the support mask.
    """

    name: str
    half_support: float
    _function: Callable[[np.ndarray], np.ndarray] = field(repr=False)

    def __post_init__(self) -> None:
        if not math.isfinite(self.half_support) or self.half_support <= 0.0:
            raise ValueError(
                f"half_support must be a positive finite number of symbol periods, "
                f"got {self.half_support!r}"
            )

    def amplitude(self, t: np.ndarray | float) -> np.ndarray:
        """Pulse amplitude (dimensionless) at time(s) ``t`` in symbol periods."""
        times = np.asarray(t, dtype=float)
        inside = np.abs(times) <= self.half_support
        out = np.zeros(times.shape, dtype=float)
        if inside.any():
            out[inside] = self._function(times[inside])
        return out

    @property
    def isi_span_symbols(self) -> int:
        """Number of neighbouring symbols on each side that can reach a sample."""
        return int(math.ceil(self.half_support))

    @property
    def value_at_one_symbol(self) -> float:
        """``p(1.0)``: the residual inter-symbol interference one symbol away."""
        return float(self.amplitude(np.array([1.0]))[0])

    def energy(self, oversample: int = 4096) -> float:
        """Pulse energy ``integral p(t)^2 dt`` in symbol periods, by trapezoid rule."""
        if oversample < 16:
            raise ValueError(f"oversample must be at least 16, got {oversample}")
        t = np.linspace(-self.half_support, self.half_support, oversample)
        return float(np.trapezoid(self.amplitude(t) ** 2, t))


def rectangular(width: float = 1.0) -> PulseShape:
    """Non-return-to-zero rectangle of the given width in symbol periods.

    ``p(t) = 1`` for ``|t| < width/2``, else 0.  The canonical OOK / NRZ shape.
    Its derivative is zero almost everywhere, so a *difference* detector sees no
    slope inside the pulse: the early-late S-curve of a rectangle is flat over
    the pulse interior, which is why this shape is included - it is the case
    where the detector gain collapses.
    """
    if width <= 0.0:
        raise ValueError(f"width must be positive in symbol periods, got {width!r}")
    half = 0.5 * width
    return PulseShape("rect", half, lambda t: np.where(np.abs(t) < half, 1.0, 0.0))


def triangular(half_width: float = 1.0) -> PulseShape:
    """Symmetric triangle, ``p(t) = max(0, 1 - |t| / half_width)``.

    The hand-computable reference shape.  Because it is exactly linear in
    ``|t|``, both the early-late and the Mueller-Mueller S-curves are exactly
    linear near the origin and their slopes can be written down by hand; see
    ``tests/test_scurve.py`` and ``docs/TIMING_MODEL.md`` section 2.
    """
    if half_width <= 0.0:
        raise ValueError(f"half_width must be positive in symbol periods, got {half_width!r}")
    return PulseShape(
        "tri", float(half_width), lambda t: 1.0 - np.abs(t) / half_width
    )


def raised_cosine_time(half_width: float = 1.0) -> PulseShape:
    """Raised cosine **in time**, ``p(t) = (1 + cos(pi t / half_width)) / 2``.

    Finite support, continuous first derivative, zero slope at the edges.  This
    is not the Nyquist raised cosine (that one is a raised cosine in
    *frequency*); use :func:`nyquist_raised_cosine` for the band-limited case.
    """
    if half_width <= 0.0:
        raise ValueError(f"half_width must be positive in symbol periods, got {half_width!r}")
    return PulseShape(
        "rc-time",
        float(half_width),
        lambda t: 0.5 * (1.0 + np.cos(np.pi * t / half_width)),
    )


def half_sine(width: float = 1.0) -> PulseShape:
    """Half-sine (return-to-zero) pulse, ``p(t) = cos(pi t / width)`` on the slot.

    Zero at the slot edges, unit peak at the centre.  A common idealisation of
    an optical return-to-zero slot.
    """
    if width <= 0.0:
        raise ValueError(f"width must be positive in symbol periods, got {width!r}")
    half = 0.5 * width
    return PulseShape("half-sine", half, lambda t: np.cos(np.pi * t / width))


def nyquist_raised_cosine(rolloff: float = 0.5, truncate_symbols: float = 4.0) -> PulseShape:
    """Frequency-domain raised cosine, truncated to ``+-truncate_symbols``.

    ``p(t) = sinc(t) * cos(pi a t) / (1 - (2 a t)^2)`` with ``a`` the rolloff.
    For ``a = 0`` this degenerates to ``sinc(t)``.  The untruncated shape has
    ``p(k) = 0`` for every non-zero integer ``k``; truncation breaks that only
    through the window edge, and :attr:`PulseShape.value_at_one_symbol` reports
    the residual.

    Parameters
    ----------
    rolloff
        Excess-bandwidth factor in ``[0, 1]``, dimensionless.
    truncate_symbols
        Half-support in symbol periods.  Smaller values make the shape cheaper
        and less Nyquist; 4 symbols is the value used throughout this package.

    Validity
    --------
    Because this shape is truncated it is the one case in this module where the
    pulse evaluated by :mod:`slotsync.scurve` differs from the mathematical
    object the name refers to.  Every result computed with it is a result about
    the truncated shape.
    """
    if not 0.0 <= rolloff <= 1.0:
        raise ValueError(f"rolloff must lie in [0, 1], got {rolloff!r}")
    if truncate_symbols < 1.0:
        raise ValueError(
            f"truncate_symbols must be at least 1 symbol period, got {truncate_symbols!r}"
        )

    def shape(t: np.ndarray) -> np.ndarray:
        sinc = np.sinc(t)
        if rolloff == 0.0:
            return sinc
        denom = 1.0 - (2.0 * rolloff * t) ** 2
        numer = np.cos(np.pi * rolloff * t)
        out = np.empty_like(t)
        singular = np.abs(denom) < 1.0e-9
        regular = ~singular
        out[regular] = sinc[regular] * numer[regular] / denom[regular]
        if singular.any():
            # l'Hopital at t = +-1/(2a): p = sinc(t) * pi / 4 * sin(pi/2) / (2a)
            # evaluated as a short-baseline central difference to avoid a
            # separate closed form that would need its own check.
            eps = 1.0e-6
            ts = t[singular]
            left = np.sinc(ts - eps) * np.cos(np.pi * rolloff * (ts - eps)) / (
                1.0 - (2.0 * rolloff * (ts - eps)) ** 2
            )
            right = np.sinc(ts + eps) * np.cos(np.pi * rolloff * (ts + eps)) / (
                1.0 - (2.0 * rolloff * (ts + eps)) ** 2
            )
            out[singular] = 0.5 * (left + right)
        return out

    return PulseShape(f"nyq-rc-{rolloff:g}", float(truncate_symbols), shape)


_BUILTIN: dict[str, Callable[[], PulseShape]] = {
    "rect": rectangular,
    "tri": triangular,
    "rc-time": raised_cosine_time,
    "half-sine": half_sine,
    "nyq-rc": lambda: nyquist_raised_cosine(0.5, 4.0),
}


def pulse_by_name(name: str) -> PulseShape:
    """Look up a pulse shape by its short name, for the CLI and examples.

    Accepted names: ``rect``, ``tri``, ``rc-time``, ``half-sine``, ``nyq-rc``.
    """
    try:
        return _BUILTIN[name]()
    except KeyError:
        raise ValueError(
            f"unknown pulse shape {name!r}; choose one of {sorted(_BUILTIN)}"
        ) from None
