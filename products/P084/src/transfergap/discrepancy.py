"""The paired simulation-to-field record, its declared range, and its validation.

What a gap is here
------------------
A digital twin predicts a scalar quantity; a field test measures the same
quantity on the same article under the same operating condition. The
**discrepancy** or **gap** of a paired measurement is

    d_i = min(|y_i - s_i|, R)                                             (1)

where ``y_i`` is the field measurement, ``s_i`` is the twin's prediction for
the same condition, and ``R`` is a **declared** finite range: the largest
discrepancy the measurement chain can report at all. Units are whatever the
predicted quantity is measured in; this package never assumes a unit and
carries the caller's unit string through every result.

``R`` is not a fitted quantity and not a convenience. Every bound in
:mod:`transfergap.bounds` that is distribution-free rests on the random
variable being **bounded**, and ``R`` is that bound. Hoeffding's inequality
with the wrong ``R`` is not conservative, it is wrong. The honest source of
``R`` is the instrument: a strain channel calibrated to +/- 40 kN.m cannot
report a discrepancy above 40 kN.m regardless of what the twin says, and a
discrepancy that saturates the channel is censored rather than measured. The
clipping in equation (1) is therefore a statement about the measurement chain
and not a smoothing step, and :func:`clip_fraction` reports how much of the
sample it touched so that a censored sample can be recognised.

Assumptions and validity range
------------------------------
- ``d_i`` lies in ``[0, R]`` with ``R > 0`` finite and declared in advance.
- The pairs are drawn from one operating-condition population. Mixing two
  test campaigns flown to different cards breaks that and is exactly what
  :mod:`transfergap.exchangeability` exists to detect.
- Independence is assumed by :mod:`transfergap.bounds`; the weaker assumption
  of exchangeability is what :mod:`transfergap.resample` needs. Neither is
  assumed silently: both are stated here and tested there.
- Nothing in this module models the twin. The gap is an observable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "GapSample",
    "clip_fraction",
    "gaps_from_pairs",
    "validate_confidence",
    "validate_gaps",
]


def validate_gaps(gaps: np.ndarray, bound_range: float) -> np.ndarray:
    """Return ``gaps`` as a validated float64 1-D array in ``[0, bound_range]``.

    Parameters
    ----------
    gaps:
        Discrepancy magnitudes, any shape coercible to 1-D, same unit as the
        predicted quantity.
    bound_range:
        The declared range ``R`` of equation (1), strictly positive, same unit.

    Raises
    ------
    TypeError
        If ``gaps`` cannot be read as a float array.
    ValueError
        If ``bound_range`` is not finite and strictly positive, if ``gaps`` is
        not one-dimensional or is empty, or if any value is non-finite,
        negative or above ``bound_range``.
    """
    if not np.isfinite(bound_range) or bound_range <= 0.0:
        raise ValueError(
            f"bound_range must be a finite positive number in the unit of the measured "
            f"quantity; got {bound_range!r}. It is the largest discrepancy the "
            f"measurement chain can report, and it is declared, never fitted."
        )
    try:
        array = np.asarray(gaps, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError(f"gaps must be coercible to a float array: {error}") from error
    if array.ndim != 1:
        raise ValueError(f"gaps must be one-dimensional; got shape {array.shape}")
    if array.size == 0:
        raise ValueError("gaps is empty; a bound cannot be computed from no measurements")
    if not np.all(np.isfinite(array)):
        raise ValueError("gaps contains a non-finite value (nan or inf)")
    if np.any(array < 0.0):
        raise ValueError(
            "gaps contains a negative value; a discrepancy magnitude is non-negative "
            "by equation (1)"
        )
    if np.any(array > bound_range):
        worst = float(np.max(array))
        raise ValueError(
            f"gaps contains {worst!r}, above the declared range {bound_range!r}. A "
            f"distribution-free bound with a range smaller than the data is not "
            f"conservative, it is wrong."
        )
    return array


def validate_confidence(confidence: float, name: str = "confidence") -> float:
    """Return ``confidence`` validated to lie strictly inside ``(0, 1)``."""
    value = float(confidence)
    if not np.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(f"{name} must lie strictly inside (0, 1); got {confidence!r}")
    return value


def gaps_from_pairs(
    field: np.ndarray, simulated: np.ndarray, bound_range: float
) -> np.ndarray:
    """Discrepancies of equation (1) from paired field and simulated values.

    Parameters
    ----------
    field, simulated:
        Equal-length 1-D arrays of the measured and predicted quantity, same
        unit.
    bound_range:
        The declared range ``R``, same unit, strictly positive.

    Returns
    -------
    numpy.ndarray
        ``min(|field - simulated|, R)``, float64, same unit.
    """
    measured = np.asarray(field, dtype=np.float64)
    predicted = np.asarray(simulated, dtype=np.float64)
    if measured.shape != predicted.shape:
        raise ValueError(
            f"field and simulated must have the same shape; got {measured.shape} "
            f"and {predicted.shape}. A gap is defined only for a PAIRED measurement."
        )
    if measured.ndim != 1:
        raise ValueError(f"field and simulated must be one-dimensional; got {measured.shape}")
    if not np.all(np.isfinite(measured)) or not np.all(np.isfinite(predicted)):
        raise ValueError("field and simulated must both be finite")
    if not np.isfinite(bound_range) or bound_range <= 0.0:
        raise ValueError(f"bound_range must be finite and positive; got {bound_range!r}")
    return np.minimum(np.abs(measured - predicted), float(bound_range))


def clip_fraction(field: np.ndarray, simulated: np.ndarray, bound_range: float) -> float:
    """Fraction of pairs whose raw discrepancy exceeds the declared range.

    A non-zero value means the sample is **censored**: those measurements
    report ``R`` rather than their true discrepancy, every bound computed from
    them understates the gap, and the right response is a wider channel rather
    than a tighter bound. Reported, never silently absorbed.
    """
    measured = np.asarray(field, dtype=np.float64)
    predicted = np.asarray(simulated, dtype=np.float64)
    if measured.shape != predicted.shape:
        raise ValueError("field and simulated must have the same shape")
    if measured.size == 0:
        raise ValueError("field is empty")
    return float(np.mean(np.abs(measured - predicted) > float(bound_range)))


@dataclass(frozen=True)
class GapSample:
    """A set of paired sim-to-field discrepancies with their operating conditions.

    Attributes
    ----------
    gaps:
        ``(n,)`` discrepancies of equation (1), in ``unit``.
    features:
        ``(n, p)`` operating conditions, one row per pair. Dimensionless or
        mixed units; the learned model standardises internally and the
        distribution-free bounds ignore them entirely.
    feature_names:
        ``p`` names, in column order.
    unit:
        The unit of ``gaps`` and of ``bound_range``, carried through every
        result so that no number in this package is printed without one.
    bound_range:
        The declared range ``R`` of equation (1).
    """

    gaps: np.ndarray
    features: np.ndarray
    feature_names: tuple[str, ...]
    unit: str
    bound_range: float

    def __post_init__(self) -> None:
        gaps = validate_gaps(self.gaps, self.bound_range)
        features = np.asarray(self.features, dtype=np.float64)
        if features.ndim != 2:
            raise ValueError(f"features must be two-dimensional; got shape {features.shape}")
        if features.shape[0] != gaps.size:
            raise ValueError(
                f"features has {features.shape[0]} rows but there are {gaps.size} gaps; "
                f"every discrepancy must carry the operating condition it was measured at"
            )
        if features.shape[1] != len(self.feature_names):
            raise ValueError(
                f"features has {features.shape[1]} columns but "
                f"{len(self.feature_names)} names were given"
            )
        if not np.all(np.isfinite(features)):
            raise ValueError("features contains a non-finite value (nan or inf)")
        object.__setattr__(self, "gaps", gaps)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "feature_names", tuple(self.feature_names))

    @property
    def size(self) -> int:
        """Number of paired measurements."""
        return int(self.gaps.size)

    def subset(self, index: np.ndarray) -> GapSample:
        """A new :class:`GapSample` holding the rows selected by ``index``."""
        selector = np.asarray(index)
        return GapSample(
            gaps=self.gaps[selector],
            features=self.features[selector],
            feature_names=self.feature_names,
            unit=self.unit,
            bound_range=self.bound_range,
        )
