"""Thin, failure-reporting wrapper over the ``sgp4`` propagator.

This package does not implement SGP4 and does not try to. ``sgp4`` 2.27 is the
reference implementation of Vallado et al. (2006) and it is both correct and
mature; the point of this module is to call it in a way that records what it
did rather than throwing the record away.

Two behaviours of ``sgp4.api.Satrec.twoline2rv`` were measured in this
environment and they are the reason the rest of this package exists. Both are
recorded in ``validation/validate_consequence_output.txt``.

1. **It does not verify the checksum.** A line whose checksum digit is wrong is
   parsed and propagated without complaint.
2. **It accepts lines it cannot possibly mean.** Two lines of 69 spaces parse
   to object number 0 with a mean motion of 0.0, and a line whose characters
   have shifted by one column parses to a mean motion of 5.50 rev/day where the
   uncorrupted line says 15.50.

Neither is a defect in ``sgp4``: parsing and propagating is what it is for, and
screening is a separate job. This module makes the separation explicit.

Reference
---------
Vallado, D. A., Crawford, P., Hujsak, R. and Kelso, T. S. (2006), *Revisiting
Spacetrack Report #3*, AIAA 2006-6753, which is the paper the ``sgp4``
distribution implements. Error codes 1 to 6 are the ones listed in
``sgp4.api.SGP4_ERRORS``.

Units
-----
Julian dates are days. Positions are kilometres in the TEME frame of date, and
velocities kilometres per second, which is what SGP4 returns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sgp4.api import SGP4_ERRORS, Satrec

from .elements import ElementSet


@dataclass(frozen=True)
class Propagation:
    """Result of propagating one element set over a time grid.

    Attributes
    ----------
    accepted:
        ``False`` when the propagator refused the lines outright, which is the
        loudest failure available to it.
    refusal:
        The exception text when ``accepted`` is ``False``, otherwise empty.
    times_jd:
        The absolute Julian dates propagated to, days.
    error_codes:
        One SGP4 error code per time. 0 means the propagator reported success.
    positions_km:
        Shape ``(len(times_jd), 3)`` TEME positions in kilometres. Rows for
        which the propagator failed are ``nan``.
    loud:
        ``True`` when the propagator refused the lines, returned a non-zero
        error code at any time, or returned a non-finite position. This is the
        measured answer to "will the propagator tell me", and it is measured
        from the propagator's own output and from nothing else.
    loud_reason:
        Why ``loud`` is set, empty when it is not.
    """

    accepted: bool
    refusal: str
    times_jd: np.ndarray
    error_codes: np.ndarray
    positions_km: np.ndarray
    loud: bool
    loud_reason: str


def make_satrec(element_set: ElementSet) -> tuple[Satrec | None, str]:
    """Hand the element set's raw lines to ``sgp4``, reporting any refusal.

    The *raw lines* are used deliberately, not this package's parsed fields:
    the question this module answers is what the propagator will do with the
    text the caller actually has.

    Returns
    -------
    tuple[Satrec | None, str]
        The propagator object and an empty string, or ``None`` and the
        exception text.
    """
    try:
        return Satrec.twoline2rv(element_set.line1, element_set.line2), ""
    except Exception as exception:  # noqa: BLE001 - the refusal text is the result
        return None, f"{type(exception).__name__}: {exception}"


def propagate(element_set: ElementSet, times_jd: np.ndarray) -> Propagation:
    """Propagate ``element_set`` to each absolute Julian date in ``times_jd``.

    Parameters
    ----------
    element_set:
        Element set whose raw lines are given to the propagator.
    times_jd:
        Absolute Julian dates, days. Must be one-dimensional, finite and
        non-empty.

    Raises
    ------
    ValueError
        If ``times_jd`` is empty, not one-dimensional, or not all finite.
    """
    times = np.asarray(times_jd, dtype=float)
    if times.ndim != 1:
        raise ValueError(f"times_jd must be one-dimensional, got shape {times.shape}")
    if times.size == 0:
        raise ValueError("times_jd must contain at least one epoch")
    if not np.isfinite(times).all():
        raise ValueError("times_jd must be all finite")

    satrec, refusal = make_satrec(element_set)
    if satrec is None:
        nan = np.full((times.size, 3), np.nan)
        return Propagation(
            accepted=False,
            refusal=refusal,
            times_jd=times,
            error_codes=np.full(times.size, -1, dtype=int),
            positions_km=nan,
            loud=True,
            loud_reason=f"the propagator refused the element set lines ({refusal})",
        )

    whole = np.floor(times - 0.5) + 0.5
    fraction = times - whole
    codes, positions, _velocities = satrec.sgp4_array(whole, fraction)
    codes = np.asarray(codes, dtype=int)
    positions = np.asarray(positions, dtype=float)

    reasons: list[str] = []
    nonzero = codes != 0
    if nonzero.any():
        first = int(np.flatnonzero(nonzero)[0])
        code = int(codes[first])
        reasons.append(
            f"SGP4 error code {code} at sample {first} of {times.size} "
            f"({SGP4_ERRORS.get(code, 'unlisted code')})"
        )
    if not np.isfinite(positions).all():
        reasons.append("the propagator returned a non-finite position")

    return Propagation(
        accepted=True,
        refusal="",
        times_jd=times,
        error_codes=codes,
        positions_km=positions,
        loud=bool(reasons),
        loud_reason="; ".join(reasons),
    )


def time_grid(start_jd: float, horizon_days: float, samples: int) -> np.ndarray:
    """Uniform absolute Julian date grid, inclusive of both endpoints.

    Raises
    ------
    ValueError
        If ``samples`` is below 2, ``horizon_days`` is not positive, or
        ``start_jd`` is not finite.
    """
    if samples < 2:
        raise ValueError(f"samples must be at least 2, got {samples}")
    if not np.isfinite(horizon_days) or horizon_days <= 0.0:
        raise ValueError(f"horizon_days must be positive and finite, got {horizon_days!r}")
    if not np.isfinite(start_jd):
        raise ValueError(f"start_jd must be finite, got {start_jd!r}")
    return start_jd + np.linspace(0.0, horizon_days, samples)
