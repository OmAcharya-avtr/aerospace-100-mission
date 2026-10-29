"""Measured consequence of a corruption, in kilometres of position error.

This module is the reason the *silently wrong* class in this package is a
measurement and not a label. The class is assigned by propagating the element
set the caller actually has and the uncorrupted element set for the same object
over the same absolute time grid with ``sgp4``, and differencing the two
positions. The number in kilometres is the evidence; the class name is a
threshold applied to it.

The three classes
-----------------
``will_fail_loudly``
    The propagator refused the lines, or returned a non-zero SGP4 error code,
    or returned a non-finite position, at any sample on the grid. This is
    decided by the propagator's own output and by nothing this package
    computes, which is the point: it is the measured answer to "will the
    propagator tell me".

``silently_wrong``
    The propagator reported success at every sample and the maximum position
    difference from the uncorrupted propagation exceeds
    ``silent_threshold_km``.

``benign``
    The propagator reported success at every sample and the maximum position
    difference does not exceed ``silent_threshold_km``.

The threshold is a declared policy value, not a physical constant. Its default
of 1.0 km is stated in the README and swept in
``validation/validate_consequence.py``, because the class boundary moves with
it and a reader who is screening for conjunction assessment and a reader who is
screening for a ground-station pass schedule do not want the same number.

Equation
--------
(11) Position error against the reference propagation:

        d_k = || r_candidate(t_k) - r_reference(t_k) ||_2,  kilometres

    reported as the maximum, the root mean square and the value at the first
    sample over the grid ``t_k``. The norm is Euclidean in the TEME frame of
    date, which is the frame SGP4 returns; no frame transformation is applied
    because the difference of two positions in the same frame does not need
    one.

Units
-----
All errors are kilometres. Horizons and grids are days.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .elements import ElementSet
from .propagate import Propagation, propagate, time_grid

WILL_FAIL_LOUDLY = "will_fail_loudly"
SILENTLY_WRONG = "silently_wrong"
BENIGN = "benign"

CLASSES: tuple[str, ...] = (WILL_FAIL_LOUDLY, SILENTLY_WRONG, BENIGN)

DEFAULT_SILENT_THRESHOLD_KM = 1.0
"""Declared policy threshold on equation (11). Not a physical constant."""


@dataclass(frozen=True)
class Consequence:
    """Measured consequence of using one element set instead of the reference.

    Attributes
    ----------
    consequence_class:
        One of :data:`CLASSES`.
    max_error_km:
        Maximum of equation (11) over the grid. ``nan`` when either
        propagation failed.
    rms_error_km:
        Root mean square of equation (11) over the grid.
    initial_error_km:
        Equation (11) at the first grid sample.
    threshold_km:
        The declared threshold the class was assigned with, carried alongside
        the class so the class cannot be read without it.
    loud_reason:
        The propagator's own reason, empty unless the class is
        ``will_fail_loudly``.
    samples:
        Grid length.
    horizon_days:
        Grid span in days.
    reference_loud:
        ``True`` when the *reference* propagation was itself loud, in which
        case no comparison is possible and the class is
        ``will_fail_loudly`` with that stated. This is a distinct outcome from
        the candidate being loud and is not collapsed into it.
    """

    consequence_class: str
    max_error_km: float
    rms_error_km: float
    initial_error_km: float
    threshold_km: float
    loud_reason: str
    samples: int
    horizon_days: float
    reference_loud: bool


def position_errors_km(candidate: Propagation, reference: Propagation) -> np.ndarray:
    """Equation (11) at every grid sample, kilometres.

    Raises
    ------
    ValueError
        If the two propagations do not share a time grid, because differencing
        positions at different epochs is a silent error of exactly the kind
        this package exists to catch.
    """
    if candidate.times_jd.shape != reference.times_jd.shape:
        raise ValueError(
            "candidate and reference propagations have different grid lengths "
            f"({candidate.times_jd.size} and {reference.times_jd.size})"
        )
    if not np.allclose(candidate.times_jd, reference.times_jd, rtol=0.0, atol=1.0e-9):
        raise ValueError(
            "candidate and reference propagations are on different time grids; "
            "equation (11) differences positions at the same epoch"
        )
    difference = candidate.positions_km - reference.positions_km
    return np.sqrt(np.sum(difference * difference, axis=1))


def measure(
    candidate: ElementSet,
    reference: ElementSet,
    start_jd: float,
    horizon_days: float = 1.0,
    samples: int = 25,
    silent_threshold_km: float = DEFAULT_SILENT_THRESHOLD_KM,
) -> Consequence:
    """Measure the consequence of ``candidate`` against ``reference``.

    Parameters
    ----------
    candidate:
        The element set being screened.
    reference:
        The uncorrupted element set for the same object. In an operational
        setting this is not available, which is exactly why screening is
        needed; it is available here because this package constructs its own
        corpus, and the README says so plainly.
    start_jd:
        First absolute Julian date of the comparison grid, days. This is the
        time the ephemeris is wanted at, not the element set epoch.
    horizon_days:
        Grid span in days.
    samples:
        Grid length, at least 2.
    silent_threshold_km:
        Declared threshold on equation (11). Must be positive and finite.

    Raises
    ------
    ValueError
        If ``silent_threshold_km`` is not positive and finite, or the grid
        parameters are invalid, or the two element sets are for different
        objects.
    """
    if not np.isfinite(silent_threshold_km) or silent_threshold_km <= 0.0:
        raise ValueError(
            "silent_threshold_km must be positive and finite, "
            f"got {silent_threshold_km!r}"
        )
    grid = time_grid(start_jd, horizon_days, samples)
    reference_propagation = propagate(reference, grid)
    candidate_propagation = propagate(candidate, grid)

    if reference_propagation.loud:
        return Consequence(
            consequence_class=WILL_FAIL_LOUDLY,
            max_error_km=float("nan"),
            rms_error_km=float("nan"),
            initial_error_km=float("nan"),
            threshold_km=silent_threshold_km,
            loud_reason=(
                "the REFERENCE element set is itself loud "
                f"({reference_propagation.loud_reason}); no consequence comparison "
                "is possible and this is not the candidate's fault"
            ),
            samples=samples,
            horizon_days=horizon_days,
            reference_loud=True,
        )

    if candidate_propagation.loud:
        return Consequence(
            consequence_class=WILL_FAIL_LOUDLY,
            max_error_km=float("nan"),
            rms_error_km=float("nan"),
            initial_error_km=float("nan"),
            threshold_km=silent_threshold_km,
            loud_reason=candidate_propagation.loud_reason,
            samples=samples,
            horizon_days=horizon_days,
            reference_loud=False,
        )

    errors = position_errors_km(candidate_propagation, reference_propagation)
    maximum = float(np.max(errors))
    return Consequence(
        consequence_class=SILENTLY_WRONG if maximum > silent_threshold_km else BENIGN,
        max_error_km=maximum,
        rms_error_km=float(np.sqrt(np.mean(errors * errors))),
        initial_error_km=float(errors[0]),
        threshold_km=silent_threshold_km,
        loud_reason="",
        samples=samples,
        horizon_days=horizon_days,
        reference_loud=False,
    )
