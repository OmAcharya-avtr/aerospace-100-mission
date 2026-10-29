"""Epoch decoding and staleness against a declared age requirement.

The epoch of a two-line element set is a two-digit year and a fractional day of
year. Two things about it are worth stating rather than leaving implicit.

**The year is ambiguous by construction.** Two digits cannot distinguish 1998
from 2098. The convention this module applies is the one stated in Vallado,
D. A., Crawford, P., Hujsak, R. and Kelso, T. S. (2006), *Revisiting Spacetrack
Report #3*, AIAA 2006-6753, section 2: values 57 to 99 are 1957 to 1999 and
values 00 to 56 are 2000 to 2056. The convention has a cliff at 2057 and this
package reports that as a determinability limit rather than a footnote.

**Staleness is not a property of the element set.** An element set three weeks
old is fine for a ground-station schedule and useless for a conjunction
assessment. The age is computed here; whether that age is acceptable is a
requirement the caller declares, and the screen reports the requirement
alongside the verdict so that a report cannot be read without it.

Equations
---------
(5) Julian date of a Gregorian calendar date at 0h UT, by the standard
    integer-arithmetic form:

        JD = 367 Y - floor(7 (Y + floor((M + 9)/12)) / 4) + floor(275 M / 9)
             + D + 1721013.5

    Source: Vallado (2013), *Fundamentals of Astrodynamics and Applications*,
    4th edition, algorithm 14 (equation 3-27).

    **Validity range, measured rather than quoted.** The stated range is
    1900 March 1 to 2100 February 28, and the boundaries were checked against
    an independent computation from the proleptic Gregorian ordinal
    (``date.toordinal() + 1721424.5``) rather than taken on trust. Equation (5)
    is one day low for 1900 January and February and one day high from 2100
    March onward; it agrees exactly, to the last bit, on every month start from
    1900 March to 2056 December. The two-digit epoch year window of 1957 to
    2056 lies entirely inside the agreeing range, so no element set this
    package can decode falls outside it; the out-of-range disagreement is
    asserted as a test rather than left as a claim.

(6) Epoch Julian date from the element set fields:

        JD_epoch = JD(year, 1, 1) + (epoch_day - 1)

    with ``epoch_day`` the fractional day of year, 1.0 being 0h UT on
    1 January. Verified against the ``sgp4`` library's own epoch decoding for
    every element set in the validation corpus; the agreement is reported in
    ``validation/validate_bounds_output.txt``.

Units
-----
Julian dates are days. Ages are days. ``requirement_days`` is days.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .elements import ElementSet

SECONDS_PER_DAY = 86400.0


def julian_date(year: int, month: int, day: float) -> float:
    """Equation (5): Julian date at the given Gregorian date, days.

    Valid from 1900 March 1 to 2100 February 28, which contains the whole
    1957-2056 window a two-digit element set epoch year can express. Outside it
    the result is one day out; see the module docstring for the measurement.
    """
    return (
        367.0 * year
        - math.floor(7.0 * (year + math.floor((month + 9.0) / 12.0)) / 4.0)
        + math.floor(275.0 * month / 9.0)
        + day
        + 1721013.5
    )


def epoch_julian_date(element_set: ElementSet) -> float:
    """Equation (6): Julian date of the element set epoch, days.

    Returns ``nan`` when the epoch fields did not parse.
    """
    if element_set.epoch_year < 0 or not math.isfinite(element_set.epoch_day):
        return float("nan")
    return julian_date(element_set.epoch_full_year, 1, 1.0) + (element_set.epoch_day - 1.0)


@dataclass(frozen=True)
class EpochVerdict:
    """Epoch age of one element set against a declared requirement.

    Attributes
    ----------
    epoch_jd:
        Julian date of the element set epoch, days. ``nan`` if unparsable.
    reference_jd:
        Julian date the age is measured against, days.
    age_days:
        ``reference_jd - epoch_jd``. Negative when the epoch is in the future
        of the reference, which is a distinct fault and is flagged separately.
    requirement_days:
        The declared maximum acceptable age, days. Reported with the verdict
        so a reader cannot see the verdict without the requirement.
    stale:
        ``age_days > requirement_days``.
    future_epoch:
        ``age_days < -future_tolerance_days``. A catalogue can legitimately
        publish an element set a fraction of a day ahead of the wall clock,
        which is why the tolerance exists and is declared.
    """

    epoch_jd: float
    reference_jd: float
    age_days: float
    requirement_days: float
    stale: bool
    future_epoch: bool
    future_tolerance_days: float


def epoch_age(
    element_set: ElementSet,
    reference_jd: float,
    requirement_days: float = 7.0,
    future_tolerance_days: float = 0.5,
) -> EpochVerdict:
    """Age of ``element_set`` at ``reference_jd`` against a declared requirement.

    Parameters
    ----------
    element_set:
        Parsed element set.
    reference_jd:
        Julian date the element set is to be used at, days.
    requirement_days:
        Declared maximum acceptable age in days. The default of 7.0 is a
        policy choice and is labelled as such in every report. There is no
        physically correct value: the requirement follows from what the
        ephemeris is for.
    future_tolerance_days:
        Epochs up to this far ahead of ``reference_jd`` are not flagged.

    Raises
    ------
    ValueError
        If ``requirement_days`` is not a positive finite number, or
        ``future_tolerance_days`` is negative, or ``reference_jd`` is not
        finite.
    """
    if not math.isfinite(requirement_days) or requirement_days <= 0.0:
        raise ValueError(
            f"requirement_days must be a positive finite number, got {requirement_days!r}"
        )
    if not math.isfinite(future_tolerance_days) or future_tolerance_days < 0.0:
        raise ValueError(
            "future_tolerance_days must be a non-negative finite number, "
            f"got {future_tolerance_days!r}"
        )
    if not math.isfinite(reference_jd):
        raise ValueError(f"reference_jd must be finite, got {reference_jd!r}")
    epoch_jd = epoch_julian_date(element_set)
    age = reference_jd - epoch_jd
    if not math.isfinite(age):
        return EpochVerdict(
            epoch_jd=epoch_jd,
            reference_jd=reference_jd,
            age_days=float("nan"),
            requirement_days=requirement_days,
            stale=False,
            future_epoch=False,
            future_tolerance_days=future_tolerance_days,
        )
    return EpochVerdict(
        epoch_jd=epoch_jd,
        reference_jd=reference_jd,
        age_days=age,
        requirement_days=requirement_days,
        stale=age > requirement_days,
        future_epoch=age < -future_tolerance_days,
        future_tolerance_days=future_tolerance_days,
    )
