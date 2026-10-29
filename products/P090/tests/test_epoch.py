"""Epoch decoding and staleness, cross-checked against sgp4's own decoding."""

from __future__ import annotations

import math

import pytest
from sgp4.api import Satrec

from tlecheck.elements import format_tle, parse_tle
from tlecheck.epoch import epoch_age, epoch_julian_date, julian_date


def test_julian_date_j2000_known_answer():
    # J2000.0 is 2000 January 1.5, Julian date 2451545.0, so 0h UT on
    # 1 January 2000 is 2451544.5. Standard reference value.
    assert julian_date(2000, 1, 1.0) == pytest.approx(2451544.5, rel=0.0, abs=1.0e-9)


def test_julian_date_1957_known_answer():
    # 1957 January 1 0h UT is JD 2435839.5, the first year the two-digit epoch
    # window can express. Standard reference value.
    assert julian_date(1957, 1, 1.0) == pytest.approx(2435839.5, rel=0.0, abs=1.0e-9)


def test_julian_date_against_independent_derivation_over_the_whole_window():
    """Cross-check against mathematics equation (5) does not use.

    The proleptic Gregorian ordinal of :mod:`datetime` plus the fixed offset
    1721424.5 is a different derivation of the same quantity: a day count from
    a different origin by a different algorithm. They agree to the last bit on
    every month start of the 1957 to 2056 epoch-year window.
    """
    from datetime import date

    for year in range(1957, 2057):
        for month in range(1, 13):
            independent = date(year, month, 1).toordinal() + 1721424.5
            assert julian_date(year, month, 1.0) == pytest.approx(
                independent, rel=0.0, abs=0.0
            )


def test_julian_date_is_one_day_out_below_its_declared_range():
    """A FAILING check, asserted rather than hidden.

    Equation (5) is one day low for 1900 January and February, which is outside
    its declared validity range of 1900 March 1 onward. No element set this
    package can decode reaches there, because the two-digit year window starts
    at 1957, but the limit is real and is asserted so that it cannot be
    forgotten.
    """
    from datetime import date

    independent = date(1900, 1, 1).toordinal() + 1721424.5
    assert julian_date(1900, 1, 1.0) == pytest.approx(
        independent - 1.0, rel=0.0, abs=0.0
    )


def test_julian_date_is_one_day_out_above_its_declared_range():
    """The other end of the same measured limit."""
    from datetime import date

    independent = date(2100, 3, 1).toordinal() + 1721424.5
    assert julian_date(2100, 3, 1.0) == pytest.approx(
        independent + 1.0, rel=0.0, abs=0.0
    )


def test_julian_date_advances_one_per_day():
    assert julian_date(2026, 3, 2.0) - julian_date(2026, 3, 1.0) == pytest.approx(
        1.0, rel=0.0, abs=1.0e-9
    )


def test_epoch_julian_date_matches_sgp4(reference_lines, reference_set):
    """Independent cross-check of equation (6) against sgp4's own decoding."""
    satrec = Satrec.twoline2rv(*reference_lines)
    theirs = satrec.jdsatepoch + satrec.jdsatepochF
    assert epoch_julian_date(reference_set) == pytest.approx(theirs, rel=0.0, abs=1.0e-9)


def test_epoch_julian_date_matches_sgp4_across_the_corpus(small_records):
    for record in small_records[:24]:
        for element_set in (*record.history, record.candidate):
            satrec = Satrec.twoline2rv(element_set.line1, element_set.line2)
            theirs = satrec.jdsatepoch + satrec.jdsatepochF
            mine = epoch_julian_date(element_set)
            if not math.isfinite(mine):
                continue
            assert mine == pytest.approx(theirs, rel=0.0, abs=1.0e-6)


def test_epoch_julian_date_nan_when_unparsable():
    element_set = parse_tle(" " * 69, " " * 69)
    assert math.isnan(epoch_julian_date(element_set))


def test_epoch_day_one_is_start_of_january():
    line1, line2 = format_tle(
        satnum=1,
        epoch_year=2026,
        epoch_day=1.0,
        inclination_deg=10.0,
        raan_deg=20.0,
        eccentricity=0.001,
        arg_perigee_deg=30.0,
        mean_anomaly_deg=40.0,
        mean_motion_rev_per_day=14.0,
    )
    assert epoch_julian_date(parse_tle(line1, line2)) == pytest.approx(
        julian_date(2026, 1, 1.0), rel=0.0, abs=1.0e-9
    )


def test_epoch_age_exact(reference_set):
    base = epoch_julian_date(reference_set)
    verdict = epoch_age(reference_set, base + 21.0, requirement_days=7.0)
    assert verdict.age_days == pytest.approx(21.0, rel=0.0, abs=1.0e-9)
    assert verdict.stale is True
    assert verdict.future_epoch is False
    assert verdict.requirement_days == 7.0


def test_epoch_age_within_requirement(reference_set):
    base = epoch_julian_date(reference_set)
    verdict = epoch_age(reference_set, base + 3.0, requirement_days=7.0)
    assert verdict.stale is False


def test_epoch_age_at_the_requirement_is_not_stale(reference_set):
    base = epoch_julian_date(reference_set)
    verdict = epoch_age(reference_set, base + 7.0, requirement_days=7.0)
    # The comparison is strictly greater than, and the age here is exactly the
    # requirement to within rounding; the test states what is guaranteed.
    assert verdict.age_days == pytest.approx(7.0, rel=0.0, abs=1.0e-9)
    assert verdict.stale is False


def test_future_epoch_detected(reference_set):
    base = epoch_julian_date(reference_set)
    verdict = epoch_age(reference_set, base - 2.0, requirement_days=7.0)
    assert verdict.future_epoch is True
    assert verdict.stale is False


def test_future_epoch_tolerance(reference_set):
    base = epoch_julian_date(reference_set)
    verdict = epoch_age(
        reference_set, base - 0.25, requirement_days=7.0, future_tolerance_days=0.5
    )
    assert verdict.future_epoch is False


def test_epoch_age_unparsable_is_nan():
    element_set = parse_tle(" " * 69, " " * 69)
    verdict = epoch_age(element_set, 2460000.0, requirement_days=7.0)
    assert math.isnan(verdict.age_days)
    assert verdict.stale is False


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_epoch_age_rejects_bad_requirement(reference_set, bad):
    with pytest.raises(ValueError, match="requirement_days"):
        epoch_age(reference_set, 2460000.0, requirement_days=bad)


@pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf")])
def test_epoch_age_rejects_bad_future_tolerance(reference_set, bad):
    with pytest.raises(ValueError, match="future_tolerance_days"):
        epoch_age(reference_set, 2460000.0, future_tolerance_days=bad)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_epoch_age_rejects_bad_reference(reference_set, bad):
    with pytest.raises(ValueError, match="reference_jd"):
        epoch_age(reference_set, bad)


def test_requirement_is_carried_with_the_verdict(reference_set):
    verdict = epoch_age(reference_set, epoch_julian_date(reference_set) + 1.0, 3.5)
    assert verdict.requirement_days == 3.5
