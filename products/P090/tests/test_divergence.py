"""Cross-source divergence, and the thing it is structurally unable to say."""

from __future__ import annotations

import math

import pytest

from tlecheck.divergence import DEFAULT_DIVERGENCE_THRESHOLD_KM, compare
from tlecheck.elements import format_tle, parse_tle
from tlecheck.epoch import epoch_julian_date

from conftest import REFERENCE_FIELDS, record_of


def _set(source: str = "a", **overrides):
    fields = dict(REFERENCE_FIELDS)
    fields.update(overrides)
    return parse_tle(*format_tle(**fields), source=source)


def test_default_threshold_is_the_documented_one():
    assert DEFAULT_DIVERGENCE_THRESHOLD_KM == 1.0


def test_identical_sources_do_not_diverge():
    first = _set("source_a")
    second = _set("source_b")
    result = compare(first, second, epoch_julian_date(first), 1.0, 9)
    assert result.usable is True
    assert result.diverged is False
    assert result.max_separation_km == pytest.approx(0.0, rel=0.0, abs=1.0e-9)


def test_perturbed_source_diverges():
    first = _set("source_a")
    second = _set("source_b", mean_motion_rev_per_day=15.50113472)
    result = compare(first, second, epoch_julian_date(first), 1.0, 9)
    assert result.diverged is True
    assert result.max_separation_km > 1.0


def test_which_source_is_correct_is_always_none():
    first = _set("source_a")
    second = _set("source_b", mean_motion_rev_per_day=15.50113472)
    result = compare(first, second, epoch_julian_date(first), 1.0, 9)
    assert result.which_source_is_correct is None


def test_comparison_is_symmetric_in_magnitude():
    """Equation (12) is symmetric, which is why the attribution is impossible."""
    first = _set("source_a")
    second = _set("source_b", mean_motion_rev_per_day=15.50113472)
    start = epoch_julian_date(first)
    forward = compare(first, second, start, 1.0, 9)
    backward = compare(second, first, start, 1.0, 9)
    assert forward.max_separation_km == pytest.approx(
        backward.max_separation_km, rel=0.0, abs=0.0
    )
    assert forward.diverged == backward.diverged
    assert forward.sources == tuple(reversed(backward.sources))


def test_sources_are_labelled():
    first = _set("space-track")
    second = _set("mission-od")
    result = compare(first, second, epoch_julian_date(first), 1.0, 5)
    assert result.sources == ("space-track", "mission-od")


def test_loud_source_makes_the_comparison_unusable():
    first = _set("source_a")
    broken = _set("source_b", eccentricity=0.9999999)
    result = compare(first, broken, epoch_julian_date(first), 1.0, 5)
    assert result.usable is False
    assert math.isnan(result.max_separation_km)
    assert "propagated loudly" in result.reason


def test_unparsable_lines_carry_object_number_minus_one():
    # This package reports an unreadable object number as -1. ``sgp4`` reads
    # the same blank columns as object 0 and propagates. The two disagree and
    # the disagreement is the point, so it is pinned here.
    blank = parse_tle(" " * 69, " " * 69, source="source_b")
    assert blank.satnum == -1
    from sgp4.api import Satrec

    assert Satrec.twoline2rv(" " * 69, " " * 69).satnum == 0


def test_different_objects_is_a_caller_error():
    first = _set("source_a")
    other = _set("source_b", satnum=11111)
    with pytest.raises(ValueError, match="needs one object"):
        compare(first, other, epoch_julian_date(first), 1.0, 5)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_rejects_bad_threshold(bad):
    first = _set("source_a")
    second = _set("source_b")
    with pytest.raises(ValueError, match="threshold_km"):
        compare(first, second, epoch_julian_date(first), 1.0, 5, bad)


def test_rms_is_bounded_by_the_maximum():
    first = _set("source_a")
    second = _set("source_b", mean_motion_rev_per_day=15.50113472)
    result = compare(first, second, epoch_julian_date(first), 1.0, 25)
    assert 0.0 <= result.rms_separation_km <= result.max_separation_km


def test_generated_divergence_record_is_usable(small_records):
    record = record_of(small_records, "cross_source_divergence")
    result = compare(
        record.candidate, record.other_sources[0], record.use_jd, 1.0, 9
    )
    assert result.usable is True
    assert result.which_source_is_correct is None
