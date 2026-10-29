"""Measured consequence: the kilometres the class labels rest on."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tlecheck.consequence import (
    BENIGN,
    CLASSES,
    DEFAULT_SILENT_THRESHOLD_KM,
    SILENTLY_WRONG,
    WILL_FAIL_LOUDLY,
    measure,
    position_errors_km,
)
from tlecheck.elements import apply_checksum, format_tle, parse_tle
from tlecheck.epoch import epoch_julian_date
from tlecheck.propagate import propagate, time_grid

from conftest import REFERENCE_FIELDS, REFERENCE_LINE2


def _set(**overrides):
    fields = dict(REFERENCE_FIELDS)
    fields.update(overrides)
    return parse_tle(*format_tle(**fields))


def test_classes_are_the_three_declared_ones():
    assert set(CLASSES) == {WILL_FAIL_LOUDLY, SILENTLY_WRONG, BENIGN}


def test_default_threshold_is_the_documented_one():
    assert DEFAULT_SILENT_THRESHOLD_KM == 1.0


def test_identical_element_sets_are_benign_with_zero_error():
    reference = _set()
    start = epoch_julian_date(reference)
    result = measure(reference, reference, start, 1.0, 9)
    assert result.consequence_class == BENIGN
    assert result.max_error_km == pytest.approx(0.0, rel=0.0, abs=1.0e-9)


def test_bstar_sign_flip_is_silently_wrong_over_a_week():
    """The textbook silent corruption, measured rather than assumed.

    Flipping the sign of the drag term leaves the element set inside every
    physical bound and inside the text format. The propagator reports success
    at every sample. Measured in this environment: 93.06 km of maximum
    position error over seven days for the published reference element set.
    """
    reference = _set()
    flipped = _set(bstar=-REFERENCE_FIELDS["bstar"])
    start = epoch_julian_date(reference)
    result = measure(flipped, reference, start, 7.0, 49)
    assert result.consequence_class == SILENTLY_WRONG
    assert result.loud_reason == ""
    assert result.max_error_km == pytest.approx(93.06, rel=0.0, abs=0.5)


def test_bstar_sign_flip_error_grows_with_the_horizon():
    reference = _set()
    flipped = _set(bstar=-REFERENCE_FIELDS["bstar"])
    start = epoch_julian_date(reference)
    one_day = measure(flipped, reference, start, 1.0, 25).max_error_km
    seven_days = measure(flipped, reference, start, 7.0, 49).max_error_km
    assert seven_days > one_day


def test_checksum_only_corruption_has_exactly_zero_consequence():
    """The screen fires and the propagated error is zero. Both are true."""
    reference = _set()
    broken_line1 = reference.line1[:68] + str((int(reference.line1[68]) + 4) % 10)
    candidate = parse_tle(broken_line1, reference.line2)
    start = epoch_julian_date(reference)
    result = measure(candidate, reference, start, 1.0, 9)
    assert result.consequence_class == BENIGN
    assert result.max_error_km == pytest.approx(0.0, rel=0.0, abs=0.0)
    assert candidate.checksum_line1_stated != candidate.checksum_line1_computed


def test_implausible_eccentricity_is_loud():
    reference = _set()
    line2 = apply_checksum(REFERENCE_LINE2[:26] + "9999999" + REFERENCE_LINE2[33:68])
    candidate = parse_tle(reference.line1, line2)
    result = measure(candidate, reference, epoch_julian_date(reference), 1.0, 9)
    assert result.consequence_class == WILL_FAIL_LOUDLY
    assert "SGP4 error code" in result.loud_reason
    assert math.isnan(result.max_error_km)


def test_loud_reference_is_reported_as_such_and_not_blamed_on_the_candidate():
    reference = parse_tle(" " * 69, " " * 69)
    candidate = _set()
    result = measure(candidate, reference, 2460000.0, 1.0, 5)
    assert result.consequence_class == WILL_FAIL_LOUDLY
    assert result.reference_loud is True
    assert "REFERENCE element set is itself loud" in result.loud_reason


def test_threshold_moves_the_class_boundary():
    reference = _set()
    flipped = _set(bstar=-REFERENCE_FIELDS["bstar"])
    start = epoch_julian_date(reference)
    tight = measure(flipped, reference, start, 1.0, 25, silent_threshold_km=1.0)
    loose = measure(flipped, reference, start, 1.0, 25, silent_threshold_km=1000.0)
    assert tight.consequence_class == SILENTLY_WRONG
    assert loose.consequence_class == BENIGN
    assert tight.max_error_km == pytest.approx(loose.max_error_km, rel=0.0, abs=0.0)


def test_threshold_is_carried_with_the_class():
    reference = _set()
    result = measure(reference, reference, epoch_julian_date(reference), 1.0, 5, 3.5)
    assert result.threshold_km == 3.5


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_measure_rejects_bad_threshold(bad):
    reference = _set()
    with pytest.raises(ValueError, match="silent_threshold_km"):
        measure(reference, reference, epoch_julian_date(reference), 1.0, 5, bad)


def test_position_errors_rejects_mismatched_grid_lengths():
    element_set = _set()
    start = epoch_julian_date(element_set)
    first = propagate(element_set, time_grid(start, 1.0, 5))
    second = propagate(element_set, time_grid(start, 1.0, 7))
    with pytest.raises(ValueError, match="different grid lengths"):
        position_errors_km(first, second)


def test_position_errors_rejects_different_grids():
    element_set = _set()
    start = epoch_julian_date(element_set)
    first = propagate(element_set, time_grid(start, 1.0, 5))
    second = propagate(element_set, time_grid(start + 0.5, 1.0, 5))
    with pytest.raises(ValueError, match="different time grids"):
        position_errors_km(first, second)


def test_position_errors_is_a_euclidean_norm():
    element_set = _set()
    start = epoch_julian_date(element_set)
    first = propagate(element_set, time_grid(start, 1.0, 5))
    errors = position_errors_km(first, first)
    assert np.allclose(errors, 0.0, rtol=0.0, atol=1.0e-9)


def test_rms_is_between_zero_and_the_maximum():
    reference = _set()
    flipped = _set(bstar=-REFERENCE_FIELDS["bstar"])
    result = measure(flipped, reference, epoch_julian_date(reference), 7.0, 49)
    assert 0.0 <= result.rms_error_km <= result.max_error_km


def test_grid_description_is_carried():
    reference = _set()
    result = measure(reference, reference, epoch_julian_date(reference), 2.5, 13)
    assert result.samples == 13
    assert result.horizon_days == 2.5
