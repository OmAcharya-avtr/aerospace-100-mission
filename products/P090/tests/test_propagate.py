"""The sgp4 wrapper, and the two behaviours of sgp4 that this package exists for."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tlecheck.elements import apply_checksum, parse_tle
from tlecheck.epoch import epoch_julian_date
from tlecheck.propagate import make_satrec, propagate, time_grid

from conftest import REFERENCE_LINE1, REFERENCE_LINE2


def test_time_grid_endpoints():
    grid = time_grid(2460000.0, 2.0, 5)
    assert grid[0] == pytest.approx(2460000.0, rel=0.0, abs=0.0)
    assert grid[-1] == pytest.approx(2460002.0, rel=0.0, abs=1.0e-12)
    assert grid.size == 5


@pytest.mark.parametrize("samples", [0, 1, -3])
def test_time_grid_rejects_too_few_samples(samples):
    with pytest.raises(ValueError, match="samples"):
        time_grid(2460000.0, 1.0, samples)


@pytest.mark.parametrize("horizon", [0.0, -1.0, float("nan"), float("inf")])
def test_time_grid_rejects_bad_horizon(horizon):
    with pytest.raises(ValueError, match="horizon_days"):
        time_grid(2460000.0, horizon, 5)


@pytest.mark.parametrize("start", [float("nan"), float("inf")])
def test_time_grid_rejects_bad_start(start):
    with pytest.raises(ValueError, match="start_jd"):
        time_grid(start, 1.0, 5)


def test_make_satrec_accepts_the_reference(reference_set):
    satrec, refusal = make_satrec(reference_set)
    assert satrec is not None
    assert refusal == ""


def test_propagate_reference_is_not_loud(reference_set):
    grid = time_grid(epoch_julian_date(reference_set), 1.0, 9)
    result = propagate(reference_set, grid)
    assert result.accepted is True
    assert result.loud is False
    assert np.all(result.error_codes == 0)
    assert np.isfinite(result.positions_km).all()


def test_propagate_position_magnitude_is_plausible(reference_set):
    grid = time_grid(epoch_julian_date(reference_set), 0.5, 5)
    result = propagate(reference_set, grid)
    radius = np.linalg.norm(result.positions_km, axis=1)
    # A 411 to 421 km altitude orbit has a geocentric radius near 6790 km.
    assert np.all(radius > 6700.0)
    assert np.all(radius < 6900.0)


def test_sgp4_does_not_verify_the_checksum():
    """Measured: a wrong checksum digit is parsed and propagated in silence.

    This is not a defect in ``sgp4`` and it is the first reason this package
    exists. The line below differs from the published one only in column 69.
    """
    broken = REFERENCE_LINE1[:68] + str((int(REFERENCE_LINE1[68]) + 3) % 10)
    element_set = parse_tle(broken, REFERENCE_LINE2)
    grid = time_grid(epoch_julian_date(element_set), 1.0, 5)
    result = propagate(element_set, grid)
    assert result.accepted is True
    assert result.loud is False


def test_sgp4_accepts_blank_lines_and_then_errors():
    """Measured: two blank lines parse without an exception and error at propagation."""
    element_set = parse_tle(" " * 69, " " * 69)
    satrec, refusal = make_satrec(element_set)
    assert satrec is not None and refusal == ""
    result = propagate(element_set, time_grid(2460000.0, 1.0, 5))
    assert result.loud is True
    assert "SGP4 error code 2" in result.loud_reason


def test_implausible_eccentricity_is_loud():
    line2 = apply_checksum(REFERENCE_LINE2[:26] + "9999999" + REFERENCE_LINE2[33:68])
    element_set = parse_tle(REFERENCE_LINE1, line2)
    result = propagate(element_set, time_grid(epoch_julian_date(element_set), 1.0, 5))
    assert result.loud is True
    assert "SGP4 error code" in result.loud_reason


def test_column_shift_changes_the_mean_motion_silently():
    """Measured: a one-column shift of the mean motion field reads 5.50 for 15.50."""
    field = REFERENCE_LINE2[52:63]
    shifted = field[1:] + field[:1]
    line2 = apply_checksum(REFERENCE_LINE2[:52] + shifted + REFERENCE_LINE2[63:68])
    element_set = parse_tle(REFERENCE_LINE1, line2)
    assert element_set.mean_motion_rev_per_day == pytest.approx(
        5.501034721, rel=1.0e-9, abs=0.0
    )
    assert element_set.checksum_line2_stated == element_set.checksum_line2_computed


def test_propagate_rejects_two_dimensional_times(reference_set):
    with pytest.raises(ValueError, match="one-dimensional"):
        propagate(reference_set, np.zeros((2, 2)) + 2460000.0)


def test_propagate_rejects_empty_times(reference_set):
    with pytest.raises(ValueError, match="at least one epoch"):
        propagate(reference_set, np.array([]))


def test_propagate_rejects_non_finite_times(reference_set):
    with pytest.raises(ValueError, match="finite"):
        propagate(reference_set, np.array([2460000.0, math.nan]))


def test_loud_reason_is_empty_when_quiet(reference_set):
    result = propagate(reference_set, time_grid(epoch_julian_date(reference_set), 1.0, 3))
    assert result.loud_reason == ""
    assert result.refusal == ""
