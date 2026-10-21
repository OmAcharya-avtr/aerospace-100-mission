"""The dilution region, its closed-form maximum, and the warning the library raises."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from conjprob.analytic import pc_foster
from conjprob.dilution import (
    DILUTION_MAHALANOBIS_THRESHOLD,
    DilutionWarning,
    assess_dilution,
    dilution_curve,
    max_probability,
)


def test_threshold_is_root_two():
    assert DILUTION_MAHALANOBIS_THRESHOLD == pytest.approx(math.sqrt(2.0), rel=1e-15)


def test_max_probability_reproduces_the_circular_closed_form():
    # For a circular covariance the maximum over covariance scale is
    # R^2 / (e d^2) (Alfano 2005b). R = 12 m, d = 120 m gives
    # 144 / (e * 14400) = 0.00367879441171442.
    pc_max, scale = max_probability(300.0, 300.0, 120.0, 0.0, 12.0)
    assert pc_max == pytest.approx(12.0**2 / (math.e * 120.0**2), rel=1e-13)
    # The scale attaining it is M / sqrt(2) with M = 120 / 300 = 0.4.
    assert scale == pytest.approx(0.4 / math.sqrt(2.0), rel=1e-13)


def test_max_probability_is_attained_by_the_numerical_maximum():
    assessment = assess_dilution(300.0, 300.0, 120.0, 0.0, 12.0, warn=False)
    assert assessment.pc_max_numerical == pytest.approx(
        assessment.pc_max_closed_form, rel=1e-4
    )
    assert assessment.scale_at_max_numerical == pytest.approx(
        assessment.scale_at_max_closed_form, rel=1e-2
    )


def test_max_probability_bounds_the_pc_at_every_scale():
    scales = np.logspace(-2.0, 2.0, 41)
    exact, _ = dilution_curve(300.0, 300.0, 120.0, 0.0, 12.0, scales)
    pc_max, _ = max_probability(300.0, 300.0, 120.0, 0.0, 12.0)
    # The closed form is the small-hard-body maximum, so the exact curve can
    # exceed it slightly where the hard body is not small. The measured excess
    # for this geometry is below one part in a thousand.
    assert exact.max() <= pc_max * (1.0 + 1e-3)


def test_max_probability_rejects_a_zero_miss_vector():
    with pytest.raises(ValueError, match="zero or underflowing miss vector"):
        max_probability(100.0, 100.0, 0.0, 0.0, 10.0)


def test_dilution_curve_is_non_monotonic_with_a_single_interior_maximum():
    # The lower end of the sweep is chosen so that Pc has not underflowed to
    # zero: at scale 0.01 the Mahalanobis distance is 40 and Pc is below the
    # smallest normal double, which would make the strict monotonicity test
    # compare zeros.
    scales = np.logspace(-1.0, 3.0, 121)
    exact, _ = dilution_curve(300.0, 300.0, 120.0, 0.0, 12.0, scales)
    peak = int(np.argmax(exact))
    assert 0 < peak < exact.size - 1
    assert np.all(np.diff(exact[: peak + 1]) > 0.0)
    assert np.all(np.diff(exact[peak:]) < 0.0)


def test_dilution_curve_small_body_tracks_the_exact_curve_where_the_body_is_small():
    scales = np.logspace(0.0, 2.0, 21)
    exact, approximate = dilution_curve(300.0, 300.0, 120.0, 0.0, 12.0, scales)
    np.testing.assert_allclose(approximate, exact, rtol=2e-3)


def test_dilution_curve_rejects_bad_scales():
    with pytest.raises(ValueError, match="non-empty"):
        dilution_curve(100.0, 100.0, 10.0, 0.0, 1.0, np.array([]))
    with pytest.raises(ValueError, match="strictly positive"):
        dilution_curve(100.0, 100.0, 10.0, 0.0, 1.0, np.array([1.0, -1.0]))


def test_warning_is_raised_well_inside_the_dilution_branch():
    with pytest.warns(DilutionWarning, match="dilution branch"):
        assess_dilution(300.0, 300.0, 60.0, 0.0, 12.0)


def test_no_warning_well_outside_the_dilution_branch():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DilutionWarning)
        assessment = assess_dilution(100.0, 100.0, 600.0, 0.0, 12.0)
    assert assessment.mahalanobis > DILUTION_MAHALANOBIS_THRESHOLD


def test_branch_flag_and_sensitivity_sign_agree_away_from_the_turning_point():
    # The flag comes from the sign of a finite-difference derivative and the
    # ratio comes from a closed form. They are two different computations of
    # the same fact, so they are only required to agree away from the turning
    # point, and near_turning_point reports when that is not the case.
    for xbar in (30.0, 60.0, 600.0, 1200.0):
        assessment = assess_dilution(300.0, 300.0, xbar, 0.0, 12.0, warn=False)
        if assessment.near_turning_point:
            continue
        assert assessment.on_dilution_branch == (assessment.dilution_ratio > 1.0)


def test_turning_point_case_is_flagged_as_near_and_not_asserted():
    # Place the Mahalanobis distance at sqrt(2), which is the turning point of
    # the small-hard-body approximation. The exact Foster-Estes maximum is at a
    # slightly different scale -- measured here at 0.99980 rather than 1 -- so
    # the derivative is small but not zero, and the boolean is not a promise.
    # The measured sensitivity is 8.0e-4 of Pc, so the test allows 1e-2 of Pc.
    sigma = 300.0
    assessment = assess_dilution(
        sigma, sigma, sigma * math.sqrt(2.0), 0.0, 12.0, warn=False
    )
    assert assessment.near_turning_point
    assert abs(assessment.log_scale_sensitivity) < 1e-2 * assessment.pc_at_unit_scale
    assert assessment.scale_at_max_numerical == pytest.approx(1.0, rel=1e-3)


def test_dilution_factor_matches_the_ratio_of_maximum_to_present():
    assessment = assess_dilution(300.0, 300.0, 120.0, 0.0, 12.0, warn=False)
    assert assessment.dilution_factor == pytest.approx(
        assessment.pc_max_numerical / assessment.pc_at_unit_scale, rel=1e-12
    )
    assert assessment.pc_at_unit_scale == pytest.approx(
        pc_foster(300.0, 300.0, 120.0, 0.0, 12.0), rel=1e-12
    )


def test_assess_dilution_rejects_a_non_positive_step():
    with pytest.raises(ValueError, match="relative_step"):
        assess_dilution(100.0, 100.0, 10.0, 0.0, 1.0, relative_step=0.0)
