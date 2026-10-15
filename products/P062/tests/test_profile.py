"""Profile-likelihood tests."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import chi2 as chi2_dist

from twincalib.fitting import fit
from twincalib.profile import chi2_threshold, profile_all, profile_likelihood


def test_threshold_matches_the_chi_squared_quantile():
    # Wilks 1938: one degree of freedom. chi2.ppf(0.95, 1) = 3.841458821.
    assert chi2_threshold(0.95) == pytest.approx(3.841458820694124, rel=1e-12)
    assert chi2_threshold(0.90) == pytest.approx(2.705543454095404, rel=1e-12)
    for level in (0.5, 0.68, 0.99):
        assert chi2_threshold(level) == pytest.approx(
            float(chi2_dist.ppf(level, 1)), rel=1e-12
        )


def test_threshold_validation():
    for bad in (0.0, 1.0, -0.1, np.nan):
        with pytest.raises(ValueError, match="level"):
            chi2_threshold(bad)


def test_profile_centre_point_has_zero_delta(exact_baseline_fit):
    problem, result, _d = exact_baseline_fit
    curve = profile_likelihood(problem, result, 0, n_points=3)
    centre = [p for p in curve.points if p.log_offset == 0.0]
    assert len(centre) == 1
    assert centre[0].delta_chi2 == pytest.approx(0.0, abs=1e-9)


def test_profile_is_non_negative_and_rises_away_from_the_optimum(exact_baseline_fit):
    problem, result, _d = exact_baseline_fit
    curve = profile_likelihood(problem, result, 2, n_points=4, refine_steps=0)
    deltas = np.array([p.delta_chi2 for p in curve.points])
    # The nested optimisations are numerical, so the profile can dip below zero
    # by a rounding amount; it must not dip meaningfully.
    assert float(np.min(deltas)) > -1e-6
    assert deltas[0] > 0.0
    assert deltas[-1] > 0.0


def test_profile_agrees_with_the_quadratic_interval_on_a_good_design(
    exact_baseline_fit,
):
    # Where the likelihood is near quadratic the two intervals must agree. The
    # measured ratios over all six parameters of this design are 1.000 to 1.006
    # (validation/validate_profile_output.txt); the test allows 0.85 to 1.20 so
    # that it is a check on the method and not on one seed.
    problem, result, _d = exact_baseline_fit
    curves = profile_all(problem, result, n_points=4, refine_steps=3)
    for c in curves:
        assert c.bounded
        assert 0.85 < c.width_ratio < 1.20


def test_profile_reports_the_structural_direction_as_open(exact_full_sun):
    # The required behaviour on the deliberately non-identifiable design. Only
    # the SUM q_sun + q_par is identifiable, so each load on its own is
    # unbounded below -- it can go to zero with the other taking up the slack --
    # and bounded above only by the positivity of the other one, at the total.
    # The upper limit is set by how far the partner load can be squeezed, and
    # the partner is itself confined to the nuisance box, so the limit lands at
    # or a little below the identified total: the measured values are 73.17 W
    # for q_sun and 68.14 W for q_par against a total of 73.32 W
    # (validation/validate_profile_output.txt). The assertion below is on that
    # structure, not on the exact fraction.
    d, _data, problem = exact_full_sun
    result = fit(problem, np.asarray(d.theta_true, dtype=float))
    total = float(result.theta[4] + result.theta[5])
    for index, name in ((4, "q_sun"), (5, "q_par")):
        c = profile_likelihood(problem, result, index, n_points=2, refine_steps=8)
        assert c.name == name
        assert c.lower is None, f"{name} must be unbounded below"
        assert c.upper is not None
        assert 0.85 * total < c.upper <= 1.02 * total
        assert not c.bounded
        # Neither interval is available for these two -- the profile side is
        # open and the quadratic one was withdrawn as infinite -- so the ratio
        # is not a number rather than infinite.
        assert not np.isfinite(c.width_ratio)
    for index in (0, 1, 2, 3):
        c = profile_likelihood(problem, result, index, n_points=3, refine_steps=2)
        assert c.bounded
    assert d.name == "full-sun"


def test_open_profile_reports_the_range_it_searched(exact_full_sun):
    # A parameter whose quadratic standard deviation is infinite gets the capped
    # search width, so the range covered is e**3 = 20.1 either side.
    d, _data, problem = exact_full_sun
    result = fit(problem, np.asarray(d.theta_true, dtype=float))
    curve = profile_likelihood(problem, result, 4, n_points=3, refine_steps=2)
    assert not curve.bounded
    assert curve.search_lower < curve.estimate < curve.search_upper
    assert np.log(curve.search_upper / curve.search_lower) == pytest.approx(
        6.0, rel=1e-9
    )


def test_interval_contains_the_estimate(exact_baseline_fit):
    problem, result, _d = exact_baseline_fit
    for c in profile_all(problem, result, n_points=3, refine_steps=3):
        assert c.lower is not None and c.upper is not None
        assert c.lower < c.estimate < c.upper
        assert c.contains(c.estimate)
        assert not c.contains(c.lower * 0.5)
        assert not c.contains(c.upper * 2.0)


def test_log_widths_are_consistent(exact_baseline_fit):
    problem, result, _d = exact_baseline_fit
    c = profile_likelihood(problem, result, 1, n_points=3, refine_steps=3)
    assert c.log_width == pytest.approx(
        float(np.log(c.upper / c.lower)), rel=1e-12
    )
    assert c.hessian_log_width == pytest.approx(
        float(np.log(c.hessian_upper / c.hessian_lower)), rel=1e-12
    )
    assert c.width_ratio == pytest.approx(
        c.log_width / c.hessian_log_width, rel=1e-12
    )


def test_tighter_level_gives_a_narrower_interval(exact_baseline_fit):
    problem, result, _d = exact_baseline_fit
    narrow = profile_likelihood(problem, result, 0, level=0.68, n_points=3)
    wide = profile_likelihood(problem, result, 0, level=0.99, n_points=3)
    assert narrow.log_width < wide.log_width
    assert narrow.threshold < wide.threshold


def test_curve_arrays_shapes(exact_baseline_fit):
    problem, result, _d = exact_baseline_fit
    c = profile_likelihood(problem, result, 3, n_points=3, refine_steps=2)
    values, deltas = c.curve_arrays()
    assert values.shape == deltas.shape
    assert values.size == len(c.points)
    assert np.all(np.diff(values) > 0.0)


def test_profile_validation(exact_baseline_fit):
    problem, result, _d = exact_baseline_fit
    with pytest.raises(ValueError, match="index must lie"):
        profile_likelihood(problem, result, 6)
    with pytest.raises(ValueError, match="n_points"):
        profile_likelihood(problem, result, 0, n_points=1)
    with pytest.raises(ValueError, match="span_sigma"):
        profile_likelihood(problem, result, 0, span_sigma=0.0)
    with pytest.raises(ValueError, match="min_log_half_width must be below"):
        profile_likelihood(
            problem, result, 0, min_log_half_width=4.0, max_log_half_width=3.0
        )


def test_profile_all_honours_an_index_subset(exact_baseline_fit):
    problem, result, _d = exact_baseline_fit
    curves = profile_all(problem, result, indices=[0, 5], n_points=3, refine_steps=1)
    assert [c.index for c in curves] == [0, 5]
    assert [c.name for c in curves] == ["C1", "q_par"]
