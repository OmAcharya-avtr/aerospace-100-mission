"""Coverage: the Monte Carlo curve and the spiral geometric closed form."""

from __future__ import annotations

import math

import numpy as np
import pytest

from acqsearch import (
    PointingUncertainty,
    cell_visit_order,
    coverage_curve,
    disc_coverage,
    spiral_closed_form_coverage,
    spiral_closed_form_valid,
    spiral_path,
    standard_paths,
)


def test_disc_coverage_is_the_rayleigh_cdf():
    u = PointingUncertainty(1.0e-4)
    assert disc_coverage(1.0e-4, u) == pytest.approx(1.0 - math.exp(-0.5), abs=1e-15)


def test_closed_form_tends_to_the_single_disc_as_the_pitch_vanishes():
    # With n = 1 and p -> 0 the covered region is the disc of radius a, so the
    # closed form must tend to the Rayleigh CDF at a.
    u = PointingUncertainty(1.5e-4)
    a = 5.0e-5
    value = spiral_closed_form_coverage(1.0e-9, 1, a, u)
    # The expression is a difference of two nearly equal error functions
    # divided by the pitch, so it loses significance as the pitch goes to
    # zero: at p = 1e-9 rad the agreement with the exact limit is 1.9e-5
    # relative, not machine precision. That is a property of the formula as
    # written and the tolerance states it rather than hiding it.
    assert value == pytest.approx(u.radial_cdf(a), rel=1e-4)


def test_closed_form_is_between_its_two_bounding_discs():
    u = PointingUncertainty(1.5e-4)
    p, a, n = 7.0e-5, 5.0e-5, 3
    value = spiral_closed_form_coverage(p, n, a, u)
    assert u.radial_cdf((n - 1) * p + a) <= value <= u.radial_cdf(n * p + a)


def test_closed_form_increases_with_turns():
    u = PointingUncertainty(1.5e-4)
    values = [spiral_closed_form_coverage(7.0e-5, n, 5.0e-5, u) for n in range(1, 8)]
    assert all(b > a for a, b in zip(values, values[1:], strict=False))


@pytest.mark.parametrize("turns", [1, 2, 3, 4, 5])
def test_closed_form_matches_monte_carlo_within_three_standard_errors(turns):
    u = PointingUncertainty(1.5e-4)
    pitch, spacing, a = 7.0710678118654756e-5, 3.5355339059327378e-5, 5.0e-5
    assert spiral_closed_form_valid(pitch, spacing, a)
    closed = spiral_closed_form_coverage(pitch, turns, a, u)
    curve = coverage_curve(spiral_path(pitch, float(turns), spacing), u, a, 60000, seed=17)
    measured = curve.final_coverage()
    se = curve.standard_error[-1]
    # Both the Monte Carlo error and the along-track discretisation of the
    # spiral contribute, so the bound is three standard errors plus 2e-3 of
    # geometric slack. Measured worst case over these five cases: see
    # validation/validate_spiral_coverage_output.txt.
    assert abs(measured - closed) <= 3.0 * se + 2.0e-3


def test_validity_predicate_rejects_a_pitch_that_leaves_gaps():
    assert not spiral_closed_form_valid(1.2e-4, 3.0e-5, 5.0e-5)
    assert spiral_closed_form_valid(7.0e-5, 3.0e-5, 5.0e-5)


def test_validity_predicate_rejects_spacing_wider_than_the_footprint():
    assert not spiral_closed_form_valid(1.0e-5, 1.5e-4, 5.0e-5)


def test_validity_predicate_rejects_non_positive_inputs():
    with pytest.raises(ValueError, match="must all be"):
        spiral_closed_form_valid(0.0, 1.0e-5, 5.0e-5)


def test_coverage_curve_is_monotone_and_bounded(grid, uncertainty, geometry):
    path = standard_paths(grid)["raster"]
    curve = coverage_curve(path, uncertainty, geometry.footprint_radius_rad, 20000, seed=5)
    assert curve.dwell_index.size == path.n_points
    assert np.all(np.diff(curve.covered_mass) >= -1e-15)
    assert 0.0 <= curve.covered_mass[0] <= curve.covered_mass[-1] <= 1.0


def test_coverage_curve_final_value_is_near_the_grid_prior_mass(grid, uncertainty, geometry):
    # The raster covers the same square region the cell grid partitions, with
    # discs rather than squares, so it covers at least the grid's prior mass.
    curve = coverage_curve(
        standard_paths(grid)["raster"], uncertainty, geometry.footprint_radius_rad, 40000, seed=9
    )
    assert curve.final_coverage() >= grid.prior_mass.sum() - 5.0e-3


def test_coverage_curve_is_seeded(grid, uncertainty, geometry):
    a = geometry.footprint_radius_rad
    path = standard_paths(grid)["spiral"]
    first = coverage_curve(path, uncertainty, a, 5000, seed=3).covered_mass
    second = coverage_curve(path, uncertainty, a, 5000, seed=3).covered_mass
    assert np.array_equal(first, second)


def test_coverage_of_the_spiral_exceeds_the_random_restart_early(grid, uncertainty, geometry):
    a = geometry.footprint_radius_rad
    paths = standard_paths(grid)
    spiral = coverage_curve(paths["spiral"], uncertainty, a, 20000, seed=13)
    random = coverage_curve(paths["random_restart"], uncertainty, a, 20000, seed=13)
    # After ten dwells the spiral has swept the centre of the prior.
    assert spiral.covered_mass[9] > random.covered_mass[9]


def test_coverage_curve_rejects_bad_arguments(grid, uncertainty):
    path = standard_paths(grid)["spiral"]
    with pytest.raises(ValueError, match="footprint_radius_rad"):
        coverage_curve(path, uncertainty, 0.0, 100)
    with pytest.raises(ValueError, match="n_samples"):
        coverage_curve(path, uncertainty, 5.0e-5, 0)


def test_closed_form_rejects_non_integer_turns():
    u = PointingUncertainty(1.0e-4)
    with pytest.raises(ValueError, match="turns"):
        spiral_closed_form_coverage(1.0e-4, 0, 5.0e-5, u)


def test_closed_form_rejects_non_positive_geometry():
    u = PointingUncertainty(1.0e-4)
    with pytest.raises(ValueError, match="must be > 0"):
        spiral_closed_form_coverage(0.0, 2, 5.0e-5, u)


def test_cell_order_and_coverage_agree_on_which_pattern_reaches_mass_first(grid, uncertainty):
    # The spiral order visits high-prior cells first, so its cumulative cell
    # mass dominates the raster order over the first quarter of the scan.
    spiral = cell_visit_order(standard_paths(grid)["spiral"], grid)
    raster = cell_visit_order(standard_paths(grid)["raster"], grid)
    quarter = grid.n_cells // 4
    assert grid.prior_mass[spiral[:quarter]].sum() > grid.prior_mass[raster[:quarter]].sum()
