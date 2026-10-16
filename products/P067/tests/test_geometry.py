"""Geometry: Rayleigh prior, terminal geometry and the cell partition."""

from __future__ import annotations

import math

import numpy as np
import pytest

from acqsearch import PointingUncertainty, TerminalGeometry, build_cell_grid


def test_radial_cdf_at_one_sigma_matches_hand_calculation():
    # 1 - exp(-sigma^2 / (2 sigma^2)) = 1 - exp(-0.5) = 0.3934693402873666
    u = PointingUncertainty(1.0e-4)
    assert u.radial_cdf(1.0e-4) == pytest.approx(0.3934693402873666, abs=1e-15)


def test_radial_cdf_at_two_and_three_sigma():
    # 1 - exp(-2) = 0.8646647167633873 ; 1 - exp(-4.5) = 0.9888910034617577
    u = PointingUncertainty(2.0e-4)
    assert u.radial_cdf(4.0e-4) == pytest.approx(0.8646647167633873, abs=1e-15)
    assert u.radial_cdf(6.0e-4) == pytest.approx(0.9888910034617577, abs=1e-15)


def test_radial_cdf_is_zero_at_zero_and_tends_to_one():
    u = PointingUncertainty(1.0e-4)
    assert u.radial_cdf(0.0) == 0.0
    assert u.radial_cdf(1.0) == pytest.approx(1.0, abs=1e-15)


def test_rectangle_mass_of_the_whole_plane_is_one():
    u = PointingUncertainty(1.5e-4)
    assert u.rectangle_mass(-1.0, 1.0, -1.0, 1.0) == pytest.approx(1.0, abs=1e-12)


def test_rectangle_mass_matches_product_of_one_dimensional_masses():
    # Central +-1 sigma in each axis: (2 Phi(1) - 1)^2 = 0.6826894921370859^2
    u = PointingUncertainty(1.0e-4)
    expected = (2.0 * 0.8413447460685429 - 1.0) ** 2
    got = u.rectangle_mass(-1.0e-4, 1.0e-4, -1.0e-4, 1.0e-4)
    assert got == pytest.approx(expected, rel=1e-12)


def test_rectangle_mass_partitions_a_strip():
    u = PointingUncertainty(1.2e-4)
    whole = u.rectangle_mass(-3.0e-4, 3.0e-4, -3.0e-4, 3.0e-4)
    left = u.rectangle_mass(-3.0e-4, 0.0, -3.0e-4, 3.0e-4)
    right = u.rectangle_mass(0.0, 3.0e-4, -3.0e-4, 3.0e-4)
    assert left + right == pytest.approx(whole, rel=1e-12)


def test_sample_reproduces_the_declared_sigma():
    u = PointingUncertainty(1.0e-4)
    draws = u.sample(200000, np.random.default_rng(3))
    assert draws.shape == (200000, 2)
    # Standard error of the sample std is sigma / sqrt(2 n) = 1.58e-7.
    assert draws.std(axis=0) == pytest.approx(1.0e-4, rel=5e-3)
    assert draws.mean(axis=0) == pytest.approx(np.zeros(2), abs=2e-6)


def test_sample_is_seeded_and_reproducible():
    u = PointingUncertainty(1.0e-4)
    a = u.sample(100, np.random.default_rng(11))
    b = u.sample(100, np.random.default_rng(11))
    assert np.array_equal(a, b)


def test_footprint_radius_is_the_tighter_of_beam_and_field_of_view():
    wide_field = TerminalGeometry(1.0e-4, 4.0e-4, 1.0e-2, 0.0)
    assert wide_field.footprint_radius_rad == pytest.approx(5.0e-5)
    narrow_field = TerminalGeometry(4.0e-4, 1.0e-4, 1.0e-2, 0.0)
    assert narrow_field.footprint_radius_rad == pytest.approx(5.0e-5)


def test_slew_time_is_linear_in_distance_plus_settle():
    g = TerminalGeometry(1.0e-4, 2.0e-4, 1.0e-2, 2.0e-3)
    # 1e-4 rad / 1e-2 rad/s = 1e-2 s, plus 2e-3 s settle.
    assert g.slew_time_s(1.0e-4) == pytest.approx(1.2e-2, rel=1e-12)
    assert g.slew_time_s(0.0) == pytest.approx(2.0e-3, rel=1e-12)
    out = g.slew_time_s(np.array([0.0, 1.0e-4]))
    assert out == pytest.approx(np.array([2.0e-3, 1.2e-2]), rel=1e-12)


def test_cell_grid_side_is_the_square_inscribed_in_the_footprint(grid, geometry):
    assert grid.side_rad == pytest.approx(geometry.footprint_radius_rad * math.sqrt(2.0), rel=1e-12)
    half_diagonal = grid.side_rad * math.sqrt(2.0) / 2.0
    assert half_diagonal == pytest.approx(geometry.footprint_radius_rad, rel=1e-12)


def test_reference_grid_has_the_expected_shape(grid):
    assert grid.n_cells == 89
    assert grid.prior_mass.sum() == pytest.approx(0.9544979085580522, rel=1e-10)
    assert grid.mass_outside == pytest.approx(1.0 - grid.prior_mass.sum(), abs=1e-15)
    assert grid.radii_rad.max() <= grid.extent_rad + 1e-18


def test_cell_masses_are_positive_and_sum_below_one(grid):
    assert np.all(grid.prior_mass > 0.0)
    assert grid.prior_mass.sum() < 1.0


def test_nearest_cell_returns_the_cell_containing_the_point(grid):
    rng = np.random.default_rng(5)
    points = grid.centres_rad + rng.uniform(-0.3, 0.3, size=grid.centres_rad.shape) * grid.side_rad
    assert np.array_equal(grid.nearest_cell(points), np.arange(grid.n_cells))


def test_larger_extent_gives_more_cells_and_more_mass(uncertainty, geometry):
    small = build_cell_grid(uncertainty, geometry, extent_sigma=2.0)
    large = build_cell_grid(uncertainty, geometry, extent_sigma=3.0)
    assert large.n_cells > small.n_cells
    assert large.prior_mass.sum() > small.prior_mass.sum()


@pytest.mark.parametrize("bad", [0.0, -1.0e-4, float("nan"), float("inf")])
def test_pointing_uncertainty_rejects_bad_sigma(bad):
    with pytest.raises(ValueError, match="sigma_rad"):
        PointingUncertainty(bad)


def test_radial_cdf_rejects_negative_radius():
    with pytest.raises(ValueError, match="radius_rad"):
        PointingUncertainty(1.0e-4).radial_cdf(-1.0)


def test_rectangle_mass_rejects_inverted_bounds():
    with pytest.raises(ValueError, match="rectangle bounds"):
        PointingUncertainty(1.0e-4).rectangle_mass(1.0, 0.0, 0.0, 1.0)


def test_sample_rejects_negative_count():
    with pytest.raises(ValueError, match="n must be"):
        PointingUncertainty(1.0e-4).sample(-1, np.random.default_rng(0))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"beam_divergence_rad": 0.0},
        {"detector_fov_rad": -1.0},
        {"slew_rate_rad_s": 0.0},
        {"settle_time_s": -1.0},
    ],
)
def test_terminal_geometry_rejects_bad_values(kwargs):
    base = {
        "beam_divergence_rad": 1.0e-4,
        "detector_fov_rad": 2.0e-4,
        "slew_rate_rad_s": 1.0e-2,
        "settle_time_s": 1.0e-3,
    }
    base.update(kwargs)
    with pytest.raises(ValueError):
        TerminalGeometry(**base)


def test_slew_time_rejects_negative_distance(geometry):
    with pytest.raises(ValueError, match="distance_rad"):
        geometry.slew_time_s(-1.0e-6)


def test_build_cell_grid_rejects_bad_extent(uncertainty, geometry):
    with pytest.raises(ValueError, match="extent_sigma"):
        build_cell_grid(uncertainty, geometry, extent_sigma=0.0)


def test_nearest_cell_rejects_wrong_shape(grid):
    with pytest.raises(ValueError, match="shape"):
        grid.nearest_cell(np.zeros((4, 3)))
