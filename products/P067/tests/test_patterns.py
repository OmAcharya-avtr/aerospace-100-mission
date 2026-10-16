"""Scan paths and the cell visit orders they induce."""

from __future__ import annotations

import math

import numpy as np
import pytest

from acqsearch import (
    PATTERN_NAMES,
    cell_visit_order,
    lissajous_path,
    random_restart_path,
    raster_path,
    spiral_path,
    standard_paths,
    tail_fraction,
)


def test_spiral_starts_at_the_origin_and_grows():
    path = spiral_path(7.0e-5, 4.0, 3.5e-5)
    assert path.points_rad[0] == pytest.approx(np.zeros(2), abs=1e-18)
    radii = np.hypot(path.points_rad[:, 0], path.points_rad[:, 1])
    assert radii[-1] > radii[0]
    assert np.all(np.diff(radii) >= -1e-12)


def test_spiral_along_track_spacing_is_uniform():
    spacing = 3.5e-5
    path = spiral_path(7.0e-5, 5.0, spacing)
    steps = np.diff(path.points_rad, axis=0)
    lengths = np.hypot(steps[:, 0], steps[:, 1])
    # The sampling is uniform in ARC length, so the chord between consecutive
    # points is never longer than the spacing and is shorter where the spiral
    # curves tightly. Near the origin the curvature is extreme: the first chord
    # is 2.32e-5 rad against a 3.5e-5 rad arc (ratio 0.6624). The ratio rises
    # monotonically: 0.9364 at the second step, 0.9778 at the fourth, and
    # within 2 percent of the arc from the fifth onwards.
    assert np.all(lengths <= spacing * 1.001)
    assert np.all(lengths[4:] >= spacing * 0.98)
    assert np.all(np.diff(lengths) >= -1e-12)


def test_spiral_end_radius_matches_pitch_times_turns():
    path = spiral_path(7.0e-5, 4.0, 3.5e-5)
    assert path.metadata["end_radius_rad"] == pytest.approx(4.0 * 7.0e-5, rel=1e-12)


def test_raster_visits_every_lattice_point_once():
    path = raster_path(2.0e-4, 1.0e-4)
    # ceil(2e-4 / 1e-4) = 2, so the axis is {-2, -1, 0, 1, 2} * 1e-4: 25 points.
    assert path.n_points == 25
    assert np.unique(np.round(path.points_rad / 1.0e-4).astype(int), axis=0).shape[0] == 25


def test_raster_rows_alternate_direction():
    path = raster_path(2.0e-4, 1.0e-4)
    first_row = path.points_rad[:5, 0]
    second_row = path.points_rad[5:10, 0]
    assert np.all(np.diff(first_row) > 0)
    assert np.all(np.diff(second_row) < 0)


def test_lissajous_stays_inside_its_amplitude():
    path = lissajous_path(3.0e-4, 400)
    assert np.max(np.abs(path.points_rad)) <= 3.0e-4 + 1e-18


def test_random_restart_is_inside_the_disc_and_seeded():
    a = random_restart_path(3.0e-4, 500, 42)
    b = random_restart_path(3.0e-4, 500, 42)
    c = random_restart_path(3.0e-4, 500, 43)
    assert np.array_equal(a.points_rad, b.points_rad)
    assert not np.array_equal(a.points_rad, c.points_rad)
    assert np.max(np.hypot(a.points_rad[:, 0], a.points_rad[:, 1])) <= 3.0e-4 + 1e-18


def test_path_length_of_a_straight_raster_row():
    path = raster_path(1.0e-4, 1.0e-4)
    # Axis {-1, 0, 1} * 1e-4, three rows: 2e-4 along each row plus 1e-4 between rows.
    assert path.path_length_rad() == pytest.approx(3.0 * 2.0e-4 + 2.0 * 1.0e-4, rel=1e-12)


def test_path_length_of_a_single_point_is_zero():
    path = random_restart_path(1.0e-4, 1, 0)
    assert path.path_length_rad() == 0.0


@pytest.mark.parametrize("name", PATTERN_NAMES)
def test_every_standard_order_is_a_permutation_of_the_grid(grid, name):
    order = cell_visit_order(standard_paths(grid)[name], grid)
    assert np.array_equal(np.sort(order), np.arange(grid.n_cells))


def test_standard_paths_cover_the_declared_names(grid):
    assert tuple(sorted(standard_paths(grid))) == tuple(sorted(PATTERN_NAMES))


def test_spiral_order_starts_at_the_centre_cell(grid, spiral_order):
    assert grid.radii_rad[spiral_order[0]] == pytest.approx(0.0, abs=1e-18)


def test_spiral_order_is_broadly_outward(grid, spiral_order):
    radii = grid.radii_rad[spiral_order]
    first_half = radii[: grid.n_cells // 2].mean()
    second_half = radii[grid.n_cells // 2 :].mean()
    assert second_half > first_half


def test_tail_fraction_is_zero_for_the_raster(grid):
    assert tail_fraction(standard_paths(grid)["raster"], grid) == pytest.approx(0.0, abs=1e-15)


def test_tail_fraction_is_reported_for_the_lissajous(grid):
    # The Lissajous curve does not enter every cell; the order is completed by
    # a nearest-neighbour tail and the fraction is reported, not hidden.
    fraction = tail_fraction(standard_paths(grid)["lissajous"], grid)
    assert 0.0 < fraction < 0.2


def test_random_restart_has_the_longest_slew(grid):
    lengths = {}
    for name, path in standard_paths(grid).items():
        order = cell_visit_order(path, grid)
        steps = np.diff(grid.centres_rad[order], axis=0)
        lengths[name] = float(np.hypot(steps[:, 0], steps[:, 1]).sum())
    assert lengths["random_restart"] == max(lengths.values())


def test_cell_visit_order_on_a_single_point_path_still_covers_the_grid(grid):
    path = random_restart_path(grid.extent_rad, 1, 0)
    order = cell_visit_order(path, grid)
    assert np.array_equal(np.sort(order), np.arange(grid.n_cells))


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
def test_spiral_rejects_bad_pitch(bad):
    with pytest.raises(ValueError, match="pitch_rad"):
        spiral_path(bad, 2.0, 1.0e-5)


def test_spiral_rejects_bad_turns_and_spacing():
    with pytest.raises(ValueError, match="turns"):
        spiral_path(1.0e-4, 0.0, 1.0e-5)
    with pytest.raises(ValueError, match="along_track_spacing_rad"):
        spiral_path(1.0e-4, 2.0, 0.0)


def test_raster_rejects_bad_arguments():
    with pytest.raises(ValueError, match="extent_rad"):
        raster_path(0.0, 1.0e-5)
    with pytest.raises(ValueError, match="spacing_rad"):
        raster_path(1.0e-4, 0.0)


def test_lissajous_rejects_bad_arguments():
    with pytest.raises(ValueError, match="extent_rad"):
        lissajous_path(0.0, 100)
    with pytest.raises(ValueError, match="n_points"):
        lissajous_path(1.0e-4, 1)


def test_random_restart_rejects_bad_arguments():
    with pytest.raises(ValueError, match="extent_rad"):
        random_restart_path(0.0, 10, 0)
    with pytest.raises(ValueError, match="n_points"):
        random_restart_path(1.0e-4, 0, 0)


def test_lissajous_frequencies_appear_in_the_metadata():
    path = lissajous_path(1.0e-4, 200, freq_x=3.0, freq_y=4.0, phase_rad=0.25 * math.pi)
    assert path.metadata["freq_x"] == 3.0
    assert path.metadata["freq_y"] == 4.0
