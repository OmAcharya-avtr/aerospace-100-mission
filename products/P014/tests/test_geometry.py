"""Unit, KAT, edge-case tests for wavelab.geometry (PupilGrid, Hudgin/Fried matrices)."""

from __future__ import annotations

import numpy as np
import pytest

from wavelab.geometry import (
    PupilGrid,
    fried_matrix,
    hudgin_matrix,
    prune_unconstrained,
    waffle_pattern,
)
from wavelab.linalg import null_space

# --------------------------------------------------------------------- PupilGrid


def test_pupil_grid_rejects_small_n():
    with pytest.raises(ValueError):
        PupilGrid(2)


def test_pupil_grid_rejects_non_integer():
    with pytest.raises(TypeError):
        PupilGrid(5.5)


def test_pupil_grid_rejects_bad_obscuration():
    with pytest.raises(ValueError):
        PupilGrid(5, obscuration=1.0)
    with pytest.raises(ValueError):
        PupilGrid(5, obscuration=-0.1)


def test_pupil_grid_hand_calc_5x5():
    # 5x5 grid over [-1,1]: axis = [-1, -0.5, 0, 0.5, 1]. Corner (-1,-1) has
    # rho = sqrt(2) > 1 so it is excluded; centre (0,0) has rho=0, included.
    grid = PupilGrid(5)
    assert grid.mask[2, 2]  # centre
    assert not grid.mask[0, 0]  # corner, rho = sqrt(2) ~ 1.41 > 1
    assert grid.mask[0, 2]  # top-middle, rho = 1.0, included by default (edge kept)


def test_pupil_grid_n_active_matches_mask_sum():
    grid = PupilGrid(9)
    assert grid.n_active == int(grid.mask.sum())
    x, y = grid.active_coords()
    assert x.shape == (grid.n_active,)
    assert np.all(np.hypot(x, y) <= 1.0 + 1e-12)


def test_pupil_grid_to_full_round_trip():
    grid = PupilGrid(7)
    values = np.arange(grid.n_active, dtype=float)
    full = grid.to_full(values)
    assert full.shape == (7, 7)
    assert np.all(np.isnan(full[~grid.mask]))
    np.testing.assert_allclose(full[grid.mask], values)


def test_pupil_grid_to_full_rejects_wrong_shape():
    grid = PupilGrid(7)
    with pytest.raises(ValueError):
        grid.to_full(np.zeros(grid.n_active + 1))


# --------------------------------------------------------------------- Hudgin matrix


def test_hudgin_matrix_hand_calc_tiny_grid():
    # 3x3 full-square mask (obscuration=0, but corners have rho=sqrt(2)>1 and
    # are excluded by the circular mask) -- use a 3x3 grid but check the
    # in-mask cross shape directly against hand-built rows.
    grid = PupilGrid(3)
    mat = hudgin_matrix(grid)
    # Every row must have exactly one +1 and one -1, summing to zero (a pure
    # finite difference), by hand: any row r satisfies sum(r) == 0.
    assert np.allclose(mat.sum(axis=1), 0.0)
    assert np.all(np.isin(mat[mat != 0], [-1.0, 1.0]))


def test_hudgin_matrix_row_has_exactly_two_nonzeros():
    grid = PupilGrid(9)
    mat = hudgin_matrix(grid)
    nnz_per_row = np.sum(mat != 0.0, axis=1)
    assert np.all(nnz_per_row == 2)


def test_hudgin_null_space_is_piston_only():
    for n in (5, 7, 9, 11):
        grid = PupilGrid(n)
        mat, keep_idx = prune_unconstrained(hudgin_matrix(grid))
        ns = null_space(mat, rel_tol=1e-6)
        assert ns.shape[1] == 1, f"n_grid={n}: expected 1D null space (piston), got {ns.shape[1]}"
        # The null vector must be (up to sign/scale) the constant vector.
        v = ns[:, 0]
        v = v / v[0]
        np.testing.assert_allclose(v, np.ones_like(v), atol=1e-6)
        assert keep_idx.size == grid.n_active  # Hudgin never drops a point on these grids


def test_hudgin_matrix_recovers_known_ramp_noise_free():
    # A linear ramp phi(x,y) = a*x has constant x-difference a * grid_spacing
    # between every adjacent pair and exactly 0 for every y-difference row
    # (hand calculation: grid spacing = 2/(5-1) = 0.5 for a 5-point [-1,1] axis).
    grid = PupilGrid(5)
    mat = hudgin_matrix(grid)
    x, _y = grid.active_coords()
    a = 0.37
    spacing = 0.5
    phi = a * x
    s = mat @ phi
    # Rows are x-differences first (see hudgin_matrix docstring), so every
    # entry from that block equals +/- a * spacing exactly; y-difference rows
    # (the tail) must be exactly zero since phi does not depend on y.
    nonzero = s[np.abs(s) > 1e-12]
    np.testing.assert_allclose(np.abs(nonzero), abs(a) * spacing, atol=1e-9)


# --------------------------------------------------------------------- Fried matrix


def test_fried_matrix_row_has_exactly_four_nonzeros():
    grid = PupilGrid(9)
    mat = fried_matrix(grid)
    nnz_per_row = np.sum(mat != 0.0, axis=1)
    assert np.all(nnz_per_row == 4)


def test_fried_matrix_row_sums_to_zero():
    grid = PupilGrid(9)
    mat = fried_matrix(grid)
    np.testing.assert_allclose(mat.sum(axis=1), 0.0, atol=1e-12)


def test_fried_null_space_is_two_dimensional_piston_and_waffle():
    for n in (5, 7, 9, 11, 13):
        grid = PupilGrid(n)
        mat, keep_idx = prune_unconstrained(fried_matrix(grid))
        ns = null_space(mat, rel_tol=1e-6)
        assert ns.shape[1] == 2, f"n_grid={n}: expected piston+waffle, got {ns.shape[1]}"


def test_waffle_pattern_is_exactly_in_fried_null_space():
    grid = PupilGrid(9)
    mat = fried_matrix(grid)
    w = waffle_pattern(grid)
    residual = mat @ w
    np.testing.assert_allclose(residual, 0.0, atol=1e-10)


def test_waffle_pattern_is_unit_norm():
    grid = PupilGrid(9)
    w = waffle_pattern(grid)
    assert np.linalg.norm(w) == pytest.approx(1.0)


def test_waffle_pattern_hand_calc_checkerboard_sign():
    # Grid index (i=0,j=0) is (0+0)%2==0 -> +1; (i=1,j=0) -> -1.
    grid = PupilGrid(9)
    n = grid.n_grid
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="xy")
    checker = np.where((ii + jj) % 2 == 0, 1.0, -1.0)
    w = waffle_pattern(grid)
    raw = checker[grid.mask]
    np.testing.assert_allclose(w, raw / np.linalg.norm(raw))


def test_fried_matrix_not_reproducible_from_piston_alone():
    # A pure piston phase (all equal) must forward-map to exactly zero slopes.
    grid = PupilGrid(9)
    mat, keep_idx = prune_unconstrained(fried_matrix(grid))
    phi = np.ones(mat.shape[1]) * 3.14
    np.testing.assert_allclose(mat @ phi, 0.0, atol=1e-10)


# --------------------------------------------------------------------- prune_unconstrained


def test_prune_unconstrained_drops_zero_columns():
    mat = np.array([[1.0, 0.0, -1.0], [0.0, 0.0, 0.0]])
    pruned, keep = prune_unconstrained(mat)
    np.testing.assert_array_equal(keep, [0, 2])
    assert pruned.shape == (2, 2)


def test_prune_unconstrained_rejects_all_zero_matrix():
    with pytest.raises(ValueError):
        prune_unconstrained(np.zeros((3, 3)))


def test_prune_unconstrained_no_op_when_all_columns_used():
    mat = np.eye(3)
    pruned, keep = prune_unconstrained(mat)
    np.testing.assert_array_equal(keep, [0, 1, 2])
    np.testing.assert_allclose(pruned, mat)
