"""Subaperture layout and the Southwell / Fried geometry matrices."""

from __future__ import annotations

import numpy as np
import pytest

from wavelab.geometry import SubapertureGeometry, build_geometry_matrices


@pytest.fixture(name="geom")
def _geom() -> SubapertureGeometry:
    return SubapertureGeometry(n_sub=8, diameter=1.0, fill_threshold=0.5)


def test_pitch_and_scaling(geom: SubapertureGeometry) -> None:
    """Hand calculation: D = 1 m over 8 subapertures gives h = 0.125 m."""
    assert geom.pitch == pytest.approx(0.125)
    # u = h * dphi/dx and dphi/dx = (2/D) dphi/dxn, so the factor is 2/n_sub.
    assert geom.scaled_slope_factor == pytest.approx(0.25)


def test_mask_is_symmetric_and_plausible(geom: SubapertureGeometry) -> None:
    mask = geom.mask
    assert mask.shape == (8, 8)
    assert np.array_equal(mask, mask[::-1])
    assert np.array_equal(mask, mask[:, ::-1])
    assert np.array_equal(mask, mask.T)
    # 52 of 64 cells are at least half inside the inscribed circle; the count is
    # bounded by the area ratio pi/4 * 64 = 50.3 and by 64.
    assert geom.n_valid_sub == 52
    assert geom.n_slopes == 104


def test_fill_threshold_monotone() -> None:
    counts = [
        SubapertureGeometry(8, 1.0, fill_threshold=f).n_valid_sub for f in (0.1, 0.5, 0.9, 1.0)
    ]
    assert counts == sorted(counts, reverse=True)


def test_subaperture_centres_inside_unit_disc(geom: SubapertureGeometry) -> None:
    cx, cy = geom.subaperture_centres()
    assert cx.shape == (52,)
    assert np.all(cx**2 + cy**2 <= 1.0)
    # First illuminated cell of the first row: iy = 0, ix = 2 -> x = 2*0.25-1+0.125
    assert cx[0] == pytest.approx(-0.375)
    assert cy[0] == pytest.approx(-0.875)


def test_corner_grid(geom: SubapertureGeometry) -> None:
    fx, fy = geom.fried_points()
    assert fx.size == int(np.count_nonzero(geom.corner_mask()))
    assert fx.size == 69
    assert np.all(np.abs(fx) <= 1.0 + 1e-12)


def test_southwell_matrix_structure(geom: SubapertureGeometry) -> None:
    gm = build_geometry_matrices(geom, "southwell")
    assert gm.geometry == "southwell"
    assert gm.n_phase == 52
    assert gm.n_slopes == 104
    # Each equation is one +1 / one -1 on the phase side and two 1/2 on the
    # slope side, by construction (Southwell 1980, Eq. 5).
    assert np.all(np.sort(gm.a, axis=1)[:, [0, -1]] == np.array([-1.0, 1.0]))
    assert np.allclose(gm.b.sum(axis=1), 1.0)
    assert np.allclose(np.abs(gm.b).sum(axis=1), 1.0)


def test_fried_matrix_structure(geom: SubapertureGeometry) -> None:
    gm = build_geometry_matrices(geom, "fried")
    assert gm.n_phase == 69
    assert gm.a.shape == (104, 69)
    # b is a permutation of the slope vector: exactly one equation per
    # measurement (rows interleave the x and y equation of each subaperture).
    assert np.array_equal(np.sort(gm.b, axis=0), np.sort(np.eye(104), axis=0))
    assert np.array_equal(gm.b.sum(axis=0), np.ones(104))
    assert np.array_equal(gm.b.sum(axis=1), np.ones(104))
    # Each Fried row has two +1/2 and two -1/2 entries.
    assert np.allclose(np.abs(gm.a).sum(axis=1), 2.0)
    assert np.allclose(gm.a.sum(axis=1), 0.0)


def test_piston_is_in_the_null_space(geom: SubapertureGeometry) -> None:
    for name in ("southwell", "fried"):
        gm = build_geometry_matrices(geom, name)
        assert np.allclose(gm.a @ np.ones(gm.n_phase), 0.0)


def test_fried_has_a_waffle_null_mode(geom: SubapertureGeometry) -> None:
    """Fried geometry is blind to the checkerboard mode; Southwell is not.

    Southwell 1980 Sec. V; Hardy 1998 ch. 5. The consequence is a second
    near-zero singular value for Fried and only one for Southwell.
    """
    sv_s = np.linalg.svd(build_geometry_matrices(geom, "southwell").a, compute_uv=False)
    sv_f = np.linalg.svd(build_geometry_matrices(geom, "fried").a, compute_uv=False)
    assert np.count_nonzero(sv_s < 1e-9 * sv_s[0]) == 1
    assert np.count_nonzero(sv_f < 1e-9 * sv_f[0]) == 2


def test_active_rows_drops_equations(geom: SubapertureGeometry) -> None:
    gm = build_geometry_matrices(geom, "fried")
    avail = np.ones(52, dtype=bool)
    assert np.count_nonzero(gm.active_rows(avail)) == 104
    avail[0] = False
    assert np.count_nonzero(gm.active_rows(avail)) == 102

    gm_s = build_geometry_matrices(geom, "southwell")
    full = np.count_nonzero(gm_s.active_rows(np.ones(52, dtype=bool)))
    one_out = np.count_nonzero(gm_s.active_rows(avail))
    assert one_out < full


def test_geometry_validation() -> None:
    with pytest.raises(ValueError, match="n_sub must be >= 2"):
        SubapertureGeometry(1, 1.0)
    with pytest.raises(TypeError, match="n_sub must be an integer"):
        SubapertureGeometry(4.0, 1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="diameter"):
        SubapertureGeometry(4, -1.0)
    with pytest.raises(ValueError, match="fill_threshold"):
        SubapertureGeometry(4, 1.0, fill_threshold=0.0)
    with pytest.raises(ValueError, match="southwell"):
        build_geometry_matrices(SubapertureGeometry(4, 1.0), "hudgin")
    with pytest.raises(TypeError, match="SubapertureGeometry"):
        build_geometry_matrices("not a geometry", "fried")  # type: ignore[arg-type]


def test_active_rows_shape_validation(geom: SubapertureGeometry) -> None:
    gm = build_geometry_matrices(geom, "fried")
    with pytest.raises(ValueError, match="sub_available"):
        gm.active_rows(np.ones(7, dtype=bool))
