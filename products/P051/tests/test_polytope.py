"""Unit, input-validation, known-answer and edge-case tests for the set algebra."""

from __future__ import annotations

import numpy as np
import pytest

from simplexguard.polytope import Box, Polytope, box, unit_normalise


def test_box_support_known_answer():
    # Hand calculation. W = {w : |w1| <= 2, |w2| <= 3}, c = (1, -4).
    # h_W(c) = sup_w c.w = max over the four vertices:
    #   (2, 3)   -> 1*2 + (-4)*3  = 2 - 12 = -10
    #   (2, -3)  -> 1*2 + (-4)*-3 = 2 + 12 =  14
    #   (-2, 3)  -> -2 - 12 = -14
    #   (-2, -3) -> -2 + 12 =  10
    # so h_W(c) = 14, which also equals |c|.r = 1*2 + 4*3 = 2 + 12 = 14.
    w = box([2.0, 3.0])
    assert w.support(np.array([1.0, -4.0])) == pytest.approx(14.0)


def test_box_support_matches_vertex_enumeration(rng):
    w = Box(np.array([-1.0, 0.5, -2.0]), np.array([3.0, 2.5, 0.0]))
    verts = w.vertices()
    assert verts.shape == (8, 3)
    for _ in range(200):
        c = rng.normal(size=3)
        assert w.support(c) == pytest.approx(float(np.max(verts @ c)), abs=1e-12)


def test_generic_polytope_support_matches_box_closed_form(rng):
    w = box([0.4, 1.7])
    generic = Polytope(w.A, w.b)
    for _ in range(50):
        c = rng.normal(size=2)
        assert generic.support(c) == pytest.approx(w.support(c), abs=1e-9)


def test_erode_known_answer():
    # Hand calculation. P = {|x1| <= 1, |x2| <= 1} with rows
    #   (1,0)<=1, (0,1)<=1, (-1,0)<=1, (0,-1)<=1
    # W = {|w1| <= 0.1, |w2| <= 0.25}. h_W((1,0)) = 0.1, h_W((0,1)) = 0.25,
    # and the same for the negated rows, so
    #   P (-) W = {|x1| <= 0.9, |x2| <= 0.75}.
    p = box([1.0, 1.0])
    eroded = p.erode(box([0.1, 0.25]))
    assert eroded.b == pytest.approx(np.array([0.9, 0.75, 0.9, 0.75]))


def test_erode_is_exactly_the_pontryagin_difference(rng):
    p = Polytope(
        np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0], [1.0, -1.0]]),
        np.array([1.0, 1.0, 1.0, 1.0]),
    )
    w = box([0.07, 0.11])
    eroded = p.erode(w)
    verts = w.vertices()
    for _ in range(500):
        x = rng.uniform(-2.0, 2.0, size=2)
        inside_eroded = eroded.contains(x, tol=0.0)
        all_shifts_inside = all(p.contains(x + v, tol=1e-12) for v in verts)
        assert inside_eroded == all_shifts_inside


def test_preimage_identity(rng):
    p = box([1.0, 2.0])
    m = np.array([[0.5, 0.25], [-1.0, 0.75]])
    pre = p.preimage(m)
    for _ in range(300):
        x = rng.uniform(-6.0, 6.0, size=2)
        assert pre.contains(x, tol=1e-12) == p.contains(m @ x, tol=1e-12)


def test_chebyshev_radius_known_answer():
    # The unit square [-1,1]^2 has an inscribed circle of radius 1.
    assert box([1.0, 1.0]).chebyshev_radius() == pytest.approx(1.0, abs=1e-9)
    # A 2-by-6 box has inscribed radius 1 (half of the short side).
    assert box([1.0, 3.0]).chebyshev_radius() == pytest.approx(1.0, abs=1e-9)


def test_empty_polytope_is_detected_and_support_is_minus_infinity():
    # {x : x <= -1 and -x <= -1} is empty. The Chebyshev programme leaves the
    # radius free, so it is feasible with a negative optimum: the two rows are
    # 2 apart in the wrong direction, each with unit normal, so the best "ball"
    # has radius (-1 + -1)/2 = -1.
    empty = Polytope(np.array([[1.0], [-1.0]]), np.array([-1.0, -1.0]))
    assert empty.is_empty()
    assert empty.chebyshev_radius() == pytest.approx(-1.0, abs=1e-9)
    assert empty.support(np.array([1.0])) == -np.inf


def test_unbounded_polytope_support_is_plus_infinity():
    half = Polytope(np.array([[1.0, 0.0]]), np.array([1.0]))
    assert half.support(np.array([-1.0, 0.0])) == np.inf
    assert half.support(np.array([0.0, 1.0])) == np.inf


def test_minimal_removes_redundant_rows_and_keeps_the_square():
    rows = np.vstack([np.eye(2), -np.eye(2), np.array([[1.0, 0.0]])])
    offs = np.array([1.0, 1.0, 1.0, 1.0, 5.0])  # last row is far away: redundant
    reduced = Polytope(rows, offs).minimal()
    assert reduced.n_halfspaces == 4
    assert reduced.area_2d() == pytest.approx(4.0, abs=1e-9)


def test_minimal_keeps_exactly_one_of_a_duplicated_row():
    rows = np.vstack([np.eye(2), -np.eye(2), np.array([[1.0, 0.0]])])
    offs = np.array([1.0, 1.0, 1.0, 1.0, 1.0])  # duplicate of row 0
    reduced = Polytope(rows, offs).minimal()
    assert reduced.n_halfspaces == 4


def test_area_and_vertices_of_a_triangle():
    # Triangle with vertices (0,0), (2,0), (0,3): area = 0.5 * 2 * 3 = 3.
    p = Polytope(
        np.array([[-1.0, 0.0], [0.0, -1.0], [1.5, 1.0]]),
        np.array([0.0, 0.0, 3.0]),
    )
    assert p.vertices_2d().shape == (3, 2)
    assert p.area_2d() == pytest.approx(3.0, abs=1e-9)


def test_contains_polytope_subset_test():
    outer = box([1.0, 1.0])
    inner = box([0.5, 0.5])
    assert outer.contains_polytope(inner)
    assert not inner.contains_polytope(outer)
    assert outer.contains_polytope(outer)


def test_contains_many_matches_contains_row_by_row(rng):
    p = box([0.3, 0.5])
    pts = rng.uniform(-1.0, 1.0, size=(200, 2))
    expected = np.array([p.contains(x) for x in pts])
    assert np.array_equal(p.contains_many(pts), expected)


def test_unit_normalise_gives_unit_rows_and_preserves_the_set():
    a = np.array([[3.0, 4.0], [-6.0, 8.0]])
    b = np.array([10.0, 20.0])
    na, nb = unit_normalise(a, b)
    assert np.allclose(np.linalg.norm(na, axis=1), 1.0)
    assert nb == pytest.approx(np.array([2.0, 2.0]))


def test_unit_normalise_leaves_a_zero_row_alone():
    a = np.array([[0.0, 0.0], [1.0, 0.0]])
    b = np.array([-1.0, 2.0])
    na, nb = unit_normalise(a, b)
    assert np.allclose(na[0], 0.0)
    assert nb[0] == pytest.approx(-1.0)


@pytest.mark.parametrize(
    ("rows", "offs", "match"),
    [
        (np.zeros((0, 2)), np.zeros(0), "at least one halfspace"),
        (np.zeros(3), np.zeros(3), "must be 2-D"),
        (np.eye(2), np.zeros((2, 1)), "must be 1-D"),
        (np.eye(2), np.zeros(3), "rows but b has"),
        (np.array([[np.nan, 0.0]]), np.zeros(1), "non-finite"),
        (np.eye(1), np.array([np.inf]), "non-finite"),
    ],
)
def test_polytope_constructor_rejects_bad_input(rows, offs, match):
    with pytest.raises(ValueError, match=match):
        Polytope(rows, offs)


def test_support_rejects_wrong_length_and_non_finite_direction():
    p = box([1.0, 1.0])
    with pytest.raises(ValueError, match="expected 2"):
        p.support(np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="non-finite"):
        p.support(np.array([np.nan, 1.0]))


def test_box_constructor_rejects_bad_bounds():
    with pytest.raises(ValueError, match="upper bound is below"):
        Box(np.array([1.0]), np.array([0.0]))
    with pytest.raises(ValueError, match="shape"):
        Box(np.array([1.0, 2.0]), np.array([3.0]))
    with pytest.raises(ValueError, match="at least one dimension"):
        Box(np.zeros(0), np.zeros(0))
    with pytest.raises(ValueError, match="finite"):
        Box(np.array([-np.inf]), np.array([1.0]))
    with pytest.raises(ValueError, match="non-negative"):
        box([-1.0])


def test_dimension_mismatch_errors():
    with pytest.raises(ValueError, match="dimension mismatch"):
        box([1.0, 1.0]).intersect(box([1.0]))
    with pytest.raises(ValueError, match="dimension mismatch"):
        box([1.0, 1.0]).erode(box([1.0]))
    with pytest.raises(ValueError, match="dimension mismatch"):
        box([1.0, 1.0]).contains_polytope(box([1.0]))
    with pytest.raises(ValueError, match="must have 2 rows"):
        box([1.0, 1.0]).preimage(np.eye(3))


def test_erode_rejects_an_unbounded_disturbance_set():
    unbounded = Polytope(np.array([[1.0, 0.0]]), np.array([1.0]))
    with pytest.raises(ValueError, match="unbounded"):
        box([1.0, 1.0]).erode(unbounded)


def test_scaled_rejects_non_positive_factors():
    with pytest.raises(ValueError, match="positive"):
        Polytope(np.eye(2), np.ones(2)).scaled(0.0)
    with pytest.raises(ValueError, match="non-negative"):
        box([1.0]).scaled(-1.0)


def test_box_scaled_about_the_centre_known_answer():
    b = Box(np.array([2.0]), np.array([6.0]))  # centre 4, half-width 2
    s = b.scaled(0.5)
    assert s.lower == pytest.approx(np.array([3.0]))
    assert s.upper == pytest.approx(np.array([5.0]))


def test_vertices_2d_requires_two_dimensions():
    with pytest.raises(ValueError, match="dim == 2"):
        box([1.0, 1.0, 1.0]).vertices_2d()


def test_vertices_2d_of_a_degenerate_set_is_empty():
    strip = Polytope(np.array([[1.0, 0.0], [-1.0, 0.0]]), np.array([1.0, 1.0]))
    assert strip.vertices_2d().shape == (0, 2)
    assert strip.area_2d() == 0.0


def test_box_refuses_to_enumerate_too_many_vertices():
    with pytest.raises(ValueError, match="refusing to enumerate"):
        box(np.ones(17)).vertices()


def test_slack_and_residual_are_negatives_of_each_other():
    p = box([1.0, 2.0])
    x = np.array([0.25, -0.5])
    assert p.slack(x) == pytest.approx(float(np.min(-p.residual(x))))
    assert p.slack(x) == pytest.approx(0.75)  # 1 - 0.25 is the tightest row
