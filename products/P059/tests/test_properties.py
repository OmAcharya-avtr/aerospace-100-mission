"""Hypothesis property tests for the set algebra.

The point of this file is the pair of identities in `setalgebra`:

    (I3)  (P (-) Q) (+) Q  subset  P          -- holds, with equality only
                                                 in special cases
    (I4)  (P (+) Q) (-) Q  =  P               -- holds for compact convex sets

**The identity that is commonly assumed, `(P (-) Q) (+) Q = P`, is false**,
and `TestPontryaginIsNotAnInverse` tests the containment that actually holds
rather than the equation that does not.  Sources: Kolmanovsky and Gilbert
1998; Blanchini, *Automatica* 35(11), 1999, pp. 1747-1768; Schneider 1993.

Generated polytopes always contain the origin in their interior and always
carry an axis-aligned bounding box, so every instance is non-empty and
bounded by construction.  That keeps the generators from spending the example
budget on degenerate cases the operations are documented not to accept.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from invariantset import (
    CONVERGED,
    EMPTY,
    Polytope,
    intersect,
    is_subset,
    maximal_robust_invariant_set,
    minkowski_sum,
    pontryagin_difference,
    verify_robust_invariance,
)

SETTINGS = settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
SLOW_SETTINGS = settings(
    max_examples=15,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

_finite = st.floats(min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False)


@st.composite
def boxes(draw, dim: int, min_half: float = 0.5, max_half: float = 2.0):
    """Axis-aligned boxes centred at the origin, half-widths in a range."""
    half = draw(
        st.lists(
            st.floats(min_value=min_half, max_value=max_half),
            min_size=dim,
            max_size=dim,
        )
    )
    return Polytope.from_box(np.zeros(dim), np.array(half))


@st.composite
def bounded_polytopes(draw, dim: int, min_rhs: float = 0.5):
    """Bounded polytopes with the origin strictly inside.

    An axis-aligned box of half-width at least `min_rhs` guarantees
    boundedness; the extra halfspaces all have right-hand side at least
    `min_rhs`, so the origin is interior and the set is non-empty.
    """
    half = draw(
        st.lists(st.floats(min_value=min_rhs, max_value=2.0), min_size=dim, max_size=dim)
    )
    box = Polytope.from_box(np.zeros(dim), np.array(half))
    n_extra = draw(st.integers(min_value=0, max_value=3))
    rows = [box.A]
    rhs = [box.b]
    for _ in range(n_extra):
        normal = np.array(
            draw(st.lists(_finite, min_size=dim, max_size=dim))
        )
        norm = float(np.linalg.norm(normal))
        assume(norm > 1e-3)
        rows.append((normal / norm).reshape(1, -1))
        rhs.append(np.array([draw(st.floats(min_value=min_rhs, max_value=3.0))]))
    return Polytope(np.vstack(rows), np.concatenate(rhs))


@st.composite
def directions(draw, dim: int):
    c = np.array(draw(st.lists(_finite, min_size=dim, max_size=dim)))
    assume(np.linalg.norm(c) > 1e-6)
    return c


class TestSupportFunctionProperties:
    @SETTINGS
    @given(P=bounded_polytopes(2), c=directions(2), t=st.floats(min_value=0.0, max_value=5.0))
    def test_positive_homogeneity(self, P, c, t):
        # h(t c) = t h(c) for t >= 0 (Rockafellar 1970).
        assert P.support(t * c) == pytest.approx(t * P.support(c), abs=1e-7, rel=1e-9)

    @SETTINGS
    @given(P=bounded_polytopes(2), c1=directions(2), c2=directions(2))
    def test_subadditivity(self, P, c1, c2):
        # h(c1 + c2) <= h(c1) + h(c2): the support function is convex.
        assert P.support(c1 + c2) <= P.support(c1) + P.support(c2) + 1e-8

    @SETTINGS
    @given(P=boxes(2), c=directions(2))
    def test_box_closed_form_matches_the_generic_lp(self, P, c):
        generic = Polytope(P.A, P.b)
        assert P.support(c) == pytest.approx(generic.support(c), abs=1e-9)

    @SETTINGS
    @given(P=bounded_polytopes(2), c=directions(2))
    def test_support_is_attained_at_a_vertex(self, P, c):
        V = P.vertices()
        assert P.support(c) == pytest.approx(float(np.max(V @ c)), abs=1e-7)


class TestMinkowskiAdditivity:
    @SLOW_SETTINGS
    @given(P=bounded_polytopes(2), Q=boxes(2, 0.05, 0.3), c=directions(2))
    def test_support_of_the_sum_adds(self, P, Q, c):
        # (I1) h_{P (+) Q}(c) = h_P(c) + h_Q(c), Schneider 1993.
        S = minkowski_sum(P, Q)
        assert S.support(c) == pytest.approx(P.support(c) + Q.support(c), abs=1e-7)

    @SETTINGS
    @given(P=boxes(2), Q=boxes(2, 0.05, 0.3))
    def test_boxes_add_half_widths(self, P, Q):
        S = minkowski_sum(P, Q)
        for axis in (np.array([1.0, 0.0]), np.array([0.0, 1.0])):
            assert S.support(axis) == pytest.approx(
                P.support(axis) + Q.support(axis), abs=1e-7
            )


class TestPontryaginIsNotAnInverse:
    """The asymmetry between (I3) and (I4) in executable form."""

    @SLOW_SETTINGS
    @given(P=bounded_polytopes(2, min_rhs=0.6), Q=boxes(2, 0.02, 0.15))
    def test_i3_erode_then_dilate_is_contained(self, P, Q):
        # (P (-) Q) (+) Q subset P.  This is the identity that holds.  It is
        # NOT an equality: see test_strictness_is_reachable below and
        # tests/test_setalgebra.py::TestTheAsymmetry.
        eroded = pontryagin_difference(P, Q)
        assume(not eroded.is_empty())
        reopened = minkowski_sum(eroded, Q)
        contained, margin = is_subset(reopened, P)
        assert contained, f"(P-Q)+Q escaped P by {margin:.3e}"
        assert margin <= 1e-7

    @SLOW_SETTINGS
    @given(P=bounded_polytopes(2, min_rhs=0.6), Q=boxes(2, 0.02, 0.15))
    def test_i4_dilate_then_erode_recovers_p(self, P, Q):
        # (P (+) Q) (-) Q = P for compact convex P, Q.  This one IS an
        # equality, and it is the direction people do not expect.
        closed = pontryagin_difference(minkowski_sum(P, Q), Q)
        forward, m_fwd = is_subset(closed, P)
        backward, m_bwd = is_subset(P, closed)
        assert forward and backward, f"margins {m_fwd:.3e}, {m_bwd:.3e}"
        assert max(m_fwd, m_bwd) <= 1e-7

    @SLOW_SETTINGS
    @given(P=bounded_polytopes(2, min_rhs=0.6), Q=boxes(2, 0.02, 0.15))
    def test_erode_then_dilate_volume_never_exceeds_p(self, P, Q):
        eroded = pontryagin_difference(P, Q)
        assume(not eroded.is_empty())
        assert minkowski_sum(eroded, Q).volume() <= P.volume() + 1e-9

    def test_strictness_is_reachable(self):
        # A single deterministic witness that (I3) is strict, so that the
        # property tests above cannot be satisfied by equality alone.
        P = Polytope(
            np.array([[-1.0, 0.0], [0.0, -1.0], [1.0, 1.0]]),
            np.array([0.0, 0.0, 1.0]),
        )
        Q = Polytope.from_box([0.0, 0.0], [0.1, 0.1])
        reopened = minkowski_sum(pontryagin_difference(P, Q), Q)
        assert reopened.volume() < P.volume() - 1e-3
        assert not is_subset(P, reopened)[0]

    @SETTINGS
    @given(Q=boxes(2, 0.02, 0.2), extra=st.floats(min_value=0.0, max_value=1.0))
    def test_pontryagin_is_monotone_in_the_first_argument(self, Q, extra):
        # P subset P' implies P (-) Q subset P' (-) Q.
        P = Polytope.from_box([0.0, 0.0], [1.0, 1.0])
        P2 = Polytope.from_box([0.0, 0.0], [1.0 + extra, 1.0 + extra])
        D1 = pontryagin_difference(P, Q)
        D2 = pontryagin_difference(P2, Q)
        assume(not D1.is_empty())
        assert is_subset(D1, D2)[0]

    @SETTINGS
    @given(P=bounded_polytopes(2), Q=boxes(2, 0.02, 0.3), c=directions(2))
    def test_pontryagin_support_is_at_least_the_difference(self, P, Q, c):
        # h_{P (-) Q}(c) >= h_P(c) - h_Q(c) is NOT generally an equality; the
        # inequality is what holds, and this test asserts only that.
        D = pontryagin_difference(P, Q)
        assume(not D.is_empty())
        assert D.support(c) <= P.support(c) - Q.support(-c) + 1e-7


class TestIntersection:
    @SETTINGS
    @given(P=bounded_polytopes(2), Q=bounded_polytopes(2), c=directions(2))
    def test_support_of_intersection_is_at_most_the_minimum(self, P, Q, c):
        R = intersect(P, Q)
        assert R.support(c) <= min(P.support(c), Q.support(c)) + 1e-8

    @SETTINGS
    @given(P=bounded_polytopes(2), Q=bounded_polytopes(2))
    def test_intersection_is_contained_in_both(self, P, Q):
        R = intersect(P, Q)
        assert is_subset(R, P)[0]
        assert is_subset(R, Q)[0]


class TestRedundancyRemoval:
    @SETTINGS
    @given(
        P=bounded_polytopes(2),
        pts=st.lists(
            st.tuples(
                st.floats(min_value=-3.0, max_value=3.0),
                st.floats(min_value=-3.0, max_value=3.0),
            ),
            min_size=1,
            max_size=25,
        ),
    )
    def test_removal_preserves_membership(self, P, pts):
        R = P.remove_redundant(tol=1e-10)
        for pt in pts:
            x = np.array(pt)
            # Points within 1e-7 of a facet are excluded: a boundary point can
            # legitimately flip when a tangent facet is dropped.
            slack = np.min(P.b - P.A @ x)
            if abs(slack) < 1e-7:
                continue
            assert P.contains(x, tol=0.0) == R.contains(x, tol=0.0)

    @SETTINGS
    @given(P=bounded_polytopes(2), c=directions(2))
    def test_removal_preserves_the_support_function(self, P, c):
        assert P.remove_redundant(tol=1e-10).support(c) == pytest.approx(
            P.support(c), abs=1e-7
        )


class TestInvariantSetProperties:
    @SLOW_SETTINGS
    @given(
        rho=st.floats(min_value=0.3, max_value=0.9),
        theta=st.floats(min_value=0.0, max_value=np.pi),
        w=st.floats(min_value=0.001, max_value=0.05),
    )
    def test_result_is_inside_x_and_invariant_when_converged(self, rho, theta, w):
        A = rho * np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        X = Polytope.from_box([0.0, 0.0], [1.0, 1.0])
        W = Polytope.from_box([0.0, 0.0], [w, w])
        res = maximal_robust_invariant_set(
            A, X, W, max_iter=30, track_geometry=False
        )
        if res.termination == EMPTY:
            assert res.polytope is None
            return
        assert is_subset(res.polytope, X, tol=1e-7)[0]
        if res.termination == CONVERGED:
            invariant, margin = verify_robust_invariance(A, res.polytope, W)
            assert invariant or margin <= 1e-7, f"margin {margin:.3e}"

    @SLOW_SETTINGS
    @given(
        rho=st.floats(min_value=0.3, max_value=0.9),
        w=st.floats(min_value=0.001, max_value=0.05),
    )
    def test_the_sequence_is_nested_and_decreasing(self, rho, w):
        A = np.array([[rho, 0.1], [0.0, rho]])
        X = Polytope.from_box([0.0, 0.0], [1.0, 1.0])
        W = Polytope.from_box([0.0, 0.0], [w, w])
        res = maximal_robust_invariant_set(A, X, W, max_iter=20, track_geometry=True)
        volumes = [r.volume for r in res.history if r.volume is not None]
        for later, earlier in zip(volumes[1:], volumes[:-1], strict=True):
            assert later <= earlier + 1e-9

    @SLOW_SETTINGS
    @given(
        lam=st.floats(min_value=0.05, max_value=0.95),
        w=st.floats(min_value=0.001, max_value=0.4),
    )
    def test_one_dimensional_emptiness_matches_the_closed_form(self, lam, w):
        # In 1-D the maximal set is X when b >= w / (1 - lambda) and empty
        # otherwise.  b is fixed at 1.  Cases within 1 % of the threshold are
        # skipped: there the answer is decided by floating point, not algebra.
        b = 1.0
        threshold = w / (1.0 - lam)
        assume(abs(b - threshold) > 0.01 * max(b, threshold))
        res = maximal_robust_invariant_set(
            np.array([[lam]]),
            Polytope.from_box([0.0], [b]),
            Polytope.from_box([0.0], [w]),
            max_iter=400,
            track_geometry=False,
        )
        if b >= threshold:
            assert res.termination == CONVERGED
            assert res.polytope.volume() == pytest.approx(2.0 * b, rel=1e-9)
        else:
            assert res.termination == EMPTY
