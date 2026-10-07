"""Hypothesis property tests for the support-function and set algebra.

The identities checked here are the ones the exact switching condition rests on.
If any of them fails, the condition in :mod:`simplexguard.guard` is not exact and
no number this package reports means anything, so they are property-tested over
generated inputs rather than spot-checked.

Identities (references in ``simplexguard/polytope.py``)
  positive homogeneity   h_P(t c) = t h_P(c) for t >= 0
  subadditivity          h_P(c1 + c2) <= h_P(c1) + h_P(c2)
  reflection             h_{-P}(c) = h_P(-c)
  box closed form        h_box(c) = c.m + |c|.r, equal to the generic LP
  Minkowski additivity   h_{P+W}(c) = h_P(c) + h_W(c)
  erosion               x in P (-) W  <=>  x + w in P for every w in W
  erosion monotonicity   W1 subset of W2  =>  P (-) W2 subset of P (-) W1
  preimage              x in M^{-1} P  <=>  M x in P
  scaling               h_{tP}(c) = t h_P(c) for t > 0 about the origin
"""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from simplexguard.polytope import Box, Polytope, box

SETTINGS = settings(
    max_examples=120,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

finite = st.floats(
    min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False
)
positive = st.floats(
    min_value=0.05, max_value=5.0, allow_nan=False, allow_infinity=False
)


def _vec(n: int):
    return st.lists(finite, min_size=n, max_size=n).map(np.array)


def _radii(n: int):
    return st.lists(positive, min_size=n, max_size=n).map(np.array)


@SETTINGS
@given(_radii(2), _vec(2), st.floats(min_value=0.0, max_value=4.0, allow_nan=False))
def test_support_is_positively_homogeneous(radii, c, t):
    w = box(radii)
    generic = Polytope(w.A, w.b)
    for p in (w, generic):
        expected = t * p.support(c)
        assert abs(p.support(t * c) - expected) <= 1e-7 * (1.0 + abs(expected))


@SETTINGS
@given(_radii(2), _vec(2), _vec(2))
def test_support_is_subadditive(radii, c1, c2):
    w = box(radii)
    assert w.support(c1 + c2) <= w.support(c1) + w.support(c2) + 1e-9


@SETTINGS
@given(_radii(3), _vec(3))
def test_support_reflection_identity(radii, c):
    w = Box(-radii, radii)
    reflected = Box(-radii, radii)  # symmetric, so -W = W
    assert reflected.support(c) == w.support(-c)


@SETTINGS
@given(_vec(2), _radii(2), _vec(2))
def test_box_closed_form_equals_the_generic_linear_programme(centre, radii, c):
    w = Box(centre - radii, centre + radii)
    generic = Polytope(w.A, w.b)
    assert abs(generic.support(c) - w.support(c)) <= 1e-7 * (1.0 + abs(w.support(c)))


@SETTINGS
@given(_radii(2), _radii(2), _vec(2))
def test_support_of_a_minkowski_sum_adds(radii_p, radii_w, c):
    # The Minkowski sum of two origin-centred boxes is the box of summed radii.
    p, w = box(radii_p), box(radii_w)
    total = box(radii_p + radii_w)
    assert abs(total.support(c) - (p.support(c) + w.support(c))) <= 1e-9 * (
        1.0 + abs(total.support(c))
    )


@SETTINGS
@given(_radii(2), _radii(2), _vec(2))
def test_erosion_is_the_pontryagin_difference(radii_p, radii_w, x):
    p, w = box(radii_p), box(radii_w)
    eroded = p.erode(w)
    inside = eroded.contains(x, tol=1e-12)
    shifted = all(p.contains(x + v, tol=1e-12) for v in w.vertices())
    if abs(eroded.slack(x)) > 1e-9:  # not on the boundary, where tol decides
        assert inside == shifted


@SETTINGS
@given(_radii(2), _radii(2), positive)
def test_erosion_is_monotone_in_the_disturbance_set(radii_p, radii_w, factor):
    assume(factor > 1.0)
    p = box(radii_p)
    small, large = box(radii_w), box(radii_w * factor)
    assert p.erode(small).contains_polytope(p.erode(large), tol=1e-9)


@SETTINGS
@given(_radii(2), _vec(2), _vec(4))
def test_preimage_membership_identity(radii, x, flat):
    p = box(radii)
    m = flat.reshape(2, 2)
    pre = p.preimage(m)
    if abs(p.slack(m @ x)) > 1e-9:
        assert pre.contains(x, tol=1e-12) == p.contains(m @ x, tol=1e-12)


@SETTINGS
@given(_radii(2), _vec(2), positive)
def test_scaling_scales_the_support(radii, c, t):
    p = box(radii)
    scaled = p.scaled(t)
    assert abs(scaled.support(c) - t * p.support(c)) <= 1e-7 * (1.0 + abs(t * p.support(c)))


@SETTINGS
@given(_radii(2))
def test_eroding_by_a_larger_box_empties_the_set(radii):
    p = box(radii)
    eroded = p.erode(box(radii * 2.0 + 0.1))
    assert eroded.is_empty()


@SETTINGS
@given(_radii(2), _vec(2))
def test_minimal_preserves_membership(radii, x):
    rows = np.vstack([np.eye(2), -np.eye(2), np.array([[1.0, 1.0], [1.0, 1.0]])])
    offs = np.concatenate([radii, radii, [radii.sum() + 1.0, radii.sum() + 1.0]])
    p = Polytope(rows, offs)
    reduced = p.minimal()
    if abs(p.slack(x)) > 1e-8:
        assert p.contains(x, tol=1e-12) == reduced.contains(x, tol=1e-12)


@SETTINGS
@given(_radii(2))
def test_box_area_known_formula(radii):
    assert abs(box(radii).area_2d() - 4.0 * radii[0] * radii[1]) <= 1e-9 * (
        1.0 + 4.0 * radii[0] * radii[1]
    )
