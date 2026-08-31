"""Hypothesis property tests over random commands and random configurations.

The load-bearing property is the last one in the file: **every allocation
produced by a bounds-aware method respects the actuator bounds**, for any
effector geometry and any commanded torque, attainable or not.
"""

import numpy as np
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from alloclab.allocation import (
    DEFAULT_TORQUE_TOL,
    allocate,
    is_attainable,
    lp_allocate,
    pseudo_inverse_allocate,
    qp_allocate,
    weighted_pseudo_inverse_allocate,
)
from alloclab.ams import attainable_moment_set, zonotope_volume
from alloclab.effectors import general_effector_set

_SETTINGS = settings(
    max_examples=120,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

finite = st.floats(min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False)
positive = st.floats(min_value=0.05, max_value=3.0, allow_nan=False, allow_infinity=False)


@st.composite
def effector_sets(draw, min_m: int = 3, max_m: int = 8, min_pair_sin: float = 0.0):
    """Random full-rank effector sets with random, possibly one-sided, bounds."""
    m = draw(st.integers(min_value=min_m, max_value=max_m))
    entries = draw(st.lists(finite, min_size=3 * m, max_size=3 * m))
    matrix = np.asarray(entries, dtype=float).reshape(3, m)
    assume(np.linalg.matrix_rank(matrix) == 3)
    assume(np.min(np.linalg.svd(matrix, compute_uv=False)) > 1e-3)
    # Exclude effectors with almost no control effectiveness. A column of norm
    # ~1e-4 is an actuator that barely moves the vehicle, and the weighted-
    # least-squares QP will happily push its command a long way for a
    # negligible torque cost -- see
    # test_weak_effector_degrades_the_qp_residual in tests/test_allocation.py,
    # where that behaviour is pinned down as a documented limitation rather
    # than filtered away.
    assume(np.min(np.linalg.norm(matrix, axis=0)) > 1e-2)
    if min_pair_sin > 0.0:
        # Keep generator directions well separated. The pairwise AMS
        # construction classifies a column as lying in a facet plane with a
        # relative tolerance, so two columns within ~1e-8 of parallel put that
        # classification on a knife edge; see README "Limitations".
        unit = matrix / np.linalg.norm(matrix, axis=0)
        for a in range(unit.shape[1]):
            for b in range(a + 1, unit.shape[1]):
                assume(np.linalg.norm(np.cross(unit[:, a], unit[:, b])) > min_pair_sin)
    lower = np.asarray(draw(st.lists(finite, min_size=m, max_size=m)), dtype=float)
    width = np.asarray(draw(st.lists(positive, min_size=m, max_size=m)), dtype=float)
    return general_effector_set(matrix, lower, lower + width)


@st.composite
def torques(draw, scale: float = 5.0):
    comps = draw(
        st.lists(
            st.floats(
                min_value=-scale, max_value=scale, allow_nan=False, allow_infinity=False
            ),
            min_size=3,
            max_size=3,
        )
    )
    return np.asarray(comps, dtype=float)


# --------------------------------------------------------------------------
# Algebraic identities
# --------------------------------------------------------------------------


@_SETTINGS
@given(eset=effector_sets(), tau=torques())
def test_pseudo_inverse_reproduces_the_torque_when_bounds_are_ignored(eset, tau):
    """B B^+ tau = tau for any full-row-rank B, regardless of the bounds."""
    res = pseudo_inverse_allocate(eset, tau)
    scale = max(1.0, float(np.linalg.norm(tau)))
    assert res.residual_norm <= 1e-8 * scale


@_SETTINGS
@given(eset=effector_sets(), tau=torques())
def test_weighted_pseudo_inverse_also_reproduces_the_torque(eset, tau):
    res = weighted_pseudo_inverse_allocate(eset, tau)
    scale = max(1.0, float(np.linalg.norm(tau)))
    assert res.residual_norm <= 1e-8 * scale


@_SETTINGS
@given(eset=effector_sets(max_m=6))
def test_zonotope_volume_matches_the_hull(eset):
    """The closed-form zonotope volume equals the computed hull volume."""
    ams = attainable_moment_set(eset)
    assume(not ams.degenerate)
    closed = zonotope_volume(eset)
    assert abs(closed - ams.volume) <= 1e-7 * max(1.0, closed)


@_SETTINGS
@given(eset=effector_sets(max_m=6, min_pair_sin=1e-3))
def test_pairwise_and_bruteforce_hulls_agree(eset):
    """Durham's pairwise construction and full box enumeration give one set.

    Compared by volume, area, and mutual containment: every vertex of each
    hull must lie inside the other. Mutual containment rather than a
    vertex-to-vertex distance, because on a near-degenerate zonotope a point
    can be a vertex of one hull and interior to the other by 1e-9 while the
    two bodies are the same to within that; a genuinely missing facet corner,
    such as the one pinned in
    ``tests/test_ams.py::test_coplanar_generators_do_not_lose_facet_vertices``,
    still shows up as a containment violation of order the vertex offset. The
    named configurations in tests/test_ams.py assert the exact vertex counts.
    """
    a = attainable_moment_set(eset, method="pairwise")
    b = attainable_moment_set(eset, method="bruteforce")
    assume(not a.degenerate and not b.degenerate)
    assert abs(a.volume - b.volume) <= 1e-7 * max(1.0, b.volume)
    assert abs(a.area - b.area) <= 1e-6 * max(1.0, b.area)
    scale = max(1.0, float(np.max(np.abs(b.vertices))))
    for src, dst in ((a, b), (b, a)):
        eq = dst.hull.equations
        outside = float(np.max(src.vertices @ eq[:, :3].T + eq[:, 3]))
        assert outside <= 1e-6 * scale


@_SETTINGS
@given(eset=effector_sets(max_m=6), tau=torques())
def test_ams_membership_agrees_with_the_lp_feasibility_test(eset, tau):
    """Hull half-space membership and the LP certificate must agree.

    Membership is decided two independent ways: the convex-hull facet
    inequalities, and the exact linear program. Commands within 1e-6 of the
    boundary are skipped, where the two disagree only at their own solver
    tolerances.
    """
    ams = attainable_moment_set(eset)
    assume(not ams.degenerate)
    scale = float(np.max(np.linalg.norm(ams.vertices, axis=1)))
    eq = ams.hull.equations
    margin = float(np.max(tau @ eq[:, :3].T + eq[:, 3]))
    assume(abs(margin) > 1e-6 * max(1.0, scale))
    assert (margin <= 0.0) == is_attainable(eset, tau, tol=1e-7 * max(1.0, scale))


# --------------------------------------------------------------------------
# The bound-respecting property
# --------------------------------------------------------------------------


@_SETTINGS
@given(eset=effector_sets(), tau=torques())
def test_qp_allocation_always_respects_actuator_bounds(eset, tau):
    res = qp_allocate(eset, tau)
    assert res.bound_violation <= 1e-9
    assert np.all(res.commands >= eset.lower - 1e-9)
    assert np.all(res.commands <= eset.upper + 1e-9)


@_SETTINGS
@given(eset=effector_sets(), tau=torques())
def test_lp_allocation_always_respects_actuator_bounds(eset, tau):
    res = lp_allocate(eset, tau, objective="min_error")
    assert res.bound_violation <= 1e-9


@_SETTINGS
@given(eset=effector_sets(), tau=torques())
def test_redistributed_pseudo_inverse_always_respects_actuator_bounds(eset, tau):
    res = allocate(eset, tau, method="rpi")
    assert res.bound_violation <= 1e-9


@_SETTINGS
@given(
    eset=effector_sets(max_m=6),
    fractions=st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=8,
        max_size=8,
    ),
)
def test_allocation_reproduces_any_attainable_command_exactly(eset, fractions):
    """The headline validation property.

    Any point of the command box maps to a torque that is by construction
    attainable, so both bounds-aware allocators must reproduce it exactly and
    stay inside the box -- whichever alternative command they choose to do it
    with.

    The QP is given ``u_pref = u_true`` here. Its objective is a *penalty*, not
    a hard constraint, so with an arbitrary effort preference it trades a small
    torque error against effort along whatever weakly-effective direction the
    active set leaves open; that trade is a real property of weighted least
    squares, is measured in
    ``tests/test_allocation.py::test_weak_effector_degrades_the_qp_residual``
    and is written up in README "Limitations". This test is about exactness,
    so the trade is removed rather than hidden.
    """
    m = eset.n_effectors
    frac = np.asarray(fractions[:m], dtype=float)
    u_true = eset.lower + frac * eset.span
    target = eset.matrix @ u_true
    scale = max(1.0, float(np.linalg.norm(target)))
    # The HiGHS primal feasibility tolerance is 1e-7 by default, and a command
    # that far outside its own box turns into a torque error of up to
    # 1e-7 * ||B||, so that is the LP's residual floor whatever torque_tol
    # says; see README "Limitations". The QP has no such floor.
    lp_tol = max(3e-7 * max(1.0, float(np.linalg.norm(eset.matrix))), DEFAULT_TORQUE_TOL * scale)
    res = qp_allocate(eset, target, u_pref=u_true)
    assert res.residual_norm <= DEFAULT_TORQUE_TOL * scale
    assert res.bound_violation <= 1e-9
    assert res.feasible
    res = lp_allocate(eset, target, torque_tol=lp_tol)
    assert res.residual_norm <= lp_tol
    assert res.bound_violation <= 1e-9
    assert res.feasible


@_SETTINGS
@given(eset=effector_sets(max_m=6), tau=torques())
def test_no_allocator_beats_the_lp_on_torque_error(eset, tau):
    """The bounds-aware LP optimum lower-bounds what any in-box command achieves.

    Compared in the 1-norm, which is what the LP minimises.
    """
    best = lp_allocate(eset, tau, objective="min_error")
    for method in ("qp", "rpi"):
        other = allocate(eset, tau, method=method)
        assert np.sum(np.abs(best.residual)) <= np.sum(np.abs(other.residual)) + 1e-6


# --------------------------------------------------------------------------
# Failure invariants
# --------------------------------------------------------------------------


@_SETTINGS
@given(eset=effector_sets(min_m=4, max_m=6), idx=st.integers(min_value=0, max_value=3))
def test_failing_an_effector_never_grows_the_attainable_set(eset, idx):
    assume(idx < eset.n_effectors)
    full = attainable_moment_set(eset)
    assume(not full.degenerate)
    part = attainable_moment_set(eset.with_failures([idx]))
    assert part.volume <= full.volume + 1e-9 * max(1.0, full.volume)
