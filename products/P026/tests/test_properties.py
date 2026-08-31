"""Hypothesis property tests for the algebraic identities.

The identities exercised here are exact, so the tolerances are round-off
tolerances and not fitted ones:

* rotation invariance -- rotating both frames by an arbitrary rotation moves
  the answer by exactly that rotation;
* orthogonality -- every returned attitude matrix is proper orthogonal;
* quaternion norm -- every returned quaternion is a unit quaternion with a
  non-negative scalar part, and reproduces the returned matrix;
* the Wahba loss is invariant under a relabelling of the observations.
"""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from wahbakit import (
    VectorObservations,
    angle_between_dcm,
    dcm_from_quat,
    is_rotation,
    olae,
    q_method,
    quest,
    triad,
    wahba_loss,
)

SETTINGS = settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.filter_too_much],
)

finite = st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)
quaternion_component = st.floats(
    min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False
)


def _dcm(components):
    q = np.array(components, dtype=float)
    if np.linalg.norm(q) < 1e-3:
        q = np.array([1.0, 0.0, 0.0, 0.0])
    return dcm_from_quat(q)


def _observations(reference_components, attitude_components, sigma):
    reference = np.array(reference_components, dtype=float).reshape(-1, 3)
    norms = np.linalg.norm(reference, axis=1)
    if np.any(norms < 1e-3):
        reference = np.eye(3)
    dcm = _dcm(attitude_components)
    obs = VectorObservations(reference @ dcm.T, reference, sigmas=np.full(len(reference), sigma))
    return obs, dcm


@SETTINGS
@given(
    reference=st.lists(st.lists(finite, min_size=3, max_size=3), min_size=3, max_size=3),
    attitude=st.lists(quaternion_component, min_size=4, max_size=4),
    extra=st.lists(quaternion_component, min_size=4, max_size=4),
    sigma=st.floats(min_value=1e-6, max_value=1e-2),
)
def test_rotation_invariance_of_the_optimal_solution(reference, attitude, extra, sigma):
    """Rotating the body frame by R maps the solution A to R A exactly."""
    obs, _ = _observations(reference, attitude, sigma)
    if obs.observability().lambda_min < 1e-3:
        return
    rotation = _dcm(extra)
    rotated = VectorObservations(obs.body @ rotation.T, obs.reference, sigmas=obs.sigmas)
    if rotated.observability().lambda_min < 1e-3:
        return
    baseline = q_method(obs, check_degeneracy=False).dcm
    rotated_solution = q_method(rotated, check_degeneracy=False).dcm
    assert angle_between_dcm(rotated_solution, rotation @ baseline) < 1e-8


@SETTINGS
@given(
    reference=st.lists(st.lists(finite, min_size=3, max_size=3), min_size=3, max_size=3),
    attitude=st.lists(quaternion_component, min_size=4, max_size=4),
    extra=st.lists(quaternion_component, min_size=4, max_size=4),
    sigma=st.floats(min_value=1e-6, max_value=1e-2),
)
def test_reference_frame_rotation_invariance(reference, attitude, extra, sigma):
    """Rotating the reference frame by R maps the solution A to A R^T."""
    obs, _ = _observations(reference, attitude, sigma)
    if obs.observability().lambda_min < 1e-3:
        return
    rotation = _dcm(extra)
    rotated = VectorObservations(obs.body, obs.reference @ rotation.T, sigmas=obs.sigmas)
    baseline = q_method(obs, check_degeneracy=False).dcm
    assert (
        angle_between_dcm(q_method(rotated, check_degeneracy=False).dcm, baseline @ rotation.T)
        < 1e-8
    )


@SETTINGS
@given(
    reference=st.lists(st.lists(finite, min_size=3, max_size=3), min_size=2, max_size=6),
    attitude=st.lists(quaternion_component, min_size=4, max_size=4),
    sigma=st.floats(min_value=1e-6, max_value=1e-1),
)
def test_every_solution_is_a_proper_rotation_with_a_unit_quaternion(reference, attitude, sigma):
    obs, _ = _observations(reference, attitude, sigma)
    if obs.observability().lambda_min < 1e-3:
        return
    solvers = [q_method, quest, olae] + ([triad] if obs.n == 2 else [])
    for solver in solvers:
        solution = solver(obs, check_degeneracy=False)
        assert is_rotation(solution.dcm)
        assert abs(float(np.linalg.norm(solution.quaternion)) - 1.0) < 1e-12
        assert solution.quaternion[0] >= 0.0
        assert np.allclose(dcm_from_quat(solution.quaternion), solution.dcm, atol=1e-12)


@SETTINGS
@given(
    reference=st.lists(st.lists(finite, min_size=3, max_size=3), min_size=3, max_size=5),
    attitude=st.lists(quaternion_component, min_size=4, max_size=4),
    sigma=st.floats(min_value=1e-6, max_value=1e-1),
)
def test_wahba_loss_is_invariant_under_relabelling(reference, attitude, sigma):
    obs, _ = _observations(reference, attitude, sigma)
    if obs.observability().lambda_min < 1e-3:
        return
    order = np.roll(np.arange(obs.n), 1)
    shuffled = obs.subset(order)
    original = q_method(obs, check_degeneracy=False)
    relabelled = q_method(shuffled, check_degeneracy=False)
    assert abs(original.loss - relabelled.loss) < 1e-12
    assert angle_between_dcm(original.dcm, relabelled.dcm) < 1e-8


@SETTINGS
@given(
    reference=st.lists(st.lists(finite, min_size=3, max_size=3), min_size=2, max_size=5),
    attitude=st.lists(quaternion_component, min_size=4, max_size=4),
)
def test_noise_free_solution_has_zero_loss(reference, attitude):
    obs, dcm = _observations(reference, attitude, 1e-6)
    if obs.observability().lambda_min < 1e-3:
        return
    solution = q_method(obs, check_degeneracy=False)
    assert abs(solution.loss) < 1e-12
    assert abs(wahba_loss(dcm, obs)) < 1e-12
    assert angle_between_dcm(solution.dcm, dcm) < 1e-8
