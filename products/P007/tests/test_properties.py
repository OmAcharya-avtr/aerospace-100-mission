"""Hypothesis property-based tests of algebraic quaternion identities."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from quatkit import (
    angle_between,
    dcm_to_quat,
    euler_zyx_to_quat,
    mrp_to_quat,
    quat_conjugate,
    quat_exp,
    quat_identity,
    quat_log,
    quat_multiply,
    quat_normalize,
    quat_rotate,
    quat_slerp,
    quat_to_dcm,
    quat_to_euler_zyx,
    quat_to_mrp,
    rodrigues_to_quat,
)
from quatkit.conversions import quat_to_rodrigues

finite = st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)


@st.composite
def unit_quaternions(draw):
    """Random unit quaternions with a well-conditioned pre-normalization norm."""
    q = np.array([draw(finite), draw(finite), draw(finite), draw(finite)])
    n = np.linalg.norm(q)
    if n < 1e-3:
        q = np.array([1.0, 0.0, 0.0, 0.0])
    return quat_normalize(q) if np.linalg.norm(q) > 0 else np.array([1.0, 0, 0, 0])


@st.composite
def vectors(draw):
    return np.array([draw(finite), draw(finite), draw(finite)])


@settings(max_examples=200, deadline=None)
@given(unit_quaternions(), unit_quaternions())
def test_product_of_units_is_unit(q1, q2):
    """|q1 ⊗ q2| = 1: the unit 3-sphere is closed under the Hamilton product."""
    assert abs(np.linalg.norm(quat_multiply(q1, q2)) - 1.0) < 1e-12


@settings(max_examples=200, deadline=None)
@given(unit_quaternions())
def test_q_times_inverse_is_identity(q):
    """q ⊗ q⁻¹ = [1, 0, 0, 0] (conjugate = inverse for unit q)."""
    prod = quat_multiply(q, quat_conjugate(q))
    assert np.max(np.abs(prod - quat_identity())) < 1e-12


@settings(max_examples=200, deadline=None)
@given(unit_quaternions(), vectors())
def test_rotation_preserves_norm(q, v):
    """|q v q*| = |v| for unit q."""
    assert abs(np.linalg.norm(quat_rotate(q, v)) - np.linalg.norm(v)) < 1e-10


@settings(max_examples=200, deadline=None)
@given(unit_quaternions(), vectors(), vectors())
def test_rotation_is_linear_and_preserves_dot(q, u, v):
    """Rotations preserve inner products (orthogonal transformation)."""
    du = quat_rotate(q, u) @ quat_rotate(q, v)
    assert abs(du - u @ v) < 1e-9


@settings(max_examples=200, deadline=None)
@given(unit_quaternions())
def test_dcm_orthogonality(q):
    """R(q)ᵀ R(q) = I and det R = +1 (proper orthogonal)."""
    r = quat_to_dcm(q)
    assert np.max(np.abs(r.T @ r - np.eye(3))) < 1e-12
    assert abs(np.linalg.det(r) - 1.0) < 1e-12


@settings(max_examples=200, deadline=None)
@given(unit_quaternions())
def test_dcm_roundtrip(q):
    """q -> DCM -> q recovers the same rotation (double cover resolved)."""
    assert float(angle_between(dcm_to_quat(quat_to_dcm(q)), q)) < 1e-10


@settings(max_examples=200, deadline=None)
@given(unit_quaternions())
def test_euler_roundtrip(q):
    """q -> ZYX Euler -> q recovers the rotation away from the gimbal margin."""
    w, x, y, z = q
    if abs(2.0 * (w * y - x * z)) > 0.99:  # skip near-singular pitch
        return
    y_, p_, r_ = quat_to_euler_zyx(q)
    assert float(angle_between(euler_zyx_to_quat(y_, p_, r_), q)) < 1e-9


@settings(max_examples=200, deadline=None)
@given(unit_quaternions())
def test_mrp_roundtrip(q):
    """q -> MRP -> q recovers the rotation (principal set)."""
    assert float(angle_between(mrp_to_quat(quat_to_mrp(q)), q)) < 1e-10


@settings(max_examples=200, deadline=None)
@given(unit_quaternions())
def test_rodrigues_roundtrip(q):
    """q -> Gibbs -> q away from the 180° singularity."""
    if abs(q[0]) < 1e-2:
        return
    assert float(angle_between(rodrigues_to_quat(quat_to_rodrigues(q)), q)) < 1e-9


@settings(max_examples=200, deadline=None)
@given(vectors())
def test_exp_log_roundtrip(rv):
    """log(exp(rv)) = rv for |rv| < π (principal branch)."""
    n = np.linalg.norm(rv)
    if n >= np.pi - 1e-3:
        rv = rv * ((np.pi - 1e-3) / n) * 0.9
    assert np.max(np.abs(quat_log(quat_exp(rv)) - rv)) < 1e-9


@settings(max_examples=200, deadline=None)
@given(unit_quaternions())
def test_exp_produces_unit(rv_q):
    """exp of any rotation vector is a unit quaternion (use q's vec scaled)."""
    rv = rv_q[1:] * 3.0
    assert abs(np.linalg.norm(quat_exp(rv)) - 1.0) < 1e-12


@settings(max_examples=100, deadline=None)
@given(unit_quaternions(), unit_quaternions(), st.floats(min_value=0.0, max_value=1.0))
def test_slerp_output_is_unit_and_bounded(q0, q1, t):
    """SLERP output is unit, and its angle from q0 never exceeds the q0-q1 angle."""
    qt = quat_slerp(q0, q1, t)
    assert abs(np.linalg.norm(qt) - 1.0) < 1e-10
    total = float(angle_between(q1, q0))
    partial = float(angle_between(qt, q0))
    assert partial <= total + 1e-7
