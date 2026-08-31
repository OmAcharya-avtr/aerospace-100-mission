"""Unit, input-validation and known-answer tests for effector configuration."""

import numpy as np
import pytest

from alloclab.effectors import (
    EffectorSet,
    general_effector_set,
    orthogonal_effectors,
    pyramid_reaction_wheels,
    reaction_wheel_array,
    thruster_cluster,
)


# --------------------------------------------------------------------------
# Known-answer tests (hand-calculated)
# --------------------------------------------------------------------------


def test_single_thruster_torque_column_hand_calculation():
    # r = (0, 2, 0) m, thrust direction on the body = +z.
    # r x F_hat = (0,2,0) x (0,0,1) = (2*1 - 0*0, 0*0 - 0*1, 0*0 - 2*0) = (2, 0, 0).
    # So 3 N of thrust produces 6 N*m about +x.
    e = thruster_cluster([[0.0, 2.0, 0.0]], [[0.0, 0.0, 1.0]], max_thrust=10.0)
    assert np.allclose(e.matrix[:, 0], [2.0, 0.0, 0.0])
    assert np.allclose(e.torque([3.0]), [6.0, 0.0, 0.0])


def test_thruster_direction_is_normalised():
    # Direction (0, 0, 5) must give the same column as (0, 0, 1).
    a = thruster_cluster([[1.0, 0.0, 0.0]], [[0.0, 0.0, 5.0]], 1.0)
    b = thruster_cluster([[1.0, 0.0, 0.0]], [[0.0, 0.0, 1.0]], 1.0)
    assert np.allclose(a.matrix, b.matrix)


def test_thruster_through_origin_has_no_torque_authority():
    # A thrust line passing through the body origin produces zero moment:
    # r x F = 0 when r is parallel to F.
    e = thruster_cluster([[0.0, 0.0, 1.0]], [[0.0, 0.0, 1.0]], 1.0)
    assert np.allclose(e.matrix, 0.0)
    assert e.rank == 0


def test_reaction_wheel_column_is_minus_spin_axis():
    # A +0.05 N*m motor torque on a wheel spinning about +z reacts as
    # -0.05 N*m about +z on the body (eq. 3).
    e = reaction_wheel_array([[0.0, 0.0, 1.0]], 0.1)
    assert np.allclose(e.matrix[:, 0], [0.0, 0.0, -1.0])
    assert np.allclose(e.torque([0.05]), [0.0, 0.0, -0.05])
    assert np.allclose(e.lower, [-0.1])
    assert np.allclose(e.upper, [0.1])


def test_pyramid_default_half_angle_is_isotropic():
    # At beta = arctan(sqrt(2)) = 54.7356 deg the four-wheel pyramid satisfies
    # B B^T = 2 sin^2(beta) (xx^T + yy^T) + 4 cos^2(beta) zz^T
    #       = 2*(2/3) I_xy + 4*(1/3) I_z = (4/3) I_3.
    e = pyramid_reaction_wheels(max_torque=0.2)
    assert e.n_effectors == 4
    assert np.allclose(e.matrix @ e.matrix.T, (4.0 / 3.0) * np.eye(3), atol=1e-12)
    assert e.rank == 3


def test_pyramid_half_angle_off_default_is_not_isotropic():
    e = pyramid_reaction_wheels(max_torque=0.2, half_angle_deg=30.0)
    gram = e.matrix @ e.matrix.T
    assert not np.allclose(gram, gram[0, 0] * np.eye(3), atol=1e-6)


def test_orthogonal_triad_is_identity():
    e = orthogonal_effectors(2.5)
    assert np.allclose(e.matrix, np.eye(3))
    assert np.allclose(e.lower, -2.5)
    assert np.allclose(e.upper, 2.5)
    assert e.names == ("x", "y", "z")


# --------------------------------------------------------------------------
# Unit behaviour
# --------------------------------------------------------------------------


def test_span_rank_and_free_mask():
    e = orthogonal_effectors(1.0)
    assert np.allclose(e.span, 2.0)
    assert e.rank == 3
    assert e.free_mask().all()


def test_batch_torque_and_bound_violation():
    e = orthogonal_effectors(1.0)
    u = np.array([[0.5, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, -3.0, 0.0]])
    assert np.allclose(e.torque(u), u)
    assert np.allclose(e.bound_violation(u), [0.0, 1.0, 2.0])
    assert np.array_equal(e.within_bounds(u), [True, False, False])


def test_clip_projects_into_box():
    e = orthogonal_effectors(1.0)
    assert np.allclose(e.clip([5.0, -5.0, 0.2]), [1.0, -1.0, 0.2])


def test_with_failures_pins_bounds_and_drops_rank():
    e = orthogonal_effectors(1.0)
    d = e.with_failures([2])
    assert d.lower[2] == 0.0 and d.upper[2] == 0.0
    assert d.rank == 2
    assert np.allclose(d.span[:2], 2.0)
    # The original set is untouched.
    assert e.rank == 3


def test_with_failures_stuck_open_keeps_torque_bias():
    # A thruster stuck at 0.8 N still applies its torque; the degraded set's
    # command box is a point in that coordinate.
    e = thruster_cluster(
        [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
        max_thrust=1.0,
    )
    d = e.with_failures([0], stuck_at=0.8)
    assert d.lower[0] == pytest.approx(0.8)
    assert d.upper[0] == pytest.approx(0.8)
    assert np.allclose(d.torque([0.8, 0.0]), [0.8, 0.0, 0.0])


def test_health_mask():
    e = orthogonal_effectors(1.0).with_failures([0, 2])
    assert np.allclose(e.health(), [0.0, 1.0, 0.0])


def test_summary_mentions_every_effector():
    e = pyramid_reaction_wheels(0.1)
    text = e.summary()
    for name in e.names:
        assert name in text
    assert "rank 3" in text


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------


def test_matrix_must_have_three_rows():
    with pytest.raises(ValueError, match="3 rows"):
        general_effector_set(np.ones((2, 4)), np.zeros(4), np.ones(4))


def test_matrix_must_be_two_dimensional():
    with pytest.raises(ValueError, match="2-D"):
        general_effector_set(np.ones(3), [0.0], [1.0])


def test_matrix_must_be_finite():
    bad = np.eye(3).copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        general_effector_set(bad, -np.ones(3), np.ones(3))


def test_bounds_shape_is_checked():
    with pytest.raises(ValueError, match=r"lower must have shape \(3,\)"):
        general_effector_set(np.eye(3), [0.0, 0.0], [1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match=r"upper must have shape \(3,\)"):
        general_effector_set(np.eye(3), [0.0, 0.0, 0.0], [1.0, 1.0])


def test_upper_below_lower_is_rejected():
    with pytest.raises(ValueError, match="upper bound must be >= lower bound"):
        general_effector_set(np.eye(3), [0.0, 1.0, 0.0], [1.0, 0.0, 1.0])


def test_infinite_bounds_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        general_effector_set(np.eye(3), [-np.inf, 0.0, 0.0], [1.0, 1.0, 1.0])


def test_names_length_is_checked():
    with pytest.raises(ValueError, match="names must have 3 entries"):
        general_effector_set(np.eye(3), -np.ones(3), np.ones(3), names=("a", "b"))


def test_torque_rejects_wrong_command_length():
    e = orthogonal_effectors(1.0)
    with pytest.raises(ValueError, match="last dimension 3"):
        e.torque([1.0, 2.0])
    with pytest.raises(ValueError, match="last dimension 3"):
        e.bound_violation([1.0, 2.0])


def test_thruster_shape_validation():
    with pytest.raises(ValueError, match=r"positions must be \(m, 3\)"):
        thruster_cluster(np.zeros((2, 2)), np.zeros((2, 2)), 1.0)
    with pytest.raises(ValueError, match="directions must match"):
        thruster_cluster(np.zeros((2, 3)), np.zeros((3, 3)), 1.0)


def test_thruster_zero_direction_is_rejected():
    with pytest.raises(ValueError, match="non-zero vector"):
        thruster_cluster([[1.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]], 1.0)


def test_thruster_negative_min_thrust_is_rejected():
    with pytest.raises(ValueError, match="cannot pull"):
        thruster_cluster([[1.0, 0.0, 0.0]], [[0.0, 0.0, 1.0]], 1.0, min_thrust=-0.5)


def test_thruster_non_positive_max_thrust_is_rejected():
    with pytest.raises(ValueError, match="max_thrust must be > 0"):
        thruster_cluster([[1.0, 0.0, 0.0]], [[0.0, 0.0, 1.0]], 0.0)


def test_wheel_validation():
    with pytest.raises(ValueError, match=r"spin_axes must be \(m, 3\)"):
        reaction_wheel_array(np.zeros((2, 4)), 0.1)
    with pytest.raises(ValueError, match="non-zero vector"):
        reaction_wheel_array([[0.0, 0.0, 0.0]], 0.1)
    with pytest.raises(ValueError, match="max_torque must be > 0"):
        reaction_wheel_array([[0.0, 0.0, 1.0]], -0.1)


def test_pyramid_validation():
    with pytest.raises(ValueError, match="n_wheels must be >= 3"):
        pyramid_reaction_wheels(0.1, n_wheels=2)
    with pytest.raises(ValueError, match=r"half_angle_deg must be in \(0, 90\]"):
        pyramid_reaction_wheels(0.1, half_angle_deg=120.0)


def test_orthogonal_validation():
    with pytest.raises(ValueError, match="max_torque must be > 0"):
        orthogonal_effectors(0.0)


def test_with_failures_index_range_is_checked():
    e = orthogonal_effectors(1.0)
    with pytest.raises(ValueError, match=r"failed indices must be in \[0, 2\]"):
        e.with_failures([5])


def test_stuck_at_outside_bounds_is_rejected():
    e = orthogonal_effectors(1.0)
    with pytest.raises(ValueError, match="inside the nominal command bounds"):
        e.with_failures([0], stuck_at=9.0)


# --------------------------------------------------------------------------
# Edge cases
# --------------------------------------------------------------------------


def test_all_effectors_failed_gives_zero_rank():
    e = orthogonal_effectors(1.0).with_failures([0, 1, 2])
    assert e.rank == 0
    assert not e.free_mask().any()


def test_zero_width_bounds_are_allowed_at_construction():
    e = EffectorSet(np.eye(3), np.zeros(3), np.zeros(3))
    assert e.rank == 0
    assert np.allclose(e.span, 0.0)


def test_single_effector_set_is_valid():
    e = general_effector_set([[1.0], [0.0], [0.0]], [0.0], [1.0])
    assert e.n_effectors == 1
    assert e.rank == 1
