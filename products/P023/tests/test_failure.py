"""Failure reallocation: meets the command when it can, says so when it cannot."""

import numpy as np
import pytest

from alloclab.allocation import DEFAULT_TORQUE_TOL, InfeasibleAllocationError
from alloclab.ams import attainable_moment_set
from alloclab.dataset import reference_thruster_cluster
from alloclab.effectors import (
    orthogonal_effectors,
    pyramid_reaction_wheels,
    thruster_cluster,
)
from alloclab.failure import failure_margin, reallocate_after_failure


@pytest.fixture
def cluster():
    return reference_thruster_cluster(1.0, 0.5)


# --------------------------------------------------------------------------
# Attainable after failure -> must be met exactly
# --------------------------------------------------------------------------


def test_small_command_still_met_after_one_thruster_fails(cluster):
    report = reallocate_after_failure(cluster, [0.05, 0.02, -0.03], failed=[0], method="qp")
    assert report.attainable
    assert report.degraded.feasible
    assert report.degraded.residual_norm <= DEFAULT_TORQUE_TOL
    assert report.degraded.bound_violation == pytest.approx(0.0, abs=1e-9)
    assert report.remaining_rank == 3


@pytest.mark.parametrize("method", ["qp", "lp"])
def test_command_inside_the_degraded_ams_is_met(cluster, method):
    degraded = cluster.with_failures([2, 5])
    ams = attainable_moment_set(degraded)
    direction = np.array([0.4, -0.7, 0.5])
    direction /= np.linalg.norm(direction)
    tau = 0.6 * ams.boundary_scale(direction) * direction
    report = reallocate_after_failure(cluster, tau, failed=[2, 5], method=method)
    assert report.attainable
    assert report.degraded.feasible, report.degraded.message
    assert report.degraded.residual_norm <= DEFAULT_TORQUE_TOL


def test_wheel_array_survives_one_wheel_loss():
    e = pyramid_reaction_wheels(0.1)
    report = reallocate_after_failure(e, [0.01, 0.005, -0.02], failed=[1], method="qp")
    assert report.remaining_rank == 3
    assert report.attainable
    assert report.degraded.feasible


# --------------------------------------------------------------------------
# Not attainable after failure -> must be reported, not clipped
# --------------------------------------------------------------------------


def test_command_outside_the_degraded_ams_is_reported_infeasible(cluster):
    degraded = cluster.with_failures([0])
    ams = attainable_moment_set(degraded)
    direction = np.array([1.0, 0.0, 0.0])
    tau = 1.4 * ams.boundary_scale(direction) * direction
    report = reallocate_after_failure(cluster, tau, failed=[0], method="qp")
    assert not report.attainable
    assert report.degraded.status == "infeasible"
    assert not report.degraded.feasible
    assert report.residual_norm > 1e-6
    assert "NOT attainable" in report.degraded.message
    # It is still a legal command, not a bound-violating one.
    assert report.degraded.bound_violation == pytest.approx(0.0, abs=1e-9)


def test_losing_a_body_axis_drops_the_rank_and_is_reported():
    # The triad has no redundancy at all: failing the z effector removes all
    # authority about z.
    e = orthogonal_effectors(1.0)
    report = reallocate_after_failure(e, [0.2, 0.1, 0.3], failed=[2], method="qp")
    assert report.remaining_rank == 2
    assert not report.attainable
    assert report.residual_norm == pytest.approx(0.3, abs=1e-6)
    assert report.volume_ratio == pytest.approx(0.0, abs=1e-12)


def test_in_plane_command_is_still_met_after_the_z_effector_fails():
    e = orthogonal_effectors(1.0)
    report = reallocate_after_failure(e, [0.2, 0.1, 0.0], failed=[2], method="qp")
    assert report.attainable
    assert report.degraded.feasible
    assert report.remaining_rank == 2


def test_require_feasible_raises_with_an_actionable_message(cluster):
    with pytest.raises(InfeasibleAllocationError) as exc:
        reallocate_after_failure(
            cluster, [5.0, 0.0, 0.0], failed=[0, 1], method="qp", require_feasible=True
        )
    text = str(exc.value)
    assert "cannot be allocated after failure of effectors [0, 1]" in text
    assert "remaining rank" in text


def test_require_feasible_does_not_raise_when_the_command_is_met(cluster):
    report = reallocate_after_failure(
        cluster, [0.02, 0.01, 0.0], failed=[0], method="qp", require_feasible=True
    )
    assert report.degraded.feasible


# --------------------------------------------------------------------------
# The nominal-versus-degraded bookkeeping
# --------------------------------------------------------------------------


def test_report_carries_both_allocations_and_the_volume_ratio(cluster):
    report = reallocate_after_failure(cluster, [0.1, 0.0, 0.0], failed=[3], method="qp")
    assert report.nominal.feasible
    assert report.failed == (3,)
    assert 0.0 < report.volume_ratio < 1.0
    assert "failed=[3]" in str(report)


def test_volume_can_be_skipped(cluster):
    report = reallocate_after_failure(
        cluster, [0.1, 0.0, 0.0], failed=[3], method="qp", compute_volume=False
    )
    assert report.volume_ratio is None


def test_stuck_open_thruster_must_be_cancelled_by_the_others():
    # Two thrusters on the same arm firing opposite ways: columns +1 x and -1 x
    # (r = (0,1,0) m, F = +/-z). With t1 stuck at 0.6 N the body sees
    # +0.6 N*m about x, and the only way to hold a zero command is t2 = 0.6 N.
    e = thruster_cluster(
        [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
        max_thrust=1.0,
    )
    report = reallocate_after_failure(
        e, np.zeros(3), failed=[0], stuck_at=0.6, method="qp"
    )
    assert report.attainable
    assert report.degraded.feasible
    assert report.degraded.commands[0] == pytest.approx(0.6)
    assert report.degraded.commands[1] == pytest.approx(0.6, abs=1e-6)
    assert np.allclose(report.degraded.achieved_torque, 0.0, atol=1e-9)


def test_stuck_open_thruster_on_the_reference_cluster_still_holds_zero(cluster):
    # Same idea with redundancy present: the QP is free to spread the
    # cancellation, so only the achieved torque is pinned down.
    report = reallocate_after_failure(
        cluster, np.zeros(3), failed=[0], stuck_at=1.0, method="qp"
    )
    assert report.attainable
    assert report.degraded.feasible
    assert report.degraded.commands[0] == pytest.approx(1.0)
    assert np.allclose(report.degraded.achieved_torque, 0.0, atol=1e-8)
    assert report.degraded.bound_violation == pytest.approx(0.0, abs=1e-9)


def test_pinv_failure_is_distinguished_from_true_infeasibility(cluster):
    # The pseudo-inverse ignores bounds, so it can return an out-of-box
    # command for a torque the degraded set can genuinely produce. The report
    # must say the command WAS attainable.
    degraded = cluster.with_failures([0])
    ams = attainable_moment_set(degraded)
    direction = np.array([-1.0, 0.2, 0.1])
    direction /= np.linalg.norm(direction)
    tau = 0.95 * ams.boundary_scale(direction) * direction
    report = reallocate_after_failure(cluster, tau, failed=[0], method="pinv")
    assert report.attainable
    assert not report.degraded.feasible
    assert "IS attainable" in report.degraded.message


# --------------------------------------------------------------------------
# failure_margin
# --------------------------------------------------------------------------


def test_failure_margin_above_one_for_a_command_still_inside(cluster):
    assert failure_margin(cluster, [0.05, 0.0, 0.0], failed=[0]) > 1.0


def test_failure_margin_below_one_for_a_command_outside(cluster):
    degraded = cluster.with_failures([0])
    scale = attainable_moment_set(degraded).boundary_scale([1.0, 0.0, 0.0])
    assert failure_margin(cluster, [1.5 * scale, 0.0, 0.0], failed=[0]) < 1.0


def test_failure_margin_of_zero_command_is_infinite(cluster):
    assert failure_margin(cluster, np.zeros(3), failed=[0]) == float("inf")


def test_failure_margin_rejects_a_bad_torque_shape(cluster):
    with pytest.raises(ValueError, match=r"torque must have shape \(3,\)"):
        failure_margin(cluster, [1.0, 2.0], failed=[0])


def test_reallocate_rejects_a_bad_torque_shape(cluster):
    with pytest.raises(ValueError, match=r"torque must have shape \(3,\)"):
        reallocate_after_failure(cluster, [1.0, 2.0], failed=[0])
