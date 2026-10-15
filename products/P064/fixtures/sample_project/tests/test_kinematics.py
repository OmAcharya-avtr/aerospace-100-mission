import pytest

from sampleproj import kinematics, vecops


def test_propagate_keeps_unit_norm():
    q = kinematics.propagate([1.0, 0.0, 0.0, 0.0], [0.01, 0.0, 0.0], 0.1)
    assert vecops.norm(q) == pytest.approx(1.0)


def test_propagate_zero_rate_is_identity():
    q0 = [1.0, 0.0, 0.0, 0.0]
    assert kinematics.propagate(q0, [0.0, 0.0, 0.0], 0.5) == pytest.approx(q0)


def test_propagate_small_angle_about_x():
    # q_dot = 0.5*q(x)[0,w]; for q0 = identity, w = [0.2,0,0], dt = 0.1
    # the x component before normalisation is 0.5*0.1*0.2 = 0.01
    q = kinematics.propagate([1.0, 0.0, 0.0, 0.0], [0.2, 0.0, 0.0], 0.1)
    assert q[1] == pytest.approx(0.01 / (1.0 + 0.01**2) ** 0.5, rel=1e-9)


def test_angular_momentum_known_answer():
    assert kinematics.angular_momentum([2.0, 3.0, 4.0], [1.0, 1.0, 1.0]) == pytest.approx(
        [2.0, 3.0, 4.0]
    )
