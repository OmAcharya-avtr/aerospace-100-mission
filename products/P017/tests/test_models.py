"""Known-answer and validation tests for the kinematic models."""

from __future__ import annotations

import numpy as np
import pytest

from estimkit import constant_velocity_cwna, constant_velocity_dwna, random_walk


def test_cwna_known_answer_unit_step():
    # T = 1 s, q~ = 1 m^2/s^3:
    #   F = [[1, 1], [0, 1]]
    #   Q = [[1/3, 1/2], [1/2, 1]]
    f, q = constant_velocity_cwna(1.0, 1.0)
    assert f == pytest.approx(np.array([[1.0, 1.0], [0.0, 1.0]]))
    assert q == pytest.approx(np.array([[1.0 / 3.0, 0.5], [0.5, 1.0]]))


def test_cwna_known_answer_half_second():
    # T = 0.5 s, q~ = 2 m^2/s^3:
    #   T^3/3 = 0.125/3 = 0.0416666...,  T^2/2 = 0.125,  T = 0.5
    #   Q = 2 * [[0.0416666..., 0.125], [0.125, 0.5]]
    _, q = constant_velocity_cwna(0.5, 2.0)
    expected = 2.0 * np.array([[0.125 / 3.0, 0.125], [0.125, 0.5]])
    assert q == pytest.approx(expected)


def test_dwna_known_answer_unit_step():
    # T = 1 s, sigma_a = 1 m/s^2:  Gamma = [0.5, 1]^T
    #   Q = [[0.25, 0.5], [0.5, 1]]
    f, q = constant_velocity_dwna(1.0, 1.0)
    assert f == pytest.approx(np.array([[1.0, 1.0], [0.0, 1.0]]))
    assert q == pytest.approx(np.array([[0.25, 0.5], [0.5, 1.0]]))


def test_dwna_process_noise_is_rank_one():
    _, q = constant_velocity_dwna(2.0, 0.3)
    assert np.linalg.matrix_rank(q, tol=1e-12) == 1
    assert np.linalg.eigvalsh(q)[0] >= -1e-12


def test_cwna_process_noise_is_positive_definite():
    _, q = constant_velocity_cwna(1.5, 0.4)
    # det Q = q~^2 (T^4/3 - T^4/4) = q~^2 T^4 / 12 > 0
    assert np.linalg.det(q) == pytest.approx(0.4**2 * 1.5**4 / 12.0)
    assert np.linalg.eigvalsh(q)[0] > 0.0


def test_random_walk_shapes_and_values():
    f, h, q, r = random_walk(0.5, 2.0)
    assert f == pytest.approx(np.array([[1.0]]))
    assert h == pytest.approx(np.array([[1.0]]))
    assert q == pytest.approx(np.array([[0.5]]))
    assert r == pytest.approx(np.array([[2.0]]))


@pytest.mark.parametrize("dt", [0.0, -1.0, np.nan, np.inf])
def test_invalid_dt_raises(dt):
    with pytest.raises(ValueError, match="dt must be a positive finite"):
        constant_velocity_cwna(dt, 1.0)


def test_negative_psd_raises():
    with pytest.raises(ValueError, match="q_psd must be a non-negative"):
        constant_velocity_cwna(1.0, -1.0)


def test_negative_sigma_a_raises():
    with pytest.raises(ValueError, match="sigma_a must be a non-negative"):
        constant_velocity_dwna(1.0, -0.1)


def test_random_walk_rejects_zero_measurement_noise():
    with pytest.raises(ValueError, match="r must be a positive finite"):
        random_walk(1.0, 0.0)


def test_random_walk_rejects_negative_process_noise():
    with pytest.raises(ValueError, match="q must be a non-negative"):
        random_walk(-1.0, 1.0)
