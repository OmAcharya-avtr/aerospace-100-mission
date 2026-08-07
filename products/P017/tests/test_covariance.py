"""Unit and known-answer tests for the covariance helpers."""

from __future__ import annotations

import numpy as np
import pytest

from estimkit import (
    covariance_health,
    is_positive_semidefinite,
    is_symmetric,
    joseph_update,
    min_eigenvalue,
    simple_update,
    symmetrize,
)


def test_symmetrize_is_bitwise_symmetric():
    p = np.array([[1.0, 0.30000000000000004], [0.3, 2.0]])
    s = symmetrize(p)
    assert s[0, 1] == s[1, 0]
    assert np.max(np.abs(s - s.T)) == 0.0


def test_symmetrize_preserves_symmetric_input_value():
    p = np.array([[2.0, -0.5], [-0.5, 3.0]])
    assert np.allclose(symmetrize(p), p)


def test_joseph_known_answer_scalar():
    # Hand calculation. P = 1, H = 1, R = 1  =>  S = 1*1*1 + 1 = 2,
    # K = P H^T / S = 1/2.  Joseph:
    #   (1 - K H)^2 P + K^2 R = (1/2)^2 * 1 + (1/2)^2 * 1 = 0.25 + 0.25 = 0.5
    p = np.array([[1.0]])
    h = np.array([[1.0]])
    r = np.array([[1.0]])
    k = np.array([[0.5]])
    assert joseph_update(p, k, h, r) == pytest.approx(np.array([[0.5]]))


def test_joseph_known_answer_two_state():
    # Hand calculation. P = [[2, 1], [1, 1]], H = [1, 0], R = 1.
    #   S = H P H^T + R = 2 + 1 = 3
    #   K = P H^T / S = [2/3, 1/3]^T
    #   A = I - K H = [[1/3, 0], [-1/3, 1]]
    #   A P A^T = [[2/9, 1/9], [1/9, 5/9]]
    #   K R K^T = [[4/9, 2/9], [2/9, 1/9]]
    #   P+      = [[6/9, 3/9], [3/9, 6/9]] = [[2/3, 1/3], [1/3, 2/3]]
    p = np.array([[2.0, 1.0], [1.0, 1.0]])
    h = np.array([[1.0, 0.0]])
    r = np.array([[1.0]])
    k = np.array([[2.0 / 3.0], [1.0 / 3.0]])
    expected = np.array([[2.0 / 3.0, 1.0 / 3.0], [1.0 / 3.0, 2.0 / 3.0]])
    assert joseph_update(p, k, h, r) == pytest.approx(expected)


def test_joseph_matches_short_form_for_optimal_gain():
    p = np.array([[3.0, 0.4], [0.4, 1.2]])
    h = np.array([[1.0, 0.5]])
    r = np.array([[0.7]])
    s = h @ p @ h.T + r
    k = p @ h.T @ np.linalg.inv(s)
    assert joseph_update(p, k, h, r) == pytest.approx(simple_update(p, k, h), abs=1e-12)


def test_joseph_stays_psd_for_suboptimal_gain_where_short_form_fails():
    # K = 1.5 x optimal. K H > I along the measured direction, so the short
    # form loses positive definiteness while the Joseph form does not.
    p = np.array([[1.0, 0.2], [0.2, 0.5]])
    h = np.array([[1.0, 0.0]])
    r = np.array([[0.01]])
    k_opt = p @ h.T @ np.linalg.inv(h @ p @ h.T + r)
    k = 1.5 * k_opt
    assert min_eigenvalue(joseph_update(p, k, h, r)) > 0.0
    assert min_eigenvalue(simple_update(p, k, h)) < 0.0


def test_joseph_zero_gain_returns_prior():
    p = np.array([[2.0, 0.1], [0.1, 0.9]])
    h = np.array([[1.0, 0.0]])
    r = np.array([[1.0]])
    k = np.zeros((2, 1))
    assert joseph_update(p, k, h, r) == pytest.approx(p)


def test_min_eigenvalue_and_psd_checks():
    p = np.diag([2.0, 0.5])
    assert min_eigenvalue(p) == pytest.approx(0.5)
    assert is_positive_semidefinite(p)
    assert is_symmetric(p)
    bad = np.diag([1.0, -0.5])
    assert not is_positive_semidefinite(bad)


def test_covariance_health_fields():
    p = np.array([[4.0, 1.0], [1.0 + 1e-9, 1.0]])
    health = covariance_health(p)
    assert health["asymmetry"] == pytest.approx(1e-9)
    assert health["trace"] == pytest.approx(5.0)
    assert health["max_eig"] > health["min_eig"] > 0.0
    assert health["condition"] == pytest.approx(health["max_eig"] / health["min_eig"])


def test_covariance_health_condition_infinite_when_singular():
    assert covariance_health(np.diag([1.0, 0.0]))["condition"] == np.inf


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"p_prior": np.eye(2), "gain": np.zeros((3, 1)), "h": np.array([[1.0, 0.0]]),
          "r": np.eye(1)}, "K must have shape"),
        ({"p_prior": np.eye(2), "gain": np.zeros((2, 1)), "h": np.array([[1.0, 0.0, 0.0]]),
          "r": np.eye(1)}, "columns"),
        ({"p_prior": np.eye(2), "gain": np.zeros((2, 1)), "h": np.array([[1.0, 0.0]]),
          "r": np.eye(2)}, "R must have shape"),
    ],
)
def test_joseph_input_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        joseph_update(**kwargs)


def test_symmetrize_rejects_non_square():
    with pytest.raises(ValueError, match="square"):
        symmetrize(np.ones((2, 3)))


def test_symmetrize_rejects_non_2d():
    with pytest.raises(ValueError, match="2-D"):
        symmetrize(np.ones(3))
