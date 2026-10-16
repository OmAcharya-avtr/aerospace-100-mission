"""The declared correlation structure and the quadratic form it defines."""

from __future__ import annotations

import math

import numpy as np
import pytest

from budgetroll import Correlation, correlation_matrix, is_positive_semidefinite, quadratic_form
from budgetroll.correlation import budget_correlation_matrix, declared_pairs
from budgetroll.errors import AggregationError


def test_matrix_is_identity_when_nothing_is_declared():
    matrix = correlation_matrix(["a", "b", "c"], ())
    assert np.allclose(matrix, np.eye(3))


def test_matrix_is_symmetric_and_placed_correctly():
    matrix = correlation_matrix(["a", "b", "c"], (Correlation("c", "a", 0.4, "r"),))
    assert matrix[0, 2] == pytest.approx(0.4)
    assert matrix[2, 0] == pytest.approx(0.4)
    assert matrix[0, 1] == 0.0


def test_matrix_ignores_a_correlation_outside_the_given_leaves():
    matrix = correlation_matrix(["a", "b"], (Correlation("a", "zz", 0.9, "r"),))
    assert np.allclose(matrix, np.eye(2))


def test_identity_is_positive_semidefinite():
    ok, smallest = is_positive_semidefinite(np.eye(4))
    assert ok
    assert smallest == pytest.approx(1.0, rel=1e-12)


def test_all_ones_matrix_is_positive_semidefinite_and_singular():
    ok, smallest = is_positive_semidefinite(np.ones((3, 3)))
    assert ok
    assert abs(smallest) < 1e-9


def test_inadmissible_triangle_is_rejected():
    # [[1, .9, .9], [.9, 1, 0], [.9, 0, 1]] has determinant
    #   1 * (1 - 0) - 0.9 * (0.9 - 0) + 0.9 * (0 - 0.9) = 1 - 0.81 - 0.81 = -0.62
    matrix = np.array([[1.0, 0.9, 0.9], [0.9, 1.0, 0.0], [0.9, 0.0, 1.0]])
    assert np.linalg.det(matrix) == pytest.approx(-0.62, rel=1e-12)
    ok, smallest = is_positive_semidefinite(matrix)
    assert not ok
    assert smallest < 0.0


def test_psd_check_rejects_non_square():
    with pytest.raises(AggregationError, match="must be square"):
        is_positive_semidefinite(np.ones((2, 3)))


def test_psd_check_rejects_asymmetric():
    with pytest.raises(AggregationError, match="must be symmetric"):
        is_positive_semidefinite(np.array([[1.0, 0.5], [0.2, 1.0]]))


def test_psd_check_on_an_empty_matrix():
    ok, smallest = is_positive_semidefinite(np.zeros((0, 0)))
    assert ok
    assert smallest == 0.0


def test_quadratic_form_rejects_shape_mismatch():
    with pytest.raises(AggregationError, match="does not match"):
        quadratic_form([1.0, 2.0], np.eye(3))


def test_quadratic_form_rejects_two_dimensional_weights():
    with pytest.raises(AggregationError, match="one-dimensional"):
        quadratic_form(np.ones((2, 2)), np.eye(2))


def test_quadratic_form_clips_a_negative_variance_from_an_inadmissible_matrix():
    # u = (1, 1, 1) with the inadmissible triangle above but signs flipped:
    #   var = 3 + 2 * (-0.9) + 2 * (-0.9) + 2 * (-0.9) = 3 - 5.4 = -2.4
    matrix = np.array([[1.0, -0.9, -0.9], [-0.9, 1.0, -0.9], [-0.9, -0.9, 1.0]])
    ok, _ = is_positive_semidefinite(matrix)
    assert not ok
    assert quadratic_form([1.0, 1.0, 1.0], matrix) == 0.0


def test_budget_matrix_orders_leaves_left_to_right(planted):
    leaf_ids, matrix = budget_correlation_matrix(planted)
    assert leaf_ids == ["s1", "s2", "s3", "a1", "a2", "p1", "t1", "t2"]
    assert matrix[0, 1] == pytest.approx(0.9)
    assert matrix[0, 2] == pytest.approx(0.9)
    assert matrix[1, 2] == 0.0


def test_declared_pairs_filters_by_membership(planted):
    assert len(declared_pairs(planted, ["s1", "s2", "s3"])) == 2
    assert declared_pairs(planted, ["s2", "s3"]) == []
    assert declared_pairs(planted, ["a1", "a2"]) == []


def test_rho_zero_is_not_a_declared_pair():
    ids = ["a", "b"]
    matrix = correlation_matrix(ids, (Correlation("a", "b", 0.0, "r"),))
    assert np.allclose(matrix, np.eye(2))


def test_quadratic_form_known_answer_three_terms():
    # u = (1, 2, 2), rho(0,1) = 0.5, others zero:
    #   var = 1 + 4 + 4 + 2 * 0.5 * 1 * 2 = 9 + 2 = 11
    matrix = correlation_matrix(
        ["a", "b", "c"], (Correlation("a", "b", 0.5, "r"),)
    )
    assert quadratic_form([1.0, 2.0, 2.0], matrix) == pytest.approx(math.sqrt(11.0), rel=1e-12)
