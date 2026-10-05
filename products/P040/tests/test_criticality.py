"""Degradation metrics, the two baselines, and the protection metric."""

from __future__ import annotations

import numpy as np
import pytest

from bitflipsim.bitlayout import float_layout
from bitflipsim.criticality import (
    _expected_top_k_sum,
    accuracy,
    evaluate_protection,
    exponent_bit_baseline_scores,
    magnitude_baseline_scores,
    mean_total_variation,
    oracle_scores,
    random_scores,
    sweep_bit_criticality,
    total_variation,
)


def test_total_variation_known_values():
    # TV([1,0],[0,1]) = 0.5 * (|1-0| + |0-1|) = 1.0
    assert total_variation(np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]]))[0] == 1.0
    # identical distributions give 0
    p = np.array([[0.2, 0.3, 0.5]])
    assert total_variation(p, p)[0] == 0.0
    # TV([0.5,0.5],[0.7,0.3]) = 0.5*(0.2+0.2) = 0.2
    assert total_variation(
        np.array([[0.5, 0.5]]), np.array([[0.7, 0.3]])
    )[0] == pytest.approx(0.2)


def test_total_variation_nan_convention_is_maximal():
    golden = np.array([[0.5, 0.5]])
    faulty = np.array([[np.nan, np.nan]])
    assert total_variation(golden, faulty)[0] == 1.0
    assert mean_total_variation(golden, faulty) == 1.0


def test_total_variation_is_bounded():
    rng = np.random.default_rng(3)
    a = rng.dirichlet(np.ones(4), size=200)
    b = rng.dirichlet(np.ones(4), size=200)
    tv = total_variation(a, b)
    assert tv.min() >= 0.0
    assert tv.max() <= 1.0


def test_total_variation_validation():
    with pytest.raises(ValueError, match="2-D shape"):
        total_variation(np.array([1.0, 0.0]), np.array([0.0, 1.0]))


def test_accuracy_counts_minus_one_as_wrong():
    assert accuracy(np.array([0, 1, -1]), np.array([0, 1, 2])) == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="shape mismatch"):
        accuracy(np.array([0, 1]), np.array([0]))


def test_magnitude_baseline_is_bit_independent(params):
    scores = magnitude_baseline_scores(params)
    assert scores.shape == (params.layout.size, 32)
    assert np.allclose(scores.std(axis=1), 0.0)
    assert np.allclose(scores[:, 0], np.abs(params.values.astype(np.float64)))


def test_exponent_baseline_is_parameter_independent_and_ordered():
    layout = float_layout("float32")
    scores = exponent_bit_baseline_scores(5, layout)
    assert scores.shape == (5, 32)
    assert np.allclose(scores.std(axis=0), 0.0)
    row = scores[0]
    # exponent bits 23..30 carry weights 1,2,4,...,128
    assert list(row[23:31]) == [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]
    # the sign bit doubles the magnitude, log2 ratio 1
    assert row[31] == 1.0
    # mantissa bit b has log2 relative change b - 23, so bit 22 -> -1, bit 0 -> -23
    assert row[22] == -1.0
    assert row[0] == -23.0
    with pytest.raises(ValueError, match="n_parameters"):
        exponent_bit_baseline_scores(0, layout)


def test_expected_top_k_sum_handles_ties_exactly():
    scores = np.array([5.0, 1.0, 1.0, 1.0, 1.0])
    values = np.array([10.0, 4.0, 8.0, 0.0, 0.0])
    # k = 1 takes the strict leader
    assert _expected_top_k_sum(scores, values, 1) == 10.0
    # k = 3 takes the leader plus 2 of 4 tied, expectation 10 + (2/4)*12 = 16
    assert _expected_top_k_sum(scores, values, 3) == pytest.approx(16.0)
    # k = 5 takes everything
    assert _expected_top_k_sum(scores, values, 5) == pytest.approx(22.0)
    assert _expected_top_k_sum(scores, values, 0) == 0.0
    with pytest.raises(ValueError, match="k must be"):
        _expected_top_k_sum(scores, values, -1)
    with pytest.raises(ValueError, match="share shape"):
        _expected_top_k_sum(scores, values[:2], 1)


def test_fully_tied_scores_give_the_random_selection_expectation():
    # A method that cannot distinguish anything must score exactly the mean.
    degradation = np.array([[1.0, 3.0], [0.0, 8.0]])
    scores = np.ones_like(degradation)
    result = evaluate_protection(scores, degradation, budget_bytes=0.25, itemsize_bytes=4,
                                 cost_model="bit", method="tied")
    # 0.25 bytes = 2 bits of 4; expectation = (2/4) * 12 = 6
    assert result.protected_units == 2
    assert result.avoided == pytest.approx(6.0)
    assert result.cost_bytes == pytest.approx(0.25)
    assert result.avoided_per_byte == pytest.approx(24.0)
    assert result.avoided_fraction == pytest.approx(0.5)


def test_oracle_is_an_upper_bound_on_every_method(params, problem):
    sweep = sweep_bit_criticality(params, problem.evaluation.x, problem.evaluation.y)
    layout = float_layout("float32")
    degradation = sweep.degradation
    methods = {
        "magnitude": magnitude_baseline_scores(params),
        "exponent": exponent_bit_baseline_scores(params.layout.size, layout),
        "random": random_scores(degradation.shape, np.random.default_rng(1)),
    }
    for cost_model in ("bit", "word"):
        for fraction in (0.05, 0.25, 0.5):
            budget = params.layout.total_bytes * fraction
            best = evaluate_protection(
                oracle_scores(degradation), degradation, budget, 4, cost_model, "oracle"
            )
            for name, scores in methods.items():
                result = evaluate_protection(
                    scores, degradation, budget, 4, cost_model, name
                )
                assert result.avoided <= best.avoided + 1e-9, (
                    cost_model, fraction, name, result.avoided, best.avoided
                )


def test_sweep_shapes_and_golden_accuracy(params, problem):
    sweep = sweep_bit_criticality(params, problem.evaluation.x, problem.evaluation.y)
    assert sweep.degradation.shape == (params.layout.size, 32)
    assert sweep.n_evaluations == params.layout.size * 32
    assert sweep.n_parameters == params.layout.size
    assert sweep.bits_per_parameter == 32
    assert 0.0 <= sweep.golden_accuracy <= 1.0
    assert sweep.degradation.min() >= 0.0
    assert sweep.degradation.max() <= 1.0
    assert sweep.flat_degradation().size == sweep.degradation.size
    assert sweep.by_bit_position().shape == (32,)
    assert sweep.by_parameter().shape == (params.layout.size,)


def test_exponent_msb_is_the_most_critical_bit_position(params, problem):
    """The premise of the exponent-bit heuristic, measured rather than assumed."""
    sweep = sweep_bit_criticality(params, problem.evaluation.x, problem.evaluation.y)
    per_bit = sweep.by_bit_position()
    assert int(np.argmax(per_bit)) == 30
    # exponent field and sign together must dominate the mantissa
    exponent_and_sign = per_bit[23:32].sum()
    mantissa = per_bit[0:23].sum()
    assert exponent_and_sign > 10.0 * mantissa


def test_protection_input_validation(params):
    degradation = np.zeros((params.layout.size, 32))
    scores = np.zeros_like(degradation)
    with pytest.raises(ValueError, match="2-D shape"):
        evaluate_protection(scores, degradation[:, :4], 1.0, 4)
    with pytest.raises(ValueError, match="budget_bytes"):
        evaluate_protection(scores, degradation, -1.0, 4)
    with pytest.raises(ValueError, match="itemsize_bytes"):
        evaluate_protection(scores, degradation, 1.0, 0)
    with pytest.raises(ValueError, match="cost_model"):
        evaluate_protection(scores, degradation, 1.0, 4, "page")  # type: ignore[arg-type]


def test_zero_budget_avoids_nothing(params):
    degradation = np.ones((params.layout.size, 32))
    result = evaluate_protection(
        oracle_scores(degradation), degradation, 0.0, 4, "bit", "oracle"
    )
    assert result.avoided == 0.0
    assert result.avoided_per_byte == 0.0
    assert result.avoided_fraction == pytest.approx(0.0)
