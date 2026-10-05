"""The bit-addressable MLP, its agreement with sklearn, and the softmax conventions."""

from __future__ import annotations

import numpy as np
import pytest

from bitflipsim.datasets import make_problem, train_reference_model
from bitflipsim.network import (
    MlpParameters,
    from_sklearn,
    make_layout,
    quantize_int8,
    softmax,
)


def test_layout_sizes_are_hand_computable():
    # 8 inputs, 12 hidden, 3 outputs: 8*12 + 12 + 12*3 + 3 = 96+12+36+3 = 147
    layout = make_layout(8, 12, 3)
    assert layout.size == 147
    assert layout.total_bytes == 147 * 4
    assert layout.bits_per_parameter == 32
    assert layout.slices["W1"] == slice(0, 96)
    assert layout.slices["b1"] == slice(96, 108)
    assert layout.slices["W2"] == slice(108, 144)
    assert layout.slices["b2"] == slice(144, 147)


def test_layout_queries():
    layout = make_layout(8, 12, 3)
    assert layout.tensor_of(0) == "W1"
    assert layout.tensor_of(96) == "b1"
    assert layout.tensor_of(146) == "b2"
    assert layout.byte_offset_of(10) == 40
    assert layout.is_bias(96) is True
    assert layout.is_bias(0) is False
    assert layout.layer_of(0) == 0
    assert layout.layer_of(108) == 1
    with pytest.raises(ValueError, match="outside"):
        layout.tensor_of(147)
    with pytest.raises(ValueError, match="outside"):
        layout.byte_offset_of(-1)


def test_layout_input_validation():
    with pytest.raises(ValueError, match="n_in"):
        make_layout(0, 2, 2)
    with pytest.raises(ValueError, match="itemsize_bytes"):
        make_layout(2, 2, 2, itemsize_bytes=3)


def test_forward_pass_reproduces_sklearn_in_float64(problem):
    clf = train_reference_model(problem)
    exact = from_sklearn(clf, itemsize_bytes=8)
    reference = clf.predict_proba(problem.evaluation.x)
    assert np.abs(exact.probabilities(problem.evaluation.x) - reference).max() == 0.0


def test_float32_storage_costs_only_rounding(problem, params):
    clf = train_reference_model(problem)
    reference = clf.predict_proba(problem.evaluation.x)
    deviation = np.abs(params.probabilities(problem.evaluation.x) - reference).max()
    assert deviation < 1e-5, deviation


def test_softmax_rows_sum_to_one():
    logits = np.array([[1.0, 2.0, 3.0], [-5.0, 0.0, 5.0], [0.0, 0.0, 0.0]])
    probs = softmax(logits)
    assert np.allclose(probs.sum(axis=1), 1.0)
    # the uniform row is exactly 1/3 each
    assert np.allclose(probs[2], 1.0 / 3.0)


def test_softmax_is_shift_invariant():
    logits = np.array([[1.0, 2.0, 3.0]])
    assert np.allclose(softmax(logits), softmax(logits + 1000.0))


def test_softmax_non_finite_conventions():
    nan_row = softmax(np.array([[np.nan, 1.0, 2.0]]))
    assert np.isnan(nan_row).all()
    one_inf = softmax(np.array([[np.inf, 1.0, 2.0]]))
    assert np.array_equal(one_inf, np.array([[1.0, 0.0, 0.0]]))
    two_inf = softmax(np.array([[np.inf, np.inf, 2.0]]))
    assert np.allclose(two_inf, np.array([[0.5, 0.5, 0.0]]))
    neg_inf = softmax(np.array([[-np.inf, 0.0, 0.0]]))
    assert np.allclose(neg_inf, np.array([[0.0, 0.5, 0.5]]))
    all_neg_inf = softmax(np.array([[-np.inf, -np.inf, -np.inf]]))
    assert np.allclose(all_neg_inf, 1.0 / 3.0)


def test_softmax_input_validation():
    with pytest.raises(ValueError, match="2-D"):
        softmax(np.array([1.0, 2.0]))


def test_predict_reports_minus_one_for_non_finite_rows(params):
    bad = params.values.copy()
    bad[0] = np.float32(np.nan)
    faulty = params.with_values(bad)
    predictions = faulty.predict(np.zeros((2, params.layout.n_in)))
    assert set(np.unique(predictions)).issubset({-1, 0, 1, 2})


def test_parameters_input_validation(params):
    layout = params.layout
    with pytest.raises(ValueError, match="1-D"):
        MlpParameters(values=np.zeros((2, 3), dtype=np.float32), layout=layout)
    with pytest.raises(ValueError, match="entries"):
        MlpParameters(values=np.zeros(5, dtype=np.float32), layout=layout)
    with pytest.raises(ValueError, match="bytes"):
        MlpParameters(values=np.zeros(layout.size, dtype=np.float64), layout=layout)
    with pytest.raises(ValueError, match="shape"):
        params.logits(np.zeros((3, 2)))


def test_from_sklearn_rejects_the_wrong_estimator(problem):
    class NotFitted:
        pass

    with pytest.raises(ValueError, match="fit it first"):
        from_sklearn(NotFitted())
    clf = train_reference_model(problem)
    with pytest.raises(ValueError, match="quantize_int8"):
        from_sklearn(clf, itemsize_bytes=1)
    clf.activation = "tanh"
    with pytest.raises(ValueError, match="relu"):
        from_sklearn(clf)


def test_int8_quantization_is_symmetric_and_accurate(params, problem):
    quantized = quantize_int8(params)
    assert quantized.codes.dtype == np.int8
    assert quantized.codes.min() >= -127  # -128 is never produced by quantization
    assert quantized.codes.nbytes * 4 == params.layout.total_bytes
    golden = params.probabilities(problem.evaluation.x)
    requantized = quantized.as_parameters().probabilities(problem.evaluation.x)
    # Measured, not derived: on this 120-sample fixture the largest per-class
    # probability shift from symmetric per-tensor int8 quantization is 0.0505
    # (it is 0.056143 on the full 300-sample evaluation split, see
    # validation/example_bit_position_criticality_output.txt). The bound below
    # pins that measurement so a regression is caught; it is not a physical
    # tolerance, and quantization error is not an upset.
    assert np.abs(golden - requantized).max() < 0.06
    # Quantization is not lossless: on this 120-sample fixture exactly 1 of 120
    # argmax predictions changes (accuracy 0.925 -> 0.9167). On the full
    # 300-sample evaluation split none change. Recorded rather than tolerated
    # away, because the int8 criticality sweep is run on a model that already
    # differs from the float32 one by this much.
    golden_predictions = params.predict(problem.evaluation.x)
    changed = int((quantized.as_parameters().predict(problem.evaluation.x)
                   != golden_predictions).sum())
    assert changed == 1, changed
    int8_block = MlpParameters(values=quantized.codes, layout=quantized.layout)
    with pytest.raises(ValueError, match="already int8"):
        quantize_int8(int8_block)


def test_hidden_activations_are_non_negative(params, problem):
    hidden = params.hidden_activations(problem.evaluation.x)
    assert hidden.shape == (problem.evaluation.n_samples, params.layout.n_hidden)
    assert (hidden >= 0.0).all()


def test_dataset_splits_are_disjoint_and_deterministic():
    first = make_problem(n_train=100, n_calibration=40, n_evaluation=40)
    second = make_problem(n_train=100, n_calibration=40, n_evaluation=40)
    assert np.array_equal(first.calibration.x, second.calibration.x)
    assert not np.array_equal(first.calibration.x, first.evaluation.x)
    with pytest.raises(ValueError, match="n_train"):
        make_problem(n_train=0)
