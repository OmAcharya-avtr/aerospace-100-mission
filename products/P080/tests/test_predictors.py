"""Predictor unit tests. Integer exactness, causality, warm-up and known answers."""

from __future__ import annotations

import numpy as np
import pytest

from boundcompress import predictors as P


def test_known_answer_previous() -> None:
    """q = [5, 5, 7, 7]; previous predicts [5, 5, 7]; residuals [0, 2, 0]."""
    engine = P.get_predictor("previous")
    assert engine.order == 1
    assert engine.residuals(np.array([5, 5, 7, 7])).tolist() == [0, 2, 0]


def test_known_answer_linear_is_exact_on_an_arithmetic_progression() -> None:
    """2q1 - q2 predicts an arithmetic progression with zero residual."""
    engine = P.get_predictor("linear")
    assert engine.order == 2
    ramp = np.arange(0, 100, 3)
    assert engine.residuals(ramp).tolist() == [0] * (ramp.size - 2)


def test_known_answer_quadratic_is_exact_on_a_quadratic_sequence() -> None:
    """3q1 - 3q2 + q3 predicts k^2 exactly: residuals are all zero."""
    engine = P.get_predictor("quadratic")
    assert engine.order == 3
    squares = (np.arange(40) ** 2).astype(np.int64)
    assert engine.residuals(squares).tolist() == [0] * (squares.size - 3)


def test_known_answer_cubic_is_exact_on_a_cubic_sequence() -> None:
    engine = P.get_predictor("cubic")
    assert engine.order == 4
    cubes = (np.arange(40) ** 3).astype(np.int64)
    assert engine.residuals(cubes).tolist() == [0] * (cubes.size - 4)


def test_linear_is_not_exact_on_a_quadratic_sequence() -> None:
    """2q1 - q2 on k^2 leaves a constant residual of 2, the second difference."""
    engine = P.get_predictor("linear")
    squares = (np.arange(20) ** 2).astype(np.int64)
    assert engine.residuals(squares).tolist() == [2] * (squares.size - 2)


def test_known_answer_predict_one() -> None:
    """Hand-calculated: history [..., 10, 14]; linear predicts 2*14 - 10 = 18."""
    assert P.get_predictor("linear").predict_one([3, 10, 14]) == 18
    assert P.get_predictor("quadratic").predict_one([1, 4, 9]) == 3 * 9 - 3 * 4 + 1
    assert P.get_predictor("previous").predict_one([1, 2, 3]) == 3


@pytest.mark.parametrize("name", P.CLASSICAL_NAMES)
def test_residuals_and_reconstruct_are_inverse(name: str) -> None:
    engine = P.get_predictor(name)
    rng = np.random.default_rng(11)
    symbols = np.cumsum(rng.integers(-40, 40, size=500)).astype(np.int64)
    residuals = engine.residuals(symbols)
    assert residuals.size == symbols.size - engine.order
    rebuilt = engine.reconstruct(symbols[: engine.order].tolist(), residuals)
    assert np.array_equal(rebuilt, symbols)


@pytest.mark.parametrize("name", P.CLASSICAL_NAMES)
def test_prediction_is_an_exact_integer(name: str) -> None:
    """No float enters the classical path, so the coder is portable bit-for-bit."""
    engine = P.get_predictor(name)
    history = [10**15, -(10**15), 3, 7][: max(4, engine.order)]
    prediction = engine.predict_one(history)
    assert isinstance(prediction, int)


def test_noise_gain_known_answers() -> None:
    """Sum of absolute coefficients: 1, 3, 7, 15."""
    assert [P.noise_gain(name) for name in P.CLASSICAL_NAMES] == [1, 3, 7, 15]


def test_unknown_predictor_is_refused() -> None:
    with pytest.raises(P.PredictorError, match="unknown predictor"):
        P.get_predictor("wiener")


@pytest.mark.parametrize("name", P.CLASSICAL_NAMES)
def test_array_no_longer_than_the_warmup_is_refused(name: str) -> None:
    engine = P.get_predictor(name)
    with pytest.raises(P.PredictorError, match="needs more than"):
        engine.residuals(np.arange(engine.order, dtype=np.int64))


@pytest.mark.parametrize("name", P.CLASSICAL_NAMES)
def test_wrong_warmup_length_is_refused(name: str) -> None:
    engine = P.get_predictor(name)
    with pytest.raises(P.PredictorError, match="exactly"):
        engine.reconstruct([0] * (engine.order + 1), np.array([1, 2]))


def test_predictor_ids_are_distinct_and_cover_the_names() -> None:
    assert len(set(P.PREDICTOR_IDS.values())) == len(P.PREDICTOR_IDS)
    for name in P.CLASSICAL_NAMES:
        assert P.PREDICTOR_NAMES[P.PREDICTOR_IDS[name]] == name
