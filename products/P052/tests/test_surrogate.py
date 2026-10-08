"""The learned component: fit, predict, the spread it reports, and validation."""

from __future__ import annotations

import numpy as np
import pytest

from falsifyloop.surrogate import DEFAULT_TREES, ForestSurrogate


def _linear_data(n: int = 80, d: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=(n, d))
    y = 2.0 * x[:, 0] - x[:, 1] + 0.5
    return x, y


def test_unfitted_surrogate_reports_itself_unfitted() -> None:
    model = ForestSurrogate()
    assert model.fitted is False
    assert "unfitted" in repr(model)
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(np.zeros((2, 3)))


def test_fit_returns_self_and_marks_itself_fitted() -> None:
    x, y = _linear_data()
    model = ForestSurrogate(n_estimators=8, random_state=1)
    assert model.fit(x, y) is model
    assert model.fitted is True
    assert "fitted" in repr(model)


def test_predict_shapes_and_nonnegative_spread() -> None:
    x, y = _linear_data()
    model = ForestSurrogate(n_estimators=8, random_state=1).fit(x, y)
    mean, spread = model.predict(x[:10])
    assert mean.shape == (10,)
    assert spread.shape == (10,)
    assert np.all(spread >= 0.0)
    assert np.all(np.isfinite(mean))


def test_predict_accepts_a_single_row() -> None:
    x, y = _linear_data()
    model = ForestSurrogate(n_estimators=8, random_state=1).fit(x, y)
    mean, spread = model.predict(x[0])
    assert mean.shape == (1,)
    assert spread.shape == (1,)


def test_the_forest_tracks_a_linear_response_it_has_seen() -> None:
    # Not a generalisation claim: on the training points a forest with
    # unrestricted leaves interpolates, so this checks the plumbing, not the
    # model. The held-out performance is measured in validation/, not here.
    x, y = _linear_data(n=200, seed=2)
    model = ForestSurrogate(n_estimators=40, random_state=2).fit(x, y)
    mean, _ = model.predict(x)
    assert float(np.corrcoef(mean, y)[0, 1]) > 0.95


def test_spread_is_larger_away_from_the_training_data() -> None:
    # Trees disagree most where they have seen nothing. Checked as an ordering,
    # which is all the acquisition rule relies on, and not as a calibrated
    # standard deviation, which it is not.
    rng = np.random.default_rng(3)
    x = rng.uniform(-1.0, -0.5, size=(120, 2))
    y = np.sin(3.0 * x[:, 0]) + x[:, 1]
    model = ForestSurrogate(n_estimators=30, random_state=3).fit(x, y)
    _, near = model.predict(x[:40])
    _, far = model.predict(rng.uniform(0.5, 1.0, size=(40, 2)))
    assert float(np.mean(far)) > float(np.mean(near))


def test_lower_confidence_bound_known_relationship() -> None:
    x, y = _linear_data()
    model = ForestSurrogate(n_estimators=8, random_state=1).fit(x, y)
    mean, spread = model.predict(x[:5])
    np.testing.assert_allclose(
        model.lower_confidence_bound(x[:5], kappa=2.0), mean - 2.0 * spread
    )
    np.testing.assert_allclose(model.lower_confidence_bound(x[:5], kappa=0.0), mean)


def test_lower_confidence_bound_rejects_a_negative_kappa() -> None:
    x, y = _linear_data()
    model = ForestSurrogate(n_estimators=8, random_state=1).fit(x, y)
    with pytest.raises(ValueError, match="non-negative"):
        model.lower_confidence_bound(x[:5], kappa=-1.0)


def test_fit_with_the_same_seed_is_bit_identical() -> None:
    x, y = _linear_data()
    first = ForestSurrogate(n_estimators=8, random_state=4).fit(x, y).predict(x[:20])
    second = ForestSurrogate(n_estimators=8, random_state=4).fit(x, y).predict(x[:20])
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_default_tree_count_is_the_documented_value() -> None:
    assert DEFAULT_TREES == 25
    assert ForestSurrogate().n_estimators == DEFAULT_TREES


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_estimators": 1}, "at least 2"),
        ({"min_samples_leaf": 0}, "at least 1"),
    ],
)
def test_constructor_validation(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        ForestSurrogate(**kwargs)


def test_fit_validation() -> None:
    model = ForestSurrogate(n_estimators=4)
    with pytest.raises(ValueError, match="two-dimensional"):
        model.fit(np.zeros(5), np.zeros(5))
    with pytest.raises(ValueError, match="rows but y has"):
        model.fit(np.zeros((5, 2)), np.zeros(4))
    with pytest.raises(ValueError, match="at least 2 training samples"):
        model.fit(np.zeros((1, 2)), np.zeros(1))
    with pytest.raises(ValueError, match="x contains a non-finite"):
        model.fit(np.full((4, 2), np.nan), np.zeros(4))
    with pytest.raises(ValueError, match="infinite robustness"):
        model.fit(np.zeros((4, 2)), np.array([0.0, 1.0, np.inf, 2.0]))


def test_predict_validation() -> None:
    x, y = _linear_data()
    model = ForestSurrogate(n_estimators=4, random_state=0).fit(x, y)
    with pytest.raises(ValueError, match="features but the surrogate was fit on"):
        model.predict(np.zeros((3, 7)))
    with pytest.raises(ValueError, match="non-finite"):
        model.predict(np.full((3, 3), np.inf))
    with pytest.raises(ValueError, match="one- or two-dimensional"):
        model.predict(np.zeros((2, 2, 2)))
