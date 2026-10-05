"""Tests for the multivariate novelty models and their confidence output."""

from __future__ import annotations

import numpy as np
import pytest

from telemetryool.novelty import (
    GmmNovelty,
    HotellingT2Q,
    IsolationForestNovelty,
)
from telemetryool.synthetic import NominalModel, equicorrelation, generate_nominal


def _nominal(n: int = 8000, channels: int = 4, cross: float = 0.6) -> np.ndarray:
    model = NominalModel(channels, correlation=equicorrelation(channels, cross))
    return generate_nominal(model, n // 100, 100, np.random.default_rng(42)).reshape(-1, channels)


MODELS = [
    lambda: HotellingT2Q(n_components=2),
    lambda: GmmNovelty(n_components=3, random_state=0),
    lambda: IsolationForestNovelty(n_estimators=30, random_state=0),
]


@pytest.mark.parametrize("factory", MODELS, ids=["t2q", "gmm", "iforest"])
def test_outlier_scores_above_the_nominal_median(factory) -> None:
    x = _nominal()
    model = factory().fit(x)
    nominal_median = float(np.median(model.score(x)))
    far = model.score(np.full((3, x.shape[1]), 6.0))
    assert np.all(far > nominal_median)


@pytest.mark.parametrize("factory", MODELS, ids=["t2q", "gmm", "iforest"])
def test_pvalue_is_small_for_a_gross_outlier_and_large_at_the_centre(factory) -> None:
    x = _nominal()
    model = factory().fit(x)
    centre, outlier = model.novelty_pvalue(
        np.vstack([np.zeros((1, x.shape[1])), np.full((1, x.shape[1]), 8.0)])
    )
    assert outlier.pvalue <= 10.0 / (1 + x.shape[0])
    assert centre.pvalue > 0.1
    assert outlier.low <= outlier.high
    assert outlier.n_calibration == x.shape[0]
    assert outlier.resolution_floor == pytest.approx(1.0 / (1 + x.shape[0]))


@pytest.mark.parametrize("factory", MODELS, ids=["t2q", "gmm", "iforest"])
def test_pvalue_cannot_go_below_the_resolution_floor(factory) -> None:
    x = _nominal(2000)
    model = factory().fit(x)
    floor = model.pvalue_resolution_floor()
    assert floor == pytest.approx(1.0 / 2001)
    pv = model.novelty_pvalue(np.full((1, x.shape[1]), 50.0))[0]
    assert pv.pvalue >= floor
    assert pv.pvalue == pytest.approx(floor)


def test_pvalues_are_approximately_uniform_on_nominal_data() -> None:
    """The p-value is an empirical tail probability under the nominal model, so
    on held-out nominal data the fraction below 0.05 should be close to 0.05.
    With 4000 held-out samples the binomial SE is 0.0034, so a 5-sigma band is
    [0.033, 0.067]."""
    train = _nominal(8000)
    model = HotellingT2Q(n_components=2).fit(train)
    held_out = generate_nominal(
        NominalModel(4, correlation=equicorrelation(4, 0.6)),
        40, 100, np.random.default_rng(777),
    ).reshape(-1, 4)
    pvals = np.array([p.pvalue for p in model.novelty_pvalue(held_out)])
    assert 0.033 < float((pvals < 0.05).mean()) < 0.067


def test_t2q_statistics_are_separately_available() -> None:
    x = _nominal()
    model = HotellingT2Q(n_components=2).fit(x)
    t2, q = model.statistics(x[:100])
    assert t2.shape == (100,) and q.shape == (100,)
    assert np.all(t2 >= 0.0) and np.all(q >= 0.0)
    assert model.n_retained == 2


def test_t2q_variance_target_selects_components() -> None:
    """With equicorrelation 0.9 on four channels the first principal component
    carries (1 + 3 * 0.9) / 4 = 0.925 of the variance, so a 0.9 target retains
    exactly one component."""
    model = HotellingT2Q(variance_target=0.9).fit(_nominal(8000, 4, 0.9))
    assert model.n_retained == 1


def test_t2q_q_is_zero_when_all_components_are_retained() -> None:
    x = _nominal(4000, 3, 0.5)
    model = HotellingT2Q(n_components=3).fit(x)
    _, q = model.statistics(x[:50])
    assert np.allclose(q, 0.0)


def test_models_are_reproducible_for_a_fixed_seed() -> None:
    x = _nominal(4000)
    for factory in (
        lambda: GmmNovelty(n_components=3, random_state=5),
        lambda: IsolationForestNovelty(n_estimators=20, random_state=5),
    ):
        a = factory().fit(x).score(x[:200])
        b = factory().fit(x).score(x[:200])
        assert np.array_equal(a, b)


def test_scoring_before_fitting_raises() -> None:
    model = GmmNovelty()
    with pytest.raises(RuntimeError, match="not fitted"):
        model.score(np.zeros((2, 3)))
    with pytest.raises(RuntimeError, match="not fitted"):
        model.novelty_pvalue(np.zeros((2, 3)))
    with pytest.raises(RuntimeError, match="not fitted"):
        model.pvalue_resolution_floor()
    with pytest.raises(RuntimeError, match="not fitted"):
        _ = model.n_features


def test_fit_input_validation() -> None:
    model = HotellingT2Q()
    with pytest.raises(ValueError, match="must be 2-D"):
        model.fit(np.zeros(10))
    with pytest.raises(ValueError, match="at least 2 samples"):
        model.fit(np.zeros((1, 3)))
    with pytest.raises(ValueError, match="non-finite"):
        model.fit(np.array([[1.0, 2.0], [np.nan, 1.0], [0.0, 0.0]]))
    with pytest.raises(ValueError, match="zero variance"):
        model.fit(np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]))


def test_score_feature_count_is_checked() -> None:
    model = HotellingT2Q(n_components=1).fit(_nominal(2000, 3, 0.4))
    assert model.n_features == 3
    with pytest.raises(ValueError, match="model was fitted on"):
        model.score(np.zeros((4, 5)))
    with pytest.raises(ValueError, match="must be 2-D"):
        model.score(np.zeros(3))


def test_constructor_validation() -> None:
    with pytest.raises(ValueError, match="n_components must be >= 1 or None"):
        HotellingT2Q(n_components=0)
    with pytest.raises(ValueError, match="variance_target must lie in"):
        HotellingT2Q(variance_target=1.0)
    with pytest.raises(ValueError, match="inner_quantile must lie in"):
        HotellingT2Q(inner_quantile=0.0)
    with pytest.raises(ValueError, match="n_components must be >= 1"):
        GmmNovelty(n_components=0)
    with pytest.raises(ValueError, match="n_estimators must be >= 1"):
        IsolationForestNovelty(n_estimators=0)
