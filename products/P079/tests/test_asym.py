"""The asymmetry correctors: features, analytic baselines, learned models.

The dataset built here is small -- 160 scenarios -- because these are unit
tests, not the benchmark. The benchmark, at the size every published number
comes from, is ``validation/validate_asymmetry_benchmark.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from timetransfer.acquire import acquire
from timetransfer.asym import (
    FEATURE_NAMES,
    MIN_WINDOW,
    GeneratorConfig,
    LearnedAsymmetryCorrector,
    MeanDifferenceEstimator,
    MedianFilterEstimator,
    RidgeAsymmetryCorrector,
    ScoreTable,
    SymmetricAssumption,
    TrainingConfig,
    build_dataset,
    centred_dwells_s,
    generate_scenario,
    score_predictions,
    window_features,
)
from timetransfer.backends import SimulatedBackend
from timetransfer.errors import ConfigurationError, InsufficientDataError
from timetransfer.exchange import ExchangeSet

from conftest import make_spec

DATASET_SIZE = 160


@pytest.fixture(scope="module")
def dataset():
    return build_dataset(DATASET_SIZE, seed=909, config=GeneratorConfig(window=60))


def _exchanges(**kwargs) -> ExchangeSet:
    backend = SimulatedBackend(make_spec(**kwargs))
    backend.open()
    out = acquire(backend, 60, min_exchanges=8).exchanges
    backend.close()
    return out


def test_feature_names_are_unique_and_sixteen() -> None:
    assert len(FEATURE_NAMES) == 16
    assert len(set(FEATURE_NAMES)) == 16


def test_centred_dwells_are_non_negative_with_a_zero() -> None:
    fwd, rev = centred_dwells_s(_exchanges())
    assert float(np.min(fwd)) == 0.0
    assert float(np.min(rev)) == 0.0
    assert float(np.max(fwd)) >= 0.0


def test_features_are_invariant_to_the_clock_offset() -> None:
    # Adding a constant to t2 and t3 shifts the remote clock by that constant.
    # The centred dwells, and therefore every feature, must not move. The
    # comparison is to tolerance, not exact: the shift is exact in integers but
    # the centring is done after conversion to float seconds.
    base = _exchanges(asymmetry_s=4e-4, timestamp_noise_s=2e-8)
    shifted = ExchangeSet(
        t1_ns=base.t1_ns,
        t2_ns=base.t2_ns + 7_000_000,
        t3_ns=base.t3_ns + 7_000_000,
        t4_ns=base.t4_ns,
        seq=base.seq,
    )
    assert window_features(shifted) == pytest.approx(window_features(base), abs=1e-14)


def test_features_are_invariant_to_a_common_time_shift() -> None:
    base = _exchanges()
    shifted = ExchangeSet(
        t1_ns=base.t1_ns + 10**9,
        t2_ns=base.t2_ns + 10**9,
        t3_ns=base.t3_ns + 10**9,
        t4_ns=base.t4_ns + 10**9,
        seq=base.seq,
    )
    assert window_features(shifted) == pytest.approx(window_features(base), abs=1e-14)


def test_features_respond_to_the_queueing_asymmetry() -> None:
    backend = SimulatedBackend(make_spec(jitter_scale_s=2e-4, seed=3))
    backend.open()
    balanced = acquire(backend, 200, min_exchanges=8).exchanges
    backend.close()
    index = FEATURE_NAMES.index("median_difference_s")
    assert abs(window_features(balanced)[index]) < 2e-4


def test_features_refuse_a_window_below_the_minimum() -> None:
    small = _exchanges().subset(slice(0, MIN_WINDOW - 1))
    with pytest.raises(InsufficientDataError, match="window features need"):
        window_features(small)


def test_symmetric_assumption_predicts_exactly_zero(dataset) -> None:
    pred = SymmetricAssumption().predict(dataset.features)
    assert pred.tolist() == [0.0] * dataset.n


def test_median_filter_is_half_the_median_difference_feature(dataset) -> None:
    index = FEATURE_NAMES.index("median_difference_s")
    pred = MedianFilterEstimator().predict(dataset.features)
    assert pred == pytest.approx(0.5 * dataset.features[:, index], rel=1e-15)


def test_mean_difference_is_half_the_mean_difference_feature(dataset) -> None:
    index = FEATURE_NAMES.index("mean_difference_s")
    pred = MeanDifferenceEstimator().predict(dataset.features)
    assert pred == pytest.approx(0.5 * dataset.features[:, index], rel=1e-15)


def test_analytic_residual_spread_is_unavailable_before_fit() -> None:
    estimator = MedianFilterEstimator()
    with pytest.raises(ConfigurationError, match="only available after fit"):
        _ = estimator.residual_std_s


def test_analytic_fit_measures_the_residual_spread(dataset) -> None:
    estimator = MedianFilterEstimator().fit(dataset.features, dataset.target_s)
    assert estimator.residual_std_s > 0.0
    assert estimator.residual_mad_s > 0.0
    mean, sigma = estimator.predict_with_uncertainty(dataset.features)
    assert mean.size == dataset.n
    assert float(np.unique(sigma).size) == 1.0


def test_analytic_estimators_reject_a_wrong_feature_width(dataset) -> None:
    with pytest.raises(ConfigurationError, match="features must have shape"):
        MedianFilterEstimator().predict(np.zeros((4, 3)))


def test_analytic_estimators_reject_non_finite_features() -> None:
    bad = np.zeros((4, 16))
    bad[0, 0] = np.nan
    with pytest.raises(ConfigurationError, match="finite"):
        MedianFilterEstimator().predict(bad)


def test_ridge_beats_the_symmetric_assumption(dataset) -> None:
    train, test = dataset.split(train_fraction=0.7)
    ridge = RidgeAsymmetryCorrector().fit(dataset.features[train], dataset.target_s[train])
    zero = SymmetricAssumption().fit(dataset.features[train], dataset.target_s[train])
    ref = score_predictions("zero", zero.predict(dataset.features[test]), dataset.target_s[test])
    scored = score_predictions(
        "ridge",
        ridge.predict(dataset.features[test]),
        dataset.target_s[test],
        reference_rmse_s=ref.rmse_s,
    )
    assert scored.rmse_s < ref.rmse_s
    assert scored.skill > 0.0
    assert ridge.coefficients.size == 16


def test_learned_corrector_trains_and_reports_an_uncertainty(dataset) -> None:
    train, test = dataset.split(train_fraction=0.7)
    model = LearnedAsymmetryCorrector(
        TrainingConfig(n_estimators=40, min_samples_leaf=4)
    ).fit(dataset.features[train], dataset.target_s[train])
    mean, sigma = model.predict_with_uncertainty(dataset.features[test])
    assert mean.size == test.size
    assert np.all(sigma >= 0.0)
    assert float(np.mean(sigma)) > 0.0
    assert model.residual_std_s > 0.0
    assert model.feature_importances.size == 16
    assert model.feature_importances.sum() == pytest.approx(1.0, rel=1e-6)
    assert 0.0 <= model.crossing_fraction <= 1.0


def test_learned_corrector_refuses_to_predict_before_fit(dataset) -> None:
    with pytest.raises(ConfigurationError, match="call fit"):
        LearnedAsymmetryCorrector().predict(dataset.features)
    with pytest.raises(ConfigurationError, match="call fit"):
        RidgeAsymmetryCorrector().predict(dataset.features)


def test_learned_corrector_refuses_too_few_rows() -> None:
    with pytest.raises(InsufficientDataError):
        LearnedAsymmetryCorrector(TrainingConfig(min_samples_leaf=8)).fit(
            np.zeros((10, 16)), np.zeros(10)
        )
    with pytest.raises(InsufficientDataError):
        RidgeAsymmetryCorrector().fit(np.zeros((10, 16)), np.zeros(10))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_estimators": 2}, "n_estimators"),
        ({"max_depth": 0}, "max_depth"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"min_samples_leaf": 0}, "min_samples_leaf"),
        ({"sigma_quantiles": (0.9, 0.1)}, "sigma_quantiles"),
    ],
)
def test_training_config_validates(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ConfigurationError, match=match):
        TrainingConfig(**kwargs)


def test_ridge_alpha_validates() -> None:
    with pytest.raises(ConfigurationError, match="alpha"):
        RidgeAsymmetryCorrector(alpha=0.0)


def test_generator_truth_decomposes_the_target() -> None:
    exchanges, truth = generate_scenario(0, seed=4242, config=GeneratorConfig(window=60))
    assert exchanges.n >= MIN_WINDOW
    assert truth["bias_s"] == pytest.approx(truth["static_bias_s"] + truth["queue_bias_s"])
    assert 0.0 <= truth["loss_fraction"] < 1.0


def test_generator_config_validates() -> None:
    with pytest.raises(ConfigurationError, match="window"):
        GeneratorConfig(window=4)
    with pytest.raises(ConfigurationError, match="coupling"):
        GeneratorConfig(coupling=-1.0)
    with pytest.raises(ConfigurationError, match="independent_sigma_s"):
        GeneratorConfig(independent_sigma_s=0.0)
    with pytest.raises(ConfigurationError, match="max_loss_probability"):
        GeneratorConfig(max_loss_probability=0.9)


def test_zero_coupling_removes_the_static_asymmetry_correlation() -> None:
    coupled = build_dataset(120, seed=31, config=GeneratorConfig(window=40, coupling=3.0))
    uncoupled = build_dataset(120, seed=31, config=GeneratorConfig(window=40, coupling=0.0))

    def correlation(ds) -> float:
        return float(np.corrcoef(ds.static_bias_s, ds.congestion_difference)[0, 1])

    assert abs(correlation(coupled)) > abs(correlation(uncoupled))


def test_dataset_split_is_deterministic_and_ordered(dataset) -> None:
    a_train, a_test = dataset.split(train_fraction=0.7)
    b_train, b_test = dataset.split(train_fraction=0.7)
    assert a_train.tolist() == b_train.tolist()
    assert a_test.tolist() == b_test.tolist()
    assert int(a_train.max()) < int(a_test.min())
    assert a_train.size + a_test.size == dataset.n


def test_dataset_split_validates(dataset) -> None:
    with pytest.raises(ConfigurationError, match="train_fraction"):
        dataset.split(train_fraction=1.0)


def test_build_dataset_validates() -> None:
    with pytest.raises(ConfigurationError, match="n_scenarios"):
        build_dataset(4)


def test_score_table_renders_and_ranks(dataset) -> None:
    train, test = dataset.split(train_fraction=0.7)
    rows = []
    reference = None
    for estimator in (SymmetricAssumption(), MedianFilterEstimator(), MeanDifferenceEstimator()):
        estimator.fit(dataset.features[train], dataset.target_s[train])
        scored = score_predictions(
            estimator.name,
            estimator.predict(dataset.features[test]),
            dataset.target_s[test],
            reference_rmse_s=reference,
        )
        if reference is None:
            reference = scored.rmse_s
        rows.append(scored)
    table = ScoreTable(rows=tuple(rows), reference_name="symmetric_assumption")
    assert "skill is relative to symmetric_assumption" in table.as_text()
    assert table.by_name("median_filter").rmse_s > 0.0
    assert table.best().rmse_s == min(row.rmse_s for row in rows)
    assert table.ratio("median_filter", "mean_difference") > 0.0
    with pytest.raises(ConfigurationError, match="no scored estimator"):
        table.by_name("nope")


def test_score_predictions_validates() -> None:
    with pytest.raises(ConfigurationError, match="match in length"):
        score_predictions("x", np.zeros(3), np.zeros(4))
    with pytest.raises(ConfigurationError, match="at least 2 rows"):
        score_predictions("x", np.zeros(1), np.zeros(1))
    bad = np.zeros(4)
    bad[0] = np.inf
    with pytest.raises(ConfigurationError, match="finite"):
        score_predictions("x", bad, np.zeros(4))


def test_skill_is_nan_without_a_reference() -> None:
    scored = score_predictions("x", np.zeros(4), np.ones(4))
    assert np.isnan(scored.skill)
