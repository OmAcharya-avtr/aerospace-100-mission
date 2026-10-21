"""The learned value model, its uncertainty output, and the baselines."""

from __future__ import annotations

import numpy as np
import pytest

from downlinkpick.dataset import VALUE_FEATURES, generate_instance, value_feature_row
from downlinkpick.inventory import DataProduct, Inventory
from downlinkpick.valuemodel import (
    DEFAULT_N_ESTIMATORS,
    SizeAgeHeuristic,
    ValueModel,
    declared_values,
    fit_size_age_heuristic,
    hybrid_values,
    size_age_values,
    spearman,
)


def test_fit_and_predict_shapes(training_table) -> None:
    model = ValueModel(n_estimators=24).fit(
        training_table.value_features, training_table.value_target
    )
    assert model.fitted
    predicted = model.predict(training_table.value_features[:10])
    assert predicted.shape == (10,)
    assert np.all(predicted >= 0.0)


def test_prediction_before_fit_raises() -> None:
    model = ValueModel(n_estimators=4)
    assert not model.fitted
    with pytest.raises(RuntimeError, match="fit must be called"):
        model.predict(np.zeros((1, len(VALUE_FEATURES))))
    with pytest.raises(RuntimeError, match="fit must be called"):
        model.feature_importances()


def test_two_fits_with_the_same_seed_agree(training_table) -> None:
    first = ValueModel(n_estimators=24, seed=5).fit(
        training_table.value_features, training_table.value_target
    )
    second = ValueModel(n_estimators=24, seed=5).fit(
        training_table.value_features, training_table.value_target
    )
    assert np.allclose(
        first.predict(training_table.value_features[:50]),
        second.predict(training_table.value_features[:50]),
        rtol=0.0,
        atol=0.0,
    )


def test_dispersion_is_non_negative_and_mean_matches_predict(training_table) -> None:
    model = ValueModel(n_estimators=32).fit(
        training_table.value_features, training_table.value_target
    )
    mean, dispersion = model.predict_with_dispersion(training_table.value_features[:40])
    assert np.all(dispersion >= 0.0)
    assert np.allclose(mean, model.predict(training_table.value_features[:40]), rtol=1e-12)


def test_predict_inventory_returns_a_usable_mapping(training_table) -> None:
    model = ValueModel(n_estimators=32).fit(
        training_table.value_features, training_table.value_target
    )
    instance = generate_instance(4, n_products=9)
    prediction = model.predict_inventory(instance.inventory)
    assert prediction.names == instance.inventory.names
    assert set(prediction.as_mapping()) == set(instance.inventory.names)
    assert set(prediction.dispersion_mapping()) == set(instance.inventory.names)
    assert np.all(prediction.lower <= prediction.value + 1e-12)
    assert np.all(prediction.upper >= prediction.value - 1e-12)
    assert np.all(prediction.lower >= 0.0)


def test_feature_importances_sum_to_one(training_table) -> None:
    model = ValueModel(n_estimators=32).fit(
        training_table.value_features, training_table.value_target
    )
    importances = model.feature_importances()
    assert set(importances) == set(VALUE_FEATURES)
    assert sum(importances.values()) == pytest.approx(1.0, rel=1e-9)


def test_model_beats_a_constant_on_the_generator(training_table) -> None:
    split = 800
    model = ValueModel(n_estimators=64).fit(
        training_table.value_features[:split], training_table.value_target[:split]
    )
    predicted = model.predict(training_table.value_features[split:])
    truth = training_table.value_target[split:]
    constant = np.full(truth.shape, training_table.value_target[:split].mean())
    assert np.abs(predicted - truth).mean() < np.abs(constant - truth).mean()


@pytest.mark.parametrize(
    ("kwargs", "fragment"),
    [
        ({"n_estimators": 1}, "n_estimators must be >= 2"),
        ({"min_samples_leaf": 0}, "min_samples_leaf must be >= 1"),
    ],
)
def test_model_construction_validation(kwargs: dict[str, int], fragment: str) -> None:
    with pytest.raises(ValueError, match=fragment):
        ValueModel(**kwargs)  # type: ignore[arg-type]


def test_fit_validation() -> None:
    model = ValueModel(n_estimators=4)
    with pytest.raises(ValueError, match="features must have shape"):
        model.fit(np.zeros((4, 3)), np.zeros(4))
    with pytest.raises(ValueError, match="rows against"):
        model.fit(np.zeros((4, len(VALUE_FEATURES))), np.zeros(3))
    with pytest.raises(ValueError, match="non-negative"):
        model.fit(np.zeros((4, len(VALUE_FEATURES))), -np.ones(4))


def test_default_forest_size() -> None:
    assert DEFAULT_N_ESTIMATORS == 300
    assert ValueModel().n_estimators == 300


def test_declared_values_fallbacks() -> None:
    inventory = Inventory(
        [
            DataProduct("a", 1.0e6, 0.0, 10.0, declared_value=2.0),
            DataProduct("b", 1.0e6, 0.0, 10.0, declared_value=6.0),
            DataProduct("c", 1.0e6, 0.0, 10.0),
        ]
    )
    assert declared_values(inventory, fallback="median")["c"] == pytest.approx(4.0)
    assert declared_values(inventory, fallback="mean")["c"] == pytest.approx(4.0)
    assert declared_values(inventory, fallback="zero")["c"] == 0.0
    assert declared_values(inventory)["a"] == 2.0
    with pytest.raises(ValueError, match="unknown fallback"):
        declared_values(inventory, fallback="nonsense")


def test_declared_values_with_no_declaration_anywhere_falls_back_to_zero() -> None:
    inventory = Inventory([DataProduct("a", 1.0e6, 0.0, 10.0)])
    assert declared_values(inventory, fallback="median")["a"] == 0.0


def test_size_age_heuristic_known_answer() -> None:
    heuristic = SizeAgeHeuristic(intercept=1.0, log_bytes_coefficient=2.0, age_coefficient=-0.5)
    # Hand calculation: 1 + 2 * 3 - 0.5 * 2 = 6.0.
    assert heuristic.predict(np.array([3.0]), np.array([2.0]))[0] == pytest.approx(6.0)
    # Clipped at zero below.
    assert heuristic.predict(np.array([0.0]), np.array([100.0]))[0] == 0.0


def test_size_age_heuristic_recovers_an_exact_linear_target() -> None:
    rng = np.random.default_rng(0)
    features = np.zeros((200, len(VALUE_FEATURES)))
    features[:, 0] = rng.uniform(5.0, 9.0, 200)
    features[:, 1] = rng.uniform(0.0, 1.0e5, 200)
    target = 3.0 + 1.5 * features[:, 0] - 2.0e-5 * features[:, 1]
    fitted = fit_size_age_heuristic(features, np.clip(target, 0.0, None))
    assert fitted.intercept == pytest.approx(3.0, abs=1e-8)
    assert fitted.log_bytes_coefficient == pytest.approx(1.5, abs=1e-9)
    assert fitted.age_coefficient == pytest.approx(-2.0e-5, abs=1e-12)


def test_size_age_fit_validation() -> None:
    with pytest.raises(ValueError, match="features must have shape"):
        fit_size_age_heuristic(np.zeros((4, 2)), np.zeros(4))


def test_size_age_values_covers_the_inventory(training_table) -> None:
    heuristic = fit_size_age_heuristic(
        training_table.value_features, training_table.value_target
    )
    instance = generate_instance(6, n_products=7)
    values = size_age_values(heuristic, instance.inventory)
    assert set(values) == set(instance.inventory.names)
    assert all(value >= 0.0 for value in values.values())


def test_hybrid_keeps_declarations_and_fills_gaps() -> None:
    inventory = Inventory(
        [
            DataProduct("a", 1.0e6, 0.0, 10.0, declared_value=2.0),
            DataProduct("b", 1.0e6, 0.0, 10.0),
        ]
    )
    hybrid = hybrid_values(inventory, {"a": 99.0, "b": 7.0})
    assert hybrid == {"a": 2.0, "b": 7.0}
    with pytest.raises(ValueError, match="no learned prediction for: b"):
        hybrid_values(inventory, {"a": 1.0})


def test_spearman_identities() -> None:
    assert spearman([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)
    assert spearman([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)
    # A monotone rescaling preserves ranks, so the correlation is unchanged.
    assert spearman([1.0, 2.0, 3.0], [10.0, 40.0, 90.0]) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="shape mismatch"):
        spearman([1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="at least two points"):
        spearman([1.0], [1.0])


def test_value_feature_row_is_what_the_model_consumes(training_table) -> None:
    model = ValueModel(n_estimators=16).fit(
        training_table.value_features, training_table.value_target
    )
    instance = generate_instance(8, n_products=6)
    rows = np.array(
        [value_feature_row(p, instance.inventory) for p in instance.inventory], dtype=float
    )
    assert rows.shape == (6, len(VALUE_FEATURES))
    direct = model.predict(rows)
    through = model.predict_inventory(instance.inventory).value
    assert np.allclose(direct, through, rtol=1e-12)
