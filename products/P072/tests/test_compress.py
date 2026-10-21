"""The compressibility estimators: the entropy floor, the table, the forest."""

from __future__ import annotations

import numpy as np
import pytest

from downlinkpick.compress import (
    DEFAULT_N_ESTIMATORS,
    GLOBAL_FALLBACK_RATIO,
    CompressibilityModel,
    entropy_floor_ratio,
    entropy_floor_ratios,
    global_mean_ratios,
    instrument_constant_ratios,
    instrument_constant_table,
    overflow_bits,
)
from downlinkpick.dataset import COMPRESSION_FEATURES, generate_instance
from downlinkpick.inventory import MIN_COMPRESSION_RATIO, DataProduct, Inventory


def test_entropy_floor_known_answer() -> None:
    # Hand calculation: 6 bits of entropy stored at 12 bits per sample gives 0.5.
    product = DataProduct("p", 1.0e6, 0.0, 10.0, entropy_bits_per_sample=6.0, bit_depth=12)
    assert entropy_floor_ratio(product) == pytest.approx(0.5, rel=1e-15)


def test_entropy_floor_is_clipped_at_one() -> None:
    product = DataProduct("p", 1.0e6, 0.0, 10.0, entropy_bits_per_sample=20.0, bit_depth=8)
    assert entropy_floor_ratio(product) == 1.0


def test_entropy_floor_is_clipped_below() -> None:
    product = DataProduct("p", 1.0e6, 0.0, 10.0, entropy_bits_per_sample=0.0, bit_depth=16)
    assert entropy_floor_ratio(product) == MIN_COMPRESSION_RATIO


def test_entropy_floor_ratios_covers_the_inventory() -> None:
    inventory = Inventory(
        [
            DataProduct("a", 1.0e6, 0.0, 10.0, entropy_bits_per_sample=4.0, bit_depth=16),
            DataProduct("b", 1.0e6, 0.0, 10.0, entropy_bits_per_sample=8.0, bit_depth=16),
        ]
    )
    assert entropy_floor_ratios(inventory) == {"a": pytest.approx(0.25), "b": pytest.approx(0.5)}


def test_instrument_constant_table_is_the_per_instrument_mean() -> None:
    table = instrument_constant_table(("x", "x", "y"), np.array([0.2, 0.4, 0.8]))
    assert table == {"x": pytest.approx(0.3), "y": pytest.approx(0.8)}


def test_instrument_constant_table_validation() -> None:
    with pytest.raises(ValueError, match="instrument labels against"):
        instrument_constant_table(("x",), np.array([0.1, 0.2]))
    with pytest.raises(ValueError, match="at least one row"):
        instrument_constant_table((), np.array([]))


def test_instrument_constant_ratios_fallback_and_strict() -> None:
    inventory = Inventory(
        [
            DataProduct("a", 1.0e6, 0.0, 10.0, instrument="known"),
            DataProduct("b", 1.0e6, 0.0, 10.0, instrument="novel"),
        ]
    )
    table = {"known": 0.4}
    ratios = instrument_constant_ratios(inventory, table)
    assert ratios == {"a": pytest.approx(0.4), "b": pytest.approx(GLOBAL_FALLBACK_RATIO)}
    with pytest.raises(ValueError, match="no constant-ratio entry for instrument\\(s\\): novel"):
        instrument_constant_ratios(inventory, table, strict=True)


def test_global_mean_ratios_is_one_value_for_everything() -> None:
    inventory = Inventory(
        [DataProduct("a", 1.0e6, 0.0, 10.0), DataProduct("b", 1.0e6, 0.0, 10.0)]
    )
    ratios = global_mean_ratios(inventory, np.array([0.2, 0.6]))
    assert ratios == {"a": pytest.approx(0.4), "b": pytest.approx(0.4)}


def test_fit_predict_and_range(training_table) -> None:
    model = CompressibilityModel(n_estimators=32).fit(
        training_table.compression_features, training_table.compression_target
    )
    assert model.fitted
    predicted = model.predict(training_table.compression_features[:50])
    assert np.all(predicted >= MIN_COMPRESSION_RATIO)
    assert np.all(predicted <= 1.0)


def test_prediction_before_fit_raises() -> None:
    model = CompressibilityModel(n_estimators=4)
    with pytest.raises(RuntimeError, match="fit must be called"):
        model.predict(np.zeros((1, len(COMPRESSION_FEATURES))))
    with pytest.raises(RuntimeError, match="fit must be called"):
        model.feature_importances()


def test_fit_validation() -> None:
    model = CompressibilityModel(n_estimators=4)
    with pytest.raises(ValueError, match="features must have shape"):
        model.fit(np.zeros((4, 2)), np.full(4, 0.5))
    with pytest.raises(ValueError, match="rows against"):
        model.fit(np.zeros((4, len(COMPRESSION_FEATURES))), np.full(3, 0.5))
    with pytest.raises(ValueError, match="must lie in"):
        model.fit(np.zeros((4, len(COMPRESSION_FEATURES))), np.full(4, 1.5))
    with pytest.raises(ValueError, match="must lie in"):
        model.fit(np.zeros((4, len(COMPRESSION_FEATURES))), np.zeros(4))


@pytest.mark.parametrize(
    ("kwargs", "fragment"),
    [
        ({"n_estimators": 1}, "n_estimators must be >= 2"),
        ({"min_samples_leaf": 0}, "min_samples_leaf must be >= 1"),
    ],
)
def test_construction_validation(kwargs: dict[str, int], fragment: str) -> None:
    with pytest.raises(ValueError, match=fragment):
        CompressibilityModel(**kwargs)  # type: ignore[arg-type]


def test_default_forest_size() -> None:
    assert DEFAULT_N_ESTIMATORS == 200
    assert CompressibilityModel().n_estimators == 200


def test_conservative_is_at_or_above_the_mean(training_table) -> None:
    model = CompressibilityModel(n_estimators=32).fit(
        training_table.compression_features, training_table.compression_target
    )
    instance = generate_instance(15, n_products=12)
    prediction = model.predict_inventory(instance.inventory, instance.autocorrelation)
    assert np.all(prediction.conservative >= prediction.ratio - 1e-12)
    assert np.all(prediction.conservative <= 1.0)
    assert np.all(prediction.dispersion >= 0.0)
    assert set(prediction.as_mapping()) == set(instance.inventory.names)
    assert set(prediction.conservative_mapping()) == set(instance.inventory.names)


def test_predict_inventory_requires_every_autocorrelation(training_table) -> None:
    model = CompressibilityModel(n_estimators=8).fit(
        training_table.compression_features, training_table.compression_target
    )
    instance = generate_instance(15, n_products=6)
    partial = dict(list(instance.autocorrelation.items())[:3])
    with pytest.raises(ValueError, match="no lag-one autocorrelation supplied"):
        model.predict_inventory(instance.inventory, partial)


def test_feature_importances_sum_to_one(training_table) -> None:
    model = CompressibilityModel(n_estimators=32).fit(
        training_table.compression_features, training_table.compression_target
    )
    importances = model.feature_importances()
    assert set(importances) == set(COMPRESSION_FEATURES)
    assert sum(importances.values()) == pytest.approx(1.0, rel=1e-9)


def test_forest_beats_the_floor_on_the_generator(training_table) -> None:
    split = 800
    model = CompressibilityModel(n_estimators=64).fit(
        training_table.compression_features[:split], training_table.compression_target[:split]
    )
    predicted = model.predict(training_table.compression_features[split:])
    truth = training_table.compression_target[split:]
    floor = training_table.entropy_floor[split:]
    assert np.abs(predicted - truth).mean() < np.abs(floor - truth).mean()


def test_overflow_bits_is_zero_where_it_fits() -> None:
    inventory = Inventory([DataProduct("a", 1.0e6, 0.0, 10.0)])
    overflow = overflow_bits(inventory, {"w1": 100.0, "w2": 300.0}, {"w1": 200.0, "w2": 200.0})
    assert overflow == {"w1": 0.0, "w2": 100.0}
