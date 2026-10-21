"""The seeded generator: determinism, shape, the knobs and the ground truth."""

from __future__ import annotations

import numpy as np
import pytest

from downlinkpick.dataset import (
    COMPRESSION_FEATURES,
    INSTRUMENT_CODE,
    INSTRUMENTS,
    VALUE_FEATURES,
    compression_feature_row,
    generate_instance,
    generate_plan,
    generate_training_table,
    instance_family,
    value_feature_row,
)
from downlinkpick.selection import sizes_from_ratios


def test_same_seed_gives_an_identical_instance() -> None:
    first = generate_instance(7, n_products=12)
    second = generate_instance(7, n_products=12)
    assert first.inventory.products == second.inventory.products
    assert first.plan.windows == second.plan.windows
    assert first.true_values == second.true_values
    assert first.true_ratios == second.true_ratios


def test_different_seeds_give_different_instances() -> None:
    assert generate_instance(1, n_products=12).inventory.products != (
        generate_instance(2, n_products=12).inventory.products
    )


def test_shape_follows_the_arguments() -> None:
    instance = generate_instance(3, n_products=17, n_windows=4)
    assert len(instance.inventory) == 17
    assert len(instance.plan) == 4
    assert set(instance.true_values) == set(instance.inventory.names)
    assert set(instance.autocorrelation) == set(instance.inventory.names)


def test_capacity_fraction_sets_the_plan_capacity() -> None:
    for fraction in (0.2, 0.45, 0.9):
        instance = generate_instance(
            11, n_products=20, n_windows=3, capacity_fraction=fraction
        )
        queue = sum(sizes_from_ratios(instance.inventory, instance.true_ratios).values())
        assert instance.plan.total_capacity_bits() == pytest.approx(
            fraction * queue, rel=1e-9
        )


def test_capacity_fraction_none_uses_the_peak_rate() -> None:
    instance = generate_instance(
        5, n_products=8, n_windows=2, capacity_fraction=None, peak_rate_bps=1.0e8
    )
    assert instance.plan.windows[0].peak_rate_bps <= 1.4e8


def test_missing_fractions_are_honoured() -> None:
    instance = generate_instance(
        13,
        n_products=20,
        missing_declared_fraction=0.5,
        missing_ratio_fraction=0.25,
    )
    assert len(instance.inventory.missing_declarations()) == 10
    assert len(instance.inventory.missing_compressibility()) == 5


def test_stale_fraction_is_honoured_among_the_declared() -> None:
    instance = generate_instance(
        13,
        n_products=20,
        missing_declared_fraction=0.5,
        stale_declared_fraction=0.6,
    )
    # Ten declarations remain; 60 per cent of them are stale.
    assert len(instance.stale) == 6
    assert all(name not in instance.inventory.missing_declarations() for name in instance.stale)


def test_a_stale_declaration_ignores_the_decay() -> None:
    instance = generate_instance(
        21, n_products=30, stale_declared_fraction=1.0, declared_noise=0.0
    )
    older = [
        name
        for name in instance.inventory.names
        if instance.inventory[name].age_s > 5.0e4
    ]
    assert older, "the generator should produce some aged products"
    # A stale declaration is the value at acquisition, so for an aged product it
    # exceeds the realised value, which has decayed.
    for name in older:
        declared = instance.inventory[name].declared_value
        assert declared is not None
        assert declared > instance.true_values[name]


def test_a_fresh_declaration_tracks_the_realised_value() -> None:
    instance = generate_instance(
        22, n_products=40, stale_declared_fraction=0.0, declared_noise=0.0
    )
    for name in instance.inventory.names:
        declared = instance.inventory[name].declared_value
        assert declared is not None
        assert declared == pytest.approx(instance.true_values[name], rel=1e-12)


def test_dependencies_are_acyclic_and_backwards_only() -> None:
    instance = generate_instance(9, n_products=30, dependency_fraction=0.8)
    order = instance.inventory.names
    for product in instance.inventory:
        for prerequisite in product.depends_on:
            assert order.index(prerequisite) < order.index(product.name)


def test_zero_dependency_fraction_gives_no_dependencies() -> None:
    instance = generate_instance(9, n_products=30, dependency_fraction=0.0)
    assert all(product.depends_on == () for product in instance.inventory)


def test_true_ratios_are_in_range() -> None:
    instance = generate_instance(31, n_products=40)
    ratios = np.array(list(instance.true_ratios.values()))
    assert np.all(ratios >= 0.02)
    assert np.all(ratios <= 1.0)


def test_true_values_are_non_negative() -> None:
    instance = generate_instance(32, n_products=40)
    assert all(value >= 0.0 for value in instance.true_values.values())


@pytest.mark.parametrize(
    ("kwargs", "fragment"),
    [
        ({"n_products": 0}, "n_products must be >= 1"),
        ({"missing_declared_fraction": 1.5}, "missing_declared_fraction must lie in"),
        ({"stale_declared_fraction": -0.1}, "stale_declared_fraction must lie in"),
        ({"missing_ratio_fraction": 2.0}, "missing_ratio_fraction must lie in"),
        ({"dependency_fraction": -1.0}, "dependency_fraction must lie in"),
        ({"declared_noise": -0.1}, "declared_noise must be non-negative"),
        ({"capacity_fraction": 0.0}, "capacity_fraction must be strictly positive"),
    ],
)
def test_generator_validation(kwargs: dict[str, object], fragment: str) -> None:
    with pytest.raises(ValueError, match=fragment):
        generate_instance(1, **kwargs)  # type: ignore[arg-type]


def test_generate_plan_validation() -> None:
    with pytest.raises(ValueError, match="n_windows must be >= 1"):
        generate_plan(1, n_windows=0)


def test_generate_plan_windows_do_not_overlap() -> None:
    for seed in range(1, 30):
        plan = generate_plan(seed, n_windows=5)
        ends = [w.end_s for w in plan]
        starts = [w.start_s for w in plan]
        assert all(start >= end for start, end in zip(starts[1:], ends[:-1], strict=False))


def test_training_table_shape_and_determinism() -> None:
    first = generate_training_table(200, 77)
    second = generate_training_table(200, 77)
    assert first.value_features.shape == (200, len(VALUE_FEATURES))
    assert first.compression_features.shape == (200, len(COMPRESSION_FEATURES))
    assert first.value_target.shape == (200,)
    assert first.compression_target.shape == (200,)
    assert len(first) == 200
    assert np.array_equal(first.value_features, second.value_features)
    assert np.array_equal(first.compression_target, second.compression_target)
    assert len(set(first.names)) == 200


def test_training_table_validation() -> None:
    with pytest.raises(ValueError, match="n_rows must be >= 1"):
        generate_training_table(0, 1)


def test_training_targets_are_in_range(training_table) -> None:
    assert np.all(training_table.value_target >= 0.0)
    assert np.all(training_table.compression_target > 0.0)
    assert np.all(training_table.compression_target <= 1.0)
    assert np.all(training_table.entropy_floor > 0.0)
    assert np.all(training_table.entropy_floor <= 1.0)


def test_entropy_floor_is_above_the_realised_ratio_on_average(training_table) -> None:
    # Equation (7) of the generator applies a correlation gain below the
    # memoryless floor of equation (6), so the floor over-predicts on average.
    assert training_table.entropy_floor.mean() > training_table.compression_target.mean()


def test_instruments_are_distinct_and_coded() -> None:
    names = [instrument.name for instrument in INSTRUMENTS]
    assert len(set(names)) == len(names)
    assert set(INSTRUMENT_CODE) == set(names)
    assert sorted(INSTRUMENT_CODE.values()) == list(range(len(names)))


def test_feature_rows_have_the_declared_width() -> None:
    instance = generate_instance(2, n_products=5)
    product = instance.inventory.products[0]
    assert len(value_feature_row(product, instance.inventory)) == len(VALUE_FEATURES)
    assert len(compression_feature_row(product, 0.5)) == len(COMPRESSION_FEATURES)


def test_value_feature_row_counts_dependents() -> None:
    instance = generate_instance(9, n_products=20, dependency_fraction=1.0)
    for product in instance.inventory:
        row = value_feature_row(product, instance.inventory)
        assert row[8] == float(len(instance.inventory.dependents(product.name)))


def test_compression_feature_row_entropy_ratio_is_clipped() -> None:
    instance = generate_instance(2, n_products=5)
    product = instance.inventory.products[0]
    row = compression_feature_row(product, 0.5)
    assert 0.0 < row[2] <= 1.0


def test_instance_family_maps_seeds() -> None:
    family = instance_family((1, 2, 3), n_products=6)
    assert [instance.seed for instance in family] == [1, 2, 3]
    assert len(family[0].inventory) == 6
