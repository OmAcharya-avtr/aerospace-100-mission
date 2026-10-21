"""Inventory and DataProduct: units, validation, dependency structure."""

from __future__ import annotations

import pytest

from downlinkpick.inventory import (
    BITS_PER_BYTE,
    MIN_COMPRESSION_RATIO,
    DataProduct,
    Inventory,
)


def test_raw_bits_is_eight_times_bytes() -> None:
    # Hand calculation: 1.5e6 bytes * 8 = 1.2e7 bits.
    product = DataProduct("p", 1.5e6, 0.0, 100.0)
    assert product.raw_bits == pytest.approx(1.2e7, rel=1e-15)
    assert BITS_PER_BYTE == 8


def test_downlink_bits_applies_the_ratio() -> None:
    # Hand calculation: 2e6 bytes * 8 bits * 0.25 = 4e6 bits.
    product = DataProduct("p", 2.0e6, 0.0, 100.0, compressibility=0.25)
    assert product.downlink_bits() == pytest.approx(4.0e6, rel=1e-15)
    assert product.downlink_bits(0.5) == pytest.approx(8.0e6, rel=1e-15)


def test_samples_from_bit_depth_and_channels() -> None:
    # Hand calculation: 1200 bytes = 9600 bits; / (12 bits * 4 channels) = 200.
    product = DataProduct("p", 1200.0, 0.0, 100.0, bit_depth=12, channels=4)
    assert product.samples == pytest.approx(200.0, rel=1e-15)


@pytest.mark.parametrize(
    ("kwargs", "fragment"),
    [
        ({"raw_bytes": 0.0}, "raw_bytes must be strictly positive"),
        ({"raw_bytes": -1.0}, "raw_bytes must be strictly positive"),
        ({"age_s": -1.0}, "age_s must be non-negative"),
        ({"expiry_s": 0.0}, "expiry_s must be strictly positive"),
        ({"declared_value": -0.5}, "declared_value must be non-negative"),
        ({"compressibility": 0.0}, "compressibility must lie in"),
        ({"compressibility": 1.5}, "compressibility must lie in"),
        ({"entropy_bits_per_sample": -1.0}, "entropy_bits_per_sample must be non-negative"),
        ({"bit_depth": 0}, "bit_depth must be >= 1"),
        ({"channels": 0}, "channels must be >= 1"),
        ({"coverage_fraction": 1.2}, "coverage_fraction must lie in"),
        ({"novelty": -0.1}, "novelty must lie in"),
    ],
)
def test_invalid_product_fields_raise(kwargs: dict[str, object], fragment: str) -> None:
    base: dict[str, object] = {"name": "p", "raw_bytes": 1.0e6, "age_s": 0.0, "expiry_s": 10.0}
    base.update(kwargs)
    with pytest.raises(ValueError, match=fragment):
        DataProduct(**base)  # type: ignore[arg-type]


def test_empty_name_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        DataProduct("   ", 1.0e6, 0.0, 10.0)


def test_self_dependency_rejected() -> None:
    with pytest.raises(ValueError, match="may not depend on itself"):
        DataProduct("p", 1.0e6, 0.0, 10.0, depends_on=("p",))


def test_repeated_dependency_rejected() -> None:
    with pytest.raises(ValueError, match="repeated name"):
        DataProduct("p", 1.0e6, 0.0, 10.0, depends_on=("a", "a"))


def test_downlink_bits_without_a_ratio_raises() -> None:
    product = DataProduct("p", 1.0e6, 0.0, 10.0)
    with pytest.raises(ValueError, match="no compression ratio available"):
        product.downlink_bits()


def test_downlink_bits_rejects_an_out_of_range_ratio() -> None:
    product = DataProduct("p", 1.0e6, 0.0, 10.0)
    with pytest.raises(ValueError, match="compression ratio must lie in"):
        product.downlink_bits(1.1)
    with pytest.raises(ValueError, match="compression ratio must lie in"):
        product.downlink_bits(MIN_COMPRESSION_RATIO / 2.0)


def test_empty_inventory_rejected() -> None:
    with pytest.raises(ValueError, match="at least one DataProduct"):
        Inventory([])


def test_duplicate_name_rejected() -> None:
    product = DataProduct("p", 1.0e6, 0.0, 10.0)
    with pytest.raises(ValueError, match="duplicate product name"):
        Inventory([product, product])


def test_unresolved_dependency_rejected() -> None:
    with pytest.raises(ValueError, match="not in the inventory"):
        Inventory([DataProduct("p", 1.0e6, 0.0, 10.0, depends_on=("absent",))])


def test_dependency_cycle_rejected() -> None:
    products = [
        DataProduct("a", 1.0e6, 0.0, 10.0, depends_on=("b",)),
        DataProduct("b", 1.0e6, 0.0, 10.0, depends_on=("a",)),
    ]
    with pytest.raises(ValueError, match="dependency cycle"):
        Inventory(products)


def test_non_product_element_rejected() -> None:
    with pytest.raises(TypeError, match="not DataProduct"):
        Inventory(["not a product"])  # type: ignore[list-item]


def test_dependency_closure_is_transitive(tiny_inventory: Inventory) -> None:
    chain = Inventory(
        [
            DataProduct("a", 1.0e6, 0.0, 10.0),
            DataProduct("b", 1.0e6, 0.0, 10.0, depends_on=("a",)),
            DataProduct("c", 1.0e6, 0.0, 10.0, depends_on=("b",)),
        ]
    )
    assert chain.dependency_closure("c") == frozenset({"a", "b"})
    assert chain.dependency_closure("a") == frozenset()
    assert tiny_inventory.dependency_closure("img") == frozenset({"cal"})


def test_dependents_is_the_inverse_of_the_closure() -> None:
    chain = Inventory(
        [
            DataProduct("a", 1.0e6, 0.0, 10.0),
            DataProduct("b", 1.0e6, 0.0, 10.0, depends_on=("a",)),
            DataProduct("c", 1.0e6, 0.0, 10.0, depends_on=("b",)),
        ]
    )
    assert chain.dependents("a") == frozenset({"b", "c"})
    assert chain.dependents("c") == frozenset()


def test_topological_order_places_prerequisites_first() -> None:
    chain = Inventory(
        [
            DataProduct("c", 1.0e6, 0.0, 10.0, depends_on=("b",)),
            DataProduct("b", 1.0e6, 0.0, 10.0, depends_on=("a",)),
            DataProduct("a", 1.0e6, 0.0, 10.0),
        ]
    )
    assert chain.topological_order() == ("a", "b", "c")


def test_dependency_pairs_are_direct_only(tiny_inventory: Inventory) -> None:
    assert tiny_inventory.dependency_pairs() == (("img", "cal"),)


def test_lookup_and_membership(tiny_inventory: Inventory) -> None:
    assert tiny_inventory["cal"].declared_value == 5.0
    assert "cal" in tiny_inventory
    assert "absent" not in tiny_inventory
    assert len(tiny_inventory) == 3
    assert tiny_inventory.names == ("cal", "img", "bulk")
    assert tiny_inventory.position("bulk") == 2
    with pytest.raises(KeyError, match="no product named"):
        tiny_inventory["absent"]
    with pytest.raises(KeyError, match="no product named"):
        tiny_inventory.position("absent")


def test_missing_declarations_and_ratios() -> None:
    inventory = Inventory(
        [
            DataProduct("a", 1.0e6, 0.0, 10.0, declared_value=1.0, compressibility=0.5),
            DataProduct("b", 1.0e6, 0.0, 10.0),
        ]
    )
    assert inventory.missing_declarations() == ("b",)
    assert inventory.missing_compressibility() == ("b",)
    assert inventory.declared_values() == {"a": 1.0, "b": None}


def test_total_bits(tiny_inventory: Inventory) -> None:
    # Hand calculation: (1 + 2 + 3) MB * 8 = 4.8e7 bits raw.
    assert tiny_inventory.total_raw_bits() == pytest.approx(4.8e7, rel=1e-15)
    # Compressed: 4e6 + 4e6 + 2.16e7 = 2.96e7 bits.
    assert tiny_inventory.total_downlink_bits() == pytest.approx(2.96e7, rel=1e-15)
    ratios = {"cal": 1.0, "img": 1.0, "bulk": 1.0}
    assert tiny_inventory.total_downlink_bits(ratios) == pytest.approx(4.8e7, rel=1e-15)


def test_iteration_order_is_insertion_order(tiny_inventory: Inventory) -> None:
    assert [p.name for p in tiny_inventory] == ["cal", "img", "bulk"]
