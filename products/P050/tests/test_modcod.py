"""MODCOD and MODCOD-set construction, ordering and validation."""

from __future__ import annotations

import numpy as np
import pytest

from coderateopt import Modcod, ModcodSet, illustrative_modcod_table


def test_net_rate_applies_overhead():
    # 2.0 bits/symbol with 12.5 % framing overhead -> 2.0 * 0.875 = 1.75
    assert Modcod("x", 2.0, 5.0, overhead_fraction=0.125).net_rate == pytest.approx(1.75)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"name": "", "rate_bits_per_symbol": 1.0, "threshold_db": 0.0}, "non-empty"),
        ({"name": "  ", "rate_bits_per_symbol": 1.0, "threshold_db": 0.0}, "non-empty"),
        ({"name": "a", "rate_bits_per_symbol": 0.0, "threshold_db": 0.0}, "rate_bits_per_symbol"),
        ({"name": "a", "rate_bits_per_symbol": -1.0, "threshold_db": 0.0}, "rate_bits_per_symbol"),
        (
            {"name": "a", "rate_bits_per_symbol": float("nan"), "threshold_db": 0.0},
            "rate_bits_per_symbol",
        ),
        ({"name": "a", "rate_bits_per_symbol": 1.0, "threshold_db": float("inf")}, "threshold_db"),
        (
            {
                "name": "a",
                "rate_bits_per_symbol": 1.0,
                "threshold_db": 0.0,
                "overhead_fraction": 1.0,
            },
            "overhead_fraction",
        ),
        (
            {
                "name": "a",
                "rate_bits_per_symbol": 1.0,
                "threshold_db": 0.0,
                "overhead_fraction": -0.1,
            },
            "overhead_fraction",
        ),
    ],
)
def test_modcod_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        Modcod(**kwargs)


def test_set_is_sorted_canonically():
    unsorted = ModcodSet.from_rows(
        [("high", 3.0, 12.0), ("low", 1.0, 3.0), ("mid", 2.0, 7.0)]
    )
    assert unsorted.names == ("low", "mid", "high")
    assert list(unsorted.thresholds_db) == [3.0, 7.0, 12.0]


def test_equal_thresholds_order_by_descending_rate_then_name():
    table = ModcodSet.from_rows(
        [("b", 1.0, 5.0), ("a", 1.0, 5.0), ("c", 2.0, 5.0)]
    )
    assert table.names == ("c", "a", "b")


def test_set_rejects_empty_and_duplicates():
    with pytest.raises(ValueError, match="at least one"):
        ModcodSet(())
    with pytest.raises(ValueError, match="duplicate"):
        ModcodSet.from_rows([("a", 1.0, 1.0), ("a", 2.0, 2.0)])


def test_index_lookup_and_error():
    table = illustrative_modcod_table()
    assert table.index("ook-r1/2") == 1
    assert table[table.index("ook-r1/2")].name == "ook-r1/2"
    with pytest.raises(KeyError, match="no MODCOD named"):
        table.index("nope")


def test_illustrative_table_shape():
    table = illustrative_modcod_table()
    assert len(table) == 9
    assert np.all(np.diff(table.thresholds_db) > 0.0)
    assert np.all(np.diff(table.rates) > 0.0)
    assert len(list(iter(table))) == 9
