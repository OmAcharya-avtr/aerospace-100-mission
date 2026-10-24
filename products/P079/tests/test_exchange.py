"""The four-timestamp record and the exchange set, including its JSON form."""

from __future__ import annotations

import numpy as np
import pytest

from timetransfer.errors import ConfigurationError
from timetransfer.exchange import Exchange, ExchangeSet


def test_exchange_derived_quantities_known_answer() -> None:
    # Hand calculation, all in nanoseconds:
    #   t1 = 0, t2 = 5_000_000, t3 = 5_100_000, t4 = 10_200_000
    #   forward dwell = t2 - t1          = 5_000_000
    #   reverse dwell = t4 - t3          = 5_100_000
    #   round trip    = (t4-t1)-(t3-t2)  = 10_200_000 - 100_000 = 10_100_000
    #   midpoint      = (t1+t4)//2       = 5_100_000
    ex = Exchange(seq=0, t1_ns=0, t2_ns=5_000_000, t3_ns=5_100_000, t4_ns=10_200_000)
    assert ex.forward_dwell_ns == 5_000_000
    assert ex.reverse_dwell_ns == 5_100_000
    assert ex.round_trip_ns == 10_100_000
    assert ex.midpoint_ns == 5_100_000


def test_exchange_rejects_reversed_timestamps() -> None:
    with pytest.raises(ConfigurationError, match="reply arrived before"):
        Exchange(seq=1, t1_ns=100, t2_ns=50, t3_ns=60, t4_ns=10)
    with pytest.raises(ConfigurationError, match="remote replied before"):
        Exchange(seq=1, t1_ns=0, t2_ns=100, t3_ns=50, t4_ns=200)


def _set(n: int = 4) -> ExchangeSet:
    base = np.arange(n, dtype=np.int64) * 1_000_000_000
    return ExchangeSet(
        t1_ns=base,
        t2_ns=base + 5_000_000,
        t3_ns=base + 5_100_000,
        t4_ns=base + 10_200_000,
        seq=np.arange(n, dtype=np.int64),
        n_lost=1,
        backend_kind="simulated",
    )


def test_exchange_set_shapes_and_counts() -> None:
    s = _set(4)
    assert s.n == 4
    assert s.n_attempted == 5
    assert s.loss_fraction == pytest.approx(0.2)
    assert s.span_s() == pytest.approx(3.0102, rel=1e-9)


def test_exchange_set_derived_arrays_are_exact_integers() -> None:
    s = _set(3)
    assert s.forward_dwell_ns().tolist() == [5_000_000] * 3
    assert s.reverse_dwell_ns().tolist() == [5_100_000] * 3
    assert s.round_trip_ns().tolist() == [10_100_000] * 3


def test_exchange_set_rejects_unequal_lengths() -> None:
    with pytest.raises(ConfigurationError, match="unequal lengths"):
        ExchangeSet(
            t1_ns=np.zeros(3, dtype=np.int64),
            t2_ns=np.zeros(2, dtype=np.int64),
            t3_ns=np.zeros(3, dtype=np.int64),
            t4_ns=np.zeros(3, dtype=np.int64),
            seq=np.arange(3, dtype=np.int64),
        )


def test_exchange_set_rejects_out_of_order_arrays() -> None:
    with pytest.raises(ConfigurationError, match="precedes"):
        ExchangeSet(
            t1_ns=np.asarray([0, 100], dtype=np.int64),
            t2_ns=np.asarray([10, 110], dtype=np.int64),
            t3_ns=np.asarray([20, 120], dtype=np.int64),
            t4_ns=np.asarray([30, 50], dtype=np.int64),
            seq=np.arange(2, dtype=np.int64),
        )


def test_exchange_set_rejects_a_truth_column_of_the_wrong_length() -> None:
    with pytest.raises(ConfigurationError, match="true_offset_s"):
        ExchangeSet(
            t1_ns=np.zeros(3, dtype=np.int64),
            t2_ns=np.zeros(3, dtype=np.int64),
            t3_ns=np.zeros(3, dtype=np.int64),
            t4_ns=np.zeros(3, dtype=np.int64),
            seq=np.arange(3, dtype=np.int64),
            true_offset_s=np.zeros(2),
        )


def test_subset_carries_truth_and_identity() -> None:
    s = _set(6)
    s.true_offset_s = np.arange(6, dtype=np.float64)
    s.true_asymmetry_s = np.arange(6, dtype=np.float64) * 2.0
    sub = s.subset(np.asarray([0, 2, 4]))
    assert sub.n == 3
    assert sub.true_offset_s is not None
    assert sub.true_offset_s.tolist() == [0.0, 2.0, 4.0]
    assert sub.backend_kind == "simulated"


def test_json_round_trip_is_byte_identical(tmp_path) -> None:
    s = _set(5)
    path = s.write_json(tmp_path / "set.json")
    back = ExchangeSet.read_json(path)
    assert back.timestamp_bytes() == s.timestamp_bytes()
    assert back.digest() == s.digest()


def test_from_dict_rejects_an_unknown_format() -> None:
    payload = _set(3).to_dict()
    payload["format"] = "something/else"
    with pytest.raises(ConfigurationError, match="unknown exchange-set format"):
        ExchangeSet.from_dict(payload)


def test_digest_changes_with_the_hardware_flag() -> None:
    a = _set(3)
    b = _set(3)
    b.is_hardware = True
    assert a.digest() != b.digest()
