"""Closed-form throughput: hand-calculated answers, identities, monotonicity."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from arqlonghaul.closedform import (
    expected_transmissions,
    gbn_throughput,
    slots_per_cycle,
    sr_throughput,
    sr_window_sweep,
    sw_throughput,
    throughput,
    window_knee,
)


def test_slots_per_cycle_hand_calculated() -> None:
    # RTT = 0.5 s, T_f = 0.01 s  =>  N = 1 + 50 = 51
    assert slots_per_cycle(0.5, 0.01) == pytest.approx(51.0)


def test_sw_hand_calculated() -> None:
    # (1 - 0.1) / 20 = 0.045 exactly
    assert sw_throughput(0.1, 20) == pytest.approx(0.045, rel=1e-12)


def test_gbn_hand_calculated() -> None:
    # (1-0.1) / (1 - 0.1 + 20*0.1) = 0.9 / 2.9 = 0.310344827586...
    assert gbn_throughput(0.1, 20) == pytest.approx(0.9 / 2.9, rel=1e-12)


def test_sr_hand_calculated() -> None:
    assert sr_throughput(0.1, 20) == pytest.approx(0.9, rel=1e-12)
    # window limited: (1-0.1) * 5/20 = 0.225
    assert sr_throughput(0.1, 20, 5) == pytest.approx(0.225, rel=1e-12)
    # at and above the knee: 1 - p
    assert sr_throughput(0.1, 20, 20) == pytest.approx(0.9, rel=1e-12)
    assert sr_throughput(0.1, 20, 200) == pytest.approx(0.9, rel=1e-12)


def test_error_free_channel_gives_unity_for_sliding_windows() -> None:
    assert gbn_throughput(0.0, 100) == pytest.approx(1.0)
    assert sr_throughput(0.0, 100) == pytest.approx(1.0)
    assert sw_throughput(0.0, 100) == pytest.approx(0.01)


def test_expected_transmissions_hand_calculated() -> None:
    assert expected_transmissions(0.0) == 1.0
    assert expected_transmissions(0.5) == pytest.approx(2.0)
    assert expected_transmissions(0.9) == pytest.approx(10.0)


def test_sw_is_the_window_one_case_of_sr() -> None:
    for n in (1, 5, 60, 1000):
        for p in (0.0, 0.01, 0.3, 0.9):
            assert sr_throughput(p, n, 1) == pytest.approx(
                sw_throughput(p, n), rel=1e-12, abs=1e-18
            )


def test_sr_dominates_gbn_dominates_sw_above_the_knee() -> None:
    n, p = 60, 0.05
    assert sr_throughput(p, n) > gbn_throughput(p, n) > sw_throughput(p, n)


def test_gbn_refuses_a_window_below_the_knee() -> None:
    with pytest.raises(ValueError, match="continuous transmission"):
        gbn_throughput(0.05, 60, 30)


def test_gbn_accepts_a_window_at_the_knee() -> None:
    assert gbn_throughput(0.05, 60.0, 60) == pytest.approx(gbn_throughput(0.05, 60.0))


@pytest.mark.parametrize("p", [-0.01, 1.0, 1.5])
def test_bad_p_raises(p: float) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        sw_throughput(p, 10)


@pytest.mark.parametrize("n", [0.0, 0.5, -3.0])
def test_bad_n_raises(n: float) -> None:
    with pytest.raises(ValueError, match="N must be"):
        sw_throughput(0.1, n)


@pytest.mark.parametrize("w", [0, -1, 2.5])
def test_bad_window_raises(w: object) -> None:
    with pytest.raises(ValueError, match="window"):
        sr_throughput(0.1, 10, w)  # type: ignore[arg-type]


def test_bad_frame_time_raises() -> None:
    with pytest.raises(ValueError, match="frame_time_s"):
        slots_per_cycle(1.0, 0.0)
    with pytest.raises(ValueError, match="rtt_s"):
        slots_per_cycle(-1.0, 0.1)


def test_throughput_dispatch() -> None:
    assert throughput("stop_and_wait", 0.1, 20) == sw_throughput(0.1, 20)
    assert throughput("go_back_n", 0.1, 20) == gbn_throughput(0.1, 20)
    assert throughput("selective_repeat", 0.1, 20, 5) == sr_throughput(0.1, 20, 5)
    with pytest.raises(ValueError, match="unknown protocol"):
        throughput("alternating_bit", 0.1, 20)


def test_window_knee_is_n() -> None:
    assert window_knee(37.5) == 37.5


def test_sr_window_sweep_matches_scalar() -> None:
    import numpy as np

    windows = np.array([1, 4, 20, 60, 400])
    got = sr_window_sweep(0.07, 20.0, windows)
    for w, g in zip(windows, got, strict=True):
        assert g == pytest.approx(sr_throughput(0.07, 20.0, int(w)))
    with pytest.raises(ValueError, match=">= 1"):
        sr_window_sweep(0.07, 20.0, np.array([0, 1]))


@given(
    p=st.floats(min_value=0.0, max_value=0.95),
    n=st.floats(min_value=1.0, max_value=1e4),
)
@settings(max_examples=300, deadline=None)
def test_ordering_and_bounds(p: float, n: float) -> None:
    sw = sw_throughput(p, n)
    gbn = gbn_throughput(p, n)
    sr = sr_throughput(p, n)
    assert 0.0 <= sw <= 1.0
    assert 0.0 <= gbn <= 1.0
    assert 0.0 <= sr <= 1.0
    assert sw <= gbn + 1e-12
    assert gbn <= sr + 1e-12


@given(
    p=st.floats(min_value=0.0, max_value=0.9),
    n=st.floats(min_value=1.0, max_value=1e3),
    w=st.integers(min_value=1, max_value=5000),
)
@settings(max_examples=300, deadline=None)
def test_sr_is_monotone_nondecreasing_in_window(p: float, n: float, w: int) -> None:
    assert sr_throughput(p, n, w + 1) >= sr_throughput(p, n, w) - 1e-15


@given(
    p1=st.floats(min_value=0.0, max_value=0.8),
    extra=st.floats(min_value=1e-3, max_value=0.19),
    n=st.floats(min_value=1.0, max_value=1e3),
)
@settings(max_examples=200, deadline=None)
def test_all_three_decrease_in_p(p1: float, extra: float, n: float) -> None:
    p2 = p1 + extra
    assert sw_throughput(p2, n) <= sw_throughput(p1, n) + 1e-15
    assert gbn_throughput(p2, n) <= gbn_throughput(p1, n) + 1e-15
    assert sr_throughput(p2, n) <= sr_throughput(p1, n) + 1e-15
