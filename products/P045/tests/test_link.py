"""Link geometry: known answers, identities, and input validation."""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from arqlonghaul.link import PRESETS, SPEED_OF_LIGHT_M_S, LinkParams, preset


def test_frame_time_hand_calculated() -> None:
    # 8920 bits at 1 Mbit/s is 8.92 ms exactly.
    link = LinkParams(rate_bps=1e6, rtt_s=0.0, frame_bits=8920)
    assert link.frame_time_s == pytest.approx(8.92e-3, rel=1e-12)


def test_n_hand_calculated() -> None:
    # T_f = 8.92 ms, RTT = 2.6144 s  =>  N = 1 + 2.6144/0.00892 = 294.0987...
    link = LinkParams(rate_bps=1e6, rtt_s=2.6144, frame_bits=8920)
    assert link.slots_per_cycle == pytest.approx(1.0 + 2.6144 / 8.92e-3, rel=1e-12)
    assert link.n_slots() == 294


def test_bdp_hand_calculated() -> None:
    # 2 Mbit/s for 0.25 s is 500000 bits = 62500 bytes = 56.0538... frames.
    link = LinkParams(rate_bps=2e6, rtt_s=0.25, frame_bits=8920)
    assert link.bdp_bits == pytest.approx(5e5)
    assert link.bdp_bytes == pytest.approx(62_500.0)
    assert link.bdp_frames == pytest.approx(5e5 / 8920)


def test_zero_rtt_gives_n_one() -> None:
    link = LinkParams(rate_bps=1e6, rtt_s=0.0, frame_bits=1000)
    assert link.slots_per_cycle == 1.0
    assert link.n_slots() == 1
    assert link.min_continuous_window == 1


def test_payload_defaults_to_frame_bits() -> None:
    link = LinkParams(rate_bps=1e6, rtt_s=1.0, frame_bits=1000)
    assert link.payload == 1000
    assert link.code_overhead == 0.0


def test_code_overhead() -> None:
    link = LinkParams(rate_bps=1e6, rtt_s=1.0, frame_bits=1000, payload_bits=800)
    assert link.code_overhead == pytest.approx(0.2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rate_bps": 0.0, "rtt_s": 1.0, "frame_bits": 100},
        {"rate_bps": -1.0, "rtt_s": 1.0, "frame_bits": 100},
        {"rate_bps": 1e6, "rtt_s": -0.1, "frame_bits": 100},
        {"rate_bps": 1e6, "rtt_s": 1.0, "frame_bits": 0},
        {"rate_bps": 1e6, "rtt_s": 1.0, "frame_bits": -5},
        {"rate_bps": 1e6, "rtt_s": 1.0, "frame_bits": 100, "payload_bits": 0},
        {"rate_bps": 1e6, "rtt_s": 1.0, "frame_bits": 100, "payload_bits": 101},
    ],
)
def test_invalid_inputs_raise(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        LinkParams(**kwargs)


def test_window_bytes_rejects_nonpositive() -> None:
    link = LinkParams(rate_bps=1e6, rtt_s=1.0, frame_bits=800)
    assert link.window_bytes(10) == pytest.approx(1000.0)
    with pytest.raises(ValueError):
        link.window_bytes(0)


def test_unknown_preset_raises() -> None:
    with pytest.raises(ValueError, match="unknown preset"):
        preset("jupiter")


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_presets_self_consistent(name: str) -> None:
    link = preset(name)
    assert link.bdp_frames == pytest.approx(link.slots_per_cycle - 1.0, rel=1e-12)
    assert link.min_continuous_window == max(1, math.ceil(link.slots_per_cycle))
    d = link.describe()
    assert d["bdp_bytes"] == pytest.approx(d["bdp_bits"] / 8.0)


def test_presets_rtt_at_least_two_way_light_time() -> None:
    # The GEO preset's RTT must exceed two-way light time to 35786 km.
    link = preset("geo")
    assert link.rtt_s > 2.0 * 35_786e3 / SPEED_OF_LIGHT_M_S


@given(
    rate=st.floats(min_value=1e3, max_value=1e10),
    rtt=st.floats(min_value=0.0, max_value=1e4),
    bits=st.integers(min_value=8, max_value=100_000),
)
@settings(max_examples=200, deadline=None)
def test_bdp_frames_equals_n_minus_one(rate: float, rtt: float, bits: int) -> None:
    link = LinkParams(rate_bps=rate, rtt_s=rtt, frame_bits=bits)
    assert link.bdp_frames == pytest.approx(link.slots_per_cycle - 1.0, rel=1e-9, abs=1e-12)
