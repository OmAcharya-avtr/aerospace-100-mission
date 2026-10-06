"""Window sizing: the knee, marginal gain, and the honest go-back-N gap."""

from __future__ import annotations

import math

import numpy as np
import pytest

from arqlonghaul import closedform as cf
from arqlonghaul.link import LinkParams, preset
from arqlonghaul.window import marginal_gain, size_window, window_sweep


def test_size_window_hand_calculated() -> None:
    # 1 Mbit/s, RTT 0.5 s, 1000-bit frames: T_f = 1 ms, N = 501,
    # BDP = 500000 bits = 62500 bytes = 500 frames, knee = 501 frames.
    link = LinkParams(rate_bps=1e6, rtt_s=0.5, frame_bits=1000)
    s = size_window(link, 0.05)
    assert s.n == pytest.approx(501.0)
    assert s.bdp_bits == pytest.approx(5e5)
    assert s.bdp_frames == pytest.approx(500.0)
    assert s.knee_frames == 501
    assert s.knee_bytes == pytest.approx(501 * 125.0)
    assert s.sr_goodput_at_knee == pytest.approx(0.95)


def test_size_window_gbn_is_far_below_sr_on_a_long_link() -> None:
    s = size_window(preset("lunar"), 0.05)
    assert s.gbn_goodput_at_knee < 0.1 * s.sr_goodput_at_knee
    d = s.as_dict()
    assert d["sr_bps_at_knee"] > d["gbn_bps_at_knee"]


def test_size_window_rejects_bad_fer() -> None:
    with pytest.raises(ValueError, match="fer"):
        size_window(preset("geo"), 1.0)


def test_marginal_gain_is_zero_above_the_knee() -> None:
    assert marginal_gain(60.0, 0.05, 60) == 0.0
    assert marginal_gain(60.0, 0.05, 120) == 0.0
    assert marginal_gain(60.0, 0.05, 30) == pytest.approx(0.95 / 60.0)


def test_marginal_gain_matches_a_finite_difference() -> None:
    for n in (10.0, 60.0, 300.0):
        for p in (0.0, 0.05, 0.4):
            for w in (1, int(n // 3), int(math.ceil(n)), int(2 * n)):
                fd = cf.sr_throughput(p, n, w + 1) - cf.sr_throughput(p, n, w)
                assert marginal_gain(n, p, w) == pytest.approx(fd, abs=1e-12)


def test_marginal_gain_validation() -> None:
    with pytest.raises(ValueError, match="window"):
        marginal_gain(10.0, 0.1, 0)
    with pytest.raises(ValueError, match="fer"):
        marginal_gain(10.0, 1.0, 5)


def test_window_sweep_marks_gbn_undefined_below_the_knee() -> None:
    out = window_sweep(60.0, 0.05, np.array([1, 10, 59, 60, 120]))
    assert np.isnan(out["go_back_n"][:3]).all()
    assert np.isfinite(out["go_back_n"][3:]).all()
    assert out["selective_repeat"][-1] == pytest.approx(0.95)
    assert out["selective_repeat"][0] < out["selective_repeat"][2]


def test_window_sweep_validation() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        window_sweep(60.0, 0.05, np.array([]))
    with pytest.raises(ValueError, match=">= 1"):
        window_sweep(60.0, 0.05, np.array([0.5, 2.0]))
