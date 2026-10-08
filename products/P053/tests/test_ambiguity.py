"""Tests for the twin-invalidation-against-asset-fault demonstration."""

from __future__ import annotations

import numpy as np
import pytest

from twininvalidate import DetectorSpec, max_absolute_difference, paired_streams


@pytest.fixture(scope="module")
def pair():
    return paired_streams(offset=0.02, onset=200, n_runs=20, n_samples=700, seed=53400)


def test_the_two_worlds_produce_the_same_residual(pair):
    # Algebraically identical; the difference is pure floating-point rounding
    # because the two worlds add the offset at different points in the
    # expression. Residuals are of order 1, so 1e-10 is about 1e6 times
    # looser than the measured difference and still a meaningful assertion.
    diff = max_absolute_difference(pair)
    assert diff < 1e-10
    assert diff > 0.0  # it is rounding, not an accidental shared buffer


def test_the_two_worlds_are_not_the_same_array_object(pair):
    assert pair.asset_fault is not pair.twin_invalid


def test_both_worlds_are_in_control_before_the_onset(pair):
    pre = pair.asset_fault[:, : pair.onset]
    se = 1.0 / np.sqrt(pre.size)
    assert abs(float(pre.mean())) < 5.0 * se
    assert float(pre.std()) == pytest.approx(1.0, abs=0.05)


def test_the_offset_shifts_the_residual_mean(pair):
    post = pair.asset_fault[:, pair.onset + 100 :]
    assert float(post.mean()) > 0.5


def test_every_detector_gives_the_identical_verdict_in_both_worlds(pair):
    for name in ("cusum", "ewma", "glr", "varcusum"):
        spec = DetectorSpec(name)
        a = spec.statistic(pair.asset_fault)
        b = spec.statistic(pair.twin_invalid)
        assert np.allclose(a, b, rtol=1e-9, atol=1e-9)


def test_recorded_settings_are_returned(pair):
    assert pair.onset == 200
    assert pair.offset == pytest.approx(0.02)
    assert pair.asset_fault.shape == (20, 700)


@pytest.mark.parametrize("onset", [-1, 700, 1000])
def test_paired_streams_validates_the_onset(onset):
    with pytest.raises(ValueError, match="onset must lie"):
        paired_streams(onset=onset, n_samples=700, n_runs=2)


def test_paired_streams_rejects_zero_runs():
    with pytest.raises(ValueError, match="must be positive"):
        paired_streams(onset=0, n_runs=0, n_samples=100)


def test_paired_streams_rejects_a_zero_length_stream():
    # The onset check fires first, because an onset of 0 cannot lie inside an
    # empty stream; either message is a refusal to simulate nothing.
    with pytest.raises(ValueError, match="onset must lie"):
        paired_streams(onset=0, n_runs=2, n_samples=0)


def test_zero_offset_leaves_both_worlds_in_control():
    pair = paired_streams(offset=0.0, onset=100, n_runs=10, n_samples=400, seed=1)
    assert max_absolute_difference(pair) == 0.0
