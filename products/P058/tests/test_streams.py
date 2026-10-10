"""Stream generator tests: determinism, declared moments, change placement."""

from __future__ import annotations

import numpy as np
import pytest

from telemdrift.streams import (
    CHANGE_TYPES,
    ChangeSpec,
    ar1_stationary,
    change_stream,
    stationary,
    transient_spike,
)


def test_stationary_is_deterministic_given_seed():
    a = stationary(1000, 7)
    b = stationary(1000, 7)
    assert np.array_equal(a, b)


def test_stationary_differs_between_seeds():
    assert not np.array_equal(stationary(1000, 7), stationary(1000, 8))


def test_stationary_moments_within_sampling_error():
    # n = 200000, so the standard error of the mean is 1/sqrt(2e5) = 2.236e-3.
    # A 5-sigma band is 1.118e-2; the test uses 0.02 to stay robust.
    x = stationary(200_000, 11)
    assert abs(x.mean()) < 0.02
    assert abs(x.std() - 1.0) < 0.02


def test_stationary_respects_sigma():
    x = stationary(200_000, 12, sigma=3.0)
    assert abs(x.std() - 3.0) < 0.06


def test_stationary_accepts_a_generator():
    g = np.random.default_rng(5)
    a = stationary(10, g)
    b = stationary(10, g)
    assert not np.array_equal(a, b)  # the generator advances


@pytest.mark.parametrize("bad", [0, -1])
def test_stationary_rejects_nonpositive_length(bad):
    with pytest.raises(ValueError, match="length must be >= 1"):
        stationary(bad, 1)


def test_stationary_rejects_nonpositive_sigma():
    with pytest.raises(ValueError, match="sigma must be > 0"):
        stationary(10, 1, sigma=0.0)


def test_ar1_has_unit_marginal_variance_and_declared_autocorrelation():
    phi = 0.8
    x = ar1_stationary(200_000, 13, phi=phi)
    assert abs(x.std() - 1.0) < 0.03
    lag1 = np.corrcoef(x[:-1], x[1:])[0, 1]
    assert abs(lag1 - phi) < 0.02


def test_ar1_with_phi_zero_matches_iid_variance():
    x = ar1_stationary(100_000, 14, phi=0.0)
    lag1 = np.corrcoef(x[:-1], x[1:])[0, 1]
    assert abs(lag1) < 0.02


@pytest.mark.parametrize("phi", [1.0, -1.0, 1.5])
def test_ar1_rejects_nonstationary_phi(phi):
    with pytest.raises(ValueError, match="abs\\(phi\\) < 1"):
        ar1_stationary(10, 1, phi=phi)


def test_change_spec_rejects_unknown_kind():
    with pytest.raises(ValueError, match="kind must be one of"):
        ChangeSpec("not_a_change", 1.0)


def test_change_spec_rejects_nonpositive_variance_multiplier():
    with pytest.raises(ValueError, match="sigma multiplier"):
        ChangeSpec("variance_step", 0.0)


def test_change_spec_rejects_zero_transient_duration():
    with pytest.raises(ValueError, match="duration must be >= 1"):
        ChangeSpec("transient", 1.0, duration=0)


def test_change_index_is_the_first_post_change_sample():
    stream, idx = change_stream(100, 50, ChangeSpec("mean_step", 100.0), 3)
    assert idx == 100
    assert len(stream) == 150
    # A 100-sigma step is unmistakable: every post-change sample is far above
    # every plausible pre-change sample.
    assert stream[idx] > 50.0
    assert stream[idx - 1] < 50.0


def test_mean_step_shifts_only_the_post_segment():
    stream, idx = change_stream(20_000, 20_000, ChangeSpec("mean_step", 1.0), 4)
    assert abs(stream[:idx].mean()) < 0.03
    assert abs(stream[idx:].mean() - 1.0) < 0.03


def test_variance_step_scales_only_the_post_segment():
    stream, idx = change_stream(20_000, 20_000, ChangeSpec("variance_step", 3.0), 5)
    assert abs(stream[:idx].std() - 1.0) < 0.04
    assert abs(stream[idx:].std() - 3.0) < 0.12


def test_drift_ramp_slope_matches_the_declared_magnitude():
    slope = 0.01
    stream, idx = change_stream(100, 5_000, ChangeSpec("drift_ramp", slope), 6)
    post = stream[idx:]
    t = np.arange(1, post.size + 1, dtype=float)
    # Least-squares slope through the post-change segment; the noise contributes
    # a standard error of sigma / sqrt(sum (t - tbar)^2) = 1 / sqrt(1.04e10)
    # = 9.8e-6, so a 1e-3 tolerance is generous by two orders of magnitude.
    fitted = np.polyfit(t, post, 1)[0]
    assert abs(fitted - slope) < 1e-3


def test_transient_returns_to_baseline():
    amp, dur = 5.0, 30
    stream, idx = change_stream(100, 2_000, ChangeSpec("transient", amp, dur), 7)
    inside = stream[idx : idx + dur]
    after = stream[idx + dur :]
    assert abs(inside.mean() - amp) < 1.0
    assert abs(after.mean()) < 0.1


def test_transient_spike_wrapper_matches_change_stream():
    a, ia = transient_spike(50, 200, 3.0, 10, 8)
    b, ib = change_stream(50, 200, ChangeSpec("transient", 3.0, 10), 8)
    assert ia == ib
    assert np.array_equal(a, b)


def test_transient_duration_is_clipped_to_the_post_segment():
    stream, idx = change_stream(10, 5, ChangeSpec("transient", 10.0, 50), 9)
    assert len(stream) == 15
    assert (stream[idx:] > 5.0).all()


@pytest.mark.parametrize("pre,post", [(0, 10), (10, 0), (-1, 10)])
def test_change_stream_rejects_empty_segments(pre, post):
    with pytest.raises(ValueError, match=">= 1"):
        change_stream(pre, post, ChangeSpec("mean_step", 1.0), 1)


def test_change_types_tuple_is_the_documented_four():
    assert CHANGE_TYPES == ("mean_step", "variance_step", "drift_ramp", "transient")
