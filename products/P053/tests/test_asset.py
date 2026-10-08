"""Tests for change injection and the seeded residual streams."""

from __future__ import annotations

import numpy as np
import pytest

from twininvalidate import (
    CHANGE_KINDS,
    AssetChange,
    StreamSpec,
    no_change,
    reference_twin,
    simulate_residuals,
)


def test_no_change_is_in_control():
    assert no_change().kind == "none"
    assert not no_change().is_change


@pytest.mark.parametrize("kind", CHANGE_KINDS)
def test_every_declared_kind_simulates(kind):
    magnitude = 2.0 if kind == "noise_variance" else -0.01
    spec = StreamSpec(
        change=AssetChange(kind=kind, onset=10, magnitude=magnitude),
        n_runs=4,
        n_samples=60,
        seed=1,
    )
    z = simulate_residuals(spec)
    assert z.shape == (4, 60)
    assert np.all(np.isfinite(z))


def test_simulation_is_deterministic_for_one_spec():
    spec = StreamSpec(change=AssetChange(), n_runs=5, n_samples=80, seed=4242)
    a = simulate_residuals(spec)
    b = simulate_residuals(spec)
    assert np.array_equal(a, b)


def test_different_seeds_give_different_streams():
    a = simulate_residuals(StreamSpec(change=AssetChange(), n_runs=5, n_samples=80, seed=1))
    b = simulate_residuals(StreamSpec(change=AssetChange(), n_runs=5, n_samples=80, seed=2))
    assert not np.allclose(a, b)


def test_in_control_residual_is_standard_normal(in_control):
    # Innovations property (Kailath 1968): z is i.i.d. N(0, 1) when the asset
    # obeys the declared model. 120 x 2000 = 240000 samples, so the standard
    # error of the mean is 1/sqrt(240000) = 0.00204; 4 standard errors is
    # 0.0082.
    n = in_control.size
    se = 1.0 / np.sqrt(n)
    assert abs(in_control.mean()) < 4.0 * se
    assert in_control.std() == pytest.approx(1.0, abs=4.0 * se)


def test_in_control_residual_is_white(in_control):
    # Lag-1 and lag-2 sample autocorrelations, same 4-sigma criterion. This is
    # the test that caught the missing A factor in the predictor gain.
    se = 1.0 / np.sqrt(in_control.size)
    lag1 = float(np.mean(in_control[:, :-1] * in_control[:, 1:]))
    lag2 = float(np.mean(in_control[:, :-2] * in_control[:, 2:]))
    assert abs(lag1) < 4.0 * se
    assert abs(lag2) < 4.0 * se


def test_parameter_step_mean_shift_matches_the_closed_form(stepped):
    # Equation (10): E[z] = C (I - (A - K C))^{-1} B delta u0 / sqrt(S) with
    # delta = -0.01 and u0 = 1.0 (the constant part of the excitation).
    twin = reference_twin()
    filt = twin.steady_state()
    m = np.eye(twin.n_states) - (twin.A - filt.K @ twin.C)
    x_ss = np.linalg.solve(m, (twin.B * -0.01).ravel())
    predicted = float((twin.C @ x_ss)[0] / np.sqrt(filt.S))
    assert predicted == pytest.approx(-0.4043, abs=5e-4)
    # Measured after the prediction-error transient has settled (500 samples,
    # spectral radius 0.917, so 0.917**500 is about 1e-19).
    measured = float(stepped[:, 500:].mean())
    assert measured == pytest.approx(predicted, rel=0.03)


def test_parameter_step_leaves_the_variance_alone(stepped):
    # A mean shift, not a scale change: the post-transient variance is within
    # 2 % of 1.
    assert float(stepped[:, 500:].var()) == pytest.approx(1.0, abs=0.02)


def test_noise_variance_change_leaves_the_mean_alone():
    z = simulate_residuals(
        StreamSpec(
            change=AssetChange("noise_variance", onset=0, magnitude=2.0),
            n_runs=120,
            n_samples=1500,
            seed=53102,
        )
    )
    post = z[:, 500:]
    se = 1.0 / np.sqrt(post.size)
    assert abs(float(post.mean())) < 4.0 * se
    # but the variance does move, by about 6 %
    assert float(post.var()) > 1.03


def test_slow_ramp_mean_grows_monotonically():
    z = simulate_residuals(
        StreamSpec(
            change=AssetChange("slow_ramp", onset=0, magnitude=-0.01, ramp_samples=400),
            n_runs=200,
            n_samples=600,
            seed=53101,
        )
    )
    blocks = [float(z[:, a : a + 100].mean()) for a in (100, 200, 300)]
    assert blocks[0] > blocks[1] > blocks[2]


def test_sensor_offset_shifts_the_mean():
    z = simulate_residuals(
        StreamSpec(
            change=AssetChange("sensor_offset", onset=200, magnitude=0.02),
            n_runs=40,
            n_samples=800,
            seed=5,
        )
    )
    assert abs(float(z[:, :200].mean())) < 0.1
    assert float(z[:, 400:].mean()) > 0.5


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"kind": "nonsense"}, "kind must be one of"),
        ({"kind": "none", "onset": -1}, "onset must be non-negative"),
        ({"kind": "none", "ramp_samples": 0}, "ramp_samples must be positive"),
        ({"kind": "noise_variance", "magnitude": 0.0}, "variance multiplier"),
        ({"kind": "noise_variance", "magnitude": -1.0}, "variance multiplier"),
        ({"kind": "parameter_step", "magnitude": -1.0}, "would reverse or"),
        ({"kind": "slow_ramp", "magnitude": -2.0}, "would reverse or"),
    ],
)
def test_asset_change_validates_its_inputs(kwargs, message):
    with pytest.raises(ValueError, match=message):
        AssetChange(**kwargs)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"n_runs": 0}, "n_runs must be positive"),
        ({"n_samples": 0}, "n_samples must be positive"),
        ({"burn_in": -1}, "burn_in must be non-negative"),
    ],
)
def test_stream_spec_validates_its_inputs(kwargs, message):
    base = {"change": AssetChange(), "n_runs": 2, "n_samples": 10, "seed": 1}
    base.update(kwargs)
    with pytest.raises(ValueError, match=message):
        StreamSpec(**base)


def test_stream_spec_rejects_an_onset_past_the_end():
    with pytest.raises(ValueError, match="at or past the end"):
        StreamSpec(
            change=AssetChange("parameter_step", onset=50, magnitude=-0.01),
            n_runs=2,
            n_samples=50,
        )
