"""Frame-error channels: analytic statistics, construction, validation."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from arqlonghaul.channel import (
    GilbertElliottChannel,
    IndependentFrameChannel,
    ber_to_fer,
    bpsk_ber,
    fer_to_ber,
)


def test_independent_channel_marginal() -> None:
    ch = IndependentFrameChannel(0.1)
    errors = ch.errors(200_000, np.random.default_rng(0))
    # se = sqrt(0.1*0.9/200000) = 6.7e-4; 5 se is 3.4e-3.
    assert abs(errors.mean() - 0.1) < 3.4e-3
    assert errors.dtype == bool
    assert errors.size == 200_000


def test_independent_channel_run_length() -> None:
    assert IndependentFrameChannel(0.0).mean_burst_slots == 1.0
    assert IndependentFrameChannel(0.5).mean_burst_slots == pytest.approx(2.0)
    assert IndependentFrameChannel(1.0).mean_burst_slots == math.inf


def test_independent_channel_autocorrelation() -> None:
    ch = IndependentFrameChannel(0.2)
    assert ch.autocorrelation(0) == 1.0
    assert ch.autocorrelation(1) == 0.0
    assert ch.autocorrelation(50) == 0.0


@pytest.mark.parametrize("fer", [-0.01, 1.01])
def test_independent_channel_rejects_bad_fer(fer: float) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        IndependentFrameChannel(fer)


def test_independent_channel_rejects_zero_slots() -> None:
    with pytest.raises(ValueError, match="n_slots"):
        IndependentFrameChannel(0.1).errors(0, np.random.default_rng(0))


def test_ge_stationary_quantities_hand_calculated() -> None:
    # p_gb = 0.01, p_bg = 0.04  =>  pi_b = 0.01/0.05 = 0.2
    # eps_g = 0, eps_b = 1      =>  p_bar = 0.2, burst = 1/0.04 = 25
    ch = GilbertElliottChannel(p_gb=0.01, p_bg=0.04, eps_g=0.0, eps_b=1.0)
    assert ch.pi_bad == pytest.approx(0.2)
    assert ch.mean_fer == pytest.approx(0.2)
    assert ch.mean_burst_slots == pytest.approx(25.0)
    assert ch.mean_good_slots == pytest.approx(100.0)
    assert ch.lam == pytest.approx(0.95)


def test_ge_autocorrelation_at_lag_zero_and_decay() -> None:
    ch = GilbertElliottChannel(p_gb=0.01, p_bg=0.04, eps_g=0.0, eps_b=1.0)
    assert ch.autocorrelation(0) == 1.0
    # With eps_g=0 and eps_b=1 the indicator IS the state, so rho_k = lam**k.
    for lag in (1, 5, 20):
        assert ch.autocorrelation(lag) == pytest.approx(ch.lam**lag, rel=1e-12)
    with pytest.raises(ValueError, match="lag"):
        ch.autocorrelation(-1)


def test_from_mean_and_burst_inverts_exactly() -> None:
    for fer in (0.01, 0.05, 0.2, 0.45):
        for burst in (1.0, 2.5, 25.0, 400.0):
            ch = GilbertElliottChannel.from_mean_and_burst(fer, burst)
            assert ch.mean_fer == pytest.approx(fer, rel=1e-12)
            assert ch.mean_burst_slots == pytest.approx(burst, rel=1e-12)


def test_from_mean_and_burst_rejects_impossible_requests() -> None:
    with pytest.raises(ValueError, match="mean_burst_slots"):
        GilbertElliottChannel.from_mean_and_burst(0.1, 0.5)
    with pytest.raises(ValueError, match="strictly between"):
        GilbertElliottChannel.from_mean_and_burst(0.0, 10.0)
    with pytest.raises(ValueError, match="strictly between"):
        GilbertElliottChannel.from_mean_and_burst(1.0, 10.0)
    with pytest.raises(ValueError, match=r"P\(G->B\)"):
        # Asking for a 90 per cent error rate with a 1-slot burst needs p_gb > 1.
        GilbertElliottChannel.from_mean_and_burst(0.9, 1.0)


def test_ge_rejects_bad_parameters() -> None:
    with pytest.raises(ValueError, match="p_gb"):
        GilbertElliottChannel(p_gb=0.0, p_bg=0.1)
    with pytest.raises(ValueError, match="p_bg"):
        GilbertElliottChannel(p_gb=0.1, p_bg=1.5)
    with pytest.raises(ValueError, match="eps_g"):
        GilbertElliottChannel(p_gb=0.1, p_bg=0.1, eps_g=-0.1)
    with pytest.raises(ValueError, match="Bad state is the worse one"):
        GilbertElliottChannel(p_gb=0.1, p_bg=0.1, eps_g=0.9, eps_b=0.1)


def test_ge_sampled_marginal_within_correlated_standard_error() -> None:
    ch = GilbertElliottChannel.from_mean_and_burst(0.1, 25.0)
    errors = ch.errors(400_000, np.random.default_rng(3))
    p = ch.mean_fer
    iid_se = math.sqrt(p * (1 - p) / errors.size)
    corr_se = iid_se * math.sqrt((1 + ch.lam) / (1 - ch.lam))
    assert abs(errors.mean() - p) < 5 * corr_se


def test_ge_state_trace_duty_cycle() -> None:
    ch = GilbertElliottChannel.from_mean_and_burst(0.2, 25.0)
    states = ch.state_trace(400_000, np.random.default_rng(5))
    assert abs(states.mean() - ch.pi_bad) < 0.03
    with pytest.raises(ValueError, match="n_slots"):
        ch.state_trace(0, np.random.default_rng(0))


def test_bpsk_ber_known_values() -> None:
    # 0.5*erfc(1) at Es/N0 = 0 dB
    assert bpsk_ber(0.0) == pytest.approx(0.078649603525, rel=1e-9)
    # monotone decreasing, bounded by 0.5
    assert bpsk_ber(-30.0) < 0.5
    assert bpsk_ber(10.0) < bpsk_ber(5.0) < bpsk_ber(0.0)


def test_ber_to_fer_hand_calculated() -> None:
    # 1 - (1 - 0.001)**1000 = 0.632304...
    assert ber_to_fer(1e-3, 1000) == pytest.approx(1.0 - 0.999**1000, rel=1e-12)
    assert ber_to_fer(0.0, 1000) == 0.0
    assert ber_to_fer(1.0, 1000) == 1.0


def test_ber_fer_roundtrip() -> None:
    for fer in (1e-6, 0.01, 0.3, 0.9):
        ber = fer_to_ber(fer, 8920)
        assert ber_to_fer(ber, 8920) == pytest.approx(fer, rel=1e-9)


def test_ber_fer_validation() -> None:
    with pytest.raises(ValueError, match="ber"):
        ber_to_fer(-0.1, 10)
    with pytest.raises(ValueError, match="frame_bits"):
        ber_to_fer(0.1, 0)
    with pytest.raises(ValueError, match="fer"):
        fer_to_ber(1.0, 10)
    with pytest.raises(ValueError, match="frame_bits"):
        fer_to_ber(0.1, -1)


@given(
    fer=st.floats(min_value=1e-4, max_value=0.45),
    burst=st.floats(min_value=1.0, max_value=500.0),
)
@settings(max_examples=250, deadline=None)
def test_from_mean_and_burst_property(fer: float, burst: float) -> None:
    ch = GilbertElliottChannel.from_mean_and_burst(fer, burst)
    assert ch.mean_fer == pytest.approx(fer, rel=1e-9)
    assert ch.mean_burst_slots == pytest.approx(burst, rel=1e-9)
    assert 0.0 < ch.pi_bad < 1.0
    assert ch.autocorrelation(1) <= 1.0 + 1e-12
