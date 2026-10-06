"""Capacity and photons per bit: closed forms, the PPM limit, data processing."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from photoncount import capacity as cap
from photoncount.ppm import PPMConfig


def test_erasure_capacity_known_answer():
    # M = 16, n_s = 2: (1 - exp(-2)) * 4 = 0.8646647168 * 4 = 3.4586588670
    out = cap.erasure_channel_capacity(PPMConfig(16, 2.0))
    assert out["bits_per_symbol"] == pytest.approx(3.4586588671, rel=1e-9)
    assert out["bits_per_slot"] == pytest.approx(3.4586588671 / 16, rel=1e-9)
    assert out["erasure_probability"] == pytest.approx(0.1353352832, rel=1e-9)


def test_erasure_capacity_limits():
    # Huge signal: capacity saturates at log2(M).
    assert cap.erasure_channel_capacity(PPMConfig(16, 50.0))["bits_per_symbol"] == pytest.approx(
        4.0, rel=1e-12
    )
    # Vanishing signal: capacity goes to zero.
    assert cap.erasure_channel_capacity(PPMConfig(16, 1e-9))["bits_per_symbol"] < 1e-8


def test_minimum_photons_per_bit_known_answers():
    assert cap.minimum_photons_per_bit(16) == 0.25
    assert cap.minimum_photons_per_bit(2) == 1.0
    assert cap.minimum_photons_per_bit(1024) == 0.1


def test_photons_per_bit_approaches_the_limit_from_above():
    """(C5): photons/bit -> 1/log2(M) as n_s -> 0, and never goes below it."""
    order = 16
    limit = cap.minimum_photons_per_bit(order)
    values = []
    for ns in (1.0, 0.1, 0.01, 1e-3, 1e-4):
        cfg = PPMConfig(order, ns)
        c = cap.erasure_channel_capacity(cfg)["bits_per_symbol"]
        values.append(cap.photons_per_bit(cfg, c))
    assert all(v > limit for v in values)
    assert all(b < a for a, b in zip(values, values[1:], strict=False))
    assert values[-1] == pytest.approx(limit, rel=2e-4)


def test_bits_per_photon_is_unbounded_in_order():
    """log2(M) grows without bound, so bits per photon has no finite ceiling."""
    bits_per_photon = [1.0 / cap.minimum_photons_per_bit(m) for m in (2, 16, 256, 4096)]
    assert bits_per_photon == pytest.approx([1.0, 4.0, 8.0, 12.0])
    assert all(b > a for a, b in zip(bits_per_photon, bits_per_photon[1:], strict=False))


def test_hard_decision_capacity_is_below_log2_m():
    cfg = PPMConfig(16, 4.0, 0.2)
    out = cap.hard_decision_capacity(cfg)
    assert 0.0 < out["bits_per_symbol"] < 4.0
    assert out["symbol_error_probability"] > 0.0


def test_hard_decision_capacity_saturates_on_a_clean_channel():
    cfg = PPMConfig(16, 60.0, 1e-9)
    assert cap.hard_decision_capacity(cfg)["bits_per_symbol"] == pytest.approx(4.0, abs=1e-9)


def test_soft_decision_rate_exceeds_hard_decision_rate():
    """Data-processing inequality: hard-deciding can only lose information."""
    for order, ns, nb in ((16, 4.0, 0.2), (8, 2.0, 0.5), (4, 1.0, 0.3)):
        cfg = PPMConfig(order, ns, nb)
        hard = cap.hard_decision_capacity(cfg)["bits_per_symbol"]
        soft = cap.soft_decision_achievable_rate(cfg, 40_000, np.random.default_rng(5))
        assert soft["bits_per_symbol"] > hard - 4 * soft["standard_error"]


def test_soft_decision_rate_is_below_log2_m():
    cfg = PPMConfig(16, 4.0, 0.2)
    soft = cap.soft_decision_achievable_rate(cfg, 40_000, np.random.default_rng(5))
    assert soft["bits_per_symbol"] < 4.0
    assert soft["standard_error"] > 0.0


def test_soft_decision_requires_background():
    with pytest.raises(ValueError, match="erasure"):
        cap.soft_decision_achievable_rate(PPMConfig(16, 2.0), 1000, np.random.default_rng(0))


def test_erasure_capacity_requires_no_background():
    with pytest.raises(ValueError, match="requires n_b"):
        cap.erasure_channel_capacity(PPMConfig(16, 2.0, 0.1))


def test_bad_rate_in_photons_per_bit_raises():
    cfg = PPMConfig(16, 2.0)
    for bad in (0.0, -1.0, float("inf")):
        with pytest.raises(ValueError):
            cap.photons_per_bit(cfg, bad)


def test_soft_decision_rejects_tiny_sample():
    with pytest.raises(ValueError):
        cap.soft_decision_achievable_rate(
            PPMConfig(16, 2.0, 0.1), 1, np.random.default_rng(0)
        )


@given(
    st.integers(min_value=2, max_value=256),
    st.floats(min_value=0.01, max_value=20.0),
)
@settings(max_examples=40, deadline=None)
def test_photons_per_bit_never_below_the_limit(order, ns):
    cfg = PPMConfig(order, ns)
    c = cap.erasure_channel_capacity(cfg)["bits_per_symbol"]
    assert cap.photons_per_bit(cfg, c) >= cap.minimum_photons_per_bit(order) - 1e-12


@given(
    st.integers(min_value=2, max_value=64),
    st.floats(min_value=0.5, max_value=20.0),
    st.floats(min_value=0.01, max_value=2.0),
)
@settings(max_examples=30, deadline=None)
def test_hard_decision_capacity_is_bounded(order, ns, nb):
    out = cap.hard_decision_capacity(PPMConfig(order, ns, nb))
    assert 0.0 <= out["bits_per_symbol"] <= np.log2(order) + 1e-12
