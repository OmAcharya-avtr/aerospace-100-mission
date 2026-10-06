"""PPM slot statistics: exact error probability, soft metrics, bit LLRs."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from photoncount import ppm


def test_background_free_symbol_error_known_answer():
    """P_e = exp(-n_s)(M-1)/M exactly: either a count arrives or the receiver guesses.

    M = 16, n_s = 2: exp(-2) * 15/16 = 0.1353352832 * 0.9375 = 0.1268768280
    """
    cfg = ppm.PPMConfig(16, 2.0)
    exact = ppm.symbol_error_probability(cfg)
    assert exact["error"] == pytest.approx(0.1268768280, rel=1e-9)
    assert exact["correct"] + exact["error"] == pytest.approx(1.0)


@pytest.mark.parametrize(("m", "ns"), [(2, 0.5), (4, 1.0), (16, 2.0), (64, 8.0), (16, 0.5)])
def test_background_free_matches_closed_form(m, ns):
    cfg = ppm.PPMConfig(m, ns)
    closed = float(np.exp(-ns) * (m - 1) / m)
    assert ppm.symbol_error_probability(cfg)["error"] == pytest.approx(closed, abs=1e-12)


@pytest.mark.parametrize(
    ("m", "ns", "nb"), [(16, 4.0, 0.2), (8, 3.0, 1.0), (256, 10.0, 0.01), (4, 1.0, 0.5)]
)
def test_exact_error_probability_agrees_with_monte_carlo(m, ns, nb):
    cfg = ppm.PPMConfig(m, ns, nb)
    exact = ppm.symbol_error_probability(cfg)["error"]
    mc = ppm.symbol_error_probability_mc(cfg, 200_000, np.random.default_rng(42))
    z = abs(mc["error"] - exact) / max(mc["standard_error"], 1e-12)
    assert z < 4.0, f"exact {exact} vs MC {mc['error']} +- {mc['standard_error']} (z={z})"


def test_erasure_probability_known_answer():
    # Background-free: P(all slots empty) = exp(-n_s) = exp(-2) = 0.1353352832
    assert ppm.erasure_probability(ppm.PPMConfig(16, 2.0)) == pytest.approx(
        0.1353352832, rel=1e-9
    )
    # With background the signal slot is not the only way to get a count.
    cfg = ppm.PPMConfig(16, 2.0, 0.1)
    assert ppm.erasure_probability(cfg) == pytest.approx(float(np.exp(-(2.0 + 1.6))), rel=1e-12)


def test_erasure_channel_probabilities_requires_background_free():
    probs = ppm.erasure_channel_probabilities(ppm.PPMConfig(8, 3.0))
    assert probs["error"] == 0.0
    assert probs["erasure"] == pytest.approx(float(np.exp(-3.0)))
    assert probs["correct"] + probs["erasure"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="background-free"):
        ppm.erasure_channel_probabilities(ppm.PPMConfig(8, 3.0, 0.1))


def test_slot_metric_scale_known_answer():
    # n_s = 4, n_0 = 0.2: ln(1 + 20) = ln(21) = 3.0445224
    cfg = ppm.PPMConfig(16, 4.0, 0.2)
    assert ppm.slot_metric_scale(cfg) == pytest.approx(3.0445224377, rel=1e-9)


def test_slot_metric_scale_diverges_without_background():
    with pytest.raises(ValueError, match="background-free"):
        ppm.slot_metric_scale(ppm.PPMConfig(16, 4.0))


def test_soft_metric_is_linear_in_the_count():
    cfg = ppm.PPMConfig(8, 2.0, 0.3)
    scale = ppm.slot_metric_scale(cfg)
    counts = np.array([0, 1, 2, 7])
    assert np.allclose(ppm.slot_metrics(counts, cfg), counts * scale)


def test_posterior_normalises_and_peaks_at_the_largest_count():
    cfg = ppm.PPMConfig(8, 3.0, 0.2)
    counts = np.array([[0, 1, 0, 5, 0, 2, 0, 0]])
    log_post = ppm.symbol_log_posterior(counts, cfg)
    assert float(np.exp(log_post).sum()) == pytest.approx(1.0, abs=1e-12)
    assert int(np.argmax(log_post)) == 3


def test_hard_decision_is_argmax_and_breaks_ties_uniformly():
    cfg = ppm.PPMConfig(4, 2.0, 0.1)
    counts = np.tile(np.array([3, 1, 0, 3]), (4000, 1))
    chosen = ppm.hard_decision(counts, cfg, np.random.default_rng(1))
    assert set(np.unique(chosen).tolist()) == {0, 3}
    frac = float(np.mean(chosen == 0))
    # 4000 tie-breaks: binomial SE is sqrt(0.25/4000) = 0.0079
    assert abs(frac - 0.5) < 5 * np.sqrt(0.25 / 4000)


def test_bit_llrs_sign_convention_and_shape():
    cfg = ppm.PPMConfig(16, 6.0, 0.05)
    counts = np.zeros((1, 16), dtype=int)
    counts[0, 0] = 9  # symbol 0 is bits 0000, so every LLR must favour 0 (positive)
    llr = ppm.bit_llrs(counts, cfg)
    assert llr.shape == (1, 4)
    assert np.all(llr > 0)
    counts2 = np.zeros((1, 16), dtype=int)
    counts2[0, 15] = 9  # symbol 15 is bits 1111
    assert np.all(ppm.bit_llrs(counts2, cfg) < 0)


def test_bit_llrs_are_zero_on_an_empty_symbol():
    """No counts means no information, so every bit LLR must be exactly zero."""
    cfg = ppm.PPMConfig(16, 2.0, 0.1)
    llr = ppm.bit_llrs(np.zeros((1, 16), dtype=int), cfg)
    assert np.allclose(llr, 0.0, atol=1e-12)


def test_bit_llrs_require_power_of_two_order():
    with pytest.raises(ValueError, match="power-of-two"):
        ppm.bit_llrs(np.zeros((1, 6), dtype=int), ppm.PPMConfig(6, 2.0, 0.1))


def test_sample_counts_matches_the_poisson_means():
    cfg = ppm.PPMConfig(8, 5.0, 0.4)
    rng = np.random.default_rng(0)
    counts = ppm.sample_counts(cfg, np.zeros(60_000, dtype=int), rng)
    means = counts.mean(axis=0)
    assert means[0] == pytest.approx(5.4, rel=0.02)
    assert np.allclose(means[1:], 0.4, rtol=0.08)


def test_slot_means_known_answer():
    cfg = ppm.PPMConfig(4, 3.0, 0.1, 0.05)
    assert ppm.slot_means(cfg, 2).tolist() == pytest.approx([0.15, 0.15, 3.15, 0.15])


def test_bits_per_symbol():
    assert ppm.bits_per_symbol(16) == 4.0
    assert ppm.bits_per_symbol(2) == 1.0
    assert ppm.bits_per_symbol(6) == pytest.approx(2.5849625007)


@pytest.mark.parametrize(
    "args", [(1, 2.0), (16, 0.0), (16, -1.0), (16, 2.0, -0.1), (16, 2.0, 0.0, -0.1)]
)
def test_invalid_config_raises(args):
    with pytest.raises(ValueError):
        ppm.PPMConfig(*args)


def test_out_of_range_slot_raises():
    cfg = ppm.PPMConfig(8, 2.0)
    with pytest.raises(ValueError):
        ppm.slot_means(cfg, 8)
    with pytest.raises(ValueError):
        ppm.sample_counts(cfg, [9], np.random.default_rng(0))


def test_wrong_last_axis_raises():
    cfg = ppm.PPMConfig(8, 2.0, 0.1)
    with pytest.raises(ValueError, match="last axis"):
        ppm.symbol_log_posterior(np.zeros((2, 7)), cfg)


def test_negative_counts_raise():
    cfg = ppm.PPMConfig(8, 2.0, 0.1)
    with pytest.raises(ValueError, match="counts must be"):
        ppm.slot_metrics(np.array([-1, 0, 0, 0, 0, 0, 0, 0]), cfg)


def test_bad_monte_carlo_size_and_truncation_raise():
    cfg = ppm.PPMConfig(8, 2.0, 0.1)
    with pytest.raises(ValueError):
        ppm.symbol_error_probability_mc(cfg, 0, np.random.default_rng(0))
    with pytest.raises(ValueError):
        ppm.symbol_error_probability(cfg, truncation=0)


@given(
    st.integers(min_value=2, max_value=64),
    st.floats(min_value=0.05, max_value=20.0),
)
@settings(max_examples=40, deadline=None)
def test_background_free_error_falls_with_signal(order, ns):
    cfg_low = ppm.PPMConfig(order, ns)
    cfg_high = ppm.PPMConfig(order, ns * 1.5)
    assert (
        ppm.symbol_error_probability(cfg_high)["error"]
        <= ppm.symbol_error_probability(cfg_low)["error"] + 1e-15
    )


@given(
    st.integers(min_value=2, max_value=32),
    st.floats(min_value=0.5, max_value=15.0),
    st.floats(min_value=0.01, max_value=3.0),
)
@settings(max_examples=40, deadline=None)
def test_exact_error_probability_is_a_probability(order, ns, nb):
    out = ppm.symbol_error_probability(ppm.PPMConfig(order, ns, nb))
    assert 0.0 <= out["error"] <= 1.0
    assert out["correct"] + out["error"] == pytest.approx(1.0, abs=1e-12)
    assert out["tail_mass"] < 1e-12
