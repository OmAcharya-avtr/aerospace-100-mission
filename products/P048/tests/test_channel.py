"""Fading-model tests: moments, quadrature, sampler, input validation."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import integrate, stats

from softdecode.channel import (
    GammaGammaFading,
    LognormalFading,
    amplitude_quadrature,
)


def test_lognormal_unit_mean_by_construction(lognormal):
    # Known answer: E[h] = exp(mu_x + sigma_x**2/2) and mu_x = -sigma_x**2/2,
    # so E[h] = 1 exactly.
    assert lognormal.mean() == pytest.approx(1.0, abs=1e-12)
    assert lognormal.variance() == pytest.approx(0.3, rel=1e-12)


def test_lognormal_sigma_x_from_scintillation_index():
    # sigma_I**2 = exp(sigma_x**2) - 1  =>  sigma_x = sqrt(log(1 + 0.3))
    model = LognormalFading(0.3)
    assert model.sigma_x == pytest.approx(np.sqrt(np.log(1.3)), rel=1e-14)
    assert model.mu_x == pytest.approx(-0.5 * np.log(1.3), rel=1e-14)


def test_gammagamma_scintillation_index_known_answer():
    # alpha = 4, beta = 2: 1/4 + 1/2 + 1/8 = 0.875 exactly.
    assert GammaGammaFading(4.0, 2.0).scintillation_index == pytest.approx(0.875, rel=1e-14)


@pytest.mark.parametrize("nodes", [10, 20, 40, 60])
def test_lognormal_quadrature_moments(lognormal, nodes):
    h, w = lognormal.quadrature(nodes)
    assert w.sum() == pytest.approx(1.0, abs=1e-12)
    assert float((w * h).sum()) == pytest.approx(1.0, rel=1e-9)
    assert float((w * h * h).sum()) == pytest.approx(1.3, rel=1e-8)


@pytest.mark.parametrize("nodes", [257, 513, 1025])
def test_gammagamma_quadrature_moments(gammagamma, nodes):
    h, w = gammagamma.quadrature(nodes)
    assert w.sum() == pytest.approx(1.0, abs=1e-10)
    assert float((w * h).sum()) == pytest.approx(1.0, rel=1e-8)
    # E[h**2] = (1 + 1/alpha)(1 + 1/beta) = 1.25 * 1.5 = 1.875
    assert float((w * h * h).sum()) == pytest.approx(1.875, rel=1e-8)


@pytest.mark.parametrize("nodes", [8, 16, 24])
def test_gammagamma_laguerre_quadrature_moments(gammagamma, nodes):
    # The Gauss-Laguerre rule is exact for the moments of each gamma factor
    # at very few nodes; it is the peaked likelihood integrand it struggles
    # with, which validate_quadrature.py measures.
    h, w = gammagamma.laguerre_quadrature(nodes)
    assert w.sum() == pytest.approx(1.0, abs=1e-10)
    assert float((w * h).sum()) == pytest.approx(1.0, rel=1e-9)
    assert float((w * h * h).sum()) == pytest.approx(1.875, rel=1e-8)


def test_gammagamma_log_pdf_matches_pdf(gammagamma):
    h = np.exp(np.linspace(-6.0, 3.0, 41))
    assert np.allclose(np.exp(gammagamma.log_pdf(h)), gammagamma.pdf(h), rtol=1e-12)
    assert np.all(np.isneginf(gammagamma.log_pdf(np.array([-1.0, 0.0]))))


def test_gammagamma_log_pdf_survives_large_parameters():
    # Evaluating the density directly overflows above a few hundred; the log
    # form does not, which is what makes the alpha = beta -> infinity limit in
    # validation/validate_awgn_limit.py computable.
    big = GammaGammaFading(5000.0, 5000.0)
    values = big.log_pdf(np.exp(np.linspace(-0.3, 0.3, 21)))
    assert np.all(np.isfinite(values))
    h, w = big.quadrature()
    assert float((w * h).sum()) == pytest.approx(1.0, rel=1e-8)


def test_gammagamma_grid_range_widens_for_heavy_tails():
    narrow = GammaGammaFading(100.0, 100.0).log_grid_range()
    wide = GammaGammaFading(1.0, 0.5).log_grid_range()
    assert wide[0] < narrow[0]
    assert wide[1] > narrow[1]


def test_gammagamma_pdf_normalises(gammagamma):
    total, _ = integrate.quad(lambda x: gammagamma.pdf(np.array([x]))[0], 1e-10, 80.0, limit=400)
    mean, _ = integrate.quad(
        lambda x: x * gammagamma.pdf(np.array([x]))[0], 1e-10, 80.0, limit=400
    )
    assert total == pytest.approx(1.0, abs=2e-6)
    assert mean == pytest.approx(1.0, abs=2e-6)


def test_lognormal_pdf_normalises(lognormal):
    total, _ = integrate.quad(lambda x: lognormal.pdf(np.array([x]))[0], 1e-12, 40.0, limit=400)
    assert total == pytest.approx(1.0, abs=1e-8)


def test_lognormal_sampler_is_lognormal(lognormal, rng):
    h = lognormal.sample(40000, rng)
    z = (np.log(h) - lognormal.mu_x) / lognormal.sigma_x
    # Standardise first: kstest with args= raises TypeError on this scipy.
    result = stats.kstest(z, "norm")
    assert result.pvalue > 0.01


def test_gammagamma_sampler_moments(gammagamma, rng):
    h = gammagamma.sample(200000, rng)
    assert h.mean() == pytest.approx(1.0, abs=0.02)
    assert h.var() == pytest.approx(gammagamma.scintillation_index, rel=0.05)


def test_pdf_zero_for_non_positive(lognormal, gammagamma):
    for model in (lognormal, gammagamma):
        assert np.all(model.pdf(np.array([-1.0, 0.0])) == 0.0)


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_lognormal_rejects_bad_sigma(bad):
    with pytest.raises(ValueError, match="sigma_i2"):
        LognormalFading(bad)


@pytest.mark.parametrize("args", [(0.0, 2.0), (4.0, -1.0), (np.nan, 2.0)])
def test_gammagamma_rejects_bad_parameters(args):
    with pytest.raises(ValueError):
        GammaGammaFading(*args)


def test_quadrature_rejects_small_n(lognormal, gammagamma):
    with pytest.raises(ValueError, match="at least 2"):
        lognormal.quadrature(1)
    with pytest.raises(ValueError, match="at least 3"):
        gammagamma.quadrature(2)
    with pytest.raises(ValueError, match="at least 2"):
        gammagamma.laguerre_quadrature(1)
    with pytest.raises(ValueError, match="log_lower < log_upper"):
        gammagamma.quadrature(129, log_lower=2.0, log_upper=-2.0)


def test_lognormal_quadrature_rejects_unstable_node_count(lognormal):
    # Above 300 nodes the Gauss-Hermite weights underflow to zero, which would
    # silently corrupt the rule, so the constructor refuses instead.
    with pytest.raises(ValueError, match="must not exceed 300"):
        lognormal.quadrature(400)


def test_amplitude_quadrature_dispatch(lognormal, gammagamma):
    h1, _ = amplitude_quadrature(lognormal)
    h2, _ = amplitude_quadrature(gammagamma)
    assert h1.size == 80
    assert h2.size == 319
    with pytest.raises(TypeError, match="unsupported fading model"):
        amplitude_quadrature(object())


@given(st.floats(min_value=1e-4, max_value=5.0))
@settings(max_examples=30, deadline=None)
def test_lognormal_quadrature_unit_mean_property(sigma_i2):
    # Algebraic identity: sum(w h) = E[h] = 1 for every scintillation index.
    h, w = LognormalFading(sigma_i2).quadrature(60)
    assert float((w * h).sum()) == pytest.approx(1.0, rel=1e-6)


@given(
    st.floats(min_value=0.5, max_value=20.0),
    st.floats(min_value=0.5, max_value=20.0),
)
@settings(max_examples=30, deadline=None)
def test_gammagamma_quadrature_unit_mean_property(alpha, beta):
    h, w = GammaGammaFading(alpha, beta).quadrature(513)
    assert float((w * h).sum()) == pytest.approx(1.0, rel=1e-4)
    h, w = GammaGammaFading(alpha, beta).laguerre_quadrature(16)
    assert float((w * h).sum()) == pytest.approx(1.0, rel=1e-6)
