"""Fade models: conversions, moment identities, monotonicity, validation."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import integrate, stats

from coderateopt import (
    EmpiricalFade,
    GammaGammaFade,
    LognormalFade,
    db_to_linear,
    gamma_gamma_pdf,
    linear_to_db,
    scintillation_from_log_amplitude,
    sigma_ln_i_from_scintillation,
)


def test_db_roundtrip():
    for value in (0.1, 1.0, 2.5, 1000.0):
        assert db_to_linear(linear_to_db(value)) == pytest.approx(value, rel=1e-12)


def test_linear_to_db_rejects_non_positive():
    with pytest.raises(ValueError, match="strictly positive"):
        linear_to_db(0.0)
    with pytest.raises(ValueError, match="strictly positive"):
        linear_to_db(np.array([1.0, -1.0]))


def test_known_answer_three_db_is_factor_two():
    # 10*log10(2) = 3.0102999566398116 dB, a textbook constant.
    assert linear_to_db(2.0) == pytest.approx(3.0102999566398116, abs=1e-12)
    assert db_to_linear(3.0102999566398116) == pytest.approx(2.0, rel=1e-12)


def test_scintillation_conversions_are_inverses():
    # sigma_I^2 = exp(4 sigma_chi^2) - 1  and  sigma_lnI = sqrt(ln(1+sigma_I^2))
    # together give sigma_lnI = 2 sigma_chi.
    for sigma_chi in (0.05, 0.1, 0.3):
        si = scintillation_from_log_amplitude(sigma_chi)
        assert sigma_ln_i_from_scintillation(si) == pytest.approx(2.0 * sigma_chi, rel=1e-12)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_scintillation_validation(bad):
    with pytest.raises(ValueError):
        sigma_ln_i_from_scintillation(bad)
    with pytest.raises(ValueError):
        scintillation_from_log_amplitude(bad)


def test_lognormal_known_answer_sigma_db():
    # sigma_I^2 = 0.2 -> sigma_lnI = sqrt(ln 1.2) = sqrt(0.1823215568) = 0.4269912842
    # sigma_dB = (10/ln10) * sigma_lnI = 4.3429448190 * 0.4269912842 = 1.8543995855 dB
    model = LognormalFade(0.2)
    assert model.sigma_ln_i == pytest.approx(math.sqrt(math.log(1.2)), rel=1e-14)
    assert model.sigma_db == pytest.approx(1.8543995855453355, rel=1e-12)
    # mean of 10log10(I) under E[I]=1 is -(10/ln10) sigma_lnI^2 / 2
    #   = -4.3429448190 * 0.1823215568 / 2 = -0.3959062302 dB
    assert model.mean_db == pytest.approx(-0.39590623023812405, rel=1e-12)


def test_lognormal_median_preserving_shifts_the_mean_only():
    a = LognormalFade(0.3)
    b = LognormalFade(0.3, median_preserving=True)
    assert a.sigma_db == pytest.approx(b.sigma_db)
    assert b.mean_db == 0.0
    assert a.mean_db < 0.0
    # The two conventions give genuinely different availabilities.
    assert a.availability(10.0, 8.0) != b.availability(10.0, 8.0)


def test_lognormal_availability_matches_normal_cdf():
    model = LognormalFade(0.15)
    got = model.availability(12.0, 9.0)
    want = stats.norm.cdf((model.mean_db - (9.0 - 12.0)) / model.sigma_db)
    assert got == pytest.approx(float(want), rel=1e-13)


def test_lognormal_samples_reproduce_the_marginal():
    model = LognormalFade(0.25)
    rng = np.random.default_rng(20261006)
    samples = model.sample_db(200_000, rng)
    assert samples.mean() == pytest.approx(model.mean_db, abs=5.0 * model.sigma_db / math.sqrt(2e5))
    assert samples.std() == pytest.approx(model.sigma_db, rel=0.02)


def test_lognormal_mean_irradiance_is_one():
    model = LognormalFade(0.4)
    rng = np.random.default_rng(7)
    irradiance = db_to_linear(model.sample_db(400_000, rng))
    assert float(np.mean(irradiance)) == pytest.approx(1.0, rel=0.01)


def test_gamma_gamma_pdf_normalises_and_has_unit_mean():
    gg = GammaGammaFade.from_scintillation(0.3)
    density = lambda t: float(gamma_gamma_pdf(np.array([t]), gg.alpha, gg.beta)[0])  # noqa: E731
    mass, _ = integrate.quad(density, 0.0, np.inf, limit=300)
    mean, _ = integrate.quad(lambda t: t * density(t), 0.0, np.inf, limit=300)
    second, _ = integrate.quad(lambda t: t * t * density(t), 0.0, np.inf, limit=300)
    assert mass == pytest.approx(1.0, abs=1e-8)
    assert mean == pytest.approx(1.0, abs=1e-8)
    # sigma_I^2 = E[I^2]/E[I]^2 - 1 must reproduce the requested 0.3.
    assert second / mean**2 - 1.0 == pytest.approx(0.3, rel=1e-6)


def test_gamma_gamma_from_scintillation_inverts_the_identity():
    for si in (0.05, 0.2, 1.0, 3.0):
        for ratio in (1.0, 0.5, 0.2):
            gg = GammaGammaFade.from_scintillation(si, ratio=ratio)
            assert gg.scintillation_index == pytest.approx(si, rel=1e-12)
            assert gg.beta / gg.alpha == pytest.approx(ratio, rel=1e-12)


def test_gamma_gamma_rejects_bad_parameters():
    with pytest.raises(ValueError):
        GammaGammaFade(alpha=0.0, beta=1.0)
    with pytest.raises(ValueError):
        gamma_gamma_pdf(np.array([1.0]), -1.0, 1.0)
    with pytest.raises(ValueError, match="ratio"):
        GammaGammaFade.from_scintillation(0.2, ratio=1.5)
    with pytest.raises(ValueError, match="scintillation_index"):
        GammaGammaFade.from_scintillation(0.0)


def test_gamma_gamma_survival_is_a_probability_and_monotone():
    gg = GammaGammaFade.from_scintillation(0.6)
    levels = np.linspace(-20.0, 10.0, 25)
    values = np.asarray(gg.exceedance_db(levels))
    assert np.all((values >= 0.0) & (values <= 1.0))
    assert np.all(np.diff(values) <= 1e-9)


def test_gamma_gamma_upper_tail_is_not_silently_zero():
    # Integrating 0..I in one quad call for large I misses the mode and returns
    # near zero, which made survival_linear report 1.0 everywhere. Guard it.
    gg = GammaGammaFade.from_scintillation(0.3)
    assert gg.survival_linear(10.0) < 1e-4
    assert gg.survival_linear(1e-3) > 0.99


def test_gamma_gamma_approaches_lognormal_in_weak_turbulence():
    # Different models, so this is agreement in a limit, not an identity.
    gg = GammaGammaFade.from_scintillation(0.02)
    ln = LognormalFade(0.02)
    for level in (-0.5, -1.0, -2.0):
        assert float(gg.exceedance_db(level)) == pytest.approx(
            float(ln.exceedance_db(level)), abs=0.01
        )


def test_empirical_survival_is_exact_on_the_sample():
    model = EmpiricalFade(np.array([-6.0, -4.0, -2.0, 0.0]))
    assert model.exceedance_db(-7.0) == 1.0
    assert model.exceedance_db(-6.0) == 1.0
    assert model.exceedance_db(-5.9) == 0.75
    assert model.exceedance_db(-2.0) == 0.5
    assert model.exceedance_db(0.0) == 0.25
    assert model.exceedance_db(0.1) == 0.0
    assert model.resolution == 0.25


def test_empirical_validation():
    with pytest.raises(ValueError, match="non-empty"):
        EmpiricalFade(np.array([]))
    with pytest.raises(ValueError, match="non-empty"):
        EmpiricalFade(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="finite"):
        EmpiricalFade(np.array([1.0, np.nan]))


def test_quantile_inverts_exceedance():
    for model in (LognormalFade(0.2), GammaGammaFade.from_scintillation(0.5)):
        for prob in (0.1, 0.5, 0.9, 0.99):
            level = model.quantile_db(prob)
            assert float(model.exceedance_db(level)) == pytest.approx(prob, abs=1e-6)


def test_quantile_validation():
    model = LognormalFade(0.2)
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="probability"):
            model.quantile_db(bad)


@settings(max_examples=60, deadline=None)
@given(
    si=st.floats(min_value=0.01, max_value=2.0),
    margin=st.floats(min_value=-20.0, max_value=40.0),
    threshold=st.floats(min_value=-20.0, max_value=40.0),
)
def test_lognormal_availability_is_in_unit_interval(si, margin, threshold):
    value = float(LognormalFade(si).availability(margin, threshold))
    assert 0.0 <= value <= 1.0


@settings(max_examples=40, deadline=None)
@given(si=st.floats(min_value=0.01, max_value=2.0), margin=st.floats(-10.0, 30.0))
def test_lognormal_availability_decreases_with_threshold(si, margin):
    model = LognormalFade(si)
    thresholds = np.linspace(margin - 10.0, margin + 10.0, 21)
    values = np.asarray(model.availability(margin, thresholds))
    assert np.all(np.diff(values) <= 1e-12)
