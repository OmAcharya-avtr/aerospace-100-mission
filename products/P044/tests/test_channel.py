"""Unit, known-answer, property and validation tests for the channel models."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy import integrate

from aperturediv.channel import (
    gamma_gamma_cdf,
    gamma_gamma_moment,
    gamma_gamma_params_from_rytov,
    gamma_gamma_pdf,
    gamma_gamma_scintillation_index,
    lognormal_cdf,
    lognormal_moment,
    lognormal_pdf,
    lognormal_quantile,
    lognormal_scintillation_index,
    lognormal_sigma_log,
    sample_gamma_gamma,
    sample_lognormal,
)


class TestLognormalSigma:
    def test_known_answer_si_one(self):
        # si = 1 -> s^2 = ln 2 = 0.6931472, s = 0.8325546 (hand: sqrt(ln 2)).
        assert lognormal_sigma_log(1.0) == pytest.approx(0.8325546111576977, rel=1e-12)

    def test_known_answer_si_point_three(self):
        # si = 0.3 -> s^2 = ln 1.3 = 0.26236426, s = 0.51221506.
        assert lognormal_sigma_log(0.3) == pytest.approx(0.5122150568535555, rel=1e-12)

    @given(st.floats(min_value=1e-4, max_value=20.0))
    def test_round_trip(self, si):
        assert lognormal_scintillation_index(lognormal_sigma_log(si)) == pytest.approx(
            si, rel=1e-9
        )

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_rejects_bad_si(self, bad):
        with pytest.raises(ValueError):
            lognormal_sigma_log(bad)

    def test_rejects_nonpositive_sigma(self):
        with pytest.raises(ValueError):
            lognormal_scintillation_index(0.0)


class TestLognormalDensity:
    def test_integrates_to_one(self):
        val, _ = integrate.quad(
            lambda t: float(lognormal_pdf(np.exp(t), 0.6)) * np.exp(t), -60.0, 15.0
        )
        assert val == pytest.approx(1.0, abs=1e-9)

    @pytest.mark.parametrize("order,si", [(1, 0.3), (2, 0.3), (1, 0.9), (2, 0.9), (3, 0.5)])
    def test_moment_against_quadrature(self, order, si):
        val, _ = integrate.quad(
            lambda t: float(lognormal_pdf(np.exp(t), si)) * np.exp(t * (order + 1)),
            -60.0,
            20.0,
            limit=400,
        )
        assert val == pytest.approx(lognormal_moment(order, si), rel=1e-7)

    def test_first_moment_is_exactly_one(self):
        for si in (0.1, 0.5, 1.0, 3.0):
            assert lognormal_moment(1, si) == pytest.approx(1.0, rel=1e-15)

    def test_second_moment_known_answer(self):
        # E[I^2] = exp(s^2) = 1 + si, so at si = 0.3 it is exactly 1.3.
        assert lognormal_moment(2, 0.3) == pytest.approx(1.3, rel=1e-12)

    def test_pdf_zero_at_zero(self):
        assert float(lognormal_pdf(0.0, 0.6)) == 0.0

    def test_pdf_rejects_negative(self):
        with pytest.raises(ValueError):
            lognormal_pdf(-1.0, 0.6)

    def test_cdf_monotone_and_bounded(self):
        grid = np.geomspace(1e-6, 1e4, 500)
        cdf = lognormal_cdf(grid, 0.6)
        assert np.all(np.diff(cdf) >= 0.0)
        assert cdf[0] >= 0.0
        assert cdf[-1] <= 1.0
        assert cdf[-1] == pytest.approx(1.0, abs=1e-12)

    def test_cdf_at_median(self):
        # The median of a unit-mean lognormal is exp(-s^2/2); F there is 0.5.
        for si in (0.2, 0.7, 1.5):
            s = lognormal_sigma_log(si)
            assert float(lognormal_cdf(np.exp(-0.5 * s * s), si)) == pytest.approx(0.5, abs=1e-12)

    @given(st.floats(min_value=1e-6, max_value=1 - 1e-6), st.floats(0.01, 5.0))
    def test_quantile_inverts_cdf(self, p, si):
        q = float(lognormal_quantile(p, si))
        assert float(lognormal_cdf(q, si)) == pytest.approx(p, abs=1e-9)

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
    def test_quantile_rejects_out_of_range(self, bad):
        with pytest.raises(ValueError):
            lognormal_quantile(bad, 0.6)


class TestLognormalSampler:
    def test_mean_and_si(self, rng):
        x = sample_lognormal(200_000, 0.6, rng)
        assert x.mean() == pytest.approx(1.0, abs=0.02)
        assert x.var() / x.mean() ** 2 == pytest.approx(0.6, abs=0.03)

    def test_strictly_positive(self, rng):
        assert np.all(sample_lognormal(10_000, 2.0, rng) > 0.0)

    def test_reproducible(self):
        a = sample_lognormal(1000, 0.5, np.random.default_rng(7))
        b = sample_lognormal(1000, 0.5, np.random.default_rng(7))
        assert np.array_equal(a, b)


class TestGammaGammaParams:
    def test_shapes_positive_and_ordered(self):
        for sr in (0.05, 0.2, 1.0, 3.0, 10.0, 25.0):
            a, b = gamma_gamma_params_from_rytov(sr)
            assert a > 0.0 and b > 0.0
            assert a >= b

    def test_si_increases_with_rytov(self):
        sis = [
            gamma_gamma_scintillation_index(*gamma_gamma_params_from_rytov(sr))
            for sr in (0.1, 0.3, 1.0, 2.0)
        ]
        assert all(y > x for x, y in zip(sis, sis[1:], strict=False))

    def test_weak_turbulence_si_approaches_rytov(self):
        # For small Rytov variance the gamma-gamma si tends to sigma_R^2.
        for sr in (0.01, 0.02, 0.05):
            si = gamma_gamma_scintillation_index(*gamma_gamma_params_from_rytov(sr))
            assert si == pytest.approx(sr, rel=0.1)

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
    def test_rejects_bad_rytov(self, bad):
        with pytest.raises(ValueError):
            gamma_gamma_params_from_rytov(bad)


class TestGammaGammaDensity:
    def test_si_closed_form_known_answer(self):
        # a = b = 2: si = 1/2 + 1/2 + 1/4 = 1.25 exactly.
        assert gamma_gamma_scintillation_index(2.0, 2.0) == pytest.approx(1.25, rel=1e-15)

    def test_moments_known_answer(self):
        # a = b = 2: E[I^2] = (3*3)/(2*2) = 2.25 exactly.
        assert gamma_gamma_moment(2, 2.0, 2.0) == pytest.approx(2.25, rel=1e-12)
        assert gamma_gamma_moment(1, 2.0, 2.0) == pytest.approx(1.0, rel=1e-12)

    def test_si_from_moments_matches_closed_form(self):
        for a, b in ((4.0, 2.0), (10.0, 1.5), (2.5, 2.5)):
            m2 = gamma_gamma_moment(2, a, b)
            assert m2 - 1.0 == pytest.approx(gamma_gamma_scintillation_index(a, b), rel=1e-12)

    def test_integrates_to_one(self):
        a, b = gamma_gamma_params_from_rytov(1.0)
        val, _ = integrate.quad(
            lambda t: float(gamma_gamma_pdf(np.exp(t), a, b)) * np.exp(t),
            -60.0,
            15.0,
            limit=400,
        )
        assert val == pytest.approx(1.0, abs=1e-8)

    def test_pdf_does_not_underflow_at_large_irradiance(self):
        # Direct evaluation of kv would underflow; the log-space form must not.
        a, b = 4.0, 2.5
        vals = gamma_gamma_pdf(np.array([50.0, 200.0, 500.0]), a, b)
        assert np.all(np.isfinite(vals))
        assert np.all(vals > 0.0)
        assert np.all(np.diff(vals) < 0.0)

    def test_pdf_zero_at_zero(self):
        assert float(gamma_gamma_pdf(0.0, 3.0, 2.0)) == 0.0

    @pytest.mark.parametrize("a,b", [(0.0, 1.0), (1.0, 0.0), (-1.0, 2.0), (float("nan"), 2.0)])
    def test_rejects_bad_shapes(self, a, b):
        with pytest.raises(ValueError):
            gamma_gamma_pdf(1.0, a, b)

    def test_cdf_monotone_and_bounded(self):
        a, b = gamma_gamma_params_from_rytov(1.0)
        grid = np.geomspace(1e-4, 50.0, 40)
        cdf = np.asarray(gamma_gamma_cdf(grid, a, b))
        assert np.all(np.diff(cdf) >= -1e-12)
        assert cdf.min() >= 0.0
        assert cdf.max() <= 1.0

    def test_cdf_against_sampler(self, rng):
        a, b = gamma_gamma_params_from_rytov(1.0)
        x = sample_gamma_gamma(200_000, a, b, rng)
        for q in (0.3, 1.0, 3.0):
            assert float(gamma_gamma_cdf(q, a, b)) == pytest.approx(
                float(np.mean(x <= q)), abs=0.005
            )

    def test_cdf_scalar_returns_scalar_shape(self):
        out = gamma_gamma_cdf(1.0, 3.0, 2.0)
        assert np.ndim(out) == 0


class TestGammaGammaSampler:
    def test_mean_and_si(self, rng):
        a, b = gamma_gamma_params_from_rytov(1.0)
        x = sample_gamma_gamma(200_000, a, b, rng)
        assert x.mean() == pytest.approx(1.0, abs=0.02)
        assert x.var() / x.mean() ** 2 == pytest.approx(
            gamma_gamma_scintillation_index(a, b), abs=0.05
        )

    def test_strictly_positive(self, rng):
        a, b = gamma_gamma_params_from_rytov(2.0)
        assert np.all(sample_gamma_gamma(5000, a, b, rng) > 0.0)
