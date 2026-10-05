"""Injected pipeline construction, validation and sampling."""

from __future__ import annotations

import math

import numpy as np
import pytest

from latencynet.pipeline import (
    PipelineSpec,
    StageSpec,
    make_lognormal_pipeline,
    sample_stage_latencies,
    sample_total_latency,
)


def test_lognormal_params_known_answer():
    # Hand-calculated for mean = 100 us, sd = 50 us, so cv = 0.5:
    #   sigma^2 = ln(1 + 0.25) = ln(1.25) = 0.223143551314209...
    #   sigma   = 0.472380727077494...
    #   mu      = ln(1e-4) - sigma^2 / 2
    #           = -9.210340371976182 - 0.111571775657104
    #           = -9.321912147633287
    stage = StageSpec("s", 100e-6, 50e-6)
    mu, sigma = stage.lognormal_params()
    assert sigma == pytest.approx(0.4723807270774944, rel=1e-13)
    assert mu == pytest.approx(-9.321912147633287, rel=1e-13)
    # Round trip: the implied mean must come back exactly.
    assert math.exp(mu + 0.5 * sigma**2) == pytest.approx(100e-6, rel=1e-14)


def test_cv_and_moment_accessors():
    spec = make_lognormal_pipeline((1e-3, 2e-3), (3e-4, 8e-4))
    assert spec.stages[0].cv == pytest.approx(0.3, rel=1e-14)
    # Equation (1): means add exactly.
    assert spec.injected_mean_s() == pytest.approx(3e-3, rel=0, abs=1e-18)
    # Equation (2) with rho = 0: variances add. 3e-4 and 8e-4 give
    # sqrt(9e-8 + 64e-8) = sqrt(7.3e-7) = 8.544003745317531e-4.
    assert spec.injected_std_s() == pytest.approx(8.544003745317531e-4, rel=1e-13)


def test_covariance_is_diagonal_when_independent():
    spec = make_lognormal_pipeline((1e-4, 2e-4, 3e-4), (1e-5, 2e-5, 3e-5))
    cov = spec.injected_covariance()
    assert np.allclose(cov, np.diag(np.diag(cov)), atol=0.0, rtol=0.0)


def test_covariance_grows_with_rho():
    low = make_lognormal_pipeline((1e-4, 2e-4), (3e-5, 6e-5), latent_rho=0.1)
    high = make_lognormal_pipeline((1e-4, 2e-4), (3e-5, 6e-5), latent_rho=0.9)
    assert high.injected_std_s() > low.injected_std_s() > 0.0


def test_constant_stage_is_deterministic():
    spec = PipelineSpec((StageSpec("c", 5e-4, 0.0, "constant"),))
    draws = sample_total_latency(spec, 64, 3)
    assert np.all(draws == 5e-4)
    assert spec.injected_std_s() == 0.0


def test_sampling_is_deterministic_in_seed():
    spec = make_lognormal_pipeline((1e-4, 2e-4), (2e-5, 4e-5))
    a = sample_stage_latencies(spec, 128, 99)
    b = sample_stage_latencies(spec, 128, 99)
    c = sample_stage_latencies(spec, 128, 100)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_sample_shapes_and_positivity():
    spec = make_lognormal_pipeline((1e-4, 2e-4, 3e-4), (1e-5, 2e-5, 3e-5))
    trace = sample_stage_latencies(spec, 50, 1)
    assert trace.shape == (50, 3)
    assert np.all(trace > 0.0)
    assert np.allclose(sample_total_latency(spec, 50, 1), trace.sum(axis=1))


def test_sampled_moments_approach_injected():
    # 200000 passes: the sample mean of a cv = 0.3 total has a relative
    # standard error of 0.3 / sqrt(2e5) = 6.7e-4, so a 1 % band is loose.
    spec = make_lognormal_pipeline((1e-4, 2e-4), (3e-5, 6e-5))
    total = sample_total_latency(spec, 200_000, 5)
    assert total.mean() == pytest.approx(spec.injected_mean_s(), rel=0.01)
    assert total.std(ddof=1) == pytest.approx(spec.injected_std_s(), rel=0.02)


def test_correlated_sampling_recovers_injected_covariance():
    spec = make_lognormal_pipeline((1e-4, 2e-4), (3e-5, 6e-5), latent_rho=0.7)
    trace = sample_stage_latencies(spec, 200_000, 6)
    measured = np.cov(trace, rowvar=False, ddof=1)
    assert measured[0, 1] == pytest.approx(spec.injected_covariance()[0, 1], rel=0.05)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": "", "mean_s": 1e-4, "std_s": 1e-5},
        {"name": "s", "mean_s": 0.0, "std_s": 1e-5},
        {"name": "s", "mean_s": -1e-4, "std_s": 1e-5},
        {"name": "s", "mean_s": 1e-4, "std_s": -1e-5},
        {"name": "s", "mean_s": float("nan"), "std_s": 1e-5},
        {"name": "s", "mean_s": 1e-4, "std_s": 0.0},
        {"name": "s", "mean_s": 1e-4, "std_s": 1e-5, "dist": "weibull"},
        {"name": "s", "mean_s": 1e-4, "std_s": 1e-5, "dist": "constant"},
    ],
)
def test_stage_validation_rejects_bad_input(kwargs):
    with pytest.raises(ValueError):
        StageSpec(**kwargs)


def test_pipeline_validation():
    with pytest.raises(ValueError, match="at least one stage"):
        PipelineSpec(())
    with pytest.raises(TypeError):
        PipelineSpec(("not a stage",))  # type: ignore[arg-type]
    # Three stages admit rho down to -1/2 only; -0.9 is not PSD.
    with pytest.raises(ValueError, match="positive semi-definite"):
        make_lognormal_pipeline((1e-4, 1e-4, 1e-4), (1e-5, 1e-5, 1e-5), latent_rho=-0.9)
    with pytest.raises(ValueError, match="positive semi-definite"):
        make_lognormal_pipeline((1e-4, 1e-4), (1e-5, 1e-5), latent_rho=1.5)


def test_negative_rho_is_sampled_by_cholesky():
    spec = make_lognormal_pipeline((1e-4, 1e-4, 1e-4), (3e-5, 3e-5, 3e-5), latent_rho=-0.4)
    trace = sample_stage_latencies(spec, 50_000, 8)
    assert np.cov(trace, rowvar=False, ddof=1)[0, 1] < 0.0


def test_make_lognormal_pipeline_validation():
    with pytest.raises(ValueError, match="match in length"):
        make_lognormal_pipeline((1e-4, 2e-4), (1e-5,))
    with pytest.raises(ValueError, match="names must match"):
        make_lognormal_pipeline((1e-4,), (1e-5,), names=("a", "b"))


def test_sample_count_validation():
    spec = make_lognormal_pipeline((1e-4,), (1e-5,))
    with pytest.raises(ValueError, match="n_samples"):
        sample_stage_latencies(spec, 0, 1)


def test_lognormal_params_rejects_constant_stage():
    with pytest.raises(ValueError, match="requires dist='lognormal'"):
        StageSpec("c", 1e-4, 0.0, "constant").lognormal_params()
