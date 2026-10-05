"""The analytic sum-of-stages model: exactness identities and known answers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from latencynet.analytic import (
    SumOfStagesModel,
    fenton_wilkinson_quantile,
    sum_of_stages_mean,
    sum_of_stages_variance,
)
from latencynet.pipeline import make_lognormal_pipeline, sample_stage_latencies


def test_mean_is_exact_known_answer():
    # Hand-calculated: 1e-3 + 2e-3 + 3e-3 = 6e-3 exactly in binary64.
    assert sum_of_stages_mean(np.array([1e-3, 2e-3, 3e-3])) == 6e-3


def test_variance_is_exact_known_answer():
    # Hand-calculated 3-4-5: 3e-3^2 + 4e-3^2 = 9e-6 + 16e-6 = 25e-6, sd 5e-3.
    var = sum_of_stages_variance(np.array([3e-3, 4e-3]))
    assert var == pytest.approx(25e-6, rel=1e-15)
    assert math.sqrt(var) == pytest.approx(5e-3, rel=1e-15)


def test_variance_with_covariance_matrix_is_the_total_sum():
    cov = np.array([[4.0, 1.0], [1.0, 9.0]])
    # Hand-calculated: 4 + 9 + 2 * 1 = 15.
    assert sum_of_stages_variance(np.array([2.0, 3.0]), cov) == pytest.approx(15.0, rel=1e-15)


def test_mean_linearity_identity_is_exact_on_a_sample():
    # Sample mean of a row sum equals the sum of column sample means, to
    # binary64 rounding, for any data. This is the identity the analytic
    # model's mean is, and it holds whatever the dependence structure.
    spec = make_lognormal_pipeline((6e-5, 1.8e-4, 3e-5), (1.2e-5, 4e-5, 6e-6), latent_rho=0.6)
    trace = sample_stage_latencies(spec, 4096, 21)
    assert sum_of_stages_mean(trace.mean(axis=0)) == pytest.approx(
        trace.sum(axis=1).mean(), rel=5e-15
    )


def test_variance_additivity_identity_is_exact_with_covariances():
    # Var of the row sum equals the total sum of the sample covariance
    # matrix, exactly, for any data. With covariances dropped the residual is
    # exactly twice the sum of the off-diagonal terms, which is what the
    # independence assumption costs.
    spec = make_lognormal_pipeline((6e-5, 1.8e-4, 3e-5), (1.2e-5, 4e-5, 6e-6), latent_rho=0.6)
    trace = sample_stage_latencies(spec, 4096, 22)
    cov = np.cov(trace, rowvar=False, ddof=1)
    total_var = trace.sum(axis=1).var(ddof=1)
    sds = np.sqrt(np.diag(cov))
    assert sum_of_stages_variance(sds, cov) == pytest.approx(total_var, rel=5e-14)
    dropped = total_var - sum_of_stages_variance(sds)
    off = cov.sum() - np.trace(cov)
    assert dropped == pytest.approx(off, rel=5e-13)


def test_fenton_wilkinson_known_answers():
    # Zero variance is the degenerate limit and must return the mean exactly.
    assert fenton_wilkinson_quantile(1.0, 0.0, 0.99) == 1.0
    # mean = 1, var = 1: sigma^2 = ln(2), mu = -ln(2)/2, so the median is
    # exp(-ln(2)/2) = 2^-0.5 = 0.7071067811865476.
    assert fenton_wilkinson_quantile(1.0, 1.0, 0.5) == pytest.approx(
        0.7071067811865476, rel=1e-14
    )
    # Quantiles must increase in p.
    qs = [fenton_wilkinson_quantile(1.0, 0.25, p) for p in (0.5, 0.9, 0.99, 0.999)]
    assert qs == sorted(qs)


def test_fenton_wilkinson_is_exact_for_a_single_lognormal_stage():
    # With one lognormal stage the sum IS a lognormal, so the moment match is
    # not an approximation and the quantile is exact.
    spec = make_lognormal_pipeline((2e-4,), (9e-5,))
    mu, sigma = spec.stages[0].lognormal_params()
    exact = math.exp(mu + sigma * 2.3263478740408408)  # Phi^-1(0.99)
    got = fenton_wilkinson_quantile(spec.injected_mean_s(), spec.injected_std_s() ** 2, 0.99)
    assert got == pytest.approx(exact, rel=1e-12)


def test_model_predict_matches_the_free_functions():
    means = np.array([6e-5, 1.8e-4, 3e-5])
    sds = np.array([1.2e-5, 4e-5, 6e-6])
    pred = SumOfStagesModel().predict(means, sds, (0.5, 0.99))
    assert pred.mean_s == sum_of_stages_mean(means)
    assert pred.std_s == pytest.approx(math.sqrt(sum_of_stages_variance(sds)), rel=1e-15)
    assert pred.quantile_s[0.99] == pytest.approx(
        fenton_wilkinson_quantile(pred.mean_s, sum_of_stages_variance(sds), 0.99), rel=1e-15
    )
    assert pred.assume_independent is True


def test_covariance_mode_requires_a_covariance():
    model = SumOfStagesModel(assume_independent=False)
    with pytest.raises(ValueError, match="requires stage_cov_s2"):
        model.predict(np.array([1e-4]), np.array([1e-5]))


def test_covariance_mode_predicts_a_wider_spread():
    means = np.array([1e-4, 2e-4])
    sds = np.array([3e-5, 6e-5])
    spec = make_lognormal_pipeline(means, sds, latent_rho=0.8)
    cov = spec.injected_covariance()
    indep = SumOfStagesModel().predict(means, sds, (0.99,))
    dep = SumOfStagesModel(assume_independent=False).predict(means, sds, (0.99,), cov)
    assert dep.std_s > indep.std_s
    assert dep.quantile_s[0.99] > indep.quantile_s[0.99]
    # Means must agree exactly: equation (1) does not care about dependence.
    assert dep.mean_s == indep.mean_s


def test_uncertainty_shrinks_as_the_probe_grows():
    means = np.array([6e-5, 1.8e-4, 3e-5])
    sds = np.array([1.2e-5, 4e-5, 6e-6])
    m4 = 3.0 * sds**4  # Gaussian fourth moment, adequate for this test
    model = SumOfStagesModel()
    u_small = model.log_quantile_uncertainty(means, sds, m4, 64, 0.99)
    u_large = model.log_quantile_uncertainty(means, sds, m4, 6400, 0.99)
    assert u_small > u_large > 0.0
    # The GUM propagation is linear in 1/sqrt(n), so a 100x probe should
    # shrink the uncertainty by a factor near 10.
    assert u_small / u_large == pytest.approx(10.0, rel=0.02)


def test_interval_brackets_the_point_and_is_log_symmetric():
    means = np.array([6e-5, 1.8e-4])
    sds = np.array([1.2e-5, 4e-5])
    m4 = 3.0 * sds**4
    lo, pt, hi = SumOfStagesModel().predict_quantile_interval(means, sds, m4, 256, 0.99, 0.9)
    assert 0.0 < lo < pt < hi
    assert math.log(pt / lo) == pytest.approx(math.log(hi / pt), rel=1e-12)


@pytest.mark.parametrize("p", [0.0, 1.0, -0.1, 1.2])
def test_quantile_probability_validation(p):
    with pytest.raises(ValueError, match="strictly in"):
        fenton_wilkinson_quantile(1.0, 1.0, p)


def test_input_validation():
    with pytest.raises(ValueError, match="non-empty"):
        sum_of_stages_mean(np.array([]))
    with pytest.raises(ValueError, match="strictly positive"):
        sum_of_stages_mean(np.array([1e-4, -1e-4]))
    with pytest.raises(ValueError, match="non-negative"):
        sum_of_stages_variance(np.array([-1.0]))
    with pytest.raises(ValueError, match="must have shape"):
        sum_of_stages_variance(np.array([1.0, 2.0]), np.eye(3))
    with pytest.raises(ValueError, match="mean_s must be finite"):
        fenton_wilkinson_quantile(0.0, 1.0, 0.5)
    with pytest.raises(ValueError, match="variance_s2 must be finite"):
        fenton_wilkinson_quantile(1.0, -1.0, 0.5)
    with pytest.raises(ValueError, match="n_probe"):
        SumOfStagesModel().log_quantile_uncertainty(
            np.array([1e-4]), np.array([1e-5]), np.array([1e-20]), 2, 0.99
        )
    with pytest.raises(ValueError, match="level"):
        SumOfStagesModel().predict_quantile_interval(
            np.array([1e-4]), np.array([1e-5]), np.array([1e-20]), 64, 0.99, level=1.0
        )
    with pytest.raises(ValueError, match="match in length"):
        SumOfStagesModel().log_quantile_uncertainty(
            np.array([1e-4, 2e-4]), np.array([1e-5]), np.array([1e-20]), 64, 0.99
        )
