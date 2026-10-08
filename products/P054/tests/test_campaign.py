"""Tests for the end-to-end campaign wrapper."""

from __future__ import annotations

import numpy as np
import pytest

from rareverify.campaign import run_campaign
from rareverify.limitstates import LinearGaussianLimitState
from rareverify.tilting import analytic_mean_shift


def test_crude_campaign_quotes_a_binomial_interval():
    state = LinearGaussianLimitState(beta=4.5, dimension=2)
    report = run_campaign(
        state, 1e-4, confidence=0.95, estimator="crude", rng=np.random.default_rng(1)
    )
    assert report.plan.n_demonstration == 29956
    assert report.interval_is_binomial is True
    assert report.interval.kind == "clopper-pearson"
    assert "simulated model" in report.describe()


def test_zero_failure_campaign_reports_the_closed_form_bound_and_meets_the_target():
    """beta = 4.5 gives p = 3.398e-6, so 29956 runs almost certainly see nothing."""
    state = LinearGaussianLimitState(beta=4.5, dimension=2)
    report = run_campaign(
        state, 1e-4, estimator="crude", rng=np.random.default_rng(3)
    )
    assert report.estimate.n_failures == 0
    assert report.zero_failure_bound == pytest.approx(
        1.0 - 0.05 ** (1 / 29956), rel=1e-12
    )
    assert report.zero_failure_bound <= 1e-4
    assert report.one_sided_upper == pytest.approx(report.zero_failure_bound, rel=1e-12)
    assert report.meets_target is True
    # The two-sided upper limit is wider and would have said the campaign
    # failed; the verdict deliberately uses the one-sided limit the plan was
    # sized from. 1 - 0.025 ** (1 / 29956) = 1.2313568e-4.
    assert report.interval.upper == pytest.approx(1.2313568e-4, rel=1e-6)
    assert report.interval.upper > 1e-4


def test_a_campaign_that_sees_violations_does_not_meet_the_target():
    """beta = 3.719 gives p = 1.000065e-4, exactly at the target, so it fails."""
    state = LinearGaussianLimitState(beta=3.719, dimension=2)
    report = run_campaign(
        state, 1e-4, estimator="crude", rng=np.random.default_rng(5)
    )
    assert report.estimate.n_failures >= 1
    assert report.zero_failure_bound is None
    assert report.meets_target is False


def test_importance_sampling_campaign_does_not_quote_a_binomial_interval():
    state = LinearGaussianLimitState(beta=3.719, dimension=2)
    report = run_campaign(
        state,
        1e-4,
        estimator="importance-sampling",
        n_samples=50_000,
        rng=np.random.default_rng(1),
    )
    assert report.interval_is_binomial is False
    assert report.interval.kind == "normal-weighted"
    assert report.zero_failure_bound is None
    assert report.one_sided_upper == pytest.approx(
        report.estimate.estimate + 1.6448536269514722 * report.estimate.standard_error,
        rel=1e-9,
    )


def test_explicit_tilt_is_honoured():
    state = LinearGaussianLimitState(beta=3.719, dimension=2)
    report = run_campaign(
        state,
        1e-4,
        estimator="importance-sampling",
        n_samples=30_000,
        tilt=analytic_mean_shift(state),
        rng=np.random.default_rng(2),
    )
    assert report.estimate.diagnostics["theta_norm"] == pytest.approx(3.719, rel=1e-12)


def test_wilson_interval_method_is_available():
    state = LinearGaussianLimitState(beta=4.5, dimension=2)
    report = run_campaign(
        state,
        1e-4,
        estimator="crude",
        n_samples=20_000,
        interval_method="wilson",
        rng=np.random.default_rng(1),
    )
    assert report.interval.kind == "wilson"


def test_campaign_validation():
    state = LinearGaussianLimitState()
    with pytest.raises(ValueError, match="estimator must be"):
        run_campaign(state, 1e-4, estimator="subset")
    with pytest.raises(ValueError, match="n_samples"):
        run_campaign(state, 1e-4, n_samples=0)
    with pytest.raises(ValueError):
        run_campaign(state, 0.0)
