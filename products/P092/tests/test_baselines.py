"""The three baselines, including hand-calculated known-answer cases."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from deadlinerisk.baselines import (
    MIN_EXCEEDANCES,
    THRESHOLD_QUANTILES,
    choose_threshold,
    counted_rate,
    gpd_tail,
    mean_plus_margin,
    threshold_stability,
)


def test_known_answer_mean_plus_margin_on_a_hand_built_sample() -> None:
    # HAND CALCULATION. Sample [1, 2, 3, 4, 5] ms, deadline 7 ms.
    #   mean = 15/5 = 3 ms
    #   sample variance (ddof=1) = ((2^2)+(1^2)+0+(1^2)+(2^2))/4 = 10/4 = 2.5
    #   sd = sqrt(2.5) = 1.5811388300841898 ms
    #   k = (7 - 3)/1.5811388300841898 = 2.5298221281347035
    #   P = norm.sf(2.5298221281347035) = 0.005706018...
    sample = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0]) * 1e-3
    result = mean_plus_margin(sample, 7e-3)
    assert result.mean_s == pytest.approx(3e-3, rel=1e-12, abs=0.0)
    assert result.sd_s == pytest.approx(1.5811388300841898e-3, rel=1e-12, abs=0.0)
    assert result.k_to_deadline == pytest.approx(2.5298221281347035, rel=1e-12, abs=0.0)
    assert result.probability == pytest.approx(
        float(stats.norm.sf(2.5298221281347035)), rel=1e-12, abs=0.0
    )
    assert result.n == 5


def test_k_is_negative_when_the_mean_already_exceeds_the_deadline() -> None:
    # HAND CALCULATION. Sample [8, 10, 12] ms, deadline 6 ms.
    #   mean = 10 ms, sd = sqrt(((-2)^2 + 0 + 2^2)/2) = sqrt(4) = 2 ms
    #   k = (6 - 10)/2 = -2
    result = mean_plus_margin(np.asarray([8e-3, 10e-3, 12e-3]), 6e-3)
    assert result.k_to_deadline == pytest.approx(-2.0, rel=1e-12, abs=0.0)
    assert result.probability > 0.5


def test_margin_probability_matches_the_k_it_reports(ground_truth) -> None:
    result = mean_plus_margin(ground_truth.response_s, ground_truth.deadline_s)
    assert result.probability == pytest.approx(
        float(stats.norm.sf(result.k_to_deadline)), rel=1e-12, abs=0.0
    )


@pytest.mark.parametrize(
    ("sample", "deadline", "pattern"),
    [
        (np.asarray([1e-3]), 0.01, "at least 2"),
        (np.asarray([1e-3, float("nan")]), 0.01, "non-finite"),
        (np.asarray([1e-3, -1e-3]), 0.01, "negative"),
        (np.asarray([1e-3, 2e-3]), 0.0, "deadline_s must be finite"),
        (np.asarray([2e-3, 2e-3]), 0.01, "zero sample standard deviation"),
    ],
)
def test_mean_plus_margin_input_validation(
    sample: np.ndarray, deadline: float, pattern: str
) -> None:
    with pytest.raises(ValueError, match=pattern):
        mean_plus_margin(sample, deadline)


def test_gpd_tail_recovers_a_known_exponential_tail(rng) -> None:
    # An exponential tail is the GPD with shape xi = 0 and scale equal to the
    # exponential's mean. Draws from Exp(mean = 1 ms) above u = 2 ms have
    # excesses that are again Exp(1 ms) by the memoryless property, so the
    # fitted shape must be near zero and the fitted scale near 1 ms. These are
    # estimates from 40000 draws, so the tolerances are sampling tolerances.
    sample = rng.exponential(1e-3, size=200_000)
    fit = gpd_tail(sample, 12e-3, 2e-3)
    assert fit.shape == pytest.approx(0.0, rel=0.0, abs=0.04)
    assert fit.scale == pytest.approx(1e-3, rel=0.06, abs=0.0)
    # P(X > 12 ms | X > 2 ms) = exp(-10) for the exponential, and
    # P(X > 2 ms) = exp(-2), so P(X > 12 ms) = exp(-12) = 6.1442e-06.
    assert fit.probability == pytest.approx(float(np.exp(-12.0)), rel=0.5, abs=0.0)


def test_gpd_modified_scale_is_beta_minus_xi_times_u(ground_truth) -> None:
    fits = threshold_stability(ground_truth.response_s, ground_truth.deadline_s)
    assert fits
    for fit in fits:
        assert fit.modified_scale == pytest.approx(
            fit.scale - fit.shape * fit.threshold_s, rel=1e-12, abs=0.0
        )


def test_gpd_zeta_is_the_measured_exceedance_rate(ground_truth) -> None:
    fits = threshold_stability(ground_truth.response_s, ground_truth.deadline_s)
    for fit in fits:
        assert fit.zeta == pytest.approx(
            fit.n_exceedances / fit.n_total, rel=1e-12, abs=0.0
        )


def test_threshold_stability_returns_one_fit_per_admissible_candidate(
    ground_truth,
) -> None:
    fits = threshold_stability(ground_truth.response_s, ground_truth.deadline_s)
    assert 0 < len(fits) <= len(THRESHOLD_QUANTILES)
    assert all(fit.n_exceedances >= MIN_EXCEEDANCES for fit in fits)
    thresholds = [fit.threshold_s for fit in fits]
    assert thresholds == sorted(thresholds)


def test_too_few_exceedances_is_refused(rng) -> None:
    sample = rng.normal(4e-3, 1e-4, size=200)
    with pytest.raises(ValueError, match="fewer than"):
        gpd_tail(sample, 0.01, float(np.quantile(sample, 0.999)))


def test_threshold_at_the_sample_maximum_is_refused(rng) -> None:
    sample = rng.normal(4e-3, 1e-4, size=1000)
    with pytest.raises(ValueError, match="sample maximum"):
        gpd_tail(sample, 0.01, float(sample.max()))


def test_choose_threshold_minimises_the_declared_spread(ground_truth) -> None:
    fits = threshold_stability(ground_truth.response_s, ground_truth.deadline_s)
    choice = choose_threshold(fits)
    spread = np.asarray(choice.shape_spread)
    chosen_index = [fit.threshold_s for fit in fits].index(choice.chosen.threshold_s)
    assert spread[chosen_index] == pytest.approx(
        float(np.min(spread)), rel=1e-12, abs=0.0
    )
    assert "Coles 2001" in choice.rule


def test_choose_threshold_refuses_an_empty_diagnostic() -> None:
    with pytest.raises(ValueError, match="no admissible candidate threshold"):
        choose_threshold([])


def test_counted_rate_is_the_accounting_total(ground_truth) -> None:
    totals = counted_rate(ground_truth)
    assert totals.miss_rate == pytest.approx(
        totals.misses / totals.releases, rel=1e-12, abs=0.0
    )


def test_negative_shape_with_endpoint_below_the_deadline_gives_zero() -> None:
    # A bounded sample: uniform on [0, 1] ms has a GPD limit with xi = -1 and a
    # finite upper endpoint at 1 ms. A deadline at 5 ms is beyond the fitted
    # endpoint, so the MODEL says the deadline cannot be exceeded. That is a
    # model statement and the flag records it as one.
    sample = np.linspace(0.0, 1e-3, 4000)
    fit = gpd_tail(sample, 5e-3, 0.9e-3)
    assert fit.shape < 0.0
    assert fit.endpoint_below_deadline is True
    assert fit.probability == 0.0
