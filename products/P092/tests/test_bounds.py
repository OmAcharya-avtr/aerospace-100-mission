"""Bound checks: mathematics the estimators do not use.

Every check here compares an estimator's output against a bound derived from
different mathematics, and every slack follows the CONDITIONING of the bound
expression rather than an operation count or an absolute constant.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from deadlinerisk.accounting import account, miss_indicator
from deadlinerisk.baselines import mean_plus_margin
from deadlinerisk.bounds import (
    bonferroni_pair_bracket,
    cantelli_slack,
    cantelli_upper_bound,
    markov_slack,
    markov_upper_bound,
)


def test_known_answer_cantelli_bound() -> None:
    # HAND CALCULATION. mean = 4 ms, sd = 1 ms, D = 7 ms, so t = 3 ms and
    #   bound = s^2/(s^2 + t^2) = 1/(1 + 9) = 0.1
    assert cantelli_upper_bound(4e-3, 1e-3, 7e-3) == pytest.approx(
        0.1, rel=1e-12, abs=0.0
    )


def test_known_answer_markov_bound() -> None:
    # HAND CALCULATION. mean = 4.5 ms, D = 10 ms, so bound = 0.45.
    assert markov_upper_bound(4.5e-3, 10e-3) == pytest.approx(0.45, rel=1e-12, abs=0.0)


def test_cantelli_is_vacuous_at_or_below_the_mean() -> None:
    assert cantelli_upper_bound(10e-3, 1e-3, 5e-3) == 1.0
    assert cantelli_upper_bound(5e-3, 1e-3, 5e-3) == 1.0


def test_markov_bound_is_clipped_at_one() -> None:
    assert markov_upper_bound(20e-3, 10e-3) == 1.0


def test_gaussian_tail_never_exceeds_the_cantelli_bound_on_the_fixture(
    campaign,
) -> None:
    # Baseline 1 computes the Gaussian tail. Cantelli uses only the first two
    # moments and no distributional form. The Gaussian tail must lie at or below
    # it for any mean and variance, and this is a bound from mathematics the
    # implementation does not use.
    for run, jobs in campaign.jobs.items():
        result = mean_plus_margin(jobs.response_s, jobs.deadline_s)
        bound = cantelli_upper_bound(result.mean_s, result.sd_s, jobs.deadline_s)
        slack = cantelli_slack(result.mean_s, result.sd_s, jobs.deadline_s)
        assert result.probability <= bound + slack, run


def test_executed_miss_rate_never_exceeds_the_markov_bound(campaign) -> None:
    # Markov: for a non-negative response R, P(R > D) <= E[R]/D. The left side
    # here is a COUNTED rate, so this is a counted number against a moment
    # bound.
    for run, jobs in campaign.jobs.items():
        totals = account(jobs)
        bound = markov_upper_bound(float(jobs.response_s.mean()), jobs.deadline_s)
        slack = markov_slack(float(jobs.response_s.mean()), jobs.deadline_s)
        assert totals.executed_miss_rate <= bound + slack, run


def test_cantelli_slack_tracks_the_conditioning_of_the_subtraction() -> None:
    # The slack must grow as the deadline approaches the mean, because that is
    # where t = D - mean loses significant digits. Two configurations with the
    # same bound magnitude but different conditioning must get different slacks.
    well_conditioned = cantelli_slack(1e-3, 1e-3, 10e-3)
    ill_conditioned = cantelli_slack(9.999e-3, 1e-3, 10e-3)
    assert ill_conditioned > well_conditioned


def test_cantelli_slack_is_not_floored_at_an_absolute_constant() -> None:
    # A bound of 1e-20 must get a slack far below 1e-20's own magnitude times
    # any fixed floor; a floor would make a narrow comparison vacuous.
    bound = cantelli_upper_bound(0.0, 1e-12, 1.0)
    slack = cantelli_slack(0.0, 1e-12, 1.0)
    assert bound == pytest.approx(1e-24, rel=1e-9, abs=0.0)
    assert slack < bound


def test_known_answer_bonferroni_bracket() -> None:
    # HAND CALCULATION. indicator = [0, 1, 1, 0, 0]; four overlapping pairs:
    #   (0,1) (1,1) (1,0) (0,0)
    #   left  members = [0, 1, 1, 0] -> p1 = 2/4 = 0.5
    #   right members = [1, 1, 0, 0] -> p2 = 2/4 = 0.5
    #   max per pair  = [1, 1, 1, 0] -> observed = 3/4 = 0.75
    #   lower = max(0.5, 0.5) = 0.5 ; upper = min(1, 1.0) = 1.0
    #   0.5 <= 0.75 <= 1.0, so the bracket holds.
    bracket = bonferroni_pair_bracket(np.asarray([0, 1, 1, 0, 0]))
    assert bracket.n_pairs == 4
    assert bracket.p_first == pytest.approx(0.5, rel=1e-12, abs=0.0)
    assert bracket.p_second == pytest.approx(0.5, rel=1e-12, abs=0.0)
    assert bracket.observed == pytest.approx(0.75, rel=1e-12, abs=0.0)
    assert bracket.lower == pytest.approx(0.5, rel=1e-12, abs=0.0)
    assert bracket.upper == pytest.approx(1.0, rel=1e-12, abs=0.0)
    assert bracket.inside is True


def test_bonferroni_bracket_holds_on_every_fixture_run(campaign) -> None:
    for run, jobs in campaign.jobs.items():
        bracket = bonferroni_pair_bracket(miss_indicator(jobs))
        assert bracket.inside, (
            f"{run}: observed {bracket.observed!r} outside "
            f"[{bracket.lower!r}, {bracket.upper!r}]"
        )


def test_bonferroni_slack_vanishes_on_an_all_zero_indicator() -> None:
    bracket = bonferroni_pair_bracket(np.zeros(100, dtype=np.int8))
    assert bracket.upper == 0.0
    assert bracket.slack == 0.0
    assert bracket.inside is True


def test_bonferroni_bracket_is_tight_on_an_alternating_indicator() -> None:
    # HAND CALCULATION. [0,1,0,1,...] of length 100 gives 99 pairs, every one
    # containing exactly one miss, so observed = 1.0. p1 and p2 are 49/99 and
    # 50/99 in some order, summing to 1.0, so the union bound is EXACTLY
    # attained and the bracket's upper end equals the observation.
    indicator = np.tile([0, 1], 50).astype(np.int8)
    bracket = bonferroni_pair_bracket(indicator)
    assert bracket.observed == pytest.approx(1.0, rel=1e-12, abs=0.0)
    assert bracket.upper == pytest.approx(1.0, rel=1e-12, abs=0.0)
    assert bracket.inside is True


@pytest.mark.parametrize(
    "series", [np.zeros(1), np.zeros((2, 2)), np.asarray([0, 2, 1])]
)
def test_bonferroni_input_validation(series: np.ndarray) -> None:
    with pytest.raises(ValueError):
        bonferroni_pair_bracket(series)


@pytest.mark.parametrize(
    ("mean", "sd", "deadline"),
    [(-1.0, 1.0, 1.0), (1.0, -1.0, 1.0), (1.0, 1.0, 0.0), (1.0, 1.0, float("nan"))],
)
def test_bound_input_validation(mean: float, sd: float, deadline: float) -> None:
    with pytest.raises(ValueError):
        if mean < 0.0 or (deadline <= 0.0 or not np.isfinite(deadline)):
            markov_upper_bound(mean, deadline)
        else:
            cantelli_upper_bound(mean, sd, deadline)


def test_cantelli_bound_is_consistent_with_the_normal_tail_analytically() -> None:
    # At k sigma above the mean the Gaussian tail is norm.sf(k) and Cantelli is
    # 1/(1 + k^2). The inequality norm.sf(k) <= 1/(1+k^2) is a statement about
    # the two functions, independent of any data, and is checked over a grid.
    for k in (0.5, 1.0, 2.0, 3.0, 5.0, 8.0):
        assert float(stats.norm.sf(k)) <= 1.0 / (1.0 + k * k)
