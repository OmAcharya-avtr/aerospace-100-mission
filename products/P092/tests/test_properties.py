"""Hypothesis property tests over the algebraic identities this product has.

Rules this file follows, from the mission's build guide:

- no bit-exact float equality over a pipeline;
- no assertion on a boolean produced by a strict threshold comparison;
- every ``pytest.approx(rel=...)`` carries an explicit ``abs=``;
- every counterexample Hypothesis finds is pinned as a named literal regression
  test in ``tests/test_regressions.py``, because ``.hypothesis/`` is gitignored
  and a falsifying example found by one run is lost on a fresh clone.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from scipy import stats

from deadlinerisk.accounting import account, autocorrelation, miss_indicator
from deadlinerisk.baselines import mean_plus_margin
from deadlinerisk.bootstrap import moving_block_index, percentile_interval
from deadlinerisk.bounds import (
    bonferroni_pair_bracket,
    cantelli_slack,
    cantelli_upper_bound,
    markov_slack,
    markov_upper_bound,
)
from deadlinerisk.predictor import target_noise_sd
from deadlinerisk.provenance import Origin
from deadlinerisk.speed_scaling import available_fraction
from deadlinerisk.trace import JobTrace

from fixtures.synthetic_trace import simulate_loop

PERIOD = 0.010

indicators = st.lists(st.integers(0, 1), min_size=2, max_size=400)
positive_times = st.floats(
    min_value=1e-6, max_value=0.2, allow_nan=False, allow_infinity=False
)


def _trace(missed: list[int], dropped: list[int]) -> JobTrace:
    n = len(missed)
    index = np.cumsum([0] + [1 + value for value in dropped[:-1]], dtype=np.int64)
    return JobTrace(
        run="property",
        period_s=PERIOD,
        deadline_s=PERIOD,
        job_index=index,
        release_s=index * PERIOD,
        start_delay_s=np.zeros(n),
        end_s=index * PERIOD + 0.004,
        exec_s=np.full(n, 0.004),
        response_s=np.full(n, 0.004),
        missed=np.asarray(missed, dtype=np.int64),
        dropped_before=np.asarray(dropped, dtype=np.int64),
        origin=Origin.SYNTHETIC,
    )


@given(
    missed=st.lists(st.integers(0, 1), min_size=1, max_size=120),
    extra=st.lists(st.integers(0, 6), min_size=1, max_size=120),
)
@settings(max_examples=200, deadline=None)
def test_release_and_miss_totals_are_exact_integer_identities(
    missed: list[int], extra: list[int]
) -> None:
    n = min(len(missed), len(extra))
    missed, dropped = missed[:n], extra[:n]
    totals = account(_trace(missed, dropped))
    # Integer arithmetic, so these are exact and equality is legitimate.
    assert totals.releases == n + sum(dropped)
    assert totals.misses == sum(missed) + sum(dropped)
    assert totals.executed == n


@given(
    missed=st.lists(st.integers(0, 1), min_size=1, max_size=80),
    extra=st.lists(st.integers(0, 4), min_size=1, max_size=80),
)
@settings(max_examples=200, deadline=None)
def test_miss_indicator_length_and_sum_match_the_totals(
    missed: list[int], extra: list[int]
) -> None:
    n = min(len(missed), len(extra))
    jobs = _trace(missed[:n], extra[:n])
    totals = account(jobs)
    indicator = miss_indicator(jobs)
    assert indicator.size == totals.releases
    assert int(indicator.sum()) == totals.misses
    assert set(np.unique(indicator).tolist()) <= {0, 1}


@given(indicator=indicators)
@settings(max_examples=300, deadline=None)
def test_bonferroni_bracket_always_contains_the_observed_pair_rate(
    indicator: list[int],
) -> None:
    bracket = bonferroni_pair_bracket(np.asarray(indicator, dtype=np.int8))
    assert bracket.lower - bracket.slack <= bracket.observed
    assert bracket.observed <= bracket.upper + bracket.slack


@given(
    mean=st.floats(min_value=0.0, max_value=0.05, allow_nan=False),
    sd=st.floats(min_value=1e-9, max_value=0.05, allow_nan=False),
    deadline=positive_times,
)
@settings(max_examples=400, deadline=None)
def test_gaussian_tail_never_exceeds_cantelli(
    mean: float, sd: float, deadline: float
) -> None:
    # norm.sf((D-mean)/sd) <= s^2/(s^2 + (D-mean)^2) for D > mean. The slack
    # follows the conditioning of D - mean, not a constant.
    assume(deadline > mean)
    k = (deadline - mean) / sd
    tail = float(stats.norm.sf(k))
    bound = cantelli_upper_bound(mean, sd, deadline)
    assert tail <= bound + cantelli_slack(mean, sd, deadline)


@given(
    responses=st.lists(positive_times, min_size=1, max_size=200),
    deadline=positive_times,
)
@settings(max_examples=300, deadline=None)
def test_counted_exceedance_rate_never_exceeds_the_markov_bound(
    responses: list[float], deadline: float
) -> None:
    values = np.asarray(responses)
    mean = float(values.mean())
    counted = float(np.count_nonzero(values > deadline) / values.size)
    assert counted <= markov_upper_bound(mean, deadline) + markov_slack(mean, deadline)


@given(
    responses=st.lists(positive_times, min_size=2, max_size=120),
    deadline=positive_times,
    scale=st.floats(min_value=1e-3, max_value=1e3, allow_nan=False),
)
@settings(max_examples=300, deadline=None)
def test_mean_plus_margin_k_is_invariant_under_a_common_rescaling(
    responses: list[float], deadline: float, scale: float
) -> None:
    values = np.asarray(responses)
    assume(float(values.std(ddof=1)) > 1e-12)
    plain = mean_plus_margin(values, deadline)
    scaled = mean_plus_margin(values * scale, deadline * scale)
    # k = (D - mean)/sd is invariant in exact arithmetic. TWO steps are
    # ill-conditioned and the tolerance has to follow both, which an earlier
    # version of this test got wrong and Hypothesis falsified; the case is
    # pinned in tests/test_regressions.py.
    #
    #   1. the subtraction D - mean, whose relative condition number is
    #      (|D| + |mean|) / |D - mean|;
    #   2. the sample variance, whose relative condition number is
    #      sum(x^2) / ((n-1) * sd^2) -- large whenever the sample's spread is
    #      small against its own magnitude, which is the two-nearly-equal-
    #      values case. sd is its square root, so its relative perturbation is
    #      half that figure.
    #
    # The tolerance is their sum times a few multiples of the unit roundoff
    # times |k|. It is the conditioning of the quantity and not a constant: it
    # collapses to a few eps when the sample is well spread and the deadline is
    # far from the mean.
    condition_subtraction = (abs(deadline) + abs(plain.mean_s)) / max(
        abs(deadline - plain.mean_s), 1e-300
    )
    condition_variance = float((values**2).sum()) / max(
        (values.size - 1) * plain.sd_s**2, 1e-300
    )
    tolerance = (
        64.0
        * np.finfo(np.float64).eps
        * (condition_subtraction + 0.5 * condition_variance)
        * max(abs(plain.k_to_deadline), 1.0)
    )
    assert scaled.k_to_deadline == pytest.approx(
        plain.k_to_deadline, rel=0.0, abs=float(tolerance)
    )


@given(
    cores=st.integers(min_value=1, max_value=64),
    workers=st.integers(min_value=0, max_value=256),
)
@settings(max_examples=300, deadline=None)
def test_available_fraction_is_a_fraction_and_monotone(
    cores: int, workers: int
) -> None:
    value = available_fraction(cores, workers)
    assert 0.0 < value <= 1.0
    assert available_fraction(cores, workers + 1) <= value + 1e-15


@given(
    n=st.integers(min_value=1, max_value=400),
    block=st.integers(min_value=1, max_value=400),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=300, deadline=None)
def test_moving_block_index_is_well_formed(n: int, block: int, seed: int) -> None:
    assume(block <= n)
    index = moving_block_index(n, block, np.random.default_rng(seed))
    assert index.size == n
    assert int(index.min()) >= 0
    assert int(index.max()) <= n - 1


@given(
    replicates=st.lists(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=200,
    ),
    confidence=st.floats(min_value=0.5, max_value=0.999, allow_nan=False),
)
@settings(max_examples=200, deadline=None)
def test_percentile_interval_is_ordered_and_inside_the_sample_range(
    replicates: list[float], confidence: float
) -> None:
    values = np.asarray(replicates)
    low, high = percentile_interval(values, confidence)
    assert low <= high
    assert float(values.min()) <= low
    assert high <= float(values.max())


@given(
    rate=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    releases=st.integers(min_value=1, max_value=10**6),
)
@settings(max_examples=300, deadline=None)
def test_target_noise_never_exceeds_the_half_over_sqrt_n_bound(
    rate: float, releases: int
) -> None:
    # p(1-p) <= 1/4 for p in [0, 1], so sqrt(p(1-p)/N) <= 0.5/sqrt(N). The
    # bound comes from the maximum of a quadratic, which the implementation
    # does not use.
    value = float(target_noise_sd(np.asarray([rate]), np.asarray([releases]))[0])
    assert value <= 0.5 / np.sqrt(releases) + 8.0 * np.finfo(np.float64).eps


@given(
    draws=st.lists(
        st.floats(min_value=1e-5, max_value=0.05, allow_nan=False), min_size=1, max_size=200
    )
)
@settings(max_examples=200, deadline=None)
def test_simulated_loop_satisfies_the_measurement_model(draws: list[float]) -> None:
    columns = simulate_loop(np.asarray(draws), PERIOD, PERIOD)
    # job_index strictly increasing, and the drop count is exactly the gap.
    assert np.all(np.diff(columns["job_index"]) >= 1)
    gaps = np.diff(columns["job_index"]) - 1
    assert np.array_equal(gaps, columns["dropped_before"][:-1])
    # response = end - release, exact in exact arithmetic, to rounding here.
    implied = columns["end_s"] - columns["release_s"]
    assert np.allclose(implied, columns["response_s"], rtol=1e-12, atol=1e-15)


@given(
    series=st.lists(
        st.floats(min_value=-1e3, max_value=1e3, allow_nan=False), min_size=3, max_size=200
    ),
    lag=st.integers(min_value=0, max_value=3),
)
@settings(max_examples=200, deadline=None)
def test_autocorrelation_is_bounded_by_one_in_magnitude(
    series: list[float], lag: int
) -> None:
    values = np.asarray(series)
    assume(lag < values.size)
    rho = autocorrelation(values, lag)
    # |rho(k)| <= 1 by Cauchy-Schwarz. The tolerance is a few roundings of the
    # inner products, which is the conditioning of the quotient.
    assert np.all(np.abs(rho) <= 1.0 + 64.0 * np.finfo(np.float64).eps)
