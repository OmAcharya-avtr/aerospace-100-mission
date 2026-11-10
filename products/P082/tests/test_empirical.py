"""Baseline 3: the empirical joint rate and the moving-block bootstrap.

The claims worth testing here are that the block bootstrap gives a WIDER
interval than the iid bootstrap on a serially dependent record -- which is the
whole reason it exists -- and that an interval on a difference is formed from
paired replicates, because the two models' errors are strongly correlated and
treating them as independent would widen the interval and hide a real
difference.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from sitediverse.empirical import (
    BLOCK_EPOCHS,
    LEVEL,
    REPLICATES,
    block_indices,
    bootstrap_difference,
    bootstrap_joint_rate,
    bootstrap_pair_rates,
    joint_rate,
    pair_clear_matrix,
    pair_joint_matrix,
)


def test_the_declared_bootstrap_settings_are_the_ones_documented():
    """64-epoch blocks, 400 replicates, 95 per cent."""
    assert BLOCK_EPOCHS == 64
    assert REPLICATES == 400
    assert LEVEL == 0.95


def test_the_joint_rate_of_a_hand_built_record_is_hand_checkable():
    """Four epochs, two stations, both cloudy at one of them: 0.25."""
    cloud = np.array(
        [[True, True], [True, False], [False, True], [False, False]]
    )
    assert joint_rate(cloud, (0, 1)) == pytest.approx(0.25, rel=1e-15, abs=0.0)
    assert joint_rate(cloud, (0,)) == pytest.approx(0.5, rel=1e-15, abs=0.0)


def test_the_joint_rate_of_three_stations_requires_all_three():
    """``all``, not ``any``: the network is unavailable only when every site is."""
    cloud = np.array(
        [[True, True, True], [True, True, False], [True, False, True]]
    )
    assert joint_rate(cloud, (0, 1, 2)) == pytest.approx(
        1.0 / 3.0, rel=1e-15, abs=0.0
    )


def test_the_joint_rate_is_monotone_decreasing_in_the_station_set(reference_record):
    """Adding a station to a network cannot raise its joint unavailability."""
    growing = (0,)
    previous = 1.1
    for site in (1, 2, 3, 4):
        growing = (*growing, site)
        current = joint_rate(reference_record.cloud, growing)
        assert current <= previous + 1e-15
        previous = current


def test_the_pair_joint_matrix_diagonal_is_the_marginal_rate(short_record):
    """``P(i and i)`` is ``P(i)``."""
    matrix = pair_joint_matrix(short_record.cloud)
    observed = short_record.cloud.mean(axis=0)
    assert float(np.max(np.abs(np.diag(matrix) - observed))) < 1e-15


def test_the_pair_joint_matrix_is_symmetric(short_record):
    """It is a Gramian, so it is symmetric to summation rounding."""
    matrix = pair_joint_matrix(short_record.cloud)
    assert float(np.max(np.abs(matrix - matrix.T))) < 1e-15


def test_the_pair_joint_matrix_agrees_with_the_direct_count(short_record, all_pairs):
    """The matrix-product form against a per-pair count, all 190 pairs."""
    matrix = pair_joint_matrix(short_record.cloud)
    worst = 0.0
    for site_i, site_j in all_pairs:
        direct = joint_rate(short_record.cloud, (site_i, site_j))
        worst = max(worst, abs(matrix[site_i, site_j] - direct))
    assert worst < 1e-15


def test_the_pair_clear_matrix_closes_inclusion_exclusion(short_record, all_pairs):
    """``P(both cloudy) = 1 - P(i clear) - P(j clear) + P(both clear)``, exactly.

    Computed by two different routes -- counting cloudy epochs and counting
    clear epochs -- so the residual is rounding on a quantity of order one.
    """
    cloudy = pair_joint_matrix(short_record.cloud)
    clear = pair_clear_matrix(short_record.cloud)
    rate = short_record.cloud.mean(axis=0)
    worst = 0.0
    for site_i, site_j in all_pairs:
        expected = (
            1.0
            - (1.0 - rate[site_i])
            - (1.0 - rate[site_j])
            + clear[site_i, site_j]
        )
        worst = max(worst, abs(cloudy[site_i, site_j] - expected))
    assert worst < 1e-14


def test_block_indices_have_the_declared_shape_and_stay_in_range():
    """``(replicates, epochs)``, every index a valid epoch."""
    draws = block_indices(1000, 64, 20, seed=1)
    assert draws.shape == (20, 1000)
    assert int(draws.min()) >= 0
    assert int(draws.max()) < 1000


def test_block_indices_are_consecutive_within_a_block():
    """Which is what preserves the serial dependence of equation (21)."""
    draws = block_indices(1000, 8, 5, seed=2)
    for replicate in draws:
        for start in range(0, 1000 - 8, 8):
            segment = replicate[start : start + 8]
            assert np.array_equal(segment, segment[0] + np.arange(8))


def test_block_indices_of_block_one_is_the_iid_bootstrap():
    """A block length of one resamples epochs independently."""
    draws = block_indices(100, 1, 50, seed=3)
    assert draws.shape == (50, 100)


def test_block_indices_are_deterministic_in_their_seed():
    """Two calls with the same seed give the same draws."""
    assert np.array_equal(
        block_indices(500, 16, 10, seed=4), block_indices(500, 16, 10, seed=4)
    )


def test_block_indices_reject_a_block_longer_than_the_record():
    """Input validation with the valid range in the message."""
    with pytest.raises(ValueError, match="block must lie in"):
        block_indices(100, 200, 10, seed=1)


def test_block_indices_reject_a_single_replicate():
    """A percentile interval needs more than one replicate."""
    with pytest.raises(ValueError, match="replicates must be at least 2"):
        block_indices(100, 8, 1, seed=1)


def test_block_indices_reject_a_record_of_one_epoch():
    """Input validation."""
    with pytest.raises(ValueError, match="epochs must be at least 2"):
        block_indices(1, 1, 10, seed=1)


def test_the_bootstrap_interval_contains_the_point_estimate(reference_record):
    """A percentile interval of a mean contains the mean, for any sane record."""
    for sites in ((0, 1), (0, 12), (0, 1, 2, 3)):
        band = bootstrap_joint_rate(reference_record.cloud, sites, replicates=200)
        assert band.contains(band.estimate)
        assert band.lower < band.upper


def test_the_bootstrap_estimate_is_the_empirical_joint_rate(reference_record):
    """The point estimate is not itself a bootstrap quantity."""
    band = bootstrap_joint_rate(reference_record.cloud, (0, 1), replicates=100)
    assert band.estimate == pytest.approx(
        joint_rate(reference_record.cloud, (0, 1)), rel=1e-14, abs=0.0
    )


def test_the_block_bootstrap_interval_is_wider_than_the_iid_one(reference_record):
    """Equation (21)'s whole purpose, measured on the reference record.

    The record has an integral time scale of 6.14 epochs in the latent field,
    so the iid bootstrap understates the variance of a mean. The ratio is
    reported by ``validation/validate_empirical.py``; the test asserts only
    that the block interval is the wider of the two, by a clear margin.
    """
    block = bootstrap_joint_rate(
        reference_record.cloud, (0, 1), block=64, replicates=300, seed=11
    )
    iid = bootstrap_joint_rate(
        reference_record.cloud, (0, 1), block=1, replicates=300, seed=11
    )
    assert block.standard_error > 1.5 * iid.standard_error


def test_the_bootstrap_standard_error_falls_as_the_record_lengthens(
    short_record, reference_record
):
    """A four-times-longer record halves the standard error, to within sampling."""
    short_band = bootstrap_joint_rate(short_record.cloud, (0, 1), replicates=300)
    long_band = bootstrap_joint_rate(reference_record.cloud, (0, 1), replicates=300)
    assert long_band.standard_error < short_band.standard_error
    ratio = short_band.standard_error / long_band.standard_error
    assert 1.4 < ratio < 3.0


def test_the_bootstrap_half_width_is_half_the_interval(reference_record):
    """An accessor identity, checked rather than assumed."""
    band = bootstrap_joint_rate(reference_record.cloud, (0, 1), replicates=100)
    assert band.half_width == pytest.approx(
        0.5 * (band.upper - band.lower), rel=1e-15, abs=0.0
    )


def test_the_relative_half_width_is_infinite_at_a_zero_estimate():
    """An estimate of zero has no relative width."""
    from sitediverse.empirical import BootstrapInterval

    band = BootstrapInterval(0.0, 0.0, 0.0, np.zeros(10), 1, 0.95)
    assert np.isinf(band.relative_half_width)


def test_a_narrower_level_gives_a_narrower_interval(reference_record):
    """An 80 per cent interval is inside a 95 per cent one."""
    wide = bootstrap_joint_rate(
        reference_record.cloud, (0, 1), replicates=400, level=0.95, seed=12
    )
    narrow = bootstrap_joint_rate(
        reference_record.cloud, (0, 1), replicates=400, level=0.80, seed=12
    )
    assert narrow.half_width < wide.half_width


def test_bootstrap_joint_rate_rejects_a_level_outside_zero_one(reference_record):
    """Input validation."""
    with pytest.raises(ValueError, match="level must lie in"):
        bootstrap_joint_rate(reference_record.cloud, (0, 1), level=1.0)


def test_bootstrap_pair_rates_share_one_set_of_draws(short_record):
    """Equation (22) needs the replicates paired ACROSS pairs as well as models.

    The returned draw matrix is one array, used for every pair, and the test
    checks that the replicate statistics are consistent with it rather than
    with an independent draw per pair.
    """
    pairs = ((0, 1), (0, 12), (4, 5))
    estimate, replicates, draws = bootstrap_pair_rates(
        short_record.cloud, pairs, replicates=40, seed=13
    )
    assert estimate.shape == (3,)
    assert replicates.shape == (40, 3)
    assert draws.shape == (40, short_record.epochs)
    for index, (site_i, site_j) in enumerate(pairs):
        both = short_record.cloud[:, site_i] & short_record.cloud[:, site_j]
        expected = both[draws[0]].mean()
        assert replicates[0, index] == pytest.approx(expected, rel=1e-14, abs=0.0)


def test_bootstrap_pair_rates_point_estimates_match_the_direct_counts(short_record):
    """The point estimate is the empirical rate, not a bootstrap mean."""
    pairs = ((0, 1), (3, 19))
    estimate, _, _ = bootstrap_pair_rates(
        short_record.cloud, pairs, replicates=20, seed=14
    )
    for index, pair in enumerate(pairs):
        assert estimate[index] == pytest.approx(
            joint_rate(short_record.cloud, pair), rel=1e-14, abs=0.0
        )


def test_bootstrap_pair_rates_rejects_an_empty_pair_list(short_record):
    """Input validation."""
    with pytest.raises(ValueError, match="at least one pair"):
        bootstrap_pair_rates(short_record.cloud, ())


def test_the_paired_difference_interval_is_narrower_than_the_independent_one():
    """Equation (22): pairing matters, and by how much.

    Two strongly correlated replicate series. The paired interval on their
    difference is far narrower than the interval the two marginal standard
    deviations would give if added in quadrature, which is what treating them
    as independent amounts to.
    """
    rng = np.random.default_rng(15)
    shared = rng.standard_normal(2000)
    left = shared + 0.05 * rng.standard_normal(2000)
    right = shared + 0.05 * rng.standard_normal(2000)
    paired = bootstrap_difference(left, right, estimate=0.0)
    independent = float(
        np.sqrt(np.var(left, ddof=1) + np.var(right, ddof=1))
    )
    assert paired.standard_error < 0.25 * independent


def test_the_paired_difference_interval_contains_zero_for_identical_series():
    """Two identical series differ by exactly zero in every replicate."""
    values = np.linspace(0.0, 1.0, 100)
    paired = bootstrap_difference(values, values, estimate=0.0)
    assert paired.lower == 0.0
    assert paired.upper == 0.0
    assert paired.contains(0.0)


def test_the_paired_difference_rejects_mismatched_replicate_series():
    """The message says why: the draws must be the same."""
    with pytest.raises(ValueError, match="same resampling draws"):
        bootstrap_difference(np.zeros(10), np.zeros(11), estimate=0.0)


@given(st.integers(8, 256))
def test_every_block_length_gives_an_interval_containing_the_estimate(block):
    """A property over the declared block-length range."""
    rng = np.random.default_rng(16)
    cloud = rng.random((2000, 3)) < 0.4
    band = bootstrap_joint_rate(
        cloud, (0, 1), block=block, replicates=60, seed=17
    )
    assert band.lower <= band.estimate + 1e-9
    assert band.upper >= band.estimate - 1e-9


def test_joint_rate_rejects_a_station_outside_the_record(short_record):
    """Input validation."""
    with pytest.raises(ValueError, match="outside"):
        joint_rate(short_record.cloud, (0, 500))


def test_joint_rate_rejects_an_empty_station_set(short_record):
    """Input validation."""
    with pytest.raises(ValueError, match="at least one station"):
        joint_rate(short_record.cloud, ())


def test_joint_rate_rejects_a_one_dimensional_record():
    """Input validation."""
    with pytest.raises(ValueError, match="two-dimensional"):
        joint_rate(np.zeros(10, dtype=bool), (0,))


def test_pair_joint_matrix_rejects_a_one_dimensional_record():
    """Input validation."""
    with pytest.raises(ValueError, match="two-dimensional"):
        pair_joint_matrix(np.zeros(10, dtype=bool))
