"""Board drift, the drift-adjusted variant, and the confounding check.

The board got 7.52 per cent slower while the campaign ran. These tests assert
what the drift machinery actually guarantees -- that the adjustment is a
per-cell multiplicative factor, hence stage-independent by construction -- and
they assert the confounding check's measured outcome, including the part of it
that does NOT support SPEC §8's claim.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from hiltimingdiff.drift import (
    cell_drift_factors,
    drift_adjusted_units,
    linear_fit,
    midranks,
    spearman_rho,
)
from hiltimingdiff.traces import CALIBRATION_BLOCKS

STAGES = ("ingest", "background", "threshold", "centroid", "gate", "track")


# --- midranks and Spearman's rho -----------------------------------------
#
# These exist because this product published a WRONG Spearman figure. The
# original implementation used ``np.argsort(np.argsort(x))``, an ORDINAL rank
# that breaks ties by position, on a series with five tied values. The
# statistic was then not invariant under a permutation of the rows and was not
# Spearman's rho. The tests below pin the tie treatment directly.


def test_midranks_of_distinct_values_are_one_based_ranks():
    assert midranks([10.0, 30.0, 20.0]).tolist() == [1.0, 3.0, 2.0]


def test_midranks_average_a_single_tied_pair():
    # Hand calculation: [10, 20, 20, 30]. The two 20s span ranks 2 and 3, so
    # each receives (2 + 3) / 2 = 2.5.
    assert midranks([10.0, 20.0, 20.0, 30.0]).tolist() == [1.0, 2.5, 2.5, 4.0]


def test_midranks_average_the_campaign_width_pattern():
    # Hand calculation on the actual campaign pattern: W over the nine cells in
    # execution order is [48, 64, 48, 64, 64, 96, 96, 64, 64]. The two 48s span
    # ranks 1-2 and share 1.5; the five 64s span ranks 3-7 and share 5; the two
    # 96s span ranks 8-9 and share 8.5.
    widths = [48.0, 64.0, 48.0, 64.0, 64.0, 96.0, 96.0, 64.0, 64.0]
    assert midranks(widths).tolist() == [1.5, 5.0, 1.5, 5.0, 5.0, 8.5, 8.5, 5.0, 5.0]


def test_midranks_of_an_all_tied_series_are_all_the_midpoint():
    assert midranks([7.0] * 5).tolist() == [3.0] * 5


def test_midranks_sum_to_the_triangular_number():
    values = np.array([3.0, 1.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0])
    n = values.size
    assert float(midranks(values).sum()) == pytest.approx(
        n * (n + 1) / 2.0, rel=1e-12, abs=0.0
    )


def test_midranks_are_invariant_under_a_permutation_of_the_input():
    values = np.array([48.0, 64.0, 48.0, 64.0, 64.0, 96.0, 96.0, 64.0, 64.0])
    permutation = np.array([3, 0, 8, 1, 7, 5, 2, 6, 4])
    assert np.allclose(
        midranks(values)[permutation], midranks(values[permutation]), rtol=0.0, atol=0.0
    )


def test_midranks_differ_from_ordinal_ranks_on_tied_data():
    # The defect, pinned. argsort(argsort(x)) gives the five tied 64s five
    # DISTINCT ranks; midranks gives them five copies of 5.
    values = np.array([48.0, 64.0, 48.0, 64.0, 64.0, 96.0, 96.0, 64.0, 64.0])
    ordinal = np.argsort(np.argsort(values)).astype(np.float64)
    assert ordinal.tolist() == [0.0, 2.0, 1.0, 3.0, 4.0, 7.0, 8.0, 5.0, 6.0]
    assert midranks(values).tolist() != (ordinal + 1.0).tolist()


def test_midranks_refuse_an_empty_array():
    with pytest.raises(ValueError, match="empty array"):
        midranks([])


def test_spearman_of_a_monotone_pair_is_one():
    assert spearman_rho([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]) == pytest.approx(
        1.0, rel=1e-12, abs=0.0
    )


def test_spearman_of_a_reversed_pair_is_minus_one():
    assert spearman_rho([1.0, 2.0, 3.0, 4.0], [40.0, 30.0, 20.0, 10.0]) == pytest.approx(
        -1.0, rel=1e-12, abs=0.0
    )


def test_spearman_hand_computed_on_a_tied_case():
    # Hand calculation. x = [1, 2, 3, 4] has ranks [1, 2, 3, 4]; y = [5, 7, 7, 9]
    # has midranks [1, 2.5, 2.5, 4]. Both series are centred at 2.5. The
    # covariance sum is (-1.5)(-1.5) + (-0.5)(0) + (0.5)(0) + (1.5)(1.5) = 4.5;
    # the x sum of squares is 5.0 and the y sum of squares is
    # 2.25 + 0 + 0 + 2.25 = 4.5. rho = 4.5 / sqrt(5.0 * 4.5) = 0.9486832980505138.
    assert spearman_rho([1.0, 2.0, 3.0, 4.0], [5.0, 7.0, 7.0, 9.0]) == pytest.approx(
        4.5 / np.sqrt(5.0 * 4.5), rel=1e-12, abs=0.0
    )


def test_spearman_is_invariant_under_a_permutation_of_the_pairs():
    first = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    second = np.array([48.0, 64.0, 48.0, 64.0, 64.0, 96.0, 96.0, 64.0, 64.0])
    permutation = np.array([3, 0, 8, 1, 7, 5, 2, 6, 4])
    assert spearman_rho(first[permutation], second[permutation]) == pytest.approx(
        spearman_rho(first, second), rel=1e-12, abs=0.0
    )


def test_ordinal_rank_spearman_is_NOT_permutation_invariant():
    # The defect this product shipped, pinned as a literal so it cannot recur
    # silently. The same nine observations give 0.85 in execution order and
    # 0.5333333333333333 in the alphabetical order a groupby produces.
    first = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    second = np.array([48.0, 64.0, 48.0, 64.0, 64.0, 96.0, 96.0, 64.0, 64.0])

    def ordinal_rho(left, right):
        return float(
            np.corrcoef(
                np.argsort(np.argsort(left)).astype(np.float64),
                np.argsort(np.argsort(right)).astype(np.float64),
            )[0, 1]
        )

    permutation = np.array([0, 2, 7, 3, 1, 8, 4, 5, 6])
    assert ordinal_rho(first, second) == pytest.approx(0.85, rel=1e-12, abs=0.0)
    assert ordinal_rho(first[permutation], second[permutation]) == pytest.approx(
        0.5333333333333333, rel=1e-12, abs=0.0
    )


def test_spearman_is_nan_when_a_series_is_constant():
    assert np.isnan(spearman_rho([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))


def test_spearman_refuses_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        spearman_rho([1.0, 2.0, 3.0], [1.0, 2.0])


def test_spearman_refuses_a_single_observation():
    with pytest.raises(ValueError, match="at least 2"):
        spearman_rho([1.0], [1.0])


def test_spearman_agrees_with_scipy_on_the_campaign_series():
    scipy_stats = pytest.importorskip(
        "scipy.stats", reason="scipy is in the test extra; the cross-check needs it"
    )
    first = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    second = np.array([48.0, 64.0, 48.0, 64.0, 64.0, 96.0, 96.0, 64.0, 64.0])
    assert spearman_rho(first, second) == pytest.approx(
        float(scipy_stats.spearmanr(first, second).statistic), rel=1e-12, abs=0.0
    )


def test_spearman_agrees_with_scipy_on_random_tied_data():
    scipy_stats = pytest.importorskip(
        "scipy.stats", reason="scipy is in the test extra; the cross-check needs it"
    )
    generator = np.random.default_rng(5309)
    disagreements = 0
    constant_cases = 0
    for _ in range(200):
        size = int(generator.integers(3, 15))
        left = generator.integers(0, 4, size=size).astype(np.float64)
        right = generator.integers(0, 4, size=size).astype(np.float64)
        ours = spearman_rho(left, right)
        # A draw can make one series constant, where rho is undefined. scipy
        # warns and returns nan there and so does this package, which is the
        # agreement being tested; the warning is silenced rather than left to
        # make the suite noisy about a case it deliberately exercises.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            theirs = float(scipy_stats.spearmanr(left, right).statistic)
        if np.isnan(ours):
            constant_cases += 1
        if np.isnan(ours) and np.isnan(theirs):
            continue
        if np.isnan(ours) != np.isnan(theirs) or abs(ours - theirs) > 1e-12:
            disagreements += 1
    assert disagreements == 0
    # The undefined cases are counted so a zero disagreement count cannot come
    # from every draw being undefined on both sides.
    assert constant_cases < 200


def test_midranks_match_scipy_rankdata_average():
    scipy_stats = pytest.importorskip(
        "scipy.stats", reason="scipy is in the test extra; the cross-check needs it"
    )
    values = np.array([48.0, 64.0, 48.0, 64.0, 64.0, 96.0, 96.0, 64.0, 64.0])
    assert np.allclose(
        midranks(values),
        scipy_stats.rankdata(values, method="average"),
        rtol=1e-12,
        atol=0.0,
    )


# --- the straight-line fit helper ----------------------------------------


def test_linear_fit_recovers_an_exact_line():
    x = np.arange(10.0)
    fit = linear_fit(x, 3.0 * x + 7.0)
    assert fit.slope == pytest.approx(3.0, rel=1e-10, abs=0.0)
    assert fit.intercept == pytest.approx(7.0, rel=1e-10, abs=0.0)


def test_linear_fit_r_squared_is_one_on_an_exact_line():
    x = np.arange(10.0)
    assert linear_fit(x, 2.0 * x - 1.0).r_squared == pytest.approx(1.0, rel=1e-10, abs=0.0)


def test_linear_fit_r_squared_is_zero_on_a_flat_response():
    x = np.arange(11.0)
    y = np.array([1.0, -1.0] * 5 + [0.0])
    assert linear_fit(x, y).r_squared < 0.05


def test_linear_fit_predicts_at_a_new_point():
    fit = linear_fit(np.arange(5.0), np.arange(5.0) * 4.0)
    assert float(fit.predict(10.0)) == pytest.approx(40.0, rel=1e-10, abs=0.0)


def test_linear_fit_records_its_labels():
    fit = linear_fit(np.arange(5.0), np.arange(5.0), x_label="t", y_label="ns")
    assert (fit.x_label, fit.y_label, fit.n) == ("t", "ns", 5)


def test_linear_fit_refuses_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        linear_fit(np.arange(5.0), np.arange(6.0))


def test_linear_fit_refuses_fewer_than_three_points():
    with pytest.raises(ValueError, match="at least 3 points"):
        linear_fit(np.arange(2.0), np.arange(2.0))


def test_linear_fit_refuses_a_constant_predictor():
    with pytest.raises(ValueError, match="constant"):
        linear_fit(np.ones(5), np.arange(5.0))


def test_linear_fit_r_squared_equals_squared_pearson_for_a_line_with_intercept():
    generator = np.random.default_rng(17)
    x = generator.normal(size=60)
    y = 2.0 * x + generator.normal(size=60)
    fit = linear_fit(x, y)
    pearson = float(np.corrcoef(x, y)[0, 1])
    assert fit.r_squared == pytest.approx(pearson**2, rel=1e-9, abs=0.0)


# --- the calibration blocks ----------------------------------------------


def test_eleven_calibration_blocks_are_fitted(drift):
    assert len(drift.blocks) == 11


def test_block_labels_are_in_the_order_they_were_timed(drift):
    assert list(drift.blocks["which"]) == list(CALIBRATION_BLOCKS)


@pytest.mark.parametrize("label", CALIBRATION_BLOCKS)
def test_every_block_has_two_thousand_reps(drift, label):
    row = drift.blocks.loc[drift.blocks["which"] == label].iloc[0]
    assert int(row["reps"]) == 2000


def test_block_elapsed_times_increase(drift):
    elapsed = drift.blocks["elapsed_s"].to_numpy()
    assert np.all(np.diff(elapsed) > 0.0)


def test_start_and_end_medians_match_the_campaign_metadata(campaign, drift):
    assert drift.start_median_ns == campaign.meta["calibration_start_median_ns"]
    assert drift.end_median_ns == campaign.meta["calibration_end_median_ns"]


def test_bracket_ratio_matches_the_recorded_drift_ratio(drift):
    assert drift.bracket_ratio == pytest.approx(drift.meta_ratio, rel=1e-15, abs=0.0)


def test_measured_drift_exceeds_seven_per_cent(drift):
    assert drift.bracket_ratio > 1.07


def test_fitted_ratio_is_below_the_bracket_ratio(drift):
    # The drift is step-like rather than linear, so a straight line through all
    # eleven blocks under-reads the end-to-start bracket. Both are reported.
    assert drift.fitted_ratio < drift.bracket_ratio


def test_straight_line_explains_most_but_not_all_of_the_drift(drift):
    assert 0.5 < drift.fit.r_squared < 0.9


def test_drift_is_not_monotonic_block_to_block(drift):
    # Measured: only some of the ten adjacent steps are increases. A monotonic
    # drift would have all ten.
    assert 0 < drift.monotonic_blocks < 10


def test_largest_single_block_step_is_a_large_share_of_the_whole_drift(drift):
    assert drift.max_block_step_ratio > 1.04


def test_drift_slope_is_positive(drift):
    assert drift.fit.slope > 0.0


def test_factor_is_one_at_the_campaign_start(drift):
    assert float(drift.factor(0.0)) == pytest.approx(1.0, rel=1e-12, abs=0.0)


def test_factor_increases_with_elapsed_time(drift):
    assert float(drift.factor(700.0)) > float(drift.factor(0.0))


def test_campaign_elapsed_is_the_recorded_seven_hundred_thirty_four(drift):
    assert drift.campaign_elapsed_s == pytest.approx(734.0, rel=0.0, abs=0.1)


# --- per-cell factors and the adjusted units -----------------------------


def test_every_cell_gets_a_drift_factor(campaign, drift):
    factors = cell_drift_factors(campaign, drift)
    assert len(factors) == 9


def test_cell_factors_are_all_above_one(campaign, drift):
    factors = cell_drift_factors(campaign, drift)
    assert min(factors.values()) > 1.0


def test_cell_factors_increase_with_execution_order(campaign, drift):
    factors = cell_drift_factors(campaign, drift)
    ordered = sorted(factors, key=campaign.order_index)
    values = [factors[cell] for cell in ordered]
    assert values == sorted(values)


def test_cell_factors_span_less_than_the_whole_campaign_drift(campaign, drift):
    factors = cell_drift_factors(campaign, drift)
    span = max(factors.values()) / min(factors.values())
    assert 1.0 < span < drift.bracket_ratio


def test_adjusted_units_divide_hardware_times_by_the_factor(campaign, units, drift):
    factors = cell_drift_factors(campaign, drift)
    adjusted = drift_adjusted_units(units, factors)
    scale = np.asarray([factors[cell] for cell in units.cells], dtype=np.float64)
    assert np.allclose(adjusted.hw * scale[:, None], units.hw, rtol=1e-12, atol=0.0)


def test_adjusted_units_leave_declared_operation_counts_alone(campaign, units, drift):
    # An operation count does not drift.
    factors = cell_drift_factors(campaign, drift)
    adjusted = drift_adjusted_units(units, factors)
    assert np.array_equal(adjusted.ops, units.ops)


def test_adjusted_units_keep_the_unit_identity(campaign, units, drift):
    factors = cell_drift_factors(campaign, drift)
    adjusted = drift_adjusted_units(units, factors)
    assert adjusted.cells == units.cells
    assert np.array_equal(adjusted.pass_index, units.pass_index)


def test_adjusted_units_refuse_a_missing_cell_factor(units):
    with pytest.raises(ValueError, match="no fitted drift factor"):
        drift_adjusted_units(units, {"w48_sparse": 1.0})


def test_adjusted_units_refuse_a_non_positive_factor(campaign, units, drift):
    factors = dict.fromkeys(set(units.cells), 1.0)
    factors["w96_dense"] = 0.0
    with pytest.raises(ValueError, match="non-positive"):
        drift_adjusted_units(units, factors)


def test_drift_adjustment_is_stage_independent_by_construction(campaign, units, drift):
    # The adjustment divides every stage of a cell by the SAME factor, so the
    # ratio between any two stages within a cell is untouched. This is a
    # structural statement about the adjustment and it is what makes drift
    # unable to explain a difference BETWEEN stages.
    factors = cell_drift_factors(campaign, drift)
    adjusted = drift_adjusted_units(units, factors)
    raw_ratio = units.hw[:, 3] / units.hw[:, 0]
    adjusted_ratio = adjusted.hw[:, 3] / adjusted.hw[:, 0]
    assert np.allclose(raw_ratio, adjusted_ratio, rtol=1e-12, atol=0.0)


# --- the raw versus adjusted comparison ----------------------------------


def test_comparison_keeps_the_raw_figure_available(comparison):
    assert comparison.raw.spread > 0.0


@pytest.mark.parametrize("stage", STAGES)
def test_adjusted_ratio_is_below_the_raw_ratio(comparison, stage):
    # Every cell's factor exceeds 1, so dividing by it lowers every R.
    assert comparison.adjusted.ratio[stage] < comparison.raw.ratio[stage]


@pytest.mark.parametrize("stage", STAGES)
def test_per_stage_shift_is_small_compared_with_the_spread(comparison, stage):
    assert abs(comparison.ratio_shift()[stage]) < 0.05


def test_spread_shift_is_small(comparison):
    assert abs(comparison.spread_relative_shift) < 0.05


def test_drift_cannot_explain_the_measured_spread(comparison):
    # Both the raw and the drift-adjusted spread are orders of magnitude above
    # the pre-registered 3. The adjustment moves the spread by a few per cent.
    assert comparison.raw.spread > 3.0
    assert comparison.adjusted.spread > 3.0


def test_comparison_frame_is_in_pipeline_order(comparison):
    assert list(comparison.frame()["stage"]) == list(STAGES)


def test_max_abs_ratio_shift_matches_the_per_stage_shifts(comparison):
    shifts = comparison.ratio_shift()
    assert comparison.max_abs_ratio_shift == pytest.approx(
        max(abs(value) for value in shifts.values()), rel=1e-12, abs=0.0
    )


def test_comparison_carries_a_factor_for_every_cell(comparison):
    assert len(comparison.factors) == 9


# --- the confounding check -----------------------------------------------


def test_confounding_covers_the_nine_cells(confound):
    assert len(confound.cells) == 9


def test_width_explains_almost_all_between_cell_variance(confound):
    assert confound.on_width.r_squared > 0.95


def test_order_explains_far_less_than_width(confound):
    assert confound.on_order.r_squared < confound.on_width.r_squared


def test_width_dominates_order(confound):
    assert confound.width_dominates is True


def test_confounding_spearman_uses_midranks(confound):
    # The corrected value, cross-checked against scipy.stats.spearmanr by the
    # coordinator and by this suite. The superseded ordinal-rank value was
    # 0.5333333333333333.
    assert confound.order_width_spearman == pytest.approx(
        0.5809475019311126, rel=1e-12, abs=0.0
    )
    assert confound.order_width_spearman != pytest.approx(
        0.5333333333333333, rel=1e-9, abs=0.0
    )


def test_confounding_spearman_matches_a_direct_midrank_computation(confound):
    assert confound.order_width_spearman == pytest.approx(
        spearman_rho(confound.order_index, confound.width), rel=1e-12, abs=0.0
    )


def test_order_and_width_are_NOT_uncorrelated_in_this_draw(confound):
    # A WEAK RESULT, asserted so it cannot be lost. SPEC §8 randomised the cell
    # order so that elapsed time would not be confounded with frame dimension.
    # With nine cells, this particular draw still leaves a substantial positive
    # correlation between order_index and W. The randomisation removed P093's
    # systematic ascending-dimension confound; it did not make the two
    # orthogonal, and this test pins the fact rather than the hope.
    assert confound.order_width_correlation > 0.4
    assert confound.order_width_spearman > 0.5


def test_order_and_width_correlation_is_below_three_quarters(confound):
    assert confound.order_width_correlation < 0.75


def test_confounding_slopes_are_positive(confound):
    assert confound.on_width.slope > 0.0
    assert confound.on_order.slope > 0.0


def test_confounding_median_totals_are_positive(confound):
    assert float(confound.median_total_ns.min()) > 0.0


def test_confounding_widths_are_the_three_frame_dimensions(confound):
    assert sorted(set(confound.width.tolist())) == [48.0, 64.0, 96.0]


def test_confounding_order_indices_are_zero_to_eight(confound):
    assert sorted(confound.order_index.tolist()) == [float(i) for i in range(9)]
