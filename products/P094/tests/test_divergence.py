"""The five SPEC §5 outputs, their invariances and the rank statistic.

The invariance tests are the load-bearing ones. ``ns_per_op`` is a single
scalar calibrated on the analysis host, so it moves between runs; the spread,
the Kendall statistic, the permutation and both arg-extrema must not move with
it, and these tests assert that rather than asserting it in a docstring.
"""

from __future__ import annotations

import numpy as np
import pytest

from hiltimingdiff.divergence import (
    PassUnits,
    divergence,
    kendall_tau_b,
    ordering,
    pass_units,
    rank_permutation,
)

STAGES = ("ingest", "background", "threshold", "centroid", "gate", "track")


def test_pass_units_has_five_hundred_forty_units(units):
    assert units.n_units == 540


def test_pass_units_has_six_stages(units):
    assert units.n_stages == 6
    assert units.stages == STAGES


def test_pass_units_cells_are_the_nine(units):
    assert len(set(units.cells)) == 9


def test_pass_units_pass_indices_span_sixty(units):
    assert sorted(set(units.pass_index.tolist())) == list(range(60))


@pytest.mark.parametrize("stage", STAGES)
def test_pass_units_hardware_times_are_positive(units, stage):
    column = units.hw[:, units.stage_column(stage)]
    assert float(column.min()) > 0.0


@pytest.mark.parametrize("stage", STAGES)
def test_pass_units_declared_ops_are_positive(units, stage):
    column = units.ops[:, units.stage_column(stage)]
    assert float(column.min()) > 0.0


@pytest.mark.parametrize("stage", STAGES)
def test_stage_column_round_trips(units, stage):
    assert units.stages[units.stage_column(stage)] == stage


def test_stage_column_refuses_an_unknown_stage(units):
    with pytest.raises(KeyError, match="is not one of"):
        units.stage_column("decimate")


def test_pass_units_refuses_mismatched_shapes():
    with pytest.raises(ValueError, match="same shape"):
        PassUnits(
            stages=("a", "b"),
            cells=("c",),
            pass_index=np.zeros(1, dtype=np.int64),
            hw=np.ones((1, 2)),
            ops=np.ones((1, 3)),
        )


def test_pass_units_refuses_a_stage_name_count_mismatch():
    with pytest.raises(ValueError, match="stage names"):
        PassUnits(
            stages=("a", "b", "c"),
            cells=("c",),
            pass_index=np.zeros(1, dtype=np.int64),
            hw=np.ones((1, 2)),
            ops=np.ones((1, 2)),
        )


def test_pass_units_refuses_a_holed_table(campaign):
    holed = campaign.stages_pass.iloc[1:]

    class Shim:
        pass

    shim = Shim()
    shim.stages_pass = holed
    shim.stage_names = campaign.stage_names
    with pytest.raises(ValueError, match="holed table"):
        pass_units(shim)


# --- Kendall's tau-b ------------------------------------------------------


def test_tau_b_of_identical_rankings_is_one():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert kendall_tau_b(values, values) == pytest.approx(1.0, rel=1e-12, abs=0.0)


def test_tau_b_of_reversed_rankings_is_minus_one():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert kendall_tau_b(values, values[::-1]) == pytest.approx(-1.0, rel=1e-12, abs=0.0)


def test_tau_b_of_one_adjacent_swap_in_four_items():
    # Hand calculation. n = 4 so n0 = 6 pairs. Swapping the top two items of a
    # perfectly concordant ranking makes exactly one pair discordant:
    # C = 5, D = 1, no ties, tau_b = (5 - 1) / 6 = 0.666666...
    first = np.array([1.0, 2.0, 3.0, 4.0])
    second = np.array([1.0, 2.0, 4.0, 3.0])
    assert kendall_tau_b(first, second) == pytest.approx(4.0 / 6.0, rel=1e-12, abs=0.0)


def test_tau_b_corrects_for_ties_on_one_side():
    # n = 4, n0 = 6. first ties the last two items: n1 = 1. second is strictly
    # increasing: n2 = 0. The tied pair is neither concordant nor discordant,
    # so C = 5, D = 0 and tau_b = 5 / sqrt(5 * 6) = 0.912870929...
    first = np.array([1.0, 2.0, 3.0, 3.0])
    second = np.array([1.0, 2.0, 3.0, 4.0])
    assert kendall_tau_b(first, second) == pytest.approx(
        5.0 / np.sqrt(5.0 * 6.0), rel=1e-12, abs=0.0
    )


def test_tau_b_is_nan_when_one_side_is_constant():
    assert np.isnan(kendall_tau_b(np.ones(5), np.arange(5.0)))


def test_tau_b_is_symmetric():
    first = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
    second = np.array([2.0, 7.0, 1.0, 8.0, 2.0])
    assert kendall_tau_b(first, second) == pytest.approx(
        kendall_tau_b(second, first), rel=1e-12, abs=0.0
    )


def test_tau_b_is_invariant_under_a_positive_rescaling():
    first = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
    second = np.array([2.0, 7.0, 1.0, 8.0, 2.0])
    assert kendall_tau_b(first, 1000.0 * second) == pytest.approx(
        kendall_tau_b(first, second), rel=1e-12, abs=0.0
    )


def test_tau_b_refuses_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        kendall_tau_b(np.ones(3), np.ones(4))


def test_tau_b_refuses_a_single_item():
    with pytest.raises(ValueError, match="at least 2"):
        kendall_tau_b(np.ones(1), np.ones(1))


# --- orderings and permutations -------------------------------------------


def test_ordering_is_ascending():
    assert ordering(np.array([5.0, 1.0, 3.0])).tolist() == [1, 2, 0]


def test_ordering_breaks_ties_by_index():
    assert ordering(np.array([2.0, 1.0, 1.0])).tolist() == [1, 2, 0]


def test_rank_permutation_of_agreeing_orders_is_the_identity():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    assert rank_permutation(values, 10.0 * values) == (0, 1, 2, 3)


def test_rank_permutation_of_a_reversal():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    assert rank_permutation(values, -values) == (3, 2, 1, 0)


def test_rank_permutation_is_a_permutation():
    declared = np.array([4.0, 1.0, 9.0, 2.0, 7.0, 3.0])
    measured = np.array([5.0, 8.0, 1.0, 3.0, 2.0, 9.0])
    assert sorted(rank_permutation(declared, measured)) == list(range(6))


# --- the measured divergence ----------------------------------------------


def test_divergence_covers_the_six_stages(divergence_result):
    assert divergence_result.stages == STAGES
    assert set(divergence_result.ratio) == set(STAGES)


@pytest.mark.parametrize("stage", STAGES)
def test_ratio_is_hardware_median_over_declared_nanoseconds(divergence_result, stage):
    expected = (
        divergence_result.hw_median_ns[stage] / divergence_result.declared_ns[stage]
    )
    assert divergence_result.ratio[stage] == pytest.approx(expected, rel=1e-12, abs=0.0)


@pytest.mark.parametrize("stage", STAGES)
def test_declared_nanoseconds_is_ops_times_ns_per_op(divergence_result, stage):
    expected = (
        divergence_result.declared_ops_median[stage] * divergence_result.ns_per_op
    )
    assert divergence_result.declared_ns[stage] == pytest.approx(
        expected, rel=1e-12, abs=0.0
    )


@pytest.mark.parametrize("stage", STAGES)
def test_ns_per_declared_op_is_free_of_ns_per_op(divergence_result, stage):
    expected = divergence_result.ratio[stage] * divergence_result.ns_per_op
    assert divergence_result.ns_per_declared_op[stage] == pytest.approx(
        expected, rel=1e-12, abs=0.0
    )


@pytest.mark.parametrize("stage", STAGES)
def test_ratio_interval_records_the_pass_unit(divergence_result, stage):
    assert divergence_result.ratio_interval[stage].unit == "pass"


@pytest.mark.parametrize("stage", STAGES)
def test_ratio_interval_records_five_hundred_forty_units(divergence_result, stage):
    assert divergence_result.ratio_interval[stage].units == 540


@pytest.mark.parametrize("stage", STAGES)
def test_ratio_interval_brackets_its_point(divergence_result, stage):
    interval = divergence_result.ratio_interval[stage]
    assert interval.low <= interval.point <= interval.high


def test_spread_is_max_over_min(divergence_result):
    values = list(divergence_result.ratio.values())
    assert divergence_result.spread == pytest.approx(
        max(values) / min(values), rel=1e-12, abs=0.0
    )


def test_spread_interval_records_the_pass_unit(divergence_result):
    assert divergence_result.spread_interval.unit == "pass"


def test_spread_interval_lower_bound_is_above_the_preregistered_three(divergence_result):
    assert divergence_result.spread_interval.low > 3.0


def test_argmax_is_the_stage_with_the_largest_ratio(divergence_result):
    assert divergence_result.argmax == max(
        divergence_result.ratio, key=divergence_result.ratio.get
    )


def test_argmin_is_the_stage_with_the_smallest_ratio(divergence_result):
    assert divergence_result.argmin == min(
        divergence_result.ratio, key=divergence_result.ratio.get
    )


def test_arg_shares_are_fractions_of_measured_frame_time(divergence_result):
    assert 0.0 < divergence_result.argmax_share < 1.0
    assert 0.0 < divergence_result.argmin_share < 1.0


def test_orders_are_permutations_of_the_stage_names(divergence_result):
    assert sorted(divergence_result.declared_order) == sorted(STAGES)
    assert sorted(divergence_result.measured_order) == sorted(STAGES)


def test_permutation_has_six_entries(divergence_result):
    assert sorted(divergence_result.permutation) == list(range(6))


def test_frame_has_one_row_per_stage_in_pipeline_order(divergence_result):
    frame = divergence_result.frame()
    assert list(frame["stage"]) == list(STAGES)


def test_as_rows_matches_the_frame(divergence_result):
    assert len(divergence_result.as_rows()) == 6


# --- the calibration-free interval, and the round trip that is not exact ---
#
# These exist because the product was BLOCKED at the release gate by an
# intermittent failure in examples/headline_divergence.py: it drew the
# calibration-free point `ns_per_declared_op` against `ratio_interval * ns_per_op`,
# and that round trip is exact only in exact arithmetic.


@pytest.mark.parametrize("stage", STAGES)
def test_calibration_free_interval_brackets_its_point(divergence_result, stage):
    interval = divergence_result.ns_per_declared_op_interval[stage]
    point = divergence_result.ns_per_declared_op[stage]
    assert interval.low <= point <= interval.high


@pytest.mark.parametrize("stage", STAGES)
def test_calibration_free_interval_records_the_pass_unit(divergence_result, stage):
    assert divergence_result.ns_per_declared_op_interval[stage].unit == "pass"


@pytest.mark.parametrize("stage", STAGES)
def test_calibration_free_interval_point_is_the_calibration_free_point(
    divergence_result, stage
):
    assert divergence_result.ns_per_declared_op_interval[stage].point == (
        divergence_result.ns_per_declared_op[stage]
    )


@pytest.mark.parametrize("factor", [1e-3, 0.5, 2.0, 1000.0])
def test_calibration_free_interval_is_BIT_IDENTICAL_under_ns_per_op(
    units, ns_per_op, factor
):
    # Not `approx`: bit-identical. The whole point of the calibration-free form
    # is that ns_per_op never enters it.
    base = divergence(units, ns_per_op, resamples=128, seed=4242)
    scaled = divergence(units, ns_per_op * factor, resamples=128, seed=4242)
    for stage in STAGES:
        assert (
            scaled.ns_per_declared_op_interval[stage].low
            == base.ns_per_declared_op_interval[stage].low
        )
        assert (
            scaled.ns_per_declared_op_interval[stage].high
            == base.ns_per_declared_op_interval[stage].high
        )


@pytest.mark.parametrize("factor", [1e-3, 0.5, 2.0, 1000.0])
def test_spread_and_its_interval_are_BIT_IDENTICAL_under_ns_per_op(
    units, ns_per_op, factor
):
    # The spread and its interval are now computed where ns_per_op never enters,
    # so the cancellation is exact in floating point and not only in exact
    # arithmetic. Before that change both moved by one ulp with the calibration.
    base = divergence(units, ns_per_op, resamples=128, seed=4242)
    scaled = divergence(units, ns_per_op * factor, resamples=128, seed=4242)
    assert scaled.spread == base.spread
    assert scaled.spread_interval.low == base.spread_interval.low
    assert scaled.spread_interval.high == base.spread_interval.high
    assert scaled.tau_b == base.tau_b


def test_spread_equals_the_ratio_of_the_calibration_free_extremes(divergence_result):
    values = list(divergence_result.ns_per_declared_op.values())
    assert divergence_result.spread == max(values) / min(values)


def test_regression_the_round_trip_through_ns_per_op_is_NOT_exact(units):
    # THE GATE BLOCKER, pinned as a defect that is present rather than described.
    # `ingest`'s interval high EQUALS its point estimate -- its per-pass medians
    # are coarsely quantised, so the observed median sits at the top of the
    # resample distribution -- which makes `high - point` mathematically zero.
    # Recovering the bound by `ratio_interval.high * ns_per_op` then rounds to
    # ONE ULP BELOW the point at this calibration, and matplotlib refuses a
    # negative `xerr`. Whether it lands negative depends on ns_per_op, which is
    # a wall-clock reading, which is why the failure was intermittent.
    blocking = 4.064160401002506
    result = divergence(units, blocking, resamples=400, seed=94094)
    point = result.ns_per_declared_op["ingest"]
    round_tripped = result.ratio_interval["ingest"].high * result.ns_per_op
    assert point == 0.46875
    assert round_tripped == 0.46874999999999994
    assert round_tripped - point == -float(np.spacing(point))
    # The calibration-free interval at the same calibration does NOT go negative.
    assert result.ns_per_declared_op_interval["ingest"].high - point == 0.0


def test_regression_the_blocking_stage_has_a_zero_width_upper_bar(divergence_result):
    # A zero-width bar is legitimate and is what the fix produces; a negative one
    # never is. Pinned so a future change that reintroduces a rescaling fails.
    interval = divergence_result.ns_per_declared_op_interval["ingest"]
    assert interval.high - divergence_result.ns_per_declared_op["ingest"] == 0.0
    assert interval.low < divergence_result.ns_per_declared_op["ingest"]


# --- invariance under the host calibration --------------------------------


@pytest.mark.parametrize("factor", [0.1, 0.5, 2.0, 10.0, 1000.0])
def test_spread_is_invariant_under_ns_per_op(units, ns_per_op, factor):
    base = divergence(units, ns_per_op, resamples=64)
    scaled = divergence(units, ns_per_op * factor, resamples=64)
    assert scaled.spread == pytest.approx(base.spread, rel=1e-12, abs=0.0)


@pytest.mark.parametrize("factor", [0.1, 2.0, 1000.0])
def test_tau_b_is_invariant_under_ns_per_op(units, ns_per_op, factor):
    base = divergence(units, ns_per_op, resamples=64)
    scaled = divergence(units, ns_per_op * factor, resamples=64)
    assert scaled.tau_b == pytest.approx(base.tau_b, rel=1e-12, abs=0.0)


@pytest.mark.parametrize("factor", [0.1, 2.0, 1000.0])
def test_permutation_and_arg_extrema_are_invariant_under_ns_per_op(
    units, ns_per_op, factor
):
    base = divergence(units, ns_per_op, resamples=64)
    scaled = divergence(units, ns_per_op * factor, resamples=64)
    assert scaled.permutation == base.permutation
    assert (scaled.argmax, scaled.argmin) == (base.argmax, base.argmin)


@pytest.mark.parametrize("factor", [0.25, 4.0])
def test_ratio_levels_scale_inversely_with_ns_per_op(units, ns_per_op, factor):
    base = divergence(units, ns_per_op, resamples=64)
    scaled = divergence(units, ns_per_op * factor, resamples=64)
    for stage in STAGES:
        assert scaled.ratio[stage] == pytest.approx(
            base.ratio[stage] / factor, rel=1e-12, abs=0.0
        )


@pytest.mark.parametrize("ns_per_op_value", [0.0, -1.0, float("nan"), float("inf")])
def test_divergence_refuses_a_non_positive_or_non_finite_ns_per_op(units, ns_per_op_value):
    with pytest.raises(ValueError, match="positive finite"):
        divergence(units, ns_per_op_value, resamples=8)


def test_divergence_is_reproducible_at_a_fixed_seed(units, ns_per_op):
    first = divergence(units, ns_per_op, resamples=128, seed=321)
    second = divergence(units, ns_per_op, resamples=128, seed=321)
    assert first.spread_interval.low == second.spread_interval.low
    assert first.spread_interval.high == second.spread_interval.high


def test_divergence_module_uses_the_declared_resampling_unit():
    # ``hiltimingdiff.divergence`` is shadowed on the package by the function of
    # the same name, so the module object is taken from sys.modules.
    import sys

    module = sys.modules["hiltimingdiff.divergence"]
    assert module.RESAMPLING_UNIT == "pass"
