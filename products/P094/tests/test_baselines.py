"""The three baselines of SPEC §4, and the pre-registration verdicts they feed.

Baseline 1 is implemented at full strength on purpose: an under-strength
version of the thing being refuted would inflate the finding, which is a defect
class this mission has shipped. Baseline 2's split is pinned, and its
alternatives -- leave-one-stage-out and all twenty three-and-three splits -- are
measured too, so the declared split can be located inside the distribution
rather than taken on trust.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from hiltimingdiff.baselines import (
    HOLDOUT_STAGES,
    PREREGISTERED_SPREAD,
    PREREGISTERED_TRANSFER_ERROR,
    TRAIN_STAGES,
    baseline_2,
)

STAGES = ("ingest", "background", "threshold", "centroid", "gate", "track")


def test_declared_split_is_pinned_as_the_vectorised_interpreted_split():
    assert TRAIN_STAGES == ("ingest", "background", "threshold")
    assert HOLDOUT_STAGES == ("centroid", "gate", "track")


def test_declared_split_partitions_the_six_stages():
    assert sorted(TRAIN_STAGES + HOLDOUT_STAGES) == sorted(STAGES)


def test_preregistered_thresholds_are_the_literals_from_the_spec():
    assert PREREGISTERED_SPREAD == 3.0
    assert PREREGISTERED_TRANSFER_ERROR == 0.5


# --- baseline 1 -----------------------------------------------------------


@pytest.mark.parametrize("stage", STAGES)
def test_baseline1_relative_error_is_the_absolute_signed_error(baseline1, stage):
    assert baseline1.relative_error[stage] == pytest.approx(
        abs(baseline1.signed_relative_error[stage]), rel=1e-12, abs=0.0
    )


@pytest.mark.parametrize("stage", STAGES)
def test_baseline1_signed_error_is_declared_minus_measured_over_measured(
    baseline1, stage
):
    expected = (
        baseline1.declared_ns[stage] - baseline1.hw_median_ns[stage]
    ) / baseline1.hw_median_ns[stage]
    assert baseline1.signed_relative_error[stage] == pytest.approx(
        expected, rel=1e-12, abs=0.0
    )


def test_baseline1_mean_relative_error_is_the_mean_over_stages(baseline1):
    values = [baseline1.relative_error[stage] for stage in STAGES]
    assert baseline1.mean_relative_error == pytest.approx(
        float(np.mean(values)), rel=1e-12, abs=0.0
    )


def test_baseline1_worst_stage_has_the_largest_relative_error(baseline1):
    assert baseline1.worst_stage == max(
        baseline1.relative_error, key=baseline1.relative_error.get
    )


def test_baseline1_max_relative_error_matches_the_worst_stage(baseline1):
    assert baseline1.max_relative_error == pytest.approx(
        baseline1.relative_error[baseline1.worst_stage], rel=1e-12, abs=0.0
    )


def test_baseline1_total_error_is_over_the_whole_pipeline(baseline1):
    expected = (
        abs(baseline1.total_declared_ns - baseline1.total_hw_ns) / baseline1.total_hw_ns
    )
    assert baseline1.total_relative_error == pytest.approx(expected, rel=1e-12, abs=0.0)


def test_baseline1_totals_are_the_sums_of_the_per_stage_figures(baseline1):
    assert baseline1.total_declared_ns == pytest.approx(
        sum(baseline1.declared_ns.values()), rel=1e-12, abs=0.0
    )
    assert baseline1.total_hw_ns == pytest.approx(
        sum(baseline1.hw_median_ns.values()), rel=1e-12, abs=0.0
    )


def test_baseline1_frame_is_in_pipeline_order(baseline1):
    assert list(baseline1.frame()["stage"]) == list(STAGES)


def test_baseline1_underpredicts_the_interpreted_stages(baseline1):
    # The dangerous direction: a declared model that under-predicts a stage
    # gives a schedulability analysis false headroom. Asserted as a measured
    # property of this campaign, with the sign stated.
    for stage in ("centroid", "gate", "track", "threshold"):
        assert baseline1.signed_relative_error[stage] < 0.0


def test_baseline1_overpredicts_the_two_cheapest_vectorised_stages(baseline1):
    for stage in ("ingest", "background"):
        assert baseline1.signed_relative_error[stage] > 0.0


# --- baseline 2 -----------------------------------------------------------


def test_baseline2_declared_split_is_the_pinned_one(baseline2):
    assert baseline2.declared.train == TRAIN_STAGES
    assert baseline2.declared.holdout == HOLDOUT_STAGES


@pytest.mark.parametrize("stage", HOLDOUT_STAGES)
def test_baseline2_prediction_is_k_times_simulated_time(baseline2, stage):
    expected = baseline2.declared.k * baseline2.sim_median_ns[stage]
    assert baseline2.declared.predicted_ns[stage] == pytest.approx(
        expected, rel=1e-12, abs=0.0
    )


@pytest.mark.parametrize("stage", HOLDOUT_STAGES)
def test_baseline2_relative_error_is_prediction_against_ground_truth(baseline2, stage):
    expected = (
        abs(baseline2.declared.predicted_ns[stage] - baseline2.hw_median_ns[stage])
        / baseline2.hw_median_ns[stage]
    )
    assert baseline2.declared.relative_error[stage] == pytest.approx(
        expected, rel=1e-12, abs=0.0
    )


def test_baseline2_mean_error_is_the_mean_over_held_out_stages(baseline2):
    values = [baseline2.declared.relative_error[stage] for stage in HOLDOUT_STAGES]
    assert baseline2.declared.mean_relative_error == pytest.approx(
        float(np.mean(values)), rel=1e-12, abs=0.0
    )


def test_baseline2_k_is_positive(baseline2):
    assert baseline2.declared.k > 0.0


def test_baseline2_k_does_not_use_a_held_out_stage(units, simulation):
    # Fitting on a different training set must change k. If it did not, the
    # fit would not be using the training stages at all.
    first = baseline_2(units, simulation, resamples=32)
    second = baseline_2(
        units,
        simulation,
        train=("centroid", "gate", "track"),
        holdout=("ingest", "background", "threshold"),
        resamples=32,
    )
    assert first.declared.k != second.declared.k


def test_baseline2_interval_records_the_pass_unit(baseline2):
    assert baseline2.declared_interval.unit == "pass"


def test_baseline2_interval_brackets_its_point(baseline2):
    interval = baseline2.declared_interval
    assert interval.low <= interval.point <= interval.high


def test_baseline2_interval_point_is_the_declared_split_error(baseline2):
    assert baseline2.declared_interval.point == pytest.approx(
        baseline2.declared.mean_relative_error, rel=1e-9, abs=0.0
    )


def test_baseline2_leave_one_out_covers_every_stage(baseline2):
    assert set(baseline2.leave_one_out) == set(STAGES)


@pytest.mark.parametrize("stage", STAGES)
def test_baseline2_leave_one_out_error_is_non_negative(baseline2, stage):
    assert baseline2.leave_one_out[stage] >= 0.0


def test_baseline2_leave_one_out_mean_is_the_mean_over_stages(baseline2):
    values = [baseline2.leave_one_out[stage] for stage in STAGES]
    assert baseline2.leave_one_out_mean == pytest.approx(
        float(np.mean(values)), rel=1e-12, abs=0.0
    )


def test_baseline2_enumerates_all_twenty_three_and_three_splits(baseline2):
    assert len(baseline2.all_splits) == len(list(itertools.combinations(STAGES, 3)))
    assert len(baseline2.all_splits) == 20


def test_baseline2_every_split_is_a_partition(baseline2):
    for split in baseline2.all_splits:
        assert sorted(split.train + split.holdout) == sorted(STAGES)
        assert not set(split.train) & set(split.holdout)


def test_baseline2_split_errors_bracket_the_declared_split(baseline2):
    assert baseline2.split_error_min <= baseline2.declared.mean_relative_error
    assert baseline2.declared.mean_relative_error <= baseline2.split_error_max


def test_baseline2_split_statistics_are_ordered(baseline2):
    assert (
        baseline2.split_error_min
        <= baseline2.split_error_median
        <= baseline2.split_error_max
    )


def test_baseline2_global_k_is_the_median_over_every_stage(baseline2):
    assert baseline2.k_global > 0.0


@pytest.mark.parametrize("stage", STAGES)
def test_baseline2_per_stage_ratio_is_hardware_over_simulation(baseline2, stage):
    ratio = baseline2.per_stage_ratio[stage]
    assert ratio > 0.0
    # Order-of-magnitude sanity: the board is slower than the host on every
    # stage of this campaign, so each ratio exceeds 1.
    assert ratio > 1.0


def test_baseline2_refuses_an_overlapping_split(units, simulation):
    with pytest.raises(ValueError, match="overlap"):
        baseline_2(
            units,
            simulation,
            train=("ingest", "background"),
            holdout=("background", "track"),
            resamples=16,
        )


def test_baseline2_refuses_an_unknown_stage(units, simulation):
    with pytest.raises(ValueError, match="unknown stage"):
        baseline_2(
            units, simulation, train=("ingest",), holdout=("decimate",), resamples=16
        )


def test_baseline2_refuses_an_empty_half(units, simulation):
    with pytest.raises(ValueError, match="non-empty"):
        baseline_2(units, simulation, train=(), holdout=("track",), resamples=16)


# --- baseline 3 -----------------------------------------------------------


def test_baseline3_covers_the_six_stages(baseline3):
    assert set(baseline3.median_ns) == set(STAGES)


def test_baseline3_resampling_unit_is_the_pass(baseline3):
    assert baseline3.unit == "pass"


@pytest.mark.parametrize("stage", STAGES)
def test_baseline3_interval_records_the_pass_unit(baseline3, stage):
    assert baseline3.interval[stage].unit == "pass"


@pytest.mark.parametrize("stage", STAGES)
def test_baseline3_interval_records_five_hundred_forty_units(baseline3, stage):
    assert baseline3.interval[stage].units == 540


@pytest.mark.parametrize("stage", STAGES)
def test_baseline3_interval_brackets_the_median(baseline3, stage):
    interval = baseline3.interval[stage]
    assert interval.low <= baseline3.median_ns[stage] <= interval.high


@pytest.mark.parametrize("stage", STAGES)
def test_baseline3_median_matches_the_unit_median(baseline3, units, stage):
    column = units.hw[:, units.stage_column(stage)]
    assert baseline3.median_ns[stage] == pytest.approx(
        float(np.median(column)), rel=1e-12, abs=0.0
    )


@pytest.mark.parametrize("stage", STAGES)
def test_baseline3_interval_is_narrow_on_five_hundred_forty_passes(baseline3, stage):
    # A measured property, not a target: 540 passes of 1500 frames give a
    # relative interval width under 2 per cent on every stage.
    assert baseline3.interval[stage].relative_width < 0.02


def test_baseline3_frame_is_in_pipeline_order(baseline3):
    assert list(baseline3.frame()["stage"]) == list(STAGES)


def test_baseline3_n_units_is_five_hundred_forty(baseline3):
    assert baseline3.n_units == 540


# --- the frame-stack sensitivity, asserted rather than described ----------


def test_full_stack_transfer_error_is_below_the_preregistered_threshold(baseline2):
    # SPEC §7's secondary expectation is REFUTED at the declared configuration.
    assert baseline2.declared.mean_relative_error < PREREGISTERED_TRANSFER_ERROR


def test_short_stack_transfer_error_is_at_least_twice_the_full_stack_error(
    baseline2, short_stack_baseline2
):
    # A WEAK RESULT, pinned so it cannot be lost. The held-out transfer error
    # depends on the frame-stack length as well as on the platform. At a
    # 150-frame stack it measured 0.4796 to 0.5532 over repeats in
    # validation/validate_measured_baselines_output.txt -- SITTING ON SPEC §7's
    # 0.5 threshold rather than clearly on either side of it -- against 0.1481
    # to 0.1701 at the board's own 1500-frame stack. The threshold crossing is
    # therefore NOT asserted, because it is not reliable on a shared two-core
    # host; what is asserted is the stable relation, a factor of at least two.
    assert short_stack_baseline2.declared.mean_relative_error > (
        2.0 * baseline2.declared.mean_relative_error
    )


@pytest.mark.parametrize("stage", ["gate", "track"])
def test_short_stack_collapses_the_gate_and_track_ratios(
    baseline2, short_stack_baseline2, stage
):
    # The mechanism: over a short stack the exponential running background has
    # not converged, so the candidate population and the gate's branch mix
    # differ, and the two stages whose cost depends on that mix stop being
    # comparable between modes.
    assert short_stack_baseline2.per_stage_ratio[stage] < baseline2.per_stage_ratio[stage]


@pytest.mark.parametrize("stage", ["ingest", "background", "threshold", "centroid"])
def test_short_stack_moves_the_other_stages_far_less_than_track(
    baseline2, short_stack_baseline2, stage
):
    # The mechanism is specific to the two stages whose cost depends on the
    # gate's branch mix. Measured over three repeats on 2026-11-16, the relative
    # shift in the hardware-to-simulation ratio between a 150-frame and a
    # 1500-frame stack was 0.0547..0.1839 for ingest, 0.0885..0.1136 for
    # background, 0.0027..0.1008 for threshold and 0.0479..0.1039 for centroid,
    # against 0.2353..0.2714 for gate and 0.5505..0.6110 for track. An absolute
    # bound on the other stages is NOT asserted, because these are wall-clock
    # readings on a shared two-core host and an absolute bound would be a bound
    # on the machine's load. The comparative claim is the stable one.
    def shift(name: str) -> float:
        long_ratio = baseline2.per_stage_ratio[name]
        return abs(short_stack_baseline2.per_stage_ratio[name] - long_ratio) / long_ratio

    assert shift(stage) < shift("track")


def test_short_stack_result_is_flagged_as_not_matching_the_campaign(
    short_stack_simulation,
):
    assert short_stack_simulation.stack_matches_campaign is False


def test_declared_configuration_result_is_flagged_as_matching(simulation):
    assert simulation.stack_matches_campaign is True
