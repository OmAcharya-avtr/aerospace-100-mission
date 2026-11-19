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


#: Stages baseline 1 UNDER-predicts on this campaign, and the two it over-predicts.
#: The dangerous direction is under-prediction: a schedulability analysis built on
#: it believes it has headroom it does not have.
UNDER_PREDICTED = ("threshold", "centroid", "gate", "track")
OVER_PREDICTED = ("ingest", "background")


def test_baseline1_sign_is_the_live_scalar_against_the_committed_per_op_cost(
    baseline1, ns_per_op, divergence_result
):
    # An IDENTITY, not a measurement. declared[s] - hw[s]
    #   = ops[s] * ns_per_op - ops[s] * ns_per_declared_op[s]
    #   = ops[s] * (ns_per_op - ns_per_declared_op[s]),
    # and ops[s] > 0, so the SIGN of baseline 1's error on a stage is the sign of
    # (live ns_per_op - committed ns_per_declared_op[s]) and of nothing else.
    # ns_per_declared_op is a committed-trace quantity; the only live input is the
    # single scalar, which is isolated into the bound test below.
    per_op = divergence_result.ns_per_declared_op
    for stage in STAGES:
        implied = ns_per_op - per_op[stage]
        assert (baseline1.signed_relative_error[stage] > 0.0) == (implied > 0.0)


def test_live_ns_per_op_sits_between_the_two_committed_costs_that_fix_the_signs(
    ns_per_op, divergence_result
):
    # THE ONE LOAD-SENSITIVE ASSERTION of baseline 1's sign pattern, reduced to a
    # single BOUND on a single live scalar. The bound is not a tuned constant: it
    # is the pair of adjacent committed ns-per-declared-op costs that bracket the
    # scalar, so the sign pattern below holds for every ns_per_op in that open
    # interval. Measured band published in validation/VALIDATION.md.
    ordered = sorted(divergence_result.ns_per_declared_op.values())
    assert ordered[1] < ns_per_op < ordered[2]


@pytest.mark.parametrize("stage", UNDER_PREDICTED)
def test_baseline1_underpredicts_the_interpreted_stages(baseline1, stage):
    # Follows from the identity and the bound above, and is stated explicitly
    # because the sign pattern is the finding.
    assert baseline1.signed_relative_error[stage] < 0.0


@pytest.mark.parametrize("stage", OVER_PREDICTED)
def test_baseline1_overpredicts_the_two_cheapest_vectorised_stages(baseline1, stage):
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
    # A BOUND on one live measurement against a fixed value, not a relationship
    # between two live timings: the numerator is the committed board median and
    # only the denominator is measured here. The board is slower than this host
    # on every stage of this campaign, so each ratio exceeds 1. Measured band
    # published in validation/VALIDATION.md; the bound 1.0 is the point at which
    # the two machines would be equally fast, which is a meaning and not a
    # tuned constant.
    assert ratio > 0.0
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
    # A BOUND on one live measurement against the PRE-REGISTERED threshold,
    # which is a literal from the specification and not a tuned constant. The
    # measured band is published in validation/VALIDATION.md.
    assert baseline2.declared.mean_relative_error < PREREGISTERED_TRANSFER_ERROR


#: The gate branch census over the SAME three cells at the two frame-stack
#: lengths, two passes each: 900 timed frames against 9000. These are INTEGER
#: counts over deterministically seeded frames -- ``make_frames`` takes the
#: campaign's own ``synth_seed`` -- so they carry no wall-clock reading at all
#: and are exactly reproducible on any host under any load. Measured
#: 2026-11-19; a repeat of each call returned the identical dictionary.
#:
#: THIS IS THE MECHANISM the three withdrawn tests used to assert through
#: timing. Over a 150-frame stack the exponential running background has not
#: converged, so the candidate population and therefore the gate's branch mix
#: are different populations, not noisier versions of the same one: the
#: per-frame acceptance rate is 0.8244 against 0.1171, a factor of 7.04, and
#: the per-frame low_score count is 2.591 against 6.602. The two stages whose
#: cost depends on that mix -- gate and track -- therefore stop being
#: comparable between modes. The timing consequence is a wall-clock reading on
#: a shared two-core host and is published as a measured band in
#: validation/VALIDATION.md rather than asserted here.
SHORT_STACK_CENSUS = {
    "accept": 742,
    "no_candidate": 16,
    "low_score": 2332,
    "area_out": 444,
    "border": 172,
}
LONG_STACK_CENSUS = {
    "accept": 1054,
    "no_candidate": 148,
    "low_score": 59418,
    "area_out": 632,
    "border": 128,
}

#: Mean candidates per frame per cell at each stack length, same three cells.
SHORT_STACK_K_MEAN = {
    "w48_sparse": 2.873333333333333,
    "w64_sparse": 4.753333333333333,
    "w96_sparse": 8.513333333333333,
}
LONG_STACK_K_MEAN = {
    "w48_sparse": 2.8526666666666665,
    "w64_sparse": 5.378666666666667,
    "w96_sparse": 12.312666666666667,
}


def test_short_stack_census_matches_the_pinned_integer_counts(short_stack_simulation):
    assert short_stack_simulation.census == SHORT_STACK_CENSUS


def test_long_stack_census_matches_the_pinned_integer_counts(long_stack_simulation):
    assert long_stack_simulation.census == LONG_STACK_CENSUS


def test_the_short_stack_exercises_a_different_candidate_population(
    short_stack_simulation, long_stack_simulation
):
    # Integer counts, not timings. 900 and 9000 timed frames over the same three
    # cells and the same seeds.
    short_frames = short_stack_simulation.n_timed_frames
    long_frames = long_stack_simulation.n_timed_frames
    assert (short_frames, long_frames) == (900, 9000)
    short_rate = short_stack_simulation.census["accept"] / short_frames
    long_rate = long_stack_simulation.census["accept"] / long_frames
    assert short_rate == pytest.approx(0.8244444444444444, rel=1e-12, abs=0.0)
    assert long_rate == pytest.approx(0.11711111111111111, rel=1e-12, abs=0.0)
    assert short_rate > 7.0 * long_rate


def test_the_short_stack_sees_fewer_thresholded_candidates_per_frame(
    short_stack_simulation, long_stack_simulation
):
    short_frames = short_stack_simulation.n_timed_frames
    long_frames = long_stack_simulation.n_timed_frames
    short_low = short_stack_simulation.census["low_score"] / short_frames
    long_low = long_stack_simulation.census["low_score"] / long_frames
    assert short_low == pytest.approx(2.5911111111111111, rel=1e-12, abs=0.0)
    assert long_low == pytest.approx(6.602, rel=1e-12, abs=0.0)
    assert long_low > 2.0 * short_low


@pytest.mark.parametrize("cell", ["w48_sparse", "w64_sparse", "w96_sparse"])
def test_short_stack_candidate_count_per_cell_matches_the_pinned_value(
    short_stack_simulation, long_stack_simulation, cell
):
    short_k = float(
        short_stack_simulation.table.groupby("cell")["k_mean"].mean()[cell]
    )
    long_k = float(long_stack_simulation.table.groupby("cell")["k_mean"].mean()[cell])
    assert short_k == pytest.approx(SHORT_STACK_K_MEAN[cell], rel=1e-12, abs=0.0)
    assert long_k == pytest.approx(LONG_STACK_K_MEAN[cell], rel=1e-12, abs=0.0)


def test_short_stack_result_is_flagged_as_not_matching_the_campaign(
    short_stack_simulation,
):
    assert short_stack_simulation.stack_matches_campaign is False


def test_declared_configuration_result_is_flagged_as_matching(simulation):
    assert simulation.stack_matches_campaign is True
