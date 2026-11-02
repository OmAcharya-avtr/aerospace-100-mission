"""The run log, the miss extraction and the two deadline accountings."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hilverdict.plant import LoopModel
from hilverdict.synthetic import generate_run
from hilverdict.timeline import (
    OverrunCounts,
    RunLog,
    miss_events,
    overrun_counts,
    utilisation,
)


def _log(durations, model=None, states=None, computed=None, prior=None):
    loop = LoopModel() if model is None else model
    steps = len(durations)
    return RunLog(
        model=loop,
        command_rad=np.zeros(steps),
        durations_s=np.asarray(durations, dtype=float),
        states=np.zeros((steps + 1, 2)) if states is None else states,
        computed_rad=np.zeros(steps) if computed is None else computed,
        saturated=np.zeros(steps, dtype=bool),
        host_time_s=np.arange(steps + 1) * loop.sampling.period_s,
        device_time_s=np.arange(steps + 1) * loop.sampling.period_s,
        prior_command_rad=prior,
        label="unit",
    )


def test_run_log_rejects_a_single_iteration():
    with pytest.raises(ValueError, match="at least 2 iterations"):
        _log([0.001])


def test_run_log_rejects_a_mismatched_command_length():
    loop = LoopModel()
    with pytest.raises(ValueError, match="command_rad must have shape"):
        RunLog(
            model=loop,
            command_rad=np.zeros(3),
            durations_s=np.zeros(4),
            states=np.zeros((5, 2)),
            computed_rad=np.zeros(4),
            saturated=np.zeros(4, dtype=bool),
            host_time_s=np.zeros(5),
            device_time_s=np.zeros(5),
        )


def test_run_log_rejects_a_mismatched_state_array():
    loop = LoopModel()
    with pytest.raises(ValueError, match="states must have shape"):
        RunLog(
            model=loop,
            command_rad=np.zeros(4),
            durations_s=np.zeros(4),
            states=np.zeros((4, 2)),
            computed_rad=np.zeros(4),
            saturated=np.zeros(4, dtype=bool),
            host_time_s=np.zeros(5),
            device_time_s=np.zeros(5),
        )


def test_run_log_rejects_a_mismatched_timestamp_series():
    loop = LoopModel()
    with pytest.raises(ValueError, match="host_time_s must have shape"):
        RunLog(
            model=loop,
            command_rad=np.zeros(4),
            durations_s=np.zeros(4),
            states=np.zeros((5, 2)),
            computed_rad=np.zeros(4),
            saturated=np.zeros(4, dtype=bool),
            host_time_s=np.zeros(4),
            device_time_s=np.zeros(5),
        )


def test_run_log_rejects_a_negative_duration():
    with pytest.raises(ValueError, match="negative execution time"):
        _log([0.001, -0.001])


def test_run_log_rejects_a_non_finite_state():
    loop = LoopModel()
    states = np.zeros((5, 2))
    states[2, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        RunLog(
            model=loop,
            command_rad=np.zeros(4),
            durations_s=np.zeros(4),
            states=states,
            computed_rad=np.zeros(4),
            saturated=np.zeros(4, dtype=bool),
            host_time_s=np.zeros(5),
            device_time_s=np.zeros(5),
        )


def test_steps_and_duration():
    log = _log([0.001] * 7)
    assert log.steps == 7
    assert log.duration_s == pytest.approx(0.07, rel=1e-14, abs=0.0)


def test_missed_mask_is_strictly_above_the_deadline():
    log = _log([0.009, 0.010, 0.011])
    assert list(log.missed) == [False, False, True]


def test_prior_command_defaults_to_the_first_computed_command():
    log = _log([0.001, 0.001], computed=np.array([0.3, 0.4]))
    assert log.prior_command == pytest.approx(0.3, rel=0.0, abs=0.0)


def test_prior_command_is_used_when_declared():
    log = _log([0.001, 0.001], computed=np.array([0.3, 0.4]), prior=0.9)
    assert log.prior_command == pytest.approx(0.9, rel=0.0, abs=0.0)


def test_peak_alpha_reads_the_state_array():
    states = np.zeros((4, 2))
    states[2, 0] = -0.07
    log = _log([0.001] * 3, states=states)
    assert log.peak_alpha_rad == pytest.approx(0.07, rel=0.0, abs=0.0)


def test_no_overrun_gives_no_miss_events():
    log = _log([0.005] * 10)
    assert miss_events(log) == ()


def test_a_single_overrun_gives_one_event():
    log = _log([0.005, 0.0125, 0.005], computed=np.array([0.0, 0.1, 0.2]))
    events = miss_events(log)
    assert len(events) == 1
    assert events[0].index == 1


def test_overrun_is_duration_minus_deadline():
    log = _log([0.005, 0.0125], computed=np.array([0.0, 0.1]))
    event = miss_events(log)[0]
    assert event.overrun_s == pytest.approx(0.0025, rel=1e-14, abs=1e-18)


def test_hold_periods_and_remainder_split_the_overrun():
    log = _log([0.005, 0.034], computed=np.array([0.0, 0.1]))
    event = miss_events(log)[0]
    assert event.hold_periods == 2
    assert event.remainder_s == pytest.approx(0.004, rel=1e-12, abs=1e-15)
    assert event.hold_periods * 0.010 + event.remainder_s == pytest.approx(
        event.overrun_s, rel=1e-14, abs=1e-18
    )


def test_overrun_periods_is_the_overrun_over_the_period():
    log = _log([0.005, 0.025], computed=np.array([0.0, 0.1]))
    event = miss_events(log)[0]
    assert event.overrun_periods == pytest.approx(1.5, rel=1e-13, abs=1e-15)


def test_exactly_one_period_of_overrun_gives_one_hold_period():
    log = _log([0.005, 0.020], computed=np.array([0.0, 0.1]))
    event = miss_events(log)[0]
    assert event.hold_periods == 1
    assert event.remainder_s == pytest.approx(0.0, rel=0.0, abs=1e-15)


def test_stale_command_is_the_previous_computed_command():
    log = _log([0.005, 0.0125, 0.005], computed=np.array([0.3, 0.1, 0.2]))
    event = miss_events(log)[0]
    assert event.stale_command_rad == pytest.approx(0.3, rel=0.0, abs=0.0)


def test_a_miss_on_iteration_zero_holds_the_declared_prior_command():
    log = _log([0.0125, 0.005], computed=np.array([0.1, 0.2]), prior=0.7)
    event = miss_events(log)[0]
    assert event.stale_command_rad == pytest.approx(0.7, rel=0.0, abs=0.0)


def test_input_step_is_the_largest_difference_over_the_hold_window():
    log = _log(
        [0.005, 0.034, 0.005, 0.005, 0.005],
        computed=np.array([1.0, 0.9, 0.5, 0.2, 0.3]),
    )
    event = miss_events(log)[0]
    # stale = 1.0; window is computed[1], computed[2], computed[3] = 0.9, 0.5, 0.2
    assert event.input_step_rad == pytest.approx(0.8, rel=1e-15, abs=0.0)


def test_input_step_is_zero_when_the_command_does_not_change():
    log = _log([0.005, 0.0125], computed=np.array([0.4, 0.4]))
    event = miss_events(log)[0]
    assert event.input_step_rad == pytest.approx(0.0, rel=0.0, abs=0.0)


def test_steps_remaining_accounts_for_the_hold():
    log = _log([0.005] * 2 + [0.034] + [0.005] * 7, computed=np.zeros(10))
    event = miss_events(log)[0]
    # index 2, p = 2, so the first clean sample instant is 5 and 5 remain.
    assert event.steps_remaining == 5


def test_steps_remaining_is_clipped_at_zero_for_a_miss_at_the_end():
    log = _log([0.005] * 8 + [0.055, 0.005], computed=np.zeros(10))
    events = [event for event in miss_events(log) if event.index == 8]
    assert len(events) == 1
    assert events[0].steps_remaining == 0


def test_event_carries_the_period_it_was_split_by():
    log = _log([0.005, 0.0125], computed=np.zeros(2))
    assert miss_events(log)[0].period_s == pytest.approx(0.010, rel=0.0, abs=0.0)


def test_events_are_in_iteration_order(loaded_run):
    indices = [event.index for event in miss_events(loaded_run)]
    assert indices == sorted(indices)


def test_overrun_counts_direct_matches_the_mask():
    log = _log([0.005, 0.012, 0.005, 0.014])
    counts = overrun_counts(log)
    assert counts.direct == 2
    assert counts.steps == 4


def test_overrun_counts_cascade_by_hand():
    # T = D = 0.010. durations 0.012, 0.009, 0.009:
    #   c0 = 0 + 0.012 = 0.012 > 0.010      -> late
    #   c1 = max(0.010, 0.012) + 0.009 = 0.021 > 0.020 -> late
    #   c2 = max(0.020, 0.021) + 0.009 = 0.030 > 0.030 -> not late (equal)
    log = _log([0.012, 0.009, 0.009])
    counts = overrun_counts(log)
    assert counts.direct == 1
    assert counts.cascade == 2


def test_cascade_never_undercounts_direct(loaded_run):
    counts = overrun_counts(loaded_run)
    assert counts.cascade >= counts.direct


def test_overrun_count_fractions():
    counts = OverrunCounts(direct=2, cascade=5, steps=10)
    assert counts.direct_fraction == pytest.approx(0.2, rel=1e-15, abs=0.0)
    assert counts.cascade_fraction == pytest.approx(0.5, rel=1e-15, abs=0.0)
    assert counts.ratio == pytest.approx(2.5, rel=1e-15, abs=0.0)


def test_overrun_count_ratio_is_infinite_with_no_direct_overrun():
    counts = OverrunCounts(direct=0, cascade=3, steps=10)
    assert math.isinf(counts.ratio)


def test_overrun_count_fractions_with_no_steps():
    counts = OverrunCounts(direct=0, cascade=0, steps=0)
    assert counts.direct_fraction == pytest.approx(0.0, rel=0.0, abs=0.0)
    assert counts.cascade_fraction == pytest.approx(0.0, rel=0.0, abs=0.0)


def test_utilisation_is_mean_duration_over_the_period():
    log = _log([0.004, 0.006])
    assert utilisation(log) == pytest.approx(0.5, rel=1e-14, abs=0.0)


def test_saturated_steps_counts_the_mask(loaded_run):
    assert loaded_run.saturated_steps == int(
        np.count_nonzero(np.asarray(loaded_run.saturated))
    )


def test_a_clean_profile_has_fewer_misses_than_a_loaded_one():
    clean = generate_run(seed=11, profile="nominal", amplitude_deg=6.0, steps=200)
    loaded = generate_run(seed=11, profile="loaded", amplitude_deg=6.0, steps=200)
    assert len(miss_events(clean)) < len(miss_events(loaded))
