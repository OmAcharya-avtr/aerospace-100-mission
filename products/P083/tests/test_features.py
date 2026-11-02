"""The cheap log features: order, units, windows and edge cases."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hilverdict.features import (
    ACTIVITY_WINDOW,
    FEATURE_NAMES,
    WINDOW,
    feature_matrix,
    feature_vector,
)
from hilverdict.plant import LoopModel
from hilverdict.synthetic import generate_run
from hilverdict.timeline import RunLog, miss_events


def _log(durations, computed, command=None, states=None):
    loop = LoopModel()
    steps = len(durations)
    return RunLog(
        model=loop,
        command_rad=np.zeros(steps) if command is None else np.asarray(command, float),
        durations_s=np.asarray(durations, dtype=float),
        states=np.zeros((steps + 1, 2)) if states is None else states,
        computed_rad=np.asarray(computed, dtype=float),
        saturated=np.zeros(steps, dtype=bool),
        host_time_s=np.arange(steps + 1) * loop.sampling.period_s,
        device_time_s=np.arange(steps + 1) * loop.sampling.period_s,
    )


def test_feature_names_are_unique():
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)


def test_feature_count_is_fourteen():
    assert len(FEATURE_NAMES) == 14


def test_declared_windows():
    assert WINDOW == 10
    assert ACTIVITY_WINDOW == 5


def test_feature_vector_length_matches_the_names(loaded_run):
    event = miss_events(loaded_run)[0]
    assert feature_vector(loaded_run, event).shape == (len(FEATURE_NAMES),)


def test_feature_vector_is_finite(long_run):
    for event in miss_events(long_run):
        assert np.all(np.isfinite(feature_vector(long_run, event)))


def test_overrun_features_match_the_event(loaded_run):
    event = miss_events(loaded_run)[0]
    vector = feature_vector(loaded_run, event)
    assert vector[FEATURE_NAMES.index("overrun_s")] == pytest.approx(
        event.overrun_s, rel=0.0, abs=0.0
    )
    assert vector[FEATURE_NAMES.index("overrun_periods")] == pytest.approx(
        event.overrun_periods, rel=0.0, abs=0.0
    )
    assert vector[FEATURE_NAMES.index("hold_periods")] == pytest.approx(
        float(event.hold_periods), rel=0.0, abs=0.0
    )
    assert vector[FEATURE_NAMES.index("input_step_rad")] == pytest.approx(
        event.input_step_rad, rel=0.0, abs=0.0
    )


def test_remaining_fraction_is_between_zero_and_one(long_run):
    index = FEATURE_NAMES.index("remaining_fraction")
    for event in miss_events(long_run):
        value = feature_vector(long_run, event)[index]
        assert 0.0 <= value <= 1.0


def test_misses_in_window_counts_only_earlier_misses():
    durations = [0.004] * 20
    for position in (5, 6, 9):
        durations[position] = 0.012
    durations[12] = 0.012
    log = _log(durations, np.arange(20) * 0.01)
    event = [item for item in miss_events(log) if item.index == 12][0]
    vector = feature_vector(log, event)
    assert vector[FEATURE_NAMES.index("misses_in_window")] == pytest.approx(
        3.0, rel=0.0, abs=0.0
    )


def test_consecutive_misses_counts_the_immediately_preceding_run():
    durations = [0.004] * 20
    for position in (7, 8, 9, 10):
        durations[position] = 0.012
    log = _log(durations, np.arange(20) * 0.01)
    event = [item for item in miss_events(log) if item.index == 10][0]
    vector = feature_vector(log, event)
    assert vector[FEATURE_NAMES.index("consecutive_misses")] == pytest.approx(
        3.0, rel=0.0, abs=0.0
    )


def test_consecutive_misses_is_zero_for_an_isolated_miss():
    durations = [0.004] * 20
    durations[10] = 0.012
    log = _log(durations, np.arange(20) * 0.01)
    event = miss_events(log)[0]
    assert feature_vector(log, event)[
        FEATURE_NAMES.index("consecutive_misses")
    ] == pytest.approx(0.0, rel=0.0, abs=0.0)


def test_command_activity_is_the_mean_absolute_difference():
    computed = np.array([0.0, 0.1, 0.3, 0.6, 1.0, 1.5, 2.1, 2.8])
    durations = [0.004] * 8
    durations[6] = 0.012
    log = _log(durations, computed)
    event = miss_events(log)[0]
    # The window is computed[index - ACTIVITY_WINDOW : index + 1] = computed[1:7]
    # = 0.1, 0.3, 0.6, 1.0, 1.5, 2.1, whose five successive differences are
    # 0.2, 0.3, 0.4, 0.5, 0.6 and whose mean is 0.4.
    assert feature_vector(log, event)[
        FEATURE_NAMES.index("command_activity_rad")
    ] == pytest.approx(0.4, rel=1e-13, abs=1e-15)


def test_command_activity_is_zero_for_a_miss_at_iteration_zero():
    durations = [0.012] + [0.004] * 5
    log = _log(durations, np.arange(6) * 0.01)
    event = miss_events(log)[0]
    assert feature_vector(log, event)[
        FEATURE_NAMES.index("command_activity_rad")
    ] == pytest.approx(0.0, rel=0.0, abs=0.0)


def test_local_utilisation_is_a_mean_over_the_window():
    durations = [0.005] * 20
    durations[15] = 0.015
    log = _log(durations, np.zeros(20))
    event = miss_events(log)[0]
    expected = float(np.mean(durations[5:16]) / 0.010)
    assert feature_vector(log, event)[
        FEATURE_NAMES.index("local_utilisation")
    ] == pytest.approx(expected, rel=1e-13, abs=0.0)


def test_steps_to_switch_is_zero_at_a_switch():
    command = np.concatenate([np.full(10, 0.1), np.full(10, -0.1)])
    durations = [0.004] * 20
    durations[10] = 0.012
    log = _log(durations, np.zeros(20), command=command)
    event = miss_events(log)[0]
    assert feature_vector(log, event)[
        FEATURE_NAMES.index("steps_to_switch")
    ] == pytest.approx(0.0, rel=0.0, abs=0.0)


def test_steps_to_switch_is_the_run_length_when_the_command_never_changes():
    durations = [0.004] * 20
    durations[5] = 0.012
    log = _log(durations, np.zeros(20), command=np.zeros(20))
    event = miss_events(log)[0]
    assert feature_vector(log, event)[
        FEATURE_NAMES.index("steps_to_switch")
    ] == pytest.approx(20.0, rel=0.0, abs=0.0)


def test_state_features_read_the_state_at_the_miss():
    states = np.zeros((11, 2))
    states[4] = [0.03, -0.07]
    durations = [0.004] * 10
    durations[4] = 0.012
    log = _log(durations, np.zeros(10), states=states)
    event = miss_events(log)[0]
    vector = feature_vector(log, event)
    assert vector[FEATURE_NAMES.index("abs_alpha_rad")] == pytest.approx(
        0.03, rel=1e-15, abs=0.0
    )
    assert vector[FEATURE_NAMES.index("abs_rate_rad_s")] == pytest.approx(
        0.07, rel=1e-15, abs=0.0
    )


def test_tracking_error_is_the_incidence_minus_the_command():
    states = np.zeros((11, 2))
    states[4] = [0.03, 0.0]
    command = np.full(10, 0.05)
    durations = [0.004] * 10
    durations[4] = 0.012
    log = _log(durations, np.zeros(10), command=command, states=states)
    event = miss_events(log)[0]
    assert feature_vector(log, event)[
        FEATURE_NAMES.index("tracking_error_rad")
    ] == pytest.approx(0.02, rel=1e-13, abs=1e-16)


def test_feature_matrix_shape(long_run):
    matrix, events = feature_matrix(long_run)
    assert matrix.shape == (len(events), len(FEATURE_NAMES))


def test_feature_matrix_rows_match_feature_vector(loaded_run):
    matrix, events = feature_matrix(loaded_run)
    for row, event in zip(matrix, events, strict=True):
        assert row == pytest.approx(
            feature_vector(loaded_run, event), rel=0.0, abs=0.0
        )


def test_feature_matrix_of_a_clean_run_is_correctly_shaped_and_empty():
    log = _log([0.004] * 10, np.zeros(10))
    matrix, events = feature_matrix(log)
    assert events == ()
    assert matrix.shape == (0, len(FEATURE_NAMES))


def test_feature_matrix_accepts_an_explicit_event_subset(long_run):
    events = miss_events(long_run)[:3]
    matrix, returned = feature_matrix(long_run, events)
    assert returned == events
    assert matrix.shape == (3, len(FEATURE_NAMES))


def test_features_are_deterministic():
    first = generate_run(seed=9, profile="loaded", amplitude_deg=6.0, steps=80)
    second = generate_run(seed=9, profile="loaded", amplitude_deg=6.0, steps=80)
    assert np.array_equal(feature_matrix(first)[0], feature_matrix(second)[0])


def test_amplitude_changes_the_input_step_feature():
    small = generate_run(seed=13, profile="loaded", amplitude_deg=3.0, steps=200)
    large = generate_run(seed=13, profile="loaded", amplitude_deg=8.0, steps=200)
    column = FEATURE_NAMES.index("input_step_rad")
    assert float(np.max(feature_matrix(large)[0][:, column])) > float(
        np.max(feature_matrix(small)[0][:, column])
    )


def test_input_step_feature_is_in_radians_and_physically_small(long_run):
    column = FEATURE_NAMES.index("input_step_rad")
    matrix, _events = feature_matrix(long_run)
    assert float(np.max(matrix[:, column])) < math.radians(90.0)
