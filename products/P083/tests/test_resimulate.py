"""Baseline 2: the simulator, the counterfactual, the early exit and the cost."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hilverdict.certificate import CertificateModel, certify_run
from hilverdict.plant import LoopModel, zoh_matrices
from hilverdict.resimulate import (
    CONSEQUENTIAL,
    INCONSEQUENTIAL,
    classify_miss,
    peak_divergence,
    reference_run,
    resimulate_run,
    simulate,
    timed_resimulation,
)
from hilverdict.synthetic import doublet_command
from hilverdict.timeline import miss_events


def test_simulate_returns_the_declared_shapes(model, command):
    trajectory = simulate(model, command, None)
    steps = command.shape[0]
    assert trajectory.states.shape == (steps + 1, 2)
    assert trajectory.computed_rad.shape == (steps,)
    assert trajectory.saturated.shape == (steps,)
    assert trajectory.steps == steps


def test_simulate_starts_at_the_given_state(model, command):
    start = np.array([0.01, -0.02])
    trajectory = simulate(model, command, None, start_state=start)
    assert trajectory.states[0] == pytest.approx(start, rel=0.0, abs=0.0)


def test_simulate_defaults_to_the_origin(model, command):
    trajectory = simulate(model, command, None)
    assert trajectory.states[0] == pytest.approx(np.zeros(2), rel=0.0, abs=0.0)


def test_simulate_is_bit_deterministic(model, command):
    first = simulate(model, command, {10: 0.013})
    second = simulate(model, command, {10: 0.013})
    assert np.array_equal(first.states, second.states)
    assert np.array_equal(first.computed_rad, second.computed_rad)


def test_simulate_counts_plant_calls(model, command):
    trajectory = simulate(model, command, None)
    assert trajectory.plant_calls == 4 * model.sampling.substeps * command.shape[0]


def test_simulate_rejects_a_two_dimensional_command(model):
    with pytest.raises(ValueError, match="one-dimensional"):
        simulate(model, np.zeros((4, 2)), None)


def test_simulate_rejects_a_start_index_outside_the_run(model, command):
    with pytest.raises(ValueError, match="start_index must lie"):
        simulate(model, command, None, start_index=command.shape[0])


def test_simulate_rejects_a_miss_before_the_start_index(model, command):
    with pytest.raises(ValueError, match="outside the simulated range"):
        simulate(model, command, {2: 0.004}, start_index=10)


def test_simulate_rejects_a_non_positive_overrun(model, command):
    with pytest.raises(ValueError, match="non-positive overrun"):
        simulate(model, command, {3: -0.001})


def test_simulate_rejects_zero_substeps(model, command):
    with pytest.raises(ValueError, match="substeps must be at least 1"):
        simulate(model, command, None, substeps=0)


def test_simulate_rejects_a_stop_index_at_or_before_the_start(model, command):
    with pytest.raises(ValueError, match="stop_index must lie"):
        simulate(model, command, None, start_index=5, stop_index=5)


def test_stop_index_truncates_the_trajectory(model, command):
    trajectory = simulate(model, command, None, stop_index=20)
    assert trajectory.states.shape == (21, 2)


def test_stop_index_agrees_with_the_full_run_on_the_shared_prefix(model, command):
    whole = simulate(model, command, None)
    piece = simulate(model, command, None, stop_index=25)
    assert np.array_equal(whole.states[:26], piece.states)


def test_reference_run_is_simulate_with_no_misses(model, command):
    assert np.array_equal(
        reference_run(model, command).states, simulate(model, command, None).states
    )


def test_a_miss_with_a_zero_input_step_changes_nothing_at_the_sample_instants(
    model, command, reference
):
    # Iteration 20 holds the command iteration 19 computed; if they are equal
    # there is no input error at all.
    index = 20
    stale = float(reference.computed_rad[index])
    perturbed = simulate(
        model,
        command,
        {index: 0.004},
        start_index=index,
        start_state=reference.states[index],
        prior_command_rad=stale,
    )
    clean = simulate(
        model, command, None, start_index=index, start_state=reference.states[index]
    )
    difference = float(np.max(model.metric.norm(perturbed.states - clean.states)))
    assert difference < 1e-8


def test_a_miss_changes_the_trajectory_when_the_command_was_about_to_change(
    model, command, reference
):
    changes = np.nonzero(np.diff(np.asarray(command)) != 0.0)[0] + 1
    index = int(changes[0])
    perturbed = simulate(
        model,
        command,
        {index: 0.008},
        start_index=index,
        start_state=reference.states[index],
        prior_command_rad=float(reference.computed_rad[index - 1]),
    )
    difference = float(
        np.max(model.metric.norm(perturbed.states - reference.states[index:]))
    )
    assert difference > 0.0


def test_a_longer_overrun_diverges_at_least_as_much(model, command, reference):
    changes = np.nonzero(np.diff(np.asarray(command)) != 0.0)[0] + 1
    index = int(changes[0])
    peaks = []
    for overrun in (0.001, 0.004, 0.009):
        perturbed = simulate(
            model,
            command,
            {index: overrun},
            start_index=index,
            start_state=reference.states[index],
            prior_command_rad=float(reference.computed_rad[index - 1]),
        )
        peaks.append(
            float(np.max(model.metric.norm(perturbed.states - reference.states[index:])))
        )
    assert peaks[1] > peaks[0]
    assert peaks[2] > peaks[1]


def test_the_linear_plant_reproduces_the_closed_form_hold_response(linear_plant):
    model = LoopModel(plant=linear_plant)
    steps = 40
    command = doublet_command(steps, math.radians(6.0))
    reference = simulate(model, command, None)
    index = 12
    overrun = 0.004
    stale = float(reference.computed_rad[index - 1])
    fresh = float(reference.computed_rad[index])
    jacobian = linear_plant.jacobian(0.0)
    column = linear_plant.input_matrix
    _state, integral = zoh_matrices(jacobian, column, overrun)
    tail, _unused = zoh_matrices(jacobian, column, model.sampling.period_s - overrun)
    expected = tail @ (integral.ravel() * (stale - fresh))
    perturbed = simulate(
        model,
        command,
        {index: overrun},
        start_index=index,
        start_state=reference.states[index],
        prior_command_rad=stale,
    )
    measured = perturbed.states[1] - reference.states[index + 1]
    assert measured == pytest.approx(expected, rel=1e-6, abs=1e-12)


def test_prefix_reuse_is_exact(model, command, reference):
    for event in miss_events(
        _fake_log(model, command, reference)
    ):
        whole = simulate(model, command, {event.index: event.overrun_s})
        tail = simulate(
            model,
            command,
            {event.index: event.overrun_s},
            start_index=event.index,
            start_state=reference.states[event.index],
            prior_command_rad=event.stale_command_rad,
        )
        assert np.array_equal(whole.states[event.index:], tail.states)


def _fake_log(model, command, reference):
    from hilverdict.timeline import RunLog

    steps = command.shape[0]
    durations = np.full(steps, 0.004)
    durations[15] = 0.016
    durations[30] = 0.0245
    return RunLog(
        model=model,
        command_rad=command,
        durations_s=durations,
        states=reference.states,
        computed_rad=reference.computed_rad,
        saturated=reference.saturated,
        host_time_s=np.arange(steps + 1) * model.sampling.period_s,
        device_time_s=np.arange(steps + 1) * model.sampling.period_s,
    )


def test_peak_divergence_returns_a_four_tuple(model, command, reference):
    log = _fake_log(model, command, reference)
    event = miss_events(log)[0]
    value, where, calls, steps = peak_divergence(
        model, command, reference.states, event
    )
    assert value >= 0.0
    assert event.index <= where <= command.shape[0]
    assert calls > 0
    assert steps > 0


def test_peak_divergence_rejects_a_mismatched_reference(model, command, reference):
    log = _fake_log(model, command, reference)
    with pytest.raises(ValueError, match="reference_states must have shape"):
        peak_divergence(model, command, reference.states[:-1], miss_events(log)[0])


def test_peak_divergence_requires_constants_for_the_early_exit(
    model, command, reference
):
    log = _fake_log(model, command, reference)
    with pytest.raises(ValueError, match="early_exit requires"):
        peak_divergence(
            model, command, reference.states, miss_events(log)[0], early_exit=True
        )


def test_early_exit_reproduces_the_full_horizon_peak(model, command, reference):
    log = _fake_log(model, command, reference)
    constants = CertificateModel.build(model, command.shape[0])
    for event in miss_events(log):
        full, _w1, calls_full, _s1 = peak_divergence(
            model, command, reference.states, event
        )
        early, _w2, calls_early, _s2 = peak_divergence(
            model, command, reference.states, event,
            constants=constants, early_exit=True,
        )
        assert early == pytest.approx(full, rel=0.0, abs=0.0)
        assert calls_early <= calls_full


def test_classify_miss_labels_a_tiny_overrun_inconsequential(
    model, command, reference
):
    from hilverdict.timeline import MissEvent

    event = MissEvent(
        index=20,
        overrun_s=1e-6,
        hold_periods=0,
        remainder_s=1e-6,
        period_s=model.sampling.period_s,
        stale_command_rad=float(reference.computed_rad[19]),
        input_step_rad=0.0,
        state_at_miss=reference.states[20].copy(),
        steps_remaining=command.shape[0] - 21,
    )
    result = classify_miss(model, command, reference.states, event, 1.0)
    assert result.verdict == INCONSEQUENTIAL
    assert result.consequential is False


def test_classify_miss_labels_a_large_overrun_consequential(model, command, reference):
    changes = np.nonzero(np.diff(np.asarray(command)) != 0.0)[0] + 1
    index = int(changes[0])
    from hilverdict.timeline import MissEvent

    event = MissEvent(
        index=index,
        overrun_s=0.028,
        hold_periods=2,
        remainder_s=0.008,
        period_s=model.sampling.period_s,
        stale_command_rad=float(reference.computed_rad[index - 1]),
        input_step_rad=abs(
            float(reference.computed_rad[index - 1] - reference.computed_rad[index])
        ),
        state_at_miss=reference.states[index].copy(),
        steps_remaining=command.shape[0] - (index + 3),
    )
    result = classify_miss(model, command, reference.states, event, 1.0)
    assert result.verdict == CONSEQUENTIAL
    assert result.consequential is True


def test_classify_miss_rejects_a_non_positive_tolerance(model, command, reference):
    log = _fake_log(model, command, reference)
    with pytest.raises(ValueError, match="tolerance must be positive"):
        classify_miss(model, command, reference.states, miss_events(log)[0], 0.0)


def test_resimulate_run_covers_every_miss(long_run):
    assert len(resimulate_run(long_run, 1.0)) == len(miss_events(long_run))


def test_resimulate_run_honours_a_subset(long_run):
    certificate = certify_run(long_run, 1.0)
    subset = tuple(item.event.index for item in certificate.undecided)
    results = resimulate_run(long_run, 1.0, subset=subset)
    assert {result.event.index for result in results} == set(subset)


def test_resimulate_run_with_an_empty_subset_returns_nothing(long_run):
    assert resimulate_run(long_run, 1.0, subset=()) == ()


def test_timed_resimulation_returns_a_positive_wall_time(loaded_run):
    results, elapsed = timed_resimulation(loaded_run, 1.0)
    assert len(results) == len(miss_events(loaded_run))
    assert elapsed > 0.0


def test_plant_calls_grow_with_the_remaining_horizon(long_run):
    results = resimulate_run(long_run, 1.0)
    pairs = sorted(
        (result.event.steps_remaining, result.plant_calls) for result in results
    )
    assert pairs[0][1] <= pairs[-1][1]


def test_substep_count_changes_the_peak_only_slightly(model, command, reference):
    log = _fake_log(model, command, reference)
    event = miss_events(log)[0]
    coarse, _w1, _c1, _s1 = peak_divergence(
        model, command, reference.states, event, substeps=4
    )
    fine, _w2, _c2, _s2 = peak_divergence(
        model, command, reference.states, event, substeps=32
    )
    assert coarse == pytest.approx(fine, rel=1e-6, abs=1e-12)
