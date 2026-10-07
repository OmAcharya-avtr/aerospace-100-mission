"""Episode simulation: pairing, references, disturbance samplers, validation."""

from __future__ import annotations

import numpy as np
import pytest

from simplexguard import (
    CostWeights,
    Mode,
    disturbance_sequence,
    simulate_baseline,
    simulate_guarded,
    simulate_unguarded,
    square_wave_reference,
)


def test_square_wave_reference_known_answer():
    ref = square_wave_reference(0.2, 3, n_states=2)
    # Half period 3: steps 0,1,2 -> +0.2; steps 3,4,5 -> -0.2; 6,7,8 -> +0.2
    expected = [0.2, 0.2, 0.2, -0.2, -0.2, -0.2, 0.2, 0.2, 0.2]
    assert [float(ref(k)[0]) for k in range(9)] == pytest.approx(expected)
    assert all(float(ref(k)[1]) == 0.0 for k in range(9))


def test_disturbance_sequence_respects_the_declared_box(plant, rng):
    for mode in ("uniform", "vertex", "zero"):
        w = disturbance_sequence(plant, 500, rng, mode)
        assert w.shape == (500, 2)
        assert np.all(plant.disturbance.contains_many(w, tol=1e-12))


def test_vertex_sampler_only_produces_vertices(plant, rng):
    w = disturbance_sequence(plant, 400, rng, "vertex")
    assert np.allclose(np.abs(w), plant.disturbance.half_widths)


def test_zero_sampler_is_exactly_zero(plant, rng):
    assert np.all(disturbance_sequence(plant, 50, rng, "zero") == 0.0)


def test_scaled_sampler_leaves_the_declared_box(plant, rng):
    w = disturbance_sequence(plant, 300, rng, "vertex", scale=2.0)
    assert not np.any(plant.disturbance.contains_many(w, tol=1e-12))
    assert np.allclose(np.abs(w), 2.0 * plant.disturbance.half_widths)


def test_guarded_run_has_no_violations_inside_the_declared_bound(
    plant, controllers, invariant_set, guard, rng
):
    w = disturbance_sequence(plant, 800, rng, "vertex")
    ref = square_wave_reference(0.18, 80, plant.n_states)
    ep = simulate_guarded(plant, guard, controllers[1], 800, ref, w)
    assert ep.constraint_violations().size == 0
    assert ep.invariant_exits(invariant_set).size == 0
    assert not np.any(ep.certificate_lost)


def test_unguarded_run_violates_on_this_scenario(plant, controllers, rng):
    w = disturbance_sequence(plant, 800, rng, "uniform")
    ref = square_wave_reference(0.18, 80, plant.n_states)
    ep = simulate_unguarded(plant, controllers[1], 800, ref, w)
    assert ep.constraint_violations().size > 0
    assert ep.worst_constraint_residual() > 0.0


def test_baseline_only_run_is_safe_and_expensive(plant, controllers, invariant_set, rng):
    w = disturbance_sequence(plant, 800, rng, "uniform")
    ref = square_wave_reference(0.18, 80, plant.n_states)
    base = simulate_baseline(plant, controllers[0], 800, ref, w)
    unguarded = simulate_unguarded(plant, controllers[1], 800, ref, w)
    assert base.constraint_violations().size == 0
    assert base.invariant_exits(invariant_set).size == 0
    assert base.cost() > unguarded.cost()
    assert base.baseline_fraction == 1.0
    assert np.all(base.modes == Mode.BASELINE.value)


def test_episodes_record_the_terminal_state(plant, controllers, guard, rng):
    w = disturbance_sequence(plant, 25, rng, "uniform")
    ref = square_wave_reference(0.1, 10, plant.n_states)
    ep = simulate_guarded(plant, guard, controllers[1], 25, ref, w)
    assert ep.states.shape == (26, 2)
    assert ep.inputs.shape == (25, 1)
    assert ep.n_steps == 25


def test_pairing_is_exact(plant, controllers, guard, rng):
    w = disturbance_sequence(plant, 100, rng, "uniform")
    ref = square_wave_reference(0.1, 20, plant.n_states)
    a = simulate_guarded(plant, guard, controllers[1], 100, ref, w)
    b = simulate_unguarded(plant, controllers[1], 100, ref, w)
    assert np.array_equal(a.disturbances, b.disturbances)


def test_simulation_is_deterministic(plant, controllers, guard, rng):
    w = disturbance_sequence(plant, 300, rng, "uniform")
    ref = square_wave_reference(0.18, 60, plant.n_states)
    a = simulate_guarded(plant, guard, controllers[1], 300, ref, w)
    b = simulate_guarded(plant, guard, controllers[1], 300, ref, w)
    assert np.array_equal(a.states, b.states)
    assert np.array_equal(a.modes, b.modes)


def test_cost_weights_known_answer():
    # states [[1, 2]], inputs [[3]], references [[0, 0]]
    # cost = 1^2 * 10 + 2^2 * 0.1 + 3^2 * 0.01 = 10 + 0.4 + 0.09 = 10.49
    weights = CostWeights()
    value = weights.evaluate(
        np.array([[1.0, 2.0]]), np.array([[3.0]]), np.array([[0.0, 0.0]])
    )
    assert value == pytest.approx(10.49)


def test_cost_weights_validate_widths():
    weights = CostWeights()
    with pytest.raises(ValueError, match="state weight has length"):
        weights.evaluate(np.zeros((2, 3)), np.zeros((2, 1)), np.zeros((2, 3)))
    with pytest.raises(ValueError, match="input weight has length"):
        weights.evaluate(np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)))


def test_simulate_validates_shapes(plant, controllers, guard, rng):
    ref = square_wave_reference(0.1, 10, plant.n_states)
    w = disturbance_sequence(plant, 10, rng, "uniform")
    with pytest.raises(ValueError, match="n_steps must be at least 1"):
        simulate_guarded(plant, guard, controllers[1], 0, ref, w)
    with pytest.raises(ValueError, match="disturbances must have shape"):
        simulate_guarded(plant, guard, controllers[1], 11, ref, w)
    with pytest.raises(ValueError, match="x0 has length"):
        simulate_guarded(plant, guard, controllers[1], 10, ref, w, x0=np.zeros(3))
    with pytest.raises(ValueError, match="x0 contains non-finite"):
        simulate_guarded(plant, guard, controllers[1], 10, ref, w, x0=np.array([np.nan, 0.0]))


def test_disturbance_sequence_validation(plant, rng):
    with pytest.raises(ValueError, match="unknown disturbance mode"):
        disturbance_sequence(plant, 10, rng, "gaussian")
    with pytest.raises(ValueError, match="n_steps must be non-negative"):
        disturbance_sequence(plant, -1, rng, "uniform")
    with pytest.raises(ValueError, match="scale must be finite"):
        disturbance_sequence(plant, 10, rng, "uniform", scale=-1.0)


def test_square_wave_reference_validation():
    with pytest.raises(ValueError, match="period_steps"):
        square_wave_reference(0.1, 0)
    with pytest.raises(ValueError, match="amplitude must be finite"):
        square_wave_reference(np.nan, 10)
    with pytest.raises(ValueError, match="n_states"):
        square_wave_reference(0.1, 10, n_states=0)
