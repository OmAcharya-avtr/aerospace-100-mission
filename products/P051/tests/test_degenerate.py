"""Edge cases and degenerate configurations.

These are the inputs a user hits by accident. Each one must produce either a
correct answer or an actionable error, never a silently wrong number.
"""

from __future__ import annotations

import numpy as np
import pytest

from simplexguard import (
    BaselineController,
    Mode,
    Plant,
    SimplexGuard,
    account,
    box,
    disturbance_sequence,
    reference_controllers,
    reference_plant,
    robust_invariant_set,
    simulate_baseline,
    simulate_guarded,
    simulate_unguarded,
    square_wave_reference,
    verify_robust_invariance,
)


def test_zero_disturbance_gives_the_maximal_output_admissible_set(plant, controllers):
    # With W = {0} the erosion is a no-op, so the recursion reduces to the
    # maximal output admissible set of Gilbert and Tan (1991). It must still
    # converge and still be invariant.
    nominal = plant.with_disturbance(box([0.0, 0.0]))
    result = robust_invariant_set(nominal, controllers[0])
    assert result.converged
    report = verify_robust_invariance(nominal, controllers[0], result.polytope)
    assert report["robustly_invariant"]
    # And it is at least as large as the set with a non-zero disturbance.
    with_w = robust_invariant_set(plant, controllers[0])
    assert result.polytope.contains_polytope(with_w.polytope, tol=1e-8)


def test_zero_disturbance_guard_has_a_zero_erosion(plant, controllers):
    nominal = plant.with_disturbance(box([0.0, 0.0]))
    invariant = robust_invariant_set(nominal, controllers[0]).polytope
    guard = SimplexGuard(nominal, controllers[0], invariant)
    assert np.all(guard.disturbance_offsets == 0.0)
    assert guard.eroded_set.b == pytest.approx(invariant.b)


def test_a_performance_controller_equal_to_the_baseline_never_fires(
    plant, controllers, invariant_set
):
    # If the performance controller IS the baseline, the guard must never take
    # over from a state inside S, because S is invariant under it.
    from simplexguard import PerformanceController

    guard = SimplexGuard(plant, controllers[0], invariant_set)
    mirror = PerformanceController(
        gain=controllers[0].gain, input_set=plant.input_constraints
    )
    rng = np.random.default_rng(11)
    w = disturbance_sequence(plant, 600, rng, "vertex")
    zero_reference = square_wave_reference(0.0, 100, plant.n_states)
    episode = simulate_guarded(plant, guard, mirror, 600, zero_reference, w)
    assert episode.baseline_fraction == 0.0
    assert np.all(episode.modes == Mode.PERFORMANCE.value)
    assert episode.constraint_violations().size == 0


def test_a_zero_reference_makes_the_guard_almost_idle(plant, controllers, invariant_set):
    guard = SimplexGuard(plant, controllers[0], invariant_set)
    rng = np.random.default_rng(12)
    w = disturbance_sequence(plant, 600, rng, "uniform")
    episode = simulate_guarded(
        plant, guard, controllers[1], 600, square_wave_reference(0.0, 100, 2), w
    )
    assert episode.baseline_fraction == 0.0


def test_an_initial_state_outside_the_invariant_set_is_reported_not_hidden(
    plant, controllers, invariant_set
):
    guard = SimplexGuard(plant, controllers[0], invariant_set)
    rng = np.random.default_rng(13)
    w = disturbance_sequence(plant, 200, rng, "uniform")
    outside = np.array([0.28, 0.45])
    assert not invariant_set.contains(outside)
    episode = simulate_guarded(
        plant, guard, controllers[1], 200, square_wave_reference(0.0, 100, 2), w, x0=outside
    )
    assert episode.certificate_lost[0]
    assert np.any(episode.certificate_lost)
    # The baseline pulls it back in, but that recovery is NOT guaranteed and the
    # record says the certificate was lost while it happened.
    assert not episode.certificate_lost[-1]


def test_a_one_step_episode_is_accountable(plant, controllers, invariant_set):
    guard = SimplexGuard(plant, controllers[0], invariant_set)
    rng = np.random.default_rng(14)
    w = disturbance_sequence(plant, 1, rng, "uniform")
    reference = square_wave_reference(0.1, 5, plant.n_states)
    guarded = simulate_guarded(plant, guard, controllers[1], 1, reference, w)
    unguarded = simulate_unguarded(plant, controllers[1], 1, reference, w)
    base_only = simulate_baseline(plant, controllers[0], 1, reference, w)
    report = account(guarded, unguarded, base_only, invariant_set)
    assert report.n_steps == 1
    assert report.n_switches == 0


def test_a_scalar_plant_runs_end_to_end():
    plant = Plant(
        A=np.array([[1.1]]),
        B=np.array([[1.0]]),
        disturbance=box([0.02]),
        state_constraints=box([1.0]),
        input_constraints=box([2.0]),
        dt=0.1,
    )
    baseline = BaselineController(gain=np.array([[0.6]]), input_set=box([2.0]))
    from simplexguard import PerformanceController

    performance = PerformanceController(gain=np.array([[2.5]]), input_set=box([2.0]))
    invariant = robust_invariant_set(plant, baseline).polytope
    guard = SimplexGuard(plant, baseline, invariant, verify=True)
    rng = np.random.default_rng(15)
    w = disturbance_sequence(plant, 400, rng, "vertex")
    reference = square_wave_reference(0.7, 40, 1)
    episode = simulate_guarded(plant, guard, performance, 400, reference, w)
    assert episode.constraint_violations().size == 0
    assert episode.invariant_exits(invariant).size == 0


def test_a_degenerate_disturbance_equal_to_the_constraint_set_empties(plant, controllers):
    huge = plant.with_disturbance(box([0.30, 0.50]))
    from simplexguard import EmptyInvariantSet

    with pytest.raises(EmptyInvariantSet):
        robust_invariant_set(huge, controllers[0])


def test_very_tight_constraints_empty_the_set(controllers):
    tiny = reference_plant(angle_limit_rad=1e-4, rate_limit_rad_s=1e-4)
    baseline, _ = reference_controllers(tiny)
    from simplexguard import EmptyInvariantSet

    with pytest.raises(EmptyInvariantSet):
        robust_invariant_set(tiny, baseline, max_iterations=40)


def test_an_enormous_input_limit_nearly_gives_the_whole_constraint_set():
    # With an effectively unlimited input and a high-authority baseline, S grows
    # to almost all of X but NOT to all of it: the corners of X where the rate is
    # at its limit still have to be eroded by one step of the disturbance, so the
    # measured area is 0.5986606 against 0.60, which is 99.777 % of X. The point
    # of the test is that "almost" is not "all", and the recursion says so.
    plant = reference_plant(accel_limit_rad_s2=500.0)
    baseline, _ = reference_controllers(plant, baseline_r=1e-4)
    result = robust_invariant_set(plant, baseline)
    assert result.converged
    area = result.polytope.area_2d()
    assert area == pytest.approx(0.5986606494, abs=1e-8)
    assert area < plant.state_constraints.area_2d()
    assert area / plant.state_constraints.area_2d() > 0.99
