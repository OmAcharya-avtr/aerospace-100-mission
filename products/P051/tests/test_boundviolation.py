"""The deliberate bound-violation experiment."""

from __future__ import annotations

import numpy as np
import pytest

from simplexguard import bound_violation_sweep


def test_scale_one_has_no_violations_and_no_exits(plant, controllers, guard, invariant_set):
    sweep = bound_violation_sweep(
        plant,
        guard,
        controllers[1],
        invariant_set,
        scales=(1.0,),
        n_episodes=3,
        n_steps=500,
        disturbance_mode="vertex",
    )
    point = sweep.points[0]
    assert point.constraint_violations == 0
    assert point.invariant_exits == 0
    assert point.fraction_steps_outside_declared_W == 0.0
    assert point.worst_constraint_residual < 0.0
    assert not np.isfinite(sweep.first_scale_with_constraint_violation)
    assert not np.isfinite(sweep.first_scale_with_invariant_exit)


def test_violations_appear_and_grow_beyond_the_declared_bound(
    plant, controllers, guard, invariant_set
):
    sweep = bound_violation_sweep(
        plant,
        guard,
        controllers[1],
        invariant_set,
        scales=(1.0, 2.0, 6.0),
        n_episodes=3,
        n_steps=600,
        disturbance_mode="vertex",
    )
    counts = [p.constraint_violations for p in sweep.points]
    exits = [p.invariant_exits for p in sweep.points]
    assert counts[0] == 0
    assert counts[1] > 0
    assert counts[2] > counts[1]
    assert exits[2] > exits[1] > exits[0]
    assert np.isfinite(sweep.first_scale_with_constraint_violation)


def test_the_certificate_is_lost_at_least_as_often_as_a_constraint_is_broken(
    plant, controllers, guard, invariant_set
):
    # S is a subset of X, so every state outside X is also outside S. The exit
    # count must therefore be at least the violation count at every scale.
    sweep = bound_violation_sweep(
        plant,
        guard,
        controllers[1],
        invariant_set,
        scales=(1.0, 1.5, 3.0, 8.0),
        n_episodes=3,
        n_steps=600,
        disturbance_mode="vertex",
    )
    for point in sweep.points:
        assert point.invariant_exits >= point.constraint_violations
        assert point.worst_invariant_residual >= point.worst_constraint_residual - 1e-12


def test_the_vertex_sampler_breaks_the_guarantee_sooner_than_uniform(
    plant, controllers, guard, invariant_set
):
    scales = (1.0, 1.25, 1.5, 2.0)
    results = {}
    for mode in ("uniform", "vertex"):
        sweep = bound_violation_sweep(
            plant,
            guard,
            controllers[1],
            invariant_set,
            scales=scales,
            n_episodes=4,
            n_steps=1000,
            disturbance_mode=mode,
        )
        results[mode] = sweep
    # The worst-case-realising sampler must not be more forgiving.
    assert (
        results["vertex"].first_scale_with_constraint_violation
        <= results["uniform"].first_scale_with_constraint_violation
    )


def test_fraction_outside_the_declared_set_rises_with_the_scale(
    plant, controllers, guard, invariant_set
):
    sweep = bound_violation_sweep(
        plant,
        guard,
        controllers[1],
        invariant_set,
        scales=(1.0, 1.25, 2.0),
        n_episodes=2,
        n_steps=400,
    )
    fracs = [p.fraction_steps_outside_declared_W for p in sweep.points]
    assert fracs[0] == 0.0
    assert fracs[1] < fracs[2]


def test_describe_reports_both_thresholds(plant, controllers, guard, invariant_set):
    sweep = bound_violation_sweep(
        plant,
        guard,
        controllers[1],
        invariant_set,
        scales=(1.0, 4.0),
        n_episodes=2,
        n_steps=400,
    )
    text = sweep.describe()
    assert "invariance certificate was lost" in text
    assert "declared state constraint was broken" in text
    assert "rho=" in text


def test_sweep_validation(plant, controllers, guard, invariant_set):
    with pytest.raises(ValueError, match="scales must be non-empty"):
        bound_violation_sweep(plant, guard, controllers[1], invariant_set, scales=())
    with pytest.raises(ValueError, match="at most 1.0"):
        bound_violation_sweep(
            plant, guard, controllers[1], invariant_set, scales=(2.0, 3.0)
        )
    with pytest.raises(ValueError, match="finite and non-negative"):
        bound_violation_sweep(
            plant, guard, controllers[1], invariant_set, scales=(1.0, -1.0)
        )
    with pytest.raises(ValueError, match="n_episodes"):
        bound_violation_sweep(
            plant, guard, controllers[1], invariant_set, scales=(1.0,), n_episodes=0
        )
