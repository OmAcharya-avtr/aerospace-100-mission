"""Regression and benchmark tests: numbers that must not drift silently.

Every value pinned here was measured in this repository by running the code, and
each is the number a README or VALIDATION.md claim rests on. A change in any of
them means the behaviour changed, and the change has to be looked at rather than
re-baselined by reflex.

Tolerances are deliberately tight where the quantity is deterministic (set
geometry, counts on a fixed seed) and loose only where a wall-clock timing is
involved, which is never pinned as a pass criterion -- only its ordering is.
"""

from __future__ import annotations

import numpy as np
import pytest

from simplexguard import (
    SimplexGuard,
    account,
    disturbance_sequence,
    guard_condition_scores,
    simulate_baseline,
    simulate_guarded,
    simulate_unguarded,
    square_wave_reference,
)
from simplexguard.predictor import build_dataset


def test_invariant_set_geometry_is_pinned(invariant_result):
    assert invariant_result.iterations == 24
    assert invariant_result.n_halfspaces == 50
    assert invariant_result.polytope.area_2d() == pytest.approx(0.513045535, abs=1e-7)
    assert invariant_result.polytope.chebyshev_radius() == pytest.approx(
        0.298406159, abs=1e-7
    )
    # Omega_0 equals the declared constraint box here: the baseline input row is
    # redundant, which is why its area is exactly 4 * 0.30 * 0.50 = 0.60.
    assert invariant_result.initial_set is not None
    assert invariant_result.initial_set.area_2d() == pytest.approx(0.60, abs=1e-9)


def test_invariant_set_covers_a_pinned_fraction_of_the_constraint_set(
    invariant_result, plant
):
    fraction = invariant_result.polytope.area_2d() / plant.state_constraints.area_2d()
    assert fraction == pytest.approx(0.85507589, abs=1e-6)


def test_baseline_gain_is_pinned(controllers):
    assert controllers[0].gain == pytest.approx(
        np.array([[0.95762716, 1.68294507]]), abs=1e-7
    )
    assert controllers[1].gain == pytest.approx(
        np.array([[25.54258981, 7.59008686]]), abs=1e-7
    )


def test_eroded_offsets_are_pinned(guard):
    offsets = guard.disturbance_offsets
    # The facet normals of S are unit-length after reduction, and the extreme
    # disturbance offsets are h_W for the two axis-aligned facets:
    #   h_W((1,0)) = 1.5e-4 and h_W((0,1)) = 6.0e-3.
    assert offsets.min() == pytest.approx(1.5e-4, abs=1e-12)
    assert offsets.max() == pytest.approx(6.0e-3, abs=1e-12)


@pytest.fixture(scope="module")
def pinned_accounting(plant, controllers, invariant_set):
    guard = SimplexGuard(plant, controllers[0], invariant_set)
    rng = np.random.default_rng(51)
    w = disturbance_sequence(plant, 2000, rng, "uniform")
    reference = square_wave_reference(0.18, 80, plant.n_states)
    guarded = simulate_guarded(plant, guard, controllers[1], 2000, reference, w)
    unguarded = simulate_unguarded(plant, controllers[1], 2000, reference, w)
    base_only = simulate_baseline(plant, controllers[0], 2000, reference, w)
    return account(guarded, unguarded, base_only, invariant_set)


def test_reference_scenario_accounting_is_pinned(pinned_accounting):
    report = pinned_accounting
    assert report.n_steps == 2000
    assert report.n_switches == 140
    assert report.baseline_fraction == pytest.approx(0.084, abs=1e-12)
    assert report.baseline_dwell.n_intervals == 70
    assert report.baseline_dwell.mean == pytest.approx(2.4, abs=1e-9)
    assert report.baseline_dwell.maximum == 5
    assert report.performance_dwell.median == pytest.approx(1.0, abs=1e-12)
    assert report.guarded_cost == pytest.approx(258.634365, abs=1e-5)
    assert report.unguarded_cost == pytest.approx(215.826971, abs=1e-5)
    assert report.baseline_cost == pytest.approx(650.257772, abs=1e-5)
    assert report.conservatism_cost_ratio == pytest.approx(1.198341, abs=1e-5)
    assert report.performance_gap_recovered == pytest.approx(0.901463, abs=1e-5)
    assert report.guarded_violations == 0
    assert report.unguarded_violations == 168
    assert report.unguarded_invariant_exits == 189
    assert report.unguarded_worst_residual == pytest.approx(0.3197937271, abs=1e-8)


def test_the_guard_chatters_on_the_reference_scenario(pinned_accounting):
    # Reported as a limitation, not hidden: most performance intervals last one
    # step, so the architecture chatters at the boundary of the eroded set.
    report = pinned_accounting
    assert report.performance_dwell.fraction_of_length_one > 0.6
    assert report.baseline_dwell.fraction_of_length_one > 0.3


def test_switch_rate_falls_with_a_minimum_baseline_dwell(plant, controllers, invariant_set):
    # Pinned numbers for the hysteresis trade. Holding the baseline longer is
    # safe, reduces the switch rate, and costs tracking performance.
    rng_seed, steps = 51, 2000
    reference = square_wave_reference(0.18, 80, plant.n_states)
    observed = []
    for dwell in (1, 4, 10):
        guard = SimplexGuard(plant, controllers[0], invariant_set, min_baseline_dwell=dwell)
        rng = np.random.default_rng(rng_seed)
        w = disturbance_sequence(plant, steps, rng, "uniform")
        guarded = simulate_guarded(plant, guard, controllers[1], steps, reference, w)
        unguarded = simulate_unguarded(plant, controllers[1], steps, reference, w)
        base_only = simulate_baseline(plant, controllers[0], steps, reference, w)
        report = account(guarded, unguarded, base_only, invariant_set)
        observed.append(
            (report.n_switches, report.baseline_fraction, report.conservatism_cost_ratio)
        )
        assert report.guarded_violations == 0
    switches = [o[0] for o in observed]
    fractions = [o[1] for o in observed]
    ratios = [o[2] for o in observed]
    assert switches == [140, 92, 48]
    assert fractions[0] < fractions[1] < fractions[2]
    assert ratios[0] < ratios[1] < ratios[2]


def test_dataset_base_rates_are_pinned(plant, controllers, invariant_set):
    guard = SimplexGuard(plant, controllers[0], invariant_set)
    lead0 = build_dataset(plant, guard, controllers[1], 8, 300, seed=5101, lead=0)
    lead5 = build_dataset(plant, guard, controllers[1], 8, 300, seed=5101, lead=5)
    assert lead0.base_rate == pytest.approx(0.0541666666667, abs=1e-9)
    assert lead5.base_rate == pytest.approx(0.1088983050847, abs=1e-9)
    assert lead0.saturation_rate == pytest.approx(0.0679166666667, abs=1e-9)


def test_the_exact_condition_stays_perfect_on_its_own_criterion(
    plant, controllers, invariant_set
):
    # The benchmark baseline. If this ever stops being exactly 1.0, the
    # switching condition and the label generator have drifted apart.
    guard = SimplexGuard(plant, controllers[0], invariant_set)
    data = build_dataset(plant, guard, controllers[1], 6, 400, seed=5101, lead=0)
    scores = guard_condition_scores(data, guard, controllers[1])
    assert scores.precision == 1.0
    assert scores.recall == 1.0
    assert scores.false_positive == 0
    assert scores.false_negative == 0


def test_exact_condition_is_cheaper_than_the_learned_model(
    plant, controllers, invariant_set
):
    # Ordering only, never an absolute timing: the container is contended and a
    # pinned microsecond count would be a flaky test, not a benchmark.
    from simplexguard import fit_switch_predictor, measure_decision_cost

    guard = SimplexGuard(plant, controllers[0], invariant_set)
    data = build_dataset(plant, guard, controllers[1], 6, 300, seed=5101, lead=0)
    model = fit_switch_predictor(
        data.select_episodes(np.arange(0, 4)),
        data.select_episodes(np.arange(4, 5)),
        "forest",
        n_estimators=40,
    )
    row = data.features[0:1]
    performance = controllers[1]
    x0, r0 = data.states[0], data.references[0]
    exact_us = measure_decision_cost(
        lambda: guard.condition_margin(x0, performance(x0, r0))[0] < 0.0, n_calls=500
    )
    learned_us = measure_decision_cost(lambda: model.predict_proba(row), n_calls=60)
    assert learned_us > 10.0 * exact_us
