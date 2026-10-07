"""End-to-end integration: the full pipeline from plant to accounting to predictor.

One test that exercises every module in sequence, the way a user does, and
checks the relationships between the stages rather than the stages in isolation.
"""

from __future__ import annotations

import numpy as np

from simplexguard import (
    ExactLeadPredictor,
    SimplexGuard,
    account,
    bound_violation_sweep,
    build_dataset,
    disturbance_sequence,
    fit_switch_predictor,
    guard_condition_scores,
    reference_controllers,
    reference_plant,
    robust_invariant_set,
    score_binary,
    simulate_baseline,
    simulate_guarded,
    simulate_unguarded,
    square_wave_reference,
    verify_robust_invariance,
)


def test_full_pipeline_end_to_end():
    plant = reference_plant()
    baseline, performance = reference_controllers(plant)

    # 1. the certificate
    result = robust_invariant_set(plant, baseline)
    assert result.converged
    report = verify_robust_invariance(plant, baseline, result.polytope)
    assert report["subset_of_X"] and report["robustly_invariant"]

    # 2. the guard
    guard = SimplexGuard(plant, baseline, result.polytope, verify=True)

    # 3. three paired episodes
    rng = np.random.default_rng(5101)
    w = disturbance_sequence(plant, 1000, rng, "uniform")
    reference = square_wave_reference(0.18, 80, plant.n_states)
    guarded = simulate_guarded(plant, guard, performance, 1000, reference, w)
    unguarded = simulate_unguarded(plant, performance, 1000, reference, w)
    base_only = simulate_baseline(plant, baseline, 1000, reference, w)

    # 4. the accounting
    acc = account(guarded, unguarded, base_only, result.polytope)
    assert acc.guarded_violations == 0
    assert acc.unguarded_violations > 0
    assert acc.unguarded_cost < acc.guarded_cost < acc.baseline_cost
    assert 0.0 < acc.baseline_fraction < 1.0
    assert acc.n_switches > 0

    # 5. the bound-violation experiment
    sweep = bound_violation_sweep(
        plant,
        guard,
        performance,
        result.polytope,
        scales=(1.0, 6.0),
        n_episodes=2,
        n_steps=600,
        disturbance_mode="vertex",
    )
    assert sweep.points[0].constraint_violations == 0
    assert sweep.points[1].constraint_violations > 0

    # 6. the exact multi-step predictor and the learned one, on the same split
    data = build_dataset(plant, guard, performance, 8, 300, seed=5101, lead=4)
    train = data.select_episodes(np.arange(0, 5))
    calib = data.select_episodes(np.arange(5, 6))
    test = data.select_episodes(np.arange(6, 8))
    exact = ExactLeadPredictor(guard, performance, 4)
    exact_pred = exact.predict_many(test.states, test.references)
    exact_scores = score_binary("exact", test.labels, exact_pred)
    model = fit_switch_predictor(train, calib, "forest", n_estimators=40)
    prob = model.predict_proba(test.features)[:, 1]
    learned_scores = score_binary("forest", test.labels, prob >= 0.5, probability=prob)
    assert exact_scores.recall > 0.6
    assert np.isfinite(learned_scores.brier)

    # 7. and on the exact condition's own criterion the exact answer is perfect
    zero_lead = build_dataset(plant, guard, performance, 3, 300, seed=5101, lead=0)
    assert guard_condition_scores(zero_lead, guard, performance).f1 == 1.0


def test_guarded_states_stay_in_the_invariant_set_for_every_sampler():
    plant = reference_plant()
    baseline, performance = reference_controllers(plant)
    invariant = robust_invariant_set(plant, baseline).polytope
    guard = SimplexGuard(plant, baseline, invariant)
    reference = square_wave_reference(0.24, 60, plant.n_states)
    for mode in ("uniform", "vertex", "zero"):
        rng = np.random.default_rng(777)
        w = disturbance_sequence(plant, 800, rng, mode)
        episode = simulate_guarded(plant, guard, performance, 800, reference, w)
        assert episode.invariant_exits(invariant).size == 0
        assert episode.constraint_violations().size == 0


def test_a_tighter_disturbance_declaration_costs_less_conservatism():
    # Declaring a smaller disturbance gives a larger invariant set and a larger
    # eroded set, so the guard fires less and the conservatism cost falls. The
    # safety statement is correspondingly weaker, which is the trade this package
    # exists to price.
    from simplexguard import box

    baseline_fractions = []
    costs = []
    for accel in (0.06, 0.12, 0.24):
        plant = reference_plant(disturbance_accel_rad_s2=accel)
        baseline, performance = reference_controllers(plant)
        invariant = robust_invariant_set(plant, baseline).polytope
        guard = SimplexGuard(plant, baseline, invariant)
        rng = np.random.default_rng(31)
        # The same realised disturbance for all three, drawn from the smallest
        # declared box, so only the declaration changes.
        smallest = box(
            reference_plant(disturbance_accel_rad_s2=0.06).disturbance.half_widths
        )
        w = rng.uniform(-1.0, 1.0, size=(800, 2)) * smallest.half_widths
        reference = square_wave_reference(0.18, 80, plant.n_states)
        episode = simulate_guarded(plant, guard, performance, 800, reference, w)
        baseline_fractions.append(episode.baseline_fraction)
        costs.append(episode.cost())
        assert episode.constraint_violations().size == 0
    assert baseline_fractions[0] <= baseline_fractions[1] <= baseline_fractions[2]
    assert costs[0] <= costs[1] <= costs[2]
