"""End-to-end: geometry to steering to a trained policy, in one small pipeline."""

import numpy as np
import pytest

from cmgsteer import (
    GradientNullMotion,
    LearnedNullMotion,
    NoNullMotion,
    classify_singularity,
    generate_policy_dataset,
    manoeuvre_suite,
    momentum_envelope,
    pyramid_array,
    roof_array,
    run_steering,
    singular_configuration,
    singularity_measure,
)


@pytest.fixture(scope="module")
def trained():
    array = pyramid_array()
    data = generate_policy_dataset(
        array, 60, seed=606, horizon=12, n_candidates=7, stride=23, n_manoeuvres=6
    )
    policy = LearnedNullMotion(
        max_null_rate=0.5, n_estimators=3, hidden_layer_sizes=(32, 16), max_iter=120
    )
    policy.fit(data.features, data.coefficients)
    return array, data, policy


class TestFullPipeline:
    def test_policy_runs_inside_a_manoeuvre(self, trained):
        array, _, policy = trained
        suite = manoeuvre_suite(
            array, 1, seed=77, n_segments=2, segment_duration=4.0, dt=0.02
        )
        history = run_steering(
            array,
            suite.initial_deltas[0],
            suite.profiles[0],
            method="sr",
            null_policy=policy,
            max_gimbal_rate=2.0,
        )
        assert history.policy == "learned"
        assert np.all(np.isfinite(history.deltas))
        assert history.min_measure > 0.0

    def test_learned_null_motion_never_changes_the_instantaneous_torque(self, trained):
        array, _, policy = trained
        suite = manoeuvre_suite(
            array, 1, seed=78, n_segments=1, segment_duration=4.0, dt=0.02
        )
        plain = run_steering(
            array, suite.initial_deltas[0], suite.profiles[0], method="pinv"
        )
        learned = run_steering(
            array, suite.initial_deltas[0], suite.profiles[0], method="pinv", null_policy=policy
        )
        assert plain.max_torque_error < 1e-12
        assert learned.max_torque_error < 1e-12

    def test_all_three_policies_run_on_the_same_suite(self, trained):
        array, _, policy = trained
        suite = manoeuvre_suite(
            array, 2, seed=79, n_segments=2, segment_duration=4.0, dt=0.02
        )
        results = {}
        for name, pol in (
            ("none", NoNullMotion()),
            ("gradient", GradientNullMotion(gain=1.0, max_rate=0.5)),
            ("learned", policy),
        ):
            total = 0.0
            for profile, start in suite:
                history = run_steering(
                    array, start, profile, method="sr", null_policy=pol, max_gimbal_rate=2.0
                )
                total += history.total_momentum_error_path
            results[name] = total
        assert set(results) == {"none", "gradient", "learned"}
        assert all(np.isfinite(v) and v > 0.0 for v in results.values())

    def test_a_singular_configuration_survives_the_whole_stack(self):
        array = pyramid_array()
        direction = np.array([0.3, -0.2, 0.9])
        direction /= np.linalg.norm(direction)
        d = singular_configuration(array, direction, np.array([1.0, -1.0, 1.0, -1.0]))
        info = classify_singularity(array, d)
        assert info.singular
        assert singularity_measure(array.jacobian(d)) < 1e-13
        assert abs(float(info.direction @ direction)) == pytest.approx(1.0, abs=1e-8)

    def test_envelope_contains_every_reachable_momentum(self):
        array = pyramid_array()
        momenta, _ = momentum_envelope(array, n_points=600)
        max_radius = float(np.linalg.norm(momenta, axis=1).max())
        rng = np.random.default_rng(808)
        sampled = np.array([array.momentum(rng.uniform(-np.pi, np.pi, 4)) for _ in range(500)])
        sampled_max = float(np.linalg.norm(sampled, axis=1).max())
        assert sampled_max <= array.total_momentum_capacity
        assert sampled_max <= max_radius + 1e-9

    def test_roof_and_pyramid_share_the_whole_api(self):
        for array in (pyramid_array(), roof_array()):
            d = np.array([0.4, -0.4, 0.4, -0.4])
            suite = manoeuvre_suite(
                array, 1, seed=81, n_segments=1, segment_duration=3.0, dt=0.05
            )
            history = run_steering(array, d, suite.profiles[0], method="sr", max_gimbal_rate=2.0)
            assert np.all(np.isfinite(history.momentum))
            assert history.measure.shape == (suite.profiles[0].n_steps + 1,)
