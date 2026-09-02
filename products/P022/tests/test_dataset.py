"""Manoeuvre suites, rollout scoring and the oracle dataset."""

import numpy as np
import pytest

from cmgsteer._fast import FastStepper
from cmgsteer.arrays import pyramid_array
from cmgsteer.dataset import (
    generate_policy_dataset,
    manoeuvre_suite,
    rollout_score,
)
from cmgsteer.nullmotion import GradientNullMotion
from cmgsteer.simulate import run_steering
from cmgsteer.steering import sr_inverse_steer


def _small_suite(array, n=2):
    return manoeuvre_suite(array, n, seed=99, n_segments=2, segment_duration=3.0, dt=0.05)


class TestManoeuvreSuite:
    def test_is_deterministic_in_the_seed(self):
        array = pyramid_array()
        a = _small_suite(array)
        b = _small_suite(array)
        assert np.allclose(a.initial_deltas, b.initial_deltas)
        for pa, pb in zip(a.profiles, b.profiles):
            assert np.allclose(pa.torques, pb.torques)

    def test_different_seeds_differ(self):
        array = pyramid_array()
        a = manoeuvre_suite(array, 2, seed=1, n_segments=2, segment_duration=3.0, dt=0.05)
        b = manoeuvre_suite(array, 2, seed=2, n_segments=2, segment_duration=3.0, dt=0.05)
        assert not np.allclose(a.profiles[0].torques, b.profiles[0].torques)

    def test_sizes_and_iteration(self):
        array = pyramid_array()
        suite = _small_suite(array, n=3)
        assert len(suite) == 3
        assert suite.n_steps == 3 * 2 * 60
        pairs = list(suite)
        assert len(pairs) == 3
        assert pairs[0][1].shape == (4,)

    def test_each_segment_returns_the_momentum_to_the_start(self):
        array = pyramid_array()
        suite = _small_suite(array)
        assert np.max(np.abs(suite.profiles[0].momentum_change)) < 1e-10

    def test_peak_momentum_is_inside_the_requested_band(self):
        array = pyramid_array()
        suite = manoeuvre_suite(
            array, 6, seed=3, n_segments=1, segment_duration=4.0, dt=0.01,
            momentum_fraction=(0.35, 0.65),
        )
        cap = array.total_momentum_capacity
        peaks = np.array([p.peak_momentum for p in suite.profiles])
        assert np.all(peaks > 0.34 * cap)
        assert np.all(peaks < 0.66 * cap)

    def test_invalid_arguments_raise(self):
        array = pyramid_array()
        with pytest.raises(ValueError, match="n_manoeuvres must be >= 1"):
            manoeuvre_suite(array, 0, seed=1)
        with pytest.raises(ValueError, match="n_segments must be >= 1"):
            manoeuvre_suite(array, 1, seed=1, n_segments=0)
        with pytest.raises(ValueError, match="momentum_fraction"):
            manoeuvre_suite(array, 1, seed=1, momentum_fraction=(0.8, 0.2))


class TestFastStepperAgreement:
    def test_matches_the_public_path_without_null_motion(self):
        array = pyramid_array()
        suite = _small_suite(array)
        profile, start = suite.profiles[0], suite.initial_deltas[0]
        fast = rollout_score(array, start, profile.torques[:40], profile.dt, 0.0, fast=True)
        slow = rollout_score(array, start, profile.torques[:40], profile.dt, 0.0, fast=False)
        assert fast == pytest.approx(slow, rel=1e-11, abs=1e-15)

    @pytest.mark.parametrize("coefficient", [-1.0, -0.25, 0.4, 1.0])
    def test_matches_the_public_path_with_null_motion(self, coefficient):
        array = pyramid_array()
        suite = _small_suite(array)
        profile, start = suite.profiles[0], suite.initial_deltas[0]
        fast = rollout_score(
            array, start, profile.torques[:40], profile.dt, coefficient, fast=True
        )
        slow = rollout_score(
            array, start, profile.torques[:40], profile.dt, coefficient, fast=False
        )
        assert fast == pytest.approx(slow, rel=1e-10, abs=1e-15)

    def test_stepper_momentum_matches_the_array(self):
        array = pyramid_array()
        stepper = FastStepper(array, 0.01, 10.0)
        rng = np.random.default_rng(21)
        for _ in range(20):
            d = rng.uniform(-np.pi, np.pi, 4)
            assert np.allclose(stepper.momentum(d), array.momentum(d), atol=1e-14)

    def test_stepper_rates_match_the_public_sr_law(self):
        array = pyramid_array()
        stepper = FastStepper(array, 0.01, 10.0, max_gimbal_rate=2.0)
        rng = np.random.default_rng(22)
        for _ in range(20):
            d = rng.uniform(-np.pi, np.pi, 4)
            tau = rng.normal(size=3) * 0.3
            expected = sr_inverse_steer(
                array, d, tau, lam0=0.01, mu=10.0, max_gimbal_rate=2.0
            ).gimbal_rates
            assert np.allclose(stepper.step(d, tau, 0.0, 0.5), expected, atol=1e-12)

    def test_stepper_gradient_mode_matches_the_public_policy(self):
        array = pyramid_array()
        stepper = FastStepper(array, 0.01, 10.0)
        policy = GradientNullMotion(gain=1.0, max_rate=0.5)
        rng = np.random.default_rng(23)
        for _ in range(15):
            d = rng.uniform(-1.0, 1.0, 4)
            tau = rng.normal(size=3) * 0.3
            null = policy.rates(array, d, tau)
            expected = sr_inverse_steer(
                array, d, tau, lam0=0.01, mu=10.0, null_rates=null
            ).gimbal_rates
            got = stepper.step(d, tau, 0.0, 0.5, gradient_gain=1.0)
            assert np.allclose(got, expected, atol=1e-11)

    def test_stepper_is_a_no_op_on_null_motion_without_a_null_space(self):
        array = pyramid_array().with_locked([0])
        stepper = FastStepper(array, 0.01, 10.0)
        d = np.array([0.2, -0.3, 0.4, 0.1])
        tau = np.array([0.05, 0.02, -0.03])
        assert np.allclose(stepper.step(d, tau, 1.0, 0.5), stepper.step(d, tau, 0.0, 0.5))


class TestPolicyDataset:
    @pytest.fixture(scope="class")
    @classmethod
    def dataset(cls):
        return generate_policy_dataset(
            pyramid_array(),
            30,
            seed=7,
            horizon=12,
            n_candidates=7,
            stride=23,
            n_manoeuvres=4,
        )

    def test_shapes_and_ranges(self, dataset):
        assert dataset.n_samples == 30
        assert dataset.features.shape == (30, 20)
        assert dataset.candidate_scores.shape == (30, 7)
        assert np.all(dataset.coefficients >= -1.0)
        assert np.all(dataset.coefficients <= 1.0)
        assert np.all(dataset.candidate_scores >= 0.0)
        assert dataset.horizon == 12

    def test_candidate_grid_contains_zero(self, dataset):
        assert np.min(np.abs(dataset.candidates)) == 0.0

    def test_best_score_never_exceeds_the_zero_score(self, dataset):
        assert np.all(dataset.best_scores <= dataset.zero_scores + 1e-15)

    def test_labels_are_near_the_best_candidate(self, dataset):
        # the parabolic refinement must not move the label more than one grid
        # spacing away from the discrete argmin
        spacing = float(dataset.candidates[1] - dataset.candidates[0])
        best = dataset.candidates[np.argmin(dataset.candidate_scores, axis=1)]
        assert np.max(np.abs(dataset.coefficients - best)) <= spacing + 1e-12

    def test_is_deterministic(self, dataset):
        again = generate_policy_dataset(
            pyramid_array(), 30, seed=7, horizon=12, n_candidates=7, stride=23, n_manoeuvres=4
        )
        assert np.allclose(dataset.features, again.features)
        assert np.allclose(dataset.coefficients, again.coefficients)
        assert np.allclose(dataset.candidate_scores, again.candidate_scores)

    def test_gradient_scores_are_recorded(self, dataset):
        assert dataset.gradient_scores.shape == (30,)
        assert np.all(dataset.gradient_scores >= 0.0)

    def test_zero_score_reproduces_a_plain_run(self):
        # rolling out k = 0 from the start of a manoeuvre must match the
        # accumulated path momentum error of the equivalent run_steering call
        array = pyramid_array()
        suite = _small_suite(array)
        profile, start = suite.profiles[0], suite.initial_deltas[0]
        score = rollout_score(
            array, start, profile.torques, profile.dt, 0.0, max_gimbal_rate=2.0, fast=False
        )
        history = run_steering(array, start, profile, method="sr", max_gimbal_rate=2.0)
        assert score == pytest.approx(history.total_momentum_error_path, rel=1e-10)

    def test_invalid_arguments_raise(self):
        array = pyramid_array()
        with pytest.raises(ValueError, match="n_samples must be >= 1"):
            generate_policy_dataset(array, 0, seed=1)
        with pytest.raises(ValueError, match="n_candidates must be odd"):
            generate_policy_dataset(array, 5, seed=1, n_candidates=8)
        with pytest.raises(ValueError, match="horizon must be >= 1"):
            generate_policy_dataset(array, 5, seed=1, horizon=0)
        with pytest.raises(ValueError, match="stride must be >= 1"):
            generate_policy_dataset(array, 5, seed=1, stride=0)
