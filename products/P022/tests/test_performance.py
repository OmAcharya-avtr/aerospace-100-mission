"""Performance benchmark.

Thresholds are deliberately loose -- roughly ten times the measured value on
the two-core reference machine -- so that the suite fails on an algorithmic
regression (an accidental O(n^2), a dropped vectorisation) rather than on a
slower runner.  ``validation/validate_performance.py`` reports the actual
numbers.
"""

import time

import numpy as np
import pytest

from cmgsteer.arrays import pyramid_array
from cmgsteer.dataset import manoeuvre_suite, rollout_score
from cmgsteer.simulate import rest_to_rest_profile, run_steering
from cmgsteer.singularity import momentum_envelope, singularity_measure
from cmgsteer.steering import gsr_inverse_steer, pseudo_inverse_steer, sr_inverse_steer

TAU = np.array([0.10, -0.05, 0.20])
D = np.array([0.30, -0.50, 0.80, 0.20])


def _time(fn, repeats):
    fn()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - start) / repeats


class TestSteeringStepCost:
    @pytest.mark.parametrize(
        ("law", "budget_us"),
        [(pseudo_inverse_steer, 2000.0), (sr_inverse_steer, 2000.0), (gsr_inverse_steer, 2000.0)],
    )
    def test_single_step_is_under_budget(self, law, budget_us):
        array = pyramid_array()
        per_call = _time(lambda: law(array, D, TAU), 400)
        assert per_call * 1e6 < budget_us

    def test_measure_evaluation_is_cheap(self):
        array = pyramid_array()
        jac = array.jacobian(D)
        per_call = _time(lambda: singularity_measure(jac), 2000)
        assert per_call * 1e6 < 500.0


class TestRunCost:
    def test_a_thousand_step_run_is_under_five_seconds(self):
        array = pyramid_array()
        profile = rest_to_rest_profile([0.2, 0.3, 0.93], 1.5, 20.0, 0.02)
        assert profile.n_steps == 1000
        start = time.perf_counter()
        run_steering(array, np.zeros(4), profile, method="sr", max_gimbal_rate=2.0)
        assert time.perf_counter() - start < 5.0

    def test_cost_is_linear_in_the_number_of_steps(self):
        array = pyramid_array()
        times = []
        for n_steps in (400, 1600):
            profile = rest_to_rest_profile([0.2, 0.3, 0.93], 1.5, n_steps * 0.02, 0.02)
            start = time.perf_counter()
            run_steering(array, np.zeros(4), profile, method="sr")
            times.append(time.perf_counter() - start)
        # a quadratic implementation would give a ratio near 16
        assert times[1] / times[0] < 8.0


class TestGenerationCost:
    def test_fast_rollout_beats_the_public_path(self):
        array = pyramid_array()
        suite = manoeuvre_suite(array, 1, seed=5, n_segments=1, segment_duration=2.0, dt=0.05)
        profile, start = suite.profiles[0], suite.initial_deltas[0]
        window = profile.torques[:40]
        fast = _time(
            lambda: rollout_score(array, start, window, profile.dt, 0.5, fast=True), 5
        )
        slow = _time(
            lambda: rollout_score(array, start, window, profile.dt, 0.5, fast=False), 5
        )
        assert fast < slow

    def test_envelope_mapping_is_under_budget(self):
        array = pyramid_array()
        start = time.perf_counter()
        momenta, _ = momentum_envelope(array, n_points=2000)
        assert momenta.shape[0] > 1900
        assert time.perf_counter() - start < 10.0
