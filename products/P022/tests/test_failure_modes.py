"""Failure modes required at validation Level 3.

Four families, all of them things a CMG steering implementation meets in
service rather than in a unit test:

1. **External (saturation) singularities** -- the array is at the boundary of
   its momentum envelope and cannot produce torque outward.
2. **Internal singularities** -- the array is inside its envelope but has lost
   a torque direction; some of these are escapable by null motion and some are
   not, and the package must say which.
3. **Gimbal-rate saturation** -- the law asks for more rate than the hardware
   has.
4. **Array degeneracy after a CMG failure** -- a locked gimbal leaves three
   free, so there is no null space at all and the remaining Jacobian can be
   singular on a two-dimensional set.
"""

import numpy as np
import pytest

from cmgsteer.arrays import pyramid_array, roof_array
from cmgsteer.nullmotion import GradientNullMotion, unit_null_vector
from cmgsteer.simulate import constant_profile, rest_to_rest_profile, run_steering
from cmgsteer.singularity import (
    classify_singularity,
    manipulability_gradient,
    null_space_basis,
    singular_configuration,
    singularity_measure,
)
from cmgsteer.steering import pseudo_inverse_steer, sr_inverse_steer

TAU = np.array([0.10, -0.05, 0.20])


class TestExternalSingularity:
    def test_pseudo_inverse_produces_a_huge_rate_approaching_saturation(self):
        array = pyramid_array()
        rates = []
        for offset in (1e-1, 1e-2, 1e-3, 1e-4):
            d = np.full(4, np.pi / 2 - offset)
            rates.append(np.max(np.abs(pseudo_inverse_steer(array, d, TAU).gimbal_rates)))
        assert all(b > a for a, b in zip(rates, rates[1:]))
        assert rates[-1] > 500.0

    def test_sr_inverse_stays_bounded_where_the_pseudo_inverse_does_not(self):
        array = pyramid_array()
        d = np.full(4, np.pi / 2 - 1e-6)
        pinv = pseudo_inverse_steer(array, d, TAU)
        sr = sr_inverse_steer(array, d, TAU, lam0=0.01, mu=10.0)
        assert np.max(np.abs(pinv.gimbal_rates)) > 1e4
        assert np.max(np.abs(sr.gimbal_rates)) < 10.0

    def test_the_lost_direction_is_the_saturation_direction(self):
        array = pyramid_array()
        d = np.full(4, np.pi / 2)
        info = classify_singularity(array, d)
        # commanded torque along +z cannot be produced at all
        result = sr_inverse_steer(array, d, np.array([0.0, 0.0, 0.1]), lam=1e-3)
        along = float(abs(result.torque_error @ info.direction))
        assert along == pytest.approx(0.1, rel=1e-6)

    def test_torque_in_the_remaining_plane_is_still_deliverable(self):
        array = pyramid_array()
        d = np.full(4, np.pi / 2)
        result = sr_inverse_steer(array, d, np.array([0.05, 0.02, 0.0]), lam=1e-8)
        assert result.torque_error_norm < 1e-6

    def test_leaving_an_external_singularity_costs_momentum(self):
        # The second-order form is definite, so every direction in null(A)
        # lowers h . u: the array can only come off a saturation singularity by
        # giving up momentum along the direction it was saturated in.  Note
        # that the manipulability gradient is NOT zero there -- a step along
        # null(A) does raise m -- so "elliptic" means the momentum cannot stay,
        # not that the measure cannot rise.
        array = pyramid_array()
        d = np.full(4, np.pi / 2)
        info = classify_singularity(array, d)
        assert info.passability == "elliptic"
        basis = null_space_basis(array.jacobian(d))
        base = float(array.momentum(d) @ info.direction)
        for j in range(basis.shape[1]):
            for eps in (-0.05, 0.05):
                moved = array.momentum(d + eps * basis[:, j]) @ info.direction
                assert moved < base - 1e-6
        grad = manipulability_gradient(array, d)
        assert np.max(np.abs(basis.T @ grad)) > 1.0


class TestInternalSingularity:
    def _internal(self):
        array = pyramid_array()
        d = np.array([1.0, 1.0, -1.0, -1.0]) * np.pi / 2
        return array, d

    def test_is_inside_the_momentum_envelope(self):
        array, d = self._internal()
        info = classify_singularity(array, d)
        assert info.singular
        assert info.kind == "internal"
        assert np.linalg.norm(info.momentum) < array.total_momentum_capacity - 1e-6

    def test_hyperbolic_internal_singularity_is_escapable(self):
        array = pyramid_array()
        found = None
        rng = np.random.default_rng(41)
        for _ in range(200):
            u = rng.normal(size=3)
            u /= np.linalg.norm(u)
            d = singular_configuration(array, u, np.array([1.0, 1.0, -1.0, -1.0]))
            info = classify_singularity(array, d)
            if info.kind == "internal" and info.passability == "hyperbolic":
                found = d
                break
        assert found is not None
        basis = null_space_basis(array.jacobian(found))
        # some null direction leaves the singular surface at second order
        improved = max(
            singularity_measure(array.jacobian(found + eps * basis[:, j]))
            for j in range(basis.shape[1])
            for eps in (-0.05, 0.05)
        )
        assert improved > 1e-4

    def test_sr_inverse_reports_the_error_it_cannot_avoid(self):
        array, d = self._internal()
        info = classify_singularity(array, d)
        tau = 0.1 * info.direction
        result = sr_inverse_steer(array, d, tau, lam=1e-4)
        assert result.torque_error_norm == pytest.approx(0.1, rel=1e-3)

    def test_null_vector_is_undefined_exactly_at_the_singularity(self):
        array, d = self._internal()
        with pytest.raises(ValueError, match="null space has dimension"):
            unit_null_vector(array, d)

    def test_gradient_policy_returns_zero_rather_than_raising(self):
        array, d = self._internal()
        rates = GradientNullMotion(gain=1.0, max_rate=0.5).rates(array, d, TAU)
        assert np.all(np.isfinite(rates))
        assert np.max(np.abs(array.jacobian(d) @ rates)) < 1e-12


class TestGimbalRateSaturation:
    def test_saturation_is_reported_not_hidden(self):
        array = pyramid_array()
        profile = rest_to_rest_profile([0.0, 0.0, 1.0], 3.0, 4.0, 0.02)
        history = run_steering(array, np.zeros(4), profile, max_gimbal_rate=0.2)
        assert history.n_rate_limited > 0
        assert history.max_torque_error > 1e-3
        assert history.peak_gimbal_rate <= 0.2 + 1e-12

    def test_saturation_costs_torque_and_path_accuracy(self):
        # The net (signed) momentum error is the wrong metric for a saturated
        # rest-to-rest run: the gimbals barely move, so the momentum stays near
        # its starting value and the net error can be *smaller* than an
        # unsaturated run's integration error.  The path-length error and the
        # instantaneous torque error are the honest ones.
        array = pyramid_array()
        profile = rest_to_rest_profile([0.3, 0.5, 0.8], 3.0, 4.0, 0.02)
        start = np.array([0.2, -0.3, 0.15, 0.4])
        free = run_steering(array, start, profile)
        capped = run_steering(array, start, profile, max_gimbal_rate=0.25)
        assert capped.total_momentum_error_path > 5.0 * free.total_momentum_error_path
        assert capped.rms_torque_error > 5.0 * free.rms_torque_error

    def test_scale_mode_keeps_the_torque_direction_while_clip_does_not(self):
        array = pyramid_array()
        profile = rest_to_rest_profile([0.3, 0.5, 0.8], 3.0, 4.0, 0.02)
        start = np.array([0.2, -0.3, 0.15, 0.4])
        clipped = run_steering(array, start, profile, max_gimbal_rate=0.25)
        scaled = run_steering(
            array, start, profile, max_gimbal_rate=0.25, saturation_mode="scale"
        )
        mask = scaled.rate_limited
        cmd = scaled.commanded_torque[mask]
        got = scaled.achieved_torque[mask]
        cos = np.sum(cmd * got, axis=1) / (
            np.linalg.norm(cmd, axis=1) * np.linalg.norm(got, axis=1)
        )
        assert np.min(cos) > 1.0 - 1e-9
        cmd_c = clipped.commanded_torque[clipped.rate_limited]
        got_c = clipped.achieved_torque[clipped.rate_limited]
        cos_c = np.sum(cmd_c * got_c, axis=1) / (
            np.linalg.norm(cmd_c, axis=1) * np.linalg.norm(got_c, axis=1)
        )
        assert np.min(cos_c) < 1.0 - 1e-6

    def test_a_command_beyond_the_rate_envelope_is_not_silently_met(self):
        array = pyramid_array()
        profile = constant_profile([0.0, 0.0, 5.0], 1.0, 0.02)
        history = run_steering(array, np.zeros(4), profile, max_gimbal_rate=0.1)
        assert history.rms_torque_error > 1.0


class TestArrayDegeneracyAfterFailure:
    def test_one_failed_cmg_removes_the_null_space(self):
        array = pyramid_array().with_locked([0])
        d = np.array([0.2, -0.4, 0.6, 0.1])
        assert null_space_basis(array.jacobian(d)).shape[1] == 0
        with pytest.raises(ValueError, match="null space has dimension 0"):
            unit_null_vector(array, d)

    def test_three_cmgs_can_still_meet_a_command_when_non_singular(self):
        array = pyramid_array().with_locked([3])
        d = np.array([0.2, -0.4, 0.6, 0.0])
        result = pseudo_inverse_steer(array, d, TAU)
        assert result.torque_error_norm < 1e-12
        assert result.gimbal_rates.shape == (3,)

    def test_the_degraded_array_has_a_smaller_measure(self):
        array = pyramid_array()
        d = np.array([0.2, -0.4, 0.6, 0.1])
        full = singularity_measure(array.jacobian(d))
        degraded = singularity_measure(array.with_locked([0]).jacobian(d))
        assert degraded < full

    def test_degraded_singular_configurations_exist_and_are_detected(self):
        array = pyramid_array().with_locked([3])
        rng = np.random.default_rng(42)
        worst = np.inf
        for _ in range(400):
            d = rng.uniform(-np.pi, np.pi, 4)
            worst = min(worst, singularity_measure(array.jacobian(d)))
        assert worst < 1e-2

    def test_two_failed_cmgs_are_under_actuated_and_say_so(self):
        array = pyramid_array().with_locked([0, 1])
        with pytest.raises(ValueError, match="at least 3 free gimbals"):
            singularity_measure(array.jacobian(np.zeros(4)))
        with pytest.raises(ValueError, match="at least 3 free gimbals"):
            pseudo_inverse_steer(array, np.zeros(4), TAU)
        with pytest.raises(ValueError, match="at least 3 free gimbals"):
            sr_inverse_steer(array, np.zeros(4), TAU)

    def test_failure_mid_manoeuvre_leaves_a_larger_error(self):
        healthy = pyramid_array()
        failed = healthy.with_locked([1])
        profile = rest_to_rest_profile([0.3, 0.2, 0.93], 1.5, 8.0, 0.02)
        start = np.array([0.1, 0.2, -0.1, 0.05])
        good = run_steering(healthy, start, profile, max_gimbal_rate=2.0)
        bad = run_steering(failed, start, profile, max_gimbal_rate=2.0)
        assert bad.min_measure < good.min_measure
        assert bad.accumulated_momentum_error > good.accumulated_momentum_error

    def test_roof_array_pair_failure_leaves_a_planar_pair(self):
        # locking one member of a roof pair leaves the surviving member's
        # momentum confined to a plane, which is the geometric reason a roof
        # array degrades faster than a pyramid
        array = roof_array().with_locked([0])
        d = np.array([0.4, -0.4, 0.4, -0.4])
        assert array.jacobian(d).shape == (3, 3)
        assert singularity_measure(array.jacobian(d)) < singularity_measure(
            roof_array().jacobian(d)
        )
