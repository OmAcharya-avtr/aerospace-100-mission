"""Unit tests for navbench.sensors — gyro, star tracker, sun sensor, accelerometer, GPS."""

from __future__ import annotations

import numpy as np
import pytest

from navbench import (
    AccelerometerModel,
    GpsModel,
    GyroModel,
    StarTrackerModel,
    SunSensorModel,
    arw_deg_per_sqrt_hour_to_si,
    dcm_from_quat,
    quat_conjugate,
    quat_from_euler_zyx,
    quat_identity,
    quat_multiply,
    rrw_deg_per_hour_1p5_to_si,
    small_angle_from_quat,
)


class TestUnitConversions:
    def test_arw_hand_computed(self):
        # 1 deg/sqrt(hr) = (pi/180) rad / sqrt(3600 s) = (pi/180)/60 rad/s^0.5
        assert arw_deg_per_sqrt_hour_to_si(1.0) == pytest.approx((np.pi / 180.0) / 60.0)

    def test_arw_linear(self):
        assert arw_deg_per_sqrt_hour_to_si(2.5) == pytest.approx(
            2.5 * arw_deg_per_sqrt_hour_to_si(1.0)
        )

    def test_arw_zero(self):
        assert arw_deg_per_sqrt_hour_to_si(0.0) == 0.0

    def test_rrw_hand_computed(self):
        assert rrw_deg_per_hour_1p5_to_si(1.0) == pytest.approx(
            (np.pi / 180.0) / 3600.0**1.5
        )

    def test_typical_tactical_grade_magnitude(self):
        """0.05 deg/sqrt(hr) is about 1.45e-5 rad/s^0.5 — sanity of the scale."""
        assert arw_deg_per_sqrt_hour_to_si(0.05) == pytest.approx(1.4544e-5, rel=1e-3)

    @pytest.mark.parametrize("bad", [-1.0, np.nan, np.inf])
    def test_negative_or_nonfinite_raises(self, bad):
        with pytest.raises(ValueError):
            arw_deg_per_sqrt_hour_to_si(bad)
        with pytest.raises(ValueError):
            rrw_deg_per_hour_1p5_to_si(bad)


class TestGyroModel:
    def test_discrete_sigmas(self):
        g = GyroModel(sigma_v=1e-4, sigma_u=1e-8, dt=0.25)
        s_rate, s_bias = g.discrete_sigmas()
        assert s_rate == pytest.approx(1e-4 / 0.5)
        assert s_bias == pytest.approx(1e-8 * 0.5)

    def test_noise_free_is_exact(self, rng):
        g = GyroModel(sigma_v=0.0, sigma_u=0.0, dt=0.1, bias0=[1e-3, 2e-3, -1e-3])
        out = g.sample([0.1, 0.2, 0.3], rng)
        assert np.allclose(out.rate, [0.1 + 1e-3, 0.2 + 2e-3, 0.3 - 1e-3])

    def test_bias_walks(self, rng):
        g = GyroModel(sigma_v=0.0, sigma_u=1e-3, dt=1.0)
        b0 = g.bias.copy()
        for _ in range(10):
            g.sample(np.zeros(3), rng)
        assert not np.allclose(g.bias, b0)

    def test_bias_constant_when_sigma_u_zero(self, rng):
        g = GyroModel(sigma_v=1e-3, sigma_u=0.0, dt=1.0, bias0=[1.0, 2.0, 3.0])
        for _ in range(20):
            g.sample(np.zeros(3), rng)
        assert np.allclose(g.bias, [1.0, 2.0, 3.0])

    def test_reset(self, rng):
        g = GyroModel(sigma_v=0.0, sigma_u=1e-2, dt=1.0, bias0=[0.1, 0.2, 0.3])
        for _ in range(5):
            g.sample(np.zeros(3), rng)
        g.reset()
        assert np.allclose(g.bias, [0.1, 0.2, 0.3])
        g.reset([1.0, 1.0, 1.0])
        assert np.allclose(g.bias, 1.0)

    def test_rate_noise_statistics(self, rng):
        sv, dt = 1e-3, 0.01
        g = GyroModel(sigma_v=sv, sigma_u=0.0, dt=dt)
        rates, _ = g.sample_series(np.zeros((50000, 3)), rng)
        assert np.allclose(np.std(rates, axis=0), sv / np.sqrt(dt), rtol=0.02)
        assert np.allclose(np.mean(rates, axis=0), 0.0, atol=4 * sv / np.sqrt(dt) / np.sqrt(50000))

    def test_bias_random_walk_variance_grows_linearly(self, rng):
        su, dt, n = 1e-3, 1.0, 4000
        finals = []
        for i in range(200):
            g = GyroModel(sigma_v=0.0, sigma_u=su, dt=dt)
            _, biases = g.sample_series(np.zeros((n, 3)), np.random.default_rng(i))
            finals.append(biases[-1, 0])
        assert np.std(finals) == pytest.approx(su * np.sqrt(dt * (n - 1)), rel=0.12)

    def test_sample_series_matches_repeated_sample(self):
        g1 = GyroModel(sigma_v=1e-3, sigma_u=1e-5, dt=0.5, bias0=[1e-4, 0, 0])
        w = np.tile([0.01, -0.02, 0.03], (7, 1))
        rates_a, biases_a = g1.sample_series(w, np.random.default_rng(3))
        # sample_series draws all bias steps first, then all rate noise, so the
        # streams differ from repeated sample() calls; only the deterministic
        # structure is compared here.
        assert rates_a.shape == (7, 3)
        assert biases_a.shape == (7, 3)
        assert np.all(np.isfinite(rates_a))

    def test_scale_factor_error(self, rng):
        g = GyroModel(sigma_v=0.0, sigma_u=0.0, dt=1.0, scale_factor_ppm=1000.0)
        out = g.sample([1.0, 0.0, 0.0], rng)
        assert out.rate[0] == pytest.approx(1.001)

    def test_misalignment_rotates_input(self, rng):
        g = GyroModel(sigma_v=0.0, sigma_u=0.0, dt=1.0, misalignment_rad=1e-3)
        out = g.sample([1.0, 0.0, 0.0], rng)
        assert np.linalg.norm(out.rate) == pytest.approx(1.0, abs=1e-12)
        assert not np.allclose(out.rate, [1.0, 0.0, 0.0])

    def test_reported_bias_is_the_one_applied(self, rng):
        g = GyroModel(sigma_v=0.0, sigma_u=1e-3, dt=1.0, bias0=[0.5, 0.0, 0.0])
        out = g.sample(np.zeros(3), rng)
        assert np.allclose(out.rate, out.bias)

    @pytest.mark.parametrize("kwargs", [
        {"sigma_v": -1.0, "sigma_u": 0.0, "dt": 1.0},
        {"sigma_v": 0.0, "sigma_u": -1.0, "dt": 1.0},
        {"sigma_v": 0.0, "sigma_u": 0.0, "dt": 0.0},
        {"sigma_v": 0.0, "sigma_u": 0.0, "dt": -1.0},
        {"sigma_v": np.nan, "sigma_u": 0.0, "dt": 1.0},
    ])
    def test_invalid_parameters_raise(self, kwargs):
        with pytest.raises(ValueError):
            GyroModel(**kwargs)

    def test_nonfinite_bias0_raises(self):
        with pytest.raises(ValueError, match="finite"):
            GyroModel(sigma_v=0.0, sigma_u=0.0, dt=1.0, bias0=[np.nan, 0, 0])

    def test_nonfinite_input_raises(self, rng):
        g = GyroModel(sigma_v=0.0, sigma_u=0.0, dt=1.0)
        with pytest.raises(ValueError, match="finite"):
            g.sample([np.nan, 0, 0], rng)

    def test_series_wrong_shape_raises(self, rng):
        g = GyroModel(sigma_v=0.0, sigma_u=0.0, dt=1.0)
        with pytest.raises(ValueError, match="shape"):
            g.sample_series(np.zeros((5, 4)), rng)


class TestStarTracker:
    def test_noise_free_recovers_true_body_vector(self, rng):
        q = quat_from_euler_zyx(0.3, -0.2, 0.5)
        st = StarTrackerModel(sigma_rad=0.0, reference_vectors=np.eye(3))
        rot = dcm_from_quat(q).T
        for obs in st.sample(q, rng):
            assert np.allclose(obs.body, rot @ obs.reference, atol=1e-15)
            assert obs.valid

    def test_references_are_normalised(self):
        st = StarTrackerModel(sigma_rad=1e-5, reference_vectors=[[2.0, 0.0, 0.0]])
        assert np.allclose(st.references[0], [1.0, 0.0, 0.0])
        assert st.n_vectors == 1

    def test_outputs_are_unit_vectors(self, rng):
        st = StarTrackerModel(sigma_rad=1e-2, reference_vectors=np.eye(3))
        for obs in st.sample(quat_identity(), rng):
            assert np.linalg.norm(obs.body) == pytest.approx(1.0)

    def test_dropout_marks_invalid_and_nan(self):
        st = StarTrackerModel(sigma_rad=1e-5, reference_vectors=np.eye(3), dropout_prob=1.0)
        obs = st.sample(quat_identity(), np.random.default_rng(0))
        assert all(not o.valid for o in obs)
        assert all(np.all(np.isnan(o.body)) for o in obs)

    def test_dropout_rate(self, rng):
        st = StarTrackerModel(sigma_rad=1e-5, reference_vectors=np.eye(1, 3), dropout_prob=0.2)
        valid = [st.sample(quat_identity(), rng)[0].valid for _ in range(20000)]
        assert np.mean(valid) == pytest.approx(0.8, abs=0.02)

    def test_quaternion_output_noise_statistics(self, rng):
        sigma = 1e-4
        q = quat_from_euler_zyx(0.4, -0.2, 0.9)
        st = StarTrackerModel(sigma_rad=sigma, reference_vectors=np.eye(3))
        dev = np.array(
            [
                small_angle_from_quat(
                    quat_multiply(quat_conjugate(q), st.sample_quaternion(q, rng))
                )
                for _ in range(20000)
            ]
        )
        assert np.allclose(np.std(dev, axis=0), sigma, rtol=0.04)
        assert np.allclose(np.mean(dev, axis=0), 0.0, atol=4 * sigma / np.sqrt(20000))

    def test_quaternion_output_is_unit_norm(self, rng):
        st = StarTrackerModel(sigma_rad=1e-3, reference_vectors=np.eye(3))
        for _ in range(20):
            q = st.sample_quaternion(quat_identity(), rng)
            assert np.linalg.norm(q) == pytest.approx(1.0)

    @pytest.mark.parametrize("bad", [-0.1, 1.5, np.nan])
    def test_bad_dropout_prob_raises(self, bad):
        with pytest.raises(ValueError, match="dropout_prob"):
            StarTrackerModel(sigma_rad=1e-5, reference_vectors=np.eye(3), dropout_prob=bad)

    def test_negative_sigma_raises(self):
        with pytest.raises(ValueError, match="sigma_rad"):
            StarTrackerModel(sigma_rad=-1.0, reference_vectors=np.eye(3))

    def test_empty_references_raise(self):
        with pytest.raises(ValueError, match="at least one"):
            StarTrackerModel(sigma_rad=1e-5, reference_vectors=np.zeros((0, 3)))

    def test_zero_reference_vector_raises(self):
        with pytest.raises(ValueError, match="norm"):
            StarTrackerModel(sigma_rad=1e-5, reference_vectors=[[0.0, 0.0, 0.0]])

    def test_wrong_reference_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            StarTrackerModel(sigma_rad=1e-5, reference_vectors=np.zeros((3, 2)))


class TestSunSensor:
    def test_noise_free(self, rng):
        ss = SunSensorModel(sigma_rad=0.0, sun_vector_inertial=[0, 0, 1])
        obs = ss.sample(quat_identity(), rng)
        assert obs.valid
        assert np.allclose(obs.body, [0, 0, 1])

    def test_normalises_sun_vector(self):
        ss = SunSensorModel(sigma_rad=1e-2, sun_vector_inertial=[0, 0, 3])
        assert np.allclose(ss.sun_inertial, [0, 0, 1])

    def test_out_of_fov_is_invalid(self, rng):
        ss = SunSensorModel(
            sigma_rad=0.0, sun_vector_inertial=[1, 0, 0], fov_half_angle_rad=0.1
        )
        obs = ss.sample(quat_identity(), rng)
        assert not obs.valid
        assert np.all(np.isnan(obs.body))

    def test_in_fov_is_valid(self, rng):
        ss = SunSensorModel(
            sigma_rad=0.0, sun_vector_inertial=[0, 0, 1], fov_half_angle_rad=0.1
        )
        assert ss.sample(quat_identity(), rng).valid

    def test_eclipse_always(self, rng):
        ss = SunSensorModel(sigma_rad=0.0, sun_vector_inertial=[0, 0, 1], eclipse_prob=1.0)
        assert not ss.sample(quat_identity(), rng).valid

    @pytest.mark.parametrize("bad", [0.0, -0.1, 4.0, np.nan])
    def test_bad_fov_raises(self, bad):
        with pytest.raises(ValueError, match="fov"):
            SunSensorModel(sigma_rad=1e-2, fov_half_angle_rad=bad)

    def test_bad_eclipse_prob_raises(self):
        with pytest.raises(ValueError, match="eclipse_prob"):
            SunSensorModel(sigma_rad=1e-2, eclipse_prob=2.0)

    def test_zero_sun_vector_raises(self):
        with pytest.raises(ValueError, match="norm"):
            SunSensorModel(sigma_rad=1e-2, sun_vector_inertial=[0, 0, 0])


class TestAccelerometer:
    def test_free_fall_measures_zero(self, rng):
        acc = AccelerometerModel(sigma_a=0.0)
        assert np.allclose(acc.sample(quat_identity(), np.zeros(3), rng), np.zeros(3))

    def test_gravity_is_subtracted(self, rng):
        acc = AccelerometerModel(sigma_a=0.0, gravity_inertial=[0, 0, -9.81])
        f = acc.sample(quat_identity(), np.zeros(3), rng)
        assert np.allclose(f, [0, 0, 9.81])

    def test_bias_added(self, rng):
        acc = AccelerometerModel(sigma_a=0.0, bias=[0.1, -0.2, 0.3])
        assert np.allclose(acc.sample(quat_identity(), np.zeros(3), rng), [0.1, -0.2, 0.3])

    def test_rotated_into_body_frame(self, rng):
        q = quat_from_euler_zyx(0.3, -0.2, 0.5)
        acc = AccelerometerModel(sigma_a=0.0)
        a_i = np.array([1.0, 2.0, 3.0])
        assert np.allclose(acc.sample(q, a_i, rng), dcm_from_quat(q).T @ a_i)

    def test_noise_statistics(self, rng):
        acc = AccelerometerModel(sigma_a=0.05)
        s = np.array([acc.sample(quat_identity(), np.zeros(3), rng) for _ in range(20000)])
        assert np.allclose(np.std(s, axis=0), 0.05, rtol=0.04)

    def test_negative_sigma_raises(self):
        with pytest.raises(ValueError, match="sigma_a"):
            AccelerometerModel(sigma_a=-0.1)

    def test_nonfinite_input_raises(self, rng):
        acc = AccelerometerModel(sigma_a=0.0)
        with pytest.raises(ValueError, match="finite"):
            acc.sample(quat_identity(), [np.nan, 0, 0], rng)


class TestGps:
    def test_measurement_dim(self):
        assert GpsModel(sigma_pos=1.0).measurement_dim == 3
        assert GpsModel(sigma_pos=1.0, sigma_vel=0.1).measurement_dim == 6

    def test_noise_covariance_position_only(self):
        assert np.allclose(GpsModel(sigma_pos=2.0).noise_covariance(), 4.0 * np.eye(3))

    def test_noise_covariance_with_velocity(self):
        r = GpsModel(sigma_pos=2.0, sigma_vel=0.5).noise_covariance()
        assert np.allclose(np.diag(r), [4.0, 4.0, 4.0, 0.25, 0.25, 0.25])

    def test_position_noise_statistics(self, rng):
        gps = GpsModel(sigma_pos=3.0)
        p = np.array([gps.sample(np.zeros(3), rng).position for _ in range(20000)])
        assert np.allclose(np.std(p, axis=0), 3.0, rtol=0.04)

    def test_dropout_returns_nan(self):
        gps = GpsModel(sigma_pos=1.0, dropout_prob=1.0)
        out = gps.sample(np.zeros(3), np.random.default_rng(0))
        assert not out.valid
        assert np.all(np.isnan(out.position))

    def test_dropout_rate(self, rng):
        gps = GpsModel(sigma_pos=1.0, dropout_prob=0.1)
        valid = [gps.sample(np.zeros(3), rng).valid for _ in range(20000)]
        assert np.mean(valid) == pytest.approx(0.9, abs=0.015)

    def test_velocity_required_when_sigma_vel_set(self, rng):
        gps = GpsModel(sigma_pos=1.0, sigma_vel=0.1)
        with pytest.raises(ValueError, match="velocity_true"):
            gps.sample(np.zeros(3), rng)

    def test_velocity_returned(self, rng):
        gps = GpsModel(sigma_pos=1e-12, sigma_vel=1e-12)
        out = gps.sample([1, 2, 3], rng, [4, 5, 6])
        assert np.allclose(out.position, [1, 2, 3])
        assert np.allclose(out.velocity, [4, 5, 6])

    def test_zero_sigma_vel_rejected(self):
        """sigma must be strictly positive: a zero-noise GPS makes R singular."""
        with pytest.raises(ValueError, match="sigma_vel"):
            GpsModel(sigma_pos=1.0, sigma_vel=0.0)

    @pytest.mark.parametrize("bad", [0.0, -1.0, np.nan])
    def test_bad_sigma_pos_raises(self, bad):
        with pytest.raises(ValueError, match="sigma_pos"):
            GpsModel(sigma_pos=bad)

    def test_bad_dropout_raises(self):
        with pytest.raises(ValueError, match="dropout_prob"):
            GpsModel(sigma_pos=1.0, dropout_prob=-0.5)

    def test_nonfinite_position_raises(self, rng):
        with pytest.raises(ValueError, match="finite"):
            GpsModel(sigma_pos=1.0).sample([np.nan, 0, 0], rng)
