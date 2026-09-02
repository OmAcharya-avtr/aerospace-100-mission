"""Plant discretisation, controller gains and their input validation.

The known-answer tests carry the hand arithmetic in comments; the numbers are
small enough to check on paper, which is the point.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fdiscope.plant import ControllerGains, PlantConfig, loop_matrices


class TestPlantConfig:
    def test_defaults_are_a_coarse_sensor_suite(self):
        # 0.05 deg = 8.7266e-4 rad -> variance 7.6154e-7 rad^2
        # 0.02 deg/s = 3.4907e-4 rad/s -> variance 1.2185e-7 (rad/s)^2
        p = PlantConfig()
        assert np.isclose(np.sqrt(p.attitude_var_rad2), np.radians(0.05), rtol=2e-5)
        assert np.isclose(np.sqrt(p.gyro_var_rad2_s2), np.radians(0.02), rtol=2e-5)

    @pytest.mark.parametrize(
        "field",
        ["inertia_kgm2", "dt_s", "attitude_var_rad2", "gyro_var_rad2_s2", "max_torque_nm"],
    )
    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_rejects_non_positive_parameters(self, field, bad):
        with pytest.raises(ValueError, match=field):
            PlantConfig(**{field: bad})

    def test_rejects_negative_noise_density(self):
        with pytest.raises(ValueError, match="torque_noise_psd"):
            PlantConfig(torque_noise_psd=-1.0)

    def test_zero_noise_density_is_allowed(self):
        assert PlantConfig(torque_noise_psd=0.0).torque_noise_psd == 0.0


class TestLoopMatrices:
    def test_known_answer_transition_and_input(self):
        # dt = 0.1 s, J = 12 kg m^2:
        #   F = [[1, 0.1], [0, 1]]
        #   G = [[0.5 * 0.1^2 / 12], [0.1 / 12]]
        #     = [[0.005 / 12], [0.1 / 12]]
        #     = [[4.1666666667e-4], [8.3333333333e-3]]
        m = loop_matrices(PlantConfig(inertia_kgm2=12.0, dt_s=0.1))
        assert np.allclose(m.f, [[1.0, 0.1], [0.0, 1.0]])
        assert np.allclose(m.g, [[4.1666666666666667e-4], [8.3333333333333333e-3]])

    def test_known_answer_process_noise(self):
        # q_t = 4e-8, J = 12, dt = 0.1:
        #   scale = 4e-8 / 144 = 2.7777777778e-10
        #   Q = scale * [[dt^3/3, dt^2/2], [dt^2/2, dt]]
        #     = 2.7777777778e-10 * [[3.3333333333e-4, 5e-3], [5e-3, 0.1]]
        #     = [[9.2592592593e-14, 1.3888888889e-12],
        #        [1.3888888889e-12, 2.7777777778e-11]]
        m = loop_matrices(PlantConfig(inertia_kgm2=12.0, dt_s=0.1, torque_noise_psd=4e-8))
        assert np.allclose(
            m.q,
            [
                [9.2592592592592e-14, 1.3888888888889e-12],
                [1.3888888888889e-12, 2.7777777777778e-11],
            ],
            rtol=1e-12,
        )

    def test_measurement_matrix_is_identity(self):
        m = loop_matrices(PlantConfig())
        assert np.allclose(m.h, np.eye(2))

    def test_measurement_covariance_is_the_sensor_variances(self):
        p = PlantConfig(attitude_var_rad2=3.0, gyro_var_rad2_s2=5.0)
        assert np.allclose(loop_matrices(p).r, np.diag([3.0, 5.0]))

    def test_process_noise_is_symmetric_positive_semidefinite(self):
        q = loop_matrices(PlantConfig()).q
        assert np.allclose(q, q.T)
        assert np.min(np.linalg.eigvalsh(q)) >= -1e-30

    def test_rejects_wrong_type(self):
        with pytest.raises(TypeError, match="PlantConfig"):
            loop_matrices({"dt_s": 0.1})

    @settings(max_examples=40, deadline=None)
    @given(
        dt=st.floats(1e-3, 1.0, allow_nan=False),
        j=st.floats(0.1, 500.0, allow_nan=False),
    )
    def test_g_matches_the_closed_form_for_any_dt_and_inertia(self, dt, j):
        m = loop_matrices(PlantConfig(dt_s=dt, inertia_kgm2=j))
        assert np.isclose(m.g[0, 0], 0.5 * dt * dt / j, rtol=1e-12)
        assert np.isclose(m.g[1, 0], dt / j, rtol=1e-12)

    @settings(max_examples=40, deadline=None)
    @given(dt=st.floats(1e-3, 1.0, allow_nan=False))
    def test_process_noise_determinant_is_non_negative(self, dt):
        q = loop_matrices(PlantConfig(dt_s=dt)).q
        # det = scale^2 (dt^4/3 - dt^4/4) = scale^2 dt^4 / 12 >= 0
        assert np.linalg.det(q) >= -1e-40


class TestControllerGains:
    def test_known_answer_gains(self):
        # omega_n = 0.35 rad/s, zeta = 0.707:
        #   kp = 0.35^2      = 0.1225 1/s^2
        #   kd = 2*0.707*0.35 = 0.4949 1/s
        g = ControllerGains(natural_freq_rad_s=0.35, damping=0.707)
        assert np.isclose(g.kp, 0.1225)
        assert np.isclose(g.kd, 0.4949)

    def test_known_answer_torque(self):
        # u = -J (kp*theta + kd*omega) with J = 10, theta = 0.02, omega = -0.01:
        #   u = -10 * (0.1225*0.02 + 0.4949*(-0.01))
        #     = -10 * (0.00245 - 0.004949)
        #     = -10 * (-0.002499) = 0.02499 N m
        g = ControllerGains()
        assert np.isclose(g.torque([0.02, -0.01], 10.0), 0.02499)

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
    def test_rejects_bad_frequency(self, bad):
        with pytest.raises(ValueError, match="natural_freq_rad_s"):
            ControllerGains(natural_freq_rad_s=bad)

    @pytest.mark.parametrize("bad", [0.0, -0.5, float("inf")])
    def test_rejects_bad_damping(self, bad):
        with pytest.raises(ValueError, match="damping"):
            ControllerGains(damping=bad)

    def test_rejects_wrong_state_length(self):
        with pytest.raises(ValueError, match="2 elements"):
            ControllerGains().torque([1.0, 2.0, 3.0], 1.0)

    def test_torque_is_linear_in_the_state(self):
        g = ControllerGains()
        a = g.torque([0.01, 0.0], 5.0)
        b = g.torque([0.02, 0.0], 5.0)
        assert np.isclose(b, 2.0 * a)
