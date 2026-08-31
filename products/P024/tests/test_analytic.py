"""Orbit-averaged field moments and the first-order detumble model."""

from __future__ import annotations

import numpy as np
import pytest

from detumblesim.analytic import (
    FieldMoments,
    damping_matrix,
    detumble_time_first_order,
    geometry_factors,
    max_torque_nm,
    modal_time_constants,
    orbit_field_moments,
    saturation_time_bound_s,
)
from detumblesim.orbit import CircularOrbit
from detumblesim.spacecraft import Magnetorquer, inertia_from_diagonal


def isotropic_moments(b_rms: float = 3.0e-5) -> FieldMoments:
    """Synthetic moments with a perfectly isotropic field-direction history."""
    b2 = b_rms**2
    return FieldMoments(
        mean_b2_t2=b2,
        outer_t2=(b2 / 3.0) * np.eye(3),
        rms_b_t=b_rms,
        n_samples=3,
        span_s=1.0,
    )


class TestFieldMoments:
    def test_rms_matches_mean_square(self):
        o = CircularOrbit(altitude_km=500.0, inclination_deg=97.4)
        m = orbit_field_moments(o, 500)
        assert np.isclose(m.rms_b_t**2, m.mean_b2_t2, rtol=1e-12)

    def test_trace_of_outer_equals_mean_b2(self):
        m = orbit_field_moments(CircularOrbit(), 400)
        assert np.isclose(float(np.trace(m.outer_t2)), m.mean_b2_t2, rtol=1e-12)

    def test_leo_rms_field_is_physical(self):
        m = orbit_field_moments(CircularOrbit(altitude_km=500.0, inclination_deg=97.4), 800)
        assert 15e-6 < m.rms_b_t < 60e-6

    @pytest.mark.parametrize("n", [0, 1])
    def test_rejects_too_few_samples(self, n):
        with pytest.raises(ValueError, match="n_samples"):
            orbit_field_moments(CircularOrbit(), n)

    def test_rejects_bad_span(self):
        with pytest.raises(ValueError, match="span_s"):
            orbit_field_moments(CircularOrbit(), 10, span_s=-1.0)


class TestGeometryFactors:
    def test_isotropic_case_is_exactly_two_thirds(self):
        g = geometry_factors(isotropic_moments())
        assert np.allclose(g, 2.0 / 3.0, atol=1e-14)

    def test_eigenvalues_always_sum_to_two(self):
        for inc in (0.0, 30.0, 63.4, 97.4):
            m = orbit_field_moments(CircularOrbit(inclination_deg=inc), 1000)
            assert np.isclose(float(np.sum(geometry_factors(m))), 2.0, atol=1e-12)

    def test_equatorial_orbit_has_a_much_smaller_minimum(self):
        eq = orbit_field_moments(CircularOrbit(inclination_deg=0.0), 2000)
        polar = orbit_field_moments(CircularOrbit(inclination_deg=97.4), 2000)
        assert geometry_factors(eq)[0] < 0.3 * geometry_factors(polar)[0]


class TestDampingMatrix:
    def test_isotropic_damping_matrix(self):
        m = isotropic_moments()
        d = damping_matrix(m, 1.5)
        assert np.allclose(d, 1.5 * (2.0 / 3.0) * m.mean_b2_t2 * np.eye(3), atol=1e-20)

    def test_is_symmetric_positive_semidefinite(self):
        m = orbit_field_moments(CircularOrbit(inclination_deg=51.6), 1000)
        d = damping_matrix(m, 1e5)
        assert np.allclose(d, d.T)
        assert np.min(np.linalg.eigvalsh(d)) >= -1e-18

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_rejects_bad_gain(self, bad):
        with pytest.raises(ValueError, match="gain"):
            damping_matrix(isotropic_moments(), bad)


class TestTimeConstants:
    def test_known_answer_isotropic(self):
        # tau = 3 j / (2 k <B^2>).  With j = 0.05, k = 1e5, B_rms = 3e-5:
        # tau = 0.15 / (2e5 * 9e-10) = 0.15 / 1.8e-4 = 833.333... s
        m = isotropic_moments(3.0e-5)
        taus = modal_time_constants(m, 1.0e5, 0.05)
        assert np.allclose(taus, 833.3333333, rtol=1e-8)

    def test_detumble_time_isotropic_known_answer(self):
        # t = tau ln(w0/wf) = 833.3333 * ln(10) = 1918.75 s
        m = isotropic_moments(3.0e-5)
        t = detumble_time_first_order(0.05, 1.0e5, m, 0.1, 0.01, "isotropic")
        assert np.isclose(t, 833.3333333 * np.log(10.0), rtol=1e-8)

    def test_modes_bracket_the_isotropic_estimate(self):
        m = orbit_field_moments(CircularOrbit(inclination_deg=97.4), 2000)
        fast = detumble_time_first_order(0.05, 1e5, m, 0.1, 0.01, "fastest")
        iso = detumble_time_first_order(0.05, 1e5, m, 0.1, 0.01, "isotropic")
        slow = detumble_time_first_order(0.05, 1e5, m, 0.1, 0.01, "slowest")
        assert fast < iso < slow

    def test_inverse_gain_scaling_is_exact_in_the_model(self):
        m = orbit_field_moments(CircularOrbit(), 1000)
        a = detumble_time_first_order(0.05, 1e4, m, 0.1, 0.01)
        b = detumble_time_first_order(0.05, 1e5, m, 0.1, 0.01)
        assert np.isclose(a / b, 10.0, rtol=1e-12)

    @pytest.mark.parametrize("bad", [0.0, -0.1])
    def test_rejects_bad_inertia(self, bad):
        with pytest.raises(ValueError, match="inertia_scalar"):
            modal_time_constants(isotropic_moments(), 1e5, bad)

    def test_rejects_bad_rates(self):
        m = isotropic_moments()
        with pytest.raises(ValueError, match="omega0_rad_s"):
            detumble_time_first_order(0.05, 1e5, m, -1.0, 0.01)
        with pytest.raises(ValueError, match="omega_target_rad_s"):
            detumble_time_first_order(0.05, 1e5, m, 0.1, 0.0)
        with pytest.raises(ValueError, match="smaller than"):
            detumble_time_first_order(0.05, 1e5, m, 0.01, 0.1)

    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError, match="mode must be"):
            detumble_time_first_order(0.05, 1e5, isotropic_moments(), 0.1, 0.01, "best")

    def test_singular_damping_matrix_is_reported(self):
        # A field confined to one inertial direction leaves that axis
        # uncontrollable: <B B^T> = <|B|^2> z z^T gives a zero eigenvalue.
        b2 = 1e-9
        m = FieldMoments(b2, np.diag([0.0, 0.0, b2]), np.sqrt(b2), 2, 1.0)
        with pytest.raises(ValueError, match="uncontrollable"):
            modal_time_constants(m, 1e5, 0.05)


class TestSaturationBound:
    def test_known_answer_max_torque(self):
        # For an isotropic 0.2 A m^2 box and B = (0, 0, 3e-5) T, the largest
        # |m x B| is at a corner: |(0.2, 0.2, *) x (0, 0, 3e-5)|
        #   = 3e-5 * sqrt(0.2^2 + 0.2^2) = 8.4853e-6 N m.
        m = Magnetorquer.isotropic(0.2)
        assert np.isclose(
            max_torque_nm(m, [0.0, 0.0, 3e-5]), 3e-5 * np.sqrt(0.08), rtol=1e-12
        )

    def test_max_torque_rejects_bad_shape(self):
        with pytest.raises(ValueError, match="shape"):
            max_torque_nm(Magnetorquer.isotropic(0.2), [1.0, 2.0])

    def test_bound_is_positive_and_shorter_than_a_real_run(self):
        mom = isotropic_moments(3e-5)
        t = saturation_time_bound_s(
            Magnetorquer.isotropic(0.2), mom, inertia_from_diagonal(0.05, 0.05, 0.05),
            np.array([0.1, 0.0, 0.0]), 0.01,
        )
        assert t > 0.0

    def test_bound_rejects_target_above_initial(self):
        with pytest.raises(ValueError, match="target momentum"):
            saturation_time_bound_s(
                Magnetorquer.isotropic(0.2), isotropic_moments(),
                inertia_from_diagonal(0.05, 0.05, 0.05), np.array([0.01, 0.0, 0.0]), 1.0,
            )
