"""Controllability-gap quantification."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from detumblesim.controllability import (
    controllability_report,
    instantaneous_projector,
    residual_rate_along,
    uncontrollable_fraction,
)
from detumblesim.orbit import CircularOrbit

vec3 = st.lists(st.floats(-3.0, 3.0), min_size=3, max_size=3)


class TestInstantaneous:
    def test_known_answer_projector(self):
        p = instantaneous_projector([0.0, 0.0, 5.0])
        assert np.allclose(p, np.diag([1.0, 1.0, 0.0]))

    def test_projector_is_idempotent_and_rank_two(self):
        p = instantaneous_projector([1.0, 2.0, -3.0])
        assert np.allclose(p @ p, p)
        assert np.isclose(float(np.trace(p)), 2.0)
        assert np.linalg.matrix_rank(p, tol=1e-10) == 2

    def test_known_answer_uncontrollable_fraction(self):
        # omega parallel to B: the whole rate is uncontrollable.
        assert np.isclose(uncontrollable_fraction([0.0, 0.0, 2.0], [0.0, 0.0, 7.0]), 1.0)
        # omega perpendicular to B: none of it is.
        assert np.isclose(uncontrollable_fraction([1.0, 0.0, 0.0], [0.0, 0.0, 7.0]), 0.0)
        # 45 degrees: cos 45 = 0.70710678
        assert np.isclose(
            uncontrollable_fraction([1.0, 0.0, 1.0], [0.0, 0.0, 1.0]), np.sqrt(0.5)
        )

    def test_zero_rate_is_reported_as_zero(self):
        assert uncontrollable_fraction([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]) == 0.0

    def test_rejects_zero_field(self):
        with pytest.raises(ValueError, match="non-zero"):
            uncontrollable_fraction([1.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="non-zero"):
            instantaneous_projector([0.0, 0.0, 0.0])

    def test_rejects_bad_shapes(self):
        with pytest.raises(ValueError, match="shape"):
            uncontrollable_fraction([1.0, 0.0], [0.0, 0.0, 1.0])
        with pytest.raises(ValueError, match="shape"):
            instantaneous_projector([1.0, 0.0])

    @given(w=vec3, b=vec3)
    @settings(max_examples=80, deadline=None)
    def test_fraction_is_in_the_unit_interval(self, w, b):
        if float(np.linalg.norm(b)) < 1e-6:
            return
        f = uncontrollable_fraction(w, b)
        assert -1e-12 <= f <= 1.0 + 1e-12


class TestReport:
    def test_eigenvalues_sum_to_two(self):
        for inc in (0.0, 45.0, 97.4):
            r = controllability_report(CircularOrbit(inclination_deg=inc), 2000)
            assert np.isclose(float(np.sum(r.weighted_eigenvalues)), 2.0, atol=1e-12)
            assert np.isclose(float(np.sum(r.direction_eigenvalues)), 2.0, atol=1e-12)

    def test_eigenvalues_are_ascending_and_non_negative(self):
        r = controllability_report(CircularOrbit(inclination_deg=63.4), 2000)
        assert np.all(np.diff(r.weighted_eigenvalues) >= -1e-12)
        assert r.weighted_eigenvalues[0] >= -1e-12

    def test_equatorial_orbit_is_far_more_anisotropic_than_polar(self):
        eq = controllability_report(CircularOrbit(inclination_deg=0.0), 3000)
        pol = controllability_report(CircularOrbit(inclination_deg=97.4), 3000)
        assert eq.anisotropy > 5.0 * pol.anisotropy
        assert eq.weighted_eigenvalues[0] < 0.15
        assert pol.weighted_eigenvalues[0] > 0.35

    def test_equatorial_weak_direction_is_close_to_the_polar_axis(self):
        eq = controllability_report(CircularOrbit(inclination_deg=0.0), 3000)
        assert abs(float(eq.weakest_direction_eci[2])) > 0.95

    def test_isotropic_reference_is_two_thirds(self):
        r = controllability_report(CircularOrbit(), 500)
        assert np.isclose(r.isotropic_reference, 2.0 / 3.0)

    def test_uncontrollable_fraction_is_reported(self):
        r = controllability_report(CircularOrbit(inclination_deg=0.0), 2000)
        assert 0.0 <= r.mean_uncontrollable_fraction <= 1.0
        # For a near-equatorial orbit the field stays near the dipole axis, so
        # the weakest direction has a large mean alignment with B.
        assert r.mean_uncontrollable_fraction > 0.8

    @pytest.mark.parametrize("n", [0, 1])
    def test_rejects_too_few_samples(self, n):
        with pytest.raises(ValueError, match="n_samples"):
            controllability_report(CircularOrbit(), n)

    def test_rejects_bad_span(self):
        with pytest.raises(ValueError, match="span_s"):
            controllability_report(CircularOrbit(), 100, span_s=0.0)


class TestResidualRate:
    def test_identity_attitude_projects_directly(self):
        w = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 1.0]])
        q = np.tile([1.0, 0.0, 0.0, 0.0], (2, 1))
        assert np.allclose(residual_rate_along(w, [0.0, 0.0, 1.0], q), [3.0, 1.0])

    def test_direction_is_normalised_internally(self):
        w = np.array([[1.0, 2.0, 3.0]])
        q = np.array([[1.0, 0.0, 0.0, 0.0]])
        assert np.allclose(
            residual_rate_along(w, [0.0, 0.0, 5.0], q),
            residual_rate_along(w, [0.0, 0.0, 1.0], q),
        )

    def test_rejects_bad_shapes(self):
        w = np.zeros((3, 3))
        with pytest.raises(ValueError, match="omega_history"):
            residual_rate_along(np.zeros((3, 2)), [0, 0, 1], np.zeros((3, 4)))
        with pytest.raises(ValueError, match="quat_history"):
            residual_rate_along(w, [0, 0, 1], np.zeros((2, 4)))
        with pytest.raises(ValueError, match="direction_eci"):
            residual_rate_along(w, [0, 1], np.zeros((3, 4)))

    def test_rejects_zero_direction(self):
        with pytest.raises(ValueError, match="non-zero"):
            residual_rate_along(np.zeros((2, 3)), [0.0, 0.0, 0.0], np.zeros((2, 4)) + [1, 0, 0, 0])
