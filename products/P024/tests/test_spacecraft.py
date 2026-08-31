"""Inertia validation and the magnetorquer hardware model."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from detumblesim.spacecraft import Magnetorquer, inertia_from_diagonal, validate_inertia


class TestInertia:
    def test_known_answer(self):
        j = inertia_from_diagonal(0.05, 0.06, 0.04)
        assert np.allclose(np.diag(j), [0.05, 0.06, 0.04])
        assert np.allclose(j - np.diag(np.diag(j)), 0.0)

    @pytest.mark.parametrize("vals", [(0.0, 1.0, 1.0), (-1.0, 1.0, 1.0)])
    def test_rejects_non_positive(self, vals):
        with pytest.raises(ValueError, match="positive"):
            inertia_from_diagonal(*vals)

    def test_rejects_triangle_inequality_violation(self):
        # Ixx + Iyy = 0.02 < Izz = 0.5 is not a physically realisable body.
        with pytest.raises(ValueError, match="triangle inequality"):
            inertia_from_diagonal(0.01, 0.01, 0.5)

    def test_validate_rejects_asymmetric(self):
        j = np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        with pytest.raises(ValueError, match="symmetric"):
            validate_inertia(j)

    def test_validate_rejects_indefinite(self):
        with pytest.raises(ValueError, match="positive definite"):
            validate_inertia(np.diag([1.0, 1.0, -1.0]))

    def test_validate_rejects_wrong_shape(self):
        with pytest.raises(ValueError, match="shape"):
            validate_inertia(np.eye(2))

    def test_validate_rejects_non_finite(self):
        with pytest.raises(ValueError, match="non-finite"):
            validate_inertia(np.diag([1.0, np.nan, 1.0]))


class TestMagnetorquer:
    def test_known_answer_saturation(self):
        m = Magnetorquer.isotropic(0.2)
        clipped, sat = m.saturate([0.3, -0.1, -0.5])
        assert np.allclose(clipped, [0.2, -0.1, -0.2])
        assert sat is True

    def test_no_saturation_flag_when_within_limits(self):
        m = Magnetorquer.isotropic(0.2)
        clipped, sat = m.saturate([0.1, -0.1, 0.05])
        assert sat is False
        assert np.allclose(clipped, [0.1, -0.1, 0.05])

    def test_per_axis_limits(self):
        m = Magnetorquer(np.array([0.1, 0.4, 0.2]))
        clipped, sat = m.saturate([0.3, 0.3, 0.3])
        assert np.allclose(clipped, [0.1, 0.3, 0.2])
        assert sat is True

    def test_max_norm_is_the_box_corner(self):
        m = Magnetorquer(np.array([0.1, 0.2, 0.2]))
        assert np.isclose(m.max_norm_am2, np.sqrt(0.01 + 0.04 + 0.04))

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("inf")])
    def test_isotropic_rejects_bad_limit(self, bad):
        with pytest.raises(ValueError, match="max_dipole_am2"):
            Magnetorquer.isotropic(bad)

    def test_rejects_bad_shape(self):
        with pytest.raises(ValueError, match="shape"):
            Magnetorquer(np.array([0.1, 0.2]))

    def test_rejects_non_positive_axis(self):
        with pytest.raises(ValueError, match="positive"):
            Magnetorquer(np.array([0.1, 0.0, 0.2]))

    def test_saturate_rejects_bad_shape(self):
        with pytest.raises(ValueError, match="shape"):
            Magnetorquer.isotropic(0.2).saturate([1.0, 2.0])

    @given(
        v=st.lists(st.floats(-5.0, 5.0), min_size=3, max_size=3),
        lim=st.floats(0.01, 1.0),
    )
    @settings(max_examples=60, deadline=None)
    def test_clipping_is_idempotent_and_bounded(self, v, lim):
        m = Magnetorquer.isotropic(lim)
        once, _ = m.saturate(v)
        twice, sat2 = m.saturate(once)
        assert np.allclose(once, twice)
        assert sat2 is False
        assert np.all(np.abs(once) <= lim + 1e-15)
