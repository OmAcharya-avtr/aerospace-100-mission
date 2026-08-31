"""Observable feature extraction."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from detumblesim.features import (
    FEATURE_NAMES,
    N_FEATURES,
    TelemetryWindow,
    rate_proxy,
)


class TestRateProxy:
    def test_known_answer(self):
        # |dB/dt| / |B| for B = (0, 0, 4e-5), dB/dt = (8e-6, 0, 0) is
        # 8e-6 / 4e-5 = 0.2 rad/s.
        assert np.isclose(rate_proxy([0.0, 0.0, 4e-5], [8e-6, 0.0, 0.0]), 0.2)

    def test_equals_omega_perp_for_a_rotating_field(self):
        # With dB/dt = -omega x B the proxy is exactly |omega_perp|.
        b = np.array([3e-5, -1e-5, 2e-5])
        w = np.array([0.05, 0.02, -0.03])
        bd = -np.cross(w, b)
        w_perp = w - (float(w @ b) / float(b @ b)) * b
        assert np.isclose(rate_proxy(b, bd), float(np.linalg.norm(w_perp)))

    def test_zero_field_gives_zero(self):
        assert rate_proxy([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]) == 0.0

    def test_rejects_bad_shape(self):
        with pytest.raises(ValueError, match="shape"):
            rate_proxy([1.0, 2.0], [1.0, 2.0, 3.0])

    @given(
        b=st.lists(st.floats(-1e-4, 1e-4), min_size=3, max_size=3),
        w=st.lists(st.floats(-1.0, 1.0), min_size=3, max_size=3),
    )
    @settings(max_examples=80, deadline=None)
    def test_proxy_never_exceeds_the_true_rate(self, b, w):
        bb, ww = np.asarray(b), np.asarray(w)
        if float(np.linalg.norm(bb)) < 1e-9:
            return
        assert rate_proxy(bb, -np.cross(ww, bb)) <= float(np.linalg.norm(ww)) + 1e-12


class TestTelemetryWindow:
    def _fill(self, w, n=80, rate=0.1, bmag=3e-5):
        for i in range(n):
            b = np.array([bmag, 0.0, 0.0])
            w.push(i * 2.0, b, np.array([0.0, rate * bmag, 0.0]), False)
        return w

    def test_length_is_bounded(self):
        w = self._fill(TelemetryWindow(10), 50)
        assert len(w) == 10

    def test_feature_vector_shape_and_names(self):
        w = self._fill(TelemetryWindow(20))
        x = w.features(0.2, 0.05)
        assert x.shape == (N_FEATURES,)
        assert len(FEATURE_NAMES) == N_FEATURES
        assert np.all(np.isfinite(x))

    def test_known_answer_constant_stream(self):
        w = self._fill(TelemetryWindow(20), rate=0.1, bmag=3e-5)
        x = w.features(0.2, 0.05)
        assert np.isclose(x[0], np.log10(0.1))  # rate proxy
        assert np.isclose(x[1], np.log10(3e-5))  # mean |B|
        assert abs(x[2]) < 1e-9  # no trend
        assert x[3] == 0.0  # no saturation
        assert np.isclose(x[4], 0.0)  # constant field magnitude
        assert np.isclose(x[5], np.log10(0.2))
        assert np.isclose(x[6], np.log10(0.05))

    def test_saturation_duty(self):
        w = TelemetryWindow(10)
        for i in range(10):
            w.push(i * 1.0, [3e-5, 0, 0], [0, 3e-6, 0], i % 2 == 0)
        assert np.isclose(w.features(0.2, 0.05)[3], 0.5)

    def test_decreasing_rate_gives_a_negative_trend(self):
        w = TelemetryWindow(40)
        for i in range(40):
            r = 0.2 * np.exp(-i / 20.0)
            w.push(i * 2.0, [3e-5, 0, 0], [0, r * 3e-5, 0], False)
        assert w.features(0.2, 0.05)[2] < 0.0

    def test_not_ready_before_two_samples(self):
        w = TelemetryWindow(5)
        assert not w.ready()
        with pytest.raises(ValueError, match="at least two samples"):
            w.features(0.2, 0.05)
        w.push(0.0, [3e-5, 0, 0], [0, 0, 0], False)
        assert not w.ready()
        w.push(1.0, [3e-5, 0, 0], [0, 0, 0], False)
        assert w.ready()

    @pytest.mark.parametrize("n", [0, 1])
    def test_rejects_short_window(self, n):
        with pytest.raises(ValueError, match="length"):
            TelemetryWindow(n)

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_rejects_bad_hardware_parameters(self, bad):
        w = self._fill(TelemetryWindow(5))
        with pytest.raises(ValueError, match="max_dipole_am2"):
            w.features(bad, 0.05)
        with pytest.raises(ValueError, match="inertia_scale_kgm2"):
            w.features(0.2, bad)

    def test_zero_rate_is_floored_not_infinite(self):
        w = TelemetryWindow(5)
        for i in range(5):
            w.push(i * 1.0, [3e-5, 0, 0], [0.0, 0.0, 0.0], False)
        x = w.features(0.2, 0.05)
        assert np.all(np.isfinite(x))
        assert x[0] == np.log10(1e-7)
