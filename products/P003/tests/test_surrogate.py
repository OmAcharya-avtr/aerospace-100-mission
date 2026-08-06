"""Surrogate pipeline and baseline tests."""

import numpy as np
import pytest

from scintinet import Surrogate, rytov_baseline, scintillation_index_weak


def _analytic_grid():
    """Small (Cn2, L, lambda, D) grid with analytic sigma_I^2 targets."""
    rows = []
    for cn2 in (1e-16, 3e-16, 1e-15):
        for ell in (1000.0, 2000.0, 3000.0):
            for lam in (8.5e-7, 1.55e-6):
                for dia in (0.0, 0.05, 0.1):
                    rows.append([cn2, ell, lam, dia])
    x = np.array(rows)
    y = rytov_baseline(x)
    return x, y


class TestBaseline:
    def test_baseline_matches_scalar_function(self):
        x = np.array([[1e-15, 2000.0, 1.55e-6, 0.1]])
        expected = scintillation_index_weak(1e-15, 1.55e-6, 2000.0, aperture_diameter=0.1)
        assert rytov_baseline(x)[0] == pytest.approx(expected, rel=1e-12)

    def test_baseline_point_receiver_for_zero_d(self):
        x = np.array([[1e-15, 2000.0, 1.55e-6, 0.0]])
        expected = scintillation_index_weak(1e-15, 1.55e-6, 2000.0)
        assert rytov_baseline(x)[0] == pytest.approx(expected, rel=1e-12)

    def test_baseline_bad_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            rytov_baseline(np.ones((3, 2)))


class TestSurrogatePipeline:
    def test_fit_predict_recovers_analytic_surface(self):
        # Train on an analytic surface; the MLP ensemble should interpolate
        # it well (median relative error under 15% on the training grid).
        x, y = _analytic_grid()
        s = Surrogate(n_members=3, hidden_layer_sizes=(16, 16), random_state=0)
        s.fit(x, y)
        pred = s.predict(x)
        rel = np.abs(pred - y) / y
        assert np.median(rel) < 0.15

    def test_return_std_shape_and_nonnegative(self):
        x, y = _analytic_grid()
        s = Surrogate(n_members=3, hidden_layer_sizes=(8,), random_state=1)
        s.fit(x, y)
        mean, std = s.predict(x[:5], return_std=True)
        assert mean.shape == (5,)
        assert std.shape == (5,)
        assert np.all(std >= 0.0)
        assert np.all(mean > 0.0)  # log-space model cannot go non-positive

    def test_deterministic_given_random_state(self):
        x, y = _analytic_grid()
        p1 = Surrogate(n_members=2, hidden_layer_sizes=(8,), random_state=7).fit(x, y).predict(x)
        p2 = Surrogate(n_members=2, hidden_layer_sizes=(8,), random_state=7).fit(x, y).predict(x)
        assert np.array_equal(p1, p2)

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="fit"):
            Surrogate().predict(np.array([[1e-15, 2000.0, 1.55e-6, 0.1]]))

    def test_single_member_rejected(self):
        with pytest.raises(ValueError, match="n_members"):
            Surrogate(n_members=1)

    def test_mismatched_lengths_raise(self):
        x, y = _analytic_grid()
        with pytest.raises(ValueError, match="rows"):
            Surrogate(n_members=2).fit(x, y[:-3])

    def test_nonpositive_targets_raise(self):
        x, y = _analytic_grid()
        y = y.copy()
        y[0] = 0.0
        with pytest.raises(ValueError, match="finite and > 0"):
            Surrogate(n_members=2).fit(x, y)

    def test_negative_inputs_raise(self):
        x = np.array([[-1e-15, 2000.0, 1.55e-6, 0.1]])
        with pytest.raises(ValueError, match="> 0"):
            rytov_baseline(x)
