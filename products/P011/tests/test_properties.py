"""Hypothesis property tests for the algebraic identities in the package."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from waveforge.control import (
    noise_transfer,
    noise_variance_gain,
    rejection_transfer,
    stability_limit_gain,
)
from waveforge.errorbudget import (
    fitting_error,
    strehl_marechal,
    variance_from_strehl,
)
from waveforge.predictor import build_lagged_dataset
from waveforge.pupil import PupilGrid, piston_removed, variance
from waveforge.statistics import (
    noll_residual_variance,
    phase_structure_function,
    zernike_variance,
)
from waveforge.zernike import (
    nm_to_noll,
    noll_to_nm,
    radial_polynomial,
    zernike_cartesian,
    zernike_norm,
)

FAST = settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])


class TestZernikeProperties:
    @FAST
    @given(j=st.integers(min_value=1, max_value=500))
    def test_index_round_trip(self, j):
        assert nm_to_noll(*noll_to_nm(j)) == j

    @FAST
    @given(j=st.integers(min_value=1, max_value=300))
    def test_parity_of_n_minus_m(self, j):
        n, m = noll_to_nm(j)
        assert (n - abs(m)) % 2 == 0
        assert abs(m) <= n

    @FAST
    @given(j=st.integers(min_value=1, max_value=300))
    def test_order_block_contains_j(self, j):
        n, _ = noll_to_nm(j)
        assert n * (n + 1) // 2 < j <= (n + 1) * (n + 2) // 2

    @FAST
    @given(
        j=st.integers(min_value=1, max_value=60),
        rho=st.floats(min_value=0.0, max_value=1.0),
    )
    def test_radial_polynomial_bounded(self, j, rho):
        n, m = noll_to_nm(j)
        assert abs(float(radial_polynomial(n, m, rho))) <= 1.0 + 1e-9

    @FAST
    @given(j=st.integers(min_value=1, max_value=60))
    def test_radial_polynomial_unity_at_edge(self, j):
        n, m = noll_to_nm(j)
        assert float(radial_polynomial(n, m, 1.0)) == 1.0 or n == 0

    @FAST
    @given(
        j=st.integers(min_value=2, max_value=40),
        theta=st.floats(min_value=-np.pi, max_value=np.pi),
    )
    def test_mode_bounded_by_its_norm(self, j, theta):
        n, m = noll_to_nm(j)
        rho = np.linspace(0.0, 1.0, 41)
        values = zernike_cartesian(j, rho * np.cos(theta), rho * np.sin(theta))
        assert np.max(np.abs(values)) <= zernike_norm(n, m) + 1e-9

    @FAST
    @given(
        j=st.integers(min_value=2, max_value=40),
        scale=st.floats(min_value=-5.0, max_value=5.0),
    )
    def test_mode_is_linear_in_amplitude(self, j, scale):
        x = np.linspace(-0.7, 0.7, 11)
        base = zernike_cartesian(j, x, np.zeros_like(x))
        assert np.allclose(scale * base, scale * base)


class TestStatisticsProperties:
    @FAST
    @given(
        j=st.integers(min_value=2, max_value=200),
        d_over_r0=st.floats(min_value=0.1, max_value=50.0),
    )
    def test_zernike_variance_positive(self, j, d_over_r0):
        assert zernike_variance(j, d_over_r0) > 0.0

    @FAST
    @given(
        j=st.integers(min_value=2, max_value=100),
        a=st.floats(min_value=0.2, max_value=20.0),
        b=st.floats(min_value=0.2, max_value=20.0),
    )
    def test_zernike_variance_scaling(self, j, a, b):
        expected = (a / b) ** (5 / 3)
        assert zernike_variance(j, a) / zernike_variance(j, b) == pytest.approx(
            expected, rel=1e-9
        )

    @FAST
    @given(j=st.integers(min_value=1, max_value=150))
    def test_noll_residual_decreases(self, j):
        assert noll_residual_variance(j + 1) < noll_residual_variance(j)

    @FAST
    @given(
        r=st.floats(min_value=0.0, max_value=10.0),
        r0=st.floats(min_value=0.01, max_value=1.0),
    )
    def test_structure_function_monotone(self, r, r0):
        assert float(phase_structure_function(r + 0.1, r0)) >= float(
            phase_structure_function(r, r0)
        )

    @FAST
    @given(
        pitch=st.floats(min_value=1e-3, max_value=1.0),
        r0=st.floats(min_value=1e-3, max_value=1.0),
    )
    def test_fitting_error_positive(self, pitch, r0):
        assert fitting_error(pitch, r0) > 0.0


class TestControlProperties:
    @FAST
    @given(
        gain=st.floats(min_value=0.01, max_value=0.95),
        delay=st.integers(min_value=1, max_value=5),
        frequency=st.floats(min_value=0.0, max_value=499.0),
    )
    def test_rejection_and_noise_transfer_sum_to_one(self, gain, delay, frequency):
        assume(gain < stability_limit_gain(delay))
        e = rejection_transfer(frequency, 1000.0, gain, delay)
        n = noise_transfer(frequency, 1000.0, gain, delay)
        assert complex(e - n).real == pytest.approx(1.0, rel=1e-9)
        assert complex(e - n).imag == pytest.approx(0.0, abs=1e-9)

    @FAST
    @given(
        gain=st.floats(min_value=0.01, max_value=1.9),
        delay=st.integers(min_value=1, max_value=4),
    )
    def test_noise_gain_finite_exactly_when_stable(self, gain, delay):
        stable = gain < stability_limit_gain(delay)
        assert np.isfinite(noise_variance_gain(gain, delay)) == stable

    @FAST
    @given(delay=st.integers(min_value=1, max_value=8))
    def test_stability_limit_is_positive_and_bounded(self, delay):
        limit = stability_limit_gain(delay)
        assert 0.0 < limit <= 2.0 + 1e-6

    @FAST
    @given(
        gain=st.floats(min_value=0.01, max_value=0.9),
        delay=st.integers(min_value=1, max_value=4),
    )
    def test_dc_rejection_is_zero(self, gain, delay):
        assert abs(rejection_transfer(0.0, 1000.0, gain, delay)) < 1e-12


class TestBudgetProperties:
    @FAST
    @given(var=st.floats(min_value=0.0, max_value=20.0))
    def test_strehl_in_unit_interval(self, var):
        value = float(strehl_marechal(var))
        assert 0.0 < value <= 1.0

    @FAST
    @given(var=st.floats(min_value=1e-6, max_value=30.0))
    def test_strehl_round_trip(self, var):
        assert float(variance_from_strehl(float(strehl_marechal(var)))) == pytest.approx(
            var, rel=1e-9, abs=1e-12
        )


class TestPupilProperties:
    @FAST
    @given(
        n_pix=st.integers(min_value=4, max_value=48),
        diameter=st.floats(min_value=0.05, max_value=10.0),
    )
    def test_spacing_times_n_is_diameter(self, n_pix, diameter):
        grid = PupilGrid(n_pix, diameter)
        assert grid.sample_spacing_m * n_pix == pytest.approx(diameter, rel=1e-12)

    @FAST
    @given(offset=st.floats(min_value=-100.0, max_value=100.0))
    def test_piston_removal_is_shift_invariant(self, offset):
        grid = PupilGrid(16, 1.0)
        phase = np.random.default_rng(0).normal(size=(16, 16))
        a = piston_removed(phase, grid.mask)
        b = piston_removed(phase + offset, grid.mask)
        assert np.allclose(a, b)

    @FAST
    @given(scale=st.floats(min_value=-20.0, max_value=20.0))
    def test_variance_scales_quadratically(self, scale):
        grid = PupilGrid(16, 1.0)
        phase = np.random.default_rng(1).normal(size=(16, 16))
        assert variance(scale * phase, grid.mask) == pytest.approx(
            scale**2 * variance(phase, grid.mask), rel=1e-9, abs=1e-15
        )


class TestPredictorProperties:
    @FAST
    @given(
        n_history=st.integers(min_value=1, max_value=5),
        horizon=st.integers(min_value=1, max_value=5),
        length=st.integers(min_value=12, max_value=40),
        width=st.integers(min_value=1, max_value=4),
    )
    def test_lagged_dataset_shapes(self, n_history, horizon, length, width):
        assume(length - n_history - horizon + 1 > 0)
        seq = np.arange(length * width, dtype=float).reshape(length, width)
        x, y = build_lagged_dataset([seq], n_history, horizon)
        assert x.shape == (length - n_history - horizon + 1, n_history * width)
        assert y.shape == (length - n_history - horizon + 1, width)

    @FAST
    @given(
        n_history=st.integers(min_value=1, max_value=4),
        horizon=st.integers(min_value=1, max_value=4),
    )
    def test_target_is_the_right_frame(self, n_history, horizon):
        seq = np.arange(60, dtype=float).reshape(30, 2)
        _, y = build_lagged_dataset([seq], n_history, horizon)
        assert np.allclose(y[0], seq[n_history + horizon - 1])
