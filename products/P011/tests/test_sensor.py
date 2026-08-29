"""Tests for waveforge.sensor."""

from __future__ import annotations

import numpy as np
import pytest

from waveforge.pupil import PupilGrid
from waveforge.sensor import ShackHartmann


@pytest.fixture(scope="module")
def sensor():
    return ShackHartmann(PupilGrid(32, 0.5), n_sub=4)


class TestGeometry:
    def test_subaperture_size(self, sensor):
        # d_sub = D / n_sub = 0.5 / 4 = 0.125 m exactly
        assert sensor.subaperture_size_m == pytest.approx(0.125)

    def test_valid_subaperture_count(self, sensor):
        # 4x4 grid on a circle inscribed in the square: all four corner
        # subapertures fall below 50 % fill, leaving 12
        assert sensor.n_valid == 12
        assert sensor.n_slopes == 24

    def test_operator_shape(self, sensor):
        assert sensor.operator.shape == (sensor.n_slopes, sensor.pupil.n_valid)

    def test_valid_mask_shape(self, sensor):
        assert sensor.valid_subapertures.shape == (4, 4)

    def test_corner_subapertures_excluded(self, sensor):
        valid = sensor.valid_subapertures
        assert not valid[0, 0] and not valid[0, -1] and not valid[-1, 0] and not valid[-1, -1]

    def test_higher_fill_threshold_keeps_fewer(self):
        strict = ShackHartmann(PupilGrid(32, 0.5), n_sub=4, fill_threshold=0.99)
        assert strict.n_valid < 12

    def test_divisibility_enforced(self):
        with pytest.raises(ValueError, match="divisible"):
            ShackHartmann(PupilGrid(32, 0.5), n_sub=5)

    @pytest.mark.parametrize("n_sub", [1, 0, 4.5])
    def test_bad_n_sub(self, n_sub):
        with pytest.raises(ValueError, match="n_sub"):
            ShackHartmann(PupilGrid(32, 0.5), n_sub=n_sub)

    @pytest.mark.parametrize("threshold", [0.0, -0.1, 1.5])
    def test_bad_fill_threshold(self, threshold):
        with pytest.raises(ValueError, match="fill_threshold"):
            ShackHartmann(PupilGrid(32, 0.5), n_sub=4, fill_threshold=threshold)

    def test_impossible_threshold_raises(self):
        # A pupil that is almost entirely obscured leaves no subaperture above
        # a 50 % fill, and the sensor must say so instead of building an
        # empty operator.
        with pytest.raises(ValueError, match="fill threshold"):
            ShackHartmann(PupilGrid(32, 0.5, obscuration=0.95), n_sub=4)


class TestKnownAnswers:
    def test_uniform_tilt_gives_uniform_slope(self, sensor):
        # A wavefront phi = g x has d(phi)/dx = g everywhere, so every
        # subaperture must report exactly g in x and 0 in y.
        g = 3.0
        x, _ = sensor.pupil.coords_m()
        slopes = sensor.true_slopes(g * x)
        assert np.allclose(slopes[: sensor.n_valid], g, atol=1e-12)
        assert np.allclose(slopes[sensor.n_valid :], 0.0, atol=1e-12)

    def test_uniform_tilt_in_y(self, sensor):
        g = -1.75
        _, y = sensor.pupil.coords_m()
        slopes = sensor.true_slopes(g * y)
        assert np.allclose(slopes[sensor.n_valid :], g, atol=1e-12)
        assert np.allclose(slopes[: sensor.n_valid], 0.0, atol=1e-12)

    def test_flat_wavefront_gives_zero(self, sensor):
        assert np.allclose(sensor.true_slopes(np.zeros((32, 32))), 0.0)

    def test_piston_gives_zero(self, sensor):
        assert np.allclose(sensor.true_slopes(np.full((32, 32), 4.2)), 0.0)

    def test_linearity(self, sensor, rng):
        a = rng.normal(size=(32, 32))
        b = rng.normal(size=(32, 32))
        assert np.allclose(
            sensor.true_slopes(2.0 * a - 3.0 * b),
            2.0 * sensor.true_slopes(a) - 3.0 * sensor.true_slopes(b),
        )

    def test_wrong_phase_shape(self, sensor):
        with pytest.raises(ValueError, match="phase shape"):
            sensor.true_slopes(np.zeros((16, 16)))


class TestNoiseModel:
    def test_noiseless_by_default(self, sensor):
        assert sensor.slope_noise_sigma() == 0.0
        assert sensor.slope_noise_variance_lambda_over_d() == (0.0, 0.0)

    def test_photon_term_known_answer(self):
        # sigma^2_photon = (pi^2/2)/N_ph * (X_T/X_D)^2; N_ph = 100, X_T = X_D
        # -> 9.8696/2/100 = 0.049348 (lambda/d)^2
        s = ShackHartmann(PupilGrid(32, 0.5), n_sub=4, photon_flux=100.0)
        photon, read = s.slope_noise_variance_lambda_over_d()
        assert photon == pytest.approx(np.pi**2 / 2 / 100.0)
        assert read == 0.0

    def test_read_term_known_answer(self):
        # sigma^2_read = (pi^2/3)(sigma_e^2/N^2)(X_S^2/X_D)^2
        # N = 100, sigma_e = 3, X_S = 6, X_D = 2
        # = 3.28987 * 9/10000 * (36/2)^2 = 3.28987 * 9e-4 * 324 = 0.95932
        s = ShackHartmann(
            PupilGrid(32, 0.5), n_sub=4, photon_flux=100.0, read_noise_e=3.0, pixels_per_sub=6
        )
        _, read = s.slope_noise_variance_lambda_over_d()
        assert read == pytest.approx(0.959324, rel=1e-5)

    def test_sigma_scales_as_inverse_sqrt_flux(self):
        a = ShackHartmann(PupilGrid(32, 0.5), n_sub=4, photon_flux=100.0).slope_noise_sigma()
        b = ShackHartmann(PupilGrid(32, 0.5), n_sub=4, photon_flux=400.0).slope_noise_sigma()
        assert a / b == pytest.approx(2.0, rel=1e-12)

    def test_read_noise_increases_sigma(self):
        clean = ShackHartmann(PupilGrid(32, 0.5), n_sub=4, photon_flux=100.0).slope_noise_sigma()
        noisy = ShackHartmann(
            PupilGrid(32, 0.5), n_sub=4, photon_flux=100.0, read_noise_e=2.0
        ).slope_noise_sigma()
        assert noisy > clean

    def test_noise_equivalent_angle_conversion(self):
        s = ShackHartmann(PupilGrid(32, 0.5), n_sub=4, photon_flux=100.0, wavelength_m=1e-6)
        assert s.noise_equivalent_angle_rad() == pytest.approx(
            s.slope_noise_sigma() * 1e-6 / (2 * np.pi)
        )

    def test_elongated_spot_increases_photon_noise(self):
        base = ShackHartmann(PupilGrid(32, 0.5), n_sub=4, photon_flux=100.0)
        wide = ShackHartmann(
            PupilGrid(32, 0.5), n_sub=4, photon_flux=100.0, spot_fwhm_pixels=4.0
        )
        assert wide.slope_noise_sigma() == pytest.approx(2.0 * base.slope_noise_sigma())

    @pytest.mark.parametrize("flux", [0.0, -1.0])
    def test_bad_flux(self, flux):
        with pytest.raises(ValueError, match="photon_flux"):
            ShackHartmann(PupilGrid(32, 0.5), n_sub=4, photon_flux=flux)

    def test_bad_read_noise(self):
        with pytest.raises(ValueError, match="read_noise_e"):
            ShackHartmann(PupilGrid(32, 0.5), n_sub=4, read_noise_e=-1.0)

    def test_bad_pixels_per_sub(self):
        with pytest.raises(ValueError, match="pixels_per_sub"):
            ShackHartmann(PupilGrid(32, 0.5), n_sub=4, pixels_per_sub=1)

    @pytest.mark.parametrize("name", ["spot_fwhm_pixels", "diffraction_fwhm_pixels"])
    def test_bad_spot_sizes(self, name):
        with pytest.raises(ValueError, match=name):
            ShackHartmann(PupilGrid(32, 0.5), n_sub=4, **{name: 0.0})

    def test_bad_wavelength(self):
        with pytest.raises(ValueError, match="wavelength_m"):
            ShackHartmann(PupilGrid(32, 0.5), n_sub=4, wavelength_m=0.0)

    def test_bad_dropout(self):
        with pytest.raises(ValueError, match="dropout_probability"):
            ShackHartmann(PupilGrid(32, 0.5), n_sub=4, dropout_probability=1.0)


class TestMeasurement:
    def test_noiseless_measurement_matches_true_slopes(self, sensor, rng):
        phase = rng.normal(size=(32, 32))
        m = sensor.measure(phase, 0)
        assert np.allclose(m.slopes, sensor.true_slopes(phase))
        assert m.valid.all()
        assert m.noise_sigma == 0.0

    def test_noise_is_reproducible(self, rng):
        s = ShackHartmann(PupilGrid(32, 0.5), n_sub=4, photon_flux=50.0)
        phase = rng.normal(size=(32, 32))
        assert np.allclose(s.measure(phase, 5).slopes, s.measure(phase, 5).slopes)

    def test_noise_changes_with_seed(self, rng):
        s = ShackHartmann(PupilGrid(32, 0.5), n_sub=4, photon_flux=50.0)
        phase = rng.normal(size=(32, 32))
        assert not np.allclose(s.measure(phase, 5).slopes, s.measure(phase, 6).slopes)

    def test_noise_statistics(self):
        s = ShackHartmann(PupilGrid(32, 0.5), n_sub=4, photon_flux=50.0)
        gen = np.random.default_rng(0)
        residuals = np.stack(
            [s.measure(np.zeros((32, 32)), gen).slopes for _ in range(400)]
        )
        assert residuals.std() == pytest.approx(s.slope_noise_sigma(), rel=0.05)
        assert abs(residuals.mean()) < 0.2 * s.slope_noise_sigma()

    def test_dropout_flags_and_zeroes(self):
        s = ShackHartmann(PupilGrid(32, 0.5), n_sub=4, dropout_probability=0.5)
        m = s.measure(np.zeros((32, 32)), 3)
        assert m.valid.dtype == bool
        assert not m.valid.all()
        assert np.all(m.slopes[: s.n_valid][~m.valid] == 0.0)
        assert np.all(m.slopes[s.n_valid :][~m.valid] == 0.0)

    def test_dropout_rate_is_about_right(self):
        s = ShackHartmann(PupilGrid(32, 0.5), n_sub=4, dropout_probability=0.3)
        gen = np.random.default_rng(1)
        flags = np.stack([s.measure(np.zeros((32, 32)), gen).valid for _ in range(300)])
        assert 1.0 - flags.mean() == pytest.approx(0.3, abs=0.03)

    def test_generator_object_accepted(self, sensor):
        gen = np.random.default_rng(2)
        assert sensor.measure(np.zeros((32, 32)), gen).slopes.shape == (sensor.n_slopes,)
