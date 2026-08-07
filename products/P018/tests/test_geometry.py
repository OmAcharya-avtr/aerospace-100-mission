"""Geometry: hand-calculated known answers, input validation, property tests."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from shacksim import AIRY_FWHM_COEFF, LensletArray


class TestDerivedQuantities:
    def test_pixel_size_hand_calculation(self):
        # pitch 500 um / 16 px = 31.25 um exactly.
        a = LensletArray(pitch=500e-6, pixels_per_sub=16)
        assert a.pixel_size == pytest.approx(31.25e-6, rel=1e-12)

    def test_pixel_angle_hand_calculation(self):
        # theta_pix = p / f = 31.25e-6 m / 50e-3 m = 6.25e-4 rad exactly.
        a = LensletArray(pitch=500e-6, pixels_per_sub=16, focal_length=50e-3)
        assert a.pixel_angle == pytest.approx(6.25e-4, rel=1e-12)

    def test_spot_fwhm_hand_calculation(self):
        # FWHM = 1.0287938 * lambda * f / d
        #      = 1.0287938 * 633e-9 * 50e-3 / 500e-6 = 6.51226e-05 m.
        a = LensletArray(wavelength=633e-9, focal_length=50e-3, pitch=500e-6)
        expected = AIRY_FWHM_COEFF * 633e-9 * 50e-3 / 500e-6
        assert a.spot_fwhm == pytest.approx(expected, rel=1e-12)
        assert a.spot_fwhm == pytest.approx(6.512265e-5, rel=1e-6)
        # ... and 6.512265e-5 / 31.25e-6 = 2.08392 px.
        assert a.spot_fwhm_px == pytest.approx(2.083925, rel=1e-6)

    def test_spot_sigma_matches_fwhm_conversion(self):
        a = LensletArray()
        assert a.spot_sigma_px == pytest.approx(a.spot_fwhm_px / 2.354820045, rel=1e-9)

    def test_spot_size_scales_with_wavelength_and_fnumber(self):
        base = LensletArray()
        assert LensletArray(wavelength=2 * base.wavelength).spot_fwhm == pytest.approx(
            2 * base.spot_fwhm, rel=1e-12
        )
        assert LensletArray(focal_length=2 * base.focal_length).spot_fwhm == pytest.approx(
            2 * base.spot_fwhm, rel=1e-12
        )
        assert LensletArray(pitch=2 * base.pitch).spot_fwhm == pytest.approx(
            0.5 * base.spot_fwhm, rel=1e-12
        )

    def test_max_slope_hand_calculation(self):
        # (16/2) px * 6.25e-4 rad/px = 5.0e-3 rad.
        assert LensletArray().max_slope == pytest.approx(5.0e-3, rel=1e-12)

    def test_image_size(self):
        assert LensletArray(n_lenslets=8, pixels_per_sub=16).image_size == 128


class TestSlopeConversion:
    def test_known_answer(self):
        # g = 1e-3 rad, f = 50 mm, p = 31.25 um -> d = 1e-3 * 50e-3 / 31.25e-6 = 1.6 px.
        a = LensletArray()
        assert float(a.slope_to_displacement(1.0e-3)) == pytest.approx(1.6, rel=1e-12)
        assert float(a.displacement_to_slope(1.6)) == pytest.approx(1.0e-3, rel=1e-12)

    @settings(max_examples=60, deadline=None)
    @given(st.floats(-1e-2, 1e-2, allow_nan=False, allow_infinity=False))
    def test_round_trip_is_identity(self, slope: float):
        a = LensletArray()
        back = a.displacement_to_slope(a.slope_to_displacement(slope))
        assert float(back) == pytest.approx(slope, abs=1e-18, rel=1e-12)

    def test_conversion_is_linear(self):
        a = LensletArray()
        s = np.array([1e-4, -3e-4])
        assert np.allclose(a.slope_to_displacement(2 * s), 2 * a.slope_to_displacement(s))


class TestPupilMask:
    def test_default_mask_counts(self):
        # 8x8 array, pupil inscribed: the four corner lenslets are outside.
        a = LensletArray(n_lenslets=8)
        mask = a.valid_mask()
        assert mask.shape == (8, 8)
        assert a.n_valid == 52
        assert not mask[0, 0] and not mask[0, 7] and not mask[7, 0] and not mask[7, 7]
        assert mask[3, 3] and mask[4, 4]

    def test_mask_is_symmetric(self):
        mask = LensletArray(n_lenslets=8).valid_mask()
        assert np.array_equal(mask, mask[::-1, :])
        assert np.array_equal(mask, mask[:, ::-1])
        assert np.array_equal(mask, mask.T)

    def test_obscuration_removes_central_subapertures(self):
        full = LensletArray(n_lenslets=8).n_valid
        obscured = LensletArray(n_lenslets=8, obscuration=0.4).n_valid
        assert obscured < full

    def test_larger_pupil_admits_more_subapertures(self):
        a = LensletArray(n_lenslets=8, pupil_diameter=8 * 500e-6 * 1.45)
        assert a.n_valid == 64

    def test_centres_are_centred_and_spaced_by_pitch(self):
        a = LensletArray(n_lenslets=4, pitch=1e-3)
        c = a.subaperture_centres()
        assert c.shape == (4, 4, 2)
        assert np.allclose(c.reshape(-1, 2).mean(axis=0), 0.0, atol=1e-15)
        assert c[0, 1, 0] - c[0, 0, 0] == pytest.approx(1e-3, rel=1e-12)
        assert c[1, 0, 1] - c[0, 0, 1] == pytest.approx(1e-3, rel=1e-12)

    def test_valid_centres_matches_mask_order(self):
        a = LensletArray(n_lenslets=6)
        assert a.valid_centres().shape == (a.n_valid, 2)

    def test_subaperture_slice(self):
        a = LensletArray(n_lenslets=4, pixels_per_sub=8)
        rs, cs = a.subaperture_slice(2, 1)
        assert (rs.start, rs.stop) == (16, 24)
        assert (cs.start, cs.stop) == (8, 16)


class TestValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"n_lenslets": 1},
            {"pixels_per_sub": 3},
            {"pitch": 0.0},
            {"pitch": -1e-3},
            {"focal_length": -1.0},
            {"wavelength": 0.0},
            {"obscuration": 1.0},
            {"obscuration": -0.1},
            {"fill_threshold": 0.0},
            {"fill_threshold": 1.5},
            {"pupil_diameter": 0.0},
            {"pitch": float("nan")},
        ],
    )
    def test_invalid_values_raise_value_error(self, kwargs):
        with pytest.raises(ValueError):
            LensletArray(**kwargs)

    @pytest.mark.parametrize("kwargs", [{"n_lenslets": 8.0}, {"pixels_per_sub": "16"}])
    def test_wrong_types_raise_type_error(self, kwargs):
        with pytest.raises(TypeError):
            LensletArray(**kwargs)

    def test_slice_out_of_range(self):
        with pytest.raises(ValueError, match="outside"):
            LensletArray(n_lenslets=4).subaperture_slice(4, 0)

    def test_summary_keys(self):
        s = LensletArray().summary()
        assert set(s) == {
            "diameter_m", "pixel_size_m", "pixel_angle_rad", "spot_fwhm_m",
            "spot_fwhm_px", "spot_sigma_px", "max_slope_rad", "n_valid",
        }
        assert all(np.isfinite(v) for v in s.values())
