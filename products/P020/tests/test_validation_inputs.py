"""Input-validation tests: every invalid input must raise, with a useful message."""

import math

import numpy as np
import pytest

from atmoprofile import (
    bufton_wind,
    constant_profile,
    fried_parameter,
    greenwood_frequency,
    hv57,
    isoplanatic_angle,
    rytov_variance,
    tabulated_profile,
    tabulated_wind,
    turbulence_moment,
    weighted_integral,
)
from atmoprofile._validate import PLANE_PARALLEL_WARN_DEG

LAM = 500e-9
HV = hv57()


class TestZenithAngle:
    def test_ninety_degrees_rejected(self):
        with pytest.raises(ValueError, match="pi/2"):
            fried_parameter(HV, LAM, zenith_rad=math.pi / 2)

    def test_beyond_ninety_degrees_rejected(self):
        # 100 deg is below the horizon: sec(zeta) is negative there and the
        # plane-parallel model is meaningless.
        with pytest.raises(ValueError, match="pi/2"):
            isoplanatic_angle(HV, LAM, zenith_rad=math.radians(100.0))

    def test_negative_zenith_rejected(self):
        with pytest.raises(ValueError, match=">= 0"):
            rytov_variance(HV, LAM, zenith_rad=-0.1)

    def test_non_finite_zenith_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            fried_parameter(HV, LAM, zenith_rad=float("nan"))
        with pytest.raises(ValueError, match="finite"):
            fried_parameter(HV, LAM, zenith_rad=float("inf"))

    def test_large_zenith_warns_about_plane_parallel_model(self):
        with pytest.warns(UserWarning, match="flat-Earth"):
            fried_parameter(HV, LAM, zenith_rad=math.radians(PLANE_PARALLEL_WARN_DEG + 5.0))

    def test_sixty_degrees_does_not_warn(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fried_parameter(HV, LAM, zenith_rad=math.radians(PLANE_PARALLEL_WARN_DEG))


class TestWavelength:
    def test_zero_wavelength_rejected(self):
        with pytest.raises(ValueError, match="> 0"):
            fried_parameter(HV, 0.0)

    def test_negative_wavelength_rejected(self):
        with pytest.raises(ValueError, match="> 0"):
            fried_parameter(HV, -500e-9)

    def test_radio_wavelength_rejected(self):
        # 3 cm (X band): the visible/IR Kolmogorov coefficients do not apply.
        with pytest.raises(ValueError, match="optical/IR band"):
            fried_parameter(HV, 0.03)

    def test_x_ray_wavelength_rejected(self):
        with pytest.raises(ValueError, match="optical/IR band"):
            fried_parameter(HV, 1e-10)


class TestAltitudes:
    def test_negative_ground_altitude_rejected(self):
        with pytest.raises(ValueError, match="negative altitude"):
            fried_parameter(HV, LAM, h_ground=-100.0)

    def test_top_below_ground_rejected(self):
        with pytest.raises(ValueError, match="strictly greater"):
            fried_parameter(HV, LAM, h_ground=5000.0, h_top=1000.0)

    def test_equal_limits_rejected(self):
        with pytest.raises(ValueError, match="strictly greater"):
            turbulence_moment(HV, 0.0, h_ground=1000.0, h_top=1000.0)

    def test_range_outside_profile_support_rejected(self):
        with pytest.raises(ValueError, match="validity range"):
            fried_parameter(HV, LAM, h_top=50_000.0)

    def test_profile_evaluated_outside_support_rejected(self):
        with pytest.raises(ValueError, match="validity range"):
            HV(25_000.0)

    def test_non_finite_altitude_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            HV(float("nan"))


class TestTabulatedProfile:
    def test_non_monotonic_heights_rejected(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            tabulated_profile([0.0, 1000.0, 500.0, 2000.0], [1e-14, 1e-15, 1e-16, 1e-17])

    def test_duplicate_heights_rejected(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            tabulated_profile([0.0, 1000.0, 1000.0], [1e-14, 1e-15, 1e-16])

    def test_negative_height_rejected(self):
        with pytest.raises(ValueError, match=">= 0 m"):
            tabulated_profile([-10.0, 1000.0], [1e-14, 1e-15])

    def test_negative_cn2_rejected(self):
        with pytest.raises(ValueError, match="must be > 0"):
            tabulated_profile([0.0, 1000.0], [1e-14, -1e-15])

    def test_zero_cn2_rejected(self):
        with pytest.raises(ValueError, match="must be > 0"):
            tabulated_profile([0.0, 1000.0], [1e-14, 0.0])

    def test_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            tabulated_profile([0.0, 1000.0, 2000.0], [1e-14, 1e-15])

    def test_single_sample_rejected(self):
        with pytest.raises(ValueError, match="at least 2 samples"):
            tabulated_profile([0.0], [1e-14])

    def test_nan_sample_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            tabulated_profile([0.0, 1000.0], [1e-14, float("nan")])


class TestProfileConstructors:
    def test_constant_profile_rejects_negative_cn2(self):
        with pytest.raises(ValueError, match="> 0"):
            constant_profile(-1e-15)

    def test_constant_profile_rejects_inverted_limits(self):
        with pytest.raises(ValueError, match="must exceed"):
            constant_profile(1e-15, 2000.0, 1000.0)

    def test_hufnagel_valley_rejects_negative_wind(self):
        from atmoprofile import hufnagel_valley

        with pytest.raises(ValueError, match="> 0"):
            hufnagel_valley(-21.0)


class TestWindValidation:
    def test_tabulated_wind_non_monotonic_rejected(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            tabulated_wind([0.0, 5000.0, 2000.0], [5.0, 20.0, 10.0])

    def test_tabulated_wind_negative_speed_rejected(self):
        with pytest.raises(ValueError, match=">= 0"):
            tabulated_wind([0.0, 5000.0], [5.0, -20.0])

    def test_wind_support_shorter_than_path_rejected(self):
        short_wind = tabulated_wind([0.0, 5000.0], [5.0, 20.0])
        with pytest.raises(ValueError, match="exceeds the support"):
            greenwood_frequency(HV, short_wind, LAM)

    def test_wind_must_be_a_wind_profile(self):
        with pytest.raises(TypeError, match="WindProfile"):
            greenwood_frequency(HV, HV, LAM)


class TestArgumentChoices:
    def test_unknown_wave_kind_rejected(self):
        with pytest.raises(ValueError, match="wave must be one of"):
            fried_parameter(HV, LAM, wave="gaussian-beam")

    def test_unknown_path_direction_rejected(self):
        with pytest.raises(ValueError, match="path must be one of"):
            fried_parameter(HV, LAM, wave="spherical", path="sideways")

    def test_unknown_integration_method_rejected(self):
        with pytest.raises(ValueError, match="method must be one of"):
            turbulence_moment(HV, 0.0, method="romberg")

    def test_profile_type_checked(self):
        with pytest.raises(TypeError, match="Cn2Profile"):
            weighted_integral(np.array([1.0, 2.0]))

    def test_too_few_simpson_nodes_rejected(self):
        with pytest.raises(ValueError, match="n_nodes must be >= 3"):
            turbulence_moment(HV, 0.0, method="simpson", n_nodes=2)

    def test_unknown_standard_profile_rejected(self):
        from atmoprofile import standard_profile

        with pytest.raises(ValueError, match="unknown profile"):
            standard_profile("hv_5_7")


class TestUndefinedResults:
    def test_greenwood_needs_a_wind_profile(self):
        # Zero wind gives f_G = 0 exactly - not an error, but it must not be
        # silently non-zero.
        from atmoprofile import constant_wind

        assert greenwood_frequency(HV, constant_wind(0.0), LAM) == 0.0

    def test_bufton_wind_rejects_negative_ground_wind(self):
        with pytest.raises(ValueError, match=">= 0"):
            bufton_wind(-5.0)
