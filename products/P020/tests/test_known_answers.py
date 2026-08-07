"""Known-answer tests: every expected value is hand-computed in the comments.

The reference case is a homogeneous slab, Cn^2 = 1e-15 m^(-2/3) from 0 to
1000 m, observed at lambda = 500 nm from the ground at zenith, for which every
integral in the package has a closed form.

Shared arithmetic used throughout this file:

    k      = 2*pi / 500e-9      = 1.2566370614e7  rad/m
    k^2                          = 1.5791367042e14 m^-2
    k^(7/6)                      = 1.9160706038e8  m^(-7/6)
    mu_0     = 1e-15 * 1000                      = 1.0e-12      m^(1/3)
    mu_(5/3) = 1e-15 * (3/8) * 1000^(8/3)        = 3.75e-8      m^2
             [ int_0^H h^(5/3) dh = H^(8/3)/(8/3) = (3/8) H^(8/3);
               1000^(8/3) = 10^8 ]
    mu_(5/6) = 1e-15 * (6/11) * 1000^(11/6)      = 1.7248787e-10 m^(7/6)
             [ int_0^H h^(5/6) dh = (6/11) H^(11/6); 1000^(11/6) = 10^5.5
               = 3.16227766e5 ]
"""

import math
import warnings

import numpy as np
import pytest
from scipy.special import beta

from atmoprofile import (
    C_FRIED,
    C_ISOPLANATIC,
    C_RYTOV,
    C_THETA0_OVER_R0,
    bufton_wind,
    constant_profile,
    constant_wind,
    effective_turbulence_height,
    fried_parameter,
    greenwood_frequency,
    hv57,
    isoplanatic_angle,
    rytov_variance,
    scintillation_index,
    slc_day,
    slc_night,
    turbulence_moment,
)
from atmoprofile.constants import C_FRIED_DEFINITION, C_GREENWOOD_LAMBDA, C_STRUCTURE_PLANE

LAM = 500e-9
SLAB = constant_profile(1e-15, 0.0, 1000.0)


class TestConstantSlabClosedForms:
    def test_moment_zero(self):
        # mu_0 = Cn^2 * H = 1e-15 * 1000 = 1.0e-12 m^(1/3)
        assert turbulence_moment(SLAB, 0.0) == pytest.approx(1.0e-12, rel=1e-12)

    def test_moment_five_thirds(self):
        # mu_(5/3) = 1e-15 * (3/8) * 1000^(8/3) = 1e-15 * 0.375 * 1e8 = 3.75e-8
        assert turbulence_moment(SLAB, 5.0 / 3.0) == pytest.approx(3.75e-8, rel=1e-10)

    def test_moment_five_sixths(self):
        # mu_(5/6) = 1e-15 * (6/11) * 3.16227766e5 = 1.72487872e-10
        assert turbulence_moment(SLAB, 5.0 / 6.0) == pytest.approx(1.72487872e-10, rel=1e-8)

    def test_fried_parameter_hand_value(self):
        # r0 = [0.423 * k^2 * mu_0]^(-3/5)
        #    = [0.423 * 1.5791367042e14 * 1.0e-12]^(-3/5)
        #    = [66.79748]^(-0.6)
        #    ln(66.79748) = 4.201655 ; * 0.6 = 2.520993 ; exp = 12.44103
        #    => r0 = 1/12.44103 = 0.0803792 m  (8.03792 cm)
        assert fried_parameter(SLAB, LAM) == pytest.approx(0.0803792, rel=1e-6)

    def test_isoplanatic_angle_hand_value(self):
        # theta0 = [2.914 * k^2 * mu_(5/3)]^(-3/5)
        #        = [2.914 * 1.5791367042e14 * 3.75e-8]^(-3/5)
        #        = [1.7256130e7]^(-0.6)
        #    ln(1.7256130e7) = 16.663645 ; * 0.6 = 9.998187 ; exp = 21957.3
        #    => theta0 = 4.54816e-5 rad = 45.4816 urad
        assert isoplanatic_angle(SLAB, LAM) == pytest.approx(4.54816e-5, rel=1e-5)

    def test_effective_height_hand_value(self):
        # h_bar = [mu_(5/3)/mu_0]^(3/5) = [3.75e-8 / 1e-12]^(0.6) = 37500^0.6
        #    ln(37500) = 10.532263 ; * 0.6 = 6.319358 ; exp = 555.16 m
        assert effective_turbulence_height(SLAB) == pytest.approx(555.16, rel=1e-4)

    def test_rytov_plane_hand_value(self):
        # sigma_R^2 = 2.25 * k^(7/6) * mu_(5/6)
        #           = 2.25 * 1.9160706e8 * 1.72487872e-10
        #           = 2.25 * 0.03304990 = 0.07436228
        assert rytov_variance(SLAB, LAM) == pytest.approx(0.0743623, rel=1e-6)

    def test_rytov_spherical_hand_value(self):
        # Spherical weight: int_0^L u^(5/6) (1-u/L)^(5/6) du = L^(11/6) B(11/6,11/6)
        # B(11/6,11/6) = Gamma(11/6)^2 / Gamma(11/3) = 0.22053566
        # mu = 1e-15 * 0.22053566 * 3.16227766e5 = 6.973668e-11
        # sigma^2 = 2.25 * 1.9160706e8 * 6.973668e-11 = 0.0300658
        assert rytov_variance(SLAB, LAM, wave="spherical") == pytest.approx(0.0300658, rel=1e-5)

    def test_spherical_beta_function_identity(self):
        # The ratio spherical/plane must equal B(11/6,11/6)/(6/11) = 0.4043154,
        # the classic "spherical wave scintillates 0.4x as much as a plane wave".
        ratio = rytov_variance(SLAB, LAM, wave="spherical") / rytov_variance(SLAB, LAM)
        assert ratio == pytest.approx(float(beta(11 / 6, 11 / 6)) / (6 / 11), rel=1e-9)
        assert ratio == pytest.approx(0.4043154, rel=1e-6)

    def test_fried_spherical_hand_value(self):
        # Spherical weight int_0^L (1-u/L)^(5/3) du = (3/8) L, so
        # mu_sph = 1e-15 * 0.375 * 1000 = 3.75e-13 and
        # r0 = [0.423 * 1.5791367e14 * 3.75e-13]^(-0.6) = [25.04905]^(-0.6)
        #    ln = 3.220889 ; *0.6 = 1.932533 ; exp = 6.906528 => 0.1447856 m
        assert fried_parameter(SLAB, LAM, wave="spherical") == pytest.approx(0.1447856, rel=1e-6)

    def test_spherical_uplink_equals_downlink_for_uniform_slab(self):
        # For a homogeneous slab the weights (u/L)^(5/3) and (1-u/L)^(5/3)
        # integrate to the same 3L/8, so the two geometries coincide.  They must
        # NOT coincide for a real (non-uniform) profile - see the HV test below.
        down = fried_parameter(SLAB, LAM, wave="spherical", path="downlink")
        up = fried_parameter(SLAB, LAM, wave="spherical", path="uplink")
        assert down == pytest.approx(up, rel=1e-9)


class TestHorizontalPathTextbookForms:
    """The slant-path constants must reproduce the textbook homogeneous forms."""

    def test_plane_wave_coefficient_is_1p23(self):
        # 2.25 * (6/11) = 1.227272..., i.e. sigma_R^2 = 1.23 Cn^2 k^(7/6) L^(11/6)
        # (Andrews & Phillips 2005).  Check the package value against the
        # textbook coefficient to within the rounding of "1.23".
        k = 2 * math.pi / LAM
        textbook = 1.23 * 1e-15 * k ** (7 / 6) * 1000 ** (11 / 6)
        assert rytov_variance(SLAB, LAM) == pytest.approx(textbook, rel=2.3e-3)
        assert C_RYTOV * (6 / 11) == pytest.approx(1.2273, rel=1e-4)

    def test_spherical_wave_coefficient_is_0p5(self):
        # 2.25 * B(11/6,11/6) = 0.4962 ~ the textbook 0.5 Cn^2 k^(7/6) L^(11/6)
        assert C_RYTOV * float(beta(11 / 6, 11 / 6)) == pytest.approx(0.4962, rel=1e-3)


class TestCoefficientDerivations:
    def test_fried_definition_constant_is_6p88(self):
        # 2 * (24/5 * Gamma(6/5))^(5/6): Gamma(1.2) = 0.9181687,
        # 4.8 * 0.9181687 = 4.4072097 ; ^(5/6): ln = 1.483241, *5/6 = 1.236034,
        # exp = 3.441939 ; * 2 = 6.883877
        assert C_FRIED_DEFINITION == pytest.approx(6.883877, rel=1e-6)

    def test_fried_coefficient_ratio(self):
        # 2.914 / 6.883877 = 0.4233080, rounded to 0.423 in the literature
        assert C_STRUCTURE_PLANE / C_FRIED_DEFINITION == pytest.approx(0.4233080, rel=1e-6)
        assert C_FRIED == 0.423

    def test_theta0_over_r0_constant_is_0p314(self):
        # (0.423 / 2.914)^(3/5): 0.423/2.914 = 0.1451613 ;
        # ln = -1.929889 ; *0.6 = -1.157933 ; exp = 0.3141308
        assert C_THETA0_OVER_R0 == pytest.approx(0.3141308, rel=1e-6)

    def test_greenwood_lambda_form_constant_is_2p31(self):
        # 0.102^(3/5) * (2 pi)^(6/5) = 0.2541913 * 9.0743 = 2.30662
        assert C_GREENWOOD_LAMBDA == pytest.approx(2.30662, rel=1e-5)

    def test_theta0_equals_0314_r0_over_hbar(self):
        # Algebraic identity implied by the two coefficients; must hold exactly
        # for any profile.
        for profile in (SLAB, hv57(), slc_day()):
            r0 = fried_parameter(profile, LAM)
            th = isoplanatic_angle(profile, LAM)
            hbar = effective_turbulence_height(profile)
            assert th == pytest.approx(C_THETA0_OVER_R0 * r0 / hbar, rel=1e-12)

    def test_isoplanatic_constant_equals_structure_constant(self):
        assert C_ISOPLANATIC == C_STRUCTURE_PLANE


class TestGreenwoodEquivalentForms:
    def test_two_published_forms_agree(self):
        # f_G = [0.102 k^2 sec(z) I]^(3/5) must equal
        # 2.3066 lambda^(-6/5) [sec(z) I]^(3/5) with I = int Cn^2 v^(5/3) dh.
        profile = hv57()
        wind = bufton_wind(5.0)
        for zen_deg in (0.0, 30.0, 55.0):
            zen = math.radians(zen_deg)
            fg = greenwood_frequency(profile, wind, LAM, zenith_rad=zen)
            # independent evaluation of the integral on a dense trapezoid grid
            h = np.linspace(0.0, 20000.0, 200001)
            integrand = np.array(profile(h)) * np.array(wind(h)) ** (5 / 3)
            integral = float(np.trapezoid(integrand, h))
            alt = C_GREENWOOD_LAMBDA * LAM ** (-6 / 5) * ((1 / math.cos(zen)) * integral) ** 0.6
            assert fg == pytest.approx(alt, rel=1e-4)

    def test_constant_wind_scales_as_v_to_the_one(self):
        # f_G ~ [int Cn^2 v^(5/3)]^(3/5): with a uniform wind this is v^(5/3 * 3/5)
        # = v^1 exactly.  Doubling the wind must double f_G.
        profile = hv57()
        f1 = greenwood_frequency(profile, constant_wind(10.0), LAM)
        f2 = greenwood_frequency(profile, constant_wind(20.0), LAM)
        assert f2 / f1 == pytest.approx(2.0, rel=1e-9)


class TestStandardProfileNamedValues:
    def test_hv57_gives_5cm_and_7urad_at_500nm(self):
        # The Hufnagel-Valley "5/7" model is named for producing r0 = 5 cm and
        # theta0 = 7 urad at 0.5 um on a vertical path (Andrews & Phillips 2005;
        # Hardy 1998).  Computed here: 4.9624 cm and 7.0109 urad.
        profile = hv57()
        r0 = fried_parameter(profile, LAM)
        th = isoplanatic_angle(profile, LAM)
        assert r0 == pytest.approx(0.05, rel=0.01)  # within 1 % of 5 cm
        assert th == pytest.approx(7e-6, rel=0.01)  # within 1 % of 7 urad
        assert r0 == pytest.approx(0.0496245, rel=1e-4)  # regression pin
        assert th == pytest.approx(7.010862e-6, rel=1e-4)  # regression pin

    def test_slc_day_is_worse_than_slc_night(self):
        # Daytime convection strengthens the surface layer, so r0(day) < r0(night).
        assert fried_parameter(slc_day(), LAM) < fried_parameter(slc_night(), LAM)

    def test_standard_profiles_r0_regression_pins(self):
        # Values produced by this build (see validation/VALIDATION.md).
        assert fried_parameter(slc_day(), LAM) == pytest.approx(0.0433900, rel=1e-4)
        assert fried_parameter(slc_night(), LAM) == pytest.approx(0.0760229, rel=1e-4)

    def test_hv57_spherical_uplink_differs_from_downlink(self):
        # Real profiles are bottom-heavy, so the uplink (source at the ground)
        # sees a much weaker effective integral than the downlink.
        profile = hv57()
        down = fried_parameter(profile, LAM, wave="spherical", path="downlink")
        up = fried_parameter(profile, LAM, wave="spherical", path="uplink")
        assert up > down * 1.5


class TestScintillation:
    def test_weak_regime_index_equals_rytov(self):
        assert scintillation_index(SLAB, LAM) == pytest.approx(rytov_variance(SLAB, LAM), rel=0)

    def test_strong_regime_warns(self):
        # A thick, strong slab drives sigma_R^2 well past 1.
        strong = constant_profile(1e-13, 0.0, 20000.0)
        with pytest.warns(UserWarning, match="Rytov variance"):
            value = scintillation_index(strong, LAM)
        assert value > 1.0

    def test_warning_can_be_suppressed(self):
        strong = constant_profile(1e-13, 0.0, 20000.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning becomes an exception
            value = scintillation_index(strong, LAM, warn_strong=False)
        assert value > 1.0
