"""Refraction: hand calculations, the plane-parallel limit, and the cross-check.

Every ``pytest.approx`` call in this file passes an explicit ``abs=``. The
default absolute tolerance of ``pytest.approx`` is 1e-12 and it is applied with
an OR against the relative bound, so a bare ``rel=`` on a small quantity is
silently replaced by the default and the assertion stops testing what it says.
``abs=0.0`` is used wherever the relative bound is the one meant.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pointmodel import refraction


class TestBennettKnownAnswers:
    """Hand calculations of equation (1)."""

    def test_forty_five_degrees(self) -> None:
        # Hand calculation. Equation (1) is R = 1 / tan(h + 7.31 / (h + 4.4))
        # in arcminutes. At h = 45: h + 4.4 = 49.4, and 7.31 / 49.4 =
        # 0.14797570850202431 degrees, so the tangent argument is
        # 45.147975708502024 degrees. In arcseconds R = 60 / tan(that), which
        # evaluates to 59.69087807141169 arcsec. The standard-conditions
        # factor of equation (2) is exactly 1 at 1010 hPa and 10 C because
        # 1010/1010 = 1 and 283/(273+10) = 1.
        argument_deg = 45.0 + 7.31 / 49.4
        assert argument_deg == pytest.approx(45.147975708502024, rel=1e-15, abs=0.0)
        expected = 60.0 / math.tan(math.radians(argument_deg))
        assert expected == pytest.approx(59.69087807141169, rel=1e-13, abs=0.0)
        got = float(refraction.bennett_refraction(45.0))
        assert got == pytest.approx(expected, rel=1e-13, abs=0.0)

    def test_thirty_degrees_has_an_exact_intermediate(self) -> None:
        # Hand calculation. At h = 30: h + 4.4 = 34.4, and 7.31 / 34.4 =
        # 0.2125 exactly, because 34.4 * 0.2125 = 7.31. So the tangent
        # argument is exactly 30.2125 degrees and R = 60 / tan(30.2125 deg) =
        # 103.03860801318639 arcsec.
        assert 7.31 / 34.4 == pytest.approx(0.2125, rel=0.0, abs=1e-15)
        expected = 60.0 / math.tan(math.radians(30.2125))
        assert expected == pytest.approx(103.03860801318639, rel=1e-13, abs=0.0)
        assert float(refraction.bennett_refraction(30.0)) == pytest.approx(
            expected, rel=1e-13, abs=0.0
        )

    def test_conditions_factor_is_a_density_ratio(self) -> None:
        # Hand calculation. Equation (2) is f = (P/1010) * (283/(273+T)). At
        # P = 505 hPa and T = 10 C, f = 0.5 * 1 = 0.5 exactly, so the
        # refraction must be exactly half the standard-conditions value in
        # exact arithmetic and to within rounding in floating point.
        standard = float(refraction.bennett_refraction(45.0, 1010.0, 10.0))
        halved = float(refraction.bennett_refraction(45.0, 505.0, 10.0))
        assert halved == pytest.approx(0.5 * standard, rel=1e-14, abs=0.0)

    def test_colder_air_refracts_more(self) -> None:
        # 283/(273+0) = 1.03663... > 1, so 0 C gives more refraction than 10 C.
        cold = float(refraction.bennett_refraction(45.0, 1010.0, 0.0))
        warm = float(refraction.bennett_refraction(45.0, 1010.0, 10.0))
        assert cold > warm
        assert cold / warm == pytest.approx(283.0 / 273.0, rel=1e-14, abs=0.0)


class TestBennettValidity:
    """The declared validity range, including the zenith sign artefact."""

    def test_zero_crossing_solves_the_quadratic(self) -> None:
        # Hand calculation. The tangent argument reaches 90 degrees when
        # h + 7.31/(h + 4.4) = 90, i.e. h^2 + 4.4h + 7.31 = 90h + 396, i.e.
        # h^2 - 85.6h - 388.69 = 0. The positive root is
        # (85.6 + sqrt(85.6^2 + 4*388.69)) / 2 = 89.92249993368347 degrees.
        root = (85.6 + math.sqrt(85.6**2 + 4.0 * 388.69)) / 2.0
        assert root == pytest.approx(89.92249993368347, rel=1e-13, abs=0.0)
        assert root + 7.31 / (root + 4.4) == pytest.approx(90.0, rel=0.0, abs=1e-12)
        # The declared upper bound sits below the root, so the formula is
        # refused before it changes sign.
        assert refraction.BENNETT_MAX_ALTITUDE_DEG < root

    def test_refused_above_the_declared_maximum(self) -> None:
        with pytest.raises(ValueError, match="refused above"):
            refraction.bennett_refraction(89.95)

    def test_refused_below_the_declared_minimum(self) -> None:
        with pytest.raises(ValueError, match="only above"):
            refraction.bennett_refraction(1.5)

    def test_refused_on_non_finite(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            refraction.bennett_refraction(np.array([30.0, np.nan]))

    def test_monotone_decreasing_over_the_range(self) -> None:
        grid = np.linspace(
            refraction.BENNETT_MIN_ALTITUDE_DEG, refraction.BENNETT_MAX_ALTITUDE_DEG, 400
        )
        values = refraction.bennett_refraction(grid)
        assert np.all(np.diff(values) < 0.0)
        assert float(values[-1]) > 0.0


class TestRefractionShape:
    """Equation (6), the column the pointing model actually uses."""

    def test_unity_at_forty_five_degrees(self) -> None:
        got = float(refraction.refraction_shape(np.array([math.radians(45.0)]))[0])
        assert got == pytest.approx(1.0, rel=1e-14, abs=0.0)

    def test_thirty_degrees_is_a_ratio_of_tangents(self) -> None:
        # Hand calculation. rho_hat(30) = R(30)/R(45) = tan(45.147975708502024
        # deg) / tan(30.2125 deg) because the leading 60 and the conditions
        # factor cancel. That evaluates to 1.726203589934048.
        expected = math.tan(math.radians(45.0 + 7.31 / 49.4)) / math.tan(
            math.radians(30.2125)
        )
        assert expected == pytest.approx(1.726203589934048, rel=1e-13, abs=0.0)
        got = float(refraction.refraction_shape(np.array([math.radians(30.0)]))[0])
        assert got == pytest.approx(expected, rel=1e-13, abs=0.0)

    def test_independent_of_site_conditions(self) -> None:
        angles = np.radians(np.array([10.0, 30.0, 60.0, 85.0]))
        standard = refraction.refraction_shape(angles, 1010.0, 10.0)
        extreme = refraction.refraction_shape(angles, 600.0, -40.0)
        assert np.allclose(standard, extreme, rtol=1e-13, atol=0.0)


class TestRaytrace:
    """The independent calculation: equation (4)."""

    def test_plane_parallel_limit_follows_the_derived_curvature_correction(self) -> None:
        # The flat-atmosphere limit of equation (4) is (n0 - 1) tan(z). The
        # departure of the spherical integral from it is derivable without
        # running either: along the ray, Snell's invariant (3) gives
        # sin(zeta) = sin(z) / (1 + h/R) * (n0/n), so to leading order
        # d(ln sin zeta) = -h/R, hence d(zeta) = -(h/R) tan(zeta) and
        #     d(tan zeta) / tan(zeta) = -(h/R) sec^2(zeta).
        # Weighting that at the refractivity scale height H = R_specific T0 / g
        # = 8290 m gives a relative departure of -(H/R_earth) sec^2(z), which
        # is 1.30e-3 * sec^2(z). The bound asserted is twice that, the factor
        # of two covering the higher-order terms and the standard atmosphere's
        # departure from isothermal. The slack therefore follows sec^2(z), the
        # conditioning of the bound expression, and not a constant: at 85
        # degrees elevation it admits 2.6e-3 and at 60 degrees 1.0e-2.
        scale_height = (
            refraction.DRY_AIR_R_J_PER_KG_K
            * (273.15 + refraction.STANDARD_TEMPERATURE_C)
            / refraction.GRAVITY_M_PER_S2
        )
        assert scale_height == pytest.approx(8290.0, rel=1e-3, abs=0.0)
        ratio = scale_height / refraction.EARTH_RADIUS_M
        assert ratio == pytest.approx(1.3014e-3, rel=1e-3, abs=0.0)
        for apparent in (85.0, 80.0, 70.0, 60.0):
            traced = float(
                np.atleast_1d(
                    refraction.raytrace_refraction(apparent, altitude_is_true=False)
                )[0]
            )
            flat = float(refraction.plane_parallel_refraction(apparent))
            zenith = math.radians(90.0 - apparent)
            bound = 2.0 * ratio / math.cos(zenith) ** 2
            assert traced < flat, f"the correction must be negative at {apparent} deg"
            assert abs(traced / flat - 1.0) < bound, f"at {apparent} deg elevation"

    def test_positive_and_decreasing(self) -> None:
        grid = np.array([10.0, 20.0, 40.0, 60.0, 80.0])
        values = np.atleast_1d(refraction.raytrace_refraction(grid))
        assert np.all(values > 0.0)
        assert np.all(np.diff(values) < 0.0)

    def test_true_and_apparent_altitudes_differ_by_the_refraction(self) -> None:
        # The fixed point is h_app = h_true - R(h_app). Solving it and then
        # integrating at the apparent altitude must reproduce the same number
        # the true-altitude mode returns.
        true_mode = float(
            np.atleast_1d(refraction.raytrace_refraction(30.0, altitude_is_true=True))[0]
        )
        apparent = 30.0 - true_mode / 3600.0
        apparent_mode = float(
            np.atleast_1d(
                refraction.raytrace_refraction(apparent, altitude_is_true=False)
            )[0]
        )
        assert apparent_mode == pytest.approx(true_mode, rel=1e-9, abs=0.0)

    def test_pressure_scales_the_result_almost_linearly(self) -> None:
        # Refractivity is linear in pressure but the refraction integral is
        # not exactly linear in refractivity, because the ray path depends on
        # it. Halving the pressure must halve the refraction to within the
        # second-order term, which is of relative order (n0 - 1) * tan^2(z);
        # at 45 degrees that is 2.8e-4. The bound follows that conditioning.
        full = float(np.atleast_1d(refraction.raytrace_refraction(45.0, 1010.0, 10.0))[0])
        half = float(np.atleast_1d(refraction.raytrace_refraction(45.0, 505.0, 10.0))[0])
        n0_minus_1 = float(refraction.refractivity(0.0, 1010.0, 10.0))
        assert abs(half / (0.5 * full) - 1.0) < 10.0 * n0_minus_1

    def test_refused_on_a_non_positive_altitude(self) -> None:
        with pytest.raises(ValueError, match="positive altitude"):
            refraction.raytrace_refraction(0.0)

    def test_refused_above_ninety(self) -> None:
        with pytest.raises(ValueError, match="above 90"):
            refraction.raytrace_refraction(90.5)


class TestRefractivity:
    """Equation (5) and the two declared constants."""

    def test_optical_constant_is_the_stated_arithmetic(self) -> None:
        # Hand calculation. K_optical = 2.7765e-4 * 288.15 / 1013.25 * 1e6,
        # the standard-air optical refractivity at 550 nm converted to the
        # K/hPa form of equation (5). That is 78.95857... K/hPa.
        expected = 2.7765e-4 * 288.15 / 1013.25 * 1e6
        assert refraction.K_OPTICAL_550NM == pytest.approx(expected, rel=1e-15, abs=0.0)
        assert expected == pytest.approx(78.95857, rel=1e-6, abs=0.0)

    def test_optical_is_above_radio_dry_by_the_measured_fraction(self) -> None:
        ratio = refraction.K_OPTICAL_550NM / refraction.K_RADIO_DRY
        assert ratio == pytest.approx(1.017507, rel=1e-5, abs=0.0)

    def test_surface_refractivity_at_standard_conditions(self) -> None:
        # Hand calculation. N = 77.6 * 1010 / 283.15 ppm, and
        # 1010 / 283.15 = 3.5670140 , so N = 276.8003 ppm and
        # n - 1 = 2.768003e-4 at the observer.
        assert 1010.0 / 283.15 == pytest.approx(3.5670140, rel=1e-7, abs=0.0)
        expected = 77.6e-6 * 1010.0 / 283.15
        assert expected == pytest.approx(2.768003e-4, rel=1e-6, abs=0.0)
        got = float(refraction.refractivity(0.0, 1010.0, 10.0))
        assert got == pytest.approx(expected, rel=1e-14, abs=0.0)

    def test_refractivity_falls_with_height(self) -> None:
        heights = np.array([0.0, 1000.0, 5000.0, 11000.0, 30000.0])
        values = refraction.refractivity(heights, 1010.0, 10.0)
        assert np.all(np.diff(values) < 0.0)
        assert float(values[-1]) / float(values[0]) < 0.05

    def test_refused_on_bad_conditions(self) -> None:
        with pytest.raises(ValueError, match="pressure must be positive"):
            refraction.refractivity(0.0, -1.0, 10.0)
        with pytest.raises(ValueError, match="above absolute zero"):
            refraction.refractivity(0.0, 1010.0, -300.0)


class TestCrossCheck:
    """The cross-check itself, as a test rather than only in validation/."""

    def test_shape_agreement_is_independent_of_the_constant(self) -> None:
        grid = np.linspace(10.0, 85.0, 8)
        radio = refraction.compare_refraction(
            grid, refractivity_constant=refraction.K_RADIO_DRY
        )
        optical = refraction.compare_refraction(
            grid, refractivity_constant=refraction.K_OPTICAL_550NM
        )
        # Changing the refractivity constant by 1.75 per cent changes the shape
        # disagreement by well under a tenth of itself, because the shape is a
        # ratio in which the constant nearly cancels.
        assert optical.max_shape_relative == pytest.approx(
            radio.max_shape_relative, rel=0.1, abs=0.0
        )
        assert radio.max_shape_relative < 0.025
        # The absolute scale, by contrast, moves with the constant.
        assert optical.max_absolute_arcsec < radio.max_absolute_arcsec

    def test_optical_absolute_agreement_is_within_bennetts_quoted_accuracy(self) -> None:
        # Meeus (1998) ch.16 states the accuracy of equation (1) as about 0.07
        # arcminutes, which is 4.2 arcsec. With the optical refractivity
        # constant the measured maximum absolute disagreement over 10 to 85
        # degrees is below that figure. This is a comparison against a
        # published accuracy claim, not against a tolerance chosen here.
        grid = np.linspace(10.0, 85.0, 8)
        optical = refraction.compare_refraction(
            grid, refractivity_constant=refraction.K_OPTICAL_550NM
        )
        assert optical.max_absolute_arcsec < 0.07 * 60.0

    def test_radio_constant_exceeds_it_and_that_is_reported(self) -> None:
        # With the radio dry-air constant the disagreement exceeds the quoted
        # accuracy, because the constant is 1.75 per cent below the optical
        # one and equation (1) is an optical fit. This is asserted so that the
        # fact is a test and not a footnote.
        grid = np.linspace(10.0, 85.0, 8)
        radio = refraction.compare_refraction(
            grid, refractivity_constant=refraction.K_RADIO_DRY
        )
        assert radio.max_absolute_arcsec > 0.07 * 60.0
        assert radio.mean_ratio > optical_mean_ratio_upper_bound()


def optical_mean_ratio_upper_bound() -> float:
    """The optical mean ratio, computed once, for the comparison above."""
    grid = np.linspace(10.0, 85.0, 8)
    return refraction.compare_refraction(
        grid, refractivity_constant=refraction.K_OPTICAL_550NM
    ).mean_ratio
