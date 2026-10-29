"""Tests for the closed-form refraction models, equations (13) to (17)."""

from __future__ import annotations

import math
from itertools import pairwise

import pytest

from refractpath.atmosphere import AtmosphericState, Observer
from refractpath.models import (
    CLOSED_FORM_BY_NAME,
    CLOSED_FORM_MODELS,
    bennett_1982,
    observer_refractivity,
    oriani_breakdown_elevation_deg,
    oriani_series,
    oriani_series_unguarded,
    oriani_truncation_ratio,
    plane_parallel,
    plane_parallel_floor_deg,
    saemundsson_1986,
    saemundsson_1986_from_true,
)

#: The conditions equations (15) and (16) were published for.
PUBLISHED = AtmosphericState(pressure_hpa=1010.0, temperature_c=10.0, wavelength_um=0.55)


class TestObserverRefractivity:
    def test_matches_the_formulation_directly(self) -> None:
        assert observer_refractivity(PUBLISHED, "ciddor_1996") == pytest.approx(
            2.8185193e-4, rel=0.0, abs=1e-11
        )

    def test_unknown_formulation_rejected_with_the_choices_named(self) -> None:
        with pytest.raises(KeyError, match=r"choose from"):
            observer_refractivity(PUBLISHED, "not_a_model")

    def test_humidity_lowers_it(self) -> None:
        humid = AtmosphericState(
            pressure_hpa=1010.0, temperature_c=20.0, relative_humidity=1.0
        )
        dry = AtmosphericState(pressure_hpa=1010.0, temperature_c=20.0)
        assert observer_refractivity(humid) < observer_refractivity(dry)


class TestPlaneParallel:
    def test_zero_at_the_zenith_exactly(self) -> None:
        # z = 0 makes sin z = 0, so asin(n * 0) - 0 = 0 identically.
        assert plane_parallel(90.0, PUBLISHED) == pytest.approx(0.0, rel=0.0, abs=1e-12)

    def test_known_answer_at_45_degrees(self) -> None:
        # Exact form at z = 45 deg with n - 1 = 2.8185193e-4:
        #   asin(1.00028185193 * sin(45 deg)) - 45 deg
        # = asin(0.707305875...) - 0.785398163... rad
        # = 2.818717e-4 rad = 58.1443 arcsec
        # Note this exceeds (n-1) tan z = 58.1413 arcsec, because the exact
        # form of equation (13) is not the first-order form.
        assert plane_parallel(45.0, PUBLISHED) == pytest.approx(
            58.1443, rel=0.0, abs=0.0002
        )

    def test_first_order_form_agrees_at_high_elevation(self) -> None:
        kappa = observer_refractivity(PUBLISHED)
        for elevation in (90.0, 80.0, 60.0, 45.0):
            first_order = (
                kappa * math.tan(math.radians(90.0 - elevation)) * 180.0 * 3600.0 / math.pi
            )
            assert plane_parallel(elevation, PUBLISHED) == pytest.approx(
                first_order, rel=2e-4, abs=0.0
            )

    def test_analytic_floor_is_where_the_equation_stops_having_a_solution(self) -> None:
        floor = plane_parallel_floor_deg(PUBLISHED)
        assert floor == pytest.approx(1.3601824, rel=0.0, abs=1e-6)
        assert math.isnan(plane_parallel(floor - 1e-6, PUBLISHED))
        assert math.isfinite(plane_parallel(floor + 1e-6, PUBLISHED))

    def test_floor_matches_its_closed_form(self) -> None:
        n = 1.0 + observer_refractivity(PUBLISHED)
        expected = 90.0 - math.degrees(math.asin(1.0 / n))
        assert plane_parallel_floor_deg(PUBLISHED) == pytest.approx(
            expected, rel=0.0, abs=1e-12
        )

    def test_floor_rises_with_refractivity(self) -> None:
        dense = AtmosphericState(pressure_hpa=1050.0, temperature_c=-20.0)
        thin = AtmosphericState(pressure_hpa=700.0, temperature_c=30.0)
        assert plane_parallel_floor_deg(dense) > plane_parallel_floor_deg(thin)

    def test_undefined_at_the_horizon(self) -> None:
        assert math.isnan(plane_parallel(0.0, PUBLISHED))

    def test_monotone_decreasing_above_the_floor(self) -> None:
        values = [plane_parallel(h, PUBLISHED) for h in (2.0, 5.0, 10.0, 30.0, 60.0, 89.0)]
        assert all(later < earlier for earlier, later in pairwise(values))


class TestOrianiSeries:
    def test_zero_at_the_zenith_exactly(self) -> None:
        assert oriani_series(90.0, PUBLISHED) == pytest.approx(0.0, rel=0.0, abs=1e-12)

    def test_known_answer_at_45_degrees(self) -> None:
        # kappa = 2.8185193e-4; H = 287.0528 * 283.15 / 9.80665 = 8288.5 m;
        # beta = 8288.5 / 6378137 = 1.299516e-3; tan z = 1 exactly at z = 45.
        #   first  term = kappa (1 - beta)         = 2.8148576e-4
        #   second term = kappa (beta - kappa / 2) = 3.26527e-7
        #   series      = 2.8115923e-4 rad = 57.99324 arcsec
        assert oriani_series(45.0, PUBLISHED) == pytest.approx(
            57.99324, rel=0.0, abs=0.0002
        )

    def test_breakdown_elevation_is_computed_not_declared(self) -> None:
        assert oriani_breakdown_elevation_deg(PUBLISHED) == pytest.approx(
            1.9507071, rel=0.0, abs=1e-6
        )

    def test_truncation_ratio_is_one_at_the_breakdown_elevation(self) -> None:
        breakdown = oriani_breakdown_elevation_deg(PUBLISHED)
        assert oriani_truncation_ratio(breakdown, PUBLISHED) == pytest.approx(
            1.0, rel=0.0, abs=1e-9
        )

    def test_guard_returns_nan_below_the_breakdown_elevation(self) -> None:
        breakdown = oriani_breakdown_elevation_deg(PUBLISHED)
        assert math.isnan(oriani_series(breakdown - 0.01, PUBLISHED))
        assert math.isfinite(oriani_series(breakdown + 0.01, PUBLISHED))

    def test_unguarded_form_returns_the_divergent_value(self) -> None:
        breakdown = oriani_breakdown_elevation_deg(PUBLISHED)
        value = oriani_series_unguarded(breakdown - 0.5, PUBLISHED)
        assert math.isfinite(value)
        # Below breakdown the tan^3 term dominates and the series goes
        # negative, which is not a small error but a reversal.
        assert value < 0.0

    def test_unguarded_form_is_undefined_at_the_horizontal(self) -> None:
        assert math.isnan(oriani_series_unguarded(0.0, PUBLISHED))
        assert math.isinf(oriani_truncation_ratio(0.0, PUBLISHED))

    def test_ratio_falls_with_elevation(self) -> None:
        values = [oriani_truncation_ratio(h, PUBLISHED) for h in (3.0, 5.0, 15.0, 45.0, 89.0)]
        assert all(later < earlier for earlier, later in pairwise(values))

    def test_observer_height_enters_through_beta(self) -> None:
        sea = oriani_series(20.0, PUBLISHED, Observer(height_m=0.0))
        high = oriani_series(20.0, PUBLISHED, Observer(height_m=4000.0))
        assert sea != high
        assert abs(sea - high) < 0.01


class TestBennett:
    def test_known_answer_at_45_degrees(self) -> None:
        # Equation (15) at h = 45: 7.31 / 49.4 = 0.1479757 deg,
        # cot(45.1479757 deg) = 0.9948534 arcmin = 59.6912 arcsec, then
        # equation (17) with p = 1010 and t = 10 is exactly 1.
        assert bennett_1982(45.0, PUBLISHED) == pytest.approx(59.69088, rel=0.0, abs=0.0002)

    def test_known_answer_at_the_horizon(self) -> None:
        # h = 0: 7.31 / 4.4 = 1.6613636 deg, cot = 34.47753 arcmin.
        assert bennett_1982(0.0, PUBLISHED) / 60.0 == pytest.approx(
            34.47753, rel=0.0, abs=0.0002
        )

    def test_negative_at_the_zenith_as_published(self) -> None:
        # The published formula adds 7.31 / 94.4 deg to 90 deg, which puts the
        # cotangent in the second quadrant. This is a property of equation (15)
        # and it is reported, not clipped.
        assert bennett_1982(90.0, PUBLISHED) < 0.0
        assert bennett_1982(90.0, PUBLISHED) == pytest.approx(-0.08109, rel=0.0, abs=1e-4)

    def test_pressure_temperature_scaling_of_equation_17(self) -> None:
        reference = bennett_1982(20.0, PUBLISHED)
        colder = bennett_1982(
            20.0, AtmosphericState(pressure_hpa=1010.0, temperature_c=-10.0)
        )
        assert colder / reference == pytest.approx(283.0 / 263.0, rel=1e-12, abs=0.0)

    def test_pressure_scaling_is_linear(self) -> None:
        half = bennett_1982(20.0, AtmosphericState(pressure_hpa=505.0, temperature_c=10.0))
        full = bennett_1982(20.0, PUBLISHED)
        assert half / full == pytest.approx(0.5, rel=1e-12, abs=0.0)

    def test_ignores_wavelength_as_the_published_formula_does(self) -> None:
        blue = AtmosphericState(pressure_hpa=1010.0, temperature_c=10.0, wavelength_um=0.4)
        red = AtmosphericState(pressure_hpa=1010.0, temperature_c=10.0, wavelength_um=1.0)
        assert bennett_1982(20.0, blue) == pytest.approx(
            bennett_1982(20.0, red), rel=0.0, abs=0.0
        )

    def test_ignores_humidity_as_the_published_formula_does(self) -> None:
        humid = AtmosphericState(
            pressure_hpa=1010.0, temperature_c=10.0, relative_humidity=1.0
        )
        assert bennett_1982(20.0, humid) == pytest.approx(
            bennett_1982(20.0, PUBLISHED), rel=0.0, abs=0.0
        )


class TestSaemundsson:
    def test_known_answer_in_the_published_direction(self) -> None:
        # Equation (16) at h_true = 45: 10.3 / 50.11 = 0.2055478 deg,
        #   tan(45.2055478 deg) = 1.0072008
        #   1.02 / 1.0071962    = 1.0127077 arcmin = 60.76246 arcsec,
        # with the equation (17) factor exactly 1 at 1010 hPa and 10 C.
        assert saemundsson_1986_from_true(45.0, PUBLISHED) == pytest.approx(
            60.76246, rel=0.0, abs=0.0005
        )

    def test_inversion_round_trips_onto_the_apparent_elevation(self) -> None:
        for apparent in (0.0, 1.0, 5.0, 20.0, 45.0, 80.0):
            refraction = saemundsson_1986(apparent, PUBLISHED)
            true_elevation = apparent - refraction / 3600.0
            assert saemundsson_1986_from_true(true_elevation, PUBLISHED) == pytest.approx(
                refraction, rel=0.0, abs=1e-9
            )

    def test_inverted_value_exceeds_the_direct_value_at_the_same_argument(self) -> None:
        # R is decreasing in elevation, and h_true < h_app, so evaluating at
        # the true elevation gives the larger number.
        assert saemundsson_1986(45.0, PUBLISHED) > saemundsson_1986_from_true(
            45.0, PUBLISHED
        )

    def test_negative_at_the_zenith_as_published(self) -> None:
        assert saemundsson_1986(90.0, PUBLISHED) < 0.0

    def test_pole_at_minus_5_11_degrees_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"-5.11 degree pole"):
            saemundsson_1986_from_true(-5.11, PUBLISHED)

    def test_non_finite_true_elevation_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"must be finite"):
            saemundsson_1986_from_true(float("nan"), PUBLISHED)


class TestSharedContract:
    @pytest.mark.parametrize("model", CLOSED_FORM_MODELS, ids=lambda m: m.name)
    def test_elevation_above_the_zenith_rejected(self, model: object) -> None:
        with pytest.raises(ValueError, match=r"must lie in \[-5, 90\]"):
            model.function(90.1, PUBLISHED)  # type: ignore[attr-defined]

    @pytest.mark.parametrize("model", CLOSED_FORM_MODELS, ids=lambda m: m.name)
    def test_elevation_far_below_the_horizon_rejected(self, model: object) -> None:
        with pytest.raises(ValueError, match=r"must lie in \[-5, 90\]"):
            model.function(-5.1, PUBLISHED)  # type: ignore[attr-defined]

    @pytest.mark.parametrize("model", CLOSED_FORM_MODELS, ids=lambda m: m.name)
    def test_non_numeric_elevation_rejected(self, model: object) -> None:
        with pytest.raises(TypeError, match=r"must be a real number"):
            model.function("20", PUBLISHED)  # type: ignore[attr-defined]

    @pytest.mark.parametrize("model", CLOSED_FORM_MODELS, ids=lambda m: m.name)
    def test_non_finite_elevation_rejected(self, model: object) -> None:
        with pytest.raises(ValueError, match=r"must be finite"):
            model.function(float("inf"), PUBLISHED)  # type: ignore[attr-defined]

    @pytest.mark.parametrize("model", CLOSED_FORM_MODELS, ids=lambda m: m.name)
    def test_positive_refraction_in_the_working_range(self, model: object) -> None:
        for elevation in (20.0, 45.0, 70.0):
            value = model.function(elevation, PUBLISHED)  # type: ignore[attr-defined]
            assert value > 0.0

    def test_every_model_declares_provenance(self) -> None:
        for model in CLOSED_FORM_MODELS:
            assert model.equation.startswith("(")
            assert len(model.citation) > 10
            assert len(model.validity_source) > 20
            assert 0.0 <= model.validity_floor_deg <= 90.0

    def test_lookup_by_name_is_complete(self) -> None:
        assert set(CLOSED_FORM_BY_NAME) == {model.name for model in CLOSED_FORM_MODELS}

    def test_the_two_published_fits_take_no_refractivity_model(self) -> None:
        assert CLOSED_FORM_BY_NAME["bennett_1982"].uses_refractivity_model is False
        assert CLOSED_FORM_BY_NAME["saemundsson_1986"].uses_refractivity_model is False
        assert CLOSED_FORM_BY_NAME["plane_parallel"].uses_refractivity_model is True
        assert CLOSED_FORM_BY_NAME["oriani_series"].uses_refractivity_model is True

    @pytest.mark.parametrize("model", CLOSED_FORM_MODELS, ids=lambda m: m.name)
    def test_refraction_decreases_with_elevation_in_the_working_range(
        self, model: object
    ) -> None:
        values = [
            model.function(h, PUBLISHED)  # type: ignore[attr-defined]
            for h in (30.0, 45.0, 60.0, 80.0)
        ]
        assert all(later < earlier for earlier, later in pairwise(values))
