"""Tests for the observer state, the atmospheric state and the layered profile."""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pytest

from refractpath.atmosphere import (
    EARTH_RADIUS_M,
    G0,
    PRESSURE_DOMAIN_HPA,
    R_DRY,
    TEMPERATURE_DOMAIN_C,
    US1976_LAYERS,
    WAVELENGTH_DOMAIN_UM,
    AtmosphericState,
    Observer,
    build_profile,
    saturation_vapour_pressure_pa,
)


class TestAtmosphericStateValidation:
    def test_defaults_are_standard_conditions(self) -> None:
        state = AtmosphericState()
        assert state.pressure_hpa == pytest.approx(1013.25, rel=0.0, abs=0.0)
        assert state.temperature_c == pytest.approx(10.0, rel=0.0, abs=0.0)
        assert state.temperature_k == pytest.approx(283.15, rel=0.0, abs=1e-12)
        assert state.pressure_pa == pytest.approx(101325.0, rel=0.0, abs=1e-9)

    @pytest.mark.parametrize("pressure", [0.0, -1.0, -1013.25])
    def test_non_positive_pressure_rejected(self, pressure: float) -> None:
        with pytest.raises(ValueError, match=r"pressure_hpa must be positive"):
            AtmosphericState(pressure_hpa=pressure)

    def test_temperature_below_absolute_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"above absolute zero"):
            AtmosphericState(temperature_c=-273.15)

    @pytest.mark.parametrize("humidity", [-0.01, 1.01, 43.0])
    def test_humidity_outside_unit_interval_rejected(self, humidity: float) -> None:
        with pytest.raises(ValueError, match=r"relative_humidity must lie in \[0, 1\]"):
            AtmosphericState(relative_humidity=humidity)

    def test_humidity_error_names_the_fraction_convention(self) -> None:
        with pytest.raises(ValueError, match=r"pass 0.43 rather than 43"):
            AtmosphericState(relative_humidity=43.0)

    @pytest.mark.parametrize("wavelength", [0.0, -0.55])
    def test_non_positive_wavelength_rejected(self, wavelength: float) -> None:
        with pytest.raises(ValueError, match=r"wavelength_um must be positive"):
            AtmosphericState(wavelength_um=wavelength)

    def test_negative_co2_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"co2_ppm must be non-negative"):
            AtmosphericState(co2_ppm=-1.0)

    def test_non_positive_vapour_scale_height_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"vapour_scale_height_m must be positive"):
            AtmosphericState(vapour_scale_height_m=0.0)

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_rejected(self, value: float) -> None:
        with pytest.raises(ValueError, match=r"must be finite"):
            AtmosphericState(pressure_hpa=value)

    def test_string_pressure_rejected_with_typeerror(self) -> None:
        with pytest.raises(TypeError, match=r"must be a real number"):
            AtmosphericState(pressure_hpa="1013.25")  # type: ignore[arg-type]

    def test_bool_is_not_a_real_number(self) -> None:
        with pytest.raises(TypeError, match=r"must be a real number"):
            AtmosphericState(pressure_hpa=True)  # type: ignore[arg-type]

    def test_state_is_frozen(self) -> None:
        state = AtmosphericState()
        with pytest.raises(Exception, match=r"cannot assign|frozen"):
            state.pressure_hpa = 900.0  # type: ignore[misc]

    @pytest.mark.parametrize("top", [0.0, -1.0, 11000.0, 20000.0])
    def test_surface_layer_top_outside_open_interval_rejected(self, top: float) -> None:
        with pytest.raises(ValueError, match=r"surface_layer_top_m must lie strictly"):
            AtmosphericState(surface_layer_top_m=top)

    def test_surface_layer_top_accepts_an_interior_value(self) -> None:
        state = AtmosphericState(surface_layer_top_m=300.0)
        assert state.surface_layer_top_m == pytest.approx(300.0, rel=0.0, abs=0.0)


class TestDeclaredDomains:
    def test_default_state_is_inside_every_domain(self) -> None:
        assert all(AtmosphericState().inside_declared_domains().values())

    def test_wavelength_outside_domain_is_reported_not_raised(self) -> None:
        report = AtmosphericState(wavelength_um=2.5).inside_declared_domains()
        assert report["wavelength_um"] is False
        assert report["pressure_hpa"] is True

    def test_pressure_outside_domain_is_reported(self) -> None:
        report = AtmosphericState(pressure_hpa=200.0).inside_declared_domains()
        assert report["pressure_hpa"] is False

    def test_temperature_outside_domain_is_reported(self) -> None:
        report = AtmosphericState(temperature_c=70.0).inside_declared_domains()
        assert report["temperature_c"] is False

    def test_domains_are_ordered_intervals(self) -> None:
        for low, high in (WAVELENGTH_DOMAIN_UM, PRESSURE_DOMAIN_HPA, TEMPERATURE_DOMAIN_C):
            assert high > low


class TestSaturationVapourPressure:
    def test_known_answer_at_20_c(self) -> None:
        # Hand calculation of equation (4) at t = 20 C:
        #   (18.678 - 20 / 234.5) = 18.678 - 0.0852878 = 18.5927122
        #   20 / (257.14 + 20)    = 20 / 277.14        = 0.07217436
        #   product               = 1.34193...
        #   611.21 * exp(1.34193) = 2338.34 Pa
        # The standard tabulated value is 2339 Pa; see
        # refractpath.references.SATURATION_VAPOUR_PRESSURE_20C.
        assert saturation_vapour_pressure_pa(20.0) == pytest.approx(2338.34, rel=0.0, abs=0.01)

    def test_known_answer_at_0_c(self) -> None:
        # Equation (4) at t = 0 collapses to the leading constant exactly.
        assert saturation_vapour_pressure_pa(0.0) == pytest.approx(611.21, rel=0.0, abs=1e-9)

    def test_increases_with_temperature(self) -> None:
        values = [saturation_vapour_pressure_pa(t) for t in (-20.0, 0.0, 10.0, 20.0, 40.0)]
        assert all(later > earlier for earlier, later in pairwise(values))

    def test_non_finite_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"must be finite"):
            saturation_vapour_pressure_pa(float("nan"))


class TestObserver:
    def test_default_observer_is_at_the_reference_sphere(self) -> None:
        assert Observer().radius_m == pytest.approx(EARTH_RADIUS_M, rel=0.0, abs=0.0)

    def test_height_adds_to_the_radius(self) -> None:
        assert Observer(height_m=2500.0).radius_m == pytest.approx(
            EARTH_RADIUS_M + 2500.0, rel=0.0, abs=0.0
        )

    def test_very_negative_height_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"height_m must be at least"):
            Observer(height_m=-600.0)

    @pytest.mark.parametrize("latitude", [-90.1, 90.1, 180.0])
    def test_latitude_outside_range_rejected(self, latitude: float) -> None:
        with pytest.raises(ValueError, match=r"latitude_deg must lie in \[-90, 90\]"):
            Observer(latitude_deg=latitude)

    def test_non_finite_height_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"must be finite"):
            Observer(height_m=float("inf"))


class TestLayeredProfile:
    def test_surface_values_are_the_observer_values(self) -> None:
        state = AtmosphericState(pressure_hpa=900.0, temperature_c=-5.0)
        profile = build_profile(state, np.array([0.0, 100.0]))
        assert profile.temperature_k[0] == pytest.approx(268.15, rel=0.0, abs=1e-12)
        assert profile.pressure_pa[0] == pytest.approx(90000.0, rel=0.0, abs=1e-9)

    def test_temperature_follows_the_declared_lapse_rate_in_the_first_layer(self) -> None:
        state = AtmosphericState(temperature_c=10.0, surface_lapse_rate_k_per_m=-0.0065)
        profile = build_profile(state, np.array([0.0, 1000.0, 5000.0]))
        # T(z) = 283.15 - 0.0065 z, exactly, from equation (1).
        assert profile.temperature_k[1] == pytest.approx(276.65, rel=0.0, abs=1e-9)
        assert profile.temperature_k[2] == pytest.approx(250.65, rel=0.0, abs=1e-9)

    def test_isothermal_layer_uses_the_exponential_branch(self) -> None:
        state = AtmosphericState(surface_lapse_rate_k_per_m=0.0)
        profile = build_profile(state, np.array([0.0, 1000.0]))
        expected = 101325.0 * math.exp(-G0 * 1000.0 / (R_DRY * 283.15))
        assert profile.pressure_pa[1] == pytest.approx(expected, rel=1e-13, abs=0.0)

    def test_pressure_decreases_monotonically(self) -> None:
        heights = np.linspace(0.0, 80000.0, 400)
        profile = build_profile(AtmosphericState(), heights)
        assert np.all(np.diff(profile.pressure_pa) < 0.0)

    def test_pressure_at_11_km_is_near_the_1976_tropopause_value(self) -> None:
        # Anchored at 1013.25 hPa and 15 C the 1976 model gives 226.32 hPa at
        # 11 km. This product anchors at the observer, so starting from 15 C
        # reproduces the published tropopause pressure to within the
        # geometric-for-geopotential substitution declared in equation (1).
        state = AtmosphericState(pressure_hpa=1013.25, temperature_c=15.0)
        profile = build_profile(state, np.array([0.0, 11000.0]))
        assert profile.pressure_pa[1] / 100.0 == pytest.approx(226.3, rel=0.0, abs=1.0)

    def test_vapour_pressure_decays_with_the_declared_scale_height(self) -> None:
        state = AtmosphericState(relative_humidity=0.5, temperature_c=20.0)
        profile = build_profile(state, np.array([0.0, 2000.0, 4000.0]))
        surface = profile.vapour_pressure_pa[0]
        assert profile.vapour_pressure_pa[1] == pytest.approx(
            surface * math.exp(-1.0), rel=1e-12, abs=0.0
        )
        assert profile.vapour_pressure_pa[2] == pytest.approx(
            surface * math.exp(-2.0), rel=1e-12, abs=0.0
        )

    def test_vapour_pressure_is_capped_below_the_total_pressure(self) -> None:
        state = AtmosphericState(
            relative_humidity=1.0, temperature_c=45.0, vapour_scale_height_m=100000.0
        )
        heights = np.linspace(0.0, 80000.0, 200)
        profile = build_profile(state, heights)
        assert np.all(profile.vapour_pressure_pa <= 0.9 * profile.pressure_pa + 1e-9)

    def test_dry_state_has_zero_vapour_everywhere(self) -> None:
        profile = build_profile(AtmosphericState(), np.linspace(0.0, 1000.0, 10))
        assert np.all(profile.vapour_pressure_pa == 0.0)

    def test_surface_inversion_layer_changes_only_the_lowest_layer(self) -> None:
        plain = AtmosphericState(surface_lapse_rate_k_per_m=0.02, surface_layer_top_m=200.0)
        profile = build_profile(plain, np.array([0.0, 200.0, 1200.0]))
        # +20 K/km over 200 m, then the 1976 -6.5 K/km over the next km.
        assert profile.temperature_k[1] == pytest.approx(283.15 + 4.0, rel=0.0, abs=1e-9)
        assert profile.temperature_k[2] == pytest.approx(283.15 + 4.0 - 6.5, rel=0.0, abs=1e-9)

    def test_layer_tops_are_reported(self) -> None:
        profile = build_profile(AtmosphericState(), np.linspace(0.0, 80000.0, 50))
        assert profile.layer_tops_m[0] == pytest.approx(11000.0, rel=0.0, abs=0.0)

    def test_length_is_the_grid_size(self) -> None:
        assert len(build_profile(AtmosphericState(), np.linspace(0.0, 100.0, 17))) == 17

    @pytest.mark.parametrize("grid", [np.array([]), np.zeros((2, 2))])
    def test_bad_grid_shape_rejected(self, grid: np.ndarray) -> None:
        with pytest.raises(ValueError, match=r"non-empty 1-D array"):
            build_profile(AtmosphericState(), grid)

    def test_negative_height_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"height_m must be non-negative"):
            build_profile(AtmosphericState(), np.array([-1.0, 0.0]))

    def test_descending_grid_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"height_m must be ascending"):
            build_profile(AtmosphericState(), np.array([0.0, 100.0, 50.0]))

    def test_non_finite_grid_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"height_m must be finite"):
            build_profile(AtmosphericState(), np.array([0.0, float("nan")]))

    def test_lapse_rate_that_freezes_the_profile_is_rejected_with_a_number(self) -> None:
        # -0.03 K/m over 11 km takes 283.15 K to -46.85 K, which equation (1)
        # cannot solve. The message must carry the offending temperature.
        with pytest.raises(ValueError, match=r"non-positive temperature"):
            build_profile(
                AtmosphericState(surface_lapse_rate_k_per_m=-0.03),
                np.linspace(0.0, 11000.0, 10),
            )

    def test_layer_structure_matches_the_1976_declaration(self) -> None:
        assert US1976_LAYERS[0] == (0.0, -0.0065)
        assert US1976_LAYERS[1] == (11000.0, 0.0)
        assert len(US1976_LAYERS) == 8
        assert all(
            later[0] > earlier[0]
            for earlier, later in pairwise(US1976_LAYERS)
        )
