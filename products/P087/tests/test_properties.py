"""Property-based tests for the algebraic identities this package relies on.

Every identity here is exact in exact arithmetic and is asserted to a stated
tolerance on both sides, never as an equality and never as a boolean produced
by a strict threshold comparison. Where Hypothesis found a falsifying example
during the build it is pinned by its literal inputs in
``tests/test_regressions.py``, because ``.hypothesis/`` is gitignored and a
counterexample found by one run is lost on a fresh clone.
"""

from __future__ import annotations

import math

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from refractpath.atmosphere import AtmosphericState, Observer, saturation_vapour_pressure_pa
from refractpath.disagreement import ensemble_at, pairwise_disagreement
from refractpath.models import (
    bennett_1982,
    observer_refractivity,
    oriani_breakdown_elevation_deg,
    oriani_truncation_ratio,
    plane_parallel,
    plane_parallel_floor_deg,
    saemundsson_1986,
    saemundsson_1986_from_true,
)
from refractpath.raytrace import raytrace_refraction
from refractpath.refractivity import (
    REFRACTIVITY_MODELS,
    birch_downs_1994,
    edlen_1966,
    refractivity_spread,
)

ARCSEC_PER_RADIAN = 180.0 * 3600.0 / math.pi

SLOW = settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
FAST = settings(max_examples=200, deadline=None)

pressures = st.floats(min_value=400.0, max_value=1080.0, allow_nan=False)
temperatures = st.floats(min_value=-60.0, max_value=45.0, allow_nan=False)
humidities = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
wavelengths = st.floats(min_value=0.31, max_value=1.65, allow_nan=False)
elevations = st.floats(min_value=0.0, max_value=90.0, allow_nan=False)
working_elevations = st.floats(min_value=5.0, max_value=90.0, allow_nan=False)


def _state(
    pressure: float, temperature: float, humidity: float = 0.0, wavelength: float = 0.55
) -> AtmosphericState:
    return AtmosphericState(
        pressure_hpa=pressure,
        temperature_c=temperature,
        relative_humidity=humidity,
        wavelength_um=wavelength,
    )


class TestRefractivityIdentities:
    @FAST
    @given(wavelengths, pressures, temperatures)
    def test_edlen_family_pressure_doubling_matches_its_closed_form_exactly(
        self, wavelength: float, pressure: float, temperature: float
    ) -> None:
        # Hypothesis falsified two successive loose bounds on "doubling the
        # pressure doubles the refractivity" -- 1e-3 at 807 hPa and -2 C, then
        # 2e-3 at 1002 hPa and -41 C -- because the departure is a known
        # second-order term, not noise. Both are pinned in
        # tests/test_regressions.py. The property asserted here is the exact
        # algebraic ratio implied by equation (7):
        #
        #   R(2p) / R(p) = 2 (1 + 1e-8 c 2p) / (1 + 1e-8 c p),  c = 0.601 - 0.00972 t
        #
        # which holds to rounding and is a far stronger statement than any
        # tolerance on 2.
        single = float(birch_downs_1994(wavelength, pressure * 100.0, temperature, 0.0))
        double = float(birch_downs_1994(wavelength, pressure * 200.0, temperature, 0.0))
        c = 0.601 - 0.00972 * temperature
        expected = (
            2.0
            * (1.0 + 1e-8 * c * pressure * 200.0)
            / (1.0 + 1e-8 * c * pressure * 100.0)
        )
        assert abs(double / single - expected) <= 1e-12 * expected

    @FAST
    @given(wavelengths, pressures, temperatures)
    def test_birch_downs_exceeds_edlen_everywhere_in_the_domain(
        self, wavelength: float, pressure: float, temperature: float
    ) -> None:
        # The 1994 correction is a positive offset on the dispersion constant,
        # and the shared equation (7) factor is positive, so the ordering holds
        # at every state in the domain.
        difference = float(
            birch_downs_1994(wavelength, pressure * 100.0, temperature, 0.0)
        ) - float(edlen_1966(wavelength, pressure * 100.0, temperature, 0.0))
        assert difference > 0.0

    @FAST
    @given(wavelengths, pressures, temperatures, humidities)
    def test_every_formulation_is_positive_in_the_domain(
        self, wavelength: float, pressure: float, temperature: float, humidity: float
    ) -> None:
        state = _state(pressure, temperature, humidity, wavelength)
        for name, function in REFRACTIVITY_MODELS.items():
            value = (
                float(
                    function(
                        wavelength,
                        state.pressure_pa,
                        temperature,
                        state.vapour_pressure_pa,
                        state.co2_ppm,
                    )
                )
                if name == "ciddor_1996"
                else float(
                    function(
                        wavelength, state.pressure_pa, temperature, state.vapour_pressure_pa
                    )
                )
            )
            assert value > 0.0

    @FAST
    @given(wavelengths, pressures, temperatures, humidities)
    def test_spread_equals_the_range_of_the_members(
        self, wavelength: float, pressure: float, temperature: float, humidity: float
    ) -> None:
        state = _state(pressure, temperature, humidity, wavelength)
        report = refractivity_spread(
            wavelength, state.pressure_pa, temperature, state.vapour_pressure_pa
        )
        members = [report[name] for name in REFRACTIVITY_MODELS]
        assert abs(report["spread"] - (max(members) - min(members))) <= 1e-18

    @FAST
    @given(temperatures)
    def test_saturation_vapour_pressure_is_positive_and_increasing(
        self, temperature: float
    ) -> None:
        here = saturation_vapour_pressure_pa(temperature)
        warmer = saturation_vapour_pressure_pa(temperature + 1.0)
        assert here > 0.0
        assert warmer > here


class TestClosedFormIdentities:
    @FAST
    @given(working_elevations, pressures, temperatures)
    def test_plane_parallel_satisfies_snells_law_by_construction(
        self, elevation: float, pressure: float, temperature: float
    ) -> None:
        # Equation (13) is defined by sin(z_true) = n sin(z_app). Recovering
        # that identity from the returned angle is a check on the inversion,
        # not a tautology: the function returns z_true - z_app, so the test
        # must reconstruct z_true.
        state = _state(pressure, temperature)
        refraction = plane_parallel(elevation, state)
        if math.isnan(refraction):
            return
        n = 1.0 + observer_refractivity(state)
        z_app = math.radians(90.0 - elevation)
        z_true = z_app + refraction / ARCSEC_PER_RADIAN
        assert abs(math.sin(z_true) - n * math.sin(z_app)) <= 1e-14

    @FAST
    @given(pressures, temperatures, wavelengths)
    def test_plane_parallel_is_undefined_exactly_below_its_analytic_floor(
        self, pressure: float, temperature: float, wavelength: float
    ) -> None:
        state = _state(pressure, temperature, 0.0, wavelength)
        floor = plane_parallel_floor_deg(state)
        # A margin of 1e-6 degrees keeps the test off the boundary itself,
        # where the comparison is a step function of a rounded quantity.
        assert math.isnan(plane_parallel(max(floor - 1e-6, -4.999), state))
        assert math.isfinite(plane_parallel(min(floor + 1e-6, 90.0), state))

    @FAST
    @given(pressures, temperatures, wavelengths)
    def test_oriani_truncation_ratio_is_one_at_the_computed_breakdown(
        self, pressure: float, temperature: float, wavelength: float
    ) -> None:
        state = _state(pressure, temperature, 0.0, wavelength)
        breakdown = oriani_breakdown_elevation_deg(state)
        assert abs(oriani_truncation_ratio(breakdown, state) - 1.0) <= 1e-9

    @FAST
    @given(elevations, pressures, temperatures)
    def test_equation_17_scales_bennett_exactly(
        self, elevation: float, pressure: float, temperature: float
    ) -> None:
        reference = bennett_1982(elevation, _state(1010.0, 10.0))
        scaled = bennett_1982(elevation, _state(pressure, temperature))
        factor = (pressure / 1010.0) * (283.0 / (273.0 + temperature))
        if abs(reference) < 1e-9:
            return
        assert abs(scaled / reference - factor) <= 1e-9 * max(1.0, abs(factor))

    @FAST
    @given(elevations, pressures, temperatures)
    def test_saemundsson_inversion_round_trips(
        self, elevation: float, pressure: float, temperature: float
    ) -> None:
        state = _state(pressure, temperature)
        refraction = saemundsson_1986(elevation, state)
        if math.isnan(refraction):
            return
        true_elevation = elevation - refraction / 3600.0
        again = saemundsson_1986_from_true(true_elevation, state)
        assert abs(again - refraction) <= 1e-7 * max(1.0, abs(refraction))

    @FAST
    @given(pressures, temperatures, wavelengths)
    def test_both_computed_floors_lie_inside_the_declared_elevation_range(
        self, pressure: float, temperature: float, wavelength: float
    ) -> None:
        state = _state(pressure, temperature, 0.0, wavelength)
        plane = plane_parallel_floor_deg(state)
        oriani = oriani_breakdown_elevation_deg(state)
        assert 0.0 < plane < 10.0
        assert 0.0 < oriani < 10.0

    @FAST
    @given(pressures, temperatures, wavelengths)
    def test_the_two_floors_cross_where_beta_equals_two_and_a_half_kappa(
        self, pressure: float, temperature: float, wavelength: float
    ) -> None:
        # Small-angle expansion of the two closed forms gives
        #   plane_parallel floor  ~ sqrt(2 kappa)        radians
        #   oriani breakdown      ~ sqrt(beta - kappa/2) radians
        # so the ordering reverses exactly where beta = 2.5 kappa. Hypothesis
        # found the reversal at 1079 hPa, -59 C and 1.0 um, where the two
        # floors differ by 1.2e-4 degrees; the counterexample is pinned in
        # tests/test_regressions.py. The ordering is asserted only where the
        # discriminant is comfortably away from zero, because the comparison
        # is a step function of a rounded quantity on the boundary itself.
        from refractpath.atmosphere import EARTH_RADIUS_M, G0, R_DRY

        state = _state(pressure, temperature, 0.0, wavelength)
        kappa = observer_refractivity(state)
        beta = (R_DRY * state.temperature_k / G0) / EARTH_RADIUS_M
        discriminant = beta - 2.5 * kappa
        if abs(discriminant) < 0.05 * beta:
            return
        plane = plane_parallel_floor_deg(state)
        oriani = oriani_breakdown_elevation_deg(state)
        assert (oriani > plane) is (discriminant > 0.0)


class TestRayTraceIdentities:
    @SLOW
    @given(working_elevations, pressures, temperatures)
    def test_ray_trace_lies_inside_the_independent_bracket(
        self, elevation: float, pressure: float, temperature: float
    ) -> None:
        # The bracket is the monotonicity plus mean-value-theorem argument of
        # tests/test_raytrace.py, which shares no mathematics with the shell
        # sum of equation (20). The slack is the bracket's own width, so it
        # follows the conditioning of the bound and not an absolute constant.
        from test_raytrace import independent_bracket_arcsec

        state = _state(pressure, temperature)
        lower, upper = independent_bracket_arcsec(elevation, state)
        measured = raytrace_refraction(elevation, state, None, "birch_downs_1994")
        width = upper - lower
        assert lower - 1e-9 * max(1.0, width) <= measured <= upper + 1e-9 * max(1.0, width)

    @SLOW
    @given(working_elevations, pressures, temperatures)
    def test_ray_trace_is_below_the_plane_parallel_answer(
        self, elevation: float, pressure: float, temperature: float
    ) -> None:
        # Curvature carries the ray out of the dense air sooner, so the
        # spherical integral is always the smaller of the two where both are
        # defined.
        # At exactly 90 degrees both are identically zero, which Hypothesis
        # found immediately; the inequality is strict only below the zenith.
        # The counterexample is pinned in tests/test_regressions.py.
        state = _state(pressure, temperature)
        plane = plane_parallel(elevation, state, None, "birch_downs_1994")
        if math.isnan(plane):
            return
        spherical = raytrace_refraction(elevation, state, None, "birch_downs_1994")
        if elevation >= 90.0:
            assert spherical == plane == 0.0
        else:
            assert spherical < plane

    @SLOW
    @given(pressures, temperatures)
    def test_ray_trace_is_zero_at_the_zenith(
        self, pressure: float, temperature: float
    ) -> None:
        # Exact: the invariant is zero, so every shell's integrand is zero.
        assert raytrace_refraction(90.0, _state(pressure, temperature)) == 0.0

    @SLOW
    @given(working_elevations, pressures, temperatures)
    def test_ray_trace_decreases_with_elevation(
        self, elevation: float, pressure: float, temperature: float
    ) -> None:
        state = _state(pressure, temperature)
        lower = raytrace_refraction(min(elevation, 89.0), state)
        higher = raytrace_refraction(min(elevation, 89.0) + 0.5, state)
        assert higher < lower

    @SLOW
    @given(working_elevations, pressures, temperatures, humidities)
    def test_humidity_lowers_the_optical_refraction(
        self, elevation: float, pressure: float, temperature: float, humidity: float
    ) -> None:
        dry = raytrace_refraction(elevation, _state(pressure, temperature, 0.0))
        humid = raytrace_refraction(elevation, _state(pressure, temperature, humidity))
        assert humid <= dry + 1e-12


class TestEnsembleIdentities:
    @SLOW
    @given(working_elevations, pressures, temperatures)
    def test_spread_equals_the_largest_pairwise_difference(
        self, elevation: float, pressure: float, temperature: float
    ) -> None:
        point = ensemble_at(elevation, _state(pressure, temperature))
        pairs = pairwise_disagreement(point)
        assert abs(max(pairs.values()) - point.spread_arcsec) <= 1e-9 * max(
            1.0, point.spread_arcsec
        )

    @SLOW
    @given(working_elevations, pressures, temperatures)
    def test_mean_lies_between_the_extremes(
        self, elevation: float, pressure: float, temperature: float
    ) -> None:
        point = ensemble_at(elevation, _state(pressure, temperature))
        values = list(point.values_arcsec.values())
        assert min(values) - 1e-12 <= point.mean_arcsec <= max(values) + 1e-12

    @SLOW
    @given(working_elevations, pressures, temperatures)
    def test_spread_is_non_negative(
        self, elevation: float, pressure: float, temperature: float
    ) -> None:
        assert ensemble_at(elevation, _state(pressure, temperature)).spread_arcsec >= 0.0

    @SLOW
    @given(st.floats(min_value=0.0, max_value=6000.0, allow_nan=False))
    def test_observer_height_leaves_the_ensemble_defined(self, height: float) -> None:
        point = ensemble_at(30.0, _state(1010.0, 10.0), Observer(height_m=height))
        assert point.undefined_members == ()
        assert point.spread_arcsec > 0.0
