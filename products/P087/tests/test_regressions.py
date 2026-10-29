"""Pinned regressions: defects found during the build, and Hypothesis counterexamples.

``.hypothesis/`` is gitignored, so a falsifying example found by one run is
lost on a fresh clone. Every counterexample Hypothesis produced during this
build is pinned here with its literal inputs in the test body, and every defect
found by hand is pinned beside it with the number it produced before the fix.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from refractpath.atmosphere import EARTH_RADIUS_M, G0, R_DRY, AtmosphericState, Observer
from refractpath.disagreement import default_ensemble, ensemble_at, validity_floor
from refractpath.models import (
    observer_refractivity,
    oriani_breakdown_elevation_deg,
    oriani_series,
    oriani_series_unguarded,
    plane_parallel,
    plane_parallel_floor_deg,
)
from refractpath.raytrace import raytrace_refraction, trace
from refractpath.refractivity import birch_downs_1994

PUBLISHED = AtmosphericState(pressure_hpa=1010.0, temperature_c=10.0, wavelength_um=0.55)


class TestHypothesisCounterexamples:
    def test_pressure_doubling_departure_at_807_hpa_and_minus_2_c(self) -> None:
        # Hypothesis falsified a 1e-3 bound on R(2p)/R(p) - 2 here. The
        # departure is the 1e-8 p term of equation (7) and is deterministic.
        # Measured: 1.0008890197918952e-3.
        single = float(birch_downs_1994(1.0, 80700.0, -2.0, 0.0))
        double = float(birch_downs_1994(1.0, 161400.0, -2.0, 0.0))
        assert abs(double / single - 2.0) == pytest.approx(
            1.00089e-3, rel=0.0, abs=1e-8
        )

    def test_pressure_doubling_departure_at_1002_hpa_and_minus_41_c(self) -> None:
        # The second counterexample, which falsified the widened 2e-3 bound.
        # Measured: 2.0010340063429943e-3. Both are now covered by an exact
        # algebraic identity rather than a tolerance.
        single = float(birch_downs_1994(1.0, 100200.0, -41.0, 0.0))
        double = float(birch_downs_1994(1.0, 200400.0, -41.0, 0.0))
        assert abs(double / single - 2.0) == pytest.approx(
            2.00103e-3, rel=0.0, abs=1e-8
        )
        c = 0.601 - 0.00972 * -41.0
        expected = 2.0 * (1.0 + 1e-8 * c * 200400.0) / (1.0 + 1e-8 * c * 100200.0)
        assert double / single == pytest.approx(expected, rel=1e-12, abs=0.0)

    def test_ray_trace_and_plane_parallel_are_both_exactly_zero_at_the_zenith(self) -> None:
        # Hypothesis falsified a strict "spherical < plane" at elevation
        # exactly 90 degrees, 400 hPa, 0 C, where both are identically zero.
        # The invariant of equation (18) is zero at the zenith, so every
        # shell's integrand is zero; this is exact, not approximate.
        state = AtmosphericState(pressure_hpa=400.0, temperature_c=0.0, wavelength_um=0.55)
        assert raytrace_refraction(90.0, state, None, "birch_downs_1994") == 0.0
        assert plane_parallel(90.0, state, None, "birch_downs_1994") == pytest.approx(
            0.0, rel=0.0, abs=0.0
        )

    def test_the_two_computed_floors_cross_at_1079_hpa_and_minus_59_c(self) -> None:
        # Hypothesis falsified "the Oriani breakdown is always above the
        # plane-parallel analytic floor" here. The two floors are
        # 1.6067910228728266 and 1.6069088355197323 degrees, 1.18e-4 degrees
        # apart, and the ordering reverses because beta falls below 2.5 kappa.
        state = AtmosphericState(
            pressure_hpa=1079.0, temperature_c=-59.0, wavelength_um=1.0
        )
        oriani = oriani_breakdown_elevation_deg(state)
        plane = plane_parallel_floor_deg(state)
        assert oriani == pytest.approx(1.6067910, rel=0.0, abs=1e-6)
        assert plane == pytest.approx(1.6069088, rel=0.0, abs=1e-6)
        assert oriani < plane
        kappa = observer_refractivity(state)
        beta = (R_DRY * state.temperature_k / G0) / EARTH_RADIUS_M
        assert beta < 2.5 * kappa


class TestDefectsFoundByHand:
    def test_the_horizon_integrand_is_not_evaluated_at_the_tangent_point(self) -> None:
        # Defect found during the build: a trapezoid rule over shell
        # BOUNDARIES evaluates tan(theta) at the observer, and for a ray
        # launched exactly horizontally sin(theta) is 1 there, so the
        # integrand is infinite and the horizon refraction came out as
        # 8.0e138 arcmin. Equation (20) evaluates at shell MIDPOINTS, which
        # never touches the tangent point. Pinned by requiring a finite,
        # physically sized answer at zero elevation.
        result = trace(0.0, PUBLISHED)
        assert math.isfinite(result.refraction_arcsec)
        assert 1800.0 < result.refraction_arcsec < 2400.0
        assert result.tangent_shell == -1

    def test_cos_squared_is_computed_without_catastrophic_cancellation(self) -> None:
        # Defect class: computing 1 - sin^2(theta) directly loses all
        # precision when sin(theta) is within 1e-14 of unity, which is the
        # first few metres of a horizon ray. Equation (21) factorises it.
        # Pinned by requiring that refining the grid by a factor of four moves
        # the horizon answer by less than 0.02 arcsec; with the cancelling
        # form the refinement changed the answer by orders of magnitude.
        coarse = trace(0.0, PUBLISHED, None, "ciddor_1996", shells=5_000)
        fine = trace(0.0, PUBLISHED, None, "ciddor_1996", shells=20_000)
        assert abs(coarse.refraction_arcsec - fine.refraction_arcsec) < 0.02

    def test_the_diverging_oriani_series_does_not_enter_the_ensemble(self) -> None:
        # Defect found during the build: without the equation (14a) guard the
        # unguarded series returns about -1.26e7 arcsec at 0.1 degrees and
        # about -2.9e47 arcsec at exactly 0 degrees in floating point, which
        # made equation (22) a measurement of a floating-point artefact rather
        # than of a disagreement. The guard returns nan instead; the
        # unguarded function is kept so the divergence can still be exhibited.
        assert math.isnan(oriani_series(0.1, PUBLISHED))
        unguarded = oriani_series_unguarded(0.1, PUBLISHED)
        assert unguarded < -1.0e6
        point = ensemble_at(0.1, PUBLISHED)
        assert "oriani_series" in point.undefined_members
        assert point.spread_arcsec < 1.0e3

    def test_the_validity_floor_takes_the_highest_crossing(self) -> None:
        # Defect class: equation (22) is not monotone, because members leave
        # the ensemble at their own floors and the spread drops when they do.
        # A search that assumed a single crossing and walked up from the
        # horizon would stop at the FIRST one. Measured at a 10 arcsec
        # tolerance: three crossings on the default grid, between 1.225 and
        # 1.344 degrees, between 1.344 and 1.469 degrees, and between 11.025
        # and 11.378 degrees. The first-crossing answer of about 1.22 degrees
        # understates the floor of 11.24 degrees by a factor of 9.2.
        floor = validity_floor(PUBLISHED, 10.0)
        exceeded = floor.spread_arcsec >= floor.tolerance_arcsec
        transitions = np.flatnonzero(exceeded[:-1] != exceeded[1:])
        assert transitions.size == 3
        first_crossing = float(floor.grid_deg[int(transitions[0])])
        assert floor.crossings == 3
        assert floor.monotone_on_grid is False
        assert floor.floor_deg == pytest.approx(11.2448, rel=0.0, abs=0.01)
        assert first_crossing == pytest.approx(1.225, rel=0.0, abs=0.01)
        assert floor.floor_deg / first_crossing == pytest.approx(9.18, rel=0.0, abs=0.05)

    def test_observer_height_raises_rather_than_lowers_the_refraction(self) -> None:
        # Defect found during the build: the test suite first asserted the
        # wrong sign. At fixed station pressure a larger radius means a
        # smaller beta and therefore a smaller curvature correction, so the
        # refraction rises toward the plane-parallel limit. Measured at 5
        # degrees: 589.6488 arcsec at sea level, 589.6995 arcsec at 5 km.
        sea = raytrace_refraction(5.0, PUBLISHED, Observer(height_m=0.0))
        high = raytrace_refraction(5.0, PUBLISHED, Observer(height_m=5000.0))
        assert sea == pytest.approx(589.6488, rel=0.0, abs=0.002)
        assert high == pytest.approx(589.6995, rel=0.0, abs=0.002)
        assert high > sea

    def test_the_ensemble_ordering_puts_the_published_fits_highest(self) -> None:
        # Pinned because it is the substance of the moderate-elevation
        # disagreement and would silently change if any model were edited.
        point = ensemble_at(45.0, PUBLISHED)
        order = sorted(point.values_arcsec, key=lambda name: point.values_arcsec[name])
        assert order == [
            "oriani_series",
            "raytrace",
            "plane_parallel",
            "bennett_1982",
            "saemundsson_1986",
        ]
        assert point.values_arcsec["raytrace"] == pytest.approx(
            57.9939, rel=0.0, abs=0.002
        )
        assert point.values_arcsec["bennett_1982"] == pytest.approx(
            59.6909, rel=0.0, abs=0.002
        )
        assert point.values_arcsec["saemundsson_1986"] == pytest.approx(
            60.7981, rel=0.0, abs=0.002
        )

    def test_the_default_ensemble_has_not_silently_changed_size(self) -> None:
        assert len(default_ensemble()) == 5
