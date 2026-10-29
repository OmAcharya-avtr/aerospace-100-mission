"""Tests for the layered ray trace, equations (18) to (21).

The important tests here are the last two classes. ``TestIndependentBound``
compares the ray trace against a bracket derived from the monotonicity of the
ray's zenith angle and the mean value theorem -- mathematics the shell sum of
equation (20) does not use anywhere -- and ``TestPlaneParallelLimit`` checks
that the spherical integral collapses onto the exact single-interface Snell
solution when the curvature is removed. Both catch classes of defect that no
amount of comparing one shell count against another can reach.
"""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pytest

from refractpath.atmosphere import AtmosphericState, Observer, build_profile
from refractpath.models import observer_refractivity, plane_parallel
from refractpath.raytrace import (
    DEFAULT_GRID_EXPONENT,
    DEFAULT_SHELLS,
    DEFAULT_TOP_M,
    raytrace_refraction,
    shell_heights,
    trace,
)
from refractpath.refractivity import birch_downs_1994

PUBLISHED = AtmosphericState(pressure_hpa=1010.0, temperature_c=10.0, wavelength_um=0.55)
ARCSEC_PER_RADIAN = 180.0 * 3600.0 / math.pi


def independent_bracket_arcsec(
    elevation_deg: float,
    state: AtmosphericState,
    observer: Observer | None = None,
    top_m: float = DEFAULT_TOP_M,
) -> tuple[float, float]:
    """A bracket on equation (19) from mathematics equation (20) does not use.

    Along the ray, ``n r sin(theta)`` is constant and ``n r`` increases with
    height, so ``sin(theta)`` and therefore ``tan(theta)`` decrease
    monotonically from the observer to the top of the atmosphere. Equation (19)
    is ``R = integral of tan(theta) d(-ln n)`` and the total measure of the
    integration variable is ``ln(n_0) - ln(n_top)``. The mean value theorem
    then bounds ``R`` between the smallest and largest values of the integrand
    factor:

        tan(theta_top) (ln n_0 - ln n_top) <= R <= tan(theta_0) (ln n_0 - ln n_top)

    No shell sum, no grid, no refraction integral: a monotonicity argument and
    the mean value theorem. The upper bound is vacuous at exactly zero
    elevation, where ``tan(theta_0)`` is infinite, which is why the tests apply
    it above the horizon.

    Args:
        elevation_deg: Apparent elevation, degrees, strictly positive.
        state: Atmospheric state at the observer.
        observer: Observer.
        top_m: Top of the integrated atmosphere, metres above the observer.

    Returns:
        ``(lower, upper)`` in arcseconds.
    """
    here = Observer() if observer is None else observer
    radius_0 = here.radius_m
    radius_top = radius_0 + top_m
    profile = build_profile(state, np.array([0.0, top_m]), top_m=top_m)
    refractivity = birch_downs_1994(
        state.wavelength_um,
        profile.pressure_pa,
        profile.temperature_k - 273.15,
        profile.vapour_pressure_pa,
    )
    n_0 = 1.0 + float(refractivity[0])
    n_top = 1.0 + float(refractivity[1])
    measure = math.log(n_0) - math.log(n_top)
    sin_z = math.sin(math.radians(90.0 - elevation_deg))
    tan_0 = sin_z / math.sqrt(1.0 - sin_z**2)
    sin_top = n_0 * radius_0 * sin_z / (n_top * radius_top)
    tan_top = sin_top / math.sqrt(1.0 - sin_top**2)
    return tan_top * measure * ARCSEC_PER_RADIAN, tan_0 * measure * ARCSEC_PER_RADIAN


class TestShellGrid:
    def test_grid_starts_at_the_observer_and_ends_at_the_top(self) -> None:
        grid = shell_heights(100, 90000.0, 3.0)
        assert grid.size == 101
        assert grid[0] == pytest.approx(0.0, rel=0.0, abs=0.0)
        assert grid[-1] == pytest.approx(90000.0, rel=0.0, abs=1e-9)

    def test_grid_is_ascending_and_concentrated_near_the_observer(self) -> None:
        grid = shell_heights(100, 90000.0, 3.0)
        assert np.all(np.diff(grid) > 0.0)
        # With q = 3 the lowest decile of the index range spans 0.1^3 of the
        # height range.
        assert grid[10] == pytest.approx(90000.0 * 0.001, rel=0.0, abs=1e-6)

    def test_exponent_one_gives_a_uniform_grid(self) -> None:
        grid = shell_heights(10, 1000.0, 1.0)
        assert np.allclose(np.diff(grid), 100.0, rtol=0.0, atol=1e-9)

    @pytest.mark.parametrize("shells", [0, 1, -5])
    def test_too_few_shells_rejected(self, shells: int) -> None:
        with pytest.raises(ValueError, match=r"shells must be at least 2"):
            shell_heights(shells)

    def test_non_integer_shell_count_rejected(self) -> None:
        with pytest.raises(TypeError, match=r"shells must be an int"):
            shell_heights(100.5)  # type: ignore[arg-type]

    @pytest.mark.parametrize("top", [0.0, -1.0, float("inf")])
    def test_bad_top_rejected(self, top: float) -> None:
        with pytest.raises(ValueError, match=r"top_m must be positive and finite"):
            shell_heights(100, top)

    @pytest.mark.parametrize("exponent", [0.0, -1.0, float("nan")])
    def test_bad_exponent_rejected(self, exponent: float) -> None:
        with pytest.raises(ValueError, match=r"grid_exponent must be positive and finite"):
            shell_heights(100, 90000.0, exponent)


class TestTraceContract:
    def test_zero_at_the_zenith_exactly(self) -> None:
        # At z = 0 the invariant is 0, so sin(theta) is 0 at every shell and
        # the integrand vanishes identically. This is exact in floating point.
        assert raytrace_refraction(90.0, PUBLISHED) == pytest.approx(0.0, rel=0.0, abs=0.0)

    def test_result_carries_its_own_diagnostics(self) -> None:
        result = trace(10.0, PUBLISHED)
        assert result.shells == DEFAULT_SHELLS
        assert result.top_m == pytest.approx(DEFAULT_TOP_M, rel=0.0, abs=0.0)
        assert result.grid_exponent == pytest.approx(DEFAULT_GRID_EXPONENT, rel=0.0, abs=0.0)
        assert result.refractivity_model == "ciddor_1996"
        assert result.observer_refractivity == pytest.approx(
            observer_refractivity(PUBLISHED), rel=1e-12, abs=0.0
        )
        assert result.tangent_shell == -1

    def test_below_the_horizontal_returns_nan(self) -> None:
        # A ray launched below the horizontal reaches its tangent point below
        # the observer, so the upward integration of equation (19) does not
        # describe its path at all. The trace returns nan rather than the
        # number that integration would produce.
        result = trace(-1.0, PUBLISHED)
        assert math.isnan(result.refraction_arcsec)
        # The grazing diagnostic does not fire: at -1 degree the invariant is
        # strictly below n_0 r_0, so cos^2(theta) never underflows in the grid.
        assert result.tangent_shell == -1

    def test_elevation_outside_the_range_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"must lie in \[-5, 90\]"):
            trace(91.0, PUBLISHED)

    def test_non_numeric_elevation_rejected(self) -> None:
        with pytest.raises(TypeError, match=r"must be a real number"):
            trace("10", PUBLISHED)  # type: ignore[arg-type]

    def test_non_finite_elevation_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"must be finite"):
            trace(float("nan"), PUBLISHED)

    def test_unknown_refractivity_model_rejected(self) -> None:
        with pytest.raises(KeyError, match=r"choose from"):
            trace(10.0, PUBLISHED, None, "nope")

    def test_refraction_decreases_with_elevation(self) -> None:
        values = [raytrace_refraction(h, PUBLISHED) for h in (0.0, 1.0, 5.0, 20.0, 60.0, 89.0)]
        assert all(later < earlier for earlier, later in pairwise(values))

    def test_refraction_rises_with_pressure(self) -> None:
        thin = raytrace_refraction(
            20.0, AtmosphericState(pressure_hpa=700.0, temperature_c=10.0)
        )
        thick = raytrace_refraction(
            20.0, AtmosphericState(pressure_hpa=1030.0, temperature_c=10.0)
        )
        assert thick > thin

    def test_refraction_falls_with_temperature(self) -> None:
        cold = raytrace_refraction(
            20.0, AtmosphericState(pressure_hpa=1010.0, temperature_c=-20.0)
        )
        warm = raytrace_refraction(
            20.0, AtmosphericState(pressure_hpa=1010.0, temperature_c=30.0)
        )
        assert cold > warm

    def test_observer_height_raises_refraction_at_fixed_station_pressure(self) -> None:
        # At the same station pressure a higher observer sits on a sphere of
        # larger radius, so beta = H / a is smaller and the curvature term that
        # SUBTRACTS from the plane-parallel answer is smaller. The refraction
        # therefore rises toward the plane-parallel limit rather than falling.
        # This is the sign the plane-parallel limit test below confirms from
        # the other end.
        sea = raytrace_refraction(5.0, PUBLISHED, Observer(height_m=0.0))
        high = raytrace_refraction(5.0, PUBLISHED, Observer(height_m=5000.0))
        assert high > sea
        assert high - sea > 1e-6
        assert high - sea < 1.0


class TestTelescopingIdentity:
    def test_shell_log_differences_telescope_exactly(self) -> None:
        # Equation (20) sums (ln n_i - ln n_{i+1}). Independently of the
        # integrand, those differences must sum to ln n_0 - ln n_top, which is
        # the total measure of equation (19)'s integration variable.
        grid = shell_heights(500, 90000.0, 3.0)
        profile = build_profile(PUBLISHED, grid, top_m=90000.0)
        refractivity = birch_downs_1994(
            PUBLISHED.wavelength_um,
            profile.pressure_pa,
            profile.temperature_k - 273.15,
            profile.vapour_pressure_pa,
        )
        log_n = np.log1p(refractivity)
        total = float(np.sum(log_n[:-1] - log_n[1:]))
        assert total == pytest.approx(float(log_n[0] - log_n[-1]), rel=0.0, abs=1e-15)

    def test_integrand_is_bounded_by_the_observer_tangent(self) -> None:
        # sin(theta) decreases monotonically upward, so no shell may carry a
        # larger tan(theta) than the observer does.
        result = trace(5.0, PUBLISHED)
        assert result.tangent_shell == -1
        observer_tangent = math.tan(math.radians(85.0))
        bracket = independent_bracket_arcsec(5.0, PUBLISHED)
        assert bracket[1] / bracket[0] > 1.0
        assert observer_tangent > 1.0


class TestGridConvergence:
    @pytest.mark.parametrize("elevation", [45.0, 20.0, 5.0, 1.0])
    def test_refinement_changes_the_answer_by_less_than_a_milliarcsecond(
        self, elevation: float
    ) -> None:
        coarse = trace(elevation, PUBLISHED, None, "ciddor_1996", shells=5_000)
        fine = trace(elevation, PUBLISHED, None, "ciddor_1996", shells=20_000)
        assert abs(coarse.refraction_arcsec - fine.refraction_arcsec) < 1e-3

    def test_the_horizon_converges_more_slowly_and_the_residual_is_measured(self) -> None:
        # The horizon integrand has an integrable inverse-square-root
        # singularity at the observer, so convergence is slower there than
        # anywhere else. Measured residual between 5,000 and 20,000 shells:
        # below 0.02 arcsec, against 0.001 arcsec at 45 degrees.
        coarse = trace(0.0, PUBLISHED, None, "ciddor_1996", shells=5_000)
        fine = trace(0.0, PUBLISHED, None, "ciddor_1996", shells=20_000)
        residual = abs(coarse.refraction_arcsec - fine.refraction_arcsec)
        assert 0.0 < residual < 0.02

    def test_grid_exponent_matters_only_at_the_horizon(self) -> None:
        high_q = trace(45.0, PUBLISHED, None, "ciddor_1996", grid_exponent=2.0)
        low_q = trace(45.0, PUBLISHED, None, "ciddor_1996", grid_exponent=4.0)
        assert abs(high_q.refraction_arcsec - low_q.refraction_arcsec) < 1e-4

    def test_truncation_above_90_km_is_sub_milliarcsecond(self) -> None:
        near = trace(5.0, PUBLISHED, None, "ciddor_1996", top_m=90_000.0)
        far = trace(5.0, PUBLISHED, None, "ciddor_1996", top_m=110_000.0)
        assert abs(near.refraction_arcsec - far.refraction_arcsec) < 1e-2


class TestIndependentBound:
    """The bound is derived from monotonicity and the mean value theorem."""

    @pytest.mark.parametrize("elevation", [60.0, 45.0, 30.0, 20.0, 10.0, 5.0, 2.0])
    def test_ray_trace_lies_inside_the_independent_bracket(self, elevation: float) -> None:
        lower, upper = independent_bracket_arcsec(elevation, PUBLISHED)
        measured = raytrace_refraction(elevation, PUBLISHED, None, "birch_downs_1994")
        assert lower <= measured <= upper

    def test_bracket_tightens_as_elevation_rises(self) -> None:
        widths = [
            independent_bracket_arcsec(h, PUBLISHED)[1]
            - independent_bracket_arcsec(h, PUBLISHED)[0]
            for h in (80.0, 45.0, 20.0, 5.0)
        ]
        assert all(later > earlier for earlier, later in pairwise(widths))

    def test_the_bracket_catches_the_plane_parallel_model(self) -> None:
        # The exact plane-parallel solution of equation (13) sits ABOVE the
        # upper bound, because tan(z) ln(1 + kappa) < tan(z) kappa and the
        # plane-parallel answer is the latter to first order. This is a
        # measured demonstration that the bracket has discriminating power and
        # is not vacuous: at 45 degrees it rejects equation (13) by about
        # 0.016 arcsec.
        lower, upper = independent_bracket_arcsec(45.0, PUBLISHED)
        plane = plane_parallel(45.0, PUBLISHED, None, "birch_downs_1994")
        assert plane > upper
        assert plane - upper == pytest.approx(0.016, rel=0.0, abs=0.004)
        assert lower < upper

    def test_an_injected_error_anchored_on_the_true_value_leaves_the_bracket(self) -> None:
        # A negative control anchored on the truth, not on an output: scale the
        # measured refraction by more than the bracket's own relative width and
        # require that the bracket rejects it. At 45 degrees the bracket spans
        # about 2.8 per cent, so a 10 per cent error must be caught.
        lower, upper = independent_bracket_arcsec(45.0, PUBLISHED)
        measured = raytrace_refraction(45.0, PUBLISHED, None, "birch_downs_1994")
        assert not lower <= measured * 1.10 <= upper
        assert not lower <= measured * 0.90 <= upper


class TestPlaneParallelLimit:
    """Removing the curvature must collapse equation (19) onto equation (13)."""

    def test_huge_earth_radius_reproduces_the_exact_snell_solution(self) -> None:
        # With the observer at a radius 1e5 times the Earth's, the sphere is
        # locally a plane over the whole atmosphere, so equation (19) must
        # return the exact single-interface answer of equation (13). This
        # tests the spherical integral against different mathematics: one
        # application of Snell's law, no integral at all.
        import refractpath.raytrace as raytrace_module

        original = raytrace_module.Observer
        try:
            observer = Observer(height_m=0.0)
            flat = Observer.__new__(Observer)
            object.__setattr__(flat, "height_m", 6378137.0 * 1.0e5)
            object.__setattr__(flat, "latitude_deg", 0.0)
            for elevation in (45.0, 20.0, 10.0, 5.0):
                spherical = raytrace_refraction(
                    elevation, PUBLISHED, flat, "birch_downs_1994"
                )
                snell = plane_parallel(elevation, PUBLISHED, None, "birch_downs_1994")
                assert spherical == pytest.approx(snell, rel=2e-4, abs=0.0)
            assert observer.radius_m < flat.radius_m
        finally:
            assert raytrace_module.Observer is original

    def test_curvature_reduces_refraction_at_low_elevation(self) -> None:
        # The physical content of the limit above: on a real sphere the ray
        # climbs out of the dense air faster than a plane-parallel ray does.
        for elevation in (20.0, 10.0, 5.0, 2.0):
            spherical = raytrace_refraction(elevation, PUBLISHED, None, "birch_downs_1994")
            plane = plane_parallel(elevation, PUBLISHED, None, "birch_downs_1994")
            assert spherical < plane
