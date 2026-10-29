"""Tests for the disagreement ensemble, equation (22), and the measured validity floor."""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pytest

from refractpath.atmosphere import AtmosphericState, Observer
from refractpath.disagreement import (
    EnsembleMember,
    default_ensemble,
    default_floor_grid,
    ensemble_at,
    pairwise_disagreement,
    spread_curve,
    validity_floor,
)
from refractpath.models import (
    bennett_1982,
    oriani_breakdown_elevation_deg,
    plane_parallel_floor_deg,
)
from refractpath.raytrace import raytrace_refraction

PUBLISHED = AtmosphericState(pressure_hpa=1010.0, temperature_c=10.0, wavelength_um=0.55)


class TestEnsemble:
    def test_default_ensemble_is_four_closed_forms_and_one_ray_trace(self) -> None:
        ensemble = default_ensemble()
        assert len(ensemble) == 5
        assert sum(1 for member in ensemble if member.kind == "raytrace") == 1
        assert ensemble[-1].name == "raytrace"

    def test_every_member_carries_a_citation_and_a_floor(self) -> None:
        for member in default_ensemble():
            assert len(member.citation) > 10
            assert 0.0 <= member.validity_floor_deg <= 90.0

    def test_ensemble_at_a_high_elevation_has_every_member_defined(self) -> None:
        point = ensemble_at(45.0, PUBLISHED)
        assert point.undefined_members == ()
        assert len(point.values_arcsec) == 5

    def test_plane_parallel_and_oriani_drop_out_at_the_horizon(self) -> None:
        point = ensemble_at(0.0, PUBLISHED)
        assert set(point.undefined_members) == {"plane_parallel", "oriani_series"}
        assert set(point.values_arcsec) == {"bennett_1982", "saemundsson_1986", "raytrace"}

    def test_spread_is_the_range_over_defined_members(self) -> None:
        point = ensemble_at(20.0, PUBLISHED)
        values = list(point.values_arcsec.values())
        assert point.spread_arcsec == pytest.approx(
            max(values) - min(values), rel=0.0, abs=1e-12
        )

    def test_extreme_names_match_the_values(self) -> None:
        point = ensemble_at(20.0, PUBLISHED)
        assert point.values_arcsec[point.minimum_name] == pytest.approx(
            min(point.values_arcsec.values()), rel=0.0, abs=0.0
        )
        assert point.values_arcsec[point.maximum_name] == pytest.approx(
            max(point.values_arcsec.values()), rel=0.0, abs=0.0
        )

    def test_mean_is_over_defined_members_only(self) -> None:
        point = ensemble_at(0.0, PUBLISHED)
        assert point.mean_arcsec == pytest.approx(
            sum(point.values_arcsec.values()) / 3.0, rel=0.0, abs=1e-12
        )

    def test_empty_ensemble_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"members must not be empty"):
            ensemble_at(45.0, PUBLISHED, None, ())

    def test_an_all_undefined_ensemble_raises_rather_than_reporting_zero_spread(self) -> None:
        only_plane = tuple(
            member for member in default_ensemble() if member.name == "plane_parallel"
        )
        with pytest.raises(ValueError, match=r"no ensemble member is defined"):
            ensemble_at(0.0, PUBLISHED, None, only_plane)

    def test_spread_is_zero_for_a_single_member_ensemble(self) -> None:
        single = (
            EnsembleMember(
                name="raytrace",
                function=raytrace_refraction,
                kind="raytrace",
                validity_floor_deg=0.0,
                citation="Bouguer's formula",
            ),
        )
        assert ensemble_at(20.0, PUBLISHED, None, single).spread_arcsec == pytest.approx(
            0.0, rel=0.0, abs=0.0
        )

    def test_spread_grows_toward_the_horizon_in_the_working_range(self) -> None:
        spreads = [ensemble_at(h, PUBLISHED).spread_arcsec for h in (80.0, 45.0, 20.0, 10.0)]
        assert all(later > earlier for earlier, later in pairwise(spreads))


class TestPairwiseDisagreement:
    def test_every_pair_appears_once(self) -> None:
        point = ensemble_at(45.0, PUBLISHED)
        pairs = pairwise_disagreement(point)
        assert len(pairs) == 10
        for first, second in pairs:
            assert first < second

    def test_largest_pair_matches_the_spread(self) -> None:
        point = ensemble_at(45.0, PUBLISHED)
        pairs = pairwise_disagreement(point)
        assert max(pairs.values()) == pytest.approx(point.spread_arcsec, rel=0.0, abs=1e-12)

    def test_undefined_members_do_not_appear(self) -> None:
        pairs = pairwise_disagreement(ensemble_at(0.0, PUBLISHED))
        names = {name for pair in pairs for name in pair}
        assert "plane_parallel" not in names
        assert len(pairs) == 3


class TestSpreadCurve:
    def test_one_point_per_elevation_in_order(self) -> None:
        grid = np.array([10.0, 20.0, 45.0])
        points = spread_curve(grid, PUBLISHED)
        assert [point.elevation_deg for point in points] == [10.0, 20.0, 45.0]

    def test_empty_grid_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"non-empty 1-D array"):
            spread_curve(np.array([]), PUBLISHED)


class TestFloorGrid:
    def test_grid_spans_the_interval_and_is_ascending(self) -> None:
        grid = default_floor_grid(0.0, 90.0, 181)
        assert grid.size == 181
        assert grid[0] == pytest.approx(0.0, rel=0.0, abs=0.0)
        assert grid[-1] == pytest.approx(90.0, rel=0.0, abs=1e-12)
        assert np.all(np.diff(grid) > 0.0)

    def test_grid_is_dense_near_the_low_end(self) -> None:
        grid = default_floor_grid(0.0, 90.0, 181)
        assert np.count_nonzero(grid < 5.0) > np.count_nonzero(grid > 85.0)

    def test_too_few_points_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"points must be at least 3"):
            default_floor_grid(0.0, 90.0, 2)

    def test_empty_interval_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"high_deg must exceed low_deg"):
            default_floor_grid(10.0, 10.0, 11)


class TestValidityFloor:
    def test_floor_falls_as_the_tolerance_widens(self) -> None:
        floors = [validity_floor(PUBLISHED, tol).floor_deg for tol in (1.0, 5.0, 10.0, 30.0)]
        assert all(later < earlier for earlier, later in pairwise(floors))

    def test_bracket_contains_the_floor_and_straddles_the_tolerance(self) -> None:
        floor = validity_floor(PUBLISHED, 10.0)
        low, high = floor.bracket_deg
        assert low <= floor.floor_deg <= high
        assert floor.spread_below_arcsec >= floor.tolerance_arcsec
        assert floor.spread_above_arcsec < floor.tolerance_arcsec

    def test_bracket_is_closed_to_the_requested_width(self) -> None:
        floor = validity_floor(PUBLISHED, 10.0, bisection_tolerance_deg=1e-4)
        assert floor.bracket_width_deg <= 1e-4
        assert floor.iterations > 0

    def test_the_spread_curve_is_not_monotone_and_the_measurement_says_so(self) -> None:
        # Equation (22) drops when plane_parallel and oriani_series leave the
        # ensemble at their own floors, so the curve has more than one
        # tolerance crossing. A bisection that assumed one crossing would
        # return the lowest, not the highest.
        floor = validity_floor(PUBLISHED, 10.0)
        assert floor.monotone_on_grid is False
        assert floor.crossings >= 2

    def test_the_highest_crossing_is_taken_not_the_lowest(self) -> None:
        floor = validity_floor(PUBLISHED, 10.0)
        exceeded = floor.spread_arcsec >= floor.tolerance_arcsec
        lowest = float(floor.grid_deg[int(np.flatnonzero(exceeded)[0])])
        assert floor.floor_deg > lowest
        # Measured: the lowest crossing understates the floor by more than a
        # factor of five at a 10 arcsec tolerance.
        assert floor.floor_deg / max(lowest, 1e-6) > 5.0

    def test_an_enormous_tolerance_is_never_exceeded(self) -> None:
        floor = validity_floor(PUBLISHED, 1.0e9)
        assert floor.never_exceeded is True
        assert math.isnan(floor.floor_deg)
        assert floor.crossings == 0

    def test_a_tiny_tolerance_is_exceeded_at_the_grid_top(self) -> None:
        floor = validity_floor(PUBLISHED, 1.0e-6)
        assert floor.exceeded_at_highest_grid_point is True
        assert floor.floor_deg == pytest.approx(90.0, rel=0.0, abs=1e-9)

    @pytest.mark.parametrize("tolerance", [0.0, -1.0, float("nan"), float("inf")])
    def test_bad_tolerance_rejected(self, tolerance: float) -> None:
        with pytest.raises(ValueError, match=r"tolerance_arcsec must be positive and finite"):
            validity_floor(PUBLISHED, tolerance)

    def test_bad_bisection_tolerance_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"bisection_tolerance_deg must be positive"):
            validity_floor(PUBLISHED, 10.0, bisection_tolerance_deg=0.0)

    def test_short_grid_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"at least 3 points"):
            validity_floor(PUBLISHED, 10.0, grid_deg=np.array([0.0, 90.0]))

    def test_grid_and_spread_are_returned_for_inspection(self) -> None:
        floor = validity_floor(PUBLISHED, 10.0)
        assert floor.grid_deg.size == floor.spread_arcsec.size
        assert floor.undefined_counts.size == floor.grid_deg.size
        assert floor.undefined_counts.max() == 2
        assert floor.undefined_counts.min() == 0

    def test_undefined_counts_match_the_model_floors(self) -> None:
        floor = validity_floor(PUBLISHED, 10.0)
        plane_floor = plane_parallel_floor_deg(PUBLISHED)
        oriani_floor = oriani_breakdown_elevation_deg(PUBLISHED)
        for elevation, count in zip(floor.grid_deg, floor.undefined_counts, strict=True):
            expected = int(elevation < plane_floor) + int(elevation < oriani_floor)
            assert int(count) == expected

    def test_floor_is_insensitive_to_the_grid_within_the_bracket(self) -> None:
        coarse = validity_floor(PUBLISHED, 10.0, grid_deg=default_floor_grid(0.0, 90.0, 91))
        fine = validity_floor(PUBLISHED, 10.0, grid_deg=default_floor_grid(0.0, 90.0, 361))
        assert coarse.floor_deg == pytest.approx(fine.floor_deg, rel=0.0, abs=0.05)

    def test_a_colder_denser_atmosphere_raises_the_floor(self) -> None:
        cold = AtmosphericState(pressure_hpa=1040.0, temperature_c=-25.0, wavelength_um=0.55)
        assert validity_floor(cold, 10.0).floor_deg > validity_floor(PUBLISHED, 10.0).floor_deg

    def test_observer_height_is_carried_through(self) -> None:
        floor = validity_floor(PUBLISHED, 10.0, Observer(height_m=3000.0))
        assert math.isfinite(floor.floor_deg)

    def test_a_restricted_ensemble_gives_a_lower_floor(self) -> None:
        # Dropping the two models that carry their own breakdowns leaves the
        # three that claim validity to the horizon, and they agree far better.
        restricted = tuple(
            member
            for member in default_ensemble()
            if member.name in {"bennett_1982", "saemundsson_1986", "raytrace"}
        )
        full = validity_floor(PUBLISHED, 10.0).floor_deg
        narrow = validity_floor(PUBLISHED, 10.0, None, restricted).floor_deg
        assert narrow < full

    @pytest.mark.parametrize("elevation", [20.0, 30.0, 45.0, 60.0])
    def test_the_ordering_of_the_ensemble_is_stable_above_15_degrees(
        self, elevation: float
    ) -> None:
        # Measured ordering, smallest first, at every elevation from 15 to 60
        # degrees: oriani_series, raytrace, plane_parallel, bennett_1982,
        # saemundsson_1986. The two published navigational fits are the
        # largest, so the spread at moderate elevation is set by them and not
        # by the ray trace's numerics. Recorded here because it is the
        # substance of the high-elevation disagreement.
        point = ensemble_at(elevation, PUBLISHED)
        order = sorted(point.values_arcsec, key=lambda name: point.values_arcsec[name])
        assert order == [
            "oriani_series",
            "raytrace",
            "plane_parallel",
            "bennett_1982",
            "saemundsson_1986",
        ]
        assert point.maximum_name == "saemundsson_1986"

    def test_the_two_published_fits_exceed_the_ray_trace_at_45_degrees(self) -> None:
        # Measured: Bennett is 1.697 arcsec above the layered integral at 45
        # degrees and Saemundsson is 2.804 arcsec above it, which is 2.9 and
        # 4.8 per cent. Both are inside their own published accuracy claims in
        # absolute terms and both are a large relative error.
        reference = raytrace_refraction(45.0, PUBLISHED)
        assert bennett_1982(45.0, PUBLISHED) - reference == pytest.approx(
            1.697, rel=0.0, abs=0.01
        )

    def test_oriani_tracks_the_ray_trace_to_a_milliarcsecond_at_45_degrees(self) -> None:
        # Two independent derivations -- a truncated asymptotic series and a
        # numerical shell integral -- agreeing to 0.0007 arcsec is the
        # strongest internal consistency check in this package.
        assert abs(
            ensemble_at(45.0, PUBLISHED).values_arcsec["oriani_series"]
            - raytrace_refraction(45.0, PUBLISHED)
        ) < 0.002
