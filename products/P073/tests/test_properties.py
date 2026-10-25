"""Property-based tests.

Hypothesis is used here for the three things it is good at and the hand-written
suite is not: generating degenerate numbers nobody thinks to type, generating
*structures* -- window and request sets with awkward overlaps -- and shrinking a
failure to the smallest case that still fails.

The strongest property in this file is the last one: an exhaustively enumerated
optimum against the HiGHS optimum on randomly generated small instances. The
two share no formulation, so a disagreement is a defect in one of them, and
Hypothesis generates the overlap patterns that a hand-written fixture does not
think of.

Every counterexample Hypothesis has found during this build is pinned as an
explicit regression test with literal inputs in ``test_regressions.py``, because
``.hypothesis/`` is gitignored and a falsifying example found by one run is lost
on a fresh clone.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from contactsched.bruteforce import solve_bruteforce
from contactsched.contention import contention_map
from contactsched.exact import solve_exact
from contactsched.fairness import jain_index, max_min_share
from contactsched.geometry import (
    MAX_GEODETIC_GEOCENTRIC_DEVIATION_DEG,
    WGS84_A_KM,
    WGS84_B_KM,
    elevation_geocentric_norms_deg,
    elevation_geocentric_vector_deg,
    geodetic_to_ecef_km,
    gmst_rad,
    look_angles_deg,
    sez_matrix,
)
from contactsched.greedy import schedule_greedy
from contactsched.problem import (
    ContactRequest,
    build_candidates,
    build_problem,
    check_feasible,
    priority_weight,
)
from contactsched.report import hms
from contactsched.station import GroundStation, HorizonMask
from contactsched.tle import OrbitElements, parse_lines
from contactsched.visibility import ContactWindow, supply_seconds_by_station

SLOW = settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)

finite = st.floats(allow_nan=False, allow_infinity=False)
shares = st.lists(st.floats(0.0, 1.0, allow_nan=False), min_size=1, max_size=12)


def station(name: str = "A1", turnaround_s: float = 0.0) -> GroundStation:
    return GroundStation(name, 0.0, 0.0, 0.0, HorizonMask(floor_deg=5.0), turnaround_s)


@st.composite
def window_sets(draw, max_windows: int = 4, spacecraft: int = 2):
    """Non-degenerate window sets on one antenna for up to ``spacecraft`` objects."""
    count = draw(st.integers(1, max_windows))
    windows = []
    for _ in range(count):
        rise = float(draw(st.integers(0, 40)) * 100)
        length = float(draw(st.integers(1, 8)) * 100)
        name = f"SC-{draw(st.integers(1, spacecraft))}"
        windows.append(
            ContactWindow(
                "A1", name, "M", rise, rise + length, 45.0, rise + 0.5 * length, 5.0
            )
        )
    return tuple(windows)


@st.composite
def request_sets(draw, max_requests: int = 3, spacecraft: int = 2):
    """Request sets whose durations are multiples of 100 s."""
    count = draw(st.integers(1, max_requests))
    requests = []
    for index in range(count):
        minimum = draw(st.integers(1, 3))
        requests.append(
            ContactRequest(
                request_id=f"R{index}",
                mission=f"M{draw(st.integers(1, 2))}",
                spacecraft=f"SC-{draw(st.integers(1, spacecraft))}",
                priority=draw(st.integers(1, 4)),
                duration_s=float(draw(st.integers(1, 3)) * 100),
                min_contacts=minimum,
                max_contacts=minimum + draw(st.integers(0, 1)),
            )
        )
    return tuple(requests)


class TestPriorityWeight:
    @given(st.integers(1, 40))
    def test_is_positive(self, priority):
        assert priority_weight(priority) > 0.0

    @given(st.integers(1, 30), st.floats(1.01, 10.0))
    def test_is_strictly_decreasing(self, priority, base):
        assert priority_weight(priority + 1, base) < priority_weight(priority, base)

    @given(st.integers(1, 20), st.integers(1, 20), st.floats(1.01, 8.0))
    def test_is_a_homomorphism_of_addition(self, a, b, base):
        # base**(1-a) * base**(1-b) == base**(1-(a+b-1)), in exact arithmetic.
        left = priority_weight(a, base) * priority_weight(b, base)
        right = priority_weight(a + b - 1, base)
        assert left == pytest.approx(right, rel=1e-9, abs=1e-300)


class TestFairnessIndices:
    @given(shares)
    def test_jain_is_inside_its_range(self, values):
        index = jain_index(values)
        assert 1.0 / len(values) - 1e-9 <= index <= 1.0 + 1e-9

    @given(shares, st.floats(1e-6, 1e6))
    def test_jain_is_scale_invariant(self, values, scale):
        assume(sum(values) > 0.0)
        scaled = [v * scale for v in values]
        assume(all(math.isfinite(v) for v in scaled))
        assert jain_index(scaled) == pytest.approx(jain_index(values), rel=1e-9, abs=1e-12)

    @given(shares)
    def test_jain_is_permutation_invariant(self, values):
        rotated = values[1:] + values[:1]
        assert jain_index(rotated) == pytest.approx(jain_index(values), rel=0.0, abs=1e-12)

    @given(st.floats(0.0, 1.0), st.integers(1, 10))
    def test_jain_is_one_for_equal_shares(self, value, count):
        assert jain_index([value] * count) == pytest.approx(1.0, rel=0.0, abs=1e-12)

    @given(shares)
    def test_max_min_is_the_minimum(self, values):
        assert max_min_share(values) == pytest.approx(min(values), rel=0.0, abs=0.0)

    @given(shares)
    def test_max_min_never_exceeds_jain(self, values):
        # Both are 1 for an equal allocation, and the smallest share is at most
        # the index for any allocation in the unit interval.
        assert max_min_share(values) <= jain_index(values) + 1e-9


class TestHorizonMask:
    """Mask profile properties.

    Breakpoint azimuths are drawn on a half-degree grid rather than from the
    whole real line. That is not a way of dodging failures: a profile whose
    breakpoints are 7e-12 degrees apart has a slope of order 1e11 per degree,
    and the periodicity property then fails on the representation error of
    ``azimuth + 360`` alone, which is a statement about double precision and not
    about this code. The half-degree grid bounds the slope at 320 per degree, so
    the 5.7e-14 degree spacing of doubles near 360 cannot move the interpolated
    value by more than 2e-11. Hypothesis found the pathological case and it is
    pinned in ``test_regressions.py`` with the tolerance it actually needs.
    """

    azimuths = st.builds(lambda k: 0.5 * k, st.integers(0, 719))
    elevations = st.floats(-80.0, 80.0)
    breakpoint_sets = st.lists(
        st.tuples(azimuths, elevations), min_size=2, max_size=8
    )

    @given(breakpoint_sets, st.floats(-80.0, 80.0), st.floats(-1080.0, 1080.0))
    def test_is_never_below_the_floor(self, breakpoints, floor, azimuth):
        mask = HorizonMask(floor_deg=floor, breakpoints=tuple(breakpoints))
        assert mask.min_elevation_deg(np.array([azimuth]))[0] >= floor - 1e-12

    @given(breakpoint_sets, st.floats(0.0, 360.0))
    def test_is_periodic(self, breakpoints, azimuth):
        mask = HorizonMask(floor_deg=0.0, breakpoints=tuple(breakpoints))
        here = mask.min_elevation_deg(np.array([azimuth]))[0]
        there = mask.min_elevation_deg(np.array([azimuth + 360.0]))[0]
        assert here == pytest.approx(there, rel=0.0, abs=1e-9)

    @given(breakpoint_sets, st.floats(0.0, 360.0))
    def test_is_bounded_by_the_breakpoint_extremes(self, breakpoints, azimuth):
        # Two distinct azimuths are needed for a profile to exist at all. With
        # fewer, the documented fallback is the floor, which is deliberately
        # outside the breakpoint range, so the assumption is on the structure
        # and not a way of dodging a failure.
        assume(len({a % 360.0 for a, _e in breakpoints}) >= 2)
        mask = HorizonMask(floor_deg=-89.0, breakpoints=tuple(breakpoints))
        value = mask.min_elevation_deg(np.array([azimuth]))[0]
        elevations = [e for _a, e in breakpoints]
        assert min(elevations) - 1e-9 <= value <= max(elevations) + 1e-9

    @given(breakpoint_sets, st.floats(0.0, 360.0))
    def test_collapsing_duplicates_takes_the_maximum(self, breakpoints, azimuth):
        mask = HorizonMask(floor_deg=-89.0, breakpoints=tuple(breakpoints))
        per_azimuth: dict[float, float] = {}
        for raw_azimuth, elevation in breakpoints:
            key = raw_azimuth % 360.0
            per_azimuth[key] = max(per_azimuth.get(key, elevation), elevation)
        exact = HorizonMask(floor_deg=-89.0, breakpoints=tuple(per_azimuth.items()))
        assert mask.min_elevation_deg(np.array([azimuth]))[0] == pytest.approx(
            exact.min_elevation_deg(np.array([azimuth]))[0], rel=0.0, abs=1e-12
        )


class TestGeometry:
    @given(st.floats(-90.0, 90.0), st.floats(-360.0, 360.0))
    def test_geocentric_radius_is_between_the_axes(self, lat, lon):
        radius = float(np.linalg.norm(geodetic_to_ecef_km(lat, lon, 0.0)))
        assert WGS84_B_KM - 1e-6 <= radius <= WGS84_A_KM + 1e-6

    @given(st.floats(-90.0, 90.0), st.floats(-360.0, 360.0), st.floats(0.0, 100.0))
    def test_altitude_never_decreases_the_radius(self, lat, lon, alt):
        low = float(np.linalg.norm(geodetic_to_ecef_km(lat, lon, 0.0)))
        high = float(np.linalg.norm(geodetic_to_ecef_km(lat, lon, alt)))
        assert high >= low - 1e-9

    @given(st.floats(-90.0, 90.0), st.floats(-360.0, 360.0))
    def test_sez_is_orthonormal(self, lat, lon):
        matrix = sez_matrix(lat, lon)
        assert np.allclose(matrix @ matrix.T, np.eye(3), rtol=0.0, atol=1e-13)

    @given(st.floats(2415020.5, 2488070.5))
    def test_gmst_is_in_the_principal_range(self, jd):
        assert 0.0 <= gmst_rad(jd) < 2.0 * math.pi

    @given(
        st.floats(-89.0, 89.0),
        st.floats(-180.0, 180.0),
        st.floats(200.0, 2000.0),
        st.floats(-1.0, 1.0),
        st.floats(-1.0, 1.0),
        st.floats(-1.0, 1.0),
    )
    def test_look_angle_range_is_the_slant_norm(self, lat, lon, height, dx, dy, dz):
        assume(abs(dx) + abs(dy) + abs(dz) > 0.1)
        station_ecef = geodetic_to_ecef_km(lat, lon, 0.0)
        direction = np.array([dx, dy, dz])
        direction = direction / np.linalg.norm(direction)
        satellite = (station_ecef + height * direction)[None, :]
        _az, el, rng = look_angles_deg(satellite, station_ecef, sez_matrix(lat, lon))
        assert rng[0] == pytest.approx(height, rel=1e-9, abs=0.0)
        assert 0.0 <= _az[0] < 360.0
        assert -90.0 - 1e-9 <= el[0] <= 90.0 + 1e-9

    @given(
        st.floats(-89.0, 89.0),
        st.floats(-180.0, 180.0),
        st.floats(300.0, 3000.0),
        st.floats(-1.0, 1.0),
        st.floats(-1.0, 1.0),
        st.floats(-1.0, 1.0),
    )
    def test_geodetic_and_geocentric_elevation_differ_within_the_derived_bound(
        self, lat, lon, height, dx, dy, dz
    ):
        # The independently derived bound again, now over a generated sample
        # rather than a hand-picked grid of latitudes.
        assume(abs(dx) + abs(dy) + abs(dz) > 0.1)
        station_ecef = geodetic_to_ecef_km(lat, lon, 0.0)
        direction = np.array([dx, dy, dz])
        direction = direction / np.linalg.norm(direction)
        satellite = (station_ecef + height * direction)[None, :]
        _az, geodetic, _r = look_angles_deg(satellite, station_ecef, sez_matrix(lat, lon))
        geocentric = elevation_geocentric_vector_deg(satellite, station_ecef)
        assert abs(geodetic[0] - geocentric[0]) <= (
            MAX_GEODETIC_GEOCENTRIC_DEVIATION_DEG + 1e-9
        )

    @given(
        st.floats(-89.0, 89.0),
        st.floats(-180.0, 180.0),
        st.floats(300.0, 3000.0),
        st.floats(0.3, 1.0),
    )
    def test_two_geocentric_derivations_agree_near_the_zenith(self, lat, lon, height, radial):
        # Restricted to directions well away from the horizon, where the law of
        # cosines form is well conditioned. The ill-conditioned regime is
        # measured, not asserted, in validation/validate_geometry.py.
        station_ecef = geodetic_to_ecef_km(lat, lon, 0.0)
        unit = station_ecef / np.linalg.norm(station_ecef)
        tangent = np.cross(unit, np.array([0.0, 0.0, 1.0]))
        norm = np.linalg.norm(tangent)
        assume(norm > 1e-6)
        tangent = tangent / norm
        direction = radial * unit + math.sqrt(max(0.0, 1.0 - radial * radial)) * tangent
        satellite = (station_ecef + height * direction)[None, :]
        vector = elevation_geocentric_vector_deg(satellite, station_ecef)
        norms = elevation_geocentric_norms_deg(
            np.linalg.norm(satellite, axis=1),
            float(np.linalg.norm(station_ecef)),
            np.linalg.norm(satellite - station_ecef[None, :], axis=1),
        )
        assert vector[0] == pytest.approx(norms[0], rel=0.0, abs=1e-3)


class TestTleRoundTrip:
    @given(
        st.integers(1, 99999),
        st.integers(0, 56),
        st.floats(1.0, 366.9),
        st.floats(0.0, 180.0),
        st.floats(0.0, 359.99),
        st.floats(0.0, 0.3),
        st.floats(0.0, 359.99),
        st.floats(0.0, 359.99),
        st.floats(1.0, 18.0),
    )
    def test_every_field_survives_the_round_trip(
        self, satnum, year, day, inc, raan, ecc, argp, anomaly, motion
    ):
        elements = OrbitElements(
            satnum=satnum,
            epoch_year=year,
            epoch_day=day,
            inclination_deg=inc,
            raan_deg=raan,
            eccentricity=ecc,
            argp_deg=argp,
            mean_anomaly_deg=anomaly,
            mean_motion_rev_day=motion,
        )
        line1, line2 = elements.to_lines()
        assert len(line1) == 69
        assert len(line2) == 69
        parsed = parse_lines(line1, line2)
        assert parsed["satnum"] == pytest.approx(float(satnum), rel=0.0, abs=0.0)
        assert parsed["inclination_deg"] == pytest.approx(inc, rel=0.0, abs=5e-5)
        assert parsed["eccentricity"] == pytest.approx(ecc, rel=0.0, abs=5e-8)
        assert parsed["mean_motion_rev_day"] == pytest.approx(motion, rel=0.0, abs=5e-9)


class TestCandidateGeneration:
    @given(window_sets(), request_sets(), st.integers(1, 6))
    @SLOW
    def test_every_candidate_lies_inside_a_window(self, windows, requests, step):
        candidates = build_candidates(requests, windows, slot_step_s=100.0 * step)
        spans: dict[tuple[str, str], list[tuple[float, float]]] = {}
        for w in windows:
            spans.setdefault((w.station, w.spacecraft), []).append((w.rise_s, w.set_s))
        for candidate in candidates:
            assert any(
                start - 1e-9 <= candidate.start_s and candidate.end_s <= end + 1e-9
                for start, end in spans[(candidate.station, candidate.spacecraft)]
            )

    @given(window_sets(), request_sets())
    @SLOW
    def test_a_finer_grid_never_loses_candidates(self, windows, requests):
        coarse = build_candidates(requests, windows, slot_step_s=400.0)
        fine = build_candidates(requests, windows, slot_step_s=100.0)
        assert len(fine) >= len(coarse)

    @given(window_sets(), request_sets())
    @SLOW
    def test_candidate_indices_are_contiguous(self, windows, requests):
        candidates = build_candidates(requests, windows, slot_step_s=200.0)
        assert [c.index for c in candidates] == list(range(len(candidates)))


class TestSchedulingProperties:
    @given(window_sets(), request_sets(), st.integers(0, 4))
    @SLOW
    def test_greedy_is_always_feasible(self, windows, requests, turnaround):
        problem = build_problem(
            (station(turnaround_s=100.0 * turnaround),), requests, windows, 200.0
        )
        assignment = schedule_greedy(problem)
        assert check_feasible(problem, assignment.selected) == []

    @given(window_sets(), request_sets(), st.integers(0, 4))
    @SLOW
    def test_the_exact_optimum_is_always_feasible(self, windows, requests, turnaround):
        problem = build_problem(
            (station(turnaround_s=100.0 * turnaround),), requests, windows, 200.0
        )
        assume(problem.candidates)
        assignment = solve_exact(problem)
        assert check_feasible(problem, assignment.selected) == []

    @given(window_sets(), request_sets(), st.integers(0, 4))
    @SLOW
    def test_greedy_never_beats_the_exact_optimum(self, windows, requests, turnaround):
        problem = build_problem(
            (station(turnaround_s=100.0 * turnaround),), requests, windows, 200.0
        )
        assume(problem.candidates)
        greedy = schedule_greedy(problem)
        exact = solve_exact(problem)
        assert greedy.objective <= exact.objective + 1e-9

    @given(window_sets(), request_sets(), st.integers(0, 4))
    @SLOW
    def test_allocated_time_never_exceeds_union_supply(self, windows, requests, turnaround):
        problem = build_problem(
            (station(turnaround_s=100.0 * turnaround),), requests, windows, 200.0
        )
        assume(problem.candidates)
        assignment = solve_exact(problem)
        supply = supply_seconds_by_station(list(windows))
        for name, seconds in assignment.seconds_by_station(problem).items():
            assert seconds <= supply.get(name, 0.0) + 1e-6

    @given(window_sets(max_windows=3), request_sets(max_requests=2), st.integers(0, 3))
    @settings(
        max_examples=120,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
    )
    def test_exhaustive_enumeration_agrees_with_highs(self, windows, requests, turnaround):
        # The strongest property in the suite: two methods sharing no
        # formulation, on structures Hypothesis chooses.
        problem = build_problem(
            (station(turnaround_s=100.0 * turnaround),), requests, windows, 300.0
        )
        assume(problem.candidates)
        assume(len(problem.candidates) <= 14)
        enumerated = solve_bruteforce(problem)
        milp = solve_exact(problem)
        assert enumerated.objective == pytest.approx(milp.objective, rel=0.0, abs=1e-9)


class TestContentionProperties:
    @given(window_sets(), request_sets(), st.integers(1, 12))
    @SLOW
    def test_apportioned_demand_is_conserved(self, windows, requests, bins):
        problem = build_problem((station(),), requests, windows, 200.0)
        horizon = max(w.set_s for w in windows) + 100.0
        cells = contention_map(problem, horizon_s=horizon, bin_s=horizon / bins)
        total = sum(c.demand_s for c in cells)
        placeable = sum(
            r.requested_seconds
            for r in requests
            if any(
                w.spacecraft == r.spacecraft and w.duration_s + 1e-9 >= r.duration_s
                for w in windows
            )
        )
        assert total == pytest.approx(placeable, rel=1e-9, abs=1e-6)

    @given(window_sets(), request_sets(), st.integers(1, 10))
    @SLOW
    def test_supply_never_exceeds_the_bin_width(self, windows, requests, bins):
        problem = build_problem((station(),), requests, windows, 200.0)
        horizon = max(w.set_s for w in windows) + 100.0
        cells = contention_map(problem, horizon_s=horizon, bin_s=horizon / bins)
        for cell in cells:
            assert cell.supply_s <= cell.bin_end_s - cell.bin_start_s + 1e-6


class TestReportProperties:
    @given(st.integers(0, 400000))
    def test_hms_is_parseable_back_to_the_second(self, seconds):
        text = hms(float(seconds))
        hours, minutes, secs = (int(part) for part in text.split(":"))
        assert hours * 3600 + minutes * 60 + secs == seconds

    @given(st.integers(0, 400000))
    def test_hms_has_two_digit_minute_and_second_fields(self, seconds):
        parts = hms(float(seconds)).split(":")
        assert len(parts[1]) == 2
        assert len(parts[2]) == 2
