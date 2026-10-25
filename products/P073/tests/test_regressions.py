"""Pinned counterexamples and pinned measured values.

Two kinds of test live here.

**Hypothesis counterexamples.** ``.hypothesis/`` is gitignored, so a falsifying
example found by one run is lost on a fresh clone. Every counterexample
Hypothesis found during this build is written out here with literal inputs, so
the defect can never return silently.

**Measured values.** Numbers produced by this repository on a named instance.
They are not reference values from the literature and they claim nothing about
the physics; they exist so that a change in behaviour shows up as a test failure
rather than as a quietly different README.
"""

from __future__ import annotations

import numpy as np
import pytest

from contactsched.benchmark import optimality_gap
from contactsched.dataset import generate_instance
from contactsched.exact import FairnessPolicy, solve_exact
from contactsched.fairness import analyse, jain_index
from contactsched.geometry import (
    MAX_GEODETIC_GEOCENTRIC_DEVIATION_DEG,
    geodetic_to_ecef_km,
    look_angles_deg,
    max_geodetic_geocentric_deviation_deg,
    sez_matrix,
)
from contactsched.greedy import schedule_greedy
from contactsched.station import HorizonMask, default_network


class TestHypothesisCounterexamples:
    def test_jain_index_of_the_smallest_positive_double(self):
        # Found by Hypothesis. The un-normalised expression computes
        # sum(v)**2 / (n * sum(v**2)); at v = 5e-324 the sum is non-zero and the
        # sum of squares underflows to exactly 0.0, so the function raised
        # ZeroDivisionError. Normalising by the largest share before squaring
        # fixes it, and the answer for a single share is 1.0 by definition.
        assert jain_index([5e-324]) == pytest.approx(1.0, rel=0.0, abs=0.0)

    def test_jain_index_of_two_equal_denormals(self):
        assert jain_index([5e-324, 5e-324]) == pytest.approx(1.0, rel=0.0, abs=1e-12)

    def test_jain_index_of_a_denormal_beside_a_normal_number(self):
        # Scale invariance means this is the same allocation as [1.0, 0.0] to
        # within rounding, so the index is the lower bound 0.5 for n = 2.
        assert jain_index([1.0, 5e-324]) == pytest.approx(0.5, rel=0.0, abs=1e-12)

    def test_azimuth_stays_below_three_hundred_and_sixty(self):
        # Found by Hypothesis at dy = -2.5878476337795604e-66. The modulo alone
        # does not close the interval: in double precision a tiny negative
        # angle reduced modulo 360 evaluates to exactly 360.0.
        station = geodetic_to_ecef_km(0.0, 0.0, 0.0)
        direction = np.array([0.0, -2.5878476337795604e-66, 1.0])
        direction = direction / np.linalg.norm(direction)
        satellite = (station + 200.0 * direction)[None, :]
        azimuth, _el, _rng = look_angles_deg(satellite, station, sez_matrix(0.0, 0.0))
        assert 0.0 <= azimuth[0] < 360.0

    def test_duplicate_mask_azimuths_are_periodic(self):
        # Found by Hypothesis. Two elevations at one azimuth do not define a
        # function, and numpy.interp broke the tie by position in its input
        # array, so the profile depended on which 360-degree period the query
        # fell in. Duplicates are now collapsed to their maximum.
        mask = HorizonMask(
            floor_deg=0.0,
            breakpoints=((1.1932097083332236, 0.0), (1.1932097083332236, 1.0)),
        )
        here = mask.min_elevation_deg(np.array([1.1932097083332236]))[0]
        there = mask.min_elevation_deg(np.array([1.1932097083332236 + 360.0]))[0]
        assert here == pytest.approx(there, rel=0.0, abs=1e-12)

    def test_duplicate_mask_azimuths_collapse_to_the_maximum(self):
        mask = HorizonMask(
            floor_deg=0.0,
            breakpoints=(
                (1.1932097083332236, 0.0),
                (1.1932097083332236, 1.0),
                (180.0, 0.0),
            ),
        )
        assert mask.min_elevation_deg(np.array([1.1932097083332236]))[0] == pytest.approx(
            1.0, rel=0.0, abs=1e-12
        )

    def test_a_near_degenerate_mask_segment_is_periodic_only_to_its_slope(self):
        # Found by Hypothesis at breakpoints 0.0 and 6.658158661298158e-12 with
        # elevations 0 and 1: a slope of 1.5e11 degrees of elevation per degree
        # of azimuth. Periodicity then fails on the 5.7e-14 degree spacing of
        # doubles near 360 alone, which is a statement about double precision
        # and not about this code. The honest assertion is the one bounded by
        # the slope: the discrepancy cannot exceed the slope times that spacing.
        mask = HorizonMask(
            floor_deg=-80.0, breakpoints=((0.0, 0.0), (6.658158661298158e-12, 1.0))
        )
        azimuth = 6.658158661298158e-12
        here = mask.min_elevation_deg(np.array([azimuth]))[0]
        there = mask.min_elevation_deg(np.array([azimuth + 360.0]))[0]
        slope = 1.0 / 6.658158661298158e-12
        spacing = np.spacing(360.0)
        assert abs(here - there) <= slope * spacing + 1e-9

    def test_the_geodetic_geocentric_constant_is_the_closed_form(self):
        # Found by this module's own known-answer test, not by Hypothesis: a
        # literal 0.19242218 stood in geometry.py and the closed form evaluates
        # to 0.19242430115956036. The bound test that used the constant was
        # failing by 1.0e-06 degrees at the same time.
        assert MAX_GEODETIC_GEOCENTRIC_DEVIATION_DEG == pytest.approx(
            max_geodetic_geocentric_deviation_deg(), rel=1e-12, abs=0.0
        )
        assert MAX_GEODETIC_GEOCENTRIC_DEVIATION_DEG == pytest.approx(
            0.19242430115956036, rel=0.0, abs=1e-15
        )


class TestMeasuredValuesSeed7:
    """Values measured on instance seed 7 over a twelve-hour horizon.

    Reproduce with ``python -m contactsched compare --seed 7 --horizon-h 12``.
    """

    @pytest.fixture(scope="class")
    @staticmethod
    def instance():
        return generate_instance(7, horizon_s=43200.0)

    def test_instance_shape(self, instance):
        assert len(instance.stations) == 4
        assert len(instance.fleet) == 5
        assert len(instance.problem.requests) == 10
        assert len(instance.problem.windows) == 61
        assert len(instance.problem.candidates) == 233

    def test_greedy_objective(self, instance):
        assignment = schedule_greedy(instance.problem)
        assert assignment.objective == pytest.approx(18.25, rel=0.0, abs=1e-12)
        assert assignment.contact_count == 36

    def test_exact_objective(self, instance):
        assignment = solve_exact(instance.problem)
        assert assignment.objective == pytest.approx(18.5, rel=0.0, abs=1e-12)
        assert assignment.contact_count == 39

    def test_optimality_gap(self, instance):
        gap = optimality_gap(
            schedule_greedy(instance.problem), solve_exact(instance.problem)
        )
        assert gap == pytest.approx(0.013513513513513514, rel=1e-9, abs=0.0)

    def test_greedy_fairness(self, instance):
        report = analyse(instance.problem, schedule_greedy(instance.problem))
        assert report.jain == pytest.approx(0.8302121926574603, rel=1e-9, abs=0.0)
        assert report.max_min == pytest.approx(0.0, rel=0.0, abs=0.0)
        assert len(report.starved) == 3
        assert report.total_seconds_short == pytest.approx(4140.0, rel=0.0, abs=1e-9)
        assert report.inversion_count == 1
        assert report.comparable_pairs == 37

    def test_exact_fairness(self, instance):
        report = analyse(instance.problem, solve_exact(instance.problem))
        assert report.jain == pytest.approx(0.8756264236902048, rel=1e-9, abs=0.0)
        assert len(report.starved) == 3
        assert report.total_seconds_short == pytest.approx(4020.0, rel=0.0, abs=1e-9)
        assert report.inversion_count == 4

    def test_the_exact_optimum_raises_jain_above_greedy_but_not_max_min(self, instance):
        greedy = analyse(instance.problem, schedule_greedy(instance.problem))
        exact = analyse(instance.problem, solve_exact(instance.problem))
        # Worth pinning because it is the product's point: a better objective
        # buys a more equal allocation here and still leaves someone on zero.
        assert exact.jain > greedy.jain
        assert exact.max_min == pytest.approx(greedy.max_min, rel=0.0, abs=0.0)

    def test_the_fairness_constrained_program_is_infeasible_and_names_the_floor(
        self, instance
    ):
        assignment = solve_exact(
            instance.problem, policy=FairnessPolicy(enforce_min_contacts=True)
        )
        assert assignment.status == "infeasible"
        joined = " ".join(assignment.notes)
        assert "R07" in joined
        assert "requires 4 contacts but has at most 1" in joined


class TestDeclaredDefaults:
    def test_network_size_and_names(self):
        names = [s.name for s in default_network()]
        assert names == ["SVALBARD-1", "SVALBARD-2", "PUNTA-ARENAS", "KOUROU"]

    def test_turnaround_times(self):
        turnarounds = {s.name: s.turnaround_s for s in default_network()}
        assert turnarounds == {
            "SVALBARD-1": 300.0,
            "SVALBARD-2": 300.0,
            "PUNTA-ARENAS": 420.0,
            "KOUROU": 360.0,
        }
