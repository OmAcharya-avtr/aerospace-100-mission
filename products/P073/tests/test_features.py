"""Pre-scheduling features and the design matrix."""

from __future__ import annotations

import numpy as np
import pytest

from contactsched.features import (
    BASELINE_FEATURES,
    FEATURE_NAMES,
    design_matrix,
    request_features,
)
from contactsched.problem import ContactRequest, build_problem
from contactsched.station import GroundStation, HorizonMask
from contactsched.visibility import ContactWindow


def station(name: str = "A1", turnaround_s: float = 0.0) -> GroundStation:
    return GroundStation(name, 0.0, 0.0, 0.0, HorizonMask(floor_deg=5.0), turnaround_s)


def window(spacecraft: str, rise: float, close: float, station_name: str = "A1") -> ContactWindow:
    return ContactWindow(
        station_name, spacecraft, "M", rise, close, 45.0, 0.5 * (rise + close), 30.0
    )


class TestFeatureNames:
    def test_names_are_unique(self):
        assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)

    def test_baseline_features_are_a_subset(self):
        assert set(BASELINE_FEATURES).issubset(set(FEATURE_NAMES))

    def test_extractor_returns_exactly_the_declared_names(self, toy_problem):
        features = request_features(toy_problem, horizon_s=5000.0)
        for values in features.values():
            assert tuple(values) == FEATURE_NAMES


class TestFeatureValues:
    @pytest.fixture
    def problem(self):
        # SC-1 has two 100 s windows; SC-2 has one. R1 declares two contacts of
        # 100 s, R2 one. The second window of SC-1 overlaps SC-2's, so R1 and R2
        # are competitors over 100 s of shared antenna time.
        requests = (
            ContactRequest("R1", "M1", "SC-1", 1, 100.0, min_contacts=2, max_contacts=2),
            ContactRequest("R2", "M2", "SC-2", 3, 100.0, min_contacts=1, max_contacts=1),
        )
        windows = (
            window("SC-1", 0.0, 100.0),
            window("SC-1", 600.0, 700.0),
            window("SC-2", 600.0, 700.0),
        )
        return build_problem((station(),), requests, windows, 100.0)

    def test_priority_and_weight(self, problem):
        features = request_features(problem, horizon_s=1000.0)
        assert features["R1"]["priority"] == pytest.approx(1.0, rel=0.0, abs=0.0)
        assert features["R1"]["weight"] == pytest.approx(1.0, rel=0.0, abs=0.0)
        assert features["R2"]["priority"] == pytest.approx(3.0, rel=0.0, abs=0.0)
        assert features["R2"]["weight"] == pytest.approx(0.25, rel=0.0, abs=0.0)

    def test_window_and_candidate_counts(self, problem):
        features = request_features(problem, horizon_s=1000.0)
        assert features["R1"]["n_windows"] == pytest.approx(2.0, rel=0.0, abs=0.0)
        assert features["R1"]["n_candidates"] == pytest.approx(2.0, rel=0.0, abs=0.0)
        assert features["R2"]["n_windows"] == pytest.approx(1.0, rel=0.0, abs=0.0)

    def test_usable_seconds(self, problem):
        features = request_features(problem, horizon_s=1000.0)
        assert features["R1"]["usable_seconds"] == pytest.approx(200.0, rel=0.0, abs=0.0)
        assert features["R2"]["usable_seconds"] == pytest.approx(100.0, rel=0.0, abs=0.0)

    def test_demand_over_usable(self, problem):
        # R1 needs 200 s and has 200 s usable: ratio 1. R2 needs 100 s of 100 s.
        features = request_features(problem, horizon_s=1000.0)
        assert features["R1"]["demand_over_usable"] == pytest.approx(1.0, rel=0.0, abs=1e-15)
        assert features["R2"]["demand_over_usable"] == pytest.approx(1.0, rel=0.0, abs=1e-15)

    def test_competitor_overlap_is_the_shared_antenna_time(self, problem):
        features = request_features(problem, horizon_s=1000.0)
        assert features["R1"]["competitor_count"] == pytest.approx(1.0, rel=0.0, abs=0.0)
        assert features["R1"]["competitor_overlap_s"] == pytest.approx(
            100.0, rel=0.0, abs=1e-12
        )

    def test_higher_priority_overlap_is_asymmetric(self, problem):
        features = request_features(problem, horizon_s=1000.0)
        # R1 is the highest priority, so nothing above it.
        assert features["R1"]["higher_priority_competitor_count"] == pytest.approx(
            0.0, rel=0.0, abs=0.0
        )
        assert features["R1"]["higher_priority_overlap_fraction"] == pytest.approx(
            0.0, rel=0.0, abs=0.0
        )
        # R2 is below R1 and shares all of its overlap with it.
        assert features["R2"]["higher_priority_competitor_count"] == pytest.approx(
            1.0, rel=0.0, abs=0.0
        )
        assert features["R2"]["higher_priority_overlap_fraction"] == pytest.approx(
            1.0, rel=0.0, abs=1e-12
        )

    def test_standalone_capacity_and_deficit(self, problem):
        features = request_features(problem, horizon_s=1000.0)
        # R1 has two disjoint placements and declares two: no deficit.
        assert features["R1"]["standalone_capacity"] == pytest.approx(2.0, rel=0.0, abs=0.0)
        assert features["R1"]["capacity_deficit"] == pytest.approx(0.0, rel=0.0, abs=0.0)

    def test_capacity_deficit_is_positive_when_over_asked(self):
        requests = (ContactRequest("R1", "M1", "SC-1", 1, 100.0, min_contacts=3, max_contacts=3),)
        windows = (window("SC-1", 0.0, 100.0),)
        problem = build_problem((station(),), requests, windows, 100.0)
        features = request_features(problem, horizon_s=1000.0)
        assert features["R1"]["capacity_deficit"] == pytest.approx(2.0, rel=0.0, abs=0.0)

    def test_window_slack_is_the_mean_duration_ratio(self, problem):
        features = request_features(problem, horizon_s=1000.0)
        assert features["R1"]["window_slack"] == pytest.approx(1.0, rel=0.0, abs=1e-15)

    def test_mean_peak_elevation_is_carried(self, problem):
        features = request_features(problem, horizon_s=1000.0)
        assert features["R1"]["mean_max_elevation_deg"] == pytest.approx(
            45.0, rel=0.0, abs=1e-12
        )

    def test_instance_level_features_are_shared(self, problem):
        features = request_features(problem, horizon_s=1000.0)
        assert features["R1"]["instance_requests"] == pytest.approx(2.0, rel=0.0, abs=0.0)
        assert features["R1"]["instance_demand_over_supply"] == pytest.approx(
            features["R2"]["instance_demand_over_supply"], rel=0.0, abs=0.0
        )

    def test_n_allowed_stations_counts_the_restriction(self):
        requests = (
            ContactRequest("R1", "M1", "SC-1", 1, 100.0, 1, 1, stations=("A1",)),
            ContactRequest("R2", "M2", "SC-1", 1, 100.0, 1, 1),
        )
        windows = (window("SC-1", 0.0, 100.0, "A1"),)
        problem = build_problem((station("A1"), station("A2")), requests, windows, 100.0)
        features = request_features(problem, horizon_s=1000.0)
        assert features["R1"]["n_allowed_stations"] == pytest.approx(1.0, rel=0.0, abs=0.0)
        assert features["R2"]["n_allowed_stations"] == pytest.approx(2.0, rel=0.0, abs=0.0)


class TestDesignMatrix:
    def test_shape_and_row_order(self, toy_problem):
        features = request_features(toy_problem, horizon_s=5000.0)
        matrix = design_matrix(features, ["R2", "R1"])
        assert matrix.shape == (2, len(FEATURE_NAMES))
        priority_column = FEATURE_NAMES.index("priority")
        assert matrix[0, priority_column] == pytest.approx(2.0, rel=0.0, abs=0.0)
        assert matrix[1, priority_column] == pytest.approx(1.0, rel=0.0, abs=0.0)

    def test_column_order_follows_feature_names(self, toy_problem):
        features = request_features(toy_problem, horizon_s=5000.0)
        matrix = design_matrix(features, ["R1"])
        for column, name in enumerate(FEATURE_NAMES):
            assert matrix[0, column] == pytest.approx(
                features["R1"][name] if np.isfinite(features["R1"][name]) else 1.0e6,
                rel=0.0,
                abs=0.0,
            )

    def test_infinities_become_the_declared_sentinel(self):
        requests = (ContactRequest("R1", "M1", "SC-1", 1, 1000.0, min_contacts=1),)
        windows = (window("SC-1", 0.0, 300.0),)
        problem = build_problem((station(),), requests, windows, 100.0)
        features = request_features(problem, horizon_s=1000.0)
        assert features["R1"]["demand_over_usable"] == float("inf")
        matrix = design_matrix(features, ["R1"])
        assert np.all(np.isfinite(matrix))
        column = FEATURE_NAMES.index("demand_over_usable")
        assert matrix[0, column] == pytest.approx(1.0e6, rel=0.0, abs=0.0)

    def test_empty_order_gives_an_empty_matrix(self, toy_problem):
        features = request_features(toy_problem, horizon_s=5000.0)
        assert design_matrix(features, []).shape == (0,)
