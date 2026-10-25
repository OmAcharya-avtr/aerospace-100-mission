"""Synthetic instance generation, labelling and the instance-level split."""

from __future__ import annotations

import numpy as np
import pytest

from contactsched.dataset import (
    INCLINATION_POOL,
    MISSION_POOL,
    STATION_POOL,
    build_dataset,
    generate_instance,
    instance_rows,
    label_instance,
    problem_of,
    split_by_instance,
)
from contactsched.features import FEATURE_NAMES
from contactsched.problem import check_feasible


class TestPools:
    def test_station_pool_names_are_unique(self):
        names = [row[0] for row in STATION_POOL]
        assert len(set(names)) == len(names)

    def test_station_pool_spans_both_hemispheres(self):
        latitudes = [row[1] for row in STATION_POOL]
        assert min(latitudes) < -20.0
        assert max(latitudes) > 60.0

    def test_inclination_pool_has_polar_and_non_polar_orbits(self):
        assert any(i > 90.0 for i in INCLINATION_POOL)
        assert any(i < 90.0 for i in INCLINATION_POOL)

    def test_mission_pool_is_unique(self):
        assert len(set(MISSION_POOL)) == len(MISSION_POOL)


class TestGenerateInstance:
    def test_is_deterministic_in_the_seed(self):
        first = generate_instance(5)
        second = generate_instance(5)
        assert [s.name for s in first.stations] == [s.name for s in second.stations]
        assert [s.name for s in first.fleet] == [s.name for s in second.fleet]
        assert len(first.problem.candidates) == len(second.problem.candidates)
        assert first.load == pytest.approx(second.load, rel=0.0, abs=0.0)

    def test_different_seeds_give_different_instances(self):
        sizes = {len(generate_instance(seed).problem.candidates) for seed in range(8)}
        assert len(sizes) > 1

    def test_sizes_can_be_forced(self):
        instance = generate_instance(9, n_stations=2, n_spacecraft=3, n_requests=4)
        assert len(instance.stations) == 2
        assert len(instance.fleet) == 3
        assert len(instance.problem.requests) == 4

    def test_station_names_are_unique_within_an_instance(self):
        for seed in range(6):
            names = [s.name for s in generate_instance(seed).stations]
            assert len(set(names)) == len(names)

    def test_spacecraft_names_are_unique_within_an_instance(self):
        for seed in range(6):
            names = [s.name for s in generate_instance(seed).fleet]
            assert len(set(names)) == len(names)

    def test_every_request_names_a_spacecraft_in_the_fleet(self):
        for seed in range(6):
            instance = generate_instance(seed)
            fleet = {s.name for s in instance.fleet}
            for request in instance.problem.requests:
                assert request.spacecraft in fleet

    def test_every_restricted_request_names_antennas_in_the_network(self):
        for seed in range(10):
            instance = generate_instance(seed)
            names = {s.name for s in instance.stations}
            for request in instance.problem.requests:
                if request.stations is not None:
                    assert set(request.stations).issubset(names)

    def test_request_mission_matches_its_spacecraft(self):
        for seed in range(6):
            instance = generate_instance(seed)
            mission_of = {s.name: s.mission for s in instance.fleet}
            for request in instance.problem.requests:
                assert request.mission == mission_of[request.spacecraft]

    def test_windows_lie_inside_the_horizon(self):
        for seed in range(6):
            instance = generate_instance(seed, horizon_s=21600.0)
            for w in instance.problem.windows:
                assert 0.0 <= w.rise_s < w.set_s <= 21600.0 + 1e-9

    def test_problem_of_returns_the_problem(self):
        instance = generate_instance(3)
        assert problem_of(instance) is instance.problem

    def test_load_is_inside_the_declared_range(self):
        for seed in range(20):
            assert 0.7 <= generate_instance(seed).load <= 2.3


class TestLabelInstance:
    def test_labels_are_binary_and_cover_every_request(self):
        instance = generate_instance(7)
        _assignment, labels = label_instance(instance)
        assert set(labels) == {r.request_id for r in instance.problem.requests}
        assert set(labels.values()).issubset({0, 1})

    def test_the_labelling_assignment_is_feasible(self):
        instance = generate_instance(7)
        assignment, _labels = label_instance(instance)
        assert check_feasible(instance.problem, assignment.selected) == []

    def test_a_label_of_one_means_the_exact_optimum_starved_the_request(self):
        instance = generate_instance(7)
        assignment, labels = label_instance(instance)
        counts = assignment.contacts_by_request(instance.problem)
        for request in instance.problem.requests:
            starved = counts[request.request_id] < request.min_contacts
            assert labels[request.request_id] == int(starved)

    def test_labels_are_deterministic(self):
        first = label_instance(generate_instance(11))[1]
        second = label_instance(generate_instance(11))[1]
        assert first == second


class TestInstanceRows:
    def test_one_row_per_request(self):
        instance = generate_instance(7)
        rows = instance_rows(instance)
        assert len(rows) == len(instance.problem.requests)

    def test_rows_carry_every_feature_and_the_label(self):
        rows = instance_rows(generate_instance(7))
        expected = {"seed", "request_id", "mission", "spacecraft", "load", "unsatisfiable"}
        expected |= set(FEATURE_NAMES)
        assert set(rows[0]) == expected

    def test_features_are_finite_or_infinite_but_never_nan(self):
        for row in instance_rows(generate_instance(7)):
            for name in FEATURE_NAMES:
                assert not np.isnan(row[name])


class TestBuildDataset:
    @pytest.fixture(scope="class")
    @staticmethod
    def frame():
        return build_dataset(range(600, 612), horizon_s=21600.0)

    def test_has_one_row_per_request_across_instances(self, frame):
        assert len(frame) == sum(
            len(generate_instance(seed, horizon_s=21600.0).problem.requests)
            for seed in range(600, 612)
        )

    def test_columns_are_in_the_declared_order(self, frame):
        expected = (
            ["seed", "request_id", "mission", "spacecraft", "load"]
            + list(FEATURE_NAMES)
            + ["unsatisfiable"]
        )
        assert list(frame.columns) == expected

    def test_label_is_binary(self, frame):
        assert set(frame["unsatisfiable"].unique()).issubset({0, 1})

    def test_both_classes_are_present(self, frame):
        # If the generator ever stopped producing starvation the benchmark would
        # become meaningless, so this is asserted rather than assumed.
        assert frame["unsatisfiable"].nunique() == 2

    def test_every_seed_appears(self, frame):
        assert sorted(frame["seed"].unique()) == list(range(600, 612))

    def test_rebuilding_gives_identical_numbers(self):
        first = build_dataset(range(700, 704), horizon_s=21600.0)
        second = build_dataset(range(700, 704), horizon_s=21600.0)
        assert first.equals(second)


class TestSplitByInstance:
    def test_splits_on_the_seed(self):
        frame = build_dataset(range(800, 808), horizon_s=21600.0)
        train, test = split_by_instance(frame, range(800, 804), range(804, 808))
        assert set(train["seed"]).isdisjoint(set(test["seed"]))
        assert len(train) + len(test) == len(frame)

    def test_overlapping_ranges_raise(self):
        frame = build_dataset(range(800, 804), horizon_s=21600.0)
        with pytest.raises(ValueError, match="overlap"):
            split_by_instance(frame, range(800, 803), range(802, 804))

    def test_no_request_appears_in_both_halves(self):
        frame = build_dataset(range(800, 808), horizon_s=21600.0)
        train, test = split_by_instance(frame, range(800, 804), range(804, 808))
        train_keys = set(zip(train["seed"], train["request_id"], strict=True))
        test_keys = set(zip(test["seed"], test["request_id"], strict=True))
        assert train_keys.isdisjoint(test_keys)
