"""JSON loading and saving."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scenariocov import (
    ScenarioSpace,
    load_campaign,
    load_class_document,
    load_space,
    save_campaign,
    save_space,
)


class TestLoading:
    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="no such file"):
            load_space(tmp_path / "absent.json")

    def test_space_round_trip(self, tmp_path: Path, rendezvous_space: ScenarioSpace) -> None:
        path = tmp_path / "space.json"
        save_space(rendezvous_space, path)
        assert load_space(path).to_dict() == rendezvous_space.to_dict()

    def test_campaign_round_trip(self, tmp_path: Path) -> None:
        records = [{"a": 1.0, "b": "x"}, {"a": 2.0, "b": "y"}]
        path = tmp_path / "campaign.json"
        save_campaign(records, path, metadata={"seed": 7})
        assert list(load_campaign(path)) == records
        assert json.loads(path.read_text())["metadata"] == {"seed": 7}

    def test_bare_list_is_accepted_as_a_campaign(self, tmp_path: Path) -> None:
        path = tmp_path / "bare.json"
        path.write_text(json.dumps([{"a": 1}]), encoding="utf-8")
        assert load_campaign(path) == ({"a": 1},)

    def test_empty_campaign_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.json"
        path.write_text(json.dumps({"scenarios": []}), encoding="utf-8")
        with pytest.raises(ValueError, match="non-empty 'scenarios'"):
            load_campaign(path)

    def test_non_object_scenario_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"scenarios": [1, 2]}), encoding="utf-8")
        with pytest.raises(ValueError, match="is not an object"):
            load_campaign(path)

    def test_class_document(self, fixture_dir: Path) -> None:
        classes, boundaries = load_class_document(fixture_dir / "classes.json")
        assert len(classes) == 15
        assert len(boundaries) == 4

    def test_class_document_must_declare_something(self, tmp_path: Path) -> None:
        path = tmp_path / "classes.json"
        path.write_text(json.dumps({"classes": [], "boundaries": []}), encoding="utf-8")
        with pytest.raises(ValueError, match="neither classes nor boundaries"):
            load_class_document(path)

    def test_class_document_must_be_an_object(self, tmp_path: Path) -> None:
        path = tmp_path / "classes.json"
        path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        with pytest.raises(ValueError, match="expected an object"):
            load_class_document(path)

    def test_unknown_parameter_kind_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "space.json"
        path.write_text(
            json.dumps({"parameters": [{"name": "x", "kind": "ordinal", "levels": ["a", "b"]}]}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="unknown parameter kind"):
            load_space(path)


class TestCommittedFixtures:
    def test_all_five_fixture_files_exist(self, fixture_dir: Path) -> None:
        for name in (
            "space.json",
            "classes.json",
            "campaign_executed.json",
            "candidate_pool.json",
            "generation_parameters.json",
        ):
            assert (fixture_dir / name).is_file(), name

    def test_campaign_and_pool_sizes(self, rendezvous_campaign, candidate_pool) -> None:
        assert len(rendezvous_campaign) == 240
        assert len(candidate_pool) == 400

    def test_every_record_carries_every_parameter(
        self, rendezvous_space, rendezvous_campaign, candidate_pool
    ) -> None:
        for records in (rendezvous_campaign, candidate_pool):
            for record in records:
                assert set(record) == set(rendezvous_space.names)

    def test_generation_parameters_record_the_seed_and_the_gaps(self, fixture_dir: Path) -> None:
        payload = json.loads((fixture_dir / "generation_parameters.json").read_text())
        assert payload["seed"] == 20261016
        assert set(payload["planted_gaps"]) == {"G1", "G2", "G3", "G4", "G5"}

    def test_no_committed_fixture_exceeds_one_megabyte(self, fixture_dir: Path) -> None:
        for path in fixture_dir.glob("*.json"):
            assert path.stat().st_size < 1_000_000, path
