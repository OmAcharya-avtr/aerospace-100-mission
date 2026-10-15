"""Tests for the committed history fixture, its loader and the test inventory."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from regressselect.history import History, Revision, load_history, save_history
from regressselect.inventory import collect_tests, module_of_node


def test_committed_history_has_the_declared_shape(history):
    assert len(history) == 320
    assert len(history.test_ids) == 24
    assert history.seed == 20261015
    assert history.project_root == "fixtures/sample_project"


def test_committed_history_regression_count_is_the_measured_one(history):
    # Measured by fixtures/regenerate_fixtures.py at seed 20261015.
    assert history.regression_count == 116


def test_every_failed_test_is_in_the_declared_inventory(history):
    known = set(history.test_ids)
    for rev in history.revisions:
        assert set(rev.failed_tests) <= known, rev.revision


def test_every_changed_module_has_a_churn_entry(history):
    for rev in history.revisions:
        assert set(rev.changed_modules) == set(rev.churn), rev.revision
        assert all(v >= 1 for v in rev.churn.values())


def test_revision_ids_are_unique_and_ordered(history):
    ids = [r.revision for r in history.revisions]
    assert len(set(ids)) == len(ids)
    assert ids == sorted(ids)


def test_churn_helpers_known_answers():
    rev = Revision("r1", ("a", "b"), {"a": 3, "b": 11}, ("t1",))
    assert rev.total_churn() == 14
    assert rev.max_churn() == 11
    assert rev.is_regression is True
    assert Revision("r2", (), {}, ()).max_churn() == 0
    assert Revision("r2", (), {}, ()).is_regression is False


def test_chronological_split_is_a_partition_in_order(history):
    train, test = history.split_chronological(0.70)
    assert len(train) + len(test) == len(history)
    assert len(train) == 224
    assert train.revisions[-1].revision < test.revisions[0].revision
    assert train.test_ids == history.test_ids


def test_split_rejects_out_of_range_fractions(history):
    for bad in (0.0, 1.0, -0.2, 1.5):
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            history.split_chronological(bad)


def test_split_rejects_a_fraction_that_empties_one_side():
    small = History(("t",), tuple(Revision(f"r{i}", ("m",), {"m": 1}) for i in range(3)))
    with pytest.raises(ValueError, match="empty side"):
        small.split_chronological(0.01)


def test_round_trip_save_and_load(tmp_path: Path, history):
    out = tmp_path / "h.json"
    save_history(history, out)
    again = load_history(out)
    assert again.revisions == history.revisions
    assert again.test_ids == history.test_ids


def test_unknown_schema_is_rejected(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema": "nope/9", "test_ids": [], "revisions": []}))
    with pytest.raises(ValueError, match="unsupported history schema"):
        load_history(bad)


def test_inventory_matches_the_history_test_ids(inventory, history):
    assert set(inventory.node_ids) == set(history.test_ids)


def test_inventory_nodes_in_module(inventory):
    nodes = inventory.nodes_in("tests.test_orbit")
    assert len(nodes) == 3
    assert all(n.startswith("tests/test_orbit.py::") for n in nodes)


def test_module_of_node_known_answers():
    assert module_of_node("tests/test_quat.py::test_rotate_90_about_z") == "tests.test_quat"
    assert module_of_node("a/b/test_c.py::test_d") == "a.b.test_c"


def test_module_of_node_rejects_malformed_ids():
    with pytest.raises(ValueError, match="not a pytest node id"):
        module_of_node("tests/test_quat.py")
    with pytest.raises(ValueError, match="not a python file"):
        module_of_node("tests/test_quat.txt::test_x")


def test_collect_tests_rejects_a_project_without_tests(tmp_path: Path):
    (tmp_path / "tests").mkdir()
    with pytest.raises(ValueError, match="no test functions found"):
        collect_tests(tmp_path)
    with pytest.raises(ValueError, match="test directory not found"):
        collect_tests(tmp_path, test_dir="nope")
