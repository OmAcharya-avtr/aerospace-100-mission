"""Tests for the seeded-fault harness and the committed junit fixture.

The harness runs pytest in a subprocess. Only two faults are materialised and
executed here, chosen to keep the suite under the three-minute budget; the
full fourteen-fault evaluation lives in validation/validate_mutation.py.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from regressselect.mutation import (
    SeededFault,
    load_faults,
    materialise_fault,
    run_suite,
)


def test_fourteen_faults_load_with_the_declared_fields(product_root: Path):
    faults = load_faults(product_root / "fixtures/faults/faults.json")
    assert len(faults) == 14
    assert [f.fault_id for f in faults] == [f"F{i:02d}" for i in range(1, 15)]
    for fault in faults:
        assert fault.module.startswith("sampleproj")
        assert fault.relative_path.endswith(".py")
        assert fault.old_text and fault.new_text
        assert fault.old_text != fault.new_text
        assert fault.operator and fault.description


def test_unknown_fault_schema_is_rejected(tmp_path: Path):
    bad = tmp_path / "f.json"
    bad.write_text(json.dumps({"schema": "nope/1", "faults": []}))
    with pytest.raises(ValueError, match="unsupported fault schema"):
        load_faults(bad)


def test_every_fault_old_text_occurs_exactly_once(product_root: Path, sample_project: Path):
    for fault in load_faults(product_root / "fixtures/faults/faults.json"):
        text = (sample_project / fault.relative_path).read_text(encoding="utf-8")
        assert text.count(fault.old_text) == 1, fault.fault_id


def test_materialise_rejects_a_fault_that_no_longer_applies(
    sample_project: Path, tmp_path: Path
):
    stale = SeededFault(
        fault_id="FXX",
        module="sampleproj.vecops",
        relative_path="sampleproj/vecops.py",
        old_text="this text is not in the file",
        new_text="x",
        operator="none",
        description="stale",
    )
    with pytest.raises(ValueError, match="occurs 0 times"):
        materialise_fault(sample_project, stale, tmp_path / "work")


def test_materialise_rejects_a_missing_target_file(sample_project: Path, tmp_path: Path):
    missing = SeededFault(
        fault_id="FYY",
        module="sampleproj.nope",
        relative_path="sampleproj/nope.py",
        old_text="a",
        new_text="b",
        operator="none",
        description="missing",
    )
    with pytest.raises(ValueError, match="missing file"):
        materialise_fault(sample_project, missing, tmp_path / "work")


def test_clean_sample_project_passes_and_cross_checks(sample_project: Path, tmp_path: Path):
    result = run_suite(sample_project, junit_path=tmp_path / "junit.xml")
    assert result.tests == 24
    assert result.failures == 0
    assert result.errors == 0
    assert result.skipped == 0
    assert result.testcase_count == 24
    assert result.cross_check() is True
    assert result.failing_nodes == ()
    assert len(result.all_nodes) == 24


def test_a_broad_fault_is_detected_by_the_modules_the_graph_predicts(
    product_root: Path, sample_project: Path, tmp_path: Path, graph
):
    # F02 replaces * with + inside vecops.dot. The import graph says
    # test_vecops, test_quat and test_kinematics can reach vecops; nothing
    # else can. The measured failing set must be a subset of that selection.
    fault = next(f for f in load_faults(product_root / "fixtures/faults/faults.json") if f.fault_id == "F02")
    work = materialise_fault(sample_project, fault, tmp_path / "F02")
    result = run_suite(work, junit_path=tmp_path / "F02.xml")
    assert result.cross_check() is True
    assert result.failures > 0
    predicted = graph.select_tests([fault.module])
    failing_modules = {n.split("::")[0][: -len(".py")].replace("/", ".") for n in result.failing_nodes}
    assert failing_modules <= predicted


def test_an_undetected_fault_stays_undetected(
    product_root: Path, sample_project: Path, tmp_path: Path
):
    # F13 deletes half of a length guard that no test exercises. The suite
    # misses it, which is why the full-suite upper bound in VALIDATION.md is
    # 0.7143 and not 1.
    fault = next(f for f in load_faults(product_root / "fixtures/faults/faults.json") if f.fault_id == "F13")
    work = materialise_fault(sample_project, fault, tmp_path / "F13")
    result = run_suite(work, junit_path=tmp_path / "F13.xml")
    assert result.cross_check() is True
    assert result.failing_nodes == ()
    assert result.failures == 0


def test_run_suite_rejects_a_missing_root(tmp_path: Path):
    with pytest.raises(ValueError, match="not a directory"):
        run_suite(tmp_path / "nope")


def test_committed_junit_fixture_is_present_and_parses(product_root: Path):
    path = product_root / "fixtures/sample_project/junit.xml"
    assert path.is_file(), (
        "fixtures/sample_project/junit.xml is a committed deliverable. If this "
        "fails on a fresh clone the .gitignore has eaten it; see README Limitations."
    )
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    declared = sum(int(s.get("tests", 0)) for s in suites)
    elements = len(list(root.iter("testcase")))
    assert declared == elements == 24
    assert sum(int(s.get("failures", 0)) for s in suites) == 0
    assert sum(int(s.get("errors", 0)) for s in suites) == 0


def test_committed_fault_matrix_fixture_is_present_and_consistent(product_root: Path):
    raw = json.loads((product_root / "fixtures/faults/fault_matrix.json").read_text())
    assert raw["schema"] == "regressselect-faultmatrix/1"
    assert len(raw["fault_ids"]) == 14
    assert len(raw["node_ids"]) == 24
    universe = set(raw["node_ids"])
    for fault_id, nodes in raw["detecting"].items():
        assert set(nodes) <= universe, fault_id
    undetected = [f for f in raw["fault_ids"] if not raw["detecting"][f]]
    assert sorted(undetected) == ["F04", "F12", "F13", "F14"]
