"""CLI tests. The CLI writes to a stream, so every assertion is on text."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from regressselect.__main__ import build_parser, main

HISTORY = "fixtures/history/history.json"
SAMPLE = "fixtures/sample_project"


def _run(argv: list[str]) -> tuple[int, str]:
    buffer = io.StringIO()
    code = main(argv, out=buffer)
    return code, buffer.getvalue()


def test_help_exits_zero():
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["--help"])
    assert exc.value.code == 0


def test_a_missing_subcommand_is_a_usage_error():
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args([])
    assert exc.value.code == 2


def test_graph_lists_the_modules_and_the_selection(product_root: Path, monkeypatch):
    monkeypatch.chdir(product_root)
    code, text = _run(["graph", SAMPLE, "--changed", "sampleproj.stats"])
    assert code == 0
    assert "sampleproj.filters -> ['sampleproj', 'sampleproj.stats']" in text
    assert "2/7 test modules (0.2857)" in text
    assert "tests.test_filters" in text


def test_graph_on_the_package_init_selects_every_test_module(product_root: Path, monkeypatch):
    monkeypatch.chdir(product_root)
    code, text = _run(["graph", SAMPLE, "--changed", "sampleproj"])
    assert code == 0
    assert "7/7 test modules (1.0000)" in text


def test_select_learned_marks_a_budget(product_root: Path, monkeypatch):
    monkeypatch.chdir(product_root)
    code, text = _run(
        [
            "select",
            SAMPLE,
            "--history",
            HISTORY,
            "--changed",
            "sampleproj.quat",
            "--churn",
            "sampleproj.quat=31",
            "--budget",
            "0.25",
            "--strategy",
            "learned",
        ]
    )
    assert code == 0
    assert "running 6/24 nodes" in text
    assert text.count("  run ") == 6
    assert text.count("  skip ") == 18


def test_select_dependency_ignores_the_budget(product_root: Path, monkeypatch):
    monkeypatch.chdir(product_root)
    code, text = _run(
        [
            "select",
            SAMPLE,
            "--history",
            HISTORY,
            "--changed",
            "sampleproj.orbit",
            "--budget",
            "0.05",
            "--strategy",
            "dependency",
        ]
    )
    assert code == 0
    assert "the budget does not apply" in text
    assert "3/24 nodes (0.1250)" in text


def test_select_history_strategy_runs(product_root: Path, monkeypatch):
    monkeypatch.chdir(product_root)
    code, text = _run(
        [
            "select",
            SAMPLE,
            "--history",
            HISTORY,
            "--changed",
            "sampleproj.stats",
            "--strategy",
            "history",
        ]
    )
    assert code == 0
    assert "strategy=history" in text


def test_select_rejects_malformed_churn(product_root: Path, monkeypatch):
    monkeypatch.chdir(product_root)
    code, _ = _run(
        [
            "select",
            SAMPLE,
            "--history",
            HISTORY,
            "--changed",
            "sampleproj.quat",
            "--churn",
            "sampleproj.quat",
        ]
    )
    assert code == 2


def test_benchmark_table_and_json(product_root: Path, monkeypatch):
    monkeypatch.chdir(product_root)
    code, text = _run(["benchmark", SAMPLE, "--history", HISTORY, "--budgets", "0.25", "1.0"])
    assert code == 0
    assert "37 evaluation revisions with a failure" in text
    assert "dependency" in text and "history" in text and "learned" in text

    code, raw = _run(
        ["benchmark", SAMPLE, "--history", HISTORY, "--budgets", "1.0", "--json"]
    )
    assert code == 0
    payload = json.loads(raw)
    assert set(payload) == {"dependency", "history", "learned"}
    assert payload["learned"]["points"][0]["safety_loss"] == pytest.approx(0.0, abs=1e-12)


def test_apfd_subcommand_known_answer(tmp_path: Path):
    order = tmp_path / "order.json"
    detecting = tmp_path / "det.json"
    order.write_text(json.dumps(["t1", "t2", "t3", "t4", "t5"]))
    detecting.write_text(json.dumps({"f1": ["t1"], "f2": ["t3"]}))
    code, text = _run(["apfd", "--order", str(order), "--detecting", str(detecting)])
    assert code == 0
    assert "APFD = 0.700000" in text
    assert "(n=5, m=2)" in text


def test_apfd_subcommand_reports_an_undetected_fault_as_an_error(tmp_path: Path):
    order = tmp_path / "order.json"
    detecting = tmp_path / "det.json"
    order.write_text(json.dumps(["t1"]))
    detecting.write_text(json.dumps({"f1": ["nope"]}))
    code, _ = _run(["apfd", "--order", str(order), "--detecting", str(detecting)])
    assert code == 2


def test_missing_history_file_is_an_error(product_root: Path, monkeypatch):
    monkeypatch.chdir(product_root)
    code, _ = _run(["benchmark", SAMPLE, "--history", "fixtures/history/nope.json"])
    assert code == 2
