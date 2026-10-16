"""The ``python -m scenariocov`` command line."""

from __future__ import annotations

from pathlib import Path

import pytest

from scenariocov.__main__ import build_parser, main


@pytest.fixture
def paths(fixture_dir: Path) -> dict[str, str]:
    return {
        "space": str(fixture_dir / "space.json"),
        "campaign": str(fixture_dir / "campaign_executed.json"),
        "classes": str(fixture_dir / "classes.json"),
        "pool": str(fixture_dir / "candidate_pool.json"),
    }


class TestParser:
    def test_a_command_is_required(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_unknown_command(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(["invent"])

    def test_report_requires_space_and_campaign(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(["report"])


class TestCommands:
    def test_report(self, paths, capsys) -> None:
        code = main(
            ["report", "--space", paths["space"], "--campaign", paths["campaign"],
             "--classes", paths["classes"]]
        )
        out = capsys.readouterr().out
        assert code == 0
        assert "223/226" in out
        assert "UNCOVERED CLASS" in out

    def test_uncovered(self, paths, capsys) -> None:
        code = main(["uncovered", "--space", paths["space"], "--campaign", paths["campaign"]])
        out = capsys.readouterr().out
        assert code == 0
        # Three planted 2-way gaps plus the trailing summary comment.
        assert len(out.strip().splitlines()) == 4
        assert out.strip().endswith("# 3 uncovered of 226 2-way tuples")

    def test_occupancy(self, paths, capsys) -> None:
        code = main(
            ["occupancy", "--space", paths["space"], "--campaign", paths["campaign"], "--blocks", "2"]
        )
        out = capsys.readouterr().out
        assert code == 0
        assert "108/144" in out
        assert out.count("empty block of") == 2

    def test_propose(self, paths, capsys) -> None:
        code = main(
            ["propose", "--space", paths["space"], "--campaign", paths["campaign"],
             "--pool", paths["pool"], "--count", "3", "--bootstrap", "4"]
        )
        out = capsys.readouterr().out
        assert code == 0
        assert "confidence" in out
        assert len(out.strip().splitlines()) == 4

    def test_propose_without_bootstrap_reports_nan_confidence(self, paths, capsys) -> None:
        main(
            ["propose", "--space", paths["space"], "--campaign", paths["campaign"],
             "--pool", paths["pool"], "--count", "2", "--bootstrap", "0"]
        )
        assert "nan" in capsys.readouterr().out

    def test_benchmark(self, paths, capsys) -> None:
        code = main(
            ["benchmark", "--space", paths["space"], "--campaign", paths["campaign"],
             "--pool", paths["pool"], "--start", "20", "--steps", "10",
             "--pool-size", "60", "--seeds", "3"]
        )
        out = capsys.readouterr().out
        assert code == 0
        for name in ("uniform_random", "greedy_maxmin", "learned_gapfinder"):
            assert name in out
        assert "best mean gain per scenario" in out

    def test_benchmark_with_reference(self, paths, capsys) -> None:
        main(
            ["benchmark", "--space", paths["space"], "--campaign", paths["campaign"],
             "--pool", paths["pool"], "--start", "20", "--steps", "6",
             "--pool-size", "40", "--seeds", "2", "--with-reference"]
        )
        assert "coverage_greedy_reference" in capsys.readouterr().out

    def test_strength_three_report(self, paths, capsys) -> None:
        main(
            ["report", "--space", paths["space"], "--campaign", paths["campaign"],
             "--strength", "3", "--max-uncovered", "3"]
        )
        assert "3-way coverage" in capsys.readouterr().out
