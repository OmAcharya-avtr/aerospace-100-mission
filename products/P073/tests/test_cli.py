"""The command line interface."""

from __future__ import annotations

import pytest

from contactsched import __version__
from contactsched.__main__ import METHODS, build_parser, main


class TestParser:
    def test_version_exits_zero(self, capsys):
        with pytest.raises(SystemExit) as exit_info:
            build_parser().parse_args(["--version"])
        assert exit_info.value.code == 0
        assert __version__ in capsys.readouterr().out

    def test_help_exits_zero(self):
        with pytest.raises(SystemExit) as exit_info:
            build_parser().parse_args(["--help"])
        assert exit_info.value.code == 0

    def test_no_subcommand_is_an_error(self):
        with pytest.raises(SystemExit) as exit_info:
            build_parser().parse_args([])
        assert exit_info.value.code == 2

    def test_unknown_subcommand_is_an_error(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["optimise"])

    def test_unknown_method_is_rejected(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["schedule", "--method", "annealing"])

    def test_every_declared_method_parses(self):
        for method in METHODS:
            args = build_parser().parse_args(["schedule", "--method", method])
            assert args.method == method


class TestSubcommands:
    def test_info_prints_versions_and_the_research_grade_statement(self, capsys):
        assert main(["info"]) == 0
        out = capsys.readouterr().out
        assert "contactsched" in out
        assert "numpy" in out
        assert "Not flight-qualified" in out

    def test_backends_prints_the_probe(self, capsys):
        assert main(["backends"]) == 0
        out = capsys.readouterr().out
        assert "highs" in out
        assert "pulp" in out

    def test_windows_prints_a_table(self, capsys):
        assert main(["windows", "--seed", "7", "--horizon-h", "6", "--limit", "5"]) == 0
        out = capsys.readouterr().out
        assert "antenna" in out
        assert "windows" in out

    @pytest.mark.parametrize("method", METHODS)
    def test_schedule_runs_for_every_method(self, capsys, method):
        code = main(["schedule", "--seed", "7", "--horizon-h", "6", "--method", method])
        assert code == 0
        assert "contacts scheduled" in capsys.readouterr().out

    def test_fairness_prints_the_accounting(self, capsys):
        assert main(["fairness", "--seed", "7", "--horizon-h", "6"]) == 0
        out = capsys.readouterr().out
        assert "max-min share" in out
        assert "priority inversions" in out

    def test_contention_prints_the_map(self, capsys):
        assert main(["contention", "--seed", "7", "--horizon-h", "6", "--bin-s", "1800"]) == 0
        out = capsys.readouterr().out
        assert "cells" in out

    def test_compare_prints_every_method_and_the_fairness_price(self, capsys):
        assert main(["compare", "--seed", "7", "--horizon-h", "6"]) == 0
        out = capsys.readouterr().out
        assert "greedy optimality gap" in out
        assert "price of enforcing the declared minimum contact counts" in out
        assert "objective_loss" in out

    def test_compare_reports_an_infeasible_fairness_program(self, capsys):
        # Seed 7 over twelve hours has a request asking for more contacts than
        # its passes can carry, so the fairness-constrained program is infeasible
        # and the CLI must say which floor binds rather than printing nothing.
        assert main(["compare", "--seed", "7", "--horizon-h", "12"]) == 0
        out = capsys.readouterr().out
        assert "infeasible" in out

    def test_a_seed_with_no_candidates_exits_two(self, capsys):
        # A one-minute horizon leaves no placeable contact anywhere.
        assert main(["schedule", "--seed", "7", "--horizon-h", "0.0003"]) == 2
        assert "nothing to schedule" in capsys.readouterr().err


class TestModuleEntryPoint:
    def test_module_is_runnable(self):
        import subprocess
        import sys
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        result = subprocess.run(
            [sys.executable, "-m", "contactsched", "--version"],
            capture_output=True,
            text=True,
            cwd=root,
            env={"PYTHONPATH": str(root / "src"), "PATH": "/usr/bin:/bin"},
            check=False,
        )
        assert result.returncode == 0
        assert __version__ in result.stdout
