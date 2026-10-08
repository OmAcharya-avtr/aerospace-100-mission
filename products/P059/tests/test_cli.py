"""CLI tests: `python -m invariantset`, run in a subprocess."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = str(Path(__file__).resolve().parents[1] / "src")


def run(*args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC + os.pathsep + env.get("PYTHONPATH", "")
    env["MPLBACKEND"] = "Agg"
    return subprocess.run(
        [sys.executable, "-m", "invariantset", *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )


class TestHelpAndVersion:
    def test_help_exits_zero(self):
        res = run("--help")
        assert res.returncode == 0
        assert "maximal robust invariant set" in res.stdout.lower()

    def test_help_names_the_set_it_computes(self):
        res = run("--help")
        assert "MAXIMAL" in res.stdout
        assert "Rakovic" in res.stdout

    def test_version(self):
        res = run("--version")
        assert res.returncode == 0
        assert "0.1.0" in res.stdout

    @pytest.mark.parametrize(
        "command", ["systems", "invariant", "tolerance-sweep", "algebra"]
    )
    def test_each_subcommand_has_help(self, command):
        res = run(command, "--help")
        assert res.returncode == 0

    def test_no_subcommand_is_an_error(self):
        res = run()
        assert res.returncode != 0


class TestSystems:
    def test_plain_listing(self):
        res = run("systems")
        assert res.returncode == 0
        assert "attitude_loop" in res.stdout
        assert "scalar_empty" in res.stdout

    def test_json_listing(self):
        res = run("systems", "--json")
        assert res.returncode == 0
        rows = json.loads(res.stdout)
        assert len(rows) == 8
        assert {r["name"] for r in rows} >= {"attitude_loop", "nilpotent_2d"}


class TestInvariant:
    def test_converging_system_exits_zero(self):
        res = run("invariant", "--system", "nilpotent_2d")
        assert res.returncode == 0
        assert "converged" in res.stdout
        assert "volume of returned set     : 3.600000000" in res.stdout

    def test_empty_system_exits_zero_and_says_empty(self):
        res = run("invariant", "--system", "scalar_empty")
        assert res.returncode == 0
        assert "EMPTY" in res.stdout

    def test_non_convergence_exits_two(self):
        res = run(
            "invariant", "--system", "slow_pair", "--max-iter", "5", "--no-geometry"
        )
        assert res.returncode == 2
        assert "DID NOT CONVERGE" in res.stdout
        assert "not S_inf" in res.stdout

    def test_json_payload(self):
        res = run("invariant", "--system", "attitude_loop", "--json")
        assert res.returncode == 0
        payload = json.loads(res.stdout)
        assert payload["termination"] == "converged"
        assert payload["iterations"] == 5
        assert payload["facets"] == 16
        assert payload["independently_verified_invariant"] is True
        assert payload["volume"] == pytest.approx(0.538347284806, rel=1e-9)
        assert len(payload["history"]) == 5

    def test_redundancy_tolerance_is_honoured(self):
        res = run(
            "invariant",
            "--system",
            "attitude_loop",
            "--redundancy-tol",
            "4e-3",
            "--max-iter",
            "40",
            "--no-geometry",
            "--json",
        )
        assert res.returncode == 2
        assert json.loads(res.stdout)["termination"] == "iteration_cap"

    def test_unknown_system_is_rejected(self):
        res = run("invariant", "--system", "no_such_system")
        assert res.returncode != 0
        assert "invalid choice" in res.stderr


class TestToleranceSweep:
    def test_sweep_reports_the_answer_change(self):
        res = run(
            "tolerance-sweep",
            "--system",
            "attitude_loop",
            "--tolerances",
            "1e-9",
            "3e-3",
            "4e-3",
            "--max-iter",
            "40",
        )
        assert res.returncode == 0
        assert "converged" in res.stdout
        assert "iteration_cap" in res.stdout

    def test_sweep_json(self):
        res = run(
            "tolerance-sweep",
            "--system",
            "decoupled_2d",
            "--tolerances",
            "1e-9",
            "1e-6",
            "--max-iter",
            "10",
            "--json",
        )
        assert res.returncode == 0
        rows = json.loads(res.stdout)
        assert len(rows) == 2
        assert all(r["termination"] == "converged" for r in rows)


class TestAlgebra:
    def test_asymmetry_is_printed(self):
        res = run("algebra")
        assert res.returncode == 0
        assert "area deficit                 = 0.040000000 (8.0000 % of P)" in res.stdout
        assert "the identity people assume, FALSE" in res.stdout
        assert "identity (I4), holds" in res.stdout

    def test_algebra_json(self):
        res = run("algebra", "--json")
        assert res.returncode == 0
        payload = json.loads(res.stdout)
        assert payload["area_P"] == pytest.approx(0.5, abs=1e-12)
        assert payload["area_erode_then_dilate"] == pytest.approx(0.46, abs=1e-9)
        assert payload["erode_then_dilate_subset_of_P"] is True
        assert payload["dilate_then_erode_equals_P"] is True
        assert payload["area_deficit_fraction"] == pytest.approx(0.08, abs=1e-9)


class TestNoPrintInLibraryCode:
    def test_only_main_writes_to_stdout(self):
        package = Path(__file__).resolve().parents[1] / "src" / "invariantset"
        offenders = []
        for path in sorted(package.glob("*.py")):
            if path.name == "__main__.py":
                continue
            for lineno, line in enumerate(path.read_text().splitlines(), start=1):
                stripped = line.strip()
                if stripped.startswith("print(") or ".write(" in stripped:
                    offenders.append(f"{path.name}:{lineno}")
        assert offenders == []
