"""The command line, in process and in a subprocess.

The subprocess tests give the child a temporary ``HOME`` and ``MPLCONFIGDIR``
outside the repository and set ``PYTHONDONTWRITEBYTECODE``, because a
validation script in an earlier product of this mission set ``HOME`` to the
product root and matplotlib wrote ``.config/matplotlib`` into the repository,
which the release gate fails twice over.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from pointmodel.__main__ import main

ROOT = Path(__file__).resolve().parent.parent


def child_environment() -> dict[str, str]:
    """An environment a spawned interpreter cannot pollute the tree with."""
    scratch = Path(tempfile.mkdtemp(prefix="pointmodel-tests-"))
    environment = dict(os.environ)
    environment.update(
        {
            "HOME": str(scratch),
            "MPLCONFIGDIR": str(scratch / "mpl"),
            "MPLBACKEND": "Agg",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(ROOT / "src"),
        }
    )
    return environment


class TestInProcess:
    """Each command returns zero and prints what it claims to."""

    def test_terms(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["terms"]) == 0
        out = capsys.readouterr().out
        assert "DECLARED POINTING MODEL TERMS" in out
        assert "NPAE" in out

    def test_identify(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["identify", "--geometry", "narrow_el", "--observations", "120"]) == 0
        out = capsys.readouterr().out
        assert "not separable" in out
        assert "terms refused:" in out

    def test_compare(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["compare", "--observations", "120"]) == 0
        out = capsys.readouterr().out
        assert "RANK VERDICT BY DECLARED GEOMETRY" in out
        for name in ("full_sky", "narrow_el", "single_az", "two_az", "low_el"):
            assert name in out

    def test_decompose(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["decompose", "--observations", "120", "--no-robust"]) == 0
        out = capsys.readouterr().out
        assert "POINTING ERROR BUDGET" in out
        assert "TRUTH, for reference" in out
        assert "not a measurement" in out

    def test_decompose_with_outliers_flags_the_disagreement(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert (
            main(
                [
                    "decompose",
                    "--observations",
                    "120",
                    "--outlier-fraction",
                    "0.08",
                ]
            )
            == 0
        )
        out = capsys.readouterr().out
        assert "flagged             : YES" in out

    def test_outliers(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["outliers", "--observations", "120"]) == 0
        out = capsys.readouterr().out
        assert "ROBUST FIT (huber)" in out
        assert "ROBUST FIT (ransac)" in out
        assert "ransac error" in out

    def test_refraction(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["refraction", "--steps", "4"]) == 0
        out = capsys.readouterr().out
        assert "REFRACTION CROSS-CHECK" in out
        assert "No refraction package" in out
        assert "radio dry" in out
        assert "optical 550 nm" in out

    def test_a_bad_geometry_is_rejected_by_argparse(self) -> None:
        with pytest.raises(SystemExit) as raised:
            main(["identify", "--geometry", "equator"])
        assert raised.value.code == 2

    def test_a_bad_value_returns_two_and_prints_to_stderr(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["decompose", "--observations", "120", "--sigma", "-1"]) == 2
        captured = capsys.readouterr()
        assert captured.err.startswith("error: ")

    def test_an_out_of_range_threshold_returns_two(self) -> None:
        assert main(["identify", "--observations", "60", "--null-threshold", "5"]) == 2


class TestSubprocess:
    """The entry point works as ``python -m pointmodel`` from a clean child."""

    def _run(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(  # noqa: S603 - fixed argv, no shell
            [sys.executable, "-m", "pointmodel", *arguments],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
            env=child_environment(),
        )

    def test_version(self) -> None:
        completed = self._run("--version")
        assert completed.returncode == 0
        assert completed.stdout.strip() == "pointmodel 0.1.0"

    def test_compare_exits_zero(self) -> None:
        completed = self._run("compare", "--observations", "100")
        assert completed.returncode == 0, completed.stderr
        assert "single_az" in completed.stdout

    def test_no_command_exits_two(self) -> None:
        completed = self._run()
        assert completed.returncode == 2

    def test_no_cache_directories_were_created(self) -> None:
        # Snapshot and diff rather than an absolute check. The parent pytest
        # process may itself have written __pycache__ into src/ if the suite
        # was run without PYTHONDONTWRITEBYTECODE, and that is the caller's
        # business; what this test owns is that the SUBPROCESS this module
        # spawns adds nothing. The absolute check over the whole tree is made
        # once, at the end of the build, by validation/validate_cli.py.
        names = ("__pycache__", ".pytest_cache", ".ruff_cache", ".hypothesis", ".config")
        before = {name: set(ROOT.rglob(name)) for name in names}
        self._run("terms")
        for name in names:
            appeared = set(ROOT.rglob(name)) - before[name]
            assert not appeared, f"{name} appeared in the tree: {sorted(appeared)}"
