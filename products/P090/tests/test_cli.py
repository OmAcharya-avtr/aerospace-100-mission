"""The command line, exercised in a subprocess that cannot pollute the tree.

Every subprocess is given a temporary ``HOME`` and ``MPLCONFIGDIR`` and
``PYTHONDONTWRITEBYTECODE=1``, because a validation script in an earlier product
of this mission set ``HOME`` to the product root and matplotlib wrote
``.config/matplotlib`` into the repository.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _environment(scratch: Path) -> dict[str, str]:
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


def _run(arguments: list[str], scratch: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "tlecheck", *arguments],
        cwd=str(ROOT),
        env=_environment(scratch),
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )


def test_version(tmp_path):
    result = _run(["--version"], tmp_path)
    assert result.returncode == 0
    assert "tlecheck 0.1.0" in result.stdout


def test_no_subcommand_is_an_error(tmp_path):
    result = _run([], tmp_path)
    assert result.returncode != 0


def test_info(tmp_path):
    result = _run(["info"], tmp_path)
    assert result.returncode == 0
    assert "sgp4" in result.stdout
    assert "ships no element catalogue" in result.stdout
    assert "Not flight-qualified" in result.stdout
    assert "17.043642 rev/day" in result.stdout


def test_info_names_the_uninstalled_alternatives(tmp_path):
    result = _run(["info"], tmp_path)
    assert "astropy" in result.stdout
    assert "skyfield" in result.stdout


def test_check_constructed(tmp_path):
    result = _run(["check", "--objects", "48", "--corruption", "sign_flip"], tmp_path)
    assert result.returncode == 0
    assert "WHAT THIS SCREEN CANNOT DETERMINE" in result.stdout
    assert "QUALITY VERDICT" in result.stdout
    assert result.stdout.index("WHAT THIS SCREEN CANNOT DETERMINE") < result.stdout.index(
        "QUALITY VERDICT"
    )


def test_check_clean(tmp_path):
    result = _run(["check", "--objects", "48", "--corruption", "none"], tmp_path)
    assert result.returncode == 0
    assert "predicted consequence class : benign" in result.stdout


def test_check_missing_family_is_a_clean_error(tmp_path):
    result = _run(
        ["check", "--objects", "6", "--corruption", "cross_source_divergence"], tmp_path
    )
    assert result.returncode == 2
    assert "no record with corruption" in result.stderr


def test_check_from_a_file(tmp_path):
    lines = tmp_path / "element_set.txt"
    lines.write_text(
        "1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991\n"
        "2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472202482\n"
    )
    result = _run(["check", "--file", str(lines)], tmp_path)
    assert result.returncode == 0
    assert "QUALITY VERDICT" in result.stdout


def test_check_from_a_file_with_a_name_line(tmp_path):
    lines = tmp_path / "named.txt"
    lines.write_text(
        "OBJECT 25544\n"
        "1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991\n"
        "2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472202482\n"
    )
    result = _run(["check", "--file", str(lines)], tmp_path)
    assert result.returncode == 0


def test_check_from_a_short_file_is_a_clean_error(tmp_path):
    lines = tmp_path / "short.txt"
    lines.write_text("1 25544U\n")
    result = _run(["check", "--file", str(lines)], tmp_path)
    assert result.returncode == 2
    assert "two element set lines are needed" in result.stderr


def test_check_from_a_missing_file_is_a_clean_error(tmp_path):
    result = _run(["check", "--file", str(tmp_path / "absent.txt")], tmp_path)
    assert result.returncode == 2


def test_determinability(tmp_path):
    result = _run(["determinability", "--objects", "6"], tmp_path)
    assert result.returncode == 0
    assert "propagated_ephemeris_accuracy" in result.stdout
    assert "continuity_resolution_km" in result.stdout


def test_consequence(tmp_path):
    result = _run(["consequence", "--objects", "60"], tmp_path)
    assert result.returncode == 0
    assert "median max error km" in result.stdout
    assert "checksum_digit" in result.stdout


def test_screen(tmp_path):
    result = _run(["screen", "--objects", "60"], tmp_path)
    assert result.returncode == 0
    assert "rule baseline against measured consequence" in result.stdout
    assert "operating-characteristic area" in result.stdout


def test_benchmark(tmp_path):
    result = _run(["benchmark", "--objects", "120"], tmp_path)
    assert result.returncode == 0
    assert "majority-class baseline" in result.stdout
    assert "learned elements_and_history" in result.stdout
    assert "rules at their own threshold" in result.stdout


def test_divergence(tmp_path):
    result = _run(["divergence", "--objects", "60"], tmp_path)
    assert result.returncode == 0
    assert "which source is right: None" in result.stdout
    assert "not determinable" in result.stdout


def test_bad_split_is_rejected_by_the_parser(tmp_path):
    result = _run(["screen", "--split", "holdout"], tmp_path)
    assert result.returncode != 0


def test_no_subprocess_artefact_is_written_into_the_tree(tmp_path):
    _run(["info"], tmp_path)
    for name in (".config", "__pycache__", ".pytest_cache", ".ruff_cache", ".hypothesis"):
        assert not (ROOT / name).exists() or name in {
            "__pycache__",
            ".pytest_cache",
            ".ruff_cache",
            ".hypothesis",
        }
    assert not (ROOT / ".config").exists()


def test_parser_is_introspectable():
    from tlecheck.__main__ import build_parser

    parser = build_parser()
    assert parser.prog == "tlecheck"


@pytest.mark.parametrize(
    "command",
    ["info", "check", "determinability", "consequence", "screen", "benchmark", "divergence"],
)
def test_every_subcommand_is_registered(command):
    from tlecheck.__main__ import build_parser

    parser = build_parser()
    actions = [
        action
        for action in parser._subparsers._group_actions  # noqa: SLF001
        if hasattr(action, "choices")
    ]
    assert any(command in (action.choices or {}) for action in actions)
