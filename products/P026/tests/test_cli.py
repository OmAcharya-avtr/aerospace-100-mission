"""CLI: exit codes, deterministic output, and importability from a subprocess."""

from __future__ import annotations

import subprocess
import sys

import pytest

from wahbakit.cli import main


def test_demo_runs_and_lists_all_four_methods(capsys):
    assert main(["demo", "--seed", "3", "--sigma", "1e-3", "--n", "4"]) == 0
    out = capsys.readouterr().out
    for method in ("triad", "q-method", "quest", "olae"):
        assert method in out
    assert "observability lambda_min" in out


def test_demo_is_deterministic(capsys):
    main(["demo", "--seed", "9"])
    first = capsys.readouterr().out
    main(["demo", "--seed", "9"])
    assert capsys.readouterr().out == first


def test_conventions_states_the_frame_order(capsys):
    assert main(["conventions"]) == 0
    out = capsys.readouterr().out
    assert "b_i ~= A r_i" in out
    assert "scalar first" in out


@pytest.mark.parametrize("argv", [["demo", "--sigma", "0"], ["demo", "--n", "1"]])
def test_invalid_arguments_exit_two(argv):
    with pytest.raises(SystemExit) as excinfo:
        main(argv)
    assert excinfo.value.code == 2


def test_unknown_subcommand_exits_two():
    with pytest.raises(SystemExit) as excinfo:
        main(["nonsense"])
    assert excinfo.value.code == 2


def test_module_entry_point_in_a_subprocess():
    # conftest.py exports PYTHONPATH so the child can import the package.
    result = subprocess.run(
        [sys.executable, "-m", "wahbakit", "demo", "--seed", "1"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "quest" in result.stdout


def test_version_flag_in_a_subprocess():
    result = subprocess.run(
        [sys.executable, "-m", "wahbakit", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "0.1.0" in result.stdout
