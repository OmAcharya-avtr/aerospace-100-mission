"""CLI tests. ``python -m softdecode --help`` must exit 0 in a clean subprocess."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import pytest

import softdecode
from softdecode.__main__ import build_parser, main

_SRC = str(pathlib.Path(softdecode.__file__).resolve().parent.parent)


def _run(args: list[str]) -> subprocess.CompletedProcess:
    """Run the CLI in a subprocess whose only path hint is the package root.

    The import path is derived from the installed package location, so the
    test works whether the package was installed or is being used from a
    source tree with no PYTHONPATH set.
    """
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = _SRC if not existing else os.pathsep.join([_SRC, existing])
    return subprocess.run(
        [sys.executable, "-m", "softdecode", *args],
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )


def test_help_exits_zero_in_a_subprocess():
    out = _run(["--help"])
    assert out.returncode == 0
    assert "softdecode" in out.stdout


def test_missing_subcommand_exits_nonzero():
    assert _run([]).returncode != 0


def test_parser_lists_every_subcommand():
    parser = build_parser()
    actions = [a for a in parser._actions if a.dest == "command"]
    assert actions
    assert set(actions[0].choices) == {
        "llr",
        "maxlog",
        "clip",
        "mismatch",
        "ppm",
        "crosscheck",
        "ldpc",
    }


def test_llr_subcommand(capsys):
    assert main(["llr", "--points", "5", "--ebn0", "6"]) == 0
    captured = capsys.readouterr().out
    assert "max-log error vs exact" in captured
    assert "max-log >= exact everywhere: True" in captured


def test_llr_gammagamma(capsys):
    assert main(["llr", "--fading", "gammagamma", "--points", "3"]) == 0
    assert "GammaGamma" in capsys.readouterr().out


def test_ldpc_subcommand(capsys):
    assert main(["ldpc"]) == 0
    captured = capsys.readouterr().out
    assert "length-4 cycles" in captured


def test_ppm_subcommand(capsys):
    assert main(["ppm", "--order", "4"]) == 0
    assert "Gray labels" in capsys.readouterr().out


def test_maxlog_subcommand(capsys):
    assert main(["maxlog", "--blocks", "40"]) == 0
    assert "decoded BER" in capsys.readouterr().out


def test_clip_subcommand(capsys):
    assert main(["clip", "--blocks", "40", "--limits", "1", "8"]) == 0
    assert "unclipped decoded BER" in capsys.readouterr().out


def test_mismatch_subcommand(capsys):
    assert main(["mismatch", "--blocks", "40", "--bias-db", "-2", "0", "2"]) == 0
    assert "plug-in BER" in capsys.readouterr().out


def test_crosscheck_subcommand(capsys):
    assert main(["crosscheck", "--blocks", "500"]) == 0
    captured = capsys.readouterr().out
    assert "soft <= hard: True" in captured


@pytest.mark.parametrize("args", [["llr", "--sigma-i2", "-1"], ["ldpc", "--dv", "1"]])
def test_invalid_arguments_raise(args):
    with pytest.raises(ValueError):
        main(args)
