"""Smoke and input-validation tests for wavelab.cli / python -m wavelab."""

from __future__ import annotations

import pytest

from wavelab.cli import build_parser, main


def test_build_parser_requires_subcommand():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_geometry_command_runs(capsys):
    rc = main(["geometry", "--n-grid", "7"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "hudgin" in out
    assert "fried" in out
    assert "null space dimension" in out


def test_reconstruct_command_runs(capsys):
    rc = main(["reconstruct", "--n-side", "8", "--j-max", "10", "--seed", "1"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "reconstruction RMS err" in out


def test_reconstruct_command_bad_r0_over_d_exits_with_error_code(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["reconstruct", "--r0-over-d", "-1.0"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "error:" in err


def test_unknown_command_exits_nonzero():
    with pytest.raises(SystemExit):
        main(["bogus-command"])


def test_geometry_default_n_grid_runs(capsys):
    rc = main(["geometry"])
    assert rc == 0


def test_demo_benchmark_small_runs(capsys):
    rc = main(
        [
            "demo-benchmark",
            "--n-side",
            "6",
            "--j-max",
            "9",
            "--n-train",
            "40",
            "--n-test",
            "20",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "baseline RMS" in out
    assert "ML RMS" in out
