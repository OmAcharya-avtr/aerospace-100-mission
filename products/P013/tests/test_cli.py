"""Tests for the python -m turbscope CLI."""

from __future__ import annotations

import pytest

from turbscope.__main__ import build_parser, main


def test_build_parser_requires_a_subcommand():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_forward_command_runs(capsys):
    rc = main(["forward", "--cn2", "1e-14", "--path-length-m", "500"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Rytov variance" in out
    assert "DIMM differential variance" in out


def test_invert_command_runs_and_reports_multivalued(capsys):
    # A sigma_i2 chosen to sit inside the saturation overshoot band.
    rc = main(["invert", "--sigma-i2", "1.05", "--path-length-m", "1000"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "weak-regime baseline Cn2" in out


def test_invert_command_weak_regime_no_warning(capsys):
    rc = main(["invert", "--sigma-i2", "0.001", "--path-length-m", "500"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "WARNING" not in out


def test_predict_command_runs(capsys):
    rc = main(
        [
            "predict",
            "--sigma-i2",
            "0.05",
            "--var-long",
            "1e-12",
            "--var-trans",
            "8e-13",
            "--path-length-m",
            "500",
            "--n-scenarios",
            "60",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "learned model" in out
    assert "scintillometer weak baseline" in out


def test_cli_reports_value_error_as_exit_code_2(capsys):
    rc = main(["forward", "--cn2", "-1.0", "--path-length-m", "500"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "error:" in err


def test_cli_unknown_wave_type_rejected_by_argparse():
    with pytest.raises(SystemExit):
        main(["forward", "--cn2", "1e-14", "--path-length-m", "500", "--wave-type", "toroidal"])
