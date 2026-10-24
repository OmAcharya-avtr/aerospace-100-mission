"""CLI tests. Exit statuses are part of the interface and are asserted."""

from __future__ import annotations

import numpy as np
import pytest

from boundcompress.__main__ import build_parser, main


def test_info_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["info"]) == 0
    captured = capsys.readouterr().out
    assert "not installed; no number here comes from either" in captured
    assert "previous=1" in captured


def test_certify_exits_zero_when_the_bound_holds(
    capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["certify", "--signal", "battery_voltage", "--samples", "256",
                 "--bound", "1e-3"]) == 0
    assert "bound verdict          : HOLDS" in capsys.readouterr().out


def test_certify_relative_mode(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["certify", "--signal", "panel_temperature", "--samples", "256",
                 "--mode", "rel", "--bound", "1e-5"]) == 0
    assert "declared mode          : rel" in capsys.readouterr().out


def test_certify_refuses_relative_mode_on_a_zero_crossing_signal() -> None:
    """wheel_rate starts at exactly 0.0, so the zero check fires before the sign check."""
    with pytest.raises(ValueError, match="undefined at zero"):
        main(["certify", "--signal", "wheel_rate", "--samples", "256",
              "--mode", "rel", "--bound", "1e-3"])


def test_negative_control_exits_zero_when_it_catches_the_violation(
    capsys: pytest.CaptureFixture[str]
) -> None:
    """Exit 0 means the control worked, which is the opposite of the bound holding."""
    assert main(["negative-control", "--signal", "magnetometer", "--samples", "512",
                 "--bound", "0.01", "--index", "100"]) == 0
    captured = capsys.readouterr().out
    assert "bound verdict          : FAILED" in captured
    assert "negative control caught: True" in captured


def test_negative_control_relative_mode(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["negative-control", "--signal", "battery_voltage", "--samples", "256",
                 "--mode", "rel", "--bound", "1e-4", "--index", "77"]) == 0
    assert "negative control caught: True" in capsys.readouterr().out


def test_sweep_prints_a_table(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["sweep", "--signal", "gyro_rate", "--samples", "256",
                 "--points", "2"]) == 0
    captured = capsys.readouterr().out
    assert "bits/sample" in captured
    assert "roundtrip" in captured
    assert " NO" not in captured


def test_input_file_is_read(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "samples.txt"
    np.savetxt(path, np.linspace(5.0, 6.0, 128))
    assert main(["certify", "--input", str(path), "--bound", "1e-3"]) == 0
    assert "samples.txt" in capsys.readouterr().out


def test_version_flag_exits_zero() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0


def test_missing_subcommand_is_refused() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code != 0


def test_parser_lists_every_subcommand() -> None:
    parser = build_parser()
    actions = [
        action for action in parser._subparsers._actions  # noqa: SLF001
        if hasattr(action, "choices") and action.choices
    ]
    names = set()
    for action in actions:
        names.update(action.choices)
    assert {"info", "certify", "sweep", "compare", "negative-control"} <= names
