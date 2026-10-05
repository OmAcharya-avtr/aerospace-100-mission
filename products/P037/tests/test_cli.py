"""CLI surface of ``python -m framesync``."""

from __future__ import annotations

import pytest

from framesync.__main__ import _snr_range, build_parser, main


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["--help"])
    assert exc.value.code == 0
    assert "frame-level link performance harness" in capsys.readouterr().out


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["--version"])
    assert exc.value.code == 0
    assert "framesync 0.1.0" in capsys.readouterr().out


def test_missing_subcommand_is_an_error():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_snr_range_forms():
    assert _snr_range("0:4:2").tolist() == [0.0, 2.0, 4.0]
    assert _snr_range("3,5.5").tolist() == [3.0, 5.5]
    with pytest.raises(Exception, match="start:stop:step"):
        _snr_range("1:2")
    with pytest.raises(Exception, match="step must be positive"):
        _snr_range("0:4:0")
    with pytest.raises(Exception, match="cannot parse"):
        _snr_range("abc")


def test_fer_analytic_only(capsys):
    assert main(["fer", "--ebn0", "8,10"]) == 0
    out = capsys.readouterr().out
    assert "FER uncoded" in out and "8936 bits" in out


def test_fer_with_measurement(capsys):
    assert main(["fer", "--ebn0", "9", "--frame-octets", "120", "--measure",
                 "--frames", "60"]) == 0
    assert "measured (Monte Carlo" in capsys.readouterr().out


def test_fer_with_rs_and_conv_measurement(capsys):
    assert main(["fer", "--ebn0", "5", "--measure-rs", "--rs-frames", "3",
                 "--measure-conv", "--conv-frames", "10", "--conv-info-bits", "64"]) == 0
    out = capsys.readouterr().out
    assert "measured RS(255,223)" in out and "d_free=10" in out


def test_falsesync(capsys):
    assert main(["falsesync", "--max-tolerance", "2"]) == 0
    out = capsys.readouterr().out
    assert "0x1ACFFC1D" in out and "2.328306e-10" in out


def test_gain(capsys):
    assert main(["gain", "--target", "1e-5"]) == 0
    out = capsys.readouterr().out
    assert "gain =" in out and "0.874510" in out


def test_sync_trace(capsys):
    assert main(["sync", "1101000", "--flywheel-max", "2"]) == 0
    out = capsys.readouterr().out
    assert "SEARCH" in out and "FLYWHEEL" in out and "final state" in out


def test_slip(capsys):
    assert main(["slip", "--frame-bits", "256", "--n-frames", "12",
                 "--slip-at-frame", "5", "--slip-bits", "3"]) == 0
    out = capsys.readouterr().out
    assert "wrong phase" in out and "state trace" in out
