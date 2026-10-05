"""Every CLI subcommand runs, exits 0, and prints numbers from the library."""

from __future__ import annotations

import pytest

from bitflipsim.__main__ import build_parser, main


def test_parser_requires_a_subcommand():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_layout_subcommand(capsys):
    assert main(["layout", "--dtype", "float32"]) == 0
    out = capsys.readouterr().out
    assert "exponent bias         127" in out
    assert "overall agree    True" in out


def test_layout_subcommand_for_integers(capsys):
    assert main(["layout", "--dtype", "int8"]) == 0
    out = capsys.readouterr().out
    assert "two's complement" in out
    assert "storage bits          8" in out


def test_flip_subcommand_float(capsys):
    assert main(["flip", "--value", "5.0", "--bit", "30"]) == 0
    out = capsys.readouterr().out
    assert "prediction matches    True" in out
    assert "regime                scaled" in out


def test_flip_subcommand_int(capsys):
    assert main(["flip", "--value", "-100", "--bit", "7", "--dtype", "int8"]) == 0
    out = capsys.readouterr().out
    assert "predicted delta       +128" in out


def test_flux_subcommand(capsys):
    assert main(
        ["flux", "--flux", "1e3", "--cross-section", "1e-14", "--bits", "4704",
         "--exposure", "1e6"]
    ) == 0
    out = capsys.readouterr().out
    assert "4.704000e-08 upsets s^-1" in out
    assert "1.693440e+05 FIT" in out
    assert "Poisson validity      True" in out


def test_campaign_subcommand(capsys):
    assert main(["campaign", "--expected", "2", "--trials", "20"]) == 0
    out = capsys.readouterr().out
    assert "standard error" in out
    assert "golden accuracy" in out


def test_campaign_subcommand_with_clamp(capsys):
    assert main(["campaign", "--expected", "2", "--trials", "20", "--clamp"]) == 0
    assert "clamp limit           4." in capsys.readouterr().out


def test_criticality_subcommand(capsys):
    assert main(["criticality", "--budget-fraction", "0.05"]) == 0
    out = capsys.readouterr().out
    for method in ("magnitude_baseline", "exponent_heuristic", "learned_predictor", "oracle"):
        assert method in out
    assert "uncertainty coverage at k=2" in out


def test_criticality_subcommand_accepts_the_word_cost_model():
    """Parsed here only. The word cost model itself is exercised at library
    level in ``test_criticality.py``; running the subcommand twice would cost
    two more full ground-truth sweeps for no extra coverage."""
    args = build_parser().parse_args(["criticality", "--cost-model", "word"])
    assert args.cost_model == "word"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["criticality", "--cost-model", "page"])


def test_mitigate_subcommand(capsys):
    assert main(["mitigate"]) == 0
    out = capsys.readouterr().out
    assert "single-upset bound" in out
    assert "selective_triplication" in out
    assert "periodic_reload" in out


def test_onnx_subcommand(capsys, tmp_path):
    target = tmp_path / "model.onnx"
    assert main(["onnx", "--out", str(target)]) == 0
    assert target.exists()
    out = capsys.readouterr().out
    assert "W1" in out and "b2" in out
    assert target.stat().st_size > 0


def test_version_exits_zero():
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
