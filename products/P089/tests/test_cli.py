"""The command line. Exercised in-process through main(), not as a subprocess."""

from __future__ import annotations

import pytest

from aligncal.__main__ import main


def test_version(capsys):
    with pytest.raises(SystemExit) as raised:
        main(["--version"])
    assert raised.value.code == 0
    assert "aligncal 0.1.0" in capsys.readouterr().out


def test_info(capsys):
    assert main(["info"]) == 0
    out = capsys.readouterr().out
    assert "observability first" in out
    assert "(8)       the observability Gramian" in out
    assert "not flight-qualified" in out
    for profile in ("single_axis", "isotropic"):
        assert profile in out


def test_observability_prints_no_estimate(capsys):
    assert main(["observability", "--samples", "400", "--profile", "single_axis"]) == 0
    out = capsys.readouterr().out
    assert "NO ESTIMATE WAS COMPUTED" in out
    assert "RANK DEFICIENT" in out
    assert "BATCH ESTIMATE" not in out


def test_align(capsys):
    assert main(["align", "--samples", "400", "--profile", "three_axis"]) == 0
    out = capsys.readouterr().out
    assert "BATCH ESTIMATE" in out
    assert "injected truth" in out
    assert "error on the unobservable subspace" in out


def test_align_withholds_on_a_single_axis_record(capsys):
    assert main(["align", "--samples", "400", "--profile", "single_axis"]) == 0
    assert "WITHHELD" in capsys.readouterr().out


def test_align_handles_several_secondaries(capsys):
    assert main(["align", "--samples", "300", "--secondaries", "2"]) == 0
    out = capsys.readouterr().out
    assert out.count("BATCH ESTIMATE") == 2


def test_compare(capsys):
    assert main(["compare", "--samples", "300"]) == 0
    out = capsys.readouterr().out
    assert "BATCH AGAINST RECURSIVE" in out
    assert "mekf sigma" in out


def test_unobservable(capsys):
    assert main(
        [
            "unobservable",
            "--profile",
            "single_axis",
            "--samples",
            "120",
            "--train",
            "200",
            "--configuration",
            "bagged_ridge",
        ]
    ) == 0
    out = capsys.readouterr().out
    assert "injected truth" in out
    assert "inf (refused)" in out
    assert "the analytic ratio is infinite" in out


def test_benchmark(capsys):
    assert main(["benchmark", "--train", "120", "--test", "40", "--samples", "120"]) == 0
    out = capsys.readouterr().out
    assert "SHORT-RECORD BENCHMARK" in out
    assert "trivial (mean)" in out
    assert "lowest total RMSE" in out


def test_bad_argument_exits_two(capsys):
    assert main(["observability", "--amplitude", "-1"]) == 2
    assert "amplitude must be finite" in capsys.readouterr().err


def test_bad_floor_factor_exits_two(capsys):
    assert main(["observability", "--floor-factor", "0.1"]) == 2
    assert "at least 1" in capsys.readouterr().err


def test_unknown_profile_is_rejected_by_argparse():
    with pytest.raises(SystemExit):
        main(["observability", "--profile", "helix"])


def test_missing_subcommand_is_rejected():
    with pytest.raises(SystemExit):
        main([])
