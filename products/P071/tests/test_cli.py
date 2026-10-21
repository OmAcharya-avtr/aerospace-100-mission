"""The command-line interface. Exercised through ``main`` with a captured stream."""

from __future__ import annotations

import io

import pytest

from conjprob.__main__ import main


def _run(argv, capture=True):
    stream = io.StringIO() if capture else None
    status = main(argv, stream=stream)
    return status, (stream.getvalue() if stream is not None else "")


def test_version_exits_zero(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert "conjprob" in capsys.readouterr().out


def test_no_subcommand_is_a_usage_error():
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2


def test_assess_prints_every_method(cases_dir):
    status, text = _run(["assess", str(cases_dir / "leo_crossing.yaml")])
    assert status == 0
    for name in ("foster", "chan", "alfano", "small_body", "quadrature"):
        assert name in text
    assert "disagreement (the primary output)" in text
    assert "DilutionWarning" in text


def test_assess_with_monte_carlo_reports_an_interval(cases_dir):
    status, text = _run(
        ["assess", str(cases_dir / "leo_crossing.yaml"), "--monte-carlo", "20000"]
    )
    assert status == 0
    assert "monte_carlo" in text
    assert "CI [" in text


def test_assess_on_a_case_without_dilution_omits_the_warning(cases_dir):
    status, text = _run(["assess", str(cases_dir / "far_miss_tight_covariance.yaml")])
    assert status == 0
    assert "DilutionWarning" not in text


def test_assess_missing_file_returns_two(capsys):
    status, _ = _run(["assess", "no_such_case.yaml"])
    assert status == 2
    assert "no_such_case.yaml" in capsys.readouterr().err


def test_dilution_prints_a_table_with_the_maximum(cases_dir):
    status, text = _run(
        ["dilution", str(cases_dir / "leo_crossing.yaml"), "--points", "7"]
    )
    assert status == 0
    assert "maximum Pc" in text
    assert "pc_foster" in text
    rows = [
        line for line in text.splitlines() if line.strip().startswith(("0.", "1", "10"))
    ]
    assert len(rows) >= 5


def test_regimes_prints_a_boundary_table_per_method():
    status, text = _run(
        ["regimes", "--points", "12", "--aspects", "1.0", "--mahalanobis", "1.0"]
    )
    assert status == 0
    for name in ("foster", "chan", "alfano", "small_body"):
        assert name in text
    assert "nan means" in text


def test_screen_prints_every_screen_and_the_learned_model():
    status, text = _run(
        ["screen", "--n-train", "2000", "--n-test", "1200", "--seed", "0"]
    )
    assert status == 0
    for name in ("miss_distance", "mahalanobis", "small_body", "learned"):
        assert name in text
    assert "recall" in text


def test_screen_accepts_an_alternative_test_regime():
    status, text = _run(
        [
            "screen",
            "--n-train",
            "2000",
            "--n-test",
            "1500",
            "--regime",
            "homogeneous",
        ]
    )
    assert status == 0
    assert "test regime homogeneous" in text


def test_screen_with_an_unknown_regime_returns_two(capsys):
    status, _ = _run(["screen", "--n-train", "500", "--n-test", "500", "--regime", "nope"])
    assert status == 2
    assert "regime" in capsys.readouterr().err
