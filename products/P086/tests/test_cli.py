"""The command line interface. Every command is executed; none is only parsed."""

from __future__ import annotations

import pytest

from telemgap.__main__ import DEFAULT_TOLERANCES, main


def test_version_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert "telemgap 0.1.0" in capsys.readouterr().out


def test_no_command_is_an_error() -> None:
    with pytest.raises(SystemExit) as exit_info:
        main([])
    assert exit_info.value.code != 0


def test_info(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["info"]) == 0
    out = capsys.readouterr().out
    assert "channel schema" in out
    assert "WHAT THIS RECORD CANNOT DETERMINE" in out
    assert "Not flight-qualified" in out
    assert "not declared" in out


def test_check_prints_determinability_first(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check", "--samples", "400"]) == 0
    out = capsys.readouterr().out
    assert out.index("WHAT THIS RECORD CANNOT DETERMINE") < out.index("INTEGRITY REPORT")
    assert "ground truth of the generated record" in out
    assert "A clean integrity report does not mean the telemetry is correct." in out


def test_check_specified_mode(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check", "--samples", "400", "--mode", "specified"]) == 0
    assert "mode=specified" in capsys.readouterr().out


def test_policy(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["policy", "--samples", "900"]) == 0
    out = capsys.readouterr().out
    assert "GAP POLICY" in out
    assert "realised reconstruction error against the noiseless truth" in out


def test_policy_without_the_resolvability_gate(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["policy", "--samples", "900", "--no-resolvability-gate"]) == 0
    assert "GAP POLICY" in capsys.readouterr().out


def test_refuse_passes_its_own_check(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["refuse", "--outage-frames", "5000"]) == 0
    out = capsys.readouterr().out
    assert "REFUSAL CHECK PASSED" in out
    assert "refusals                : 4" in out


def test_determinability_withholding(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["determinability", "--samples", "400", "--withhold", "none"]) == 0
    baseline = capsys.readouterr().out
    assert main(["determinability", "--samples", "400", "--withhold", "validity"]) == 0
    without_validity = capsys.readouterr().out
    assert main(["determinability", "--samples", "400", "--withhold", "raw"]) == 0
    without_raw = capsys.readouterr().out
    assert "withheld: none" in baseline
    assert "no per-frame error-control verdict is present" in without_validity
    assert "the raw words of equation (1) were not retained" in without_raw


def test_benchmark(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["benchmark", "--samples", "1200"]) == 0
    out = capsys.readouterr().out
    assert "all/rules_specified" in out
    assert "all/rules_strengthened" in out
    assert "all/learned" in out
    assert "all/majority" in out
    assert "baseline versus learned, per class and metric" in out
    assert "This model is not certified for operational flight use." in out


def test_default_tolerances_cover_every_generated_channel() -> None:
    from telemgap import dataset

    assert set(DEFAULT_TOLERANCES) == set(dataset.schema().channel_names)
    assert all(value > 0.0 for value in DEFAULT_TOLERANCES.values())
