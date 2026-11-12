"""The command-line interface, including the required explicit origin."""

from __future__ import annotations

import json

import pytest

from thermalbudget import VALIDATION_LEVEL
from thermalbudget.__main__ import build_parser, main


def test_help_exits_zero():
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args(["--help"])
    assert exit_info.value.code == 0


def test_version_exits_zero_and_prints_the_version(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert "thermalbudget 0.1.0" in capsys.readouterr().out


def test_a_missing_subcommand_is_an_error():
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args([])
    assert exit_info.value.code == 2


def test_schema_subcommand_prints_the_binding_schema(capsys):
    assert main(["schema"]) == 0
    out = capsys.readouterr().out
    assert "temp_tj-thermal" in out
    assert "never a value" in out
    assert "cv0-thermal" in out
    assert "inferred from the data, not assumed" in out


def test_limits_subcommand_prints_all_four_limits_and_every_named_claim(capsys):
    from thermalbudget import limits

    assert main(["limits"]) == 0
    out = capsys.readouterr().out
    for item in limits.DECLARED_LIMITS:
        assert item.key in out
    for claim in limits.UNMEASURED_BY_NAME:
        assert claim.key in out
    assert "STRUCTURALLY UNMEASURABLE" in out


def test_validation_level_subcommand_states_the_level_with_its_scope(capsys):
    """The level may be claimed; a level claimed without its scope may not.

    Updated 2026-11-12, when the coordinator set the achieved level to 4 after
    the campaign ran on the board. Before that the constant read "Level 4 not
    claimed here" and this test asserted exactly that. The invariant worth
    pinning was never the particular level -- it is that the level never
    travels without the scope that bounds it, and that the unmeasurable axis
    stays visible next to it.
    """
    assert main(["validation-level"]) == 0
    out = capsys.readouterr().out
    assert "Level 4" in out
    assert "MAXN_SUPER" in out
    assert "STRUCTURALLY UNMEASURABLE" in out


def test_origin_is_required_on_every_loading_subcommand():
    for command in ("equilibrium", "tau"):
        with pytest.raises(SystemExit) as exit_info:
            build_parser().parse_args([command, "some.csv"])
        assert exit_info.value.code == 2
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args(["report", "somewhere"])
    assert exit_info.value.code == 2


def test_an_invalid_origin_is_rejected():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["equilibrium", "x.csv", "--origin", "measuredish"])


def test_equilibrium_subcommand_on_a_synthetic_trace(capsys, no_throttle_campaign):
    path = no_throttle_campaign["soak_1.00_r1"].path
    assert main(["equilibrium", str(path), "--origin", "synthetic"]) == 0
    out = capsys.readouterr().out
    assert "origin SYNTHETIC" in out
    assert "EQUILIBRATED" in out
    assert "threshold ladder" in out


def test_tau_subcommand_prints_every_route_or_its_refusal(capsys, no_throttle_campaign):
    path = no_throttle_campaign["soak_1.00_r1"].path
    assert main(["tau", str(path), "--origin", "synthetic"]) == 0
    out = capsys.readouterr().out
    assert "least squares" in out
    assert "log-differences" in out or "REFUSED" in out
    assert "area-rule" in out or "REFUSED" in out


def test_report_subcommand_renders_and_the_json_form_agrees(capsys, no_throttle_campaign):
    directory = str(no_throttle_campaign.directory)
    assert main(["report", directory, "--origin", "synthetic", "--resamples", "500",
                 "--no-learned"]) == 0
    text = capsys.readouterr().out
    assert "HEADLINE" in text
    assert main(["report", directory, "--origin", "synthetic", "--resamples", "500",
                 "--no-learned", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["origin"] == "SYNTHETIC"
    assert payload["verdict"] in ("INDISTINGUISHABLE", "OVERSTATES", "UNDERSTATES")
    assert len(payload["overstatement_ci"]) == 2
    # The JSON form must carry the same level string as the constant, scope and
    # all, so a machine consumer cannot read a level shorn of its bounds.
    assert payload["validation_level"] == VALIDATION_LEVEL
    assert "MAXN_SUPER" in payload["validation_level"]


def test_an_absent_trace_exits_two_and_names_the_file(capsys, tmp_path):
    assert main(["equilibrium", str(tmp_path / "soak_1.00_r1.csv"), "--origin",
                 "measured"]) == 2
    assert "DATA ABSENT" in capsys.readouterr().err


def test_an_absent_directory_exits_two(capsys, tmp_path):
    assert main(["report", str(tmp_path / "traces"), "--origin", "measured"]) == 2
    assert "DATA ABSENT" in capsys.readouterr().err


def test_a_damaged_trace_exits_one(capsys, tmp_path):
    path = tmp_path / "soak_1.00_r1.csv"
    path.write_text("a,b,c\n1,2,3\n")
    assert main(["equilibrium", str(path), "--origin", "measured"]) == 1
    assert "ERROR" in capsys.readouterr().err


def test_measured_evidence_reports_an_absent_directory_without_guessing(tmp_path):
    from thermalbudget import measured_evidence

    evidence = measured_evidence(str(tmp_path))
    assert evidence["present"] is False
    assert evidence["csv_count"] == 0
    assert evidence["runs"] == []
    assert evidence["manifest"] is False
    assert "not present" in evidence["statement"]


def test_measured_evidence_counts_what_is_actually_there(tmp_path):
    from thermalbudget import measured_evidence

    traces = tmp_path / "data" / "traces"
    traces.mkdir(parents=True)
    (traces / "soak_1.00_r1.csv").write_text("x\n")
    (traces / "burst_1.00_a.csv").write_text("x\n")
    (traces / "manifest.json").write_text("{}")
    evidence = measured_evidence(str(tmp_path))
    assert evidence["present"] is True
    assert evidence["csv_count"] == 2
    assert evidence["runs"] == ["burst_1.00_a", "soak_1.00_r1"]
    assert evidence["manifest"] is True


def test_measured_evidence_distinguishes_empty_from_absent(tmp_path):
    from thermalbudget import measured_evidence

    (tmp_path / "data" / "traces").mkdir(parents=True)
    evidence = measured_evidence(str(tmp_path))
    assert evidence["present"] is False
    assert "holds no CSV" in evidence["statement"]


def test_the_validation_level_constant_carries_its_scope():
    """A bare level is not a result. See the note on the subcommand test above."""
    from thermalbudget import VALIDATION_BASIS, VALIDATION_LEVEL

    assert "Level 4" in VALIDATION_LEVEL
    # The four scope qualifications that bound every measured figure. A level
    # stated without them would overclaim, which is the failure this pins.
    assert "MAXN_SUPER" in VALIDATION_LEVEL
    assert "ambient" in VALIDATION_LEVEL
    assert "CPU-only" in VALIDATION_LEVEL
    assert "720" in VALIDATION_LEVEL
    assert "STRUCTURALLY UNMEASURABLE" in VALIDATION_BASIS
    assert "MAXN_SUPER" in VALIDATION_BASIS
