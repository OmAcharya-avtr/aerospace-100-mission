"""CLI: every subcommand runs, exits zero, and prints what it claims to."""

from __future__ import annotations

import json

import pytest

from faultinject.__main__ import build_parser, main


def test_help_lists_every_subcommand(capsys):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for name in ("taxonomy", "cells", "run", "replay", "campaign", "benchmark"):
        assert name in out


def test_taxonomy_command(capsys):
    assert main(["taxonomy"]) == 0
    out = capsys.readouterr().out
    assert "16 kinds, 248 coverage cells" in out
    assert "sensor_bias" in out
    assert "numerical_overflow" in out


def test_cells_command(capsys):
    assert main(["cells"]) == 0
    assert "cells: 248" in capsys.readouterr().out
    assert main(["cells", "--kind", "sensor_stuck", "--list"]) == 0
    out = capsys.readouterr().out
    assert "cells: 8" in out
    assert "sensor_stuck/pos/s0/d0/p[]" in out


def test_run_command_emits_json(capsys):
    code = main(
        [
            "run",
            "--kind",
            "sensor_bias",
            "--channel",
            "pos",
            "--param",
            "offset=3.0",
            "--start",
            "40",
            "--duration",
            "100",
            "--seed",
            "7",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["case"]["case_id"] == "bd32605725b734a5"
    assert payload["cell"] == "sensor_bias/pos/s0/d1/p[2]"
    assert payload["severity"]["label"] == "severe"


def test_run_command_rejects_bad_param(capsys):
    with pytest.raises(SystemExit, match="name=value"):
        main(["run", "--kind", "sensor_bias", "--channel", "pos", "--param", "offset"])
    with pytest.raises(SystemExit, match="not a number"):
        main(["run", "--kind", "sensor_bias", "--channel", "pos", "--param", "offset=x"])


def test_run_command_reports_invalid_fault(capsys):
    code = main(["run", "--kind", "sensor_bias", "--channel", "u", "--param", "offset=1.0"])
    assert code == 2
    assert "error:" in capsys.readouterr().err


def test_replay_command(capsys):
    case_json = (
        '{"case_id":"bd32605725b734a5","injection":{"channel":"pos",'
        '"duration_steps":100,"kind":"sensor_bias","params":{"offset":3.0},'
        '"start_step":40},"n_steps":150,"seed":7}'
    )
    assert main(["replay", "--json", case_json]) == 0
    out = capsys.readouterr().out
    assert "bit identical  True" in out
    assert "bd32605725b734a5" in out


def test_replay_command_from_file(tmp_path, capsys):
    path = tmp_path / "case.json"
    path.write_text(
        '{"injection":{"channel":"pos","duration_steps":10,"kind":"sensor_stuck",'
        '"params":{},"start_step":5},"n_steps":50,"seed":2}',
        encoding="utf-8",
    )
    assert main(["replay", "--file", str(path)]) == 0
    assert "bit identical  True" in capsys.readouterr().out


def test_replay_requires_a_source():
    with pytest.raises(SystemExit, match="--file or --json"):
        main(["replay"])


def test_campaign_command(capsys):
    code = main(
        [
            "campaign",
            "--strategy",
            "coverage_greedy",
            "--budget",
            "20",
            "--replicates",
            "1",
            "--worst",
            "3",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "strategy            coverage_greedy" in out
    assert "budget              20" in out
    assert "worst cases found:" in out


def test_campaign_command_learned(capsys):
    code = main(
        [
            "campaign",
            "--strategy",
            "learned",
            "--budget",
            "20",
            "--warmup",
            "10",
            "--replicates",
            "1",
            "--worst",
            "0",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "interval coverage" in out


def test_benchmark_command(capsys):
    code = main(["benchmark", "--budget", "20", "--pools", "1", "--seeds", "2", "--warmup", "8"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["budget"] == 20
    assert "verdict_vs_uniform_random" in payload
    assert set(payload["strategies"]) == {
        "uniform_random",
        "coverage_greedy",
        "kind_mean",
        "learned",
    }
