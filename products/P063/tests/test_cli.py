"""The CLI: every subcommand, every exit code."""

from __future__ import annotations

import json

import pytest

from safemodetree.__main__ import main


def run(capsys, argv: list[str]) -> tuple[int, str, str]:
    """Run the CLI in-process and return (code, stdout, stderr)."""
    code = main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_version_exits_zero(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0


def test_info_prints_the_summary(capsys):
    code, out, _err = run(capsys, ["info"])
    assert code == 0
    assert "8 faults" in out
    assert "indistinguishable: st_blinding, st_latchup" in out


def test_info_json_is_parseable(capsys):
    code, out, _err = run(capsys, ["info", "--json"])
    assert code == 0
    payload = json.loads(out)
    assert payload["n_states"] == 36
    assert payload["safe_modes"] == ["sun_safe", "survival"]
    assert len(payload["ambiguity_groups"]) == 2


def test_check_reports_findings_and_exits_zero_without_the_flag(capsys):
    code, out, _err = run(capsys, ["check"])
    assert code == 0
    assert "total findings: 6" in out


def test_check_exits_one_with_the_flag(capsys):
    code, out, _err = run(capsys, ["check", "--fail-on-findings"])
    assert code == 1
    assert "vacuous_responses" in out


def test_check_json_lists_the_conflicting_classes(capsys):
    code, out, _err = run(capsys, ["check", "--json"])
    assert code == 0
    payload = json.loads(out)
    assert payload["findings"]["conflicting_ambiguity_classes"] == 4
    assert len(payload["conflicting_classes"]) == 4
    assert payload["vacuous"] == ["rw_swap_to_spare", "st_power_cycle_in_safe"]


def test_check_on_the_planted_declaration(capsys):
    code, out, _err = run(capsys, ["check", "--spec", "planted_unreachable", "--json"])
    assert code == 0
    payload = json.loads(out)
    assert payload["unreachable_fault_states"] == ["(ops, sensor_fail)"]
    assert payload["unreachable_safe_modes"] == ["orphan_safe"]


def test_check_with_a_latency_budget_changes_the_findings(capsys):
    code, out, _err = run(capsys, ["check", "--latency-budget", "60", "--json"])
    assert code == 0
    payload = json.loads(out)
    assert payload["findings"]["conflicting_ambiguity_classes"] == 2
    assert "60 s" in payload["assumption"]


def test_check_with_perfect_detection(capsys):
    code, out, _err = run(capsys, ["check", "--assume-perfect-detection"])
    assert code == 0
    assert "perfect" in out


def test_reach_table_and_json(capsys):
    code, out, _err = run(capsys, ["reach"])
    assert code == 0
    assert "coarse_downmode -> safe_mode_entry" in out
    code, out, _err = run(capsys, ["reach", "--json"])
    payload = json.loads(out)
    assert payload["all_reachable"] is True
    assert len(payload["rows"]) == 16


def test_reach_exits_one_on_the_planted_declaration(capsys):
    code, out, _err = run(
        capsys, ["reach", "--spec", "planted_unreachable", "--fail-on-findings"]
    )
    assert code == 1
    assert "NO" in out


def test_ambiguity_table_and_json(capsys):
    code, out, _err = run(capsys, ["ambiguity"])
    assert code == 0
    assert "st_blinding+st_latchup" in out
    code, out, _err = run(capsys, ["ambiguity", "--json"])
    payload = json.loads(out)
    assert len(payload["classes"]) == 4


def test_ambiguity_says_so_when_there_is_none(capsys):
    code, out, _err = run(capsys, ["ambiguity", "--spec", "planted_unreachable"])
    assert code == 0
    assert "no declared ambiguity class" in out


def test_benchmark_prints_both_detectors(capsys):
    code, out, _err = run(
        capsys, ["benchmark", "--noise-scale", "2.0", "--n-per-fault", "80"]
    )
    assert code == 0
    assert "threshold logic" in out
    assert "learned (matched FAR)" in out
    assert "accuracy ceiling" in out


def test_benchmark_json_has_the_three_scores(capsys):
    code, out, _err = run(
        capsys, ["benchmark", "--noise-scale", "2.0", "--n-per-fault", "80", "--json"]
    )
    assert code == 0
    payload = json.loads(out)
    assert set(payload["scores"]) == {"baseline", "learned", "learned_argmax"}
    assert payload["scores"]["baseline"]["brier"] is None


def test_policy_prints_the_cost_of_conservatism(capsys):
    code, out, _err = run(
        capsys, ["policy", "--noise-scale", "2.0", "--n-per-fault", "80"]
    )
    assert code == 0
    assert "cost of conservatism against the oracle" in out
    assert "mis-routes avoided" in out


def test_policy_sweep_and_json(capsys):
    code, out, _err = run(
        capsys, ["policy", "--noise-scale", "2.0", "--n-per-fault", "80", "--sweep"]
    )
    assert code == 0
    assert "routing threshold sweep" in out
    code, out, _err = run(
        capsys,
        ["policy", "--noise-scale", "2.0", "--n-per-fault", "80", "--json",
         "--detector", "baseline"],
    )
    payload = json.loads(out)
    assert payload["detector"] == "baseline"
    assert set(payload["policies"]) == {
        "oracle", "always_specific", "confidence_routed", "always_conservative"
    }


def test_unknown_declaration_exits_two(capsys):
    code, _out, err = run(capsys, ["check", "--spec", "no_such_declaration"])
    assert code == 2
    assert "no packaged declaration" in err


def test_unknown_mode_exits_two(capsys):
    code, _out, err = run(
        capsys, ["policy", "--mode", "nowhere", "--n-per-fault", "60"]
    )
    assert code == 2
    assert "no declared mode" in err


def test_invalid_declaration_file_exits_two(capsys, tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("name: broken\n", encoding="utf-8")
    code, _out, err = run(capsys, ["check", "--spec", str(path)])
    assert code == 2
    assert "missing required key" in err


def test_a_declaration_can_be_loaded_from_a_path(capsys, tmp_path, minimal_yaml):
    path = tmp_path / "ok.yaml"
    path.write_text(minimal_yaml, encoding="utf-8")
    code, out, _err = run(capsys, ["info", "--spec", str(path)])
    assert code == 0
    assert "minimal" in out


def test_no_subcommand_is_refused(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2
