"""The CLI and the report assembly, exercised through the public entry point."""

from __future__ import annotations

import pytest

from reliablock import __version__
from reliablock.__main__ import main
from reliablock.cases import CASES, MISSIONS
from reliablock.quantify import quantify
from reliablock.report import (
    common_cause_comparison,
    cut_set_report,
    diagram_report,
    importance_report,
    mission_report,
    quantity_tree,
)


class TestSubcommands:
    def test_info(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["info"]) == 0
        out = capsys.readouterr().out
        assert __version__ in out
        assert "no learned component" in out
        assert "reliability" in out
        for name in CASES:
            assert name in out
        for name in MISSIONS:
            assert name in out

    def test_info_carries_the_research_grade_statement(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["info"])
        out = capsys.readouterr().out
        # The statement is wrapped across two lines in the output, so the test
        # checks the collapsed text rather than a line-dependent fragment.
        collapsed = " ".join(out.split()).lower()
        assert "not flight-qualified" in collapsed
        assert "not a safety case" in collapsed

    @pytest.mark.parametrize("case_name", sorted(CASES))
    def test_show(self, case_name: str, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["show", case_name]) == 0
        out = capsys.readouterr().out
        assert "minimal cut sets" in out
        assert CASES[case_name].name in out

    def test_show_with_enumeration(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["show", "voting_triple", "--enumerate"]) == 0
        out = capsys.readouterr().out
        assert "exhaustive enumeration" in out
        assert "identical                    : True" in out

    @pytest.mark.parametrize("case_name", sorted(CASES))
    def test_quantify(self, case_name: str, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["quantify", case_name]) == 0
        out = capsys.readouterr().out
        assert "system reliability" in out
        assert "per-node breakdown" in out

    @pytest.mark.parametrize("case_name", sorted(CASES))
    def test_compare(self, case_name: str, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["compare", case_name]) == 0
        out = capsys.readouterr().out
        assert "independence assumed" in out
        assert "declared common cause" in out

    def test_compare_says_so_when_no_beta_is_declared(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["compare", "warm_standby"]) == 0
        out = capsys.readouterr().out
        assert "No beta factor is declared" in out

    def test_importance(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["importance", "voting_triple"]) == 0
        out = capsys.readouterr().out
        assert "Birnbaum" in out
        assert "voter" in out

    def test_importance_refusal_is_reported_not_raised(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["importance", "load_sharing_pair"]) == 0
        out = capsys.readouterr().out
        assert "refused" in out
        assert "dependent quantification" in out

    def test_mission(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["mission", "smallsat_bus", "--mission", "leo_phased"]) == 0
        out = capsys.readouterr().out
        assert "launch_and_leop" in out
        assert "effective age (h)" in out

    def test_mission_with_explicit_hours(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["quantify", "redundant_pair", "--hours", "1000"]) == 0
        assert "1000 h" in capsys.readouterr().out

    def test_simulate(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["simulate", "redundant_pair", "--trials", "20000", "--seed", "5"]) == 0
        out = capsys.readouterr().out
        assert "closed form inside : True" in out
        assert "UNSTABLE" in out

    def test_simulate_sizes_its_own_budget(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["simulate", "voting_triple", "--target", "0.2", "--seed", "5"]) == 0
        out = capsys.readouterr().out
        assert "interval half-width" in out

    def test_bridge(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["bridge"]) == 0
        out = capsys.readouterr().out
        assert "0.978480000000000" in out
        assert "DERIVED" in out
        assert "No page" in out
        assert out.count("0.978480000000000") == 4

    def test_version(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exit_info:
            main(["--version"])
        assert exit_info.value.code == 0
        assert __version__ in capsys.readouterr().out

    def test_help_exits_zero(self) -> None:
        with pytest.raises(SystemExit) as exit_info:
            main(["--help"])
        assert exit_info.value.code == 0

    def test_a_missing_subcommand_is_an_error(self) -> None:
        with pytest.raises(SystemExit) as exit_info:
            main([])
        assert exit_info.value.code != 0

    def test_an_unknown_case_is_an_error(self) -> None:
        with pytest.raises(SystemExit):
            main(["show", "not_a_case"])


class TestDeclaredFiles:
    def test_a_declared_diagram_file_is_loaded(
        self, tmp_path, capsys: pytest.CaptureFixture[str]
    ) -> None:  # type: ignore[no-untyped-def]
        path = tmp_path / "d.yaml"
        path.write_text(CASES["redundant_pair"].to_yaml())
        assert main(["compare", "--diagram", str(path)]) == 0
        assert "declared common cause" in capsys.readouterr().out

    def test_a_declared_mission_file_is_loaded(
        self, tmp_path, capsys: pytest.CaptureFixture[str]
    ) -> None:  # type: ignore[no-untyped-def]
        path = tmp_path / "m.yaml"
        path.write_text(MISSIONS["one_year"].to_yaml())
        assert main(["quantify", "redundant_pair", "--mission-file", str(path)]) == 0
        assert "system reliability" in capsys.readouterr().out


class TestReportAssembly:
    def test_diagram_report_names_the_absence_of_a_beta(self) -> None:
        lines = diagram_report(CASES["warm_standby"])
        assert any("assumed independent" in line for line in lines)

    def test_diagram_report_warns_about_a_dynamic_node(self) -> None:
        lines = diagram_report(CASES["load_sharing_pair"])
        assert any("not the" in line for line in lines)

    def test_cut_set_report_calls_out_singletons(self) -> None:
        lines = cut_set_report(CASES["voting_triple"])
        assert any("single points of failure (1)" in line for line in lines)

    def test_cut_set_report_says_none_when_there_are_none(self) -> None:
        lines = cut_set_report(CASES["redundant_pair"])
        assert any("single points of failure: none" in line for line in lines)

    def test_cut_set_report_respects_its_limit(self) -> None:
        lines = cut_set_report(CASES["smallsat_bus"], limit=3)
        assert any("more not shown" in line for line in lines)

    def test_quantity_tree_indents_the_children(self) -> None:
        lines = quantity_tree(quantify(CASES["smallsat_bus"], 8766.0))
        assert lines[0].startswith("bus")
        assert any(line.startswith("  ") for line in lines[1:])

    def test_importance_report_respects_its_limit(self) -> None:
        lines = importance_report(CASES["voting_triple"], 8766.0, limit=2)
        assert any("more" in line for line in lines)

    def test_mission_report_rows_match_the_blocks(self) -> None:
        lines = mission_report(CASES["voting_triple"], MISSIONS["one_year"])
        for name in CASES["voting_triple"].block_names():
            assert any(name in line for line in lines)

    def test_common_cause_comparison_reports_the_per_group_effect(self) -> None:
        lines = common_cause_comparison(CASES["smallsat_bus"], MISSIONS["three_year"])
        text = "\n".join(lines)
        assert "per-group effect" in text
        assert "reaction_wheels" in text
        assert "increases Q" in text

    def test_common_cause_comparison_warns_on_a_non_conservative_group(self) -> None:
        # A 3-of-4 group driven past the crossing: the beta factor reduces the
        # computed unreliability, and the report says so rather than leaving it
        # for the reader to notice.
        from reliablock.diagram import Block, Diagram, KOfN
        from reliablock.distributions import Exponential

        diagram = Diagram(
            "past the crossing",
            KOfN(
                k=3,
                items=tuple(
                    Block(name=f"u{i}", distribution=Exponential(3.0e-4)) for i in range(4)
                ),
                beta_factor=0.1,
                label="group",
            ),
        )
        text = "\n".join(common_cause_comparison(diagram, 1000.0))
        assert "WARNING" in text
        assert "DECREASES" in text
        assert "multiple Greek letter" in text

    def test_no_report_line_contains_marketing_language(self) -> None:
        banned = (
            "revolutionary",
            "cutting-edge",
            "powerful",
            "seamless",
            "game-changing",
            "!",
            "flight-safe",
            "certified",
            "mission-ready",
            "production-ready",
        )
        texts = []
        for case_name, diagram in CASES.items():
            del case_name
            texts.append("\n".join(diagram_report(diagram)))
            texts.append("\n".join(cut_set_report(diagram)))
            texts.append("\n".join(common_cause_comparison(diagram, 8766.0)))
        blob = "\n".join(texts).lower()
        for word in banned:
            if word == "certified":
                # "not certified" is required; the bare claim is not.
                assert "is certified" not in blob
                continue
            assert word not in blob
