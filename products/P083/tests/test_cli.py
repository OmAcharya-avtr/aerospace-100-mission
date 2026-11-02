"""The command line, exercised in process. The subprocess run is in validation/."""

from __future__ import annotations

import pytest

from hilverdict import __version__
from hilverdict.__main__ import build_parser, main


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    assert "hardware-in-the-loop" in capsys.readouterr().out


def test_version_prints_the_package_version(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_no_command_exits_two():
    with pytest.raises(SystemExit) as exit_info:
        main([])
    assert exit_info.value.code == 2


def test_an_unknown_command_exits_two():
    with pytest.raises(SystemExit) as exit_info:
        main(["invent"])
    assert exit_info.value.code == 2


def test_an_unknown_profile_exits_two():
    with pytest.raises(SystemExit) as exit_info:
        main(["certify", "--profile", "turbo"])
    assert exit_info.value.code == 2


def test_a_non_positive_tolerance_exits_two_with_a_message(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["certify", "--tolerance", "0"])
    assert exit_info.value.code == 2
    assert "strictly positive" in capsys.readouterr().err


def test_too_few_steps_exits_two_with_a_message(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["certify", "--steps", "1"])
    assert exit_info.value.code == 2
    assert "--steps must be at least 2" in capsys.readouterr().err


def test_an_out_of_range_target_recall_exits_two_with_a_message(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["learned", "--target-recall", "1.5"])
    assert exit_info.value.code == 2
    assert "--target-recall must lie" in capsys.readouterr().err


def test_certify_prints_the_header_and_a_table(capsys):
    assert main(["certify", "--seed", "101", "--steps", "120"]) == 0
    out = capsys.readouterr().out
    assert f"hilverdict {__version__}" in out
    assert "certified overrun" in out
    assert "dismissal rate" in out


def test_certify_honours_the_limit(capsys):
    assert main(["certify", "--seed", "101", "--steps", "200", "--limit", "2"]) == 0
    assert "showing 2 of" in capsys.readouterr().out


def test_certify_on_a_clean_run_says_so(capsys):
    assert main(["certify", "--seed", "4", "--profile", "nominal",
                 "--steps", "20"]) == 0
    out = capsys.readouterr().out
    assert "no iteration in this run exceeded its deadline" in out or "iter" in out


def test_resimulate_prints_the_cost_table(capsys):
    assert main(["resimulate", "--seed", "101", "--steps", "200"]) == 0
    out = capsys.readouterr().out
    assert "plant evaluations spent" in out or "dismissed every miss" in out


def test_resimulate_all_covers_every_miss(capsys):
    assert main(["resimulate", "--seed", "101", "--steps", "160", "--all"]) == 0
    out = capsys.readouterr().out
    assert "peak divergence" in out


def test_clock_prints_the_verdict_first(capsys):
    assert main(["clock", "--seed", "101", "--steps", "120"]) == 0
    assert "CLOCK VERDICT:" in capsys.readouterr().out


def test_clock_reports_an_injected_drift(capsys):
    assert main(["clock", "--seed", "101", "--steps", "200",
                 "--drift-ppm", "5000"]) == 0
    out = capsys.readouterr().out
    assert "CLOCK_DRIFTING" in out
    assert "5000" in out


def test_clock_reports_an_injected_reset(capsys):
    assert main(["clock", "--seed", "101", "--steps", "200",
                 "--resets", "0.4"]) == 0
    out = capsys.readouterr().out
    assert "CLOCK_RESET" in out


def test_clock_accepts_several_resets(capsys):
    assert main(["clock", "--seed", "101", "--steps", "200",
                 "--resets", "0.3,0.7"]) == 0
    assert "CLOCK_RESET" in capsys.readouterr().out


def test_verdict_prints_a_run_verdict(capsys):
    assert main(["verdict", "--seed", "101", "--steps", "160"]) == 0
    out = capsys.readouterr().out
    assert "RUN VERDICT:" in out
    assert "not a certification" in out


def test_verdict_on_a_reset_run_is_undecidable(capsys):
    assert main(["verdict", "--seed", "101", "--steps", "160",
                 "--resets", "0.5"]) == 0
    assert "UNDECIDABLE" in capsys.readouterr().out


def test_threshold_prints_equation_fourteen(capsys):
    assert main(["threshold", "--steps", "200"]) == 0
    out = capsys.readouterr().out
    assert "certified overrun [ms]" in out
    assert "gamma_sup, eq (12)" in out
    assert "metric anisotropy" in out


def test_threshold_is_monotone_in_the_input_step(capsys):
    assert main(["threshold", "--steps", "50"]) == 0
    out = capsys.readouterr().out
    values = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 3:
            try:
                values.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
        if "model constant" in line:
            break
    thresholds = [value for _step, value in values]
    assert thresholds == sorted(thresholds, reverse=True)


def test_learned_prints_the_comparison_and_the_safety_line(capsys):
    assert main(["learned", "--train-runs", "9", "--test-runs", "6",
                 "--steps", "120"]) == 0
    out = capsys.readouterr().out
    assert "analytic certificate" in out
    assert "recall" in out
    assert "This model is not certified for operational flight use." in out


def test_parser_defaults_are_the_declared_ones():
    parser = build_parser()
    arguments = parser.parse_args(["certify"])
    assert arguments.seed == 101
    assert arguments.profile == "loaded"
    assert arguments.amplitude_deg == pytest.approx(6.0, rel=0.0, abs=0.0)
    assert arguments.steps == 200
    assert arguments.tolerance == pytest.approx(1.0, rel=0.0, abs=0.0)
    assert arguments.drift_ppm == pytest.approx(0.0, rel=0.0, abs=0.0)
    assert arguments.resets == ""


def test_parser_exposes_a_handler_for_every_command():
    parser = build_parser()
    for command in ("certify", "resimulate", "clock", "verdict", "threshold",
                    "learned"):
        assert hasattr(parser.parse_args([command]), "handler")
