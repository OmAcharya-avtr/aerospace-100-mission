"""The command-line interface, exercised through ``main`` and its parser."""

from __future__ import annotations

import pytest

from orbitdrift.__main__ import DEMO_TLE, build_parser, main


def test_parser_requires_a_subcommand() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_version_prints_the_environment_and_names_the_absent_packages(capsys) -> None:
    """The version block must say what is **not** installed.

    ``astropy`` and ``torch`` are named as absent in the output, so a reader
    of a transcript cannot come away thinking a number here was cross-checked
    against either.
    """
    assert main(["version"]) == 0
    out = capsys.readouterr().out
    assert "orbitdrift 0.1.0" in out
    assert "astropy      not installed" in out
    assert "torch        not installed" in out
    assert "not flight-qualified" in out


def test_vectors_reproduces_the_published_set(capsys) -> None:
    assert main(["vectors"]) == 0
    out = capsys.readouterr().out
    assert "33 cases, 666 rows compared, 1 rows skipped" in out
    assert "worst position disagreement" in out


def test_vectors_verbose_prints_a_line_per_case(capsys) -> None:
    assert main(["vectors", "--verbose"]) == 0
    out = capsys.readouterr().out
    assert out.count("\n") > 40


def test_atmosphere_prints_the_table_and_its_caveat(capsys) -> None:
    assert main(["atmosphere"]) == 0
    out = capsys.readouterr().out
    assert "Table 8-4" in out
    assert "This model is static" in out
    assert "3.725000e-12" in out


def test_drift_runs_on_the_built_in_element_set(capsys) -> None:
    assert main(["drift", "--span-days", "0.6", "--steps-per-rev", "90"]) == 0
    out = capsys.readouterr().out
    assert "model divergence" in out
    assert "lower bound on larger model error" in out
    assert "analytic growth forms fitted" in out


def test_drift_accepts_two_lines_on_the_command_line(capsys) -> None:
    assert (
        main(
            [
                "drift",
                "--line1",
                DEMO_TLE[0],
                "--line2",
                DEMO_TLE[1],
                "--span-days",
                "0.5",
                "--steps-per-rev",
                "90",
            ]
        )
        == 0
    )
    assert "catalogue 25544" in capsys.readouterr().out


def test_drift_reads_a_file(tmp_path, capsys) -> None:
    path = tmp_path / "object.tle"
    path.write_text("\n".join(DEMO_TLE) + "\n")
    assert (
        main(["drift", "--tle-file", str(path), "--span-days", "0.5", "--steps-per-rev", "90"])
        == 0
    )
    assert "catalogue 25544" in capsys.readouterr().out


def test_drift_reports_a_file_with_no_element_lines(tmp_path, capsys) -> None:
    path = tmp_path / "empty.tle"
    path.write_text("nothing useful here\n")
    assert main(["drift", "--tle-file", str(path)]) == 2
    assert "does not contain a line starting" in capsys.readouterr().err


def test_drift_term_switches_change_the_force_label(capsys) -> None:
    assert (
        main(
            [
                "drift",
                "--span-days",
                "0.5",
                "--steps-per-rev",
                "90",
                "--no-drag",
                "--no-third-body",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "WGS72: two-body + J2-J4," in out
    assert "drag" not in out.split("divergence source")[1].split("\n")[1]


def test_stale_prints_both_horizons_and_the_unbacked_warning(capsys) -> None:
    assert (
        main(["stale", "--span-days", "0.6", "--steps-per-rev", "90", "--tolerance-km", "20"])
        == 0
    )
    out = capsys.readouterr().out
    assert "median horizon" in out
    assert "conservative horizon" in out
    assert "no measured backing" in out


def test_stale_with_an_age_reports_a_verdict(capsys) -> None:
    assert (
        main(
            [
                "stale",
                "--span-days",
                "0.6",
                "--steps-per-rev",
                "90",
                "--tolerance-km",
                "20",
                "--age-days",
                "0.4",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "element set age" in out
    assert "verdict" in out


def test_stale_refuses_a_conjunction_purpose_with_exit_status_three(capsys) -> None:
    status = main(
        [
            "stale",
            "--span-days",
            "0.4",
            "--steps-per-rev",
            "90",
            "--purpose",
            "covariance for conjunction screening",
        ]
    )
    assert status == 3
    err = capsys.readouterr().err
    assert "refused:" in err
    assert "P071" in err


def test_help_text_says_what_the_tool_does_not_produce(capsys) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--help"])
    assert caught.value.code == 0
    out = capsys.readouterr().out
    assert "No output is a covariance" in out
    assert "P071 ConjProb" in out
    assert "not flight-qualified" in out
    assert "does not" in out
