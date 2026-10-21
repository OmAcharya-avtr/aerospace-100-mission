"""The command-line interface, in process."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from downlinkpick import __version__
from downlinkpick.__main__ import SELECTORS, build_parser, main

ROOT = Path(__file__).resolve().parent.parent
CASE = str(ROOT / "examples" / "cases" / "three_pass_declared.yaml")
SMALL = str(ROOT / "examples" / "cases" / "single_short_pass.yaml")
MISSING = str(ROOT / "examples" / "cases" / "missing_declarations.yaml")


def run(argv: list[str]) -> tuple[int, str]:
    stream = io.StringIO()
    status = main(argv, stream=stream)
    return status, stream.getvalue()


def test_version_flag() -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code in (0, None)


def test_no_subcommand_is_an_error() -> None:
    with pytest.raises(SystemExit):
        main([])


def test_info_reports_the_backend() -> None:
    status, text = run(["info"])
    assert status == 0
    assert f"downlinkpick     : {__version__}" in text
    assert "pulp solvers" in text
    assert "exact backend" in text


def test_select_prints_the_full_report() -> None:
    status, text = run(["select", SMALL, "--method", "exact"])
    assert status == 0
    assert "contact plan" in text
    assert "left behind" in text
    assert "marginal value of contact time" in text
    assert "invariants: all hold" in text


def test_select_with_greedy() -> None:
    status, text = run(["select", SMALL, "--method", "greedy"])
    assert status == 0
    assert "method greedy" in text


def test_compare_reports_the_gap() -> None:
    status, text = run(["compare", CASE])
    assert status == 0
    assert "greedy value" in text
    assert "exact value" in text
    assert "relative gap" in text
    assert "LP relaxation bound" in text
    assert "Dantzig bound" in text


def test_marginal_prices_both_selectors() -> None:
    status, text = run(["marginal", SMALL, "--delta-s", "30"])
    assert status == 0
    assert "selector: greedy" in text
    assert "selector: exact" in text


def test_stability_prints_a_curve() -> None:
    status, text = run(
        ["stability", SMALL, "--trials", "4", "--sigmas", "0.0", "0.1"]
    )
    assert status == 0
    lines = [line for line in text.splitlines() if line.strip()]
    assert len(lines) == 3
    assert "sigma" in lines[0]


def test_lp_writes_a_file(tmp_path: Path) -> None:
    target = tmp_path / "m.lp"
    status, text = run(["lp", SMALL, str(target)])
    assert status == 0
    assert target.exists()
    assert "wrote" in text
    assert "Maximize" in target.read_text()


def test_missing_file_exits_two() -> None:
    status, _text = run(["select", str(ROOT / "examples" / "cases" / "absent.yaml")])
    assert status == 2


def test_undeclared_values_exit_two() -> None:
    status, _text = run(["select", MISSING])
    assert status == 2


def test_a_file_that_is_not_a_case_exits_two() -> None:
    status, _text = run(["compare", str(ROOT / "pyproject.toml")])
    assert status == 2


def test_parser_exposes_the_subcommands() -> None:
    parser = build_parser()
    assert parser.prog == "downlinkpick"
    args = parser.parse_args(["select", CASE])
    assert args.method == "exact"
    assert args.delta_s == 1.0
    assert set(SELECTORS) == {"greedy", "exact"}


def test_parser_rejects_an_unknown_method() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["select", CASE, "--method", "psychic"])


def test_broken_pipe_exits_quietly() -> None:
    """A CLI piped into `head` must not print a traceback.

    `python -m downlinkpick info | head -3` closes the pipe while the handler is
    still writing. Without handling, the `print` raises BrokenPipeError, the
    traceback reaches the user and the exit status is 1.
    """

    class ClosedStream(io.StringIO):
        def write(self, _text: str) -> int:
            raise BrokenPipeError(32, "Broken pipe")

    status = main(["info"], stream=ClosedStream())
    assert status == 0
