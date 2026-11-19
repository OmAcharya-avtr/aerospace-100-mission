"""The CLI: every subcommand, its output, and its error handling.

The CLI is exercised in-process through :func:`flakerank.__main__.main` rather
than by spawning a subprocess, because a subprocess would add a second
interpreter start to every test and the gate already runs
``python -m flakerank --help`` independently.
"""

from __future__ import annotations

import pytest

from flakerank.__main__ import build_parser, main


def test_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    """``--help`` is the gate's own smoke test and must exit 0."""
    with pytest.raises(SystemExit) as raised:
        main(["--help"])
    assert raised.value.code == 0
    captured = capsys.readouterr()
    assert "cost of their intermittency" in captured.out
    assert "not flight-qualified" in captured.out.lower()


def test_no_subcommand_is_a_usage_error() -> None:
    """A subcommand is required, and argparse reports that as exit 2."""
    with pytest.raises(SystemExit) as raised:
        main([])
    assert raised.value.code == 2


def test_geometric_subcommand_prints_the_hand_checkable_values(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """p = 0.25, t = 8 s, unbounded: E[F] = 1/3 and wasted = 8/3 s.

    Hand arithmetic: 0.25/0.75 = 0.3333333333333333; times 8 =
    2.6666666666666665.
    """
    assert main(
        ["geometric", "--failure-probability", "0.25", "--failing-attempt-seconds", "8"]
    ) == 0
    captured = capsys.readouterr()
    assert "0.3333333333333333" in captured.out
    assert "2.6666666666666665" in captured.out
    assert "unbounded" in captured.out


def test_geometric_subcommand_with_a_retry_budget(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """p = 0.25, m = 3: E[F_3] = 0.328125, and at t = 8 s that is 2.625 s.

    Hand arithmetic: 0.328125 * 8 = 2.625.
    """
    assert main(
        [
            "geometric",
            "--failure-probability",
            "0.25",
            "--failing-attempt-seconds",
            "8",
            "--retry-budget",
            "3",
        ]
    ) == 0
    captured = capsys.readouterr()
    assert "0.328125" in captured.out
    assert "2.625" in captured.out


def test_geometric_subcommand_reports_the_divergence_as_an_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """p = 1 with no budget is a message and exit 2, never a traceback or ``inf``."""
    assert main(
        ["geometric", "--failure-probability", "1.0", "--failing-attempt-seconds", "8"]
    ) == 2
    captured = capsys.readouterr()
    assert "diverges" in captured.err
    assert "retry_budget" in captured.err
    assert "inf" not in captured.out


def test_geometric_subcommand_rejects_an_out_of_range_probability(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A probability above 1 is a user error with an actionable message."""
    assert main(
        ["geometric", "--failure-probability", "1.5", "--failing-attempt-seconds", "8"]
    ) == 2
    assert "must lie in [0, 1]" in capsys.readouterr().err


def test_rank_subcommand_on_the_committed_history(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The default history path resolves without the caller's working directory."""
    assert main(["rank", "--resamples", "30", "--top", "3", "--movers", "2"]) == 0
    captured = capsys.readouterr()
    assert "THE HEADLINE" in captured.out
    assert "soak/test_validation_probe_1_in_44" in captured.out
    assert "research-grade" in captured.out


def test_disagree_subcommand_prints_only_the_headline(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The one-screen form carries both statistics and their units."""
    assert main(["disagree", "--resamples", "30"]) == 0
    captured = capsys.readouterr()
    assert "Kendall tau-b" in captured.out
    assert "Spearman rho" in captured.out
    assert "THE HEADLINE" not in captured.out


def test_rank_subcommand_reports_a_missing_history(
    capsys: pytest.CaptureFixture[str], tmp_path
) -> None:
    """A missing file is a message naming the generator, and exit 2."""
    assert main(["rank", "--history", str(tmp_path / "absent.json")]) == 2
    assert "data/generate_history.py" in capsys.readouterr().err


def test_rank_subcommand_reports_an_invalid_history(
    capsys: pytest.CaptureFixture[str], tmp_path
) -> None:
    """A file with the wrong schema is a message, not a traceback."""
    bad = tmp_path / "bad.json"
    bad.write_text('{"schema": "nope"}', encoding="utf-8")
    assert main(["rank", "--history", str(bad)]) == 2
    assert "unsupported history schema" in capsys.readouterr().err


def test_parser_declares_every_subcommand() -> None:
    """Three subcommands, so a future addition cannot be silently undocumented."""
    parser = build_parser()
    actions = [
        action for action in parser._subparsers._actions  # type: ignore[union-attr]
        if hasattr(action, "choices") and action.choices
    ]
    assert actions
    assert set(actions[-1].choices) == {"rank", "disagree", "geometric"}
