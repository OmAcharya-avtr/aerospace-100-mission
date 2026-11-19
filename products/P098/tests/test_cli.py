"""The command-line interface, exercised in-process and as a subprocess."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from evidencestale.__main__ import build_parser, main


def test_help_exits_zero() -> None:
    with pytest.raises(SystemExit) as caught:
        build_parser().parse_args(["--help"])
    assert caught.value.code == 0


def test_version_exits_zero() -> None:
    with pytest.raises(SystemExit) as caught:
        build_parser().parse_args(["--version"])
    assert caught.value.code == 0


def test_a_subcommand_is_required() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_report_runs_from_the_product_root(
    product_root: Path, capsys: pytest.CaptureFixture[str], monkeypatch
) -> None:
    monkeypatch.chdir(product_root)
    assert main(["report", "--corpus", "data/corpus_cyclic.json"]) == 0
    captured = capsys.readouterr().out
    assert "HEADLINE -- stale fraction" in captured
    assert "not flight-qualified" in captured


def test_bracket_runs_and_reports_the_four_bound_inputs(
    product_root: Path, capsys: pytest.CaptureFixture[str], monkeypatch
) -> None:
    monkeypatch.chdir(product_root)
    assert main(["bracket"]) == 0
    captured = capsys.readouterr().out
    for line in (
        "changed artifacts (k)",
        "max out-degree (d)",
        "depth limit (L)",
        "artifacts (N)",
        "bracket holds            : True",
    ):
        assert line in captured


def test_benchmark_runs_on_the_small_cyclic_corpus(
    product_root: Path, capsys: pytest.CaptureFixture[str], monkeypatch
) -> None:
    monkeypatch.chdir(product_root)
    assert main(["benchmark", "--corpus", "data/corpus_cyclic.json"]) == 0
    captured = capsys.readouterr().out
    assert "baseline2_closure_hash" in captured
    assert "learned_forest" in captured


def test_a_missing_corpus_exits_two_with_a_message_on_stderr(
    product_root: Path, capsys: pytest.CaptureFixture[str], monkeypatch
) -> None:
    monkeypatch.chdir(product_root)
    assert main(["report", "--corpus", "data/not_a_corpus.json"]) == 2
    assert "does not exist" in capsys.readouterr().err


def test_an_unknown_declarations_view_is_rejected_by_the_parser() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["report", "--declarations", "guessed"])


def test_module_help_in_a_clean_subprocess(product_root: Path) -> None:
    """``python -m evidencestale --help`` is what the release gate runs."""
    with tempfile.TemporaryDirectory(prefix="evidencestale-home-") as temporary:
        environment = dict(os.environ)
        environment["HOME"] = temporary
        environment["MPLCONFIGDIR"] = str(Path(temporary) / "matplotlib")
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["PYTHONPATH"] = str(product_root / "src")
        completed = subprocess.run(  # noqa: S603 - fixed argument vector
            [sys.executable, "-m", "evidencestale", "--help"],
            cwd=str(product_root),
            capture_output=True,
            env=environment,
            timeout=120,
            check=False,
        )
    assert completed.returncode == 0
    assert b"assurance case" in completed.stdout
