"""Tests for the ``python -m zernkit`` command-line interface."""

from __future__ import annotations

import pytest

from zernkit.cli import main


def test_index_from_noll(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["index", "--noll", "8"]) == 0
    out = capsys.readouterr().out
    assert "3   +1" in out
    assert "horizontal coma" in out
    assert out.splitlines()[-1].split()[1] == "8"  # OSA index of Noll 8


def test_index_from_osa(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["index", "--osa", "0"]) == 0
    assert "piston" in capsys.readouterr().out


def test_index_from_nm(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["index", "--nm", "2", "0"]) == 0
    assert "defocus" in capsys.readouterr().out


def test_index_table(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["index", "--max-order", "2"]) == 0
    out = capsys.readouterr().out
    assert len(out.strip().splitlines()) == 2 + 6  # title + header + 6 modes


def test_noll_table(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["noll-table", "--j-max", "5"]) == 0
    out = capsys.readouterr().out
    assert "1.0299" in out
    assert "%" in out


def test_invalid_input_exits_with_code_2() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["index", "--noll", "0"])
    assert exc.value.code == 2


def test_missing_subcommand_exits() -> None:
    with pytest.raises(SystemExit):
        main([])
