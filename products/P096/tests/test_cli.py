"""The CLI: every subcommand, its exit status and its refusals."""

from __future__ import annotations

import pytest

from handoverloss.__main__ import build_parser, main

SEQUENCE_ARGUMENTS = [
    "sequence",
    "--source-rate", "2e8",
    "--downlink-rate", "1e9",
    "--buffer", "3.2e10",
    "--occupancy", "2.9e10",
    "--acquisition", "40",
    "--restart", "12",
    "--ramp", "60",
    "--horizon", "700",
]


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as raised:
        build_parser().parse_args(["--help"])
    assert raised.value.code == 0
    assert "handoverloss" in capsys.readouterr().out


def test_version_exits_zero(capsys):
    with pytest.raises(SystemExit) as raised:
        main(["--version"])
    assert raised.value.code == 0
    assert "0.1.0" in capsys.readouterr().out


def test_no_subcommand_is_refused():
    with pytest.raises(SystemExit) as raised:
        main([])
    assert raised.value.code == 2


def test_sequence_subcommand_prints_the_decomposition_and_the_bracket(capsys):
    assert main(SEQUENCE_ARGUMENTS) == 0
    out = capsys.readouterr().out
    # Hand-calculated values, as in tests/test_baselines.py.
    assert "5.000000e+09" in out
    assert "2.400000e+09" in out
    assert "1.200000e+09" in out
    assert "8.600000e+09" in out
    assert "dominant       acquisition" in out
    assert "independently derived bracket" in out
    assert "brackets the fluid total: True" in out


def test_sequence_subcommand_rejects_an_overfull_buffer(capsys):
    arguments = list(SEQUENCE_ARGUMENTS)
    arguments[arguments.index("--occupancy") + 1] = "4e10"
    assert main(arguments) == 2
    assert "exceeds buffer_capacity_bits" in capsys.readouterr().err


def test_sequence_subcommand_rejects_a_short_horizon(capsys):
    arguments = list(SEQUENCE_ARGUMENTS)
    arguments[arguments.index("--horizon") + 1] = "50"
    assert main(arguments) == 2
    assert "is shorter than" in capsys.readouterr().err


def test_attribute_subcommand(capsys):
    assert main(["attribute", "--events", "48", "--resamples", "120"]) == 0
    out = capsys.readouterr().out
    assert "corpus sha256" in out
    assert "resampling unit        : handover event" in out
    assert "decomposition residual" in out
    assert "events                 : 48" in out


@pytest.mark.parametrize("split", ["station_pairs", "time"])
def test_baselines_subcommand(capsys, split):
    assert main(["baselines", "--split", split, "--resamples", "100"]) == 0
    out = capsys.readouterr().out
    assert "baseline1_acquisition_only" in out
    assert "baseline2_analytic_fluid" in out
    assert f"split: {split}" in out


def test_learned_subcommand_prints_the_baselines_first(capsys):
    assert main(["learned", "--split", "time", "--resamples", "100"]) == 0
    out = capsys.readouterr().out
    assert "baselines first" in out
    baseline_position = out.index("baseline1_acquisition_only")
    learned_position = out.index("learned_random_forest")
    assert baseline_position < learned_position
    assert "highest dominance accuracy on this split:" in out
    assert "mean dominance confidence" in out


def test_learned_subcommand_rejects_an_unknown_split():
    with pytest.raises(SystemExit) as raised:
        main(["learned", "--split", "random"])
    assert raised.value.code == 2


def test_missing_corpus_gives_exit_two(monkeypatch, capsys, tmp_path):
    from handoverloss import __main__ as module
    from handoverloss.corpus import CorpusAbsentError

    def absent(*_args, **_kwargs):
        raise CorpusAbsentError("the committed corpus is absent; generate_corpus.py")

    monkeypatch.setattr(module, "load_corpus", absent)
    assert main(["baselines"]) == 2
    assert "generate_corpus.py" in capsys.readouterr().err
