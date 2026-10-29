"""The command-line interface, exercised in-process through main()."""

from __future__ import annotations

import re

import pytest

from tmtcdiff import __version__
from tmtcdiff.__main__ import EXIT_SILENT_FOUND, build_parser, main
from tmtcdiff.changes import ChangeClass
from tmtcdiff.parse import dumps
from tmtcdiff.synth import baseline_dictionary, inject_changes, synthetic_corpus


@pytest.fixture
def pair_files(tmp_path):
    old = baseline_dictionary(40, 2401)
    pair = inject_changes(
        old,
        (ChangeClass.RESCALED, ChangeClass.REMOVED, ChangeClass.RENAMED),
        2402,
    )
    old_path = tmp_path / "old.json"
    new_path = tmp_path / "new.json"
    old_path.write_text(dumps(pair.old), encoding="utf-8")
    new_path.write_text(dumps(pair.new), encoding="utf-8")
    return pair, str(old_path), str(new_path)


@pytest.fixture
def corpus_directory(tmp_path, pair_files):
    pair, _, _ = pair_files
    root = tmp_path / "corpus"
    corpus = synthetic_corpus(pair.old, 2403, names=[c.name for c in pair.truth])
    for source in corpus.sources:
        target = root / source.label
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source.text, encoding="utf-8")
    (root / "notes.bin").write_bytes(b"\x00binary")
    return str(root)


class TestParser:
    def test_a_subcommand_is_required(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_version_exits_zero(self, capsys) -> None:
        with pytest.raises(SystemExit) as excinfo:
            main(["--version"])
        assert excinfo.value.code == 0
        assert __version__ in capsys.readouterr().out

    def test_an_unknown_subcommand_exits_two(self) -> None:
        with pytest.raises(SystemExit) as excinfo:
            main(["nonsense"])
        assert excinfo.value.code == 2


class TestInfo:
    def test_info_runs(self, capsys) -> None:
        assert main(["info"]) == 0
        output = capsys.readouterr().out
        assert "tmtcdiff" in output
        assert "python" in output

    def test_info_prints_the_taxonomy(self, capsys) -> None:
        main(["info"])
        assert "rescaled" in capsys.readouterr().out

    def test_info_names_deepdiff_as_an_alternative(self, capsys) -> None:
        main(["info"])
        assert "deepdiff" in capsys.readouterr().out

    def test_taxonomy_subcommand(self, capsys) -> None:
        assert main(["taxonomy"]) == 0
        assert "argument_reordered" in capsys.readouterr().out


class TestDiffSubcommand:
    def test_diff_runs(self, capsys, pair_files) -> None:
        _, old_path, new_path = pair_files
        assert main(["diff", old_path, new_path]) == 0
        assert "SILENT CHANGES FIRST" in capsys.readouterr().out

    def test_fail_on_silent_changes_the_exit_status(self, pair_files) -> None:
        _, old_path, new_path = pair_files
        assert main(["diff", old_path, new_path, "--fail-on-silent"]) == EXIT_SILENT_FOUND

    def test_no_renames_option(self, capsys, pair_files) -> None:
        _, old_path, new_path = pair_files
        assert main(["diff", old_path, new_path, "--no-renames"]) == 0
        assert "rename detection was disabled" in capsys.readouterr().out

    def test_greedy_strategy_option(self, pair_files) -> None:
        _, old_path, new_path = pair_files
        assert main(["diff", old_path, new_path, "--strategy", "greedy"]) == 0

    def test_a_missing_file_exits_one(self, pair_files, capsys) -> None:
        _, _, new_path = pair_files
        with pytest.raises(SystemExit) as excinfo:
            main(["diff", "absent.json", new_path])
        assert excinfo.value.code == 1
        assert "cannot read" in capsys.readouterr().err

    def test_a_bad_threshold_exits_one(self, pair_files, capsys) -> None:
        _, old_path, new_path = pair_files
        assert main(["diff", old_path, new_path, "--threshold", "5.0"]) == 1
        assert "rename_threshold" in capsys.readouterr().err

    def test_max_rows_is_honoured(self, capsys, pair_files) -> None:
        _, old_path, new_path = pair_files
        main(["diff", old_path, new_path, "--max-rows", "1"])
        assert "more" in capsys.readouterr().out


class TestImpactSubcommand:
    def test_impact_runs(self, capsys, pair_files, corpus_directory) -> None:
        _, old_path, new_path = pair_files
        assert main(["impact", old_path, new_path, "--corpus", corpus_directory]) == 0
        output = capsys.readouterr().out
        assert "coverage verdict" in output

    def test_the_unreadable_source_is_named(self, capsys, pair_files, corpus_directory) -> None:
        _, old_path, new_path = pair_files
        main(["impact", old_path, new_path, "--corpus", corpus_directory])
        assert "notes.bin" in capsys.readouterr().out

    def test_a_missing_corpus_directory_exits_one(self, capsys, pair_files, tmp_path) -> None:
        _, old_path, new_path = pair_files
        assert (
            main(["impact", old_path, new_path, "--corpus", str(tmp_path / "absent")]) == 1
        )
        assert "not a directory" in capsys.readouterr().err


class TestGeneratedSubcommands:
    def test_demo_runs(self, capsys) -> None:
        assert main(["demo", "--size", "30", "--max-rows", "2"]) == 0
        output = capsys.readouterr().out
        assert "injected changes" in output
        assert "interpretation" in output

    def test_demo_can_fail_on_silent(self) -> None:
        assert main(["demo", "--size", "30", "--fail-on-silent"]) == EXIT_SILENT_FOUND

    def test_sweep_runs(self, capsys) -> None:
        assert main(["sweep", "--sizes", "30", "--thresholds", "0.6", "0.8"]) == 0
        assert "false-match rate" in capsys.readouterr().out

    def test_negative_control_runs_and_holds(self, capsys) -> None:
        assert main(["negative-control", "--size", "30"]) == 0
        output = capsys.readouterr().out
        assert "semantic changes reported  : 0" in output
        assert "control held" in output

    def test_negative_control_reports_a_large_text_difference(self, capsys) -> None:
        main(["negative-control", "--size", "30"])
        line = next(
            line
            for line in capsys.readouterr().out.splitlines()
            if line.startswith("text lines differing")
        )
        assert int(line.split(":")[1]) > 100


class TestOutputHygiene:
    @pytest.mark.parametrize(
        "arguments",
        [
            ["info"],
            ["taxonomy"],
            ["demo", "--size", "30", "--max-rows", "2"],
            ["negative-control", "--size", "30"],
        ],
    )
    @pytest.mark.parametrize(
        "word", ["safe", "certified", "flight-qualified", "production-ready"]
    )
    def test_no_subcommand_claims_safety(self, capsys, arguments, word: str) -> None:
        # Word boundaries, not substrings: the generator mints enumeration
        # labels such as SAFE_MODE, and matching those would make this check
        # fail on generated data rather than on a claim.
        main(arguments)
        assert re.search(rf"\b{re.escape(word)}\b", capsys.readouterr().out, re.IGNORECASE) is None

    def test_no_output_contains_an_absolute_path(self, capsys) -> None:
        # The prefixes are assembled from os.sep rather than written as
        # literals: a repository-wide absolute-path scanner would otherwise
        # report this test as the defect it exists to prevent. Three sessions
        # of this mission introduced that defect in the act of writing about
        # it.
        import os

        main(["demo", "--size", "30", "--max-rows", "2"])
        output = capsys.readouterr().out
        prefixes = [os.sep + name + os.sep for name in ("home", "Users", "root")]
        assert not any(prefix in output for prefix in prefixes)
