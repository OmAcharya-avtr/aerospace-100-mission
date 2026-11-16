"""Pristine-copy execution and the two-arrangement classifier."""

from __future__ import annotations

from pathlib import Path

import pytest

from seedaudit import dynamic


def test_the_classifier_cross_tabulation() -> None:
    # A verdict that changes between two PRISTINE runs is intrinsic. One that
    # changes between run 1 and run 2 of the SAME tree while the pristine pair
    # agrees is self-pollution. That distinction is the detection mechanism.
    assert dynamic.classify(True, False) == dynamic.SELF_POLLUTING
    assert dynamic.classify(False, False) == dynamic.INTRINSIC
    assert dynamic.classify(False, True) == dynamic.INTRINSIC_MASKED
    assert dynamic.classify(True, True) == dynamic.NO_EVIDENCE
    assert (
        len(
            {
                dynamic.INTRINSIC,
                dynamic.INTRINSIC_MASKED,
                dynamic.SELF_POLLUTING,
                dynamic.NO_EVIDENCE,
            }
        )
        == 4
    )


def test_the_subprocess_environment_keeps_its_writes_outside_the_tree(tmp_path: Path) -> None:
    # An earlier product of this mission set HOME to the product root and
    # matplotlib wrote .config/matplotlib into the repository.
    env = dynamic.subprocess_environment(tmp_path / "home")
    assert env["HOME"] == str(tmp_path / "home")
    assert env["MPLCONFIGDIR"].startswith(env["HOME"])
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert env["MPLBACKEND"] == "Agg"
    assert "PYTHONPATH" not in env


def test_the_hash_seed_is_random_by_default_and_pinnable(tmp_path: Path) -> None:
    # The power model assumes independent repeats, and a fixed hash seed makes
    # container iteration order identical across them.
    assert dynamic.subprocess_environment(tmp_path)["PYTHONHASHSEED"] == "random"
    assert dynamic.subprocess_environment(tmp_path, "0")["PYTHONHASHSEED"] == "0"


def test_a_pristine_copy_never_touches_the_source(tmp_path: Path) -> None:
    source = tmp_path / "tree"
    source.mkdir()
    (source / "a.txt").write_text("original", encoding="utf-8")
    with dynamic.pristine_copy(source) as (copy, _home):
        (copy / "a.txt").write_text("modified", encoding="utf-8")
        (copy / "new.txt").write_text("added", encoding="utf-8")
    assert (source / "a.txt").read_text(encoding="utf-8") == "original"
    assert not (source / "new.txt").exists()


def test_a_pristine_copy_omits_caches_and_puts_home_outside_itself(tmp_path: Path) -> None:
    # Copying a cache would carry one run's state into the next, which is the
    # confound the two arrangements exist to separate.
    source = tmp_path / "tree"
    (source / "__pycache__").mkdir(parents=True)
    (source / "__pycache__" / "x.pyc").write_bytes(b"\x00")
    (source / ".ruff_cache").mkdir()
    (source / "keep.py").write_text("x = 1\n", encoding="utf-8")
    with dynamic.pristine_copy(source) as (copy, home):
        assert (copy / "keep.py").is_file()
        assert not (copy / "__pycache__").exists()
        assert not (copy / ".ruff_cache").exists()
        assert home.is_dir()
        assert not str(home).startswith(str(copy) + "/")


def test_a_pristine_copy_rejects_a_file(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="existing directory"):
        with dynamic.pristine_copy(target):
            pass


def test_a_run_captures_stdout_and_the_exit_code(tmp_path: Path) -> None:
    source = tmp_path / "tree"
    source.mkdir()
    (source / "s.py").write_text("print('hello')\n", encoding="utf-8")
    result = dynamic.run_once(source, "s.py", timeout=30.0)
    assert (result.exit_code, result.ok, result.timed_out) == (0, True, False)
    assert result.stdout.strip() == "hello"
    (source / "bad.py").write_text("import sys\nsys.exit(3)\n", encoding="utf-8")
    failed = dynamic.run_once(source, "bad.py", timeout=30.0)
    assert failed.exit_code == 3
    assert not failed.ok


def test_a_run_reads_its_named_artifact_and_reports_a_missing_one(tmp_path: Path) -> None:
    source = tmp_path / "tree"
    source.mkdir()
    (source / "s.py").write_text(
        "from pathlib import Path\n"
        "Path(__file__).resolve().parent.joinpath('out.txt').write_text('42\\n')\n",
        encoding="utf-8",
    )
    result = dynamic.run_once(source, "s.py", "out.txt", timeout=30.0)
    assert result.artifact == "42\n"
    assert result.verdict == (0, "42\n")
    assert dynamic.run_once(source, "s.py", "absent.txt", timeout=30.0).artifact is None


def test_a_source_transform_is_applied_inside_the_copy_only(tmp_path: Path) -> None:
    # This is how a seed is changed: the rewrite happens in the copy and the
    # audited tree is never touched.
    source = tmp_path / "tree"
    source.mkdir()
    (source / "s.py").write_text("print(1)\n", encoding="utf-8")
    result = dynamic.run_once(
        source, "s.py", timeout=30.0, transform=lambda text: text.replace("1", "2")
    )
    assert result.stdout.strip() == "2"
    assert (source / "s.py").read_text(encoding="utf-8") == "print(1)\n"


def test_a_timeout_is_reported_rather_than_raised(tmp_path: Path) -> None:
    source = tmp_path / "tree"
    source.mkdir()
    (source / "s.py").write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    result = dynamic.run_once(source, "s.py", timeout=1.0)
    assert result.timed_out
    assert result.exit_code is None


def test_repeat_runs_returns_one_result_per_repeat(tmp_path: Path) -> None:
    source = tmp_path / "tree"
    source.mkdir()
    (source / "s.py").write_text("print('x')\n", encoding="utf-8")
    results = dynamic.repeat_runs(source, "s.py", 3, timeout=30.0)
    assert len(results) == 3
    assert all(result.ok for result in results)


def test_input_validation(tmp_path: Path) -> None:
    source = tmp_path / "tree"
    source.mkdir()
    (source / "s.py").write_text("pass\n", encoding="utf-8")
    with pytest.raises(ValueError, match="script not found"):
        dynamic.run_once(source, "absent.py", timeout=5.0)
    with pytest.raises(ValueError, match="at least 1"):
        dynamic.repeat_runs(source, "s.py", 0)
    with pytest.raises(ValueError, match="at least 2 runs"):
        dynamic.two_arrangements(source, "s.py", pristine_repeats=1)
    with dynamic.pristine_copy(source) as (copy, home):
        with pytest.raises(ValueError, match="timeout must be positive"):
            dynamic.execute(copy, home, "s.py", timeout=0.0)


def test_the_two_arrangements_detect_the_self_polluting_check(polluting_tree: Path) -> None:
    # The signature: two pristine runs agree, the reused tree disagrees.
    result = dynamic.two_arrangements(
        polluting_tree, "self_polluting.py", "self_polluting_output.txt", timeout=60.0
    )
    assert result.pristine_agree
    assert not result.reused_agree
    assert result.classification == dynamic.SELF_POLLUTING


def test_the_two_arrangements_stay_silent_on_a_deterministic_control(
    polluting_tree: Path,
) -> None:
    result = dynamic.two_arrangements(
        polluting_tree, "deterministic_control.py", None, timeout=60.0
    )
    assert result.pristine_agree
    assert result.reused_agree
    assert result.classification == dynamic.NO_EVIDENCE
    assert result.escaped_bound == pytest.approx(0.7763932022500211, rel=1e-12, abs=0.0)


def test_an_always_manifesting_script_is_invisible_to_a_reference_free_design(
    polluting_tree: Path,
) -> None:
    from seedaudit.corpus import rate_script_name

    result = dynamic.two_arrangements(polluting_tree, rate_script_name(1.0), None, timeout=60.0)
    # At rate 1.0 every run manifests, so every run agrees with every other.
    # NO_EVIDENCE is the honest answer for a design with no committed reference.
    assert result.classification == dynamic.NO_EVIDENCE


def test_result_accessors_and_the_declared_default_timeout() -> None:
    empty = dynamic.ArrangementResult(script="s.py")
    assert empty.pristine_agree and empty.reused_agree
    with_artifact = dynamic.RunResult(
        exit_code=0, stdout="printed", stderr="", duration=0.1, artifact="file", timed_out=False
    )
    without = dynamic.RunResult(
        exit_code=0, stdout="printed", stderr="", duration=0.1, artifact=None, timed_out=False
    )
    assert with_artifact.verdict == (0, "file")
    assert without.verdict == (0, "printed")
    assert dynamic.DEFAULT_TIMEOUT == 75.0
