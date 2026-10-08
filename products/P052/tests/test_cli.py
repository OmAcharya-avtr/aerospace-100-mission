"""The CLI: every subcommand, every exit status, every validation path."""

from __future__ import annotations

import io
import subprocess
import sys

import pytest

from falsifyloop import __version__
from falsifyloop.__main__ import main


def _run(argv) -> tuple[int, str]:
    out = io.StringIO()
    status = main(argv, out=out)
    return status, out.getvalue()


def test_help_exits_zero() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_module_help_exits_zero_as_a_subprocess() -> None:
    # The build guide requires `python -m falsifyloop --help` to exit 0, so it is
    # checked through the actual module entry point and not only through main().
    # The package directory is put on PYTHONPATH explicitly so the test works
    # from a cold clone, where nothing is installed and pytest's own pythonpath
    # setting does not reach a child process.
    import os
    import pathlib

    import falsifyloop

    env = dict(os.environ)
    src = str(pathlib.Path(falsifyloop.__file__).resolve().parents[1])
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [sys.executable, "-m", "falsifyloop", "--help"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert completed.returncode == 0
    assert "falsifyloop" in completed.stdout


def test_version_exits_zero_and_reports_the_package_version() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ == "0.1.0"


def test_no_subcommand_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


def test_instances_lists_the_whole_suite() -> None:
    status, text = _run(["instances"])
    assert status == 0
    assert "overshoot-loose" in text
    assert "rate-envelope" in text
    assert "design target" in text


def test_falsify_reports_a_found_violation() -> None:
    status, text = _run(
        ["falsify", "--instance", "overshoot-loose", "--budget", "60", "--seed", "0"]
    )
    assert status == 0
    assert "VIOLATION FOUND at simulation" in text


def test_falsify_reports_nothing_found_without_implying_correctness() -> None:
    status, text = _run(
        ["falsify", "--instance", "rate-envelope", "--budget", "5", "--seed", "0"]
    )
    assert status == 0
    assert "NO VIOLATION FOUND within 5 simulations." in text
    assert "not evidence of correctness" in text


def test_exit_on_violation_changes_only_the_found_case() -> None:
    found, _ = _run(
        [
            "falsify",
            "--instance",
            "overshoot-loose",
            "--budget",
            "60",
            "--seed",
            "0",
            "--exit-on-violation",
        ]
    )
    nothing, _ = _run(
        [
            "falsify",
            "--instance",
            "rate-envelope",
            "--budget",
            "5",
            "--seed",
            "0",
            "--exit-on-violation",
        ]
    )
    assert found == 1
    # The absence of a violation is never encoded in the exit status.
    assert nothing == 0


def test_falsify_with_repeats_prints_the_curve() -> None:
    status, text = _run(
        [
            "falsify",
            "--instance",
            "overshoot-tight",
            "--strategy",
            "surrogate-guided",
            "--budget",
            "20",
            "--repeats",
            "4",
            "--seed",
            "3",
        ]
    )
    assert status == 0
    assert "first-violation simulation index per seed" in text
    assert "P(found by n)" in text
    assert "95% band" in text


def test_falsify_rejects_a_non_positive_budget_with_status_two() -> None:
    status, _ = _run(
        ["falsify", "--instance", "overshoot-loose", "--budget", "0", "--seed", "0"]
    )
    assert status == 2


def test_falsify_rejects_a_non_positive_repeat_count() -> None:
    status, _ = _run(
        [
            "falsify",
            "--instance",
            "overshoot-loose",
            "--budget",
            "10",
            "--repeats",
            "0",
        ]
    )
    assert status == 2


def test_benchmark_prints_per_instance_before_aggregate() -> None:
    status, text = _run(
        [
            "benchmark",
            "--instances",
            "overshoot-loose",
            "overshoot-tight",
            "--strategies",
            "uniform-random",
            "latin-hypercube",
            "--budget",
            "15",
            "--repeats",
            "3",
        ]
    )
    assert status == 0
    per_instance = text.index("median sims")
    aggregate = text.index("mean P (aggregate)")
    assert per_instance < aggregate, "the aggregate table must come after the per-instance one"
    assert "wall clock" in text
    assert "not a hardware characteristic" in text


def test_evaluate_reports_robustness_at_one_point() -> None:
    status, text = _run(
        [
            "evaluate",
            "--instance",
            "overshoot-loose",
            "--input",
            "5,1.9,0.3,2.5,10,0.8",
        ]
    )
    assert status == 0
    assert "robustness  :" in text
    assert "VIOLATION" in text


def test_evaluate_reports_a_satisfying_point_without_claiming_correctness() -> None:
    status, text = _run(
        ["evaluate", "--instance", "rate-envelope", "--input", "1,0.6,1.2,0.6,0,0.1"]
    )
    assert status == 0
    assert "NO VIOLATION at this single point" in text
    assert "not evidence of correctness" in text


@pytest.mark.parametrize(
    "value",
    ["1,2,3", "1,2,3,4,5,6,7", "a,b,c,d,e,f", "100,1,1,1,1,1"],
)
def test_evaluate_input_validation(value) -> None:
    status, _ = _run(["evaluate", "--instance", "overshoot-loose", "--input", value])
    assert status == 2


def test_evaluate_accepts_spaces_in_the_input_list() -> None:
    status, _ = _run(
        ["evaluate", "--instance", "overshoot-loose", "--input", "3.5, 1.3, 0.7, 1.8, 7.5, 1.5"]
    )
    assert status == 0


def test_difficulty_reports_counts_and_an_exact_interval() -> None:
    status, text = _run(
        ["difficulty", "--instances", "overshoot-loose", "--draws", "40", "--seed", "1"]
    )
    assert status == 0
    assert "p measured" in text
    assert "Clopper-Pearson exact" in text
    assert "widest interval:" in text


def test_difficulty_rejects_a_non_positive_draw_count() -> None:
    status, _ = _run(["difficulty", "--draws", "0"])
    assert status == 2


def test_unknown_instance_is_a_usage_error_from_argparse() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["falsify", "--instance", "nope"])
    assert exc.value.code == 2


def test_unknown_strategy_is_a_usage_error_from_argparse() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["falsify", "--instance", "overshoot-loose", "--strategy", "nope"])
    assert exc.value.code == 2
