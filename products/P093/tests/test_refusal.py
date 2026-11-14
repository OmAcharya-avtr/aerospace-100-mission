"""The negative control for the whole discipline: the refusal path exits non-zero.

A validation script that needs measured data and cannot find it must print why
and exit non-zero. It must never silently fall back to the synthetic generator.
These tests verify that the refusal FIRES, as a subprocess, with the exit status
a caller would see.

They are hermetic in both directions. ``POWERENVELOPE_TRACES`` points the
required directory at an empty temporary directory, so the result does not
change now that the campaign's traces ARE committed, and nothing here writes
into ``validation/``. That matters more than it did before the measurement
landed: a refusal test that read the default directory would now pass for the
wrong reason, by finding real data, and would stop testing the refusal at all.
``test_the_refusal_tests_do_not_read_the_committed_campaign`` asserts the
isolation directly.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from fixtures import synthetic_campaign

ROOT = Path(__file__).resolve().parent.parent
VALIDATION = ROOT / "validation"

def _refusal_program() -> str:
    """A program that exercises the refusal and nothing else.

    It touches no committed output file, and it sets its own ``sys.path`` rather
    than relying on ``PYTHONPATH``, because the subprocess runs in isolated mode
    where ``PYTHONPATH`` is ignored. Exit status 7 is chosen so that it cannot
    be confused with pytest's or the interpreter's own statuses.
    """
    paths = [str(ROOT / "src"), str(ROOT / "tests"), str(VALIDATION)]
    return (
        "import sys\n"
        "sys.dont_write_bytecode = True\n"
        f"sys.path[:0] = {paths!r}\n"
        "import _bootstrap\n"
        "_bootstrap.bootstrap()\n"
        "from powerenvelope.trace import TraceDataAbsentError\n"
        "try:\n"
        "    _bootstrap.require_measured_campaign()\n"
        "except TraceDataAbsentError as error:\n"
        "    sys.stderr.write(str(error))\n"
        "    raise SystemExit(7)\n"
        "raise SystemExit(0)\n"
    )


def _environment(traces: Path, tmp_path: Path) -> dict[str, str]:
    import os

    environment = dict(os.environ)
    environment.update(
        {
            "POWERENVELOPE_TRACES": str(traces),
            "PYTHONPATH": ":".join([str(ROOT / "src"), str(ROOT / "tests"), str(VALIDATION)]),
            "PYTHONDONTWRITEBYTECODE": "1",
            "HOME": str(tmp_path / "home"),
            "MPLCONFIGDIR": str(tmp_path / "mpl"),
        }
    )
    return environment


def _run_refusal(traces: Path, tmp_path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-I", "-c", _refusal_program()],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=_environment(traces, tmp_path),
        timeout=300,
        check=False,
    )


def test_refusal_on_an_empty_directory_exits_non_zero(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    completed = _run_refusal(empty, tmp_path)
    assert completed.returncode != 0
    assert completed.returncode == 7, completed.stderr
    assert "MEASURED TRACES ABSENT" in completed.stderr
    assert "no synthetic fallback" in completed.stderr


def test_refusal_names_every_required_cell(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    completed = _run_refusal(empty, tmp_path)
    for cell in ("floor", "w64_d050", "w64_d050_rep4", "w64_d050_permissive"):
        assert cell in completed.stderr


def test_refusal_on_an_incomplete_campaign_exits_non_zero(tmp_path):
    directory = synthetic_campaign.write_campaign_dir(
        tmp_path / "partial", cycles=3, complete=False
    )
    completed = _run_refusal(directory, tmp_path)
    assert completed.returncode == 7
    assert "CAMPAIGN_COMPLETE is absent" in completed.stderr
    assert "writes that marker last of all" in completed.stderr


def test_refusal_on_a_self_declared_synthetic_campaign_exits_non_zero(tmp_path):
    directory = synthetic_campaign.write_campaign_dir(tmp_path / "synthetic", cycles=3)
    completed = _run_refusal(directory, tmp_path)
    assert completed.returncode == 7
    assert "declares itself SYNTHETIC" in completed.stderr
    assert "will not relabel generated data as measurement" in completed.stderr


def test_refusal_does_not_substitute_the_generator(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    completed = _run_refusal(empty, tmp_path)
    # Nothing on stdout at all: no banner, no figures, no partial report.
    assert completed.stdout.strip() == ""


def test_cli_refusal_exits_three_as_a_subprocess(tmp_path):
    import os

    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": str(ROOT / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "HOME": str(tmp_path / "home"),
            "MPLCONFIGDIR": str(tmp_path / "mpl"),
        }
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "powerenvelope",
            "--traces",
            "data/traces_absent_probe",
            "report",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=environment,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 3
    assert "CAMPAIGN_COMPLETE is absent" in completed.stderr
    assert "SYNTHETIC" not in completed.stdout


def test_the_refusal_tests_do_not_read_the_committed_campaign(tmp_path):
    # The measured campaign is now committed under data/traces/. Every refusal
    # test above must still be testing the refusal, which means none of them
    # may reach that directory. The override is asserted to take effect: the
    # message names the temporary directory's own basename and not data/traces.
    empty = tmp_path / "empty"
    empty.mkdir()
    completed = _run_refusal(empty, tmp_path)
    assert completed.returncode == 7
    assert "data/traces/" not in completed.stderr
    assert empty.name in completed.stderr


def test_the_committed_campaign_is_present_and_is_accepted(tmp_path):
    # The other side of the same coin: with no override, the default directory
    # is data/traces and it now holds a complete, non-synthetic campaign, so
    # the same call that refuses above returns. If this fails and the refusal
    # tests pass, the measurement has gone missing.
    import os

    traces = ROOT / "data" / "traces"
    if not (traces / "CAMPAIGN_COMPLETE").exists():
        pytest.skip(
            "the measured campaign is not on disk at data/traces/; the refusal tests "
            "above are the ones that matter in that state"
        )
    environment = _environment(traces, tmp_path)
    environment.pop("POWERENVELOPE_TRACES")
    completed = subprocess.run(
        [sys.executable, "-I", "-c", _refusal_program()],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=environment,
        timeout=600,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stderr.strip() == ""
    assert os.environ.get("POWERENVELOPE_TRACES") is None


@pytest.mark.parametrize(
    "script",
    [
        "validate_measured_headline.py",
        "validate_measured_baselines.py",
        "validate_measured_learned.py",
    ],
)
def test_every_measured_script_declares_the_refusal_and_the_token(script):
    # A structural check, so a future edit cannot remove the refusal without
    # this failing. It reads the source rather than running the script, because
    # running it would overwrite the committed output file.
    text = (VALIDATION / script).read_text()
    assert "require_measured_campaign" in text
    assert "REFUSED -- no measured campaign" in text
    assert "MEASURED_PENDING" in text
    assert "return 1" in text
    assert "synthetic_campaign" not in text.split("def main")[-1]
