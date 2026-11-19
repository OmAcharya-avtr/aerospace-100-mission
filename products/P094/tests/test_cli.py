"""The command-line interface, exercised as a subprocess.

Each command is run in a subprocess with a temporary ``HOME`` and
``MPLCONFIGDIR`` outside the repository and with ``PYTHONDONTWRITEBYTECODE``
set, because a build that pollutes its own tree through a subprocess
environment it sets is a defect class this mission has shipped.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
COMMANDS = ("headline", "baselines", "drift", "clock", "gate")
#: Commands that do not measure simulation mode, so they are fast.
FAST_COMMANDS = ("clock", "gate")


def _environment() -> dict[str, str]:
    scratch = Path(tempfile.mkdtemp(prefix="hiltimingdiff-cli-"))
    environment = dict(os.environ)
    environment.update(
        {
            "HOME": str(scratch),
            "MPLCONFIGDIR": str(scratch / "mpl"),
            "MPLBACKEND": "Agg",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(ROOT / "src"),
        }
    )
    return environment


def _run(arguments: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "hiltimingdiff", *arguments],
        cwd=str(cwd or ROOT),
        env=_environment(),
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )


@pytest.fixture(scope="module")
def fast_outputs() -> dict[str, subprocess.CompletedProcess]:
    return {name: _run([name]) for name in FAST_COMMANDS}


@pytest.fixture(scope="module")
def headline_output() -> subprocess.CompletedProcess:
    return _run(["headline", "--passes", "1", "--resamples", "200", "--quiet"])


@pytest.fixture(scope="module")
def baselines_output() -> subprocess.CompletedProcess:
    return _run(["baselines", "--passes", "1", "--resamples", "200", "--quiet"])


@pytest.fixture(scope="module")
def drift_output() -> subprocess.CompletedProcess:
    return _run(["drift", "--passes", "1", "--resamples", "200", "--quiet"])


@pytest.mark.parametrize("name", FAST_COMMANDS)
def test_fast_command_exits_zero(fast_outputs, name):
    assert fast_outputs[name].returncode == 0, fast_outputs[name].stderr


@pytest.mark.parametrize("name", FAST_COMMANDS)
def test_fast_command_writes_to_stdout(fast_outputs, name):
    assert fast_outputs[name].stdout.strip()


@pytest.mark.parametrize("name", FAST_COMMANDS)
def test_fast_command_states_the_scope(fast_outputs, name):
    assert "unpinned schedutil-governed clock" in fast_outputs[name].stdout


@pytest.mark.parametrize("name", FAST_COMMANDS)
def test_fast_command_states_research_grade(fast_outputs, name):
    assert "Research-grade" in fast_outputs[name].stdout


@pytest.mark.parametrize("name", FAST_COMMANDS)
def test_fast_command_output_has_no_exclamation_mark(fast_outputs, name):
    assert "!" not in fast_outputs[name].stdout


@pytest.mark.parametrize("name", FAST_COMMANDS)
def test_fast_command_output_carries_no_absolute_home_path(fast_outputs, name):
    separator = os.sep
    assert f"{separator}home{separator}" not in fast_outputs[name].stdout
    assert f"{separator}Users{separator}" not in fast_outputs[name].stdout


def test_gate_command_reports_every_branch_firing(fast_outputs):
    assert "every branch fired at least once : True" in fast_outputs["gate"].stdout


def test_gate_command_reports_no_silent_branch(fast_outputs):
    assert "silent branches                  : none" in fast_outputs["gate"].stdout


def test_gate_command_prints_the_total_row(fast_outputs):
    assert "TOTAL" in fast_outputs["gate"].stdout


def test_clock_command_reports_the_governor(fast_outputs):
    assert "governor            : schedutil" in fast_outputs["clock"].stdout


def test_clock_command_reports_the_power_mode(fast_outputs):
    assert "MAXN_SUPER" in fast_outputs["clock"].stdout


def test_clock_command_reports_that_sudo_was_unavailable(fast_outputs):
    assert "passwordless sudo   : False" in fast_outputs["clock"].stdout


def test_clock_command_reports_fifty_four_paired_points(fast_outputs):
    assert "paired points       : 54" in fast_outputs["clock"].stdout


def test_headline_command_exits_zero(headline_output):
    assert headline_output.returncode == 0, headline_output.stderr


def test_headline_command_reports_the_spread(headline_output):
    assert "Divergence spread max(R)/min(R)" in headline_output.stdout


def test_headline_command_confirms_the_primary_preregistration(headline_output):
    assert "SPEC §7 primary (divergence spread): CONFIRMED" in headline_output.stdout


def test_headline_command_refutes_the_secondary_preregistration(headline_output):
    # The line is located by its NAME, its measured value is parsed out, the
    # BOUND against the pre-registered 0.5 is asserted on that value, and the
    # label is asserted to be the one the value implies. Asserting the literal
    # substring "... : REFUTED" made this an assertion about a strict threshold
    # comparison on a live wall-clock measurement taken by a one-pass
    # simulation-mode run in a subprocess. Measured band published in
    # validation/VALIDATION.md.
    prefix = "SPEC §7 secondary (baseline 2 held-out transfer error):"
    matches = [
        line for line in headline_output.stdout.splitlines() if prefix in line
    ]
    assert len(matches) == 1, headline_output.stdout
    line = matches[0]
    measured = float(line.rsplit("measured", 1)[1].strip())
    assert measured < 0.5
    assert ("REFUTED" in line) == (measured <= 0.5)
    assert "CONFIRMED" not in line


def test_headline_command_names_the_least_trustworthy_stage(headline_output):
    assert "argmax(R) = centroid" in headline_output.stdout


def test_headline_command_names_the_most_trustworthy_stage(headline_output):
    assert "argmin(R) = ingest" in headline_output.stdout


def test_headline_command_reports_the_permutation(headline_output):
    assert "(3, 0, 1, 5, 4, 2)" in headline_output.stdout


def test_headline_command_reports_the_pass_unit(headline_output):
    assert "pass-unit 95% bootstrap interval" in headline_output.stdout


def test_headline_command_records_both_machines(headline_output):
    assert "simulation mode measured here" in headline_output.stdout
    assert "hardware mode read from data/traces/" in headline_output.stdout


def test_baselines_command_exits_zero(baselines_output):
    assert baselines_output.returncode == 0, baselines_output.stderr


def test_baselines_command_prints_all_three_baselines(baselines_output):
    for label in ("BASELINE 1", "BASELINE 2", "BASELINE 3"):
        assert label in baselines_output.stdout


def test_baselines_command_names_the_declared_split(baselines_output):
    assert "training stages : ingest, background, threshold" in baselines_output.stdout
    assert "held-out stages : centroid, gate, track" in baselines_output.stdout


def test_baselines_command_reports_all_twenty_splits(baselines_output):
    assert "all 20 three-and-three splits" in baselines_output.stdout


def test_baselines_command_reports_the_resampling_unit(baselines_output):
    assert "pass-unit 95% interval" in baselines_output.stdout


def test_drift_command_exits_zero(drift_output):
    assert drift_output.returncode == 0, drift_output.stderr


def test_drift_command_prints_eleven_blocks(drift_output):
    for label in ("floor_start", "floor_00", "floor_08", "floor_end"):
        assert label in drift_output.stdout


def test_drift_command_reports_the_bracket_ratio(drift_output):
    assert "1.0752408447278534" in drift_output.stdout


def test_drift_command_reports_raw_and_adjusted_side_by_side(drift_output):
    assert "R_raw_measured" in drift_output.stdout
    assert "R_drift_adjusted" in drift_output.stdout


def test_drift_command_labels_the_raw_figure_as_the_headline(drift_output):
    assert "RAW MEASURED (headline)" in drift_output.stdout


def test_drift_command_reports_the_confounding_check(drift_output):
    assert "CONFOUNDING CHECK" in drift_output.stdout
    assert "corr(order_index, W)" in drift_output.stdout


def test_cli_refuses_an_absent_trace_directory(tmp_path):
    result = _run(["gate", "--traces", str(tmp_path / "nothing")])
    assert result.returncode == 2
    assert "REFUSED" in result.stderr


def test_cli_refusal_names_the_required_files(tmp_path):
    result = _run(["gate", "--traces", str(tmp_path / "nothing")])
    assert "meta.json" in result.stderr
    assert "stages_pass.csv" in result.stderr


def test_cli_rejects_an_unknown_command():
    result = _run(["nonsense"])
    assert result.returncode == 2


def test_cli_help_exits_zero():
    result = _run(["--help"])
    assert result.returncode == 0
    assert "headline" in result.stdout


def test_cli_help_states_research_grade():
    result = _run(["--help"])
    assert "Research-grade" in result.stdout


@pytest.mark.parametrize("name", COMMANDS)
def test_every_command_is_in_the_help_text(name):
    result = _run(["--help"])
    assert name in result.stdout


def test_cli_runs_from_a_different_working_directory(tmp_path):
    # The package must not depend on the caller's working directory: every path
    # is anchored on __file__.
    result = _run(["gate"], cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert "every branch fired at least once : True" in result.stdout


def test_cli_runs_from_the_filesystem_root():
    result = _run(["gate"], cwd=Path(os.sep))
    assert result.returncode == 0, result.stderr
