"""CLI tests. Every one runs the module in a clean subprocess and asserts on the
return code as well as the output, so the documented exit statuses are actually
tested rather than described.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = str(Path(__file__).resolve().parents[1] / "src")


#: Every subcommand that calibrates is run at the reduced budget, so the suite
#: finishes in minutes rather than tens of minutes. The full budget is exercised
#: by validation/validate_cli.py, whose output is committed.
QUICK = ("--budget", "quick")


def cli(*args, timeout=600):
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC + os.pathsep + env.get("PYTHONPATH", "")
    env["MPLBACKEND"] = "Agg"
    # argparse wraps --help to the terminal width and silently drops the
    # description when the width it infers is very small, which it is under
    # pytest's capture. Pinning COLUMNS makes the help output deterministic;
    # without it the help assertions pass or fail depending on the harness.
    env["COLUMNS"] = "100"
    return subprocess.run(
        [sys.executable, "-m", "telemdrift", *args],
        capture_output=True, text=True, env=env, timeout=timeout,
    )


def test_help_exits_zero_and_names_the_subcommands():
    res = cli("--help")
    assert res.returncode == 0
    for word in ("detectors", "defaults", "calibrate", "arl", "tradeoff",
                 "transient", "trace"):
        assert word in res.stdout


def test_help_states_that_this_is_not_a_streaming_framework():
    res = cli("--help")
    assert res.returncode == 0
    assert "benchmark harness" in res.stdout
    assert "river" in res.stdout


def test_help_carries_the_research_grade_statement():
    res = cli("--help")
    assert "not flight-qualified" in res.stdout
    assert "not certified" in res.stdout


def test_help_explains_exit_status_two():
    res = cli("--help")
    assert "status 2" in res.stdout


def test_version_exits_zero_and_prints_the_version():
    res = cli("--version")
    assert res.returncode == 0
    assert "telemdrift 0.1.0" in res.stdout


def test_no_subcommand_exits_nonzero():
    """argparse requires a subcommand; the failure must be a non-zero status."""
    res = cli()
    assert res.returncode != 0
    assert "required" in (res.stderr + res.stdout).lower()


def test_unknown_subcommand_exits_nonzero():
    res = cli("kalman")
    assert res.returncode == 2
    assert "invalid choice" in res.stderr


def test_detectors_lists_all_six_with_their_scalars():
    res = cli("detectors")
    assert res.returncode == 0
    for key in ("cusum", "page_hinkley", "ewma", "ks", "adwin", "learned"):
        assert key in res.stdout
    assert "p*" in res.stdout
    assert "in order to be measured, not used" in res.stdout


def test_detectors_json_is_parseable_and_complete():
    res = cli("detectors", "--json")
    assert res.returncode == 0
    payload = json.loads(res.stdout)
    assert len(payload["detectors"]) == 6
    keys = {d["key"] for d in payload["detectors"]}
    assert keys == {"cusum", "page_hinkley", "ewma", "ks", "adwin", "learned"}
    for d in payload["detectors"]:
        assert d["default_threshold"] > 0.0


def test_budget_quick_announces_that_it_is_not_the_published_configuration():
    res = cli("defaults", *QUICK)
    assert res.returncode == 0
    assert "REDUCED BUDGET" in res.stdout
    assert "not the README numbers" in res.stdout


def test_budget_full_does_not_claim_a_reduced_budget():
    res = cli("detectors")
    assert res.returncode == 0
    assert "REDUCED BUDGET" not in res.stdout


def test_invalid_budget_exits_nonzero():
    res = cli("defaults", "--budget", "enormous")
    assert res.returncode == 2
    assert "invalid choice" in res.stderr


def test_trace_reports_a_delay_for_every_detector():
    res = cli("trace", "--change", "mean_step", "--magnitude", "2.0", "--post", "400", *QUICK)
    assert res.returncode == 0
    assert "pre-alarms" in res.stdout
    assert "peak ratio" in res.stdout


def test_trace_json_separates_pre_change_alarms_from_detections():
    res = cli("trace", "--change", "mean_step", "--magnitude", "2.0", "--post", "400",
              "--json", *QUICK)
    assert res.returncode == 0
    payload = json.loads(res.stdout)
    assert payload["change_index"] == 1000
    for key, row in payload["detectors"].items():
        assert "pre_change_alarms" in row, key
        assert "first_alarm_after_change" in row, key


def test_arl_on_a_transient_prints_the_negative_control_warning():
    res = cli("arl", "--change", "transient", "--magnitude", "4.0", "--replicates", "20",
              *QUICK)
    assert res.returncode in (0, 2)
    assert "negative control" in res.stdout
    assert "FALSE alarm" in res.stdout


def test_arl_exits_two_when_an_arl1_is_censored():
    """A censored ARL1 is a lower bound, and the documented exit status for a
    finding is 2. An unreachable threshold guarantees censoring: an ARL0 target
    of 10^8 cannot be bracketed, so the detectors end up far too wide to detect
    anything inside the budget."""
    res = cli("arl", "--target-arl0", "100000000", "--replicates", "12", *QUICK)
    assert res.returncode == 2, res.stdout[-2000:]
    assert "LOWER BOUND" in res.stdout


def test_calibrate_exits_two_on_a_bracketing_failure():
    res = cli("calibrate", "--target-arl0", "100000000", *QUICK)
    assert res.returncode == 2, res.stdout[-2000:]
    assert "BRACKETING FAILED" in res.stdout


def test_calibrate_json_reports_the_bracketing_flag():
    res = cli("calibrate", "--target-arl0", "300", "--json", *QUICK)
    assert res.returncode in (0, 2)
    payload = json.loads(res.stdout)
    assert set(payload["detectors"]) == {"cusum", "page_hinkley", "ewma", "ks", "adwin"}
    for row in payload["detectors"].values():
        assert "bracketing_failed" in row
        assert "achieved_arl0" in row
        assert "achieved_sem" in row


def test_tradeoff_prints_a_curve_with_standard_errors():
    res = cli("tradeoff", "--detector", "cusum", "--points", "3", "--span", "1.5",
              "--replicates", "20", *QUICK)
    assert res.returncode in (0, 2)
    assert "ARL0 SEM" in res.stdout
    assert "ARL1 SEM" in res.stdout


def test_tradeoff_json_has_one_entry_per_threshold():
    res = cli("tradeoff", "--detector", "ewma", "--points", "3", "--span", "1.4",
              "--replicates", "20", "--json", *QUICK)
    payload = json.loads(res.stdout)
    assert len(payload["points"]) == 3
    for pt in payload["points"]:
        assert pt["arl0"] > 0.0


@pytest.mark.parametrize("detector", ["cusum", "page_hinkley", "ewma", "ks", "adwin"])
def test_tradeoff_runs_for_every_analytic_detector(detector):
    res = cli("tradeoff", "--detector", detector, "--points", "2", "--span", "1.3",
              "--replicates", "10", *QUICK)
    assert res.returncode in (0, 2), res.stderr[-1000:]
    assert "threshold" in res.stdout


def test_transient_reports_wilson_intervals():
    res = cli("transient", "--replicates", "25", *QUICK)
    assert res.returncode == 0
    assert "Wilson" in res.stdout
    assert "every alarm below is a false alarm" in res.stdout.lower()


def test_defaults_reports_the_spread_between_default_thresholds():
    res = cli("defaults", *QUICK)
    assert res.returncode == 0
    assert "Spread between the widest and narrowest default" in res.stdout
    assert "compares" in res.stdout


def test_defaults_json_reports_the_spread_factor():
    """Run at the FULL budget deliberately.

    The spread between the five default thresholds is the product's opening
    claim, and at the reduced budget the windowed KS test's very long ARL0 is
    censored away (a 16000-sample stream cannot measure an 8000-sample ARL0), so
    the measured spread collapses to about 6x. The claim has to be checked
    against a budget that can see it. This subcommand does no calibration, so
    the full budget costs only a few seconds.
    """
    res = cli("defaults", "--json")
    payload = json.loads(res.stdout)
    assert payload["spread_factor"] > 20.0, payload["spread_factor"]
    assert len(payload["detectors"]) == 5


def test_invalid_change_type_exits_nonzero():
    res = cli("arl", "--change", "earthquake")
    assert res.returncode == 2
    assert "invalid choice" in res.stderr


def test_invalid_detector_choice_exits_nonzero():
    res = cli("tradeoff", "--detector", "learned")
    assert res.returncode == 2
    assert "invalid choice" in res.stderr


def test_package_imports_in_a_fresh_interpreter():
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC + os.pathsep + env.get("PYTHONPATH", "")
    res = subprocess.run(
        [sys.executable, "-c",
         "import telemdrift; print(telemdrift.__version__); "
         "print(len(telemdrift.__all__))"],
        capture_output=True, text=True, env=env, timeout=120,
    )
    assert res.returncode == 0, res.stderr
    assert res.stdout.splitlines()[0] == "0.1.0"
