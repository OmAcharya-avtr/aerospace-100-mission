"""CLI: exit codes, convention output, and the four subcommands."""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

import dopplerkit
from dopplerkit.cli import main

# The subprocess tests must find the package whether the suite is run with
# PYTHONPATH=src, from an editable install, or through the pyproject
# `pythonpath = ["src"]` setting, which does not reach a child process.
_SRC = str(pathlib.Path(dopplerkit.__file__).resolve().parent.parent)


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = _SRC + (os.pathsep + existing if existing else "")
    return env

ISS_LINE1 = "1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991"
ISS_LINE2 = "2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472202482"


def test_module_help_exits_zero():
    """`python -m dopplerkit --help` must exit 0 in a fresh interpreter."""
    proc = subprocess.run(
        [sys.executable, "-m", "dopplerkit", "--help"],
        capture_output=True, text=True, check=False, env=_child_env(),
    )
    assert proc.returncode == 0
    assert "convention" in proc.stdout


def test_convention_subcommand(capsys):
    assert main(["convention"]) == 0
    out = capsys.readouterr().out
    assert "RECEDING" in out
    assert "UP-SHIFT, which means APPROACHING" in out
    assert "INVARIANT under reversing the link direction" in out
    assert "O(beta^1)" in out


def test_analytic_table(capsys):
    assert main(["analytic", "--samples", "5"]) == 0
    out = capsys.readouterr().out
    assert "Circular overhead pass" in out
    assert "peak Doppler" in out
    assert "Doppler zero crossing" in out
    assert "convention:" in out


def test_analytic_json_has_the_conventions_and_arrays(capsys):
    assert main(["analytic", "--samples", "9", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["times_s"]) == 9
    assert len(payload["doppler_hz"]) == 9
    assert "receding" in payload["range_rate_convention"]
    assert "approaching" in payload["doppler_convention"]
    assert payload["ways"] == 1
    assert payload["peak_doppler_hz"] > 0.0


def test_analytic_two_way_doubles_the_peak(capsys):
    assert main(["analytic", "--samples", "5", "--json"]) == 0
    one = json.loads(capsys.readouterr().out)
    assert main(["analytic", "--samples", "5", "--two-way", "--json"]) == 0
    two = json.loads(capsys.readouterr().out)
    assert two["ways"] == 2
    assert two["peak_doppler_hz"] == pytest.approx(2.0 * one["peak_doppler_hz"], rel=1e-12)


def test_analytic_rejects_too_few_samples(capsys):
    assert main(["analytic", "--samples", "1"]) == 2
    assert "--samples must be >= 2" in capsys.readouterr().err


def test_relativistic_reports_magnitudes(capsys):
    assert main(["relativistic"]) == 0
    out = capsys.readouterr().out
    assert "O(beta^2) SR term at TCA" in out
    assert "gravitational (static, est.)" in out
    assert "REPORTED, not applied" in out
    assert "Dropped entirely" in out


def test_tle_subcommand(capsys):
    assert main([
        "tle", "--line1", ISS_LINE1, "--line2", ISS_LINE2,
        "--lat-deg", "51.1450", "--lon-deg", "-1.4365", "--alt-m", "100",
        "--epoch", "2019-12-09T16:35:00", "--duration-s", "300", "--step-s", "60",
    ]) == 0
    out = capsys.readouterr().out
    assert "TLE pass profile" in out
    assert "convention:" in out


def test_tle_json(capsys):
    assert main([
        "tle", "--line1", ISS_LINE1, "--line2", ISS_LINE2,
        "--lat-deg", "51.1450", "--lon-deg", "-1.4365",
        "--epoch", "2019-12-09T16:35:00", "--duration-s", "120", "--step-s", "60",
        "--json",
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["times_s"]) == 3
    assert payload["ways"] == 1
    assert "receding" in payload["range_rate_convention"]
    # Pre-compensation must be the negative of the Doppler, in the JSON too.
    for d, p in zip(payload["doppler_hz"], payload["precomp_offset_hz"]):
        assert p == pytest.approx(-d, rel=1e-15)


def test_tle_rejects_a_short_line(capsys):
    assert main([
        "tle", "--line1", "too short", "--line2", ISS_LINE2,
        "--lat-deg", "0", "--lon-deg", "0", "--epoch", "2019-12-09T16:35:00",
    ]) == 2
    assert "--line1 must be 69 characters" in capsys.readouterr().err


def test_tle_rejects_a_bad_epoch(capsys):
    assert main([
        "tle", "--line1", ISS_LINE1, "--line2", ISS_LINE2,
        "--lat-deg", "0", "--lon-deg", "0", "--epoch", "not-a-date",
    ]) == 2
    assert "--epoch must be ISO-8601" in capsys.readouterr().err


def test_tle_rejects_a_bad_step(capsys):
    assert main([
        "tle", "--line1", ISS_LINE1, "--line2", ISS_LINE2,
        "--lat-deg", "0", "--lon-deg", "0", "--epoch", "2019-12-09T16:35:00",
        "--step-s", "0",
    ]) == 2
    assert "--step-s must be > 0" in capsys.readouterr().err


def test_tle_rejects_a_duration_shorter_than_the_step(capsys):
    assert main([
        "tle", "--line1", ISS_LINE1, "--line2", ISS_LINE2,
        "--lat-deg", "0", "--lon-deg", "0", "--epoch", "2019-12-09T16:35:00",
        "--duration-s", "10", "--step-s", "30",
    ]) == 2
    assert "must exceed" in capsys.readouterr().err


def test_tle_rejects_an_out_of_range_latitude(capsys):
    assert main([
        "tle", "--line1", ISS_LINE1, "--line2", ISS_LINE2,
        "--lat-deg", "100", "--lon-deg", "0", "--epoch", "2019-12-09T16:35:00",
    ]) == 2
    assert "lat_deg must be in" in capsys.readouterr().err


def test_missing_subcommand_exits_two():
    proc = subprocess.run(
        [sys.executable, "-m", "dopplerkit"], capture_output=True, text=True,
        check=False, env=_child_env(),
    )
    assert proc.returncode == 2
