"""Command-line interface."""

from __future__ import annotations

import io

import pytest

from constellink.cli import build_parser, main


def test_help_exits_zero():
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["--help"])
    assert exc.value.code == 0


def test_version_exits_zero():
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["--version"])
    assert exc.value.code == 0


def test_a_command_is_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_contacts_command():
    out = io.StringIO()
    assert main(["contacts", "--n-total", "8", "--n-planes", "2",
                 "--hours", "1", "--limit", "5"], out=out) == 0
    text = out.getvalue()
    assert "Walker-53:8/2/1" in text
    assert "windows:" in text
    assert "union graph:" in text


def test_contacts_command_without_stations():
    out = io.StringIO()
    assert main(["contacts", "--n-total", "8", "--n-planes", "2",
                 "--hours", "1", "--no-stations", "--limit", "3"],
                out=out) == 0
    assert "ground" in out.getvalue()


def test_capacity_command():
    out = io.StringIO()
    assert main(["capacity", "--range-km", "1500"], out=out) == 0
    text = out.getvalue()
    assert "free-space loss" in text
    assert "achievable rate" in text
    assert "Optical" in text


def test_capacity_command_rejects_bad_range():
    out = io.StringIO()
    assert main(["capacity", "--range-km", "0"], out=out) == 2


def test_route_command():
    out = io.StringIO()
    code = main(["route", "--n-total", "8", "--n-planes", "2", "--hours", "1",
                 "--source", "W00-00", "--destination", "AWARUA"], out=out)
    assert code == 0
    assert "time-expanded graph" in out.getvalue()


def test_route_command_unknown_node_returns_two():
    out = io.StringIO()
    assert main(["route", "--n-total", "8", "--n-planes", "2", "--hours", "1",
                 "--source", "NOPE", "--destination", "AWARUA"], out=out) == 2


def test_flow_command():
    out = io.StringIO()
    code = main(["flow", "--n-total", "8", "--n-planes", "2", "--hours", "1",
                 "--source", "W00-00", "--destination", "AWARUA",
                 "--unit-mbit", "6000"], out=out)
    assert code == 0
    assert "delivered:" in out.getvalue()


def test_flow_command_pulp_backend_reports_the_missing_solver():
    out = io.StringIO()
    from constellink.flow import pulp_solver_available
    code = main(["flow", "--n-total", "8", "--n-planes", "2", "--hours", "1",
                 "--source", "W00-00", "--destination", "AWARUA",
                 "--unit-mbit", "6000", "--backend", "pulp"], out=out)
    assert code == (0 if pulp_solver_available() else 2)


def test_predict_command():
    out = io.StringIO()
    assert main(["predict", "--hours", "4"], out=out) == 0
    text = out.getvalue()
    assert "climatology (base)" in text
    assert "logistic (base)" in text
    assert "best calibrated" in text


def test_benchmark_command():
    out = io.StringIO()
    assert main(["benchmark", "--n-total", "8", "--n-planes", "2",
                 "--hours", "1", "--repeats", "1"], out=out) == 0
    text = out.getvalue()
    assert "clock_resolution_s" in text
    assert "tracemalloc lower bound" in text


def test_module_entry_point_help_exits_zero():
    import subprocess
    import sys

    res = subprocess.run([sys.executable, "-m", "constellink", "--help"],
                         capture_output=True, text=True, check=False,
                         env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"})
    assert res.returncode == 0
    assert "constellink" in res.stdout
