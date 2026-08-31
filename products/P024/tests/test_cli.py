"""Command-line interface, including as a spawned subprocess."""

from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

import pytest

from detumblesim.cli import build_parser, main

ROOT = Path(__file__).resolve().parents[1]


def run(argv):
    buf = io.StringIO()
    code = main(argv, out=buf)
    return code, buf.getvalue()


class TestParser:
    def test_requires_a_command(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_rejects_an_unknown_command(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["nope"])


class TestCommands:
    def test_field(self):
        code, out = run(["field"])
        assert code == 0
        assert "geomagnetic north pole" in out
        assert "29733.4 nT" in out

    def test_detumble(self):
        code, out = run(["detumble", "--gain", "2e5", "--duration-s", "8000",
                         "--control-dt-s", "4"])
        assert code == 0
        assert "detumble time (simulated)" in out
        assert "first-order model" in out
        assert "dipole-limit lower bound" in out

    def test_detumble_reports_failure_plainly(self):
        code, out = run(["detumble", "--gain", "1.0", "--duration-s", "4000",
                         "--control-dt-s", "4"])
        assert code == 0
        assert "NOT REACHED" in out

    def test_sweep(self):
        code, out = run(["sweep", "--n-gains", "3", "--duration-s", "8000"])
        assert code == 0
        assert len(out.strip().splitlines()) == 4

    def test_controllability(self):
        code, out = run(["controllability", "--inclination-deg", "0", "--orbits", "3"])
        assert code == 0
        assert "weighted geometry factors" in out
        assert "anisotropy" in out

    def test_bad_input_returns_exit_code_two(self):
        code, _ = run(["detumble", "--inertia", "0.01", "0.01", "0.5",
                       "--duration-s", "1000"])
        assert code == 2


class TestSubprocess:
    def test_module_entry_point(self):
        proc = subprocess.run(
            [sys.executable, "-m", "detumblesim", "field"],
            cwd=ROOT, capture_output=True, text=True, timeout=180,
        )
        assert proc.returncode == 0, proc.stderr
        assert "dipole tilt" in proc.stdout
