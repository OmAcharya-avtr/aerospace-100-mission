"""Tests for the command-line interface, in-process and as a subprocess."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import pytest

import aperturediv
from aperturediv.__main__ import build_parser, main


def _child_env() -> dict[str, str]:
    """Environment for a subprocess that must import the package.

    The package directory is taken from the imported module rather than
    assumed, so the test passes both from a source tree and from an
    installed distribution.
    """
    env = dict(os.environ)
    package_root = str(pathlib.Path(aperturediv.__file__).resolve().parents[1])
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        package_root if not existing else package_root + os.pathsep + existing
    )
    return env


class TestSubprocess:
    def test_help_exits_zero_in_a_clean_subprocess(self):
        proc = subprocess.run(
            [sys.executable, "-m", "aperturediv", "--help"],
            capture_output=True,
            text=True,
            check=False,
            env=_child_env(),
        )
        assert proc.returncode == 0
        assert "aperturediv" in proc.stdout
        assert "channel" in proc.stdout

    def test_version_exits_zero(self):
        proc = subprocess.run(
            [sys.executable, "-m", "aperturediv", "--version"],
            capture_output=True,
            text=True,
            check=False,
            env=_child_env(),
        )
        assert proc.returncode == 0
        assert "0.1.0" in proc.stdout

    def test_no_subcommand_is_an_error(self):
        proc = subprocess.run(
            [sys.executable, "-m", "aperturediv"],
            capture_output=True,
            text=True,
            check=False,
            env=_child_env(),
        )
        assert proc.returncode != 0


class TestParser:
    def test_subcommands_present(self):
        parser = build_parser()
        for cmd in ("channel", "aperture", "outage", "combiner", "ber"):
            args = parser.parse_args([cmd])
            assert args.command == cmd
            assert callable(args.func)

    def test_unknown_subcommand_exits(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["nonexistent"])


class TestCommands:
    def test_channel_default(self, capsys):
        assert main(["channel"]) == 0
        out = capsys.readouterr().out
        assert "scintillation index" in out
        assert "not flight-qualified" in out

    def test_channel_with_rytov(self, capsys):
        assert main(["channel", "--rytov", "1.0"]) == 0
        out = capsys.readouterr().out
        assert "gamma-gamma alpha" in out
        assert "gamma-gamma beta" in out

    def test_channel_rejects_bad_si(self):
        with pytest.raises(ValueError):
            main(["channel", "--si", "0"])

    def test_aperture_with_explicit_scale(self, capsys):
        assert main(["aperture", "--correlation-scale", "0.05",
                     "--diameter", "0.02", "0.1"]) == 0
        out = capsys.readouterr().out
        assert "aperture averaging" in out.lower() or "A(D)" in out
        assert "0.0200" in out

    def test_aperture_derives_the_fresnel_scale(self, capsys):
        assert main(["aperture", "--wavelength", "1.55e-6", "--path-length", "2000"]) == 0
        assert "Fresnel scale" in capsys.readouterr().out

    def test_aperture_exponential_model(self, capsys):
        assert main(["aperture", "--correlation-scale", "0.05", "--model",
                     "exponential", "--diameter", "0.05"]) == 0
        assert "exponential" in capsys.readouterr().out

    def test_ber(self, capsys):
        assert main(["ber", "--bits", "200000", "--ebn0", "8", "--si", "0.3"]) == 0
        out = capsys.readouterr().out
        assert "sample BER" in out
        assert "binomial SE" in out

    def test_outage_small(self, capsys):
        assert main([
            "outage", "--samples", "40000", "--max-apertures", "2",
            "--snr-min", "0", "--snr-max", "40", "--snr-step", "1.0",
            "--window", "0.001", "0.1",
        ]) == 0
        out = capsys.readouterr().out
        assert "independent" in out
        assert "correlated" in out
        assert "diversity order" in out

    def test_outage_reports_unmeasurable_windows_instead_of_crashing(self, capsys):
        assert main([
            "outage", "--samples", "20000", "--max-apertures", "1",
            "--snr-min", "0", "--snr-max", "5", "--snr-step", "5.0",
            "--window", "1e-6", "1e-5",
        ]) == 0
        out = capsys.readouterr().out
        assert "not measurable" in out
        # A single-aperture run has no adjacent pair to report; it must say so
        # rather than index a 1x1 correlation matrix out of bounds.
        assert "n/a (single aperture)" in out

    def test_combiner_small(self, capsys):
        assert main([
            "combiner", "--samples", "12000", "--apertures", "2",
        ]) == 0
        out = capsys.readouterr().out
        assert "mrc_true" in out
        assert "egc" in out
        assert "learned" in out
        assert "Cauchy-Schwarz" in out
