"""Tests for the waveforge command-line interface."""

from __future__ import annotations

import pytest

from waveforge import __version__
from waveforge.cli import build_parser, main

SMALL = ["--n-pix", "32", "--n-sub", "4", "--n-act", "5", "--screen-pixels", "256"]


class TestParser:
    def test_requires_a_subcommand(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_version(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main(["--version"])
        assert exc.value.code == 0
        assert __version__ in capsys.readouterr().out

    def test_unknown_subcommand(self):
        with pytest.raises(SystemExit):
            main(["nonsense"])


class TestNollCommand:
    def test_runs(self, capsys):
        assert main(["noll", "--j-max", "10"]) == 0
        out = capsys.readouterr().out
        assert "published" in out
        assert "worst relative difference" in out

    def test_reports_close_agreement(self, capsys):
        main(["noll", "--j-max", "21"])
        line = [
            row for row in capsys.readouterr().out.splitlines() if "worst relative" in row
        ][0]
        worst = float(line.split(":")[1].strip().rstrip("%"))
        assert worst < 1.0

    def test_beyond_the_published_table(self, capsys):
        main(["noll", "--j-max", "25"])
        assert capsys.readouterr().out.count("-") > 0

    def test_scaling_argument(self, capsys):
        main(["noll", "--j-max", "3", "--d-over-r0", "5"])
        assert "D/r0 = 5" in capsys.readouterr().out


class TestScreenCommand:
    def test_runs(self, capsys):
        assert main(["screen", "--n-pix", "64", "--subharmonics", "0"]) == 0
        out = capsys.readouterr().out
        assert "D_bandlim" in out
        assert "variance" in out

    def test_subharmonics_reported(self, capsys):
        main(["screen", "--n-pix", "32", "--subharmonics", "2"])
        assert "subharmonic levels : 2" in capsys.readouterr().out

    def test_invalid_screen_size(self):
        with pytest.raises(SystemExit):
            main(["screen", "--n-pix", "2"])


class TestBudgetCommand:
    def test_runs(self, capsys):
        assert main(["budget", *SMALL]) == 0
        out = capsys.readouterr().out
        assert "total_rad2" in out
        assert "dominant term" in out

    def test_reports_noise_when_flux_is_finite(self, capsys):
        main(["budget", *SMALL, "--flux", "500", "--read-noise", "1.0"])
        out = capsys.readouterr().out
        assert "slope noise sigma" in out
        noise_line = [row for row in out.splitlines() if "noise_rad2" in row][0]
        assert float(noise_line.split(":")[1]) > 0.0

    def test_invalid_gain_is_reported(self):
        with pytest.raises(SystemExit):
            main(["budget", *SMALL, "--gain", "0"])


class TestLoopCommand:
    def test_runs(self, capsys):
        assert main(["loop", *SMALL, "--frames", "60", "--warmup", "20"]) == 0
        out = capsys.readouterr().out
        assert "residual variance" in out
        assert "diverged           : False" in out

    def test_reports_rejection(self, capsys):
        main(["loop", *SMALL, "--frames", "60", "--warmup", "20"])
        line = [row for row in capsys.readouterr().out.splitlines() if "rejection" in row][0]
        assert float(line.split(":")[1].split()[0]) > 0.0

    def test_invalid_warmup_is_reported(self):
        with pytest.raises(SystemExit):
            main(["loop", *SMALL, "--frames", "10", "--warmup", "10"])


class TestPredictCommand:
    def test_runs_and_reports_a_verdict(self, capsys):
        code = main(
            [
                "predict",
                *SMALL,
                "--frames",
                "80",
                "--warmup",
                "20",
                "--train-frames",
                "80",
                "--history",
                "3",
            ]
        )
        assert code == 0
        out = capsys.readouterr().out
        assert "classical integrator" in out
        assert "pure-delay POL" in out
        assert "learned predictor" in out
        assert "wins" in out
