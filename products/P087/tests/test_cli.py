"""Tests for the command line of ``python -m refractpath``."""

from __future__ import annotations

import pytest

from refractpath.__main__ import build_parser, main


class TestParser:
    def test_a_command_is_required(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_version_exits_zero(self) -> None:
        with pytest.raises(SystemExit) as excinfo:
            build_parser().parse_args(["--version"])
        assert excinfo.value.code == 0

    def test_every_subcommand_is_registered(self) -> None:
        parser = build_parser()
        for command in ("info", "refract", "curve", "floor", "dispersion", "references"):
            assert parser.parse_args([command]).command == command


class TestInfo:
    def test_info_exits_zero(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["info"]) == 0
        out = capsys.readouterr().out
        assert "refractpath" in out
        assert "raytrace" in out
        assert "Research grade" in out
        assert "Not flight-qualified" in out

    def test_info_lists_every_method_and_reference(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["info"])
        out = capsys.readouterr().out
        for name in (
            "plane_parallel",
            "oriani_series",
            "bennett_1982",
            "saemundsson_1986",
            "raytrace",
        ):
            assert name in out
        for key in ("refractivity_standard_air_633nm", "bennett_accuracy_claim"):
            assert key in out

    def test_info_states_what_cannot_be_determined(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["info"])
        assert "Cannot be determined" in capsys.readouterr().out


class TestRefract:
    def test_refract_prints_the_verdict_first(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["refract", "--elevation", "45"]) == 0
        out = capsys.readouterr().out
        assert out.startswith("VERDICT:")
        assert "cross-model disagreement" in out
        assert "profile envelope" in out

    def test_refract_at_low_elevation_refuses(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["refract", "--elevation", "2", "--tolerance", "5"])
        assert "VERDICT: REFUSE" in capsys.readouterr().out

    def test_refract_at_high_elevation_is_usable(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["refract", "--elevation", "80", "--tolerance", "10"])
        assert "VERDICT: USABLE" in capsys.readouterr().out

    def test_refract_accepts_the_full_state(self, capsys: pytest.CaptureFixture[str]) -> None:
        status = main(
            [
                "refract",
                "--elevation",
                "30",
                "--pressure",
                "880",
                "--temperature",
                "-5",
                "--humidity",
                "0.3",
                "--wavelength",
                "0.8",
                "--co2",
                "420",
                "--lapse",
                "-9.8",
                "--inversion-depth",
                "150",
                "--observer-height",
                "1200",
                "--refractivity",
                "edlen_1966",
            ]
        )
        assert status == 0
        assert "VERDICT:" in capsys.readouterr().out

    def test_bad_humidity_is_reported_and_exits_two(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["refract", "--humidity", "43"]) == 2
        assert "relative_humidity" in capsys.readouterr().err

    def test_bad_elevation_is_reported_and_exits_two(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["refract", "--elevation", "120"]) == 2
        assert "refractpath:" in capsys.readouterr().err


class TestCurve:
    def test_curve_prints_one_row_per_grid_point(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["curve", "--points", "7"]) == 0
        lines = capsys.readouterr().out.strip().splitlines()
        # Header, rule, then seven rows.
        assert len(lines) == 9

    def test_curve_marks_undefined_methods(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(["curve", "--low", "0", "--high", "1", "--points", "4"])
        assert "undefined" in capsys.readouterr().out

    def test_curve_names_every_method_in_the_header(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["curve", "--points", "4"])
        header = capsys.readouterr().out.splitlines()[0]
        assert "spread" in header
        assert "raytrace" in header


class TestFloor:
    def test_floor_reports_a_measured_bracket(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["floor", "--tolerance", "10"]) == 0
        out = capsys.readouterr().out
        assert "measured validity floor" in out
        assert "bracket" in out
        assert "tolerance crossings on grid" in out

    def test_floor_states_the_operational_limit(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["floor", "--tolerance", "10"])
        out = capsys.readouterr().out
        assert "suitable for operational pointing below" in out
        assert out.startswith("declared tolerance")

    def test_floor_rejects_a_zero_tolerance(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["floor", "--tolerance", "0"]) == 2
        assert "tolerance_arcsec" in capsys.readouterr().err


class TestDispersion:
    def test_dispersion_prints_every_formulation(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["dispersion", "--points", "5"]) == 0
        out = capsys.readouterr().out
        for name in (
            "iau_explanatory_supplement",
            "edlen_1966",
            "birch_downs_1994",
            "ciddor_1996",
        ):
            assert name in out
        assert "rel spread" in out

    def test_dispersion_rows_match_the_point_count(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["dispersion", "--points", "6"])
        assert len(capsys.readouterr().out.strip().splitlines()) == 8


class TestReferences:
    def test_references_exits_one_because_one_check_fails(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # The non-zero exit is the point: the Bennett claim is not reproduced
        # against this package's ray trace below about 19 degrees, and the CLI
        # says so rather than reporting six passes.
        assert main(["references"]) == 1
        out = capsys.readouterr().out
        assert "FAIL" in out
        assert "bennett_accuracy_claim" in out
        assert "failures: 1" in out

    def test_references_prints_the_note_for_the_failure(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["references"])
        assert "note:" in capsys.readouterr().out

    def test_references_reports_five_passes(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["references"])
        out = capsys.readouterr().out
        assert out.count("PASS") == 5
