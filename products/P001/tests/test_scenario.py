"""Scenario loading, configuration variants, and failure-mode tests."""

from __future__ import annotations

import json
import math

import pytest
import yaml

from beamtwin.budget import kim_attenuation_db_per_km
from beamtwin.scenario import (
    ScenarioError,
    format_report_text,
    load_scenario,
    report_to_json,
    run_twin,
    scenario_from_dict,
)

MINIMAL = {"link": {"range_km": 5.0}}


def _write(tmp_path, data, name="s.yaml"):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return p


class TestScenarioLoading:
    def test_minimal_scenario_loads(self):
        s = scenario_from_dict(MINIMAL)
        assert s.link.range_m == pytest.approx(5000.0)

    def test_defaults_applied(self):
        s = scenario_from_dict(MINIMAL)
        assert s.link.wavelength_m == pytest.approx(1550e-9)
        assert s.n_samples == 100_000
        assert s.seed == 0

    def test_full_scenario_from_file(self, tmp_path):
        data = {
            "name": "unit-test-link",
            "link": {
                "wavelength_nm": 1550,
                "tx_power_dbm": 23.0,
                "tx_efficiency": 0.75,
                "rx_efficiency": 0.7,
                "beam_waist_radius_m": 0.03,
                "rx_aperture_radius_m": 0.08,
                "range_km": 8.0,
                "pointing_bias_urad": 1.5,
                "attenuation_db_per_km": 1.2,
                "rx_sensitivity_dbm": -35.0,
            },
            "channel": {"cn2": 2.0e-15, "pointing_jitter_urad": 4.0},
            "monte_carlo": {"n_samples": 5000, "seed": 17},
        }
        s = load_scenario(_write(tmp_path, data))
        assert s.name == "unit-test-link"
        assert s.link.tx_power_dbm == pytest.approx(23.0)
        assert s.link.range_m == pytest.approx(8000.0)
        assert s.link.pointing_bias_rad == pytest.approx(1.5e-6)
        assert s.channel.pointing_jitter_rad == pytest.approx(4e-6)
        assert s.n_samples == 5000 and s.seed == 17

    def test_name_defaults_to_filename_stem(self, tmp_path):
        s = load_scenario(_write(tmp_path, MINIMAL, name="my-link.yaml"))
        assert s.name == "my-link"

    def test_unit_conversions_km_and_urad_and_nm(self):
        s = scenario_from_dict(
            {
                "link": {"range_km": 12.0, "pointing_bias_urad": 3.0, "wavelength_nm": 850},
                "channel": {"pointing_jitter_urad": 7.0},
            }
        )
        assert s.link.range_m == pytest.approx(12_000.0)
        assert s.link.pointing_bias_rad == pytest.approx(3e-6)
        assert s.link.wavelength_m == pytest.approx(850e-9)
        assert s.channel.pointing_jitter_rad == pytest.approx(7e-6)

    def test_shipped_example_scenario_loads(self):
        from pathlib import Path

        p = Path(__file__).resolve().parents[1] / "examples" / "link_10km.yaml"
        s = load_scenario(p)
        assert s.name == "terrestrial-10km"
        assert s.link.range_m == pytest.approx(10_000.0)


class TestVisibilityConfiguration:
    def test_visibility_maps_to_kim_attenuation(self):
        s = scenario_from_dict({"link": {"range_km": 5.0, "visibility_km": 7.0}})
        expected = kim_attenuation_db_per_km(7.0, 1550e-9)
        assert s.link.attenuation_db_per_km == pytest.approx(expected)

    def test_visibility_and_attenuation_conflict_rejected(self):
        with pytest.raises(ScenarioError, match="not both"):
            scenario_from_dict(
                {"link": {"range_km": 5.0, "visibility_km": 7.0, "attenuation_db_per_km": 1.0}}
            )

    def test_invalid_visibility_rejected(self):
        with pytest.raises(ScenarioError, match="visibility"):
            scenario_from_dict({"link": {"range_km": 5.0, "visibility_km": 0.0}})

    def test_visibility_wavelength_dependence(self):
        a = scenario_from_dict(
            {"link": {"range_km": 5.0, "visibility_km": 10.0, "wavelength_nm": 850}}
        )
        b = scenario_from_dict(
            {"link": {"range_km": 5.0, "visibility_km": 10.0, "wavelength_nm": 1550}}
        )
        assert b.link.attenuation_db_per_km < a.link.attenuation_db_per_km


class TestFailureModes:
    def test_missing_file(self, tmp_path):
        with pytest.raises(ScenarioError, match="not found"):
            load_scenario(tmp_path / "nope.yaml")

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.yaml"
        p.write_text("", encoding="utf-8")
        with pytest.raises(ScenarioError, match="empty"):
            load_scenario(p)

    def test_malformed_yaml(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("link: {range_km: 5.0\n  bad indent: [", encoding="utf-8")
        with pytest.raises(ScenarioError, match="invalid YAML"):
            load_scenario(p)

    def test_root_not_mapping(self):
        with pytest.raises(ScenarioError, match="mapping"):
            scenario_from_dict([1, 2, 3])

    def test_missing_link_section(self):
        with pytest.raises(ScenarioError, match="'link'"):
            scenario_from_dict({"channel": {"cn2": 1e-15}})

    def test_unknown_top_level_key(self):
        with pytest.raises(ScenarioError, match="unknown top-level"):
            scenario_from_dict({"link": {"range_km": 5.0}, "bogus": 1})

    def test_unknown_link_key(self):
        with pytest.raises(ScenarioError, match="unknown key"):
            scenario_from_dict({"link": {"range_km": 5.0, "typo_here": 1}})

    def test_unknown_channel_key(self):
        with pytest.raises(ScenarioError, match="unknown key"):
            scenario_from_dict({"link": {"range_km": 5.0}, "channel": {"cn_2": 1e-15}})

    def test_non_numeric_value(self):
        with pytest.raises(ScenarioError, match="must be a number"):
            scenario_from_dict({"link": {"range_km": "five"}})

    def test_boolean_rejected_as_number(self):
        with pytest.raises(ScenarioError, match="must be a number"):
            scenario_from_dict({"link": {"range_km": True}})

    def test_nan_rejected(self):
        with pytest.raises(ScenarioError):
            scenario_from_dict({"link": {"range_km": float("nan")}})

    def test_negative_range_rejected(self):
        with pytest.raises(ScenarioError):
            scenario_from_dict({"link": {"range_km": -1.0}})

    def test_out_of_band_wavelength_rejected(self):
        with pytest.raises(ScenarioError):
            scenario_from_dict({"link": {"range_km": 5.0, "wavelength_nm": 50_000_000}})

    def test_efficiency_above_one_rejected(self):
        with pytest.raises(ScenarioError):
            scenario_from_dict({"link": {"range_km": 5.0, "tx_efficiency": 1.4}})

    def test_negative_cn2_rejected(self):
        with pytest.raises(ScenarioError):
            scenario_from_dict({"link": {"range_km": 5.0}, "channel": {"cn2": -1e-15}})

    @pytest.mark.parametrize("bad", [0, -5, 2.5, "many", True])
    def test_invalid_n_samples_rejected(self, bad):
        with pytest.raises(ScenarioError, match="n_samples"):
            scenario_from_dict({"link": {"range_km": 5.0}, "monte_carlo": {"n_samples": bad}})

    @pytest.mark.parametrize("bad", [-1, 1.5, "seed"])
    def test_invalid_seed_rejected(self, bad):
        with pytest.raises(ScenarioError, match="seed"):
            scenario_from_dict({"link": {"range_km": 5.0}, "monte_carlo": {"seed": bad}})

    def test_empty_name_rejected(self):
        with pytest.raises(ScenarioError, match="name"):
            scenario_from_dict({"name": "   ", "link": {"range_km": 5.0}})

    def test_channel_not_mapping_rejected(self):
        with pytest.raises(ScenarioError, match="channel"):
            scenario_from_dict({"link": {"range_km": 5.0}, "channel": [1, 2]})

    def test_sensitivity_above_max_power_gives_negative_flagged_margin(self):
        # Documented failure mode: unachievable sensitivity -> negative margin,
        # flagged, and a fade probability of 1.
        s = scenario_from_dict(
            {
                "link": {"range_km": 10.0, "rx_sensitivity_dbm": 30.0},
                "monte_carlo": {"n_samples": 2000, "seed": 1},
            }
        )
        report = run_twin(s)
        assert report["budget"]["margin_db"] < 0
        assert report["budget"]["margin_negative"] is True
        assert report["monte_carlo"]["fade_probability"] == pytest.approx(1.0)
        assert "NEGATIVE MARGIN" in format_report_text(report)


class TestRunTwin:
    def test_report_structure(self):
        s = scenario_from_dict(
            {"link": {"range_km": 5.0}, "monte_carlo": {"n_samples": 2000, "seed": 4}}
        )
        r = run_twin(s)
        for key in ("name", "budget", "channel", "monte_carlo", "analytic_baseline", "surrogate"):
            assert key in r
        assert r["surrogate"] is None  # no surrogate supplied

    def test_report_is_json_serialisable(self):
        s = scenario_from_dict(
            {"link": {"range_km": 5.0}, "monte_carlo": {"n_samples": 1000, "seed": 4}}
        )
        parsed = json.loads(report_to_json(run_twin(s)))
        assert parsed["name"] == "scenario"

    def test_report_text_contains_key_sections(self):
        s = scenario_from_dict(
            {"link": {"range_km": 5.0}, "monte_carlo": {"n_samples": 1000, "seed": 4}}
        )
        text = format_report_text(run_twin(s))
        for token in ("LINK BUDGET", "CHANNEL", "MONTE CARLO", "not certified"):
            assert token in text

    def test_weak_regime_warning_in_text(self):
        s = scenario_from_dict(
            {
                "link": {"range_km": 15.0},
                "channel": {"cn2": 5e-14},
                "monte_carlo": {"n_samples": 1000, "seed": 1},
            }
        )
        report = run_twin(s)
        assert report["channel"]["weak_regime_valid"] is False
        assert "INVALID" in format_report_text(report)

    def test_reproducible_same_seed(self):
        data = {"link": {"range_km": 8.0}, "monte_carlo": {"n_samples": 5000, "seed": 42}}
        a = run_twin(scenario_from_dict(data))
        b = run_twin(scenario_from_dict(data))
        assert report_to_json(a) == report_to_json(b)

    def test_different_seed_changes_mc_but_not_budget(self):
        # Uses a fade-producing link (margin ~12 dB); a high-margin link would
        # give P_fade = 0 for both seeds and the comparison would be vacuous.
        link = {"range_km": 10.0, "attenuation_db_per_km": 2.5, "rx_sensitivity_dbm": -30.0}
        channel = {"cn2": 5.0e-16, "pointing_jitter_urad": 5.0}
        a = run_twin(
            scenario_from_dict(
                {"link": link, "channel": channel,
                 "monte_carlo": {"n_samples": 20_000, "seed": 1}}
            )
        )
        b = run_twin(
            scenario_from_dict(
                {"link": link, "channel": channel,
                 "monte_carlo": {"n_samples": 20_000, "seed": 2}}
            )
        )
        assert a["budget"] == b["budget"]
        assert a["monte_carlo"]["fade_probability"] != b["monte_carlo"]["fade_probability"]

    def test_analytic_baseline_present_and_finite(self):
        s = scenario_from_dict(
            {"link": {"range_km": 10.0}, "monte_carlo": {"n_samples": 1000, "seed": 1}}
        )
        p = run_twin(s)["analytic_baseline"]["fade_probability_scintillation_only"]
        assert 0.0 <= p <= 1.0 and math.isfinite(p)
