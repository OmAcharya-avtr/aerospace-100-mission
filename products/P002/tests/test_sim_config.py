"""Configuration tests: Scenario validation and YAML round-trips."""

from __future__ import annotations

import pytest
import yaml

from trackbench.sim import DEFAULT_SCENARIO, Scenario, load_scenario


def write(tmp_path, data: dict, name: str = "s.yaml"):
    """Helper: dump ``data`` to a YAML file and return its path."""
    p = tmp_path / name
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return p


def test_default_scenario_is_valid():
    assert DEFAULT_SCENARIO.name == "default"
    assert DEFAULT_SCENARIO.dt > 0
    assert DEFAULT_SCENARIO.spike_time < DEFAULT_SCENARIO.track_duration


def test_scenario_to_dict_round_trips_through_yaml(tmp_path):
    d = DEFAULT_SCENARIO.to_dict()
    p = write(tmp_path, d)
    sc = load_scenario(p)
    assert sc.to_dict() == d


def test_load_scenario_applies_overrides(tmp_path):
    p = write(tmp_path, {"name": "custom", "bandwidth_hz": 12.0, "controller": "lqr"})
    sc = load_scenario(p)
    assert sc.name == "custom" and sc.bandwidth_hz == 12.0 and sc.controller == "lqr"
    assert sc.dt == DEFAULT_SCENARIO.dt  # unspecified keys keep defaults


def test_load_scenario_rejects_unknown_keys(tmp_path):
    p = write(tmp_path, {"nmae": "typo"})
    with pytest.raises(ValueError, match="unknown scenario keys"):
        load_scenario(p)


def test_load_scenario_missing_file():
    with pytest.raises(FileNotFoundError):
        load_scenario("/nonexistent/scenario.yaml")


def test_load_scenario_rejects_non_mapping(tmp_path):
    p = tmp_path / "list.yaml"
    p.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(TypeError):
        load_scenario(p)


def test_load_empty_scenario_gives_defaults(tmp_path):
    p = tmp_path / "empty.yaml"
    p.write_text("", encoding="utf-8")
    assert load_scenario(p).to_dict() == DEFAULT_SCENARIO.to_dict()


@pytest.mark.parametrize("kw", [
    {"pattern": "zigzag"},
    {"controller": "fuzzy"},
    {"reacq_policy": "guess"},
    {"dt": 0.0},
    {"beam_radius": -1e-5},
    {"track_threshold": 0.0},
    {"p_dwell": 0.0},
    {"spike_time": -1.0},
])
def test_scenario_field_validation(kw):
    with pytest.raises(ValueError):
        Scenario(**kw)


def test_spike_after_track_duration_is_rejected():
    with pytest.raises(ValueError, match="spike_time"):
        Scenario(spike_time=5.0, track_duration=2.0)


def test_reacq_config_derived_from_scenario():
    sc = Scenario()
    cfg = sc.reacq_config()
    assert cfg.p_detect == sc.p_dwell
    assert cfg.cone_radius == pytest.approx(sc.sigma_uncertainty * 3.0)
    assert cfg.coverage_rate > 0


def test_reacq_config_overrides_are_applied():
    sc = Scenario(reacq={"max_time": 12.0, "p_detect": 0.5})
    cfg = sc.reacq_config()
    assert cfg.max_time == 12.0 and cfg.p_detect == 0.5


def test_shipped_example_scenarios_load():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "examples"
    files = sorted(root.glob("*.yaml"))
    assert files, "no example scenarios found"
    for f in files:
        sc = load_scenario(f)
        assert sc.name
