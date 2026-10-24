"""YAML case loading: the shipped cases, the schema and its refusals."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from pointahead.cases import case_to_mapping, dump_case, load_case

SHIPPED = ("leo_crossing.yaml", "leo_trailing.yaml", "geo_relay_to_leo.yaml")


def test_every_shipped_case_loads(case_dir: Path) -> None:
    for name in SHIPPED:
        case = load_case(case_dir / name)
        assert case.source == name
        assert case.link.range_m > 1.0e5
        assert case.inputs.beam.wavelength_m > 0.0


def test_shipped_cases_round_trip_through_the_serialiser(case_dir: Path) -> None:
    for name in SHIPPED:
        case = load_case(case_dir / name)
        reloaded = yaml.safe_load(dump_case(case))
        assert reloaded == case_to_mapping(case)


def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_case(tmp_path / "absent.yaml")


def _write(tmp_path: Path, document: dict) -> Path:
    path = tmp_path / "case.yaml"
    path.write_text(yaml.safe_dump(document))
    return path


def _valid_document() -> dict:
    return {
        "label": "t",
        "trajectory_model": "linear",
        "local": {"position_km": [7000.0, 0.0, 0.0], "velocity_m_s": [0.0, 7500.0, 0.0]},
        "remote": {"position_km": [7000.0, 2000.0, 0.0], "velocity_m_s": [0.0, 7500.0, 0.0]},
        "beam": {"wavelength_nm": 1550.0, "waist_mm": 25.0},
        "budget": {
            "sigma_velocity_per_axis_m_s": 0.05,
            "ephemeris_age_s": 30.0,
            "light_time_iterations": 2,
        },
    }


def test_a_minimal_valid_document_loads(tmp_path: Path) -> None:
    case = load_case(_write(tmp_path, _valid_document()))
    assert case.link.range_m == pytest.approx(2.0e6, rel=1e-12)
    assert case.inputs.aberration_route == "momentum_boost"
    assert case.inputs.angle_method == "closed_form"
    assert case.inputs.loss_budget_db == 1.0


def test_an_unknown_top_level_key_is_refused(tmp_path: Path) -> None:
    document = _valid_document()
    document["trajectory"] = "kepler"
    with pytest.raises(ValueError, match="unknown key"):
        load_case(_write(tmp_path, document))


def test_an_unknown_budget_key_is_refused(tmp_path: Path) -> None:
    # A silently ignored sigma_velocity_m_s that should have been
    # sigma_velocity_per_axis_m_s would change the headline number invisibly.
    document = _valid_document()
    document["budget"]["sigma_velocity_m_s"] = 0.1
    with pytest.raises(ValueError, match="unknown key"):
        load_case(_write(tmp_path, document))


@pytest.mark.parametrize("section", ["local", "remote", "beam", "budget"])
def test_a_missing_section_is_named(tmp_path: Path, section: str) -> None:
    document = _valid_document()
    del document[section]
    with pytest.raises(ValueError, match=f"missing the '{section}' section"):
        load_case(_write(tmp_path, document))


def test_a_missing_state_field_is_named(tmp_path: Path) -> None:
    document = _valid_document()
    del document["local"]["velocity_m_s"]
    with pytest.raises(ValueError, match="velocity_m_s"):
        load_case(_write(tmp_path, document))


def test_a_missing_beam_field_is_named(tmp_path: Path) -> None:
    document = _valid_document()
    del document["beam"]["waist_mm"]
    with pytest.raises(ValueError, match="waist_mm"):
        load_case(_write(tmp_path, document))


def test_a_missing_budget_field_is_named(tmp_path: Path) -> None:
    document = _valid_document()
    del document["budget"]["ephemeris_age_s"]
    with pytest.raises(ValueError, match="ephemeris_age_s"):
        load_case(_write(tmp_path, document))


def test_a_non_numeric_position_is_refused(tmp_path: Path) -> None:
    document = _valid_document()
    document["local"]["position_km"] = ["a", "b", "c"]
    with pytest.raises(ValueError, match="position_km must be three numbers"):
        load_case(_write(tmp_path, document))


def test_a_non_mapping_section_is_refused(tmp_path: Path) -> None:
    document = _valid_document()
    document["beam"] = [1550.0, 25.0]
    with pytest.raises(ValueError, match="must be a mapping"):
        load_case(_write(tmp_path, document))


def test_kilometres_and_nanometres_are_converted(tmp_path: Path) -> None:
    case = load_case(_write(tmp_path, _valid_document()))
    assert case.link.local.position_m[0] == pytest.approx(7.0e6, rel=1e-15)
    assert case.inputs.beam.wavelength_m == pytest.approx(1550.0e-9, rel=1e-15)
    assert case.inputs.beam.waist_m == pytest.approx(0.025, rel=1e-15)
