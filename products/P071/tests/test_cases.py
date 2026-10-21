"""Loading declared conjunction cases from YAML."""

from __future__ import annotations

import numpy as np
import pytest

from conjprob.cases import load_case


def test_every_shipped_case_loads(cases_dir):
    paths = sorted(cases_dir.glob("*.yaml"))
    assert len(paths) == 3
    for path in paths:
        case = load_case(path)
        assert case.name
        assert case.radius_m > 0.0
        assert case.description
        assert np.all(np.isfinite(case.primary.covariance_m2))


def test_rtn_sigma_declaration_produces_the_expected_spectrum(cases_dir):
    case = load_case(cases_dir / "leo_crossing.yaml")
    eigenvalues = np.sort(np.linalg.eigvalsh(case.primary.covariance_m2))
    np.testing.assert_allclose(eigenvalues, [400.0, 400.0, 10000.0], rtol=1e-10)


def test_missing_file_message_quotes_the_path_as_given():
    with pytest.raises(FileNotFoundError, match="relative/path/to/case.yaml"):
        load_case("relative/path/to/case.yaml")


def test_rejects_a_non_mapping_document(tmp_path):
    path = tmp_path / "case.yaml"
    path.write_text("- 1\n- 2\n")
    with pytest.raises(ValueError, match="YAML mapping"):
        load_case(path)


def test_rejects_missing_keys(tmp_path):
    path = tmp_path / "case.yaml"
    path.write_text("name: x\n")
    with pytest.raises(ValueError, match="missing required keys"):
        load_case(path)


def test_rejects_both_or_neither_covariance_declaration(tmp_path):
    base = (
        "name: x\nhard_body_radius_m: 1.0\n"
        "primary:\n  position_m: [7000000.0, 0.0, 0.0]\n"
        "  velocity_mps: [0.0, 7500.0, 0.0]\n"
    )
    secondary = (
        "secondary:\n  position_m: [7000010.0, 0.0, 0.0]\n"
        "  velocity_mps: [0.0, 7500.0, 10.0]\n  sigma_rtn_m: [1.0, 1.0, 1.0]\n"
    )
    neither = tmp_path / "neither.yaml"
    neither.write_text(base + secondary)
    with pytest.raises(ValueError, match="exactly one of"):
        load_case(neither)
    both = tmp_path / "both.yaml"
    both.write_text(
        base
        + "  sigma_rtn_m: [1.0, 1.0, 1.0]\n"
        + "  covariance_m2: [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]\n"
        + secondary
    )
    with pytest.raises(ValueError, match="exactly one of"):
        load_case(both)


def test_rejects_a_non_positive_radius(tmp_path):
    path = tmp_path / "case.yaml"
    path.write_text(
        "name: x\nhard_body_radius_m: 0.0\n"
        "primary:\n  position_m: [7000000.0, 0.0, 0.0]\n"
        "  velocity_mps: [0.0, 7500.0, 0.0]\n  sigma_rtn_m: [1.0, 1.0, 1.0]\n"
        "secondary:\n  position_m: [7000010.0, 0.0, 0.0]\n"
        "  velocity_mps: [0.0, 7500.0, 10.0]\n  sigma_rtn_m: [1.0, 1.0, 1.0]\n"
    )
    with pytest.raises(ValueError, match="hard_body_radius_m"):
        load_case(path)


def test_rejects_a_state_that_is_not_a_mapping(tmp_path):
    path = tmp_path / "case.yaml"
    path.write_text(
        "name: x\nhard_body_radius_m: 1.0\nprimary: 3\n"
        "secondary:\n  position_m: [7000010.0, 0.0, 0.0]\n"
        "  velocity_mps: [0.0, 7500.0, 10.0]\n  sigma_rtn_m: [1.0, 1.0, 1.0]\n"
    )
    with pytest.raises(ValueError, match="must be a mapping"):
        load_case(path)


def test_rejects_a_state_missing_a_position(tmp_path):
    path = tmp_path / "case.yaml"
    path.write_text(
        "name: x\nhard_body_radius_m: 1.0\n"
        "primary:\n  velocity_mps: [0.0, 7500.0, 0.0]\n  sigma_rtn_m: [1.0, 1.0, 1.0]\n"
        "secondary:\n  position_m: [7000010.0, 0.0, 0.0]\n"
        "  velocity_mps: [0.0, 7500.0, 10.0]\n  sigma_rtn_m: [1.0, 1.0, 1.0]\n"
    )
    with pytest.raises(ValueError, match="position_m"):
        load_case(path)
