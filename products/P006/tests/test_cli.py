"""CLI tests: table output, JSON output, error handling."""

import json

import pytest

from linkbudgetx.cli import main

CONFIG = """\
tx_power_dbm: 20.0
wavelength_nm: 1550.0
beam_divergence_rad: 1.0e-3
range_km: 10.0
rx_aperture_diameter_m: 0.1
rx_sensitivity_dbm: -40.0
tx_optics_efficiency: 0.8
rx_optics_efficiency: 0.8
pointing_error_rad: 0.25e-3
atmos_attenuation_db_per_km: 0.5
beam_profile: gaussian
uncertainties:
  tx_power_dbm: 0.5
  atmos_attenuation_db_per_km: 0.05
"""


@pytest.fixture()
def config_file(tmp_path):
    p = tmp_path / "link.yaml"
    p.write_text(CONFIG)
    return p


def test_table_output(config_file, capsys):
    assert main(["--config", str(config_file)]) == 0
    out = capsys.readouterr().out
    assert "Link margin" in out
    assert "1-sigma" in out


def test_json_output(config_file, capsys):
    assert main(["--config", str(config_file), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    # Known answer from tests/test_core.py hand calc: margin = 13.9003 dB.
    assert payload["budget"]["margin_db"] == pytest.approx(13.9003, abs=5e-4)
    assert payload["uncertainty"]["sigma_margin_db"] == pytest.approx(0.7071, abs=5e-3)


def test_missing_config_returns_error(tmp_path, capsys):
    assert main(["--config", str(tmp_path / "nope.yaml")]) == 2
    assert "error:" in capsys.readouterr().err


def test_invalid_physics_returns_error(tmp_path, capsys):
    p = tmp_path / "bad.yaml"
    p.write_text(CONFIG.replace("range_km: 10.0", "range_km: -3.0"))
    assert main(["--config", str(p)]) == 2
    assert "range_km" in capsys.readouterr().err


def test_unknown_key_returns_error(tmp_path, capsys):
    p = tmp_path / "bad.yaml"
    p.write_text(CONFIG + "bogus_key: 1\n")
    assert main(["--config", str(p)]) == 2
    assert "bogus_key" in capsys.readouterr().err
