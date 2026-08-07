"""Tests for the ``python -m estimkit`` command-line interface."""

from __future__ import annotations

import json

import numpy as np
import pytest

from estimkit.cli import main


def test_steady_state_random_walk_matches_hand_solution(capsys):
    assert main(["--json", "steady-state", "--model", "random-walk", "--q", "1", "--r", "1"]) == 0
    payload = json.loads(capsys.readouterr().out)
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    assert payload["P_prior"][0][0] == pytest.approx(phi, abs=1e-12)
    assert payload["K"][0][0] == pytest.approx(1.0 / phi, abs=1e-12)


def test_steady_state_constant_velocity_table(capsys):
    assert main(["steady-state", "--model", "constant-velocity", "--q", "0.01", "--r", "4"]) == 0
    out = capsys.readouterr().out
    assert "P_prior" in out
    assert "K" in out


def test_track_reports_smoother_below_filter(capsys):
    assert main(["--json", "track", "--steps", "150", "--seed", "5"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["rms_position_smoother_m"] < payload["rms_position_filter_m"]
    assert payload["rms_velocity_smoother_mps"] < payload["rms_velocity_filter_mps"]


def test_track_is_deterministic_for_a_fixed_seed(capsys):
    main(["--json", "track", "--steps", "50", "--seed", "77"])
    first = capsys.readouterr().out
    main(["--json", "track", "--steps", "50", "--seed", "77"])
    assert capsys.readouterr().out == first


def test_invalid_measurement_noise_exits_with_code_2(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["steady-state", "--model", "random-walk", "--q", "1", "--r", "-1"])
    assert exc.value.code == 2
    assert "r must be a positive finite" in capsys.readouterr().err


def test_missing_subcommand_exits_nonzero():
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code != 0
