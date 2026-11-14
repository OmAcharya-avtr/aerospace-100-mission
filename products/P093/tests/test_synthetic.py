"""The synthetic generator: deterministic, schema-correct, and labelled synthetic."""

from __future__ import annotations

import json

import numpy as np
import pytest

from powerenvelope.schema import AREA_MAX_DECLARED, AREA_MIN_DECLARED, REJECT_REASONS
from powerenvelope.synthetic import (
    DEFAULT_PLAN,
    SYNTHETIC_TAU_PERMISSIVE,
    SYNTHETIC_TAU_STRICT,
    SYNTHETIC_V_MV,
    gate_outcome,
    synthetic_score,
    write_campaign,
)
from powerenvelope.trace import COMPLETION_MARKER, load_campaign


def test_regeneration_at_a_fixed_seed_is_byte_identical(tmp_path):
    first = write_campaign(tmp_path / "a", cycles=4)
    second = write_campaign(tmp_path / "b", cycles=4)
    for path in sorted(first.iterdir()):
        assert path.read_bytes() == (second / path.name).read_bytes()


def test_jitter_changes_the_trace(tmp_path):
    plain = write_campaign(tmp_path / "plain", cycles=4)
    jittered = write_campaign(tmp_path / "jittered", cycles=4, current_jitter_lsb=2)
    assert (plain / "power_w64_d050.csv").read_bytes() != (
        jittered / "power_w64_d050.csv"
    ).read_bytes()


def test_every_expected_file_is_written(tmp_path):
    directory = write_campaign(tmp_path / "c", cycles=4)
    names = {path.name for path in directory.iterdir()}
    assert "manifest.json" in names
    assert "campaign.log" in names
    assert "SYNTHETIC" in names
    assert COMPLETION_MARKER in names
    for label, _dim, _duty, _tau, pipeline in DEFAULT_PLAN:
        assert f"power_{label}.csv" in names
        assert f"monitor_{label}.csv" in names
        assert (f"decisions_{label}.csv" in names) == pipeline


def test_incomplete_campaign_omits_the_marker(tmp_path):
    directory = write_campaign(tmp_path / "c", cycles=4, complete=False)
    assert not (directory / COMPLETION_MARKER).exists()


def test_manifest_declares_itself_synthetic(tmp_path):
    directory = write_campaign(tmp_path / "c", cycles=4)
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["synthetic"] is True
    assert manifest["origin"] == "synthetic"
    assert "not measured" in manifest["note"]
    assert manifest["ina3221"]["samples"] == "512"


def test_written_power_column_is_voltage_times_current(tmp_path):
    directory = write_campaign(tmp_path / "c", cycles=4)
    campaign = load_campaign(directory, directory_label="c")
    for cell in campaign.cells.values():
        power = cell.power
        assert np.all(power.vdd_in_mv == SYNTHETIC_V_MV)
        assert power.vdd_in_mw == pytest.approx(
            power.vdd_in_mv * power.vdd_in_ma / 1000.0, rel=1e-6, abs=1e-3
        )


def test_current_is_a_whole_number_of_quanta(tmp_path):
    directory = write_campaign(tmp_path / "c", cycles=4)
    campaign = load_campaign(directory, directory_label="c")
    current = campaign.require("w64_d050").power.vdd_in_ma
    assert np.all(np.mod(current, 8.0) == 0.0)


def test_rejects_cycles_below_one(tmp_path):
    with pytest.raises(ValueError, match="cycles must be at least 1"):
        write_campaign(tmp_path / "c", cycles=0)


def test_rejects_a_dimension_with_no_declared_latency(tmp_path):
    with pytest.raises(ValueError, match="no declared"):
        write_campaign(tmp_path / "c", plan=(("w7_d050", 7, 0.5, 1.0, True),), cycles=2)


def test_rejects_a_duty_the_latency_does_not_tile(tmp_path):
    with pytest.raises(ValueError, match="not an exact multiple"):
        write_campaign(tmp_path / "c", plan=(("w64_dx", 64, 0.3333, 1.0, True),), cycles=2)


def test_score_grid_is_uniform_over_the_reachable_levels():
    # ``(idx * 7) % 100`` is a bijection on the residues, so the grid has 100
    # levels; but every index that is a multiple of 25 is a no-candidate frame
    # and scores zero instead, and those indices are exactly the ones that map
    # to the residues 0, 25, 50 and 75. So 96 of the 100 levels are reachable
    # with a positive score, and the four that are not are 20.0, 32.5, 45.0 and
    # 57.5. This is a property of the declared generator, counted here rather
    # than assumed.
    scores = {synthetic_score(index) for index in range(1, 2500) if index % 25}
    assert len(scores) == 96
    assert min(scores) == pytest.approx(20.5, rel=0.0, abs=0.0)
    assert max(scores) == pytest.approx(69.5, rel=0.0, abs=0.0)
    assert not scores & {20.0, 32.5, 45.0, 57.5}


def test_no_candidate_frames_score_zero():
    assert synthetic_score(0) == pytest.approx(0.0, rel=0.0, abs=0.0)
    assert synthetic_score(25) == pytest.approx(0.0, rel=0.0, abs=0.0)


def test_gate_outcome_follows_the_campaign_order():
    # frame 0: no candidate wins over everything else.
    assert gate_outcome(0, 0.0)[1] == "no_candidate"
    # A border frame is rejected for the border even when its score passes.
    accepted, reason, score, _candidates, _area, border = gate_outcome(17, 0.0)
    assert accepted == 0 and reason == "border" and border == 1 and score > 0.0
    # An out-of-area frame is rejected for the area even when its score passes.
    accepted, reason, _score, _candidates, area, _border = gate_outcome(23, 0.0)
    assert accepted == 0 and reason == "area_out"
    assert area > AREA_MAX_DECLARED or area < AREA_MIN_DECLARED


def test_gate_outcome_reasons_are_all_declared():
    reasons = {gate_outcome(index, SYNTHETIC_TAU_STRICT[64])[1] for index in range(500)}
    assert reasons <= set(REJECT_REASONS)
    assert reasons == set(REJECT_REASONS)


def test_permissive_tau_accepts_strictly_more():
    strict = sum(gate_outcome(i, SYNTHETIC_TAU_STRICT[64])[0] for i in range(1000))
    permissive = sum(gate_outcome(i, SYNTHETIC_TAU_PERMISSIVE)[0] for i in range(1000))
    assert permissive > strict


def test_accepted_rows_carry_a_score_at_or_above_tau():
    tau = SYNTHETIC_TAU_STRICT[64]
    for index in range(1000):
        accepted, _reason, score, _candidates, _area, _border = gate_outcome(index, tau)
        if accepted:
            assert score >= tau
