"""The loader: schema enforcement, the missing sentinel, and carried origin."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from thermalbudget import schema
from thermalbudget.provenance import Origin
from thermalbudget.trace import (
    DECLARED_BOARD,
    TraceDataAbsentError,
    load_campaign,
    load_trace,
)

#: The five real rows quoted in docs/SPEC.md, verbatim, as the shape contract.
REAL_HEADER = (
    "t_wall,t_rel,iters_cum,iters_per_s,achieved_duty,loadavg1,foreign_cpu_pct,"
    "temp_cpu-thermal,temp_gpu-thermal,temp_soc0-thermal,temp_soc1-thermal,"
    "temp_soc2-thermal,temp_tj-thermal,cpu0_khz,cpu1_khz,cpu2_khz,cpu3_khz,cpu4_khz,"
    "cpu5_khz,gpu_hz"
)
REAL_ROWS = (
    "1794457189.964,1.001,0,19447.34,0.7512,0.38,20.10,45875,46000,45812,45125,44531,"
    "45875,1728000,1728000,1728000,1728000,1728000,1728000,306000000",
    "1794457191.0,2.051,32760,31200.88,1.1942,0.40,20.10,46343,46400,46312,45625,45031,"
    "46343,1728000,1728000,1728000,1728000,1728000,1728000,306000000",
    "1794457195.2,5.191,100120,25168.28,0.9620,0.52,20.10,46906,47000,46812,46125,45531,"
    "46906,1728000,1728000,1728000,1728000,1728000,1728000,306000000",
    "1794457204.6,14.601,340210,29038.24,1.1222,0.80,20.00,47781,47900,47712,47025,46431,"
    "47781,1728000,1728000,1728000,1728000,1728000,1728000,306000000",
    "1794457210.9,20.879,510880,27503.54,1.0825,0.95,19.80,48687,48800,48712,48025,47431,"
    "48687,1728000,1728000,1728000,1728000,1728000,1728000,306000000",
)


def _write(path, rows=REAL_ROWS, header=REAL_HEADER):
    path.write_text(header + "\n" + "\n".join(rows) + "\n")
    return path


def test_the_five_real_rows_load_and_convert(tmp_path):
    trace = _write(tmp_path / "burst_1.00_a.csv")
    loaded = load_trace(trace, origin=Origin.MEASURED)
    assert loaded.n_rows == 5
    assert loaded.temperature_divisor == 1000.0
    # Hand check: 45875 milli-degC is 45.875 degC.
    assert loaded.die_temperature_c[0] == pytest.approx(45.875, rel=0.0, abs=1e-12)
    assert loaded.die_temperature_c[-1] == pytest.approx(48.687, rel=0.0, abs=1e-12)
    assert loaded.throughput_iters_per_s[1] == pytest.approx(31200.88, rel=0.0, abs=1e-9)
    # All six cores at cpuinfo_max_freq in all five of these rows. The claim for
    # the WHOLE run is a separate measurement and is not made here: it is in
    # validation/validate_measured_campaign_output.txt, where all 120 per-core
    # samples of each burst run are reported at 1728000 kHz.
    assert loaded.max_cpu_khz == pytest.approx(1728000.0, rel=0.0, abs=1e-9)
    assert loaded.duty == pytest.approx(1.0, rel=0.0, abs=0.0)
    assert loaded.origin is Origin.MEASURED


def test_the_missing_sentinel_becomes_nan_and_is_counted(tmp_path):
    rows = list(REAL_ROWS)
    rows[2] = rows[2].replace(",46906,1728000", ",-1,1728000")
    trace = load_trace(_write(tmp_path / "soak_1.00_r1.csv", rows), origin=Origin.MEASURED)
    assert math.isnan(trace.die_temperature_c[2])
    assert trace.missing[schema.DIE_COLUMN] == 1
    # And it is NOT -0.001 degC, which is what dividing the sentinel would give.
    assert not np.any(trace.die_temperature_c < 0.0)
    assert trace.finite(schema.DIE_COLUMN).size == 4


def test_an_all_cores_missing_row_gives_nan_mean_clock_without_warning(tmp_path):
    rows = list(REAL_ROWS)
    rows[1] = ",".join(
        field if index < 13 or index == 19 else "-1"
        for index, field in enumerate(rows[1].split(","))
    )
    trace = load_trace(_write(tmp_path / "soak_1.00_r1.csv", rows), origin=Origin.MEASURED)
    assert math.isnan(trace.mean_cpu_khz[1])
    assert np.isfinite(trace.mean_cpu_khz[0])


def test_a_negative_value_in_a_non_sentinel_column_is_a_damaged_file(tmp_path):
    rows = [REAL_ROWS[0].replace(",19447.34,", ",-19447.34,")]
    with pytest.raises(schema.TraceSchemaError, match="iters_per_s"):
        load_trace(_write(tmp_path / "x.csv", rows), origin=Origin.MEASURED)


def test_an_empty_file_and_a_header_only_file_both_refuse(tmp_path):
    (tmp_path / "empty.csv").write_text("")
    with pytest.raises(schema.TraceSchemaError, match="no header row"):
        load_trace(tmp_path / "empty.csv", origin=Origin.MEASURED)
    (tmp_path / "head.csv").write_text(REAL_HEADER + "\n")
    with pytest.raises(schema.TraceSchemaError, match="no data rows"):
        load_trace(tmp_path / "head.csv", origin=Origin.MEASURED)


def test_a_ragged_row_is_refused_with_its_position(tmp_path):
    rows = [REAL_ROWS[0], REAL_ROWS[1].rsplit(",", 1)[0]]
    with pytest.raises(schema.TraceSchemaError, match="field count"):
        load_trace(_write(tmp_path / "x.csv", rows), origin=Origin.MEASURED)


def test_an_absent_file_raises_the_data_absent_error_naming_it(tmp_path):
    with pytest.raises(TraceDataAbsentError, match="soak_1.00_r1.csv"):
        load_trace(tmp_path / "soak_1.00_r1.csv", origin=Origin.MEASURED)


def test_origin_is_required_and_is_not_inferred_from_the_filename(tmp_path):
    path = _write(tmp_path / "SYNTHETIC_soak_1.00_r1.csv")
    with pytest.raises(TypeError):
        load_trace(path)  # type: ignore[call-arg]
    # The prefix is stripped from the LABEL but the origin is whatever the caller
    # says: a filename is not evidence.
    loaded = load_trace(path, origin=Origin.MEASURED)
    assert loaded.label == "soak_1.00_r1"
    assert loaded.origin is Origin.MEASURED


def test_duty_parsed_from_the_label_when_no_manifest(tmp_path):
    assert load_trace(_write(tmp_path / "soak_0.75.csv"), origin=Origin.MEASURED).duty == 0.75
    assert load_trace(_write(tmp_path / "cool_r1.csv"), origin=Origin.MEASURED).duty == 0.0
    assert load_trace(_write(tmp_path / "idle_pre.csv"), origin=Origin.MEASURED).duty == 0.0
    assert math.isnan(load_trace(_write(tmp_path / "zzz.csv"), origin=Origin.MEASURED).duty)


def test_window_and_tail_select_by_relative_time(tmp_path):
    trace = load_trace(_write(tmp_path / "burst_1.00_a.csv"), origin=Origin.MEASURED)
    assert trace.window(0.0, 5.2).n_rows == 3
    assert trace.tail(7.0).n_rows == 2  # t_rel 14.601 and 20.879 within 20.879-7
    assert trace.tail(1000.0).n_rows == 5


def test_an_absent_directory_raises_data_absent_not_an_empty_campaign(tmp_path):
    with pytest.raises(TraceDataAbsentError, match="does not exist"):
        load_campaign(tmp_path / "traces", origin=Origin.MEASURED)


def test_an_empty_directory_refuses_rather_than_returning_zero_runs(tmp_path):
    (tmp_path / "traces").mkdir()
    with pytest.raises(TraceDataAbsentError, match="empty input set"):
        load_campaign(tmp_path / "traces", origin=Origin.MEASURED)


def test_a_required_run_absent_is_named(tmp_path):
    directory = tmp_path / "traces"
    directory.mkdir()
    _write(directory / "burst_1.00_a.csv")
    with pytest.raises(TraceDataAbsentError, match="soak_1.00_r1"):
        load_campaign(directory, origin=Origin.MEASURED, require=["soak_1.00_r1"])


def test_the_manifest_supplies_duty_and_the_board_match_is_checked(tmp_path):
    directory = tmp_path / "traces"
    directory.mkdir()
    _write(directory / "soak_0.50.csv")
    (directory / "manifest.json").write_text(
        json.dumps({"model": DECLARED_BOARD, "runs": [{"label": "soak_0.50", "duty": 0.5}]})
    )
    campaign = load_campaign(directory, origin=Origin.MEASURED)
    assert campaign.board_matches_declared_target()
    assert campaign["soak_0.50"].duty == 0.5
    with pytest.raises(TraceDataAbsentError, match="not in"):
        campaign["soak_1.00_r1"]


def test_a_synthetic_manifest_does_not_match_the_declared_board(no_throttle_campaign):
    assert not no_throttle_campaign.board_matches_declared_target()
    assert no_throttle_campaign.origin is Origin.SYNTHETIC
    assert "not a board" in no_throttle_campaign.board_model


def test_campaign_selectors(no_throttle_campaign):
    assert len(no_throttle_campaign) == 17
    assert len(no_throttle_campaign.bursts()) == 2
    assert len(no_throttle_campaign.soaks()) == 6
    assert len(no_throttle_campaign.soaks(1.0)) == 3
    assert len(no_throttle_campaign.cooldowns()) == 8
    assert no_throttle_campaign.duty_cycles == (0.25, 0.5, 0.75, 1.0)
    assert math.isfinite(no_throttle_campaign.idle_floor_c())
