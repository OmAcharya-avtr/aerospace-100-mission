"""The loader contract: what it accepts, and every way it refuses."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from powerenvelope.provenance import Origin
from powerenvelope.schema import SchemaError
from powerenvelope.trace import (
    CYCLE_S,
    FLOOR_CELL,
    PERMISSIVE_CELL,
    PLANNED_CELLS,
    REFERENCE_REPEATS,
    TraceDataAbsentError,
    campaign_is_complete,
    load_campaign,
    load_decisions,
    load_monitor,
    load_power,
)

from fixtures import synthetic_campaign


def test_loads_every_planned_cell(campaign):
    assert set(campaign.cells) == set(PLANNED_CELLS)
    assert len(campaign) == 20


def test_origin_is_carried_and_is_synthetic(campaign):
    assert campaign.provenance.origin is Origin.SYNTHETIC
    assert not campaign.provenance.is_measured
    assert "synthetic" in campaign.provenance.banner()


def test_directory_label_is_relative(campaign):
    assert "/" not in campaign.provenance.directory


def test_floor_cell_has_no_decisions(campaign):
    floor = campaign.require(FLOOR_CELL)
    assert not floor.has_pipeline
    assert floor.decisions is None
    with pytest.raises(TraceDataAbsentError, match="background floor"):
        floor.require_decisions()


def test_pipeline_cells_exclude_the_floor(campaign):
    labels = {cell.label for cell in campaign.pipeline_cells()}
    assert FLOOR_CELL not in labels
    assert len(labels) == 19


def test_grid_cells_are_the_fifteen(campaign):
    grid = campaign.grid_cells()
    assert len(grid) == 15
    assert PERMISSIVE_CELL not in {cell.label for cell in grid}
    assert all("rep" not in cell.label for cell in grid)


def test_reference_repeats_are_all_present(campaign):
    for label in REFERENCE_REPEATS:
        assert campaign.require(label).duty == pytest.approx(0.50, rel=0.0, abs=0.0)


def test_cycle_s_matches_the_declared_constant(campaign):
    assert campaign.cycle_s == pytest.approx(CYCLE_S, rel=0.0, abs=0.0)


def test_power_column_equals_voltage_times_current(reference_cell):
    power = reference_cell.power
    recomputed = power.vdd_in_mv * power.vdd_in_ma / 1000.0
    assert recomputed == pytest.approx(power.vdd_in_mw, rel=1e-12, abs=1e-6)


def test_decision_accounting_is_counted_not_asserted(reference_cell):
    decisions = reference_cell.require_decisions()
    counts = decisions.reason_counts()
    assert sum(counts.values()) == decisions.inferences
    assert counts["accept"] == decisions.accepted_count
    assert decisions.acceptance_rate == decisions.accepted_count / decisions.inferences


def test_latency_is_positive(reference_cell):
    assert float(reference_cell.require_decisions().latency_s.min()) > 0.0


def test_clk_mean_ignores_unreadable_cores(campaign):
    monitor = campaign.require(FLOOR_CELL).monitor
    assert float(monitor.clk_mean_khz.min()) == pytest.approx(1728000.0, rel=0.0, abs=0.0)


def test_nominal_column_substitutes_five_volts(reference_cell):
    power = reference_cell.power
    assert power.nominal_mw == pytest.approx(
        5000.0 * power.vdd_in_ma / 1000.0, rel=1e-12, abs=0.0
    )


def test_quantisation_floor_is_forty_milliwatts_at_five_volts(reference_cell):
    # 5088 mV * 8 mA / 1000 = 40.704 mW. Hand calculation.
    floor = reference_cell.power.quantisation_floor_mw
    assert float(floor[0]) == pytest.approx(40.704, rel=1e-12, abs=1e-9)


def test_absent_directory_refuses_by_name(tmp_path):
    with pytest.raises(TraceDataAbsentError, match="does not exist"):
        load_campaign(tmp_path / "nothing-here")


def test_absent_manifest_refuses_and_lists_the_csvs(tmp_path):
    directory = synthetic_campaign.write_campaign_dir(tmp_path / "c")
    (directory / "manifest.json").unlink()
    with pytest.raises(TraceDataAbsentError, match="manifest.json is absent"):
        load_campaign(directory)


def test_required_cell_absent_refuses_by_name(tmp_path):
    directory = synthetic_campaign.write_campaign_dir(tmp_path / "c")
    with pytest.raises(TraceDataAbsentError, match="w64_d999"):
        load_campaign(directory, require=("w64_d999",))


def test_manifest_without_a_run_record_refuses(tmp_path):
    directory = synthetic_campaign.write_campaign_dir(tmp_path / "c")
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["runs"] = [run for run in manifest["runs"] if run["cell"] != "w48_d050"]
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(TraceDataAbsentError, match="no run record for cell"):
        load_campaign(directory)


def test_manifest_that_is_not_json_refuses(tmp_path):
    directory = synthetic_campaign.write_campaign_dir(tmp_path / "c")
    (directory / "manifest.json").write_text("{not json")
    with pytest.raises(SchemaError, match="not valid JSON"):
        load_campaign(directory)


def test_manifest_with_a_different_cycle_is_refused(tmp_path):
    directory = synthetic_campaign.write_campaign_dir(tmp_path / "c")
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["cycle_s"] = 1.0
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(SchemaError, match="different cycle period"):
        load_campaign(directory)


def test_completion_marker_is_detected_both_ways(tmp_path):
    complete = synthetic_campaign.write_campaign_dir(tmp_path / "yes")
    partial = synthetic_campaign.write_campaign_dir(tmp_path / "no", complete=False)
    assert campaign_is_complete(complete)
    assert not campaign_is_complete(partial)
    assert load_campaign(complete).complete
    assert not load_campaign(partial).complete


def _rewrite_header(path: Path, header: str) -> None:
    lines = path.read_text().splitlines()
    path.write_text(header + "\n" + "\n".join(lines[1:]) + "\n")


def test_power_header_mismatch_refuses(tmp_path):
    directory = synthetic_campaign.write_campaign_dir(tmp_path / "c")
    target = directory / "power_w64_d050.csv"
    _rewrite_header(
        target,
        "t_mono,vdd_in_mv,vdd_in_ma,cpu_gpu_cv_mv,cpu_gpu_cv_ma,soc_mv,soc_ma,watts",
    )
    with pytest.raises(SchemaError, match="power CSV header"):
        load_power(target)


def test_power_header_reorder_refuses(tmp_path):
    directory = synthetic_campaign.write_campaign_dir(tmp_path / "c")
    target = directory / "power_w64_d050.csv"
    _rewrite_header(
        target,
        "vdd_in_mv,t_mono,vdd_in_ma,cpu_gpu_cv_mv,cpu_gpu_cv_ma,soc_mv,soc_ma,vdd_in_mw",
    )
    with pytest.raises(SchemaError, match="wrong order"):
        load_power(target)


def test_monitor_header_mismatch_refuses(tmp_path):
    directory = synthetic_campaign.write_campaign_dir(tmp_path / "c")
    target = directory / "monitor_w64_d050.csv"
    _rewrite_header(
        target,
        "t_mono,cpu_busy,tj_c,load1,clk0_khz,clk1_khz,clk2_khz,clk3_khz,clk4_khz,clk5_khz",
    )
    with pytest.raises(SchemaError, match="monitor CSV header"):
        load_monitor(target)


def test_decision_header_mismatch_refuses(tmp_path):
    directory = synthetic_campaign.write_campaign_dir(tmp_path / "c")
    target = directory / "decisions_w64_d050.csv"
    _rewrite_header(
        target,
        "t_start,t_end,cycle,frame_idx,score,n_candidates,area,touches_border,ok,reject_reason",
    )
    with pytest.raises(SchemaError, match="decisions CSV header"):
        load_decisions(target)


def test_empty_file_refuses(tmp_path):
    target = tmp_path / "power_x.csv"
    target.write_text("")
    with pytest.raises(SchemaError, match="no header row"):
        load_power(target)


def test_header_only_file_refuses(tmp_path):
    directory = synthetic_campaign.write_campaign_dir(tmp_path / "c")
    target = directory / "power_w64_d050.csv"
    target.write_text(target.read_text().splitlines()[0] + "\n")
    with pytest.raises(SchemaError, match="no data rows"):
        load_power(target)


def test_ragged_row_refuses(tmp_path):
    directory = synthetic_campaign.write_campaign_dir(tmp_path / "c")
    target = directory / "power_w64_d050.csv"
    lines = target.read_text().splitlines()
    lines[3] = lines[3] + ",99"
    target.write_text("\n".join(lines) + "\n")
    with pytest.raises(SchemaError, match="fields, header has"):
        load_power(target)


def test_non_numeric_field_refuses(tmp_path):
    directory = synthetic_campaign.write_campaign_dir(tmp_path / "c")
    target = directory / "power_w64_d050.csv"
    lines = target.read_text().splitlines()
    fields = lines[3].split(",")
    fields[2] = "nope"
    lines[3] = ",".join(fields)
    target.write_text("\n".join(lines) + "\n")
    with pytest.raises(SchemaError, match="non-numeric"):
        load_power(target)


def test_decreasing_timestamp_refuses(tmp_path):
    directory = synthetic_campaign.write_campaign_dir(tmp_path / "c")
    target = directory / "power_w64_d050.csv"
    lines = target.read_text().splitlines()
    fields = lines[5].split(",")
    fields[0] = "1.0"
    lines[5] = ",".join(fields)
    target.write_text("\n".join(lines) + "\n")
    with pytest.raises(SchemaError, match="decreasing t_mono"):
        load_power(target)


def test_power_column_that_was_not_v_times_i_refuses(tmp_path):
    directory = synthetic_campaign.write_campaign_dir(tmp_path / "c")
    target = directory / "power_w64_d050.csv"
    lines = target.read_text().splitlines()
    fields = lines[4].split(",")
    fields[7] = "%.3f" % (float(fields[7]) + 1.0)
    lines[4] = ",".join(fields)
    target.write_text("\n".join(lines) + "\n")
    with pytest.raises(SchemaError, match="not produced by the committed campaign script"):
        load_power(target)


def _first_accepted_line(lines: list[str]) -> int:
    for index, line in enumerate(lines[1:], start=1):
        if line.endswith(",accept"):
            return index
    raise AssertionError("the fixture has no accepted decision to corrupt")


def test_accepted_disagreeing_with_reason_refuses(tmp_path):
    directory = synthetic_campaign.write_campaign_dir(tmp_path / "c")
    target = directory / "decisions_w64_d050.csv"
    lines = target.read_text().splitlines()
    index = _first_accepted_line(lines)
    fields = lines[index].split(",")
    fields[8] = "0"
    lines[index] = ",".join(fields)
    target.write_text("\n".join(lines) + "\n")
    with pytest.raises(SchemaError, match="disagrees with reject_reason"):
        load_decisions(target)


def test_accepted_outside_zero_one_refuses(tmp_path):
    directory = synthetic_campaign.write_campaign_dir(tmp_path / "c")
    target = directory / "decisions_w64_d050.csv"
    lines = target.read_text().splitlines()
    fields = lines[1].split(",")
    fields[8] = "2"
    lines[1] = ",".join(fields)
    target.write_text("\n".join(lines) + "\n")
    with pytest.raises(SchemaError, match="not 0 or 1"):
        load_decisions(target)


def test_negative_inference_interval_refuses(tmp_path):
    directory = synthetic_campaign.write_campaign_dir(tmp_path / "c")
    target = directory / "decisions_w64_d050.csv"
    lines = target.read_text().splitlines()
    fields = lines[1].split(",")
    fields[1] = "%.6f" % (float(fields[0]) - 1.0)
    lines[1] = ",".join(fields)
    target.write_text("\n".join(lines) + "\n")
    with pytest.raises(SchemaError, match="t_end < t_start"):
        load_decisions(target)


def test_unknown_reject_reason_in_a_file_refuses(tmp_path):
    directory = synthetic_campaign.write_campaign_dir(tmp_path / "c")
    target = directory / "decisions_w64_d050.csv"
    lines = target.read_text().splitlines()
    fields = lines[1].split(",")
    fields[9] = "exploded"
    lines[1] = ",".join(fields)
    target.write_text("\n".join(lines) + "\n")
    with pytest.raises(SchemaError, match="exploded"):
        load_decisions(target)


def test_require_names_the_cells_that_are_present(campaign):
    with pytest.raises(TraceDataAbsentError, match="cells present"):
        campaign.require("not_a_cell")


# ------------------------------------------------- the header-only floor file


def _header_only(path: Path) -> None:
    path.write_text(path.read_text().splitlines()[0] + "\n")


def test_header_only_decisions_file_for_the_floor_cell_loads(tmp_path):
    # The campaign opens and closes a decisions file for the background floor
    # cell too, so decisions_floor.csv exists with a header and no rows. That is
    # not an error: the cell ran no inferences by construction.
    directory = synthetic_campaign.write_campaign_dir(tmp_path / "c")
    floor_decisions = directory / "decisions_floor.csv"
    floor_decisions.write_text(
        "t_start,t_end,cycle,frame_idx,score,n_candidates,area,touches_border,"
        "accepted,reject_reason\n"
    )
    campaign = load_campaign(directory)
    assert campaign.require(FLOOR_CELL).decisions is None
    assert len(campaign) == 20


def test_load_decisions_allow_empty_returns_an_empty_trace(tmp_path):
    directory = synthetic_campaign.write_campaign_dir(tmp_path / "c")
    target = directory / "decisions_w64_d050.csv"
    _header_only(target)
    trace = load_decisions(target, allow_empty=True)
    assert len(trace) == 0
    assert trace.inferences == 0
    assert trace.accepted_count == 0
    assert trace.reason_counts() == {}
    with pytest.raises(ValueError, match="no inferences is undefined"):
        _ = trace.acceptance_rate


def test_header_only_decisions_file_for_a_pipeline_cell_still_refuses(tmp_path):
    # A pipeline cell whose decisions file is empty means the run produced
    # nothing, which is an error and is not excused by the floor cell's case.
    directory = synthetic_campaign.write_campaign_dir(tmp_path / "c")
    _header_only(directory / "decisions_w64_d050.csv")
    with pytest.raises(SchemaError, match="no data rows"):
        load_campaign(directory)


def test_a_floor_decisions_file_with_rows_is_a_contradiction(tmp_path):
    directory = synthetic_campaign.write_campaign_dir(tmp_path / "c")
    pipeline_rows = (directory / "decisions_w64_d050.csv").read_text()
    (directory / "decisions_floor.csv").write_text(pipeline_rows)
    with pytest.raises(SchemaError, match="manifest and the trace disagree"):
        load_campaign(directory)


def test_a_floor_decisions_file_with_a_bad_header_is_still_refused(tmp_path):
    directory = synthetic_campaign.write_campaign_dir(tmp_path / "c")
    (directory / "decisions_floor.csv").write_text("a,b,c\n")
    with pytest.raises(SchemaError, match="decisions CSV header"):
        load_campaign(directory)
