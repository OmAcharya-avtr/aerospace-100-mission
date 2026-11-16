"""The campaign loader: what it requires, what it refuses, and what it carries.

The refusal paths matter as much as the happy path. A loader that quietly
produced a plausible number when the campaign was absent would make a simulated
figure indistinguishable from a measurement, which is the one failure mode a
differential timing product cannot survive.
"""

from __future__ import annotations

import json

import pytest

from hiltimingdiff import traces
from hiltimingdiff.traces import (
    CALIBRATION_BLOCKS,
    COMPLETION_TOKEN,
    EXPECTED_ROWS,
    GATE_BRANCHES,
    REQUIRED_CELLS,
    REQUIRED_FILES,
    TraceDataAbsentError,
)

STAGES = ("ingest", "background", "threshold", "centroid", "gate", "track")


def test_required_cells_are_the_nine_in_randomised_order():
    assert REQUIRED_CELLS == (
        "w48_dense",
        "w64_sparse_rep1",
        "w48_sparse",
        "w64_sparse",
        "w64_sparse_rep3",
        "w96_dense",
        "w96_sparse",
        "w64_dense",
        "w64_sparse_rep2",
    )


def test_calibration_blocks_are_eleven_not_the_ten_the_spec_says():
    # SPEC §8 says ten blocks. The campaign program emits floor_start, one block
    # before each of the nine cells, and floor_end: eleven. The discrepancy is
    # reported rather than reconciled by dropping a block.
    assert len(CALIBRATION_BLOCKS) == 11
    assert CALIBRATION_BLOCKS[0] == "floor_start"
    assert CALIBRATION_BLOCKS[-1] == "floor_end"


@pytest.mark.parametrize("index", range(9))
def test_calibration_block_labels_are_zero_padded(index):
    assert CALIBRATION_BLOCKS[index + 1] == f"floor_{index:02d}"


def test_gate_branches_are_accept_plus_four_reject_reasons():
    assert GATE_BRANCHES == ("accept", "no_candidate", "low_score", "area_out", "border")


@pytest.mark.parametrize("name", REQUIRED_FILES)
def test_required_file_is_present(campaign, name):
    assert (campaign.directory / name).is_file()


@pytest.mark.parametrize("cell", REQUIRED_CELLS)
def test_raw_array_is_present_for_every_cell(campaign, cell):
    assert (campaign.directory / "raw" / f"{cell}.npy").is_file()


@pytest.mark.parametrize(("name", "rows"), sorted(EXPECTED_ROWS.items()))
def test_pinned_row_count(campaign, name, rows):
    loaded = {
        "stages_pass.csv": campaign.stages_pass,
        "stages.csv": campaign.stages_subsampled,
        "runs.csv": campaign.runs,
        "env.csv": campaign.env,
        "calibration.csv": campaign.calibration,
    }[name]
    assert len(loaded) == rows


def test_label_is_relative_and_carries_no_absolute_path(campaign):
    assert campaign.label == "data/traces"
    assert not campaign.label.startswith("/")


def test_campaign_is_complete(campaign):
    assert traces.campaign_is_complete(campaign.directory)
    assert COMPLETION_TOKEN in campaign.log


def test_log_records_the_measured_drift_ratio(campaign):
    assert "drift_ratio 1.075241" in campaign.log


def test_sample_every_is_three(campaign):
    assert campaign.sample_every == 3


def test_stage_names_from_meta_match_the_board_source(campaign):
    assert campaign.stage_names == STAGES


def test_meta_records_the_campaign_seed(campaign):
    assert campaign.meta["campaign_seed"] == 20261115


def test_meta_records_the_committed_permutation(campaign):
    assert campaign.meta["permutation"] == [1, 6, 0, 2, 8, 5, 4, 3, 7]


def test_meta_records_an_unpinned_governor(campaign):
    assert campaign.meta["governor"] == "schedutil"


def test_meta_records_maxn_super(campaign):
    assert "MAXN_SUPER" in campaign.meta["nvpmodel"]


def test_meta_records_that_passwordless_sudo_was_unavailable(campaign):
    assert campaign.meta["sudo_passwordless"] is False


def test_meta_records_the_board_architecture(campaign):
    assert campaign.meta["machine"] == "aarch64"


def test_meta_records_ninety_thousand_timed_frames_per_cell(campaign):
    assert campaign.meta["n_timed_frames_per_run"] == 90000
    assert campaign.meta["n_frames_per_stack"] == 1500
    assert campaign.meta["n_passes_per_run"] == 60


def test_meta_drift_ratio_is_end_over_start(campaign):
    expected = (
        campaign.meta["calibration_end_median_ns"]
        / campaign.meta["calibration_start_median_ns"]
    )
    assert campaign.meta["calibration_drift_ratio"] == pytest.approx(
        expected, rel=1e-15, abs=0.0
    )


def test_randomised_order_is_a_permutation_of_the_unrandomised_one(campaign):
    assert sorted(campaign.meta["randomised_order"]) == sorted(
        campaign.meta["unrandomised_order"]
    )


def test_randomised_order_is_not_the_unrandomised_one(campaign):
    assert campaign.meta["randomised_order"] != campaign.meta["unrandomised_order"]


def test_permutation_reproduces_the_randomised_order(campaign):
    unrandomised = campaign.meta["unrandomised_order"]
    permutation = campaign.meta["permutation"]
    assert [unrandomised[index] for index in permutation] == campaign.meta[
        "randomised_order"
    ]


@pytest.mark.parametrize("cell", REQUIRED_CELLS)
def test_raw_array_is_two_dimensional_with_six_stage_columns(campaign, cell):
    array = campaign.raw(cell)
    assert array.ndim == 2
    assert array.shape[1] == 6


@pytest.mark.parametrize("cell", REQUIRED_CELLS)
def test_raw_array_rows_match_the_subsampled_frame_count(campaign, cell):
    # 90000 timed frames subsampled every third frame gives 30000 rows.
    assert campaign.raw(cell).shape[0] == 30000


@pytest.mark.parametrize("cell", REQUIRED_CELLS)
def test_raw_array_holds_no_negative_duration(campaign, cell):
    assert int(campaign.raw(cell).min()) >= 0


def test_raw_refuses_an_unknown_cell(campaign):
    with pytest.raises(TraceDataAbsentError, match="is absent"):
        campaign.raw("w1024_absurd")


@pytest.mark.parametrize(
    ("cell", "width"),
    [
        ("w48_sparse", 48),
        ("w48_dense", 48),
        ("w64_sparse", 64),
        ("w64_dense", 64),
        ("w64_sparse_rep1", 64),
        ("w64_sparse_rep2", 64),
        ("w64_sparse_rep3", 64),
        ("w96_sparse", 96),
        ("w96_dense", 96),
    ],
)
def test_cell_width(campaign, cell, width):
    assert campaign.cell_width(cell) == width


@pytest.mark.parametrize(
    ("cell", "order"),
    [
        ("w48_dense", 0),
        ("w64_sparse_rep1", 1),
        ("w48_sparse", 2),
        ("w64_sparse", 3),
        ("w64_sparse_rep3", 4),
        ("w96_dense", 5),
        ("w96_sparse", 6),
        ("w64_dense", 7),
        ("w64_sparse_rep2", 8),
    ],
)
def test_order_index_matches_the_randomised_order(campaign, cell, order):
    assert campaign.order_index(cell) == order


def test_cell_width_refuses_an_unknown_cell(campaign):
    with pytest.raises(TraceDataAbsentError, match="not in"):
        campaign.cell_width("nope")


def test_order_index_refuses_an_unknown_cell(campaign):
    with pytest.raises(TraceDataAbsentError, match="not in"):
        campaign.order_index("nope")


def test_frequency_columns_are_the_six_cores(campaign):
    assert campaign.frequency_columns == tuple(f"freq_cpu{index}" for index in range(6))


@pytest.mark.parametrize("stage", STAGES)
def test_every_stage_appears_in_the_pass_table(campaign, stage):
    assert (campaign.stages_pass["stage"] == stage).sum() == 540


@pytest.mark.parametrize("cell", REQUIRED_CELLS)
def test_every_cell_has_sixty_passes_per_stage(campaign, cell):
    rows = campaign.stages_pass.loc[campaign.stages_pass["cell"] == cell]
    assert len(rows) == 360
    assert sorted(rows["pass_idx"].unique()) == list(range(60))


def test_pass_summaries_cover_every_frame_of_the_pass(campaign):
    assert set(campaign.stages_pass["n_frames"].unique()) == {1500}


@pytest.mark.parametrize("branch", GATE_BRANCHES)
def test_runs_table_has_a_column_for_every_gate_branch(campaign, branch):
    assert branch in campaign.runs.columns


def test_clock_resolution_is_one_nanosecond_in_every_cell(campaign):
    assert set(campaign.runs["clock_res_ns"].unique()) == {1.0}


def test_bracket_cost_is_measured_and_positive_in_every_cell(campaign):
    assert float(campaign.runs["bracket_ns"].min()) > 0.0
    assert float(campaign.runs["bracket_ns"].max()) < 200.0


def test_subsampled_table_covers_every_cell(campaign):
    assert set(campaign.stages_subsampled["cell"].unique()) == set(REQUIRED_CELLS)


def test_subsampled_frame_indices_step_by_three(campaign):
    frames = campaign.stages_subsampled["frame_idx"].to_numpy()
    assert int(frames.min()) == 0
    assert set(frames[:5].tolist()) == {0, 3, 6, 9, 12}


def test_calibration_has_two_thousand_reps_per_block(campaign):
    counts = campaign.calibration.groupby("which").size()
    assert set(counts.unique()) == {2000}


def test_hardware_pass_table_adds_declared_ops_and_per_op_cost(campaign):
    table = traces.hardware_pass_table(campaign)
    assert "declared_ops" in table.columns
    assert "ns_per_declared_op" in table.columns
    assert len(table) == 3240


def test_hardware_pass_table_declared_ops_are_all_positive(campaign):
    table = traces.hardware_pass_table(campaign)
    assert float(table["declared_ops"].min()) > 0.0


def test_hardware_pass_table_per_op_is_median_over_ops(campaign):
    table = traces.hardware_pass_table(campaign)
    row = table.iloc[0]
    assert row["ns_per_declared_op"] == pytest.approx(
        row["median_ns"] / row["declared_ops"], rel=1e-12, abs=0.0
    )


def test_loader_refuses_a_missing_directory(tmp_path):
    with pytest.raises(TraceDataAbsentError, match="MEASURED CAMPAIGN ABSENT"):
        traces.load_campaign(tmp_path / "nothing-here")


def test_loader_refuses_an_empty_directory(tmp_path):
    with pytest.raises(TraceDataAbsentError, match="MEASURED CAMPAIGN INCOMPLETE"):
        traces.load_campaign(tmp_path)


def test_loader_names_every_missing_file(tmp_path):
    with pytest.raises(TraceDataAbsentError) as excinfo:
        traces.load_campaign(tmp_path)
    for name in REQUIRED_FILES:
        assert name in str(excinfo.value)


def test_loader_refuses_a_campaign_without_the_completion_token(campaign, tmp_path):
    import shutil

    shutil.copytree(campaign.directory, tmp_path / "copy")
    (tmp_path / "copy" / "campaign.log").write_text("[FULL] started and stopped\n")
    with pytest.raises(TraceDataAbsentError, match="DID NOT COMPLETE"):
        traces.load_campaign(tmp_path / "copy")


def test_loader_refuses_a_campaign_missing_a_cell(campaign, tmp_path):
    import shutil

    import pandas as pd

    shutil.copytree(campaign.directory, tmp_path / "copy")
    runs = pd.read_csv(tmp_path / "copy" / "runs.csv")
    runs = runs.loc[runs["cell"] != "w96_dense"]
    runs.to_csv(tmp_path / "copy" / "runs.csv", index=False)
    with pytest.raises(TraceDataAbsentError, match="MISSING CELLS"):
        traces.load_campaign(tmp_path / "copy", require_rows=False)


def test_loader_refuses_a_campaign_missing_a_raw_array(campaign, tmp_path):
    import shutil

    shutil.copytree(campaign.directory, tmp_path / "copy")
    (tmp_path / "copy" / "raw" / "w96_dense.npy").unlink()
    with pytest.raises(TraceDataAbsentError, match="MISSING RAW ARRAY"):
        traces.load_campaign(tmp_path / "copy", require_rows=False)


def test_loader_refuses_a_truncated_table(campaign, tmp_path):
    import shutil

    import pandas as pd

    shutil.copytree(campaign.directory, tmp_path / "copy")
    table = pd.read_csv(tmp_path / "copy" / "stages_pass.csv")
    table.iloc[:1000].to_csv(tmp_path / "copy" / "stages_pass.csv", index=False)
    with pytest.raises(TraceDataAbsentError, match="ROW COUNTS CHANGED"):
        traces.load_campaign(tmp_path / "copy")


def test_loader_row_count_refusal_names_both_counts(campaign, tmp_path):
    import shutil

    import pandas as pd

    shutil.copytree(campaign.directory, tmp_path / "copy")
    table = pd.read_csv(tmp_path / "copy" / "stages_pass.csv")
    table.iloc[:1000].to_csv(tmp_path / "copy" / "stages_pass.csv", index=False)
    with pytest.raises(TraceDataAbsentError) as excinfo:
        traces.load_campaign(tmp_path / "copy")
    assert "3240" in str(excinfo.value)
    assert "1000" in str(excinfo.value)


def test_environment_variable_moves_the_required_directory(tmp_path, monkeypatch):
    monkeypatch.setenv(traces.TRACES_ENVIRONMENT_VARIABLE, str(tmp_path / "elsewhere"))
    with pytest.raises(TraceDataAbsentError, match="elsewhere"):
        traces.load_campaign()


def test_traces_label_falls_back_to_the_directory_name(tmp_path):
    assert traces.traces_label(tmp_path / "somewhere") == "somewhere"


def test_meta_json_is_valid_json_on_disk(campaign):
    assert isinstance(json.loads((campaign.directory / "meta.json").read_text()), dict)
