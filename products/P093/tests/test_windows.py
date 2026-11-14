"""The predictor's dataset: windows, descriptors, the imputation flag, splits."""

from __future__ import annotations

import numpy as np
import pytest

from powerenvelope.windows import (
    FEATURE_NAMES,
    MIN_ACTIVE_S,
    WINDOW_S,
    extract_windows,
    grouped_splits,
    pipeline_free_monitor_mask,
)


def test_window_count_is_two_per_pipeline_cell(campaign):
    # Ten cycles is 20 s per run and the window is 8 s, so two whole windows
    # per pipeline cell: 19 cells, 38 windows.
    windows = extract_windows(campaign)
    assert len(windows) == 38
    assert windows.features.shape == (38, len(FEATURE_NAMES))


def test_floor_cell_contributes_nothing_and_says_so(campaign):
    windows = extract_windows(campaign)
    assert "floor" not in set(windows.cells)
    reasons = [reason for cell, reason in windows.dropped if cell == "floor"]
    assert reasons and "no pipeline" in reasons[0]


def test_target_is_a_positive_power_in_milliwatts(campaign):
    windows = extract_windows(campaign)
    assert float(windows.targets.min()) > 1000.0
    assert float(windows.targets.max()) < 10000.0


def test_descriptor_columns_are_the_licensed_names_in_order(campaign):
    assert extract_windows(campaign).feature_names == FEATURE_NAMES


def test_configuration_columns_carry_the_declared_values(campaign):
    windows = extract_windows(campaign)
    assert set(np.unique(windows.column("duty"))) == {0.10, 0.25, 0.50, 0.75, 1.00}
    assert set(np.unique(windows.column("frame_dim"))) == {48.0, 64.0, 96.0}
    assert set(np.unique(windows.column("cycle_s"))) == {2.0}
    assert set(np.unique(windows.column("cores_available"))) == {6.0}


def test_environment_columns_are_finite_and_in_range(campaign):
    windows = extract_windows(campaign)
    for name, low, high in (
        ("env_tj_c", 20.0, 120.0),
        ("env_load1", 0.0, 50.0),
        ("env_clk_mean_khz", 1.0, 1e7),
        ("env_foreign_busy", 0.0, 1.0),
    ):
        column = windows.column(name)
        assert np.all(np.isfinite(column))
        assert low <= float(column.min()) and float(column.max()) <= high


def test_foreign_occupancy_is_imputed_only_at_high_duty(campaign):
    windows = extract_windows(campaign)
    flag = windows.column("env_foreign_imputed")
    duty = windows.column("duty")
    assert windows.foreign_imputed == int(flag.sum())
    assert windows.foreign_imputed > 0
    assert set(np.unique(duty[flag == 1.0])) <= {0.75, 1.00}
    assert set(np.unique(duty[flag == 0.0])) <= {0.10, 0.25, 0.50}


def test_imputed_value_is_the_floor_cells_own_occupancy(campaign):
    windows = extract_windows(campaign)
    flag = windows.column("env_foreign_imputed")
    imputed = windows.column("env_foreign_busy")[flag == 1.0]
    assert np.allclose(imputed, windows.floor_foreign_busy, rtol=0.0, atol=1e-12)


def test_pipeline_free_mask_is_empty_at_full_duty(campaign):
    assert not np.any(pipeline_free_monitor_mask(campaign.require("w64_d100")))


def test_pipeline_free_mask_is_non_empty_at_low_duty(campaign):
    assert np.any(pipeline_free_monitor_mask(campaign.require("w64_d010")))


def test_pipeline_free_mask_is_all_true_for_the_floor_cell(campaign):
    assert np.all(pipeline_free_monitor_mask(campaign.require("floor")))


def test_window_without_removes_the_named_columns(campaign):
    windows = extract_windows(campaign)
    matrix, names = windows.without(("env_load1", "env_foreign_busy"))
    assert matrix.shape[1] == len(FEATURE_NAMES) - 2
    assert "env_load1" not in names and "env_foreign_busy" not in names


def test_window_without_refuses_to_drop_everything(campaign):
    windows = extract_windows(campaign)
    with pytest.raises(ValueError, match="cannot drop every descriptor"):
        windows.without(FEATURE_NAMES)


def test_column_rejects_an_unknown_name(campaign):
    with pytest.raises(KeyError, match="is not a descriptor"):
        extract_windows(campaign).column("vdd_in_mw")


def test_extract_rejects_a_non_positive_window(campaign):
    with pytest.raises(ValueError, match="window_s must be positive"):
        extract_windows(campaign, 0.0)


def test_extract_refuses_when_no_window_survives(campaign):
    with pytest.raises(ValueError, match="no window survived"):
        extract_windows(campaign, 1e6)


def test_grouped_splits_by_duty_are_disjoint_and_cover_everything(campaign):
    windows = extract_windows(campaign)
    splits = grouped_splits(windows, "duty")
    assert len(splits) == 5
    seen: set[int] = set()
    for _label, train, test in splits:
        assert not set(train.tolist()) & set(test.tolist())
        assert len(train) + len(test) == len(windows)
        seen |= set(test.tolist())
    assert seen == set(range(len(windows)))


def test_grouped_splits_by_dimension_hold_out_a_whole_workload(campaign):
    windows = extract_windows(campaign)
    for label, train, test in grouped_splits(windows, "frame_dim"):
        held = set(np.unique(windows.frame_dims[test]).tolist())
        assert len(held) == 1
        assert not held & set(np.unique(windows.frame_dims[train]).tolist())
        assert label.startswith("frame_dim=")


def test_grouped_splits_never_share_a_cell_between_train_and_test(campaign):
    windows = extract_windows(campaign)
    for grouping in ("duty", "frame_dim"):
        for _label, train, test in grouped_splits(windows, grouping):
            assert not set(windows.cells[train]) & set(windows.cells[test])


def test_grouped_splits_rejects_an_unknown_grouping(campaign):
    with pytest.raises(ValueError, match="must be 'duty' or 'frame_dim'"):
        grouped_splits(extract_windows(campaign), "tau")


def test_grouped_splits_refuses_a_single_group(tmp_path):
    from fixtures import synthetic_campaign

    single = synthetic_campaign.campaign(
        tmp_path / "one",
        plan=(
            ("floor", 64, 0.0, 0.0, False),
            ("w64_d050", 64, 0.50, synthetic_campaign.tau_for(64), True),
        ),
    )
    with pytest.raises(ValueError, match="leave-one-out needs at least two"):
        grouped_splits(extract_windows(single), "duty")


def test_declared_window_constants():
    assert WINDOW_S == pytest.approx(8.0, rel=0.0, abs=0.0)
    assert MIN_ACTIVE_S == pytest.approx(0.2, rel=0.0, abs=0.0)
