"""Windowing and descriptor extraction, including the two split definitions."""

from __future__ import annotations

import numpy as np
import pytest

from deadlinerisk.windows import (
    FEATURE_NAMES,
    extract_windows,
    grouped_splits,
    time_split,
)


def test_windows_carry_one_row_per_feature_name(campaign) -> None:
    windows = extract_windows(campaign)
    assert windows.features.shape[1] == len(FEATURE_NAMES)
    assert windows.feature_names == FEATURE_NAMES
    assert len(windows) == windows.targets.size == windows.runs.size


def test_window_targets_are_rates_in_the_unit_interval(campaign) -> None:
    windows = extract_windows(campaign)
    assert np.all(windows.targets >= 0.0)
    assert np.all(windows.targets <= 1.0)


def test_window_release_counts_sum_to_at_most_the_run_totals(campaign) -> None:
    # Windows tile a run without overlap and the last partial window is
    # dropped, so the summed release count per run cannot exceed the run's own.
    windows = extract_windows(campaign)
    for run in set(windows.runs.tolist()):
        mask = windows.runs == run
        jobs = campaign.jobs[run]
        total = int((1 + jobs.dropped_before).sum())
        assert int(windows.releases[mask].sum()) <= total


def test_window_count_matches_the_declared_window_length(campaign) -> None:
    # The fixture's gt_long is 60 s at a 10 s window, so at most 6 windows.
    windows = extract_windows(campaign, window_s=10.0)
    assert int(np.count_nonzero(windows.runs == "gt_long")) <= 6


def test_n_workers_column_is_the_sum_of_the_two_stressor_columns(campaign) -> None:
    windows = extract_windows(campaign)
    cpu = windows.features[:, FEATURE_NAMES.index("n_cpu_stress")]
    mem = windows.features[:, FEATURE_NAMES.index("n_mem_stress")]
    workers = windows.features[:, FEATURE_NAMES.index("n_workers")]
    assert np.allclose(workers, cpu + mem, rtol=0.0, atol=0.0)


def test_foreign_occupancy_is_measured_not_assumed_absent(campaign) -> None:
    # The fixture writes a non-zero foreign baseline on purpose: the measured
    # campaign shares its board, and a descriptor set that assumed zero foreign
    # load would not exercise the column the product reports.
    windows = extract_windows(campaign)
    foreign = windows.features[:, FEATURE_NAMES.index("foreign_pct")]
    assert np.all(foreign >= 0.0)
    assert float(foreign.max()) > 0.0


@pytest.mark.parametrize("window_s", [0.0, -1.0, float("nan")])
def test_window_length_validation(campaign, window_s: float) -> None:
    with pytest.raises(ValueError, match="window_s must be"):
        extract_windows(campaign, window_s=window_s)


def test_min_releases_too_high_discards_everything(campaign) -> None:
    with pytest.raises(ValueError, match="no window survived"):
        extract_windows(campaign, min_releases=10**9)


def test_grouped_splits_never_leak_a_run_into_training(campaign) -> None:
    windows = extract_windows(campaign)
    splits = grouped_splits(windows)
    assert len(splits) == len(set(windows.runs.tolist()))
    for label, train, test in splits:
        assert set(windows.runs[test].tolist()) == {label}
        assert label not in set(windows.runs[train].tolist())
        assert set(train.tolist()) & set(test.tolist()) == set()


def test_time_split_holds_out_only_later_windows(campaign) -> None:
    windows = extract_windows(campaign)
    train, test = time_split(windows, 0.3)
    assert set(train.tolist()) & set(test.tolist()) == set()
    for run in set(windows.runs.tolist()):
        train_runs = train[windows.runs[train] == run]
        test_runs = test[windows.runs[test] == run]
        if train_runs.size and test_runs.size:
            assert (
                windows.window_start_s[train_runs].max()
                < windows.window_start_s[test_runs].min()
            )


def test_time_split_covers_every_window(campaign) -> None:
    windows = extract_windows(campaign)
    train, test = time_split(windows, 0.3)
    assert train.size + test.size == len(windows)


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.2, 1.4])
def test_time_split_fraction_validation(campaign, fraction: float) -> None:
    windows = extract_windows(campaign)
    with pytest.raises(ValueError, match="holdout_fraction must be"):
        time_split(windows, fraction)
