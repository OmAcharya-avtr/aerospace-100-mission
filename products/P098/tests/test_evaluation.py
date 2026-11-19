"""Held-out splits and the scoring helpers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from evidencestale.evaluation import (
    artifact_splits,
    score_binary,
    score_probabilistic,
    summarise,
    time_splits,
)


def test_grouped_split_never_puts_one_subject_on_both_sides() -> None:
    groups = [f"g{index // 2}" for index in range(20)]
    for split in artifact_splits(groups, n_splits=5):
        train = {groups[index] for index in split.train}
        test = {groups[index] for index in split.test}
        assert not (train & test)
        assert split.kind == "artifact"


def test_grouped_split_covers_every_item_exactly_once_across_folds() -> None:
    groups = [f"g{index // 2}" for index in range(20)]
    seen: list[int] = []
    for split in artifact_splits(groups, n_splits=5):
        seen.extend(int(index) for index in split.test)
    assert sorted(seen) == list(range(20))


def test_grouped_split_rejects_too_few_groups() -> None:
    with pytest.raises(ValueError, match="at least that many distinct"):
        artifact_splits(["a", "a", "b"], n_splits=5)


def test_time_split_trains_only_on_earlier_items() -> None:
    revisions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for split in time_splits(revisions):
        cutoff = max(revisions[index] for index in split.train)
        assert all(revisions[index] > cutoff for index in split.test)
        assert split.kind == "time"


def test_time_split_rejects_concentrated_revisions() -> None:
    with pytest.raises(ValueError, match="too\n?\\s*concentrated|too concentrated"):
        time_splits([5, 5, 5, 5])


def test_score_binary_known_answer() -> None:
    split = artifact_splits([f"g{index}" for index in range(10)], n_splits=5)[0]
    decisions = [1] * 10
    truth = [1, 0] * 5
    score = score_binary("all_stale", split, decisions, truth)
    assert score.n_test == 2
    assert score.errors.false_negative == 0
    assert score.errors.true_positive + score.errors.false_positive == 2


def test_score_probabilistic_requires_test_ordered_predictions() -> None:
    split = artifact_splits([f"g{index}" for index in range(10)], n_splits=5)[0]
    with pytest.raises(ValueError, match="in test order"):
        score_probabilistic("m", split, [0.5], [1] * 10)


def test_score_probabilistic_known_answer() -> None:
    split = artifact_splits([f"g{index}" for index in range(10)], n_splits=5)[0]
    truth = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    probabilities = [0.9] * len(split.test)
    score = score_probabilistic("m", split, probabilities, truth)
    # Brier of a constant 0.9 forecast: mean of (0.9 - t)^2 over the test items.
    expected = float(
        np.mean([(0.9 - truth[index]) ** 2 for index in split.test])
    )
    assert score.brier == pytest.approx(expected, rel=1e-12, abs=0.0)


def test_summarise_counts_nan_entries_separately() -> None:
    split = artifact_splits([f"g{index}" for index in range(10)], n_splits=5)[0]
    truth = [1] * 10  # one class only, so AUC is nan
    score = score_binary("m", split, [1] * 10, truth)
    assert math.isnan(score.auc)
    summary = summarise([score])
    assert summary["folds"] == 1.0
    assert summary["nan_auc"] == 1.0
    assert math.isnan(summary["auc"])


def test_summarise_rejects_an_empty_score_list() -> None:
    with pytest.raises(ValueError, match="at least one score"):
        summarise([])
