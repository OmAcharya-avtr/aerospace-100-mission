"""Bootstrap intervals, and the requirement that the resampling unit be declared."""

from __future__ import annotations

import pytest

from evidencestale.bootstrap import UNITS, bootstrap_stale_fraction


def test_unit_has_no_default_so_it_must_be_passed() -> None:
    with pytest.raises(TypeError):
        bootstrap_stale_fraction([1, 0], ["a", "b"])  # type: ignore[call-arg]


def test_unknown_unit_rejected_and_names_the_implemented_units() -> None:
    with pytest.raises(ValueError, match="unstated unit is not a reported"):
        bootstrap_stale_fraction([1, 0], ["a", "b"], unit="block")


def test_units_are_cluster_and_item() -> None:
    assert UNITS == ("cluster", "item")


def test_point_estimate_is_the_sample_mean() -> None:
    interval = bootstrap_stale_fraction(
        [1, 1, 0, 0, 0], ["a", "b", "c", "d", "e"], unit="item", resamples=200
    )
    assert interval.point == pytest.approx(0.4, rel=1e-15, abs=0.0)


def test_interval_brackets_the_point_estimate() -> None:
    interval = bootstrap_stale_fraction(
        [1, 1, 1, 0, 0, 0, 0, 1] * 4,
        [f"g{index // 2}" for index in range(32)],
        unit="cluster",
        resamples=500,
    )
    assert interval.low <= interval.point <= interval.high
    assert interval.contains(interval.point)
    assert interval.half_width > 0.0


def test_the_same_seed_gives_the_same_interval() -> None:
    arguments = ([1, 0, 1, 1, 0, 0], ["a", "a", "b", "b", "c", "c"])
    first = bootstrap_stale_fraction(*arguments, unit="cluster", resamples=300, seed=7)
    second = bootstrap_stale_fraction(*arguments, unit="cluster", resamples=300, seed=7)
    assert (first.low, first.high) == (second.low, second.high)


def test_a_different_seed_is_allowed_to_move_the_endpoints() -> None:
    arguments = ([1, 0, 1, 1, 0, 0], ["a", "a", "b", "b", "c", "c"])
    first = bootstrap_stale_fraction(*arguments, unit="cluster", resamples=300, seed=7)
    second = bootstrap_stale_fraction(*arguments, unit="cluster", resamples=300, seed=8)
    assert first.point == pytest.approx(second.point, rel=0.0, abs=0.0)
    # The endpoints are resampling-dependent; only the point estimate is not.
    assert isinstance(second.low, float)


def test_degenerate_data_gives_a_degenerate_interval() -> None:
    interval = bootstrap_stale_fraction(
        [0, 0, 0, 0], ["a", "b", "c", "d"], unit="item", resamples=100
    )
    assert interval.point == 0.0
    assert interval.low == 0.0
    assert interval.high == 0.0
    assert interval.half_width == 0.0


def test_misaligned_groups_rejected() -> None:
    with pytest.raises(ValueError, match="aligned with its label"):
        bootstrap_stale_fraction([1, 0], ["a"], unit="item")


def test_empty_input_rejected() -> None:
    with pytest.raises(ValueError, match="at least one evidence item"):
        bootstrap_stale_fraction([], [], unit="item")


def test_confidence_outside_the_open_unit_interval_rejected() -> None:
    with pytest.raises(ValueError, match=r"confidence must lie in \(0, 1\)"):
        bootstrap_stale_fraction([1, 0], ["a", "b"], unit="item", confidence=1.0)


def test_too_few_resamples_rejected() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        bootstrap_stale_fraction([1, 0], ["a", "b"], unit="item", resamples=1)


def test_cluster_unit_counts_clusters_and_item_unit_counts_items(
    layered_corpus,
) -> None:
    labels = layered_corpus.labels()
    subjects = layered_corpus.subjects()
    cluster = bootstrap_stale_fraction(labels, subjects, unit="cluster", resamples=400)
    item = bootstrap_stale_fraction(labels, subjects, unit="item", resamples=400)
    assert cluster.n_units == 240
    assert item.n_units == 480
    assert cluster.point == pytest.approx(item.point, rel=0.0, abs=0.0)


def test_cluster_interval_is_not_narrower_than_the_item_interval_on_this_corpus(
    layered_corpus,
) -> None:
    # Every artifact carries two evidence items that go stale together, so
    # resampling items treats correlated observations as independent draws and
    # under-states the variability. The direction of the inequality is a
    # property of this corpus's cluster structure, which is why it is asserted
    # here and reported in VALIDATION.md rather than claimed in general.
    labels = layered_corpus.labels()
    subjects = layered_corpus.subjects()
    cluster = bootstrap_stale_fraction(labels, subjects, unit="cluster", resamples=2000)
    item = bootstrap_stale_fraction(labels, subjects, unit="item", resamples=2000)
    assert cluster.half_width >= item.half_width
