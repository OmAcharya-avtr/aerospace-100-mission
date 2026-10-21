"""The scoring harness: arm isolation, regret, gap statistics."""

from __future__ import annotations

import numpy as np
import pytest

from downlinkpick.benchmark import (
    benchmark_ratio_sources,
    benchmark_value_sources,
    gap_statistics,
)
from downlinkpick.dataset import generate_instance
from downlinkpick.exact import exact_select
from downlinkpick.greedy import greedy_select
from downlinkpick.valuemodel import declared_values

INSTANCES = [
    generate_instance(seed, n_products=10, n_windows=2, capacity_fraction=0.4)
    for seed in (101, 102, 103)
]


def test_oracle_is_added_and_has_zero_regret() -> None:
    results = benchmark_value_sources(
        INSTANCES,
        {"declared": lambda instance: declared_values(instance.inventory)},
        selector=exact_select,
    )
    names = [result.name for result in results]
    assert "oracle" in names
    oracle = next(result for result in results if result.name == "oracle")
    assert oracle.regret == pytest.approx(0.0, abs=1e-12)
    assert oracle.mae == pytest.approx(0.0, abs=1e-12)
    assert oracle.spearman == pytest.approx(1.0, rel=1e-9)
    assert oracle.regret_fraction == pytest.approx(0.0, abs=1e-12)


def test_regret_is_non_negative_for_every_arm() -> None:
    results = benchmark_value_sources(
        INSTANCES,
        {
            "declared": lambda instance: declared_values(instance.inventory),
            "flat": lambda instance: dict.fromkeys(instance.inventory.names, 1.0),
        },
        selector=exact_select,
    )
    for result in results:
        assert result.regret >= -1e-9
        assert result.regret_fraction >= -1e-9
        assert result.worst_regret_fraction >= result.regret_fraction - 1e-9
        assert result.n_instances == len(INSTANCES)


def test_an_exact_copy_of_the_truth_has_zero_regret() -> None:
    results = benchmark_value_sources(
        INSTANCES,
        {"copy": lambda instance: dict(instance.true_values)},
        selector=exact_select,
    )
    copy = next(result for result in results if result.name == "copy")
    assert copy.regret == pytest.approx(0.0, abs=1e-12)


def test_a_monotone_rescaling_of_the_truth_has_zero_regret() -> None:
    # A knapsack cares about the ordering, and the per-window capacity rows are
    # unaffected by a uniform value rescaling, so doubling every value selects
    # the same products and the regret is zero even though the MAE is large.
    results = benchmark_value_sources(
        INSTANCES,
        {"doubled": lambda instance: {k: 2.0 * v for k, v in instance.true_values.items()}},
        selector=exact_select,
    )
    doubled = next(result for result in results if result.name == "doubled")
    assert doubled.regret == pytest.approx(0.0, abs=1e-9)
    assert doubled.mae > 0.5
    assert doubled.spearman == pytest.approx(1.0, rel=1e-9)


def test_empty_instance_list_rejected() -> None:
    with pytest.raises(ValueError, match="at least one instance"):
        benchmark_value_sources([], {}, selector=exact_select)
    with pytest.raises(ValueError, match="at least one instance"):
        benchmark_ratio_sources([], {}, selector=exact_select)


def test_ratio_arms_report_overflow_and_delivery() -> None:
    results = benchmark_ratio_sources(
        INSTANCES,
        {
            "true": lambda instance: dict(instance.true_ratios),
            "half": lambda instance: {k: 0.5 * v for k, v in instance.true_ratios.items()},
        },
        selector=exact_select,
    )
    lookup = {result.name: result for result in results}
    assert lookup["oracle"].overflow_rate == 0.0
    assert lookup["oracle"].delivered_fraction == pytest.approx(1.0, rel=1e-12)
    assert lookup["true"].overflow_rate == 0.0
    # Halving every estimate under-reserves capacity, so passes overflow.
    assert lookup["half"].overflow_rate > 0.0
    assert lookup["half"].bias < 0.0
    assert lookup["half"].delivered_fraction < 1.0
    assert 0.0 <= lookup["oracle"].unspent_fraction <= 1.0


def test_an_over_estimated_ratio_never_overflows() -> None:
    results = benchmark_ratio_sources(
        INSTANCES,
        {"double": lambda instance: {
            k: min(2.0 * v, 1.0) for k, v in instance.true_ratios.items()
        }},
        selector=exact_select,
    )
    doubled = next(result for result in results if result.name == "double")
    assert doubled.overflow_rate == 0.0
    assert doubled.mean_overflow_bits == 0.0
    assert doubled.bias > 0.0
    assert doubled.unspent_fraction > 0.0


def test_the_ranking_is_computed_through_whichever_selector_is_passed() -> None:
    arms = {"declared": lambda instance: declared_values(instance.inventory)}
    by_exact = benchmark_value_sources(INSTANCES, arms, selector=exact_select)
    by_greedy = benchmark_value_sources(INSTANCES, arms, selector=greedy_select)
    exact_oracle = next(r for r in by_exact if r.name == "oracle").oracle_value
    greedy_oracle = next(r for r in by_greedy if r.name == "oracle").oracle_value
    # The optimum is at least as good as greedy on the same ground truth.
    assert exact_oracle >= greedy_oracle - 1e-9


def test_gap_statistics_known_answer() -> None:
    gaps = [0.0, 0.0, 0.1, 0.2, 0.3]
    seeds = [1, 2, 3, 4, 5]
    statistics = gap_statistics(gaps, seeds, quantile_levels=(0.5,))
    assert statistics.n_instances == 5
    assert statistics.n_optimal == 2
    # Hand calculation: mean of [0, 0, 0.1, 0.2, 0.3] is 0.12, median 0.1.
    assert statistics.mean == pytest.approx(0.12, rel=1e-12)
    assert statistics.median == pytest.approx(0.1, rel=1e-12)
    assert statistics.quantiles == ((0.5, pytest.approx(0.1)),)
    assert statistics.maximum == pytest.approx(0.3)
    assert statistics.worst_seed == 5
    assert sum(statistics.histogram_counts) == 5
    assert len(statistics.histogram_edges) == len(statistics.histogram_counts) + 1


def test_gap_statistics_optimal_tolerance_is_two_sided() -> None:
    # A gap of 1e-12 is agreement to within rounding and counts as optimal; a
    # gap of 1e-6 does not. This is a tolerance, not an equality test.
    statistics = gap_statistics([1e-12, 1e-6], [1, 2], optimal_tolerance=1e-9)
    assert statistics.n_optimal == 1


def test_gap_statistics_validation() -> None:
    with pytest.raises(ValueError, match="at least one gap"):
        gap_statistics([], [])
    with pytest.raises(ValueError, match="seeds against"):
        gap_statistics([0.1, 0.2], [1])


def test_gap_statistics_histogram_covers_the_sample() -> None:
    rng = np.random.default_rng(3)
    gaps = rng.uniform(0.0, 0.5, 200)
    statistics = gap_statistics(gaps, list(range(200)))
    assert sum(statistics.histogram_counts) == 200
