"""The benchmark harness, and the pinned head-to-head result.

The result pinned here is the headline of README.md and MODEL_CARD.md: the
learned gap finder is NOT separated from the greedy max-min baseline by this
experiment, and the deterministic coverage-greedy reference beats all three
decisively. A change that reverses either of those has to change this file,
which is the point of pinning them.
"""

from __future__ import annotations

import numpy as np
import pytest

from scenariocov import (
    CoverageGreedyReference,
    GreedyMaxMinStrategy,
    LearnedGapFinderStrategy,
    Parameter,
    ScenarioSpace,
    UniformRandomStrategy,
    coverage_benchmark,
    incremental_coverage_curve,
)
from scenariocov.benchmark import incremental_occupancy_curve

SEEDS = tuple(range(12))


@pytest.fixture(scope="module")
def small_benchmark(request):
    """A 12-seed benchmark on the committed fixtures, shared by the tests here."""
    from scenariocov import load_campaign, load_space

    root = request.path.parents[1]
    space = load_space(root / "fixtures" / "space.json")
    campaign = load_campaign(root / "fixtures" / "campaign_executed.json")
    pool = load_campaign(root / "fixtures" / "candidate_pool.json")
    return coverage_benchmark(
        space,
        campaign,
        pool,
        [
            UniformRandomStrategy(),
            GreedyMaxMinStrategy(),
            LearnedGapFinderStrategy(n_bootstrap=0),
            CoverageGreedyReference(t=2),
        ],
        n_start=20,
        n_steps=20,
        pool_size=200,
        seeds=SEEDS,
        t=2,
    )


class TestIncrementalCurves:
    def test_curve_length_and_monotonicity(self, tiny_space: ScenarioSpace) -> None:
        start = tiny_space.encode_many([{"mode": "safe", "speed": 0.5, "light": "sun"}])
        added = tiny_space.encode_many(
            [
                {"mode": "nominal", "speed": 1.5, "light": "dusk"},
                {"mode": "nominal", "speed": 0.5, "light": "dark"},
            ]
        )
        curve = incremental_coverage_curve(tiny_space, start, added, t=2)
        assert curve.shape == (3,)
        # Coverage is a set size over a fixed denominator: adding a scenario
        # can never lower it. The underlying counts are exact integers.
        assert all(b >= a for a, b in zip(curve, curve[1:], strict=False))

    def test_curve_matches_a_direct_coverage_computation(self, tiny_space) -> None:
        from scenariocov import tway_coverage

        records = [
            {"mode": "safe", "speed": 0.5, "light": "sun"},
            {"mode": "nominal", "speed": 1.5, "light": "dusk"},
            {"mode": "safe", "speed": 1.5, "light": "dark"},
        ]
        start = tiny_space.encode_many(records[:1])
        added = tiny_space.encode_many(records[1:])
        curve = incremental_coverage_curve(tiny_space, start, added, t=2)
        direct = tway_coverage(tiny_space, tiny_space.encode_many(records), t=2).fraction
        assert curve[-1] == pytest.approx(direct, abs=1e-12)

    def test_occupancy_curve_is_monotone_and_bounded(self, tiny_space: ScenarioSpace) -> None:
        start = tiny_space.encode_many([{"mode": "safe", "speed": 0.5, "light": "sun"}])
        added = tiny_space.encode_many([{"mode": "nominal", "speed": 1.5, "light": "dusk"}])
        curve = incremental_occupancy_curve(tiny_space, start, added)
        assert curve.shape == (2,)
        # Only one continuous parameter with 2 bins: 1/2 then 2/2.
        assert curve == pytest.approx([0.5, 1.0], abs=1e-12)

    def test_occupancy_curve_needs_a_continuous_parameter(self) -> None:
        space = ScenarioSpace([Parameter.categorical("c", ["a", "b"])])
        encoded = space.encode_many([{"c": "a"}])
        with pytest.raises(ValueError, match="no continuous parameters"):
            incremental_occupancy_curve(space, encoded, encoded)


class TestBenchmarkValidation:
    @pytest.fixture
    def minimal_args(self, tiny_space: ScenarioSpace):
        records = [
            {"mode": m, "speed": s, "light": c}
            for m in ("safe", "nominal")
            for s in (0.5, 1.5)
            for c in ("sun", "dusk", "dark")
        ]
        return tiny_space, records, records

    def test_rejects_a_start_of_one(self, minimal_args) -> None:
        space, executed, pool = minimal_args
        with pytest.raises(ValueError, match="n_start must be at least 2"):
            coverage_benchmark(space, executed, pool, [UniformRandomStrategy()], n_start=1)

    def test_rejects_zero_steps(self, minimal_args) -> None:
        space, executed, pool = minimal_args
        with pytest.raises(ValueError, match="n_steps must be at least 1"):
            coverage_benchmark(
                space, executed, pool, [UniformRandomStrategy()], n_start=2, n_steps=0
            )

    def test_rejects_a_pool_smaller_than_the_step_count(self, minimal_args) -> None:
        space, executed, pool = minimal_args
        with pytest.raises(ValueError, match="pool_size must be at least"):
            coverage_benchmark(
                space, executed, pool, [UniformRandomStrategy()],
                n_start=2, n_steps=5, pool_size=3,
            )

    def test_rejects_a_campaign_too_small_for_the_start(self, minimal_args) -> None:
        space, executed, pool = minimal_args
        with pytest.raises(ValueError, match="executed campaign has"):
            coverage_benchmark(
                space, executed, pool, [UniformRandomStrategy()],
                n_start=100, n_steps=2, pool_size=4,
            )

    def test_rejects_a_pool_too_small(self, minimal_args) -> None:
        space, executed, pool = minimal_args
        with pytest.raises(ValueError, match="candidate pool has"):
            coverage_benchmark(
                space, executed, pool, [UniformRandomStrategy()],
                n_start=2, n_steps=2, pool_size=1000,
            )

    def test_rejects_an_empty_strategy_list(self, minimal_args) -> None:
        space, executed, pool = minimal_args
        with pytest.raises(ValueError, match="at least one strategy"):
            coverage_benchmark(space, executed, pool, [], n_start=2, n_steps=2, pool_size=4)

    def test_rejects_an_unknown_metric(self, minimal_args) -> None:
        space, executed, pool = minimal_args
        with pytest.raises(ValueError, match="metric must be"):
            coverage_benchmark(
                space, executed, pool, [UniformRandomStrategy()],
                n_start=2, n_steps=2, pool_size=4, metric="auc",
            )


class TestStrategyCurves:
    def test_shapes_and_summary_keys(self, small_benchmark) -> None:
        curves = small_benchmark.by_strategy["greedy_maxmin"]
        assert curves.curves.shape == (len(SEEDS), 21)
        assert set(curves.summary()) == {
            "final_mean", "final_sd", "final_min", "final_max",
            "gain_mean", "gain_sd", "seconds",
        }

    def test_spread_is_reported_and_non_negative(self, small_benchmark) -> None:
        for curves in small_benchmark.by_strategy.values():
            assert np.all(curves.sd >= 0.0)
            assert np.all(curves.maximum >= curves.mean - 1e-12)
            assert np.all(curves.minimum <= curves.mean + 1e-12)

    def test_single_seed_reports_zero_spread(self, request) -> None:
        from scenariocov import load_campaign, load_space

        root = request.path.parents[1]
        space = load_space(root / "fixtures" / "space.json")
        campaign = load_campaign(root / "fixtures" / "campaign_executed.json")
        pool = load_campaign(root / "fixtures" / "candidate_pool.json")
        result = coverage_benchmark(
            space, campaign, pool, [UniformRandomStrategy()],
            n_start=20, n_steps=5, pool_size=50, seeds=(0,),
        )
        curves = result.by_strategy["uniform_random"]
        assert np.all(curves.sd == 0.0)
        assert curves.summary()["final_sd"] == pytest.approx(0.0, abs=0.0)


class TestPinnedHeadToHead:
    """The result that decides whether the learned component is worth running."""

    def test_the_coverage_greedy_reference_beats_every_strategy_under_test(
        self, small_benchmark
    ) -> None:
        reference = small_benchmark.by_strategy["coverage_greedy_reference"].final.mean()
        for name in ("uniform_random", "greedy_maxmin", "learned_gapfinder"):
            assert reference > small_benchmark.by_strategy[name].final.mean() + 0.01

    def test_the_reference_wins_on_every_single_seed(self, small_benchmark) -> None:
        diff = small_benchmark.paired_difference("coverage_greedy_reference", "learned_gapfinder")
        assert diff["right_ahead_on"] == 0
        assert diff["left_ahead_on"] >= len(SEEDS) - diff["tied_on"]

    def test_the_learned_model_is_not_separated_from_greedy_maxmin(
        self, small_benchmark
    ) -> None:
        # The headline negative result. The paired mean difference is smaller
        # than the standard deviation of that same paired difference, so the
        # experiment does not separate the two strategies. This assertion is
        # deliberately on the magnitude of the difference relative to its own
        # spread, and NOT on which of the two has the larger mean: the sign is
        # not something 12 or 60 seeds establish.
        diff = small_benchmark.paired_difference("learned_gapfinder", "greedy_maxmin")
        assert abs(diff["mean"]) < diff["sd"]

    def test_the_learned_model_loses_on_a_substantial_minority_of_seeds(
        self, small_benchmark
    ) -> None:
        diff = small_benchmark.paired_difference("learned_gapfinder", "greedy_maxmin")
        assert diff["right_ahead_on"] >= 2
        assert diff["left_ahead_on"] + diff["tied_on"] + diff["right_ahead_on"] == len(SEEDS)

    def test_coverage_only_ever_increases_along_every_curve(self, small_benchmark) -> None:
        for curves in small_benchmark.by_strategy.values():
            diffs = np.diff(curves.curves, axis=1)
            assert np.all(diffs >= -1e-12)

    def test_ranking_is_ordered_by_mean_gain(self, small_benchmark) -> None:
        ranking = small_benchmark.ranking()
        gains = [row[1] for row in ranking]
        assert gains == sorted(gains, reverse=True)
        assert ranking[0][0] == "coverage_greedy_reference"

    def test_learned_strategy_costs_far_more_wall_clock_than_max_min(
        self, small_benchmark
    ) -> None:
        learned = small_benchmark.by_strategy["learned_gapfinder"].seconds
        maxmin = small_benchmark.by_strategy["greedy_maxmin"].seconds
        # Measured at roughly 40x in the build environment. The assertion is a
        # loose inequality because wall clock is machine dependent.
        assert learned > 3.0 * maxmin
