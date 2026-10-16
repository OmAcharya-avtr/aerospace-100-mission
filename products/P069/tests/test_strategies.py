"""The three strategies and the coverage-greedy reference."""

from __future__ import annotations

import numpy as np
import pytest

from scenariocov import (
    CoverageGreedyReference,
    GreedyMaxMinStrategy,
    LearnedGapFinderStrategy,
    Parameter,
    ScenarioSpace,
    SelectionContext,
    Strategy,
    UniformRandomStrategy,
    default_strategies,
)


@pytest.fixture
def line_space() -> ScenarioSpace:
    """One continuous parameter with 4 bins over [0, 1]: distances are readable by hand."""
    return ScenarioSpace([Parameter.continuous("x", [0.0, 0.25, 0.5, 0.75, 1.0])])


def context_from(space: ScenarioSpace, executed: list[dict], pool: list[dict]) -> SelectionContext:
    return SelectionContext(
        space=space,
        executed_encoded=space.encode_many(executed),
        executed_norm=space.normalise(executed),
        candidate_encoded=space.encode_many(pool),
        candidate_norm=space.normalise(pool),
    )


class TestSelectionContext:
    def test_mismatched_executed_rows(self, line_space: ScenarioSpace) -> None:
        with pytest.raises(ValueError, match="executed_encoded and executed_norm"):
            SelectionContext(
                space=line_space,
                executed_encoded=np.zeros((2, 1), dtype=np.int64),
                executed_norm=np.zeros((3, 1)),
                candidate_encoded=np.zeros((1, 1), dtype=np.int64),
                candidate_norm=np.zeros((1, 1)),
            )

    def test_mismatched_candidate_rows(self, line_space: ScenarioSpace) -> None:
        with pytest.raises(ValueError, match="candidate_encoded and candidate_norm"):
            SelectionContext(
                space=line_space,
                executed_encoded=np.zeros((1, 1), dtype=np.int64),
                executed_norm=np.zeros((1, 1)),
                candidate_encoded=np.zeros((2, 1), dtype=np.int64),
                candidate_norm=np.zeros((3, 1)),
            )

    def test_mismatched_embedding_width(self, line_space: ScenarioSpace) -> None:
        with pytest.raises(ValueError, match="different widths"):
            SelectionContext(
                space=line_space,
                executed_encoded=np.zeros((1, 1), dtype=np.int64),
                executed_norm=np.zeros((1, 2)),
                candidate_encoded=np.zeros((1, 1), dtype=np.int64),
                candidate_norm=np.zeros((1, 3)),
            )


class TestBaseClass:
    def test_base_select_is_not_implemented(self, line_space: ScenarioSpace) -> None:
        context = context_from(line_space, [{"x": 0.1}], [{"x": 0.9}])
        with pytest.raises(NotImplementedError):
            Strategy().select(context, 1, np.random.default_rng(0))

    @pytest.mark.parametrize("strategy", [UniformRandomStrategy(), GreedyMaxMinStrategy()])
    def test_batch_size_bounds(self, strategy: Strategy, line_space: ScenarioSpace) -> None:
        context = context_from(line_space, [{"x": 0.1}], [{"x": 0.9}, {"x": 0.5}])
        for n in (0, 3):
            with pytest.raises(ValueError, match="pool size"):
                strategy.select(context, n, np.random.default_rng(0))

    def test_default_strategies_bundle(self) -> None:
        names = [s.name for s in default_strategies(t=2)]
        assert names == [
            "uniform_random", "greedy_maxmin", "learned_gapfinder", "coverage_greedy_reference",
        ]


class TestUniformRandom:
    def test_returns_distinct_indices(self, line_space: ScenarioSpace) -> None:
        pool = [{"x": v} for v in (0.1, 0.3, 0.6, 0.9)]
        context = context_from(line_space, [{"x": 0.1}], pool)
        picks = UniformRandomStrategy().select(context, 3, np.random.default_rng(0))
        assert len(set(picks)) == 3

    def test_same_seed_same_picks(self, line_space: ScenarioSpace) -> None:
        pool = [{"x": v} for v in (0.1, 0.3, 0.6, 0.9)]
        context = context_from(line_space, [{"x": 0.1}], pool)
        a = UniformRandomStrategy().select(context, 2, np.random.default_rng(4))
        b = UniformRandomStrategy().select(context, 2, np.random.default_rng(4))
        assert a == b


class TestGreedyMaxMin:
    def test_hand_worked_first_two_picks(self, line_space: ScenarioSpace) -> None:
        # Executed at x = 0.0, so in the normalised embedding the executed
        # point is 0. Candidates at 0.1, 0.4, 0.95. Distances to the executed
        # set are 0.1, 0.4, 0.95, so the first pick is index 2 (0.95). The
        # minimum distance then becomes min(d to 0, d to 0.95): candidate 0 has
        # min(0.1, 0.85) = 0.1 and candidate 1 has min(0.4, 0.55) = 0.4, so the
        # second pick is index 1.
        pool = [{"x": 0.1}, {"x": 0.4}, {"x": 0.95}]
        context = context_from(line_space, [{"x": 0.0}], pool)
        picks = GreedyMaxMinStrategy().select(context, 2, np.random.default_rng(0))
        assert picks == (2, 1)

    def test_never_repeats_a_candidate(self, line_space: ScenarioSpace) -> None:
        pool = [{"x": v} for v in np.linspace(0.0, 1.0, 9)]
        context = context_from(line_space, [{"x": 0.5}], pool)
        picks = GreedyMaxMinStrategy().select(context, 9, np.random.default_rng(0))
        assert len(set(picks)) == 9

    def test_is_deterministic_regardless_of_the_generator(self, line_space) -> None:
        pool = [{"x": v} for v in (0.05, 0.45, 0.85)]
        context = context_from(line_space, [{"x": 0.0}], pool)
        a = GreedyMaxMinStrategy().select(context, 3, np.random.default_rng(1))
        b = GreedyMaxMinStrategy().select(context, 3, np.random.default_rng(99))
        assert a == b

    def test_empty_executed_set_still_works(self, line_space: ScenarioSpace) -> None:
        pool = [{"x": v} for v in (0.1, 0.5, 0.9)]
        context = SelectionContext(
            space=line_space,
            executed_encoded=np.zeros((0, 1), dtype=np.int64),
            executed_norm=np.zeros((0, 1)),
            candidate_encoded=line_space.encode_many(pool),
            candidate_norm=line_space.normalise(pool),
        )
        picks = GreedyMaxMinStrategy().select(context, 2, np.random.default_rng(0))
        assert len(set(picks)) == 2


class TestLearnedGapFinderStrategy:
    def test_picks_the_sparse_end(self, line_space: ScenarioSpace) -> None:
        executed = [{"x": v} for v in np.linspace(0.0, 0.3, 20)]
        pool = [{"x": 0.15}, {"x": 0.2}, {"x": 0.95}]
        context = context_from(line_space, executed, pool)
        picks = LearnedGapFinderStrategy(n_bootstrap=0).select(
            context, 1, np.random.default_rng(0)
        )
        assert picks == (2,)

    def test_batch_size_bounds(self, line_space: ScenarioSpace) -> None:
        executed = [{"x": v} for v in np.linspace(0.0, 0.3, 10)]
        context = context_from(line_space, executed, [{"x": 0.9}])
        with pytest.raises(ValueError, match="pool size"):
            LearnedGapFinderStrategy().select(context, 2, np.random.default_rng(0))


class TestCoverageGreedyReference:
    def test_picks_the_candidate_adding_the_most_new_tuples(self) -> None:
        # Space 2x2x2, executed only 000. Candidate 111 adds 3 new pairs
        # ((1,1) of each of the 3 parameter pairs); candidate 001 adds 2
        # (a=0&c=1 and b=0&c=1), because a=0&b=0 is already covered.
        # Hand-counted, so the greedy pick must be 111.
        space = ScenarioSpace([Parameter.categorical(n, ["0", "1"]) for n in "abc"])
        executed = [{"a": "0", "b": "0", "c": "0"}]
        pool = [{"a": "0", "b": "0", "c": "1"}, {"a": "1", "b": "1", "c": "1"}]
        context = context_from(space, executed, pool)
        picks = CoverageGreedyReference(t=2).select(context, 1, np.random.default_rng(0))
        assert picks == (1,)

    def test_t_must_be_at_least_one(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            CoverageGreedyReference(t=0)

    def test_t_beyond_the_parameter_count(self, line_space: ScenarioSpace) -> None:
        context = context_from(line_space, [{"x": 0.1}], [{"x": 0.9}])
        with pytest.raises(ValueError, match="exceeds the parameter count"):
            CoverageGreedyReference(t=3).select(context, 1, np.random.default_rng(0))

    def test_picks_are_distinct(self, rendezvous_space, rendezvous_campaign, candidate_pool):
        context = SelectionContext(
            space=rendezvous_space,
            executed_encoded=rendezvous_space.encode_many(rendezvous_campaign[:20]),
            executed_norm=rendezvous_space.normalise(rendezvous_campaign[:20]),
            candidate_encoded=rendezvous_space.encode_many(candidate_pool[:60]),
            candidate_norm=rendezvous_space.normalise(candidate_pool[:60]),
        )
        picks = CoverageGreedyReference(t=2).select(context, 10, np.random.default_rng(0))
        assert len(set(picks)) == 10
