"""t-way coverage: hand-worked answers, exhaustive cross-checks and input validation."""

from __future__ import annotations

from itertools import combinations, product

import numpy as np
import pytest

from scenariocov import (
    Parameter,
    ScenarioSpace,
    count_tway_tuples,
    tway_coverage,
)


def brute_force_coverage(space: ScenarioSpace, encoded: np.ndarray, t: int) -> tuple[int, int]:
    """Independent reference implementation: enumerate T_t and test membership.

    Deliberately written the slow, obvious way -- build every t-way value tuple
    and ask whether any scenario realises it -- so that it shares no code path
    with :func:`scenariocov.tway_coverage` beyond the space object itself.
    """
    counts = space.level_counts
    total = 0
    covered = 0
    rows = [tuple(int(v) for v in row) for row in encoded]
    for subset in combinations(range(len(counts)), t):
        for values in product(*(range(counts[i]) for i in subset)):
            total += 1
            if any(all(row[i] == v for i, v in zip(subset, values, strict=True)) for row in rows):
                covered += 1
    return covered, total


class TestCountTwayTuples:
    def test_hand_worked_counts(self) -> None:
        # Level counts (2, 2, 3). |T_1| = 2 + 2 + 3 = 7.
        # |T_2| = 2*2 + 2*3 + 2*3 = 4 + 6 + 6 = 16. |T_3| = 2*2*3 = 12.
        assert count_tway_tuples([2, 2, 3], 1) == 7
        assert count_tway_tuples([2, 2, 3], 2) == 16
        assert count_tway_tuples([2, 2, 3], 3) == 12

    def test_committed_fixture_space_counts(self) -> None:
        # Level counts (4, 4, 3, 3, 3, 3, 3): 16 + 2*5*12 + C(5,2)*9
        # = 16 + 120 + 90 = 226 pairs.
        assert count_tway_tuples([4, 4, 3, 3, 3, 3, 3], 2) == 226
        assert count_tway_tuples([4, 4, 3, 3, 3, 3, 3], 7) == 4 * 4 * 3**5

    def test_t_equal_to_k_is_the_full_cross_product(self) -> None:
        assert count_tway_tuples([3, 5, 2], 3) == 30

    @pytest.mark.parametrize(("counts", "t"), [([2, 2], 0), ([2, 2], 3), ([0, 2], 1)])
    def test_invalid_arguments_raise(self, counts: list[int], t: int) -> None:
        with pytest.raises(ValueError):
            count_tway_tuples(counts, t)


class TestTwayCoverageKnownAnswers:
    def test_two_opposite_corners_of_a_three_bit_space(self) -> None:
        # Space 2x2x2. |T_2| = 3 pairs x 4 = 12. Scenarios 000 and 111 cover
        # (0,0) and (1,1) of each pair, so 6 of 12 and the 6 uncovered tuples
        # are the mixed assignments. Hand-counted.
        space = ScenarioSpace([Parameter.categorical(n, ["0", "1"]) for n in "abc"])
        encoded = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int64)
        result = tway_coverage(space, encoded, t=2)
        assert (result.total, result.covered) == (12, 6)
        assert result.fraction == pytest.approx(0.5, abs=1e-12)
        assert result.n_uncovered == 6
        assert len(result.uncovered) == 6
        assert {u.as_text() for u in result.uncovered} == {
            "a=0 & b=1", "a=1 & b=0", "a=0 & c=1", "a=1 & c=0", "b=0 & c=1", "b=1 & c=0",
        }

    def test_one_way_coverage_counts_levels_seen(self, tiny_space: ScenarioSpace) -> None:
        # |T_1| = 7. One scenario touches one level of each of 3 parameters.
        encoded = tiny_space.encode_many([{"mode": "safe", "speed": 0.5, "light": "sun"}])
        result = tway_coverage(tiny_space, encoded, t=1)
        assert (result.total, result.covered) == (7, 3)

    def test_exhaustive_campaign_covers_everything_exactly(self, tiny_space) -> None:
        rows = np.array(list(product(*(range(c) for c in tiny_space.level_counts))), dtype=np.int64)
        for t in (1, 2, 3):
            result = tway_coverage(tiny_space, rows, t=t)
            # Integer identity, not a floating-point one.
            assert result.covered == result.total
            assert result.uncovered == ()

    def test_empty_campaign_covers_nothing(self, tiny_space: ScenarioSpace) -> None:
        result = tway_coverage(tiny_space, tiny_space.encode_many([]), t=2)
        assert result.covered == 0
        assert len(result.uncovered) == result.total == 16

    def test_covered_plus_uncovered_is_the_total(self, tiny_space: ScenarioSpace) -> None:
        encoded = tiny_space.encode_many(
            [
                {"mode": "safe", "speed": 0.5, "light": "sun"},
                {"mode": "nominal", "speed": 1.5, "light": "dark"},
            ]
        )
        result = tway_coverage(tiny_space, encoded, t=2)
        assert result.covered + len(result.uncovered) == result.total


class TestExhaustiveCrossCheck:
    @pytest.mark.parametrize("t", [1, 2, 3])
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_against_the_brute_force_reference(self, tiny_space: ScenarioSpace, t: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        rows = np.column_stack(
            [rng.integers(0, c, size=7) for c in tiny_space.level_counts]
        ).astype(np.int64)
        result = tway_coverage(tiny_space, rows, t=t)
        covered, total = brute_force_coverage(tiny_space, rows, t)
        assert (result.covered, result.total) == (covered, total)

    def test_uncovered_tuples_are_genuinely_absent(self, tiny_space: ScenarioSpace) -> None:
        records = [
            {"mode": "safe", "speed": 0.5, "light": "sun"},
            {"mode": "nominal", "speed": 1.5, "light": "dusk"},
        ]
        result = tway_coverage(tiny_space, tiny_space.encode_many(records), t=2)
        for tup in result.uncovered:
            assert not any(tup.matches(r, tiny_space) for r in records)


class TestPerCombinationBreakdown:
    def test_entry_per_parameter_subset(self, tiny_space: ScenarioSpace) -> None:
        encoded = tiny_space.encode_many([{"mode": "safe", "speed": 0.5, "light": "sun"}])
        result = tway_coverage(tiny_space, encoded, t=2)
        assert len(result.per_combination) == 3  # C(3, 2)
        assert sum(c.covered for c in result.per_combination) == result.covered
        assert sum(c.total for c in result.per_combination) == result.total

    def test_worst_combinations_are_ordered_by_fraction(self, rendezvous_space, rendezvous_campaign):
        result = tway_coverage(
            rendezvous_space, rendezvous_space.encode_many(rendezvous_campaign), t=2
        )
        worst = result.worst_combinations(3)
        fractions = [c.fraction for c in worst]
        assert fractions == sorted(fractions)
        assert len(result.worst_combinations(0)) == 0
        with pytest.raises(ValueError, match="non-negative"):
            result.worst_combinations(-1)

    def test_uncovered_involving_filters_by_parameter(self, rendezvous_space, rendezvous_campaign):
        result = tway_coverage(
            rendezvous_space, rendezvous_space.encode_many(rendezvous_campaign), t=2
        )
        subset = result.uncovered_involving("closing_speed")
        assert len(subset) == 2
        assert all("closing_speed" in u.parameters for u in subset)


class TestInputValidation:
    def test_t_out_of_range(self, tiny_space: ScenarioSpace) -> None:
        encoded = tiny_space.encode_many([{"mode": "safe", "speed": 0.5, "light": "sun"}])
        for t in (0, 4):
            with pytest.raises(ValueError, match="1 <= t <="):
                tway_coverage(tiny_space, encoded, t=t)

    def test_wrong_column_count(self, tiny_space: ScenarioSpace) -> None:
        with pytest.raises(ValueError, match="shape"):
            tway_coverage(tiny_space, np.zeros((3, 2), dtype=np.int64), t=2)

    def test_float_array_is_rejected(self, tiny_space: ScenarioSpace) -> None:
        with pytest.raises(TypeError, match="integer level indices"):
            tway_coverage(tiny_space, np.zeros((2, 3), dtype=float), t=2)

    def test_negative_level_index(self, tiny_space: ScenarioSpace) -> None:
        with pytest.raises(ValueError, match="negative level index"):
            tway_coverage(tiny_space, -np.ones((2, 3), dtype=np.int64), t=2)

    def test_level_index_beyond_the_declaration(self, tiny_space: ScenarioSpace) -> None:
        bad = np.array([[0, 0, 9]], dtype=np.int64)
        with pytest.raises(ValueError, match="levels but encoded"):
            tway_coverage(tiny_space, bad, t=2)

    def test_max_tuples_refuses_a_blow_up(self) -> None:
        space = ScenarioSpace([Parameter.categorical(f"p{i}", list("abcd")) for i in range(12)])
        encoded = np.zeros((1, 12), dtype=np.int64)
        with pytest.raises(ValueError, match="exceeds max_tuples"):
            tway_coverage(space, encoded, t=5, max_tuples=1000)
