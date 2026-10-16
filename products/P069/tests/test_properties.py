"""Property-based tests, plus every counterexample Hypothesis has found, pinned.

``.hypothesis/`` is gitignored, so a falsifying example found by one run is
lost on a fresh clone. Anything Hypothesis finds here is written out as an
explicit regression test with literal inputs in the body, in
:class:`TestPinnedCounterexamples` at the end of the file.
"""

from __future__ import annotations

import math
from itertools import product

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from scenariocov import (
    GreedyMaxMinStrategy,
    Parameter,
    ScenarioSpace,
    SelectionContext,
    count_tway_tuples,
    expected_occupied_cells,
    grid_occupancy,
    tway_coverage,
)

SLOW = settings(
    max_examples=120,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

level_counts = st.lists(st.integers(min_value=2, max_value=4), min_size=1, max_size=5)


def space_from_counts(counts: list[int]) -> ScenarioSpace:
    """A categorical space with the given level counts."""
    return ScenarioSpace(
        [Parameter.categorical(f"p{i}", [f"l{j}" for j in range(c)]) for i, c in enumerate(counts)]
    )


def rows_from(counts: list[int], draws: list[list[int]]) -> np.ndarray:
    """Clamp arbitrary integers into valid level indices."""
    if not draws:
        return np.zeros((0, len(counts)), dtype=np.int64)
    return np.asarray(
        [[d % c for d, c in zip(row, counts, strict=True)] for row in draws], dtype=np.int64
    )


class TestCombinatorialIdentities:
    @given(counts=level_counts, data=st.data())
    @settings(max_examples=150, deadline=None)
    def test_count_matches_enumeration(self, counts: list[int], data) -> None:
        t = data.draw(st.integers(min_value=1, max_value=len(counts)))
        space = space_from_counts(counts)
        result = tway_coverage(space, np.zeros((0, len(counts)), dtype=np.int64), t=t)
        # The closed form and the enumeration must agree; both are integers.
        assert count_tway_tuples(counts, t) == result.total == len(result.uncovered)

    @given(counts=level_counts, data=st.data())
    @settings(max_examples=150, deadline=None)
    def test_covered_plus_uncovered_is_total(self, counts: list[int], data) -> None:
        t = data.draw(st.integers(min_value=1, max_value=len(counts)))
        n = data.draw(st.integers(min_value=0, max_value=8))
        draws = data.draw(
            st.lists(
                st.lists(st.integers(0, 50), min_size=len(counts), max_size=len(counts)),
                min_size=n, max_size=n,
            )
        )
        space = space_from_counts(counts)
        result = tway_coverage(space, rows_from(counts, draws), t=t)
        assert result.covered + len(result.uncovered) == result.total
        assert 0 <= result.fraction <= 1.0

    @given(counts=level_counts, data=st.data())
    @settings(max_examples=120, deadline=None)
    def test_adding_a_scenario_never_lowers_coverage(self, counts: list[int], data) -> None:
        t = data.draw(st.integers(min_value=1, max_value=len(counts)))
        draws = data.draw(
            st.lists(
                st.lists(st.integers(0, 50), min_size=len(counts), max_size=len(counts)),
                min_size=1, max_size=6,
            )
        )
        space = space_from_counts(counts)
        rows = rows_from(counts, draws)
        before = tway_coverage(space, rows[:-1], t=t).covered
        after = tway_coverage(space, rows, t=t).covered
        # Exact integer monotonicity: the covered set only ever grows.
        assert after >= before

    @given(counts=level_counts, data=st.data())
    @settings(max_examples=60, deadline=None)
    def test_exhaustive_campaign_covers_everything(self, counts: list[int], data) -> None:
        assume(math.prod(counts) <= 256)
        t = data.draw(st.integers(min_value=1, max_value=len(counts)))
        space = space_from_counts(counts)
        rows = np.asarray(list(product(*(range(c) for c in counts))), dtype=np.int64)
        result = tway_coverage(space, rows, t=t)
        assert result.covered == result.total
        assert result.uncovered == ()

    @given(counts=level_counts, data=st.data())
    @settings(max_examples=80, deadline=None)
    def test_full_strength_coverage_counts_distinct_rows(self, counts: list[int], data) -> None:
        draws = data.draw(
            st.lists(
                st.lists(st.integers(0, 20), min_size=len(counts), max_size=len(counts)),
                min_size=1, max_size=8,
            )
        )
        space = space_from_counts(counts)
        rows = rows_from(counts, draws)
        distinct = len({tuple(int(v) for v in row) for row in rows})
        assert tway_coverage(space, rows, t=len(counts)).covered == distinct

    @given(counts=level_counts)
    @settings(max_examples=80, deadline=None)
    def test_one_way_total_is_the_sum_of_level_counts(self, counts: list[int]) -> None:
        assert count_tway_tuples(counts, 1) == sum(counts)


class TestOccupancyIdentities:
    @given(
        n=st.integers(min_value=0, max_value=500),
        c=st.integers(min_value=1, max_value=500),
    )
    @settings(max_examples=200, deadline=None)
    def test_expected_occupancy_is_bounded_by_n_and_c(self, n: int, c: int) -> None:
        value = expected_occupied_cells(n, c)
        # Cannot exceed either the number of throws or the number of cells.
        assert -1e-9 <= value <= min(n, c) + 1e-9

    @given(
        n=st.integers(min_value=0, max_value=200),
        c=st.integers(min_value=1, max_value=200),
    )
    @settings(max_examples=150, deadline=None)
    def test_expected_occupancy_is_monotone_in_n(self, n: int, c: int) -> None:
        assert expected_occupied_cells(n + 1, c) >= expected_occupied_cells(n, c) - 1e-12

    @given(counts=st.lists(st.integers(2, 4), min_size=1, max_size=3), data=st.data())
    @settings(max_examples=100, deadline=None)
    def test_occupied_equals_distinct_cells(self, counts: list[int], data) -> None:
        space = ScenarioSpace(
            [
                Parameter.continuous(f"p{i}", [float(j) for j in range(c + 1)])
                for i, c in enumerate(counts)
            ]
        )
        draws = data.draw(
            st.lists(
                st.lists(st.integers(0, 10), min_size=len(counts), max_size=len(counts)),
                min_size=0, max_size=10,
            )
        )
        rows = rows_from(counts, draws)
        occ = grid_occupancy(space, rows)
        distinct = len({tuple(int(v) for v in row) for row in rows})
        assert occ.occupied == distinct
        assert occ.occupied + occ.empty == occ.n_cells
        assert sum(b.size for b in occ.empty_blocks()) == occ.empty


class TestEmbeddingIdentities:
    @given(
        values=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=1, max_size=6,
        )
    )
    @settings(max_examples=120, deadline=None)
    def test_continuous_column_stays_in_the_unit_interval(self, values: list[float]) -> None:
        space = ScenarioSpace([Parameter.continuous("x", [0.0, 0.5, 1.0])])
        column = space.normalise([{"x": v} for v in values])[:, 0]
        assert np.all(column >= -1e-12)
        assert np.all(column <= 1.0 + 1e-12)

    @given(
        value=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        label=st.sampled_from(["a", "b", "c"]),
    )
    @settings(max_examples=100, deadline=None)
    def test_identical_records_are_at_zero_distance(self, value: float, label: str) -> None:
        space = ScenarioSpace(
            [Parameter.continuous("x", [0.0, 0.5, 1.0]), Parameter.categorical("c", ["a", "b", "c"])]
        )
        points = space.normalise([{"x": value, "c": label}] * 2)
        assert float(np.linalg.norm(points[0] - points[1])) == pytest.approx(0.0, abs=1e-12)


class TestMaxMinIdentity:
    @given(
        executed=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=1, max_size=6,
        ),
        pool=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=1, max_size=8,
        ),
    )
    @settings(max_examples=120, deadline=None)
    def test_first_pick_maximises_the_distance_to_the_executed_set(
        self, executed: list[float], pool: list[float]
    ) -> None:
        space = ScenarioSpace([Parameter.continuous("x", [0.0, 0.5, 1.0])])
        context = SelectionContext(
            space=space,
            executed_encoded=space.encode_many([{"x": v} for v in executed]),
            executed_norm=space.normalise([{"x": v} for v in executed]),
            candidate_encoded=space.encode_many([{"x": v} for v in pool]),
            candidate_norm=space.normalise([{"x": v} for v in pool]),
        )
        pick = GreedyMaxMinStrategy().select(context, 1, np.random.default_rng(0))[0]
        distances = [min(abs(c - e) for e in executed) for c in pool]
        # Equation (5) is an argmax over floating-point distances, so a tie
        # can be resolved either way by rounding. The guarantee asserted is on
        # the VALUE: the chosen candidate's distance agrees with the maximum
        # to within rounding, never that the index is the first maximiser.
        assert distances[pick] >= max(distances) - 1e-12


class TestPinnedCounterexamples:
    """Literal regression tests for falsifying examples Hypothesis has found.

    Empty of counterexamples as of the build session of 2026-10-16: the
    property tests above were run at the example counts declared in their
    settings and none of them failed. The class is kept so that the next
    counterexample has an obvious home, and so that a reader can see that the
    absence of pinned cases is a measured result rather than an omission.
    """

    def test_single_level_parameter_is_rejected_rather_than_mishandled(self) -> None:
        # Found while writing the generators above: a parameter with one level
        # is meaningless for coverage and must be refused at declaration time
        # rather than silently producing |T_1| = 1.
        with pytest.raises(ValueError, match="at least 2 levels"):
            Parameter.categorical("p", ["only"])

    def test_zero_scenarios_with_t_equal_to_k(self) -> None:
        # Found while writing test_full_strength_coverage_counts_distinct_rows:
        # the empty campaign must give covered == 0 and the whole cross product
        # uncovered, not an empty uncovered list.
        space = space_from_counts([2, 3])
        result = tway_coverage(space, np.zeros((0, 2), dtype=np.int64), t=2)
        assert result.covered == 0
        assert len(result.uncovered) == 6
