"""Grid occupancy, the Feller reference and the empty-block flood fill."""

from __future__ import annotations

import numpy as np
import pytest

from scenariocov import (
    Parameter,
    ScenarioSpace,
    expected_occupied_cells,
    grid_occupancy,
)


@pytest.fixture
def grid_space() -> ScenarioSpace:
    """Two continuous parameters with 3 and 4 bins: a 12-cell grid."""
    return ScenarioSpace(
        [
            Parameter.continuous("x", [0.0, 1.0, 2.0, 3.0]),
            Parameter.continuous("y", [0.0, 0.25, 0.5, 0.75, 1.0]),
        ]
    )


class TestExpectedOccupiedCells:
    def test_hand_worked_values(self) -> None:
        # One throw always occupies one cell, for any C.
        assert expected_occupied_cells(1, 7) == pytest.approx(1.0, abs=1e-12)
        # Zero throws occupy nothing.
        assert expected_occupied_cells(0, 7) == pytest.approx(0.0, abs=1e-12)
        # C = 2, n = 2: 2 * (1 - (1/2)^2) = 1.5, by hand.
        assert expected_occupied_cells(2, 2) == pytest.approx(1.5, abs=1e-12)
        # C = 4, n = 3: 4 * (1 - (3/4)^3) = 4 * (1 - 27/64) = 2.3125, by hand.
        assert expected_occupied_cells(3, 4) == pytest.approx(2.3125, abs=1e-12)

    def test_single_cell_is_always_occupied_after_one_throw(self) -> None:
        assert expected_occupied_cells(5, 1) == pytest.approx(1.0, abs=1e-12)

    def test_monotone_in_the_number_of_throws(self) -> None:
        values = [expected_occupied_cells(n, 10) for n in range(0, 20)]
        assert all(b >= a for a, b in zip(values, values[1:], strict=False))

    @pytest.mark.parametrize(("n", "c"), [(-1, 4), (3, 0)])
    def test_invalid_arguments(self, n: int, c: int) -> None:
        with pytest.raises(ValueError):
            expected_occupied_cells(n, c)

    def test_monte_carlo_agreement_on_an_equiprobable_grid(self) -> None:
        # The model equation (3) describes exactly: independent uniform throws
        # into equiprobable cells. 2000 replicates of n = 30 into C = 25.
        # Compared with a tolerance on both sides, never for equality.
        rng = np.random.default_rng(20261016)
        n, c, reps = 30, 25, 2000
        counts = [len(set(rng.integers(0, c, size=n).tolist())) for _ in range(reps)]
        measured = float(np.mean(counts))
        predicted = expected_occupied_cells(n, c)
        standard_error = float(np.std(counts, ddof=1)) / np.sqrt(reps)
        assert abs(measured - predicted) < 4.0 * standard_error


class TestGridOccupancy:
    def test_hand_counted_occupancy(self, grid_space: ScenarioSpace) -> None:
        # Values chosen well inside their bins, so no bin-edge comparison is
        # involved. Three scenarios land in cells (0,0), (0,0) and (2,3).
        records = [
            {"x": 0.5, "y": 0.1},
            {"x": 0.7, "y": 0.2},
            {"x": 2.5, "y": 0.9},
        ]
        occ = grid_occupancy(grid_space, grid_space.encode_many(records))
        assert occ.shape == (3, 4)
        assert occ.n_cells == 12
        assert occ.occupied == 2
        assert occ.empty == 10
        assert occ.fraction == pytest.approx(2 / 12, abs=1e-12)
        assert int(occ.counts[0, 0]) == 2
        assert int(occ.counts[2, 3]) == 1

    def test_empty_campaign_occupies_nothing(self, grid_space: ScenarioSpace) -> None:
        occ = grid_occupancy(grid_space, grid_space.encode_many([]))
        assert occ.occupied == 0
        assert occ.largest_empty_block() == 12

    def test_full_grid_has_no_empty_block(self, grid_space: ScenarioSpace) -> None:
        records = [
            {"x": 0.5 + i, "y": 0.125 + 0.25 * j} for i in range(3) for j in range(4)
        ]
        occ = grid_occupancy(grid_space, grid_space.encode_many(records))
        assert occ.occupied == occ.n_cells
        assert occ.empty_cells() == ()
        assert occ.empty_blocks() == ()
        assert occ.largest_empty_block() == 0

    def test_empty_blocks_use_von_neumann_connectivity(self, grid_space) -> None:
        # Occupy everything except cells (0,0) and (1,1), which touch only at a
        # corner, so they must be two blocks of 1, not one block of 2.
        records = [
            {"x": 0.5 + i, "y": 0.125 + 0.25 * j}
            for i in range(3)
            for j in range(4)
            if (i, j) not in {(0, 0), (1, 1)}
        ]
        occ = grid_occupancy(grid_space, grid_space.encode_many(records))
        blocks = occ.empty_blocks()
        assert [b.size for b in blocks] == [1, 1]
        assert occ.largest_empty_block() == 1

    def test_a_contiguous_empty_strip_is_one_block(self, grid_space: ScenarioSpace) -> None:
        records = [
            {"x": 0.5 + i, "y": 0.125 + 0.25 * j}
            for i in range(3)
            for j in range(4)
            if i != 1
        ]
        occ = grid_occupancy(grid_space, grid_space.encode_many(records))
        blocks = occ.empty_blocks()
        assert len(blocks) == 1
        assert blocks[0].size == 4

    def test_explicit_parameter_selection(self, grid_space: ScenarioSpace) -> None:
        occ = grid_occupancy(
            grid_space, grid_space.encode_many([{"x": 0.5, "y": 0.1}]), parameters=["y"]
        )
        assert occ.shape == (4,)
        assert occ.n_cells == 4

    def test_duplicate_parameters_are_rejected(self, grid_space: ScenarioSpace) -> None:
        encoded = grid_space.encode_many([{"x": 0.5, "y": 0.1}])
        with pytest.raises(ValueError, match="must not repeat"):
            grid_occupancy(grid_space, encoded, parameters=["x", "x"])

    def test_empty_parameter_list_is_rejected(self, grid_space: ScenarioSpace) -> None:
        encoded = grid_space.encode_many([{"x": 0.5, "y": 0.1}])
        with pytest.raises(ValueError, match="at least one parameter"):
            grid_occupancy(grid_space, encoded, parameters=[])

    def test_purely_categorical_space_needs_explicit_parameters(self) -> None:
        space = ScenarioSpace([Parameter.categorical("c", ["a", "b"])])
        encoded = space.encode_many([{"c": "a"}])
        with pytest.raises(ValueError, match="no continuous parameters"):
            grid_occupancy(space, encoded)

    def test_max_cells_refuses_a_blow_up(self) -> None:
        space = ScenarioSpace(
            [Parameter.continuous(f"p{i}", list(range(11))) for i in range(8)]
        )
        encoded = np.zeros((1, 8), dtype=np.int64)
        with pytest.raises(ValueError, match="exceeds max_cells"):
            grid_occupancy(space, encoded, max_cells=1000)

    def test_wrong_shape_is_rejected(self, grid_space: ScenarioSpace) -> None:
        with pytest.raises(ValueError, match="shape"):
            grid_occupancy(grid_space, np.zeros((2, 5), dtype=np.int64))

    def test_committed_campaign_known_answer(self, rendezvous_space, rendezvous_campaign) -> None:
        # Known answer from the committed fixture at seed 20261016: 108 of the
        # 144 cells of the 4x4x3x3 continuous grid are occupied, and the
        # planted G3 region (closing_speed bin 3 crossed with range bin 3,
        # 1 x 1 x 3 x 3 = 9 cells) is entirely empty.
        occ = grid_occupancy(rendezvous_space, rendezvous_space.encode_many(rendezvous_campaign))
        assert occ.shape == (4, 4, 3, 3)
        assert occ.n_cells == 144
        assert occ.occupied == 108
        planted = occ.counts[3, 3, :, :]
        assert int(planted.sum()) == 0
        assert planted.size == 9
