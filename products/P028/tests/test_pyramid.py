"""The Pyramid scan order and the fourth-star confirmation (Eq. Y1)."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from skymatch.catalogue import StarCatalogue
from skymatch.pairtable import PairTable
from skymatch.pyramid import confirm_with_fourth_star, pyramid_triple_order
from skymatch.triangle import triangle_edge_angles


@pytest.fixture(scope="module")
def toy_table(toy_catalogue: StarCatalogue) -> PairTable:
    return PairTable(toy_catalogue, np.radians(20.0))


class TestScanOrder:
    def test_four_stars_by_hand(self) -> None:
        # dj = 1: dk = 1 -> i = 0, 1  giving (0,1,2), (1,2,3)
        #         dk = 2 -> i = 0     giving (0,1,3)
        # dj = 2: dk = 1 -> i = 0     giving (0,2,3)
        assert pyramid_triple_order(4) == [(0, 1, 2), (1, 2, 3), (0, 1, 3), (0, 2, 3)]

    @pytest.mark.parametrize("n", [3, 4, 5, 7, 10, 14])
    def test_enumerates_every_triple_exactly_once(self, n: int) -> None:
        order = pyramid_triple_order(n)
        assert len(order) == n * (n - 1) * (n - 2) // 6
        assert sorted(order) == sorted(combinations(range(n), 3))
        assert len(set(order)) == len(order)

    def test_indices_are_increasing_within_a_triple(self) -> None:
        assert all(i < j < k for i, j, k in pyramid_triple_order(9))

    def test_consecutive_triples_change_spots(self) -> None:
        # The point of the gap ordering: the first few attempts must not all
        # share the same spot, so a single false detection cannot poison a
        # long run of consecutive attempts.
        order = pyramid_triple_order(8)
        assert len(set(order[0]) & set(order[1])) <= 2

    def test_rejects_too_few(self) -> None:
        with pytest.raises(ValueError, match="at least 3"):
            pyramid_triple_order(2)


class TestConfirmation:
    def test_confirms_the_known_pyramid(
        self, toy_catalogue: StarCatalogue, toy_table: PairTable
    ) -> None:
        # Triangle (0, 1, 2) confirmed by star 4: its three angles to the
        # triangle are looked up from the same catalogue, so the match is
        # exact and unique.
        v = toy_catalogue.vectors
        t_ar = triangle_edge_angles(v, 0, 4, 1)[0]
        t_br = triangle_edge_angles(v, 1, 4, 2)[0]
        t_cr = triangle_edge_angles(v, 2, 4, 0)[0]
        n, d, rms = confirm_with_fourth_star(
            toy_table, np.array([0]), np.array([1]), np.array([2]), t_ar, t_br, t_cr, 1e-9
        )
        assert n[0] == 1
        assert d[0] == 4
        assert rms[0] < 1e-9

    def test_rejects_a_wrong_triangle(
        self, toy_catalogue: StarCatalogue, toy_table: PairTable
    ) -> None:
        v = toy_catalogue.vectors
        t_ar = triangle_edge_angles(v, 0, 4, 1)[0]
        t_br = triangle_edge_angles(v, 1, 4, 2)[0]
        t_cr = triangle_edge_angles(v, 2, 4, 0)[0]
        # Same fourth-star angles, but the wrong candidate triangle.
        n, d, _ = confirm_with_fourth_star(
            toy_table, np.array([1]), np.array([2]), np.array([3]), t_ar, t_br, t_cr, 1e-9
        )
        assert n[0] == 0
        assert d[0] == -1

    def test_batched_over_candidates_and_fourth_spots(
        self, toy_catalogue: StarCatalogue, toy_table: PairTable
    ) -> None:
        v = toy_catalogue.vectors
        good = (
            triangle_edge_angles(v, 0, 4, 1)[0],
            triangle_edge_angles(v, 1, 4, 2)[0],
            triangle_edge_angles(v, 2, 4, 0)[0],
        )
        a = np.array([0, 1])
        b = np.array([1, 2])
        c = np.array([2, 3])
        t_ar = np.array([good[0], good[0]])
        t_br = np.array([good[1], good[1]])
        t_cr = np.array([good[2], good[2]])
        n, d, _ = confirm_with_fourth_star(toy_table, a, b, c, t_ar, t_br, t_cr, 1e-9)
        assert n.tolist() == [1, 0]
        assert d.tolist() == [4, -1]

    def test_empty_input(self, toy_table: PairTable) -> None:
        empty = np.empty(0, dtype=np.int64)
        n, d, rms = confirm_with_fourth_star(toy_table, empty, empty, empty, 0.1, 0.1, 0.1, 1e-6)
        assert n.size == d.size == rms.size == 0

    def test_a_huge_tolerance_makes_the_match_ambiguous(
        self, toy_catalogue: StarCatalogue, toy_table: PairTable
    ) -> None:
        # With a 20 deg window every catalogue star fits, so the fourth spot
        # cannot single one out and the candidate is NOT confirmed. That is
        # the behaviour that keeps the Pyramid rule's false-ID rate low.
        v = toy_catalogue.vectors
        t_ar = triangle_edge_angles(v, 0, 4, 1)[0]
        n, d, _ = confirm_with_fourth_star(
            toy_table, np.array([0]), np.array([1]), np.array([2]),
            t_ar, t_ar, t_ar, np.radians(20.0),
        )
        assert n[0] > 1
        assert d[0] == -1

    def test_validation(self, toy_table: PairTable) -> None:
        one = np.array([0])
        with pytest.raises(ValueError, match="tolerance_rad"):
            confirm_with_fourth_star(toy_table, one, one, one, 0.1, 0.1, 0.1, 0.0)
        with pytest.raises(ValueError, match="same shape"):
            confirm_with_fourth_star(
                toy_table, np.array([0, 1]), one, one, 0.1, 0.1, 0.1, 1e-6
            )
