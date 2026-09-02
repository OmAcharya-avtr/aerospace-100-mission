"""The pair table and its three indexes."""

from __future__ import annotations

import numpy as np
import pytest

from skymatch.camera import CameraModel
from skymatch.catalogue import StarCatalogue, generate_catalogue
from skymatch.geometry import angular_separation
from skymatch.pairtable import PairTable, expected_pair_count


@pytest.fixture(scope="module")
def toy_table(toy_catalogue: StarCatalogue) -> PairTable:
    return PairTable(toy_catalogue, np.radians(20.0))


class TestKnownAnswers:
    def test_expected_pair_count_by_hand(self) -> None:
        # C(100, 2) * (1 - cos(60 deg)) / 2 = 4950 * 0.25 = 1237.5.
        assert expected_pair_count(100, np.pi / 3) == pytest.approx(1237.5)

    def test_toy_pair_count(self, toy_table: PairTable) -> None:
        # 5 stars, all pairs within 20 deg except (star 2 at dec +4, ra 0) to
        # (star 3 at ra 8): those are 8.94 deg apart, still inside. So all
        # C(5,2) = 10 pairs are stored.
        assert toy_table.n_pairs == 10
        assert toy_table.n_stars == 5

    def test_toy_separations_by_hand(self, toy_table: PairTable) -> None:
        # Stars 0 and 1 are 3.0000 deg apart on the equator; 0 and 2 are
        # 4.0000 deg apart on the ra = 0 meridian; 0 and 3 are 8.0000 deg.
        for a, b, deg in ((0, 1, 3.0), (0, 2, 4.0), (0, 3, 8.0)):
            got = toy_table.separation_lookup(np.array([a]), np.array([b]))[0]
            assert np.degrees(got) == pytest.approx(deg, abs=1e-12)

    def test_separations_are_sorted(self, toy_table: PairTable) -> None:
        assert np.all(np.diff(toy_table.separations) >= 0.0)

    def test_lookup_is_symmetric(self, toy_table: PairTable) -> None:
        a = np.array([0, 1, 2, 3])
        b = np.array([1, 2, 3, 4])
        assert np.allclose(
            toy_table.separation_lookup(a, b), toy_table.separation_lookup(b, a)
        )

    def test_lookup_returns_nan_for_a_missing_pair(
        self, toy_catalogue: StarCatalogue
    ) -> None:
        narrow = PairTable(toy_catalogue, np.radians(3.5))
        # Only the 3.0 deg pair (0, 1) and the 3.28 deg pair (1, 4) fit.
        assert np.isnan(narrow.separation_lookup(np.array([0]), np.array([3]))[0])
        assert not np.isnan(narrow.separation_lookup(np.array([0]), np.array([1]))[0])

    def test_lookup_of_a_star_with_itself_is_nan(self, toy_table: PairTable) -> None:
        assert np.isnan(toy_table.separation_lookup(np.array([2]), np.array([2]))[0])


class TestQueries:
    def test_ordered_range_returns_both_orientations(self, toy_table: PairTable) -> None:
        lo, hi = np.radians(2.9), np.radians(3.1)
        a, b = toy_table.ordered_range(lo, hi)
        assert a.size == 2  # one pair, two orientations
        assert set(zip(a.tolist(), b.tolist(), strict=True)) == {(0, 1), (1, 0)}

    def test_ordered_range_empty(self, toy_table: PairTable) -> None:
        a, b = toy_table.ordered_range(np.radians(19.0), np.radians(19.5))
        assert a.size == 0 and b.size == 0

    def test_neighbours_range(self, toy_table: PairTable) -> None:
        rows, nb = toy_table.neighbours_range(
            np.array([0]), np.radians(2.5), np.radians(4.5)
        )
        # From star 0: star 1 at 3.0 deg and star 2 at 4.0 deg.
        assert sorted(nb.tolist()) == [1, 2]
        assert set(rows.tolist()) == {0}

    def test_neighbours_range_broadcasts_bounds(self, toy_table: PairTable) -> None:
        stars = np.array([0, 0])
        lo = np.radians([2.5, 3.5])
        hi = np.radians([3.5, 4.5])
        rows, nb = toy_table.neighbours_range(stars, lo, hi)
        assert nb[rows == 0].tolist() == [1]
        assert nb[rows == 1].tolist() == [2]

    def test_neighbours_range_empty_input(self, toy_table: PairTable) -> None:
        rows, nb = toy_table.neighbours_range(np.array([], dtype=int), 0.0, 1.0)
        assert rows.size == 0 and nb.size == 0

    def test_matches_brute_force(self, camera: CameraModel) -> None:
        cat = generate_catalogue(4.5, seed=9)
        table = PairTable(cat, camera.max_separation_rad)
        full = np.degrees(
            angular_separation(
                np.repeat(cat.vectors, cat.n_stars, axis=0),
                np.tile(cat.vectors, (cat.n_stars, 1)),
            )
        ).reshape(cat.n_stars, cat.n_stars)
        iu = np.triu_indices(cat.n_stars, k=1)
        brute = int(np.sum(full[iu] <= np.degrees(camera.max_separation_rad)))
        assert table.n_pairs == brute

    def test_size_matches_eq_p1_within_sampling_error(self, table: PairTable) -> None:
        predicted = expected_pair_count(table.n_stars, table.max_separation_rad)
        assert abs(table.n_pairs / predicted - 1.0) < 0.02

    def test_nbytes_is_positive(self, toy_table: PairTable) -> None:
        assert toy_table.nbytes > 0

    def test_separations_view_is_read_only(self, toy_table: PairTable) -> None:
        with pytest.raises(ValueError):
            toy_table.separations[0] = 1.0


class TestValidation:
    def test_rejects_bad_separation(self, toy_catalogue: StarCatalogue) -> None:
        for bad in (0.0, -1.0, 4.0, np.nan):
            with pytest.raises(ValueError, match="max_separation_rad"):
                PairTable(toy_catalogue, bad)

    def test_rejects_tiny_catalogue(self, toy_catalogue: StarCatalogue) -> None:
        one = StarCatalogue(
            ra=toy_catalogue.ra[:1],
            dec=toy_catalogue.dec[:1],
            magnitude=toy_catalogue.magnitude[:1],
            vectors=toy_catalogue.vectors[:1],
            magnitude_limit=6.0,
            seed=0,
        )
        with pytest.raises(ValueError, match="at least 2 stars"):
            PairTable(one, 0.1)

    def test_rejects_a_separation_with_no_pairs(
        self, toy_catalogue: StarCatalogue
    ) -> None:
        with pytest.raises(ValueError, match="no star pairs closer"):
            PairTable(toy_catalogue, np.radians(0.001))

    def test_rejects_out_of_range_indices(self, toy_table: PairTable) -> None:
        with pytest.raises(ValueError, match="out of range"):
            toy_table.separation_lookup(np.array([99]), np.array([0]))
        with pytest.raises(ValueError, match="out of range"):
            toy_table.separation_lookup(np.array([0]), np.array([99]))
        with pytest.raises(ValueError, match="out of range"):
            toy_table.neighbours_range(np.array([99]), 0.0, 1.0)

    def test_rejects_inverted_range(self, toy_table: PairTable) -> None:
        with pytest.raises(ValueError, match="hi_rad"):
            toy_table.ordered_range(1.0, 0.5)
        with pytest.raises(ValueError, match="hi_rad"):
            toy_table.neighbours_range(np.array([0]), 1.0, 0.5)

    def test_rejects_mismatched_lookup_shapes(self, toy_table: PairTable) -> None:
        with pytest.raises(ValueError, match="same shape"):
            toy_table.separation_lookup(np.array([0, 1]), np.array([1]))

    def test_expected_pair_count_validation(self) -> None:
        with pytest.raises(ValueError, match="n_stars"):
            expected_pair_count(-1, 0.1)
        with pytest.raises(ValueError, match="max_separation_rad"):
            expected_pair_count(10, 4.0)
