"""Triangle matching (Eq. T1) and the tolerance (Eq. T2)."""

from __future__ import annotations

import numpy as np
import pytest

from skymatch.catalogue import StarCatalogue
from skymatch.geometry import ARCSEC, random_rotation
from skymatch.pairtable import PairTable
from skymatch.triangle import (
    separation_tolerance,
    triangle_candidates,
    triangle_edge_angles,
)


@pytest.fixture(scope="module")
def toy_table(toy_catalogue: StarCatalogue) -> PairTable:
    return PairTable(toy_catalogue, np.radians(20.0))


class TestTolerance:
    def test_by_hand(self) -> None:
        # tau = 3 * sqrt(2) * 10 arcsec = 42.4264 arcsec = 2.05714e-4 rad.
        tau = separation_tolerance(10.0)
        assert tau / ARCSEC == pytest.approx(42.42640687, rel=1e-9)

    def test_scales_with_k(self) -> None:
        assert separation_tolerance(10.0, 6.0) == pytest.approx(
            2.0 * separation_tolerance(10.0, 3.0)
        )

    def test_zero_noise_gives_a_positive_floor(self) -> None:
        assert separation_tolerance(0.0) == 1e-9

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="centroid_sigma_arcsec"):
            separation_tolerance(-1.0)
        with pytest.raises(ValueError, match="k_sigma"):
            separation_tolerance(1.0, 0.0)


class TestEdgeAngles:
    def test_known_angles(self, toy_catalogue: StarCatalogue) -> None:
        # Stars 0, 1, 2 of the toy catalogue: (0,0), (3 deg, 0), (0, 4 deg).
        # t_01 = 3.0000 deg, t_02 = 4.0000 deg, and t_12 is the hypotenuse of
        # a spherical right triangle: cos(t_12) = cos(3 deg) cos(4 deg)
        # = 0.9961965, so t_12 = 4.99854 deg -- a little UNDER the flat-sky
        # 5.0, because the spherical Pythagoras law contracts rather than
        # expands. Getting the sign of that 1.5e-3 deg wrong would be a
        # 5.3 arcsec bias, a quarter of the default tolerance at sigma = 5.
        t01, t02, t12 = triangle_edge_angles(toy_catalogue.vectors, 0, 1, 2)
        assert np.degrees(t01) == pytest.approx(3.0, abs=1e-12)
        assert np.degrees(t02) == pytest.approx(4.0, abs=1e-12)
        expected = np.degrees(np.arccos(np.cos(np.radians(3.0)) * np.cos(np.radians(4.0))))
        assert np.degrees(t12) == pytest.approx(expected, abs=1e-12)
        assert np.degrees(t12) == pytest.approx(4.998537, abs=1e-5)

    def test_validation(self, toy_catalogue: StarCatalogue) -> None:
        v = toy_catalogue.vectors
        with pytest.raises(ValueError, match="distinct"):
            triangle_edge_angles(v, 0, 0, 1)
        with pytest.raises(ValueError, match="out of range"):
            triangle_edge_angles(v, 0, 1, 99)
        with pytest.raises(ValueError, match=r"shape \(N, 3\)"):
            triangle_edge_angles(np.zeros((3, 2)), 0, 1, 2)


class TestCandidates:
    def test_finds_the_known_triangle(
        self, toy_catalogue: StarCatalogue, toy_table: PairTable
    ) -> None:
        t01, t02, t12 = triangle_edge_angles(toy_catalogue.vectors, 0, 1, 2)
        a, b, c, res = triangle_candidates(toy_table, t01, t02, t12, 1e-9)
        assert list(zip(a.tolist(), b.tolist(), c.tolist(), strict=True)) == [(0, 1, 2)]
        assert np.allclose(res, 0.0, atol=1e-9)

    def test_ordering_is_preserved(
        self, toy_catalogue: StarCatalogue, toy_table: PairTable
    ) -> None:
        # Asking for (t_02, t_01, t_12) -- swapping the roles of the second and
        # third observed stars -- must return the catalogue triple swapped the
        # same way.
        t01, t02, t12 = triangle_edge_angles(toy_catalogue.vectors, 0, 1, 2)
        a, b, c, _ = triangle_candidates(toy_table, t02, t01, t12, 1e-9)
        assert list(zip(a.tolist(), b.tolist(), c.tolist(), strict=True)) == [(0, 2, 1)]

    def test_no_match_returns_empty(self, toy_table: PairTable) -> None:
        a, b, c, res = triangle_candidates(
            toy_table, np.radians(19.5), np.radians(19.5), np.radians(19.5), 1e-6
        )
        assert a.size == b.size == c.size == 0
        assert res.shape == (0, 3)

    def test_rotation_invariance(
        self, catalogue: StarCatalogue, table: PairTable, camera
    ) -> None:
        # The whole point of inter-star angles: rotating the observation
        # cannot change which catalogue triangle matches.
        from skymatch.scene import SceneConfig, simulate_scene

        rng = np.random.default_rng(21)
        cfg = SceneConfig(camera=camera, centroid_sigma_arcsec=0.0, magnitude_sigma=0.0)
        scene = simulate_scene(catalogue, cfg, rng)
        tol = separation_tolerance(1.0)
        angles = triangle_edge_angles(scene.vectors, 0, 1, 2)
        first = triangle_candidates(table, *angles, tol)
        rotated = scene.vectors @ random_rotation(rng).T
        second = triangle_candidates(table, *triangle_edge_angles(rotated, 0, 1, 2), tol)
        assert np.array_equal(first[0], second[0])
        assert np.array_equal(first[1], second[1])
        assert np.array_equal(first[2], second[2])

    def test_truth_is_found_under_noise(
        self, catalogue: StarCatalogue, table: PairTable, camera
    ) -> None:
        from skymatch.scene import SceneConfig, simulate_scene

        rng = np.random.default_rng(22)
        cfg = SceneConfig(camera=camera, centroid_sigma_arcsec=5.0)
        tol = separation_tolerance(5.0)
        found = 0
        for _ in range(40):
            scene = simulate_scene(catalogue, cfg, rng)
            a, b, c, _ = triangle_candidates(
                table, *triangle_edge_angles(scene.vectors, 0, 1, 2), tol
            )
            truth = scene.truth_index[:3]
            if np.any((a == truth[0]) & (b == truth[1]) & (c == truth[2])):
                found += 1
        assert found == 40

    def test_wider_tolerance_finds_at_least_as_many(
        self, catalogue: StarCatalogue, table: PairTable, camera
    ) -> None:
        from skymatch.scene import SceneConfig, simulate_scene

        rng = np.random.default_rng(23)
        scene = simulate_scene(
            catalogue, SceneConfig(camera=camera, centroid_sigma_arcsec=5.0), rng
        )
        angles = triangle_edge_angles(scene.vectors, 0, 1, 2)
        narrow = triangle_candidates(table, *angles, separation_tolerance(5.0))[0].size
        wide = triangle_candidates(table, *angles, separation_tolerance(40.0))[0].size
        assert wide >= narrow

    def test_max_candidates_truncates(
        self, catalogue: StarCatalogue, table: PairTable, camera
    ) -> None:
        from skymatch.scene import SceneConfig, simulate_scene

        rng = np.random.default_rng(24)
        scene = simulate_scene(
            catalogue, SceneConfig(camera=camera, centroid_sigma_arcsec=5.0), rng
        )
        angles = triangle_edge_angles(scene.vectors, 0, 1, 2)
        big_tol = separation_tolerance(300.0)
        full = triangle_candidates(table, *angles, big_tol)[0].size
        cut = triangle_candidates(table, *angles, big_tol, max_candidates=3)[0].size
        assert cut == min(full, 3)

    def test_validation(self, toy_table: PairTable) -> None:
        with pytest.raises(ValueError, match="tolerance_rad"):
            triangle_candidates(toy_table, 0.1, 0.1, 0.1, 0.0)
        with pytest.raises(ValueError, match="t_ij"):
            triangle_candidates(toy_table, -0.1, 0.1, 0.1, 1e-6)
        with pytest.raises(ValueError, match="t_jk"):
            triangle_candidates(toy_table, 0.1, 0.1, np.nan, 1e-6)
        with pytest.raises(ValueError, match="max_candidates"):
            triangle_candidates(toy_table, 0.1, 0.1, 0.1, 1e-6, max_candidates=-1)
