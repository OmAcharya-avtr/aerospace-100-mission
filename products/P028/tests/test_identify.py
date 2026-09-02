"""Candidate gathering, the classical decision rules, and resolution."""

from __future__ import annotations

import numpy as np
import pytest

from skymatch.camera import CameraModel
from skymatch.catalogue import StarCatalogue
from skymatch.geometry import ARCSEC, angle_between_dcm
from skymatch.identify import (
    FEATURE_NAMES,
    SearchConfig,
    gather_candidates,
    observed_separations,
    pyramid_decision,
    resolve,
    triangle_decision,
    triple_scan_order,
)
from skymatch.pairtable import PairTable
from skymatch.scene import Scene, SceneConfig, simulate_scene
from skymatch.triangle import separation_tolerance


def make_scene(catalogue, camera, seed, **kwargs) -> Scene:
    cfg = SceneConfig(camera=camera, **kwargs)
    return simulate_scene(catalogue, cfg, np.random.default_rng(seed))


class TestSeparationMatrix:
    def test_matches_pairwise_angles(self) -> None:
        v = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
        sep = observed_separations(v)
        assert np.allclose(np.diag(sep), 0.0, atol=1e-15)
        assert np.allclose(sep[0, 1], np.pi / 2)
        assert np.allclose(sep, sep.T)

    def test_rejects_bad_shape(self) -> None:
        with pytest.raises(ValueError, match=r"shape \(n, 3\)"):
            observed_separations(np.zeros((3, 2)))


class TestScanOrderCache:
    def test_returns_the_same_object(self) -> None:
        assert triple_scan_order(6) is triple_scan_order(6)

    def test_candidates_are_not_naively_comparable(
        self, catalogue: StarCatalogue, table: PairTable, camera: CameraModel
    ) -> None:
        # Candidate carries NumPy arrays, so dataclass equality is disabled
        # (it would raise on the array comparison). Two distinct objects
        # describing the same match are not ==; compare the index tuples.
        scene = make_scene(catalogue, camera, 30, centroid_sigma_arcsec=5.0)
        tol = separation_tolerance(5.0)
        first, _ = gather_candidates(scene.vectors, scene.magnitudes, table, tol, camera)
        second, _ = gather_candidates(scene.vectors, scene.magnitudes, table, tol, camera)
        assert first[0] is not second[0]
        assert first[0] != second[0]
        assert (first[0].observed, first[0].catalogue) == (
            second[0].observed, second[0].catalogue
        )


class TestGather:
    def test_finds_the_truth_on_a_clean_frame(
        self, catalogue: StarCatalogue, table: PairTable, camera: CameraModel
    ) -> None:
        scene = make_scene(catalogue, camera, 31, centroid_sigma_arcsec=5.0)
        tol = separation_tolerance(5.0)
        candidates, diag = gather_candidates(scene.vectors, scene.magnitudes, table, tol, camera)
        assert candidates
        assert any(c.is_correct(scene.truth_index) for c in candidates)
        assert diag["triples_tried"] == 25
        assert diag["n_spots"] == scene.n_spots

    def test_feature_vector_shape_and_finiteness(
        self, catalogue: StarCatalogue, table: PairTable, camera: CameraModel
    ) -> None:
        scene = make_scene(catalogue, camera, 32, centroid_sigma_arcsec=5.0, n_false_stars=3)
        candidates, _ = gather_candidates(
            scene.vectors, scene.magnitudes, table, separation_tolerance(5.0), camera
        )
        for cand in candidates:
            assert cand.features.shape == (len(FEATURE_NAMES),)
            assert np.all(np.isfinite(cand.features))

    def test_too_few_spots_returns_nothing(
        self, table: PairTable, camera: CameraModel
    ) -> None:
        v = np.array([[0.0, 0.0, 1.0], [0.01, 0.0, 1.0], [0.0, 0.01, 1.0]])
        v /= np.linalg.norm(v, axis=1)[:, None]
        candidates, diag = gather_candidates(v, np.zeros(3), table, 1e-4, camera)
        assert candidates == []
        assert diag["triples_tried"] == 0

    def test_early_exit_returns_fewer_candidates(
        self, catalogue: StarCatalogue, table: PairTable, camera: CameraModel
    ) -> None:
        scene = make_scene(catalogue, camera, 33, centroid_sigma_arcsec=5.0)
        tol = separation_tolerance(5.0)
        full, _ = gather_candidates(scene.vectors, scene.magnitudes, table, tol, camera)
        early, _ = gather_candidates(
            scene.vectors, scene.magnitudes, table, tol, camera, None, True
        )
        assert 0 < len(early) <= len(full)
        a, b = pyramid_decision(early), pyramid_decision(full)
        assert (a.observed, a.catalogue) == (b.observed, b.catalogue)

    def test_search_limits_are_respected(
        self, catalogue: StarCatalogue, table: PairTable, camera: CameraModel
    ) -> None:
        scene = make_scene(catalogue, camera, 34, centroid_sigma_arcsec=5.0)
        cfg = SearchConfig(max_triples=3, max_confirm_stars=1, max_candidates=5)
        candidates, diag = gather_candidates(
            scene.vectors, scene.magnitudes, table, separation_tolerance(5.0), camera, cfg
        )
        assert diag["triples_tried"] <= 3
        assert len(candidates) <= 5

    def test_confirmations_scale_with_evidence(
        self, catalogue: StarCatalogue, table: PairTable, camera: CameraModel
    ) -> None:
        scene = make_scene(catalogue, camera, 35, centroid_sigma_arcsec=5.0)
        candidates, _ = gather_candidates(
            scene.vectors, scene.magnitudes, table, separation_tolerance(5.0), camera
        )
        correct = [c for c in candidates if c.is_correct(scene.truth_index)]
        assert correct
        assert all(c.n_confirm >= 1 for c in correct)
        assert all(len(c.all_observed) == len(c.all_catalogue) for c in candidates)

    def test_validation(self, table: PairTable, camera: CameraModel) -> None:
        v = np.zeros((5, 3))
        v[:, 2] = 1.0
        with pytest.raises(ValueError, match=r"shape \(n, 3\)"):
            gather_candidates(np.zeros((5, 2)), np.zeros(5), table, 1e-4, camera)
        with pytest.raises(ValueError, match="magnitudes has length"):
            gather_candidates(v, np.zeros(3), table, 1e-4, camera)
        with pytest.raises(ValueError, match="tolerance_rad"):
            gather_candidates(v, np.zeros(5), table, 0.0, camera)

    def test_search_config_validation(self) -> None:
        for kwargs in (
            {"max_triples": 0},
            {"max_candidates_per_triple": -1},
            {"max_confirm_stars": 0},
            {"max_candidates": 0},
        ):
            with pytest.raises(ValueError, match="integer >= 1"):
                SearchConfig(**kwargs)


class TestDecisions:
    def test_both_rules_agree_on_a_clean_frame(
        self, catalogue: StarCatalogue, table: PairTable, camera: CameraModel
    ) -> None:
        scene = make_scene(catalogue, camera, 36, centroid_sigma_arcsec=2.0)
        candidates, _ = gather_candidates(
            scene.vectors, scene.magnitudes, table, separation_tolerance(2.0), camera
        )
        tri = triangle_decision(candidates)
        pyr = pyramid_decision(candidates)
        assert tri is not None and pyr is not None
        assert tri.is_correct(scene.truth_index)
        assert pyr.is_correct(scene.truth_index)

    def test_empty_candidates_give_no_decision(self) -> None:
        assert triangle_decision([]) is None
        assert pyramid_decision([]) is None

    def test_pyramid_needs_a_confirmation(
        self, catalogue: StarCatalogue, table: PairTable, camera: CameraModel
    ) -> None:
        scene = make_scene(catalogue, camera, 37, centroid_sigma_arcsec=5.0)
        candidates, _ = gather_candidates(
            scene.vectors, scene.magnitudes, table, separation_tolerance(5.0), camera
        )
        chosen = pyramid_decision(candidates)
        assert chosen is not None and chosen.n_confirm >= 1


class TestResolve:
    def test_attitude_is_recovered(
        self, catalogue: StarCatalogue, table: PairTable, camera: CameraModel
    ) -> None:
        scene = make_scene(catalogue, camera, 38, centroid_sigma_arcsec=5.0)
        tol = separation_tolerance(5.0)
        candidates, diag = gather_candidates(
            scene.vectors, scene.magnitudes, table, tol, camera
        )
        chosen = pyramid_decision(candidates)
        ident = resolve(chosen, scene.vectors, catalogue, camera, tol,
                        n_candidates=len(candidates), diagnostics=diag)
        assert ident.identified
        assert ident.attitude is not None
        error = np.degrees(angle_between_dcm(ident.attitude, scene.attitude)) * 3600.0
        assert error < 120.0
        assert ident.confidence == 1.0
        assert ident.n_candidates == len(candidates)

    def test_extension_matches_more_stars(
        self, catalogue: StarCatalogue, table: PairTable, camera: CameraModel
    ) -> None:
        scene = make_scene(catalogue, camera, 39, centroid_sigma_arcsec=5.0)
        tol = separation_tolerance(5.0)
        candidates, _ = gather_candidates(scene.vectors, scene.magnitudes, table, tol, camera)
        chosen = pyramid_decision(candidates)
        ident = resolve(chosen, scene.vectors, catalogue, camera, tol)
        assert len(ident.observed_indices) >= len(chosen.all_observed)
        for spot, star in zip(ident.observed_indices, ident.catalogue_indices, strict=True):
            assert scene.truth_index[spot] == star

    def test_no_candidate_gives_no_solution(
        self, catalogue: StarCatalogue, camera: CameraModel
    ) -> None:
        ident = resolve(None, np.zeros((4, 3)), catalogue, camera, 1e-4)
        assert not ident.identified
        assert ident.status == "no_solution"
        assert ident.attitude is None
        assert ident.confidence == 0.0
        assert ident.observed_indices.size == 0

    def test_attitude_error_scales_with_noise(
        self, catalogue: StarCatalogue, table: PairTable, camera: CameraModel
    ) -> None:
        errors = {}
        for sigma in (2.0, 20.0):
            tol = separation_tolerance(sigma)
            found = []
            rng = np.random.default_rng(40)
            cfg = SceneConfig(camera=camera, centroid_sigma_arcsec=sigma)
            for _ in range(12):
                scene = simulate_scene(catalogue, cfg, rng)
                candidates, _ = gather_candidates(
                    scene.vectors, scene.magnitudes, table, tol, camera
                )
                chosen = pyramid_decision(candidates)
                if chosen is None or not chosen.is_correct(scene.truth_index):
                    continue
                ident = resolve(chosen, scene.vectors, catalogue, camera, tol)
                found.append(angle_between_dcm(ident.attitude, scene.attitude) / ARCSEC)
            errors[sigma] = float(np.median(found))
        assert errors[20.0] > 4.0 * errors[2.0]
