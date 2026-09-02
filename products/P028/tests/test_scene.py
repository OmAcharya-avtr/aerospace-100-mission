"""Frame simulation and its truth bookkeeping."""

from __future__ import annotations

import numpy as np
import pytest

from skymatch.camera import CameraModel
from skymatch.catalogue import StarCatalogue
from skymatch.geometry import ARCSEC, angular_separation, random_rotation
from skymatch.scene import SceneConfig, simulate_scene


class TestBehaviour:
    def test_noise_free_spots_land_on_their_stars(
        self, small_catalogue: StarCatalogue, camera: CameraModel
    ) -> None:
        cfg = SceneConfig(camera=camera, centroid_sigma_arcsec=0.0, magnitude_sigma=0.0)
        rng = np.random.default_rng(1)
        # At magnitude limit 5.0 a 12 deg field holds about 5 stars, so a
        # given pointing can hold fewer than four; draw until one does not.
        scene = simulate_scene(small_catalogue, cfg, rng)
        while scene.n_spots < 4:
            scene = simulate_scene(small_catalogue, cfg, rng)
        assert scene.n_spots >= 4
        for spot, star in enumerate(scene.truth_index):
            expected = small_catalogue.vectors[star] @ scene.attitude.T
            assert angular_separation(scene.vectors[spot], expected)[0] < 1e-12

    def test_noise_scale_matches_the_configured_sigma(
        self, small_catalogue: StarCatalogue, camera: CameraModel
    ) -> None:
        sigma = 20.0
        cfg = SceneConfig(camera=camera, centroid_sigma_arcsec=sigma, max_stars=20)
        rng = np.random.default_rng(2)
        errors = []
        for _ in range(60):
            scene = simulate_scene(small_catalogue, cfg, rng)
            for spot, star in enumerate(scene.truth_index):
                truth = small_catalogue.vectors[star] @ scene.attitude.T
                errors.append(angular_separation(scene.vectors[spot], truth)[0] / ARCSEC)
        # Two independent axes of sigma each: the radial error has mean
        # sigma * sqrt(pi/2) = 1.2533 sigma. Allow 10% for the off-axis plate
        # scale and finite sampling.
        assert np.mean(errors) == pytest.approx(sigma * np.sqrt(np.pi / 2), rel=0.10)

    def test_false_stars_are_marked(
        self, small_catalogue: StarCatalogue, camera: CameraModel
    ) -> None:
        cfg = SceneConfig(camera=camera, n_false_stars=5, max_stars=20)
        rng = np.random.default_rng(3)
        scene = simulate_scene(small_catalogue, cfg, rng)
        assert scene.n_false_stars == int(np.sum(scene.truth_index < 0))
        assert scene.n_true_stars + scene.n_false_stars == scene.n_spots
        assert scene.false_fraction == pytest.approx(scene.n_false_stars / scene.n_spots)

    def test_bright_false_stars_displace_real_ones(
        self, catalogue: StarCatalogue, camera: CameraModel
    ) -> None:
        # The frame is sorted by brightness and truncated, so false detections
        # drawn at magnitude 0-1 take the whole spot list.
        cfg = SceneConfig(
            camera=camera,
            n_false_stars=12,
            max_stars=10,
            false_star_magnitude_range=(0.0, 1.0),
        )
        rng = np.random.default_rng(4)
        scene = simulate_scene(catalogue, cfg, rng)
        assert scene.n_spots == 10
        assert scene.n_true_stars == 0

    def test_spots_are_sorted_by_magnitude(
        self, small_catalogue: StarCatalogue, camera: CameraModel
    ) -> None:
        cfg = SceneConfig(camera=camera, n_false_stars=3)
        rng = np.random.default_rng(5)
        scene = simulate_scene(small_catalogue, cfg, rng)
        assert np.all(np.diff(scene.magnitudes) >= 0.0)

    def test_max_stars_is_respected(
        self, catalogue: StarCatalogue, camera: CameraModel
    ) -> None:
        cfg = SceneConfig(camera=camera, max_stars=6)
        rng = np.random.default_rng(6)
        for _ in range(10):
            assert simulate_scene(catalogue, cfg, rng).n_spots <= 6

    def test_dropout_removes_stars(
        self, catalogue: StarCatalogue, camera: CameraModel
    ) -> None:
        rng = np.random.default_rng(7)
        cfg_none = SceneConfig(camera=camera, dropout_prob=0.0, max_stars=40)
        cfg_all = SceneConfig(camera=camera, dropout_prob=1.0, max_stars=40)
        attitude = random_rotation(rng)
        kept = simulate_scene(catalogue, cfg_none, rng, attitude=attitude)
        dropped = simulate_scene(catalogue, cfg_all, rng, attitude=attitude)
        assert kept.n_spots > 0
        assert dropped.n_spots == 0
        assert dropped.n_in_field == kept.n_in_field

    def test_fixed_attitude_is_used(
        self, small_catalogue: StarCatalogue, camera: CameraModel
    ) -> None:
        rng = np.random.default_rng(8)
        attitude = random_rotation(rng)
        scene = simulate_scene(small_catalogue, SceneConfig(camera=camera), rng, attitude)
        assert np.array_equal(scene.attitude, attitude)

    def test_determinism(self, small_catalogue: StarCatalogue, camera: CameraModel) -> None:
        cfg = SceneConfig(camera=camera, n_false_stars=2)
        a = simulate_scene(small_catalogue, cfg, np.random.default_rng(9))
        b = simulate_scene(small_catalogue, cfg, np.random.default_rng(9))
        assert np.array_equal(a.vectors, b.vectors)
        assert np.array_equal(a.truth_index, b.truth_index)

    def test_mean_stars_in_field_matches_the_density(
        self, catalogue: StarCatalogue, camera: CameraModel
    ) -> None:
        rng = np.random.default_rng(10)
        cfg = SceneConfig(camera=camera)
        counts = [simulate_scene(catalogue, cfg, rng).n_in_field for _ in range(200)]
        predicted = catalogue.expected_in_solid_angle(camera.solid_angle_sr)
        assert np.mean(counts) == pytest.approx(predicted, rel=0.12)


class TestValidation:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"centroid_sigma_arcsec": -1.0}, "centroid_sigma_arcsec"),
            ({"centroid_sigma_arcsec": np.inf}, "centroid_sigma_arcsec"),
            ({"n_false_stars": -1}, "n_false_stars"),
            ({"dropout_prob": 1.5}, "dropout_prob"),
            ({"max_stars": 3}, "max_stars"),
            ({"magnitude_sigma": -0.1}, "magnitude_sigma"),
        ],
    )
    def test_scene_config_validation(self, kwargs: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            SceneConfig(**kwargs)

    def test_rejects_bad_attitude_shape(
        self, small_catalogue: StarCatalogue, camera: CameraModel
    ) -> None:
        with pytest.raises(ValueError, match=r"attitude must be \(3, 3\)"):
            simulate_scene(
                small_catalogue, SceneConfig(camera=camera), np.random.default_rng(0),
                attitude=np.eye(2),
            )

    def test_rejects_inverted_false_magnitude_range(
        self, small_catalogue: StarCatalogue, camera: CameraModel
    ) -> None:
        cfg = SceneConfig(camera=camera, n_false_stars=2, false_star_magnitude_range=(6.0, 2.0))
        with pytest.raises(ValueError, match="increasing"):
            simulate_scene(small_catalogue, cfg, np.random.default_rng(0))
