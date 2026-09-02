"""The pinhole camera model (Eq. K1-K3)."""

from __future__ import annotations

import numpy as np
import pytest

from skymatch.camera import CameraModel
from skymatch.geometry import angular_separation


class TestKnownAnswers:
    def test_focal_length_by_hand(self) -> None:
        # Eq. K2 with FOV = 90 deg and 100 pixels:
        # f = (100/2) / tan(45 deg) = 50 / 1 = 50 pixels, exactly.
        assert CameraModel(fov_deg=90.0, pixels=100).focal_length_px == pytest.approx(50.0)

    def test_plate_scale_by_hand(self) -> None:
        # Eq. K3: 12 deg over 1024 px = 12 * 3600 / 1024 = 42.1875 arcsec/pixel.
        assert CameraModel(12.0, 1024).arcsec_per_pixel == pytest.approx(42.1875)

    def test_solid_angle_by_hand(self) -> None:
        # For a square gnomonic field with a = tan(FOV/2) = tan(45) = 1,
        # Omega = 4 arcsin(a^2/(1+a^2)) = 4 arcsin(1/2) = 4 * pi/6 = 2.0943951.
        assert CameraModel(90.0, 128).solid_angle_sr == pytest.approx(2.0943951023931953)

    def test_half_diagonal_by_hand(self) -> None:
        # FOV 90 deg, f = 50 px, corner at (50, 50): angle from boresight is
        # atan(sqrt(50^2 + 50^2) / 50) = atan(sqrt(2)) = 54.7356 deg.
        cam = CameraModel(90.0, 100)
        assert np.degrees(cam.half_diagonal_rad) == pytest.approx(54.735610317245346)
        assert cam.max_separation_rad == pytest.approx(2 * cam.half_diagonal_rad)

    def test_boresight_projects_to_centre(self) -> None:
        cam = CameraModel()
        assert np.allclose(cam.project([[0.0, 0.0, 1.0]]), 0.0)

    def test_sigma_pixels(self) -> None:
        # 42.1875 arcsec/pixel, so 42.1875 arcsec is exactly one pixel.
        cam = CameraModel(12.0, 1024)
        assert cam.sigma_pixels(42.1875) == pytest.approx(1.0)
        assert cam.sigma_pixels(0.0) == 0.0


class TestBehaviour:
    def test_round_trip(self, camera: CameraModel) -> None:
        rng = np.random.default_rng(11)
        half = camera.pixels / 2.0
        px = rng.uniform(-half, half, size=(500, 2))
        assert np.allclose(camera.project(camera.unproject(px)), px, atol=1e-9)

    def test_in_field_matches_the_detector_edge(self, camera: CameraModel) -> None:
        half = camera.pixels / 2.0
        inside = camera.unproject([[half - 0.5, 0.0]])
        outside = camera.unproject([[half + 0.5, 0.0]])
        assert camera.in_field(inside)[0]
        assert not camera.in_field(outside)[0]

    def test_in_field_rejects_behind(self, camera: CameraModel) -> None:
        assert not camera.in_field([[0.0, 0.0, -1.0]])[0]

    def test_corner_angle_matches_half_diagonal(self, camera: CameraModel) -> None:
        half = camera.pixels / 2.0
        corner = camera.unproject([[half, half]])
        ang = angular_separation(corner, [[0.0, 0.0, 1.0]])[0]
        assert ang == pytest.approx(camera.half_diagonal_rad, abs=1e-14)

    def test_solid_angle_is_below_the_naive_square(self, camera: CameraModel) -> None:
        # A 12 deg square field is 143.48 sq.deg, not 144: the sphere is curved.
        assert camera.solid_angle_sqdeg == pytest.approx(143.477245, rel=1e-6)
        assert camera.solid_angle_sqdeg < camera.fov_deg**2


class TestValidation:
    @pytest.mark.parametrize("fov", [0.0, -1.0, 120.0, 200.0, np.nan])
    def test_rejects_bad_fov(self, fov: float) -> None:
        with pytest.raises(ValueError, match="fov_deg"):
            CameraModel(fov_deg=fov)

    @pytest.mark.parametrize("pixels", [0, 15, -4, 10.5])
    def test_rejects_bad_pixel_count(self, pixels: float) -> None:
        with pytest.raises(ValueError, match="pixels"):
            CameraModel(pixels=pixels)

    def test_project_rejects_behind_the_focal_plane(self, camera: CameraModel) -> None:
        with pytest.raises(ValueError, match="v_z <= 0"):
            camera.project([[0.0, 0.0, -1.0]])

    def test_unproject_rejects_wrong_shape(self, camera: CameraModel) -> None:
        with pytest.raises(ValueError, match=r"shape \(2,\) or \(N, 2\)"):
            camera.unproject(np.zeros((3, 3)))

    def test_unproject_rejects_non_finite(self, camera: CameraModel) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            camera.unproject([[np.nan, 0.0]])

    def test_sigma_pixels_rejects_negative(self, camera: CameraModel) -> None:
        with pytest.raises(ValueError, match="sigma_arcsec"):
            camera.sigma_pixels(-1.0)
