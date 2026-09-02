"""Pinhole star-camera model: field of view, projection, and the noise scale.

The camera frame has boresight ``+z``, detector axes ``+x`` (columns) and
``+y`` (rows). A direction ``v`` in the camera frame lands at

.. math:: (x, y) = f\\,(v_x/v_z,\; v_y/v_z)                         (Eq. K1)

with the focal length in pixels fixed by the *full* field of view across the
detector width,

.. math:: f = \\frac{n_{\\text{pix}}/2}{\\tan(\\text{FOV}/2)}          (Eq. K2)

This is the ideal gnomonic (tangent-plane) projection with no distortion, no
misalignment and no pixel-response variation. Real star cameras carry radial
and decentring distortion of several pixels at the field edge, which shifts
the *inter-star angles* an identification algorithm depends on. Nothing here
represents that; see ``DATASET_CARD.md``.

The plate scale used to convert a centroiding error in pixels to an angular
error is the field-average value ``s = FOV / n_pix`` [rad/pixel]   (Eq. K3).

A gnomonic projection has no single plate scale: the local scale is
``cos^2(r)/f`` and so falls off away from the boresight. Measured on the
reference 12 deg / 1024 px camera in ``validation/validate_geometry.py``
section 1e, the true local scale runs from 1.0037x Eq. K3 at the field centre
to 0.9874x at the corner -- inside 1.3% everywhere. The package uses Eq. K3
only to *set* a noise level in pixels and a matching tolerance in radians; the
projection itself is exact, so this is a labelling approximation and not an
error in the geometry.

Typical values for a small-satellite star tracker, all of which the defaults
sit inside: fields of 10-25 deg, detectors of 512-2048 pixels, and centroiding
to 0.05-0.2 pixel for a well-sampled defocused star image (Liebe 2002).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .geometry import normalise

__all__ = ["CameraModel"]


@dataclass(frozen=True)
class CameraModel:
    """A square pinhole star camera.

    Parameters
    ----------
    fov_deg
        Full field of view across the detector width [deg]. Must be in
        ``(0, 120)``.
    pixels
        Detector width and height in pixels. Must be >= 16.
    """

    fov_deg: float = 12.0
    pixels: int = 1024

    def __post_init__(self) -> None:
        if not np.isfinite(self.fov_deg) or not (0.0 < self.fov_deg < 120.0):
            raise ValueError(f"fov_deg must be in (0, 120), got {self.fov_deg}")
        if int(self.pixels) != self.pixels or self.pixels < 16:
            raise ValueError(f"pixels must be an integer >= 16, got {self.pixels}")

    @property
    def fov_rad(self) -> float:
        """Full field of view across the detector width [rad]."""
        return np.radians(self.fov_deg)

    @property
    def focal_length_px(self) -> float:
        """Eq. K2, focal length [pixels]."""
        return (self.pixels / 2.0) / np.tan(0.5 * self.fov_rad)

    @property
    def arcsec_per_pixel(self) -> float:
        """Eq. K3, on-axis plate scale [arcsec/pixel]."""
        return self.fov_deg * 3600.0 / self.pixels

    @property
    def half_diagonal_rad(self) -> float:
        """Angle [rad] from the boresight to a detector corner."""
        half = self.pixels / 2.0
        return float(np.arctan(np.hypot(half, half) / self.focal_length_px))

    @property
    def max_separation_rad(self) -> float:
        """Largest possible angle [rad] between two stars in the field.

        Twice the half-diagonal, which is what the pair table must be built to.
        """
        return 2.0 * self.half_diagonal_rad

    @property
    def solid_angle_sr(self) -> float:
        """Solid angle of the square field [sr], by exact spherical excess.

        For a square gnomonic field of half-width ``a`` in tangent-plane units
        (``a = tan(FOV/2)``), the solid angle is ``4 arcsin(a^2/(1+a^2))``.
        """
        a = np.tan(0.5 * self.fov_rad)
        return float(4.0 * np.arcsin(a * a / (1.0 + a * a)))

    @property
    def solid_angle_sqdeg(self) -> float:
        """Solid angle of the field [square degrees]."""
        return self.solid_angle_sr * (180.0 / np.pi) ** 2

    def in_field(self, vectors: ArrayLike) -> np.ndarray:
        """Boolean mask ``(N,)`` of camera-frame directions that land on the detector."""
        v = normalise(vectors, "vectors")
        half = self.pixels / 2.0
        ahead = v[:, 2] > 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            x = self.focal_length_px * v[:, 0] / v[:, 2]
            y = self.focal_length_px * v[:, 1] / v[:, 2]
        return ahead & (np.abs(x) <= half) & (np.abs(y) <= half)

    def project(self, vectors: ArrayLike) -> np.ndarray:
        """Eq. K1. Camera-frame directions -> ``(N, 2)`` pixel offsets from the centre.

        Raises ``ValueError`` if any direction is at or behind the focal plane
        (``v_z <= 0``), where the projection does not exist.
        """
        v = normalise(vectors, "vectors")
        if np.any(v[:, 2] <= 0.0):
            raise ValueError(
                f"{int(np.sum(v[:, 2] <= 0.0))} direction(s) have v_z <= 0 and cannot be "
                "projected; filter with in_field() first"
            )
        f = self.focal_length_px
        return np.stack([f * v[:, 0] / v[:, 2], f * v[:, 1] / v[:, 2]], axis=1)

    def unproject(self, pixels: ArrayLike) -> np.ndarray:
        """Inverse of Eq. K1. ``(N, 2)`` pixel offsets -> unit directions ``(N, 3)``."""
        p = np.asarray(pixels, dtype=float)
        if p.ndim == 1:
            p = p[None, :]
        if p.ndim != 2 or p.shape[1] != 2:
            raise ValueError(f"pixels must have shape (2,) or (N, 2), got {np.shape(pixels)}")
        if not np.all(np.isfinite(p)):
            raise ValueError("pixels contains non-finite values")
        f = self.focal_length_px
        v = np.stack([p[:, 0], p[:, 1], np.full(p.shape[0], f)], axis=1)
        return v / np.linalg.norm(v, axis=1)[:, None]

    def sigma_pixels(self, sigma_arcsec: float) -> float:
        """Convert a centroid error [arcsec] to pixels through Eq. K3."""
        if sigma_arcsec < 0.0:
            raise ValueError(f"sigma_arcsec must be >= 0, got {sigma_arcsec}")
        return float(sigma_arcsec) / self.arcsec_per_pixel
