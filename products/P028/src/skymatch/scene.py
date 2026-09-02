"""Simulate what a star camera hands the identification algorithm.

A lost-in-space identification starts from a list of detected spots with no
attitude prior at all. This module produces that list, together with the truth
needed to score an identification: for every spot, the catalogue index it came
from, or -1 if it is a false detection.

The pipeline, in order, because the order is what makes the false-star regime
realistic:

1. draw a uniform random attitude (:func:`skymatch.geometry.random_rotation`);
2. rotate the catalogue into the camera frame and keep the stars on the
   detector (Eq. K1);
3. drop each remaining star independently with probability ``dropout_prob``
   -- a star lost to a hot-pixel mask, a cosmic ray or a low signal-to-noise
   ratio;
4. project to pixels and add independent Gaussian centroid noise of
   ``centroid_sigma_arcsec`` on each axis;
5. add ``n_false_stars`` false detections at uniform random pixel positions
   with magnitudes drawn uniformly over the detectable range;
6. sort everything by measured magnitude, brightest first, and keep the
   brightest ``max_stars``.

Step 6 after step 5 is deliberate. A star tracker picks the brightest spots it
has, so a bright false detection *displaces a real star* rather than being
appended to it. Sorting after, not before, is what makes the dense false-star
regime in ``validation/validate_failure_regime.py`` behave the way the real
failure does.

Noise model: independent zero-mean Gaussians on the two focal-plane axes, the
standard centroiding-error model (Liebe 2002). Real centroid error is
correlated with the star's position within a pixel, its brightness and the
point-spread function shape; none of that is here. Magnitudes get independent
Gaussian noise of ``magnitude_sigma``, with no colour term and no
signal-to-noise dependence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .camera import CameraModel
from .catalogue import StarCatalogue
from .geometry import random_rotation

__all__ = ["Scene", "SceneConfig", "simulate_scene"]


@dataclass(frozen=True)
class SceneConfig:
    """Observation conditions for one simulated frame.

    Parameters
    ----------
    camera
        The camera model.
    centroid_sigma_arcsec
        Per-axis centroid noise [arcsec]. 0 is allowed and gives noise-free
        geometry, which is what the exactness checks in validation use.
    n_false_stars
        Number of false detections added to the frame.
    dropout_prob
        Probability that a real star in the field is not detected.
    max_stars
        Number of spots handed to the matcher, brightest first. Real star
        trackers use 5-20.
    magnitude_sigma
        Instrument magnitude noise [mag].
    false_star_magnitude_range
        Magnitude range false detections are drawn from. ``None`` uses
        ``(2.0, catalogue.magnitude_limit)``.
    """

    camera: CameraModel = field(default_factory=CameraModel)
    centroid_sigma_arcsec: float = 5.0
    n_false_stars: int = 0
    dropout_prob: float = 0.0
    max_stars: int = 10
    magnitude_sigma: float = 0.1
    false_star_magnitude_range: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.centroid_sigma_arcsec < 0.0 or not np.isfinite(self.centroid_sigma_arcsec):
            raise ValueError(
                f"centroid_sigma_arcsec must be finite and >= 0, got {self.centroid_sigma_arcsec}"
            )
        if self.n_false_stars < 0:
            raise ValueError(f"n_false_stars must be >= 0, got {self.n_false_stars}")
        if not 0.0 <= self.dropout_prob <= 1.0:
            raise ValueError(f"dropout_prob must be in [0, 1], got {self.dropout_prob}")
        if self.max_stars < 4:
            raise ValueError(
                f"max_stars must be >= 4 (a pyramid needs four stars), got {self.max_stars}"
            )
        if self.magnitude_sigma < 0.0:
            raise ValueError(f"magnitude_sigma must be >= 0, got {self.magnitude_sigma}")


@dataclass(frozen=True)
class Scene:
    """One simulated frame, plus the truth needed to score an identification.

    Attributes
    ----------
    vectors
        ``(n, 3)`` measured unit directions in the camera frame, brightest
        first. This is the only field an identification algorithm may read.
    pixels
        ``(n, 2)`` measured pixel offsets from the detector centre.
    magnitudes
        ``(n,)`` measured instrument magnitudes.
    truth_index
        ``(n,)`` catalogue index of each spot, ``-1`` for a false detection.
        Truth: for scoring only.
    attitude
        ``(3, 3)`` true DCM, catalogue frame -> camera frame. Truth.
    n_in_field
        Real stars on the detector before dropout and truncation.
    n_true_stars
        Real stars actually present among ``vectors``.
    n_false_stars
        False detections actually present among ``vectors``.
    """

    vectors: np.ndarray
    pixels: np.ndarray
    magnitudes: np.ndarray
    truth_index: np.ndarray
    attitude: np.ndarray
    n_in_field: int
    n_true_stars: int
    n_false_stars: int

    @property
    def n_spots(self) -> int:
        """Number of spots handed to the matcher."""
        return int(self.vectors.shape[0])

    @property
    def false_fraction(self) -> float:
        """Fraction of the handed spots that are false detections."""
        return self.n_false_stars / self.n_spots if self.n_spots else 0.0


def simulate_scene(
    catalogue: StarCatalogue,
    config: SceneConfig,
    rng: np.random.Generator,
    attitude: np.ndarray | None = None,
) -> Scene:
    """Simulate one frame. Deterministic given ``rng``'s state.

    Parameters
    ----------
    catalogue
        The star catalogue to observe.
    config
        Observation conditions.
    rng
        A ``numpy.random.Generator``; advanced in place.
    attitude
        Optional ``(3, 3)`` DCM to use instead of a random one, for repeating
        a specific pointing.

    Returns a :class:`Scene`, possibly with fewer than four spots if the field
    is empty. Callers must handle that: it is the dominant failure mode at a
    bright magnitude limit and is measured in validation, not hidden here.
    """
    cam = config.camera
    att = random_rotation(rng) if attitude is None else np.asarray(attitude, dtype=float)
    if att.shape != (3, 3):
        raise ValueError(f"attitude must be (3, 3), got {att.shape}")

    v_cam = catalogue.vectors @ att.T
    on_detector = cam.in_field(v_cam)
    idx = np.flatnonzero(on_detector)
    n_in_field = int(idx.size)

    if config.dropout_prob > 0.0 and idx.size:
        idx = idx[rng.random(idx.size) >= config.dropout_prob]

    if idx.size:
        px = cam.project(v_cam[idx])
        sigma_px = cam.sigma_pixels(config.centroid_sigma_arcsec)
        if sigma_px > 0.0:
            px = px + sigma_px * rng.normal(size=px.shape)
        mags = catalogue.magnitude[idx] + config.magnitude_sigma * rng.normal(size=idx.size)
        truth = idx.astype(np.int64)
    else:
        px = np.empty((0, 2))
        mags = np.empty(0)
        truth = np.empty(0, dtype=np.int64)

    if config.n_false_stars > 0:
        half = cam.pixels / 2.0
        fpx = rng.uniform(-half, half, size=(config.n_false_stars, 2))
        lo, hi = config.false_star_magnitude_range or (2.0, catalogue.magnitude_limit)
        if hi <= lo:
            raise ValueError(f"false_star_magnitude_range must be increasing, got ({lo}, {hi})")
        fmag = rng.uniform(lo, hi, size=config.n_false_stars)
        px = np.vstack([px, fpx])
        mags = np.concatenate([mags, fmag])
        truth = np.concatenate([truth, np.full(config.n_false_stars, -1, dtype=np.int64)])

    order = np.argsort(mags, kind="stable")[: config.max_stars]
    px, mags, truth = px[order], mags[order], truth[order]
    vectors = cam.unproject(px) if px.shape[0] else np.empty((0, 3))

    return Scene(
        vectors=vectors,
        pixels=px,
        magnitudes=mags,
        truth_index=truth,
        attitude=att,
        n_in_field=n_in_field,
        n_true_stars=int(np.sum(truth >= 0)),
        n_false_stars=int(np.sum(truth < 0)),
    )
