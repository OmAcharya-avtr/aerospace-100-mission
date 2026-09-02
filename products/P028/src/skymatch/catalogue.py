"""Synthetic star catalogue generation and preparation.

No real catalogue is bundled or downloaded. Stars are drawn from a generative
model with a realistic magnitude distribution and a realistic whole-sky
density, so that the identification results characterise the *algorithms*
rather than predict on-sky performance. ``DATASET_CARD.md`` states what is and
is not modelled; the short version is that this is an isotropic sky of
point sources with no proper motion, no binaries, no spectral response and no
optical distortion.

Magnitude model
---------------
Whole-sky cumulative counts follow a power law in magnitude,

.. math:: N(<m) = N_{\\text{ref}} \\, 10^{\\,b\\,(m - m_{\\text{ref}})}   (Eq. C1)

with defaults ``N_ref = 4800`` at ``m_ref = 6.0`` and ``b = 0.52``. The
constant is the order of magnitude of the naked-eye star count quoted in
standard references; the slope is in the range expected for a locally uniform
stellar distribution seen through a shallow magnitude cut (a strictly uniform
space density gives ``b = 0.6``, and real counts flatten below that as the
sample leaves the galactic disc). **No fit to a real catalogue was performed**,
and Eq. C1 has no galactic-latitude dependence, which is its largest single
departure from the real sky: measured counts vary by roughly an order of
magnitude between the galactic pole and the galactic plane.

Sampling magnitudes from Eq. C1 by inverse transform on ``[m_min, m_limit]``:

.. math:: m = \\frac{1}{b}\\log_{10}\\!\\big[10^{b m_{\\min}}
          + u\\,(10^{b m_{\\lim}} - 10^{b m_{\\min}})\\big],\\quad u\\sim U(0,1)
                                                                     (Eq. C2)

Positions are uniform on the sphere: ``ra ~ U(0, 2*pi)``, ``sin(dec) ~ U(-1, 1)``.

Catalogue preparation
---------------------
A real star tracker's catalogue is *prepared*, not used raw: stars closer
together than the instrument can resolve blend into one centroid at the wrong
place, so close pairs are removed. :func:`remove_close_pairs` does that, and
the number it removes is checked in validation against the analytic
expectation for a uniform sphere,

.. math:: E[\\text{pairs closer than }\\theta]
          = \\binom{N}{2}\\,\\frac{1 - \\cos\\theta}{2}                 (Eq. C3)

Units: magnitudes are dimensionless (visual-band-like, but see the dataset
card); angles are radians unless the name ends in ``_deg`` or ``_arcsec``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .geometry import unit_vectors_from_radec

__all__ = [
    "DEFAULT_SLOPE",
    "REFERENCE_COUNT",
    "REFERENCE_MAGNITUDE",
    "StarCatalogue",
    "expected_close_pairs",
    "generate_catalogue",
    "predicted_count",
    "remove_close_pairs",
]

#: Whole-sky count brighter than :data:`REFERENCE_MAGNITUDE` in Eq. C1.
REFERENCE_COUNT = 4800.0
#: The magnitude at which :data:`REFERENCE_COUNT` applies.
REFERENCE_MAGNITUDE = 6.0
#: Slope ``b`` of Eq. C1.
DEFAULT_SLOPE = 0.52

_FULL_SKY_STERADIAN = 4.0 * np.pi
_FULL_SKY_SQDEG = _FULL_SKY_STERADIAN * (180.0 / np.pi) ** 2


@dataclass(frozen=True)
class StarCatalogue:
    """A prepared star catalogue.

    Attributes
    ----------
    ra, dec
        ``(N,)`` right ascension and declination [rad].
    magnitude
        ``(N,)`` apparent magnitude [dimensionless], ascending is brighter.
    vectors
        ``(N, 3)`` unit vectors in the catalogue frame (Eq. G1).
    magnitude_limit
        The limiting magnitude the catalogue was generated to.
    seed
        The integer seed that reproduces it exactly.
    removed_close_pairs
        How many stars :func:`remove_close_pairs` deleted during preparation.
    min_separation_rad
        The separation below which stars were removed, or 0.0 if none were.
    """

    ra: np.ndarray
    dec: np.ndarray
    magnitude: np.ndarray
    vectors: np.ndarray
    magnitude_limit: float
    seed: int
    removed_close_pairs: int = 0
    min_separation_rad: float = 0.0

    @property
    def n_stars(self) -> int:
        """Number of stars in the catalogue."""
        return int(self.magnitude.shape[0])

    @property
    def density_per_steradian(self) -> float:
        """Mean sky density [stars / sr]."""
        return self.n_stars / _FULL_SKY_STERADIAN

    def expected_in_solid_angle(self, solid_angle_sr: float) -> float:
        """Expected star count in a field of ``solid_angle_sr`` [sr]."""
        if solid_angle_sr < 0.0:
            raise ValueError(f"solid_angle_sr must be >= 0, got {solid_angle_sr}")
        return self.density_per_steradian * float(solid_angle_sr)

    def brighter_than(self, magnitude: float) -> StarCatalogue:
        """A new catalogue containing only stars brighter than ``magnitude``."""
        keep = self.magnitude < float(magnitude)
        return StarCatalogue(
            ra=self.ra[keep],
            dec=self.dec[keep],
            magnitude=self.magnitude[keep],
            vectors=self.vectors[keep],
            magnitude_limit=min(float(magnitude), self.magnitude_limit),
            seed=self.seed,
            removed_close_pairs=self.removed_close_pairs,
            min_separation_rad=self.min_separation_rad,
        )

    def stars_within(self, direction: np.ndarray, radius_rad: float) -> np.ndarray:
        """Indices of stars within ``radius_rad`` of a unit ``direction``.

        Brute force over the whole catalogue; ``O(N)`` per call. Fine for the
        catalogue sizes here (``N <= 2e4``), and used only in verification.
        """
        if radius_rad < 0.0:
            raise ValueError(f"radius_rad must be >= 0, got {radius_rad}")
        d = np.asarray(direction, dtype=float).reshape(3)
        d = d / np.linalg.norm(d)
        return np.flatnonzero(self.vectors @ d >= np.cos(radius_rad))


def predicted_count(
    magnitude_limit: float,
    magnitude_min: float = -1.5,
    reference_count: float = REFERENCE_COUNT,
    reference_magnitude: float = REFERENCE_MAGNITUDE,
    slope: float = DEFAULT_SLOPE,
) -> float:
    """Eq. C1: whole-sky count between ``magnitude_min`` and ``magnitude_limit``."""
    if slope <= 0.0:
        raise ValueError(f"slope must be > 0, got {slope}")
    if magnitude_limit <= magnitude_min:
        raise ValueError(
            f"magnitude_limit ({magnitude_limit}) must exceed magnitude_min ({magnitude_min})"
        )
    hi = 10.0 ** (slope * (magnitude_limit - reference_magnitude))
    lo = 10.0 ** (slope * (magnitude_min - reference_magnitude))
    return float(reference_count * (hi - lo))


def expected_close_pairs(n_stars: int, separation_rad: float) -> float:
    """Eq. C3: expected number of pairs closer than ``separation_rad`` on a uniform sphere."""
    if n_stars < 0:
        raise ValueError(f"n_stars must be >= 0, got {n_stars}")
    if separation_rad < 0.0:
        raise ValueError(f"separation_rad must be >= 0, got {separation_rad}")
    return 0.5 * n_stars * (n_stars - 1) * 0.5 * (1.0 - np.cos(separation_rad))


def remove_close_pairs(
    catalogue: StarCatalogue, min_separation_rad: float
) -> tuple[StarCatalogue, int]:
    """Delete every star that has a neighbour closer than ``min_separation_rad``.

    Both members of a close pair go. That is deliberately conservative: an
    unresolved pair produces one centroid somewhere between the two stars, so
    keeping either one would put a wrong position in the catalogue.

    Returns ``(prepared_catalogue, n_removed)``.
    """
    if min_separation_rad < 0.0:
        raise ValueError(f"min_separation_rad must be >= 0, got {min_separation_rad}")
    if min_separation_rad == 0.0 or catalogue.n_stars < 2:
        return catalogue, 0
    chord = 2.0 * np.sin(0.5 * min_separation_rad)
    tree = cKDTree(catalogue.vectors)
    pairs = tree.query_pairs(chord, output_type="ndarray")
    if pairs.size == 0:
        drop = np.zeros(catalogue.n_stars, dtype=bool)
    else:
        drop = np.zeros(catalogue.n_stars, dtype=bool)
        drop[np.unique(pairs.ravel())] = True
    keep = ~drop
    n_removed = int(drop.sum())
    return (
        StarCatalogue(
            ra=catalogue.ra[keep],
            dec=catalogue.dec[keep],
            magnitude=catalogue.magnitude[keep],
            vectors=catalogue.vectors[keep],
            magnitude_limit=catalogue.magnitude_limit,
            seed=catalogue.seed,
            removed_close_pairs=n_removed,
            min_separation_rad=float(min_separation_rad),
        ),
        n_removed,
    )


def generate_catalogue(
    magnitude_limit: float = 6.0,
    seed: int = 20260902,
    *,
    magnitude_min: float = -1.5,
    reference_count: float = REFERENCE_COUNT,
    reference_magnitude: float = REFERENCE_MAGNITUDE,
    slope: float = DEFAULT_SLOPE,
    min_separation_rad: float = 0.0,
) -> StarCatalogue:
    """Generate a synthetic catalogue to ``magnitude_limit``, sorted by magnitude.

    Deterministic in ``seed`` through ``numpy.random.default_rng``: the same
    seed and parameters reproduce the same catalogue bit for bit.

    Parameters
    ----------
    magnitude_limit
        Faintest magnitude retained [dimensionless]. The star count follows
        Eq. C1 and grows by ``10**slope`` per magnitude — a factor 3.3 per
        magnitude at the default slope — so the pair table (see
        :mod:`skymatch.pairtable`) grows roughly 11x per magnitude.
    seed
        Integer seed.
    magnitude_min
        Brightest magnitude in the model; -1.5 is about Sirius.
    reference_count, reference_magnitude, slope
        Eq. C1 parameters.
    min_separation_rad
        If > 0, :func:`remove_close_pairs` is applied as a preparation step.

    Raises ``ValueError`` on an out-of-range magnitude limit or a
    non-positive slope.
    """
    if not np.isfinite(magnitude_limit):
        raise ValueError("magnitude_limit must be finite")
    if magnitude_limit <= magnitude_min:
        raise ValueError(
            f"magnitude_limit ({magnitude_limit}) must exceed magnitude_min ({magnitude_min})"
        )
    if magnitude_limit > 12.0:
        n_pred = predicted_count(
            magnitude_limit, magnitude_min, reference_count, reference_magnitude, slope
        )
        raise ValueError(
            f"magnitude_limit {magnitude_limit} exceeds the supported range: Eq. C1 predicts "
            f"{n_pred:.3e} stars, and the pair table is O(N^2). Limit is 12.0."
        )
    n_expected = predicted_count(
        magnitude_limit, magnitude_min, reference_count, reference_magnitude, slope
    )
    n_stars = int(round(n_expected))
    if n_stars < 4:
        raise ValueError(
            f"magnitude_limit {magnitude_limit} gives only {n_stars} stars; "
            "at least 4 are needed for a pyramid"
        )

    rng = np.random.default_rng(int(seed))
    u = rng.random(n_stars)
    lo = 10.0 ** (slope * magnitude_min)
    hi = 10.0 ** (slope * magnitude_limit)
    magnitude = np.log10(lo + u * (hi - lo)) / slope  # Eq. C2
    order = np.argsort(magnitude, kind="stable")
    magnitude = magnitude[order]

    ra = rng.random(n_stars) * 2.0 * np.pi
    dec = np.arcsin(2.0 * rng.random(n_stars) - 1.0)
    ra, dec = ra[order], dec[order]

    catalogue = StarCatalogue(
        ra=ra,
        dec=dec,
        magnitude=magnitude,
        vectors=unit_vectors_from_radec(ra, dec),
        magnitude_limit=float(magnitude_limit),
        seed=int(seed),
    )
    if min_separation_rad > 0.0:
        catalogue, _ = remove_close_pairs(catalogue, min_separation_rad)
    return catalogue
