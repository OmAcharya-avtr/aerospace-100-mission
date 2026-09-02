"""Shared fixtures. Everything is small: the tests check behaviour, not scale."""

from __future__ import annotations

import numpy as np
import pytest

from skymatch.camera import CameraModel
from skymatch.catalogue import StarCatalogue, generate_catalogue
from skymatch.geometry import unit_vectors_from_radec
from skymatch.pairtable import PairTable

SEED = 20260902


@pytest.fixture(scope="session")
def camera() -> CameraModel:
    """The reference 12 deg / 1024 px camera."""
    return CameraModel(fov_deg=12.0, pixels=1024)


@pytest.fixture(scope="session")
def small_catalogue() -> StarCatalogue:
    """Magnitude limit 5.0: 1449 stars, enough for a populated field and fast."""
    return generate_catalogue(5.0, seed=SEED)


@pytest.fixture(scope="session")
def catalogue() -> StarCatalogue:
    """Magnitude limit 6.0: the catalogue the validation scripts use."""
    return generate_catalogue(6.0, seed=SEED)


@pytest.fixture(scope="session")
def table(catalogue: StarCatalogue, camera: CameraModel) -> PairTable:
    """Pair table over the magnitude-6.0 catalogue for the reference camera."""
    return PairTable(catalogue, camera.max_separation_rad)


@pytest.fixture(scope="session")
def toy_catalogue() -> StarCatalogue:
    """Five stars at hand-chosen positions, for known-answer tests.

    Positions in (ra, dec) degrees, all near (0, 0) so the small-angle
    intuition holds:

    ==  ======  ======  =========================================
    id  ra deg  dec deg  note
    ==  ======  ======  =========================================
    0     0.0     0.0    on the x axis
    1     3.0     0.0    3.0000 deg from star 0 (equator, exact)
    2     0.0     4.0    4.0000 deg from star 0 (meridian, exact)
    3     8.0     0.0    8.0000 deg from star 0
    4    -5.0     2.0    a fourth star for pyramid confirmation
    ==  ======  ======  =========================================
    """
    ra = np.radians([0.0, 3.0, 0.0, 8.0, -5.0])
    dec = np.radians([0.0, 0.0, 4.0, 0.0, 2.0])
    mag = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    return StarCatalogue(
        ra=ra,
        dec=dec,
        magnitude=mag,
        vectors=unit_vectors_from_radec(ra, dec),
        magnitude_limit=6.0,
        seed=0,
    )
